# This tool learns where to split text into chunks.

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Patterns for finding words and list markers
_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b")
_STRUCTURE_START_PATTERN = re.compile(r"^\s*(?:\(?\d+\)|\d+[.)]|[-•])\s+")
_SENTENCE_END_PUNCT = re.compile(r"[:;]\s*$")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "will",
    "with", "who", "which", "what", "when", "where", "why", "how",
}
_SCORER_CACHE: dict[tuple, "LearnableBoundaryScorer"] = {}

# Split text into a list of sentences
def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences or [text.strip()]

# Get important words from a sentence
def _keyword_tokens(text: str) -> set[str]:
    tokens = set()
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        tokens.add(token)
    return tokens

# Count words that start with a capital letter
def _capitalized_count(text: str) -> int:
    count = 0
    for token in _TOKEN_PATTERN.findall(text):
        if len(token) > 1 and token[0].isupper() and not token.isupper():
            count += 1
    return count

# Count words that contain numbers
def _numeric_count(text: str) -> int:
    return sum(1 for token in _TOKEN_PATTERN.findall(text) if any(ch.isdigit() for ch in token))

# Check if a sentence starts a list or a new point
def _starts_structure(text: str) -> float:
    lowered = text.strip().lower()
    if _STRUCTURE_START_PATTERN.match(text):
        return 1.0
    if lowered.startswith(("first", "second", "third", "finally", "however", "meanwhile", "in contrast")):
        return 1.0
    return 0.0

# Check if a sentence ends with a colon or semicolon
def _ends_with_structure_punct(text: str) -> float:
    return 1.0 if _SENTENCE_END_PUNCT.search(text) else 0.0

# Build a list of numbers to describe the gap between two sentences
def _feature_vector(prev_sentence: str, next_sentence: str) -> np.ndarray:
    prev_tokens = _keyword_tokens(prev_sentence)
    next_tokens = _keyword_tokens(next_sentence)
    overlap = len(prev_tokens & next_tokens)
    overlap_ratio = overlap / max(1, min(len(prev_tokens), len(next_tokens)))

    prev_words = len(prev_sentence.split())
    next_words = len(next_sentence.split())
    prev_caps = _capitalized_count(prev_sentence)
    next_caps = _capitalized_count(next_sentence)
    prev_nums = _numeric_count(prev_sentence)
    next_nums = _numeric_count(next_sentence)

    # Return a normalized vector of features
    return np.asarray(
        [
            prev_words / 40.0,
            next_words / 40.0,
            len(prev_tokens) / 12.0,
            len(next_tokens) / 12.0,
            overlap / 6.0,
            overlap_ratio,
            prev_caps / 4.0,
            next_caps / 4.0,
            prev_nums / 3.0,
            next_nums / 3.0,
            _starts_structure(next_sentence),
            _ends_with_structure_punct(prev_sentence),
        ],
        dtype=np.float32,
    )

# The AI model structure
class _BoundaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# Main class for scoring split points
@dataclass
class LearnableBoundaryScorer:
    model: _BoundaryMLP
    mean: np.ndarray
    std: np.ndarray

    # Predict the chance of a split between two sentences
    def predict_proba(self, prev_sentence: str, next_sentence: str) -> float:
        feats = _feature_vector(prev_sentence, next_sentence)
        feats = (feats - self.mean) / self.std
        with torch.no_grad():
            tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            logits = self.model(tensor)
            return float(torch.sigmoid(logits).item())

# Find examples of "good splits" and "bad splits" in the data
def _build_training_examples(corpus_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    negative_examples: list[np.ndarray] = []
    positive_examples: list[np.ndarray] = []

    for _, row in corpus_df.iterrows():
        paragraphs = row.get("paragraphs", [])
        if not isinstance(paragraphs, (list, np.ndarray)):
            continue
        paragraphs = [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]
        if not paragraphs:
            continue

        # Sentences inside one paragraph should stay together
        split_paragraphs = [_split_sentences(p) for p in paragraphs]
        for sentences in split_paragraphs:
            if len(sentences) < 2:
                continue
            for prev_sentence, next_sentence in zip(sentences[:-1], sentences[1:], strict=False):
                negative_examples.append(_feature_vector(prev_sentence, next_sentence))

        # Sentences across different paragraphs can be split
        for prev_sentences, next_sentences in zip(split_paragraphs[:-1], split_paragraphs[1:], strict=False):
            if not prev_sentences or not next_sentences:
                continue
            positive_examples.append(_feature_vector(prev_sentences[-1], next_sentences[0]))

    if not negative_examples or not positive_examples:
        raise ValueError("Need more data to train.")

    # Balance the number of examples
    n = min(len(negative_examples), len(positive_examples))
    rng = random.Random(42)
    negative_examples = rng.sample(negative_examples, n)
    positive_examples = rng.sample(positive_examples, n)

    x = np.vstack(negative_examples + positive_examples).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
    return x, y

# Train the AI to recognize split points
def train_boundary_scorer(
    corpus_df: pd.DataFrame,
    *,
    hidden_dim: int = 32,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> LearnableBoundaryScorer:
    # Check if we already trained this model
    cache_key = (
        hashlib.sha256(",".join(map(str, corpus_df["wikipedia_id"].astype(int).tolist())).encode("utf-8")).hexdigest(),
        hidden_dim, epochs, batch_size, learning_rate, seed,
    )
    if cache_key in _SCORER_CACHE:
        return _SCORER_CACHE[cache_key]

    # Prepare data for training
    x, y = _build_training_examples(corpus_df)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    # Setup the AI training
    torch.manual_seed(seed)
    model = _BoundaryMLP(input_dim=x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Run the training loop
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    scorer = LearnableBoundaryScorer(model=model, mean=mean, std=std)
    _SCORER_CACHE[cache_key] = scorer
    return scorer