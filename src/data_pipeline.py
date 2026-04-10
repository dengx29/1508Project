# Functions for building the corpus, splitting text, and preparing queries

import gc
import hashlib
import json as _json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import spacy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.learnable_boundary import LearnableBoundaryScorer, train_boundary_scorer
from src.profiler import Profiler

# URL for Wikipedia data
KILT_WIKIPEDIA_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"
_SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "will",
    "with", "who", "which", "what", "when", "where", "why", "how",
}
_SENTENCE_ENCODER_CACHE: dict[str, SentenceTransformer] = {}


# Load the NaturalQuestions dataset from KILT
def load_kilt_nq_dev() -> list[dict]:
    ds = load_dataset("facebook/kilt_tasks", "nq", split="validation", trust_remote_code=True)
    records = list(ds)
    # Free up memory
    del ds
    return records


# Get all page IDs that contain answers
def _extract_gold_page_ids(nq_records: list[dict]) -> set[int]:
    gold_ids: set[int] = set()
    for rec in nq_records:
        for out in rec.get("output", []):
            for prov in out.get("provenance", []):
                wid = prov.get("wikipedia_id")
                if wid is not None:
                    gold_ids.add(int(wid))
    return gold_ids


# Read Wikipedia data one page at a time from the web
def _stream_kilt_wikipedia():
    resp = requests.get(KILT_WIKIPEDIA_URL, stream=True, timeout=60)
    resp.raise_for_status()
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.extend(chunk)
        while b"\n" in buf:
            idx = buf.index(b"\n")
            line = bytes(buf[:idx]).strip()
            del buf[:idx + 1]
            if line:
                yield _json.loads(line)


# Get the path for the cache metadata file
def _cache_metadata_path(cache_path: str) -> Path:
    return Path(f"{cache_path}.meta.json")


# Create a unique ID for the cache based on settings
def _build_cache_signature(nq_records: list[dict], target_size: int, seed: int) -> dict:
    gold_ids = sorted(_extract_gold_page_ids(nq_records))
    gold_digest = hashlib.sha256(",".join(map(str, gold_ids)).encode("utf-8")).hexdigest()
    return {
        "target_size": target_size,
        "seed": seed,
        "gold_page_count": len(gold_ids),
        "gold_page_digest": gold_digest,
    }


# Try to load the corpus from a local file
def _load_cached_corpus(cache_path: str, expected_signature: dict) -> pd.DataFrame | None:
    cache_file = Path(cache_path)
    metadata_file = _cache_metadata_path(cache_path)
    if not cache_file.exists() or not metadata_file.exists():
        return None

    try:
        metadata = _json.loads(metadata_file.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError):
        return None

    # Check if the file settings match
    if metadata != expected_signature:
        return None
    return pd.read_parquet(cache_file)


# Save the corpus to a local file
def _save_cached_corpus(cache_path: str, corpus_df: pd.DataFrame, signature: dict) -> None:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(cache_file, index=False)
    _cache_metadata_path(cache_path).write_text(
        _json.dumps(signature, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# Build a smaller version of Wikipedia with answer pages and random pages
def build_reduced_corpus(
    nq_records: list[dict],
    target_size: int = 10_000,
    cache_path: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    cache_signature = _build_cache_signature(nq_records, target_size=target_size, seed=seed)
    if cache_path:
        cached = _load_cached_corpus(cache_path, cache_signature)
        if cached is not None:
            return cached

    gold_ids = _extract_gold_page_ids(nq_records)
    gold_ids_remaining = set(gold_ids)
    rng = random.Random(seed)

    gold_pages: list[dict] = []
    distractor_reservoir: list[dict] = []
    reservoir_count = 0
    needed_distractors = max(0, target_size - len(gold_ids))
    scanned = 0

    print(f"  Streaming KILT Wikipedia...")

    for page in _stream_kilt_wikipedia():
        wid_str = page.get("wikipedia_id", "")
        try:
            wid = int(wid_str)
        except (ValueError, TypeError):
            continue

        scanned += 1
        # Save gold pages and pick random pages for the rest
        if wid in gold_ids_remaining:
            gold_pages.append(page)
            gold_ids_remaining.discard(wid)
        else:
            reservoir_count += 1
            if len(distractor_reservoir) < needed_distractors:
                distractor_reservoir.append(page)
            else:
                j = rng.randint(0, reservoir_count - 1)
                if j < needed_distractors:
                    distractor_reservoir[j] = page

        # Stop if we have enough pages
        if len(gold_ids_remaining) == 0 and len(distractor_reservoir) >= needed_distractors:
            break
        if scanned >= 2_000_000 and len(distractor_reservoir) >= needed_distractors:
            break

    all_pages = gold_pages + distractor_reservoir[:needed_distractors]
    del gold_pages, distractor_reservoir

    # Clean up the page data format
    def _extract_page(p: dict) -> dict:
        text = p.get("text", [])
        if isinstance(text, list):
            paragraphs = text
        elif isinstance(text, dict):
            paragraphs = text.get("paragraph", text.get("paragraph_text", []))
        else:
            paragraphs = []
        return {
            "wikipedia_id": int(p.get("wikipedia_id", 0)),
            "title": p.get("wikipedia_title", p.get("title", "")),
            "paragraphs": paragraphs,
        }

    df = pd.DataFrame([_extract_page(p) for p in all_pages])
    del all_pages

    # Save to file
    if cache_path:
        _save_cached_corpus(cache_path, df, cache_signature)

    return df


# Split a paragraph into sentences
def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_SPLITTER.split(text) if part.strip()]
    return sentences or [text.strip()]


# Get important words from text
def _keyword_tokens(text: str) -> set[str]:
    tokens = set()
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        tokens.add(token)
    return tokens


# Load and reuse the AI model for sentence meaning
def _get_sentence_encoder(model_name: str) -> SentenceTransformer:
    if model_name not in _SENTENCE_ENCODER_CACHE:
        _SENTENCE_ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _SENTENCE_ENCODER_CACHE[model_name]


# Split text using a moving window of sentences
def _sentence_window_chunks(
    paragraph_text: str,
    window_size: int,
    stride: int,
) -> list[str]:
    sentences = _split_sentences(paragraph_text)
    if len(sentences) <= window_size:
        return [" ".join(sentences).strip()]

    chunks = []
    step = max(1, stride)
    for start in range(0, len(sentences), step):
        window = sentences[start:start + window_size]
        if not window:
            continue
        chunks.append(" ".join(window).strip())
        if start + window_size >= len(sentences):
            break
    return [chunk for chunk in chunks if chunk]


# Split text by choosing the best break points
def _adaptive_sentence_chunks(
    paragraph_text: str,
    min_words: int,
    max_words: int,
    keyword_slack_words: int,
    keyword_min_overlap: int,
    boundary_scorer: LearnableBoundaryScorer,
    split_threshold: float,
) -> list[str]:
    sentences = _split_sentences(paragraph_text)
    if len(sentences) <= 1:
        return [paragraph_text.strip()]

    target_words = (min_words + max_words) // 2
    hard_max_words = max_words + max(0, keyword_slack_words)
    sentence_words = [len(sentence.split()) for sentence in sentences]
    sentence_keywords = [_keyword_tokens(sentence) for sentence in sentences]

    chunks: list[str] = []
    start = 0

    while start < len(sentences):
        end = start + 1
        current_words = sentence_words[start]
        current_keywords = set(sentence_keywords[start])
        candidates: list[tuple[int, float]] = []

        while end < len(sentences):
            next_sentence = sentences[end]
            next_words = sentence_words[end]
            next_keywords = sentence_keywords[end]
            overlap = len(current_keywords & next_keywords)
            effective_max_words = hard_max_words if overlap >= keyword_min_overlap else max_words

            # Score the current gap as a place to split
            if current_words >= min_words:
                boundary_prob = boundary_scorer.predict_proba(sentences[end - 1], next_sentence)
                size_penalty = abs(current_words - target_words) / max(1, max_words - min_words)
                adjusted_score = boundary_prob - 0.15 * size_penalty
                if boundary_prob >= split_threshold:
                    adjusted_score += 0.05
                candidates.append((end, adjusted_score))

            if current_words + next_words > effective_max_words:
                break

            current_words += next_words
            current_keywords.update(next_keywords)
            end += 1

        if end >= len(sentences):
            chunks.append(" ".join(sentences[start:]).strip())
            break

        # Pick the best split point from the options
        if candidates:
            split_at = max(candidates, key=lambda item: item[1])[0]
            if split_at <= start:
                split_at = end
        else:
            split_at = end

        chunks.append(" ".join(sentences[start:split_at]).strip())
        start = split_at

    return [chunk for chunk in chunks if chunk]


# Split text based on changes in meaning
def _semantic_similarity_chunks(
    paragraph_text: str,
    model_name: str,
    similarity_threshold: float,
    min_words: int,
    max_words: int,
    min_sentences: int,
    max_sentences: int,
) -> list[str]:
    sentences = _split_sentences(paragraph_text)
    if len(sentences) <= 1:
        return [" ".join(sentences).strip()]

    encoder = _get_sentence_encoder(model_name)
    embeddings = np.asarray(
        encoder.encode(sentences, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_embeddings: list[np.ndarray] = []
    current_words = 0

    for sentence, embedding in zip(sentences, embeddings, strict=False):
        sentence_words = len(sentence.split())

        # Check if the text is too long
        should_split_for_size = (
            current_sentences
            and (
                len(current_sentences) >= max_sentences
                or current_words + sentence_words > max_words
            )
        )

        # Check if meaning has changed too much
        should_split_for_semantics = False
        if current_sentences:
            centroid = np.mean(np.stack(current_embeddings, axis=0), axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            similarity = float(np.dot(embedding, centroid))
            enough_context = current_words >= min_words or len(current_sentences) >= min_sentences
            should_split_for_semantics = enough_context and similarity < similarity_threshold

        if should_split_for_size or should_split_for_semantics:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = []
            current_embeddings = []
            current_words = 0

        current_sentences.append(sentence)
        current_embeddings.append(embedding)
        current_words += sentence_words

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [chunk for chunk in chunks if chunk]


# Main function to split the corpus into small pieces
def chunk_corpus(corpus_df: pd.DataFrame, chunking_config: dict | None = None) -> list[dict]:
    chunking_config = chunking_config or {}
    strategy = chunking_config.get("strategy", "paragraph")
    # Set default values for all settings
    sentence_window_size = max(1, int(chunking_config.get("sentence_window_size", 3)))
    sentence_window_stride = max(1, int(chunking_config.get("sentence_window_stride", 2)))
    adaptive_min_words = max(1, int(chunking_config.get("adaptive_min_words", 80)))
    adaptive_max_words = max(adaptive_min_words, int(chunking_config.get("adaptive_max_words", 160)))
    adaptive_keyword_slack_words = max(0, int(chunking_config.get("adaptive_keyword_slack_words", 40)))
    adaptive_keyword_min_overlap = max(1, int(chunking_config.get("adaptive_keyword_min_overlap", 1)))
    learned_boundary_hidden_dim = max(8, int(chunking_config.get("learned_boundary_hidden_dim", 32)))
    learned_boundary_epochs = max(1, int(chunking_config.get("learned_boundary_epochs", 15)))
    learned_boundary_batch_size = max(16, int(chunking_config.get("learned_boundary_batch_size", 128)))
    learned_boundary_learning_rate = float(chunking_config.get("learned_boundary_learning_rate", 1e-3))
    learned_boundary_split_threshold = float(chunking_config.get("learned_boundary_split_threshold", 0.55))
    semantic_model = chunking_config.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
    semantic_similarity_threshold = float(chunking_config.get("semantic_similarity_threshold", 0.72))
    semantic_min_words = max(1, int(chunking_config.get("semantic_min_words", 80)))
    semantic_max_words = max(semantic_min_words, int(chunking_config.get("semantic_max_words", 160)))
    semantic_min_sentences = max(1, int(chunking_config.get("semantic_min_sentences", 2)))
    semantic_max_sentences = max(semantic_min_sentences, int(chunking_config.get("semantic_max_sentences", 6)))
    
    # Train the AI for adaptive split if needed
    learned_boundary_scorer = None
    if strategy == "adaptive_sentence":
        learned_boundary_scorer = train_boundary_scorer(
            corpus_df,
            hidden_dim=learned_boundary_hidden_dim,
            epochs=learned_boundary_epochs,
            batch_size=learned_boundary_batch_size,
            learning_rate=learned_boundary_learning_rate,
        )

    chunks: list[dict] = []
    for _, row in corpus_df.iterrows():
        wid = int(row["wikipedia_id"])
        paragraphs = row.get("paragraphs", [])

        if not isinstance(paragraphs, (list, np.ndarray)):
            paragraphs = []
        else:
            paragraphs = list(paragraphs)

        for para_idx, para_text in enumerate(paragraphs):
            if not isinstance(para_text, str):
                continue
            para_text = para_text.strip()
            if not para_text:
                continue

            # Pick the split method
            if strategy == "paragraph":
                derived_chunks = [para_text]
            elif strategy == "sentence_window":
                derived_chunks = _sentence_window_chunks(
                    para_text,
                    window_size=sentence_window_size,
                    stride=sentence_window_stride,
                )
            elif strategy == "adaptive_sentence":
                derived_chunks = _adaptive_sentence_chunks(
                    para_text,
                    min_words=adaptive_min_words,
                    max_words=adaptive_max_words,
                    keyword_slack_words=adaptive_keyword_slack_words,
                    keyword_min_overlap=adaptive_keyword_min_overlap,
                    boundary_scorer=learned_boundary_scorer,
                    split_threshold=learned_boundary_split_threshold,
                )
            elif strategy == "semantic_similarity":
                derived_chunks = _semantic_similarity_chunks(
                    para_text,
                    model_name=semantic_model,
                    similarity_threshold=semantic_similarity_threshold,
                    min_words=semantic_min_words,
                    max_words=semantic_max_words,
                    min_sentences=semantic_min_sentences,
                    max_sentences=semantic_max_sentences,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Save the final chunks
            for sub_idx, chunk_text in enumerate(derived_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                suffix = "" if strategy == "paragraph" else f"_c{sub_idx}"
                chunk_id = f"{wid}_{para_idx}{suffix}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "wikipedia_id": wid,
                        "paragraph_index": para_idx,
                        "text": chunk_text,
                        "chunking_strategy": strategy,
                    }
                )
    return chunks


# Split text into chunks with a set number of words
def chunk_corpus_fixed(corpus_df: pd.DataFrame, max_tokens: int = 256) -> list[dict]:
    chunks: list[dict] = []
    for _, row in corpus_df.iterrows():
        wid = int(row["wikipedia_id"])
        paragraphs = row.get("paragraphs", [])

        if not isinstance(paragraphs, (list, np.ndarray)):
            paragraphs = []
        else:
            paragraphs = list(paragraphs)

        full_text = " ".join(
            p.strip() for p in paragraphs if isinstance(p, str) and p.strip()
        )
        if not full_text:
            continue

        tokens = full_text.split()
        chunk_idx = 0
        for start in range(0, len(tokens), max_tokens):
            chunk_text = " ".join(tokens[start : start + max_tokens])
            chunks.append(
                {
                    "chunk_id": f"{wid}_{chunk_idx}",
                    "wikipedia_id": wid,
                    "paragraph_index": chunk_idx,
                    "text": chunk_text,
                }
            )
            chunk_idx += 1
    return chunks


# Choose a split method based on a name
def chunk_corpus_by_strategy(corpus_df: pd.DataFrame, strategy: str = "paragraph") -> list[dict]:
    if strategy == "paragraph":
        return chunk_corpus(corpus_df)
    if strategy.startswith("fixed-"):
        max_tokens = int(strategy.split("-", 1)[1])
        return chunk_corpus_fixed(corpus_df, max_tokens=max_tokens)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


# Group questions by how many names/places they mention
def classify_queries_ner(
    nq_records: list[dict],
    spacy_model: str = "en_core_web_sm",
    entity_threshold: int = 2,
) -> list[dict]:
    nlp = spacy.load(spacy_model)
    classified: list[dict] = []

    for rec in nq_records:
        query = rec.get("input", "")
        doc = nlp(query)
        entities = [ent.text for ent in doc.ents]
        entity_count = len(entities)
        group = "multi-entity" if entity_count >= entity_threshold else "single-entity"

        classified.append(
            {
                "query": query,
                "entity_count": entity_count,
                "entity_list": entities,
                "entity_group": group,
                "record": rec,
            }
        )
    return classified


# Pick a random set of questions from each group
def sample_balanced_queries(
    classified_queries: list[dict],
    sample_size_per_group: int = 500,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for q in classified_queries:
        groups.setdefault(q["entity_group"], []).append(q)

    sampled: list[dict] = []
    for group_name, members in groups.items():
        n = min(sample_size_per_group, len(members))
        sampled.extend(rng.sample(members, n))
    return sampled


# Find the correct chunk IDs for a question
def extract_ground_truth(query_record: dict) -> set[str]:
    gt_ids: set[str] = set()
    rec = query_record.get("record", query_record)
    for out in rec.get("output", []):
        for prov in out.get("provenance", []):
            wid = prov.get("wikipedia_id")
            start_par = prov.get("start_paragraph_id", 0)
            end_par = prov.get("end_paragraph_id", start_par)
            if wid is not None:
                for pidx in range(start_par, end_par + 1):
                    gt_ids.add(f"{int(wid)}_{pidx}")
    return gt_ids


# Run the full data prep process
def run_data_pipeline(config: dict, profiler: Profiler) -> dict:
    sample_per_group = config["queries"]["sample_size_per_group"]
    query_seed = config["queries"].get("seed", 42)

    # Load questions
    profiler.start_stage("data_loading")
    nq_records = load_kilt_nq_dev()
    profiler.end_stage("data_loading")

    # Label and group questions
    profiler.start_stage("ner_classification")
    candidate_size = min(len(nq_records), sample_per_group * 8)
    rng = random.Random(query_seed)
    candidates = rng.sample(nq_records, candidate_size)
    del nq_records

    classified = classify_queries_ner(
        candidates,
        spacy_model=config["spacy"]["model"],
        entity_threshold=config["spacy"]["entity_threshold"],
    )
    del candidates
    profiler.end_stage("ner_classification")

    # Pick final questions
    profiler.start_stage("query_sampling")
    sampled = sample_balanced_queries(
        classified,
        sample_size_per_group=sample_per_group,
        seed=query_seed,
    )
    profiler.end_stage("query_sampling")

    del classified
    gc.collect()

    # Build the Wikipedia pages
    profiler.start_stage("corpus_construction")
    corpus_df = build_reduced_corpus(
        [q["record"] for q in sampled],
        target_size=config["corpus"]["target_size"],
        cache_path=config["corpus"].get("local_corpus_cache"),
        seed=config["corpus"].get("seed", 42),
    )
    profiler.end_stage("corpus_construction")

    # Split pages into pieces
    profiler.start_stage("chunking")
    chunks = chunk_corpus(corpus_df, chunking_config=config.get("chunking", {}))
    corpus_size = len(corpus_df)
    del corpus_df
    gc.collect()
    profiler.end_stage("chunking")

    # Add correct answers to each question
    for q in sampled:
        q["ground_truth_chunk_ids"] = list(extract_ground_truth(q))

    # Remove the heavy raw records
    for q in sampled:
        q.pop("record", None)

    # Store stats
    profiler.data["metadata"]["corpus_size"] = corpus_size
    profiler.data["metadata"]["total_chunks"] = len(chunks)
    profiler.data["metadata"]["total_queries"] = len(sampled)
    profiler.data["metadata"]["chunking"] = config.get("chunking", {})
    profiler.data["metadata"]["queries_per_group"] = {
        g: sum(1 for q in sampled if q["entity_group"] == g)
        for g in {"single-entity", "multi-entity"}
    }

    return {
        "chunks": chunks,
        "sampled_queries": sampled,
    }