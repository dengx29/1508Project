# Build a simple knowledge graph for searching through chunks

import gc
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import spacy

from src.biencoder_retrieval import compute_recall_at_k
from src.benchmark_runner import load_config
from src.data_pipeline import run_data_pipeline
from src.profiler import Profiler

_NORMALIZE_RE = re.compile(r"\s+")

# Clean up entity text and make it lowercase
def normalize_entity(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.strip().lower())

# Find and list all names or places in each chunk
def extract_chunk_entities(chunks: list[dict], spacy_model: str) -> dict[str, list[str]]:
    nlp = spacy.load(spacy_model)
    chunk_entities: dict[str, list[str]] = {}
    for chunk in chunks:
        doc = nlp(chunk["text"])
        entities = []
        seen = set()
        for ent in doc.ents:
            normalized = normalize_entity(ent.text)
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            entities.append(normalized)
        chunk_entities[chunk["chunk_id"]] = entities
    return chunk_entities

# Connect chunks based on shared entities
def build_knowledge_graph(chunks: list[dict], spacy_model: str = "en_core_web_sm") -> dict:
    chunk_entities = extract_chunk_entities(chunks, spacy_model=spacy_model)
    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    entity_to_pages: dict[str, set[int]] = defaultdict(set)
    entity_degree: Counter = Counter()
    cooccurrence_weights: Counter = Counter()

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    for chunk_id, entities in chunk_entities.items():
        unique_entities = sorted(set(entities))
        chunk = chunk_lookup[chunk_id]
        for entity in unique_entities:
            entity_to_chunks[entity].add(chunk_id)
            entity_to_pages[entity].add(chunk["wikipedia_id"])
            entity_degree[entity] += 1

        # Track which entities appear together in the same chunk
        for idx, entity_a in enumerate(unique_entities):
            for entity_b in unique_entities[idx + 1:]:
                cooccurrence_weights[(entity_a, entity_b)] += 1
                cooccurrence_weights[(entity_b, entity_a)] += 1

    # Save the graph structure as a dictionary
    graph = {
        "metadata": {
            "num_chunks": len(chunks),
            "num_entities": len(entity_to_chunks),
            "spacy_model": spacy_model,
        },
        "chunk_entities": chunk_entities,
        "entity_to_chunks": {key: sorted(value) for key, value in entity_to_chunks.items()},
        "entity_to_pages": {key: sorted(value) for key, value in entity_to_pages.items()},
        "entity_degree": dict(entity_degree),
        "cooccurrence_weights": {f"{src}|||{dst}": weight for (src, dst), weight in cooccurrence_weights.items()},
    }
    return graph

# Extract entities from the user query
def extract_query_entities(query: str, spacy_model: str = "en_core_web_sm") -> list[str]:
    nlp = spacy.load(spacy_model)
    doc = nlp(query)
    entities = []
    seen = set()
    for ent in doc.ents:
        normalized = normalize_entity(ent.text)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        entities.append(normalized)
    return entities

# Check how often two entities appear together
def _cooccurrence_lookup(graph: dict, source: str, target: str) -> int:
    return int(graph["cooccurrence_weights"].get(f"{source}|||{target}", 0))

# Search for chunks using the graph
def retrieve_top_k_graph(
    graph: dict,
    query: str,
    k: int = 20,
    spacy_model: str = "en_core_web_sm",
) -> tuple[list[str], list[str]]:
    query_entities = extract_query_entities(query, spacy_model=spacy_model)
    if not query_entities:
        return [], []

    scores: defaultdict[str, float] = defaultdict(float)
    chunk_entities = graph["chunk_entities"]
    num_chunks = graph["metadata"].get("num_chunks", 1000)

    # Calculate importance of an entity (IDF)
    def get_idf(ent: str) -> float:
        degree = graph["entity_degree"].get(ent, 0)
        if degree == 0:
            return 0.0
        return max(0.1, math.log(num_chunks / degree))

    # Give points for direct matches
    for entity in query_entities:
        idf = get_idf(entity)
        for chunk_id in graph["entity_to_chunks"].get(entity, []):
            scores[chunk_id] += 10.0 * idf

    # Give points for related entities through the graph
    for chunk_id, entities in chunk_entities.items():
        for query_entity in query_entities:
            for entity in entities:
                if entity == query_entity:
                    continue
                weight = _cooccurrence_lookup(graph, query_entity, entity)
                # Only use strong links
                if weight >= 2:
                    idf = get_idf(entity)
                    scores[chunk_id] += (math.log1p(weight) * idf) * 0.5

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in ranked[:k]], query_entities

# Run graph search for a list of queries
def run_graph_retrieval(
    graph: dict,
    sampled_queries: list[dict],
    k_values: list[int],
    spacy_model: str = "en_core_web_sm",
) -> None:
    max_k = max(k_values)
    for q in sampled_queries:
        retrieved_ids, query_entities = retrieve_top_k_graph(
            graph=graph,
            query=q["query"],
            k=max_k,
            spacy_model=spacy_model,
        )
        gt_ids = set(q["ground_truth_chunk_ids"])
        q["graph_query_entities"] = query_entities
        q["graph_retrieved_ids"] = retrieved_ids
        q["graph_recall_at_k"] = {
            str(k): compute_recall_at_k(retrieved_ids, gt_ids, k) for k in k_values
        }

# Setup the graph using the main config
def build_graph_from_config(config: dict) -> dict:
    profiler = Profiler(config=config)
    pipeline_out = run_data_pipeline(config, profiler)
    chunks = pipeline_out["chunks"]
    sampled_queries = pipeline_out["sampled_queries"]
    graph = build_knowledge_graph(chunks, spacy_model=config["spacy"]["model"])
    return {
        "chunks": chunks,
        "sampled_queries": sampled_queries,
        "graph": graph,
    }

# Save the graph to a JSON file
def save_graph(graph: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

# Get average scores for the graph search
def graph_summary_dataframe(sampled_queries: list[dict]) -> pd.DataFrame:
    rows = []
    for q in sampled_queries:
        for k_str, recall in q.get("graph_recall_at_k", {}).items():
            rows.append(
                {
                    "entity_group": q["entity_group"],
                    "k": int(k_str),
                    "graph_recall": recall,
                }
            )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["entity_group", "k"])["graph_recall"]
        .mean()
        .reset_index()
    )

    overall = (
        df.groupby("k")["graph_recall"]
        .mean()
        .reset_index()
    )
    overall["entity_group"] = "overall"
    return pd.concat([grouped, overall], ignore_index=True)