# Ablation testing script for retrieval performance under different conditions
import gc
import json
import os

import pandas as pd
import yaml

from src.ablation import build_ablation_test_set
from src.biencoder_retrieval import (
    build_faiss_index,
    compute_recall_at_k,
    encode_chunks,
    retrieve_top_k,
)
from src.colbert_retrieval import (
    build_colbert_index,
    compute_recall_at_k as colbert_recall_at_k,
    retrieve_top_k_colbert,
)
from src.data_pipeline import chunk_corpus_by_strategy
from src.profiler import Profiler


# Run search tests and save results
def _run_retrieval_condition(
    chunks: list[dict],
    sampled_queries: list[dict],
    config: dict,
    k_values: list[int],
    condition_dir: str,
) -> dict:
    os.makedirs(condition_dir, exist_ok=True)
    profiler = Profiler(config=config)
    max_k = max(k_values)

    # Bi-encoder steps
    profiler.start_stage("biencoder_encoding")
    model, embeddings = encode_chunks(chunks, model_name=config["models"]["biencoder"])
    profiler.end_stage("biencoder_encoding")

    profiler.start_stage("faiss_indexing")
    index = build_faiss_index(embeddings)
    del embeddings
    gc.collect()
    profiler.end_stage("faiss_indexing")

    profiler.start_stage("biencoder_retrieval")
    for q in sampled_queries:
        gt_ids = set(q["ground_truth_chunk_ids"])
        retrieved_ids, latency_ms = retrieve_top_k(model, index, chunks, q["query"], k=max_k)
        q["biencoder_retrieved_ids"] = retrieved_ids
        q["biencoder_recall_at_k"] = {str(k): compute_recall_at_k(retrieved_ids, gt_ids, k) for k in k_values}
        q["biencoder_latency_ms"] = round(latency_ms, 4)
    profiler.end_stage("biencoder_retrieval")

    del model, index
    gc.collect()

    # ColBERT steps
    colbert_index_dir = os.path.join(condition_dir, "colbert_index")
    profiler.start_stage("colbert_indexing")
    bsize = config.get("colbert", {}).get("bsize", 32)
    rag = build_colbert_index(chunks, index_path=colbert_index_dir, model_name=config["models"]["colbert"], bsize=bsize)
    profiler.end_stage("colbert_indexing")

    profiler.start_stage("colbert_retrieval")
    for q in sampled_queries:
        gt_ids = set(q["ground_truth_chunk_ids"])
        retrieved_ids, latency_ms = retrieve_top_k_colbert(rag, q["query"], k=max_k)
        q["colbert_retrieved_ids"] = retrieved_ids
        q["colbert_recall_at_k"] = {str(k): colbert_recall_at_k(retrieved_ids, gt_ids, k) for k in k_values}
        q["colbert_latency_ms"] = round(latency_ms, 4)
    profiler.end_stage("colbert_retrieval")

    del rag
    gc.collect()

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save the log data
    profiler.data["metadata"]["total_chunks"] = len(chunks)
    profiler.data["metadata"]["total_queries"] = len(sampled_queries)
    profiler.data["metadata"]["k_values"] = k_values
    profiler.data["metadata"]["models"] = config["models"]

    for q in sampled_queries:
        record = {
            "query": q["query"],
            "entity_count": q["entity_count"],
            "entity_list": q["entity_list"],
            "entity_group": q["entity_group"],
            "ground_truth_chunk_ids": q["ground_truth_chunk_ids"],
        }
        for key in ("biencoder_retrieved_ids", "biencoder_recall_at_k", "biencoder_latency_ms",
                     "colbert_retrieved_ids", "colbert_recall_at_k", "colbert_latency_ms"):
            if key in q:
                record[key] = q[key]
        profiler.log_query(record)

    log_path = os.path.join(condition_dir, "benchmark_log.json")
    profiler.save(log_path)
    return profiler.data


# Clear old results to reuse queries
def _reset_query_results(queries: list[dict]) -> None:
    keys_to_remove = [
        "biencoder_retrieved_ids", "biencoder_recall_at_k", "biencoder_latency_ms",
        "colbert_retrieved_ids", "colbert_recall_at_k", "colbert_latency_ms",
    ]
    for q in queries:
        for k in keys_to_remove:
            q.pop(k, None)


# Keep correct pages and add random ones
def _subsample_corpus(corpus_df: pd.DataFrame, size: int, gold_page_ids: set[int], seed: int) -> pd.DataFrame:
    if size >= len(corpus_df):
        return corpus_df

    is_gold = corpus_df["wikipedia_id"].astype(int).isin(gold_page_ids)
    gold_df = corpus_df[is_gold]
    distractor_df = corpus_df[~is_gold]
    needed = max(0, size - len(gold_df))
    if needed < len(distractor_df):
        distractor_df = distractor_df.sample(n=needed, random_state=seed)
    return pd.concat([gold_df, distractor_df], ignore_index=True)


# Get page IDs from chunk strings
def _get_gold_page_ids(sampled_queries: list[dict]) -> set[int]:
    ids = set()
    for q in sampled_queries:
        for cid in q.get("ground_truth_chunk_ids", []):
            wid = cid.rsplit("_", 1)[0]
            ids.add(int(wid))
    return ids


# Test how corpus size affects results
def run_corpus_scale_ablation(config: dict, test_data: dict, results_dir: str) -> list[dict]:
    dim_config = config["dimensions"]["corpus_scale"]
    corpus_sizes = sorted(dim_config["values"])
    chunking = dim_config.get("default_chunking", "paragraph")
    k_values = config["dimensions"]["top_k"]["values"]
    seed = config["test_set"].get("seed", 42)

    corpus_df = test_data["corpus_df"]
    queries = test_data["sampled_queries"]
    gold_ids = _get_gold_page_ids(queries)

    all_results = []
    for size in corpus_sizes:
        print(f"\n  [corpus_scale] Running with {size} pages...")
        condition_dir = os.path.join(results_dir, "corpus_scale", str(size))

        sub_corpus = _subsample_corpus(corpus_df, size, gold_ids, seed)
        chunks = chunk_corpus_by_strategy(sub_corpus, chunking)
        print(f"    Corpus: {len(sub_corpus)} pages, {len(chunks)} chunks")

        queries_copy = [dict(q) for q in queries]
        _reset_query_results(queries_copy)

        data = _run_retrieval_condition(chunks, queries_copy, config, k_values, condition_dir)
        all_results.append({"dimension": "corpus_scale", "condition": str(size), "data": data})
        del chunks, queries_copy, sub_corpus
        gc.collect()

    return all_results


# Test how chunking methods affect results
def run_chunk_granularity_ablation(config: dict, test_data: dict, results_dir: str) -> list[dict]:
    dim_config = config["dimensions"]["chunk_granularity"]
    strategies = dim_config["values"]
    default_size = dim_config.get("default_corpus_size", 50)
    k_values = config["dimensions"]["top_k"]["values"]
    seed = config["test_set"].get("seed", 42)

    corpus_df = test_data["corpus_df"]
    queries = test_data["sampled_queries"]
    gold_ids = _get_gold_page_ids(queries)
    sub_corpus = _subsample_corpus(corpus_df, default_size, gold_ids, seed)

    all_results = []
    for strategy in strategies:
        print(f"\n  [chunk_granularity] Running with strategy={strategy}...")
        condition_dir = os.path.join(results_dir, "chunk_granularity", strategy)

        chunks = chunk_corpus_by_strategy(sub_corpus, strategy)
        print(f"    Chunks: {len(chunks)}")

        queries_copy = [dict(q) for q in queries]
        _reset_query_results(queries_copy)

        data = _run_retrieval_condition(chunks, queries_copy, config, k_values, condition_dir)
        all_results.append({"dimension": "chunk_granularity", "condition": strategy, "data": data})
        del chunks, queries_copy
        gc.collect()

    return all_results


# Test results at different k levels
def run_top_k_ablation(config: dict, test_data: dict, results_dir: str) -> list[dict]:
    k_values = config["dimensions"]["top_k"]["values"]
    chunking = config["dimensions"]["corpus_scale"].get("default_chunking", "paragraph")
    seed = config["test_set"].get("seed", 42)

    corpus_df = test_data["corpus_df"]
    queries = test_data["sampled_queries"]
    gold_ids = _get_gold_page_ids(queries)
    default_size = config["dimensions"]["chunk_granularity"].get("default_corpus_size", 50)
    sub_corpus = _subsample_corpus(corpus_df, default_size, gold_ids, seed)

    print(f"\n  [top_k] Running with k={k_values}...")
    condition_dir = os.path.join(results_dir, "top_k", "default")

    chunks = chunk_corpus_by_strategy(sub_corpus, chunking)
    print(f"    Chunks: {len(chunks)}")

    queries_copy = [dict(q) for q in queries]
    _reset_query_results(queries_copy)

    data = _run_retrieval_condition(chunks, queries_copy, config, k_values, condition_dir)
    all_results = [{"dimension": "top_k", "condition": "default", "data": data}]
    del chunks, queries_copy
    gc.collect()
    return all_results


# Main script execution
def main():
    with open("configs/ablation_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Ablation config loaded.")

    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Prepare data
    print("\n1. Building ablation test set")
    test_data = build_ablation_test_set(config)
    print(f"  Queries: {len(test_data['sampled_queries'])}")
    print(f"  Corpus pages: {len(test_data['corpus_df'])}")

    # Run all tests
    all_results = []

    print("\n2. Corpus Scale Ablation")
    all_results.extend(run_corpus_scale_ablation(config, test_data, results_dir))

    print("\n3. Chunk Granularity Ablation")
    all_results.extend(run_chunk_granularity_ablation(config, test_data, results_dir))

    print("\n4. Top-k Depth Ablation")
    all_results.extend(run_top_k_ablation(config, test_data, results_dir))

    # Create charts
    print("\n5. Generating Ablation Visualizations")
    from src.ablation_visualize import run_ablation_visualization
    run_ablation_visualization(all_results, config)

    print("\nDONE")


if __name__ == "__main__":
    main()