# Helper functions to run experiments

import gc
import json
from copy import deepcopy
from pathlib import Path

import yaml

from src.biencoder_retrieval import run_biencoder_retrieval
from src.colbert_retrieval import run_colbert_retrieval
from src.data_pipeline import run_data_pipeline
from src.evaluation import run_evaluation
from src.profiler import Profiler
from src.visualize import run_visualization
from src.generation import run_generation_step


# Open and read the YAML config file
def load_config(path: str = "configs/experiment_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Merge two dictionaries without changing the original
def deep_update(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


# Setup folders for saving results
def configure_run_paths(base_config: dict, run_subdir: str, cache_suffix: str | None = None) -> dict:
    config = deepcopy(base_config)
    result_root = Path(base_config["paths"]["results_dir"]) / run_subdir
    result_root.mkdir(parents=True, exist_ok=True)

    config["paths"]["results_dir"] = str(result_root)
    config["paths"]["faiss_index_dir"] = str(result_root / "faiss_index")
    config["paths"]["colbert_index_dir"] = str(result_root / "colbert_index")
    config["paths"]["json_log"] = str(result_root / "benchmark_log.json")
    config["paths"]["charts_dir"] = str(result_root / "charts")
    config["paths"]["csv_output"] = str(result_root / "summary_statistics.csv")

    # Add a suffix to the cache filename if needed
    if cache_suffix:
        cache_file = Path(base_config["corpus"]["local_corpus_cache"])
        config["corpus"]["local_corpus_cache"] = str(
            cache_file.with_name(f"{cache_file.stem}_{cache_suffix}{cache_file.suffix}")
        )
    return config


# Change settings and folders for one test
def build_variant_config(
    base_config: dict,
    run_subdir: str,
    overrides: dict | None = None,
    cache_suffix: str | None = None,
) -> dict:
    config = deep_update(base_config, overrides or {})
    return configure_run_paths(config, run_subdir=run_subdir, cache_suffix=cache_suffix)


# Save search results to the log
def _log_query_results(sampled_queries: list[dict], profiler: Profiler, config: dict) -> None:
    for q in sampled_queries:
        record = {
            "query": q["query"],
            "entity_count": q["entity_count"],
            "entity_list": q["entity_list"],
            "entity_group": q["entity_group"],
            "ground_truth_chunk_ids": q["ground_truth_chunk_ids"],
        }

        if "generated_answer" in q:
            record["generated_answer"] = q["generated_answer"]

        for key in (
            "biencoder_retrieved_ids",
            "biencoder_recall_at_k",
            "biencoder_latency_ms",
            "colbert_retrieved_ids",
            "colbert_recall_at_k",
            "colbert_latency_ms",
        ):
            if key in q:
                record[key] = q[key]
        profiler.log_query(record)

    profiler.data["metadata"]["models"] = config["models"]
    profiler.data["metadata"]["k_values"] = config["retrieval"]["k_values"]


# Run the whole test from start to finish
def run_full_benchmark(
    config: dict,
    save_outputs: bool = True,
    generate_visualizations: bool = True,
) -> dict:
    profiler = Profiler(config=config)

    # Get data and queries
    pipeline_out = run_data_pipeline(config, profiler)
    chunks = pipeline_out["chunks"]
    sampled_queries = pipeline_out["sampled_queries"]
    del pipeline_out
    gc.collect()

    # Run different search methods
    run_biencoder_retrieval(chunks, sampled_queries, config, profiler)
    run_colbert_retrieval(chunks, sampled_queries, config, profiler)

    # Let the LLM answer questions
    profiler.start_stage("llm_generation")
    run_generation_step(sampled_queries, chunks, top_k=3)
    profiler.end_stage("llm_generation")

    _log_query_results(sampled_queries, profiler, config)

    log_path = config["paths"]["json_log"]
    summary = None
    output_paths = {}

    # Save files and draw charts
    if save_outputs:
        profiler.save(log_path)
        summary = run_evaluation(log_path)
        if generate_visualizations:
            output_paths = run_visualization(
                log_path=log_path,
                charts_dir=config["paths"]["charts_dir"],
                csv_path=config["paths"]["csv_output"],
            )
    else:
        summary = run_evaluation_from_data(profiler.data)

    return {
        "config": config,
        "profiler_data": profiler.data,
        "summary": summary,
        "log_path": log_path,
        "output_paths": output_paths,
        "chunks": chunks,
        "sampled_queries": sampled_queries,
    }


# Evaluate results without saving to disk first
def run_evaluation_from_data(log_data: dict):
    temp_path = Path("results") / "_temp_eval_log.json"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
    return run_evaluation(str(temp_path))