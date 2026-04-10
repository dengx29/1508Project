# Functions to compare search results from different tests

import json
from pathlib import Path

import pandas as pd


# Read a log file from the computer
def load_log(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Get the recall score for a specific k value
def _recall_value(query_record: dict, retriever: str, k: int) -> float:
    key = f"{retriever}_recall_at_k"
    return float(query_record.get(key, {}).get(str(k), 0.0))


# Compare two different test logs to see which is better
def compare_variant_logs(
    baseline_log_path: str,
    candidate_log_path: str,
    retriever: str = "colbert",
    k: int = 10,
) -> pd.DataFrame:
    baseline = load_log(baseline_log_path)
    candidate = load_log(candidate_log_path)

    # Match queries that appear in both files
    baseline_queries = {q["query"]: q for q in baseline["queries"]}
    candidate_queries = {q["query"]: q for q in candidate["queries"]}
    shared_queries = sorted(set(baseline_queries) & set(candidate_queries))

    rows = []
    for query in shared_queries:
        baseline_q = baseline_queries[query]
        candidate_q = candidate_queries[query]
        baseline_recall = _recall_value(baseline_q, retriever, k)
        candidate_recall = _recall_value(candidate_q, retriever, k)
        delta = candidate_recall - baseline_recall

        # Check if the score went up or down
        if delta > 0:
            outcome = "improved"
        elif delta < 0:
            outcome = "regressed"
        else:
            outcome = "unchanged"

        # Save details for each query
        rows.append(
            {
                "query": query,
                "entity_group": baseline_q["entity_group"],
                "ground_truth_chunk_ids": baseline_q["ground_truth_chunk_ids"],
                "baseline_recall": baseline_recall,
                "candidate_recall": candidate_recall,
                "delta": delta,
                "outcome": outcome,
                "baseline_top_5": baseline_q.get(f"{retriever}_retrieved_ids", [])[:5],
                "candidate_top_5": candidate_q.get(f"{retriever}_retrieved_ids", [])[:5],
            }
        )

    # Return a table sorted by the best improvements
    return pd.DataFrame(rows).sort_values(["delta", "query"], ascending=[False, True]).reset_index(drop=True)


# Create a short summary of the comparison results
def summarize_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame(columns=["entity_group", "outcome", "count", "mean_delta"])

    # Count outcomes for each group
    summary = (
        comparison_df.groupby(["entity_group", "outcome"])
        .agg(count=("query", "count"), mean_delta=("delta", "mean"))
        .reset_index()
        .sort_values(["entity_group", "outcome"])
        .reset_index(drop=True)
    )
    return summary


# Save the comparison tables to CSV files
def save_comparison_outputs(
    comparison_df: pd.DataFrame,
    output_dir: str,
    prefix: str,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    detail_path = output_root / f"{prefix}_details.csv"
    summary_path = output_root / f"{prefix}_summary.csv"

    # Save files to disk
    comparison_df.to_csv(detail_path, index=False)
    summarize_comparison(comparison_df).to_csv(summary_path, index=False)

    return {
        "details_csv": str(detail_path),
        "summary_csv": str(summary_path),
    }