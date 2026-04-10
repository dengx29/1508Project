# Tools to combine results from search models and graphs

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.biencoder_retrieval import compute_recall_at_k


# Combine multiple search lists into one better list
def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    max_candidates: int | None = None,
) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    weights = weights or {}

    # Score each item based on its position in every list
    for source, ranked_ids in rankings.items():
        weight = weights.get(source, 1.0)
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            # Items at the top of a list get more points
            scores[chunk_id] += weight / (rrf_k + rank)

    # Sort items by their final score
    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused_ids = [chunk_id for chunk_id, _ in fused]
    
    # Return only the number of results requested
    if max_candidates is not None:
        return fused_ids[:max_candidates]
    return fused_ids


# Mix results from a search model and a graph search
def run_hybrid_retrieval(
    sampled_queries: list[dict],
    *,
    primary_key: str,
    graph_key: str = "graph",
    output_prefix: str,
    k_values: list[int],
    primary_weight: float = 1.0,
    graph_weight: float = 0.3,
    rrf_k: int = 60,
) -> None:
    max_k = max(k_values)
    primary_ids_field = f"{primary_key}_retrieved_ids"
    graph_ids_field = f"{graph_key}_retrieved_ids"
    output_ids_field = f"{output_prefix}_retrieved_ids"
    output_recall_field = f"{output_prefix}_recall_at_k"

    for q in sampled_queries:
        primary_ranking = q.get(primary_ids_field, [])
        graph_ranking = q.get(graph_ids_field, [])
        gt_ids = set(q.get("ground_truth_chunk_ids", []))

        # Merge the two lists
        fused_ids = reciprocal_rank_fusion(
            {
                primary_key: primary_ranking,
                graph_key: graph_ranking,
            },
            weights={
                primary_key: primary_weight,
                graph_key: graph_weight,
            },
            rrf_k=rrf_k,
            max_candidates=max_k,
        )

        # Save the combined list and the score
        q[output_ids_field] = fused_ids
        q[output_recall_field] = {
            str(k): compute_recall_at_k(fused_ids, gt_ids, k) for k in k_values
        }


# Create a table of combined search results
def hybrid_summary_dataframe(sampled_queries: list[dict], output_prefix: str) -> pd.DataFrame:
    recall_field = f"{output_prefix}_recall_at_k"
    rows = []
    for q in sampled_queries:
        for k_str, recall in q.get(recall_field, {}).items():
            rows.append(
                {
                    "entity_group": q["entity_group"],
                    "k": int(k_str),
                    "hybrid_recall": recall,
                }
            )

    df = pd.DataFrame(rows)
    # Get average scores for each group
    grouped = (
        df.groupby(["entity_group", "k"])["hybrid_recall"]
        .mean()
        .reset_index()
    )

    # Get the average score for all queries
    overall = (
        df.groupby("k")["hybrid_recall"]
        .mean()
        .reset_index()
    )
    overall["entity_group"] = "overall"
    return pd.concat([grouped, overall], ignore_index=True)


# Only combine lists for specific types of queries
def run_selective_hybrid_retrieval(
    sampled_queries: list[dict],
    *,
    primary_key: str,
    graph_key: str = "graph",
    output_prefix: str,
    k_values: list[int],
    fuse_entity_groups: set[str] | None = None,
    primary_weight: float = 1.0,
    graph_weight: float = 0.7,
    rrf_k: int = 60,
) -> None:
    max_k = max(k_values)
    primary_ids_field = f"{primary_key}_retrieved_ids"
    graph_ids_field = f"{graph_key}_retrieved_ids"
    output_ids_field = f"{output_prefix}_retrieved_ids"
    output_recall_field = f"{output_prefix}_recall_at_k"
    fuse_entity_groups = fuse_entity_groups or set()

    for q in sampled_queries:
        primary_ranking = q.get(primary_ids_field, [])
        gt_ids = set(q.get("ground_truth_chunk_ids", []))

        # Combine results only if the query group matches
        if q.get("entity_group") in fuse_entity_groups:
            graph_ranking = q.get(graph_ids_field, [])
            fused_ids = reciprocal_rank_fusion(
                {
                    primary_key: primary_ranking,
                    graph_key: graph_ranking,
                },
                weights={
                    primary_key: primary_weight,
                    graph_key: graph_weight,
                },
                rrf_k=rrf_k,
                max_candidates=max_k,
            )
        else:
            # Otherwise, use only the main search list
            fused_ids = primary_ranking[:max_k]

        q[output_ids_field] = fused_ids
        q[output_recall_field] = {
            str(k): compute_recall_at_k(fused_ids, gt_ids, k) for k in k_values
        }