# Calculate scores and show summary tables from the logs

import json
import pandas as pd


# Read the JSON log file
def load_log(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Calculate average recall for each group and model
def compute_grouped_recall(log_data: dict) -> pd.DataFrame:
    rows = []
    for q in log_data["queries"]:
        group = q["entity_group"]
        for retriever in ("biencoder", "colbert"):
            recall_key = f"{retriever}_recall_at_k"
            if recall_key not in q:
                continue
            for k_str, recall_val in q[recall_key].items():
                rows.append(
                    {
                        "retriever": retriever,
                        "entity_group": group,
                        "k": int(k_str),
                        "recall": recall_val,
                    }
                )

    df = pd.DataFrame(rows)
    # Group results by model, category, and k value
    grouped = (
        df.groupby(["retriever", "entity_group", "k"])["recall"]
        .mean()
        .reset_index()
        .rename(columns={"recall": "mean_recall"})
    )
    return grouped


# Calculate average recall for everything together
def compute_overall_recall(log_data: dict) -> pd.DataFrame:
    rows = []
    for q in log_data["queries"]:
        for retriever in ("biencoder", "colbert"):
            recall_key = f"{retriever}_recall_at_k"
            if recall_key not in q:
                continue
            for k_str, recall_val in q[recall_key].items():
                rows.append(
                    {
                        "retriever": retriever,
                        "k": int(k_str),
                        "recall": recall_val,
                    }
                )

    df = pd.DataFrame(rows)
    overall = (
        df.groupby(["retriever", "k"])["recall"]
        .mean()
        .reset_index()
        .rename(columns={"recall": "mean_recall"})
    )
    overall["entity_group"] = "overall"
    return overall


# Create a table comparing models and showing the difference
def generate_summary_table(log_data: dict) -> pd.DataFrame:
    grouped = compute_grouped_recall(log_data)
    overall = compute_overall_recall(log_data)
    combined = pd.concat([grouped, overall], ignore_index=True)

    # Put biencoder and colbert results side by side
    pivot = combined.pivot_table(
        index=["entity_group", "k"],
        columns="retriever",
        values="mean_recall",
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(
        columns={"biencoder": "biencoder_recall", "colbert": "colbert_recall"}
    )
    # Find the difference between the two models
    pivot["delta"] = pivot["colbert_recall"] - pivot["biencoder_recall"]

    # Sort groups in a clear order
    group_order = {"single-entity": 0, "multi-entity": 1, "overall": 2}
    pivot["_sort"] = pivot["entity_group"].map(group_order)
    pivot = pivot.sort_values(["_sort", "k"]).drop(columns="_sort").reset_index(drop=True)

    return pivot


# Main function to get the final summary
def run_evaluation(log_path: str) -> pd.DataFrame:
    log_data = load_log(log_path)
    return generate_summary_table(log_data)