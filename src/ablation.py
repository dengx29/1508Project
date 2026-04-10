# Create a small test set for experiments
# Reuses the saved corpus to save memory

import gc
import random

import pandas as pd

from src.data_pipeline import (
    chunk_corpus,
    chunk_corpus_by_strategy,
    classify_queries_ner,
    extract_ground_truth,
    load_kilt_nq_dev,
    sample_balanced_queries,
)


# Build a small test set from the local cache
def build_ablation_test_set(config: dict) -> dict:
    ts = config["test_set"]
    seed = ts.get("seed", 42)
    queries_per_group = ts.get("queries_per_group", 15)
    main_corpus_cache = ts.get("main_corpus_cache", "data/corpus_cache.parquet")

    # Load the saved corpus from the local file
    corpus_df = pd.read_parquet(main_corpus_cache)
    corpus_page_ids = set(corpus_df["wikipedia_id"].astype(int).tolist())
    print(f"  Loaded cached corpus: {len(corpus_df)} pages")

    # Split the corpus into chunks to get valid IDs
    all_chunks = chunk_corpus(corpus_df)
    valid_chunk_ids = {c["chunk_id"] for c in all_chunks}
    del all_chunks

    # Load questions from the dataset
    nq_records = load_kilt_nq_dev()

    # Find questions where the answer pages are in our local file
    eligible = []
    for rec in nq_records:
        gold_ids = set()
        for out in rec.get("output", []):
            for prov in out.get("provenance", []):
                wid = prov.get("wikipedia_id")
                if wid is not None:
                    gold_ids.add(int(wid))
        # Only keep the question if we have all its answer pages
        if gold_ids and gold_ids.issubset(corpus_page_ids):
            eligible.append(rec)
    del nq_records
    print(f"  Eligible queries (gold pages in corpus): {len(eligible)}")

    # Label questions based on the names/entities they contain
    classified = classify_queries_ner(
        eligible,
        spacy_model=config["spacy"]["model"],
        entity_threshold=config["spacy"]["entity_threshold"],
    )
    del eligible

    # Select a balanced mix of questions
    sampled = sample_balanced_queries(
        classified,
        sample_size_per_group=queries_per_group,
        seed=seed,
    )
    del classified
    gc.collect()

    # Find the specific chunk IDs for the answers
    for q in sampled:
        raw_gt = extract_ground_truth(q)
        q["ground_truth_chunk_ids"] = [cid for cid in raw_gt if cid in valid_chunk_ids]
        q.pop("record", None)

    return {
        "sampled_queries": sampled,
        "corpus_df": corpus_df,
    }