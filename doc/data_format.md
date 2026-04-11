# Data Format Reference

This document describes the data formats used throughout the benchmark pipeline.

## Input Data

### KILT NaturalQuestions

Downloaded automatically from HuggingFace (`facebook/kilt_tasks`, `nq` config, `validation` split).

Each record contains:
- `input`: The natural language question
- `output`: List of answer objects, each with `provenance` containing:
  - `wikipedia_id`: Gold Wikipedia page identifier
  - `start_paragraph_id` / `end_paragraph_id`: Passage boundaries

### KILT Wikipedia Corpus

Streamed from HuggingFace (`facebook/kilt_knowledgesource`). Each page has:
- `wikipedia_id`: Unique page identifier
- `wikipedia_title`: Page title
- `text`: List of paragraph strings (index 0 = title, index 1+ = body paragraphs)

## Intermediate Data

### Corpus Cache (`data/corpus_cache.parquet`)

Parquet file caching the reduced Wikipedia corpus to avoid re-streaming. Columns:
- `wikipedia_id` (str): Page identifier
- `wikipedia_title` (str): Page title
- `text` (list[str]): Paragraph list

### Chunk Records

In-memory list of dicts, each representing one retrievable unit:

```python
{
    "chunk_id": "12345_p2",        # {wikipedia_id}_p{paragraph_index}
    "wikipedia_id": "12345",
    "text": "The paragraph content...",
    "title": "Page Title"
}
```

### Query Records

In-memory list of dicts after NER classification:

```python
{
    "query": "Who directed Inception?",
    "entity_count": 1,
    "entity_list": ["Inception"],
    "entity_group": "single-entity",   # or "multi-entity"
    "ground_truth_chunk_ids": ["12345_p3", "12345_p4"],
    # After retrieval:
    "biencoder_retrieved_ids": [...],
    "biencoder_recall_at_k": {"1": 0.0, "5": 1.0, "10": 1.0, "20": 1.0},
    "biencoder_latency_ms": 4.2,
    "colbert_retrieved_ids": [...],
    "colbert_recall_at_k": {"1": 1.0, "5": 1.0, "10": 1.0, "20": 1.0},
    "colbert_latency_ms": 23.5
}
```

## Output Data

### Benchmark Log (`results/benchmark_log.json`)

Top-level structure:

```json
{
    "metadata": {
        "timestamp": "2026-04-01T12:00:00+00:00",
        "config": { ... },
        "gpu_device": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "gpu_total_vram_bytes": 8585740288,
        "corpus_size": 224,
        "total_chunks": 11901,
        "total_queries": 260,
        "queries_per_group": {"single-entity": 130, "multi-entity": 130},
        "models": {
            "biencoder": "sentence-transformers/all-MiniLM-L6-v2",
            "colbert": "colbert-ir/colbertv2.0"
        },
        "k_values": [1, 5, 10, 20]
    },
    "stages": {
        "data_loading": {
            "duration_seconds": 12.5,
            "peak_vram_bytes": null,
            "rss_bytes": 524288000
        },
        "biencoder_encoding": { ... },
        "biencoder_retrieval": { ... },
        "colbert_indexing": { ... },
        "colbert_retrieval": { ... }
    },
    "queries": [ ... ],
    "disk_sizes": {
        "faiss_index": 18300000,
        "colbert_index": 0
    }
}
```

### Summary Statistics CSV (`results/summary_statistics.csv`)

| Column | Description |
|---|---|
| `entity_group` | "single-entity", "multi-entity", or "overall" |
| `k` | Top-k value |
| `biencoder_recall` | Mean Recall@k for bi-encoder |
| `colbert_recall` | Mean Recall@k for ColBERTv2 |
| `delta` | ColBERT recall minus bi-encoder recall |

### Ablation Log Files

Each ablation condition produces a separate JSON log (`results/ablation/<condition>_log.json`) with the same structure as the main benchmark log, but using the smaller test set.
