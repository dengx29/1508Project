# Architecture Overview

This document describes the modular architecture of the ColBERTv2 vs Bi-Encoder retrieval benchmark system.

## System Architecture

```
                    ┌─────────────────┐
                    │  Configuration  │
                    │  (YAML files)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Data Pipeline  │
                    │ (data_pipeline) │
                    └──┬──────────┬───┘
                       │          │
              ┌────────▼──┐  ┌───▼──────────┐
              │ Bi-Encoder │  │  ColBERTv2   │
              │  + FAISS   │  │ + RAGatouille│
              └────────┬───┘  └───┬──────────┘
                       │          │
                    ┌──▼──────────▼───┐
                    │   Evaluation    │
                    │  (Recall@k)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Visualization  │
                    │ (charts + CSV)  │
                    └─────────────────┘
```

## Module Descriptions

### `src/data_pipeline.py`

Handles the complete data preparation workflow:

1. **KILT NQ Loading** -- Downloads the NaturalQuestions dev split from HuggingFace.
2. **Corpus Construction** -- Builds a reduced Wikipedia corpus by extracting gold pages referenced by queries and sampling distractor pages.
3. **Chunking** -- Splits documents into retrievable units using paragraph-level or fixed-length strategies.
4. **NER Classification** -- Uses spaCy to classify queries by named-entity count (single-entity vs. multi-entity).
5. **Ground-Truth Extraction** -- Maps each query to its gold chunk IDs using KILT provenance annotations.

Local caching (`data/corpus_cache.parquet`) avoids re-downloading the ~35 GB Wikipedia stream on subsequent runs.

### `src/biencoder_retrieval.py`

Implements the bi-encoder retrieval pipeline:

- Encodes all document chunks and queries using `all-MiniLM-L6-v2` (384-dim embeddings).
- Builds a FAISS `IndexFlatIP` for exact inner-product search.
- Retrieves top-k passages for each query and records per-query latency.

### `src/colbert_retrieval.py`

Implements ColBERTv2 late-interaction retrieval:

- Uses the RAGatouille library to build a PLAID index over document chunks.
- Retrieves top-k passages using MaxSim token-level scoring.
- Includes Windows compatibility patches for C++ extension loading.

### `src/evaluation.py`

Computes Recall@k metrics from the JSON benchmark log. Groups results by entity complexity (single-entity vs. multi-entity) and generates summary statistics DataFrames.

### `src/visualize.py`

Generates all output artifacts from the JSON log:

- Recall@k bar charts (grouped by entity type and retrieval model)
- Per-query recall histograms at each k value
- Latency comparison charts
- Index size comparison charts
- CSV summary table

### `src/profiler.py`

Lightweight profiling utility that tracks:

- Stage-level wall-clock timing
- GPU VRAM peak usage (via PyTorch CUDA)
- Process RSS memory snapshots (via psutil)
- Disk sizes of generated indexes
- Per-query result logging

All measurements are accumulated into a single JSON log file.

### `src/ablation.py`

Constructs balanced test sets for ablation experiments by sampling queries from the cached main corpus. Supports ablation across three dimensions: corpus scale, chunk granularity, and top-k depth.

### `src/ablation_visualize.py`

Generates ablation-specific visualizations: multi-panel charts comparing recall across different corpus sizes, chunking strategies, and k values.

## Data Flow

1. **Input**: KILT NQ dev split (HuggingFace) + KILT Wikipedia (streamed, cached locally)
2. **Intermediate**: Chunked documents, FAISS/ColBERT indexes, per-query retrieval results
3. **Output**: `results/benchmark_log.json` (single source of truth), charts, CSV summary

The JSON log serves as the single source of truth -- evaluation and visualization stages read exclusively from this log, ensuring reproducibility and separation of concerns.
