# Experiment Guide

This guide explains how to configure and run experiments with the benchmark system.

## Main Benchmark

### Running via Notebook

```bash
jupyter notebook notebooks/run_benchmark.ipynb
```

Run all cells sequentially. The notebook covers the full pipeline: data loading, bi-encoder retrieval, ColBERTv2 retrieval, evaluation, and visualization.

### Running via Script

```bash
python run_benchmark.py
```

This executes the same pipeline as the notebook in a single command.

### Configuration

Edit `configs/experiment_config.yaml` to customize the main benchmark:

| Parameter | Default | Description |
|---|---|---|
| `corpus.target_size` | 500 | Number of Wikipedia pages in the reduced corpus |
| `corpus.seed` | 42 | Random seed for corpus sampling |
| `chunking.strategy` | paragraph | Chunking method (paragraph-level) |
| `queries.sample_size_per_group` | 150 | Queries per entity group |
| `retrieval.k_values` | [1, 5, 10, 20] | Top-k values for retrieval evaluation |
| `colbert.bsize` | 32 | ColBERT indexing batch size (lower = less VRAM) |
| `spacy.entity_threshold` | 2 | NER count threshold for multi-entity classification |

**VRAM Tip**: If ColBERTv2 indexing fails with OOM, reduce `corpus.target_size` to 200 or lower `colbert.bsize` to 16.

## Ablation Experiments

### Running

```bash
python run_ablation.py
```

Ablation experiments systematically vary one factor at a time while holding others at defaults.

### Configuration

Edit `configs/ablation_config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `test_set.num_queries` | 30 | Total queries (balanced across entity groups) |
| `dimensions.corpus_scale.values` | [50, 100, 200] | Corpus sizes to test |
| `dimensions.chunk_granularity.values` | [paragraph, fixed-256, fixed-512] | Chunking strategies to compare |
| `dimensions.top_k.values` | [1, 3, 5, 10, 20] | k values to evaluate |

### Ablation Dimensions

1. **Corpus Scale** -- Varies the number of Wikipedia pages (50, 100, 200) while using paragraph chunking. Tests how retrieval quality degrades as the distractor pool grows.

2. **Chunk Granularity** -- Compares paragraph-level splitting vs. fixed-length splitting (256 and 512 tokens) on a 50-page corpus. Isolates the impact of chunking strategy on recall.

3. **Top-k Depth** -- Evaluates recall at k = 1, 3, 5, 10, 20 to characterize the precision-recall tradeoff for each retrieval model.

## Output Structure

### Main Benchmark

```
results/
├── benchmark_log.json       # Complete results (timing, metrics, per-query data)
├── summary_statistics.csv   # Recall@k summary table
├── charts/
│   ├── recall_bar_chart.png
│   ├── recall_histogram_k1.png
│   ├── recall_histogram_k5.png
│   ├── recall_histogram_k10.png
│   ├── recall_histogram_k20.png
│   ├── latency_comparison.png
│   └── index_size_comparison.png
├── faiss_index/             # Saved FAISS index
└── colbert_index/           # Saved ColBERT PLAID index
```

### Ablation Experiments

```
results/ablation/
├── <condition_name>_log.json   # Per-condition benchmark log
├── charts/
│   ├── ablation_chunk_granularity.png
│   ├── ablation_corpus_scale.png
│   └── ablation_topk_depth.png
└── ablation_summary.csv        # Combined summary across conditions
```

## Interpreting Results

### JSON Log Structure

The `benchmark_log.json` file is the single source of truth. Key sections:

- **`metadata`**: Run configuration, hardware info, corpus/query counts
- **`stages`**: Per-stage timing, VRAM peak, RSS memory
- **`queries`**: Per-query retrieval results, recall@k, latency
- **`disk_sizes`**: Index file sizes on disk

### Recall@k Metrics

- **Recall@1**: Fraction of queries where the top-ranked passage is correct. Measures precision at the very top.
- **Recall@5/10/20**: Fraction of queries where at least one correct passage appears in the top k results. Higher k gives a more lenient measure of retrieval quality.

### Entity Groups

- **Single-entity**: Queries containing 0-1 named entities (simpler, typically factoid lookups)
- **Multi-entity**: Queries containing 2+ named entities (more complex, requiring matching multiple constraints)
