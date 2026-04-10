# ColBERTv2 search using RAGatouille and PLAID indexing

import gc
import logging
import time
from pathlib import Path

import torch
import numpy as np
import colbert.modeling.colbert as _cm
import colbert.indexing.codecs.residual as _cr
import colbert.search.strided_tensor as _st
import colbert.search.index_storage as _is

# Fix ColBERT for Windows because it usually needs a C++ compiler
@classmethod
def _patched_try_load(cls, use_gpu):
    if not hasattr(cls, "loaded_extensions"):
        cls.loaded_extensions = True

_cm.ColBERT.try_load_torch_extensions = _patched_try_load
_cr.ResidualCodec.try_load_torch_extensions = _patched_try_load
_st.StridedTensor.try_load_torch_extensions = _patched_try_load
_is.IndexScorer.try_load_torch_extensions = _patched_try_load

# Fallback code for GPU tasks when C++ extensions are missing
@staticmethod
def _fallback_packbits(tensor):
    packed = np.packbits(np.asarray(tensor.contiguous().cpu()))
    return torch.as_tensor(packed, dtype=torch.uint8)

@staticmethod
def _fallback_decompress_residuals(
    residuals, bucket_weights, reversed_bit_map,
    decompression_lookup_table, codes, centroids, dim, nbits,
):
    centroids_ = centroids[codes.long()]
    residuals_ = reversed_bit_map[residuals.long()]
    residuals_ = decompression_lookup_table[residuals_.long()]
    residuals_ = residuals_.reshape(residuals_.shape[0], -1)
    residuals_ = bucket_weights[residuals_.long()]
    centroids_ = centroids_ + residuals_
    return centroids_

if not hasattr(_cr.ResidualCodec, "packbits"):
    _cr.ResidualCodec.packbits = _fallback_packbits
if not hasattr(_cr.ResidualCodec, "decompress_residuals"):
    _cr.ResidualCodec.decompress_residuals = _fallback_decompress_residuals

from ragatouille import RAGPretrainedModel
from src.profiler import Profiler

logger = logging.getLogger(__name__)
COLBERT_INDEX_NAME = "colbert_benchmark"

# Build the ColBERT index from text chunks
def build_colbert_index(
    chunks: list[dict],
    index_path: str = "results/colbert_index",
    model_name: str = "colbert-ir/colbertv2.0",
    bsize: int = 32,
) -> RAGPretrainedModel:
    index_root = Path(index_path)
    index_root.mkdir(parents=True, exist_ok=True)
    rag = RAGPretrainedModel.from_pretrained(model_name, index_root=str(index_root))

    texts = [c["text"] for c in chunks]
    doc_ids = [c["chunk_id"] for c in chunks]

    # Create the index on disk
    rag.index(
        collection=texts,
        document_ids=doc_ids,
        index_name=COLBERT_INDEX_NAME,
        split_documents=False,
        max_document_length=180,
        bsize=bsize,
    )
    return rag

# Find where the index is saved on the computer
def resolve_colbert_index_path(index_path: str) -> Path:
    configured = Path(index_path)
    candidates = [
        configured / COLBERT_INDEX_NAME,
        configured,
        Path(".ragatouille") / "colbert" / "indexes" / COLBERT_INDEX_NAME,
        Path(".ragatouille") / "indexes" / COLBERT_INDEX_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return configured

# Search for the best matches using the model
def retrieve_top_k_colbert(
    rag: RAGPretrainedModel,
    query: str,
    k: int = 20,
) -> tuple[list[str], float]:
    start = time.perf_counter()
    results = rag.search(query=query, k=k)
    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_ids = [r["document_id"] for r in results]
    return retrieved_ids, latency_ms

# Calculate how many correct items were found
def compute_recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]

    matched = 0
    for ground_truth_id in ground_truth_ids:
        if any(
            retrieved_id == ground_truth_id
            or retrieved_id.startswith(f"{ground_truth_id}_c")
            for retrieved_id in top_k
        ):
            matched += 1
    return matched / len(ground_truth_ids)

# Run the full ColBERT test and handle memory errors
def run_colbert_retrieval(
    chunks: list[dict],
    sampled_queries: list[dict],
    config: dict,
    profiler: Profiler,
) -> None:
    model_name = config["models"]["colbert"]
    k_values = config["retrieval"]["k_values"]
    max_k = max(k_values)
    index_path = config["paths"]["colbert_index_dir"]

    # Start indexing
    profiler.start_stage("colbert_indexing")
    try:
        bsize = config.get("colbert", {}).get("bsize", 32)
        rag = build_colbert_index(chunks, index_path=index_path, model_name=model_name, bsize=bsize)
    except (RuntimeError, MemoryError) as e:
        profiler.end_stage("colbert_indexing")
        msg = f"ColBERT failed. GPU memory might be full: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    profiler.end_stage("colbert_indexing")

    # Save index size info
    resolved_index_path = resolve_colbert_index_path(index_path)
    profiler.record_disk_size("colbert_index", str(resolved_index_path))
    profiler.data["metadata"]["colbert_index_path"] = str(resolved_index_path)

    # Start searching
    profiler.start_stage("colbert_retrieval")
    for q in sampled_queries:
        gt_ids = set(q["ground_truth_chunk_ids"])
        retrieved_ids, latency_ms = retrieve_top_k_colbert(rag, q["query"], k=max_k)

        # Get recall for different k values
        recall_at_k = {str(k): compute_recall_at_k(retrieved_ids, gt_ids, k) for k in k_values}

        q["colbert_retrieved_ids"] = retrieved_ids
        q["colbert_recall_at_k"] = recall_at_k
        q["colbert_latency_ms"] = round(latency_ms, 4)
    profiler.end_stage("colbert_retrieval")

    # Clean up memory
    del rag
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()