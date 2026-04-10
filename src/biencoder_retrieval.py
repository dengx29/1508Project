# Bi-encoder search using sentence-transformers and FAISS

import gc
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.profiler import Profiler


# Turn text chunks into vector numbers
def encode_chunks(
    chunks: list[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> tuple[SentenceTransformer, np.ndarray]:
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    # Normalize embeddings to help with cosine similarity
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return model, np.asarray(embeddings, dtype=np.float32)


# Build a FAISS index for fast searching
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# Find the best matches for a query
def retrieve_top_k(
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    query: str,
    k: int = 20,
) -> tuple[list[str], float]:
    start = time.perf_counter()
    # Turn the query into a vector
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    # Search the index for the top results
    _, indices = index.search(q_emb, k)
    latency_ms = (time.perf_counter() - start) * 1000

    # Get the chunk IDs from the search results
    retrieved_ids = [chunks[int(idx)]["chunk_id"] for idx in indices[0] if idx < len(chunks)]
    return retrieved_ids, latency_ms


# Calculate how many correct answers were found
def compute_recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]

    matched = 0
    for ground_truth_id in ground_truth_ids:
        # Check if the result matches the correct answer
        if any(
            retrieved_id == ground_truth_id
            or retrieved_id.startswith(f"{ground_truth_id}_c")
            for retrieved_id in top_k
        ):
            matched += 1
    return matched / len(ground_truth_ids)


# Run the full search test and track performance
def run_biencoder_retrieval(
    chunks: list[dict],
    sampled_queries: list[dict],
    config: dict,
    profiler: Profiler,
) -> None:
    model_name = config["models"]["biencoder"]
    k_values = config["retrieval"]["k_values"]
    max_k = max(k_values)

    # Convert corpus to vectors
    profiler.start_stage("biencoder_encoding")
    model, embeddings = encode_chunks(chunks, model_name=model_name)
    profiler.end_stage("biencoder_encoding")

    # Build the search index and clear extra memory
    profiler.start_stage("faiss_indexing")
    index = build_faiss_index(embeddings)
    del embeddings
    gc.collect()
    profiler.end_stage("faiss_indexing")

    # Save the index and check file size
    index_path = config["paths"]["faiss_index_dir"]
    import os
    os.makedirs(index_path, exist_ok=True)
    faiss_file = os.path.join(index_path, "index.faiss")
    faiss.write_index(index, faiss_file)
    profiler.record_disk_size("faiss_index", index_path)

    # Search for each question
    profiler.start_stage("biencoder_retrieval")
    for q in sampled_queries:
        gt_ids = set(q["ground_truth_chunk_ids"])
        retrieved_ids, latency_ms = retrieve_top_k(model, index, chunks, q["query"], k=max_k)

        # Calculate scores at different k levels
        recall_at_k = {str(k): compute_recall_at_k(retrieved_ids, gt_ids, k) for k in k_values}

        q["biencoder_retrieved_ids"] = retrieved_ids
        q["biencoder_recall_at_k"] = recall_at_k
        q["biencoder_latency_ms"] = round(latency_ms, 4)
    profiler.end_stage("biencoder_retrieval")

    # Clean up memory for the next stage
    del model, index
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()