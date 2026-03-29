import numpy as np

from src.offline.bm25 import tokenize
from src.db.init_db import fetch_chunks_by_positions, fetch_chunk_indices_by_pages
from src.db.qdrant_db import query_pages


def retrieve_pipeline_a(query: str, bm25, top_k: int = 5) -> list[dict]:
    """
    Score all chunks against the query using BM25 and return the top-K.
    Uses global_index to fetch only the needed chunks from SQLite.
    """
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    # top-K positions (these correspond to global_index in SQLite)
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = {int(i): float(scores[i]) for i in top_indices}

    # fetch only the top-K chunks from SQLite
    chunks = fetch_chunks_by_positions(list(top_scores.keys()))

    # attach scores and sort by score descending
    for chunk in chunks:
        chunk["score"] = top_scores[chunk["global_index"]]
    chunks.sort(key=lambda c: c["score"], reverse=True)

    return chunks


def retrieve_pipeline_b(
    query: str,
    bm25,
    query_embeddings: list[list[float]],
    qdrant_client,
    top_p: int = 20,
    top_k: int = 5,
) -> list[dict]:
    """ColPali MaxSim -> page filter -> BM25 re-rank on candidate chunks."""

    # 1. Qdrant MaxSim: find top-P pages
    page_hits = query_pages(qdrant_client, query_embeddings, top_p)
    pages = {(h["pmcid"], h["page_num"]) for h in page_hits}

    # 2. Map pages -> chunk global_indices via SQLite
    candidate_idx = fetch_chunk_indices_by_pages(pages)
    if not candidate_idx:
        return []

    # 3. BM25 score only the candidate chunks
    tokens = tokenize(query)
    all_scores = bm25.get_scores(tokens)
    scored = {i: float(all_scores[i]) for i in candidate_idx}

    # 4. Top-K from candidates
    top_indices = sorted(scored, key=scored.get, reverse=True)[:top_k]
    top_scores = {i: scored[i] for i in top_indices}

    chunks = fetch_chunks_by_positions(top_indices)
    for chunk in chunks:
        chunk["score"] = top_scores[chunk["global_index"]]
    chunks.sort(key=lambda c: c["score"], reverse=True)

    return chunks
