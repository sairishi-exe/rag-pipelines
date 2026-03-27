import numpy as np

from src.offline.bm25 import tokenize
from src.db.init_db import fetch_chunks_by_positions


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
