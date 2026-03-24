import numpy as np

from src.offline.bm25 import tokenize


def retrieve_pipeline_a(query: str, chunks: list[dict], bm25, top_k: int = 5) -> list[dict]:
    """
    Score all chunks against the query using BM25 and return the top-K.
    Caller is responsible for loading bm25 and chunks once and passing them in.
    """
    # tokenize exact way tokenized corpus
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    # sort in reverse and return top k indices that map to chunks
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        chunk = chunks[i].copy()
        chunk["score"] = float(scores[i])
        results.append(chunk)

    return results
