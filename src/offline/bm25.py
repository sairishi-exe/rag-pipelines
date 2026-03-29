import os
import pickle
import re
import sqlite3

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_DIR, DB_PATH, VERBOSE

BM25_INDEX_PATH = os.path.join(BM25_INDEX_DIR, "bm25_index.pkl")

# Rebuild index if rebuild chunks since BM25 position = global_index in SQLite


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. Used at both build and query time."""
    return re.findall(r'[a-z0-9]+', text.lower())


def load_chunks_from_db(db_path: str) -> list[dict]:
    """
    Load text and pmcid from SQLite, ordered by global_index.
    This order must match BM25 positional indexing.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT pmcid, text FROM chunks ORDER BY global_index"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]



def build_index(chunks: list[dict]) -> BM25Okapi:
    """Tokenize all chunks and build a BM25Okapi index."""
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def save_index(bm25: BM25Okapi, index_path: str) -> None:
    """Persist BM25 index to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(bm25, f)


def load_index(index_path: str = BM25_INDEX_PATH) -> BM25Okapi:
    """Load BM25 index from disk. Used by the online retriever."""
    with open(index_path, "rb") as f:
        return pickle.load(f)


def main():
    # Step 1: load chunks from SQLite
    if not os.path.isfile(DB_PATH):
        print(f"Database not found: {DB_PATH}. Run chunker first.")
        return

    chunks = load_chunks_from_db(DB_PATH)
    if not chunks:
        print("No chunks found in database. Run chunker first.")
        return

    # Step 2: build index
    if VERBOSE:
        print(f"Building BM25 index from {len(chunks)} chunks...")

    bm25 = build_index(chunks)
    save_index(bm25, BM25_INDEX_PATH)

    # check stats
    unique_docs = len(set(c["pmcid"] for c in chunks))
    print(f"\nChunks indexed: {len(chunks)}  |  Documents: {unique_docs}")
    print(f"Index saved to: {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
