import json
import os
import pickle
import re

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH, BM25_INDEX_DIR, VERBOSE

BM25_INDEX_PATH = os.path.join(BM25_INDEX_DIR, "bm25_index.pkl")

# Rebuild index if rebuild chunks since using implicit mapping of position 0 in scores = line 0 in chunks.jsonl

def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. Used at both build and query time."""
    return re.findall(r'[a-z0-9]+', text.lower())


def load_chunks(chunks_path: str) -> list[dict]:
    """Load all chunk records from a JSONL file."""
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


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
    # Step 1: load chunks
    if not os.path.isfile(CHUNKS_PATH):
        print(f"Chunks file not found: {CHUNKS_PATH}. Run chunker first.")
        return

    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        print("No chunks found. Run chunker first.")
        return

    # Step 2: build index
    if VERBOSE:
        print(f"Building BM25 index from {len(chunks)} chunks...")

    bm25 = build_index(chunks)
    save_index(bm25, BM25_INDEX_PATH)

    unique_docs = len(set(c["pmcid"] for c in chunks))
    print(f"\nChunks indexed: {len(chunks)}  |  Documents: {unique_docs}")
    print(f"Index saved to: {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
