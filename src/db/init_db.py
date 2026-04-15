import os
import sqlite3

from src.config import DB_PATH


def init_db():
    """Create pipeline.db and the chunks table if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id     TEXT PRIMARY KEY,
            pmcid        TEXT NOT NULL,
            page_start   INTEGER NOT NULL,
            page_end     INTEGER NOT NULL,
            chunk_index  INTEGER NOT NULL,
            global_index INTEGER NOT NULL,
            text         TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def fetch_chunks_by_positions(positions: list[int]) -> list[dict]:
    """Fetch chunks from SQLite by their global_index values."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    qmarks = ",".join("?" for _ in positions)
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE global_index IN ({qmarks})", positions
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_chunk_indices_by_pages(pages: set[tuple[str, int]]) -> list[int]:
    """Map (pmcid, page_num) pairs to chunk global_index values.

    Used by Pipeline B: ColPali returns top pages, this finds which
    chunks overlap those pages so BM25 can score just the candidates.
    """
    conn = sqlite3.connect(DB_PATH)
    indices = []
    for pmcid, page_num in pages:
        rows = conn.execute(
            "SELECT global_index FROM chunks "
            "WHERE pmcid = ? AND page_start <= ? AND page_end >= ?",
            (pmcid, page_num, page_num),
        ).fetchall()
        indices.extend(r[0] for r in rows)
    conn.close()
    return list(set(indices))


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {DB_PATH}")
