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
    """Fetch chunks from SQLite by their global_index values.

    Uses parameterized IN (...) query — SQLite requires one '?' placeholder
    per value to prevent SQL injection, so we build the placeholder string
    dynamically based on the number of positions requested.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access on rows
    qmarks = ",".join("?" for _ in positions) # SQL injection prevention (prod-like purposes)
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE global_index IN ({qmarks})", positions
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {DB_PATH}")
