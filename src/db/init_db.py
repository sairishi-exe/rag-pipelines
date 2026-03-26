import os
import sqlite3

from src.config import DB_PATH


def init_db():
    """Create pipeline.db and the chunks table if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            pmcid       TEXT NOT NULL,
            page_start  INTEGER NOT NULL,
            page_end    INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {DB_PATH}")
