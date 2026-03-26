import json
import os
import sqlite3

from src.config import OCR_DIR, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP, VERBOSE, DB_PATH
from src.db.init_db import init_db

# minimum new words the last chunk must contribute; otherwise merge into previous
MIN_TAIL_WORDS = 50


def get_ocr_dirs(ocr_dir: str) -> list[str]:
    """Return sorted list of PMCID subdirectory names in ocr_dir."""
    return sorted(
        d for d in os.listdir(ocr_dir)
        if os.path.isdir(os.path.join(ocr_dir, d))
    )


def is_already_chunked(chunks_path: str, ocr_pmcids: list[str]) -> bool:
    """
    Check if chunks.jsonl exists and contains chunks for every OCR'd PMCID.
    Returns False if the file is missing or any PMCID is not covered.
    """
    if not os.path.isfile(chunks_path):
        return False
    chunked_pmcids = set()
    with open(chunks_path) as f:
        for line in f:
            rec = json.loads(line)
            chunked_pmcids.add(rec["pmcid"])
    return all(pmcid in chunked_pmcids for pmcid in ocr_pmcids)


def read_document_pages(ocr_dir: str, pmcid: str) -> list[tuple[int, list[str]]]:
    """
    Read all per-page OCR text files for one document.
    Returns a list of (page_number, words) tuples sorted by page number.
    Page number is 1-indexed, extracted from filename (p0001.txt -> 1).
    """
    doc_dir = os.path.join(ocr_dir, pmcid)
    txt_files = sorted(f for f in os.listdir(doc_dir) if f.endswith(".txt"))

    pages = []
    for txt_file in txt_files:
        page_num = int(txt_file[1:5])  # "p0001.txt" -> 1
        with open(os.path.join(doc_dir, txt_file)) as f:
            words = f.read().split()
        pages.append((page_num, words))

    return pages


def chunk_document(
    pmcid: str,
    pages: list[tuple[int, list[str]]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Chunk a single document's pages into fixed-window overlapping chunks.

    1. Flatten all words into one list, tracking which page each word came from.
    2. Slide a window of chunk_size words with step = chunk_size - chunk_overlap.
    3. Merge short tail if the last window adds fewer than MIN_TAIL_WORDS new words.
    4. Return list of chunk records with metadata.
    """
    # step 1: flatten words and build page tracking array
    all_words = []
    word_page = []
    for page_num, words in pages:
        for w in words:
            all_words.append(w)
            word_page.append(page_num)

    total = len(all_words)
    if total == 0:
        return []

    # step 2: generate chunk windows
    step = chunk_size - chunk_overlap
    windows = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        windows.append((start, end))
        if end == total:
            break
        start += step

    # step 3: short tail merge
    if len(windows) >= 2:
        prev_start, prev_end = windows[-2]
        last_start, last_end = windows[-1]
        new_words_in_last = last_end - prev_end
        if new_words_in_last < MIN_TAIL_WORDS:
            windows[-2] = (prev_start, last_end)
            windows.pop()

    # step 4: build chunk records
    chunks = []
    for idx, (start, end) in enumerate(windows):
        chunks.append({
            "chunk_id": f"{pmcid}_{idx}",
            "pmcid": pmcid,
            "page_start": word_page[start],
            "page_end": word_page[end - 1],
            "chunk_index": idx,
            "text": " ".join(all_words[start:end]),
        })

    return chunks


def insert_chunks_to_db(chunks: list[dict]) -> None:
    """Insert all chunks into SQLite. Wipes existing data first for idempotency."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chunks")
    conn.executemany(
        "INSERT INTO chunks (chunk_id, pmcid, page_start, page_end, chunk_index, text) "
        "VALUES (:chunk_id, :pmcid, :page_start, :page_end, :chunk_index, :text)",
        chunks
    )
    conn.commit()
    conn.close()


def main():
    # Step 1: discover all OCR'd documents
    pmcids = get_ocr_dirs(OCR_DIR)
    if not pmcids:
        print("No OCR output found. Run OCR first.")
        return

    # Step 2: idempotency check
    if is_already_chunked(CHUNKS_PATH, pmcids):
        print(f"Chunking already complete ({CHUNKS_PATH} covers all {len(pmcids)} documents). Skipping.")
        return

    all_chunks = []
    processed, failed = 0, 0

    # Step 3: chunk each document
    for i, pmcid in enumerate(pmcids, 1):
        pages = read_document_pages(OCR_DIR, pmcid)

        if not pages:
            failed += 1
            if VERBOSE:
                print(f"  [{i}/{len(pmcids)}] {pmcid} -- FAILED (no pages)")
            continue

        chunks = chunk_document(pmcid, pages, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        processed += 1

        if VERBOSE:
            total_words = sum(len(words) for _, words in pages)
            print(f"  [{i}/{len(pmcids)}] {pmcid} -- {len(pages)} pages, "
                  f"{total_words} words, {len(chunks)} chunks")

    # Step 4: write all chunks to JSONL
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    # Step 5: insert into SQLite
    insert_chunks_to_db(all_chunks)

    print(f"\nChunked: {processed}  |  Failed: {failed}")
    print(f"Total chunks: {len(all_chunks)}  |  Output: {CHUNKS_PATH}")
    print(f"SQLite: {DB_PATH}")


if __name__ == "__main__":
    main()
