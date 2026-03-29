import os
import time

import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

from src.config import IMAGES_DIR, VERBOSE
from src.db.qdrant_db import get_client, init_collection, is_already_embedded, upsert_page

MODEL_NAME = "vidore/colpali-v1.3-hf"


def init_model() -> tuple[ColPaliForRetrieval, ColPaliProcessor]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ColPaliForRetrieval.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    ).to(device).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def embed_page(model, processor, image_path: str) -> list[list[float]]:
    """Full multi-vector for a page image — (num_patches, 128)."""
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        embeddings = model(**inputs).embeddings  # (1, num_patches, dim)

    return embeddings[0].cpu().tolist()


def embed_query(model, processor, query: str) -> list[list[float]]:
    """Full multi-vector for a text query — (num_tokens, 128)."""
    device = next(model.parameters()).device
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        embeddings = model(**inputs).embeddings  # (1, num_tokens, dim)

    return embeddings[0].cpu().tolist()


def get_image_dirs(images_dir: str) -> list[str]:
    """Return sorted list of PMCID subdirectory names."""
    return sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )


def get_page_images(images_dir: str, pmcid: str) -> list[tuple[int, str]]:
    """Return sorted list of (page_num, image_path) for a document."""
    doc_dir = os.path.join(images_dir, pmcid)
    pages = []
    for f in sorted(os.listdir(doc_dir)):
        if f.endswith(".png"):
            page_num = int(f[1:5])
            pages.append((page_num, os.path.join(doc_dir, f)))
    return pages


def embed_document(model, processor, client, pmcid, pages) -> int:
    count = 0
    for page_num, image_path in pages:
        try:
            vectors = embed_page(model, processor, image_path)
            upsert_page(client, pmcid, page_num, vectors)
            count += 1
        except Exception as e:
            print(f"  [warn] Failed {pmcid} p{page_num}: {e}")
    return count


def main():
    model, processor = init_model()
    client = get_client()
    init_collection(client)

    pmcids = get_image_dirs(IMAGES_DIR)
    if not pmcids:
        print("No images found. Run pdf_to_images first.")
        return

    embedded, skipped, failed = 0, 0, 0
    total_pages, total_time = 0, 0.0

    for i, pmcid in enumerate(pmcids, 1):
        pages = get_page_images(IMAGES_DIR, pmcid)

        if is_already_embedded(client, pmcid, len(pages)):
            skipped += 1
            if VERBOSE:
                print(f"  [{i}/{len(pmcids)}] {pmcid} -- skipped")
            continue

        start = time.time()
        n = embed_document(model, processor, client, pmcid, pages)
        elapsed = time.time() - start

        if n > 0:
            embedded += 1
            total_pages += n
            total_time += elapsed
            if VERBOSE:
                print(f"  [{i}/{len(pmcids)}] {pmcid} -- {n} pages in {elapsed:.1f}s "
                      f"(avg {total_time/total_pages:.1f}s/page)")
        else:
            failed += 1
            if VERBOSE:
                print(f"  [{i}/{len(pmcids)}] {pmcid} -- FAILED")

    print(f"\nEmbedded: {embedded}  |  Skipped: {skipped}  |  Failed: {failed}")
    if total_pages:
        print(f"Pages: {total_pages}  |  Time: {total_time:.0f}s  |  Avg: {total_time/total_pages:.1f}s/page")


if __name__ == "__main__":
    main()
