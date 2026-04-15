import os
import shutil
import time
import psutil
from paddleocr import PaddleOCR

from src.config import IMAGES_DIR, OCR_DIR, VERBOSE

# stop if memory usage exceeds this limit (in GB), adjust based on your machine
MAX_MEMORY_GB = 14


def init_ocr() -> PaddleOCR:
    """Initialize PaddleOCR v3 engine once (model loading is expensive)."""
    return PaddleOCR(
        lang='en',
        use_doc_orientation_classify=False,  # not needed for digital PDF renders
        use_doc_unwarping=False,             # not needed for digital PDF renders
        use_textline_orientation=False,      # not needed for digital PDF renders
        # mobile model is good enough for clean 300 DPI images, no need for the heavier server version
        text_detection_model_name='PP-OCRv5_mobile_det',  # lighter model, sufficient for clean PDFs
    )


def get_image_dirs(images_dir: str) -> list[str]:
    """Return sorted list of PMCID subdirectory names in images_dir."""
    return sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )


def is_already_ocrd(pmcid: str, ocr_dir: str, images_dir: str) -> bool:
    """
    Check if OCR is complete for this document.
    Compares number of .txt files in OCR dir against number of .png files
    in images dir to detect partial runs.
    """
    out_dir = os.path.join(ocr_dir, pmcid)
    img_dir = os.path.join(images_dir, pmcid)
    if not os.path.isdir(out_dir):
        return False
    num_txts = len([f for f in os.listdir(out_dir) if f.endswith(".txt")])
    num_pngs = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
    if num_txts == 0:
        return False
    return num_txts == num_pngs


def ocr_page(ocr_engine: PaddleOCR, image_path: str) -> str:
    """
    Run OCR on a single page image using PaddleOCR v3 API.
    Returns extracted text with lines sorted in reading order (top-to-bottom,
    left-to-right). Returns empty string if no text is detected or on failure.
    """
    try:
        # v3 API: predict() returns a list of result objects
        results = ocr_engine.predict(image_path)
    except Exception as e:
        print(f"  [warn] OCR failed for {image_path}: {e}")
        return ""

    if not results:
        return ""

    res = results[0]

    # v3 result is a dict-like OCRResult with:
    #   rec_texts: list of recognized text strings
    #   dt_polys: list of numpy arrays, each with 4 corner points [[x,y], ...]
    #   rec_scores: list of confidence scores
    texts = res["rec_texts"]
    polys = res["dt_polys"]

    if not texts:
        return ""

    # sort text lines by reading order using bounding polygon coordinates
    # each polygon is a numpy array of [[x,y], ...] corner points
    # sort by min y (top of region), then min x (left edge) as tiebreaker
    if polys is not None and len(polys) == len(texts):
        paired = list(zip(polys, texts))
        paired.sort(key=lambda pt: (pt[0][:, 1].min(), pt[0][:, 0].min()))
        texts = [t for _, t in paired]

    return "\n".join(texts)


def ocr_document(ocr_engine: PaddleOCR, image_dir: str, out_dir: str) -> int:
    """
    OCR all pages of a single document.
    Clears any partial output before processing.
    Returns the number of pages processed.
    """
    # clear partial output from a previous interrupted run
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # get sorted list of page images
    pages = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    count = 0

    for page_file in pages:
        image_path = os.path.join(image_dir, page_file)
        text = ocr_page(ocr_engine, image_path)

        # save text file with same name as image but .txt extension
        # write even if empty so the skip check counts match
        txt_name = page_file.replace(".png", ".txt")
        try:
            with open(os.path.join(out_dir, txt_name), "w") as f:
                f.write(text)
            count += 1
        except Exception as e:
            print(f"  [warn] Failed to save {txt_name}: {e}")

    return count


def check_memory():
    """Return current process memory usage in GB. Abort if over limit."""
    mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    if mem_gb > MAX_MEMORY_GB:
        print(f"\n[ABORT] Memory usage {mem_gb:.1f} GB exceeds {MAX_MEMORY_GB} GB limit. Stopping.")
        exit(1)
    return mem_gb


def main():
    # Step 1: initialize OCR engine (one-time model load)
    ocr_engine = init_ocr()

    # Step 2: get all document directories
    pmcids = get_image_dirs(IMAGES_DIR)
    os.makedirs(OCR_DIR, exist_ok=True)

    converted, skipped, failed = 0, 0, 0
    total_pages = 0
    total_time = 0.0

    # Step 3: OCR each document
    for i, pmcid in enumerate(pmcids, 1):
        # skip documents that have already been fully OCR'd
        if is_already_ocrd(pmcid, OCR_DIR, IMAGES_DIR):
            skipped += 1
            if VERBOSE:
                print(f"  [{i}/{len(pmcids)}] {pmcid} -- skipped (already OCR'd)")
            continue

        image_dir = os.path.join(IMAGES_DIR, pmcid)
        out_dir = os.path.join(OCR_DIR, pmcid)

        start = time.time()
        num_pages = ocr_document(ocr_engine, image_dir, out_dir)
        elapsed = time.time() - start

        if num_pages > 0:
            converted += 1
            total_pages += num_pages
            total_time += elapsed
            avg_per_page = total_time / total_pages
            mem_gb = check_memory()
            if VERBOSE:
                print(f"  [{i}/{len(pmcids)}] {pmcid} -- {num_pages} pages in {elapsed:.1f}s "
                      f"(avg {avg_per_page:.1f}s/page, mem {mem_gb:.1f}GB)")
        else:
            failed += 1
            if VERBOSE:
                print(f"  [{i}/{len(pmcids)}] {pmcid} -- FAILED")

    print(f"\nOCR'd: {converted}  |  Skipped: {skipped}  |  Failed: {failed}")
    if total_pages > 0:
        print(f"Total pages: {total_pages}  |  Total time: {total_time:.0f}s  |  Avg: {total_time/total_pages:.1f}s/page")


if __name__ == "__main__":
    main()
