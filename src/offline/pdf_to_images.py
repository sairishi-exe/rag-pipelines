import os
import shutil
from pdf2image import convert_from_path, pdfinfo_from_path

from src.config import PDF_DIR, IMAGES_DIR, VERBOSE
from src.utils import cache_data

DPI = 300


def get_pdf_paths(pdf_dir: str) -> list[str]:
    """Return sorted list of absolute paths to all .pdf files in pdf_dir."""
    return sorted(
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    )


def is_already_converted(pdf_path: str, pmcid: str, images_dir: str) -> bool:
    """
    Check if all pages of the PDF have been converted.
    Compares the number of existing PNGs against the PDF's actual page count
    to detect partial conversions from interrupted runs.
    """
    out_dir = os.path.join(images_dir, pmcid)
    if not os.path.isdir(out_dir):
        return False
    num_pngs = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if num_pngs == 0:
        return False
    # get the actual page count from the PDF without rendering
    try:
        info = pdfinfo_from_path(pdf_path)
        expected_pages = info["Pages"]
    except Exception:
        # if we can't read page count, assume complete if any PNGs exist
        return True
    return num_pngs == expected_pages


def convert_pdf_to_images(pdf_path: str, out_dir: str, dpi: int = DPI) -> int:
    """
    Convert a single PDF to per-page PNGs.
    Clears any existing partial output before converting to avoid stale files.
    
    Returns the number of pages converted, or 0 on failure.
    """
    # clear any partial output from a previous interrupted run
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"  [error] Cannot convert {pdf_path}: {e}")
        return 0

    count = 0
    for i, image in enumerate(images, 1):
        try:
            image.save(os.path.join(out_dir, f"p{i:04d}.png"), "PNG")
            count += 1
        except Exception as e:
            print(f"  [warn] Failed to save page {i} of {pdf_path}: {e}")

    return count


def main():
    # Step 1: collect all PDF paths from the corpus directory
    pdf_paths = get_pdf_paths(PDF_DIR)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    converted, skipped, failed = 0, 0, 0
    page_counts = {}  # {pmcid: num_pages} list for downstream use

    # Step 2: convert each PDF to per-page PNGs
    for i, pdf_path in enumerate(pdf_paths, 1):
        pmcid = os.path.splitext(os.path.basename(pdf_path))[0]

        # skip PDFs that have already been fully converted
        if is_already_converted(pdf_path, pmcid, IMAGES_DIR):
            skipped += 1
            existing_dir = os.path.join(IMAGES_DIR, pmcid)
            page_counts[pmcid] = len([f for f in os.listdir(existing_dir) if f.endswith(".png")])
            if VERBOSE:
                print(f"  [{i}/{len(pdf_paths)}] {pmcid} -- skipped (already converted)")
            continue

        # convert and save page images
        num_pages = convert_pdf_to_images(pdf_path, os.path.join(IMAGES_DIR, pmcid))

        if num_pages > 0:
            converted += 1
            page_counts[pmcid] = num_pages
            if VERBOSE:
                print(f"  [{i}/{len(pdf_paths)}] {pmcid} -- {num_pages} pages")
        else:
            failed += 1
            if VERBOSE:
                print(f"  [{i}/{len(pdf_paths)}] {pmcid} -- FAILED")

    # Step 3: save page count manifest so OCR/chunker knows what to expect
    if page_counts:
        cache_data(page_counts=page_counts)

    print(f"\nConverted: {converted}  |  Skipped: {skipped}  |  Failed: {failed}")
    print(f"Total pages: {sum(page_counts.values())}")


if __name__ == "__main__":
    main()
