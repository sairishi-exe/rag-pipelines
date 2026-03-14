import os
import time
import requests
import pandas as pd

from src.config import OA_PDF_CSV, PDF_DIR, VERBOSE
from src.utils import load_from_cache

FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc"
DOWNLOAD_SLEEP = 0.5


def load_pdf_path_map(csv_path: str) -> dict[str, str]:
    """
    Parse oa_non_comm_use_pdf.csv into {pmcid: file_path}.
    file_path is the relative FTP path, e.g. 'oa_pdf/e7/38/....PMC466938.pdf'
    """
    df = pd.read_csv(csv_path, usecols=["Accession ID", "File"])
    return dict(zip(df["Accession ID"].str.strip(), df["File"].str.strip()))


def download_pdf(pmcid: str, file_path: str, out_dir: str) -> str:
    """
    Download a single PDF from the PMC FTP server.
    Returns 'downloaded', 'exists', or 'failed'.
    """
    dest = os.path.join(out_dir, f"{pmcid}.pdf")

    if os.path.exists(dest):
        return "exists"

    url = f"{FTP_BASE}/{file_path}"

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(dest, "wb") as f:
                f.write(response.content)
            return "downloaded"
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] failed to download {pmcid}: {e}")
                return "failed"


if __name__ == "__main__":
    subset = load_from_cache("document_id_subset")[0]
    pmcids = subset["pmcids"]

    pdf_path_map = load_pdf_path_map(OA_PDF_CSV)
    os.makedirs(PDF_DIR, exist_ok=True)

    downloaded, existed, failed, missing = 0, 0, 0, 0

    for pmcid in pmcids:
        file_path = pdf_path_map.get(pmcid)
        if not file_path:
            if VERBOSE:
                print(f"  [skip] {pmcid} not in PDF list")
            missing += 1
            continue

        result = download_pdf(pmcid, file_path, PDF_DIR)

        if result == "downloaded":
            downloaded += 1
        elif result == "exists":
            existed += 1
        else:
            failed += 1

        time.sleep(DOWNLOAD_SLEEP)

    print(f"\nDownloaded: {downloaded}  |  Already existed: {existed}  |  Failed: {failed}  |  Not in PDF list: {missing}")
