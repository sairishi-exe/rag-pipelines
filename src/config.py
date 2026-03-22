RAW_DATA_PATH = "data/raw/training13b.json"

# This file is used to filter the dataset to only include documents that have a directly downloadable PDF.
# Found on: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_non_comm_use_pdf.csv
OA_PDF_CSV = "data/raw/oa_non_comm_use_pdf.csv"

# This directory is used to cache intermediate dataframes and mappings.
CACHE_DIR = "data/cache"

# This directory is used to store the downloaded PDFs.
PDF_DIR = "data/raw/pdfs"

# This directory is used to store the processed data.
IMAGES_DIR = "data/processed/images"
OCR_DIR = "data/processed/ocr"
CHUNKS_PATH = "data/processed/chunks.jsonl"
CHUNK_SIZE = 400       # words per chunk
CHUNK_OVERLAP = 100    # overlap in words
# INDEXES_DIR = "data/indexes"

# Target corpus size to download
TARGET_ARTICLES = 120
VERBOSE = True
