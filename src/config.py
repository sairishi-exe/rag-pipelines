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
CHUNK_SIZE = 400       # words per chunk
CHUNK_OVERLAP = 100    # overlap in words
BM25_INDEX_DIR = "data/indexes/bm25"
DB_PATH = "data/indexes/pipeline.db"

# Qdrant (ColPali multi-vector storage)
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "colpali_pages"
COLPALI_DIM = 128

# Retrieval
CONTEXT_TOP_K = 5
TOP_P_PAGES = 20  # ColPali page candidates for Pipeline B

# LLM
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Target corpus size to download
TARGET_ARTICLES = 120
VERBOSE = True
