# Hybrid Visual-Lexical RAG Evaluation (BioASQ + PMC)

We evaluate a cost-constrained RAG scenario: a team with a text-only LLM, a standard layout-aware OCR tool, and a fixed-window chunking strategy wants to know whether adding ColPali as a visual page pre-filter is a smart investment, or whether the text pipeline alone is good enough.

**Primary question:** does ColPali page pre-filtering improve Recall@K (surfacing the right pages more reliably) compared to running BM25 over the full corpus?

**Secondary question:** if Recall@K improves, does that translate to better LLM answer quality (ROUGE-L, BERTScore), or is the real bottleneck the document parsing pipeline rather than the retrieval step?

**If ColPali helps:** it demonstrates that a one-time offline embedding pass delivers meaningful ROI (better retrieval precision and potentially better answers) at the tradeoff of added query latency and amortized preprocessing cost. For a team building an internal knowledge tool over a stable document corpus, this tradeoff is likely worth it.

**If ColPali improves Recall@K but not answer quality:** it still has value as a citation layer, surfacing the exact PDF page alongside the answer so users can verify or dig deeper, while suggesting the true bottleneck is in how well the text is extracted and chunked, not in how pages are ranked.

**If ColPali adds nothing:** the text pipeline is sufficient at this scale and corpus type, and the overhead of visual page embeddings is not justified without upgrading to a vision-capable LLM for end-to-end gains.

## What We Compare

### Pipeline A (Lexical Baseline)
1) OCR over all pages (offline)
2) Chunk OCR text (fixed window; 400 words with overlap)
3) Global BM25 retrieval over all chunks → fill fixed context budget
4) Send top-K chunks to text-only LLM → answer

### Pipeline B (Visual-Guided Hybrid)
1) Compute ColPali page embeddings (offline)
2) Same OCR + chunking as Pipeline A
3) Query-time: ColPali retrieves top-P pages, filter chunks to those pages only
4) BM25 retrieval on filtered chunks → fill same fixed context budget → LLM → answer

## Metrics
- Answer quality: Exact Match (prediction vs exact_answer), ROUGE-L (prediction vs ideal_answer), BERTScore (prediction vs ideal_answer)
- Retrieval quality: Hit Rate@1 / Hit Rate@3, Recall@1 / Recall@3
- Efficiency: average latency per query
- ColPali analysis: Page Hit Rate, Page Recall (Pipeline B only)

## Folder Structure

```
project/
  README.md
  requirements.txt
  docker-compose.yml      # Qdrant vector DB for ColPali embeddings
  src/
    prepare_data/
      build_dataset.py    # load BioASQ, resolve PMIDs → PMCIDs, cache
      filter_dataset.py   # compute OA ratio, scope to ~120 documents
      download_corpus.py  # download PMC PDFs
    offline/
      pdf_to_images.py    # rasterize PDFs to per-page images
      ocr.py              # layout-aware OCR (PaddleOCR)
      chunker.py          # split OCR text into fixed-token overlapping chunks → SQLite
      colpali.py          # ColPali page embeddings → Qdrant
      bm25.py             # BM25 index builder (reads from SQLite)
    online/
      retriever.py        # query-time retrieval (Pipeline A and B)
      llm.py              # LLM answerer (Ollama / Llama 3.1 8B)
    db/
      init_db.py          # SQLite schema + chunk queries
      qdrant_db.py        # Qdrant collection + ColPali vector queries
    eval/
      eval.py             # run evaluation (both pipelines, all metrics)
      metrics.py          # Exact Match, ROUGE-L, BERTScore, Hit Rate, Recall
    utils.py              # shared helpers (cache_data, print_example_row)
  data/
    raw/
      training13b.json    # BioASQ training data (downloaded externally)
      pdfs/               # downloaded PMC PDFs
    processed/
      images/             # per-page images: {pmcid}/p0001.png
      ocr/                # per-page OCR text
    cache/                # intermediate DataFrames and mappings
    eval/                 # evaluation results (CSV/JSON)
    indexes/
      bm25/               # BM25 pickle index
      qdrant/             # Qdrant vector DB storage (Docker volume)
      pipeline.db         # SQLite chunk store
```


## Setup

### 1) Create environment and install dependencies

> **Python 3.9–3.12 required** (PaddlePaddle does not support 3.13+)

Mac/Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```bash
python3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Download BioASQ training data
Register and download `BioASQ-training13b.zip` from [bioasq.org](http://bioasq.org), then place the training13b.json file at:
```
data/raw/training13b.json
```

### 3) Download PMC OA PDF file list
Download the non-commercial PDF file list from NCBI's FTP server:
```
https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_non_comm_use_pdf.csv
```
Place it at:
```
data/raw/oa_non_comm_use_pdf.csv
```
This file is used by `filter_dataset.py` to confirm which PMC articles have a directly downloadable PDF.

## Run Instructions

### A) Build dataset (resolve PMIDs → PMCIDs)
```bash
python -m src.prepare_data.build_dataset
```
Outputs: `data/cache/dataframe_cache.json`, `data/cache/document_id_map.json`

### B) Filter subset (~100 - 120 documents, high OA ratio)
```bash
python -m src.prepare_data.filter_dataset
```
Output: `data/cache/dataframe_subset_cache.json`, `data/cache/document_id_subset.json`

### C) Download PMC PDFs
```bash
python -m src.prepare_data.download_corpus
```
Output: `data/raw/pdfs/{pmcid}.pdf`

### D) Offline preprocessing (PDF → images → OCR → chunks → indexes)
```bash
python -m src.offline.pdf_to_images    # PDFs → per-page PNGs
python -m src.offline.ocr              # PNGs → per-page OCR text
python -m src.offline.chunker          # OCR text → SQLite chunks + JSONL backup
python -m src.offline.bm25             # SQLite chunks → BM25 index
```

### E) ColPali page embeddings (Pipeline B only)
Requires Docker for Qdrant:
```bash
docker compose up -d                   # start Qdrant vector DB
python -m src.offline.colpali          # page images → ColPali embeddings → Qdrant
```

### F) Run evaluation
Requires Ollama (with llama3.1:8b) and Qdrant running:
```bash
python -m src.eval.eval
```
This runs both Pipeline A and Pipeline B on all QA pairs, then computes all metrics.

The number of ColPali candidate pages for Pipeline B can be configured via `TOP_P_PAGES` in `src/config.py`. We tested P=5 and P=10 in our evaluation. Change this value and rerun to test other settings.

Outputs:
- `data/eval/eval_results_p{TOP_P_PAGES}.csv` - aggregate comparison for reports
- `data/eval/eval_results_p{TOP_P_PAGES}.json` - per-question results for analysis


## Authors
- Sai Subramanian
- Khush Patel
