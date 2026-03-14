# Hybrid Visual–Lexical RAG Evaluation (BioASQ + PMC)

We evaluate a cost-constrained RAG scenario: a team with a text-only LLM, a standard layout-aware OCR tool, and a fixed-window chunking strategy wants to know whether adding ColPali as a visual page pre-filter is a smart investment — or whether the text pipeline alone is good enough.

**Primary question:** does ColPali page pre-filtering improve Recall@K — surfacing the right pages more reliably — compared to running BM25 over the full corpus?

**Secondary question:** if Recall@K improves, does that translate to better LLM answer quality (ROUGE-L, BERTScore), or is the real bottleneck the document parsing pipeline rather than the retrieval step?

**If ColPali helps:** it demonstrates that a one-time offline embedding pass delivers meaningful ROI — better retrieval precision and potentially better answers — at the tradeoff of added query latency and amortized preprocessing cost. For a team building an internal knowledge tool over a stable document corpus, this tradeoff is likely worth it.

**If ColPali improves Recall@K but not answer quality:** it still has value as a citation layer — surfacing the exact PDF page alongside the answer so users can verify or dig deeper — while suggesting the true bottleneck is in how well the text is extracted and chunked, not in how pages are ranked.

**If ColPali adds nothing:** the text pipeline is sufficient at this scale and corpus type, and the overhead of visual page embeddings is not justified without upgrading to a vision-capable LLM for end-to-end gains.

## What We Compare

### Pipeline A — Lexical Baseline
1) OCR over all pages (offline)
2) Chunk OCR text (fixed window; 400 tokens with overlap)
3) Global BM25 retrieval over all chunks → fill fixed context budget
4) Send top-K chunks to text-only LLM → answer

### Pipeline B — Visual-Guided Hybrid
1) Compute ColPali page embeddings (offline) using 2x2 pooling (1024 → 256 vectors/page)
2) Same OCR + chunking as Pipeline A
3) Query-time: ColPali retrieves top-P pages → filter chunks to those pages only
4) BM25 retrieval on filtered chunks → fill same fixed context budget → LLM → answer

## Metrics
- Answer quality: ROUGE-L, BERTScore (prediction vs ground-truth answer)
- Retrieval quality: Hit Rate@1 / Hit Rate@3, Recall@1 / Recall@3
- Efficiency: latency and reduction in search space (chunks scanned)

## Folder Structure

```
project/
  README.md
  requirements.txt
  src/
    prepare_data/
      build_dataset.py    # load BioASQ, resolve PMIDs → PMCIDs, cache
      filter_dataset.py   # compute OA ratio, scope to ~500 pages
      download_corpus.py  # download PMC PDFs
    offline/
      pdf_to_images.py    # rasterize PDFs to per-page images
      ocr.py              # layout-aware OCR
      chunker.py          # split OCR text into fixed-token overlapping chunks
      colpali.py          # ColPali page embeddings
      bm25.py             # BM25 index builder
    online/
      retriever.py        # query-time retrieval (Pipeline A and B)
      llm.py              # LLM answerer
    eval/
      eval.py             # run evaluation
      metrics.py          # ROUGE-L, BERTScore, Hit Rate, Recall
    utils.py              # shared helpers (cache_data, print_example_row)
  data/
    raw/
      training13b.json    # BioASQ training data (downloaded externally)
      pdfs/               # downloaded PMC PDFs
    processed/
      images/             # per-page images: {pmcid}/p0001.png
      ocr/                # per-page OCR text
      chunks.jsonl
    cache/                # intermediate DataFrames and mappings
    indexes/
      bm25/
      colpali/
```


## Setup

### 1) Create environment and install dependencies

Mac/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```bash
python -m venv .venv
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


## Authors
- Sai Subramanian
- Khush Patel
