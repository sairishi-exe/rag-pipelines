import csv
import json
import os
import time
import requests

from src.config import CACHE_DIR, CONTEXT_TOP_K, DB_PATH, EVAL_DIR, TOP_P_PAGES
from src.offline.bm25 import load_index, BM25_INDEX_PATH
from src.offline.colpali import init_model, embed_query
from src.online.retriever import retrieve_pipeline_a, retrieve_pipeline_b
from src.online.llm import answer_question
from src.db.qdrant_db import get_client
from src.eval.metrics import (
    exact_match, rouge_l, bert_score,
    hit_rate_at_k, recall_at_k,
    page_hit_rate, page_recall,
)


### Data loading

def load_qa_dataset(path: str) -> list[dict]:
    """Load QA pairs from the subset cache JSONL."""
    with open(path) as f:
        return [json.loads(line) for line in f]


### Setup verification

def verify_setup():
    """Make sure BM25 index, SQLite, Ollama, and Qdrant are all available."""
    errors = []

    # BM25 index
    if not os.path.isfile(BM25_INDEX_PATH):
        errors.append(f"BM25 index not found: {BM25_INDEX_PATH}")

    # SQLite
    if not os.path.isfile(DB_PATH):
        errors.append(f"SQLite DB not found: {DB_PATH}")

    # Ollama (LLM server on port 11434)
    try:
        requests.get("http://localhost:11434/api/version", timeout=3)
    except Exception:
        errors.append("Ollama not reachable at localhost:11434")

    # Qdrant (vector DB on port 6333)
    try:
        client = get_client()
        client.get_collections()
    except Exception:
        errors.append("Qdrant not reachable at localhost:6333")

    if errors:
        print("Setup check failed:")
        for e in errors:
            print(f"  - {e}")
        return False
    return True


### Per-question evaluation

def run_pipeline_a(query, bm25):
    """Run Pipeline A on a single question.
    Returns (prediction, retrieved_pmcids, retrieval_latency, llm_latency)."""
    # retrieval: BM25 over full corpus
    t0 = time.time()
    chunks = retrieve_pipeline_a(query, bm25, CONTEXT_TOP_K)
    retrieval_latency = time.time() - t0

    # LLM answer
    t1 = time.time()
    try:
        prediction = answer_question(query, chunks)
    except Exception:
        prediction = "unanswerable"
    llm_latency = time.time() - t1

    retrieved_pmcids = [c["pmcid"] for c in chunks]
    return prediction, retrieved_pmcids, retrieval_latency, llm_latency


def run_pipeline_b(query, bm25, model, processor, qdrant_client):
    """Run Pipeline B on a single question.
    Returns (prediction, retrieved_pmcids, page_hits, retrieval_latency, llm_latency)."""
    # retrieval: ColPali embedding + Qdrant query + BM25 re-rank on candidates
    t0 = time.time()
    query_embeddings = embed_query(model, processor, query)
    chunks, page_hits = retrieve_pipeline_b(
        query, bm25, query_embeddings, qdrant_client, TOP_P_PAGES, CONTEXT_TOP_K
    )
    retrieval_latency = time.time() - t0

    # LLM answer
    t1 = time.time()
    try:
        prediction = answer_question(query, chunks)
    except Exception:
        prediction = "unanswerable"
    llm_latency = time.time() - t1

    retrieved_pmcids = [c["pmcid"] for c in chunks]
    return prediction, retrieved_pmcids, page_hits, retrieval_latency, llm_latency


### Output formatting

def save_csv(results_a, results_b, path):
    """Save comparison table as CSV."""
    rows = [
        ["Metric", "Pipeline A (BM25 only)", "Pipeline B (ColPali + BM25)"],
        ["Exact Match", f"{results_a['exact_match']:.1%}", f"{results_b['exact_match']:.1%}"],
        ["ROUGE-L (F1)", f"{results_a['rouge_l_f1']:.3f}", f"{results_b['rouge_l_f1']:.3f}"],
        ["BERTScore (F1)", f"{results_a['bert_score_f1']:.3f}", f"{results_b['bert_score_f1']:.3f}"],
        ["Hit Rate@1", f"{results_a['hit_rate_1']:.1%}", f"{results_b['hit_rate_1']:.1%}"],
        ["Hit Rate@3", f"{results_a['hit_rate_3']:.1%}", f"{results_b['hit_rate_3']:.1%}"],
        ["Recall@1", f"{results_a['recall_1']:.3f}", f"{results_b['recall_1']:.3f}"],
        ["Recall@3", f"{results_a['recall_3']:.3f}", f"{results_b['recall_3']:.3f}"],
        ["Retrieval Latency (s)", f"{results_a['avg_retrieval_latency']:.2f}", f"{results_b['avg_retrieval_latency']:.2f}"],
        ["LLM Latency (s)", f"{results_a['avg_llm_latency']:.1f}", f"{results_b['avg_llm_latency']:.1f}"],
        ["Total Latency (s)", f"{results_a['avg_total_latency']:.1f}", f"{results_b['avg_total_latency']:.1f}"],
        ["Page Hit Rate", "—", f"{results_b['page_hit_rate']:.1%}"],
        ["Page Recall", "—", f"{results_b['page_recall']:.3f}"],
    ]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"CSV saved: {path}")


def save_json(results_a, results_b, per_question, path):
    """Save full results (aggregate + per-question) as JSON."""
    output = {
        "aggregate": {
            "pipeline_a": results_a,
            "pipeline_b": results_b,
        },
        "per_question": per_question,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved: {path}")


### Main Run
# Pipelines run separately to avoid LLM warm/cold variance.
# Pipeline A runs all questions first, then Pipeline B runs all questions.

def main():
    # verify setup
    if not verify_setup():
        return

    # load QA dataset
    qa_dataset = load_qa_dataset(os.path.join(CACHE_DIR, "dataframe_subset_cache.json"))
    n = len(qa_dataset)

    # collect predictions + ideal answers for BERTScore (batched after each pipeline)
    per_question = [{
        "question": qa["body"],
        "exact_answer": qa["exact_answer"],
        "gold_pmcids": qa["pmcid_oa"],
    } for qa in qa_dataset]
    all_ideal_answers = [qa["ideal_answer"] for qa in qa_dataset]

    # accumulators: each key is a metric, each value is a list that grows by one per question
    # after both pipelines, we average each list to get the final aggregate score
    a_metrics = {"em": [], "rouge_f1": [], "hr1": [], "hr3": [], "r1": [], "r3": [],
                 "ret_lat": [], "llm_lat": []}
    b_metrics = {"em": [], "rouge_f1": [], "hr1": [], "hr3": [], "r1": [], "r3": [],
                 "ret_lat": [], "llm_lat": [], "page_hr": [], "page_r": []}
    a_predictions, b_predictions = [], []

    # --- Pipeline A: BM25 over full corpus ---
    print("Loading BM25 index...")
    bm25 = load_index()
    print(f"\n--- Pipeline A (BM25 only) --- {n} questions\n")

    for i, qa in enumerate(qa_dataset, 1):
        query = qa["body"]
        exact_answers = qa["exact_answer"]
        ideal_answers = qa["ideal_answer"]
        gold_pmcids = qa["pmcid_oa"]

        print(f"  [A {i}/{n}] {query[:70]}...")

        pred_a, pmcids_a, ret_lat_a, llm_lat_a = run_pipeline_a(query, bm25)
        a_predictions.append(pred_a)

        # answer quality: exact match vs exact_answer, ROUGE-L vs ideal_answer
        a_metrics["em"].append(exact_match(pred_a, exact_answers))
        a_metrics["rouge_f1"].append(rouge_l(pred_a, ideal_answers)["f1"])
        # retrieval quality: do retrieved chunks come from gold documents?
        a_metrics["hr1"].append(hit_rate_at_k(pmcids_a, gold_pmcids, k=1))
        a_metrics["hr3"].append(hit_rate_at_k(pmcids_a, gold_pmcids, k=3))
        a_metrics["r1"].append(recall_at_k(pmcids_a, gold_pmcids, k=1))
        a_metrics["r3"].append(recall_at_k(pmcids_a, gold_pmcids, k=3))
        a_metrics["ret_lat"].append(ret_lat_a)
        a_metrics["llm_lat"].append(llm_lat_a)

        per_question[i - 1]["pipeline_a"] = {
            "prediction": pred_a,
            "retrieved_pmcids": pmcids_a,
            "retrieval_latency": ret_lat_a,
            "llm_latency": llm_lat_a,
            "exact_match": a_metrics["em"][-1],
            "rouge_l_f1": a_metrics["rouge_f1"][-1],
            "hit_rate_1": a_metrics["hr1"][-1],
            "hit_rate_3": a_metrics["hr3"][-1],
            "recall_1": a_metrics["r1"][-1],
            "recall_3": a_metrics["r3"][-1],
        }

    # --- Pipeline B: ColPali pre-filter + BM25 re-rank ---
    # Load ColPali model only when needed (frees memory during Pipeline A)
    print("\nLoading ColPali model + Qdrant client...")
    model, processor = init_model()
    qdrant_client = get_client()
    print(f"\n--- Pipeline B (ColPali + BM25) --- {n} questions\n")

    for i, qa in enumerate(qa_dataset, 1):
        query = qa["body"]
        exact_answers = qa["exact_answer"]
        ideal_answers = qa["ideal_answer"]
        gold_pmcids = qa["pmcid_oa"]

        print(f"  [B {i}/{n}] {query[:70]}...")

        pred_b, pmcids_b, page_hits_b, ret_lat_b, llm_lat_b = run_pipeline_b(
            query, bm25, model, processor, qdrant_client
        )
        b_predictions.append(pred_b)

        # answer quality
        b_metrics["em"].append(exact_match(pred_b, exact_answers))
        b_metrics["rouge_f1"].append(rouge_l(pred_b, ideal_answers)["f1"])
        # retrieval quality
        b_metrics["hr1"].append(hit_rate_at_k(pmcids_b, gold_pmcids, k=1))
        b_metrics["hr3"].append(hit_rate_at_k(pmcids_b, gold_pmcids, k=3))
        b_metrics["r1"].append(recall_at_k(pmcids_b, gold_pmcids, k=1))
        b_metrics["r3"].append(recall_at_k(pmcids_b, gold_pmcids, k=3))
        b_metrics["ret_lat"].append(ret_lat_b)
        b_metrics["llm_lat"].append(llm_lat_b)
        # page-level: how well did ColPali's pre-filter surface gold documents?
        b_metrics["page_hr"].append(page_hit_rate(page_hits_b, gold_pmcids))
        b_metrics["page_r"].append(page_recall(page_hits_b, gold_pmcids))

        per_question[i - 1]["pipeline_b"] = {
            "prediction": pred_b,
            "retrieved_pmcids": pmcids_b,
            "retrieval_latency": ret_lat_b,
            "llm_latency": llm_lat_b,
            "exact_match": b_metrics["em"][-1],
            "rouge_l_f1": b_metrics["rouge_f1"][-1],
            "hit_rate_1": b_metrics["hr1"][-1],
            "hit_rate_3": b_metrics["hr3"][-1],
            "recall_1": b_metrics["r1"][-1],
            "recall_3": b_metrics["r3"][-1],
            "page_hit_rate": b_metrics["page_hr"][-1],
            "page_recall": b_metrics["page_r"][-1],
        }

    # --- BERTScore + aggregation + output ---
    # BERTScore computed after both pipelines in one batch for efficiency
    print("\nComputing BERTScore...")
    a_bert = bert_score(a_predictions, all_ideal_answers)
    b_bert = bert_score(b_predictions, all_ideal_answers)

    # aggregate: average each metric's per-question scores across all questions
    avg = lambda vals: sum(vals) / n

    results_a = {
        "exact_match": avg(a_metrics["em"]),
        "rouge_l_f1": avg(a_metrics["rouge_f1"]),
        "bert_score_f1": avg([s["f1"] for s in a_bert]),
        "hit_rate_1": avg(a_metrics["hr1"]),
        "hit_rate_3": avg(a_metrics["hr3"]),
        "recall_1": avg(a_metrics["r1"]),
        "recall_3": avg(a_metrics["r3"]),
        "avg_retrieval_latency": avg(a_metrics["ret_lat"]),
        "avg_llm_latency": avg(a_metrics["llm_lat"]),
        "avg_total_latency": avg([r + l for r, l in zip(a_metrics["ret_lat"], a_metrics["llm_lat"])]),
    }

    results_b = {
        "exact_match": avg(b_metrics["em"]),
        "rouge_l_f1": avg(b_metrics["rouge_f1"]),
        "bert_score_f1": avg([s["f1"] for s in b_bert]),
        "hit_rate_1": avg(b_metrics["hr1"]),
        "hit_rate_3": avg(b_metrics["hr3"]),
        "recall_1": avg(b_metrics["r1"]),
        "recall_3": avg(b_metrics["r3"]),
        "avg_retrieval_latency": avg(b_metrics["ret_lat"]),
        "avg_llm_latency": avg(b_metrics["llm_lat"]),
        "avg_total_latency": avg([r + l for r, l in zip(b_metrics["ret_lat"], b_metrics["llm_lat"])]),
        "page_hit_rate": avg(b_metrics["page_hr"]),
        "page_recall": avg(b_metrics["page_r"]),
    }

    # output results as CSV (for Google Sheets) and JSON (for re-analysis)
    save_csv(results_a, results_b, os.path.join(EVAL_DIR, f"eval_results_p{TOP_P_PAGES}.csv"))
    save_json(results_a, results_b, per_question, os.path.join(EVAL_DIR, f"eval_results_p{TOP_P_PAGES}.json"))


if __name__ == "__main__":
    main()
