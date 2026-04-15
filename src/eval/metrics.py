from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

# .score(ref, pred) returns {"rougeL": Score(precision, recall, fmeasure)}
# use_stemmer=True so morphological variants match (e.g. "approved" ≈ "approving")
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


### Answer quality metrics

def exact_match(prediction: str, exact_answers: list[str]) -> int:
    """
    1 if prediction contains any of the exact answers (case-insensitive), else 0.
    """
    pred_lower = prediction.lower()
    return int(any(ans.lower() in pred_lower for ans in exact_answers))


def rouge_l(prediction: str, ideal_answers: list[str] | str) -> dict:
    """
    ROUGE-L: measures word-level overlap using longest common subsequence (LCS).
             Captures word ordering without requiring consecutive matches.
    Returns dict with precision, recall, and f1 (F1 combines both — primary metric).
    """
    if isinstance(ideal_answers, str):
        ideal_answers = [ideal_answers]
    best = max(
        (_scorer.score(ref, prediction)["rougeL"] for ref in ideal_answers),
        key=lambda s: s.fmeasure
    )
    return {"precision": best.precision, "recall": best.recall, "f1": best.fmeasure}


def bert_score(predictions: list[str], ideal_answers: list[list[str] | str]) -> list[dict]:
    """
    BERTScore: measures semantic similarity using contextual embeddings (RoBERTa-large).
               Captures meaning-level matches that word overlap misses (e.g. "FTY720" ≈ "Fingolimod").

    Returns list of dicts with precision, recall, and f1 
    (F1 is the primary metric that combines both precision and recall).
    For questions with multiple ideal answers, scores against each and takes the max by F1.
    """
    # bert_score_fn expects two equal-length lists and scores them pairwise
    # (index 0 vs index 0, index 1 vs index 1, etc.)
    # so if a question has 2 ideal variants, we repeat the prediction twice:
    #   preds_repeated = ["FTY720 is the drug",  "FTY720 is the drug"]
    #   ideal           = ["Fingolimod is the drug", "FTY720 is the drug"]
    #
    # all pairs are flattened into a single batched call to avoid repeated model
    # loading overhead, then regrouped per question to take the max F1.
    all_preds = []
    all_refs = []
    group_sizes = []
    for pred, ideal in zip(predictions, ideal_answers):
        if isinstance(ideal, str):
            ideal = [ideal]
        all_preds.extend([pred] * len(ideal))
        all_refs.extend(ideal)
        group_sizes.append(len(ideal))

    # returns (Precision, Recall, F1) — each a tensor with one value per pair
    # e.g. F1 = tensor([0.90, 1.00]) for 2 ideal variants
    # F1 combines precision and recall — primary metric
    P, R, F1 = bert_score_fn(all_preds, all_refs, lang="en", verbose=False)

    # regroup by question and take the best-scoring variant (by F1)
    results = []
    idx = 0
    for size in group_sizes:
        chunk_f1 = F1[idx:idx + size]
        best = int(chunk_f1.argmax())
        results.append({
            "precision": float(P[idx + best]),
            "recall": float(R[idx + best]),
            "f1": float(F1[idx + best]),
        })
        idx += size

    return results


### Retrieval quality metrics

def hit_rate_at_k(retrieved_pmcids: list[str], gold_pmcids: list[str], k: int) -> int:
    """
    Hit Rate@K: did at least one gold PMCID appear in the top-K retrieved chunks?

    Returns 1 if yes, 0 if no. Evaluated at K=1 and K=3.
    """
    top_k = set(retrieved_pmcids[:k])
    return int(bool(top_k & set(gold_pmcids)))


def recall_at_k(retrieved_pmcids: list[str], gold_pmcids: list[str], k: int) -> float:
    """
    Recall@K: what fraction of gold PMCIDs were found in the top-K retrieved chunks?
    
    Returns 0.0–1.0. Evaluated at K=1 and K=3.
    """
    top_k = set(retrieved_pmcids[:k])
    found = len(top_k & set(gold_pmcids))
    return found / len(gold_pmcids)


### Page-level retrieval metrics (Pipeline B ColPali analysis)

def page_hit_rate(page_hits: list[dict], gold_pmcids: list[str]) -> int:
    """
    Answers: Did ColPali's top-P pages include at least one page from a gold PMCID?
    page_hits: list of {"pmcid": ..., "page_num": ..., "score": ...} from Qdrant.

    Returns 1 if yes, 0 if no.
    """
    retrieved_pmcids = {h["pmcid"] for h in page_hits}
    return int(bool(retrieved_pmcids & set(gold_pmcids)))


def page_recall(page_hits: list[dict], gold_pmcids: list[str]) -> float:
    """
    Answers: What fraction of gold PMCIDs appear in ColPali's top-P pages?
    
    Returns 0.0–1.0.
    """
    retrieved_pmcids = {h["pmcid"] for h in page_hits}
    found = len(retrieved_pmcids & set(gold_pmcids))
    return found / len(gold_pmcids)
