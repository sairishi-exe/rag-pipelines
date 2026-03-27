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
    all_f1s = []
    for pred, ideal in zip(predictions, ideal_answers):
        if isinstance(ideal, str):
            ideal = [ideal]

        # bert_score_fn expects two equal-length lists and scores them pairwise
        # (index 0 vs index 0, index 1 vs index 1, etc.)
        # so if a question has 2 ideal variants, we repeat the prediction twice:
        #   preds_repeated = ["FTY720 is the drug",  "FTY720 is the drug"]
        #   ideal           = ["Fingolimod is the drug", "FTY720 is the drug"]
        preds_repeated = [pred] * len(ideal)

        # returns (Precision, Recall, F1) — each a tensor with one value per pair
        # e.g. F1 = tensor([0.90, 1.00]) for 2 ideal variants
        # F1 combines precision and recall — primary metric
        P, R, F1 = bert_score_fn(preds_repeated, ideal, lang="en", verbose=False)

        # take the best-scoring variant (by F1) and store all three scores
        best_idx = int(F1.argmax())
        all_f1s.append({
            "precision": float(P[best_idx]),
            "recall": float(R[best_idx]),
            "f1": float(F1[best_idx]),
        })

    return all_f1s


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
