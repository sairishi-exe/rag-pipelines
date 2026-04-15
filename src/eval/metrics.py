from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

# .score(ref, pred) returns {"rougeL": Score(precision, recall, fmeasure)}
# use_stemmer=True so morphological variants match (e.g. "approved" ≈ "approving")
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


### Answer quality metrics

def exact_match(prediction: str, exact_answers: list[str]) -> int:
    """
    Returns 1 if prediction contains any of the exact answers (case-insensitive), else 0.
    """
    pred_lower = prediction.lower()
    return int(any(ans.lower() in pred_lower for ans in exact_answers))


def rouge_l(prediction: str, ideal_answers: list[str] | str) -> dict:
    """
    Score prediction against ideal answer(s) using ROUGE-L. 

    Returns best {precision, recall, f1}.
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
    Score all predictions against ideal answers using BERTScore.
    For questions with multiple ideal answers, takes the max F1.

    Returns list of {precision, recall, f1} dicts.
    """
    # flatten all (prediction, ideal) pairs so we can score them in one batch call
    # questions with multiple ideal answers get repeated predictions
    all_preds = []
    all_refs = []
    group_sizes = []
    for pred, ideal in zip(predictions, ideal_answers):
        if isinstance(ideal, str):
            ideal = [ideal]
        all_preds.extend([pred] * len(ideal))
        all_refs.extend(ideal)
        group_sizes.append(len(ideal))

    # returns (Precision, Recall, F1) tensors, one value per pair
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
    """Returns 1 if any gold PMCID appears in the top-K retrieved chunks, else 0."""
    top_k = set(retrieved_pmcids[:k])
    return int(bool(top_k & set(gold_pmcids)))


def recall_at_k(retrieved_pmcids: list[str], gold_pmcids: list[str], k: int) -> float:
    """Returns fraction of gold PMCIDs found in the top-K retrieved chunks."""
    top_k = set(retrieved_pmcids[:k])
    found = len(top_k & set(gold_pmcids))
    return found / len(gold_pmcids)


### Page-level retrieval metrics (Pipeline B ColPali analysis)

def page_hit_rate(page_hits: list[dict], gold_pmcids: list[str]) -> int:
    """Returns 1 if ColPali's top-P pages include at least one gold PMCID, else 0."""
    retrieved_pmcids = {h["pmcid"] for h in page_hits}
    return int(bool(retrieved_pmcids & set(gold_pmcids)))


def page_recall(page_hits: list[dict], gold_pmcids: list[str]) -> float:
    """Returns fraction of gold PMCIDs that appear in ColPali's top-P pages."""
    retrieved_pmcids = {h["pmcid"] for h in page_hits}
    found = len(retrieved_pmcids & set(gold_pmcids))
    return found / len(gold_pmcids)
