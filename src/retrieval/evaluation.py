import numpy as np
from typing import List, Dict


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Hitung Discounted Cumulative Gain pada posisi k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = [rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)]
    return sum(gains)


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Hitung Normalized DCG pada posisi k."""
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mean_reciprocal_rank(relevances_list: List[List[int]]) -> float:
    """
    Hitung Mean Reciprocal Rank (MRR) dari semua kueri.
    MRR mengukur seberapa cepat dokumen relevan pertama ditemukan.
    """
    rr_scores = []
    for relevances in relevances_list:
        rr = 0.0
        for rank, rel in enumerate(relevances, start=1):
            if rel > 0:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def recall_at_k(relevances: List[int], total_relevant: int, k: int) -> float:
    """Hitung Recall pada posisi k."""
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(relevances[:k])
    return retrieved_relevant / total_relevant


def average_precision(relevances: List[int]) -> float:
    """Hitung Average Precision untuk satu kueri."""
    hits, sum_precisions = 0, 0.0
    for rank, rel in enumerate(relevances, start=1):
        if rel > 0:
            hits += 1
            sum_precisions += hits / rank
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0
    return sum_precisions / total_relevant


def mean_average_precision(relevances_list: List[List[int]]) -> float:
    """Hitung MAP dari semua kueri."""
    ap_scores = [average_precision(rels) for rels in relevances_list]
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def cohens_kappa(annotations_a: List[int], annotations_b: List[int]) -> float:
    """
    Hitung Cohen's Kappa untuk Inter-Annotator Agreement.
    Mengukur kesepakatan anotasi antara dua anotator.
    """
    n = len(annotations_a)
    if n == 0:
        return 0.0
    agree = sum(a == b for a, b in zip(annotations_a, annotations_b))
    p_o = agree / n  # observed agreement

    # Expected agreement
    classes = set(annotations_a + annotations_b)
    p_e = sum(
        (annotations_a.count(c) / n) * (annotations_b.count(c) / n)
        for c in classes
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def evaluate_all_metrics(
    relevances_per_query: Dict[str, List[int]],
    total_relevant_per_query: Dict[str, int],
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluasi semua metrik sekaligus.

    Args:
        relevances_per_query: {query: [1,0,1,0,...]} — urutan relevansi hasil retrieval
        total_relevant_per_query: {query: jumlah_total_dokumen_relevan_di_dataset}
        k: cutoff (default 10)

    Returns:
        Dict berisi nDCG@k, MRR, Recall@k, MAP
    """
    all_relevances = list(relevances_per_query.values())
    ndcg_scores = [ndcg_at_k(rels, k) for rels in all_relevances]
    recall_scores = [
        recall_at_k(rels, total_relevant_per_query.get(q, 1), k)
        for q, rels in relevances_per_query.items()
    ]

    return {
        f'nDCG@{k}': float(np.mean(ndcg_scores)),
        'MRR': mean_reciprocal_rank(all_relevances),
        f'Recall@{k}': float(np.mean(recall_scores)),
        'MAP': mean_average_precision(all_relevances),
    }