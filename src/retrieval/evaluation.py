import numpy as np
from typing import List, Dict


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Hitung Discounted Cumulative Gain pada posisi k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = [rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)]
    return sum(gains)


def ndcg_at_k(relevances: List[int], k: int, total_relevant_in_pool: int) -> float:
    """
    Hitung Normalized DCG pada posisi k.

    PENTING: ideal_dcg dihitung dari total dokumen relevan di SELURUH POOL
    (gabungan semua metode), bukan hanya dari dokumen yang berhasil diambil
    metode ini. Ini memastikan ideal_dcg SAMA untuk semua metode pada query
    yang sama, sehingga nDCG antar metode benar-benar comparable.

    Args:
        relevances: relevansi dokumen sesuai urutan ranking metode ini, mis. [1,0,1,0,...]
        k: cutoff
        total_relevant_in_pool: jumlah dokumen relevan di seluruh pool untuk query ini
    """
    actual_dcg = dcg_at_k(relevances, k)

    n_relevant_capped = min(total_relevant_in_pool, k)
    ideal_relevances = [1] * n_relevant_capped + [0] * (k - n_relevant_capped)
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
    p_o = agree / n

    classes = set(annotations_a + annotations_b)
    p_e = sum(
        (annotations_a.count(c) / n) * (annotations_b.count(c) / n)
        for c in classes
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def cohens_kappa_per_query(
    annotations_a: Dict[str, Dict[str, int]],
    annotations_b: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """
    Hitung Cohen's Kappa PER QUERY (lebih valid secara metodologis
    dibanding flatten semua query jadi satu list).

    Args:
        annotations_a: {query: {doc_id_str: 0/1}}
        annotations_b: {query: {doc_id_str: 0/1}}

    Returns:
        Dict {query: kappa_score}, plus key '__average__' untuk rata-rata.
    """
    common_queries = [q for q in annotations_a if q in annotations_b]
    per_query_kappa = {}

    for q in common_queries:
        doc_ids = sorted(set(annotations_a[q].keys()) & set(annotations_b[q].keys()))
        if not doc_ids:
            continue
        flat_a = [annotations_a[q][d] for d in doc_ids]
        flat_b = [annotations_b[q][d] for d in doc_ids]
        per_query_kappa[q] = cohens_kappa(flat_a, flat_b)

    if per_query_kappa:
        per_query_kappa['__average__'] = float(np.mean(list(per_query_kappa.values())))

    return per_query_kappa


def evaluate_all_metrics(
    relevances_per_query: Dict[str, List[int]],
    total_relevant_per_query: Dict[str, int],
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluasi semua metrik sekaligus.

    Args:
        relevances_per_query: {query: [1,0,1,0,...]} — relevansi sesuai urutan
                                ranking metode TERTENTU untuk query tersebut
        total_relevant_per_query: {query: jumlah_total_dokumen_relevan_DI_POOL}
                                    HARUS sama untuk semua metode pada query yang sama
        k: cutoff (default 10)

    Returns:
        Dict berisi nDCG@k, MRR, Recall@k, MAP
    """
    missing = [q for q in relevances_per_query if q not in total_relevant_per_query]
    if missing:
        raise ValueError(
            f"total_relevant_per_query tidak lengkap untuk query: {missing}. "
            f"Setiap query harus punya total_relevant yang dihitung dari pool lengkap."
        )

    all_relevances = list(relevances_per_query.values())

    ndcg_scores = [
        ndcg_at_k(rels, k, total_relevant_per_query[q])
        for q, rels in relevances_per_query.items()
    ]
    recall_scores = [
        recall_at_k(rels, total_relevant_per_query[q], k)
        for q, rels in relevances_per_query.items()
    ]

    return {
        f'nDCG@{k}': float(np.mean(ndcg_scores)),
        'MRR': mean_reciprocal_rank(all_relevances),
        f'Recall@{k}': float(np.mean(recall_scores)),
        'MAP': mean_average_precision(all_relevances),
    }