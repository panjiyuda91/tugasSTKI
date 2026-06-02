from collections import defaultdict
from typing import List, Dict, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF).

    Args:
        ranked_lists: List of ranked document ID lists dari masing-masing retriever.
                      Contoh: [[doc_id_1, doc_id_2, ...], [doc_id_3, ...], ...]
        k: Konstanta smoothing (default=60, standard di literatur).

    Returns:
        List of (doc_id, rrf_score) diurutkan dari skor tertinggi.
    """
    scores: Dict[int, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def get_top_k(fused_results: List[Tuple[int, float]], top_k: int = 10) -> List[int]:
    """Ambil top-K doc_id dari hasil RRF."""
    return [doc_id for doc_id, _ in fused_results[:top_k]]