from __future__ import annotations

def reciprocal_rank_fusion(
    *ranked_lists: list[str],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Menggabungkan beberapa ranked list menjadi satu ranking tunggal via RRF.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]

def build_ranked_id_list(retrieval_results, id_column: str = 'Place_Id') -> list[str]:
    """
    Helper: ubah DataFrame hasil retrieval menjadi list doc_id terurut.
    """
    return retrieval_results[id_column].tolist()