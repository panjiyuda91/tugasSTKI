# evaluation.py
import json
import pytrec_eval

def build_pool(query_id, tfidf_ids, bm25_ids, dense_ids, rrf_ids, depth=10):
    """
    Kumpulkan dokumen unik dari top-depth masing-masing metode.
    Inilah yang akan dianotasi manual — bukan seluruh 400 dokumen.
    """
    pool = set(tfidf_ids[:depth]) | set(bm25_ids[:depth]) \
         | set(dense_ids[:depth]) | set(rrf_ids[:depth])
    return {query_id: list(pool)}


def load_qrels(path='qrels.json') -> dict:
    """
    Format qrels.json yang diisi setelah anotasi manual:
    {
      "q01": {"place_42": 1, "place_17": 0, "place_88": 1, ...},
      "q02": {...}
    }
    """
    with open(path) as f:
        return json.load(f)


def run_evaluation(qrels: dict, run: dict) -> dict:
    """
    qrels : hasil anotasi manual (format di atas)
    run   : hasil retrieval satu metode, format sama dengan qrels
            {"q01": {"place_42": 0.91, "place_17": 0.76, ...}, ...}
    """
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'ndcg_cut_10', 'recip_rank', 'recall_10'}
    )
    per_query = evaluator.evaluate(run)
    # Rata-rata semua query
    avg = {
        metric: sum(v[metric] for v in per_query.values()) / len(per_query)
        for metric in ['ndcg_cut_10', 'recip_rank', 'recall_10']
    }
    return avg