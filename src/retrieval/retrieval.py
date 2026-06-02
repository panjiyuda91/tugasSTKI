import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.preprocessing import clean_text_semantic, clean_text_lexical
from src.retrieval.fusion import reciprocal_rank_fusion, get_top_k


class HybridRetriever:
    def __init__(self, tfidf_vectorizer, tfidf_matrix, bm25,
                 faiss_index, sbert_model, stemmer, stopword_remover):
        self.tfidf_vec  = tfidf_vectorizer
        self.tfidf_mat  = tfidf_matrix
        self.bm25       = bm25
        self.faiss_idx  = faiss_index
        self.sbert      = sbert_model
        self.stemmer    = stemmer
        self.sw_remover = stopword_remover

    def tfidf(self, query, top_k=10):
        q = clean_text_lexical(query, self.stemmer, self.sw_remover)
        v = self.tfidf_vec.transform([q])
        s = cosine_similarity(v, self.tfidf_mat).flatten()
        return np.argsort(s)[::-1][:top_k].tolist()

    def bm25_search(self, query, top_k=10):
        q = clean_text_lexical(query, self.stemmer, self.sw_remover).split()
        s = self.bm25.get_scores(q)
        return np.argsort(s)[::-1][:top_k].tolist()

    def dense(self, query, top_k=10):
        q = clean_text_semantic(query)
        v = self.sbert.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        ).astype('float32')
        _, idx = self.faiss_idx.search(v, top_k)
        return idx[0].tolist()

    def hybrid(self, query, top_k=10):
        t = self.tfidf(query, top_k * 2)
        b = self.bm25_search(query, top_k * 2)
        d = self.dense(query, top_k * 2)
        fused = reciprocal_rank_fusion([t, b, d])
        return {
            'hybrid': get_top_k(fused, top_k),
            'tfidf':  t[:top_k],
            'bm25':   b[:top_k],
            'dense':  d[:top_k],
        }