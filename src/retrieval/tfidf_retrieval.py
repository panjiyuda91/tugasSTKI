from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import build_stemmed_text

def retrieve_tfidf(query, vectorizer, tfidf_matrix, df, top_k=10):
    # Kueri harus di-stemming agar adil
    query_stemmed = build_stemmed_text(query)
    query_vec = vectorizer.transform([query_stemmed])
    
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    
    results = df.iloc[top_idx].copy()
    results['score'] = scores[top_idx]
    return results