from retrieval.preprocessing import build_stemmed_text, tokenize

def retrieve_bm25(query, bm25, df, top_k=10):
    query_stemmed = build_stemmed_text(query)
    tokens = tokenize(query_stemmed)
    
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[::-1][:top_k]
    
    results = df.iloc[top_idx].copy()
    results['score'] = scores[top_idx]
    return results