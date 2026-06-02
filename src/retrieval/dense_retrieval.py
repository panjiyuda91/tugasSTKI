import numpy as np
from retrieval.preprocessing import build_clean_text

def retrieve_dense(query, model, faiss_index, df, top_k=10):
    # Kueri HANYA dibersihkan tanda bacanya, tanpa stemming
    query_clean = build_clean_text(query)
    
    q_emb = model.encode([query_clean])
    q_emb = q_emb / np.linalg.norm(q_emb) # Normalisasi vektor
    
    scores, indices = faiss_index.search(q_emb, top_k)
    
    results = df.iloc[indices[0]].copy()
    results['score'] = scores[0]
    return results