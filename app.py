import streamlit as st
import pandas as pd
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Impor mesin pencari yang baru saja kita buat
from src.retrieval.tfidf_retrieval import retrieve_tfidf
from src.retrieval.bm25_retrieval import retrieve_bm25
from src.retrieval.dense_retrieval import retrieve_dense
from src.fusion import reciprocal_rank_fusion

# Konfigurasi Halaman Web
st.set_page_config(page_title="Pencarian Wisata", page_icon="🏝️", layout="wide")

# Muat data ke memori (Hanya dijalankan sekali agar web tidak lemot)
@st.cache_resource
def load_systems():
    df = pd.read_pickle('index/dataset_processed.pkl')
    vectorizer, tfidf_matrix = pickle.load(open('index/tfidf.pkl', 'rb'))
    bm25 = pickle.load(open('index/bm25.pkl', 'rb'))
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    faiss_index = faiss.read_index('index/dense.faiss')
    return df, vectorizer, tfidf_matrix, bm25, model, faiss_index

df, vectorizer, tfidf_matrix, bm25, model, faiss_index = load_systems()

# --- TAMPILAN ANTARMUKA ---
st.title("🏝️ Mesin Pencari Destinasi Wisata Indonesia")
st.markdown("Coba ketik kriteria wisata menggunakan bahasa sehari-hari. Contoh: *pantai pasir putih yang sepi di Lombok*")

query = st.text_input("🔍 Masukkan kriteria wisata Anda:")

col1, col2 = st.columns([1, 3])
with col1:
    method = st.selectbox("Pilih Mesin Pencari:", ["Hybrid (RRF) ⭐", "Semantic (Dense)", "BM25", "TF-IDF"])
    top_k = st.slider("Jumlah hasil:", 5, 20, 10)

if st.button("Cari") and query:
    if method == "TF-IDF":
        results = retrieve_tfidf(query, vectorizer, tfidf_matrix, df, top_k)
    elif method == "BM25":
        results = retrieve_bm25(query, bm25, df, top_k)
    elif method == "Semantic (Dense)":
        results = retrieve_dense(query, model, faiss_index, df, top_k)
    else:
        # Menjalankan Hybrid RRF
        tfidf_r = retrieve_tfidf(query, vectorizer, tfidf_matrix, df, top_k=30)
        bm25_r  = retrieve_bm25(query, bm25, df, top_k=30)
        dense_r = retrieve_dense(query, model, faiss_index, df, top_k=30)
        
        # Gabungkan ranking
        rrf_ranking = reciprocal_rank_fusion(
            tfidf_r['Place_Id'].tolist(),
            bm25_r['Place_Id'].tolist(),
            dense_r['Place_Id'].tolist(),
            k=60, top_k=top_k
        )
        
        # Susun ulang hasil
        rrf_ids = [doc_id for doc_id, _ in rrf_ranking]
        scores_dict = dict(rrf_ranking)
        results = df[df['Place_Id'].isin(rrf_ids)].copy()
        results['score'] = results['Place_Id'].map(scores_dict)
        results = results.sort_values('score', ascending=False)

    # Menampilkan Hasil ke Layar
    st.divider()
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        with st.expander(f"#{i} - 📍 {row['Place_Name']} — {row['City']} ({row['Category']})"):
            st.caption(f"Skor Relevansi: {row['score']:.4f}")
            st.write(row['Description'])