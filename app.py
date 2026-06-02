import os
import sys
import pickle
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from src.retrieval.fusion import reciprocal_rank_fusion, get_top_k
from src.retrieval.preprocessing import (
    create_stemmer, create_stopword_remover,
    clean_text_semantic, clean_text_lexical
)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
INDEX_DIR = os.path.join(os.path.dirname(__file__), 'index')

st.set_page_config(
    page_title="Search Engine Wisata Indonesia",
    page_icon="🗺️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# LOAD RESOURCES (cached agar tidak reload setiap interaksi)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    df = pd.read_pickle(f'{INDEX_DIR}/dataset_processed.pkl')

    tfidf_data = pickle.load(open(f'{INDEX_DIR}/tfidf.pkl', 'rb'))
    tfidf_vectorizer = tfidf_data['vectorizer']
    tfidf_matrix     = tfidf_data['matrix']

    bm25 = pickle.load(open(f'{INDEX_DIR}/bm25.pkl', 'rb'))

    faiss_index = faiss.read_index(f'{INDEX_DIR}/dense.faiss')

    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    stemmer          = create_stemmer()
    stopword_remover = create_stopword_remover()

    return df, tfidf_vectorizer, tfidf_matrix, bm25, faiss_index, sbert_model, stemmer, stopword_remover

with st.spinner("⏳ Memuat index dan model..."):
    df, tfidf_vec, tfidf_mat, bm25, faiss_idx, sbert, stemmer, sw_remover = load_resources()

# ──────────────────────────────────────────────────────────────
# RETRIEVAL FUNCTIONS
# ──────────────────────────────────────────────────────────────
def retrieve_tfidf(query, top_k=10):
    q = clean_text_lexical(query, stemmer, sw_remover)
    v = tfidf_vec.transform([q])
    s = cosine_similarity(v, tfidf_mat).flatten()
    return np.argsort(s)[::-1][:top_k].tolist()

def retrieve_bm25(query, top_k=10):
    q = clean_text_lexical(query, stemmer, sw_remover).split()
    s = bm25.get_scores(q)
    return np.argsort(s)[::-1][:top_k].tolist()

def retrieve_dense(query, top_k=10):
    q = clean_text_semantic(query)
    v = sbert.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    _, idx = faiss_idx.search(v, top_k)
    return idx[0].tolist()

def hybrid_search(query, top_k=10):
    t = retrieve_tfidf(query, top_k * 2)
    b = retrieve_bm25(query, top_k * 2)
    d = retrieve_dense(query, top_k * 2)
    fused = reciprocal_rank_fusion([t, b, d])
    return get_top_k(fused, top_k), t[:top_k], b[:top_k], d[:top_k]

# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────
st.title("🗺️ Search Engine Destinasi Wisata Indonesia")
st.caption("Hybrid Retrieval: TF-IDF + BM25 + Semantic Search (RRF Fusion)")

st.markdown("---")

col_search, col_config = st.columns([3, 1])

with col_search:
    query = st.text_input(
        "🔍 Cari destinasi wisata...",
        placeholder="Contoh: pantai untuk snorkeling, wisata alam menenangkan, Taman Nasional Komodo...",
    )

with col_config:
    method = st.selectbox(
        "Metode",
        options=["Hybrid (Rekomendasi)", "TF-IDF", "BM25", "Semantic (Dense)"],
        index=0,
    )
    top_k = st.slider("Jumlah Hasil", 5, 20, 10)

# Filter sidebar
with st.sidebar:
    st.header("🔧 Filter")
    all_categories = sorted(df['Category'].dropna().unique())
    all_cities     = sorted(df['City'].dropna().unique())

    selected_categories = st.multiselect("Kategori", all_categories)
    selected_cities     = st.multiselect("Kota/Daerah", all_cities)
    min_rating          = st.slider("Rating Minimum", 0.0, 5.0, 0.0, 0.1)
    max_price           = st.number_input("Harga Tiket Maks (Rp)", 0, 500000, 500000, 10000)

    st.markdown("---")
    st.markdown("**Tentang Sistem**")
    st.markdown(
        "Sistem ini menggunakan **Hybrid Retrieval** yang "
        "menggabungkan pencarian leksikal (TF-IDF & BM25) "
        "dan semantik (Sentence Transformer + FAISS) "
        "melalui **Reciprocal Rank Fusion (RRF)**."
    )

# ──────────────────────────────────────────────────────────────
# SEARCH & DISPLAY
# ──────────────────────────────────────────────────────────────
if query:
    with st.spinner("🔍 Mencari..."):
        if method == "Hybrid (Rekomendasi)":
            indices, _, _, _ = hybrid_search(query, top_k * 3)
        elif method == "TF-IDF":
            indices = retrieve_tfidf(query, top_k * 3)
        elif method == "BM25":
            indices = retrieve_bm25(query, top_k * 3)
        else:
            indices = retrieve_dense(query, top_k * 3)

        results = df.iloc[indices].copy()

        # Terapkan filter
        if selected_categories:
            results = results[results['Category'].isin(selected_categories)]
        if selected_cities:
            results = results[results['City'].isin(selected_cities)]
        results = results[results['Rating'] >= min_rating]
        results = results[results['Price'] <= max_price]
        results = results.head(top_k)

    st.success(f"Ditemukan {len(results)} destinasi untuk: **'{query}'**")
    st.markdown("---")

    if results.empty:
        st.warning("Tidak ada hasil yang cocok dengan filter. Coba longgarkan filter.")
    else:
        for rank, (_, row) in enumerate(results.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### {rank}. {row['Place_Name']}")
                    st.markdown(f"📍 **{row['City']}** &nbsp;|&nbsp; 🏷️ **{row['Category']}**")
                    st.write(row['Description'])
                with col2:
                    st.metric("⭐ Rating", f"{row['Rating']:.1f}")
                    price_display = "Gratis" if row['Price'] == 0 else f"Rp {int(row['Price']):,}"
                    st.metric("🎫 Tiket", price_display)
                    if pd.notna(row.get('Time_minutes')) and row['Time_minutes'] > 0:
                        st.metric("⏱️ Waktu", f"{int(row['Time_minutes'])} mnt")
                st.markdown("---")
else:
    st.info("💡 Ketik kata kunci di atas untuk mulai mencari destinasi wisata.")

    # Contoh query
    st.markdown("### 💡 Coba query ini:")
    example_queries = [
        "pantai indah untuk snorkeling",
        "wisata alam menenangkan",
        "tempat bersejarah Yogyakarta",
        "taman hiburan keluarga",
        "Gunung Bromo"
    ]
    cols = st.columns(len(example_queries))
    for col, eq in zip(cols, example_queries):
        col.button(eq, use_container_width=True)