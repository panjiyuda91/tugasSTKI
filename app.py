import os
import sys
import pickle
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# Setup path agar Python bisa mengenali folder src
sys.path.insert(0, os.path.dirname(__file__))

# Cukup import class utama dan fungsi preprocessing, fusion sudah ditangani di dalam class
from src.retrieval.retrieval import HybridRetriever
from src.retrieval.preprocessing import create_stemmer, create_stopword_remover

# ──────────────────────────────────────────────────────────────
# CONFIG & LOAD RESOURCES
# ──────────────────────────────────────────────────────────────
INDEX_DIR = os.path.join(os.path.dirname(__file__), 'index')

st.set_page_config(
    page_title="Search Engine Wisata Indonesia",
    page_icon="🗺️",
    layout="wide",
)

@st.cache_resource
def load_resources():
    # Load dataset
    df = pd.read_pickle(f'{INDEX_DIR}/dataset_processed.pkl')

    # Load file index lexical
    tfidf_data = pickle.load(open(f'{INDEX_DIR}/tfidf.pkl', 'rb'))
    bm25 = pickle.load(open(f'{INDEX_DIR}/bm25.pkl', 'rb'))
    
    # Load file index semantic
    faiss_index = faiss.read_index(f'{INDEX_DIR}/dense.faiss')
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Inisialisasi Preprocessing
    stemmer = create_stemmer()
    stopword_remover = create_stopword_remover()

    # Inisialisasi Otak Sistem
    retriever = HybridRetriever(
        tfidf_vectorizer=tfidf_data['vectorizer'],
        tfidf_matrix=tfidf_data['matrix'],
        bm25=bm25,
        faiss_index=faiss_index,
        sbert_model=sbert_model,
        stemmer=stemmer,
        stopword_remover=stopword_remover
    )

    return df, retriever

with st.spinner("⏳ Memuat index dan model..."):
    df, retriever = load_resources()
    
def set_query(q):
    st.session_state.search_query = q

# ──────────────────────────────────────────────────────────────
# UI: LAYOUT & INPUT
# ──────────────────────────────────────────────────────────────
st.title("🗺️ Search Engine Destinasi Wisata Indonesia")
st.caption("Hybrid Retrieval: TF-IDF + BM25 + Semantic Search (RRF Fusion)")
st.markdown("---")

col_search, col_config = st.columns([3, 1])

with col_search:
    query = st.text_input(
        "🔍 Cari destinasi wisata...",
        placeholder="Contoh: pantai untuk snorkeling, wisata alam menenangkan...",
        key="search_query"
    )

with col_config:
    method = st.selectbox(
        "Metode",
        options=["Hybrid (Rekomendasi)", "TF-IDF", "BM25", "Semantic (Dense)"],
        index=0,
    )
    top_k = st.slider("Jumlah Hasil", 5, 20, 10)

# UI: SIDEBAR FILTER
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
# CORE LOGIC: SEARCH & DISPLAY
# ──────────────────────────────────────────────────────────────
if query:
    with st.spinner("🔍 Mencari..."):
        # Logika pemanggilan method yang sangat clean
        if method == "Hybrid (Rekomendasi)":
            indices = retriever.hybrid(query, top_k)['hybrid']
        elif method == "TF-IDF":
            indices = retriever.tfidf(query, top_k)
        elif method == "BM25":
            indices = retriever.bm25_search(query, top_k)
        else: # Semantic
            indices = retriever.dense(query, top_k)

        # Ambil dataframe berdasarkan index hasil retrieval
        results = df.iloc[indices].copy()

        # Terapkan filter sidebar
        if selected_categories:
            results = results[results['Category'].isin(selected_categories)]
        if selected_cities:
            results = results[results['City'].isin(selected_cities)]
        results = results[results['Rating'] >= min_rating]
        results = results[results['Price'] <= max_price]
        
        # Potong ulang sesuai top_k setelah difilter (opsional, agar konsisten dengan UI)
        results = results.head(top_k)

    # Render Hasil
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

    # Contoh query interaktif
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
        col.button(eq, use_container_width=True, on_click=set_query, args=(eq,))
        # if col.button(eq, use_container_width=True):
        #     # Trick Streamlit: Jika tombol diklik, isi session state agar query jalan
            