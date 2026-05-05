import os
import pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Mengimpor fungsi dari file preprocessing yang sudah diperbarui
from src.preprocessing import load_and_build_corpus

def main():
    # 1. Pastikan folder 'index' tercipta otomatis agar tidak error 'File Not Found'
    os.makedirs('index', exist_ok=True)

    print("1/4 - Memuat dataset dan melakukan preprocessing (Dual-Corpus)...")
    # Ini akan agak lama karena Sastrawi sedang memotong imbuhan kata
    df = load_and_build_corpus('data/tourism_with_id.csv')

    print("2/4 - Membangun TF-IDF Index...")
    vectorizer = TfidfVectorizer()
    # PENTING: TF-IDF disuapi dengan 'text_stemmed' (kata dasarnya saja)
    tfidf_matrix = vectorizer.fit_transform(df['text_stemmed'])
    pickle.dump((vectorizer, tfidf_matrix), open('index/tfidf.pkl', 'wb'))

    print("3/4 - Membangun BM25 Index...")
    # PENTING: BM25 disuapi dengan 'tokens' (list kata dasar)
    bm25 = BM25Okapi(df['tokens'].tolist())
    pickle.dump(bm25, open('index/bm25.pkl', 'wb'))

    print("4/4 - Membangun Dense Index (FAISS)...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    # PENTING: FAISS disuapi dengan 'text_clean' (teks asli utuh agar maknanya tidak hilang)
    embeddings = model.encode(df['text_clean'].tolist(), show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1]) 
    index.add(embeddings)
    faiss.write_index(index, 'index/dense.faiss')
    np.save('index/embeddings.npy', embeddings)

    # Kita simpan juga dataframe aslinya agar gampang dipanggil di app.py nanti
    df.to_pickle('index/dataset_processed.pkl')

    print("SELESAI! Semua index dan dataset berhasil dibangun dan disimpan di folder 'index/'.")

if __name__ == '__main__':
    main()