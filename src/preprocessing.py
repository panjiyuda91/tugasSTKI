# preprocessing.py
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi PySastrawi (lakukan sekali, mahal jika diulang)
_stemmer = StemmerFactory().create_stemmer()
_stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


def _base_clean(text: str) -> str:
    """Case folding + punctuation removal. Dipakai oleh kedua corpus."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)   # ganti tanda baca dengan spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_clean_text(text: str) -> str:
    """
    Corpus ASLI — hanya base clean.
    Digunakan untuk: Dense Retrieval (FAISS + SentenceTransformer).
    Mempertahankan bentuk kata asli agar embedding semantik tidak terdistorsi.
    """
    return _base_clean(text)


def build_stemmed_text(text: str) -> str:
    """
    Corpus STEMMED — base clean + stopword removal + stemming PySastrawi.
    Digunakan untuk: TF-IDF dan BM25.
    Menyamakan 'wisata', 'pariwisata', 'berwisata' → 'wisata'.
    """
    text = _base_clean(text)
    text = _stopword_remover.remove(text)   # hapus stopword Bahasa Indonesia
    text = _stemmer.stem(text)              # stemming PySastrawi
    return text


def tokenize(text: str) -> list[str]:
    """Tokenisasi sederhana berbasis spasi. Dipakai BM25."""
    return text.split()


def load_and_build_corpus(csv_path: str) -> pd.DataFrame:
    """
    Load dataset dan hasilkan dua kolom corpus sekaligus.

    Kolom output:
        text_clean   → untuk Dense Retrieval
        text_stemmed → untuk TF-IDF dan BM25
        tokens       → list token dari text_stemmed, langsung untuk BM25Okapi
    """
    df = pd.read_csv(csv_path)

    # Gabung nama tempat + deskripsi sebagai unit dokumen
    raw = df['Place_Name'].fillna('') + '. ' + df['Description'].fillna('')

    df['text_clean']   = raw.apply(build_clean_text)
    df['text_stemmed'] = raw.apply(build_stemmed_text)
    df['tokens']       = df['text_stemmed'].apply(tokenize)

    return df