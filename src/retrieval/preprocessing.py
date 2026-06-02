import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def create_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


def create_stopword_remover():
    factory = StopWordRemoverFactory()
    return factory.create_stop_word_remover()


def clean_text_semantic(text: str) -> str:
    """
    Preprocessing untuk Semantic Search (TANPA stemming).
    Hanya lowercase + hapus karakter aneh.
    Tujuan: menjaga konteks makna untuk Sentence Transformer.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_lexical(text: str, stemmer, stopword_remover) -> str:
    """
    Preprocessing untuk Lexical Search (DENGAN stemming + stopword removal).
    Tujuan: meningkatkan akurasi pencocokan kata kunci TF-IDF & BM25.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_combined_text(row: pd.Series) -> str:
    """Gabungkan kolom-kolom penting menjadi satu field teks."""
    parts = [
        str(row.get('Place_Name', '') or ''),
        str(row.get('Description', '') or ''),
        str(row.get('Category', '') or ''),
        str(row.get('City', '') or ''),
    ]
    return ' '.join(filter(None, parts))