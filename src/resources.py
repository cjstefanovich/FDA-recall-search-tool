import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from preprocessor import preprocess_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data/corpus"
INDEX_DIR = PROJECT_ROOT / "index"

INVERTED_INDEX_PATH = INDEX_DIR / "inverted_index.json"
INDEX_STATS_PATH = INDEX_DIR / "index_stats.json"
METADATA_PATH = INDEX_DIR / "doc_metadata.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_search_resources():
    """
    Load and cache inverted index, stats, and metadata.
    """
    inverted_index = load_json(INVERTED_INDEX_PATH)
    index_stats = load_json(INDEX_STATS_PATH)
    metadata = pd.read_csv(METADATA_PATH).fillna("")
    metadata["doc_id"] = metadata["doc_id"].astype(str)

    return inverted_index, index_stats, metadata


@lru_cache(maxsize=1)
def load_metadata_lookup():
    """
    Build a doc_id keyed metadata lookup once per process.
    """
    _, _, metadata = load_search_resources()
    return metadata.set_index("doc_id", drop=False)


@lru_cache(maxsize=None)
def load_document_tokens(doc_id):
    """
    Load and preprocess one document file once, then reuse it.
    """
    file_path = CORPUS_DIR / f"{doc_id}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    return tuple(preprocess_file(file_path, use_stemming=True))


@lru_cache(maxsize=None)
def load_document_vector(doc_id):
    """
    Build and cache one document TF-IDF vector.
    """
    _, index_stats, _ = load_search_resources()
    idf_values = index_stats["inverse_document_frequency"]

    term_counts = Counter(load_document_tokens(doc_id))
    vector = {}

    for term, tf in term_counts.items():
        if term in idf_values:
            vector[term] = tf * idf_values[term]

    return vector
