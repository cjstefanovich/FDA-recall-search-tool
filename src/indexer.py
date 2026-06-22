import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from preprocessor import preprocess_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data/corpus"
INDEX_DIR = PROJECT_ROOT / "index"

INVERTED_INDEX_PATH = INDEX_DIR / "inverted_index.json"
VOCABULARY_PATH = INDEX_DIR / "vocabulary.json"
INDEX_STATS_PATH = INDEX_DIR / "index_stats.json"


def build_inverted_index(corpus_dir):
    """
    Build an inverted index.

    Format:
    {
        term: {
            doc_id: term_frequency
        }
    }
    """
    inverted_index = defaultdict(dict)
    document_lengths = {}
    document_term_counts = {}
    doc_ids = []

    txt_files = sorted(corpus_dir.glob("*.txt"))

    print(f"Found {len(txt_files)} documents.")

    for file_path in txt_files:
        doc_id = file_path.stem
        doc_ids.append(doc_id)

        tokens = preprocess_file(file_path, use_stemming=True)
        term_counts = Counter(tokens)

        document_lengths[doc_id] = len(tokens)
        document_term_counts[doc_id] = dict(term_counts)

        for term, count in term_counts.items():
            inverted_index[term][doc_id] = count

    return inverted_index, document_lengths, document_term_counts, doc_ids


def compute_index_statistics(inverted_index, document_lengths, document_term_counts, doc_ids):
    """
    Compute document frequency, IDF, and full document vector norms.
    """
    total_documents = len(doc_ids)

    document_frequency = {}
    inverse_document_frequency = {}

    for term, postings in inverted_index.items():
        df = len(postings)
        document_frequency[term] = df

        # Smoothed IDF formula.
        idf = math.log((1 + total_documents) / (1 + df)) + 1
        inverse_document_frequency[term] = idf

    # Precompute full-document TF-IDF norms for cosine scoring.
    document_norms = {}

    for doc_id, term_counts in document_term_counts.items():
        squared_sum = 0.0

        for term, tf in term_counts.items():
            idf = inverse_document_frequency.get(term, 0.0)
            weight = tf * idf
            squared_sum += weight ** 2

        document_norms[doc_id] = math.sqrt(squared_sum)

    avg_doc_length = sum(document_lengths.values()) / total_documents

    stats = {
        "total_documents": total_documents,
        "average_document_length": avg_doc_length,
        "document_lengths": document_lengths,
        "document_norms": document_norms,
        "document_frequency": document_frequency,
        "inverse_document_frequency": inverse_document_frequency,
    }

    return stats


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    inverted_index, document_lengths, document_term_counts, doc_ids = build_inverted_index(CORPUS_DIR)

    stats = compute_index_statistics(
        inverted_index,
        document_lengths,
        document_term_counts,
        doc_ids
    )

    vocabulary = sorted(inverted_index.keys())

    save_json(inverted_index, INVERTED_INDEX_PATH)
    save_json(vocabulary, VOCABULARY_PATH)
    save_json(stats, INDEX_STATS_PATH)

    print("\nIndex built successfully.")
    print(f"Total documents: {len(doc_ids)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Average document length: {stats['average_document_length']:.2f}")

    print("\nSaved files:")
    print(INVERTED_INDEX_PATH)
    print(VOCABULARY_PATH)
    print(INDEX_STATS_PATH)

    print("\nSample terms:")
    for term in vocabulary[:20]:
        print(term)

    print("\nSample posting for term 'recal' if available:")
    if "recal" in inverted_index:
        sample_postings = list(inverted_index["recal"].items())[:10]
        print(sample_postings)
    else:
        print("Term 'recal' not found.")

    print("\nSample document norm:")
    first_doc = doc_ids[0]
    print(first_doc, stats["document_norms"][first_doc])


if __name__ == "__main__":
    main()