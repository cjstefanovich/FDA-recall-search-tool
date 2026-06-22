import math
from collections import defaultdict

from resources import load_document_vector, load_metadata_lookup, load_search_resources


def more_like_this(doc_id, top_k=10):
    """
    Query-by-example search using the selected document as the query vector.
    """
    metadata_lookup = load_metadata_lookup()
    inverted_index, index_stats, _ = load_search_resources()

    if doc_id not in metadata_lookup.index:
        return []

    example_vector = load_document_vector(doc_id)
    if not example_vector:
        return []

    idf_values = index_stats["inverse_document_frequency"]
    document_norms = index_stats["document_norms"]
    example_norm = math.sqrt(sum(weight ** 2 for weight in example_vector.values()))

    if example_norm == 0:
        return []

    candidate_scores = defaultdict(float)

    # Score only documents that share terms with the example.
    for term, example_weight in example_vector.items():
        postings = inverted_index.get(term, {})
        term_idf = idf_values.get(term)

        if term_idf is None:
            continue

        for candidate_doc_id, term_frequency in postings.items():
            if candidate_doc_id == doc_id:
                continue

            candidate_weight = term_frequency * term_idf
            candidate_scores[candidate_doc_id] += example_weight * candidate_weight

    scores = []

    for candidate_doc_id, dot_product in candidate_scores.items():
        candidate_norm = document_norms.get(candidate_doc_id, 0.0)

        if candidate_norm == 0:
            continue

        score = dot_product / (example_norm * candidate_norm)

        if score > 0:
            scores.append((candidate_doc_id, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    top_scores = scores[:top_k]

    results = []

    for rank, (candidate_doc_id, score) in enumerate(top_scores, start=1):
        if candidate_doc_id not in metadata_lookup.index:
            continue

        row = metadata_lookup.loc[candidate_doc_id]

        results.append({
            "rank": rank,
            "doc_id": candidate_doc_id,
            "score": round(score, 4),
            "recall_number": row["recall_number"],
            "product_description": row["product_description"],
            "reason_for_recall": row["reason_for_recall"],
            "recalling_firm": row["recalling_firm"],
            "classification": row["classification"],
            "status": row["status"],
            "state": row["state"],
            "product_type": row["product_type"],
            "recall_initiation_date": row["recall_initiation_date"],
        })

    return results