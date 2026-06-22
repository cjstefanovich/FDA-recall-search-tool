import math
from collections import Counter, defaultdict

from preprocessor import preprocess_text
from resources import load_metadata_lookup, load_search_resources


def build_query_vector(query, idf_values):
    """
    Build a TF-IDF vector for the query.
    """
    tokens = preprocess_text(query, use_stemming=True)
    query_term_counts = Counter(tokens)

    query_vector = {}

    for term, tf in query_term_counts.items():
        if term in idf_values:
            query_vector[term] = tf * idf_values[term]

    return query_vector, tokens


def compute_vector_norm(vector):
    return math.sqrt(sum(weight ** 2 for weight in vector.values()))


def empty_search_output(query, query_tokens, message):
    return {
        "query": query,
        "query_tokens": query_tokens,
        "results": [],
        "message": message,
    }


def search(query, top_k=10, candidate_doc_ids=None):
    """
    Rank documents with TF-IDF cosine similarity.

    Optional candidate_doc_ids limits scoring to pre-filtered documents.
    """
    inverted_index, index_stats, _ = load_search_resources()
    metadata_lookup = load_metadata_lookup()

    idf_values = index_stats["inverse_document_frequency"]
    document_norms = index_stats["document_norms"]

    query_vector, query_tokens = build_query_vector(query, idf_values)

    if not query_vector:
        return empty_search_output(query, query_tokens, "No query terms found in vocabulary.")

    allowed_doc_ids = None

    if candidate_doc_ids is not None:
        allowed_doc_ids = {str(doc_id) for doc_id in candidate_doc_ids}

        if not allowed_doc_ids:
            return empty_search_output(query, query_tokens, "No documents matched the selected filters.")

    query_norm = compute_vector_norm(query_vector)

    candidate_scores = defaultdict(float)

    # Accumulate dot products over shared index terms.
    for term, query_weight in query_vector.items():
        postings = inverted_index.get(term, {})

        for doc_id, term_frequency in postings.items():
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue

            doc_weight = term_frequency * idf_values[term]
            candidate_scores[doc_id] += query_weight * doc_weight

    ranked_results = []

    for doc_id, dot_product in candidate_scores.items():
        doc_norm = document_norms.get(doc_id, 0.0)

        if doc_norm == 0 or query_norm == 0:
            score = 0.0
        else:
            score = dot_product / (query_norm * doc_norm)

        ranked_results.append((doc_id, score))

    ranked_results.sort(key=lambda x: x[1], reverse=True)

    if top_k is None or top_k <= 0:
        top_results = ranked_results
    else:
        top_results = ranked_results[:top_k]

    output_rows = []

    for rank, (doc_id, score) in enumerate(top_results, start=1):
        if doc_id not in metadata_lookup.index:
            continue

        row = metadata_lookup.loc[doc_id]

        output_rows.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(score, 4),
            "recall_number": row["recall_number"],
            "product_description": row["product_description"],
            "reason_for_recall": row["reason_for_recall"],
            "recalling_firm": row["recalling_firm"],
            "classification": row["classification"],
            "status": row["status"],
            "state": row["state"],
            "city": row["city"],
            "country": row["country"],
            "product_type": row["product_type"],
            "recall_initiation_date": row["recall_initiation_date"],
            "termination_date": row["termination_date"],
            "report_date": row["report_date"],
        })

    return {
        "query": query,
        "query_tokens": query_tokens,
        "results": output_rows,
        "message": "Search completed."
    }


def print_results(search_output):
    print("\nQuery:", search_output["query"])
    print("Processed query tokens:", search_output["query_tokens"])
    print("Message:", search_output["message"])

    results = search_output["results"]

    if not results:
        print("\nNo results found.")
        return

    print("\nTop results:")

    for result in results:
        print("=" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Recall Number: {result['recall_number']}")
        print(f"Product: {result['product_description'][:250]}")
        print(f"Reason: {result['reason_for_recall'][:250]}")
        print(f"Class: {result['classification']}")
        print(f"Status: {result['status']}")
        print(f"State: {result['state']}")
        print(f"Date: {result['recall_initiation_date']}")


def main():
    test_queries = [
        "milk allergy",
        "salmonella contamination",
        "baby formula",
        "metal fragments",
        "lettuce recall",
    ]

    for query in test_queries:
        output = search(query, top_k=5)
        print_results(output)
        print("\n\n")


if __name__ == "__main__":
    main()