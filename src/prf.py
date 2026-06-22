import math
from collections import Counter

from preprocessor import preprocess_text
from ranker import search
from resources import load_document_tokens, load_document_vector


EXCLUDED_PRF_TERMS = {
    "document",
    "recal",
    "number",
    "product",
    "descript",
    "reason",
    "firm",
    "distribut",
    "pattern",
    "classif",
    "status",
    "state",
    "citi",
    "countri",
    "type",
    "voluntari",
    "mandat",
    "initi",
    "notif",
    "code",
    "info",
    "contain",
    "contamin",
    "packag",
    "receiv",
    "supplier",
    "retail",
    "store",
    "north",
    "label",
    "food",
    "class",
    "risk",
    "potenti",
    "account",
    "telephon",
    "custom",
    "plastic",
    "pound",
    "warehous",
    "ingredi",
    "sell",
    "danger",
    "defici",
    "evid",
    "relat",
    "particularli",
    "cross",
    "caramel",
    "cluster",
    "candi",
    "chef",
    "middlefield",
    "purato",
    "termin",
    "statu",
    "unit",
    "human",
    "process",
    "harm",
    "juic",
    "could",
    "case",
    "consum",
    "life",
    "peopl",
    "reaction",
    "seriou",
    "sever",
    "sensit",
    "threaten",
    "agenc",
    "inform",
    "mail",
    "philadelphia",
    "cleveland",
    "awrey",
    "minni",
    "everpress",
}

NO_PRF_QUERIES = {
    "cheese listeria",
    "e coli",
    "foreign material",
    "infant formula",
    "metal fragments",
    "mislabeling",
}

ALLERGY_QUERY_TOKENS = {"allergy", "allergies"}


def normalize_query_text(query):
    """
    Normalize a query string for PRF safety checks.
    """
    cleaned = str(query).lower().strip()
    cleaned = cleaned.replace(".", " ")
    cleaned = cleaned.replace("-", " ")
    return " ".join(cleaned.split())


def get_prf_skip_reason(query):
    """
    Return a user-facing reason when PRF should not run for this query.
    """
    normalized_query = normalize_query_text(query)
    tokens = set(normalized_query.split())

    if normalized_query in NO_PRF_QUERIES:
        return "This query pattern is already specific enough that PRF is more likely to hurt than help."

    if "recall" in tokens:
        return "Recall-focused queries already retrieve specific matches, so PRF is skipped to avoid drift."

    if normalized_query.startswith("undeclared "):
        return "Undeclared-allergen queries are already specific, so PRF is skipped to avoid drift."

    if normalized_query.endswith(" contamination"):
        return "Contamination queries are already specific, so PRF is skipped to avoid drift."

    if normalized_query.endswith(" allergen"):
        return "Allergen queries with an explicit allergen label are already specific, so PRF is skipped."

    if len(tokens) >= 3 and tokens & ALLERGY_QUERY_TOKENS:
        return "This multi-concept allergy query is already narrow enough that PRF is more likely to overfit."

    return ""


def build_prf_fallback_output(
    query,
    top_k,
    candidate_doc_ids,
    reason,
    prf_doc_ids=None,
):
    """
    Return baseline ranking with a PRF skip or rollback message.
    """
    reranked_output = search(
        query,
        top_k=top_k,
        candidate_doc_ids=candidate_doc_ids
    )
    reranked_output["prf_terms"] = []
    reranked_output["prf_doc_ids"] = prf_doc_ids or []
    reranked_output["prf_expanded_query"] = query
    reranked_output["prf_status_message"] = reason
    return reranked_output


def should_keep_prf_rerank(baseline_results, reranked_results):
    """
    Decide whether a PRF rerank is close enough to the baseline to keep.

    Keeps the rerank only when:
    - the baseline rank-1 doc is still in the reranked top 3
    - at least 3 of the baseline top 5 docs appear in the reranked top 5
    """
    if not baseline_results or not reranked_results:
        return False

    baseline_doc_ids = [row["doc_id"] for row in baseline_results[:5]]
    reranked_doc_ids = [row["doc_id"] for row in reranked_results[:5]]

    if baseline_doc_ids[0] not in reranked_doc_ids[:3]:
        return False

    overlap = len(set(baseline_doc_ids) & set(reranked_doc_ids))
    return overlap >= 3


def select_prf_terms(query, feedback_results, max_terms=5):
    """
    Pick PRF expansion terms shared across feedback documents.

    Steps:
    - sum TF-IDF weights for each candidate term across feedback docs
    - skip query terms, short tokens, digits, and EXCLUDED_PRF_TERMS
    - keep terms that appear in at least 67% of feedback docs (minimum 2)
    - return the highest-scoring terms up to max_terms
    """
    query_terms = set(preprocess_text(query, use_stemming=True))
    scored_terms = Counter()
    document_frequency = Counter()

    for result in feedback_results:
        document_vector = load_document_vector(result["doc_id"])
        seen_in_document = set()

        for term, weight in document_vector.items():
            if term in query_terms or term in EXCLUDED_PRF_TERMS:
                continue

            if len(term) < 4 or term.isdigit():
                continue

            scored_terms[term] += weight
            seen_in_document.add(term)

        document_frequency.update(seen_in_document)

    min_shared_docs = max(2, math.ceil(len(feedback_results) * 0.67))
    # Keep terms that appear in most feedback documents.
    shared_terms = [
        term
        for term, doc_count in document_frequency.items()
        if doc_count >= min_shared_docs
    ]

    if shared_terms:
        shared_terms.sort(key=lambda term: scored_terms[term], reverse=True)
        return shared_terms[:max_terms]

    return []


def select_feedback_documents(query, feedback_results, min_documents=2):
    """
    Filter pseudo-relevant docs down to query-coherent feedback.

    Keeps only feedback documents whose preprocessed tokens contain every
    query term. Returns an empty list unless at least min_documents pass.
    """
    query_terms = set(preprocess_text(query, use_stemming=True))

    if not query_terms:
        return []

    coherent_results = []

    for result in feedback_results:
        document_terms = set(load_document_tokens(result["doc_id"]))
        overlap_count = len(query_terms & document_terms)

        # Require full query-term overlap in each feedback document.
        if overlap_count == len(query_terms):
            coherent_results.append(result)

    if len(coherent_results) >= min_documents:
        return coherent_results

    return []


def run_prf_search(
    query,
    original_query=None,
    top_k=10,
    candidate_doc_ids=None,
    feedback_docs=3,
    expansion_terms=5
):
    """
    Run pseudo-relevance feedback with conservative guardrails.

    Steps:
    - score the query and take the top feedback_docs as pseudo-relevant
    - skip PRF when the query pattern is already specific enough
    - keep only feedback docs that contain every query term
    - pick shared high-weight terms from those docs
    - rerank with an expanded query and roll back if the top results shift too much

    Returns the same search output shape as ranker.search, with optional
    prf_terms, prf_doc_ids, prf_expanded_query, and prf_status_message fields.
    """
    baseline_output = search(
        query,
        top_k=feedback_docs,
        candidate_doc_ids=candidate_doc_ids
    )
    feedback_results = baseline_output["results"]
    selection_query = original_query or query

    skip_reason = get_prf_skip_reason(selection_query)

    if skip_reason:
        return build_prf_fallback_output(
            query,
            top_k,
            candidate_doc_ids,
            skip_reason,
        )

    if not feedback_results:
        baseline_output["prf_terms"] = []
        baseline_output["prf_doc_ids"] = []
        baseline_output["prf_expanded_query"] = query
        baseline_output["prf_status_message"] = "PRF could not run because the baseline search returned no feedback documents."
        return baseline_output

    coherent_feedback_results = select_feedback_documents(
        selection_query,
        feedback_results
    )

    if not coherent_feedback_results:
        # Weak feedback docs: keep the baseline ranking.
        return build_prf_fallback_output(
            query,
            top_k,
            candidate_doc_ids,
            "Top results did not agree closely enough, so PRF was skipped to avoid drift."
        )

    prf_terms = select_prf_terms(
        selection_query,
        coherent_feedback_results,
        max_terms=expansion_terms
    )

    if not prf_terms:
        return build_prf_fallback_output(
            query,
            top_k,
            candidate_doc_ids,
            "Feedback documents did not share any strong extra terms, so the original query was kept.",
            prf_doc_ids=[row["doc_id"] for row in coherent_feedback_results],
        )

    expanded_query = query + " " + " ".join(prf_terms)
    baseline_full_output = search(
        query,
        top_k=top_k,
        candidate_doc_ids=candidate_doc_ids
    )

    reranked_output = search(
        expanded_query,
        top_k=top_k,
        candidate_doc_ids=candidate_doc_ids
    )

    if not should_keep_prf_rerank(
        baseline_full_output["results"],
        reranked_output["results"]
    ):
        return build_prf_fallback_output(
            query,
            top_k,
            candidate_doc_ids,
            "PRF expansion changed the ranking too much, so the system fell back to the safer baseline ranking.",
            prf_doc_ids=[row["doc_id"] for row in coherent_feedback_results],
        )

    reranked_output["prf_terms"] = prf_terms
    reranked_output["prf_doc_ids"] = [row["doc_id"] for row in coherent_feedback_results]
    reranked_output["prf_expanded_query"] = expanded_query
    reranked_output["prf_status_message"] = ""

    return reranked_output
