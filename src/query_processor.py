import argparse
from datetime import datetime

from ranker import search, print_results
from prf import run_prf_search
from query_expansion import expand_query
from query_normalization import normalize_search_query
from facets import get_matching_doc_ids, normalize_filters, sort_results


def parse_cli_date(value):
    """
    Parse CLI dates in YYYY-MM-DD format.
    """
    if not value:
        return None

    return datetime.strptime(value, "%Y-%m-%d").date()


def process_query(
    query,
    top_k=10,
    use_expansion=False,
    use_prf=False,
    prf_feedback_docs=3,
    prf_expansion_terms=5,
    filters=None,
    start_date=None,
    end_date=None,
    sort_by="relevance"
):
    """
    Run the full retrieval pipeline.

    Steps:
    - normalize query text (aliases and typo fixes)
    - optionally apply rule-based expansion
    - pre-filter candidates by metadata and date range
    - rank with TF-IDF cosine similarity (optional PRF reranking)
    - sort by relevance or newest recall date
    - return top_k results plus query and feature metadata

    When sort_by is newest, all filtered candidates are scored before
    date sort so newer recalls are not cut off early.
    """
    query_details = normalize_search_query(query)
    corrected_query = query_details["corrected_query"]
    final_query = corrected_query
    expansion_terms = []
    expansion_status_message = ""
    filters = normalize_filters(filters)

    if use_expansion:
        final_query, expansion_terms, expansion_status_message = expand_query(corrected_query)

    candidate_doc_ids = None

    if filters or start_date or end_date:
        candidate_doc_ids = get_matching_doc_ids(
            filters,
            start_date=start_date,
            end_date=end_date
        )

    retrieval_k = top_k

    if sort_by == "newest":
        # Newest sort ranks all filtered candidates before sorting by date.
        retrieval_k = None

    if use_prf:
        search_output = run_prf_search(
            final_query,
            original_query=corrected_query,
            top_k=retrieval_k,
            candidate_doc_ids=candidate_doc_ids,
            feedback_docs=prf_feedback_docs,
            expansion_terms=prf_expansion_terms,
        )
    else:
        search_output = search(
            final_query,
            top_k=retrieval_k,
            candidate_doc_ids=candidate_doc_ids
        )

    results = sort_results(search_output["results"], sort_by=sort_by)
    search_output["results"] = results[:top_k]
    search_output["original_query"] = query
    search_output["corrected_query"] = corrected_query
    search_output["rule_expanded_query"] = final_query
    search_output["expanded_query"] = search_output.get("prf_expanded_query", final_query)
    search_output["expansion_terms"] = expansion_terms
    search_output["expansion_status_message"] = expansion_status_message
    search_output["use_expansion"] = use_expansion
    search_output["use_prf"] = use_prf
    search_output["prf_terms"] = search_output.get("prf_terms", [])
    search_output["prf_doc_ids"] = search_output.get("prf_doc_ids", [])
    search_output["prf_status_message"] = search_output.get("prf_status_message", "")
    search_output["prf_feedback_docs"] = prf_feedback_docs
    search_output["prf_expansion_terms"] = prf_expansion_terms
    search_output["query_corrections"] = query_details["corrections"]
    search_output["highlight_query"] = corrected_query or query
    search_output["filters"] = filters
    search_output["start_date"] = start_date.isoformat() if start_date else ""
    search_output["end_date"] = end_date.isoformat() if end_date else ""
    search_output["sort_by"] = sort_by

    return search_output


def print_query_processing_output(output):
    print("\nOriginal query:", output["original_query"])
    print("Corrected query:", output["corrected_query"])
    print("Use expansion:", output["use_expansion"])
    print("Use PRF:", output["use_prf"])

    if output["query_corrections"]:
        print("Query corrections:", output["query_corrections"])

    if output["use_expansion"]:
        print("Expansion terms:", output["expansion_terms"])
        print("Rule-expanded query:", output["rule_expanded_query"])
        if output["expansion_status_message"]:
            print("Expansion status:", output["expansion_status_message"])

    if output["use_prf"]:
        print("PRF terms:", output["prf_terms"])
        print("Final PRF query:", output["expanded_query"])
        print("PRF feedback docs:", output["prf_feedback_docs"])
        print("PRF expansion terms:", output["prf_expansion_terms"])
        if output["prf_status_message"]:
            print("PRF status:", output["prf_status_message"])

    if output["filters"]:
        print("Filters:", output["filters"])

    if output["start_date"] or output["end_date"]:
        print("Date range:", output["start_date"], "to", output["end_date"])

    print("Sort by:", output["sort_by"])

    print_results(output)


def main():
    parser = argparse.ArgumentParser(description="Search the FDA recall corpus.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--expand", action="store_true", help="Use FDA recall-specific query expansion")
    parser.add_argument("--prf", action="store_true", help="Use pseudo-relevance feedback reranking")
    parser.add_argument("--prf_feedback_docs", type=int, default=3, help="Number of top documents to use for PRF")
    parser.add_argument("--prf_expansion_terms", type=int, default=5, help="Number of terms to add during PRF")
    parser.add_argument("--classification", type=str, default="", help="Filter by recall classification")
    parser.add_argument("--status", type=str, default="", help="Filter by recall status")
    parser.add_argument("--state", type=str, default="", help="Filter by state")
    parser.add_argument("--product_type", type=str, default="", help="Filter by product type")
    parser.add_argument("--start_date", type=str, default="", help="Filter start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="", help="Filter end date (YYYY-MM-DD)")
    parser.add_argument(
        "--sort_by",
        type=str,
        default="relevance",
        choices=["relevance", "newest"],
        help="Sort search results by relevance or newest recall date"
    )

    args = parser.parse_args()

    filters = {
        "classification": args.classification,
        "status": args.status,
        "state": args.state,
        "product_type": args.product_type,
    }

    # Drop empty filter values
    filters = {key: value for key, value in filters.items() if value}

    output = process_query(
        args.query,
        top_k=args.top_k,
        use_expansion=args.expand,
        use_prf=args.prf,
        prf_feedback_docs=args.prf_feedback_docs,
        prf_expansion_terms=args.prf_expansion_terms,
        filters=filters,
        start_date=parse_cli_date(args.start_date),
        end_date=parse_cli_date(args.end_date),
        sort_by=args.sort_by
    )

    print_query_processing_output(output)


if __name__ == "__main__":
    main()