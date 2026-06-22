import math
import statistics
import time


def precision_at_k(ranked_doc_ids, relevant_doc_ids, k):
    """
    Compute Precision@k for one query.
    """
    if k <= 0:
        return 0.0

    top_docs = ranked_doc_ids[:k]

    if not top_docs:
        return 0.0

    relevant_found = sum(1 for doc_id in top_docs if doc_id in relevant_doc_ids)
    return relevant_found / k


def recall_at_k(ranked_doc_ids, relevant_doc_ids, k):
    """
    Compute Recall@k for one query.
    """
    if not relevant_doc_ids:
        return 0.0

    top_docs = ranked_doc_ids[:k]
    relevant_found = sum(1 for doc_id in top_docs if doc_id in relevant_doc_ids)
    return relevant_found / len(relevant_doc_ids)


def average_precision(ranked_doc_ids, relevant_doc_ids):
    """
    Compute Average Precision for one query.

    AP is averaged over the length of ranked_doc_ids, not the full corpus.
    When the caller passes only top_k doc ids, this is AP@k (AP@10 in the
    benchmark when top_k is 10).
    """
    if not relevant_doc_ids:
        return 0.0

    precision_sum = 0.0
    relevant_seen = 0

    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    if relevant_seen == 0:
        return 0.0

    return precision_sum / len(relevant_doc_ids)


def dcg_at_k(ranked_doc_ids, relevance_lookup, k):
    """
    Compute discounted cumulative gain for the top k results.
    """
    dcg = 0.0

    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        relevance = relevance_lookup.get(doc_id, 0)

        if relevance <= 0:
            continue

        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)
        dcg += gain / discount

    return dcg


def ndcg_at_k(ranked_doc_ids, relevance_lookup, k):
    """
    Compute NDCG@k for one query.
    """
    ideal_relevances = sorted(relevance_lookup.values(), reverse=True)

    if not ideal_relevances:
        return 0.0

    ideal_dcg = 0.0

    for rank, relevance in enumerate(ideal_relevances[:k], start=1):
        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)
        ideal_dcg += gain / discount

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(ranked_doc_ids, relevance_lookup, k) / ideal_dcg


def build_relevance_lookup(judgments_df):
    """
    Build a nested lookup keyed by query_id then doc_id.
    """
    lookup = {}

    for _, row in judgments_df.iterrows():
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        relevance = int(row["relevance"])

        if query_id not in lookup:
            lookup[query_id] = {}

        lookup[query_id][doc_id] = relevance

    return lookup


def evaluate_query(ranked_doc_ids, relevance_lookup, k_values=(5, 10)):
    """
    Compute the main ranking metrics for one query.
    """
    relevant_doc_ids = {
        doc_id
        for doc_id, relevance in relevance_lookup.items()
        if relevance > 0
    }

    metrics = {
        "ap": average_precision(ranked_doc_ids, relevant_doc_ids),
    }

    for k in k_values:
        metrics[f"p@{k}"] = precision_at_k(ranked_doc_ids, relevant_doc_ids, k)
        metrics[f"r@{k}"] = recall_at_k(ranked_doc_ids, relevant_doc_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_doc_ids, relevance_lookup, k)

    return metrics


def summarize_query_metrics(query_metrics):
    """
    Average per-query metrics into one run summary.
    """
    if not query_metrics:
        return {}

    metric_names = query_metrics[0].keys()
    summary = {}

    for metric_name in metric_names:
        summary[metric_name] = statistics.mean(
            metrics[metric_name]
            for metrics in query_metrics
        )

    return summary


def measure_latency(query_fn, queries):
    """
    Measure average and p95 latency in milliseconds.
    """
    latencies_ms = []

    for query_text in queries:
        start_time = time.perf_counter()
        query_fn(query_text)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies_ms.append(elapsed_ms)

    if not latencies_ms:
        return {
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)

    return {
        "avg_latency_ms": statistics.mean(sorted_latencies),
        "p95_latency_ms": sorted_latencies[p95_index],
    }


def run_config(
    config_name,
    queries_df,
    relevance_lookup,
    process_query_fn,
    use_expansion=False,
    use_prf=False,
    prf_feedback_docs=3,
    prf_expansion_terms=5,
    top_k=10
):
    """
    Run one retrieval configuration across the benchmark queries.

    Steps:
    - run process_query_fn on each query with the given expansion and PRF flags
    - compute P@5, R@10, AP, and NDCG@10 from the top_k ranked doc ids
    - run a second latency pass over the same queries

    top_k sets both retrieval depth and AP depth (AP@10 when top_k is 10).

    Returns (summary_row, per_query_rows).
    """
    query_metrics = []
    per_query_rows = []

    for _, row in queries_df.iterrows():
        query_id = row["query_id"]
        query_text = row["query_text"]

        output = process_query_fn(
            query=query_text,
            top_k=top_k,
            use_expansion=use_expansion,
            use_prf=use_prf,
            prf_feedback_docs=prf_feedback_docs,
            prf_expansion_terms=prf_expansion_terms,
        )
        ranked_doc_ids = [result["doc_id"] for result in output["results"]]
        metrics = evaluate_query(
            ranked_doc_ids,
            relevance_lookup.get(query_id, {}),
            k_values=(5, 10)
        )
        query_metrics.append(metrics)

        per_query_row = {
            "config": config_name,
            "query_id": query_id,
            "query_text": query_text,
        }
        per_query_row.update(metrics)
        per_query_rows.append(per_query_row)

    summary = summarize_query_metrics(query_metrics)
    latency = measure_latency(
        lambda query_text: process_query_fn(
            query=query_text,
            top_k=top_k,
            use_expansion=use_expansion,
            use_prf=use_prf,
            prf_feedback_docs=prf_feedback_docs,
            prf_expansion_terms=prf_expansion_terms,
        ),
        queries_df["query_text"].tolist()
    )

    summary_row = {
        "config": config_name,
        "prf_feedback_docs": prf_feedback_docs,
        "prf_expansion_terms": prf_expansion_terms,
    }
    summary_row.update(summary)
    summary_row.update(latency)

    return summary_row, per_query_rows
