import sys
from pathlib import Path

import pandas as pd


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from query_processor import process_query


QUERIES_PATH = EVALUATION_DIR / "test_queries.csv"
POOL_OUTPUT_PATH = EVALUATION_DIR / "judgment_pool.csv"


def main():
    """
    Export pooled top-10 candidates from all benchmark configs for judging.
    """
    queries_df = pd.read_csv(QUERIES_PATH)
    configs = [
        ("baseline", False, False),
        ("expansion", True, False),
        ("prf", False, True),
        ("full", True, True),
    ]
    pooled_rows = []

    for _, query_row in queries_df.iterrows():
        seen_doc_ids = set()

        for config_name, use_expansion, use_prf in configs:
            output = process_query(
                query=query_row["query_text"],
                top_k=10,
                use_expansion=use_expansion,
                use_prf=use_prf,
            )

            for result in output["results"]:
                doc_id = result["doc_id"]

                if doc_id in seen_doc_ids:
                    continue

                seen_doc_ids.add(doc_id)
                pooled_rows.append({
                    "query_id": query_row["query_id"],
                    "query_text": query_row["query_text"],
                    "config": config_name,
                    "rank": result["rank"],
                    "doc_id": doc_id,
                    "recall_number": result["recall_number"],
                    "classification": result["classification"],
                    "status": result["status"],
                    "state": result["state"],
                    "product_description": result["product_description"],
                    "reason_for_recall": result["reason_for_recall"],
                })

    pool_df = pd.DataFrame(pooled_rows)
    pool_df.to_csv(POOL_OUTPUT_PATH, index=False)

    print("Saved pooled judgment candidates to:", POOL_OUTPUT_PATH)
    print("Total pooled rows:", len(pool_df))


if __name__ == "__main__":
    main()
