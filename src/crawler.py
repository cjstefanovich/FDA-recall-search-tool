import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.fda.gov/food/enforcement.json"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUTPUT_PATH = PROJECT_ROOT / "data/raw/openfda_food_enforcement_raw.json"
CLEAN_OUTPUT_PATH = PROJECT_ROOT / "data/processed/recalls_clean.csv"


FIELDS_TO_KEEP = [
    "recall_number",
    "product_description",
    "reason_for_recall",
    "recalling_firm",
    "distribution_pattern",
    "classification",
    "status",
    "state",
    "city",
    "country",
    "product_type",
    "recall_initiation_date",
    "termination_date",
    "report_date",
    "voluntary_mandated",
    "initial_firm_notification",
    "code_info",
]


def fetch_page(session, url, params=None, max_retries=5):
    """
    Fetch one API page with retries for temporary service failures.
    """
    retryable_status_codes = {429, 500, 502, 503, 504}

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            response = None
            error_message = str(exc)
            status_code = None
        else:
            status_code = response.status_code

            if status_code == 200:
                return response

            error_message = response.text[:500]

        should_retry = (
            attempt < max_retries
            and (status_code is None or status_code in retryable_status_codes)
        )

        if not should_retry:
            raise RuntimeError(
                f"OpenFDA request failed on attempt {attempt}. "
                f"Status code: {status_code}. Response: {error_message}"
            )

        wait_seconds = min(30, 2 ** attempt)
        print(
            f"Temporary request failure on attempt {attempt}. "
            f"Retrying in {wait_seconds} seconds..."
        )
        time.sleep(wait_seconds)


def fetch_openfda_records(max_records=None, batch_size=1000):
    """
    Fetch OpenFDA food enforcement records.

    Uses search_after paging. Optional max_records caps the download.
    """
    all_records = []
    page_number = 1
    next_url = BASE_URL
    params = {
        "limit": batch_size,
        # Stable sort field required for search_after paging.
        "sort": "report_date:asc",
    }

    with requests.Session() as session:
        while True:
            print(f"Fetching page {page_number}...")

            response = fetch_page(session, next_url, params=params)

            data = response.json()
            records = data.get("results", [])
            total_available = data.get("meta", {}).get("results", {}).get("total")

            if page_number == 1 and total_available is not None:
                print(f"OpenFDA reports {total_available} total food enforcement records.")

            if not records:
                print("No more records found.")
                break

            all_records.extend(records)
            print(f"Collected {len(all_records)} total records so far.")

            if max_records is not None and len(all_records) >= max_records:
                all_records = all_records[:max_records]
                print(f"Reached requested limit of {max_records} records.")
                break

            next_url = response.links.get("next", {}).get("url")

            if not next_url:
                print("No next page link found.")
                break

            params = None
            page_number += 1

            # Brief pause between API requests.
            time.sleep(0.2)

    return all_records


def normalize_identifier_part(value):
    """
    Normalize an identifier part so it is stable and file-system friendly.
    """
    if value is None:
        return ""

    text = str(value).strip().upper()

    if not text:
        return ""

    text = re.sub(r"[^A-Z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def build_stable_doc_id(record, used_doc_ids, fallback_number):
    """
    Assign a stable, file-safe doc_id for one recall record.

    Resolution order on collision:
    - recall_number alone
    - recall_number plus report_date
    - recall_number plus numeric suffix (2, 3, ...)
    - FDA-RECALL-{report_date}
    - FDA-RECALL-{fallback_number}
    """
    recall_number = normalize_identifier_part(record.get("recall_number", ""))
    report_date = normalize_identifier_part(record.get("report_date", ""))

    if recall_number and recall_number not in used_doc_ids:
        return recall_number

    if recall_number and report_date:
        candidate = f"{recall_number}-{report_date}"

        if candidate not in used_doc_ids:
            return candidate

    if recall_number:
        suffix = 2

        while True:
            candidate = f"{recall_number}-{suffix}"

            if candidate not in used_doc_ids:
                return candidate

            suffix += 1

    if report_date:
        candidate = f"FDA-RECALL-{report_date}"

        if candidate not in used_doc_ids:
            return candidate

    return f"FDA-RECALL-{fallback_number:05d}"


def clean_records(records):
    """
    Keep only important fields and convert missing values to empty strings.
    """
    cleaned = []
    used_doc_ids = set()

    for i, record in enumerate(records, start=1):
        doc_id = build_stable_doc_id(record, used_doc_ids, i)
        used_doc_ids.add(doc_id)
        row = {"doc_id": doc_id}

        for field in FIELDS_TO_KEEP:
            value = record.get(field, "")

            if value is None:
                value = ""

            row[field] = value

        cleaned.append(row)

    return pd.DataFrame(cleaned).sort_values("doc_id").reset_index(drop=True)


def main():
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = fetch_openfda_records(batch_size=1000)

    print(f"\nTotal records fetched: {len(records)}")

    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    df = clean_records(records)
    df.to_csv(CLEAN_OUTPUT_PATH, index=False)

    print(f"Raw JSON saved to: {RAW_OUTPUT_PATH}")
    print(f"Clean CSV saved to: {CLEAN_OUTPUT_PATH}")
    print("\nClean data shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample records:")
    print(df[["doc_id", "product_description", "reason_for_recall", "classification", "status"]].head())


if __name__ == "__main__":
    main()