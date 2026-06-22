from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data/processed/recalls_clean.csv"
CORPUS_DIR = PROJECT_ROOT / "data/corpus"
METADATA_OUTPUT_PATH = PROJECT_ROOT / "index/doc_metadata.csv"


TEXT_FIELDS = [
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
    "voluntary_mandated",
    "initial_firm_notification",
    "code_info",
]


METADATA_FIELDS = [
    "doc_id",
    "recall_number",
    "product_description",
    "reason_for_recall",
    "recalling_firm",
    "classification",
    "status",
    "state",
    "city",
    "country",
    "product_type",
    "recall_initiation_date",
    "termination_date",
    "report_date",
]


FIELD_WEIGHTS = {
    "reason_for_recall": 4,
    "product_description": 3,
    "classification": 2,
    "status": 1,
    "recalling_firm": 1,
    "distribution_pattern": 1,
    "state": 1,
    "city": 1,
    "country": 1,
    "product_type": 1,
    "voluntary_mandated": 1,
    "initial_firm_notification": 1,
    "code_info": 1,
}


def safe_text(value):
    """
    Convert missing values to empty strings.
    """
    if pd.isna(value):
        return ""
    return str(value)


def build_document_text(row):
    """
    Build one weighted corpus document from recall fields.

    High-signal fields are repeated using FIELD_WEIGHTS.
    """
    lines = []

    lines.append(f"Document ID: {safe_text(row['doc_id'])}")
    lines.append(f"Recall Number: {safe_text(row['recall_number'])}")
    lines.append("")

    for field in TEXT_FIELDS:
        field_label = field.replace("_", " ").title()
        field_value = safe_text(row[field])
        weight = FIELD_WEIGHTS.get(field, 1)

        for _ in range(weight):
            lines.append(f"{field_label}: {field_value}")

    return "\n".join(lines)


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Clear old corpus files before rebuild.
    for existing_file in CORPUS_DIR.glob("*.txt"):
        existing_file.unlink()

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} recall records.")

    metadata_rows = []

    for _, row in df.iterrows():
        doc_id = row["doc_id"]
        document_text = build_document_text(row)

        output_file = CORPUS_DIR / f"{doc_id}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(document_text)

        metadata_rows.append(row[METADATA_FIELDS].to_dict())

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(METADATA_OUTPUT_PATH, index=False)

    print(f"Created {len(df)} weighted text documents in: {CORPUS_DIR}")
    print(f"Saved metadata to: {METADATA_OUTPUT_PATH}")

    sample_file = CORPUS_DIR / f"{df.iloc[0]['doc_id']}.txt"
    print("\nSample document path:")
    print(sample_file)

    print("\nSample document preview:")
    with open(sample_file, "r", encoding="utf-8") as f:
        print(f.read()[:1500])


if __name__ == "__main__":
    main()