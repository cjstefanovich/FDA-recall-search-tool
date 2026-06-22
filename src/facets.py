from datetime import datetime, timedelta

from resources import load_search_resources


DATE_FIELD = "recall_initiation_date"
DATE_PRESET_WINDOWS = {
    "Past week": 7,
    "Past month": 30,
    "Past 6 months": 183,
    "Past year": 365,
}


def load_metadata():
    _, _, metadata = load_search_resources()
    return metadata


def normalize_filters(filters):
    """
    Drop empty filter values.
    """
    if not filters:
        return {}

    return {
        field: value
        for field, value in filters.items()
        if value is not None and value != ""
    }


def get_available_facets():
    """
    Return available values for each filter/facet.
    """
    metadata = load_metadata()

    facet_fields = [
        "classification",
        "status",
        "state",
        "product_type",
    ]

    facets = {}

    for field in facet_fields:
        values = sorted([value for value in metadata[field].unique() if value != ""])
        facets[field] = values

    return facets


def parse_recall_date(value):
    """
    Parse FDA dates stored as YYYYMMDD strings.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    if "." in text:
        text = text.split(".")[0]

    digits_only = "".join(char for char in text if char.isdigit())

    if len(digits_only) != 8:
        return None

    try:
        return datetime.strptime(digits_only, "%Y%m%d").date()
    except ValueError:
        return None


def format_recall_date(value):
    """
    Format FDA date strings consistently for the UI.
    """
    parsed_date = parse_recall_date(value)

    if parsed_date is None:
        return ""

    return parsed_date.isoformat()


def get_date_bounds():
    """
    Return the minimum and maximum recall dates in the metadata.
    """
    metadata = load_metadata()
    parsed_dates = metadata[DATE_FIELD].apply(parse_recall_date)
    valid_dates = [date_value for date_value in parsed_dates.tolist() if date_value is not None]

    if not valid_dates:
        return None, None

    return min(valid_dates), max(valid_dates)


def clamp_date_range(start_date, end_date, min_date, max_date):
    """
    Keep a date range inside the dataset bounds.
    """
    if not min_date or not max_date:
        return start_date, end_date

    start_date = max(start_date, min_date)
    end_date = min(end_date, max_date)

    if start_date > end_date:
        start_date = end_date

    return start_date, end_date


def get_preset_date_range(preset_label, min_date, max_date):
    """
    Build a preset date range anchored on the newest recall in the corpus.
    """
    if not min_date or not max_date:
        return None, None

    if preset_label == "Entire dataset":
        return min_date, max_date

    window_days = DATE_PRESET_WINDOWS.get(preset_label)

    if window_days is None:
        return min_date, max_date

    start_date = max_date - timedelta(days=window_days - 1)
    return clamp_date_range(start_date, max_date, min_date, max_date)


def filter_metadata(metadata, filters, start_date=None, end_date=None):
    """
    Filter the metadata table before retrieval.

    Applies facet filters and optional recall_initiation_date bounds so
    ranking can score only matching documents instead of filtering after search.
    """
    normalized_filters = normalize_filters(filters)

    if not normalized_filters and not start_date and not end_date:
        return metadata

    filtered_metadata = metadata

    for field, selected_value in normalized_filters.items():
        if field not in filtered_metadata.columns:
            continue

        filtered_metadata = filtered_metadata[
            filtered_metadata[field].astype(str) == str(selected_value)
        ]

    if start_date or end_date:
        parsed_dates = filtered_metadata[DATE_FIELD].apply(parse_recall_date)
        keep_mask = parsed_dates.apply(lambda value: value is not None)

        if start_date:
            keep_mask = keep_mask & parsed_dates.apply(
                lambda value: value is not None and value >= start_date
            )

        if end_date:
            keep_mask = keep_mask & parsed_dates.apply(
                lambda value: value is not None and value <= end_date
            )

        filtered_metadata = filtered_metadata[keep_mask]

    return filtered_metadata


def get_matching_doc_ids(filters, start_date=None, end_date=None):
    """
    Return doc_ids whose metadata matches the active filters.

    Used by process_query to limit TF-IDF scoring to pre-filtered candidates.
    """
    metadata = load_metadata()
    filtered_metadata = filter_metadata(
        metadata,
        filters,
        start_date=start_date,
        end_date=end_date
    )
    return filtered_metadata["doc_id"].tolist()


def renumber_results(results):
    """
    Re-number ranks after filtering or re-sorting.
    """
    for i, result in enumerate(results, start=1):
        result["rank"] = i

    return results


def sort_results(results, sort_by="relevance"):
    """
    Sort already-retrieved results by relevance or newest-first.
    """
    sorted_results = list(results)

    if sort_by == "newest":
        sorted_results.sort(
            key=lambda result: (
                parse_recall_date(result.get(DATE_FIELD, "")).toordinal()
                if parse_recall_date(result.get(DATE_FIELD, "")) is not None
                else -1
            ),
            reverse=True
        )

    return renumber_results(sorted_results)


def main():
    facets = get_available_facets()

    print("Available facets:")

    for field, values in facets.items():
        print("=" * 80)
        print(field)
        print(values[:20])
        print(f"Total values: {len(values)}")


if __name__ == "__main__":
    main()