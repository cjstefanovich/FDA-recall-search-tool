import re

from query_normalization import normalize_search_query


EXPANSION_RULES = {
    "allergy": [
        "allergen",
        "undeclared allergen",
        "mislabeling",
    ],
    "allergies": [
        "allergen",
        "undeclared allergen",
        "mislabeling",
    ],
    "milk": [
        "milk allergen",
        "undeclared milk",
        "dairy allergen",
        "undeclared dairy",
    ],
    "peanut": [
        "peanut allergen",
        "undeclared peanut",
        "peanut protein",
    ],
    "formula": [
        "infant formula",
        "powdered infant formula",
        "cronobacter",
        "formula recall",
    ],
    "salmonella": [
        "salmonella contamination",
        "potential salmonella",
    ],
    "listeria": [
        "listeria monocytogenes",
    ],
    "metal": [
        "foreign material",
        "metal fragments",
        "wire mesh",
        "stainless steel",
    ],
    "fragments": [
        "foreign material",
        "metal fragments",
        "plastic fragments",
        "wire mesh",
    ],
    "lettuce": [
        "romaine lettuce",
        "iceberg lettuce",
        "shredded lettuce",
    ],
}

PHRASE_EXPANSION_RULES = {
    "baby formula": [
        "infant formula",
        "powdered infant formula",
        "cronobacter",
        "formula recall",
    ],
    "infant formula": [
        "powdered infant formula",
        "cronobacter",
        "formula recall",
    ],
    "peanut butter salmonella": [
        "peanut butter contaminated with salmonella",
        "salmonella peanut butter",
        "contaminated peanut butter",
    ],
    "ice cream listeria": [
        "listeria monocytogenes",
    ],
    "cheese listeria": [
        "listeria monocytogenes",
        "cheese recall",
    ],
    "lettuce recall": [
        "romaine lettuce",
        "iceberg lettuce",
        "shredded lettuce",
    ],
}

NO_EXPANSION_QUERIES = {
    "e coli",
    "foreign material",
    "ice cream listeria",
    "infant formula",
    "lettuce recall",
    "listeria",
    "metal fragments",
    "milk allergy",
    "peanut allergen",
    "peanut butter salmonella",
    "sesame allergen",
    "undeclared milk",
}

MAX_EXPANSION_TERMS = 4


def normalize_key(term):
    """
    Normalize a query word so it can match expansion rules.
    """
    term = term.lower().strip()
    term = term.replace(".", "")
    term = term.replace("-", "")
    return term


def normalize_query_text(query):
    """
    Normalize the full query for phrase matching.
    """
    cleaned = str(query).lower().strip()
    cleaned = cleaned.replace(".", " ")
    cleaned = cleaned.replace("-", " ")
    return " ".join(cleaned.split())


def tokenize_variant_text(text):
    """
    Tokenize short display phrases for readable suggestion merging.
    """
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def build_suggestion_variant(query, term):
    """
    Merge query and expansion phrase without repeating tokens.
    """
    query_text = " ".join(str(query).split())
    term_text = " ".join(str(term).split())
    query_tokens = tokenize_variant_text(query_text)
    term_tokens = tokenize_variant_text(term_text)

    if not term_tokens:
        return query_text

    if set(query_tokens) & set(term_tokens):
        return term_text

    return f"{query_text} {term_text}".strip()


def build_expanded_query(query, expansion_terms):
    """
    Merge expansion terms into the query string.

    Only trims redundant overlap at token boundaries between the current
    query and each new term. Full phrases like "undeclared milk" stay intact.
    """
    combined_tokens = tokenize_variant_text(query)

    for term in expansion_terms:
        term_tokens = tokenize_variant_text(term)

        if not term_tokens:
            continue

        overlap = 0
        max_overlap = min(len(combined_tokens), len(term_tokens))

        for candidate_overlap in range(max_overlap, 0, -1):
            if combined_tokens[-candidate_overlap:] == term_tokens[:candidate_overlap]:
                overlap = candidate_overlap
                break

        combined_tokens.extend(term_tokens[overlap:])

    return " ".join(combined_tokens)


def should_skip_expansion(normalized_query):
    """
    Skip expansion for queries that are already specific enough.
    """
    if normalized_query in NO_EXPANSION_QUERIES:
        return True

    if normalized_query.startswith("undeclared "):
        return True

    if normalized_query.endswith(" contamination"):
        return True

    return False


def get_expansion_status_message(query, expansion_terms):
    """
    User-facing note on why expansion ran or was skipped.
    """
    normalized_query = normalize_query_text(query)

    if expansion_terms:
        return ""

    if should_skip_expansion(normalized_query):
        return "This query is already specific enough that rule expansion was skipped to avoid drift."

    return "No rule-based expansion pattern matched this query. Expansion only adds terms for a small set of FDA-specific query patterns."


def expand_query(query):
    """
    Apply FDA rule-based query expansion.

    Steps:
    - skip queries that are already specific enough
    - match phrase rules first, then single-token rules
    - dedupe terms and cap at MAX_EXPANSION_TERMS
    - merge terms into the query with boundary overlap trimming

    Returns (expanded_query, added_terms, status_message).
    """
    original_terms = query.split()
    expansion_terms = []
    normalized_query = normalize_query_text(query)
    covered_tokens = set()

    if should_skip_expansion(normalized_query):
        return query, [], get_expansion_status_message(query, [])

    for phrase, phrase_expansions in PHRASE_EXPANSION_RULES.items():
        if phrase in normalized_query:
            expansion_terms.extend(phrase_expansions)
            covered_tokens.update(phrase.split())

    for term in original_terms:
        key = normalize_key(term)

        if key in covered_tokens:
            continue

        if key in EXPANSION_RULES:
            expansion_terms.extend(EXPANSION_RULES[key])

    # Dedupe while preserving order
    seen = set()
    unique_expansions = []

    for term in expansion_terms:
        if term not in seen:
            unique_expansions.append(term)
            seen.add(term)

    unique_expansions = unique_expansions[:MAX_EXPANSION_TERMS]

    expanded_query = query

    if unique_expansions:
        expanded_query = build_expanded_query(query, unique_expansions)

    return expanded_query, unique_expansions, get_expansion_status_message(query, unique_expansions)


def get_query_suggestions(query):
    """
    Return expansion-based suggestions using the same normalization as retrieval.
    """
    corrected_query = normalize_search_query(query)["corrected_query"]
    expanded_query, expansion_terms, _ = expand_query(corrected_query)

    suggestions = []
    seen = set()

    for term in expansion_terms[:5]:
        suggestion = build_suggestion_variant(corrected_query, term)
        normalized_suggestion = normalize_query_text(suggestion)

        if normalized_suggestion in seen:
            continue

        seen.add(normalized_suggestion)
        suggestions.append(suggestion)

    return suggestions


def main():
    test_queries = [
        "baby formula",
        "milk allergy",
        "metal fragments",
        "lettuce recall",
        "salmonella contamination",
    ]

    for query in test_queries:
        expanded_query, terms, status_message = expand_query(query)
        suggestions = get_query_suggestions(query)

        print("=" * 80)
        print("Original query:", query)
        print("Expansion terms:", terms)
        print("Expanded query:", expanded_query)
        print("Status:", status_message)
        print("Suggestions:", suggestions)


if __name__ == "__main__":
    main()