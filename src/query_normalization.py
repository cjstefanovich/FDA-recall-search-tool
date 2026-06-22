import difflib
import re


CONTROLLED_QUERY_TERMS = {
    "allergen",
    "allergens",
    "allergy",
    "baby",
    "cheese",
    "coli",
    "contamination",
    "cronobacter",
    "dairy",
    "e",
    "ecoli",
    "escherichia",
    "foreign",
    "formula",
    "fragments",
    "ice",
    "listeria",
    "lettuce",
    "metal",
    "milk",
    "mislabeling",
    "peanut",
    "potato",
    "recall",
    "salad",
    "salmonella",
    "sesame",
    "undeclared",
}

COMMON_MISSPELLINGS = {
    "salmonela": "salmonella",
    "samonella": "salmonella",
    "listera": "listeria",
    "cronobactor": "cronobacter",
    "undeclard": "undeclared",
}

E_COLI_PATTERN = re.compile(
    r"\becoli\b|\be\s*\.?\s*coli\b|\bescherichia\s+coli\b",
    re.IGNORECASE,
)


def normalize_whitespace(text):
    """
    Collapse repeated whitespace.
    """
    return " ".join(str(text).strip().split())


def tokenize_query_text(text):
    """
    Tokenize a short user query into alphanumeric terms.
    """
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def correct_query_spelling(query):
    """
    Fix common recall-search typos against a controlled term list.
    """
    tokens = tokenize_query_text(query)

    if not tokens:
        return "", []

    corrected_tokens = []
    corrections = []

    for token in tokens:
        replacement = token

        if token in COMMON_MISSPELLINGS:
            replacement = COMMON_MISSPELLINGS[token]
        elif len(token) >= 4 and token not in CONTROLLED_QUERY_TERMS:
            matches = difflib.get_close_matches(
                token,
                sorted(CONTROLLED_QUERY_TERMS),
                n=1,
                cutoff=0.88,
            )

            if matches:
                replacement = matches[0]

        corrected_tokens.append(replacement)

        if replacement != token:
            corrections.append({
                "from": token,
                "to": replacement,
            })

    return " ".join(corrected_tokens), corrections


def normalize_query_aliases(query):
    """
    Canonicalize common query variants before typo correction.
    """
    normalized_query = normalize_whitespace(query).lower()
    normalized_query = E_COLI_PATTERN.sub("e coli", normalized_query)
    return normalize_whitespace(normalized_query)


def normalize_search_query(query):
    """
    Return corrected query text plus any applied typo corrections.
    """
    aliased_query = normalize_query_aliases(query)
    corrected_query, corrections = correct_query_spelling(aliased_query)

    if not corrected_query:
        corrected_query = normalize_whitespace(query).lower()

    return {
        "corrected_query": corrected_query,
        "corrections": corrections,
    }
