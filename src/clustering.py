from collections import defaultdict

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from resources import load_document_vector


KEYWORD_CLUSTERS = [
    {
        "label": "Infant / Baby",
        "keywords": [
            "baby",
            "infant",
            "formula",
            "toddler",
            "cronobacter",
        ],
    },
    {
        "label": "Bacterial Contamination",
        "keywords": [
            "salmonella",
            "listeria",
            "e coli",
            "escherichia coli",
            "bacteria",
            "microbial",
            "pathogen",
            "contamination",
        ],
    },
    {
        "label": "Allergen / Mislabeling",
        "keywords": [
            "allergen",
            "undeclared",
            "mislabel",
            "milk",
            "peanut",
            "tree nut",
            "soy",
            "wheat",
            "sesame",
            "egg",
        ],
    },
    {
        "label": "Foreign Material",
        "keywords": [
            "metal",
            "glass",
            "plastic",
            "foreign material",
            "fragment",
            "piece of",
            "rubber",
        ],
    },
    {
        "label": "Chemical / Residue",
        "keywords": [
            "chemical",
            "pesticide",
            "herbicide",
            "residue",
            "lead",
            "arsenic",
        ],
    },
]

CLUSTER_TERM_STOPWORDS = {
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
    "food",
    "term",
    "date",
}


def build_cluster_text(result):
    """
    Combine product, reason, and class fields for keyword matching.
    """
    parts = [
        str(result.get("product_description", "")),
        str(result.get("reason_for_recall", "")),
        str(result.get("classification", "")),
    ]
    return " ".join(parts).lower()


def assign_keyword_cluster(result):
    """
    Assign one result to the first matching keyword bucket.
    """
    text = build_cluster_text(result)

    for cluster in KEYWORD_CLUSTERS:
        matched_terms = [
            keyword
            for keyword in cluster["keywords"]
            if keyword in text
        ]

        if matched_terms:
            return cluster["label"], matched_terms

    return "Other", []


def build_keyword_clusters(results):
    """
    Group results into simple deterministic keyword buckets.
    """
    grouped = {}

    for result in results:
        label, matched_terms = assign_keyword_cluster(result)

        if label not in grouped:
            grouped[label] = {
                "label": label,
                "size": 0,
                "term_label": "Matched terms",
                "match_terms": set(),
                "results": [],
            }

        grouped[label]["size"] += 1
        grouped[label]["match_terms"].update(matched_terms)
        grouped[label]["results"].append(result)

    clusters = []

    for cluster in grouped.values():
        cluster["results"].sort(key=lambda result: result["rank"])
        clusters.append({
            "label": cluster["label"],
            "size": cluster["size"],
            "term_label": cluster["term_label"],
            "match_terms": sorted(cluster["match_terms"])[:5],
            "results": cluster["results"],
        })

    clusters.sort(key=lambda cluster: (-cluster["size"], cluster["label"]))
    return clusters


def build_kmeans_matrix(results):
    """
    Build a sparse TF-IDF matrix for the retrieved results only.
    """
    feature_index = {}
    rows = []
    cols = []
    data = []

    for row_index, result in enumerate(results):
        document_vector = load_document_vector(result["doc_id"])

        for term, weight in document_vector.items():
            if term not in feature_index:
                feature_index[term] = len(feature_index)

            rows.append(row_index)
            cols.append(feature_index[term])
            data.append(weight)

    if not feature_index:
        return None, []

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(results), len(feature_index))
    )
    feature_names = [None] * len(feature_index)

    for term, index in feature_index.items():
        feature_names[index] = term

    return matrix, feature_names


def select_representative_terms(centroid, feature_names, max_terms=5):
    """
    Choose readable cluster terms from the centroid weights.
    """
    scored_terms = []

    for feature_index, weight in enumerate(centroid):
        term = feature_names[feature_index]

        if term in CLUSTER_TERM_STOPWORDS:
            continue

        if len(term) < 4 or term.isdigit():
            continue

        scored_terms.append((term, weight))

    scored_terms.sort(key=lambda item: item[1], reverse=True)
    return [term for term, _ in scored_terms[:max_terms]]


def build_kmeans_clusters(results, n_clusters=5):
    """
    Cluster result TF-IDF vectors with k-means.
    """
    if len(results) < 2:
        return [{
            "label": "Cluster 1",
            "size": len(results),
            "term_label": "Representative terms",
            "match_terms": [],
            "results": list(results),
        }]

    matrix, feature_names = build_kmeans_matrix(results)

    if matrix is None or matrix.shape[1] == 0:
        return build_keyword_clusters(results)

    cluster_count = max(2, min(n_clusters, len(results)))
    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(matrix)
    grouped = defaultdict(list)

    for result, label in zip(results, labels):
        grouped[int(label)].append(result)

    clusters = []

    for cluster_index, cluster_results_list in grouped.items():
        cluster_results_list.sort(key=lambda result: result["rank"])
        # Top centroid terms label this cluster.
        representative_terms = select_representative_terms(
            model.cluster_centers_[cluster_index],
            feature_names
        )
        clusters.append({
            "label": "Cluster",
            "size": len(cluster_results_list),
            "term_label": "Representative terms",
            "match_terms": representative_terms,
            "results": cluster_results_list,
        })

    clusters.sort(key=lambda cluster: (-cluster["size"], cluster["label"]))

    for cluster_number, cluster in enumerate(clusters, start=1):
        cluster["label"] = f"Cluster {cluster_number}"

    return clusters


def cluster_results(results, method="keyword", n_clusters=5):
    """
    Group search results into topical buckets.

    keyword uses rule-based buckets. kmeans clusters TF-IDF result vectors.
    """
    if not results:
        return []

    if method == "keyword":
        return build_keyword_clusters(results)

    if method == "kmeans":
        return build_kmeans_clusters(results, n_clusters=n_clusters)

    raise ValueError(f"Unsupported clustering method: {method}")
