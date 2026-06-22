import sys
from datetime import date
from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st

# files inside src/
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from query_processor import process_query
from facets import (
    format_recall_date,
    get_available_facets,
    get_date_bounds,
    get_preset_date_range,
)
from query_expansion import get_query_suggestions
from feedback import more_like_this
from clustering import cluster_results


st.set_page_config(
    page_title="FDA Recall Search Assistant",
    page_icon="🔎",
    layout="wide"
)

st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #2e7d32;
        color: white;
        border: 1px solid #2e7d32;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #256628;
        border-color: #256628;
        color: white;
    }
    div[data-testid="stButton"] > button[kind="tertiary"] {
        border: 1px solid #d9d9d9;
        border-radius: 0.5rem;
        padding: 0.25rem 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Session state
# -----------------------------

if "search_output" not in st.session_state:
    st.session_state.search_output = None

if "more_like_results" not in st.session_state:
    st.session_state.more_like_results = []

if "more_like_source" not in st.session_state:
    st.session_state.more_like_source = None

if "selected_more_like_doc" not in st.session_state:
    st.session_state.selected_more_like_doc = None

if "more_like_loading_doc" not in st.session_state:
    st.session_state.more_like_loading_doc = None

if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0

if "search_requested" not in st.session_state:
    st.session_state.search_requested = False

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


# -----------------------------
# Helper functions
# -----------------------------

def clean_title(text, max_chars=90):
    """
    Shorten long product descriptions for cleaner card titles.
    """
    if not text:
        return "No product description available"

    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def make_safe_key(text):
    """
    Convert labels into Streamlit-safe key fragments.
    """
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_").lower()


def display_metadata_row(item):
    """
    Display common recall metadata in four columns.
    """
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

    with meta_col1:
        st.write(f"**Class:** {item['classification']}")

    with meta_col2:
        st.write(f"**Status:** {item['status']}")

    with meta_col3:
        st.write(f"**State:** {item['state']}")

    with meta_col4:
        st.write(f"**Date:** {format_recall_date(item['recall_initiation_date'])}")


def build_highlighted_html(text, query_text):
    """
    Highlight original query terms inside result text.
    """
    if not text:
        return ""

    highlighted = html.escape(str(text))
    normalized_query = " ".join(str(query_text).lower().split())

    if normalized_query == "e coli":
        highlighted = re.sub(
            r"\bE\s*\.?\s*coli\b",
            lambda match: f"<mark>{match.group(0)}</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )

    query_terms = [
        term
        for term in str(query_text).split()
        if len(term) >= 3
    ]

    if not query_terms:
        return highlighted

    pattern = re.compile(
        "(" + "|".join(re.escape(term) for term in sorted(set(query_terms), key=len, reverse=True)) + ")",
        re.IGNORECASE
    )
    return pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", highlighted)


def mark_search_requested():
    """
    Let Enter in the text input trigger a search rerun.
    """
    if st.session_state.get("query_input", "").strip():
        st.session_state.search_requested = True


def reset_more_like_state():
    """
    Clear the query-by-example UI state.
    """
    st.session_state.more_like_results = []
    st.session_state.more_like_source = None
    st.session_state.selected_more_like_doc = None
    st.session_state.more_like_loading_doc = None
    st.session_state.carousel_index = 0


def get_result_by_doc_id(results, doc_id):
    """
    Return one result dictionary from the current result set by doc_id.
    """
    for result in results:
        if result["doc_id"] == doc_id:
            return result
    return None


def get_more_like_view_state():
    """
    Determine whether the app should show the dedicated similar-recalls view.
    """
    output = st.session_state.search_output
    selected_doc_id = st.session_state.selected_more_like_doc

    if output is None or not selected_doc_id:
        return False, None, None

    source_result = get_result_by_doc_id(output.get("results", []), selected_doc_id)

    if source_result is None:
        return False, output, None

    is_active = (
        st.session_state.more_like_loading_doc == selected_doc_id
        or (
            st.session_state.more_like_results
            and st.session_state.more_like_source == selected_doc_id
        )
    )
    return is_active, output, source_result


def format_query_corrections(corrections):
    """
    Format query corrections for user-facing feedback.
    """
    return ", ".join(
        f"{item['from']} -> {item['to']}"
        for item in corrections
    )


def display_result_analytics(results):
    """
    Show lightweight result-set summaries to support exploratory browsing.
    """
    results_df = pd.DataFrame(results)

    if results_df.empty:
        st.caption("No analytics available for an empty result set.")
        return

    results_df["classification"] = results_df["classification"].replace("", "Unknown")
    results_df["state"] = results_df["state"].replace("", "Unknown")
    results_df["recalling_firm"] = results_df["recalling_firm"].replace("", "Unknown")

    recall_dates = pd.to_datetime(
        results_df["recall_initiation_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
    monthly_counts = (
        recall_dates.dropna()
        .dt.to_period("M")
        .astype(str)
        .value_counts()
        .sort_index()
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Recall trend over time**")

        if monthly_counts.empty:
            st.caption("No valid recall dates are available for the time-series chart.")
        else:
            st.line_chart(monthly_counts)

    with chart_col2:
        st.markdown("**By classification**")
        st.bar_chart(results_df["classification"].value_counts())

    chart_col3, chart_col4, chart_col5 = st.columns(3)

    with chart_col3:
        st.markdown("**By state**")
        st.bar_chart(results_df["state"].value_counts().head(10))

    with chart_col4:
        st.markdown("**By status**")
        st.bar_chart(results_df["status"].replace("", "Unknown").value_counts())

    with chart_col5:
        st.markdown("**By recalling firm**")
        st.bar_chart(results_df["recalling_firm"].value_counts().head(10))


def display_more_like_carousel(source_doc_id, source_product):
    """
    Display one similar recall at a time with Previous/Next buttons.
    """
    results = st.session_state.more_like_results

    if not results:
        return

    total = len(results)

    # Keep index safe
    if st.session_state.carousel_index >= total:
        st.session_state.carousel_index = 0

    current_index = st.session_state.carousel_index
    current = results[current_index]

    st.markdown("#### Similar recalls")
    st.caption(
        f"Query-by-example results for **{source_doc_id}**: "
        f"{clean_title(source_product, max_chars=100)}"
    )

    nav_col1, nav_col2, nav_col3, nav_spacer = st.columns([1.2, 1.8, 1.2, 5.8])

    with nav_col1:
        if st.button(
            "← Previous",
            key=f"prev_{source_doc_id}",
            type="tertiary",
            disabled=(total <= 1)
        ):
            st.session_state.carousel_index = (
                st.session_state.carousel_index - 1
            ) % total
            st.rerun()

    with nav_col2:
        st.markdown(
            f"<div style='text-align:left; font-weight:600; padding-top:0.35rem;'>"
            f"Similar result {current_index + 1} of {total}"
            f"</div>",
            unsafe_allow_html=True
        )

    with nav_col3:
        if st.button(
            "Next →",
            key=f"next_{source_doc_id}",
            type="tertiary",
            disabled=(total <= 1)
        ):
            st.session_state.carousel_index = (
                st.session_state.carousel_index + 1
            ) % total
            st.rerun()

    with st.container(border=True):
        st.markdown(
            "<h3>"
            f"<span style='color:#c62828; font-weight:700;'>Similar Recall #{current['rank']}:</span> "
            f"{html.escape(clean_title(current['product_description']))}"
            "</h3>",
            unsafe_allow_html=True
        )
        st.caption(
            f"Doc ID: {current['doc_id']} | "
            f"Recall Number: {current['recall_number']}"
        )

        st.markdown(
            f"<div><span style='color:#c62828; font-weight:700;'>Similarity Score:</span> "
            f"{current['score']}</div>",
            unsafe_allow_html=True
        )

        display_metadata_row(current)

        st.markdown("**Reason for Recall:**")
        st.write(current["reason_for_recall"])

        with st.expander("View full product description"):
            st.write(current["product_description"])

        with st.expander("View additional details"):
            st.write(f"**Firm:** {current['recalling_firm']}")
            st.write(f"**Recall Number:** {current['recall_number']}")
            st.write(f"**Doc ID:** {current['doc_id']}")


def display_result_card(result, button_prefix, query_text="", show_more_like_button=True):
    """
    Render one recall result card.
    """
    with st.container(border=True):
        st.markdown(
            "<h3>"
            f"<span style='color:#c62828; font-weight:700;'>Rank {result['rank']}:</span> "
            f"{html.escape(clean_title(result['product_description']))}"
            "</h3>",
            unsafe_allow_html=True
        )
        st.caption(
            f"Doc ID: {result['doc_id']} | "
            f"Recall Number: {result['recall_number']}"
        )

        st.markdown(
            f"<div><span style='color:#c62828; font-weight:700;'>Score:</span> "
            f"{result['score']}</div>",
            unsafe_allow_html=True
        )

        display_metadata_row(result)

        st.markdown("**Reason for Recall:**")
        st.markdown(
            f"<div>{build_highlighted_html(result['reason_for_recall'], query_text)}</div>",
            unsafe_allow_html=True
        )

        with st.expander("View full product description"):
            st.markdown(
                f"<div>{build_highlighted_html(result['product_description'], query_text)}</div>",
                unsafe_allow_html=True
            )

        with st.expander("View additional details"):
            st.write(f"**Firm:** {result['recalling_firm']}")
            st.write(f"**Recall Number:** {result['recall_number']}")
            st.write(f"**Doc ID:** {result['doc_id']}")

        if show_more_like_button and st.button(
            "More like this",
            key=f"{button_prefix}_more_like_{result['doc_id']}",
            type="secondary"
        ):
            st.session_state.selected_more_like_doc = result["doc_id"]
            st.session_state.more_like_source = result["doc_id"]
            st.session_state.more_like_loading_doc = result["doc_id"]
            st.session_state.more_like_results = []
            st.session_state.carousel_index = 0
            st.rerun()


# -----------------------------
# Sidebar filters/options
# -----------------------------

active_more_like_view, active_output, active_source_result = get_more_like_view_state()

if active_more_like_view:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("FDA Recall Search Assistant")
    st.write(
        "Search a local corpus of FDA food recall records using ranked retrieval, "
        "query expansion, faceted filtering, and query-by-example."
    )

use_expansion = False
use_prf = False
top_k = st.session_state.get("top_k", 10)
cluster_method_label = st.session_state.get("cluster_method_label", "Keyword")
cluster_method = "kmeans" if cluster_method_label == "K-means" else "keyword"
cluster_count = st.session_state.get("cluster_count", 5)
sort_label = st.session_state.get("sort_label", "Relevance")
sort_by = "newest" if sort_label == "Newest first" else "relevance"


# Load available facets
facets = get_available_facets()
min_date, max_date = get_date_bounds()


def reset_search_state():
    """
    Restore the search UI to its default state.
    """
    reset_more_like_state()
    st.session_state.search_output = None
    st.session_state.search_requested = False
    st.session_state.query_input = ""
    st.session_state.use_expansion = False
    st.session_state.use_prf = False
    st.session_state.top_k = 10
    st.session_state.cluster_method_label = "Keyword"
    st.session_state.cluster_count = 5
    st.session_state.sort_label = "Relevance"
    st.session_state.selected_classification = ""
    st.session_state.selected_status = ""
    st.session_state.selected_state = ""
    st.session_state.use_date_filter = False
    st.session_state.date_preset = "Entire dataset"
    if min_date and max_date:
        st.session_state.custom_date_range = (min_date, max_date)


classification_options = [""] + facets.get("classification", [])
status_options = [""] + facets.get("status", [])
state_options = [""] + facets.get("state", [])
selected_classification = ""
selected_status = ""
selected_state = ""
use_date_filter = False

start_date = None
end_date = None

if not active_more_like_view:
    st.sidebar.header("Search Options")

    use_expansion = st.sidebar.checkbox(
        "Use query expansion",
        value=False,
        help="Adds FDA recall-specific terms to vague queries.",
        key="use_expansion"
    )

    use_prf = st.sidebar.checkbox(
        "Use pseudo-relevance feedback",
        value=False,
        help="Expands the query with strong terms from the top retrieved recall records.",
        key="use_prf"
    )

    top_k = st.sidebar.slider(
        "Number of results",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        key="top_k"
    )

    cluster_method_label = st.sidebar.selectbox(
        "Cluster method",
        ["Keyword", "K-means"],
        help="Keyword clusters are simpler and more explainable. K-means clusters result vectors automatically.",
        key="cluster_method_label"
    )
    cluster_method = "kmeans" if cluster_method_label == "K-means" else "keyword"
    cluster_count = 5

    if cluster_method == "kmeans":
        cluster_count = st.sidebar.slider(
            "K-means clusters",
            min_value=2,
            max_value=min(8, top_k),
            value=min(5, top_k),
            step=1,
            key="cluster_count"
        )

    sort_label = st.sidebar.selectbox(
        "Sort results",
        ["Relevance", "Newest first"],
        key="sort_label"
    )
    sort_by = "newest" if sort_label == "Newest first" else "relevance"

    selected_classification = st.sidebar.selectbox(
        "Classification",
        classification_options,
        format_func=lambda x: "Any" if x == "" else x,
        key="selected_classification"
    )

    selected_status = st.sidebar.selectbox(
        "Status",
        status_options,
        format_func=lambda x: "Any" if x == "" else x,
        key="selected_status"
    )

    selected_state = st.sidebar.selectbox(
        "State",
        state_options,
        format_func=lambda x: "Any" if x == "" else x,
        key="selected_state"
    )

    use_date_filter = st.sidebar.checkbox(
        "Filter by recall date",
        value=False,
        help="Restrict results to recalls initiated within a selected date range.",
        key="use_date_filter"
    )

    if use_date_filter and min_date and max_date:
        st.sidebar.caption(
            f"Available recall dates in this local dataset: "
            f"{min_date.isoformat()} to {max_date.isoformat()}"
        )

        if max_date < date.today():
            st.sidebar.caption(
                f"Preset ranges are anchored to {max_date.isoformat()}, "
                f"the newest recall date currently indexed."
            )

        date_preset = st.sidebar.selectbox(
            "Choose a date range",
            ["Entire dataset", "Past week", "Past month", "Past 6 months", "Past year", "Custom range"],
            key="date_preset"
        )

        if date_preset == "Custom range":
            start_date, end_date = st.sidebar.slider(
                "Recall date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD",
                key="custom_date_range"
            )
        else:
            start_date, end_date = get_preset_date_range(date_preset, min_date, max_date)
            st.sidebar.caption(
                f"Selected range: {start_date.isoformat()} to {end_date.isoformat()}"
            )
    elif use_date_filter:
        st.sidebar.caption("No recall dates are available for filtering.")

    st.sidebar.button(
        "Reset",
        type="tertiary",
        use_container_width=True,
        on_click=reset_search_state
    )


filters = {}

if selected_classification:
    filters["classification"] = selected_classification

if selected_status:
    filters["status"] = selected_status

if selected_state:
    filters["state"] = selected_state

# -----------------------------
# Main search input
# -----------------------------

query = st.session_state.get("query_input", "")
search_requested = False

if not active_more_like_view:
    query = st.text_input(
        "Enter a food recall search query",
        key="query_input",
        on_change=mark_search_requested,
        placeholder="Examples: milk allergy, salmonella contamination, baby formula, metal fragments"
    )

    st.markdown(
        "**Try examples:** `milk allergy`, `baby formula`, "
        "`salmonella contamination`, `metal fragments`, `lettuce recall`"
    )

    # Suggested query variants
    if query:
        suggestions = get_query_suggestions(query)

        if suggestions:
            st.subheader("Suggested query variants")
            for suggestion in suggestions:
                st.write(f"- {suggestion}")

    search_button = st.button(
        "Search",
        type="primary",
        use_container_width=True
    )
    search_requested = search_button or st.session_state.search_requested


# -----------------------------
# Run a new search
# -----------------------------

if search_requested and query:
    st.session_state.search_requested = False
    st.session_state.search_output = process_query(
        query=query,
        top_k=top_k,
        use_expansion=use_expansion,
        use_prf=use_prf,
        filters=filters,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_by
    )

    # Clear old query-by-example results when a new search is submitted
    st.session_state.more_like_results = []
    st.session_state.more_like_source = None
    st.session_state.selected_more_like_doc = None
    st.session_state.carousel_index = 0


elif search_requested and not query:
    st.session_state.search_requested = False
    st.warning("Please enter a query first.")


# -----------------------------
# Display saved search results
# -----------------------------

if active_more_like_view and active_output is not None and active_source_result is not None:
    st.subheader("Similar Recalls")

    if st.button("Back to search results", type="tertiary"):
        reset_more_like_state()
        st.rerun()

    st.caption(
        f"Finding recalls similar to **{active_source_result['doc_id']}**: "
        f"{clean_title(active_source_result['product_description'], max_chars=110)}"
    )
    st.caption(
        "Query-by-example uses TF-IDF cosine similarity over the indexed recall text. "
        "It is independent of the selected cluster method."
    )

    st.markdown("#### Selected recall")
    display_result_card(
        active_source_result,
        button_prefix="focused_source",
        query_text=active_output.get("highlight_query", active_output["original_query"]),
        show_more_like_button=False
    )

    if st.session_state.more_like_loading_doc == active_source_result["doc_id"]:
        with st.spinner("Searching for similar recalls...", show_time=True):
            st.session_state.more_like_results = more_like_this(
                active_source_result["doc_id"],
                top_k=5
            )
            st.session_state.more_like_source = active_source_result["doc_id"]
            st.session_state.more_like_loading_doc = None
        st.rerun()

    if st.session_state.more_like_results:
        display_more_like_carousel(
            source_doc_id=active_source_result["doc_id"],
            source_product=active_source_result["product_description"]
        )
    else:
        st.info("No similar recalls were found for this record.")

elif st.session_state.search_output is not None:
    output = st.session_state.search_output

    st.divider()

    st.subheader("Search Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Original Query", output["original_query"])

    with col2:
        st.metric("Expansion Used", "Yes" if output["use_expansion"] else "No")

    with col3:
        st.metric("Results Returned", len(output["results"]))

    with col4:
        st.metric("Sort", "Newest" if output["sort_by"] == "newest" else "Relevance")

    with col5:
        st.metric("PRF Used", "Yes" if output["use_prf"] else "No")

    st.caption(
        f"Cluster method: {cluster_method_label}"
        + (f" ({cluster_count} clusters)" if cluster_method == "kmeans" else "")
    )

    if output["query_corrections"]:
        st.info(
            "Auto-corrected query terms: "
            + format_query_corrections(output["query_corrections"])
        )

    if output["corrected_query"] and output["corrected_query"] != output["original_query"]:
        st.markdown("**Corrected query used for retrieval:**")
        st.write(output["corrected_query"])

    if output["use_expansion"]:
        st.markdown("**Rule-expanded query:**")
        st.write(output["rule_expanded_query"])

        if output["expansion_terms"]:
            st.markdown("**Rule expansion terms:**")
            st.write(", ".join(output["expansion_terms"]))
        elif output["expansion_status_message"]:
            st.warning(f"Rule expansion not applied: {output['expansion_status_message']}")

    if output["use_prf"]:
        if output["prf_terms"]:
            st.markdown("**PRF terms:**")
            st.write(", ".join(output["prf_terms"]))
        elif output["prf_status_message"]:
            st.warning(f"PRF not applied: {output['prf_status_message']}")

        st.markdown("**Final reranking query:**")
        st.write(output["expanded_query"])

    if output["filters"]:
        st.markdown("**Applied filters:**")
        st.write(output["filters"])

    if output["start_date"] or output["end_date"]:
        st.markdown("**Recall date range:**")
        st.write(f"{output['start_date']} to {output['end_date']}")

    st.divider()

    results = output["results"]
    if not results:
        st.warning(output.get("message", "No results found."))
        st.caption("Try removing filters, correcting the query, or using query expansion.")

    else:
        clusters = cluster_results(
            results,
            method=cluster_method,
            n_clusters=cluster_count
        )
        ranked_tab, clustered_tab, analytics_tab = st.tabs(
            ["Ranked Results", "Clustered View", "Analytics"]
        )

        with ranked_tab:
            for result in results:
                display_result_card(
                    result,
                    button_prefix="ranked",
                    query_text=output.get("highlight_query", output["original_query"])
                )

        with clustered_tab:
            for cluster in clusters:
                cluster_key = make_safe_key(cluster["label"])

                with st.expander(
                    f"{cluster['label']} ({cluster['size']})",
                    expanded=True
                ):
                    if cluster["match_terms"]:
                        st.markdown(
                            "<div>"
                            f"<span style='color:#c62828; font-weight:700;'>{cluster['term_label']}:</span> "
                            f"{', '.join(cluster['match_terms'])}"
                            "</div>",
                            unsafe_allow_html=True
                        )

                    for result in cluster["results"]:
                        display_result_card(
                            result,
                            button_prefix=f"cluster_{cluster_key}",
                            query_text=output.get("highlight_query", output["original_query"])
                        )

        with analytics_tab:
            display_result_analytics(results)