import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
PLOTS_DIR = EVALUATION_DIR / "plots"

sys.path.insert(0, str(SRC_DIR))

from evaluation import build_relevance_lookup, run_config
from query_processor import process_query


QUERIES_PATH = EVALUATION_DIR / "test_queries.csv"
JUDGMENTS_PATH = EVALUATION_DIR / "relevance_judgments.csv"
SUMMARY_RESULTS_PATH = EVALUATION_DIR / "results.csv"
PER_QUERY_RESULTS_PATH = EVALUATION_DIR / "per_query_results.csv"
HYPERPARAMETER_RESULTS_PATH = EVALUATION_DIR / "hyperparameter_results.csv"


def plot_summary_metrics(summary_df):
    """
    Save config comparison plots for effectiveness and latency.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    effectiveness_metrics = ["p@5", "r@10", "ap", "ndcg@10"]
    latency_metrics = ["avg_latency_ms", "p95_latency_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    summary_df.plot(
        x="config",
        y=effectiveness_metrics,
        kind="bar",
        ax=axes[0]
    )
    axes[0].set_title("Effectiveness Metrics by Configuration")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=0)

    summary_df.plot(
        x="config",
        y=latency_metrics,
        kind="bar",
        ax=axes[1]
    )
    axes[1].set_title("Latency by Configuration")
    axes[1].set_ylabel("Milliseconds")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "config_comparison.png", dpi=200)
    plt.close(fig)


def plot_hyperparameter_study(summary_df):
    """
    Save PRF hyperparameter sweep plots.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.copy()
    plot_df["config_label"] = plot_df.apply(
        lambda row: f"docs={int(row['prf_feedback_docs'])}, terms={int(row['prf_expansion_terms'])}",
        axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_df.plot(
        x="config_label",
        y=["ap", "ndcg@10"],
        kind="bar",
        ax=axes[0]
    )
    axes[0].set_title("PRF Hyperparameter Effectiveness")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=20)

    plot_df.plot(
        x="config_label",
        y=["avg_latency_ms", "p95_latency_ms"],
        kind="bar",
        ax=axes[1]
    )
    axes[1].set_title("PRF Hyperparameter Latency")
    axes[1].set_ylabel("Milliseconds")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prf_hyperparameter_study.png", dpi=200)
    plt.close(fig)


def main():
    """
    Run benchmark configs and save CSV results plus plots.
    """
    queries_df = pd.read_csv(QUERIES_PATH)
    judgments_df = pd.read_csv(JUDGMENTS_PATH)
    relevance_lookup = build_relevance_lookup(judgments_df)

    configs = [
        {
            "name": "baseline",
            "use_expansion": False,
            "use_prf": False,
            "prf_feedback_docs": 3,
            "prf_expansion_terms": 5,
        },
        {
            "name": "expansion",
            "use_expansion": True,
            "use_prf": False,
            "prf_feedback_docs": 3,
            "prf_expansion_terms": 5,
        },
        {
            "name": "prf",
            "use_expansion": False,
            "use_prf": True,
            "prf_feedback_docs": 3,
            "prf_expansion_terms": 5,
        },
        {
            "name": "full",
            "use_expansion": True,
            "use_prf": True,
            "prf_feedback_docs": 3,
            "prf_expansion_terms": 5,
        },
    ]
    hyperparameter_configs = [
        {
            "name": "prf_docs2_terms5",
            "use_expansion": False,
            "use_prf": True,
            "prf_feedback_docs": 2,
            "prf_expansion_terms": 5,
        },
        {
            "name": "prf_docs3_terms5",
            "use_expansion": False,
            "use_prf": True,
            "prf_feedback_docs": 3,
            "prf_expansion_terms": 5,
        },
        {
            "name": "prf_docs5_terms5",
            "use_expansion": False,
            "use_prf": True,
            "prf_feedback_docs": 5,
            "prf_expansion_terms": 5,
        },
        {
            "name": "prf_docs3_terms8",
            "use_expansion": False,
            "use_prf": True,
            "prf_feedback_docs": 3,
            "prf_expansion_terms": 8,
        },
    ]

    summary_rows = []
    per_query_rows = []
    hyperparameter_rows = []

    for config in configs:
        summary_row, config_query_rows = run_config(
            config_name=config["name"],
            queries_df=queries_df,
            relevance_lookup=relevance_lookup,
            process_query_fn=process_query,
            use_expansion=config["use_expansion"],
            use_prf=config["use_prf"],
            prf_feedback_docs=config["prf_feedback_docs"],
            prf_expansion_terms=config["prf_expansion_terms"],
            top_k=10,
        )
        summary_rows.append(summary_row)
        per_query_rows.extend(config_query_rows)

    for config in hyperparameter_configs:
        summary_row, _ = run_config(
            config_name=config["name"],
            queries_df=queries_df,
            relevance_lookup=relevance_lookup,
            process_query_fn=process_query,
            use_expansion=config["use_expansion"],
            use_prf=config["use_prf"],
            prf_feedback_docs=config["prf_feedback_docs"],
            prf_expansion_terms=config["prf_expansion_terms"],
            top_k=10,
        )
        hyperparameter_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    per_query_df = pd.DataFrame(per_query_rows)
    hyperparameter_df = pd.DataFrame(hyperparameter_rows)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)
    per_query_df.to_csv(PER_QUERY_RESULTS_PATH, index=False)
    hyperparameter_df.to_csv(HYPERPARAMETER_RESULTS_PATH, index=False)
    plot_summary_metrics(summary_df)
    plot_hyperparameter_study(hyperparameter_df)

    print("Saved evaluation summary to:", SUMMARY_RESULTS_PATH)
    print("Saved per-query results to:", PER_QUERY_RESULTS_PATH)
    print("Saved hyperparameter results to:", HYPERPARAMETER_RESULTS_PATH)
    print("Saved plots to:", PLOTS_DIR)
    print("\nSummary:")
    print(summary_df.round(4))
    print("\nHyperparameter study:")
    print(hyperparameter_df.round(4))


if __name__ == "__main__":
    main()
