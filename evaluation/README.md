# **Evaluation Benchmark**

This folder contains a lightweight benchmark for comparing retrieval settings in the FDA Recall Search Assistant.

The benchmark uses 20 queries and 237 graded relevance pairs over stable document IDs derived from normalized FDA `recall_number` values, so judgments remain meaningful after corpus rebuilds.

## **Files**

Source files:

- `test_queries.csv`: 20-query benchmark set
- `relevance_judgments.csv`: graded relevance labels (`0` absent, `1` relevant, `2` highly relevant)
- `run_experiments.py`: runs baseline vs enhanced configurations and the PRF hyperparameter sweep
- `export_judgment_pool.py`: exports pooled candidate documents for manual judgment expansion

Generated outputs:

- `results.csv`: summary metrics from `run_experiments.py`
- `per_query_results.csv`: per-query metrics from `run_experiments.py`
- `hyperparameter_results.csv`: PRF hyperparameter sweep from `run_experiments.py`
- `judgment_pool.csv`: pooled candidate set from `export_judgment_pool.py`
- `plots/`: matplotlib plots written by `run_experiments.py` (`config_comparison.png`, `prf_hyperparameter_study.png`)

## **Judgment Rubric**

- `2`: highly relevant
  - directly matches the product, contamination, allergen, or recall issue named in the query
- `1`: relevant
  - related recall that partially matches the query intent or uses a close variant of the same issue
- `0`: not included in the CSV
  - all unlisted documents are treated as non-relevant for that query

## **Configurations**

The experiment runner currently evaluates:

- `baseline`: no rule expansion, no PRF
- `expansion`: rule-based query expansion only
- `prf`: pseudo-relevance feedback only, with fallback to baseline when feedback is weak or unstable
- `full`: rule expansion plus guarded PRF

It also runs a PRF hyperparameter study over feedback-document counts (`2`, `3`, `5`) and feedback-term counts (`5`, `8`). Because PRF is intentionally conservative, some settings may produce identical output when the safeguard decides not to expand.

## **Run**

From the project root:

```powershell
python evaluation/run_experiments.py
python evaluation/export_judgment_pool.py
```

`run_experiments.py` saves summary metrics, per-query metrics, and comparison plots in this folder. `export_judgment_pool.py` writes `judgment_pool.csv`.

**MAP (reported as `ap` in CSV output).** The experiment runner evaluates each query at `top_k=10`, so mean average precision is computed over those top 10 ranked documents only (AP@10). That matches how we compare configs in this benchmark: all runs use the same depth, and most judged relevant recalls appear in the top 10 for our query set. It is a deliberate lightweight choice for a course-project benchmark, not full-corpus MAP.

Because this is a lightweight benchmark, the judgments are intentionally sparse compared with the full rebuilt corpus. Metric values are most useful for relative comparison between configs rather than as a complete measure of real-world retrieval quality.
