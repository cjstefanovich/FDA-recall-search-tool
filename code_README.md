# **FDA Recall Search Assistant**

FDA Recall Search Assistant is a local search system for FDA food enforcement recall records. 

The project includes a retrieval pipeline for ingestion, preprocessing, indexing, and ranking, plus an interactive Streamlit interface with filters, query expansion, clustered browsing, pseudo-relevance feedback, and query-by-example.

## **Team**

- Connor Stefanovich

## **Main Features**

- OpenFDA crawler for food enforcement recall records
- Local cleaned dataset and weighted text corpus
- Inverted index with TF-IDF cosine ranking
- Rule-based FDA query expansion
- Query normalization for common punctuation variants and a few high-value misspellings
- Faceted filtering by classification, status, state, and recall date
- Date range filtering and newest-first sorting
- Keyword and k-means clustered browsing views
- Lightweight analytics panel for result exploration
- Pseudo-relevance feedback reranking
- "More like this" query-by-example search
- Evaluation pipeline with Precision@k, Recall@k, MAP, NDCG, and latency

## **Project Structure**

```text
fda-recall-search/
|---- app.py
|---- run_app.bat
|---- requirements.txt
|---- README.md
|---- data/
|   |---- raw/
|   |---- processed/
|   |---- corpus/
|---- evaluation/
|   |---- README.md
|   |---- test_queries.csv
|   |---- relevance_judgments.csv
|   |---- run_experiments.py
|   |---- export_judgment_pool.py
|   |---- plots/
|---- index/
|---- src/
    |---- clustering.py
    |---- corpus_builder.py
    |---- crawler.py
    |---- evaluation.py
    |---- facets.py
    |---- feedback.py
    |---- indexer.py
    |---- preprocessor.py
    |---- prf.py
    |---- query_expansion.py
    |---- query_normalization.py
    |---- query_processor.py
    |---- ranker.py
    |---- resources.py
```

## **Requirements**

- Python 3.10+
- Python packages in `requirements.txt`

Dependencies (also in `requirements.txt`):

- `pandas`
- `numpy`
- `requests`
- `nltk`
- `scikit-learn`
- `scipy`
- `streamlit`
- `matplotlib`

## **Setup**

Install dependencies in your preferred Python environment, such as in a virtual environment:

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the plain `streamlit` command is not available in your shell, use `python -m streamlit` instead.

The preprocessing code will download the NLTK English stopword list automatically the first time it is needed.

## **Data Source And Snapshot Access**

Source API:

- [OpenFDA Food Enforcement API](https://api.fda.gov/food/enforcement.json)

The crawler supports fetching the full available food enforcement snapshot from OpenFDA. At the time of the latest rebuild in this repository, the full snapshot contains about 28,890 food enforcement records.

Document IDs are stable and are derived from normalized FDA `recall_number` values. This makes rebuilds, evaluation judgments, and demo references more reproducible.

The larger rebuilt dataset and index is available in Google Drive with this link:

`https://drive.google.com/file/d/17LMlR0DgaI4zBP5bbDXJTTzVA7rui-JI/view?usp=sharing`

If you download the ready-to-run snapshot from the shared drive, place the extracted folders directly inside `fda-recall-search/` so the structure looks like this:

```text
fda-recall-search/
|---- app.py
|---- README.md
|---- requirements.txt
|---- run_app.bat
|---- data/
|   |---- raw/
|   |---- processed/
|   |---- corpus/
|---- evaluation/
|   |---- # evaluation files
|---- index/
|   |---- doc_metadata.csv
|   |---- index_stats.json
|   |---- inverted_index.json
|   |---- vocabulary.json
|---- src/
|   |---- # python code files
```

If those `data/` and `index/` folders are already populated with the downloaded data, you can skip the rebuild steps and run the app immediately.

If needed, the dataset can also be recreated locally from the public API by running the crawler/build pipeline described below.

## **Rebuild The Local Search Index**

Run these commands from the `fda-recall-search/` directory:

```powershell
python src/crawler.py
python src/corpus_builder.py
python src/indexer.py
```

Pipeline summary:

1. `src/crawler.py` downloads recall records and saves raw JSON plus a cleaned CSV.
  - It uses stable recall-based document IDs and OpenFDA `search_after` paging to fetch beyond the `skip` limit.
2. `src/corpus_builder.py` turns each recall into a weighted text document and metadata row.
3. `src/indexer.py` builds the inverted index, vocabulary, and corpus statistics.

## **Run The App**

From the project root, the easiest Windows launch option is to use `run_app.bat`. Simply double-click it to run the following command automatically:

```powershell
streamlit run app.py
```

## **CLI Search**

You can also search from the command line:

```powershell
python src/query_processor.py "baby formula" --expand --top_k 10
python src/query_processor.py "listeria" --prf --classification "Class I" --sort_by newest
```

Useful flags include `--expand`, `--prf`, `--classification`, `--status`, `--state`, `--start_date`, `--end_date`, and `--sort_by` (`relevance` or `newest`).

## **Search Options In The UI**

- Query expansion
- Pseudo-relevance feedback
- Result count slider (up to 50)
- Sort by relevance or newest first
- Metadata filters for classification, status, and state
- Recall date range filter
- Ranked results tab
- Clustered view tab (keyword or k-means with user-defined number of clusters)
- Analytics tab with trend/count visualizations
- Typo-aware query normalization for a few common search variants
- Dedicated more-like-this browsing page

## **Demo Queries**

These are useful test queries:

- `baby formula`
- `infant formula`
- `milk allergy`
- `undeclared milk`
- `salmonella contamination`
- `metal fragments`
- `foreign material`
- `lettuce recall`
- `ice cream listeria`
- `potato salad recall`

Good before/after demo ideas:

- `baby formula` with query expansion off vs. on
- `milk allergy` with and without PRF
- `metal fragments` in ranked view vs clustered view

## **Evaluation**

The evaluation benchmark lives in `evaluation/`. There is also a more detailed `README.md` file in that folder that goes into more detail about how evaluation works.

Run experiments from the project root:

```powershell
python evaluation/run_experiments.py
```

This writes:

- `evaluation/results.csv`
- `evaluation/per_query_results.csv`
- `evaluation/hyperparameter_results.csv`
- `evaluation/plots/config_comparison.png`
- `evaluation/plots/prf_hyperparameter_study.png`

The `evaluation/plots/` folder is created when you run the script. Pre-generated plot files are not required for normal use.

The current benchmark compares:

- Baseline
- Rule expansion
- PRF
- Full pipeline (`expansion + PRF`)

Metrics implemented:

- `Precision@5`
- `Recall@10`
- `MAP`
- `NDCG@5`
- `NDCG@10`
- average latency
- p95 latency

The repository also includes a pooled candidate export script to help expand judgments on the larger corpus:

```powershell
python evaluation/export_judgment_pool.py
```

## **Implementation Notes**

- Field weighting is implemented in `src/corpus_builder.py` by repeating high-signal fields in the document text.
- Search resources are cached in `src/resources.py` so repeated queries do not keep reloading index files from disk.
- Filters are applied at the candidate-selection stage rather than only after taking a fixed top-50 pool.
- Keyword clustering is intentionally simple.
- PRF is implemented as a separate feature from rule-based expansion so they can be compared in experiments.
- PRF includes guardrails that skip or roll back expansion when the pseudo-feedback evidence is too narrow, which keeps the interface stable on already-specific queries.
- "More like this" feature uses index-backed TF-IDF cosine similarity for the sake of shortening execution time, and is independent of the user's selected cluster method.

## **Known Limitations**

- The current clustering view is keyword-based; optional k-means clustering is not yet the default.
- PRF is conservative and may deliberately fall back to the baseline ranking when pseudo-feedback looks too query-specific.
- The benchmark judgments are lightweight and intended for course-project comparison rather than large-scale IR evaluation.
- The project depends on locally built `data/` and `index/` artifacts.
- The corpus builder uses weighted text repetition instead of a fully separate fielded index.

