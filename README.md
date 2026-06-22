# **FDA-recall-search-tool**

I completed this project for an image processing class for my graduate degree in Spring 2026. It uses intelligent information retreival techniques with a UI to help users search for food recalls issued by the FDA.

The dataset comes from the [OpenFDA Food Enforcement API](https://open.fda.gov/apis/food/enforcement/), where each result describes one enforcement action with semi-structured text and filter attributes (classification, status, geographic state, recall dates, product type). reason_for_recall and product_description are the primary retrieval targets; firm, location, and code fields provide secondary text.

The project combines a custom search engine with an enhanced retrieval interface. The search engine includes separate crawler, indexer, and query-processing components; builds an inverted-file index over a local OpenFDA corpus with TF-IDF weighting; and persists index artifacts for reload across sessions. The interface supports query formulation through normalization and optional rule expansion, metadata facets, relevance feedback, query-by-example browsing, and clustering and analytics in Streamlit. All retrieval runs against a local corpus snapshot with no live OpenFDA calls at query time.

## **Methodology**

Offline, crawler.py, corpus_builder.py, and indexer.py ingest OpenFDA records, build weighted corpus files, and serialize the inverted index under index/. Online, query_processor.py coordinates normalization, optional expansion and PRF, pre-retrieval filtering, TF-IDF ranking, and result packaging; resources.py caches index artifacts in memory. 

### **Data aquisition and corpus construction:**

The crawler paginates up to 1,000 records per request sorted by report_date, following OpenFDA next link pagination with exponential backoff on failure. One recall equals one document, identified by a normalized recall_number stable across rebuilds and aligned with evaluation judgments. Raw JSON and a cleaned table (recalls_clean.csv) are stored under data/raw/ and data/processed/; each recall is written to data/corpus/{doc_id}.txt. The repository rebuild contains 28,890 records (recall initiation dates 2008-02-22 through 2026-05-01). Thirteen text fields are concatenated for indexing; high-signal fields are repeated in corpus files rather than stored in a separate fielded index. Recall dates and display columns live in doc_metadata.csv for facets and UI cards.

### **Preprocessing, indexing, and ranking:**

Documents and queries share preprocessing in preprocessor.py: lowercase conversion, punctuation removal, alphanumeric tokenization, English stopword removal, and Porter stemming via NLTK. The indexer accumulates within-document term frequencies and writes an inverted index {term: {doc_id: tf}}. For a corpus of N documents, smoothed inverse document frequency is:

$$
idf(t) = \log\left(\frac{1 + N}{1 + df(t)} + 1\right)
$$

Where df(t) is the document frequency of term t. We precompute each document’s TF-IDF vector norm for cosine scoring at query time. The vocabulary contains approximately 112,706 distinct terms; average document length is about 209 tokens after preprocessing. Persisted artifacts include inverted_index.json, index_stats.json, vocabulary.json, and doc_metadata.csv.
For query q and document d, cosine similarity over TF-IDF weights is:

$$
sim(q, d) =
\frac{\sum_{t \in q \cap d} w_{t,q} w_{t,d}}{\lVert q \rVert \lVert d \rVert}
$$

Where q is the query, d is the document, w_{t,q} is the TF-IDF weight of term t in the query, w_{t,d} is the TF-IDF weight of term t in the document, and q∩d is terms appearing in both query and document. Therefore, the similarity of query q and document d is the sum over shared terms t of (query weight * document weight), divided by the product of the query and document vector norms.

ranker.py accumulates dot products via inverted-list postings and normalizes by query and document norms from index_stats.json.

Each online query follows five steps: (1) normalize and optionally apply rule expansion or PRF; (2) restrict candidates with active facet and date filters; (3) look up query terms in the inverted index; (4) rank candidates with TF-IDF cosine similarity; (5) truncate to top-k and return metadata with highlighted terms. Facet and date constraints apply before scoring (facets.py) to avoid losing strong matches to a fixed top-k cutoff. By default, results sort by descending similarity; the UI can re-sort by recall initiation date when needed.

### **Query enhancement:**

Rule expansion and PRF are independent UI toggles, enabling clean ablation in experiments.

Query normalization. query_normalization.py runs on every query before expansion or PRF. It canonicalizes common variants (for example, "E.coli", "ecoli" to e coli) and applies a misspelling table plus fuzzy matching against a controlled domain term list (for example, "salmonela" to salmonella).

Rule-based query expansion. query_expansion.py adds FDA-domain terms when a query matches a narrow token or phrase rule, such as mapping "formula" toward infant-formula vocabulary or "metal" toward foreign-material phrases. At most, four terms are appended per query. Skip lists and pattern checks leave already-specific queries unchanged (for instance, queries beginning with undeclared or ending with contamination). When no rule fires, the UI reports that expansion was skipped. Broad expansion hurt precision in further experiments (Section 4.4).

Guarded pseudo-relevance feedback. prf.py re-queries using terms drawn from the TF-IDF vectors of top baseline results treated as pseudo-relevant documents. Guardrails limit unstable reranking:

-	Query-pattern skips for already-narrow inputs (allergen phrases, contamination wording, etc.).
-	Each feedback document must contain every preprocessed query term.
-	Proposed PRF terms must appear in at least two-thirds of selected feedback documents.
-	Keep a PRF rerank only if rank-1 remains in the top three and at least three of the original top-five documents remain in the reranked top five.

If any check fails, the system returns the baseline ranking with a short UI explanation. Default settings use three feedback documents and five candidate expansion terms.

### **User interface:**

The Streamlit app provides ranked results, clustered view, analytics, and a "More like this" page for query-by-example similarity. The sidebar exposes classification, status, state, and optional recall initiation date filters. Clustering defaults to keyword-based groups (infant products, bacterial contamination, allergens, foreign material, chemical/residue) with optional k-means over retrieved TF-IDF vectors; hierarchical clustering was dropped due to high latency. Query-by-example search in feedback.py uses posting-list intersection rather than brute-force corpus comparison, reducing latency from tens of seconds to roughly 200 ms in testing.

The UI has the following features:

- Ranked results: Top-k retrieval with cosine scores and highlighted query terms
- Clustered view: Exploratory grouping of the current result set
- Analytics view: Aggregate views (counts by categories, recalls over time)
- "More like this" page: Query-by-example similarity from a selected recall

### **Design decisions:**

I implemented the inverted index, TF-IDF ranker, expansion rules, PRF logic, facet restriction, and posting-based similarity; external libraries handle HTTP (requests), tables (pandas), tokenization primitives (nltk), vector operations (numpy, scipy), k-means (scikit-learn), the UI (streamlit), and plots for analytics/evaluation (matplotlib).

Key decisions and their rationale:

- Stable recall_number doc IDs: Reproducible rebuilds and stable judgments
- Field weighting via text repetition: Boost key fields without a separate fielded index
- Pre-retrieval facet filtering: Avoid losing strong matches to a fixed top-k cutoff
- Separate expansion and PRF toggles: Clean ablation and user control
- TF-IDF + cosine ranking: Strong benchmark accuracy at low latency
- Cached index loading: Avoid reloading large JSON/CSV per query

## **Experiments and results**

### **Evaluation design:**

I built a 20-query benchmark (test_queries.csv) covering infant formula, allergens, pathogens, foreign material, and related topics. Relevance labels are graded (2 = highly relevant, 1 = relevant, 0 = non-relevant). relevance_judgments.csv contains 237 labeled pairs across 206 unique recalls, built from a pooled top-10 candidate set across all four configurations: baseline (no expansion, no PRF), expansion, PRF, and full (expansion + PRF). We report P@5, R@10, MAP, NDCG@10, and average latency. For P@k, R@k, and MAP, grades 1 and 2 count as relevant. All runs use top_k = 10, normalization, and no facet filters, so MAP is computed over each query’s top 10 ranked documents only. run_experiments.py regenerates result CSVs and plots.

### **Quantitative results:**

| **Config**       | **MAP**      | **P@5**     | **R@10**     | **NDCG@10**  | **Avg. latency (ms)** |
|------------------|--------------|-------------|--------------|--------------|-----------------------|
|     Baseline     |     0.560    |     0.90    |     0.584    |     0.735    |     11.3              |
|     Expansion    |     0.592    |     0.95    |     0.613    |     0.771    |     14.5              |
|     PRF          |     0.560    |     0.90    |     0.584    |     0.735    |     26.8              |
|     Full         |     0.592    |     0.95    |     0.613    |     0.771    |     30.4              |

Rule expansion raises MAP to 0.592 and P@5 to 0.95 at 14.5 ms average latency (+3.2 ms over baseline). Gains come mainly from "baby formula" (success case, Q01: P@5 0.0 to 1.0). "peanut allergen" (failure case, Q05) remains weak across configs (P@5 = 0.4), consistent with vocabulary mismatch. PRF matched baseline on every query because guardrails always returned the baseline ranking; full matched expansion exactly with added latency.

Hyperparameter study. Table 5 reports a PRF sweep with rule expansion off. Three feedback documents and five terms yield the best MAP (0.560) and P@5 (0.90). Using only two feedback documents lowered MAP to 0.545; increasing feedback depth to five documents did not help. prf_docs3_terms8 ties prf_docs3_terms5 because guardrails did not select additional terms. These results confirm the default PRF settings used in the main benchmark.

| **Setting**             | **Feedback docs** | **PRF terms** | **MAP**      | **P@5**     | **NDCG@10**  | **Avg. latency (ms)** |
|-------------------------|-------------------|---------------|--------------|-------------|--------------|-----------------------|
|     prf_docs2_terms5    |     2             |     5         |     0.545    |     0.89    |     0.720    |     23.4              |
|     prf_docs3_terms5    |     3             |     5         |     0.560    |     0.90    |     0.735    |     24.2              |
|     prf_docs5_terms5    |     5             |     5         |     0.549    |     0.90    |     0.726    |     25.0              |
|     prf_docs3_terms8    |     3             |     8         |     0.560    |     0.90    |     0.735    |     24.2              |

### **Qualitative results:**

Two patterns recur across benchmark and illustrative queries. Expansion closes vocabulary gaps when broad terms miss document phrasing (success case, Q01). Skip logic preserves precision on already-specific inputs such as undeclared milk and metal fragments (P@5 = 1.0 without expansion).

The query "fargments" is not in the benchmark but illustrates normalization and enhancement behavior. Normalization corrects the typo to fragments, expansion adds foreign-material vocabulary, and PRF declines to rerank because feedback is too noisy. Narrower inputs such as "wheat" can activate PRF when feedback is coherent. Clustering and analytics help when ranked lists alone are hard to interpret; those UI features are not reflected in P@5.

### **Further exploratory experiments:**

After the main benchmark was stable, we ran a separate branch of further experiments to test alternative rankers and enhancement strategies before finalizing the codebase.

We first replaced TF-IDF with BM25 and tuned BM25 hyperparameters. Tuned BM25 reached only MAP = 0.232 and NDCG@10 = 0.412, far below the TF-IDF baseline (MAP = 0.560), with noticeably higher latency. We also tried aggressive query expansion rules that appended many related terms per query. On "milk allergy", average precision fell from 0.505 to 0.007, showing that broad expansion caused query drift rather than better recall. Unrestricted PRF diagnostics matched the guarded baseline (MAP = 0.560) but did not improve rankings. Rocchio-style vector feedback reached MAP = 0.533 and NDCG@10 = 0.718, below baseline on both metrics.

We also experimented with field importance directly in the ranker. Standalone fielded TF-IDF performed poorly (MAP = 0.083; NDCG@10 = 0.175), likely because our repetition-based weighting already captured much of the same signal. A blended fielded rerank came closest to the expansion configuration (NDCG@10 = 0.772 versus 0.771 for expansion alone), but still scored below rule-based expansion on MAP (0.590 versus 0.592) while adding latency.

These prototypes were not merged into the finalized codebase because none offered a clear benefit on the 20-query benchmark. We therefore shipped TF-IDF + conservative expansion.

## **Discussion**

### **Analysis of results:**

Baseline TF-IDF already reaches P@5 = 0.90 on our 20 recall-focused queries, which suggests that field weighting, preprocessing, and inverted-index retrieval work well for this corpus. Rule-based expansion is the only setting that improves aggregate effectiveness: MAP and P@5 rose (0.560 to 0.592 and 0.90 to 0.95 respectively) with a modest latency cost (+3.2 ms). The largest gain comes from "baby formula" (Q01), where expansion bridges a synonym gap; "peanut allergen" (Q05) remains weak across all configurations (P@5 = 0.4), indicating a vocabulary mismatch that enhancement alone cannot fix. PRF and the full configuration did not change benchmark rankings because guardrails returned the baseline ordering on every query, adding latency without further benefit on this query set.

### **Hyperparameters:**

Expansion is capped at four terms with explicit skip lists because aggressive rules from further experiments collapsed precision. Skip logic preserved strong performance on already-specific queries such as "undeclared milk". The PRF sweep (Table 5) shows that three feedback documents and five terms yield the best MAP and P@5; two feedback documents lowered MAP to 0.545, and increasing terms to eight tied the default because guardrails did not adopt additional terms.

### **What worked, what didn't, and next steps**

OpenFDA crawling, offline index persistence, pre-retrieval facets, and query normalization worked well; clustering, analytics, and query-by-example complemented ranked lists. Guarded PRF did not improve benchmark metrics, and BM25, Rocchio feedback, aggressive expansion, and direct fielded TF-IDF were abandoned after further experiments showed no net benefit. Evaluation is also limited by sparse judgments: with only 237 labeled pairs across 28,890 documents, unjudged recalls are treated as non-relevant, which may understate performance when expansion retrieves plausible matches outside the pooled set, as on "peanut allergen" (Q05). Future work could include richer pooled judgments, data-driven expansion rules, or lightweight semantic reranking with strict latency caps.
