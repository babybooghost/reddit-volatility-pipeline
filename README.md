# Reddit Volatility Pipeline

Does retail discussion carry information about the next 48 hours of market volatility? This project turns posts from four finance subreddits into dense vectors, reduces them with PCA, and evaluates three classifiers on later market data.

The core constraint is time: every text snapshot, engagement count, and training label must be available before it can enter the model. Python 3.11 or later is required.

## Browser demo

[Open the dashboard](https://babybooghost.github.io/reddit-volatility-pipeline/). No terminal, account, or API key is needed.

The dashboard shows actual Python model outputs from **synthetic inputs**. Change the decision threshold, inspect errors, and download the results. Browser controls recalculate decisions from saved scores; they do not retrain models. All displayed metrics describe this generated experiment, not real-world performance.

To regenerate the checked-in dataset after installing dependencies:

```sh
.venv/bin/python build_demo.py
.venv/bin/python -m http.server 8765 --directory docs
```

Open `http://localhost:8765`. GitHub Pages serves `docs/` from `main`. Commit regenerated `docs/data.json` to publish new results. The dataset tests independently recompute ROC-AUC, average precision, and ROC points from the saved predictions.

## Install and verify

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/mypy volatility_pipeline.py research_utils.py
```

`requirements.lock` records the environment used for verification. Reproduce it with `uv pip sync --python .venv/bin/python requirements.lock`, then install this repository using `uv pip install --python .venv/bin/python --no-deps -e .`.

## Reddit sentiment and volatility

```bash
.venv/bin/python volatility_pipeline.py \
  --posts data/posts.parquet --prices data/hourly_prices.parquet \
  --cutoff 2024-07-01 --offline --output volatility_report.json
```

Post schema:

| Field | Contract |
| --- | --- |
| `post_id` | Nonempty identifier. Earliest observed snapshot wins. |
| `subreddit` | wallstreetbets, stocks, investing, or options, with optional r/ prefix. |
| `created_utc` | UTC timestamp or numeric Unix seconds. |
| `observed_at` | UTC timestamp when text and engagement counts were collected. |
| `text` | Scraped text, or provide `title` and `selftext`. |
| `score`, `num_comments` | Engagement counts known at observed_at. |

Default filters require 40 characters, score of at least 2, and at least 1 comment. URL/user mentions, removed text, discussion megathreads, and selected promotional noise are filtered. Text and counts must reflect the historical snapshot, not later edits or final popularity.

Prices require `timestamp` and positive `close` for one instrument on a complete UTC hourly grid. Duplicate hours and missing hours fail validation. The pipeline does not forward-fill closed equity markets. For exchange-session data, first define a separate trading-time research protocol instead of calling 48 trading bars 48 elapsed hours.

Each post becomes usable at the next hourly boundary after observation. Vectors are averaged across available posts within each decision hour. Each hour receives one market label, regardless of post count or subreddit. The target is:

```text
trailing_rv(t) = sqrt(sum(log_return(i)^2 for i in (t-48h, t]))
forward_rv(t)  = sqrt(sum(log_return(i)^2 for i in (t, t+48h]))
spike(t)       = forward_rv(t) > 1.5 * trailing_rv(t)
```

The ratio threshold is fixed in advance. Rows without a full history/future or with zero trailing RV are excluded. Training rows whose target ends at or after the cutoff are purged. This also keeps the holdout's overlapping outcomes out of the training set. Test windows can overlap one another, so raw row counts are not independent sample counts.

Dense vectors have 1,536 coordinates. Offline mode uses deterministic per-text pseudorandom unit vectors and is explicitly recorded in reports. It validates execution only. It cannot establish semantic predictive power. No API key is read, and no paid request is issued implicitly.

For real embeddings, replace `--offline` with `--embedding-results completed_batch.jsonl`. `OpenAIBatchEmbeddings` validates result IDs, HTTP status, model, dimensions, and finite values, then restores input order. Missing, duplicate, unknown, or failed requests stop execution. The CLI does not submit paid requests.

`format_embedding_requests` generates OpenAI Batch API request records with `text-embedding-3-small` and explicit dimensions. Serialize each record as one JSON line. Generate requests from the filtered posts, and keep that input snapshot unchanged until results are ingested. Completed result IDs must exactly cover those posts.

Alternatively, inject a callable using the OpenAI embeddings API or another encoder. The callable receives a sequence of texts and returns a finite `(batch_size, 1536)` matrix in request order. Supply a stable `provider_id` containing the provider/model/preprocessing revision. Injected providers manage their own retries and rate limits. Provider exceptions propagate without silently replacing real vectors with simulated ones.

Caches use parquet through PyArrow. Keys include text, provider identity, model, and dimensions. Batch writes use atomic replacement, and reads validate keys and shapes. Changed batch boundaries create separate files. Offline and real vectors cannot share cache entries.

PCA uses centered full SVD, with no whitening or coordinate scaling. It optimizes global linear variance within the training corpus. It does not optimize local neighborhoods. The default retains at least 95% of training variance. Held-out vectors only use `transform`. Classifier scaling occurs after PCA, inside each fitted model.

Models: Logistic Regression, RBF SVM, and Random Forest. SVM evaluation uses decision margins without probability calibration. Reports include ROC-AUC, average precision (the chosen PR-AUC convention), prevalence, confusion matrices, and permutation importance by global component. Logistic coefficients and forest impurity importance are also included. Permutation importance uses the holdout only for descriptive attribution, never feature selection or refitting.

The supplied 0.55 historical price-only AUC is an unverified external reference. Reported differences are descriptive. No matched price-only model, significance test, or claim of outperforming the historical study is implied. Synthetic tests make no empirical performance claim.

## Verification

Tests cover filtering, cache integrity, embedding batch order, PCA isolation, exact forward windows, temporal purging, metrics, and both CLI modes. GitHub Actions runs the suite, Ruff, and mypy on Python 3.11 and 3.12. Warnings fail the test suite.

## References

- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
