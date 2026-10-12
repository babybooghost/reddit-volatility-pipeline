"""Point-in-time Reddit embeddings, global PCA, and forward volatility models.

Offline embeddings are deterministic fixtures, not OpenAI semantic embeddings.
PCA centers dense vectors without whitening or coordinate standardization, so
its components preserve global linear variance within the training corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from research_utils import (
    FloatArray,
    FrameSource,
    binary_labels,
    classification_report,
    cutoff_utc,
    matrix,
    read_frame,
    require_columns,
    utc,
    write_json,
)

LOGGER = logging.getLogger(__name__)
EmbeddingProvider = Callable[[Sequence[str]], Any]


class OpenAIBatchEmbeddings:
    """Validate completed OpenAI Batch JSONL and restore request order by ID."""

    def __init__(self, posts: pd.DataFrame, result_path: str | Path) -> None:
        """Require exactly one valid result for each filtered post."""
        require_columns(posts, ["post_id", "text"])
        if posts.post_id.astype(str).duplicated().any():
            raise ValueError("Batch request IDs must be unique")
        lookup = dict(zip(posts.post_id.astype(str), posts.text.astype(str), strict=True))
        self.vectors: dict[str, FloatArray] = {}
        seen: set[str] = set()
        content = Path(result_path).read_bytes()
        self.provider_id = "openai-batch-" + hashlib.sha256(content).hexdigest()
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                identifier = str(result["custom_id"])
                if identifier not in lookup or identifier in seen:
                    raise ValueError("Unknown or duplicate batch result ID")
                response = result["response"]
                if result.get("error") or response["status_code"] != 200:
                    raise ValueError(f"Embedding request failed for {identifier}")
                body = response["body"]
                if (
                    body["model"] != RedditVolatilityPipeline.EMBEDDING_MODEL
                    or len(body["data"]) != 1
                ):
                    raise ValueError("Unexpected embedding model or result count")
                vector = matrix([body["data"][0]["embedding"]], width=1536)[0]
                text = lookup[identifier]
                if text in self.vectors and not np.allclose(
                    self.vectors[text], vector, rtol=1e-6, atol=1e-8
                ):
                    raise ValueError("Identical texts have inconsistent batch vectors")
                self.vectors[text] = vector
                seen.add(identifier)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid batch result at line {line_number}: {error}") from error
        if seen != set(lookup):
            raise ValueError(f"Batch results missing {len(set(lookup) - seen)} request IDs")

    def __call__(self, texts: Sequence[str]) -> FloatArray:
        """Return validated vectors aligned to a requested text batch."""
        try:
            return np.stack([self.vectors[text] for text in texts])
        except KeyError as error:
            raise ValueError("Text is missing from completed batch results") from error


class RedditVolatilityPipeline:
    """Research pipeline with explicit provider identity and purged OOT evaluation."""

    EMBEDDING_MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    SUBREDDITS = frozenset({"wallstreetbets", "stocks", "investing", "options"})
    NOISE_PATTERN = (
        r"(?i)\b(?:daily discussion|weekend discussion|join (?:my|our) discord|referral code)\b"
    )

    def __init__(
        self,
        cache_dir: str | Path = ".cache/embeddings",
        *,
        embedding_provider: EmbeddingProvider | None = None,
        provider_id: str | None = None,
        batch_size: int = 128,
        min_length: int = 40,
        min_score: int = 2,
        min_comments: int = 1,
        n_components: int | float = 0.95,
        random_state: int = 42,
    ) -> None:
        """Configure signal filters, vector caching, and global PCA retention."""
        if batch_size < 1 or min_length < 1 or min_comments < 0:
            raise ValueError("Invalid batch size or filter thresholds")
        if isinstance(n_components, bool) or not (
            isinstance(n_components, int)
            and n_components >= 1
            or isinstance(n_components, float)
            and 0 < n_components < 1
        ):
            raise ValueError("n_components must be a positive integer or variance fraction")
        if embedding_provider is not None and not provider_id:
            raise ValueError("Custom providers need a stable provider_id for cache isolation")
        self.cache_dir = Path(cache_dir)
        self.embedding_provider = embedding_provider
        self.provider_id = (
            provider_id if embedding_provider else f"offline-sha256-normal-v1-seed-{random_state}"
        )
        self.batch_size = batch_size
        self.min_length = min_length
        self.min_score = min_score
        self.min_comments = min_comments
        self.n_components = n_components
        self.random_state = random_state
        self.pca: PCA | None = None
        self.models: dict[str, Any] = {}

    def ingest_posts(self, source: FrameSource) -> pd.DataFrame:
        """Filter snapshots using engagement known at observed_at, never final counts."""
        frame = read_frame(source)
        require_columns(
            frame, ["post_id", "subreddit", "created_utc", "observed_at", "score", "num_comments"]
        )
        if "text" not in frame and not {"title", "selftext"}.issubset(frame.columns):
            raise ValueError("Provide text or both title and selftext")
        if frame.post_id.isna().any() or frame.post_id.astype(str).str.strip().eq("").any():
            raise ValueError("post_id cannot be missing or blank")
        frame["post_id"] = frame.post_id.astype(str)
        created = frame["created_utc"]
        frame["created_utc"] = (
            utc(pd.to_datetime(created, unit="s", utc=True))
            if pd.api.types.is_numeric_dtype(created)
            else utc(created)
        )
        frame["observed_at"] = utc(frame.observed_at)
        if (frame.observed_at < frame.created_utc).any():
            raise ValueError("observed_at must be at or after creation")
        frame["subreddit"] = (
            frame.subreddit.astype("string").str.lower().str.replace(r"^r/", "", regex=True)
        )
        raw = (
            frame["text"]
            if "text" in frame
            else frame.title.fillna("") + "\n" + frame.selftext.fillna("")
        )
        noise = raw.fillna("").str.contains(self.NOISE_PATTERN, regex=True)
        frame["text"] = (
            raw.fillna("")
            .astype(str)
            .str.replace(r"https?://\S+|www\.\S+|\bu/\w+", " ", regex=True)
            .str.replace(r"\[(?:removed|deleted)\]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        for column in ("score", "num_comments"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"{column} must be finite")
        if (frame.num_comments < 0).any():
            raise ValueError("num_comments cannot be negative")
        # Earliest snapshot wins before filtering; later popularity cannot rescue it.
        frame = (
            frame.assign(_noise=noise)
            .sort_values(["observed_at", "post_id"], kind="stable")
            .drop_duplicates("post_id")
        )
        keep = (
            frame.subreddit.isin(self.SUBREDDITS)
            & ~frame._noise
            & frame.text.str.len().ge(self.min_length)
            & frame.text.str.contains(r"[A-Za-z]{3}", regex=True)
            & frame.score.ge(self.min_score)
            & frame.num_comments.ge(self.min_comments)
        )
        frame = frame.loc[keep].drop(columns="_noise").reset_index(drop=True)
        if frame.empty:
            raise ValueError("No posts passed the signal filters")
        frame["prediction_at"] = frame.observed_at.dt.ceil("h")
        LOGGER.info("Retained %d posts across %d subreddits", len(frame), frame.subreddit.nunique())
        return frame

    def format_embedding_requests(self, posts: pd.DataFrame) -> list[dict[str, Any]]:
        """Build OpenAI Batch API JSONL records with stable IDs and explicit dimensions."""
        require_columns(posts, ["post_id", "text"])
        return [
            {
                "custom_id": str(row.post_id),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": self.EMBEDDING_MODEL,
                    "input": row.text,
                    "dimensions": self.DIMENSIONS,
                },
            }
            for row in posts[["post_id", "text"]].itertuples(index=False)
        ]

    def _key(self, text: str) -> str:
        """Hash the complete encoder identity and normalized input text."""
        payload = json.dumps(
            [self.provider_id, self.EMBEDDING_MODEL, self.DIMENSIONS, text], ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _offline_embeddings(self, texts: Sequence[str]) -> FloatArray:
        """Generate reproducible simulated unit vectors without fitting the corpus."""
        vectors = np.stack(
            [
                np.random.default_rng(int(self._key(text)[:16], 16)).standard_normal(
                    self.DIMENSIONS
                )
                for text in texts
            ]
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def cache_embeddings(self, posts: pd.DataFrame) -> FloatArray:
        """Cache immutable content-addressed batches as parquet, preserving row order.

        Providers must return a matrix in request order. Errors never silently
        switch real embeddings to offline vectors. Concurrent identical writes
        are atomic; stale content and provider changes use separate cache keys.
        """
        require_columns(posts, ["text"])
        if posts.text.isna().any() or posts.text.astype(str).str.strip().eq("").any():
            raise ValueError("Embedding text cannot be empty")
        texts = posts.text.astype(str).tolist()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        collected: list[FloatArray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            keys = [self._key(text) for text in batch]
            digest = hashlib.sha256("".join(keys).encode()).hexdigest()
            path = self.cache_dir / f"{digest}.parquet"
            if path.exists():
                cached = pd.read_parquet(path)
                require_columns(cached, ["key", "vector"])
                if cached.key.tolist() != keys:
                    raise ValueError(f"Cache key mismatch: {path}")
                vectors = matrix(np.stack(cached.vector.tolist()), width=self.DIMENSIONS)
            else:
                provider = self.embedding_provider or self._offline_embeddings
                vectors = matrix(provider(batch), width=self.DIMENSIONS)
                if len(vectors) != len(batch):
                    raise ValueError("Embedding provider returned wrong batch length")
                descriptor, temporary = tempfile.mkstemp(dir=self.cache_dir, suffix=".parquet.tmp")
                os.close(descriptor)
                try:
                    pd.DataFrame({"key": keys, "vector": list(vectors)}).to_parquet(
                        temporary, index=False
                    )
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            if len(vectors) != len(batch):
                raise ValueError("Cached vector count mismatch")
            collected.append(vectors)
        LOGGER.info("Loaded %d embeddings (%s)", len(texts), self.provider_id)
        return np.vstack(collected)

    def apply_pca(self, embeddings: Any, *, fit: bool = False) -> FloatArray:
        """Fit on training vectors only; transform held-out vectors without refitting."""
        vectors = matrix(embeddings, width=self.DIMENSIONS)
        if fit:
            if len(vectors) < 2 or not np.any(np.var(vectors, axis=0) > 0):
                raise ValueError("PCA requires at least two nonidentical vectors")
            if isinstance(self.n_components, int) and self.n_components > min(vectors.shape):
                raise ValueError("n_components exceeds training matrix rank bound")
            self.pca = PCA(n_components=self.n_components, svd_solver="full", whiten=False)
            return np.asarray(self.pca.fit_transform(vectors), dtype=float)
        if self.pca is None:
            raise RuntimeError("Fit PCA on training vectors before transforming")
        return np.asarray(self.pca.transform(vectors), dtype=float)

    @staticmethod
    def build_volatility_targets(
        prices: FrameSource, *, spike_multiple: float = 1.5
    ) -> pd.DataFrame:
        """Define 48h RV as sqrt(sum(log-return squared)) on a complete hourly grid.

        A spike exceeds spike_multiple times trailing 48h RV. The rule uses no
        fitted full-sample quantiles. Missing bars are rejected, never filled.
        Input covers one instrument, in elapsed hours including weekends.
        """
        if not np.isfinite(spike_multiple) or spike_multiple <= 1:
            raise ValueError("spike_multiple must be finite and greater than one")
        prices = read_frame(prices)
        require_columns(prices, ["timestamp", "close"])
        prices["timestamp"] = utc(prices.timestamp)
        prices = prices.sort_values("timestamp").reset_index(drop=True)
        close = pd.to_numeric(prices.close, errors="raise").to_numpy(dtype=float)
        if len(close) < 97 or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("Need at least 97 finite positive hourly prices")
        times = prices.timestamp
        if (
            not times.eq(times.dt.floor("h")).all()
            or not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
        ):
            raise ValueError("Prices must form a unique, complete UTC hourly grid")
        squares = np.square(np.diff(np.log(close)))
        cumulative = np.r_[0.0, np.cumsum(squares)]
        positions = np.arange(48, len(close) - 48)
        past = np.sqrt(np.maximum(0, cumulative[positions] - cumulative[positions - 48]))
        future = np.sqrt(np.maximum(0, cumulative[positions + 48] - cumulative[positions]))
        valid = past > np.finfo(float).eps
        positions, past, future = positions[valid], past[valid], future[valid]
        if not len(positions):
            raise ValueError("No targets with nonzero trailing volatility")
        return pd.DataFrame(
            {
                "prediction_at": times.iloc[positions].to_numpy(),
                "label_end": times.iloc[positions + 48].to_numpy(),
                "trailing_rv": past,
                "forward_rv": future,
                "target": (future > spike_multiple * past).astype(int),
            }
        )

    def train_classifiers(
        self, x_train: Any, y_train: Any, x_test: Any, y_test: Any
    ) -> dict[str, Any]:
        """Fit fixed models; report held-out metrics and permutation component attribution."""
        train, test = matrix(x_train), matrix(x_test)
        if train.shape[1] != test.shape[1]:
            raise ValueError("Train and test feature widths differ")
        train_y = binary_labels(y_train, len(train))
        test_y = binary_labels(y_test, len(test))
        self.models = {
            "logistic_regression": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=5000, random_state=self.random_state)
            ),
            "svm": make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=1.0, probability=False, random_state=self.random_state),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3, random_state=self.random_state, n_jobs=1
            ),
        }
        reports: dict[str, Any] = {}
        for name, model in self.models.items():
            model.fit(train, train_y)
            scores = (
                model.decision_function(test) if name == "svm" else model.predict_proba(test)[:, 1]
            )
            report = classification_report(test_y, scores, threshold=0.0 if name == "svm" else 0.5)
            attribution = permutation_importance(
                model,
                test,
                test_y,
                scoring="roc_auc",
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=1,
            )
            report["component_attribution"] = {
                f"PC{i + 1}": {"auc_decrease": float(mean), "std": float(std)}
                for i, (mean, std) in enumerate(
                    zip(attribution.importances_mean, attribution.importances_std, strict=True)
                )
            }
            if name == "logistic_regression":
                report["standardized_component_coefficients"] = model[-1].coef_[0].tolist()
            elif name == "random_forest":
                report["component_impurity_importance"] = model.feature_importances_.tolist()
            reports[name] = report
            LOGGER.info("%s OOT ROC-AUC %.4f", name, report["roc_auc"])
        return reports

    @staticmethod
    def benchmark_against_baseline(
        reports: dict[str, Any], baseline_auc: float = 0.55
    ) -> dict[str, Any]:
        """Compare to a supplied historical reference, not a paired statistical test."""
        if not np.isfinite(baseline_auc) or not 0 <= baseline_auc <= 1:
            raise ValueError("baseline_auc must be within [0, 1]")
        return {
            name: {
                "historical_price_only_auc": baseline_auc,
                "auc_difference": report["roc_auc"] - baseline_auc,
                "exceeds_reference": bool(report["roc_auc"] > baseline_auc),
                "comparison": "unpaired historical reference; no significance claim",
            }
            for name, report in reports.items()
        }

    def run(
        self, posts: FrameSource, prices: FrameSource, *, cutoff: str, spike_multiple: float = 1.5
    ) -> dict[str, Any]:
        """Aggregate snapshots per decision hour, purge overlapping labels, fit, evaluate."""
        clean = self.ingest_posts(posts)
        embeddings = self.cache_embeddings(clean)
        targets = self.build_volatility_targets(prices, spike_multiple=spike_multiple)
        vectors = pd.DataFrame(embeddings)
        vectors["prediction_at"] = clean.prediction_at.to_numpy()
        hourly = vectors.groupby("prediction_at", sort=True).mean()
        joined = hourly.join(targets.set_index("prediction_at"), how="inner")
        boundary = cutoff_utc(cutoff)
        train = joined.loc[(joined.index < boundary) & (joined.label_end < boundary)]
        test = joined.loc[joined.index >= boundary]
        if train.empty or test.empty:
            raise ValueError("No aligned train/test samples after 48h purge")
        columns = list(range(self.DIMENSIONS))
        train_y = binary_labels(train.target, len(train))
        test_y = binary_labels(test.target, len(test))
        x_train = self.apply_pca(train[columns], fit=True)
        x_test = self.apply_pca(test[columns])
        reports = self.train_classifiers(x_train, train_y, x_test, test_y)
        assert self.pca is not None
        return {
            "embedding_provider": self.provider_id,
            "offline": self.embedding_provider is None,
            "train_samples": len(train),
            "test_samples": len(test),
            "purged_samples": int(
                ((joined.index < boundary) & (joined.label_end >= boundary)).sum()
            ),
            "cutoff": boundary.isoformat(),
            "horizon_hours": 48,
            "spike_multiple": spike_multiple,
            "pca_explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "models": reports,
            "baseline_comparison": self.benchmark_against_baseline(reports),
        }


def main() -> None:
    """Run offline or completed-batch embeddings through the full research workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", type=Path, default=Path("volatility_report.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/embeddings"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--offline",
        action="store_true",
        help="Acknowledge simulated embeddings; no semantic performance claims",
    )
    mode.add_argument(
        "--embedding-results",
        type=Path,
        help="Completed OpenAI Batch JSONL matching filtered post IDs",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        pipeline = RedditVolatilityPipeline(args.cache_dir)
        if args.embedding_results:
            provider = OpenAIBatchEmbeddings(
                pipeline.ingest_posts(args.posts), args.embedding_results
            )
            pipeline = RedditVolatilityPipeline(
                args.cache_dir, embedding_provider=provider, provider_id=provider.provider_id
            )
        report = pipeline.run(args.posts, args.prices, cutoff=args.cutoff)
        write_json(args.output, report)
    except (ValueError, OSError, RuntimeError) as error:
        LOGGER.error("Pipeline failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
