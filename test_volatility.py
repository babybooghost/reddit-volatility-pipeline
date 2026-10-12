"""Data filtering, cache integrity, PCA leakage, and forward-window regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volatility_pipeline import RedditVolatilityPipeline


def test_ingestion_filters_and_earliest_snapshot(posts: pd.DataFrame) -> None:
    dirty = posts.iloc[:8].copy()
    dirty.loc[0, "text"] = "[removed]"
    dirty.loc[1, "subreddit"] = "cats"
    dirty.loc[2, "text"] = (
        "Join our discord for all the best trade ideas and advice on the market today"
    )
    dirty.loc[3, "score"] = 0
    dirty.loc[4, "num_comments"] = 0
    dirty.loc[5, "text"] = "x" * 8
    duplicate = dirty.iloc[[3]].assign(
        score=100, observed_at=dirty.observed_at.iloc[3] + pd.Timedelta(days=1)
    )
    clean = RedditVolatilityPipeline().ingest_posts(pd.concat([dirty, duplicate]))
    assert clean.post_id.tolist() == dirty.post_id.iloc[6:].tolist()
    assert clean.subreddit.isin(RedditVolatilityPipeline.SUBREDDITS).all()
    assert (clean.prediction_at >= clean.observed_at).all()


@pytest.mark.parametrize(
    "failure", ["missing_observation", "early_observation", "bad_score", "empty", "bad_id"]
)
def test_bad_ingestion(posts: pd.DataFrame, failure: str) -> None:
    if failure == "missing_observation":
        posts = posts.drop(columns="observed_at")
    elif failure == "early_observation":
        posts.loc[0, "observed_at"] = pd.Timestamp("2000", tz="UTC")
    elif failure == "bad_score":
        posts["score"] = posts.score.astype(float)
        posts.loc[0, "score"] = np.inf
    elif failure == "bad_id":
        posts.loc[0, "post_id"] = None
    else:
        posts = posts.iloc[:0]
    with pytest.raises(ValueError):
        RedditVolatilityPipeline().ingest_posts(posts)


def test_batch_format(posts: pd.DataFrame) -> None:
    pipeline = RedditVolatilityPipeline()
    clean = pipeline.ingest_posts(posts)
    request = pipeline.format_embedding_requests(clean)[0]
    assert request["custom_id"] == clean.post_id.iloc[0]
    assert request["body"]["model"] == "text-embedding-3-small"
    assert request["body"]["dimensions"] == 1536


def test_cache_roundtrip_order_and_content(posts: pd.DataFrame, tmp_path: Path) -> None:
    pipeline = RedditVolatilityPipeline(tmp_path, batch_size=3)
    clean = pipeline.ingest_posts(posts.iloc[:7])
    first = pipeline.cache_embeddings(clean)
    second = pipeline.cache_embeddings(clean)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (7, 1536)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1)
    reversed_vectors = pipeline.cache_embeddings(clean.iloc[::-1])
    np.testing.assert_array_equal(reversed_vectors, first[::-1])
    edited = clean.copy()
    edited.loc[0, "text"] += " new material"
    updated = pipeline.cache_embeddings(edited)
    assert not np.array_equal(updated[0], first[0])
    np.testing.assert_array_equal(updated[1:], first[1:])
    assert not list(tmp_path.glob("*.tmp"))


def test_provider_called_once_and_cache_isolation(posts: pd.DataFrame, tmp_path: Path) -> None:
    calls: list[int] = []

    def provider(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return np.ones((len(texts), 1536))

    pipeline = RedditVolatilityPipeline(
        tmp_path, embedding_provider=provider, provider_id="real-v1"
    )
    pipeline.cache_embeddings(posts.iloc[:3])
    pipeline.cache_embeddings(posts.iloc[:3])
    assert calls == [3]
    offline = RedditVolatilityPipeline(tmp_path).cache_embeddings(posts.iloc[:3])
    assert not np.all(offline == 1)


@pytest.mark.parametrize("bad", [np.ones((2, 5)), np.ones((1, 1536)), np.full((2, 1536), np.nan)])
def test_provider_validation(posts: pd.DataFrame, tmp_path: Path, bad: np.ndarray) -> None:
    pipeline = RedditVolatilityPipeline(
        tmp_path, embedding_provider=lambda _: bad, provider_id="broken"
    )
    with pytest.raises(ValueError):
        pipeline.cache_embeddings(posts.iloc[:2])
    assert not list(tmp_path.glob("*.parquet"))


def test_corrupt_cache_rejected(posts: pd.DataFrame, tmp_path: Path) -> None:
    pipeline = RedditVolatilityPipeline(tmp_path)
    pipeline.cache_embeddings(posts.iloc[:2])
    path = next(tmp_path.glob("*.parquet"))
    cache = pd.read_parquet(path)
    cache.loc[0, "key"] = "wrong"
    cache.to_parquet(path)
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.cache_embeddings(posts.iloc[:2])


def test_pca_preserves_global_variance_without_test_leakage() -> None:
    rng = np.random.default_rng(4)
    x = np.zeros((100, 1536))
    x[:, :3] = rng.normal(size=(100, 3)) * [100, 2, 0.1]
    pipeline = RedditVolatilityPipeline(n_components=0.95)
    with pytest.raises(RuntimeError):
        pipeline.apply_pca(x)
    components = pipeline.apply_pca(x, fit=True)
    assert components.shape == (100, 1)
    assert pipeline.pca.explained_variance_ratio_.sum() >= 0.95
    assert abs(pipeline.pca.components_[0, 0]) > 0.99
    original_mean = pipeline.pca.mean_.copy()
    original_components = pipeline.pca.components_.copy()
    pipeline.apply_pca(x + 1000)
    np.testing.assert_array_equal(pipeline.pca.mean_, original_mean)
    np.testing.assert_array_equal(pipeline.pca.components_, original_components)
    assert not pipeline.pca.whiten


def test_forward_targets_exact_window(prices: pd.DataFrame) -> None:
    targets = RedditVolatilityPipeline.build_volatility_targets(prices)
    row = targets.iloc[23]
    index = int(prices.index[prices.timestamp.eq(row.prediction_at)][0])
    returns = np.diff(np.log(prices.close.to_numpy()))
    assert row.trailing_rv == pytest.approx(np.sqrt(np.sum(returns[index - 48 : index] ** 2)))
    assert row.forward_rv == pytest.approx(np.sqrt(np.sum(returns[index : index + 48] ** 2)))
    assert row.label_end - row.prediction_at == pd.Timedelta(hours=48)
    assert row.target == int(row.forward_rv > 1.5 * row.trailing_rv)
    assert targets.prediction_at.max() == prices.timestamp.iloc[-49]
    modified = prices.copy()
    modified.loc[index + 49 :, "close"] *= 10
    same = RedditVolatilityPipeline.build_volatility_targets(modified).iloc[23]
    assert same.forward_rv == pytest.approx(row.forward_rv)


@pytest.mark.parametrize("failure", ["gap", "duplicate", "zero", "infinite", "short", "constant"])
def test_bad_prices(prices: pd.DataFrame, failure: str) -> None:
    if failure == "gap":
        prices = prices.drop(index=50)
    elif failure == "duplicate":
        prices = pd.concat([prices, prices.iloc[[50]]])
    elif failure == "zero":
        prices.loc[50, "close"] = 0
    elif failure == "infinite":
        prices.loc[50, "close"] = np.inf
    elif failure == "constant":
        prices.loc[:, "close"] = 100
    else:
        prices = prices.iloc[:50]
    with pytest.raises(ValueError):
        RedditVolatilityPipeline.build_volatility_targets(prices)


def test_all_classifiers_and_reference() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
    pipeline = RedditVolatilityPipeline()
    result = pipeline.train_classifiers(x[:65], y[:65], x[65:], y[65:])
    assert set(result) == {"logistic_regression", "svm", "random_forest"}
    baseline = pipeline.benchmark_against_baseline(result)
    for name, report in result.items():
        assert 0 <= report["roc_auc"] <= 1
        assert 0 <= report["pr_auc"] <= 1
        assert np.asarray(report["confusion_matrix"]).sum() == 35
        assert len(report["component_attribution"]) == 4
        assert baseline[name]["auc_difference"] == pytest.approx(report["roc_auc"] - 0.55)
    with pytest.raises(ValueError, match="Both classes"):
        pipeline.train_classifiers(x[:65], np.zeros(65), x[65:], y[65:])


def test_end_to_end_purge_and_training_only_pca(
    posts: pd.DataFrame, prices: pd.DataFrame, tmp_path: Path
) -> None:
    pipeline = RedditVolatilityPipeline(tmp_path, n_components=5)
    cutoff = prices.timestamp.iloc[650]
    report = pipeline.run(posts, prices, cutoff=cutoff.isoformat())
    assert report["offline"] is True
    assert report["purged_samples"] > 0
    assert report["train_samples"] > 0 and report["test_samples"] > 0
    clean = pipeline.ingest_posts(posts)
    expected = clean.loc[clean.prediction_at + pd.Timedelta(hours=48) < cutoff]
    expected_vectors = pipeline.cache_embeddings(expected)
    np.testing.assert_allclose(pipeline.pca.mean_, expected_vectors.mean(axis=0), atol=1e-12)
    assert pipeline.pca.n_samples_ == report["train_samples"]


def test_hourly_aggregation_prevents_duplicate_targets(
    posts: pd.DataFrame, prices: pd.DataFrame, tmp_path: Path
) -> None:
    doubled = pd.concat([posts, posts.assign(post_id=posts.post_id + "copy")], ignore_index=True)
    pipeline = RedditVolatilityPipeline(tmp_path, n_components=2)
    report = pipeline.run(doubled, prices, cutoff=prices.timestamp.iloc[650].isoformat())
    assert report["train_samples"] + report["test_samples"] + report["purged_samples"] == len(posts)


def test_real_batch_results_restored_by_id(posts: pd.DataFrame, tmp_path: Path) -> None:
    import json

    from volatility_pipeline import OpenAIBatchEmbeddings

    selected = posts.iloc[:3]
    records = [
        {
            "custom_id": str(row.post_id),
            "error": None,
            "response": {
                "status_code": 200,
                "body": {
                    "model": "text-embedding-3-small",
                    "data": [{"embedding": [float(i + 1)] * 1536}],
                },
            },
        }
        for i, row in enumerate(selected.itertuples())
    ]
    path = tmp_path / "batch.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records[::-1]), encoding="utf-8")
    provider = OpenAIBatchEmbeddings(selected, path)
    pipeline = RedditVolatilityPipeline(
        tmp_path / "cache", embedding_provider=provider, provider_id=provider.provider_id
    )
    result = pipeline.cache_embeddings(selected)
    np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])
    path.write_text(json.dumps(records[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        OpenAIBatchEmbeddings(selected, path)
    records[0]["response"]["status_code"] = 429
    path.write_text(json.dumps(records[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="failed"):
        OpenAIBatchEmbeddings(selected, path)


@pytest.mark.parametrize("mode", ["offline", "batch"])
def test_cli_end_to_end(
    posts: pd.DataFrame, prices: pd.DataFrame, tmp_path: Path, mode: str
) -> None:
    import json
    import subprocess
    import sys

    posts.to_csv(tmp_path / "posts.csv", index=False)
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    arguments = [
        sys.executable,
        str(Path(__file__).with_name("volatility_pipeline.py")),
        "--posts",
        str(tmp_path / "posts.csv"),
        "--prices",
        str(tmp_path / "prices.parquet"),
        "--cutoff",
        prices.timestamp.iloc[650].isoformat(),
        "--output",
        str(tmp_path / "report.json"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    if mode == "offline":
        arguments.append("--offline")
    else:
        clean = RedditVolatilityPipeline().ingest_posts(posts)
        # Low-rank vectors exercise batch ingestion and PCA without large fixtures.
        rng = np.random.default_rng(31)
        basis = rng.normal(size=(4, 1536))
        vectors = rng.normal(size=(len(clean), 4)) @ basis
        records = [
            {
                "custom_id": str(identifier),
                "error": None,
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "text-embedding-3-small",
                        "data": [{"embedding": vector.tolist()}],
                    },
                },
            }
            for identifier, vector in zip(clean.post_id, vectors, strict=True)
        ]
        result_path = tmp_path / "results.jsonl"
        result_path.write_text(
            "\n".join(json.dumps(record) for record in records[::-1]), encoding="utf-8"
        )
        arguments += ["--embedding-results", str(result_path)]
    result = subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["offline"] is (mode == "offline")
    assert set(report["models"]) == {"logistic_regression", "svm", "random_forest"}
    assert "Traceback" not in result.stderr
