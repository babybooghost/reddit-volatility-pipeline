"""Deterministic synthetic inputs for pipeline invariants, not performance claims."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    count = 1100
    sigma = np.where((np.arange(count) // 70) % 2 == 0, 0.001, 0.009)
    returns = rng.normal(size=count) * sigma
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC"),
            "close": 100 * np.exp(np.cumsum(returns)),
        }
    )


@pytest.fixture
def posts(prices: pd.DataFrame) -> pd.DataFrame:
    times = prices.timestamp.iloc[70:-70:6].reset_index(drop=True)
    return pd.DataFrame(
        {
            "post_id": [f"post_{i}" for i in range(len(times))],
            "subreddit": np.resize(
                ["r/wallstreetbets", "stocks", "investing", "options"], len(times)
            ),
            "created_utc": times - pd.Timedelta(minutes=20),
            "observed_at": times,
            "text": [
                f"Market earnings and options positioning change risk expectations for portfolio number {i}."
                for i in range(len(times))
            ],
            "score": 10,
            "num_comments": 3,
        }
    )
