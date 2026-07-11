from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantloom.config.schema import Config, Direction, UniverseConfig


@pytest.fixture
def raw_ohlcv_frame():
    """Factory fixture: synthetic OHLCV frame with a random-walk close, shared by the
    indicators/signals/strategy pipeline-stage tests that all need the same shape."""

    def _make(n: int = 120, seed: int = 0) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=n, freq="h")
        rng = np.random.default_rng(seed=seed)
        close = pd.Series(100 + rng.normal(size=n).cumsum(), index=index)
        return pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000},
            index=index,
        )

    return _make


@pytest.fixture
def make_config():
    """Factory fixture: minimal valid Config for a given direction set, at packaged-default
    values otherwise."""

    def _make(directions: frozenset[Direction]) -> Config:
        return Config(
            data_dir="./data",
            directions=directions,
            universe=UniverseConfig(
                start_date="2024-01-01", train_test_split_date="2024-03-01", end_date="2024-06-01"
            ),
        )

    return _make
