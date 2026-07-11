from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantloom import _talib_compat
from quantloom.config import Direction
from quantloom.config.schema import IndicatorConfig
from quantloom.indicators.core import (
    candlestick_signal,
    compute_indicators,
    rolling_volatility,
    rsi,
    stochastic,
)


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.Series(values, index=index, dtype=float)


def test_rsi_is_bounded_and_named() -> None:
    close = _series([float(x) for x in range(1, 21)])  # strictly increasing -> RSI should hit 100
    result = rsi(close, length=5)

    assert result.name == "rsi"
    assert result.iloc[-1] == pytest.approx(100.0)
    assert result.iloc[:4].isna().all()  # warmup period


def test_rolling_volatility_matches_hand_computed_std() -> None:
    close = _series([100.0, 101.0, 99.0, 102.0, 98.0, 103.0])
    length = 3

    result = rolling_volatility(close, length)

    pct_change = close.pct_change()
    expected_last = pct_change.iloc[-3:].std(ddof=1)
    assert result.iloc[:3].isna().all()  # pct_change's own NaN + rolling window not yet full
    assert result.iloc[-1] == pytest.approx(expected_last)
    assert result.name == "volatility"


def test_stochastic_returns_k_and_d_columns_matching_talib_directly() -> None:
    close = _series([float(x) for x in range(1, 21)])
    high = close + 1
    low = close - 1

    result = stochastic(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    expected_k, expected_d = _talib_compat.stoch(
        high, low, close, fastk_period=5, slowk_period=3, slowd_period=3
    )

    assert list(result.columns) == ["stoch_k", "stoch_d"]
    pd.testing.assert_series_equal(result["stoch_k"], expected_k, check_names=False)
    pd.testing.assert_series_equal(result["stoch_d"], expected_d, check_names=False)


def test_candlestick_signal_matches_the_or_of_its_component_patterns() -> None:
    # A plausible-looking OHLC series (not hand-crafted to match any specific pattern) --
    # this tests the OR-across-patterns/boolean-conversion logic, not talib's own pattern math.
    rng = np.random.default_rng(seed=1)
    n = 30
    close_values = 100 + rng.normal(scale=1.5, size=n).cumsum()
    open_ = _series((close_values + rng.normal(scale=0.5, size=n)).tolist())
    close = _series(close_values.tolist())
    high = _series(np.maximum(open_, close) + np.abs(rng.normal(scale=0.3, size=n)))
    low = _series(np.minimum(open_, close) - np.abs(rng.normal(scale=0.3, size=n)))

    result = candlestick_signal(open_, high, low, close, Direction.LONG)

    expected = pd.Series(False, index=close.index)
    for pattern_name, target_value in [
        ("CDLHAMMER", 100),
        ("CDLENGULFING", 100),
        ("CDLMORNINGSTAR", 100),
        ("CDL3WHITESOLDIERS", 100),
    ]:
        values = _talib_compat.candle_pattern(pattern_name, open_, high, low, close)
        expected = expected | (values == target_value)

    assert result.name == "candlestick_long"
    assert result.dtype == bool
    pd.testing.assert_series_equal(result, expected, check_names=False)
    assert result.any(), "fixture should trigger at least one pattern to make this test meaningful"


def test_compute_indicators_only_produces_configured_directions() -> None:
    n = 40
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed=0)
    close = pd.Series(100 + rng.normal(size=n).cumsum(), index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
        },
        index=index,
    )
    config = IndicatorConfig(
        rsi_length=5,
        stochastic_fastk_period=5,
        stochastic_slowk_period=3,
        stochastic_slowd_period=3,
    )

    result = compute_indicators(frame, config, frozenset({Direction.LONG}))

    assert set(result.columns) == {
        "rsi",
        "stoch_k",
        "stoch_d",
        "candlestick_long",
    }
    assert len(result) == n
