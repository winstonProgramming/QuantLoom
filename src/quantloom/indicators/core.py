"""Vectorized technical indicators: RSI, Stochastic, rolling volatility, and candlestick pattern
detection.

Long/short is threaded through as a single `Direction`-parametrized implementation rather than
duplicated per-direction branches.
"""

from __future__ import annotations

import pandas as pd

from quantloom import _talib_compat
from quantloom.config import Direction, IndicatorConfig

# (talib CDL function name, value that counts as a hit) per direction. Values come from talib's
# convention of returning +100/-100 for a confirmed bullish/bearish pattern, 0 otherwise.
_CANDLESTICK_PATTERNS: dict[Direction, list[tuple[str, int]]] = {
    Direction.LONG: [
        ("CDLHAMMER", 100),
        ("CDLENGULFING", 100),
        ("CDLMORNINGSTAR", 100),
        ("CDL3WHITESOLDIERS", 100),
    ],
    Direction.SHORT: [
        ("CDLSHOOTINGSTAR", -100),
        ("CDLENGULFING", -100),
        ("CDLEVENINGSTAR", -100),
        ("CDL3BLACKCROWS", -100),
    ],
}


def rsi(close: pd.Series, length: int) -> pd.Series:
    return _talib_compat.rsi(close, length).rename("rsi")


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    fastk_period: int,
    slowk_period: int,
    slowd_period: int,
) -> pd.DataFrame:
    k, d = _talib_compat.stoch(
        high,
        low,
        close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def rolling_volatility(close: pd.Series, length: int) -> pd.Series:
    return close.pct_change().rolling(length).std(ddof=1).rename("volatility")


def candlestick_signal(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, direction: Direction
) -> pd.Series:
    """True on bars matching any of a small set of classic reversal candlestick patterns."""
    hit = pd.Series(False, index=close.index)
    for pattern_name, target_value in _CANDLESTICK_PATTERNS[direction]:
        values = _talib_compat.candle_pattern(pattern_name, open_, high, low, close)
        hit = hit | (values == target_value)
    return hit.rename(f"candlestick_{direction.value}")


def compute_indicators(
    frame: pd.DataFrame, config: IndicatorConfig, directions: frozenset[Direction]
) -> pd.DataFrame:
    """All indicator columns to attach to a ticker's stored frame, for the configured directions."""
    columns: dict[str, pd.Series] = {
        "rsi": rsi(frame["close"], config.rsi_length),
    }

    stoch = stochastic(
        frame["high"],
        frame["low"],
        frame["close"],
        fastk_period=config.stochastic_fastk_period,
        slowk_period=config.stochastic_slowk_period,
        slowd_period=config.stochastic_slowd_period,
    )
    columns["stoch_k"] = stoch["stoch_k"]
    columns["stoch_d"] = stoch["stoch_d"]

    for direction in directions:
        series = candlestick_signal(
            frame["open"], frame["high"], frame["low"], frame["close"], direction
        )
        columns[str(series.name)] = series

    return pd.DataFrame(columns, index=frame.index)
