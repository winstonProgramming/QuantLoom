"""Thin wrapper isolating a talib typing quirk to one place.

talib's bundled type stubs declare numpy-ndarray-only signatures, but the installed build
(verified empirically) accepts and returns pandas Series when given Series input. Treating the
module as `Any` here keeps that mismatch out of indicators/core.py instead of scattering
`# type: ignore` across every call site.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import talib as _talib_typed

_talib: Any = _talib_typed


def rsi(close: pd.Series, timeperiod: int) -> pd.Series:
    return _talib.RSI(close, timeperiod=timeperiod)


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    fastk_period: int,
    slowk_period: int,
    slowd_period: int,
) -> tuple[pd.Series, pd.Series]:
    return _talib.STOCH(
        high,
        low,
        close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )


def candle_pattern(
    name: str, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    func = getattr(_talib, name)
    return func(open_, high, low, close)
