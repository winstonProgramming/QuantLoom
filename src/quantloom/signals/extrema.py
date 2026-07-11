"""Vectorized swing-point (local high/low) detection.

A bar is a confirmed swing high/low if it's strictly more extreme than every one of the `before`
bars preceding it and every one of the `after` bars following it -- checked via a handful of
vectorized shift-and-compare passes. The same before/after check applies whether the series is
price or RSI, so `find_swing_highs`/`find_swing_lows` are generic over the input series.

Look-ahead bias: confirming a swing point at bar `d` requires seeing `after` bars past `d` --
it is not actually knowable until bar `d + after`. Every downstream consumer (divergences,
support/resistance) needs to treat it that way, not as if it were known at `d` itself.
`find_swing_highs`/`find_swing_lows` shift the confirmed value forward by `window.after` bars
before returning it, so a non-NaN value at row `d` in the returned series reflects only
information genuinely available by bar `d`.
"""

from __future__ import annotations

import pandas as pd

from quantloom.config import ExtremaWindow


def _is_strictly_more_extreme_than_neighbors(
    values: pd.Series, before: int, after: int
) -> pd.Series:
    """True where `values` is strictly greater than every one of the `before` bars preceding it
    and every one of the `after` bars following it. Negate the input to test for a low instead."""
    is_extreme = pd.Series(True, index=values.index)
    for offset in range(1, before + 1):
        is_extreme &= values > values.shift(offset)
    for offset in range(1, after + 1):
        is_extreme &= values > values.shift(-offset)
    return is_extreme.fillna(False)


def find_swing_highs(values: pd.Series, window: ExtremaWindow) -> pd.Series:
    """Confirmed swing-high values of `values` (price, RSI, whatever), indexed at the bar where
    they become knowable -- NaN everywhere else, including at the extremum's own bar."""
    is_high = _is_strictly_more_extreme_than_neighbors(values, window.before, window.after)
    return values.where(is_high).shift(window.after)


def find_swing_lows(values: pd.Series, window: ExtremaWindow) -> pd.Series:
    """Confirmed swing-low values of `values`, indexed at the bar where they become knowable."""
    is_low = _is_strictly_more_extreme_than_neighbors(-values, window.before, window.after)
    return values.where(is_low).shift(window.after)
