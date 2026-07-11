from __future__ import annotations

import math

import pandas as pd

from quantloom.config import ExtremaWindow
from quantloom.signals.extrema import find_swing_highs, find_swing_lows


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.Series(values, index=index, dtype=float)


def test_find_swing_highs_detects_a_peak_and_shifts_it_forward_by_after() -> None:
    #                index: 0    1    2    3    4    5    6
    values = _series([1.0, 2.0, 5.0, 3.0, 2.0, 1.0, 1.0])
    window = ExtremaWindow(before=2, after=2)

    result = find_swing_highs(values, window)

    # THE critical look-ahead-bias regression check: the peak at index 2 is not knowable until
    # index 2 + after(2) = 4 -- it must be NaN at its own bar, not just "eventually correct".
    assert math.isnan(result.iloc[2])
    assert result.iloc[4] == 5.0
    # nowhere else should a value appear
    assert result.drop(result.index[4]).isna().all()


def test_find_swing_lows_detects_a_trough_and_shifts_it_forward_by_after() -> None:
    values = _series([5.0, 4.0, 1.0, 3.0, 4.0, 5.0, 5.0])
    window = ExtremaWindow(before=2, after=2)

    result = find_swing_lows(values, window)

    assert math.isnan(result.iloc[2])
    assert result.iloc[4] == 1.0
    assert result.drop(result.index[4]).isna().all()


def test_a_tie_with_a_neighbor_does_not_count_as_a_swing_high() -> None:
    # index 2 (5.0) ties index 4 (5.0) -- strict inequality means index 2 is NOT a confirmed high
    values = _series([1.0, 2.0, 5.0, 3.0, 5.0, 1.0, 1.0])
    window = ExtremaWindow(before=2, after=2)

    result = find_swing_highs(values, window)

    assert result.isna().all()


def test_series_shorter_than_the_window_produces_no_confirmations() -> None:
    values = _series([1.0, 5.0, 1.0])
    window = ExtremaWindow(before=3, after=3)

    result = find_swing_highs(values, window)

    assert result.isna().all()


def test_works_identically_on_any_series_not_just_price() -> None:
    # same function, called on an RSI-shaped series -- there's no separate RSI-specific
    # implementation, since the before/after extremum check doesn't care what the series represents.
    rsi_like = _series([50.0, 60.0, 85.0, 55.0, 40.0])
    window = ExtremaWindow(before=1, after=1)

    result = find_swing_highs(rsi_like, window)

    assert result.iloc[3] == 85.0
