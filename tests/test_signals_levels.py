from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantloom.signals.levels import support_resistance_levels


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.Series(values, index=index, dtype=float)


def _nan_series(n: int) -> pd.Series:
    return _series([math.nan] * n)


def test_hand_traced_high_only_envelope() -> None:
    # matches the by-hand walkthrough: 10, 15, 12, 20, 8 as successive confirmed highs
    highs = _series([10.0, 15.0, 12.0, 20.0, 8.0])
    lows = _nan_series(5)

    result = support_resistance_levels(highs, lows)

    assert result.iloc[0] == [10.0]
    assert result.iloc[1] == [15.0]
    assert result.iloc[2] == [12.0, 15.0]
    assert result.iloc[3] == [20.0]
    assert result.iloc[4] == [8.0, 20.0]


def test_lone_high_with_no_low_yet_is_not_discarded() -> None:
    # a confirmed high with no low yet must produce [high], not an accidentally-emptied list --
    # the two sides' stacks are independent, so one side having nothing yet can't wipe the other.
    highs = _series([10.0, math.nan])
    lows = _nan_series(2)

    result = support_resistance_levels(highs, lows)

    assert result.iloc[0] == [10.0]
    assert result.iloc[1] == [10.0]


def test_lone_low_with_no_high_yet_is_not_discarded() -> None:
    highs = _nan_series(2)
    lows = _series([5.0, math.nan])

    result = support_resistance_levels(highs, lows)

    assert result.iloc[0] == [5.0]


def test_no_extrema_yet_produces_empty_level_list() -> None:
    result = support_resistance_levels(_nan_series(3), _nan_series(3))

    assert result.tolist() == [[], [], []]


def test_combines_and_sorts_highs_and_lows() -> None:
    highs = _series([110.0, math.nan, 130.0])
    lows = _series([math.nan, 90.0, math.nan])

    result = support_resistance_levels(highs, lows)

    assert result.iloc[2] == [90.0, 130.0]


def _brute_force_reference(highs: pd.Series, lows: pd.Series) -> list[list[float]]:
    """Day-by-day backward rescan, used as a slow reference implementation to cross-check the
    monotonic-stack version in signals/levels.py. Tracks the high-side and low-side running
    records *independently* -- comparing a new high against the combined high+low list's overall
    max instead would let an unrelated low (if numerically larger after a strong uptrend) evict an
    older high from the envelope, conflating two conceptually independent things."""
    highs_list = highs.tolist()
    lows_list = lows.tolist()
    results = []
    for day in range(len(highs_list)):
        newer_highs = [x for x in reversed(highs_list[: day + 1]) if not math.isnan(x)]
        newer_lows = [x for x in reversed(lows_list[: day + 1]) if not math.isnan(x)]

        high_levels: list[float] = []
        for x in newer_highs:
            if not high_levels or x > high_levels[-1]:
                high_levels.append(x)
        low_levels: list[float] = []
        for x in newer_lows:
            if not low_levels or x < low_levels[-1]:
                low_levels.append(x)

        results.append(sorted(high_levels + low_levels))
    return results


def test_matches_brute_force_reference_on_random_data() -> None:
    rng = np.random.default_rng(seed=42)
    n = 200
    prices = 100 + rng.normal(scale=2, size=n).cumsum()

    # sparsely scatter confirmed highs/lows, as extrema.find_swing_highs/lows would produce
    high_mask = rng.random(n) < 0.1
    low_mask = (~high_mask) & (rng.random(n) < 0.1)
    highs = pd.Series(np.where(high_mask, prices, np.nan))
    lows = pd.Series(np.where(low_mask, prices, np.nan))

    result = support_resistance_levels(highs, lows)
    expected = _brute_force_reference(highs, lows)

    assert result.tolist() == expected
