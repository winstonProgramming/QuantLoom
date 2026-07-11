from __future__ import annotations

import pandas as pd

from quantloom.config import Direction
from quantloom.config.schema import (
    DivergenceConfig,
    ExtremaConfig,
    ExtremaWindow,
    RelativeThreshold,
    StochasticCrossoverConfig,
)
from quantloom.signals.divergence import find_divergences, find_stochastic_crossovers


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.Series(values, index=index, dtype=float)


_SMALL_EXTREMA = ExtremaConfig(
    divergence_first=ExtremaWindow(before=2, after=3),  # "anchor": stricter/slower
    divergence_second=ExtremaWindow(before=2, after=1),  # "trigger": looser/faster
    support_resistance=ExtremaWindow(before=8, after=8),
)


def test_bullish_divergence_fires_only_once_both_extrema_are_confirmed() -> None:
    n = 25
    price = [100.0] * n
    rsi = [50.0] * n
    price[5], rsi[5] = 90.0, 20.0  # anchor low: lower price, lower RSI
    price[15], rsi[15] = 80.0, 30.0  # trigger low: even lower price, but HIGHER RSI -> divergence

    result = find_divergences(
        _series(price),
        _series(rsi),
        Direction.LONG,
        _SMALL_EXTREMA,
        DivergenceConfig(expiration_bars=30),
    )

    # anchor confirmed at 5+3=8, trigger confirmed at 15+1=16 -> pair knowable at max(8,16)=16
    confirmation_bar = 16
    assert not result["divergence_long"].iloc[:confirmation_bar].any()
    assert result["divergence_long"].iloc[confirmation_bar]
    assert not result["divergence_long"].iloc[confirmation_bar + 1 :].any()


def test_pair_rejected_when_the_anchor_extremum_fails_the_stricter_window() -> None:
    # position 5 ("P1") would satisfy the raw bullish-divergence condition against position 8
    # ("spoiler") -- but a lower price 3 bars later (within the anchor's after=3 window, but
    # outside its own before=2 window so it's independently valid) means position 5 fails the
    # anchor (before=2, after=3) check, even though it still passes the looser trigger
    # (before=2, after=1) check. Without the anchor-membership fix, this pair would fire.
    n = 20
    price = [100.0] * n
    rsi = [50.0] * n
    price[5], rsi[5] = 90.0, 20.0
    price[8], rsi[8] = 85.0, 30.0

    result = find_divergences(
        _series(price),
        _series(rsi),
        Direction.LONG,
        _SMALL_EXTREMA,
        DivergenceConfig(expiration_bars=30),
    )

    assert not result["divergence_long"].any()


def test_expiration_bars_blocks_pairs_that_are_too_far_apart() -> None:
    n = 40
    price = [100.0] * n
    rsi = [50.0] * n
    price[5], rsi[5] = 90.0, 20.0
    price[35], rsi[35] = 80.0, 30.0  # 30 bars apart

    result = find_divergences(
        _series(price),
        _series(rsi),
        Direction.LONG,
        _SMALL_EXTREMA,
        DivergenceConfig(expiration_bars=30),  # gap (30) is not < expiration_bars (30)
    )

    assert not result["divergence_long"].any()


def test_bearish_divergence_mirrors_the_bullish_case_using_highs() -> None:
    n = 25
    price = [100.0] * n
    rsi = [50.0] * n
    price[5], rsi[5] = 110.0, 80.0  # anchor high
    price[15], rsi[15] = 120.0, 70.0  # trigger high: higher price, but LOWER RSI -> divergence

    result = find_divergences(
        _series(price),
        _series(rsi),
        Direction.SHORT,
        _SMALL_EXTREMA,
        DivergenceConfig(expiration_bars=30),
    )

    confirmation_bar = 16
    assert result["divergence_short"].iloc[confirmation_bar]


def _stoch_config(
    *, flexible_cross_level: bool = False, cross_level: float = 20.0, **overrides: object
) -> StochasticCrossoverConfig:
    defaults: dict[str, object] = dict(
        extreme_threshold=50.0,
        cross_level=RelativeThreshold(flexible=flexible_cross_level, value=cross_level),
        expiration_bars=10,
    )
    defaults.update(overrides)
    return StochasticCrossoverConfig(**defaults)  # type: ignore[arg-type]


def test_stochastic_crossover_fires_once_k_crosses_d_after_entering_the_extreme_zone() -> None:
    n = 10
    k = pd.Series([65.0] * n)
    d = pd.Series([70.0] * n)
    k.iloc[2], d.iloc[2] = 30.0, 35.0  # enters extreme zone (k < extreme_threshold(50)) -> arms
    k.iloc[3], d.iloc[3] = 40.0, 45.0  # still not crossed
    k.iloc[4], d.iloc[4] = 60.0, 50.0  # crosses above d AND above cross_level(20) -> fires here

    result = find_stochastic_crossovers(k, d, Direction.LONG, _stoch_config())

    assert result["stochastic_crossover_long"].tolist() == [i == 4 for i in range(n)]


def test_stochastic_crossover_expires_if_no_crossover_within_the_window() -> None:
    n = 10
    # %K starts in the extreme zone and never crosses %D at all
    k = pd.Series([20.0] * n)
    d = pd.Series([40.0] * n)

    result = find_stochastic_crossovers(k, d, Direction.LONG, _stoch_config(expiration_bars=3))

    assert not result["stochastic_crossover_long"].any()


def test_stochastic_crossover_flexible_level_measures_from_the_arming_bars_own_k() -> None:
    n = 8
    k = pd.Series([60.0] * n)
    d = pd.Series([70.0] * n)
    k.iloc[1] = 15.0  # enters extreme zone here -> arms, initial_k = 15
    k.iloc[3] = 30.0  # 30 > 15 + cross_level(10) and 30 > d? no, d is 70 here -> not crossed yet
    d.iloc[3] = 20.0  # now k(30) > d(20) AND k(30) > initial_k(15) + cross_level(10)=25 -> fires

    result = find_stochastic_crossovers(
        k,
        d,
        Direction.LONG,
        _stoch_config(flexible_cross_level=True, cross_level=10.0, expiration_bars=5),
    )

    assert result["stochastic_crossover_long"].tolist() == [i == 3 for i in range(n)]
