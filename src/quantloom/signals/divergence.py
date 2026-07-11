"""RSI/price divergence detection, and a standalone stochastic %K/%D crossover signal.

The two signals are independent (`find_divergences` / `find_stochastic_crossovers`) -- neither
takes the other as input. strategy/buy_rules.py's generic buy_signal_order allows for chaining.

Design: a divergence pair has two roles. The *anchor* extremum (older, the reference point) is
confirmed with a slower, stricter window -- since it's already in the past by the time the pair
matters, using a stricter window costs no extra real-world latency. The *trigger* extremum (the
actionable, most-recent one) is confirmed with a faster, looser window, since the overall signal's
timeliness is dominated by it. An anchor-only extremum on its own doesn't fire anything; a
divergence fires when a trigger extremum is preceded by another extremum that independently also
qualifies as an anchor extremum (every anchor is automatically also a trigger, since its window
is strictly stricter -- but not every trigger is an anchor).

Matching the two confirmation sequences is a single set-membership check over integer bar positions.

Look-ahead bias: a divergence is only exposed at `max(anchor confirmation bar, trigger
confirmation bar)`, not at the trigger extremum's own raw bar -- neither half of the pair is known
to be a genuine extremum until each has individually survived its own confirmation window.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantloom.config import Direction, DivergenceConfig, ExtremaConfig, StochasticCrossoverConfig
from quantloom.signals.extrema import find_swing_highs, find_swing_lows


def _joint_extrema_positions(
    price_confirmed: pd.Series, rsi_confirmed: pd.Series, after: int
) -> np.ndarray:
    """Raw bar positions where price and RSI both confirm an extremum on the same bar (both
    series are shifted-confirmed by the same `after`, so a shared non-NaN confirmation position
    implies a shared underlying raw position)."""
    both_confirmed = price_confirmed.notna().to_numpy() & rsi_confirmed.notna().to_numpy()
    confirmation_positions = np.flatnonzero(both_confirmed)
    return confirmation_positions - after


def find_divergences(
    close: pd.Series,
    rsi: pd.Series,
    direction: Direction,
    extrema: ExtremaConfig,
    divergence: DivergenceConfig,
) -> pd.DataFrame:
    """Bullish (LONG, using lows) or bearish (SHORT, using highs) divergence signal.

    A value only appears once the latter of the pair's two confirmations has occurred -- i.e.
    once the pair is genuinely, fully knowable, not just once the trigger extremum's own window
    has elapsed (that extremum could still be confirmed before its anchor, if the two extrema
    are close together in time; the anchor uses a slower window).
    """
    find_extrema = find_swing_lows if direction is Direction.LONG else find_swing_highs
    anchor_window = extrema.divergence_first
    trigger_window = extrema.divergence_second

    anchor_positions = set(
        _joint_extrema_positions(
            find_extrema(close, anchor_window),
            find_extrema(rsi, anchor_window),
            anchor_window.after,
        ).tolist()
    )
    trigger_positions = _joint_extrema_positions(
        find_extrema(close, trigger_window),
        find_extrema(rsi, trigger_window),
        trigger_window.after,
    )

    n = len(close)
    divergence_flag = np.zeros(n, dtype=bool)

    pairs = zip(trigger_positions[:-1], trigger_positions[1:], strict=True)
    for anchor_pos, actionable_pos in pairs:
        if anchor_pos not in anchor_positions:
            continue
        if actionable_pos - anchor_pos >= divergence.expiration_bars:
            continue

        price1, price2 = close.iloc[anchor_pos], close.iloc[actionable_pos]
        rsi1, rsi2 = rsi.iloc[anchor_pos], rsi.iloc[actionable_pos]

        is_divergence = (
            (price1 > price2 and rsi1 < rsi2)
            if direction is Direction.LONG
            else (price1 < price2 and rsi1 > rsi2)
        )
        if not is_divergence:
            continue

        confirmation_pos = max(
            anchor_pos + anchor_window.after, actionable_pos + trigger_window.after
        )
        if confirmation_pos >= n:
            continue

        divergence_flag[confirmation_pos] = True

    suffix = direction.value
    return pd.DataFrame({f"divergence_{suffix}": divergence_flag}, index=close.index)


def find_stochastic_crossovers(
    stoch_k: pd.Series,
    stoch_d: pd.Series,
    direction: Direction,
    config: StochasticCrossoverConfig,
) -> pd.DataFrame:
    """Fires when %K crosses %D (a momentum turn) after %K has entered an extreme
    (oversold/overbought) zone, within `config.expiration_bars` of first entering that zone.
    Standalone -- doesn't take a divergence as input; a strategy chains this signal after
    `"rsi_divergence"` in `buy_signal_order` if it wants confirmation-style behavior. Purely
    sequential/stateful (a "pending watch" persists across bars), so this is a single O(n)
    forward pass rather than a vectorized transform.
    """
    n = len(stoch_k)
    confirmed = np.zeros(n, dtype=bool)

    is_long = direction is Direction.LONG
    pending_since: int | None = None
    pending_initial_k = math.nan

    for day in range(n):
        k, d = stoch_k.iloc[day], stoch_d.iloc[day]

        in_extreme_zone = (
            k < config.extreme_threshold if is_long else k > 100 - config.extreme_threshold
        )
        if in_extreme_zone and pending_since is None:
            pending_since = day
            pending_initial_k = k

        if pending_since is None:
            continue
        if day - pending_since >= config.expiration_bars:
            pending_since = None
            continue

        crossed = k > d if is_long else k < d
        cross_level = config.cross_level.value
        if config.cross_level.flexible:
            level_reached = (
                k > pending_initial_k + cross_level
                if is_long
                else k < pending_initial_k - cross_level
            )
        else:
            level_reached = k > cross_level if is_long else k < 100 - cross_level

        if crossed and level_reached:
            confirmed[day] = True
            pending_since = None

    suffix = direction.value
    return pd.DataFrame({f"stochastic_crossover_{suffix}": confirmed}, index=stoch_k.index)
