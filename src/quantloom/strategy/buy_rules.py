"""Buy-signal engine: fires when a configured chronological sequence of signals (RSI divergence,
stochastic crossover, candlestick patterns) all occurred, each within its own expiration window
of the next.

The chain is resolved as a backward search anchored at the *previous* (more recent) signal's found
position, cascading back through the chain -- e.g. for [rsi_divergence, candle sticks], candle
sticks is searched for within its own expiration window ending "now", and once found,
rsi_divergence is searched within its window ending at that position, not at "now" directly.
This same generic chaining is also what lets a strategy compose "rsi_divergence" and
"stochastic_crossover" (two independent signals -- see signals/divergence.py) into
confirmation-style behavior.

`buy_signal_expiration_bars` has one entry per gap between consecutive signal names in
`buy_signal_order`'s flattened order (stages concatenated, ties in listed order). The last
flattened name never needs a window (see `_find_ordered_sequence`), so a single-signal
`buy_signal_order` needs zero entries and an N-signal one needs N-1, regardless of how those signals
are grouped into stages or tied within a stage.

Iterates over the sparse bars where the chain's last-stage signal fires (not every bar), matching
the sparse-event style used throughout signals.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain, permutations, product

import numpy as np
import pandas as pd

from quantloom.config import Direction, StrategyConfig

_SIGNAL_COLUMN = {
    "rsi_divergence": "divergence_{dir}",
    "stochastic_crossover": "stochastic_crossover_{dir}",
    "candle sticks": "candlestick_{dir}",
}


def _stage_orderings(stages: Sequence[Sequence[str]]) -> list[list[str]]:
    """Every full signal ordering consistent with `stages`: stages stay in sequence, but names
    tied within the same stage may occur in either order."""
    stage_perms = [list(permutations(stage)) for stage in stages]
    return [list(chain.from_iterable(combo)) for combo in product(*stage_perms)]


def _find_ordered_sequence(
    signals: dict[str, pd.Series], ordering: list[str], expirations: list[int], anchor_day: int
) -> list[int] | None:
    """Search backward from `anchor_day` for `ordering[-1]` within its expiration window; once
    found, search backward from *that* position for `ordering[-2]` within its own window; and so
    on. Returns each signal's found bar position (same order as `ordering`), or None if the
    chain breaks anywhere. A signal may be found on its anchor day itself (offset 0).

    `expirations[i]` bounds the search for `ordering[i]` relative to `ordering[i+1]`'s already-
    found position -- `expirations` has exactly `len(ordering) - 1` entries (StrategyConfig's own
    `_check_expiration_length` enforces this), one per gap between consecutive flattened
    positions. `ordering[-1]` needs no entry: `anchor_day` is already the bar where it fired (the
    caller only ever calls this for such bars), so it's found trivially without a search."""
    positions = [0] * len(ordering)
    positions[-1] = anchor_day
    anchor = anchor_day
    for i in reversed(range(len(ordering) - 1)):
        series = signals[ordering[i]]
        found = None
        for offset in range(expirations[i] + 1):
            candidate = anchor - offset
            if candidate < 0:
                break
            if series.iloc[candidate]:
                found = candidate
                break
        if found is None:
            return None
        positions[i] = found
        anchor = found
    return positions


def find_buy_signals(
    frame: pd.DataFrame, direction: Direction, strategy: StrategyConfig
) -> pd.DataFrame:
    suffix = direction.value
    signals = {name: frame[column.format(dir=suffix)] for name, column in _SIGNAL_COLUMN.items()}

    orderings = _stage_orderings(strategy.buy_signal_order)
    expirations = strategy.buy_signal_expiration_bars

    n = len(frame)
    buy = np.zeros(n, dtype=bool)

    for ordering in orderings:
        last_signal = signals[ordering[-1]]
        for raw_day in np.flatnonzero(last_signal.to_numpy()):
            day = int(raw_day)
            if buy[day]:
                continue
            if _find_ordered_sequence(signals, ordering, expirations, day) is not None:
                buy[day] = True

    return pd.DataFrame({f"buy_signal_{suffix}": buy}, index=frame.index)
