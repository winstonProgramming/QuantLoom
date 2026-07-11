"""Support/resistance level tracking via a monotonic-stack "envelope of unbroken extrema".

The question at every bar is "of all confirmed highs/lows so far, which ones has nothing since
exceeded?" -- exactly the classic monotonic-stack problem: each side's stack only pops an entry
when a new value dominates it, so both stacks are built in a single O(n) forward pass. The two
sides (highs, lows) are independent stacks, so one side updating never disturbs the other.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _running_envelope(
    confirmed_values: pd.Series, *, keep_if_greater: bool
) -> Iterable[list[float]]:
    """For each bar, the current stack of "unbroken" extrema up to and including that bar.

    An extremum survives only while nothing more recent has dominated it: pop anything the new
    value dominates, then push the new value. `keep_if_greater=True` for highs (an entry
    survives while nothing later is >= it), `False` for lows (survives while nothing later is
    <= it).
    """
    stack: list[float] = []
    for value in confirmed_values:
        if pd.notna(value):
            while stack and (
                (keep_if_greater and stack[-1] <= value)
                or (not keep_if_greater and stack[-1] >= value)
            ):
                stack.pop()
            stack.append(value)
        yield list(stack)


def support_resistance_levels(confirmed_highs: pd.Series, confirmed_lows: pd.Series) -> pd.Series:
    """Sorted list of currently-relevant support/resistance price levels for each bar: the most
    recent confirmed high and low, plus any older extremum still unbroken by a more recent one."""
    high_envelope = _running_envelope(confirmed_highs, keep_if_greater=True)
    low_envelope = _running_envelope(confirmed_lows, keep_if_greater=False)

    levels = [
        sorted(highs + lows)
        for highs, lows in zip(high_envelope, low_envelope, strict=True)
    ]
    return pd.Series(levels, index=confirmed_highs.index, name="support_resistance_levels")
