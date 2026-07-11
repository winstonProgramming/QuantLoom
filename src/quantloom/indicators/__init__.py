from __future__ import annotations

import pandas as pd

from quantloom.config import Config

from .core import candlestick_signal, compute_indicators, rolling_volatility, rsi, stochastic

__all__ = [
    "candlestick_signal",
    "compute_all",
    "compute_indicators",
    "rolling_volatility",
    "rsi",
    "stochastic",
]


def compute_all(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """All indicator columns, computed in one pass. A thin pass-through today (there's only one
    indicator stage), kept as its own entry point since callers shouldn't need to know how many
    internal stages indicator computation is split into."""
    return compute_indicators(frame, config.indicators, config.directions)
