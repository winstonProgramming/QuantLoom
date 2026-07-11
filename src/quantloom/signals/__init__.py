"""Top-level signals pipeline stage: combines extrema, support/resistance, RSI divergence, and
the standalone stochastic-crossover signal into the columns added to a ticker's stored frame.

Requires `frame` to already carry indicators.compute_all's output (rsi, stoch_k, stoch_d)
alongside the raw OHLCV columns.
"""

from __future__ import annotations

import pandas as pd

from quantloom.config import Config
from quantloom.signals.divergence import find_divergences, find_stochastic_crossovers
from quantloom.signals.extrema import find_swing_highs, find_swing_lows
from quantloom.signals.levels import support_resistance_levels

__all__ = ["compute_signals"]


def compute_signals(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    sr_window = config.extrema.support_resistance
    confirmed_highs = find_swing_highs(frame["high"], sr_window)
    confirmed_lows = find_swing_lows(frame["low"], sr_window)

    columns: dict[str, pd.Series] = {
        "support_resistance_levels": support_resistance_levels(confirmed_highs, confirmed_lows)
    }

    for direction in config.directions:
        divergence_df = find_divergences(
            frame["close"],
            frame["rsi"],
            direction,
            config.extrema,
            config.divergence,
        )
        for column_name in divergence_df.columns:
            columns[column_name] = divergence_df[column_name]

        crossover_df = find_stochastic_crossovers(
            frame["stoch_k"], frame["stoch_d"], direction, config.stochastic_crossover
        )
        for column_name in crossover_df.columns:
            columns[column_name] = crossover_df[column_name]

    return pd.DataFrame(columns, index=frame.index)
