"""Top-level strategy pipeline stage: buy-signal engine + sell-rule engine."""

from __future__ import annotations

import pandas as pd

from quantloom.config import Config
from quantloom.strategy.buy_rules import find_buy_signals
from quantloom.strategy.sell_rules import find_sell_signals

__all__ = ["compute_strategy_signals"]


def compute_strategy_signals(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}

    for direction in config.directions:
        buy_df = find_buy_signals(frame, direction, config.strategy)
        for column_name in buy_df.columns:
            columns[column_name] = buy_df[column_name]

        sell_df = find_sell_signals(
            frame,
            buy_df[f"buy_signal_{direction.value}"],
            direction,
            config.strategy.sell_rule_groups,
        )
        for column_name in sell_df.columns:
            columns[column_name] = sell_df[column_name]

    return pd.DataFrame(columns, index=frame.index)
