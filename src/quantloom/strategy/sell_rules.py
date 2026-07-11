"""Sell-rule engine: per-rule exit conditions combined into groups (all rules in a group must
agree -- AND within a group), any one of which can close the position (OR across groups).

Each rule type handles LONG/SHORT via its own Direction-aware logic rather than duplicating full
method bodies. Every rule recomputes its condition fresh every bar -- "is my condition true today"
-- with no persisted state beyond what the rule itself needs (e.g. the entry indicator value for a
flexible threshold, or the entry-bar support/resistance levels).

Exit price resolution: a rule that fires via an intrabar touch (e.g. take-profit/stop-loss or a
support/resistance breach checked against the bar's high/low) reports the specific level
breached, standing in for a resting order filled exactly there; a rule that fires via a close-
based check reports no level, meaning "exit at this bar's close". When multiple rules across
fired groups disagree, the most conservative level wins: the lowest breached level for LONG (you
would have been filled first at whichever threshold price rose through), the highest for SHORT
(mirror, falling price).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantloom.config import (
    Direction,
    MarginRule,
    SellIndicatorRule,
    SellIndicatorType,
    SellRule,
    SupportResistanceRule,
    TimeRule,
)
from quantloom.indicators import rolling_volatility

_NO_LEVEL = math.nan


def _select_indicator(frame: pd.DataFrame, indicator: SellIndicatorType, suffix: str) -> pd.Series:
    if indicator is SellIndicatorType.RSI:
        return frame["rsi"]
    if indicator is SellIndicatorType.STOCH_K:
        return frame["stoch_k"]
    return frame["stoch_d"]


def _indicator_rule(
    frame: pd.DataFrame,
    buy_signal: pd.Series,
    direction: Direction,
    rule: SellIndicatorRule,
    suffix: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = _select_indicator(frame, rule.indicator, suffix)
    n = len(values)
    triggered = np.zeros(n, dtype=bool)
    is_long = direction is Direction.LONG

    armed = False
    entry_value = math.nan
    for day in range(n):
        if buy_signal.iloc[day]:
            armed = True
            entry_value = values.iloc[day]
        if not armed:
            continue
        value = values.iloc[day]
        threshold = rule.threshold.value
        if rule.threshold.flexible:
            delta = value - entry_value
            hit = delta >= threshold if is_long else delta <= -threshold
        else:
            hit = value >= threshold if is_long else value <= 100 - threshold
        triggered[day] = hit

    return triggered, np.full(n, _NO_LEVEL)


def _nearest_above(levels: Sequence[float], threshold: float) -> float:
    for level in levels:  # ascending
        if level > threshold:
            return level
    return _NO_LEVEL


def _nearest_below(levels: Sequence[float], threshold: float) -> float:
    for level in reversed(levels):  # descending
        if level < threshold:
            return level
    return _NO_LEVEL


def _support_resistance_rule(
    frame: pd.DataFrame,
    buy_signal: pd.Series,
    direction: Direction,
    rule: SupportResistanceRule,
    suffix: str,
) -> tuple[np.ndarray, np.ndarray]:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    levels_per_bar = frame["support_resistance_levels"]
    n = len(close)
    triggered = np.zeros(n, dtype=bool)
    level_out = np.full(n, _NO_LEVEL)

    resistance_price = math.nan
    support_price = math.nan
    for day in range(n):
        levels = levels_per_bar.iloc[day]
        entry_close = close.iloc[day]
        if buy_signal.iloc[day]:
            if direction is Direction.LONG:
                resistance_price = _nearest_above(
                    levels, entry_close * rule.resistance_min_distance
                )
                support_price = _nearest_below(levels, entry_close * rule.support_min_distance)
            else:
                resistance_price = _nearest_above(levels, entry_close / rule.support_min_distance)
                support_price = _nearest_below(levels, entry_close / rule.resistance_min_distance)

        hit = False
        level = _NO_LEVEL
        if direction is Direction.LONG:
            if not rule.resistance_triggers_on_high:
                if close.iloc[day] >= resistance_price:
                    hit = True
            elif high.iloc[day] >= resistance_price:
                hit = True
                level = resistance_price
            if close.iloc[day] <= support_price:
                hit = True
                level = _NO_LEVEL
        else:
            if not rule.resistance_triggers_on_high:
                if close.iloc[day] <= support_price:
                    hit = True
            elif low.iloc[day] <= support_price:
                hit = True
                level = support_price
            if close.iloc[day] >= resistance_price:
                hit = True
                level = _NO_LEVEL

        triggered[day] = hit
        level_out[day] = level

    return triggered, level_out


def _margin_rule(
    frame: pd.DataFrame, buy_signal: pd.Series, direction: Direction, rule: MarginRule, suffix: str
) -> tuple[np.ndarray, np.ndarray]:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volatility = rolling_volatility(close, rule.volatility_length)
    n = len(close)
    triggered = np.zeros(n, dtype=bool)
    level_out = np.full(n, _NO_LEVEL)

    take_profit_price = math.nan
    stop_loss_price = math.nan
    for day in range(n):
        if buy_signal.iloc[day]:
            vol = volatility.iloc[day]
            if direction is Direction.LONG:
                take_profit_price = close.iloc[day] * (1 + vol * rule.take_profit_multiplier)
                stop_loss_price = close.iloc[day] * (1 - vol * rule.stop_loss_multiplier)
            else:
                take_profit_price = close.iloc[day] / (1 + vol * rule.take_profit_multiplier)
                stop_loss_price = close.iloc[day] / (1 - vol * rule.stop_loss_multiplier)

        hit = False
        level = _NO_LEVEL
        if direction is Direction.LONG:
            if not rule.take_profit_triggers_on_high:
                if close.iloc[day] >= take_profit_price:
                    hit = True
            elif high.iloc[day] >= take_profit_price:
                hit = True
                level = take_profit_price
            if close.iloc[day] <= stop_loss_price:
                hit = True
                level = _NO_LEVEL
        else:
            if not rule.take_profit_triggers_on_high:
                if close.iloc[day] <= take_profit_price:
                    hit = True
            elif low.iloc[day] <= take_profit_price:
                hit = True
                level = take_profit_price
            if close.iloc[day] >= stop_loss_price:
                hit = True
                level = _NO_LEVEL

        triggered[day] = hit
        level_out[day] = level

    return triggered, level_out


def _time_rule(
    frame: pd.DataFrame, buy_signal: pd.Series, direction: Direction, rule: TimeRule, suffix: str
) -> tuple[np.ndarray, np.ndarray]:
    n = len(buy_signal)
    triggered = np.zeros(n, dtype=bool)
    entry_day: int | None = None
    for day in range(n):
        if buy_signal.iloc[day]:
            entry_day = day
        if entry_day is not None and day - entry_day >= rule.max_bars_held:
            triggered[day] = True
    return triggered, np.full(n, _NO_LEVEL)


def _evaluate_rule(
    frame: pd.DataFrame, buy_signal: pd.Series, direction: Direction, rule: SellRule, suffix: str
) -> tuple[np.ndarray, np.ndarray]:
    if rule.kind == "indicator":
        return _indicator_rule(frame, buy_signal, direction, rule, suffix)
    if rule.kind == "support_resistance":
        return _support_resistance_rule(frame, buy_signal, direction, rule, suffix)
    if rule.kind == "margin":
        return _margin_rule(frame, buy_signal, direction, rule, suffix)
    return _time_rule(frame, buy_signal, direction, rule, suffix)


def find_sell_signals(
    frame: pd.DataFrame,
    buy_signal: pd.Series,
    direction: Direction,
    sell_rule_groups: list[list[SellRule]],
) -> pd.DataFrame:
    n = len(frame)
    suffix = direction.value

    group_results = [
        [_evaluate_rule(frame, buy_signal, direction, rule, suffix) for rule in group]
        for group in sell_rule_groups
    ]

    sold = np.zeros(n, dtype=bool)
    sell_price = np.full(n, np.nan)
    close = frame["close"].to_numpy()

    for day in range(n):
        candidate_levels: list[float] = []
        any_group_fired = False
        for group in group_results:
            fired = all(triggered[day] for triggered, _ in group)
            if not fired:
                continue
            any_group_fired = True
            for _, level in group:
                if not math.isnan(level[day]):
                    candidate_levels.append(level[day])

        if not any_group_fired:
            continue

        sold[day] = True
        if candidate_levels:
            sell_price[day] = (
                min(candidate_levels) if direction is Direction.LONG else max(candidate_levels)
            )
        else:
            sell_price[day] = close[day]

    return pd.DataFrame(
        {f"sell_signal_{suffix}": sold, f"sell_price_{suffix}": sell_price}, index=frame.index
    )
