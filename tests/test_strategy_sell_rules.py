from __future__ import annotations

import pandas as pd
import pytest

from quantloom.config import Direction
from quantloom.config.schema import (
    MarginRule,
    SellIndicatorRule,
    SupportResistanceRule,
    TimeRule,
)
from quantloom.indicators import rolling_volatility
from quantloom.strategy.sell_rules import find_sell_signals


def _frame(n: int, **overrides: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    base = {
        "close": [100.0] * n,
        "high": [100.0] * n,
        "low": [100.0] * n,
        "rsi": [50.0] * n,
        "stoch_k": [50.0] * n,
        "stoch_d": [50.0] * n,
        "support_resistance_levels": [[] for _ in range(n)],
    }
    base.update(overrides)
    return pd.DataFrame(base, index=index)


def _buy_at(n: int, day: int) -> pd.Series:
    flags = [False] * n
    flags[day] = True
    return pd.Series(flags, index=pd.date_range("2024-01-01", periods=n, freq="h"))


def test_indicator_rule_long_non_flexible_fires_when_rsi_crosses_threshold() -> None:
    n = 10
    rsi = [50.0] * n
    rsi[5] = 80.0
    frame = _frame(n, rsi=rsi)
    buy = _buy_at(n, 0)
    rule = SellIndicatorRule(indicator="rsi", threshold={"flexible": False, "value": 70.0})

    result = find_sell_signals(frame, buy, Direction.LONG, [[rule]])

    assert result["sell_signal_long"].tolist() == [i == 5 for i in range(n)]
    assert result["sell_price_long"].iloc[5] == 100.0  # close-based exit


def test_indicator_rule_short_flexible_fires_on_a_drop_not_a_100_minus_threshold_level() -> None:
    # regression test for the fixed bug: SHORT + flexible must compare a DELTA against
    # -threshold, not reuse the absolute-level "100 - threshold" formula from non-flexible mode.
    n = 10
    rsi = [50.0] * n
    rsi[5] = 30.0  # dropped by 20 from the entry value (50)
    frame = _frame(n, rsi=rsi)
    buy = _buy_at(n, 0)
    rule = SellIndicatorRule(indicator="rsi", threshold={"flexible": True, "value": 15.0})

    result = find_sell_signals(frame, buy, Direction.SHORT, [[rule]])

    # drop of 20 >= threshold(15) -> fires at day 5. With the old buggy formula
    # (diff <= 100-15=85) it would have fired on day 0 already (50-50=0 <= 85), which this
    # assertion also guards against.
    assert result["sell_signal_short"].tolist() == [i == 5 for i in range(n)]


def test_margin_rule_long_take_profit_via_high_reports_the_level_not_the_close() -> None:
    # take-profit/stop-loss now scale with the ticker's own rolling volatility, so the entry
    # needs enough preceding history (volatility_length bars) for that to be defined -- entry
    # fires on day 4, not day 0, and the expected level is derived from the same
    # rolling_volatility() the rule itself uses rather than a hardcoded price.
    n = 10
    entry_day = 4
    close = [100.0, 102.0, 99.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    high = list(close)

    rule = MarginRule(
        take_profit_multiplier=2.0,
        stop_loss_multiplier=2.0,
        volatility_length=3,
        take_profit_triggers_on_high=True,
    )
    vol = rolling_volatility(pd.Series(close), rule.volatility_length).iloc[entry_day]
    assert not pd.isna(vol)  # sanity check: history is long enough for this test to be meaningful
    take_profit_level = close[entry_day] * (1 + vol * rule.take_profit_multiplier)

    day_hit = 7
    high[day_hit] = take_profit_level + 1.0  # spikes through take-profit intrabar
    close[day_hit] = take_profit_level - 0.5  # close itself doesn't reach take-profit

    frame = _frame(n, high=high, close=close)
    buy = _buy_at(n, entry_day)

    result = find_sell_signals(frame, buy, Direction.LONG, [[rule]])

    assert result["sell_signal_long"].iloc[day_hit]
    assert result["sell_price_long"].iloc[day_hit] == pytest.approx(take_profit_level)


def test_margin_rule_stop_loss_always_overrides_take_profit_on_the_same_bar() -> None:
    # contrived same-bar case: both TP (via high) and SL (via close) conditions are met.
    # Stop-loss is checked second and unconditionally overwrites TP when both fire the same bar.
    n = 8
    entry_day = 4
    close = [100.0, 102.0, 99.0, 101.0, 100.0, 100.0, 100.0, 100.0]
    high = list(close)

    rule = MarginRule(
        take_profit_multiplier=2.0,
        stop_loss_multiplier=2.0,
        volatility_length=3,
        take_profit_triggers_on_high=True,
    )
    vol = rolling_volatility(pd.Series(close), rule.volatility_length).iloc[entry_day]
    take_profit_level = close[entry_day] * (1 + vol * rule.take_profit_multiplier)
    stop_loss_level = close[entry_day] * (1 - vol * rule.stop_loss_multiplier)

    day_hit = 6
    high[day_hit] = take_profit_level + 1.0  # breaches TP intrabar (via high)...
    close[day_hit] = stop_loss_level - 0.5  # ...but SL (via close) fires on the same bar

    frame = _frame(n, high=high, close=close)
    buy = _buy_at(n, entry_day)

    result = find_sell_signals(frame, buy, Direction.LONG, [[rule]])

    assert result["sell_signal_long"].iloc[day_hit]
    # close-based SL price wins, not the TP level
    assert result["sell_price_long"].iloc[day_hit] == pytest.approx(close[day_hit])


def test_margin_rule_take_profit_level_scales_with_ticker_volatility() -> None:
    """The same multiplier should produce a WIDER absolute band for a more volatile ticker --
    this is the whole point of scaling by volatility instead of a fixed percentage."""
    n = 10
    entry_day = 4
    calm_close = [100.0, 100.5, 99.7, 100.3, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    volatile_close = [100.0, 108.0, 92.0, 106.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    rule = MarginRule(take_profit_multiplier=2.0, stop_loss_multiplier=2.0, volatility_length=3)
    buy = _buy_at(n, entry_day)

    def _take_profit_level(close: list[float]) -> float:
        high = list(close)
        high[entry_day + 1] = close[entry_day] + 1000.0  # guaranteed breach the bar after entry
        frame = _frame(n, close=close, high=high)
        result = find_sell_signals(frame, buy, Direction.LONG, [[rule]])
        return float(result["sell_price_long"].iloc[entry_day + 1])

    calm_level = _take_profit_level(calm_close)
    volatile_level = _take_profit_level(volatile_close)

    assert volatile_level > calm_level > calm_close[entry_day]


def test_margin_rule_volatility_length_is_independently_configurable() -> None:
    # a shorter lookback that only sees the calm history right before entry should produce a
    # tighter band than a longer lookback that still reaches back into an earlier volatile
    # stretch -- proving volatility_length actually controls the window, not just accepted and
    # ignored in favor of indicators.rolling_volatility_length.
    n = 13
    entry_day = 11
    close = [100.0, 130.0, 70.0, 120.0, 80.0] + [100.0] * (n - 5)
    buy = _buy_at(n, entry_day)

    def _take_profit_level(volatility_length: int) -> float:
        rule = MarginRule(
            take_profit_multiplier=2.0,
            stop_loss_multiplier=2.0,
            volatility_length=volatility_length,
        )
        high = list(close)
        high[entry_day + 1] = close[entry_day] + 1000.0
        frame = _frame(n, close=close, high=high)
        result = find_sell_signals(frame, buy, Direction.LONG, [[rule]])
        return float(result["sell_price_long"].iloc[entry_day + 1])

    short_window_level = _take_profit_level(3)
    long_window_level = _take_profit_level(9)

    assert short_window_level == pytest.approx(close[entry_day])  # zero vol in the calm window
    assert long_window_level > short_window_level


def test_support_resistance_rule_long_resistance_via_high() -> None:
    n = 10
    levels = [[] for _ in range(n)]
    levels[0] = [90.0, 105.0]  # support below, resistance above at entry
    for i in range(1, n):
        levels[i] = levels[0]
    high = [100.0] * n
    high[4] = 106.0
    frame = _frame(n, high=high, **{"support_resistance_levels": levels})
    buy = _buy_at(n, 0)
    rule = SupportResistanceRule(
        resistance_min_distance=1.0, support_min_distance=1.0, resistance_triggers_on_high=True
    )

    result = find_sell_signals(frame, buy, Direction.LONG, [[rule]])

    assert result["sell_signal_long"].iloc[4]
    assert result["sell_price_long"].iloc[4] == 105.0


def test_time_rule_fires_after_max_bars_held() -> None:
    n = 10
    frame = _frame(n)
    buy = _buy_at(n, 2)
    rule = TimeRule(max_bars_held=5)

    result = find_sell_signals(frame, buy, Direction.LONG, [[rule]])

    assert result["sell_signal_long"].tolist() == [i >= 7 for i in range(n)]


def test_group_requires_all_rules_and_any_group_can_fire() -> None:
    n = 10
    rsi = [50.0] * n
    rsi[3] = 80.0  # indicator condition true only on day 3 -- but time isn't satisfied yet
    rsi[5] = 80.0  # both indicator and time (>=5 bars held) are true together here
    frame = _frame(n, rsi=rsi)
    buy = _buy_at(n, 0)
    indicator = SellIndicatorRule(indicator="rsi", threshold={"flexible": False, "value": 70.0})
    time_rule = TimeRule(max_bars_held=5)

    result = find_sell_signals(frame, buy, Direction.LONG, [[indicator, time_rule]])

    # day 3: indicator true, time not yet -- group doesn't fire. day 5: both true -- fires.
    assert result["sell_signal_long"].tolist() == [i == 5 for i in range(n)]


def test_or_across_groups_uses_the_most_conservative_level_for_long() -> None:
    n = 10
    entry_day = 4
    close = [100.0, 102.0, 99.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    high = list(close)

    margin = MarginRule(
        take_profit_multiplier=2.0,
        stop_loss_multiplier=2.0,
        volatility_length=3,
        take_profit_triggers_on_high=True,
    )
    vol = rolling_volatility(pd.Series(close), margin.volatility_length).iloc[entry_day]
    entry_close = close[entry_day]
    take_profit_level = entry_close * (1 + vol * margin.take_profit_multiplier)
    resistance_level = entry_close + (take_profit_level - entry_close) / 2  # strictly in between

    day_hit = 7
    high[day_hit] = take_profit_level + 10.0  # breaches both margin TP and resistance intrabar

    frame = _frame(n, high=high, close=close)
    frame["support_resistance_levels"] = [[90.0, resistance_level]] * n
    buy = _buy_at(n, entry_day)
    support_resistance = SupportResistanceRule(
        resistance_min_distance=1.0, support_min_distance=1.0, resistance_triggers_on_high=True
    )

    result = find_sell_signals(frame, buy, Direction.LONG, [[margin], [support_resistance]])

    assert result["sell_signal_long"].iloc[day_hit]
    # margin level > resistance level -> the lower (resistance) level wins
    assert result["sell_price_long"].iloc[day_hit] == pytest.approx(resistance_level)
