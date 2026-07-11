from __future__ import annotations

import pandas as pd

from quantloom.config import Direction
from quantloom.config.schema import StrategyConfig
from quantloom.strategy.buy_rules import find_buy_signals


def _base_frame(n: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "divergence_long": [False] * n,
            "stochastic_crossover_long": [False] * n,
            "candlestick_long": [False] * n,
        },
        index=index,
    )


def test_single_stage_divergence_fires() -> None:
    n = 30
    frame = _base_frame(n)
    frame.loc[frame.index[20], "divergence_long"] = True
    strategy = StrategyConfig(buy_signal_order=[["rsi_divergence"]], buy_signal_expiration_bars=[])

    result = find_buy_signals(frame, Direction.LONG, strategy)

    assert result["buy_signal_long"].tolist() == [i == 20 for i in range(n)]


def test_two_stage_chain_anchors_on_the_later_signal() -> None:
    n = 30
    frame = _base_frame(n)
    frame.loc[frame.index[10], "divergence_long"] = True
    frame.loc[frame.index[15], "candlestick_long"] = True
    strategy = StrategyConfig(
        buy_signal_order=[["rsi_divergence"], ["candle sticks"]],
        buy_signal_expiration_bars=[8],
    )

    result = find_buy_signals(frame, Direction.LONG, strategy)

    # fires at the LAST stage's bar (15), not the divergence's own bar (10)
    assert result["buy_signal_long"].tolist() == [i == 15 for i in range(n)]


def test_chain_breaks_when_earlier_signal_is_outside_its_expiration_window() -> None:
    n = 30
    frame = _base_frame(n)
    frame.loc[frame.index[0], "divergence_long"] = True  # 15 bars before the candlestick
    frame.loc[frame.index[15], "candlestick_long"] = True
    strategy = StrategyConfig(
        buy_signal_order=[["rsi_divergence"], ["candle sticks"]],
        buy_signal_expiration_bars=[8],  # divergence must be within 8 bars of the candlestick
    )

    result = find_buy_signals(frame, Direction.LONG, strategy)

    assert not result["buy_signal_long"].any()


def test_tied_stage_accepts_either_order() -> None:
    n = 30
    frame = _base_frame(n)
    # divergence happens BEFORE the candlestick here -- the other permutation order
    frame.loc[frame.index[10], "divergence_long"] = True
    frame.loc[frame.index[12], "candlestick_long"] = True
    strategy = StrategyConfig(
        buy_signal_order=[["rsi_divergence", "candle sticks"]],
        buy_signal_expiration_bars=[5],
    )

    result = find_buy_signals(frame, Direction.LONG, strategy)

    # divergence(10) and candlestick(12) are both within 5 bars of each other, so the chain
    # completes regardless of which name is assumed to come "last" in the tied stage
    assert result["buy_signal_long"].any()


def test_rsi_divergence_and_stochastic_crossover_chain_independently_of_each_other() -> None:
    # the two signals are computed independently (see signals/divergence.py) -- buy_rules'
    # generic chaining is what requires one to follow the other, same as any other pairing.
    n = 30
    frame = _base_frame(n)
    frame.loc[frame.index[10], "divergence_long"] = True
    frame.loc[frame.index[15], "stochastic_crossover_long"] = True
    strategy = StrategyConfig(
        buy_signal_order=[["rsi_divergence"], ["stochastic_crossover"]],
        buy_signal_expiration_bars=[8],
    )

    result = find_buy_signals(frame, Direction.LONG, strategy)

    # fires at the LAST stage's bar (15), not the divergence's own bar (10)
    assert result["buy_signal_long"].tolist() == [i == 15 for i in range(n)]
