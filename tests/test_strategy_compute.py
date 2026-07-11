from __future__ import annotations

import pandas as pd

from quantloom.config.schema import Direction
from quantloom.indicators import compute_all
from quantloom.signals import compute_signals
from quantloom.strategy import compute_strategy_signals


def test_compute_strategy_signals_produces_expected_columns(
    raw_ohlcv_frame, make_config
) -> None:
    config = make_config(frozenset({Direction.LONG}))
    raw = raw_ohlcv_frame(n=150, seed=1)
    frame = pd.concat([raw, compute_all(raw, config)], axis=1)
    frame = pd.concat([frame, compute_signals(frame, config)], axis=1)

    result = compute_strategy_signals(frame, config)

    assert "buy_signal_long" in result.columns
    assert "sell_signal_long" in result.columns
    assert "sell_price_long" in result.columns
    assert len(result) == len(frame)


def test_compute_strategy_signals_both_directions(raw_ohlcv_frame, make_config) -> None:
    config = make_config(frozenset({Direction.LONG, Direction.SHORT}))
    raw = raw_ohlcv_frame(n=150, seed=1)
    frame = pd.concat([raw, compute_all(raw, config)], axis=1)
    frame = pd.concat([frame, compute_signals(frame, config)], axis=1)

    result = compute_strategy_signals(frame, config)

    for suffix in ("long", "short"):
        assert f"buy_signal_{suffix}" in result.columns
        assert f"sell_signal_{suffix}" in result.columns
