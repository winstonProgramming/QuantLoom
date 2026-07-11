from __future__ import annotations

import pandas as pd

from quantloom.config.schema import Direction
from quantloom.indicators import compute_all
from quantloom.signals import compute_signals


def test_compute_signals_produces_expected_columns_for_one_direction(
    raw_ohlcv_frame, make_config
) -> None:
    config = make_config(frozenset({Direction.LONG}))
    raw = raw_ohlcv_frame()
    frame = pd.concat([raw, compute_all(raw, config)], axis=1)

    result = compute_signals(frame, config)

    assert "support_resistance_levels" in result.columns
    assert "divergence_long" in result.columns
    assert "stochastic_crossover_long" in result.columns
    assert len(result) == len(frame)


def test_compute_signals_produces_both_directions_when_configured(
    raw_ohlcv_frame, make_config
) -> None:
    config = make_config(frozenset({Direction.LONG, Direction.SHORT}))
    raw = raw_ohlcv_frame()
    frame = pd.concat([raw, compute_all(raw, config)], axis=1)

    result = compute_signals(frame, config)

    for suffix in ("long", "short"):
        assert f"divergence_{suffix}" in result.columns
        assert f"stochastic_crossover_{suffix}" in result.columns
