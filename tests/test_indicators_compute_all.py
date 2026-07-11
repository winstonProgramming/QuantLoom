from __future__ import annotations

import pandas as pd

from quantloom.config.schema import Direction
from quantloom.indicators import compute_all

# compute_all's own direction-loop behavior (e.g. both LONG and SHORT producing their own
# candlestick column) is exercised directly against compute_indicators in
# test_indicators_core.py::test_compute_indicators_only_produces_configured_directions --
# compute_all is a documented pure pass-through to it (see indicators/__init__.py), so it isn't
# re-tested per-direction here too.


def test_compute_all_produces_expected_indicator_columns(raw_ohlcv_frame, make_config) -> None:
    frame = raw_ohlcv_frame(n=40)
    config = make_config(frozenset({Direction.LONG}))

    result = compute_all(frame, config)

    assert {"rsi", "stoch_k", "stoch_d", "candlestick_long"} <= set(result.columns)
    assert len(result) == len(frame)


def test_compute_all_is_safe_to_rerun_on_an_already_processed_frame(
    raw_ohlcv_frame, make_config
) -> None:
    # regression test: a live smoke test caught this crashing when re-running the indicators
    # stage on a frame that already carries a previous run's indicator columns (e.g. re-running
    # a pipeline stage) -- concatenating the stale frame with freshly computed columns produced
    # a duplicate "rsi" column, turning frame["rsi"] into a DataFrame instead of a Series.
    frame = raw_ohlcv_frame(n=40)
    config = make_config(frozenset({Direction.LONG}))

    once = compute_all(frame, config)
    already_processed = pd.concat([frame, once], axis=1)
    twice = compute_all(already_processed, config)

    assert isinstance(twice["rsi"], pd.Series)
    pd.testing.assert_series_equal(twice["rsi"], once["rsi"])
