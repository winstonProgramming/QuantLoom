from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantloom.data.store import MarketDataStore


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(next(iter(values.values()))), freq="h")
    return pd.DataFrame(values, index=index)


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path, "1h")
    frame = _frame({"open": [1.0, 2.0], "close": [1.5, 2.5]})

    store.write("AAPL", frame)

    assert store.exists("AAPL")
    # Parquet doesn't round-trip a DatetimeIndex's `freq` metadata -- only compare values/dtypes.
    pd.testing.assert_frame_equal(store.read("AAPL"), frame, check_freq=False)


def test_tickers_lists_written_files(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path, "1h")
    store.write("MSFT", _frame({"open": [1.0]}))
    store.write("AAPL", _frame({"open": [1.0]}))

    assert store.tickers() == ["AAPL", "MSFT"]


def test_separate_candle_lengths_do_not_collide(tmp_path: Path) -> None:
    store_1h = MarketDataStore(tmp_path, "1h")
    store_1d = MarketDataStore(tmp_path, "1d")
    store_1h.write("AAPL", _frame({"open": [1.0]}))

    assert store_1h.exists("AAPL")
    assert not store_1d.exists("AAPL")
