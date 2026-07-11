from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from quantloom.config.schema import UniverseConfig
from quantloom.data.ingest import (
    _alpaca_credentials,
    _alpaca_download,
    _alpaca_timeframe,
    _reshape_bars_to_ticker_columns,
    ingest,
)
from quantloom.data.store import MarketDataStore

# `quantloom.data`'s __init__.py does `from .ingest import ingest`, which shadows the `ingest`
# attribute on the package (the submodule) with the function of the same name -- so
# `from quantloom.data import ingest` resolves to the function, not the submodule.
# importlib.import_module sidesteps that shadowing and gives the actual submodule, needed here
# to monkeypatch its module-level StockHistoricalDataClient/DataFeed imports.
ingest_module = importlib.import_module("quantloom.data.ingest")


def _universe() -> UniverseConfig:
    return UniverseConfig(
        start_date="2024-01-01", train_test_split_date="2024-01-15", end_date="2024-02-01"
    )


def _batch(tickers: list[str], *, rows: int = 2, empty_for: set[str] = frozenset()) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    columns = pd.MultiIndex.from_product([tickers, ["open", "high", "low", "close", "volume"]])
    data = pd.DataFrame(1.0, index=index, columns=columns)
    for ticker in empty_for:
        data[ticker] = float("nan")
    return data


def test_ingest_writes_all_tickers_across_chunks(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path, "1h")
    calls: list[list[str]] = []

    def fake_download(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
        calls.append(tickers)
        return _batch(tickers)

    succeeded = ingest(
        ["AAPL", "MSFT", "GOOGL"],
        _universe(),
        store,
        chunk_size=2,
        download=fake_download,
    )

    assert succeeded == ["AAPL", "MSFT", "GOOGL"]
    assert calls == [["AAPL", "MSFT"], ["GOOGL"]]
    assert store.tickers() == ["AAPL", "GOOGL", "MSFT"]
    assert list(store.read("AAPL").columns) == ["open", "high", "low", "close", "volume"]


def test_ingest_skips_ticker_with_no_data(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path, "1h")

    def fake_download(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
        return _batch(tickers, empty_for={"BADTICKER"})

    succeeded = ingest(["AAPL", "BADTICKER"], _universe(), store, download=fake_download)

    assert succeeded == ["AAPL"]
    assert not store.exists("BADTICKER")


def test_ingest_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = MarketDataStore(tmp_path, "1h")
    attempts = {"count": 0}

    def flaky_download(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ConnectionError("simulated network failure")
        return _batch(tickers)

    monkeypatch.setattr("time.sleep", lambda _: None)
    succeeded = ingest(
        ["AAPL"],
        _universe(),
        store,
        max_retries=3,
        retry_backoff_seconds=0,
        download=flaky_download,
    )

    assert succeeded == ["AAPL"]
    assert attempts["count"] == 2


def test_ingest_gives_up_after_max_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = MarketDataStore(tmp_path, "1h")

    def always_fails(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr("time.sleep", lambda _: None)
    succeeded = ingest(
        ["AAPL"], _universe(), store, max_retries=2, retry_backoff_seconds=0, download=always_fails
    )

    assert succeeded == []
    assert store.tickers() == []


@pytest.mark.parametrize(
    ("candle_length", "expected_value"),
    [
        ("1m", "1Min"),
        ("2m", "2Min"),
        ("5m", "5Min"),
        ("15m", "15Min"),
        ("30m", "30Min"),
        ("1h", "1Hour"),
        ("1d", "1Day"),
        ("1wk", "1Week"),
        ("1mo", "1Month"),
        ("3mo", "3Month"),
    ],
)
def test_alpaca_timeframe_maps_every_candle_length(candle_length: str, expected_value: str) -> None:
    assert _alpaca_timeframe(candle_length).value == expected_value


def test_alpaca_credentials_raises_a_clear_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        _alpaca_credentials()


def test_alpaca_credentials_reads_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "key-123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret-456")

    assert _alpaca_credentials() == ("key-123", "secret-456")


def _fake_bars_df(tickers: list[str], rows: int = 2) -> pd.DataFrame:
    calendar = pd.date_range("2024-01-01", periods=rows, freq="h")
    records = [
        {
            "symbol": ticker,
            "timestamp": timestamp,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100.0,
            "trade_count": 10.0,
            "vwap": 1.4,
        }
        for ticker in tickers
        for timestamp in calendar
    ]
    return pd.DataFrame.from_records(records).set_index(["symbol", "timestamp"])


def test_reshape_bars_to_ticker_columns_pivots_symbol_index_into_columns() -> None:
    bars_df = _fake_bars_df(["AAPL", "MSFT"])

    wide = _reshape_bars_to_ticker_columns(bars_df, ["AAPL", "MSFT"])

    assert set(wide.columns.get_level_values(0)) == {"AAPL", "MSFT"}
    assert list(wide["AAPL"].columns) == ["close", "high", "low", "open", "volume"]
    assert len(wide) == 2
    assert wide[("AAPL", "close")].iloc[0] == 1.5


def test_reshape_bars_to_ticker_columns_handles_empty_response() -> None:
    wide = _reshape_bars_to_ticker_columns(pd.DataFrame(), ["AAPL", "MSFT"])

    assert wide.empty
    assert set(wide.columns.get_level_values(0)) == {"AAPL", "MSFT"}


class _FakeBarSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


def test_alpaca_download_uses_iex_feed_and_reshapes_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "key-123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret-456")

    captured_requests = []

    class _FakeClient:
        def __init__(self, api_key: str, secret_key: str) -> None:
            assert (api_key, secret_key) == ("key-123", "secret-456")

        def get_stock_bars(self, request_params):
            captured_requests.append(request_params)
            return _FakeBarSet(_fake_bars_df(["AAPL", "MSFT"]))

    monkeypatch.setattr(ingest_module, "StockHistoricalDataClient", _FakeClient)

    result = _alpaca_download(["AAPL", "MSFT"], "2024-01-01", "2024-02-01", "1h")

    assert set(result.columns.get_level_values(0)) == {"AAPL", "MSFT"}
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.feed == ingest_module.DataFeed.IEX
    assert request.timeframe.value == "1Hour"
