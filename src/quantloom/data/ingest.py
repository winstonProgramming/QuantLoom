"""Downloads OHLCV data from Alpaca's Market Data API and writes it into the Parquet store.

Chosen over yfinance: yfinance caps 1h/60m bars at ~730 days of lookback regardless of requested
date range, and Alpaca covers the SEC-sourced universe (data/universe.py) far more completely than
yfinance's rate limits tolerate at that scale. Tickers are downloaded with retry+backoff, and a
ticker Alpaca couldn't fill (delisted, bad symbol, no IEX prints, etc.) is logged and skipped
rather than crashing -- or silently corrupting -- the whole run.

Alpaca's free-plan historical bars do NOT go back as far as `start_date` may request -- confirmed
empirically (not documented anywhere by Alpaca) that they're only available for roughly the
trailing ~6 years from today, on a rolling basis, regardless of bar size. A request for an earlier
`start_date` doesn't error; Alpaca silently returns however much it actually has, starting from
wherever its real historical window begins. This module has no way to detect or warn about that
truncation -- a caller comparing the requested vs. actually-ingested date range is the only way to
catch it (see main.py's `_ensure_data`, which at least validates ticker *presence* against
`refresh_data`, though not date-range completeness).

Requires the ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables (a free Alpaca account's
keys work -- no funding or brokerage subscription needed for historical market data).
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime
from typing import Protocol, cast

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from quantloom.config import UniverseConfig
from quantloom.data.store import MarketDataStore

logger = logging.getLogger(__name__)

_OHLCV_FIELDS = ["open", "high", "low", "close", "volume"]


class BatchDownloader(Protocol):
    def __call__(self, tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame: ...


def _alpaca_timeframe(candle_length: str) -> TimeFrame:
    """Maps this project's CandleLength literal (config/schema.py) onto an Alpaca TimeFrame.
    Alpaca allows Minute amounts 1-59, Hour amounts 1-23, Day/Week amount exactly 1, and
    Month amounts in {1,2,3,6,12} -- CandleLength's value set was chosen to fit those constraints
    exactly (see config/schema.py), so this mapping is total over it."""
    mapping = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "2m": TimeFrame(2, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
        "1wk": TimeFrame(1, TimeFrameUnit.Week),
        "1mo": TimeFrame(1, TimeFrameUnit.Month),
        "3mo": TimeFrame(3, TimeFrameUnit.Month),
    }
    return mapping[candle_length]


def _alpaca_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Alpaca ingestion requires the ALPACA_API_KEY and ALPACA_SECRET_KEY environment "
            "variables. A free Alpaca account's API keys work -- no funding or brokerage "
            "subscription needed for historical market data. Sign up at https://alpaca.markets."
        )
    return api_key, secret_key


def _reshape_bars_to_ticker_columns(bars_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Alpaca's `BarSet.df` is indexed by (symbol, timestamp) with one column per OHLCV field.
    Reshape into the layout the rest of `ingest()` expects: a plain datetime index with
    MultiIndex columns of (ticker, field), so `_extract_ticker_frame`'s
    `.xs(ticker, axis=1, level=0)` works regardless of which backend produced the batch."""
    if bars_df.empty:
        return pd.DataFrame(columns=pd.MultiIndex.from_product([tickers, _OHLCV_FIELDS]))

    fields = [field for field in _OHLCV_FIELDS if field in bars_df.columns]
    # bars_df[fields] is always a DataFrame (fields is a list, never a bare column name), so
    # unstack() always returns a DataFrame here -- the cast just narrows past the stub's more
    # conservative DataFrame | Series overload.
    wide = cast(pd.DataFrame, bars_df[fields].unstack(level=0))
    return wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)


def _alpaca_download(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    api_key, secret_key = _alpaca_credentials()
    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=_alpaca_timeframe(interval),
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        # Explicitly IEX: the feed guaranteed on every free plan -- switching requires an
        # account actually entitled to SIP, or requesting an unentitled feed errors.
        feed=DataFeed.IEX,
        # No extended-hours parameter exists to request/exclude premarket or after-hours bars --
        # see README.md#market-hours-coverage for what IEX actually returns.
    )
    # get_stock_bars returns BarSet unless the client was constructed with raw_data=True, which
    # this project never does.
    bars = cast(BarSet, client.get_stock_bars(request))
    return _reshape_bars_to_ticker_columns(bars.df, tickers)


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_ticker_frame(batch: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if ticker not in batch.columns.get_level_values(0):
        return None
    frame = pd.DataFrame(batch.xs(ticker, axis=1, level=0)).dropna(how="all")
    if frame.empty:
        return None
    frame = frame.rename(columns=str.lower)
    frame.columns.name = None  # drop any leftover MultiIndex level name from the batch source
    frame.index.name = "datetime"
    return frame


def _download_chunk_with_retry(
    chunk: list[str],
    universe: UniverseConfig,
    download: BatchDownloader,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame | None:
    for attempt in range(1, max_retries + 1):
        try:
            return download(
                chunk, str(universe.start_date), str(universe.end_date), universe.candle_length
            )
        except Exception:
            logger.warning(
                "Download failed for chunk (attempt %d/%d): %s",
                attempt,
                max_retries,
                chunk,
                exc_info=True,
            )
            if attempt < max_retries:
                # Exponential (not linear) backoff, plus jitter: a transient failure (a 429, a
                # dropped connection, ...) can easily outlast a few seconds -- long enough to
                # silently drop tickers that were only ever temporarily unavailable, not
                # actually unfetchable.
                backoff = retry_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(backoff + random.uniform(0, backoff * 0.5))
    logger.error("Giving up on chunk after %d attempts: %s", max_retries, chunk)
    return None


def ingest(
    tickers: list[str],
    universe: UniverseConfig,
    store: MarketDataStore,
    *,
    chunk_size: int = 50,
    max_retries: int = 5,
    retry_backoff_seconds: float = 5.0,
    download: BatchDownloader = _alpaca_download,
) -> list[str]:
    """Download raw OHLCV for `tickers` into `store`, one chunk at a time, sequentially --
    concurrent chunk downloads reliably trigger 429s (Alpaca's real capacity limit is tighter
    than its documented 200 req/min), silently dropping tickers whose chunk exhausts retries.
    Returns the subset that actually got data, in the same order as `tickers`."""
    succeeded: list[str] = []
    for chunk in _chunk(tickers, chunk_size):
        batch = _download_chunk_with_retry(
            chunk, universe, download, max_retries, retry_backoff_seconds
        )
        if batch is None:
            continue

        for ticker in chunk:
            frame = _extract_ticker_frame(batch, ticker)
            if frame is None:
                logger.warning("No usable data returned for %s, skipping", ticker)
                continue
            store.write(ticker, frame)
            succeeded.append(ticker)

    logger.info("Ingested %d/%d requested tickers", len(succeeded), len(tickers))
    return succeeded
