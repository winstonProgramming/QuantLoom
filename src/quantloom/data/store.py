"""Parquet-backed storage for per-ticker raw OHLCV market data.

Each ticker gets exactly one Parquet file holding ingested OHLCV -- this is the ONLY module that
builds a data file path; every other module reads/writes through it. Indicator/signal/strategy
columns are derived per-config and computed in memory by the pipeline (see main.py's
`_compute_report`) rather than written back here, since they're cheap to recompute and specific to
whichever config produced them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class MarketDataStore:
    """One Parquet file per ticker at `{data_dir}/{candle_length}/{ticker}.parquet`."""

    def __init__(self, data_dir: Path, candle_length: str) -> None:
        self._root = Path(data_dir) / candle_length
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        return self._root / f"{ticker}.parquet"

    def exists(self, ticker: str) -> bool:
        return self._path(ticker).exists()

    def read(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(ticker))

    def write(self, ticker: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self._path(ticker))

    def tickers(self) -> list[str]:
        return sorted(path.stem for path in self._root.glob("*.parquet"))
