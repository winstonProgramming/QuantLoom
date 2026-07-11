"""CLI entry point: loads config (optionally a grid of configs, see config/grid.py), runs the
full pipeline, and opens one HTML report (reporting/grid_report.py) comparing every combination's
train/test Sharpe -- a single run is just a one-row grid.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from quantloom.backtest.engine import run_simulation
from quantloom.backtest.metrics import BacktestReport, compute_report
from quantloom.config import Config, GridPoint, load_config_grid
from quantloom.data import MarketDataStore, ingest, resolve_universe
from quantloom.indicators import compute_all
from quantloom.logging_utils import configure_logging
from quantloom.reporting.grid_report import build_grid_report_html
from quantloom.signals import compute_signals
from quantloom.strategy import compute_strategy_signals

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = "SPY"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QuantLoom backtest pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local.yaml"),
        help="Path to a local config override YAML (default: configs/local.yaml, if present).",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip re-downloading market data; reuse whatever is already in the store.",
    )
    return parser.parse_args(argv)


def _ensure_data(
    config: Config, *, refresh: bool
) -> tuple[MarketDataStore, MarketDataStore, list[str]]:
    """Ingests raw OHLCV (if `refresh`) using `config`, and returns the (store, benchmark_store,
    tickers) shared across every grid combination that follows -- raw data doesn't depend on
    which combination's config is being evaluated (candle indicators/signals do, which is why
    only THIS step is shared; see `_compute_report`). If a grid search sweeps a
    universe/ingestion field, only the first combination's value is actually used to ingest.

    The universe (the `stock_number` largest US companies by market cap) is always resolved fresh
    from the SEC's live ticker list -- a free, fast lookup, not rate-limited like Alpaca -- so
    `stock_number` takes effect regardless of `refresh`. `refresh` only controls whether OHLCV
    data for that universe gets (re-)downloaded: if False, any resolved ticker missing from
    `store` is logged and skipped (e.g. a ticker Alpaca has never been able to fill -- see
    data/ingest.py's "No usable data returned" warning -- will never appear in `store` no matter
    how many times you refresh, so silently dropping it here rather than hard-erroring is the
    right default) rather than silently falling back to whatever ticker set happens to be sitting
    on disk from an earlier run with a different `stock_number`. This only checks presence, not
    date-range completeness -- see data/ingest.py's module docstring on Alpaca silently
    truncating a `start_date` it can't fill. Raises only if *none* of the resolved tickers are
    present at all.
    """
    store = MarketDataStore(config.data_dir, config.universe.candle_length)
    # A separate root, not just a separate file, so the benchmark ticker never leaks into the
    # tradable universe (e.g. via store.tickers(), or the missing-ticker check below).
    benchmark_store = MarketDataStore(
        config.data_dir / "benchmarks", config.universe.candle_length
    )

    universe_tickers = resolve_universe(config.universe.stock_number)

    if refresh:
        tickers = ingest(universe_tickers, config.universe, store)
        ingest([BENCHMARK_TICKER], config.universe, benchmark_store)
    else:
        tickers = []
        missing: list[str] = []
        for ticker in universe_tickers:
            (tickers if store.exists(ticker) else missing).append(ticker)

        if missing:
            shown = ", ".join(missing[:20])
            more = f", and {len(missing) - 20} more" if len(missing) > 20 else ""
            logger.warning(
                "%d/%d universe ticker(s) are missing from %s and will be skipped: %s%s",
                len(missing),
                len(universe_tickers),
                config.data_dir,
                shown,
                more,
            )
        if not tickers:
            raise RuntimeError(
                f"None of the {len(universe_tickers)} resolved universe tickers are present "
                f"under {config.data_dir!s}. Run once with refresh_data: true (or without "
                "--no-refresh) first."
            )

    if not benchmark_store.exists(BENCHMARK_TICKER):
        logger.warning(
            "Benchmark ticker %s not found under %s (run without --no-refresh at least once "
            "to fetch it) -- report will omit the benchmark comparison.",
            BENCHMARK_TICKER,
            config.data_dir / "benchmarks" / config.universe.candle_length,
        )

    return store, benchmark_store, tickers


def _merge_new_columns(frame: pd.DataFrame, new_columns: pd.DataFrame) -> pd.DataFrame:
    """Drop-then-join on overlapping column names, so recomputing a stage replaces its own
    previous columns instead of erroring on a name collision."""
    overlap = frame.columns.intersection(new_columns.columns)
    return frame.drop(columns=overlap).join(new_columns, how="left")


def _compute_report(
    config: Config, store: MarketDataStore, benchmark_store: MarketDataStore, tickers: list[str]
) -> BacktestReport:
    """Indicators/signals/strategy are computed fresh in memory and never written back to `store`."""
    ticker_frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = store.read(ticker)
        frame = _merge_new_columns(frame, compute_all(frame, config))
        frame = _merge_new_columns(frame, compute_signals(frame, config))
        frame = _merge_new_columns(frame, compute_strategy_signals(frame, config))
        ticker_frames[ticker] = frame

    simulator = run_simulation(ticker_frames, config.directions, config.position_sizing)
    price_series = {ticker: frame["close"] for ticker, frame in ticker_frames.items()}
    benchmark_prices = None
    if benchmark_store.exists(BENCHMARK_TICKER):
        benchmark_prices = benchmark_store.read(BENCHMARK_TICKER)["close"]

    return compute_report(
        simulator,
        price_series,
        config.risk,
        benchmark_prices=benchmark_prices,
        benchmark_label=BENCHMARK_TICKER,
        train_test_split_date=config.universe.train_test_split_date,
    )


def _open_grid_report(
    results: list[tuple[GridPoint, BacktestReport]], *, include_equity_graphs: bool
) -> None:
    """Writes the grid search's comparison HTML (see reporting/grid_report.py) to a temp file and
    opens it in the default browser. Each row's equity curve is embedded in that page's detail
    pane (shown on click, alongside the comprehensive report)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(build_grid_report_html(results, include_equity_graphs=include_equity_graphs))
        report_path = Path(handle.name)
    webbrowser.open(report_path.as_uri())


def _run_grid_search(
    grid_points: list[GridPoint],
    store: MarketDataStore,
    benchmark_store: MarketDataStore,
    tickers: list[str],
) -> list[tuple[GridPoint, BacktestReport]]:
    """Runs every combination's `_compute_report` in its own process. Each combination's
    indicator/signal/strategy computation and backtest simulation is CPU-bound (pandas/TA-Lib), so
    threads would just serialize on the GIL for the expensive part and buy nothing -- separate
    processes give real parallelism instead. Every worker independently re-reads `tickers` from
    `store` (the sequential version already did this once per combination too), so the only added
    cost is one-time process-pool startup, not extra disk I/O."""
    max_workers = min(len(grid_points), os.cpu_count() or 1)
    logger.info(
        "running grid search: %d combinations across up to %d worker process(es)",
        len(grid_points),
        max_workers,
    )
    reports: dict[int, BacktestReport] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_report, point.config, store, benchmark_store, tickers): index
            for index, point in enumerate(grid_points)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            reports[index] = future.result()
            logger.info(
                "grid combination %d/%d complete: %s",
                completed,
                len(grid_points),
                grid_points[index].overrides,
            )

    return [(point, reports[i]) for i, point in enumerate(grid_points)]


def run_pipeline(grid_points: list[GridPoint], *, refresh: bool) -> None:
    store, benchmark_store, tickers = _ensure_data(grid_points[0].config, refresh=refresh)

    if len(grid_points) > 1:
        grid_results = _run_grid_search(grid_points, store, benchmark_store, tickers)
    else:
        report = _compute_report(grid_points[0].config, store, benchmark_store, tickers)
        grid_results = [(grid_points[0], report)]

    # Always on -- not configurable, see Config's docstring in config/schema.py.
    _open_grid_report(grid_results, include_equity_graphs=True)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    grid_points = load_config_grid(args.config)
    refresh = grid_points[0].config.universe.refresh_data and not args.no_refresh
    run_pipeline(grid_points, refresh=refresh)


if __name__ == "__main__":
    main()
