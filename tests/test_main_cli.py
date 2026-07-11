from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import pytest

from quantloom.config import GridPoint
from quantloom.config.schema import Config, Direction, IndicatorConfig, UniverseConfig
from quantloom.data.store import MarketDataStore
from quantloom.main import _parse_args, run_pipeline


def test_parse_args_defaults() -> None:
    args = _parse_args([])

    assert args.config == Path("configs/local.yaml")
    assert args.no_refresh is False


def test_parse_args_no_refresh_flag() -> None:
    args = _parse_args(["--no-refresh", "--config", "custom.yaml"])

    assert args.config == Path("custom.yaml")
    assert args.no_refresh is True


def test_run_pipeline_raises_a_clear_error_when_no_universe_tickers_are_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quantloom.main.resolve_universe", lambda stock_number: ["AAA", "BBB"])
    config = Config(
        data_dir=tmp_path,
        directions=frozenset({Direction.LONG}),
        universe=UniverseConfig(
            start_date="2024-01-01",
            train_test_split_date="2024-03-01",
            end_date="2024-06-01",
            refresh_data=False,
        ),
    )

    with pytest.raises(RuntimeError, match="None of the 2 resolved universe tickers"):
        run_pipeline([GridPoint(overrides={}, config=config)], refresh=False)


def test_run_pipeline_warns_and_skips_when_some_universe_tickers_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = MarketDataStore(tmp_path, "1h")
    store.write("AAA", _synthetic_frame())
    # BBB is resolved as part of the universe but was never ingested (e.g. Alpaca could never
    # fill it) -- must warn and proceed with just AAA, not hard-error.
    monkeypatch.setattr("quantloom.main.resolve_universe", lambda stock_number: ["AAA", "BBB"])
    monkeypatch.setattr("quantloom.main.webbrowser.open", lambda url: None)

    config = _grid_config(tmp_path, rsi_length=14)

    with caplog.at_level("WARNING"):
        run_pipeline([GridPoint(overrides={}, config=config)], refresh=False)

    assert "1/2 universe ticker(s) are missing" in caplog.text
    assert "BBB" in caplog.text


def _synthetic_frame(n: int = 150) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed=2)
    close = pd.Series(100 + rng.normal(size=n).cumsum(), index=index)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000},
        index=index,
    )


def _grid_config(tmp_path: Path, rsi_length: int) -> Config:
    return Config(
        data_dir=tmp_path,
        directions=frozenset({Direction.LONG}),
        universe=UniverseConfig(
            start_date="2024-01-01",
            train_test_split_date="2024-03-01",
            end_date="2024-06-08",
            refresh_data=False,
        ),
        indicators=IndicatorConfig(rsi_length=rsi_length),
    )


def test_run_pipeline_multi_combination_grid_opens_an_html_report_and_skips_the_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MarketDataStore(tmp_path, "1h")
    frame = _synthetic_frame()
    for ticker in ("AAA", "BBB"):
        store.write(ticker, frame)
    original_columns = set(frame.columns)

    monkeypatch.setattr("quantloom.main.resolve_universe", lambda stock_number: ["AAA", "BBB"])
    opened_urls: list[str] = []
    monkeypatch.setattr("quantloom.main.webbrowser.open", opened_urls.append)

    grid_points = [
        GridPoint(
            overrides={"indicators.rsi_length": rsi_length},
            config=_grid_config(tmp_path, rsi_length),
        )
        for rsi_length in (10, 14)
    ]

    run_pipeline(grid_points, refresh=False)

    # no per-combination console report is printed for a grid search anymore
    output = capsys.readouterr().out
    assert output == ""

    assert len(opened_urls) == 1
    report_path = Path(url2pathname(urlparse(opened_urls[0]).path))
    report_html = report_path.read_text(encoding="utf-8")
    assert "indicators.rsi_length = 10" in report_html
    assert "indicators.rsi_length = 14" in report_html

    # in-memory grid combinations must never write derived columns back to the shared store
    assert set(store.read("AAA").columns) == original_columns
    assert set(store.read("BBB").columns) == original_columns


def test_run_pipeline_single_combination_skips_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single (non-grid) run takes the same in-memory path as a grid combination -- it must not
    write indicator/signal/strategy columns back to the shared store either."""
    store = MarketDataStore(tmp_path, "1h")
    frame = _synthetic_frame()
    store.write("AAA", frame)
    original_columns = set(frame.columns)

    monkeypatch.setattr("quantloom.main.resolve_universe", lambda stock_number: ["AAA"])
    monkeypatch.setattr("quantloom.main.webbrowser.open", lambda url: None)

    config = _grid_config(tmp_path, rsi_length=14)
    run_pipeline([GridPoint(overrides={}, config=config)], refresh=False)

    assert set(store.read("AAA").columns) == original_columns


def test_run_pipeline_always_embeds_equity_graph_in_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equity-curve charting is always on (not configurable -- see Config's docstring). A single
    combination is just a one-row grid now (see main.py's module docstring) -- its equity curve
    is embedded in the HTML report's detail pane, same as a multi-combination grid, never popped
    open as its own browser tab."""
    store = MarketDataStore(tmp_path, "1h")
    store.write("AAA", _synthetic_frame())

    monkeypatch.setattr("quantloom.main.resolve_universe", lambda stock_number: ["AAA"])
    opened_urls: list[str] = []
    monkeypatch.setattr("quantloom.main.webbrowser.open", opened_urls.append)

    config = _grid_config(tmp_path, rsi_length=14)
    run_pipeline([GridPoint(overrides={}, config=config)], refresh=False)
    report_html = Path(url2pathname(urlparse(opened_urls[-1]).path)).read_text(encoding="utf-8")
    assert "plotly.js v" in report_html


def test_run_pipeline_never_opens_more_than_one_browser_tab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regardless of combination count, exactly one HTML report is opened -- equity curves are
    embedded per-row in that report's detail pane, never popped open as their own tab per
    combination."""
    store = MarketDataStore(tmp_path, "1h")
    for ticker in ("AAA", "BBB"):
        store.write(ticker, _synthetic_frame())

    monkeypatch.setattr("quantloom.main.resolve_universe", lambda stock_number: ["AAA", "BBB"])
    opened_urls: list[str] = []
    monkeypatch.setattr("quantloom.main.webbrowser.open", opened_urls.append)

    grid_points = [
        GridPoint(
            overrides={"indicators.rsi_length": rsi_length},
            config=_grid_config(tmp_path, rsi_length),
        )
        for rsi_length in (10, 14)
    ]

    run_pipeline(grid_points, refresh=False)

    assert len(opened_urls) == 1
