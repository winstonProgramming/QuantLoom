from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantloom.backtest.metrics import (
    BacktestReport,
    ModelPeriodReport,
    PeriodReport,
    compute_trade_statistics,
)
from quantloom.config import GridPoint
from quantloom.config.schema import Config, Direction, UniverseConfig
from quantloom.reporting.grid_report import (
    build_grid_report_html,
    build_grid_rows,
    train_test_sharpe_correlation,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path,
        directions=frozenset({Direction.LONG}),
        universe=UniverseConfig(
            start_date="2024-01-01",
            train_test_split_date="2024-03-01",
            end_date="2024-06-01",
            refresh_data=False,
        ),
    )


def _period_report(label: str, sharpe_ratio: float | None) -> PeriodReport:
    calendar = pd.date_range("2024-01-01", periods=5, freq="D")
    return PeriodReport(
        label=label,
        start=calendar[0],
        end=calendar[-1],
        period_return=0.02,
        annualized_return=None,
        volatility=0.1,
        sharpe_ratio=sharpe_ratio,
        days_won=1,
        days_lost=1,
        days_flat=0,
        weeks_won=0,
        weeks_lost=0,
        weeks_flat=0,
        months_won=0,
        months_lost=0,
        months_flat=0,
        profit_odds=None,
    )


def _model_period(label: str, sharpe_ratio: float | None) -> ModelPeriodReport:
    return ModelPeriodReport(
        stats=_period_report(label, sharpe_ratio),
        trade_statistics=compute_trade_statistics([], position_count_history=[]),
        average_fraction_invested=0.5,
    )


def _sub_period(sharpe_ratio: float | None) -> ModelPeriodReport | None:
    return _model_period("period", sharpe_ratio) if sharpe_ratio is not None else None


def _report(train_sharpe: float | None, test_sharpe: float | None) -> BacktestReport:
    calendar = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = pd.Series([1.0, 1.01, 1.02, 1.01, 1.03], index=calendar)
    return BacktestReport(
        equity_curve=equity,
        overall=_model_period("overall", None),
        holding_period_hours=168,
        risk_free_rate=0.045,
        train=_sub_period(train_sharpe),
        test=_sub_period(test_sharpe),
    )


def _point(tmp_path: Path, overrides: dict[str, object]) -> GridPoint:
    return GridPoint(overrides=overrides, config=_config(tmp_path))


def test_build_grid_rows_sorts_by_train_sharpe_descending_with_missing_values_last(
    tmp_path: Path,
) -> None:
    results = [
        (_point(tmp_path, {"a": 1}), _report(train_sharpe=0.5, test_sharpe=0.4)),
        (_point(tmp_path, {"a": 2}), _report(train_sharpe=1.2, test_sharpe=0.1)),
        (_point(tmp_path, {"a": 3}), _report(train_sharpe=None, test_sharpe=0.9)),
    ]

    rows = build_grid_rows(results)

    assert [row.label for row in rows] == ["a = 2", "a = 1", "a = 3"]
    assert rows[0].train_sharpe == pytest.approx(1.2)
    assert rows[-1].train_sharpe is None


def test_build_grid_rows_labels_empty_overrides_as_baseline(tmp_path: Path) -> None:
    results = [(_point(tmp_path, {}), _report(train_sharpe=1.0, test_sharpe=1.0))]

    rows = build_grid_rows(results)

    assert rows[0].label == "(baseline)"


def test_train_test_sharpe_correlation_matches_hand_computed_value(tmp_path: Path) -> None:
    # perfectly linearly related -> correlation of exactly 1.0
    results = [
        (_point(tmp_path, {"a": i}), _report(train_sharpe=float(i), test_sharpe=float(i) * 2 + 1))
        for i in range(5)
    ]

    rows = build_grid_rows(results)
    correlation = train_test_sharpe_correlation(rows)

    assert correlation == pytest.approx(1.0)


def test_train_test_sharpe_correlation_is_none_with_fewer_than_two_complete_pairs(
    tmp_path: Path,
) -> None:
    results = [
        (_point(tmp_path, {"a": 1}), _report(train_sharpe=1.0, test_sharpe=None)),
        (_point(tmp_path, {"a": 2}), _report(train_sharpe=2.0, test_sharpe=1.5)),
    ]

    rows = build_grid_rows(results)

    assert train_test_sharpe_correlation(rows) is None


def test_build_grid_report_html_includes_rows_correlation_and_full_report_text(
    tmp_path: Path,
) -> None:
    results = [
        (_point(tmp_path, {"a": 1}), _report(train_sharpe=0.5, test_sharpe=0.4)),
        (_point(tmp_path, {"a": 2}), _report(train_sharpe=1.2, test_sharpe=0.8)),
    ]

    page = build_grid_report_html(results)

    assert "Grid Search Report" in page
    assert "a = 1" in page
    assert "a = 2" in page
    assert "0.500" in page
    assert "1.200" in page
    # the full comprehensive report text for each row is embedded for the click-to-reveal panel
    assert "trades: 0" in page
    assert "no trades taken" in page


def test_build_grid_report_html_escapes_hyperparameter_labels(tmp_path: Path) -> None:
    results = [
        (_point(tmp_path, {"a": "<script>alert(1)</script>"}), _report(1.0, 1.0)),
    ]

    page = build_grid_report_html(results)

    # the table cell (rendered as HTML) must not contain a live tag
    assert "<td" in page
    assert "&lt;script&gt;" in page
    # the JS string embedded in <script> must not contain a literal "</script>", which would
    # close the surrounding <script> tag early regardless of being inside a string literal
    assert "alert(1)</script>" not in page
    assert "alert(1)<\\/script>" in page


def test_build_grid_report_html_shows_na_when_correlation_undefined(tmp_path: Path) -> None:
    results = [(_point(tmp_path, {"a": 1}), _report(train_sharpe=1.0, test_sharpe=None))]

    page = build_grid_report_html(results)

    assert "N/A</strong>" in page


def test_build_grid_rows_includes_equity_json_by_default(tmp_path: Path) -> None:
    results = [(_point(tmp_path, {"a": 1}), _report(train_sharpe=1.0, test_sharpe=1.0))]

    rows = build_grid_rows(results)

    assert rows[0].equity_json is not None
    assert '"data"' in rows[0].equity_json


def test_build_grid_rows_omits_equity_json_when_disabled(tmp_path: Path) -> None:
    results = [(_point(tmp_path, {"a": 1}), _report(train_sharpe=1.0, test_sharpe=1.0))]

    rows = build_grid_rows(results, include_equity_graphs=False)

    assert rows[0].equity_json is None


def test_build_grid_report_html_embeds_plotly_and_equity_data_when_enabled(
    tmp_path: Path,
) -> None:
    results = [(_point(tmp_path, {"a": 1}), _report(train_sharpe=1.0, test_sharpe=1.0))]

    page = build_grid_report_html(results, include_equity_graphs=True)

    # the plotly.js bundle is inlined exactly once, not once per row, and the click handler
    # renders each row's equity curve into a shared div rather than popping open a browser tab
    # per row
    assert page.count("plotly.js v") == 1
    assert '"data"' in page
    assert 'id="detail-equity"' in page
    assert "Plotly.react" in page


def test_build_grid_report_html_omits_plotly_bundle_when_equity_graphs_disabled(
    tmp_path: Path,
) -> None:
    results = [(_point(tmp_path, {"a": 1}), _report(train_sharpe=1.0, test_sharpe=1.0))]

    page = build_grid_report_html(results, include_equity_graphs=False)

    assert "plotly.js v" not in page
