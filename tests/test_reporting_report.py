from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantloom.backtest.engine import PortfolioSimulator
from quantloom.backtest.metrics import PeriodReport, compute_report
from quantloom.config import Direction
from quantloom.config.schema import PositionSizingConfig, RiskConfig
from quantloom.reporting.report import _GREEN, _RED, _comparison_table, format_report


def _sizing() -> PositionSizingConfig:
    return PositionSizingConfig(max_positions=1)


def test_format_report_includes_key_figures() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)

    sim = PortfolioSimulator(config=_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 500, calendar[500], prices.iloc[500])

    report = compute_report(sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=100))
    text = format_report(report)

    assert "start date" in text
    assert "trades: 1" in text
    assert "sharpe ratio" in text
    assert "chance of profit" in text
    assert "chance of beating risk-free" in text


def test_format_report_handles_zero_trades_gracefully() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    prices = pd.Series([100.0] * len(calendar), index=calendar)
    sim = PortfolioSimulator(config=_sizing())

    report = compute_report(sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=50))
    text = format_report(report)

    assert "trades: 0" in text
    assert "no trades taken" in text


def test_format_report_includes_benchmark_section_when_present() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)
    benchmark_prices = pd.Series(400 + walk * 0.5, index=calendar)

    sim = PortfolioSimulator(config=_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 500, calendar[500], prices.iloc[500])

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
    )
    text = format_report(report)

    # comparison table header names the benchmark column, and the model-only "beat benchmark"
    # line (no benchmark-side equivalent -- it can't beat itself) names it too
    assert "SPY" in text
    assert "return" in text
    assert "sharpe ratio" in text
    assert "days won" in text
    assert "weeks won" in text
    assert "months won" in text
    assert "chance of profit" in text
    assert "chance of beating risk-free" in text
    assert "chance of beating SPY" in text
    assert "correlation to SPY" in text


def test_format_report_prints_correlation_beneath_chance_of_beating_in_all_three_sections() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 90, freq="h")
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)
    benchmark_prices = pd.Series(400 + walk * 0.5, index=calendar)

    sim = PortfolioSimulator(config=_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 1500, calendar[1500], prices.iloc[1500])

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
        train_test_split_date=date(2024, 2, 15),
    )
    lines = format_report(report).split("\n")

    beating_lines = [i for i, line in enumerate(lines) if "chance of beating SPY" in line]
    assert len(beating_lines) == 3  # OVERALL, TRAIN, TEST
    for i in beating_lines:
        assert lines[i + 1].startswith("correlation to SPY:")


def test_format_report_omits_benchmark_section_when_absent() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)

    sim = PortfolioSimulator(config=_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 500, calendar[500], prices.iloc[500])

    report = compute_report(sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=100))
    text = format_report(report)

    assert "benchmark" not in text


def _period(
    period_return: float, sharpe_ratio: float, volatility: float, **overrides: object
) -> PeriodReport:
    calendar = pd.date_range("2024-01-01", periods=5, freq="D")
    defaults: dict[str, object] = dict(
        label="period",
        start=calendar[0],
        end=calendar[-1],
        period_return=period_return,
        annualized_return=None,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        days_won=5,
        days_lost=3,
        days_flat=1,
        weeks_won=2,
        weeks_lost=1,
        weeks_flat=0,
        months_won=1,
        months_lost=0,
        months_flat=0,
        profit_odds=None,
    )
    defaults.update(overrides)
    return PeriodReport(**defaults)  # type: ignore[arg-type]


def test_comparison_table_colors_the_higher_return_green_and_lower_red() -> None:
    model = _period(period_return=0.10, sharpe_ratio=1.0, volatility=0.1)
    benchmark = _period(period_return=0.05, sharpe_ratio=1.0, volatility=0.1)

    lines = _comparison_table(model, benchmark, "SPY", holding_period_hours=168)
    return_line = next(line for line in lines if line.startswith("return"))

    # model's cell (higher return, better) comes first and should be green; benchmark's
    # (lower, worse) comes second and should be red
    assert return_line.index(_GREEN) < return_line.index(_RED)


def test_comparison_table_colors_lower_volatility_green_not_higher() -> None:
    # volatility is inverted vs. return/sharpe -- LOWER is better
    model = _period(period_return=0.05, sharpe_ratio=1.0, volatility=0.05)
    benchmark = _period(period_return=0.05, sharpe_ratio=1.0, volatility=0.20)

    lines = _comparison_table(model, benchmark, "SPY", holding_period_hours=168)
    vol_line = next(line for line in lines if line.startswith("volatility"))

    # model has the lower (better) volatility -- its cell (first) should be green
    assert vol_line.index(_GREEN) < vol_line.index(_RED)


def test_comparison_table_flat_periods_highlight_the_higher_count_red() -> None:
    # a flat period "loses to the risk-free rate" -- more of them is worse, so the HIGHER flat
    # count must be red (same polarity as "lost", opposite of "won")
    model = _period(period_return=0.05, sharpe_ratio=1.0, volatility=0.1, days_flat=10)
    benchmark = _period(period_return=0.05, sharpe_ratio=1.0, volatility=0.1, days_flat=2)

    lines = _comparison_table(model, benchmark, "SPY", holding_period_hours=168)
    flat_line = next(line for line in lines if line.startswith("days flat"))

    # model (higher flat count, worse) comes first and should be red
    assert flat_line.index(_RED) < flat_line.index(_GREEN)


def test_comparison_table_ties_are_not_colored() -> None:
    model = _period(period_return=0.05, sharpe_ratio=1.0, volatility=0.1)
    benchmark = _period(period_return=0.05, sharpe_ratio=1.0, volatility=0.1)

    lines = _comparison_table(model, benchmark, "SPY", holding_period_hours=168)
    return_line = next(line for line in lines if line.startswith("return"))

    assert _GREEN not in return_line
    assert _RED not in return_line


def test_comparison_table_no_benchmark_has_no_color() -> None:
    model = _period(period_return=0.10, sharpe_ratio=1.0, volatility=0.1)

    lines = _comparison_table(model, None, "SPY", holding_period_hours=168)
    return_line = next(line for line in lines if line.startswith("return"))

    assert _GREEN not in return_line
    assert _RED not in return_line


def test_format_report_separates_overall_train_and_test_with_blank_lines() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)
    sim = PortfolioSimulator(config=_sizing())

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=50),
        train_test_split_date=date(2024, 1, 20),
    )
    lines = format_report(report).split("\n")

    train_index = lines.index("TRAIN")
    test_index = lines.index("TEST")
    assert lines[train_index - 2] == ""  # blank line right before TRAIN's opening rule
    assert lines[test_index - 2] == ""  # blank line right before TEST's opening rule


def test_format_report_dates_have_no_time_component() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    prices = pd.Series(100.0, index=calendar)
    sim = PortfolioSimulator(config=_sizing())

    report = compute_report(sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=50))
    text = format_report(report)

    assert "start date: 2024-01-01" in text
    assert ":00:00" not in text


def test_format_report_orders_beat_benchmark_and_volatility_ratio_correctly() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)
    benchmark_prices = pd.Series(400 + walk * 0.5, index=calendar)
    sim = PortfolioSimulator(config=_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 500, calendar[500], prices.iloc[500])

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
    )
    lines = format_report(report).split("\n")

    beat_index = next(i for i, line in enumerate(lines) if line.startswith("chance of beating SPY"))
    avg_invested_index = next(
        i for i, line in enumerate(lines) if line.startswith("average % of portfolio")
    )
    vol_ratio_index = next(
        i for i, line in enumerate(lines) if line.startswith("volatility / average")
    )
    trades_index = next(i for i, line in enumerate(lines) if line.startswith("trades:"))

    # "chance of beating SPY" belongs to the first (comparison table) section -- the rule
    # separating section 1 from section 2 sits between it and "average % of portfolio"
    assert beat_index < avg_invested_index
    assert lines[avg_invested_index - 1] == "-" * 60
    assert beat_index < lines.index("-" * 60, beat_index)  # a rule follows it before section 2
    assert avg_invested_index < vol_ratio_index < trades_index
