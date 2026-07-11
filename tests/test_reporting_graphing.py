from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from quantloom.backtest.engine import PortfolioSimulator
from quantloom.backtest.metrics import compute_report
from quantloom.config import Direction
from quantloom.config.schema import PositionSizingConfig, RiskConfig
from quantloom.reporting.graphing import plot_equity_curve


def test_plot_equity_curve_includes_model_and_risk_free_traces_without_benchmark() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    prices = pd.Series(100.0, index=calendar)
    sim = PortfolioSimulator(config=PositionSizingConfig(max_positions=1))
    sim.buy("AAPL", 5, calendar[5], 100.0, direction=Direction.LONG)
    sim.sell("AAPL", 100, calendar[100], 105.0)

    report = compute_report(sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=50))
    fig = plot_equity_curve(report)

    assert isinstance(fig, go.Figure)
    names = [trace.name for trace in fig.data]
    assert names == ["model", "risk-free rate"]


def test_plot_equity_curve_includes_benchmark_trace_when_present() -> None:
    calendar = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    prices = pd.Series(100.0, index=calendar)
    benchmark_prices = pd.Series(100.0 * 1.0001 ** np.arange(len(calendar)), index=calendar)
    sim = PortfolioSimulator(config=PositionSizingConfig(max_positions=1))
    sim.buy("AAPL", 5, calendar[5], 100.0, direction=Direction.LONG)
    sim.sell("AAPL", 100, calendar[100], 105.0)

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=50),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
    )
    fig = plot_equity_curve(report)

    names = [trace.name for trace in fig.data]
    assert names == ["model", "SPY", "risk-free rate"]


def test_plot_equity_curve_risk_free_trace_compounds_at_the_configured_rate() -> None:
    calendar = pd.date_range("2024-01-01", periods=365, freq="D")  # exactly 1 year
    prices = pd.Series(100.0, index=calendar)
    sim = PortfolioSimulator(config=PositionSizingConfig(max_positions=1))

    risk_config = RiskConfig(monte_carlo_paths=50, risk_free_rate=0.05)
    report = compute_report(sim, {"AAPL": prices}, risk_config)
    fig = plot_equity_curve(report)

    risk_free_trace = next(trace for trace in fig.data if trace.name == "risk-free rate")
    # ~1 year elapsed -> the curve should have compounded to roughly (1 + rate)
    assert risk_free_trace.y[0] == pytest.approx(1.0, abs=1e-6)
    assert risk_free_trace.y[-1] == pytest.approx(1.05, abs=0.01)
