"""Plotly chart for a backtest's equity curve."""

from __future__ import annotations

import plotly.graph_objects as go

from quantloom.backtest.metrics import BacktestReport

# Categorical slots 1/2/3 from the dataviz reference palette (references/palette.md), fixed
# order -- validated (node scripts/validate_palette.js) for CVD-safe adjacent separation. The
# aqua/yellow slots fall below 3:1 contrast on a white surface (the palette's documented relief
# case), mitigated here by the legend (always shown for 2+ traces) plus a dashed line style on
# the risk-free trace as a secondary, color-independent encoding.
_MODEL_COLOR = "#2a78d6"
_BENCHMARK_COLOR = "#1baf7a"
_RISK_FREE_COLOR = "#eda100"


def plot_equity_curve(report: BacktestReport) -> go.Figure:
    """Model equity vs. the benchmark's own equity curve (if available) vs. a hypothetical
    risk-free-rate compounding curve over the same dates -- so "did this beat doing nothing" and
    "did this beat the market" are both visible on the same chart, not just in the text report's
    profit-odds numbers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=report.equity_curve.index,
            y=report.equity_curve.values,
            name="model",
            line=dict(color=_MODEL_COLOR),
        )
    )
    if report.benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=report.benchmark.equity_curve.index,
                y=report.benchmark.equity_curve.values,
                name=report.benchmark.label,
                line=dict(color=_BENCHMARK_COLOR),
            )
        )

    elapsed_years = (report.equity_curve.index - report.equity_curve.index[0]).days / 365.25
    risk_free_curve = (1 + report.risk_free_rate) ** elapsed_years
    fig.add_trace(
        go.Scatter(
            x=report.equity_curve.index,
            y=risk_free_curve,
            name="risk-free rate",
            line=dict(color=_RISK_FREE_COLOR, dash="dash"),
        )
    )

    fig.update_layout(
        title="Portfolio equity curve", xaxis_title="date", yaxis_title="portfolio value"
    )
    return fig
