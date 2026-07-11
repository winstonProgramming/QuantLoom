"""Formats a BacktestReport into an HTML-embeddable report: three identically-formatted blocks
(overall, train, test), each a two-column model-vs-benchmark comparison table (better/worse
highlighted green/red) followed by model-only trade/exposure stats with no benchmark equivalent.

The output contains inline `<span style="color:...">` markup for the better/worse highlighting,
so it must be rendered via `innerHTML` (e.g. reporting/grid_report.py's detail pane), not
`textContent` -- `print_report`'s plain terminal printing will show the raw `<span>` tags
literally, since ANSI terminal color and HTML markup can't both be produced from one string.
"""

from __future__ import annotations

import html
from collections.abc import Callable

from quantloom.backtest.metrics import BacktestReport, ModelPeriodReport, PeriodReport

_RULE = "-" * 60
_LABEL_WIDTH = 32
_COL_WIDTH = 14
_GREEN = "#2da44e"
_RED = "#cf222e"


def _pct(value: float | None) -> str:
    return f"{value * 100:.2f}%" if value is not None else "N/A"


def _num(value: float | None, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _int(value: float | None) -> str:
    return str(int(value)) if value is not None else "N/A"


def _cell(value: str, width: int, *, color: str | None = None) -> str:
    """Right-aligns `value` to `width` BEFORE wrapping it in a color span -- padding an
    already-wrapped string would count the invisible `<span>` markup toward the column width and
    throw off alignment in the (monospace) rendered `<pre>` block."""
    padded = f"{value:>{width}}"
    return padded if color is None else f'<span style="color:{color}">{padded}</span>'


def _row(label: str, model_cell: str, benchmark_cell: str | None) -> str:
    if benchmark_cell is None:
        return f"{label:<{_LABEL_WIDTH}}{model_cell}"
    return f"{label:<{_LABEL_WIDTH}}{model_cell}{benchmark_cell}"


def _better_worse_colors(
    model_value: float | None, benchmark_value: float | None, *, higher_is_better: bool
) -> tuple[str | None, str | None]:
    """(model_color, benchmark_color) -- both None if not comparable (either side missing, or
    tied -- a tie has no winner to highlight)."""
    if model_value is None or benchmark_value is None or model_value == benchmark_value:
        return None, None
    model_wins = (model_value > benchmark_value) == higher_is_better
    return (_GREEN, _RED) if model_wins else (_RED, _GREEN)


def _comparison_table(
    model: PeriodReport,
    benchmark: PeriodReport | None,
    benchmark_label: str,
    holding_period_hours: int,
) -> list[str]:
    header_benchmark = (
        _cell(html.escape(benchmark_label), _COL_WIDTH) if benchmark is not None else None
    )
    lines = [_row("", _cell("model", _COL_WIDTH), header_benchmark)]

    def add(
        label: str,
        model_raw: float | None,
        benchmark_raw: float | None,
        fmt: Callable[[float | None], str],
        *,
        higher_is_better: bool,
    ) -> None:
        if benchmark is None:
            lines.append(_row(label, _cell(fmt(model_raw), _COL_WIDTH), None))
            return
        model_color, benchmark_color = _better_worse_colors(
            model_raw, benchmark_raw, higher_is_better=higher_is_better
        )
        model_cell = _cell(fmt(model_raw), _COL_WIDTH, color=model_color)
        benchmark_cell = _cell(fmt(benchmark_raw), _COL_WIDTH, color=benchmark_color)
        lines.append(_row(label, model_cell, benchmark_cell))

    b = benchmark

    def bval(getter: Callable[[PeriodReport], float | None]) -> float | None:
        return getter(b) if b is not None else None

    def add_higher_better(
        label: str, model_raw: float, get: Callable[[PeriodReport], float]
    ) -> None:
        add(label, model_raw, bval(get), _int, higher_is_better=True)

    def add_lower_better(
        label: str, model_raw: float, get: Callable[[PeriodReport], float]
    ) -> None:
        add(label, model_raw, bval(get), _int, higher_is_better=False)

    add("return", model.period_return, bval(lambda p: p.period_return), _pct, higher_is_better=True)
    add(
        "sharpe ratio",
        model.sharpe_ratio,
        bval(lambda p: p.sharpe_ratio),
        _num,
        higher_is_better=True,
    )
    add("volatility", model.volatility, bval(lambda p: p.volatility), _pct, higher_is_better=False)
    add_higher_better("days won", model.days_won, lambda p: p.days_won)
    add_lower_better("days lost", model.days_lost, lambda p: p.days_lost)
    add_lower_better("days flat", model.days_flat, lambda p: p.days_flat)
    add_higher_better("weeks won", model.weeks_won, lambda p: p.weeks_won)
    add_lower_better("weeks lost", model.weeks_lost, lambda p: p.weeks_lost)
    add_lower_better("weeks flat", model.weeks_flat, lambda p: p.weeks_flat)
    add_higher_better("months won", model.months_won, lambda p: p.months_won)
    add_lower_better("months lost", model.months_lost, lambda p: p.months_lost)
    add_lower_better("months flat", model.months_flat, lambda p: p.months_flat)

    if model.profit_odds is not None:
        b_odds = b.profit_odds if b is not None else None
        add(
            f"chance of profit ({holding_period_hours} bars)",
            model.profit_odds.profit_within_holding_period,
            b_odds.profit_within_holding_period if b_odds else None,
            _pct,
            higher_is_better=True,
        )
        add(
            "chance of beating risk-free",
            model.profit_odds.beat_risk_free_rate,
            b_odds.beat_risk_free_rate if b_odds else None,
            _pct,
            higher_is_better=True,
        )

    return lines


def _trade_stats_lines(report: ModelPeriodReport) -> list[str]:
    lines = []

    avg_invested_pct = report.average_fraction_invested * 100
    lines.append(f"average % of portfolio in the market: {avg_invested_pct:.2f}%")
    if report.stats.volatility is not None and report.average_fraction_invested:
        lines.append(
            "volatility / average % of portfolio in the market: "
            f"{report.stats.volatility * 100 / report.average_fraction_invested:.2f}%"
        )

    stats = report.trade_statistics
    lines.append(f"trades: {stats.trade_count}")
    if stats.trade_count == 0:
        lines.append("(no trades taken)")
        return lines

    lines += [
        f"trades won: {stats.trades_won}",
        f"trades lost: {stats.trades_lost}",
        (
            "average profit per trade (discounting trade size): "
            f"{stats.average_profit_per_trade * 100:.2f}%"
        ),
        (
            "average profit per trade (accounting trade size): "
            f"{stats.average_profit_per_trade_weighted * 100:.2f}%"
        ),
        f"average profit of winning trade: {stats.average_win * 100:.2f}%",
        f"average profit of losing trade: {stats.average_loss * 100:.2f}%",
        f"average trade duration (bars): {stats.average_duration_bars:.1f}",
        f"most positions at once: {stats.most_concurrent_positions}",
    ]
    return lines


def _format_block(
    title: str,
    model: ModelPeriodReport,
    benchmark: PeriodReport | None,
    benchmark_label: str,
    holding_period_hours: int,
) -> list[str]:
    lines = [_RULE, title, _RULE]
    lines.append(f"start date: {model.stats.start.strftime('%Y-%m-%d')}")
    lines.append(f"end date: {model.stats.end.strftime('%Y-%m-%d')}")
    lines.append("")
    lines += _comparison_table(model.stats, benchmark, benchmark_label, holding_period_hours)
    escaped_label = html.escape(benchmark_label)
    odds = model.stats.profit_odds
    if odds is not None and odds.beat_benchmark is not None:
        lines.append(f"chance of beating {escaped_label}: {_pct(odds.beat_benchmark)}")
    if model.benchmark_correlation is not None:
        lines.append(f"correlation to {escaped_label}: {model.benchmark_correlation:.3f}")
    lines.append(_RULE)
    lines += _trade_stats_lines(model)
    lines.append(_RULE)
    return lines


def format_report(report: BacktestReport) -> str:
    benchmark_label = report.benchmark.label if report.benchmark is not None else "benchmark"

    lines = _format_block(
        "OVERALL",
        report.overall,
        report.benchmark.overall if report.benchmark is not None else None,
        benchmark_label,
        report.holding_period_hours,
    )

    if report.train is not None:
        lines.append("")
        lines += _format_block(
            "TRAIN",
            report.train,
            report.benchmark.train if report.benchmark is not None else None,
            benchmark_label,
            report.holding_period_hours,
        )
    if report.test is not None:
        lines.append("")
        lines += _format_block(
            "TEST",
            report.test,
            report.benchmark.test if report.benchmark is not None else None,
            benchmark_label,
            report.holding_period_hours,
        )

    return "\n".join(lines)


def print_report(report: BacktestReport) -> None:
    """Prints the raw HTML-markup report text to the terminal -- the `<span>` color tags will
    show up literally, since this function isn't the report's real consumer anymore (the CLI
    always opens the HTML grid report; see main.py). Kept for debugging/programmatic use."""
    print(format_report(report))
