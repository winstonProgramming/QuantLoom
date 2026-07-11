"""Backtest performance metrics: equity curve construction, Sharpe ratio/volatility, trade
statistics, and a block-bootstrap Monte Carlo profit-odds simulation.

Two modeling choices worth calling out:

1. The Monte Carlo simulation block-bootstraps the strategy's own realized per-bar returns rather
   than drawing each simulated bar i.i.d. from a symmetric distribution (e.g. Uniform(mean-std,
   mean+std)) -- a block bootstrap preserves the empirical return distribution's actual shape
   (skew, fat tails) and its short-range autocorrelation, instead of assuming a shape that may not
   match the data.
2. Annualization uses a factor derived from the data's own observed bar density (`_bars_per_year`)
   rather than a constant tied to one specific candle length, so the same code stays correct as
   `candle_length` changes.

Report shape: every period (overall/train/test) is a `PeriodReport` -- the same shape for both
the model and the benchmark, so they can be compared side by side in the console report's
two-column tables. Model-only stats with no benchmark equivalent (trade statistics, portfolio
exposure -- a buy-and-hold benchmark has neither) live alongside a `PeriodReport` in a
`ModelPeriodReport`, one per period, attached to `BacktestReport.overall/train/test`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quantloom.backtest.engine import ClosedTrade, PortfolioSimulator, Position
from quantloom.config import Direction, RiskConfig


def _mark_to_market(
    direction: Direction, entry_price: float, trade_size: float, aligned: pd.Series
) -> pd.Series:
    if direction is Direction.LONG:
        return trade_size * (aligned / entry_price)
    return trade_size * (2 - aligned / entry_price)


def _trade_contribution(
    trade: ClosedTrade, prices: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    """This trade's contribution to portfolio value at every bar in `calendar`: its trade size
    marked to the current price while held, zero outside its holding interval."""
    aligned = prices.reindex(calendar, method="ffill")
    in_position = (calendar >= trade.entry_date) & (calendar < trade.exit_date)
    marked = _mark_to_market(trade.direction, trade.entry_price, trade.trade_size, aligned)
    return marked.where(in_position, 0.0)


def _open_position_contribution(
    position: Position, prices: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    """A still-open position's contribution to portfolio value at every bar in `calendar`: its
    trade size marked to the current price from entry through the end of the calendar (it has no
    exit yet). Without this, `buy()` deducts an open position's cost from cash the moment it
    opens, but that capital was never marked back to market anywhere -- it would silently vanish
    from the equity curve for any position still open when the report window ends, understating
    `model return` by exactly that position's cost basis regardless of how it's actually
    performing."""
    aligned = prices.reindex(calendar, method="ffill")
    in_position = calendar >= position.entry_date
    marked = _mark_to_market(position.direction, position.entry_price, position.trade_size, aligned)
    return marked.where(in_position, 0.0)


def _cash_curve(
    cash_history: list[tuple[pd.Timestamp, float]], calendar: pd.DatetimeIndex
) -> pd.Series:
    """Cash at every bar in `calendar`: flat between trade events (it only changes when a trade
    executes), forward-filled from the event history, 1.0 (all starting capital, uninvested)
    before the first trade. When multiple events land on the same timestamp -- a same-bar round
    trip, or several tickers trading at once -- only the last event's cash value for that
    timestamp is kept; reindexing requires a unique source index."""
    dates = pd.DatetimeIndex([d for d, _ in cash_history])
    # dtype=float matters even (especially) when cash_history is empty -- an untyped empty
    # Series defaults to dtype=object, which silently survives reindex/ffill/fillna and only
    # breaks much later (a numpy ufunc casting error deep inside the Monte Carlo simulation).
    values = pd.Series([c for _, c in cash_history], index=dates, dtype=float)
    values = values[~values.index.duplicated(keep="last")]
    union_index = calendar.union(pd.DatetimeIndex(values.index))
    return values.reindex(union_index).sort_index().ffill().reindex(calendar).fillna(1.0)


def build_equity_curve(
    simulator: PortfolioSimulator, price_series: dict[str, pd.Series], calendar: pd.DatetimeIndex
) -> pd.Series:
    """Bar-by-bar mark-to-market portfolio value: cash plus each closed trade's value marked to
    the bar's close over its holding interval."""
    cash = _cash_curve(simulator.cash_history, calendar)

    position_value = pd.Series(0.0, index=calendar)
    for trade in simulator.closed_trades:
        position_value = position_value + _trade_contribution(
            trade, price_series[trade.ticker], calendar
        )
    for position in simulator.positions.values():
        position_value = position_value + _open_position_contribution(
            position, price_series[position.ticker], calendar
        )

    return cash + position_value


def _bars_per_year(calendar: pd.DatetimeIndex) -> float:
    elapsed_years = (calendar[-1] - calendar[0]).days / 365.25
    return len(calendar) / elapsed_years


def annualization_factor(calendar: pd.DatetimeIndex) -> float:
    """sqrt(bars per year), derived from the data's own observed bar density rather than a
    constant tied to one specific candle length."""
    return math.sqrt(_bars_per_year(calendar))


@dataclass
class RiskMetrics:
    yearly_return: float
    annual_volatility: float
    sharpe_ratio: float


def compute_risk_metrics(equity_curve: pd.Series, risk_free_rate: float) -> RiskMetrics:
    non_positive = equity_curve[equity_curve <= 0]
    if not non_positive.empty:
        bad_date = non_positive.index[0]
        raise ValueError(
            f"equity curve went non-positive at {bad_date} (value={non_positive.iloc[0]!r}) -- "
            "yearly return, volatility, and Sharpe ratio are undefined once total portfolio "
            "equity is zero or negative. This is most likely an unbounded short-side loss "
            "(a short's downside isn't capped at -100% the way a long's is) that wasn't closed "
            "before exceeding the portfolio's capital -- tighten the short side's stop-loss/exit "
            "rules rather than relying on this check."
        )

    elapsed_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    yearly_return = equity_curve.iloc[-1] ** (1 / elapsed_years) - 1

    bar_returns = equity_curve.pct_change().dropna()
    calendar = pd.DatetimeIndex(equity_curve.index)
    annual_volatility = bar_returns.std() * annualization_factor(calendar)

    sharpe_ratio = (yearly_return - risk_free_rate) / annual_volatility
    return RiskMetrics(yearly_return, annual_volatility, sharpe_ratio)


def average_fraction_invested(equity_curve: pd.Series, cash_curve: pd.Series) -> float:
    return float((1 - cash_curve / equity_curve).mean())


def _periodic_outcomes(equity_curve: pd.Series, freq: str) -> tuple[int, int, int]:
    """(periods won, periods lost, periods flat), comparing period-over-period portfolio value,
    where `freq` is a pandas resample alias (e.g. "D", "W", "ME")."""
    period = equity_curve.resample(freq).last().dropna()
    changes = period.diff().dropna()
    return int((changes > 0).sum()), int((changes < 0).sum()), int((changes == 0).sum())


def daily_outcomes(equity_curve: pd.Series) -> tuple[int, int, int]:
    """(days won, days lost, days flat), comparing day-over-day portfolio value."""
    return _periodic_outcomes(equity_curve, "D")


def weekly_outcomes(equity_curve: pd.Series) -> tuple[int, int, int]:
    """(weeks won, weeks lost, weeks flat), comparing week-over-week portfolio value."""
    return _periodic_outcomes(equity_curve, "W")


def monthly_outcomes(equity_curve: pd.Series) -> tuple[int, int, int]:
    """(months won, months lost, months flat), comparing month-over-month portfolio value."""
    return _periodic_outcomes(equity_curve, "ME")


@dataclass
class TradeStatistics:
    trade_count: int
    trades_won: int
    trades_lost: int
    average_profit_per_trade: float
    average_profit_per_trade_weighted: float
    average_win: float
    average_loss: float
    average_duration_bars: float
    most_concurrent_positions: int


def _geometric_mean_return(values: list[float]) -> float:
    if not values:
        return math.nan
    return statistics.geometric_mean([v + 1 for v in values]) - 1


def compute_trade_statistics(
    trades: list[ClosedTrade], position_count_history: list[tuple[pd.Timestamp, int]]
) -> TradeStatistics:
    """Returns an all-NaN/zero-count result (rather than raising) when `trades` is empty --
    a real scenario (a tight sell rule or a short backtest window can easily produce zero
    trades), not just a theoretical edge case."""
    profits = [t.profit for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    weighted_profits = [t.profit * t.trade_size for t in trades]

    return TradeStatistics(
        trade_count=len(trades),
        trades_won=len(wins),
        trades_lost=len(losses),
        average_profit_per_trade=_geometric_mean_return(profits),
        average_profit_per_trade_weighted=_geometric_mean_return(weighted_profits),
        average_win=_geometric_mean_return(wins),
        average_loss=_geometric_mean_return(losses),
        average_duration_bars=(
            statistics.mean(t.duration_bars for t in trades) if trades else math.nan
        ),
        most_concurrent_positions=max((c for _, c in position_count_history), default=0),
    )


@dataclass
class ProfitOdds:
    """Empirical odds, from the same simulated `holding_period_hours`-long paths (see
    simulate_profit_odds), of clearing three different bars -- each compared against that one
    distribution rather than three separately-simulated ones, so the three numbers are directly
    comparable (a stricter bar can only have lower or equal odds than a looser one)."""

    profit_within_holding_period: float
    beat_risk_free_rate: float
    beat_benchmark: float | None


def _compounded_threshold(
    annual_rate: float, bars_per_year: float, holding_period_hours: int
) -> float:
    """Converts an annualized rate into the compounded growth multiple expected over a
    holding_period_hours-bar horizon"""
    return (1 + annual_rate) ** (holding_period_hours / bars_per_year)


# Hardcoded, not configurable: every run (and every grid combination) should be directly
# comparable against every other
_MONTE_CARLO_SEED = 42


def simulate_profit_odds(
    hourly_returns: pd.Series,
    config: RiskConfig,
    *,
    bars_per_year: float,
    benchmark_annual_return: float | None = None,
    rng: np.random.Generator | None = None,
) -> ProfitOdds:
    """Block-bootstraps the strategy's own realized per-bar returns into `monte_carlo_paths`
    simulated `holding_period_hours`-long paths (resampling contiguous blocks of
    `monte_carlo_block_size` returns at a time, preserving short-range autocorrelation instead of
    treating each bar as independent), then reports the empirical odds of the compounded path
    clearing: any profit at all (> breakeven), the risk-free rate compounded over the same
    horizon, and -- only when `benchmark_annual_return` is given -- the benchmark's own realized
    annualized return compounded over the same horizon."""
    rng = rng if rng is not None else np.random.default_rng(_MONTE_CARLO_SEED)
    returns = hourly_returns.dropna().to_numpy()
    n = len(returns)
    block = min(config.monte_carlo_block_size, n)

    paths = np.ones(config.monte_carlo_paths)
    remaining = config.holding_period_hours
    while remaining > 0:
        this_block = min(block, remaining)
        starts = rng.integers(0, n - this_block + 1, size=config.monte_carlo_paths)
        offsets = np.arange(this_block)
        block_returns = returns[starts[:, None] + offsets[None, :]]
        paths *= np.prod(1 + block_returns, axis=1)
        remaining -= this_block

    risk_free_threshold = _compounded_threshold(
        config.risk_free_rate, bars_per_year, config.holding_period_hours
    )
    beat_benchmark = None
    if benchmark_annual_return is not None:
        benchmark_threshold = _compounded_threshold(
            benchmark_annual_return, bars_per_year, config.holding_period_hours
        )
        beat_benchmark = float((paths > benchmark_threshold).mean())

    return ProfitOdds(
        profit_within_holding_period=float((paths > 1.0).mean()),
        beat_risk_free_rate=float((paths > risk_free_threshold).mean()),
        beat_benchmark=beat_benchmark,
    )


def _benchmark_correlation(
    equity_curve: pd.Series, benchmark_equity_curve: pd.Series
) -> float | None:
    """Pearson correlation between the strategy's and the benchmark's bar-over-bar returns, over
    whatever dates the two series actually share -- None (rather than NaN) if fewer than two
    dates overlap, since a correlation isn't meaningfully defined there."""
    returns = equity_curve.pct_change().dropna()
    benchmark_returns = benchmark_equity_curve.pct_change().dropna()
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner")
    if len(aligned) < 2:
        return None
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(correlation) if not math.isnan(correlation) else None


@dataclass
class PeriodReport:
    """Performance stats for one series (model or benchmark) over one chronological period
    (overall/train/test) -- every field here is comparable side by side between the model and a
    benchmark, which is exactly what the console report's two-column tables do. Model-only stats
    with no benchmark equivalent (trade statistics, portfolio exposure -- a buy-and-hold
    benchmark has neither) live on `ModelPeriodReport` instead."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    period_return: float
    annualized_return: float | None
    volatility: float | None
    sharpe_ratio: float | None
    days_won: int
    days_lost: int
    days_flat: int
    weeks_won: int
    weeks_lost: int
    weeks_flat: int
    months_won: int
    months_lost: int
    months_flat: int
    profit_odds: ProfitOdds | None


def _risk_and_profit_odds_for_period(
    equity_curve: pd.Series,
    risk_config: RiskConfig,
    benchmark_annual_return: float | None,
    *,
    strict: bool,
) -> tuple[float | None, float | None, float | None, ProfitOdds | None]:
    """(annualized_return, volatility, sharpe_ratio, profit_odds) for `equity_curve`.

    strict=True (the overall/full period) raises via `compute_risk_metrics` if equity ever goes
    non-positive -- an actual bug worth surfacing loudly, same as before this function existed.
    strict=False (a train/test sub-period) instead returns all-None if there's nothing meaningful
    to compute (e.g. under a year of data) -- a short sub-window failing to produce a full set of
    annualized stats is an expected, common case, not a bug."""
    if strict:
        risk = compute_risk_metrics(equity_curve, risk_config.risk_free_rate)
        profit_odds = None
        if risk_config.simulate_profit_odds:
            bar_returns = equity_curve.pct_change().dropna()
            bars_per_year = _bars_per_year(pd.DatetimeIndex(equity_curve.index))
            profit_odds = simulate_profit_odds(
                bar_returns,
                risk_config,
                bars_per_year=bars_per_year,
                benchmark_annual_return=benchmark_annual_return,
            )
        return risk.yearly_return, risk.annual_volatility, risk.sharpe_ratio, profit_odds

    elapsed_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if elapsed_years <= 0:
        return None, None, None, None
    annualized_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / elapsed_years) - 1
    bar_returns = equity_curve.pct_change().dropna()
    calendar = pd.DatetimeIndex(equity_curve.index)
    vol = bar_returns.std() * annualization_factor(calendar)
    if not vol or math.isnan(vol):
        return annualized_return, None, None, None
    sharpe_ratio = (annualized_return - risk_config.risk_free_rate) / vol
    profit_odds = None
    if risk_config.simulate_profit_odds:
        bars_per_year = _bars_per_year(calendar)
        profit_odds = simulate_profit_odds(
            bar_returns,
            risk_config,
            bars_per_year=bars_per_year,
            benchmark_annual_return=benchmark_annual_return,
        )
    return annualized_return, vol, sharpe_ratio, profit_odds


def _period_report(
    label: str,
    equity_curve: pd.Series,
    risk_config: RiskConfig,
    benchmark_annual_return: float | None,
    *,
    strict: bool,
) -> PeriodReport | None:
    """`PeriodReport` for `equity_curve` as given -- callers slice first (e.g. via
    `.loc[start:end]`), so this works uniformly for the overall period, a train/test sub-period,
    the model, or the benchmark. None (non-strict only) if the slice is too short or starts
    non-positive to compute anything meaningful from."""
    if not strict and (len(equity_curve) < 2 or equity_curve.iloc[0] <= 0):
        return None

    period_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    days_won, days_lost, days_flat = daily_outcomes(equity_curve)
    weeks_won, weeks_lost, weeks_flat = weekly_outcomes(equity_curve)
    months_won, months_lost, months_flat = monthly_outcomes(equity_curve)
    annualized_return, volatility, sharpe_ratio, profit_odds = _risk_and_profit_odds_for_period(
        equity_curve, risk_config, benchmark_annual_return, strict=strict
    )

    return PeriodReport(
        label=label,
        start=equity_curve.index[0],
        end=equity_curve.index[-1],
        period_return=period_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        days_won=days_won,
        days_lost=days_lost,
        days_flat=days_flat,
        weeks_won=weeks_won,
        weeks_lost=weeks_lost,
        weeks_flat=weeks_flat,
        months_won=months_won,
        months_lost=months_lost,
        months_flat=months_flat,
        profit_odds=profit_odds,
    )


def _period_report_strict(
    label: str,
    equity_curve: pd.Series,
    risk_config: RiskConfig,
    benchmark_annual_return: float | None,
) -> PeriodReport:
    """Non-Optional wrapper over `_period_report(strict=True)`, which never actually returns None
    (it raises instead) -- lets `overall` stay typed as a plain `PeriodReport` everywhere."""
    report = _period_report(label, equity_curve, risk_config, benchmark_annual_return, strict=True)
    assert report is not None
    return report


@dataclass
class ModelPeriodReport:
    """One report block for the model over one chronological period: comparable stats
    (`PeriodReport`) plus model-only trade/portfolio-exposure stats with no benchmark
    equivalent. `benchmark_correlation` is computed per period (not just once overall) so it
    reflects how the model actually tracked the benchmark within that specific window, same as
    every other stat here."""

    stats: PeriodReport
    trade_statistics: TradeStatistics
    average_fraction_invested: float
    benchmark_correlation: float | None = None


def _model_period_report(
    label: str,
    equity_curve: pd.Series,
    cash_curve: pd.Series,
    trades: list[ClosedTrade],
    position_count_history: list[tuple[pd.Timestamp, int]],
    risk_config: RiskConfig,
    benchmark_annual_return: float | None,
    benchmark_equity_curve: pd.Series | None,
    *,
    strict: bool,
) -> ModelPeriodReport | None:
    stats = _period_report(label, equity_curve, risk_config, benchmark_annual_return, strict=strict)
    if stats is None:
        return None
    start, end = equity_curve.index[0], equity_curve.index[-1]
    sub_trades = [t for t in trades if start <= t.entry_date <= end]
    sub_position_history = [(t, c) for t, c in position_count_history if start <= t <= end]
    benchmark_correlation = None
    if risk_config.calculate_spy_correlation and benchmark_equity_curve is not None:
        benchmark_correlation = _benchmark_correlation(equity_curve, benchmark_equity_curve)
    return ModelPeriodReport(
        stats=stats,
        trade_statistics=compute_trade_statistics(sub_trades, sub_position_history),
        average_fraction_invested=average_fraction_invested(equity_curve, cash_curve),
        benchmark_correlation=benchmark_correlation,
    )


def _model_period_report_strict(
    label: str,
    equity_curve: pd.Series,
    cash_curve: pd.Series,
    trades: list[ClosedTrade],
    position_count_history: list[tuple[pd.Timestamp, int]],
    risk_config: RiskConfig,
    benchmark_annual_return: float | None,
    benchmark_equity_curve: pd.Series | None,
) -> ModelPeriodReport:
    report = _model_period_report(
        label,
        equity_curve,
        cash_curve,
        trades,
        position_count_history,
        risk_config,
        benchmark_annual_return,
        benchmark_equity_curve,
        strict=True,
    )
    assert report is not None
    return report


@dataclass
class BenchmarkReport:
    label: str
    equity_curve: pd.Series
    overall: PeriodReport
    train: PeriodReport | None
    test: PeriodReport | None
    holding_period_hours: int


def compute_benchmark_report(
    prices: pd.Series,
    risk_config: RiskConfig,
    label: str = "SPY",
    train_test_split_date: date | None = None,
) -> BenchmarkReport:
    """Buy-and-hold performance of a single reference ticker (e.g. SPY) over its own historical
    prices, run through the exact same metrics as the strategy itself (day/week/month win counts,
    Sharpe/volatility, block-bootstrap profit odds) -- including, when `train_test_split_date` is
    given, the same train/test breakdown -- so the two are directly comparable period by period."""
    equity_curve = prices / prices.iloc[0]

    overall = _period_report_strict("overall", equity_curve, risk_config, None)

    train = test = None
    if train_test_split_date is not None:
        tz = pd.DatetimeIndex(equity_curve.index).tz
        split = pd.Timestamp(train_test_split_date, tz=tz)
        train = _period_report("train", equity_curve.loc[:split], risk_config, None, strict=False)
        test = _period_report("test", equity_curve.loc[split:], risk_config, None, strict=False)

    return BenchmarkReport(
        label=label,
        equity_curve=equity_curve,
        overall=overall,
        train=train,
        test=test,
        holding_period_hours=risk_config.holding_period_hours,
    )


@dataclass
class BacktestReport:
    equity_curve: pd.Series
    overall: ModelPeriodReport
    train: ModelPeriodReport | None
    test: ModelPeriodReport | None
    holding_period_hours: int
    risk_free_rate: float
    benchmark: BenchmarkReport | None = None


def compute_report(
    simulator: PortfolioSimulator,
    price_series: dict[str, pd.Series],
    risk_config: RiskConfig,
    benchmark_prices: pd.Series | None = None,
    benchmark_label: str = "SPY",
    train_test_split_date: date | None = None,
) -> BacktestReport:
    calendar = pd.DatetimeIndex(sorted(set().union(*(s.index for s in price_series.values()))))
    warmup_bars = min(risk_config.portfolio_warmup_bars, len(calendar) - 1)
    report_calendar = calendar[warmup_bars:]
    equity_curve = build_equity_curve(simulator, price_series, report_calendar)
    cash_curve = _cash_curve(simulator.cash_history, report_calendar)
    # Rescale so the reportable window itself starts at 1.0 -- every metric below (yearly_return,
    # "model return", average_fraction_invested) assumes equity_curve.iloc[0] == 1.0, the same
    # assumption that held for the untrimmed curve before the warmup window was cut off the front.
    baseline = equity_curve.iloc[0]
    equity_curve = equity_curve / baseline
    cash_curve = cash_curve / baseline

    benchmark = None
    if benchmark_prices is not None:
        benchmark = compute_benchmark_report(
            benchmark_prices,
            risk_config,
            label=benchmark_label,
            train_test_split_date=train_test_split_date,
        )

    report_start = equity_curve.index[0]
    reportable_trades = [t for t in simulator.closed_trades if t.entry_date >= report_start]
    reportable_position_history = [
        (t, count) for t, count in simulator.position_count_history if t >= report_start
    ]

    # Each period's beat_benchmark threshold uses that SAME period's benchmark annualized return
    # (not the overall benchmark return for every period), which is why benchmark is computed
    # first above.
    overall = _model_period_report_strict(
        "overall",
        equity_curve,
        cash_curve,
        reportable_trades,
        reportable_position_history,
        risk_config,
        benchmark.overall.annualized_return if benchmark is not None else None,
        benchmark.equity_curve if benchmark is not None else None,
    )

    train = test = None
    if train_test_split_date is not None:
        tz = pd.DatetimeIndex(equity_curve.index).tz
        split = pd.Timestamp(train_test_split_date, tz=tz)
        train = _model_period_report(
            "train",
            equity_curve.loc[:split],
            cash_curve.loc[:split],
            reportable_trades,
            reportable_position_history,
            risk_config,
            (
                benchmark.train.annualized_return
                if benchmark is not None and benchmark.train is not None
                else None
            ),
            benchmark.equity_curve.loc[:split] if benchmark is not None else None,
            strict=False,
        )
        test = _model_period_report(
            "test",
            equity_curve.loc[split:],
            cash_curve.loc[split:],
            reportable_trades,
            reportable_position_history,
            risk_config,
            (
                benchmark.test.annualized_return
                if benchmark is not None and benchmark.test is not None
                else None
            ),
            benchmark.equity_curve.loc[split:] if benchmark is not None else None,
            strict=False,
        )

    return BacktestReport(
        equity_curve=equity_curve,
        overall=overall,
        train=train,
        test=test,
        holding_period_hours=risk_config.holding_period_hours,
        risk_free_rate=risk_config.risk_free_rate,
        benchmark=benchmark,
    )
