from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantloom.backtest.engine import ClosedTrade, PortfolioSimulator
from quantloom.backtest.metrics import (
    build_equity_curve,
    compute_benchmark_report,
    compute_report,
    compute_risk_metrics,
    compute_trade_statistics,
    daily_outcomes,
    monthly_outcomes,
    simulate_profit_odds,
    weekly_outcomes,
)
from quantloom.config import Direction
from quantloom.config.schema import PositionSizingConfig, RiskConfig


def _calendar(n: int, freq: str = "h") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=freq)


def _position_sizing() -> PositionSizingConfig:
    return PositionSizingConfig(max_positions=1)


def test_build_equity_curve_reflects_cash_before_during_and_after_a_trade() -> None:
    calendar = _calendar(10)
    sim = PortfolioSimulator(config=_position_sizing())
    sim.buy("AAPL", 2, calendar[2], 100.0, direction=Direction.LONG)
    sim.sell("AAPL", 6, calendar[6], 110.0)

    prices = pd.Series([100.0] * 3 + [105.0, 108.0, 110.0, 110.0] + [110.0] * 3, index=calendar)

    equity = build_equity_curve(sim, {"AAPL": prices}, calendar)

    # before the trade: all cash, no position -> equity == starting cash (1.0)
    assert equity.iloc[0] == pytest.approx(1.0)
    # mid-trade (bar 4): cash (0.0, all committed) + trade_size * (price/entry)
    trade_size = 1.0  # cash/(max_positions-0)=1 with the sizing config above
    assert equity.iloc[4] == pytest.approx(0.0 + trade_size * (108.0 / 100.0))
    # after the trade closes: back to being all cash, now at the realized P&L
    assert equity.iloc[7] == pytest.approx(1.0 * 1.10)


def test_build_equity_curve_marks_still_open_positions_to_market() -> None:
    # A position still open at the end of the report window must not just vanish from equity --
    # buy() already deducted its cost from cash, so if it's never marked back to market, its
    # capital silently disappears regardless of whether it's actually winning or losing.
    calendar = _calendar(10)
    sim = PortfolioSimulator(config=_position_sizing())
    sim.buy("AAPL", 2, calendar[2], 100.0, direction=Direction.LONG)

    prices = pd.Series([100.0] * 3 + [105.0, 108.0, 110.0, 120.0] + [120.0] * 3, index=calendar)

    equity = build_equity_curve(sim, {"AAPL": prices}, calendar)

    # all cash committed to the still-open position, marked to the current price
    assert equity.iloc[4] == pytest.approx(0.0 + 1.0 * (108.0 / 100.0))
    assert equity.iloc[-1] == pytest.approx(0.0 + 1.0 * (120.0 / 100.0))


def test_compute_risk_metrics_hand_computed() -> None:
    calendar = _calendar(24 * 30)  # 30 days of hourly bars
    # simple deterministic geometric growth: 0.01% per bar, zero volatility
    growth = 1.0001
    equity = pd.Series(growth ** np.arange(len(calendar)), index=calendar)

    result = compute_risk_metrics(equity, risk_free_rate=0.0)

    elapsed_years = (calendar[-1] - calendar[0]).days / 365.25
    expected_yearly_return = equity.iloc[-1] ** (1 / elapsed_years) - 1
    assert result.yearly_return == pytest.approx(expected_yearly_return)
    # deterministic growth -> zero volatility
    assert result.annual_volatility == pytest.approx(0.0, abs=1e-9)


def test_compute_risk_metrics_raises_on_non_positive_equity() -> None:
    # an unbounded short-side loss (or any other cause) can drive total equity to zero/negative --
    # yearly_return/volatility/sharpe are all undefined at that point, so this must raise loudly
    # rather than silently produce NaN via a fractional power of a negative number.
    calendar = _calendar(5, freq="D")
    equity = pd.Series([1.0, 0.5, -0.1, -0.6, -0.6], index=calendar)

    with pytest.raises(ValueError, match="non-positive"):
        compute_risk_metrics(equity, risk_free_rate=0.0)


def test_daily_outcomes_counts_up_down_flat_days() -> None:
    days = pd.date_range("2024-01-01", periods=4, freq="D")
    equity = pd.Series([1.0, 1.1, 1.05, 1.05], index=days)

    won, lost, flat = daily_outcomes(equity)

    assert (won, lost, flat) == (1, 1, 1)


def test_weekly_outcomes_counts_up_down_flat_weeks() -> None:
    weeks = pd.date_range("2024-01-07", periods=4, freq="W")
    equity = pd.Series([1.0, 1.1, 1.05, 1.05], index=weeks)

    won, lost, flat = weekly_outcomes(equity)

    assert (won, lost, flat) == (1, 1, 1)


def test_monthly_outcomes_counts_up_down_flat_months() -> None:
    months = pd.date_range("2024-01-31", periods=4, freq="ME")
    equity = pd.Series([1.0, 1.1, 1.05, 1.05], index=months)

    won, lost, flat = monthly_outcomes(equity)

    assert (won, lost, flat) == (1, 1, 1)


def test_compute_trade_statistics_hand_computed() -> None:
    trades = [
        ClosedTrade(
            "A", Direction.LONG,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"),
            100, 110, 0.1, 0.10, 5,
        ),
        ClosedTrade(
            "B", Direction.LONG,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"),
            100, 90, 0.1, -0.10, 10,
        ),
    ]
    history = [(pd.Timestamp("2024-01-01"), 2)]

    stats = compute_trade_statistics(trades, position_count_history=history)

    assert stats.trade_count == 2
    assert stats.trades_won == 1
    assert stats.trades_lost == 1
    assert stats.average_duration_bars == pytest.approx(7.5)
    assert stats.most_concurrent_positions == 2
    assert stats.average_win == pytest.approx(0.10)
    assert stats.average_loss == pytest.approx(-0.10)


def test_compute_trade_statistics_handles_zero_trades_without_raising() -> None:
    # a real scenario (a tight sell rule, short backtest window), not just a theoretical edge
    # case -- statistics.geometric_mean/mean both raise on an empty input if not guarded.
    stats = compute_trade_statistics([], position_count_history=[])

    assert stats.trade_count == 0
    assert stats.most_concurrent_positions == 0
    assert math.isnan(stats.average_profit_per_trade)
    assert math.isnan(stats.average_win)
    assert math.isnan(stats.average_duration_bars)


def test_simulate_profit_odds_is_reproducible_with_a_seed() -> None:
    returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, size=500))
    config = RiskConfig(
        monte_carlo_paths=1000,
        holding_period_hours=50,
        monte_carlo_block_size=5,
    )

    seed = 7
    result_a = simulate_profit_odds(
        returns, config, bars_per_year=252 * 7, rng=np.random.default_rng(seed)
    )
    result_b = simulate_profit_odds(
        returns, config, bars_per_year=252 * 7, rng=np.random.default_rng(seed)
    )

    assert result_a == result_b


def test_simulate_profit_odds_responds_to_the_underlying_return_distribution() -> None:
    always_positive = pd.Series([0.01] * 200)
    always_negative = pd.Series([-0.01] * 200)
    config = RiskConfig(
        monte_carlo_paths=200,
        holding_period_hours=20,
        monte_carlo_block_size=5,
    )

    odds_positive = simulate_profit_odds(
        always_positive, config, bars_per_year=252 * 7, rng=np.random.default_rng(1)
    )
    odds_negative = simulate_profit_odds(
        always_negative, config, bars_per_year=252 * 7, rng=np.random.default_rng(1)
    )

    assert odds_positive.profit_within_holding_period == pytest.approx(1.0)
    assert odds_negative.profit_within_holding_period == pytest.approx(0.0)
    # a strictly negative path can never clear the (positive) risk-free rate either
    assert odds_negative.beat_risk_free_rate == pytest.approx(0.0)


def test_simulate_profit_odds_computes_beat_benchmark_only_when_given() -> None:
    # bars_per_year == holding_period_hours makes the holding period exactly one simulated
    # "year", so _compounded_threshold(annual_rate, ...) reduces to (1 + annual_rate) directly --
    # keeps the arithmetic here easy to hand-check.
    returns = pd.Series([0.01] * 200)  # deterministic path: 1.01**20 ≈ 1.22
    config = RiskConfig(monte_carlo_paths=50, holding_period_hours=20)

    without_benchmark = simulate_profit_odds(
        returns, config, bars_per_year=20, rng=np.random.default_rng(1)
    )
    assert without_benchmark.beat_benchmark is None

    with_easy_benchmark = simulate_profit_odds(
        returns,
        config,
        bars_per_year=20,
        benchmark_annual_return=0.0,  # threshold 1.0 -- 1.22 clears it
        rng=np.random.default_rng(1),
    )
    with_hard_benchmark = simulate_profit_odds(
        returns,
        config,
        bars_per_year=20,
        benchmark_annual_return=2.0,  # threshold 3.0 -- 1.22 doesn't clear it
        rng=np.random.default_rng(1),
    )
    assert with_easy_benchmark.beat_benchmark == pytest.approx(1.0)
    assert with_hard_benchmark.beat_benchmark == pytest.approx(0.0)


def test_compute_report_end_to_end() -> None:
    calendar = _calendar(24 * 60)
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)

    sim = PortfolioSimulator(config=_position_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 500, calendar[500], prices.iloc[500])

    report = compute_report(sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=100))

    assert report.overall.trade_statistics.trade_count == 1
    assert report.overall.stats.sharpe_ratio is not None
    assert report.overall.stats.profit_odds is not None
    assert not math.isnan(report.overall.stats.sharpe_ratio)
    assert report.benchmark is None


def test_compute_benchmark_report_hand_computed() -> None:
    calendar = _calendar(24 * 30, freq="D")
    # deterministic 0.1%/day growth -- known total return, zero volatility
    growth = 1.001
    prices = pd.Series(100 * growth ** np.arange(len(calendar)), index=calendar)

    benchmark = compute_benchmark_report(prices, RiskConfig(monte_carlo_paths=100), label="SPY")

    assert benchmark.label == "SPY"
    expected_return = prices.iloc[-1] / prices.iloc[0] - 1
    assert (benchmark.equity_curve.iloc[-1] - 1) == pytest.approx(expected_return)
    assert benchmark.overall.days_won == len(calendar) - 1  # every day is a strict improvement
    assert benchmark.overall.days_lost == 0
    assert benchmark.overall.volatility == pytest.approx(0.0, abs=1e-9)
    assert benchmark.overall.profit_odds is not None


def test_compute_report_includes_benchmark_when_prices_given() -> None:
    calendar = _calendar(24 * 60)
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)
    benchmark_prices = pd.Series(400 + walk * 0.5, index=calendar)

    sim = PortfolioSimulator(config=_position_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 500, calendar[500], prices.iloc[500])

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
    )

    assert report.benchmark is not None
    assert report.benchmark.label == "SPY"
    assert report.benchmark.overall.sharpe_ratio is not None
    # the position is only held over part of the window, but while held it's the same
    # underlying random walk as the benchmark -> positively correlated
    assert report.overall.benchmark_correlation is not None
    assert report.overall.benchmark_correlation > 0.3


def test_compute_report_omits_benchmark_correlation_when_disabled() -> None:
    calendar = _calendar(24 * 60)
    walk = np.random.default_rng(0).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)
    benchmark_prices = pd.Series(400 + walk * 0.5, index=calendar)
    sim = PortfolioSimulator(config=_position_sizing())

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100, calculate_spy_correlation=False),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
    )

    assert report.benchmark is not None
    assert report.overall.benchmark_correlation is None


def test_compute_report_omits_train_test_split_when_no_date_given() -> None:
    calendar = _calendar(24 * 60)
    prices = pd.Series(100.0, index=calendar)
    sim = PortfolioSimulator(config=_position_sizing())

    report = compute_report(sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=50))

    assert report.train is None
    assert report.test is None


def test_compute_report_excludes_warmup_bars_from_equity_curve_and_trade_statistics() -> None:
    calendar = _calendar(24 * 60)
    prices = pd.Series(100.0, index=calendar)
    sim = PortfolioSimulator(config=_position_sizing())
    # entirely inside the 10-bar warmup window -- should be excluded from the report
    sim.buy("AAPL", 2, calendar[2], 100.0, direction=Direction.LONG)
    sim.sell("AAPL", 4, calendar[4], 105.0)
    # after the warmup window -- should be the only trade counted
    sim.buy("AAPL", 20, calendar[20], 100.0, direction=Direction.LONG)
    sim.sell("AAPL", 30, calendar[30], 110.0)

    report = compute_report(
        sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=50, portfolio_warmup_bars=10)
    )

    assert report.equity_curve.index[0] == calendar[10]
    assert report.equity_curve.iloc[0] == pytest.approx(1.0)
    assert report.overall.trade_statistics.trade_count == 1


def test_compute_report_zero_warmup_bars_matches_untrimmed_behavior() -> None:
    calendar = _calendar(24 * 60)
    prices = pd.Series(100.0, index=calendar)
    sim = PortfolioSimulator(config=_position_sizing())
    sim.buy("AAPL", 2, calendar[2], 100.0, direction=Direction.LONG)
    sim.sell("AAPL", 4, calendar[4], 105.0)

    report = compute_report(
        sim, {"AAPL": prices}, RiskConfig(monte_carlo_paths=50, portfolio_warmup_bars=0)
    )

    assert report.equity_curve.index[0] == calendar[0]
    assert report.overall.trade_statistics.trade_count == 1


def test_compute_report_splits_train_and_test_periods_by_entry_date() -> None:
    calendar = _calendar(24 * 60)  # 60 days of hourly bars starting 2024-01-01
    walk = np.random.default_rng(1).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)

    sim = PortfolioSimulator(config=_position_sizing())
    # one trade well before the split, one well after
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 100, calendar[100], prices.iloc[100])
    sim.buy("AAPL", 800, calendar[800], prices.iloc[800], direction=Direction.LONG)
    sim.sell("AAPL", 900, calendar[900], prices.iloc[900])

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100),
        train_test_split_date=date(2024, 1, 20),
    )

    assert report.train is not None
    assert report.test is not None
    assert report.train.trade_statistics.trade_count == 1
    assert report.test.trade_statistics.trade_count == 1
    split = pd.Timestamp(date(2024, 1, 20), tz=report.equity_curve.index.tz)
    assert report.train.stats.start < split <= report.test.stats.start
    assert report.train.stats.end <= split
    assert report.test.stats.end == report.equity_curve.index[-1]


def test_compute_benchmark_report_splits_train_and_test_when_split_date_given() -> None:
    # the benchmark's own train/test breakdown didn't exist before this -- only the model's did --
    # so the console report's two-column tables had nothing to compare against for train/test.
    calendar = _calendar(24 * 60, freq="D")
    growth = 1.001
    prices = pd.Series(100 * growth ** np.arange(len(calendar)), index=calendar)

    benchmark = compute_benchmark_report(
        prices,
        RiskConfig(monte_carlo_paths=100),
        label="SPY",
        train_test_split_date=date(2024, 1, 30),
    )

    assert benchmark.train is not None
    assert benchmark.test is not None
    split = pd.Timestamp(date(2024, 1, 30), tz=benchmark.equity_curve.index.tz)
    assert benchmark.train.start < split <= benchmark.test.start
    assert benchmark.train.end <= split
    assert benchmark.test.end == benchmark.equity_curve.index[-1]
    # deterministic, positive, monotonic growth in both sub-periods -> a real positive return
    assert benchmark.train.period_return > 0
    assert benchmark.test.period_return > 0


def test_compute_report_beat_benchmark_uses_the_matching_sub_periods_own_benchmark_return() -> None:
    # each period's beat_benchmark threshold must come from the BENCHMARK's return over that same
    # period, not the benchmark's overall return reused for every period.
    calendar = _calendar(24 * 60)
    walk = np.random.default_rng(2).normal(size=len(calendar)).cumsum()
    prices = pd.Series(100 + walk, index=calendar)
    benchmark_prices = pd.Series(100.0 * 1.0002 ** np.arange(len(calendar)), index=calendar)

    sim = PortfolioSimulator(config=_position_sizing())
    sim.buy("AAPL", 10, calendar[10], prices.iloc[10], direction=Direction.LONG)
    sim.sell("AAPL", 900, calendar[900], prices.iloc[900])

    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100, simulate_profit_odds=True),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
        train_test_split_date=date(2024, 1, 20),
    )

    assert report.benchmark is not None
    assert report.train is not None and report.test is not None
    assert report.train.stats.profit_odds is not None
    assert report.test.stats.profit_odds is not None
    # both periods have a benchmark to compare against -- beat_benchmark must be populated (not
    # None) for each, using that period's own benchmark sub-report
    assert report.train.stats.profit_odds.beat_benchmark is not None
    assert report.test.stats.profit_odds.beat_benchmark is not None


def test_compute_report_benchmark_correlation_is_computed_per_period_not_reused_overall() -> None:
    # each period's correlation must come from THAT period's own (equity, benchmark) slice --
    # constructed here so the train and test windows are correlated oppositely, which only a
    # genuinely independent per-period computation can distinguish.
    calendar = _calendar(24 * 60)
    n = len(calendar)
    split_idx = n // 2
    rng = np.random.default_rng(3)
    shared_walk = rng.normal(size=split_idx).cumsum()
    train_prices = 100 + shared_walk
    train_benchmark = 400 + shared_walk * 0.5  # train: perfectly correlated with the model
    test_prices = 100 + rng.normal(size=n - split_idx).cumsum()
    test_benchmark = 400 - rng.normal(size=n - split_idx).cumsum()  # test: independent draws
    prices = pd.Series(np.concatenate([train_prices, test_prices]), index=calendar)
    benchmark_prices = pd.Series(
        np.concatenate([train_benchmark, test_benchmark]), index=calendar
    )

    sim = PortfolioSimulator(config=_position_sizing())
    sim.buy("AAPL", 0, calendar[0], prices.iloc[0], direction=Direction.LONG)
    sim.sell("AAPL", n - 1, calendar[n - 1], prices.iloc[n - 1])

    split_date = calendar[split_idx].date()
    report = compute_report(
        sim,
        {"AAPL": prices},
        RiskConfig(monte_carlo_paths=100),
        benchmark_prices=benchmark_prices,
        benchmark_label="SPY",
        train_test_split_date=split_date,
    )

    assert report.train is not None and report.test is not None
    assert report.overall.benchmark_correlation is not None
    assert report.train.benchmark_correlation is not None
    assert report.test.benchmark_correlation is not None
    assert report.train.benchmark_correlation > 0.9
    # train and test must differ -- proves test wasn't just the overall/train value reused
    assert report.train.benchmark_correlation != report.test.benchmark_correlation
