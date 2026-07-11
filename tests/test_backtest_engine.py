from __future__ import annotations

import pandas as pd
import pytest

from quantloom.backtest.engine import (
    PortfolioSimulator,
    Position,
    extract_ticker_events,
    run_simulation,
)
from quantloom.config import Direction
from quantloom.config.schema import PositionSizingConfig


def _frame(n: int, **overrides: list) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    base = {
        "close": [100.0] * n,
        "buy_signal_long": [False] * n,
        "sell_signal_long": [False] * n,
        "sell_price_long": [100.0] * n,
        "buy_signal_short": [False] * n,
        "sell_signal_short": [False] * n,
        "sell_price_short": [100.0] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base, index=index)


def _flags(n: int, *true_positions: int) -> list[bool]:
    values = [False] * n
    for pos in true_positions:
        values[pos] = True
    return values


def test_simple_long_buy_then_sell() -> None:
    n = 10
    frame = _frame(
        n,
        buy_signal_long=_flags(n, 2),
        sell_signal_long=_flags(n, 5),
        sell_price_long=[110.0] * n,
    )

    events = extract_ticker_events(frame, frozenset({Direction.LONG}))

    assert [(e.position, e.kind, e.direction) for e in events] == [
        (2, "buy", Direction.LONG),
        (5, "sell", Direction.LONG),
    ]


def test_repeated_buy_signal_while_holding_is_ignored() -> None:
    n = 10
    frame = _frame(
        n,
        buy_signal_long=_flags(n, 2, 3, 4),  # fires 3 days in a row
        sell_signal_long=_flags(n, 6),
    )

    events = extract_ticker_events(frame, frozenset({Direction.LONG}))

    assert [(e.position, e.kind) for e in events] == [(2, "buy"), (6, "sell")]


def test_same_bar_round_trip() -> None:
    n = 10
    frame = _frame(
        n,
        buy_signal_long=_flags(n, 2),
        sell_signal_long=_flags(n, 2),  # fires the SAME bar as the buy
        sell_price_long=[105.0] * n,
    )

    events = extract_ticker_events(frame, frozenset({Direction.LONG}))

    assert [(e.position, e.kind, e.direction) for e in events] == [
        (2, "buy", Direction.LONG),
        (2, "sell", Direction.LONG),
    ]


def test_opposite_direction_signal_forces_close_and_flip() -> None:
    n = 10
    frame = _frame(
        n,
        buy_signal_long=_flags(n, 2),
        buy_signal_short=_flags(n, 5),  # fires while still holding long -> flip
    )

    events = extract_ticker_events(frame, frozenset({Direction.LONG, Direction.SHORT}))

    # the resulting SHORT position never gets its own sell signal, so it's force-closed
    # on the final bar, same as any other still-open position at the end of the series
    assert [(e.position, e.kind, e.direction) for e in events] == [
        (2, "buy", Direction.LONG),
        (5, "sell", Direction.LONG),
        (5, "buy", Direction.SHORT),
        (n - 1, "sell", Direction.SHORT),
    ]


def test_position_still_open_at_end_is_force_closed_on_the_last_bar() -> None:
    n = 10
    frame = _frame(n, buy_signal_long=_flags(n, 2))

    events = extract_ticker_events(frame, frozenset({Direction.LONG}))

    assert events[-1].position == n - 1
    assert events[-1].kind == "sell"


def _sizing(**overrides) -> PositionSizingConfig:
    defaults = dict(max_positions=2)
    defaults.update(overrides)
    return PositionSizingConfig(**defaults)


def test_simulator_buy_and_sell_realizes_correct_profit() -> None:
    sim = PortfolioSimulator(config=_sizing(max_positions=1))
    date0 = pd.Timestamp("2024-01-01")
    date1 = pd.Timestamp("2024-01-02")

    sim.buy("AAPL", 0, date0, 100.0, direction=Direction.LONG)
    assert sim.cash == pytest.approx(0.0)  # trade_size = cash/(max_positions-0) = 1.0, all of it

    sim.sell("AAPL", 1, date1, 110.0)

    assert len(sim.closed_trades) == 1
    trade = sim.closed_trades[0]
    assert trade.profit == pytest.approx(0.10)
    assert sim.cash == pytest.approx(1.10)


def test_simulator_short_profit_is_correct_when_price_falls() -> None:
    sim = PortfolioSimulator(config=_sizing(max_positions=1))
    sim.buy("AAPL", 0, pd.Timestamp("2024-01-01"), 100.0, direction=Direction.SHORT)
    sim.sell("AAPL", 1, pd.Timestamp("2024-01-02"), 90.0)

    trade = sim.closed_trades[0]
    assert trade.profit == pytest.approx(0.10)  # price fell 10% -> short profits 10%


def test_simulator_enforces_hard_position_cap() -> None:
    sim = PortfolioSimulator(config=_sizing(max_positions=1))
    date = pd.Timestamp("2024-01-01")

    sim.buy("AAPL", 0, date, 100.0, direction=Direction.LONG)
    sim.buy("MSFT", 0, date, 100.0, direction=Direction.LONG)  # rejected: at cap

    assert set(sim.positions.keys()) == {"AAPL"}


def test_trade_size_splits_current_cash_evenly_across_remaining_slots() -> None:
    # trade_size = cash / (max_positions - currently_open) is self-limiting by construction --
    # it's derived from CURRENT cash, not a stale estimated_value that ignores unrealized losses
    # on still-open positions, so it can never overspend into negative cash.
    # Concretely (max_positions=2): buy A and B at trade_size=0.5 each (cash=0). Sell A at a 50%
    # loss -> cash=0.25. B alone means 1 open slot is free, so C is sized at all remaining cash
    # (0.25 / 1 remaining slot = 0.25), landing exactly at zero, never negative.
    sim = PortfolioSimulator(config=_sizing(max_positions=2))
    date = pd.Timestamp("2024-01-01")

    sim.buy("A", 0, date, 100.0, direction=Direction.LONG)
    sim.buy("B", 0, date, 100.0, direction=Direction.LONG)
    assert sim.cash == pytest.approx(0.0)

    sim.sell("A", 1, date, 50.0)  # 50% loss
    assert sim.cash == pytest.approx(0.25)

    sim.buy("C", 2, date, 100.0, direction=Direction.LONG)

    assert "C" in sim.positions
    assert sim.positions["C"].trade_size == pytest.approx(0.25)
    assert sim.cash == pytest.approx(0.0)


def _daily_returns(dates: pd.DatetimeIndex, values: list[float]) -> pd.Series:
    return pd.Series(values, index=dates)


def test_correlated_entry_is_rejected_at_or_above_threshold() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.01, -0.01, 0.02, -0.02]
    a_returns = _daily_returns(dates, values)
    aligned_returns = pd.DataFrame({"A": a_returns, "B": a_returns})  # B tracks A exactly -> r=1.0

    sizing = _sizing(
        max_positions=2, reject_correlated_entries=True,
        correlation_reject_threshold=0.75, correlation_lookback_bars=90,
    )
    sim = PortfolioSimulator(config=sizing, aligned_returns=aligned_returns)
    sim.positions["A"] = Position("A", Direction.LONG, 1.0, 100.0, dates[0], 0, 0.5)

    sim.buy("B", 5, dates[5], 100.0, direction=Direction.LONG)

    assert "B" not in sim.positions


def test_uncorrelated_entry_is_not_rejected() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    a_returns = _daily_returns(dates, [0.01, -0.02, 0.015, -0.03, 0.02, 0.005, -0.01, 0.025])
    b_returns = _daily_returns(dates, [0.02, 0.01, -0.01, -0.005, 0.03, -0.02, 0.015, 0.01])
    aligned_returns = pd.DataFrame({"A": a_returns, "B": b_returns})
    assert a_returns.corr(b_returns) < 0.75  # sanity check on the fixture itself

    sizing = _sizing(
        max_positions=2, reject_correlated_entries=True,
        correlation_reject_threshold=0.75, correlation_lookback_bars=90,
    )
    sim = PortfolioSimulator(config=sizing, aligned_returns=aligned_returns)
    sim.positions["A"] = Position("A", Direction.LONG, 1.0, 100.0, dates[0], 0, 0.5)

    sim.buy("B", 7, dates[-1], 100.0, direction=Direction.LONG)

    assert "B" in sim.positions


def test_negatively_correlated_entry_is_not_rejected() -> None:
    # a strong hedge (returns move opposite to a holding) must never be blocked -- the gate uses
    # SIGNED correlation, not absolute value, since a negatively-correlated position reduces
    # portfolio risk rather than duplicating it.
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.01, -0.01, 0.02, -0.02]
    a_returns = _daily_returns(dates, values)
    b_returns = -a_returns  # perfectly INVERSE -- correlation -1.0

    sizing = _sizing(
        max_positions=2, reject_correlated_entries=True,
        correlation_reject_threshold=0.75, correlation_lookback_bars=90,
    )
    sim = PortfolioSimulator(
        config=sizing, aligned_returns=pd.DataFrame({"A": a_returns, "B": b_returns})
    )
    sim.positions["A"] = Position("A", Direction.LONG, 1.0, 100.0, dates[0], 0, 0.5)

    sim.buy("B", 5, dates[5], 100.0, direction=Direction.LONG)

    assert "B" in sim.positions


def test_rejection_triggers_on_worst_pairing_not_the_average() -> None:
    # candidate C is uncorrelated with A but perfectly correlated with B -- a book-average
    # correlation would dilute this away, but checking the single worst pairing catches it.
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    c_values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.01, -0.01, 0.02, -0.02]
    c_returns = _daily_returns(dates, c_values)
    a_values = [0.02, 0.01, -0.01, -0.005, 0.03, -0.02, 0.015, 0.01, 0.0, 0.02]
    a_returns = _daily_returns(dates, a_values)
    aligned_returns = pd.DataFrame({"A": a_returns, "B": c_returns, "C": c_returns})

    sizing = _sizing(
        max_positions=3, reject_correlated_entries=True,
        correlation_reject_threshold=0.75, correlation_lookback_bars=90,
    )
    sim = PortfolioSimulator(config=sizing, aligned_returns=aligned_returns)
    sim.positions["A"] = Position("A", Direction.LONG, 1.0, 100.0, dates[0], 0, 0.3)
    sim.positions["B"] = Position("B", Direction.LONG, 1.0, 100.0, dates[0], 0, 0.3)

    sim.buy("C", 5, dates[5], 100.0, direction=Direction.LONG)

    assert "C" not in sim.positions


def test_correlation_reject_threshold_is_configurable() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    a_returns = _daily_returns(dates, [0.01, -0.02, 0.015, -0.03, 0.02, 0.005, -0.01, 0.025])
    b_returns = _daily_returns(dates, [0.02, 0.01, -0.01, -0.005, 0.03, -0.02, 0.015, 0.01])
    correlation = a_returns.corr(b_returns)
    aligned_returns = pd.DataFrame({"A": a_returns, "B": b_returns})

    lenient = _sizing(
        max_positions=2, reject_correlated_entries=True,
        correlation_reject_threshold=correlation + 0.01, correlation_lookback_bars=90,
    )
    strict = _sizing(
        max_positions=2, reject_correlated_entries=True,
        correlation_reject_threshold=correlation - 0.01, correlation_lookback_bars=90,
    )

    sim_lenient = PortfolioSimulator(config=lenient, aligned_returns=aligned_returns)
    sim_lenient.positions["A"] = Position("A", Direction.LONG, 1.0, 100.0, dates[0], 0, 0.5)
    sim_lenient.buy("B", 7, dates[-1], 100.0, direction=Direction.LONG)
    assert "B" in sim_lenient.positions

    sim_strict = PortfolioSimulator(config=strict, aligned_returns=aligned_returns)
    sim_strict.positions["A"] = Position("A", Direction.LONG, 1.0, 100.0, dates[0], 0, 0.5)
    sim_strict.buy("B", 7, dates[-1], 100.0, direction=Direction.LONG)
    assert "B" not in sim_strict.positions


def test_simulator_rejects_duplicate_buy_for_already_held_ticker() -> None:
    sim = PortfolioSimulator(config=_sizing(max_positions=5))
    date = pd.Timestamp("2024-01-01")

    sim.buy("AAPL", 0, date, 100.0, direction=Direction.LONG)
    cash_after_first_buy = sim.cash
    sim.buy("AAPL", 1, date, 105.0, direction=Direction.LONG)

    assert sim.cash == cash_after_first_buy
    assert len(sim.positions) == 1


def test_run_simulation_merges_multiple_tickers_chronologically() -> None:
    n = 10
    frame_a = _frame(n, buy_signal_long=_flags(n, 1), sell_signal_long=_flags(n, 4))
    frame_b = _frame(n, buy_signal_long=_flags(n, 2), sell_signal_long=_flags(n, 6))

    simulator = run_simulation(
        {"A": frame_a, "B": frame_b}, frozenset({Direction.LONG}), _sizing(max_positions=5)
    )

    assert len(simulator.closed_trades) == 2
    tickers_closed = {trade.ticker for trade in simulator.closed_trades}
    assert tickers_closed == {"A", "B"}
