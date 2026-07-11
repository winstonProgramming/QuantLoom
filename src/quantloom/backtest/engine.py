"""Event-driven portfolio simulation.

Each ticker's signal columns are walked once into a small ordered list of buy/sell events via an
explicit state machine (flat -> long -> short -> ...); combination cases (e.g. a same-bar round
trip, or an opposite-direction signal forcing an immediate close-and-flip) fall out of the state
machine directly instead of needing their own named branch.

Same-bar-close execution: both entries and exits fill at the bar's close (or, for an intrabar-level
sell trigger, at that specific level -- see strategy/sell_rules.py). This avoids the look-ahead
bias a same-day-open fill would introduce: filling at the open would act on a signal computed from
that same bar's close, information not actually available yet at the fill price.

Two enforcement details worth being explicit about:
1. `max_positions` is a hard concurrent-position cap, enforced unconditionally in `buy()` --
   not just a divisor in the position-sizing formula.
2. The trade size deducted from cash and the trade size recorded for later P&L attribution are
   the same value, computed once in `_trade_size()` -- keeping these in sync matters because
   `estimated_value` doesn't mark still-open positions to market, so deriving the recorded size
   from it separately would let realized P&L drift from the capital actually risked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantloom.config import Direction, PositionSizingConfig


@dataclass(frozen=True)
class RawEvent:
    position: int
    date: pd.Timestamp
    kind: str  # "buy" or "sell"
    direction: Direction
    price: float


def extract_ticker_events(frame: pd.DataFrame, directions: frozenset[Direction]) -> list[RawEvent]:
    """One ticker's buy/sell events, enforcing "at most one open position at a time".

    Priority when multiple signals fire on the same bar: a bullish buy is checked before a
    bullish sell, before a bearish buy, before a bearish sell. An opposite-direction buy
    signal while holding closes the current position first (a flip); a sell signal firing
    on the very same bar a position opens closes it immediately (a same-bar round trip) --
    both fall out of the state machine below without a dedicated branch.
    """
    has_long = Direction.LONG in directions
    has_short = Direction.SHORT in directions
    n = len(frame)
    close = frame["close"]
    dates = frame.index

    def buy_event(day: int, direction: Direction) -> RawEvent:
        return RawEvent(day, dates[day], "buy", direction, close.iloc[day])

    def sell_event(day: int, direction: Direction, *, at_close: bool = False) -> RawEvent:
        suffix = direction.value
        price = close.iloc[day] if at_close else frame[f"sell_price_{suffix}"].iloc[day]
        return RawEvent(day, dates[day], "sell", direction, price)

    events: list[RawEvent] = []
    state: Direction | None = None

    for day in range(n):
        if has_long and frame["buy_signal_long"].iloc[day] and state is not Direction.LONG:
            if state is Direction.SHORT:
                events.append(sell_event(day, Direction.SHORT, at_close=True))
            events.append(buy_event(day, Direction.LONG))
            state = Direction.LONG
            if frame["sell_signal_long"].iloc[day]:
                events.append(sell_event(day, Direction.LONG))
                state = None
            continue

        if has_long and state is Direction.LONG and frame["sell_signal_long"].iloc[day]:
            events.append(sell_event(day, Direction.LONG))
            state = None
            continue

        if has_short and frame["buy_signal_short"].iloc[day] and state is not Direction.SHORT:
            if state is Direction.LONG:
                events.append(sell_event(day, Direction.LONG, at_close=True))
            events.append(buy_event(day, Direction.SHORT))
            state = Direction.SHORT
            if frame["sell_signal_short"].iloc[day]:
                events.append(sell_event(day, Direction.SHORT))
                state = None
            continue

        if has_short and state is Direction.SHORT and frame["sell_signal_short"].iloc[day]:
            events.append(sell_event(day, Direction.SHORT))
            state = None
            continue

    if state is not None:
        events.append(RawEvent(n - 1, dates[-1], "sell", state, close.iloc[-1]))

    return events


@dataclass
class Position:
    ticker: str
    direction: Direction
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_position: int
    trade_size: float


@dataclass
class ClosedTrade:
    ticker: str
    direction: Direction
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    trade_size: float
    profit: float
    duration_bars: int


def _trade_profit(direction: Direction, entry_price: float, exit_price: float) -> float:
    ratio = exit_price / entry_price
    return ratio - 1 if direction is Direction.LONG else 1 - ratio


@dataclass
class PortfolioSimulator:
    config: PositionSizingConfig
    # None unless config.reject_correlated_entries is on (see run_simulation) -- the correlation
    # check below always returns False (never rejects) when this isn't wired up, so constructing
    # a PortfolioSimulator directly without it (as most tests do) is always safe.
    aligned_returns: pd.DataFrame | None = None
    cash: float = 1.0
    estimated_value: float = 1.0
    positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    cash_history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    estimated_value_history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    position_count_history: list[tuple[pd.Timestamp, int]] = field(default_factory=list)

    def _is_too_correlated_with_held(self, ticker: str, date: pd.Timestamp) -> bool:
        """True if `ticker`'s trailing returns (over correlation_lookback_bars, ending at `date`
        -- same-bar-close convention, no future information) are correlated at or above
        correlation_reject_threshold with ANY single currently-held ticker.

        Deliberately checks the single worst (most positively correlated) pairing rather than an
        average across the book -- one dangerously correlated pair is exactly the risk this gate
        exists to catch, and an average could dilute it away. Uses SIGNED correlation, not
        absolute value: a candidate strongly *negatively* correlated with a holding is a natural
        hedge, not redundant risk, so it's never rejected on that basis. Bar-based (not calendar
        days) so the estimate's statistical reliability (driven by observation count) stays
        consistent regardless of candle_length -- see PositionSizingConfig.correlation_lookback_bars.
        """
        if self.aligned_returns is None or ticker not in self.aligned_returns:
            return False
        held_tickers = [t for t in self.positions if t != ticker and t in self.aligned_returns]
        if not held_tickers:
            return False
        window = self.aligned_returns.loc[:date].tail(self.config.correlation_lookback_bars)
        if len(window) < 2:
            return False
        correlations = window[held_tickers].corrwith(window[ticker])
        max_correlation = correlations.max()
        return bool(max_correlation >= self.config.correlation_reject_threshold)

    def _trade_size(self) -> float:
        """Splits current cash evenly across the remaining open slots -- self-limiting by
        construction (remaining_slots >= 1 whenever this is reached, since `buy` already
        rejected the call once at the max_positions cap), so this can never overspend cash the
        way an estimated_value-based formula can (estimated_value doesn't mark still-open
        positions to market, so it can silently overstate what's actually available)."""
        remaining_slots = self.config.max_positions - len(self.positions)
        return self.cash / remaining_slots

    def _record(self, date: pd.Timestamp) -> None:
        self.cash_history.append((date, self.cash))
        self.estimated_value_history.append((date, self.estimated_value))
        self.position_count_history.append((date, len(self.positions)))

    def buy(
        self,
        ticker: str,
        position: int,
        date: pd.Timestamp,
        price: float,
        direction: Direction,
    ) -> None:
        if ticker in self.positions or len(self.positions) >= self.config.max_positions:
            return
        if self.config.reject_correlated_entries and self._is_too_correlated_with_held(
            ticker, date
        ):
            return
        trade_size = self._trade_size()
        held_size = sum(p.trade_size for p in self.positions.values())
        self.cash -= trade_size
        self.estimated_value = self.cash + held_size + trade_size
        self.positions[ticker] = Position(
            ticker, direction, trade_size / price, price, date, position, trade_size
        )
        self._record(date)

    def sell(self, ticker: str, position: int, date: pd.Timestamp, price: float) -> None:
        held = self.positions.pop(ticker, None)
        if held is None:
            return
        profit = _trade_profit(held.direction, held.entry_price, price)
        self.cash += held.trade_size * (1 + profit)
        self.estimated_value = self.cash + sum(p.trade_size for p in self.positions.values())
        self.closed_trades.append(
            ClosedTrade(
                ticker=ticker,
                direction=held.direction,
                entry_date=held.entry_date,
                exit_date=date,
                entry_price=held.entry_price,
                exit_price=price,
                trade_size=held.trade_size,
                profit=profit,
                duration_bars=position - held.entry_position,
            )
        )
        self._record(date)


def run_simulation(
    ticker_frames: dict[str, pd.DataFrame],
    directions: frozenset[Direction],
    config: PositionSizingConfig,
) -> PortfolioSimulator:
    """Merge every ticker's events into one chronological stream and simulate them in order."""
    all_events: list[tuple[str, RawEvent]] = []
    for ticker, frame in ticker_frames.items():
        for event in extract_ticker_events(frame, directions):
            all_events.append((ticker, event))
    all_events.sort(key=lambda item: (item[1].date, 0 if item[1].kind == "sell" else 1))

    aligned_returns = None
    if config.reject_correlated_entries and ticker_frames:
        aligned_returns = pd.concat(
            {ticker: frame["close"].pct_change() for ticker, frame in ticker_frames.items()},
            axis=1,
        )

    simulator = PortfolioSimulator(config=config, aligned_returns=aligned_returns)
    for ticker, event in all_events:
        if event.kind == "buy":
            simulator.buy(ticker, event.position, event.date, event.price, event.direction)
        else:
            simulator.sell(ticker, event.position, event.date, event.price)
    return simulator
