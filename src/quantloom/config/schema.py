"""Typed, validated configuration for the backtesting pipeline.

Every pipeline stage takes an explicit `Config` instance rather than reading module-level
globals, which keeps the pipeline reentrant (multiple configs can be evaluated in one process,
e.g. for parameter sweeps) and turns invalid settings into a `ValidationError` at load time
instead of a mid-run crash.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so a typo'd YAML field fails loudly."""

    model_config = ConfigDict(extra="forbid")


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class Equity(StrEnum):
    STOCKS = "stocks"

CandleLength = Literal["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo", "3mo"]


class UniverseConfig(_Strict):
    equity: Equity = Equity.STOCKS
    stock_number: int = Field(
        25,
        gt=0,
        le=10_500,
        description=(
            "Size of the ticker universe to trade: the N largest US companies by market cap, "
            "sourced from the SEC's own company_tickers.json (data/universe.py) -- about 10,400 "
            "tickers available as of writing. Each ticker is a separate Alpaca ingestion request "
            "(batched, but still rate-limited -- see data/ingest.py) plus its own indicator/signal "
            "computation, so both download time and compute cost scale roughly linearly with this "
            "number. Recommended: 5-20 for a quick smoke-test run (seconds to a couple minutes), "
            "50-100 for a realistic single backtest (several minutes), a few hundred to low "
            "thousands for a serious run -- ingestion is strictly sequential and a large "
            "multi-year hourly backfill is genuinely slow: measured at ~20s per 10,000-row page on "
            "a 9-year hourly range, a 10,000-ticker universe would take on the order of days, not "
            "the 'about an hour' a naive 200-requests/minute calculation would suggest. This is a "
            "one-time cost either way, since refresh_data=False reuses whatever was already "
            "downloaded on later runs -- but budget accordingly, and prefer a much smaller "
            "stock_number (or a coarser candle_length, which needs far fewer bars/pages) unless "
            "you actually need thousands of tickers. The le=10_500 cap here is just a sanity check "
            "against an obvious typo (e.g. an extra zero), not a recommendation -- requesting more "
            "than the ~10,400 actually available silently returns however many the source has."
        ),
    )
    candle_length: CandleLength = Field(
        "1h",
        description=(
            "Alpaca's free-plan historical bars are still only available for "
            "roughly the trailing ~6 years from today (rolling, undocumented by Alpaca, "
            "confirmed empirically), regardless of bar size. A start_date further back than that "
            "doesn't error -- Alpaca silently returns however much it actually has."
        ),
    )
    start_date: date = Field(date(2021, 1, 1))
    train_test_split_date: date = Field(
        date(2025, 1, 1),
        description=(
            "Boundary between the train and test sub-periods reported alongside the overall "
            "backtest: every report additionally breaks out return/volatility/Sharpe/trade "
            "stats separately for [start_date, train_test_split_date) and "
            "[train_test_split_date, end_date]. This exists because tuning any parameter "
            "(e.g. a correlation threshold) by looking only at combined-period performance risks "
            "picking a value that overfits -- train and test performance can "
            " move in opposite directions as a parameter is tuned. Must be "
            "strictly between start_date and end_date."
        ),
    )
    end_date: date = Field(date(2025, 12, 31))
    refresh_data: bool = Field(
        False,
        description="Re-download raw OHLCV before running. False reuses already-downloaded data.",
    )

    @model_validator(mode="after")
    def _check_date_order(self) -> UniverseConfig:
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be after start_date ({self.start_date})"
            )
        if not (self.start_date < self.train_test_split_date < self.end_date):
            raise ValueError(
                f"train_test_split_date ({self.train_test_split_date}) must be strictly between "
                f"start_date ({self.start_date}) and end_date ({self.end_date})"
            )
        return self


class IndicatorConfig(_Strict):
    rsi_length: int = Field(12, gt=1, description="Lookback window (bars) for RSI.")
    stochastic_fastk_period: int = Field(
        5, gt=0, description="Lookback window (bars) for the raw (%K) Stochastic oscillator."
    )
    stochastic_slowk_period: int = Field(
        3, gt=0, description="Smoothing window (bars) applied to raw %K to produce slow %K."
    )
    stochastic_slowd_period: int = Field(
        3, gt=0, description="Smoothing window (bars) applied to slow %K to produce %D."
    )

    @property
    def warmup_bars(self) -> int:
        """Longest lookback any single indicator needs before it starts producing values."""
        return max(
            self.rsi_length,
            self.stochastic_fastk_period,
            self.stochastic_slowk_period,
            self.stochastic_slowd_period,
        )


class ExtremaWindow(_Strict):
    """Bars required strictly before/after a candidate bar to confirm it as a swing point.

    A swing point using this window is only knowable `after` bars later than the candidate
    bar -- signals/extrema.py re-indexes confirmations forward by `after` bars specifically to
    avoid look-ahead bias, so treat `after` as a real reporting delay, not just a shape parameter.
    """

    before: int = Field(..., gt=0)
    after: int = Field(..., gt=0)


class ExtremaConfig(_Strict):
    divergence_first: ExtremaWindow = Field(
        default_factory=lambda: ExtremaWindow(before=5, after=5),
        description=(
            "Confirmation window for a divergence's older 'anchor' extremum. Stricter/slower "
            "than divergence_second is fine here since the anchor is already in the past by "
            "the time the pair matters, so its confirmation delay doesn't hold up the signal."
        ),
    )
    divergence_second: ExtremaWindow = Field(
        default_factory=lambda: ExtremaWindow(before=5, after=1),
        description=(
            "Confirmation window for a divergence's newer 'trigger' extremum -- keep `after` "
            "small here, since it dominates how quickly the overall signal can fire."
        ),
    )
    support_resistance: ExtremaWindow = Field(
        default_factory=lambda: ExtremaWindow(before=8, after=8),
        description=(
            "Confirmation window for the swing highs/lows that seed support/resistance levels."
        ),
    )


class DivergenceConfig(_Strict):
    expiration_bars: int = Field(
        30,
        gt=0,
        description=(
            "Bars allowed to elapse between the anchor and trigger extrema before the pair is "
            "considered stale and discarded. No closed value set: larger values allow "
            "slower-forming divergences at the cost of pairing extrema that are less related to "
            "each other."
        ),
    )


class RelativeThreshold(_Strict):
    """A threshold that's either a fixed absolute level (`flexible=False`, `value` used as-is) or
    a required move from some reference point captured earlier -- entry value, divergence-bar
    value, etc; whichever field holds this explains what that reference point is (`flexible=True`,
    `value` used as a delta). Bundling `flexible`+`value` into one field, rather than two separate
    ones, means a grid search can sweep both together as a single axis -- e.g.
    `grid: {...: [{flexible: false, value: 50.0}, {flexible: true, value: 10.0}]}` is 2
    combinations, not the 4 you'd get sweeping two independent scalar fields' cartesian product."""

    flexible: bool
    value: float


class StochasticCrossoverConfig(_Strict):
    """Tuning for the standalone `"stochastic_crossover"` buy signal (see
    `strategy/buy_rules.py`) -- fires when %K crosses %D after having entered an extreme
    (oversold/overbought) zone. Independent of RSI divergence; a strategy chains it together with
    `"rsi_divergence"` via `buy_signal_order` if it wants confirmation-style behavior, or uses
    either signal on its own."""

    extreme_threshold: float = Field(
        50.0,
        ge=0,
        le=100,
        description=(
            "The crossover watch arms once %K reaches this level (< threshold for long, "
            "> 100-threshold for short) -- guards against arming on an already-overbought/"
            "oversold reading that never actually entered the zone."
        ),
    )
    cross_level: RelativeThreshold = Field(
        default_factory=lambda: RelativeThreshold(flexible=True, value=10.0),
        description=(
            "cross_level.flexible=True (default): cross_level.value is the required %K move "
            "from the value %K had when the watch armed. cross_level.flexible=False: "
            "cross_level.value is a fixed absolute %K level regardless of where %K started."
        ),
    )
    expiration_bars: int = Field(
        10,
        gt=0,
        description="Bars the pending crossover watch stays armed before expiring unfired.",
    )


class SellIndicatorType(StrEnum):
    RSI = "rsi"
    STOCH_K = "k"
    STOCH_D = "d"


class SellIndicatorRule(_Strict):
    """Exits when the chosen indicator crosses a threshold. Evaluated fresh every bar -- no
    state persists once armed at entry other than the entry indicator value itself (needed for
    threshold.flexible=True)."""

    kind: Literal["indicator"] = "indicator"
    indicator: SellIndicatorType = Field(
        SellIndicatorType.RSI,
        description="Which indicator to compare against threshold: RSI, %K, or %D.",
    )
    threshold: RelativeThreshold = Field(
        default_factory=lambda: RelativeThreshold(flexible=False, value=50.0),
        description=(
            "threshold.flexible=False (default): threshold.value is a fixed absolute level "
            "regardless of the entry value. threshold.flexible=True: threshold.value is a "
            "required move from the value at entry."
        ),
    )


class SupportResistanceRule(_Strict):
    """Exits when price breaches a known support/resistance level (from
    signals/levels.py:support_resistance_levels). Levels are recomputed relative to the entry
    close the bar a position opens, then checked every subsequent bar."""

    kind: Literal["support_resistance"] = "support_resistance"
    resistance_min_distance: float = Field(
        1.0,
        ge=1.0,
        description=(
            "The nearest known level above entry_close * this multiplier is used as the "
            "resistance target -- must be >= 1.0 since resistance is above the entry price. "
            "1.0 = the nearest level at or above entry price itself; e.g. 1.02 requires the "
            "level to be at least 2% above entry."
        ),
    )
    support_min_distance: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "The nearest known level below entry_close * this multiplier is used as the "
            "support target -- must be in [0, 1] since support is below the entry price. 1.0 = "
            "the nearest level at or below entry price itself; e.g. 0.98 requires the level to "
            "be at least 2% below entry."
        ),
    )
    resistance_triggers_on_high: bool = Field(
        True,
        description=(
            "If True, the resistance side triggers on an intrabar touch (the bar's high "
            "reaching the level, modeling a resting order). If False, it only triggers if the "
            "bar's close itself reaches the level. The support side always triggers on close, "
            "regardless of this setting -- there's no equivalent toggle for it."
        ),
    )


class MarginRule(_Strict):
    """A take-profit/stop-loss band scaled by the ticker's own historical volatility at entry,
    as a multiple of price*volatility -- see docs/CONFIGURATION.md#margin-rule for worked
    examples. A fixed percentage band doesn't make sense across tickers of very different
    volatility -- a 1% band is tight for a volatile stock and needlessly loose for a calm one --
    so the band scales with the ticker's own realized volatility instead of being a flat constant."""

    kind: Literal["margin"] = "margin"
    take_profit_multiplier: float = Field(
        2.0,
        gt=0,
        description=(
            "Take-profit level = entry_close * (1 + volatility_at_entry * this) [long] or "
            "entry_close / (1 + volatility_at_entry * this) [short], where volatility_at_entry "
            "is the rolling volatility (see volatility_length below) as of the entry bar. "
            "Higher = a more distant, less frequently hit target."
        ),
    )
    stop_loss_multiplier: float = Field(
        2.0,
        gt=0,
        description=(
            "Stop-loss level = entry_close * (1 - volatility_at_entry * this) [long] or "
            "entry_close / (1 - volatility_at_entry * this) [short]. Higher = a wider stop, "
            "less frequently hit. Stop-loss always triggers on the bar's close, not intrabar -- "
            "there's no equivalent of take_profit_triggers_on_high for it."
        ),
    )
    volatility_length: int = Field(
        30,
        gt=1,
        description=(
            "Lookback window (bars) for the rolling volatility used to scale take_profit/"
            "stop_loss -- independent of indicators.rolling_volatility_length, so this rule can "
            "react faster or slower to changing volatility than the rest of the pipeline. If a "
            "position opens before this many bars of history exist, volatility_at_entry is NaN "
            "and the rule never fires for that position (falls back to whatever other sell "
            "rules are configured)."
        ),
    )
    take_profit_triggers_on_high: bool = Field(
        True,
        description=(
            "If True, take-profit triggers on an intrabar touch (the bar's high reaching the "
            "level, modeling a resting order). If False, it only triggers if the bar's close "
            "itself reaches the level."
        ),
    )


class TimeRule(_Strict):
    kind: Literal["time"] = "time"
    max_bars_held: int = Field(
        60, gt=0, description="Exit unconditionally once a position has been held this many bars."
    )


SellRule = Annotated[
    SellIndicatorRule | SupportResistanceRule | MarginRule | TimeRule,
    Field(discriminator="kind"),
]


def _default_sell_rule_groups() -> list[list[SellRule]]:
    return [
        [SellIndicatorRule()],
        [MarginRule()],
        [SupportResistanceRule()],
        [TimeRule()],
    ]


# The buy-signal names strategy/buy_rules.py knows how to look up (see its _SIGNAL_COLUMN
# mapping). "rsi_divergence" and "stochastic_crossover" are independent signals: chain them
# together in buy_signal_order for divergence-confirmed-by-crossover behavior, or use either
# alone.
BuySignalName = Literal["rsi_divergence", "stochastic_crossover", "candle sticks"]


def _default_buy_signal_order() -> list[list[BuySignalName]]:
    return [["rsi_divergence"]]


class StrategyConfig(_Strict):
    buy_signal_order: list[list[BuySignalName]] = Field(
        default_factory=_default_buy_signal_order,
        description=(
            "Outer list = chronological stages, all of which must fire in order; inner list = "
            "signal names tied for that stage (any order among them satisfies it). Valid names: "
            "'rsi_divergence', 'stochastic_crossover', 'candle sticks'."
        ),
    )
    buy_signal_expiration_bars: list[int] = Field(
        default_factory=list,
        description=(
            "One entry per GAP between consecutive signal names in buy_signal_order's flattened "
            "order (stages concatenated in order, tied names within a stage in the order listed) "
            "-- i.e. exactly one fewer entry than the total signal-name count across every stage, "
            "regardless of how those names are grouped into stages. Entry i bounds how many bars "
            "back flattened signal i is allowed to search for its own occurrence, relative to "
            "flattened signal i+1's already-found position, before the chain is considered "
            "broken. The very last flattened signal needs no entry: it's always the bar being "
            "evaluated (buy_rules.py only considers bars where it already fired), so there's "
            "nothing left to bound it against. A single-signal buy_signal_order (e.g. "
            "[['rsi_divergence']], the default) needs zero entries for the same reason -- there's "
            "no second signal to measure a gap to."
        ),
    )
    sell_rule_groups: list[list[SellRule]] = Field(
        default_factory=_default_sell_rule_groups,
        description=(
            "A sell fires when ALL rules in any one group are satisfied "
            "(OR across groups, AND within a group)."
        ),
    )

    @model_validator(mode="after")
    def _check_expiration_length(self) -> StrategyConfig:
        total_signals = sum(len(stage) for stage in self.buy_signal_order)
        expected = max(total_signals - 1, 0)
        if len(self.buy_signal_expiration_bars) != expected:
            raise ValueError(
                "buy_signal_expiration_bars must have exactly one fewer entry than the total "
                "number of signal names across all stages of buy_signal_order "
                f"({total_signals} signal(s) -> {expected} entries expected), got "
                f"{len(self.buy_signal_expiration_bars)}"
            )
        return self


class PositionSizingConfig(_Strict):
    """See docs/CONFIGURATION.md#position-sizing for the full trade-size formula and worked
    examples."""

    max_positions: int = Field(
        4,
        gt=0,
        description=(
            "Portfolio holds at most this many concurrent positions (a hard cap). Also sets "
            "each trade's base size: trade_size = cash / (max_positions - currently_open), an "
            "even split of remaining cash across remaining open slots -- self-limiting by "
            "construction (can never exceed available cash, unlike a prior estimated_value-based "
            "formula that could silently overspend). reject_correlated_ should generally be used"
            "to limit bet sizing instead of max_positions since it hedges against asset correlation"
        ),
    )
    reject_correlated_entries: bool = Field(
        True,
        description=(
            "If True, a candidate ticker's trade is refused entirely (not just sized down) if its "
            "trailing returns (over correlation_lookback_bars) are correlated at or above "
            "correlation_reject_threshold with ANY single currently-held ticker -- checks the "
            "single worst pairing, not a book average, since one dangerously correlated pair is "
            "exactly the risk this exists to catch. Uses signed correlation: a candidate strongly "
            "negatively correlated with a holding is a natural hedge, never rejected on that "
            "basis. Directly targets the failure mode where a cluster of correlated positions "
            "(e.g. a broad market dip triggering many mean-reversion entries at once) gets treated "
            "as independent bets when it isn't. Prior size-dampening versions of this, and a "
            "separate inverse-volatility size dampener, were both tried and removed due to poor"
            "performance"
        ),
    )
    correlation_reject_threshold: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Minimum (signed) correlation with a held ticker's trailing returns that refuses a "
            "new trade outright, when reject_correlated_entries is enabled. Higher = more "
            "permissive (only near-duplicate bets get blocked); lower = stricter (blocks more "
            "moderately-correlated pairs too)."
        ),
    )
    correlation_lookback_bars: int = Field(
        60,
        gt=1,
        description=(
            "Trailing window, in bars (not calendar time), of returns used to estimate "
            "correlation between a candidate ticker and each currently-held ticker, when "
            "reject_correlated_entries is enabled. Bar count rather than calendar days "
            "deliberately: every other lookback in this schema is bar-based, and a correlation "
            "estimate's reliability is driven by the number of observations, not calendar "
            "duration -- fixing calendar days would mean too few observations for a stable "
            "estimate on slow candles (e.g. ~13 weekly bars in 90 days) and an unnecessarily "
            "expensive number on fast ones (millions of 1-second bars), where the correlation "
            "measured would reflect microstructure noise rather than the 'same bet' risk this "
            "gate exists to catch."
        ),
    )

class RiskConfig(_Strict):
    portfolio_warmup_bars: int = Field(
        15,
        ge=0,
        description=(
            "Bars at the start of the backtest excluded from every reported performance metric "
            "(return, volatility, Sharpe, trade stats, win/loss counts, train/test sub-periods) -- "
            "the portfolio starts entirely in cash and takes time to ramp up to its steady-state "
            "exposure, and that ramp is a flat/near-zero-return stretch that dilutes headline "
            "numbers if left in. Trading itself is unaffected -- signals can fire and trades can "
            "open during this window, they just aren't counted in the reported metrics until it "
            "ends. Set to 0 to disable and report from the true start_date. Does not affect the "
            "benchmark, which is meant to reflect true buy-and-hold performance over the full "
            "requested date range."
        ),
    )
    risk_free_rate: float = Field(
        0.04,
        description=(
            "Annualized risk-free rate. Used in the Sharpe ratio numerator, and (compounded over "
            "holding_period_hours) as one of the profit-odds simulation's thresholds below."
        ),
    )
    simulate_profit_odds: bool = Field(
        True,
        description=(
            "If True, run the block-bootstrap Monte Carlo simulation below, reporting the "
            "empirical odds -- over a holding_period_hours-long simulated horizon -- of clearing "
            "three bars: any profit at all (> breakeven), the risk_free_rate compounded over "
            "that horizon, and (only when a benchmark is available) the benchmark's own realized "
            "annualized return compounded over that horizon."
        ),
    )
    monte_carlo_paths: int = Field(
        10_000,
        gt=0,
        description="Number of simulated price paths in the block-bootstrap Monte Carlo.",
    )
    monte_carlo_block_size: int = Field(
        10,
        gt=0,
        description=(
            "Block bootstrap resamples contiguous runs of this many real historical returns at "
            "a time (preserving short-range autocorrelation) rather than drawing i.i.d. samples."
        ),
    )
    holding_period_hours: int = Field(
        2016,
        gt=0,
        description=(
            "Length (in bars) of each simulated Monte Carlo path -- default is ~1 year on this "
            "project's hourly candles."
        ),
    )
    calculate_spy_correlation: bool = Field(
        True,
        description=(
            "If True, report the Pearson correlation between the strategy's bar-over-bar equity "
            "curve returns and SPY's over the same aligned dates -- how much of the strategy's "
            "movement tracks the broad market rather than its own independent edge. Requires the "
            "SPY benchmark to be available; silently omitted from the report otherwise."
        ),
    )


class Config(_Strict):
    data_dir: Path = Field(Path("./sample_data"))
    directions: frozenset[Direction] = Field(default_factory=lambda: frozenset({Direction.LONG}))
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    extrema: ExtremaConfig = Field(default_factory=ExtremaConfig)
    divergence: DivergenceConfig = Field(default_factory=DivergenceConfig)
    stochastic_crossover: StochasticCrossoverConfig = Field(
        default_factory=StochasticCrossoverConfig
    )
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    @model_validator(mode="after")
    def _check_directions(self) -> Config:
        if not self.directions:
            raise ValueError(
                "directions must contain at least one of Direction.LONG / Direction.SHORT"
            )
        return self