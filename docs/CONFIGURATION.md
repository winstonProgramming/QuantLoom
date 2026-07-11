# Configuration Reference

`src/quantloom/config/schema.py` is the source of truth: every field is a validated
pydantic model, and every field carries a comprehensive `description` you can read directly
(e.g. via `Config.model_fields` or by opening the file) -- mechanics, formulas, and defaults are
all there, not just a one-line hint. This document holds only what a field description
structurally can't: closed value sets not obvious from a type alone, worked numeric examples/
tables, and cross-field or cross-file mechanics that don't belong to any single field. If a field's
description already answers your question, there's deliberately no matching section here.

## How configuration loads

Two YAML files are merged: the packaged defaults
(`src/quantloom/config/default.yaml`) plus an optional local override
(`configs/local.yaml`, gitignored -- copy `configs/local.example.yaml` to create it). The local
file only needs to contain the keys you want to change; everything else falls back to the
packaged default. Invalid values (wrong type, out of range, unknown key) fail immediately at
load time with a pydantic `ValidationError`.

## Grid search

A top-level `grid:` section maps dotted config paths to a list of candidate values to sweep, and
`python -m quantloom.main` will run every combination (Cartesian product across all listed axes)
instead of just one. For example:

```yaml
indicators:
  rsi_length: 12          # baseline value -- used as-is for any field not listed under grid:
divergence:
  expiration_bars: 30

grid:
  indicators.rsi_length: [10, 12, 14]
  divergence.expiration_bars: [20, 30]
```

runs all 3 x 2 = 6 combinations. `grid:` is its own namespace, separate from the fields it
overrides -- this is a deliberate choice to avoid ambiguity for any field whose own type is
already list-shaped (e.g. `strategy.buy_signal_order: list[list[str]]`).

```yaml
grid:
  strategy.buy_signal_order:
    - [["rsi_divergence"]]
    - [["rsi_divergence"], ["candle sticks"]]
```

A path segment that's purely digits is treated as a list index when the current node is a
list, so a single rule buried inside a list-of-lists field can be targeted directly instead of
having to restate the entire structure per candidate -- e.g. to sweep just the first sell rule
group's first rule:

```yaml
grid:
  strategy.sell_rule_groups.0.0.threshold:
    - { flexible: false, value: 50.0 }
    - { flexible: true, value: 10.0 }
```

That example also shows why `flexible`+`value` are bundled into one `RelativeThreshold` field
(`SellIndicatorRule.threshold`, `StochasticCrossoverConfig.cross_level`) instead of two separate
scalar fields -- sweeping the two independently would Cartesian-product into meaningless combinations;
bundling them means one grid axis, so listing exactly the pairs you care about gives exactly that many
combinations.

A key missing its section prefix (e.g. bare `rsi_length` instead of `indicators.rsi_length`) is
auto-resolved against the schema rather than silently becoming a bogus top-level field -- so
`grid: {rsi_length: [10, 12, 14]}` works the same as the fully-qualified form above. This only
works for a name that's unique across the whole schema; `expiration_bars` alone is rejected with
a clear "ambiguous" error (it exists on both `divergence` and `stochastic_crossover`) rather than
silently guessing, and a name that matches nothing anywhere raises immediately too -- both fail
at grid-parse time, before any combination is built. This resolution only applies to plain named
fields; an indexed path (containing a list index, like `strategy.sell_rule_groups.0.0.threshold`
above) is left untouched either way.

Every run opens an HTML comparison report (see `reporting/grid_report.py`). Each combination in
a real sweep runs in its own process (`main.py`'s `_run_grid_search`, via `ProcessPoolExecutor`)
since the indicator/signal/backtest computation is CPU-bound.

Note that a grid search is meant for sweeping strategy/signal/sizing parameters, not the ticker
universe or date range, per combination.

Indicators/signals/strategy columns themselves are always computed in memory and never written
back to the on-disk store for performance reasons -- the on-disk Parquet store holds only raw
OHLCV. A `--no-refresh` run reuses the ingested OHLCV, not any previously computed
indicator/signal/strategy columns -- those are recomputed fresh every run.

`load_config` (the plain, non-grid loader) ignores a `grid:` section if present, rather than
erroring on the unrecognized key -- it just uses whatever literal/baseline values are given
elsewhere. Use `load_config_grid`/`GridPoint` in `config/grid.py` to actually expand a sweep, for
programmatic use (e.g. from a notebook or a custom script) instead of the CLI.

## Train/test split

The train/test breakdown is read directly off the single continuous equity curve and closed-trade
list the full-period report already computes -- there is no separate re-simulation for the train
and test windows, so portfolio state (cash, open positions) stays continuous across the
`train_test_split_date` boundary (see `universe.train_test_split_date`'s description in schema.py
for what gets reported and why this split exists at all).

When using [grid search](#grid-search) to sweep a parameter, prefer the value that holds up on
the test period over the one that merely maximizes the combined or train period numbers -- train
and test performance can move in opposite directions as a parameter is tuned.

## Universe

| Field | Valid values                                                    |
|---|-----------------------------------------------------------------|
| `equity` | `stocks` (the only supported market type as of now)             |
| `candle_length` | `1m`, `2m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`, `1mo`, `3mo` |

Sourcing the universe (`data/universe.py`) is free, no API key, no HTML scraping -- and (verified
empirically against the live file) the SEC's `company_tickers.json` is already ordered by market
cap descending, so truncating to `stock_number` is a meaningful "N biggest" (see `stock_number`'s
description in schema.py for the rest). Note that `stock_number` is capped at roughly 10,400.

Ingestion is via Alpaca's Market Data API (`data/ingest.py`), which requires the
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` environment variables -- a free Alpaca account's API keys
work, no funding or brokerage subscription needed. Ingestion requests the IEX feed explicitly (the
feed guaranteed on every free plan); historical (>15-minutes-old) requests may also be entitled to
the fuller SIP consolidated tape depending on your plan, but that isn't assumed by default since
requesting an unentitled feed errors. See `candle_length`'s description in schema.py for Alpaca's
undocumented rolling lookback limit.

Market hours: ingested bars are not limited to the regular 9:30am-4:00pm ET session --
a partial premarket window is included (the first bar of a trading day typically starts around
8:00am ET -- not the full 4:00am ET premarket session, just whatever earlier trading IEX itself
reports), but no after-hours data at all (the last bar ends exactly at the 4:00pm ET close).
`StockBarsRequest` has no extended-hours parameter to control this either way, and nothing
downstream -- indicators, signals, or the strategy engine -- filters by time of day, so every bar
IEX returns is treated identically regardless of session. See the README's
[Market hours coverage](../README.md#market-hours-coverage) section for the full detail; there is
currently no config option to filter premarket bars out.

Ingestion downloads one ticker at a time, strictly sequentially -- multithreading fails due to
server API request limitations (Alpaca's backend appears to have a concurrency-sensitive capacity
limit tighter than its documented 200 requests/minute figure), with failed chunks dropping tickers
once retries were exhausted.

## Indicators

Straightforward lookback-window parameters for RSI and the Stochastic oscillator -- see the
`description` on each field in `IndicatorConfig`. Larger windows smooth more and react to slower
trends, smaller windows react faster and noisier.

## Strategy

### Custom strategies

`strategy:` normally takes an inline block (`buy_signal_order`, `buy_signal_expiration_bars`,
`sell_rule_groups` -- see `StrategyConfig`'s fields in schema.py). It can instead be a plain
string naming a preset defined under a top-level `strategies:` section, so you can define
several reusable strategies once and reference (or sweep between) them by name:

```yaml
strategies:
  fast_divergence:
    buy_signal_order: [["rsi_divergence"]]
    buy_signal_expiration_bars: []
    sell_rule_groups:
      - - kind: indicator
          indicator: rsi
          threshold: { flexible: false, value: 50.0 }
      - - kind: time
          max_bars_held: 60
  candlestick_confirmed_divergence:
    buy_signal_order: [["rsi_divergence"], ["candle sticks"]]
    buy_signal_expiration_bars: [8]
    sell_rule_groups:
      - - kind: time
          max_bars_held: 120

strategy: fast_divergence
```

`strategies:` is not a `Config` field -- it's a config-loading-time lookup table, resolved
(`config/loader.py`'s `_resolve_named_strategy`) before `strategy:` is validated, then discarded.
Every entry under `strategies:` is validated unconditionally at load time, not just whichever one
`strategy:` currently references, so a typo in a preset you aren't using yet still fails loudly
immediately rather than lurking until you finally switch to (or grid-sweep to) it.

Because resolution happens by name, [grid search](#grid-search) can sweep between whole named
strategies as a single axis, same as any other field:

```yaml
grid:
  strategy: [fast_divergence, candlestick_confirmed_divergence]
```

This is the reason `strategies:` exists as a separate section rather than just letting
`buy_signal_order`/`sell_rule_groups`/etc. be swept individually inside `strategy:` -- you often
want to compare whole, internally-consistent strategy configurations against each other (a
particular buy chain paired with a particular exit scheme), not the Cartesian product of every
field varied independently, most of which wouldn't be meaningful pairings.

One limitation: a `grid:` path can't reach inside a strategy referenced by name (e.g.
`strategy.buy_signal_expiration_bars` while `strategy: fast_divergence`) -- resolution happens
after grid expansion, so at that point `strategy` is still the plain string `"fast_divergence"`,
not yet the dict it names. Either grid-sweep `strategy` itself (a list of preset names) or inline
the strategy block instead of naming it if you need to sweep one of its sub-fields.

#### `default_strategy:` -- presets as deltas, not full restatements

A top-level `default_strategy:` section, if present, is itself a full strategy definition that
every entry under `strategies:` is deep-merged onto (`config/loader.py`'s
`_merge_strategy_onto_default`) before validation. A preset only needs to state what's
different from `default_strategy` -- any field it omits falls back to `default_strategy`'s
value for that field:

```yaml
default_strategy:
  buy_signal_order: [["rsi_divergence"]]
  buy_signal_expiration_bars: []
  sell_rule_groups:
    - - kind: indicator
        indicator: rsi
        threshold: { flexible: false, value: 50.0 }
    - - kind: margin
        take_profit_multiplier: 2.0
        stop_loss_multiplier: 2.0
    - - kind: time
        max_bars_held: 60

strategies:
  tight_stop:
    sell_rule_groups:
      - - kind: margin
          stop_loss_multiplier: 5.0   # take_profit_multiplier still inherits 2.0
```

Most fields (`buy_signal_order`, `buy_signal_expiration_bars`) merge as a plain "override
replaces default if given, else keep the default" -- but `sell_rule_groups` is a
`list[list[dict]]`, not a plain dict, so it merges by rule `kind`, not by list position:

- A preset is free to include an entirely different subset or combination of sell-rule kinds than
  `default_strategy` -- `default_strategy` having an `indicator` rule doesn't force every preset
  to have one; a preset's `sell_rule_groups` lists exactly the kinds it wants, no more, no less.
- For a kind the preset does include, any field it doesn't specify on that rule falls back to
  `default_strategy`'s rule of the same kind, if one exists. In the example above, `tight_stop`'s
  `margin` rule only sets `stop_loss_multiplier`; `take_profit_multiplier` is inherited from
  `default_strategy`'s `margin` rule unchanged.
- If `default_strategy` has no rule of a kind the preset introduces, that rule is used exactly as
  given (nothing to inherit from).

Without a `default_strategy:` section, `strategies:` entries are used exactly as given (each must
be fully self-contained) -- the simpler behavior shown at the top of this section.

#### Shipped presets

`configs/default.yaml`'s packaged `strategies:` section ships several presets as deltas against
its `default_strategy:` -- read `default.yaml` directly for the current set and their exact
`buy_signal_order`/`sell_rule_groups`, since both are actively iterated on. The general shapes in
play:

- A raw-divergence entry with the full exit stack (indicator + margin + support/resistance +
  time) -- exits on whichever of four independent conditions triggers first.
- Divergence confirmed by a subsequent stochastic crossover, exiting only on an RSI level
  crossing or a hard time stop -- both halves of the entry are about not acting until momentum has
  actually turned (see [Stochastic Crossover](#stochastic-crossover) below).
- Divergence, then a candlestick pattern, exiting on margin/support-resistance/time -- don't
  act on the divergence alone, wait for an actual reversal candle within the next few bars.
- Divergence and candlestick in either order, same window -- rarer, higher-conviction setups
  than the strict "confirmation" ordering above.
- Divergence with a support/resistance-only exit ("buy near support, sell at resistance") --
  no volatility bracket at all, a range/chop trading style.
- Divergence, confirmed by a stochastic crossover, then a candlestick pattern too -- the most
  selective entry, and the exit requires agreement too: the RSI level and a support/resistance
  breach must fire together (grouped into one AND'd group, not two independent ones), with a
  volatility-scaled margin hit or a 60-bar time stop as the other two (still independent)
  ways out. See [Sell rules](#sell-rules) below for the AND-within-a-group/OR-across-groups
  mechanics this relies on.

### Buy signal chains

See `StrategyConfig.buy_signal_order`'s description in schema.py for the stage/tie-breaking
mechanics and the full list of valid signal names.

### Sell rules

`sell_rule_groups` is a list of groups; a sell fires when all rules within any one group
are satisfied (OR across groups, AND within a group). Four rule kinds (`kind` discriminator) --
see each rule's own fields in schema.py for its exact mechanics:

| `kind` | Purpose |
|---|---|
| `indicator` | Exit when RSI, %K, or %D (`indicator: rsi \| k \| d`) crosses a threshold |
| `margin` | Take-profit/stop-loss band, scaled by the ticker's own volatility at entry (see [Margin Rule](#margin-rule)) |
| `support_resistance` | Exit when price breaches a known support/resistance level |
| `time` | Exit unconditionally after `max_bars_held` bars |

### Margin Rule

Worked example of `MarginRule`'s volatility-scaled take-profit/stop-loss band (see its class
docstring and `take_profit_multiplier`/`stop_loss_multiplier`'s descriptions in schema.py for the
exact formula), `entry_close = 100`:

| Volatility at entry | `multiplier=1.0` | `multiplier=2.0` (default) | `multiplier=3.0` |
|---|---|---|---|
| 1% (calm) | ±1.00 (±1.00%) | ±2.00 (±2.00%) | ±3.00 (±3.00%) |
| 2% (typical) | ±2.00 (±2.00%) | ±4.00 (±4.00%) | ±6.00 (±6.00%) |
| 5% (volatile) | ±5.00 (±5.00%) | ±10.00 (±10.00%) | ±15.00 (±15.00%) |

Reading a row: at 5% realized volatility, the default `multiplier=2.0` puts take-profit/stop-loss
10% away from entry in both directions -- five times further than the same multiplier would place
them for a calm 1%-volatility ticker. `take_profit_multiplier` and `stop_loss_multiplier` don't
need to be equal; a higher take-profit multiplier than stop-loss multiplier gives an asymmetric
risk/reward band (a bigger target than the risk taken to reach it), and vice versa.

## Stochastic Crossover

`"stochastic_crossover"` is an ordinary buy signal (see [Buy signal chains](#buy-signal-chains)
above), not a global toggle -- a strategy opts in by referencing it in its own `buy_signal_order`,
typically chained after `"rsi_divergence"` for confirmation-style behavior (see
[Shipped presets](#shipped-presets) above). A strategy that doesn't reference
`"stochastic_crossover"` at all is entirely unaffected by `StochasticCrossoverConfig`'s tuning,
which is otherwise fully documented on its own fields in schema.py.

## Position Sizing

`max_positions` doubles as both the hard cap on concurrent positions and the basis for each new
trade's size: `trade_size = cash / (max_positions - currently_open)`, an even split of whatever
cash remains across whatever open slots remain, recomputed fresh at every entry rather than fixed
at portfolio-open time. This is self-limiting by construction -- a trade can never draw more than
the cash actually on hand; see `max_positions`'s description in schema.py for more information

Worked example, `cash = $10,000`, `max_positions = 4`:

| Currently open | Remaining slots | This trade's size |
|---|---|---|
| 0 | 4 | $2,500.00 |
| 1 | 3 | $3,333.33 |
| 2 | 2 | $5,000.00 |
| 3 | 1 | $10,000.00 (all remaining cash) |

Reading a row: trade size grows as slots fill because the same remaining cash is split across
fewer remaining slots, not because entries get more aggressive -- the last open slot always
claims whatever cash happens to be left. `max_positions` is a hard cap, not a sizing dial.

### Correlation-based rejection

See `reject_correlated_entries`/`correlation_reject_threshold`/`correlation_lookback_bars`'s
descriptions in schema.py for the full mechanism (signed correlation, single-worst-pairing check,
bar-based lookback).

Worked example, `correlation_reject_threshold = 0.35`, a candidate ticker's trailing returns
(over `correlation_lookback_bars`) against each currently-held ticker's:

| Held ticker | Correlation with candidate | Outcome |
|---|---|---|
| A | 0.82 | Trade refused -- this pairing alone exceeds the threshold |
| B | 0.10 | Would not refuse on its own |
| C | -0.60 | Would not refuse -- negative correlation is a hedge, never rejected |

Reading this: the candidate is refused because of ticker A alone, even though B sits well under
the threshold and C is negatively correlated -- the check looks at the single worst pairing in
the current book, not a book-wide average, since one dangerously correlated pair is exactly the
risk this exists to catch.

## Risk

### Portfolio warmup

`portfolio_warmup_bars`'s description in schema.py covers what's excluded from reported metrics
and why. One cross-section detail worth knowing: a trade still open when the warmup cutoff passes
has its mark-to-market value already baked into the reportable equity curve's starting point, the
same way a position open across the `train_test_split_date` boundary carries into the test period
(see [Train/test split](#train-test-split)).

The packaged default overrides `holding_period_hours` to `2016` (~1 year on this project's hourly
candles).
