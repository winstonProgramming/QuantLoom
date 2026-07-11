from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantloom.config import Direction
from quantloom.config.loader import load_config
from quantloom.config.schema import Config, RiskConfig, StrategyConfig, UniverseConfig


def test_packaged_default_config_loads() -> None:
    config = load_config()
    assert config.directions == frozenset({Direction.LONG})
    assert config.universe.candle_length == "1h"
    assert config.indicators.warmup_bars == 12  # rsi_length dominates
    assert config.risk.portfolio_warmup_bars == 15


def test_local_override_is_deep_merged(tmp_path: Path) -> None:
    override = tmp_path / "local.yaml"
    override.write_text('data_dir: "/tmp/custom"\nindicators:\n  rsi_length: 21\n')

    config = load_config(override)

    assert config.data_dir == Path("/tmp/custom")
    assert config.indicators.rsi_length == 21
    # untouched sibling fields keep their packaged defaults
    assert config.indicators.stochastic_fastk_period == 5


def test_missing_override_path_is_ignored(tmp_path: Path) -> None:
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config.universe.stock_number == 25


def test_end_date_before_start_date_rejected() -> None:
    with pytest.raises(ValidationError):
        UniverseConfig(start_date="2025-01-01", end_date="2024-01-01")


def test_train_test_split_date_outside_range_rejected() -> None:
    with pytest.raises(ValidationError):
        UniverseConfig(
            start_date="2024-01-01", train_test_split_date="2025-01-01", end_date="2024-06-01"
        )


def test_train_test_split_date_within_range_accepted() -> None:
    universe = UniverseConfig(
        start_date="2024-01-01", train_test_split_date="2024-03-01", end_date="2024-06-01"
    )
    assert str(universe.train_test_split_date) == "2024-03-01"


def test_empty_directions_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(data_dir="./data", directions=frozenset(), universe=UniverseConfig(
            start_date="2024-01-01", train_test_split_date="2024-03-01", end_date="2024-06-01",
        ))


def test_buy_signal_expiration_length_mismatch_rejected() -> None:
    # two total signal names requires exactly one expiration entry (one fewer than the total
    # signal count), not two
    with pytest.raises(ValidationError):
        StrategyConfig(
            buy_signal_order=[["rsi_divergence"], ["candle sticks"]],
            buy_signal_expiration_bars=[8, 4],
        )


def test_buy_signal_expiration_length_matching_signal_count_minus_one_accepted() -> None:
    strategy = StrategyConfig(
        buy_signal_order=[["rsi_divergence"], ["candle sticks"]], buy_signal_expiration_bars=[8]
    )
    assert strategy.buy_signal_expiration_bars == [8]


def test_buy_signal_expiration_empty_list_accepted_for_a_single_signal() -> None:
    # a lone signal has no "next" signal to measure a gap against, so it needs zero entries
    strategy = StrategyConfig(
        buy_signal_order=[["rsi_divergence"]], buy_signal_expiration_bars=[]
    )
    assert strategy.buy_signal_expiration_bars == []


def test_buy_signal_expiration_counts_tied_names_within_a_stage_too() -> None:
    # a single tied stage with 2 signal names still needs exactly 1 entry (2 signals - 1), same
    # as if they were two separate untied stages -- ties don't change the total signal count
    strategy = StrategyConfig(
        buy_signal_order=[["rsi_divergence", "candle sticks"]], buy_signal_expiration_bars=[5]
    )
    assert strategy.buy_signal_expiration_bars == [5]


def test_sell_rule_discriminated_union_parses_each_kind() -> None:
    strategy = StrategyConfig(
        sell_rule_groups=[
            [{"kind": "indicator", "indicator": "k"}],
            [{"kind": "margin", "take_profit_multiplier": 3.0}],
            [{"kind": "support_resistance"}],
            [{"kind": "time", "max_bars_held": 40}],
        ]
    )
    kinds = [rule.kind for group in strategy.sell_rule_groups for rule in group]
    assert kinds == ["indicator", "margin", "support_resistance", "time"]


def test_negative_portfolio_warmup_bars_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(portfolio_warmup_bars=-1)


def test_zero_portfolio_warmup_bars_accepted() -> None:
    assert RiskConfig(portfolio_warmup_bars=0).portfolio_warmup_bars == 0


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        UniverseConfig(
            start_date="2024-01-01", end_date="2024-06-01", not_a_real_field=True,
        )


def test_load_config_resolves_a_named_strategy_reference(tmp_path: Path) -> None:
    override = tmp_path / "local.yaml"
    override.write_text(
        "strategies:\n"
        "  my_test_strategy:\n"
        '    buy_signal_order: [["candle sticks"]]\n'
        "    buy_signal_expiration_bars: []\n"
        "strategy: my_test_strategy\n"
    )

    config = load_config(override)

    assert config.strategy.buy_signal_order == [["candle sticks"]]
    assert config.strategy.buy_signal_expiration_bars == []


def test_load_config_raises_a_clear_error_for_an_unknown_strategy_name(tmp_path: Path) -> None:
    override = tmp_path / "local.yaml"
    override.write_text("strategy: does_not_exist\n")

    with pytest.raises(ValueError, match="not a name defined under strategies"):
        load_config(override)


def test_load_config_validates_every_defined_strategy_eagerly(tmp_path: Path) -> None:
    # broken_strategy is invalid (buy_signal_expiration_bars must have exactly one fewer entry
    # than the total signal count in buy_signal_order) but never referenced by `strategy:` --
    # still must fail at load time.
    override = tmp_path / "local.yaml"
    override.write_text(
        "strategies:\n"
        "  broken_strategy:\n"
        '    buy_signal_order: [["rsi_divergence"], ["candle sticks"]]\n'
        "    buy_signal_expiration_bars: [8, 4]\n"
        "strategy: fast_divergence\n"
    )

    with pytest.raises(ValueError, match="broken_strategy is not a valid strategy definition"):
        load_config(override)


def test_load_config_still_accepts_an_inline_strategy_block(tmp_path: Path) -> None:
    override = tmp_path / "local.yaml"
    override.write_text('strategy:\n  buy_signal_order: [["candle sticks"]]\n')

    config = load_config(override)

    assert config.strategy.buy_signal_order == [["candle sticks"]]


_DEFAULT_STRATEGY_YAML = (
    "default_strategy:\n"
    '  buy_signal_order: [["rsi_divergence"]]\n'
    "  buy_signal_expiration_bars: []\n"
    "  sell_rule_groups:\n"
    "    - - kind: indicator\n"
    "        indicator: rsi\n"
    "        threshold: { flexible: false, value: 50.0 }\n"
    "    - - kind: margin\n"
    "        take_profit_multiplier: 2.0\n"
    "        stop_loss_multiplier: 3.0\n"
    "        volatility_length: 30\n"
    "        take_profit_triggers_on_high: true\n"
    "    - - kind: time\n"
    "        max_bars_held: 60\n"
)


def test_load_config_named_strategy_inherits_omitted_rule_fields_from_default(
    tmp_path: Path,
) -> None:
    # the margin rule below overrides stop_loss_multiplier but never mentions
    # take_profit_multiplier -- that field must fall back to default_strategy's own margin rule.
    override = tmp_path / "local.yaml"
    override.write_text(
        _DEFAULT_STRATEGY_YAML + "strategies:\n"
        "  tight_stop:\n"
        "    sell_rule_groups:\n"
        "      - - kind: margin\n"
        "          stop_loss_multiplier: 5.0\n"
        "strategy: tight_stop\n"
    )

    config = load_config(override)

    margin_rule = next(
        rule
        for group in config.strategy.sell_rule_groups
        for rule in group
        if rule.kind == "margin"
    )
    assert margin_rule.stop_loss_multiplier == 5.0  # explicitly overridden
    assert margin_rule.take_profit_multiplier == 2.0  # inherited from default_strategy


def test_load_config_named_strategy_controls_which_sell_rule_kinds_are_present(
    tmp_path: Path,
) -> None:
    # default_strategy has 3 groups (indicator, margin, time); a named strategy can drop down to
    # just one kind -- merge-by-kind never re-adds kinds the strategy didn't itself list.
    override = tmp_path / "local.yaml"
    override.write_text(
        _DEFAULT_STRATEGY_YAML + "strategies:\n"
        "  time_only:\n"
        "    sell_rule_groups:\n"
        "      - - kind: time\n"
        "strategy: time_only\n"
    )

    config = load_config(override)

    kinds = [rule.kind for group in config.strategy.sell_rule_groups for rule in group]
    assert kinds == ["time"]
    time_rule = config.strategy.sell_rule_groups[0][0]
    assert time_rule.max_bars_held == 60  # inherited from default_strategy


def test_load_config_named_strategy_inherits_omitted_top_level_fields(tmp_path: Path) -> None:
    override = tmp_path / "local.yaml"
    override.write_text(
        _DEFAULT_STRATEGY_YAML + "strategies:\n"
        "  candlestick_only:\n"
        '      buy_signal_order: [["candle sticks"]]\n'
        "strategy: candlestick_only\n"
    )

    config = load_config(override)

    assert config.strategy.buy_signal_order == [["candle sticks"]]
    assert config.strategy.buy_signal_expiration_bars == []  # inherited, unchanged


def test_load_config_rejects_an_invalid_default_strategy(tmp_path: Path) -> None:
    override = tmp_path / "local.yaml"
    override.write_text(
        "default_strategy:\n"
        '  buy_signal_order: [["rsi_divergence"], ["candle sticks"]]\n'
        "  buy_signal_expiration_bars: [8, 4]\n"  # length mismatch (should be 1 entry, not 2)
    )

    with pytest.raises(ValueError, match="default_strategy is not a valid strategy definition"):
        load_config(override)