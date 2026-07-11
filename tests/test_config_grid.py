from __future__ import annotations

import math
from pathlib import Path

import pytest

from quantloom.config.grid import (
    _known_schema_paths,
    _resolve_grid_path,
    _set_by_path,
    expand_grid,
    load_config_grid,
)
from quantloom.config.loader import _load_merged_dict


def test_set_by_path_replaces_nested_value_without_mutating_input() -> None:
    data = {"indicators": {"rsi_length": 10, "ema_length": 12}}

    result = _set_by_path(data, "indicators.rsi_length", 12)

    assert result["indicators"]["rsi_length"] == 12
    assert result["indicators"]["ema_length"] == 12
    assert data["indicators"]["rsi_length"] == 10  # original untouched


def test_expand_grid_produces_cartesian_product_of_all_axes() -> None:
    data = {
        "indicators": {"rsi_length": 12},
        "divergence": {"expiration_bars": 30},
        "grid": {
            "indicators.rsi_length": [10, 12, 14],
            "divergence.expiration_bars": [20, 30],
        },
    }

    combos = expand_grid(data)

    assert len(combos) == 6
    rsi_values = {overrides["indicators.rsi_length"] for overrides, _ in combos}
    expiration_values = {overrides["divergence.expiration_bars"] for overrides, _ in combos}
    assert rsi_values == {10, 12, 14}
    assert expiration_values == {20, 30}
    # every combination's expanded dict actually has the scalar substituted in place, and the
    # grid key itself is stripped out (not a valid Config field)
    for overrides, expanded in combos:
        assert "grid" not in expanded
        assert expanded["indicators"]["rsi_length"] == overrides["indicators.rsi_length"]
        assert expanded["divergence"]["expiration_bars"] == overrides["divergence.expiration_bars"]


def test_expand_grid_resolves_a_bare_field_name_missing_its_section_prefix() -> None:
    # rsi_length actually lives at indicators.rsi_length -- previously a bare key like this
    # silently became a bogus top-level field, only caught later as a cryptic pydantic
    # extra_forbidden error once every combination had already been built.
    data = {"indicators": {"rsi_length": 12}, "grid": {"rsi_length": [10, 14]}}

    combos = expand_grid(data)

    assert len(combos) == 2
    for overrides, expanded in combos:
        assert "rsi_length" not in overrides  # rewritten to the resolved path
        assert overrides["indicators.rsi_length"] == expanded["indicators"]["rsi_length"]


def test_expand_grid_rejects_an_ambiguous_bare_field_name() -> None:
    # expiration_bars exists on both divergence and stochastic_crossover -- must not silently
    # guess which one was meant.
    data = {
        "divergence": {"expiration_bars": 30},
        "stochastic_crossover": {"expiration_bars": 10},
        "grid": {"expiration_bars": [20, 30]},
    }

    with pytest.raises(ValueError, match="is ambiguous"):
        expand_grid(data)


def test_expand_grid_rejects_a_grid_path_matching_no_config_field() -> None:
    data = {"indicators": {"rsi_length": 12}, "grid": {"not_a_real_field": [1, 2]}}

    with pytest.raises(ValueError, match="doesn't match any Config field"):
        expand_grid(data)


def test_expand_grid_leaves_an_indexed_path_unresolved() -> None:
    # a numeric segment indexes into a list, which isn't enumerable from the schema alone --
    # resolution must leave it untouched rather than mistaking it for an unknown field.
    data = {
        "strategy": {"sell_rule_groups": [[{"kind": "time", "max_bars_held": 60}]]},
        "grid": {
            "strategy.sell_rule_groups.0.0.max_bars_held": [30, 60],
        },
    }

    combos = expand_grid(data)

    assert len(combos) == 2
    chosen = {overrides["strategy.sell_rule_groups.0.0.max_bars_held"] for overrides, _ in combos}
    assert chosen == {30, 60}


def test_expand_grid_sweeps_a_list_typed_field_without_ambiguity() -> None:
    # strategy.buy_signal_order's own type is list[list[str]] -- the old bare-list-at-its-own-path
    # convention could never sweep this (it was hardcoded into a literal-list denylist instead).
    # The grid section resolves the ambiguity: candidates live in their own namespace, so a
    # list-typed field's candidates (each itself a list) are unambiguous with its literal value.
    data = {
        "strategy": {"buy_signal_order": [["rsi_divergence"]]},
        "grid": {
            "strategy.buy_signal_order": [
                [["rsi_divergence"]],
                [["rsi_divergence"], ["candle sticks"]],
            ]
        },
    }

    combos = expand_grid(data)

    assert len(combos) == 2
    chosen_values = [overrides["strategy.buy_signal_order"] for overrides, _ in combos]
    assert [["rsi_divergence"]] in chosen_values
    assert [["rsi_divergence"], ["candle sticks"]] in chosen_values
    for overrides, expanded in combos:
        assert expanded["strategy"]["buy_signal_order"] == overrides["strategy.buy_signal_order"]


def test_expand_grid_returns_single_combo_when_no_grid_section_present() -> None:
    data = {"indicators": {"rsi_length": 12}, "directions": ["long"]}

    combos = expand_grid(data)

    assert combos == [({}, data)]


def test_expand_grid_returns_single_combo_when_grid_section_is_empty() -> None:
    data = {"indicators": {"rsi_length": 12}, "grid": {}}

    combos = expand_grid(data)

    assert combos == [({}, {"indicators": {"rsi_length": 12}})]


def test_expand_grid_rejects_non_list_axis() -> None:
    data = {"indicators": {"rsi_length": 12}, "grid": {"indicators.rsi_length": 14}}

    with pytest.raises(ValueError, match="non-empty YAML list"):
        expand_grid(data)


def test_expand_grid_rejects_empty_axis() -> None:
    data = {"indicators": {"rsi_length": 12}, "grid": {"indicators.rsi_length": []}}

    with pytest.raises(ValueError, match="non-empty YAML list"):
        expand_grid(data)


def test_load_config_grid_expands_the_packaged_default_yaml() -> None:
    # default.yaml ships a real grid: {...} section (multiple axes) -- this is an end-to-end
    # check that loader.py + grid.py + schema.py agree on that syntax. Reads the candidate lists
    # from the YAML itself rather than hardcoding them, since they're live values the project's
    # own default config actively sweeps and edits over time. Resolves each raw key the same way
    # expand_grid does, since the packaged config may use a bare `rsi_length` rather than the
    # fully-qualified `indicators.rsi_length` -- see _resolve_grid_path.
    merged = _load_merged_dict()
    known_paths = _known_schema_paths()
    resolved_grid = {
        _resolve_grid_path(path, known_paths): candidates
        for path, candidates in merged["grid"].items()
    }
    expected_rsi_lengths = set(resolved_grid["indicators.rsi_length"])
    expected_total = math.prod(len(candidates) for candidates in resolved_grid.values())

    points = load_config_grid()

    assert len(points) == expected_total
    assert {p.overrides["indicators.rsi_length"] for p in points} == expected_rsi_lengths
    assert {p.config.indicators.rsi_length for p in points} == expected_rsi_lengths


def test_load_config_grid_sweeps_named_strategies(tmp_path: Path) -> None:
    override = tmp_path / "local.yaml"
    override.write_text(
        "strategies:\n"
        "  strat_a:\n"
        "    sell_rule_groups:\n"
        "      - - kind: time\n"
        "          max_bars_held: 10\n"
        "  strat_b:\n"
        "    sell_rule_groups:\n"
        "      - - kind: time\n"
        "          max_bars_held: 20\n"
        "grid:\n"
        "  strategy: [strat_a, strat_b]\n"
    )

    points = load_config_grid(override)

    max_bars_by_name = {
        p.overrides["strategy"]: p.config.strategy.sell_rule_groups[0][0].max_bars_held
        for p in points
    }
    assert max_bars_by_name == {"strat_a": 10, "strat_b": 20}
    assert {p.overrides["strategy"] for p in points} == {"strat_a", "strat_b"}


def test_load_config_grid_raises_a_clear_error_sweeping_inside_an_unresolved_strategy(
    tmp_path: Path,
) -> None:
    # strategy is still a plain string ("fast_divergence") at grid-expansion time -- resolution
    # to the actual strategy dict happens after, so a sub-field grid path can't reach inside it.
    override = tmp_path / "local.yaml"
    override.write_text(
        'strategy: fast_divergence\ngrid:\n  strategy.buy_signal_order: [[["rsi_divergence"]]]\n'
    )

    with pytest.raises(ValueError, match="reach inside a plain string value"):
        load_config_grid(override)
