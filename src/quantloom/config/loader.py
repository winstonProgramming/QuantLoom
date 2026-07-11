"""YAML loading for `Config`: packaged defaults, deep-merged with optional local overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import Config, StrategyConfig

_PACKAGE_DEFAULT = Path(__file__).parent / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_merged_dict(*overrides: Path) -> dict[str, Any]:
    """Load the packaged default config, then deep-merge override YAML files in order.

    Missing override paths are silently skipped so a repo without `configs/local.yaml` still runs.
    """
    with _PACKAGE_DEFAULT.open() as f:
        merged: dict[str, Any] = yaml.safe_load(f) or {}

    for override_path in overrides:
        if not override_path.exists():
            continue
        with override_path.open() as f:
            override_data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, override_data)

    return merged


def _merge_rule(
    default_rule: dict[str, Any] | None, override_rule: dict[str, Any]
) -> dict[str, Any]:
    """Deep-merges `override_rule`'s fields onto the default strategy's rule of the SAME `kind`,
    if one exists -- a field the override rule doesn't specify falls back to that default rule's
    value for the same field. A rule whose kind has no match in the default (or when there's no
    default strategy at all) is used exactly as given, since there's nothing to inherit fields
    from."""
    if default_rule is None or default_rule.get("kind") != override_rule.get("kind"):
        return override_rule
    return _deep_merge(default_rule, override_rule)


def _merge_sell_rule_groups(
    default_groups: list[list[dict[str, Any]]], override_groups: list[list[dict[str, Any]]]
) -> list[list[dict[str, Any]]]:
    """Matches override rules to the default strategy's rules by KIND, not position -- a named
    strategy is free to pick an entirely different subset/combination of sell-rule kinds (and
    group them differently) than the default, but a rule it does include still inherits any field
    it doesn't itself specify from the default's rule of that same kind, if the default has one."""
    default_by_kind = {
        rule["kind"]: rule
        for group in default_groups
        for rule in group
        if isinstance(rule, dict) and "kind" in rule
    }
    return [
        [_merge_rule(default_by_kind.get(rule.get("kind")), rule) for rule in group]
        for group in override_groups
    ]


def _merge_strategy_onto_default(
    default_strategy: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Merges a named strategy's definition onto `default_strategy` -- most fields
    (`buy_signal_order`, `buy_signal_expiration_bars`) use the override's value if given, else the
    default's, completely unchanged. `sell_rule_groups` merges by rule KIND
    instead (see `_merge_sell_rule_groups`), since a named strategy often selects a different
    combination of sell-rule kinds than the default but still wants to inherit unset field values
    within whichever kinds it does include -- e.g. a strategy that includes a `margin` rule but
    doesn't specify `take_profit_multiplier` inherits that one field from the default strategy's
    own `margin` rule, without having to restate the rest of that rule or any other group."""
    merged = dict(default_strategy)
    for key, value in override.items():
        if key == "sell_rule_groups" and isinstance(default_strategy.get("sell_rule_groups"), list):
            merged[key] = _merge_sell_rule_groups(default_strategy["sell_rule_groups"], value)
        else:
            merged[key] = value
    return merged


def _resolve_named_strategy(data: dict[str, Any]) -> dict[str, Any]:
    """If `data["strategy"]` is a plain string, replaces it with the matching named preset from
    the top-level `strategies:` section (see docs/CONFIGURATION.md#custom-strategies) -- lets a
    config reference a reusable, named strategy definition instead of always inlining the full
    `strategy:` block. Always strips `default_strategy` and `strategies` from the returned dict
    either way -- neither is a `Config` field, both are purely config-loading-time inputs.

    A top-level `default_strategy:` section, if present, is itself a full strategy definition
    that every entry under `strategies:` is deep-merged onto (see `_merge_strategy_onto_default`)
    before validation -- a named preset only needs to specify what's DIFFERENT from the default,
    not restate every field. Without a `default_strategy:` section, `strategies:` entries are used
    exactly as given (each must be fully self-contained).

    Every entry under `strategies:` (and `default_strategy` itself) is validated as a
    `StrategyConfig` unconditionally (not just the one currently referenced), so a typo in a
    preset you aren't using right now still fails loudly at load time instead of lurking until
    you finally grid-sweep to it.

    Called *after* grid expansion (see `expand_grid`) rather than before, so
    `grid: {strategy: [name1, name2]}` can sweep between named presets -- each expanded
    combination's `strategy` value (a plain string at that point) is resolved independently here.
    """
    data = dict(data)
    default_strategy = data.pop("default_strategy", None)
    strategies = data.pop("strategies", None) or {}

    if default_strategy is not None:
        try:
            StrategyConfig.model_validate(default_strategy)
        except ValidationError as exc:
            raise ValueError(
                f"default_strategy is not a valid strategy definition: {exc}"
            ) from exc

    resolved_strategies: dict[str, Any] = {}
    for name, definition in strategies.items():
        merged = (
            _merge_strategy_onto_default(default_strategy, definition)
            if default_strategy is not None
            else definition
        )
        try:
            StrategyConfig.model_validate(merged)
        except ValidationError as exc:
            raise ValueError(
                f"strategies.{name} is not a valid strategy definition: {exc}"
            ) from exc
        resolved_strategies[name] = merged

    strategy = data.get("strategy")
    if isinstance(strategy, str):
        if strategy not in resolved_strategies:
            available = ", ".join(sorted(resolved_strategies)) or "(none defined)"
            raise ValueError(
                f"strategy: {strategy!r} is not a name defined under strategies: -- available: "
                f"{available}"
            )
        data["strategy"] = resolved_strategies[strategy]

    return data


def load_config(*overrides: Path) -> Config:
    """Load the packaged default config, then deep-merge override YAML files in order.

    Typical use: `load_config(Path("configs/local.yaml"))` to layer machine-specific settings
    (e.g. `data_dir`) on top of the checked-in defaults without editing tracked files.

    Expects every field to be a single, concrete value. A top-level `grid:` section (see
    config/grid.py) is ignored here if present -- it's only interpreted by `load_config_grid`,
    which sweeps every axis it lists into its own `Config`; use that instead to actually run a
    sweep. `strategy` may be a plain string naming a preset under `strategies:` (see
    `_resolve_named_strategy`) instead of an inline block. Any other unrecognized field still
    raises a normal pydantic `ValidationError`.
    """
    merged = dict(_load_merged_dict(*overrides))
    merged.pop("grid", None)
    merged = _resolve_named_strategy(merged)
    return Config.model_validate(merged)