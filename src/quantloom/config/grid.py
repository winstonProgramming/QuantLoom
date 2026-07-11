"""Grid-search expansion over `default.yaml`/local-override config.

A top-level `grid:` section maps dotted config paths to a list of candidate values to sweep into
the Cartesian product of every listed axis -- its own namespace so any field, including ones
already list-shaped, can be swept without ambiguity against its literal value. A bare key missing
its section prefix is auto-resolved against the schema (see `_resolve_grid_path`). See
docs/CONFIGURATION.md#grid-search for the full mechanics and worked examples.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .loader import _load_merged_dict, _resolve_named_strategy
from .schema import Config


def _known_schema_paths(model: type[BaseModel] = Config, prefix: str = "") -> set[str]:
    """Every dotted path reachable by walking nested pydantic BaseModel fields from `model`,
    named-attribute access only -- does NOT enumerate into list/dict-typed fields (e.g. the
    individual rules inside `strategy.sell_rule_groups`), since those are only reachable
    positionally, at a depth the schema alone can't predict. Used by `_resolve_grid_path` to
    repair a grid key that's missing its section prefix."""
    paths: set[str] = set()
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        paths.add(path)
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths |= _known_schema_paths(annotation, prefix=f"{path}.")
    return paths


def _resolve_grid_path(path: str, known_paths: set[str]) -> str:
    """Resolves a grid key that's missing its section prefix (e.g. `rsi_length` instead of
    `indicators.rsi_length`) against the schema, so any bare or under-qualified field name works
    regardless of how deeply it's actually nested.

    A path already matching a known schema path, or indexing into a list (a numeric segment --
    not enumerable from the schema alone, e.g. `strategy.sell_rule_groups.0.0.threshold`), is
    returned unchanged. A path matching more than one field anywhere in the schema (e.g. bare
    `expiration_bars`, which exists on both `divergence` and `stochastic_crossover`) raises
    rather than guessing."""
    if path in known_paths or any(part.isdigit() for part in path.split(".")):
        return path
    suffix = f".{path}"
    matches = sorted(p for p in known_paths if p.endswith(suffix))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"grid path {path!r} is ambiguous -- it matches multiple config fields: "
            f"{', '.join(matches)}. Use the full dotted path (e.g. {matches[0]!r}) to "
            "disambiguate."
        )
    raise ValueError(
        f"grid path {path!r} doesn't match any Config field, at any nesting depth. Grid keys "
        "are dotted paths from the config root (e.g. 'indicators.rsi_length', not just "
        "'rsi_length') -- see docs/CONFIGURATION.md#grid-search."
    )


def _check_not_a_string(node: Any, path: str) -> None:
    if isinstance(node, str):
        raise ValueError(
            f"grid.{path} tries to reach inside a plain string value ({node!r}) -- this usually "
            "means `strategy` is still a named-preset reference (see "
            "docs/CONFIGURATION.md#custom-strategies) at this point in the path, which can't be "
            "indexed into. Sweep `strategy` itself (a list of preset names) instead of a "
            "sub-field of one, or inline the strategy block instead of naming it."
        )


def _get_child(node: Any, part: str, path: str) -> Any:
    _check_not_a_string(node, path)
    return node[int(part)] if isinstance(node, list) else node[part]


def _set_child(node: Any, part: str, value: Any, path: str) -> None:
    _check_not_a_string(node, path)
    if isinstance(node, list):
        node[int(part)] = value
    else:
        node[part] = value


def _set_by_path(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """A deep copy of `data` with the value at dotted `path` replaced by `value`. A path segment
    that's purely digits is treated as a list index when the current node is a list -- e.g.
    `strategy.sell_rule_groups.0.0.threshold` reaches into the first rule of the first group, so a
    single rule's field can be swept without having to restate the entire sell_rule_groups
    structure per grid candidate -- and as a dict key otherwise."""
    result = copy.deepcopy(data)
    node = result
    *parents, leaf = path.split(".")
    traversed: list[str] = []
    for part in parents:
        traversed.append(part)
        node = _get_child(node, part, ".".join(traversed))
    _set_child(node, leaf, value, path)
    return result


def expand_grid(data: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """[(chosen_overrides, expanded_data), ...] for every combination of the top-level `grid`
    section's axes. `chosen_overrides` maps each swept dotted path to the one value used in that
    combination -- meant for labeling a run's output, not for re-loading. `expanded_data` has the
    `grid` key removed and each axis's dotted path set to its chosen value for that combination.
    Returns a single `({}, data-without-grid)` pair, unchanged, when there's no `grid` section (or
    it's empty) -- the common case."""
    data = dict(data)
    grid_section = data.pop("grid", None) or {}

    for path, candidates in grid_section.items():
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                f"grid.{path} must be a non-empty YAML list of candidate values, got "
                f"{candidates!r}"
            )

    if not grid_section:
        return [({}, data)]

    known_paths = _known_schema_paths()
    grid_section = {
        _resolve_grid_path(path, known_paths): candidates
        for path, candidates in grid_section.items()
    }

    paths = list(grid_section.keys())
    value_lists = list(grid_section.values())
    combos = []
    for combo in itertools.product(*value_lists):
        chosen = dict(zip(paths, combo, strict=True))
        expanded = data
        for path, value in chosen.items():
            expanded = _set_by_path(expanded, path, value)
        combos.append((chosen, expanded))
    return combos


@dataclass
class GridPoint:
    # dotted path -> value used in this combo, e.g. {"indicators.rsi_length": 12}
    overrides: dict[str, Any]
    config: Config


def load_config_grid(*overrides: Path) -> list[GridPoint]:
    """Like `load_config`, but expands every grid axis found across the merged YAML into one
    `Config` per combination. A config with no swept fields returns a single-element list (with
    an empty `overrides` dict), so callers can always iterate uniformly rather than branching on
    whether a grid search was actually requested.

    `strategy` may be a plain string naming a preset under `strategies:` (see
    `loader._resolve_named_strategy`) instead of an inline block -- resolved per combination
    (after grid expansion, not before), so `grid: {strategy: [name1, name2]}` sweeps between named
    presets just like any other field."""
    merged = _load_merged_dict(*overrides)
    return [
        GridPoint(
            overrides=chosen, config=Config.model_validate(_resolve_named_strategy(expanded))
        )
        for chosen, expanded in expand_grid(merged)
    ]
