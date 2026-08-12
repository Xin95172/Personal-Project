"""8-Max preflop solver JSON 設定。"""

import json
from pathlib import Path

from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.table import Position
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.range_spec import expand_range_spec
from poker_solver.solver_core.river_mccfr import Combo, WeightedRange


def load_preflop_trainer(path: str | Path) -> MultiwayPreflopMCCFRTrainer:
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)
    try:
        ranges = _load_ranges(raw)
        sizing = raw.get("sizing_policy", {})
        return MultiwayPreflopMCCFRTrainer(
            ranges=ranges,
            sizing_policy=PreflopSizingPolicy(
                open_sizes_bb=tuple(sizing.get("open_sizes_bb", (2.0, 2.5, 3.0))),
                re_raise_multipliers=tuple(sizing.get("re_raise_multipliers", (2.5, 3.5))),
                include_all_in=bool(sizing.get("include_all_in", True)),
                max_raises=sizing.get("max_raises", 3),
            ),
            stack_bb=raw.get("stack_bb", 100),
            seed=int(raw.get("seed", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid preflop solve configuration: {error}") from error


def _load_ranges(raw: dict[str, object]) -> dict[Position, WeightedRange]:
    """載入明確 range，或展開完整 1,326 種起手牌組合。"""
    if "ranges" in raw:
        entries_by_position = raw["ranges"]
        if not isinstance(entries_by_position, dict):
            raise ValueError("ranges must be an object")
        return {
            position: WeightedRange(
                tuple(Combo(tuple(entry["cards"]), float(entry.get("weight", 1.0))) for entry in entries_by_position[position.value])  # type: ignore[index,arg-type]
            )
            for position in Position
        }
    spec = raw.get("range_spec")
    if not isinstance(spec, dict):
        raise ValueError("range_spec must be an object")
    overrides = spec.get("percent_by_position", {})
    if not isinstance(overrides, dict):
        raise ValueError("percent_by_position must be an object")
    return {
        position: expand_range_spec({**spec, "percent": overrides.get(position.value, spec.get("percent"))})
        for position in Position
    }
