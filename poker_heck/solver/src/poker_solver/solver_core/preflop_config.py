"""8-Max preflop solver JSON 設定。"""

import json
from pathlib import Path

from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.table import Position
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.integrated_solver import ContinuationSettings, attach_postflop_continuation
from poker_solver.solver_core.range_spec import expand_range_spec
from poker_solver.solver_core.river_mccfr import Combo, WeightedRange


def load_preflop_trainer(path: str | Path) -> MultiwayPreflopMCCFRTrainer:
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)
    try:
        ranges = _load_ranges(raw)
        sizing = raw.get("sizing_policy", {})
        trainer = MultiwayPreflopMCCFRTrainer(
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
        continuation = raw.get("continuation")
        if continuation is not None:
            if not isinstance(continuation, dict) or not bool(continuation.get("enabled", False)):
                raise ValueError("continuation must be an object with enabled=true")
            attach_postflop_continuation(
                trainer,
                ContinuationSettings(
                    subgame_iterations=int(continuation.get("subgame_iterations", 100)),
                    value_rollouts=int(continuation.get("value_rollouts", 4)),
                    max_cached_subgames=continuation.get("max_cached_subgames"),
                    bet_sizes=tuple(continuation.get("bet_sizes", (0.33, 0.5, 0.75, 1.0, 1.5, 2.0))),
                    raise_sizes=tuple(continuation.get("raise_sizes", (0.33, 0.5, 0.75, 1.0, 1.5, 2.0))),
                    include_all_in=bool(continuation.get("include_all_in", True)),
                    max_re_raises=continuation.get("max_re_raises", 1),
                    strategy_db=str((Path(path).parent / continuation["strategy_db"]).resolve()) if continuation.get("strategy_db") else None,
                    range_profile_id=str(continuation.get("range_profile_id", raw.get("range_spec", {}).get("kind", "all_combos"))),
                    solver_version=str(continuation.get("solver_version", "multiway-postflop-grid-v1")),
                ),
            )
        return trainer
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
