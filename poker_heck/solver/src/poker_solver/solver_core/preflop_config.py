"""8-Max preflop solver JSON 設定。"""

import json
from itertools import combinations
from math import ceil
from pathlib import Path

from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.table import Position
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
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
    kind = spec.get("kind")
    if kind == "all_combos":
        full_range = _all_combos()
        return {position: full_range for position in Position}
    if kind == "top_percent":
        default_percent = spec.get("percent")
        overrides = spec.get("percent_by_position", {})
        if not isinstance(overrides, dict):
            raise ValueError("percent_by_position must be an object")
        return {
            position: _top_percent(float(overrides.get(position.value, default_percent)))
            for position in Position
        }
    raise ValueError("range_spec.kind must be all_combos or top_percent")


def _all_combos() -> WeightedRange:
    return WeightedRange(tuple(Combo(cards) for cards in combinations(_DECK, 2)))


def _top_percent(percent: float) -> WeightedRange:
    """依可重現的 preflop 起手牌強度排序取前百分比，不指定個別實體牌。"""
    if not 0 < percent <= 100:
        raise ValueError("top_percent percent must be in (0, 100]")
    ordered = sorted(combinations(_DECK, 2), key=lambda cards: (_hand_strength(cards), cards), reverse=True)
    count = ceil(len(ordered) * percent / 100)
    return WeightedRange(tuple(Combo(cards) for cards in ordered[:count]))


def _hand_strength(cards: tuple[str, str]) -> int:
    """簡潔、可重現的牌力排序；pair 優先，其次高張、suited 與連接性。"""
    first, second = ("23456789TJQKA".index(card[0]) + 2 for card in cards)
    high, low = max(first, second), min(first, second)
    if high == low:
        return 10_000 + high
    suited_bonus = 8 if cards[0][1] == cards[1][1] else 0
    gap_penalty = max(0, high - low - 1) * 3
    return high * 100 + low * 5 + suited_bonus - gap_penalty


_DECK = tuple(f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs")
