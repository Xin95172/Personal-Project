"""將宣告式 range spec 展開為可供 MCCFR 抽樣的 weighted range。"""

from __future__ import annotations

from itertools import combinations
from math import ceil
from typing import Any

from poker_solver.engine.chance import FULL_DECK
from poker_solver.solver_core.river_mccfr import Combo, WeightedRange


def expand_range_spec(spec: dict[str, Any], *, excluded_cards: tuple[str, ...] = ()) -> WeightedRange:
    """支援完整 combo 或依可重現牌力排序取前百分比。"""
    available = tuple(card for card in FULL_DECK if card not in excluded_cards)
    combos = tuple(combinations(available, 2))
    kind = spec["kind"]
    if kind == "all_combos":
        return WeightedRange(tuple(Combo(cards) for cards in combos))
    if kind != "top_percent":
        raise ValueError("range_spec.kind must be all_combos or top_percent")
    percent = float(spec["percent"])
    if not 0 < percent <= 100:
        raise ValueError("range_spec.percent must be in (0, 100]")
    ordered = sorted(combos, key=lambda cards: (_hand_strength(cards), cards), reverse=True)
    return WeightedRange(tuple(Combo(cards) for cards in ordered[:ceil(len(ordered) * percent / 100)]))


def _hand_strength(cards: tuple[str, str]) -> int:
    ranks = "23456789TJQKA"
    first, second = (ranks.index(card[0]) + 2 for card in cards)
    high, low = max(first, second), min(first, second)
    if high == low:
        return 10_000 + high
    return high * 100 + low * 5 + (8 if cards[0][1] == cards[1][1] else 0) - max(0, high - low - 1) * 3
