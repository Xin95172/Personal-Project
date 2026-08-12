"""多人主池、邊池建立與攤牌分配。"""

from dataclasses import dataclass
from typing import Callable, Iterable

from poker_solver.engine.table import Position, TablePlayer


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: frozenset[Position]


def build_pots(players: Iterable[TablePlayer]) -> tuple[Pot, ...]:
    """依總投入額建立主池與所有邊池；fold 玩家仍貢獻金額但不可贏池。"""
    values = tuple(players)
    levels = sorted({player.committed_total for player in values if player.committed_total > 0})
    previous = 0
    pots: list[Pot] = []
    for level in levels:
        contributors = [player for player in values if player.committed_total >= level]
        amount = (level - previous) * len(contributors)
        eligible = frozenset(player.position for player in contributors if not player.folded)
        if amount:
            pots.append(Pot(amount, eligible))
        previous = level
    return tuple(pots)


def settle_pots(
    pots: Iterable[Pot],
    hand_strength: Callable[[Position], tuple[int, ...]],
) -> dict[Position, int]:
    """依牌力函式分配所有 pots；餘數按 Position 宣告順序發放。"""
    payouts: dict[Position, int] = {}
    for pot in pots:
        if not pot.eligible:
            continue
        best = max(hand_strength(position) for position in pot.eligible)
        winners = sorted(
            [position for position in pot.eligible if hand_strength(position) == best],
            key=lambda position: list(Position).index(position),
        )
        share, remainder = divmod(pot.amount, len(winners))
        for index, winner in enumerate(winners):
            payouts[winner] = payouts.get(winner, 0) + share + (1 if index < remainder else 0)
    return payouts
