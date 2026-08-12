"""公共牌 chance node 的牌組管理與 turn→river 轉換。"""

from dataclasses import dataclass
from random import Random
from typing import Iterable

from poker_solver.engine.river_game import RiverGameState, create_river_game


RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = tuple(f"{rank}{suit}" for rank in RANKS for suit in SUITS)


def remaining_cards(known_cards: Iterable[str]) -> tuple[str, ...]:
    """回傳扣除已知牌後、可供 chance node 使用的牌。"""
    known = tuple(known_cards)
    if len(set(known)) != len(known):
        raise ValueError("known cards must not contain duplicates")
    if any(card not in FULL_DECK for card in known):
        raise ValueError("known cards contain an invalid card")
    return tuple(card for card in FULL_DECK if card not in known)


@dataclass(frozen=True)
class TurnToRiverScenario:
    """已完成 turn 下注、即將發 river 的雙人局面。

    這是 chance node 的資料模型；turn 的下注樹將在下一階段接到它前面。
    """

    turn_board: tuple[str, str, str, str]
    oop_hole_cards: tuple[str, str]
    ip_hole_cards: tuple[str, str]
    pot_bb: int | float | str = 10
    effective_stack_bb: int | float | str = 95

    def __post_init__(self) -> None:
        if len(self.turn_board) != 4:
            raise ValueError("turn board must contain four cards")
        if len(self.oop_hole_cards) != 2 or len(self.ip_hole_cards) != 2:
            raise ValueError("each player must have two hole cards")
        # remaining_cards performs the canonical uniqueness and card validation.
        remaining_cards((*self.turn_board, *self.oop_hole_cards, *self.ip_hole_cards))

    def legal_river_cards(self) -> tuple[str, ...]:
        return remaining_cards((*self.turn_board, *self.oop_hole_cards, *self.ip_hole_cards))

    def deal_river(self, river_card: str) -> RiverGameState:
        if river_card not in self.legal_river_cards():
            raise ValueError("river card is not available in this scenario")
        return create_river_game(
            (*self.turn_board, river_card),
            self.oop_hole_cards,
            self.ip_hole_cards,
            initial_pot_bb=self.pot_bb,
            effective_stack_bb=self.effective_stack_bb,
        )

    def sample_river(self, rng: Random) -> RiverGameState:
        cards = self.legal_river_cards()
        return self.deal_river(cards[rng.randrange(len(cards))])
