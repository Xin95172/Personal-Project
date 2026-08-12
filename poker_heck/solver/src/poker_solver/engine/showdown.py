"""多人 showdown 與 side-pot 結算。"""

from typing import Mapping, Sequence

from poker_solver.engine.pots import build_pots, settle_pots
from poker_solver.engine.river_game import evaluate_seven_cards
from poker_solver.engine.table import Position, TablePlayer


def settle_multiway_showdown(
    players: Sequence[TablePlayer],
    board: Sequence[str],
    hole_cards: Mapping[Position, tuple[str, str]],
) -> tuple[dict[Position, int], dict[Position, int]]:
    """回傳各位置的 payout 與相對本手投入的 net utility。"""
    if len(board) != 5:
        raise ValueError("showdown requires a five-card board")
    active = [player for player in players if not player.folded]
    if any(player.position not in hole_cards for player in active):
        raise ValueError("each active player requires hole cards")

    strengths = {
        player.position: evaluate_seven_cards((*hole_cards[player.position], *board))
        for player in active
    }
    payouts = settle_pots(build_pots(players), strengths.__getitem__)
    utility = {player.position: payouts.get(player.position, 0) - player.committed_total for player in players}
    if sum(utility.values()) != 0:
        raise RuntimeError("showdown utility must be zero-sum")
    return payouts, utility
