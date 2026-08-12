from poker_solver.engine.showdown import settle_multiway_showdown
from poker_solver.engine.table import Position, TablePlayer


def player(position, committed, folded=False):
    return TablePlayer(position, stack=0, committed_total=committed, folded=folded)


def test_multiway_showdown_settles_main_and_side_pots():
    players = (player(Position.UTG, 100), player(Position.UTG_1, 50), player(Position.MP, 25))
    board = ("As", "Kd", "Qh", "Jc", "2s")
    holes = {
        Position.UTG: ("8h", "8c"),
        Position.UTG_1: ("9h", "9c"),
        Position.MP: ("Ts", "3d"),
    }
    payouts, utility = settle_multiway_showdown(players, board, holes)
    assert payouts == {Position.MP: 75, Position.UTG_1: 50, Position.UTG: 50}
    assert utility == {Position.UTG: -50, Position.UTG_1: 0, Position.MP: 50}


def test_folded_player_requires_no_cards_and_cannot_win():
    players = (player(Position.UTG, 50), player(Position.UTG_1, 50, folded=True), player(Position.MP, 50))
    board = ("As", "Kd", "Qh", "Jc", "2s")
    holes = {Position.UTG: ("Ts", "3d"), Position.MP: ("9h", "9c")}
    payouts, utility = settle_multiway_showdown(players, board, holes)
    assert payouts == {Position.UTG: 150}
    assert utility[Position.UTG] == 100
    assert utility[Position.UTG_1] == -50
    assert utility[Position.MP] == -50
