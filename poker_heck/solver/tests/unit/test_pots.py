from poker_solver.engine.pots import build_pots, settle_pots
from poker_solver.engine.table import Position, TablePlayer


def player(position, committed, folded=False):
    return TablePlayer(position, stack=0, committed_total=committed, folded=folded)


def test_build_pots_creates_main_and_multiple_side_pots():
    pots = build_pots(
        (
            player(Position.UTG, 100),
            player(Position.UTG_1, 50),
            player(Position.MP, 25),
        )
    )
    assert [pot.amount for pot in pots] == [75, 50, 50]
    assert pots[0].eligible == {Position.UTG, Position.UTG_1, Position.MP}
    assert pots[1].eligible == {Position.UTG, Position.UTG_1}
    assert pots[2].eligible == {Position.UTG}


def test_folded_player_contributes_but_cannot_win():
    pots = build_pots((player(Position.UTG, 50), player(Position.UTG_1, 50, folded=True), player(Position.MP, 25)))
    assert [pot.amount for pot in pots] == [75, 50]
    assert Position.UTG_1 not in pots[0].eligible


def test_settlement_awards_each_side_pot_to_eligible_winner():
    pots = build_pots((player(Position.UTG, 100), player(Position.UTG_1, 50), player(Position.MP, 25)))
    strengths = {Position.UTG: (1,), Position.UTG_1: (3,), Position.MP: (5,)}
    payouts = settle_pots(pots, strengths.__getitem__)
    assert payouts == {Position.MP: 75, Position.UTG_1: 50, Position.UTG: 50}


def test_settlement_splits_odd_chip_by_position_order():
    pots = build_pots((player(Position.UTG, 5), player(Position.UTG_1, 5), player(Position.MP, 5)))
    strengths = {Position.UTG: (5,), Position.UTG_1: (5,), Position.MP: (1,)}
    payouts = settle_pots(pots, strengths.__getitem__)
    assert payouts == {Position.UTG: 8, Position.UTG_1: 7}
