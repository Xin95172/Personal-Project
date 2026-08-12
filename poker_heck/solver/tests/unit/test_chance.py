from random import Random

import pytest

from poker_solver.engine.chance import FULL_DECK, TurnToRiverScenario, remaining_cards


def scenario():
    return TurnToRiverScenario(
        turn_board=("As", "Kd", "Qh", "Jc"),
        oop_hole_cards=("Ts", "3d"),
        ip_hole_cards=("9h", "9c"),
        pot_bb=20,
        effective_stack_bb=80,
    )


def test_remaining_cards_excludes_known_cards():
    cards = remaining_cards(("As", "Kd", "Qh", "Jc", "Ts", "3d", "9h", "9c"))
    assert len(cards) == 44
    assert all(card not in cards for card in ("As", "Kd", "Qh", "Jc", "Ts", "3d", "9h", "9c"))
    assert set(cards).issubset(FULL_DECK)


@pytest.mark.parametrize("known_cards", [("As", "As"), ("As", "ZZ")])
def test_remaining_cards_rejects_bad_input(known_cards):
    with pytest.raises(ValueError):
        remaining_cards(known_cards)


def test_turn_to_river_scenario_deals_only_legal_cards():
    turn = scenario()
    assert len(turn.legal_river_cards()) == 44
    river = turn.deal_river("2s")
    assert river.board == ("As", "Kd", "Qh", "Jc", "2s")
    assert river.pot == 2000
    assert river.oop.stack == 8000


def test_turn_to_river_sampling_is_seeded_and_legal():
    turn = scenario()
    first = turn.sample_river(Random(5))
    second = turn.sample_river(Random(5))
    assert first.board == second.board
    assert first.board[-1] in turn.legal_river_cards()


@pytest.mark.parametrize(
    "river_card",
    ["As", "Ts", "ZZ"],
)
def test_turn_to_river_rejects_unavailable_cards(river_card):
    with pytest.raises(ValueError):
        scenario().deal_river(river_card)
