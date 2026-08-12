import pytest

from poker_solver.engine.river_game import evaluate_seven_cards


@pytest.mark.parametrize(
    ("cards", "category"),
    [
        (("As", "Kd", "9h", "7c", "4s", "3d", "2c"), 0),  # high card
        (("As", "Ad", "9h", "7c", "4s", "3d", "2c"), 1),  # pair
        (("As", "Ad", "9h", "9c", "4s", "3d", "2c"), 2),  # two pair
        (("As", "Ad", "Ah", "7c", "4s", "3d", "2c"), 3),  # trips
        (("As", "Kd", "Qh", "Jc", "Ts", "3d", "2c"), 4),  # straight
        (("As", "Js", "9s", "7s", "4s", "3d", "2c"), 5),  # flush
        (("As", "Ad", "Ah", "9c", "9s", "3d", "2c"), 6),  # full house
        (("As", "Ad", "Ah", "Ac", "9s", "3d", "2c"), 7),  # quads
        (("As", "Ks", "Qs", "Js", "Ts", "3d", "2c"), 8),  # straight flush
    ],
)
def test_all_hand_categories(cards, category):
    assert evaluate_seven_cards(cards)[0] == category


def test_wheel_is_lower_than_six_high_straight():
    wheel = evaluate_seven_cards(("As", "5d", "4h", "3c", "2s", "Kd", "Qh"))
    six_high = evaluate_seven_cards(("6s", "5d", "4h", "3c", "2s", "Kd", "Qh"))
    assert wheel < six_high


def test_kicker_breaks_a_pair_tie():
    ace_kicker = evaluate_seven_cards(("As", "Ad", "Kc", "7h", "4s", "3d", "2c"))
    queen_kicker = evaluate_seven_cards(("Ah", "Ac", "Qc", "7h", "4s", "3d", "2c"))
    assert ace_kicker > queen_kicker


@pytest.mark.parametrize(
    "cards",
    [
        ("As", "Ad", "Ah", "Ac", "As", "3d", "2c"),
        ("As", "Kd", "Qh", "Jc", "Ts", "3d"),
    ],
)
def test_evaluator_rejects_invalid_card_collections(cards):
    with pytest.raises(ValueError):
        evaluate_seven_cards(cards)
