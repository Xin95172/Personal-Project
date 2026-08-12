import pytest

from poker_solver.engine.money import bb_to_units as bb
from poker_solver.engine.river_game import (
    Action,
    ActionType,
    Player,
    SizingPolicy,
    abstract_actions,
    all_legal_actions,
    apply_action,
    current_player,
    format_action_as_pot_ratio,
    infoset_key,
    is_legal_action,
    is_terminal,
    terminal_utility,
)


def test_unopened_pot_exposes_all_integer_bets_and_all_in(winning_oop_game):
    actions = all_legal_actions(winning_oop_game)
    assert actions[0] == Action(ActionType.CHECK)
    assert Action(ActionType.BET, 1) in actions  # 0.01 BB 的最小內部單位
    assert Action(ActionType.BET, bb(94)) in actions
    assert Action(ActionType.ALL_IN) in actions


def test_ip_can_bet_after_oop_checks(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.CHECK))
    assert state.current_player is Player.IP
    assert is_legal_action(state, Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.BET, bb(5)))
    assert state.current_player is Player.OOP
    assert state.pot == bb(15)


def test_current_player_public_api(winning_oop_game):
    assert current_player(winning_oop_game) is Player.OOP


def test_minimum_raise_boundary(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(5)))
    assert is_legal_action(state, Action(ActionType.RAISE, bb(10)))
    with pytest.raises(ValueError, match="illegal action"):
        apply_action(state, Action(ActionType.RAISE, bb(9)))


def test_full_re_raise_updates_minimum_raise(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.RAISE, bb(15)))
    assert state.last_full_raise_size == bb(10)
    assert is_legal_action(state, Action(ActionType.RAISE, bb(25)))
    with pytest.raises(ValueError, match="illegal action"):
        apply_action(state, Action(ActionType.RAISE, bb(24)))


def test_multiple_re_raises_then_call(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.RAISE, bb(15)))
    state = apply_action(state, Action(ActionType.RAISE, bb(30)))
    state = apply_action(state, Action(ActionType.RAISE, bb(50)))
    assert is_legal_action(state, Action(ActionType.CALL))
    state = apply_action(state, Action(ActionType.CALL))
    assert is_terminal(state)
    assert state.pot == bb(110)
    assert terminal_utility(state) == (bb(55), -bb(55))


def test_all_in_requires_call_not_an_all_in_alias(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.ALL_IN))
    assert is_legal_action(state, Action(ActionType.CALL))
    assert not is_legal_action(state, Action(ActionType.ALL_IN))
    state = apply_action(state, Action(ActionType.CALL))
    assert is_terminal(state)


def test_full_enumeration_is_available_only_as_a_diagnostic_api(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(5)))
    actions = all_legal_actions(state)
    assert Action(ActionType.FOLD) in actions
    assert Action(ActionType.CALL) in actions
    assert Action(ActionType.RAISE, bb(10)) in actions


@pytest.mark.parametrize(
    "first_action",
    [Action(ActionType.FOLD), Action(ActionType.CALL), Action(ActionType.RAISE, 1)],
)
def test_illegal_actions_are_rejected_before_a_bet(winning_oop_game, first_action):
    with pytest.raises(ValueError, match="illegal action"):
        apply_action(winning_oop_game, first_action)


def test_no_action_is_legal_after_fold(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.FOLD))
    assert all_legal_actions(state) == ()
    with pytest.raises(ValueError, match="illegal action"):
        apply_action(state, Action(ActionType.CHECK))


def test_infoset_excludes_opponent_hole_cards(winning_oop_game):
    key = infoset_key(winning_oop_game, Player.OOP)
    assert winning_oop_game.oop.hole_cards in key
    assert winning_oop_game.ip.hole_cards not in key


def test_abstract_actions_use_configured_pot_fractions(winning_oop_game):
    policy = SizingPolicy(bet_sizes=(0.5, 1.5), raise_sizes=(0.75,), include_all_in=False)
    assert abstract_actions(winning_oop_game, policy) == (
        Action(ActionType.CHECK),
        Action(ActionType.BET, bb(5)),
        Action(ActionType.BET, bb(15)),
    )

    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(5)))
    assert abstract_actions(state, policy) == (
        Action(ActionType.FOLD),
        Action(ActionType.CALL),
        Action(ActionType.RAISE, bb(20)),
    )


def test_action_formatting_uses_pot_ratios_not_units(winning_oop_game):
    assert format_action_as_pot_ratio(winning_oop_game, Action(ActionType.BET, bb(5))) == "bet 50% pot"
    assert format_action_as_pot_ratio(winning_oop_game, Action(ActionType.ALL_IN)) == "all-in (950% pot)"
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(5)))
    assert format_action_as_pot_ratio(state, Action(ActionType.RAISE, bb(20))) == "raise 75% pot-after-call"


def test_abstract_actions_filters_sizes_that_are_not_legal(winning_oop_game):
    policy = SizingPolicy(bet_sizes=(0.0001, 100.0), include_all_in=True)
    assert abstract_actions(winning_oop_game, policy) == (
        Action(ActionType.CHECK),
        Action(ActionType.BET, 1),
        Action(ActionType.ALL_IN),
    )


def test_abstract_actions_reject_non_positive_pot_fractions(winning_oop_game):
    with pytest.raises(ValueError, match="fractions"):
        abstract_actions(winning_oop_game, SizingPolicy(bet_sizes=(0.0,)))


def test_abstract_actions_is_empty_at_a_terminal_state(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert abstract_actions(state, SizingPolicy()) == ()


def test_default_action_abstraction_contains_requested_pot_sizes():
    policy = SizingPolicy()
    assert policy.bet_sizes == (0.33, 0.50, 0.75, 1.0, 1.5, 2.0)
    assert policy.raise_sizes == (0.33, 0.50, 0.75, 1.0, 1.5, 2.0)
