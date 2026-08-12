import pytest

from poker_solver.engine.money import bb_to_units as bb
from poker_solver.engine.river_game import Action, ActionType, apply_action, create_river_game, is_terminal, terminal_utility


def test_check_check_reaches_showdown(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert is_terminal(state)
    assert terminal_utility(state) == (bb(5), -bb(5))


def test_bet_fold_awards_the_whole_pot(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(8)))
    state = apply_action(state, Action(ActionType.FOLD))
    assert state.pot == bb(18)
    assert terminal_utility(state) == (bb(5), -bb(5))


def test_oop_fold_awards_the_whole_pot_to_ip(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.FOLD))
    assert terminal_utility(state) == (-bb(5), bb(5))


def test_bet_call_reaches_showdown(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.BET, bb(8)))
    state = apply_action(state, Action(ActionType.CALL))
    assert is_terminal(state)
    assert state.pot == bb(26)
    assert terminal_utility(state) == (bb(13), -bb(13))


def test_all_in_call_reaches_showdown(winning_oop_game):
    state = apply_action(winning_oop_game, Action(ActionType.ALL_IN))
    state = apply_action(state, Action(ActionType.CALL))
    assert state.pot == bb(200)
    assert terminal_utility(state) == (bb(100), -bb(100))


def test_split_pot_is_zero_sum(split_pot_game):
    state = apply_action(split_pot_game, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert terminal_utility(state) == (0, 0)


def test_ip_can_win_at_showdown():
    state = create_river_game(("As", "Kd", "Qh", "Jc", "2s"), ("9h", "9c"), ("Ts", "3d"))
    state = apply_action(state, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert terminal_utility(state) == (-bb(5), bb(5))


def test_utility_requires_a_terminal_state(winning_oop_game):
    with pytest.raises(ValueError, match="terminal"):
        terminal_utility(winning_oop_game)
