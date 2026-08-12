from poker_solver.engine.money import bb_to_units as bb
from poker_solver.engine.postflop_game import (
    Street,
    abstract_actions,
    advance_to_next_street,
    advance_to_river,
    apply_action,
    create_turn_game,
    create_flop_game,
    is_chance_node,
    is_terminal,
    terminal_utility,
)
from poker_solver.engine.river_game import Action, ActionType, SizingPolicy


def game():
    return create_turn_game(
        ("As", "Kd", "Qh", "Jc"),
        ("Ts", "3d"),
        ("9h", "9c"),
        initial_pot_bb=10,
        effective_stack_bb=95,
    )


def test_turn_check_check_becomes_chance_node_then_river():
    state = apply_action(game(), Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert is_chance_node(state)
    assert not is_terminal(state)
    assert abstract_actions(state, SizingPolicy()) == ()

    state = advance_to_river(state, "2s")
    assert state.street is Street.RIVER
    assert state.current_player.value == "oop"
    assert state.pot == bb(10)
    assert state.board[-1] == "2s"


def test_turn_bet_call_then_river_showdown():
    state = apply_action(game(), Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.CALL))
    assert is_chance_node(state)
    assert state.pot == bb(20)

    state = advance_to_river(state, "2s")
    state = apply_action(state, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert is_terminal(state)
    assert terminal_utility(state) == (bb(10), -bb(10))


def test_turn_all_in_call_deals_river_then_ends_immediately():
    state = apply_action(game(), Action(ActionType.ALL_IN))
    state = apply_action(state, Action(ActionType.CALL))
    assert is_chance_node(state)
    state = advance_to_river(state, "2s")
    assert is_terminal(state)
    assert terminal_utility(state) == (bb(100), -bb(100))


def test_re_raise_limit_resets_when_turn_advances_to_river():
    policy = SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1)
    state = apply_action(game(), Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.RAISE, bb(20)))
    assert not any(action.kind is ActionType.RAISE for action in abstract_actions(state, policy))
    state = apply_action(state, Action(ActionType.CALL))
    state = advance_to_river(state, "2s")
    state = apply_action(state, Action(ActionType.BET, bb(10)))
    assert any(action.kind is ActionType.RAISE for action in abstract_actions(state, policy))


def test_turn_fold_is_terminal_without_dealing_river():
    state = apply_action(game(), Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.FOLD))
    assert is_terminal(state)
    assert terminal_utility(state) == (bb(5), -bb(5))


def test_advance_to_river_rejects_invalid_state_and_cards():
    state = game()
    try:
        advance_to_river(state, "2s")
    except ValueError:
        pass
    else:
        raise AssertionError("must not deal river before turn betting ends")

    state = apply_action(state, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    for card in ("As", "Ts", "ZZ"):
        try:
            advance_to_river(state, card)
        except ValueError:
            pass
        else:
            raise AssertionError("must reject unavailable river card")


def test_flop_to_turn_to_river_transition():
    state = create_flop_game(("As", "Kd", "Qh"), ("Ts", "3d"), ("9h", "9c"))
    state = apply_action(state, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert is_chance_node(state)
    state = advance_to_next_street(state, "Jc")
    assert state.street is Street.TURN
    state = apply_action(state, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    state = advance_to_next_street(state, "2s")
    assert state.street is Street.RIVER
    state = apply_action(state, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert is_terminal(state)


def test_flop_all_in_deals_both_remaining_public_cards():
    state = create_flop_game(("As", "Kd", "Qh"), ("Ts", "3d"), ("9h", "9c"))
    state = apply_action(state, Action(ActionType.ALL_IN))
    state = apply_action(state, Action(ActionType.CALL))
    state = advance_to_next_street(state, "Jc")
    assert is_chance_node(state)
    assert state.street is Street.TURN
    state = advance_to_next_street(state, "2s")
    assert is_terminal(state)
