import pytest

from poker_solver.engine.money import bb_to_units as bb
from poker_solver.engine.preflop_policy import PreflopSizingPolicy, abstract_actions, format_action
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import Position, apply_action, create_8max_preflop


def test_opening_policy_generates_configured_open_sizes():
    state = create_8max_preflop()
    actions = abstract_actions(state, PreflopSizingPolicy(open_sizes_bb=(2.0, 2.5), include_all_in=False))
    assert actions == (
        Action(ActionType.FOLD),
        Action(ActionType.CALL),
        Action(ActionType.RAISE, bb(2)),
        Action(ActionType.RAISE, bb(2.5)),
    )


def test_policy_uses_re_raise_multipliers_after_open():
    state = apply_action(create_8max_preflop(), Action(ActionType.RAISE, bb(2.5)))
    actions = abstract_actions(state, PreflopSizingPolicy(re_raise_multipliers=(3.0,), include_all_in=False))
    assert state.current_player is Position.UTG_1
    assert Action(ActionType.RAISE, bb(7.5)) in actions


def test_policy_caps_number_of_preflop_raises():
    policy = PreflopSizingPolicy(max_raises=1)
    state = apply_action(create_8max_preflop(), Action(ActionType.RAISE, bb(2.5)))
    actions = abstract_actions(state, policy)
    assert not any(action.kind is ActionType.RAISE for action in actions)


def test_policy_validates_configuration():
    with pytest.raises(ValueError, match="positive"):
        PreflopSizingPolicy(open_sizes_bb=(0.0,))


def test_preflop_action_formatting_uses_bb():
    state = create_8max_preflop()
    assert format_action(state, Action(ActionType.RAISE, bb(2.5))) == "raise to 2.5 BB"
