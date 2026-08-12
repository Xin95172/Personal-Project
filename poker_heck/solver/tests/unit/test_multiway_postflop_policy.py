from poker_solver.engine.money import bb_to_units as bb
import pytest
from poker_solver.engine.multiway_postflop_policy import (
    MultiwayPostflopSizingPolicy,
    abstract_multiway_postflop_actions,
    format_multiway_postflop_action_as_pot_ratio,
)
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import Position, advance_preflop_to_flop, apply_action, apply_multiway_postflop_action, create_8max_preflop


def _flop_state():
    preflop = create_8max_preflop()
    for _ in range(7):
        preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    return advance_preflop_to_flop(preflop, ("As", "Kd", "Qh"))


def test_multiway_postflop_policy_uses_pot_ratios_for_bets():
    state = _flop_state()
    actions = abstract_multiway_postflop_actions(state, MultiwayPostflopSizingPolicy(bet_sizes=(0.5, 1.0)))
    assert actions == (
        Action(ActionType.CHECK),
        Action(ActionType.BET, bb(4)),
        Action(ActionType.BET, bb(8)),
        Action(ActionType.ALL_IN),
    )


def test_multiway_postflop_policy_offers_fold_call_and_ratio_raises_when_facing_bet():
    state = apply_multiway_postflop_action(_flop_state(), Action(ActionType.BET, bb(4)))
    assert state.current_player is Position.BB
    actions = abstract_multiway_postflop_actions(state, MultiwayPostflopSizingPolicy(raise_sizes=(1.0,), include_all_in=False))
    assert actions == (Action(ActionType.FOLD), Action(ActionType.CALL), Action(ActionType.RAISE, bb(20)))
    assert format_multiway_postflop_action_as_pot_ratio(state, actions[-1]) == "加注 100% pot-after-call"


def test_multiway_postflop_policy_validates_sizes_and_formats_all_action_types():
    with pytest.raises(ValueError, match="正數"):
        MultiwayPostflopSizingPolicy(bet_sizes=(0,))
    with pytest.raises(ValueError, match="非負"):
        MultiwayPostflopSizingPolicy(max_re_raises=-1)
    state = _flop_state()
    assert format_multiway_postflop_action_as_pot_ratio(state, Action(ActionType.CHECK)) == "check"
    assert format_multiway_postflop_action_as_pot_ratio(state, Action(ActionType.BET, bb(4))) == "下注 50% pot"
    assert "全下下注" in format_multiway_postflop_action_as_pot_ratio(state, Action(ActionType.ALL_IN))
    facing_bet = apply_multiway_postflop_action(state, Action(ActionType.BET, bb(4)))
    assert format_multiway_postflop_action_as_pot_ratio(facing_bet, Action(ActionType.FOLD)) == "fold"
    assert format_multiway_postflop_action_as_pot_ratio(facing_bet, Action(ActionType.CALL)) == "call"
    assert "全下加注" in format_multiway_postflop_action_as_pot_ratio(facing_bet, Action(ActionType.ALL_IN))
