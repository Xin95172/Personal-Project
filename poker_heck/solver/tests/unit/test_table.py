from poker_solver.engine.money import bb_to_units as bb
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import (
    Position,
    advance_multiway_postflop_street,
    advance_preflop_to_flop,
    apply_action,
    apply_multiway_postflop_action,
    call_amount,
    create_8max_preflop,
    is_legal_action,
    is_multiway_postflop_terminal,
    is_legal_multiway_postflop_action,
    is_terminal,
    settle_multiway_postflop,
)


def test_8max_preflop_posts_blinds_and_starts_utg():
    state = create_8max_preflop()
    assert state.pot == bb(1.5)
    assert state.current_player is Position.UTG
    assert state.player(Position.SB).committed_this_street == bb(0.5)
    assert state.player(Position.BB).committed_this_street == bb(1)
    assert call_amount(state, Position.UTG) == bb(1)


def test_unraised_preflop_round_ends_after_big_blind_checks():
    state = create_8max_preflop()
    for _ in range(6):  # UTG through BTN call 1 BB
        state = apply_action(state, Action(ActionType.CALL))
    state = apply_action(state, Action(ActionType.CALL))  # SB completes 0.5 BB
    assert state.current_player is Position.BB
    assert is_legal_action(state, Action(ActionType.CHECK))
    state = apply_action(state, Action(ActionType.CHECK))
    assert is_terminal(state)
    assert state.betting_complete
    assert state.pot == bb(8)


def test_completed_preflop_advances_to_flop_with_small_blind_first():
    state = create_8max_preflop()
    for _ in range(7):
        state = apply_action(state, Action(ActionType.CALL))
    state = apply_action(state, Action(ActionType.CHECK))
    flop = advance_preflop_to_flop(state, ("As", "Kd", "Qh"))
    assert flop.current_player is Position.SB
    assert flop.pending_players == frozenset(Position)
    assert all(player.committed_this_street == 0 for player in flop.players)
    assert flop.call_amount(Position.SB) == 0
    assert flop.actable_positions == frozenset(Position)
    assert not flop.hand_ended


def test_preflop_raise_resets_pending_players():
    state = create_8max_preflop()
    state = apply_action(state, Action(ActionType.RAISE, bb(3)))
    assert state.current_bet == bb(3)
    assert state.current_player is Position.UTG_1
    assert Position.UTG not in state.pending_players
    assert Position.BB in state.pending_players
    assert is_legal_action(state, Action(ActionType.CALL))


def test_preflop_raise_then_everyone_folds_ends_hand():
    state = apply_action(create_8max_preflop(), Action(ActionType.RAISE, bb(3)))
    while not is_terminal(state):
        state = apply_action(state, Action(ActionType.FOLD))
    assert state.hand_ended
    assert not state.player(Position.UTG).folded
    assert state.pot == bb(4.5)


def test_preflop_rejects_check_before_big_blind_option():
    state = create_8max_preflop()
    assert not is_legal_action(state, Action(ActionType.CHECK))


def test_different_effective_stacks_support_short_all_in_call():
    state = create_8max_preflop(stacks_bb={Position.UTG: 20})
    state = apply_action(state, Action(ActionType.ALL_IN))  # UTG opens for 20 BB
    assert state.player(Position.UTG).all_in
    state = apply_action(state, Action(ActionType.RAISE, bb(40)))  # UTG+1 isolates to 40 BB
    # UTG is already all-in; later players can still participate in a side pot.
    assert state.current_bet == bb(40)
    assert state.player(Position.UTG).committed_total == bb(20)


def _limped_eight_way_flop():
    state = create_8max_preflop()
    for _ in range(7):
        state = apply_action(state, Action(ActionType.CALL))
    state = apply_action(state, Action(ActionType.CHECK))
    return advance_preflop_to_flop(state, ("As", "Kd", "Qh"))


def test_multiway_flop_full_raise_requires_every_remaining_player_to_respond():
    state = _limped_eight_way_flop()
    state = apply_multiway_postflop_action(state, Action(ActionType.BET, bb(4)))
    assert state.current_player is Position.BB
    state = apply_multiway_postflop_action(state, Action(ActionType.RAISE, bb(12)))
    assert state.current_bet == bb(12)
    assert state.last_full_raise_size == bb(8)
    assert state.current_player is Position.UTG
    assert Position.BB not in state.pending_players

    # UTG through BTN all fold; SB must still respond to BB's raise.
    for _ in range(6):
        state = apply_multiway_postflop_action(state, Action(ActionType.FOLD))
    assert state.current_player is Position.SB
    state = apply_multiway_postflop_action(state, Action(ActionType.CALL))
    assert state.betting_complete
    assert state.current_player is None
    assert state.pot == bb(32)


def test_multiway_postflop_checks_advance_to_turn_and_river_terminal():
    state = _limped_eight_way_flop()
    for _ in range(8):
        assert is_legal_multiway_postflop_action(state, Action(ActionType.CHECK))
        state = apply_multiway_postflop_action(state, Action(ActionType.CHECK))
    assert state.betting_complete
    turn = advance_multiway_postflop_street(state, "2c")
    assert turn.street == "turn"
    assert turn.current_player is Position.SB
    for _ in range(8):
        turn = apply_multiway_postflop_action(turn, Action(ActionType.CHECK))
    river = advance_multiway_postflop_street(turn, "3c")
    for _ in range(8):
        river = apply_multiway_postflop_action(river, Action(ActionType.CHECK))
    assert is_multiway_postflop_terminal(river)


def test_short_all_in_raise_does_not_reopen_raise_for_player_who_already_called():
    preflop = create_8max_preflop(stacks_bb={Position.BB: 16})
    for _ in range(7):
        preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    state = advance_preflop_to_flop(preflop, ("As", "Kd", "Qh"))
    state = apply_multiway_postflop_action(state, Action(ActionType.BET, bb(10)))
    state = apply_multiway_postflop_action(state, Action(ActionType.ALL_IN))
    assert state.current_bet == bb(15)
    for _ in range(6):
        state = apply_multiway_postflop_action(state, Action(ActionType.FOLD))
    assert state.current_player is Position.SB
    assert not is_legal_multiway_postflop_action(state, Action(ActionType.RAISE, bb(30)))
    assert is_legal_multiway_postflop_action(state, Action(ActionType.CALL))


def test_multiway_postflop_fold_win_settles_without_hole_cards():
    state = _limped_eight_way_flop()
    state = apply_multiway_postflop_action(state, Action(ActionType.BET, bb(2)))
    while not state.hand_ended:
        state = apply_multiway_postflop_action(state, Action(ActionType.FOLD))
    payouts, utility = settle_multiway_postflop(state, {})
    assert payouts == {Position.SB: bb(10)}
    assert sum(utility.values()) == 0
    assert utility[Position.SB] == bb(7)


def test_all_in_players_automatically_run_out_remaining_streets():
    preflop = create_8max_preflop(stacks_bb={Position.SB: 16, Position.BB: 16})
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    flop = advance_preflop_to_flop(preflop, ("As", "Kd", "Qh"))
    flop = apply_multiway_postflop_action(flop, Action(ActionType.ALL_IN))
    flop = apply_multiway_postflop_action(flop, Action(ActionType.ALL_IN))
    assert flop.betting_complete
    turn = advance_multiway_postflop_street(flop, "2c")
    assert turn.betting_complete
    river = advance_multiway_postflop_street(turn, "3c")
    assert is_multiway_postflop_terminal(river)


def test_preflop_all_in_players_enter_a_completed_flop():
    preflop = create_8max_preflop(stacks_bb={Position.SB: 16, Position.BB: 16})
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.ALL_IN))
    preflop = apply_action(preflop, Action(ActionType.CALL))

    flop = advance_preflop_to_flop(preflop, ("As", "Kd", "Qh"))

    assert flop.current_player is None
    assert flop.betting_complete


def test_last_eligible_player_all_in_raise_finishes_the_street():
    preflop = create_8max_preflop(stacks_bb={Position.SB: 16, Position.BB: 20})
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    flop = advance_preflop_to_flop(preflop, ("As", "Kd", "Qh"))
    flop = apply_multiway_postflop_action(flop, Action(ActionType.ALL_IN))
    flop = apply_multiway_postflop_action(flop, Action(ActionType.ALL_IN))
    assert flop.current_player is None
    assert flop.betting_complete
