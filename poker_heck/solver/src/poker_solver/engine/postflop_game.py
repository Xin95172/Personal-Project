"""Turn 與 river 共用的 heads-up postflop 遊戲狀態。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import ceil
from typing import Optional, Sequence

from poker_solver.engine.money import bb_to_units
from poker_solver.engine.river_game import Action, ActionType, Player, PlayerState, SizingPolicy, evaluate_seven_cards


class Street(Enum):
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


@dataclass(frozen=True)
class PostflopGameState:
    street: Street
    board: tuple[str, ...]
    pot: int
    oop: PlayerState
    ip: PlayerState
    current_player: Optional[Player] = Player.OOP
    current_bet: int = 0
    last_full_raise_size: int = 1
    checks_since_last_bet: int = 0
    raise_count_this_street: int = 0
    action_history: tuple[Action, ...] = field(default_factory=tuple)
    chance_pending: bool = False
    terminal: bool = False

    def player_state(self, player: Player) -> PlayerState:
        return self.oop if player is Player.OOP else self.ip

    def with_player(self, player: Player, value: PlayerState) -> "PostflopGameState":
        return replace(self, oop=value) if player is Player.OOP else replace(self, ip=value)


def create_turn_game(
    board: Sequence[str],
    oop_hole_cards: Sequence[str],
    ip_hole_cards: Sequence[str],
    *,
    initial_pot_bb: int | float | str = 10,
    effective_stack_bb: int | float | str = 95,
) -> PostflopGameState:
    return _create_game(Street.TURN, board, oop_hole_cards, ip_hole_cards, initial_pot_bb, effective_stack_bb)


def create_flop_game(
    board: Sequence[str],
    oop_hole_cards: Sequence[str],
    ip_hole_cards: Sequence[str],
    *,
    initial_pot_bb: int | float | str = 10,
    effective_stack_bb: int | float | str = 95,
) -> PostflopGameState:
    return _create_game(Street.FLOP, board, oop_hole_cards, ip_hole_cards, initial_pot_bb, effective_stack_bb)


def _create_game(
    street: Street,
    board: Sequence[str],
    oop_hole_cards: Sequence[str],
    ip_hole_cards: Sequence[str],
    initial_pot_bb: int | float | str,
    effective_stack_bb: int | float | str,
) -> PostflopGameState:
    expected_board_size = 3 if street is Street.FLOP else 4
    if len(board) != expected_board_size or len(oop_hole_cards) != 2 or len(ip_hole_cards) != 2:
        raise ValueError(f"{street.value} requires {expected_board_size} board cards and two hole cards per player")
    cards = tuple(board) + tuple(oop_hole_cards) + tuple(ip_hole_cards)
    if len(set(cards)) != len(cards) or any(not _valid_card(card) for card in cards):
        raise ValueError("cards must be valid and distinct")
    pot = bb_to_units(initial_pot_bb)
    stack = bb_to_units(effective_stack_bb)
    if pot <= 0 or pot % 2 or stack <= 0:
        raise ValueError("pot must be positive/even and effective stack must be positive")
    prior = pot // 2
    return PostflopGameState(
        street=street,
        board=tuple(board),
        pot=pot,
        oop=PlayerState(Player.OOP, tuple(oop_hole_cards), stack, committed_total=prior),  # type: ignore[arg-type]
        ip=PlayerState(Player.IP, tuple(ip_hole_cards), stack, committed_total=prior),  # type: ignore[arg-type]
    )


def is_terminal(state: PostflopGameState) -> bool:
    return state.terminal


def is_chance_node(state: PostflopGameState) -> bool:
    return state.chance_pending


def call_amount(state: PostflopGameState, player: Player) -> int:
    return state.current_bet - state.player_state(player).committed_this_street


def is_legal_action(state: PostflopGameState, action: Action) -> bool:
    if state.terminal or state.chance_pending or state.current_player is None:
        return False
    actor = state.player_state(state.current_player)
    if actor.folded or actor.all_in:
        return False
    to_call = call_amount(state, actor.player)
    if action.kind is ActionType.FOLD:
        return to_call > 0
    if action.kind is ActionType.CHECK:
        return to_call == 0
    if action.kind is ActionType.CALL:
        return 0 < to_call <= actor.stack
    if action.kind is ActionType.BET:
        return to_call == 0 and action.amount is not None and 1 <= action.amount < actor.stack
    if action.kind is ActionType.RAISE:
        return (
            to_call > 0
            and action.amount is not None
            and state.current_bet + state.last_full_raise_size <= action.amount < actor.committed_this_street + actor.stack
        )
    if action.kind is ActionType.ALL_IN:
        target = actor.committed_this_street + actor.stack
        return to_call == 0 or target > state.current_bet
    return False


def abstract_actions(state: PostflopGameState, policy: SizingPolicy) -> tuple[Action, ...]:
    if state.terminal or state.chance_pending:
        return ()
    assert state.current_player is not None
    actor = state.player_state(state.current_player)
    to_call = call_amount(state, actor.player)
    if to_call == 0:
        candidates = [Action(ActionType.CHECK)] + [Action(ActionType.BET, ceil(state.pot * size)) for size in policy.bet_sizes]
    else:
        candidates = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        if policy.max_re_raises is None or state.raise_count_this_street < policy.max_re_raises:
            candidates.extend(Action(ActionType.RAISE, state.current_bet + ceil((state.pot + to_call) * size)) for size in policy.raise_sizes)
    if policy.include_all_in:
        candidates.append(Action(ActionType.ALL_IN))
    return tuple(dict.fromkeys(action for action in candidates if is_legal_action(state, action)))


def apply_action(state: PostflopGameState, action: Action) -> PostflopGameState:
    if not is_legal_action(state, action):
        raise ValueError(f"illegal action: {action}")
    assert state.current_player is not None
    actor_id = state.current_player
    opponent_id = actor_id.opponent
    actor = state.player_state(actor_id)
    history = state.action_history + (action,)
    to_call = call_amount(state, actor_id)

    if action.kind is ActionType.FOLD:
        return replace(state.with_player(actor_id, replace(actor, folded=True)), current_player=None, terminal=True, action_history=history)
    if action.kind is ActionType.CHECK:
        if state.checks_since_last_bet == 1:
            return _close_betting_round(replace(state, checks_since_last_bet=2, action_history=history))
        return replace(state, current_player=opponent_id, checks_since_last_bet=state.checks_since_last_bet + 1, action_history=history)
    if action.kind is ActionType.CALL:
        return _close_betting_round(_commit(state, actor_id, to_call, history))

    target = actor.committed_this_street + actor.stack if action.kind is ActionType.ALL_IN else action.amount
    assert target is not None
    result = _commit(state, actor_id, target - actor.committed_this_street, history)
    new_bet = max(state.current_bet, target)
    raise_size = new_bet - state.current_bet
    return replace(
        result,
        current_bet=new_bet,
        last_full_raise_size=raise_size if raise_size >= state.last_full_raise_size else state.last_full_raise_size,
        checks_since_last_bet=0,
        raise_count_this_street=state.raise_count_this_street + (action.kind is ActionType.RAISE),
        current_player=opponent_id,
    )


def advance_to_next_street(state: PostflopGameState, card: str) -> PostflopGameState:
    if not state.chance_pending or state.street is Street.RIVER:
        raise ValueError("state is not waiting for a public card")
    known = (*state.board, *state.oop.hole_cards, *state.ip.hole_cards)
    if card in known or not _valid_card(card):
        raise ValueError("public card must be valid and unavailable")
    oop = replace(state.oop, committed_this_street=0)
    ip = replace(state.ip, committed_this_street=0)
    next_street = Street.TURN if state.street is Street.FLOP else Street.RIVER
    all_in = oop.all_in and ip.all_in
    all_in_showdown = all_in and next_street is Street.RIVER
    return replace(
        state,
        street=next_street,
        board=(*state.board, card),
        oop=oop,
        ip=ip,
        current_player=None if all_in else Player.OOP,
        current_bet=0,
        last_full_raise_size=1,
        checks_since_last_bet=0,
        raise_count_this_street=0,
        chance_pending=all_in and not all_in_showdown,
        terminal=all_in_showdown,
    )


def advance_to_river(state: PostflopGameState, river_card: str) -> PostflopGameState:
    if state.street is not Street.TURN:
        raise ValueError("advance_to_river requires a turn chance node")
    return advance_to_next_street(state, river_card)


def terminal_utility(state: PostflopGameState) -> tuple[int, int]:
    if not state.terminal:
        raise ValueError("utility is only defined at terminal states")
    if state.oop.folded:
        return -state.oop.committed_total, state.pot - state.ip.committed_total
    if state.ip.folded:
        return state.pot - state.oop.committed_total, -state.ip.committed_total
    oop_rank = evaluate_seven_cards((*state.oop.hole_cards, *state.board))
    ip_rank = evaluate_seven_cards((*state.ip.hole_cards, *state.board))
    if oop_rank > ip_rank:
        return state.pot - state.oop.committed_total, -state.ip.committed_total
    if ip_rank > oop_rank:
        return -state.oop.committed_total, state.pot - state.ip.committed_total
    return state.pot // 2 - state.oop.committed_total, state.pot // 2 - state.ip.committed_total


def infoset_key(state: PostflopGameState, player: Player) -> tuple[object, ...]:
    return (
        state.street.value,
        player.value,
        state.player_state(player).hole_cards,
        state.board,
        state.pot,
        state.current_bet,
        state.last_full_raise_size,
        tuple((action.kind.value, action.amount) for action in state.action_history),
    )


def _commit(state: PostflopGameState, player: Player, amount: int, history: tuple[Action, ...]) -> PostflopGameState:
    actor = state.player_state(player)
    updated = replace(
        actor,
        stack=actor.stack - amount,
        committed_this_street=actor.committed_this_street + amount,
        committed_total=actor.committed_total + amount,
        all_in=actor.stack == amount,
    )
    return replace(state.with_player(player, updated), pot=state.pot + amount, action_history=history)


def _close_betting_round(state: PostflopGameState) -> PostflopGameState:
    if state.street is Street.RIVER:
        return replace(state, current_player=None, terminal=True)
    return replace(state, current_player=None, chance_pending=True)


def _valid_card(card: str) -> bool:
    return len(card) == 2 and card[0] in "23456789TJQKA" and card[1] in "cdhs"
