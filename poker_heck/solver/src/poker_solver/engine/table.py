"""8-Max preflop 的多人座位、盲注與下注輪狀態。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Optional

from poker_solver.engine.money import bb_to_units
from poker_solver.engine.river_game import Action, ActionType


class Position(Enum):
    UTG = "utg"
    UTG_1 = "utg+1"
    MP = "mp"
    HJ = "hj"
    CO = "co"
    BTN = "btn"
    SB = "sb"
    BB = "bb"


SEAT_ORDER = tuple(Position)


@dataclass(frozen=True)
class TablePlayer:
    position: Position
    stack: int
    committed_this_street: int = 0
    committed_total: int = 0
    folded: bool = False
    all_in: bool = False


@dataclass(frozen=True)
class PreflopState:
    players: tuple[TablePlayer, ...]
    pot: int
    current_bet: int
    last_full_raise_size: int
    current_player: Optional[Position]
    pending_players: frozenset[Position]
    action_history: tuple[Action, ...] = ()
    betting_complete: bool = False
    hand_ended: bool = False

    def player(self, position: Position) -> TablePlayer:
        return next(player for player in self.players if player.position is position)

    def with_player(self, position: Position, value: TablePlayer) -> "PreflopState":
        return replace(self, players=tuple(value if player.position is position else player for player in self.players))


@dataclass(frozen=True)
class MultiwayPostflopState:
    """多人 postflop 的街別起點；籌碼與總投入直接承接 preflop。"""

    board: tuple[str, ...]
    players: tuple[TablePlayer, ...]
    pot: int
    current_player: Optional[Position]
    pending_players: frozenset[Position]
    current_bet: int = 0
    last_full_raise_size: int = bb_to_units(1)
    street: str = "flop"
    action_history: tuple[Action, ...] = ()
    raise_allowed_players: frozenset[Position] = frozenset()
    betting_complete: bool = False
    hand_ended: bool = False

    def player(self, position: Position) -> TablePlayer:
        return next(player for player in self.players if player.position is position)

    def call_amount(self, position: Position) -> int:
        """指定座位在本街需要補齊的籌碼；全下或 fold 玩家由呼叫端排除。"""
        return self.current_bet - self.player(position).committed_this_street

    def with_player(self, position: Position, value: TablePlayer) -> "MultiwayPostflopState":
        return replace(self, players=tuple(value if player.position is position else player for player in self.players))

    @property
    def actable_positions(self) -> frozenset[Position]:
        return frozenset(player.position for player in self.players if not player.folded and not player.all_in)



def create_8max_preflop(
    *,
    stack_bb: int | float | str = 100,
    small_blind_bb: int | float | str = 0.5,
    big_blind_bb: int | float | str = 1,
    ante_bb: int | float | str = 0,
    stacks_bb: Mapping[Position, int | float | str] | None = None,
) -> PreflopState:
    stack = bb_to_units(stack_bb)
    sb = bb_to_units(small_blind_bb)
    bb = bb_to_units(big_blind_bb)
    ante = bb_to_units(ante_bb)
    if stack <= 0 or sb <= 0 or bb <= 0 or sb > bb:
        raise ValueError("stacks and blinds must be positive and small blind cannot exceed big blind")
    stacks = {position: bb_to_units(stacks_bb[position]) if stacks_bb and position in stacks_bb else stack for position in SEAT_ORDER}

    players = []
    for position in SEAT_ORDER:
        blind = sb if position is Position.SB else bb if position is Position.BB else 0
        if stacks[position] < blind + ante:
            raise ValueError(f"{position.value} stack is too small for blind and ante")
        committed = blind + ante
        players.append(
            TablePlayer(
                position=position,
                stack=stacks[position] - committed,
                committed_this_street=blind,
                committed_total=committed,
                all_in=(stacks[position] == committed),
            )
        )
    current_bet = bb
    pending = frozenset(position for position in SEAT_ORDER if not next(player for player in players if player.position is position).all_in)
    return PreflopState(
        players=tuple(players),
        pot=sum(player.committed_total for player in players),
        current_bet=current_bet,
        last_full_raise_size=bb,
        current_player=Position.UTG,
        pending_players=pending,
    )


def is_terminal(state: PreflopState) -> bool:
    return state.hand_ended or state.betting_complete


def advance_preflop_to_flop(state: PreflopState, board: tuple[str, str, str]) -> MultiwayPostflopState:
    """將正常結束的 preflop 帶入 flop，並由 SB 起依序行動。"""
    if not state.betting_complete or state.hand_ended:
        raise ValueError("only a completed non-fold preflop round can advance to flop")
    if len(board) != 3 or len(set(board)) != 3:
        raise ValueError("flop requires three distinct board cards")
    players = tuple(replace(player, committed_this_street=0) for player in state.players)
    pending = frozenset(player.position for player in players if not player.folded and not player.all_in)
    first = _next_position(Position.BTN, pending)
    return MultiwayPostflopState(
        board=board,
        players=players,
        pot=state.pot,
        current_player=first,
        pending_players=pending,
        raise_allowed_players=pending,
    )


def is_multiway_postflop_terminal(state: MultiwayPostflopState) -> bool:
    """此手是否已因棄牌結束，或已完成 river 下注。"""
    return state.hand_ended or (state.betting_complete and len(state.board) == 5)


def is_legal_multiway_postflop_action(state: MultiwayPostflopState, action: Action) -> bool:
    """檢查多人 postflop 的一個具體動作是否合法。"""
    if state.hand_ended or state.betting_complete or state.current_player is None:
        return False
    actor = state.player(state.current_player)
    if actor.folded or actor.all_in:
        return False
    to_call = state.call_amount(actor.position)
    if to_call < 0:
        raise ValueError("current bet cannot be lower than the actor's commitment")
    if action.kind is ActionType.FOLD:
        return to_call > 0
    if action.kind is ActionType.CHECK:
        return to_call == 0
    if action.kind is ActionType.CALL:
        return 0 < to_call <= actor.stack
    if action.kind is ActionType.BET:
        return to_call == 0 and action.amount is not None and bb_to_units(1) <= action.amount < actor.stack
    if action.kind is ActionType.RAISE:
        return (
            to_call > 0
            and actor.position in state.raise_allowed_players
            and action.amount is not None
            and state.current_bet + state.last_full_raise_size <= action.amount < actor.committed_this_street + actor.stack
        )
    if action.kind is ActionType.ALL_IN:
        target = actor.committed_this_street + actor.stack
        if actor.stack <= 0 or (actor.stack > to_call and target <= state.current_bet):
            return False
        return target <= state.current_bet or actor.position in state.raise_allowed_players
    return False


def apply_multiway_postflop_action(state: MultiwayPostflopState, action: Action) -> MultiwayPostflopState:
    """套用一個 postflop 動作，並決定下一位行動者或完成本街。"""
    if not is_legal_multiway_postflop_action(state, action):
        raise ValueError(f"illegal postflop action: {action}")
    assert state.current_player is not None
    actor_id = state.current_player
    actor = state.player(actor_id)
    history = state.action_history + (action,)
    to_call = state.call_amount(actor_id)

    if action.kind is ActionType.FOLD:
        result = state.with_player(actor_id, replace(actor, folded=True))
        return _advance_multiway_postflop_non_raise(result, actor_id, history)
    if action.kind is ActionType.CHECK:
        return _advance_multiway_postflop_non_raise(state, actor_id, history)
    if action.kind is ActionType.CALL:
        result = _commit_postflop(state, actor_id, to_call, history)
        return _advance_multiway_postflop_non_raise(result, actor_id, history)

    target = actor.committed_this_street + actor.stack if action.kind is ActionType.ALL_IN else action.amount
    assert target is not None
    result = _commit_postflop(state, actor_id, target - actor.committed_this_street, history)
    if target <= state.current_bet:  # short-stack all-in call
        return _advance_multiway_postflop_non_raise(result, actor_id, history)

    raise_size = target - state.current_bet
    full_raise = raise_size >= state.last_full_raise_size
    active = _actable_multiway_postflop_positions(result) - {actor_id}
    eligible = frozenset(active) if full_raise else state.raise_allowed_players & frozenset(active)
    return replace(
        result,
        current_bet=target,
        last_full_raise_size=raise_size if full_raise else state.last_full_raise_size,
        pending_players=frozenset(active),
        current_player=_next_position(actor_id, active),
        raise_allowed_players=eligible,
        action_history=history,
    )


def advance_multiway_postflop_street(state: MultiwayPostflopState, card: str) -> MultiwayPostflopState:
    """在已完成 flop/turn 下注後發一張 turn/river，並重設本街下注。"""
    if state.hand_ended or not state.betting_complete or len(state.board) not in {3, 4}:
        raise ValueError("only a completed flop or turn round can advance")
    if card in state.board:
        raise ValueError("board cards must be distinct")
    players = tuple(replace(player, committed_this_street=0) for player in state.players)
    pending = frozenset(player.position for player in players if not player.folded and not player.all_in)
    return MultiwayPostflopState(
        board=state.board + (card,),
        players=players,
        pot=state.pot,
        current_player=_next_position(Position.BTN, pending),
        pending_players=pending,
        street="turn" if len(state.board) == 3 else "river",
        raise_allowed_players=pending,
        betting_complete=not pending,
    )


def settle_multiway_postflop(
    state: MultiwayPostflopState, hole_cards: Mapping[Position, tuple[str, str]]
) -> tuple[dict[Position, int], dict[Position, int]]:
    """結算已結束的多人手牌，回傳 ``(payouts, utilities)``。

    翻牌／轉牌因所有人棄牌結束時不需要 hole cards；river 完成後則交由
    side-pot 與牌力比較模組處理，因而支援不同有效籌碼。
    """
    if not state.hand_ended and not (state.betting_complete and len(state.board) == 5):
        raise ValueError("only a finished hand can be settled")
    remaining = [player for player in state.players if not player.folded]
    if len(remaining) == 1:
        winner = remaining[0].position
        payouts = {winner: state.pot}
        utility = {player.position: payouts.get(player.position, 0) - player.committed_total for player in state.players}
        return payouts, utility
    from poker_solver.engine.showdown import settle_multiway_showdown

    return settle_multiway_showdown(state.players, state.board, hole_cards)


def call_amount(state: PreflopState, position: Position) -> int:
    return state.current_bet - state.player(position).committed_this_street


def is_legal_action(state: PreflopState, action: Action) -> bool:
    if is_terminal(state) or state.current_player is None:
        return False
    actor = state.player(state.current_player)
    if actor.folded or actor.all_in:
        return False
    to_call = call_amount(state, actor.position)
    if action.kind is ActionType.FOLD:
        return to_call > 0
    if action.kind is ActionType.CHECK:
        return to_call == 0
    if action.kind is ActionType.CALL:
        return 0 < to_call <= actor.stack
    if action.kind is ActionType.RAISE:
        return (
            to_call > 0
            and action.amount is not None
            and state.current_bet + state.last_full_raise_size <= action.amount < actor.committed_this_street + actor.stack
        )
    if action.kind is ActionType.ALL_IN:
        target = actor.committed_this_street + actor.stack
        return actor.stack > 0 and (actor.stack <= to_call or target > state.current_bet)
    return False


def apply_action(state: PreflopState, action: Action) -> PreflopState:
    if not is_legal_action(state, action):
        raise ValueError(f"illegal action: {action}")
    assert state.current_player is not None
    actor_id = state.current_player
    actor = state.player(actor_id)
    history = state.action_history + (action,)
    to_call = call_amount(state, actor_id)

    if action.kind is ActionType.FOLD:
        updated = replace(actor, folded=True)
        result = state.with_player(actor_id, updated)
        return _advance_after_non_raise(result, actor_id, history)
    if action.kind is ActionType.CHECK:
        return _advance_after_non_raise(state, actor_id, history)
    if action.kind is ActionType.CALL:
        return _advance_after_non_raise(_commit(state, actor_id, to_call, history), actor_id, history)

    target = actor.committed_this_street + actor.stack if action.kind is ActionType.ALL_IN else action.amount
    assert target is not None
    result = _commit(state, actor_id, target - actor.committed_this_street, history)
    is_raise = target > state.current_bet
    if not is_raise:
        return _advance_after_non_raise(result, actor_id, history)
    active = _actable_positions(result) - {actor_id}
    new_bet = target
    return replace(
        result,
        current_bet=new_bet,
        last_full_raise_size=max(state.last_full_raise_size, new_bet - state.current_bet),
        pending_players=frozenset(active),
        current_player=_next_position(actor_id, active),
        action_history=history,
    )


def _commit_postflop(
    state: MultiwayPostflopState, position: Position, amount: int, history: tuple[Action, ...]
) -> MultiwayPostflopState:
    actor = state.player(position)
    if amount < 0 or amount > actor.stack:
        raise ValueError("commitment exceeds stack")
    updated = replace(
        actor,
        stack=actor.stack - amount,
        committed_this_street=actor.committed_this_street + amount,
        committed_total=actor.committed_total + amount,
        all_in=(actor.stack == amount),
    )
    return replace(state.with_player(position, updated), pot=state.pot + amount, action_history=history)


def _advance_multiway_postflop_non_raise(
    state: MultiwayPostflopState, actor: Position, history: tuple[Action, ...]
) -> MultiwayPostflopState:
    active = _actable_multiway_postflop_positions(state)
    remaining = (state.pending_players - {actor}) & active
    not_folded = [player for player in state.players if not player.folded]
    if len(not_folded) <= 1:
        return replace(
            state,
            current_player=None,
            pending_players=frozenset(),
            raise_allowed_players=frozenset(),
            hand_ended=True,
            action_history=history,
        )
    if not remaining:
        return replace(
            state,
            current_player=None,
            pending_players=frozenset(),
            raise_allowed_players=frozenset(),
            betting_complete=True,
            action_history=history,
        )
    return replace(
        state,
        current_player=_next_position(actor, remaining),
        pending_players=frozenset(remaining),
        action_history=history,
    )


def _actable_multiway_postflop_positions(state: MultiwayPostflopState) -> set[Position]:
    return {player.position for player in state.players if not player.folded and not player.all_in}


def _commit(state: PreflopState, position: Position, amount: int, history: tuple[Action, ...]) -> PreflopState:
    actor = state.player(position)
    if amount < 0 or amount > actor.stack:
        raise ValueError("commitment exceeds stack")
    updated = replace(
        actor,
        stack=actor.stack - amount,
        committed_this_street=actor.committed_this_street + amount,
        committed_total=actor.committed_total + amount,
        all_in=(actor.stack == amount),
    )
    return replace(state.with_player(position, updated), pot=state.pot + amount, action_history=history)


def _advance_after_non_raise(state: PreflopState, actor: Position, history: tuple[Action, ...]) -> PreflopState:
    active = _actable_positions(state)
    remaining = (state.pending_players - {actor}) & active
    not_folded = [player for player in state.players if not player.folded]
    if len(not_folded) == 1:
        return replace(state, current_player=None, pending_players=frozenset(), hand_ended=True, action_history=history)
    if not remaining:
        return replace(state, current_player=None, pending_players=frozenset(), betting_complete=True, action_history=history)
    return replace(state, current_player=_next_position(actor, remaining), pending_players=frozenset(remaining), action_history=history)


def _actable_positions(state: PreflopState) -> set[Position]:
    return {player.position for player in state.players if not player.folded and not player.all_in}


def _next_position(after: Position, candidates: set[Position] | frozenset[Position]) -> Optional[Position]:
    start = SEAT_ORDER.index(after)
    for offset in range(1, len(SEAT_ORDER) + 1):
        position = SEAT_ORDER[(start + offset) % len(SEAT_ORDER)]
        if position in candidates:
            return position
    return None
