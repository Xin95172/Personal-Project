"""Solver-oriented, river-only heads-up no-limit Hold'em game engine.

The engine supports every legal *integer-chip* bet or raise.  A solver should
later select a smaller subset through an action-abstraction policy; the poker
rules themselves are intentionally not restricted to preset pot percentages.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations
from math import ceil
from typing import Optional, Sequence

from poker_solver.engine.money import bb_to_units


class Player(Enum):
    OOP = "oop"
    IP = "ip"

    @property
    def opponent(self) -> "Player":
        return Player.IP if self is Player.OOP else Player.OOP


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class Action:
    """A concrete action.

    ``amount`` for BET and RAISE is the player's *total commitment in units*
    on this street after the action, not the number of units added by action.
    """

    kind: ActionType
    amount: Optional[int] = None

    def __post_init__(self) -> None:
        needs_amount = self.kind in {ActionType.BET, ActionType.RAISE}
        if needs_amount != (self.amount is not None):
            raise ValueError("only BET and RAISE actions require an amount")


@dataclass(frozen=True)
class SizingPolicy:
    """Solver 使用的底池比例動作抽象設定。"""

    bet_sizes: tuple[float, ...] = (0.33, 0.50, 0.75, 1.0, 1.5, 2.0)
    raise_sizes: tuple[float, ...] = (0.33, 0.50, 0.75, 1.0, 1.5, 2.0)
    include_all_in: bool = True
    max_re_raises: int | None = 2

    def __post_init__(self) -> None:
        if self.max_re_raises is not None and self.max_re_raises < 0:
            raise ValueError("max_re_raises must be non-negative or None")


@dataclass(frozen=True)
class PlayerState:
    player: Player
    hole_cards: tuple[str, str]
    stack: int
    committed_this_street: int = 0
    committed_total: int = 0
    folded: bool = False
    all_in: bool = False


@dataclass(frozen=True)
class RiverGameState:
    """Immutable state for one fixed-board, heads-up river subgame."""

    board: tuple[str, str, str, str, str]
    pot: int
    oop: PlayerState
    ip: PlayerState
    current_player: Optional[Player] = Player.OOP
    current_bet: int = 0
    last_full_raise_size: int = 1
    checks_since_last_bet: int = 0
    action_history: tuple[Action, ...] = field(default_factory=tuple)

    def player_state(self, player: Player) -> PlayerState:
        return self.oop if player is Player.OOP else self.ip

    def with_player(self, player: Player, new_state: PlayerState) -> "RiverGameState":
        return replace(self, oop=new_state) if player is Player.OOP else replace(self, ip=new_state)


def create_river_game(
    board: Sequence[str],
    oop_hole_cards: Sequence[str],
    ip_hole_cards: Sequence[str],
    *,
    initial_pot_bb: int | float | str = 10,
    effective_stack_bb: int | float | str = 95,
) -> RiverGameState:
    """Create a valid default river state with OOP to act.

    Cards use the compact notation ``As`` / ``Td`` / ``2c``. Amounts are
    supplied in BB and converted to internal 0.01-BB units.
    """
    if len(board) != 5 or len(oop_hole_cards) != 2 or len(ip_hole_cards) != 2:
        raise ValueError("river requires five board cards and two hole cards per player")
    cards = tuple(board) + tuple(oop_hole_cards) + tuple(ip_hole_cards)
    if len(set(cards)) != len(cards):
        raise ValueError("board and hole cards must not overlap")
    for card in cards:
        _parse_card(card)
    initial_pot = bb_to_units(initial_pot_bb)
    effective_stack = bb_to_units(effective_stack_bb)
    if initial_pot <= 0 or initial_pot % 2:
        raise ValueError("initial_pot_bb must be positive and split evenly between players")
    if effective_stack <= 0:
        raise ValueError("effective_stack_bb must be positive")
    prior_commitment = initial_pot // 2

    state = RiverGameState(
        board=tuple(board),  # type: ignore[arg-type]
        pot=initial_pot,
        oop=PlayerState(Player.OOP, tuple(oop_hole_cards), effective_stack, committed_total=prior_commitment),  # type: ignore[arg-type]
        ip=PlayerState(Player.IP, tuple(ip_hole_cards), effective_stack, committed_total=prior_commitment),  # type: ignore[arg-type]
    )
    _validate_state(state)
    return state


def is_terminal(state: RiverGameState) -> bool:
    return state.current_player is None


def current_player(state: RiverGameState) -> Optional[Player]:
    return state.current_player


def call_amount(state: RiverGameState, player: Player) -> int:
    return state.current_bet - state.player_state(player).committed_this_street


def is_legal_action(state: RiverGameState, action: Action) -> bool:
    """Return whether one concrete action follows the poker rules.

    This function does not enumerate the action space.  It is the rule-level
    boundary used by both external input and solver-generated candidates.
    """
    if is_terminal(state):
        return False
    assert state.current_player is not None
    actor = state.player_state(state.current_player)
    if actor.folded or actor.all_in:
        return False

    to_call = call_amount(state, actor.player)
    if to_call < 0:
        raise ValueError("current bet cannot be lower than the actor's commitment")

    if action.kind is ActionType.FOLD:
        return to_call > 0
    if action.kind is ActionType.CHECK:
        return to_call == 0
    if action.kind is ActionType.CALL:
        return 0 < to_call <= actor.stack
    if action.kind is ActionType.BET:
        return to_call == 0 and action.amount is not None and 1 <= action.amount < actor.stack
    if action.kind is ActionType.RAISE:
        if to_call == 0 or action.amount is None:
            return False
        minimum_target = state.current_bet + state.last_full_raise_size
        maximum_non_all_in_target = actor.committed_this_street + actor.stack - 1
        return minimum_target <= action.amount <= maximum_non_all_in_target
    if action.kind is ActionType.ALL_IN:
        target = actor.committed_this_street + actor.stack
        # When all-in only matches an existing bet, CALL is the sole semantic
        # action; all-in must be an opening bet or a raise.
        return to_call == 0 or target > state.current_bet
    raise ValueError(f"unknown action type: {action.kind}")


def abstract_actions(state: RiverGameState, policy: SizingPolicy) -> tuple[Action, ...]:
    """Return a small, policy-selected action set for a solver traversal."""
    if is_terminal(state):
        return ()
    assert state.current_player is not None
    actor = state.player_state(state.current_player)
    to_call = call_amount(state, actor.player)
    candidates: list[Action]

    if to_call == 0:
        candidates = [Action(ActionType.CHECK)]
        candidates.extend(Action(ActionType.BET, _pot_fraction_to_chips(state.pot, size)) for size in policy.bet_sizes)
    else:
        candidates = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        pot_after_call = state.pot + to_call
        raise_count = sum(action.kind is ActionType.RAISE for action in state.action_history)
        if policy.max_re_raises is None or raise_count < policy.max_re_raises:
            candidates.extend(
                Action(ActionType.RAISE, state.current_bet + _pot_fraction_to_chips(pot_after_call, size))
                for size in policy.raise_sizes
            )
    if policy.include_all_in:
        candidates.append(Action(ActionType.ALL_IN))

    # Multiple percentages may round to the same integer amount.  Preserve
    # order while emitting each legal concrete action once.
    return tuple(dict.fromkeys(action for action in candidates if is_legal_action(state, action)))


def all_legal_actions(state: RiverGameState) -> tuple[Action, ...]:
    """Enumerate every integer-chip action; use only for diagnostics/tests."""
    if is_terminal(state):
        return ()
    assert state.current_player is not None
    actor = state.player_state(state.current_player)
    to_call = call_amount(state, actor.player)
    if to_call == 0:
        candidates = [Action(ActionType.CHECK)]
        candidates.extend(Action(ActionType.BET, amount) for amount in range(1, actor.stack))
    else:
        candidates = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        minimum_target = state.current_bet + state.last_full_raise_size
        maximum_target = actor.committed_this_street + actor.stack
        candidates.extend(Action(ActionType.RAISE, target) for target in range(minimum_target, maximum_target))
    candidates.append(Action(ActionType.ALL_IN))
    return tuple(dict.fromkeys(action for action in candidates if is_legal_action(state, action)))


def format_action_as_pot_ratio(state: RiverGameState, action: Action) -> str:
    """將動作格式化為供策略輸出使用的底池比例文字。"""
    if action.kind is ActionType.FOLD:
        return "fold"
    if action.kind is ActionType.CHECK:
        return "check"
    if action.kind is ActionType.CALL:
        return "call"

    assert state.current_player is not None
    actor = state.player_state(state.current_player)
    to_call = call_amount(state, actor.player)
    if action.kind is ActionType.BET:
        assert action.amount is not None
        return f"bet {100 * action.amount / state.pot:g}% pot"
    if action.kind is ActionType.RAISE:
        assert action.amount is not None
        pot_after_call = state.pot + to_call
        raise_increment = action.amount - state.current_bet
        return f"raise {100 * raise_increment / pot_after_call:g}% pot-after-call"
    if action.kind is ActionType.ALL_IN:
        target = actor.committed_this_street + actor.stack
        if to_call == 0:
            return f"all-in ({100 * target / state.pot:g}% pot)"
        pot_after_call = state.pot + to_call
        raise_increment = target - state.current_bet
        return f"all-in (raise {100 * raise_increment / pot_after_call:g}% pot-after-call)"
    raise ValueError(f"unknown action type: {action.kind}")


def _pot_fraction_to_chips(pot: int, fraction: float) -> int:
    if fraction <= 0:
        raise ValueError("pot fractions must be positive")
    return ceil(pot * fraction)


def apply_action(state: RiverGameState, action: Action) -> RiverGameState:
    """Apply one legal action and return a new immutable state."""
    if not is_legal_action(state, action):
        raise ValueError(f"illegal action: {action}")
    assert state.current_player is not None
    actor_id = state.current_player
    opponent_id = actor_id.opponent
    actor = state.player_state(actor_id)
    to_call = call_amount(state, actor_id)
    history = state.action_history + (action,)

    if action.kind is ActionType.FOLD:
        updated = replace(actor, folded=True)
        result = state.with_player(actor_id, updated)
        result = replace(result, current_player=None, action_history=history)
        _validate_state(result)
        return result

    if action.kind is ActionType.CHECK:
        if state.checks_since_last_bet == 1:
            result = replace(state, current_player=None, checks_since_last_bet=2, action_history=history)
        else:
            result = replace(
                state,
                current_player=opponent_id,
                checks_since_last_bet=state.checks_since_last_bet + 1,
                action_history=history,
            )
        _validate_state(result)
        return result

    if action.kind is ActionType.CALL:
        # Equal effective stacks are an MVP invariant, so a call always fully matches.
        result = _commit(state, actor_id, to_call, history)
        result = replace(result, current_player=None)
        _validate_state(result)
        return result

    if action.kind is ActionType.ALL_IN:
        target = actor.committed_this_street + actor.stack
    else:
        assert action.amount is not None
        target = action.amount

    amount_to_add = target - actor.committed_this_street
    result = _commit(state, actor_id, amount_to_add, history)
    new_current_bet = max(state.current_bet, target)
    raise_size = new_current_bet - state.current_bet
    result = replace(
        result,
        current_bet=new_current_bet,
        last_full_raise_size=raise_size if raise_size >= state.last_full_raise_size else state.last_full_raise_size,
        checks_since_last_bet=0,
        current_player=opponent_id,
    )
    _validate_state(result)
    return result


def terminal_utility(state: RiverGameState) -> tuple[int, int]:
    """Return zero-sum (OOP, IP) net chip utility at a terminal state."""
    if not is_terminal(state):
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
    if state.pot % 2:
        raise ValueError("this MVP requires an even pot for split pots")
    return state.pot // 2 - state.oop.committed_total, state.pot // 2 - state.ip.committed_total


def infoset_key(state: RiverGameState, player: Player) -> tuple[object, ...]:
    """Return information observable by ``player``; it excludes opponent cards."""
    own = state.player_state(player)
    return (
        player.value,
        own.hole_cards,
        state.board,
        state.pot,
        state.current_bet,
        state.last_full_raise_size,
        tuple((action.kind.value, action.amount) for action in state.action_history),
    )


def _commit(state: RiverGameState, player: Player, amount: int, history: tuple[Action, ...]) -> RiverGameState:
    actor = state.player_state(player)
    if amount < 0 or amount > actor.stack:
        raise ValueError("commitment exceeds stack")
    updated = replace(
        actor,
        stack=actor.stack - amount,
        committed_this_street=actor.committed_this_street + amount,
        committed_total=actor.committed_total + amount,
        all_in=(actor.stack == amount),
    )
    result = state.with_player(player, updated)
    return replace(result, pot=state.pot + amount, action_history=history)


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _parse_card(card: str) -> tuple[int, str]:
    if len(card) != 2 or card[0] not in _RANKS or card[1] not in _SUITS:
        raise ValueError(f"invalid card: {card!r}; use notation such as 'As' or 'Td'")
    return _RANKS.index(card[0]) + 2, card[1]


def evaluate_seven_cards(cards: Sequence[str]) -> tuple[int, ...]:
    """Evaluate seven cards; larger tuples represent stronger poker hands."""
    if len(cards) != 7 or len(set(cards)) != 7:
        raise ValueError("exactly seven distinct cards are required")
    parsed = tuple(_parse_card(card) for card in cards)
    return max(_evaluate_five_cards(combo) for combo in combinations(parsed, 5))


def _evaluate_five_cards(cards: Sequence[tuple[int, str]]) -> tuple[int, ...]:
    ranks = sorted((rank for rank, _ in cards), reverse=True)
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({suit for _, suit in cards}) == 1
    unique_desc = sorted(set(ranks), reverse=True)
    straight_high = 0
    if len(unique_desc) == 5:
        if unique_desc == [14, 5, 4, 3, 2]:
            straight_high = 5
        elif unique_desc[0] - unique_desc[4] == 4:
            straight_high = unique_desc[0]

    if flush and straight_high:
        return 8, straight_high
    if groups[0][0] == 4:
        quad = groups[0][1]
        return 7, quad, next(rank for rank in ranks if rank != quad)
    if groups[0][0] == 3 and groups[1][0] == 2:
        return 6, groups[0][1], groups[1][1]
    if flush:
        return 5, *ranks
    if straight_high:
        return 4, straight_high
    if groups[0][0] == 3:
        trip = groups[0][1]
        return 3, trip, *(rank for rank in ranks if rank != trip)
    if groups[0][0] == 2 and groups[1][0] == 2:
        high_pair, low_pair = sorted((groups[0][1], groups[1][1]), reverse=True)
        kicker = next(rank for rank in ranks if rank not in {high_pair, low_pair})
        return 2, high_pair, low_pair, kicker
    if groups[0][0] == 2:
        pair = groups[0][1]
        return 1, pair, *(rank for rank in ranks if rank != pair)
    return 0, *ranks


def _validate_state(state: RiverGameState) -> None:
    if state.pot != state.oop.committed_total + state.ip.committed_total:
        raise ValueError("pot must equal both players' total commitments")
    if any(value < 0 for value in (state.pot, state.current_bet, state.last_full_raise_size)):
        raise ValueError("chip values must be non-negative")
    if state.current_bet != max(state.oop.committed_this_street, state.ip.committed_this_street):
        raise ValueError("current_bet must equal the highest street commitment")
