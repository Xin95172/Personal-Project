"""8-Max preflop 的可配置 action abstraction。"""

from dataclasses import dataclass
from math import ceil

from poker_solver.engine.money import bb_to_units
from poker_solver.engine.money import format_bb
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import PreflopState, call_amount, is_legal_action


@dataclass(frozen=True)
class PreflopSizingPolicy:
    open_sizes_bb: tuple[float, ...] = (2.0, 2.5, 3.0)
    re_raise_multipliers: tuple[float, ...] = (2.5, 3.5)
    include_all_in: bool = True
    max_raises: int | None = 3

    def __post_init__(self) -> None:
        if any(size <= 0 for size in (*self.open_sizes_bb, *self.re_raise_multipliers)):
            raise ValueError("preflop sizing values must be positive")
        if self.max_raises is not None and self.max_raises < 0:
            raise ValueError("max_raises must be non-negative or None")


def abstract_actions(state: PreflopState, policy: PreflopSizingPolicy) -> tuple[Action, ...]:
    """回傳 solver 應展開的 preflop 候選動作，而非所有合法金額。"""
    if state.current_player is None:
        return ()
    actor = state.player(state.current_player)
    to_call = call_amount(state, actor.position)
    raise_count = sum(action.kind is ActionType.RAISE for action in state.action_history)
    candidates: list[Action]

    if to_call == 0:
        candidates = [Action(ActionType.CHECK)]
    else:
        candidates = [Action(ActionType.FOLD), Action(ActionType.CALL)]

    if policy.max_raises is None or raise_count < policy.max_raises:
        if state.current_bet == bb_to_units(1):
            candidates.extend(Action(ActionType.RAISE, bb_to_units(size)) for size in policy.open_sizes_bb)
        else:
            candidates.extend(Action(ActionType.RAISE, ceil(state.current_bet * multiple)) for multiple in policy.re_raise_multipliers)
    if policy.include_all_in:
        candidates.append(Action(ActionType.ALL_IN))
    return tuple(dict.fromkeys(action for action in candidates if is_legal_action(state, action)))


def format_action(state: PreflopState, action: Action) -> str:
    """將 preflop 動作格式化為 BB 表示。"""
    if action.kind in {ActionType.FOLD, ActionType.CALL, ActionType.CHECK, ActionType.ALL_IN}:
        return action.kind.value.replace("_", "-")
    assert action.amount is not None
    return f"raise to {format_bb(action.amount)}"
