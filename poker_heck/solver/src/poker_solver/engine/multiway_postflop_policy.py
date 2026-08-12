"""多人 postflop solver 的動作抽象化。

規則引擎不限制下注額；本模組只為訓練時選取少量、以底池比例表示的候選動作。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import MultiwayPostflopState, is_legal_multiway_postflop_action


@dataclass(frozen=True)
class MultiwayPostflopSizingPolicy:
    bet_sizes: tuple[float, ...] = (0.33, 0.50, 0.75, 1.0, 1.5, 2.0)
    raise_sizes: tuple[float, ...] = (0.33, 0.50, 0.75, 1.0, 1.5, 2.0)
    include_all_in: bool = True
    max_re_raises: int | None = 2

    def __post_init__(self) -> None:
        if any(size <= 0 for size in (*self.bet_sizes, *self.raise_sizes)):
            raise ValueError("下注與加注比例必須為正數")
        if self.max_re_raises is not None and self.max_re_raises < 0:
            raise ValueError("max_re_raises 必須是非負整數或 None")


def abstract_multiway_postflop_actions(
    state: MultiwayPostflopState, policy: MultiwayPostflopSizingPolicy
) -> tuple[Action, ...]:
    """回傳目前節點的抽象動作，所有比例均以當下底池計算。"""
    if state.current_player is None or state.betting_complete or state.hand_ended:
        return ()
    actor = state.player(state.current_player)
    to_call = state.call_amount(actor.position)
    if to_call == 0:
        candidates = [Action(ActionType.CHECK)]
        candidates.extend(Action(ActionType.BET, ceil(state.pot * size)) for size in policy.bet_sizes)
    else:
        candidates = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        raise_count = sum(action.kind is ActionType.RAISE for action in state.action_history)
        if policy.max_re_raises is None or raise_count < policy.max_re_raises:
            pot_after_call = state.pot + to_call
            candidates.extend(
                Action(ActionType.RAISE, state.current_bet + ceil(pot_after_call * size))
                for size in policy.raise_sizes
            )
    if policy.include_all_in:
        candidates.append(Action(ActionType.ALL_IN))
    return tuple(dict.fromkeys(action for action in candidates if is_legal_multiway_postflop_action(state, action)))


def format_multiway_postflop_action_as_pot_ratio(state: MultiwayPostflopState, action: Action) -> str:
    """將抽象後的動作顯示成使用者閱讀的底池比例。"""
    if action.kind in {ActionType.FOLD, ActionType.CHECK, ActionType.CALL}:
        return action.kind.value
    if state.current_player is None:
        raise ValueError("terminal state has no acting player")
    actor = state.player(state.current_player)
    to_call = state.call_amount(actor.position)
    if action.kind is ActionType.BET:
        assert action.amount is not None
        return f"下注 {100 * action.amount / state.pot:g}% pot"
    if action.kind is ActionType.RAISE:
        assert action.amount is not None
        return f"加注 {100 * (action.amount - state.current_bet) / (state.pot + to_call):g}% pot-after-call"
    if action.kind is ActionType.ALL_IN:
        target = actor.committed_this_street + actor.stack
        basis = state.pot if to_call == 0 else state.pot + to_call
        label = "全下下注" if to_call == 0 else "全下加注"
        amount = target if to_call == 0 else target - state.current_bet
        return f"{label} {100 * amount / basis:g}% pot"
    raise ValueError(f"unsupported action: {action}")
