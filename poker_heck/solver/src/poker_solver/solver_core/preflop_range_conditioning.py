"""依 preflop 平均策略，把初始 range 更新為進入 postflop 的後驗 range。"""

from __future__ import annotations

from poker_solver.engine.preflop_policy import abstract_actions
from poker_solver.engine.table import Position, PreflopState, apply_action, create_8max_preflop
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.river_mccfr import Combo, WeightedRange


def condition_ranges_for_terminal(
    trainer: MultiwayPreflopMCCFRTrainer,
    terminal_state: PreflopState,
    *,
    excluded_cards: tuple[str, ...] = (),
) -> dict[Position, WeightedRange]:
    """回傳在指定 preflop 行動歷史後、且排除公共牌的邊際後驗 ranges。"""
    result: dict[Position, WeightedRange] = {}
    blocked = set(excluded_cards)
    for position, initial_range in trainer.ranges.items():
        combos: list[Combo] = []
        for combo in initial_range.combos:
            if blocked.intersection(combo.cards):
                continue
            likelihood = _history_likelihood(trainer, terminal_state, position, combo.cards)
            if likelihood > 0:
                combos.append(Combo(combo.cards, combo.weight * likelihood))
        if not combos:
            # Counterfactual action branches may have zero average probability while
            # MCCFR is still evaluating their value.  Keep the prior range rather
            # than creating an invalid empty postflop subgame.
            combos = [Combo(combo.cards, combo.weight) for combo in initial_range.combos if not blocked.intersection(combo.cards)]
        if not combos:
            raise ValueError(f"postflop range for {position.value} is empty after card exclusion")
        result[position] = WeightedRange(tuple(combos))
    return result


def _history_likelihood(
    trainer: MultiwayPreflopMCCFRTrainer,
    terminal_state: PreflopState,
    position: Position,
    cards: tuple[str, str],
) -> float:
    state = create_8max_preflop(stack_bb=trainer.stack_bb)
    likelihood = 1.0
    for observed in terminal_state.action_history:
        assert state.current_player is not None
        actor = state.current_player
        if actor is position:
            actions = abstract_actions(state, trainer.sizing_policy)
            try:
                strategy = trainer.strategy_for(state, position, cards)
            except KeyError:
                strategy = {action: 1.0 / len(actions) for action in actions}
            likelihood *= strategy.get(observed, 0.0)
        state = apply_action(state, observed)
    return likelihood
