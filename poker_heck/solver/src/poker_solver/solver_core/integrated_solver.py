"""以 postflop 子局 EV 回饋 preflop 的整合式訓練。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from poker_solver.engine.chance import remaining_cards
from poker_solver.engine.multiway_postflop_policy import MultiwayPostflopSizingPolicy
from poker_solver.engine.table import Position, PreflopState, advance_preflop_to_flop
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.preflop_range_conditioning import condition_ranges_for_terminal
from poker_solver.solver_core.strategy_store import StrategyContext, StrategyStore, export_multiway_postflop_infosets
from poker_solver.engine.river_game import Action, ActionType


@dataclass(frozen=True)
class ContinuationSettings:
    subgame_iterations: int = 100
    value_rollouts: int = 4
    max_cached_subgames: int | None = None
    bet_sizes: tuple[float, ...] = (0.33, 0.5, 0.75, 1.0, 1.5, 2.0)
    raise_sizes: tuple[float, ...] = (0.33, 0.5, 0.75, 1.0, 1.5, 2.0)
    include_all_in: bool = True
    max_re_raises: int | None = 1
    strategy_db: str | None = None
    range_profile_id: str = "all_combos"
    solver_version: str = "multiway-postflop-grid-v1"


class PostflopContinuationOracle:
    def __init__(self, trainer: MultiwayPreflopMCCFRTrainer, settings: ContinuationSettings) -> None:
        if settings.subgame_iterations <= 0 or settings.value_rollouts <= 0:
            raise ValueError("continuation iterations and rollouts must be positive")
        self.trainer = trainer
        self.settings = settings
        self.cache: dict[tuple[object, ...], MultiwayPostflopMCCFRTrainer] = {}
        self.store = StrategyStore(Path(settings.strategy_db)) if settings.strategy_db else None

    def utility(self, state: PreflopState, holes: dict[Position, tuple[str, str]]) -> dict[Position, float]:
        known = tuple(card for cards in holes.values() for card in cards)
        flop = tuple(self.trainer.rng.sample(remaining_cards(known), 3))
        key = (tuple((action.kind.value, action.amount) for action in state.action_history), flop)
        solver = self.cache.get(key)
        if solver is None:
            if self.settings.max_cached_subgames is not None and len(self.cache) >= self.settings.max_cached_subgames:
                self.cache.clear()
            ranges = condition_ranges_for_terminal(self.trainer, state, excluded_cards=flop)
            solver = MultiwayPostflopMCCFRTrainer(
                initial_state=advance_preflop_to_flop(state, flop), ranges=ranges,
                sizing_policy=MultiwayPostflopSizingPolicy(self.settings.bet_sizes, self.settings.raise_sizes, self.settings.include_all_in, self.settings.max_re_raises),
                seed=self.trainer.rng.randrange(2**31),
            )
            root_has_strategy = (
                solver.initial_state.current_player is not None
                and self._database_policy(
                    solver.initial_state,
                    solver.initial_state.current_player,
                    holes[solver.initial_state.current_player],
                    max(player.stack for player in solver.initial_state.players),
                ) is not None
            )
            if not root_has_strategy:
                solver.train(self.settings.subgame_iterations)
            self.cache[key] = solver
            if self.store is not None and not root_has_strategy:
                export_multiway_postflop_infosets(self.store, solver, range_profile_id=self.settings.range_profile_id, solver_version=self.settings.solver_version)
        totals = {position: 0.0 for position in Position}
        effective_stack_units = max(player.stack for player in solver.initial_state.players)
        for _ in range(self.settings.value_rollouts):
            policy = lambda current, player, cards: self._database_policy(current, player, cards, effective_stack_units)
            for position, value in solver.average_strategy_utility(solver.initial_state, holes, policy).items():
                totals[position] += value / self.settings.value_rollouts
        return totals

    def _database_policy(self, state, position: Position, cards: tuple[str, str], effective_stack_units: int) -> dict[Action, float] | None:
        if self.store is None:
            return None
        context = StrategyContext("multiway_postflop", state.street, sum(not player.folded for player in state.players), position.value, tuple(sorted(cards)), state.board, state.pot, effective_stack_units, tuple((action.kind.value, action.amount) for action in state.action_history), self.settings.range_profile_id, self.settings.solver_version)
        stored = self.store.lookup(context)
        if stored is None:
            return None
        return {Action(ActionType(item.kind), item.amount_units): item.probability for item in stored.actions}


def attach_postflop_continuation(trainer: MultiwayPreflopMCCFRTrainer, settings: ContinuationSettings) -> PostflopContinuationOracle:
    oracle = PostflopContinuationOracle(trainer, settings)
    trainer.continuation_utility = oracle.utility
    return oracle
