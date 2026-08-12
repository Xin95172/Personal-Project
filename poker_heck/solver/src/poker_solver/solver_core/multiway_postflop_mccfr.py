"""8-Max 多人 postflop 的 external-sampling MCCFR 訓練器。

此訓練器由一個已完成 preflop 的 flop/turn/river 狀態開始，依序抽樣未知
公共牌，並用 side-pot showdown 作為終局效用。這是抽象動作樹的近似解，不是
對無限下注額的精確均衡。
"""

from __future__ import annotations

from random import Random
from time import perf_counter
from typing import Mapping

from poker_solver.engine.chance import remaining_cards
from poker_solver.engine.multiway_postflop_policy import MultiwayPostflopSizingPolicy, abstract_multiway_postflop_actions
from poker_solver.engine.river_game import Action
from poker_solver.engine.table import (
    MultiwayPostflopState,
    Position,
    advance_multiway_postflop_street,
    apply_multiway_postflop_action,
    is_multiway_postflop_terminal,
    settle_multiway_postflop,
)
from poker_solver.solver_core.river_mccfr import InfoSet, TrainingStats, WeightedRange, _sample_action


class MultiwayPostflopMCCFRTrainer:
    """從指定多人 postflop 根節點訓練各位置的策略。"""

    def __init__(
        self,
        *,
        initial_state: MultiwayPostflopState,
        ranges: Mapping[Position, WeightedRange],
        sizing_policy: MultiwayPostflopSizingPolicy | None = None,
        seed: int = 0,
    ) -> None:
        if set(ranges) != set(Position):
            raise ValueError("ranges must provide every 8-Max position")
        if len(initial_state.board) not in {3, 4, 5}:
            raise ValueError("initial_state must be a flop, turn, or river state")
        self.initial_state = initial_state
        self.ranges = dict(ranges)
        self.sizing_policy = sizing_policy or MultiwayPostflopSizingPolicy()
        self.rng = Random(seed)
        self.infosets: dict[tuple[object, ...], InfoSet] = {}
        self.iterations_completed = 0

    def train(self, iterations: int) -> TrainingStats:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        started = perf_counter()
        for _ in range(iterations):
            holes = self._sample_deal()
            for traverser in Position:
                self._traverse(self.initial_state, holes, traverser, {position: 1.0 for position in Position})
            self.iterations_completed += 1
        elapsed = perf_counter() - started
        positive = [max(0.0, regret) for node in self.infosets.values() for regret in node.regret_sum.values()]
        return TrainingStats(iterations, self.iterations_completed, len(self.infosets), elapsed, sum(positive) / len(positive) if positive else 0.0)

    def strategy_for(self, state: MultiwayPostflopState, player: Position, hole_cards: tuple[str, str]) -> dict[Action, float]:
        key = _infoset_key(state, player, hole_cards)
        if key not in self.infosets:
            raise KeyError("this information set has not been visited during training")
        return self.infosets[key].average_strategy()

    def average_strategy_utility(self, state: MultiwayPostflopState, holes: dict[Position, tuple[str, str]]) -> dict[Position, int]:
        """以目前平均策略 rollout 一次，回傳所有位置的 continuation utility。"""
        if is_multiway_postflop_terminal(state):
            return settle_multiway_postflop(state, holes)[1]
        if state.betting_complete:
            card = self.rng.choice(remaining_cards((*state.board, *(card for combo in holes.values() for card in combo))))
            return self.average_strategy_utility(advance_multiway_postflop_street(state, card), holes)
        assert state.current_player is not None
        actions = abstract_multiway_postflop_actions(state, self.sizing_policy)
        key = _infoset_key(state, state.current_player, holes[state.current_player])
        strategy = self.infosets[key].average_strategy() if key in self.infosets else {action: 1.0 / len(actions) for action in actions}
        action = _sample_action(strategy, self.rng)
        return self.average_strategy_utility(apply_multiway_postflop_action(state, action), holes)

    def _sample_deal(self) -> dict[Position, tuple[str, str]]:
        board = self.initial_state.board
        for _ in range(10_000):
            holes = {position: self.ranges[position].sample(self.rng).cards for position in Position}
            cards = [*board, *(card for combo in holes.values() for card in combo)]
            if len(cards) == len(set(cards)):
                return holes
        raise ValueError("ranges cannot produce a non-overlapping 8-player deal")

    def _traverse(
        self,
        state: MultiwayPostflopState,
        holes: dict[Position, tuple[str, str]],
        traverser: Position,
        reach: dict[Position, float],
    ) -> float:
        if is_multiway_postflop_terminal(state):
            return float(settle_multiway_postflop(state, holes)[1][traverser])
        if state.betting_complete:
            card = self.rng.choice(remaining_cards((*state.board, *(card for combo in holes.values() for card in combo))))
            return self._traverse(advance_multiway_postflop_street(state, card), holes, traverser, reach)

        assert state.current_player is not None
        actor = state.current_player
        actions = abstract_multiway_postflop_actions(state, self.sizing_policy)
        if not actions:
            raise RuntimeError("a non-terminal state must have an abstract action")
        key = _infoset_key(state, actor, holes[actor])
        node = self.infosets.setdefault(key, InfoSet(actions))
        if node.actions != actions:
            raise RuntimeError("an information set produced inconsistent abstract actions")
        strategy = node.current_strategy()
        node.accumulate_average_strategy(strategy, reach[actor])
        if actor is traverser:
            values = {
                action: self._traverse(
                    apply_multiway_postflop_action(state, action), holes, traverser,
                    {**reach, actor: reach[actor] * strategy[action]},
                )
                for action in actions
            }
            node_value = sum(strategy[action] * values[action] for action in actions)
            for action in actions:
                node.regret_sum[action] += values[action] - node_value
            return node_value
        action = _sample_action(strategy, self.rng)
        return self._traverse(
            apply_multiway_postflop_action(state, action), holes, traverser,
            {**reach, actor: reach[actor] * strategy[action]},
        )


def _infoset_key(state: MultiwayPostflopState, player: Position, hole_cards: tuple[str, str]) -> tuple[object, ...]:
    return (
        player.value,
        hole_cards,
        state.street,
        state.board,
        state.pot,
        state.current_bet,
        state.last_full_raise_size,
        tuple((seat.position.value, seat.committed_this_street, seat.folded, seat.all_in) for seat in state.players),
        tuple((action.kind.value, action.amount) for action in state.action_history),
    )
