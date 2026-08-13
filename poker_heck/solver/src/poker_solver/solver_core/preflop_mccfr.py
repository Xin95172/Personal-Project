"""8-Max preflop 的 multiway External Sampling MCCFR 近似訓練器。"""

from random import Random
from time import perf_counter
from typing import Callable, Mapping

from poker_solver.engine.chance import remaining_cards
from poker_solver.engine.preflop_policy import PreflopSizingPolicy, abstract_actions
from poker_solver.engine.river_game import Action
from poker_solver.engine.showdown import settle_multiway_showdown
from poker_solver.engine.table import Position, PreflopState, apply_action, create_8max_preflop, is_terminal
from poker_solver.solver_core.river_mccfr import InfoSet, TrainingStats, WeightedRange, _sample_action


ContinuationUtility = Callable[[PreflopState, dict[Position, tuple[str, str]]], Mapping[Position, int | float]]


class MultiwayPreflopMCCFRTrainer:
    """以抽樣 showdown rollout 訓練 8-Max preflop 的近似策略 profile。"""

    def __init__(
        self,
        *,
        ranges: Mapping[Position, WeightedRange],
        sizing_policy: PreflopSizingPolicy | None = None,
        stack_bb: int | float | str = 100,
        seed: int = 0,
        continuation_utility: ContinuationUtility | None = None,
    ) -> None:
        if set(ranges) != set(Position):
            raise ValueError("ranges must provide every 8-Max position")
        self.ranges = dict(ranges)
        self.sizing_policy = sizing_policy or PreflopSizingPolicy()
        self.stack_bb = stack_bb
        self.rng = Random(seed)
        self.continuation_utility = continuation_utility
        self.infosets: dict[tuple[object, ...], InfoSet] = {}
        self.iterations_completed = 0

    def train(self, iterations: int) -> TrainingStats:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        started = perf_counter()
        for _ in range(iterations):
            state, holes = self._sample_deal()
            for traverser in Position:
                self._traverse(state, holes, traverser, {position: 1.0 for position in Position})
            self.iterations_completed += 1
        elapsed = perf_counter() - started
        positive = [max(0.0, regret) for node in self.infosets.values() for regret in node.regret_sum.values()]
        return TrainingStats(iterations, self.iterations_completed, len(self.infosets), elapsed, sum(positive) / len(positive) if positive else 0.0)

    def strategy_for(self, state: PreflopState, player: Position, hole_cards: tuple[str, str]) -> dict[Action, float]:
        key = _infoset_key(state, player, hole_cards)
        if key not in self.infosets:
            raise KeyError("this information set has not been visited during training")
        return self.infosets[key].average_strategy()

    def _sample_deal(self) -> tuple[PreflopState, dict[Position, tuple[str, str]]]:
        for _ in range(10_000):
            holes = {position: self.ranges[position].sample(self.rng).cards for position in Position}
            if len({card for cards in holes.values() for card in cards}) == 16:
                return create_8max_preflop(stack_bb=self.stack_bb), holes
        raise ValueError("ranges cannot produce a non-overlapping 8-player deal")

    def _traverse(
        self,
        state: PreflopState,
        holes: dict[Position, tuple[str, str]],
        traverser: Position,
        reach: dict[Position, float],
    ) -> float:
        if is_terminal(state):
            utility = self._terminal_utility(state, holes)
            return utility[traverser]

        assert state.current_player is not None
        actor = state.current_player
        actions = abstract_actions(state, self.sizing_policy)
        key = _infoset_key(state, actor, holes[actor])
        node = self.infosets.get(key)
        if node is None:
            node = InfoSet(actions)
            self.infosets[key] = node
        elif node.actions != actions:
            raise RuntimeError("preflop information set produced inconsistent actions")
        strategy = node.current_strategy()
        node.accumulate_average_strategy(strategy, reach[actor])

        if actor is traverser:
            values = {
                action: self._traverse(apply_action(state, action), holes, traverser, {**reach, actor: reach[actor] * strategy[action]})
                for action in actions
            }
            node_value = sum(strategy[action] * values[action] for action in actions)
            for action in actions:
                node.observe_action_value(action, values[action])
                node.regret_sum[action] += values[action] - node_value
            return node_value

        action = _sample_action(strategy, self.rng)
        return self._traverse(apply_action(state, action), holes, traverser, {**reach, actor: reach[actor] * strategy[action]})

    def _terminal_utility(self, state: PreflopState, holes: dict[Position, tuple[str, str]]) -> dict[Position, float]:
        active = [player for player in state.players if not player.folded]
        if len(active) == 1:
            winner = active[0].position
            return {player.position: (state.pot - player.committed_total if player.position is winner else -player.committed_total) for player in state.players}
        if self.continuation_utility is not None:
            utility = self.continuation_utility(state, holes)
            if set(utility) != set(Position):
                raise ValueError("continuation utility must provide every 8-Max position")
            return {position: float(utility[position]) for position in Position}
        known = [card for cards in holes.values() for card in cards]
        board = remaining_cards(known)
        sampled_board = tuple(board[index] for index in self.rng.sample(range(len(board)), 5))
        _, utility = settle_multiway_showdown(state.players, sampled_board, holes)
        return utility


def _infoset_key(state: PreflopState, player: Position, hole_cards: tuple[str, str]) -> tuple[object, ...]:
    return (
        player.value,
        hole_cards,
        state.pot,
        state.current_bet,
        state.last_full_raise_size,
        tuple((seat.position.value, seat.committed_this_street, seat.folded, seat.all_in) for seat in state.players),
        tuple((action.kind.value, action.amount) for action in state.action_history),
    )
