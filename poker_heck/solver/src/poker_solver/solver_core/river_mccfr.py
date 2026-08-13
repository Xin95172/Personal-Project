"""River-only、heads-up External Sampling MCCFR 的最小 solver。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from random import Random
from time import perf_counter
from typing import Iterable

from poker_solver.engine.chance import TurnToRiverScenario
from poker_solver.engine.river_game import (
    Action,
    Player,
    RiverGameState,
    SizingPolicy,
    abstract_actions,
    apply_action,
    create_river_game,
    infoset_key,
    is_terminal,
    terminal_utility,
)


@dataclass(frozen=True)
class Combo:
    """一個兩張私牌 combo 與其 range 權重。"""

    cards: tuple[str, str]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.cards) != 2 or len(set(self.cards)) != 2:
            raise ValueError("a combo requires two distinct cards")
        if self.weight <= 0:
            raise ValueError("combo weight must be positive")


@dataclass(frozen=True)
class WeightedRange:
    """可用於 chance sampling 的加權私牌範圍。"""

    combos: tuple[Combo, ...]

    def __post_init__(self) -> None:
        if not self.combos:
            raise ValueError("a range must contain at least one combo")
        if len({combo.cards for combo in self.combos}) != len(self.combos):
            raise ValueError("a range cannot contain duplicate combos")

    @classmethod
    def from_cards(cls, combos: Iterable[tuple[str, str]]) -> "WeightedRange":
        return cls(tuple(Combo(cards) for cards in combos))

    def sample(self, rng: Random) -> Combo:
        total = sum(combo.weight for combo in self.combos)
        threshold = rng.random() * total
        running = 0.0
        for combo in self.combos:
            running += combo.weight
            if running >= threshold:
                return combo
        return self.combos[-1]  # 浮點數邊界保護。


@dataclass
class InfoSet:
    """一個資訊集的 cumulative regret 與平均策略累積值。"""

    actions: tuple[Action, ...]
    regret_sum: dict[Action, float] = field(init=False)
    strategy_sum: dict[Action, float] = field(init=False)
    action_value_count: dict[Action, int] = field(init=False)
    action_value_sum: dict[Action, float] = field(init=False)
    action_value_sum_squares: dict[Action, float] = field(init=False)
    visit_count: int = 0

    def __post_init__(self) -> None:
        self.regret_sum = {action: 0.0 for action in self.actions}
        self.strategy_sum = {action: 0.0 for action in self.actions}
        self.action_value_count = {action: 0 for action in self.actions}
        self.action_value_sum = {action: 0.0 for action in self.actions}
        self.action_value_sum_squares = {action: 0.0 for action in self.actions}

    def observe_action_value(self, action: Action, value: float) -> None:
        self.action_value_count[action] += 1
        self.action_value_sum[action] += value
        self.action_value_sum_squares[action] += value * value

    def action_value_stats(self, action: Action) -> dict[str, float | int | None]:
        count = self.action_value_count[action]
        if count == 0:
            return {"samples": 0, "ev_mean": None, "ev_stddev": None, "ev_stderr": None, "ci95_low": None, "ci95_high": None}
        mean = self.action_value_sum[action] / count
        variance = max(0.0, self.action_value_sum_squares[action] / count - mean * mean)
        stddev = sqrt(variance)
        stderr = stddev / sqrt(count)
        return {"samples": count, "ev_mean": mean, "ev_stddev": stddev, "ev_stderr": stderr, "ci95_low": mean - 1.96 * stderr, "ci95_high": mean + 1.96 * stderr}

    def current_strategy(self) -> dict[Action, float]:
        positive = {action: max(0.0, regret) for action, regret in self.regret_sum.items()}
        normalizer = sum(positive.values())
        if normalizer == 0:
            probability = 1.0 / len(self.actions)
            return {action: probability for action in self.actions}
        return {action: positive[action] / normalizer for action in self.actions}

    def accumulate_average_strategy(self, strategy: dict[Action, float], weight: float = 1.0) -> None:
        for action, probability in strategy.items():
            self.strategy_sum[action] += weight * probability
        self.visit_count += 1

    def average_strategy(self) -> dict[Action, float]:
        normalizer = sum(self.strategy_sum.values())
        if normalizer == 0:
            return self.current_strategy()
        return {action: self.strategy_sum[action] / normalizer for action in self.actions}


@dataclass(frozen=True)
class TrainingStats:
    iterations: int
    total_iterations: int
    infosets: int
    elapsed_seconds: float
    mean_positive_regret: float

    @property
    def iterations_per_second(self) -> float:
        return self.iterations / self.elapsed_seconds if self.elapsed_seconds else 0.0


class RiverMCCFRTrainer:
    """以 External Sampling MCCFR 訓練固定 river board 的雙人策略。"""

    def __init__(
        self,
        *,
        board: tuple[str, str, str, str, str],
        oop_range: WeightedRange,
        ip_range: WeightedRange,
        sizing_policy: SizingPolicy | None = None,
        initial_pot_bb: int | float | str = 10,
        effective_stack_bb: int | float | str = 95,
        seed: int = 0,
    ) -> None:
        self.board = board
        self.oop_range = oop_range
        self.ip_range = ip_range
        self.sizing_policy = sizing_policy or SizingPolicy()
        self.initial_pot_bb = initial_pot_bb
        self.effective_stack_bb = effective_stack_bb
        self.rng = Random(seed)
        self.infosets: dict[tuple[object, ...], InfoSet] = {}
        self.iterations_completed = 0

    def train(self, iterations: int) -> TrainingStats:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        started = perf_counter()
        for _ in range(iterations):
            state = self._sample_initial_state()
            self._traverse(state, Player.OOP, {Player.OOP: 1.0, Player.IP: 1.0})
            self._traverse(state, Player.IP, {Player.OOP: 1.0, Player.IP: 1.0})
            self.iterations_completed += 1
        elapsed = perf_counter() - started
        return TrainingStats(
            iterations=iterations,
            total_iterations=self.iterations_completed,
            infosets=len(self.infosets),
            elapsed_seconds=elapsed,
            mean_positive_regret=self.mean_positive_regret(),
        )

    def mean_positive_regret(self) -> float:
        """回傳資訊集 action 的平均正 regret，作為訓練趨勢指標。"""
        regrets = [max(0.0, regret) for node in self.infosets.values() for regret in node.regret_sum.values()]
        return sum(regrets) / len(regrets) if regrets else 0.0

    def strategy_for(self, state: RiverGameState, player: Player) -> dict[Action, float]:
        """取得某個已被造訪資訊集的平均策略。"""
        key = infoset_key(state, player)
        if key not in self.infosets:
            raise KeyError("this information set has not been visited during training")
        return self.infosets[key].average_strategy()

    def _sample_initial_state(self) -> RiverGameState:
        oop_combo, ip_combo = self._sample_compatible_combos(self.board)
        return create_river_game(
            self.board,
            oop_combo.cards,
            ip_combo.cards,
            initial_pot_bb=self.initial_pot_bb,
            effective_stack_bb=self.effective_stack_bb,
        )

    def _sample_compatible_combos(self, board: tuple[str, ...]) -> tuple[Combo, Combo]:
        board_cards = set(board)
        for _ in range(10_000):
            oop_combo = self.oop_range.sample(self.rng)
            ip_combo = self.ip_range.sample(self.rng)
            cards = board_cards | set(oop_combo.cards) | set(ip_combo.cards)
            if len(cards) == len(board) + 4:
                return oop_combo, ip_combo
        raise ValueError("the supplied ranges cannot produce a non-overlapping deal")

    def _infoset_for(self, state: RiverGameState, player: Player, actions: tuple[Action, ...]) -> InfoSet:
        key = infoset_key(state, player)
        existing = self.infosets.get(key)
        if existing is None:
            existing = InfoSet(actions)
            self.infosets[key] = existing
        elif existing.actions != actions:
            raise RuntimeError("an information set produced inconsistent abstract actions")
        return existing

    def _traverse(self, state: RiverGameState, traverser: Player, reach: dict[Player, float]) -> float:
        """External-sampling traversal，回傳 traverser 的 utility estimate。"""
        if is_terminal(state):
            oop_utility, ip_utility = terminal_utility(state)
            return oop_utility if traverser is Player.OOP else ip_utility

        assert state.current_player is not None
        actor = state.current_player
        actions = abstract_actions(state, self.sizing_policy)
        if not actions:
            raise RuntimeError("a non-terminal state must have an abstract action")
        node = self._infoset_for(state, actor, actions)
        strategy = node.current_strategy()
        node.accumulate_average_strategy(strategy, reach[actor])

        if actor is traverser:
            action_values = {
                action: self._traverse(
                    apply_action(state, action),
                    traverser,
                    {**reach, actor: reach[actor] * strategy[action]},
                )
                for action in actions
            }
            node_value = sum(strategy[action] * action_values[action] for action in actions)
            for action in actions:
                node.observe_action_value(action, action_values[action])
                node.regret_sum[action] += action_values[action] - node_value
            return node_value

        sampled_action = _sample_action(strategy, self.rng)
        return self._traverse(
            apply_action(state, sampled_action),
            traverser,
            {**reach, actor: reach[actor] * strategy[sampled_action]},
        )


def _sample_action(strategy: dict[Action, float], rng: Random) -> Action:
    threshold = rng.random()
    running = 0.0
    for action, probability in strategy.items():
        running += probability
        if running >= threshold:
            return action
    return next(reversed(strategy))  # 浮點數邊界保護。


class TurnRiverMCCFRTrainer(RiverMCCFRTrainer):
    """在固定 turn board 下抽樣 river 的 River MCCFR 訓練器。

    此類別只讓 river 策略跨不同 river card 訓練；turn 的下注決策樹尚未加入。
    """

    def __init__(
        self,
        *,
        turn_board: tuple[str, str, str, str],
        oop_range: WeightedRange,
        ip_range: WeightedRange,
        sizing_policy: SizingPolicy | None = None,
        initial_pot_bb: int | float | str = 10,
        effective_stack_bb: int | float | str = 95,
        seed: int = 0,
    ) -> None:
        if len(turn_board) != 4:
            raise ValueError("turn_board must contain four cards")
        self.turn_board = turn_board
        # board 在這個子類別不會用於 create_river_game；保留五張型別相容的佔位資料。
        super().__init__(
            board=(*turn_board, "2c"),
            oop_range=oop_range,
            ip_range=ip_range,
            sizing_policy=sizing_policy,
            initial_pot_bb=initial_pot_bb,
            effective_stack_bb=effective_stack_bb,
            seed=seed,
        )

    def _sample_initial_state(self) -> RiverGameState:
        oop_combo, ip_combo = self._sample_compatible_combos(self.turn_board)
        scenario = TurnToRiverScenario(
            turn_board=self.turn_board,
            oop_hole_cards=oop_combo.cards,
            ip_hole_cards=ip_combo.cards,
            pot_bb=self.initial_pot_bb,
            effective_stack_bb=self.effective_stack_bb,
        )
        return scenario.sample_river(self.rng)
