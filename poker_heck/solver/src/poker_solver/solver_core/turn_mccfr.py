"""包含 turn 決策、river chance node 與 river 決策的 MCCFR trainer。"""

from random import Random
from time import perf_counter

from poker_solver.engine.chance import remaining_cards
from poker_solver.engine.postflop_game import (
    PostflopGameState,
    abstract_actions,
    advance_to_next_street,
    apply_action,
    create_flop_game,
    create_turn_game,
    infoset_key,
    is_chance_node,
    is_terminal,
    terminal_utility,
)
from poker_solver.engine.river_game import Action, Player, SizingPolicy
from poker_solver.solver_core.river_mccfr import InfoSet, TrainingStats, WeightedRange, _sample_action


class TurnMCCFRTrainer:
    """固定 turn board 的 heads-up External Sampling MCCFR trainer。"""

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
        regrets = [max(0.0, value) for node in self.infosets.values() for value in node.regret_sum.values()]
        return TrainingStats(iterations, self.iterations_completed, len(self.infosets), elapsed, sum(regrets) / len(regrets) if regrets else 0.0)

    def strategy_for(self, state: PostflopGameState, player: Player) -> dict[Action, float]:
        key = infoset_key(state, player)
        if key not in self.infosets:
            raise KeyError("this information set has not been visited during training")
        return self.infosets[key].average_strategy()

    def _sample_initial_state(self) -> PostflopGameState:
        board = set(self.turn_board)
        for _ in range(10_000):
            oop = self.oop_range.sample(self.rng)
            ip = self.ip_range.sample(self.rng)
            if len(board | set(oop.cards) | set(ip.cards)) == 8:
                return create_turn_game(
                    self.turn_board,
                    oop.cards,
                    ip.cards,
                    initial_pot_bb=self.initial_pot_bb,
                    effective_stack_bb=self.effective_stack_bb,
                )
        raise ValueError("the supplied ranges cannot produce a non-overlapping deal")

    def _node(self, state: PostflopGameState, player: Player, actions: tuple[Action, ...]) -> InfoSet:
        key = infoset_key(state, player)
        node = self.infosets.get(key)
        if node is None:
            node = InfoSet(actions)
            self.infosets[key] = node
        elif node.actions != actions:
            raise RuntimeError("an information set produced inconsistent abstract actions")
        return node

    def _traverse(self, state: PostflopGameState, traverser: Player, reach: dict[Player, float]) -> float:
        if is_terminal(state):
            oop_utility, ip_utility = terminal_utility(state)
            return oop_utility if traverser is Player.OOP else ip_utility
        if is_chance_node(state):
            cards = remaining_cards((*state.board, *state.oop.hole_cards, *state.ip.hole_cards))
            return self._traverse(advance_to_next_street(state, cards[self.rng.randrange(len(cards))]), traverser, reach)

        assert state.current_player is not None
        actor = state.current_player
        actions = abstract_actions(state, self.sizing_policy)
        node = self._node(state, actor, actions)
        strategy = node.current_strategy()
        node.accumulate_average_strategy(strategy, reach[actor])
        if actor is traverser:
            values = {
                action: self._traverse(apply_action(state, action), traverser, {**reach, actor: reach[actor] * strategy[action]})
                for action in actions
            }
            node_value = sum(strategy[action] * values[action] for action in actions)
            for action in actions:
                node.observe_action_value(action, values[action])
                node.regret_sum[action] += values[action] - node_value
            return node_value

        action = _sample_action(strategy, self.rng)
        return self._traverse(apply_action(state, action), traverser, {**reach, actor: reach[actor] * strategy[action]})


class FlopMCCFRTrainer(TurnMCCFRTrainer):
    """固定 flop board 的 heads-up MCCFR trainer，包含 turn 與 river chance node。"""

    def __init__(
        self,
        *,
        flop_board: tuple[str, str, str],
        oop_range: WeightedRange,
        ip_range: WeightedRange,
        sizing_policy: SizingPolicy | None = None,
        initial_pot_bb: int | float | str = 10,
        effective_stack_bb: int | float | str = 95,
        seed: int = 0,
    ) -> None:
        if len(flop_board) != 3:
            raise ValueError("flop_board must contain three cards")
        self.flop_board = flop_board
        self.oop_range = oop_range
        self.ip_range = ip_range
        self.sizing_policy = sizing_policy or SizingPolicy()
        self.initial_pot_bb = initial_pot_bb
        self.effective_stack_bb = effective_stack_bb
        self.rng = Random(seed)
        self.infosets = {}
        self.iterations_completed = 0

    def _sample_initial_state(self) -> PostflopGameState:
        board = set(self.flop_board)
        for _ in range(10_000):
            oop = self.oop_range.sample(self.rng)
            ip = self.ip_range.sample(self.rng)
            if len(board | set(oop.cards) | set(ip.cards)) == 7:
                return create_flop_game(
                    self.flop_board,
                    oop.cards,
                    ip.cards,
                    initial_pot_bb=self.initial_pot_bb,
                    effective_stack_bb=self.effective_stack_bb,
                )
        raise ValueError("the supplied ranges cannot produce a non-overlapping deal")
