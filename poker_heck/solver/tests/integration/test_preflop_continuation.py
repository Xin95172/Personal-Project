from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import Position, apply_action, create_8max_preflop
from poker_solver.engine.table import advance_preflop_to_flop
from poker_solver.solver_core import integrated_solver
from poker_solver.solver_core.integrated_solver import ContinuationSettings, PostflopContinuationOracle
from poker_solver.solver_core.preflop_config import load_preflop_trainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.preflop_range_conditioning import condition_ranges_for_terminal
from poker_solver.solver_core.river_mccfr import WeightedRange
from poker_solver.solver_core.multiway_postflop_config import load_multiway_postflop_trainer
from poker_solver.solver_core.multiway_postflop_config import _load_ranges
from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
import pytest
from dataclasses import replace


def _ranges():
    cards = (("As", "Kd"), ("Qh", "Jc"), ("Ts", "9d"), ("8h", "7c"), ("6s", "5d"), ("4h", "3c"), ("2s", "Ac"), ("Kh", "Qd"))
    return {position: WeightedRange.from_cards((cards[index],)) for index, position in enumerate(Position)}


def _limped_terminal_state():
    state = create_8max_preflop()
    while not state.betting_complete:
        state = apply_action(state, Action(ActionType.CHECK if state.current_player is Position.BB else ActionType.CALL))
    return state


def test_range_conditioning_replays_observed_preflop_action():
    trainer = MultiwayPreflopMCCFRTrainer(ranges=_ranges(), sizing_policy=PreflopSizingPolicy(include_all_in=False, max_raises=0), seed=1)
    trainer.train(1)
    state = _limped_terminal_state()
    excluded = ("Ah", "Ad", "2h")
    conditioned = condition_ranges_for_terminal(trainer, state, excluded_cards=excluded)
    assert set(conditioned) == set(Position)
    assert all(not set(combo.cards) & set(excluded) for value in conditioned.values() for combo in value.combos)


def test_continuation_oracle_caches_subgame_and_returns_utility(monkeypatch):
    class FakeSubgame:
        def __init__(self, *, initial_state, **_kwargs):
            self.initial_state = initial_state
            self.iterations_completed = 1
            self.infosets = {}

        def train(self, _iterations):
            return None

        def average_strategy_utility(self, _state, _holes, _policy):
            return {position: 0.5 for position in Position}

    monkeypatch.setattr(integrated_solver, "MultiwayPostflopMCCFRTrainer", FakeSubgame)
    trainer = MultiwayPreflopMCCFRTrainer(ranges=_ranges(), sizing_policy=PreflopSizingPolicy(include_all_in=False, max_raises=0), seed=1)
    oracle = PostflopContinuationOracle(trainer, ContinuationSettings(subgame_iterations=1, value_rollouts=2, max_cached_subgames=1))
    holes = {position: value.combos[0].cards for position, value in trainer.ranges.items()}
    values = oracle.utility(_limped_terminal_state(), holes)
    assert values == {position: 0.5 for position in Position}
    assert len(oracle.cache) == 1


def test_preflop_configuration_enables_database_backed_continuation(tmp_path):
    config = tmp_path / "preflop.json"
    config.write_text(
        '{"range_spec":{"kind":"top_percent","percent":20},"continuation":{"enabled":true,"strategy_db":"strategies.sqlite3","subgame_iterations":1,"value_rollouts":1}}',
        encoding="utf-8",
    )
    trainer = load_preflop_trainer(config)
    assert trainer.continuation_utility is not None
    assert (tmp_path / "strategies.sqlite3").exists()


def test_continuation_uses_only_exact_database_strategy_context():
    class Store:
        def lookup(self, context):
            self.context = context
            return type("Stored", (), {"actions": (type("Item", (), {"kind": "check", "amount_units": None, "probability": 1.0})(),)})()

    trainer = MultiwayPreflopMCCFRTrainer(ranges=_ranges(), seed=1)
    oracle = PostflopContinuationOracle(trainer, ContinuationSettings())
    oracle.store = Store()
    state = advance_preflop_to_flop(_limped_terminal_state(), ("Ah", "Ad", "2h"))
    strategy = oracle._database_policy(state, state.current_player, ("As", "Kd"), 9_900)
    assert strategy == {Action(ActionType.CHECK): 1.0}
    assert oracle.store.context.board == ("Ah", "Ad", "2h")


def test_multiway_postflop_config_builds_a_completed_preflop_subgame(tmp_path):
    config = tmp_path / "multiway.json"
    actions = [{"kind": "call"}] * 7 + [{"kind": "check"}]
    config.write_text(
        __import__("json").dumps({"stack_bb": 40, "board": ["Ah", "Ad", "2h"], "preflop_actions": actions, "range_spec": {"kind": "top_percent", "percent": 20}, "sizing_policy": {"bet_sizes": [0.5], "raise_sizes": [1.0], "include_all_in": False, "max_re_raises": 0}}),
        encoding="utf-8",
    )
    trainer = load_multiway_postflop_trainer(config)
    assert trainer.initial_state.board == ("Ah", "Ad", "2h")
    assert len(trainer.ranges[Position.UTG].combos) > 0


def test_database_root_hit_skips_subgame_retraining(monkeypatch):
    class FakeSubgame:
        def __init__(self, *, initial_state, **_kwargs):
            self.initial_state = initial_state
            self.iterations_completed = 0
            self.infosets = {}

        def train(self, _iterations):
            raise AssertionError("資料庫命中時不應重訓子局")

        def average_strategy_utility(self, _state, _holes, _policy):
            return {position: 0.0 for position in Position}

    class Store:
        def lookup(self, _context):
            return type("Stored", (), {"actions": (type("Item", (), {"kind": "check", "amount_units": None, "probability": 1.0})(),)})()

    monkeypatch.setattr(integrated_solver, "MultiwayPostflopMCCFRTrainer", FakeSubgame)
    trainer = MultiwayPreflopMCCFRTrainer(ranges=_ranges(), seed=1)
    oracle = PostflopContinuationOracle(trainer, ContinuationSettings(subgame_iterations=1, value_rollouts=1))
    oracle.store = Store()
    holes = {position: value.combos[0].cards for position, value in trainer.ranges.items()}
    oracle.utility(_limped_terminal_state(), holes)


def test_continuation_checkpoint_can_be_saved_and_loaded(tmp_path):
    config = tmp_path / "preflop.json"
    config.write_text(
        '{"range_spec":{"kind":"top_percent","percent":20},"continuation":{"enabled":true,"strategy_db":"strategies.sqlite3","subgame_iterations":1,"value_rollouts":1}}',
        encoding="utf-8",
    )
    saved = save_checkpoint(load_preflop_trainer(config), tmp_path / "preflop.pkl")
    resumed = load_checkpoint(saved)
    assert resumed.continuation_utility is not None


@pytest.mark.parametrize(
    "payload",
    [
        {"preflop_actions": [], "board": ["Ah", "Ad", "2h"]},
        {"preflop_actions": [{"kind": "call"}] * 7 + [{"kind": "check"}], "board": ["Ah", "Ad"]},
    ],
)
def test_multiway_config_rejects_incomplete_preflop_or_invalid_board(tmp_path, payload):
    config = tmp_path / "invalid.json"
    config.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid multiway postflop solve configuration"):
        load_multiway_postflop_trainer(config)


def test_multiway_range_spec_rejects_non_mapping_position_overrides():
    with pytest.raises(ValueError, match="percent_by_position"):
        _load_ranges({"range_spec": {"kind": "top_percent", "percent": 20, "percent_by_position": []}}, ("Ah", "Ad", "2h"))


def test_multiway_trainer_rejects_incomplete_ranges_and_nonpositive_training():
    state = advance_preflop_to_flop(_limped_terminal_state(), ("Ah", "Ad", "2h"))
    with pytest.raises(ValueError, match="every"):
        MultiwayPostflopMCCFRTrainer(initial_state=state, ranges={})
    trainer = MultiwayPostflopMCCFRTrainer(initial_state=state, ranges=_ranges())
    with pytest.raises(ValueError, match="positive"):
        trainer.train(0)
    with pytest.raises(ValueError, match="flop, turn, or river"):
        MultiwayPostflopMCCFRTrainer(initial_state=replace(state, board=()), ranges=_ranges())
