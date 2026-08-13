from poker_solver.engine.multiway_postflop_policy import MultiwayPostflopSizingPolicy
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import Position, advance_preflop_to_flop, apply_action, create_8max_preflop
from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.multiway_postflop_config import load_multiway_postflop_trainer
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.river_mccfr import WeightedRange
import pytest


def test_multiway_postflop_trainer_smoke_runs_from_flop_to_showdown():
    preflop = create_8max_preflop()
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    state = advance_preflop_to_flop(preflop, ("As", "Kd", "Qh"))
    combos = (("2c", "2d"), ("3c", "3d"), ("4c", "4d"), ("5c", "5d"), ("6c", "6d"), ("7c", "7d"), ("8c", "8d"), ("9c", "9d"))
    trainer = MultiwayPostflopMCCFRTrainer(
        initial_state=state,
        ranges={position: WeightedRange.from_cards((combos[index],)) for index, position in enumerate(Position)},
        sizing_policy=MultiwayPostflopSizingPolicy(bet_sizes=(0.5,), raise_sizes=(), include_all_in=False, max_re_raises=0),
        seed=7,
    )
    stats = trainer.train(1)
    assert stats.total_iterations == 1
    assert stats.infosets > 0


def test_multiway_postflop_checkpoint_round_trip(tmp_path):
    preflop = create_8max_preflop()
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    combos = (("2c", "2d"), ("3c", "3d"), ("4c", "4d"), ("5c", "5d"), ("6c", "6d"), ("7c", "7d"), ("8c", "8d"), ("9c", "9d"))
    trainer = MultiwayPostflopMCCFRTrainer(
        initial_state=advance_preflop_to_flop(preflop, ("As", "Kd", "Qh")),
        ranges={position: WeightedRange.from_cards((combos[index],)) for index, position in enumerate(Position)},
        sizing_policy=MultiwayPostflopSizingPolicy(bet_sizes=(0.5,), raise_sizes=(), include_all_in=False, max_re_raises=0),
    )
    trainer.train(1)
    path = save_checkpoint(trainer, tmp_path / "multiway-postflop.pkl")
    resumed = load_checkpoint(path)
    assert isinstance(resumed, MultiwayPostflopMCCFRTrainer)
    assert resumed.iterations_completed == 1


@pytest.mark.parametrize(("mode", "expected_traversals"), [("single_random", 1), ("all_players", 8)])
def test_multiway_traverser_mode_controls_regret_update_traversals(monkeypatch, mode, expected_traversals):
    preflop = create_8max_preflop()
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    combos = (("2c", "2d"), ("3c", "3d"), ("4c", "4d"), ("5c", "5d"), ("6c", "6d"), ("7c", "7d"), ("8c", "8d"), ("9c", "9d"))
    trainer = MultiwayPostflopMCCFRTrainer(
        initial_state=advance_preflop_to_flop(preflop, ("As", "Kd", "Qh")),
        ranges={position: WeightedRange.from_cards((combos[index],)) for index, position in enumerate(Position)},
        sizing_policy=MultiwayPostflopSizingPolicy(bet_sizes=(0.5,), raise_sizes=(), include_all_in=False, max_re_raises=0),
        traverser_mode=mode,
    )
    calls = []
    monkeypatch.setattr(trainer, "_traverse", lambda _state, _holes, traverser, _reach: calls.append(traverser) or 0.0)
    trainer.train(1)
    assert len(calls) == expected_traversals


def test_multiway_postflop_json_config_builds_trainer(tmp_path):
    import json

    cards = (("2c", "2d"), ("3c", "3d"), ("4c", "4d"), ("5c", "5d"), ("6c", "6d"), ("7c", "7d"), ("8c", "8d"), ("9c", "9d"))
    raw = {
        "flop": ["As", "Kd", "Qh"],
        "preflop_actions": [{"kind": "fold"}] * 6 + [{"kind": "call"}, {"kind": "check"}],
        "ranges": {position.value: [{"cards": list(cards[index])}] for index, position in enumerate(Position)},
        "sizing_policy": {"bet_sizes": [0.5], "raise_sizes": [], "include_all_in": False, "max_re_raises": 0},
    }
    path = tmp_path / "postflop.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    trainer = load_multiway_postflop_trainer(path)
    assert trainer.initial_state.current_player is Position.SB
    assert trainer.train(1).infosets > 0
