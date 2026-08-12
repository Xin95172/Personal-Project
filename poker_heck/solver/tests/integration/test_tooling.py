import csv
import json
import pickle

import pytest

from poker_solver.engine.river_game import Player, create_river_game
from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.config import load_config
from poker_solver.solver_core.reporting import action_probabilities, export_strategy_csv
from poker_solver.solver_core.reporting import export_preflop_strategy_csv
from poker_solver.solver_core.river_mccfr import TurnRiverMCCFRTrainer
from poker_solver.solver_core.river_mccfr import WeightedRange
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.table import Position, create_8max_preflop


def write_config(path, **changes):
    config = {
        "board": ["As", "Kd", "Qh", "Jc", "2s"],
        "initial_pot_bb": 10,
        "effective_stack_bb": 95,
        "seed": 7,
        "sizing_policy": {"bet_sizes": [0.5], "raise_sizes": [0.75], "max_re_raises": 1},
        "oop_range": [{"cards": ["Ts", "3d"]}],
        "ip_range": [{"cards": ["Tc", "4d"]}, {"cards": ["8h", "8c"]}],
    }
    config.update(changes)
    path.write_text(json.dumps(config), encoding="utf-8")


def test_json_config_creates_a_trainable_solver(tmp_path):
    path = tmp_path / "river.json"
    write_config(path)
    config = load_config(path)
    trainer = config.create_trainer()
    stats = trainer.train(5)
    assert stats.iterations == 5
    assert trainer.initial_pot_bb == 10
    assert trainer.effective_stack_bb == 95


def test_json_config_with_four_board_cards_creates_turn_chance_trainer(tmp_path):
    path = tmp_path / "turn.json"
    write_config(path, board=["As", "Kd", "Qh", "Jc"])
    trainer = load_config(path).create_trainer()
    assert isinstance(trainer, TurnRiverMCCFRTrainer)
    assert trainer.train(3).infosets > 0


def test_json_config_turn_mode_creates_full_turn_solver(tmp_path):
    path = tmp_path / "turn.json"
    write_config(path, board=["As", "Kd", "Qh", "Jc"], mode="turn")
    trainer = load_config(path).create_trainer()
    assert isinstance(trainer, TurnMCCFRTrainer)
    assert trainer.train(3).infosets > 0


def test_json_config_with_flop_board_creates_flop_solver(tmp_path):
    path = tmp_path / "flop.json"
    write_config(path, board=["As", "Kd", "Qh"])
    trainer = load_config(path).create_trainer()
    assert isinstance(trainer, FlopMCCFRTrainer)
    assert trainer.train(3).infosets > 0


@pytest.mark.parametrize(
    "changes",
    [
        {"board": ["As"]},
        {"oop_range": "not-a-list"},
        {"oop_range": [{"cards": ["As"]}]},
    ],
)
def test_json_config_rejects_invalid_shapes(tmp_path, changes):
    path = tmp_path / "invalid.json"
    write_config(path, **changes)
    with pytest.raises(ValueError, match="invalid river solve configuration"):
        load_config(path)


def test_checkpoint_can_resume_training(tmp_path):
    path = tmp_path / "river.checkpoint"
    config_path = tmp_path / "river.json"
    write_config(config_path)
    trainer = load_config(config_path).create_trainer()
    trainer.train(5)
    save_checkpoint(trainer, path)
    resumed = load_checkpoint(path)
    resumed.train(3)
    assert resumed.iterations_completed == 8
    assert len(resumed.infosets) > 0


def test_checkpoint_resume_is_identical_to_uninterrupted_training(tmp_path):
    config_path = tmp_path / "river.json"
    checkpoint_path = tmp_path / "river.checkpoint"
    write_config(config_path)

    uninterrupted = load_config(config_path).create_trainer()
    uninterrupted.train(8)

    paused = load_config(config_path).create_trainer()
    paused.train(5)
    save_checkpoint(paused, checkpoint_path)
    resumed = load_checkpoint(checkpoint_path)
    resumed.train(3)

    assert resumed.iterations_completed == uninterrupted.iterations_completed
    assert resumed.infosets == uninterrupted.infosets


def test_checkpoint_rejects_unrelated_pickle(tmp_path):
    path = tmp_path / "wrong.checkpoint"
    with path.open("wb") as file:
        pickle.dump({"not": "a trainer"}, file)
    with pytest.raises(ValueError, match="does not contain"):
        load_checkpoint(path)


def test_strategy_export_uses_bb_and_pot_ratio_output(tmp_path):
    config_path = tmp_path / "river.json"
    write_config(config_path)
    trainer = load_config(config_path).create_trainer()
    trainer.train(10)
    state = create_river_game(("As", "Kd", "Qh", "Jc", "2s"), ("Ts", "3d"), ("Tc", "4d"))
    output = export_strategy_csv(trainer, state, Player.OOP, tmp_path / "strategy.csv")
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert rows
    assert {row["pot"] for row in rows} == {"10 BB"}
    assert all("pot" in row["action"] or row["action"] in {"check", "all-in"} for row in rows)

    pairs = action_probabilities(trainer.strategy_for(state, Player.OOP), state)
    assert sum(probability for _, probability in pairs) == pytest.approx(1.0)


def test_preflop_checkpoint_and_csv_export_can_resume(tmp_path):
    cards = (("As", "Kd"), ("Qh", "Jc"), ("Ts", "9d"), ("8h", "7c"), ("6s", "5d"), ("4h", "3c"), ("2s", "Ac"), ("Kh", "Qd"))
    ranges = {position: WeightedRange.from_cards((cards[index],)) for index, position in enumerate(Position)}
    trainer = MultiwayPreflopMCCFRTrainer(ranges=ranges, sizing_policy=PreflopSizingPolicy(include_all_in=False, max_raises=0), seed=3)
    trainer.train(1)
    checkpoint = tmp_path / "preflop.pkl"
    save_checkpoint(trainer, checkpoint)
    resumed = load_checkpoint(checkpoint)
    assert isinstance(resumed, MultiwayPreflopMCCFRTrainer)
    resumed.train(1)
    root = create_8max_preflop(stack_bb=resumed.stack_bb)
    hole_cards = ranges[Position.UTG].combos[0].cards
    output = export_preflop_strategy_csv(resumed, root, Position.UTG, hole_cards, tmp_path / "preflop.csv")
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert rows
    assert {row["pot"] for row in rows} == {"1.5 BB"}
    assert all("BB" in row["action"] or row["action"] in {"fold", "call", "check", "all-in"} for row in rows)
