import pytest
import json

from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.table import Position
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.preflop_config import load_preflop_trainer
from poker_solver.solver_core.river_mccfr import WeightedRange


def ranges():
    cards = (("As", "Kd"), ("Qh", "Jc"), ("Ts", "9d"), ("8h", "7c"), ("6s", "5d"), ("4h", "3c"), ("2s", "Ac"), ("Kh", "Qd"))
    return {position: WeightedRange.from_cards((cards[index],)) for index, position in enumerate(Position)}


def test_multiway_preflop_mccfr_trains_a_strategy_profile():
    trainer = MultiwayPreflopMCCFRTrainer(
        ranges=ranges(),
        sizing_policy=PreflopSizingPolicy(include_all_in=False, max_raises=0),
        seed=1,
    )
    stats = trainer.train(1)
    assert stats.infosets > 0
    assert all(sum(node.average_strategy().values()) == pytest.approx(1.0) for node in trainer.infosets.values())


def test_multiway_preflop_requires_all_positions_to_have_ranges():
    incomplete = ranges()
    incomplete.pop(Position.BB)
    with pytest.raises(ValueError, match="every"):
        MultiwayPreflopMCCFRTrainer(ranges=incomplete)


def test_preflop_json_config_loads_and_trains(tmp_path):
    cards = (("As", "Kd"), ("Qh", "Jc"), ("Ts", "9d"), ("8h", "7c"), ("6s", "5d"), ("4h", "3c"), ("2s", "Ac"), ("Kh", "Qd"))
    raw = {
        "sizing_policy": {"include_all_in": False, "max_raises": 0},
        "ranges": {position.value: [{"cards": cards[index]}] for index, position in enumerate(Position)},
    }
    path = tmp_path / "preflop.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    trainer = load_preflop_trainer(path)
    assert trainer.train(1).infosets > 0


def test_preflop_top_percent_range_expands_without_fixed_cards(tmp_path):
    raw = {
        "range_spec": {
            "kind": "top_percent",
            "percent": 10,
            "percent_by_position": {"utg": 5, "btn": 40},
        },
        "sizing_policy": {"include_all_in": False, "max_raises": 0},
    }
    path = tmp_path / "top-percent.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    trainer = load_preflop_trainer(path)
    assert len(trainer.ranges[Position.UTG].combos) == 67
    assert len(trainer.ranges[Position.BTN].combos) == 531
    assert len(trainer.ranges[Position.MP].combos) == 133
