import pytest
import json

from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.table import Position, create_8max_preflop
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.preflop_config import load_preflop_trainer
from poker_solver.solver_core.multiway_postflop_config import _load_ranges as load_multiway_ranges
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


def test_action_value_statistics_include_standard_error_and_confidence_interval():
    trainer = MultiwayPreflopMCCFRTrainer(
        ranges=ranges(), sizing_policy=PreflopSizingPolicy(include_all_in=False, max_raises=0), seed=1,
    )
    trainer.train(2)
    node = next(iter(trainer.infosets.values()))
    stats = node.action_value_stats(node.actions[0])
    assert stats["samples"] > 0
    assert stats["ev_stderr"] is not None
    assert stats["ci95_low"] <= stats["ev_mean"] <= stats["ci95_high"]


def test_preflop_continuation_preserves_fractional_ev():
    trainer = MultiwayPreflopMCCFRTrainer(
        ranges=ranges(), continuation_utility=lambda _state, _holes: {position: 0.5 for position in Position},
    )
    holes = {position: trainer.ranges[position].combos[0].cards for position in Position}
    utility = trainer._terminal_utility(create_8max_preflop(), holes)
    assert utility[Position.UTG] == 0.5


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


def test_multiway_top_percent_range_uses_shared_range_spec():
    ranges = load_multiway_ranges(
        {"range_spec": {"kind": "top_percent", "percent": 10, "percent_by_position": {"btn": 40}}},
        ("As", "Kd", "Qh"),
    )
    # 剩餘 49 張牌共有 1,176 combos。
    assert len(ranges[Position.UTG].combos) == 118
    assert len(ranges[Position.BTN].combos) == 471
