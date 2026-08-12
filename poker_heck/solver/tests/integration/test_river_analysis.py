import json

import pytest

from poker_solver.engine.river_game import SizingPolicy
from poker_solver.solver_core.river_analysis import analyze_river_profile, write_river_quality_report
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer, WeightedRange


def test_river_profile_analysis_returns_finite_zero_sum_quality_metrics():
    trainer = RiverMCCFRTrainer(
        board=("As", "Kd", "Qh", "Jc", "2s"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1),
        seed=3,
    )
    trainer.train(8)
    report = analyze_river_profile(trainer)
    assert report.compatible_combo_pairs == 1
    assert report.oop_best_response_bb >= report.profile_ev_oop_bb
    assert report.ip_best_response_bb >= -report.profile_ev_oop_bb
    assert report.nash_conv_bb == pytest.approx(report.oop_best_response_bb + report.ip_best_response_bb)
    assert report.nash_conv_initial_pot_fraction == pytest.approx(report.nash_conv_bb / 10)
    assert report.nash_conv_effective_stack_fraction == pytest.approx(report.nash_conv_bb / 95)


def test_river_quality_report_is_machine_readable_json(tmp_path):
    trainer = RiverMCCFRTrainer(
        board=("As", "Kd", "Qh", "Jc", "2s"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=0),
    )
    trainer.train(2)
    path = write_river_quality_report(analyze_river_profile(trainer), tmp_path / "quality.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "nash_conv_initial_pot_percent" in payload
    assert "nash_conv_effective_stack_percent" in payload
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["scope"].startswith("heads-up river")
    assert data["compatible_combo_pairs"] == 1
