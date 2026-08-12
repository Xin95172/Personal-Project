import json
from pathlib import Path

from poker_solver.generators.multiway import build_jobs
from poker_solver.generators.heads_up import build_jobs as build_heads_up_jobs
from poker_solver.generators.preflop import build_jobs as build_preflop_jobs


def test_full_grid_generator_creates_every_selected_combination_in_priority_order():
    raw = json.loads((Path(__file__).resolve().parents[2] / "configs" / "multiway_solution_grid.json").read_text(encoding="utf-8"))
    raw["max_canonical_boards"] = 2
    raw["max_preflop_routes_per_stack"] = 1
    jobs = build_jobs(raw, Path("generated/multiway_packs"))

    # 3 stacks × 5 preflop routes × 2 canonical flops.
    assert len(jobs) == 6
    first_job, first_config = jobs[0]
    assert "canonical_000001_40bb" in first_job["config"]
    assert first_config["stack_bb"] == 40
    assert len(first_config["board"]) == 3
    assert first_config["completed_street_actions"] == []
    assert first_config["range_spec"] == {"kind": "all_combos"}
    assert first_config["sizing_policy"]["max_re_raises"] == 1
    assert len({job["range_profile_id"] for job, _ in jobs}) == 6


def test_heads_up_grid_generator_covers_every_selected_combination_in_priority_order():
    raw = json.loads((Path(__file__).resolve().parents[2] / "configs" / "heads_up_solution_grid.json").read_text(encoding="utf-8"))
    raw["max_canonical_boards_per_street"] = 2
    jobs = build_heads_up_jobs(raw, Path("generated/heads_up_packs"))

    # 3 stacks × 2 pot profiles × 3 streets × 2 canonical boards.
    assert len(jobs) == 36
    first_job, first_config = jobs[0]
    assert first_job["game_type"] == "river_heads_up"
    assert first_config["effective_stack_bb"] == 100
    assert len(first_config["board"]) == 5


def test_preflop_grid_generator_covers_stack_full_range_and_action_profiles():
    raw = json.loads((Path(__file__).resolve().parents[2] / "configs" / "preflop_solution_grid.json").read_text(encoding="utf-8"))
    jobs = build_preflop_jobs(raw, Path("generated/preflop_packs"))
    assert len(jobs) == 6
    first_job, first_config = jobs[0]
    assert first_job["game_type"] == "preflop_8max"
    assert first_config["stack_bb"] == 100
    assert first_config["range_spec"] == {"kind": "all_combos"}
