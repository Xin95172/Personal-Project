import json
from pathlib import Path

from poker_solver.generators.multiway import build_jobs
from poker_solver.generators.heads_up import build_jobs as build_heads_up_jobs
from poker_solver.generators.preflop import build_jobs as build_preflop_jobs
from poker_solver.generators.conditional_subgames import build_jobs as build_subgame_jobs


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
    assert first_config["range_spec"] == {"kind": "top_percent", "percent": 20}
    assert first_config["sizing_policy"]["max_re_raises"] == 1
    assert {job["range_profile_id"] for job, _ in jobs} == {"top_percent_20"}
    assert {job["traverser_mode"] for job, _ in jobs} <= {"single_random", "all_players"}
    assert all("traverser" in job["solver_version"] for job, _ in jobs)
    assert len({json.dumps({"stack": config["stack_bb"], "actions": config["preflop_actions"], "board": config["board"]}, sort_keys=True) for _, config in jobs}) == 6
    assert len({config["seed"] for _, config in jobs}) == 6


def test_multiway_route_offset_selects_the_next_batch_without_repeating_routes():
    raw = json.loads((Path(__file__).resolve().parents[2] / "configs" / "multiway_solution_grid.json").read_text(encoding="utf-8"))
    raw["stack_bb"] = [40]
    raw["max_canonical_boards"] = 1
    raw["max_preflop_routes_per_stack"] = 1
    raw["preflop_route_offset_per_stack"] = 0
    first = build_jobs(raw, Path("generated/first"))[0][1]["preflop_actions"]
    raw["preflop_route_offset_per_stack"] = 1
    second = build_jobs(raw, Path("generated/second"))[0][1]["preflop_actions"]
    assert first != second


def test_conditional_subgame_generator_creates_turn_job_with_full_history():
    raw = {
        "seed_start": 1, "solver_version": "test-v1", "iterations_per_pack": 2, "checkpoint_every": 1,
        "checkpoint_dir": "../checkpoints",
        "subgames": [{
            "board": ["As", "Kd", "Qh", "2c"], "preflop_actions": [],
            "completed_street_actions": [[]], "range_spec": {"kind": "all_combos"},
            "sizing_policy": {}, "stack_bb": 40, "range_profile_id": "test-range",
        }],
    }
    job, config = build_subgame_jobs(raw, Path("generated/subgames"))[0]
    assert job["game_type"] == "multiway_postflop"
    assert config["solve_scope"] == "conditional_subgame"


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


def test_preflop_grid_generator_uses_one_rule_set_for_every_stack():
    raw = json.loads((Path(__file__).resolve().parents[2] / "configs" / "preflop_solution_grid.json").read_text(encoding="utf-8"))
    jobs = build_preflop_jobs(raw, Path("generated/preflop_packs"))
    assert len(jobs) == 3
    first_job, first_config = jobs[0]
    assert first_job["game_type"] == "preflop_8max"
    assert first_config["stack_bb"] == 40
    assert first_config["range_spec"] == {"kind": "top_percent", "percent": 20}
    assert first_config["sizing_policy"]["max_raises"] == 2
