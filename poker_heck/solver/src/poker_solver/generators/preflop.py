"""依 preflop 網格產生 8-Max 訓練 pack。"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from itertools import product
from pathlib import Path
from typing import Any


def main() -> None:
    parser = ArgumentParser(description="產生 8-Max preflop 全網格訓練 pack")
    parser.add_argument("grid", type=Path)
    args = parser.parse_args()
    grid_path = args.grid.resolve()
    raw = json.loads(grid_path.read_text(encoding="utf-8"))
    output_dir = _resolve(grid_path.parent, raw["output_dir"])
    manifest_path = _resolve(grid_path.parent, raw["manifest"])
    jobs = build_jobs(raw, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for job, config in jobs:
        (output_dir / Path(job["config"]).name).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"strategy_db": raw["strategy_db"], "jobs": [job for job, _ in jobs]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已產生 {len(jobs)} 個 preflop pack：{output_dir}")


def build_jobs(raw: dict[str, Any], output_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    stacks = tuple(int(value) for value in raw["stack_bb"])
    ranges = raw["range_profiles"]
    policies = raw["action_profiles"]
    if not stacks or not ranges or not policies or any(value <= 0 for value in stacks):
        raise ValueError("stack_bb, range_profiles, and action_profiles must be non-empty")
    keys = [(stack, range_name, action_name) for stack, range_name, action_name in product(stacks, ranges, policies)]
    ordered: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str, str]] = set()
    for tier in raw["priority_tiers"]:
        for key in keys:
            if key not in seen and (not tier.get("stack_bb") or key[0] in tier["stack_bb"]) and (not tier.get("range_profiles") or key[1] in tier["range_profiles"]) and (not tier.get("action_profiles") or key[2] in tier["action_profiles"]):
                ordered.append(key)
                seen.add(key)
    ordered.extend(key for key in keys if key not in seen)
    iterations = int(raw["iterations_per_pack"])
    checkpoint_every = int(raw["checkpoint_every"])
    seed_start = int(raw["seed_start"])
    job_options = raw["job_options"]
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, (stack, range_name, action_name) in enumerate(ordered, start=seed_start):
        stem = f"pf_{index:03d}_{range_name}_{action_name}_{stack}bb"
        config = {"stack_bb": stack, "range_spec": ranges[range_name], "sizing_policy": policies[action_name], "seed": index}
        jobs.append(({
            "game_type": "preflop_8max", "config": f"{output_dir.name}/{stem}.json",
            "range_profile_id": f"pf-grid-{range_name}-{action_name}-{stack}bb-{raw['solver_version']}", "solver_version": raw["solver_version"],
            "iterations": iterations, "checkpoint": f"{raw['checkpoint_dir']}/{stem}.pkl", "checkpoint_every": checkpoint_every,
            "export_all_routes": bool(job_options["export_all_routes"]), "quality_report": bool(job_options["quality_report"]),
        }, config))
    return jobs


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    main()
