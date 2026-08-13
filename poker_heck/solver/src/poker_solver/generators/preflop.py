"""依抽象規則產生 8-Max preflop 訓練 pack。"""
from __future__ import annotations
from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def main() -> None:
    parser = ArgumentParser(description="產生 8-Max preflop 訓練 pack")
    parser.add_argument("grid", type=Path)
    args = parser.parse_args()
    path = args.grid.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    output, manifest = _resolve(path.parent, raw["output_dir"]), _resolve(path.parent, raw["manifest"])
    jobs = build_jobs(raw, output)
    output.mkdir(parents=True, exist_ok=True)
    for job, config in tqdm(jobs, desc="寫入 pack", unit="pack", dynamic_ncols=True):
        (output / Path(job["config"]).name).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"strategy_db": raw["strategy_db"], "jobs": [job for job, _ in jobs]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已產生 {len(jobs)} 個 preflop pack：{output}")


def build_jobs(raw: dict[str, Any], output: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    stacks = tuple(int(value) for value in raw["stack_bb"])
    if not stacks or any(value <= 0 for value in stacks):
        raise ValueError("stack_bb 必須是非空的正數陣列")
    jobs = []
    for seed, stack in enumerate(stacks, start=int(raw["seed_start"])):
        stem = f"pf_{seed:03d}_{stack}bb"
        config = {"stack_bb": stack, "range_spec": raw["range_spec"], "sizing_policy": raw["sizing_policy"], "seed": seed}
        if "continuation" in raw:
            config["continuation"] = raw["continuation"]
        options = raw["job_options"]
        jobs.append(({
            "game_type": "preflop_8max", "config": f"{output.name}/{stem}.json",
            "range_profile_id": f"{_range_profile_id(raw['range_spec'])}-{stack}bb", "solver_version": raw["solver_version"],
            "iterations": int(raw["iterations_per_pack"]), "checkpoint": f"{raw['checkpoint_dir']}/{stem}.pkl",
            "checkpoint_every": int(raw["checkpoint_every"]), "export_all_routes": bool(options["export_all_routes"]),
            "quality_report": bool(options["quality_report"]),
        }, config))
    return jobs


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _range_profile_id(spec: dict[str, Any]) -> str:
    return f"top_percent_{float(spec['percent']):g}" if spec["kind"] == "top_percent" else str(spec["kind"])


if __name__ == "__main__":
    main()
