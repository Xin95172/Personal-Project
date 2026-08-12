"""將明確的 turn／river 條件局面產生為可獨立求解的 multiway pack。"""
from __future__ import annotations
from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = ArgumentParser(description="產生 multiway conditional subgame pack")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    path = args.config.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = _resolve(path.parent, raw["output_dir"])
    manifest = _resolve(path.parent, raw["manifest"])
    jobs = build_jobs(raw, output)
    output.mkdir(parents=True, exist_ok=True)
    for job, config in jobs:
        (output / Path(job["config"]).name).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"strategy_db": raw["strategy_db"], "jobs": [job for job, _ in jobs]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_jobs(raw: dict[str, Any], output: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    jobs = []
    for index, scenario in enumerate(raw["subgames"], start=int(raw["seed_start"])):
        board = scenario["board"]
        if len(board) not in {4, 5}:
            raise ValueError("conditional subgame board 必須為 turn（4 張）或 river（5 張）")
        street = "turn" if len(board) == 4 else "river"
        stem = f"conditional_{street}_{index:06d}"
        config = {**scenario, "solve_scope": "conditional_subgame", "seed": index}
        jobs.append(({
            "game_type": "multiway_postflop", "config": f"{output.name}/{stem}.json",
            "range_profile_id": scenario["range_profile_id"], "solver_version": raw["solver_version"],
            "iterations": int(raw["iterations_per_pack"]), "checkpoint": f"{raw['checkpoint_dir']}/{stem}.pkl",
            "checkpoint_every": int(raw["checkpoint_every"]), "export_all_routes": True, "quality_report": False,
        }, config))
    return jobs


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


if __name__ == "__main__":
    main()
