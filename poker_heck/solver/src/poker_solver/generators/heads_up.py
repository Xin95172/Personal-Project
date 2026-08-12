"""依 heads-up 網格產生 flop、turn、river 訓練 pack。"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from itertools import product
from pathlib import Path
from typing import Any

from poker_solver.engine.board_catalog import iter_canonical_boards


def main() -> None:
    parser = ArgumentParser(description="產生 heads-up 全網格訓練 pack")
    parser.add_argument("grid", type=Path, help="網格與優先順序 JSON")
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
    print(f"已產生 {len(jobs)} 個 heads-up pack：{output_dir}")
    print(f"訓練 manifest：{manifest_path}")


def build_jobs(raw: dict[str, Any], output_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    stacks = tuple(int(value) for value in raw["stack_bb"])
    pots = {str(name): float(value) for name, value in raw["pot_profiles_bb"].items()}
    streets = tuple(str(value) for value in raw["streets"])
    boards = _boards_by_street(raw, streets)
    if not stacks or not pots or not streets or any(stack <= 0 for stack in stacks):
        raise ValueError("stack_bb, pot_profiles_bb, and streets must be non-empty")
    if any(street not in boards for street in streets) or any(street not in {"flop", "turn", "river"} for street in streets):
        raise ValueError("each street must be flop, turn, or river and provide board buckets")
    expected_cards = {"flop": 3, "turn": 4, "river": 5}
    if any(len(board) != expected_cards[street] for street in streets for board in boards[street].values()):
        raise ValueError("board bucket has an invalid card count")

    all_keys = [(street, stack, pot_name, board_name) for street, stack, pot_name in product(streets, stacks, pots) for board_name in boards[street]]
    ordered: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for tier in raw["priority_tiers"]:
        for key in _tier_matches(tier, all_keys):
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    ordered.extend(key for key in all_keys if key not in seen)

    iterations = int(raw["iterations_per_pack"])
    checkpoint_every = int(raw["checkpoint_every"])
    if iterations <= 0 or checkpoint_every <= 0:
        raise ValueError("iterations_per_pack and checkpoint_every must be positive")
    sizing = raw["sizing_policy"]
    seed_start = int(raw["seed_start"])
    job_options = raw["job_options"]
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, (street, stack, pot_name, board_name) in enumerate(ordered, start=seed_start):
        stem = f"hu_{index:03d}_{street}_{pot_name}_{board_name}_{stack}bb"
        config: dict[str, Any] = {
            "board": list(boards[street][board_name]), "initial_pot_bb": pots[pot_name], "effective_stack_bb": stack,
            "oop_range_spec": raw["oop_range_spec"], "ip_range_spec": raw["ip_range_spec"], "sizing_policy": sizing, "seed": index,
        }
        if street == "turn":
            config["mode"] = "turn"
        jobs.append(({
            "game_type": f"{street}_heads_up", "config": f"{output_dir.name}/{stem}.json",
            "range_profile_id": f"hu-grid-{street}-{pot_name}-{board_name}-{stack}bb-{raw['solver_version']}",
            "solver_version": raw["solver_version"], "iterations": iterations,
            "checkpoint": f"{raw['checkpoint_dir']}/{stem}.pkl", "checkpoint_every": checkpoint_every,
            "quality_report": street == "river" and bool(job_options["quality_report_on_river"]), "export_all_routes": bool(job_options["export_all_routes"]),
        }, config))
    return jobs


def _tier_matches(tier: dict[str, Any], keys: list[tuple[str, int, str, str]]) -> list[tuple[str, int, str, str]]:
    wanted_streets = set(tier.get("streets", []))
    wanted_stacks = set(tier.get("stack_bb", []))
    wanted_pots = set(tier.get("pot_profiles", []))
    wanted_boards = set(tier.get("board_buckets", []))
    return [key for key in keys if (not wanted_streets or key[0] in wanted_streets) and (not wanted_stacks or key[1] in wanted_stacks) and (not wanted_pots or key[2] in wanted_pots) and (not wanted_boards or key[3] in wanted_boards)]


def _boards_by_street(raw: dict[str, Any], streets: tuple[str, ...]) -> dict[str, dict[str, tuple[str, ...]]]:
    if raw["board_source"] == "canonical":
        limit = raw["max_canonical_boards_per_street"]
        return {
            street: {f"canonical_{index:06d}": board for index, board in enumerate(iter_canonical_boards({"flop": 3, "turn": 4, "river": 5}[street], limit=limit), start=1)}
            for street in streets
        }
    return {str(street): {str(name): tuple(cards) for name, cards in values.items()} for street, values in raw["board_buckets"].items()}


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


if __name__ == "__main__":
    main()
