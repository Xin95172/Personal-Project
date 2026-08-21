"""產生 multiway postflop 訓練工作。"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from poker_solver.engine.board_catalog import iter_canonical_boards
from poker_solver.engine.preflop_policy import PreflopSizingPolicy, abstract_actions
from poker_solver.engine.river_game import Action
from poker_solver.engine.table import apply_action, create_8max_preflop, is_terminal


class _RouteLimitReached(Exception):
    pass


def main() -> None:
    parser = ArgumentParser(description="產生 multiway 訓練工作")
    parser.add_argument("grid", type=Path)
    args = parser.parse_args()
    path = args.grid.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    output, manifest = _resolve(path.parent, raw["output_dir"]), _resolve(path.parent, raw["manifest"])
    print("正在列舉 preflop 路線與 canonical flop…", flush=True)
    jobs = build_jobs(raw, output)
    output.mkdir(parents=True, exist_ok=True)
    for job, config in tqdm(jobs, desc="寫入 pack", unit="pack", dynamic_ncols=True):
        (output / Path(job["config"]).name).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"strategy_db": str(_resolve(path.parent, raw["strategy_db"])), "jobs": [job for job, _ in jobs]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已產生 {len(jobs)} 個 multiway pack：{output}")


def build_jobs(raw: dict[str, Any], output: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if raw["solve_scope"] != "flop_full_tree":
        raise ValueError("multiway grid solve_scope 必須是 flop_full_tree")
    stacks = tuple(int(value) for value in raw["stack_bb"])
    counts = set(raw["player_counts"])
    boards = _boards(raw)
    policy = PreflopSizingPolicy(**raw["preflop_route_policy"])
    routes = [(stack, route) for stack in stacks for route in _routes(stack, policy, counts, raw["preflop_route_offset_per_stack"], raw["max_preflop_routes_per_stack"])]
    jobs = []
    profile = f"top_percent_{float(raw['range_spec']['percent']):g}" if raw["range_spec"]["kind"] == "top_percent" else str(raw["range_spec"]["kind"])
    combinations = [
        (stack, route_id, player_count, actions, board_id, board)
        for stack, (route_id, player_count, actions) in routes
        for board_id, board in boards.items()
    ]
    for seed, (stack, route_id, player_count, actions, board_id, board) in enumerate(combinations, start=int(raw["seed_start"])):
        stem = f"mw_{seed:06d}_{route_id}_{board_id}_{stack}bb"
        requested_mode = str(raw["traverser_mode"])
        resolved_mode = _resolve_traverser_mode(raw, requested_mode, player_count)
        config = {"solve_scope": "flop_full_tree", "stack_bb": stack, "board": list(board), "completed_street_actions": [], "preflop_actions": actions, "range_spec": raw["range_spec"], "sizing_policy": raw["sizing_policy"], "traverser_mode": resolved_mode, "requested_traverser_mode": requested_mode, "postflop_player_count": player_count, "seed": seed}
        options = raw["job_options"]
        jobs.append(({"game_type": "multiway_postflop", "config": f"{output.name}/{stem}.json", "range_profile_id": profile, "solver_version": f"{raw['solver_version']}-{_mode_label(resolved_mode)}", "iterations": int(raw["iterations_per_pack"]), "checkpoint": f"{raw['checkpoint_dir']}/{stem}.pkl", "checkpoint_every": int(raw["checkpoint_every"]), "export_all_routes": bool(options["export_all_routes"]), "quality_report": bool(options["quality_report"]), "cleanup_checkpoint_after_export": bool(options.get("cleanup_checkpoint_after_export", False)), "traverser_mode": resolved_mode}, config))
    return jobs


def _routes(stack: int, policy: PreflopSizingPolicy, counts: set[int], offset: int, limit: int | None):
    if offset < 0 or (limit is not None and limit <= 0):
        raise ValueError("preflop route offset 必須非負，limit 必須是正數或 null")
    result, matched = [], 0

    def walk(state):
        nonlocal matched
        if is_terminal(state):
            active_players = [player for player in state.players if not player.folded]
            # 所有人都已全下時，後面只剩發牌與攤牌，沒有 postflop 決策可訓練。
            has_postflop_decision = any(not player.all_in for player in active_players)
            if not state.hand_ended and has_postflop_decision and len(active_players) in counts:
                matched += 1
                if matched > offset:
                    active = len(active_players)
                    result.append((f"p{active}_{matched:06d}", active, [_action(action) for action in state.action_history]))
                    if limit is not None and len(result) >= limit:
                        raise _RouteLimitReached
            return
        for action in abstract_actions(state, policy):
            walk(apply_action(state, action))

    try:
        walk(create_8max_preflop(stack_bb=stack))
    except _RouteLimitReached:
        pass
    return result


def _action(action: Action) -> dict[str, Any]:
    result = {"kind": action.kind.value}
    if action.amount is not None:
        result["amount_bb"] = action.amount / 100
    return result


def _resolve_traverser_mode(raw: dict[str, Any], requested_mode: str, player_count: int) -> str:
    if requested_mode in {"single_random", "all_players"}:
        return requested_mode
    if requested_mode != "adaptive":
        raise ValueError("traverser_mode 必須是 adaptive、single_random 或 all_players")
    threshold = int(raw["all_players_max_active_players"])
    if not 2 <= threshold <= 8:
        raise ValueError("all_players_max_active_players 必須介於 2 與 8")
    return "all_players" if player_count <= threshold else "single_random"


def _mode_label(mode: str) -> str:
    return "single-traverser" if mode == "single_random" else "all-traversers"


def _boards(raw: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    if raw["board_source"] == "explicit":
        return {name: tuple(cards) for name, cards in raw["board_buckets"].items()}
    return {f"canonical_{index:06d}": board for index, board in enumerate(iter_canonical_boards(3, limit=raw["max_canonical_boards"]), start=1)}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    main()
