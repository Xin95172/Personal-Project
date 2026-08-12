"""依 manifest 批次訓練多種局面，寫入統一 SQLite 預訓練策略庫。"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from poker_solver.engine.postflop_game import create_flop_game, create_turn_game
from poker_solver.engine.river_game import Player, create_river_game
from poker_solver.engine.table import Position, create_8max_preflop
from poker_solver.solver_core.config import load_config
from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.multiway_postflop_config import load_multiway_postflop_trainer
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.preflop_config import load_preflop_trainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.river_analysis import analyze_river_profile
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer
from poker_solver.solver_core.strategy_store import (
    StrategyStore,
    export_river_tree,
    export_preflop_infosets,
    export_multiway_postflop_infosets,
    export_heads_up_postflop_infosets,
    store_multiway_postflop_root_strategy,
    store_preflop_root_strategy,
    store_river_root_strategy,
    store_heads_up_postflop_root_strategy,
)


def main() -> None:
    parser = ArgumentParser(description="批次建立預訓練撲克策略 SQLite 資料庫")
    parser.add_argument("manifest", type=Path, help="策略庫建置 manifest JSON")
    arguments = parser.parse_args()
    manifest_path = arguments.manifest.resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = StrategyStore(_resolve(manifest_path.parent, raw["strategy_db"]))
    jobs = raw.get("jobs", [])
    if not jobs:
        parser.error("manifest 至少需要一個 jobs 項目")

    for index, job in enumerate(jobs, start=1):
        _run_job(store, manifest_path.parent, job, index, len(jobs), parser)
    print(f"策略資料庫完成：{store.path}")


def _run_job(store: StrategyStore, base: Path, job: dict[str, object], index: int, total: int, parser: ArgumentParser) -> None:
    game_type = str(job.get("game_type", "river_heads_up"))
    config_path = _resolve(base, str(job["config"]))
    iterations = int(job["iterations"])
    if iterations <= 0:
        parser.error(f"job {index} 的 iterations 必須是正整數")
    range_profile = str(job.get("range_profile_id", "default"))
    solver_version = str(job.get("solver_version", f"{game_type}-v1"))
    checkpoint_value = job.get("checkpoint")
    checkpoint_path = _resolve(base, str(checkpoint_value)) if checkpoint_value else None
    checkpoint_every = int(job.get("checkpoint_every", 1000))
    print(f"[{index}/{total}] {game_type}：{config_path.name}，{iterations} iteration", flush=True)

    if game_type == "river_heads_up":
        trainer = load_checkpoint(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else load_config(config_path).create_trainer()
        if type(trainer) is not RiverMCCFRTrainer:
            parser.error(f"job {index} 的 config 必須是五張 board river 設定")
        _train_with_progress(trainer, iterations, checkpoint_path, checkpoint_every)
        oop = trainer.oop_range.combos[0]
        ip = next((combo for combo in trainer.ip_range.combos if not (set(combo.cards) & set(oop.cards))), None)
        if ip is None:
            parser.error(f"job {index} 的 OOP/IP ranges 沒有可相容 combo")
        state = create_river_game(trainer.board, oop.cards, ip.cards, initial_pot_bb=trainer.initial_pot_bb, effective_stack_bb=trainer.effective_stack_bb)
        stored = store_river_root_strategy(
            store, trainer, state, Player.OOP, range_profile_id=range_profile, solver_version=solver_version,
            quality=analyze_river_profile(trainer) if bool(job.get("quality_report", True)) else None,
        )
        if bool(job.get("export_all_routes", True)):
            report = export_river_tree(
                store, trainer, range_profile_id=range_profile, solver_version=solver_version,
                quality=analyze_river_profile(trainer) if bool(job.get("quality_report", True)) else None,
            )
            print(
                f"  river 路徑覆蓋：{report.stored_infosets}/{report.reachable_infosets} "
                f"（未訓練 {report.unvisited_infosets}）",
                flush=True,
            )
    elif game_type == "preflop_8max":
        trainer = load_checkpoint(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else load_preflop_trainer(config_path)
        if not isinstance(trainer, MultiwayPreflopMCCFRTrainer):
            parser.error(f"job {index} 的 checkpoint 不是 8-Max preflop trainer")
        _train_with_progress(trainer, iterations, checkpoint_path, checkpoint_every)
        state = create_8max_preflop(stack_bb=trainer.stack_bb)
        cards = trainer.ranges[Position.UTG].combos[0].cards
        stored = store_preflop_root_strategy(store, trainer, state, Position.UTG, cards, range_profile_id=range_profile, solver_version=solver_version)
        if bool(job.get("export_all_routes", True)):
            report = export_preflop_infosets(store, trainer, range_profile_id=range_profile, solver_version=solver_version)
            print(f"  preflop 已訓練路徑匯出：{report.stored_infosets} 個", flush=True)
    elif game_type in {"flop_heads_up", "turn_heads_up"}:
        trainer = load_checkpoint(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else load_config(config_path).create_trainer()
        expected_type = FlopMCCFRTrainer if game_type == "flop_heads_up" else TurnMCCFRTrainer
        if type(trainer) is not expected_type:
            expected_board_cards = 3 if game_type == "flop_heads_up" else 4
            parser.error(f"job {index} must use a {expected_board_cards}-card board for {game_type}")
        _train_with_progress(trainer, iterations, checkpoint_path, checkpoint_every)
        oop = trainer.oop_range.combos[0]
        ip = next((combo for combo in trainer.ip_range.combos if not (set(combo.cards) & set(oop.cards))), None)
        if ip is None:
            parser.error(f"job {index} cannot find non-overlapping OOP/IP range combo")
        state = (
            create_flop_game(trainer.flop_board, oop.cards, ip.cards, initial_pot_bb=trainer.initial_pot_bb, effective_stack_bb=trainer.effective_stack_bb)
            if game_type == "flop_heads_up"
            else create_turn_game(trainer.turn_board, oop.cards, ip.cards, initial_pot_bb=trainer.initial_pot_bb, effective_stack_bb=trainer.effective_stack_bb)
        )
        stored = store_heads_up_postflop_root_strategy(
            store, trainer, state, Player.OOP, game_type=game_type,
            range_profile_id=range_profile, solver_version=solver_version,
        )
        if bool(job.get("export_all_routes", True)):
            report = export_heads_up_postflop_infosets(
                store, trainer, game_type=game_type, range_profile_id=range_profile, solver_version=solver_version,
            )
            print(f"  {game_type} trained infosets exported: {report.stored_infosets}", flush=True)
    elif game_type == "multiway_postflop":
        trainer = load_checkpoint(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else load_multiway_postflop_trainer(config_path)
        if not isinstance(trainer, MultiwayPostflopMCCFRTrainer):
            parser.error(f"job {index} 的 checkpoint 不是多人 postflop trainer")
        _train_with_progress(trainer, iterations, checkpoint_path, checkpoint_every)
        state = trainer.initial_state
        if state.current_player is None:
            parser.error(f"job {index} 的多人 postflop 根節點沒有行動者")
        cards = trainer.ranges[state.current_player].combos[0].cards
        stored = store_multiway_postflop_root_strategy(
            store, trainer, state, state.current_player, cards, range_profile_id=range_profile, solver_version=solver_version
        )
        if bool(job.get("export_all_routes", True)):
            report = export_multiway_postflop_infosets(store, trainer, range_profile_id=range_profile, solver_version=solver_version)
            print(f"  多人 postflop 已訓練路徑匯出：{report.stored_infosets} 個", flush=True)
    else:
        parser.error(f"job {index} 不支援 game_type：{game_type}")
    print(f"  已寫入：{stored.strategy_key[:12]}…，資訊集 {len(trainer.infosets)} 個", flush=True)


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _train_with_progress(trainer, iterations: int, checkpoint_path: Path | None, checkpoint_every: int) -> None:
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every 必須是正整數")
    remaining = iterations
    while remaining:
        batch = min(remaining, checkpoint_every)
        stats = trainer.train(batch)
        remaining -= batch
        if checkpoint_path:
            save_checkpoint(trainer, checkpoint_path)
        print(
            f"  進度：{iterations - remaining}/{iterations}；累計 {stats.total_iterations}；"
            f"資訊集 {stats.infosets}；{stats.iterations_per_second:.0f} it/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
