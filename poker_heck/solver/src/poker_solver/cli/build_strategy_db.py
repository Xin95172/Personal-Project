"""依 manifest 訓練工作並把已訪問資訊集寫入 SQLite。"""
from __future__ import annotations
from argparse import ArgumentParser
import json
from pathlib import Path

from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.config import load_config
from poker_solver.solver_core.multiway_postflop_config import load_multiway_postflop_trainer
from poker_solver.solver_core.preflop_config import load_preflop_trainer
from poker_solver.solver_core.river_analysis import analyze_river_profile
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer
from poker_solver.solver_core.strategy_store import StrategyStore, export_heads_up_postflop_infosets, export_multiway_postflop_infosets, export_preflop_infosets, export_river_tree
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer


def main() -> None:
    parser = ArgumentParser(description="執行 manifest 並建立 SQLite 策略庫")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    if not jobs:
        parser.error("manifest.jobs 不可為空")
    store = StrategyStore(_resolve(manifest_path.parent, manifest["strategy_db"]))
    for index, job in enumerate(jobs, 1):
        _run_job(store, manifest_path.parent, job, index, len(jobs))
    print(f"策略庫完成：{store.path}")


def _run_job(store: StrategyStore, base: Path, job: dict, index: int, total: int, _parser=None) -> None:
    game_type = job["game_type"]
    config_path = _resolve(base, job["config"])
    checkpoint = _resolve(base, job["checkpoint"])
    trainer = load_checkpoint(checkpoint) if checkpoint.exists() else _new_trainer(game_type, config_path)
    print(f"[{index}/{total}] {game_type}：{config_path.name}", flush=True)
    _train(trainer, int(job["iterations"]), checkpoint, int(job["checkpoint_every"]))
    if not job["export_all_routes"]:
        print("  已儲存 checkpoint；依設定不匯出策略。", flush=True)
        return
    profile, version = job.get("range_profile_id", "default"), job.get("solver_version", f"{game_type}-v1")
    if game_type == "river_heads_up":
        report = export_river_tree(store, trainer, range_profile_id=profile, solver_version=version, quality=analyze_river_profile(trainer) if job.get("quality_report", True) else None)
    elif game_type in {"flop_heads_up", "turn_heads_up"}:
        report = export_heads_up_postflop_infosets(store, trainer, game_type=game_type, range_profile_id=profile, solver_version=version)
    elif game_type == "preflop_8max":
        report = export_preflop_infosets(store, trainer, range_profile_id=profile, solver_version=version)
    elif game_type == "multiway_postflop":
        report = export_multiway_postflop_infosets(store, trainer, range_profile_id=profile, solver_version=version)
    else:
        raise ValueError(f"不支援 game_type：{game_type}")
    print(f"  已匯出 {report.stored_infosets} 個已訪問資訊集。", flush=True)


def _new_trainer(game_type: str, config_path: Path):
    if game_type == "preflop_8max":
        return load_preflop_trainer(config_path)
    if game_type == "multiway_postflop":
        return load_multiway_postflop_trainer(config_path)
    trainer = load_config(config_path).create_trainer()
    expected = {"river_heads_up": RiverMCCFRTrainer, "flop_heads_up": FlopMCCFRTrainer, "turn_heads_up": TurnMCCFRTrainer}[game_type]
    if type(trainer) is not expected:
        raise ValueError(f"{game_type} 的 board／mode 與 trainer 不符")
    return trainer


def _train(trainer, iterations: int, checkpoint: Path, checkpoint_every: int) -> None:
    if iterations <= 0 or checkpoint_every <= 0:
        raise ValueError("iterations 與 checkpoint_every 必須為正數")
    for completed in range(0, iterations, checkpoint_every):
        trainer.train(min(checkpoint_every, iterations - completed))
        save_checkpoint(trainer, checkpoint)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    main()
