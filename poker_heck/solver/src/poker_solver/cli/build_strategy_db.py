"""依 manifest 訓練各 pack，並匯出策略至 SQLite。"""
from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from pickle import UnpicklingError

from tqdm import tqdm

from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.config import load_config
from poker_solver.solver_core.multiway_postflop_config import load_multiway_postflop_trainer
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.preflop_config import load_preflop_trainer
from poker_solver.solver_core.river_analysis import analyze_river_profile
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer
from poker_solver.solver_core.strategy_store import StrategyStore, export_heads_up_postflop_infosets, export_multiway_postflop_root_infosets, export_preflop_infosets, export_river_tree
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer


def main() -> None:
    parser = ArgumentParser(description="依 manifest 訓練並建立策略資料庫")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    run_manifest(args.manifest)


def run_manifest(path: Path) -> None:
    manifest_path = path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    if not jobs:
        raise ValueError("manifest.jobs 不可為空")
    store = StrategyStore(_resolve(manifest_path.parent, manifest["strategy_db"]))
    with tqdm(jobs, desc="訓練 pack", unit="pack", dynamic_ncols=True, position=1) as packs:
        for index, job in enumerate(packs, 1):
            mode = job.get("traverser_mode")
            mode_label = "單 traverser" if mode == "single_random" else "全玩家" if mode == "all_players" else ""
            packs.set_postfix_str(f"訓練：{Path(job['config']).name} {mode_label}", refresh=False)
            _run_job(store, manifest_path.parent, job, index, len(jobs), progress=None)
    print(f"策略資料庫已建立：{store.path}")


def _run_job(store: StrategyStore, base: Path, job: dict, index: int, total: int, _parser=None, progress=None) -> None:
    game_type = job["game_type"]
    config_path = _resolve(base, job["config"])
    checkpoint = _resolve(base, job["checkpoint"])
    profile = job.get("range_profile_id", "default")
    version = job.get("solver_version", f"{game_type}-v1")
    export_key = _pack_export_key(config_path, game_type, profile, version)
    if job["export_all_routes"] and store.has_completed_pack_export(export_key):
        if job.get("cleanup_checkpoint_after_export", False):
            checkpoint.unlink(missing_ok=True)
        if progress is not None:
            progress.set_postfix_str(f"撌脰歲??綽?{config_path.name}", refresh=True)
        return
    expected_trainer = _new_trainer(game_type, config_path)
    if checkpoint.exists():
        try:
            loaded_trainer = load_checkpoint(checkpoint)
        except (EOFError, UnpicklingError, OSError, ValueError, AttributeError, ImportError):
            # 寫檔中斷可能留下不完整 pickle；它無法安全恢復，改由本 pack
            # 的初始局面重新訓練，並移除壞檔以免下次再次停止。
            checkpoint.unlink(missing_ok=True)
            trainer = expected_trainer
        else:
            trainer = loaded_trainer if _checkpoint_matches_config(loaded_trainer, expected_trainer) else expected_trainer
    else:
        trainer = expected_trainer
    _train(trainer, int(job["iterations"]), checkpoint, int(job["checkpoint_every"]), description=f"{index}/{total} {config_path.name}")
    if not job["export_all_routes"]:
        return
    infoset_count = _export_count(game_type, trainer)
    if store.is_pack_export_complete(export_key, trained_iterations=trainer.iterations_completed, infoset_count=infoset_count):
        if job.get("cleanup_checkpoint_after_export", False):
            checkpoint.unlink(missing_ok=True)
        if progress is not None:
            progress.set_postfix_str(f"已跳過匯出：{config_path.name}", refresh=True)
        return
    if progress is not None:
        progress.set_postfix_str(f"匯出：{config_path.name}", refresh=True)
    with tqdm(total=infoset_count, desc="匯出 infoset", unit="node", dynamic_ncols=True, position=0, leave=False) as exported:
        with store.buffered_upserts(progress=exported):
            if game_type == "river_heads_up":
                export_river_tree(store, trainer, range_profile_id=profile, solver_version=version, quality=analyze_river_profile(trainer) if job.get("quality_report", True) else None)
            elif game_type in {"flop_heads_up", "turn_heads_up"}:
                export_heads_up_postflop_infosets(store, trainer, game_type=game_type, range_profile_id=profile, solver_version=version)
            elif game_type == "preflop_8max":
                export_preflop_infosets(store, trainer, range_profile_id=profile, solver_version=version)
            elif game_type == "multiway_postflop":
                export_multiway_postflop_root_infosets(store, trainer, range_profile_id=profile, solver_version=version)
            else:
                raise ValueError(f"不支援的 game_type：{game_type}")
    store.mark_pack_export_complete(export_key, trained_iterations=trainer.iterations_completed, infoset_count=infoset_count)
    if job.get("cleanup_checkpoint_after_export", False):
        checkpoint.unlink(missing_ok=True)


def _new_trainer(game_type: str, config_path: Path):
    if game_type == "preflop_8max":
        return load_preflop_trainer(config_path)
    if game_type == "multiway_postflop":
        return load_multiway_postflop_trainer(config_path)
    trainer = load_config(config_path).create_trainer()
    expected = {"river_heads_up": RiverMCCFRTrainer, "flop_heads_up": FlopMCCFRTrainer, "turn_heads_up": TurnMCCFRTrainer}[game_type]
    if type(trainer) is not expected:
        raise ValueError(f"{game_type} 的 trainer 類型不符")
    return trainer


def _checkpoint_matches_config(trainer, expected) -> bool:
    """只在 checkpoint 與目前 pack 完全相容時才接續訓練。

    pack 產生規則更新後，舊的檔名序號可能指向不同局面；若仍直接讀取
    同名 checkpoint，會把舊策略錯套到新局面。這裡保留相容的 checkpoint，
    但安全地略過不相容者。
    """
    if type(trainer) is not type(expected):
        return False
    for attribute in ("initial_state", "ranges", "sizing_policy", "traverser_mode", "stack_bb"):
        if hasattr(expected, attribute) and getattr(trainer, attribute, object()) != getattr(expected, attribute):
            return False
    # multiway infoset v2 新增了剩餘籌碼、總投入與加注資格；含有舊 key
    # 的 checkpoint 不能與新版節點混用，必須從該 pack 的初始局面重建。
    if isinstance(expected, MultiwayPostflopMCCFRTrainer):
        return all(len(key) == 11 for key in trainer.infosets)
    return True


def _train(trainer, iterations: int, checkpoint: Path, checkpoint_every: int, *, description: str) -> None:
    if iterations <= 0 or checkpoint_every <= 0:
        raise ValueError("iterations 與 checkpoint_every 必須是正整數")
    completed_before = min(int(getattr(trainer, "iterations_completed", 0)), iterations)
    remaining = iterations - completed_before
    with tqdm(total=iterations, initial=completed_before, desc="目前 pack 迭代", unit="iter", leave=False, dynamic_ncols=True, position=0) as bar:
        for completed in range(1, remaining + 1):
            trainer.train(1)
            bar.update(1)
            bar.set_postfix_str(description, refresh=False)
            if completed % checkpoint_every == 0 or completed == remaining:
                save_checkpoint(trainer, checkpoint)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _pack_export_key(config_path: Path, game_type: str, range_profile_id: str, solver_version: str) -> str:
    """設定內容或求解版本變動時，強制重新匯出該 pack。"""
    payload = {
        "config_path": str(config_path.resolve()),
        "config_digest": sha256(config_path.read_bytes()).hexdigest(),
        "game_type": game_type,
        "range_profile_id": range_profile_id,
        "solver_version": solver_version,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _export_count(game_type: str, trainer) -> int:
    if game_type != "multiway_postflop":
        return len(trainer.infosets)
    state = trainer.initial_state
    actor = state.current_player
    return sum(
        key[0] == actor.value and tuple(key[3]) == state.board and tuple(key[-1]) == ()
        for key in trainer.infosets
    ) if actor is not None else 0


if __name__ == "__main__":
    main()
