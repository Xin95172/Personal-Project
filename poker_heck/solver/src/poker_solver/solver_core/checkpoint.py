"""本機訓練 checkpoint 的保存與載入。"""

from pathlib import Path
from pickle import dump, load

from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer

SupportedTrainer = RiverMCCFRTrainer | TurnMCCFRTrainer | FlopMCCFRTrainer | MultiwayPreflopMCCFRTrainer | MultiwayPostflopMCCFRTrainer

def save_checkpoint(trainer: SupportedTrainer, path: str | Path) -> Path:
    """保存 trainer、資訊集與隨機數狀態，供後續續訓。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:
        dump(trainer, file)
    return target


def load_checkpoint(path: str | Path) -> SupportedTrainer:
    """讀取由本專案相同版本建立的 checkpoint。"""
    with Path(path).open("rb") as file:
        trainer = load(file)
    if not isinstance(trainer, (RiverMCCFRTrainer, TurnMCCFRTrainer, FlopMCCFRTrainer, MultiwayPreflopMCCFRTrainer, MultiwayPostflopMCCFRTrainer)):
        raise ValueError("checkpoint does not contain a supported trainer")
    return trainer
