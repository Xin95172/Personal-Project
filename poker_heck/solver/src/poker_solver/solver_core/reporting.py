"""將訓練後策略匯出為可閱讀的 CSV。"""

import csv
from pathlib import Path

from poker_solver.engine.money import format_bb
from poker_solver.engine.river_game import Action, Player, RiverGameState, format_action_as_pot_ratio
from poker_solver.engine.preflop_policy import format_action as format_preflop_action
from poker_solver.engine.multiway_postflop_policy import format_multiway_postflop_action_as_pot_ratio
from poker_solver.engine.table import MultiwayPostflopState, Position, PreflopState
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer


def export_strategy_csv(
    trainer: RiverMCCFRTrainer,
    state: RiverGameState,
    player: Player,
    path: str | Path,
) -> Path:
    """匯出單一資訊集的平均策略，動作以 pot 比例呈現。"""
    strategy = trainer.strategy_for(state, player)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("position", "board", "hole_cards", "pot", "action", "probability"))
        writer.writeheader()
        for action, probability in strategy.items():
            writer.writerow(
                {
                    "position": player.value,
                    "board": " ".join(state.board),
                    "hole_cards": " ".join(state.player_state(player).hole_cards),
                    "pot": format_bb(state.pot),
                    "action": format_action_as_pot_ratio(state, action),
                    "probability": f"{probability:.10f}",
                }
            )
    return target


def action_probabilities(strategy: dict[Action, float], state: RiverGameState) -> list[tuple[str, float]]:
    """提供 Notebook、CLI 共用的底池比例策略表資料。"""
    return [(format_action_as_pot_ratio(state, action), probability) for action, probability in strategy.items()]


def export_preflop_strategy_csv(
    trainer: MultiwayPreflopMCCFRTrainer,
    state: PreflopState,
    player: Position,
    hole_cards: tuple[str, str],
    path: str | Path,
) -> Path:
    """匯出指定 preflop 資訊集的平均策略；所有尺寸都以 BB 顯示。"""
    strategy = trainer.strategy_for(state, player, hole_cards)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("position", "hole_cards", "pot", "action", "probability"))
        writer.writeheader()
        for action, probability in strategy.items():
            writer.writerow({"position": player.value, "hole_cards": " ".join(hole_cards), "pot": format_bb(state.pot), "action": format_preflop_action(state, action), "probability": f"{probability:.10f}"})
    return target


def export_multiway_postflop_strategy_csv(
    trainer: MultiwayPostflopMCCFRTrainer,
    state: MultiwayPostflopState,
    player: Position,
    hole_cards: tuple[str, str],
    path: str | Path,
) -> Path:
    """匯出多人 flop/turn/river 指定資訊集的策略，下注額以底池比例呈現。"""
    strategy = trainer.strategy_for(state, player, hole_cards)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("position", "street", "board", "hole_cards", "pot", "action", "probability"))
        writer.writeheader()
        for action, probability in strategy.items():
            writer.writerow(
                {
                    "position": player.value,
                    "street": state.street,
                    "board": " ".join(state.board),
                    "hole_cards": " ".join(hole_cards),
                    "pot": format_bb(state.pot),
                    "action": format_multiway_postflop_action_as_pot_ratio(state, action),
                    "probability": f"{probability:.10f}",
                }
            )
    return target
