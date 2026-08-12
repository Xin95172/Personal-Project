"""以 JSON 設定訓練 8-Max 多人 postflop MCCFR。"""

from argparse import ArgumentParser
from pathlib import Path

from poker_solver.engine.multiway_postflop_policy import format_multiway_postflop_action_as_pot_ratio
from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.multiway_postflop_config import load_multiway_postflop_trainer
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.reporting import export_multiway_postflop_strategy_csv


def main() -> None:
    parser = ArgumentParser(description="訓練 8-Max 多人 postflop MCCFR 模型")
    parser.add_argument("config", nargs="?", help="訓練設定 JSON")
    parser.add_argument("--iterations", type=int, required=True, help="本次訓練 iteration 數")
    parser.add_argument("--checkpoint", type=Path, help="輸出 checkpoint 路徑")
    parser.add_argument("--resume", type=Path, help="讀取既有 checkpoint")
    parser.add_argument("--export", type=Path, help="匯出根節點策略 CSV")
    arguments = parser.parse_args()
    if arguments.resume:
        trainer = load_checkpoint(arguments.resume)
        if not isinstance(trainer, MultiwayPostflopMCCFRTrainer):
            parser.error("--resume 不是多人 postflop checkpoint")
    elif arguments.config:
        trainer = load_multiway_postflop_trainer(arguments.config)
    else:
        parser.error("必須提供 config 或 --resume")

    stats = trainer.train(arguments.iterations)
    print(f"完成 {stats.iterations} 次 iteration，累計 {stats.total_iterations} 次；資訊集 {stats.infosets} 個。")
    root = trainer.initial_state
    assert root.current_player is not None
    hero = root.current_player
    cards = trainer.ranges[hero].combos[0].cards
    strategy = trainer.strategy_for(root, hero, cards)
    print(f"{hero.value.upper()} {cards[0]}{cards[1]} 的根節點策略：")
    for action, probability in strategy.items():
        print(f"{format_multiway_postflop_action_as_pot_ratio(root, action):32} {probability:.2%}")
    if arguments.checkpoint:
        print(f"checkpoint：{save_checkpoint(trainer, arguments.checkpoint)}")
    if arguments.export:
        print(f"CSV：{export_multiway_postflop_strategy_csv(trainer, root, hero, cards, arguments.export)}")


if __name__ == "__main__":
    main()
