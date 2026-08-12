"""從 JSON 設定訓練、續訓與匯出 8-Max preflop 策略。"""

from argparse import ArgumentParser
from pathlib import Path

from poker_solver.engine.preflop_policy import format_action
from poker_solver.engine.table import Position, create_8max_preflop
from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.preflop_config import load_preflop_trainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.reporting import export_preflop_strategy_csv


def main() -> None:
    parser = ArgumentParser(description="訓練 8-Max preflop multiway MCCFR profile")
    parser.add_argument("config", nargs="?", help="JSON 訓練設定檔")
    parser.add_argument("--iterations", type=int, required=True, help="本次訓練 iteration 數")
    parser.add_argument("--checkpoint", type=Path, help="訓練後儲存 checkpoint 的路徑")
    parser.add_argument("--resume", type=Path, help="載入既有 checkpoint 後續訓")
    parser.add_argument("--export", type=Path, help="匯出 UTG 根節點策略 CSV 的路徑")
    arguments = parser.parse_args()

    if arguments.resume:
        trainer = load_checkpoint(arguments.resume)
        if not isinstance(trainer, MultiwayPreflopMCCFRTrainer):
            parser.error("--resume 必須是 preflop checkpoint")
    elif arguments.config:
        trainer = load_preflop_trainer(arguments.config)
    else:
        parser.error("請提供 config，或使用 --resume 載入 checkpoint")

    stats = trainer.train(arguments.iterations)
    print(f"完成 {stats.iterations} 次 iteration；累計 {stats.total_iterations} 次；資訊集 {stats.infosets} 個")
    print(f"速度：{stats.iterations_per_second:.0f} iterations / 秒")
    print(f"平均正 regret：{stats.mean_positive_regret:.6f}（這是多人的近似策略 profile，不是已收斂的 GTO 證明）")

    root = create_8max_preflop(stack_bb=trainer.stack_bb)
    hero_cards = trainer.ranges[Position.UTG].combos[0].cards
    strategy = trainer.strategy_for(root, Position.UTG, hero_cards)
    print(f"UTG {hero_cards[0]}{hero_cards[1]} 的根節點平均策略：")
    for action, probability in strategy.items():
        print(f"{format_action(root, action):24} {probability:.2%}")
    if arguments.checkpoint:
        print(f"checkpoint：{save_checkpoint(trainer, arguments.checkpoint)}")
    if arguments.export:
        print(f"策略 CSV：{export_preflop_strategy_csv(trainer, root, Position.UTG, hero_cards, arguments.export)}")


if __name__ == "__main__":
    main()
