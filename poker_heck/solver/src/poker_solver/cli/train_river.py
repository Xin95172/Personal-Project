"""由 JSON 設定訓練 River MCCFR solver。"""

from argparse import ArgumentParser
from pathlib import Path

from poker_solver.engine.chance import TurnToRiverScenario
from poker_solver.engine.postflop_game import create_flop_game, create_turn_game
from poker_solver.engine.river_game import Player, create_river_game
from poker_solver.solver_core.checkpoint import load_checkpoint, save_checkpoint
from poker_solver.solver_core.config import load_config
from poker_solver.solver_core.reporting import action_probabilities, export_strategy_csv
from poker_solver.solver_core.river_analysis import analyze_river_profile, write_river_quality_report
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer, TurnRiverMCCFRTrainer
from poker_solver.solver_core.strategy_store import StrategyStore, store_river_root_strategy
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer


def main() -> None:
    parser = ArgumentParser(description="訓練 River heads-up MCCFR solver")
    parser.add_argument("config", nargs="?", help="JSON 設定檔路徑；續訓時可省略")
    parser.add_argument("--iterations", type=int, required=True, help="本次訓練 iterations")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint 輸出路徑")
    parser.add_argument("--resume", type=Path, help="既有 checkpoint 路徑")
    parser.add_argument("--export", type=Path, help="匯出 OOP 根節點策略的 CSV 路徑")
    parser.add_argument("--river-card", help="turn chance 模式匯出策略時指定的 river card")
    parser.add_argument("--quality-report", type=Path, help="輸出 River profile EV / best-response / NashConv JSON")
    parser.add_argument("--progress-every", type=int, default=100, help="每隔多少 iteration 顯示一次進度；設為 0 可關閉")
    parser.add_argument("--strategy-db", type=Path, help="將已訓練的 river 根節點策略寫入 SQLite 資料庫")
    parser.add_argument("--range-profile", default="default", help="策略庫使用的 range profile 識別名稱")
    parser.add_argument("--solver-version", default="river-v1", help="策略庫使用的 solver 版本識別名稱")
    arguments = parser.parse_args()

    if arguments.resume:
        trainer = load_checkpoint(arguments.resume)
    elif arguments.config:
        trainer = load_config(arguments.config).create_trainer()
    else:
        parser.error("請提供 config，或使用 --resume 指定 checkpoint")

    if arguments.progress_every < 0:
        parser.error("--progress-every 必須是 0 或正整數")
    remaining = arguments.iterations
    chunk_size = remaining if arguments.progress_every == 0 else min(arguments.progress_every, remaining)
    stats = None
    while remaining:
        batch = min(chunk_size, remaining)
        stats = trainer.train(batch)
        remaining -= batch
        if arguments.progress_every:
            print(
                f"訓練進度：{arguments.iterations - remaining}/{arguments.iterations} "
                f"iteration，資訊集 {stats.infosets} 個。",
                flush=True,
            )
    assert stats is not None
    print(f"完成 {stats.iterations} 次 iteration；累積 {stats.total_iterations} 次；資訊集 {stats.infosets} 個。")
    print(f"速度：{stats.iterations_per_second:.0f} iterations / 秒")
    print(f"平均正 regret：{stats.mean_positive_regret:.6f}（僅供觀察訓練趨勢，不等同 exploitability）")

    if arguments.checkpoint:
        print(f"checkpoint：{save_checkpoint(trainer, arguments.checkpoint)}")

    report = None
    if arguments.quality_report:
        if type(trainer) is not RiverMCCFRTrainer:
            parser.error("--quality-report 目前只支援固定五張公共牌的 RiverMCCFRTrainer")
        report = analyze_river_profile(trainer)
        output = write_river_quality_report(report, arguments.quality_report)
        print(
            f"River NashConv：{report.nash_conv_bb:.6f} BB "
            f"（初始底池 {100 * report.nash_conv_initial_pot_fraction:.4f}%；"
            f"有效籌碼 {100 * report.nash_conv_effective_stack_fraction:.4f}%）；品質報告：{output}"
        )

    if arguments.strategy_db:
        if type(trainer) is not RiverMCCFRTrainer:
            parser.error("--strategy-db 目前只支援 RiverMCCFRTrainer")
        oop_combo = trainer.oop_range.combos[0]
        ip_combo = trainer.ip_range.combos[0]
        state = create_river_game(
            trainer.board,
            oop_combo.cards,
            ip_combo.cards,
            initial_pot_bb=trainer.initial_pot_bb,
            effective_stack_bb=trainer.effective_stack_bb,
        )
        stored = store_river_root_strategy(
            StrategyStore(arguments.strategy_db),
            trainer,
            state,
            Player.OOP,
            range_profile_id=arguments.range_profile,
            solver_version=arguments.solver_version,
            quality=report,
        )
        print(f"策略資料庫：{arguments.strategy_db}（key：{stored.strategy_key[:12]}…）")

    if arguments.export:
        oop_combo = trainer.oop_range.combos[0]
        ip_combo = trainer.ip_range.combos[0]
        if isinstance(trainer, FlopMCCFRTrainer):
            state = create_flop_game(
                trainer.flop_board,
                oop_combo.cards,
                ip_combo.cards,
                initial_pot_bb=trainer.initial_pot_bb,
                effective_stack_bb=trainer.effective_stack_bb,
            )
        elif isinstance(trainer, TurnMCCFRTrainer):
            state = create_turn_game(
                trainer.turn_board,
                oop_combo.cards,
                ip_combo.cards,
                initial_pot_bb=trainer.initial_pot_bb,
                effective_stack_bb=trainer.effective_stack_bb,
            )
        elif isinstance(trainer, TurnRiverMCCFRTrainer):
            if not arguments.river_card:
                parser.error("turn chance 模式匯出策略時必須提供 --river-card")
            state = TurnToRiverScenario(
                turn_board=trainer.turn_board,
                oop_hole_cards=oop_combo.cards,
                ip_hole_cards=ip_combo.cards,
                pot_bb=trainer.initial_pot_bb,
                effective_stack_bb=trainer.effective_stack_bb,
            ).deal_river(arguments.river_card)
        else:
            state = create_river_game(
                trainer.board,
                oop_combo.cards,
                ip_combo.cards,
                initial_pot_bb=trainer.initial_pot_bb,
                effective_stack_bb=trainer.effective_stack_bb,
            )
        strategy = trainer.strategy_for(state, Player.OOP)
        for action, probability in action_probabilities(strategy, state):
            print(f"{action:45} {probability:.2%}")
        print(f"策略 CSV：{export_strategy_csv(trainer, state, Player.OOP, arguments.export)}")


if __name__ == "__main__":
    main()
