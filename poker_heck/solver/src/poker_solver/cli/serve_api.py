"""以 checkpoint 啟動 River solver API。"""

from argparse import ArgumentParser

import uvicorn

from poker_solver.api.app import create_app
from poker_solver.solver_core.checkpoint import load_checkpoint
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.strategy_store import StrategyStore


def main() -> None:
    parser = ArgumentParser(description="啟動 River solver FastAPI")
    parser.add_argument("checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--strategy-db", help="預訓練策略 SQLite 資料庫路徑")
    arguments = parser.parse_args()
    trainer = load_checkpoint(arguments.checkpoint)
    if not isinstance(trainer, (RiverMCCFRTrainer, MultiwayPostflopMCCFRTrainer)):
        parser.error("API 目前只支援 RiverMCCFRTrainer checkpoint")
    store = StrategyStore(arguments.strategy_db) if arguments.strategy_db else None
    uvicorn.run(create_app(trainer, strategy_store=store), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
