"""單一設定檔的完整訓練入口。"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from poker_solver.cli.build_strategy_db import run_manifest
from poker_solver.generators import conditional_subgames, heads_up, multiway, preflop


def main() -> None:
    parser = ArgumentParser(description="從設定檔產生訓練 pack 並立刻建立策略資料庫")
    parser.add_argument("config", type=Path, help="configs 目錄中的 JSON 設定檔")
    parser.add_argument("--generate-only", action="store_true", help="只產生 manifest 與 pack，不開始訓練")
    args = parser.parse_args()
    config = args.config.resolve()
    raw = json.loads(config.read_text(encoding="utf-8"))
    _generate(raw, config)
    if not args.generate_only:
        run_manifest(_resolve(config.parent, raw["manifest"]))


def _generate(raw: dict, config: Path) -> None:
    if "player_counts" in raw:
        _invoke(multiway.main, config)
    elif "streets" in raw:
        _invoke(heads_up.main, config)
    elif "subgames" in raw:
        _invoke(conditional_subgames.main, config)
    elif "range_spec" in raw and "stack_bb" in raw:
        _invoke(preflop.main, config)
    else:
        raise ValueError("無法辨識設定檔類型")


def _invoke(entry, config: Path) -> None:
    import sys
    previous = sys.argv
    try:
        sys.argv = [previous[0], str(config)]
        entry()
    finally:
        sys.argv = previous


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


if __name__ == "__main__":
    main()
