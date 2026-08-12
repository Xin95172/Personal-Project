"""清理可再生的 solver 輸出與開發快取。"""

from argparse import ArgumentParser
from pathlib import Path
import shutil

from poker_solver.paths import ARTIFACTS_DIR


def main() -> None:
    parser = ArgumentParser(description="清理 solver 的可再生輸出")
    parser.add_argument("--all", action="store_true", help="刪除整個 artifacts 目錄，包含 checkpoint 與策略資料庫")
    parser.add_argument("--generated", action="store_true", help="只刪除產生的 manifest 與訓練 pack")
    parser.add_argument("--smoke-data", action="store_true", help="只刪除測試用 SQLite 策略庫")
    parser.add_argument("--cache", action="store_true", help="刪除 pytest、coverage、Python 快取與安裝 metadata")
    parser.add_argument("--empty", action="store_true", help="刪除空的舊目錄")
    args = parser.parse_args()
    if not args.all and not args.generated and not args.smoke_data and not args.cache and not args.empty:
        parser.error("請至少指定 --generated、--smoke-data、--cache、--empty 或 --all")

    targets: list[Path] = []
    if args.all:
        targets.append(ARTIFACTS_DIR)
    elif args.generated:
        targets.append(ARTIFACTS_DIR / "generated")
    if args.smoke_data:
        targets.append(ARTIFACTS_DIR / "data" / "store_smoke.sqlite3")
    if args.cache:
        root = ARTIFACTS_DIR.parent
        targets.extend((root / ".pytest_cache", root / "__pycache__", root / "poker_heck_solver.egg-info", ARTIFACTS_DIR / ".coverage"))
        targets.extend(root.rglob("__pycache__"))
        targets.extend(root.rglob("*.egg-info"))

    for target in targets:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            print(f"已刪除：{target}")
        else:
            print(f"不存在，略過：{target}")

    if args.empty:
        root = ARTIFACTS_DIR.parent
        for target in (root / "examples", root / "notebooks"):
            if target.is_dir() and not any(target.iterdir()):
                target.rmdir()
                print(f"已刪除空目錄：{target}")


if __name__ == "__main__":
    main()
