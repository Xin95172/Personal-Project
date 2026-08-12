"""清除可再生的執行產物，不碰原始碼與設定檔。"""

from argparse import ArgumentParser
from pathlib import Path
import shutil

from poker_solver.paths import ARTIFACTS_DIR


def main() -> None:
    parser = ArgumentParser(description="清除 solver 的可再生訓練產物")
    parser.add_argument("--all", action="store_true", help="刪除 artifacts 內所有 checkpoint、資料庫、匯出與生成檔")
    parser.add_argument("--generated", action="store_true", help="只刪除自動生成的 pack")
    parser.add_argument("--smoke-data", action="store_true", help="只刪除 smoke 測試留下的 SQLite 資料")
    parser.add_argument("--cache", action="store_true", help="只刪除 pytest、coverage、Python 與安裝 metadata 快取")
    parser.add_argument("--empty", action="store_true", help="刪除目前已空的可選資料夾")
    args = parser.parse_args()
    if not args.all and not args.generated and not args.smoke_data and not args.cache and not args.empty:
        parser.error("請指定 --generated、--smoke-data、--cache、--empty 或 --all")
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
    for target in targets:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            print(f"已移除：{target}")
        else:
            print(f"不存在，略過：{target}")
    if args.empty:
        root = ARTIFACTS_DIR.parent
        for target in (root / "examples", root / "notebooks"):
            if target.is_dir() and not any(target.iterdir()):
                target.rmdir()
                print(f"已移除空資料夾：{target}")


if __name__ == "__main__":
    main()
