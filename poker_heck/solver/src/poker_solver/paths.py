"""集中管理專案執行產物的預設路徑。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
DATA_DIR = ARTIFACTS_DIR / "data"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"
GENERATED_DIR = ARTIFACTS_DIR / "generated"


def ensure_artifact_directories() -> None:
    """建立會由 CLI 寫入的產物資料夾。"""
    for directory in (CHECKPOINTS_DIR, DATA_DIR, EXPORTS_DIR, GENERATED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
