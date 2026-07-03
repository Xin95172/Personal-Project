from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.pipeline_modeling import run_step3_batch

ARTIFACTS = ROOT / "artifacts" / "reports"
FEATURES = ROOT / "artifacts" / "features" / "dtm"
MNIR_FEATURES = ROOT / "artifacts" / "features" / "mnir"

DATASET_NAMES = [
    "criminal_win_lose_mixed",
]
REPRESENTATIONS = ["bow", "tf", "tfidf"]
CHI2_K_GRID = [1000, 3000, 5000, 10000]
SVM_C_GRID = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def ensure_r_runtime() -> None:
    r_home = Path(r"C:\Program Files\R\R-4.6.0")
    if r_home.exists():
        os.environ["R_HOME"] = str(r_home)
        bin_path = r_home / "bin" / "x64"
        os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ.get("PATH", "")


def save_batch_outputs(result: dict, run_tag: str) -> None:
    out_dir = ARTIFACTS / "step3_batch_runs" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result["summary_df"].to_csv(out_dir / "batch_summary.csv", index=False, encoding="utf-8-sig")
    result["failures_df"].to_csv(out_dir / "batch_failures.csv", index=False, encoding="utf-8-sig")
    print(f"Batch summary saved to: {out_dir / 'batch_summary.csv'}")
    print(f"Batch failures saved to: {out_dir / 'batch_failures.csv'}")
    print(f"Completed rows: {len(result['summary_df'])}")
    print(f"Failures: {len(result['failures_df'])}")
    if not result["failures_df"].empty:
        print(result["failures_df"].to_string(index=False))


def run_main() -> None:
    run_tag = "full_20260504_154912"
    result = run_step3_batch(
        dataset_names=DATASET_NAMES,
        remove_leakage_values=[False, True],
        representations=REPRESENTATIONS,
        run_mode="full",
        run_tag=run_tag,
        skip_existing=False,
        allow_new_runs=True,
        patch_missing_no_chi2=True,
        max_rows=None,
        chi2_k_grid=CHI2_K_GRID,
        svm_c_grid=SVM_C_GRID,
        svm_class_weight=None,
        mnir_no_chi2_feature_limit=None,
        mnir_no_chi2_min_train_df=1,
        include_no_chi2=False,
        train_size=0.70,
        val_size=0.10,
        test_size=0.20,
        random_state=42,
        features_folder=str(FEATURES),
        artifacts_folder=str(ARTIFACTS),
        mnir_features_folder=str(MNIR_FEATURES),
    )
    save_batch_outputs(result, run_tag)


def run_balanced() -> None:
    run_tag = "full_20260504_154912_class_weight_balanced"
    result = run_step3_batch(
        dataset_names=DATASET_NAMES,
        remove_leakage_values=[True],
        representations=REPRESENTATIONS,
        run_mode="full",
        run_tag=run_tag,
        skip_existing=False,
        allow_new_runs=True,
        patch_missing_no_chi2=True,
        max_rows=None,
        chi2_k_grid=CHI2_K_GRID,
        svm_c_grid=SVM_C_GRID,
        svm_class_weight="balanced",
        mnir_no_chi2_feature_limit=None,
        mnir_no_chi2_min_train_df=1,
        include_no_chi2=False,
        train_size=0.70,
        val_size=0.10,
        test_size=0.20,
        random_state=42,
        features_folder=str(FEATURES),
        artifacts_folder=str(ARTIFACTS),
        mnir_features_folder=str(MNIR_FEATURES),
    )
    save_batch_outputs(result, run_tag)


def main() -> None:
    ensure_r_runtime()
    run_main()
    run_balanced()


if __name__ == "__main__":
    main()
