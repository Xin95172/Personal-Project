from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
SUMMARY_DIR = REPORTS / "claimant_view_update"
LABEL_SWAP = {"Win": "Lose", "Lose": "Win"}
LABEL_ORDER = ["Lose", "Mixed", "Win"]


def swap_label_value(value: object) -> object:
    return LABEL_SWAP.get(value, value)


def update_count_file(path: Path) -> bool:
    df = pd.read_csv(path)
    changed = False
    for col in ["label", "VERDICT", "verdict"]:
        if col in df.columns:
            before = df[col].copy()
            df[col] = df[col].map(swap_label_value)
            changed = changed or not before.equals(df[col])
    if changed:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return changed


def update_report_file(path: Path) -> bool:
    df = pd.read_csv(path, index_col=0)
    old_index = df.index.tolist()
    df = df.rename(index=LABEL_SWAP)
    ordered = [x for x in LABEL_ORDER if x in df.index]
    rest = [x for x in df.index if x not in ordered]
    df = df.loc[ordered + rest]
    changed = old_index != df.index.tolist()
    if changed:
        df.to_csv(path, encoding="utf-8-sig")
    return changed


def update_confusion_file(path: Path) -> bool:
    df = pd.read_csv(path, index_col=0)
    before_index = df.index.tolist()
    before_cols = df.columns.tolist()
    df = df.rename(index=LABEL_SWAP, columns=LABEL_SWAP)
    ordered_index = [x for x in LABEL_ORDER if x in df.index]
    ordered_cols = [x for x in LABEL_ORDER if x in df.columns]
    df = df.loc[ordered_index, ordered_cols]
    changed = before_index != df.index.tolist() or before_cols != df.columns.tolist()
    if changed:
        df.to_csv(path, encoding="utf-8-sig")
    return changed


def redraw_final_confusion_png(run_dir: Path) -> bool:
    csv_path = run_dir / "final_confusion_matrix.csv"
    if not csv_path.exists():
        return False
    cm = pd.read_csv(csv_path, index_col=0)
    if cm.empty:
        return False
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(np.arange(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_title("Final Test Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.iat[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(run_dir / "final_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return True


def current_global_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(REPORTS / "judgment_labels.csv", encoding="utf-8-sig")
    summary = pd.crosstab(labels["JTYPE"], labels["VERDICT"], margins=True)
    valid_mask = labels["JTYPE"].isin(["ADMINISTRATIVE", "CIVIL", "CRIMINAL", "CWC"]) & labels["VERDICT"].isin(LABEL_ORDER)
    valid_summary = pd.crosstab(labels.loc[valid_mask, "JTYPE"], labels.loc[valid_mask, "VERDICT"], margins=True)
    return summary, valid_summary


def update_global_summary_files() -> int:
    summary, valid_summary = current_global_summaries()
    count = 0
    for run_dir in REPORTS.rglob("step3_runs/*"):
        if not run_dir.is_dir():
            continue
        summary.to_csv(run_dir / "jtype_verdict_summary.csv", encoding="utf-8-sig")
        valid_summary.to_csv(run_dir / "valid_target_summary.csv", encoding="utf-8-sig")
        count += 2
    return count


def update_criminal_run_dirs() -> pd.DataFrame:
    rows = []
    for run_dir in sorted((REPORTS / "criminal_win_lose_mixed").rglob("step3_runs/*")):
        if not run_dir.is_dir():
            continue
        changed_files = 0
        for path in run_dir.glob("*label_counts.csv"):
            changed_files += int(update_count_file(path))
        for path in run_dir.glob("*test_report.csv"):
            changed_files += int(update_report_file(path))
        for path in run_dir.glob("*confusion_matrix.csv"):
            changed_files += int(update_confusion_file(path))
        changed_files += int(redraw_final_confusion_png(run_dir))
        rows.append({
            "run_dir": str(run_dir),
            "changed_files_or_png": changed_files,
        })
    return pd.DataFrame(rows)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    criminal_updates = update_criminal_run_dirs()
    global_count = update_global_summary_files()
    criminal_updates.to_csv(SUMMARY_DIR / "criminal_result_label_swap_summary.csv", index=False, encoding="utf-8-sig")
    print("Updated criminal result labels.")
    print(criminal_updates.to_string(index=False))
    print(f"Updated global summary files: {global_count}")


if __name__ == "__main__":
    main()
