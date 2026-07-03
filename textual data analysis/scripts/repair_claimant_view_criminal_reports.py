from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Github\Personal-Project\textual data analysis")
REPORTS = ROOT / "artifacts" / "reports"
CRIMINAL_ROOT = REPORTS / "criminal_win_lose_mixed"
LABEL_SWAP = {"Win": "Lose", "Lose": "Win"}
LABEL_ORDER = ["Lose", "Mixed", "Win"]


def git_root() -> Path:
    output = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    return Path(output)


GIT_ROOT = git_root()


def git_blob(path: Path) -> str | None:
    rel = path.relative_to(GIT_ROOT).as_posix()
    try:
        data = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=GIT_ROOT, stderr=subprocess.DEVNULL)
        return data.decode("utf-8-sig")
    except subprocess.CalledProcessError:
        return None


def swap_label(value: object) -> object:
    return LABEL_SWAP.get(value, value)


def clean_actual(value: object) -> str:
    return str(value).replace("Actual_", "").replace("Actual ", "").strip()


def clean_predicted(value: object) -> str:
    return str(value).replace("Predicted_", "").replace("Predicted ", "").strip()


def repair_count_file(path: Path) -> bool:
    raw = git_blob(path)
    if raw is None:
        return False
    df = pd.read_csv(io.StringIO(raw))
    for col in ["label", "VERDICT", "verdict"]:
        if col in df.columns:
            df[col] = df[col].map(swap_label)
            if "count" in df.columns:
                order = {label: i for i, label in enumerate(LABEL_ORDER)}
                df = df.sort_values(by=col, key=lambda s: s.map(order).fillna(99))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return True


def repair_report_file(path: Path) -> bool:
    raw = git_blob(path)
    if raw is None:
        return False
    df = pd.read_csv(io.StringIO(raw), index_col=0)
    df.index = [swap_label(x) for x in df.index]
    ordered = [x for x in LABEL_ORDER if x in df.index]
    rest = [x for x in df.index if x not in ordered]
    df = df.loc[ordered + rest]
    df.to_csv(path, encoding="utf-8-sig")
    return True


def repair_confusion_file(path: Path) -> bool:
    raw = git_blob(path)
    if raw is None:
        return False
    df = pd.read_csv(io.StringIO(raw), index_col=0)
    if df.empty:
        return False

    df.index = [swap_label(clean_actual(x)) for x in df.index]
    df.columns = [swap_label(clean_predicted(c)) for c in df.columns]
    ordered_rows = [x for x in LABEL_ORDER if x in df.index]
    ordered_cols = [x for x in LABEL_ORDER if x in df.columns]
    df = df.loc[ordered_rows, ordered_cols]
    df.index = [f"Actual {x}" for x in df.index]
    df.columns = [f"Predicted {x}" for x in df.columns]
    df.to_csv(path, encoding="utf-8-sig")
    return True


def redraw_final_confusion_png(run_dir: Path) -> bool:
    csv_path = run_dir / "final_confusion_matrix.csv"
    if not csv_path.exists():
        return False
    cm = pd.read_csv(csv_path, index_col=0)
    if cm.empty:
        return False
    clean_rows = [clean_actual(x) for x in cm.index]
    clean_cols = [clean_predicted(c) for c in cm.columns]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(np.arange(len(clean_cols)))
    ax.set_xticklabels(clean_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(clean_rows)))
    ax.set_yticklabels(clean_rows)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Final Test Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.iat[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(run_dir / "final_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    rows = []
    for run_dir in sorted(CRIMINAL_ROOT.rglob("step3_runs/*")):
        if not run_dir.is_dir():
            continue
        repaired = 0
        for path in run_dir.glob("*label_counts.csv"):
            repaired += int(repair_count_file(path))
        for path in run_dir.glob("*test_report.csv"):
            repaired += int(repair_report_file(path))
        for path in run_dir.glob("*confusion_matrix.csv"):
            repaired += int(repair_confusion_file(path))
        repaired += int(redraw_final_confusion_png(run_dir))
        rows.append({"run_dir": str(run_dir), "repaired_files_or_png": repaired})

    out = REPORTS / "claimant_view_update" / "criminal_report_repair_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
