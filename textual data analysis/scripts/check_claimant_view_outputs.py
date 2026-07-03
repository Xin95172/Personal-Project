from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\Github\Personal-Project\textual data analysis")
REPORTS = ROOT / "artifacts" / "reports"
OUT_DIR = REPORTS / "claimant_view_update"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = REPORTS / "judgment_labels.csv"
DATASETS = [
    "administrative_win_lose_mixed",
    "civil_win_lose_mixed",
    "criminal_win_lose_mixed",
    "cwc_win_lose_mixed",
]
LEAKAGES = ["no_leakage", "with_leakage"]
REPS = ["bow", "tf", "tfidf"]
REQUIRED_RUN_FILES = [
    "model_comparison.csv",
    "final_test_metrics.csv",
    "final_test_report.csv",
    "final_confusion_matrix.csv",
    "final_confusion_matrix.png",
    "jtype_verdict_summary.csv",
    "valid_target_summary.csv",
    "run_config.csv",
    "artifact_index.csv",
]
CLASS_ORDER = ["Lose", "Mixed", "Win"]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def latest_run_dir(dataset: str, leakage: str, rep: str) -> Path | None:
    base = REPORTS / dataset / leakage / rep / "step3_runs"
    if not base.exists():
        return None
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("full_")]
    if not runs:
        return None
    runs.sort(key=lambda p: ("class_weight_balanced" in p.name, p.stat().st_mtime), reverse=True)
    # Prefer the main unweighted full run for thesis/PPT unless only balanced exists.
    main_runs = [p for p in runs if "class_weight_balanced" not in p.name]
    return max(main_runs or runs, key=lambda p: p.stat().st_mtime)


def counts_dict(df: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    out = df.groupby(group_cols, dropna=False).size().reset_index(name="count")
    return out.to_dict(orient="records")


def compare_dataset_labels(labels: pd.DataFrame) -> list[dict]:
    rows = []
    by_jid = labels[["JID", "JTYPE", "VERDICT"]].drop_duplicates("JID")
    for dataset in DATASETS:
        for leakage in LEAKAGES:
            base = REPORTS / dataset / leakage
            doc_ids_path = base / "doc_ids.csv"
            verdict_xlsx_path = base / "verdict_results.xlsx"
            if not doc_ids_path.exists():
                rows.append({"check": "dataset_doc_ids", "dataset": dataset, "leakage": leakage, "status": "missing", "path": str(doc_ids_path)})
                continue

            docs = read_csv(doc_ids_path)
            merged = docs.merge(by_jid, on="JID", how="left", suffixes=("_dataset", "_labels"))
            mismatches = merged[
                merged["VERDICT_dataset"].notna()
                & merged["VERDICT_labels"].notna()
                & (merged["VERDICT_dataset"] != merged["VERDICT_labels"])
            ]
            rows.append(
                {
                    "check": "dataset_doc_ids",
                    "dataset": dataset,
                    "leakage": leakage,
                    "status": "ok" if len(mismatches) == 0 else "mismatch",
                    "rows": len(docs),
                    "mismatches": len(mismatches),
                    "counts": json.dumps(docs["VERDICT"].value_counts().sort_index().to_dict(), ensure_ascii=False),
                    "path": str(doc_ids_path),
                }
            )

            if verdict_xlsx_path.exists():
                verdict_df = pd.read_excel(verdict_xlsx_path)
                merged_xlsx = verdict_df.merge(by_jid, on="JID", how="left", suffixes=("_dataset", "_labels"))
                mismatches_xlsx = merged_xlsx[
                    merged_xlsx["VERDICT_dataset"].notna()
                    & merged_xlsx["VERDICT_labels"].notna()
                    & (merged_xlsx["VERDICT_dataset"] != merged_xlsx["VERDICT_labels"])
                ]
                rows.append(
                    {
                        "check": "dataset_verdict_results_xlsx",
                        "dataset": dataset,
                        "leakage": leakage,
                        "status": "ok" if len(mismatches_xlsx) == 0 else "mismatch",
                        "rows": len(verdict_df),
                        "mismatches": len(mismatches_xlsx),
                        "path": str(verdict_xlsx_path),
                    }
                )
            else:
                rows.append({"check": "dataset_verdict_results_xlsx", "dataset": dataset, "leakage": leakage, "status": "missing", "path": str(verdict_xlsx_path)})
    return rows


def check_run_outputs() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for leakage in LEAKAGES:
            for rep in REPS:
                run_dir = latest_run_dir(dataset, leakage, rep)
                if run_dir is None:
                    rows.append({"check": "latest_full_run", "dataset": dataset, "leakage": leakage, "representation": rep, "status": "missing"})
                    continue

                missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
                rows.append(
                    {
                        "check": "required_run_files",
                        "dataset": dataset,
                        "leakage": leakage,
                        "representation": rep,
                        "status": "ok" if not missing else "missing",
                        "missing": ", ".join(missing),
                        "run_dir": str(run_dir),
                    }
                )

                report_path = run_dir / "final_test_report.csv"
                cm_path = run_dir / "final_confusion_matrix.csv"
                png_path = run_dir / "final_confusion_matrix.png"
                if report_path.exists():
                    report = pd.read_csv(report_path, index_col=0)
                    labels = [x for x in report.index.astype(str).tolist() if x in CLASS_ORDER]
                    rows.append(
                        {
                            "check": "final_test_report_label_order",
                            "dataset": dataset,
                            "leakage": leakage,
                            "representation": rep,
                            "status": "ok" if labels == CLASS_ORDER else "unexpected",
                            "labels": ", ".join(labels),
                            "path": str(report_path),
                        }
                    )
                if cm_path.exists():
                    cm = pd.read_csv(cm_path, index_col=0)
                    if cm.empty or len(cm.columns) == 0:
                        rows.append(
                            {
                                "check": "final_confusion_matrix_label_order",
                                "dataset": dataset,
                                "leakage": leakage,
                                "representation": rep,
                                "status": "empty",
                                "path": str(cm_path),
                            }
                        )
                        continue
                    row_labels = [str(x).replace("Actual_", "").replace("Actual ", "") for x in cm.index.tolist()]
                    col_labels = [str(c).replace("Predicted_", "").replace("Predicted ", "") for c in cm.columns]
                    rows.append(
                        {
                            "check": "final_confusion_matrix_label_order",
                            "dataset": dataset,
                            "leakage": leakage,
                            "representation": rep,
                            "status": "ok" if row_labels == CLASS_ORDER and col_labels == CLASS_ORDER else "unexpected",
                            "row_labels": ", ".join(row_labels),
                            "col_labels": ", ".join(col_labels),
                            "path": str(cm_path),
                        }
                    )
                if png_path.exists():
                    rows.append(
                        {
                            "check": "final_confusion_matrix_png",
                            "dataset": dataset,
                            "leakage": leakage,
                            "representation": rep,
                            "status": "ok" if png_path.stat().st_size > 0 else "empty",
                            "size_bytes": png_path.stat().st_size,
                            "modified": pd.Timestamp(png_path.stat().st_mtime, unit="s").isoformat(),
                            "path": str(png_path),
                        }
                    )
    return rows


def main() -> None:
    labels = read_csv(LABELS_PATH)
    summary = {
        "judgment_labels_rows": int(len(labels)),
        "overall_verdict_counts": labels["VERDICT"].value_counts().sort_index().to_dict(),
        "valid_target_counts": labels[labels["VERDICT"].isin(CLASS_ORDER)]["VERDICT"].value_counts().sort_index().to_dict(),
        "jtype_verdict_counts": counts_dict(labels, ["JTYPE", "VERDICT"]),
    }
    criminal = labels[labels["JTYPE"] == "CRIMINAL"]
    summary["criminal_verdict_counts"] = criminal["VERDICT"].value_counts().sort_index().to_dict()

    rows = []
    rows.extend(compare_dataset_labels(labels))
    rows.extend(check_run_outputs())
    audit = pd.DataFrame(rows)

    audit_path = OUT_DIR / "claimant_view_audit.csv"
    summary_path = OUT_DIR / "claimant_view_audit_summary.json"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    status_counts = audit["status"].value_counts(dropna=False).to_dict()
    print(json.dumps({"audit_path": str(audit_path), "summary_path": str(summary_path), "status_counts": status_counts, **summary}, ensure_ascii=False, indent=2))
    if any(status not in {"ok"} for status in audit["status"].dropna().unique()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
