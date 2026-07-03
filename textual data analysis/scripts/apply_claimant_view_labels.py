from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
SUMMARY_DIR = REPORTS / "claimant_view_update"


def flip_criminal_verdict(value: object) -> object:
    if value == "Win":
        return "Lose"
    if value == "Lose":
        return "Win"
    return value


def crosstab(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["JTYPE"], df["VERDICT"], margins=True)


def update_main_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_csv = REPORTS / "judgment_labels.csv"
    labels_xlsx = REPORTS / "judgment_labels.xlsx"
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    before = crosstab(labels)

    mask = labels["JTYPE"].eq("CRIMINAL") & labels["VERDICT"].isin(["Win", "Lose"])
    labels.loc[mask, "VERDICT"] = labels.loc[mask, "VERDICT"].map(flip_criminal_verdict)
    after = crosstab(labels)

    labels.to_csv(labels_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(labels_xlsx, engine="openpyxl") as writer:
        labels.to_excel(writer, sheet_name="Labels", index=False)

    return before, after


def update_dataset_labels(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    label_map = labels.set_index("JID")["VERDICT"]
    for dataset_dir in sorted(REPORTS.glob("*_win_lose_mixed")):
        for leakage_dir in [dataset_dir / "with_leakage", dataset_dir / "no_leakage"]:
            doc_ids_path = leakage_dir / "doc_ids.csv"
            verdict_path = leakage_dir / "verdict_results.xlsx"
            if not doc_ids_path.exists() or not verdict_path.exists():
                rows.append({
                    "dataset": dataset_dir.name,
                    "leakage": leakage_dir.name,
                    "status": "missing",
                    "rows": 0,
                    "changed": 0,
                })
                continue

            doc_ids = pd.read_csv(doc_ids_path, encoding="utf-8-sig")
            old_verdict = doc_ids["VERDICT"].copy()
            doc_ids["VERDICT"] = doc_ids["JID"].map(label_map)
            if doc_ids["VERDICT"].isna().any():
                missing = int(doc_ids["VERDICT"].isna().sum())
                raise ValueError(f"{doc_ids_path} has {missing} JID values missing from judgment_labels")

            doc_ids.to_csv(doc_ids_path, index=False, encoding="utf-8-sig")
            verdict_results = pd.DataFrame({"VERDICT": doc_ids["VERDICT"].to_numpy()}, index=doc_ids["JID"])
            verdict_results.to_excel(verdict_path)

            rows.append({
                "dataset": dataset_dir.name,
                "leakage": leakage_dir.name,
                "status": "updated",
                "rows": int(len(doc_ids)),
                "changed": int((old_verdict != doc_ids["VERDICT"]).sum()),
                "Lose": int((doc_ids["VERDICT"] == "Lose").sum()),
                "Mixed": int((doc_ids["VERDICT"] == "Mixed").sum()),
                "Win": int((doc_ids["VERDICT"] == "Win").sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    before, after = update_main_labels()
    labels = pd.read_csv(REPORTS / "judgment_labels.csv", encoding="utf-8-sig")
    dataset_summary = update_dataset_labels(labels)

    before.to_csv(SUMMARY_DIR / "jtype_verdict_before.csv", encoding="utf-8-sig")
    after.to_csv(SUMMARY_DIR / "jtype_verdict_after.csv", encoding="utf-8-sig")
    dataset_summary.to_csv(SUMMARY_DIR / "dataset_label_update_summary.csv", index=False, encoding="utf-8-sig")

    print("Updated claimant-view labels.")
    print("Before:")
    print(before.to_string())
    print("\nAfter:")
    print(after.to_string())
    print("\nDataset files:")
    print(dataset_summary.to_string(index=False))


if __name__ == "__main__":
    main()
