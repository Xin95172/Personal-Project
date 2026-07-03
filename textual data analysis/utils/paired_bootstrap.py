from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from utils.pipeline_modeling import (
    apply_chi2_feature_selection,
    evaluate_svm_grid,
    load_modeling_data,
    run_mnir_feature_extraction,
    split_modeling_data,
    stratified_sample_modeling_data,
)


def paired_bootstrap_macro_f1(
    y_true,
    y_pred_svm,
    y_pred_mnir_svm,
    n_bootstrap: int = 5000,
    random_state: int = 42,
    labels=("Lose", "Mixed", "Win"),
    bootstrap_method: str = "stratified_paired",
    confidence_level: float = 0.95,
):
    """Paired bootstrap for Delta = MacroF1(Chi-square + SVM) - MacroF1(MNIR + SVM)."""
    y_true = np.asarray(y_true)
    y_pred_svm = np.asarray(y_pred_svm)
    y_pred_mnir_svm = np.asarray(y_pred_mnir_svm)
    if not (len(y_true) == len(y_pred_svm) == len(y_pred_mnir_svm)):
        raise ValueError("y_true, y_pred_svm, and y_pred_mnir_svm must have the same length.")

    bootstrap_method = bootstrap_method.lower().replace("-", "_")
    if bootstrap_method not in {"paired", "stratified_paired"}:
        raise ValueError("bootstrap_method must be 'paired' or 'stratified_paired'.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    label_list = list(labels)
    class_indices = {label: np.flatnonzero(y_true == label) for label in label_list}
    all_classes_present = all(len(idx) > 0 for idx in class_indices.values())
    if bootstrap_method == "stratified_paired" and not all_classes_present:
        missing = [label for label, idx in class_indices.items() if len(idx) == 0]
        raise ValueError(f"Cannot run stratified paired bootstrap because test set lacks classes: {missing}")

    deltas = np.empty(n_bootstrap, dtype=float)
    invalid_resamples = 0

    for b in range(n_bootstrap):
        if bootstrap_method == "stratified_paired":
            idx = np.concatenate(
                [rng.choice(class_idx, size=len(class_idx), replace=True) for class_idx in class_indices.values()]
            )
            rng.shuffle(idx)
        else:
            # Use the same sampled index for both models to preserve pairing.
            idx = rng.integers(0, n, size=n)
            if not all(np.any(y_true[idx] == label) for label in label_list):
                invalid_resamples += 1
        svm_f1 = f1_score(
            y_true[idx],
            y_pred_svm[idx],
            average="macro",
            labels=label_list,
            zero_division=0,
        )
        mnir_f1 = f1_score(
            y_true[idx],
            y_pred_mnir_svm[idx],
            average="macro",
            labels=label_list,
            zero_division=0,
        )
        deltas[b] = svm_f1 - mnir_f1

    observed_svm_f1 = f1_score(y_true, y_pred_svm, average="macro", labels=label_list, zero_division=0)
    observed_mnir_f1 = f1_score(y_true, y_pred_mnir_svm, average="macro", labels=label_list, zero_division=0)
    observed_delta = observed_svm_f1 - observed_mnir_f1
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    summary = pd.DataFrame(
        [
            {
                "n_test": n,
                "n_bootstrap": n_bootstrap,
                "bootstrap_method": bootstrap_method,
                "ci_method": "percentile",
                "confidence_level": confidence_level,
                "bootstrap_random_seed": random_state,
                "all_classes_present": all_classes_present,
                "n_invalid_resamples": invalid_resamples,
                "svm_macro_f1": observed_svm_f1,
                "mnir_svm_macro_f1": observed_mnir_f1,
                "delta_svm_minus_mnir": observed_delta,
                "ci_2.5%": ci_low,
                "ci_97.5%": ci_high,
                "ci_excludes_0": (ci_low > 0) or (ci_high < 0),
                "interpretation_scope": "fixed trained models and fixed test set only",
            }
        ]
    )
    return summary, deltas


def _load_doc_ids(modeling_data):
    report_folder = Path(modeling_data["verdict_results_path"]).parent
    doc_ids_path = report_folder / "doc_ids.csv"
    if not doc_ids_path.exists():
        return None
    return pd.read_csv(doc_ids_path)


def _split_sample_indices(sample_indices, y_sample, train_size, val_size, test_size, random_state, stratify=True):
    y_sample = np.asarray(y_sample).ravel()
    first_stratify = y_sample if stratify else None
    train_idx, temp_idx, _, y_temp = train_test_split(
        sample_indices,
        y_sample,
        train_size=train_size,
        random_state=random_state,
        stratify=first_stratify,
    )
    relative_test_size = test_size / (val_size + test_size)
    second_stratify = y_temp if stratify else None
    _, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=second_stratify,
    )
    return np.asarray(test_idx)


def _apply_label_permutation(values, label_permutation):
    values = np.asarray(values)
    if not label_permutation:
        return values
    return np.array([label_permutation.get(value, value) for value in values])


def _coerce_optional(value):
    if pd.isna(value) or value == "":
        return None
    return value


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _coerce_int_or_none(value):
    value = _coerce_optional(value)
    if value is None or str(value).strip().lower() == "all":
        return None
    return int(float(value))


def _normalize_representation(value):
    representation = str(value).lower().replace("-", "_")
    return "tfidf" if representation == "tf_idf" else representation


def _metadata_from_best_run_configs(
    svm_config,
    mnir_config,
    label_view_for_modeling="current",
):
    dataset_name = str(svm_config["dataset_name"])
    remove_leakage = _coerce_bool(svm_config["remove_leakage"])
    train_size = float(svm_config["train_size"])
    val_size = float(svm_config["val_size"])
    test_size = float(svm_config["test_size"])
    random_state = int(svm_config["random_state"])
    svm_class_weight = _coerce_optional(svm_config.get("svm_class_weight"))
    return {
        "dataset_name": dataset_name,
        "remove_leakage": remove_leakage,
        "text_version": "no_leakage" if remove_leakage else "with_leakage",
        "svm_representation": _normalize_representation(svm_config["representation"]),
        "mnir_representation": _normalize_representation(mnir_config["representation"]),
        "svm_chi2_k": _coerce_int_or_none(svm_config["best_direct_chi2_k"]),
        "svm_c": float(svm_config["best_direct_svm_c"]),
        "mnir_chi2_k": _coerce_int_or_none(mnir_config["best_chi2_k"]),
        "mnir_svm_c": float(mnir_config["best_svm_c"]),
        "svm_class_weight": svm_class_weight,
        "random_seed": random_state,
        "split_id": f"random_state_{random_state}_train{train_size}_val{val_size}_test{test_size}",
        "parameter_source": "saved_step3_test_predictions",
        "label_view_for_modeling": label_view_for_modeling,
    }


def _load_step3_test_predictions_from_run_dirs(
    svm_run_dir,
    mnir_run_dir,
    label_permutation_for_modeling=None,
):
    svm_path = Path(svm_run_dir) / "test_predictions.csv"
    mnir_path = Path(mnir_run_dir) / "test_predictions.csv"
    if not svm_path.exists() or not mnir_path.exists():
        return None

    svm_predictions = pd.read_csv(svm_path, index_col=0)
    mnir_predictions = pd.read_csv(mnir_path, index_col=0)
    required = {"test_row", "source_row", "y_true", "y_pred_svm", "y_pred_mnir_svm"}
    missing_svm = required - set(svm_predictions.columns)
    missing_mnir = required - set(mnir_predictions.columns)
    if missing_svm:
        raise ValueError(f"Missing columns in {svm_path}: {sorted(missing_svm)}")
    if missing_mnir:
        raise ValueError(f"Missing columns in {mnir_path}: {sorted(missing_mnir)}")

    for col in ["test_row", "source_row", "y_true"]:
        if not np.array_equal(svm_predictions[col].astype(str).to_numpy(), mnir_predictions[col].astype(str).to_numpy()):
            raise RuntimeError(f"SVM and MNIR saved test_predictions are not aligned on {col}.")

    keep_meta = [
        col
        for col in ["test_row", "source_row", "JID", "JTYPE", "doc_verdict"]
        if col in svm_predictions.columns
    ]
    predictions = svm_predictions[keep_meta + ["y_true", "y_pred_svm"]].copy()
    predictions["y_pred_mnir_svm"] = mnir_predictions["y_pred_mnir_svm"].to_numpy()
    for col in ["y_true", "y_pred_svm", "y_pred_mnir_svm"]:
        predictions[col] = _apply_label_permutation(predictions[col].to_numpy(), label_permutation_for_modeling)
    if "doc_verdict" in predictions.columns:
        predictions["doc_verdict"] = _apply_label_permutation(
            predictions["doc_verdict"].to_numpy(),
            label_permutation_for_modeling,
        )
        if not np.array_equal(
            predictions["y_true"].astype(str).to_numpy(),
            predictions["doc_verdict"].astype(str).to_numpy(),
        ):
            raise RuntimeError("Internal alignment check failed: y_true does not match doc_verdict.")
    return predictions


def build_chi2_svm_vs_mnir_test_predictions_fixed_params(
    dataset_name: str,
    remove_leakage: bool,
    representation: str,
    svm_chi2_k,
    svm_c,
    mnir_chi2_k,
    mnir_svm_c,
    max_rows=None,
    train_size: float = 0.70,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
    svm_class_weight=None,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    mnir_features_folder="../artifacts/features/mnir",
    label_permutation_for_modeling=None,
    label_view_for_modeling="current",
):
    """Rebuild paired test predictions using fixed K/C parameters from a completed run."""
    representation = representation.lower().replace("-", "_")
    if representation == "tf_idf":
        representation = "tfidf"

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    y_for_modeling = _apply_label_permutation(modeling_data["y"], label_permutation_for_modeling)
    sampled_data = stratified_sample_modeling_data(
        modeling_data["x"],
        y_for_modeling,
        max_rows=max_rows,
        random_state=random_state,
    )
    test_source_rows = _split_sample_indices(
        sampled_data["sample_indices"],
        sampled_data["y"],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    split_data = split_modeling_data(
        sampled_data["x"],
        sampled_data["y"],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    mnir_chi2 = apply_chi2_feature_selection(
        split_data["x_train"],
        split_data["y_train"],
        split_data["x_val"],
        split_data["x_test"],
        k=mnir_chi2_k,
        output_dir=None,
        representation=representation,
    )
    mnir_data = run_mnir_feature_extraction(
        mnir_chi2["x_train"],
        split_data["y_train"],
        mnir_chi2["x_val"],
        mnir_chi2["x_test"],
        dataset_slug=modeling_data["dataset_slug"],
        leakage_variant=modeling_data["leakage_variant"],
        feature_variant=mnir_chi2["feature_variant"],
        mnir_features_folder=mnir_features_folder,
    )
    mnir_results = evaluate_svm_grid(
        mnir_data["z_train"],
        split_data["y_train"],
        mnir_data["z_val"],
        split_data["y_val"],
        mnir_data["z_test"],
        split_data["y_test"],
        svm_grid=[float(mnir_svm_c)],
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )

    svm_chi2 = apply_chi2_feature_selection(
        split_data["x_train"],
        split_data["y_train"],
        split_data["x_val"],
        split_data["x_test"],
        k=svm_chi2_k,
        output_dir=None,
        representation=representation,
    )
    svm_results = evaluate_svm_grid(
        svm_chi2["x_train"],
        split_data["y_train"],
        svm_chi2["x_val"],
        split_data["y_val"],
        svm_chi2["x_test"],
        split_data["y_test"],
        svm_grid=[float(svm_c)],
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )

    predictions = pd.DataFrame(
        {
            "test_row": np.arange(len(split_data["y_test"])),
            "source_row": test_source_rows,
            "y_true": _apply_label_permutation(split_data["y_test"], label_permutation_for_modeling),
            "y_pred_svm": _apply_label_permutation(svm_results["y_test_pred"], label_permutation_for_modeling),
            "y_pred_mnir_svm": _apply_label_permutation(mnir_results["y_test_pred"], label_permutation_for_modeling),
        }
    )
    doc_ids = _load_doc_ids(modeling_data)
    if doc_ids is not None:
        meta_cols = [col for col in ["JID", "JTYPE", "VERDICT"] if col in doc_ids.columns]
        doc_meta = doc_ids.iloc[test_source_rows][meta_cols].reset_index(drop=True)
        insert_at = 2
        for col in meta_cols:
            out_col = "doc_" + col.lower() if col == "VERDICT" else col
            predictions.insert(insert_at, out_col, doc_meta[col].to_numpy())
            insert_at += 1

    if "doc_verdict" in predictions.columns and not np.array_equal(
        predictions["y_true"].astype(str).to_numpy(),
        predictions["doc_verdict"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Internal alignment check failed: y_true does not match doc_ids VERDICT.")

    metadata = {
        "dataset_name": dataset_name,
        "remove_leakage": remove_leakage,
        "text_version": "no_leakage" if remove_leakage else "with_leakage",
        "representation": representation,
        "mnir_chi2_k": mnir_chi2_k,
        "mnir_svm_c": float(mnir_svm_c),
        "svm_chi2_k": svm_chi2_k,
        "svm_c": float(svm_c),
        "svm_class_weight": svm_class_weight,
        "random_seed": random_state,
        "split_id": f"random_state_{random_state}_train{train_size}_val{val_size}_test{test_size}",
        "parameter_source": "fixed_params",
        "label_view_for_modeling": label_view_for_modeling,
    }
    return predictions, metadata


def run_chi2_svm_vs_mnir_paired_bootstrap_fixed_params(
    dataset_name: str,
    remove_leakage: bool,
    representation: str,
    svm_chi2_k,
    svm_c,
    mnir_chi2_k,
    mnir_svm_c,
    max_rows=None,
    train_size: float = 0.70,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
    svm_class_weight=None,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    mnir_features_folder="../artifacts/features/mnir",
    n_bootstrap: int = 5000,
    bootstrap_random_state: int = 20260703,
    bootstrap_method: str = "stratified_paired",
    confidence_level: float = 0.95,
    output_dir="../artifacts/reports/paired_bootstrap",
    output_stem_suffix="fixed_params",
    label_permutation_for_modeling=None,
    label_view_for_modeling="current",
):
    """Run paired bootstrap with fixed model-selection parameters."""
    predictions, metadata = build_chi2_svm_vs_mnir_test_predictions_fixed_params(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        svm_chi2_k=svm_chi2_k,
        svm_c=svm_c,
        mnir_chi2_k=mnir_chi2_k,
        mnir_svm_c=mnir_svm_c,
        max_rows=max_rows,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        svm_class_weight=svm_class_weight,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
        mnir_features_folder=mnir_features_folder,
        label_permutation_for_modeling=label_permutation_for_modeling,
        label_view_for_modeling=label_view_for_modeling,
    )

    summary, deltas = paired_bootstrap_macro_f1(
        predictions["y_true"].to_numpy(),
        predictions["y_pred_svm"].to_numpy(),
        predictions["y_pred_mnir_svm"].to_numpy(),
        n_bootstrap=n_bootstrap,
        random_state=bootstrap_random_state,
        labels=("Lose", "Mixed", "Win"),
        bootstrap_method=bootstrap_method,
        confidence_level=confidence_level,
    )
    for idx, (key, value) in enumerate(metadata.items()):
        summary.insert(idx, key, value)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leakage_slug = "no_leakage" if remove_leakage else "with_leakage"
    class_weight_slug = "none" if svm_class_weight is None else str(svm_class_weight).replace(" ", "_")
    stem = f"{dataset_name}_{leakage_slug}_{metadata['representation']}_class_weight_{class_weight_slug}_{output_stem_suffix}"
    pred_path = out_dir / f"{stem}_test_predictions.csv"
    summary_path = out_dir / f"{stem}_paired_bootstrap_summary.csv"
    deltas_path = out_dir / f"{stem}_paired_bootstrap_deltas.csv"
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "bootstrap_id": np.arange(len(deltas)),
            "delta_svm_minus_mnir": deltas,
        }
    ).to_csv(deltas_path, index=False, encoding="utf-8-sig")
    return {
        "summary": summary,
        "deltas": deltas,
        "predictions": predictions,
        "paths": {
            "predictions": pred_path,
            "summary": summary_path,
            "deltas": deltas_path,
        },
    }


def build_chi2_svm_vs_mnir_test_predictions_from_run_configs(
    svm_config,
    mnir_config,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    mnir_features_folder="../artifacts/features/mnir",
    label_permutation_for_modeling=None,
    label_view_for_modeling="current",
):
    """Build paired predictions when Chi-square + SVM and MNIR + SVM use different best runs."""
    dataset_name = str(svm_config["dataset_name"])
    if dataset_name != str(mnir_config["dataset_name"]):
        raise ValueError("SVM and MNIR run configs must use the same dataset_name.")

    remove_leakage = _coerce_bool(svm_config["remove_leakage"])
    if remove_leakage != _coerce_bool(mnir_config["remove_leakage"]):
        raise ValueError("SVM and MNIR run configs must use the same remove_leakage setting.")

    shared_keys = ["max_rows", "train_size", "val_size", "test_size", "random_state", "svm_class_weight"]
    for key in shared_keys:
        if str(_coerce_optional(svm_config.get(key))) != str(_coerce_optional(mnir_config.get(key))):
            raise ValueError(f"SVM and MNIR run configs must use the same {key}.")

    train_size = float(svm_config["train_size"])
    val_size = float(svm_config["val_size"])
    test_size = float(svm_config["test_size"])
    random_state = int(svm_config["random_state"])
    max_rows = _coerce_int_or_none(svm_config.get("max_rows"))
    svm_class_weight = _coerce_optional(svm_config.get("svm_class_weight"))
    svm_representation = _normalize_representation(svm_config["representation"])
    mnir_representation = _normalize_representation(mnir_config["representation"])

    svm_modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=svm_representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    mnir_modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=mnir_representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    svm_y_for_modeling = _apply_label_permutation(svm_modeling_data["y"], label_permutation_for_modeling)
    mnir_y_for_modeling = _apply_label_permutation(mnir_modeling_data["y"], label_permutation_for_modeling)
    if not np.array_equal(svm_y_for_modeling, mnir_y_for_modeling):
        raise RuntimeError("SVM and MNIR modeling labels are not aligned.")

    sampled_data = stratified_sample_modeling_data(
        svm_modeling_data["x"],
        svm_y_for_modeling,
        max_rows=max_rows,
        random_state=random_state,
    )
    test_source_rows = _split_sample_indices(
        sampled_data["sample_indices"],
        sampled_data["y"],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    svm_split_data = split_modeling_data(
        sampled_data["x"],
        sampled_data["y"],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    mnir_sampled_data = {
        "x": mnir_modeling_data["x"][sampled_data["sample_indices"]],
        "y": sampled_data["y"],
    }
    mnir_split_data = split_modeling_data(
        mnir_sampled_data["x"],
        mnir_sampled_data["y"],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    if not np.array_equal(svm_split_data["y_test"], mnir_split_data["y_test"]):
        raise RuntimeError("SVM and MNIR test labels are not aligned.")

    svm_chi2 = apply_chi2_feature_selection(
        svm_split_data["x_train"],
        svm_split_data["y_train"],
        svm_split_data["x_val"],
        svm_split_data["x_test"],
        k=_coerce_int_or_none(svm_config["best_direct_chi2_k"]),
        output_dir=None,
        representation=svm_representation,
    )
    svm_results = evaluate_svm_grid(
        svm_chi2["x_train"],
        svm_split_data["y_train"],
        svm_chi2["x_val"],
        svm_split_data["y_val"],
        svm_chi2["x_test"],
        svm_split_data["y_test"],
        svm_grid=[float(svm_config["best_direct_svm_c"])],
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )

    mnir_chi2 = apply_chi2_feature_selection(
        mnir_split_data["x_train"],
        mnir_split_data["y_train"],
        mnir_split_data["x_val"],
        mnir_split_data["x_test"],
        k=_coerce_int_or_none(mnir_config["best_chi2_k"]),
        output_dir=None,
        representation=mnir_representation,
    )
    mnir_data = run_mnir_feature_extraction(
        mnir_chi2["x_train"],
        mnir_split_data["y_train"],
        mnir_chi2["x_val"],
        mnir_chi2["x_test"],
        dataset_slug=mnir_modeling_data["dataset_slug"],
        leakage_variant=mnir_modeling_data["leakage_variant"],
        feature_variant=mnir_chi2["feature_variant"],
        mnir_features_folder=mnir_features_folder,
    )
    mnir_results = evaluate_svm_grid(
        mnir_data["z_train"],
        mnir_split_data["y_train"],
        mnir_data["z_val"],
        mnir_split_data["y_val"],
        mnir_data["z_test"],
        mnir_split_data["y_test"],
        svm_grid=[float(mnir_config["best_svm_c"])],
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )

    predictions = pd.DataFrame(
        {
            "test_row": np.arange(len(svm_split_data["y_test"])),
            "source_row": test_source_rows,
            "y_true": _apply_label_permutation(svm_split_data["y_test"], label_permutation_for_modeling),
            "y_pred_svm": _apply_label_permutation(svm_results["y_test_pred"], label_permutation_for_modeling),
            "y_pred_mnir_svm": _apply_label_permutation(mnir_results["y_test_pred"], label_permutation_for_modeling),
        }
    )
    doc_ids = _load_doc_ids(svm_modeling_data)
    if doc_ids is not None:
        meta_cols = [col for col in ["JID", "JTYPE", "VERDICT"] if col in doc_ids.columns]
        doc_meta = doc_ids.iloc[test_source_rows][meta_cols].reset_index(drop=True)
        insert_at = 2
        for col in meta_cols:
            out_col = "doc_" + col.lower() if col == "VERDICT" else col
            predictions.insert(insert_at, out_col, doc_meta[col].to_numpy())
            insert_at += 1

    if "doc_verdict" in predictions.columns and not np.array_equal(
        predictions["y_true"].astype(str).to_numpy(),
        predictions["doc_verdict"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Internal alignment check failed: y_true does not match doc_ids VERDICT.")

    metadata = {
        "dataset_name": dataset_name,
        "remove_leakage": remove_leakage,
        "text_version": "no_leakage" if remove_leakage else "with_leakage",
        "svm_representation": svm_representation,
        "mnir_representation": mnir_representation,
        "svm_chi2_k": _coerce_int_or_none(svm_config["best_direct_chi2_k"]),
        "svm_c": float(svm_config["best_direct_svm_c"]),
        "mnir_chi2_k": _coerce_int_or_none(mnir_config["best_chi2_k"]),
        "mnir_svm_c": float(mnir_config["best_svm_c"]),
        "svm_class_weight": svm_class_weight,
        "random_seed": random_state,
        "split_id": f"random_state_{random_state}_train{train_size}_val{val_size}_test{test_size}",
        "parameter_source": "fixed_best_run_configs",
        "label_view_for_modeling": label_view_for_modeling,
    }
    return predictions, metadata


def run_chi2_svm_vs_mnir_paired_bootstrap_from_best_run_dirs(
    svm_run_dir,
    mnir_run_dir,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    mnir_features_folder="../artifacts/features/mnir",
    n_bootstrap: int = 5000,
    bootstrap_random_state: int = 20260703,
    bootstrap_method: str = "stratified_paired",
    confidence_level: float = 0.95,
    output_dir="../artifacts/reports/paired_bootstrap",
    use_legacy_criminal_label_view=True,
):
    """Run paired bootstrap for thesis-style best SVM vs best MNIR + SVM comparison."""
    svm_run_dir = Path(svm_run_dir)
    mnir_run_dir = Path(mnir_run_dir)
    svm_config_path = svm_run_dir / "run_config.csv"
    mnir_config_path = mnir_run_dir / "run_config.csv"
    if not svm_config_path.exists():
        raise FileNotFoundError(f"run_config.csv not found: {svm_config_path}")
    if not mnir_config_path.exists():
        raise FileNotFoundError(f"run_config.csv not found: {mnir_config_path}")
    svm_config = pd.read_csv(svm_config_path).iloc[0]
    mnir_config = pd.read_csv(mnir_config_path).iloc[0]
    dataset_name = str(svm_config["dataset_name"])
    legacy_criminal_swap = {"Win": "Lose", "Lose": "Win"} if (
        use_legacy_criminal_label_view and dataset_name == "criminal_win_lose_mixed"
    ) else None
    label_view_for_modeling = "legacy_defendant_view_then_claimant_view_output" if legacy_criminal_swap else "current"
    predictions = _load_step3_test_predictions_from_run_dirs(
        svm_run_dir,
        mnir_run_dir,
        label_permutation_for_modeling=legacy_criminal_swap,
    )
    if predictions is None:
        predictions, metadata = build_chi2_svm_vs_mnir_test_predictions_from_run_configs(
            svm_config=svm_config,
            mnir_config=mnir_config,
            features_folder=features_folder,
            artifacts_folder=artifacts_folder,
            mnir_features_folder=mnir_features_folder,
            label_permutation_for_modeling=legacy_criminal_swap,
            label_view_for_modeling=label_view_for_modeling,
        )
    else:
        metadata = _metadata_from_best_run_configs(
            svm_config=svm_config,
            mnir_config=mnir_config,
            label_view_for_modeling=label_view_for_modeling,
        )
    summary, deltas = paired_bootstrap_macro_f1(
        predictions["y_true"].to_numpy(),
        predictions["y_pred_svm"].to_numpy(),
        predictions["y_pred_mnir_svm"].to_numpy(),
        n_bootstrap=n_bootstrap,
        random_state=bootstrap_random_state,
        labels=("Lose", "Mixed", "Win"),
        bootstrap_method=bootstrap_method,
        confidence_level=confidence_level,
    )
    metadata = {
        "svm_run_tag": str(svm_config["run_tag"]),
        "mnir_run_tag": str(mnir_config["run_tag"]),
        "svm_run_dir": str(svm_run_dir),
        "mnir_run_dir": str(mnir_run_dir),
        **metadata,
    }
    for idx, (key, value) in enumerate(metadata.items()):
        summary.insert(idx, key, value)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_weight_slug = "none" if metadata["svm_class_weight"] is None else str(metadata["svm_class_weight"]).replace(" ", "_")
    stem = (
        f"{dataset_name}_{metadata['text_version']}_svm_{metadata['svm_representation']}"
        f"_mnir_{metadata['mnir_representation']}_class_weight_{class_weight_slug}_best_runs"
    )
    pred_path = out_dir / f"{stem}_test_predictions.csv"
    summary_path = out_dir / f"{stem}_paired_bootstrap_summary.csv"
    deltas_path = out_dir / f"{stem}_paired_bootstrap_deltas.csv"
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "bootstrap_id": np.arange(len(deltas)),
            "delta_svm_minus_mnir": deltas,
        }
    ).to_csv(deltas_path, index=False, encoding="utf-8-sig")
    return {
        "summary": summary,
        "deltas": deltas,
        "predictions": predictions,
        "paths": {
            "predictions": pred_path,
            "summary": summary_path,
            "deltas": deltas_path,
        },
    }


def run_chi2_svm_vs_mnir_paired_bootstrap_from_run_dir(
    run_dir,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    mnir_features_folder="../artifacts/features/mnir",
    n_bootstrap: int = 5000,
    bootstrap_random_state: int = 20260703,
    bootstrap_method: str = "stratified_paired",
    confidence_level: float = 0.95,
    output_dir="../artifacts/reports/paired_bootstrap",
    use_legacy_criminal_label_view=True,
):
    """Read a completed Step 3 run_config.csv and run fixed-parameter paired bootstrap."""
    run_dir = Path(run_dir)
    config_path = run_dir / "run_config.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"run_config.csv not found: {config_path}")
    config = pd.read_csv(config_path).iloc[0]
    run_tag = str(config["run_tag"])
    dataset_name = str(config["dataset_name"])
    legacy_criminal_swap = {"Win": "Lose", "Lose": "Win"} if (
        use_legacy_criminal_label_view and dataset_name == "criminal_win_lose_mixed"
    ) else None
    result = run_chi2_svm_vs_mnir_paired_bootstrap_fixed_params(
        dataset_name=dataset_name,
        remove_leakage=_coerce_bool(config["remove_leakage"]),
        representation=str(config["representation"]),
        svm_chi2_k=_coerce_int_or_none(config["best_direct_chi2_k"]),
        svm_c=float(config["best_direct_svm_c"]),
        mnir_chi2_k=_coerce_int_or_none(config["best_chi2_k"]),
        mnir_svm_c=float(config["best_svm_c"]),
        max_rows=_coerce_int_or_none(config.get("max_rows")),
        train_size=float(config["train_size"]),
        val_size=float(config["val_size"]),
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
        svm_class_weight=_coerce_optional(config.get("svm_class_weight")),
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
        mnir_features_folder=mnir_features_folder,
        n_bootstrap=n_bootstrap,
        bootstrap_random_state=bootstrap_random_state,
        bootstrap_method=bootstrap_method,
        confidence_level=confidence_level,
        output_dir=output_dir,
        output_stem_suffix=f"{run_tag}_fixed_params",
        label_permutation_for_modeling=legacy_criminal_swap,
        label_view_for_modeling="legacy_defendant_view_then_claimant_view_output" if legacy_criminal_swap else "current",
    )
    result["summary"].insert(0, "run_tag", run_tag)
    result["summary"].insert(1, "run_dir", str(run_dir))
    result["summary"].to_csv(result["paths"]["summary"], index=False, encoding="utf-8-sig")
    return result
