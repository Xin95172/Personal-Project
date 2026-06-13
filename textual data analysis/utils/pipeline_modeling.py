import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from algos.svm import SVMClassifier

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def check_mnir_runtime(require_textir=True):
    """Return MNIR R runtime diagnostics for the active Python environment."""
    diagnostics = {
        'python_executable': os.sys.executable,
        'R_HOME': os.environ.get('R_HOME'),
        'R_on_path': shutil.which('R'),
        'Rscript_on_path': shutil.which('Rscript'),
        'R_version': None,
        'Matrix': None,
        'textir': None,
        'error': None,
        'ok': False,
    }

    if diagnostics['Rscript_on_path'] is None:
        diagnostics['error'] = 'Rscript not found on PATH.'
        return diagnostics

    packages = "c('Matrix','textir')" if require_textir else "c('Matrix')"
    r_code = (
        "cat(R.version.string, '\\n'); "
        f"for (pkg in {packages}) {{ "
        "cat(pkg, ':', requireNamespace(pkg, quietly=TRUE), '\\n') "
        "}"
    )
    completed = subprocess.run(
        [diagnostics['Rscript_on_path'], '-e', r_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        diagnostics['error'] = completed.stderr.strip() or completed.stdout.strip()
        return diagnostics

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if lines:
        diagnostics['R_version'] = lines[0]
    for line in lines[1:]:
        if line.startswith('Matrix :'):
            diagnostics['Matrix'] = 'ok' if line.endswith('TRUE') else 'missing'
        elif line.startswith('textir :'):
            diagnostics['textir'] = 'ok' if line.endswith('TRUE') else 'missing'
    if not require_textir:
        diagnostics['textir'] = 'not_checked'

    diagnostics['ok'] = (
        diagnostics['Rscript_on_path'] is not None
        and diagnostics['Matrix'] == 'ok'
        and (diagnostics['textir'] == 'ok' or not require_textir)
    )
    return diagnostics


def assert_mnir_runtime_ready():
    diagnostics = check_mnir_runtime(require_textir=True)
    if diagnostics['ok']:
        return diagnostics

    fix_hint = (
        "MNIR runtime is not ready in the active environment.\n"
        f"Python: {diagnostics['python_executable']}\n"
        f"R_HOME: {diagnostics['R_HOME']}\n"
        f"R on PATH: {diagnostics['R_on_path']}\n"
        f"Rscript on PATH: {diagnostics['Rscript_on_path']}\n"
        f"R version: {diagnostics['R_version']}\n"
        f"Matrix: {diagnostics['Matrix']}\n"
        f"textir: {diagnostics['textir']}\n"
        f"Error: {diagnostics['error']}\n\n"
        "Fix in the tm environment by installing R packages Matrix, gamlr, distrom, "
        "and textir, then restart the notebook kernel."
    )
    raise RuntimeError(fix_hint)


def _load_mnir_functions():
    assert_mnir_runtime_ready()
    try:
        from algos.MNIR import load_or_fit_feature_splits, summarize_predictions
    except Exception as exc:
        raise RuntimeError(
            "MNIR requires the optional R/rpy2 runtime used by algos.MNIR. "
            "Check that R, rpy2, and the MNIR dependencies are installed."
        ) from exc

    return load_or_fit_feature_splits, summarize_predictions


def _safe_slug(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip().lower())
    return value.strip('_') or 'subset'


def _batch_combinations(dataset_names, remove_leakage_values, representations, desc='Batch'):
    combos = [
        (dataset_name, remove_leakage, representation)
        for dataset_name in dataset_names
        for remove_leakage in remove_leakage_values
        for representation in representations
    ]
    if tqdm is None:
        return combos
    return tqdm(combos, total=len(combos), desc=desc)


def resolve_modeling_paths(
    dataset_name=None,
    remove_leakage=False,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    dtm_bow_path=None,
    dtm_tf_path=None,
    dtm_tfidf_path=None,
    verdict_results_path=None,
):
    """Resolve Step 3 input paths for one dataset/leakage variant."""
    leakage_variant = 'no_leakage' if remove_leakage else 'with_leakage'
    dataset_slug = _safe_slug(dataset_name) if dataset_name else 'all'

    if dataset_name:
        dtm_folder = os.path.join(features_folder, dataset_slug, leakage_variant)
        report_folder = os.path.join(artifacts_folder, dataset_slug, leakage_variant)
    elif remove_leakage:
        dtm_folder = os.path.join(features_folder, leakage_variant)
        report_folder = os.path.join(artifacts_folder, leakage_variant)
    else:
        dtm_folder = features_folder
        report_folder = artifacts_folder

    return {
        'dataset_slug': dataset_slug,
        'leakage_variant': leakage_variant,
        'dtm_bow_path': dtm_bow_path or os.path.join(dtm_folder, 'dtm_csr_BoW.npz'),
        'dtm_tf_path': dtm_tf_path or os.path.join(dtm_folder, 'dtm_csr_TF.npz'),
        'dtm_tfidf_path': dtm_tfidf_path or os.path.join(dtm_folder, 'dtm_csr_TF_IDF.npz'),
        'verdict_results_path': verdict_results_path or os.path.join(report_folder, 'verdict_results.xlsx'),
    }


def load_modeling_data(
    dataset_name=None,
    remove_leakage=False,
    representation='bow',
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    dtm_bow_path=None,
    dtm_tf_path=None,
    dtm_tfidf_path=None,
    verdict_results_path=None,
):
    """Load one Step 3 feature representation and labels."""
    paths = resolve_modeling_paths(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
        dtm_bow_path=dtm_bow_path,
        dtm_tf_path=dtm_tf_path,
        dtm_tfidf_path=dtm_tfidf_path,
        verdict_results_path=verdict_results_path,
    )

    representation = representation.lower().replace('-', '_')
    representation_paths = {
        'bow': paths['dtm_bow_path'],
        'tf': paths['dtm_tf_path'],
        'tfidf': paths['dtm_tfidf_path'],
        'tf_idf': paths['dtm_tfidf_path'],
    }
    if representation not in representation_paths:
        raise ValueError("representation must be one of: bow, tf, tfidf")

    dtm_path = representation_paths[representation]
    representation_slug = 'tfidf' if representation == 'tf_idf' else representation

    if not os.path.exists(dtm_path):
        raise FileNotFoundError(f"DTM file not found: {dtm_path}")
    if not os.path.exists(paths['verdict_results_path']):
        raise FileNotFoundError(f"Label file not found: {paths['verdict_results_path']}")

    x = sp.load_npz(dtm_path)
    y = pd.read_excel(paths['verdict_results_path'], index_col=0).to_numpy().ravel()

    if x.shape[0] != len(y):
        raise ValueError(
            f"Feature/label row mismatch: X has {x.shape[0]} rows, y has {len(y)} labels."
        )

    label_counts = pd.Series(y, name='VERDICT').value_counts().rename_axis('label').reset_index(name='count')
    return {
        **paths,
        'representation': representation_slug,
        'dtm_path': dtm_path,
        'x': x,
        'x_bow': x,
        'y': y,
        'label_counts': label_counts,
    }


def load_jtype_verdict_summary(artifacts_folder='../artifacts/reports'):
    """Load the project-level JTYPE x VERDICT distribution table."""
    labels_csv = os.path.join(artifacts_folder, 'judgment_labels.csv')
    labels_xlsx = os.path.join(artifacts_folder, 'judgment_labels.xlsx')
    if os.path.exists(labels_csv):
        labels = pd.read_csv(labels_csv, encoding='utf-8-sig')
    elif os.path.exists(labels_xlsx):
        labels = pd.read_excel(labels_xlsx)
    else:
        raise FileNotFoundError(
            f"Could not find judgment_labels.csv or judgment_labels.xlsx in {artifacts_folder}."
        )

    required = {'JTYPE', 'VERDICT'}
    missing = required - set(labels.columns)
    if missing:
        raise KeyError(f"judgment_labels is missing required columns: {sorted(missing)}")

    summary = pd.crosstab(labels['JTYPE'], labels['VERDICT'], margins=True)
    valid_mask = labels['JTYPE'].isin(['ADMINISTRATIVE', 'CIVIL', 'CRIMINAL', 'CWC']) & labels['VERDICT'].isin(
        ['Win', 'Lose', 'Mixed']
    )
    valid_summary = pd.crosstab(labels.loc[valid_mask, 'JTYPE'], labels.loc[valid_mask, 'VERDICT'], margins=True)
    return {
        'jtype_verdict_summary': summary,
        'valid_target_summary': valid_summary,
    }


def stratified_sample_modeling_data(x, y, max_rows=None, random_state=42):
    """Return a stratified row sample for quick pipeline tests."""
    y = np.asarray(y).ravel()
    if max_rows is None or max_rows >= x.shape[0]:
        label_counts = pd.Series(y, name='VERDICT').value_counts().rename_axis('label').reset_index(name='count')
        return {
            'x': x,
            'y': y,
            'sample_indices': np.arange(x.shape[0]),
            'label_counts': label_counts,
        }

    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer or None.")

    indices = np.arange(x.shape[0])
    sample_indices, _ = train_test_split(
        indices,
        train_size=max_rows,
        random_state=random_state,
        stratify=y,
    )
    sample_indices = np.sort(sample_indices)
    y_sample = y[sample_indices]
    label_counts = pd.Series(y_sample, name='VERDICT').value_counts().rename_axis('label').reset_index(name='count')
    return {
        'x': x[sample_indices],
        'y': y_sample,
        'sample_indices': sample_indices,
        'label_counts': label_counts,
    }


def split_modeling_data(
    x,
    y,
    train_size=0.7,
    val_size=0.1,
    test_size=0.2,
    random_state=42,
    stratify=True,
):
    """Create train/validation/test splits, stratified by label by default."""
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_size + val_size + test_size must equal 1.0, got {total}.")

    y = np.asarray(y).ravel()
    first_stratify = y if stratify else None
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=first_stratify,
    )

    relative_test_size = test_size / (val_size + test_size)
    second_stratify = y_temp if stratify else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=second_stratify,
    )

    return {
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y_train': np.asarray(y_train).ravel(),
        'y_val': np.asarray(y_val).ravel(),
        'y_test': np.asarray(y_test).ravel(),
        'split_summary': pd.DataFrame(
            [
                {'split': 'train', 'rows': x_train.shape[0], 'features': x_train.shape[1]},
                {'split': 'validation', 'rows': x_val.shape[0], 'features': x_val.shape[1]},
                {'split': 'test', 'rows': x_test.shape[0], 'features': x_test.shape[1]},
            ]
        ),
    }


def apply_chi2_feature_selection(
    x_train,
    y_train,
    x_val,
    x_test,
    k=5000,
    output_dir=None,
    representation='bow',
):
    """Fit chi-square feature selection on train only, then transform val/test."""
    representation = _safe_slug(representation)
    if k is None:
        return {
            'x_train': x_train,
            'x_val': x_val,
            'x_test': x_test,
            'selector': None,
            'selected_feature_indices': None,
            'feature_variant': representation,
            'summary': pd.DataFrame(
                [{'step': 'chi2', 'original_features': x_train.shape[1], 'selected_features': x_train.shape[1]}]
            ),
        }

    requested_k = int(k)
    if requested_k <= 0:
        raise ValueError("k must be a positive integer or None.")

    actual_k = min(requested_k, x_train.shape[1])
    selector = SelectKBest(score_func=chi2, k=actual_k)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_val_selected = selector.transform(x_val)
    x_test_selected = selector.transform(x_test)
    selected_feature_indices = selector.get_support(indices=True)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'chi2_selected_feature_indices.npy'), selected_feature_indices)

    return {
        'x_train': x_train_selected,
        'x_val': x_val_selected,
        'x_test': x_test_selected,
        'selector': selector,
        'selected_feature_indices': selected_feature_indices,
        'feature_variant': f'{representation}_chi2_k{actual_k}',
        'summary': pd.DataFrame(
            [{'step': 'chi2', 'original_features': x_train.shape[1], 'selected_features': actual_k}]
        ),
    }


def prepare_mnir_no_chi2_input(
    no_chi2_data,
    representation='bow',
    feature_limit=None,
    min_train_df=1,
):
    """
    Keep Direct SVM no-chi-square inputs untouched, but optionally trim rare
    columns before MNIR when the full DTM is too wide for textir::srproj.
    """
    representation = _safe_slug(str(representation).lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'

    x_train = no_chi2_data['x_train'].tocsc()
    x_val = no_chi2_data['x_val'].tocsc()
    x_test = no_chi2_data['x_test'].tocsc()
    original_features = int(x_train.shape[1])
    feature_limit = None if feature_limit is None else int(feature_limit)
    min_train_df = int(min_train_df)

    train_df = np.diff(x_train.indptr)
    zero_train_features = int((train_df == 0).sum())
    over_feature_limit = feature_limit is not None and original_features > feature_limit
    should_filter = zero_train_features > 0 or over_feature_limit
    if not should_filter:
        summary = {
            'mnir_no_chi2_lowfreq_filter_applied': False,
            'mnir_no_chi2_feature_limit': feature_limit,
            'mnir_no_chi2_min_train_df': min_train_df,
            'mnir_no_chi2_original_features': original_features,
            'mnir_no_chi2_selected_features': original_features,
            'mnir_no_chi2_removed_features': 0,
            'mnir_no_chi2_zero_train_features': zero_train_features,
            'mnir_no_chi2_filter_reason': 'not_over_feature_limit',
        }
        return {
            **no_chi2_data,
            'feature_variant': representation,
            'selected_feature_indices': None,
            'mnir_filter_summary': summary,
        }

    keep_mask = train_df >= min_train_df
    selected_feature_indices = np.flatnonzero(keep_mask)
    filter_reason = f'train_df_gte_{min_train_df}'

    if feature_limit is not None and selected_feature_indices.size > feature_limit:
        # If a min-df trim is still too wide, keep the most frequent columns.
        order = np.lexsort((selected_feature_indices, -train_df[selected_feature_indices]))
        selected_feature_indices = np.sort(selected_feature_indices[order[:feature_limit]])
        filter_reason = f'train_df_gte_{min_train_df}_then_top_{feature_limit}_by_train_df'

    if selected_feature_indices.size == 0:
        raise ValueError(
            "MNIR no-chi-square low-frequency filter removed all features. "
            f"Lower min_train_df; current min_train_df={min_train_df}."
        )

    selected_features = int(selected_feature_indices.size)
    removed_features = int(original_features - selected_features)
    suffix = f'mnir_min_df{min_train_df}'
    if feature_limit is not None and selected_features == feature_limit:
        suffix += f'_top{feature_limit}'
    feature_variant = f'{representation}_no_chi2_{suffix}'
    summary = {
        'mnir_no_chi2_lowfreq_filter_applied': True,
        'mnir_no_chi2_feature_limit': feature_limit,
        'mnir_no_chi2_min_train_df': min_train_df,
        'mnir_no_chi2_original_features': original_features,
        'mnir_no_chi2_selected_features': selected_features,
        'mnir_no_chi2_removed_features': removed_features,
        'mnir_no_chi2_zero_train_features': zero_train_features,
        'mnir_no_chi2_filter_reason': filter_reason,
    }
    print(
        "[MNIR no-chi2] low-frequency filter applied: "
        f"{original_features} -> {selected_features} features "
        f"(removed {removed_features}; {filter_reason})"
    )
    return {
        'x_train': x_train[:, selected_feature_indices],
        'x_val': x_val[:, selected_feature_indices],
        'x_test': x_test[:, selected_feature_indices],
        'selector': None,
        'selected_feature_indices': selected_feature_indices,
        'feature_variant': feature_variant,
        'summary': pd.DataFrame(
            [{
                'step': 'mnir_no_chi2_lowfreq_filter',
                'original_features': original_features,
                'selected_features': selected_features,
                'removed_features': removed_features,
                'feature_limit': feature_limit,
                'min_train_df': min_train_df,
                'filter_reason': filter_reason,
            }]
        ),
        'mnir_filter_summary': summary,
    }


def _annotate_no_chi2_mnir_results(results, filter_summary):
    metrics = results.get('test_metrics')
    if metrics is not None and not metrics.empty:
        for key, value in filter_summary.items():
            metrics[key] = value
    return results


def _save_mnir_no_chi2_filter_artifacts(mnir_data, mnir_input):
    output_dir = mnir_data.get('mnir_output_dir')
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    selected = mnir_input.get('selected_feature_indices')
    if selected is not None:
        np.save(os.path.join(output_dir, 'mnir_no_chi2_selected_feature_indices.npy'), selected)
    summary = mnir_input.get('mnir_filter_summary')
    if summary is not None:
        pd.DataFrame([summary]).to_csv(
            os.path.join(output_dir, 'mnir_no_chi2_lowfreq_filter_summary.csv'),
            index=False,
            encoding='utf-8-sig',
        )


def run_mnir_feature_extraction(
    x_train,
    y_train,
    x_val,
    x_test,
    dataset_slug='all',
    leakage_variant='with_leakage',
    feature_variant='bow',
    mnir_features_folder='../artifacts/features/mnir',
):
    """Run or load MNIR features for already prepared train/val/test matrices."""
    load_or_fit_feature_splits, _ = _load_mnir_functions()
    mnir_output_dir = os.path.join(mnir_features_folder, dataset_slug, leakage_variant, feature_variant)
    feature_res = load_or_fit_feature_splits(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        X_test=x_test,
        output_dir=mnir_output_dir,
        train_name='mnir_z_train.npy',
        val_name='mnir_z_val.npy',
        test_name='mnir_z_test.npy',
        model_name='mnir_mnlm.rds',
    )
    return {
        **feature_res,
        'mnir_output_dir': mnir_output_dir,
    }


def evaluate_svm_grid(
    z_train,
    y_train,
    z_val,
    y_val,
    z_test,
    y_test,
    svm_grid=None,
    svm_class_weight=None,
    evaluate_test=True,
):
    """Tune SVM C on validation data, then optionally report final test metrics."""
    svm_grid = sorted(svm_grid or [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    best_score = -np.inf
    best_params = None
    validation_rows = []

    for c in svm_grid:
        clf = SVMClassifier(C=c, max_iter=50000, class_weight=svm_class_weight)
        clf.fit(z_train, y_train)
        y_val_pred = clf.predict(z_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        validation_rows.append(
            {'Model': f'SVM (C={c})', 'C': c, 'Validation Accuracy': val_accuracy, 'Validation Macro F1': val_macro_f1}
        )
        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_params = {'C': c, 'max_iter': 50000, 'class_weight': svm_class_weight}

    result = {
        'validation_results': pd.DataFrame(validation_rows),
        'best_params': best_params,
    }
    if not evaluate_test:
        return result

    if z_test is None or y_test is None:
        raise ValueError("z_test and y_test are required when evaluate_test=True.")

    if sp.issparse(z_train) or sp.issparse(z_val):
        x_final = sp.vstack([z_train, z_val])
    else:
        x_final = np.vstack([z_train, z_val])
    y_final = np.concatenate([y_train, y_val])
    final_clf = SVMClassifier(**best_params)
    final_clf.fit(x_final, y_final)
    y_test_pred = final_clf.predict(z_test)

    test_metrics = pd.DataFrame(
        [
            {
                'Model': 'SVM',
                'Best C': best_params['C'],
                'Test Accuracy': accuracy_score(y_test, y_test_pred),
                'Test Macro F1': f1_score(y_test, y_test_pred, average='macro'),
            }
        ]
    )
    test_report = pd.DataFrame(
        classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    ).transpose()
    labels = sorted(pd.unique(np.concatenate([y_test, y_test_pred])))
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_test_pred, labels=labels),
        index=[f'Actual {label}' for label in labels],
        columns=[f'Predicted {label}' for label in labels],
    )

    result.update(
        {
            'test_metrics': test_metrics,
            'test_report': test_report,
            'confusion_matrix': cm,
            'y_test_pred': y_test_pred,
        }
    )
    return result


def evaluate_majority_baseline(y_train, y_test):
    """Evaluate a majority-class baseline on the test set."""
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    majority_label = pd.Series(y_train).value_counts().idxmax()
    y_pred = np.repeat(majority_label, len(y_test))
    labels = sorted(pd.unique(np.concatenate([y_test, y_pred])))
    metrics = pd.DataFrame(
        [
            {
                'Model': 'Majority Class',
                'Best C': np.nan,
                'Test Accuracy': accuracy_score(y_test, y_pred),
                'Test Macro F1': f1_score(y_test, y_pred, average='macro'),
            }
        ]
    )
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=[f'Actual {label}' for label in labels],
        columns=[f'Predicted {label}' for label in labels],
    )
    return {
        'test_metrics': metrics,
        'test_report': report,
        'confusion_matrix': cm,
        'majority_label': majority_label,
        'y_test_pred': y_pred,
    }


def build_model_comparison(proposed_results, baseline_results=None, proposed_name='Proposed MNIR + SVM'):
    """Combine final metrics from proposed and baseline models."""
    rows = []
    if baseline_results:
        for name, result in baseline_results.items():
            metrics = result.get('test_metrics')
            if metrics is not None and not metrics.empty:
                row = metrics.iloc[0].to_dict()
                row['Model'] = name
                rows.append(row)

    metrics = proposed_results.get('test_metrics') if proposed_results else None
    if metrics is not None and not metrics.empty:
        row = metrics.iloc[0].to_dict()
        row['Model'] = proposed_name
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def tune_chi2_k_with_direct_svm(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    k_values,
    svm_grid=None,
    svm_class_weight=None,
    representation='bow',
):
    """
    Select chi-square K using Direct SVM validation Macro F1.

    This is the independent Pipeline A tuner:
    DTM -> train-only chi-square -> SVM validation grid.
    """
    if not k_values:
        raise ValueError("k_values must contain at least one K value or None.")

    tuning_rows = []
    all_validation_rows = []
    best_score = -np.inf
    best_result = None

    for k in k_values:
        chi2_res = apply_chi2_feature_selection(
            x_train,
            y_train,
            x_val,
            x_test,
            k=k,
            output_dir=None,
            representation=representation,
        )
        svm_res = evaluate_svm_grid(
            chi2_res['x_train'],
            y_train,
            chi2_res['x_val'],
            y_val,
            chi2_res['x_test'],
            None,
            svm_grid=svm_grid,
            svm_class_weight=svm_class_weight,
            evaluate_test=False,
        )

        validation_results = svm_res['validation_results'].assign(
            K='all' if k is None else int(k),
            Selected_Features=int(chi2_res['summary'].iloc[0]['selected_features']),
            Feature_Variant=chi2_res['feature_variant'],
        )
        all_validation_rows.append(validation_results)
        best_svm_row = validation_results.sort_values('Validation Macro F1', ascending=False).iloc[0]
        selected_features = int(chi2_res['summary'].iloc[0]['selected_features'])
        row = {
            'K': 'all' if k is None else int(k),
            'Selected Features': selected_features,
            'Feature Variant': chi2_res['feature_variant'],
            'Best C': svm_res['best_params']['C'],
            'Validation Accuracy': float(best_svm_row['Validation Accuracy']),
            'Validation Macro F1': float(best_svm_row['Validation Macro F1']),
        }
        tuning_rows.append(row)

        if row['Validation Macro F1'] > best_score:
            best_score = row['Validation Macro F1']
            best_result = {
                'best_k': k,
                'best_row': row,
                'chi2': chi2_res,
                'svm_validation': svm_res,
            }

    tuning_results = pd.DataFrame(tuning_rows).sort_values('Validation Macro F1', ascending=False).reset_index(drop=True)
    all_validation_results = pd.concat(all_validation_rows, ignore_index=True)
    return {
        'tuning_results': tuning_results,
        'all_validation_results': all_validation_results,
        **best_result,
    }


def tune_chi2_k_with_validation(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    k_values,
    dataset_slug='all',
    leakage_variant='with_leakage',
    mnir_features_folder='../artifacts/features/mnir',
    svm_grid=None,
    svm_class_weight=None,
    representation='bow',
):
    """
    Select chi-square K using validation Macro F1.

    The selector is always fitted on train only. Test features may be transformed
    for cache consistency, but test metrics are not computed during K selection.
    """
    if not k_values:
        raise ValueError("k_values must contain at least one K value or None.")

    tuning_rows = []
    all_validation_rows = []
    best_score = -np.inf
    best_result = None

    for k in k_values:
        chi2_res = apply_chi2_feature_selection(
            x_train,
            y_train,
            x_val,
            x_test,
            k=k,
            output_dir=None,
            representation=representation,
        )
        feature_variant = chi2_res['feature_variant']
        mnir_res = run_mnir_feature_extraction(
            chi2_res['x_train'],
            y_train,
            chi2_res['x_val'],
            chi2_res['x_test'],
            dataset_slug=dataset_slug,
            leakage_variant=leakage_variant,
            feature_variant=feature_variant,
            mnir_features_folder=mnir_features_folder,
        )
        svm_res = evaluate_svm_grid(
            mnir_res['z_train'],
            y_train,
            mnir_res['z_val'],
            y_val,
            mnir_res['z_test'],
            None,
            svm_grid=svm_grid,
            svm_class_weight=svm_class_weight,
            evaluate_test=False,
        )

        validation_results = svm_res['validation_results']
        validation_results = validation_results.assign(
            K='all' if k is None else int(k),
            Selected_Features=int(chi2_res['summary'].iloc[0]['selected_features']),
            Feature_Variant=feature_variant,
        )
        all_validation_rows.append(validation_results)
        best_svm_row = validation_results.sort_values('Validation Macro F1', ascending=False).iloc[0]
        selected_features = int(chi2_res['summary'].iloc[0]['selected_features'])
        row = {
            'K': 'all' if k is None else int(k),
            'Selected Features': selected_features,
            'Feature Variant': feature_variant,
            'Best C': svm_res['best_params']['C'],
            'Validation Accuracy': float(best_svm_row['Validation Accuracy']),
            'Validation Macro F1': float(best_svm_row['Validation Macro F1']),
        }
        tuning_rows.append(row)

        if row['Validation Macro F1'] > best_score:
            best_score = row['Validation Macro F1']
            best_result = {
                'best_k': k,
                'best_row': row,
                'chi2': chi2_res,
                'mnir': mnir_res,
                'svm_validation': svm_res,
            }

    tuning_results = pd.DataFrame(tuning_rows).sort_values('Validation Macro F1', ascending=False).reset_index(drop=True)
    all_validation_results = pd.concat(all_validation_rows, ignore_index=True)
    return {
        'tuning_results': tuning_results,
        'all_validation_results': all_validation_results,
        **best_result,
    }


def save_step3_artifacts(
    output_dir,
    chi2_tuning=None,
    svm_results=None,
    modeling_data=None,
    sampled_data=None,
    split_data=None,
    run_config=None,
    baseline_results=None,
    model_comparison=None,
    dataset_summary=None,
    direct_svm_tuning=None,
):
    """Save Step 3 tables and parameter-surface plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    def _save_df(name, df):
        if df is None:
            return
        path = os.path.join(output_dir, name)
        df.to_csv(path, index=True, encoding='utf-8-sig')
        saved_paths[name] = path

    def _k_sort_key(value):
        return float('inf') if value == 'all' else float(value)

    if run_config is not None:
        run_config_df = pd.DataFrame([run_config])
        _save_df('run_config.csv', run_config_df)

    if dataset_summary is not None:
        _save_df('jtype_verdict_summary.csv', dataset_summary.get('jtype_verdict_summary'))
        _save_df('valid_target_summary.csv', dataset_summary.get('valid_target_summary'))

    if modeling_data is not None:
        _save_df('original_label_counts.csv', modeling_data.get('label_counts'))

    if sampled_data is not None:
        _save_df('active_label_counts.csv', sampled_data.get('label_counts'))

    if split_data is not None:
        _save_df('split_summary.csv', split_data.get('split_summary'))

    if chi2_tuning is not None:
        tuning_results = chi2_tuning.get('tuning_results')
        all_validation_results = chi2_tuning.get('all_validation_results')
        _save_df('chi2_k_tuning_summary.csv', tuning_results)
        _save_df('chi2_k_svm_validation_grid.csv', all_validation_results)

        if tuning_results is not None and not tuning_results.empty:
            plot_df = tuning_results.copy()
            plot_df = plot_df.sort_values('K', key=lambda s: s.map(_k_sort_key))
            x_labels = plot_df['K'].astype(str).tolist()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(x_labels, plot_df['Validation Macro F1'], marker='o')
            ax.set_xlabel('Chi-square K')
            ax.set_ylabel('Validation Macro F1')
            ax.set_title('Chi-square K Tuning')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, 'chi2_k_validation_macro_f1.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved_paths['chi2_k_validation_macro_f1.png'] = path

        if all_validation_results is not None and not all_validation_results.empty:
            heat_df = all_validation_results.copy()
            heat_df['K_label'] = heat_df['K'].astype(str)
            heat_df = heat_df.sort_values('K', key=lambda s: s.map(_k_sort_key))
            pivot = heat_df.pivot_table(
                index='K_label',
                columns='C',
                values='Validation Macro F1',
                aggfunc='max',
            )
            row_order = sorted(pivot.index, key=_k_sort_key)
            pivot = pivot.loc[row_order]
            fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(pivot.columns)), max(4, 0.6 * len(pivot.index))))
            im = ax.imshow(pivot.to_numpy(), aspect='auto', cmap='viridis')
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha='right')
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('SVM C')
            ax.set_ylabel('Chi-square K')
            ax.set_title('Validation Macro F1 Surface')
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iat[i, j]
                    if pd.notna(value):
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white', fontsize=8)
            fig.colorbar(im, ax=ax, label='Validation Macro F1')
            fig.tight_layout()
            path = os.path.join(output_dir, 'chi2_k_svm_c_validation_heatmap.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved_paths['chi2_k_svm_c_validation_heatmap.png'] = path

    if direct_svm_tuning is not None:
        direct_tuning_results = direct_svm_tuning.get('tuning_results')
        direct_validation_results = direct_svm_tuning.get('all_validation_results')
        _save_df('direct_svm_chi2_k_tuning_summary.csv', direct_tuning_results)
        _save_df('direct_svm_chi2_k_svm_validation_grid.csv', direct_validation_results)

        if direct_tuning_results is not None and not direct_tuning_results.empty:
            plot_df = direct_tuning_results.copy()
            plot_df = plot_df.sort_values('K', key=lambda s: s.map(_k_sort_key))
            x_labels = plot_df['K'].astype(str).tolist()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(x_labels, plot_df['Validation Macro F1'], marker='o')
            ax.set_xlabel('Chi-square K')
            ax.set_ylabel('Validation Macro F1')
            ax.set_title('Direct SVM Chi-square K Tuning')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, 'direct_svm_chi2_k_validation_macro_f1.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved_paths['direct_svm_chi2_k_validation_macro_f1.png'] = path

        if direct_validation_results is not None and not direct_validation_results.empty:
            heat_df = direct_validation_results.copy()
            heat_df['K_label'] = heat_df['K'].astype(str)
            heat_df = heat_df.sort_values('K', key=lambda s: s.map(_k_sort_key))
            pivot = heat_df.pivot_table(
                index='K_label',
                columns='C',
                values='Validation Macro F1',
                aggfunc='max',
            )
            row_order = sorted(pivot.index, key=_k_sort_key)
            pivot = pivot.loc[row_order]
            fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(pivot.columns)), max(4, 0.6 * len(pivot.index))))
            im = ax.imshow(pivot.to_numpy(), aspect='auto', cmap='viridis')
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha='right')
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('SVM C')
            ax.set_ylabel('Chi-square K')
            ax.set_title('Direct SVM Validation Macro F1 Surface')
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iat[i, j]
                    if pd.notna(value):
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white', fontsize=8)
            fig.colorbar(im, ax=ax, label='Validation Macro F1')
            fig.tight_layout()
            path = os.path.join(output_dir, 'direct_svm_chi2_k_svm_c_validation_heatmap.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved_paths['direct_svm_chi2_k_svm_c_validation_heatmap.png'] = path

    if svm_results is not None:
        validation_results = svm_results.get('validation_results')
        _save_df('final_svm_validation_grid.csv', validation_results)
        _save_df('final_test_metrics.csv', svm_results.get('test_metrics'))
        _save_df('final_test_report.csv', svm_results.get('test_report'))
        _save_df('final_confusion_matrix.csv', svm_results.get('confusion_matrix'))

        if validation_results is not None and not validation_results.empty:
            plot_df = validation_results.sort_values('C')
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(plot_df['C'].astype(str), plot_df['Validation Macro F1'], marker='o')
            ax.set_xlabel('SVM C')
            ax.set_ylabel('Validation Macro F1')
            ax.set_title('Final SVM C Tuning')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, 'final_svm_c_validation_macro_f1.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved_paths['final_svm_c_validation_macro_f1.png'] = path

        cm = svm_results.get('confusion_matrix')
        if cm is not None and not cm.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm.to_numpy(), cmap='Blues')
            ax.set_xticks(np.arange(len(cm.columns)))
            ax.set_xticklabels(cm.columns, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(cm.index)))
            ax.set_yticklabels(cm.index)
            ax.set_title('Final Test Confusion Matrix')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm.iat[i, j]), ha='center', va='center', color='black', fontsize=9)
            fig.colorbar(im, ax=ax, label='Count')
            fig.tight_layout()
            path = os.path.join(output_dir, 'final_confusion_matrix.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved_paths['final_confusion_matrix.png'] = path

    if baseline_results is not None:
        for baseline_name, result in baseline_results.items():
            slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', baseline_name.strip().lower()).strip('_')
            _save_df(f'baseline_{slug}_test_metrics.csv', result.get('test_metrics'))
            _save_df(f'baseline_{slug}_test_report.csv', result.get('test_report'))
            _save_df(f'baseline_{slug}_confusion_matrix.csv', result.get('confusion_matrix'))
            _save_df(f'baseline_{slug}_validation_grid.csv', result.get('validation_results'))

    if model_comparison is not None and not model_comparison.empty:
        _save_df('model_comparison.csv', model_comparison)
        metric_cols = [col for col in ['Test Accuracy', 'Test Macro F1'] if col in model_comparison.columns]
        if metric_cols:
            plot_df = model_comparison.set_index('Model')[metric_cols]
            fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(plot_df.index)), 4.8))
            plot_df.plot(kind='bar', ax=ax)
            ax.set_ylim(0, max(1.0, float(plot_df.max().max()) * 1.15))
            ax.set_ylabel('Score')
            ax.set_title('Model Comparison on Test Set')
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend(loc='best')
            fig.tight_layout()
            path = os.path.join(output_dir, 'model_comparison_test_metrics.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved_paths['model_comparison_test_metrics.png'] = path

    saved_index = pd.DataFrame(
        [{'artifact': name, 'path': path} for name, path in sorted(saved_paths.items())]
    )
    if not saved_index.empty:
        saved_index_path = os.path.join(output_dir, 'artifact_index.csv')
        saved_index.to_csv(saved_index_path, index=False, encoding='utf-8-sig')
        saved_paths['artifact_index.csv'] = saved_index_path

    return saved_paths


def run_step3_experiment(
    dataset_name,
    remove_leakage,
    representation='bow',
    run_mode='quick',
    run_tag=None,
    max_rows=1200,
    chi2_k_grid=None,
    svm_c_grid=None,
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
    svm_class_weight=None,
    mnir_no_chi2_feature_limit=None,
    mnir_no_chi2_min_train_df=1,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    mnir_features_folder='../artifacts/features/mnir',
    include_dataset_summary=True,
):
    """Run one complete Step 3 experiment and save its artifacts."""
    representation = _safe_slug(representation.lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    run_tag = run_tag or run_mode
    chi2_k_grid = chi2_k_grid or [500, 1000]
    svm_c_grid = svm_c_grid or [0.25, 1.0]

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    sampled_data = stratified_sample_modeling_data(
        modeling_data['x'],
        modeling_data['y'],
        max_rows=max_rows,
        random_state=random_state,
    )
    split_data = split_modeling_data(
        sampled_data['x'],
        sampled_data['y'],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    chi2_tuning = tune_chi2_k_with_validation(
        split_data['x_train'],
        split_data['y_train'],
        split_data['x_val'],
        split_data['y_val'],
        split_data['x_test'],
        k_values=chi2_k_grid,
        dataset_slug=modeling_data['dataset_slug'],
        leakage_variant=modeling_data['leakage_variant'],
        mnir_features_folder=mnir_features_folder,
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        representation=representation,
    )

    chi2_data = chi2_tuning['chi2']
    mnir_data = chi2_tuning['mnir']
    best_chi2_k = chi2_tuning['best_k']

    if chi2_data['selected_feature_indices'] is not None:
        os.makedirs(mnir_data['mnir_output_dir'], exist_ok=True)
        selected_idx_path = os.path.join(mnir_data['mnir_output_dir'], 'chi2_selected_feature_indices.npy')
        np.save(selected_idx_path, chi2_data['selected_feature_indices'])

    svm_results = evaluate_svm_grid(
        mnir_data['z_train'],
        split_data['y_train'],
        mnir_data['z_val'],
        split_data['y_val'],
        mnir_data['z_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )

    direct_svm_tuning = tune_chi2_k_with_direct_svm(
        split_data['x_train'],
        split_data['y_train'],
        split_data['x_val'],
        split_data['y_val'],
        split_data['x_test'],
        k_values=chi2_k_grid,
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        representation=representation,
    )
    direct_chi2_data = direct_svm_tuning['chi2']
    direct_svm_results = evaluate_svm_grid(
        direct_chi2_data['x_train'],
        split_data['y_train'],
        direct_chi2_data['x_val'],
        split_data['y_val'],
        direct_chi2_data['x_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )
    no_chi2_data = apply_chi2_feature_selection(
        split_data['x_train'],
        split_data['y_train'],
        split_data['x_val'],
        split_data['x_test'],
        k=None,
        output_dir=None,
        representation=representation,
    )
    no_chi2_svm_results = evaluate_svm_grid(
        no_chi2_data['x_train'],
        split_data['y_train'],
        no_chi2_data['x_val'],
        split_data['y_val'],
        no_chi2_data['x_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )
    no_chi2_mnir_input = prepare_mnir_no_chi2_input(
        no_chi2_data,
        representation=representation,
        feature_limit=mnir_no_chi2_feature_limit,
        min_train_df=mnir_no_chi2_min_train_df,
    )
    no_chi2_mnir_data = run_mnir_feature_extraction(
        no_chi2_mnir_input['x_train'],
        split_data['y_train'],
        no_chi2_mnir_input['x_val'],
        no_chi2_mnir_input['x_test'],
        dataset_slug=modeling_data['dataset_slug'],
        leakage_variant=modeling_data['leakage_variant'],
        feature_variant=no_chi2_mnir_input['feature_variant'],
        mnir_features_folder=mnir_features_folder,
    )
    _save_mnir_no_chi2_filter_artifacts(no_chi2_mnir_data, no_chi2_mnir_input)
    no_chi2_mnir_svm_results = evaluate_svm_grid(
        no_chi2_mnir_data['z_train'],
        split_data['y_train'],
        no_chi2_mnir_data['z_val'],
        split_data['y_val'],
        no_chi2_mnir_data['z_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )
    no_chi2_mnir_svm_results = _annotate_no_chi2_mnir_results(
        no_chi2_mnir_svm_results,
        no_chi2_mnir_input['mnir_filter_summary'],
    )

    baseline_results = {
        'Majority Class': evaluate_majority_baseline(
            split_data['y_train'],
            split_data['y_test'],
        ),
        f'{representation.upper()} + SVM': direct_svm_results,
        f'{representation.upper()} + SVM (No Chi-square)': no_chi2_svm_results,
        'Proposed MNIR + SVM (No Chi-square)': no_chi2_mnir_svm_results,
    }
    model_comparison = build_model_comparison(svm_results, baseline_results)

    output_dir = os.path.join(
        artifacts_folder,
        modeling_data['dataset_slug'],
        modeling_data['leakage_variant'],
        representation,
        'step3_runs',
        run_tag,
    )
    run_config = {
        'run_mode': run_mode,
        'run_tag': run_tag,
        'dataset_name': dataset_name,
        'dataset_slug': modeling_data['dataset_slug'],
        'remove_leakage': remove_leakage,
        'leakage_variant': modeling_data['leakage_variant'],
        'representation': representation,
        'dtm_path': modeling_data['dtm_path'],
        'max_rows': max_rows,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'random_state': random_state,
        'chi2_k_grid': chi2_k_grid,
        'svm_c_grid': svm_c_grid,
        'svm_class_weight': svm_class_weight,
        'best_chi2_k': 'all' if best_chi2_k is None else best_chi2_k,
        'best_svm_c': svm_results['best_params']['C'],
        'best_direct_chi2_k': 'all' if direct_svm_tuning['best_k'] is None else direct_svm_tuning['best_k'],
        'best_direct_svm_c': direct_svm_results['best_params']['C'],
        'no_chi2_svm_c': no_chi2_svm_results['best_params']['C'],
        'no_chi2_mnir_svm_c': no_chi2_mnir_svm_results['best_params']['C'],
        **no_chi2_mnir_input['mnir_filter_summary'],
        'direct_svm_tuning': 'independent_validation_macro_f1',
        'baseline_models': list(baseline_results.keys()),
    }
    dataset_summary = load_jtype_verdict_summary(artifacts_folder=artifacts_folder) if include_dataset_summary else None
    saved_paths = save_step3_artifacts(
        output_dir=output_dir,
        chi2_tuning=chi2_tuning,
        svm_results=svm_results,
        modeling_data=modeling_data,
        sampled_data=sampled_data,
        split_data=split_data,
        run_config=run_config,
        baseline_results=baseline_results,
        model_comparison=model_comparison,
        dataset_summary=dataset_summary,
        direct_svm_tuning=direct_svm_tuning,
    )

    return {
        'run_config': run_config,
        'output_dir': output_dir,
        'saved_paths': saved_paths,
        'modeling_data': modeling_data,
        'sampled_data': sampled_data,
        'split_data': split_data,
        'chi2_tuning': chi2_tuning,
        'chi2_data': chi2_data,
        'mnir_data': mnir_data,
        'svm_results': svm_results,
        'direct_svm_tuning': direct_svm_tuning,
        'direct_chi2_data': direct_chi2_data,
        'direct_svm_results': direct_svm_results,
        'no_chi2_data': no_chi2_data,
        'no_chi2_svm_results': no_chi2_svm_results,
        'no_chi2_mnir_input': no_chi2_mnir_input,
        'no_chi2_mnir_data': no_chi2_mnir_data,
        'no_chi2_mnir_svm_results': no_chi2_mnir_svm_results,
        'baseline_results': baseline_results,
        'model_comparison': model_comparison,
    }


def _step3_output_dir(artifacts_folder, dataset_name, remove_leakage, representation, run_tag):
    dataset_slug = _safe_slug(dataset_name) if dataset_name else 'all'
    leakage_variant = 'no_leakage' if remove_leakage else 'with_leakage'
    representation = _safe_slug(str(representation).lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    return os.path.join(
        artifacts_folder,
        dataset_slug,
        leakage_variant,
        representation,
        'step3_runs',
        run_tag,
    )


def _read_step3_run_config(output_dir):
    path = os.path.join(output_dir, 'run_config.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, index_col=0)
    return df.iloc[0].to_dict() if not df.empty else {}


def _read_step3_model_comparison(output_dir):
    path = os.path.join(output_dir, 'model_comparison.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


def _step3_required_model_names(representation):
    representation = _safe_slug(str(representation).lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    rep = representation.upper()
    return {
        'Majority Class',
        f'{rep} + SVM',
        f'{rep} + SVM (No Chi-square)',
        'Proposed MNIR + SVM',
        'Proposed MNIR + SVM (No Chi-square)',
    }


def _model_metric_row(model_comparison, model_name):
    if model_comparison is None or model_comparison.empty or 'Model' not in model_comparison.columns:
        return {}
    rows = model_comparison[model_comparison['Model'] == model_name]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _step3_summary_from_existing(output_dir, dataset_name, remove_leakage, representation, run_tag, status='skipped_existing'):
    run_config = _read_step3_run_config(output_dir)
    model_comparison = _read_step3_model_comparison(output_dir)
    rep = _safe_slug(str(representation).lower().replace('-', '_'))
    if rep == 'tf_idf':
        rep = 'tfidf'

    mnir = _model_metric_row(model_comparison, 'Proposed MNIR + SVM')
    direct = _model_metric_row(model_comparison, f'{rep.upper()} + SVM')
    no_chi2_direct = _model_metric_row(model_comparison, f'{rep.upper()} + SVM (No Chi-square)')
    no_chi2_mnir = _model_metric_row(model_comparison, 'Proposed MNIR + SVM (No Chi-square)')

    return {
        **run_config,
        'run_tag': run_config.get('run_tag', run_tag),
        'dataset_name': run_config.get('dataset_name', dataset_name),
        'dataset_slug': run_config.get('dataset_slug', _safe_slug(dataset_name)),
        'remove_leakage': run_config.get('remove_leakage', remove_leakage),
        'leakage_variant': run_config.get('leakage_variant', 'no_leakage' if remove_leakage else 'with_leakage'),
        'representation': run_config.get('representation', rep),
        'output_dir': output_dir,
        'run_status': status,
        'Test Accuracy': mnir.get('Test Accuracy'),
        'Test Macro F1': mnir.get('Test Macro F1'),
        'Direct SVM Independent Accuracy': direct.get('Test Accuracy'),
        'Direct SVM Independent Macro F1': direct.get('Test Macro F1'),
        'Direct SVM No Chi-square Accuracy': no_chi2_direct.get('Test Accuracy'),
        'Direct SVM No Chi-square Macro F1': no_chi2_direct.get('Test Macro F1'),
        'MNIR + SVM No Chi-square Accuracy': no_chi2_mnir.get('Test Accuracy'),
        'MNIR + SVM No Chi-square Macro F1': no_chi2_mnir.get('Test Macro F1'),
    }


def patch_step3_no_chi2_results(
    dataset_name,
    remove_leakage,
    representation='bow',
    run_mode='full',
    run_tag='full',
    max_rows=None,
    svm_c_grid=None,
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
    svm_class_weight=None,
    mnir_no_chi2_feature_limit=None,
    mnir_no_chi2_min_train_df=1,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    mnir_features_folder='../artifacts/features/mnir',
    **_ignored,
):
    """Patch only no-chi-square Step 3 comparison rows into an existing run."""
    representation = _safe_slug(str(representation).lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    svm_c_grid = svm_c_grid or [0.25, 1.0]

    output_dir = _step3_output_dir(
        artifacts_folder,
        dataset_name,
        remove_leakage,
        representation,
        run_tag,
    )
    model_comparison = _read_step3_model_comparison(output_dir)
    if model_comparison is None or model_comparison.empty:
        raise FileNotFoundError(f"Cannot patch no-chi-square rows without model_comparison.csv: {output_dir}")

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    sampled_data = stratified_sample_modeling_data(
        modeling_data['x'],
        modeling_data['y'],
        max_rows=max_rows,
        random_state=random_state,
    )
    split_data = split_modeling_data(
        sampled_data['x'],
        sampled_data['y'],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    no_chi2_data = apply_chi2_feature_selection(
        split_data['x_train'],
        split_data['y_train'],
        split_data['x_val'],
        split_data['x_test'],
        k=None,
        output_dir=None,
        representation=representation,
    )
    no_chi2_svm_results = evaluate_svm_grid(
        no_chi2_data['x_train'],
        split_data['y_train'],
        no_chi2_data['x_val'],
        split_data['y_val'],
        no_chi2_data['x_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )
    no_chi2_mnir_input = prepare_mnir_no_chi2_input(
        no_chi2_data,
        representation=representation,
        feature_limit=mnir_no_chi2_feature_limit,
        min_train_df=mnir_no_chi2_min_train_df,
    )
    no_chi2_mnir_data = run_mnir_feature_extraction(
        no_chi2_mnir_input['x_train'],
        split_data['y_train'],
        no_chi2_mnir_input['x_val'],
        no_chi2_mnir_input['x_test'],
        dataset_slug=modeling_data['dataset_slug'],
        leakage_variant=modeling_data['leakage_variant'],
        feature_variant=no_chi2_mnir_input['feature_variant'],
        mnir_features_folder=mnir_features_folder,
    )
    _save_mnir_no_chi2_filter_artifacts(no_chi2_mnir_data, no_chi2_mnir_input)
    no_chi2_mnir_svm_results = evaluate_svm_grid(
        no_chi2_mnir_data['z_train'],
        split_data['y_train'],
        no_chi2_mnir_data['z_val'],
        split_data['y_val'],
        no_chi2_mnir_data['z_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )
    no_chi2_mnir_svm_results = _annotate_no_chi2_mnir_results(
        no_chi2_mnir_svm_results,
        no_chi2_mnir_input['mnir_filter_summary'],
    )

    new_rows = []
    for model_name, result in {
        f'{representation.upper()} + SVM (No Chi-square)': no_chi2_svm_results,
        'Proposed MNIR + SVM (No Chi-square)': no_chi2_mnir_svm_results,
    }.items():
        metrics = result.get('test_metrics')
        if metrics is not None and not metrics.empty:
            row = metrics.iloc[0].to_dict()
            row['Model'] = model_name
            new_rows.append(row)

    remove_models = {row['Model'] for row in new_rows}
    model_comparison = model_comparison[~model_comparison['Model'].isin(remove_models)]
    model_comparison = pd.concat([model_comparison, pd.DataFrame(new_rows)], ignore_index=True)

    run_config = _read_step3_run_config(output_dir)
    run_config.update({
        'run_mode': run_config.get('run_mode', run_mode),
        'run_tag': run_config.get('run_tag', run_tag),
        'dataset_name': run_config.get('dataset_name', dataset_name),
        'dataset_slug': run_config.get('dataset_slug', modeling_data['dataset_slug']),
        'remove_leakage': run_config.get('remove_leakage', remove_leakage),
        'leakage_variant': run_config.get('leakage_variant', modeling_data['leakage_variant']),
        'representation': run_config.get('representation', representation),
        'svm_class_weight': run_config.get('svm_class_weight', svm_class_weight),
        'no_chi2_svm_c': no_chi2_svm_results['best_params']['C'],
        'no_chi2_mnir_svm_c': no_chi2_mnir_svm_results['best_params']['C'],
        **no_chi2_mnir_input['mnir_filter_summary'],
        'no_chi2_patch_run_mode': run_mode,
        'no_chi2_patch_run_tag': run_tag,
    })

    saved_paths = save_step3_artifacts(
        output_dir=output_dir,
        run_config=run_config,
        baseline_results={
            f'{representation.upper()} + SVM (No Chi-square)': no_chi2_svm_results,
            'Proposed MNIR + SVM (No Chi-square)': no_chi2_mnir_svm_results,
        },
        model_comparison=model_comparison,
    )
    return {
        'run_config': run_config,
        'output_dir': output_dir,
        'saved_paths': saved_paths,
        'no_chi2_data': no_chi2_data,
        'no_chi2_svm_results': no_chi2_svm_results,
        'no_chi2_mnir_input': no_chi2_mnir_input,
        'no_chi2_mnir_data': no_chi2_mnir_data,
        'no_chi2_mnir_svm_results': no_chi2_mnir_svm_results,
        'model_comparison': model_comparison,
    }


def run_step3_batch(
    dataset_names,
    remove_leakage_values,
    representations,
    run_mode='quick',
    run_tag=None,
    skip_existing=True,
    patch_missing_no_chi2=True,
    allow_new_runs=True,
    **kwargs,
):
    """Run Step 3 over dataset x leakage x representation combinations."""
    results = []
    failures = []
    run_tag = run_tag or run_mode

    for dataset_name, remove_leakage, representation in _batch_combinations(
        dataset_names,
        remove_leakage_values,
        representations,
        desc='Step 3 batch',
    ):
        try:
            output_dir = _step3_output_dir(
                kwargs.get('artifacts_folder', '../artifacts/reports'),
                dataset_name,
                remove_leakage,
                representation,
                run_tag,
            )
            existing_comparison = _read_step3_model_comparison(output_dir)
            if skip_existing and existing_comparison is not None and 'Model' in existing_comparison.columns:
                existing_models = set(existing_comparison['Model'].astype(str))
                required_models = _step3_required_model_names(representation)
                if required_models.issubset(existing_models):
                    print(
                        f"[Step3 skip] {dataset_name} / "
                        f"{'no_leakage' if remove_leakage else 'with_leakage'} / {representation}"
                    )
                    results.append(
                        _step3_summary_from_existing(
                            output_dir,
                            dataset_name,
                            remove_leakage,
                            representation,
                            run_tag,
                            status='skipped_existing',
                        )
                    )
                    continue

                rep_slug = _safe_slug(str(representation).lower().replace('-', '_'))
                if rep_slug == 'tf_idf':
                    rep_slug = 'tfidf'
                no_chi2_models = {
                    f'{rep_slug.upper()} + SVM (No Chi-square)',
                    'Proposed MNIR + SVM (No Chi-square)',
                }
                missing_no_chi2 = no_chi2_models - existing_models
                base_models = required_models - no_chi2_models
                if patch_missing_no_chi2 and missing_no_chi2 and base_models.issubset(existing_models):
                    print(
                        f"[Step3 patch no-chi2] {dataset_name} / "
                        f"{'no_leakage' if remove_leakage else 'with_leakage'} / {representation}"
                    )
                    patch_step3_no_chi2_results(
                        dataset_name=dataset_name,
                        remove_leakage=remove_leakage,
                        representation=representation,
                        run_mode=run_mode,
                        run_tag=run_tag,
                        **kwargs,
                    )
                    results.append(
                        _step3_summary_from_existing(
                            output_dir,
                            dataset_name,
                            remove_leakage,
                            representation,
                            run_tag,
                            status='patched_no_chi2',
                        )
                    )
                    continue

                if not allow_new_runs:
                    print(
                        f"[Step3 skip incomplete] {dataset_name} / "
                        f"{'no_leakage' if remove_leakage else 'with_leakage'} / {representation}"
                    )
                    results.append(
                        _step3_summary_from_existing(
                            output_dir,
                            dataset_name,
                            remove_leakage,
                            representation,
                            run_tag,
                            status='skipped_incomplete_existing',
                        )
                    )
                    continue

            if not allow_new_runs:
                print(
                    f"[Step3 skip no existing run] {dataset_name} / "
                    f"{'no_leakage' if remove_leakage else 'with_leakage'} / {representation}"
                )
                failures.append({
                    'dataset_name': dataset_name,
                    'remove_leakage': remove_leakage,
                    'representation': representation,
                    'error': 'Skipped because allow_new_runs=False and no complete reusable Step 3 run exists.',
                })
                continue

            print(
                f"[Step3] {dataset_name} / "
                f"{'no_leakage' if remove_leakage else 'with_leakage'} / {representation}"
            )
            result = run_step3_experiment(
                dataset_name=dataset_name,
                remove_leakage=remove_leakage,
                representation=representation,
                run_mode=run_mode,
                run_tag=run_tag,
                **kwargs,
            )
            metrics = result['svm_results']['test_metrics'].iloc[0].to_dict()
            direct_metrics = result['direct_svm_results']['test_metrics'].iloc[0].to_dict()
            no_chi2_metrics = result['no_chi2_svm_results']['test_metrics'].iloc[0].to_dict()
            no_chi2_mnir_metrics = result['no_chi2_mnir_svm_results']['test_metrics'].iloc[0].to_dict()
            results.append({
                **result['run_config'],
                'output_dir': result['output_dir'],
                'run_status': 'ran_full',
                'Test Accuracy': metrics.get('Test Accuracy'),
                'Test Macro F1': metrics.get('Test Macro F1'),
                'Direct SVM Independent Accuracy': direct_metrics.get('Test Accuracy'),
                'Direct SVM Independent Macro F1': direct_metrics.get('Test Macro F1'),
                'Direct SVM No Chi-square Accuracy': no_chi2_metrics.get('Test Accuracy'),
                'Direct SVM No Chi-square Macro F1': no_chi2_metrics.get('Test Macro F1'),
                'MNIR + SVM No Chi-square Accuracy': no_chi2_mnir_metrics.get('Test Accuracy'),
                'MNIR + SVM No Chi-square Macro F1': no_chi2_mnir_metrics.get('Test Macro F1'),
            })
        except Exception as exc:
            failures.append({
                'dataset_name': dataset_name,
                'remove_leakage': remove_leakage,
                'representation': representation,
                'error': repr(exc),
            })

    summary_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)
    return {
        'summary_df': summary_df,
        'failures_df': failures_df,
    }

def run_direct_svm_experiment(
    dataset_name,
    remove_leakage,
    representation='bow',
    run_mode='quick',
    run_tag=None,
    max_rows=1200,
    chi2_k_grid=None,
    svm_c_grid=None,
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
    svm_class_weight=None,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    include_dataset_summary=True,
):
    """Run Pipeline A: DTM -> chi-square K tuning -> Direct SVM."""
    representation = _safe_slug(representation.lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    run_tag = run_tag or run_mode
    chi2_k_grid = chi2_k_grid or [500, 1000]
    svm_c_grid = svm_c_grid or [0.25, 1.0]

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    sampled_data = stratified_sample_modeling_data(
        modeling_data['x'],
        modeling_data['y'],
        max_rows=max_rows,
        random_state=random_state,
    )
    split_data = split_modeling_data(
        sampled_data['x'],
        sampled_data['y'],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    chi2_tuning = tune_chi2_k_with_direct_svm(
        split_data['x_train'],
        split_data['y_train'],
        split_data['x_val'],
        split_data['y_val'],
        split_data['x_test'],
        k_values=chi2_k_grid,
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        representation=representation,
    )
    chi2_data = chi2_tuning['chi2']
    best_chi2_k = chi2_tuning['best_k']
    direct_svm_tuning = chi2_tuning

    output_dir = os.path.join(
        artifacts_folder,
        modeling_data['dataset_slug'],
        modeling_data['leakage_variant'],
        representation,
        'step4_direct_svm_runs',
        run_tag,
    )
    if chi2_data['selected_feature_indices'] is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'direct_svm_chi2_selected_feature_indices.npy'), chi2_data['selected_feature_indices'])

    direct_svm_results = evaluate_svm_grid(
        chi2_data['x_train'],
        split_data['y_train'],
        chi2_data['x_val'],
        split_data['y_val'],
        chi2_data['x_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )
    baseline_results = {
        'Majority Class': evaluate_majority_baseline(
            split_data['y_train'],
            split_data['y_test'],
        ),
    }
    direct_model_name = f'{representation.upper()} + Direct SVM'
    model_comparison = build_model_comparison(
        direct_svm_results,
        baseline_results,
        proposed_name=direct_model_name,
    )

    run_config = {
        'run_mode': run_mode,
        'run_tag': run_tag,
        'dataset_name': dataset_name,
        'dataset_slug': modeling_data['dataset_slug'],
        'remove_leakage': remove_leakage,
        'leakage_variant': modeling_data['leakage_variant'],
        'representation': representation,
        'dtm_path': modeling_data['dtm_path'],
        'max_rows': max_rows,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'random_state': random_state,
        'chi2_k_grid': chi2_k_grid,
        'svm_c_grid': svm_c_grid,
        'svm_class_weight': svm_class_weight,
        'best_direct_chi2_k': 'all' if best_chi2_k is None else best_chi2_k,
        'best_direct_svm_c': direct_svm_results['best_params']['C'],
        'model_name': direct_model_name,
    }
    dataset_summary = load_jtype_verdict_summary(artifacts_folder=artifacts_folder) if include_dataset_summary else None
    saved_paths = save_step3_artifacts(
        output_dir=output_dir,
        chi2_tuning=chi2_tuning,
        svm_results=direct_svm_results,
        modeling_data=modeling_data,
        sampled_data=sampled_data,
        split_data=split_data,
        run_config=run_config,
        baseline_results=baseline_results,
        model_comparison=model_comparison,
        dataset_summary=dataset_summary,
        direct_svm_tuning=direct_svm_tuning,
    )

    return {
        'run_config': run_config,
        'output_dir': output_dir,
        'saved_paths': saved_paths,
        'modeling_data': modeling_data,
        'sampled_data': sampled_data,
        'split_data': split_data,
        'chi2_tuning': chi2_tuning,
        'chi2_data': chi2_data,
        'direct_svm_results': direct_svm_results,
        'baseline_results': baseline_results,
        'model_comparison': model_comparison,
    }


def run_direct_svm_batch(
    dataset_names,
    remove_leakage_values,
    representations,
    run_mode='quick',
    run_tag=None,
    **kwargs,
):
    """Run Pipeline A over dataset x leakage x representation combinations."""
    results = []
    failures = []
    run_tag = run_tag or run_mode
    for dataset_name, remove_leakage, representation in _batch_combinations(
        dataset_names,
        remove_leakage_values,
        representations,
        desc='Direct SVM batch',
    ):
        try:
            print(
                f"[Step4 Direct SVM] {dataset_name} / "
                f"{'no_leakage' if remove_leakage else 'with_leakage'} / {representation}"
            )
            result = run_direct_svm_experiment(
                dataset_name=dataset_name,
                remove_leakage=remove_leakage,
                representation=representation,
                run_mode=run_mode,
                run_tag=run_tag,
                **kwargs,
            )
            metrics = result['direct_svm_results']['test_metrics'].iloc[0].to_dict()
            results.append({
                **result['run_config'],
                'output_dir': result['output_dir'],
                'Test Accuracy': metrics.get('Test Accuracy'),
                'Test Macro F1': metrics.get('Test Macro F1'),
            })
        except Exception as exc:
            failures.append({
                'dataset_name': dataset_name,
                'remove_leakage': remove_leakage,
                'representation': representation,
                'error': repr(exc),
            })

    return {
        'summary_df': pd.DataFrame(results),
        'failures_df': pd.DataFrame(failures),
    }

def patch_step3_direct_svm_baseline(
    dataset_name,
    remove_leakage,
    representation='bow',
    run_mode='full',
    run_tag='full_20260504_154912',
    max_rows=None,
    chi2_k_grid=None,
    svm_c_grid=None,
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
    svm_class_weight=None,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
):
    """
    Recompute only the Direct SVM baseline and patch an existing Step 3 run.

    Existing MNIR + SVM final_* files are read from the Step 3 run folder and
    left untouched. This updates the baseline files, direct-SVM tuning files,
    model_comparison.csv, model_comparison_test_metrics.png, and run_config.csv.
    """
    representation = _safe_slug(representation.lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    chi2_k_grid = chi2_k_grid or [1000, 3000, 5000, 10000]
    svm_c_grid = svm_c_grid or [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    output_dir = os.path.join(
        artifacts_folder,
        modeling_data['dataset_slug'],
        modeling_data['leakage_variant'],
        representation,
        'step3_runs',
        run_tag,
    )
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Existing Step 3 run folder not found: {output_dir}")

    sampled_data = stratified_sample_modeling_data(
        modeling_data['x'],
        modeling_data['y'],
        max_rows=max_rows,
        random_state=random_state,
    )
    split_data = split_modeling_data(
        sampled_data['x'],
        sampled_data['y'],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    direct_svm_tuning = tune_chi2_k_with_direct_svm(
        split_data['x_train'],
        split_data['y_train'],
        split_data['x_val'],
        split_data['y_val'],
        split_data['x_test'],
        k_values=chi2_k_grid,
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        representation=representation,
    )
    direct_chi2_data = direct_svm_tuning['chi2']
    direct_svm_results = evaluate_svm_grid(
        direct_chi2_data['x_train'],
        split_data['y_train'],
        direct_chi2_data['x_val'],
        split_data['y_val'],
        direct_chi2_data['x_test'],
        split_data['y_test'],
        svm_grid=svm_c_grid,
        svm_class_weight=svm_class_weight,
        evaluate_test=True,
    )
    majority_results = evaluate_majority_baseline(split_data['y_train'], split_data['y_test'])

    proposed_results = {
        'test_metrics': pd.read_csv(os.path.join(output_dir, 'final_test_metrics.csv'), index_col=0),
        'test_report': pd.read_csv(os.path.join(output_dir, 'final_test_report.csv'), index_col=0),
        'confusion_matrix': pd.read_csv(os.path.join(output_dir, 'final_confusion_matrix.csv'), index_col=0),
    }
    validation_path = os.path.join(output_dir, 'final_svm_validation_grid.csv')
    if os.path.exists(validation_path):
        proposed_results['validation_results'] = pd.read_csv(validation_path, index_col=0)

    baseline_results = {
        'Majority Class': majority_results,
        f'{representation.upper()} + SVM': direct_svm_results,
    }
    model_comparison = build_model_comparison(proposed_results, baseline_results)

    # Preserve existing run_config columns and append patch metadata.
    run_config_path = os.path.join(output_dir, 'run_config.csv')
    if os.path.exists(run_config_path):
        run_config_df = pd.read_csv(run_config_path)
        run_config = run_config_df.iloc[0].to_dict() if not run_config_df.empty else {}
    else:
        run_config = {}
    run_config.update(
        {
            'direct_svm_tuning': 'independent_validation_macro_f1',
            'best_direct_chi2_k': 'all' if direct_svm_tuning['best_k'] is None else direct_svm_tuning['best_k'],
            'best_direct_svm_c': direct_svm_results['best_params']['C'],
            'svm_class_weight': run_config.get('svm_class_weight', svm_class_weight),
            'direct_svm_patch_run_mode': run_mode,
            'direct_svm_patch_run_tag': run_tag,
            'direct_svm_patch_max_rows': max_rows,
        }
    )

    saved_paths = save_step3_artifacts(
        output_dir=output_dir,
        baseline_results=baseline_results,
        model_comparison=model_comparison,
        run_config=run_config,
        direct_svm_tuning=direct_svm_tuning,
    )

    return {
        'run_config': run_config,
        'output_dir': output_dir,
        'saved_paths': saved_paths,
        'direct_svm_tuning': direct_svm_tuning,
        'direct_chi2_data': direct_chi2_data,
        'direct_svm_results': direct_svm_results,
        'baseline_results': baseline_results,
        'model_comparison': model_comparison,
    }


def patch_step3_direct_svm_batch(
    dataset_names,
    remove_leakage_values,
    representations,
    run_mode='full',
    run_tag='full_20260504_154912',
    **kwargs,
):
    """Patch Direct SVM baselines for existing Step 3 run folders."""
    results = []
    failures = []
    for dataset_name, remove_leakage, representation in _batch_combinations(
        dataset_names,
        remove_leakage_values,
        representations,
        desc='Patch Direct SVM batch',
    ):
        try:
            print(
                f"[Patch Direct SVM] {dataset_name} / "
                f"{'no_leakage' if remove_leakage else 'with_leakage'} / {representation}"
            )
            result = patch_step3_direct_svm_baseline(
                dataset_name=dataset_name,
                remove_leakage=remove_leakage,
                representation=representation,
                run_mode=run_mode,
                run_tag=run_tag,
                **kwargs,
            )
            metrics = result['direct_svm_results']['test_metrics'].iloc[0].to_dict()
            results.append({
                **result['run_config'],
                'output_dir': result['output_dir'],
                'Direct SVM Independent Accuracy': metrics.get('Test Accuracy'),
                'Direct SVM Independent Macro F1': metrics.get('Test Macro F1'),
            })
        except Exception as exc:
            failures.append({
                'dataset_name': dataset_name,
                'remove_leakage': remove_leakage,
                'representation': representation,
                'error': repr(exc),
            })

    return {
        'summary_df': pd.DataFrame(results),
        'failures_df': pd.DataFrame(failures),
    }


def _load_vocab_for_modeling_data(modeling_data, representation):
    vocab_file = {
        'bow': 'vocab_BoW.npy',
        'tf': 'vocab_TF.npy',
        'tfidf': 'vocab_TF_IDF.npy',
    }[representation]
    vocab_path = Path(modeling_data['dtm_path']).parent / vocab_file
    if not vocab_path.exists():
        return None
    vocab = np.load(vocab_path, allow_pickle=True).astype(str)
    if len(vocab) != modeling_data['x'].shape[1]:
        return None
    return vocab


def preview_original_dtm(
    dataset_name,
    remove_leakage=False,
    representation='tfidf',
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    n_docs=5,
    top_terms_per_doc=20,
):
    """Return a long-format preview of nonzero full-DTM values before chi-square."""
    representation = _safe_slug(str(representation).lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    paths = resolve_modeling_paths(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    labels = pd.read_excel(paths['verdict_results_path'], index_col=0)
    doc_ids = labels.index.to_numpy()
    vocab = _load_vocab_for_modeling_data(modeling_data, representation)
    x = modeling_data['x'].tocsr()

    rows = []
    for doc_position in range(min(int(n_docs), x.shape[0])):
        row = x.getrow(doc_position)
        order = np.argsort(row.data)[::-1][:int(top_terms_per_doc)]
        for pos in order:
            feature_index = int(row.indices[pos])
            rows.append({
                'dataset': dataset_name,
                'leakage': modeling_data['leakage_variant'],
                'representation': representation,
                'doc_position': doc_position,
                'doc_id': doc_ids[doc_position],
                'label': modeling_data['y'][doc_position],
                'feature_index': feature_index,
                'term': vocab[feature_index] if vocab is not None else f'feature_{feature_index}',
                'value': float(row.data[pos]),
                'vocab_available': vocab is not None,
            })
    return pd.DataFrame(rows)


def preview_original_dtm_matrix(
    dataset_name,
    remove_leakage=False,
    representation='tfidf',
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    n_docs=8,
    n_terms=30,
    column_mode='nonzero',
):
    """Return a small document-by-term matrix preview before chi-square."""
    representation = _safe_slug(str(representation).lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    if column_mode not in {'nonzero', 'first'}:
        raise ValueError("column_mode must be 'nonzero' or 'first'.")
    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    paths = resolve_modeling_paths(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    labels = pd.read_excel(paths['verdict_results_path'], index_col=0)
    doc_ids = labels.index.to_numpy()
    vocab = _load_vocab_for_modeling_data(modeling_data, representation)
    x = modeling_data['x'].tocsr()
    n_docs = min(int(n_docs), x.shape[0])
    n_terms = min(int(n_terms), x.shape[1])
    x_rows = x[:n_docs]
    if column_mode == 'first':
        selected_cols = np.arange(n_terms)
    else:
        selected_cols = np.unique(x_rows.indices)[:n_terms]
        if len(selected_cols) == 0:
            selected_cols = np.arange(n_terms)
    column_names = [str(vocab[i]) if vocab is not None else f'feature_{int(i)}' for i in selected_cols]
    df = pd.DataFrame(x_rows[:, selected_cols].toarray(), columns=column_names)
    df.insert(0, 'label', modeling_data['y'][:n_docs])
    df.insert(0, 'doc_id', doc_ids[:n_docs])
    df.insert(0, 'doc_position', np.arange(n_docs))
    df.attrs['dtm_shape'] = x.shape
    df.attrs['selected_feature_indices'] = selected_cols.tolist()
    df.attrs['vocab_available'] = vocab is not None
    return df


def preview_dtm_before_after_chi2(
    dataset_name,
    remove_leakage=False,
    representation='tfidf',
    k=3000,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
    split='train',
    n_docs=8,
    samples_per_label=None,
    label_order=None,
    n_terms=30,
    column_mode='nonzero',
):
    """Return DTM matrix previews for the same sampled documents before/after chi-square."""
    representation = _safe_slug(str(representation).lower().replace('-', '_'))
    if representation == 'tf_idf':
        representation = 'tfidf'
    if split not in {'train', 'validation', 'test'}:
        raise ValueError("split must be one of: train, validation, test.")
    if column_mode not in {'nonzero', 'first'}:
        raise ValueError("column_mode must be 'nonzero' or 'first'.")

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    paths = resolve_modeling_paths(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    labels = pd.read_excel(paths['verdict_results_path'], index_col=0)
    doc_ids = labels.index.to_numpy()
    vocab = _load_vocab_for_modeling_data(modeling_data, representation)
    x = modeling_data['x'].tocsr()
    y = np.asarray(modeling_data['y']).ravel()

    indices = np.arange(x.shape[0])
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, y, train_size=train_size, random_state=random_state, stratify=y
    )
    relative_test_size = test_size / (val_size + test_size)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )
    split_indices = {'train': train_idx, 'validation': val_idx, 'test': test_idx}
    all_row_indices = split_indices[split]
    split_labels = y[all_row_indices]
    if samples_per_label is not None:
        rng = np.random.default_rng(random_state)
        positions = []
        labels_to_sample = list(label_order) if label_order is not None else sorted(pd.unique(split_labels))
        for label in labels_to_sample:
            label_positions = np.flatnonzero(split_labels == label)
            if len(label_positions):
                take = min(int(samples_per_label), len(label_positions))
                positions.extend(rng.choice(label_positions, size=take, replace=False).tolist())
        split_positions = np.asarray(positions, dtype=int)
    else:
        split_positions = np.arange(min(int(n_docs), len(all_row_indices)))
    row_indices = all_row_indices[split_positions]

    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    chi2_res = apply_chi2_feature_selection(
        x_train, y[train_idx], x_val, x_test, k=k, output_dir=None, representation=representation
    )
    after_matrix = {'train': chi2_res['x_train'], 'validation': chi2_res['x_val'], 'test': chi2_res['x_test']}[split].tocsr()
    selected_feature_indices = chi2_res['selected_feature_indices']
    if selected_feature_indices is None:
        selected_feature_indices = np.arange(x.shape[1])

    def feature_names(feature_indices):
        return [str(vocab[i]) if vocab is not None else f'feature_{int(i)}' for i in feature_indices]

    def matrix_preview(matrix, matrix_positions, original_rows, feature_indices):
        matrix_rows = matrix[matrix_positions]
        col_count = min(int(n_terms), len(feature_indices))
        if column_mode == 'first':
            local_cols = np.arange(col_count)
        else:
            local_cols = np.unique(matrix_rows.indices)[:col_count]
            if len(local_cols) == 0:
                local_cols = np.arange(col_count)
        df = pd.DataFrame(
            matrix_rows[:, local_cols].toarray(),
            columns=[feature_names(feature_indices)[i] for i in local_cols],
        )
        df.insert(0, 'label', y[original_rows])
        df.insert(0, 'doc_id', doc_ids[original_rows])
        df.insert(0, 'original_row', original_rows)
        return df, [int(feature_indices[i]) for i in local_cols]

    before_df, before_features = matrix_preview(x, row_indices, row_indices, np.arange(x.shape[1]))
    after_df, after_features = matrix_preview(after_matrix, split_positions, row_indices, selected_feature_indices)
    summary_df = pd.DataFrame([{
        'dataset': dataset_name,
        'leakage': modeling_data['leakage_variant'],
        'representation': representation,
        'split': split,
        'chi2_k': 'all' if k is None else int(k),
        'original_dtm_shape': x.shape,
        'preview_rows': len(row_indices),
        'samples_per_label': samples_per_label,
        'before_preview_shape': before_df.shape,
        'after_chi2_shape': after_matrix.shape,
        'after_preview_shape': after_df.shape,
        'selected_features': int(chi2_res['summary'].iloc[0]['selected_features']),
        'vocab_available': vocab is not None,
    }])
    return {
        'summary_df': summary_df,
        'before_chi2_df': before_df,
        'after_chi2_df': after_df,
        'before_visible_feature_indices': before_features,
        'after_visible_feature_indices': after_features,
        'chi2_summary_df': chi2_res['summary'],
    }


def build_dtm_chi2_diagnostic(
    dataset_names,
    remove_leakage_values,
    representations,
    run_tag=None,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
    split_representation='tfidf',
):
    """Summarize original DTM shapes and saved chi-square tuning coverage."""
    artifacts_root = Path(artifacts_folder)
    if run_tag is None:
        batch_dirs = sorted((artifacts_root / 'step3_batch_runs').glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
        run_tag = batch_dirs[0].name if batch_dirs else None

    dtm_rows = []
    for dataset_name in dataset_names:
        for remove_leakage in remove_leakage_values:
            leakage = 'no_leakage' if remove_leakage else 'with_leakage'
            for representation in representations:
                data = load_modeling_data(
                    dataset_name=dataset_name,
                    remove_leakage=remove_leakage,
                    representation=representation,
                    features_folder=features_folder,
                    artifacts_folder=artifacts_folder,
                )
                x = data['x']
                dtm_rows.append({
                    'dataset': dataset_name,
                    'leakage': leakage,
                    'representation': representation,
                    'docs': x.shape[0],
                    'original_terms': x.shape[1],
                    'nonzero_values': x.nnz,
                    'density_pct': round((x.nnz / (x.shape[0] * x.shape[1]) * 100) if x.shape[0] * x.shape[1] else 0, 4),
                    'dtm_path': data['dtm_path'],
                })
    dtm_shape_df = pd.DataFrame(dtm_rows)

    split_rows = []
    for dataset_name in dataset_names:
        for remove_leakage in remove_leakage_values:
            leakage = 'no_leakage' if remove_leakage else 'with_leakage'
            data = load_modeling_data(
                dataset_name=dataset_name,
                remove_leakage=remove_leakage,
                representation=split_representation,
                features_folder=features_folder,
                artifacts_folder=artifacts_folder,
            )
            split_data = split_modeling_data(
                data['x'], data['y'], train_size=train_size, val_size=val_size,
                test_size=test_size, random_state=random_state, stratify=True
            )
            for _, row in split_data['split_summary'].iterrows():
                split_rows.append({
                    'dataset': dataset_name,
                    'leakage': leakage,
                    'split': row['split'],
                    'rows': int(row['rows']),
                    'features_before_chi2': int(row['features']),
                })
    split_shape_df = pd.DataFrame(split_rows)

    tuning_files = list(artifacts_root.rglob('*chi2*k*tuning_summary.csv'))
    files_with_no_chi2 = []
    for path in tuning_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'K' in df.columns and df['K'].astype(str).str.lower().isin(['all', 'none']).any():
            files_with_no_chi2.append(path)

    run_rows = []
    if run_tag is not None:
        for dataset_name in dataset_names:
            for remove_leakage in remove_leakage_values:
                leakage = 'no_leakage' if remove_leakage else 'with_leakage'
                for representation in representations:
                    run_dir = artifacts_root / dataset_name / leakage / representation / 'step3_runs' / run_tag
                    for result_type, filename in [
                        ('MNIR + SVM chi2 tuning', 'chi2_k_tuning_summary.csv'),
                        ('Direct SVM chi2 tuning', 'direct_svm_chi2_k_tuning_summary.csv'),
                    ]:
                        path = run_dir / filename
                        if not path.exists():
                            continue
                        df = pd.read_csv(path)
                        k_values = df['K'].astype(str).tolist() if 'K' in df.columns else []
                        best = df.iloc[0] if not df.empty else pd.Series(dtype=object)
                        run_rows.append({
                            'run_tag': run_tag,
                            'dataset': dataset_name,
                            'leakage': leakage,
                            'representation': representation,
                            'result_type': result_type,
                            'has_no_chi2_candidate': any(str(k).lower() in ['all', 'none'] for k in k_values),
                            'k_candidates': ', '.join(k_values),
                            'best_k': best.get('K'),
                            'best_selected_features': best.get('Selected Features'),
                            'best_feature_variant': best.get('Feature Variant'),
                            'best_validation_macro_f1': best.get('Validation Macro F1'),
                        })
    run_chi2_check_df = pd.DataFrame(run_rows)
    return {
        'run_tag': run_tag,
        'dtm_shape_df': dtm_shape_df,
        'split_shape_df': split_shape_df,
        'no_chi2_files_df': pd.DataFrame({'path': [str(p) for p in files_with_no_chi2]}),
        'run_chi2_check_df': run_chi2_check_df,
        'summary': {
            'tuning_files_scanned': len(tuning_files),
            'files_with_no_chi2_candidate': len(files_with_no_chi2),
            'selected_run_has_no_chi2_candidate': (
                bool(run_chi2_check_df['has_no_chi2_candidate'].any()) if not run_chi2_check_df.empty else False
            ),
        },
    }


def evaluate_all_models(
    dtm_bow_path=None,
    dtm_tfidf_path=None,
    verdict_results_path=None,
    dataset_name=None,
    remove_leakage=False,
    features_folder='../artifacts/features/dtm',
    artifacts_folder='../artifacts/reports',
    mnir_features_folder='../artifacts/features/mnir',
    chi2_k=None,
    return_full=False,
):
    """
    Backward-compatible Step 3 runner.

    For notebook work, prefer calling the individual functions above so chi-square
    selection and modeling outputs are visible in separate cells.
    """
    data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
        dtm_bow_path=dtm_bow_path,
        dtm_tfidf_path=dtm_tfidf_path,
        verdict_results_path=verdict_results_path,
    )
    print("Loaded BoW:", data['x_bow'].shape)
    print(data['label_counts'].to_string(index=False))

    split = split_modeling_data(data['x_bow'], data['y'], train_size=0.7, val_size=0.1, test_size=0.2)
    print(split['split_summary'].to_string(index=False))

    feature_variant = 'bow'
    chi2_output_dir = os.path.join(
        mnir_features_folder,
        data['dataset_slug'],
        data['leakage_variant'],
        f'bow_chi2_k{chi2_k}' if chi2_k is not None else 'bow',
    )
    chi2_res = apply_chi2_feature_selection(
        split['x_train'],
        split['y_train'],
        split['x_val'],
        split['x_test'],
        k=chi2_k,
        output_dir=chi2_output_dir if chi2_k is not None else None,
    )
    feature_variant = chi2_res['feature_variant']
    print(chi2_res['summary'].to_string(index=False))

    mnir_res = run_mnir_feature_extraction(
        chi2_res['x_train'],
        split['y_train'],
        chi2_res['x_val'],
        chi2_res['x_test'],
        dataset_slug=data['dataset_slug'],
        leakage_variant=data['leakage_variant'],
        feature_variant=feature_variant,
        mnir_features_folder=mnir_features_folder,
    )

    svm_res = evaluate_svm_grid(
        mnir_res['z_train'],
        split['y_train'],
        mnir_res['z_val'],
        split['y_val'],
        mnir_res['z_test'],
        split['y_test'],
    )
    print(svm_res['test_metrics'].to_string(index=False))

    full_result = {
        **data,
        **split,
        'chi2': chi2_res,
        'mnir': mnir_res,
        'svm': svm_res,
    }
    if return_full:
        return full_result

    validation_results = svm_res['validation_results']
    validation_results.attrs['full_result'] = full_result
    return validation_results


if __name__ == '__main__':
    result = evaluate_all_models(return_full=True)
    print(result['svm']['test_metrics'])
