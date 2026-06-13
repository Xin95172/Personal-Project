import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC

from utils.dtm_utils import custom_tokenizer
from utils.pipeline_modeling import (
    apply_chi2_feature_selection,
    load_modeling_data,
    prepare_mnir_no_chi2_input,
    split_modeling_data,
    stratified_sample_modeling_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_text_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _normalize_token(token):
    return unicodedata.normalize("NFKC", str(token)).lower().strip()


def _strip_phrases_series(series, phrases):
    s = series.astype(str).copy()
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        phrase_norm = _normalize_token(phrase)
        if not phrase_norm or " " not in phrase_norm:
            continue
        pat = re.compile(re.escape(phrase_norm))
        s = s.apply(lambda x: pat.sub(" ", _normalize_token(x)))
    return s


def _make_analyzer(delete_words):
    banned = {_normalize_token(t) for t in delete_words if t}

    def analyzer(doc):
        tokens = [_normalize_token(t) for t in custom_tokenizer(doc)]
        return [t for t in tokens if t not in banned]

    return analyzer


def configure_chinese_matplotlib_fonts():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_or_rebuild_vocab(
    dataset_name,
    remove_leakage,
    representation,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
):
    representation = representation.lower().replace("-", "_")
    if representation == "tf_idf":
        representation = "tfidf"

    leakage_variant = "no_leakage" if remove_leakage else "with_leakage"
    dtm_dir = Path(features_folder) / dataset_name / leakage_variant
    vocab_file = {
        "bow": "vocab_BoW.npy",
        "tf": "vocab_TF.npy",
        "tfidf": "vocab_TF_IDF.npy",
    }[representation]
    vocab_path = dtm_dir / vocab_file
    if vocab_path.exists():
        return np.load(vocab_path, allow_pickle=True).astype(str)

    word_seg_path = Path(artifacts_folder) / dataset_name / leakage_variant / "word_seg.xlsx"
    if not word_seg_path.exists():
        raise FileNotFoundError(f"word_seg.xlsx not found: {word_seg_path}")

    lexicon_dir = PROJECT_ROOT / "resources" / "lexicons"
    stop_words = _read_text_lines(lexicon_dir / "stopwords-ch-jiebar-zht.txt")
    delete_words = _read_text_lines(lexicon_dir / "delete_vocab.txt")
    phrases = [
        word
        for words in (stop_words, delete_words)
        for word in words
        if isinstance(word, str) and " " in word
    ]

    df_word_seg = pd.read_excel(word_seg_path).set_index("JID")
    texts = _strip_phrases_series(df_word_seg["Word Segmentation"], phrases)
    analyzer = _make_analyzer(delete_words)

    if representation == "bow":
        vectorizer = CountVectorizer(
            analyzer=analyzer,
            preprocessor=None,
            token_pattern=None,
            min_df=2,
            max_df=0.98,
        )
    elif representation == "tf":
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            norm="l2",
            use_idf=False,
            preprocessor=None,
            token_pattern=None,
            min_df=2,
            max_df=0.98,
        )
    elif representation == "tfidf":
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            norm="l2",
            use_idf=True,
            preprocessor=None,
            token_pattern=None,
            min_df=2,
            max_df=0.98,
        )
    else:
        raise ValueError("representation must be one of: bow, tf, tfidf")

    vectorizer.fit(texts)
    return vectorizer.get_feature_names_out().astype(str)


def rebuild_dtm_from_word_seg(
    dataset_name,
    remove_leakage,
    representation,
    artifacts_folder="../artifacts/reports",
):
    representation = representation.lower().replace("-", "_")
    if representation == "tf_idf":
        representation = "tfidf"

    leakage_variant = "no_leakage" if remove_leakage else "with_leakage"
    word_seg_path = Path(artifacts_folder) / dataset_name / leakage_variant / "word_seg.xlsx"
    verdict_path = Path(artifacts_folder) / dataset_name / leakage_variant / "verdict_results.xlsx"
    if not word_seg_path.exists():
        raise FileNotFoundError(f"word_seg.xlsx not found: {word_seg_path}")
    if not verdict_path.exists():
        raise FileNotFoundError(f"verdict_results.xlsx not found: {verdict_path}")

    lexicon_dir = PROJECT_ROOT / "resources" / "lexicons"
    stop_words = _read_text_lines(lexicon_dir / "stopwords-ch-jiebar-zht.txt")
    delete_words = _read_text_lines(lexicon_dir / "delete_vocab.txt")
    phrases = [
        word
        for words in (stop_words, delete_words)
        for word in words
        if isinstance(word, str) and " " in word
    ]

    df_word_seg = pd.read_excel(word_seg_path).set_index("JID")
    texts = _strip_phrases_series(df_word_seg["Word Segmentation"], phrases)
    analyzer = _make_analyzer(delete_words)

    if representation == "bow":
        vectorizer = CountVectorizer(
            analyzer=analyzer,
            preprocessor=None,
            token_pattern=None,
            min_df=2,
            max_df=0.98,
        )
    elif representation == "tf":
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            norm="l2",
            use_idf=False,
            preprocessor=None,
            token_pattern=None,
            min_df=2,
            max_df=0.98,
        )
    elif representation == "tfidf":
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            norm="l2",
            use_idf=True,
            preprocessor=None,
            token_pattern=None,
            min_df=2,
            max_df=0.98,
        )
    else:
        raise ValueError("representation must be one of: bow, tf, tfidf")

    x = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out().astype(str)
    labels = pd.read_excel(verdict_path, index_col=0).loc[df_word_seg.index].to_numpy().ravel()
    return x, labels, vocab


def latest_direct_svm_config(
    dataset_name,
    remove_leakage,
    representation,
    artifacts_folder="../artifacts/reports",
):
    patch_root = Path(artifacts_folder) / "step4_direct_svm_patch_runs"
    summary_paths = sorted(
        patch_root.glob("*/direct_svm_patch_summary.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summary_paths:
        raise FileNotFoundError(
            "No direct_svm_patch_summary.csv found. Run Step 4.2 first, or set K/C manually."
        )

    leakage_variant = "no_leakage" if remove_leakage else "with_leakage"
    for path in summary_paths:
        df = pd.read_csv(path)
        mask = (
            (df["dataset_slug"].astype(str) == dataset_name)
            & (df["leakage_variant"].astype(str) == leakage_variant)
            & (df["representation"].astype(str) == representation)
        )
        if mask.any():
            row = df.loc[mask].iloc[0].to_dict()
            row["summary_path"] = str(path)
            return row
    raise ValueError(f"No patched Direct SVM config found for {dataset_name}/{leakage_variant}/{representation}.")


def latest_mnir_svm_config(
    dataset_name,
    remove_leakage,
    representation,
    artifacts_folder="../artifacts/reports",
    run_tag=None,
):
    representation = representation.lower().replace("-", "_")
    if representation == "tf_idf":
        representation = "tfidf"

    summary_root = Path(artifacts_folder) / "step3_batch_runs"
    summary_paths = list(summary_root.glob("*/batch_summary.csv"))
    if run_tag is not None:
        summary_paths = [path for path in summary_paths if path.parent.name == run_tag]
    summary_paths = sorted(
        summary_paths,
        key=lambda p: (not p.parent.name.startswith("full"), -p.stat().st_mtime),
    )
    if not summary_paths:
        raise FileNotFoundError("No batch_summary.csv found. Run Step 3 first.")

    leakage_variant = "no_leakage" if remove_leakage else "with_leakage"
    for path in summary_paths:
        df = pd.read_csv(path)
        mask = (
            (df["dataset_slug"].astype(str) == dataset_name)
            & (df["leakage_variant"].astype(str) == leakage_variant)
            & (df["representation"].astype(str) == representation)
        )
        if mask.any():
            row = df.loc[mask].iloc[0].to_dict()
            row["summary_path"] = str(path)
            return row
    raise ValueError(f"No MNIR + SVM config found for {dataset_name}/{leakage_variant}/{representation}.")


def _normalise_model_variant(model_variant):
    model_variant = str(model_variant).lower().replace("-", "_")
    aliases = {
        "chi2": "with_chi2",
        "with_chi_square": "with_chi2",
        "no_chi_square": "no_chi2",
        "none": "no_chi2",
        "all": "no_chi2",
    }
    model_variant = aliases.get(model_variant, model_variant)
    if model_variant not in {"with_chi2", "no_chi2"}:
        raise ValueError("model_variant must be 'with_chi2' or 'no_chi2'.")
    return model_variant


def _cfg_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _cfg_optional_int(value):
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def _resolve_mnir_feature_dir(cfg, representation, mnir_features_folder, model_variant="with_chi2"):
    model_variant = _normalise_model_variant(model_variant)
    if model_variant == "no_chi2":
        if _cfg_bool(cfg.get("mnir_no_chi2_lowfreq_filter_applied")):
            min_train_df = _cfg_optional_int(cfg.get("mnir_no_chi2_min_train_df")) or 1
            feature_variant = f"{representation}_no_chi2_mnir_min_df{min_train_df}"
            feature_limit = _cfg_optional_int(cfg.get("mnir_no_chi2_feature_limit"))
            selected_features = _cfg_optional_int(cfg.get("mnir_no_chi2_selected_features"))
            if feature_limit is not None and selected_features == feature_limit:
                feature_variant += f"_top{feature_limit}"
        else:
            feature_variant = representation
        return Path(mnir_features_folder) / cfg["dataset_slug"] / cfg["leakage_variant"] / feature_variant

    best_k = cfg["best_chi2_k"]
    feature_variant = representation if str(best_k).lower() == "all" else f"{representation}_chi2_k{int(float(best_k))}"
    return Path(mnir_features_folder) / cfg["dataset_slug"] / cfg["leakage_variant"] / feature_variant


def _load_mnir_z_splits(feature_dir):
    feature_dir = Path(feature_dir)
    paths = {
        "z_train": feature_dir / "mnir_z_train.npy",
        "z_val": feature_dir / "mnir_z_val.npy",
        "z_test": feature_dir / "mnir_z_test.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MNIR z feature files:\n" + "\n".join(missing))
    return {name: np.load(path) for name, path in paths.items()}


def _fit_linear_svm_and_rank_features(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    feature_names,
    best_c,
    class_label,
    top_n,
):
    x_final = np.vstack([np.asarray(x_train), np.asarray(x_val)])
    y_final = np.concatenate([y_train, y_val])
    x_test = np.asarray(x_test)

    model = LinearSVC(C=best_c, max_iter=10000)
    model.fit(x_final, y_final)

    classes = list(model.classes_)
    if class_label is None:
        class_index = 0
        class_label = classes[class_index]
    else:
        class_index = classes.index(class_label)

    coef = model.coef_[class_index]
    background_mean = np.asarray(x_final.mean(axis=0)).ravel()
    shap_values = (x_test - background_mean) * coef
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    ranked = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
                "svm_coef": coef,
                "mean_test_value": np.asarray(x_test.mean(axis=0)).ravel(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return ranked, model, class_label, class_index


def mnir_feature_shap_svm(
    dataset_name,
    remove_leakage,
    representation,
    top_n=30,
    class_label=None,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    mnir_features_folder="../artifacts/features/mnir",
    run_tag=None,
    model_variant="with_chi2",
    show_plot=True,
):
    """Compute linear SHAP-style contributions for the MNIR z features used by MNIR + SVM."""
    configure_chinese_matplotlib_fonts()
    model_variant = _normalise_model_variant(model_variant)
    representation = representation.lower().replace("-", "_")
    if representation == "tf_idf":
        representation = "tfidf"

    cfg = latest_mnir_svm_config(
        dataset_name,
        remove_leakage,
        representation,
        artifacts_folder=artifacts_folder,
        run_tag=run_tag,
    )
    best_c = float(cfg["no_chi2_mnir_svm_c"] if model_variant == "no_chi2" else cfg["best_svm_c"])
    max_rows = None if pd.isna(cfg.get("max_rows")) else int(cfg["max_rows"])
    feature_dir = _resolve_mnir_feature_dir(cfg, representation, mnir_features_folder, model_variant=model_variant)
    z = _load_mnir_z_splits(feature_dir)
    feature_names = np.array([f"mnir_z_{idx + 1}" for idx in range(z["z_train"].shape[1])])

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    sampled = stratified_sample_modeling_data(
        modeling_data["x"],
        modeling_data["y"],
        max_rows=max_rows,
        random_state=int(cfg["random_state"]),
    )
    split = split_modeling_data(
        sampled["x"],
        sampled["y"],
        train_size=float(cfg["train_size"]),
        val_size=float(cfg["val_size"]),
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_state"]),
        stratify=True,
    )

    ranked, _, class_label, _ = _fit_linear_svm_and_rank_features(
        z["z_train"],
        split["y_train"],
        z["z_val"],
        split["y_val"],
        z["z_test"],
        feature_names,
        best_c,
        class_label,
        top_n,
    )

    out_dir = Path(artifacts_folder) / dataset_name / cfg["leakage_variant"] / representation / "mnir_feature_shap" / model_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"mnir_svm_z_feature_shap_{model_variant}_{class_label}.csv"
    out_png = out_dir / f"mnir_svm_z_feature_shap_{model_variant}_{class_label}.png"
    ranked.to_csv(out_csv, index=False, encoding="utf-8-sig")

    ax = ranked.sort_values("mean_abs_shap").plot.barh(
        x="feature",
        y="mean_abs_shap",
        figsize=(8, max(5, top_n * 0.25)),
        legend=False,
    )
    ax.set_title(f"MNIR + SVM z-feature SHAP-style contributions ({model_variant}): {class_label}")
    ax.set_xlabel("mean |linear SHAP contribution|")
    ax.set_ylabel("MNIR feature")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()

    print("Config summary:", cfg["summary_path"])
    print("MNIR feature dir:", feature_dir)
    print("Model variant:", model_variant)
    print("Best K:", "all" if model_variant == "no_chi2" else cfg["best_chi2_k"], "Best C:", best_c, "Class:", class_label)
    print("Saved CSV:", out_csv)
    print("Saved plot:", out_png)
    return ranked


def _find_rscript():
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript

    for candidate in [
        Path(sys.prefix) / "Scripts" / "Rscript.exe",
        Path(sys.prefix) / "bin" / "Rscript.exe",
        Path(sys.prefix) / "bin" / "Rscript",
    ]:
        if candidate.exists():
            return str(candidate)

    user_r_root = Path.home() / "R"
    if user_r_root.exists():
        for candidate in sorted(user_r_root.glob("R-*/bin/Rscript.exe"), reverse=True):
            if candidate.exists():
                return str(candidate)

    r_home = os.environ.get("R_HOME")
    if r_home:
        for candidate in [
            Path(r_home) / "bin" / "Rscript.exe",
            Path(r_home) / "bin" / "x64" / "Rscript.exe",
            Path(r_home) / "bin" / "Rscript",
        ]:
            if candidate.exists():
                return str(candidate)
    return None


def r_runtime_diagnostic():
    r_home = os.environ.get("R_HOME")
    candidates = [
        shutil.which("Rscript"),
        str(Path(sys.prefix) / "Scripts" / "Rscript.exe"),
        str(Path(sys.prefix) / "bin" / "Rscript.exe"),
        str(Path(sys.prefix) / "bin" / "Rscript"),
    ]
    user_r_root = Path.home() / "R"
    if user_r_root.exists():
        candidates.extend(str(path) for path in sorted(user_r_root.glob("R-*/bin/Rscript.exe"), reverse=True))
    if r_home:
        candidates.extend(
            [
                str(Path(r_home) / "bin" / "Rscript.exe"),
                str(Path(r_home) / "bin" / "x64" / "Rscript.exe"),
                str(Path(r_home) / "bin" / "Rscript"),
            ]
        )
    return {
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "R_HOME": r_home,
        "R_HOME_exists": bool(r_home and Path(r_home).exists()),
        "Rscript_on_PATH": shutil.which("Rscript"),
        "checked_candidates": [candidate for candidate in candidates if candidate],
        "found_Rscript": _find_rscript(),
    }


def _load_mnir_coefficients_from_rds(model_path):
    rscript = _find_rscript()
    if rscript is None:
        diag = r_runtime_diagnostic()
        raise RuntimeError(
            "Rscript not found, so Python cannot read mnir_mnlm.rds for projected keyword importance. "
            "Install/activate an R runtime with the textir package, or set R_HOME/PATH to a valid R installation. "
            f"Diagnostic: {diag}"
        )

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MNIR model file not found: {model_path}")

    with tempfile.TemporaryDirectory(prefix="mnir_coef_") as tmp:
        tmp_dir = Path(tmp)
        script_path = tmp_dir / "read_mnir_coef.R"
        coef_path = tmp_dir / "mnir_coef.csv"
        script_path.write_text(
            """
args <- commandArgs(trailingOnly = TRUE)
model_path <- args[[1]]
coef_path <- args[[2]]
suppressPackageStartupMessages(library(textir))
model <- readRDS(model_path)
coefs <- as.matrix(coef(model))
write.table(coefs, file = coef_path, sep = ",", row.names = FALSE, col.names = FALSE)
""",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [rscript, str(script_path), str(model_path), str(coef_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to read MNIR coefficients from RDS.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
        coefs = np.loadtxt(coef_path, delimiter=",")
        if coefs.ndim == 1:
            coefs = coefs.reshape(-1, 1)
        return coefs


def mnir_projected_keyword_importance(
    dataset_name,
    remove_leakage,
    representation,
    top_n=30,
    class_label=None,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    mnir_features_folder="../artifacts/features/mnir",
    run_tag=None,
    model_variant="with_chi2",
    show_plot=True,
):
    """Project MNIR + SVM weights back to selected keyword columns using MNIR coefficients."""
    configure_chinese_matplotlib_fonts()
    model_variant = _normalise_model_variant(model_variant)
    representation = representation.lower().replace("-", "_")
    if representation == "tf_idf":
        representation = "tfidf"

    cfg = latest_mnir_svm_config(
        dataset_name,
        remove_leakage,
        representation,
        artifacts_folder=artifacts_folder,
        run_tag=run_tag,
    )
    best_k = None if model_variant == "no_chi2" or str(cfg["best_chi2_k"]).lower() == "all" else int(float(cfg["best_chi2_k"]))
    best_c = float(cfg["no_chi2_mnir_svm_c"] if model_variant == "no_chi2" else cfg["best_svm_c"])
    max_rows = None if pd.isna(cfg.get("max_rows")) else int(cfg["max_rows"])
    feature_dir = _resolve_mnir_feature_dir(cfg, representation, mnir_features_folder, model_variant=model_variant)
    z = _load_mnir_z_splits(feature_dir)

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    vocab = load_or_rebuild_vocab(
        dataset_name,
        remove_leakage,
        representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    if len(vocab) != modeling_data["x"].shape[1]:
        print(
            "Vocabulary length does not match saved DTM columns; "
            "rebuilding DTM and vocabulary from word_seg.xlsx for this explanation only. "
            f"vocab={len(vocab)}, saved_dtm_columns={modeling_data['x'].shape[1]}"
        )
        x_rebuilt, y_rebuilt, vocab = rebuild_dtm_from_word_seg(
            dataset_name,
            remove_leakage,
            representation,
            artifacts_folder=artifacts_folder,
        )
        modeling_data = {
            **modeling_data,
            "x": x_rebuilt,
            "y": y_rebuilt,
        }

    sampled = stratified_sample_modeling_data(
        modeling_data["x"],
        modeling_data["y"],
        max_rows=max_rows,
        random_state=int(cfg["random_state"]),
    )
    split = split_modeling_data(
        sampled["x"],
        sampled["y"],
        train_size=float(cfg["train_size"]),
        val_size=float(cfg["val_size"]),
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_state"]),
        stratify=True,
    )
    chi2_data = apply_chi2_feature_selection(
        split["x_train"],
        split["y_train"],
        split["x_val"],
        split["x_test"],
        k=best_k,
        representation=representation,
    )
    selected_idx = chi2_data["selected_feature_indices"]
    if model_variant == "no_chi2":
        selected_terms = vocab
        mnir_selected_idx_path = feature_dir / "mnir_no_chi2_selected_feature_indices.npy"
        if mnir_selected_idx_path.exists():
            selected_terms = vocab[np.load(mnir_selected_idx_path)]
    else:
        selected_terms = vocab if selected_idx is None else vocab[selected_idx]

    feature_names = np.array([f"mnir_z_{idx + 1}" for idx in range(z["z_train"].shape[1])])
    _, model, class_label, class_index = _fit_linear_svm_and_rank_features(
        z["z_train"],
        split["y_train"],
        z["z_val"],
        split["y_val"],
        z["z_test"],
        feature_names,
        best_c,
        class_label,
        top_n,
    )

    coefs = _load_mnir_coefficients_from_rds(feature_dir / "mnir_mnlm.rds")
    if coefs.shape[0] != len(selected_terms) and coefs.shape[1] == len(selected_terms):
        coefs = coefs.T
    if coefs.shape[0] != len(selected_terms):
        raise ValueError(
            f"MNIR coefficient rows ({coefs.shape[0]}) do not match selected terms ({len(selected_terms)})."
        )
    if coefs.shape[1] != model.coef_.shape[1]:
        raise ValueError(
            f"MNIR coefficient columns ({coefs.shape[1]}) do not match z features ({model.coef_.shape[1]})."
        )

    projected_weight = coefs @ model.coef_[class_index]
    ranked = (
        pd.DataFrame(
            {
                "term": selected_terms,
                "mnir_projected_weight": projected_weight,
                "abs_mnir_projected_weight": np.abs(projected_weight),
            }
        )
        .sort_values("abs_mnir_projected_weight", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    out_dir = Path(artifacts_folder) / dataset_name / cfg["leakage_variant"] / representation / "mnir_projected_keywords" / model_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"mnir_svm_projected_keyword_importance_{model_variant}_{class_label}.csv"
    out_png = out_dir / f"mnir_svm_projected_keyword_importance_{model_variant}_{class_label}.png"
    ranked.to_csv(out_csv, index=False, encoding="utf-8-sig")

    ax = ranked.sort_values("abs_mnir_projected_weight").plot.barh(
        x="term",
        y="abs_mnir_projected_weight",
        figsize=(8, max(5, top_n * 0.25)),
        legend=False,
    )
    ax.set_title(f"MNIR + SVM projected keyword importance ({model_variant}): {class_label}")
    ax.set_xlabel("|projected MNIR keyword weight|")
    ax.set_ylabel("keyword")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()

    print("Config summary:", cfg["summary_path"])
    print("MNIR feature dir:", feature_dir)
    print("Model variant:", model_variant)
    print("Best K:", "all" if model_variant == "no_chi2" else cfg["best_chi2_k"], "Best C:", best_c, "Class:", class_label)
    print("Saved CSV:", out_csv)
    print("Saved plot:", out_png)
    return ranked


def keyword_linear_shap_direct_svm(
    dataset_name,
    remove_leakage,
    representation,
    top_n=30,
    class_label=None,
    features_folder="../artifacts/features/dtm",
    artifacts_folder="../artifacts/reports",
    run_tag=None,
    model_variant="with_chi2",
    show_plot=True,
):
    """Compute keyword-level linear SHAP-style contributions for Direct SVM."""
    configure_chinese_matplotlib_fonts()
    model_variant = _normalise_model_variant(model_variant)
    representation = representation.lower().replace("-", "_")
    if representation == "tf_idf":
        representation = "tfidf"

    if model_variant == "no_chi2":
        cfg = latest_mnir_svm_config(
            dataset_name,
            remove_leakage,
            representation,
            artifacts_folder=artifacts_folder,
            run_tag=run_tag,
        )
        best_k = None
        best_c = float(cfg["no_chi2_svm_c"])
        max_rows = None if pd.isna(cfg.get("max_rows")) else int(cfg["max_rows"])
    else:
        cfg = latest_direct_svm_config(
            dataset_name,
            remove_leakage,
            representation,
            artifacts_folder=artifacts_folder,
        )
        best_k = None if str(cfg["best_direct_chi2_k"]).lower() == "all" else int(float(cfg["best_direct_chi2_k"]))
        best_c = float(cfg["best_direct_svm_c"])
        max_rows = None if pd.isna(cfg.get("direct_svm_patch_max_rows")) else int(cfg["direct_svm_patch_max_rows"])

    modeling_data = load_modeling_data(
        dataset_name=dataset_name,
        remove_leakage=remove_leakage,
        representation=representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    vocab = load_or_rebuild_vocab(
        dataset_name,
        remove_leakage,
        representation,
        features_folder=features_folder,
        artifacts_folder=artifacts_folder,
    )
    if len(vocab) != modeling_data["x"].shape[1]:
        print(
            "Vocabulary length does not match saved DTM columns; "
            "rebuilding DTM and vocabulary from word_seg.xlsx for this explanation only. "
            f"vocab={len(vocab)}, saved_dtm_columns={modeling_data['x'].shape[1]}"
        )
        x_rebuilt, y_rebuilt, vocab = rebuild_dtm_from_word_seg(
            dataset_name,
            remove_leakage,
            representation,
            artifacts_folder=artifacts_folder,
        )
        modeling_data = {
            **modeling_data,
            "x": x_rebuilt,
            "y": y_rebuilt,
        }

    sampled = stratified_sample_modeling_data(
        modeling_data["x"],
        modeling_data["y"],
        max_rows=max_rows,
        random_state=int(cfg["random_state"]),
    )
    split = split_modeling_data(
        sampled["x"],
        sampled["y"],
        train_size=float(cfg["train_size"]),
        val_size=float(cfg["val_size"]),
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_state"]),
        stratify=True,
    )
    chi2_data = apply_chi2_feature_selection(
        split["x_train"],
        split["y_train"],
        split["x_val"],
        split["x_test"],
        k=best_k,
        representation=representation,
    )

    selected_idx = chi2_data["selected_feature_indices"]
    selected_terms = vocab if selected_idx is None else vocab[selected_idx]
    x_final = sp.vstack([chi2_data["x_train"], chi2_data["x_val"]])
    y_final = np.concatenate([split["y_train"], split["y_val"]])

    model = LinearSVC(C=best_c, max_iter=10000)
    model.fit(x_final, y_final)

    classes = list(model.classes_)
    if class_label is None:
        class_index = 0
        class_label = classes[class_index]
    else:
        class_index = classes.index(class_label)

    coef = model.coef_[class_index]
    background_mean = np.asarray(x_final.mean(axis=0)).ravel()
    shap_values = chi2_data["x_test"].multiply(coef).toarray() - (background_mean * coef)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    keyword_shap = (
        pd.DataFrame(
            {
                "term": selected_terms,
                "mean_abs_shap": mean_abs_shap,
                "svm_coef": coef,
                "mean_test_value": np.asarray(chi2_data["x_test"].mean(axis=0)).ravel(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    leakage_variant = "no_leakage" if remove_leakage else "with_leakage"
    out_dir = Path(artifacts_folder) / dataset_name / leakage_variant / representation / "keyword_shap" / model_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"direct_svm_keyword_shap_{model_variant}_{class_label}.csv"
    out_png = out_dir / f"direct_svm_keyword_shap_{model_variant}_{class_label}.png"
    keyword_shap.to_csv(out_csv, index=False, encoding="utf-8-sig")

    ax = keyword_shap.sort_values("mean_abs_shap").plot.barh(
        x="term",
        y="mean_abs_shap",
        figsize=(8, max(5, top_n * 0.25)),
        legend=False,
    )
    ax.set_title(f"Direct SVM keyword SHAP-style contributions ({model_variant}): {class_label}")
    ax.set_xlabel("mean |linear SHAP contribution|")
    ax.set_ylabel("keyword")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()

    print("Config summary:", cfg["summary_path"])
    print("Model variant:", model_variant)
    print("Best K:", "all" if best_k is None else best_k, "Best C:", best_c, "Class:", class_label)
    print("Saved CSV:", out_csv)
    print("Saved plot:", out_png)
    return keyword_shap
