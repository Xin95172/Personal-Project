from __future__ import annotations

from typing import Any, Callable, Protocol, cast

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite

robjects = None
RS4 = Any
importr = None
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "artifacts" / "features"
DEFAULT_TRAIN_FEATURES_PATH = DEFAULT_FEATURES_DIR / "mnir_z_train.npy"
DEFAULT_VAL_FEATURES_PATH = DEFAULT_FEATURES_DIR / "mnir_z_val.npy"
DEFAULT_TEST_FEATURES_PATH = DEFAULT_FEATURES_DIR / "mnir_z_test.npy"
DEFAULT_MODEL_PATH = DEFAULT_FEATURES_DIR / "mnir_mnlm_model.rds"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
DEFAULT_LABEL_ORDER = ["勝訴", "敗訴", "部分勝訴"]

# 啟用 NumPy <-> R 物件自動轉換


def _as_callable(obj: Any, name: str) -> Callable[..., Any]:
    """把動態取得的 rpy2 物件視為可呼叫函式，並在執行期檢查。"""
    fn = cast(Callable[..., Any], obj)
    if not callable(fn):
        raise TypeError(f"{name} 不是可呼叫函式。")
    return fn


def _run_mnir_rscript(
    X_train: sp.spmatrix,
    y_train: np.ndarray,
    X_val: sp.spmatrix | None,
    X_test: sp.spmatrix | None,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript not found on PATH. Activate the tm environment and install r-base.")

    y_arr = _to_1d_labels(y_train, name="y_train")
    if np.issubdtype(y_arr.dtype, np.number):
        y_for_r = y_arr.astype(float)
    else:
        y_for_r = LabelEncoder().fit_transform(y_arr).astype(float)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mnir_r_") as tmp:
        tmp_dir = Path(tmp)
        train_mtx = tmp_dir / "x_train.mtx"
        val_mtx = tmp_dir / "x_val.mtx"
        test_mtx = tmp_dir / "x_test.mtx"
        y_path = tmp_dir / "y_train.csv"
        z_train_path = tmp_dir / "z_train.csv"
        z_val_path = tmp_dir / "z_val.csv"
        z_test_path = tmp_dir / "z_test.csv"
        script_path = tmp_dir / "run_mnir.R"

        mmwrite(train_mtx, sp.csc_matrix(X_train))
        pd.Series(y_for_r).to_csv(y_path, index=False, header=False)
        has_val = X_val is not None
        has_test = X_test is not None
        if has_val:
            mmwrite(val_mtx, sp.csc_matrix(X_val))
        if has_test:
            mmwrite(test_mtx, sp.csc_matrix(X_test))

        script_path.write_text(
            """
args <- commandArgs(trailingOnly = TRUE)
x_train_path <- args[[1]]
y_path <- args[[2]]
model_path <- args[[3]]
z_train_path <- args[[4]]
x_val_path <- args[[5]]
z_val_path <- args[[6]]
x_test_path <- args[[7]]
z_test_path <- args[[8]]

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(textir))

read_counts <- function(path) {
  x <- Matrix::readMM(path)
  as(x, "dgCMatrix")
}

write_z <- function(z, path) {
  z <- as.matrix(z)
  write.table(z, file = path, sep = ",", row.names = FALSE, col.names = FALSE)
}

x_train <- read_counts(x_train_path)
y <- scan(y_path, what = numeric(), sep = ",", quiet = TRUE)
model <- textir::mnlm(cl = NULL, covars = y, counts = x_train)
saveRDS(model, model_path)
write_z(textir::srproj(model, counts = x_train), z_train_path)

if (nzchar(x_val_path)) {
  write_z(textir::srproj(model, counts = read_counts(x_val_path)), z_val_path)
}
if (nzchar(x_test_path)) {
  write_z(textir::srproj(model, counts = read_counts(x_test_path)), z_test_path)
}
""",
            encoding="utf-8",
        )

        cmd = [
            rscript,
            str(script_path),
            str(train_mtx),
            str(y_path),
            str(model_path),
            str(z_train_path),
            str(val_mtx if has_val else ""),
            str(z_val_path),
            str(test_mtx if has_test else ""),
            str(z_test_path),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Rscript MNIR run failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        def _read_z(path: Path, expected_rows: int) -> np.ndarray:
            z = np.loadtxt(path, delimiter=",")
            if z.ndim == 1:
                z = z.reshape(1, -1) if expected_rows == 1 else z.reshape(-1, 1)
            return z

        z_train = _read_z(z_train_path, X_train.shape[0])
        z_val = _read_z(z_val_path, X_val.shape[0]) if has_val else None
        z_test = _read_z(z_test_path, X_test.shape[0]) if has_test else None
        return z_train, z_val, z_test


class RTextirWrapper:
    """Python 對 R 套件 textir 的簡單包裝。"""

    def __init__(self, auto_install: bool = False, cran_mirror: str = "https://cloud.r-project.org"):
        global robjects, RS4, importr
        if robjects is None or importr is None:
            try:
                import rpy2.robjects as _robjects
                from rpy2.robjects import RS4 as _RS4
                from rpy2.robjects.packages import importr as _importr
                robjects = _robjects
                RS4 = _RS4
                importr = _importr
            except Exception as err:
                raise RuntimeError(
                    "rpy2 is not usable in this environment. Use the Rscript MNIR path instead."
                ) from err

        self.model: Any = None
        self.base: Any = None
        self.utils: Any = None
        self.textir: Any = None
        self.matrix_pkg: Any = None

        print("[MNIR] 初始化 R 環境...")
        self.base = importr("base")
        self.utils = importr("utils")
        self.matrix_pkg = importr("Matrix")

        try:
            self.textir = importr("textir")
            print("[MNIR] 已載入 R 套件: textir")
        except Exception as err:
            if not auto_install:
                raise RuntimeError(
                    "找不到 R 套件 textir，且 auto_install=False，無法繼續。"
                ) from err

            print("[MNIR] 找不到 textir，嘗試從 CRAN 安裝...")
            install_packages = _as_callable(
                getattr(self.utils, "install_packages", None),
                "utils::install_packages",
            )
            # Force non-interactive CRAN mirror to avoid console prompts in notebooks.
            install_packages("textir", repos=cran_mirror)
            self.textir = importr("textir")
            print("[MNIR] textir 安裝並載入完成")

    def convert_to_r_format(
        self, X: sp.spmatrix, Y: np.ndarray
    ) -> tuple[RS4, robjects.FloatVector]:
        """
        把 SciPy 稀疏矩陣轉成 R 的稀疏矩陣格式
        並把 Y 轉成 R 的向量格式。
        """
        X_csc = sp.csc_matrix(X)

        data_arr = np.asarray(X_csc.data, dtype=np.float64)
        indices_arr = np.asarray(X_csc.indices, dtype=np.int32)
        indptr_arr = np.asarray(X_csc.indptr, dtype=np.int32)
        shape_arr = np.asarray(X_csc.shape, dtype=np.int32)

        data = robjects.FloatVector(data_arr.tolist())
        indices = robjects.IntVector(indices_arr.tolist())
        indptr = robjects.IntVector(indptr_arr.tolist())
        shape = robjects.IntVector(shape_arr.tolist())

        r_sparse_matrix = self.matrix_pkg.sparseMatrix(
            i=indices,
            p=indptr,
            x=data,
            dims=shape,
            index1=False,
        )

        y_vector = robjects.FloatVector(np.asarray(Y, dtype=float).tolist())
        return r_sparse_matrix, y_vector

    def fit_mnlm(self, X: RS4, Y: robjects.FloatVector) -> Any:
        """訓練 MultiNomial Linear Model 模型。"""
        self.model = self.textir.mnlm(cl=robjects.NULL, covars=Y, counts=X)
        return self.model

    def transform_srproj(self, X: RS4) -> np.ndarray:
        """
        srproj: sufficient reduction projection
        使用訓練好的 mnlm 模型進行降維投影 (Sufficient Reduction)。
        將高維度的詞頻矩陣壓縮成對應目標變數的特徵分數 (Z scores)。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練！請先呼叫 fit_mnlm。")

        z_scores_r = self.textir.srproj(self.model, counts=X)
        return np.asarray(z_scores_r)

    def get_coefs(self) -> np.ndarray:
        """提取 mnlm 模型中各字詞的迴歸係數 (權重)。"""
        if self.model is None:
            raise ValueError("模型尚未訓練！請先呼叫 fit_mnlm。")

        coefs_r = self.base.coef(self.model)
        return np.asarray(coefs_r)

    def save_model(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        if self.model is None:
            raise ValueError("MNIR model is not fitted. Call fit_mnlm() before save_model().")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_rds = _as_callable(getattr(self.base, "saveRDS", None), "base::saveRDS")
        save_rds(self.model, file=str(save_path))
        print(f"[MNIR] saved: {save_path}")
        return save_path

    def load_model(self, path: str | Path = DEFAULT_MODEL_PATH) -> Any:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"找不到 MNIR model 檔案: {load_path}")

        read_rds = _as_callable(getattr(self.base, "readRDS", None), "base::readRDS")
        self.model = read_rds(str(load_path))
        print(f"[MNIR] loaded: {load_path}")
        return self.model


class FitPredictModel(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        ...


def _to_1d_labels(y: np.ndarray, name: str = "Y") -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    elif y_arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={y_arr.shape}")
    return y_arr


def _build_local_model(model_name: str, **kwargs: Any) -> FitPredictModel:
    """用專案內的 algos 模組建立最終預測模型。"""
    name = model_name.lower()
    if name in {"rf", "random_forest"}:
        from .random_forest import RFClassifier

        return cast(FitPredictModel, RFClassifier(**kwargs))
    if name in {"svm"}:
        from .svm import SVMClassifier

        return cast(FitPredictModel, SVMClassifier(**kwargs))
    if name in {"knn"}:
        from .knn import KNNClassifier

        return cast(FitPredictModel, KNNClassifier(**kwargs))
    if name in {"naive_bayes", "nb"}:
        from .naive_bayes import NaiveBayesClassifier

        return cast(FitPredictModel, NaiveBayesClassifier(**kwargs))

    raise ValueError(
        "不支援的 model_name。可用值: rf, random_forest, svm, knn, naive_bayes, nb"
    )


def _ensure_reports_dir(output_dir: str | Path | None = None) -> Path:
    reports_dir = Path(output_dir) if output_dir else DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _ensure_features_dir(output_dir: str | Path | None = None) -> Path:
    features_dir = Path(output_dir) if output_dir else DEFAULT_FEATURES_DIR
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def _normalize_labels(y: np.ndarray, label_order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    y_arr = _to_1d_labels(y, name="y").astype(str)
    labels = label_order or DEFAULT_LABEL_ORDER
    labels = [label for label in labels if label in set(y_arr.tolist())]
    if not labels:
        labels = sorted(pd.Series(y_arr).unique().tolist())
    return y_arr, labels


def build_sample_distribution_report(
    y: np.ndarray,
    output_prefix: str = "mnir_sample_distribution",
    label_order: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    y_arr, labels = _normalize_labels(y, label_order)
    reports_dir = _ensure_reports_dir(output_dir)

    counts = pd.Series(y_arr).value_counts().reindex(labels, fill_value=0)
    total = int(counts.sum())
    distribution_df = pd.DataFrame(
        {
            "count": counts.astype(int),
            "ratio": (counts / total).round(6),
            "ratio_pct": ((counts / total) * 100).round(2),
        }
    )
    distribution_df.loc["總樣本數", "count"] = total
    distribution_df.loc["總樣本數", ["ratio", "ratio_pct"]] = [1.0, 100.0]

    csv_path = reports_dir / f"{_safe_stem(output_prefix)}.csv"
    distribution_df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"[MNIR] saved: {csv_path}")
    return distribution_df


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_prefix: str,
    label_order: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true_arr, labels = _normalize_labels(y_true, label_order)
    y_pred_arr = _to_1d_labels(y_pred, name="y_pred").astype(str)
    reports_dir = _ensure_reports_dir(output_dir)
    stem = _safe_stem(output_prefix)

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    main_df = pd.DataFrame(
        {
            "accuracy_mean": [accuracy],
            "macro_f1_mean": [macro_f1],
            "weighted_f1_mean": [weighted_f1],
            "macro_f1_std": [np.nan],
        },
        index=["single_run"],
    )

    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        zero_division=0,
    )
    class_report_df = pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_values,
            "support": support.astype(int),
        }
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    main_path = reports_dir / f"{stem}_main_summary.csv"
    class_path = reports_dir / f"{stem}_classification_report.csv"
    cm_path = reports_dir / f"{stem}_confusion_matrix.csv"
    fig_path = reports_dir / f"{stem}_confusion_matrix.png"

    main_df.to_csv(main_path, encoding="utf-8-sig")
    class_report_df.to_csv(class_path, index=False, encoding="utf-8-sig")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[MNIR] saved: {main_path}")
    print(f"[MNIR] saved: {class_path}")
    print(f"[MNIR] saved: {cm_path}")
    print(f"[MNIR] saved: {fig_path}")
    return main_df, class_report_df, cm_df


def extract_keyword_loadings(
    extractor: "MNIRFeatureExtractor",
    vocabulary: list[str] | np.ndarray,
    output_prefix: str = "mnir_keyword_loadings",
    label_order: list[str] | None = None,
    top_k: int = 20,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    reports_dir = _ensure_reports_dir(output_dir)
    stem = _safe_stem(output_prefix)
    labels = label_order or DEFAULT_LABEL_ORDER
    vocab = list(vocabulary)

    try:
        coefs = np.asarray(extractor.r_wrapper.get_coefs())
    except Exception:
        note_df = pd.DataFrame(
            {"note": ["MNIR model coefficients unavailable; skipped keyword loading export."]}
        )
        note_path = reports_dir / f"{stem}_keywords_skipped.csv"
        note_df.to_csv(note_path, index=False, encoding="utf-8-sig")
        print(f"[MNIR] saved: {note_path}")
        return note_df

    if coefs.ndim == 1:
        coefs = coefs.reshape(-1, 1)
    if coefs.shape[0] != len(vocab):
        note_df = pd.DataFrame(
            {
                "note": [
                    f"Coefficient rows ({coefs.shape[0]}) do not match vocabulary size ({len(vocab)}); skipped."
                ]
            }
        )
        note_path = reports_dir / f"{stem}_keywords_skipped.csv"
        note_df.to_csv(note_path, index=False, encoding="utf-8-sig")
        print(f"[MNIR] saved: {note_path}")
        return note_df

    usable_labels = labels[: coefs.shape[1]]
    rows: list[dict[str, Any]] = []
    for col_idx, label in enumerate(usable_labels):
        coef_col = coefs[:, col_idx]
        abs_idx = np.argsort(np.abs(coef_col))[::-1][:top_k]
        for rank, vocab_idx in enumerate(abs_idx, start=1):
            rows.append(
                {
                    "class": label,
                    "rank": rank,
                    "term": vocab[vocab_idx],
                    "loading": float(coef_col[vocab_idx]),
                    "abs_loading": float(abs(coef_col[vocab_idx])),
                }
            )

    keywords_df = pd.DataFrame(rows)
    csv_path = reports_dir / f"{stem}.csv"
    keywords_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[MNIR] saved: {csv_path}")
    return keywords_df


def analyze_gamma_lasso_sparsity(
    extractor: "MNIRFeatureExtractor",
    vocabulary: list[str] | np.ndarray,
    output_prefix: str = "mnir_sparsity",
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    reports_dir = _ensure_reports_dir(output_dir)
    stem = _safe_stem(output_prefix)

    try:
        coefs = np.asarray(extractor.r_wrapper.get_coefs())
    except Exception:
        note_df = pd.DataFrame({"note": ["MNIR model coefficients unavailable; skipped sparsity analysis."]})
        note_path = reports_dir / f"{stem}_skipped.csv"
        note_df.to_csv(note_path, index=False, encoding="utf-8-sig")
        print(f"[MNIR] saved: {note_path}")
        return note_df

    coef_arr = np.asarray(coefs)
    nonzero_mask = coef_arr != 0
    nonzero_count = int(nonzero_mask.sum())
    total_vocab = int(len(vocabulary))
    zero_count = int(total_vocab - np.any(nonzero_mask, axis=1).sum()) if coef_arr.ndim > 1 else int(total_vocab - nonzero_count)
    zero_ratio = zero_count / total_vocab if total_vocab else np.nan

    sparsity_df = pd.DataFrame(
        {
            "raw_vocab_size": [total_vocab],
            "nonzero_coefficients": [nonzero_count],
            "zeroed_terms": [zero_count],
            "zeroed_ratio": [zero_ratio],
        }
    )
    csv_path = reports_dir / f"{stem}.csv"
    sparsity_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[MNIR] saved: {csv_path}")
    return sparsity_df


def plot_sr_scores(
    z_scores: np.ndarray,
    y: np.ndarray,
    output_prefix: str = "mnir_sr_scores",
    label_order: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    reports_dir = _ensure_reports_dir(output_dir)
    stem = _safe_stem(output_prefix)
    y_arr, labels = _normalize_labels(y, label_order)
    z = np.asarray(z_scores)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    fig_path = reports_dir / f"{stem}.png"
    color_map = {
        labels[idx]: color
        for idx, color in enumerate(["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"])
    }

    if z.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in labels:
            mask = y_arr == label
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                label=label,
                alpha=0.6,
                s=18,
                color=color_map.get(label, "#333333"),
            )
        ax.set_xlabel("SR Score 1")
        ax.set_ylabel("SR Score 2")
        ax.set_title("MNIR SR Score Scatter")
        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in labels:
            mask = y_arr == label
            ax.hist(
                z[mask, 0],
                bins=30,
                alpha=0.5,
                label=label,
                color=color_map.get(label, "#333333"),
            )
        ax.set_xlabel("SR Score 1")
        ax.set_ylabel("Count")
        ax.set_title("MNIR SR Score Distribution")
        ax.legend()

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[MNIR] saved: {fig_path}")
    return fig_path


def _write_note_csv(
    note: str,
    output_prefix: str,
    suffix: str = "note",
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    reports_dir = _ensure_reports_dir(output_dir)
    csv_path = reports_dir / f"{_safe_stem(output_prefix)}_{suffix}.csv"
    note_df = pd.DataFrame({"note": [note]})
    note_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[MNIR] saved: {csv_path}")
    return note_df


def save_feature_matrix(features: np.ndarray, path: str | Path) -> Path:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, np.asarray(features))
    print(f"[MNIR] saved: {save_path}")
    return save_path


def load_feature_matrix(path: str | Path) -> np.ndarray:
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"找不到 MNIR 特徵檔案: {load_path}")

    features = np.load(load_path)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    return features


def save_feature_splits(
    z_train: np.ndarray,
    z_val: np.ndarray | None = None,
    z_test: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    train_name: str = "mnir_z_train.npy",
    val_name: str = "mnir_z_val.npy",
    test_name: str = "mnir_z_test.npy",
) -> dict[str, Path]:
    features_dir = _ensure_features_dir(output_dir)
    saved_paths = {
        "train": save_feature_matrix(z_train, features_dir / train_name),
    }
    if z_val is not None:
        saved_paths["val"] = save_feature_matrix(z_val, features_dir / val_name)
    if z_test is not None:
        saved_paths["test"] = save_feature_matrix(z_test, features_dir / test_name)
    return saved_paths


def fit_and_save_feature_splits(
    X_train: sp.spmatrix,
    y_train: np.ndarray,
    X_val: sp.spmatrix | None = None,
    X_test: sp.spmatrix | None = None,
    output_dir: str | Path | None = None,
    train_name: str = "mnir_z_train.npy",
    val_name: str = "mnir_z_val.npy",
    test_name: str = "mnir_z_test.npy",
    model_name: str = "mnir_mnlm_model.rds",
    auto_install: bool = False,
) -> dict[str, Any]:
    features_dir = _ensure_features_dir(output_dir)
    model_path = features_dir / model_name

    if robjects is None or importr is None:
        z_train, z_val, z_test = _run_mnir_rscript(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            X_test=X_test,
            model_path=model_path,
        )
        extractor = None
    else:
        extractor = MNIRFeatureExtractor(RTextirWrapper(auto_install=auto_install))
        extractor.fit(
            X_train,
            y_train,
            load_cached=False,
            cache_path=features_dir / train_name,
            model_path=model_path,
        )

        z_train = extractor.get_train_features()
        z_val = extractor.transform(X_val) if X_val is not None else None
        z_test = extractor.transform(X_test) if X_test is not None else None

    saved_paths = save_feature_splits(
        z_train=z_train,
        z_val=z_val,
        z_test=z_test,
        output_dir=output_dir,
        train_name=train_name,
        val_name=val_name,
        test_name=test_name,
    )
    return {
        "extractor": extractor,
        "z_train": z_train,
        "z_val": z_val,
        "z_test": z_test,
        "paths": saved_paths,
        "model_path": model_path,
    }


def load_or_fit_feature_splits(
    X_train: sp.spmatrix,
    y_train: np.ndarray,
    X_val: sp.spmatrix | None = None,
    X_test: sp.spmatrix | None = None,
    output_dir: str | Path | None = None,
    train_name: str = "mnir_z_train.npy",
    val_name: str = "mnir_z_val.npy",
    test_name: str = "mnir_z_test.npy",
    model_name: str = "mnir_mnlm_model.rds",
    auto_install: bool = False,
    require_model: bool = True,
) -> dict[str, Any]:
    features_dir = _ensure_features_dir(output_dir)
    train_path = features_dir / train_name
    val_path = features_dir / val_name
    test_path = features_dir / test_name
    model_path = features_dir / model_name

    required_paths = [train_path]
    if X_val is not None:
        required_paths.append(val_path)
    if X_test is not None:
        required_paths.append(test_path)
    if require_model:
        required_paths.append(model_path)

    cache_ready = all(path.exists() for path in required_paths)
    if cache_ready:
        z_train = load_feature_matrix(train_path)
        z_val = load_feature_matrix(val_path) if X_val is not None else None
        z_test = load_feature_matrix(test_path) if X_test is not None else None
        cache_shape_matches = (
            z_train.shape[0] == X_train.shape[0]
            and (X_val is None or z_val.shape[0] == X_val.shape[0])
            and (X_test is None or z_test.shape[0] == X_test.shape[0])
        )
        if not cache_shape_matches:
            print(
                "[MNIR] cached feature shape does not match current split; "
                "recomputing MNIR features."
            )
        else:
            result: dict[str, Any] = {
                "z_train": z_train,
                "z_val": z_val,
                "z_test": z_test,
                "paths": {
                    "train": train_path,
                    **({"val": val_path} if X_val is not None else {}),
                    **({"test": test_path} if X_test is not None else {}),
                },
                "model_path": model_path,
                "loaded_from_cache": True,
            }
            if require_model and robjects is not None and importr is not None:
                extractor = MNIRFeatureExtractor(RTextirWrapper(auto_install=auto_install))
                extractor.load_model(model_path)
                extractor.load_train_features(train_path)
                result["extractor"] = extractor
            return result

    result = fit_and_save_feature_splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        X_test=X_test,
        output_dir=output_dir,
        train_name=train_name,
        val_name=val_name,
        test_name=test_name,
        model_name=model_name,
        auto_install=auto_install,
    )
    result["loaded_from_cache"] = False
    return result


def _fit_local_model(model_name: str, model_kwargs: dict[str, Any] | None, x_train: np.ndarray, y_train: np.ndarray) -> FitPredictModel:
    model = _build_local_model(model_name, **(model_kwargs or {}))
    try:
        model.fit(x_train=x_train, y_train=y_train)
    except TypeError:
        model.fit(x_train, y_train)
    return model


def export_single_split_outputs(
    X_train: sp.spmatrix,
    y_train: np.ndarray,
    X_eval: sp.spmatrix,
    y_eval: np.ndarray,
    vocabulary: list[str] | np.ndarray | None = None,
    model_name: str = "rf",
    model_kwargs: dict[str, Any] | None = None,
    output_prefix: str = "mnir_single",
    label_order: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Fit one MNIR + downstream model pipeline and export paper-ready outputs.

    Outputs:
    - sample distribution table
    - prediction summary table
    - per-class classification report
    - confusion matrix csv + png
    - prediction detail csv
    - keyword loadings / sparsity (if vocabulary is provided and coefficients are available)
    - SR score plot for evaluation split
    """
    reports_dir = _ensure_reports_dir(output_dir)
    stem = _safe_stem(output_prefix)
    y_train_arr, labels = _normalize_labels(y_train, label_order)
    y_eval_arr = _to_1d_labels(y_eval, name="y_eval").astype(str)

    predictor = MNIRPredictor(
        r_wrapper=RTextirWrapper(auto_install=False),
        final_model=model_name,
        model_kwargs=model_kwargs,
    )
    predictor.fit(X_train, y_train_arr)

    z_train = predictor.get_train_features()
    z_eval = predictor.transform_features(X_eval)
    y_pred = predictor.predict(X_eval)

    sample_df = build_sample_distribution_report(
        np.concatenate([y_train_arr, y_eval_arr]),
        output_prefix=f"{stem}_sample_distribution",
        label_order=labels,
        output_dir=reports_dir,
    )
    main_df, class_report_df, cm_df = summarize_predictions(
        y_eval_arr,
        y_pred,
        output_prefix=f"{stem}_eval",
        label_order=labels,
        output_dir=reports_dir,
    )

    pred_detail_df = pd.DataFrame(
        {
            "set": "eval",
            "y_true": y_eval_arr,
            "y_pred": _to_1d_labels(y_pred, name="y_pred").astype(str),
        }
    )
    pred_detail_path = reports_dir / f"{stem}_prediction_detail.csv"
    pred_detail_df.to_csv(pred_detail_path, index=False, encoding="utf-8-sig")
    print(f"[MNIR] saved: {pred_detail_path}")

    z_train_path = reports_dir / f"{stem}_z_train.csv"
    z_eval_path = reports_dir / f"{stem}_z_eval.csv"
    pd.DataFrame(z_train).to_csv(z_train_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(z_eval).to_csv(z_eval_path, index=False, encoding="utf-8-sig")
    print(f"[MNIR] saved: {z_train_path}")
    print(f"[MNIR] saved: {z_eval_path}")

    sr_plot_path = plot_sr_scores(
        z_eval,
        y_eval_arr,
        output_prefix=f"{stem}_sr_scores",
        label_order=labels,
        output_dir=reports_dir,
    )

    if vocabulary is not None:
        keyword_df = extract_keyword_loadings(
            predictor.feature_extractor,
            vocabulary=vocabulary,
            output_prefix=f"{stem}_keyword_loadings",
            label_order=labels,
            output_dir=reports_dir,
        )
        sparsity_df = analyze_gamma_lasso_sparsity(
            predictor.feature_extractor,
            vocabulary=vocabulary,
            output_prefix=f"{stem}_sparsity",
            output_dir=reports_dir,
        )
    else:
        keyword_df = _write_note_csv(
            "Vocabulary not provided; skipped keyword loading export.",
            output_prefix=f"{stem}_keyword_loadings",
            output_dir=reports_dir,
        )
        sparsity_df = _write_note_csv(
            "Vocabulary not provided; skipped sparsity analysis.",
            output_prefix=f"{stem}_sparsity",
            output_dir=reports_dir,
        )

    return {
        "predictor": predictor,
        "sample_distribution_df": sample_df,
        "main_summary_df": main_df,
        "class_report_df": class_report_df,
        "confusion_matrix_df": cm_df,
        "prediction_detail_df": pred_detail_df,
        "keyword_df": keyword_df,
        "sparsity_df": sparsity_df,
        "sr_plot_path": sr_plot_path,
        "z_train": z_train,
        "z_eval": z_eval,
    }


# --- Leakage / 單一折輔助函數 ---
def remove_leakage_phrases(
    text: str,
    exact_patterns: list[str] | None = None,
    soft_patterns: list[str] | None = None,
    remove_soft: bool = False,
) -> str:
    """
    移除會洩漏判決結果的字詞或句型。
    支援透過確切字詞 (exact) 或正則表達式 (soft) 取代。
    """
    if not isinstance(text, str):
        return ""
    if exact_patterns:
        for p in exact_patterns:
            text = text.replace(p, "")
    if remove_soft and soft_patterns:
        for p in soft_patterns:
            text = re.sub(p, "", text)
    return text.strip()

def prepare_model_text(
    text_series: pd.Series,
    exact_patterns: list[str] | None = None,
    soft_patterns: list[str] | None = None,
    remove_soft: bool = False,
) -> pd.Series:
    """
    Notebook 友善的前處理函式，將 leakage 清理接到資料流程。
    自動補齊空值轉字串，並套用 remove_leakage_phrases。
    """
    s = text_series.fillna("").astype(str)
    return s.apply(lambda x: remove_leakage_phrases(x, exact_patterns, soft_patterns, remove_soft))

def run_single_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: sp.spmatrix,
    y: np.ndarray,
    model_name: str = "rf",
    model_kwargs: dict[str, Any] | None = None,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    執行單一 Fold 的完整流程 (MNIR 特徵抽取與下游分類)。
    回傳 y_true, y_pred, 各類指標以及降維特徵。
    """
    y_arr, labels = _normalize_labels(y, label_order)
    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = y_arr[train_idx], y_arr[val_idx]

    predictor = MNIRPredictor(
        r_wrapper=RTextirWrapper(auto_install=False),
        final_model=model_name,
        model_kwargs=model_kwargs,
    )
    predictor.fit(x_train, y_train)

    z_train = predictor.get_train_features()
    z_val = predictor.transform_features(x_val)
    y_pred = _to_1d_labels(predictor.predict(x_val), name="y_pred").astype(str)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    cls_report = classification_report(y_val, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    return {
        "train_idx": train_idx,
        "valid_idx": val_idx,
        "y_true": y_val,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": cls_report,
        "confusion_matrix": cm,
        "Z_train": z_train,
        "Z_valid": z_val,
        "extractor": predictor.feature_extractor,
        "model": predictor.final_model,
        "predictor": predictor
    }

def export_repeated_cv_outputs(
    X: sp.spmatrix,
    y: np.ndarray,
    vocabulary: list[str] | np.ndarray | None = None,
    model_name: str = "rf",
    model_kwargs: dict[str, Any] | None = None,
    output_prefix: str = "mnir_cv",
    label_order: list[str] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    output_dir: str | Path | None = None,
    refit_full_for_interpretability: bool = True,
) -> dict[str, Any]:
    """
    Run repeated stratified k-fold CV and export paper-ready outputs.

    Saved outputs include:
    - sample distribution
    - per-fold metrics
    - summary metrics (mean accuracy / mean macro-F1 / mean weighted-F1 / macro-F1 std)
    - pooled classification report
    - pooled confusion matrix csv + png
    - all validation predictions across repeats/folds
    - optional full-data refit keyword loadings / sparsity / SR score plot
    """
    reports_dir = _ensure_reports_dir(output_dir)
    stem = _safe_stem(output_prefix)
    y_arr, labels = _normalize_labels(y, label_order)

    sample_df = build_sample_distribution_report(
        y_arr,
        output_prefix=f"{stem}_sample_distribution",
        label_order=labels,
        output_dir=reports_dir,
    )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for split_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_arr), start=1):
        repeat_id = ((split_idx - 1) // n_splits) + 1
        fold_id = ((split_idx - 1) % n_splits) + 1

        fold_res = run_single_fold(
            train_idx, val_idx, X, y_arr,
            model_name=model_name, model_kwargs=model_kwargs, label_order=labels
        )

        y_val = fold_res["y_true"]
        y_pred = fold_res["y_pred"]

        fold_rows.append(
            {
                "repeat": repeat_id,
                "fold": fold_id,
                "accuracy": fold_res["accuracy"],
                "macro_f1": fold_res["macro_f1"],
                "weighted_f1": fold_res["weighted_f1"],
            }
        )

        for row_idx, true_label, pred_label in zip(val_idx.tolist(), y_val.tolist(), y_pred.tolist()):
            prediction_rows.append(
                {
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "row_index": row_idx,
                    "y_true": true_label,
                    "y_pred": pred_label,
                }
            )

        print(
            f"[MNIR-CV] repeat={repeat_id:02d} fold={fold_id:02d} "
            f"macro_f1={fold_rows[-1]['macro_f1']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    prediction_df = pd.DataFrame(prediction_rows)

    fold_path = reports_dir / f"{stem}_fold_metrics.csv"
    prediction_path = reports_dir / f"{stem}_prediction_detail.csv"
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    print(f"[MNIR] saved: {fold_path}")
    print(f"[MNIR] saved: {prediction_path}")

    main_summary_df = pd.DataFrame(
        {
            "accuracy_mean": [fold_df["accuracy"].mean()],
            "macro_f1_mean": [fold_df["macro_f1"].mean()],
            "weighted_f1_mean": [fold_df["weighted_f1"].mean()],
            "macro_f1_std": [fold_df["macro_f1"].std(ddof=0)],
        },
        index=["mnir"],
    )
    main_summary_path = reports_dir / f"{stem}_main_summary.csv"
    main_summary_df.to_csv(main_summary_path, encoding="utf-8-sig")
    print(f"[MNIR] saved: {main_summary_path}")

    pooled_main_df, class_report_df, cm_df = summarize_predictions(
        prediction_df["y_true"].to_numpy(),
        prediction_df["y_pred"].to_numpy(),
        output_prefix=f"{stem}_pooled",
        label_order=labels,
        output_dir=reports_dir,
    )
    pooled_main_df.iloc[0, pooled_main_df.columns.get_loc("macro_f1_std")] = float(fold_df["macro_f1"].std(ddof=0))
    pooled_main_df.to_csv(reports_dir / f"{stem}_pooled_main_summary.csv", encoding="utf-8-sig")

    keyword_df: pd.DataFrame
    sparsity_df: pd.DataFrame
    sr_plot_path: Path | None
    full_fit_predictor: MNIRPredictor | None = None

    if refit_full_for_interpretability:
        full_fit_predictor = MNIRPredictor(
            r_wrapper=RTextirWrapper(auto_install=False),
            final_model=model_name,
            model_kwargs=model_kwargs,
        )
        full_fit_predictor.fit(X, y_arr)
        z_full = full_fit_predictor.get_train_features()
        z_full_path = reports_dir / f"{stem}_z_full.csv"
        pd.DataFrame(z_full).to_csv(z_full_path, index=False, encoding="utf-8-sig")
        print(f"[MNIR] saved: {z_full_path}")

        sr_plot_path = plot_sr_scores(
            z_full,
            y_arr,
            output_prefix=f"{stem}_sr_scores",
            label_order=labels,
            output_dir=reports_dir,
        )

        if vocabulary is not None:
            keyword_df = extract_keyword_loadings(
                full_fit_predictor.feature_extractor,
                vocabulary=vocabulary,
                output_prefix=f"{stem}_keyword_loadings",
                label_order=labels,
                output_dir=reports_dir,
            )
            sparsity_df = analyze_gamma_lasso_sparsity(
                full_fit_predictor.feature_extractor,
                vocabulary=vocabulary,
                output_prefix=f"{stem}_sparsity",
                output_dir=reports_dir,
            )
        else:
            keyword_df = _write_note_csv(
                "Vocabulary not provided; skipped keyword loading export.",
                output_prefix=f"{stem}_keyword_loadings",
                output_dir=reports_dir,
            )
            sparsity_df = _write_note_csv(
                "Vocabulary not provided; skipped sparsity analysis.",
                output_prefix=f"{stem}_sparsity",
                output_dir=reports_dir,
            )
    else:
        sr_plot_path = None
        keyword_df = _write_note_csv(
            "Skipped keyword loading export because refit_full_for_interpretability=False.",
            output_prefix=f"{stem}_keyword_loadings",
            output_dir=reports_dir,
        )
        sparsity_df = _write_note_csv(
            "Skipped sparsity analysis because refit_full_for_interpretability=False.",
            output_prefix=f"{stem}_sparsity",
            output_dir=reports_dir,
        )
        _write_note_csv(
            "Skipped SR score plot because refit_full_for_interpretability=False.",
            output_prefix=f"{stem}_sr_scores",
            output_dir=reports_dir,
        )

    return {
        "sample_df": sample_df,
        "fold_df": fold_df,
        "prediction_df": prediction_df,
        "main_summary_df": main_summary_df,
        "pooled_main_df": pooled_main_df,
        "class_report_df": class_report_df,
        "cm_df": cm_df,
        "keyword_df": keyword_df,
        "sparsity_df": sparsity_df,
        "sr_plot_path": sr_plot_path,
        "full_fit_predictor": full_fit_predictor,
    }


class MNIRFeatureExtractor:
    """抽取 MNIR 特徵，可重複用於多個下游模型。"""

    def __init__(self, r_wrapper: RTextirWrapper):
        self.r_wrapper = r_wrapper
        self.is_fitted = False
        self._z_train_cache: np.ndarray | None = None

    def fit(
        self,
        X: sp.spmatrix,
        Y: np.ndarray,
        load_cached: bool = False,
        cache_path: str | Path = DEFAULT_TRAIN_FEATURES_PATH,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> "MNIRFeatureExtractor":
        cache_file = Path(cache_path)
        model_file = Path(model_path)
        if load_cached and cache_file.exists() and model_file.exists():
            self.load_train_features(cache_file)
            self.load_model(model_file)
            return self

        y_arr = _to_1d_labels(Y, name="Y")

        # R textir::mnlm requires numeric covariates.
        if np.issubdtype(y_arr.dtype, np.number):
            y_for_r = y_arr.astype(float)
        else:
            y_for_r = LabelEncoder().fit_transform(y_arr).astype(float)

        r_X, r_Y = self.r_wrapper.convert_to_r_format(X, y_for_r)
        self.r_wrapper.fit_mnlm(r_X, r_Y)

        z_train = self.r_wrapper.transform_srproj(r_X)
        if z_train.ndim == 1:
            z_train = z_train.reshape(-1, 1)

        self._z_train_cache = z_train
        self.is_fitted = True
        self.save_train_features(cache_file)
        self.save_model(model_file)
        return self

    def transform(self, X: sp.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MNIRFeatureExtractor 尚未訓練！請先呼叫 fit。")

        dummy_y = np.zeros(X.shape[0], dtype=float)
        r_X, _ = self.r_wrapper.convert_to_r_format(X, dummy_y)
        z = self.r_wrapper.transform_srproj(r_X)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        return z

    def get_train_features(self) -> np.ndarray:
        if self._z_train_cache is None:
            raise RuntimeError("尚未有訓練特徵快取，請先呼叫 fit。")
        return self._z_train_cache

    def save_train_features(self, path: str | Path = DEFAULT_TRAIN_FEATURES_PATH) -> Path:
        if self._z_train_cache is None:
            raise RuntimeError("尚未有訓練特徵快取，請先呼叫 fit。")

        return save_feature_matrix(self._z_train_cache, path)

    def load_train_features(self, path: str | Path = DEFAULT_TRAIN_FEATURES_PATH) -> np.ndarray:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"找不到 MNIR 訓練特徵檔案: {load_path}")

        z_train = np.load(load_path)
        if z_train.ndim == 1:
            z_train = z_train.reshape(-1, 1)

        self._z_train_cache = z_train
        self.is_fitted = False
        return z_train

    def save_model(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        return self.r_wrapper.save_model(path)

    def load_model(self, path: str | Path = DEFAULT_MODEL_PATH) -> Any:
        model = self.r_wrapper.load_model(path)
        self.is_fitted = True
        return model

class MNIRPredictor:
    """全功能 MNIR 預測器，後端使用你在 algos 寫好的模型。"""

    def __init__(
        self,
        r_wrapper: RTextirWrapper,
        final_model: FitPredictModel | str,
        model_kwargs: dict[str, Any] | None = None,
        feature_extractor: MNIRFeatureExtractor | None = None,
    ):
        """
        :param r_wrapper: RTextirWrapper 實例
        :param final_model: 你在 algos 寫好的模型實例，或模型名稱字串（例如 "rf"）
        :param model_kwargs: 當 final_model 是字串時，傳給模型建構子的參數
        :param feature_extractor: 可重用的 MNIR 特徵抽取器；可跨多個模型共用
        """
        self.r_wrapper = r_wrapper
        self.feature_extractor = feature_extractor or MNIRFeatureExtractor(r_wrapper)
        if isinstance(final_model, str):
            self.final_model = _build_local_model(final_model, **(model_kwargs or {}))
        else:
            self.final_model = final_model
        self.is_fitted = False

    def fit(self, X: sp.spmatrix, Y: np.ndarray, reuse_mnir: bool = True) -> "MNIRPredictor":
        print(f"[MNIR] 開始訓練流程，後端模型: {type(self.final_model).__name__}")

        y_arr = _to_1d_labels(Y, name="Y")

        if reuse_mnir and self.feature_extractor.is_fitted:
            z_features = self.feature_extractor.transform(X)
        else:
            self.feature_extractor.fit(X, y_arr)
            z_features = self.feature_extractor.get_train_features()

        try:
            self.final_model.fit(x_train=z_features, y_train=y_arr)
        except TypeError:
            self.final_model.fit(z_features, y_arr)

        self.is_fitted = True
        print("[MNIR] 訓練完成。")
        return self

    def get_train_features(self) -> np.ndarray:
        """Expose cached MNIR training features for inspection or downstream analysis."""
        return self.feature_extractor.get_train_features()

    def transform_features(self, X: sp.spmatrix) -> np.ndarray:
        """Transform raw input into MNIR feature space without running the final model."""
        return self.feature_extractor.transform(X)

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型尚未訓練！")

        z_test = self.feature_extractor.transform(X)
        return self.final_model.predict(z_test)


import warnings

def build_vectorizer(config: dict[str, Any]) -> Any:
    """依據設定建立 Vectorizer (TFIDF 或 BoW)。"""
    v_type = config.get("vectorizer_type", "tfidf").lower()
    max_features = config.get("max_features", 10000)
    ngram_range = config.get("ngram_range", (1, 1))
    min_df = config.get("min_df", 1)
    max_df = config.get("max_df", 1.0)

    if v_type == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
    elif v_type in ["count", "bow"]:
        return CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
    else:
        raise ValueError("不支援的 vectorizer_type，請使用 'tfidf' 或 'count'。")


def compare_representations_cv(
    X_bow: sp.spmatrix,
    X_tfidf: sp.spmatrix,
    y: np.ndarray,
    model_name: str = "rf",
    model_kwargs: dict[str, Any] | None = None,
    label_order: list[str] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    [LEGACY] Compare BoW and TF-IDF under repeated stratified k-fold CV with MNIR features.
    This function is maintained for backward compatibility but is no longer the main notebook workflow.
    Please use `export_repeated_cv_outputs` and isolated model flows.
    """
    warnings.warn(
        "compare_representations_cv is a legacy function and is no longer the main workflow. "
        "Please rely on export_repeated_cv_outputs with a specific model setting.",
        DeprecationWarning,
        stacklevel=2,
    )
    y_arr = _to_1d_labels(y, name="y")
    labels = label_order or sorted(pd.Series(y_arr).astype(str).unique().tolist())
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X_bow, y_arr), start=1):
        x_train_bow, x_val_bow = X_bow[train_idx], X_bow[val_idx]
        x_train_tfidf, x_val_tfidf = X_tfidf[train_idx], X_tfidf[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        bow_wrapper = RTextirWrapper(auto_install=False)
        bow_extractor = MNIRFeatureExtractor(bow_wrapper)
        bow_extractor.fit(x_train_bow, y_train, load_cached=False)
        z_train_bow = bow_extractor.get_train_features()
        z_val_bow = bow_extractor.transform(x_val_bow)

        bow_model = _build_local_model(model_name, **(model_kwargs or {}))
        try:
            bow_model.fit(x_train=z_train_bow, y_train=y_train)
        except TypeError:
            bow_model.fit(z_train_bow, y_train)
        bow_pred = bow_model.predict(z_val_bow)
        bow_acc = accuracy_score(y_val, bow_pred)
        bow_f1 = f1_score(y_val, bow_pred, average="macro")
        bow_weighted_f1 = f1_score(y_val, bow_pred, average="weighted")
        bow_report = classification_report(
            y_val,
            bow_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        tfidf_wrapper = RTextirWrapper(auto_install=False)
        tfidf_extractor = MNIRFeatureExtractor(tfidf_wrapper)
        tfidf_extractor.fit(x_train_tfidf, y_train, load_cached=False)
        z_train_tfidf = tfidf_extractor.get_train_features()
        z_val_tfidf = tfidf_extractor.transform(x_val_tfidf)

        tfidf_model = _build_local_model(model_name, **(model_kwargs or {}))
        try:
            tfidf_model.fit(x_train=z_train_tfidf, y_train=y_train)
        except TypeError:
            tfidf_model.fit(z_train_tfidf, y_train)
        tfidf_pred = tfidf_model.predict(z_val_tfidf)
        tfidf_acc = accuracy_score(y_val, tfidf_pred)
        tfidf_f1 = f1_score(y_val, tfidf_pred, average="macro")
        tfidf_weighted_f1 = f1_score(y_val, tfidf_pred, average="weighted")
        tfidf_report = classification_report(
            y_val,
            tfidf_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        bow_row = {
            "fold": fold_id,
            "representation": "bow",
            "accuracy": bow_acc,
            "macro_f1": bow_f1,
            "weighted_f1": bow_weighted_f1,
        }
        tfidf_row = {
            "fold": fold_id,
            "representation": "tfidf",
            "accuracy": tfidf_acc,
            "macro_f1": tfidf_f1,
            "weighted_f1": tfidf_weighted_f1,
        }
        for label in labels:
            bow_row[f"{label}_f1"] = bow_report.get(label, {}).get("f1-score", 0.0)
            tfidf_row[f"{label}_f1"] = tfidf_report.get(label, {}).get("f1-score", 0.0)

        fold_rows.extend([bow_row, tfidf_row])
        print(f"[MNIR-CV] fold={fold_id:02d} | BoW={bow_f1:.4f} | TF-IDF={tfidf_f1:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    main_summary_df = (
        fold_df.groupby("representation")[["accuracy", "macro_f1", "weighted_f1"]]
        .mean()
        .rename(columns={"accuracy": "accuracy_mean", "macro_f1": "macro_f1_mean", "weighted_f1": "weighted_f1_mean"})
    )
    main_summary_df["macro_f1_std"] = fold_df.groupby("representation")["macro_f1"].std(ddof=0)
    main_summary_df = main_summary_df[["accuracy_mean", "macro_f1_mean", "weighted_f1_mean", "macro_f1_std"]]

    class_cols = [f"{label}_f1" for label in labels]
    class_f1_summary_df = fold_df.groupby("representation")[class_cols].mean()
    class_f1_summary_df.columns = labels

    output_prefix = f"mnir_{model_name}_cv"
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = DEFAULT_REPORTS_DIR / f"{output_prefix}_fold_detail.csv"
    main_path = DEFAULT_REPORTS_DIR / f"{output_prefix}_main_summary.csv"
    class_path = DEFAULT_REPORTS_DIR / f"{output_prefix}_class_f1_summary.csv"
    workbook_path = DEFAULT_REPORTS_DIR / f"{output_prefix}_report.xlsx"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    main_summary_df.to_csv(main_path, encoding="utf-8-sig")
    class_f1_summary_df.to_csv(class_path, encoding="utf-8-sig")
    with pd.ExcelWriter(workbook_path) as writer:
        fold_df.to_excel(writer, sheet_name="fold_detail", index=False)
        main_summary_df.to_excel(writer, sheet_name="main_summary")
        class_f1_summary_df.to_excel(writer, sheet_name="class_f1_summary")

    print(f"[MNIR-CV] saved: {fold_path}")
    print(f"[MNIR-CV] saved: {main_path}")
    print(f"[MNIR-CV] saved: {class_path}")
    print(f"[MNIR-CV] saved: {workbook_path}")
    return fold_df, main_summary_df, class_f1_summary_df

def build_classification_report_table(y_true: np.ndarray, y_pred: np.ndarray, label_order: list[str]) -> pd.DataFrame:
    """依據 y_true 與 y_pred 建立 DataFrame 形式的 Classification Report"""
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0,
    )
    class_report_df = pd.DataFrame(
        {
            "class": label_order,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_values,
            "support": support.astype(int),
        }
    )
    return class_report_df

def build_pooled_confusion_matrix(prediction_df: pd.DataFrame, label_order: list[str]) -> pd.DataFrame:
    """依據總體的預測集建立 Pooled Confusion Matrix DataFrame"""
    cm = confusion_matrix(prediction_df["y_true"], prediction_df["y_pred"], labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    return cm_df

def run_repeated_cv(X: sp.spmatrix, y: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """最小可用版本的 CV 執行函式 (包裝 export_repeated_cv_outputs 避免巨幅重構)。"""
    return export_repeated_cv_outputs(
        X, y,
        model_name=config.get("model_name", "rf"),
        model_kwargs=config.get("model_kwargs", None),
        label_order=config.get("keep_labels", ["勝訴", "敗訴", "部分勝訴"]),
        n_splits=config.get("n_splits", 5),
        n_repeats=config.get("n_repeats", 3),
        refit_full_for_interpretability=False
    )

def summarize_cv_results(cv_results: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """最小可用版本：從 run_repeated_cv 結果中萃取 fold_df, summary_df, pred_df"""
    return cv_results["fold_df"], cv_results["main_summary_df"], cv_results["prediction_df"]

def run_full_experiment(X: sp.spmatrix, y: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """
    輕量高層函式：串接 CV、整理 summary 表格、建立 class report 與 confusion matrix。
    不強制存檔，回傳完整分析物件供 Notebook 中探索或後續寫入硬碟。
    """
    labels = config.get("keep_labels", ["勝訴", "敗訴", "部分勝訴"])

    cv_results = run_repeated_cv(X, y, config)
    fold_df, summary_df, pred_df = summarize_cv_results(cv_results)

    class_report_df = build_classification_report_table(pred_df["y_true"], pred_df["y_pred"], labels)
    cm_df = build_pooled_confusion_matrix(pred_df, labels)

    return {
        "cv_results": cv_results,
        "fold_df": fold_df,
        "summary_df": summary_df,
        "prediction_df": pred_df,
        "class_report_df": class_report_df,
        "confusion_matrix_df": cm_df
    }
