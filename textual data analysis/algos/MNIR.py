from typing import Any, Callable, Protocol, cast

import numpy as np
import scipy.sparse as sp
import rpy2.robjects as robjects
from rpy2.robjects import RS4
from rpy2.robjects.packages import importr
from sklearn.preprocessing import LabelEncoder

# 啟用 NumPy <-> R 物件自動轉換


def _as_callable(obj: Any, name: str) -> Callable[..., Any]:
    """把動態取得的 rpy2 物件視為可呼叫函式，並在執行期檢查。"""
    fn = cast(Callable[..., Any], obj)
    if not callable(fn):
        raise TypeError(f"{name} 不是可呼叫函式。")
    return fn


class RTextirWrapper:
    """Python 對 R 套件 textir 的簡單包裝。"""

    def __init__(self, auto_install: bool = False, cran_mirror: str = "https://cloud.r-project.org"):
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


class MNIRFeatureExtractor:
    """抽取 MNIR 特徵，可重複用於多個下游模型。"""

    def __init__(self, r_wrapper: RTextirWrapper):
        self.r_wrapper = r_wrapper
        self.is_fitted = False
        self._z_train_cache: np.ndarray | None = None

    def fit(self, X: sp.spmatrix, Y: np.ndarray) -> "MNIRFeatureExtractor":
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
        return self

    def fit_transform(self, X: sp.spmatrix, Y: np.ndarray) -> np.ndarray:
        self.fit(X, Y)
        return self.get_train_features()

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
            raise RuntimeError("尚未有訓練特徵快取，請先呼叫 fit 或 fit_transform。")
        return self._z_train_cache


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
            z_features = self.feature_extractor.fit_transform(X, y_arr)

        try:
            self.final_model.fit(x_train=z_features, y_train=y_arr)
        except TypeError:
            self.final_model.fit(z_features, y_arr)

        self.is_fitted = True
        print("[MNIR] 訓練完成。")
        return self

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型尚未訓練！")

        z_test = self.feature_extractor.transform(X)
        return self.final_model.predict(z_test)
