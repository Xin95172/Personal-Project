from typing import Any, Callable, cast

import numpy as np
import scipy.sparse as sp
import rpy2.robjects as robjects
from rpy2.robjects import RS4, numpy2ri
from rpy2.robjects.packages import importr

# 啟用 NumPy <-> R 物件自動轉換
numpy2ri.activate()


def _as_callable(obj: Any, name: str) -> Callable[..., Any]:
    """把動態取得的 rpy2 物件視為可呼叫函式，並在執行期檢查。"""
    fn = cast(Callable[..., Any], obj)
    if not callable(fn):
        raise TypeError(f"{name} 不是可呼叫函式。")
    return fn


class RTextirWrapper:
    """Python 對 R 套件 textir 的簡單包裝。"""

    def __init__(self, auto_install: bool = True):
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
            install_packages("textir")
            self.textir = importr("textir")
            print("[MNIR] textir 安裝並載入完成")

    def convert_to_r_format(self, X: sp.spmatrix, Y: np.ndarray) -> tuple[RS4, robjects.FloatVector]:
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

        Y_vector = robjects.FloatVector(Y)

        return r_sparse_matrix, Y_vector

    def fit_mnlm(self, X: RS4, Y: robjects.FloatVector) -> Any:
        """訓練 MultiNomial Linear Model 模型。"""
        self.model = self.textir.mnlm(covars=Y, counts=X)
        return self.model
    
    def transform_srproj(self, X: RS4) -> np.ndarray:
        """
        srproj: sufficient reduction projection
        使用訓練好的 mnlm 模型進行降維投影 (Sufficient Reduction)。
        將高維度的詞頻矩陣，壓縮成對應目標變數的特徵分數 (Z scores)。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練！請先呼叫 fit_mnlm。")

        # 呼叫 R 的 srproj 函數
        z_scores_r = self.textir.srproj(self.model, counts=X)

        # 將 R 的結果安全地轉換回 Python 的 NumPy 二維陣列
        z_scores_np = np.asarray(z_scores_r)
        
        return z_scores_np

    def get_coefs(self) -> np.ndarray:
        """
        提取 mnlm 模型中各字詞的迴歸係數 (權重)。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練！請先呼叫 fit_mnlm。")
            
        coefs_r = self.base.coef(self.model)
        
        return np.asarray(coefs_r)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

class MNIRPredictor:
    """
    全功能 MNIR 預測器：
    支持多種後端模型 (Regression 或 Classification)。
    """
    def __init__(self, r_wrapper: RTextirWrapper, final_model: Any):
        """
        :param r_wrapper: 你的 RTextirWrapper 實例
        :param final_model: 任何符合 sklearn 介面的模型實例 (例如 RandomForestRegressor())
        """
        self.r_wrapper = r_wrapper
        self.final_model = final_model
        self.is_fitted = False

    def fit(self, X: sp.spmatrix, Y: np.ndarray):
        print(f"[MNIR] 開始訓練流程，後端模型: {type(self.final_model).__name__}")
        
        # 1. 轉換並訓練 R 端的 mnlm
        r_X, r_Y = self.r_wrapper.convert_to_r_format(X, Y)
        self.r_wrapper.fit_mnlm(r_X, r_Y)
        
        # 2. 提取 Z 分數 (降維特徵)
        z_features = self.r_wrapper.transform_srproj(r_X)
        
        # 3. 訓練 Python 端的最終模型
        # 注意：Z 分數通常是 (n_samples, 1) 或 (n_samples, n_targets)
        self.final_model.fit(z_features, Y)
        
        self.is_fitted = True
        print("[MNIR] 訓練完成。")
        return self

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型尚未訓練！")
            
        # 轉換測試資料 (Y 給空值即可)
        dummy_y = np.zeros(X.shape[0])
        r_X_test, _ = self.r_wrapper.convert_to_r_format(X, dummy_y)
        
        # 提取測試集的 Z 分數
        z_test = self.r_wrapper.transform_srproj(r_X_test)
        
        # 最終預測
        return self.final_model.predict(z_test)