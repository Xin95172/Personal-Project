from typing import Literal
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier


WEIGHTS = Literal["uniform", "distance"]
ALGORITHMS = Literal["auto", "ball_tree", "kd_tree", "brute"]
METRICS = Literal["minkowski", "euclidean", "manhattan", "cosine"]
class KNNClassifier:
    """
    initialize KNN Classifier
    """
    def __init__(
        self,
        n_neighbors: int = 5,                     # K 值（鄰居數量）
        weights: WEIGHTS = "uniform",
        algorithm: ALGORITHMS = "auto",
        metric: METRICS = "minkowski",
        p: int = 2,                               # Minkowski 次方（p = 2 -> 歐幾里得距離，p = 1 -> 曼哈頓距離）
        n_jobs: int = -1                          # CPU 核心數（-1 表示使用全部核心）
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.is_fitted = False

        # 特殊情況：metric=cosine 時必須搭配 algorithm='brute'
        if self.metric == "cosine" and self.algorithm != "brute":
            print("提示：metric = 'cosine' 需搭配 algorithm = 'brute'，已自動切換")
            self.algorithm = "brute"

        self.model = KNeighborsClassifier(
            n_neighbors = self.n_neighbors,     # K 值（鄰居數量）
            weights = self.weights,
            algorithm = self.algorithm,
            metric = self.metric,
            p = self.p,                         # Minkowski 次方（p = 2 -> 歐幾里得距離，p = 1 -> 曼哈頓距離）
            n_jobs = self.n_jobs   
        )

    def fit(self, x_train: np.ndarray | sp.spmatrix, y_train: np.ndarray) -> None:
        """
        訓練 KNN 分類器
        """
        # 確保 y_train 是一維 NumPy 陣列
        y_arr = np.asarray(y_train)
        if y_arr.ndim != 1:
            raise ValueError(f"y_train 必須是一維陣列，目前形狀: {y_arr.shape}")

        self.model.fit(x_train, y_train)
        self.is_fitted = True

    def predict(self, x_test: np.ndarray | sp.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("go fitting the model first")
        return self.model.predict(x_test)