from typing import Literal

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier


WEIGHTS = Literal["uniform", "distance"]
ALGORITHMS = Literal["auto", "ball_tree", "kd_tree", "brute"]
METRICS = Literal["minkowski", "euclidean", "manhattan", "cosine"]


class KNNClassifier:
    """initialize KNN Classifier"""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: WEIGHTS = "uniform",
        algorithm: ALGORITHMS = "auto",
        metric: METRICS = "minkowski",
        p: int = 2,
        n_jobs: int = -1,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.is_fitted = False

        if self.metric == "cosine" and self.algorithm != "brute":
            print("metric='cosine' requires algorithm='brute'; switching automatically.")
            self.algorithm = "brute"

        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )

    def fit(self, x_train: np.ndarray | sp.spmatrix, y_train: np.ndarray) -> None:
        """Fit KNN model."""
        y_arr = np.asarray(y_train)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        elif y_arr.ndim != 1:
            raise ValueError(f"y_train 必須是一維陣列，目前形狀: {y_arr.shape}")

        self.model.fit(x_train, y_arr)
        self.is_fitted = True

    def predict(self, x_test: np.ndarray | sp.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("go fitting the model first")
        return self.model.predict(x_test)
