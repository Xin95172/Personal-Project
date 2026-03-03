from typing import Literal

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier


DTType = Literal["gini", "entropy"]


def _ensure_1d_labels(y: np.ndarray, name: str = "y_train") -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    elif y_arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={y_arr.shape}")
    return y_arr


class RFClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: DTType = "entropy",
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: float | Literal["sqrt", "log2"] = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, x_train: np.ndarray | sp.spmatrix, y_train: np.ndarray) -> None:
        y_arr = _ensure_1d_labels(y_train)
        self.model.fit(x_train, y_arr)
        self.is_fitted = True

    def predict(self, x_test: np.ndarray | sp.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(x_test)
