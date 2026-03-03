import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC


def _ensure_1d_labels(y: np.ndarray, name: str = "y_train") -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    elif y_arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={y_arr.shape}")
    return y_arr


class SVMClassifier:
    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        self.model = LinearSVC(max_iter=max_iter, C=C)
        self.is_fitted = False

    def fit(self, x_train: sp.csr_matrix, y_train: np.ndarray) -> None:
        y_arr = _ensure_1d_labels(y_train)
        self.model.fit(x_train, y_arr)
        self.is_fitted = True

    def predict(self, x_test: np.ndarray | sp.csr_matrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Please call 'fit' method first.")
        return self.model.predict(x_test)
