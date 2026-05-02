from typing import Literal

import numpy as np
import scipy.sparse as sp
from sklearn.tree import DecisionTreeClassifier


DTType = Literal['gini', 'entropy']


def _ensure_1d_labels(y: np.ndarray, name: str = 'y_train') -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    elif y_arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={y_arr.shape}")
    return y_arr


def fit_decision_tree_classifier(
    x_train: np.ndarray | sp.spmatrix,
    y_train: np.ndarray,
    criterion: DTType = 'entropy',
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    y_arr = _ensure_1d_labels(y_train)
    clf.fit(x_train, y_arr)
    return clf
