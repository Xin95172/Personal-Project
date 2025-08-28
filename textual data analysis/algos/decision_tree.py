from typing import Literal
import numpy as np
import scipy.sparse as sp
from sklearn.tree import DecisionTreeClassifier

DTType = Literal["gini", "entropy"]

def fit_decision_tree_classifier(
    x_train: np.ndarray | sp.spmatrix,   # 特徵矩陣 (BoW/TF-IDF)，稠密或稀疏
    y_train: np.ndarray,                 # 標籤向量
    criterion: DTType = "entropy",       # 分裂準則，[gini, entropy]
    max_depth: int | None = None,        # 限制深度以防過擬合
    min_samples_leaf: int = 1,           # 葉節點最少樣本數
    random_state: int = 42               # 固定隨機種子，結果可重現
) -> DecisionTreeClassifier:
    """
    訓練 Decision Tree classifier
    """
    clf = DecisionTreeClassifier(
        criterion = criterion,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        random_state = random_state
    )
    clf.fit(x_train, y_train)
    return clf