from typing import Literal
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier

DTType = Literal["gini", "entropy"]

def fit_random_forest_classifier(
    x_train: np.ndarray | sp.spmatrix,    # 特徵矩陣 (BoW/TF-IDF)，稠密或稀疏
    y_train: np.ndarray,                  # 標籤向量
    n_estimators: int = 100,              # 樹的數量
    criterion: DTType = "entropy",        # "gini" 或 "entropy"
    max_depth: int | None = None,         # 限制深度以防過擬合
    min_samples_leaf: int = 1,            # 葉節點最少樣本數
    max_features: float | Literal["sqrt", "log2"] = "sqrt",  # 每次分裂考慮的特徵數，float：比例；"sqrt"：平方根；"log2"：對數
    n_jobs: int = -1,                     # 平行運算核心數 (-1 = 全部核心)
    random_state: int = 42                # 固定亂數種子，方便重現結果
) -> RandomForestClassifier:
    """
    訓練 Random Forest 分類器
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion = criterion,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        n_jobs = n_jobs,
        random_state = random_state
    )
    clf.fit(x_train, y_train)
    return clf