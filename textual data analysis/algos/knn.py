from typing import Literal
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier

def fit_knn_classifier(
    x_train: np.ndarray | sp.spmatrix,  # 訓練特徵矩陣（BoW / TF-IDF，可為稠密或稀疏格式）
    y_train: np.ndarray,                # 標籤向量（長度必須等於樣本數）
    n_neighbors: int = 5,               # K 值（鄰居數量）
    weights: Literal["uniform", "distance"] = "uniform",
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    metric: Literal["minkowski", "euclidean", "manhattan", "cosine"] = "minkowski",
    p: int = 2,                         # Minkowski 次方（p = 2 -> 歐幾里得距離，p = 1 -> 曼哈頓距離）
    n_jobs: int = -1                    # CPU 核心數（-1 表示使用全部核心）
) -> KNeighborsClassifier:
    """
    訓練 KNN 分類器
    """
    # 確保 y_train 是一維 NumPy 陣列
    y_arr = np.asarray(y_train)
    if y_arr.ndim != 1:
        raise ValueError(f"y_train 必須是一維陣列，目前形狀: {y_arr.shape}")

    # 特殊情況：metric=cosine 時必須搭配 algorithm='brute'
    if metric == "cosine" and algorithm != "brute":
        print("提示：metric = 'cosine' 需搭配 algorithm = 'brute'，已自動切換")
        algorithm = "brute"

    clf = KNeighborsClassifier(
        n_neighbors = n_neighbors,
        weights = weights,
        algorithm = algorithm,
        metric = metric,
        p = p,            # 僅在 metric='minkowski' 時有效
        n_jobs = n_jobs
    )
    clf.fit(x_train, y_arr)
    return clf