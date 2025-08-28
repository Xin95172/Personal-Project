from typing import Literal
import numpy as np
import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier

def fit_neural_network_classifier(
    x_train: np.ndarray | sp.spmatrix,  # 特徵矩陣（BoW / TF-IDF）
    y_train: np.ndarray,                # 標籤向量
    hidden_layer_sizes: tuple[int, ...] = (256, 128),  # 隱藏層結構
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu",           # 激活函數: 'identity', 'logistic', 'tanh', 'relu'
    solver: Literal["lbfgs", "sgd", "adam"] = "adam",                # 優化器: 'lbfgs', 'sgd', 'adam'
    alpha: float = 0.0001,               # L2 正則化係數
    batch_size: int | str = "auto",      # 小批次大小
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant",     # 學習率調整策略
    learning_rate_init: float = 0.001,   # 初始學習率
    max_iter: int = 200,                 # 最大迭代次數
    random_state: int = 42               # 固定隨機種子，確保可重現
) -> MLPClassifier:
    """
    訓練全連接神經網路分類器
    """
    clf = MLPClassifier(
        hidden_layer_sizes = hidden_layer_sizes,
        activation = activation,
        solver = solver,
        alpha = alpha,
        batch_size = batch_size,
        learning_rate = learning_rate,
        learning_rate_init = learning_rate_init,
        max_iter = max_iter,
        random_state = random_state
    )
    clf.fit(x_train, y_train)
    return clf