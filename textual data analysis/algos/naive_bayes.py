from typing import Literal, Union
import numpy as np
import scipy
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB


NBType = Literal["multinomial", "bernoulli", "complement"]
NBModels = Union[MultinomialNB, BernoulliNB, ComplementNB]

def fit_naive_bayes_classifier(
    x_train: np.ndarray | scipy.sparse.spmatrix, # Bow 或 TF-IDF，shape = (n_samples, n_features)
    y_train: np.ndarray, # 標籤向量，長度 = n_samples
    model: NBType = "multinomial",
    alpha: float = 1.0, # 平滑參數
) -> NBModels:
    """
    訓練 naive bayes classifier
    """
    if model == "multinomial":
        clf = MultinomialNB(alpha = alpha)
    elif model == "bernoulli":
        clf = BernoulliNB(alpha = alpha)
    elif model == "complement":
        clf = ComplementNB(alpha = alpha)

    clf.fit(x_train, y_train)

    return clf

def predict_naive_bayes_classifier(
        clf: NBModels, # 用 fit_naive_nayes_classifier 訓練好的 model
        x_test: np.ndarray|scipy.sparse.spmatrix # Bow 或 TF-IDF，shape = (n_samples, n_features)
) -> np.ndarray:
    """
    使用 naive bayes classifier 預測
    """
    y_pred = clf.predict(x_test)

    return y_pred