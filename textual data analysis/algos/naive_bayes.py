from typing import Literal

import numpy as np
import scipy
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB


NBType = Literal['multinomial', 'bernoulli', 'complement', 'gaussian']
NBModels = MultinomialNB | BernoulliNB | ComplementNB | GaussianNB


def _ensure_1d_labels(y: np.ndarray, name: str = 'y_train') -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    elif y_arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={y_arr.shape}")
    return y_arr


class NaiveBayesClassifier:
    def __init__(
        self,
        model: NBType = 'multinomial',
        alpha: float = 1.0,
    ):
        self.model = model
        self.alpha = alpha

        if model == 'multinomial':
            self.clf = MultinomialNB(alpha=alpha)
        elif model == 'bernoulli':
            self.clf = BernoulliNB(alpha=alpha)
        elif model == 'gaussian':
            self.clf = GaussianNB()
        else:
            self.clf = ComplementNB(alpha=alpha)

        self.is_fitted = False

    def fit(
        self,
        x_train: np.ndarray | scipy.sparse.spmatrix,
        y_train: np.ndarray,
    ) -> NBModels:
        y_arr = _ensure_1d_labels(y_train)
        x_fit = x_train.toarray() if (self.model == 'gaussian' and scipy.sparse.issparse(x_train)) else x_train
        self.clf.fit(x_fit, y_arr)
        self.is_fitted = True
        return self.clf

    def predict(self, x_test: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("go fitting the model first")
        x_pred = x_test.toarray() if (self.model == 'gaussian' and scipy.sparse.issparse(x_test)) else x_test
        return self.clf.predict(x_pred)