from typing import Literal
import numpy as np
import scipy
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB


NBType = Literal["multinomial", "bernoulli", "complement"]
NBModels = MultinomialNB | BernoulliNB | ComplementNB

class NaiveBayesClassifier:
    def __init__(
            self,
            model: NBType = "multinomial",                  # "multinomial", "bernoulli", "complement"
            alpha: float = 1.0                              # 平滑參數
    ):
        self.model = model
        self.alpha = alpha
        if model == "multinomial":
            self.clf = MultinomialNB(alpha = alpha)
        elif model == "bernoulli":
            self.clf = BernoulliNB(alpha = alpha)
        elif model == "complement":
            self.clf = ComplementNB(alpha = alpha)
        self.is_fitted = False

    def fit(
            self,
            x_train: np.ndarray | scipy.sparse.spmatrix,    # Bow 或 TF-IDF，shape = (n_samples, n_features)
            y_train: np.ndarray,                            # 標籤向量，長度 = n_samples
    ) -> NBModels:
        """
        fit the model
        """
        self.clf.fit(x_train, y_train)
        self.is_fitted = True

        return self.clf

    def predict(
            self,
            x_test: np.ndarray | scipy.sparse.spmatrix # Bow 或 TF-IDF，shape = (n_samples, n_features)
    ) -> np.ndarray:
        """
        make prediction
        """
        if not self.is_fitted:
            raise RuntimeError("go fitting the model first")
        y_pred = self.clf.predict(x_test)

        return y_pred