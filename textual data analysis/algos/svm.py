from sklearn.svm import LinearSVC
import scipy.sparse as sp
import numpy as np


class SVMClassifier:
    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        """
        initialize svm classifier
        :max_iter: maximum number of iterations
        :C: regularization 強度，越小越強
        """
        self.model = LinearSVC(max_iter = max_iter, C = C)
        self.is_fitted = False

    def fit(self, x_train: sp.csr_matrix, y_train: np.ndarray) -> None:
        """
        fit the model
        :x_train: training data, shape = (n_samples, n_features)
        :y_train: labels, shape = (n_samples, )
        """
        self.model.fit(x_train, y_train)
        self.is_fitted = True

    def predict(self, x_test: np.ndarray | sp.csr_matrix):
        """
        predict the labels
        :x_test: test data, shape = (n_samples, n_features)
        :return: predicted labels, shape = (n_samples, )
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Please call 'fit' method first.")
        return self.model.predict(x_test)