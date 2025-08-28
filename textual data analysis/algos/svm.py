from sklearn.svm import LinearSVC
import scipy
import numpy as np

def fit_svm_classifier(
        x_train: np.ndarray | scipy.sparse.spmatrix, # Bow 或 TF-IDF，shape = (n_samples, n_features)
        y_train: np.ndarray # 標籤向量，長度 = n_samples
) -> LinearSVC:
    svm_clf = LinearSVC()
    svm_clf.fit(x_train, y_train)
    return svm_clf

def predict_svm_classifier(
        clf: LinearSVC,
        x_test: np.ndarray | scipy.sparse.spmatrix
) -> np.ndarray:
    y_pred = clf.predict(x_test)
    return y_pred