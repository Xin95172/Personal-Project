from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd


def evaluate_metrics(
    y_true: np.ndarray, # true Labels
    y_pred: np.ndarray, # predicted results
    labels: list = None, # ordinize labels
) -> None:
    """
    輸出 classification report 和 confusion matrix
    """
    print("====== classification report ======")
    print(classification_report(y_true, y_pred, digits = 5))

    print("\n====== confusion matrix ======")
    if labels:
        cm = confusion_matrix(y_true, y_pred, labels = labels)
        print(pd.DataFrame(cm, index = labels, columns = labels))