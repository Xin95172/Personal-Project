from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd


def evaluate_metrics(
    y_true: np.ndarray, # true Labels
    y_pred: np.ndarray, # predicted results
    labels: list | None = None, # ordinize labels
    zero_division: int | str = 0, # control undefined metric behavior (0, 1, or "warn")
) -> None:
    """
    輸出 classification report 和 confusion matrix
    """
    print("====== classification report ======")
    print(classification_report(
        y_true,
        y_pred,
        digits = 5,
        zero_division = zero_division,
        labels = labels if labels else None,
    ))

    if labels:
        print("====== confusion matrix ======")
        cm = confusion_matrix(y_true, y_pred, labels = labels)
        print(pd.DataFrame(
            cm,
            index = pd.Index(labels, name = 'True'),
            columns = pd.Index(labels, name = 'Predicted')
        ))
