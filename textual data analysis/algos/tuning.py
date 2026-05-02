from typing import Any, Callable

import numpy as np
from sklearn.metrics import f1_score


def tune_with_val(
    build_model: Callable[..., Any],
    param_grid: list[dict[str, Any]],
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    y_val: np.ndarray,
):
    """Tune model hyperparameters with a fixed validation set.

    The model is built by calling ``build_model(**params)`` for each param dict.
    This helper supports both API styles:
    - fit(x_train=..., y_train=...), predict(x_test=...)
    - fit(X, y), predict(X)
    """
    y_train_1d = np.asarray(y_train).ravel()
    y_val_1d = np.asarray(y_val).ravel()

    if len(param_grid) == 0:
        raise ValueError("param_grid is empty")

    best_score = -1.0
    best_params: dict[str, Any] | None = None
    best_model: Any = None

    for params in param_grid:
        model = build_model(**params)
        try:
            model.fit(x_train=x_train, y_train=y_train_1d)
            pred_val = model.predict(x_test=x_val)
        except TypeError:
            model.fit(x_train, y_train_1d)
            pred_val = model.predict(x_val)

        score = f1_score(
            y_val_1d,
            np.asarray(pred_val).ravel(),
            average='macro',
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was selected during tuning")

    return best_model, best_params, best_score
