"""Performance measures: MSE, error rate, macro F1, RMSE and the
generalisation factor rho = E_G / E_T.
"""

from __future__ import annotations

import numpy as np


def mse(O: np.ndarray, T: np.ndarray) -> float:
    """Mean squared error over all patterns and all K output units."""
    return float(np.mean((T - O) ** 2))


def rmse_raw(O: np.ndarray, T: np.ndarray, t_lo: float, t_hi: float) -> float:
    """Root mean squared error in the units of the original target."""
    span = (t_hi - t_lo) / 0.8          # targets were scaled into [0.1, 0.9]
    return float(np.sqrt(np.mean(((T - O) * span) ** 2)))


def accuracy(O: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(O, axis=1) == y))


def macro_f1(O: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    pred = np.argmax(O, axis=1)
    scores = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores))


def generalisation_factor(e_gen: float, e_train: float) -> float:
    """Roebel's generalisation factor, rho = E_G / E_T, with E an error.

    A value near one indicates that the network performs on unseen data as it
    does on training data. Larger values indicate overfitting of the training
    data.
    """
    return float(e_gen / max(e_train, 1e-12))


def evaluate(net, Z, T, y, task, n_classes, t_lo=0.0, t_hi=1.0) -> dict:
    """All performance measures for one network on one partition."""
    O = net.predict(Z)
    out = {"mse": mse(O, T)}
    if task == "classification":
        out["accuracy"] = accuracy(O, y)
        out["macro_f1"] = macro_f1(O, y, n_classes)
        out["error_rate"] = 1.0 - out["accuracy"]
    else:
        out["rmse"] = rmse_raw(O, T, t_lo, t_hi)
    return out
