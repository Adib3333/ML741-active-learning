"""Loads the six problems and splits them.

Iris, Breast Cancer and Diabetes come from scikit-learn. Optical Digits, Fuel
Consumption (auto-mpg) and Concrete Strength are read from data/.

Each run draws a new 60/20/20 split, stratified for classification. Inputs are
scaled to [-1, 1] and targets to [0.1, 0.9] using the training part only, so
nothing leaks. Missing horsepower values get the training mean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw")

T_LO, T_HI = 0.1, 0.9


@dataclass
class Problem:
    """One benchmark problem, before any partitioning or scaling."""

    key: str
    name: str
    task: str  # "classification" or "regression"
    X: np.ndarray  # (n, I) raw inputs
    Y: np.ndarray  # (n,) integer labels, or (n, K) raw continuous targets
    n_classes: int  # 0 for regression
    hidden: int  # the overestimated hidden layer size used for every algorithm
    note: str = ""

    @property
    def n_inputs(self) -> int:
        return self.X.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.n_classes if self.task == "classification" else self.Y.shape[1]


# --- loaders

def _iris() -> Problem:
    from sklearn.datasets import load_iris

    d = load_iris()
    return Problem("iris", "Iris", "classification", d.data.astype(float),
                   d.target.astype(int), 3, hidden=20,
                   note="Balanced three-class problem, four continuous attributes.")


def _breast_cancer() -> Problem:
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer()
    return Problem("breast_cancer", "Breast Cancer", "classification",
                   d.data.astype(float), d.target.astype(int), 2, hidden=40,
                   note="Thirty continuous attributes, mild class imbalance (37/63).")


def _optdigits() -> Problem:
    path = os.path.join(RAW, "optdigits_all.csv")
    raw = np.loadtxt(path, delimiter=",")
    X, y = raw[:, :64], raw[:, 64].astype(int)
    return Problem("optdigits", "Optical Digits", "classification", X, y, 10, hidden=100,
                   note="Ten classes, 64 attributes formed from 8x8 blocks of a 32x32 bitmap.")


def _auto_mpg() -> Problem:
    """Auto MPG. Six horsepower values are missing and are mean imputed,
    following the convention used by Engelbrecht (2001)."""
    path = os.path.join(RAW, "auto-mpg.data")
    cols = []
    with open(path) as fh:
        for line in fh:
            head = line.split("\t")[0]
            parts = head.split()
            if len(parts) < 8:
                continue
            cols.append(parts[:8])
    arr = np.array(cols, dtype=object)
    mpg = arr[:, 0].astype(float)
    cyl = arr[:, 1].astype(float)
    disp = arr[:, 2].astype(float)
    hp = np.array([np.nan if v == "?" else float(v) for v in arr[:, 3]])
    wt = arr[:, 4].astype(float)
    acc = arr[:, 5].astype(float)
    yr = arr[:, 6].astype(float)
    org = arr[:, 7].astype(float).astype(int)

    # The six missing values stay NaN here. make_split replaces them by the mean
    # of the training partition, so that no held-out value influences training.

    origin = np.zeros((len(org), 3))
    origin[np.arange(len(org)), org - 1] = 1.0
    X = np.column_stack([cyl, disp, hp, wt, acc, yr, origin])
    return Problem("auto_mpg", "Fuel Consumption", "regression", X, mpg.reshape(-1, 1), 0, hidden=30,
                   note="Six missing horsepower values mean imputed; origin one-hot encoded.")


def _diabetes() -> Problem:
    from sklearn.datasets import load_diabetes

    d = load_diabetes()
    return Problem("diabetes", "Diabetes Progression", "regression",
                   d.data.astype(float), d.target.astype(float).reshape(-1, 1), 0, hidden=30,
                   note="Noisy target with a weak signal, providing a stress test.")


def _concrete() -> Problem:
    path = os.path.join(RAW, "Concrete_Data.csv")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return Problem("concrete", "Concrete Strength", "regression",
                   raw[:, :8], raw[:, 8].reshape(-1, 1), 0, hidden=50,
                   note="Strongly non-linear relationship between mixture and strength.")


LOADERS = {
    "iris": _iris,
    "breast_cancer": _breast_cancer,
    "optdigits": _optdigits,
    "auto_mpg": _auto_mpg,
    "diabetes": _diabetes,
    "concrete": _concrete,
}

CLASSIFICATION = ["iris", "breast_cancer", "optdigits"]
REGRESSION = ["auto_mpg", "diabetes", "concrete"]
ALL = CLASSIFICATION + REGRESSION

_CACHE: dict[str, Problem] = {}


def load(key: str) -> Problem:
    if key not in _CACHE:
        _CACHE[key] = LOADERS[key]()
    return _CACHE[key]


# --- partitioning and scaling

@dataclass
class Split:
    """One training / validation / generalisation partition, fully scaled."""

    Zc: np.ndarray   # candidate (training) inputs with bias column, (nC, I+1)
    Tc: np.ndarray   # candidate targets scaled to [0.1, 0.9], (nC, K)
    yc: np.ndarray   # candidate integer labels, classification only
    Zv: np.ndarray
    Tv: np.ndarray
    yv: np.ndarray
    Zg: np.ndarray
    Tg: np.ndarray
    yg: np.ndarray
    task: str
    n_classes: int
    t_lo: float = 0.0   # raw target minimum, for unscaling regression predictions
    t_hi: float = 1.0


def _add_bias(Z: np.ndarray) -> np.ndarray:
    out = np.empty((Z.shape[0], Z.shape[1] + 1))
    out[:, :-1] = Z
    out[:, -1] = -1.0
    return out


def _stratified_indices(y: np.ndarray, fracs, rng) -> list[np.ndarray]:
    """Split indices into len(fracs) groups, preserving class proportions."""
    groups = [[] for _ in fracs]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cuts = np.cumsum([int(round(f * len(idx))) for f in fracs[:-1]])
        for g, part in enumerate(np.split(idx, cuts)):
            groups[g].append(part)
    return [np.concatenate(g) for g in groups]


def make_split(prob: Problem, seed: int, train=0.6, val=0.2) -> Split:
    """Build one scaled train/validation/generalisation partition.

    Classification splits are stratified so that every partition carries the
    class distribution of the full problem. Scaling parameters come from the
    training partition alone.
    """
    rng = np.random.default_rng(seed)
    n = prob.X.shape[0]
    fracs = [train, val, 1.0 - train - val]

    if prob.task == "classification":
        tr, va, ge = _stratified_indices(prob.Y, fracs, rng)
    else:
        idx = rng.permutation(n)
        cuts = [int(round(train * n)), int(round((train + val) * n))]
        tr, va, ge = np.split(idx, cuts)

    X = prob.X
    if np.isnan(X).any():
        # mean imputation fitted on the training partition only
        X = X.copy()
        col_mean = np.nanmean(X[tr], axis=0)
        r, c = np.where(np.isnan(X))
        X[r, c] = col_mean[c]
    Xtr = X[tr]
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)

    def scale_x(X):
        return np.clip(2.0 * (X - lo) / span - 1.0, -3.0, 3.0)

    Zc, Zv, Zg = (_add_bias(scale_x(X[i])) for i in (tr, va, ge))

    if prob.task == "classification":
        K = prob.n_classes
        def onehot(y):
            T = np.full((len(y), K), T_LO)
            T[np.arange(len(y)), y] = T_HI
            return T
        yc, yv, yg = prob.Y[tr], prob.Y[va], prob.Y[ge]
        return Split(Zc, onehot(yc), yc, Zv, onehot(yv), yv, Zg, onehot(yg), yg,
                     prob.task, K)

    Ytr = prob.Y[tr]
    tlo, thi = Ytr.min(axis=0), Ytr.max(axis=0)
    tspan = np.where(thi - tlo < 1e-12, 1.0, thi - tlo)

    def scale_t(Y):
        return T_LO + (T_HI - T_LO) * (Y - tlo) / tspan

    return Split(Zc, scale_t(prob.Y[tr]), None,
                 Zv, scale_t(prob.Y[va]), None,
                 Zg, scale_t(prob.Y[ge]), None,
                 prob.task, 0, float(tlo[0]), float(thi[0]))


def summary_table() -> list[dict]:
    """Characteristics of all six problems, for the report."""
    rows = []
    for key in ALL:
        p = load(key)
        row = dict(key=key, name=p.name, task=p.task, patterns=p.X.shape[0],
                   inputs=p.n_inputs, outputs=p.n_outputs, hidden=p.hidden,
                   note=p.note)
        if p.task == "classification":
            counts = np.bincount(p.Y, minlength=p.n_classes)
            row["distribution"] = " / ".join(f"{100.0 * c / len(p.Y):.1f}" for c in counts)
            row["imbalance"] = float(counts.max() / counts.min())
        else:
            row["distribution"] = f"[{p.Y.min():.2f}, {p.Y.max():.2f}]"
            row["imbalance"] = float("nan")
        rows.append(row)
    return rows


if __name__ == "__main__":
    for r in summary_table():
        print(f"{r['name']:<22} {r['task']:<15} n={r['patterns']:>5} I={r['inputs']:>3} "
              f"K={r['outputs']:>3} J={r['hidden']:>3}  {r['distribution']}")
