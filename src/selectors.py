"""The three ways of choosing training patterns.

  passive      - every pattern, every epoch
  SASLA        - keeps patterns with informativeness above (1 - beta) * mean
  uncertainty  - starts small and adds the most uncertain patterns whenever
                 the validation/subset MSE ratio goes above theta
"""

from __future__ import annotations

import numpy as np


# --- uncertainty measures

def entropy_uncertainty(O: np.ndarray) -> np.ndarray:
    """Conditional entropy of the normalised output vector, per pattern.

    Sigmoid output units do not sum to one, so the output vector is
    normalised into a distribution before the entropy is taken.
    """
    P = O / np.maximum(O.sum(axis=1, keepdims=True), 1e-12)
    return -np.sum(P * np.log(np.maximum(P, 1e-12)), axis=1)


def committee_uncertainty(preds: np.ndarray) -> np.ndarray:
    """Mean prediction variance across a committee. preds is (C, n, K)."""
    return preds.var(axis=0).mean(axis=1)


# --- strategies

class Passive:
    """Fixed set learning. The control against which both active learners
    are measured."""

    key = "passive"
    label = "Passive"
    n_models = 1

    def __init__(self, **_):
        self.calls = 0          # applications of a selection operator (none)
        self.evaluations = 0    # candidate patterns evaluated by the operator

    def initial(self, split, rng):
        return np.arange(split.Zc.shape[0])

    def update(self, nets, split, current, rng, rho):
        return current


class SASLA:
    """Sensitivity analysis selective learning (Engelbrecht, 2001).

    Parameters
    ----------
    beta : float in (0, 1]
        Subset selection constant. A pattern is selected when its
        informativeness reaches (1 - beta) times the mean informativeness
        over the candidate set. Larger beta lowers the threshold and selects
        more patterns; beta = 1 degenerates to fixed set learning.
    interval : int
        Number of epochs between selection intervals.
    """

    key = "sasla"
    label = "SASLA"
    n_models = 1

    def __init__(self, beta: float = 0.9, interval: int = 1, **_):
        self.beta = beta
        self.interval = interval
        self._epoch = 0
        self.calls = 0          # selection intervals at which the operator was applied
        self.evaluations = 0    # candidate patterns evaluated by the operator

    def initial(self, split, rng):
        self._epoch = 0
        return np.arange(split.Zc.shape[0])  # training starts on the whole candidate set

    def update(self, nets, split, current, rng, rho):
        self._epoch += 1
        if self._epoch % self.interval:
            return current
        phi = nets[0].informativeness(split.Zc)
        self.calls += 1
        self.evaluations += split.Zc.shape[0]
        threshold = (1.0 - self.beta) * phi.mean()
        sel = np.where(phi > threshold)[0]
        if sel.size < 2:  # never let the subset collapse entirely
            sel = np.argsort(phi)[-2:]
        return sel


class UncertaintySampling:
    """Incremental active learning driven by an uncertainty measure.

    Parameters
    ----------
    seed_frac : float
        Fraction of the candidate set forming the initial training subset.
    growth : float
        Fraction of the candidate set added at each growth step.
    rho_max : float
        Generalisation factor threshold. Patterns are added only once the
        validation error exceeds rho_max times the training error on the
        current subset, which signals that the subset has been learned and
        further information is required (Roebel, 1994).
    committee : int
        Number of networks used to estimate uncertainty on function
        approximation problems.
    """

    key = "uncertainty"
    label = "Uncertainty"

    def __init__(self, seed_frac: float = 0.10, growth: float = 0.05,
                 rho_max: float = 1.15, committee: int = 3, task: str = "classification", **_):
        self.seed_frac = seed_frac
        self.growth = growth
        self.rho_max = rho_max
        self.n_models = 1 if task == "classification" else committee
        self.task = task
        self._pool = None
        self.calls = 0          # growth steps at which the operator was applied
        self.evaluations = 0    # candidate evaluations, counted per committee member

    def initial(self, split, rng):
        n = split.Zc.shape[0]
        k = max(4, int(round(self.seed_frac * n)))
        if split.task == "classification":
            chosen = []
            for c in np.unique(split.yc):
                idx = np.where(split.yc == c)[0]
                take = max(2, int(round(self.seed_frac * len(idx))))
                chosen.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
            sel = np.unique(np.concatenate(chosen))
        else:
            sel = rng.choice(n, size=min(k, n), replace=False)
        self._pool = np.setdiff1d(np.arange(n), sel)
        return np.sort(sel)

    def _uncertainty(self, nets, Z):
        if self.task == "classification":
            return entropy_uncertainty(nets[0].predict(Z))
        preds = np.stack([m.predict(Z) for m in nets], axis=0)
        return committee_uncertainty(preds)

    def update(self, nets, split, current, rng, rho):
        if self._pool.size == 0 or rho is None or rho <= self.rho_max:
            return current
        n_add = max(1, int(round(self.growth * split.Zc.shape[0])))
        n_add = min(n_add, self._pool.size)
        u = self._uncertainty(nets, split.Zc[self._pool])
        self.calls += 1
        self.evaluations += self._pool.size * len(nets)
        take = self._pool[np.argsort(u)[-n_add:]]
        self._pool = np.setdiff1d(self._pool, take)
        return np.sort(np.concatenate([current, take]))


STRATEGIES = {"passive": Passive, "sasla": SASLA, "uncertainty": UncertaintySampling}
ORDER = ["passive", "sasla", "uncertainty"]
