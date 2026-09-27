"""Statistics: Wilcoxon with Holm correction, Vargha-Delaney effect size,
Friedman with tie correction, Iman-Davenport and the Nemenyi critical difference.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

# Critical values of the studentised range statistic divided by sqrt(2),
# at a significance level of 0.05, indexed by the number of algorithms.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def ranks_per_problem(matrix: np.ndarray) -> np.ndarray:
    """Rank algorithms on each problem, rank 1 being the best (lowest error).

    matrix is (n_problems, n_algorithms). Ties receive the average rank.
    """
    return np.vstack([sps.rankdata(row) for row in matrix])


def friedman(matrix: np.ndarray) -> dict:
    """Friedman test with the Iman-Davenport correction.

    matrix is (N, k): performance of k algorithms on N problems, lower better.
    """
    N, k = matrix.shape
    R = ranks_per_problem(matrix)
    avg = R.mean(axis=0)
    chi2 = (12.0 * N / (k * (k + 1))) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4.0)
    # correction for tied ranks (Hollander and Wolfe), as in scipy.stats.friedmanchisquare
    ties = sum(float(np.sum(c ** 3 - c)) for c in (np.unique(row, return_counts=True)[1] for row in R))
    chi2 = chi2 / (1.0 - ties / (N * k * (k * k - 1)))
    p_chi = float(sps.chi2.sf(chi2, k - 1))
    denom = N * (k - 1) - chi2
    if denom <= 0:
        ff, p_f = float("inf"), 0.0
    else:
        ff = (N - 1) * chi2 / denom
        p_f = float(sps.f.sf(ff, k - 1, (k - 1) * (N - 1)))
    return {"N": N, "k": k, "avg_ranks": avg, "ranks": R,
            "chi2": float(chi2), "p_chi2": p_chi,
            "F": float(ff), "p_F": p_f, "df1": k - 1, "df2": (k - 1) * (N - 1)}


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    """Critical difference in average ranks for the Nemenyi post-hoc test."""
    q = NEMENYI_Q05[k] if abs(alpha - 0.05) < 1e-9 else NEMENYI_Q05[k]
    return float(q * np.sqrt(k * (k + 1) / (6.0 * N)))


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm step-down procedure controlling the family-wise error rate.

    Returns, for each comparison, the raw p-value, the Holm adjusted p-value
    and whether the null hypothesis is rejected at the given level.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[name] = {"p": float(p), "p_holm": float(adj), "reject": bool(adj < alpha)}
    return out


def pairwise_wilcoxon(samples: dict[str, np.ndarray], alpha: float = 0.05) -> dict:
    """Paired Wilcoxon signed-rank tests between every pair, Holm corrected.

    samples maps a strategy name to its vector of per-run results, with the
    runs of every strategy aligned on the same seeds.
    """
    names = list(samples)
    raw, detail = {}, {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            x, y = samples[a], samples[b]
            # Rounding removes floating point noise, so that equal differences
            # (for example one misclassified pattern) receive tied ranks.
            d = np.round(x - y, 12)
            if np.allclose(d, 0.0):
                p, stat = 1.0, 0.0
            else:
                stat, p = sps.wilcoxon(d, zero_method="wilcox")
            key = f"{a} vs {b}"
            raw[key] = float(p)
            # The metric is an error, so the smaller mean is the better one.
            # Medians are compared only as a tie breaker, because coarse error
            # rates on small partitions produce many exactly zero differences.
            if np.mean(x) < np.mean(y):
                better = a
            elif np.mean(y) < np.mean(x):
                better = b
            else:
                better = "tie"
            detail[key] = {"statistic": float(stat),
                           "mean_diff": float(np.mean(d)),
                           "median_diff": float(np.median(d)),
                           "better": better}
    adj = holm(raw, alpha)
    for key, d in detail.items():
        adj[key].update(d)
    return adj


def effect_size_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha and Delaney's A12: the probability that a randomly drawn
    observation from x exceeds one drawn from y, with ties counted as half.

    A value of 0.5 indicates no effect. Since the metric being compared is an
    error, a value below 0.5 means that x tends to produce the smaller error
    and is therefore the better of the two.
    """
    nx, ny = len(x), len(y)
    r = sps.rankdata(np.concatenate([x, y]))[:nx].sum()
    return float(((r / nx) - (nx + 1) / 2.0) / ny)


def describe(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)), "min": float(np.min(values)),
            "max": float(np.max(values))}
