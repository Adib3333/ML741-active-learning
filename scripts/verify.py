"""Checks the code against independent calculations: the sensitivity and the
gradient against finite differences, and the Friedman test against SciPy.

  results/verification.json
"""

import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as sps

from src.data import ALL, load
from src.network import FFNN
from src.stats import effect_size_a12, friedman, holm
from src.sweep import RESULTS


def forward_out(net, z):
    return net.predict(z[None, :])[0]


def check_sensitivity(h=1e-6):
    worst = 0.0
    rng = np.random.default_rng(7)
    for key in ALL:
        p = load(key)
        net = FFNN(p.n_inputs, p.hidden, p.n_outputs, rng)
        Z = np.column_stack([rng.uniform(-1, 1, (5, p.n_inputs)), -np.ones(5)])
        S = net.sensitivity(Z)
        for n in range(Z.shape[0]):
            for i in range(p.n_inputs):
                zp, zm = Z[n].copy(), Z[n].copy()
                zp[i] += h
                zm[i] -= h
                num = (forward_out(net, zp) - forward_out(net, zm)) / (2 * h)
                worst = max(worst, float(np.max(np.abs(num - S[n, :, i]))))
    return worst


def check_informativeness():
    worst = 0.0
    rng = np.random.default_rng(11)
    for key in ALL:
        p = load(key)
        net = FFNN(p.n_inputs, p.hidden, p.n_outputs, rng)
        Z = np.column_stack([rng.uniform(-1, 1, (5, p.n_inputs)), -np.ones(5)])
        phi = net.informativeness(Z)
        Hb, O = net.forward(Z)
        for n in range(Z.shape[0]):
            per_output = []
            for k in range(p.n_outputs):
                total = 0.0
                for i in range(p.n_inputs):
                    s = 0.0
                    for j in range(p.hidden):
                        y = Hb[n, j]
                        s += net.W[k, j] * y * (1.0 - y) * net.V[j, i]
                    s *= O[n, k] * (1.0 - O[n, k])
                    total += s * s
                per_output.append(np.sqrt(total))
            worst = max(worst, abs(max(per_output) - phi[n]))
    return float(worst)


def check_gradient(h=1e-6, eta=1e-3, lam=1e-2):
    rng = np.random.default_rng(3)
    I, J, K = 3, 4, 2
    net = FFNN(I, J, K, rng)
    z = np.append(rng.uniform(-1, 1, I), -1.0)
    t = np.array([0.9, 0.1])

    def objective(V, W):
        yb = np.append(1.0 / (1.0 + np.exp(-(V @ z))), -1.0)
        o = 1.0 / (1.0 + np.exp(-(W @ yb)))
        return 0.5 * np.sum((t - o) ** 2) + lam * 0.5 * (np.sum(V ** 2) + np.sum(W ** 2))

    V0, W0 = net.V.copy(), net.W.copy()
    gV, gW = np.zeros_like(V0), np.zeros_like(W0)
    for idx in itertools.product(range(V0.shape[0]), range(V0.shape[1])):
        Vp, Vm = V0.copy(), V0.copy()
        Vp[idx] += h
        Vm[idx] -= h
        gV[idx] = (objective(Vp, W0) - objective(Vm, W0)) / (2 * h)
    for idx in itertools.product(range(W0.shape[0]), range(W0.shape[1])):
        Wp, Wm = W0.copy(), W0.copy()
        Wp[idx] += h
        Wm[idx] -= h
        gW[idx] = (objective(V0, Wp) - objective(V0, Wm)) / (2 * h)

    net.train_epoch(z[None, :], t[None, :], eta, 0.0, lam, np.random.default_rng(0))
    stepV = (net.V - V0) / (-eta)
    stepW = (net.W - W0) / (-eta)
    return float(max(np.max(np.abs(stepV - gV)), np.max(np.abs(stepW - gW))))


def check_statistics():
    rng = np.random.default_rng(5)
    M = rng.random((6, 3))
    fr = friedman(M)
    ref = sps.friedmanchisquare(*M.T)
    friedman_diff = abs(fr["chi2"] - ref.statistic)
    # the same comparison with tied ranks, as on the Iris problem of the study
    Mt = M.copy()
    Mt[0, 1] = Mt[0, 0]
    Mt[3, 2] = Mt[3, 1]
    friedman_diff = max(friedman_diff, abs(friedman(Mt)["chi2"] - sps.friedmanchisquare(*Mt.T).statistic))

    # hand calculation: ranks (1,2,3), (1,3,2), (2,1,3), (1,2,3), (1,3,2), (1,2,3)
    Mh = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3], [1, 2, 3], [1, 3, 2], [1, 2, 3]], float)
    frh = friedman(Mh)
    R = np.array([7, 13, 16]) / 6.0
    chi2_hand = 12 * 6 / (3 * 4) * (np.sum(R ** 2) - 3 * 16 / 4.0)
    ff_hand = 5 * chi2_hand / (6 * 2 - chi2_hand)
    friedman_hand_diff = abs(frh["chi2"] - chi2_hand) + abs(frh["F"] - ff_hand)

    # Holm: sorted p = 0.01, 0.03, 0.04 -> 0.03, 0.06, max(0.06, 0.04) = 0.06
    h = holm({"a": 0.04, "b": 0.01, "c": 0.03})
    holm_diff = abs(h["b"]["p_holm"] - 0.03) + abs(h["c"]["p_holm"] - 0.06) + abs(h["a"]["p_holm"] - 0.06)

    x, y = rng.integers(0, 5, 30).astype(float), rng.integers(0, 5, 30).astype(float)
    direct = np.mean([(a > b) + 0.5 * (a == b) for a in x for b in y])
    a12_diff = abs(effect_size_a12(x, y) - direct)
    return float(friedman_diff), float(friedman_hand_diff), float(holm_diff), float(a12_diff)


def main():
    out = {}
    out["sensitivity_max_abs_diff"] = check_sensitivity()
    out["informativeness_max_abs_diff"] = check_informativeness()
    out["gradient_max_abs_diff"] = check_gradient()
    f1, f2, hd, ad = check_statistics()
    out["friedman_vs_scipy_abs_diff"] = f1
    out["friedman_vs_hand_abs_diff"] = f2
    out["holm_vs_hand_abs_diff"] = hd
    out["a12_vs_direct_abs_diff"] = ad
    for k, v in out.items():
        print(f"  {k:<34} {v:.3e}")
    with open(os.path.join(RESULTS, "verification.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {os.path.join(RESULTS, 'verification.json')}")


if __name__ == "__main__":
    main()
