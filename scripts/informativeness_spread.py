"""Why SASLA prunes classification problems harder.

Retrains the passive networks of the first 10 final runs and measures how
spread out the informativeness is, split into the output and hidden factors.

  results/informativeness_spread.csv, .json, .npz
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.data import ALL, load, make_split
from src.network import FFNN
from src.sweep import BUDGET, FINAL_SEED_BASE, RESULTS, write_csv

N_SEEDS = 10


def one(args):
    key, seed, eta, alpha, lam = args
    prob = load(key)
    rng = np.random.default_rng(seed)
    split = make_split(prob, seed)
    net = FFNN(prob.n_inputs, prob.hidden, prob.n_outputs, rng)
    for _ in range(BUDGET[key]["final_epochs"]):
        net.train_epoch(split.Zc, split.Tc, eta, alpha, lam, rng)
    phi = net.informativeness(split.Zc)
    r = phi / phi.mean()
    Hb, O = net.forward(split.Zc)
    H = Hb[:, :net.J]
    # decomposition of the informativeness into the output factor o_k(1-o_k) and the
    # hidden factor ||sum_j w_kj y_j(1-y_j) v_ji||, both at the output unit k that attains the maximum
    S = net.sensitivity(split.Zc)
    norms = np.sqrt(np.sum(S ** 2, axis=2))                 # (n, K)
    kstar = np.argmax(norms, axis=1)
    idx = np.arange(len(kstar))
    a_out = (O * (1.0 - O))[idx, kstar]
    b_hid = norms[idx, kstar] / a_out
    cv_of = lambda x: float(np.std(x) / np.mean(x))
    row = dict(problem=key, seed=seed, max_over_mean=float(r.max()),
               cv=float(r.std()), below_0_1=float(np.mean(r < 0.1)),
               below_0_5=float(np.mean(r < 0.5)), p10=float(np.percentile(r, 10)),
               p90=float(np.percentile(r, 90)),
               # saturation of the hidden units, which drives the spread of informativeness
               hidden_saturated=float(np.mean((H < 0.05) | (H > 0.95))),
               hidden_derivative=float(np.mean(H * (1.0 - H))),
               output_derivative=float(np.mean(O * (1.0 - O))),
               cv_output_factor=cv_of(a_out), cv_hidden_factor=cv_of(b_hid),
               output_factor_ratio=float(a_out.max() / a_out.min()),
               hidden_factor_ratio=float(b_hid.max() / b_hid.min()),
               logvar_share_output=float(np.var(np.log(a_out)) / np.var(np.log(phi))))
    return row, r


def main():
    lam = json.load(open(os.path.join(RESULTS, "chosen_lambda.json")))
    ea = json.load(open(os.path.join(RESULTS, "chosen_eta_alpha.json")))
    jobs = [(k, FINAL_SEED_BASE + i, ea[k]["eta"], ea[k]["alpha"], lam[k])
            for k in ALL for i in range(N_SEEDS)]
    print(f"informativeness spread: {len(jobs)} passive networks", flush=True)
    rows, pooled = [], {k: [] for k in ALL}
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        for row, r in ex.map(one, jobs, chunksize=1):
            rows.append(row)
            pooled[row["problem"]].append(r)
    write_csv(rows, "informativeness_spread.csv")
    np.savez_compressed(os.path.join(RESULTS, "informativeness_spread.npz"),
                        **{k: np.concatenate(v) for k, v in pooled.items()})
    summary = {}
    print(f"\n{'problem':<16}{'max/mean':>12}{'cv':>12}{'<0.1 mean %':>14}{'<0.5 mean %':>14}")
    for k in ALL:
        sel = [r for r in rows if r["problem"] == k]
        s = {f: (float(np.mean([r[f] for r in sel])), float(np.std([r[f] for r in sel], ddof=1)))
             for f in ("max_over_mean", "cv", "below_0_1", "below_0_5",
                       "hidden_saturated", "hidden_derivative", "output_derivative",
                       "cv_output_factor", "cv_hidden_factor", "output_factor_ratio",
                       "hidden_factor_ratio", "logvar_share_output")}
        summary[k] = s
        print(f"{k:<16}{s['max_over_mean'][0]:>8.2f}+-{s['max_over_mean'][1]:<4.2f}"
              f"{s['cv'][0]:>8.2f}+-{s['cv'][1]:<4.2f}"
              f"{100 * s['below_0_1'][0]:>10.1f}{100 * s['below_0_5'][0]:>14.1f}"
              f"   saturated hidden {100 * s['hidden_saturated'][0]:5.1f}%"
              f"   mean y(1-y) {s['hidden_derivative'][0]:.3f}"
              f"   mean o(1-o) {s['output_derivative'][0]:.3f}"
              f"   cv out {s['cv_output_factor'][0]:.2f} cv hid {s['cv_hidden_factor'][0]:.2f}"
              f"   ratio out {s['output_factor_ratio'][0]:.1f} ratio hid {s['hidden_factor_ratio'][0]:.1f}"
              f"   logvar share out {s['logvar_share_output'][0]:.2f}")
    with open(os.path.join(RESULTS, "informativeness_spread.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
