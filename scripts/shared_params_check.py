"""Where phase 4d found a better pair, reruns the 30 final runs with that pair
and redoes the tests.

  results/shared_params_check.json
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALL, load
from src.stats import friedman, pairwise_wilcoxon
from src.sweep import BUDGET, FINAL_SEED_BASE, RESULTS, read_csv, run_cached
from scripts.phase5_compare import N_RUNS


def main():
    test = json.load(open(os.path.join(RESULTS, "shared_params_test.json")))
    lam = json.load(open(os.path.join(RESULTS, "chosen_lambda.json")))
    beta = json.load(open(os.path.join(RESULTS, "chosen_beta.json")))
    unc = json.load(open(os.path.join(RESULTS, "chosen_uncertainty.json")))
    final = read_csv("phase5_runs.csv")

    def key_measure(r):
        return r["gen_error_rate"] if load(r["problem"]).task == "classification" else r["gen_mse"]

    def vec(rows, key, strat):
        sel = sorted([r for r in rows if r["problem"] == key and r["strategy"] == strat],
                     key=lambda r: int(r["seed"]))
        return np.array([key_measure(r) for r in sel], float)

    out = {"cases": {}}
    replaced = {}
    for key in ALL:
        for strat in ("sasla", "uncertainty"):
            t = test[key][strat]
            if t["within_one_se"]:
                continue
            sk = dict(beta=beta[key]) if strat == "sasla" else dict(unc[key])
            jobs = [(key, strat, FINAL_SEED_BASE + i, dict(
                eta=t["best_eta"], alpha=t["best_alpha"], lam=lam[key],
                max_epochs=BUDGET[key]["final_epochs"], keep_history=False,
                strategy_kwargs=sk)) for i in range(N_RUNS)]
            rows = run_cached(f"sharedcheck_{key}_{strat}", jobs)
            replaced[(key, strat)] = rows
            new = vec(rows, key, strat)
            old = vec(final, key, strat)
            pas = vec(final, key, "passive")
            pres_new = float(np.mean([r["presentations_total"] for r in rows]))
            pres_pas = float(np.mean([r["presentations_total"] for r in final
                                      if r["problem"] == key and r["strategy"] == "passive"]))
            samples = {"Passive": pas,
                       "SASLA": new if strat == "sasla" else vec(final, key, "sasla"),
                       "Uncertainty": new if strat == "uncertainty" else vec(final, key, "uncertainty")}
            tests = pairwise_wilcoxon(samples)
            out["cases"][f"{key}|{strat}"] = dict(
                eta=t["best_eta"], alpha=t["best_alpha"],
                mean_old=float(old.mean()), mean_new=float(new.mean()),
                mean_passive=float(pas.mean()),
                presentations_rel_new=100.0 * pres_new / pres_pas,
                tests={k: dict(p_holm=v["p_holm"], reject=v["reject"], better=v["better"])
                       for k, v in tests.items()})
            print(f"{key:<16}{strat:<13} eta={t['best_eta']:g} alpha={t['best_alpha']:g}  "
                  f"mean {old.mean():.5f} -> {new.mean():.5f}  (passive {pas.mean():.5f})")
            for k, v in tests.items():
                print(f"    {k:<24} p_holm={v['p_holm']:.3f}  {'SIGNIFICANT' if v['reject'] else ''}")

    # Friedman across problems with the repeated runs in place
    M = []
    for key in ALL:
        row = []
        for strat in ("passive", "sasla", "uncertainty"):
            if (key, strat) in replaced:
                row.append(vec(replaced[(key, strat)], key, strat).mean())
            else:
                row.append(vec(final, key, strat).mean())
        M.append(row)
    fr = friedman(np.array(M))
    out["friedman"] = dict(avg_ranks=list(map(float, fr["avg_ranks"])), chi2=fr["chi2"], p_chi2=fr["p_chi2"],
                           F=fr["F"], p_F=fr["p_F"])
    print(f"Friedman with the repeated runs: ranks {np.round(fr['avg_ranks'], 3)}  "
          f"chi2={fr['chi2']:.3f} p={fr['p_chi2']:.3f}  F={fr['F']:.3f} p={fr['p_F']:.3f}")
    with open(os.path.join(RESULTS, "shared_params_check.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
