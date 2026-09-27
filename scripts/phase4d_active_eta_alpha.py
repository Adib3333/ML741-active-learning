"""Phase 4d: is sharing passive learning's learning rate and momentum fair?
Searches the same grid for each active approach.

  results/phase4d_active_eta_alpha.csv, shared_params_test.json
"""

import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALL
from src.sweep import BUDGET, RESULTS, TUNE_SEED_BASE, run_cached, write_csv
from scripts.phase4_params import ETAS, ALPHAS, cell, load_json


def strategy_kwargs(strategy, key, beta, unc):
    if strategy == "sasla":
        return dict(beta=beta[key])
    sk = dict(seed_frac=unc[key]["seed_frac"], growth=unc[key]["growth"], rho_max=unc[key]["rho_max"])
    if "committee" in unc[key]:
        sk["committee"] = unc[key]["committee"]
    return sk


def main():
    t0 = time.perf_counter()
    lam = load_json("chosen_lambda.json")
    ea = load_json("chosen_eta_alpha.json")
    beta = load_json("chosen_beta.json")
    unc = load_json("chosen_uncertainty.json")

    rows = []
    for key in ALL:
        b = BUDGET[key]
        for strategy in ("sasla", "uncertainty"):
            jobs = []
            for eta, alpha in itertools.product(ETAS, ALPHAS):
                for s in range(b["tune_seeds"]):
                    jobs.append((key, strategy, TUNE_SEED_BASE + s, dict(
                        eta=eta, alpha=alpha, lam=lam[key],
                        max_epochs=b["tune_epochs"], keep_history=False,
                        strategy_kwargs=strategy_kwargs(strategy, key, beta, unc))))
            rows += run_cached(f"phase4d_{key}_{strategy}", jobs)
    write_csv(rows, "phase4d_active_eta_alpha.csv")

    out = {}
    print(f"\n{'problem':<16}{'strategy':<13}{'best eta':>9}{'alpha':>7}{'best val':>11}{'se':>9}"
          f"{'inherited':>11}{'val':>11}  within 1 se")
    for key in ALL:
        for strategy in ("sasla", "uncertainty"):
            sub = [r for r in rows if r["strategy"] == strategy]
            grid = {(e, a): cell(sub, key, eta=e, alpha=a) for e, a in itertools.product(ETAS, ALPHAS)}
            best = min(grid, key=lambda g: grid[g]["val"])
            inh = (ea[key]["eta"], ea[key]["alpha"])
            within = grid[inh]["val"] <= grid[best]["val"] + grid[best]["se"]
            out.setdefault(key, {})[strategy] = dict(
                best_eta=best[0], best_alpha=best[1], best_val=grid[best]["val"], best_se=grid[best]["se"],
                inherited_eta=inh[0], inherited_alpha=inh[1], inherited_val=grid[inh]["val"],
                inherited_is_best=bool(best == inh), within_one_se=bool(within),
                relative_excess=float(grid[inh]["val"] / grid[best]["val"] - 1.0))
            print(f"{key:<16}{strategy:<13}{best[0]:>9g}{best[1]:>7g}{grid[best]['val']:>11.5f}"
                  f"{grid[best]['se']:>9.5f}{str(inh):>11}{grid[inh]['val']:>11.5f}  {within}")
    with open(os.path.join(RESULTS, "shared_params_test.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nelapsed {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
