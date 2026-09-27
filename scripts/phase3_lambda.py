"""Phase 3: picks the weight decay coefficient lambda per problem.

  results/phase3_lambda.csv, chosen_lambda.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALL
from src.sweep import BUDGET, DEFAULTS, RESULTS, TUNE_SEED_BASE, read_csv, run_many, write_csv

LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def report(rows):
    chosen = {}
    print(f"\n{'problem':<16}{'lambda':>10}{'val mse':>12}{'std':>10}{'rho':>8}")
    for key in ALL:
        stats = []
        for lam in LAMBDAS:
            sel = [r for r in rows if r["problem"] == key and abs(r["lam"] - lam) < 1e-15]
            v = np.array([r["val_mse"] for r in sel])
            rho = float(np.mean([r["val_mse"] / r["train_mse"] for r in sel]))
            stats.append((lam, v.mean(), v.std(ddof=1), rho))
        pick = min([s for s in stats if s[0] > 0.0], key=lambda s: s[1])
        chosen[key] = pick[0]
        for lam, m, sd, rho in stats:
            mark = "  <- selected" if lam == pick[0] else ("  (unregularised reference)" if lam == 0.0 else "")
            print(f"{key if lam == LAMBDAS[0] else '':<16}{lam:>10.0e}{m:>12.5f}{sd:>10.5f}{rho:>8.2f}{mark}")

    with open(os.path.join(RESULTS, "chosen_lambda.json"), "w") as fh:
        json.dump(chosen, fh, indent=2)
    print("\nselected:", chosen)
    return chosen


def main():
    """Run the sweep, or with --summary reprint the table from the saved CSV."""
    t0 = time.perf_counter()
    if "--summary" in sys.argv:
        rows = read_csv("phase3_lambda.csv")
        print(f"Phase 3: penalty coefficient sweep, {len(rows)} runs (summary of saved results)")
    else:
        jobs = []
        for key in ALL:
            b = BUDGET[key]
            for lam in LAMBDAS:
                for s in range(b["tune_seeds"]):
                    jobs.append((key, "passive", TUNE_SEED_BASE + s, dict(
                        eta=DEFAULTS["eta"], alpha=DEFAULTS["alpha"], lam=lam,
                        max_epochs=b["tune_epochs"], keep_history=False)))
        print(f"Phase 3: penalty coefficient sweep, {len(jobs)} runs")
        rows = run_many(jobs)
        write_csv(rows, "phase3_lambda.csv")
    report(rows)
    print(f"elapsed {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
