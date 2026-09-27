"""Does weight decay cause the weak pruning of SASLA at beta = 0.9?
Runs SASLA with and without the penalty on the classification problems.

  results/decay_effect.csv, decay_effect.json
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import CLASSIFICATION, load, make_split
from src.sweep import BUDGET, RESULTS, TUNE_SEED_BASE, run_many, write_csv


def main():
    lam = json.load(open(os.path.join(RESULTS, "chosen_lambda.json")))
    ea = json.load(open(os.path.join(RESULTS, "chosen_eta_alpha.json")))
    BETA = 0.9

    jobs = []
    for key in CLASSIFICATION:
        b = BUDGET[key]
        for penalty in (lam[key], 0.0):
            for s in range(b["tune_seeds"]):
                jobs.append((key, "sasla", TUNE_SEED_BASE + s, dict(
                    eta=ea[key]["eta"], alpha=ea[key]["alpha"], lam=penalty,
                    max_epochs=b["tune_epochs"], keep_history=False,
                    strategy_kwargs=dict(beta=BETA))))
    print(f"weight decay effect on pruning: {len(jobs)} runs", flush=True)
    rows = run_many(jobs)
    write_csv(rows, "decay_effect.csv")

    out = {}
    print(f"\n{'problem':<16}{'beta':>6}{'with penalty %':>16}{'without penalty %':>19}")
    for key in CLASSIFICATION:
        vals = {}
        for penalty, name in ((lam[key], "with"), (0.0, "without")):
            sel = [r for r in rows if r["problem"] == key and abs(r["lam"] - penalty) < 1e-15]
            vals[name] = 100.0 * float(np.mean([r["subset_final"] / make_split(load(key), int(r["seed"])).Zc.shape[0]
                                                for r in sel]))
        out[key] = vals
        print(f"{key:<16}{BETA:>6.1f}{vals['with']:>16.1f}{vals['without']:>19.1f}")
    with open(os.path.join(RESULTS, "decay_effect.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {os.path.join(RESULTS, 'decay_effect.json')}")


if __name__ == "__main__":
    main()
