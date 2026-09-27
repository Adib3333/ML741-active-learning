"""Phase 2: is the hidden layer really an overestimate?

Trains unregularised passive networks at several sizes. The sufficient size is
the smallest one within one standard error of the best. The adopted size must
be bigger.

  results/phase2_architecture.csv, sufficient_hidden.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALL, load
from src.sweep import BUDGET, DEFAULTS, RESULTS, TUNE_SEED_BASE, read_csv, run_many, write_csv

SIZES = {
    "iris":          [1, 2, 3, 5, 8, 12, 20],
    "breast_cancer": [1, 2, 4, 8, 16, 28, 40],
    "optdigits":     [4, 8, 16, 24, 36, 48, 60, 80, 100],
    "auto_mpg":      [1, 2, 4, 8, 14, 22, 30],
    "diabetes":      [1, 2, 4, 8, 14, 22, 30],
    "concrete":      [2, 4, 8, 16, 28, 40, 50],
}


def summarise(rows, key):
    out = []
    for J in SIZES[key]:
        sel = [r for r in rows if r["problem"] == key and r["hidden"] == J]
        v = np.array([r["val_mse"] for r in sel])
        t = np.array([r["train_mse"] for r in sel])
        out.append(dict(J=J, train=t.mean(), val=v.mean(),
                        se=v.std(ddof=1) / np.sqrt(len(v)),
                        rho=float(np.mean(v / t))))
    return out


def report(rows):
    sufficient = {}
    print(f"\n{'problem':<16}{'J':>5}{'train mse':>12}{'val mse':>12}{'se':>10}{'rho':>7}")
    for key in ALL:
        stats = summarise(rows, key)
        best = min(stats, key=lambda s: s["val"])
        suff = min((s for s in stats if s["val"] <= best["val"] + best["se"]),
                   key=lambda s: s["J"])
        sufficient[key] = suff["J"]
        adopted = load(key).hidden
        for s in stats:
            tag = ""
            if s["J"] == suff["J"]:
                tag += "  sufficient"
            if s["J"] == best["J"]:
                tag += "  lowest"
            if s["J"] == adopted:
                tag += "  <- adopted"
            print(f"{key if s['J'] == SIZES[key][0] else '':<16}{s['J']:>5}{s['train']:>12.5f}"
                  f"{s['val']:>12.5f}{s['se']:>10.5f}{s['rho']:>7.2f}{tag}")
        verdict = "OVERESTIMATE" if adopted > suff["J"] else "NOT AN OVERESTIMATE"
        print(f"{'':<16}sufficient J = {suff['J']}, adopted J = {adopted}: {verdict}")

    with open(os.path.join(RESULTS, "sufficient_hidden.json"), "w") as fh:
        json.dump(sufficient, fh, indent=2)
    return sufficient


def main():
    """Run the probe, or with --summary reprint the table from the saved CSV."""
    t0 = time.perf_counter()
    if "--summary" in sys.argv:
        rows = read_csv("phase2_architecture.csv")
        print(f"Phase 2: architecture probe, {len(rows)} runs (summary of saved results)")
    else:
        jobs = []
        for key in ALL:
            b = BUDGET[key]
            for J in SIZES[key]:
                for s in range(b["tune_seeds"]):
                    jobs.append((key, "passive", TUNE_SEED_BASE + s, dict(
                        eta=DEFAULTS["eta"], alpha=DEFAULTS["alpha"], lam=0.0,
                        hidden=J, max_epochs=b["tune_epochs"], keep_history=False)))
        print(f"Phase 2: architecture probe, {len(jobs)} runs")
        rows = run_many(jobs)
        write_csv(rows, "phase2_architecture.csv")
    report(rows)
    print(f"\nelapsed {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
