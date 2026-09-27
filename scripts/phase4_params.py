"""Phase 4: tunes the control parameters.

  4a  learning rate and momentum, on passive learning
  4b  beta for SASLA
  4c  seed fraction, growth fraction, threshold and committee size for
      uncertainty sampling

The active approaches use the one standard error rule: cheapest setting within
one standard error of the best validation MSE.

  results/phase4a_eta_alpha.csv, phase4b_beta.csv, phase4c_uncertainty.csv
  results/chosen_eta_alpha.json, chosen_beta.json, chosen_uncertainty.json
"""

import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALL, load
from src.sweep import BUDGET, RESULTS, TUNE_SEED_BASE, read_csv, run_cached, run_many, write_csv

ETAS = [0.01, 0.05, 0.1, 0.3]
ALPHAS = [0.0, 0.5, 0.9]
BETAS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS_FRAC = [0.05, 0.10, 0.20]
GROWTHS = [0.02, 0.05, 0.10]
THETAS = [1.05, 1.15, 1.30]
COMMITTEES = [3, 5]


def load_json(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def cell(rows, key, **match):
    sel = [r for r in rows if r["problem"] == key and all(
        (abs(r[k] - v) < 1e-12) if isinstance(v, float) else r[k] == v
        for k, v in match.items())]
    v = np.array([r["val_mse"] for r in sel], float)
    pres = np.array([r["presentations_total"] for r in sel], float)
    frac = np.array([r["subset_frac_mean"] for r in sel], float)
    return dict(val=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v)),
                pres=pres.mean(), frac=100.0 * frac.mean(), n=len(v))


def one_se(cands):
    """cands: list of (label, stats). Most parsimonious within 1 SE of the best."""
    best = min(cands, key=lambda c: c[1]["val"])
    tol = best[1]["val"] + best[1]["se"]
    within = [c for c in cands if c[1]["val"] <= tol]
    return min(within, key=lambda c: c[1]["pres"]), best, tol


def stage_a(lam, rows=None):
    if rows is None:
        jobs = []
        for key in ALL:
            b = BUDGET[key]
            for eta, alpha in itertools.product(ETAS, ALPHAS):
                for s in range(b["tune_seeds"]):
                    jobs.append((key, "passive", TUNE_SEED_BASE + s, dict(
                        eta=eta, alpha=alpha, lam=lam[key],
                        max_epochs=b["tune_epochs"], keep_history=False)))
        print(f"Phase 4a: learning rate and momentum, {len(jobs)} runs", flush=True)
        rows = run_many(jobs)
        write_csv(rows, "phase4a_eta_alpha.csv")
    else:
        print(f"Phase 4a: learning rate and momentum, {len(rows)} runs (summary of saved results)")
    chosen = {}
    print(f"\n{'problem':<16}{'eta':>7}{'alpha':>7}{'val mse':>12}")
    for key in ALL:
        grid = [((e, a), cell(rows, key, eta=e, alpha=a)) for e, a in itertools.product(ETAS, ALPHAS)]
        best = min(grid, key=lambda g: g[1]["val"])
        chosen[key] = {"eta": best[0][0], "alpha": best[0][1]}
        for (e, a), st in grid:
            mark = "  <- selected" if (e, a) == best[0] else ""
            print(f"{key if (e, a) == (ETAS[0], ALPHAS[0]) else '':<16}{e:>7.2f}{a:>7.1f}{st['val']:>12.5f}{mark}")
    with open(os.path.join(RESULTS, "chosen_eta_alpha.json"), "w") as fh:
        json.dump(chosen, fh, indent=2)
    return chosen


def stage_b(lam, ea, rows=None):
    if rows is None:
        jobs = []
        for key in ALL:
            b = BUDGET[key]
            for beta in BETAS:
                for s in range(b["tune_seeds"]):
                    jobs.append((key, "sasla", TUNE_SEED_BASE + s, dict(
                        eta=ea[key]["eta"], alpha=ea[key]["alpha"], lam=lam[key],
                        max_epochs=b["tune_epochs"], keep_history=False,
                        strategy_kwargs=dict(beta=beta))))
        print(f"\nPhase 4b: SASLA selection constant, {len(jobs)} runs", flush=True)
        rows = run_many(jobs)
        write_csv(rows, "phase4b_beta.csv")
    else:
        print(f"\nPhase 4b: SASLA selection constant, {len(rows)} runs (summary of saved results)")
    chosen = {}
    print(f"\n{'problem':<16}{'beta':>6}{'val mse':>11}{'se':>9}{'subset %':>10}{'presentations':>15}")
    for key in ALL:
        cands = [(b_, cell(rows, key, p_beta=b_)) for b_ in BETAS]
        pick, best, tol = one_se(cands)
        chosen[key] = pick[0]
        for b_, st in cands:
            tag = "  <- selected" if b_ == pick[0] else ("  (lowest error)" if b_ == best[0] else ("  within 1 se" if st["val"] <= tol else ""))
            print(f"{key if b_ == BETAS[0] else '':<16}{b_:>6.1f}{st['val']:>11.5f}{st['se']:>9.5f}"
                  f"{st['frac']:>10.1f}{st['pres']:>15,.0f}{tag}")
    with open(os.path.join(RESULTS, "chosen_beta.json"), "w") as fh:
        json.dump(chosen, fh, indent=2)
    return chosen


def stage_c(lam, ea, rows=None):
    if rows is None:
        rows = []
        n_total = 0
        blocks = []
        for key in ALL:
            b = BUDGET[key]
            comms = COMMITTEES if load(key).task == "regression" else [1]
            # one block per problem, committee size and tuning seed, so that an
            # interrupted phase loses at most one small block of runs
            for c in comms:
                for s_ in range(b["tune_seeds"]):
                    jobs = []
                    for sf, g, th in itertools.product(SEEDS_FRAC, GROWTHS, THETAS):
                        sk = dict(seed_frac=sf, growth=g, rho_max=th)
                        if c > 1:
                            sk["committee"] = c
                        jobs.append((key, "uncertainty", TUNE_SEED_BASE + s_, dict(
                            eta=ea[key]["eta"], alpha=ea[key]["alpha"], lam=lam[key],
                            max_epochs=b["tune_epochs"], keep_history=False,
                            strategy_kwargs=sk)))
                    blocks.append((f"phase4c_{key}_c{c}_s{s_}", jobs))
                    n_total += len(jobs)
        print(f"\nPhase 4c: uncertainty sampling parameters, {n_total} runs", flush=True)
        for tag, jobs in blocks:
            rows += run_cached(tag, jobs)
        for r in rows:
            r.setdefault("p_committee", 1)
            if r.get("p_committee") is None:
                r["p_committee"] = 1
        write_csv(rows, "phase4c_uncertainty.csv")
    else:
        print(f"\nPhase 4c: uncertainty sampling parameters, {len(rows)} runs (summary of saved results)")
    chosen = {}
    print(f"\n{'problem':<16}{'seed':>6}{'growth':>7}{'theta':>7}{'C':>3}{'val mse':>11}{'se':>9}{'subset %':>10}{'presentations':>15}")
    for key in ALL:
        comms = COMMITTEES if load(key).task == "regression" else [1]
        cands = []
        for sf, g, th, c in itertools.product(SEEDS_FRAC, GROWTHS, THETAS, comms):
            st = cell(rows, key, p_seed_frac=sf, p_growth=g, p_rho_max=th, p_committee=c)
            cands.append(((sf, g, th, c), st))
        pick, best, tol = one_se(cands)
        sf, g, th, c = pick[0]
        chosen[key] = {"seed_frac": sf, "growth": g, "rho_max": th}
        if c > 1:
            chosen[key]["committee"] = c
        for lbl, st in cands:
            tag = "  <- selected" if lbl == pick[0] else ("  (lowest error)" if lbl == best[0] else "")
            if tag:
                print(f"{key:<16}{lbl[0]:>6.2f}{lbl[1]:>7.2f}{lbl[2]:>7.2f}{lbl[3]:>3}{st['val']:>11.5f}"
                      f"{st['se']:>9.5f}{st['frac']:>10.1f}{st['pres']:>15,.0f}{tag}")
        n_within = sum(1 for _, st in cands if st["val"] <= tol)
        print(f"{'':<16}{n_within} of {len(cands)} settings within one standard error")
    with open(os.path.join(RESULTS, "chosen_uncertainty.json"), "w") as fh:
        json.dump(chosen, fh, indent=2)
    return chosen


def main():
    """Run the selected stages (default a, b and c). With --summary, reprint
    the selection tables of the selected stages from the saved CSV files."""
    t0 = time.perf_counter()
    lam = load_json("chosen_lambda.json")
    summary = "--summary" in sys.argv
    stages = [a for a in sys.argv[1:] if not a.startswith("--")] or ["a", "b", "c"]
    saved = (lambda name: read_csv(name)) if summary else (lambda name: None)
    ea = stage_a(lam, saved("phase4a_eta_alpha.csv")) if "a" in stages else load_json("chosen_eta_alpha.json")
    if "b" in stages:
        stage_b(lam, ea, saved("phase4b_beta.csv"))
    if "c" in stages:
        stage_c(lam, ea, saved("phase4c_uncertainty.csv"))
    print(f"\nelapsed {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
