"""Phase 5: the final comparison. 30 paired runs per approach and problem.

  results/phase5_runs.csv, phase5_histories.npz
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALL, load, make_split
from src.selectors import ORDER
from src.sweep import (BUDGET, CACHE, FINAL_SEED_BASE, RESULTS, read_csv_path,
                       write_csv, write_csv_path)

N_RUNS = 30
HISTORY_SEEDS = 5


def load_json(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def build_blocks():
    """One block of jobs per problem and strategy, so that an interrupted run
    resumes at the first unfinished block."""
    lam = load_json("chosen_lambda.json")
    ea = load_json("chosen_eta_alpha.json")
    beta = load_json("chosen_beta.json")
    unc = load_json("chosen_uncertainty.json")

    blocks = []
    for key in ALL:
        b = BUDGET[key]
        for strat in ORDER:
            if strat == "sasla":
                sk = dict(beta=beta[key])
            elif strat == "uncertainty":
                sk = dict(unc[key])
            else:
                sk = {}
            jobs = []
            for i in range(N_RUNS):
                seed = FINAL_SEED_BASE + i
                jobs.append((key, strat, seed, dict(
                    eta=ea[key]["eta"], alpha=ea[key]["alpha"], lam=lam[key],
                    max_epochs=b["final_epochs"],
                    keep_history=(i < HISTORY_SEEDS),
                    strategy_kwargs=sk)))
            blocks.append((f"{key}_{strat}", jobs))
    return blocks


def _run_one(j):
    """Module level so that it can be dispatched to a worker process."""
    from src.experiment import run as _run
    key, strat, seed, kw = j
    r = _run(load(key), strat, seed, **kw)
    row = r.flat()
    row.update({k: v for k, v in kw.items() if k in ("eta", "alpha", "lam")})
    for k, v in (kw.get("strategy_kwargs") or {}).items():
        if k != "task":
            row[f"p_{k}"] = v
    return row, ((f"{key}|{strat}|{seed}", r.history) if r.history else None)


def run_block(tag, jobs):
    """Execute one block, or reload the block from the cache."""
    csv_path = os.path.join(CACHE, f"phase5_{tag}.csv")
    npz_path = os.path.join(CACHE, f"phase5_{tag}.npz")
    if os.path.exists(csv_path) and os.path.exists(npz_path):
        npz = np.load(npz_path)
        print(f"  cached block {tag}", flush=True)
        return read_csv_path(csv_path), {k: npz[k] for k in npz.files}

    from concurrent.futures import ProcessPoolExecutor
    rows, flat = [], {}
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        for row, hist in ex.map(_run_one, jobs, chunksize=1):
            rows.append(row)
            if hist:
                for field_, vals in hist[1].items():
                    flat[f"{hist[0]}|{field_}"] = np.asarray(vals)
    write_csv_path(rows, csv_path)
    np.savez_compressed(npz_path, **flat)
    return rows, flat


def main():
    blocks = build_blocks()
    n = sum(len(j) for _, j in blocks)
    print(f"Phase 5: comparison, {n} runs "
          f"({N_RUNS} seeds x {len(ORDER)} strategies x {len(ALL)} problems)")
    t0 = time.perf_counter()

    rows, flat = [], {}
    for tag, jobs in blocks:
        r, f = run_block(tag, jobs)
        rows += r
        flat.update(f)
        print(f"  {len(rows)}/{n} runs  [{time.perf_counter()-t0:.0f}s]", flush=True)

    write_csv(rows, "phase5_runs.csv")
    np.savez_compressed(os.path.join(RESULTS, "phase5_histories.npz"), **flat)
    print(f"  wrote phase5_histories.npz  ({len(flat)} series)")
    print(f"\nelapsed {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
