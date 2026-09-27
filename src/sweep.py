"""Shared bits for the phases: epoch budgets, seeds, parallel runs and caching.
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .data import load, make_split
from .experiment import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

N_WORKERS = max(1, (os.cpu_count() or 2))

# Epoch budgets. The larger problems cost more per epoch, so the tuning
# phases use a reduced budget while the final comparison uses the full one.
BUDGET = {
    "iris":          dict(tune_epochs=200, final_epochs=300, tune_seeds=5),
    "breast_cancer": dict(tune_epochs=150, final_epochs=250, tune_seeds=5),
    "optdigits":     dict(tune_epochs=60,  final_epochs=150, tune_seeds=3),
    "auto_mpg":      dict(tune_epochs=200, final_epochs=300, tune_seeds=5),
    "diabetes":      dict(tune_epochs=200, final_epochs=300, tune_seeds=5),
    "concrete":      dict(tune_epochs=200, final_epochs=300, tune_seeds=5),
}

# Baseline control parameter values, used until each phase replaces them.
DEFAULTS = dict(eta=0.1, alpha=0.9, lam=1e-4)

TUNE_SEED_BASE = 1000     # seeds reserved for tuning
FINAL_SEED_BASE = 1       # seeds reserved for the final comparison


def _job(args):
    key, strategy, seed, kw = args
    prob = load(key)
    r = run(prob, strategy, seed, **kw)
    row = r.flat()
    row.update({k: v for k, v in kw.items()
                if k in ("eta", "alpha", "lam", "hidden")})
    sk = kw.get("strategy_kwargs") or {}
    for k, v in sk.items():
        if k != "task":
            row[f"p_{k}"] = v
    return row


def run_many(jobs, workers: int | None = None) -> list[dict]:
    """Execute a list of (problem_key, strategy, seed, kwargs) jobs."""
    workers = workers or N_WORKERS
    if workers == 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs, chunksize=1))


CACHE = os.path.join(RESULTS, "cache")
os.makedirs(CACHE, exist_ok=True)


def run_cached(tag: str, jobs) -> list[dict]:
    """Execute a block of jobs, or reload the block from the cache.

    Long phases are executed one block at a time and every finished block is
    written to results/cache. A phase that is interrupted therefore resumes at
    the first unfinished block instead of repeating the whole phase.
    """
    path = os.path.join(CACHE, tag + ".csv")
    if os.path.exists(path):
        rows = read_csv_path(path)
        print(f"  cached block {tag}: {len(rows)} runs", flush=True)
        return rows
    rows = run_many(jobs)
    write_csv_path(rows, path)
    print(f"  block {tag}: {len(rows)} runs", flush=True)
    return rows


def write_csv_path(rows: list[dict], path: str) -> str:
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def write_csv(rows: list[dict], name: str) -> str:
    path = os.path.join(RESULTS, name)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")
    return path


def read_csv(name: str) -> list[dict]:
    return read_csv_path(os.path.join(RESULTS, name))


def read_csv_path(path: str) -> list[dict]:
    with open(path) as fh:
        out = []
        for row in csv.DictReader(fh):
            conv = {}
            for k, v in row.items():
                if v == "":
                    conv[k] = None
                    continue
                try:
                    conv[k] = int(v)
                except ValueError:
                    try:
                        conv[k] = float(v)
                    except ValueError:
                        conv[k] = v
            out.append(conv)
    return out


def key_metric(task: str) -> str:
    """The single measure used for selection and for statistical ranking."""
    return "gen_error_rate" if task == "classification" else "gen_mse"


def agg(rows, field_) -> tuple[float, float]:
    vals = np.array([r[field_] for r in rows], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
