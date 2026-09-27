"""Reruns every phase for one problem only, keeping the saved results of the
other problems. Used after the Fuel Consumption imputation fix.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from src.sweep import read_csv
import src.sweep as sweep

KEY = sys.argv[1] if len(sys.argv) > 1 else "auto_mpg"


def filtered(module, csv_name):
    """Replace module.run_many by a version that runs only the jobs of KEY
    and returns them together with the saved rows of the other problems."""
    def run(jobs, *a, **k):
        mine = [j for j in jobs if j[0] == KEY]
        print(f"  rerunning {len(mine)} of {len(jobs)} runs ({KEY})", flush=True)
        new = sweep.run_many(mine, *a, **k)
        old = [r for r in read_csv(csv_name) if r["problem"] != KEY]
        return old + new
    module.run_many = run


if __name__ == "__main__":     # needed on Windows, where worker processes re-import this file
    t0 = time.perf_counter()
    sys.argv = [sys.argv[0]]
    p2 = importlib.import_module("scripts.phase2_architecture")
    filtered(p2, "phase2_architecture.csv"); p2.main()
    p3 = importlib.import_module("scripts.phase3_lambda")
    filtered(p3, "phase3_lambda.csv"); p3.main()
    p4 = importlib.import_module("scripts.phase4_params")
    lam = p4.load_json("chosen_lambda.json")
    filtered(p4, "phase4a_eta_alpha.csv"); ea = p4.stage_a(lam)
    filtered(p4, "phase4b_beta.csv"); p4.stage_b(lam, ea)
    p4.stage_c(lam, ea)          # block cache: only the blocks of KEY are missing
    p5 = importlib.import_module("scripts.phase5_compare")
    p5.main()                    # block cache: only the blocks of KEY are missing
    print(f"\nrerun of {KEY} finished in {time.perf_counter()-t0:.0f}s", flush=True)
