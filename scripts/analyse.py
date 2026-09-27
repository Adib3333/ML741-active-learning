"""Statistics on the final runs.

  results/analysis.json, analysis.txt
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import ALL, load
from src.selectors import ORDER
from src.stats import (describe, effect_size_a12, friedman, nemenyi_cd,
                       pairwise_wilcoxon)
from src.sweep import RESULTS, read_csv

LABEL = {"passive": "Passive", "sasla": "SASLA", "uncertainty": "Uncertainty"}


def by(rows, **m):
    return [r for r in rows if all(r[k] == v for k, v in m.items())]


def vec(rows, field_):
    return np.array([r[field_] for r in rows], float)


def main():
    rows = read_csv("phase5_runs.csv")
    out = {"per_problem": {}, "across": {}}

    # --- per problem summary
    print("=" * 96)
    print("TABLE: generalisation performance, mean +/- standard deviation over 30 independent runs")
    print("=" * 96)
    for key in ALL:
        prob = load(key)
        metric = "gen_error_rate" if prob.task == "classification" else "gen_mse"
        unit = "gen_macro_f1" if prob.task == "classification" else "gen_rmse"
        print(f"\n{prob.name}  ({prob.task}, J={prob.hidden})")
        hdr = (f"  {'strategy':<13}{'train mse':>18}{'gen mse':>18}"
               f"{'rho':>14}{'subset %':>12}{'presentations':>16}")
        print(hdr)
        entry = {}
        for s in ORDER:
            rs = by(rows, problem=key, strategy=s)
            tr, ge = describe(vec(rs, "train_mse")), describe(vec(rs, "gen_mse"))
            rho = describe(vec(rs, "rho"))
            frac = describe(100.0 * vec(rs, "subset_frac_mean"))
            pres = describe(vec(rs, "presentations_total"))
            sec = describe(vec(rs, unit))
            key_m = describe(vec(rs, metric))
            entry[s] = {"train_mse": tr, "gen_mse": ge, "rho": rho,
                        "subset_pct": frac, "presentations": pres,
                        "secondary": sec, "key": key_m, "n": len(rs),
                        "val_mse": describe(vec(rs, "val_mse")),
                        "best_epoch": describe(vec(rs, "best_epoch")),
                        "pres_best": describe(vec(rs, "presentations")),
                        "op_evals": describe(vec(rs, "operator_evals")),
                        "op_calls": describe(vec(rs, "operator_calls")),
                        "subset_final": describe(vec(rs, "subset_final"))}
            if prob.task == "classification":
                entry[s]["train_error_rate"] = describe(vec(rs, "train_error_rate"))
            print(f"  {LABEL[s]:<13}{tr['mean']:>10.5f}+-{tr['std']:<7.5f}"
                  f"{ge['mean']:>10.5f}+-{ge['std']:<7.5f}"
                  f"{rho['mean']:>8.2f}+-{rho['std']:<5.2f}"
                  f"{frac['mean']:>9.1f}  {pres['mean']:>14,.0f}")
        sec_name = "macro F1" if prob.task == "classification" else "RMSE (original units)"
        print(f"  secondary measure, {sec_name}: " + "  ".join(
            f"{LABEL[s]} {entry[s]['secondary']['mean']:.4f}+-{entry[s]['secondary']['std']:.4f}"
            for s in ORDER))

        # --- paired tests within the problem
        samples = {LABEL[s]: vec(by(rows, problem=key, strategy=s), metric) for s in ORDER}
        res = pairwise_wilcoxon(samples)
        print("  paired Wilcoxon signed-rank, Holm corrected, on "
              + ("generalisation error rate" if prob.task == "classification" else "generalisation mse"))
        for name, d in res.items():
            a, b = name.split(" vs ")
            a12 = effect_size_a12(samples[a], samples[b])
            d["a12"] = a12
            verdict = "significant" if d["reject"] else "not significant"
            print(f"    {name:<28} p={d['p']:.4g}  p_holm={d['p_holm']:.4g}  "
                  f"{verdict:<16} better={d['better']:<12} A12={a12:.3f}")
        entry["_tests"] = {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                               for kk, vv in v.items()} for k, v in res.items()}

        # cost comparison
        cost = {LABEL[s]: vec(by(rows, problem=key, strategy=s), "presentations_total") for s in ORDER}
        base = cost["Passive"].mean()
        print("  pattern presentations relative to the control: " + "  ".join(
            f"{s} {100.0 * cost[s].mean() / base:.1f}%" for s in ["SASLA", "Uncertainty"]))
        entry["_cost_ratio"] = {s: float(cost[s].mean() / base) for s in ["SASLA", "Uncertainty"]}
        out["per_problem"][key] = entry

    # --- across problems
    print("\n" + "=" * 96)
    print("Friedman test across the six problems")
    print("=" * 96)
    for what, field_, lower_better in [
            ("solution quality", "key", True),
            ("computational cost", "presentations", True)]:
        M = []
        for key in ALL:
            if what == "solution quality":
                row = [out["per_problem"][key][s]["key"]["mean"] for s in ORDER]
            else:
                row = [out["per_problem"][key][s]["presentations"]["mean"] for s in ORDER]
            M.append(row)
        M = np.array(M, float)
        fr = friedman(M)
        cd = nemenyi_cd(fr["k"], fr["N"])
        print(f"\n{what}:")
        print("  average ranks: " + "  ".join(
            f"{LABEL[s]}={r:.3f}" for s, r in zip(ORDER, fr["avg_ranks"])))
        print(f"  chi2_F = {fr['chi2']:.4f}  p = {fr['p_chi2']:.4g}   "
              f"Iman-Davenport F({fr['df1']},{fr['df2']}) = {fr['F']:.3f}  p = {fr['p_F']:.4g}")
        print(f"  Nemenyi critical difference = {cd:.4f}")
        for i in range(len(ORDER)):
            for j in range(i + 1, len(ORDER)):
                d = abs(fr["avg_ranks"][i] - fr["avg_ranks"][j])
                print(f"    {LABEL[ORDER[i]]:<12} vs {LABEL[ORDER[j]]:<12} "
                      f"rank difference = {d:.3f}  "
                      f"{'SIGNIFICANT' if d >= cd else 'not significant'}")
        out["across"][what] = {
            "avg_ranks": {s: float(r) for s, r in zip(ORDER, fr["avg_ranks"])},
            "chi2": fr["chi2"], "p_chi2": fr["p_chi2"], "F": fr["F"], "p_F": fr["p_F"],
            "df1": fr["df1"], "df2": fr["df2"], "cd": cd,
            "ranks_per_problem": {k: dict(zip(ORDER, fr["ranks"][i].tolist()))
                                  for i, k in enumerate(ALL)},
            "matrix": {k: dict(zip(ORDER, M[i].tolist())) for i, k in enumerate(ALL)},
        }

    with open(os.path.join(RESULTS, "analysis.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(RESULTS, 'analysis.json')}")


if __name__ == "__main__":
    main()
