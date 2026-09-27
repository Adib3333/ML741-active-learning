"""Runs one training session: one problem, one approach, one seed.

Same seed = same split and same starting weights for all three approaches, so
the comparison is paired. Records errors, presentations and the best epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Problem, Split, make_split
from .metrics import evaluate, generalisation_factor, mse
from .network import FFNN
from .selectors import STRATEGIES


@dataclass
class RunResult:
    problem: str
    strategy: str
    seed: int
    epochs_run: int
    best_epoch: int
    presentations: int          # cumulative pattern presentations to the best epoch
    presentations_total: int
    train: dict = field(default_factory=dict)
    val: dict = field(default_factory=dict)
    gen: dict = field(default_factory=dict)
    rho: float = float("nan")
    subset_final: int = 0
    subset_mean: float = 0.0
    subset_frac_mean: float = 0.0
    converged: bool = False
    operator_calls: int = 0     # applications of the selection operator
    operator_evals: int = 0     # candidate patterns evaluated by the operator
    history: dict = field(default_factory=dict)

    def flat(self) -> dict:
        row = {
            "problem": self.problem, "strategy": self.strategy, "seed": self.seed,
            "epochs_run": self.epochs_run, "best_epoch": self.best_epoch,
            "presentations": self.presentations,
            "presentations_total": self.presentations_total,
            "rho": self.rho, "subset_final": self.subset_final,
            "subset_mean": self.subset_mean, "subset_frac_mean": self.subset_frac_mean,
            "converged": int(self.converged),
            "operator_calls": self.operator_calls,
            "operator_evals": self.operator_evals,
        }
        for part, d in (("train", self.train), ("val", self.val), ("gen", self.gen)):
            for k, v in d.items():
                row[f"{part}_{k}"] = v
        return row


def run(prob: Problem, strategy_key: str, seed: int, *, eta: float, alpha: float,
        lam: float, max_epochs: int = 250,
        hidden: int | None = None, strategy_kwargs: dict | None = None,
        split: Split | None = None, keep_history: bool = True,
        converge_at: float | None = None) -> RunResult:
    """Execute one independent run over a fixed epoch budget."""
    rng = np.random.default_rng(seed)
    if split is None:
        split = make_split(prob, seed)
    J = hidden if hidden is not None else prob.hidden
    kwargs = dict(strategy_kwargs or {})
    kwargs.setdefault("task", prob.task)
    strat = STRATEGIES[strategy_key](**kwargs)

    nets = [FFNN(prob.n_inputs, J, prob.n_outputs, rng) for _ in range(strat.n_models)]
    idx = strat.initial(split, rng)

    best_val, best_epoch, best_pres = np.inf, 0, 0
    presentations = 0
    subsets = []
    hist = {"epoch": [], "e_train": [], "e_val": [], "e_gen": [], "subset": [],
            "presentations": []}
    epoch = 0
    nC = split.Zc.shape[0]

    for epoch in range(1, max_epochs + 1):
        Zs, Ts = split.Zc[idx], split.Tc[idx]
        subsets.append(len(idx))
        for m in nets:
            presentations += m.train_epoch(Zs, Ts, eta, alpha, lam, rng)

        e_sub = mse(nets[0].predict(Zs), Ts)              # error on the current subset
        e_val = mse(nets[0].predict(split.Zv), split.Tv)
        rho_now = generalisation_factor(e_val, e_sub)

        if keep_history:
            e_gen = mse(nets[0].predict(split.Zg), split.Tg)
            hist["epoch"].append(epoch)
            hist["e_train"].append(e_sub)
            hist["e_val"].append(e_val)
            hist["e_gen"].append(e_gen)
            hist["subset"].append(int(len(idx)))
            hist["presentations"].append(presentations)

        if e_val < best_val - 1e-9:
            best_val, best_epoch, best_pres = e_val, epoch, presentations

        if not np.isfinite(e_val):                        # divergence guard only
            break

        idx = strat.update(nets, split, idx, rng, rho_now)

    net = nets[0]
    train_m = evaluate(net, split.Zc, split.Tc, split.yc, split.task,
                       split.n_classes, split.t_lo, split.t_hi)
    val_m = evaluate(net, split.Zv, split.Tv, split.yv, split.task,
                     split.n_classes, split.t_lo, split.t_hi)
    gen_m = evaluate(net, split.Zg, split.Tg, split.yg, split.task,
                     split.n_classes, split.t_lo, split.t_hi)

    key_metric = "error_rate" if split.task == "classification" else "mse"
    converged = bool(converge_at is None or gen_m[key_metric] <= converge_at)

    return RunResult(
        problem=prob.key, strategy=strategy_key, seed=seed,
        epochs_run=epoch, best_epoch=best_epoch,
        presentations=best_pres, presentations_total=presentations,
        train=train_m, val=val_m, gen=gen_m,
        rho=generalisation_factor(gen_m["mse"], train_m["mse"]),
        subset_final=int(len(idx)),
        subset_mean=float(np.mean(subsets)),
        subset_frac_mean=float(np.mean(subsets) / nC),
        converged=converged,
        operator_calls=int(strat.calls),
        operator_evals=int(strat.evaluations),
        history=hist if keep_history else {},
    )
