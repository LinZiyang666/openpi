"""Task-conditional dispatch headroom (exploratory, development set).

A dispatcher that may use a different arm per task reallocates the compute
budget to the tasks where a millisecond buys the most success. Every arm was
rolled out on the same 300 A' episodes, so the task-conditional value can be
computed from the logged sufficient statistics with the same budget-mixture
semantics (ratio-of-sums cost, episode-level mixtures, Lagrangian sweep).

Two estimates per family roster:
  * in-sample: mixture chosen and evaluated on all 30 inits per task (winner's
    curse included, same optimism for every roster);
  * cross-fitted: mixture chosen on the even-index inits, evaluated on the odd
    ones and vice versa (honest out-of-sample value; the realised cost on the
    evaluation half may miss the budget by a few percent -- reported).

Also runs the same two estimates for the ordinary single-family mixture (one
policy for every task), so the gain is measured against the estimand the
confirmation plan uses. Nothing here feeds a verdict.

Usage: see ``oracle_headroom.py`` for the --source syntax.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exp.dispatch_surface.analysis.oracle_headroom import family_of, load_source  # noqa: E402
from exp.dispatch_surface.run_precheck import NUM_TASKS, official_test_inits  # noqa: E402

LAMBDAS = np.concatenate([[0.0], np.logspace(-6, 1, 600)])


def _stats(cells: dict, subset) -> tuple[float, float, float]:
    t = float(np.mean([cells[c][0] for c in subset]))
    d = float(np.mean([cells[c][1] for c in subset]))
    s = float(np.mean([1.0 if cells[c][2] else 0.0 for c in subset]))
    return t, d, s


def solve(arms, mode, st, B):
    """Fractional Lagrangian optimum on ``st[(arm, task)] = (t, d, s)``.

    Returns [(weight, {task: arm}), ...] -- one assignment, or a tight two-point
    mixture of a feasible and an infeasible assignment."""
    sols = []
    for lam in LAMBDAS:
        if mode == "single":
            obj = [sum(st[(a, t)][2] - lam * (st[(a, t)][0] - B * st[(a, t)][1]) for t in range(NUM_TASKS)) for a in arms]
            a = arms[int(np.argmax(obj))]
            pick = {t: a for t in range(NUM_TASKS)}
        else:
            pick = {t: arms[int(np.argmax([st[(a, t)][2] - lam * (st[(a, t)][0] - B * st[(a, t)][1]) for a in arms]))]
                    for t in range(NUM_TASKS)}
        slack = sum(st[(pick[t], t)][0] - B * st[(pick[t], t)][1] for t in range(NUM_TASKS))
        sr = sum(st[(pick[t], t)][2] for t in range(NUM_TASKS)) / NUM_TASKS
        sols.append((slack, sr, pick))
    feas = [x for x in sols if x[0] <= 1e-9]
    infeas = [x for x in sols if x[0] > 1e-9]
    if not feas:
        return None
    best_f = max(feas, key=lambda x: x[1])
    best, best_v = [(1.0, best_f[2])], best_f[1]
    for f in feas:
        for g in infeas:
            w = g[0] / (g[0] - f[0])
            v = w * f[1] + (1.0 - w) * g[1]
            if v > best_v + 1e-12:
                best_v, best = v, [(w, f[2]), (1.0 - w, g[2])]
    return best


def evaluate(mix, st):
    sr = sum(w * sum(st[(p[t], t)][2] for t in range(NUM_TASKS)) / NUM_TASKS for w, p in mix)
    tt = sum(w * sum(st[(p[t], t)][0] for t in range(NUM_TASKS)) for w, p in mix)
    dd = sum(w * sum(st[(p[t], t)][1] for t in range(NUM_TASKS)) for w, p in mix)
    return sr, tt / dd


def run_roster(per_arm: dict, arms: list[str], budgets: np.ndarray, grid) -> dict:
    out = {}
    by_task_all = {t: [c for c in grid if c[0] == t] for t in range(NUM_TASKS)}
    st_all = {(a, t): _stats(per_arm[a], by_task_all[t]) for a in arms for t in range(NUM_TASKS)}
    for mode in ("single", "task"):
        ins = [evaluate(solve(arms, mode, st_all, B), st_all)[0] for B in budgets]
        srs, costs = [], []
        for fold in (0, 1):
            sel = {t: [c for c in grid if c[0] == t and c[1] % 2 == fold] for t in range(NUM_TASKS)}
            ev = {t: [c for c in grid if c[0] == t and c[1] % 2 != fold] for t in range(NUM_TASKS)}
            st_a = {(a, t): _stats(per_arm[a], sel[t]) for a in arms for t in range(NUM_TASKS)}
            st_b = {(a, t): _stats(per_arm[a], ev[t]) for a in arms for t in range(NUM_TASKS)}
            fs, fc = [], []
            for B in budgets:
                mix = solve(arms, mode, st_a, B)
                if mix is None:
                    fs.append(np.nan)
                    fc.append(np.nan)
                    continue
                sr, c = evaluate(mix, st_b)
                fs.append(sr)
                fc.append(c)
            srs.append(fs)
            costs.append(fc)
        out[mode] = {"in_sample": np.asarray(ins), "cross_fit": np.nanmean(srs, axis=0),
                     "cross_fit_cost_ratio": np.nanmean(costs, axis=0) / budgets}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", required=True, help="tag:arm_matrix:journal:per_step")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--outcome-design", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    args = ap.parse_args()
    officials = official_test_inits(args.split_manifest, args.trials)
    per_arm = {}
    for spec in args.source:
        _tag, data = load_source(spec, officials, args.trials)
        per_arm.update(data)
    grid = sorted(next(iter(per_arm.values())))
    design = json.loads(pathlib.Path(args.outcome_design).read_text())
    b_l, b_h = design["interval"]
    budgets = np.linspace(b_l, b_h, 25)
    rosters = {
        "threshold": sorted(a for a in per_arm if family_of(a) == "threshold"),
        "surface (sv + s0)": sorted(a for a in per_arm if family_of(a) in ("sv", "s0")),
        "all arms": sorted(per_arm),
    }
    results = {name: run_roster(per_arm, arms, budgets, grid) for name, arms in rosters.items()}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(budgets, res["single"]["in_sample"], "-", color="#888888", lw=1.6, label="single policy for all tasks — in-sample")
        ax.plot(budgets, res["single"]["cross_fit"], "--", color="#888888", lw=1.6, label="single policy — cross-fitted")
        ax.plot(budgets, res["task"]["in_sample"], "-", color="#c0392b", lw=2.2, label="task-conditional — in-sample")
        ax.plot(budgets, res["task"]["cross_fit"], "--", color="#c0392b", lw=2.2, label="task-conditional — cross-fitted")
        ax.set_title(f"{name} ({len(rosters[name])} arms)")
        ax.set_xlabel("compute budget B (ms per decision)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("success rate (A′ development set)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Task-conditional budget allocation vs one policy for every task (budget-mixture semantics; exploratory)")
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=170)
    fig.savefig(str(pathlib.Path(args.out_fig).with_suffix(".pdf")))

    summary = {}
    for name, res in results.items():
        summary[name] = {
            "n_arms": len(rosters[name]),
            "single_in_sample": float(np.nanmean(res["single"]["in_sample"])),
            "single_cross_fit": float(np.nanmean(res["single"]["cross_fit"])),
            "single_cross_fit_cost_ratio": float(np.nanmean(res["single"]["cross_fit_cost_ratio"])),
            "task_in_sample": float(np.nanmean(res["task"]["in_sample"])),
            "task_cross_fit": float(np.nanmean(res["task"]["cross_fit"])),
            "task_cross_fit_cost_ratio": float(np.nanmean(res["task"]["cross_fit_cost_ratio"])),
        }
        s = summary[name]
        print(f"{name:20s} single in/cross {s['single_in_sample']:.3f}/{s['single_cross_fit']:.3f}  "
              f"task-cond in/cross {s['task_in_sample']:.3f}/{s['task_cross_fit']:.3f}  cross-fit gain {s['task_cross_fit'] - s['single_cross_fit']:+.3f}  "
              f"(eval cost/B {s['task_cross_fit_cost_ratio']:.3f})")
    pathlib.Path(args.out_json).write_text(json.dumps({
        "protocol": "dispatch_surface_rev2_task_conditional_headroom", "posthoc_exploratory": True,
        "interval": [b_l, b_h], "cross_fit": "even/odd subset index folds", "rosters": summary,
        "arms": {k: v for k, v in rosters.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
