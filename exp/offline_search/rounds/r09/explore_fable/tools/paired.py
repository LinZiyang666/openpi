"""Paired closed-loop outcomes across R8 arms and the dose-assignment simulator.

Every R8 arm ran the same 500 (task, init) pairs with the same environment seed,
so for one cell (model, suite, library) the outcome of pair *i* under arm *a* is a
real closed-loop observation. A *dose assignment* picks, for every pair, which
arm's outcome to count. If the choice depends only on information available
before the episode starts (task identity, the step-0 observation, a pre-fitted
per-task score) the mixture's success rate and owner IR are unbiased paired
estimates of the mixed controller (up to single-run noise of each arm).

Owner IR of a mixture = sum(cost) / sum(decisions) over the chosen arms.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common


def load_episodes(derived=common.DERIVED):
    frames = [pd.read_parquet(p) for p in sorted((Path(derived) / "episodes").glob("*.parquet"))]
    if not frames:
        raise FileNotFoundError("no extracted episodes under %s" % derived)
    df = pd.concat(frames, ignore_index=True)
    df["lib"] = df.lib.astype(int)
    return df


def cell_matrix(episodes, model, suite, lib):
    """Wide table indexed by (task_id, init): columns ``<variant>__{success,cost,n_dec}``.

    The pure-policy arm (no library) is attached to both libraries of its suite.
    """
    sel = episodes[(episodes.model == model) & (episodes.suite == suite) & ((episodes.lib == lib) | (episodes.variant == "P10"))]
    wide = sel.pivot_table(index=["task_id", "init"], columns="variant", values=["success", "cost", "n_dec"], aggfunc="first")
    wide.columns = [f"{v}__{k}" for k, v in wide.columns]
    wide = wide.sort_index()
    return wide


def variants_of(matrix):
    return sorted({c.split("__")[0] for c in matrix.columns})


def mixture(matrix, assignment):
    """Per-pair arrays (success, cost, n_dec) under an assignment Series (index = matrix index, values = variant)."""
    assignment = assignment.reindex(matrix.index)
    n = len(matrix)
    succ, cost, ndec = np.empty(n), np.empty(n), np.empty(n)
    for v in assignment.unique():
        m = (assignment == v).values
        succ[m] = matrix[f"{v}__success"].values[m]
        cost[m] = matrix[f"{v}__cost"].values[m]
        ndec[m] = matrix[f"{v}__n_dec"].values[m]
    if np.isnan(succ).any():
        raise ValueError("assignment references a variant missing for some pairs")
    return succ, cost, ndec


def summarize(matrix, assignment, n_boot=2000, seed=0, reference=None):
    """SR, IR and task-stratified bootstrap intervals of a mixture; optional paired delta vs a reference variant."""
    succ, cost, ndec = mixture(matrix, assignment)
    task = matrix.index.get_level_values("task_id").values
    cols = [succ, cost, ndec]
    if reference is not None:
        rs, rc, rn = mixture(matrix, pd.Series(reference, index=matrix.index))
        cols += [rs, rc, rn]
    out = dict(n=int(len(succ)), sr=float(succ.mean()), ir=float(cost.sum() / ndec.sum()), sr_ci=None, ir_ci=None)
    boot = None
    if n_boot > 0:
        boot = common.task_strat_bootstrap(np.column_stack(cols), task, n_boot=n_boot, seed=seed, stat=np.sum)
        out.update(sr_ci=[float(x) for x in common.ci(boot[:, 0] / len(succ))],
                   ir_ci=[float(x) for x in common.ci(boot[:, 1] / boot[:, 2])])
    if reference is not None:
        out.update(ref=reference, ref_sr=float(rs.mean()), ref_ir=float(rc.sum() / rn.sum()),
                   delta_sr=float(succ.mean() - rs.mean()), delta_sr_ci=None,
                   wins=int(((succ == 1) & (rs == 0)).sum()), losses=int(((succ == 0) & (rs == 1)).sum()))
        if boot is not None:
            out["delta_sr_ci"] = [float(x) for x in common.ci((boot[:, 0] - boot[:, 3]) / len(succ))]
    return out


def per_task_table(matrix, variants=None, inits=None):
    """Per-task SR and per-decision IR of every variant (optionally restricted to inits)."""
    m = matrix if inits is None else matrix[matrix.index.get_level_values("init").isin(list(inits))]
    variants = variants or variants_of(m)
    rows = []
    for t, grp in m.groupby(level="task_id"):
        row = dict(task_id=int(t), n=int(len(grp)))
        for v in variants:
            row[f"{v}_sr"] = float(grp[f"{v}__success"].mean())
            row[f"{v}_ir"] = float(grp[f"{v}__cost"].sum() / grp[f"{v}__n_dec"].sum())
            row[f"{v}_cost"] = float(grp[f"{v}__cost"].mean())
            row[f"{v}_ndec"] = float(grp[f"{v}__n_dec"].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("task_id")


def lagrangian_task_allocation(table, variants, lam):
    """For every task pick the variant maximizing SR - lam * cost (cost = mean owner cost per episode)."""
    choice = {}
    for t, row in table.iterrows():
        scores = {v: row[f"{v}_sr"] - lam * row[f"{v}_cost"] for v in variants}
        best = max(variants, key=lambda v: (scores[v], -row[f"{v}_cost"]))
        choice[int(t)] = best
    return choice


def task_frontier(matrix, variants, fit_inits, eval_inits, lams=None, n_boot=500, seed=0):
    """Sweep the Lagrange multiplier: per-task choices fitted on ``fit_inits`` outcomes, scored on ``eval_inits``.

    Returns a list of points (lam, choice, fit-SR/IR, eval-SR/IR). With fit == eval this is the in-sample oracle
    hull; with disjoint init folds it is an honest cross-validated estimate.
    """
    fit_t = per_task_table(matrix, variants, fit_inits)
    eval_m = matrix[matrix.index.get_level_values("init").isin(list(eval_inits))]
    fit_m = matrix[matrix.index.get_level_values("init").isin(list(fit_inits))]
    lams = np.geomspace(0.02, 50, 60) if lams is None else lams
    seen, points = set(), []
    for lam in list(lams) + [0.0, 1e9]:
        choice = lagrangian_task_allocation(fit_t, variants, lam)
        key = tuple(sorted(choice.items()))
        if key in seen:
            continue
        seen.add(key)
        assign_eval = pd.Series([choice[t] for t in eval_m.index.get_level_values("task_id")], index=eval_m.index)
        assign_fit = pd.Series([choice[t] for t in fit_m.index.get_level_values("task_id")], index=fit_m.index)
        ev = summarize(eval_m, assign_eval, n_boot=n_boot, seed=seed)
        fi = summarize(fit_m, assign_fit, n_boot=0, seed=seed)
        points.append(dict(lam=float(lam), choice=choice, fit_sr=fi["sr"], fit_ir=fi["ir"], eval_sr=ev["sr"], eval_ir=ev["ir"],
                           eval_sr_ci=ev["sr_ci"], eval_ir_ci=ev["ir_ci"]))
    points.sort(key=lambda p: p["eval_ir"])
    return points


def cheapest_reaching(points, target_sr, key_sr="eval_sr", key_ir="eval_ir"):
    ok = [p for p in points if p[key_sr] >= target_sr]
    return min(ok, key=lambda p: p[key_ir]) if ok else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(common.OUT / "paired"))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = load_episodes()
    disc = list(common.DISCOVERY_INITS)
    fold_a, fold_b = disc[:15], disc[15:]
    report = {}
    for model, suite, lib in common.CELLS:
        M = cell_matrix(eps, model, suite, lib)
        variants = variants_of(M)
        call_variants = [v for v in ("A", "O5b", "O5a", "IP", "CU", "CT", "P10") if v in variants]
        cell = f"{model}_{suite}_{lib}"
        target = common.P10_SR[(model, suite)]
        res = dict(variants=variants, call_variants=call_variants, target_p10=target)
        # per-task table on discovery inits
        res["per_task_discovery"] = per_task_table(M, variants, disc).round(4).reset_index().to_dict("records")
        # arm-level on discovery
        res["arms_discovery"] = {v: summarize(M[M.index.get_level_values("init").isin(disc)], pd.Series(v, index=M[M.index.get_level_values("init").isin(disc)].index), n_boot=a.n_boot) for v in variants}
        # in-sample oracle hull (discovery inits) and 2-fold cross-validated hull
        res["task_oracle_discovery"] = task_frontier(M, call_variants, disc, disc, n_boot=200)
        cv = []
        for fit, ev in ((fold_a, fold_b), (fold_b, fold_a)):
            cv.append(task_frontier(M, call_variants, fit, ev, n_boot=200))
        res["task_cv_folds"] = cv
        report[cell] = res
        common.write_json(out / f"{cell}.json", res)
        print(cell, "oracle cheapest reaching P10:", cheapest_reaching(res["task_oracle_discovery"], target))
    return 0


if __name__ == "__main__":
    main()


def cv_frontier(matrix, variants, folds, lams=None, n_boot=300, seed=0):
    """K-fold cross-validated per-task allocation hull.

    For every Lagrange multiplier the per-task choice is fitted on the other folds' outcomes and applied to the
    held-out fold; the held-out mixtures of all folds are pooled before computing SR / IR.
    """
    lams = np.geomspace(0.02, 50, 60) if lams is None else lams
    inits_all = sorted(set(itertools.chain.from_iterable(folds)))
    m_all = matrix[matrix.index.get_level_values("init").isin(inits_all)]
    points = []
    for lam in list(lams) + [0.0, 1e9]:
        assign = pd.Series(index=m_all.index, dtype=object)
        choices = []
        for k, ev in enumerate(folds):
            fit = sorted(set(inits_all) - set(ev))
            choice = lagrangian_task_allocation(per_task_table(matrix, variants, fit), variants, lam)
            choices.append(choice)
            sel = m_all.index.get_level_values("init").isin(list(ev))
            assign[sel] = [choice[t] for t in m_all.index.get_level_values("task_id")[sel]]
        s = summarize(m_all, assign, n_boot=n_boot, seed=seed)
        points.append(dict(lam=float(lam), choices=choices, sr=s["sr"], ir=s["ir"], sr_ci=s["sr_ci"], ir_ci=s["ir_ci"]))
    points.sort(key=lambda p: p["ir"])
    return points


def hull(points, key_sr="sr", key_ir="ir"):
    """Upper-left staircase: drop points dominated by a cheaper-or-equal point with >= SR."""
    out = []
    best = -1.0
    for p in sorted(points, key=lambda p: (p[key_ir], -p[key_sr])):
        if p[key_sr] > best:
            out.append(p)
            best = p[key_sr]
    return out
