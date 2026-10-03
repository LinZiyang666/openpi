"""Per-task budget allocation: how much of the per-task heterogeneity is predictable.

Two ways to decide which tasks get policy calls:

* ``outcome``: per-task closed-loop outcomes on a calibration fold (what a few
  recorded A episodes per task would give);
* ``signal``: a label-free per-task score from the shadows on the calibration fold
  (mean served-vs-policy gap, gripper disagreement, retrieval distance) or the
  per-task A success on that fold.

Allocations are evaluated on the held-out fold through the paired outcome matrix
(``paired.cell_matrix``): hard tasks (top-k by score) get a high dose variant,
the rest a low dose variant; sweeping k and the two dose levels traces a
frontier whose points are real closed-loop mixtures.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import common, paired

LADDER_ALL = ("W10", "SW", "FL", "SF1", "A", "O5b", "O5a", "IP", "CU", "CT", "P10")
SIGNALS = ("gap_all", "gap_first5", "grip_all", "d1_mean_looks", "gap_looks")


def shadow_episodes(model, suite, lib, shadow_dir=common.OUT / "shadow"):
    p = Path(shadow_dir) / f"episodes_{common.arm_name(model, suite, lib, 'A')}.parquet"
    return pd.read_parquet(p)


def task_scores(epi, inits, signals=SIGNALS):
    """Per-task mean of each signal over the given inits (A arm shadows), plus A failure rate on those inits."""
    sel = epi[epi["init"].isin(list(inits))]
    g = sel.groupby("task_id")
    out = g[list(signals)].mean()
    out["a_fail"] = 1 - g.success.mean()
    return out


def rank_allocation(scores, k_hard, high, low):
    order = scores.sort_values(ascending=False).index.tolist()
    return {int(t): (high if i < k_hard else low) for i, t in enumerate(order)}


def signal_frontier(matrix, epi, signal, folds, ladder, n_boot=0):
    """Cross-validated frontier of top-k allocations driven by one per-task signal."""
    inits_all = sorted(set(itertools.chain.from_iterable(folds)))
    m_all = matrix[matrix.index.get_level_values("init").isin(inits_all)]
    tasks_in = m_all.index.get_level_values("task_id")
    points = []
    levels = [v for v in ladder if f"{v}__success" in matrix.columns]
    for k_hard in range(0, 11):
        for hi, lo in itertools.product(levels, levels):
            if levels.index(hi) < levels.index(lo):
                continue
            if k_hard == 0 and hi != lo:
                continue
            assign = pd.Series(index=m_all.index, dtype=object)
            for ev in folds:
                fit = sorted(set(inits_all) - set(ev))
                sc = task_scores(epi, fit)[signal]
                choice = rank_allocation(sc, k_hard, hi, lo)
                sel = m_all.index.get_level_values("init").isin(list(ev))
                assign[sel] = [choice[t] for t in tasks_in[sel]]
            s = paired.summarize(m_all, assign, n_boot=n_boot)
            points.append(dict(k_hard=k_hard, high=hi, low=lo, sr=s["sr"], ir=s["ir"]))
    return points


UNIFORM_REF = ("A", "IP", "CU", "P10")   # the uniform-dose family: pure cache, p=.25 coin, rho=.3/.18 lottery, pure policy


def interp_uniform(points_arms, ir, ref=UNIFORM_REF):
    """Piecewise-linear SR of the uniform-dose arms at owner IR ``ir`` (the reference every allocation is judged against)."""
    xs = sorted((a["ir"], a["sr"]) for v, a in points_arms.items() if v in ref)
    return float(np.interp(ir, [x for x, _ in xs], [y for _, y in xs]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(common.OUT / "task_alloc"))
    ap.add_argument("--ladder", default="A,O5b,IP,CU,CT,P10")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ladder = tuple(a.ladder.split(","))
    eps = paired.load_episodes()
    disc = list(common.DISCOVERY_INITS)
    folds = [disc[0::3], disc[1::3], disc[2::3]]
    report = {}
    for model, suite, lib in common.CELLS:
        cell = f"{model}_{suite}_{lib}"
        M = paired.cell_matrix(eps, model, suite, lib)
        epi = shadow_episodes(model, suite, lib)
        T = paired.per_task_table(M, [v for v in ("A", "P10") if f"{v}__success" in M.columns], disc)
        gap = (T["P10_sr"] - T["A_sr"])
        sc = task_scores(epi, disc)
        corr = {s: dict(spearman=float(stats.spearmanr(sc[s], gap.loc[sc.index]).statistic),
                        pearson=float(np.corrcoef(sc[s], gap.loc[sc.index])[0, 1])) for s in list(SIGNALS) + ["a_fail"]}
        # fold-to-fold stability of the per-task ranking (how reproducible is a 10-init calibration)
        stab = {}
        for s in list(SIGNALS) + ["a_fail"]:
            rs = []
            for f1, f2 in itertools.combinations(folds, 2):
                rs.append(float(stats.spearmanr(task_scores(epi, f1)[s], task_scores(epi, f2)[s]).statistic))
            stab[s] = float(np.mean(rs))
        arms = {v: paired.summarize(M[M.index.get_level_values("init").isin(disc)],
                                    pd.Series(v, index=M[M.index.get_level_values("init").isin(disc)].index), n_boot=0)
                for v in paired.variants_of(M)}
        fronts = {}
        for s in ("gap_all", "grip_all", "d1_mean_looks", "a_fail"):
            pts = paired.hull(signal_frontier(M, epi, s, folds, ladder))
            fronts[s] = [dict(p, gain_vs_uniform=p["sr"] - interp_uniform(arms, p["ir"])) for p in pts]
        cv_outcome = [dict(sr=p["sr"], ir=p["ir"], gain_vs_uniform=p["sr"] - interp_uniform(arms, p["ir"]))
                      for p in paired.hull(paired.cv_frontier(M, [v for v in ladder if f"{v}__success" in M.columns], folds, n_boot=0))]
        report[cell] = dict(gap_p10_minus_a=gap.round(3).to_dict(), scores=sc.round(4).reset_index().to_dict("records"),
                            corr_with_gap=corr, fold_stability=stab, arms={k: dict(sr=v["sr"], ir=v["ir"]) for k, v in arms.items()},
                            signal_fronts=fronts,
                            outcome_cv_front=cv_outcome)
        print(f"\n=== {cell}  corr(signal, P10-A gap): " + "  ".join(f"{s}:{c['spearman']:+.2f}" for s, c in corr.items()))
        print("   fold stability (spearman between folds): " + "  ".join(f"{s}:{v:+.2f}" for s, v in stab.items()))
        for s, pts in fronts.items():
            print(f"   {s:14s} CV front: " + "  ".join(f"{p['sr']:.3f}@{p['ir']:.3f}({100*p['gain_vs_uniform']:+.1f})" for p in pts))
        print("   outcome-CV front:    " + "  ".join(f"{p['sr']:.3f}@{p['ir']:.3f}({100*p['gain_vs_uniform']:+.1f})" for p in cv_outcome))
    common.write_json(out / "task_alloc.json", report)
    return 0


# ----------------------------------------------------------------------------- pre-declared rules (honest CV)
RULES = [dict(k=k, high=hi, low=lo) for k in (2, 3, 4, 5) for hi in ("IP", "CU", "P10") for lo in ("A",)]


def uniform_match(matrix, ir_target, lo="A", hi="CU"):
    """Per-pair expected outcome of the uniform-dose reference at the same owner IR: a random fraction f of
    episodes run ``hi`` and the rest ``lo``; f is solved so the pooled IR equals ``ir_target``.
    Returns fractional success per pair and (cost, n_dec) per pair; None when the target is outside [lo, hi]."""
    cl, nl = matrix[f"{lo}__cost"].values, matrix[f"{lo}__n_dec"].values
    ch, nh = matrix[f"{hi}__cost"].values, matrix[f"{hi}__n_dec"].values
    ir_lo, ir_hi = cl.sum() / nl.sum(), ch.sum() / nh.sum()
    if not (min(ir_lo, ir_hi) - 1e-9 <= ir_target <= max(ir_lo, ir_hi) + 1e-9):
        return None
    # pooled IR of the mixture is (f*Ch + (1-f)*Cl) / (f*Nh + (1-f)*Nl) = target  ->  solve for f
    Cl, Nl, Ch, Nh = cl.sum(), nl.sum(), ch.sum(), nh.sum()
    f = (ir_target * Nl - Cl) / ((Ch - Cl) - ir_target * (Nh - Nl))
    f = float(np.clip(f, 0, 1))
    succ = f * matrix[f"{hi}__success"].values + (1 - f) * matrix[f"{lo}__success"].values
    return dict(f=f, succ=succ, cost=f * ch + (1 - f) * cl, n_dec=f * nh + (1 - f) * nl, lo=lo, hi=hi)


def evaluate_rule(matrix, epi, signal, folds, rule, n_boot=2000, seed=0):
    """CV evaluation of one pre-declared top-k rule; paired bootstrap of the gain against the IR-matched uniform mix."""
    inits_all = sorted(set(itertools.chain.from_iterable(folds)))
    m_all = matrix[matrix.index.get_level_values("init").isin(inits_all)]
    tasks_in = m_all.index.get_level_values("task_id")
    assign = pd.Series(index=m_all.index, dtype=object)
    for ev in folds:
        fit = sorted(set(inits_all) - set(ev))
        sc = task_scores(epi, fit)[signal] if signal != "oracle" else -pd.Series(
            paired.per_task_table(matrix, ["A"], fit)["A_sr"])
        choice = rank_allocation(sc, rule["k"], rule["high"], rule["low"])
        sel = m_all.index.get_level_values("init").isin(list(ev))
        assign[sel] = [choice[t] for t in tasks_in[sel]]
    succ, cost, ndec = paired.mixture(m_all, assign)
    ir = cost.sum() / ndec.sum()
    ref = None
    for lo, hi in ((rule["low"], "IP"), (rule["low"], "CU"), ("IP", "CU"), ("CU", "P10"), (rule["low"], "P10")):
        if f"{hi}__success" in matrix.columns and f"{lo}__success" in matrix.columns:
            ref = uniform_match(m_all, ir, lo, hi)
            if ref is not None:
                break
    out = dict(rule=rule, signal=signal, sr=float(succ.mean()), ir=float(ir), n=int(len(succ)))
    if ref is not None:
        task = m_all.index.get_level_values("task_id").values
        boot = common.task_strat_bootstrap(np.column_stack([succ, ref["succ"]]), task, n_boot=n_boot, seed=seed, stat=np.mean)
        lo_, hi_ = common.ci(boot[:, 0] - boot[:, 1])
        out.update(ref_sr=float(ref["succ"].mean()), ref_mix=f"{ref['lo']}/{ref['hi']} f={ref['f']:.2f}",
                   gain=float(succ.mean() - ref["succ"].mean()), gain_ci=[float(lo_), float(hi_)])
    return out


def calibration_size_stability(epi, signal, sizes=(3, 5, 10, 20), n_rep=200, seed=0):
    """Spearman between the per-task ranking from n random inits per task and the ranking from all 30 discovery inits."""
    rng = np.random.default_rng(seed)
    disc = list(common.DISCOVERY_INITS)
    full = task_scores(epi, disc)[signal]
    out = {}
    for n in sizes:
        rs = []
        for _ in range(n_rep):
            sub = list(rng.choice(disc, n, replace=False))
            rs.append(float(stats.spearmanr(task_scores(epi, sub)[signal].reindex(full.index), full).statistic))
        out[n] = dict(mean=float(np.mean(rs)), p10=float(np.percentile(rs, 10)))
    return out


def rules_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.OUT / "task_alloc"))
    ap.add_argument("--signals", default="gap_all,grip_all,d1_mean_looks,a_fail")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = paired.load_episodes()
    disc = list(common.DISCOVERY_INITS)
    folds = [disc[0::3], disc[1::3], disc[2::3]]
    report = {}
    for model, suite, lib in common.CELLS:
        cell = f"{model}_{suite}_{lib}"
        M = paired.cell_matrix(eps, model, suite, lib)
        epi = shadow_episodes(model, suite, lib)
        rows = []
        for signal in a.signals.split(","):
            for rule in RULES:
                if f"{rule['high']}__success" not in M.columns:
                    continue
                rows.append(evaluate_rule(M, epi, signal, folds, rule, n_boot=a.n_boot))
        stab = {s: calibration_size_stability(epi, s) for s in a.signals.split(",")}
        report[cell] = dict(rules=rows, calibration_size_stability=stab)
        print(f"\n=== {cell}")
        for r in rows:
            g = f"{100*r['gain']:+.1f} [{100*r['gain_ci'][0]:+.1f},{100*r['gain_ci'][1]:+.1f}] vs {r['ref_mix']}" if "gain" in r else "no matched uniform reference"
            print(f"  {r['signal']:14s} top{r['rule']['k']} {r['rule']['high']:>3s}/{r['rule']['low']}: SR {r['sr']:.3f} @ IR {r['ir']:.3f}   gain {g}")
        for s, st in stab.items():
            print(f"  calibration-size stability {s}: " + "  ".join(f"n={n}: mean {v['mean']:.2f} p10 {v['p10']:.2f}" for n, v in st.items()))
    common.write_json(out / "rules.json", report)


if __name__ == "__main__":
    import sys
    if "--rules" in sys.argv:
        sys.argv.remove("--rules")
        rules_main()
    else:
        main()
