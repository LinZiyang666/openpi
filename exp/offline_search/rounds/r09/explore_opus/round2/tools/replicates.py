"""Replicate-pooled outcome decomposition (O1).

Pools every closed-loop replicate of the *same* controller on the same (task, init) pair across run roots,
fleets and rounds -- discovery inits 0-29 only (``catalog.py`` already dropped everything else; this module
re-asserts it).  Answers:

* run-to-run churn of a controller against itself (the null for paired "+b/-c" counts);
* how much of a cell's pure-cache deficit sits in pairs that always fail (structural) vs pairs that flip
  between replicates (knife-edge);
* where a new controller (corrector, guards, pure policy) gains, stratified by the pair's pooled pure-cache
  success probability.

Outputs ``OUT/replicates/<cell>.json`` and a summary ``OUT/replicates/summary.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .common import DERIVED, OUT, dump

CELLS = [(m, s, n) for m in ("pi05", "groot") for s in ("l10", "spatial") for n in (50, 500)]
CACHE_FITS = {"r5t_p_l10_50_tail1uc.pkl", "r5t_p_l10_500_tail1uc.pkl", "r5t_p_sp_50_tail1uc.pkl",
              "r4b3_p_sp_500_tail1uc.pkl", "r5x_g_l10_50_tail1u.pkl", "r5x_g_l10_500_tail1u.pkl",
              "r5x_g_sp_50_tail1u.pkl", "r5x_g_sp_500_tail1u.pkl"}
CACHE_FLAGS = "--os-no-shadow-native --os-blind"
P10_FITS_PREFIX = ("r8_pi05_", "r8_groot_")


def family(row) -> str | None:
    m = row.method.split(":")[-1]
    if m == "BlindAWM" and row.fit in CACHE_FITS and row.flags == CACHE_FLAGS:
        return "cache"
    if m == "PolicyEveryTen":
        return "policy10"
    if m == "ResidualCache":
        return "corrector"
    if m in ("CommitJudge", "GrootCommitJudge") and row.fit and "_c10_" in row.fit and "cap" not in row.arm:
        return "guards_B"
    if m in ("TriggerCommitJudge", "TriggerGrootCommitJudge") and "onlynp" in row.arm:
        return "only_no_progress"
    if m in ("TriggerCommitJudge", "TriggerGrootCommitJudge") and "no_progress" in row.arm:
        return "B_minus_no_progress"
    if m == "CallController" and "_CU" in row.arm:
        return "uniform_calls"
    if m == "CallController" and "_CT" in row.arm:
        return "stage_tilted_calls"
    return None


def load():
    arms = pd.read_parquet(DERIVED / "catalog" / "arms.parquet")
    outc = pd.read_parquet(DERIVED / "catalog" / "outcomes.parquet")
    if len(outc) and outc["init"].max() >= 30:
        raise RuntimeError("RULE 1: catalog contains init >= 30")
    arms = arms[arms.n_pairs >= 290].copy()       # complete discovery populations only
    arms["family"] = [family(r) for r in arms.itertuples()]
    arms = arms[arms.family.notna()]
    # pure-policy arms have no library; attach them to both library sizes of their suite
    rows = []
    for r in arms.itertuples():
        libs = [50, 500] if r.family == "policy10" else [int(r.lib)] if r.lib == r.lib else []
        for lib in libs:
            rows.append(dict(root=r.root, arm=r.arm, model=r.model, suite=r.suite, lib=lib, family=r.family))
    arms = pd.DataFrame(rows)
    outc = outc.merge(arms, on=["root", "arm"])
    # one accepted attempt per (arm, pair)
    outc = outc.sort_values("attempt").drop_duplicates(["root", "arm", "lib", "task", "init"], keep="last")
    return arms, outc


def matrix(df):
    """pairs x replicates success matrix (NaN where a replicate lacks the pair)."""
    df = df.assign(rep=df.root + "/" + df.arm)
    return df.pivot_table(index=["task", "init"], columns="rep", values="success", aggfunc="first").astype(float)


def churn(M):
    """Discordant counts for every pair of replicate columns."""
    out = []
    for a, b in itertools.combinations(M.columns, 2):
        ok = M[a].notna() & M[b].notna()
        x, y = M.loc[ok, a].values, M.loc[ok, b].values
        out.append(dict(a=a, b=b, n=int(ok.sum()), a_only=int(((x == 1) & (y == 0)).sum()),
                        b_only=int(((x == 0) & (y == 1)).sum())))
    return out


def betabin_fit(k, n):
    """MLE of a beta-binomial over pairs; returns (alpha, beta)."""
    k, n = np.asarray(k, float), np.asarray(n, float)

    def nll(theta):
        a, b = np.exp(theta)
        return -np.sum(special.betaln(k + a, n - k + b) - special.betaln(a, b))
    best = optimize.minimize(nll, x0=np.log([1.0, 0.3]), method="Nelder-Mead", options=dict(maxiter=4000))
    return tuple(np.exp(best.x))


def cell_report(arms, outc, model, suite, lib, rng):
    cell = f"{model}_{suite}_{lib}"
    d = outc[(outc.model == model) & (outc.suite == suite) & (outc.lib == lib)]
    rep = {}
    mats = {fam: matrix(g) for fam, g in d.groupby("family")}
    for fam, M in mats.items():
        c = churn(M)
        disc = [x["a_only"] + x["b_only"] for x in c]
        rep[fam] = dict(replicates=list(M.columns), n_rep=M.shape[1],
                        sr_per_replicate={k: float(v) for k, v in M.mean().items()},
                        sr_pooled=float(np.nanmean(M.values)),
                        churn_pairs=c, discordant_mean=float(np.mean(disc)) if disc else None,
                        discordant_range=[int(min(disc)), int(max(disc))] if disc else None)
    if "cache" not in mats:
        return cell, rep
    C = mats["cache"]
    k, n = np.nansum(C.values, 1), np.sum(np.isfinite(C.values), 1)
    p = k / n
    a, b = betabin_fit(k, n)
    # implied distribution of per-pair success probability under the cache
    q = stats.beta(a, b)
    rep["cache_decomposition"] = dict(
        pairs=int(len(p)), replicates_per_pair_min=int(n.min()), replicates_per_pair_max=int(n.max()),
        always_fail=float(np.mean(k == 0)), always_succeed=float(np.mean(k == n)), mixed=float(np.mean((k > 0) & (k < n))),
        deficit_total=float(1 - p.mean()),
        deficit_from_always_fail=float(np.mean(k == 0)),
        deficit_from_mixed=float(np.mean(np.where((k > 0) & (k < n), 1 - p, 0))),
        betabin_alpha=float(a), betabin_beta=float(b),
        betabin_mass_p_below_10=float(q.cdf(.1)), betabin_mass_p_above_90=float(1 - q.cdf(.9)),
        betabin_mass_between=float(q.cdf(.9) - q.cdf(.1)),
        expected_churn_two_runs=float(2 * np.sum(p * (1 - p) * n / np.maximum(n - 1, 1))),
    )
    pooled = pd.Series(p, index=C.index)
    bins = [("p=0", lambda x: x == 0), ("0<p<.5", lambda x: (x > 0) & (x < .5)),
            (".5<=p<1", lambda x: (x >= .5) & (x < 1)), ("p=1", lambda x: x == 1)]
    strat = {}
    for fam, M in mats.items():
        if fam == "cache":
            continue
        idx = M.index.intersection(pooled.index)
        y = M.loc[idx].mean(1)                 # pooled success of the other controller per pair
        x = pooled.loc[idx]
        rows = {}
        for name, f in bins:
            sel = f(x.values)
            rows[name] = dict(pairs=int(sel.sum()), cache_p=float(x.values[sel].mean()) if sel.any() else None,
                              other_sr=float(y.values[sel].mean()) if sel.any() else None,
                              gain_pairs=float((y.values[sel] - x.values[sel]).sum()) if sel.any() else None)
        # discordance of each replicate of the other family against each cache replicate vs cache-vs-cache
        vs = []
        for col in M.columns:
            for cc in C.columns:
                ok = M[col].notna() & C[cc].notna()
                u, v = M.loc[ok, col].values, C.loc[ok, cc].values
                vs.append(dict(other=col, cache=cc, other_only=int(((u == 1) & (v == 0)).sum()),
                               cache_only=int(((u == 0) & (v == 1)).sum())))
        strat[fam] = dict(by_cache_p=rows, vs_cache=vs,
                          total_gain_pp=float(100 * (y.mean() - x.mean())))
    rep["stratified"] = strat
    # per-task descriptive breakdown (diagnostic only, never used as a switch)
    tk = {}
    for t in range(10):
        sel = C.index.get_level_values(0) == t
        tk[t] = dict(cache_p=float(p[sel].mean()), always_fail=int(np.sum(k[sel] == 0)),
                     mixed=int(np.sum((k[sel] > 0) & (k[sel] < n[sel]))))
        for fam, M in mats.items():
            if fam != "cache":
                tk[t][fam] = float(np.nanmean(M.loc[M.index.get_level_values(0) == t].values))
    rep["per_task_descriptive"] = tk
    return cell, rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    arms, outc = load()
    rng = np.random.default_rng(20261002)
    summary = {}
    for m, s, n in CELLS:
        cell, rep = cell_report(arms, outc, m, s, n, rng)
        dump(OUT / "replicates" / f"{cell}.json", rep)
        dec = rep.get("cache_decomposition", {})
        summary[cell] = dict({fam: dict(n_rep=r["n_rep"], sr=r["sr_pooled"], churn=r["discordant_mean"])
                              for fam, r in rep.items() if isinstance(r, dict) and "n_rep" in r},
                             decomposition=dec,
                             gains={fam: dict(total_pp=v["total_gain_pp"],
                                              by_bin={b: (x["pairs"], x["cache_p"], x["other_sr"]) for b, x in v["by_cache_p"].items()})
                                    for fam, v in rep.get("stratified", {}).items()})
    dump(OUT / "replicates" / "summary.json", summary)
    for cell, s in summary.items():
        print("==", cell)
        for fam, v in s.items():
            if fam not in ("decomposition", "gains"):
                print(f"  {fam:22s} reps={v['n_rep']} sr={v['sr']:.3f} churn={v['churn']}")
        d = s["decomposition"]
        if d:
            print("  cache: always_fail={always_fail:.3f} mixed={mixed:.3f} always_succ={always_succeed:.3f} "
                  "deficit={deficit_total:.3f} (fail0 {deficit_from_always_fail:.3f}, mixed {deficit_from_mixed:.3f}) "
                  "betabin a={betabin_alpha:.2f} b={betabin_beta:.2f} exp_churn={expected_churn_two_runs:.1f}".format(**d))
        for fam, g in s["gains"].items():
            print(f"  gain {fam:20s} {g['total_pp']:+.1f}pp", {b: (x[0], None if x[1] is None else round(x[1], 2), None if x[2] is None else round(x[2], 2)) for b, x in g["by_bin"].items()})


if __name__ == "__main__":
    main()
