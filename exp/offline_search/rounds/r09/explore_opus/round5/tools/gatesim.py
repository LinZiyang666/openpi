"""Prefix-exact success simulation of two guard-call gates, fitted on discovery inits 0-19 only (round 5, Q2).

Gate P ("on-pace silence", threshold L0): a no-progress verdict at a look whose pace lag <= L0 is dropped and the
controller behaves exactly like the pure cache (no forced look at the next decision).  Hence, per episode, the gated
controller IS the pure cache until the first look where the guard fires with lag > L0.  Simulation: take every
pure-cache replicate episode; if it never fires with lag > L0 its real outcome is the gated outcome (exact); else the
episode is cut at that look and the suffix success is the judge arms' success after their first call with lag > L0,
matched on (decision bin, lag bin).  The same construction with L0 = -inf reproduces the plain judge (calibration).

Gate C ("call budget", threshold C): at most C guard calls per episode; afterwards pure cache.  Per judge-arm
episode the gated controller is identical up to its (C+1)-th call; the suffix success is the pure cache's success
after a stall firing at the same (decision bin, span bin) -- the cache's own recovery from a stall.

Uncertainty: pair bootstrap (resampling (task, init) pairs; all arms of a pair move together), 1000 draws.
RULE 1: inputs are ``waste.py`` look tables restricted to inits 0-19 (asserted).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import dump
from exp.offline_search.rounds.r09.explore_opus.round5.tools import OUT
from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import clean

STEP_BINS = [-1, 20, 40, 60, 80, 200]
LAG_BINS = [-99, 0, 4, 8, 12, 20, 999]
SPAN_BINS = [-1, 4, 8, 12, 16, 24, 32, 999]


def _tables(model):
    L = pd.read_parquet(OUT / f"looks_{model}_fit.parquet")
    Lc = pd.read_parquet(OUT / f"looks_cache_{model}_fit.parquet")
    assert L["init"].max() < 20 and Lc["init"].max() < 20, "fit inits only"
    return L, Lc


def _bin(df, col, edges):
    return pd.cut(df[col], edges).cat.codes


def pace_sr(L, Lc, L0):
    """Simulated success of gate P (and of the plain judge for L0=None) on the pure-cache episodes."""
    thr = -np.inf if L0 is None else L0
    epc = Lc.groupby(["arm", "task", "init"]).success.first()
    fc = Lc[Lc.fire & (Lc.lag > thr)].groupby(["arm", "task", "init"])[["step", "lag"]].first()
    fj = L[L.np_call & (L.lag > thr)].groupby(["arm", "task", "init"])[["step", "lag", "success"]].first()
    for f in (fc, fj):
        f["sb"], f["lb"] = _bin(f, "step", STEP_BINS), _bin(f, "lag", LAG_BINS)
    r = fj.groupby(["sb", "lb"]).success.mean()
    r_all = fj.success.mean() if len(fj) else 0.0
    est = fc.join(r.rename("r"), on=["sb", "lb"]).r.fillna(r_all)
    clean_eps = epc[~epc.index.isin(fc.index)]
    return float((clean_eps.sum() + est.sum()) / len(epc)), float(len(fc) / len(epc))


def cap_sr(L, Lc, C):
    """Simulated success of gate C on the judge-arm episodes."""
    ep = L.groupby(["arm", "task", "init"]).success.first()
    c = L[L.call].copy()
    c["idx"] = c.groupby(["arm", "task", "init"]).cumcount() + 1
    over = c[c.idx == C + 1].copy()
    cf = Lc[Lc.fire].copy()
    for f in (over, cf):
        f["sb"], f["pb"] = _bin(f, "step", STEP_BINS), _bin(f, "span", SPAN_BINS)
    rc = cf.groupby(["sb", "pb"]).success.mean()
    over = over.join(rc.rename("rc"), on=["sb", "pb"])
    over["rc"] = over.rc.fillna(cf.success.mean())
    # an episode cut at its (C+1)-th call keeps a real failure as a failure (no credit), a success becomes rc
    est = ep.astype(float).copy()
    idx = over.set_index(["arm", "task", "init"]).index
    est.loc[idx] = np.where(over.success.values, over.rc.values, 0.0)
    saved = float((c.idx > C).sum() / len(ep))
    return float(est.mean()), saved


def bootstrap(L, Lc, fn, arg, base_arg, draws, seed):
    pairs = pd.MultiIndex.from_frame(Lc[["task", "init"]].drop_duplicates()).sort_values()
    rng = np.random.default_rng(seed)
    pt, pb = fn(L, Lc, arg)[0], fn(L, Lc, base_arg)[0]
    diffs = []
    Lg = {k: g for k, g in L.groupby(["task", "init"])}
    Lcg = {k: g for k, g in Lc.groupby(["task", "init"])}
    for _ in range(draws):
        pick = rng.integers(0, len(pairs), len(pairs))
        parts_j, parts_c = [], []
        for n, j in enumerate(pick):
            k = pairs[j]
            if k in Lg:
                parts_j.append(Lg[k].assign(init=Lg[k].init * 1000 + n))
            parts_c.append(Lcg[k].assign(init=Lcg[k].init * 1000 + n))
        Lb, Lcb = pd.concat(parts_j), pd.concat(parts_c)
        diffs.append(fn(Lb, Lcb, arg)[0] - fn(Lb, Lcb, base_arg)[0])
    d = np.array(diffs)
    return dict(point=pt - pb, lo90=float(np.quantile(d, .05)), hi90=float(np.quantile(d, .95)), sd=float(d.std()))


PACE_GRID = (-1, 0, 1, 2, 3, 4)
CAP_GRID = (10, 12, 15, 20, 25)


def _job(args):
    model, kind, thr, draws = args
    L, Lc = _tables(model)
    if kind == "pace":
        sr, extra = pace_sr(L, Lc, thr)
        bs = bootstrap(L, Lc, pace_sr, thr, None, draws, 20261002 + thr)
        return model, kind, thr, dict(sr=sr, reach_offpace=extra, delta=bs)
    sr, extra = cap_sr(L, Lc, thr)
    bs = bootstrap(L, Lc, cap_sr, thr, 10_000, draws, 20261102 + thr)
    return model, kind, thr, dict(sr=sr, calls_saved_per_ep=extra, delta=bs)


def main(argv=None):
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args(argv)
    rep = {}
    for model in ("pi05", "groot"):
        L, Lc = _tables(model)
        rep[model] = dict(judge_sr_real=float(L.groupby(["arm", "task", "init"]).success.first().mean()),
                          cache_sr_real=float(Lc.groupby(["arm", "task", "init"]).success.first().mean()),
                          pace_sim_baseline=pace_sr(L, Lc, None)[0], pace={}, cap={})
    jobs = [(m, "pace", t, a.draws) for m in rep for t in PACE_GRID] + [(m, "cap", t, a.draws) for m in rep for t in CAP_GRID]
    with ProcessPoolExecutor(a.workers) as ex:
        for model, kind, thr, res in ex.map(_job, jobs):
            rep[model][kind][thr] = res
            d = res["delta"]
            print(f"{model} {kind} {thr}: sim SR {res['sr']:.3f} delta {d['point']*100:+.2f}pp 90% [{d['lo90']*100:+.2f},{d['hi90']*100:+.2f}] "
                  + (f"reach off-pace {res['reach_offpace']:.3f}" if kind == "pace" else f"calls saved {res['calls_saved_per_ep']:.2f}/ep"),
                  flush=True)
    dump(OUT / "gatesim.json", clean(rep))


if __name__ == "__main__":
    main()
