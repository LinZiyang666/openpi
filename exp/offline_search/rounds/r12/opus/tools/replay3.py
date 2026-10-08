"""R12 opus: third replay pass -- can the one-camera retrieval itself be made closer to two-camera? (library only)

Variants (same folds / recipe as replay.py):
  Wl3, Wl1     wrist-only metric with ridge lam .3 / 1.0 (deployed F uses .1)
  Wnn8         wrist-only metric with 8 action-neighbour pairs per row (deployed 3)
  WC           wrist-only distance + continuity: dt = d/median(d) + .5 * c / s_c, with c the RMS (sigma) distance
               between a candidate's executed head and the continuation head of the previous look's F kernel
               (its members advanced two rows); s_c = median c over the task's training rows (AWM regime-1 form)
  FC           the same continuity term on the two-camera distance (reference)
  WLC          WL (wrist + agentview from the previous look, lag-matched) + continuity
Output: out/replay3/<cell>_<size>_f<fold>.npz (served, mem, w, d1 per variant; row order = replay.py).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

OUTD = cm.OUT / "replay3"
VARIANTS = ("Wl3", "Wl1", "Wnn8", "WC", "FC", "WLC")
CELLS = [(m, s, n) for n in (50, 500, 200) for m in ("pi05", "groot") for s in ("l10", "spatial")]


class M:
    def __init__(self, X, heads, ep, step, lam=0.1, nn=3):
        self.mean, self.std, self.W = fit_metric(X, heads, ep, nn=nn, lam=lam)
        m0 = step <= 2
        self.mean0, self.std0, self.W0 = fit_metric(X[m0], heads[m0], ep[m0], nn=nn, lam=lam)

    def code(self, X):
        return ((X - self.mean) / self.std) @ self.W

    def code0(self, X):
        return ((X - self.mean0) / self.std0) @ self.W0


def run_job(job):
    model, suite, size, fold = job
    out_path = OUTD / f"{model}_{suite}_{size}_f{fold}.npz"
    if out_path.exists():
        return str(out_path), 0.0
    t0 = time.time()
    C = cm.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, size)
    tr = fold_of_row != fold
    P0, P1, src = cm.projections(model, suite, size, fold, rows, tr, keep=128)
    P0, P1 = P0[:, :64].astype(np.float64), P1[:, :64].astype(np.float64)
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    lag2 = cm.lag_index(C, rows, 2)
    nxt2 = cm.next_index(C, rows, 2)
    sig_m = act[tr][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
    kref = cm.kref_for(size)
    XF = np.concatenate([P0, P1, rs], 1)
    XW = np.concatenate([P1, rs], 1)
    XWL = np.concatenate([P0[lag2], P1, rs], 1)
    res = {"row": []}
    per = {v: {k: [] for k in ("served", "mem", "w", "d1")} for v in VARIANTS}

    def dists(met, X, itr, ite):
        Zc, Zc0, Zq, Zq0 = met.code(X[itr]), met.code0(X[itr]), met.code(X[ite]), met.code0(X[ite])
        D = core.pdist(Zq, Zc)
        s0 = step[ite] == 0
        d1 = D.min(1)
        if s0.any():
            D[s0] = core.pdist(Zq0[s0], Zc0)
        return D, d1, s0

    def put(v, D, d1, itr):
        idx, Dk, w = cm.topk(D, kref)
        per[v]["served"].append(np.einsum("qk,qktc->qtc", w, act[itr][idx]).astype(np.float32))
        per[v]["mem"].append(rows[itr][idx]); per[v]["w"].append(w.astype(np.float32)); per[v]["d1"].append(d1)

    for t in range(10):
        itr = np.flatnonzero(tr & (task == t))
        ite = np.flatnonzero(~tr & (task == t))
        if not len(ite):
            continue
        res["row"].append(rows[ite])
        for v, lam, nn in (("Wl3", .3, 3), ("Wl1", 1.0, 3), ("Wnn8", .1, 8)):
            D, d1, _ = dists(M(XW[itr], heads[itr], ep[itr], step[itr], lam=lam, nn=nn), XW, itr, ite)
            put(v, D, d1, itr)
        # continuity target: previous look's F kernel advanced two rows
        mF = M(XF[itr], heads[itr], ep[itr], step[itr])
        Dp, _, _ = dists(mF, XF, itr, lag2[ite])
        idx_p, _, w_p = cm.topk(Dp, kref)
        adv = nxt2[itr][idx_p]
        cont = np.einsum("qk,qkd->qd", w_p, heads[adv])            # continuation head (sigma units, 35-d)
        Hc = heads[itr]
        c = np.sqrt(np.maximum((cont * cont).sum(1)[:, None] - 2 * cont @ Hc.T + (Hc * Hc).sum(1)[None], 0) / 35)
        # s_c: median nearest continuity of the task's training rows (own successor vs other episodes' heads)
        tails = heads[nxt2[itr]]
        cc = np.sqrt(np.maximum((tails * tails).sum(1)[:, None] - 2 * tails @ Hc.T + (Hc * Hc).sum(1)[None], 0) / 35)
        cc[ep[itr][:, None] == ep[itr][None, :]] = np.inf
        s_c = float(np.median(cc.min(1))) + 1e-6
        first = step[ite] < 2
        for v, met, X in (("WC", M(XW[itr], heads[itr], ep[itr], step[itr]), XW), ("FC", mF, XF),
                          ("WLC", M(XWL[itr], heads[itr], ep[itr], step[itr]), XWL)):
            D, d1, s0 = dists(met, X, itr, ite)
            med = np.median(D, axis=1, keepdims=True) + 1e-12
            Dt = D / med + 0.5 * c / s_c
            Dt[first] = D[first]                                       # no previous look before step 2
            put(v, Dt, d1, itr)
    arrays = {k: np.concatenate(a) for k, a in res.items()}
    for v in VARIANTS:
        for k, a in per[v].items():
            arrays[f"{v}|{k}"] = np.concatenate(a)
    meta = dict(model=model, suite=suite, size=size, fold=fold, pca=src, kref=kref, variants=list(VARIANTS),
                wall_s=time.time() - t0)
    OUTD.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_path)
    return str(out_path), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    cells = CELLS if not a.only else [(s.split(":")[0], s.split(":")[1], int(s.split(":")[2]))
                                      for s in a.only.split(",")]
    jobs = [(m, s, n, f) for m, s, n in cells for f in range(5)]
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for path, dt in pool.imap_unordered(run_job, jobs):
            print(f"done {path} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
