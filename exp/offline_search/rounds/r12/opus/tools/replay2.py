"""R12 opus: second replay pass (library only, same folds/metric recipe as replay.py).

Variants
  Fl3   F features, metric ridge lam .3 instead of .1          (reference: an equally reasonable full metric)
  Fpca  F features on the stored r01 500-episode PCA basis     (reference: another valid PCA; sizes < 500 only)
  WL4/WL6/WL8  wrist fresh + agentview L decisions old (library side lagged identically), own metric per lag
  WI    wrist fresh + agentview imputed from the previous look: the previous look's F kernel members (training rows)
        advanced by two rows, weighted mean of their PCA64 v0; scored with the F metric (library side true v0)
  WIL   as WI but the imputed agentview is scored with the WL metric against library lag-2 features (no: true v0);
        kept simple: WI only uses F's metric.

Output: out/replay2/<cell>_<size>_f<fold>.npz (same layout as replay.py; served chunks, members, weights, d1).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

OUTD = cm.OUT / "replay2"
VARIANTS = ("Fl3", "Fpca", "WL4", "WL6", "WL8", "WI")
CELLS = [(m, s, n) for n in (50, 500, 200) for m in ("pi05", "groot") for s in ("l10", "spatial")]


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
    P0 = P0[:, :64].astype(np.float64)
    P1 = P1[:, :64].astype(np.float64)
    act = C["act"][rows]
    rs = C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    lags = {L: cm.lag_index(C, rows, L) for L in (2, 4, 6, 8)}
    nxt2 = cm.next_index(C, rows, 2)
    sig_m = act[tr][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
    kref = cm.kref_for(size)
    if size < 500:
        S0, S1 = core.stored_pca(model, suite)
        Q0, Q1 = S0[rows].astype(np.float64), S1[rows].astype(np.float64)
    res = {k: [] for k in ("row",)}
    per = {v: {k: [] for k in ("served", "mem", "w", "d1")} for v in VARIANTS}

    def serve(X, met, itr, ite, Xq=None):
        Xq = X[ite] if Xq is None else Xq
        Zc, Zc0 = met.code(X[itr]), met.code0(X[itr])
        Zq, Zq0 = met.code(Xq), met.code0(Xq)
        D = core.pdist(Zq, Zc)
        s0 = step[ite] == 0
        d1 = D.min(1)
        if s0.any():
            D[s0] = core.pdist(Zq0[s0], Zc0)
        idx, Dk, w = cm.topk(D, kref)
        served = np.einsum("qk,qktc->qtc", w, act[itr][idx]).astype(np.float32)
        return served, idx, w, d1

    def put(v, out, itr):
        served, idx, w, d1 = out
        per[v]["served"].append(served); per[v]["mem"].append(rows[itr][idx])
        per[v]["w"].append(w.astype(np.float32)); per[v]["d1"].append(d1)

    for t in range(10):
        itr = np.flatnonzero(tr & (task == t))
        ite = np.flatnonzero(~tr & (task == t))
        if not len(ite):
            continue
        res["row"].append(rows[ite])
        XF = np.concatenate([P0, P1, rs], 1)
        mF = cm.Metric(XF[itr], heads[itr], ep[itr], step[itr])
        m3 = cm.Metric.__new__(cm.Metric)
        from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric
        m3.mean, m3.std, m3.W = fit_metric(XF[itr], heads[itr], ep[itr], nn=3, lam=0.3)
        e0 = step[itr] <= 2
        m3.mean0, m3.std0, m3.W0 = fit_metric(XF[itr][e0], heads[itr][e0], ep[itr][e0], nn=3, lam=0.3)
        put("Fl3", serve(XF, m3, itr, ite), itr)
        if size < 500:
            XQ = np.concatenate([Q0, Q1, rs], 1)
            put("Fpca", serve(XQ, cm.Metric(XQ[itr], heads[itr], ep[itr], step[itr]), itr, ite), itr)
        else:
            put("Fpca", serve(XF, mF, itr, ite), itr)
        for L in (4, 6, 8):
            X = np.concatenate([P0[lags[L]], P1, rs], 1)
            put(f"WL{L}", serve(X, cm.Metric(X[itr], heads[itr], ep[itr], step[itr]), itr, ite), itr)
        # WI: previous look (two rows earlier in the held-out episode) served by F; its members advanced by two rows
        prev = lags[2][ite]
        _, idx_p, w_p, _ = serve(XF, mF, itr, prev)
        adv = nxt2[itr][idx_p]                                   # positions (within rows) two rows later
        P0hat = np.einsum("qk,qkd->qd", w_p, P0[adv])
        Xq = np.concatenate([np.where((step[ite] >= 2)[:, None], P0hat, P0[ite]), P1[ite], rs[ite]], 1)
        put("WI", serve(XF, mF, itr, ite, Xq=Xq), itr)
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
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for path, dt in pool.imap_unordered(run_job, jobs):
            print(f"done {path} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
