"""R12 opus: library-only k-fold replay of full (two-camera) vs single-camera ("look half") retrieval.

For every R10 nested cell-size and fold f, every held-out row is served by the training-fold cache with several
feature sets, each with its OWN per-task AWM Mahalanobis metric (``awm.fit_metric``: nn 3, lam .1, main + early)
fitted on training-fold rows only:

  F     [PCA64 v0, PCA64 v1, rs8]                deployed two-camera features (reference)
  W     [PCA64 v1, rs8]                          wrist only (v1 = eye-in-hand)
  W128  [PCA128 v1, rs8]                         wrist only, same vision width as F
  A     [PCA64 v0, rs8]                          third-person only (v0 = agentview)
  A128  [PCA128 v0, rs8]
  R     [rs8]                                    no vision at all (floor)
  WL    [PCA64 v0(t-2), PCA64 v1(t), rs8(t)]     wrist fresh + agentview from the previous look (10 controls old)
  AL    [PCA64 v0(t), PCA64 v1(t-2), rs8(t)]     agentview fresh + wrist from the previous look
  FL    [PCA64 v0(t-2), PCA64 v1(t-2), rs8(t)]   both cameras stale (no new vision, state refreshed)

Library candidates carry the same lag structure (their own episode's row two decisions earlier), so the query and
the library always compare like with like. Step-0 rows use each variant's early metric. Regime 2 (after a HIT).

Output per (cell, size, fold): out/replay/<model>_<suite>_<size>_f<fold>.npz with per-held-out-row metadata and,
per variant, the served chunk [10,7], top-16 parent rows + kernel weights, d1.

Run (opus CPUs):
  taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
    .venv/bin/python -m exp.offline_search.rounds.r12.opus.tools.replay --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

OUTD = cm.OUT / "replay"
VARIANTS = ("F", "W", "W128", "A", "A128", "R", "WL", "AL", "FL")
CELLS = [(m, s, n) for n in (50, 200, 500) for m in ("pi05", "groot") for s in ("l10", "spatial")]


def features(P0, P1, rs, lag2, name):
    p0, p1 = P0[:, :64], P1[:, :64]
    cols = {"F": [p0, p1, rs], "W": [p1, rs], "W128": [P1, rs], "A": [p0, rs], "A128": [P0, rs], "R": [rs],
            "WL": [p0[lag2], p1, rs], "AL": [p0, p1[lag2], rs], "FL": [p0[lag2], p1[lag2], rs]}[name]
    return np.concatenate(cols, 1).astype(np.float64)


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
    P0 = P0.astype(np.float64)
    P1 = P1.astype(np.float64)
    act = C["act"][rows]
    rs = C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    lag2 = cm.lag_index(C, rows, 2)
    sig_m = act[tr][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
    kref = cm.kref_for(size)
    te_all = np.flatnonzero(~tr)
    res = {k: [] for k in ("row", "task", "ep", "step", "ep_len", "success", "prog")}
    per = {v: {k: [] for k in ("served", "mem", "w", "d1")} for v in VARIANTS}
    for t in range(10):
        itr = np.flatnonzero(tr & (task == t))
        ite = np.flatnonzero(~tr & (task == t))
        if not len(ite):
            continue
        res["row"].append(rows[ite]); res["task"].append(task[ite]); res["ep"].append(ep[ite])
        res["step"].append(step[ite]); res["ep_len"].append(C["ep_len"][rows[ite]])
        res["success"].append(C["success"][rows[ite]]); res["prog"].append(C["prog"][rows[ite]])
        for v in VARIANTS:
            X = features(P0, P1, rs, lag2, v)
            met = cm.Metric(X[itr], heads[itr], ep[itr], step[itr])
            Zc, Zc0 = met.code(X[itr]), met.code0(X[itr])
            Zq, Zq0 = met.code(X[ite]), met.code0(X[ite])
            D = core.pdist(Zq, Zc)
            s0 = step[ite] == 0
            d1 = D.min(1)
            if s0.any():
                D[s0] = core.pdist(Zq0[s0], Zc0)
            idx, Dk, w = cm.topk(D, kref)
            served = np.einsum("qk,qktc->qtc", w, act[itr][idx]).astype(np.float32)
            per[v]["served"].append(served)
            per[v]["mem"].append(rows[itr][idx])
            per[v]["w"].append(w.astype(np.float32))
            per[v]["d1"].append(d1)
    arrays = {k: np.concatenate(a) for k, a in res.items()}
    arrays["act"] = np.concatenate([act[np.flatnonzero(~tr & (task == t))] for t in range(10)])
    for v in VARIANTS:
        for k, a in per[v].items():
            arrays[f"{v}|{k}"] = np.concatenate(a)
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    meta = dict(model=model, suite=suite, size=size, fold=fold, pca=src, kref=kref, sig_m=sig_m.tolist(),
                sig_r=sig_r.tolist(), variants=list(VARIANTS), wall_s=time.time() - t0)
    OUTD.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_path)
    return str(out_path), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma list model:suite:size")
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
