"""Vectorized full-cell reference of V4 / V5 (verification tool, not a method; reads GT for the metric only).

Uses the method's own fit (same task blocks, scales, basis) and evaluates every decision of a cell in batch
(numpy GEMMs per task), so that
  1. the full-cell err / AURC can be compared with ideation B's measured numbers (NOTES_ideation_B.md F1/F2/F7),
  2. a harness smoke npz of the same method can be checked decision by decision (topk + synthesized segment).
Also reports the gripper-sign-faithful err (dim 6 of the executed segment snapped to its sign before scoring).

    OMP_NUM_THREADS=4 taskset -c 9-17,53-61 .venv/bin/python exp/offline_search/rounds/r02/g2_pcacos/_ref_check.py \
        --cell pi05_spatial_cache --cls PcaCosV4 --kwargs '{}' [--npz <smoke npz>] [--out json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[4]))

from exp.offline_search.harness import api, dims, store  # noqa: E402
import method as M  # noqa: E402

ROOT = "/dev/shm/offline_search_store"
QCACHE = M.DERIVED / "scratch" / "qproj"


def aurc(conf, err):
    o = np.lexsort((np.arange(conf.size), -conf))
    e = err[o]
    return float((np.cumsum(e) / np.arange(1, e.size + 1)).mean())


def qproj(cell, fitsrc, f, k, meth):
    QCACHE.mkdir(parents=True, exist_ok=True)
    p = QCACHE / f"{cell}_fit{fitsrc}_{f}.npy"
    if p.exists():
        return np.load(p)[:, :k]
    mu, B = M.load_basis(meth.key, fitsrc, f, 128)
    X = np.load(pathlib.Path(ROOT) / "queries" / cell / f"key_{f}.npy", mmap_mode="r")
    out = np.empty((X.shape[0], 128), np.float32)
    muB = mu @ B
    for lo in range(0, X.shape[0], 2048):
        out[lo:lo + 2048] = np.asarray(X[lo:lo + 2048], np.float32) @ B - muB
    np.save(p, out)
    return out[:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--cls", default="PcaCosV4")
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="npz of per-decision err / regime / confidence features")
    a = ap.parse_args()
    cell = a.cell
    model, suite, arm = store.parse_cell(cell)
    meth = getattr(M, a.cls)(**json.loads(a.kwargs))
    lib = store.LibraryView(ROOT, store.lib_key(cell), "current")
    ctx = api.Context(root=ROOT, cell=cell, seed=0, scratch=M.DERIVED / "scratch" / "refctx")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    meth.fit(lib, ctx)
    qd = pathlib.Path(ROOT) / "queries" / cell
    eps = json.loads((qd / "episodes.json").read_text())
    ep = np.load(qd / "ep.npy").astype(np.int64)
    step = np.load(qd / "step.npy").astype(np.int64)
    N = ep.size
    task = np.asarray([e["task_id"] for e in eps], np.int64)[ep]
    rs = np.asarray(np.load(qd / "rs.npy", mmap_mode="r")[:, :M.RSV], np.float32)
    a_inf = np.asarray(np.load(qd / "a_inf.npy", mmap_mode="r")[:, :5, :7], np.float64)
    a_exec = np.asarray(np.load(qd / "a_exec.npy", mmap_mode="r")[:, :10, :7], np.float64)
    sig = meth.sig
    prev_idx = np.full(N, -1)
    same = np.r_[False, (ep[1:] == ep[:-1]) & (step[1:] == step[:-1] + 1)]
    prev_idx[same] = np.arange(N)[same] - 1
    regime = np.where(step == 0, 0, 1 if arm == "inf" else 2)
    Q = {f: qproj(cell, meth.fitsrc, f, meth.k, meth) for f in ("v0", "v1")}
    fresh_branch = isinstance(meth, M.PcaCosV5Fresh)
    act = np.load(meth._act_path, mmap_mode="r")
    TOP = np.full((N, 10), -1, np.int64)
    SEG = np.zeros((N, 5, 7))
    CONF = np.zeros(N)
    FE = {k: np.full(N, np.nan) for k in ("dnn", "disp", "over", "cmax", "c0n", "vis0")}
    for t, T in meth.tasks.items():
        qi = np.nonzero(task == t)[0]
        if qi.size == 0:
            continue
        C0 = (M._unit_rows(Q["v0"][qi] - T.m0) @ T.Z0.T).astype(np.float64)
        C1 = (M._unit_rows(Q["v1"][qi] - T.m1) @ T.Z1.T).astype(np.float64)
        allm = np.ones_like(C0, bool)
        Z0, Z1 = M._zmask(C0, allm), M._zmask(C1, allm)
        D = M._pair_l2(rs[qi], T.RS).astype(np.float64)
        ZD = M._zmask(-D, allm)
        S = Z0 + Z1 + meth.st * ZD
        elig = np.ones_like(S, bool)
        if meth.align0 and T.has0:
            s0 = step[qi] == 0
            elig[s0] = T.s0mask[None, :]
            S = np.where(elig, S, M.NEG)
        branch = np.zeros(qi.size, bool)
        if fresh_branch:
            fr = regime[qi] == 1
            if fr.any():
                tail = (a_exec[prev_idx[qi[fr]], 5:10] / sig).reshape(fr.sum(), M.NH)
                Cc = np.sqrt(np.maximum(np.einsum("ij,ij->i", tail, tail)[:, None] - 2 * tail @ T.HD.T.astype(np.float64)
                                        + T.h2[None, :], 0) / M.NH)
                F = -Cc / meth.s_c + meth.lam * (Z0[fr] + Z1[fr]) / 2 + meth.alpha * ZD[fr]
                S[fr] = F
                branch[fr] = True
        o = np.argsort(-S, axis=1, kind="stable")[:, :max(10, meth.kk)]
        So = np.take_along_axis(S, o, 1)
        TOP[qi] = T.rows[o[:, :10]]
        kk = meth.kk
        w = np.exp((So[:, :kk] - So[:, :1]) / meth.T) if meth.synth == "kmean" else np.ones((qi.size, kk))
        w = np.where(So[:, :kk] > M.NEG / 2, w, 0.0)
        w /= w.sum(1, keepdims=True)
        H = np.asarray(act[T.rows[o[:, :kk]], :5, :7], np.float64)          # [n, kk, 5, 7]
        SEG[qi] = (H * w[:, :, None, None]).sum(1)
        # confidence (V4 branch rows)
        Hs = (H / sig).reshape(qi.size, kk, M.NH)
        disp = np.array([M._disp(Hs[i][w[i] > 0]) if (w[i] > 0).sum() > 1 else 0.0 for i in range(qi.size)])
        dnn = np.where(elig, D, np.inf).min(1)
        cmax = np.where(elig, C0, -np.inf).max(1) + np.where(elig, C1, -np.inf).max(1)
        over = step[qi] / meth.medlen[int(t)]
        sc = meth.scales["v4"]
        tot, _ = meth._conf_terms_v4(dnn, disp, over, cmax, sc)
        CONF[qi] = tot / sc["sum_sd"]
        FE["dnn"][qi], FE["disp"][qi], FE["over"][qi], FE["cmax"][qi] = dnn, disp, over, cmax
        if fresh_branch and branch.any():
            fr = np.nonzero(branch)[0]
            j = o[fr, 0]
            c0n = Cc[np.arange(fr.size), j] / meth.s_c
            vis0 = (Z0[fr, j] + Z1[fr, j]) / 2
            scf = meth.scales["fresh"]
            totf, _ = meth._conf_terms_fresh(c0n, disp[fr], vis0, scf)
            CONF[qi[fr]] = totf / scf["sum_sd"]
            FE["c0n"][qi[fr]], FE["vis0"][qi[fr]] = c0n, vis0
    d = (SEG - a_inf) / sig
    err = np.sqrt((d * d).mean(axis=(1, 2)))
    snap = SEG.copy()
    snap[:, :, 6] = np.where(snap[:, :, 6] >= 0, 1.0, -1.0)
    d2 = (snap - a_inf) / sig
    err_snap = np.sqrt((d2 * d2).mean(axis=(1, 2)))
    rep = {"cell": cell, "method": meth.name, "N": int(N), "all": {"err": float(err.mean()), "err_snap": float(err_snap.mean()),
                                                                     "aurc": aurc(CONF, err)}}
    for r, rn in ((0, "step0"), (1, "fresh"), (2, "stale")):
        m = regime == r
        if m.any():
            rep[rn] = {"n": int(m.sum()), "err": float(err[m].mean()), "err_snap": float(err_snap[m].mean()),
                       "aurc": aurc(CONF[m], err[m])}
    if a.npz:
        z = np.load(a.npz)
        rows = z["row"]
        topk = z["topk"]
        same_top = (topk[:, :10] == TOP[rows]).all(1)
        seg = z["synth_seg"].astype(np.float64)
        dseg = np.abs(seg - SEG[rows]).max(axis=(1, 2))
        rep["npz_check"] = {"n": int(rows.size), "topk_equal": float(same_top.mean()), "top1_equal": float((topk[:, 0] == TOP[rows, 0]).mean()),
                            "seg_maxabs": float(dseg.max()), "err_npz": float(z["err"].mean()), "err_ref_same_rows": float(err[rows].mean()),
                            "conf_maxabs": float(np.abs(z["confidence"] - CONF[rows]).max())}
    if a.dump:
        np.savez(a.dump, err=err, err_snap=err_snap, regime=regime, conf=CONF, ep=ep, step=step, **FE)
    print(json.dumps(rep, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
