"""R12 opus: full vs one-camera retrieval on CLOSED-LOOP cache-driven states (R6 B-val recordings, non-test).

Source: /home/weiland/trace_runs/os_closed_loop/r06_c_cal (BRIEF section 5: allowed). These are pure-cache (R6 A,
anchor-tail blind block) episodes on one B-val init per task, with both raw camera keys at every look. The B-val
init's own library episode is excluded from the candidates (R6 exclusion list). Nothing from test set A is read.

For each look of each recording, the R10 subset library of the same cell (size 50 / 500) serves the state with the
feature-set variants of replay.py (own per-task metric fitted on ALL subset rows, PCA = the deployed R10Recipe
artifact basis). Reported: served-chunk difference to F (dF), top-1 agreement, kernel overlap, and the same numbers
on library replay for comparison (from out/replay_summary.json). No policy chunk exists for these states, so no
error-vs-policy is computed here.
"""
from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

R6 = Path("/home/weiland/trace_runs/os_closed_loop/r06_c_cal")
ART = Path(__file__).resolve().parents[3] / "r10" / "recipe" / "artifacts"
EXCL = {"l10": {8, 58, 104, 152, 209, 259, 311, 350, 409, 452},
        "spatial": {9, 56, 105, 150, 211, 259, 309, 350, 409, 452}}


def deployed_pca(model, suite, size):
    r = pickle.load(open(ART / f"r10_recipe_{model}_{suite}_{size}.pkl", "rb"))["method"]
    b = r.inner.base
    return (b.B0T, b.muB0), (b.B1T, b.muB1), np.asarray(r.row_subset)


def run(model, suite, size):
    C = cm.load_cell(model, suite)
    (B0T, m0), (B1T, m1), rows = deployed_pca(model, suite, size)
    keep = ~np.isin(C["ep"][rows], list(EXCL[suite]))
    rows = rows[keep]
    K0, K1 = core.gather_keys(model, suite, rows)
    P0 = (K0 @ B0T.T - m0).astype(np.float64)
    P1 = (K1 @ B1T.T - m1).astype(np.float64)
    del K0, K1
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    lag2 = cm.lag_index(C, rows, 2)
    sig = act[:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    heads = (act[:, :5, :7] / sig).reshape(len(act), -1)
    kref = cm.kref_for(size)
    feats = {"F": lambda a, b, la, s: [a, b, s], "W": lambda a, b, la, s: [b, s], "A": lambda a, b, la, s: [a, s],
             "R": lambda a, b, la, s: [s], "WL": lambda a, b, la, s: [la, b, s]}
    mets = {}
    for t in range(10):
        it = np.flatnonzero(task == t)
        for v, f in feats.items():
            X = np.concatenate(f(P0, P1, P0[lag2], rs), 1)
            mets[(t, v)] = (cm.Metric(X[it], heads[it], ep[it], step[it]), X[it], it)
    cell = f"{model}_{suite}_{size}"
    arm = R6 / "runs" / f"r6c_cal_{cell}"
    out = {v: dict(dF=[], top1=[], ovl=[]) for v in feats}
    cw = dict(score=[], dF=[])
    n_look = 0
    for fpath in sorted(glob.glob(str(arm / "server_*" / "inputs" / "*.npz"))):
        z = np.load(fpath)
        meta = json.loads(str(z["meta"]))
        t = int(meta["task_id"])
        vis = np.flatnonzero(z["vision"])
        k0 = z["key_v0"][vis].astype(np.float32)
        k1 = z["key_v1"][vis].astype(np.float32)
        q0 = (k0 @ B0T.T - m0).astype(np.float64)
        q1 = (k1 @ B1T.T - m1).astype(np.float64)
        qrs = z["rs"][vis][:, :8].astype(np.float64)
        st = z["step"][vis]
        ql0 = np.concatenate([q0[:1], q0[:-1]])            # previous look's agentview (first look: itself)
        res = {}
        for v, f in feats.items():
            met, Xc, it = mets[(t, v)]
            Xq = np.concatenate(f(q0, q1, ql0, qrs), 1)
            Zc, Zc0, Zq, Zq0 = met.code(Xc), met.code0(Xc), met.code(Xq), met.code0(Xq)
            D = core.pdist(Zq, Zc)
            s0 = st == 0
            if s0.any():
                D[s0] = core.pdist(Zq0[s0], Zc0)
            idx, Dk, w = cm.topk(D, kref)
            res[v] = (np.einsum("qk,qktc->qtc", w, act[it][idx]), it[idx], w)
        nz = st > 0
        n_look += int(nz.sum())
        sW, sWL, sF = res["W"][0], res["WL"][0], res["F"][0]
        sc = cm.motion_err(sW, sWL, sig) + np.mean(np.sign(sW[:, :5, 6]) != np.sign(sWL[:, :5, 6]), axis=1)
        cw["score"].append(sc[nz]); cw["dF"].append(cm.motion_err(sF, sW, sig)[nz])
        for v in feats:
            s, mem, w = res[v]
            out[v]["dF"].append(cm.motion_err(res["F"][0], s, sig)[nz])
            out[v]["top1"].append((mem[:, 0] == res["F"][1][:, 0])[nz])
            out[v]["ovl"].append(np.array([w[i][np.isin(mem[i], res["F"][1][i])].sum() for i in range(len(w))])[nz])
    summ = {v: {k: float(np.mean(np.concatenate(a))) for k, a in d.items()} for v, d in out.items()}
    summ["_n_looks"] = n_look
    sc, dF = np.concatenate(cw["score"]), np.concatenate(cw["dF"])
    summ["_cw"] = {str(t): dict(half=float((sc <= t).mean()), dF_half=float(dF[sc <= t].mean()) if (sc <= t).any() else None,
                                dF_rest=float(dF[sc > t].mean()) if (sc > t).any() else None) for t in (0.1, 0.15, 0.2, 0.3)}
    return summ


def main():
    lib = json.loads((cm.OUT / "replay_summary.json").read_text())
    res = {}
    for model in ("pi05", "groot"):
        for suite in ("l10", "spatial"):
            for size in (50, 500):
                cell = f"{model}_{suite}_{size}"
                s = run(model, suite, size)
                res[cell] = s
                print(f"\n== {cell}: closed-loop looks {s['_n_looks']} (R6 B-val, pure cache)")
                print("var   CL dF  lib dF | CL top1 lib top1 | CL ovl lib ovl")
                for v in ("W", "A", "R", "WL"):
                    L = lib.get(cell, {}).get(v, {})
                    print(f"{v:4s} {s[v]['dF']:6.3f} {L.get('dF', float('nan')):6.3f} | {s[v]['top1']:6.3f} "
                          f"{L.get('top1', float('nan')):6.3f} | {s[v]['ovl']:6.3f} {L.get('ovl', float('nan')):6.3f}")
                for t, d in s["_cw"].items():
                    print(f"  cw tau {t}: half {d['half']:.3f} dF(half) {d['dF_half']} dF(rest) {d['dF_rest']}")
    cm.save_json(res, cm.OUT / "cl_agreement_r6.json")


if __name__ == "__main__":
    main()
