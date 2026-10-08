"""R12 opus: what should the layer-3 corrector see on a half look? (library only, k-fold by episode)

The deployed corrector head takes [PCA64 v0, PCA64 v1, rs8, served chunk/sigma, step, task one-hot]. On a wrist-only
look v0 is missing. Per fold and task a LOEO head is trained exactly like R10 (training-fold rows, regime-2 LOEO
synthesis with the F metric, 384 RFF, ridge 100, episode weights) and applied to held-out looks:
  F_true   F served chunk, true v0            (the deployed two-camera answer, reference)
  W_none   W served chunk, no correction
  W_true   W served chunk, true v0            (oracle: what the corrector would do with the real agentview)
  W_stale  W served chunk, v0 from the previous look (10 controls old)
  W_imp    W served chunk, v0 = kernel-weighted mean v0 of W's own members (library rows)
  W_zero   W served chunk, v0 = training mean (uninformative)
Correction blend .5 on motion channels, 10 steps (distance attenuation omitted: identical across options).
Reported: mean e10 of the corrected chunk vs the row's own policy chunk (look rows, step > 0).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

OUTD = cm.OUT / "corrector"


def run_job(job):
    model, suite, size, fold = job
    C = cm.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, size)
    tr = fold_of_row != fold
    P0, P1, _ = cm.projections(model, suite, size, fold, rows, tr, keep=128)
    P0, P1 = P0[:, :64].astype(np.float64), P1[:, :64].astype(np.float64)
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    lag2 = cm.lag_index(C, rows, 2)
    sig_m = act[tr][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
    kref = cm.kref_for(size)
    XF = np.concatenate([P0, P1, rs], 1)
    XW = np.concatenate([P1, rs], 1)
    P = np.concatenate([P0, P1], 1)
    res = {k: [] for k in ("F_none", "F_true", "W_none", "W_true", "W_stale", "W_imp", "W_zero", "W_hh",
                           "WL_none", "WL_stale", "WL_hh")}
    for t in range(10):
        itr = np.flatnonzero(tr & (task == t))
        ite = np.flatnonzero(~tr & (task == t) & (step % 2 == 0) & (step > 0))
        if not len(ite):
            continue
        mF = core.TaskMetric(XF[itr], heads[itr], ep[itr], step[itr])
        mW = core.TaskMetric(XW[itr], heads[itr], ep[itr], step[itr])
        s_tr, *_ = core.serve(XF[itr], step[itr], ep[itr], mF, XF[itr], None, ep[itr], act[itr], kref)
        Xh = core.features(P[itr], rs[itr], s_tr, step[itr], task[itr], sig_m)
        head = core.Head(Xh, core.residual(act[itr], s_tr, sig_m), core.episode_weights(ep[itr]), total=len(itr))
        sF, *_ = core.serve(XF[ite], step[ite], ep[ite], mF, XF[itr], None, ep[itr], act[itr], kref, exclude_own=False)
        sW, _, _, memW, wW = core.serve(XW[ite], step[ite], ep[ite], mW, XW[itr], None, ep[itr], act[itr], kref,
                                        exclude_own=False)
        P0imp = np.einsum("qk,qkd->qd", wW, P0[itr][memW])
        XWL = np.concatenate([P0[lag2], P1, rs], 1)
        mWL = core.TaskMetric(XWL[itr], heads[itr], ep[itr], step[itr])
        sWL, *_ = core.serve(XWL[ite], step[ite], ep[ite], mWL, XWL[itr], None, ep[itr], act[itr], kref, exclude_own=False)
        # half head: the same recipe trained on W's own LOEO synthesis with wrist-only features
        sW_tr, *_ = core.serve(XW[itr], step[itr], ep[itr], mW, XW[itr], None, ep[itr], act[itr], kref)
        hh = core.Head(core.features(P1[itr], rs[itr], sW_tr, step[itr], task[itr], sig_m),
                       core.residual(act[itr], sW_tr, sig_m), core.episode_weights(ep[itr]), total=len(itr))
        sig6 = sig_m[:6]

        def corrected(s, p0):
            Pq = np.concatenate([p0, P1[ite]], 1)
            c = head.predict(core.features(Pq, rs[ite], s, step[ite], task[ite], sig_m)).reshape(-1, 10, 6) * sig6
            out = s.copy()
            out[:, :10, :6] += 0.5 * c
            return core.motion_err(act[ite], out, sig_r)

        res["F_none"].append(core.motion_err(act[ite], sF, sig_r))
        res["F_true"].append(corrected(sF, P0[ite]))
        res["W_none"].append(core.motion_err(act[ite], sW, sig_r))
        res["W_true"].append(corrected(sW, P0[ite]))
        res["W_stale"].append(corrected(sW, P0[lag2[ite]]))
        res["W_imp"].append(corrected(sW, P0imp))
        res["W_zero"].append(corrected(sW, np.broadcast_to(P0[itr].mean(0), P0[ite].shape)))
        ch = hh.predict(core.features(P1[ite], rs[ite], sW, step[ite], task[ite], sig_m)).reshape(-1, 10, 6) * sig6
        o = sW.copy()
        o[:, :10, :6] += 0.5 * ch
        res["W_hh"].append(core.motion_err(act[ite], o, sig_r))
        res["WL_none"].append(core.motion_err(act[ite], sWL, sig_r))
        res["WL_stale"].append(corrected(sWL, P0[lag2[ite]]))
        ch2 = hh.predict(core.features(P1[ite], rs[ite], sWL, step[ite], task[ite], sig_m)).reshape(-1, 10, 6) * sig6
        o2 = sWL.copy()
        o2[:, :10, :6] += 0.5 * ch2
        res["WL_hh"].append(core.motion_err(act[ite], o2, sig_r))
    return job, {k: np.concatenate(v) for k, v in res.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05:l10:50,groot:l10:50,pi05:spatial:50,groot:spatial:50,pi05:l10:200,groot:l10:200")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    cells = [(s.split(":")[0], s.split(":")[1], int(s.split(":")[2])) for s in a.cells.split(",")]
    jobs = [(m, s, n, f) for m, s, n in cells for f in range(5)]
    acc = {}
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for job, r in pool.imap_unordered(run_job, jobs):
            key = f"{job[0]}_{job[1]}_{job[2]}"
            for k, v in r.items():
                acc.setdefault(key, {}).setdefault(k, []).append(v)
    out = {}
    for key in sorted(acc):
        d = {k: np.concatenate(v) for k, v in acc[key].items()}
        base = d["F_true"]
        out[key] = {k: dict(e10=float(v.mean()), d_vs_Ftrue=float((v - base).mean()),
                            se=float((v - base).std() / np.sqrt(len(v)))) for k, v in d.items()}
        print(f"\n== {key} (n={len(base)})")
        for k, v in out[key].items():
            print(f"{k:8s} e10 {v['e10']:.4f}  vs F_true {v['d_vs_Ftrue']:+.4f} (se {v['se']:.4f})")
    OUTD.mkdir(parents=True, exist_ok=True)
    cm.save_json(out, OUTD / "corrector_check_wl.json")


if __name__ == "__main__":
    main()
