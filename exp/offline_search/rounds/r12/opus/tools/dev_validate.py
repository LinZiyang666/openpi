"""R12 opus: post-freeze validation of the frozen cascade on CLOSED-LOOP dev looks (B-pool dev inits only).

Source: the coordinator's dev recordings `os_closed_loop/r12_devlooks_*` (deployed R10Recipe, knob off, unchanged
two-camera serving, every look recorded with both keys; L10-50 also with the policy's shadow chunk). Read only after
PREDICTION.md was frozen (12:15 CDT); nothing here changes SPEC (tau, metrics and heads stay library-only).

Library side exactly as SPEC 2.1: the deployed R10 artifact's PCA bases and row subset; per-task AWM metrics for
F = [v0, v1, rs], W = [v1, rs], WL_L = [v0 lagged L, v1, rs] (L = 2, 4, 6, 8) fitted on the served library rows.
For every recorded episode the cascade's camera decision is replayed look by look on the recorded states
(look 0 full; lag = decisions since the last fresh agentview: step 0, a completion, or a recorded policy call;
lag > 8 -> complete; score = |W - WL_lag| + first-5-step gripper-sign disagreement; half iff score <= tau(N)).
Reported: one-camera share, served-chunk difference to F on one-camera looks, and (with shadow chunks) the paired
error of W vs F against the policy's own chunk at the visited state (the closed-loop counterpart of dq).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop.look_recording import read_looks
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

RUNS = Path("/home/weiland/trace_runs/os_closed_loop")
ART = Path(__file__).resolve().parents[3] / "r10" / "recipe" / "artifacts"
LAGS = (2, 4, 6, 8)


def fit_library(model, suite, size):
    r = pickle.load(open(ART / f"r10_recipe_{model}_{suite}_{size}.pkl", "rb"))["method"]
    b = r.inner.base
    rows = np.asarray(r.row_subset)
    C = cm.load_cell(model, suite)
    K0, K1 = core.gather_keys(model, suite, rows)
    P0 = (K0 @ b.B0T.T - b.muB0).astype(np.float64)
    P1 = (K1 @ b.B1T.T - b.muB1).astype(np.float64)
    del K0, K1
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    sig = act[:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    heads = (act[:, :5, :7] / sig).reshape(len(act), -1)
    lag = {L: cm.lag_index(C, rows, L) for L in LAGS}
    T = {}
    for t in range(10):
        it = np.flatnonzero(task == t)
        d = {"it": it}
        for name, X in [("F", np.concatenate([P0, P1, rs], 1)), ("W", np.concatenate([P1, rs], 1))] + \
                       [(f"WL{L}", np.concatenate([P0[lag[L]], P1, rs], 1)) for L in LAGS]:
            met = cm.Metric(X[it], heads[it], ep[it], step[it])
            d[name] = (met, met.code(X[it]), met.code0(X[it]))
        T[t] = d
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    return dict(b=b, rows=rows, act=act, T=T, kref=cm.kref_for(size), sig_r=sig_r, size=size, parent_rows=rows)


def serve(L, t, name, x, step0):
    met, Z, Z0 = L["T"][t][name]
    it = L["T"][t]["it"]
    if step0:
        d = core.pdist(met.code0(x[None]), Z0)
    else:
        d = core.pdist(met.code(x[None]), Z)
    idx, Dk, w = cm.topk(d, L["kref"])
    s = np.einsum("qk,qktc->qtc", w, L["act"][it][idx])[0]
    return s, L["rows"][it][idx[0][0]]


def episodes(arm_dir):
    eps = defaultdict(dict)
    for d in sorted(glob.glob(str(arm_dir / "server_*" / "looks"))):
        for lk in read_looks(d):
            m = lk["meta"]
            key = (m["uid"], int(m["attempt"]))
            eps[key][int(lk["step"])] = lk
    # keep the latest attempt per uid
    best = {}
    for (uid, att) in eps:
        best[uid] = max(best.get(uid, 0), att)
    return {uid: eps[(uid, att)] for uid, att in best.items()}


_CACHE = {}


def run(root, arm, model, suite, size, tau=None):
    key = (model, suite, size)
    if key not in _CACHE:
        _CACHE[key] = fit_library(model, suite, size)
    L = _CACHE[key]
    b, sig_r = L["b"], L["sig_r"]
    tau = 0.15 * np.sqrt(50 / size) if tau is None else tau
    ek = (root, arm)
    if ek not in _CACHE:
        _CACHE[ek] = episodes(RUNS / root / "runs" / arm)
    E = _CACHE[ek]
    rec = []
    agree_top1 = []
    for uid, looks in E.items():
        steps = sorted(looks)
        last = None                                    # (step, P0) of the last fresh agentview
        for s in steps:
            lk = looks[s]
            t = int(lk["task_id"])
            P0 = (b.B0T @ lk["key_v0"] - b.muB0).astype(np.float64)
            P1 = (b.B1T @ lk["key_v1"] - b.muB1).astype(np.float64)
            rs8 = np.asarray(lk["rs"][:8], np.float64)
            sF, top1F = serve(L, t, "F", np.concatenate([P0, P1, rs8]), s == 0)
            if bool(lk["hit"]) and s > 0 and lk["guard_state"].get("look_reason") != 6:
                agree_top1.append(int(top1F) == int(lk["topk"][0]))
            if s == 0 or last is None:
                last = (s, P0)
                continue
            lag = s - last[0]
            sW, _ = serve(L, t, "W", np.concatenate([P1, rs8]), False)
            if lag > 8:
                kind, score = "comp_lag", np.nan
            else:
                Lq = min(LAGS, key=lambda x: abs(x - lag))
                sWL, _ = serve(L, t, f"WL{Lq}", np.concatenate([last[1], P1, rs8]), False)
                score = float(cm.motion_err(sW, sWL, sig_r)) + float(np.mean(np.sign(sW[:5, 6]) != np.sign(sWL[:5, 6])))
                kind = "half" if score <= tau else "comp"
            r = dict(uid=uid, step=s, kind=kind, score=score, lag=lag, hit=bool(lk["hit"]),
                     dF=float(cm.motion_err(sF, sW, sig_r)))
            sh = lk.get("shadow_policy")
            if sh is not None and np.isfinite(sh).all():
                pol = np.asarray(sh)[:10, :7]
                r["eW"] = float(cm.motion_err(pol, sW[:10, :7], sig_r))
                r["eF"] = float(cm.motion_err(pol, sF[:10, :7], sig_r))
            rec.append(r)
            if kind != "half" or not bool(lk["hit"]):
                last = (s, P0)                         # completion or a recorded policy call refreshes the agentview
    kinds = np.array([r["kind"] for r in rec])
    half = kinds == "half"
    out = dict(root=root, arm=arm, tau=float(tau), n_episodes=len(E), n_looks=len(rec),
               half_share=float(half.mean()), comp_consistency=float((kinds == "comp").mean()),
               comp_lag=float((kinds == "comp_lag").mean()),
               dF_half=float(np.mean([r["dF"] for r in rec if r["kind"] == "half"])) if half.any() else None,
               dF_rest=float(np.mean([r["dF"] for r in rec if r["kind"] != "half"])),
               top1_F_vs_deployed=float(np.mean(agree_top1)) if agree_top1 else None)
    if all("eW" in r for r in rec):
        dq = np.array([(r["eW"] - r["eF"]) if r["kind"] == "half" else 0.0 for r in rec])
        allw = np.array([r["eW"] - r["eF"] for r in rec])
        out.update(dq_closed_loop=float(dq.mean()), dq_se=float(dq.std() / np.sqrt(len(dq))),
                   all_wrist_dq=float(allw.mean()), all_wrist_dq_se=float(allw.std() / np.sqrt(len(allw))),
                   eF_mean=float(np.mean([r["eF"] for r in rec])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="")
    ap.add_argument("--arms", default="r12_devlooks_50L:r11_dev_pi05_l10_50:pi05:l10:50,"
                                      "r12_devlooks_50S:r11_dev_pi05_spatial_50:pi05:spatial:50,"
                                      "r12_devlooks_50S:r11_dev_groot_spatial_50:groot:spatial:50")
    a = ap.parse_args(argv)
    res = []
    for spec in a.arms.split(","):
        root, arm, m, s, n = spec.split(":")
        r = run(root, arm, m, s, int(n))
        res.append(r)
        print(json.dumps(r), flush=True)
        for t in [float(x) for x in a.taus.split(",") if x]:
            r2 = run(root, arm, m, s, int(n), tau=t)
            res.append(r2)
            print("  tau", t, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r2.items()
                               if k in ("half_share", "dF_half", "dq_closed_loop", "dq_se")}, flush=True)
    cm.save_json(res, cm.OUT / ("dev_validate.json" if not a.taus else "dev_validate_taus.json"))


if __name__ == "__main__":
    main()
