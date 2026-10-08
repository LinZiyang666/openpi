"""R12 opus (directive 2): per-look feature table on the B-pool dev look recordings (closed loop, deployed system).

For every recorded look (two-camera serving, both keys recorded) the served library of the arm re-serves the state
with several camera configurations, each with its own per-task AWM metric fitted on the served rows (deployed PCA):
  F   [v0, v1, rs]           two fresh cameras (replica of the deployed retrieval)
  W   [v1, rs]               fresh wrist only
  A   [v0, rs]               fresh agentview only
  WLk [v0 from k decisions earlier, v1, rs]   k = 2, 4, 6, 8 (library side lagged identically)
  AL2 [v0, v1 from 2 decisions earlier, rs]   fresh agentview + previous wrist
Stored per look: served (uncorrected) chunks [10, 7] of every configuration, top-1 rows, nearest distances, the
policy's shadow chunk when recorded, recorded hit flag / look reason / guard span, step, task, progress of top-1 rows,
and simple motion cues (change of each camera's PCA code and of rs since the previous look).
Output: out/dev/<root>__<arm>.npz. Library-only fitting; recordings are B-pool dev inits only.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop.look_recording import read_looks
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

RUNS = Path("/home/weiland/trace_runs/os_closed_loop")
ART = Path(__file__).resolve().parents[3] / "r10" / "recipe" / "artifacts"
OUTD = cm.OUT / "dev"
LAGS = (2, 4, 6, 8)
NAMES = ("F", "W", "A", "AL2") + tuple(f"WL{L}" for L in LAGS)
ARMS = [("r12_devlooks_50L", "r11_dev_pi05_l10_50", "pi05", "l10", 50),
        ("r12_devlooks_50L", "r11_dev_groot_l10_50", "groot", "l10", 50),
        ("r12_devlooks_50S", "r11_dev_pi05_spatial_50", "pi05", "spatial", 50),
        ("r12_devlooks_50S", "r11_dev_groot_spatial_50", "groot", "spatial", 50),
        ("r12_devlooks_200", "r11_dev_pi05_l10_200", "pi05", "l10", 200),
        ("r12_devlooks_200", "r11_dev_pi05_spatial_200", "pi05", "spatial", 200),
        ("r12_devlooks_400", "r11_dev_pi05_l10_400", "pi05", "l10", 400),
        ("r12_devlooks_400", "r11_dev_groot_l10_400", "groot", "l10", 400),
        ("r12_devlooks_knob", "r11_devknob_pi05_l10_50_periodic_ir32", "pi05", "l10", 50),
        ("r12_devlooks_knob", "r11_devknob_groot_l10_50_periodic_ir32", "groot", "l10", 50)]


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
    feats = {"F": [P0, P1, rs], "W": [P1, rs], "A": [P0, rs], "AL2": [P0, P1[lag[2]], rs]}
    for L in LAGS:
        feats[f"WL{L}"] = [P0[lag[L]], P1, rs]
    T = {}
    for t in range(10):
        it = np.flatnonzero(task == t)
        d = {"it": it}
        for name, cols in feats.items():
            X = np.concatenate(cols, 1)
            met = cm.Metric(X[it], heads[it], ep[it], step[it])
            d[name] = (met, met.code(X[it]), met.code0(X[it]))
        T[t] = d
    return dict(b=b, rows=rows, act=act, T=T, kref=cm.kref_for(size), prog=C["prog"][rows],
                eplen=C["ep_len"][rows], sig_r=C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0),
                sig=sig, succ=C["success"][rows])


def episodes(arm_dir):
    eps = defaultdict(dict)
    for d in sorted(glob.glob(str(arm_dir / "server_*" / "looks"))):
        for lk in read_looks(d):
            m = lk["meta"]
            eps[(m["uid"], int(m["attempt"]))][int(lk["step"])] = lk
    best = {}
    for uid, att in eps:
        best[uid] = max(best.get(uid, 0), att)
    return {uid: eps[(uid, att)] for uid, att in best.items()}


def run(spec):
    root, arm, model, suite, size = spec
    out_path = OUTD / f"{root}__{arm}.npz"
    if out_path.exists():
        return str(out_path)
    L = fit_library(model, suite, size)
    b = L["b"]
    E = episodes(RUNS / root / "runs" / arm)
    cols = defaultdict(list)
    served = {n: [] for n in NAMES}
    for uid, looks in sorted(E.items()):
        steps = sorted(looks)
        P0s, P1s, RSs = {}, {}, {}
        for s in steps:
            lk = looks[s]
            P0s[s] = (b.B0T @ lk["key_v0"] - b.muB0).astype(np.float64)
            P1s[s] = (b.B1T @ lk["key_v1"] - b.muB1).astype(np.float64)
            RSs[s] = np.asarray(lk["rs"][:8], np.float64)
        prev = None
        for s in steps:
            lk = looks[s]
            t = int(lk["task_id"])
            P0, P1, rs8 = P0s[s], P1s[s], RSs[s]

            def lagged(dct, L):
                k = max(s - L, 0)
                while k not in dct and k > 0:
                    k -= 1
                return dct.get(k, dct[steps[0]])

            q = {"F": [P0, P1, rs8], "W": [P1, rs8], "A": [P0, rs8], "AL2": [P0, lagged(P1s, 2), rs8]}
            for Lg in LAGS:
                q[f"WL{Lg}"] = [lagged(P0s, Lg), P1, rs8]
            it = L["T"][t]["it"]
            for n in NAMES:
                met, Z, Z0 = L["T"][t][n]
                x = np.concatenate(q[n])[None]
                d = core.pdist(met.code0(x), Z0) if s == 0 else core.pdist(met.code(x), Z)
                idx, Dk, w = cm.topk(d, L["kref"])
                served[n].append(np.einsum("qk,qktc->qtc", w, L["act"][it][idx])[0].astype(np.float32))
                cols[f"{n}_top1"].append(int(L["rows"][it][idx[0, 0]]))
                cols[f"{n}_top1_prog"].append(float(L["prog"][it][idx[0, 0]]))
                cols[f"{n}_top1_eplen"].append(int(L["eplen"][it][idx[0, 0]]))
                cols[f"{n}_d1"].append(float(Dk[0, 0]))
                cols[f"{n}_dk"].append(float(Dk[0, min(L['kref'], Dk.shape[1]) - 1]))
                cols[f"{n}_wsucc"].append(float(w[0] @ L["succ"][it][idx[0]]))
            gs = lk["guard_state"] or {}
            ex = gs.get("extras", {}) if isinstance(gs.get("extras"), dict) else {}
            cols["uid"].append(uid)
            cols["step"].append(s)
            cols["task"].append(t)
            cols["hit"].append(bool(lk["hit"]))
            cols["look_reason"].append(float(gs.get("look_reason", np.nan)) if gs.get("look_reason") is not None else np.nan)
            cols["noprog_span"].append(float(ex.get("noprog_span", np.nan)))
            cols["os_reason"].append(float(ex.get("os_reason", np.nan)))
            cols["dP0"].append(float(np.linalg.norm(P0 - P0s[prev])) if prev is not None else np.nan)
            cols["dP1"].append(float(np.linalg.norm(P1 - P1s[prev])) if prev is not None else np.nan)
            cols["drs"].append(float(np.linalg.norm(rs8 - RSs[prev])) if prev is not None else np.nan)
            sh = lk.get("shadow_policy")
            cols["shadow"].append(np.asarray(sh)[:10, :7].astype(np.float32) if sh is not None
                                  else np.full((10, 7), np.nan, np.float32))
            cols["executed"].append(np.asarray(lk["action"])[:10, :7].astype(np.float32))
            prev = s
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    for n in NAMES:
        arrays[f"S_{n}"] = np.asarray(served[n])
    meta = dict(root=root, arm=arm, model=model, suite=suite, size=size, kref=L["kref"], sig_r=L["sig_r"].tolist(),
                sig=L["sig"].tolist(), n_episodes=len(E))
    OUTD.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_path)
    return str(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    specs = [s for s in ARMS if not a.only or s[1] in a.only.split(",")]
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for p in pool.imap_unordered(run, specs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
