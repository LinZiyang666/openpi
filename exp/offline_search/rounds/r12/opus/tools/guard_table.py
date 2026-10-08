"""R12 opus (guard task): per-look guard dataset on the B-pool dev recordings (closed loop, deployed system).

For every recorded look: outcome labels and candidate guard signals that do NOT need the agentview.
Labels
  called / os_reason      the deployed decision (4 = no-progress guard, 62 = periodic knob)
  D                       divergence between the cache chunk (two-camera retrieval F, uncorrected kernel mean) and the
                          policy's own shadow chunk at this state (motion RMS, sigma units; 50L arms only): how much a
                          policy call would change what is executed
  fail                    the episode failed; pure_ok: the pure policy succeeds from this (task, init) (library
                          rollout of the same B init) -> "rescuable" = fail and pure_ok
Signals (computed from the episode's own history up to this look)
  speed, speed2           |delta rs[:3]| over the last 10 controls / mean of the last two intervals (rs = valid state)
  rot                     |delta rs[3:6]|
  gripw, dgrip            gripper opening (rs[6] - rs[7]) and its change
  mismatch                |realised delta rs[:3] - predicted from the commanded actions| / library scale
                          (ridge map fitted on the B library: delta rs over two decisions vs the two executed heads)
  osc                     xyz direction reversals over the last three intervals
  cmdgrip_incons          commanded gripper (served head) disagrees with the current opening trend
  W_adv                   library-progress advance of the wrist-only top-1 vs the previous look's served top-1
  W_d1rel                 wrist-only nearest distance / its arm median
  lagrel                  (elapsed step - library step of the wrist-only top-1) / its episode length
  cons                    |W - WL2| + gripper-sign disagreement (needs the remembered agentview)
  step, prevcall
Output: out/guard/<arm>.npz
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop.look_recording import read_looks
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load, score
from exp.offline_search.rounds.r12.opus.tools.dev_features import ARMS, RUNS

OUTD = cm.OUT / "guard"


def mismatch_map(model, suite):
    """Ridge map: delta rs[:3] over two decisions ~ [sum of row r's executed head, sum of row r+1's executed head]."""
    C = cm.load_cell(model, suite)
    d = core.lib_dir(model, suite)
    nxt = np.load(d / "next.npy").astype(np.int64)
    act = C["act"]
    rs = C["rs"]
    r = np.flatnonzero(nxt >= 0)
    r = r[nxt[r] >= 0]
    r2 = nxt[r]
    ok = nxt[r2] >= 0
    r, r2 = r[ok], r2[ok]
    r3 = nxt[r2]
    X = np.c_[np.ones(len(r)), act[r, :5, :3].sum(1), act[r2, :5, :3].sum(1), rs[r, :8]]
    Y = rs[r3, :3] - rs[r, :3]
    lam = 1e-3 * np.trace(X.T @ X) / X.shape[1]
    B = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    res = np.linalg.norm(Y - X @ B, axis=1)
    return B, float(np.median(res)) + 1e-9


def pure_outcomes(model, suite):
    d = core.lib_dir(model, suite)
    eps = json.load(open(d / "episodes.json"))
    out = {}
    for e in eps:
        init = e.get("orig_init_state_idx")
        init = int(e["episode_id"]) % 50 if init is None else int(init)     # groot spatial: init = id % 50
        out[(int(e["task_id"]), init)] = bool(e["success"])
    return out


def journal(arm_dir):
    out = {}
    for line in open(arm_dir / "client" / "journal.jsonl"):
        r = json.loads(line)
        if r.get("accepted", True):
            out[r["task_uid"]] = bool(r["success"])
    return out


def run(spec):
    root, arm, model, suite, size = spec
    out_path = OUTD / f"{arm}.npz"
    if out_path.exists():
        return str(out_path)
    dev_p = cm.OUT / "dev" / f"{root}__{arm}.npz"
    D, meta = load(dev_p)
    sig = np.asarray(meta["sig_r"])
    idx = {(u, int(s)): i for i, (u, s) in enumerate(zip(D["uid"], D["step"]))}
    B, scale = mismatch_map(model, suite)
    pure = pure_outcomes(model, suite)
    succ = journal(RUNS / root / "runs" / arm)
    eps = defaultdict(dict)
    for dd in sorted(glob.glob(str(RUNS / root / "runs" / arm / "server_*" / "looks"))):
        for lk in read_looks(dd):
            m = lk["meta"]
            eps[(m["uid"], int(m["attempt"]))][int(lk["step"])] = dict(
                rs=np.asarray(lk["rs"][:8], np.float64), act=np.asarray(lk["action"])[:10, :7].astype(np.float64),
                task=int(lk["task_id"]), init=int(lk["init"]) if lk["init"] is not None else -1)
    best = {}
    for u, a in eps:
        best[u] = max(best.get(u, 0), a)
    R = defaultdict(list)
    medW = np.median(D["W_d1"])
    for u, a in best.items():
        looks = eps[(u, a)]
        steps = sorted(looks)
        hist = []
        prev_served_prog = None
        for k, s in enumerate(steps):
            i = idx.get((u, s))
            if i is None:
                continue
            L = looks[s]
            rs = L["rs"]
            hist.append(rs)
            dv = [np.linalg.norm(hist[j][:3] - hist[j - 1][:3]) for j in range(max(1, len(hist) - 3), len(hist))]
            vec = [hist[j][:3] - hist[j - 1][:3] for j in range(max(1, len(hist) - 3), len(hist))]
            osc = sum(float(np.dot(vec[j], vec[j - 1]) < 0) for j in range(1, len(vec)))
            mis = np.nan
            if k >= 1 and (s - steps[k - 1]) == 2:
                pa = looks[steps[k - 1]]["act"]
                x = np.r_[1.0, pa[:5, :3].sum(0), pa[5:10, :3].sum(0), looks[steps[k - 1]]["rs"]]
                mis = float(np.linalg.norm(rs[:3] - looks[steps[k - 1]]["rs"][:3] - x @ B)) / scale
            gw = rs[6] - rs[7]
            gw_prev = (hist[-2][6] - hist[-2][7]) if len(hist) >= 2 else gw
            cmd = float(np.mean(np.sign(L["act"][:5, 6])))
            prog_W = float(D["W_top1_prog"][i]); el_W = int(D["W_top1_eplen"][i])
            R["uid"].append(u); R["step"].append(s); R["task"].append(L["task"])
            R["called"].append(not bool(D["hit"][i])); R["os_reason"].append(float(D["os_reason"][i]))
            R["fail"].append(not succ.get(u, True)); R["pure_ok"].append(pure.get((L["task"], L["init"]), True))
            sh = D["shadow"][i]
            R["D"].append(float(cm.motion_err(sh, D["S_F"][i], sig)) if np.isfinite(sh).all() else np.nan)
            R["speed"].append(dv[-1] if dv else np.nan); R["speed2"].append(float(np.mean(dv[-2:])) if dv else np.nan)
            R["rot"].append(float(np.linalg.norm(rs[3:6] - hist[-2][3:6])) if len(hist) >= 2 else np.nan)
            R["gripw"].append(gw); R["dgrip"].append(gw - gw_prev)
            R["mismatch"].append(mis); R["osc"].append(osc)
            R["cmdgrip_incons"].append(float(cmd * np.sign(gw - gw_prev + 1e-12) > 0.5 and abs(gw - gw_prev) > 1e-3))
            R["W_adv"].append((prog_W - prev_served_prog) * max(el_W - 1, 1) if prev_served_prog is not None else np.nan)
            R["W_d1rel"].append(float(D["W_d1"][i] / medW))
            R["lagrel"].append((s - prog_W * max(el_W - 1, 1)) / max(el_W, 1))
            R["cons"].append(score(D["S_W"][i], D["S_WL2"][i], sig) if k >= 1 else np.nan)
            R["prevcall"].append(bool(k >= 1 and R["called"][-2]) if k >= 1 else False)
            R["F_adv"].append((float(D["F_top1_prog"][i]) - prev_served_prog) * max(int(D["F_top1_eplen"][i]) - 1, 1)
                              if prev_served_prog is not None else np.nan)
            prev_served_prog = float(D["F_top1_prog"][i])
    arrays = {k: np.asarray(v) for k, v in R.items()}
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta_json=np.array(json.dumps(dict(meta, mismatch_scale=scale))), **arrays)
    return str(out_path)


if __name__ == "__main__":
    import multiprocessing as mp
    with mp.get_context("fork").Pool(4, maxtasksperchild=1) as pool:
        for p in pool.imap_unordered(run, ARMS):
            print("done", p, flush=True)
