"""Fit the model-space <-> wire action map and the wire-action -> end-effector displacement map (inits 0-19 only).

Data: R8 pure-cache debug arms.  Per episode, the client ``episode.json`` identity is read first; episodes with init >=
20 are skipped before any control array is opened (RULE 1 + disjoint fit).  Served model-space actions come from the
standard server logs (``served_head``: the 5 executed rows of each decision), admitted through the uid filter.

Outputs ``out/action_map_<model>.json``: per action dim d the affine  wire_d = a_d * model_d + b_d  (max residual), and
the 3x3 least-squares map  delta_eef_pos(control) ~= M @ wire[0:3]  with its R^2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, iter_jsonl_discovery, parse_uid, dump

HERE = Path(__file__).resolve().parents[1]
ARMS = {"pi05": "r8_pi05_l10_50_A", "groot": "r8_groot_l10_50_A"}
FIT_MAX_INIT = 20


def served_heads(arm):
    out = {}
    for path in sorted((RUNS / "r08_main" / "runs" / arm).glob("server_*/decisions*.jsonl")):
        for r in iter_jsonl_discovery(path, uid_key="uid"):
            if r.get("ev") != "dec" or r.get("served_head") is None:
                continue
            t, i = parse_uid(r["uid"])
            if i >= FIT_MAX_INIT:
                continue
            out[(t, i, int(r.get("attempt", 1)), int(r["step"]))] = np.asarray(r["served_head"], float)
    return out


def fit(model, max_episodes=120):
    arm = ARMS[model]
    heads = served_heads(arm)
    X, Y, dP, A = [], [], [], []
    n = 0
    for ep_dir in sorted((RUNS / "r08_main" / "runs" / arm / "debug" / "client").iterdir()):
        meta = json.loads((ep_dir / "episode.json").read_text())
        t, i = parse_uid(meta["task_uid"])
        if i >= FIT_MAX_INIT:
            continue                          # identity checked before any array is read
        att = int(meta.get("attempt", 1))
        parts = [np.load(p) for p in sorted(ep_dir.glob("controls_*.npz"))]
        c = {k: np.concatenate([p[k] for p in parts]) for k in ("action", "decision_seq", "chunk_offset", "eef_pos", "is_settle")}
        for j in range(len(c["action"])):
            if c["is_settle"][j]:
                continue
            key = (t, i, att, int(c["decision_seq"][j]))
            off = int(c["chunk_offset"][j]) % 5
            if key in heads and off < len(heads[key]):
                X.append(heads[key][off])
                Y.append(c["action"][j][:7])
            if j > 0 and not c["is_settle"][j - 1]:
                dP.append(c["eef_pos"][j] - c["eef_pos"][j - 1])
                A.append(c["action"][j][:3])
        n += 1
        if n >= max_episodes:
            break
    X, Y, dP, A = map(np.asarray, (X, Y, dP, A))
    aff = []
    for d in range(7):
        a, b = np.polyfit(X[:, d], Y[:, d], 1)
        res = np.abs(Y[:, d] - (a * X[:, d] + b)).max()
        aff.append(dict(a=float(a), b=float(b), max_abs_residual=float(res)))
    M, *_ = np.linalg.lstsq(A, dP, rcond=None)
    pred = A @ M
    r2 = 1 - ((dP - pred) ** 2).sum(0) / ((dP - dP.mean(0)) ** 2).sum(0)
    out = dict(model=model, arm=arm, fit_inits=f"0-{FIT_MAX_INIT - 1}", episodes=n, pairs=int(len(X)), steps=int(len(A)),
               affine=aff, M=M.T.tolist(), r2=r2.tolist(),
               wire_pos_p95=np.quantile(np.abs(Y[:, :3]), .95, axis=0).tolist(),
               model_pos_p95=np.quantile(np.abs(X[:, :3]), .95, axis=0).tolist())
    dump(HERE / "out" / f"action_map_{model}.json", out)
    return out


if __name__ == "__main__":
    for m in ARMS:
        o = fit(m)
        print(m, json.dumps({k: o[k] for k in ("episodes", "pairs", "steps", "r2")}))
        for d, f in enumerate(o["affine"]):
            print("  dim", d, f)
        print("  M (rows: dx,dy,dz per wire x,y,z):", np.round(np.asarray(o["M"]), 5).tolist())
