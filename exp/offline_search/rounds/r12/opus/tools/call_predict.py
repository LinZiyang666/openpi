"""R12 opus (directive 2): predict, from a one-camera look, whether the two-camera guard would call the policy.

Closed-loop dev looks (deployed system, two cameras; recorded call = the deployed no-progress guard fired on F; my F
replica reproduces every recorded call). Features available at a wrist-first look before any agentview encode,
with the previous look's served top-1 progress p_prev as reference:
  dW   = (prog(W top-1) - p_prev) * (eplen - 1)          wrist-only retrieval
  dWL  = (prog(WL_2 top-1) - p_prev) * (eplen - 1)       wrist + remembered agentview
  drs  = |rs_now - rs_prev| (robot-state motion since the previous look), prevcall (previous look called)
Rule family (suspicion -> complete the agentview and let the deployed guard decide on F):
  flag if min(dW, dWL) <= .5 + m   or   (prevcall and c_prev)   or   drs <= r
Reported: recall of the deployed calls (share of calls that would still happen) and the share of non-call looks that
would still be served with one camera (before the separate consistency test).
"""
from __future__ import annotations

import glob
from collections import defaultdict

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load


def table(d):
    by = defaultdict(list)
    for i, u in enumerate(d["uid"]):
        by[u].append(i)
    R = defaultdict(list)
    for u, idx in by.items():
        idx = sorted(idx, key=lambda i: d["step"][i])
        for a, b in zip(idx[:-1], idx[1:]):
            pp = float(d["F_top1_prog"][a])
            for n in ("W", "WL2", "A", "F"):
                R[f"d{n}"].append((float(d[f"{n}_top1_prog"][b]) - pp) * max(int(d[f"{n}_top1_eplen"][b]) - 1, 1))
            R["drs"].append(float(d["drs"][b])); R["dP1"].append(float(d["dP1"][b]))
            R["prevcall"].append(not bool(d["hit"][a])); R["called"].append(not bool(d["hit"][b]))
            R["step"].append(int(d["step"][b]))
    return {k: np.asarray(v) for k, v in R.items()}


def main():
    out = {}
    for p in sorted(glob.glob(str(cm.OUT / "dev" / "*.npz"))):
        d, meta = load(p)
        T = table(d)
        key = f"{meta['model']}_{meta['suite']}_{meta['size']}"
        c = T["called"]
        res = {}
        print(f"\n== {key}: looks {len(c)}, calls {c.mean():.3f}, calls after a call {(c & T['prevcall']).sum() / max(c.sum(), 1):.2f}")
        rs_q = np.quantile(T["drs"], [0.05, 0.1, 0.2])
        for m in (0, 0.5, 1, 1.5, 2, 3):
            for usewl in (False, True):
                for pc in (False, True):
                    for rq, rthr in ((None, -1), ("q10", rs_q[1])):
                        dmin = np.minimum(T["dW"], T["dWL2"]) if usewl else T["dW"]
                        flag = (dmin <= 0.5 + m) | (pc & T["prevcall"]) | (T["drs"] <= rthr)
                        rec = float(flag[c].mean()); keep = float((~flag[~c]).mean())
                        name = f"m{m}{'_WL' if usewl else ''}{'_pc' if pc else ''}{'_rs' if rq else ''}"
                        res[name] = dict(recall=rec, nocall_kept_half=keep)
        best = sorted(res.items(), key=lambda kv: (-kv[1]["recall"], -kv[1]["nocall_kept_half"]))
        for name, r in sorted(res.items(), key=lambda kv: kv[1]["recall"]):
            if r["recall"] >= 0.85:
                print(f"  {name:18s} recall {r['recall']:.3f}  one-camera share of non-call looks {r['nocall_kept_half']:.3f}")
        out[key] = res
    cm.save_json(out, cm.OUT / "call_predict.json")


if __name__ == "__main__":
    main()
