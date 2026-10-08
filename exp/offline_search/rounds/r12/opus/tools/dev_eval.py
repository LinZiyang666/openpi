"""R12 opus (directive 2): evaluate one-camera schedules on the closed-loop dev feature tables (out/dev/*.npz).

Compute-share prices (owner ruling 2026-10-03): full look .152 (GR00T .148), one camera .076 (.074), completion of
the other camera .076 (.074); a policy call = full look + .848 (.852).

The recorded runs served two cameras at every look, so every look has fresh keys of both cameras plus the keys of
earlier looks; each schedule is replayed look by look on the recorded states (open loop on closed-loop states).
Recorded policy calls (MISS looks) are kept where they happened: they need both cameras, cost a full look, and
execute the policy (no cache error). Schedules:
  full          deployed two-camera looks
  allW          wrist only at every non-call look (serve W)
  casc(t)       wrist first; complete when |W - WL_lag| + gripper-sign disagreement > t or lag > 8; serve W
  cascL(t)      as casc, but serve WL_lag (fresh wrist + remembered agentview) when the test passes
  altF          alternate: after a fresh-agentview look, one look wrist + remembered agentview (WL2), then a full look
  altWA         every non-call look encodes ONE camera, alternating: wrist looks serve WL2 (agentview from the
                previous look), agentview looks serve AL2 (wrist from the previous look)
  altWA_c(t)    altWA plus the consistency test on wrist looks (complete if |W - WL2| > t) and on agentview looks
                (complete if |A - AL2| > t)
Reported: one-camera share, vision cost per look, IR change per slot (v = .505), retrieval cost dq = mean over looks
of (error of the served chunk - error of F) against the policy's shadow chunk (0 on call looks), and the own-guard
call-rate change (top-1 progress rule on the served sequence vs F's).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm

V = 0.505
PRICE = {"pi05": (0.152, 0.076, 0.076), "groot": (0.148, 0.074, 0.074)}


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    d = {k: z[k] for k in z.files if k != "meta_json"}
    return d, meta


def score(a, b, sig):
    return float(cm.motion_err(a, b, sig)) + float(np.mean(np.sign(a[:5, 6]) != np.sign(b[:5, 6])))


def simulate(d, meta, rule, t=np.inf, maxlag=8):
    sig = np.asarray(meta["sig_r"])
    full, half, comp = PRICE[meta["model"]]
    has_shadow = np.isfinite(d["shadow"]).all(axis=(1, 2))
    by = defaultdict(list)
    for i, u in enumerate(d["uid"]):
        by[u].append(i)
    recs = []
    for u, idx in by.items():
        idx = sorted(idx, key=lambda i: d["step"][i])
        lag_a = lag_w = 0                      # decisions since the last fresh agentview / wrist key
        next_cam = "w"
        prev_top1 = prevF_top1 = None
        alt_phase = 0
        for k, i in enumerate(idx):
            s = int(d["step"][i])
            lag_a += 2 if k else 0
            lag_w += 2 if k else 0
            hit = bool(d["hit"][i])
            kind, served = "full", "F"
            if k == 0 or not hit or rule == "full":
                kind = "full"
            elif rule == "allW":
                kind, served = "half", "W"
            elif rule in ("casc", "cascL"):
                if lag_a > maxlag:
                    kind = "comp"
                else:
                    L = min(8, max(2, lag_a))
                    sc = score(d["S_W"][i], d[f"S_WL{L}"][i], sig)
                    if sc <= t:
                        kind, served = "half", ("W" if rule == "casc" else f"WL{L}")
                    else:
                        kind = "comp"
            elif rule == "altF":
                if lag_a == 2:
                    kind, served = "half", "WL2"
                else:
                    kind = "full"
            elif rule in ("altWA", "altWA_c"):
                if next_cam == "w" and lag_a <= 2:
                    ok = rule == "altWA" or score(d["S_W"][i], d["S_WL2"][i], sig) <= t
                    kind, served = ("half", "WL2") if ok else ("comp", "F")
                elif next_cam == "a" and lag_w <= 2:
                    ok = rule == "altWA" or score(d["S_A"][i], d["S_AL2"][i], sig) <= t
                    kind, served = ("half", "AL2") if ok else ("comp", "F")
                else:
                    kind = "comp"
            # camera bookkeeping
            if kind in ("full", "comp"):
                lag_a = lag_w = 0
                next_cam = "w"
            elif served.startswith("W"):
                lag_w = 0
                next_cam = "a"
            elif served.startswith("A"):
                lag_a = 0
                next_cam = "w"
            top1 = int(d[f"{served}_top1"][i])
            pr, el = float(d[f"{served}_top1_prog"][i]), int(d[f"{served}_top1_eplen"][i])
            prF, elF = float(d["F_top1_prog"][i]), int(d["F_top1_eplen"][i])
            g = prev_top1 is not None and (pr - prev_top1) * max(el - 1, 1) <= 0.5
            gF = prevF_top1 is not None and (prF - prevF_top1) * max(elF - 1, 1) <= 0.5
            prev_top1, prevF_top1 = pr, prF
            if k == 0:
                continue
            e = eF = np.nan
            if has_shadow[i]:
                e = float(cm.motion_err(d["shadow"][i], d[f"S_{served}"][i], sig))
                eF = float(cm.motion_err(d["shadow"][i], d["S_F"][i], sig))
            dfull = float(cm.motion_err(d["S_F"][i], d[f"S_{served}"][i], sig))
            cost = {"full": full, "half": half, "comp": half + comp}[kind]
            recs.append((kind, cost, (e - eF) if hit else 0.0, dfull if kind == "half" else np.nan, g, gF, hit))
    kinds = np.array([r[0] for r in recs])
    cost = np.array([r[1] for r in recs])
    dq = np.array([r[2] for r in recs], float)
    dF = np.array([r[3] for r in recs], float)
    g = np.array([r[4] for r in recs]); gF = np.array([r[5] for r in recs]); hit = np.array([r[6] for r in recs])
    out = dict(n=len(recs), half=float((kinds == "half").mean()), comp=float((kinds == "comp").mean()),
               look_cost=float(cost.mean()), dIR_vision=V * float(cost.mean() - full),
               dF_half=float(np.nanmean(dF)) if np.isfinite(dF).any() else None,
               dg_own=float(g[hit].mean() - gF[hit].mean()))
    if np.isfinite(dq).all():
        out.update(dq=float(dq.mean()), dq_se=float(dq.std() / np.sqrt(len(dq))))
    return out


RULES = [("full", "full", None), ("allW", "allW", None), ("altF", "altF", None), ("altWA", "altWA", None)] + \
        [(f"casc{t}", "casc", t) for t in (0.05, 0.075, 0.1, 0.15)] + \
        [(f"cascL{t}", "cascL", t) for t in (0.075, 0.1, 0.15, 0.2)] + \
        [(f"altWA_c{t}", "altWA_c", t) for t in (0.1, 0.15, 0.2, 0.3)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*")
    a = ap.parse_args(argv)
    res = {}
    for p in sorted(glob.glob(str(cm.OUT / "dev" / f"{a.glob}.npz"))):
        d, meta = load(p)
        key = f"{meta['model']}_{meta['suite']}_{meta['size']}"
        res[key] = {}
        print(f"\n== {key} ({meta['root']}, {meta['n_episodes']} episodes, {len(d['step'])} looks)")
        print(f"{'rule':12s} {'half':>5s} {'comp':>5s} {'look$':>6s} {'dIRv':>7s} {'dq':>8s} {'se':>7s} {'dF_h':>6s} {'dg':>6s}")
        for name, rule, t in RULES:
            r = simulate(d, meta, rule, t if t is not None else np.inf)
            res[key][name] = r
            dq = f"{r['dq']:+8.4f} {r['dq_se']:7.4f}" if "dq" in r else f"{'—':>8s} {'':7s}"
            print(f"{name:12s} {r['half']:5.2f} {r['comp']:5.2f} {r['look_cost']:6.4f} {r['dIR_vision']:+7.4f} {dq} "
                  f"{(r['dF_half'] or 0):6.3f} {r['dg_own']:+6.3f}")
    cm.save_json(res, cm.OUT / "dev_eval.json")


if __name__ == "__main__":
    main()
