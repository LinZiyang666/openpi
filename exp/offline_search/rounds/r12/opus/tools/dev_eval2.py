"""R12 opus (directive 2): one-camera schedules on closed-loop dev looks WITH the guard interaction modelled.

Compute-share prices: full .152 (GR00T .148), one camera .076 (.074), completion .076 (.074), call = full + .848/.852.
Replayed look by look on the recorded states of the deployed system (which served two cameras everywhere). At a
recorded policy call the deployed guard (on the two-camera answer F) fired. Under a schedule:
  * a look planned with both cameras (step 0, a completion, a forced full look) reproduces the deployed decision;
  * a one-camera look first evaluates the frozen only-no-progress rule on its OWN served top-1 (progress vs the
    previous look's served top-1), with sensitivity margin m (flag if advance <= .5 + m library steps). If it flags,
    the missing camera is completed and the DEPLOYED guard decides on F ("re-decide"): at a recorded call this
    reproduces the call; at a recorded HIT it costs one completion and serves F. If it does not flag at a recorded
    call, the call is MISSED and the one-camera chunk is executed instead of the policy chunk.
Costs per look: full .152; one-camera .076; one-camera + completion .152; calls add .848 (unchanged vs deployed
except missed calls, which save .848 + .076). Retrieval cost dq: mean over looks of (error of what is executed -
error of what the deployed system executed) against the policy's shadow chunk (0 for an executed policy chunk).
Schedules: allW, casc(t) [wrist first, complete if |W - WL_lag| + grip > t or lag > 8], cascL(t) [serve WL_lag when
the test passes], altWA / altWA_c(t) [alternate the encoded camera, serve with the other camera remembered from the
previous look; optional consistency test], altF [one remembered-agentview look between full looks].
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load, score

V = 0.505
PRICE = {"pi05": (0.152, 0.076, 0.076, 0.848), "groot": (0.148, 0.074, 0.074, 0.852)}


def plan(rule, t, d, i, sig, lag_a, lag_w, next_cam, maxlag=8):
    """Camera plan for a non-first look: ('full'|'half'|'comp', served variant)."""
    if rule == "allW":
        return "half", "W"
    if rule in ("casc", "cascL"):
        if lag_a > maxlag:
            return "comp", "F"
        L = min(8, max(2, lag_a))
        if score(d["S_W"][i], d[f"S_WL{L}"][i], sig) <= t:
            return "half", ("W" if rule == "casc" else f"WL{L}")
        return "comp", "F"
    if rule == "altF":
        return ("half", "WL2") if lag_a == 2 else ("full", "F")
    if rule in ("altWA", "altWA_c"):
        if next_cam == "w" and lag_a <= 2:
            ok = rule == "altWA" or score(d["S_W"][i], d["S_WL2"][i], sig) <= t
            return ("half", "WL2") if ok else ("comp", "F")
        if next_cam == "a" and lag_w <= 2:
            ok = rule == "altWA" or score(d["S_A"][i], d["S_AL2"][i], sig) <= t
            return ("half", "AL2") if ok else ("comp", "F")
        return "comp", "F"
    return "full", "F"


def simulate(d, meta, rule, t=np.inf, m=0.0):
    sig = np.asarray(meta["sig_r"])
    full, half, comp, call = PRICE[meta["model"]]
    shadow_ok = np.isfinite(d["shadow"]).all(axis=(1, 2))
    by = defaultdict(list)
    for i, u in enumerate(d["uid"]):
        by[u].append(i)
    R = defaultdict(list)
    for u, idx in by.items():
        idx = sorted(idx, key=lambda i: d["step"][i])
        lag_a = lag_w = 0
        next_cam = "w"
        prev_prog = None
        for k, i in enumerate(idx):
            if k:
                lag_a += 2
                lag_w += 2
            called = not bool(d["hit"][i])                      # deployed decision at this look
            if k == 0 or rule == "full":
                kind, served = "full", "F"
            else:
                kind, served = plan(rule, t, d, i, sig, lag_a, lag_w, next_cam)
            missed = False
            if kind == "half":
                pr = float(d[f"{served}_top1_prog"][i]); el = int(d[f"{served}_top1_eplen"][i])
                flag = prev_prog is not None and (pr - prev_prog) * max(el - 1, 1) <= 0.5 + m
                if flag:
                    kind, served = "comp", "F"                    # complete, the deployed guard decides on F
                elif called:
                    missed = True                                 # the deployed system called here; we do not
            exec_policy = called and not missed
            cost = {"full": full, "half": half, "comp": half + comp}[kind] + (call if exec_policy else 0.0)
            cost_dep = full + (call if called else 0.0)
            if kind in ("full", "comp"):
                lag_a = lag_w = 0
                next_cam = "w"
            elif served.startswith("W"):
                lag_w = 0
                next_cam = "a"
            elif served.startswith("A"):
                lag_a = 0
                next_cam = "w"
            prev_prog = float(d[f"{served}_top1_prog"][i])
            if k == 0:
                continue
            e = np.nan
            if shadow_ok[i]:
                e_dep = 0.0 if called else float(cm.motion_err(d["shadow"][i], d["S_F"][i], sig))
                e_new = 0.0 if exec_policy else float(cm.motion_err(d["shadow"][i], d[f"S_{served}"][i], sig))
                e = e_new - e_dep
            R["kind"].append(kind); R["cost"].append(cost); R["cost_dep"].append(cost_dep); R["dq"].append(e)
            R["missed"].append(missed); R["called"].append(called)
            R["spurious_comp"].append(kind == "comp" and served == "F" and not called and rule != "full")
    kind = np.array(R["kind"]); dq = np.array(R["dq"], float)
    out = dict(n=len(kind), half=float((kind == "half").mean()), comp=float((kind == "comp").mean()),
               dIR=V * float(np.mean(R["cost"]) - np.mean(R["cost_dep"])),
               call_rate=float(np.mean(R["called"])), missed_calls=float(np.mean(R["missed"])),
               missed_share_of_calls=float(np.sum(R["missed"]) / max(np.sum(R["called"]), 1)))
    if np.isfinite(dq).all():
        out.update(dq=float(dq.mean()), dq_se=float(dq.std() / np.sqrt(len(dq))))
    return out


RULES = [("allW", "allW", None), ("altF", "altF", None), ("altWA", "altWA", None)] + \
        [(f"casc{t}", "casc", t) for t in (0.075, 0.1, 0.15)] + \
        [(f"cascL{t}", "cascL", t) for t in (0.1, 0.15, 0.2)] + \
        [(f"altWA_c{t}", "altWA_c", t) for t in (0.15, 0.2, 0.3)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*")
    ap.add_argument("--margins", default="0,1,2")
    a = ap.parse_args(argv)
    res = {}
    for p in sorted(glob.glob(str(cm.OUT / "dev" / f"{a.glob}.npz"))):
        d, meta = load(p)
        key = f"{meta['model']}_{meta['suite']}_{meta['size']}"
        res[key] = {}
        print(f"\n== {key} ({meta['n_episodes']} episodes, {len(d['step'])} looks)")
        print(f"{'rule':12s} {'m':>3s} {'half':>5s} {'comp':>5s} {'dIR':>7s} {'calls':>6s} {'missed':>7s} {'miss%':>6s} {'dq':>8s} {'se':>7s}")
        for mg in [float(x) for x in a.margins.split(",")]:
            for name, rule, t in RULES:
                r = simulate(d, meta, rule, t if t is not None else np.inf, mg)
                res[key][f"{name}_m{mg:g}"] = r
                dq = f"{r['dq']:+8.4f} {r['dq_se']:7.4f}" if "dq" in r else f"{'—':>8s} {'':7s}"
                print(f"{name:12s} {mg:3g} {r['half']:5.2f} {r['comp']:5.2f} {r['dIR']:+7.4f} {r['call_rate']:6.3f} "
                      f"{r['missed_calls']:7.4f} {r['missed_share_of_calls']:6.3f} {dq}")
    cm.save_json(res, cm.OUT / "dev_eval2.json")


if __name__ == "__main__":
    main()
