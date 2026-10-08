"""R12 opus: sequential open-loop simulation of look-half rules (v2, library only).

Walks every held-out library episode of the k-fold replay at the deployed cadence (a look every two decisions;
look 0 always two-camera). At each later look a rule decides how to look and what to serve:

  kinds   full  planned two-camera look (decided before looking)
          half  wrist-only look, the one-camera answer H is used
          comp  wrist-only look, then the missing camera is completed and the two-camera answer F is used
Prices (fraction of a policy call):  owner basis full .152 | half .0646 | completion .0502 (R7 Addendum A);
          "fair" basis: full .1148 (= wrist + completion; a full look with the masked dummy camera cached).
          GR00T: full .148; half/completion are NOT measured -> .074/.074 placeholders.
Guard (layer 2, frozen): flag iff the served top-1's library progress advanced <= .5 library steps since the
          previous look's served top-1. A flagged look calls the policy (MISS; executes the policy chunk, so the
          look's served-chunk error counts as 0). On a half look a call needs the missing camera: the look becomes
          `comp` (cost half + completion) and, in guard mode `redecide`, the call is re-decided on F (deployed rule).
          In mode `own`, H's own flag is final.
Rules: full | all (every look wrist-first) | sw (R7 stage-wrist eligibility, decided before the look) |
       cw(t) (wrist-first; complete when |W - WL_lag| + gripper-sign disagreement > t, or the agent key is older than
       `maxlag` decisions; WL_lag = wrist + the most recent agentview key, lag-matched library features)
Reported per rule (paired with `full` on the same looks): half / comp / full shares, look cost on both price
bases, call rate g, e10_eff (served-chunk error vs the policy's own chunk; 0 on calls), its paired difference,
missed / extra calls vs `full`, and the owner IR per decision slot  v * (look cost + c_m * g)  with v = .505.
"""
from __future__ import annotations

import argparse

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.cascade_sim import Cell

PROG_EPS = 0.5
V = 0.505
PRICES = {"pi05": dict(owner=(0.152, 0.0646, 0.0502), fair=(0.1148, 0.0646, 0.0502), cm=0.848),
          "groot": dict(owner=(0.148, 0.074, 0.074), fair=(0.148, 0.074, 0.074), cm=0.852)}


def flag_of(cell, prev_top1, top1, margin=0.0):
    if prev_top1 is None:
        return False
    return (cell.prog[top1] - cell.prog[prev_top1]) * max(cell.eplen[top1] - 1, 1) <= PROG_EPS + margin


def simulate(cell, rule, H="W", thr=np.inf, margin=0.0, guard="redecide", maxlag=8, knob=0):
    """knob = periodic layer-4 gap cap in looks (0 = off): a look whose distance to the last policy call (guard or
    knob) reaches `knob` looks is a planned knob call (two-camera look, MISS)."""
    Z, sig = cell.Z, cell.sig
    act = Z["act"]
    rec = []
    for e, looks in cell.eps.items():
        prevF = None
        prev_top1 = None
        prev_kernel = None
        lag = 0
        since = 0                      # looks since the last policy call of the rule's sequence
        sinceF = 0
        for k, i in enumerate(looks):
            lag += 2
            since += 1
            sinceF += 1
            knob_call = bool(knob) and k > 0 and since >= knob
            knobF = bool(knob) and k > 0 and sinceF >= knob
            memF, wF, sF = Z["F|mem"][i], Z["F|w"][i], Z["F|served"][i]
            flagF = flag_of(cell, prevF, memF[0]) or knobF
            if flagF:
                sinceF = 0
            prevF = memF[0]
            kind = "full"
            if k > 0 and rule != "full" and not knob_call:
                memH, sH = Z[f"{H}|mem"][i], Z[f"{H}|served"][i]
                if rule == "sw":
                    kind = "half" if cell.sw_eligible(*prev_kernel) else "full"
                elif rule == "all":
                    kind = "half"
                elif rule == "cw":
                    if lag > maxlag:
                        kind = "comp"
                    else:
                        tag = "WL" if lag == 2 else f"WL{lag}"
                        sWL = Z[f"{tag}|served"][i]
                        score = float(cm.motion_err(sH, sWL, sig)) + float(np.mean(np.sign(sH[:5, 6]) != np.sign(sWL[:5, 6])))
                        kind = "half" if score <= thr else "comp"
            if kind == "half":
                if guard == "own":
                    flag = flag_of(cell, prev_top1, memH[0])
                    if flag:
                        kind = "comp"                       # the policy call needs both cameras
                    top1, served = memH[0], sH
                    kern = (memH, Z[f"{H}|w"][i])
                else:                                       # redecide: H's widened flag -> complete, F decides
                    if flag_of(cell, prev_top1, memH[0], margin):
                        kind = "comp"
                        flag = flag_of(cell, prev_top1, memF[0])
                        top1, served, kern = memF[0], sF, (memF, wF)
                    else:
                        flag = False
                        top1, served, kern = memH[0], sH, (memH, Z[f"{H}|w"][i])
            else:
                flag = flag_of(cell, prev_top1, memF[0]) or knob_call
                top1, served, kern = memF[0], sF, (memF, wF)
            if flag:
                since = 0
            if kind != "half":
                lag = 0
            prev_top1, prev_kernel = top1, kern
            if k == 0:
                continue
            e = 0.0 if flag else float(cm.motion_err(act[i], served, sig))
            eF = 0.0 if flagF else float(cm.motion_err(act[i], sF, sig))
            gm = 0.0 if flag else float(cm.grip_mis(act[i], served))
            gmF = 0.0 if flagF else float(cm.grip_mis(act[i], sF))
            # retrieval-quality effect only: one-camera chunk vs two-camera chunk on looks served half
            dq = (float(cm.motion_err(act[i], served, sig)) - float(cm.motion_err(act[i], sF, sig))) if kind == "half" else 0.0
            rec.append((kind, flag, flagF, e, eF, gm, gmF, dq))
    kinds = np.array([r[0] for r in rec])
    flag = np.array([r[1] for r in rec])
    flagF = np.array([r[2] for r in rec])
    e, eF = np.array([r[3] for r in rec]), np.array([r[4] for r in rec])
    gm, gmF = np.array([r[5] for r in rec]), np.array([r[6] for r in rec])
    P = PRICES[cell.model]
    out = dict(n=len(rec), half=float((kinds == "half").mean()), comp=float((kinds == "comp").mean()),
               full=float((kinds == "full").mean()), g=float(flag.mean()), gF=float(flagF.mean()),
               missed=float((flagF & ~flag).mean()), extra=float((flag & ~flagF).mean()),
               e_eff=float(e.mean()), de_eff=float((e - eF).mean()), de_se=float((e - eF).std() / np.sqrt(len(e))),
               dgrip=float((gm - gmF).mean()), dq=float(np.mean([r[7] for r in rec])),
               dq_se=float(np.std([r[7] for r in rec]) / np.sqrt(len(rec))))
    for basis in ("owner", "fair"):
        f, h, c = P[basis]
        look = (kinds == "full") * f + (kinds == "half") * h + (kinds == "comp") * (h + c)
        out[f"look_{basis}"] = float(look.mean())
        out[f"ir_{basis}"] = V * float(look.mean() + P["cm"] * flag.mean())
        out[f"irF_{basis}"] = V * float(f + P["cm"] * flagF.mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05:l10:50,pi05:spatial:50,groot:l10:50,groot:spatial:50")
    ap.add_argument("--H", default="W")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    res = {}
    for spec in a.cells.split(","):
        m, s, n = spec.split(":")
        cell = Cell(m, s, int(n))
        key = f"{m}_{s}_{n}"
        rows = [("full", simulate(cell, "full")), ("sw_own", simulate(cell, "sw", a.H, guard="own")),
                ("all_own", simulate(cell, "all", a.H, guard="own"))]
        for t in (0.1, 0.15, 0.2, 0.3):
            rows.append((f"cw{t}_own", simulate(cell, "cw", a.H, thr=t, guard="own")))
        res[key] = dict(rows)
        print(f"\n== {key} H={a.H}")
        print(f"{'rule':13s} {'half':>5s} {'comp':>5s} {'full':>5s} {'lookO':>6s} {'lookF':>6s} {'g':>6s} {'gF':>6s} "
              f"{'miss':>6s} {'extra':>6s} {'dq':>7s} {'dq_se':>6s} {'de_eff':>7s} {'se':>6s} {'dgrip':>6s} {'dIR_O':>7s} {'dIR_F':>7s}")
        for name, r in rows:
            print(f"{name:13s} {r['half']:5.2f} {r['comp']:5.2f} {r['full']:5.2f} {r['look_owner']:6.4f} "
                  f"{r['look_fair']:6.4f} {r['g']:6.3f} {r['gF']:6.3f} {r['missed']:6.3f} {r['extra']:6.3f} "
                  f"{r['dq']:+7.4f} {r['dq_se']:6.4f} {r['de_eff']:+7.4f} {r['de_se']:6.4f} {r['dgrip']:+6.3f} "
                  f"{r['ir_owner'] - r['irF_owner']:+7.4f} {r['ir_fair'] - r['irF_fair']:+7.4f}")
    cm.save_json(res, cm.OUT / f"sim2_{a.H}{a.tag}.json")


if __name__ == "__main__":
    main()
