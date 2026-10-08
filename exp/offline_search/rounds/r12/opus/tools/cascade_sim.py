"""R12 opus: sequential open-loop simulation of look-half rules on library replay (library only).

Each held-out library episode (k-fold replay outputs) is walked at the deployed cadence (a look every two decisions).
Look 0 is always full. At every later look the rule chooses between
  * serving the one-camera retrieval H (a "half look"), or
  * completing the missing camera and serving the full retrieval F (the deployed two-camera answer),
and the frozen only-no-progress guard is applied to whatever was served (top-1 progress must advance > .5 library
steps since the previous look's served top-1). A guard flag on a half look always completes first (a policy call
needs both cameras), and the call decision is then re-taken on F, exactly as deployed.

Rules (all decide from information available at that look; no future, no policy chunk):
  full      every look F (deployed baseline)
  all       every look H; guard flag on H -> complete, re-decide on F
  sw        R7 stage-wrist eligibility from the previous served kernel (unanimous gripper mode, no failed-rollout
            member, every member's chain stays unanimous and away from gripper events through the next head);
            eligible -> as `all`, else full
  casc(t)   cascade: H first; complete when the doubt score > t or H's guard (widened by margin m) flags
Doubt score: RMS (sigma units, 10 steps x 6 motion ch.) between H's served chunk and the continuation of the
previous served kernel (its members advanced two rows), plus gripper-sign disagreement of the first five steps.

Outputs per cell-size and rule: half share, completion share, look cost (owner prices), mean e10 over looks and
paired de10 vs `full`, gripper mismatch, guard rate g and its paired difference, missed full-guard flags.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.analyze_replay import load

PROG_EPS = 0.5
PRICES = {"pi05": dict(full=0.152, half=0.0646, comp=0.0502), "groot": dict(full=0.148, half=0.074, comp=0.074)}


class Cell:
    def __init__(self, model, suite, size):
        self.model, self.suite, self.size = model, suite, size
        Z = load(model, suite, size)
        self.Z = Z
        C = cm.load_cell(model, suite)
        self.C = C
        S = cm.stage_arrays(model, suite)
        self.S = S
        self.sig = Z["sig"]
        self.act_parent = C["act"]
        # episode -> ordered look positions
        order = np.lexsort((Z["step"], Z["ep"]))
        self.eps = {}
        for i in order:
            if Z["step"][i] % 2 == 0:
                self.eps.setdefault(int(Z["ep"][i]), []).append(int(i))
        self.prog = C["prog"]
        self.eplen = C["ep_len"]

    def cont(self, mem, w):
        adv = cm.advance(self.S["next"], mem, 2)
        ok = adv >= 0
        adv = np.where(ok, adv, mem)                    # terminal: hold the last row
        return np.einsum("k,ktc->tc", w, self.act_parent[adv]), adv, ok

    def sw_eligible(self, mem, w):
        S = self.S
        if (S["mode"][mem] < 0).any():
            return False
        m0 = S["mode"][mem]
        if not np.all(m0 == m0[0]):
            return False
        for depth in range(3):
            adv = cm.advance(S["next"], mem, depth)
            if (adv < 0).any():
                return False
            if not np.all(S["mode"][adv] == m0[0]) or S["event"][adv].any():
                return False
        return True


def simulate(cell, H, rule, thr=np.inf, margin=0.0, gate_event=False, maxlag=8, serve_wl=False):
    Z, sig = cell.Z, cell.sig
    act = Z["act"]
    out = dict(n=0, half=0, comp=0, full=0, cost=0.0, e10=[], e10F=[], grip=[], gripF=[], flag=[], flagF=[],
               missed=0, extra=0, dF=[])
    for e, looks in cell.eps.items():
        prevF = None                   # previous look's top-1 for the all-F baseline sequence
        prev = None                    # previous served (mem, w) of the rule's sequence
        agent_lag = 0                  # decisions since the last fresh agentview key (full or completed look)
        for k, i in enumerate(looks):
            agent_lag += 2
            memF, wF = Z["F|mem"][i], Z["F|w"][i].astype(np.float64)
            sF = Z["F|served"][i]
            # baseline guard (all-F sequence)
            pF = cell.prog[memF[0]]
            LF = max(cell.eplen[memF[0]] - 1, 1)
            flagF = prevF is not None and (pF - prevF) * LF <= PROG_EPS
            prevF = pF
            if k == 0 or rule == "full":
                served, mem, w, kind = sF, memF, wF, "full"
                agent_lag = 0
            else:
                memH, wH = Z[f"{H}|mem"][i], Z[f"{H}|w"][i].astype(np.float64)
                sH = Z[f"{H}|served"][i]
                pm, pw = prev
                use_half = True
                if rule == "sw":
                    use_half = cell.sw_eligible(pm, pw)
                elif rule == "cw":
                    # self-consistency: does the stalest-allowed agentview change the wrist-only answer?
                    if agent_lag > maxlag:
                        use_half = False
                    else:
                        sWL = Z[f"WL{agent_lag if agent_lag > 2 else ''}|served"][i]
                        score = float(cm.motion_err(sH, sWL, sig)) + float(np.mean(np.sign(sH[:5, 6]) != np.sign(sWL[:5, 6])))
                        use_half = score <= thr
                        if use_half and serve_wl:
                            sH = sWL
                            memH, wH = Z[f"WL{agent_lag if agent_lag > 2 else ''}|mem"][i], Z[f"WL{agent_lag if agent_lag > 2 else ''}|w"][i].astype(np.float64)
                elif rule == "casc":
                    contc, adv, ok = cell.cont(pm, pw)
                    dmot = float(cm.motion_err(sH, contc, sig))
                    gdis = float(np.mean(np.sign(sH[:5, 6]) != np.sign(contc[:5, 6])))
                    score = dmot + gdis
                    use_half = score <= thr
                    if gate_event and cell.S["event"][cm.advance(cell.S["next"], pm, 2)].any():
                        use_half = False
                if use_half:
                    pp = cell.prog[pm[0]]
                    pH = cell.prog[memH[0]]
                    LH = max(cell.eplen[memH[0]] - 1, 1)
                    if (pH - pp) * LH <= PROG_EPS + margin:
                        use_half = False                   # guard doubt -> complete, re-decide on F
                        kind = "comp"
                    else:
                        kind = "half"
                else:
                    kind = "full"
                if kind == "half":
                    served, mem, w = sH, memH, wH
                else:
                    served, mem, w = sF, memF, wF
                    agent_lag = 0
            # rule's guard on what was served
            if prev is None:
                flag = False
            else:
                pp = cell.prog[prev[0][0]]
                flag = (cell.prog[mem[0]] - pp) * max(cell.eplen[mem[0]] - 1, 1) <= PROG_EPS
            prev = (mem, w)
            if k == 0:
                continue
            out["n"] += 1
            out[kind] += 1
            out["e10"].append(float(cm.motion_err(act[i], served, sig)))
            out["e10F"].append(float(cm.motion_err(act[i], sF, sig)))
            out["grip"].append(float(cm.grip_mis(act[i], served)))
            out["gripF"].append(float(cm.grip_mis(act[i], sF)))
            out["dF"].append(float(cm.motion_err(sF, served, sig)))
            out["flag"].append(flag)
            out["flagF"].append(flagF)
            out["missed"] += int(flagF and not flag)
            out["extra"] += int(flag and not flagF)
    n = max(out["n"], 1)
    P = PRICES[cell.model]
    look_cost = (out["full"] * P["full"] + out["half"] * P["half"] + out["comp"] * (P["half"] + P["comp"])) / n
    e, eF = np.array(out["e10"]), np.array(out["e10F"])
    return dict(n=out["n"], half=out["half"] / n, comp=out["comp"] / n, full=out["full"] / n, look_cost=look_cost,
                e10=float(e.mean()), de10=float((e - eF).mean()), de10_se=float((e - eF).std() / np.sqrt(n)),
                dgrip=float(np.mean(out["grip"]) - np.mean(out["gripF"])), dF=float(np.mean(out["dF"])),
                g=float(np.mean(out["flag"])), gF=float(np.mean(out["flagF"])), missed=out["missed"] / n,
                extra=out["extra"] / n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05:l10:50,pi05:spatial:50,groot:l10:50,groot:spatial:50")
    ap.add_argument("--H", default="W")
    a = ap.parse_args(argv)
    res = {}
    for spec in a.cells.split(","):
        m, s, n = spec.split(":")
        cell = Cell(m, s, int(n))
        key = f"{m}_{s}_{n}"
        rows = [("full", simulate(cell, a.H, "full")), ("all", simulate(cell, a.H, "all")),
                ("all_m1", simulate(cell, a.H, "all", margin=1.0)), ("sw", simulate(cell, a.H, "sw"))]
        for t in (0.3, 0.5):
            rows.append((f"casc{t}_m1", simulate(cell, a.H, "casc", thr=t, margin=1.0)))
        for t in (0.1, 0.15, 0.2, 0.3):
            for m_ in (0.0, 1.0):
                rows.append((f"cw{t}_m{m_:g}", simulate(cell, a.H, "cw", thr=t, margin=m_)))
            rows.append((f"cw{t}_m1_wl", simulate(cell, a.H, "cw", thr=t, margin=1.0, serve_wl=True)))
        rows.append(("cw0.2_m1_L4", simulate(cell, a.H, "cw", thr=0.2, margin=1.0, maxlag=4)))
        res[key] = dict(rows)
        print(f"\n== {key} H={a.H}")
        print(f"{'rule':12s} {'half':>5s} {'comp':>5s} {'look$':>6s} {'e10':>6s} {'de10':>7s} {'se':>6s} {'dgrip':>6s} "
              f"{'dF':>6s} {'g':>6s} {'gF':>6s} {'miss':>6s} {'extra':>6s}")
        for name, r in rows:
            print(f"{name:12s} {r['half']:5.2f} {r['comp']:5.2f} {r['look_cost']:6.4f} {r['e10']:6.3f} {r['de10']:+7.4f} "
                  f"{r['de10_se']:6.4f} {r['dgrip']:+6.3f} {r['dF']:6.3f} {r['g']:6.3f} {r['gF']:6.3f} "
                  f"{r['missed']:6.3f} {r['extra']:6.3f}")
    cm.save_json(res, cm.OUT / f"cascade_sim_{a.H}.json")


if __name__ == "__main__":
    main()
