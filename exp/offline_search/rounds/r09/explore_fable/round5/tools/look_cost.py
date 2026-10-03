"""Offline look-cost analysis on the R8 debug collection (pi0.5, deferred camera shadows), discovery inits 0-29 only.

Q1  Does a wrist-only look retrieve something different from the full look, and does the difference track the gates
    (R7 stage "easy", pace lag, retrieval confidence)?  Metrics at every full-look decision of the R8 pure-cache arm:
    top-1 agreement full vs wrist retrieval, kernel-set overlap, motion distance to the deferred policy chunk for the
    served (full) chunk and the wrist chunk, gripper-command disagreement.
Q2  What would gated follow have served instead of the look at 10 controls?  At every look decision that follows one
    blind block (look reason 1), the stage-table successor block of the previous anchor vs the fresh retrieval vs the
    policy chunk, stratified by the follow gates (stage unanimity/chain, state valve).

Every table is filtered to inits < 30 immediately after load; the camera-shadow arrays are fetched by decision id for
those rows only.  Nothing here is task-indexed: stratifications use stage structure, pace and confidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r07.stages.stages import StageTable

DERIVED = Path("/home/weiland/trace_runs/offline_search_store/derived/r09_fable")
R8 = Path("/home/weiland/trace_runs/os_closed_loop/r08_main")
R07_FITS = Path("/home/weiland/trace_runs/os_closed_loop/r07_main/fits")
A_FIT = {"pi05_l10_50": "/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl",
         "pi05_spatial_50": "/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_sp_50_tail1uc.pkl",
         "pi05_l10_500": "/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_l10_500_tail1uc.pkl"}
STAGES = {"pi05_l10_50": "stages_pi05_l10_50.pkl", "pi05_spatial_50": "stages_pi05_sp_50.pkl", "pi05_l10_500": "stages_pi05_l10_500.pkl"}
OUT = Path(__file__).resolve().parents[1] / "out"
MAX_INIT = 30


def load_cell(cell):
    arm = f"r8_{cell}_A"
    dec = pd.read_parquet(DERIVED / "decisions" / f"{arm}.parquet")
    dec = dec[dec["init"] < MAX_INIT].reset_index(drop=True)                      # rule 1, before anything else
    arr = np.load(DERIVED / "arrays" / f"{arm}.npz")
    ids = {d: i for i, d in enumerate(arr["decision_id"])}
    keep = np.array([ids[d] for d in dec.decision_id])
    arrays = {k: arr[k][keep] for k in ("served", "shadow", "state", "rows", "weights")}
    with open(A_FIT[cell], "rb") as f:
        afit = FitUnpickler(f).load()["method"]
    stages = StageTable.load(R07_FITS / STAGES[cell])
    return arm, dec, arrays, afit, stages


def motion_dist(a, b, n=5):
    return np.sqrt(np.mean((np.asarray(a)[:, :n, :6] - np.asarray(b)[:, :n, :6]) ** 2, axis=(1, 2)))


def easy_flags(stages, rows, weights, afit, target=2):
    """R7 stage-wrist plan at an anchor: unanimity (2), chain valid (3), no event near along the chain (4)."""
    n = len(rows)
    unanimous, chain_ok, event_free = np.zeros(n, bool), np.ones(n, bool), np.ones(n, bool)
    for i in range(n):
        r, w = rows[i], weights[i]
        info = stages.online(r, w)
        unanimous[i] = info["unanimous"] and info["unknown_mass"] == 0
        for depth in range(target + 1):
            adv = stages.advance(r, depth)
            if np.any(adv < 0):
                chain_ok[i] = False
                break
            sub = stages.online(adv, w)
            if not sub["unanimous"] or np.any(stages.event_near[adv]):
                event_free[i] = False
                break
    return unanimous, chain_ok, event_free


def q1(cell, dec, arrays, afit, stages, arm):
    look = dec[dec.vision & dec.hit.fillna(True).astype(bool)].copy()
    data = reader.open_arm(R8, arm)
    data.cache_enabled = False
    cs = data.aug("camera_shadow", list(look.decision_id))                      # only the selected (inits < 30) ids
    order = {d: i for i, d in enumerate(cs["decision_id"])}
    sel = np.array([order[d] for d in look.decision_id])
    full_rows, full_w = arrays["rows"][look.index], arrays["weights"][look.index]
    served, policy = arrays["served"][look.index], arrays["shadow"][look.index]
    wrist_rows, wrist_chunk = cs["wrist_rows"][sel], cs["wrist_cache_chunk"][sel][:, :, :7]
    third_rows, third_chunk = cs["third_rows"][sel], cs["third_cache_chunk"][sel][:, :, :7]
    look["top1_same_wrist"] = full_rows[:, 0] == wrist_rows[:, 0]
    look["top1_same_third"] = full_rows[:, 0] == third_rows[:, 0]
    look["overlap_wrist"] = [len(set(a) & set(b)) / len(set(a)) for a, b in zip(full_rows, wrist_rows)]
    look["d_full_policy"] = motion_dist(served, policy)
    look["d_wrist_policy"] = motion_dist(wrist_chunk, policy)
    look["d_third_policy"] = motion_dist(third_chunk, policy)
    look["d_wrist_full"] = motion_dist(wrist_chunk, served)
    g = lambda c: np.sign(np.median(c[:, :5, 6], axis=1))
    look["grip_disagree_wrist"] = g(wrist_chunk) != g(served)
    look["lag"] = look.decision_seq.values - look.x_lib_step.values
    un, ch, ev = easy_flags(stages, full_rows, full_w, afit)
    look["easy"] = un & ch & ev
    look["on_pace"] = look.lag <= 1
    look["conf_hi"] = look.d1 <= np.nanmedian(look.d1)          # d1 is a distance: small = confident
    look["wrist_conf"], look["full_conf"] = cs["wrist_conf"][sel], look.conf
    cols = ["top1_same_wrist", "overlap_wrist", "d_full_policy", "d_wrist_policy", "d_third_policy", "d_wrist_full", "grip_disagree_wrist"]
    rows = []
    for name, mask in (("all looks", np.ones(len(look), bool)), ("easy (R7 stage plan)", look.easy.values), ("not easy", ~look.easy.values),
                       ("on pace (lag<=1)", look.on_pace.values), ("behind (lag>1)", ~look.on_pace.values),
                       ("confident (d1<=median)", look.conf_hi.values), ("unsure (d1>median)", ~look.conf_hi.values),
                       ("easy & on pace", look.easy.values & look.on_pace.values)):
        sub = look[mask]
        rows.append(dict(stratum=name, n=len(sub), share=len(sub) / len(look), **{c: float(sub[c].mean()) for c in cols},
                         wrist_minus_full=float((sub.d_wrist_policy - sub.d_full_policy).mean()),
                         wrist_worse_frac=float((sub.d_wrist_policy > sub.d_full_policy).mean())))
    table = pd.DataFrame(rows)
    # episode level: does wrist/full disagreement at looks go with failure in the pure-cache arm?
    ep = look.groupby("episode_key").agg(success=("journal_success", "first"), disagree=("top1_same_wrist", lambda s: 1 - s.mean()),
                                         d_gap=("d_wrist_full", "mean"), n=("top1_same_wrist", "size"))
    corr = dict(disagree_succ=float(ep[ep.success].disagree.mean()), disagree_fail=float(ep[~ep.success].disagree.mean()),
                gap_succ=float(ep[ep.success].d_gap.mean()), gap_fail=float(ep[~ep.success].d_gap.mean()), n_ep=int(len(ep)))
    return table, corr, look


def q2(cell, dec, arrays, afit, stages):
    """At looks after one blind block (reason 1): the follow block vs the fresh look vs the policy."""
    idx = {d: i for i, d in enumerate(dec.decision_id)}
    look = dec[dec.vision & (dec.look_reason == 1) & dec.hit.fillna(True).astype(bool)].copy()
    anchors = look.anchor_decision_id.map(idx)
    # the anchor of the look decision is the look itself; the previous anchor is two decisions earlier in the episode
    prev = look.index.values - 2
    ok = (prev >= 0)
    ok &= dec.episode_key.values[np.clip(prev, 0, None)] == look.episode_key.values
    ok &= dec.vision.values[np.clip(prev, 0, None)].astype(bool)
    look, prev = look[ok], prev[ok]
    arows, aw = arrays["rows"][prev], arrays["weights"][prev]
    astate, cstate = arrays["state"][prev], arrays["state"][look.index]
    served, policy = arrays["served"][look.index], arrays["shadow"][look.index]
    act = np.asarray(afit.act)[:, :, :7]
    follow = np.zeros((len(look), 5, 7), np.float32)
    structural = np.ones(len(look), bool)
    valve_ok = np.zeros(len(look), bool)
    stage_ok = np.zeros(len(look), bool)
    for i in range(len(look)):
        r, w = arows[i], aw[i]
        succ = stages.advance(r, 2)
        if (succ < 0).any():
            structural[i] = False
            continue
        follow[i] = np.tensordot(w, act[succ, :5, :], 1)
        info = stages.online(r, w)
        stage_ok[i] = bool(info["unanimous"]) and all(bool(np.all(stages.mode[stages.advance(r, a)] == info["mode"])) for a in (1, 2))
        d, supported = stages.displacement(cstate[i], astate[i], r, w, 2)
        valve_ok[i] = supported and d <= stages.valve_radius
    look["structural"], look["stage_ok"], look["valve_ok"] = structural, stage_ok, valve_ok
    look["d_look_policy"] = motion_dist(served, policy)
    look["d_follow_policy"] = np.where(structural, motion_dist(follow, policy), np.nan)
    look["d_follow_look"] = np.where(structural, motion_dist(follow, served), np.nan)
    look["grip_disagree_follow"] = np.where(structural, np.sign(np.median(follow[:, :, 6], 1)) != np.sign(np.median(served[:, :5, 6], 1)), np.nan)
    gated = structural & stage_ok & valve_ok
    valve_only = structural & valve_ok
    rows = []
    for name, mask in (("all looks after one blind block", structural), ("gate pass (stage + valve)", gated), ("gate fail", structural & ~gated),
                       ("valve only pass", valve_only), ("valve only fail", structural & ~valve_ok)):
        sub = look[mask]
        rows.append(dict(stratum=name, n=len(sub), share=len(sub) / max(len(look), 1), d_look_policy=float(sub.d_look_policy.mean()),
                         d_follow_policy=float(sub.d_follow_policy.mean()), d_follow_look=float(sub.d_follow_look.mean()),
                         follow_minus_look=float((sub.d_follow_policy - sub.d_look_policy).mean()),
                         follow_worse_frac=float((sub.d_follow_policy > sub.d_look_policy).mean()),
                         grip_disagree=float(sub.grip_disagree_follow.mean())))
    return pd.DataFrame(rows), look


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05_l10_50,pi05_spatial_50,pi05_l10_500")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cell in a.cells.split(","):
        arm, dec, arrays, afit, stages = load_cell(cell)
        t1, corr, look1 = q1(cell, dec, arrays, afit, stages, arm)
        t2, look2 = q2(cell, dec, arrays, afit, stages)
        print(f"\n== {cell}: Q1 wrist-only vs full look at {len(look1)} look decisions (inits 0-29)")
        print(t1.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("episode-level (pure cache):", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in corr.items()}))
        print(f"== {cell}: Q2 follow block vs fresh look at {len(look2)} looks after one blind block")
        print(t2.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        t1.to_csv(OUT / f"q1_wrist_{cell}.csv", index=False)
        t2.to_csv(OUT / f"q2_follow_{cell}.csv", index=False)
        summary[cell] = dict(q1=t1.to_dict("records"), q1_episode=corr, q2=t2.to_dict("records"), n_look=int(len(look1)), n_follow=int(len(look2)))
    (OUT / "look_cost_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
