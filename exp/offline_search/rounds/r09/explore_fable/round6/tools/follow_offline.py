"""Offline follow-block comparison with the R8 deferred policy shadows (pure-cache A arms, inits 0-29 only).

At every look that follows one blind block, three candidate chunks for the next five controls are compared with the
policy's own chunk at that observation:
  fresh     the chunk the fresh look actually served (kernel mean of the new retrieval),
  tail      the anchor chunk's controls 10:15 (what R7's extension serves for GR00T, H=16: ``os_sf_source 1``),
  successor the stage-table successor rows' heads (what it serves for pi0.5, H=10: ``os_sf_source 2``),
plus the ordinary blind block (anchor controls 5:10 at the blind decision) to show how error grows with blind age.
Stage tables come from the round-5 fitted artifacts (same frozen retrieval); tables are filtered to inits < 30 on load.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler

DERIVED = Path("/home/weiland/trace_runs/offline_search_store/derived/r09_fable")
R5 = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r5")
A_FIT = {"pi05_l10_50": "/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl",
         "groot_l10_50": "/home/weiland/trace_runs/os_closed_loop/r05_x/fits/r5x_g_l10_50_tail1u.pkl"}
STAGE_ART = {"pi05_l10_50": R5 / "fits" / "r9f5_pi05_l10_50_esc_fg.pkl", "groot_l10_50": R5 / "fits" / "r9f5_groot_l10_50_np_fg.pkl"}
OUT = Path(__file__).resolve().parents[1] / "out"


def rms(a, b):
    return np.sqrt(np.mean((np.asarray(a)[:, :5, :6] - np.asarray(b)[:, :5, :6]) ** 2, axis=(1, 2)))


def run(cell):
    arm = f"r8_{cell}_A"
    dec = pd.read_parquet(DERIVED / "decisions" / f"{arm}.parquet")
    dec = dec[dec["init"] < 30].reset_index(drop=True)
    arr = np.load(DERIVED / "arrays" / f"{arm}.npz")
    ids = {d: i for i, d in enumerate(arr["decision_id"])}
    keep = np.array([ids[d] for d in dec.decision_id])
    served, policy, rows, weights, state = (arr[k][keep] for k in ("served", "shadow", "rows", "weights", "state"))
    with open(A_FIT[cell], "rb") as f:
        act = np.asarray(FitUnpickler(f).load()["method"].act)[:, :, :7]
    with open(STAGE_ART[cell], "rb") as f:
        stages = pickle.load(f)["method"].lc_stages
    H = act.shape[1]
    look = dec[dec.vision & (dec.look_reason == 1) & dec.hit.fillna(True).astype(bool)]
    prev = look.index.values - 2
    ok = (prev >= 0)
    ok &= dec.episode_key.values[np.clip(prev, 0, None)] == look.episode_key.values
    ok &= dec.vision.values[np.clip(prev, 0, None)].astype(bool)
    look, prev = look[ok], prev[ok]
    arows, aw = rows[prev], weights[prev]
    out = []
    for i, (li, pi) in enumerate(zip(look.index.values, prev)):
        r, w = arows[i], aw[i]
        succ = stages.advance(r, 2)
        structural = not (succ < 0).any()
        tail = np.tensordot(w, act[r, 10:15, :], 1) if H >= 15 else None
        successor = np.tensordot(w, act[succ, :5, :], 1) if structural else None
        blind_block = np.tensordot(w, act[r, 5:10, :], 1)
        pol, srv = policy[li][:5, :7], served[li][:5, :7]
        pol_blind = policy[pi + 1][:5, :7]
        info = stages.online(r, w)
        stage_ok = structural and bool(info["unanimous"]) and all(bool(np.all(stages.mode[stages.advance(r, a)] == info["mode"])) for a in (1, 2))
        d, supported = stages.displacement(state[li], state[pi], r, w, 2)
        gate = stage_ok and supported and d <= stages.valve_radius
        out.append(dict(structural=structural, gate=gate,
                        d_fresh=float(rms(srv[None], pol[None])[0]),
                        d_tail=float(rms(tail[None], pol[None])[0]) if tail is not None else np.nan,
                        d_successor=float(rms(successor[None], pol[None])[0]) if successor is not None else np.nan,
                        d_blind_block=float(rms(blind_block[None], pol_blind[None])[0]),
                        d_anchor_head=float(rms(served[pi][None], policy[pi][None])[0]),
                        grip_tail=float(np.sign(np.median(tail[:, 6])) != np.sign(np.median(pol[:, 6]))) if tail is not None else np.nan,
                        grip_successor=float(np.sign(np.median(successor[:, 6])) != np.sign(np.median(pol[:, 6]))) if successor is not None else np.nan,
                        grip_fresh=float(np.sign(np.median(srv[:, 6])) != np.sign(np.median(pol[:, 6])))))
    t = pd.DataFrame(out)
    rows_ = []
    for name, mask in (("all (structural)", t.structural.values), ("gate pass", t.gate.values), ("gate fail", t.structural.values & ~t.gate.values)):
        s = t[mask]
        rows_.append(dict(stratum=name, n=len(s), share=len(s) / len(t), anchor_head=s.d_anchor_head.mean(), blind_block=s.d_blind_block.mean(),
                          fresh_look=s.d_fresh.mean(), follow_tail=s.d_tail.mean(), follow_successor=s.d_successor.mean(),
                          tail_minus_fresh=(s.d_tail - s.d_fresh).mean(), successor_minus_fresh=(s.d_successor - s.d_fresh).mean(),
                          grip_fresh=s.grip_fresh.mean(), grip_tail=s.grip_tail.mean(), grip_successor=s.grip_successor.mean()))
    tab = pd.DataFrame(rows_)
    print(f"\n== {cell}: {len(t)} looks after one blind block (inits 0-29), H={H}; RMS to the policy chunk, motion dims")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tab.to_csv(OUT / f"follow_offline_{cell}.csv", index=False)
    return tab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in ("pi05_l10_50", "groot_l10_50"):
        run(cell)


if __name__ == "__main__":
    main()
