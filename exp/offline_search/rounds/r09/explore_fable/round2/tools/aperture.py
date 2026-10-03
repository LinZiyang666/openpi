"""Empty-grasp detector study from robot-observable signals (R8 pure-cache arms, inits 0-29 ONLY).

For every control: executed gripper command (+1 close), finger aperture (robot state dims 6/7), the forensic
truth stage (carry = object held). Calibrate the empty-close aperture threshold on inits 0-19 (holding vs
empty closes), evaluate detection on inits 20-29: does "close command held for 2 decisions and aperture below
threshold" flag the forensic grasp-miss episodes, how early, and how often does it fire in successes?
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader

RUN = "/home/weiland/trace_runs/os_closed_loop/r08_main"
PF = Path(RUN) / "profile" / "forensics"


def episode_table(args):
    arm, ek = args
    d = reader.open_arm(RUN, arm)
    d.cache_enabled = False
    c = d.controls(ek, ["control_idx", "decision_seq", "action", "gripper_qpos", "is_settle"])
    g = c["gripper_qpos"]
    ap = (g[:, 0] - g[:, 1]) / 2.0 if g.shape[1] >= 2 else g[:, 0]
    return pd.DataFrame(dict(episode_key=ek, control_idx=c["control_idx"], decision_seq=c["decision_seq"],
                             cmd=c["action"][:, 6], aperture=ap, settle=c["is_settle"]))


def load_arm(arm, workers=12):
    d = reader.open_arm(RUN, arm)
    d.cache_enabled = False
    eps = d.episodes()
    eps = eps[eps["init"] < 30]                                   # rule 1: discovery inits only
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(episode_table, [(arm, ek) for ek in eps.episode_key]))
    ctrl = pd.concat(frames, ignore_index=True)
    truth = pd.read_csv(PF / arm / "controls_truth.csv", usecols=["episode_key", "control_idx", "truth_stage", "init", "task_id", "success"])
    truth = truth[truth["init"] < 30]
    ctrl = ctrl.merge(truth, on=["episode_key", "control_idx"], how="left")
    ef = pd.read_csv(PF / arm / "episodes_forensics.csv", usecols=["episode_key", "init", "task_id", "success", "label", "onset_control"])
    ef = ef[ef["init"] < 30]
    return ctrl, ef


def closed_runs(ctrl, hold_controls=10):
    """Mark controls where the close command has been held for >= hold_controls controls."""
    out = np.zeros(len(ctrl), bool)
    for ek, idx in ctrl.groupby("episode_key").indices.items():
        idx = idx[np.argsort(ctrl.control_idx.values[idx])]
        closed = ctrl.cmd.values[idx] > 0
        run = 0
        for j, c in enumerate(closed):
            run = run + 1 if c else 0
            out[idx[j]] = run >= hold_controls
    return out


def main(argv=None):
    arms = argv or ["r8_pi05_l10_50_A", "r8_pi05_spatial_50_A", "r8_groot_l10_50_A"]
    for arm in arms:
        ctrl, ef = load_arm(arm)
        ctrl["held"] = closed_runs(ctrl)
        cal = ctrl[(ctrl["init"] < 20) & ctrl.held & ~ctrl.settle]
        hold = cal[cal.truth_stage.eq("carry")].aperture
        empty = cal[cal.truth_stage.isin(["approach", "grasp_window"])].aperture
        print(f"\n== {arm} (calibration inits 0-19): aperture while holding (truth carry) p1/p10/p50 = {hold.quantile([.01,.1,.5]).round(4).tolist()}  n={len(hold)}"
              f"\n   aperture with close held but no object (approach/grasp_window) p50/p90/p99 = {empty.quantile([.5,.9,.99]).round(4).tolist()}  n={len(empty)}")
        thr = float(hold.quantile(0.01))
        for t in (thr, 0.5 * thr, hold.quantile(0.001)):
            ev = ctrl[(ctrl["init"] >= 20) & ~ctrl.settle].copy()
            ev["flag"] = ev.held & (ev.aperture < t)
            first = ev[ev.flag].groupby("episode_key").control_idx.min()
            ep = ef[ef["init"] >= 20].set_index("episode_key")
            ep["first_flag"] = first
            fails = ep[~ep.success.astype(bool)]
            gm = fails[fails.label.eq("grasp_miss")]
            succ = ep[ep.success.astype(bool)]
            lead = (gm.onset_control - gm.first_flag).dropna()
            print(f"   thr={t:.4f}: grasp-miss failures flagged {gm.first_flag.notna().sum()}/{len(gm)} (all failures {fails.first_flag.notna().sum()}/{len(fails)}); "
                  f"flag in successes {succ.first_flag.notna().sum()}/{len(succ)}; flag minus onset control p50={lead.median() if len(lead) else float('nan'):.0f} (negative = flag before onset)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
