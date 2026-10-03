"""Script exhaustion: when does the retrieved demonstration reach its end, what happens afterwards (round 6, Q1),
and the fitted thresholds of the exhaustion trigger (Q2, inits 0-19 only).

Per episode (any ledger from ``ledger5`` / ``ledger6``): the first fresh look whose top-1 row is within ``k`` rows of
its demonstration's end (k = 0: the demo's last row), the last decision, success, the first guard call at/after that
look, the pace-lag trigger (lag >= 12 at a fresh decision <= 80, the frozen escalation rule), the policy's share of the
decisions after the end, and the exhaustion trigger "top-1 at the end AND >= d decisions after the first end look".

Threshold rule (stated before applying it): among end_rows k in {0,1,2} and dwell d in {0,2,4,6}, keep the pairs whose
trigger fires in at most 10 % of SUCCESSFUL episodes of the fit judge runs (only-no-progress + 4 B replicates,
inits 0-19) in both models; pick the one that fires earliest after the first end look (smallest d, then smallest k).
RULE 1: ledgers via ``ledger5.load`` (init < 30 asserted); fit quantities restricted to init < 20.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import dump
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog
from exp.offline_search.rounds.r09.explore_opus.round5.tools.ledger5 import load
from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import CACHE, JUDGE, clean
from exp.offline_search.rounds.r09.explore_opus.round6.tools import OUT

FA_MAX = 0.10
GRID_K, GRID_D = (0, 1, 2), (0, 2, 4, 6)
LAG_THR, LAG_DEADLINE = 12, 80

SCREEN = {  # inits 20-29 (descriptive only)
    "pi05": {"cache": CACHE["pi05"] + [("r09_opus_escalation", "r9o_pi05_l10_50_cache"), ("r09_opus_r3", "r9o3_pi05_l10_50_cache"),
                                       ("r09_opus_r4", "r9o4_pi05_l10_50_cache")],
             "corrector": [("r09_fable_r2", "r9f2_pi05_l10_50_corr05pt"), ("r09_opus_r4", "r9o4_pi05_l10_50_corr")],
             "only_no_progress": [("r08_abl_t107", "r8abl_onlynp_p_l10_50"), ("r09_opus_escalation", "r9o_pi05_l10_50_onlynp")],
             "cache+escalation": [("r09_opus_escalation", "r9o_pi05_l10_50_esc"), ("r09_opus_escalation", "r9o_pi05_l10_50_esc_w24"),
                                  ("r09_opus_r3", "r9o3_pi05_l10_50_esc_w24")],
             "only_no_progress+escalation": [("r09_opus_escalation", "r9o_pi05_l10_50_onlynp_esc")],
             "stack": [("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_esc"), ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_gm_esc"),
                       ("r09_opus_r5", "r9o5_pi05_l10_50_stack")]},
    "groot": {"cache": CACHE["groot"] + [("r09_opus_escalation", "r9o_groot_l10_50_cache"), ("r09_opus_r3", "r9o3_groot_l10_50_cache"),
                                         ("r09_opus_r4", "r9o4_groot_l10_50_cache")],
              "corrector": [("r09_fable_r2", "r9f2_groot_l10_50_corr05pt"), ("r09_opus_r4", "r9o4_groot_l10_50_corr")],
              "only_no_progress": [("r08_abl_t107", "r8abl_onlynp_g_l10_50"), ("r09_opus_escalation", "r9o_groot_l10_50_onlynp")],
              "cache+escalation": [("r09_opus_escalation", "r9o_groot_l10_50_esc"), ("r09_opus_escalation", "r9o_groot_l10_50_esc_w24"),
                                   ("r09_opus_r3", "r9o3_groot_l10_50_esc_w24")],
              "only_no_progress+escalation": [("r09_opus_escalation", "r9o_groot_l10_50_onlynp_esc")],
              "stack": [("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05"), ("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05_gmS"),
                        ("r09_opus_r5", "r9o5_groot_l10_50_stack")]}}


def episodes(model, arms, lo, hi, k=0, d=2):
    cat = catalog((model, "l10", 50))
    rte, lstep = (cat.ep_len - 1 - cat.step).values, cat.step.values
    rows = []
    for root, arm in arms:
        D, E = load(root, arm)
        D = D[(D.init >= lo) & (D.init < hi)]
        E = E.set_index(["task", "init"])
        for (t, i), e in D.groupby(["task", "init"]):
            e = e.sort_values("step")
            v = e[e.vision & (e.top1 >= 0)]
            s, r = v.step.values, rte[v.top1.values]
            lag = s - lstep[v.top1.values]
            lt = s[(lag >= LAG_THR) & (s <= LAG_DEADLINE)]
            at = np.flatnonzero(r <= k)
            t_end = s[at[0]] if len(at) else np.nan
            fire = np.nan
            if len(at):
                cand = np.flatnonzero((s >= t_end + d) & (r <= k))
                fire = s[cand[0]] if len(cand) else np.nan
            after = e[e.step > t_end] if len(at) else e.iloc[:0]
            calls_after = e[e.call & (e.step >= t_end)] if len(at) else e.iloc[:0]
            rows.append(dict(arm=arm, task=t, init=i, success=bool(E.loc[(t, i), "success"]), t_last=int(e.step.max()),
                             t_end=t_end, fire=fire, t_lag=lt[0] if len(lt) else np.nan,
                             first_call_after_end=calls_after.step.min() if len(calls_after) else np.nan,
                             policy_share_after_end=float(after.src.isin(["policy", "policy_tail"]).mean()) if len(after) else np.nan,
                             calls_after_end=int(len(calls_after))))
    return pd.DataFrame(rows)


def fit_thresholds():
    grid = []
    for k in GRID_K:
        for d in GRID_D:
            row = dict(k=k, d=d)
            for model in ("pi05", "groot"):
                X = episodes(model, JUDGE[model], 0, 20, k, d)
                s, f = X[X.success], X[~X.success]
                row[f"{model}_fa"] = float(s.fire.notna().mean())
                row[f"{model}_detect"] = float(f.fire.notna().mean())
                row[f"{model}_fire_minus_guard"] = float(np.nanmedian((f.fire - f.first_call_after_end)[f.fire.notna()]))
            grid.append(row)
    G = pd.DataFrame(grid)
    ok = G[(G.pi05_fa <= FA_MAX) & (G.groot_fa <= FA_MAX)].sort_values(["d", "k"])
    best = ok.iloc[0]
    return G, int(best.k), int(best.d)


def summarize(X):
    s, f = X[X.success], X[~X.success]
    out = dict(episodes=int(len(X)), sr=float(X.success.mean()),
               fail_reach_end=float(f.t_end.notna().mean()) if len(f) else None,
               fail_t_end_median=float(np.nanmedian(f.t_end)) if f.t_end.notna().any() else None,
               succ_reach_end=float(s.t_end.notna().mean()) if len(s) else None,
               succ_gap_after_end_q50_q90_q95=[float(x) for x in np.nanpercentile(s.t_last - s.t_end, [50, 90, 95])]
               if s.t_end.notna().any() else None)
    for dd in (2, 4, 6, 8):
        run = X[X.t_end.notna() & (X.t_last >= X.t_end + dd)]
        out[f"still_running_at_end+{dd}"] = dict(n=int(len(run)), share=float(len(run) / len(X)),
                                                 sr=float(run.success.mean()) if len(run) else None)
    ff = f[f.t_end.notna()]
    out["fail_first_guard_call_after_end_median"] = float(np.nanmedian(ff.first_call_after_end - ff.t_end)) if len(ff) else None
    out["fail_lag_trigger_minus_end_median"] = float(np.nanmedian(ff.t_lag - ff.t_end)) if ff.t_lag.notna().any() else None
    out["fail_lag_trigger_before_end_share"] = float((ff.t_lag < ff.t_end).mean()) if len(ff) else None
    out["fail_policy_share_after_end"] = float(ff.policy_share_after_end.mean()) if len(ff) else None
    return out


def by_end_time(X):
    run = X[X.t_end.notna() & (X.t_last >= X.t_end + 4)].copy()
    run["bin"] = pd.cut(run.t_end, [0, 30, 40, 50, 60, 80, 110]).astype(str)
    return {b: dict(n=int(len(g)), sr=float(g.success.mean())) for b, g in run.groupby("bin")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    pd.set_option("display.width", 250)
    G, k, d = fit_thresholds()
    print(G.round(3).to_string())
    print(f"fitted: end_rows k = {k}, dwell d = {d}")
    rep = dict(fit_grid=G.to_dict("records"), end_rows=k, dwell=d, fa_max=FA_MAX, groups={})
    for model in ("pi05", "groot"):
        groups = {"fit 0-19 cache": (CACHE[model], 0, 20), "fit 0-19 judge (only-np + B)": (JUDGE[model], 0, 20)}
        groups.update({f"20-29 {g}": (arms, 20, 30) for g, arms in SCREEN[model].items()})
        for name, (arms, lo, hi) in groups.items():
            X = episodes(model, arms, lo, hi, k, d)
            sm = summarize(X)
            if name.startswith("fit"):
                sm["recovery_by_end_time"] = by_end_time(X)
            s = X[X.success]
            f = X[~X.success]
            sm["trigger_fires_in_successes"] = float(s.fire.notna().mean()) if len(s) else None
            sm["trigger_fires_in_failures"] = float(f.fire.notna().mean()) if len(f) else None
            rep["groups"][f"{model} | {name}"] = sm
            r = sm["still_running_at_end+6"]
            print(f"{model} | {name:40s} SR {sm['sr']:.2f} | fail reach end {sm['fail_reach_end']:.2f} (median dec {sm['fail_t_end_median']}) "
                  f"| run>=end+6: {r['share']:.2f} SR {r['sr'] if r['sr'] is None else round(r['sr'], 2)} "
                  f"| guard call {sm['fail_first_guard_call_after_end_median']} dec after end, lag trigger {sm['fail_lag_trigger_minus_end_median']} "
                  f"| policy share after end {sm['fail_policy_share_after_end'] if sm['fail_policy_share_after_end'] is None else round(sm['fail_policy_share_after_end'], 2)}")
        X = episodes(model, JUDGE[model], 0, 20, k, d)
        print(model, "recovery (running >= 4 dec after end) by first-end decision, fit judge:", by_end_time(X))
    dump(OUT / "exhaust.json", clean(rep))


if __name__ == "__main__":
    main()
