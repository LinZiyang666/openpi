"""Round-5 per-decision ledger with the guard's own extras (standard-mode server logs, any judge stack).

One row per committed decision of the accepted attempt: decision index, look/hit/source, the retrieved top-1 row,
the guard verdict (``os_reason``: 4 = no-progress, 91 = pace-lag escalation, 93 = empty grasp), the no-progress
span, retrieval confidence, motion, the first 8 robot-state dims, and the escalation extras when present.
The pace lag (decision index - library step of the top-1 row) is added from the r08_main catalog.

RULE 1: server records and journals are admitted through ``iter_jsonl_discovery`` which drops init >= 30 BEFORE
decoding; holdout roots (``r09_holdout*``, ``r09_astra_holdout*``) are refused by ``check_root``; the output is
asserted to contain inits 0-29 only.  Output: ``DERIVED/r5ledger/<root>__<arm>.parquet`` (+ ``_ep`` episodes).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import (
    RUNS, DERIVED, PRICE, iter_jsonl_discovery, parse_uid, check_root)
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog

OUTDIR = DERIVED / "r5ledger"

# (root, arm, model) -- L10, 50-demo library, both models.  FIT runs cover inits 0-29 (0-19 used for fitting);
# screen runs cover inits 20-29 only.
ARMS = [
    # fit: R8 ablation on timan107 (cache A, B, only-no-progress)
    ("r08_abl_t107", "r8abl_onlynp_p_l10_50", "pi05"), ("r08_abl_t107", "r8abl_onlynp_g_l10_50", "groot"),
    ("r08_abl_t107", "r8abl_ctrlA_p_l10_50", "pi05"), ("r08_abl_t107", "r8abl_ctrlA_g_l10_50", "groot"),
    ("r08_abl_t107", "r8abl_ctrlB_p_l10_50", "pi05"), ("r08_abl_t107", "r8abl_ctrlB_g_l10_50", "groot"),
    # screen: leading stacks and their relatives (fable r3c)
    ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_esc", "pi05"), ("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05", "groot"),
    ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_gm", "pi05"), ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_gm_esc", "pi05"),
    ("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05_gmS", "groot"), ("r09_fable_r3c", "r9f3c_groot_l10_50_corr05_gmS", "groot"),
    # screen: my round-2 escalation runs (only-no-progress with and without escalation, cache)
    ("r09_opus_escalation", "r9o_pi05_l10_50_onlynp", "pi05"), ("r09_opus_escalation", "r9o_groot_l10_50_onlynp", "groot"),
    ("r09_opus_escalation", "r9o_pi05_l10_50_onlynp_esc", "pi05"), ("r09_opus_escalation", "r9o_groot_l10_50_onlynp_esc", "groot"),
    ("r09_opus_escalation", "r9o_pi05_l10_50_cache", "pi05"), ("r09_opus_escalation", "r9o_groot_l10_50_cache", "groot"),
    # screen: fable round 2 "np_corr05pt" (corrector not active on the judge path => only-no-progress)
    ("r09_fable_r2", "r9f2_pi05_l10_50_np_corr05pt", "pi05"),
    # replicates on inits 0-29: B (all four guards; no-progress dominates its calls) and the pure cache
    ("r05_q1", "r5q1_c10_p_l10_50", "pi05"), ("r06_paper", "r5q1_c10_p_l10_50_rep2", "pi05"),
    ("r06_paper", "r5q1_c10_p_l10_50_rep3", "pi05"), ("r06_paper", "r6p1_c10_g_l10_50", "groot"),
    ("r06_paper", "r6p1_c10_g_l10_50_rep2", "groot"), ("r06_paper", "r6p1_c10_g_l10_50_rep3", "groot"),
    ("r05_ptail", "r5t_p_l10_50_tail1uc", "pi05"), ("r06_paper", "r5t_p_l10_50_tail1uc_rep2", "pi05"),
    ("r06_paper", "r5t_p_l10_50_tail1uc_rep3", "pi05"), ("r08_main", "r8_pi05_l10_50_A", "pi05"),
    ("r05_x", "r5x_g_l10_50_tail1u", "groot"), ("r06_paper", "r5x_g_l10_50_tail1u_rep2", "groot"),
    ("r06_paper", "r5x_g_l10_50_tail1u_rep3", "groot"), ("r08_main", "r8_groot_l10_50_A", "groot"),
    # screen inits 20-29: corrector-only arms (the gated stack's prefix controller) and same-batch caches
    ("r09_fable_r2", "r9f2_pi05_l10_50_corr05pt", "pi05"), ("r09_fable_r2", "r9f2_groot_l10_50_corr05pt", "groot"),
    ("r09_opus_r4", "r9o4_pi05_l10_50_corr", "pi05"), ("r09_opus_r4", "r9o4_groot_l10_50_corr", "groot"),
    ("r09_opus_r4", "r9o4_pi05_l10_50_cache", "pi05"), ("r09_opus_r4", "r9o4_groot_l10_50_cache", "groot"),
    ("r09_opus_r4", "r9o4_pi05_l10_50_corr_esc_w24", "pi05"), ("r09_opus_r4", "r9o4_groot_l10_50_corr_esc_w24", "groot"),
    ("r09_opus_r3", "r9o3_pi05_l10_50_cache", "pi05"), ("r09_opus_r3", "r9o3_groot_l10_50_cache", "groot"),
]


def journal(root, arm):
    out = {}
    for r in iter_jsonl_discovery(check_root(RUNS / root) / "runs" / arm / "client" / "journal.jsonl"):
        if r.get("accepted") and r.get("phase") in (None, "eval"):
            t, i = parse_uid(r["task_uid"])
            assert i < 30
            out[(t, i)] = (bool(r.get("success")), int(r.get("attempt", 1)))
    return out


def _f(ex, k):
    v = ex.get(k)
    return float(v) if v is not None else np.nan


def extract(root, arm, model, overwrite=False):
    out = OUTDIR / f"{root}__{arm}.parquet"
    if out.exists() and not overwrite:
        return str(out)
    J = journal(root, arm)
    st = catalog((model, "l10", 50)).step.values
    recs = []
    for path in sorted((check_root(RUNS / root) / "runs" / arm).glob("server_*/decisions*.jsonl")):
        for r in iter_jsonl_discovery(path, uid_key="uid"):          # init >= 30 dropped before decoding
            if r.get("ev") != "dec":
                continue
            t, i = parse_uid(r["uid"])
            if (t, i) not in J or int(r.get("attempt", 1)) != J[(t, i)][1]:
                continue
            ex = r.get("extras") or {}
            rs = list(r.get("robot_state") or [])[:8]
            rs += [np.nan] * (8 - len(rs))
            top1 = int(r["top1"]) if r.get("top1") is not None else -1
            recs.append(dict(task=t, init=i, step=int(r.get("step", -1)), vision=bool(r.get("vision")),
                             hit=bool(r.get("hit")), src=str(r.get("src")), top1=top1,
                             pace_lag=float(int(r.get("step", -1)) - st[top1]) if (top1 >= 0 and r.get("vision")) else np.nan,
                             reason=_f(ex, "os_reason"), flags=_f(ex, "os_flags"), span=_f(ex, "noprog_span"),
                             conf=_f(ex, "os_conf_raw"), motion=_f(ex, "motion"), top1_prog=_f(ex, "top1_prog"),
                             gexec=_f(ex, "gexec"), r9o_lag=_f(ex, "r9o_lag"), r9o_esc=_f(ex, "r9o_escalated"),
                             gm_burst=_f(ex, "r9f3_grasp_burst"), look_reason=float(r.get("look_reason") or 0),
                             **{f"rs{k}": float(rs[k]) for k in range(8)}))
    D = pd.DataFrame(recs)
    if D.empty:
        return None
    assert D["init"].max() < 30, "RULE 1"
    D = D.sort_values(["task", "init", "step"]).drop_duplicates(["task", "init", "step"], keep="last").reset_index(drop=True)
    v, m = PRICE[model]
    D["call"] = D.vision & ~D.hit
    D["cost"] = np.where(D.vision, v, 0.0) + np.where(D.call, m, 0.0)
    E = D.groupby(["task", "init"]).agg(decisions=("step", "size"), looks=("vision", "sum"), calls=("call", "sum"),
                                         cost=("cost", "sum"), last_step=("step", "max")).reset_index()
    E["success"] = [J[(int(t), int(i))][0] for t, i in zip(E.task, E.init)]
    E["root"], E["arm"], E["model"] = root, arm, model
    OUTDIR.mkdir(parents=True, exist_ok=True)
    D.to_parquet(out)
    E.to_parquet(OUTDIR / f"{root}__{arm}_ep.parquet")
    return str(out)


def load(root, arm):
    D = pd.read_parquet(OUTDIR / f"{root}__{arm}.parquet")
    E = pd.read_parquet(OUTDIR / f"{root}__{arm}_ep.parquet")
    assert D["init"].max() < 30 and E["init"].max() < 30, "RULE 1"
    return D, E


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(extract, r, arm, m, a.overwrite) for r, arm, m in ARMS]
        for (r, arm, m), f in zip(ARMS, futs):
            print(r, arm, f.result())


if __name__ == "__main__":
    main()
