"""Residual failure anatomy of the leading stacks on screen inits 20-29 (round 5, question 3).

Which (task, init) pairs fail under the stack but succeed under the pure policy, and what kind of failure is it?
No simulator truth or debug predicate exists in standard mode, so failures are typed from the controller's own
signals: the retrieved demo progress (top-1 library row), the guard verdicts, the pace lag and the gripper.

Types (task-agnostic; from the peak retrieved demo progress):
  demo_end_miss  the retrieved demo was played to (near) its end (peak progress >= 0.95) and the episode still timed
                 out: the cache "finished the script" but the task was not accomplished
  stall_late     peak progress in [0.6, 0.95): stuck in the second half of the demo
  stall_mid      peak in [0.3, 0.6)
  stall_early    peak < 0.3
plus whether the policy held control at the end (escalated, or >= 5 guard calls in the last 20 decisions).

Replicates on inits 20-29: pure policy (P10) 5 runs pi0.5 / 4 runs GR00T, pure cache (catalog 7 + 3 later runs),
corrector-only 2 runs, the leading stack plus its near-identical "+ empty-grasp" sibling (grasp calls fire in
<= 0.11 episodes' worth of decisions; reported separately).

RULE 1: journals via ``iter_jsonl_discovery`` (init >= 30 dropped before decoding), ledgers via ``ledger5.load``;
holdout roots refused by ``check_root``.  Only inits 20-29 are analysed here (screen inits; nothing is fitted).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, dump
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog
from exp.offline_search.rounds.r09.explore_opus.round3.tools.dissect import journal
from exp.offline_search.rounds.r09.explore_opus.round5.tools import OUT
from exp.offline_search.rounds.r09.explore_opus.round5.tools.ledger5 import load
from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import clean

P10 = {"pi05": [("r08_main", "r8_pi05_l10_P10"), ("r09_astra_confirmation", "r9_astra_pi05_l10_P10"),
                ("r09_astra_round2_eval", "r9r2_astra_pi05_l10_P10"), ("r09_astra_round2b", "r9r2b_astra_pi05_l10_P10"),
                ("r09_fable_grown", "r9f_P10_p_l10")],
       "groot": [("r08_main", "r8_groot_l10_P10"), ("r09_astra_confirmation", "r9_astra_groot_l10_P10"),
                 ("r09_astra_round2_eval", "r9r2_astra_groot_l10_P10"), ("r09_astra_round2b_g", "r9r2b_astra_groot_l10_P10")]}
CACHE20 = {"pi05": [("r09_opus_escalation", "r9o_pi05_l10_50_cache"), ("r09_opus_r3", "r9o3_pi05_l10_50_cache"),
                    ("r09_opus_r4", "r9o4_pi05_l10_50_cache")],
           "groot": [("r09_opus_escalation", "r9o_groot_l10_50_cache"), ("r09_opus_r3", "r9o3_groot_l10_50_cache"),
                     ("r09_opus_r4", "r9o4_groot_l10_50_cache")]}
CORR = {"pi05": [("r09_fable_r2", "r9f2_pi05_l10_50_corr05pt"), ("r09_opus_r4", "r9o4_pi05_l10_50_corr")],
        "groot": [("r09_fable_r2", "r9f2_groot_l10_50_corr05pt"), ("r09_opus_r4", "r9o4_groot_l10_50_corr")]}
STACK = {"pi05": ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_esc"), "groot": ("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05")}
SIBLING = {"pi05": ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_gm_esc"), "groot": ("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05_gmS")}


def outcomes(arms):
    rows = []
    for root, arm in arms:
        for (t, i), (s, _) in journal(RUNS / root, arm).items():
            if 20 <= i < 30:
                rows.append(dict(task=t, init=i, success=float(s), arm=arm))
    return pd.DataFrame(rows)


def pair_table(model):
    from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import pair_p
    pp = pair_p(model)
    pp = pp[pp.init >= 20]
    P = outcomes(P10[model]).groupby(["task", "init"]).success.agg(p_policy="mean", n_policy="size")
    Cx = outcomes(CACHE20[model])
    cat_cache = pp.set_index(["task", "init"])[["p_cache", "n_p_cache"]]
    cx = Cx.groupby(["task", "init"]).success.agg(["sum", "size"])
    k = cat_cache.p_cache * cat_cache.n_p_cache + cx["sum"].reindex(cat_cache.index, fill_value=0)
    n = cat_cache.n_p_cache + cx["size"].reindex(cat_cache.index, fill_value=0)
    C = pd.DataFrame(dict(p_cache=k / n, n_cache=n))
    R = outcomes(CORR[model]).groupby(["task", "init"]).success.agg(p_corr="mean", n_corr="size")
    S = outcomes([STACK[model]]).set_index(["task", "init"]).success.rename("s_stack")
    S2 = outcomes([SIBLING[model]]).set_index(["task", "init"]).success.rename("s_sibling")
    T = pd.concat([P, C, R, S, S2], axis=1).reset_index()
    assert T.init.min() >= 20 and T.init.max() < 30
    return T


def failure_features(model, root, arm):
    D, E = load(root, arm)
    cat = catalog((model, "l10", 50))
    prog = cat.progress.values
    rows = []
    for (t, i), e in D.groupby(["task", "init"]):
        if bool(E[(E.task == t) & (E.init == i)].success.iloc[0]):
            continue
        v = e[e.vision & (e.top1 >= 0)].sort_values("step")
        p = prog[v.top1.values]
        best = np.maximum.accumulate(p)
        last_adv = v.step.values[np.flatnonzero(np.diff(best, prepend=-1) > 0.02)[-1]]
        n_dec = int(e.step.max()) + 1
        tail = e[e.step >= n_dec - 20]
        tail_calls = int(tail.call.sum())
        stalled_tail = (n_dec - 1 - last_adv) >= 20
        peak = float(best[-1])
        esc = e[(e.reason == 91.0) & e.call]
        grip = e.sort_values("step").iloc[-1]
        ftype = ("demo_end_miss" if peak >= 0.95 else "stall_late" if peak >= 0.6 else "stall_mid" if peak >= 0.3
                 else "stall_early")
        takeover = "policy_took_over" if (len(esc) or tail_calls >= 5) else "cache_only_tail"
        rows.append(dict(task=t, init=i, decisions=n_dec, calls=int(e.call.sum()), np_calls=int((e.call & (e.reason == 4.0)).sum()),
                         esc_calls=int(len(esc)), esc_step=float(esc.step.min()) if len(esc) else np.nan,
                         peak_progress=round(peak, 3), last_progress_gain_step=int(last_adv), tail20_calls=tail_calls,
                         final_lag=float(v.pace_lag.values[-1]), max_lag=float(np.nanmax(v.pace_lag.values)),
                         final_state_aperture=float(grip.rs6 - grip.rs7), type=ftype, tail=takeover))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    pd.set_option("display.width", 250)
    rep = {}
    for model in ("pi05", "groot"):
        T = pair_table(model)
        F = failure_features(model, *STACK[model]).merge(T, on=["task", "init"], how="left")
        F2 = failure_features(model, *SIBLING[model]).merge(T, on=["task", "init"], how="left")
        lost = F[F.p_policy >= 0.5]
        print(f"===== {model}: stack failures {len(F)}; of which pure policy succeeds (>= half of {int(T.n_policy.max())} runs) {len(lost)}")
        print(F.sort_values(["type", "task"]).to_string(index=False))
        print("sibling failures:")
        print(F2.sort_values(["type", "task"])[["task", "init", "type", "peak_progress", "calls", "esc_calls", "p_policy", "p_cache", "p_corr", "s_stack"]].to_string(index=False))
        both = T[(T.s_stack == 0) & (T.s_sibling == 0)]
        print(f"pairs failing in both stack runs: {len(both)}; with p_policy>=.5: {int((both.p_policy >= .5).sum())}")
        # pairs where pure policy beats the stack's pooled rate
        T["p_stack"] = T[["s_stack", "s_sibling"]].mean(1)
        rep[model] = dict(failures=F.to_dict("records"), sibling_failures=F2.to_dict("records"),
                          type_counts=F.type.value_counts().to_dict(), type_counts_sibling=F2.type.value_counts().to_dict(),
                          lost_vs_policy=int(len(lost)), both_fail=int(len(both)), both_fail_policy_ok=int((both.p_policy >= .5).sum()),
                          sr=dict(stack=float(T.s_stack.mean()), sibling=float(T.s_sibling.mean()), policy=float(T.p_policy.mean()),
                                  cache=float(T.p_cache.mean()), corrector=float(T.p_corr.mean())),
                          policy_minus_stack_by_pcache={b: dict(pairs=int(len(g)), p_stack=float(g.p_stack.mean()), p_policy=float(g.p_policy.mean()))
                                                        for b, g in T.groupby(pd.cut(T.p_cache, [-.01, 0, .5, .99, 1.0]).astype(str))})
        T.to_parquet(OUT / f"pairs20_{model}.parquet")
    dump(OUT / "anatomy5.json", clean(rep))


if __name__ == "__main__":
    main()
