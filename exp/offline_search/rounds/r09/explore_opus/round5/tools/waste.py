"""Which no-progress guard calls are wasted?  (round 5, question 1)

Inputs: ``ledger5`` per-decision ledgers (inits 0-29 only, asserted) of the only-no-progress arm, the B replicates
(their calls are mostly no-progress calls), the leading stacks (inits 20-29) and the pure-cache replicates; pooled
per-pair success of the pure cache (7 replicates) and of the pure policy (2 replicates) from the round-2 catalog
(inits 0-29 only).

The guard statistic is replayed exactly from the logged top-1 rows (``span``: decisions since the retrieved demo
progress last advanced by more than half a library step; validated: 0 mismatches against the logged span).  On the
pure-cache arms the same replay gives *virtual* firings -- where the guard would have fired -- so the cache's own
recovery after a stall can be compared with the recovery after a real call.

Per call / firing:  step, pace lag (decision - library step of top-1), span, index of the call inside its stall run
(a run = consecutive firings without the span resetting), calls before it in the episode, whether the next look
shows progress (span reset), whether the episode succeeded, and the pair's pooled pure-cache success.

RULE 1: everything is read through ``ledger5.load`` / the round-2 catalog (both asserted init < 30).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import dump
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog
from exp.offline_search.rounds.r09.explore_opus.round5.tools.ledger5 import load
from exp.offline_search.rounds.r09.explore_opus.round5.tools import OUT

PROG_EPS = 0.5
NOPROG_N = 3                       # fires when span >= NOPROG_N - 1 (frozen B)

JUDGE = {"pi05": [("r08_abl_t107", "r8abl_onlynp_p_l10_50"), ("r05_q1", "r5q1_c10_p_l10_50"),
                  ("r06_paper", "r5q1_c10_p_l10_50_rep2"), ("r06_paper", "r5q1_c10_p_l10_50_rep3"),
                  ("r08_abl_t107", "r8abl_ctrlB_p_l10_50")],
         "groot": [("r08_abl_t107", "r8abl_onlynp_g_l10_50"), ("r06_paper", "r6p1_c10_g_l10_50"),
                   ("r06_paper", "r6p1_c10_g_l10_50_rep2"), ("r06_paper", "r6p1_c10_g_l10_50_rep3"),
                   ("r08_abl_t107", "r8abl_ctrlB_g_l10_50")]}
CACHE = {"pi05": [("r05_ptail", "r5t_p_l10_50_tail1uc"), ("r06_paper", "r5t_p_l10_50_tail1uc_rep2"),
                  ("r06_paper", "r5t_p_l10_50_tail1uc_rep3"), ("r08_main", "r8_pi05_l10_50_A"),
                  ("r08_abl_t107", "r8abl_ctrlA_p_l10_50")],
         "groot": [("r05_x", "r5x_g_l10_50_tail1u"), ("r06_paper", "r5x_g_l10_50_tail1u_rep2"),
                   ("r06_paper", "r5x_g_l10_50_tail1u_rep3"), ("r08_main", "r8_groot_l10_50_A"),
                   ("r08_abl_t107", "r8abl_ctrlA_g_l10_50")]}
STACK = {"pi05": [("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_esc"), ("r09_opus_escalation", "r9o_pi05_l10_50_onlynp_esc"),
                  ("r09_opus_escalation", "r9o_pi05_l10_50_onlynp"), ("r09_fable_r2", "r9f2_pi05_l10_50_np_corr05pt")],
         "groot": [("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05"), ("r09_opus_escalation", "r9o_groot_l10_50_onlynp"),
                   ("r09_opus_escalation", "r9o_groot_l10_50_onlynp_esc")]}
LAG_BINS = [-1e9, -4, 0, 4, 8, 12, 20, 1e9]
LAG_LABELS = ["<=-4", "-3..0", "1..4", "5..8", "9..12", "13..20", ">20"]


def pair_p(model):
    """Pooled per-pair success (inits 0-29) of the pure cache and the pure policy (round-2 catalog)."""
    from exp.offline_search.rounds.r09.explore_opus.round2.tools.replicates import load as rload
    _, outc = rload()
    d = outc[(outc.model == model) & (outc.suite == "l10") & (outc.lib == 50)]
    assert d["init"].max() < 30
    out = {}
    for fam, key in (("cache", "p_cache"), ("policy10", "p_policy")):
        g = d[d.family == fam].groupby(["task", "init"]).success
        out[key] = g.mean()
        out["n_" + key] = g.size()
    return pd.DataFrame(out).reset_index()


def looks(D, model):
    """Every look with a retrieval; replayed span, firing, run structure, resolution at the next look."""
    cat = catalog((model, "l10", 50))
    prog, eplen, lstep = cat.progress.values, np.maximum(cat.ep_len.values - 1, 1), cat.step.values
    out = []
    for (t, i), e in D.groupby(["task", "init"], sort=True):
        e = e.sort_values("step")
        v = e[e.vision & (e.top1 >= 0)]
        hist, span, run, k_run, n_calls, n_fire = [], 0, -1, 0, 0, 0
        rows = []
        for r in v.itertuples():
            hist = [x for x in hist if x[0] < r.step] + [(r.step, prog[r.top1], eplen[r.top1])]
            span = 0
            for a, b in zip(hist[:-1], hist[1:]):
                span = span + b[0] - a[0] if (b[1] - a[1]) * b[2] <= PROG_EPS else 0
            fire = span >= NOPROG_N - 1
            if fire and (not rows or not rows[-1]["fire"]):
                run += 1
                k_run = 0
            if fire:
                k_run += 1
            reason = r.reason if r.reason == r.reason else 0.0
            call = bool(r.call)
            rows.append(dict(task=t, init=i, step=int(r.step), lag=float(r.step - lstep[r.top1]), span=int(span), fire=bool(fire),
                             call=call, reason=float(reason), np_call=bool(call and reason == 4.0),
                             run=run if fire else -1, k_run=k_run if fire else 0, n_np_before=n_calls, n_fire_before=n_fire,
                             conf=float(r.conf), xyz=(r.rs0, r.rs1, r.rs2)))
            n_calls += int(call and reason == 4.0)
            n_fire += int(fire)
        for a, b in zip(rows[:-1], rows[1:]):
            b["move_prev"] = float(np.linalg.norm(np.subtract(b["xyz"], a["xyz"])))
            b["gap_prev"] = b["step"] - a["step"]
            a["resolved_next"] = b["span"] == 0
            a["next_gap"] = b["step"] - a["step"]
            a["move_next"] = float(np.linalg.norm(np.subtract(b["xyz"], a["xyz"])))
        if rows:
            rows[0]["move_prev"], rows[0]["gap_prev"] = np.nan, np.nan
            rows[-1]["resolved_next"], rows[-1]["next_gap"], rows[-1]["move_next"] = np.nan, np.nan, np.nan
        out += rows
    L = pd.DataFrame(out).drop(columns=["xyz"])
    L["lag_bin"] = pd.cut(L.lag, LAG_BINS, labels=LAG_LABELS)
    return L


def annotate(L, E, P):
    L = L.merge(E[["task", "init", "success", "decisions"]], on=["task", "init"]).merge(P, on=["task", "init"], how="left")
    runs = L[L.fire].groupby(["task", "init", "run"]).agg(run_fires=("fire", "size"), run_calls=("np_call", "sum"))
    return L.merge(runs.reset_index(), on=["task", "init", "run"], how="left")


def anatomy(L, E, model, label):
    """Shares of no-progress call cost by ex-post category (descriptive, not a rule)."""
    c = L[L.np_call]
    n = max(len(c), 1)
    on_pace = c.lag <= 4
    repeat = c.k_run >= 2
    unresolved = c.resolved_next == False  # noqa: E712
    fail = ~c.success
    sure = c.success & (c.p_cache == 1)
    useful = c.success & (c.p_cache < 1)
    late = c.n_np_before >= 10
    res = dict(arm=label, model=model, episodes=int(len(E)), np_calls=int(len(c)), np_calls_per_ep=len(c) / len(E),
               ir=float(E.cost.sum() / E.decisions.sum()), sr=float(E.success.mean()),
               np_ir_share=float(len(c) * 1.0 / E.decisions.sum()),
               on_pace=float(on_pace.mean()), repeat_in_run=float(repeat.mean()), unresolved_next=float(unresolved.mean()),
               in_failed_episode=float(fail.mean()), in_always_succeed_pair_and_succeeded=float(sure.mean()),
               in_succeeded_pair_cache_below_1=float(useful.mean()), after_10_calls=float(late.mean()),
               after_10_calls_in_failed=float((late & fail).mean()),
               on_pace_and_succeeded=float((on_pace & c.success).mean()),
               on_pace_resolved=float((on_pace & (c.resolved_next == True)).mean()),  # noqa: E712
               union_wasted_expost=float((fail | sure).mean()))
    return res


def resolution(Lj, Lc):
    """P(progress at the next look) after a real call vs after a virtual firing of the pure cache, by lag bin,
    first firing in a run only (the call/firing that starts the stall)."""
    rows = []
    for name, L, sel in (("call", Lj, Lj.np_call & (Lj.k_run == 1)), ("cache_virtual", Lc, Lc.fire & (Lc.k_run == 1))):
        x = L[sel & L.resolved_next.notna()]
        for b, g in x.groupby("lag_bin", observed=True):
            rows.append(dict(kind=name, lag_bin=str(b), n=int(len(g)), resolved_next=float(g.resolved_next.astype(float).mean()),
                             episode_success=float(g.success.mean()), run_len=float(g.run_fires.mean())))
    return pd.DataFrame(rows)


def survival(L, E, ks=(3, 5, 8, 10, 12, 15, 20, 25)):
    """Per-episode call cap: calls beyond k and the successes among episodes that needed more than k calls."""
    c = L[L.np_call].groupby(["task", "init"]).size().reindex(pd.MultiIndex.from_frame(E[["task", "init"]]), fill_value=0)
    s = E.set_index(["task", "init"]).success.reindex(c.index)
    rows = []
    for k in ks:
        over = c > k
        rows.append(dict(cap=k, eps_over=int(over.sum()), sr_over=float(s[over].mean()) if over.any() else np.nan,
                         successes_over=int(s[over].sum()), calls_saved_share=float((c - k).clip(lower=0).sum() / max(c.sum(), 1)),
                         calls_saved_per_ep=float((c - k).clip(lower=0).sum() / len(E))))
    return pd.DataFrame(rows)


def build(model, part):
    P = pair_p(model)
    lo, hi = (0, 20) if part == "fit" else (20, 30)
    Ls, Es = [], []
    for root, arm in JUDGE[model] + (STACK[model] if part == "eval" else []):
        D, E = load(root, arm)
        D, E = D[(D.init >= lo) & (D.init < hi)], E[(E.init >= lo) & (E.init < hi)]
        if E.empty:
            continue
        L = annotate(looks(D, model), E, P)
        L["arm"] = arm
        Ls.append(L)
        Es.append(E.assign(arm=arm))
    Lc = []
    for root, arm in CACHE[model]:
        D, E = load(root, arm)
        D, E = D[(D.init >= lo) & (D.init < hi)], E[(E.init >= lo) & (E.init < hi)]
        L = annotate(looks(D, model), E, P)
        L["arm"] = arm
        Lc.append(L)
    return pd.concat(Ls), pd.concat(Es), pd.concat(Lc), P


def clean(x):
    """NaN -> None recursively (strict JSON)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    return x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    report = {}
    pd.set_option("display.width", 250)
    for model in ("pi05", "groot"):
        for part in ("fit", "eval"):
            Lj, Ej, Lc, P = build(model, part)
            an = [anatomy(Lj[Lj.arm == a], Ej[Ej.arm == a], model, a) for a in Ej.arm.unique()]
            res = resolution(Lj[Lj.arm.isin([a for _, a in JUDGE[model]])], Lc)
            sv = {a: survival(Lj[Lj.arm == a], Ej[Ej.arm == a]).to_dict("records") for a in Ej.arm.unique()}
            report[f"{model}_{part}"] = dict(anatomy=an, resolution=res.to_dict("records"), survival=sv)
            print(f"===== {model} {part} (inits {'0-19' if part == 'fit' else '20-29'})")
            print(pd.DataFrame(an).round(3).to_string())
            print(res.round(3).to_string())
            Lj.to_parquet(OUT / f"looks_{model}_{part}.parquet")
            Lc.to_parquet(OUT / f"looks_cache_{model}_{part}.parquet")
    dump(OUT / "waste.json", clean(report))


if __name__ == "__main__":
    main()
