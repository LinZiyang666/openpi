"""Predicted screen outcome (inits 20-29) of the gated stacks; basis of PREDICTION.md.

IR: start from the stack's own screen ledger (fable r3c, inits 20-29) and remove the calls the gates would drop.
  Gate P drops a no-progress call with pace lag <= L0; the stall then resolves at the next look with the pure cache's
  own probability at that lag (fit inits 0-19, ``waste.resolution``); otherwise the call comes back one look later,
  so the expected saving per dropped call is res_cache(lag) calls.  Gate C then caps the remaining calls at C per
  episode.  A dropped call saves the policy price (.848 pi0.5 / .852 GR00T); the look itself stays.
SR: (a) the fit-inits prefix simulation deltas (``gatesim.json``), added to the stack's screen SR;
    (b) cross-check on inits 20-29 with the same construction, using the two corrector-only runs as the gated
        prefix and the stack's own episodes for the suffix (descriptive; nothing is fitted on 20-29).
RULE 1: ledgers via ``ledger5.load`` (init < 30 asserted); fit quantities from inits 0-19 only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import PRICE, dump
from exp.offline_search.rounds.r09.explore_opus.round5.tools import OUT
from exp.offline_search.rounds.r09.explore_opus.round5.tools.gatesim import pace_sr, cap_sr
from exp.offline_search.rounds.r09.explore_opus.round5.tools.ledger5 import load
from exp.offline_search.rounds.r09.explore_opus.round5.tools.prepare_arms import BUDGET_C, PACE_L0
from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import annotate, clean, looks, pair_p

STACK = {"pi05": ("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_esc"), "groot": ("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05")}
CORR = {"pi05": [("r09_fable_r2", "r9f2_pi05_l10_50_corr05pt"), ("r09_opus_r4", "r9o4_pi05_l10_50_corr")],
        "groot": [("r09_fable_r2", "r9f2_groot_l10_50_corr05pt"), ("r09_opus_r4", "r9o4_groot_l10_50_corr")]}


def res_cache_by_lag(model):
    """P(progress at the next look | pure-cache stall firing at this lag), fit inits 0-19, pooled replicates."""
    Lc = pd.read_parquet(OUT / f"looks_cache_{model}_fit.parquet")
    assert Lc["init"].max() < 20
    x = Lc[Lc.fire & Lc.resolved_next.notna()]
    r = x.groupby(x.lag.clip(-6, 10)).resolved_next.apply(lambda s: s.astype(float).mean())
    return r


def ir_prediction(model, L0, C):
    D, E = load(*STACK[model])
    v, m = PRICE[model]
    P = pair_p(model)
    L = annotate(looks(D, model), E, P)
    rc = res_cache_by_lag(model)
    calls = L[L.call].copy()
    calls["drop_p"] = 0.0
    if L0 is not None:
        sel = calls.np_call & (calls.lag <= L0)
        calls.loc[sel, "drop_p"] = calls.loc[sel, "lag"].clip(-6, 10).map(rc).fillna(rc.mean()).values
    base_cost, dec = float(E.cost.sum()), float(E.decisions.sum())
    saved = float(calls.drop_p.sum())
    out = dict(ir_stack=base_cost / dec, calls_per_ep=float(len(calls) / len(E)), np_calls_per_ep=float(calls.np_call.sum() / len(E)),
               pace_dropped_calls_per_ep=saved / len(E))
    out["ir_pace"] = (base_cost - m * saved) / dec
    if C is not None:
        kept = (1 - calls.drop_p).groupby([calls.task, calls.init]).sum()
        over = (kept - C).clip(lower=0)
        out["budget_dropped_calls_per_ep"] = float(over.sum() / len(E))
        out["episodes_over_budget"] = int((over > 0).sum())
        succ = E.set_index(["task", "init"]).success
        out["successes_over_budget"] = int(succ.reindex(over[over > 0].index).sum())
        out["ir_pace_budget"] = (base_cost - m * (saved + over.sum())) / dec
    out["sr_stack"] = float(E.success.mean())
    return out


def sr_crosscheck(model, L0, C):
    """Inits 20-29: corrector-only runs as the gated prefix, the stack's episodes as the judge."""
    P = pair_p(model)
    D, E = load(*STACK[model])
    Lj = annotate(looks(D, model), E, P).assign(arm="stack")
    parts = []
    for root, arm in CORR[model]:
        Dc, Ec = load(root, arm)
        parts.append(annotate(looks(Dc, model), Ec, P).assign(arm=arm))
    Lc = pd.concat(parts)
    base = pace_sr(Lj, Lc, None)[0]
    pace = pace_sr(Lj, Lc, L0)[0]
    cap = cap_sr(Lj, Lc, C)[0] if C is not None else None
    real = float(E.success.mean())
    return dict(sim_stack=base, sim_pace=pace, delta_pace=pace - base, sim_cap=cap,
                delta_cap=(cap - real) if cap is not None else None, stack_real=real,
                corrector_real=float(Lc.groupby(["arm", "task", "init"]).success.first().mean()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    gs = json.loads((OUT / "gatesim.json").read_text())
    rep = {}
    for model in ("pi05", "groot"):
        ir = ir_prediction(model, PACE_L0, BUDGET_C)
        xc = sr_crosscheck(model, PACE_L0, BUDGET_C)
        fit_p = gs[model]["pace"][str(PACE_L0)]["delta"]
        fit_c = gs[model]["cap"][str(BUDGET_C)]["delta"]
        sr0 = ir["sr_stack"]
        pred = dict(stack=dict(sr=sr0, ir=ir["ir_stack"]),
                    pace=dict(sr=sr0 + fit_p["point"], sr_lo90=sr0 + fit_p["lo90"], sr_hi90=sr0 + fit_p["hi90"], ir=ir["ir_pace"],
                              ir_cut=1 - ir["ir_pace"] / ir["ir_stack"]),
                    pace_budget=dict(sr=sr0 + fit_p["point"] + fit_c["point"], sr_lo90=sr0 + fit_p["lo90"] + fit_c["lo90"],
                                     sr_hi90=sr0 + fit_p["hi90"] + fit_c["hi90"], ir=ir["ir_pace_budget"],
                                     ir_cut=1 - ir["ir_pace_budget"] / ir["ir_stack"]))
        rep[model] = dict(L0=PACE_L0, C=BUDGET_C, ir_detail=ir, crosscheck_20_29=xc, fit_delta_pace=fit_p, fit_delta_budget=fit_c,
                          prediction=pred)
        print(model, json.dumps(clean(pred), indent=1))
        print(" crosscheck", json.dumps(clean(xc)))
        print(" ir detail", json.dumps(clean(ir)))
    dump(OUT / "predict.json", clean(rep))


if __name__ == "__main__":
    main()
