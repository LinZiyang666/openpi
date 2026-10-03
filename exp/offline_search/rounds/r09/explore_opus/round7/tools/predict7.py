"""Predicted effect of the call budget alone (no pace gate) on the leading stacks; basis of round7/PREDICTION.md.

IR: each stack screen run's own ledger (fable r3c run and the round-5 same-batch control, inits 20-29) with every
guard call beyond the C-th per episode removed (the look stays; policy price .848 / .852 saved).  SR: the fit-inits
(0-19) prefix-exact simulation of the budget (round5 ``gatesim.json``) and the round-5 screen contrast
P2C20 vs P2 (budget on top of the pace gate).
RULE 1: ledgers via ``ledger5.load`` (init < 30 asserted).
"""
from __future__ import annotations

import json

import numpy as np

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import PRICE, dump
from exp.offline_search.rounds.r09.explore_opus.round5.tools import OUT as OUT5
from exp.offline_search.rounds.r09.explore_opus.round5.tools.ledger5 import load
from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import clean
from exp.offline_search.rounds.r09.explore_opus.round7.tools import OUT

RUNS = {"pi05": [("r09_fable_r3c", "r9f3c_pi05_l10_50_np_corr05_esc"), ("r09_opus_r5", "r9o5_pi05_l10_50_stack")],
        "groot": [("r09_fable_r3c", "r9f3c_groot_l10_50_np_corr05"), ("r09_opus_r5", "r9o5_groot_l10_50_stack")]}


def budget_ir(root, arm, model, C):
    D, E = load(root, arm)
    m = PRICE[model][1]
    calls = D[D.call].sort_values(["task", "init", "step"])
    idx = calls.groupby(["task", "init"]).cumcount() + 1
    over = calls[idx > C]
    succ = E.set_index(["task", "init"]).success
    over_eps = over.groupby(["task", "init"]).size()
    return dict(ir=float(E.cost.sum() / E.decisions.sum()), ir_budget=float((E.cost.sum() - m * len(over)) / E.decisions.sum()),
                calls_removed_per_ep=float(len(over) / len(E)), episodes_over=int(len(over_eps)),
                successes_over=int(succ.reindex(over_eps.index).sum()), sr=float(E.success.mean()))


def main():
    gs = json.loads((OUT5 / "gatesim.json").read_text())
    rep = {}
    for model in ("pi05", "groot"):
        for C in (20, 15):
            runs = {arm: budget_ir(root, arm, model, C) for root, arm in RUNS[model]}
            cut = [1 - r["ir_budget"] / r["ir"] for r in runs.values()]
            fit = gs[model]["cap"][str(C)]["delta"]
            rep[f"{model}_C{C}"] = dict(runs=runs, ir_cut_range=[float(min(cut)), float(max(cut))], fit_sim_delta_sr=fit)
            print(model, C, json.dumps(clean(rep[f"{model}_C{C}"])))
    dump(OUT / "predict7.json", clean(rep))


if __name__ == "__main__":
    main()
