"""Predicted screen effect of the exhaustion takeover (basis of PREDICTION.md; inits 20-29 descriptive data only).

Replays the fitted trigger (end_rows k, dwell d from ``exhaust.json``) on the three screen runs of each leading stack
(fable r3c, its "+ empty grasp" sibling, the round-5 same-batch control; 300 episodes per model) and counts, inside
each would-be takeover window, the fresh decisions the stack did NOT already hand to the policy: those become calls
(IR) -- the only thing the takeover changes in an exhausted episode.  For the pi0.5 "escalation replaced" arm,
escalation calls outside the takeover window are removed and replaced by the no-progress guard's own call rate after
the demo end (measured on the only-no-progress runs).  SR: no fitted model of policy recovery from an exhausted state
exists (no fit-inits run has a takeover there); the evidence is the recovery of comparable controllers (see
DATA_ANALYSIS.md) and is summarized as an interval.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import PRICE, dump
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog
from exp.offline_search.rounds.r09.explore_opus.round5.tools.ledger5 import load
from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import clean
from exp.offline_search.rounds.r09.explore_opus.round6.tools import OUT
from exp.offline_search.rounds.r09.explore_opus.round6.tools.exhaust import SCREEN

WINDOW = 24


def replay(model, arms, k, d, replace_escalation=False, np_rate_after_end=None):
    cat = catalog((model, "l10", 50))
    rte = (cat.ep_len - 1 - cat.step).values
    v_price, m_price = PRICE[model]
    rows = []
    for root, arm in arms:
        D, E = load(root, arm)
        E = E.set_index(["task", "init"])
        for (t, i), e in D.groupby(["task", "init"]):
            e = e.sort_values("step")
            v = e[e.vision & (e.top1 >= 0)]
            s, r = v.step.values, rte[v.top1.values]
            at = np.flatnonzero(r <= k)
            fire = np.nan
            if len(at):
                cand = np.flatnonzero((s >= s[at[0]] + d) & (r <= k))
                fire = s[cand[0]] if len(cand) else np.nan
            cost = float(e.cost.sum())
            extra = 0.0
            if fire == fire:
                win = e[(e.step >= fire) & (e.step < fire + WINDOW) & e.vision]
                extra = float((~win.call).sum())          # fresh looks the stack served from the cache
            removed = 0.0
            if replace_escalation:
                esc = e[e.call & (e.reason == 91.0)]
                if fire == fire:
                    esc = esc[(esc.step < fire) | (esc.step >= fire + WINDOW)]
                removed = float(len(esc)) * (1.0 - np_rate_after_end)
            rows.append(dict(arm=arm, task=t, init=i, success=bool(E.loc[(t, i), "success"]), fired=fire == fire,
                             decisions=len(e), cost=cost, cost_new=cost + m_price * (extra - removed), extra_calls=extra,
                             removed_calls=removed))
    return pd.DataFrame(rows)


def np_rate_after_end(model, k):
    """Share of fresh looks after the demo end that the only-no-progress guard turns into calls (screen runs)."""
    cat = catalog((model, "l10", 50))
    rte = (cat.ep_len - 1 - cat.step).values
    n = c = 0
    for root, arm in SCREEN[model]["only_no_progress"]:
        D, _ = load(root, arm)
        D = D[D.init >= 20]
        for _, e in D.groupby(["task", "init"]):
            v = e[e.vision & (e.top1 >= 0)].sort_values("step")
            at = np.flatnonzero(rte[v.top1.values] <= k)
            if len(at):
                after = v[v.step > v.step.values[at[0]]]
                n += len(after)
                c += int(after.call.sum())
    return c / max(n, 1)


def main():
    fit = json.loads((OUT / "exhaust.json").read_text())
    k, d = int(fit["end_rows"]), int(fit["dwell"])
    rep = dict(end_rows=k, dwell=d, window=WINDOW)
    for model in ("pi05", "groot"):
        arms = SCREEN[model]["stack"]
        X = replay(model, arms, k, d)
        out = dict(episodes=int(len(X)), sr=float(X.success.mean()), ir=float(X.cost.sum() / X.decisions.sum()),
                   ir_takeover=float(X.cost_new.sum() / X.decisions.sum()), fired=int(X.fired.sum()),
                   fired_in_successes=int((X.fired & X.success).sum()), fired_in_failures=int((X.fired & ~X.success).sum()),
                   failures=int((~X.success).sum()), extra_calls_per_fired=float(X[X.fired].extra_calls.mean()) if X.fired.any() else 0.0,
                   per_run={a: dict(sr=float(g.success.mean()), ir=float(g.cost.sum() / g.decisions.sum()),
                                    ir_takeover=float(g.cost_new.sum() / g.decisions.sum()), fired=int(g.fired.sum()))
                            for a, g in X.groupby("arm")})
        rep[f"{model}_stack_plus_takeover"] = out
        print(model, "stack + takeover", json.dumps(clean({kk: vv for kk, vv in out.items() if kk != "per_run"})))
        if model == "pi05":
            rate = np_rate_after_end(model, k)
            Y = replay(model, arms, k, d, replace_escalation=True, np_rate_after_end=rate)
            o2 = dict(np_guard_call_rate_after_end=rate, ir=float(Y.cost.sum() / Y.decisions.sum()),
                      ir_replace=float(Y.cost_new.sum() / Y.decisions.sum()),
                      escalation_calls_outside_window_per_ep=float(Y.removed_calls.sum() / (1 - rate) / len(Y)) if rate < 1 else None)
            rep["pi05_escalation_replaced"] = o2
            print(model, "escalation replaced", json.dumps(clean(o2)))
    dump(OUT / "predict6.json", clean(rep))


if __name__ == "__main__":
    main()
