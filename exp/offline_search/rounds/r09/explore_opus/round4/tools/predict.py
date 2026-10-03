"""Pre-registered predictions for round 4: escalation window 24 (lag 12, deadline 80) on the half corrector.

Inputs (inits 20-29 only, read-only): fable's round-2 standalone corrector arms r9f2_{pi05,groot}_l10_50_corr05pt and
corr0 / cache controls (timan107), per-decision top-1 rows from their standard server logs (uid filter drops init >= 30).
The exact-prefix simulator (round2/tools/triggers.simulate, window=24) keeps each corrector episode up to the frozen
trigger and replaces the rest by a 24-decision policy window with success probability r.  r is bracketed by what the
window achieved on triggered episodes in my round-2 screen (both-crossed cell: pi0.5 .46, GR00T .40) +/- .10.
"""
import json
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, dump
from exp.offline_search.rounds.r09.explore_opus.round2.tools.serverledger import extract, episodes
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog, signals, first_trigger, simulate
from exp.offline_search.rounds.r09.explore_opus.round3.tools.dissect import journal

HERE = Path(__file__).resolve().parents[1]
ROOT = "r09_fable_r2"
R_MID = {"pi05": 0.46, "groot": 0.40}


def main():
    out = {}
    for model in ("pi05", "groot"):
        arm = f"r9f2_{model}_l10_50_corr05pt"
        J = journal(RUNS / ROOT, arm)
        path = extract(ROOT, arm, {k: v[1] for k, v in J.items()})
        eps = episodes(path, {k: v[0] for k, v in J.items()}, model)
        assert all(20 <= e.init < 30 for e in eps)
        cell = (model, "l10", 50)
        cat = catalog(cell)
        trig = [first_trigger(e, signals(e, cat), "lag", 12, 0, 80) for e in eps]
        base = simulate(eps, [None] * len(eps), 0, model, "l10")
        t = [e for e, k in zip(eps, trig) if k is not None]
        res = dict(arm=arm, n=len(eps), corrector_sr=base["sr"], corrector_ir=base["ir"], trigger_rate=len(t) / len(eps),
                   self_recovery_after_trigger=float(np.mean([e.success for e in t])) if t else None)
        for r in (R_MID[model] - .10, R_MID[model], R_MID[model] + .10):
            s = simulate(eps, trig, r, model, "l10", window=24)
            res[f"stack_r{r:.2f}"] = dict(sr=s["sr"], ir=s["ir"])
        out[model] = res
        print(model, json.dumps(res))
    dump(HERE / "out" / "predictions.json", out)


if __name__ == "__main__":
    main()
