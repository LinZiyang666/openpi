"""Choose the two gate thresholds on discovery inits 0-19 only (pre-stated rules; output ``out/thresholds.json``).

L0 (gate P): the largest lag L such that, for every lag value l in [-3, L], the next-look recovery after a REAL
    no-progress call exceeds the pure cache's own recovery after a stall firing at the same lag by less than 5
    percentage points (first firing of a stall; pooled over both models and all fit replicates: 5 judge runs and 5
    cache runs per model).  Rationale: below L0 a call does not end a stall more often than doing nothing.
C (gate C): the smallest budget on the grid {10, 12, 15, 20, 25} whose prefix-exact simulated success loss
    (``gatesim``) is at most 1 percentage point for BOTH models.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import dump
from exp.offline_search.rounds.r09.explore_opus.round5.tools import OUT
from exp.offline_search.rounds.r09.explore_opus.round5.tools.waste import clean

MARGIN = 0.05
LOSS = 0.01


def recovery_table():
    rows = []
    for model in ("pi05", "groot"):
        L = pd.read_parquet(OUT / f"looks_{model}_fit.parquet")
        Lc = pd.read_parquet(OUT / f"looks_cache_{model}_fit.parquet")
        assert L["init"].max() < 20 and Lc["init"].max() < 20
        a = L[L.np_call & (L.k_run == 1) & L.resolved_next.notna()].assign(kind="call", model=model)
        b = Lc[Lc.fire & (Lc.k_run == 1) & Lc.resolved_next.notna()].assign(kind="cache", model=model)
        rows.append(pd.concat([a, b])[["model", "kind", "lag", "resolved_next"]])
    X = pd.concat(rows)
    X["resolved_next"] = X.resolved_next.astype(float)
    T = X[(X.lag >= -3) & (X.lag <= 8)].groupby(["lag", "kind"]).resolved_next.agg(["mean", "size"]).unstack("kind")
    T.columns = [f"{a}_{b}" for a, b in T.columns]
    T["call_minus_cache"] = T["mean_call"] - T["mean_cache"]
    per_model = X[(X.lag >= -3) & (X.lag <= 8)].groupby(["model", "lag", "kind"]).resolved_next.mean().unstack("kind")
    per_model["diff"] = per_model["call"] - per_model["cache"]
    return T, per_model


def main():
    T, per_model = recovery_table()
    L0 = None
    for lag in range(-3, 9):
        if T.loc[lag, "call_minus_cache"] >= MARGIN:
            break
        L0 = lag
    gs = json.loads((OUT / "gatesim.json").read_text())
    C = None
    for c in (10, 12, 15, 20, 25):
        if all(-gs[m]["cap"][str(c)]["delta"]["point"] <= LOSS for m in ("pi05", "groot")):
            C = c
            break
    out = dict(L0=L0, C=C, margin=MARGIN, loss=LOSS, recovery_pooled=T.reset_index().to_dict("records"),
               recovery_per_model=per_model.reset_index().to_dict("records"),
               pace_sim={m: {k: v["delta"] for k, v in gs[m]["pace"].items()} for m in gs},
               cap_sim={m: {k: v["delta"] for k, v in gs[m]["cap"].items()} for m in gs})
    dump(OUT / "thresholds.json", clean(out))
    pd.set_option("display.width", 200)
    print(T.round(3).to_string())
    print(per_model.round(3).to_string())
    print("L0 =", L0, " C =", C)


if __name__ == "__main__":
    main()
