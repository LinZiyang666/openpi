"""R11 opus: check the IR model's overlap rule against R6's arm-level ledger aggregates (disclosed; no fitting).

Reads only rounds/r06/frontier_final/frontier_points.csv (arm-level N/V/M/SR aggregates, the table behind R6
frontier.md); no episode-level or os_closed_loop file. For each R6 "B + random anchor dose" arm (R6 ExtraDose:
B's guard verdict first, then an independent keyed coin with probability `dose` on the anchors where no guard
fired), the guard-first rule predicts the miss rate from the same cell's B replicates alone:
    f_pred = g_B + (1 - g_B) * dose,   m_pred = v * f_pred,   IR_pred = c_v v + c_m m_pred,
with g_B = M/V of the pooled B replicates. Nothing is fitted; the residual measures how much extra policy calls
change the closed-loop guard rate (the one closed-loop quantity the R11 offline model cannot see).

  .venv/bin/python -m exp.offline_search.rounds.r11.opus.r6_overlap_check
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "r06" / "frontier_final" / "frontier_points.csv"
COSTS = {"pi05": (0.152, 0.848), "groot": (0.148, 0.852)}


def main():
    rows = [r for r in csv.DictReader(SRC.open()) if r["eligible"] == "True"]
    B = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r["method_family"] == "B_guard_committed_rescue" and "cap" not in r["arm"]:
            for i, k in enumerate("NVM"):
                B[r["cell"]][i] += int(r[k])
    out = []
    for r in rows:
        if r["method_family"] != "B_plus_random_anchor_dose":
            continue
        cell, model = r["cell"], r["model"]
        dose = float(r["arm"].split("dose")[1].replace("p", "."))
        N, V, M = B[cell]
        g = M / V
        N1, V1, M1 = int(r["N"]), int(r["V"]), int(r["M"])
        v1 = V1 / N1
        f_pred = g + (1 - g) * dose
        c_v, c_m = COSTS[model]
        out.append(dict(cell=cell, dose=dose, g_B=g, f_obs=M1 / V1, f_pred=f_pred,
                        IR_obs=float(r["owner_IR"]), IR_pred=c_v * v1 + c_m * v1 * f_pred,
                        implied_g_under_dose=(M1 / V1 - dose) / (1 - dose)))
    print("| cell | dose | B guard rate | miss frac obs | pred | IR obs | IR pred | IR pred - obs | guard rate implied under dose |")
    print("|---|---|---|---|---|---|---|---|---|")
    for o in sorted(out, key=lambda o: (o["cell"], o["dose"])):
        print(f"| {o['cell']} | {o['dose']:g} | {o['g_B']:.3f} | {o['f_obs']:.3f} | {o['f_pred']:.3f} | "
              f"{o['IR_obs']:.3f} | {o['IR_pred']:.3f} | {o['IR_pred'] - o['IR_obs']:+.4f} | {o['implied_g_under_dose']:.3f} |")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "r6_overlap_check.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
