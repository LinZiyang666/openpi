"""R11 opus: disclosed accuracy check of the library base-IR prediction against R10's published aggregates.

Nothing here feeds any knob setting. R10 3-layer (GC_dist) owner IRs are copied from rounds/r10/REPORT.md
(test set A aggregates, 500 episodes per arm); the library side comes from out/curves.json.
Implied closed-loop guard rate: g_cl = (IR / v_lib - c_v) / c_m.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COSTS = {"pi05": (0.152, 0.848), "groot": (0.148, 0.852)}
R10_GC_DIST_IR = {  # rounds/r10/REPORT.md appendix tables, column GC_dist
    "pi05_l10_50": .168, "pi05_l10_200": .150, "pi05_l10_500": .150, "pi05_spatial_50": .135,
    "pi05_spatial_200": .101, "pi05_spatial_500": .095,
    "groot_l10_50": .207, "groot_l10_200": .191, "groot_l10_500": .175, "groot_spatial_50": .127,
    "groot_spatial_200": .114, "groot_spatial_500": .108}


def main():
    res = json.loads((HERE / "out" / "curves.json").read_text())
    rows = []
    print("| cell-size | library v | library g | library base IR | R10 3-layer IR (test A aggregate) | implied closed-loop g | library g - implied |")
    print("|---|---|---|---|---|---|---|")
    for r in res:
        c = r["cell"]
        if c not in R10_GC_DIST_IR:
            continue
        c_v, c_m = COSTS[r["model"]]
        b = r["base"]
        g_cl = (R10_GC_DIST_IR[c] / b["v"] - c_v) / c_m
        rows.append(dict(cell=c, v=b["v"], g_lib=b["g"], ir_lib=b["IR"], ir_r10=R10_GC_DIST_IR[c], g_cl=g_cl))
        print(f"| {c} | {b['v']:.4f} | {b['g']:.3f} | {b['IR']:.3f} | {R10_GC_DIST_IR[c]:.3f} | {g_cl:.3f} | {b['g'] - g_cl:+.3f} |")
    (HERE / "out" / "r10_check.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
