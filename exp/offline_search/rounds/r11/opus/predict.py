"""R11 opus: written predictions for the proposed arms (before any R11 closed-loop result).

IR: library-model prediction and the "informed" prediction from out/arm_grid.json (see arm_grid.py).
SR: a prior, not a fit. Base SR / IR of the 3-layer design and the pure-policy SR are R10's published test-set-A
aggregates (rounds/r10/REPORT.md and the R11 BRIEF table). The gap-closing curve phi(dIR) = dIR^2/(dIR^2 + .12^2)
is a hand-set shape read off R6's published "B + random dose" aggregates (R6 ANALYSIS / frontier, disclosed):
~20-35% of the base-to-pure gap closed at +.08 IR, ~65-70% at +.16, ~80-90% at +.24.
Schedule bonuses are hand-set priors (R6 periodic hints, see PREDICTION.md); they are not estimated here.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BASE = {  # R10 GC_dist (3-layer) SR @ IR, test set A aggregates
    "pi05_l10_50": (.826, .168), "pi05_l10_200": (.904, .150), "pi05_l10_500": (.894, .150),
    "pi05_spatial_50": (.910, .135), "groot_l10_50": (.754, .207), "groot_l10_200": (.820, .191),
    "groot_l10_500": (.906, .175), "groot_spatial_50": (.896, .127)}
PURE = {"pi05_l10": .908, "groot_l10": .898, "pi05_spatial": .988, "groot_spatial": .940}
SCALE = .12


def phi(d):
    d = max(d, 0.0)
    return d * d / (d * d + SCALE * SCALE)


def bonus(cell, method, target):
    weak_l10 = cell in ("pi05_l10_50", "groot_l10_50", "groot_l10_200")
    sp50 = cell in ("pi05_spatial_50", "groot_spatial_50")
    per = 0.0
    if method.startswith("periodic"):
        per = (1.0 if weak_l10 else .5 if sp50 else 0.0) if target <= .32 else 0.0
        if method == "periodic_pgt1":
            per += .3
    if method == "random_tail2":
        per = -.5
    return per


def main():
    rows = json.loads((OUT / "arm_grid.json").read_text())
    out = []
    for r in rows:
        cell = r["cell"]
        sr0, ir0 = BASE[cell]
        pure = PURE["_".join(cell.split("_")[:2])]
        gap = pure - sr0
        d = r["pred_IR_informed"] - ir0
        dsr = 100 * max(gap, 0) * phi(d) + bonus(cell, r["method"], r["target"])
        if gap <= .005:          # base already at pure: extra calls are a coin flip with a slight harm prior
            dsr = -0.5 + (0.0 if not r["method"].startswith("periodic") else 0.0)
        sr = min(sr0 + dsr / 100, pure + .01)
        out.append(dict(r, base_SR=sr0, base_IR=ir0, pure_SR=pure, pred_dSR_pp=round(100 * (sr - sr0), 1),
                        pred_SR=round(sr, 3)))
    (OUT / "predictions.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["| # | cell-size | target | method | setting | pred IR library | pred IR informed | 3-layer base SR @ IR | pred SR | pred change vs base (pp) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(out, 1):
        L.append(f"| {i} | {r['cell']} | {r['target']:.2f} | {r['method']} | {r['knob']} {r['setting']:.4f} | "
                 f"{r['pred_IR_lib']:.3f} | {r['pred_IR_informed']:.3f} | {r['base_SR']:.3f} @ {r['base_IR']:.3f} | "
                 f"{r['pred_SR']:.3f} | {r['pred_dSR_pp']:+.1f} |")
    (OUT / "predictions.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
