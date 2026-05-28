"""Plot the SR x inference_ratio Pareto for the threshold sweep.

Joins per-yaml success_rate (from summarize.py) with per-yaml inference_ratio
(from summarize_inf_ratio.py) and scatters each cell on the (inference_ratio,
success_rate) plane, drawing the upper-left Pareto frontier. Anchors the
always-hit (inf_ratio≈0) and all-MISS raw-policy (inf_ratio=1) points if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pareto_front(points):
    """Upper-left frontier: sort by inf asc, keep points with strictly higher SR."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1.0
    for inf, sr, yid in pts:
        if sr > best:
            front.append((inf, sr, yid))
            best = sr
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", required=True, help="per-yaml success_rate JSON (summarize.py)")
    ap.add_argument("--inf-ratio", required=True, help="per-yaml inference_ratio JSON (summarize_inf_ratio.py)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "threshold_pareto.png"))
    args = ap.parse_args()

    sr = _load(args.sr)
    inf = _load(args.inf_ratio)
    pts = []
    for yid, srrec in sr.items():
        if yid not in inf:
            continue
        s = float(srrec["success_rate"] if isinstance(srrec, dict) else srrec) * 100
        ir = float(inf[yid]["inference_ratio"])
        pts.append((ir, s, yid))
    if not pts:
        print("no overlapping (SR, inf_ratio) yamls to plot")
        return

    front = _pareto_front(pts)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=30, alpha=0.6, label="cells")
    ax.plot([p[0] for p in front], [p[1] for p in front], "r-o", lw=2, ms=5, label="Pareto frontier")
    for ir, s, yid in pts:
        tag = "anchor" if "anchor" in yid else ("warm0" if "__warmup" in yid else "")
        if tag:
            ax.annotate(tag, (ir, s), fontsize=7)
    ax.set_xlabel("inference_ratio  (lower = more cache reuse / less compute)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Weighted-sum threshold sweep: SR vs inference_ratio Pareto")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out.replace(".png", f".{ext}"), dpi=200)
    print(f"saved {args.out} ({len(pts)} cells, {len(front)} on frontier)")


if __name__ == "__main__":
    main()
