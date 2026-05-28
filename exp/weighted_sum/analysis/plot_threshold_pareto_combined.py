"""Combined Pareto — 3 frontier lines only (no scatter clutter).

  - r/p baseline frontier (gray dashed)
  - kinematic verdict frontier (purple solid)
  - threshold-pareto envelope across all 4 bases (teal solid, the outer best-of-d1/d3/d4/d5)
  - small ★ at inf~0 for each base's wsweep always_hit anchor (74/72/72/72)

The 6-line per-base version is in plot_threshold_pareto_per_base.py (kept for detail).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from exp.verdict_factor_judge.analysis.phase3.plot_pareto import (  # noqa: E402
    _load_random_periodic, pareto_upper_frontier,
)

_TAG_RE = re.compile(r"__d(\d+)__")
_ALWAYS_HIT_SR = {1: 0.74, 3: 0.72, 4: 0.72, 5: 0.72}
_BASE_COLOR = {1: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c", 5: "#d62728"}
WARM_COST = 0.75


def _compute_inf(row):
    n = row.get("n_eval_verdicts") or 0
    if not n: return 0.0
    return (row.get("n_full_hit", 0) * 0.0 + row.get("n_warm_start", 0) * WARM_COST + row.get("n_miss", 0) * 1.0) / n


def main():
    repo = Path(__file__).resolve().parents[3]
    # threshold-pareto: all 4 bases combined into one envelope
    sr = json.loads((repo / "exp/weighted_sum/data/threshold_pareto/eval_results.json").read_text())
    inf = json.loads((repo / "exp/weighted_sum/data/threshold_pareto/eval_inf_ratio.json").read_text())
    by_base = defaultdict(list)
    all_threshold = []
    for yid in sr:
        m = _TAG_RE.search(yid)
        if not m or yid not in inf:
            continue
        d = int(m.group(1))
        s = sr[yid]["success_rate"] if isinstance(sr[yid], dict) else sr[yid]
        ir = inf[yid].get("inference_ratio")
        if ir is None:
            continue
        by_base[d].append((ir, float(s)))
        all_threshold.append((ir, float(s)))

    # verdict r/p baseline
    rp_pts = [p for p in _load_random_periodic(repo / "exp/random_periodic_gate/analysis/aggregate.csv") if p[0] > 0]
    # verdict phase5 = kinematic
    p5_path = repo / "exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl"
    p5_pts = []
    for line in p5_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "error" in d or not d.get("n_eval_verdicts"):
            continue
        ir = _compute_inf(d)
        s = d.get("success_rate", 0.0)
        if ir > 0:
            p5_pts.append((ir, s))

    rp_front = pareto_upper_frontier(list(rp_pts)) if rp_pts else []
    p5_front = pareto_upper_frontier(p5_pts) if p5_pts else []
    th_front = pareto_upper_frontier(all_threshold) if all_threshold else []

    fig, ax = plt.subplots(figsize=(12, 8))
    # 1) baseline
    if rp_front:
        fx, fy = zip(*rp_front)
        ax.plot(fx, fy, "--", color="#888888", lw=2.0, alpha=0.85,
                label=f"r/p baseline ({len(rp_front)} pts)", zorder=2)
    # 2) kinematic
    if p5_front:
        fx, fy = zip(*p5_front)
        ax.plot(fx, fy, "-", color="#7e3eb5", lw=2.5, alpha=0.95,
                label=f"kinematic verdict ({len(p5_front)} pts)", zorder=3)
    # 3) threshold envelope (4 bases combined)
    if th_front:
        fx, fy = zip(*th_front)
        ax.plot(fx, fy, "-o", color="#0f766e", lw=2.8, ms=5, alpha=0.95,
                label=f"threshold-pareto envelope ({len(th_front)} pts, all 4 base)", zorder=4)
    # 4) ★ always_hit anchors per base
    for d in (1, 3, 4, 5):
        ax.scatter([0], [_ALWAYS_HIT_SR[d]], marker="*", s=130, color=_BASE_COLOR[d],
                   edgecolor="black", linewidth=0.9, zorder=5,
                   label=f"d{d} always_hit anchor ({_ALWAYS_HIT_SR[d]:.2f})")

    ax.set_xlabel("inference_ratio (lower = more cache reuse)", fontsize=12)
    ax.set_ylabel("success_rate", fontsize=12)
    ax.set_title("Pareto comparison: threshold-pareto vs verdict (kinematic) vs r/p baseline\n"
                 "★ = wsweep always_hit anchor (inf≈0); per-base detail in per_base.png",
                 fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.55, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.4, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=9, ncol=1, framealpha=0.95)
    fig.tight_layout()
    out = Path(__file__).resolve().parent
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pareto_combined.{ext}", dpi=180, bbox_inches="tight")
    print(f"saved combined Pareto (3-line) to {out}/pareto_combined.{{png,pdf}}")
    print(f"frontier sizes: r/p={len(rp_front)}, kinematic={len(p5_front)}, threshold={len(th_front)}")


if __name__ == "__main__":
    main()
