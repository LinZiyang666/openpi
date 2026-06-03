"""Combined Pareto (FULL version, v1) — per-base detail + verdict overlay.

Same data sources as plot_threshold_pareto_combined.py but keeps the per-base
detail visible (scatter + 4 frontiers) instead of collapsing into one envelope.
Use this when you want to see how each of d1/d3/d4/d5 compares individually
against r/p baseline + kinematic verdict.

  - r/p baseline:         light-gray scatter + gray dashed frontier
  - kinematic verdict:    light-purple scatter + purple solid frontier
  - threshold per base:   per-base light scatter + colored frontier (4 lines)
  - ★ inf~0 always_hit anchors (74/72/72/72)

Output: pareto_combined_full.{png,pdf}
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
_BASE_LIGHT = {1: "#a6c8e0", 3: "#ffc580", 4: "#a6dcb1", 5: "#f0a3a3"}
WARM_COST = 0.75


def _compute_inf(row):
    n = row.get("n_eval_verdicts") or 0
    if not n: return 0.0
    return (row.get("n_full_hit", 0) * 0.0 + row.get("n_warm_start", 0) * WARM_COST + row.get("n_miss", 0) * 1.0) / n


def _load_threshold_maps(repo: Path):
    root = repo / "exp/weighted_sum/data/libero_spatial/threshold_pareto"
    combined_sr = root / "eval_results.json"
    combined_inf = root / "eval_inf_ratio.json"
    if combined_sr.exists() and combined_inf.exists():
        return json.loads(combined_sr.read_text()), json.loads(combined_inf.read_text())

    sr = {}
    inf = {}
    for depth in (1, 3, 4, 5):
        droot = root / f"d{depth}"
        sr.update(json.loads((droot / "results.json").read_text()))
        inf.update(json.loads((droot / "inf_ratio.json").read_text()))
    return sr, inf


def main():
    repo = Path(__file__).resolve().parents[3]
    # threshold-pareto by base
    sr, inf = _load_threshold_maps(repo)
    by_base = defaultdict(list)
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

    # verdict r/p baseline
    rp_pts = [p for p in _load_random_periodic(repo / "exp/random_periodic_gate/analysis/aggregate.csv") if p[0] > 0]
    # verdict phase5 kinematic
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

    fig, ax = plt.subplots(figsize=(14, 9))
    # 1) baseline (gray scatter + dashed frontier)
    if rp_pts:
        ax.scatter([p[0] for p in rp_pts], [p[1] for p in rp_pts], s=18, c="#c0c0c0", alpha=0.5,
                   label=f"r/p baseline ({len(rp_pts)} pts)", zorder=1)
        rp_front = pareto_upper_frontier(list(rp_pts))
        if rp_front:
            fx, fy = zip(*rp_front)
            ax.plot(fx, fy, "--", color="#666666", lw=2.6, alpha=0.95,
                    label=f"r/p baseline frontier ({len(rp_front)} pts)", zorder=2)
    # 2) kinematic (purple scatter + solid frontier)
    if p5_pts:
        ax.scatter([p[0] for p in p5_pts], [p[1] for p in p5_pts], s=18, c="#cbb3e8", alpha=0.5,
                   label=f"kinematic verdict ({len(p5_pts)} pts)", zorder=2)
        p5_front = pareto_upper_frontier(p5_pts)
        if p5_front:
            fx, fy = zip(*p5_front)
            ax.plot(fx, fy, "-", color="#7e3eb5", lw=2.8, alpha=0.95,
                    label=f"kinematic frontier ({len(p5_front)} pts)", zorder=4)
    # 3) per-base scatter + frontier (4 lines)
    for d in (1, 3, 4, 5):
        pts = by_base.get(d, [])
        if not pts:
            continue
        c = _BASE_COLOR[d]; lc = _BASE_LIGHT[d]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=lc, alpha=0.45, zorder=3)
        front = pareto_upper_frontier(pts)
        if front:
            fx, fy = zip(*front)
            ax.plot(fx, fy, "-o", color=c, lw=2.2, ms=5, alpha=0.95,
                    label=f"threshold d{d} ({len(front)} front pts, mean SR {sum(p[1] for p in pts)/len(pts)*100:.1f}%)",
                    zorder=5)
        ax.scatter([0], [_ALWAYS_HIT_SR[d]], marker="*", s=180, color=c, edgecolor="black", linewidth=0.9, zorder=6)

    ax.set_xlabel("inference_ratio (lower = more cache reuse)", fontsize=12)
    ax.set_ylabel("success_rate", fontsize=12)
    ax.set_title("Combined Pareto (FULL): per-base d1/d3/d4/d5 vs verdict (kinematic, r/p baseline)\n"
                 "★ = wsweep always_hit anchor for each threshold base", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.55, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.4, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.95)
    fig.tight_layout()
    out = repo / "exp/weighted_sum/analysis/libero_spatial/threshold_pareto"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pareto_combined_full.{ext}", dpi=180, bbox_inches="tight")
    print(f"saved combined Pareto (FULL) to {out}/pareto_combined_full.{{png,pdf}}")


if __name__ == "__main__":
    main()
