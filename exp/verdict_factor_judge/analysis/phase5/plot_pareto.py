"""Phase 5 Pareto overlay — 240 cells × 5 group, colored per group.

Reuses (inference_ratio, success_rate) plane from `plot_pareto_phase3`.
Layered:
  - random/periodic baseline cloud
  - always-WARM markers
  - phase3 100ep cells (faded gray)
  - phase4 stage5 500ep cells (small black)
  - phase5 240 cells colored per group (G1-G5)
  - gold-circle Pareto-positive vs r/p + always-WARM

Inputs (defaults, repo-relative):
  exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl
  exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl
  exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl   (stage5 500ep)
  exp/random_periodic_gate/analysis/aggregate.csv

Output: exp/verdict_factor_judge/analysis/phase5/pareto.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from exp.verdict_factor_judge.analysis.phase3.plot_pareto import (  # noqa: E402
    _load_phase3,
    _load_random_periodic,
    is_pareto_dominated,
    pareto_upper_frontier,
    warm_cost,
)


WARM_COST = 0.75
WARM_SR_SPATIAL16 = {0.30: 0.942, 0.50: 0.952, 0.70: 0.976}


def _compute_inf(row: dict) -> float:
    n = row.get("n_eval_verdicts") or 0
    if not n:
        return 0.0
    return (
        row.get("n_full_hit", 0) * 0.0
        + row.get("n_warm_start", 0) * WARM_COST
        + row.get("n_miss", 0) * 1.0
    ) / n


def _load_phase5(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "error" in d or not d.get("n_eval_verdicts"):
            continue
        d["inf"] = _compute_inf(d)
        d["sr"] = d.get("success_rate") or 0.0
        rows.append(d)
    return rows


def main() -> None:
    repo = Path(__file__).resolve().parents[4]
    rp_pts = _load_random_periodic(repo / "exp/random_periodic_gate/analysis/aggregate.csv")
    phase3_rows = _load_phase3(repo / "exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl")
    stage5_path = repo / "exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl"
    stage5_rows = _load_phase5(stage5_path)
    phase5_path = repo / "exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl"
    phase5_rows = _load_phase5(phase5_path)

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    fig.suptitle(
        "Phase 5 systematic sweep — 240 cell × 100ep on Pareto plane "
        f"(spatial16; warm cost {WARM_COST} @ start_t=0.5)",
        fontsize=14, fontweight="bold",
    )

    # 1) random / periodic cloud
    rx, ry = (zip(*rp_pts) if rp_pts else ([], []))
    ax.scatter(rx, ry, s=22, c="lightgray", marker="o", alpha=0.55,
               label=f"random/periodic ({len(rp_pts)} pts)", zorder=1)

    # 2) always-WARM
    warm_pts = [(warm_cost(t), WARM_SR_SPATIAL16[t]) for t in (0.30, 0.50, 0.70)]
    wx = [p[0] for p in warm_pts]
    wy = [p[1] for p in warm_pts]
    ax.scatter(wx, wy, s=320, c="red", marker="*", edgecolors="darkred",
               linewidths=1.4, label="always-WARM", zorder=4)
    ax.plot(wx, wy, "--", color="red", alpha=0.4, linewidth=1.0, zorder=3)

    # 3) phase 3 100ep cells (faded gray)
    if phase3_rows:
        ax.scatter([r["inf"] for r in phase3_rows], [r["sr"] for r in phase3_rows],
                   s=20, c="silver", alpha=0.35, marker="o",
                   label=f"phase3 100ep ({len(phase3_rows)} cells)", zorder=2)

    # 4) phase4 stage5 500ep cells (small black)
    if stage5_rows:
        ax.scatter([r["inf"] for r in stage5_rows], [r["sr"] for r in stage5_rows],
                   s=28, c="black", alpha=0.55, marker="x",
                   label=f"phase4 stage5 500ep ({len(stage5_rows)} cells)", zorder=3)

    # 5) phase 5 cells per group
    group_color = {
        "g1": ("#1f77b4", "o", "G1 single-window"),
        "g2": ("#ff7f0e", "s", "G2 multi-window"),
        "g3": ("#2ca02c", "^", "G3 weight pattern"),
        "g4": ("#d62728", "D", "G4 multi-factor subset"),
        "g5": ("#9467bd", "P", "G5 threshold grid"),
    }
    for g, (color, marker, label) in group_color.items():
        sub = [r for r in phase5_rows if r.get("group") == g]
        if not sub:
            continue
        ax.scatter([r["inf"] for r in sub], [r["sr"] for r in sub],
                   s=120, color=color, marker=marker, edgecolors="black",
                   linewidths=0.9, alpha=0.92,
                   label=f"{label} ({len(sub)})", zorder=5)

    # 6) gold-circle Pareto positives vs r/p + warm
    all_base = list(rp_pts) + warm_pts
    n_gold = 0
    for r in phase5_rows:
        if not is_pareto_dominated(r["inf"], r["sr"], all_base):
            ax.scatter([r["inf"]], [r["sr"]], s=260, facecolors="none",
                       edgecolors="gold", linewidths=2.4, zorder=6)
            n_gold += 1

    # 7) phase5 own Pareto frontier line
    p5_front = pareto_upper_frontier([(r["inf"], r["sr"]) for r in phase5_rows])
    if p5_front:
        fx, fy = zip(*p5_front)
        ax.plot(fx, fy, "-", color="black", alpha=0.6, linewidth=1.5,
                label=f"phase5 frontier ({len(p5_front)} pts)", zorder=4)

    ax.set_xlabel("inference_ratio", fontsize=12)
    ax.set_ylabel("success_rate", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.55, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.4, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95, ncol=2,
              title=f"gold = Pareto positive ({n_gold}/{len(phase5_rows)})")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out = repo / "exp/verdict_factor_judge/analysis/phase5/pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}  size: {out.stat().st_size / 1024:.1f} KB  "
          f"({n_gold}/{len(phase5_rows)} gold)")


if __name__ == "__main__":
    main()
