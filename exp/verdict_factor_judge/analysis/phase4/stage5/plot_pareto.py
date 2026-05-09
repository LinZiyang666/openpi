"""Phase 4 Stage 5 Pareto overlay — 48 cells x 500 ep true-value retest.

Reuses the (inference_ratio, success_rate) plane from
`plot_pareto_phase3.py`. Cells from stage 5 are colored per group:

  - Group A: phase3 g1 / g10 anchors (8)
  - Group B+C: phase4 R2 desc patterns (16)
  - Group D: phase4 R1 alpha sweep (8)
  - Group E: phase4 R4 W-FUT window (8)
  - Group F: phase3 g6 + g4/g8/g9/g11 (8)

Phase 3 100ep cloud is faded gray. Stage 5 cells get gold circle if
Pareto-positive vs random/periodic + always-WARM baselines.

Inputs:
  /tmp/phase4_stage5_unpack/exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl
  exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl
  exp/random_periodic_gate/analysis/aggregate.csv

Output: exp/verdict_factor_judge/analysis/phase4/stage5/pareto.png
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

STAGE5_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/phase5/per_yaml_summary.jsonl"
)


def _compute_inf(row: dict) -> float:
    n = row.get("n_eval_verdicts") or 0
    if not n:
        return 0.0
    return (
        row.get("n_full_hit", 0) * 0.0
        + row.get("n_warm_start", 0) * WARM_COST
        + row.get("n_miss", 0) * 1.0
    ) / n


def _load_stage5(path: Path) -> list[dict]:
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


def _classify(row: dict) -> str:
    yid = row["yaml_id"]
    rid = row["recipe_id"]
    if "phase4" in yid and "__r1_a" in yid:
        return "D"
    if "phase4" in yid and "__r2_a" in yid:
        return "BC"
    if "phase4" in yid and "__r4_a" in yid:
        return "E"
    if "phase3_g1_" in yid or "phase3_g10_" in yid:
        return "A"
    return "F"


def main() -> None:
    repo = Path(__file__).resolve().parents[5]
    rp_pts = _load_random_periodic(
        repo / "exp/random_periodic_gate/analysis/aggregate.csv"
    )
    phase3_rows = _load_phase3(
        repo / "exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl"
    )
    stage5_rows = _load_stage5(STAGE5_PATH)

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    fig.suptitle(
        "Phase 4 Stage 5 — 48-cell × 500ep true-value retest on Pareto plane "
        f"(spatial16; warm cost {WARM_COST} @ start_t=0.5)",
        fontsize=14, fontweight="bold",
    )

    # 1) random / periodic cloud + frontier
    rx, ry = (zip(*rp_pts) if rp_pts else ([], []))
    ax.scatter(rx, ry, s=22, c="lightgray", marker="o", alpha=0.55,
               label=f"random / periodic ({len(rp_pts)} pts)", zorder=1)
    rp_front = pareto_upper_frontier(rp_pts)
    if rp_front:
        fx, fy = zip(*rp_front)
        ax.plot(fx, fy, "-", color="gray", alpha=0.35, linewidth=1.1, zorder=2)

    # 2) always-WARM baselines
    warm_pts = [
        (warm_cost(t), WARM_SR_SPATIAL16[t]) for t in (0.30, 0.50, 0.70)
    ]
    wx = [p[0] for p in warm_pts]
    wy = [p[1] for p in warm_pts]
    ax.scatter(wx, wy, s=320, c="red", marker="*", edgecolors="darkred",
               linewidths=1.4, label="always-WARM (start_t)", zorder=4)
    for (x, y), t in zip(warm_pts, (0.30, 0.50, 0.70)):
        ax.annotate(f" t={t}", (x, y), xytext=(8, -4),
                    textcoords="offset points", fontsize=10, color="darkred",
                    fontweight="bold")
    ax.plot(wx, wy, "--", color="red", alpha=0.4, linewidth=1.0, zorder=3)

    # 3) phase 3 100ep cloud — faded reference
    p3_x = [r["inf"] for r in phase3_rows]
    p3_y = [r["sr"] for r in phase3_rows]
    ax.scatter(p3_x, p3_y, s=32, c="silver", marker="o", alpha=0.5,
               edgecolors="gray", linewidths=0.5,
               label=f"phase 3 100ep ({len(phase3_rows)} cells, faded)",
               zorder=2)

    # 4) Stage 5 — colored per group, larger marker
    group_color = {
        "A":  ("#1f77b4", "o", "Group A: phase3 g1/g10 anchor (8)"),
        "BC": ("#ff7f0e", "s", "Group B+C: phase4 R2 desc patterns (16)"),
        "D":  ("#2ca02c", "^", "Group D: phase4 R1 α sweep (8)"),
        "E":  ("#d62728", "D", "Group E: phase4 R4 W-FUT window (8)"),
        "F":  ("#9467bd", "P", "Group F: phase3 g6 + g4/g8/g9/g11 (8)"),
    }
    by_group: dict[str, list[dict]] = {k: [] for k in group_color}
    for r in stage5_rows:
        by_group[_classify(r)].append(r)

    for grp, (color, marker, label) in group_color.items():
        sub = by_group[grp]
        if not sub:
            continue
        xs = [r["inf"] for r in sub]
        ys = [r["sr"] for r in sub]
        ax.scatter(xs, ys, s=140, color=color, marker=marker,
                   edgecolors="black", linewidths=1.0, alpha=0.95,
                   label=label, zorder=5)

    # 5) gold-circle stage5 cells: strict Pareto positive vs r/p + always-WARM
    all_base = list(rp_pts) + warm_pts
    n_gold = 0
    for r in stage5_rows:
        if not is_pareto_dominated(r["inf"], r["sr"], all_base):
            ax.scatter([r["inf"]], [r["sr"]], s=280, facecolors="none",
                       edgecolors="gold", linewidths=2.5, zorder=6)
            n_gold += 1

    # 6) Stage 5 own Pareto upper frontier (for visual)
    s5_front = pareto_upper_frontier([(r["inf"], r["sr"]) for r in stage5_rows])
    if s5_front:
        fx, fy = zip(*s5_front)
        ax.plot(fx, fy, "-", color="black", alpha=0.55, linewidth=1.6,
                label=f"stage5 upper frontier ({len(s5_front)} pts)", zorder=4)

    ax.set_xlabel("inference_ratio  (lower = more cache reuse)", fontsize=12)
    ax.set_ylabel("success_rate", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.55, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(which="major", alpha=0.4, linewidth=0.7)
    ax.grid(which="minor", alpha=0.18, linewidth=0.4)
    ax.tick_params(axis="both", which="major", labelsize=11)

    ax.legend(
        loc="lower right", fontsize=9, framealpha=0.95, ncol=2,
        title=f"gold-circle = Pareto-positive vs r/p+warm ({n_gold}/{len(stage5_rows)})",
        title_fontsize=10,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out = repo / "exp/verdict_factor_judge/analysis/phase4/stage5/pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}  size: {out.stat().st_size / 1024:.1f} KB  "
          f"({n_gold}/{len(stage5_rows)} gold)")


if __name__ == "__main__":
    main()
