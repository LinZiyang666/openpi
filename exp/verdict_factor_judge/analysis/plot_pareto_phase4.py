"""Phase 4 Pareto overlay — phase 3 (faded) + phase 4 R1/R2/R3 cells.

Overlays the phase 4 weight-sweep results on the same (inference_ratio,
success_rate) plane that `plot_pareto_phase3.py` already established.
The phase 3 cells appear as a light-gray cloud for orientation; the
phase 4 R1/R2/R3 cells are colored per recipe + per round so the
operator can see whether weight fusion shifted any cell off the
phase 3 frontier.

Inputs:
  * exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl
  * exp/verdict_factor_judge/data/phase4/r{1,2,3}_*/per_yaml_summary.jsonl
  * exp/random_periodic_gate/analysis/aggregate.csv

Output: exp/verdict_factor_judge/analysis/phase4_pareto.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless / no Qt
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from exp.verdict_factor_judge.analysis.plot_pareto_phase3 import (  # noqa: E402
    _load_phase3,
    _load_random_periodic,
    is_pareto_dominated,
    pareto_upper_frontier,
    warm_cost,
)


WARM_COST = 0.75
CFG_KEYBUILDER = "spatial16_w8_d4"
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


def _load_phase4_round(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    rows: list[dict] = []
    for line in summary_path.read_text().splitlines():
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
    repo = Path(__file__).resolve().parents[3]
    rp_pts = _load_random_periodic(
        repo / "exp/random_periodic_gate/analysis/aggregate.csv"
    )
    phase3_rows = _load_phase3(
        repo / "exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl"
    )

    phase4_dirs = {
        "R1": repo / "exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl",
        "R2": repo / "exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl",
        "R3": repo / "exp/verdict_factor_judge/data/phase4/r3_online_desc/per_yaml_summary.jsonl",
    }
    phase4_rows = {tag: _load_phase4_round(p) for tag, p in phase4_dirs.items()}

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    fig.suptitle(
        "Phase 4 weight sweep on phase 3 Pareto plane (spatial16; "
        "phase 4 cells use locked per-recipe (FH, WS) and warm cost 0.75)",
        fontsize=14, fontweight="bold",
    )

    # 1) random / periodic baseline cloud + frontier (lightest)
    rx, ry = (zip(*rp_pts) if rp_pts else ([], []))
    ax.scatter(
        rx, ry, s=22, c="lightgray", marker="o", alpha=0.55,
        label=f"random / periodic ({len(rp_pts)} pts)", zorder=1,
    )
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
    ax.scatter(
        wx, wy, s=320, c="red", marker="*", edgecolors="darkred",
        linewidths=1.4, label="always-WARM (start_t)", zorder=4,
    )
    for (x, y), t in zip(warm_pts, (0.30, 0.50, 0.70)):
        ax.annotate(f" t={t}", (x, y), xytext=(8, -4),
                    textcoords="offset points", fontsize=10, color="darkred",
                    fontweight="bold")
    ax.plot(wx, wy, "--", color="red", alpha=0.4, linewidth=1.0, zorder=3)

    # 3) phase 3 cells — faded grey reference cloud
    p3_x = [r["inf"] for r in phase3_rows]
    p3_y = [r["sr"] for r in phase3_rows]
    ax.scatter(
        p3_x, p3_y, s=42, c="silver", marker="o", alpha=0.7,
        edgecolors="gray", linewidths=0.6,
        label=f"phase 3 ({len(phase3_rows)} cells)", zorder=2,
    )

    # 4) phase 4 cells — per (round, recipe) marker + color
    round_marker = {"R1": "o", "R2": "s", "R3": "^"}
    recipe_color = {
        "p1_state_fut_online_act": "tab:blue",
        "p2_action_fut_online_act": "tab:orange",
    }
    for tag, rows in phase4_rows.items():
        for rid, color in recipe_color.items():
            sub = [r for r in rows if r.get("recipe_id") == rid]
            if not sub:
                continue
            xs = [r["inf"] for r in sub]
            ys = [r["sr"] for r in sub]
            ax.scatter(
                xs, ys, s=110, color=color, marker=round_marker[tag],
                edgecolors="black", linewidths=0.8, alpha=0.92,
                label=f"phase 4 {tag} {rid.split('_')[0]} ({len(sub)})",
                zorder=5,
            )

    # 5) gold-circle phase 4 cells (strict Pareto positive vs r/p + always-WARM)
    all_base = list(rp_pts) + warm_pts
    n_p4_gold = 0
    for tag, rows in phase4_rows.items():
        for r in rows:
            if not is_pareto_dominated(r["inf"], r["sr"], all_base):
                ax.scatter(
                    [r["inf"]], [r["sr"]], s=240, facecolors="none",
                    edgecolors="gold", linewidths=2.4, zorder=6,
                )
                n_p4_gold += 1

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

    n_total = sum(len(rows) for rows in phase4_rows.values())
    ax.legend(
        loc="lower right", fontsize=9, framealpha=0.95, ncol=2,
        title=f"gold-circle = strict Pareto positive ({n_p4_gold}/{n_total})",
        title_fontsize=10,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out = repo / "exp/verdict_factor_judge/analysis/phase4_pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(
        f"saved -> {out}  size: {out.stat().st_size / 1024:.1f} KB  "
        f"({n_p4_gold}/{n_total} gold)"
    )


if __name__ == "__main__":
    main()
