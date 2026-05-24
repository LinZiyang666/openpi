"""Phase 5 libero_10 Pareto plot — 240 cells × 5 group on (inference_ratio, success_rate) plane.

Inference ratio computed per `analysis/phase5/plot_pareto.py` convention:
  inf = (n_full_hit*0.0 + n_warm_start*0.75 + n_miss*1.0) / n_eval_verdicts

No external baselines (libero_10 has no random/periodic / always-WARM ground truth in this repo).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

WARM_COST = 0.75


def _compute_inf(row: dict) -> float:
    n = row.get("n_eval_verdicts") or 0
    if not n:
        return 0.0
    return (
        row.get("n_full_hit", 0) * 0.0
        + row.get("n_warm_start", 0) * WARM_COST
        + row.get("n_miss", 0) * 1.0
    ) / n


def _load(path: Path) -> list[dict]:
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


def _pareto_upper_frontier(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return points that are not dominated by any other (lower inf is better,
    higher sr is better)."""
    by_x = sorted(pts, key=lambda p: (p[0], -p[1]))
    front: list[tuple[float, float]] = []
    best_y = -1.0
    for x, y in by_x:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


def main() -> None:
    repo = Path(__file__).resolve().parents[4]
    summary_path = repo / "exp/verdict_factor_judge/data/phase5_libero10_systematic/per_yaml_summary.jsonl"
    out_path = repo / "exp/verdict_factor_judge/analysis/phase5_libero10/pareto.png"
    rows = _load(summary_path)
    print(f"loaded {len(rows)} cells from {summary_path}")

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.suptitle(
        f"Phase 5 libero_10 systematic sweep — 240 cell × 100ep on Pareto plane "
        f"(spatial16; warm cost {WARM_COST} @ start_t=0.5)",
        fontsize=13, fontweight="bold",
    )

    GROUP_STYLE = {
        "g1": dict(color="#1f77b4", marker="o", label="G1 (online single window)"),
        "g2": dict(color="#ff7f0e", marker="s", label="G2 (online multi-window)"),
        "g3": dict(color="#2ca02c", marker="^", label="G3 (weight pattern sweep)"),
        "g4": dict(color="#9467bd", marker="D", label="G4 (factor subset)"),
        "g5": dict(color="#d62728", marker="*", label="G5 (threshold sweep)"),
    }
    for g, style in GROUP_STYLE.items():
        gx = [r["inf"] for r in rows if r.get("group") == g]
        gy = [r["sr"] for r in rows if r.get("group") == g]
        ax.scatter(gx, gy, s=80 if g == "g5" else 50,
                   color=style["color"], marker=style["marker"],
                   edgecolors="black", linewidths=0.4, alpha=0.85,
                   label=f"{style['label']} (n={len(gx)})", zorder=3)

    # Pareto upper frontier across all 240 cells
    all_pts = [(r["inf"], r["sr"]) for r in rows]
    frontier = _pareto_upper_frontier(all_pts)
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, "-", color="gold", linewidth=2.5, alpha=0.9,
            label=f"Pareto upper frontier ({len(frontier)} pts)", zorder=4)
    ax.scatter(fx, fy, s=180, facecolors="none", edgecolors="gold",
               linewidths=2.0, zorder=5)

    # π0.5 no-cache baseline on libero-10 (OpenPI repo): 93.0% sr (at theoretical inf=1.0)
    PI05_LIBERO10_SR = 0.930
    ax.axhline(PI05_LIBERO10_SR, color="#444", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"π0.5 no-cache baseline sr={PI05_LIBERO10_SR:.3f} (libero-10)",
               zorder=2)

    ax.set_xlabel("inference_ratio (cost per step)", fontsize=12)
    ax.set_ylabel("success_rate", fontsize=12)
    # Data ranges: inf ∈ [0.29, 0.79], sr ∈ [0.58, 0.93]
    ax.set_xlim(0.25, 0.82)
    ax.set_ylim(0.55, 1.00)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.grid(which="major", alpha=0.45, linestyle="-")
    ax.grid(which="minor", alpha=0.18, linestyle=":")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
