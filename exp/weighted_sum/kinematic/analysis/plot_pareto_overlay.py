"""Plot a 4-frontier (inf, SR) Pareto overlay.

Frontiers drawn (per plan §7.1 decision rule):

  - r/p baseline frontier (gray dashed) — gate-level no-signal floor;
    valid across retrievals since the gate is retrieval-agnostic.
  - threshold_pareto d1 envelope (teal solid) — same d1 retrieval as this
    experiment but with cp1_score-direct judging.
  - This experiment's 237-cell frontier (red solid) — colored by group
    (g1..g5) for scatter; one overall frontier curve.
  - phase5 d4 native frontier (purple solid, reference only) — *different*
    retrieval (weighted_rrf_knn d4); labeled with cross-retrieval caveat.

Self-measured always-WARM 3 anchors (red stars) replace phase5's d4 rrf
reused (0.942 / 0.952 / 0.976) values for ceiling marking; see plan §1.5
on why cross-retrieval ceiling reuse is invalid.

Usage:
    PYTHONPATH=. uv run exp/weighted_sum/kinematic/analysis/plot_pareto_overlay.py \\
        --summary exp/weighted_sum/data/kinematic_phase5/per_yaml_summary.jsonl \\
        --out-dir exp/weighted_sum/data/kinematic_phase5
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

WARM_COST = 0.75
_GROUP_COLOR = {
    "g1": ("#1f77b4", "o", "G1 single-window"),
    "g2": ("#ff7f0e", "s", "G2 multi-window"),
    "g3": ("#2ca02c", "^", "G3 weight pattern"),
    "g4": ("#d62728", "D", "G4 multi-factor subset"),
    "g5": ("#9467bd", "P", "G5 threshold grid"),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _compute_inf(row: dict) -> float:
    """row → inference_ratio (warm=0.75, miss=1.0, hit=0.0)."""
    n = row.get("n_eval_verdicts") or (
        row.get("n_full_hit", 0) + row.get("n_warm_start", 0) + row.get("n_miss", 0)
    )
    if not n:
        return 0.0
    return (
        row.get("n_full_hit", 0) * 0.0
        + row.get("n_warm_start", 0) * WARM_COST
        + row.get("n_miss", 0) * 1.0
    ) / n


def _pareto_upper_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort by inf asc, keep only points with strictly higher SR than prefix."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front: list[tuple[float, float]] = []
    best = -1.0
    for inf, sr in pts:
        if sr > best:
            front.append((inf, sr))
            best = sr
    return front


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in d:
            continue
        rows.append(d)
    return rows


def _load_random_periodic_csv(path: Path) -> list[tuple[float, float]]:
    """Load r/p baseline (inf, SR) pairs from aggregate.csv if present."""
    if not path.exists():
        return []
    try:
        from exp.verdict_factor_judge.analysis.phase3.plot_pareto import (
            _load_random_periodic,
        )

        return list(_load_random_periodic(path))
    except ImportError:
        return []


def _load_threshold_pareto_d1(repo: Path) -> list[tuple[float, float]]:
    """Load threshold_pareto d1 (inf, SR) pairs from existing csv if present."""
    csv_path = repo / "exp/weighted_sum/analysis/threshold_pareto_per_yaml.csv"
    if not csv_path.exists():
        return []
    import csv

    out: list[tuple[float, float]] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("base_depth") != "1":
                continue
            try:
                inf = float(row["inf_ratio"])
                sr = float(row["success_rate_pct"]) / 100.0
            except (KeyError, ValueError):
                continue
            out.append((inf, sr))
    return out


def _load_phase5_d4_native(repo: Path) -> list[tuple[float, float]]:
    """Load phase5 d4 native frontier from existing per_yaml_summary.jsonl."""
    p5_path = repo / "exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl"
    if not p5_path.exists():
        return []
    out: list[tuple[float, float]] = []
    for row in _load_jsonl(p5_path):
        if not row.get("n_eval_verdicts"):
            continue
        out.append((_compute_inf(row), row.get("success_rate", 0.0)))
    return [p for p in out if p[0] > 0]


def _load_self_always_warm(data_dir: Path) -> list[tuple[float, float]]:
    """Load self-measured always-warm 3 anchors from always_warm_results.json."""
    p = data_dir / "always_warm_results.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    # Schema: {"start_t_0.3": {"success_rate": 0.94, "inf": ...}, ...}
    out: list[tuple[float, float]] = []
    for v in d.values():
        if not isinstance(v, dict):
            continue
        inf = v.get("inf")
        sr = v.get("success_rate")
        if inf is None or sr is None:
            continue
        out.append((float(inf), float(sr)))
    return out


# ----------------------------------------------------------------------
# Main plot
# ----------------------------------------------------------------------


def main(*, summary_path: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build the 4-frontier overlay and save it to out_dir/pareto_overlay.png."""
    repo = Path(__file__).resolve().parents[4]
    summary_path = summary_path or (
        repo / "exp/weighted_sum/data/kinematic_phase5/per_yaml_summary.jsonl"
    )
    out_dir = out_dir or summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data sources
    # ------------------------------------------------------------------
    rp_pts = _load_random_periodic_csv(
        repo / "exp/random_periodic_gate/analysis/aggregate.csv"
    )
    rp_pts = [p for p in rp_pts if p[0] > 0]
    th_d1_pts = _load_threshold_pareto_d1(repo)
    p5_pts = _load_phase5_d4_native(repo)
    self_warm = _load_self_always_warm(out_dir)

    # This experiment: 237-cell summary
    this_rows = _load_jsonl(summary_path)
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    this_pts: list[tuple[float, float]] = []
    for r in this_rows:
        if not r.get("n_eval_verdicts"):
            continue
        inf = _compute_inf(r)
        sr = r.get("success_rate", 0.0)
        if inf <= 0:
            continue
        by_group[r.get("group", "?")].append((inf, sr))
        this_pts.append((inf, sr))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    # r/p baseline
    rp_front = _pareto_upper_frontier(rp_pts) if rp_pts else []
    if rp_front:
        fx, fy = zip(*rp_front)
        ax.plot(
            fx,
            fy,
            "--",
            color="#888888",
            lw=2.0,
            alpha=0.85,
            label=f"r/p baseline ({len(rp_front)} pts)",
            zorder=2,
        )

    # threshold_pareto d1 envelope
    th_d1_front = _pareto_upper_frontier(th_d1_pts) if th_d1_pts else []
    if th_d1_front:
        fx, fy = zip(*th_d1_front)
        ax.plot(
            fx,
            fy,
            "-",
            color="#0f766e",
            lw=2.5,
            alpha=0.95,
            label=f"threshold_pareto d1 cp1_score ({len(th_d1_front)} pts)",
            zorder=3,
        )

    # phase5 d4 reference (cross-retrieval — labeled as such)
    p5_front = _pareto_upper_frontier(p5_pts) if p5_pts else []
    if p5_front:
        fx, fy = zip(*p5_front)
        ax.plot(
            fx,
            fy,
            ":",
            color="#7e3eb5",
            lw=2.0,
            alpha=0.80,
            label=f"phase5 d4 ref (different retrieval; {len(p5_front)} pts)",
            zorder=3,
        )

    # This experiment — per-group scatter + overall frontier
    for g, (color, marker, label) in _GROUP_COLOR.items():
        sub = by_group.get(g, [])
        if not sub:
            continue
        ax.scatter(
            [p[0] for p in sub],
            [p[1] for p in sub],
            s=22,
            color=color,
            marker=marker,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.4,
            label=f"{label} ({len(sub)})",
            zorder=4,
        )
    this_front = _pareto_upper_frontier(this_pts) if this_pts else []
    if this_front:
        fx, fy = zip(*this_front)
        ax.plot(
            fx,
            fy,
            "-o",
            color="#d62728",
            lw=2.8,
            ms=5,
            alpha=1.0,
            label=f"kinematic-on-ws-d1 frontier ({len(this_front)} pts)",
            zorder=6,
        )

    # Self-measured always-WARM anchors (red stars)
    if self_warm:
        ax.scatter(
            [p[0] for p in self_warm],
            [p[1] for p in self_warm],
            marker="*",
            s=240,
            color="red",
            edgecolor="darkred",
            linewidths=1.2,
            zorder=7,
            label=f"self always-WARM d1 ({len(self_warm)} anchors)",
        )

    # ------------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------------
    ax.set_xlabel("inference_ratio (lower = more cache reuse / less compute)", fontsize=12)
    ax.set_ylabel("success_rate", fontsize=12)
    ax.set_title(
        "kinematic-on-weighted_sum d1 (237 cell) vs r/p / threshold_pareto / phase5 d4\n"
        "★ = self-measured always-WARM ceiling (phase5 d4 anchors NOT reused — cross-retrieval invalid)",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.55, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.4, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95, ncol=2)

    fig.tight_layout()
    out_path = out_dir / "pareto_overlay.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "pareto_overlay.pdf", bbox_inches="tight")
    logger.info("[plot] saved %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(
        summary_path=Path(args.summary) if args.summary else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    _cli()
