"""Phase 4 R1 alpha-sweep plot — 2 panels (p1, p2), x=alpha, y_left=SR,
y_right=inference_ratio.

Inputs:
  exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl

Output:
  exp/verdict_factor_judge/analysis/phase4_alpha_sweep.png

Each subplot includes a horizontal anchor line at the recipe's phase 3
SR anchor (p1=0.95, p2=0.96). For G_R1 the operator inspects whether
each curve has a clear argmax over alpha and whether SR(alpha*) sits
within 2pp of the anchor (continue rule §3.1).

Usage::

    MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.plot_alpha_sweep_phase4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless / no Qt
import matplotlib.pyplot as plt  # noqa: E402

from exp.verdict_factor_judge.phase4_spec import ANCHOR_SR  # noqa: E402


WARM_COST = 0.75


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _compute_inf(row: dict) -> float:
    n = row.get("n_eval_verdicts") or 0
    if not n:
        return 0.0
    return (
        row.get("n_full_hit", 0) * 0.0
        + row.get("n_warm_start", 0) * WARM_COST
        + row.get("n_miss", 0) * 1.0
    ) / n


def _load_summary(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"phase4 R1 summary not found: {path}")
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_one_panel(ax, recipe_id: str, rows: list[dict]) -> None:
    rs = sorted(
        (r for r in rows if r.get("recipe_id") == recipe_id),
        key=lambda r: float(r.get("alpha", 0.0)),
    )
    if not rs:
        ax.text(0.5, 0.5, f"no data for {recipe_id}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(recipe_id, fontsize=11, fontweight="bold")
        return

    alphas = [float(r["alpha"]) for r in rs]
    srs = [r.get("success_rate") or 0.0 for r in rs]
    infs = [_compute_inf(r) for r in rs]

    sr_line, = ax.plot(
        alphas, srs, "o-", color="steelblue", linewidth=2.0, markersize=8,
        label="success rate",
    )
    ax.set_xlabel("alpha (offline weight share)", fontsize=11)
    ax.set_ylabel("success rate", color="steelblue", fontsize=11)
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 1.02)
    ax.grid(which="major", alpha=0.4, linewidth=0.6)

    # phase 3 anchor SR
    anchor = ANCHOR_SR.get(recipe_id)
    anchor_line = None
    if anchor is not None:
        anchor_line = ax.axhline(
            anchor, color="gray", linestyle=":", linewidth=1.5,
            label=f"phase3 anchor SR={anchor:.2f}",
        )
        ax.axhline(
            anchor - 0.02, color="lightgray", linestyle=":", linewidth=0.8,
            alpha=0.6,
        )

    ax2 = ax.twinx()
    inf_line, = ax2.plot(
        alphas, infs, "s--", color="crimson", linewidth=1.6, markersize=7,
        label="inference ratio",
    )
    ax2.set_ylabel("inference ratio", color="crimson", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(-0.02, 1.02)

    handles = [sr_line, inf_line]
    if anchor_line is not None:
        handles.append(anchor_line)
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_title(recipe_id, fontsize=11, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl",
    )
    parser.add_argument(
        "--out",
        default="exp/verdict_factor_judge/analysis/phase4_alpha_sweep.png",
    )
    args = parser.parse_args()

    rows = _load_summary(Path(args.summary))
    fig, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        "Phase 4 R1 — alpha sweep: SR (left, blue) and inference_ratio "
        "(right, red) per recipe",
        fontsize=14, fontweight="bold",
    )
    _plot_one_panel(ax_p1, "p1_state_fut_online_act", rows)
    _plot_one_panel(ax_p2, "p2_action_fut_online_act", rows)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(
        f"saved -> {out_path}  size: {out_path.stat().st_size / 1024:.1f} KB"
    )


if __name__ == "__main__":
    main()
