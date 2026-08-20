"""Figures for the cache-effectiveness ablation (Phase 5), rendered from one data file.

  fig1  ablation_sr_matrix.{png,pdf}
        Per-arm SR with Wilson 95% CIs, both suites, teacher anchor lines.
  fig2  ablation_pareto_inference_rate.{png,pdf}
        SR vs teacher-inference rate. Overlays the 4b kinematic-verdict threshold
        sweep, the retrieval-threshold routing points, and the pure-student /
        student-at-miss / teacher anchors.

This module reads **only** ``plot_data.json`` (built by ``emit_plot_data.py``)
and never touches the conductor journals or per-step exports. That indirection
is the point: the raw tree is gitignored and lives off this disk, so the figures
have to be reproducible from a small versioned file. Every abscissa, ordinate,
CI, series membership and legend label comes from the data file; what stays here
is presentation only -- colours, marker shapes, axis furniture.

Run:  uv run python exp/ablation_study/analysis/plot_ablation.py
      [--data <plot_data.json>] [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DIR = pathlib.Path("exp/ablation_study/analysis")
SUITES = ("libero_spatial", "libero_10")

SWEEP_SERIES = "kinematic_verdict_sweep"
SWEEP_COLOUR = "#1f77b4"
TEACHER_COLOUR = "crimson"

# Marker shape / size / colour per arm. Presentation only: the arm's meaning and
# its coordinates live in plot_data.json.
STYLE: dict[str, tuple[str, float, str]] = {
    "hit_act": ("*", 230, "#d62728"),
    "hit_smolvla": ("*", 230, "#9467bd"),
    "cache_baseline": ("*", 230, "#2ca02c"),
    "pure_act": ("^", 70, "#d62728"),
    "pure_smolvla": ("^", 70, "#9467bd"),
    "miss_act": ("v", 70, "#8c564b"),
    "miss_smolvla": ("v", 70, "#e377c2"),
}

# fig-2 draw order, which is also the legend order.
ANCHOR_ORDER = ("hit_act", "hit_smolvla", "cache_baseline",
                "pure_act", "pure_smolvla", "miss_act", "miss_smolvla")


# ------------------------------------------------------------------
# Data access
# ------------------------------------------------------------------
def load(path: pathlib.Path) -> dict:
    """Read the data file and check it carries the families the figures need."""
    data = json.loads(path.read_text())
    families = data.get("families", {})
    missing = [s for s in SUITES if s not in families]
    if missing:
        raise SystemExit(
            f"{path}: no data for {missing}; collect them with emit_plot_data.py "
            "before plotting (the figures never read the raw journals)"
        )
    return data


def by_arm(family: dict) -> dict[str, dict]:
    return {p["arm"]: p for p in family["points"]}


def sweep_points(family: dict) -> list[dict]:
    pts = [p for p in family["points"] if p["series"] == SWEEP_SERIES]
    return sorted(pts, key=lambda p: p["sweep_order"])


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def fig1(data: dict, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, suite in zip(axes, SUITES):
        fam = data["families"][suite]
        arms = fam["matrix_arm_order"]
        pts = by_arm(fam)
        teacher = fam["teacher_anchor"]
        xs = range(len(arms))
        ys = [pts[a]["success_rate"] for a in arms]
        los = [pts[a]["success_rate"] - pts[a]["success_rate_ci95"][0] for a in arms]
        his = [pts[a]["success_rate_ci95"][1] - pts[a]["success_rate"] for a in arms]
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", ms=6, capsize=3,
                    color="#1f77b4", ecolor="#1f77b4", lw=0, elinewidth=1.4)
        ax.axhline(teacher["success_rate"], ls="--", lw=1.2, color=TEACHER_COLOUR)
        ax.text(0.02, teacher["success_rate"] + 0.006, teacher["label"],
                color=TEACHER_COLOUR, fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_xticks(list(xs))
        ax.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=8)
        ax.set_title(suite)
        n = pts[arms[0]]["n_episodes"]
        ax.set_ylabel(f"success rate ({n} ep, Wilson 95% CI)")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Main matrix: executor substitution under a byte-identical "
                 "retrieval/verdict chain")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ablation_sr_matrix.{ext}", dpi=200)
    plt.close(fig)


def fig2(data: dict, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, suite in zip(axes, SUITES):
        fam = data["families"][suite]
        pts = by_arm(fam)
        teacher = fam["teacher_anchor"]

        sweep = sweep_points(fam)
        ax.plot([p["teacher_inference_rate"] for p in sweep],
                [p["success_rate"] for p in sweep],
                "-o", ms=5, lw=1.8, color=SWEEP_COLOUR,
                label=f"{sweep[0]['label']} (threshold sweep)")
        for p in sweep:
            ax.annotate(p["threshold_tag"].replace("fh", "fh."),
                        xy=(p["teacher_inference_rate"], p["success_rate"]),
                        xytext=(4, -11), textcoords="offset points",
                        fontsize=7.5, color=SWEEP_COLOUR)

        for arm in ANCHOR_ORDER:
            p = pts[arm]
            marker, size, colour = STYLE[arm]
            ax.scatter([p["teacher_inference_rate"]], [p["success_rate"]],
                       marker=marker, s=size, color=colour,
                       zorder=5 if marker == "*" else 4, label=p["label"])

        ax.scatter([teacher["teacher_inference_rate"]], [teacher["success_rate"]],
                   marker="s", s=70, color=TEACHER_COLOUR, zorder=5,
                   label=teacher["label"])
        ax.axhline(teacher["success_rate"], ls="--", lw=1.0,
                   color=TEACHER_COLOUR, alpha=0.5)

        ax.set_xlabel("inference rate  (fraction of steps on teacher Stage1-3 "
                      "= 1 - hit rate)")
        ax.set_ylabel("success rate")
        ax.set_title(suite)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.6,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("SR vs teacher-inference rate: student routing frontiers and anchors")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ablation_pareto_inference_rate.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DIR / "plot_data.json"),
                    help="data file built by emit_plot_data.py")
    ap.add_argument("--out-dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()

    data = load(pathlib.Path(args.data))
    out = pathlib.Path(args.out_dir)
    fig1(data, out)
    fig2(data, out)
    print("wrote:", [str(p) for p in sorted(out.glob("ablation_*.png"))])


if __name__ == "__main__":
    main()
