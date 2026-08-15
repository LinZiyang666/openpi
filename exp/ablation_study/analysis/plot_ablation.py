"""Figures for the cache-effectiveness ablation (Phase 5).

Reads the conductor journals + per_step exports under data/runs/ and renders:

  fig1  ablation_sr_matrix.{png,pdf}
        Per-arm SR with Wilson 95% CIs, both suites, teacher anchor lines.
  fig2  ablation_pareto_inference_rate.{png,pdf}
        SR vs inference rate (fraction of steps executed by the teacher's
        full Stage1-3 pipeline = 1 - FULL_HIT rate, owner-ruled axis; the
        historical warmup-cost axis is deliberately NOT used). Overlays the
        kinematic-verdict threshold sweep (4b) with the retrieval-threshold
        verdict points and the replay/pure/teacher anchors from the main
        matrix.

Run:  PYTHONPATH=. uv run python exp/ablation_study/analysis/plot_ablation.py
Outputs land in exp/ablation_study/analysis/ (tracked, non-gitignored).
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = pathlib.Path("exp/ablation_study/data/runs")
OUT = pathlib.Path("exp/ablation_study/analysis")

SUITES = ["libero_spatial", "libero_10"]
TEACHER = {"libero_spatial": 0.974, "libero_10": 0.868}
MAIN_ARMS = ["cache_baseline", "hit_act", "hit_smolvla", "miss_act",
             "miss_smolvla", "pure_act", "pure_smolvla"]
FH_ORDER = ["fh67", "fh49", "fh40", "fh25", "fh11"]  # decreasing threshold

MAIN_JOURNAL = {"libero_spatial": "p4_libero_spatial_journal.jsonl",
                "libero_10": "p4_libero_10_journal.jsonl"}
MAIN_PERSTEP = {"libero_spatial": "p4_libero_spatial_per_step.jsonl",
                "libero_10": "p4_libero_10_per_step.jsonl"}
P4B_JOURNAL = {"libero_spatial": "p4b_sp_journal.jsonl",
               "libero_10": "p4b_l10_journal.jsonl"}
P4B_PERSTEP = {"libero_spatial": "p4b_sp_per_step.jsonl",
               "libero_10": "p4b_l10_per_step.jsonl"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def arm_sr(journal: pathlib.Path) -> dict[str, tuple[float, int, tuple[float, float]]]:
    n = defaultdict(int)
    k = defaultdict(int)
    for line in journal.read_text().splitlines():
        r = json.loads(line)
        a = r["yaml_id"]
        n[a] += 1
        k[a] += 1 if r.get("success") else 0
    return {a: (k[a] / n[a], n[a], wilson(k[a], n[a])) for a in n}


def full_hit_rate(per_step: pathlib.Path) -> dict[str, float]:
    tot = defaultdict(int)
    hit = defaultdict(int)
    for line in per_step.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        a = r.get("yaml_id")
        tot[a] += 1
        if "FULL_HIT" in str(r.get("hit_type")):
            hit[a] += 1
    return {a: hit[a] / tot[a] for a in tot}


def fig1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, suite in zip(axes, SUITES):
        sr = arm_sr(RUNS / MAIN_JOURNAL[suite])
        xs = range(len(MAIN_ARMS))
        ys = [sr[a][0] for a in MAIN_ARMS]
        los = [sr[a][0] - sr[a][2][0] for a in MAIN_ARMS]
        his = [sr[a][2][1] - sr[a][0] for a in MAIN_ARMS]
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", ms=6, capsize=3,
                    color="#1f77b4", ecolor="#1f77b4", lw=0, elinewidth=1.4)
        ax.axhline(TEACHER[suite], ls="--", lw=1.2, color="crimson")
        ax.text(0.02, TEACHER[suite] + 0.006, "teacher (all steps Pi0.5)",
                color="crimson", fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_xticks(list(xs))
        ax.set_xticklabels([a.replace("_", "\n") for a in MAIN_ARMS], fontsize=8)
        ax.set_title(suite)
        ax.set_ylabel("success rate (500 ep, Wilson 95% CI)")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Main matrix: executor substitution under a byte-identical retrieval/verdict chain")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ablation_sr_matrix.{ext}", dpi=200)
    plt.close(fig)


def fig2() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, suite in zip(axes, SUITES):
        m_sr = arm_sr(RUNS / MAIN_JOURNAL[suite])
        m_fh = full_hit_rate(RUNS / MAIN_PERSTEP[suite])
        b_sr = arm_sr(RUNS / P4B_JOURNAL[suite])
        b_fh = full_hit_rate(RUNS / P4B_PERSTEP[suite])

        # kinematic-verdict sweep (4b): x = 1 - FULL_HIT rate, MISS runs teacher
        pts = []
        for tag in FH_ORDER:
            arm = f"kinroute_act_{tag}"
            pts.append((1 - b_fh[arm], b_sr[arm][0], tag))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=5, lw=1.8,
                color="#1f77b4", label="kinematic verdict -> ACT at hit (threshold sweep)")
        for x, y, tag in pts:
            ax.annotate(tag.replace("fh", "fh."), xy=(x, y), xytext=(4, -11),
                        textcoords="offset points", fontsize=7.5, color="#1f77b4")

        # retrieval-threshold verdict points (main matrix, MISS runs teacher)
        for arm, color, label in [
            ("hit_act", "#d62728", "retrieval threshold -> ACT at hit"),
            ("hit_smolvla", "#9467bd", "retrieval threshold -> SmolVLA at hit"),
            ("cache_baseline", "#2ca02c", "cache replay at hit (full cache system)"),
        ]:
            x = 1 - m_fh[arm]
            ax.scatter([x], [m_sr[arm][0]], marker="*", s=230, color=color,
                       zorder=5, label=label)

        # zero-teacher-inference anchors (x = 0)
        for arm, marker, color, label in [
            ("pure_act", "^", "#d62728", "pure ACT (no teacher steps)"),
            ("pure_smolvla", "^", "#9467bd", "pure SmolVLA (no teacher steps)"),
            ("miss_act", "v", "#8c564b", "replay at hit + ACT at miss"),
            ("miss_smolvla", "v", "#e377c2", "replay at hit + SmolVLA at miss"),
        ]:
            ax.scatter([0], [m_sr[arm][0]], marker=marker, s=70, color=color,
                       zorder=4, label=label)

        # teacher anchor (x = 1)
        ax.scatter([1], [TEACHER[suite]], marker="s", s=70, color="crimson",
                   zorder=5, label="teacher (all steps Pi0.5)")
        ax.axhline(TEACHER[suite], ls="--", lw=1.0, color="crimson", alpha=0.5)

        ax.set_xlabel("inference rate  (fraction of steps on teacher Stage1-3 = 1 - hit rate)")
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
        fig.savefig(OUT / f"ablation_pareto_inference_rate.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote:", [str(p) for p in sorted(OUT.glob("ablation_*.png"))])
