"""Draw the LIBERO x GR00T accuracy-cost frontier, one panel per suite.

The x axis is the *realized* inference ratio -- what each arm actually spent,
priced from its stored per-step verdicts -- not the ratio it was addressed at.
That distinction is the whole point: a grid addressed at 20..95 lands where the
gate and the closed-loop score distribution let it land, and the frontier is a
statement about what was spent, not about what was asked for.

Four families share the axis:

  RIT k=1/2/3   one tolerance inverted into one cut per rung; the ladder depth
                is the series, and at a fixed budget the three are directly
                comparable because they were addressed at the same one.
  GST           percent triples on the cost simplex, the grid-search reference.
  anchors       the two ends the axis is pinned to: every step answered from
                the cache (the cheap end) and every step answered by the
                teacher (IR = 100 by construction).

The Pareto front is drawn over all sweep arms together, since a practitioner
picking an operating point does not care which rule produced it.

Usage:
  uv run python -m exp.libero_groot.plot_rit_frontier
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openpi.cache.types import groot_n15_schedule  # noqa: E402

from exp.robocasa365 import rit_cost_rc as rc  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL = ROOT / "exp/libero_groot/data/rit/eval"
CFG = ROOT / "exp/libero_groot/config/rit"
#: Figures and their backing data land beside the Pi0.5 line's, so the two
#: teachers' frontiers are read from one directory. Names carry the teacher
#: because that directory already holds the Pi0.5 panels.
OUT = ROOT / "exp/rit_pareto/analysis/figures"
STEM = "groot_libero_frontier"
SUITES = ("libero_spatial", "libero_10")
TITLE = {"libero_spatial": "LIBERO-Spatial", "libero_10": "LIBERO-10"}

INK, MUTED, GRID = "#1B2124", "#6B7780", "#DFE4E7"
#: Ladder depth is the categorical dimension; hues are assigned in a fixed
#: order and never cycled. GST is the reference rule and takes the neutral.
SERIES = {
    "rit1": ("#2E6FD9", "o", "RIT  k=1  (hit / miss)"),
    "rit2": ("#C8641E", "s", "RIT  k=2  (+ warm, 2 steps left)"),
    "rit3": ("#7A4FBF", "^", "RIT  k=3  (+ warm, 4 steps left)"),
    "gst": ("#8A9299", "D", "GST  grid-searched thresholds"),
}


def load_cost() -> rc.StageCost:
    d = json.loads((CFG / "cost_groot_libero_measured.json").read_text(encoding="utf-8"))
    return rc.StageCost(
        teacher=d["teacher"],
        schedule=groot_n15_schedule(int(d["num_steps"])),
        stage1_ms=float(d["stage1_ms"]),
        stage2_ms=float(d["stage2_ms"]),
        stage3_head_ms=float(d["stage3_head_ms"]),
        stage3_step_ms=float(d["stage3_step_ms"]),
        provenance=str(d["provenance"]),
        linear_stage3=bool(d.get("linear_stage3", False)),
    )


def realized_ir(counts: dict, cost: rc.StageCost) -> float | None:
    """Priced verdict mix as a percent of the all-MISS cost.

    Warm counts come from the per-tier keys, never the pooled total: the two
    rungs differ by 17 points of the axis, so pooling them would erase most of
    what the ladder is measuring.
    """
    n = sum(v for k, v in counts.items()
            if k in ("FULL_HIT", "MISS") or k.startswith("WARM_START@"))
    if not n:
        return None
    spend = counts.get("FULL_HIT", 0) * rc.tier_cost(cost, "FULL_HIT", None)
    spend += counts.get("MISS", 0) * rc.miss_cost(cost)
    for key, value in counts.items():
        if key.startswith("WARM_START@"):
            spend += value * rc.tier_cost(cost, "WARM_START", float(key.split("@")[1]))
    return 100.0 * spend / (n * rc.miss_cost(cost))


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated points, cheapest first: nothing else is both no dearer and no worse."""
    keep = [p for p in points
            if not any(q is not p and q[0] <= p[0] and q[1] >= p[1]
                       and (q[0] < p[0] or q[1] > p[1]) for q in points)]
    return sorted(keep)


def collect(suite: str, cost: rc.StageCost) -> tuple[dict, dict]:
    agg = json.loads((EVAL / f"aggregate_{suite}.json").read_text(encoding="utf-8"))
    rec = json.loads((CFG / suite / "arm_record.json").read_text(encoding="utf-8"))["arms"]
    series: dict[str, list[tuple[float, float]]] = {k: [] for k in SERIES}
    anchors: dict[str, tuple[float, float]] = {}
    for arm, r in agg.items():
        if r["n_ep"] < 500:
            continue
        ir = realized_ir(r["counts"], cost)
        if ir is None:
            continue
        point = (ir, r["success_rate"])
        meta = rec.get(arm, {})
        rule = meta.get("rule")
        if rule == "anchor":
            anchors["cache" if "cache" in arm else "teacher"] = point
        elif rule == "gst":
            series["gst"].append(point)
        elif rule == "rit":
            series[f"rit{meta['k']}"].append(point)
    return series, anchors


def draw(ax, suite: str, series: dict, anchors: dict) -> None:
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    teacher = anchors.get("teacher")
    if teacher:
        ax.axhline(teacher[1], color=INK, linewidth=1.1, linestyle=(0, (5, 3)), zorder=2)
        ax.annotate("teacher, every step", xy=(2, teacher[1]), xytext=(2, teacher[1] + 0.008),
                    color=INK, fontsize=8.5)

    for key, (color, marker, _) in SERIES.items():
        pts = series.get(key, [])
        if pts:
            ax.plot(*zip(*pts), linestyle="none", marker=marker, color=color,
                    markersize=5.5, markeredgecolor="white", markeredgewidth=0.9,
                    alpha=0.85, zorder=3)

    everything = [p for pts in series.values() for p in pts]
    front = pareto(everything)
    if len(front) > 1:
        ax.plot(*zip(*front), "-", color=INK, linewidth=1.6, alpha=0.55, zorder=4)

    for name, (color, label) in (("cache", (INK, "cache, every step")),
                                 ("teacher", (INK, None))):
        p = anchors.get(name)
        if p and name == "cache":
            ax.plot([p[0]], [p[1]], marker="*", color=color, markersize=13,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=5)
            ax.annotate(label, xy=p, xytext=(p[0] + 1.5, p[1] - 0.004),
                        color=INK, fontsize=8.5)

    ax.set_xlim(0, 104)
    ax.set_title(TITLE[suite], color=INK, fontsize=12, loc="left", pad=8)
    ax.set_xlabel("realized inference ratio  (% of the all-MISS cost)",
                  color=MUTED, fontsize=10)


def main() -> None:
    cost = load_cost()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for ax, suite in zip(axes, SUITES):
        series, anchors = collect(suite, cost)
        draw(ax, suite, series, anchors)
    axes[0].set_ylabel("success rate  (500 episodes per arm)", color=MUTED, fontsize=10)

    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=6,
                          markeredgecolor="white", markeredgewidth=0.9, label=lab)
               for c, m, lab in SERIES.values()]
    handles.append(plt.Line2D([], [], color=INK, marker="*", linestyle="none",
                              markersize=10, label="cache-only / teacher-only anchors"))
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.6, alpha=0.55,
                              label="Pareto front over all sweep arms"))
    axes[1].legend(handles=handles, loc="lower right", frameon=False,
                   fontsize=9, labelcolor=INK)

    fig.suptitle(
        "GR00T N1.5 warm-start cache on LIBERO -- accuracy against inference cost",
        color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985,
    )
    fig.text(0.006, 0.005,
             f"cost model: stage1 {cost.stage1_ms:.2f} / stage2 {cost.stage2_ms:.2f} / "
             f"stage3 {cost.stage3_ms(8):.2f} ms (k=8, RTX 4090, CUDA graph); "
             f"FULL {100 * rc.tier_cost(cost, 'FULL_HIT', None) / rc.miss_cost(cost):.1f}%, "
             f"warm rungs {100 * rc.tier_cost(cost, 'WARM_START', 0.75) / rc.miss_cost(cost):.1f}% "
             f"and {100 * rc.tier_cost(cost, 'WARM_START', 0.5) / rc.miss_cost(cost):.1f}% of MISS",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}/{STEM}.png / .pdf")

    # The figure's own data, so the panels can be re-read or re-drawn without
    # going back to the run directories on the worker boxes.
    data = {
        "schema": "libero_groot.rit_frontier/v1",
        "teacher": "groot_n15",
        "episodes_per_arm": 500,
        "x_label": "realized inference ratio (% of the all-MISS cost)",
        "y_label": "success rate",
        "cost_model": {
            "stage1_ms": cost.stage1_ms,
            "stage2_ms": cost.stage2_ms,
            "stage3_full_loop_ms": cost.stage3_ms(8),
            "num_steps": 8,
            "unit_cost_ms": {
                "FULL_HIT": rc.tier_cost(cost, "FULL_HIT", None),
                "WARM_START@0.75": rc.tier_cost(cost, "WARM_START", 0.75),
                "WARM_START@0.5": rc.tier_cost(cost, "WARM_START", 0.5),
                "MISS": rc.miss_cost(cost),
            },
            "provenance": cost.provenance,
        },
        "suites": {},
    }
    for suite in SUITES:
        series, anchors = collect(suite, cost)
        everything = [p for pts in series.values() for p in pts]
        data["suites"][suite] = {
            "series": {k: [{"ir": x, "sr": y} for x, y in sorted(v)]
                       for k, v in series.items()},
            "anchors": {k: {"ir": v[0], "sr": v[1]} for k, v in anchors.items()},
            "pareto_front": [{"ir": x, "sr": y} for x, y in pareto(everything)],
        }
    (OUT / f"{STEM}.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}/{STEM}.json")


if __name__ == "__main__":
    main()
