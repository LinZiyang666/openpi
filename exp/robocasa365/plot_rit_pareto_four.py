"""Draw both teachers' RIT frontiers on one axis: four lines, two references.

Takes no arguments. It reads every figure spec in ``analysis/figures`` and
draws one line per (teacher, ladder depth), so this figure and the per-teacher
panels are the same numbers -- edit a point in the editor and both follow.

Each teacher's x axis is its own inference ratio -- percent of *that* teacher's
all-MISS cost -- because the two models' absolute latencies differ by 2x and a
shared millisecond axis would say the cheap teacher is always ahead. The ratio
is what the ladder is addressed by, so it is also what makes the curves
comparable.

Identity is carried twice over, never by colour alone: hue names the teacher
and dash pattern names the ladder depth. Four series therefore live inside a
two-hue palette rather than inventing two more hues, and the figure survives
both colour-vision deficiency and a greyscale print.

Usage:
  uv run python -m exp.robocasa365.plot_rit_pareto_four
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp.robocasa365.render_rit_figure import (  # noqa: E402
    FIGURES_DIR, GRID, INK, MUTED, SCHEMA, XDOM, anchor_of, overall_of, pareto, teacher_of,
)

#: Hue per teacher, fixed order, never cycled.
TEACHERS = {"groot_tp": ("#2E6FD9", "GR00T N1.5"), "pi05": ("#C8641E", "pi0.5")}
#: Dash per ladder depth: the shallower ladder is solid.
DASHES = ((0, ()), (0, (6, 2.5)))
OUT_STEM = FIGURES_DIR / "rit_pareto_both_teachers"


def load_specs() -> dict[str, dict]:
    out = {}
    for path in sorted(FIGURES_DIR.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(spec, dict) and spec.get("schema") == SCHEMA and spec.get("teacher"):
            out[spec["teacher"]] = spec
    return out


def main() -> None:
    specs = load_specs()
    if not specs:
        raise SystemExit(f"no {SCHEMA} spec with a teacher stamp under {FIGURES_DIR}")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    handles = []
    for teacher, (color, tlabel) in TEACHERS.items():
        spec = specs.get(teacher)
        if spec is None:
            continue
        for si, series in enumerate(spec["series"]):
            pts = [v for v in (overall_of(p, spec["tasks"]) for p in series["points"]) if v]
            if not pts:
                continue
            dash = DASHES[si % len(DASHES)]
            front = pareto(pts)
            if len(front) > 1:
                ax.plot(*zip(*front), color=color, linewidth=2.0, linestyle=dash, zorder=3)
            ax.plot(*zip(*pts), linestyle="none", marker="o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=4)
            depth = series["key"].split()[0]
            handles.append(plt.Line2D([], [], color=color, linestyle=dash, marker="o",
                                      linewidth=2.0, markersize=6, markeredgecolor="white",
                                      markeredgewidth=1.5, label=f"{tlabel}  {depth}"))
        anchor = anchor_of(spec, None)
        if anchor:
            ax.plot([anchor[0]], [anchor[1]], marker="D", color=color, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=5)
        th = teacher_of(spec, None)
        if th is not None:
            ax.axhline(th, color=color, linewidth=1.2, linestyle=(0, (2, 3)),
                       alpha=0.85, zorder=2)

    handles.append(plt.Line2D([], [], color=INK, marker="D", linestyle="none", markersize=7,
                              markeredgecolor="white", markeredgewidth=1.5,
                              label="all-FULL_HIT arm"))
    handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (2, 3)), linewidth=1.2,
                              label="teacher-only (no cache)"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5,
              labelcolor=INK, handlelength=2.8)

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlim(*XDOM)
    ax.set_title("RoboCasa365 warm-start RIT frontier -- both teachers, 13 tasks x 50 episodes",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("realized inference ratio  (% of that teacher's all-MISS cost)",
                  color=MUTED, fontsize=10)
    ax.set_ylabel("success rate  (macro over 13 tasks)", color=MUTED, fontsize=10)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_STEM}.png / .pdf  ({', '.join(sorted(specs))})")


if __name__ == "__main__":
    main()
