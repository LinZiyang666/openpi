"""Render every RIT figure spec in ``analysis/figures`` to png and pdf.

Takes no arguments. The spec is the source of truth and the editor is the only
thing that writes it, so rendering is a pure function of what is on disk: the
same thirteen task panels the editor shows, the same overall panel derived from
them, and the same Pareto-frontier connection.

Layout mirrors the editor's: the overall panel is a two-by-two tile in the
top-left of a six-by-three grid and the thirteen tasks fill the rest, so the
exported figure reads as the page it was edited on.

Usage:
  uv run python -m exp.robocasa365.render_rit_figure
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES_DIR = pathlib.Path(__file__).with_name("analysis") / "figures"
SCHEMA = "robocasa365.rit_figure/v1"

COLORS = ("#2E6FD9", "#C8641E")
INK, MUTED, GRID = "#1B2124", "#6B7780", "#DFE4E7"
XDOM = (0.0, 105.0)
#: The overall tile takes the top-left two-by-two block; the tasks fill the
#: rest of a six-by-three grid, row-major.
SLOTS = [(0, 2), (0, 3), (0, 4), (0, 5),
         (1, 2), (1, 3), (1, 4), (1, 5),
         (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]


def overall_of(point: dict, tasks: list[str]) -> tuple[float, float] | None:
    """Weighted-pooled x and macro y -- the same derivation the editor shows."""
    w = wx = 0.0
    ys = []
    for task in tasks:
        c = point["per_task"].get(task)
        if not c:
            continue
        w += c["w"]
        wx += c["w"] * c["x"]
        ys.append(c["y"])
    if not ys:
        return None
    return (wx / w if w else 0.0), sum(ys) / len(ys)


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated points, cheapest first: nothing else is both no dearer and no worse."""
    keep = [p for p in points
            if not any(q is not p and q[0] <= p[0] and q[1] >= p[1]
                       and (q[0] < p[0] or q[1] > p[1]) for q in points)]
    return sorted(keep)


def coords(spec: dict, task: str | None) -> list[list[tuple[float, float]]]:
    out = []
    for series in spec["series"]:
        pts = []
        for p in series["points"]:
            if task is None:
                v = overall_of(p, spec["tasks"])
            else:
                c = p["per_task"].get(task)
                v = (c["x"], c["y"]) if c else None
            if v:
                pts.append(v)
        out.append(pts)
    return out


def anchor_of(spec: dict, task: str | None) -> tuple[float, float] | None:
    if not spec.get("anchors"):
        return None
    a = spec["anchors"][0]
    if task is None:
        return overall_of(a, spec["tasks"])
    c = a["per_task"].get(task)
    return (c["x"], c["y"]) if c else None


def teacher_of(spec: dict, task: str | None) -> float | None:
    per = spec["teacher_only"]["per_task"]
    if task is not None:
        return per.get(task)
    vals = [per[t] for t in spec["tasks"] if t in per]
    return sum(vals) / len(vals) if vals else None


def ydomain(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    a, b = lo - 0.15, hi + 0.15
    if b - a < 0.3:
        c = (a + b) / 2
        a, b = c - 0.15, c + 0.15
    return max(0.0, a), min(1.0, b)


def draw(ax, spec: dict, task: str | None, *, big: bool) -> None:
    series = coords(spec, task)
    anchor = anchor_of(spec, task)
    teacher = teacher_of(spec, task)

    vals = [y for pts in series for _, y in pts]
    if anchor:
        vals.append(anchor[1])
    if teacher is not None:
        vals.append(teacher)
    lo, hi = ydomain(vals or [0.0, 1.0])

    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8 if big else 7, length=0)

    if teacher is not None:
        ax.axhline(teacher, color=INK, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
    for i, pts in enumerate(series):
        if not pts:
            continue
        front = pareto(pts)
        if len(front) > 1:
            ax.plot(*zip(*front), "-", color=COLORS[i % len(COLORS)],
                    linewidth=2.0 if big else 1.6, solid_joinstyle="round", zorder=3)
        ax.plot(*zip(*pts), linestyle="none", marker="o", color=COLORS[i % len(COLORS)],
                markersize=6 if big else 4.2, markeredgecolor="white",
                markeredgewidth=1.3 if big else 1.0, zorder=4)
    if anchor:
        ax.plot([anchor[0]], [anchor[1]], marker="D", color=INK,
                markersize=7 if big else 5, markeredgecolor="white",
                markeredgewidth=1.3, linestyle="none", zorder=5)

    ax.set_xlim(*XDOM)
    ax.set_ylim(lo, hi)
    title = f"all {len(spec['tasks'])} tasks  (macro over tasks)" if task is None else task
    ax.set_title(title, color=INK, fontsize=10.5 if big else 8.5, loc="left", pad=6)
    if big:
        ax.set_xlabel(spec["x_label"], color=MUTED, fontsize=9)
        ax.set_ylabel(spec["y_label"], color=MUTED, fontsize=9)


def render(spec: dict, stem: pathlib.Path) -> None:
    fig = plt.figure(figsize=(16.5, 7.6))
    gs = fig.add_gridspec(3, 6, hspace=0.42, wspace=0.30)
    draw(fig.add_subplot(gs[0:2, 0:2]), spec, None, big=True)
    for task, (r, c) in zip(spec["tasks"], SLOTS):
        draw(fig.add_subplot(gs[r, c]), spec, task, big=False)

    handles = [plt.Line2D([], [], color=COLORS[i % len(COLORS)], marker="o", linewidth=2.0,
                          markersize=6, markeredgecolor="white", markeredgewidth=1.3,
                          label=s["key"])
               for i, s in enumerate(spec["series"])]
    if spec.get("anchors"):
        handles.append(plt.Line2D([], [], color=INK, marker="D", linestyle="none",
                                  markersize=6.5, markeredgecolor="white",
                                  markeredgewidth=1.3, label=spec["anchors"][0]["label"]))
    handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (5, 3)), linewidth=1.2,
                              label=spec["teacher_only"]["label"]))
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, 0.02),
               frameon=False, fontsize=9.5, labelcolor=INK, ncol=2)
    fig.suptitle(spec["title"], color=INK, fontsize=13, x=0.006, ha="left", y=0.995)

    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def main() -> None:
    found = 0
    for path in sorted(FIGURES_DIR.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
            continue
        render(spec, path.with_suffix(""))
        found += 1
    if not found:
        raise SystemExit(f"no {SCHEMA} spec under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
