"""Draw the RIT frontier: one overall panel and one small multiple per task.

x is the realized inference ratio (percent of always-MISS cost), y is success
rate. One line per ladder depth; the all-FULL_HIT arm and the teacher-only arm are
drawn as reference marks rather than points on a frontier, because neither is
addressed by an IR target.

Usage:
  uv run python -m exp.robocasa365.plot_rit_pareto \\
      --frontier <frontier.json> --floor <teacher_floor.json> --out-stem <path>
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Two ladder depths -> two categorical hues, assigned in fixed order and never
# cycled; the reference marks wear ink tokens, not a third series colour.
SERIES = {2: ("#2E6FD9", "k=2  FULL + WARM@1 step"),
          3: ("#C8641E", "k=3  FULL + WARM@1 + WARM@2 steps")}
INK = "#1B2124"
MUTED = "#6B7780"
GRID = "#DFE4E7"


def curves(points: dict, key_ir: str, key_y: str, task: str | None = None,
           require_tasks: int | None = None):
    """Points of each ladder depth, cheapest first.

    ``require_tasks`` drops a cell whose task coverage is short of the full
    roster. A partially dispatched cell has a macro over whichever tasks the
    scheduler reached first, which is a different quantity from the macro every
    other point carries -- plotting the two together reads as a trend and is a
    coverage artefact.
    """
    out = {}
    for cid, pt in points.items():
        if "missing" in pt or pt.get("k") not in SERIES:
            continue
        if require_tasks is not None and len(pt.get("per_task", {})) < require_tasks:
            continue
        if task is None:
            ir, y = pt.get(key_ir), pt.get(key_y)
        else:
            sub = pt.get("per_task", {}).get(task)
            if not sub:
                continue
            ir, y = sub.get("realized_ir"), sub.get("sr")
        if ir is None or y is None:
            continue
        out.setdefault(pt["k"], []).append((ir, y))
    return {k: sorted(v) for k, v in out.items()}


def anchors(points: dict, task: str | None = None):
    """The all-FULL_HIT arm, drawn as a reference mark."""
    pt = points.get("always_hit")
    if not pt or "missing" in pt:
        return None
    if task is None:
        return pt.get("realized_ir"), pt.get("macro_sr")
    sub = pt.get("per_task", {}).get(task)
    return (sub.get("realized_ir"), sub.get("sr")) if sub else None


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def draw(ax, data, anchor, floor, title, ylab=True):
    for k, (color, _) in SERIES.items():
        pts = data.get(k)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, linewidth=2.0, markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    if floor is not None:
        ax.axhline(floor, color=INK, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
    if anchor and anchor[0] is not None and anchor[1] is not None:
        ax.plot([anchor[0]], [anchor[1]], marker="D", color=INK, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("realized inference ratio  (% of all-MISS)", color=MUTED, fontsize=9)
    if ylab:
        ax.set_ylabel("success rate", color=MUTED, fontsize=9)
    ax.set_xlim(0, 105)
    style(ax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--floor", default="")
    ap.add_argument("--title", default="RoboCasa365 x GR00T N1.5 -- warm-start RIT frontier")
    ap.add_argument("--out-stem", required=True)
    ap.add_argument("--require-tasks", type=int, default=None,
                    help="drop cells covering fewer tasks than this (partial runs)")
    args = ap.parse_args()

    fr = json.loads(pathlib.Path(args.frontier).read_text())
    points = fr["points"]
    floor = None
    floor_tasks = {}
    if args.floor:
        fl = json.loads(pathlib.Path(args.floor).read_text())
        floor = fl["macro_sr"]
        floor_tasks = {t: v["sr"] for t, v in fl["tasks"].items()}

    tasks = sorted({t for pt in points.values() if "per_task" in pt for t in pt["per_task"]})

    fig = plt.figure(figsize=(13.5, 4.6 + 3.0 * ((len(tasks) + 3) // 4)))
    gs = fig.add_gridspec((len(tasks) + 3) // 4 + 1, 4, hspace=0.55, wspace=0.28,
                          height_ratios=[1.5] + [1] * ((len(tasks) + 3) // 4))

    ax0 = fig.add_subplot(gs[0, :2])
    n_tasks = max((len(pt.get("per_task", {})) for pt in points.values()), default=0)
    draw(ax0, curves(points, "realized_ir", "macro_sr", require_tasks=args.require_tasks),
         anchors(points), floor, f"all {n_tasks} tasks  (macro over tasks)")

    handles = [plt.Line2D([], [], color=c, marker="o", linewidth=2.0, markersize=6,
                          markeredgecolor="white", markeredgewidth=1.5, label=lab)
               for c, lab in SERIES.values()]
    handles.append(plt.Line2D([], [], color=INK, marker="D", linestyle="none",
                              markersize=7, markeredgecolor="white",
                              markeredgewidth=1.5, label="all-FULL_HIT arm"))
    if floor is not None:
        handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (5, 3)),
                                  linewidth=1.4, label="teacher-only (no cache)"))
    lg = fig.add_subplot(gs[0, 2:])
    lg.axis("off")
    lg.legend(handles=handles, loc="center left", frameon=False, fontsize=10,
              labelcolor=INK)

    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        draw(ax, curves(points, "realized_ir", "sr", task), anchors(points, task),
             floor_tasks.get(task), task, ylab=(i % 4 == 0))

    fig.suptitle(args.title, color=INK, fontsize=14, x=0.008, ha="left", y=0.995)
    stem = pathlib.Path(args.out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {stem}.png / .pdf  ({len(tasks)} task panels)")


if __name__ == "__main__":
    main()
