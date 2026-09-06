"""Render a figure spec to PNG and PDF -- stage two of the figure chain.

This module is deliberately ignorant of the experiment: it knows points, styles
and legend text, never an arm name or a cost model. Its only derived quantity is
the Pareto frontier, which is recomputed from the points on every render rather
than stored, so a point moved in the editor moves the frontier with it.

A series carrying ``source_figure`` owns no points of its own: they are read from
that sibling spec at render time (the K=3 figures reference the K=2 no-gate series
this way), which keeps one set of measurements behind every line that shows them.

Input is a ``rit_pareto.figure/v1`` spec written by ``build_figure``; output is
``<out_dir>/<figure_id>.{png,pdf}``, defaulting to the spec's own directory so a
re-render overwrites the published figures in place.

A spec that has no figures on disk yet is refused unless ``--new`` is passed: some
specs only hold points for other figures to reference, and rendering a whole
directory must not turn those into published figures by accident.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

SCHEMA = "rit_pareto.figure/v1"


def load_spec(path: pathlib.Path) -> dict:
    """Read a figure spec and reject anything the renderer does not understand."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {spec.get('schema')!r}, expected {SCHEMA!r}")
    return spec


def pareto_front(points: list[dict]) -> list[dict]:
    """Non-dominated subset of ``{"x", "y"}`` records: minimise x, maximise y.

    Sorted by x ascending with ties broken by higher y; a point survives only if
    its y strictly exceeds every cheaper survivor, so the result is the upper-left
    staircase and never a polyline through every point.
    """
    front: list[dict] = []
    best = -math.inf
    for p in sorted(points, key=lambda q: (q["x"], -q["y"])):
        if p["y"] > best:
            front.append(p)
            best = p["y"]
    return front


def series_points(series: dict, base_dir: pathlib.Path, warnings: list[str] | None = None) -> list[dict]:
    """Points of a series: its own, or the referenced figure's if it names one.

    A missing or unusable reference yields no points and a warning rather than an
    error -- the line drops out of the figure, which is visible, instead of taking
    the whole render down.
    """
    if "source_figure" not in series:
        return series.get("points", [])
    path = base_dir / f"{series['source_figure']}.json"
    key = series.get("source_series")
    note = None
    if not path.is_file():
        note = f"{series['key']}: {path.name} not found, reference line skipped"
    else:
        other = json.loads(path.read_text(encoding="utf-8"))
        match = next((s for s in other.get("series", []) if s.get("key") == key), None)
        if match is None:
            note = f"{series['key']}: {path.name} has no series {key!r}, reference line skipped"
        elif "source_figure" in match:
            note = f"{series['key']}: {path.name} series {key!r} is itself a reference, not followed"
        else:
            return match.get("points", [])
    if warnings is not None:
        warnings.append(note)
    return []


def _fill(template: str, **counts: int) -> str:
    """Legend text with ``{n_arms}`` / ``{n_front}`` resolved against the drawn points."""
    try:
        return template.format(**counts)
    except (KeyError, IndexError, ValueError):
        return template


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

def render(spec: dict, out_dir: pathlib.Path, *, formats: tuple[str, ...] = ("png", "pdf"),
           base_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Draw the spec and write one file per format; returns the paths written.

    ``base_dir`` is where referenced figures are looked up; it defaults to the
    output directory, which is where a spec and its siblings normally live.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings: list[str] = []
    fig, ax = plt.subplots(figsize=tuple(spec.get("figsize", (8, 5.5))))
    for series in spec["series"]:
        if series.get("hidden"):
            continue
        pts = [p for p in series_points(series, base_dir or out_dir, warnings) if not p.get("hidden")]
        if not pts:
            continue
        st = series["style"]
        front = pareto_front(pts)
        counts = {"n_arms": len(pts), "n_front": len(front)}
        if series.get("show_points", True):
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], color=st["color"], marker=st["marker"],
                       s=st.get("marker_size", 34), alpha=st.get("alpha", 1.0),
                       zorder=st.get("point_zorder", 3), label=_fill(series["scatter_label"], **counts))
        if series.get("show_frontier", True):
            ax.plot([p["x"] for p in front], [p["y"] for p in front], linestyle=st.get("linestyle", "-"),
                    color=st["color"], lw=st.get("linewidth", 2.0), zorder=st.get("line_zorder", 2),
                    label=_fill(series["frontier_label"], **counts))
        if series.get("annotate", False):
            for p in pts:
                if p.get("label"):
                    ax.annotate(p["label"], xy=(p["x"], p["y"]), xytext=tuple(p.get("label_offset", (4, 4))),
                                textcoords="offset points", fontsize=st.get("annotate_fontsize", 7),
                                color=st["color"])
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["y_label"])
    ax.set_title(spec["title"])
    ax.grid(alpha=spec.get("grid_alpha", 0.3))
    legend = spec.get("legend", {})
    ax.legend(fontsize=legend.get("fontsize", 7), loc=legend.get("loc", "lower right"))
    for note in warnings:
        print(f"warning: {note}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out_dir / f"{spec['figure_id']}.{ext}"
        fig.savefig(path, dpi=spec.get("dpi", 160), bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def render_file(spec_path: pathlib.Path, out_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Render a spec file next to itself unless another directory is given."""
    spec = load_spec(spec_path)
    return render(spec, out_dir if out_dir is not None else spec_path.parent, base_dir=spec_path.parent)


def is_published(spec_path: pathlib.Path, out_dir: pathlib.Path) -> bool:
    """Whether this spec already has figures on disk, i.e. re-rendering only refreshes."""
    return any((out_dir / f"{spec_path.stem}.{ext}").is_file() for ext in ("png", "pdf"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", required=True, help="figure spec json written by build_figure")
    ap.add_argument("--out-dir", default="", help="defaults to the directory holding the spec")
    ap.add_argument("--new", action="store_true",
                    help="allow creating figures for a spec that has none yet")
    args = ap.parse_args()
    spec_path = pathlib.Path(args.figure)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else spec_path.parent
    if not args.new and not is_published(spec_path, out_dir):
        raise SystemExit(f"{spec_path.stem}: no figure on disk; it is a data source for other "
                         f"figures. Pass --new to publish one.")
    for path in render_file(spec_path, out_dir):
        print(path)


if __name__ == "__main__":
    main()
