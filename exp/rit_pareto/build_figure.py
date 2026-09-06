"""Build the figure spec of a RIT-Pareto frontier plot from the run aggregates.

Stage one of the three-stage figure chain::

    build_figure   aggregate.json (+ GTP plot_data.json)  ->  <figure_id>.json
    render_figure  <figure_id>.json                       ->  <figure_id>.{png,pdf}
    edit_figure    <figure_id>.json                       ->  browser editor, writes both

Everything that knows what the experiment *means* lives in this module: which
run aggregate feeds which series, how an arm name becomes an annotation, which
cost authority prices the x axis, how the series are styled and named. The spec
it writes is self-contained -- neither the renderer nor the editor ever reads an
aggregate again -- so an arm re-run or added later is picked up by rebuilding the
spec, and a hand-placed point survives every later re-render.

A point carries only what the figure needs to draw it: position, annotation, and
the episode count behind its success rate (the editor snaps y to multiples of
1/n_ep).

Subcommands ``rit`` (K=2 layers + GTP GST reference) and ``k3`` (K=3 ladder, RIT
vs GST, optionally the GST cells re-run behind the H gate) reproduce the four
published figures; ``k3-gsth`` / ``k3-rith`` write the gated K=3 re-runs as their own
specs, each referencing the matching no-gate frontier of ``pareto_k3_<suite>``. The
same gated cells can be overlaid on the K=2 figure too; there they are a reference
into the K=3 spec, which owns them.

A series may name another figure instead of carrying points: the K=3 figures draw
the K=2 no-gate frontier by pointing at ``pareto_rit_<suite>``, so that line is
resolved from the K=2 spec at render time and cannot drift away from the figure
that owns it -- editing the K=2 points moves the K=3 reference with them.

"""

from __future__ import annotations

import argparse
import json
import pathlib

from exp.rit_pareto.aggregate_rit import arm_label

SCHEMA = "rit_pareto.figure/v1"

X_LABEL = "measured inference ratio (% of always-full GPU cost, three-tier)"
Y_LABEL = "success rate"

# ------------------------------------------------------------------
# Series styling
# ------------------------------------------------------------------

#: Scatter + solid-frontier look shared by every measured series.
_PRIMARY = {"marker_size": 34, "alpha": 0.85, "point_zorder": 3,
            "linestyle": "-", "linewidth": 2.4, "line_zorder": 2}

STYLES = {
    "no gate": {"color": "#0b7a75", "marker": "o", **_PRIMARY},
    "H gate": {"color": "#b8336a", "marker": "s", **_PRIMARY},
    "RIT-K3": {"color": "#0b7a75", "marker": "o", **_PRIMARY},
    "GST-K3": {"color": "#d1495b", "marker": "^", **_PRIMARY},
    "GST-K3 + H gate": {"color": "#6a3d9a", "marker": "D", **_PRIMARY},
    "RIT-K3 + H gate": {"color": "#d4780f", "marker": "P", **_PRIMARY},
}
_FALLBACK_STYLE = {"color": "gray", "marker": "^", **_PRIMARY}

#: The GTP GST sweep of the same library: grey crosses, dashed frontier.
_GTP_STYLE = {"color": "0.45", "marker": "x", "marker_size": 28, "alpha": 1.0, "point_zorder": 3,
              "linestyle": "--", "linewidth": 1.4, "line_zorder": 2}

#: A reference series contributes its frontier only, dotted and behind everything.
_REFERENCE_STYLE = {"color": "#4c4c9d", "marker": "o", "marker_size": 20, "alpha": 1.0, "point_zorder": 1,
                    "linestyle": ":", "linewidth": 1.6, "line_zorder": 1}

#: The GST-K3 cells re-run behind the K=2 hysteresis gate, overlaid on the K=3 figure.
_GSTH_STYLE = {"color": "#6a3d9a", "marker": "D", "marker_size": 34, "alpha": 0.85, "point_zorder": 3,
               "linestyle": "-", "linewidth": 2.4, "line_zorder": 2}


# ------------------------------------------------------------------
# Series assembly
# ------------------------------------------------------------------

def _points(arms: dict) -> list[dict]:
    """One record per priceable arm, sorted by measured IR."""
    pts = []
    for name, rec in arms.items():
        x = rec.get("ir_percent")
        if x is None:
            continue
        y = rec["success_rate"]
        pts.append({
            "id": name,
            "x": float(x), "y": float(y),
            "label": rec.get("label") or arm_label(name),
            "label_offset": [4, 4],
            # A success rate is a count over n_ep episodes; the editor keeps y on that grid.
            "n_ep": rec.get("n_ep"),
        })
    return sorted(pts, key=lambda p: p["x"])


def _episode_tag(points: list[dict]) -> str:
    """`` x 500 ep`` when every arm ran the same episode count, else empty."""
    counts = {p.get("n_ep") for p in points}
    return f" x {counts.pop()} ep" if len(counts) == 1 and None not in counts else ""


def series_from_arms(key: str, arms: dict, *, style: dict | None = None, prefix: str = "",
                     annotate: bool = True, show_points: bool = True, show_frontier: bool = True,
                     scatter_label: str | None = None, frontier_label: str | None = None) -> dict:
    """A drawable series: the arms as points plus the labels and style the renderer needs.

    ``{n_arms}`` / ``{n_front}`` in a legend label are filled in at render time,
    so the counts stay correct after the editor adds, hides or moves a point.
    """
    pts = _points(arms)
    st = dict(style if style is not None else STYLES.get(key, _FALLBACK_STYLE))
    st["annotate_fontsize"] = 6 if len(pts) > 20 else 7
    return {
        "key": key,
        "style": st,
        "show_points": show_points,
        "show_frontier": show_frontier,
        "annotate": annotate,
        "scatter_label": scatter_label if scatter_label is not None
        else f"{prefix}{key}: arms ({{n_arms}}{_episode_tag(pts)})",
        "frontier_label": frontier_label if frontier_label is not None
        else f"{prefix}{key}: Pareto frontier ({{n_front}} non-dominated)",
        "points": pts,
    }


def reference_series(key: str, source_figure: str, source_series: str, *,
                     style: dict | None = None, show_points: bool = False, annotate: bool = False,
                     annotate_fontsize: int = 7, scatter_label: str | None = None,
                     frontier_label: str | None = None) -> dict:
    """A series whose points live in another figure's spec and are read from there.

    It stores the pointer only. The renderer and the editor resolve it against the
    sibling spec, so the two figures cannot disagree about the same measurements.
    By default only the frontier is drawn; ``show_points`` / ``annotate`` overlay
    the points themselves.
    """
    st = dict(style if style is not None else _REFERENCE_STYLE)
    st["annotate_fontsize"] = annotate_fontsize
    return {
        "key": key,
        "style": st,
        "show_points": show_points,
        "show_frontier": True,
        "annotate": annotate,
        "source_figure": source_figure,
        "source_series": source_series,
        "scatter_label": scatter_label if scatter_label is not None else f"{key}: arms ({{n_arms}})",
        "frontier_label": frontier_label if frontier_label is not None else f"{key} (frontier, reference)",
    }


def overlay_series(source_figure: str, figures_dir: pathlib.Path) -> list[dict]:
    """Reference every series another figure owns, keeping its look and legend text.

    Used to lay a whole experiment's lines over an existing figure without copying
    the points: they stay in ``<source_figure>.json``, which the renderer reads and
    the editor writes back to.
    """
    owner = json.loads((figures_dir / f"{source_figure}.json").read_text())
    out = []
    for series in owner["series"]:
        if "source_figure" in series:
            continue  # do not chain references
        out.append({
            "key": series["key"],
            "style": dict(series["style"]),
            "show_points": series.get("show_points", True),
            "show_frontier": series.get("show_frontier", True),
            "annotate": series.get("annotate", False),
            "source_figure": source_figure,
            "source_series": series["key"],
            "scatter_label": series["scatter_label"],
            "frontier_label": series["frontier_label"],
        })
    return out


def gsth_overlay(suite: str) -> dict:
    """The GST-K3 + H gate cells of ``suite``, read from the K=3 figure that owns them."""
    key = "GST-K3 + H gate"
    return reference_series(key, f"pareto_k3_{suite}", key, style=_GSTH_STYLE, show_points=True,
                            annotate=True, annotate_fontsize=6,
                            scatter_label=f"{key}: arms ({{n_arms}} x 500 ep)",
                            frontier_label=f"{key}: Pareto frontier ({{n_front}} non-dominated)")


def gtp_gst_arms(plot_data: pathlib.Path, suite: str, lib: str = "ws") -> dict:
    """GTP GST arms of ``lib`` mapped onto the IR axis (the two-tier special case)."""
    from exp.gate_threshold_pareto.analyze_gtp import SUITE_TAG, inference_ratio

    data = json.loads(plot_data.read_text())
    prefix = f"gtp_{lib}_{SUITE_TAG[suite]}_fh"
    out = {}
    for arm, rec in data["suites"][suite].items():
        if arm.startswith(prefix) and rec.get("teacher_ratio") is not None:
            out[arm] = {"ir_percent": 100.0 * inference_ratio(rec["teacher_ratio"]),
                        "success_rate": rec["success_rate"], "label": None,
                        "n_ep": rec.get("n_ep")}
    return out


def _spec(figure_id: str, title: str, series: list[dict]) -> dict:
    return {
        "schema": SCHEMA,
        "figure_id": figure_id,
        "title": title,
        "x_label": X_LABEL,
        "y_label": Y_LABEL,
        "figsize": [8, 5.5],
        "dpi": 160,
        "grid_alpha": 0.3,
        "legend": {"fontsize": 7, "loc": "lower right"},
        "series": series,
    }


# ------------------------------------------------------------------
# The two published figures
# ------------------------------------------------------------------

def build_rit(suite: str, nogate: pathlib.Path, hgate: pathlib.Path | None = None,
              gst_plot_data: pathlib.Path | None = None, *, gsth: bool = False,
              acb: pathlib.Path | None = None) -> dict:
    """K=2 figure: the no-gate and H-gate RIT-PL layers over the GTP GST reference.

    With ``gsth`` the GST-K3 + H gate cells are overlaid as a reference into
    ``pareto_k3_<suite>``, which must sit in the same directory when rendering.
    """
    series = [series_from_arms("no gate", json.loads(nogate.read_text()), prefix="RIT-PL ")]
    if hgate is not None:
        series.append(series_from_arms("H gate", json.loads(hgate.read_text()), prefix="RIT-PL "))
    if gst_plot_data is not None:
        series.append(series_from_arms(
            "GST (GTP)", gtp_gst_arms(gst_plot_data, suite), style=_GTP_STYLE, annotate=False,
            scatter_label="GST hysteresis-gate sweep, same library (GTP): arms",
            frontier_label="GST: Pareto frontier (reference)"))
    if gsth:
        series.append(gsth_overlay(suite))
    if acb is not None:
        series.extend(overlay_series(f"pareto_acb_{suite}", acb))
    return _spec(f"pareto_rit_{suite}",
                 f"{suite}: RIT-PL frontiers on the official pruned-500 pool", series)


def build_k3(suite: str, rit: pathlib.Path, gst: pathlib.Path | None = None,
             gsth: pathlib.Path | None = None, *, acb: pathlib.Path | None = None) -> dict:
    """K=3 figure: RIT-K3 against GST-K3 (and, with ``gsth``, the same GST cells behind
    the H gate), over the K=2 no-gate frontier of the same suite.

    The K=2 line is not copied here: it points at ``pareto_rit_<suite>``, which must
    sit in the same directory when this figure is rendered.
    """
    series = [series_from_arms("RIT-K3", json.loads(rit.read_text()))]
    if gst is not None:
        series.append(series_from_arms("GST-K3", json.loads(gst.read_text())))
    if gsth is not None:
        series.append(series_from_arms("GST-K3 + H gate", json.loads(gsth.read_text()), style=_GSTH_STYLE))
    series.append(reference_series("RIT-PL K=2 no gate", f"pareto_rit_{suite}", "no gate"))
    if acb is not None:
        series.extend(overlay_series(f"pareto_acb_{suite}", acb))
    return _spec(f"pareto_k3_{suite}",
                 f"{suite}: K=3 ladder (FULL / WARM@0.3 / WARM@0.5), no gate, pruned-500 pool", series)


def build_k3_gsth(suite: str, gsth: pathlib.Path) -> dict:
    """K=3 GST cells behind the production H gate (the ``gsth`` groups of 2026-09-03/04).

    Same 34 cut triples as the no-gate GST-K3 arms, with the K=2 score-hysteresis
    gate in front of the judge. The no-gate GST-K3 frontier is drawn as a pointer
    into ``pareto_k3_<suite>`` (series ``GST-K3``), not copied, so the two figures
    keep sharing one set of no-gate measurements.
    """
    series = [series_from_arms("GST-K3 + H gate", json.loads(gsth.read_text())),
              reference_series("GST-K3 no gate", f"pareto_k3_{suite}", "GST-K3")]
    return _spec(f"pareto_k3_gsth_{suite}",
                 f"{suite}: K=3 GST cells + H gate (score_hysteresis, K=2 theta), pruned-500 pool", series)


def build_k3_rith(suite: str, rith: pathlib.Path) -> dict:
    """K=3 RIT arms behind the production H gate (the ``rith`` groups of 2026-09-05).

    Same 16 IR-addressed cut triples as the no-gate RIT-K3 arms, with the K=2
    score-hysteresis gate in front of the judge. The no-gate RIT-K3 frontier is drawn
    as a pointer into ``pareto_k3_<suite>`` (series ``RIT-K3``), not copied, so the two
    figures keep sharing one set of no-gate measurements.
    """
    series = [series_from_arms("RIT-K3 + H gate", json.loads(rith.read_text())),
              reference_series("RIT-K3 no gate", f"pareto_k3_{suite}", "RIT-K3")]
    return _spec(f"pareto_k3_rith_{suite}",
                 f"{suite}: K=3 RIT arms + H gate (score_hysteresis, K=2 theta), pruned-500 pool", series)


def write_spec(spec: dict, out_dir: pathlib.Path) -> pathlib.Path:
    """Write ``<out_dir>/<figure_id>.json`` -- the path the renderer and editor take."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec['figure_id']}.json"
    path.write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _opt(value: str) -> pathlib.Path | None:
    return pathlib.Path(value) if value else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rit", help="K=2 layers (no gate / H gate) + GTP GST reference")
    r.add_argument("--suite", required=True)
    r.add_argument("--nogate", required=True, help="aggregate json of the no-gate layer")
    r.add_argument("--hgate", default="", help="aggregate json of the H-gate layer")
    r.add_argument("--gst-plot-data", default="", help="GTP plot_data.json for the GST reference series")
    r.add_argument("--gsth", action="store_true",
                   help="overlay the GST-K3 + H gate cells, read from pareto_k3_<suite>.json")
    r.add_argument("--acb", action="store_true",
                   help="overlay the ActionCache baseline lines, read from pareto_acb_<suite>.json")
    r.add_argument("--out-dir", required=True)
    k = sub.add_parser("k3", help="K=3 ladder: RIT vs GST, K=2 no-gate frontier as reference")
    k.add_argument("--suite", required=True)
    k.add_argument("--rit", required=True, help="aggregate json of the K3 RIT group")
    k.add_argument("--gst", default="", help="aggregate json of the K3 GST group")
    k.add_argument("--gsth", default="", help="aggregate json of the K3 GST + H gate group (overlaid)")
    k.add_argument("--acb", action="store_true",
                   help="overlay the ActionCache baseline lines, read from pareto_acb_<suite>.json")
    k.add_argument("--out-dir", required=True)
    h = sub.add_parser("k3-gsth", help="K=3 GST cells + H gate; no-gate GST-K3 frontier of pareto_k3_<suite> as reference")
    h.add_argument("--suite", required=True)
    h.add_argument("--gsth", required=True, help="aggregate json of the K3 GST + H-gate group")
    h.add_argument("--out-dir", required=True)
    t = sub.add_parser("k3-rith", help="K=3 RIT arms + H gate; no-gate RIT-K3 frontier of pareto_k3_<suite> as reference")
    t.add_argument("--suite", required=True)
    t.add_argument("--rith", required=True, help="aggregate json of the K3 RIT + H-gate group")
    t.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "rit":
        out_dir = pathlib.Path(args.out_dir)
        spec = build_rit(args.suite, pathlib.Path(args.nogate), _opt(args.hgate), _opt(args.gst_plot_data),
                         gsth=args.gsth, acb=out_dir if args.acb else None)
    elif args.cmd == "k3-gsth":
        spec = build_k3_gsth(args.suite, pathlib.Path(args.gsth))
    elif args.cmd == "k3-rith":
        spec = build_k3_rith(args.suite, pathlib.Path(args.rith))
    else:
        spec = build_k3(args.suite, pathlib.Path(args.rit), _opt(args.gst), _opt(args.gsth),
                        acb=pathlib.Path(args.out_dir) if args.acb else None)
    print(write_spec(spec, pathlib.Path(args.out_dir)))


if __name__ == "__main__":
    main()
