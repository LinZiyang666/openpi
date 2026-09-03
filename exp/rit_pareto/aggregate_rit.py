"""Aggregate RIT-Pareto rollouts and draw the four frontiers.

``aggregate``: from one run directory (``journal.jsonl`` + ``per_step.jsonl``
written by ``run_gtp``) to per-arm success rate and the three-tier inference
ratio. Episode acceptance follows ``analyze_gtp``: only terminal, accepted
journal rows count, and per-step rows are matched to the accepted attempt.
The cost of a decision is ``unit_cost(hit_type, start_t)`` from the single
cost authority (``exp.dispatch_surface.analysis.analytic_cost``), so

    IR% = 100 * sum(cost) / (n_decisions * unit_cost("MISS"))

is the same estimand ``rit_pl.predicted_ir`` addressed offline; the GST
inference_ratio of the GTP plots is its two-tier special case.

``plot``: one figure per suite. Every arm is a scatter point on the measured
IR axis (annotated with the addressed target); the line of each series is its
Pareto frontier -- the non-dominated arms (lower IR, higher success) sorted by
IR -- not a polyline through every arm. The GTP GST series of the same library
(``plot_data.json``) can be overlaid the same way for reference.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from exp.dispatch_surface.analysis.analytic_cost import VERDICTS, unit_cost
from exp.rit_pareto.rit_k import tier_cost

_MISS_MS = unit_cost("MISS", None)
_TARGET = re.compile(r"_ir(\d+)(?:p(\d+))?$")
_GST_CELL = re.compile(r"_gst_f(\d+)w(\d+)v(\d+)$")


def target_of(arm: str) -> float | None:
    m = _TARGET.search(arm)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}") if m.group(2) else float(m.group(1))


def gst_cell_of(arm: str) -> tuple[int, int, int] | None:
    m = _GST_CELL.search(arm)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def arm_label(arm: str) -> str | None:
    """Short annotation: the addressed IR of a RIT arm or the fh/w3/w5 shares of a GST cell."""
    t = target_of(arm)
    if t is not None:
        return f"IR={t:g}"
    c = gst_cell_of(arm)
    return f"{c[0]}/{c[1]}/{c[2]}" if c else None


def decision_cost(hit_type: str, start_t) -> float:
    """Per-decision cost; WARM_START at any canonical tier (K>2 ladders) via ``tier_cost``."""
    if hit_type == "WARM_START":
        return tier_cost(hit_type, start_t)
    return unit_cost(hit_type, start_t)


def aggregate(data_dir: pathlib.Path) -> dict:
    """Per-arm ``{n_ep, success_rate, decisions, counts, ir_percent, target_ir}``."""
    accepted_attempt: dict[str, int] = {}
    episodes: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    with (data_dir / "journal.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("status") not in ("done", "failed") or not row.get("accepted"):
                continue
            accepted_attempt[row["task_uid"]] = row["attempt"]
            episodes[row["yaml_id"]][row["task_uid"]] = row["status"] == "done"

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    cost: dict[str, float] = collections.defaultdict(float)
    with (data_dir / "per_step.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            hit_type = row.get("hit_type")
            if hit_type is None:
                continue
            if accepted_attempt.get(row["task_uid"]) != row.get("attempt"):
                continue
            if hit_type not in VERDICTS:
                raise SystemExit(f"unpriceable hit_type {hit_type!r} in {data_dir}")
            key = hit_type if hit_type != "WARM_START" else f"WARM_START@{float(row.get('start_t')):g}"
            counts[row["yaml_id"]][key] += 1
            cost[row["yaml_id"]] += decision_cost(hit_type, row.get("start_t"))

    out = {}
    for yaml_id in sorted(episodes):
        eps = episodes[yaml_id]
        n_dec = sum(counts[yaml_id].values())
        c = counts[yaml_id]
        warm_keys = sorted(k for k in c if k.startswith("WARM_START@"))
        out[yaml_id] = {
            "n_ep": len(eps),
            "success_rate": sum(eps.values()) / len(eps),
            "decisions": n_dec,
            "counts": {"FULL_HIT": c.get("FULL_HIT", 0),
                       "WARM_START": sum(c[k] for k in warm_keys),
                       "MISS": c.get("MISS", 0),
                       **{k: c[k] for k in warm_keys}},
            "ir_percent": 100.0 * cost[yaml_id] / (n_dec * _MISS_MS) if n_dec else None,
            "target_ir": target_of(yaml_id),
            "gst_cell": gst_cell_of(yaml_id),
            "label": arm_label(yaml_id),
        }
    return out


def _series(arms: dict) -> list[tuple[float, float, str | None, str]]:
    pts = [(a["ir_percent"], a["success_rate"], a.get("label") or (f"IR={a['target_ir']:g}" if a.get("target_ir") is not None else None), name)
           for name, a in arms.items() if a["ir_percent"] is not None]
    return sorted(pts, key=lambda p: p[0])


STYLES = {
    "no gate": {"color": "#0b7a75", "marker": "o"},
    "H gate": {"color": "#b8336a", "marker": "s"},
    "RIT-K3": {"color": "#0b7a75", "marker": "o"},
    "GST-K3": {"color": "#d1495b", "marker": "^"},
}


def pareto_front(points):
    """Non-dominated subset: minimise IR (first field), maximise success (second).

    Sorted by IR ascending with ties broken by higher success; a point survives
    only if its success strictly exceeds every cheaper survivor, so the result
    is the upper-left staircase and never a polyline through every arm.
    """
    front, best = [], -1.0
    for p in sorted(points, key=lambda q: (q[0], -q[1])):
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


def plot_suite(suite: str, series: dict[str, dict], out_dir: pathlib.Path,
               gst: list[tuple[float, float, str]] | None = None, *,
               reference: dict[str, dict] | None = None, stem: str = "pareto_rit",
               title: str | None = None, series_prefix: str = "RIT-PL ") -> list[pathlib.Path]:
    """One figure: every arm of every series as a scatter point, each series' Pareto
    frontier as a line; ``reference`` series are drawn as dashed frontiers only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, arms in series.items():
        pts = _series(arms)
        if not pts:
            continue
        st = STYLES.get(label, {"color": "gray", "marker": "^"})
        # Scatter layer: every arm. Line layer: the Pareto frontier only.
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=st["color"], marker=st["marker"],
                   s=34, alpha=0.85, zorder=3, label=f"{series_prefix}{label}: arms ({len(pts)} x 500 ep)")
        front = pareto_front(pts)
        ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=st["color"], lw=2.4,
                zorder=2, label=f"{series_prefix}{label}: Pareto frontier ({len(front)} non-dominated)")
        for x, y, t, _ in pts:
            if t is not None:
                ax.annotate(t, xy=(x, y), xytext=(4, 4), textcoords="offset points",
                            fontsize=6 if len(pts) > 20 else 7, color=st["color"])
    for label, arms in (reference or {}).items():
        pts = _series(arms)
        if not pts:
            continue
        front = pareto_front(pts)
        ax.plot([p[0] for p in front], [p[1] for p in front], ":", color="#4c4c9d", lw=1.6,
                zorder=1, label=f"{label} (frontier, reference)")
    if gst:
        gpts = sorted(gst)
        gfront = pareto_front(gpts)
        ax.scatter([p[0] for p in gpts], [p[1] for p in gpts], color="0.45", marker="x", s=28,
                   zorder=3, label="GST hysteresis-gate sweep, same library (GTP): arms")
        ax.plot([p[0] for p in gfront], [p[1] for p in gfront], "--", color="0.45", lw=1.4,
                zorder=2, label="GST: Pareto frontier (reference)")
    ax.set_xlabel("measured inference ratio (% of always-full GPU cost, three-tier)")
    ax.set_ylabel("success rate")
    ax.set_title(title or f"{suite}: RIT-PL frontiers on the official pruned-500 pool")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}_{suite}.{ext}"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def gst_series(plot_data: pathlib.Path, suite: str, lib: str = "ws") -> list[tuple[float, float, str]]:
    """GTP GST points of ``lib`` on the IR axis (two-tier special case)."""
    from exp.gate_threshold_pareto.analyze_gtp import SUITE_TAG, inference_ratio

    data = json.loads(plot_data.read_text())
    prefix = f"gtp_{lib}_{SUITE_TAG[suite]}_fh"
    out = []
    for arm, rec in data["suites"][suite].items():
        if arm.startswith(prefix) and rec.get("teacher_ratio") is not None:
            out.append((100.0 * inference_ratio(rec["teacher_ratio"]), rec["success_rate"], arm))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--data-dir", required=True)
    a.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--suite", required=True)
    p.add_argument("--nogate", required=True, help="aggregate json of the no-gate layer")
    p.add_argument("--hgate", default="", help="aggregate json of the H-gate layer")
    p.add_argument("--gst-plot-data", default="", help="GTP plot_data.json for the GST reference series")
    p.add_argument("--out-dir", required=True)
    k = sub.add_parser("plot-k3", help="RIT-K3 vs GST-K3 with the K=2 no-gate RIT frontier as reference")
    k.add_argument("--suite", required=True)
    k.add_argument("--rit", required=True, help="aggregate json of the K3 RIT group")
    k.add_argument("--gst", default="", help="aggregate json of the K3 GST group (optional until that group has run)")
    k.add_argument("--k2-nogate", default="", help="aggregate json of the K=2 no-gate RIT group (reference)")
    k.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "aggregate":
        res = aggregate(pathlib.Path(args.data_dir))
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        for arm, r in res.items():
            print(f"{arm}: n={r['n_ep']} SR={r['success_rate']:.3f} IR={r['ir_percent']:.2f}% "
                  f"counts={r['counts']}")
    elif args.cmd == "plot-k3":
        series = {"RIT-K3": json.loads(pathlib.Path(args.rit).read_text())}
        if args.gst:
            series["GST-K3"] = json.loads(pathlib.Path(args.gst).read_text())
        ref = {"RIT-PL K=2 no gate": json.loads(pathlib.Path(args.k2_nogate).read_text())} if args.k2_nogate else None
        for path in plot_suite(args.suite, series, pathlib.Path(args.out_dir), None, reference=ref,
                               stem="pareto_k3", series_prefix="",
                               title=f"{args.suite}: K=3 ladder (FULL / WARM@0.3 / WARM@0.5), no gate, pruned-500 pool"):
            print(path)
    else:
        series = {"no gate": json.loads(pathlib.Path(args.nogate).read_text())}
        if args.hgate:
            series["H gate"] = json.loads(pathlib.Path(args.hgate).read_text())
        gst = gst_series(pathlib.Path(args.gst_plot_data), args.suite) if args.gst_plot_data else None
        for path in plot_suite(args.suite, series, pathlib.Path(args.out_dir), gst):
            print(path)


if __name__ == "__main__":
    main()
