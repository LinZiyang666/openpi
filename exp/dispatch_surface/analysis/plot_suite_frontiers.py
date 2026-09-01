"""Suite-level Pareto frontier figure from ``sgrid_sweep summarize`` outputs alone.

The l10 dense figure (plot_budget_amendment) requires the confirmation outcome
design; the spatial chain (goals 2-4) and the eval500 line produce only
summarize outputs. This script draws the same picture -- per-family staircase
frontier (solid), two-arm mixture envelope (dotted), anchor star, percent-cost
axis -- directly from up to three summaries: the RIT (s0/sv) density sweep, the
GST threshold grid, and the gated RIT (s0) system points.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp.dispatch_surface.analysis.frontier_hull import upper_concave_hull  # noqa: E402
from exp.dispatch_surface.analysis.plot_budget_amendment import (  # noqa: E402
    FAMILY_COLOR,
    FAMILY_LABEL,
    FAMILY_MARKER,
    SYSGATE_COLOR,
    _pareto_staircase,
    _short,
)

_QUANT = re.compile(r"^dsp_s0_p(\d+)(?:wp(\d+))?$")
_PL = re.compile(r"^dsp_s0_pl_(?:ir([0-9]+(?:p[0-9]+)?)|p([0-9]+))$")


def _label(arm: str) -> str:
    """Self-explanatory arm label.

    ``dsp_s0_p9362``       -> ``q.9362``        (one delta drives both cuts)
    ``dsp_s0_p9362wp9542`` -> ``q.9362→.9542``  (full cut from the first
                              quantile, warm cut from the second: a decoupled
                              cut vector, not a point on the single-delta ladder)
    """
    pl = _PL.match(arm)
    if pl:
        # RIT-PL: ir82p5 -> IR82.5 (budget-addressed), p925 -> q.925 (delta-addressed)
        return f"IR{pl.group(1).replace('p', '.')}" if pl.group(1) else f"q.{pl.group(2)}"
    m = _QUANT.match(arm)
    if not m:
        return _short(arm)
    full = f"q.{m.group(1)}"
    return full if not m.group(2) else f"{full}→.{m.group(2)}"


def _load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def _series(ax, arms_dict, color, marker, label, pct, style="-", offset=(3, 3), z=4):
    arms = sorted(arms_dict)
    pts = [(arms_dict[a]["cost"], arms_dict[a]["sr"]) for a in arms]
    front = _pareto_staircase(pts)
    ax.plot([pct(c) for c, _ in front], [s for _, s in front], style, color=color, lw=2.4, zorder=z + 1,
            label=f"{label} — Pareto frontier ({len(front)} / {len(pts)} arms)")
    if len(pts) >= 2:
        hull = upper_concave_hull(pts)
        ax.plot([pct(c) for c, _ in hull], [s for _, s in hull], ":", color=color, lw=1.1, alpha=0.55, zorder=2)
    on = {(round(c, 9), round(s, 9)) for c, s in front}
    face = [color if (round(c, 9), round(s, 9)) in on else "white" for c, s in pts]
    ax.scatter([pct(c) for c, _ in pts], [s for _, s in pts], marker=marker, s=50,
               facecolors=face, edgecolors=color, linewidths=1.2, zorder=z + 2)
    for a, (c, s) in zip(arms, pts):
        ax.annotate(_label(a), (pct(c), s), xytext=offset, textcoords="offset points",
                    fontsize=6.5, color=color, alpha=0.9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sgrid-summary", required=True, help="s0/sv density sweep summarize output")
    ap.add_argument("--tgrid-summary", default="", help="GST threshold-grid summarize output (orange line)")
    ap.add_argument("--sysgate-summary", default="", help="gated RIT (s0) summarize output (system line)")
    ap.add_argument("--phase0-summary", default="", help="anchor source (always_full_inference arm)")
    ap.add_argument("--anchor", type=float, nargs=2, default=None, metavar=("COST_MS", "SR"),
                    help="anchor given directly (for lines with no phase0 summary, e.g. eval500)")
    ap.add_argument("--reference-curve", default="",
                    help="JSON {label, color, points: [[cost_pct, sr], ...]} drawn as a reference "
                         "line (e.g. the GTP GST+gate curve on the same pool)")
    ap.add_argument("--extra-points", default="",
                    help="JSON {family: {arm: {cost, sr}}} merged into the family series "
                         "(e.g. Rev 1 measurements the sweeps do not re-measure)")
    ap.add_argument("--interval", type=float, nargs=2, default=None, help="budget interval in ms, shaded")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="", help="filename tag; default = suite")
    args = ap.parse_args()

    sgrid = _load(args.sgrid_summary)
    suite = sgrid["suite"]
    summaries = {"sgrid": sgrid}
    for name, path in (("tgrid", args.tgrid_summary), ("sysgate", args.sysgate_summary)):
        if path:
            summaries[name] = _load(path)
            if summaries[name].get("suite") != suite:
                raise SystemExit(f"{name} summary suite != {suite}")
    extra = _load(args.extra_points) if args.extra_points else {}
    extra = {k: v for k, v in extra.items() if not k.startswith("_")}
    if args.anchor:
        full_cost, full_sr = float(args.anchor[0]), float(args.anchor[1])
    elif args.phase0_summary:
        p0 = _load(args.phase0_summary)
        anchor = p0["arms"]["always_full_inference"]
        full_cost, full_sr = float(anchor["realized_cost_ms"]), float(anchor["sr_recorded_not_judged"])
    else:
        full_cost, full_sr = 67.51859499999996, None   # cost basis only; no anchor drawn
    pct = lambda c: 100.0 * c / full_cost  # noqa: E731

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    if args.interval:
        b_l, b_h = args.interval
        ax.axvspan(pct(b_l), pct(b_h), color="#bbbbbb", alpha=0.25, lw=0,
                   label=f"budget interval [{b_l}, {b_h}] ms = [{pct(b_l):.1f}, {pct(b_h):.1f}] %")
    if full_sr is not None:
        ax.axhline(full_sr, color="#444444", ls=":", lw=1)
        ax.scatter([100.0], [full_sr], marker="*", s=320, color="#333333", zorder=6,
                   label=f"always full inference: 100 % cost ({full_cost:.1f} ms), SR {full_sr:.3f}")

    if "tgrid" in summaries:
        thr = dict(summaries["tgrid"]["arms"])
        thr.update(extra.get("threshold", {}))
        _series(ax, thr, FAMILY_COLOR["threshold"], FAMILY_MARKER["threshold"],
                FAMILY_LABEL["threshold"], pct)
    by_family: dict[str, dict] = {}
    for arm, a in sgrid["arms"].items():
        by_family.setdefault(a["family"], {})[arm] = a
    for fam, pts in extra.items():
        if fam != "threshold":
            by_family.setdefault(fam, {}).update(pts)
    for fam in ("s0", "s0_pl", "sv"):
        if fam in by_family:
            _series(ax, by_family[fam], FAMILY_COLOR[fam], FAMILY_MARKER[fam],
                    FAMILY_LABEL[fam], pct, offset=(-28, 7))
    if "sysgate" in summaries:
        _series(ax, summaries["sysgate"]["arms"], SYSGATE_COLOR, "^",
                "RIT ladder (s-only) + production gate (score_hysteresis, j=3, probe=3, L=6)",
                pct, style="--", offset=(4, -12), z=6)

    if args.reference_curve:
        ref = _load(args.reference_curve)
        color = ref.get("color", "#8a8a8a")
        pts = sorted((float(c), float(s)) for c, s in ref["points"])
        front = _pareto_staircase(pts)
        ax.plot([c for c, _ in front], [s for _, s in front], "-", color=color, lw=2.2, zorder=3,
                label=f"{ref['label']} — Pareto frontier ({len(front)} / {len(pts)} cells)")
        on = {(round(c, 9), round(s, 9)) for c, s in front}
        face = [color if (round(c, 9), round(s, 9)) in on else "white" for c, s in pts]
        ax.scatter([c for c, _ in pts], [s for _, s in pts], marker="P", s=58, facecolors=face,
                   edgecolors=color, linewidths=1.2, zorder=4)
    ax.scatter([], [], marker="o", facecolors="k", edgecolors="k", s=40, label="arm on its family's frontier")
    ax.scatter([], [], marker="o", facecolors="white", edgecolors="k", s=40, label="arm dominated within its family")
    ax.plot([], [], ":", color="#666666", lw=1.1, label="two-arm mixture envelope (upper concave hull)")
    ax.set_xlabel("analytic compute cost per decision, % of always-full inference")
    ax.set_ylabel("success rate")
    ax.set_title(f"{suite}: measured arms by family — solid = per-family Pareto frontier, dotted = mixture envelope\n"
                 "GST labels = fh/ws percentiles; RIT labels = calibration quantile (development points, exploratory)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.95)
    fig.tight_layout()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or suite
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pareto_frontiers_{tag}.{ext}", dpi=170)
    plt.close(fig)
    print(f"wrote {out}/pareto_frontiers_{tag}.png|pdf")


if __name__ == "__main__":
    main()
