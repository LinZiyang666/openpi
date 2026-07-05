"""Stage 2b -- fair (SR, inference_ratio) Pareto overlay (offline, 0 GPU).

Builds two overlay figures (libero_spatial, libero_10) that are STRUCTURALLY
ASYMMETRIC by design:

- **load-bearing layer** (same protocol as the 8 live points: always_search +
  ThresholdJudge + 50-init): the Stage-0 d1 frontier anchors (3 spatial / 4 l10,
  SR + exact inf computed directly from the gate_rows) + the 4 live points per
  suite (2 N1 + 2 periodic). The F11 verdict "does periodic raise the existing
  frontier" is decided ONLY on this layer.
- **reference layer** (libero_spatial ONLY): the RPG periodic/random anchors from
  ``random_periodic_gate/analysis/aggregate.csv``. Different judge (AlwaysHitJudge)
  + old keybuilders + ``inference_ratio_source=derived`` -> plotted hollow with a
  ``(different search/judge)`` legend, never connected / interpolated / used in
  the verdict. libero_10 has NO RPG anchor.

Reads only recorded gate_rows + the RPG aggregate csv + run manifests; writes one
markdown report and the two figures. No ``src`` / inference-path dependency.

Usage:
    python -m exp.gate_research.stage2b_pareto_overlay \
        --spatial-rows <gate_rows> --l10-rows <gate_rows> \
        --manifests <m.json> [...] --rpg-csv <aggregate.csv> \
        --out <report.md> --fig-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

from exp.gate_research.analyze_n1_live import (
    baseline_inf_ratio,
    load_jsonl,
    run_metrics,
)
from exp.gate_research.stage2_common import load_stage0_episodes


# ------------------------------------------------------------------
# Anchor / point assembly (pure)
# ------------------------------------------------------------------
def _yaml_short(yaml_id: str) -> str:
    return yaml_id.split("__d1__")[-1]


def stage0_yaml_ids(gate_rows_path) -> list[str]:
    """Distinct per-step ``yaml_id``s present in a Stage-0 gate_rows file."""
    seen = []
    for r in load_jsonl(gate_rows_path):
        if r.get("_kind") == "episode_summary":
            continue
        y = r.get("yaml_id")
        if y is not None and y not in seen:
            seen.append(y)
    return seen


def stage0_anchors(gate_rows_path, replan_steps: int = 5) -> list[dict]:
    """Protocol-matched d1 frontier anchors: SR (per-step success) + exact inf
    (warm-cost weighted) for every config in the Stage-0 gate_rows."""
    out = []
    for yid in stage0_yaml_ids(gate_rows_path):
        eps = load_stage0_episodes(gate_rows_path, yid, replan_steps=replan_steps)
        sr = 100.0 * sum(e.success for e in eps) / len(eps)
        inf = baseline_inf_ratio(gate_rows_path, yid, replan_steps)
        out.append({"label": _yaml_short(yid), "sr": sr, "inf": inf,
                    "layer": "loadbearing", "kind": "d1_anchor"})
    return out


def live_points(manifests: list[dict]) -> list[dict]:
    """SR / inf / skip for each live run (reuses the verified ``run_metrics``)."""
    out = []
    for m in manifests:
        met = run_metrics(m)
        out.append({"label": m["run_id"], "sr": 100.0 * met["sr"],
                    "inf": met["live_inf_ratio"], "skip": 100.0 * met["skip_pct"],
                    "gate_type": m["gate_type"], "suite": m["suite"],
                    "layer": "loadbearing", "kind": m["gate_type"]})
    return out


def parse_rpg_anchors(csv_path) -> list[dict]:
    """RPG periodic/random anchors (libero_spatial only, reference layer).

    Marked ``layer='reference'`` with the different-protocol caveat; the
    ``inference_ratio_source`` (derived/expected) is carried through verbatim.
    """
    out = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append({
                "label": f"{row['cfg']}/{row['param_slug']}",
                "cfg": row["cfg"], "gate_type": row["gate_type"],
                "sr": 100.0 * float(row["success_rate"]),
                "inf": float(row["mean_inference_ratio"]),
                "inf_source": row["inference_ratio_source"],
                "layer": "reference", "kind": "rpg",
            })
    return out


def parse_extra_anchors(specs) -> dict:
    """Optional extra load-bearing anchors (e.g. the roadmap's pure-inference
    ceiling used to bound l10, whose live points sit beyond the d1-anchor inf
    range). Each spec is ``suite:inf:sr:label``. Default off -> the tool reports
    out-of-range live points honestly rather than extrapolating. The injected
    anchor's SR protocol is the CALLER's responsibility (report as such)."""
    out = collections.defaultdict(list)
    for s in specs or []:
        suite, inf, sr, label = s.split(":")
        out[suite].append({"label": label, "inf": float(inf), "sr": float(sr),
                           "layer": "loadbearing", "kind": "pure_inf"})
    return out


def _frontier_anchors(loadbearing: list[dict]) -> list[dict]:
    """Anchor points that define the frontier line (d1 anchors + any injected
    pure-inference ceiling); live gate points are NOT anchors."""
    return [p for p in loadbearing if p["kind"] in ("d1_anchor", "pure_inf")]


def build_overlay_layers(suite: str, anchors: list[dict], live: list[dict],
                         rpg: list[dict]) -> dict:
    """Assemble the two layers for one suite. The RPG reference layer is included
    ONLY for libero_spatial (l10 has no RPG anchor); every RPG point MUST be
    ``layer='reference'`` (never load-bearing)."""
    load_bearing = list(anchors) + [p for p in live if p["suite"] == suite]
    reference = []
    if suite == "libero_spatial":
        assert all(p["layer"] == "reference" for p in rpg), "RPG points must be reference-only"
        reference = list(rpg)
    return {"suite": suite, "loadbearing": load_bearing, "reference": reference}


# ------------------------------------------------------------------
# Frontier verdict (pure) -- load-bearing layer only
# ------------------------------------------------------------------
def frontier_interp(anchors: list[dict], inf: float):
    """Piecewise-linear SR of the d1-anchor frontier at ``inf``. Returns ``None``
    outside the anchor inf range (no extrapolation -- an out-of-range live point
    is reported as such, not silently extrapolated)."""
    pts = sorted(((a["inf"], a["sr"]) for a in anchors))
    if not pts or inf < pts[0][0] or inf > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= inf <= x1:
            if x1 == x0:
                return max(y0, y1)
            return y0 + (y1 - y0) * (inf - x0) / (x1 - x0)
    return pts[-1][1]


def frontier_gain(point: dict, anchors: list[dict]) -> dict:
    """Vertical SR gap of a live point above the d1-anchor frontier at its inf
    (positive = the point sits above the existing frontier). ``interp=None`` when
    the point's inf is outside the anchor range."""
    interp = frontier_interp(anchors, point["inf"])
    return {"label": point["label"], "inf": point["inf"], "sr": point["sr"],
            "frontier_sr": interp,
            "gain_pp": (point["sr"] - interp) if interp is not None else None}


def pareto_dominates(p: dict, q: dict) -> bool:
    """``p`` dominates ``q`` in (SR up, inf down): >= on both and > on one."""
    return (p["sr"] >= q["sr"] and p["inf"] <= q["inf"]
            and (p["sr"] > q["sr"] or p["inf"] < q["inf"]))


# ------------------------------------------------------------------
# Plot + report
# ------------------------------------------------------------------
def plot_overlay(layers: dict, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    anchors = _frontier_anchors(layers["loadbearing"])
    fr = sorted(((a["inf"], a["sr"]) for a in anchors))
    if fr:
        ax.plot([x for x, _ in fr], [y for _, y in fr], "-o", color="#1f4e79",
                label="d1 frontier (same protocol)", zorder=3)
        for a in anchors:
            ax.annotate(a["label"], (a["inf"], a["sr"]), fontsize=6, alpha=0.7)
    for p in layers["loadbearing"]:
        if p["kind"] in ("d1_anchor", "pure_inf"):
            continue
        c = "#c00000" if p["kind"] == "periodic" else "#2e8b57"
        ax.scatter(p["inf"], p["sr"], s=70, marker="D", color=c, zorder=4,
                   label=p["kind"])
        ax.annotate(p["label"], (p["inf"], p["sr"]), fontsize=6)
    for p in layers["reference"]:
        ax.scatter(p["inf"], p["sr"], s=22, facecolors="none",
                   edgecolors="#888", alpha=0.5, zorder=2)
    if layers["reference"]:
        ax.scatter([], [], s=22, facecolors="none", edgecolors="#888",
                   label="RPG (different search/judge)")
    ax.set_xlabel("inference_ratio")
    ax.set_ylabel("SR (%)")
    ax.set_title(f"Stage 2b Pareto overlay — {layers['suite']}")
    # de-dup legend labels
    h, lab = ax.get_legend_handles_labels()
    seen = dict(zip(lab, h))
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    fig.savefig(str(fig_path).replace(".png", ".pdf"))
    plt.close(fig)


def render_md(overlays: list[dict]) -> str:
    L = ["# Stage 2b — 公平 Pareto overlay（RPG 坐标）", "",
         "承重层 = 同协议（always_search + ThresholdJudge + 50-init）；"
         "RPG = 参照层（libero_spatial only，异 judge/keybuilder，不承重）。", ""]
    for ov in overlays:
        anchors = _frontier_anchors(ov["loadbearing"])
        L += [f"## {ov['suite']}", "",
              "| 点 | 层 | inf | SR% | frontier SR | gain pp |", "|---|---|---|---|---|---|"]
        for a in sorted(anchors, key=lambda x: x["inf"]):
            tag = "pure_inf" if a["kind"] == "pure_inf" else "d1"
            L.append(f"| {a['label']} | {tag} | {a['inf']:.3f} | {a['sr']:.1f} | — | — |")
        for p in [p for p in ov["loadbearing"] if p["kind"] not in ("d1_anchor", "pure_inf")]:
            g = frontier_gain(p, anchors)
            fr = f"{g['frontier_sr']:.1f}" if g["frontier_sr"] is not None else "OOR"
            gp = f"{g['gain_pp']:+.1f}" if g["gain_pp"] is not None else "—(inf 越界)"
            L.append(f"| {p['label']} | {p['kind']} | {p['inf']:.3f} | {p['sr']:.1f} | {fr} | {gp} |")
        if ov["reference"]:
            L.append(f"| RPG×{len(ov['reference'])} | 参照(不承重) | — | — | — | — |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Stage 2b fair Pareto overlay")
    ap.add_argument("--spatial-rows", required=True)
    ap.add_argument("--l10-rows", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--rpg-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--extra-anchor", nargs="*", default=[],
                    help="optional load-bearing anchors 'suite:inf:sr:label' "
                         "(e.g. libero_10:1.0:83:pure_inf); SR protocol is the caller's")
    a = ap.parse_args()

    import json
    manifests = [json.loads(Path(p).read_text()) for p in a.manifests]
    live = live_points(manifests)
    rpg = parse_rpg_anchors(a.rpg_csv)
    extra = parse_extra_anchors(a.extra_anchor)
    overlays = [
        build_overlay_layers("libero_spatial",
                             stage0_anchors(a.spatial_rows) + extra.get("libero_spatial", []),
                             live, rpg),
        build_overlay_layers("libero_10",
                             stage0_anchors(a.l10_rows) + extra.get("libero_10", []),
                             live, rpg),
    ]
    fig_dir = Path(a.fig_dir)
    for ov, name in zip(overlays, ("stage2_pareto_spatial.png", "stage2_pareto_l10.png")):
        plot_overlay(ov, fig_dir / name)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(overlays))
    print(f"[stage2b] wrote {out} + figures in {fig_dir}")


if __name__ == "__main__":
    main()
