"""Export the ActionCache-baseline arms as a ``rit_pareto.figure/v1`` spec (points only).

Writes ``exp/rit_pareto/analysis/figures/pareto_acb_<suite>.json`` in the same
schema the RIT-Pareto figure chain uses (``exp/rit_pareto/build_figure.py``),
so the renderer / editor can draw it and other figures can reference it. Each
point is one arm: ``x`` = measured inference ratio (CUDA-graph analytic tiers,
``% of always-full``), ``y`` = success rate over ``n_ep`` episodes, ``label`` =
tier + target (``N0 IR65``, ``N1 θ=.85``). Four measured series per suite —
library regime (50-trajectory / S6 full) × N_hit (0 = FULL_HIT, 1 = WARM@0.1) —
plus a pointer to the same-library RIT-PL K=2 no-gate frontier of
``pareto_rit_<suite>`` (resolved at render time, never copied).

The arms are read straight from the raw runs through ``aggregate`` (its
completeness / purity gates stay fail-closed); no aggregate file is needed.

Usage:
  uv run python -m exp.actioncache_baseline.export_figure_points --suite libero_spatial \\
      --lib50-run exp/actioncache_baseline/data/runs/libero_spatial_lib50 \\
      --s6-run exp/actioncache_baseline/data/runs/libero_spatial_s6 \\
      --config-root exp/actioncache_baseline/config --out-dir exp/rit_pareto/analysis/figures
"""

from __future__ import annotations

import argparse
import json
import pathlib

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.aggregate import aggregate
from exp.rit_pareto.build_figure import _spec, reference_series, series_from_arms, write_spec

LIB_NAMES = {"lib50": "50-traj lib", "s6": "S6 full lib"}
TIER_NAMES = {"n0": "N_hit=0 (FULL)", "n1": "N_hit=1 (WARM@0.1)"}
_PRIMARY = {"marker_size": 34, "alpha": 0.85, "point_zorder": 3, "linestyle": "-", "linewidth": 2.0, "line_zorder": 2}
STYLES = {
    ("lib50", "n0"): {"color": "#e07b39", "marker": "o", **_PRIMARY},
    ("lib50", "n1"): {"color": "#e07b39", "marker": "s", **_PRIMARY},
    ("s6", "n0"): {"color": "#3b6fb6", "marker": "o", **_PRIMARY},
    ("s6", "n1"): {"color": "#3b6fb6", "marker": "s", **_PRIMARY},
}


def point_label(arm: str) -> str:
    p = libs.parse_arm(arm)
    if p is None:
        return arm
    tier = p["tier"].upper()
    t = p["target"]
    if t.startswith("ref"):
        return f"{tier} θ=.{t[3:].rstrip('0') or '0'}"
    return f"{tier} IR{t[2:]}"


def series_arms(arms: dict, lib: str, tier: str) -> dict:
    """``{arm: {ir_percent, success_rate, label, n_ep}}`` for one (library, tier)."""
    out = {}
    for name, rec in arms.items():
        p = libs.parse_arm(name)
        if p is None or p["lib"] != lib or p["tier"] != tier:
            continue
        out[name] = {"ir_percent": rec["ir_percent"], "success_rate": rec["success_rate"],
                     "label": point_label(name), "n_ep": rec["n_ep"]}
    return out


def build_acb(suite: str, runs: dict[str, pathlib.Path], config_root: pathlib.Path) -> dict:
    series = []
    for lib, run_dir in runs.items():
        record = json.loads((config_root / f"{suite}_{lib}" / "export_record.json").read_text())
        arms = aggregate(run_dir, expect_episodes=500, export_record=record, suite=suite)["arms"]
        for tier in ("n0", "n1"):
            key = f"ActionCache {LIB_NAMES[lib]} {TIER_NAMES[tier]}"
            series.append(series_from_arms(key, series_arms(arms, lib, tier), style=STYLES[(lib, tier)]))
    series.append(reference_series("RIT-PL K=2 no gate (50-traj lib)", f"pareto_rit_{suite}", "no gate"))
    return _spec(f"pareto_acb_{suite}",
                 f"{suite}: ActionCache-style post-backbone (CP2) single-threshold arms, pruned-500 pool", series)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(libs.SUITE_TAGS))
    ap.add_argument("--lib50-run", required=True)
    ap.add_argument("--s6-run", required=True)
    ap.add_argument("--config-root", default="exp/actioncache_baseline/config")
    ap.add_argument("--out-dir", default="exp/rit_pareto/analysis/figures")
    args = ap.parse_args()
    spec = build_acb(args.suite, {"lib50": pathlib.Path(args.lib50_run), "s6": pathlib.Path(args.s6_run)},
                     pathlib.Path(args.config_root))
    print(write_spec(spec, pathlib.Path(args.out_dir)))


if __name__ == "__main__":
    main()
