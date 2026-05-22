"""Aggregate random_periodic_gate JSONL results into CSV + Pareto plots.

Inputs:
  - Per-batch results under ``exp/random_periodic_gate/data/batch{N}/results.jsonl``
  - Baseline sidecar: ``exp/trajectory_deviation/data/cache_eval_results.json``
    (3000 rows; 6 config_id x 500 ep — cache + pure inference endpoints).
  - Step 3 overlay: ``exp/trajectory_deviation/data/step3/summary.csv``
    (failure-subset trade-off; converted to full-population coords for overlay).

Outputs (written to the same ``analysis/`` directory):
  - ``aggregate.csv``         per-(cfg, gate_type, slug) aggregation
  - ``pareto_all.png``        1x3 figure: success_rate vs inference_ratio with
                              baseline endpoints overlaid, one subplot per cfg
  - ``heatmap_all_periodic.png``  1x3 figure: cache_len x inference_len Periodic
                                  success_rate grid, shared color scale across cfgs

Plan: ``logs/random_periodic_gate_plan.log.md`` §5.7 / §9 (baseline join).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)


CFG_NAMES = ("clip_w7_d4", "spatial16_w8_d4", "max_pool_w3_d5")

# Step 3 (trajectory_deviation) overlay source. summary.csv is on the failure
# subset; convert to full-population coordinates so it overlays correctly with
# the random_periodic_gate full-population scatter.
STEP3_SUMMARY_PATH = Path("exp/trajectory_deviation/data/step3/summary.csv")

# Step 3 marker style — kept byte-equivalent to plot_step3_tradeoff_total.py
# so the overlay reproduces that figure's scatter unchanged.
STEP3_TAUS = (3, 5, 7, 10)
STEP3_NS = (1, 2, 3, 5, 10)
STEP3_N_MARKERS = {1: "o", 2: "s", 3: "^", 5: "D", 10: "P"}

# RPG Periodic encoding — shape encodes n (shared with Step 3 map so same
# shape means same n across both experiments); color encodes cache_len k via
# a Blues colormap that stays visually distinct from Step 3's plasma palette.
RPG_PERIODIC_KS = (1, 2, 5, 10)
RPG_PERIODIC_NS = (1, 2, 3, 5, 10)
RPG_PERIODIC_N_MARKERS = STEP3_N_MARKERS


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _iter_jsonl_rows(root: Path) -> list[dict[str, Any]]:
    """Read every ``results.jsonl`` under batch directories."""
    rows: list[dict[str, Any]] = []
    for batch_dir in sorted(root.glob("batch*")):
        path = batch_dir / "results.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_baseline_rows(baseline_json: Path) -> list[dict[str, Any]]:
    """Load ``cache_eval_results.json``. Must exist; fail loud otherwise."""
    if not baseline_json.exists():
        raise FileNotFoundError(
            f"baseline source missing: {baseline_json}. "
            "See plan §2.4 — the analysis expects cache_eval_results.json "
            "from the trajectory_deviation experiment."
        )
    return json.loads(baseline_json.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group per-episode rows into per-(cfg, gate_type, key) summaries.

    For random gate, the key drops the seed so all 3 seeds for a given
    p_inference collapse into a single 1500-episode bucket. Periodic gate is
    deterministic in its skip schedule, so each (cache_len, inference_len)
    is its own 500-episode bucket as before.
    """
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["cfg"], row["gate_type"], _bucket_slug(row))
        buckets[key].append(row)

    out: list[dict[str, Any]] = []
    for (cfg, gate_type, slug), bucket in sorted(buckets.items()):
        success_rate = mean(1.0 if r["success"] else 0.0 for r in bucket)
        inference_ratios = [r["inference_ratio"] for r in bucket]
        record = {
            "cfg": cfg,
            "gate_type": gate_type,
            "param_slug": slug,
            "episodes": len(bucket),
            "success_rate": success_rate,
            "mean_inference_ratio": mean(inference_ratios),
            "inference_ratio_source": bucket[0].get("inference_ratio_source", ""),
        }
        params = bucket[0].get("gate_params", {})
        record["p_inference"] = params.get("p_inference")
        # Random buckets pool 3 seeds; periodic has none. Reflect that.
        seeds = sorted({r.get("seed") for r in bucket if r.get("seed") is not None})
        record["seed"] = ",".join(str(s) for s in seeds) if seeds else None
        record["cache_len"] = params.get("cache_len")
        record["inference_len"] = params.get("inference_len")
        out.append(record)
    return out


def _bucket_slug(row: dict[str, Any]) -> str:
    """Slug used for aggregation bucketing — drops seed for random gate so
    seeds are pooled into a single mean point per p_inference.
    """
    params = row.get("gate_params", {})
    if row["gate_type"] == "periodic":
        return f"periodic_k{params['cache_len']}_n{params['inference_len']}"
    if row["gate_type"] == "random":
        p = params["p_inference"]
        return f"random_p{f'{p:.2f}'.replace('.', 'p')}"
    raise ValueError(f"unknown gate_type in row: {row['gate_type']}")


def _slug_of(row: dict[str, Any]) -> str:
    """Reconstruct the original per-seed slug from an in-run row."""
    params = row.get("gate_params", {})
    if row["gate_type"] == "periodic":
        return f"periodic_k{params['cache_len']}_n{params['inference_len']}"
    if row["gate_type"] == "random":
        p = params["p_inference"]
        return f"random_p{f'{p:.2f}'.replace('.', 'p')}_s{row['seed']}"
    raise ValueError(f"unknown gate_type in row: {row['gate_type']}")


# ---------------------------------------------------------------------------
# Baseline endpoints
# ---------------------------------------------------------------------------


def baseline_endpoints(baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute (cfg, endpoint) success rates from cache_eval_results.json.

    Endpoint = "cache" for config_id in CFG_NAMES (cache-on rollouts);
    "pure_inference" for config_id == "inference_<cfg>" (cache-off rollouts).
    """
    cache_success: dict[str, list[bool]] = defaultdict(list)
    inf_success: dict[str, list[bool]] = defaultdict(list)
    for row in baseline_rows:
        cid = row["config_id"]
        succ = bool(row["success"])
        if cid in CFG_NAMES:
            cache_success[cid].append(succ)
        elif cid.startswith("inference_"):
            base_cfg = cid[len("inference_"):]
            if base_cfg in CFG_NAMES:
                inf_success[base_cfg].append(succ)

    out: list[dict[str, Any]] = []
    for cfg in CFG_NAMES:
        if cfg in cache_success:
            out.append({
                "cfg": cfg, "endpoint": "cache", "inference_ratio": 0.0,
                "episodes": len(cache_success[cfg]),
                "success_rate": mean(1.0 if s else 0.0 for s in cache_success[cfg]),
            })
        if cfg in inf_success:
            out.append({
                "cfg": cfg, "endpoint": "pure_inference", "inference_ratio": 1.0,
                "episodes": len(inf_success[cfg]),
                "success_rate": mean(1.0 if s else 0.0 for s in inf_success[cfg]),
            })
    return out


# ---------------------------------------------------------------------------
# CSV + plots
# ---------------------------------------------------------------------------


def write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    if not records:
        raise ValueError("no records to write")
    fieldnames = [
        "cfg", "gate_type", "param_slug",
        "p_inference", "seed", "cache_len", "inference_len",
        "episodes", "success_rate", "mean_inference_ratio",
        "inference_ratio_source",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fieldnames})


def load_step3_overlay(
    summary_path: Path,
    baselines: dict[str, float],
) -> dict[str, list[dict[str, float]]]:
    """Load Step 3 trade-off rows; mirror plot_step3_tradeoff_total.py exactly.

    Step 3 ``summary.csv`` reports success_rate and mean_inference_ratio over
    the failure subset only (episodes where pure cache failed). The original
    Step 3 plot keeps x on the failure-subset axis and converts only y to a
    full-population total success rate:

      x = step3_inf_ratio        (failure-subset inference cost — unchanged)
      y = baseline + (1 - baseline) * step3_sr   (full-population total SR)

    Overlaying on the rpg axes therefore mixes two different x meanings:
    Step 3 x is "extra inference per failure episode" and rpg x is "inference
    per any episode". This is a faithful reproduction of the source figures'
    axes; comparing them at face value is what the source figures already do.
    """
    if not summary_path.exists():
        logger.warning("step3 summary missing: %s — overlay disabled", summary_path)
        return {cfg: [] for cfg in CFG_NAMES}

    out: dict[str, list[dict[str, float]]] = {cfg: [] for cfg in CFG_NAMES}
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            cfg = row["cfg"]
            if cfg not in baselines:
                continue
            b = baselines[cfg]
            sr = float(row["success_rate"])
            ir = float(row["mean_inference_ratio"])
            out[cfg].append({
                "tau": int(row["tau"]),
                "n": int(row["n"]),
                "x": ir,
                "y": b + (1.0 - b) * sr,
            })
    return out


def _pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Return indices on the (min X, max Y) Pareto front, sorted by X."""
    idxs = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    best_y = float("-inf")
    front: list[int] = []
    for i in idxs:
        _, y = points[i]
        if y > best_y:
            front.append(i)
            best_y = y
    return front


def _draw_pareto(
    ax,
    cfg: str,
    agg: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    step3_rows: list[dict[str, float]],
    tau_colors: dict[int, Any],
    k_colors: dict[int, Any],
    *,
    include_step3: bool = True,
) -> None:
    """Render one Pareto subplot on the given axis with Step 3 overlay.

    Two Pareto fronts are drawn (only when ``include_step3`` is True for Step 3):
      - Step 3 front (gray dashed) — only Step 3 points, mirroring
        plot_step3_tradeoff_total.py.
      - RPG front (red dashed) — only random_periodic_gate points
        (Periodic + Random combined).
    """
    cache_baseline = next(
        (ep["success_rate"] for ep in endpoints
         if ep["cfg"] == cfg and ep["endpoint"] == "cache"),
        None,
    )

    # ------------------------------------------------------------------
    # Step 3 scatter — collapsed to a single green triangle per point so the
    # tau/n encoding from the source figure stops competing with the
    # Periodic gate encoding for visual attention.
    # ------------------------------------------------------------------
    if include_step3:
        step3_pts: list[tuple[float, float]] = [(r["x"], r["y"]) for r in step3_rows]
        if step3_pts:
            sx = [p[0] for p in step3_pts]
            sy = [p[1] for p in step3_pts]
            ax.scatter(
                sx, sy, s=55, marker="^", color="#2ca02c",
                edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3,
            )
            front_idx = _pareto_front(step3_pts)
            fx = [step3_pts[i][0] for i in front_idx]
            fy = [step3_pts[i][1] for i in front_idx]
            ax.plot(fx, fy, linestyle="--", color="#555555", linewidth=1.0, zorder=2)

    # ------------------------------------------------------------------
    # RPG Periodic scatter — shape by n (shared with Step 3), color by k.
    # ------------------------------------------------------------------
    rpg_pts: list[tuple[float, float]] = []
    for r in agg:
        if r["cfg"] != cfg or r["gate_type"] != "periodic":
            continue
        x = r["mean_inference_ratio"]
        y = r["success_rate"]
        ax.scatter(
            x, y, s=85,
            color=k_colors[int(r["cache_len"])],
            marker=RPG_PERIODIC_N_MARKERS[int(r["inference_len"])],
            edgecolor="black", linewidth=0.6, alpha=0.95, zorder=3,
        )
        rpg_pts.append((x, y))

    # ------------------------------------------------------------------
    # RPG Random scatter — keep single orange x; only one parameter (p).
    # ------------------------------------------------------------------
    rx = [r["mean_inference_ratio"] for r in agg if r["cfg"] == cfg and r["gate_type"] == "random"]
    ry = [r["success_rate"] for r in agg if r["cfg"] == cfg and r["gate_type"] == "random"]
    ax.scatter(rx, ry, marker="x", color="#ff7f0e", s=70, linewidth=1.5, zorder=3)
    rpg_pts.extend(zip(rx, ry))

    if rpg_pts:
        front_idx = _pareto_front(rpg_pts)
        fx = [rpg_pts[i][0] for i in front_idx]
        fy = [rpg_pts[i][1] for i in front_idx]
        ax.plot(fx, fy, linestyle="--", color="red", linewidth=1.4, zorder=4)

    # ------------------------------------------------------------------
    # Pure-cache baseline horizontal red dotted line.
    # ------------------------------------------------------------------
    if cache_baseline is not None:
        ax.axhline(cache_baseline, linestyle=":", color="#b23a48",
                   linewidth=1.0, zorder=1)
        ax.text(0.98, cache_baseline + 0.005,
                f"pure cache = {cache_baseline*100:.1f}%",
                ha="right", va="bottom", fontsize=8, color="#b23a48",
                transform=ax.get_yaxis_transform())

    # x-axis mixes two scopes: Step 3 reports failure-subset ratio, rpg reports
    # full-population ratio. Label kept generic to mirror the source figures.
    ax.set_xlabel("inference_ratio")
    ax.set_ylabel("success_rate (full population)")
    title_baseline = f" (baseline {cache_baseline*100:.1f}%)" if cache_baseline is not None else ""
    ax.set_title(f"{cfg}{title_baseline}", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.60, 1.02)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def _make_pareto_legend(
    fig,
    tau_colors: dict[int, Any],  # noqa: ARG001 — kept for signature stability
    k_colors: dict[int, Any],
    *,
    include_step3: bool = True,
) -> None:
    """Build a single combined legend below the figure."""
    from matplotlib.lines import Line2D

    td_handle = Line2D(
        [0], [0], marker="^", linestyle="", markersize=8,
        markerfacecolor="#2ca02c", markeredgecolor="black",
        label="Trajectory Deviation",
    )
    # RPG Periodic color (cache_len) — circle marker stand-in.
    k_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=k_colors[k], markeredgecolor="black",
               label=f"Periodic cache_len = {k}")
        for k in RPG_PERIODIC_KS
    ]
    n_handles = [
        Line2D([0], [0], marker=RPG_PERIODIC_N_MARKERS[n], linestyle="",
               markersize=8, markerfacecolor="lightgray",
               markeredgecolor="black", label=f"Periodic inference_len = {n}")
        for n in RPG_PERIODIC_NS
    ]
    random_handle = Line2D(
        [0], [0], marker="x", linestyle="", markersize=8,
        color="#ff7f0e", markeredgewidth=1.5, label="Random gate",
    )
    line_handles: list[Line2D] = []
    if include_step3:
        line_handles.append(
            Line2D([0], [0], linestyle="--", color="#555555",
                   label="Trajectory Deviation Pareto front")
        )
    line_handles.extend([
        Line2D([0], [0], linestyle="--", color="red", label="RPG Pareto front"),
        Line2D([0], [0], linestyle=":", color="#b23a48", label="pure-cache baseline"),
    ])
    head = [td_handle] if include_step3 else []
    handles = head + k_handles + n_handles + [random_handle] + line_handles
    fig.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=5, frameon=False, fontsize=9,
    )


def _build_periodic_grid(cfg: str, agg: list[dict[str, Any]]):
    """Return (data, PERIODIC_CACHE_LENS, PERIODIC_INFERENCE_LENS) for one cfg."""
    import numpy as np

    from exp.random_periodic_gate.generate_batches import (
        PERIODIC_CACHE_LENS, PERIODIC_INFERENCE_LENS,
    )

    data = np.full((len(PERIODIC_CACHE_LENS), len(PERIODIC_INFERENCE_LENS)), np.nan)
    for r in agg:
        if r["cfg"] != cfg or r["gate_type"] != "periodic":
            continue
        try:
            i = PERIODIC_CACHE_LENS.index(r["cache_len"])
            j = PERIODIC_INFERENCE_LENS.index(r["inference_len"])
        except ValueError:
            continue
        data[i, j] = r["success_rate"]
    return data, PERIODIC_CACHE_LENS, PERIODIC_INFERENCE_LENS


def _draw_heatmap(ax, cfg: str, data, cache_lens, inference_lens, vmin: float, vmax: float):
    """Render one Periodic heatmap subplot on the given axis. Returns the AxesImage."""
    import numpy as np

    im = ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(inference_lens)))
    ax.set_xticklabels(inference_lens)
    ax.set_yticks(range(len(cache_lens)))
    ax.set_yticklabels(cache_lens)
    ax.set_xlabel("inference_len")
    ax.set_ylabel("cache_len")
    ax.set_title(f"PeriodicGate success_rate ({cfg})")
    midpoint = (vmin + vmax) / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if v < midpoint else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=9)
    return im


def _plot_pareto_combined(
    agg: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    step3_by_cfg: dict[str, list[dict[str, float]]],
    out_path: Path,
    *,
    include_step3: bool = True,
) -> None:
    """One figure with three side-by-side Pareto subplots, one per cfg."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        logger.warning("matplotlib unavailable; skipping plot %s", out_path)
        return

    tau_colors = {
        t: cm.plasma(i / (len(STEP3_TAUS) - 1)) for i, t in enumerate(STEP3_TAUS)
    }
    # Use Blues colormap from mid-light to dark so all 4 k shades are clearly
    # distinguishable and stay visually disjoint from the warm plasma palette.
    k_colors = {
        k: cm.Blues(0.45 + 0.5 * i / max(1, len(RPG_PERIODIC_KS) - 1))
        for i, k in enumerate(RPG_PERIODIC_KS)
    }

    import textwrap

    fig, axes = plt.subplots(1, len(CFG_NAMES), figsize=(6 * len(CFG_NAMES), 7.4))
    for ax, cfg in zip(axes, CFG_NAMES):
        _draw_pareto(ax, cfg, agg, endpoints, step3_by_cfg.get(cfg, []),
                     tau_colors, k_colors, include_step3=include_step3)
    _make_pareto_legend(fig, tau_colors, k_colors, include_step3=include_step3)
    # Bottom band layout: subplots top, legend middle-bottom, caption very bottom.
    fig.tight_layout(rect=(0, 0.32, 1, 1))
    caption = textwrap.fill(
        "Experiment: on every replan cycle of a LIBERO libero_spatial rollout, a "
        "server-side gate decides whether to reuse the cached action (search) or fall "
        "through to the real policy (skip -> inference). PeriodicGate(cache_len k, "
        "inference_len n) alternates k cached cycles followed by n inference cycles "
        "in a fixed repeating schedule (counter resets per episode), while "
        "RandomGate(p_inference, seed) Bernoulli-samples each cycle independently with "
        "probability p_inference of choosing inference; 3 seeds per p are averaged "
        "to wash out per-rollout randomness.",
        width=180,
    )
    fig.text(
        0.5, 0.015, caption,
        ha="center", va="bottom", fontsize=10,
    )
    plt.subplots_adjust(bottom=0.32)
    # Re-anchor the legend ABOVE the caption band.
    if fig.legends:
        fig.legends[0].set_bbox_to_anchor((0.5, 0.16), transform=fig.transFigure)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_heatmap_combined(agg: list[dict[str, Any]], out_path: Path) -> None:
    """One figure with three side-by-side Periodic heatmaps; shared color scale
    across cfgs so the 3 panels are directly comparable.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy unavailable; skipping plot %s", out_path)
        return

    grids = [_build_periodic_grid(cfg, agg) for cfg in CFG_NAMES]
    finite_all = np.concatenate([d[np.isfinite(d)].ravel() for d, *_ in grids])
    vmin = float(finite_all.min()) - 0.005 if finite_all.size else 0.0
    vmax = float(finite_all.max()) + 0.005 if finite_all.size else 1.0

    fig, axes = plt.subplots(1, len(CFG_NAMES), figsize=(5 * len(CFG_NAMES), 4))
    last_im = None
    for ax, cfg, (data, cache_lens, inference_lens) in zip(axes, CFG_NAMES, grids):
        last_im = _draw_heatmap(ax, cfg, data, cache_lens, inference_lens, vmin, vmax)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate random_periodic_gate results")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("exp/random_periodic_gate/data"),
        help="Root directory containing batch{N}/results.jsonl",
    )
    p.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("exp/trajectory_deviation/data/cache_eval_results.json"),
        help="Baseline source: trajectory_deviation cache_eval_results.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exp/random_periodic_gate/analysis"),
    )
    p.add_argument(
        "--no-step3",
        action="store_true",
        help="Skip the Trajectory Deviation (Step 3) overlay and emit "
             "pareto_no_step3.png instead of pareto_all.png.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _iter_jsonl_rows(args.data_root)
    if not rows:
        logger.warning("no results.jsonl found under %s", args.data_root)
        return
    agg = aggregate_rows(rows)
    baseline = baseline_endpoints(load_baseline_rows(args.baseline_json))

    csv_path = args.out_dir / "aggregate.csv"
    write_csv(agg, csv_path)
    logger.info("wrote %s (%d rows)", csv_path, len(agg))

    cache_baselines = {
        ep["cfg"]: ep["success_rate"]
        for ep in baseline if ep["endpoint"] == "cache"
    }
    step3_by_cfg = (
        {cfg: [] for cfg in CFG_NAMES}
        if args.no_step3
        else load_step3_overlay(STEP3_SUMMARY_PATH, cache_baselines)
    )
    pareto_name = "pareto_no_step3.png" if args.no_step3 else "pareto_all.png"
    _plot_pareto_combined(
        agg, baseline, step3_by_cfg, args.out_dir / pareto_name,
        include_step3=not args.no_step3,
    )
    _plot_heatmap_combined(agg, args.out_dir / "heatmap_all_periodic.png")


if __name__ == "__main__":
    main()
