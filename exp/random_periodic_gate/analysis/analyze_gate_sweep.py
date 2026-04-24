"""Aggregate random_periodic_gate JSONL results into CSV + Pareto plots.

Inputs:
  - Per-batch results under ``exp/random_periodic_gate/data/batch{N}/results.jsonl``
  - Baseline sidecar: ``exp/trajectory_deviation/data/cache_eval_results.json``
    (3000 rows; 6 config_id x 500 ep — cache + pure inference endpoints).

Outputs (written to the same ``analysis/`` directory):
  - ``aggregate.csv``        per-(cfg, gate_type, slug) aggregation
  - ``pareto_<cfg>.png``     success_rate vs inference_ratio with baseline
                             endpoints overlaid
  - ``heatmap_<cfg>_periodic.png``  cache_len x inference_len success_rate grid

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
    """Group per-episode rows into per-(cfg, gate_type, slug) summaries."""
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["cfg"], row["gate_type"], _slug_of(row))
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
        # Flatten the specific gate params for downstream pivot tables.
        params = bucket[0].get("gate_params", {})
        record["p_inference"] = params.get("p_inference")
        record["seed"] = bucket[0].get("seed")
        record["cache_len"] = params.get("cache_len")
        record["inference_len"] = params.get("inference_len")
        out.append(record)
    return out


def _slug_of(row: dict[str, Any]) -> str:
    """Reconstruct the slug from an in-run row (mirrors generator rules)."""
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


def _plot_pareto(
    cfg: str,
    agg: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping plot %s", out_path)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    # PeriodicGate points.
    px = [r["mean_inference_ratio"] for r in agg if r["cfg"] == cfg and r["gate_type"] == "periodic"]
    py = [r["success_rate"] for r in agg if r["cfg"] == cfg and r["gate_type"] == "periodic"]
    ax.scatter(px, py, label="Periodic", marker="o")
    # RandomGate points.
    rx = [r["mean_inference_ratio"] for r in agg if r["cfg"] == cfg and r["gate_type"] == "random"]
    ry = [r["success_rate"] for r in agg if r["cfg"] == cfg and r["gate_type"] == "random"]
    ax.scatter(rx, ry, label="Random", marker="x")
    # Baseline endpoints.
    for ep in endpoints:
        if ep["cfg"] != cfg:
            continue
        ax.scatter(
            [ep["inference_ratio"]],
            [ep["success_rate"]],
            marker="*",
            s=150,
            label=ep["endpoint"],
        )
    ax.set_xlabel("inference_ratio")
    ax.set_ylabel("success_rate")
    ax.set_title(f"Pareto ({cfg})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_heatmap(cfg: str, agg: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy unavailable; skipping plot %s", out_path)
        return

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

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(PERIODIC_INFERENCE_LENS)))
    ax.set_xticklabels(PERIODIC_INFERENCE_LENS)
    ax.set_yticks(range(len(PERIODIC_CACHE_LENS)))
    ax.set_yticklabels(PERIODIC_CACHE_LENS)
    ax.set_xlabel("inference_len")
    ax.set_ylabel("cache_len")
    ax.set_title(f"PeriodicGate success_rate ({cfg})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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

    for cfg in CFG_NAMES:
        _plot_pareto(cfg, agg, baseline, args.out_dir / f"pareto_{cfg}.png")
        _plot_heatmap(cfg, agg, args.out_dir / f"heatmap_{cfg}_periodic.png")


if __name__ == "__main__":
    main()
