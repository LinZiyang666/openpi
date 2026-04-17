"""Plot Step 2 deviate-score distributions across cache key builders.

Reads ``deviate_score_{cfg}.json`` files produced by
``exp.trajectory_deviation.compute_deviate_scores`` and writes:

- a continuous histogram plot, colored by threshold bands;
- a Markdown summary with per-cycle and per-episode band counts;
- a JSON summary for downstream notebooks.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


_DEFAULT_CONFIGS = ("clip_w7_d4", "spatial16_w8_d4", "max_pool_w3_d5")
_COLORS = ("#2f9e44", "#228be6", "#f08c00", "#e03131", "#862e9c", "#495057")


def _score_path(score_dir: Path, cfg: str) -> Path:
    return score_dir / f"deviate_score_{cfg}.json"


def _load_scores(path: Path) -> dict[str, dict[str, list[float]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by episode")
    return data


def _build_bands(thresholds: list[float]) -> list[dict[str, Any]]:
    if any(t <= 0 for t in thresholds):
        raise ValueError(f"thresholds must be positive, got {thresholds}")
    thresholds = sorted(set(float(t) for t in thresholds))
    edges = [0.0, *thresholds, math.inf]
    bands = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if lo == 0.0:
            label = f"<{hi:g}"
        elif math.isinf(hi):
            label = f">={lo:g}"
        else:
            label = f"{lo:g}-{hi:g}"
        bands.append({
            "label": label,
            "lo": lo,
            "hi": hi,
            "color": _COLORS[i % len(_COLORS)],
        })
    return bands


def _band_mask(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if math.isinf(hi):
        return values >= lo
    return (values >= lo) & (values < hi)


def summarize_config(
    data: dict[str, dict[str, list[float]]],
    bands: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[float] = []
    per_ep_total: list[int] = []
    per_ep_band_counts: dict[str, list[int]] = {b["label"]: [] for b in bands}
    bad_episodes: list[dict[str, Any]] = []

    for ep, record in data.items():
        keys = set(record)
        expected = {"background_l2", "cache_l2", "deviate_score"}
        if keys != expected:
            bad_episodes.append({"episode": ep, "reason": "keys", "keys": sorted(keys)})
            continue
        lengths = {k: len(record[k]) for k in expected}
        if len(set(lengths.values())) != 1 or lengths["deviate_score"] == 0:
            bad_episodes.append({"episode": ep, "reason": "lengths", "lengths": lengths})
            continue

        arr = np.asarray(record["deviate_score"], dtype=float)
        values.extend(arr.tolist())
        per_ep_total.append(int(arr.size))
        for band in bands:
            per_ep_band_counts[band["label"]].append(
                int(np.sum(_band_mask(arr, band["lo"], band["hi"])))
            )

    values_np = np.asarray(values, dtype=float)
    if values_np.size == 0:
        raise ValueError("No valid deviate scores found")

    band_rows = []
    for band in bands:
        label = band["label"]
        count = int(np.sum(_band_mask(values_np, band["lo"], band["hi"])))
        ep_counts = per_ep_band_counts[label]
        band_rows.append({
            "band": label,
            "count": count,
            "pct": float(count / values_np.size * 100),
            "per_episode_mean": float(statistics.mean(ep_counts)),
            "per_episode_median": float(statistics.median(ep_counts)),
        })

    summary = {
        "episodes": len(data),
        "valid_episodes": len(per_ep_total),
        "bad_episodes": bad_episodes,
        "cycles": int(values_np.size),
        "cycle_per_episode_mean": float(statistics.mean(per_ep_total)),
        "score_mean": float(np.mean(values_np)),
        "score_median": float(np.median(values_np)),
        "score_p90": float(np.percentile(values_np, 90)),
        "score_p95": float(np.percentile(values_np, 95)),
        "score_p99": float(np.percentile(values_np, 99)),
        "score_max": float(np.max(values_np)),
        "bands": band_rows,
    }
    return values_np, summary


def write_markdown(summary: OrderedDict[str, dict[str, Any]], out_path: Path) -> None:
    band_labels = [row["band"] for row in next(iter(summary.values()))["bands"]]
    header = [
        "config",
        "episodes",
        "cycles",
        "mean cycles/ep",
        "mean",
        "median",
        "p95",
        "max",
        *(f"{label} / ep" for label in band_labels),
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
    ]
    for cfg, data in summary.items():
        band_by_label = {row["band"]: row for row in data["bands"]}
        row = [
            cfg,
            str(data["episodes"]),
            str(data["cycles"]),
            f"{data['cycle_per_episode_mean']:.2f}",
            f"{data['score_mean']:.2f}",
            f"{data['score_median']:.2f}",
            f"{data['score_p95']:.2f}",
            f"{data['score_max']:.2f}",
            *(f"{band_by_label[label]['per_episode_mean']:.2f}" for label in band_labels),
        ]
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def plot_distributions(
    values_by_cfg: OrderedDict[str, np.ndarray],
    bands: list[dict[str, Any]],
    out_path: Path,
    *,
    bins: int,
    xmax: float | None,
) -> None:
    if xmax is None:
        xmax = max(float(np.percentile(v, 99.7)) for v in values_by_cfg.values())
        xmax = max(8.0, min(30.0, math.ceil(xmax)))
    bin_edges = np.linspace(0.0, xmax, bins + 1)

    hist_cache = {}
    max_density = 0.0
    for cfg, values in values_by_cfg.items():
        hist, edges = np.histogram(values, bins=bin_edges, density=True)
        hist_cache[cfg] = (hist, edges)
        max_density = max(max_density, float(hist.max()))

    fig_h = max(3.0, 2.8 * len(values_by_cfg))
    fig, axes = plt.subplots(
        len(values_by_cfg), 1,
        figsize=(12, fig_h),
        sharex=True,
        constrained_layout=True,
    )
    if len(values_by_cfg) == 1:
        axes = [axes]

    finite_thresholds = [b["lo"] for b in bands[1:] if math.isfinite(b["lo"])]
    for ax, (cfg, values) in zip(axes, values_by_cfg.items()):
        hist, edges = hist_cache[cfg]
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2
        colors = []
        for center in centers:
            color = bands[-1]["color"]
            for band in bands:
                if center >= band["lo"] and (math.isinf(band["hi"]) or center < band["hi"]):
                    color = band["color"]
                    break
            colors.append(color)
        ax.bar(
            edges[:-1],
            hist,
            width=widths,
            align="edge",
            color=colors,
            edgecolor=colors,
            alpha=0.82,
        )
        for threshold in finite_thresholds:
            if threshold <= xmax:
                ax.axvline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.65)
                ax.text(
                    threshold,
                    max_density * 0.92,
                    f"{threshold:g}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    backgroundcolor="white",
                )
        tail = int(np.sum(values > xmax))
        ax.set_title(
            f"{cfg}: n={len(values)}, mean={np.mean(values):.2f}, "
            f"median={np.median(values):.2f}, max={np.max(values):.2f}, >xmax={tail}"
        )
        ax.set_ylabel("Density")
        ax.set_ylim(0, max_density * 1.08)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("deviate_score = cache_l2 / max(background_l2, floor)")
    fig.legend(
        handles=[
            Patch(facecolor=band["color"], edgecolor=band["color"], label=band["label"])
            for band in bands
        ],
        loc="outside upper center",
        ncol=min(4, len(bands)),
        frameon=False,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-dir",
        default="data/deviation_experiment/deviate_scores",
        help="Directory containing deviate_score_{cfg}.json files.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(_DEFAULT_CONFIGS),
        help="Config ids to plot.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 5.0],
        help="Score thresholds used to color continuous histogram bands.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to SCORE_DIR/plots.",
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--xmax", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir) if args.out_dir else score_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = _build_bands(args.thresholds)

    values_by_cfg: OrderedDict[str, np.ndarray] = OrderedDict()
    summary: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for cfg in args.configs:
        values, cfg_summary = summarize_config(_load_scores(_score_path(score_dir, cfg)), bands)
        values_by_cfg[cfg] = values
        summary[cfg] = cfg_summary

    plot_path = out_dir / "deviate_score_distribution_colored_by_threshold.png"
    summary_json_path = out_dir / "deviate_score_distribution_summary.json"
    summary_md_path = out_dir / "deviate_score_distribution_summary.md"

    plot_distributions(values_by_cfg, bands, plot_path, bins=args.bins, xmax=args.xmax)
    summary_json_path.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, summary_md_path)

    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")


if __name__ == "__main__":
    main()
