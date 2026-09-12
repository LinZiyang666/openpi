"""Render complete pruning curves and paired diagnostics as standalone PNG/PDF.

Plots consume the integrity-gated analysis and optional four-source retrieval
microbenchmarks. All ten points and all four sources remain visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import check_seal, new_directory, read_json, require


def plot_prune(
    analysis: dict, out_dir: str | Path, *, latency: list[dict] = ()
) -> list[str]:
    """Publish SR, online/offline cost, paired length and per-task figures."""
    check_seal(analysis)
    require(len(analysis["curves"]) == 4, "figures require all four curves")
    latency_map = {}
    for item in latency:
        check_seal(item)
        require(
            item["freeze_digest"] == analysis["freeze_digest"],
            "latency comes from another experiment",
        )
        pair = item["suite"], item["regime"]
        require(
            pair not in latency_map and len(item["summary"]) == 10,
            "duplicate/incomplete latency curve",
        )
        latency_map[pair] = {p["point"]: p for p in item["summary"]}
    require(
        not latency_map or len(latency_map) == 4,
        "report retrieval costs for all four sources together",
    )
    written = []
    with new_directory(out_dir) as temporary:
        for figure_type in ("success_rate", "cost", "paired_length", "per_task"):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            for ax, curve in zip(axes.flat, analysis["curves"]):
                subset = curve["subsets"]["common_unseen"]
                require(
                    subset["estimable"] and len(subset["points"]) == 10,
                    "primary curve is incomplete",
                )
                points = subset["points"]
                libraries = curve["libraries"]
                x = np.array([p["actual_rate"] * 100 for p in libraries])
                ax.set_title(f"{curve['suite']} / {curve['regime']}")
                if figure_type == "success_rate":
                    y = np.array([p["sr"] * 100 for p in points])
                    ci = np.array([p["sr_ci95"] for p in points]) * 100
                    ax.plot(x, y, "o-", label="Common unseen")
                    ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=0.18)
                    ax.plot(
                        x,
                        [p["sr"] * 100 for p in curve["subsets"]["full"]["points"]],
                        "s--",
                        label="Full 500",
                        alpha=0.75,
                    )
                    ax.set_ylabel("Success rate (%)")
                    full_values = [
                        p["sr"] * 100 for p in curve["subsets"]["full"]["points"]
                    ]
                    ax.set_ylim(
                        max(0, min(float(ci[:, 0].min()), min(full_values)) - 3),
                        min(100, max(float(ci[:, 1].max()), max(full_values)) + 3),
                    )
                    ax.legend()
                elif figure_type == "cost":
                    ax.plot(
                        x,
                        [p["online_ms_per_call"] for p in points],
                        "o-",
                        label="Online Σinfer_ms / Σinfers",
                    )
                    ax.set_ylabel("Online mean call latency (ms)")
                    pair = curve["suite"], curve["regime"]
                    if pair in latency_map:
                        twin = ax.twinx()
                        micro = latency_map[pair]
                        twin.plot(
                            x,
                            [micro[p["point"]]["p50_ms"] for p in points],
                            "s--",
                            color="tab:orange",
                            label="Retrieval p50",
                        )
                        twin.plot(
                            x,
                            [micro[p["point"]]["p95_ms"] for p in points],
                            "^:",
                            color="tab:red",
                            label="Retrieval p95",
                        )
                        twin.set_ylabel("Library-query retrieval (ms)")
                        twin.legend(loc="upper right", fontsize=8)
                    ax.legend(loc="upper left", fontsize=8)
                elif figure_type == "paired_length":
                    values = [
                        p["mean_control_steps_delta"]
                        if p["mean_control_steps_delta"] is not None
                        else np.nan
                        for p in points
                    ]
                    ax.plot(x, values, "o-")
                    ax.axhline(0, color="grey", linewidth=0.8)
                    ax.set_ylabel("Control steps Δ vs P00, common successes")
                    for x_i, y_i, point in zip(x, values, points):
                        if np.isfinite(y_i):
                            ax.annotate(
                                f"n={point['common_success_n']}",
                                (x_i, y_i),
                                xytext=(0, 5),
                                textcoords="offset points",
                                fontsize=6,
                            )
                else:
                    image = ax.imshow(
                        np.array([p["per_task_sr"] for p in points]).T * 100,
                        vmin=0,
                        vmax=100,
                        aspect="auto",
                        origin="lower",
                        cmap="viridis",
                    )
                    ax.set_xticks(
                        range(10),
                        [f"P{i:02d}\n{x[i]:.1f}%" for i in range(10)],
                        fontsize=7,
                    )
                    ax.set_yticks(range(10))
                    ax.set_ylabel("Task ID")
                    fig.colorbar(image, ax=ax, label="Success rate (%)")
                ax.set_xlabel(
                    "Point / actual deletion"
                    if figure_type == "per_task"
                    else "Actual entries removed (%)"
                )
                if figure_type != "per_task":
                    ax.grid(alpha=0.2)
            for suffix in ("png", "pdf"):
                filename = f"cache_prune_{figure_type}.{suffix}"
                fig.savefig(temporary / filename, dpi=180)
                written.append(str(Path(out_dir).resolve() / filename))
            plt.close(fig)
    return written


def main() -> None:
    """Render publication-ready artifacts from complete experiment analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--latency", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot_prune(
        read_json(args.analysis), args.out, latency=[read_json(p) for p in args.latency]
    )


if __name__ == "__main__":
    main()
