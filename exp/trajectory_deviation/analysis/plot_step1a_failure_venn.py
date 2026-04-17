"""Draw the Step 1a failure-set Venn diagram.

The diagram uses the three non-inference cache builders only. Each circle is a
failure set over shared (task_id, init_state_idx) units.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openpi_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


BUILDERS = {
    "clip_w7_d4": "Clip failures",
    "max_pool_w3_d5": "Max-pool failures",
    "spatial16_w8_d4": "Spatial16 failures",
}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_failure_sets(results_path: Path) -> tuple[dict[str, set[tuple[int, int]]], set[tuple[int, int]]]:
    rows = json.loads(results_path.read_text())
    failure_sets = {builder: set() for builder in BUILDERS}
    all_units: set[tuple[int, int]] = set()

    for row in rows:
        builder = row.get("config_id")
        if builder not in BUILDERS:
            continue
        unit = (int(row["task_id"]), int(row["init_state_idx"]))
        all_units.add(unit)
        if not bool(row["success"]):
            failure_sets[builder].add(unit)

    return failure_sets, all_units


def _regions(failure_sets: dict[str, set[tuple[int, int]]], all_units: set[tuple[int, int]]) -> dict[str, int]:
    clip = failure_sets["clip_w7_d4"]
    max_pool = failure_sets["max_pool_w3_d5"]
    spatial16 = failure_sets["spatial16_w8_d4"]

    return {
        "outside": len(all_units - (clip | max_pool | spatial16)),
        "clip_only": len(clip - max_pool - spatial16),
        "max_only": len(max_pool - clip - spatial16),
        "spatial_only": len(spatial16 - clip - max_pool),
        "clip_max": len((clip & max_pool) - spatial16),
        "clip_spatial": len((clip & spatial16) - max_pool),
        "max_spatial": len((max_pool & spatial16) - clip),
        "all_three": len(clip & max_pool & spatial16),
        "clip_total": len(clip),
        "max_total": len(max_pool),
        "spatial_total": len(spatial16),
        "universe": len(all_units),
    }


def _add_label(ax, x: float, y: float, text: str, *, size: int = 12, weight: str = "normal") -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color="#111111",
    )


def draw_venn(counts: dict[str, int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    circles = [
        ((0.0, 0.72), "#E76F51", f"Clip failures\n{counts['clip_total']}"),
        ((-0.88, -0.58), "#2A9D8F", f"Max-pool failures\n{counts['max_total']}"),
        ((0.88, -0.58), "#457B9D", f"Spatial16 failures\n{counts['spatial_total']}"),
    ]
    radius = 1.55
    for (x, y), color, _label in circles:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=color,
                edgecolor=color,
                linewidth=2.6,
                alpha=0.30,
            )
        )

    # Circle titles sit outside the dense overlap area.
    ax.text(0.0, 2.56, circles[0][2], ha="center", va="center", fontsize=13, fontweight="bold", color="#8F2D1F")
    ax.text(-1.95, -2.15, circles[1][2], ha="center", va="center", fontsize=13, fontweight="bold", color="#17665D")
    ax.text(1.95, -2.15, circles[2][2], ha="center", va="center", fontsize=13, fontweight="bold", color="#285274")

    _add_label(ax, 0.0, 1.45, f"Clip only\n{counts['clip_only']}")
    _add_label(ax, -1.25, -0.95, f"Max only\n{counts['max_only']}")
    _add_label(ax, 1.25, -0.95, f"Spatial16 only\n{counts['spatial_only']}")
    _add_label(ax, -0.64, 0.25, f"Clip + Max\nonly\n{counts['clip_max']}", size=11)
    _add_label(ax, 0.64, 0.25, f"Clip + Spatial16\nonly\n{counts['clip_spatial']}", size=11)
    _add_label(ax, 0.0, -0.86, f"Max + Spatial16\nonly\n{counts['max_spatial']}", size=11)
    _add_label(ax, 0.0, -0.18, f"All 3 fail\n{counts['all_three']}", size=13, weight="bold")

    ax.text(
        2.85,
        1.82,
        f"All 3 pass\n{counts['outside']}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "#F6F6F6",
            "edgecolor": "#999999",
            "linewidth": 0.9,
        },
    )

    ax.set_title(
        "Step 1a failure overlap by keybuilder",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        -2.82,
        f"Universe: {counts['universe']} shared (task_id, init_state_idx) units. "
        "Circles are failures, not successes.",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    ax.set_xlim(-3.2, 3.25)
    ax.set_ylim(-3.0, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=repo_root / "exp/trajectory_deviation/data/cache_eval_results.json",
        help="Path to cache_eval_results.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("step1a_failure_venn.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    failure_sets, all_units = _load_failure_sets(args.results)
    counts = _regions(failure_sets, all_units)
    draw_venn(counts, args.out)

    print(f"Wrote {args.out}")
    for key in [
        "universe",
        "outside",
        "clip_only",
        "max_only",
        "spatial_only",
        "clip_max",
        "clip_spatial",
        "max_spatial",
        "all_three",
    ]:
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
