"""Plot Step 1a per-task failure counts as one line per cfg.

x-axis: task_id 0..9 (10 libero_spatial subtasks).
y-axis: failure count out of 50 shared init states for that task.
Three lines, one per non-inference cache builder (clip / max-pool / spatial16).

This view makes per-cfg specialisation visible in absolute terms: where two
lines cross or diverge, that subtask is a builder-preference signal; where all
three cluster, the subtask is uniformly hard or uniformly easy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openpi_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


BUILDERS: list[tuple[str, str, str, str]] = [
    # config_id, display label, color, marker
    ("clip_w7_d4", "clip_w7_d4", "#E76F51", "o"),
    ("max_pool_w3_d5", "max_pool_w3_d5", "#2A9D8F", "s"),
    ("spatial16_w8_d4", "spatial16_w8_d4", "#457B9D", "^"),
]
NUM_TASKS = 10
INITS_PER_TASK = 50


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_failure_matrix(results_path: Path) -> np.ndarray:
    """Return failure count matrix shaped (n_builders, NUM_TASKS)."""
    rows = json.loads(results_path.read_text())
    builder_index = {cfg: i for i, (cfg, *_rest) in enumerate(BUILDERS)}
    fails = np.zeros((len(BUILDERS), NUM_TASKS), dtype=int)
    seen_units: list[set[tuple[int, int]]] = [set() for _ in BUILDERS]
    for row in rows:
        cfg = row.get("config_id")
        if cfg not in builder_index:
            continue
        bi = builder_index[cfg]
        task_id = int(row["task_id"])
        unit = (task_id, int(row["init_state_idx"]))
        seen_units[bi].add(unit)
        if not bool(row["success"]) and 0 <= task_id < NUM_TASKS:
            fails[bi, task_id] += 1
    # Sanity: each builder should cover NUM_TASKS * INITS_PER_TASK shared units.
    for bi, (cfg, *_rest) in enumerate(BUILDERS):
        expected = NUM_TASKS * INITS_PER_TASK
        if len(seen_units[bi]) != expected:
            print(f"WARN: builder {cfg} has {len(seen_units[bi])} units, expected {expected}")
    return fails


def draw(fails: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(NUM_TASKS)
    for (cfg_id, label, color, marker), row in zip(BUILDERS, fails):
        ax.plot(
            x,
            row,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.2,
            label=label,
        )
        for xi, yi in zip(x, row):
            ax.annotate(
                f"{int(yi)}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in range(NUM_TASKS)], fontsize=11)
    ax.set_xlabel("subtask (task_id)", fontsize=11)
    ax.set_ylabel(f"failures (out of {INITS_PER_TASK})", fontsize=11)
    ax.set_ylim(0, max(fails.max() + 6, 30))
    ax.set_title(
        "Step 1a per-task failures by cache builder",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(loc="upper left", fontsize=10, frameon=True)
    legend.get_frame().set_edgecolor("#cccccc")

    fig.text(
        0.5,
        -0.02,
        "Each task has 50 shared init states. Higher = more failures = worse.",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )

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
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("step1a_failure_by_task.png"),
    )
    args = parser.parse_args()

    fails = _load_failure_matrix(args.results)
    draw(fails, args.out)
    print(f"Wrote {args.out}")
    print("failure matrix (rows=builders, cols=tasks 0..9):")
    for (cfg, *_rest), row in zip(BUILDERS, fails):
        print(f"  {cfg:>20s}: {row.tolist()} (total={int(row.sum())})")


if __name__ == "__main__":
    main()
