"""Step 2 / Step 3 analysis + visualisation (plan §12).

Takes two artifact families produced by earlier layers:

- ``deviate_score_{cfg}.json`` (written by ``exp/trajectory_deviation/compute_deviate_scores.py``)
  — per-episode per-cycle background/cache L2 + deviate_score.
- ``spawn_aggregate.csv`` (written by ``exp/trajectory_deviation/run_spawn_experiment.py``) —
  one row per completed spawn unit with its (config, strategy, episode,
  s, n, k_idx, success, env_steps_executed).

Produces four plot families under ``<out-dir>/figures/``:

1. Deviate-score histogram per config (all episodes flattened), with a
   threshold=1 reference line. Answers: "how heavy-tailed is the cache
   error relative to natural model variance?"
2. Top-k coverage (ROC-like): x = top-k% of cycles by deviate-score,
   y = fraction of *spawn-success-failure* cycles captured. Treats
   "spawn failed → cache really pulled us off" as ground truth. A higher
   AUC means deviate-score is a useful ranker.
3. Success-rate heatmap over the (n, k_idx) grid, per config. Answers:
   "how far ahead of the crisis point do we need to teleport, and how
   many high-score points per episode?"
4. Strategy comparison bar chart: top-k vs random vs equidistant at the
   best (n, k_idx) cell per config.

The module keeps the pure math (grid aggregation, histogram binning, ROC
curve) as standalone helpers that are test-friendly; ``matplotlib`` is
imported lazily inside each plotting function so unit tests don't need
an X/GL display. Run with ``uv run python -m exp.trajectory_deviation.analyze_deviation_results
--help`` for the full CLI.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_deviate_scores(path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Parse ``deviate_score_{cfg}.json`` — maps episode → score dict."""
    return json.loads(Path(path).read_text())


def load_spawn_rows(path: Path) -> List[Dict[str, Any]]:
    """Parse ``spawn_aggregate.csv`` into a list of typed row dicts.

    Booleans in CSV land as strings, integers as strings. We coerce back
    here (once) so every downstream analysis helper can trust types.
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "config": r["config"],
                "strategy": r["strategy"],
                "episode": r["episode"],
                "s": int(r["s"]),
                "n": int(r["n"]),
                "k_idx": int(r["k_idx"]),
                "success": r["success"].strip().lower() == "true",
                "env_steps_executed": int(r["env_steps_executed"]),
            })
    return rows


def filter_rows(
    rows: Iterable[Dict[str, Any]], *,
    config: Optional[str] = None,
    strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if config is not None and r["config"] != config:
            continue
        if strategy is not None and r["strategy"] != strategy:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Analysis 1 — deviate-score distribution
# ---------------------------------------------------------------------------


def flatten_deviate_scores(scores: Dict[str, Dict[str, List[float]]]) -> np.ndarray:
    """All per-cycle deviate scores across episodes, concatenated.

    Used to plot one histogram per config (keeping distinct episodes
    per-config is overkill for the "how tailed is this?" question).
    """
    out: List[float] = []
    for ep_scores in scores.values():
        out.extend(ep_scores.get("deviate_score") or [])
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Analysis 2 — Top-k coverage (ROC-like curve)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoveragePoint:
    """One point on the top-k coverage curve.

    ``top_k_fraction`` ∈ [0, 1] — the cumulative fraction of cycles we
    rank as "most deviating". ``recall`` ∈ [0, 1] — of the ground-truth
    failure cycles, how many are in that fraction.
    """

    top_k_fraction: float
    recall: float


def _build_failure_cycle_set(
    rows: Iterable[Dict[str, Any]], *, n_threshold: int
) -> Dict[Tuple[str, int], bool]:
    """Ground-truth: a (ep, s) pair is "truly deviating" if at least one
    spawn unit with ``n <= n_threshold`` starting from ``s`` failed to
    recover. A small ``n`` means "we teleported close to the crisis, and
    the cache STILL couldn't save it" — that's the signal we want
    deviate-score to catch.

    Returns ``{(ep, s): True}`` for failure cycles only so a missing key
    can be read as "recoverable / not relevant".
    """
    # Collect, per (ep, s), whether every small-n spawn succeeded.
    per_pair_failures: Dict[Tuple[str, int], List[bool]] = defaultdict(list)
    for r in rows:
        if r["n"] > n_threshold:
            continue
        per_pair_failures[(r["episode"], r["s"])].append(not r["success"])
    out: Dict[Tuple[str, int], bool] = {}
    for key, fails in per_pair_failures.items():
        # A pair is "truly deviating" iff *any* close-in spawn failed.
        # Using ``any`` (not ``all``) is deliberate: failure on just one
        # k_idx still proves the cycle was hard. ``all`` would be more
        # stringent but far too sparse for the coverage curve.
        if any(fails):
            out[key] = True
    return out


def compute_topk_coverage(
    scores: Dict[str, Dict[str, List[float]]],
    spawn_rows: Iterable[Dict[str, Any]],
    *,
    config: str,
    n_threshold: int = 3,
    resolution: int = 50,
) -> List[CoveragePoint]:
    """Build the ROC-like coverage curve.

    For each cycle across all episodes, we emit a (score, is_failure)
    pair, then sort by descending score. The "top-k% of cycles" curve is
    computed by sweeping the cumulative fraction at ``resolution`` evenly
    spaced thresholds.
    """
    cfg_rows = filter_rows(spawn_rows, config=config, strategy="top_k")
    failure_set = _build_failure_cycle_set(cfg_rows, n_threshold=n_threshold)

    pairs: List[Tuple[float, bool]] = []
    for ep, ep_scores in scores.items():
        for s, score in enumerate(ep_scores.get("deviate_score") or []):
            is_fail = failure_set.get((ep, s), False)
            pairs.append((float(score), bool(is_fail)))
    if not pairs:
        return []
    # Sort by descending score.
    pairs.sort(key=lambda p: -p[0])
    total = len(pairs)
    total_fails = sum(1 for _, f in pairs if f)
    if total_fails == 0:
        # No ground-truth failures → recall is undefined. Returning empty
        # keeps the plot empty rather than drawing a misleading line.
        logger.warning(
            "compute_topk_coverage: config=%s has 0 failure cycles (n<=%d)",
            config, n_threshold,
        )
        return []
    out: List[CoveragePoint] = []
    for i in range(1, resolution + 1):
        cutoff = max(1, int(total * i / resolution))
        captured = sum(1 for _, f in pairs[:cutoff] if f)
        out.append(CoveragePoint(
            top_k_fraction=cutoff / total,
            recall=captured / total_fails,
        ))
    return out


# ---------------------------------------------------------------------------
# Analysis 3 — success-rate grid over (n, k_idx)
# ---------------------------------------------------------------------------


def compute_success_rate_grid(
    rows: Iterable[Dict[str, Any]],
    *,
    config: str,
    strategy: str = "top_k",
) -> Dict[Tuple[int, int], float]:
    """Mean success rate per (n, k_idx) cell.

    Used for the §12.3 heatmap; keeping it as a pure function lets us
    test without matplotlib. An absent (n, k_idx) cell means no unit
    ran there — callers should either skip or render NaN as appropriate.
    """
    bucket: Dict[Tuple[int, int], List[bool]] = defaultdict(list)
    for r in filter_rows(rows, config=config, strategy=strategy):
        bucket[(r["n"], r["k_idx"])].append(bool(r["success"]))
    return {k: float(np.mean(v)) for k, v in bucket.items() if v}


def best_cell(
    grid: Dict[Tuple[int, int], float],
) -> Optional[Tuple[int, int]]:
    """Return the (n, k_idx) cell with the highest success rate.

    Ties broken by smaller ``n`` first (we prefer "teleport closer to
    the crisis"), then smaller ``k_idx`` (smaller k is cheaper).
    """
    if not grid:
        return None
    return max(
        grid.items(),
        key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]),
    )[0]


# ---------------------------------------------------------------------------
# Analysis 4 — strategy comparison at best (n, k_idx)
# ---------------------------------------------------------------------------


def compute_strategy_comparison(
    rows: Iterable[Dict[str, Any]],
    *,
    config: str,
    n: int,
    k_idx: int,
    strategies: Sequence[str] = ("top_k", "random", "equidistant"),
) -> Dict[str, float]:
    """Mean success rate per strategy at a single (n, k_idx) cell."""
    rows = list(rows)
    out: Dict[str, float] = {}
    for strat in strategies:
        scoped = [
            r for r in rows
            if r["config"] == config and r["strategy"] == strat
            and r["n"] == n and r["k_idx"] == k_idx
        ]
        if not scoped:
            continue
        out[strat] = float(np.mean([r["success"] for r in scoped]))
    return out


# ---------------------------------------------------------------------------
# Plotting (thin wrappers; matplotlib imported lazily so tests stay lean)
# ---------------------------------------------------------------------------


def plot_score_histogram(
    scores_flat: np.ndarray,
    *,
    title: str,
    out_path: Path,
    threshold: float = 1.0,
    bins: int = 50,
) -> Path:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores_flat, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
    ax.set_xlabel("deviate_score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_topk_coverage(
    curve: Sequence[CoveragePoint],
    *,
    title: str,
    out_path: Path,
) -> Path:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]

    xs = [p.top_k_fraction for p in curve]
    ys = [p.recall for p in curve]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xs, ys, marker="o", markersize=3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("top-k fraction of cycles (ranked by deviate_score)")
    ax.set_ylabel("recall of failure cycles")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_success_heatmap(
    grid: Dict[Tuple[int, int], float],
    *,
    title: str,
    out_path: Path,
) -> Path:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]

    if not grid:
        logger.warning("plot_success_heatmap: empty grid for %s", title)
        return out_path
    ns = sorted({n for n, _ in grid})
    ks = sorted({k for _, k in grid})
    mat = np.full((len(ks), len(ns)), np.nan, dtype=np.float64)
    for (n, k), v in grid.items():
        mat[ks.index(k), ns.index(n)] = v
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("n (cycles after s before teleport)")
    ax.set_ylabel("k_idx (which top-k point)")
    ax.set_title(title)
    for i, k in enumerate(ks):
        for j, n in enumerate(ns):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v < 0.5 else "black"), fontsize=8)
    fig.colorbar(im, ax=ax, label="success rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_strategy_comparison(
    values: Dict[str, float],
    *,
    title: str,
    out_path: Path,
) -> Path:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]

    strategies = list(values.keys())
    rates = [values[s] for s in strategies]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(strategies, rates, color=["steelblue", "coral", "seagreen"][: len(strategies)])
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze_config(
    cfg: str,
    *,
    deviate_score_dir: Path,
    spawn_rows: List[Dict[str, Any]],
    out_dir: Path,
    n_threshold: int = 3,
) -> List[Path]:
    """Emit all four figure families for a single config. Returns the
    written paths for easy aggregation in the CLI driver."""
    scores_path = Path(deviate_score_dir) / f"deviate_score_{cfg}.json"
    if not scores_path.exists():
        logger.warning("Missing deviate_score JSON for %s; skipping", cfg)
        return []
    scores = load_deviate_scores(scores_path)
    figure_paths: List[Path] = []

    # 1. Histogram.
    scores_flat = flatten_deviate_scores(scores)
    figure_paths.append(plot_score_histogram(
        scores_flat,
        title=f"deviate_score distribution — {cfg}",
        out_path=Path(out_dir) / f"hist_{cfg}.png",
    ))

    # 2. ROC-like coverage curve.
    curve = compute_topk_coverage(
        scores, spawn_rows, config=cfg, n_threshold=n_threshold,
    )
    if curve:
        figure_paths.append(plot_topk_coverage(
            curve,
            title=f"top-k coverage — {cfg}",
            out_path=Path(out_dir) / f"coverage_{cfg}.png",
        ))

    # 3. Success heatmap (top-k strategy).
    grid = compute_success_rate_grid(spawn_rows, config=cfg, strategy="top_k")
    if grid:
        figure_paths.append(plot_success_heatmap(
            grid,
            title=f"spawn success rate (top-k) — {cfg}",
            out_path=Path(out_dir) / f"heatmap_{cfg}.png",
        ))

    # 4. Strategy comparison at the best cell.
    best = best_cell(grid) if grid else None
    if best is not None:
        comparison = compute_strategy_comparison(
            spawn_rows, config=cfg, n=best[0], k_idx=best[1],
        )
        if comparison:
            figure_paths.append(plot_strategy_comparison(
                comparison,
                title=f"strategies @ n={best[0]} k_idx={best[1]} — {cfg}",
                out_path=Path(out_dir) / f"strategy_{cfg}.png",
            ))

    return figure_paths


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deviate-score-dir", required=True, help="Directory with deviate_score_{cfg}.json")
    ap.add_argument("--spawn-csv", required=True, help="Path to spawn_aggregate.csv")
    ap.add_argument("--out-dir", required=True, help="Directory for figures/*.png")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Cache configs to plot (must match the JSON/CSV content)")
    ap.add_argument("--n-threshold", type=int, default=3,
                    help="Close-in spawn cutoff for ground-truth failure labelling")
    return ap.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    rows = load_spawn_rows(Path(args.spawn_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in args.configs:
        paths = analyze_config(
            cfg,
            deviate_score_dir=Path(args.deviate_score_dir),
            spawn_rows=rows,
            out_dir=out_dir,
            n_threshold=args.n_threshold,
        )
        for p in paths:
            logger.info("Wrote %s", p)


if __name__ == "__main__":
    main()
