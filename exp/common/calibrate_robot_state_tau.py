from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    task: str
    success: bool
    states: np.ndarray  # [num_steps, state_dim]


def _sorted_step_names(keys: Any) -> list[str]:
    return sorted(key for key in keys if key.startswith("step_"))


def _parse_positive_deltas(text: str) -> tuple[int, ...]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"positive deltas must be > 0, got {value}")
        values.append(value)
    if not values:
        raise ValueError("positive deltas cannot be empty")
    return tuple(sorted(set(values)))


def _load_episode(path: Path) -> EpisodeRecord:
    with h5py.File(path, "r") as h5_file:
        step_names = _sorted_step_names(h5_file.keys())
        if not step_names:
            raise ValueError(f"No step_* groups found in file: {path}")
        states = []
        for step_name in step_names:
            group = h5_file[step_name]
            if "robot_state" not in group:
                raise ValueError(f"Missing robot_state in {path}:{step_name}")
            state = np.asarray(group["robot_state"], dtype=np.float32)
            if state.ndim != 1:
                raise ValueError(f"{path}:{step_name}/robot_state must be 1D, got shape={state.shape}")
            states.append(state)
        task = str(h5_file.attrs.get("task", ""))
        success = bool(h5_file.attrs.get("success", False))
    return EpisodeRecord(path=path, task=task, success=success, states=np.stack(states, axis=0))


def load_episodes(data_dir: Path, *, success_only: bool) -> list[EpisodeRecord]:
    paths = sorted(data_dir.rglob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found under: {data_dir}")

    episodes = []
    for path in paths:
        episode = _load_episode(path)
        if success_only and not episode.success:
            continue
        episodes.append(episode)

    if not episodes:
        raise ValueError("No episodes left after filtering")
    return episodes


def compute_positive_distances(episodes: list[EpisodeRecord], positive_deltas: tuple[int, ...]) -> np.ndarray:
    distances: list[np.ndarray] = []
    for episode in episodes:
        num_steps = int(episode.states.shape[0])
        for delta in positive_deltas:
            if delta >= num_steps:
                continue
            diffs = episode.states[delta:] - episode.states[:-delta]
            distances.append(np.linalg.norm(diffs, axis=1))
    if not distances:
        raise ValueError("No positive pairs were generated; check positive_deltas and episode lengths")
    return np.concatenate(distances, axis=0).astype(np.float64, copy=False)


def compute_cross_episode_negative_distances(
    episodes: list[EpisodeRecord],
    *,
    negatives_per_step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if negatives_per_step <= 0:
        return np.empty((0,), dtype=np.float64)

    task_groups: dict[str, list[EpisodeRecord]] = {}
    for episode in episodes:
        task_groups.setdefault(episode.task, []).append(episode)

    distances: list[np.ndarray] = []
    for group in task_groups.values():
        if len(group) < 2:
            continue
        for index, episode in enumerate(group):
            other_states = [other.states for j, other in enumerate(group) if j != index]
            if not other_states:
                continue
            other_pool = np.concatenate(other_states, axis=0)
            choice_idx = rng.integers(
                low=0,
                high=other_pool.shape[0],
                size=(episode.states.shape[0], negatives_per_step),
                endpoint=False,
            )
            sampled = other_pool[choice_idx]
            diffs = episode.states[:, None, :] - sampled
            distances.append(np.linalg.norm(diffs, axis=2).reshape(-1))

    if not distances:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(distances, axis=0).astype(np.float64, copy=False)


def compute_far_step_negative_distances(
    episodes: list[EpisodeRecord],
    *,
    min_delta: int,
    negatives_per_step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if negatives_per_step <= 0:
        return np.empty((0,), dtype=np.float64)
    if min_delta <= 0:
        raise ValueError(f"negative min delta must be > 0, got {min_delta}")

    distances: list[float] = []
    for episode in episodes:
        num_steps = int(episode.states.shape[0])
        if num_steps <= min_delta:
            continue
        for anchor_idx in range(num_steps):
            candidates = [j for j in range(num_steps) if abs(j - anchor_idx) >= min_delta]
            if not candidates:
                continue
            count = min(negatives_per_step, len(candidates))
            sampled_idx = rng.choice(candidates, size=count, replace=False)
            diffs = episode.states[sampled_idx] - episode.states[anchor_idx]
            distances.extend(np.linalg.norm(diffs, axis=1).tolist())

    return np.asarray(distances, dtype=np.float64)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def build_report(
    *,
    data_dir: Path,
    episodes: list[EpisodeRecord],
    positive_deltas: tuple[int, ...],
    cross_episode_negatives_per_step: int,
    far_negative_min_delta: int,
    far_negatives_per_step: int,
    positive_distances: np.ndarray,
    negative_distances: np.ndarray,
) -> dict[str, Any]:
    tau = float(np.median(positive_distances) / math.log(2.0))
    tau_sweep = [float(scale * tau) for scale in (0.5, 0.75, 1.0, 1.5, 2.0)]

    combined_distances = np.concatenate([positive_distances, negative_distances], axis=0)
    similarity_scores = np.exp(-combined_distances / tau)

    return {
        "data_dir": str(data_dir),
        "episodes_used": len(episodes),
        "tasks_used": len({episode.task for episode in episodes}),
        "total_steps": int(sum(episode.states.shape[0] for episode in episodes)),
        "positive_deltas": list(positive_deltas),
        "cross_episode_negatives_per_step": int(cross_episode_negatives_per_step),
        "far_negative_min_delta": int(far_negative_min_delta),
        "far_negatives_per_step": int(far_negatives_per_step),
        "positive_pair_count": int(positive_distances.size),
        "negative_pair_count": int(negative_distances.size),
        "positive_distance_stats": _percentiles(positive_distances),
        "negative_distance_stats": _percentiles(negative_distances),
        "recommended_tau": tau,
        "tau_formula": "median(d_pos) / ln(2)",
        "tau_sweep": tau_sweep,
        "robot_similarity_stats_at_recommended_tau": _percentiles(similarity_scores),
    }


def _format_stats(name: str, stats: dict[str, float]) -> list[str]:
    if not stats:
        return [f"{name}: <empty>"]
    return [
        (
            f"{name}: min={stats['min']:.6f} p5={stats['p5']:.6f} p25={stats['p25']:.6f} "
            f"p50={stats['p50']:.6f} p75={stats['p75']:.6f} p95={stats['p95']:.6f} "
            f"max={stats['max']:.6f} mean={stats['mean']:.6f} std={stats['std']:.6f}"
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate robot_state tau for Weighted Score Sum using LIBERO-style HDF5 episodes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("exp/common/data/libero_spatial"),
        help="Directory containing episode_*.h5 files",
    )
    parser.add_argument(
        "--positive-deltas",
        type=str,
        default="1,2",
        help="Comma-separated same-episode step deltas treated as positive pairs",
    )
    parser.add_argument(
        "--cross-episode-negatives-per-step",
        type=int,
        default=2,
        help="Number of same-task, different-episode negatives sampled per anchor step",
    )
    parser.add_argument(
        "--far-negative-min-delta",
        type=int,
        default=8,
        help="Minimum |delta step| used for same-episode far negatives",
    )
    parser.add_argument(
        "--far-negatives-per-step",
        type=int,
        default=1,
        help="Number of same-episode far negatives sampled per anchor step",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed episodes instead of filtering to success-only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for negative-pair sampling",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_deltas = _parse_positive_deltas(args.positive_deltas)
    rng = np.random.default_rng(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    episodes = load_episodes(data_dir, success_only=not args.include_failed)
    positive_distances = compute_positive_distances(episodes, positive_deltas)
    negative_distances = np.concatenate(
        [
            compute_cross_episode_negative_distances(
                episodes,
                negatives_per_step=args.cross_episode_negatives_per_step,
                rng=rng,
            ),
            compute_far_step_negative_distances(
                episodes,
                min_delta=args.far_negative_min_delta,
                negatives_per_step=args.far_negatives_per_step,
                rng=rng,
            ),
        ],
        axis=0,
    )
    if negative_distances.size == 0:
        raise ValueError("No negative pairs were generated; adjust the negative sampling settings")

    report = build_report(
        data_dir=data_dir,
        episodes=episodes,
        positive_deltas=positive_deltas,
        cross_episode_negatives_per_step=args.cross_episode_negatives_per_step,
        far_negative_min_delta=args.far_negative_min_delta,
        far_negatives_per_step=args.far_negatives_per_step,
        positive_distances=positive_distances,
        negative_distances=negative_distances,
    )

    lines = [
        f"data_dir={report['data_dir']}",
        f"episodes_used={report['episodes_used']}",
        f"tasks_used={report['tasks_used']}",
        f"total_steps={report['total_steps']}",
        f"positive_deltas={report['positive_deltas']}",
        f"positive_pair_count={report['positive_pair_count']}",
        f"negative_pair_count={report['negative_pair_count']}",
    ]
    lines.extend(_format_stats("positive_distance_stats", report["positive_distance_stats"]))
    lines.extend(_format_stats("negative_distance_stats", report["negative_distance_stats"]))
    lines.append(f"recommended_tau={report['recommended_tau']:.6f}")
    lines.append(f"tau_formula={report['tau_formula']}")
    lines.append("tau_sweep=" + ", ".join(f"{value:.6f}" for value in report["tau_sweep"]))
    lines.extend(
        _format_stats(
            "robot_similarity_stats_at_recommended_tau",
            report["robot_similarity_stats_at_recommended_tau"],
        )
    )
    print("\n".join(lines))

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote_json={output_path}")


if __name__ == "__main__":
    main()
