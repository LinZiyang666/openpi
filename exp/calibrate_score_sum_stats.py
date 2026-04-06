"""Compute per-field p5/p95 statistics for weighted score sum normalization.

Usage:
    uv run exp/calibrate_score_sum_stats.py \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --output data/cache_artifacts/libero_spatial/calibration.json

Output JSON format:
    {
      "cp1_mean_pool": {
        "vision_0":    {"p5": 0.82, "p95": 0.99, "mean": 0.91, "std": 0.04},
        "vision_1":    {"p5": ..., "p95": ...},
        "prompt_emb":  {"p5": ..., "p95": ...},
        "robot_state": {"p5": ..., "p95": ...}
      },
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Field similarity definitions (fixed for this experiment round)
_FIELD_SIM_TYPE: dict[str, str] = {
    "vision_0": "cosine",
    "vision_1": "cosine",
    "prompt_emb": "cosine",
    "robot_state": "l2",
}

_ROBOT_STATE_TAU = 0.334717


def compute_field_stats(
    entries: list,
    field_name: str,
    sim_type: str,
    tau: float = _ROBOT_STATE_TAU,
    num_pairs: int = 50000,
) -> dict[str, float]:
    """Sample pairs and compute similarity distribution stats.

    Similarity is direction-unified to [0, 1]:
      - cosine: (cos + 1) / 2
      - l2: exp(-d / tau)

    Returns: {"p5": float, "p95": float, "mean": float, "std": float}
    """
    valid = [e for e in entries if field_name in e.query_keys]
    n = len(valid)
    if n < 2:
        logger.warning("Field %s: only %d entries with this field, skipping", field_name, n)
        return {"p5": 0.0, "p95": 1.0, "mean": 0.5, "std": 0.0}

    scores = []
    for _ in range(num_pairs):
        i, j = random.sample(range(n), 2)
        qi = valid[i].query_keys[field_name]
        ej = valid[j].query_keys[field_name]

        if sim_type == "cosine":
            cos = float(F.cosine_similarity(qi.unsqueeze(0), ej.unsqueeze(0)))
            s = (cos + 1.0) / 2.0
        elif sim_type == "l2":
            d = float(torch.norm(qi.float() - ej.float(), p=2))
            s = math.exp(-d / tau)
        else:
            raise ValueError(f"Unknown sim_type: {sim_type}")
        scores.append(s)

    scores_np = np.array(scores)
    return {
        "p5":   float(np.percentile(scores_np, 5)),
        "p95":  float(np.percentile(scores_np, 95)),
        "mean": float(np.mean(scores_np)),
        "std":  float(np.std(scores_np)),
    }


def sanity_check(
    entries: list,
    field_name: str,
    sim_type: str,
    tau: float = _ROBOT_STATE_TAU,
    num_pairs: int = 10000,
) -> dict[str, float]:
    """Same-task vs cross-task similarity distribution comparison.

    Returns: {"same_task_mean": float, "cross_task_mean": float, "separation": float}
    separation = same_task_mean - cross_task_mean (higher = better discrimination)
    """
    valid = [e for e in entries if field_name in e.query_keys]
    if len(valid) < 10:
        return {"same_task_mean": 0.0, "cross_task_mean": 0.0, "separation": 0.0}

    by_task: dict[str, list[int]] = {}
    for idx, e in enumerate(valid):
        tk = e.payload.task_key
        by_task.setdefault(tk, []).append(idx)

    tasks = [t for t, idxs in by_task.items() if len(idxs) >= 2]
    if len(tasks) < 2:
        return {"same_task_mean": 0.0, "cross_task_mean": 0.0, "separation": 0.0}

    def _sim(i: int, j: int) -> float:
        qi = valid[i].query_keys[field_name]
        ej = valid[j].query_keys[field_name]
        if sim_type == "cosine":
            return (float(F.cosine_similarity(qi.unsqueeze(0), ej.unsqueeze(0))) + 1.0) / 2.0
        else:
            d = float(torch.norm(qi.float() - ej.float(), p=2))
            return math.exp(-d / tau)

    same_scores = []
    cross_scores = []
    for _ in range(num_pairs):
        if random.random() < 0.5 and tasks:
            # same task
            t = random.choice(tasks)
            i, j = random.sample(by_task[t], 2)
            same_scores.append(_sim(i, j))
        else:
            # cross task
            t1, t2 = random.sample(tasks, 2)
            i = random.choice(by_task[t1])
            j = random.choice(by_task[t2])
            cross_scores.append(_sim(i, j))

    same_mean = float(np.mean(same_scores)) if same_scores else 0.0
    cross_mean = float(np.mean(cross_scores)) if cross_scores else 0.0
    return {
        "same_task_mean": same_mean,
        "cross_task_mean": cross_mean,
        "separation": same_mean - cross_mean,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Calibrate score normalization stats")
    parser.add_argument("--artifact-dir", required=True, help="Directory with .pkl artifacts")
    parser.add_argument("--output", required=True, help="Output calibration JSON path")
    parser.add_argument("--num-pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    artifact_dir = Path(args.artifact_dir)
    pkl_files = sorted(artifact_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {artifact_dir}")
        return

    result: dict[str, dict] = {}

    for pkl_path in pkl_files:
        builder_type = pkl_path.stem
        logger.info("Processing %s ...", builder_type)

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        entries = data["entries"]
        if not entries:
            logger.warning("%s: no entries, skipping", builder_type)
            continue

        builder_stats: dict[str, dict] = {}
        for field_name, sim_type in _FIELD_SIM_TYPE.items():
            stats = compute_field_stats(entries, field_name, sim_type, num_pairs=args.num_pairs)
            sc = sanity_check(entries, field_name, sim_type)
            stats["_sanity"] = sc
            builder_stats[field_name] = stats

            # Print sanity check
            sep = sc["separation"]
            warn = " WARNING: low separation!" if sep < 0.05 else ""
            logger.info(
                "  %s (%s): p5=%.4f p95=%.4f mean=%.4f | "
                "sanity: same=%.4f cross=%.4f sep=%.4f%s",
                field_name, sim_type,
                stats["p5"], stats["p95"], stats["mean"],
                sc["same_task_mean"], sc["cross_task_mean"], sep, warn,
            )

        result[builder_type] = builder_stats

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nCalibration saved to {args.output}")


if __name__ == "__main__":
    main()
