"""Batch-generate CP1 temporal prune experiment YAML configs.

Generates YAML configs for all combinations of:
  - reducer:  mean_pool, max_pool
  - window:   3, 4, 5, 6
  - kr:       0.25, 0.5, 0.75
  - weights:  2 sets (WA: vision-light, WB: vision-heavy)

Total: 2 reducers × 4 windows × 3 kr × 2 weights = 48 YAMLs

Usage:
    # Generate all 48 YAMLs
    uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
        --artifact-dir data/cache_artifacts/libero_spatial/temporal_prune \
        --output-dir configs/cache_runs/temporal_prune

    # Dry run: print config count without writing
    uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
        --artifact-dir data/cache_artifacts/libero_spatial/temporal_prune \
        --output-dir configs/cache_runs/temporal_prune \
        --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

REDUCERS = ["mean_pool", "max_pool"]
WINDOWS = [3, 4, 5, 6]
KEEP_RATIOS = [0.25, 0.5, 0.75]

WEIGHT_GRID = {
    "wa": {"vision_0": 0.1,  "vision_1": 0.1,  "robot_state": 0.8},   # robot_state dominant
    "wb": {"vision_0": 0.5,  "vision_1": 0.25, "robot_state": 0.25},  # vision dominant
}

# Both mean_pool and max_pool output 2048-dim vision vectors
_VECTOR_DIMS = {
    "vision_0": 2048,
    "vision_1": 2048,
    "prompt_emb": 2048,
    "robot_state": 32,
}

_FIELD_SIMILARITY = {
    "vision_0":    {"type": "cosine"},
    "vision_1":    {"type": "cosine"},
    "prompt_emb":  {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
}


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def artifact_stem(reducer: str, window: int, kr: float) -> str:
    """e.g., cp1_tp_mean_3w_025kr"""
    r_short = reducer.replace("_pool", "")  # mean, max
    kr_str = str(kr).replace(".", "")       # 025, 05, 075
    return f"cp1_tp_{r_short}_{window}w_{kr_str}kr"


def yaml_filename(run_idx: int, reducer: str, window: int, kr: float, weight_id: str) -> str:
    """e.g., tp_run_001_mean_3w_025kr_wa.yaml"""
    r_short = reducer.replace("_pool", "")
    kr_str = str(kr).replace(".", "")
    return f"tp_run_{run_idx:03d}_{r_short}_{window}w_{kr_str}kr_{weight_id}.yaml"


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


def render_yaml(
    reducer: str,
    window: int,
    kr: float,
    weights: dict[str, float],
    artifact_dir: str,
) -> str:
    """Render a complete temporal prune experiment YAML."""
    preload_path = f"{artifact_dir}/{artifact_stem(reducer, window, kr)}.pkl"

    config = {
        "enabled": True,
        "timer": {"enabled": True, "buffer_size": 10000, "output_csv_dir": None},
        "keys": {
            "vision_0":    {"enabled": True,  "weight": weights.get("vision_0", 0.0)},
            "vision_1":    {"enabled": True,  "weight": weights.get("vision_1", 0.0)},
            "vision_2":    {"enabled": False, "weight": 0.0},
            "prompt_emb":  {"enabled": True,  "weight": 0.0},
            "robot_state": {"enabled": True,  "weight": weights.get("robot_state", 0.0)},
        },
        "key_builder": {
            "type": "cp1_temporal_prune",
            "prune_window_size": window,
            "temporal_keep_ratio": kr,
            "reducer": {"type": reducer},
        },
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": {"type": "always_hit"},
                "search_strategy": {
                    "type": "weighted_rrf_knn",
                    "top_k": 1,
                    "step_filter": "all",
                    "rrf_k": 60,
                    "trajectory_depth": 1,
                    "field_similarity": _FIELD_SIMILARITY,
                },
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": _VECTOR_DIMS,
            "in_memory": {"preload_path": preload_path, "index_type": "brute_force"},
        },
    }

    return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_all(artifact_dir: str, output_dir: str, dry_run: bool = False) -> int:
    """Generate all YAML configs. Returns total count."""
    out_path = Path(output_dir)
    run_idx = 0

    for reducer in REDUCERS:
        for window in WINDOWS:
            for kr in KEEP_RATIOS:
                for weight_id, weights in WEIGHT_GRID.items():
                    run_idx += 1
                    fname = yaml_filename(run_idx, reducer, window, kr, weight_id)
                    if not dry_run:
                        yaml_str = render_yaml(reducer, window, kr, weights, artifact_dir)
                        fpath = out_path / fname
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        fpath.write_text(yaml_str)

    return run_idx


# ---------------------------------------------------------------------------
# Artifact build commands
# ---------------------------------------------------------------------------


def print_artifact_commands(data_dir: str, artifact_dir: str) -> None:
    """Print shell commands to build all 24 artifacts."""
    print("# Build all 24 temporal prune artifacts")
    print(f"mkdir -p {artifact_dir}\n")
    for reducer in REDUCERS:
        for window in WINDOWS:
            for kr in KEEP_RATIOS:
                stem = artifact_stem(reducer, window, kr)
                print(
                    f"uv run python exp/cache_experiment/build_in_memory_cache_artifact.py \\\n"
                    f"    --data-dir {data_dir} \\\n"
                    f"    --builder-type cp1_temporal_prune \\\n"
                    f"    --reducer-type {reducer} \\\n"
                    f"    --prune-window-size {window} \\\n"
                    f"    --temporal-keep-ratio {kr} \\\n"
                    f"    --output {artifact_dir}/{stem}.pkl \\\n"
                    f"    --workers -1\n"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate CP1 temporal prune experiment YAMLs"
    )
    parser.add_argument(
        "--artifact-dir", required=True,
        help="Directory containing temporal prune .pkl artifacts"
    )
    parser.add_argument(
        "--output-dir", default="configs/cache_runs/temporal_prune",
        help="Output directory for YAML files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print count without writing files"
    )
    parser.add_argument(
        "--print-artifact-commands", action="store_true",
        help="Print shell commands to build all 24 artifacts"
    )
    parser.add_argument(
        "--data-dir", default="data/db/libero_cache/libero_spatial",
        help="HDF5 data directory (for --print-artifact-commands)"
    )
    args = parser.parse_args()

    if args.print_artifact_commands:
        print_artifact_commands(args.data_dir, args.artifact_dir)
        return

    count = generate_all(args.artifact_dir, args.output_dir, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry run: would generate {count} YAML files in {args.output_dir}")
    else:
        print(f"Generated {count} YAML files in {args.output_dir}")


if __name__ == "__main__":
    main()
