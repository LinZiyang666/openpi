"""Batch-generate CP1 experiment YAML configs.

Usage:
    # Phase 1: 8 combos x 8 weights = 64 YAMLs (written to <output-dir>/phase1/)
    uv run exp/common/generate_cache_run_yamls.py \
        --phase 1 \
        --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
        --calibration-file exp/common/data/cache_artifacts/libero_spatial/calibration.json \
        --output-dir exp/common/config

    # Phase 1.5: top 3 combos x ~15 fine weights = ~45 YAMLs (-> <output-dir>/phase1_5/)
    uv run exp/common/generate_cache_run_yamls.py \
        --phase 1.5 \
        --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
        --calibration-file exp/common/data/cache_artifacts/libero_spatial/calibration.json \
        --phase1-analysis exp/common/data/phase1/analysis.json \
        --output-dir exp/common/config

    # Phase 2: ~3 YAMLs with prompt_emb weight (-> <output-dir>/phase2/)
    uv run exp/common/generate_cache_run_yamls.py \
        --phase 2 \
        --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
        --calibration-file exp/common/data/cache_artifacts/libero_spatial/calibration.json \
        --phase1-5-analysis exp/common/data/phase1_5/analysis.json \
        --output-dir exp/common/config
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentCombo:
    builder_type: str          # "cp1_mean_pool", "clip", ...
    builder_abbrev: str        # "a", "b1", "b2", "c", "d"
    strategy_type: str         # "weighted_rrf_knn" | "weighted_score_sum_knn"
    strategy_abbrev: str       # "rrf" | "sum"
    vector_dims: dict[str, int]
    artifact_name: str = ""    # pkl filename stem; defaults to builder_type

    def __post_init__(self):
        if not self.artifact_name:
            self.artifact_name = self.builder_type


COMBOS = [
    ExperimentCombo("cp1_mean_pool",       "a",  "weighted_rrf_knn",       "rrf", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_mean_pool",       "a",  "weighted_score_sum_knn", "sum", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_16", "b1", "weighted_rrf_knn",       "rrf", {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_16", "b1", "weighted_score_sum_knn", "sum", {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_64", "b2", "weighted_rrf_knn",       "rrf", {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_64", "b2", "weighted_score_sum_knn", "sum", {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_max_pool",        "c",  "weighted_rrf_knn",       "rrf", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_max_pool",        "c",  "weighted_score_sum_knn", "sum", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("clip",                "d",  "weighted_rrf_knn",       "rrf", {"vision_0": 512, "vision_1": 512, "prompt_emb": 2048, "robot_state": 32}, artifact_name="clip_vit_b_32"),
    ExperimentCombo("clip",                "d",  "weighted_score_sum_knn", "sum", {"vision_0": 512, "vision_1": 512, "prompt_emb": 2048, "robot_state": 32}, artifact_name="clip_vit_b_32"),
]

# Set to True to skip all weighted_score_sum_knn combos (only generate RRF).
SKIP_SCORE_SUM = True

WEIGHT_GRID_PHASE1 = [
    {"vision_0": 1.0,  "vision_1": 0.0,  "robot_state": 0.0},   # W1: vision_0 only
    {"vision_0": 0.0,  "vision_1": 0.0,  "robot_state": 1.0},   # W2: robot_state only
    {"vision_0": 0.5,  "vision_1": 0.0,  "robot_state": 0.5},   # W3: v0 + rs equal
    {"vision_0": 0.25, "vision_1": 0.0,  "robot_state": 0.75},  # W4: rs dominant
    {"vision_0": 0.25, "vision_1": 0.25, "robot_state": 0.5},   # W5: balanced
    {"vision_0": 0.15, "vision_1": 0.1,  "robot_state": 0.75},  # W6: rs heavy
    {"vision_0": 0.1,  "vision_1": 0.1,  "robot_state": 0.8},   # W7: rs very heavy
    {"vision_0": 0.5,  "vision_1": 0.25, "robot_state": 0.25},  # W8: vision dominant
]


def _find_combo(combo_key: str) -> ExperimentCombo:
    """Find combo by 'builder_abbrev_strategy_abbrev' key, e.g. 'a_rrf'."""
    for c in COMBOS:
        if f"{c.builder_abbrev}_{c.strategy_abbrev}" == combo_key:
            return c
    raise ValueError(f"Unknown combo key: {combo_key}")


def _generate_fine_grid(
    center: dict[str, float],
    step: float = 0.1,
    radius: float = 0.2,
) -> list[dict[str, float]]:
    """Generate fine weight grid around center, step=0.1, radius=0.2.

    Weights must be >= 0 and sum to 1.0 (within tolerance).
    """
    fields = ["vision_0", "vision_1", "robot_state"]
    results = []
    steps = [round(i * step, 2) for i in range(-int(radius / step), int(radius / step) + 1)]

    for dv0 in steps:
        for dv1 in steps:
            for drs in steps:
                w = {
                    "vision_0": round(center.get("vision_0", 0.5) + dv0, 4),
                    "vision_1": round(center.get("vision_1", 0.0) + dv1, 4),
                    "robot_state": round(center.get("robot_state", 0.0) + drs, 4),
                }
                if any(v < 0 for v in w.values()):
                    continue
                total = sum(w.values())
                if abs(total - 1.0) > 0.01:
                    continue
                # Normalize to exactly 1.0
                w = {k: round(v / total, 4) for k, v in w.items()}
                # Deduplicate
                key = tuple(sorted(w.items()))
                if key not in {tuple(sorted(r.items())) for r in results}:
                    results.append(w)

    return results


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------

_FIELD_SIMILARITY = {
    "vision_0":    {"type": "cosine"},
    "vision_1":    {"type": "cosine"},
    "prompt_emb":  {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
}


def render_yaml(
    combo: ExperimentCombo,
    weights: dict[str, float],
    weight_id: str,
    artifact_dir: str,
    calibration: dict,
) -> str:
    """Render a complete run YAML string."""
    full_weights = {"prompt_emb": 0.0, **weights}
    preload_path = f"{artifact_dir}/{combo.artifact_name}.pkl"

    search_strategy: dict = {
        "type": combo.strategy_type,
        "top_k": 1,
        "step_filter": "all",
        "field_similarity": _FIELD_SIMILARITY,
    }

    if combo.strategy_type == "weighted_rrf_knn":
        search_strategy["rrf_k"] = 60
    elif combo.strategy_type == "weighted_score_sum_knn":
        cal = calibration.get(combo.artifact_name, {})
        # Require calibration for all fields with weight > 0
        weighted_fields = [f for f, w in full_weights.items() if w > 0]
        missing = [f for f in weighted_fields if f not in cal]
        if missing:
            raise ValueError(
                f"Calibration for {combo.artifact_name} is missing stats for "
                f"weighted fields: {missing}. Re-run calibrate_score_sum_stats.py."
            )
        norm_fields = {}
        for fname in ("vision_0", "vision_1", "prompt_emb", "robot_state"):
            if fname in cal:
                norm_fields[fname] = {"p5": cal[fname]["p5"], "p95": cal[fname]["p95"]}
        search_strategy["score_normalization"] = {
            "type": "percentile",
            "fields": norm_fields,
        }

    config = {
        "enabled": True,
        "timer": {"enabled": True, "buffer_size": 10000, "output_csv_dir": None},
        "keys": {
            "vision_0":    {"enabled": True, "weight": full_weights.get("vision_0", 0.0)},
            "vision_1":    {"enabled": True, "weight": full_weights.get("vision_1", 0.0)},
            "vision_2":    {"enabled": False, "weight": 0.0},
            "prompt_emb":  {"enabled": True, "weight": full_weights.get("prompt_emb", 0.0)},
            "robot_state": {"enabled": True, "weight": full_weights.get("robot_state", 0.0)},
        },
        "key_builder": {"type": combo.builder_type},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": {"type": "always_hit"},
                "search_strategy": search_strategy,
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": combo.vector_dims,
            "in_memory": {"preload_path": preload_path, "index_type": "brute_force"},
        },
    }

    return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Phase generators
# ---------------------------------------------------------------------------


def _active_combos() -> list[ExperimentCombo]:
    if SKIP_SCORE_SUM:
        return [c for c in COMBOS if c.strategy_type != "weighted_score_sum_knn"]
    return list(COMBOS)


def _generate_phase1(args, calibration: dict) -> None:
    run_idx = 0
    out_dir = Path(args.output_dir) / "phase1"
    for combo in _active_combos():
        for w_idx, weights in enumerate(WEIGHT_GRID_PHASE1):
            run_idx += 1
            filename = f"phase1_run_{run_idx:03d}_{combo.builder_abbrev}_{combo.strategy_abbrev}_w{w_idx + 1}.yaml"
            yaml_str = render_yaml(combo, weights, f"w{w_idx + 1}", args.artifact_dir, calibration)
            out_path = out_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_str)
    print(f"Phase 1: generated {run_idx} YAML files in {out_dir}")


def _generate_phase1_5(args, calibration: dict, analysis: dict) -> None:
    top3 = analysis["top3"]
    run_idx = 0
    out_dir = Path(args.output_dir) / "phase1_5"
    for entry in top3:
        combo = _find_combo(entry["combo"])
        center = entry["best_weights"]
        fine_weights = _generate_fine_grid(center, step=0.1, radius=0.2)
        for w_idx, weights in enumerate(fine_weights):
            run_idx += 1
            filename = f"phase1_5_run_{run_idx:03d}_{combo.builder_abbrev}_{combo.strategy_abbrev}_f{w_idx + 1}.yaml"
            yaml_str = render_yaml(combo, weights, f"f{w_idx + 1}", args.artifact_dir, calibration)
            out_path = out_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_str)
    print(f"Phase 1.5: generated {run_idx} YAML files in {out_dir}")


def _generate_phase2(args, calibration: dict, analysis: dict) -> None:
    best = analysis["best"]
    run_idx = 0
    out_dir = Path(args.output_dir) / "phase2"
    combo = _find_combo(best["combo"])
    for prompt_w in [0.0, 0.1, 0.2]:
        run_idx += 1
        weights = {**best["best_weights"], "prompt_emb": prompt_w}
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
        filename = f"phase2_run_{run_idx:03d}_prompt_{prompt_w:.1f}.yaml"
        yaml_str = render_yaml(combo, weights, f"p{prompt_w}", args.artifact_dir, calibration)
        out_path = out_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_str)
    print(f"Phase 2: generated {run_idx} YAML files in {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate CP1 experiment YAML configs")
    parser.add_argument("--phase", required=True, choices=["1", "1.5", "2"])
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Parent directory for generated YAMLs. The script appends a per-phase "
            "subdirectory (phase1/, phase1_5/, phase2/). Example: "
            "--output-dir exp/common/config writes Phase 1 YAMLs to "
            "exp/common/config/phase1/."
        ),
    )
    parser.add_argument("--phase1-analysis", default=None, help="Phase 1 analysis JSON (for phase 1.5)")
    parser.add_argument("--phase1-5-analysis", default=None, help="Phase 1.5 analysis JSON (for phase 2)")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration_file).read_text())

    if args.phase == "1":
        _generate_phase1(args, calibration)
    elif args.phase == "1.5":
        if not args.phase1_analysis:
            parser.error("--phase1-analysis required for phase 1.5")
        analysis = json.loads(Path(args.phase1_analysis).read_text())
        _generate_phase1_5(args, calibration, analysis)
    elif args.phase == "2":
        if not getattr(args, "phase1_5_analysis", None):
            parser.error("--phase1-5-analysis required for phase 2")
        analysis = json.loads(Path(args.phase1_5_analysis).read_text())
        _generate_phase2(args, calibration, analysis)


if __name__ == "__main__":
    main()
