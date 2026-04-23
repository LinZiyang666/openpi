"""Generate phase1 sweep YAMLs for cp1_llm_layer_extract on libero_spatial.

Matrix:
  - 4 per_modality_* reducers × 4 extract_layer × 12 weight tags (w1..w8 + p1..p4)
    = 192 yamls, keys + field_similarity inherited from phase1/libero_10 peers.
  - prefix_mean_pool × 4 extract_layer × 1 fixed weight config (single-field)
    = 4 yamls (keys forced to vision_0=1.0, robot_state=1.0 only; other disabled).

Legacy group <-> new reducer mapping (same emit structure):
  a  -> cp1_mean_pool         -> per_modality_mean_pool
  c  -> cp1_max_pool          -> per_modality_max_pool
  b1 -> cp1_spatial_pool_16   -> per_modality_spatial_pool_16
  b2 -> cp1_spatial_pool_64*  -> per_modality_spatial_pool_4
  (* legacy _64 == canonical _4; the key/enable layout is identical.)
  e  -> prefix_mean_pool (new group tag, no legacy equivalent)

Layout:
  exp/common/config/phase1/libero_spatial_llm/
      batch1/  w1,w2  -> 4 reducer × 4 layer × 2 weight = 32
      batch2/  w3,w4                                     = 32
      batch3/  w5,w6                                     = 32
      batch4/  w7,w8                                     = 32
      batch5/  p1,p2                                     = 32
      batch6/  p3,p4                                     = 32  +  4 prefix_mean_pool  =  36
  total = 192 + 4 = 196 yamls.

Usage:
    uv run python exp/common/generate_phase1_llm_yamls.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # openpi repo root
LEGACY_DIR = ROOT / "exp" / "common" / "config" / "phase1" / "libero_10"
OUT_DIR = ROOT / "exp" / "common" / "config" / "phase1" / "libero_spatial_llm"
PKL_SUBDIR = "exp/common/data/cache_artifacts/libero_spatial/llm_layer_extract"


# ----------------------------------------------------------------------
# Configuration tables
# ----------------------------------------------------------------------


# New reducer -> (legacy_group_tag, vector_dims)
REDUCER_TO_LEGACY_GROUP: dict[str, tuple[str, dict[str, int]]] = {
    "per_modality_mean_pool": (
        "a",
        {"vision_0": 2048, "vision_1": 2048, "vision_2": 2048,
         "prompt_emb": 2048, "robot_state": 32},
    ),
    "per_modality_max_pool": (
        "c",
        {"vision_0": 2048, "vision_1": 2048, "vision_2": 2048,
         "prompt_emb": 2048, "robot_state": 32},
    ),
    "per_modality_spatial_pool_16": (
        "b1",
        {"vision_0": 32768, "vision_1": 32768, "vision_2": 32768,
         "prompt_emb": 2048, "robot_state": 32},
    ),
    "per_modality_spatial_pool_4": (
        "b2",
        {"vision_0": 8192, "vision_1": 8192, "vision_2": 8192,
         "prompt_emb": 2048, "robot_state": 32},
    ),
}

# Batch assignment: each batch carries two weight tags.
BATCH_WEIGHT_GROUPS: list[tuple[str, list[str]]] = [
    ("batch1", ["w1", "w2"]),
    ("batch2", ["w3", "w4"]),
    ("batch3", ["w5", "w6"]),
    ("batch4", ["w7", "w8"]),
    ("batch5", ["p1", "p2"]),
    ("batch6", ["p3", "p4"]),
]

# Within a batch, iterate reducers in this order for naming determinism.
REDUCER_ORDER = [
    "per_modality_mean_pool",
    "per_modality_spatial_pool_16",
    "per_modality_spatial_pool_4",
    "per_modality_max_pool",
]

LAYERS = [0, 1, 2, 3]


# ----------------------------------------------------------------------
# Legacy yaml loader
# ----------------------------------------------------------------------


def _load_legacy_snippet(group: str, weight_tag: str) -> tuple[dict, dict]:
    """Return (keys_cfg, field_similarity_cfg) from the matching libero_10 yaml."""
    matches = list(LEGACY_DIR.glob(f"batch*/phase1_run_*_{group}_rrf_{weight_tag}.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No legacy yaml for group={group!r} weight_tag={weight_tag!r} under {LEGACY_DIR}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple legacy yamls match {group}/{weight_tag}: {matches}")
    with open(matches[0]) as f:
        doc = yaml.safe_load(f)
    keys_cfg = doc["keys"]
    field_sim = doc["checkpoints"]["cp1"]["search_strategy"]["field_similarity"]
    return keys_cfg, field_sim


# ----------------------------------------------------------------------
# Document builder
# ----------------------------------------------------------------------


def _build_doc(
    reducer: str,
    layer: int,
    keys_cfg: dict,
    field_sim: dict,
    vector_dims: dict[str, int],
    pkl_name: str,
) -> dict:
    """Assemble a single cache config yaml body."""
    return {
        "enabled": True,
        "timer": {"enabled": True, "buffer_size": 10000, "output_csv_dir": None},
        "keys": keys_cfg,
        "key_builder": {
            "type": "cp1_llm_layer_extract",
            "extract_layer": layer,
            "prefix_reducer": {"type": reducer},
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
                    "field_similarity": field_sim,
                    "rrf_k": 60,
                },
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": vector_dims,
            "in_memory": {
                "preload_path": f"{PKL_SUBDIR}/{pkl_name}",
                "index_type": "brute_force",
            },
        },
    }


def _prefix_mean_pool_doc(layer: int) -> dict:
    """prefix_mean_pool has no meaningful multi-field weight sweep; single-field
    vision_0 (+ robot_state) with constant weights = 1.0."""
    keys_cfg = {
        "vision_0":    {"enabled": True,  "weight": 1.0},
        "vision_1":    {"enabled": False, "weight": 0.0},
        "vision_2":    {"enabled": False, "weight": 0.0},
        "prompt_emb":  {"enabled": False, "weight": 0.0},
        "robot_state": {"enabled": True,  "weight": 1.0},
    }
    field_sim = {
        "vision_0":    {"type": "cosine"},
        "robot_state": {
            "type": "l2",
            "to_similarity": {"type": "exp", "tau": 0.334717},
        },
    }
    vector_dims = {"vision_0": 2048, "robot_state": 32}
    pkl_name = f"cp1_llm_l{layer}_prefix_mean_pool.pkl"
    return _build_doc(
        reducer="prefix_mean_pool",
        layer=layer,
        keys_cfg=keys_cfg,
        field_sim=field_sim,
        vector_dims=vector_dims,
        pkl_name=pkl_name,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _dump_yaml(doc: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for batch_name, _ in BATCH_WEIGHT_GROUPS:
        (OUT_DIR / batch_name).mkdir(exist_ok=True)

    # Pre-load every needed (group, weight_tag) snippet from legacy.
    legacy_cache: dict[tuple[str, str], tuple[dict, dict]] = {}
    for reducer, (group, _vdims) in REDUCER_TO_LEGACY_GROUP.items():
        for batch_name, weight_tags in BATCH_WEIGHT_GROUPS:
            for wt in weight_tags:
                if (group, wt) not in legacy_cache:
                    legacy_cache[(group, wt)] = _load_legacy_snippet(group, wt)

    run_idx = 1
    written: list[Path] = []

    # per_modality_* sweep: 192 yamls, 32 per batch.
    for batch_name, weight_tags in BATCH_WEIGHT_GROUPS:
        for reducer in REDUCER_ORDER:
            group, vdims = REDUCER_TO_LEGACY_GROUP[reducer]
            for layer in LAYERS:
                for wt in weight_tags:
                    keys_cfg, field_sim = legacy_cache[(group, wt)]
                    pkl_name = f"cp1_llm_l{layer}_{reducer}.pkl"
                    doc = _build_doc(reducer, layer, keys_cfg, field_sim, vdims, pkl_name)
                    fname = f"phase1_run_{run_idx:03d}_{group}_l{layer}_rrf_{wt}.yaml"
                    out_path = OUT_DIR / batch_name / fname
                    _dump_yaml(doc, out_path)
                    written.append(out_path)
                    run_idx += 1

    # prefix_mean_pool: 4 yamls, piggy-backed onto batch6.
    for layer in LAYERS:
        doc = _prefix_mean_pool_doc(layer)
        fname = f"phase1_run_{run_idx:03d}_e_l{layer}_rrf_const.yaml"
        out_path = OUT_DIR / "batch6" / fname
        _dump_yaml(doc, out_path)
        written.append(out_path)
        run_idx += 1

    # Summary
    print(f"Wrote {len(written)} yamls to {OUT_DIR}")
    for batch_name, _ in BATCH_WEIGHT_GROUPS:
        n = len(list((OUT_DIR / batch_name).glob("*.yaml")))
        print(f"  {batch_name}: {n}")


if __name__ == "__main__":
    main()
