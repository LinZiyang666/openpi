"""Emit Phase-2 eval YAMLs for the weighted-sum weight search.

Reads the Phase-1 calibration artifact (``calibration_normalizers.json``) and
writes one cache-config YAML per (keybuilder, weight-config) cell. Each YAML:

- uses ``search_strategy.type = weighted_score_sum_knn`` with the per-field
  Layer-1 normalizers selected in Phase 1 (``score_normalization.type =
  per_field``);
- uses an ``always_hit`` judge (pure cache replay) to isolate retrieval quality;
- masks ``prompt_emb`` via ``keys.prompt_emb.enabled: false`` (dropped from the
  experiment);
- sets ``write_policy: {type: never}`` so the C2 write-frozen server accepts the
  bundle at load (the default ``on_any_miss`` would fail fast).

Weight configs come from two families: per-modality isolation (one field at a
time) and a coarse weight grid over the useful modalities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# Fields eligible for weighting (prompt_emb is dropped; vision_2 only if present).
WEIGHTED_FIELDS = ("vision_0", "vision_1", "vision_2", "robot_state")
_ALL_KEY_FIELDS = ("vision_0", "vision_1", "vision_2", "prompt_emb", "robot_state")
# cp1_* and clip key_builders hard-require these enabled (config validator).
_REQUIRED_KEYBUILDER_FIELDS = ("vision_0", "robot_state")


def build_eval_config(
    *,
    builder_type: str,
    vector_dims: dict[str, int],
    preload_path: str,
    weights: dict[str, float],
    fields_calib: dict[str, dict],
    trajectory_depth: int = 1,
    trajectory_weights: list[float] | None = None,
) -> dict:
    """Build one eval cache-config dict (validated downstream by the loader)."""
    keys = {f: {"enabled": False, "weight": 0.0} for f in _ALL_KEY_FIELDS}
    vdims: dict[str, int] = {}
    field_similarity: dict[str, dict] = {}
    sn_fields: dict[str, dict] = {}

    for field in WEIGHTED_FIELDS:
        w = float(weights.get(field, 0.0))
        if w <= 0:
            continue
        if field not in vector_dims:
            raise ValueError(f"{builder_type}: field {field!r} has no vector dim in calibration")
        if field not in fields_calib:
            raise ValueError(f"{builder_type}: field {field!r} has no Phase-1 calibration")
        keys[field] = {"enabled": True, "weight": w}
        vdims[field] = int(vector_dims[field])
        sim_type = fields_calib[field]["sim_type"]
        selected = fields_calib[field]["selected"]
        if sim_type == "l2":
            tau = float(selected["params"].get("tau", 1.0))
            field_similarity[field] = {"type": "l2", "to_similarity": {"type": "exp", "tau": tau}}
        else:
            field_similarity[field] = {"type": "cosine"}
        sn_fields[field] = {"method": selected["method"], "params": selected["params"]}

    if not sn_fields:
        raise ValueError("no weighted fields enabled (all weights <= 0)")

    # cp1_* / clip key_builders require vision_0 + robot_state enabled. Force them
    # on at weight 0 (no _iter_active_fields contribution, no normalizer, no
    # score_normalization entry) so modality-isolation / partial-grid YAMLs still
    # load. This keeps the leak-free weighted sum unchanged.
    for req in _REQUIRED_KEYBUILDER_FIELDS:
        if not keys[req]["enabled"]:
            if req not in vector_dims:
                raise ValueError(f"{builder_type}: required field {req!r} missing vector dim")
            keys[req] = {"enabled": True, "weight": 0.0}
            vdims[req] = int(vector_dims[req])

    search_strategy = {
        "type": "weighted_score_sum_knn",
        "top_k": 1,
        "step_filter": "all",
        "field_similarity": field_similarity,
        "score_normalization": {"type": "per_field", "fields": sn_fields},
    }
    if trajectory_depth > 1:
        search_strategy["trajectory_depth"] = trajectory_depth
        search_strategy["trajectory_weights"] = trajectory_weights

    return {
        "enabled": True,
        "timer": {"enabled": False},
        "keys": keys,
        "key_builder": {"type": builder_type},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": {"type": "always_hit"},
                "search_strategy": search_strategy,
            }
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": vdims,
            "in_memory": {"preload_path": preload_path, "index_type": "brute_force"},
        },
        # C2: the runtime backend is write-frozen; non-"never" fails fast at load.
        "write_policy": {"type": "never"},
    }


def isolation_weight_configs(fields: list[str]) -> dict[str, dict[str, float]]:
    """One config per field with that field at weight 1.0 (modality isolation)."""
    return {f"iso_{f}": {f: 1.0} for f in fields}


def grid_weight_configs(fields: list[str], levels=(0.25, 0.5, 0.75)) -> dict[str, dict[str, float]]:
    """Coarse normalized weight grid over the given fields (2-field combos)."""
    configs: dict[str, dict[str, float]] = {}
    for i, fa in enumerate(fields):
        for fb in fields[i + 1:]:
            for la in levels:
                lb = round(1.0 - la, 4)
                cid = f"grid_{fa}@{int(la * 100)}_{fb}@{int(lb * 100)}"
                configs[cid] = {fa: la, fb: lb}
    return configs


def main():
    ap = argparse.ArgumentParser(description="Emit Phase-2 weighted-sum eval YAMLs")
    ap.add_argument("--calibration", required=True, help="calibration_normalizers.json")
    ap.add_argument("--stem", required=True, help="artifact stem key in the calibration JSON")
    ap.add_argument("--preload-path", required=True, help="library .pkl loaded by the server")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["isolation", "grid", "both"], default="both")
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if args.stem not in calib:
        raise SystemExit(f"stem {args.stem!r} not in calibration; have {sorted(calib)}")
    entry = calib[args.stem]
    builder_type = entry["builder_type"]
    vector_dims = entry["vector_dims"]
    fields_calib = entry["fields"]
    useful = [f for f in WEIGHTED_FIELDS if f in fields_calib]

    configs: dict[str, dict[str, float]] = {}
    if args.mode in ("isolation", "both"):
        configs.update(isolation_weight_configs(useful))
    if args.mode in ("grid", "both"):
        configs.update(grid_weight_configs(useful))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid, weights in configs.items():
        cfg = build_eval_config(
            builder_type=builder_type,
            vector_dims=vector_dims,
            preload_path=args.preload_path,
            weights=weights,
            fields_calib=fields_calib,
        )
        yaml_id = f"{args.stem}__{cid}"
        (out_dir / f"{yaml_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote {len(configs)} eval YAMLs to {out_dir}")


if __name__ == "__main__":
    main()
