"""Emit ws2 (text-IVF round) search YAMLs over the full704 bucketed library.

Two arms out of one weight matrix (plan §2-D4 / §3-W2):

- **main** (all 132 cells): the round-1 recipe with three text-IVF keys fixed
  up on the built dict — ``search_strategy.type -> text_ivf_knn``,
  ``backend.in_memory.index_type -> text_ivf`` and
  ``keys.prompt_emb -> {enabled: true, weight: 0}`` (the validator's screening
  requirement; the bucket replaces task scoping, prompt_emb never scores).
- **control** (the 12 manifest ws2c cells only): the round-1 shape verbatim
  (``weighted_score_sum_knn`` + task filter + brute_force) over the SAME
  full704 preload — the matched arm that separates library growth from bucket
  scoping.

Reuses ``exp.weighted_sum.emit_yamls.build_eval_config`` and the round-1
``weight_matrix``/cp3-pin unchanged; the round-1 emitter and its outputs are
not touched. Every emitted yaml is re-loaded through ``load_cache_config`` +
``validate_cache_config`` before the index is written (fail-fast, no partial
trees), and shape invariants are asserted per arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from exp.robocasa365.emit_ws_search_yamls import weight_matrix
from exp.weighted_sum.emit_yamls import build_eval_config

TEACHER = "groot_tp"
CALIB_STEM = "groot_tp_spatial_pool_16_full704"
# The servers resolve this on weilandserver, where /data is local.
PRELOAD = "/data/robocasa365_cache/cache_artifacts_text_ivf/groot_tp_spatial_pool_16_full704.pkl"

# Round-1's real-load-proven cp3 pin (see emit_ws_search_yamls docstring for
# the exact trap it guards against).
CP3_PIN = {"enabled": False, "search_strategy": {"type": "weighted_rrf_knn"}}


def build_cell(weights: dict[str, float], calib_entry: dict, *, text_ivf: bool) -> dict:
    """Build one cell config: the round-1 recipe, plus the text-IVF keys for the main arm."""
    cfg = build_eval_config(
        builder_type=calib_entry["builder_type"],
        vector_dims=calib_entry["vector_dims"],
        preload_path=PRELOAD,
        weights=weights,
        fields_calib=calib_entry["fields"],
    )
    cfg["checkpoints"]["cp3"] = dict(CP3_PIN)
    if text_ivf:
        cfg["checkpoints"]["cp1"]["search_strategy"]["type"] = "text_ivf_knn"
        cfg["backend"]["in_memory"]["index_type"] = "text_ivf"
        cfg["keys"]["prompt_emb"] = {"enabled": True, "weight": 0.0}
    return cfg


def verify_cell(cfg: dict, cid: str, *, text_ivf: bool) -> None:
    """Arm-shape invariants + the real validator on a round-tripped load."""
    ss = cfg["checkpoints"]["cp1"]["search_strategy"]
    weights = {f: k["weight"] for f, k in cfg["keys"].items() if k.get("enabled")}
    assert abs(sum(weights.values()) - 1.0) < 1e-6, f"{cid}: enabled weights must sum to 1"
    if text_ivf:
        assert ss["type"] == "text_ivf_knn", cid
        assert cfg["backend"]["in_memory"]["index_type"] == "text_ivf", cid
        assert cfg["keys"]["prompt_emb"] == {"enabled": True, "weight": 0.0}, cid
    else:
        assert ss["type"] == "weighted_score_sum_knn", cid
        assert cfg["backend"]["in_memory"]["index_type"] == "brute_force", cid
        assert not cfg["keys"]["prompt_emb"]["enabled"], cid
    assert cfg["write_policy"] == {"type": "never"}, cid
    assert cfg["backend"]["vector_dims"]["prompt_emb"] == 2048, cid


def validate_on_disk(path: Path) -> None:
    """Load and validate an emitted YAML through the production loader."""
    from openpi.cache.config import load_cache_config, validate_cache_config

    validate_cache_config(load_cache_config(path))


def emit_arm(
    out_dir: Path, cids: list[str], configs: dict, calib_entry: dict, *, text_ivf: bool
) -> dict:
    """Emit one arm's YAMLs plus its index.json, validating every file as it lands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for cid in cids:
        weights = configs[cid]
        cfg = build_cell(weights, calib_entry, text_ivf=text_ivf)
        verify_cell(cfg, cid, text_ivf=text_ivf)
        path = out_dir / f"{cid}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        validate_on_disk(path)
        index[cid] = {"file": path.name, "weights": weights}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", required=True,
                    help="full704 Phase-1 calibration json (round-1's n5 json is stale)")
    ap.add_argument("--manifest", required=True,
                    help="selection_manifest.json with the ws2c segment (control-arm cells)")
    ap.add_argument("--out-root", default="exp/robocasa365/config/ws_search2")
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text())
    calib_entry = calib[CALIB_STEM]
    manifest = json.loads(Path(args.manifest).read_text())
    control_cells = manifest["segments"]["ws2c"]["cells"]

    configs = weight_matrix()
    unknown = sorted(set(control_cells) - set(configs))
    if unknown:
        raise SystemExit(f"manifest ws2c cells not in the weight matrix: {unknown}")

    out_root = Path(args.out_root) / TEACHER
    main_index = emit_arm(out_root / "main", sorted(configs), configs, calib_entry, text_ivf=True)
    ctrl_index = emit_arm(out_root / "control", list(control_cells), configs, calib_entry, text_ivf=False)
    assert len(main_index) == 132, f"main arm must be 132 cells, got {len(main_index)}"
    assert len(ctrl_index) == 12, f"control arm must be 12 cells, got {len(ctrl_index)}"
    print(f"[emit] main={len(main_index)} control={len(ctrl_index)} -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
