"""Emit round-1 weighted-sum search YAMLs for the RoboCasa365 pure-cache search.

Mirrors the earliest LIBERO weighted-sum search (``exp/weighted_sum``) with the
keybuilder axis removed: the builder is frozen per teacher
(``cp1_spatial_pool_16`` / ``cp1_groot_spatial_pool_16``), the library is the
T6 n5 artifact, and the weight matrix is the same three families over four
fields (vision_0/1/2, robot_state): iso 4 + grid2 C(4,2)*7=42 + grid3
rs-dominant 30 = 76 cells per teacher.

Reuses ``exp.weighted_sum.emit_yamls.build_eval_config`` verbatim so the
recipe shape (always_search + always_hit + top_k 1 + per_field normalization +
write_policy never + timer off) stays aligned with the LIBERO line, then
applies two RoboCasa-specific fixups:

- an explicit DISABLED cp3 block pinned to ``weighted_rrf_knn``. The trap it
  guards against (review-verified 2026-08-21): a cp3 block that is PRESENT but
  lacks ``search_strategy.type`` defaults to the qdrant strategy and the
  validator type-checks even disabled checkpoints — rejected against the
  in_memory backend (what ``groot_cache_smoke_n5.yaml`` hit on its first real
  load). Omitting cp3 entirely would also validate; the pinned form is kept
  because it is the shape that passed a real server load in the T7 smoke;
- ``preload_path`` pointing at the absolute /data path of the T6 n5 artifact
  (the servers resolve it on weilandserver, where /data is local).

Note: the ``to_similarity: {type: exp, tau: 1.0}`` block on ``robot_state``
(inherited verbatim from the LIBERO emitter) is inert in the per_field
weighted-score-sum path — the zscore normalizer orients raw l2 distances as
``-d`` internally; nothing applies ``exp(-d/tau)`` first.

Usage::

    uv run exp/robocasa365/emit_ws_search_yamls.py \
        --calibration exp/robocasa365/config/ws_search/calibration_normalizers.json \
        --out-root exp/robocasa365/config/ws_search
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from exp.weighted_sum.emit_yamls import (
    build_eval_config,
    grid3_weight_configs,
    grid_weight_configs,
    isolation_weight_configs,
)

# Weighted fields for RoboCasa365 (3 cameras + proprioception); prompt_emb is
# dropped from the search exactly as in LIBERO plan D7 (task-constant field).
FIELDS = ["vision_0", "vision_1", "vision_2", "robot_state"]

# Teacher -> (calibration stem, preload path as the server resolves it).
TEACHERS = {
    "groot_tp": {
        "stem": "groot_tp_spatial_pool_16_n5",
        "preload": "/data/robocasa365_cache/cache_artifacts_l1s1/groot_tp_spatial_pool_16_n5.pkl",
    },
    "pi05": {
        "stem": "pi05_spatial_pool_16_n5",
        "preload": "/data/robocasa365_cache/cache_artifacts_l1s1/pi05_spatial_pool_16_n5.pkl",
    },
}


def _simplex(fields: tuple[str, ...], *, step: float = 0.125, prefix: str) -> dict[str, dict[str, float]]:
    """All positive weightings of `fields` on a `step`-spaced simplex (sum 1)."""
    from itertools import product

    n = round(1.0 / step)
    configs: dict[str, dict[str, float]] = {}
    k = len(fields)
    for units in product(range(1, n - k + 2), repeat=k - 1):
        last = n - sum(units)
        if last < 1:
            continue
        weights = {f: round(u * step, 4) for f, u in zip(fields, (*units, last))}
        cid = prefix + "_" + "_".join(f"{f}@{int(w * 100)}" for f, w in weights.items())
        configs[cid] = weights
    return configs


def weight_matrix() -> dict[str, dict[str, float]]:
    """The round-1 matrix.

    Families (all statically sized):
    - iso 4 + grid2 42 + grid3(rs-dominant) 30: the LIBERO-mirrored core.
    - grid3v 21: the {v0,v1,v2} three-camera simplex — LIBERO's grid3 WAS its
      full field set, so a faithful 4-field port must cover the all-camera
      face too (owner correction 2026-08-22).
    - grid4 35: the full four-field simplex interior.
    """
    configs: dict[str, dict[str, float]] = {}
    configs.update(isolation_weight_configs(FIELDS))
    configs.update(grid_weight_configs(FIELDS))
    configs.update(grid3_weight_configs(FIELDS, dominant="robot_state"))
    configs.update(_simplex(("vision_0", "vision_1", "vision_2"), prefix="grid3v"))
    configs.update(_simplex(("vision_0", "vision_1", "vision_2", "robot_state"), prefix="grid4"))
    return configs


def emit_teacher(teacher: str, calib: dict, out_root: Path) -> list[str]:
    spec = TEACHERS[teacher]
    entry = calib[spec["stem"]]
    out_dir = out_root / teacher
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = weight_matrix()
    index = {}
    for cid, weights in configs.items():
        cfg = build_eval_config(
            builder_type=entry["builder_type"],
            vector_dims=entry["vector_dims"],
            preload_path=spec["preload"],
            weights=weights,
            fields_calib=entry["fields"],
        )
        # Pin a disabled, in_memory-compatible cp3 explicitly (see module
        # docstring for the exact trap; this is the real-load-proven shape).
        cfg["checkpoints"]["cp3"] = {
            "enabled": False,
            "search_strategy": {"type": "weighted_rrf_knn"},
        }
        path = out_dir / f"{cid}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        index[cid] = {"file": path.name, "weights": weights}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    return sorted(index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--calibration",
        default="exp/robocasa365/config/ws_search/calibration_normalizers.json",
    )
    ap.add_argument("--out-root", default="exp/robocasa365/config/ws_search")
    ap.add_argument("--teachers", default="groot_tp,pi05")
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text())
    out_root = Path(args.out_root)
    for teacher in args.teachers.split(","):
        cids = emit_teacher(teacher, calib, out_root)
        print(f"{teacher}: wrote {len(cids)} yamls to {out_root / teacher}")


if __name__ == "__main__":
    main()
