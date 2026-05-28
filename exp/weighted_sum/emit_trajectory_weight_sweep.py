"""Emit a spatial_16 weight-sweep × depth grid to re-search weights under the
trajectory regime.

Follow-up to the trajectory experiment: the global top-10 (all spatial_16 /
max_pool) regressed −6.4pp under trajectory. This sweep tests whether that is
(H-3a) the d1-tuned weights being overfit to the single-step regime — curable by
re-searching weights at each depth — or (H-mechanism) an intrinsic penalty of
trajectory smoothing on an already-precise single-step index.

Only ``cp1_spatial_pool_16`` (the top-10 workhorse). For each weight triple on a
densified ``robot_state``-dominant simplex × each depth in {1,3,4,5}, emit one
eval YAML. **depth 1 is included on purpose**: the new grid points are not in the
original weighted-sum sweep, so each needs its own unbiased d1 baseline to judge
whether the optimal weight vector drifts with depth. depth 6 is dropped (known to
regress). Everything else (zscore Layer-1, always_hit, prompt_emb mask,
write_policy=never, in_memory) is inherited verbatim via build_eval_config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from openpi.cache.config import load_cache_config

from exp.weighted_sum.emit_yamls import build_eval_config, grid3_weight_configs
from exp.weighted_sum.emit_trajectory_yamls import DEPTH_WEIGHTS

_STEM = "cp1_spatial_pool_16"
_FIELDS = ("vision_0", "vision_1", "robot_state")


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Emit spatial_16 weight-sweep × depth trajectory YAMLs")
    ap.add_argument("--calibration", default=str(repo / "exp/weighted_sum/data/calibration_normalizers.json"))
    ap.add_argument("--artifact-dir", default="exp/common/data/cache_artifacts/libero_spatial",
                    help="relative — server resolves against its own CWD")
    ap.add_argument("--stem", default=_STEM)
    ap.add_argument("--output-dir", default=str(repo / "exp/weighted_sum/config/trajectory_wsweep"))
    ap.add_argument("--depths", default="1,3,4,5")
    ap.add_argument("--step", type=float, default=0.0625, help="simplex grid step (1/16 default)")
    ap.add_argument("--dom-min", type=float, default=0.1875, help="robot_state lower bound on the grid")
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if args.stem not in calib:
        raise SystemExit(f"stem {args.stem!r} not in calibration; have {sorted(calib)}")
    entry = calib[args.stem]

    depths = [int(d) for d in args.depths.split(",") if d != ""]
    for d in depths:
        if d != 1 and d not in DEPTH_WEIGHTS:
            raise SystemExit(f"no trajectory_weights for depth {d}; have d=1 or {sorted(DEPTH_WEIGHTS)}")

    weight_cfgs = grid3_weight_configs(list(_FIELDS), step=args.step, dom_min=args.dom_min)
    print(f"weight configs: {len(weight_cfgs)} (step={args.step}, rs>={args.dom_min}); depths={depths}")
    print(f"=> total yaml = {len(weight_cfgs) * len(depths)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid, weights in weight_cfgs.items():
        for depth in depths:
            kw = {} if depth == 1 else {"trajectory_depth": depth, "trajectory_weights": DEPTH_WEIGHTS[depth]}
            cfg = build_eval_config(
                builder_type=entry["builder_type"],
                vector_dims=entry["vector_dims"],
                preload_path=f"{args.artifact_dir}/{args.stem}.pkl",
                weights=weights,
                fields_calib=entry["fields"],
                **kw,
            )
            yaml_id = f"{args.stem}__{cid}__d{depth}"
            path = out_dir / f"{yaml_id}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            load_cache_config(path)  # schema self-check (no pkl load)
            n += 1

    expected = len(weight_cfgs) * len(depths)
    assert n == expected, f"wrote {n} != expected {expected}"
    print(f"Wrote {n} YAMLs to {out_dir}")


if __name__ == "__main__":
    main()
