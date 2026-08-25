"""Emit the GR00T×LIBERO weighted-sum search matrix.

Mirrors the Pi0.5 LIBERO line (``exp/weighted_sum``) and its RoboCasa365 port,
reusing ``build_eval_config`` verbatim so the recipe shape stays identical:
``always_search`` + ``always_hit`` + ``top_k=1`` + per-field Layer-1 normalizers
+ ``write_policy: never``. ``always_hit`` is what isolates retrieval quality --
with no teacher fallback the success rate is a pure function of what came back.

Two differences from both predecessors:

*   **Three weighted fields, not four.** LIBERO GR00T feeds two cameras
    (``video.image`` / ``video.wrist_image``); ``prompt_emb`` is dropped as it is
    for every LIBERO run (task-constant within an episode).
*   **A closed simplex instead of named families.** With three fields the whole
    weight space fits in one uniform grid, so the isolation corners and the
    two-field edges are grid points rather than separate families. That matters
    here: the RoboCasa365 result put the useful region *on an edge* (two fields
    carrying everything), which an interior-only grid would have missed.

``trajectory_depth`` stays 1 -- single-step retrieval only.

Usage::

    uv run exp/libero_groot/emit_search_yamls.py \
        --calibration /data/libero_cache/calib_input/libero_spatial_calibration.json \
        --stem libero_spatial_sp16_S6 \
        --preload /data/libero_cache/libraries/libero_spatial/libero_spatial_sp16_S6.pkl \
        --out-dir exp/libero_groot/config/search/libero_spatial
"""

from __future__ import annotations

import argparse
import json
import pathlib

import yaml

from exp.weighted_sum.emit_yamls import build_eval_config

# Weighted fields for LIBERO GR00T: two cameras + proprioception.
FIELDS: tuple[str, ...] = ("vision_0", "vision_1", "robot_state")
_SHORT = {"vision_0": "v0", "vision_1": "v1", "robot_state": "rs"}
# Set by main() from --field-min / --field-max; None means the whole simplex.
_REGION: tuple[tuple[int, ...], tuple[int, ...]] | None = None


def in_region(units: tuple[int, ...], lo: tuple[int, ...], hi: tuple[int, ...]) -> bool:
    """Per-field unit bounds, inclusive. Used to aim a fine grid at a region the
    coarse round already localised, instead of spending it where the coarse
    round showed the cells are statistically indistinguishable."""
    return all(low <= u <= high for u, low, high in zip(units, lo, hi, strict=True))


def closed_simplex(fields: tuple[str, ...], steps: int) -> dict[str, dict[str, float]]:
    """Every weighting on the ``steps``-spaced simplex, edges and corners included.

    Enumerated in integer units so the weights sum to exactly 1 in the units and
    the cell id is exact; the float division happens once, at the end.
    """
    configs: dict[str, dict[str, float]] = {}
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            units = (a, b, steps - a - b)
            if _REGION is not None and not in_region(units, *_REGION):
                continue
            weights = {f: u / steps for f, u in zip(fields, units, strict=True)}
            cid = "_".join(f"{_SHORT[f]}@{u}" for f, u in zip(fields, units, strict=True))
            configs[cid] = weights
    return configs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--stem", required=True, help="Artifact stem key inside the calibration JSON")
    ap.add_argument("--preload", required=True, help="Library path as the SERVER resolves it")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=16, help="Simplex resolution (1/steps per axis)")
    ap.add_argument("--field-min", default=None,
                    help="Per-field lower bound in units, e.g. '2,2,1' for v0,v1,rs.")
    ap.add_argument("--field-max", default=None,
                    help="Per-field upper bound in units, e.g. '9,9,4'.")
    args = ap.parse_args()

    if args.field_min or args.field_max:
        global _REGION  # noqa: PLW0603 - module-level knob read by closed_simplex
        lo = tuple(int(x) for x in (args.field_min or "0,0,0").split(","))
        hi = tuple(int(x) for x in (args.field_max or f"{args.steps},{args.steps},{args.steps}").split(","))
        _REGION = (lo, hi)

    calib = json.loads(pathlib.Path(args.calibration).read_text())
    if args.stem not in calib:
        raise SystemExit(f"stem {args.stem!r} not in calibration; have {sorted(calib)}")
    entry = calib[args.stem]

    configs = closed_simplex(FIELDS, args.steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for cid, weights in configs.items():
        cfg = build_eval_config(
            builder_type=entry["builder_type"],
            vector_dims=entry["vector_dims"],
            preload_path=args.preload,
            weights=weights,
            fields_calib=entry["fields"],
        )
        # A cp3 block that is PRESENT but lacks ``search_strategy.type`` defaults
        # to the qdrant strategy, and the validator type-checks even disabled
        # checkpoints -- rejected against the in_memory backend. This pinned
        # disabled form is the shape that passed a real server load.
        cfg["checkpoints"]["cp3"] = {
            "enabled": False,
            "search_strategy": {"type": "weighted_rrf_knn"},
        }
        (args.out_dir / f"{cid}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        index[cid] = {"file": f"{cid}.yaml", "weights": weights}

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    print(f"wrote {len(index)} yamls to {args.out_dir} (builder={entry['builder_type']})")


if __name__ == "__main__":
    main()
