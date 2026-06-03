"""Generate round-N refinement YAMLs around the current best region.

Reads a ``summarize`` results.json + the Phase-1 calibration artifact, picks the
best keybuilder (highest single-yaml success_rate), and emits a denser weight
grid around the rs-dominant optimum plus optional shortlist-2nd-normalizer
variants. Reuses :func:`exp.weighted_sum.emit_yamls.build_eval_config` so every
C2 / prompt_emb-mask / vector_dims-full-set / write_policy invariant is kept
identical to the baseline (no server restart needed — the conductor's
``load_cache_config`` hot-swaps the bundle).

Strategy (round 1, but parameters generalize to rounds 2-5):
  - lock the winning keybuilder;
  - dense rs-dominant 3-field simplex at ``step`` (default 1/16) over an rs band
    that brackets the baseline top configs, keeping only the *new* interpolation
    points the 1/8 baseline grid did not cover (>=1 weight on an odd 1/16 unit);
  - dense rs-dominant 2-field grid (single vision + robot_state);
  - for the top-K baseline weight configs, a ``__norm2`` variant that swaps each
    weighted field's normalizer to its calibration shortlist 2nd candidate.

Usage:
    python -m exp.weighted_sum.refine_round \
        --results exp/weighted_sum/data/libero_spatial/phase2/results_preview.json \
        --calibration exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json \
        --preload-path exp/common/data/cache_artifacts/libero_spatial/cp1_max_pool.pkl \
        --output-dir exp/weighted_sum/config/round_1 \
        --round 1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from exp.weighted_sum.emit_yamls import build_eval_config

_FIELDS3 = ("vision_0", "vision_1", "robot_state")


def pick_best_keybuilder(results: dict) -> tuple[str, list[tuple[float, str, dict]]]:
    """Return (best_stem, sorted [(sr, yaml_id, weights)] of that stem)."""
    by_kb: dict[str, list] = defaultdict(list)
    for yid, v in results.items():
        if v.get("n", 0) < 50:  # ignore under-sampled
            continue
        stem = yid.split("__")[0]
        by_kb[stem].append((v["success_rate"], yid, _parse_weights(yid)))
    best_stem = max(by_kb, key=lambda s: max(r for r, _, _ in by_kb[s]))
    return best_stem, sorted(by_kb[best_stem], reverse=True)


def rank_keybuilder(results: dict, stem: str) -> list[tuple[float, str, dict]]:
    """Return sorted ``[(sr, yaml_id, weights)]`` for a SPECIFIC keybuilder stem.

    Companion to :func:`pick_best_keybuilder` for the winner-tie case: when
    Stage 1's top-2 keybuilders tie within 1 SE the plan refines BOTH, forcing
    each via ``--stem``. ``main`` must then center / norm2 on the forced stem's
    OWN ranked list. Without this, ``main`` keeps the global-best stem's ranked
    list even when ``--stem`` forces a different keybuilder, refining the wrong
    neighborhood and applying norm2 to the wrong keybuilder's top weights
    (plan §5 / §9, G1 R2 Item1).
    """
    ranked = sorted(
        (
            (v["success_rate"], yid, _parse_weights(yid))
            for yid, v in results.items()
            if v.get("n", 0) >= 50 and yid.split("__")[0] == stem
        ),
        reverse=True,
    )
    if not ranked:
        raise ValueError(f"no configs (n>=50) for stem {stem!r} in results")
    return ranked


def _parse_weights(yaml_id: str) -> dict[str, float]:
    w: dict[str, float] = {}
    cfg = yaml_id.split("__", 1)[1] if "__" in yaml_id else yaml_id
    for field, pct in re.findall(r"(vision_0|vision_1|robot_state)@(\d+)", cfg):
        w[field] = int(pct) / 100.0
    return w


def _cid(weights: dict[str, float]) -> str:
    # Use the SAME percent truncation as emit_yamls.py (``int(x * 100)``), not
    # rounding: emit_yamls generated the baseline yaml stems and the data CSVs
    # are keyed by those stems, so a value like 0.375 must encode to "37" here
    # exactly as it did there ("38" under round() would break the join and the
    # center-mode reuse of baseline optima).
    parts = [f"{f}@{int(weights[f] * 100)}" for f in _FIELDS3 if weights.get(f, 0) > 0]
    return "grid_" + "_".join(parts) if len(parts) == 2 else "grid3_" + "_".join(parts)


def dense_grid3(step: float, rs_lo: float, rs_hi: float) -> list[dict[str, float]]:
    """rs-dominant 3-field simplex at ``step``, keeping only points NOT on the
    1/8 baseline grid (at least one unit odd in 1/16 units)."""
    n = round(1.0 / step)            # total units (16 for step 1/16)
    out = []
    for rs_u in range(round(rs_lo * n), round(rs_hi * n) + 1):
        vtot = n - rs_u
        for v0_u in range(1, vtot):
            v1_u = vtot - v0_u
            # skip baseline 1/8-grid points (all three even in 1/16 units)
            if v0_u % 2 == 0 and v1_u % 2 == 0 and rs_u % 2 == 0:
                continue
            out.append({
                "vision_0": round(v0_u * step, 4),
                "vision_1": round(v1_u * step, 4),
                "robot_state": round(rs_u * step, 4),
            })
    return out


def dense_2field(step: float, rs_lo: float, rs_hi: float) -> list[dict[str, float]]:
    """rs-dominant 2-field (single vision + robot_state) at ``step``, new points."""
    n = round(1.0 / step)
    out = []
    for rs_u in range(round(rs_lo * n), round(rs_hi * n) + 1):
        v_u = n - rs_u
        if v_u <= 0:
            continue
        if v_u % 2 == 0 and rs_u % 2 == 0:
            continue  # baseline 1/8 point
        for vfield in ("vision_0", "vision_1"):
            out.append({vfield: round(v_u * step, 4), "robot_state": round(rs_u * step, 4)})
    return out


def local_dense_around(centers: list[dict], step: float, radius_units: int) -> list[dict]:
    """3-field simplex points in a +/- radius_units neighborhood of each center
    weight config (in 1/step units), INCLUDING the center itself (so the round
    re-runs the baseline optimum to confirm SR vs statistical noise). Dedups."""
    n = round(1.0 / step)
    out, seen = [], set()
    for c in centers:
        c0, c1 = round(c.get("vision_0", 0) * n), round(c.get("vision_1", 0) * n)
        for d0 in range(-radius_units, radius_units + 1):
            for d1 in range(-radius_units, radius_units + 1):
                v0, v1 = c0 + d0, c1 + d1
                rs = n - v0 - v1
                if v0 >= 1 and v1 >= 1 and rs >= 1 and (v0, v1, rs) not in seen:
                    seen.add((v0, v1, rs))
                    out.append({"vision_0": round(v0 * step, 4), "vision_1": round(v1 * step, 4),
                                "robot_state": round(rs * step, 4)})
    return out


def _fields_calib(entry: dict, *, variant: bool) -> dict:
    """Build the fields_calib dict for build_eval_config. ``variant`` swaps each
    field's selected normalizer to its shortlist 2nd candidate."""
    out = {}
    for f, fc in entry["fields"].items():
        sel = fc["selected"]
        if variant:
            sl = fc.get("shortlist") or []
            if len(sl) >= 2:
                sel = sl[1]
        out[f] = {"sim_type": fc["sim_type"], "selected": {"method": sel["method"], "params": sel["params"]}}
    return out


def main():
    ap = argparse.ArgumentParser(description="Emit round-N refinement YAMLs")
    ap.add_argument("--results", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--preload-path", required=True, help="relative library .pkl path for the winning keybuilder")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--stem", default="", help="force a keybuilder stem; default = best from results")
    ap.add_argument("--step", type=float, default=1 / 16)
    ap.add_argument("--rs-lo", type=float, default=0.3125)
    ap.add_argument("--rs-hi", type=float, default=0.6875)
    ap.add_argument("--rs-lo-2f", type=float, default=0.5)
    ap.add_argument("--rs-hi-2f", type=float, default=0.8125)
    ap.add_argument("--topk-norm2", type=int, default=6, help="apply shortlist-2nd normalizer to top-K baseline configs")
    ap.add_argument("--mode", choices=["dense", "center"], default="dense",
                    help="dense=global rs-band interpolation grid; center=local dense neighborhood around the top-K configs (re-runs them to confirm SR)")
    ap.add_argument("--center-topk", type=int, default=3, help="center mode: number of top configs to center neighborhoods on")
    ap.add_argument("--center-radius", type=int, default=2, help="center mode: neighborhood radius in 1/step units")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    calib = json.loads(Path(args.calibration).read_text())

    if args.stem:
        # Forced stem (winner-tie dual refine): rank THAT stem's own configs so
        # center/norm2 below operate on the forced keybuilder's optimum, not the
        # global-best stem's (G1 R2 Item1).
        stem = args.stem
        ranked = rank_keybuilder(results, stem)
    else:
        stem, ranked = pick_best_keybuilder(results)
    entry = calib[stem]
    builder_type, vector_dims = entry["builder_type"], entry["vector_dims"]
    fc_sel = _fields_calib(entry, variant=False)
    fc_v2 = _fields_calib(entry, variant=True)

    print(f"[refine] round={args.round} using stem={stem} ({'forced --stem' if args.stem else 'best-of-results'})")
    print(f"[refine] top baseline configs of {stem}:")
    for sr, yid, w in ranked[:5]:
        print(f"  {100*sr:.0f}%  {yid.split('__',1)[1]}")

    # weight configs
    weight_cfgs: dict[str, dict] = {}
    if args.mode == "center":
        centers = [w for _, _, w in ranked[:args.center_topk] if w]
        print(f"[refine] center mode: {len(centers)} centers, radius={args.center_radius} units (step={args.step:.4f})")
        for w in local_dense_around(centers, args.step, args.center_radius):
            weight_cfgs[_cid(w)] = w
    else:
        for w in dense_grid3(args.step, args.rs_lo, args.rs_hi):
            weight_cfgs[_cid(w)] = w
        for w in dense_2field(args.step, args.rs_lo_2f, args.rs_hi_2f):
            weight_cfgs[_cid(w)] = w

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cid, weights in weight_cfgs.items():
        cfg = build_eval_config(
            builder_type=builder_type, vector_dims=vector_dims,
            preload_path=args.preload_path, weights=weights, fields_calib=fc_sel,
        )
        (out_dir / f"{stem}__{cid}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        written += 1

    # normalizer-2nd-candidate variants on the top-K baseline weight configs
    for sr, yid, weights in ranked[:args.topk_norm2]:
        if not weights:
            continue
        cfg = build_eval_config(
            builder_type=builder_type, vector_dims=vector_dims,
            preload_path=args.preload_path, weights=weights, fields_calib=fc_v2,
        )
        (out_dir / f"{stem}__{_cid(weights)}__norm2.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        written += 1

    print(f"[refine] wrote {written} YAMLs to {out_dir} ({len(weight_cfgs)} weight-grid + {min(args.topk_norm2, len(ranked))} norm2)")


if __name__ == "__main__":
    main()
