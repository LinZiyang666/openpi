"""Emit trajectory-search eval YAMLs on top of the weighted-sum best configs.

Layers ``trajectory_depth > 1`` (multi-step query-history aggregation) onto the
18 base configs selected from the weighted-sum Phase-2 results, mirroring what
the old trajectory experiment did to Phase 1 — but with the two-layer
``weighted_score_sum_knn`` retrieval as the base instead of ``weighted_rrf_knn``.

Two base groups (run together, deduplicated):

- **Group 1** (per-keybuilder, old-experiment selection): for each of the 4
  CP1 keybuilders, take top-1 + top-2 + second-worst. "Second-worst" uses the
  **A criterion** (owner decision): regular weight grid only, same ``zscore``
  normalizer — ``__norm2`` (normalizer swap) and ``iso_`` (single-modality) are
  excluded so the only variable is the weight vector.
- **Group 2**: the global SR top-10 (``config/top10/``).

The two groups overlap on 4 configs (spatial_16 / max_pool top-1 + top-2 are all
in the top-10); the union is 18 distinct base configs. Each (base, depth) is
emitted exactly once; the 4 dual-role configs are referenced by both group
lenses at analysis time without rerunning episodes.

The base selection is recomputed programmatically from the suite-specific
``data/<suite>/phase2/all_results.csv`` (not a hand-written list) and
cross-checked against ``config/top10/<suite>/`` so the dedup stays reproducible
despite tied SRs.

Each YAML reuses :func:`exp.weighted_sum.emit_yamls.build_eval_config` (the
validated weighted-sum builder, unchanged) with ``trajectory_depth`` /
``trajectory_weights`` set; the Layer-1 ``zscore`` normalizers, ``always_hit``
judge, ``prompt_emb`` mask and ``write_policy: never`` are all inherited
verbatim. yaml_id = ``<base_id>__d{depth}``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import statistics
from pathlib import Path

import yaml

from openpi.cache.config import load_cache_config

from exp.weighted_sum.emit_yamls import build_eval_config

# ----------------------------------------------------------------------
# Trajectory schedule — fixed decreasing weights reused from the old
# trajectory experiment (newest-first, sums to 1).
# ----------------------------------------------------------------------
DEPTH_WEIGHTS: dict[int, list[float]] = {
    3: [0.5, 0.3, 0.2],
    4: [0.4, 0.3, 0.2, 0.1],
    5: [0.35, 0.25, 0.2, 0.12, 0.08],
    6: [0.3, 0.25, 0.2, 0.12, 0.08, 0.05],
}

_KEYBUILDERS = ("cp1_spatial_pool_16", "cp1_max_pool", "cp1_mean_pool", "cp1_spatial_pool_64")
_WEIGHT_FIELDS = ("vision_0", "vision_1", "vision_2", "robot_state")

# Reference invariants for the libero_spatial run only. The base-union size and
# the group1/top10 overlap are arithmetic facts of a SPECIFIC results table, not
# a cross-suite invariant: on libero_10 the overlap is almost never 4 and N_base
# is therefore data-dependent (see plan §3.1 / §6a). Kept for documentation; the
# emit-time check below uses suite-agnostic bounds, not these exact values.
_LIBERO_SPATIAL_BASE = 18
_LIBERO_SPATIAL_OVERLAP = 4


# ----------------------------------------------------------------------
# Base-config selection (recomputed from the Phase-2 results CSV)
# ----------------------------------------------------------------------
def _parse_weights(yaml_id: str) -> dict[str, float]:
    """Parse ``field@<pct>`` tokens from a yaml_id into fractional weights."""
    weights: dict[str, float] = {}
    for field in _WEIGHT_FIELDS:
        m = re.search(rf"{field}@(\d+)", yaml_id)
        if m:
            weights[field] = int(m.group(1)) / 100.0
    if not weights:
        raise ValueError(f"no weight tokens parsed from yaml_id {yaml_id!r}")
    return weights


def select_base_configs(results_csv: Path, top10_dir: Path) -> tuple[
    list[tuple[str, str, str, float]], set[str], set[str]
]:
    """Recompute the (data-dependent) base union from the results CSV + top10 dir.

    Returns ``(group1, top10_ids, base_ids)`` where ``group1`` is a list of
    ``(keybuilder, role, yaml_id, mean_sr)`` (up to 4 keybuilders x 3 roles) and
    ``base_ids`` is the deduplicated union of group1 + top10. The union size
    ``N_base`` is suite-dependent (the libero_spatial run happened to give 18
    with overlap 4 — not a cross-suite invariant; see plan §3.1).
    """
    rows = list(csv.DictReader(results_csv.open()))
    agg: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for r in rows:
        agg[(r["keybuilder"], r["yaml_id"])].append(float(r["success_rate"]))
    mean_sr = {k: statistics.mean(v) for k, v in agg.items()}

    top10_ids = {p.stem for p in top10_dir.glob("*.yaml")}
    # Option A invariant (G1 R3 / G2 R1 Item2): the top10 dir must hold EXACTLY
    # 10 yamls. emit_top10.py guarantees this; assert here so a stale / short /
    # miscounted dir can't silently change N_base downstream.
    if len(top10_ids) != 10:
        raise ValueError(
            f"top10 dir must contain exactly 10 yamls (Option A), found "
            f"{len(top10_ids)} under {top10_dir} — re-run emit_top10.py"
        )

    group1: list[tuple[str, str, str, float]] = []
    for kb in _KEYBUILDERS:
        ranked = sorted(
            ((y, sr) for (k, y), sr in mean_sr.items() if k == kb),
            key=lambda x: -x[1],
        )
        # A criterion: regular grid only — drop normalizer-swap and single-modality.
        reg = [(y, sr) for y, sr in ranked if "__norm2" not in y and "__iso_" not in y]
        if len(reg) < 3:
            raise ValueError(f"{kb}: fewer than 3 regular-grid configs ({len(reg)})")
        for role, (y, sr) in zip(("top1", "top2", "2nd-worst"), (reg[0], reg[1], reg[-2])):
            group1.append((kb, role, y, sr))

    base_ids = {g[2] for g in group1} | top10_ids
    return group1, top10_ids, base_ids


# ----------------------------------------------------------------------
# Emission
# ----------------------------------------------------------------------
def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Emit trajectory eval YAMLs over weighted-sum best configs")
    ap.add_argument(
        "--calibration",
        default=str(repo / "exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json"),
    )
    # Relative on purpose: the server resolves preload_path against its own CWD
    # (the openpi repo root on each machine), so the YAMLs stay portable across
    # client/server hosts. Matches the existing top10/round YAML convention.
    ap.add_argument("--artifact-dir", default="exp/common/data/cache_artifacts/libero_spatial")
    ap.add_argument(
        "--results-csv",
        default=str(repo / "exp/weighted_sum/data/libero_spatial/phase2/all_results.csv"),
    )
    ap.add_argument("--top10-dir", default=str(repo / "exp/weighted_sum/config/top10/libero_spatial"))
    ap.add_argument("--output-dir", default=str(repo / "exp/weighted_sum/config/trajectory/libero_spatial"))
    ap.add_argument("--depths", default="3,4,5,6", help="comma-separated trajectory depths")
    args = ap.parse_args()

    import json

    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    depths = [int(d) for d in args.depths.split(",") if d != ""]
    for d in depths:
        if d not in DEPTH_WEIGHTS:
            raise SystemExit(f"no trajectory_weights schedule for depth {d}; have {sorted(DEPTH_WEIGHTS)}")

    group1, top10_ids, base_ids = select_base_configs(Path(args.results_csv), Path(args.top10_dir))

    # ── Dedup assertions (Reviewer G1 R1 suggestion: explicit + printed) ──
    group1_ids = {g[2] for g in group1}
    overlap = group1_ids & top10_ids
    print("=== base selection ===")
    print(f"  group1 logical entries : {len(group1)} (4 keybuilders x 3 roles)")
    print(f"  group1 distinct ids    : {len(group1_ids)}")
    print(f"  top10 ids              : {len(top10_ids)}")
    print(f"  group1 ∩ top10 overlap : {len(overlap)} -> {sorted(overlap)}")
    print(f"  deduplicated base union: {len(base_ids)}")
    for kb, role, y, sr in group1:
        print(f"    {kb:22} {role:10} {sr * 100:4.0f}%  {'[DUP]' if y in top10_ids else '[NEW]'}  {y}")
    # Suite-agnostic sanity bounds (G1 R2 Item3): N_base is data-dependent
    # (the libero_spatial run gave 18 / overlap 4 — NOT a cross-suite invariant).
    # top10 is pinned to EXACTLY 10 inside select_base_configs (G2 R1 Item2);
    # group1 contributes up to 4 keybuilders x 3 roles distinct ids.
    assert len(group1_ids) <= 4 * 3, f"group1 distinct ids {len(group1_ids)} > 12 (4kb x 3roles)"
    print(f"  -> N_base = {len(base_ids)} (libero_spatial was {_LIBERO_SPATIAL_BASE}; suite-dependent)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for base_id in sorted(base_ids):
        stem, sep, cid = base_id.partition("__")
        if not sep:
            raise ValueError(f"base_id {base_id!r} has no '__' keybuilder/config separator")
        if stem not in calib:
            raise SystemExit(f"stem {stem!r} (from {base_id}) not in calibration; have {sorted(calib)}")
        entry = calib[stem]
        weights = _parse_weights(base_id)
        preload_path = str(Path(args.artifact_dir) / f"{stem}.pkl")

        for depth in depths:
            cfg = build_eval_config(
                builder_type=entry["builder_type"],
                vector_dims=entry["vector_dims"],
                preload_path=preload_path,
                weights=weights,
                fields_calib=entry["fields"],
                trajectory_depth=depth,
                trajectory_weights=DEPTH_WEIGHTS[depth],
            )
            yaml_id = f"{base_id}__d{depth}"
            path = out_dir / f"{yaml_id}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            # Schema self-check: load_cache_config runs validate_cache_config
            # (trajectory_depth/weights legality, per_field zscore, in_memory,
            # write_policy=never). Pure schema — does not load the pkl.
            load_cache_config(path)
            n_written += 1

    expected = len(base_ids) * len(depths)
    assert n_written == expected, f"wrote {n_written} != expected {expected}"
    print(f"\nWrote {n_written} trajectory YAMLs to {out_dir} ({len(base_ids)} base x {len(depths)} depths)")


if __name__ == "__main__":
    main()
