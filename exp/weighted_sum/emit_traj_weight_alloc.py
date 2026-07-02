"""Emit the trajectory step-weight screening search over the weighted-sum bases.

Fixes each depth's own base (modality weights + Layer-1 zscore normalizers +
keybuilder + ``always_hit`` judge + ``write_policy: never``) and sweeps ONLY the
per-step ``trajectory_weights``. Under ``always_hit`` the judge always replays the
top-1 entry, so only the RANKING (argmax) affects success rate — the search is
therefore over the relative *shape* of the step weights.

Search set = union of three families (canonicalized to 6 decimals, deduped):

- **S1** current-step-dominant x tail-shape gradient: ``c in {0.2..0.9}`` (8) x
  geometric tail ratio ``q in {0.5, 1.0, 2.0}`` (3). ``w[0]=c``;
  ``w[1+j] = (1-c)*q**j / sum_k q**k``. Reaches the near-boundary (``c=0.9``)
  region where a trajectory-hurts optimum would live; also supplies the d5
  ``uniform`` point (``c=0.2, q=1``). The d4 uniform (``.25x4``) is NOT in S1
  (``c=0.25`` is not on the c-grid) — it comes from S2; d3 has no exact uniform.
- **S2** interior shape lattice: positive-integer compositions of ``n=1/step``
  into ``depth`` parts, divided by ``n`` (``step`` = 0.1 for d3, 0.125 for d4/d5).
  Covers the monotone-increasing family, single peaks, U shapes, balanced
  interior. NOTE: the d5 uniform ``.2x5`` is NOT on the 8-into-5 lattice; it comes
  from S1 (this is interior-shape coverage, not an unbiased exhaustion).
- **S3** incumbent (the historical decreasing template per depth), reused from
  :data:`exp.weighted_sum.emit_trajectory_yamls.DEPTH_WEIGHTS`, so the incumbent
  is evaluated in the same batch as every new shape.

Reproducibility: the exact modality weights are NOT hand-transcribed (that risks
0.31-vs-real-0.3125 rounding); they are looked up from the tracked
:func:`exp.weighted_sum.emit_yamls.grid3_weight_configs` by the winner cid, so a
clean clone regenerates every YAML from tracked inputs (calibration JSON + this
script). ``d1`` never enters the eval set (no trajectory; existing data reused).

yaml_id = ``cp1_spatial_pool_16__<winner_cid>__d{depth}__<label>`` where label is
``s1_c{cc}_q{qq}`` / ``s2_{parts}`` / ``incumbent`` (a unique tag only; the true
weights are read back from the YAML by the analysis, never parsed from the id).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from openpi.cache.config import load_cache_config

from exp.weighted_sum.emit_yamls import build_eval_config, grid3_weight_configs
from exp.weighted_sum.emit_trajectory_yamls import DEPTH_WEIGHTS

# ----------------------------------------------------------------------
# Fixed per-depth base identity, per task suite.
#
# libero_spatial: every winner cid is a clean grid3 point, so the EXACT 1/16
# modality weights resolve via grid3_weight_configs — no hand-copied decimals.
#
# libero_10: the winners are NOT clean grid points, and BOTH the cid and the
# all_results.csv v0/v1/rs columns are lossy (they come from int(w*100), see
# build_all_results.py). The exact per-depth modality weights are therefore
# frozen in LIBERO10_BASE_MANIFEST below, read from the actual evaluated base
# YAMLs. NEVER reconstruct libero_10 weights from the cid, all_results.csv, or
# the grid functions.
# ----------------------------------------------------------------------
STEM = "cp1_spatial_pool_16"
FIELDS = ("vision_0", "vision_1", "robot_state")

WINNER_CID_BY_SUITE: dict[str, dict[int, str]] = {
    "libero_spatial": {
        3: "grid3_vision_0@31_vision_1@12_robot_state@56",   # -> 0.3125 / 0.125  / 0.5625
        4: "grid3_vision_0@56_vision_1@18_robot_state@25",   # -> 0.5625 / 0.1875 / 0.25
        5: "grid3_vision_0@31_vision_1@6_robot_state@62",    # -> 0.3125 / 0.0625 / 0.625
    },
    "libero_10": {
        3: "grid_vision_0@62_vision_1@37",                   # grid2 (rs=0)
        4: "grid3_vision_0@25_vision_1@43_robot_state@31",   # grid3
        5: "grid_vision_0@50_vision_1@50",                   # grid2 (rs=0)
    },
}
# Backward-compatible alias: bare WINNER_CID stays the libero_spatial set so the
# original single-suite contract is byte-identical.
WINNER_CID: dict[int, str] = WINNER_CID_BY_SUITE["libero_spatial"]

# Tracked, non-lossy per-depth base manifest for libero_10 (source of truth for
# the modality weights). Values are the exact floats read from the actual base
# YAMLs (a zero-diff build_eval_config rebuild + a tracked-fixture deep-diff test
# lock them). The cid / all_results.csv are lossy and MUST NOT be used to derive
# these — see the module docstring and the class-level note above.
LIBERO10_BASE_MANIFEST: dict[int, dict] = {
    3: {"cid": "grid_vision_0@62_vision_1@37",
        "weights": {"vision_0": 0.62, "vision_1": 0.37, "robot_state": 0.0},
        "provenance": "config/trajectory/libero_10/cp1_spatial_pool_16__grid_vision_0@62_vision_1@37__d3.yaml"},
    4: {"cid": "grid3_vision_0@25_vision_1@43_robot_state@31",
        "weights": {"vision_0": 0.25, "vision_1": 0.4375, "robot_state": 0.3125},
        "provenance": "config/stage2/libero_10/cp1_spatial_pool_16__grid3_vision_0@25_vision_1@43_robot_state@31__d4.yaml"},
    5: {"cid": "grid_vision_0@50_vision_1@50",
        "weights": {"vision_0": 0.5, "vision_1": 0.5, "robot_state": 0.0},
        "provenance": "config/trajectory/libero_10/cp1_spatial_pool_16__grid_vision_0@50_vision_1@50__d5.yaml"},
}

# grid3 params that produced the libero_spatial trajectory_wsweep bases.
GRID3_STEP = 0.0625
GRID3_DOM_MIN = 0.1875

# Search knobs.
C_GRID = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
Q_GRID = (0.5, 1.0, 2.0)
S2_STEP: dict[int, float] = {3: 0.1, 4: 0.125, 5: 0.125}
DEFAULT_DEPTHS = (3, 4, 5)

# Locked union sizes (throwaway-script verified; asserted by tests + emitter).
EXPECTED_UNION: dict[int, int] = {3: 52, 4: 60, 5: 59}

# Per-suite path defaults, resolved post-parse from --task-suite so an explicit
# CLI flag always wins (see main). Kept module-level for reuse + testing.
_REPO = Path(__file__).resolve().parents[2]
SUITE_DEFAULTS: dict[str, dict[str, str]] = {
    "libero_spatial": {
        "calibration": str(_REPO / "exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json"),
        "artifact_dir": "exp/common/data/cache_artifacts/libero_spatial",
        "output_dir": str(_REPO / "exp/weighted_sum/config/trajectory_weight_alloc/libero_spatial/eval"),
    },
    "libero_10": {
        "calibration": str(_REPO / "exp/weighted_sum/data/libero_10/phase1/calibration_normalizers.json"),
        "artifact_dir": "exp/common/data/cache_artifacts/libero_10",
        "output_dir": str(_REPO / "exp/weighted_sum/config/trajectory_weight_alloc/libero_10/eval"),
    },
}


# ----------------------------------------------------------------------
# Weight-set construction
# ----------------------------------------------------------------------
def _canon(vec: list[float]) -> tuple[float, ...]:
    """Canonical identity for dedup: each component rounded to 6 decimals."""
    return tuple(round(x, 6) for x in vec)


def _q_label(q: float) -> str:
    return f"{int(round(q * 10)):02d}"


def s1_vector(c: float, q: float, depth: int) -> list[float]:
    """Current-dominant vector: w[0]=c, geometric tail q**j over depth-1 steps."""
    m = depth - 1
    raw = [q ** j for j in range(m)]
    s = sum(raw)
    tail = [(1.0 - c) * r / s for r in raw]
    return [c] + tail


def _compositions(n: int, k: int):
    """Yield every composition of n into exactly k positive integer parts."""
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in _compositions(n - first, k - 1):
            yield (first,) + rest


def build_depth_configs(depth: int) -> list[tuple[tuple[float, ...], str, list[float]]]:
    """Return the deduped ordered (canonical, label, weights) list for one depth.

    Insertion order encodes the label precedence S3 incumbent > S1 > S2: a
    canonical vector present in several families keeps the first (highest-
    precedence) label. Count is order-independent (= |S1 u S2 u S3|).
    """
    if depth not in S2_STEP:
        raise ValueError(f"unsupported depth {depth}; have {sorted(S2_STEP)}")
    seen: dict[tuple[float, ...], tuple[str, list[float]]] = {}
    order: list[tuple[float, ...]] = []

    def add(vec: list[float], label: str) -> None:
        key = _canon(vec)
        if key in seen:
            return
        # Store the canonicalized (6-decimal) weights so the emitted YAML is clean
        # and byte-stable, matching the dedup identity.
        seen[key] = (label, list(key))
        order.append(key)

    # S3 incumbent (highest precedence).
    add(list(DEPTH_WEIGHTS[depth]), "incumbent")
    # S1 current-dominant x tail-shape.
    for c in C_GRID:
        for q in Q_GRID:
            add(s1_vector(c, q, depth), f"s1_c{int(round(c * 100))}_q{_q_label(q)}")
    # S2 interior shape lattice.
    n = round(1.0 / S2_STEP[depth])
    for parts in _compositions(n, depth):
        add([p / n for p in parts], "s2_" + "-".join(str(p) for p in parts))

    return [(key, seen[key][0], seen[key][1]) for key in order]


# ----------------------------------------------------------------------
# Emission
# ----------------------------------------------------------------------
def _modality_weights(suite: str, winner_cid: dict[int, str]) -> dict[int, dict[str, float]]:
    """Exact per-depth modality weights for a suite.

    libero_10 reads the frozen non-lossy LIBERO10_BASE_MANIFEST (the cid and the
    all_results.csv columns are lossy, so they are never used here).
    libero_spatial resolves the cid against grid3_weight_configs (its winners are
    exact 1/16 grid points).
    """
    if suite == "libero_10":
        out: dict[int, dict[str, float]] = {}
        for depth, cid in winner_cid.items():
            entry = LIBERO10_BASE_MANIFEST.get(depth)
            if entry is None:
                raise SystemExit(f"no libero_10 base manifest for d{depth}")
            if entry["cid"] != cid:
                raise SystemExit(f"manifest cid {entry['cid']!r} != winner cid {cid!r} (d{depth})")
            out[depth] = dict(entry["weights"])
        return out
    cfgs = grid3_weight_configs(list(FIELDS), step=GRID3_STEP, dom_min=GRID3_DOM_MIN)
    out = {}
    for depth, cid in winner_cid.items():
        if cid not in cfgs:
            raise SystemExit(f"winner cid {cid!r} (d{depth}) not in grid3 output — grid params drifted")
        out[depth] = cfgs[cid]
    return out


def expected_ids(depths=DEFAULT_DEPTHS, *, winner_cid: dict[int, str] = WINNER_CID) -> set[str]:
    """The exact set of yaml_id stems the emitter will write (for stale-guard).

    ``depths`` keeps its original positional slot (backward-compatible with the
    single-suite callers, e.g. ``expected_ids((3,))``); ``winner_cid`` is
    keyword-only and defaults to the libero_spatial set. The analyzer passes the
    run's suite winner set.
    """
    ids: set[str] = set()
    for depth in depths:
        for _key, label, _w in build_depth_configs(depth):
            ids.add(f"{STEM}__{winner_cid[depth]}__d{depth}__{label}")
    return ids


def emit(output_dir: Path, calibration: Path, artifact_dir: str, depths=DEFAULT_DEPTHS, *,
         suite: str = "libero_spatial", winner_cid: dict[int, str] | None = None) -> set[str]:
    """Build + write every eval YAML; return the set of written yaml_id stems.

    Rebuilds each depth's base via build_eval_config from the tracked calibration
    JSON and the winner modality weights, overlaying the swept trajectory_weights.
    Every file is validated by load_cache_config (no pkl load). Stale guard: only
    ``*.yaml`` under exp/weighted_sum/config are removed, then the written set is
    asserted equal to expected_ids.
    """
    repo = Path(__file__).resolve().parents[2]
    for depth in depths:
        if depth not in S2_STEP:
            raise SystemExit(f"depth {depth} unsupported (d1 has no trajectory and is excluded by design)")

    calib = json.loads(Path(calibration).read_text(encoding="utf-8"))
    if STEM not in calib:
        raise SystemExit(f"stem {STEM!r} not in calibration; have {sorted(calib)}")
    entry = calib[STEM]
    if winner_cid is None:
        winner_cid = WINNER_CID_BY_SUITE[suite]
    modality = _modality_weights(suite, winner_cid)

    # Stale guard: refuse paths outside the config tree; delete only *.yaml.
    resolved = output_dir.resolve()
    guard = (repo / "exp/weighted_sum/config").resolve()
    if guard != resolved and guard not in resolved.parents:
        raise SystemExit(f"refuse to touch {output_dir}: not under exp/weighted_sum/config")
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.glob("*.yaml"):
        p.unlink()

    written: set[str] = set()
    for depth in depths:
        for _key, label, tw in build_depth_configs(depth):
            cfg = build_eval_config(
                builder_type=entry["builder_type"],
                vector_dims=entry["vector_dims"],
                preload_path=f"{artifact_dir}/{STEM}.pkl",
                weights=modality[depth],
                fields_calib=entry["fields"],
                trajectory_depth=depth,
                trajectory_weights=tw,
            )
            yaml_id = f"{STEM}__{winner_cid[depth]}__d{depth}__{label}"
            path = output_dir / f"{yaml_id}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            load_cache_config(path)  # schema self-check (validate_cache_config; no pkl load)
            if yaml_id in written:
                raise SystemExit(f"duplicate yaml_id {yaml_id!r} — label collision")
            written.add(yaml_id)

    # Post-emit hard gate: directory holds exactly the expected set, no stray files.
    on_disk = {p.stem for p in output_dir.glob("*.yaml")}
    stray = [p.name for p in output_dir.iterdir() if p.suffix != ".yaml"]
    if stray:
        raise SystemExit(f"stray non-yaml files in {output_dir}: {stray}")
    exp_ids = expected_ids(depths, winner_cid=winner_cid)
    if on_disk != exp_ids:
        raise SystemExit(
            f"emitted set != expected: missing={sorted(exp_ids - on_disk)}, "
            f"extra={sorted(on_disk - exp_ids)}"
        )
    return written


def main():
    ap = argparse.ArgumentParser(description="Emit trajectory step-weight screening YAMLs")
    ap.add_argument("--task-suite", choices=tuple(SUITE_DEFAULTS), default="libero_spatial")
    # Path defaults stay None so --task-suite resolves them post-parse; an
    # explicit flag always wins. Avoids silently reusing one suite's
    # calibration/artifact/output when emitting for another.
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--artifact-dir", default=None,
                    help="relative — the server resolves preload_path against its own CWD")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--depths", default="3,4,5")
    args = ap.parse_args()

    suite = args.task_suite
    sd = SUITE_DEFAULTS[suite]
    calibration = args.calibration or sd["calibration"]
    artifact_dir = args.artifact_dir or sd["artifact_dir"]
    output_dir = args.output_dir or sd["output_dir"]
    winner_cid = WINNER_CID_BY_SUITE[suite]

    depths = tuple(int(d) for d in args.depths.split(",") if d != "")
    per_depth = {d: len(build_depth_configs(d)) for d in depths}
    total = sum(per_depth.values())
    print(f"=== trajectory step-weight search matrix ({suite}) ===")
    for d in depths:
        locked = EXPECTED_UNION.get(d)
        tag = "" if locked is None else (" OK" if per_depth[d] == locked else f" != locked {locked}")
        print(f"  d{d}: {per_depth[d]} configs{tag}")
    print(f"  total = {total} configs")
    if set(depths) == set(DEFAULT_DEPTHS):
        assert total == 171, f"expected 171 configs for depths {DEFAULT_DEPTHS}, got {total}"

    written = emit(Path(output_dir), Path(calibration), artifact_dir, depths, suite=suite, winner_cid=winner_cid)
    print(f"Wrote {len(written)} eval YAMLs to {output_dir}")


if __name__ == "__main__":
    main()
