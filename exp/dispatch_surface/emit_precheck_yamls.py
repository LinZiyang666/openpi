"""Emit the precheck eval yamls + arm matrix (dynamic 5-7 arms).

Arms (plan section 4.6): T1-T3 calibrated two-threshold baselines
(start_t=0.3 tier, thresholds solved by the single shared
``derive_thresholds`` implementation on the fit-union-cal score
distribution), S0 (RIT s-only surface artifact at delta*), SV (RIT primary surface),
and SV-/SV+ only when the corresponding grid neighbour artifact exists.

Every yaml is derived from the gate-line template: judge section replaced,
``preload_path`` pointed at the rebuilt dispatch library (with an assert that
the preload file's sha256 equals the artifact contract's library_sha256), and
``write_policy: never`` enforced. The emitted arm matrix records the actual
arm set for the launch manifest.

The GATE is re-solved too, not inherited. ``solve_gtp`` states the rule the
GST threshold line runs on: the gate theta and the judge cuts are quantiles of the
SAME score distribution and are "re-derived per library rather than carried
over -- carrying a threshold across libraries is exactly the mistake the
ratio-based design exists to prevent". This line rebuilds the library, so an
inherited theta would place the gate at a different operating point than the
one it was calibrated for. Both numbers therefore come from this run's
fit-union-cal scores, through the same imported ``derive_thresholds``.

One theta is shared by EVERY arm (thresholds and surfaces alike) and held
fixed across the sweep. The gate decides whether a step is probed at all; if
it moved between arms, a difference in SR or cost could not be attributed to
the verdict rule under test.

Usage:
  uv run python -m exp.dispatch_surface.emit_precheck_yamls \
      --template exp/gate_research/config/libero_spatial/n4_server/<base>.yaml \
      --table exp/dispatch_surface/data/dispatch_table_fresh.jsonl \
      --fit-dir exp/dispatch_surface/data/surface_fit \
      --library-pkl exp/dispatch_surface/data/cache_artifacts/dispatch_lib_cp1_spatial_pool_16.pkl \
      --out-dir exp/dispatch_surface/config/precheck
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import pathlib

import yaml

from exp.dispatch_surface.phase0_roster import FAMILY_S0, FAMILY_SV, PROTOCOL_PHASE0
from exp.dispatch_surface.template_parity import (
    assert_export_record_schema,
    assert_no_placeholders,
    assert_template_parity,
)

from exp.gate_threshold_pareto.emit_gtp_yamls import (
    GATE_J,
    GATE_L,
    GATE_PROBE_INTERVAL,
)

# Pre-registered baseline grid: (f_FH %, f_WS %) cells around the known
# threshold-pareto sweet spots.
THRESHOLD_CELLS = ((30, 20), (50, 20), (70, 10))
WS_START_T = 0.3

# Rev 1 runs two pre-registered layers. Primary isolates the verdict rule by
# probing every step; secondary re-attaches the production gate and is
# descriptive only. Both rosters are frozen here, before any rollout, so the
# secondary arm set can never be chosen after seeing primary results.
LAYER_PRIMARY = "primary"
LAYER_SECONDARY = "secondary"
# Rev 2 Phase 0: post-hoc exploratory arms on the old A' (development set).
# The roster is frozen in phase0_roster; nothing here is confirmatory.
LAYER_EXPLORATORY = "exploratory"
# Rev 2 confirmation plan: dense GST threshold grid on the development set
# (plan section 3.2). Cells are frozen in phase0_roster; nothing is chosen here.
LAYER_TGRID = "exploratory_tgrid"
# Effective retrieval widths. The on-disk yaml stays at configured top_k=1 for
# every arm so search_digest keeps matching the calibration table; these are the
# widths the judges' min_required_top_k hint lifts to at runtime.
SV_EFFECTIVE_TOP_K = 5
S0_EFFECTIVE_TOP_K = 1

PRIMARY_CORE_ARMS = frozenset(
    {"dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10", "dsp_s0", "dsp_sv"}
)
SECONDARY_ARMS = frozenset(
    {"dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10", "dsp_sv"}
)
PROTOCOL = "dispatch_surface_rev1"


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _load_scores(table_path: str) -> list[float]:
    scores = []
    for line in open(table_path):
        row = json.loads(line)
        if row["split"] in ("fit", "cal"):
            scores.append(float(row["s"]))
    if not scores:
        raise SystemExit(f"no fit/cal scores in {table_path}")
    return scores


def gate_section(layer: str, theta: float) -> dict:
    """The gate every arm in a layer shares.

    Primary probes every step. The gate cuts on ``s``, which is the same signal
    the threshold arms and the surface cut on, so leaving it in would let it
    truncate exactly the ambiguous region where ``v`` is supposed to earn its
    keep -- and a null result could then not be attributed to the verdict rather
    than to the gate. Secondary puts the production gate back, unchanged and
    identical across its arms, purely as external validity.
    """
    if layer == LAYER_PRIMARY:
        return {"type": "always_search"}
    return {
        "type": "score_hysteresis",
        "theta_low": theta,
        "theta_high": theta,
        "j": GATE_J,
        "probe_interval": GATE_PROBE_INTERVAL,
        "L": GATE_L,
    }


def _emit(template: dict, out_path: pathlib.Path, judge_section: dict,
          preload_path: str, theta: float, layer: str) -> None:
    doc = json.loads(json.dumps(template))  # deep copy without yaml anchors
    cp1 = doc["checkpoints"]["cp1"]
    cp1["judge"] = judge_section
    # Written explicitly rather than inherited: the two suites' templates differ
    # (spatial carries a stale score_hysteresis theta, l10 carries
    # always_search) and neither is this run's operating point.
    cp1["gate"] = gate_section(layer, theta)
    doc["backend"]["in_memory"]["preload_path"] = preload_path
    doc["write_policy"] = {"type": "never"}
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False))


# ----------------------------------------------------------------------
# Rev 2 Phase 0: exploratory layer (plan section 3.2)
# ----------------------------------------------------------------------

def _export_record_arms(records: list[dict]) -> dict[str, dict]:
    """arm -> {family, quantile, artifact, delta, export_record_index, output_sha256}."""
    out: dict[str, dict] = {}
    for idx, rec in enumerate(records):
        if rec.get("protocol") != PROTOCOL_PHASE0 or rec.get("posthoc_exploratory") is not True:
            raise SystemExit(f"export record {idx} is not a Phase 0 exploratory record")
        family = rec.get("family")
        if family not in (FAMILY_SV, FAMILY_S0):
            raise SystemExit(f"export record {idx} has family {family!r}")
        for name, art in (rec.get("artifacts") or {}).items():
            arm = f"dsp_{family}_{name}"
            if arm in out:
                raise SystemExit(f"duplicate exploratory arm {arm}")
            out[arm] = {"family": family, "quantile": float(art["quantile"]),
                        "artifact": art["path"], "delta": float(art["delta"]),
                        "output_sha256": art["output_sha256"], "export_record_index": idx}
    return out


def emit_exploratory(args) -> None:
    from exp.dispatch_surface import rev1_package as pkgmod
    from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest, cost_model_payload
    from exp.dispatch_surface.phase0_roster import (
        ANCHOR_ARM,
        ANCHOR_ROLE,
        ANCHOR_THRESHOLD,
        CONTRACT_ANCHOR_ARM,
        FAMILY_ANCHOR,
        assert_roster,
        roster_spec,
        roster_spec_digest,
    )
    from openpi.cache.components.surface_judge import load_surface_artifact

    for flag in ("suite", "export_records", "rev1_package_manifest"):
        if not getattr(args, flag):
            raise SystemExit(f"--{flag.replace('_', '-')} is required for the exploratory layer")
    suite = args.suite
    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    if manifest.get("suite") != suite:
        raise SystemExit(f"package suite {manifest.get('suite')!r} != --suite {suite!r}")
    rev1_matrix = pkgmod.load_json_member(manifest, pkg, "matrix")
    template = yaml.safe_load(open(args.template))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_sha = _file_sha256(pathlib.Path(args.library_pkl))
    if lib_sha != rev1_matrix.get("library_sha256"):
        raise SystemExit("--library-pkl is not the library the Rev 1 matrix froze")
    theta = float(rev1_matrix["gate_theta"])

    record_paths = [pathlib.Path(p) for p in args.export_records.split(",") if p.strip()]
    records = [json.loads(p.read_text()) for p in record_paths]
    record_sha = [_file_sha256(p) for p in record_paths]
    for idx, rec in enumerate(records):
        assert_export_record_schema(rec, what=f"export record {idx}", cost_model_digest=cost_model_digest(),
                                    protocol=PROTOCOL_PHASE0)
        if rec.get("rev1_package_manifest_sha256") != manifest_sha:
            raise SystemExit("an export record was produced against a different Rev 1 package")
    surface_arms = _export_record_arms(records)
    roster_arms = {arm: {"family": spec["family"], "quantile": spec["quantile"]}
                   for arm, spec in surface_arms.items()}
    roster_arms[ANCHOR_ARM] = {"family": FAMILY_ANCHOR, "quantile": None}
    assert_roster(suite, roster_arms)

    arms: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    artifact_sha256: dict[str, str] = {}
    families: dict[str, str] = {}
    quantiles: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for arm, spec in surface_arms.items():
        path = pathlib.Path(spec["artifact"])
        if not path.is_file() or _file_sha256(path) != spec["output_sha256"]:
            raise SystemExit(f"{arm}: artifact missing or drifted from its export record")
        art = load_surface_artifact(str(path))
        if art.meta.get("posthoc_exploratory") is not True:
            raise SystemExit(f"{arm}: artifact is not marked posthoc_exploratory")
        rec = records[spec["export_record_index"]]
        source_role = rec["source_role"]
        source = load_surface_artifact(str(pkgmod.verify_member(manifest, pkg, source_role)))
        if (art.meta.get("source_artifact_sha256") != pkgmod.member_sha(manifest, source_role)
                or rec["source_artifact_sha256"] != pkgmod.member_sha(manifest, source_role)):
            raise SystemExit(f"{arm}: source artifact SHA chain broken")
        fit_role = "fit.s0" if spec["family"] == FAMILY_S0 else "fit.sv"
        if rec["source_fit_record_sha256"] != pkgmod.member_sha(manifest, fit_role):
            raise SystemExit(f"{arm}: source fit record SHA chain broken")
        assert_template_parity(art, source, what=arm)
        assert_no_placeholders(art.meta, what=arm)
        if art.uses_disagreement != (spec["family"] == FAMILY_SV):
            raise SystemExit(f"{arm}: family/uses_disagreement mismatch")
        if art.delta != spec["delta"]:
            raise SystemExit(f"{arm}: artifact delta != export record delta")
        judge = {"type": "dispatch_surface", "surface_artifact_path": str(path)}
        ypath = out_dir / f"{arm}.yaml"
        _emit(template, ypath, judge, args.library_pkl, theta, LAYER_PRIMARY)
        arms[arm] = str(ypath)
        artifact_paths[arm] = str(path.resolve())
        artifact_sha256[arm] = spec["output_sha256"]
        families[arm] = spec["family"]
        quantiles[arm] = spec["quantile"]
        deltas[arm] = spec["delta"]
    anchor_judge = {"type": "threshold", "threshold": float(ANCHOR_THRESHOLD)}
    apath = out_dir / f"{ANCHOR_ARM}.yaml"
    _emit(template, apath, anchor_judge, args.library_pkl, theta, LAYER_PRIMARY)
    arms[ANCHOR_ARM] = str(apath)
    families[ANCHOR_ARM] = FAMILY_ANCHOR

    spec = roster_spec(suite)
    spec_path = out_dir / "roster_spec.json"
    # canonical form: the digest hashes exactly these bytes
    spec_path.write_text(json.dumps(spec, sort_keys=True, separators=(",", ":")))
    matrix = {
        "protocol": PROTOCOL_PHASE0,
        "layer": LAYER_EXPLORATORY,
        "posthoc_exploratory": True,
        "suite": suite,
        "arms": arms,
        "families": families,
        "quantiles": quantiles,
        "deltas": deltas,
        "gate_type": gate_section(LAYER_PRIMARY, theta)["type"],
        "gate_theta": theta,
        "gate_theta_top_fraction": rev1_matrix.get("gate_theta_top_fraction"),
        "gate_params": rev1_matrix.get("gate_params"),
        "arm_yaml_sha256": {arm: _file_sha256(pathlib.Path(p)) for arm, p in arms.items()},
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
        "certification_mode": rev1_matrix.get("certification_mode"),
        "judge_role": {ANCHOR_ARM: ANCHOR_ROLE},
        "anchor_threshold": float(ANCHOR_THRESHOLD),
        "contract_anchor_arm": CONTRACT_ANCHOR_ARM[suite],
        "core_arms": [],
        "descriptive_arms": sorted(arms),
        "roster_spec_path": str(spec_path.resolve()),
        "roster_spec_sha256": roster_spec_digest(suite),
        "rev1_package_manifest_path": str(pathlib.Path(args.rev1_package_manifest).resolve()),
        "rev1_package_manifest_sha256": manifest_sha,
        "rev1_matrix_sha256": pkgmod.member_sha(manifest, "matrix"),
        "export_record_paths": [str(p.resolve()) for p in record_paths],
        "export_record_sha256": record_sha,
        "cost_model": cost_model_payload(),
        "cost_model_digest": cost_model_digest(),
        "library_pkl": args.library_pkl,
        "library_sha256": lib_sha,
        "template": str(args.template),
    }
    if _file_sha256(spec_path) != roster_spec_digest(suite):
        raise SystemExit("roster spec digest mismatch")
    matrix_path = out_dir / f"arm_matrix_{LAYER_EXPLORATORY}.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    print(f"emitted {len(arms)} exploratory arms for {suite} -> {matrix_path}")


# ----------------------------------------------------------------------
# Rev 2 confirmation plan: dense GST threshold grid (plan section 3.2)
# ----------------------------------------------------------------------

def threshold_pair_digest(t_fh: float, t_ws: float | None) -> str:
    return hashlib.sha256(json.dumps([float(t_fh), None if t_ws is None else float(t_ws)],
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def threshold_pair_rollup(arms: list[str], digests: dict[str, str]) -> str:
    h = hashlib.sha256()
    for arm in arms:
        h.update(digests[arm].encode("utf-8"))
    return h.hexdigest()


def nominal_cost_ms(fh: int, ws: int) -> float:
    """Analytic cost if the FULL / WARM / MISS shares were exactly fh% / ws% / rest."""
    from exp.dispatch_surface.analysis.analytic_cost import unit_cost_table

    u = unit_cost_table()
    return (fh / 100.0) * u["FULL_HIT"] + (ws / 100.0) * u["WARM_START"] + (1.0 - (fh + ws) / 100.0) * u["MISS"]


def emit_tgrid(args) -> None:
    """Emit the 29 frozen threshold-grid arms for libero_10 on the development set.

    Thresholds come from the SAME ``derive_thresholds`` on the SAME fit-union-
    cal score list as Rev 1 (the table's content SHA must equal the archived
    fit record's ``input_digests.table``); gate theta is read from the Rev 1
    matrix, never recomputed. ``ws = 0`` cells carry no ``warm_tiers`` key;
    ``ws > 0`` cells carry exactly one tier at ``start_t = 0.3`` and require
    ``t_fh > t_ws``. Two cells deriving the same threshold pair are refused.
    """
    from exp.dispatch_surface import rev1_package as pkgmod
    from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest, cost_model_payload
    from exp.dispatch_surface.analysis.estimator_version import budget_mixture_digest
    from exp.dispatch_surface.analysis.precheck_io import (
        load_accepted_cells_costonly,
        load_cost_cells_costonly,
    )
    from exp.dispatch_surface.phase0_roster import (
        FAMILY_THRESHOLD,
        PROTOCOL_TGRID,
        REV1_THRESHOLD_CELLS,
        tgrid_arm_id,
        tgrid_cells,
        tgrid_roster_spec,
        tgrid_roster_spec_digest,
    )
    from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, official_test_inits
    from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds

    for flag in ("suite", "rev1_package_manifest", "table"):
        if not getattr(args, flag):
            raise SystemExit(f"--{flag.replace('_', '-')} is required for the threshold-grid layer")
    suite = args.suite
    spec = tgrid_roster_spec(suite)
    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    if manifest.get("suite") != suite:
        raise SystemExit(f"package suite {manifest.get('suite')!r} != --suite {suite!r}")
    rev1_matrix = pkgmod.load_json_member(manifest, pkg, "matrix")
    fit_sv = pkgmod.load_json_member(manifest, pkg, "fit.sv")
    table_sha = _file_sha256(pathlib.Path(args.table))
    if table_sha != (fit_sv.get("input_digests") or {}).get("table"):
        raise SystemExit("--table is not the calibration table the archived SV fit record binds")
    template = yaml.safe_load(open(args.template))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} must be empty")
    lib_sha = _file_sha256(pathlib.Path(args.library_pkl))
    if lib_sha != rev1_matrix.get("library_sha256"):
        raise SystemExit("--library-pkl is not the library the Rev 1 matrix froze")
    theta = float(rev1_matrix["gate_theta"])
    scores = _load_scores(args.table)

    arms: dict[str, str] = {}
    nominal: dict[str, dict] = {}
    pairs: dict[str, list] = {}
    pair_digests: dict[str, str] = {}
    nominal_cost: dict[str, float] = {}
    seen_pairs: dict[tuple, str] = {}
    conflicts = []
    grid_arms = [tgrid_arm_id(fh, ws) for fh, ws in tgrid_cells()]
    for fh, ws in tgrid_cells():
        if fh + ws > 100:
            raise SystemExit(f"illegal cell ({fh}, {ws}): fh + ws > 100")
        arm = tgrid_arm_id(fh, ws)
        t_fh, t_ws = derive_thresholds(scores, fh / 100.0, ws / 100.0)
        if not (math.isfinite(t_fh) and math.isfinite(t_ws)):
            raise SystemExit(f"{arm}: non-finite derived thresholds")
        if ws == 0:
            judge = {"type": "threshold", "threshold": float(t_fh)}
            pair = (float(t_fh), None)
        else:
            if not t_fh > t_ws:
                raise SystemExit(f"{arm}: degenerate cell, t_fh={t_fh} is not > t_ws={t_ws}")
            judge = {"type": "threshold", "threshold": float(t_fh),
                     "warm_tiers": [{"threshold": float(t_ws), "start_t": WS_START_T}]}
            pair = (float(t_fh), float(t_ws))
        if pair in seen_pairs:
            conflicts.append((seen_pairs[pair], arm, pair))
        seen_pairs[pair] = arm
        ypath = out_dir / f"{arm}.yaml"
        _emit(template, ypath, judge, args.library_pkl, theta, LAYER_PRIMARY)
        arms[arm] = str(ypath)
        nominal[arm] = {"fh": fh, "ws": ws}
        pairs[arm] = [pair[0], pair[1]]
        pair_digests[arm] = threshold_pair_digest(pair[0], pair[1])
        nominal_cost[arm] = nominal_cost_ms(fh, ws)
    if conflicts:
        raise SystemExit(f"threshold-grid cells derive identical threshold pairs: {conflicts}")
    if sorted(arms) != sorted(grid_arms) or sorted(arms) != sorted(spec["arms"]):
        raise SystemExit("emitted arms != frozen threshold grid")

    # Rev 1 reference cells: nominal vs realized (cost rows only, outcome-blind)
    officials = official_test_inits(str(pkgmod.verify_member(manifest, pkg, "split_manifest")), FORMAL_TRIALS)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    rev1_arms = [f"dsp_t_fh{fh}_ws{ws}" for fh, ws in REV1_THRESHOLD_CELLS]
    accepted = load_accepted_cells_costonly(str(pkgmod.verify_member(manifest, pkg, "journal")), rev1_arms, grid)
    cells, _summary = load_cost_cells_costonly(str(pkgmod.verify_member(manifest, pkg, "per_step")), rev1_arms, accepted, officials)
    rev1_ref = {}
    for (fh, ws), arm in zip(REV1_THRESHOLD_CELLS, rev1_arms):
        num = sum(c for c, _n in cells[arm].values())
        den = sum(n for _c, n in cells[arm].values())
        realized = num / den
        rev1_ref[arm] = {"fh": fh, "ws": ws, "nominal_cost_ms": nominal_cost_ms(fh, ws),
                         "realized_cost_ms": realized, "realized_minus_nominal_ms": realized - nominal_cost_ms(fh, ws)}

    spec_path = out_dir / "tgrid_roster_spec.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True, separators=(",", ":")))
    if _file_sha256(spec_path) != tgrid_roster_spec_digest(suite):
        raise SystemExit("tgrid roster spec digest mismatch")
    matrix = {
        "protocol": PROTOCOL_TGRID,
        "layer": LAYER_TGRID,
        "posthoc_exploratory": True,
        "suite": suite,
        "arms": arms,
        "families": {arm: FAMILY_THRESHOLD for arm in arms},
        "nominal": nominal,
        "threshold_pairs": pairs,
        "threshold_pair_digests": pair_digests,
        "threshold_pair_rollup_sha256": threshold_pair_rollup(grid_arms, pair_digests),
        "nominal_cost_ms": nominal_cost,
        "rev1_reference_cells": rev1_ref,
        "gate_type": gate_section(LAYER_PRIMARY, theta)["type"],
        "gate_theta": theta,
        "gate_theta_top_fraction": rev1_matrix.get("gate_theta_top_fraction"),
        "gate_params": rev1_matrix.get("gate_params"),
        "arm_yaml_sha256": {arm: _file_sha256(pathlib.Path(p)) for arm, p in arms.items()},
        "artifact_paths": {},
        "artifact_sha256": {},
        "certification_mode": rev1_matrix.get("certification_mode"),
        "core_arms": [],
        "descriptive_arms": sorted(arms),
        "tgrid_roster_spec_path": str(spec_path.resolve()),
        "tgrid_roster_spec_sha256": tgrid_roster_spec_digest(suite),
        "rev1_package_manifest_path": str(pathlib.Path(args.rev1_package_manifest).resolve()),
        "rev1_package_manifest_sha256": manifest_sha,
        "rev1_matrix_sha256": pkgmod.member_sha(manifest, "matrix"),
        "table_sha256": table_sha,
        "cost_model": cost_model_payload(),
        "cost_model_digest": cost_model_digest(),
        "estimator_version": budget_mixture_digest(),
        "contract_source": spec["contract_source"],
        "library_pkl": args.library_pkl,
        "library_sha256": lib_sha,
        "template": str(args.template),
    }
    matrix_path = out_dir / f"arm_matrix_{LAYER_TGRID}.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    print(f"emitted {len(arms)} threshold-grid arms for {suite} -> {matrix_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", required=True)
    ap.add_argument("--table", default=None)
    ap.add_argument("--fit-dir", default=None,
                    help="fit_surface output dir (artifacts + fit_record.json)")
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--layer", required=True,
                    choices=[LAYER_PRIMARY, LAYER_SECONDARY, LAYER_EXPLORATORY, LAYER_TGRID])
    ap.add_argument("--out-dir", required=True)
    # -- Rev 2 Phase 0 exploratory layer only ------------------------
    ap.add_argument("--suite", default=None, help="exploratory: suite whose frozen roster to emit")
    ap.add_argument("--export-records", default=None,
                    help="exploratory: comma list of export_record.json (SV and S0)")
    ap.add_argument("--rev1-package-manifest", default=None,
                    help="exploratory: Rev 1 discipline package MANIFEST.json")
    args = ap.parse_args()
    if args.layer == LAYER_EXPLORATORY:
        emit_exploratory(args)
        return
    if args.layer == LAYER_TGRID:
        emit_tgrid(args)
        return
    if not args.table or not args.fit_dir:
        ap.error("--table and --fit-dir are required for the primary/secondary layers")

    from exp.gate_threshold_pareto.solve_gtp import MIN_SCORES, THETA_TOP_FRACTION
    from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds
    from openpi.cache.components.surface_judge import (
        CERTIFICATION_EMPIRICAL,
        load_surface_artifact,
    )

    template = yaml.safe_load(open(args.template))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_sha = _file_sha256(pathlib.Path(args.library_pkl))

    arms: dict[str, str] = {}

    # -- gate theta, shared by every arm ---------------------------
    scores = _load_scores(args.table)
    if len(scores) < MIN_SCORES:
        raise SystemExit(
            f"only {len(scores)} fit/cal scores (need >= {MIN_SCORES}) — a quantile "
            "cut this fine would rest on too few samples; extend the cohort rather "
            "than lowering the bar"
        )
    theta = derive_thresholds(scores, THETA_TOP_FRACTION, 0.0)[0]

    # -- threshold baseline arms -----------------------------------
    for fh, ws in THRESHOLD_CELLS:
        t_fh, t_ws = derive_thresholds(scores, fh / 100.0, ws / 100.0)
        arm_id = f"dsp_t_fh{fh}_ws{ws}"
        judge = {
            "type": "threshold",
            "threshold": float(t_fh),
            "warm_tiers": [{"threshold": float(t_ws), "start_t": WS_START_T}],
        }
        path = out_dir / f"{arm_id}.yaml"
        _emit(template, path, judge, args.library_pkl, theta, args.layer)
        arms[arm_id] = str(path)

    # -- surface arms (dynamic cardinality) ------------------------
    fit_dir = pathlib.Path(args.fit_dir)
    fit_record_paths = {
        "sv": fit_dir / "fit_record.json",
        "s0": fit_dir / "fit_record_s_only.json",
    }
    if any(not path.is_file() for path in fit_record_paths.values()):
        raise SystemExit("fit directory must contain both SV and S0 fit records")
    fit_records = {name: json.loads(path.read_text()) for name, path in fit_record_paths.items()}
    sv_record, s0_record = fit_records["sv"], fit_records["s0"]
    if sv_record.get("s_only") is not False or s0_record.get("s_only") is not True:
        raise SystemExit("SV/S0 fit records carry the wrong fit mode")
    for name, rec in fit_records.items():
        if rec.get("certification_mode") != CERTIFICATION_EMPIRICAL or rec.get("stop_loss") is not None:
            raise SystemExit(f"{name} fit record is not a completed empirical Rev 1 fit")
        if not rec.get("d0_binding") or not rec.get("dev_membership_sha256") \
                or not rec.get("fold_map_sha256") or not rec.get("final_fit_digests"):
            raise SystemExit(f"{name} fit record lacks D0/split/final-fit audit fields")
    for field in ("delta_star", "input_digests", "d0_binding", "dev_membership_sha256", "fold_map_sha256"):
        if sv_record.get(field) != s0_record.get(field):
            raise SystemExit(f"SV and S0 fit records differ on nested-ablation field {field}")

    surface_specs = [
        ("dsp_s0", fit_dir / "surface_s_only_primary.npz"),
        ("dsp_sv", fit_dir / "surface_sv_primary.npz"),
        ("dsp_sv_minus", fit_dir / "surface_sv_minus.npz"),
        ("dsp_sv_plus", fit_dir / "surface_sv_plus.npz"),
    ]
    deltas: dict[str, float] = {}
    artifact_paths: dict[str, str] = {}
    artifact_sha256: dict[str, str] = {}
    for arm_id, artifact_path in surface_specs:
        if not artifact_path.is_file():
            if arm_id in ("dsp_s0", "dsp_sv"):
                raise SystemExit(f"required artifact missing: {artifact_path}")
            continue  # SV+/- are optional when delta* sits at a grid endpoint
        artifact = load_surface_artifact(str(artifact_path))
        contract_sha = artifact.retrieval_contract.get("library_sha256")
        if contract_sha != lib_sha:
            raise SystemExit(
                f"{artifact_path}: contract library_sha256={contract_sha} does not "
                f"match preload {args.library_pkl} sha256={lib_sha}"
            )
        # Rev 1 emits empirical artifacts only. A conformal artifact reaching
        # this line would be reported downstream as a certificate the run does
        # not hold, so it is refused here rather than relabelled.
        if artifact.certification_mode != CERTIFICATION_EMPIRICAL:
            raise SystemExit(
                f"{artifact_path}: certification_mode="
                f"{artifact.certification_mode!r}; Rev 1 arms must be "
                f"{CERTIFICATION_EMPIRICAL!r} (no finite-sample certificate is claimed)"
            )
        if artifact.conformal_c != 0.0 or artifact.n_calibration_episodes != 0:
            raise SystemExit(
                f"{artifact_path}: empirical artifact carries conformal_c="
                f"{artifact.conformal_c} / n_calibration_episodes="
                f"{artifact.n_calibration_episodes}"
            )
        expected_record = s0_record if arm_id == "dsp_s0" else sv_record
        expected_name = "primary"
        if arm_id == "dsp_sv_minus":
            expected_name = "minus"
        elif arm_id == "dsp_sv_plus":
            expected_name = "plus"
        recorded_path = (expected_record.get("artifacts") or {}).get(expected_name)
        if recorded_path is None or pathlib.Path(recorded_path).resolve() != artifact_path.resolve():
            raise SystemExit(f"{artifact_path}: fit record does not bind this artifact path")
        meta = artifact.meta
        for field in ("input_digests", "d0_binding", "dev_membership_sha256",
                      "fold_map_sha256", "final_fit_digests"):
            if meta.get(field) != expected_record.get(field):
                raise SystemExit(f"{artifact_path}: artifact/fit-record drift on {field}")
        # Effective retrieval width: SV needs 5 candidates for v, S0 needs 1.
        # The judge lifts the width from the yaml's configured 1 via
        # min_required_top_k, so this is the only place the intent is checked.
        expected_k = SV_EFFECTIVE_TOP_K if arm_id.startswith("dsp_sv") else S0_EFFECTIVE_TOP_K
        if artifact.k != expected_k or artifact.retrieval_contract.get("top_k") != expected_k:
            raise SystemExit(
                f"{artifact_path}: expected k == contract.top_k == {expected_k} for "
                f"{arm_id}, got k={artifact.k} contract.top_k="
                f"{artifact.retrieval_contract.get('top_k')}"
            )
        deltas[arm_id] = artifact.delta
        artifact_paths[arm_id] = str(artifact_path.resolve())
        artifact_sha256[arm_id] = _file_sha256(artifact_path)
        judge = {
            "type": "dispatch_surface",
            "surface_artifact_path": str(artifact_path),
        }
        path = out_dir / f"{arm_id}.yaml"
        _emit(template, path, judge, args.library_pkl, theta, args.layer)
        arms[arm_id] = str(path)

    # Nested-ablation contract: S0 must run at the SAME frozen delta* as SV,
    # or Gate 2 compares two operating points instead of the v axis (G2-B2),
    # and must be fitted from byte-identical inputs (G2R2-B6).
    if deltas["dsp_s0"] != deltas["dsp_sv"]:
        raise SystemExit(
            f"S0 delta {deltas['dsp_s0']} != SV delta {deltas['dsp_sv']} — the s-only "
            "fit must inherit the SV frozen delta (fit_surface --frozen-record)"
        )
    sv_meta = load_surface_artifact(str(fit_dir / "surface_sv_primary.npz")).meta
    s0_meta = load_surface_artifact(str(fit_dir / "surface_s_only_primary.npz")).meta
    if not sv_meta.get("input_digests") or \
            sv_meta.get("input_digests") != s0_meta.get("input_digests"):
        raise SystemExit(
            "SV and S0 artifacts were fitted from different inputs "
            f"(SV={sv_meta.get('input_digests')}, S0={s0_meta.get('input_digests')}) — "
            "same-delta alone is not the nested ablation"
        )

    if args.layer == LAYER_SECONDARY:
        # Descriptive layer: fixed four-arm roster, frozen here rather than
        # picked from primary's winners.
        arms = {a: p for a, p in arms.items() if a in SECONDARY_ARMS}
        artifact_paths = {a: p for a, p in artifact_paths.items() if a in SECONDARY_ARMS}
        artifact_sha256 = {a: p for a, p in artifact_sha256.items() if a in SECONDARY_ARMS}
        missing = SECONDARY_ARMS - set(arms)
        if missing:
            raise SystemExit(f"secondary roster incomplete: missing {sorted(missing)}")
        core = set(SECONDARY_ARMS)
    else:
        core = set(PRIMARY_CORE_ARMS)
    matrix = {
        "protocol": PROTOCOL,
        "arms": arms,
        "layer": args.layer,
        "gate_type": gate_section(args.layer, theta)["type"],
        "gate_theta": theta,
        "gate_theta_top_fraction": THETA_TOP_FRACTION,
        "gate_params": {"j": GATE_J, "probe_interval": GATE_PROBE_INTERVAL, "L": GATE_L},
        "gate_theta_scores_n": len(scores),
        "arm_yaml_sha256": {arm: _file_sha256(pathlib.Path(p)) for arm, p in arms.items()},
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
        # Declared at matrix level too: the runner checks each artifact's own
        # mode, but the matrix is what the launch ledger freezes, so the claim
        # this sweep makes has to be visible there as well.
        "certification_mode": CERTIFICATION_EMPIRICAL,
        "fit_record_paths": {k: str(v.resolve()) for k, v in fit_record_paths.items()},
        "fit_record_sha256": {k: _file_sha256(v) for k, v in fit_record_paths.items()},
        "core_arms": sorted(core),
        "descriptive_arms": sorted(set(arms) - core),
        "library_pkl": args.library_pkl,
        "library_sha256": lib_sha,
        "template": str(args.template),
    }
    matrix_path = out_dir / f"arm_matrix_{args.layer}.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    print(f"emitted {len(arms)} {args.layer} arms "
          f"(gate={matrix['gate_type']}) -> {matrix_path}")


if __name__ == "__main__":
    main()
