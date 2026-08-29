"""Emit the precheck eval yamls + arm matrix (dynamic 5-7 arms).

Arms (plan section 4.6): T1-T3 calibrated two-threshold baselines
(start_t=0.3 tier, thresholds solved by the single shared
``derive_thresholds`` implementation on the fit-union-cal score
distribution), S0 (s-only surface artifact at delta*), SV (primary surface),
and SV-/SV+ only when the corresponding grid neighbour artifact exists.

Every yaml is derived from the gate-line template: judge section replaced,
``preload_path`` pointed at the rebuilt dispatch library (with an assert that
the preload file's sha256 equals the artifact contract's library_sha256), and
``write_policy: never`` enforced. The emitted arm matrix records the actual
arm set for the launch manifest.

The GATE is re-solved too, not inherited. ``solve_gtp`` states the rule the
threshold line runs on: the gate theta and the judge cuts are quantiles of the
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
import json
import pathlib

import yaml

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--fit-dir", required=True,
                    help="fit_surface output dir (artifacts + fit_record.json)")
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--layer", required=True, choices=[LAYER_PRIMARY, LAYER_SECONDARY])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

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
