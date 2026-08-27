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

# Pre-registered baseline grid: (f_FH %, f_WS %) cells around the known
# threshold-pareto sweet spots.
THRESHOLD_CELLS = ((30, 20), (50, 20), (70, 10))
WS_START_T = 0.3


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


def _emit(template: dict, out_path: pathlib.Path, judge_section: dict,
          preload_path: str) -> None:
    doc = json.loads(json.dumps(template))  # deep copy without yaml anchors
    doc["checkpoints"]["cp1"]["judge"] = judge_section
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
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds
    from openpi.cache.components.surface_judge import load_surface_artifact

    template = yaml.safe_load(open(args.template))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_sha = _file_sha256(pathlib.Path(args.library_pkl))

    arms: dict[str, str] = {}

    # -- threshold baseline arms -----------------------------------
    scores = _load_scores(args.table)
    for fh, ws in THRESHOLD_CELLS:
        t_fh, t_ws = derive_thresholds(scores, fh / 100.0, ws / 100.0)
        arm_id = f"dsp_t_fh{fh}_ws{ws}"
        judge = {
            "type": "threshold",
            "threshold": float(t_fh),
            "warm_tiers": [{"threshold": float(t_ws), "start_t": WS_START_T}],
        }
        path = out_dir / f"{arm_id}.yaml"
        _emit(template, path, judge, args.library_pkl)
        arms[arm_id] = str(path)

    # -- surface arms (dynamic cardinality) ------------------------
    fit_dir = pathlib.Path(args.fit_dir)
    surface_specs = [
        ("dsp_s0", fit_dir / "surface_s_only_primary.npz"),
        ("dsp_sv", fit_dir / "surface_sv_primary.npz"),
        ("dsp_sv_minus", fit_dir / "surface_sv_minus.npz"),
        ("dsp_sv_plus", fit_dir / "surface_sv_plus.npz"),
    ]
    deltas: dict[str, float] = {}
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
        deltas[arm_id] = artifact.delta
        judge = {
            "type": "dispatch_surface",
            "surface_artifact_path": str(artifact_path),
        }
        path = out_dir / f"{arm_id}.yaml"
        _emit(template, path, judge, args.library_pkl)
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

    core = {"dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10", "dsp_s0", "dsp_sv"}
    matrix = {
        "arms": arms,
        "arm_yaml_sha256": {arm: _file_sha256(pathlib.Path(p)) for arm, p in arms.items()},
        "core_arms": sorted(core),
        "descriptive_arms": sorted(set(arms) - core),
        "library_pkl": args.library_pkl,
        "library_sha256": lib_sha,
        "template": str(args.template),
    }
    matrix_path = out_dir / "arm_matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    print(f"emitted {len(arms)} arms -> {matrix_path}")


if __name__ == "__main__":
    main()
