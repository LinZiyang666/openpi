"""Export exploratory (post-hoc) dispatch-surface artifacts at extra deltas.

Rev 2 Phase 0 (plan section 3.1). A Rev 1 primary artifact is the IMMUTABLE
template: ``w``, ``active_mask``, ``retrieval_contract``, ``quantile_alpha``,
``certification_mode``, ``k``, ``h_exec``, ``start_t_ws``, ``uses_disagreement``
and ``v_bin_edges`` are copied field by field. Only the final q-grid, the
delta and the exported boundaries are recomputed -- from the frozen table and
the fit record's exact development membership -- and the recomputed grid must
reproduce the fit record's ``final_fit_digests`` or the export is refused.

Provenance is a digest chain resolved through the Rev 1 discipline package
by ROLE (never by the historical absolute paths inside the frozen files):

    export_record -> output artifact
    export_record -> source artifact (package member) -> matrix declared SHA -> verdict
    export_record -> source fit record (package member) -> matrix -> verdict
    fit_record.input_digests.{table, weights_npz, cache_yaml, d0_record, rebuild_record, split_manifest}
    d0.inputs.files.{table, weights_npz, cache_yaml, library_pkl, noise_sidecar}  (declared digests only)

No historical file is re-opened: ``validate_export_d0_binding`` compares
DECLARED digests and recomputes the D0 attestation rollup from the record's
own dict (G1R3-B1, G1R4-B1). The artifact meta never carries a placeholder.

Usage:
  uv run python -m exp.dispatch_surface.export_exploratory_surface \
      --rev1-package-manifest <pkg>/MANIFEST.json --source-role artifact.dsp_sv \
      --table <table.jsonl> --quantiles 0.85 --out-dir <empty dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest
from exp.dispatch_surface.d0_check import D0_PROTOCOL, _canonical_digest
from exp.dispatch_surface.template_parity import assert_no_placeholders, assert_template_parity
from exp.dispatch_surface.fit_surface import (
    GRID_LADDER_S_ONLY,
    GRID_LADDER_SV,
    Table,
    _digest_obj,
    export_boundaries,
    final_fit,
    load_table,
)
from openpi.cache.components.surface_judge import (
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    load_surface_artifact,
    save_surface_artifact,
)

QUANTILE_METHOD = "linear"
D0_FILE_KEYS = frozenset({"table", "library_pkl", "noise_sidecar", "cache_yaml", "weights_npz"})
SOURCE_ROLE_TO_FIT = {"artifact.dsp_sv": "fit.sv", "artifact.dsp_s0": "fit.s0"}
EXPORT_RECORD_NAME = "export_record.json"


def delta_at_quantile(y10_dev: np.ndarray, q: float) -> float:
    """The Rev 1 grid call (``fit_surface.py`` delta grid), one quantile at a time."""
    return float(np.percentile(np.asarray(y10_dev, dtype=np.float64), 100.0 * q, method=QUANTILE_METHOD))


def dev_mask_from_membership(table: Table, membership) -> np.ndarray:
    """Rebuild the development mask from the fit record's exact episode list."""
    # fit_surface._development_audit records (episode, task, init) per episode
    wanted = {str(m[0]) if isinstance(m, (list, tuple)) else str(m) for m in membership}
    eps = table.episode.astype(str)
    mask = np.isin(eps, list(wanted))
    if len(set(eps[mask])) != len(wanted):
        raise SystemExit("table does not contain every development episode of the fit record")
    return mask


def validate_export_d0_binding(
    d0: dict, fit_record: dict, manifest: dict, package_dir: pathlib.Path,
    table_sha256: str, source_artifact: SurfaceArtifact,
) -> None:
    """Digest-chain binding of the archived D0 record. Opens no historical path."""
    if d0.get("protocol") != D0_PROTOCOL or d0.get("D0") != "PASS":
        raise SystemExit("D0 record is not a PASS of the frozen D0 protocol")
    inputs = d0.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("files"), dict):
        raise SystemExit("D0 record lacks its input attestation")
    files = inputs["files"]
    if set(files) != D0_FILE_KEYS:
        raise SystemExit(f"D0 attestation file set {sorted(files)} != {sorted(D0_FILE_KEYS)}")
    for section in ("query_h5", "library_h5", "policy"):
        if section not in inputs:
            raise SystemExit(f"D0 attestation lacks {section}")
    rollup = inputs.get("rollup_sha256")
    body = {k: v for k, v in inputs.items() if k != "rollup_sha256"}
    if rollup != _canonical_digest(body):
        raise SystemExit("D0 attestation rollup does not match its own content")
    if not (d0.get("census") or {}).get("passed"):
        raise SystemExit("D0 census did not pass")
    for check, flag in (("check1_self_resume_parity", "passed"),
                        ("check2_payload_sidecar_identity", "passed"),
                        ("check3_path_decomposition", "table_semantics_passed")):
        if not (d0.get(check) or {}).get(flag):
            raise SystemExit(f"D0 {check} did not pass")
    digests = fit_record.get("input_digests") or {}
    if files["table"].get("sha256") != table_sha256:
        raise SystemExit("D0 attested table != the table supplied to the exporter")
    if digests.get("table") != table_sha256:
        raise SystemExit("fit record input table digest != the table supplied to the exporter")
    for key in ("weights_npz", "cache_yaml"):
        if files[key].get("sha256") != digests.get(key):
            raise SystemExit(f"D0 attested {key} != fit record input digest")
    if fit_record.get("d0_record_sha256") != digests.get("d0_record"):
        raise SystemExit("fit record d0_record digests disagree")
    rebuild = pkgmod.load_json_member(manifest, package_dir, "rebuild")
    if pkgmod.member_sha(manifest, "rebuild") != digests.get("rebuild_record"):
        raise SystemExit("package rebuild member != fit record rebuild_record digest")
    if pkgmod.member_sha(manifest, "split_manifest") != digests.get("split_manifest"):
        raise SystemExit("package split_manifest member != fit record split_manifest digest")
    if rebuild.get("split_manifest_sha256") != pkgmod.member_sha(manifest, "split_manifest"):
        raise SystemExit("rebuild record split_manifest_sha256 != package split_manifest member")
    if files["library_pkl"].get("sha256") != rebuild.get("library_sha256"):
        raise SystemExit("D0 attested library_pkl != rebuild record library_sha256")
    if files["noise_sidecar"].get("sha256") != rebuild.get("noise_sidecar_sha256"):
        raise SystemExit("D0 attested noise_sidecar != rebuild record noise_sidecar_sha256")
    contract = source_artifact.retrieval_contract
    if contract.get("library_sha256") != rebuild.get("library_sha256"):
        raise SystemExit("source artifact contract library_sha256 != rebuild record")
    if contract.get("library_entry_count") != rebuild.get("entry_count"):
        raise SystemExit("source artifact contract library_entry_count != rebuild record")
    policy = inputs.get("policy") or {}
    if contract.get("policy_fingerprint") != policy.get("policy_fingerprint"):
        raise SystemExit("source artifact contract policy_fingerprint != D0 attested policy")
    if d0.get("suite") != fit_record.get("d0_suite"):
        raise SystemExit("D0 suite != fit record d0_suite")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance best effort, recorded as unknown
        return "unknown"


def export(args) -> dict:
    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    if args.source_role not in SOURCE_ROLE_TO_FIT:
        raise SystemExit(f"--source-role must be one of {sorted(SOURCE_ROLE_TO_FIT)}")
    fit_role = SOURCE_ROLE_TO_FIT[args.source_role]
    source_path = pkgmod.verify_member(manifest, pkg, args.source_role)
    source_sha = pkgmod.member_sha(manifest, args.source_role)
    fit_path = pkgmod.verify_member(manifest, pkg, fit_role)
    fit_sha = pkgmod.member_sha(manifest, fit_role)
    fit_record = json.loads(fit_path.read_text())
    matrix = pkgmod.load_json_member(manifest, pkg, "matrix")
    verdict = pkgmod.load_json_member(manifest, pkg, "verdict")
    arm = args.source_role.split(".", 1)[1]
    if (verdict["discipline"]["artifact_sha256"].get(arm) != source_sha
            or matrix["artifact_sha256"].get(arm) != source_sha):
        raise SystemExit("source artifact SHA is not the archived verdict/matrix authority")
    if matrix["fit_record_sha256"].get(fit_role.split(".", 1)[1]) != fit_sha:
        raise SystemExit("source fit record SHA is not the matrix authority")
    s_only = fit_record.get("s_only")
    if s_only not in (True, False) or (s_only != (args.source_role == "artifact.dsp_s0")):
        raise SystemExit("source role and fit record mode disagree")
    if fit_record.get("stop_loss") is not None:
        raise SystemExit("fit record is a stop-loss, not a completed fit")

    source = load_surface_artifact(str(source_path))
    if source.meta.get("posthoc_exploratory"):
        raise SystemExit("source artifact is itself exploratory; only Rev 1 primaries are templates")
    if source.delta != fit_record.get("delta_star"):
        raise SystemExit("source artifact delta != fit record delta_star")

    table_path = pathlib.Path(args.table)
    table_sha = pkgmod.file_sha256(table_path)
    d0 = pkgmod.load_json_member(manifest, pkg, "d0")
    if pkgmod.member_sha(manifest, "d0") != fit_record.get("d0_record_sha256"):
        raise SystemExit("package d0 member != fit record d0_record_sha256")
    validate_export_d0_binding(d0, fit_record, manifest, pkg, table_sha, source)

    table = load_table(str(table_path), ref_mode="fresh")
    membership = fit_record.get("dev_membership")
    if not membership or _digest_obj(membership) != fit_record.get("dev_membership_sha256"):
        raise SystemExit("fit record development membership is missing or its digest drifted")
    dev_mask = dev_mask_from_membership(table, membership)
    ladder = GRID_LADDER_S_ONLY if s_only else GRID_LADDER_SV
    ff = final_fit(table, dev_mask, alpha=float(fit_record["quantile_alpha"]), ladder=ladder)
    if ff is None:
        raise SystemExit("final fit hit the sparse-cell stop-loss on the frozen inputs")
    digests = {
        "s_edges": _digest_obj(np.asarray(ff.s_edges).tolist()),
        "v_edges": _digest_obj(np.asarray(ff.v_edges).tolist()),
        "q_deploy": _digest_obj(np.asarray(ff.q_hat).tolist()),
        "n_dev_rows": int(dev_mask.sum()),
    }
    if digests != fit_record.get("final_fit_digests"):
        raise SystemExit("recomputed final fit does not reproduce the fit record's final_fit_digests")
    if not np.array_equal(np.asarray(ff.v_edges, dtype=np.float64), np.asarray(source.v_bin_edges, dtype=np.float64)):
        raise SystemExit("recomputed v edges differ from the source artifact")
    grid = fit_record.get("delta_grid") or []
    y10_dev = table.y10[dev_mask]
    for q, expected in ((0.8, grid[-2] if len(grid) >= 2 else None), (0.9, grid[-1] if grid else None)):
        if expected is not None and abs(delta_at_quantile(y10_dev, q) - float(expected)) > 1e-9:
            raise SystemExit(f"p{int(q*100)} does not reproduce the fit record delta grid")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} must be empty")
    quantiles = [float(q) for q in args.quantiles.split(",") if q.strip()]
    if not quantiles or any(not (0.0 < q < 1.0) for q in quantiles) or len(set(quantiles)) != len(quantiles):
        raise SystemExit("--quantiles must be distinct values in (0, 1)")
    tag = "s_only" if s_only else "sv"
    record = {
        "protocol": "dispatch_surface_rev2_phase0",
        "posthoc_exploratory": True,
        "source_role": args.source_role,
        "family": "s0" if s_only else "sv",
        "rev1_package_manifest_sha256": manifest_sha,
        "source_artifact_sha256": source_sha,
        "source_fit_record_sha256": fit_sha,
        "table_sha256": table_sha,
        "d0_record_sha256": fit_record["d0_record_sha256"],
        "dev_membership_sha256": fit_record["dev_membership_sha256"],
        "final_fit_digests": digests,
        "quantile_method": QUANTILE_METHOD,
        "cost_model_digest": cost_model_digest(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "artifacts": {},
    }
    for q in quantiles:
        name = f"p{q * 100:g}".replace(".", "")
        delta = delta_at_quantile(y10_dev, q)
        full, warm = export_boundaries(ff.q_hat, ff.s_edges, delta)
        meta = dict(source.meta)
        meta.update({
            "posthoc_exploratory": True,
            "delta_quantile": q,
            "delta_name": name,
            "quantile_method": QUANTILE_METHOD,
            "source_role": args.source_role,
            "source_artifact_sha256": source_sha,
            "source_fit_record_sha256": fit_sha,
            "rev1_package_manifest_sha256": manifest_sha,
        })
        assert_no_placeholders(meta, what=f"artifact {name}")
        artifact = SurfaceArtifact(
            schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION,
            k=source.k, h_exec=source.h_exec, w=source.w, active_mask=source.active_mask,
            start_t_ws=source.start_t_ws, delta=float(delta),
            quantile_alpha=source.quantile_alpha, certification_mode=source.certification_mode,
            uses_disagreement=source.uses_disagreement, v_bin_edges=source.v_bin_edges,
            s_min_full=full, s_min_warm=warm, conformal_c=source.conformal_c,
            n_calibration_episodes=source.n_calibration_episodes,
            retrieval_contract=dict(source.retrieval_contract), meta=meta,
        )
        assert_template_parity(artifact, source, what=f"artifact {name}")
        path = out_dir / f"surface_{tag}_{name}.npz"
        save_surface_artifact(artifact, str(path))
        record["artifacts"][name] = {
            "path": str(path.resolve()), "quantile": q, "delta": float(delta),
            "output_sha256": pkgmod.file_sha256(path),
        }
    rec_path = out_dir / EXPORT_RECORD_NAME
    rec_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    record["export_record_sha256"] = hashlib.sha256(rec_path.read_bytes()).hexdigest()
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--source-role", required=True, choices=sorted(SOURCE_ROLE_TO_FIT))
    ap.add_argument("--table", required=True)
    ap.add_argument("--quantiles", required=True, help="comma list of D_dev.y10 quantiles in (0,1)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    record = export(args)
    print(json.dumps({k: v for k, v in record.items() if k in ("source_role", "artifacts", "export_record_sha256")}, indent=2))


if __name__ == "__main__":
    main()
