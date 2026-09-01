"""Export RIT-PL artifacts (piecewise-linear risk curves, IR- or delta-addressed)
from the Rev 1 s-only template (plan ``logs/rit_pl_ir_ladder_plan.log.md``).

Provenance is the Phase 0 exporter's digest chain resolved through the Rev 1
discipline package by ROLE (``artifact.dsp_s0`` template, ``fit.s0`` for the
exact development membership, archived D0 record), with one difference: the
frozen 12-bin ``final_fit_digests`` are NOT reproduced -- RIT-PL is a new
estimator with its own ``fit_record_pl.json`` which is written FIRST and bound
by SHA into every artifact meta and into the export record (write order:
fit record -> artifacts -> export record).

Addressing (mutually exclusive):
  --target-ir 50,55,...   invert the predicted inference ratio on the
                          development rows (nearest attainable point, gap
                          recorded; |gap| > IR_MAX_GAP refused)
  --quantiles 0.85,0.925  delta = quantile of D_dev.y10 (numpy ``linear``,
                          the Phase 0 call)

Artifacts keep the runtime schema unchanged (two cut constants); ``meta``
carries the estimator tag, addressing, floor information and the fit-record
SHA. Nothing here reads a rollout outcome.

Usage:
  uv run python -m exp.dispatch_surface.export_rit_pl \
      --rev1-package-manifest <pkg>/MANIFEST.json --table <table.jsonl> \
      --target-ir 50,55,60,65,70,75,80,85,90,95 --out-dir <empty dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest
from exp.dispatch_surface.export_exploratory_surface import (
    EXPORT_RECORD_NAME,
    QUANTILE_METHOD,
    _git_commit,
    delta_at_quantile,
    dev_mask_from_membership,
    validate_export_d0_binding,
)
from exp.dispatch_surface.fit_surface import _digest_obj, load_table
from exp.dispatch_surface.rit_pl import (
    EPS_TOTAL,
    ESTIMATOR,
    IR_MAX_GAP,
    IR_TOL,
    KNOT_LADDER,
    PROTOCOL_RIT_PL,
    attainable_range,
    choose_knots,
    cuts,
    delta_for_ir,
    ecdf_quantile,
    fit_pl_quantile,
    fit_record_fields,
    floor_info,
    ir_curve,
    pl_fit_digests,
    predicted_ir,
)
from exp.dispatch_surface.template_parity import (
    RIT_PL_FAMILY,
    RIT_PL_SOURCE_ROLE,
    assert_no_placeholders,
    assert_rit_pl_artifact_coherence,
    assert_rit_pl_export_record_schema,
    assert_rit_pl_fit_record,
    assert_template_parity,
)
from openpi.cache.components.surface_judge import (
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    load_surface_artifact,
    save_surface_artifact,
)

FIT_ROLE = "fit.s0"
FIT_RECORD_NAME = "fit_record_pl.json"
ECDF_METHOD = "ecdf_right_continuous"

__all__ = ["PROTOCOL_RIT_PL", "FIT_RECORD_NAME", "export", "main"]


# ------------------------------------------------------------------
# Argument parsing helpers
# ------------------------------------------------------------------


def parse_targets(text: str, kind: str) -> list[float]:
    """Comma list of strictly increasing floats; ``kind`` is ``"ir"`` (0 < x < 100)
    or ``"q"`` (0 < x < 1). Duplicates, non-increasing order and out-of-range
    values are refused."""
    items = [t.strip() for t in text.split(",") if t.strip()]
    if not items:
        raise SystemExit(f"--{'target-ir' if kind == 'ir' else 'quantiles'} needs at least one value")
    try:
        values = [float(t) for t in items]
    except ValueError as exc:
        raise SystemExit(f"non-numeric value in list: {exc}") from exc
    lo, hi = (0.0, 100.0) if kind == "ir" else (0.0, 1.0)
    if any(not (lo < v < hi) for v in values):
        raise SystemExit(f"every value must lie in ({lo:g}, {hi:g}) exclusive")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise SystemExit("values must be strictly increasing (no duplicates)")
    return values


def artifact_name(addressing: str, value: float) -> str:
    """``ir80`` / ``ir82p5`` for target-IR addressing, ``p85`` / ``p925`` for quantiles."""
    if addressing == "target_ir":
        return "ir" + f"{value:g}".replace(".", "p")
    return "p" + f"{value * 100:g}".replace(".", "")


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def export(args) -> dict:
    """Export one RIT-PL ladder from the Rev 1 s-only template.

    ``args`` carries ``rev1_package_manifest``, ``table``, ``out_dir`` and
    exactly one of ``target_ir`` / ``quantiles`` (comma lists). Writes the PL
    fit record, the artifacts and the export record into the empty ``out_dir``
    and returns the export record plus ``export_record_sha256`` and the
    attainable IR range. Every refusal is a ``SystemExit``."""
    if bool(args.target_ir) == bool(args.quantiles):
        raise SystemExit("exactly one of --target-ir / --quantiles is required")
    addressing = "target_ir" if args.target_ir else "quantile"
    values = parse_targets(args.target_ir, "ir") if addressing == "target_ir" else parse_targets(args.quantiles, "q")

    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    source_path = pkgmod.verify_member(manifest, pkg, RIT_PL_SOURCE_ROLE)
    source_sha = pkgmod.member_sha(manifest, RIT_PL_SOURCE_ROLE)
    fit_path = pkgmod.verify_member(manifest, pkg, FIT_ROLE)
    fit_sha = pkgmod.member_sha(manifest, FIT_ROLE)
    fit_record = json.loads(fit_path.read_text())
    matrix = pkgmod.load_json_member(manifest, pkg, "matrix")
    verdict = pkgmod.load_json_member(manifest, pkg, "verdict")
    if (verdict["discipline"]["artifact_sha256"].get("dsp_s0") != source_sha
            or matrix["artifact_sha256"].get("dsp_s0") != source_sha):
        raise SystemExit("source artifact SHA is not the archived verdict/matrix authority")
    if matrix["fit_record_sha256"].get("s0") != fit_sha:
        raise SystemExit("source fit record SHA is not the matrix authority")
    if fit_record.get("s_only") is not True:
        raise SystemExit("RIT-PL is exported from the s-only fit record only")
    if fit_record.get("stop_loss") is not None:
        raise SystemExit("fit record is a stop-loss, not a completed fit")

    source = load_surface_artifact(str(source_path))
    if source.meta.get("posthoc_exploratory"):
        raise SystemExit("source artifact is itself exploratory; only Rev 1 primaries are templates")
    if source.delta != fit_record.get("delta_star"):
        raise SystemExit("source artifact delta != fit record delta_star")
    if source.uses_disagreement:
        raise SystemExit("RIT-PL template must be the s-only artifact")

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
    s_dev, y7_dev, y10_dev = table.s[dev_mask], table.y7[dev_mask], table.y10[dev_mask]
    alpha = float(fit_record["quantile_alpha"])

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} must be empty")

    chosen = choose_knots(s_dev, KNOT_LADDER)
    if chosen is None:
        raise SystemExit("knot ladder exhausted on the development rows (stop-loss)")
    knots, n_seg_req = chosen
    fit = fit_pl_quantile(s_dev, y7_dev, y10_dev, knots, n_seg_req=n_seg_req, alpha=alpha, eps_total=EPS_TOTAL)
    digests = pl_fit_digests(fit)
    ir_lo, ir_hi = attainable_range(fit, s_dev)

    # -- per-target solve (before any file is written) ------------------
    solved: dict[str, dict] = {}
    for value in values:
        name = artifact_name(addressing, value)
        if name in solved:
            raise SystemExit(f"duplicate artifact name {name}")
        if addressing == "target_ir":
            try:
                res = delta_for_ir(fit, s_dev, value, tol=IR_TOL)
            except ValueError as exc:
                raise SystemExit(f"target IR {value:g}: {exc}") from exc
            if abs(res["ir_gap"]) > IR_MAX_GAP:
                raise SystemExit(
                    f"target IR {value:g} is not attainable within {IR_MAX_GAP} pt: nearest predicted "
                    f"{res['predicted_ir']:.3f} (bracket {res['bracket']['ir_hi']:.3f} .. {res['bracket']['ir_lo']:.3f})")
            if abs(res["ir_gap"]) > IR_TOL:
                print(f"warning: target IR {value:g} attained at {res['predicted_ir']:.3f} (gap {res['ir_gap']:+.3f} pt)")
            delta, theta_full, theta_warm = res["delta"], res["theta_full"], res["theta_warm"]
            quantile, target_ir, ir_gap = ecdf_quantile(y10_dev, delta), float(value), float(res["ir_gap"])
        else:
            delta = delta_at_quantile(y10_dev, value)
            theta_full, theta_warm = cuts(fit, delta)
            quantile, target_ir, ir_gap = float(value), None, None
        if not (math.isfinite(delta) and delta > 0.0):
            raise SystemExit(f"{name}: delta {delta!r} violates the artifact contract (finite, > 0)")
        if math.isinf(theta_full) and math.isinf(theta_warm):
            raise SystemExit(f"{name}: both cuts are +inf (all-MISS); the always-full anchor is measured, not exported")
        solved[name] = {"value": float(value), "delta": float(delta), "theta_full": float(theta_full),
                        "theta_warm": float(theta_warm), "predicted_ir": float(predicted_ir(s_dev, theta_full, theta_warm)),
                        "quantile": float(quantile), "target_ir": target_ir, "ir_gap": ir_gap,
                        "floor_info": floor_info(fit, s_dev, delta)}

    # -- 1. PL fit record (written first, bound by SHA below) -----------
    identity = {
        "rev1_package_manifest_sha256": manifest_sha,
        "source_artifact_sha256": source_sha,
        "source_fit_record_sha256": fit_sha,
        "d0_record_sha256": fit_record["d0_record_sha256"],
        "table_sha256": table_sha,
        "dev_membership_sha256": fit_record["dev_membership_sha256"],
        "cost_model_digest": cost_model_digest(),
        "estimator": ESTIMATOR,
    }
    fit_rec = dict(identity)
    fit_rec.update(fit_record_fields(fit))
    fit_rec.update({
        "n_dev_rows": int(dev_mask.sum()),
        "pl_fit_digests": digests,
        "ir_curve": [[d, ir] for d, ir in ir_curve(fit, s_dev)],
        "s_range": [float(s_dev.min()), float(s_dev.max())],
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    })
    fit_rec_path = out_dir / FIT_RECORD_NAME
    fit_rec_path.write_text(json.dumps(fit_rec, indent=2, sort_keys=True))
    fit_rec_sha = _sha(fit_rec_path)

    # -- 2. artifacts ---------------------------------------------------
    record = dict(identity)
    record.update({
        "protocol": PROTOCOL_RIT_PL,
        "posthoc_exploratory": True,
        "source_role": RIT_PL_SOURCE_ROLE,
        "family": RIT_PL_FAMILY,
        "addressing": addressing,
        "pl_fit_record_path": str(fit_rec_path.resolve()),
        "pl_fit_record_sha256": fit_rec_sha,
        "pl_fit_digests": digests,
        "eps_total": float(fit.eps_total),
        "n_seg": int(fit.n_seg),
        "quantile_method": QUANTILE_METHOD if addressing == "quantile" else ECDF_METHOD,
        "git_commit": fit_rec["git_commit"],
        "python": fit_rec["python"],
        "numpy": fit_rec["numpy"],
        "artifacts": {},
    })
    artifact_cuts: dict[str, tuple[float, float]] = {}
    for name, sol in solved.items():
        meta = dict(source.meta)
        meta.update({
            "posthoc_exploratory": True,
            "estimator": ESTIMATOR,
            "family": RIT_PL_FAMILY,
            "addressing": addressing,
            "n_seg_req": int(fit.n_seg_req),
            "n_seg": int(fit.n_seg),
            "eps_total": float(fit.eps_total),
            "target_ir": sol["target_ir"],
            "predicted_ir": sol["predicted_ir"],
            "ir_gap": sol["ir_gap"],
            "quantile": sol["quantile"],
            "floor_info": sol["floor_info"],
            "s_range": fit_rec["s_range"],
            "source_role": RIT_PL_SOURCE_ROLE,
            "source_artifact_sha256": source_sha,
            "source_fit_record_sha256": fit_sha,
            "rev1_package_manifest_sha256": manifest_sha,
            "pl_fit_record_sha256": fit_rec_sha,
            "pl_fit_digests": digests,
        })
        assert_no_placeholders(meta, what=f"artifact {name}")
        artifact = SurfaceArtifact(
            schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION,
            k=source.k, h_exec=source.h_exec, w=source.w, active_mask=source.active_mask,
            start_t_ws=source.start_t_ws, delta=sol["delta"],
            quantile_alpha=source.quantile_alpha, certification_mode=source.certification_mode,
            uses_disagreement=False, v_bin_edges=source.v_bin_edges,
            s_min_full=np.asarray([sol["theta_full"]], dtype=np.float64),
            s_min_warm=np.asarray([sol["theta_warm"]], dtype=np.float64),
            conformal_c=source.conformal_c, n_calibration_episodes=source.n_calibration_episodes,
            retrieval_contract=dict(source.retrieval_contract), meta=meta,
        )
        assert_template_parity(artifact, source, what=f"artifact {name}")
        path = out_dir / f"surface_{RIT_PL_FAMILY}_{name}.npz"
        save_surface_artifact(artifact, str(path))
        artifact_cuts[name] = (sol["theta_full"], sol["theta_warm"])
        record["artifacts"][name] = {
            "path": str(path.resolve()), "addressing": addressing, "target_ir": sol["target_ir"],
            "predicted_ir": sol["predicted_ir"], "ir_gap": sol["ir_gap"], "quantile": sol["quantile"],
            "delta": sol["delta"], "theta_full": sol["theta_full"], "theta_warm": sol["theta_warm"],
            "floor_info": sol["floor_info"], "output_sha256": _sha(path),
        }

    # -- 3. export record; self-check with the emitter's validators -------
    assert_rit_pl_export_record_schema(record, what="export record", cost_model_digest=cost_model_digest())
    assert_rit_pl_fit_record(fit_rec, record, what="fit record", artifact_cuts=artifact_cuts)
    for name, art_rec in record["artifacts"].items():
        assert_rit_pl_artifact_coherence(load_surface_artifact(art_rec["path"]), art_rec, record, fit_rec, what=name)
    rec_path = out_dir / EXPORT_RECORD_NAME
    rec_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    record["export_record_sha256"] = _sha(rec_path)
    record["attainable_ir_range"] = [ir_lo, ir_hi]
    return record


def main() -> None:
    """Command-line entry point (see the module docstring for usage)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--table", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target-ir", default="", help="comma list of target inference ratios in percent, strictly increasing")
    mode.add_argument("--quantiles", default="", help="comma list of D_dev.y10 quantiles in (0,1), strictly increasing")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    record = export(args)
    print(json.dumps({k: v for k, v in record.items()
                      if k in ("addressing", "artifacts", "export_record_sha256", "attainable_ir_range")}, indent=2))


if __name__ == "__main__":
    main()
