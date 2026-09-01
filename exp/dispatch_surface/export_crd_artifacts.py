"""Export Hysteretic Cumulative-Risk Re-anchoring (H-CRD) artifacts from a
Rev 1 primary surface (exploratory prototype, 2026-08-30; design in
logs/dispatch_surface_rev2_amendment_result.md section 12).

Same provenance chain as ``export_exploratory_surface`` (Rev 1 package by ROLE,
frozen table, exact development membership, recomputed final fit must
reproduce ``final_fit_digests``). Each artifact is a regular surface artifact
at the per-step cap ``delta = quantile(D_dev.y10, q)`` plus:

  * ``q_hat``          upper grid at level 1 - quantile_alpha (local admission u_a)
  * ``q_hat_central``  median grid (level 0.5) on the SAME edges (debt increment d_a)
  * ``s_edges``
  * ``meta.crd``       {gamma, beta, budget_mult, j_bad, l_max, min_recovery_misses, delta_reopen,
                       task_scale, upper_grid_sha256, central_grid_sha256}

``beta = budget_mult * delta`` (``budget_mult`` may be "inf"); ``task_scale``
is the per-task median of y10 on the development rows divided by the pooled
median (calibration-only, never reads a rollout outcome). No rollout outcome
enters any quantity. The export record keeps the frozen Phase 0 schema so the
sgrid emit path consumes it unchanged; the artifact NAME encodes the knobs.

Usage:
  python -m exp.dispatch_surface.export_crd_artifacts \
      --rev1-package-manifest <pkg>/MANIFEST.json --source-role artifact.dsp_sv \
      --table <table.jsonl> --quantiles 0.85 --gammas 1.0 --budget-mults 2,4 \
      --j-bad 3 --l-max 6 --out-dir <empty dir>
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
    SOURCE_ROLE_TO_FIT,
    _git_commit,
    delta_at_quantile,
    dev_mask_from_membership,
    validate_export_d0_binding,
)
from exp.dispatch_surface.fit_surface import (
    GRID_LADDER_S_ONLY,
    GRID_LADDER_SV,
    _digest_obj,
    export_boundaries,
    final_fit,
    load_table,
)
from exp.dispatch_surface.template_parity import assert_no_placeholders, assert_template_parity
from openpi.cache.components.crd_judge import CumulativeRiskJudge, JUDGE_VARIANT
from openpi.cache.components.surface_judge import (
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    load_surface_artifact,
)


def save_crd_artifact(artifact: SurfaceArtifact, q_hat: np.ndarray, q_central: np.ndarray,
                      s_edges: np.ndarray, path: pathlib.Path) -> None:
    """Surface artifact NPZ layout plus the CRD extras (same key names)."""
    artifact.validate()
    scalars = {
        "schema_version": artifact.schema_version, "k": artifact.k, "h_exec": artifact.h_exec,
        "start_t_ws": artifact.start_t_ws, "delta": artifact.delta, "quantile_alpha": artifact.quantile_alpha,
        "certification_mode": artifact.certification_mode, "uses_disagreement": artifact.uses_disagreement,
        "conformal_c": "inf" if artifact.conformal_c == np.inf else artifact.conformal_c,
        "n_calibration_episodes": artifact.n_calibration_episodes,
    }
    np.savez(
        path,
        w=np.asarray(artifact.w, dtype=np.float32),
        active_mask=np.asarray(artifact.active_mask, dtype=bool),
        v_bin_edges=np.asarray(artifact.v_bin_edges, dtype=np.float64),
        s_min_full=np.asarray(artifact.s_min_full, dtype=np.float64),
        s_min_warm=np.asarray(artifact.s_min_warm, dtype=np.float64),
        q_hat=np.asarray(q_hat, dtype=np.float64),
        q_hat_central=np.asarray(q_central, dtype=np.float64),
        s_edges=np.asarray(s_edges, dtype=np.float64),
        scalars_json=np.frombuffer(json.dumps(scalars).encode("utf-8"), dtype=np.uint8),
        contract_json=np.frombuffer(json.dumps(artifact.retrieval_contract, sort_keys=True).encode("utf-8"), dtype=np.uint8),
        meta_json=np.frombuffer(json.dumps(artifact.meta).encode("utf-8"), dtype=np.uint8),
    )


def _fmt(x) -> str:
    return "inf" if x is None or (isinstance(x, float) and np.isinf(x)) else f"{x:g}"


def _name(q: float, gamma: float, mult, j_bad, l_max) -> str:
    return (f"crd_q{q * 100:g}_g{_fmt(gamma)}_m{_fmt(mult)}_j{_fmt(j_bad)}_L{'none' if l_max is None else l_max}").replace(".", "")


def _parse_list(text: str, kind):
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() in ("inf", "none"):
            out.append(None if kind is int else float("inf"))
        else:
            out.append(kind(tok))
    return out


def export(args) -> dict:
    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    fit_role = SOURCE_ROLE_TO_FIT[args.source_role]
    source_path = pkgmod.verify_member(manifest, pkg, args.source_role)
    source_sha = pkgmod.member_sha(manifest, args.source_role)
    fit_path = pkgmod.verify_member(manifest, pkg, fit_role)
    fit_sha = pkgmod.member_sha(manifest, fit_role)
    fit_record = json.loads(fit_path.read_text())
    matrix = pkgmod.load_json_member(manifest, pkg, "matrix")
    verdict = pkgmod.load_json_member(manifest, pkg, "verdict")
    arm = args.source_role.split(".", 1)[1]
    if verdict["discipline"]["artifact_sha256"].get(arm) != source_sha or matrix["artifact_sha256"].get(arm) != source_sha:
        raise SystemExit("source artifact SHA is not the archived verdict/matrix authority")
    if matrix["fit_record_sha256"].get(fit_role.split(".", 1)[1]) != fit_sha:
        raise SystemExit("source fit record SHA is not the matrix authority")
    s_only = fit_record.get("s_only")
    if s_only not in (True, False) or (s_only != (args.source_role == "artifact.dsp_s0")):
        raise SystemExit("source role and fit record mode disagree")
    source = load_surface_artifact(str(source_path))
    if source.meta.get("posthoc_exploratory") or source.delta != fit_record.get("delta_star"):
        raise SystemExit("source artifact is not the Rev 1 primary at delta_star")
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
    alpha = float(fit_record["quantile_alpha"])
    ladder = GRID_LADDER_S_ONLY if s_only else GRID_LADDER_SV
    ff = final_fit(table, dev_mask, alpha=alpha, ladder=ladder)
    if ff is None:
        raise SystemExit("final fit hit the sparse-cell stop-loss on the frozen inputs")
    digests = {"s_edges": _digest_obj(np.asarray(ff.s_edges).tolist()), "v_edges": _digest_obj(np.asarray(ff.v_edges).tolist()),
               "q_deploy": _digest_obj(np.asarray(ff.q_hat).tolist()), "n_dev_rows": int(dev_mask.sum())}
    if digests != fit_record.get("final_fit_digests"):
        raise SystemExit("recomputed final fit does not reproduce the fit record's final_fit_digests")
    if not np.array_equal(np.asarray(ff.v_edges, dtype=np.float64), np.asarray(source.v_bin_edges, dtype=np.float64)):
        raise SystemExit("recomputed v edges differ from the source artifact")
    # Central (median) grid on the SAME edges: the debt increment d_a.
    fc = final_fit(table, dev_mask, alpha=0.5, edges=(ff.s_edges, ff.v_edges))
    if fc is None:
        raise SystemExit("central fit failed")
    if np.any(np.asarray(fc.q_hat) > np.asarray(ff.q_hat) + 1e-12):
        raise SystemExit("central (median) grid exceeds the upper grid; refusing to export a debt model above the admission risk")
    y10_dev = table.y10[dev_mask]
    tasks_dev = table.task[dev_mask]
    pooled = float(np.median(y10_dev))
    if not math.isfinite(pooled) or pooled <= 0:
        raise SystemExit(f"pooled development y10 median must be finite and > 0, got {pooled!r}")
    task_ids = sorted(int(t) for t in np.unique(tasks_dev))
    if task_ids != list(range(len(task_ids))):
        raise SystemExit(f"development task ids must be the complete contiguous range 0..N-1, got {task_ids}")
    task_scale = {int(t): float(np.median(y10_dev[tasks_dev == t]) / pooled) for t in np.unique(tasks_dev)}
    if any(not math.isfinite(scale) or scale <= 0 for scale in task_scale.values()):
        raise SystemExit(f"per-task development y10 scales must be finite and > 0, got {task_scale}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} must be empty")
    quantiles = _parse_list(args.quantiles, float)
    gammas = _parse_list(args.gammas, float)
    mults = _parse_list(args.budget_mults, float)
    j_bads = _parse_list(args.j_bad, int)
    l_maxs = _parse_list(args.l_max, int)
    if not all((quantiles, gammas, mults, j_bads, l_maxs)):
        raise SystemExit("every CRD parameter list must contain at least one value")
    if any(not math.isfinite(q) or not (0.0 < q < 1.0) for q in quantiles) \
            or any(not math.isfinite(g) or not (0.0 <= g <= 1.0) for g in gammas) \
            or any((not math.isfinite(m) and not math.isinf(m)) or m < 1.0 for m in mults):
        raise SystemExit("--quantiles in (0,1), --gammas in [0,1], --budget-mults >= 1 (or inf)")
    if any(j is not None and j < 1 for j in j_bads) or any(limit is not None and limit < 1 for limit in l_maxs) \
            or isinstance(args.min_recovery_misses, bool) or args.min_recovery_misses < 0:
        raise SystemExit("--j-bad/--l-max must be positive or inf/none; --min-recovery-misses must be >= 0")
    tag = "s_only" if s_only else "sv"
    record = {
        "protocol": "dispatch_surface_rev2_phase0", "posthoc_exploratory": True, "source_role": args.source_role,
        "family": "s0" if s_only else "sv", "rev1_package_manifest_sha256": manifest_sha,
        "source_artifact_sha256": source_sha, "source_fit_record_sha256": fit_sha, "table_sha256": table_sha,
        "d0_record_sha256": fit_record["d0_record_sha256"], "dev_membership_sha256": fit_record["dev_membership_sha256"],
        "final_fit_digests": digests, "quantile_method": QUANTILE_METHOD, "cost_model_digest": cost_model_digest(),
        "git_commit": _git_commit(), "python": sys.version.split()[0], "numpy": np.__version__, "artifacts": {},
    }
    for q in quantiles:
        delta = delta_at_quantile(y10_dev, q)
        full, warm = export_boundaries(ff.q_hat, ff.s_edges, delta)
        for gamma in gammas:
            for mult in mults:
                for j_bad in j_bads:
                    for l_max in l_maxs:
                        name = _name(q, gamma, mult, j_bad, l_max)
                        beta = None if np.isinf(mult) else mult * float(delta)
                        meta = dict(source.meta)
                        meta.update({
                            "posthoc_exploratory": True, "delta_quantile": q, "delta_name": name, "quantile_method": QUANTILE_METHOD,
                            "source_role": args.source_role, "source_artifact_sha256": source_sha,
                            "source_fit_record_sha256": fit_sha, "rev1_package_manifest_sha256": manifest_sha,
                            "judge_variant": JUDGE_VARIANT,
                            "crd": {"gamma": gamma, "beta": beta, "budget_mult": None if np.isinf(mult) else mult,
                                    "j_bad": j_bad, "l_max": l_max, "delta_reopen": float(delta),
                                    "min_recovery_misses": int(args.min_recovery_misses),
                                    "task_scale": {str(k): v for k, v in task_scale.items()},
                                    "central_level": 0.5, "upper_level": 1.0 - alpha,
                                    "upper_grid_sha256": _digest_obj(np.asarray(ff.q_hat).tolist()),
                                    "central_grid_sha256": _digest_obj(np.asarray(fc.q_hat).tolist())},
                        })
                        assert_no_placeholders(meta, what=f"artifact {name}")
                        artifact = SurfaceArtifact(
                            schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION, k=source.k, h_exec=source.h_exec, w=source.w,
                            active_mask=source.active_mask, start_t_ws=source.start_t_ws, delta=float(delta),
                            quantile_alpha=source.quantile_alpha, certification_mode=source.certification_mode,
                            uses_disagreement=source.uses_disagreement, v_bin_edges=source.v_bin_edges,
                            s_min_full=full, s_min_warm=warm, conformal_c=source.conformal_c,
                            n_calibration_episodes=source.n_calibration_episodes,
                            retrieval_contract=dict(source.retrieval_contract), meta=meta,
                        )
                        assert_template_parity(artifact, source, what=f"artifact {name}")
                        path = out_dir / f"surface_{tag}_{name}.npz"
                        save_crd_artifact(artifact, ff.q_hat, fc.q_hat, ff.s_edges, path)
                        # Exercise the production loader before the artifact is
                        # admitted to the export record.  This prevents an
                        # exporter/runtime schema drift from producing a
                        # cryptographically frozen but unloadable arm.
                        CumulativeRiskJudge(str(path))
                        record["artifacts"][name] = {"path": str(path.resolve()), "quantile": q, "delta": float(delta),
                                                     "output_sha256": pkgmod.file_sha256(path)}
                        print(f"{name}: delta {delta:.4f} beta {beta} j_bad {j_bad} l_max {l_max} "
                              f"min_recovery_misses {args.min_recovery_misses} -> {path.name}")
    rec_path = out_dir / EXPORT_RECORD_NAME
    rec_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    record["export_record_sha256"] = hashlib.sha256(rec_path.read_bytes()).hexdigest()
    print("task_scale:", {k: round(v, 3) for k, v in task_scale.items()})
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--source-role", required=True, choices=sorted(SOURCE_ROLE_TO_FIT))
    ap.add_argument("--table", required=True)
    ap.add_argument("--quantiles", required=True, help="per-step cap delta as D_dev.y10 quantiles, e.g. 0.85,0.90")
    ap.add_argument("--gammas", required=True, help="debt decay per step, e.g. 1.0,0.8")
    ap.add_argument("--budget-mults", required=True, help="beta as multiples of delta, e.g. 2,4,inf")
    ap.add_argument("--j-bad", default="3", help="consecutive region-MISS steps before RECOVERY, e.g. 3,inf")
    ap.add_argument("--l-max", default="6", help="FULL_HIT run fuse, e.g. 6,none")
    ap.add_argument("--min-recovery-misses", type=int, default=2,
                    help="executed MISSes required inside RECOVERY before reopening (production probe_interval - 1)")
    ap.add_argument("--out-dir", required=True)
    rec = export(ap.parse_args())
    print(json.dumps({"artifacts": sorted(rec["artifacts"]), "export_record_sha256": rec["export_record_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
