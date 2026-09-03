"""RIT-PL export for the RIT-Pareto line: shadow table -> IR-addressed artifacts.

Two subcommands:

  calib-yaml  The calibration retrieval yaml for one suite: the GTP server
              template (``exp.gate_threshold_pareto.libraries.TEMPLATE``) with
              ``judge: always_hit`` / ``gate: always_search`` and the library
              to calibrate on. ``build_dispatch_table`` refuses any yaml whose
              ``preload_path`` differs from its ``--library-pkl``.

  fit         Fit the RIT-PL curves (``exp.dispatch_surface.rit_pl``) on EVERY
              row of the shadow table (fit + cal; RIT-PL carries no conformal
              split), invert each target inference ratio to a delta and the two
              cuts, and write one ``dispatch_surface`` judge artifact per target
              with the retrieval contract of the calibrated library. The gate
              theta for the H-gate layer is solved on the same shadow scores
              with the GTP convention (top ``THETA_TOP_FRACTION`` admitted).

Provenance is a flat record (table / weights / yaml / library digests, cost
model digest, fit digests, per-target solution) rather than the Rev 1
package chain: this line is a post-hoc Pareto trace on the GTP libraries, not
a confirmation-chain input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import yaml

from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest, unit_cost_table
from exp.dispatch_surface.fit_surface import N_STEPS, load_table
from exp.dispatch_surface.rit_pl import (
    EPS_TOTAL,
    ESTIMATOR,
    IR_MAX_GAP,
    KNOT_LADDER,
    attainable_range,
    choose_knots,
    cuts,
    delta_for_ir,
    fit_pl_quantile,
    fit_record_fields,
    floor_info,
    ir_curve,
    pl_fit_digests,
    predicted_ir,
)
from exp.gate_threshold_pareto import libraries as libs
from exp.gate_threshold_pareto.solve_gtp import THETA_TOP_FRACTION
from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds
from openpi.cache.components.surface_judge import (
    CERTIFICATION_EMPIRICAL,
    PINNED_START_T_WS,
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    load_surface_artifact,
    save_surface_artifact,
)

PROTOCOL = "rit_pareto_v1"
FAMILY = "rit"
DEFAULT_TARGETS = tuple(float(x) for x in range(20, 100, 5))  # 20 .. 95
DEFAULT_ALPHA = 0.05
DEFAULT_H_EXEC = 5
FIT_RECORD_NAME = "fit_record.json"
EXPORT_RECORD_NAME = "export_record.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(libs.REPO_ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def target_name(target: float) -> str:
    """``ir20`` for integral targets, ``ir82p5`` otherwise."""
    if float(target).is_integer():
        return f"ir{int(target):02d}"
    return "ir" + f"{target:g}".replace(".", "p")


def parse_targets(text: str) -> list[float]:
    vals = [float(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise SystemExit("--target-ir is empty")
    if any(not (0.0 < v < 100.0) for v in vals):
        raise SystemExit(f"targets must lie in (0, 100): {vals}")
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise SystemExit(f"targets must be strictly increasing: {vals}")
    return vals


def build_calibration_yaml(suite: str, library_pkl: str) -> dict:
    """GTP template of ``suite`` with the calibration judge/gate and the library."""
    path = libs.REPO_ROOT / libs.TEMPLATE[suite]
    doc = copy.deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    cp1 = doc["checkpoints"]["cp1"]
    cp1["judge"] = {"type": "always_hit"}
    cp1["gate"] = {"type": "always_search"}
    doc["backend"]["in_memory"]["preload_path"] = library_pkl
    doc["write_policy"] = {"type": "never"}
    return doc


def solve_gate_theta(scores, top_fraction: float = THETA_TOP_FRACTION) -> float:
    """GTP hysteresis theta: the score cut admitting the top ``top_fraction``."""
    return float(derive_thresholds([float(x) for x in scores], top_fraction, 0.0)[0])


# ------------------------------------------------------------------
# Core export (pure: takes a prebuilt retrieval contract)
# ------------------------------------------------------------------


def fit_export(
    *,
    s,
    y7,
    y10,
    targets: list[float],
    alpha: float,
    h_exec: int,
    w,
    active_mask,
    contract: dict,
    identity: dict,
    out_dir: pathlib.Path,
    theta_top_fraction: float = THETA_TOP_FRACTION,
) -> dict:
    """Fit, invert every target, write fit record + artifacts + export record.

    ``identity`` carries the caller's provenance digests (table / weights /
    yaml / library / suite / ref_mode). Returns the export record.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir must be empty: {out_dir}")
    s = np.asarray(s, dtype=np.float64)
    y7 = np.asarray(y7, dtype=np.float64)
    y10 = np.asarray(y10, dtype=np.float64)
    if not (len(s) == len(y7) == len(y10)) or len(s) == 0:
        raise SystemExit("s / y7 / y10 must be equally long and non-empty")
    if int(contract.get("action_dim", -1)) != len(w):
        raise SystemExit(f"contract action_dim={contract.get('action_dim')} != len(w)={len(w)}")

    picked = choose_knots(s, KNOT_LADDER)
    if picked is None:
        raise SystemExit(f"knot ladder {KNOT_LADDER} exhausted on {len(s)} rows (stop-loss)")
    knots, n_seg_req = picked
    fit = fit_pl_quantile(s, y7, y10, knots, n_seg_req=n_seg_req, alpha=alpha, eps_total=EPS_TOTAL)
    ir_lo, ir_hi = attainable_range(fit, s)
    digests = pl_fit_digests(fit)
    gate_theta = solve_gate_theta(s, theta_top_fraction)

    solved: dict[str, dict] = {}
    for target in targets:
        sol = delta_for_ir(fit, s, float(target))
        delta = float(sol["delta"])
        if not (math.isfinite(delta) and delta > 0.0):
            raise SystemExit(f"target IR {target}: delta {delta!r} is not a positive finite number")
        if abs(float(sol["ir_gap"])) > IR_MAX_GAP:
            raise SystemExit(
                f"target IR {target}: nearest attainable IR {sol['predicted_ir']:.3f} is "
                f"{sol['ir_gap']:+.3f} pt away (> {IR_MAX_GAP}); attainable range [{ir_lo:.2f}, {ir_hi:.2f}]"
            )
        theta_full, theta_warm = cuts(fit, delta)
        if theta_full == math.inf and theta_warm == math.inf:
            raise SystemExit(f"target IR {target}: both cuts are +inf (all-MISS artifact)")
        solved[target_name(target)] = {
            "target_ir": float(target),
            "delta": delta,
            "theta_full": float(theta_full),
            "theta_warm": float(theta_warm),
            "predicted_ir": float(predicted_ir(s, theta_full, theta_warm)),
            "ir_gap": float(sol["ir_gap"]),
            "floor_info": floor_info(fit, s, delta),
        }

    fit_rec = dict(identity)
    fit_rec.update({
        "protocol": PROTOCOL,
        "estimator": ESTIMATOR,
        "quantile_alpha": float(alpha),
        "h_exec": int(h_exec),
        "n_rows": int(len(s)),
        "cost_model_digest": cost_model_digest(),
        "unit_cost_ms": unit_cost_table(),
        "pl_fit_digests": digests,
        "attainable_ir_range": [float(ir_lo), float(ir_hi)],
        "ir_curve": [[float(d), float(ir)] for d, ir in ir_curve(fit, s)],
        "s_range": [float(s.min()), float(s.max())],
        "gate_theta": gate_theta,
        "gate_theta_top_fraction": float(theta_top_fraction),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    })
    fit_rec.update(fit_record_fields(fit))
    fit_rec_path = out_dir / FIT_RECORD_NAME
    fit_rec_path.write_text(json.dumps(fit_rec, indent=2, sort_keys=True) + "\n")
    fit_rec_sha = _sha(fit_rec_path)

    artifacts: dict[str, dict] = {}
    for name, sol in solved.items():
        meta = {
            "protocol": PROTOCOL,
            "family": FAMILY,
            "estimator": ESTIMATOR,
            "posthoc_exploratory": True,
            "addressing": "ir",
            "target_ir": sol["target_ir"],
            "predicted_ir": sol["predicted_ir"],
            "ir_gap": sol["ir_gap"],
            "floor_info": sol["floor_info"],
            "n_seg_req": int(fit.n_seg_req),
            "n_seg": int(fit.n_seg),
            "eps_total": float(fit.eps_total),
            "s_range": fit_rec["s_range"],
            "fit_record_sha256": fit_rec_sha,
            "pl_fit_digests": digests,
            "gate_theta": gate_theta,
            **identity,
        }
        artifact = SurfaceArtifact(
            schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION,
            k=1,
            h_exec=int(h_exec),
            w=np.asarray(w, dtype=np.float32),
            active_mask=np.asarray(active_mask, dtype=bool),
            start_t_ws=PINNED_START_T_WS,
            delta=sol["delta"],
            quantile_alpha=float(alpha),
            certification_mode=CERTIFICATION_EMPIRICAL,
            uses_disagreement=False,
            v_bin_edges=np.asarray([-np.inf, np.inf], dtype=np.float64),
            s_min_full=np.asarray([sol["theta_full"]], dtype=np.float64),
            s_min_warm=np.asarray([sol["theta_warm"]], dtype=np.float64),
            conformal_c=0.0,
            n_calibration_episodes=0,
            retrieval_contract=dict(contract),
            meta=meta,
        )
        path = out_dir / f"surface_{FAMILY}_{name}.npz"
        save_surface_artifact(artifact, str(path))
        back = load_surface_artifact(str(path))  # round-trip self-check
        if (float(back.s_min_full[0]) != sol["theta_full"]
                or float(back.s_min_warm[0]) != sol["theta_warm"]
                or float(back.delta) != sol["delta"]):
            raise SystemExit(f"artifact {name}: round-trip changed the cuts")
        artifacts[name] = {**sol, "path": str(path.resolve()), "output_sha256": _sha(path)}

    record = dict(identity)
    record.update({
        "protocol": PROTOCOL,
        "family": FAMILY,
        "estimator": ESTIMATOR,
        "posthoc_exploratory": True,
        "addressing": "ir",
        "quantile_alpha": float(alpha),
        "h_exec": int(h_exec),
        "targets": [float(t) for t in targets],
        "attainable_ir_range": [float(ir_lo), float(ir_hi)],
        "gate_theta": gate_theta,
        "gate_theta_top_fraction": float(theta_top_fraction),
        "cost_model_digest": cost_model_digest(),
        "fit_record_path": str(fit_rec_path.resolve()),
        "fit_record_sha256": fit_rec_sha,
        "pl_fit_digests": digests,
        "retrieval_contract": dict(contract),
        "git_commit": fit_rec["git_commit"],
        "artifacts": artifacts,
    })
    rec_path = out_dir / EXPORT_RECORD_NAME
    rec_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def cmd_calib_yaml(args) -> None:
    from openpi.cache.config import load_cache_config

    doc = build_calibration_yaml(args.suite, args.library_pkl)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    load_cache_config(str(out))  # strict schema self-check
    print(f"calibration yaml -> {out}")


def build_contract(*, cache_yaml: str, library_pkl: str, checkpoint_dir: str, config_name: str,
                   action_dim: int, h_exec: int) -> dict:
    """Yaml-side contract plus the library / policy identity the judge checks."""
    from openpi.cache.config import compute_surface_retrieval_contract, load_cache_config
    from openpi.serving.policy_identity import compute_policy_fingerprint, resolve_checkpoint_root

    config = load_cache_config(cache_yaml)
    if config.backend.in_memory.preload_path != library_pkl:
        raise SystemExit(
            f"cache yaml preload_path={config.backend.in_memory.preload_path} != "
            f"--library-pkl {library_pkl}; the contract must name the library the table was built on"
        )
    contract = compute_surface_retrieval_contract(config)
    with open(library_pkl, "rb") as f:
        lib = pickle.load(f)
    contract.update({
        "library_sha256": _sha(pathlib.Path(library_pkl)),
        "library_entry_count": int(len(lib["entries"])),
        "action_dim": int(action_dim),
        "num_steps": N_STEPS,
        "h_exec": int(h_exec),
        "policy_fingerprint": compute_policy_fingerprint(
            str(resolve_checkpoint_root(checkpoint_dir)), config_name,
        ),
        "top_k": 1,
    })
    return contract


def cmd_fit(args) -> None:
    table_path = pathlib.Path(args.table)
    weights_path = pathlib.Path(args.weights_npz)
    table = load_table(str(table_path), ref_mode=args.ref_mode)
    weights = np.load(weights_path)
    w = np.asarray(weights["w"], dtype=np.float32)
    active_mask = np.asarray(weights["active_mask"], dtype=bool)
    contract = build_contract(
        cache_yaml=args.cache_yaml, library_pkl=args.library_pkl,
        checkpoint_dir=args.checkpoint_dir, config_name=args.config_name,
        action_dim=int(w.shape[0]), h_exec=args.h_exec,
    )
    identity = {
        "suite": args.suite,
        "ref_mode": args.ref_mode,
        "table_path": str(table_path.resolve()),
        "table_sha256": _sha(table_path),
        "weights_sha256": _sha(weights_path),
        "cache_yaml_sha256": _sha(pathlib.Path(args.cache_yaml)),
        "library_pkl": args.library_pkl,
        "library_sha256": contract["library_sha256"],
        "n_episodes": int(len(set(table.episode.tolist()))),
        "n_tasks": int(len(set(table.task.tolist()))),
    }
    record = fit_export(
        s=table.s, y7=table.y7, y10=table.y10, targets=parse_targets(args.target_ir),
        alpha=args.alpha, h_exec=args.h_exec, w=w, active_mask=active_mask,
        contract=contract, identity=identity, out_dir=pathlib.Path(args.out_dir),
        theta_top_fraction=args.theta_top_fraction,
    )
    print(f"attainable IR range {record['attainable_ir_range']}; gate theta {record['gate_theta']:.6f}")
    for name, art in record["artifacts"].items():
        print(f"  {name}: target {art['target_ir']:5.1f} -> predicted {art['predicted_ir']:6.2f} "
              f"(gap {art['ir_gap']:+.3f}) delta {art['delta']:.4f} "
              f"theta_full {art['theta_full']:.6f} theta_warm {art['theta_warm']:.6f}")
    print(f"export record -> {pathlib.Path(args.out_dir) / EXPORT_RECORD_NAME}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calib-yaml")
    p.add_argument("--suite", required=True, choices=tuple(libs.TEMPLATE))
    p.add_argument("--library-pkl", required=True, help="server-side path written into preload_path")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calib_yaml)
    f = sub.add_parser("fit")
    f.add_argument("--suite", required=True, choices=tuple(libs.TEMPLATE))
    f.add_argument("--table", required=True)
    f.add_argument("--weights-npz", required=True)
    f.add_argument("--cache-yaml", required=True)
    f.add_argument("--library-pkl", required=True)
    f.add_argument("--checkpoint-dir", required=True)
    f.add_argument("--config-name", default="pi05_libero")
    f.add_argument("--ref-mode", default="tau1", choices=("tau1", "uncoupled", "fresh"))
    f.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    f.add_argument("--h-exec", type=int, default=DEFAULT_H_EXEC)
    f.add_argument("--target-ir", default=",".join(f"{t:g}" for t in DEFAULT_TARGETS))
    f.add_argument("--theta-top-fraction", type=float, default=THETA_TOP_FRACTION)
    f.add_argument("--out-dir", required=True)
    f.set_defaults(func=cmd_fit)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
