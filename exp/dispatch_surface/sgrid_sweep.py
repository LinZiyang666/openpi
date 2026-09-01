"""Exploratory RIT-density sweep ("sgrid"): fill the RIT (SV / S0) families
with extra calibration quantiles so their frontiers are as dense as the GST
threshold grid. Owner-authorised post-hoc development rollout (2026-08-30); it does NOT
enter the frozen confirmation chain (no roster spec, no seal input).

Re-uses the disciplined components unchanged:
  * ``export_exploratory_surface`` produced the artifacts (delta = quantile of
    D_dev.y10, digest chain to the Rev 1 package);
  * ``emit_precheck_yamls._emit`` writes each arm's yaml from the same template;
  * ``run_precheck.PrecheckSweepStrategy`` / ``_launch_fresh_pool_run`` schedule
    the same A' cells (official init index stamped) on the same fleet, with the
    journal / per_step / launch ledger written the same way;
  * ``precheck_io`` cost-only readers form the ratio-of-sums analytic cost.

Sub-commands:
  emit       export records + Rev 1 package + template -> yamls + arm_matrix_sgrid.json
  run        launch (or --dry-validate) the sweep on A'
  summarize  journal + per_step -> per-arm {cost, sr, t, d} in ``measured_policies`` shape
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import logging
import pathlib

import yaml

PROTOCOL_SGRID = "dispatch_surface_rev2_sgrid_dev"
PROTOCOL_SYSGATE = "dispatch_surface_rev2_sysgate_dev"   # same sweep with the production hysteresis gate re-attached
LAYER_SGRID = "sgrid"
SGRID_FROZEN_KEYS = (
    "protocol", "layer", "suite", "trials_per_task", "replan_steps", "env_seed", "policy_fingerprint",
    "library_sha256", "aprime_content_sha256", "split_manifest_sha256", "arm_matrix_sha256",
    "frozen_yaml_sha256", "artifact_sha256", "rev1_package_manifest_sha256", "export_record_sha256",
    "cost_model_digest", "contract_arm", "posthoc_exploratory",
)
logger = logging.getLogger("dispatch_surface.sgrid")


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def pl_arm_fields(matrix: dict, arm: str) -> dict:
    """RIT-PL addressing fields of an arm as recorded in the matrix (empty for
    a matrix emitted before the RIT-PL keys existed)."""
    return {key: matrix[key].get(arm) for key in ("target_ir", "predicted_ir", "estimator")
            if isinstance(matrix.get(key), dict)}


def _load_pl_fit_record(rec: dict, *, what: str) -> dict:
    """Legs (i) path and (ii) bytes of the PL fit-record validation; the
    semantic leg runs once every artifact of the record has been loaded."""
    path = pathlib.Path(rec["pl_fit_record_path"])
    if not path.is_file():
        raise SystemExit(f"{what}: PL fit record missing at {path}")
    if _sha(path) != rec["pl_fit_record_sha256"]:
        raise SystemExit(f"{what}: PL fit record drifted from the export record SHA")
    return json.loads(path.read_text())



# ----------------------------------------------------------------------
# emit
# ----------------------------------------------------------------------

def emit(args) -> None:
    from exp.dispatch_surface import rev1_package as pkgmod
    from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest, cost_model_payload
    from exp.dispatch_surface.emit_precheck_yamls import (
        LAYER_PRIMARY,
        LAYER_SECONDARY,
        _emit,
        _export_record_arms,
        gate_section,
    )
    from exp.dispatch_surface.phase0_roster import FAMILY_S0, FAMILY_S0_PL, FAMILY_SV, PROTOCOL_PHASE0
    from exp.dispatch_surface.rit_pl import ESTIMATOR, PROTOCOL_RIT_PL
    from exp.dispatch_surface.template_parity import (
        assert_export_record_schema,
        assert_no_placeholders,
        assert_rit_pl_artifact_coherence,
        assert_rit_pl_export_record_schema,
        assert_rit_pl_fit_record,
        assert_template_parity,
    )
    from openpi.cache.components.surface_judge import load_surface_artifact

    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    suite = manifest["suite"]
    rev1_matrix = pkgmod.load_json_member(manifest, pkg, "matrix")
    lib_sha = _sha(args.library_pkl)
    if lib_sha != rev1_matrix.get("library_sha256"):
        raise SystemExit("--library-pkl is not the library the Rev 1 matrix froze")
    theta = float(rev1_matrix["gate_theta"])
    emit_layer = LAYER_SECONDARY if getattr(args, "gate_layer", "primary") == "secondary" else LAYER_PRIMARY
    protocol = PROTOCOL_SYSGATE if emit_layer == LAYER_SECONDARY else PROTOCOL_SGRID
    template = yaml.safe_load(open(args.template))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} must be empty")

    record_paths = [pathlib.Path(p) for p in args.export_records.split(",") if p.strip()]
    records = [json.loads(p.read_text()) for p in record_paths]
    record_sha = [_sha(p) for p in record_paths]
    pl_fit_records: dict[int, dict] = {}
    for idx, rec in enumerate(records):
        what = f"export record {idx}"
        proto = rec.get("protocol")
        if proto == PROTOCOL_PHASE0:
            assert_export_record_schema(rec, what=what, cost_model_digest=cost_model_digest(), protocol=PROTOCOL_PHASE0)
        elif proto == PROTOCOL_RIT_PL:
            assert_rit_pl_export_record_schema(rec, what=what, cost_model_digest=cost_model_digest())
            pl_fit_records[idx] = _load_pl_fit_record(rec, what=what)
        else:
            raise SystemExit(f"{what}: protocol {proto!r} is not accepted by the sweep")
        if rec.get("rev1_package_manifest_sha256") != manifest_sha:
            raise SystemExit("an export record was produced against a different Rev 1 package")
    surface_arms = _export_record_arms(records, protocols=(PROTOCOL_PHASE0, PROTOCOL_RIT_PL),
                                       families=(FAMILY_SV, FAMILY_S0, FAMILY_S0_PL))

    arms, artifact_paths, artifact_sha256, families, quantiles, deltas = {}, {}, {}, {}, {}, {}
    target_ir, predicted_ir, estimator, pl_sha = {}, {}, {}, {}
    pl_cuts: dict[int, dict[str, tuple[float, float]]] = {}
    for arm, spec in surface_arms.items():
        path = pathlib.Path(spec["artifact"])
        if not path.is_file() or _sha(path) != spec["output_sha256"]:
            raise SystemExit(f"{arm}: artifact missing or drifted from its export record ({path})")
        art = load_surface_artifact(str(path))
        if art.meta.get("posthoc_exploratory") is not True:
            raise SystemExit(f"{arm}: artifact is not marked posthoc_exploratory")
        rec = records[spec["export_record_index"]]
        source_role = rec["source_role"]
        source = load_surface_artifact(str(pkgmod.verify_member(manifest, pkg, source_role)))
        if (art.meta.get("source_artifact_sha256") != pkgmod.member_sha(manifest, source_role)
                or rec["source_artifact_sha256"] != pkgmod.member_sha(manifest, source_role)):
            raise SystemExit(f"{arm}: source artifact SHA chain broken")
        fit_role = "fit.sv" if spec["family"] == FAMILY_SV else "fit.s0"
        if rec["source_fit_record_sha256"] != pkgmod.member_sha(manifest, fit_role):
            raise SystemExit(f"{arm}: source fit record SHA chain broken")
        assert_template_parity(art, source, what=arm)
        assert_no_placeholders(art.meta, what=arm)
        if art.uses_disagreement != (spec["family"] == FAMILY_SV):
            raise SystemExit(f"{arm}: family/uses_disagreement mismatch")
        if art.delta != spec["delta"]:
            raise SystemExit(f"{arm}: artifact delta != export record delta")
        extra = spec.get("extra")
        if extra is not None:
            # RIT-PL: the artifact must bind the same PL fit record as its export record.
            if art.meta.get("pl_fit_record_sha256") != rec["pl_fit_record_sha256"]:
                raise SystemExit(f"{arm}: artifact meta does not bind the record's PL fit record")
            name = arm[len(f"dsp_{spec['family']}_"):]
            # Close the record <-> artifact <-> meta triangle before any value enters the matrix.
            assert_rit_pl_artifact_coherence(art, rec["artifacts"][name], rec, pl_fit_records[spec["export_record_index"]], what=arm)
            pl_cuts.setdefault(spec["export_record_index"], {})[name] = (float(art.s_min_full[0]), float(art.s_min_warm[0]))
            target_ir[arm], predicted_ir[arm] = extra["target_ir"], extra["predicted_ir"]
            estimator[arm], pl_sha[arm] = ESTIMATOR, rec["pl_fit_record_sha256"]
        else:
            target_ir[arm] = predicted_ir[arm] = estimator[arm] = pl_sha[arm] = None
        judge = {"type": "dispatch_surface", "surface_artifact_path": str(path)}
        if art.meta.get("judge_variant") == "cumulative_risk":
            judge["export_factor_outputs"] = True   # persist the CRD commit diagnostics per step
        ypath = out_dir / f"{arm}.yaml"
        _emit(template, ypath, judge, args.library_pkl, theta, emit_layer)
        arms[arm] = str(ypath)
        artifact_paths[arm] = str(path.resolve())
        artifact_sha256[arm] = spec["output_sha256"]
        families[arm] = spec["family"]
        quantiles[arm] = spec["quantile"]
        deltas[arm] = spec["delta"]
    for idx, fit_rec in pl_fit_records.items():
        # Semantic leg: schema, digests, identity and per-arm cut recomputation.
        assert_rit_pl_fit_record(fit_rec, records[idx], what=f"export record {idx}", artifact_cuts=pl_cuts.get(idx, {}))
    sv_arms = sorted(a for a in arms if families[a] == FAMILY_SV)
    if not sv_arms:
        raise SystemExit("the sweep needs at least one SV arm to carry the launch contract")
    contract_arm = sv_arms[0]
    matrix = {
        "protocol": protocol,
        "layer": LAYER_SGRID,
        "posthoc_exploratory": True,
        "suite": suite,
        "arms": arms,
        "families": families,
        "quantiles": quantiles,
        "deltas": deltas,
        "target_ir": target_ir,
        "predicted_ir": predicted_ir,
        "estimator": estimator,
        "pl_fit_record_sha256": pl_sha,
        "gate_type": gate_section(emit_layer, theta)["type"],
        "gate_theta": theta,
        "gate_theta_top_fraction": rev1_matrix.get("gate_theta_top_fraction"),
        "gate_params": rev1_matrix.get("gate_params"),
        "arm_yaml_sha256": {arm: _sha(p) for arm, p in arms.items()},
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
        "certification_mode": rev1_matrix.get("certification_mode"),
        "core_arms": [],
        "descriptive_arms": sorted(arms),
        "contract_arm": contract_arm,
        "contract_artifact": artifact_paths[contract_arm],
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
    matrix_path = out_dir / f"arm_matrix_{LAYER_SGRID}.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    print(f"emitted {len(arms)} sgrid arms for {suite} -> {matrix_path}")


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------

def _check_ledger(ledger: dict, entry: dict) -> None:
    if ledger.get("schema_version") != 2 or not isinstance(ledger.get("launches"), list):
        raise SystemExit("launch ledger is not schema_version 2; use a fresh output path")
    for idx, prior in enumerate(ledger["launches"]):
        for key in SGRID_FROZEN_KEYS:
            if prior.get(key) != entry.get(key):
                raise SystemExit(f"launch ledger entry {idx} differs in frozen key {key!r}; refusing to resume")


def run(args) -> None:
    from exp.dispatch_surface.run_precheck import (
        LAYER_EXPLORATORY,
        NUM_TASKS,
        PrecheckSweepStrategy,
        ServerEndpoint,
        _launch_fresh_pool_run,
        arms_with_accepted_work_left,
        assert_launch_contract,
        official_test_inits,
        validate_aprime_pool,
        validate_precheck_arms,
    )

    if args.workers <= 0 or args.gpus <= 0:
        raise SystemExit("--workers and --gpus must be positive")
    matrix = json.loads(pathlib.Path(args.arm_matrix).read_text())
    protocol = matrix.get("protocol")
    if protocol not in (PROTOCOL_SGRID, PROTOCOL_SYSGATE) or matrix.get("layer") != LAYER_SGRID \
            or matrix.get("posthoc_exploratory") is not True:
        raise SystemExit("arm matrix is not an sgrid matrix")
    if matrix.get("suite") != args.task_suite:
        raise SystemExit("arm matrix suite != --task-suite")
    expected_gate = "always_search" if protocol == PROTOCOL_SGRID else "score_hysteresis"
    if matrix.get("gate_type") != expected_gate:
        raise SystemExit(f"{protocol} arms must carry gate {expected_gate!r}, got {matrix.get('gate_type')!r}")
    arm_paths: dict[str, str] = dict(matrix["arms"])
    if {a: _sha(p) for a, p in arm_paths.items()} != matrix["arm_yaml_sha256"]:
        raise SystemExit("arm matrix YAML digests do not match the files about to run")
    for arm, path in matrix["artifact_paths"].items():
        if _sha(path) != matrix["artifact_sha256"][arm]:
            raise SystemExit(f"{arm}: surface artifact drifted since emit")
    if args.arms:
        wanted = set(args.arms.split(","))
        missing = wanted - set(arm_paths)
        if missing:
            raise SystemExit(f"unknown arms requested: {sorted(missing)}")
        arm_paths = {a: p for a, p in arm_paths.items() if a in wanted}
    if not args.no_resume_filter:
        remaining, _counts = arms_with_accepted_work_left(args.journal, list(arm_paths), expected=NUM_TASKS * args.trials)
        arm_paths = {a: p for a, p in arm_paths.items() if a in set(remaining)}
        if not arm_paths:
            logger.info("every arm is complete; nothing to run")
            return
    from exp.dispatch_surface.run_precheck import LAYER_SECONDARY as _LSEC
    anchor = matrix.get("contract_anchor_arm")
    yaml_paths = validate_precheck_arms(arm_paths, LAYER_EXPLORATORY if protocol == PROTOCOL_SGRID else _LSEC,
                                        anchor_arms=frozenset([anchor]) if anchor else frozenset())

    servers = []
    for spec in args.servers.split(","):
        if ":" not in spec:
            raise SystemExit(f"--servers entry {spec!r} must be host:port")
        host, port = spec.rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))
    contract_binding = assert_launch_contract(matrix["contract_artifact"], args.replan_steps, servers)

    per_step_path = pathlib.Path(args.per_step_out)
    per_step_path.parent.mkdir(parents=True, exist_ok=True)
    pool_dir = args.pool_dir or str(pathlib.Path(args.split_manifest).resolve().parent / "test_aprime")
    pool = validate_aprime_pool(args.split_manifest, pool_dir, args.trials)
    if pool["suite"] != args.task_suite:
        raise SystemExit("A' split suite != --task-suite")
    officials = official_test_inits(args.split_manifest, args.trials)

    launch_entry = {
        "protocol": protocol, "layer": LAYER_SGRID, "suite": args.task_suite,
        "executed_arms": sorted(yaml_paths), "core_arms": [], "descriptive_arms": sorted(matrix["descriptive_arms"]),
        "trials_per_task": args.trials, "replan_steps": args.replan_steps, "env_seed": args.seed,
        "policy_fingerprint": contract_binding["policy_fingerprint"], "contract_binding": contract_binding,
        "library_sha256": matrix["library_sha256"], "aprime_content_sha256": pool["rollup_sha256"],
        "split_manifest": args.split_manifest, "split_manifest_sha256": pool["split_manifest_sha256"],
        "arm_matrix_sha256": _sha(args.arm_matrix), "frozen_yaml_sha256": dict(matrix["arm_yaml_sha256"]),
        "artifact_sha256": dict(matrix["artifact_sha256"]), "fit_record_sha256": None,
        "executed_yaml_sha256": {arm: _sha(p) for arm, p in yaml_paths.items()}, "pool": pool,
        "posthoc_exploratory": True, "rev1_package_manifest_sha256": matrix["rev1_package_manifest_sha256"],
        "export_record_sha256": list(matrix["export_record_sha256"]), "cost_model_digest": matrix["cost_model_digest"],
        "contract_arm": matrix["contract_arm"],
    }
    launch_path = pathlib.Path(str(per_step_path) + ".launch.json")
    ledger = {"schema_version": 2, "launches": []}
    if launch_path.is_file():
        ledger = json.loads(launch_path.read_text())
    _check_ledger(ledger, launch_entry)
    logger.info("sgrid launch bound to A' pool rollup %s", pool["rollup_sha256"])
    _launch_fresh_pool_run(args, lambda: PrecheckSweepStrategy(args.task_suite, yaml_paths, args.trials, officials),
                           yaml_paths, servers, launch_entry, ledger, launch_path, per_step_path, pool["apool_dir"],
                           layer=LAYER_SGRID, frozen_keys=SGRID_FROZEN_KEYS,
                           summary={"contract_binding": contract_binding, "n_arms": len(yaml_paths)})


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------

_CRD_NAME = re.compile(r"crd_q(?P<q>\d+)_g(?P<g>[0-9p]+)_m(?P<m>\d+|inf)_j(?P<j>\d+|inf)_L(?P<L>\d+|none)$")


def crd_params(arm: str) -> dict | None:
    """H-CRD knobs encoded in an artifact / arm name (``crd_q85_g1_m2_j3_L6``);
    ``None`` for a plain surface arm. ``m``/``j`` are ``inf`` and ``L`` is
    ``None`` for the degenerate (ablation) settings."""
    m = _CRD_NAME.search(arm)
    if not m:
        return None
    return {
        "budget_mult": float("inf") if m.group("m") == "inf" else int(m.group("m")),
        "j_bad": float("inf") if m.group("j") == "inf" else int(m.group("j")),
        "l_max": None if m.group("L") == "none" else int(m.group("L")),
        "gamma": float(m.group("g").replace("p", ".")),
    }


def summarize(args) -> None:
    from exp.dispatch_surface.analysis.precheck_io import load_accepted_cells_costonly, load_cost_cells_costonly
    from exp.dispatch_surface.run_precheck import NUM_TASKS, official_test_inits

    from exp.dispatch_surface.phase0_roster import PROTOCOL_TGRID

    matrix = json.loads(pathlib.Path(args.arm_matrix).read_text())
    if matrix.get("protocol") not in (PROTOCOL_SGRID, PROTOCOL_SYSGATE, PROTOCOL_TGRID):
        raise SystemExit("arm matrix is not an sgrid/tgrid matrix")
    is_tgrid = matrix.get("protocol") == PROTOCOL_TGRID
    arms = sorted(matrix["arms"])
    subset = bool(getattr(args, "arms", ""))
    if subset:
        wanted = set(args.arms.split(","))
        missing = wanted - set(arms)
        if missing:
            raise SystemExit(f"unknown arms requested: {sorted(missing)}")
        arms = sorted(wanted)                                  # completeness is still enforced per summarized arm
    officials = official_test_inits(args.split_manifest, args.trials)
    grid = {(t, i) for t in range(NUM_TASKS) for i in range(args.trials)}
    accepted = load_accepted_cells_costonly(args.journal, arms, grid)
    successes = {a: 0 for a in arms}
    for line in open(args.journal):
        row = json.loads(line)
        if row.get("yaml_id") in successes and row.get("accepted") is True and row.get("success") is True:
            successes[row["yaml_id"]] += 1
    cells, cost_summary = load_cost_cells_costonly(args.per_step, arms, accepted, officials)
    out_arms = {}
    for arm in arms:
        n_ep = len(accepted[arm])
        if n_ep != len(grid):
            raise SystemExit(f"{arm}: {n_ep} accepted episodes, expected {len(grid)} (sweep incomplete)")
        total_cost = sum(c for c, _n in cells[arm].values())
        total_dec = sum(n for _c, n in cells[arm].values())
        out_arms[arm] = {
            "family": matrix["families"][arm],
            "quantile": None if is_tgrid else matrix["quantiles"][arm],
            "delta": None if is_tgrid else matrix["deltas"][arm],
            "cost": total_cost / total_dec, "sr": successes[arm] / n_ep,
            "t": total_cost / n_ep, "d": total_dec / n_ep, "episodes": n_ep, "successes": successes[arm],
        }
        if is_tgrid:
            out_arms[arm]["fh"] = matrix["nominal"][arm]["fh"]
            out_arms[arm]["ws"] = matrix["nominal"][arm]["ws"]
            out_arms[arm]["threshold_pair"] = matrix["threshold_pairs"][arm]
        out_arms[arm].update(pl_arm_fields(matrix, arm))

        knobs = crd_params(arm)
        if knobs is not None:
            out_arms[arm]["crd"] = {k: (None if v is None else (v if v != float("inf") else "inf")) for k, v in knobs.items()}
    ledger = json.loads(pathlib.Path(args.launch_manifest).read_text())
    out = {
        "protocol": matrix["protocol"], "layer": matrix.get("layer", LAYER_SGRID), "suite": matrix["suite"], "posthoc_exploratory": True,
        "trials_per_task": args.trials, "arms": out_arms,
        "arms_subset": subset, "matrix_arm_count": len(matrix["arms"]),
        "run_ids": [e.get("run_id") for e in ledger.get("launches", [])],
        "cost_summary": cost_summary,
        "input_sha256": {"arm_matrix": _sha(args.arm_matrix), "journal": _sha(args.journal),
                         "per_step": _sha(args.per_step), "launch_manifest": _sha(args.launch_manifest),
                         "split_manifest": _sha(args.split_manifest)},
        "cost_model_digest": matrix["cost_model_digest"],
        "rev1_package_manifest_sha256": matrix["rev1_package_manifest_sha256"],
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    for arm in arms:
        a = out_arms[arm]
        tag = f"fh{a['fh']}/ws{a['ws']}" if is_tgrid else f"q={a['quantile']}"
        print(f"{arm:16s} {tag:<10} cost {a['cost']:6.2f} ms  sr {a['sr']:.3f}")
    print(f"summary -> {args.out} sha256={_sha(args.out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--rev1-package-manifest", required=True)
    e.add_argument("--export-records", required=True, help="comma list of export_record.json (sv and/or s0)")
    e.add_argument("--template", required=True)
    e.add_argument("--library-pkl", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--gate-layer", choices=["primary", "secondary"], default="primary",
                   help="primary = always_search (default); secondary = re-attach the production score_hysteresis gate (theta from the Rev 1 matrix)")
    r = sub.add_parser("run")
    r.add_argument("--arm-matrix", required=True)
    r.add_argument("--task-suite", required=True)
    r.add_argument("--servers", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--server-workers", default="")
    r.add_argument("--arms", default="")
    r.add_argument("--no-resume-filter", action="store_true")
    r.add_argument("--trials", type=int, required=True)
    r.add_argument("--replan-steps", type=int, required=True)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--journal", required=True)
    r.add_argument("--per-step-out", required=True)
    r.add_argument("--split-manifest", required=True)
    r.add_argument("--pool-dir", default="")
    r.add_argument("--bind-host", default="127.0.0.1")
    r.add_argument("--episode-timeout-s", type=float, default=1800.0)
    r.add_argument("--eval-concurrency", type=int, default=0)
    r.add_argument("--gpus", type=int, default=1)
    r.add_argument("--conda-env", default="")
    r.add_argument("--dry-validate", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--arm-matrix", required=True)
    s.add_argument("--journal", required=True)
    s.add_argument("--per-step", required=True)
    s.add_argument("--launch-manifest", required=True)
    s.add_argument("--split-manifest", required=True)
    s.add_argument("--trials", type=int, default=30)
    s.add_argument("--arms", default="", help="comma list; summarize only this subset of the matrix (recorded as arms_subset)")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    {"emit": emit, "run": run, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
