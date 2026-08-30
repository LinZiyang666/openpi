"""Discipline validator for Phase 0 exploratory runs (plan section 3.5, G2R1-B1/B4).

Same strength as the Rev 1 ``check_discipline``: one frozen exploratory
matrix, its Rev 1 package, the split manifest and a v2 launch ledger must
agree on every experiment-wide field; every ledger entry is checked for its
executed-arm/YAML map, contract binding, A' pool attestation and uniqueness
of run ids; YAML bytes are recomputed; the anchor arm's judge shape is
re-validated. The returned ``executed_arms_by_run`` is what every consumer
uses to refuse accepted rows no launch actually executed. Nothing here reads
outcomes.
"""

from __future__ import annotations

import json
import pathlib

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis.analytic_cost import (
    assert_unit_costs_match,
    cost_model_digest,
)
from exp.dispatch_surface.phase0_roster import ANCHOR_ARM, LAYER_EXPLORATORY, PROTOCOL_PHASE0
from exp.dispatch_surface.run_precheck import (
    EXPLORATORY_FROZEN_LAUNCH_KEYS,
    FORMAL_H_EXEC,
    FROZEN_LAUNCH_KEYS,
    NUM_TASKS,
    _file_sha256,
    validate_exploratory_matrix_artifacts,
    validate_precheck_arms,
)


def executed_arms_by_run(launches: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for launch in launches:
        out[str(launch["run_id"])] = set(launch.get("executed_arms") or [])
    return out


def assert_rows_claimed(accepted: dict, executed: dict[str, set[str]], *, what: str) -> None:
    """Refuse any accepted (run_id, arm) that no ledger launch executed (G2R1-B1)."""
    for arm, cells in accepted.items():
        for key, rec in cells.items():
            run_id = str(rec.get("run_id"))
            if run_id not in executed or arm not in executed[run_id]:
                raise SystemExit(
                    f"{what}: arm {arm} cell {key} was accepted under run {run_id!r}, "
                    "which no launch in the ledger executed for that arm"
                )


def validate(matrix_path: str, launch_manifest_path: str, split_manifest_path: str,
             *, trials: int) -> dict:
    """Return the frozen discipline context or raise SystemExit."""
    from openpi.cache.components.surface_judge import load_surface_artifact

    matrix_path = pathlib.Path(matrix_path)
    matrix = json.loads(matrix_path.read_text())
    if matrix.get("layer") != LAYER_EXPLORATORY or matrix.get("protocol") != PROTOCOL_PHASE0:
        raise SystemExit("not a Phase 0 exploratory arm matrix")
    ctx = validate_exploratory_matrix_artifacts(matrix)
    matrix_arms = dict(matrix["arms"])
    # YAML bytes recomputed here, not trusted from the matrix.
    recorded = matrix.get("arm_yaml_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(matrix_arms):
        raise SystemExit("exploratory matrix does not freeze exactly one YAML digest per arm")
    arm_yaml_sha = {}
    for arm, path in matrix_arms.items():
        yp = pathlib.Path(path)
        if not yp.is_file():
            raise SystemExit(f"arm {arm}: yaml missing on disk: {yp}")
        actual = _file_sha256(yp)
        if actual != recorded[arm]:
            raise SystemExit(f"arm {arm}: yaml content drifted since emit")
        arm_yaml_sha[arm] = actual
    # Anchor judge shape and gate, re-validated from the yaml itself.
    validate_precheck_arms(matrix_arms, LAYER_EXPLORATORY, frozenset(matrix.get("judge_role") or {}))
    contract_art = load_surface_artifact(matrix["artifact_paths"][matrix["contract_anchor_arm"]])
    fp = contract_art.retrieval_contract.get("policy_fingerprint")
    lib_sha = contract_art.retrieval_contract.get("library_sha256")
    if matrix.get("library_sha256") != lib_sha:
        raise SystemExit("exploratory matrix library_sha256 != contract artifact")

    ledger = json.loads(pathlib.Path(launch_manifest_path).read_text())
    launches = ledger.get("launches") or []
    if ledger.get("schema_version") != 2 or not isinstance(launches, list) or not launches:
        raise SystemExit("launch ledger is not a v2 ledger with launches")
    matrix_sha = _file_sha256(matrix_path)
    split_path = pathlib.Path(split_manifest_path)
    split_sha = _file_sha256(split_path)
    split = json.loads(split_path.read_text())
    first = launches[0]
    frozen = FROZEN_LAUNCH_KEYS + EXPLORATORY_FROZEN_LAUNCH_KEYS
    for key in frozen:
        if key not in first:
            raise SystemExit(f"launch ledger entry lacks {key}")
        for idx, other in enumerate(launches[1:], start=1):
            if other.get(key) != first.get(key):
                raise SystemExit(f"launch ledger entry {idx} drifts on frozen key {key}")
    run_ids = [e.get("run_id") for e in launches]
    if any(not isinstance(r, str) or not r for r in run_ids) or len(set(run_ids)) != len(run_ids):
        raise SystemExit("launch ledger has missing or duplicated run ids")
    if first.get("protocol") != PROTOCOL_PHASE0 or first.get("layer") != LAYER_EXPLORATORY:
        raise SystemExit("launch ledger protocol/layer != exploratory")
    if first.get("posthoc_exploratory") is not True:
        raise SystemExit("launch ledger is not marked posthoc_exploratory")
    if first.get("suite") != matrix.get("suite") or split.get("suite") != matrix.get("suite"):
        raise SystemExit("launch/split/matrix suites disagree")
    if first.get("trials_per_task") != trials:
        raise SystemExit(f"launch trials_per_task {first.get('trials_per_task')} != {trials}")
    if not isinstance(first.get("env_seed"), int) or isinstance(first.get("env_seed"), bool):
        raise SystemExit("launch env_seed must be an integer")
    if first.get("replan_steps") != contract_art.h_exec or contract_art.h_exec != FORMAL_H_EXEC:
        raise SystemExit("launch replan_steps != contract artifact h_exec")
    if first.get("arm_matrix_sha256") != matrix_sha:
        raise SystemExit("launch ledger binds a different arm matrix")
    if first.get("split_manifest_sha256") != split_sha:
        raise SystemExit("launch ledger split manifest != the one supplied")
    for key in ("roster_spec_sha256", "rev1_package_manifest_sha256", "cost_model_digest",
                "contract_anchor_arm", "library_sha256", "artifact_sha256"):
        if first.get(key) != matrix.get(key):
            raise SystemExit(f"launch {key} != matrix")
    if list(first.get("export_record_sha256") or []) != list(matrix.get("export_record_sha256") or []):
        raise SystemExit("launch export records != matrix export records")
    if first.get("cost_model_digest") != cost_model_digest():
        raise SystemExit("cost model digest drift between ledger and the cost authority")
    assert_unit_costs_match((matrix.get("cost_model") or {}).get("unit_cost_ms"), what="exploratory matrix")
    if first.get("frozen_yaml_sha256") != arm_yaml_sha:
        raise SystemExit("launch frozen YAML digests != recomputed matrix YAML digests")
    if first.get("policy_fingerprint") != fp:
        raise SystemExit("launch policy fingerprint != contract artifact")
    if first.get("aprime_content_sha256") is None:
        raise SystemExit("launch lacks an A' content digest")
    split_state_sha = {name: info.get("sha256")
                       for name, info in ((split.get("pool_digests") or {}).get("test_aprime") or {}).items()}
    for idx, launch in enumerate(launches):
        if launch.get("core_arms") != []:
            raise SystemExit(f"launch {idx} core arms must be empty for the exploratory layer")
        if sorted(launch.get("descriptive_arms") or []) != sorted(matrix_arms):
            raise SystemExit(f"launch {idx} descriptive roster != matrix roster")
        binding = launch.get("contract_binding") or {}
        if binding.get("policy_fingerprint") != fp or binding.get("h_exec") != launch.get("replan_steps"):
            raise SystemExit(f"launch {idx} contract binding != contract artifact")
        pool = launch.get("pool") or {}
        if (pool.get("suite") != launch.get("suite")
                or pool.get("total_inits") != NUM_TASKS * trials
                or pool.get("rollup_sha256") != launch.get("aprime_content_sha256")
                or pool.get("split_manifest_sha256") != launch.get("split_manifest_sha256")
                or pool.get("state_content_sha256") != split_state_sha):
            raise SystemExit(f"launch {idx} A' pool attestation is inconsistent")
        executed = launch.get("executed_arms")
        executed_sha = launch.get("executed_yaml_sha256")
        if not isinstance(executed, list) or not executed:
            raise SystemExit(f"launch {idx} has no executed arms")
        if set(executed) != set(executed_sha or {}):
            raise SystemExit(f"launch {idx} executed arm/YAML keys disagree")
        if not set(executed).issubset(matrix_arms):
            raise SystemExit(f"launch {idx} executed unknown arms")
        for arm in executed:
            if executed_sha[arm] != arm_yaml_sha[arm]:
                raise SystemExit(f"launch {idx} executed a different YAML for {arm}")
    by_run = executed_arms_by_run(launches)
    # Rev 1 package cross-checks: same suite, policy, library, A', split.
    verdict = pkgmod.load_json_member(ctx["manifest"], ctx["package_dir"], "verdict")
    disc = verdict.get("discipline") or {}
    if disc.get("policy_fingerprint") != fp or disc.get("library_sha256") != lib_sha:
        raise SystemExit("policy/library differ from the Rev 1 verdict")
    if disc.get("aprime_content_sha256") != first.get("aprime_content_sha256"):
        raise SystemExit("A' pool content differs from the Rev 1 verdict")
    if disc.get("split_manifest_sha256") != split_sha:
        raise SystemExit("split manifest differs from the Rev 1 verdict")
    assert_unit_costs_match((disc.get("cost_inputs") or {}).get("unit_cost_ms"), what="Rev 1 verdict")
    executed_all = set().union(*by_run.values())
    return {
        "suite": matrix["suite"],
        "trials": trials,
        "arm_matrix_sha256": matrix_sha,
        "split_manifest_sha256": split_sha,
        "launch_manifest_sha256": _file_sha256(pathlib.Path(launch_manifest_path)),
        "roster_spec_sha256": matrix["roster_spec_sha256"],
        "rev1_package_manifest_sha256": matrix["rev1_package_manifest_sha256"],
        "export_record_sha256": list(matrix["export_record_sha256"]),
        "cost_model_digest": cost_model_digest(),
        "policy_fingerprint": fp,
        "library_sha256": lib_sha,
        "aprime_content_sha256": first.get("aprime_content_sha256"),
        "launch_run_ids": run_ids,
        "executed_arms_by_run": {k: sorted(v) for k, v in by_run.items()},
        "executed_arms": sorted(executed_all),
        "roster_complete": executed_all == set(matrix_arms),
        "arms": sorted(matrix_arms),
        "families": dict(matrix["families"]),
        "quantiles": dict(matrix.get("quantiles") or {}),
        "deltas": dict(matrix.get("deltas") or {}),
        "artifact_paths": dict(matrix["artifact_paths"]),
        "anchor_arm": ANCHOR_ARM,
        "contract_anchor_arm": matrix["contract_anchor_arm"],
        "posthoc_exploratory": True,
        "_manifest": ctx["manifest"],
        "_package_dir": ctx["package_dir"],
    }


# ----------------------------------------------------------------------
# Rev 2 confirmation plan: threshold-grid development layer (plan 3.2-g)
# ----------------------------------------------------------------------

def validate_tgrid(matrix_path: str, launch_manifest_path: str, split_manifest_path: str,
                   *, trials: int, yaml_paths: dict[str, str] | None = None,
                   rev1_manifest_path: str | None = None) -> dict:
    """Same strength as ``validate`` for the dense threshold-grid layer: frozen
    grid roster, Rev 1 package binding, recomputed YAML bytes, threshold-pair
    digests, contract binding via the package's SV artifact, A' attestation,
    unique run ids and frozen ledger keys. Reads no outcome.

    ``yaml_paths`` (arm -> file) lets a finalised package supply its own YAML
    members instead of the execution-time paths the matrix records, and
    ``rev1_manifest_path`` the local copy of the Rev 1 package; both are still
    bound by content to the digests frozen in the matrix (G2R1-B8)."""
    from openpi.cache.components.surface_judge import load_surface_artifact

    from exp.dispatch_surface.phase0_roster import LAYER_TGRID, PROTOCOL_TGRID
    from exp.dispatch_surface.run_precheck import (
        TGRID_FROZEN_LAUNCH_KEYS,
        validate_tgrid_arms,
        validate_tgrid_matrix_artifacts,
    )

    matrix_path = pathlib.Path(matrix_path)
    matrix = json.loads(matrix_path.read_text())
    if matrix.get("layer") != LAYER_TGRID or matrix.get("protocol") != PROTOCOL_TGRID:
        raise SystemExit("not a threshold-grid development arm matrix")
    ctx = validate_tgrid_matrix_artifacts(matrix, rev1_manifest_path=rev1_manifest_path)
    matrix_arms = dict(matrix["arms"])
    recorded = matrix.get("arm_yaml_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(matrix_arms):
        raise SystemExit("threshold-grid matrix does not freeze exactly one YAML digest per arm")
    if yaml_paths is not None:
        if set(yaml_paths) != set(matrix_arms):
            raise SystemExit("resolved yaml roles != matrix arms")
        resolved = {arm: str(yaml_paths[arm]) for arm in matrix_arms}
    else:
        resolved = matrix_arms
    arm_yaml_sha = {}
    for arm, path in resolved.items():
        yp = pathlib.Path(path)
        if not yp.is_file():
            raise SystemExit(f"arm {arm}: yaml missing on disk: {yp}")
        actual = _file_sha256(yp)
        if actual != recorded[arm]:
            raise SystemExit(f"arm {arm}: yaml content drifted since emit")
        arm_yaml_sha[arm] = actual
    validate_tgrid_arms(resolved, matrix)
    contract_art = load_surface_artifact(ctx["contract_artifact"])
    fp = contract_art.retrieval_contract.get("policy_fingerprint")
    lib_sha = contract_art.retrieval_contract.get("library_sha256")
    if matrix.get("library_sha256") != lib_sha:
        raise SystemExit("threshold-grid matrix library_sha256 != contract artifact")

    ledger = json.loads(pathlib.Path(launch_manifest_path).read_text())
    launches = ledger.get("launches") or []
    if ledger.get("schema_version") != 2 or not isinstance(launches, list) or not launches:
        raise SystemExit("launch ledger is not a v2 ledger with launches")
    matrix_sha = _file_sha256(matrix_path)
    split_path = pathlib.Path(split_manifest_path)
    split_sha = _file_sha256(split_path)
    split = json.loads(split_path.read_text())
    first = launches[0]
    for key in FROZEN_LAUNCH_KEYS + TGRID_FROZEN_LAUNCH_KEYS:
        if key not in first:
            raise SystemExit(f"launch ledger entry lacks {key}")
        for idx, other in enumerate(launches[1:], start=1):
            if other.get(key) != first.get(key):
                raise SystemExit(f"launch ledger entry {idx} drifts on frozen key {key}")
    run_ids = [e.get("run_id") for e in launches]
    if any(not isinstance(r, str) or not r for r in run_ids) or len(set(run_ids)) != len(run_ids):
        raise SystemExit("launch ledger has missing or duplicated run ids")
    if first.get("protocol") != PROTOCOL_TGRID or first.get("layer") != LAYER_TGRID:
        raise SystemExit("launch ledger protocol/layer != threshold grid")
    if first.get("posthoc_exploratory") is not True:
        raise SystemExit("launch ledger is not marked posthoc_exploratory")
    if first.get("suite") != matrix.get("suite") or split.get("suite") != matrix.get("suite"):
        raise SystemExit("launch/split/matrix suites disagree")
    if first.get("trials_per_task") != trials:
        raise SystemExit(f"launch trials_per_task {first.get('trials_per_task')} != {trials}")
    if not isinstance(first.get("env_seed"), int) or isinstance(first.get("env_seed"), bool):
        raise SystemExit("launch env_seed must be an integer")
    if first.get("replan_steps") != contract_art.h_exec or contract_art.h_exec != FORMAL_H_EXEC:
        raise SystemExit("launch replan_steps != contract artifact h_exec")
    if first.get("arm_matrix_sha256") != matrix_sha:
        raise SystemExit("launch ledger binds a different arm matrix")
    if first.get("split_manifest_sha256") != split_sha:
        raise SystemExit("launch ledger split manifest != the one supplied")
    for key in ("tgrid_roster_spec_sha256", "threshold_pair_rollup_sha256", "contract_source",
                "estimator_version", "rev1_package_manifest_sha256", "cost_model_digest", "library_sha256"):
        if first.get(key) != matrix.get(key):
            raise SystemExit(f"launch {key} != matrix")
    if first.get("cost_model_digest") != cost_model_digest():
        raise SystemExit("cost model digest drift between ledger and the cost authority")
    if first.get("frozen_yaml_sha256") != arm_yaml_sha:
        raise SystemExit("launch frozen YAML digests != recomputed matrix YAML digests")
    if first.get("policy_fingerprint") != fp:
        raise SystemExit("launch policy fingerprint != contract artifact")
    if first.get("aprime_content_sha256") is None:
        raise SystemExit("launch lacks an A' content digest")
    split_state_sha = {name: info.get("sha256")
                       for name, info in ((split.get("pool_digests") or {}).get("test_aprime") or {}).items()}
    for idx, launch in enumerate(launches):
        if launch.get("core_arms") != []:
            raise SystemExit(f"launch {idx} core arms must be empty for the threshold-grid layer")
        if sorted(launch.get("descriptive_arms") or []) != sorted(matrix_arms):
            raise SystemExit(f"launch {idx} descriptive roster != matrix roster")
        binding = launch.get("contract_binding") or {}
        if binding.get("policy_fingerprint") != fp or binding.get("h_exec") != launch.get("replan_steps"):
            raise SystemExit(f"launch {idx} contract binding != contract artifact")
        pool = launch.get("pool") or {}
        if (pool.get("suite") != launch.get("suite")
                or pool.get("total_inits") != NUM_TASKS * trials
                or pool.get("rollup_sha256") != launch.get("aprime_content_sha256")
                or pool.get("split_manifest_sha256") != launch.get("split_manifest_sha256")
                or pool.get("state_content_sha256") != split_state_sha):
            raise SystemExit(f"launch {idx} A' pool attestation is inconsistent")
        executed = launch.get("executed_arms")
        executed_sha = launch.get("executed_yaml_sha256")
        if not isinstance(executed, list) or not executed:
            raise SystemExit(f"launch {idx} has no executed arms")
        if set(executed) != set(executed_sha or {}):
            raise SystemExit(f"launch {idx} executed arm/YAML keys disagree")
        if not set(executed).issubset(matrix_arms):
            raise SystemExit(f"launch {idx} executed unknown arms")
        for arm in executed:
            if executed_sha[arm] != arm_yaml_sha[arm]:
                raise SystemExit(f"launch {idx} executed a different YAML for {arm}")
    by_run = executed_arms_by_run(launches)
    verdict = pkgmod.load_json_member(ctx["manifest"], ctx["package_dir"], "verdict")
    disc = verdict.get("discipline") or {}
    if disc.get("policy_fingerprint") != fp or disc.get("library_sha256") != lib_sha:
        raise SystemExit("policy/library differ from the Rev 1 verdict")
    if disc.get("aprime_content_sha256") != first.get("aprime_content_sha256"):
        raise SystemExit("A' pool content differs from the Rev 1 verdict")
    if disc.get("split_manifest_sha256") != split_sha:
        raise SystemExit("split manifest differs from the Rev 1 verdict")
    assert_unit_costs_match((disc.get("cost_inputs") or {}).get("unit_cost_ms"), what="Rev 1 verdict")
    executed_all = set().union(*by_run.values())
    return {
        "suite": matrix["suite"],
        "trials": trials,
        "arm_matrix_sha256": matrix_sha,
        "split_manifest_sha256": split_sha,
        "launch_manifest_sha256": _file_sha256(pathlib.Path(launch_manifest_path)),
        "tgrid_roster_spec_sha256": matrix["tgrid_roster_spec_sha256"],
        "threshold_pair_rollup_sha256": matrix["threshold_pair_rollup_sha256"],
        "rev1_package_manifest_sha256": matrix["rev1_package_manifest_sha256"],
        "estimator_version": matrix["estimator_version"],
        "cost_model_digest": cost_model_digest(),
        "policy_fingerprint": fp,
        "library_sha256": lib_sha,
        "aprime_content_sha256": first.get("aprime_content_sha256"),
        "launch_run_ids": run_ids,
        "executed_arms_by_run": {k: sorted(v) for k, v in by_run.items()},
        "executed_arms": sorted(executed_all),
        "roster_complete": executed_all == set(matrix_arms),
        "arms": sorted(matrix_arms),
        "families": dict(matrix["families"]),
        "nominal": dict(matrix["nominal"]),
        "threshold_pairs": dict(matrix["threshold_pairs"]),
        "posthoc_exploratory": True,
        "_manifest": ctx["manifest"],
        "_package_dir": ctx["package_dir"],
    }
