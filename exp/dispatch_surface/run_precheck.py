"""Closed-loop precheck runner (SR axis): dynamic 5-7 arm paired sweep.

Structure mirrors ``exp/gate_threshold_pareto/run_gtp.py`` (whose snapshot,
resume-filter and A-pool attestation machinery is imported verbatim), with two
deliberate differences:

  * Arm validation accepts exactly the two precheck verdict families —
    ``threshold`` WITH a single start_t=0.3 warm tier, and
    ``dispatch_surface`` — instead of the gate line's warm-free contract.
  * Launch-contract fail-fast (plan section 4.6): before any episode runs,
    the primary surface artifact's retrieval contract is loaded, the explicit
    ``--replan-steps`` argument must equal its ``h_exec``, and the policy
    fingerprint REPORTED BY THE SERVER metadata must equal the contract's
    ``policy_fingerprint``. Both are recorded in the launch manifest.

Pairing discipline: all arms run on the same A' init pool (materialised by
``split_init_pools``), same env seed, same conductor seed; the manifest binds
their digests.

Usage mirrors run_gtp; pool record/dir point at the A' pool and
``--arm-matrix`` at ``emit_precheck_yamls``'s arm_matrix.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import threading
import time

from exp.ablation_study.cache_size.run_size_eval import _snapshot_loop, _write_snapshot, merge_snapshot
from exp.ablation_study.cache_size.verify_apool import (
    digest_init_file,
    load_init_states,
    rollup_digest,
)
from openpi.cache.config import load_cache_config
from openpi.conductor import (
    ConductorDriver,
    ExperimentStrategy,
    ServerEndpoint,
    WorkerAgent,
    WorkerSpec,
)

logger = logging.getLogger("dispatch_surface.precheck")

NUM_TASKS = 10
FORMAL_TRIALS = 30
FORMAL_H_EXEC = 5
WS_START_T = 0.3
# Rev 1 runs two pre-registered layers off two separate matrices. Primary
# isolates the verdict rule (every step probed); secondary re-attaches the
# production gate and is descriptive. Their rosters and gates differ, so the
# runner validates the matrix against the layer it declares instead of assuming
# one shape -- a secondary matrix used to be rejected outright here.
LAYER_PRIMARY = "primary"
LAYER_SECONDARY = "secondary"
SECONDARY_CORE_ARMS = {
    "dsp_t_fh30_ws20",
    "dsp_t_fh50_ws20",
    "dsp_t_fh70_ws10",
    "dsp_sv",
}
LAYER_EXPECTED_GATE = {LAYER_PRIMARY: "always_search", LAYER_SECONDARY: "score_hysteresis"}

FORMAL_CORE_ARMS = {
    "dsp_t_fh30_ws20",
    "dsp_t_fh50_ws20",
    "dsp_t_fh70_ws10",
    "dsp_s0",
    "dsp_sv",
}
FROZEN_LAUNCH_KEYS = (
    "protocol",
    "layer",
    "suite",
    "core_arms",
    "descriptive_arms",
    "trials_per_task",
    "replan_steps",
    "env_seed",
    "policy_fingerprint",
    "library_sha256",
    "aprime_content_sha256",
    "split_manifest_sha256",
    "arm_matrix_sha256",
    "frozen_yaml_sha256",
    "artifact_sha256",
    "fit_record_sha256",
)
PROTOCOL = "dispatch_surface_rev1"


def _file_sha256(path: pathlib.Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def arms_with_accepted_work_left(
    journal_path: str | pathlib.Path, arms: list[str], *, expected: int,
) -> tuple[list[str], dict[str, int]]:
    """Resume-filter by accepted eval cells, never by fenced terminal rows.

    The generic GTP helper counts every terminal task_uid. Dispatch journals
    fenced/stale terminal reports too; counting those can declare an arm done
    even though no accepted outcome exists for some cells. Under the formal
    analyzer that state is unrecoverable, so filter only scheduler-accepted
    eval records here.
    """
    path = pathlib.Path(journal_path)
    if not path.exists():
        return list(arms), {}
    seen: dict[str, set[str]] = {arm: set() for arm in arms}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            arm = row.get("yaml_id")
            if (arm in seen and row.get("phase") == "eval"
                    and row.get("accepted") is True and row.get("task_uid")):
                seen[arm].add(str(row["task_uid"]))
    counts = {arm: len(seen[arm]) for arm in arms}
    return [arm for arm in arms if counts[arm] < expected], counts


class PrecheckSweepStrategy(ExperimentStrategy):
    """Sweep strategy that stamps the OFFICIAL init index on every episode.

    ``run_gtp.SweepStrategy`` writes ``orig_init_state_idx=episode_idx``, which
    is correct only when the pool IS the official pool. A' is 30 of the
    official 50 materialised into a fresh file, so its subset position 0..29 is
    NOT the official index -- reusing that strategy would label every episode
    with a provenance it does not have, and the A'/C/D_lib disjointness
    evidence would be unverifiable downstream (G2 B3).

    ``episode_idx`` stays the subset position (the client indexes the
    materialised pool with it); ``orig_init_state_idx`` carries the official
    0..49 index read from the frozen split manifest.
    """

    def __init__(self, task_suite: str, yaml_paths: dict[str, str], trials: int,
                 official_by_task: dict[int, list[int]]):
        self._task_suite = task_suite
        self._yaml_paths = yaml_paths
        self._trials = trials
        self._official = official_by_task

    def plan(self, yamls, server_assignment):
        from openpi.conductor import task as _task

        graph = _task.TaskGraph()
        for yaml_id in yamls:
            server = server_assignment[yaml_id]
            stage = _task.Stage(
                stage_id=f"eval__{yaml_id}",
                yaml_id=yaml_id,
                phase="eval",
                server=server,
                setup={"yaml_path": self._yaml_paths[yaml_id]},
            )
            for task_id in range(NUM_TASKS):
                officials = self._official[task_id]
                for ep_idx in range(self._trials):
                    stage.episodes.append(
                        _task.EpisodeTask(
                            task_uid=_task.make_task_uid(
                                yaml_id, "eval", task_id, ep_idx
                            ),
                            yaml_id=yaml_id,
                            phase="eval",
                            experiment=self._task_suite,
                            task_id=task_id,
                            episode_idx=ep_idx,
                            orig_init_state_idx=officials[ep_idx],
                            server_host=server.host,
                            server_port=server.port,
                            bundle_id=yaml_id,
                            extra={"num_trials_per_task": self._trials},
                        )
                    )
            graph.add_stage(stage)
        return graph

    def on_stage_begin(self, stage, ctl, ctx) -> None:
        """Load the arm bundle before any episode in its stage becomes ready."""
        yaml_path = stage.setup["yaml_path"]
        ctl.load_cache_config(
            yaml_content=pathlib.Path(yaml_path).read_text(encoding="utf-8"),
            yaml_id=stage.yaml_id,
            bundle_id=stage.yaml_id,
        )
        logger.info("arm %s: bundle loaded from %s", stage.yaml_id, yaml_path)


def official_test_inits(split_manifest_path: str, trials: int) -> dict[int, list[int]]:
    """task -> the official init indices A' holds, in materialised order.

    ``split_init_pools.materialize_pool`` writes ``states[sorted(indices)]``,
    so subset position i is the i-th smallest official index of that task.
    """
    manifest = json.loads(pathlib.Path(split_manifest_path).read_text())
    quota = (manifest.get("quota") or {}).get("test")
    if quota != trials:
        raise SystemExit(
            f"split manifest freezes {quota} test inits/task but --trials is {trials}"
        )
    out: dict[int, list[int]] = {}
    for tid_str, info in manifest["assignment"].items():
        officials = sorted(int(i) for i in info["test"])
        if len(officials) != trials or len(set(officials)) != trials:
            raise SystemExit(
                f"task {tid_str}: A' must hold exactly {trials} distinct official "
                f"inits, got {len(officials)} entries / {len(set(officials))} distinct"
            )
        if officials and (officials[0] < 0 or officials[-1] >= 50):
            raise SystemExit(f"task {tid_str}: official test indices outside 0..49")
        out[int(tid_str)] = officials
    if sorted(out) != list(range(NUM_TASKS)):
        raise SystemExit(f"split manifest task ids {sorted(out)} != 0..{NUM_TASKS - 1}")
    return out


def _state_content_sha256(states) -> str:
    """The canonical state-content digest used by split_init_pools."""
    import hashlib

    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()


def validate_aprime_pool(
    split_manifest_path: str, pool_dir: str | pathlib.Path, trials: int,
) -> dict:
    """Bind the worker's actual 30/task pool to the frozen split assignment.

    The cache-size helper deliberately freezes a different experiment's
    50/task pool. Reusing it here rejects the formal A' before launch and, even
    if its count were relaxed, would not prove that subset position ``i`` is
    the official init claimed by the dispatch split. The split manifest
    already carries the authoritative state-content digest and index mapping;
    recompute both against the exact directory handed to WorkerSpec.
    """
    split_path = pathlib.Path(split_manifest_path)
    manifest = json.loads(split_path.read_text())
    officials = official_test_inits(split_manifest_path, trials)
    if manifest.get("suite") is None:
        raise SystemExit("split manifest lacks suite")
    expected_digests = (manifest.get("pool_digests") or {}).get("test_aprime")
    if not isinstance(expected_digests, dict):
        raise SystemExit("split manifest lacks pool_digests.test_aprime")

    root = pathlib.Path(pool_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"A' pool directory does not exist: {root}")
    expected_names = {
        str(info["task_name"]): int(tid) for tid, info in manifest["assignment"].items()
    }
    if len(expected_names) != NUM_TASKS:
        raise SystemExit("split manifest task_name values are not ten distinct tasks")
    files = {p.stem: p for p in root.glob("*.init")}
    if set(files) != set(expected_names):
        raise SystemExit(
            "A' task files differ from the frozen split manifest: "
            f"missing={sorted(set(expected_names) - set(files))[:3]}, "
            f"extra={sorted(set(files) - set(expected_names))[:3]}"
        )
    if set(expected_digests) != set(expected_names):
        raise SystemExit("split manifest assignment and test_aprime digest keys disagree")

    raw_digests: dict[str, str] = {}
    state_digests: dict[str, str] = {}
    for task_name, task_id in sorted(expected_names.items()):
        states = load_init_states(files[task_name])
        frozen = expected_digests[task_name]
        if len(states) != trials or int(frozen.get("count", -1)) != trials:
            raise SystemExit(
                f"{task_name}: actual/frozen A' count is {len(states)}/"
                f"{frozen.get('count')}, expected {trials}"
            )
        if list(frozen.get("indices") or []) != officials[task_id]:
            raise SystemExit(
                f"{task_name}: pool digest indices disagree with split assignment"
            )
        content_sha = _state_content_sha256(states)
        if frozen.get("sha256") != content_sha:
            raise SystemExit(
                f"{task_name}: actual A' state bytes do not match the frozen split digest"
            )
        raw_digests[task_name] = digest_init_file(files[task_name])
        state_digests[task_name] = content_sha

    return {
        "suite": manifest["suite"],
        "apool_dir": str(root),
        "total_inits": NUM_TASKS * trials,
        "per_task_digests": raw_digests,
        "state_content_sha256": state_digests,
        "rollup_sha256": rollup_digest(raw_digests),
        "split_manifest_sha256": _file_sha256(split_path),
    }


def validate_optional_pool_record(path: str, pool: dict) -> None:
    """If supplied, require the legacy raw-file record to attest this pool."""
    if not path:
        return
    import yaml

    record = yaml.safe_load(pathlib.Path(path).read_text())
    for key in ("suite", "total_inits", "rollup_sha256", "per_task_digests"):
        if record.get(key) != pool.get(key):
            raise SystemExit(f"A' pool record {key} does not match the split-bound pool")


def validate_precheck_arms(arm_paths: dict[str, str], layer: str) -> dict[str, str]:
    """Accept exactly the two precheck verdict families; reject anything else."""
    for arm, path in arm_paths.items():
        cfg = load_cache_config(path)
        if cfg.routing is not None:
            raise SystemExit(f"arm {arm}: precheck has no executor routing ({path})")
        cp1 = cfg.checkpoints.get("cp1")
        if cp1 is None:
            raise SystemExit(f"arm {arm}: missing cp1 checkpoint ({path})")
        if cp1.gate.type != LAYER_EXPECTED_GATE[layer]:
            raise SystemExit(
                f"arm {arm}: gate {cp1.gate.type!r} does not match {layer} layer "
                f"contract {LAYER_EXPECTED_GATE[layer]!r}"
            )
        jt = cp1.judge.type
        if jt == "threshold":
            tiers = cp1.judge.warm_tiers or []
            if len(tiers) != 1 or tiers[0].get("start_t") != WS_START_T:
                raise SystemExit(
                    f"arm {arm}: threshold baseline must carry exactly one warm tier "
                    f"at start_t={WS_START_T}, got {tiers}"
                )
        elif jt != "dispatch_surface":
            raise SystemExit(
                f"arm {arm}: judge {jt!r} is not a precheck family "
                "(threshold+tier or dispatch_surface)"
            )
        if cfg.write_policy.type != "never":
            raise SystemExit(f"arm {arm}: write_policy must be 'never'")
    return arm_paths


def validate_matrix_artifacts(matrix: dict) -> None:
    """Re-attest every mutable fit/artifact path before server contact."""
    from openpi.cache.components.surface_judge import (
        CERTIFICATION_EMPIRICAL,
        load_surface_artifact,
    )

    if matrix.get("protocol") != PROTOCOL:
        raise SystemExit("arm matrix is not a frozen dispatch_surface_rev1 matrix")
    artifact_paths = matrix.get("artifact_paths")
    artifact_sha = matrix.get("artifact_sha256")
    record_paths = matrix.get("fit_record_paths")
    record_sha = matrix.get("fit_record_sha256")
    if not all(isinstance(value, dict) for value in (
        artifact_paths, artifact_sha, record_paths, record_sha,
    )):
        raise SystemExit("arm matrix lacks artifact/fit-record content bindings")
    expected_surface_arms = {
        arm for arm in matrix.get("arms", {}) if arm.startswith("dsp_s")
    }
    if set(artifact_paths) != expected_surface_arms or set(artifact_sha) != expected_surface_arms:
        raise SystemExit("arm matrix surface roster and artifact bindings disagree")
    if set(record_paths) != {"sv", "s0"} or set(record_sha) != {"sv", "s0"}:
        raise SystemExit("arm matrix must bind exactly the SV and S0 fit records")
    records = {}
    for name, raw in record_paths.items():
        path = pathlib.Path(raw).resolve()
        if not path.is_file() or _file_sha256(path) != record_sha[name]:
            raise SystemExit(f"fit record {name} is missing or content-drifted")
        records[name] = json.loads(path.read_text())
    if records["sv"].get("s_only") is not False or records["s0"].get("s_only") is not True:
        raise SystemExit("fit record modes do not form SV/S0")
    for name, record in records.items():
        if record.get("certification_mode") != CERTIFICATION_EMPIRICAL \
                or record.get("stop_loss") is not None:
            raise SystemExit(f"fit record {name} is not a completed empirical Rev 1 fit")
        for field in ("d0_binding", "dev_membership_sha256", "fold_map_sha256",
                      "final_fit_digests", "input_digests"):
            if not record.get(field):
                raise SystemExit(f"fit record {name} lacks audit field {field}")
    for field in ("delta_star", "d0_binding", "dev_membership_sha256",
                  "fold_map_sha256", "input_digests"):
        if records["sv"].get(field) != records["s0"].get(field):
            raise SystemExit(f"SV/S0 fit records differ on nested-ablation field {field}")

    import yaml

    for arm, raw in artifact_paths.items():
        path = pathlib.Path(raw).resolve()
        if not path.is_file() or _file_sha256(path) != artifact_sha[arm]:
            raise SystemExit(f"surface artifact for {arm} is missing or content-drifted")
        yaml_path = pathlib.Path(matrix["arms"][arm])
        yaml_artifact = yaml.safe_load(yaml_path.read_text())["checkpoints"]["cp1"]["judge"].get(
            "surface_artifact_path"
        )
        if pathlib.Path(str(yaml_artifact)).resolve() != path:
            raise SystemExit(f"arm {arm} YAML points at an artifact outside the matrix binding")
        artifact = load_surface_artifact(str(path))
        record = records["s0"] if arm == "dsp_s0" else records["sv"]
        expected_k = 1 if arm == "dsp_s0" else 5
        if (artifact.certification_mode != CERTIFICATION_EMPIRICAL
                or artifact.conformal_c != 0.0
                or artifact.n_calibration_episodes != 0
                or artifact.h_exec != FORMAL_H_EXEC
                or artifact.k != expected_k
                or artifact.retrieval_contract.get("top_k") != expected_k):
            raise SystemExit(f"surface artifact for {arm} violates the formal Rev 1 contract")
        if artifact.uses_disagreement != (arm != "dsp_s0"):
            raise SystemExit(f"surface artifact for {arm} has the wrong disagreement mode")
        expected_name = "primary" if arm in {"dsp_s0", "dsp_sv"} else arm.removeprefix("dsp_sv_")
        recorded_artifact = (record.get("artifacts") or {}).get(expected_name)
        if recorded_artifact is None or pathlib.Path(recorded_artifact).resolve() != path:
            raise SystemExit(f"surface artifact for {arm} is not the fit record's {expected_name}")
        if arm in {"dsp_s0", "dsp_sv"} and artifact.delta != record.get("delta_star"):
            raise SystemExit(f"primary surface artifact for {arm} has an unbound delta")
        if artifact.meta.get("delta_name") != expected_name:
            raise SystemExit(f"surface artifact for {arm} carries the wrong delta label")
        meta = artifact.meta
        for field in ("input_digests", "d0_binding", "dev_membership_sha256",
                      "fold_map_sha256", "final_fit_digests"):
            if meta.get(field) != record.get(field):
                raise SystemExit(f"surface artifact for {arm} drifts from fit record on {field}")


def assert_launch_contract(
    primary_artifact_path: str, replan_steps: int, servers: list[ServerEndpoint],
) -> dict:
    """Fail-fast binding of h_exec and the server-attested policy fingerprint."""
    from openpi.cache.components.surface_judge import load_surface_artifact

    artifact = load_surface_artifact(primary_artifact_path)
    contract = artifact.retrieval_contract
    if replan_steps != contract["h_exec"]:
        raise SystemExit(
            f"--replan-steps {replan_steps} != artifact h_exec {contract['h_exec']}; "
            "the calibrated execution window does not match this launch"
        )
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    attested = {}
    for ep in servers:
        client = WebsocketClientPolicy(host=ep.host, port=ep.port)
        try:
            meta = client.get_server_metadata()
        finally:
            close = getattr(client, "close", None)
            if close:
                close()
        fp = meta.get("policy_fingerprint")
        if fp != contract["policy_fingerprint"]:
            raise SystemExit(
                f"server {ep.host}:{ep.port} attests policy_fingerprint={fp!r} but the "
                f"surface contract requires {contract['policy_fingerprint']!r} — the "
                "server is not running the calibrated policy"
            )
        attested[f"{ep.host}:{ep.port}"] = {"policy_fingerprint": fp,
                                            "monitor_level": meta.get("monitor_level")}
    return {"h_exec": contract["h_exec"],
            "policy_fingerprint": contract["policy_fingerprint"],
            "servers": attested}


def validate_existing_launch_ledger(ledger: dict, new_entry: dict) -> None:
    """Reject resume drift before constructing a driver or running an episode.

    A resume may execute a strict subset of arms, so ``executed_*`` is
    intentionally per-launch. Everything that defines the experiment itself
    is frozen across the ledger.
    """
    if ledger.get("schema_version") != 2 or not isinstance(ledger.get("launches"), list):
        raise SystemExit("launch ledger is not schema_version 2; use a fresh output path")
    seen_run_ids: set[str] = set()
    frozen_yamls = new_entry["frozen_yaml_sha256"]
    for idx, prior in enumerate(ledger["launches"]):
        for key in FROZEN_LAUNCH_KEYS:
            if prior.get(key) != new_entry.get(key):
                raise SystemExit(f"launch ledger entry {idx} drifts on frozen key {key}")
        run_id = prior.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen_run_ids:
            raise SystemExit(f"launch ledger entry {idx} has missing/duplicate run_id")
        seen_run_ids.add(run_id)
        executed = prior.get("executed_arms")
        executed_sha = prior.get("executed_yaml_sha256")
        if not isinstance(executed, list) or not executed:
            raise SystemExit(f"launch ledger entry {idx} has no executed_arms")
        if set(executed) != set(executed_sha or {}):
            raise SystemExit(f"launch ledger entry {idx} executed arm/YAML keys disagree")
        if not set(executed).issubset(frozen_yamls):
            raise SystemExit(f"launch ledger entry {idx} contains an unknown arm")
        for arm in executed:
            if executed_sha[arm] != frozen_yamls[arm]:
                raise SystemExit(f"launch ledger entry {idx} YAML digest drift for {arm}")


def build_worker_specs(
    worker_server_keys: list[str], *, gpus: int, conda_env: str,
    task_suite: str, pool_dir: str, replan_steps: int, seed: int,
) -> list[WorkerSpec]:
    """Build the formal A′ fleet with execution/provenance indexing separated."""
    if not worker_server_keys or gpus <= 0:
        raise SystemExit("precheck requires at least one worker and one GPU slot")
    return [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=server_key,
            gpu_id=str(i % gpus),
            conda_env=conda_env,
            task_suite_name=task_suite,
            init_states_dir=pool_dir,
            init_state_index_mode="subset",
            replan_steps=replan_steps,
            seed=seed,
        )
        for i, server_key in enumerate(worker_server_keys)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-matrix", required=True, help="emit_precheck_yamls arm_matrix.json")
    ap.add_argument("--layer", required=True, choices=[LAYER_PRIMARY, LAYER_SECONDARY],
                    help="must match the arm matrix; the layers keep separate ledgers")
    ap.add_argument("--task-suite", required=True)
    ap.add_argument("--servers", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--server-workers", default="")
    ap.add_argument("--arms", default="", help="comma list; empty = all")
    ap.add_argument("--no-resume-filter", action="store_true")
    ap.add_argument("--trials", type=int, required=True, help="episodes per task (= A' inits)")
    ap.add_argument("--replan-steps", type=int, required=True,
                    help="explicit, no default: must equal the artifact h_exec")
    ap.add_argument("--seed", type=int, default=7, help="LIBERO environment seed")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step-out", required=True)
    ap.add_argument("--split-manifest", required=True,
                    help="frozen split manifest; supplies the official init identity")
    ap.add_argument(
        "--pool-record",
        default="",
        help="optional legacy raw-file digest record; the split manifest is authoritative",
    )
    ap.add_argument(
        "--pool-dir",
        default="",
        help="A' init directory (default: test_aprime beside the split manifest)",
    )
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--eval-concurrency", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--conda-env", default="")
    ap.add_argument(
        "--dry-validate", action="store_true",
        help="run every launch-gate check against the real files and server, then "
             "exit without running an episode or appending to the launch ledger",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.trials != FORMAL_TRIALS:
        raise SystemExit(f"formal precheck is frozen at --trials {FORMAL_TRIALS}")
    if args.workers <= 0 or args.gpus <= 0:
        raise SystemExit("--workers and --gpus must be positive")

    matrix = json.loads(pathlib.Path(args.arm_matrix).read_text())
    matrix_arm_ids = set(matrix.get("arms") or {})
    layer = matrix.get("layer")
    if layer not in LAYER_EXPECTED_GATE:
        raise SystemExit(
            f"arm matrix declares layer {layer!r}; expected one of "
            f"{sorted(LAYER_EXPECTED_GATE)} — a matrix without a layer predates Rev 1"
        )
    if layer != args.layer:
        raise SystemExit(
            f"--layer {args.layer!r} does not match the matrix's layer {layer!r}; "
            "the two layers keep separate ledgers and must not be crossed"
        )
    expected_core = FORMAL_CORE_ARMS if layer == LAYER_PRIMARY else SECONDARY_CORE_ARMS
    if set(matrix.get("core_arms") or []) != expected_core:
        raise SystemExit(
            f"{layer} arm matrix must contain exactly {sorted(expected_core)}"
        )
    if matrix.get("gate_type") != LAYER_EXPECTED_GATE[layer]:
        raise SystemExit(
            f"{layer} arm matrix declares gate_type={matrix.get('gate_type')!r}; "
            f"expected {LAYER_EXPECTED_GATE[layer]!r}"
        )
    validate_matrix_artifacts(matrix)
    if set(matrix.get("descriptive_arms") or []) != matrix_arm_ids - expected_core:
        raise SystemExit("arm matrix descriptive arm roster is inconsistent")
    arm_paths: dict[str, str] = dict(matrix["arms"])
    if args.arms:
        wanted = set(args.arms.split(","))
        missing = wanted - set(arm_paths)
        if missing:
            raise SystemExit(f"unknown arms requested: {sorted(missing)}")
        arm_paths = {a: p for a, p in arm_paths.items() if a in wanted}
    if not args.no_resume_filter:
        expected = NUM_TASKS * args.trials
        remaining, _counts = arms_with_accepted_work_left(
            args.journal, list(arm_paths), expected=expected
        )
        arm_paths = {a: p for a, p in arm_paths.items() if a in set(remaining)}
        if not arm_paths:
            logger.info("every arm is complete; nothing to run")
            return
    yaml_paths = validate_precheck_arms(arm_paths, layer)

    servers = []
    for spec in args.servers.split(","):
        if ":" not in spec:
            raise SystemExit(f"--servers entry {spec!r} must be host:port")
        host, port = spec.rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))

    # Launch contract before anything runs.
    sv_yaml = matrix["arms"].get("dsp_sv")
    if sv_yaml is None:
        raise SystemExit("arm matrix has no primary surface arm 'dsp_sv'")
    import yaml as _yaml

    primary_artifact = _yaml.safe_load(open(sv_yaml))["checkpoints"]["cp1"]["judge"][
        "surface_artifact_path"
    ]
    contract_binding = assert_launch_contract(primary_artifact, args.replan_steps, servers)

    per_step_path = pathlib.Path(args.per_step_out)
    per_step_path.parent.mkdir(parents=True, exist_ok=True)

    pool_dir = args.pool_dir or str(
        pathlib.Path(args.split_manifest).resolve().parent / "test_aprime"
    )
    pool = validate_aprime_pool(args.split_manifest, pool_dir, args.trials)
    validate_optional_pool_record(args.pool_record, pool)
    if pool["suite"] != args.task_suite:
        raise SystemExit(
            f"A' split suite {pool['suite']!r} != --task-suite {args.task_suite!r}"
        )

    matrix_arms = dict(matrix["arms"])
    matrix_yaml_sha = {
        arm: _file_sha256(pathlib.Path(path)) for arm, path in matrix_arms.items()
    }
    recorded_yaml_sha = matrix.get("arm_yaml_sha256")
    if recorded_yaml_sha != matrix_yaml_sha:
        raise SystemExit("arm matrix YAML digests do not match the files about to run")
    # Freeze WHAT IS ABOUT TO RUN, not what happens to be on disk at analysis
    # time: the arm matrix and every executed yaml are digested here and the
    # driver's run id is recorded, so the adjudicator can bind each accepted
    # episode back to the configuration that produced it (G2 B4). Resume
    # appends another entry. The analyzer requires experiment-wide fields and
    # all matrix YAMLs to stay frozen, while each entry records the subset that
    # this run actually executed.
    launch_entry = {
        "protocol": PROTOCOL,
        "layer": layer,
        "suite": args.task_suite,
        "executed_arms": sorted(yaml_paths),
        "core_arms": sorted(matrix["core_arms"]),
        "descriptive_arms": sorted(matrix["descriptive_arms"]),
        "trials_per_task": args.trials,
        "replan_steps": args.replan_steps,
        "env_seed": args.seed,
        "policy_fingerprint": contract_binding["policy_fingerprint"],
        "contract_binding": contract_binding,
        "library_sha256": matrix["library_sha256"],
        "aprime_content_sha256": pool["rollup_sha256"],
        "split_manifest": args.split_manifest,
        "split_manifest_sha256": pool["split_manifest_sha256"],
        "arm_matrix_sha256": _file_sha256(pathlib.Path(args.arm_matrix)),
        "frozen_yaml_sha256": matrix_yaml_sha,
        "artifact_sha256": matrix["artifact_sha256"],
        "fit_record_sha256": matrix["fit_record_sha256"],
        "executed_yaml_sha256": {
            arm: _file_sha256(pathlib.Path(path)) for arm, path in yaml_paths.items()
        },
        "pool": pool,
    }
    launch_path = pathlib.Path(str(per_step_path) + ".launch.json")
    ledger = {"schema_version": 2, "launches": []}
    if launch_path.is_file():
        prior = json.loads(launch_path.read_text())
        if prior.get("schema_version") != 2:
            raise SystemExit(
                "pre-ledger launch manifest cannot be resumed safely; use fresh output paths"
            )
        ledger = prior
    validate_existing_launch_ledger(ledger, launch_entry)
    logger.info("launch bound to A' pool rollup %s", pool["rollup_sha256"])

    if args.server_workers:
        counts = [int(x) for x in args.server_workers.split(",")]
        if len(counts) != len(servers):
            raise SystemExit("--server-workers count mismatch with --servers")
        if any(count <= 0 for count in counts):
            raise SystemExit("--server-workers entries must be positive")
        worker_server_keys = [s.key for s, c in zip(servers, counts) for _ in range(c)]
        server_capacities = {s.key: c for s, c in zip(servers, counts)}
    else:
        worker_server_keys = [servers[i % len(servers)].key for i in range(args.workers)]
        server_capacities = None

    from examples.libero.episode_runner import default_client_factory

    per_step_lock = threading.Lock()
    merged = merge_snapshot(per_step_path, per_step_path.with_suffix(".snapshot.jsonl"))
    if merged:
        logger.info("merged %d snapshot rows on resume", merged)

    def _per_step_writer(yaml_id: str, rows: list[dict]) -> None:
        with per_step_lock, per_step_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"yaml_id": yaml_id, **row}) + "\n")

    official = official_test_inits(args.split_manifest, args.trials)

    if args.dry_validate:
        # Everything above is the launch gate: matrix layer/roster/gate, artifact
        # and fit-record content bindings, A' pool attestation, the launch
        # contract read back from the SERVER's own metadata, and the resume
        # ledger. Stopping here exercises all of it against the real files and
        # the real server without running an episode -- and without appending a
        # ledger entry, which would burn a run_id and make the eventual real
        # launch look like a resume.
        summary = {
            "dry_validate": True,
            "layer": layer,
            "suite": args.task_suite,
            "arms": sorted(yaml_paths),
            "trials_per_task": args.trials,
            "replan_steps": args.replan_steps,
            "env_seed": args.seed,
            "official_inits_per_task": {t: len(v) for t, v in sorted(official.items())},
            "contract_binding": contract_binding,
            "would_append_launch": {
                k: launch_entry.get(k) for k in sorted(FROZEN_LAUNCH_KEYS)
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        logger.info("dry validation passed for layer %s; no episode was run", layer)
        return

    strategy = PrecheckSweepStrategy(
        args.task_suite, yaml_paths, args.trials, official
    )
    driver = ConductorDriver(
        strategy,
        yaml_weights={arm: 100 for arm in yaml_paths},
        servers=servers,
        journal_path=args.journal,
        ctl_factory=default_client_factory,
        episode_timeout_s=args.episode_timeout_s,
        bind_host=args.bind_host,
        scheduler_kwargs=(
            {"eval_concurrency": args.eval_concurrency} if args.eval_concurrency else None
        ),
        server_capacities=server_capacities,
        per_step_writer=_per_step_writer,
    )
    launch_entry["run_id"] = driver.run_id
    ledger["launches"].append(launch_entry)
    launch_path.write_text(json.dumps(ledger, indent=2))
    logger.info("launch %s recorded in %s", driver.run_id, launch_path)

    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    snapshot_stop = threading.Event()
    snapshot_thread = threading.Thread(
        target=_snapshot_loop, args=(driver, per_step_path, snapshot_stop), daemon=True
    )
    snapshot_thread.start()
    while driver.port is None:
        time.sleep(0.05)
    logger.info("driver pull port = %d", driver.port)

    specs = build_worker_specs(
        worker_server_keys,
        gpus=args.gpus,
        conda_env=args.conda_env,
        task_suite=args.task_suite,
        pool_dir=pool["apool_dir"],
        replan_steps=args.replan_steps,
        seed=args.seed,
    )
    agent = WorkerAgent(specs, driver_host=args.bind_host, driver_port=driver.port)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    try:
        driver_thread.join()
    finally:
        snapshot_stop.set()
        snapshot_thread.join(timeout=10)
        try:
            n = _write_snapshot(driver, per_step_path)
            logger.info("final snapshot: %d in-memory rows dumped", n)
        except Exception:  # noqa: BLE001 - bookkeeping must not mask the run outcome
            logger.exception("final snapshot failed; trailing rows may be missing")
        agent.stop()
        agent_thread.join(timeout=30)


if __name__ == "__main__":
    main()
