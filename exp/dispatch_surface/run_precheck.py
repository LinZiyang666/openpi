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

from exp.ablation_study.cache_size.run_size_eval import (
    _snapshot_loop,
    _write_snapshot,
    load_apool_digest,
    merge_snapshot,
)
from exp.gate_threshold_pareto.run_gtp import SweepStrategy, arms_with_work_left
from openpi.cache.config import load_cache_config
from openpi.conductor import ConductorDriver, ServerEndpoint, WorkerAgent, WorkerSpec

logger = logging.getLogger("dispatch_surface.precheck")

NUM_TASKS = 10
WS_START_T = 0.3


def validate_precheck_arms(arm_paths: dict[str, str]) -> dict[str, str]:
    """Accept exactly the two precheck verdict families; reject anything else."""
    for arm, path in arm_paths.items():
        cfg = load_cache_config(path)
        if cfg.routing is not None:
            raise SystemExit(f"arm {arm}: precheck has no executor routing ({path})")
        cp1 = cfg.checkpoints.get("cp1")
        if cp1 is None:
            raise SystemExit(f"arm {arm}: missing cp1 checkpoint ({path})")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-matrix", required=True, help="emit_precheck_yamls arm_matrix.json")
    ap.add_argument("--task-suite", required=True)
    ap.add_argument("--servers", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--server-workers", default="")
    ap.add_argument("--arms", default="", help="comma list; empty = all")
    ap.add_argument("--no-resume-filter", action="store_true")
    ap.add_argument("--trials", type=int, required=True, help="episodes per task (= A' inits)")
    ap.add_argument("--replan-steps", type=int, required=True,
                    help="explicit, no default: must equal the artifact h_exec")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step-out", required=True)
    ap.add_argument("--pool-record", required=True, help="A' pool digest record yaml")
    ap.add_argument("--pool-dir", default="")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--eval-concurrency", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--conda-env", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    matrix = json.loads(pathlib.Path(args.arm_matrix).read_text())
    arm_paths: dict[str, str] = dict(matrix["arms"])
    if args.arms:
        wanted = set(args.arms.split(","))
        missing = wanted - set(arm_paths)
        if missing:
            raise SystemExit(f"unknown arms requested: {sorted(missing)}")
        arm_paths = {a: p for a, p in arm_paths.items() if a in wanted}
    if not args.no_resume_filter:
        expected = NUM_TASKS * args.trials
        remaining, _counts = arms_with_work_left(
            args.journal, list(arm_paths), expected=expected
        )
        arm_paths = {a: p for a, p in arm_paths.items() if a in set(remaining)}
        if not arm_paths:
            logger.info("every arm is complete; nothing to run")
            return
    yaml_paths = validate_precheck_arms(arm_paths)

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

    if args.pool_dir:
        import yaml as _y

        record = _y.safe_load(pathlib.Path(args.pool_record).read_text())
        record["apool_dir"] = args.pool_dir
        relocated = per_step_path.parent / f"aprime_{args.task_suite}.local.yaml"
        relocated.write_text(_y.safe_dump(record, sort_keys=False))
        pool = load_apool_digest(str(relocated), required=True, verify_contents=True)
    else:
        pool = load_apool_digest(args.pool_record, required=True, verify_contents=True)

    import hashlib as _hashlib

    _aprime_h = _hashlib.sha256()
    for f in sorted(pathlib.Path(pool["apool_dir"]).glob("*.init")):
        _aprime_h.update(f.name.encode("utf-8"))
        _aprime_h.update(f.read_bytes())
    pathlib.Path(str(per_step_path) + ".launch.json").write_text(json.dumps({
        "suite": args.task_suite,
        "arms": sorted(yaml_paths),
        "core_arms": matrix["core_arms"],
        "descriptive_arms": matrix["descriptive_arms"],
        "trials_per_task": args.trials,
        "replan_steps": args.replan_steps,
        "contract_binding": contract_binding,
        "library_sha256": matrix["library_sha256"],
        "aprime_content_sha256": _aprime_h.hexdigest(),
        "pool": pool,
    }, indent=2))
    logger.info("launch bound to A' pool rollup %s", pool["rollup_sha256"])

    if args.server_workers:
        counts = [int(x) for x in args.server_workers.split(",")]
        if len(counts) != len(servers):
            raise SystemExit("--server-workers count mismatch with --servers")
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

    strategy = SweepStrategy(args.task_suite, yaml_paths, args.trials)
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

    specs = [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=worker_server_keys[i],
            gpu_id=str(i % args.gpus),
            conda_env=args.conda_env,
            task_suite_name=args.task_suite,
            init_states_dir=pool["apool_dir"],
        )
        for i in range(len(worker_server_keys))
    ]
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
