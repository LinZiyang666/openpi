"""Conductor driver for the hybrid-gate threshold sweep (warmup and eval).

One runner serves both phases because they differ only in the arm matrix and
the trial count: the warmup arms carry ``always_search`` + a force-MISS
threshold, the eval arms carry the N4 hybrid gate + a solved threshold, and both
roll out over the same frozen A-pool with the same episode identity.

Generic conductor plumbing -- A-pool re-hashing, snapshot merge, crash-time
snapshots -- is imported from the cache-size runner rather than copied. That
runner's *gates* are not reused: its pure-cache validator demands
``judge: always_hit`` + ``gate: always_search`` and would reject every arm here,
and its FULL_HIT witness asserts a premise this experiment deliberately breaks
(a thresholded verdict is supposed to serve MISS steps).

The gate this runner does enforce is the shape of the arms themselves: warm
tiers absent, routing absent, and -- for eval arms -- the hybrid gate actually
configured with ``L``. An arm that silently lost its ``L`` would run as pure N1
and be indistinguishable in the results from one that kept it.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import threading
import time

import yaml

from openpi.cache.config import load_cache_config
from openpi.conductor import ConductorDriver
from openpi.conductor import ServerEndpoint
from openpi.conductor import WorkerAgent
from openpi.conductor import WorkerSpec
from openpi.conductor import strategy as _strat
from openpi.conductor import task as _task

from exp.ablation_study.cache_size.run_size_eval import (
    _snapshot_loop,
    _write_snapshot,
    load_apool_digest,
    merge_snapshot,
)
from exp.gate_threshold_pareto.emit_gtp_yamls import GATE_L

logger = logging.getLogger(__name__)

NUM_TASKS = 10


class SweepStrategy(_strat.ExperimentStrategy):
    """One independent stage per arm yaml; the bundle is hot-swapped per stage."""

    def __init__(
        self, task_suite: str, yaml_paths: dict[str, str], trials: int
    ) -> None:
        self._task_suite = task_suite
        self._yaml_paths = yaml_paths
        self._trials = trials

    def plan(self, yamls, server_assignment) -> _task.TaskGraph:
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
                            orig_init_state_idx=ep_idx,
                            server_host=server.host,
                            server_port=server.port,
                            bundle_id=yaml_id,
                            extra={"num_trials_per_task": self._trials},
                        )
                    )
            graph.add_stage(stage)
        return graph

    def on_stage_begin(self, stage, ctl, ctx) -> None:
        yaml_path = stage.setup["yaml_path"]
        ctl.load_cache_config(
            yaml_content=pathlib.Path(yaml_path).read_text(encoding="utf-8"),
            yaml_id=stage.yaml_id,
            bundle_id=stage.yaml_id,
        )
        logger.info("arm %s: bundle loaded from %s", stage.yaml_id, yaml_path)


def arms_with_work_left(
    journal_path: str | pathlib.Path, arms: list[str], *, expected: int
) -> tuple[list[str], dict[str, int]]:
    """Drop arms whose episodes are already all in the journal.

    Resume is episode-level, so a completed arm contributes no work -- but the
    driver still walks its stage, and every stage walked calls
    ``ctl.load_cache_config``. On a resume late in a sweep that becomes a burst
    of back-to-back bundle swaps: the server tears down and rebuilds its cache
    backend, reloading a gigabyte-scale library each time, while replicas are
    concurrently serving. Measured on 2026-08-20: 23 swaps inside a few seconds
    put the GPU into an MMU fault (Xid 31, null-address read from a freed
    structure) within a minute, twice, on a run that had been stable for four
    and a half hours at one swap per eight minutes.

    Filtering here removes the burst at its source. Counting is over distinct
    ``task_uid`` so a retried episode is not double-counted into a false
    "complete".
    """
    journal_path = pathlib.Path(journal_path)
    if not journal_path.exists():
        return list(arms), {}

    seen: dict[str, set[str]] = {}
    with journal_path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("task_uid", "")
            arm = rec.get("yaml_id") or (uid.split(":", 1)[0] if uid else None)
            if arm and uid:
                seen.setdefault(arm, set()).add(uid)

    counts = {arm: len(seen.get(arm, ())) for arm in arms}
    remaining = [arm for arm in arms if counts[arm] < expected]
    return remaining, counts


def validate_arms(
    arm_rows: list[dict], *, phase: str, expected_l: int = GATE_L
) -> dict[str, str]:
    """Every arm must be warm-tier-free, routing-free, and carry the right gate."""
    yaml_paths: dict[str, str] = {}
    for row in arm_rows:
        arm, path = row["arm"], row["yaml"]
        cfg = load_cache_config(path)
        if cfg.routing is not None:
            raise SystemExit(f"arm {arm}: this sweep has no executor routing ({path})")
        cp1 = cfg.checkpoints.get("cp1")
        if cp1 is None:
            raise SystemExit(f"arm {arm}: missing cp1 checkpoint ({path})")
        if cp1.judge.type != "threshold":
            raise SystemExit(
                f"arm {arm}: judge is {cp1.judge.type!r}, expected 'threshold'"
            )
        if cp1.judge.warm_tiers:
            raise SystemExit(
                f"arm {arm}: warm tier present ({cp1.judge.warm_tiers}). The warm-start "
                "route is disabled for this experiment; a surviving tier would make the "
                "verdict three-way and the inference-ratio axis incomparable."
            )
        if phase == "warmup":
            if cp1.gate.type != "always_search":
                raise SystemExit(
                    f"arm {arm}: warmup gate is {cp1.gate.type!r}, expected 'always_search' "
                    "-- a gated warmup skips steps, and the skipped steps carry no score, "
                    "so the solved quantiles would describe a censored distribution."
                )
        else:
            if cp1.gate.type != "score_hysteresis":
                raise SystemExit(
                    f"arm {arm}: eval gate is {cp1.gate.type!r}, expected 'score_hysteresis'"
                )
            if cp1.gate.L != expected_l:
                raise SystemExit(
                    f"arm {arm}: gate L is {cp1.gate.L!r}, expected {expected_l}. Without L the "
                    "gate degrades to pure N1 and the run would be silently mislabelled."
                )
        yaml_paths[arm] = path
    if not yaml_paths:
        raise SystemExit("no arms selected")
    return yaml_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid-gate threshold sweep runner")
    ap.add_argument("--arm-matrix", required=True)
    ap.add_argument("--phase", choices=("warmup", "eval"), required=True)
    ap.add_argument("--task-suite", required=True)
    ap.add_argument("--servers", required=True, help="host:port[,host:port...]")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--server-workers", default="")
    ap.add_argument("--arms", default="", help="comma list; empty = all")
    ap.add_argument(
        "--no-resume-filter",
        action="store_true",
        help="walk completed arms too; only for reproducing the bundle-swap burst",
    )
    ap.add_argument("--trials", type=int, required=True, help="episodes per task")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step-out", required=True)
    ap.add_argument("--apool-record", required=True)
    ap.add_argument(
        "--apool-dir",
        default="",
        help="where the frozen pool lives on THIS host; the record stays unedited and "
        "its digests are still re-hashed from these files, so a substituted pool fails",
    )
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--eval-concurrency", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--conda-env", default="")
    ap.add_argument(
        "--gate-l",
        type=int,
        default=GATE_L,
        help="expected gate lockout L for arm validation (gate-only ablation uses 8)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    matrix = yaml.safe_load(pathlib.Path(args.arm_matrix).read_text(encoding="utf-8"))
    rows = matrix["arms"]
    if args.arms:
        wanted = set(args.arms.split(","))
        rows = [r for r in rows if r["arm"] in wanted]
        missing = wanted - {r["arm"] for r in rows}
        if missing:
            raise SystemExit(f"unknown arms requested: {sorted(missing)}")
    if not args.no_resume_filter:
        expected = NUM_TASKS * args.trials
        remaining, counts = arms_with_work_left(
            args.journal, [r["arm"] for r in rows], expected=expected
        )
        dropped = [a for a in counts if a not in set(remaining)]
        if dropped:
            logger.info(
                "resume: %d/%d arm(s) already complete, not walking their stages "
                "(avoids the bundle-swap burst): %s",
                len(dropped),
                len(rows),
                ", ".join(sorted(dropped)),
            )
            rows = [r for r in rows if r["arm"] in set(remaining)]
        if not rows:
            logger.info("every arm is complete; nothing to run")
            return

    yaml_paths = validate_arms(rows, phase=args.phase, expected_l=args.gate_l)

    per_step_path = pathlib.Path(args.per_step_out)
    per_step_path.parent.mkdir(parents=True, exist_ok=True)

    # The A-pool is re-hashed from disk, not read back out of the record: the
    # record is a claim, the files are the evidence.
    if args.apool_dir:
        # The record was frozen on the host that materialized the pool; on any
        # other host its absolute path is meaningless. Rewriting only the path
        # (never the digests) keeps the attestation intact -- verify_contents
        # re-hashes the files at the new location and fails if they differ.
        record = yaml.safe_load(
            pathlib.Path(args.apool_record).read_text(encoding="utf-8")
        )
        record["apool_dir"] = args.apool_dir
        relocated = per_step_path.parent / f"apool_{args.task_suite}.local.yaml"
        relocated.parent.mkdir(parents=True, exist_ok=True)
        relocated.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
        apool = load_apool_digest(str(relocated), required=True, verify_contents=True)
    else:
        apool = load_apool_digest(
            args.apool_record, required=True, verify_contents=True
        )
    if apool["suite"] != args.task_suite:
        raise SystemExit(
            f"A-pool record is for {apool['suite']!r}, run is for {args.task_suite!r}"
        )

    pathlib.Path(str(per_step_path) + ".launch.json").write_text(
        json.dumps(
            {
                "phase": args.phase,
                "suite": args.task_suite,
                "arms": sorted(yaml_paths),
                "trials_per_task": args.trials,
                "gate_L": args.gate_l,
                "warm_start": "disabled",
                "apool": apool,
            },
            indent=2,
        )
    )
    logger.info("launch bound to A-pool rollup %s", apool["rollup_sha256"])

    servers = []
    for spec in args.servers.split(","):
        if ":" not in spec:
            raise SystemExit(f"--servers entry {spec!r} must be host:port")
        host, port = spec.rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))

    if args.server_workers:
        counts = [int(x) for x in args.server_workers.split(",")]
        if len(counts) != len(servers):
            raise SystemExit("--server-workers count mismatch with --servers")
        worker_server_keys = [s.key for s, c in zip(servers, counts) for _ in range(c)]
        server_capacities = {s.key: c for s, c in zip(servers, counts)}
    else:
        worker_server_keys = [
            servers[i % len(servers)].key for i in range(args.workers)
        ]
        server_capacities = None

    from examples.libero.episode_runner import default_client_factory

    per_step_lock = threading.Lock()
    merged = merge_snapshot(per_step_path, per_step_path.with_suffix(".snapshot.jsonl"))
    if merged:
        logger.info("merged %d snapshot rows into %s on resume", merged, per_step_path)

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
            {"eval_concurrency": args.eval_concurrency}
            if args.eval_concurrency
            else None
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
            init_states_dir=apool["apool_dir"],
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
            logger.info("final snapshot: %d in-memory rows dumped before merge", n)
        except Exception:  # noqa: BLE001 - bookkeeping must not mask the run outcome
            logger.exception("final snapshot failed; trailing rows may be missing")
        agent.stop()
        agent_thread.join(timeout=30)


if __name__ == "__main__":
    main()
