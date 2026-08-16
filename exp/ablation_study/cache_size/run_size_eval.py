"""Conductor driver for the cache-size ablation's paired A-pool evaluation.

A dedicated runner rather than a reuse of the executor-substitution driver: that
one requires every non-baseline arm to carry a ``routing`` section, and all 16
arms here are pure-cache by design (no sidecar, no executor override), so it
rejects them before a single episode starts. Its TOST power gate also encodes a
different statistical contract (paired equivalence at delta=3pp), whereas this
experiment pre-registers an eight-test Holm family over task-level clusters.

Three gates specific to this experiment:

*   **pure-cache validation** -- every arm must have no ``routing``,
    ``judge: always_hit`` and ``gate: always_search``. An arm that quietly
    carried a threshold verdict would reintroduce the calibration degree of
    freedom the design exists to remove.
*   **A-pool binding** -- the evaluation inits are the frozen official pool; the
    run records the digest it was launched against so the analysis can prove
    which 500 episodes it is talking about.
*   **FULL_HIT witness** -- retrieval is task-scoped, so a tier whose library
    misses a task silently serves that task from the teacher. Every routed step
    is expected to be a FULL_HIT; the rate is measured per arm and checked.

Episode identity, resume and per-step capture follow the established conductor
contract (``task_uid`` = ``<yaml_id>:<phase>:<task_id>:<episode_idx>``), so the
journal this writes is directly consumable by ``analysis/analyze_size.py``.
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

logger = logging.getLogger(__name__)

NUM_TASKS = 10
TRIALS_PER_TASK = 50  # official A-pool protocol: 50 eval inits per task


class PureCacheEvalStrategy(_strat.ExperimentStrategy):
    """One independent eval stage per arm yaml; no routing, no sidecar."""

    def __init__(self, task_suite: str, yaml_paths: dict[str, str], trials: int) -> None:
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
                            task_uid=_task.make_task_uid(yaml_id, "eval", task_id, ep_idx),
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


def validate_pure_cache_arms(arm_rows: list[dict]) -> dict[str, str]:
    """Every arm must be a threshold-free, routing-free pure-cache config."""
    yaml_paths: dict[str, str] = {}
    for row in arm_rows:
        arm, path = row["arm"], row["yaml"]
        if row.get("sidecar") is not None:
            raise SystemExit(f"arm {arm}: pure-cache arms must not declare a sidecar")
        cfg = load_cache_config(path)  # raises on allowlist violations
        if cfg.routing is not None:
            raise SystemExit(
                f"arm {arm}: pure-cache arms must not carry a routing section ({path})"
            )
        cp1 = cfg.checkpoints.get("cp1")
        if cp1 is None:
            raise SystemExit(f"arm {arm}: missing cp1 checkpoint ({path})")
        if cp1.judge.type != "always_hit":
            raise SystemExit(
                f"arm {arm}: judge is {cp1.judge.type!r}, expected 'always_hit' -- a "
                "threshold verdict would reintroduce the calibration degree of freedom"
            )
        if cp1.gate.type != "always_search":
            raise SystemExit(
                f"arm {arm}: gate is {cp1.gate.type!r}, expected 'always_search'"
            )
        # The loader only requires preload_path to be non-empty; existence is
        # checked server-side at bundle load. When the path happens to resolve
        # locally we can surface a typo now instead of after three setup retries.
        preload = cfg.backend.in_memory.preload_path
        if preload and not pathlib.Path(preload).exists():
            logger.warning(
                "arm %s: preload_path %s does not exist on this host. Harmless if the "
                "server resolves it elsewhere; a typo otherwise, and the failure would "
                "surface only as retried stage setup errors.",
                arm, preload,
            )
        yaml_paths[arm] = path
    if not yaml_paths:
        raise SystemExit("no arms selected")
    return yaml_paths


def rehash_apool(apool_dir: pathlib.Path, *, expect_per_task: int = TRIALS_PER_TASK) -> dict:
    """Recompute per-task digests, the rollup, and the init count, from the files.

    A record is a claim, not evidence. Reading its digests back and declaring a
    match proves only that the file is internally consistent -- swap any ``.init``
    after the record was written and the run still "passes" while every worker
    loads the substituted pool. So the digests reported here are computed from
    the bytes on disk at launch time, and the caller compares them item by item.
    """
    from exp.ablation_study.cache_size.verify_apool import digest_init_file, rollup_digest

    import torch  # noqa: PLC0415 -- client env only, and heavy

    files = sorted(apool_dir.glob("*.init"))
    if len(files) != NUM_TASKS:
        raise SystemExit(
            f"A-pool directory {apool_dir} holds {len(files)} .init files, expected {NUM_TASKS}"
        )
    digests = {f.stem: digest_init_file(f) for f in files}
    counts = {}
    for f in files:
        states = torch.load(f, weights_only=False)
        counts[f.stem] = len(states)
    bad = {k: v for k, v in counts.items() if v != expect_per_task}
    if bad:
        raise SystemExit(
            f"A-pool tasks with the wrong init count (expected {expect_per_task}): {bad}. "
            "The evaluation grid is 10 tasks x 50 inits; a short task silently "
            "shrinks the denominator of that task's success rate."
        )
    return {
        "per_task_digests": digests,
        "rollup_sha256": rollup_digest(digests),
        "total_inits": sum(counts.values()),
    }


def load_apool_digest(path: str | None, *, required: bool = True,
                      verify_contents: bool = True) -> dict | None:
    """Read the frozen A-pool record and check it against the files on disk.

    The record must name a directory, not just a digest: that directory is what
    every worker is pointed at, so the digest attests the pool the run actually
    used. Without it a record produced in one environment could ride along with
    a different environment's default pool and still "pass". And the digests are
    re-derived from that directory here -- see ``rehash_apool`` for why reading
    them back out of the record would attest nothing.
    """
    if not path:
        if required:
            raise SystemExit(
                "--apool-record is required: without it the workers fall back to "
                "whatever init pool their environment ships, and the 500 evaluated "
                "episodes cannot be attested. Use --smoke for unbound subset runs."
            )
        return None
    record = yaml.safe_load(pathlib.Path(path).read_text())
    for key in ("suite", "total_inits", "rollup_sha256", "apool_dir", "per_task_digests"):
        if key not in record:
            raise SystemExit(f"A-pool record {path} lacks {key!r}")
    expected = NUM_TASKS * TRIALS_PER_TASK
    if record["total_inits"] != expected:
        raise SystemExit(
            f"A-pool record declares {record['total_inits']} inits, expected {expected}"
        )
    apool_dir = pathlib.Path(record["apool_dir"])
    if not apool_dir.is_dir():
        raise SystemExit(f"A-pool directory {apool_dir} from the record does not exist")
    digests = record["per_task_digests"]
    if len(digests) != NUM_TASKS:
        raise SystemExit(
            f"A-pool record lists {len(digests)} per-task digests, expected {NUM_TASKS}"
        )
    if not verify_contents:
        return record

    actual = rehash_apool(apool_dir)
    if set(actual["per_task_digests"]) != set(digests):
        only_disk = sorted(set(actual["per_task_digests"]) - set(digests))
        only_rec = sorted(set(digests) - set(actual["per_task_digests"]))
        raise SystemExit(
            f"A-pool task names differ from the record: on disk only {only_disk}, "
            f"in record only {only_rec}"
        )
    changed = sorted(k for k, v in digests.items() if actual["per_task_digests"][k] != v)
    if changed:
        raise SystemExit(
            f"A-pool contents changed since the record was written: {changed}. "
            "The frozen evaluation pool is not what is on disk, so the run would "
            "attest a digest it never used."
        )
    if actual["rollup_sha256"] != record["rollup_sha256"]:
        raise SystemExit(
            f"A-pool rollup mismatch: recomputed {actual['rollup_sha256']}, "
            f"record says {record['rollup_sha256']}"
        )
    if actual["total_inits"] != expected:
        raise SystemExit(
            f"A-pool holds {actual['total_inits']} inits on disk, expected {expected}"
        )
    return record


def assert_full_hit(rates: dict[str, float], expected_arms: set[str], minimum: float) -> None:
    """Every expected arm must be present AND at or above the floor.

    Absence is the dangerous case: an arm with no per-step rows is exactly what a
    crashed or never-started arm looks like, and a gate that only inspects the
    arms it happens to find would wave it through. So missing evidence fails the
    same way bad evidence does.
    """
    missing = sorted(expected_arms - set(rates))
    if missing:
        raise SystemExit(
            f"no per-step evidence for arm(s) {missing}; cannot verify the pure-cache "
            "premise. Absent rows mean the arm never ran, crashed before its stage "
            "flush, or wrote elsewhere -- all of which invalidate its success rate."
        )
    offenders = {a: r for a, r in rates.items() if a in expected_arms and r < minimum}
    if offenders:
        raise SystemExit(
            "pure-cache premise violated -- arms served non-FULL_HIT steps, which means "
            f"some task had no library entries and fell back to the teacher: {offenders}"
        )


def full_hit_rates(per_step_path: pathlib.Path) -> dict[str, float]:
    """Per-arm FULL_HIT fraction from the per-step sink."""
    totals: dict[str, list[int]] = {}
    if not per_step_path.exists():
        return {}
    with per_step_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec.get("task_uid", "")
            arm = rec.get("yaml_id") or (uid.split(":", 1)[0] if uid else None)
            hit = rec.get("hit_type")
            if arm is None or hit is None:
                continue
            slot = totals.setdefault(arm, [0, 0])
            slot[1] += 1
            if hit == "FULL_HIT":
                slot[0] += 1
    return {arm: (h / n if n else 0.0) for arm, (h, n) in totals.items()}


def assert_accepted_full_hit(journal_path: pathlib.Path, per_step_path: pathlib.Path,
                             expected_arms: set[str]) -> dict:
    """Runner-side twin of the analyzer's per-episode FULL_HIT gate.

    Deliberately the *same* functions, not a parallel implementation: a runner
    gate that admits what the analyzer later rejects is worse than no gate, since
    the episodes are already spent by then.
    """
    from exp.ablation_study.cache_size.full_hit import (
        assert_full_hit_per_episode,
        load_per_episode_hits,
    )
    from exp.common.conductor_journal import load_accepted

    arms = load_accepted([journal_path])
    per_ep, _ = load_per_episode_hits(per_step_path, require_attempt=True)
    summary = {}
    for arm in sorted(expected_arms):
        if arm not in arms:
            raise SystemExit(
                f"journal has no accepted episodes for arm {arm!r}; the arm never ran "
                "or crashed before its first terminal record"
            )
        attempts = {uid: r.attempt for uid, r in arms[arm].items()}
        summary[arm] = assert_full_hit_per_episode(arm, attempts, per_ep.get(arm, {}))
    logger.info("per-episode FULL_HIT witness: %s", summary)
    return summary


def canonical_row(line: str) -> str:
    """Stable identity for a per-step JSONL row (key order independent)."""
    return json.dumps(json.loads(line), sort_keys=True)


def merge_snapshot(per_step_path: pathlib.Path, snapshot_path: pathlib.Path) -> int:
    """Fold a crash-time snapshot into the authoritative JSONL, deduplicating on
    canonical row identity, then remove the snapshot. Returns merged count.

    Dedup is canonical rather than byte-wise: the snapshot writer and the sink
    writer can serialize the same row with different key order, and a byte-wise
    check would fold such a row in twice. Retiring the snapshot here is what
    keeps a later run that reuses this output path from merging stale evidence.
    """
    if not snapshot_path.exists():
        return 0
    seen = set()
    if per_step_path.exists():
        with per_step_path.open(encoding="utf-8") as f:
            seen = {canonical_row(line) for line in f if line.strip()}
    merged = 0
    with per_step_path.open("a", encoding="utf-8") as out, \
            snapshot_path.open(encoding="utf-8") as snap:
        for line in snap:
            row = line.strip()
            if not row:
                continue
            key = canonical_row(row)
            if key not in seen:
                seen.add(key)  # dedup within the snapshot itself too
                out.write(row + "\n")
                merged += 1
    snapshot_path.unlink()
    return merged


def _snapshot_loop(driver, per_step_path: pathlib.Path, stop: threading.Event,
                   interval_s: float = 60.0) -> None:
    """Atomically dump the driver's in-memory rows so a crash cannot erase them."""
    import os

    snap = per_step_path.with_suffix(".snapshot.jsonl")
    while not stop.wait(interval_s):
        rows = list(driver.per_step_rows)
        tmp = snap.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, snap)


def journal_shortfall(journal_path: pathlib.Path, arms: set[str], expected: int) -> dict[str, int]:
    """Arms whose journal row count falls short of the planned episode count."""
    if not journal_path.exists():
        return {a: 0 for a in sorted(arms)}
    seen: dict[str, set[str]] = {}
    with journal_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") not in ("done", "failed"):
                continue
            seen.setdefault(rec["yaml_id"], set()).add(rec["task_uid"])
    return {a: len(seen.get(a, ())) for a in sorted(arms) if len(seen.get(a, ())) < expected}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-matrix", required=True)
    ap.add_argument("--task-suite", required=True)
    ap.add_argument("--servers", required=True, help="host:port[,host:port...]")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--server-workers", default="")
    ap.add_argument("--arms", default="", help="comma list; empty = all arms")
    ap.add_argument("--trials", type=int, default=TRIALS_PER_TASK)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step-out", required=True)
    ap.add_argument("--apool-record", default=None,
                    help="verify_apool.py output; binds the run to the frozen pool. "
                         "Required unless --smoke.")
    ap.add_argument("--smoke", action="store_true",
                    help="unbound subset run: skips the A-pool requirement. NEVER for "
                         "the reported experiment.")
    ap.add_argument("--min-full-hit", type=float, default=1.0,
                    help="post-run gate on each arm's FULL_HIT rate; only lowerable "
                         "under --smoke, since the frozen design requires 1.0")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--eval-concurrency", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--conda-env", default="")
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
    yaml_paths = validate_pure_cache_arms(rows)

    apool = load_apool_digest(args.apool_record, required=not args.smoke,
                              verify_contents=not args.smoke)
    if apool and apool["suite"] != args.task_suite:
        raise SystemExit(
            f"A-pool record is for {apool['suite']!r}, run is for {args.task_suite!r}"
        )

    per_step_path = pathlib.Path(args.per_step_out)
    per_step_path.parent.mkdir(parents=True, exist_ok=True)
    launch_record = {
        "suite": args.task_suite,
        "arms": sorted(yaml_paths),
        "trials_per_task": args.trials,
        "smoke": bool(args.smoke),
        "apool": apool,
    }
    pathlib.Path(str(per_step_path) + ".launch.json").write_text(
        json.dumps(launch_record, indent=2)
    )
    logger.info("launch bound to A-pool rollup %s",
                (apool or {}).get("rollup_sha256", "<unbound>"))

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
        worker_server_keys = [servers[i % len(servers)].key for i in range(args.workers)]
        server_capacities = None

    from examples.libero.episode_runner import default_client_factory

    per_step_lock = threading.Lock()

    # Resume-aware merge: fold any crash-time snapshot into the authoritative
    # JSONL before a resumed run can append past it. Journal resume skips
    # already-done episodes, so their step evidence would otherwise be lost for
    # good -- and by the gate above, an arm with no evidence must fail.
    merged = merge_snapshot(per_step_path, per_step_path.with_suffix(".snapshot.jsonl"))
    if merged:
        logger.info("merged %d snapshot rows into %s on resume", merged, per_step_path)

    def _per_step_writer(yaml_id: str, rows: list[dict]) -> None:
        # N4: the driver passes the stage's yaml_id here, not a task_uid; every
        # row already carries its own task_uid from the episode runner.
        with per_step_lock, per_step_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"yaml_id": yaml_id, **row}) + "\n")

    strategy = PureCacheEvalStrategy(args.task_suite, yaml_paths, args.trials)
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

    # The driver only drains per-step rows at stage completion, so an arm that
    # crashes mid-stage would lose all of its evidence. Snapshot periodically.
    snapshot_stop = threading.Event()
    snapshot_thread = threading.Thread(
        target=_snapshot_loop,
        args=(driver, per_step_path, snapshot_stop),
        daemon=True,
    )
    snapshot_thread.start()

    while driver.port is None:
        time.sleep(0.05)
    logger.info("driver pull port = %d", driver.port)

    # Point every worker at the frozen pool. Leaving this empty would let each
    # worker load whatever init states its own environment ships, which is what
    # makes an unbound run unattestable.
    init_states_dir = apool["apool_dir"] if apool else ""
    specs = [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=worker_server_keys[i],
            gpu_id=str(i % args.gpus),
            conda_env=args.conda_env,
            task_suite_name=args.task_suite,
            init_states_dir=init_states_dir,
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
        agent.stop()
        agent_thread.join(timeout=30)

    # Canonical merge + retire the snapshot, so a later run reusing this output
    # path cannot fold stale evidence into fresh results.
    merged_final = merge_snapshot(per_step_path, per_step_path.with_suffix(".snapshot.jsonl"))
    if merged_final:
        logger.info("merged %d trailing snapshot rows at exit", merged_final)

    min_full_hit = args.min_full_hit
    if not args.smoke and min_full_hit != 1.0:
        raise SystemExit(
            f"--min-full-hit={min_full_hit} is only permitted under --smoke; the frozen "
            "design requires every routed step to be a FULL_HIT"
        )
    rates = full_hit_rates(per_step_path)
    logger.info("FULL_HIT rates (arm-level, informational): %s", rates)
    # The arm-level ratio is usable only for smoke runs, which have no accepted
    # journal contract. In a formal run it includes stale attempts by design;
    # applying the 1.0 floor there would let a stale MISS reject a clean accepted
    # retry before the accepted-aware gate can filter it.
    if args.smoke:
        assert_full_hit(rates, set(yaml_paths), min_full_hit)
    else:
        # The formal gate is the same per-episode join the analyzer runs, on
        # (task_uid, accepted attempt), so the operator learns here rather than
        # hours later in analysis.
        assert_accepted_full_hit(pathlib.Path(args.journal), per_step_path, set(yaml_paths))

    # N1: a uid whose retries are exhausted is terminal in the scheduler but is
    # never journaled, so the loss would only surface hours later in analysis.
    # Read the journal back here and say so while the operator is still present.
    shortfall = journal_shortfall(pathlib.Path(args.journal), set(yaml_paths),
                                  NUM_TASKS * args.trials)
    if shortfall:
        raise SystemExit(
            f"journal is short for arm(s) {shortfall}; relaunch with the same journal "
            "to re-run the missing episodes (resume reads the journal, so they will "
            "be re-dispatched)"
        )


if __name__ == "__main__":
    main()
