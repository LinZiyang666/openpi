"""Drive the GR00T x LIBERO gate-threshold sweep through the conductor.

What this buys over ``orchestrate_search.py``
---------------------------------------------
The cell scheduler dispatches *cells*: one arm goes to one slot, which launches
its own sim clients against its own server. That is fine while there are more
arms than slots, but a phase holding a single arm then runs at 1/N of the pool.
Measured on the 2026-08-24 run: libero_spatial's warmup (16.0 min) and
gate-only (32.0 min) phases used one of six slots, together 48.0 of the suite's
190.5 minutes -- 25.2% of the wall clock spent with five sixths of the fleet
idle.

The conductor dispatches *episodes*, and ``sharding.shard_eval_stage`` puts one
sibling stage of each arm on every server, so a single-arm phase draws on every
worker. Dynamic dispatch also beats pre-cutting the pool into lanes: a lane
file fixes each worker's share up front, and at 48 lanes of ~10 episodes the
slowest lane runs 27% past an even split (measured -- see
``analysis/gate_pareto/shard_imbalance_probe.py``), whereas a worker that
finishes early simply pulls the next episode.

Topology
--------
Six ``serve_groot_libero.py --concurrent --allow-dynamic-bundles`` processes on
the serving box, the driver beside them, workers on the sim box:

    # serving box
    python -m exp.libero_groot.run_conductor --role driver \
        --servers h:23160,h:23161,... --driver-port 23190 --bind-host 0.0.0.0
    # sim box
    python -m exp.libero_groot.run_conductor --role agent \
        --driver-host <serving box> --driver-port 23190 --workers 48

Six independent endpoints rather than one ``--replicas 6`` endpoint: that is
what ``ServerEndpoint``'s deployment invariant prescribes, and it is what the
scheduler can balance over -- N endpoints are N units it can place work on,
while one routed endpoint is a single unit whose internal fan-out it cannot
see. The router's only added value here is stealing work across endpoints,
worth 2.3-6.0% on this line's measured episode-cost spread, against a
supervisor whose watchdog takes all six replicas down with any one of them.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import threading
import time

from openpi.conductor import ConductorDriver
from openpi.conductor.journal import Journal
from openpi.conductor import ServerEndpoint
from openpi.conductor import WorkerAgent
from openpi.conductor import WorkerSpec
from openpi.conductor import strategy as _strat
from openpi.conductor import task as _task
from openpi.conductor.sharding import shard_eval_stage


logger = logging.getLogger("groot_conductor")


# ----------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------


class GrootSweepStrategy(_strat.ExperimentStrategy):
    """One arm -> one sibling eval stage per server; the bundle is hot-swapped.

    ``server_assignment`` is ignored on purpose: it maps each yaml to exactly
    one endpoint, which is the placement this strategy exists to widen. The
    full endpoint list comes in through the constructor instead, because
    ``plan`` is never handed one.
    """

    def __init__(
        self,
        *,
        task_suite: str,
        yaml_paths: dict[str, str],
        all_arms: list[str],
        servers: list[ServerEndpoint],
        trials: int,
        num_tasks: int,
        done_uids: set[str] | None = None,
    ) -> None:
        self._task_suite = task_suite
        self._yaml_paths = yaml_paths
        self._servers = servers
        self._trials = trials
        self._num_tasks = num_tasks
        self._done_uids = set(done_uids or ())
        from examples.libero.collect_util import compute_global_episode_id

        #: uid -> the identity fields the analysis joins on. The wire result
        #: carries none of them (``EpisodeResult`` is uid + outcome) and the
        #: journal only echoes uid/yaml_id, so this map is the only place the
        #: run can recover ``task_id`` / ``orig_init_state_idx`` / ``episode_id``.
        #:
        #: Built over **every** arm, not just the ones with work left: on a
        #: resume the finished arms are exactly the ones whose artifacts a
        #: crashed predecessor never got to write, so dropping them here would
        #: make "resume until it passes" impossible.
        self.episode_index: dict[str, dict] = {
            _task.make_task_uid(arm, "eval", task_id, ep_idx): {
                "task_id": task_id,
                "init_state_idx": ep_idx,
                "orig_init_state_idx": ep_idx,
                # The canonical global id, from the one helper both the serial
                # and concurrent paths use. Deriving it any other way would
                # silently fail to join against the arms the cell scheduler has
                # already produced.
                "episode_id": compute_global_episode_id(task_id, ep_idx, trials),
            }
            for arm in all_arms
            for task_id in range(num_tasks)
            for ep_idx in range(trials)
        }

    def plan(self, yamls, server_assignment) -> _task.TaskGraph:  # noqa: ARG002
        graph = _task.TaskGraph()
        for yaml_id in yamls:
            episodes = [
                _task.EpisodeTask(
                    task_uid=uid,
                    yaml_id=yaml_id,
                    phase="eval",
                    experiment=self._task_suite,
                    task_id=task_id,
                    episode_idx=ep_idx,
                    orig_init_state_idx=ep_idx,
                    # Rewritten per shard by shard_eval_stage.
                    server_host=self._servers[0].host,
                    server_port=self._servers[0].port,
                    bundle_id=yaml_id,
                    # Producer contract (task.py): the runner derives the global
                    # episode id from this and fails fast without it.
                    extra={"num_trials_per_task": self._trials},
                )
                for task_id in range(self._num_tasks)
                for ep_idx in range(self._trials)
                # Episodes already in the journal are dropped from the plan, not
                # merely skipped by the scheduler. The scheduler's own resume
                # empties ``pending`` but leaves ``Stage.episodes`` full, and a
                # strategy can only see the latter -- so without this a sibling
                # with nothing left to run would still look non-empty at
                # ``on_stage_begin`` and pay a full bundle reload for no work.
                if (uid := _task.make_task_uid(yaml_id, "eval", task_id, ep_idx))
                not in self._done_uids
            ]
            for stage in shard_eval_stage(
                stage_id=f"eval__{yaml_id}",
                yaml_id=yaml_id,
                episodes=episodes,
                servers=self._servers,
                # Gate-Pareto eval episodes are pure rollouts: they write only
                # their own result + per-step rows, both keyed by episode id.
                episodes_are_idempotent=True,
                setup={"yaml_path": self._yaml_paths[yaml_id]},
            ):
                graph.add_stage(stage)
        graph.validate()
        return graph

    def on_stage_begin(self, stage, ctl, ctx) -> None:  # noqa: ARG002
        # An empty sibling still reaches this hook: the driver runs setup first
        # and only then learns there is nothing to dispatch. A bundle load is a
        # teardown and a gigabyte-scale rebuild, so paying it for zero episodes
        # is exactly the burst that put the GPU into an MMU fault on the pi0.5
        # line. Two ways a sibling ends up empty, and both are covered because
        # ``plan`` drops journalled episodes outright: fewer episodes than
        # servers, and a resume where this shard's work is already done.
        if not stage.episodes:
            return
        yaml_path = stage.setup["yaml_path"]
        ctl.load_cache_config(
            yaml_content=pathlib.Path(yaml_path).read_text(encoding="utf-8"),
            yaml_id=stage.yaml_id,
            # Never "default": with hot-swap enabled that slot is writable, so
            # binding under it would make every later connection's provenance
            # depend on load order.
            bundle_id=stage.yaml_id,
        )
        logger.info("%s: bundle loaded from %s", stage.stage_id, yaml_path)


# ----------------------------------------------------------------------
# Resume: never walk a finished arm's stages
# ----------------------------------------------------------------------


def arms_with_work_left(
    journal_path: str | pathlib.Path, arms: list[str], *, expected: int
) -> tuple[list[str], dict[str, int]]:
    """Drop arms whose episodes are already all in the journal.

    Resume is episode-level, so a completed arm contributes no work -- but the
    driver still walks its stages, and every stage walked calls
    ``ctl.load_cache_config``. Late in a sweep that is a burst of back-to-back
    bundle swaps, each tearing down and rebuilding a gigabyte-scale library
    while other connections are serving. Measured on 2026-08-20 (pi0.5 line):
    23 swaps inside a few seconds put the GPU into an MMU fault (Xid 31) within
    a minute, twice, on a run that had been stable for four and a half hours at
    one swap per eight minutes.

    Sharding multiplies the exposure -- an arm now owns one stage per server --
    so the filter matters more here than it did there, not less.

    Counting is over distinct ``task_uid`` so a retried episode is not
    double-counted into a false "complete".
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
            if not Journal.record_counts_as_done(rec):
                # A fenced stale result is not completed work. Counting it is
                # how an arm gets filtered out of a resume while one of its
                # episodes has never actually run to an accepted result.
                continue
            uid = rec.get("task_uid", "")
            arm = rec.get("yaml_id") or (uid.split(":", 1)[0] if uid else None)
            if arm and uid:
                seen.setdefault(arm, set()).add(uid)

    counts = {arm: len(seen.get(arm, ())) for arm in arms}
    remaining = [arm for arm in arms if counts[arm] < expected]
    return remaining, counts


# ----------------------------------------------------------------------
# Evidence: per-step rows, results, and the merge sidecar
# ----------------------------------------------------------------------


def keep_accepted_rows(rows: list[dict], outcomes: dict[str, dict]) -> list[dict]:
    """Keep only the rows belonging to each episode's *accepted* dispatch.

    Three ways one episode ends up with more than one set of rows, and taking
    the highest ``attempt`` resolves only the first:

      * a retried episode reports once per attempt at increasing generations;
      * a **fenced** dispatch reports at the *same* generation as the accepted
        one -- the scheduler distinguishes them, the generation number does not,
        so max-attempt keeps both and the integrity gate then rejects the arm
        for duplicate ``(episode_id, step_idx)``;
      * a resumed run restarts generations at 1, so rows written before a crash
        tie *exactly* with rows written after it -- same ``accepted``, same
        ``attempt`` -- which is why the driver stamps a per-run id on both the
        rows and the journal line, and that is what separates them here.

    The journal's accepted outcome is the only record of which report the run
    actually used, so it is what selects here. Rows the driver stamped
    ``accepted: false`` are dropped outright; among what remains, the attempt
    named by the accepted outcome wins, and rows for an episode with no accepted
    outcome are dropped -- that episode did not complete, so its rows would
    describe an episode absent from the results side and fail I3.
    """
    kept: list[dict] = []
    for row in rows:
        uid = row.get("task_uid")
        if uid is None:
            kept.append(row)  # not an episode row; nothing to select on
            continue
        if row.get("accepted") is False:
            continue
        rec = outcomes.get(uid)
        if rec is None:
            continue
        # ``run_id`` first: after a crash the dispatch counter restarts at 1, so
        # rows written before the crash and rows from the re-run carry the same
        # (accepted, attempt) and only the producing run tells them apart. Fall
        # back to attempt when either side predates the field.
        want_run = rec.get("run_id")
        row_run = row.get("run_id")
        if want_run is not None and row_run is not None:
            if row_run != want_run:
                continue
            kept.append(row)
            continue
        want_attempt = rec.get("attempt")
        if want_attempt is None or int(row.get("attempt", 1)) == int(want_attempt):
            kept.append(row)
    return kept


def read_journal_outcomes(journal_path: pathlib.Path) -> dict[str, dict]:
    """uid -> the record describing its *accepted* dispatch.

    Two terminal records can share a uid. A timed-out episode is re-dispatched
    at a higher generation, and when the original worker finally reports, the
    scheduler fences that result -- but the driver journals it anyway, marked
    ``accepted: false``, precisely so an offline reader can tell the two apart.
    Taking the last line instead would let a stale *fatal* error overwrite the
    retry that actually succeeded, and the arm would report one fewer success
    than it earned, with nothing anywhere disagreeing.

    Records that predate the ``accepted`` field carry neither value; ties fall
    back to the highest attempt, which is the live dispatch by construction.
    """
    best: dict[str, dict] = {}
    if not journal_path.exists():
        return best
    with journal_path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("task_uid") or not Journal.record_counts_as_done(rec):
                continue
            uid = rec["task_uid"]
            prev = best.get(uid)
            if prev is None or int(rec.get("attempt", 1)) >= int(prev.get("attempt", 1)):
                best[uid] = rec
    return best


def write_arm_artifacts(
    *,
    results_dir: pathlib.Path,
    per_step_dir: pathlib.Path,
    arm: str,
    planned_uids: list[str],
    outcomes: dict[str, dict],
    episode_index: dict[str, dict],
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ``<arm>.json`` and ``<arm>.merge.json`` for the analysis gate.

    The results rows carry the identity fields the wire result does not, joined
    back from the plan.

    The sidecar's two counts are deliberately drawn from *different* places:
    ``episodes_expected`` from the plan, ``episodes_reported`` from the journal.
    That is what makes it evidence rather than a tautology -- an episode whose
    retries are exhausted never receives a terminal journal record, so the two
    genuinely disagree exactly when something was lost. A count derived from the
    results file would agree with the results file by construction, which is the
    self-confirming sidecar this gate exists to refuse.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    per_step_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid in planned_uids:
        rec = outcomes.get(uid)
        if rec is None:
            continue
        ident = episode_index[uid]
        rows.append({**ident, "seed": 7, "success": bool(rec.get("success"))})
    results_path = results_dir / f"{arm}.json"
    results_path.write_text(json.dumps(rows), encoding="utf-8")

    # Beside the per-step file, not beside the results: ``aggregate`` globs the
    # results directory for ``*.json`` and treats every stem it does not
    # recognise as an unexpected arm, so a sidecar left there fails the phase
    # under the name ``<arm>.merge``.
    sidecar = per_step_dir / f"{arm}.merge.json"
    sidecar.write_text(
        json.dumps(
            {
                # The per-worker file that I6 watches for does not exist on this
                # transport: rows ride back with their episode's result over the
                # same connection, so "a worker's file never appeared" is not a
                # reachable failure. Say so rather than fabricating lane counts
                # that would always agree.
                "transport": "tcp",
                "episodes_expected": len(planned_uids),
                "episodes_reported": sum(1 for uid in planned_uids if uid in outcomes),
            }
        ),
        encoding="utf-8",
    )
    return results_path, sidecar


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def _groot_specs(
    *, n: int, server_keys: list[str], gpus: int, conda_env: str, task_suite: str,
    init_states_dir: str,
) -> list[WorkerSpec]:
    return [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=server_keys[i],
            gpu_id=str(i % gpus),
            conda_env=conda_env,
            task_suite_name=task_suite,
            init_states_dir=init_states_dir,
            # A GR00T checkpoint needs the raw render; the 224 default would
            # crop twice and the wire contract rejects the frame -- after the
            # whole fleet is already up.
            resize_size=256,
            replan_steps=5,
            # CUDA_VISIBLE_DEVICES steers the policy client; EGL picks its
            # render device separately, and unset it lands every worker on GPU 0.
            env={"MUJOCO_EGL_DEVICE_ID": str(i % gpus)},
        )
        for i in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="all", choices=("driver", "agent", "all"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--yaml-dir", default="")
    ap.add_argument("--servers", default="", help="host:port,host:port,... (driver role)")
    ap.add_argument("--journal", default="", help="episode ledger path (driver role)")
    ap.add_argument(
        "--results-dir",
        default="",
        help="where <arm>.json lands; the first positional of the analyzer's "
        "aggregate. Keep it free of anything else ending in .json.",
    )
    ap.add_argument(
        "--per-step-dir",
        default="",
        help="where <arm>.jsonl and <arm>.merge.json land; the analyzer's "
        "second positional.",
    )
    ap.add_argument("--bind-host", default="0.0.0.0")
    ap.add_argument(
        "--driver-port",
        type=int,
        default=0,
        help="driver role: the pull port to bind (fixed, not ephemeral, so the "
        "agent on another host can be pointed at it). agent role: where to connect.",
    )
    ap.add_argument("--driver-host", default="", help="agent role: the driver's host")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--conda-env", default="")
    ap.add_argument("--init-states-dir", default="")
    ap.add_argument("--eval-concurrency", type=int, default=0)
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    # Defaults come from the experiment's binding table, resolved lazily: the
    # agent role dispatches workers and has no use for it, and requiring it
    # would put an experiment-specific file on every sim box.
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--num-tasks", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


    if not args.servers:
        raise SystemExit("--servers is required: it is how workers are bound to endpoints")
    servers = []
    for spec in args.servers.split(","):
        if ":" not in spec:
            raise SystemExit(f"--servers entry {spec!r} must be host:port")
        host, port = spec.rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))

    if args.role == "agent":
        # Workers are strictly affine: one only ever receives episodes whose
        # stage sits on its bound endpoint. The round-robin below must therefore
        # match what the driver assumes, which is why --servers is required on
        # both sides rather than negotiated.
        if not (args.driver_host and args.driver_port):
            raise SystemExit("--role agent requires --driver-host and --driver-port")
        server_keys = [servers[i % len(servers)].key for i in range(args.workers)]
        agent = WorkerAgent(
            _groot_specs(
                n=args.workers,
                server_keys=server_keys,
                gpus=args.gpus,
                conda_env=args.conda_env,
                task_suite=args.suite,
                init_states_dir=args.init_states_dir,
            ),
            driver_host=args.driver_host,
            driver_port=args.driver_port,
        )
        logger.info(
            "agent: %d workers -> %s (driver %s:%d)",
            args.workers,
            [s.key for s in servers],
            args.driver_host,
            args.driver_port,
        )
        agent.run()
        return

    # Resolved here, past the agent-role return: only the driver plans episodes,
    # so a sim box never needs the experiment's binding table on disk.
    if args.trials is None or args.num_tasks is None:
        from exp.libero_groot import gate_pareto_bindings as bindings

        if args.trials is None:
            args.trials = bindings.APOOL_TRIALS
        if args.num_tasks is None:
            args.num_tasks = bindings.NUM_TASKS

    yaml_dir = pathlib.Path(args.yaml_dir)
    yaml_paths = {p.stem: str(p) for p in sorted(yaml_dir.glob("*.yaml"))}
    if not yaml_paths:
        raise SystemExit(f"no arm recipes under {yaml_dir}")

    journal = pathlib.Path(args.journal)
    expected = args.num_tasks * args.trials
    remaining, counts = arms_with_work_left(journal, sorted(yaml_paths), expected=expected)
    if len(remaining) != len(yaml_paths):
        logger.info(
            "resume: %d/%d arms already complete, not walking their stages (%s)",
            len(yaml_paths) - len(remaining),
            len(yaml_paths),
            {a: counts[a] for a in sorted(counts) if counts[a] >= expected},
        )
    active_paths = {arm: yaml_paths[arm] for arm in remaining}

    results_dir = pathlib.Path(args.results_dir)
    per_step_dir = pathlib.Path(args.per_step_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    per_step_dir.mkdir(parents=True, exist_ok=True)

    done_uids = set(read_journal_outcomes(journal))
    strategy = GrootSweepStrategy(
        task_suite=args.suite,
        yaml_paths=active_paths,
        # Every arm, so a resume can still write the artifacts a crashed
        # predecessor never got to.
        all_arms=sorted(yaml_paths),
        servers=servers,
        trials=args.trials,
        num_tasks=args.num_tasks,
        done_uids=done_uids,
    )

    def finalize() -> None:
        """Deduplicate the per-arm evidence and write what the analyzer reads.

        Runs on every exit path, including "nothing left to run" -- that is
        exactly the state a crashed run resumes into, and its artifacts are the
        ones that were never written.
        """
        outcomes = read_journal_outcomes(journal)
        by_arm: dict[str, list[str]] = collections.defaultdict(list)
        for uid in strategy.episode_index:
            by_arm[uid.split(":", 1)[0]].append(uid)
        for arm, uids in sorted(by_arm.items()):
            path = per_step_dir / f"{arm}.jsonl"
            if path.exists():
                # Retries wrote a second full set of rows for one episode, and
                # the integrity gate rejects duplicate (episode_id, step_idx).
                # Deduplicating only now, rather than at append time, keeps the
                # hot path a plain append -- which is what makes a crash
                # survivable.
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                kept = keep_accepted_rows(rows, outcomes)
                if len(kept) != len(rows):
                    path.write_text(
                        "".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8"
                    )
                    logger.info(
                        "%s: dropped %d superseded per-step rows", arm, len(rows) - len(kept)
                    )
            write_arm_artifacts(
                results_dir=results_dir,
                per_step_dir=per_step_dir,
                arm=arm,
                planned_uids=sorted(uids),
                outcomes=outcomes,
                episode_index=strategy.episode_index,
            )
        logger.info("wrote results + merge sidecars for %d arms", len(by_arm))

    if not active_paths:
        # Not a no-op: the artifacts are what a resume is for.
        logger.info("every arm is complete; writing artifacts and exiting")
        finalize()
        return

    per_step_lock = threading.Lock()

    def _per_step_writer(yaml_id: str, rows: list[dict]) -> None:
        # Appended immediately, per arm, rather than accumulated in memory. The
        # journal is written per episode, so a crash-and-resume replays only the
        # *missing* episodes -- any rows still buffered for the ones already
        # journalled would be gone for good, and the integrity gate requires the
        # per-step episode set to equal the results episode set exactly.
        #
        # One file per arm because that is what the analyzer reads; the driver
        # calls this once per sibling stage, each with only its own shard's rows.
        with per_step_lock, (per_step_dir / f"{yaml_id}.jsonl").open(
            "a", encoding="utf-8"
        ) as fh:
            for row in rows:
                fh.write(json.dumps({"yaml_id": yaml_id, **row}) + "\n")

    from examples.libero.episode_runner import default_client_factory

    driver = ConductorDriver(
        strategy,
        yaml_weights={arm: expected for arm in active_paths},
        servers=servers,
        journal_path=str(journal),
        ctl_factory=default_client_factory,
        episode_timeout_s=args.episode_timeout_s,
        bind_host=args.bind_host,
        bind_port=args.driver_port,
        scheduler_kwargs=(
            {"eval_concurrency": args.eval_concurrency} if args.eval_concurrency else None
        ),
        per_step_writer=_per_step_writer,
    )

    # Warn once per unknown server key. Worker affinity is a *string* match on
    # "host:port", so a driver started with 127.0.0.1 and an agent pointed at the
    # public hostname describe the same processes and never agree: next_task
    # returns None for every pull, the worker backs off and retries, and the run
    # idles indefinitely without printing anything. Measured cost of learning
    # this the hard way: twelve minutes of a silent four-worker fleet.
    _known_keys = {s.key for s in servers}
    _warned: set[str] = set()
    _inner_pull = driver.handle_pull

    def _pull_with_affinity_warning(server_key: str):
        key = server_key
        if key is not None and key not in _known_keys and key not in _warned:
            _warned.add(key)
            logger.error(
                "worker reports server_key=%r, which is not among the driver's "
                "endpoints %s. Worker affinity is an exact string match, so these "
                "workers will never be given an episode. Pass the *same* --servers "
                "string to both roles.",
                key,
                sorted(_known_keys),
            )
        return _inner_pull(server_key)

    driver.handle_pull = _pull_with_affinity_warning

    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    # Guarded on the thread, not just the port: --driver-port is fixed, so
    # EADDRINUSE (a leftover driver, or the port block taken by another session)
    # is the likeliest startup failure -- and it kills the thread while leaving
    # ``port`` None, which an unguarded wait would spin on forever.
    while driver.port is None and driver_thread.is_alive():
        time.sleep(0.05)
    if driver.port is None:
        raise SystemExit(
            f"driver thread exited before binding {args.bind_host}:{args.driver_port} "
            "(see the traceback above; a stale driver holding the port is the usual cause)"
        )
    logger.info("driver pull port = %d", driver.port)

    agent = None
    agent_thread = None
    if args.role == "all":
        server_keys = [servers[i % len(servers)].key for i in range(args.workers)]
        specs = _groot_specs(
            n=args.workers,
            server_keys=server_keys,
            gpus=args.gpus,
            conda_env=args.conda_env,
            task_suite=args.suite,
            init_states_dir=args.init_states_dir,
        )
        # Loopback, not --bind-host: the listener binds 0.0.0.0 so remote agents
        # can reach it, but handing 0.0.0.0 to a local connect() only works
        # because Linux happens to map INADDR_ANY to loopback.
        agent = WorkerAgent(specs, driver_host="127.0.0.1", driver_port=driver.port)
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()

    try:
        driver_thread.join()
    finally:
        # Rows a stage never flushed -- a sibling that ended FAILED never runs
        # _complete_stage, so its rows are still sitting in the driver.
        stranded = collections.defaultdict(list)
        for row in list(driver.per_step_rows):
            arm = row.get("yaml_id") or (row.get("task_uid", "").split(":", 1)[0])
            if arm:
                stranded[arm].append(row)
        for arm, rows in stranded.items():
            with (per_step_dir / f"{arm}.jsonl").open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps({"yaml_id": arm, **row}) + "\n")
        if stranded:
            logger.info(
                "recovered %d unflushed per-step rows across %d arm(s)",
                sum(len(v) for v in stranded.values()),
                len(stranded),
            )
        finalize()

        if agent is not None:
            agent.stop()
            if agent_thread is not None:
                agent_thread.join(timeout=30)


if __name__ == "__main__":
    main()
