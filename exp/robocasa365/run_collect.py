"""Driver/agent entry for RoboCasa365 teacher-library collection over the conductor.

One invocation = ONE teacher (frozen teacher<->server affinity): core
``assign_servers`` greedily balances every yaml over every endpoint with no
model-type notion, so mixing teachers in one graph could dispatch pi0.5
episodes to a GR00T server. ``--teacher`` is mandatory and every ``--servers``
endpoint must belong to that teacher's group in the env-config file — checked
BEFORE the graph is built. The endpoint check validates configuration grouping
only; it cannot prove which model actually listens on a port. That mapping is
an operator invariant: whoever launches the servers from the env-config is
responsible, and the manual gates archive the raw server metadata + launch
command + checkpoint sha as provenance.

Collection topology (frozen D-L): one ``serve_policy.py --collect
--non-concurrent`` process <-> one connection <-> one worker, horizontally
scaled by launching N server processes (``server_capacities={key: 1}``).
The injected ``_NoOpCtl`` keeps the driver's control connection from consuming
the single-connection server's only slot. EVAL runs (with ``--cache_config``,
concurrent) must switch back to a real ctl factory
(``examples.libero.episode_runner.default_client_factory``) — the no-op ctl is
correct for collection ONLY.

Before dispatch the exact TaskGraph is persisted as an immutable per-batch
run-plan JSON (params + every expected task_uid + output prefixes +
``plan_hash``). It is the ONLY executable source of the expected-UID set for
``verify_collection_artifacts.py`` (``--run-plan``); resume recomputes the
hash and refuses to start on a mismatch so changed parameters cannot silently
continue an old journal.

Roles (driver and agents may live on different machines)::

    --role driver   build graph + run-plan, serve the pull port, write journal
    --role agent    fork + supervise this machine's workers
    --role all      single-machine convenience: driver thread + agent
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any

# Run as a plain script (`python exp/robocasa365/run_collect.py`, the form the
# collection guide documents) and sys.path[0] is this file's directory, so the
# first-party `exp.` import below cannot resolve. Run as a module or imported by
# run_ws_search it resolves fine. Put the repo root on the path so both callers
# keep working -- the alternative is a silent ModuleNotFoundError for whoever
# still uses the documented command.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from openpi.conductor import task as _task
from openpi.conductor.agent import WorkerAgent, WorkerSpec
from exp.robocasa365.pinned_objects import (
    compute_pin_task_id,
    load_pin_manifest,
    resolve_manifest_path,
)
from openpi.conductor.driver import ConductorDriver
from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.task import EpisodeTask, ServerEndpoint, Stage, TaskGraph, make_task_uid

TEACHERS = ("pi05", "groot_tp")

# ------------------------------------------------------------------
# Identity formulas (frozen §4.3.2)
# ------------------------------------------------------------------


def build_run_id(layout: int, style: int, teacher: str) -> str:
    return f"collect_l{layout}s{style}_{teacher}"


def build_yaml_id(run_id: str, task_name: str) -> str:
    return f"{run_id}__{task_name}"


def output_prefix(teacher: str, task_name: str, episode_idx: int) -> str:
    """Relative artifact prefix (attempt suffix + .h5 appended by the auditor)."""
    return f"{teacher}/{task_name}/episode_{episode_idx:04d}"


def canonical_collect_root(raw: str) -> str:
    """Canonical spelling of the collection root for the run-plan hash payload.

    ABSOLUTE paths only: a relative root hashes to the same string across runs
    while actually pointing wherever the server's cwd happens to be — the exact
    "one journal silently split across two roots" failure the hash exists to
    prevent. Beyond that, pure-lexical normalization only (normpath +
    trailing-slash strip): the root lives on the SERVER machine, so resolving
    symlinks locally on the driver would canonicalize against the wrong
    filesystem.
    """
    text = str(raw)
    if not os.path.isabs(text):
        raise ValueError(
            f"collect root must be an absolute path, got {text!r}: a relative "
            "root keeps the same hash while resolving against whatever cwd the "
            "server has, splitting one journal across multiple output roots"
        )
    return os.path.normpath(text).rstrip("/") or "/"


# ------------------------------------------------------------------
# Strategy
# ------------------------------------------------------------------


# First seed of the evaluation segment. Collection uses 0.. and eval 1_000_000..
# so the two never share an initial state (analysis/robocasa365_seed_anatomy.md).
EVAL_SEED_BASE = 1_000_000


class RobocasaCollectStrategy(ExperimentStrategy):
    """Static collection plan: one eval stage per task, no calibration, no deps.

    ``tasks`` is an ordered list of ``(task_name, n_episodes)``; the position is
    the ``task_id``, so the ordering is part of the identity and must stay
    stable across resumes (pinned by the run-plan hash).
    """

    def __init__(
        self,
        *,
        teacher: str,
        layout: int,
        style: int,
        base_seed: int,
        replan_steps: int,
        tasks: list[tuple[str, int]],
        batch: int = 1,
        episode_lo: dict[str, int] | None = None,
        pin_id: str | None = None,
        pinned_objects: dict[str, dict[str, str]] | None = None,
    ) -> None:
        if teacher not in TEACHERS:
            raise ValueError(f"unknown teacher {teacher!r}; expected one of {TEACHERS}")
        self._teacher = teacher
        self._layout = int(layout)
        self._style = int(style)
        self._base_seed = int(base_seed)
        self._replan_steps = int(replan_steps)
        self._tasks = list(tasks)
        self._batch = int(batch)
        # Extension batches continue the episode range where the prior batch
        # ended (seeds stay contiguous and non-overlapping, §4.3.6-(4)).
        self._episode_lo = dict(episode_lo or {})
        self._pin_id = pin_id
        self._pinned_objects = pinned_objects
        if (pin_id is None) != (pinned_objects is None):
            raise ValueError("pin_id and pinned_objects must be supplied together")
        if pinned_objects is not None:
            unknown = sorted({name for name, _ in self._tasks} - set(pinned_objects))
            if unknown:
                raise ValueError(
                    f"pin table has no slot map for {unknown}; a task dispatched "
                    "without pins would silently run random objects"
                )
        # The eval segment starts at 1_000_000 (analysis/robocasa365_seed_anatomy.md).
        # Nothing enforced that until now: a large enough extension batch would
        # walk collection seeds straight into eval's range and the two would
        # share initial states with no error anywhere. The guard applies only to
        # runs that START in the collection segment -- an eval run's base_seed
        # IS 1_000_000 by design, so testing it against the same bound would
        # reject every eval strategy.
        highest = max(
            (int(self._episode_lo.get(name, 0)) + int(n) - 1 for name, n in self._tasks),
            default=0,
        )
        if self._base_seed < EVAL_SEED_BASE and self._base_seed + highest >= EVAL_SEED_BASE:
            raise ValueError(
                f"seed {self._base_seed + highest} reaches the eval segment "
                f"(base {EVAL_SEED_BASE}); collection and eval must not share seeds"
            )
        self.run_id = build_run_id(self._layout, self._style, teacher)

    @property
    def yaml_ids(self) -> list[str]:
        return [build_yaml_id(self.run_id, name) for name, _ in self._tasks]

    def plan(
        self,
        yamls: list[str],
        server_assignment: dict[str, _task.ServerEndpoint],
    ) -> TaskGraph:
        del yamls  # identity comes from the ordered task list, not the sorted arg
        graph = TaskGraph()
        for task_id, (task_name, n_episodes) in enumerate(self._tasks):
            yaml_id = build_yaml_id(self.run_id, task_name)
            server = server_assignment[yaml_id]
            lo = int(self._episode_lo.get(task_name, 0))
            episodes = []
            for offset in range(int(n_episodes)):
                episode_idx = lo + offset
                episodes.append(
                    EpisodeTask(
                        task_uid=make_task_uid(yaml_id, "eval", task_id, episode_idx),
                        yaml_id=yaml_id,
                        phase="eval",
                        experiment=self._teacher,  # collector directory level; must not contain "/"
                        task_id=task_id,
                        episode_idx=episode_idx,
                        # RoboCasa has no init pool: this field carries the seed
                        # offset (seed = base_seed + orig_init_state_idx).
                        orig_init_state_idx=episode_idx,
                        server_host=server.host,
                        server_port=server.port,
                        bundle_id="default",  # the only id a bare collection server acks
                        extra={
                            "task_name": task_name,
                            "layout": self._layout,
                            "style": self._style,
                            "teacher": self._teacher,
                            "base_seed": self._base_seed,
                            "replan_steps": self._replan_steps,
                            "batch": self._batch,
                            # The pin payload travels WITH the task: a hash
                            # cannot build an environment, only the slot map
                            # can. Absent for unpinned runs so their wire
                            # format is unchanged.
                            **(
                                {}
                                if self._pinned_objects is None
                                else {
                                    "pin_id": self._pin_id,
                                    "pin_task_id": compute_pin_task_id(
                                        task_name, self._pinned_objects[task_name]
                                    ),
                                    "pinned_objects": self._pinned_objects[task_name],
                                }
                            ),
                        },
                    )
                )
            graph.add_stage(
                Stage(stage_id=yaml_id, yaml_id=yaml_id, phase="eval", server=server, episodes=episodes)
            )
        graph.validate()
        return graph


class _NoOpCtl:
    """Socket-free stand-in for the driver's per-server control connection.

    Collection stages are eval-phase with no calibration, so the driver's stage
    lifecycle only ever needs these no-ops. Correct for collection ONLY — eval
    runs need the real ctl (load_cache_config / select_bundle / preload).
    """

    def unload_warmup_buffer(self, *_args: Any, **_kwargs: Any) -> dict:
        return {}

    def close(self) -> None:
        return None


# ------------------------------------------------------------------
# Env-config + teacher<->endpoint affinity
# ------------------------------------------------------------------


def load_env_config(path: str | pathlib.Path) -> dict[str, str]:
    """KEY=VALUE lines; '#' comments and blank lines ignored."""
    values: dict[str, str] = {}
    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"malformed env-config line (expected KEY=VALUE): {raw!r}")
        values[key.strip()] = value.strip()
    return values


def teacher_endpoint_group(env_config: dict[str, str], teacher: str) -> set[str]:
    key = f"{teacher.upper()}_SERVERS"
    raw = env_config.get(key, "")
    return {spec.strip() for spec in raw.split(",") if spec.strip()}


def validate_teacher_endpoints(teacher: str, servers: list[ServerEndpoint], env_config: dict[str, str]) -> None:
    """Refuse BEFORE graph construction if any endpoint is outside the teacher's group."""
    group = teacher_endpoint_group(env_config, teacher)
    if not group:
        raise SystemExit(f"env-config declares no {teacher.upper()}_SERVERS group; refusing to guess")
    stray = [s.key for s in servers if s.key not in group]
    if stray:
        raise SystemExit(
            f"servers {stray} are not in the {teacher!r} endpoint group {sorted(group)}; "
            "one driver invocation serves exactly one teacher's homogeneous endpoints"
        )


# ------------------------------------------------------------------
# Run-plan artifact (frozen §4.3.6-(6))
# ------------------------------------------------------------------


def compute_plan_hash(payload: dict[str, Any]) -> str:
    """sha256 over canonical JSON with the ``plan_hash`` field itself excluded."""
    body = {key: value for key, value in payload.items() if key != "plan_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_run_plan(strategy: RobocasaCollectStrategy, graph: TaskGraph, collect_root: str) -> dict[str, Any]:
    """Serialize the ACTUAL dispatch graph — no second UID formula anywhere."""
    uids: list[str] = []
    prefixes: dict[str, str] = {}
    for yaml_id in strategy.yaml_ids:  # deterministic order = the task-list order
        stage = graph.stages[yaml_id]
        for episode in stage.episodes:
            uids.append(episode.task_uid)
            prefixes[episode.task_uid] = output_prefix(
                episode.extra["teacher"], episode.extra["task_name"], episode.episode_idx
            )
    payload: dict[str, Any] = {
        "params": {
            "teacher": strategy._teacher,  # noqa: SLF001 - own module
            "layout": strategy._layout,  # noqa: SLF001
            "style": strategy._style,  # noqa: SLF001
            "base_seed": strategy._base_seed,  # noqa: SLF001
            "replan_steps": strategy._replan_steps,  # noqa: SLF001
            "batch": strategy._batch,  # noqa: SLF001
            # Frozen schema (§4.3.6-(6)): per-task inclusive episode range —
            # extension batches shift episode_lo, so the range IS the batch
            # boundary and lands in the hash.
            "tasks": [
                {
                    "task_name": name,
                    "task_id": task_id,
                    "episode_lo": int(strategy._episode_lo.get(name, 0)),  # noqa: SLF001
                    "episode_hi": int(strategy._episode_lo.get(name, 0)) + n - 1,  # noqa: SLF001
                }
                for task_id, (name, n) in enumerate(strategy._tasks)  # noqa: SLF001
            ],
            # Output-affecting parameter, canonicalized: a resume must not be
            # able to split one journal across two spellings of the same root.
            "collect_root": canonical_collect_root(collect_root),
            # Identity of the object pinning, so a resume cannot join a pinned
            # batch onto an unpinned journal (or onto a different pin table).
            "pin_id": strategy._pin_id,  # noqa: SLF001 - own module
        },
        "uids": uids,
        "prefixes": prefixes,
    }
    payload["plan_hash"] = compute_plan_hash(payload)
    return payload


def write_run_plan(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    """Atomic, never-overwriting write; on an existing file, verify hash equality.

    Resume contract: identical parameters recompute the identical plan_hash and
    proceed; anything else aborts. A new batch must go to a new ``_bNN`` file.
    """
    path = pathlib.Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        stored = existing.get("plan_hash")
        recomputed = compute_plan_hash(existing)
        if stored != recomputed:
            raise SystemExit(f"run-plan {path} is corrupt: stored hash {stored} != recomputed {recomputed}")
        if stored != payload["plan_hash"]:
            raise SystemExit(
                f"run-plan mismatch at {path}: existing plan_hash {stored} != new {payload['plan_hash']}. "
                "Parameters changed since the journal was started; refusing to resume. "
                "Start an extension batch (--batch N+1) instead."
            )
        return  # byte-compatible resume
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=1))
    try:
        os.link(tmp, path)  # fails with FileExistsError instead of overwriting
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------
# Worker spawn (agent side)
# ------------------------------------------------------------------


def robocasa_spawn_fn(
    spec: WorkerSpec,
    driver_host: str,
    driver_port: int,
    *,
    worker_python: str,
    robocasa_cwd: str,
    repo_root: str,
    egl_lib_dir: str,
    egl_vendor_dir: str,
    teacher: str,
    connect_deadline_s: float,
    episode_deadline_s: float,
    terminate_grace_s: float,
    max_cached_envs: int | None = None,
    pinned_objects_path: str | None = None,
) -> subprocess.Popen:
    """Launch one island-A worker. All paths come from parameters — never hardcoded.

    ``start_new_session=True`` is a SAFETY requirement, not an option:
    ``WorkerAgent.stop()`` signals ``os.getpgid(pid)``, which without a fresh
    session is the agent's own process group.
    """
    cmd = [
        worker_python,
        "-m",
        "exp.robocasa365.worker_entry",
        "--worker-id", spec.worker_id,
        "--server-key", spec.server_key,
        "--driver-host", driver_host,
        "--driver-port", str(driver_port),
        "--teacher", teacher,
        "--connect-deadline-s", str(connect_deadline_s),
        "--episode-deadline-s", str(episode_deadline_s),
        "--terminate-grace-s", str(terminate_grace_s),
    ]
    if max_cached_envs is not None:
        # Optional so every existing collection invocation stays byte-identical.
        cmd += ["--max-cached-envs", str(max_cached_envs)]
    if pinned_objects_path:
        cmd += ["--pinned-objects", pinned_objects_path]
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    # Island A has openpi_client but not openpi/exp; both come off the repo.
    env["PYTHONPATH"] = os.pathsep.join((repo_root, os.path.join(repo_root, "src")))
    env["MUJOCO_GL"] = "egl"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (egl_lib_dir, env.get("LD_LIBRARY_PATH", "")) if p
    )
    env["__EGL_VENDOR_LIBRARY_DIRS"] = egl_vendor_dir
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    # Same MuJoCo glibc-arena tuning the LIBERO spawn uses (memory plateau).
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
    return subprocess.Popen(cmd, env=env, cwd=robocasa_cwd, start_new_session=True)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_tasks(raw: str, episodes: int) -> list[tuple[str, int]]:
    """``A,B,C`` with a shared count, or ``A:12,B:34`` with per-task counts."""
    tasks: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, count = item.partition(":")
        tasks.append((name, int(count) if sep else int(episodes)))
    if not tasks:
        raise SystemExit("--tasks resolved to an empty list")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", required=True, choices=("driver", "agent", "all"))
    ap.add_argument("--teacher", required=True, choices=TEACHERS)
    ap.add_argument("--servers", required=True, help="comma-separated host:port, ONE teacher's endpoints only")
    ap.add_argument("--tasks", required=True, help="TaskA,TaskB or TaskA:256,TaskB:126 (ordered; order is identity)")
    ap.add_argument("--episodes", type=int, default=1, help="shared per-task episode count when --tasks has no :N")
    ap.add_argument("--layout", type=int, required=True)
    ap.add_argument("--style", type=int, required=True)
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--episode-lo", default="", help="TaskA:20,TaskB:15 — extension-batch episode range starts")
    ap.add_argument(
        "--pinned-objects",
        default="",
        help="pin table path; pins every object slot to one exact mesh. The "
        "driver re-derives its pin_id and every worker re-reads the file.",
    )
    ap.add_argument("--collect-root", required=True, help="scene root the servers write under (hashed into run-plan)")
    ap.add_argument("--journal", default="", help="default: exp/robocasa365/data/journal_<runid>.jsonl")
    ap.add_argument("--run-plan-dir", default="", help="default: alongside the journal")
    ap.add_argument("--env-config", required=True, help="env-config file with per-teacher endpoint groups + paths")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--driver-host", default="", help="agent role: where the driver's pull port lives")
    ap.add_argument("--driver-port", type=int, default=0, help="agent role: the driver's pull port")
    ap.add_argument("--gpu-ids", default="0", help="comma-separated CUDA slots, one worker per bound server")
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--connect-deadline-s", type=float, required=True)
    ap.add_argument("--episode-deadline-s", type=float, required=True)
    ap.add_argument("--terminate-grace-s", type=float, required=True)
    args = ap.parse_args()

    env_config = load_env_config(args.env_config)
    servers = []
    for spec in args.servers.split(","):
        host, port = spec.strip().rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))
    # Affinity gate BEFORE any graph is built (frozen §4.3.4a).
    validate_teacher_endpoints(args.teacher, servers, env_config)

    tasks = parse_tasks(args.tasks, args.episodes)
    episode_lo = {}
    for item in args.episode_lo.split(","):
        item = item.strip()
        if item:
            name, _, lo = item.partition(":")
            episode_lo[name] = int(lo)
    # Resolved once here, on the driver: the worker's cwd is the external
    # RoboCasa checkout, so a relative path forwarded verbatim opens nothing.
    pin_path = resolve_manifest_path(args.pinned_objects) if args.pinned_objects else ""
    pin_id, pinned_objects = (None, None)
    if pin_path:
        pin_id, pinned_objects = load_pin_manifest(pin_path)
        print(f"[run_collect] pin_id={pin_id} manifest={pin_path}", flush=True)
    strategy = RobocasaCollectStrategy(
        teacher=args.teacher,
        layout=args.layout,
        style=args.style,
        base_seed=args.base_seed,
        replan_steps=args.replan_steps,
        tasks=tasks,
        batch=args.batch,
        episode_lo=episode_lo,
        pin_id=pin_id,
        pinned_objects=pinned_objects,
    )
    run_id = strategy.run_id
    data_dir = pathlib.Path(__file__).resolve().parent / "data"
    journal_path = pathlib.Path(args.journal) if args.journal else data_dir / f"journal_{run_id}.jsonl"
    plan_dir = pathlib.Path(args.run_plan_dir) if args.run_plan_dir else journal_path.parent
    run_plan_path = plan_dir / f"run_plan_{run_id}_b{args.batch:02d}.json"

    yaml_weights = {yid: n for yid, (_, n) in zip(strategy.yaml_ids, tasks)}
    server_capacities = {s.key: 1 for s in servers}  # frozen D-L: 1 connection per server process

    agent = None
    driver = None
    if args.role in ("driver", "all"):
        from openpi.conductor.driver import assign_servers

        # Persist the run-plan from the same strategy/assignment inputs the
        # driver ctor uses; plan() is deterministic (test-pinned), so the
        # serialized UIDs are value-identical to the dispatched ones.
        assignment = assign_servers(yaml_weights, servers, None, server_capacities)
        graph = strategy.plan(sorted(yaml_weights), assignment)
        run_plan = build_run_plan(strategy, graph, args.collect_root)
        write_run_plan(run_plan_path, run_plan)
        print(f"[run_collect] run-plan {run_plan_path} plan_hash={run_plan['plan_hash']}", flush=True)

        driver = ConductorDriver(
            strategy,
            yaml_weights=yaml_weights,
            servers=servers,
            journal_path=str(journal_path),
            ctl_factory=lambda _server: _NoOpCtl(),  # collection only; eval needs the real ctl
            episode_timeout_s=args.episode_timeout_s,
            bind_host=args.bind_host,
            server_capacities=server_capacities,
        )
        driver_thread = threading.Thread(target=driver.run, daemon=True)
        driver_thread.start()
        while driver.port is None:
            time.sleep(0.05)
        print(f"[run_collect] driver pull port = {driver.port}", flush=True)

    if args.role in ("agent", "all"):
        driver_host = args.driver_host or args.bind_host
        driver_port = args.driver_port or (driver.port if driver is not None else 0)
        if not driver_port:
            raise SystemExit("--role agent requires --driver-host/--driver-port of a running driver")
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
        specs = [
            WorkerSpec(worker_id=f"w{i}", server_key=server.key, gpu_id=gpu_ids[i % len(gpu_ids)])
            for i, server in enumerate(servers)
        ]
        spawn = functools.partial(
            robocasa_spawn_fn,
            worker_python=env_config["WORKER_PYTHON"],
            robocasa_cwd=env_config["ROBOCASA_CWD"],
            repo_root=env_config["REPO_ROOT"],
            egl_lib_dir=env_config["EGL_LIB_DIR"],
            egl_vendor_dir=env_config["EGL_VENDOR_DIR"],
            teacher=args.teacher,
            connect_deadline_s=args.connect_deadline_s,
            episode_deadline_s=args.episode_deadline_s,
            terminate_grace_s=args.terminate_grace_s,
            pinned_objects_path=pin_path or None,
        )
        agent = WorkerAgent(specs, driver_host=driver_host, driver_port=driver_port, spawn_fn=spawn)
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()
        print(f"[run_collect] agent supervising {len(specs)} worker(s)", flush=True)

    try:
        if driver is not None:
            while driver_thread.is_alive():
                driver_thread.join(timeout=5.0)
            print("[run_collect] driver finished", flush=True)
        else:
            while True:
                time.sleep(5.0)
    finally:
        if agent is not None:
            agent.stop()


if __name__ == "__main__":
    main()
