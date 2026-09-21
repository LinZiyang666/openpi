"""RoboCasa365 conductor driver of the step-vs-warm-start line (plan §3.2).

One invocation = one ``(teacher, arm, lane)`` cell: the driver dispatches the cell's task subset
to a worker fleet (``--role agent`` on a timan box; ``worker_entry.StepDiagEpisodeRunner`` delegates
to the production ``RobocasaEpisodeRunner`` and only adds the client proxy / counting env /
summary row) against the server processes that serve exactly this arm (``serve_diag_pi05`` /
``serve_diag_groot``, one config, one or more processes).

Why a strategy of its own: ``run_ws_search`` guards the PnP lane with the frozen five-task x 50
design, which refuses this line's task subsets and 10-episode shadow cells; this strategy keeps
its own explicit identity instead (task ids taken from the 13-task roster, never renumbered;
pin table resolved with the production helpers; the exact expected episode set written into
the launch manifest and re-checked by the summary).

Evidence written by the driver (everything else lives on the server side):

* ``journal_<run_id>.jsonl``      the conductor's terminal records (outcome authority);
* ``run_plan_<run_id>.json``       the dispatched graph + identity (``build_run_plan``);
* ``per_step_<yaml_id>.jsonl``     the worker-side rows: the runner's per-decision ``__hit_meta__``
                                   rows plus one ``episode_summary`` row per episode
                                   (``reported_n_steps`` / ``n_env_steps`` / ``n_decisions``),
                                   drained per episode by the driver's ``per_step_writer``;
* ``launch_<launch_id>.json``      ``driver.run_id`` <-> launch id, arm, config sha, the expected
                                   ``(task, env_seed, init_idx, lane, pin)`` set;
* ``summary_<run_id>.json``        SR per task reconciled against the expected uid set.

usage (driver + local agent, or ``--role driver`` / ``--role agent`` on separate hosts)::

    python -m exp.step_diag.run_diag --teacher pi05 --servers h100:23240,h100:23241,h100:23242 \\
        --arm-id plain_k2 --experiment-id sdiag_rc_v1 --lane main --tasks CloseFridge,OpenCabinet,OpenDrawer \\
        --episodes 50 --episodes-map '{"OpenDrawer": 100}' --base-seed 2000000 --config-sha <sha> \\
        --env-config exp/step_diag/config/rc_timan.env --role all --workers-per-server 1

Serving topology: the arm's servers run single-connection (``serve_diag_pi05`` passes
``--non-concurrent``; the GR00T RC server has no ``--concurrent``), so every server process takes
exactly one worker and the tasks are spread over the servers by ``assign_servers`` (one task
never straddles two servers). ``--workers-per-server`` above 1 is only for a server mode that
accepts several connections.
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import threading
import time
import uuid
from typing import Any

from openpi.conductor import ServerEndpoint, WorkerAgent, WorkerSpec
from openpi.conductor.driver import ConductorDriver, assign_servers
from openpi.conductor.task import EpisodeTask, Stage, TaskGraph, make_task_uid

from exp.robocasa365.pinned_objects import load_pin_manifest, resolve_manifest_path
from exp.robocasa365.run_collect import (
    RobocasaCollectStrategy,
    _NoOpCtl,
    build_run_plan as _build_run_plan,
    compute_plan_hash,
    build_yaml_id,
    compute_pin_task_id,
    load_env_config,
    parse_tasks,
    validate_teacher_endpoints,
    write_run_plan,
)
from exp.robocasa365.run_ws_search import DEFAULT_EVAL_TASKS, EVAL_NO_COLLECT_ROOT, summarize_journal
from exp.step_diag import envs as _envs
from exp.step_diag.worker_entry import step_diag_spawn_fn

ROSTER: tuple[str, ...] = tuple(DEFAULT_EVAL_TASKS.split(","))
DEFAULT_OUT_ROOT = pathlib.Path(__file__).resolve().parent / "data" / "rc"


class StepDiagStrategy(RobocasaCollectStrategy):
    """Collection strategy with the arm folded into the identity and roster task ids."""

    def __init__(self, *, arm_id: str, experiment_id: str, lane: str, launch_id: str = "", config_sha: str = "",
                 run_prefix: str = "sdiag", slots=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._arm_id = str(arm_id)
        self._experiment_id = str(experiment_id)
        self._lane = str(lane)
        # Stamped into every task's extra and, by the worker's client proxy, into episode_start:
        # the server-side rows join on (launch_id, arm_id, config_sha) — see worker_entry.
        self._launch_id = str(launch_id)
        self._config_sha = str(config_sha)
        self._slots = list(slots or [])
        identity = _envs.sha256_json([experiment_id, config_sha])[:12]
        self.run_id = f"{run_prefix}-{self._arm_id}-{lane}-{identity}__l{self._layout}s{self._style}_{self._teacher}"
        unknown = [name for name, _ in self._tasks if name not in ROSTER]
        if unknown:
            raise ValueError(f"tasks outside the 13-task roster: {unknown}")
        for name, _ in self._tasks:
            if _envs.lane_of(name) != self._lane:
                raise ValueError(f"task {name} is not in lane {self._lane!r}")

    def plan(self, yamls: list[str], server_assignment: dict[str, ServerEndpoint]) -> TaskGraph:
        del yamls
        graph = TaskGraph()
        for task_name, n_episodes in self._tasks:
            task_id = ROSTER.index(task_name)  # roster position, never the subset position
            yaml_id = build_yaml_id(self.run_id, task_name)
            server = self._slots[task_id % len(self._slots)] if self._slots else server_assignment[yaml_id]
            lo = int(self._episode_lo.get(task_name, 0))
            episodes = []
            for offset in range(int(n_episodes)):
                episode_idx = lo + offset
                extra: dict[str, Any] = {
                    "task_name": task_name, "layout": self._layout, "style": self._style,
                    "teacher": self._teacher, "base_seed": self._base_seed, "replan_steps": self._replan_steps,
                    "batch": self._batch, "arm_id": self._arm_id, "experiment_id": self._experiment_id,
                    "lane": self._lane, "launch_id": self._launch_id, "config_sha": self._config_sha,
                }
                if self._pinned_objects is not None:
                    extra.update({
                        "pin_id": self._pin_id,
                        "pin_task_id": compute_pin_task_id(task_name, self._pinned_objects[task_name]),
                        "pinned_objects": self._pinned_objects[task_name],
                    })
                episodes.append(EpisodeTask(
                    task_uid=make_task_uid(yaml_id, "eval", task_id, episode_idx), yaml_id=yaml_id, phase="eval",
                    experiment=self._teacher, task_id=task_id, episode_idx=episode_idx,
                    orig_init_state_idx=episode_idx, server_host=server.host, server_port=server.port,
                    bundle_id="default", extra=extra,
                ))
            graph.add_stage(Stage(stage_id=yaml_id, yaml_id=yaml_id, phase="eval", server=server, episodes=episodes))
        graph.validate()
        return graph

    def expected_identities(self) -> list[dict]:
        """The exact episode set of this cell: what the analysis must find, no more, no less."""
        out = []
        for task_name, n in self._tasks:
            lo = int(self._episode_lo.get(task_name, 0))
            for idx in range(lo, lo + int(n)):
                out.append({"task": task_name, "task_id": ROSTER.index(task_name), "init_idx": idx,
                            "env_seed": self._base_seed + idx, "lane": self._lane, "pin_id": self._pin_id,
                            "layout": self._layout, "style": self._style,
                            "task_uid": make_task_uid(build_yaml_id(self.run_id, task_name), "eval",
                                                      ROSTER.index(task_name), idx)})
        return out


def build_run_plan(strategy, graph, collect_root):
    """Bind resume to the experiment and configuration as well as the production graph."""
    payload = _build_run_plan(strategy, graph, collect_root)
    payload["step_diag"] = {"experiment_id": strategy._experiment_id, "arm_id": strategy._arm_id,
                            "lane": strategy._lane, "config_sha": strategy._config_sha,
                            "expected": strategy.expected_identities()}
    payload["step_diag"]["assignment"] = {sid: stage.server.key for sid, stage in graph.stages.items()}
    for task in payload.get("params", {}).get("tasks", []):
        if isinstance(task, dict) and task.get("task_name") in ROSTER:
            task["task_id"] = ROSTER.index(task["task_name"])
    payload["plan_hash"] = compute_plan_hash(payload)
    return payload


def per_step_writer_for(out_dir: pathlib.Path):
    """Append the drained per-step rows of one yaml to ``per_step_<yaml_id>.jsonl``."""
    lock = threading.Lock()

    def _write(yaml_id: str, rows: list[dict]) -> None:
        with lock, (out_dir / f"per_step_{yaml_id}.jsonl").open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return _write


def build_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    tasks = parse_tasks(args.tasks, args.episodes)
    if args.episodes_map:
        override = json.loads(args.episodes_map)
        tasks = [(name, int(override.get(name, n))) for name, n in tasks]
    if not tasks or len({name for name, _ in tasks}) != len(tasks) or any(n < 1 for _, n in tasks):
        raise ValueError("tasks must be nonempty, unique and have positive episode counts")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, choices=("pi05", "groot_tp"))
    ap.add_argument("--servers", required=True,
                    help="host:port[,host:port...] of the processes serving THIS arm (all the same config)")
    ap.add_argument("--arm-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--lane", required=True, choices=("main", "pnp"))
    ap.add_argument("--tasks", required=True, help="comma-separated task names of one lane")
    ap.add_argument("--episodes", type=int, default=_envs.QB_EPISODES)
    ap.add_argument("--episodes-map", default="", help='json {task: n} overriding --episodes per task')
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=_envs.RC_FORMAL_BASE_SEED)
    ap.add_argument("--replan-steps", type=int, default=_envs.REPLAN_STEPS)
    ap.add_argument("--pinned-objects", default="", help="pnp lane: the pinned-object manifest")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--launch-id", default="")
    ap.add_argument("--config-sha", required=True, help="config_sha of the served arm (server manifest_<arm>.json)")
    ap.add_argument("--role", choices=("driver", "agent", "all"), default="all")
    ap.add_argument("--workers-per-server", type=int, default=1,
                    help="sim workers bound to each server (1: the servers run single-connection)")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--env-config", default="", help="agent role: KEY=VALUE env file of the worker box")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--bind-port", type=int, default=0)
    ap.add_argument("--driver-host", default="")
    ap.add_argument("--driver-port", type=int, default=0)
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--connect-deadline-s", type=float, default=600.0)
    ap.add_argument("--episode-deadline-s", type=float, default=1500.0)
    ap.add_argument("--terminate-grace-s", type=float, default=60.0)
    args = ap.parse_args()

    if not args.config_sha.strip() or args.workers_per_server != 1:
        ap.error("a nonempty --config-sha and exactly one worker per server are required")

    if (args.lane == "pnp") != bool(args.pinned_objects):
        ap.error("--lane pnp requires --pinned-objects (and main forbids it)")
    slots = []
    for spec in args.servers.split(","):
        host, _, port = spec.strip().rpartition(":")
        slots.append(ServerEndpoint(host, int(port)))
    if len({s.key for s in slots}) != len(slots):
        ap.error("--servers lists an endpoint twice")
    env_config = load_env_config(args.env_config) if args.env_config else {}
    if env_config:
        validate_teacher_endpoints(args.teacher, slots, env_config)

    pin_id, pinned_objects = (None, None)
    if args.pinned_objects:
        pin_path = resolve_manifest_path(args.pinned_objects)
        pin_id, pinned_objects = load_pin_manifest(pin_path)
        print(f"[step_diag] pin_id={pin_id} manifest={pin_path}", flush=True)

    tasks = build_tasks(args)
    policy = "pi05" if args.teacher == "pi05" else "groot"
    if args.replan_steps != 5 or (args.layout, args.style) != (1, 1):
        ap.error("this experiment freezes replan=5 and layout/style=1")
    if args.base_seed == _envs.RC_SMOKE_BASE_SEED:
        if "smoke" not in args.experiment_id.lower():
            ap.error("smoke seeds require a separate experiment id containing 'smoke'")
    elif args.base_seed == _envs.RC_FORMAL_BASE_SEED:
        if "smoke" in args.experiment_id.lower():
            ap.error("formal seeds cannot use a smoke experiment id")
        allowed = {"shadow", "full", *(f"plain_k{k}" for k in _envs.QB_PLAIN_KS[policy]),
                   *(f"warm_t{t:g}" for t in _envs.QB_WARM_TS[policy])}
        if args.arm_id not in allowed:
            ap.error("unknown formal arm")
        for name, n in tasks:
            want = 10 if args.arm_id == "shadow" else _envs.qb_episode_count(policy, name, args.arm_id)
            if n != want or (args.arm_id != "shadow" and name not in _envs.qb_tasks(policy)):
                ap.error(f"{name}/{args.arm_id}: expected {want} episodes in the frozen task set")
    else:
        ap.error("base seed must be the frozen formal or smoke seed")
    if pin_id is not None and pin_id != _envs.canonical_pin_id():
        ap.error("PnP requires the canonical frozen pin table")
    launch_id = args.launch_id or uuid.uuid4().hex[:10]
    strategy = StepDiagStrategy(
        arm_id=args.arm_id, experiment_id=args.experiment_id, lane=args.lane, launch_id=launch_id,
        config_sha=args.config_sha, teacher=args.teacher, layout=args.layout, style=args.style,
        slots=slots,
        base_seed=args.base_seed, replan_steps=args.replan_steps, tasks=tasks, pin_id=pin_id,
        pinned_objects=pinned_objects,
    )
    run_id = strategy.run_id
    yaml_weights = {yid: n for yid, (_, n) in zip(strategy.yaml_ids, tasks)}
    server_capacities = {s.key: args.workers_per_server for s in slots}

    driver = driver_thread = None
    run_plan = None
    out_dir = pathlib.Path(args.out_root) / args.teacher / args.arm_id
    journal_path = out_dir / f"journal_{run_id}.jsonl"
    summary_path = out_dir / f"summary_{run_id}.json"
    if args.role in ("driver", "all"):
        out_dir.mkdir(parents=True, exist_ok=True)
        assignment = assign_servers(yaml_weights, slots, None, server_capacities)
        graph = strategy.plan(sorted(yaml_weights), assignment)
        assignment = {sid: stage.server for sid, stage in graph.stages.items()}
        run_plan = build_run_plan(strategy, graph, EVAL_NO_COLLECT_ROOT)
        write_run_plan(out_dir / f"run_plan_{run_id}.json", run_plan)
        driver = ConductorDriver(
            strategy, yaml_weights=yaml_weights, servers=slots, journal_path=str(journal_path),
            ctl_factory=lambda _server: _NoOpCtl(), episode_timeout_s=args.episode_timeout_s,
            bind_host=args.bind_host, bind_port=args.bind_port, server_capacities=server_capacities,
            per_step_writer=per_step_writer_for(out_dir),
        )
        launch = {
            "launch_id": launch_id, "driver_run_id": driver.run_id, "run_id": run_id, "arm_id": args.arm_id,
            "experiment_id": args.experiment_id, "teacher": args.teacher, "lane": args.lane,
            "servers": [s.key for s in slots], "workers_per_server": args.workers_per_server,
            "assignment": {yid: assignment[yid].key for yid in sorted(assignment)},
            "config_sha": args.config_sha, "plan_hash": run_plan["plan_hash"], "tasks": tasks, "base_seed": args.base_seed,
            "replan_steps": args.replan_steps, "pin_id": pin_id, "expected": strategy.expected_identities(),
            "layout": args.layout, "style": args.style,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_dir / f"launch_{launch_id}.json").write_text(json.dumps(launch, indent=1))
        driver_thread = threading.Thread(target=driver.run, daemon=True)
        driver_thread.start()
        deadline = time.monotonic() + 30
        while driver.port is None:
            if not driver_thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("conductor driver failed to start")
            time.sleep(0.05)
        print(f"[step_diag] driver run_id={driver.run_id} launch={launch_id} pull port={driver.port} "
              f"expected={len(launch['expected'])} episodes", flush=True)

    agent = None
    if args.role in ("agent", "all"):
        if not env_config:
            raise SystemExit("--role agent/all requires --env-config")
        driver_host = args.driver_host or args.bind_host
        driver_port = args.driver_port or (driver.port if driver is not None else 0)
        if not driver_port:
            raise SystemExit("--role agent requires --driver-host/--driver-port of a running driver")
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
        specs = []
        for slot in slots:
            for w in range(args.workers_per_server):
                i = len(specs)
                specs.append(WorkerSpec(worker_id=f"w{i}", server_key=slot.key, gpu_id=gpu_ids[i % len(gpu_ids)]))
        spawn = functools.partial(
            step_diag_spawn_fn, worker_python=env_config["WORKER_PYTHON"], robocasa_cwd=env_config["ROBOCASA_CWD"],
            repo_root=env_config["REPO_ROOT"], egl_lib_dir=env_config["EGL_LIB_DIR"],
            egl_vendor_dir=env_config["EGL_VENDOR_DIR"], teacher=args.teacher,
            connect_deadline_s=args.connect_deadline_s, episode_deadline_s=args.episode_deadline_s,
            terminate_grace_s=args.terminate_grace_s,
            pinned_objects_path=resolve_manifest_path(args.pinned_objects) if args.pinned_objects else None,
            max_cached_envs=1,
        )
        agent = WorkerAgent(specs, driver_host=driver_host, driver_port=driver_port, spawn_fn=spawn)
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()
        print(f"[step_diag] agent supervising {len(specs)} worker(s) -> {[s.key for s in slots]} "
              f"({driver_host}:{driver_port})", flush=True)

    try:
        if driver_thread is not None:
            while driver_thread.is_alive():
                driver_thread.join(timeout=5.0)
        else:
            while True:
                time.sleep(5.0)
    finally:
        if agent is not None:
            agent.stop()

    if driver is not None:
        summary = summarize_journal(journal_path, expected_uids=list(run_plan["uids"]))
        summary_path.write_text(json.dumps({"arm_id": args.arm_id, "teacher": args.teacher, "lane": args.lane,
                                            "launch_id": launch_id, "driver_run_id": driver.run_id, **summary},
                                           indent=1))
        print(f"[step_diag] {'DONE' if summary['complete'] else 'INCOMPLETE'} arm={args.arm_id} "
              f"macro_sr={summary['macro_sr']} n_err={summary['n_err']} n_missing={summary['n_missing']} "
              f"-> {summary_path}", flush=True)
        if not summary["complete"]:
            # Operators chain cells on the exit code; an incomplete cell (retries exhausted, missing
            # identities) must not read as success. Re-running the same command resumes it.
            raise SystemExit(1)


if __name__ == "__main__":
    main()
