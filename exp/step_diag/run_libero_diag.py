"""Conductor LIBERO driver and worker of the step-vs-warm-start line, using the production episode kernel.

One invocation = one ``(env, arm)`` cell over a LIBERO task subset. Run with the simulator's Python
directly; no conda executable is needed. One worker per server also supports the single-connection
pi0.5 server. Every episode has a launch manifest entry (its expected identity: task x pool init_idx,
env seed 7, pool content digest), an accepted journal record and a worker ``episode_summary`` row.

* ``--arm-id shadow`` (default): the LIBERO shadow, all 10 tasks x init_idx 0..9, written to
  ``<out-root>/<teacher>/shadow_<env_id>/`` under its historical run id ``sdiag-lib-<digest>``.
* any served arm (``full``, ``plain_k<k>``, ``warm_t<t>``, the warm-reset and self-start variants):
  ``--tasks`` (LIBERO task ids, default all 10) x init_idx 0..``--episodes``-1, written to
  ``<out-root>/<teacher>/<env_id>/<arm_id>/`` (one arm id runs on both suites). ``--run-prefix`` names
  the journal / run plan, so sequential single-task cells of one arm on one slot never share a run id
  (as ``run_diag``). The self-start round (``envs.LIBERO_SELF_EXPERIMENT_ID``) admits its arm table only,
  50 episodes per task (the task's whole pruned A pool) and its own out root; a smoke experiment id
  (containing ``smoke``) admits any arm id and 1..50 episodes; any other experiment id runs the shadow only.

usage (one self-start cell: GR00T libero_10, task 3, 50 episodes)::

    python -m exp.step_diag.run_libero_diag --env-id groot_libero_10 --arm-id selfmidreset_t0.75_n1 \\
        --experiment-id sdiag_libero_self --servers weilandserver:23160 --pool <pruned A pool dir> \\
        --config-sha <sha> --tasks 3 --episodes 50 --run-prefix sdq23160t3 \\
        --out-root exp/step_diag/data/libero_self
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import uuid

from openpi.conductor import ServerEndpoint, WorkerAgent, WorkerSpec
from openpi.conductor.driver import ConductorDriver
from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.task import EpisodeTask, Stage, TaskGraph, make_task_uid
from openpi.conductor.worker import WorkerLoop
from examples.libero.episode_runner import LiberoEpisodeRunner, default_client_factory
from exp.robocasa365.run_collect import _NoOpCtl, compute_plan_hash, write_run_plan
from exp.step_diag import envs as E
from exp.step_diag.run_diag import per_step_writer_for
from exp.step_diag.worker_entry import _ClientProxy, _CountingEnv, worker_runtime

DEFAULT_OUT_ROOT = "exp/step_diag/data/libero"
SHADOW_RUN_PREFIX = "sdiag-lib"


class LiberoDiagRunner(LiberoEpisodeRunner):
    def close_current_client(self):
        self.close()

    def __init__(
        self,
        args,
        setup,
        *,
        client_factory=default_client_factory,
        pool_sha=None,
        **kwargs,
    ):
        self._counting = None
        self._pool_sha = pool_sha

        def counted(task):
            env, initial, prompt, limit = setup(task)
            self._counting = _CountingEnv(env, lambda _: None)
            return self._counting, initial, prompt, limit

        super().__init__(
            args,
            counted,
            client_factory=lambda s: _ClientProxy(client_factory(s)),
            **kwargs,
        )

    def run(self, task, report):
        if (
            self._pool_sha is not None
            and task.extra.get("init_pool_sha256") != self._pool_sha
        ):
            raise ValueError("worker pool bytes differ from the driver's frozen pool")
        client = self._ensure_client(task)
        client.stamp = {
            k: task.extra[k]
            for k in (
                "launch_id",
                "arm_id",
                "experiment_id",
                "config_sha",
                "init_pool_sha256",
                "seed",
            )
        }
        try:
            result = super().run(task, report)
            result.per_step_rows.append(
                {
                    "row": "episode_summary",
                    "step_idx": -2,
                    "task_uid": task.task_uid,
                    "attempt": task.attempt,
                    "task_id": task.task_id,
                    "init_idx": task.orig_init_state_idx,
                    "n_decisions": client.n_infer,
                    # counted from the reset: the settle steps before the first decision are included
                    "n_env_steps": self._counting.n_steps,
                    "num_steps_wait": getattr(self._args, "num_steps_wait", None),
                    "success": result.success,
                    "error": result.error,
                    "worker_runtime": worker_runtime(),
                    **client.stamp,
                }
            )
            return result
        finally:
            client.stamp = {}
            if self._counting is not None:
                self._counting.close()
                self._counting = None


class LiberoDiagStrategy(ExperimentStrategy):
    """One ``(env, arm)`` cell: ``tasks`` x init_idx ``0..episodes-1`` of the frozen pool, LIBERO env seed 7.

    Task ids are suite positions (a subset never renumbers them). The run id folds in the environment,
    experiment, served configuration and pool bytes, and off the shadow the prefix and arm, so a changed
    pool or configuration never resumes another cell's journal; the shadow keeps its historical run id.
    """

    def __init__(self, args, names, pool_sha, launch, *, arm_id="shadow", tasks=None,
                 episodes=E.SHADOW_EPISODES, run_prefix=SHADOW_RUN_PREFIX):
        self.args, self.names, self.pool_sha, self.launch = (
            args,
            names,
            pool_sha,
            launch,
        )
        self.arm_id = str(arm_id)
        self.tasks = tuple(range(len(names))) if tasks is None else tuple(int(t) for t in tasks)
        self.episodes = int(episodes)
        if (not self.tasks or len(set(self.tasks)) != len(self.tasks)
                or any(not 0 <= t < len(names) for t in self.tasks) or self.episodes < 1):
            raise ValueError("tasks must be unique suite task ids and episodes positive")
        digest = E.sha256_json([args.env_id, args.experiment_id, args.config_sha, pool_sha])[:16]
        self.run_id = (f"{run_prefix}-{digest}" if self.arm_id == "shadow"
                       else f"{run_prefix}-{self.arm_id}-{digest}")
        self.yaml_ids = [f"{self.run_id}_t{t}" for t in self.tasks]

    def plan(self, yamls, server_assignment):
        graph = TaskGraph()
        for task_id, yid in zip(self.tasks, self.yaml_ids):
            server = server_assignment[yid]
            tasks = [
                EpisodeTask(
                    task_uid=make_task_uid(yid, "eval", task_id, i),
                    yaml_id=yid,
                    phase="eval",
                    experiment=E.resolve_env(self.args.env_id).benchmark,
                    task_id=task_id,
                    episode_idx=i,
                    orig_init_state_idx=i,
                    server_host=server.host,
                    server_port=server.port,
                    bundle_id="default",
                    extra={
                        "num_trials_per_task": self.episodes,
                        "seed": E.LIBERO_ENV_SEED,
                        "init_pool_sha256": self.pool_sha,
                        "launch_id": self.launch,
                        "arm_id": self.arm_id,
                        "experiment_id": self.args.experiment_id,
                        "config_sha": self.args.config_sha,
                    },
                )
                for i in range(self.episodes)
            ]
            graph.add_stage(
                Stage(
                    stage_id=yid,
                    yaml_id=yid,
                    phase="eval",
                    server=server,
                    episodes=tasks,
                )
            )
        graph.validate()
        return graph

    def expected(self):
        return [
            {
                "task_uid": make_task_uid(y, "eval", t, i),
                "task": self.names[t],
                "task_id": t,
                "init_idx": i,
                "env_seed": E.LIBERO_ENV_SEED,
                "init_pool_sha256": self.pool_sha,
            }
            for t, y in zip(self.tasks, self.yaml_ids)
            for i in range(self.episodes)
        ]


def parse_tasks(spec: str) -> tuple[int, ...]:
    """``--tasks``: comma-separated LIBERO task ids; empty = all of the suite."""
    if not spec.strip():
        return tuple(range(E.LIBERO_N_TASKS))
    tasks = tuple(int(x) for x in spec.split(",") if x.strip())
    if not tasks or len(set(tasks)) != len(tasks) or any(not 0 <= t < E.LIBERO_N_TASKS for t in tasks):
        raise ValueError(f"--tasks must list unique LIBERO task ids in 0..{E.LIBERO_N_TASKS - 1}")
    return tasks


def _known_arm_id(arm_id: str) -> bool:
    if arm_id == "full":
        return True
    if arm_id.startswith("plain_k"):
        return arm_id[len("plain_k"):].isdigit()
    try:
        E.warm_t_of(arm_id)
    except ValueError:
        return False
    return True


def check_cell(a: argparse.Namespace, env: E.EnvSpec) -> tuple[tuple[int, ...], int]:
    """Validate one driver invocation against its experiment; returns ``(task ids, episodes per task)``."""
    tasks = parse_tasks(a.tasks)
    if a.arm_id == "shadow":
        episodes = E.SHADOW_EPISODES if a.episodes is None else int(a.episodes)
        if tasks != tuple(range(E.LIBERO_N_TASKS)) or episodes != E.SHADOW_EPISODES:
            raise ValueError(f"the LIBERO shadow runs all {E.LIBERO_N_TASKS} tasks x init_idx 0..{E.SHADOW_EPISODES - 1}")
        return tasks, episodes
    if a.experiment_id == E.LIBERO_SELF_EXPERIMENT_ID:
        episodes = E.LIBERO_SELF_EPISODES if a.episodes is None else int(a.episodes)
        if env.benchmark not in E.LIBERO_SELF_SUITES:
            raise ValueError(f"{E.LIBERO_SELF_EXPERIMENT_ID} runs on {list(E.LIBERO_SELF_SUITES)}")
        if a.arm_id not in E.LIBERO_SELF_ARMS_BY_POLICY[env.policy]:
            raise ValueError(f"{a.arm_id}: not an arm of {E.LIBERO_SELF_EXPERIMENT_ID} ({env.policy})")
        if episodes != E.LIBERO_SELF_EPISODES:
            raise ValueError(f"{E.LIBERO_SELF_EXPERIMENT_ID} runs {E.LIBERO_SELF_EPISODES} episodes per task")
        if pathlib.Path(a.out_root).resolve() == pathlib.Path(DEFAULT_OUT_ROOT).resolve():
            raise ValueError(f"{E.LIBERO_SELF_EXPERIMENT_ID} needs its own --out-root (not the shadow root)")
        return tasks, episodes
    if "smoke" in a.experiment_id.lower():
        if a.episodes is None or not 1 <= int(a.episodes) <= E.LIBERO_SELF_EPISODES:
            raise ValueError(f"a smoke cell names --episodes in 1..{E.LIBERO_SELF_EPISODES}")
        if not _known_arm_id(a.arm_id):
            raise ValueError(f"{a.arm_id}: not an arm id of this line")
        return tasks, int(a.episodes)
    raise ValueError(f"{a.experiment_id}: only the shadow runs outside {E.LIBERO_SELF_EXPERIMENT_ID} and smoke ids")


def cell_dir(out_root: str | pathlib.Path, teacher: str, env_id: str, arm_id: str) -> pathlib.Path:
    """Driver output directory of one cell (the shadow keeps its historical layout)."""
    root = pathlib.Path(out_root) / teacher
    return root / f"shadow_{env_id}" if arm_id == "shadow" else root / env_id / arm_id


def build_parser() -> argparse.ArgumentParser:
    """The driver / worker command line (``--role all`` runs the driver and spawns one worker per server;
    ``--role worker`` is the spawned worker). ``check_cell`` validates a parsed driver invocation."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("all", "worker"), default="all")
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--config-sha", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--servers", default="")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--arm-id", default="shadow", help="the arm the servers serve (default: the shadow)")
    ap.add_argument("--tasks", default="", help="comma-separated LIBERO task ids (default: all 10)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes per task = init_idx 0..n-1 of the pool (default: shadow 10, self-start round 50)")
    ap.add_argument("--run-prefix", default=SHADOW_RUN_PREFIX,
                    help="run id / journal prefix (one per sequential or concurrent cell of the same arm)")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--worker-id")
    ap.add_argument("--server-key")
    ap.add_argument("--driver-host", default="127.0.0.1")
    ap.add_argument("--driver-port", type=int)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    env = E.resolve_env(a.env_id)
    if not env.benchmark.startswith("libero") or not a.config_sha:
        ap.error("requires a LIBERO environment and configuration digest")
    if a.role == "worker":
        pool_sha = E.sha256_tree(a.pool)
        from examples.libero import main as m
        from examples.libero.worker_entry import _build_episode_setup

        args = m.Args(
            task_suite_name=env.benchmark,
            seed=E.LIBERO_ENV_SEED,
            replan_steps=5,
            resize_size=224 if env.policy == "pi05" else 256,
        )
        setup = _build_episode_setup(args, E.LIBERO_ENV_SEED, a.pool)
        runner = LiberoDiagRunner(args, setup, pool_sha=pool_sha)
        from exp.robocasa365.episode_runner import WatchdogRunner

        runner = WatchdogRunner(runner, episode_deadline_s=1500, terminate_grace_s=60)
        WorkerLoop(
            a.worker_id,
            a.server_key,
            runner,
            connect=lambda: socket.create_connection((a.driver_host, a.driver_port)),
        ).run_forever()
        return
    try:
        tasks, episodes = check_cell(a, env)
    except ValueError as exc:
        ap.error(str(exc))
    pool_sha = E.sha256_tree(a.pool)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[env.benchmark]()
    names = [suite.get_task(i).language for i in range(E.LIBERO_N_TASKS)]
    slots = [
        ServerEndpoint(h, int(p))
        for h, p in (s.rsplit(":", 1) for s in a.servers.split(","))
    ]
    if len({s.key for s in slots}) != len(slots):
        ap.error("duplicate servers")
    launch_id = uuid.uuid4().hex
    strategy = LiberoDiagStrategy(a, names, pool_sha, launch_id, arm_id=a.arm_id, tasks=tasks, episodes=episodes,
                                  run_prefix=a.run_prefix)
    teacher = "pi05" if env.policy == "pi05" else "groot_tp"
    out = cell_dir(a.out_root, teacher, a.env_id, a.arm_id)
    out.mkdir(parents=True, exist_ok=True)
    expected = strategy.expected()
    plan = {
        "expected": expected,
        "config_sha": a.config_sha,
        "experiment_id": a.experiment_id,
        "servers": [s.key for s in slots],
        "env_id": a.env_id,
        "resize_size": 224 if env.policy == "pi05" else 256,
    }
    plan["plan_hash"] = compute_plan_hash(plan)
    write_run_plan(out / f"run_plan_{strategy.run_id}.json", plan)
    driver = ConductorDriver(
        strategy,
        yaml_weights={y: episodes for y in strategy.yaml_ids},
        servers=slots,
        journal_path=str(out / f"journal_{strategy.run_id}.jsonl"),
        ctl_factory=lambda _: _NoOpCtl(),
        server_capacities={s.key: 1 for s in slots},
        per_step_writer=per_step_writer_for(out),
        episode_timeout_s=1800,
    )
    launch = {
        **plan,
        "expected": expected,
        "launch_id": launch_id,
        "driver_run_id": driver.run_id,
        "run_id": strategy.run_id,
        "arm_id": a.arm_id,
        "teacher": teacher,
        "replan_steps": 5,
        "tasks": list(tasks),
        "episodes": episodes,
        "env_seed": E.LIBERO_ENV_SEED,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / f"launch_{launch_id}.json").write_text(json.dumps(launch, indent=1))
    thread = threading.Thread(target=driver.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while driver.port is None:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("conductor driver failed to start")
        time.sleep(0.05)
    print(f"[step_diag] libero driver arm={a.arm_id} env={a.env_id} run_id={strategy.run_id} launch={launch_id} "
          f"expected={len(expected)} episodes -> {out}", flush=True)
    gpu_ids = a.gpu_ids.split(",")
    specs = [
        WorkerSpec(
            worker_id=f"w{i}", server_key=s.key, gpu_id=gpu_ids[i % len(gpu_ids)]
        )
        for i, s in enumerate(slots)
    ]

    def spawn(spec, host, port):
        cmd = [
            sys.executable,
            "-m",
            "exp.step_diag.run_libero_diag",
            "--role",
            "worker",
            "--env-id",
            a.env_id,
            "--experiment-id",
            a.experiment_id,
            "--config-sha",
            a.config_sha,
            "--pool",
            a.pool,
            "--worker-id",
            spec.worker_id,
            "--server-key",
            spec.server_key,
            "--driver-host",
            host,
            "--driver-port",
            str(port),
        ]
        # EGL enumerates physical GPUs independently of CUDA_VISIBLE_DEVICES: render on the
        # worker's own slot (the standalone launcher's t % NGPU spread), not always on GPU 0.
        child_env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=spec.gpu_id,
            MUJOCO_EGL_DEVICE_ID=spec.gpu_id,
            MUJOCO_GL="egl",
        )
        return subprocess.Popen(cmd, env=child_env, start_new_session=True)

    agent = WorkerAgent(
        specs, driver_host="127.0.0.1", driver_port=driver.port, spawn_fn=spawn
    )
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    try:
        while thread.is_alive():
            thread.join(5)
    finally:
        agent.stop()
    from exp.robocasa365.run_ws_search import summarize_journal

    summary = summarize_journal(
        out / f"journal_{strategy.run_id}.jsonl",
        expected_uids=[e["task_uid"] for e in expected],
    )
    summary_path = out / f"summary_{launch_id}.json"
    summary_path.write_text(json.dumps({"arm_id": a.arm_id, "env_id": a.env_id, "run_id": strategy.run_id,
                                        "launch_id": launch_id, "driver_run_id": driver.run_id, **summary}, indent=1))
    print(f"[step_diag] {'DONE' if summary['complete'] else 'INCOMPLETE'} arm={a.arm_id} env={a.env_id} "
          f"macro_sr={summary['macro_sr']} n_err={summary['n_err']} n_missing={summary['n_missing']} -> {summary_path}",
          flush=True)
    if not summary["complete"]:
        # Operators chain cells on the exit code; re-running the same command resumes the cell from its journal.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
