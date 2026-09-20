"""Conductor LIBERO shadow driver and worker, using the production episode kernel.

Run with the simulator's Python directly; no conda executable is needed. One worker
per server also supports the single-connection pi0.5 server. All episodes have a
launch, accepted journal record, worker summary and pool content identity.
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
                    "n_env_steps": self._counting.n_steps,
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
    def __init__(self, args, names, pool_sha, launch):
        self.args, self.names, self.pool_sha, self.launch = (
            args,
            names,
            pool_sha,
            launch,
        )
        self.run_id = (
            "sdiag-lib-"
            + E.sha256_json(
                [args.env_id, args.experiment_id, args.config_sha, pool_sha]
            )[:16]
        )
        self.yaml_ids = [f"{self.run_id}_t{i}" for i in range(10)]

    def plan(self, yamls, server_assignment):
        graph = TaskGraph()
        for task_id, yid in enumerate(self.yaml_ids):
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
                        "num_trials_per_task": 10,
                        "seed": 7,
                        "init_pool_sha256": self.pool_sha,
                        "launch_id": self.launch,
                        "arm_id": "shadow",
                        "experiment_id": self.args.experiment_id,
                        "config_sha": self.args.config_sha,
                    },
                )
                for i in range(10)
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
                "env_seed": 7,
                "init_pool_sha256": self.pool_sha,
            }
            for t, y in enumerate(self.yaml_ids)
            for i in range(10)
        ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", choices=("all", "worker"), default="all")
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--config-sha", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--servers", default="")
    ap.add_argument("--out-root", default="exp/step_diag/data/libero")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--worker-id")
    ap.add_argument("--server-key")
    ap.add_argument("--driver-host", default="127.0.0.1")
    ap.add_argument("--driver-port", type=int)
    a = ap.parse_args()
    env = E.resolve_env(a.env_id)
    if not env.benchmark.startswith("libero") or not a.config_sha:
        ap.error("requires a LIBERO environment and configuration digest")
    pool_sha = E.sha256_tree(a.pool)
    if a.role == "worker":
        from examples.libero import main as m
        from examples.libero.worker_entry import _build_episode_setup

        args = m.Args(
            task_suite_name=env.benchmark,
            seed=7,
            replan_steps=5,
            resize_size=224 if env.policy == "pi05" else 256,
        )
        setup = _build_episode_setup(args, 7, a.pool)
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
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[env.benchmark]()
    names = [suite.get_task(i).language for i in range(10)]
    slots = [
        ServerEndpoint(h, int(p))
        for h, p in (s.rsplit(":", 1) for s in a.servers.split(","))
    ]
    if len({s.key for s in slots}) != len(slots):
        ap.error("duplicate servers")
    launch_id = uuid.uuid4().hex
    strategy = LiberoDiagStrategy(a, names, pool_sha, launch_id)
    teacher = "pi05" if env.policy == "pi05" else "groot_tp"
    out = pathlib.Path(a.out_root) / teacher / f"shadow_{a.env_id}"
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
        yaml_weights={y: 10 for y in strategy.yaml_ids},
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
        "arm_id": "shadow",
        "teacher": teacher,
        "replan_steps": 5,
    }
    (out / f"launch_{launch_id}.json").write_text(json.dumps(launch, indent=1))
    thread = threading.Thread(target=driver.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while driver.port is None:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("conductor driver failed to start")
        time.sleep(0.05)
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
    (out / f"summary_{launch_id}.json").write_text(json.dumps(summary, indent=1))
    if not summary["complete"]:
        raise SystemExit("LIBERO shadow incomplete; inspect journal")


if __name__ == "__main__":
    main()
