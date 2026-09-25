"""Dispatch frozen warm reset plans through the standard conductor and workers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from pathlib import Path

from exp.step_diag.envs import ENVS
from openpi.conductor import ConductorDriver
from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.task import (
    EpisodeTask,
    ServerEndpoint,
    Stage,
    TaskGraph,
    make_task_uid,
)


def manifest_digest(manifest: dict) -> str:
    """Bind all task, rollout, arm and endpoint facts to the driver execution."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()


class WarmResetStrategy(ExperimentStrategy):
    """One eval stage per arm, with unchanged episode identity across all arms."""

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.arms = {a["yaml_id"]: a for a in manifest["arms"]}
        self.tasks: list[EpisodeTask] = []

    def plan(self, yamls, server_assignment):
        """Build the actual dispatched graph and retain its trusted episode roster."""
        graph = TaskGraph()
        self.tasks = []
        env = ENVS[self.manifest["env_id"]]
        rollout = self.manifest["rollout"]
        for yaml_id in yamls:
            server = server_assignment[yaml_id]
            episodes = []
            for task in self.manifest["tasks"]:
                extra = {"num_trials_per_task": len(task["init_indices"])}
                experiment = env.benchmark
                if env.benchmark == "robocasa365":
                    experiment = f"warm_reset_{self.manifest['token']}"
                    extra.update(
                        task_name=task["name"],
                        teacher="groot_tp" if env.policy == "groot" else "pi05",
                        **{
                            k: rollout[k]
                            for k in ("base_seed", "layout", "style", "replan_steps")
                        },
                    )
                for ep, original in enumerate(task["init_indices"]):
                    episodes.append(
                        EpisodeTask(
                            task_uid=make_task_uid(
                                yaml_id, "eval", task["task_id"], ep
                            ),
                            yaml_id=yaml_id,
                            phase="eval",
                            experiment=experiment,
                            task_id=task["task_id"],
                            episode_idx=ep,
                            orig_init_state_idx=original,
                            server_host=server.host,
                            server_port=server.port,
                            bundle_id=yaml_id,
                            extra=dict(extra),
                        )
                    )
            graph.add_stage(
                Stage(f"{yaml_id}:eval", yaml_id, "eval", server, episodes=episodes)
            )
            self.tasks.extend(episodes)
        return graph

    def on_stage_begin(self, stage, ctl, ctx):
        """Hot-load the exact frozen YAML into this stage's named bundle."""
        ctl.load_cache_config(
            yaml_content=self.arms[stage.yaml_id]["yaml"], yaml_id=stage.yaml_id
        )


def build_driver(
    root: Path,
    manifest: dict,
    *,
    bind_host: str,
    port: int,
    concurrency: int,
    ctl_factory=None,
    episode_timeout_s: float = 1800,
):
    """Claim a fresh execution and persist its graph/run identity before dispatch."""
    if concurrency < 1 or episode_timeout_s <= 0:
        raise ValueError("concurrency and episode timeout must be positive")
    # No resume across driver processes: attempt numbers restart there. A fresh
    # plan/token is required after a crash, preventing ambiguous server evidence.
    claim = root / "execution.json"
    with claim.open("x") as fh:
        fh.write("{}\n")
    if (root / "journal.jsonl").exists() or (root / "per_step.jsonl").exists():
        raise ValueError("fresh execution contains old evidence")
    lock = threading.Lock()

    def write_rows(yaml_id, rows):
        del yaml_id
        with lock, (root / "per_step.jsonl").open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row, allow_nan=False) + "\n")
            fh.flush()

    clients = []
    if ctl_factory is None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        def ctl_factory(endpoint):
            client = WebsocketClientPolicy(host=endpoint.host, port=endpoint.port)
            clients.append(client)
            return client

    strategy = WarmResetStrategy(manifest)
    driver = ConductorDriver(
        strategy,
        yaml_weights={a["yaml_id"]: 1.0 for a in manifest["arms"]},
        servers=[ServerEndpoint(**s) for s in manifest["servers"]],
        journal_path=str(root / "journal.jsonl"),
        ctl_factory=ctl_factory,
        bind_host=bind_host,
        bind_port=port,
        scheduler_kwargs={"eval_concurrency": concurrency},
        episode_timeout_s=episode_timeout_s,
        per_step_writer=write_rows,
    )
    record = {
        "run_id": driver.run_id,
        "token": manifest["token"],
        "manifest_sha256": manifest_digest(manifest),
        "tasks": [dataclasses.asdict(t) for t in strategy.tasks],
    }
    claim.write_text(json.dumps(record, indent=2) + "\n")
    return driver, clients


def worker_agent(
    manifest: dict,
    *,
    server: str,
    driver_host: str,
    driver_port: int,
    gpus: list[str],
    workers_per_gpu: int,
    prefix: str,
    conda_env: str = "",
    rc_options: dict | None = None,
):
    """Build standard LIBERO or RoboCasa workers with the plan's rollout knobs."""
    from functools import partial

    from openpi.conductor.agent import WorkerAgent, WorkerSpec

    env = ENVS[manifest["env_id"]]
    allowed = {f"{s['host']}:{s['port']}" for s in manifest["servers"]}
    if (
        server not in allowed
        or workers_per_gpu < 1
        or not gpus
        or not all(gpus)
        or not prefix
    ):
        raise ValueError("invalid worker server, GPU list, count or prefix")
    rollout = manifest["rollout"]
    specs = [
        WorkerSpec(
            worker_id=f"{prefix}-{i}-{j}",
            server_key=server,
            gpu_id=gpu,
            conda_env=conda_env,
            task_suite_name=env.benchmark,
            init_states_dir=rollout.get("init_states_dir", ""),
            resize_size=256 if env.policy == "groot" else 224,
            replan_steps=rollout["replan_steps"],
            seed=rollout.get("seed"),
            # EGL picks its render device from this variable, and robosuite
            # asserts it names a device inside CUDA_VISIBLE_DEVICES (= gpu).
            env={"MUJOCO_EGL_DEVICE_ID": gpu},
        )
        for i, gpu in enumerate(gpus)
        for j in range(workers_per_gpu)
    ]
    if env.benchmark == "robocasa365":
        from exp.robocasa365.run_collect import robocasa_spawn_fn

        if not rc_options or any(
            not rc_options.get(k)
            for k in ("worker_python", "robocasa_cwd", "egl_lib_dir", "egl_vendor_dir")
        ):
            raise ValueError(
                "RoboCasa requires worker Python, cwd and EGL library/vendor paths"
            )
        spawn = partial(
            robocasa_spawn_fn,
            repo_root=str(Path(__file__).resolve().parents[2]),
            teacher="groot_tp" if env.policy == "groot" else "pi05",
            **rc_options,
        )
        return WorkerAgent(specs, driver_host, driver_port, spawn_fn=spawn)
    return WorkerAgent(specs, driver_host, driver_port)
