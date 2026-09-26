"""pi0.5 x MetaWorld MT50 as an environment of the warm reset experiment entry.

Registered at import time through the entry's plugin hook
(``exp.warm_reset.envs``: ``exp/<package>/warm_reset_env.py`` is discovered by
convention), so ``python -m exp.warm_reset.run {tasks,prepare,run,agent,admit,
analysis} --env pi05_metaworld`` works without touching the entry package.

* ``ENV`` -- ``EntryEnv("pi05_metaworld")``: policy pi0.5, benchmark
  ``metaworld_mt50``, action horizon 5 (the checkpoint's; replan bound), K = 10
  on schedule ``pi05_v1``; ``--arms all`` = ``ARMS``.
* ``ARMS`` -- the six arms of the owner's 2026-09-25 decision (no cache library,
  no collection): ``full`` (K = 10), ``plain_k2``, ``plain_k1`` (per-bundle
  ``miss`` blocks) and the three self warm reset arms at start_t 0.2 (library
  free with ``prepare --self-trigger always``).
* ``MetaworldAdapter`` -- task export from the MT50 table (``--task-names`` or
  ``--task-ids``; ``init_indices`` are ``MT1.train_tasks`` indices), the fixed
  rollout (bench seed 7, replan 5, no init pool), episode experiment
  ``metaworld_mt50`` with the task name stamped into ``extra`` (the worker
  cross-checks it against ``task_id``), the trusted identity key ``seed``, the
  pairing identity ``(task, idx, bench seed)`` and the worker launch
  (``spawn_worker``: ``exp.metaworld.worker_entry`` in the simulator venv).

Import-light by the hook's contract: no simulator or model import here.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
import subprocess
from typing import Any

from exp.metaworld import tasks as T
from exp.warm_reset.envs import BenchmarkAdapter, EntryEnv, register_env

ENV_ID = "pi05_metaworld"
ARMS = (
    "full",
    "plain_k2",
    "plain_k1",
    "selfwarmreset_t0.2",
    "selfresetfinal_t0.2",
    "selfmidfinal_t0.2",
)
EPISODES_PER_TASK = T.EPISODES_PER_TASK
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_WORKER_PYTHON = str(pathlib.Path("~/metaworld_sim/bin/python").expanduser())


# ------------------------------------------------------------------
# Worker launch
# ------------------------------------------------------------------


def worker_command(
    spec, driver_host: str, driver_port: int, *, worker_python: str
) -> list[str]:
    """``exp.metaworld.worker_entry`` argv for one ``WorkerSpec``."""
    cmd = [
        worker_python,
        "-m",
        "exp.metaworld.worker_entry",
        "--worker-id",
        spec.worker_id,
        "--server-key",
        spec.server_key,
        "--driver-host",
        driver_host,
        "--driver-port",
        str(driver_port),
    ]
    if spec.seed is not None:
        cmd += ["--seed", str(spec.seed)]
    if spec.replan_steps is not None:
        cmd += ["--replan-steps", str(spec.replan_steps)]
    return cmd


def worker_environ(spec, *, repo_root: str) -> dict[str, str]:
    """Process environment: the simulator venv's own packages plus the repo, EGL on ``spec.gpu_id``."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
    }
    env["PYTHONPATH"] = os.pathsep.join((repo_root, os.path.join(repo_root, "src")))
    env["MUJOCO_GL"] = "egl"
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    # MuJoCo EGL picks its render device from MUJOCO_EGL_DEVICE_ID; default it to
    # the worker's GPU so it always lies inside CUDA_VISIBLE_DEVICES.
    env["MUJOCO_EGL_DEVICE_ID"] = spec.gpu_id
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.update(spec.env)
    return env


def spawn_worker(
    spec,
    driver_host: str,
    driver_port: int,
    *,
    worker_python: str = DEFAULT_WORKER_PYTHON,
    repo_root: str = str(REPO_ROOT),
) -> subprocess.Popen:
    """``WorkerAgent`` spawn function for MetaWorld workers.

    ``start_new_session=True`` is required: ``WorkerAgent.stop()`` signals the
    worker's process group, which would otherwise be the agent's own.
    """
    return subprocess.Popen(
        worker_command(spec, driver_host, driver_port, worker_python=worker_python),
        env=worker_environ(spec, repo_root=repo_root),
        start_new_session=True,
    )


# ------------------------------------------------------------------
# Entry adapter and registration
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MetaworldAdapter(BenchmarkAdapter):
    """MetaWorld MT50 as seen by the warm reset entry (see module docstring).

    A field-less frozen dataclass, so two instances compare equal and a
    re-import re-registers the same environment without conflict.
    """

    name = "metaworld"

    def validate_rollout(self, env: EntryEnv, rollout: dict) -> None:
        """MetaWorld runs the bench seed and the checkpoint's 5-step open-loop chunk only."""
        if rollout.get("seed") != T.BENCH_SEED:
            raise ValueError(
                f"MetaWorld requires rollout seed {T.BENCH_SEED} (the MT1 bench seed)"
            )
        if rollout.get("replan_steps") != T.REPLAN_STEPS:
            raise ValueError(f"MetaWorld requires replan_steps {T.REPLAN_STEPS}")
        if rollout.get("init_states_dir"):
            raise ValueError(
                "MetaWorld has no init-state pool; idx selects MT1.train_tasks"
            )

    def export_tasks(
        self,
        env: EntryEnv,
        *,
        task_ids: str,
        task_names: str,
        episodes: int,
        init_offset: int,
    ) -> list[dict]:
        """MT50 names from ``--task-names``, else ``--task-ids`` (``all`` = the 50 in table order)."""
        names = [n.strip() for n in task_names.split(",") if n.strip()]
        if not names and task_ids != "all":
            names = [T.TASK_NAMES[int(i)] for i in task_ids.split(",")]
        return T.export_tasks(names or None, episodes, init_offset)

    def experiment(self, env: EntryEnv, manifest: dict) -> str:
        """The benchmark id: self-start noise is common across runs of the same identity."""
        return env.benchmark

    def episode_extra(self, env: EntryEnv, manifest: dict, task: dict) -> dict:
        """The task name, after checking the roster uses MT50's canonical ``task_id``."""
        if T.task_id_of(task["name"]) != task["task_id"]:
            raise ValueError(
                f"task {task['name']!r} has MT50 task_id {T.task_id_of(task['name'])}, "
                f"not {task['task_id']}"
            )
        return {"task_name": task["name"]}

    def identity(self, env: EntryEnv, manifest: dict, episode: Any) -> dict:
        """The worker's episode_start ``seed`` (the MT1 bench seed of the plan)."""
        return {"seed": manifest["rollout"]["seed"]}

    def pairing(self, env: EntryEnv, manifest: dict, episode: Any) -> dict:
        """``(idx, bench seed)``; the entry adds the task name."""
        return {
            "init_idx": episode.orig_init_state_idx,
            "env_seed": manifest["rollout"]["seed"],
        }

    def worker_agent(
        self,
        env: EntryEnv,
        manifest: dict,
        *,
        specs: list,
        driver_host: str,
        driver_port: int,
        options: dict,
    ) -> Any:
        """``WorkerAgent`` spawning ``spawn_worker`` (``--worker-python`` overrides the venv)."""
        from openpi.conductor.agent import WorkerAgent

        if options.get("conda_env") or options.get("pinned_objects"):
            raise ValueError(
                "MetaWorld workers take --worker-python, not --conda-env / --pinned-objects"
            )
        python = (options.get("rc") or {}).get("worker_python") or DEFAULT_WORKER_PYTHON
        spawn = functools.partial(spawn_worker, worker_python=python)
        return WorkerAgent(specs, driver_host, driver_port, spawn_fn=spawn)


ENV = register_env(
    EntryEnv(
        env_id=ENV_ID,
        policy="pi05",
        benchmark=T.BENCHMARK,
        action_horizon=5,
        k_full=10,
        schedule_id="pi05_v1",
        adapter=MetaworldAdapter(),
        default_arms=ARMS,
    )
)
