"""Worker process entry point (plan §3/§4.2): one GPU/EGL slot, one server.

Launched by ``WorkerAgent`` as ``python -m examples.libero.worker_entry ...``.
Builds a ``LiberoEpisodeRunner`` (real WebSocket client + LIBERO env) and drives
a ``WorkerLoop`` that pulls episodes from the driver and reports results.

This is GPU/LIBERO-only (manual): it lazily imports LIBERO so the module stays
importable in CI, but ``main()`` requires a real environment + a running driver.
"""

from __future__ import annotations

import argparse
import socket

from openpi.conductor.worker import WorkerLoop


def _init_state_index(task, mode: str) -> int:
    if mode == "orig":
        return int(task.orig_init_state_idx)
    if mode == "subset":
        return int(task.episode_idx)
    raise ValueError(f"unknown init-state index mode {mode!r}")


def _build_episode_setup(args, seed: int, init_states_dir: str, index_mode: str = "orig"):
    """Build the (env, initial_state, task_description, max_steps) provider.

    Uses the existing helpers in ``examples.libero.main`` and the standard
    LIBERO benchmark API (``main.py`` imports ``from libero.libero import
    benchmark``). Lazy so importing this module needs no LIBERO/CUDA.
    """
    from libero.libero import benchmark

    from examples.libero import main as m

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()

    def setup(task):
        libero_task = task_suite.get_task(task.task_id)
        env, task_description = m._get_libero_env(libero_task, m.LIBERO_ENV_RESOLUTION, seed)  # noqa: SLF001
        init_states = m._load_init_states(libero_task, task_suite, task.task_id, init_states_dir)  # noqa: SLF001
        index = _init_state_index(task, index_mode)
        if index < 0 or index >= len(init_states):
            raise IndexError(
                f"task {task.task_uid}: {index_mode} init index {index} outside "
                f"materialised pool of {len(init_states)} states"
            )
        initial_state = init_states[index]
        max_steps = m._get_max_steps(task.experiment)  # noqa: SLF001
        return env, initial_state, task_description, max_steps

    return setup


def main() -> None:
    ap = argparse.ArgumentParser(description="conductor LIBERO worker")
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--server-key", required=True, help='bound server endpoint "host:port"')
    ap.add_argument("--driver-host", required=True)
    ap.add_argument("--driver-port", type=int, required=True)
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument("--init-states-dir", default="")
    ap.add_argument("--init-state-index-mode", choices=("orig", "subset"), default="orig")
    ap.add_argument("--seed", type=int, default=7)
    # Both default to main.Args' values, which are the Pi0.5 LIBERO convention.
    # A GR00T checkpoint needs --resize-size 256: the official evaluator feeds
    # the raw render and lets the transform chain crop to 224, so the 224
    # default would crop twice and change the field of view. The wire contract
    # rejects a 224 frame outright, but only after the fleet is already up.
    ap.add_argument("--resize-size", type=int, default=None)
    ap.add_argument("--replan-steps", type=int, default=None)
    a = ap.parse_args()

    from examples.libero import main as m
    from examples.libero.episode_runner import LiberoEpisodeRunner

    overrides = {}
    if a.resize_size is not None:
        overrides["resize_size"] = a.resize_size
    if a.replan_steps is not None:
        overrides["replan_steps"] = a.replan_steps
    args = m.Args(task_suite_name=a.task_suite_name, seed=a.seed, **overrides)
    runner = LiberoEpisodeRunner(
        args,
        _build_episode_setup(a, a.seed, a.init_states_dir, a.init_state_index_mode),
    )

    def connect():
        return socket.create_connection((a.driver_host, a.driver_port))

    WorkerLoop(a.worker_id, a.server_key, runner, connect=connect).run_forever()


if __name__ == "__main__":
    main()
