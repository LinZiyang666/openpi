"""MetaWorld conductor worker process: one EGL slot, one inference server.

Launched by ``WorkerAgent`` through ``exp.metaworld.warm_reset_env.spawn_worker``
inside the simulator venv (``~/metaworld_sim``: Python 3.11, metaworld 3.0.0,
mujoco, openpi-client) with ``PYTHONPATH=<repo>:<repo>/src``. Builds a
``MetaworldEpisodeRunner`` and drives a ``WorkerLoop`` that pulls episodes from
the driver and reports results. ``metaworld`` is imported only when an episode
builds its environment.
"""

from __future__ import annotations

import argparse
import os
import socket

from exp.metaworld import tasks as T


def parser() -> argparse.ArgumentParser:
    """Worker command line (the agent-side spawn function builds it)."""
    ap = argparse.ArgumentParser(description="conductor MetaWorld MT50 worker")
    ap.add_argument("--worker-id", required=True)
    ap.add_argument(
        "--server-key", required=True, help='bound server endpoint "host:port"'
    )
    ap.add_argument("--driver-host", required=True)
    ap.add_argument("--driver-port", type=int, required=True)
    ap.add_argument("--seed", type=int, default=T.BENCH_SEED, help="metaworld.MT1 seed")
    ap.add_argument("--replan-steps", type=int, default=T.REPLAN_STEPS)
    return ap


def main(argv: list[str] | None = None) -> None:
    """Pull and run episodes until the driver sends shutdown."""
    a = parser().parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")

    from exp.metaworld.episode_runner import MetaworldEpisodeRunner
    from openpi.conductor.worker import WorkerLoop

    runner = MetaworldEpisodeRunner(seed=a.seed, replan_steps=a.replan_steps)

    def connect():
        return socket.create_connection((a.driver_host, a.driver_port))

    WorkerLoop(a.worker_id, a.server_key, runner, connect=connect).run_forever()


if __name__ == "__main__":
    main()
