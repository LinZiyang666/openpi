"""Worker process entry point for RoboCasa365 collection (island A).

Launched by ``WorkerAgent`` (via ``run_collect.py --role agent|all``) as
``<island-A-python> -m exp.robocasa365.worker_entry ...`` with
``cwd=<robocasa-cwd>`` (robocasa resolves its assets relative to the working
directory) and the EGL triple exported. Builds a ``RobocasaEpisodeRunner``
wrapped in the ``WatchdogRunner`` (armed BEFORE ``runner.run()`` so the
unbounded client constructor is covered too) and drives a ``WorkerLoop``.

Imports of robocasa / gymnasium / websockets happen lazily inside the runner's
default collaborators, so this module stays importable in the main venv for
tests.
"""

from __future__ import annotations

import argparse
import socket

from openpi.conductor.worker import WorkerLoop

from exp.robocasa365.episode_runner import (
    ADAPTERS,
    RobocasaEpisodeRunner,
    WatchdogRunner,
)


def build_runner(args: argparse.Namespace) -> WatchdogRunner:
    """Adapter + runner + watchdog from parsed CLI args (test seam)."""
    try:
        adapter_factory = ADAPTERS[args.teacher]
    except KeyError:
        raise SystemExit(f"unknown --teacher {args.teacher!r}; expected one of {sorted(ADAPTERS)}") from None
    runner_cls = RobocasaEpisodeRunner
    if getattr(args, "episode_header_rows", False):
        # Lazy import: the evidence runner exists only for the ws2 search
        # round; the default path must not even load it.
        from exp.robocasa365.ws2_episode_runner import Ws2EpisodeRunner

        runner_cls = Ws2EpisodeRunner
    runner = runner_cls(
        adapter_factory(),
        connect_deadline_s=args.connect_deadline_s,
        connect_retries=args.connect_retries,
        max_cached_envs=None if args.max_cached_envs < 1 else args.max_cached_envs,
    )
    return WatchdogRunner(
        runner,
        episode_deadline_s=args.episode_deadline_s,
        terminate_grace_s=args.terminate_grace_s,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the worker CLI (test seam; ``main`` uses the process argv)."""
    ap = argparse.ArgumentParser(description="conductor RoboCasa365 collection worker")
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--server-key", required=True, help='bound server endpoint "host:port"')
    ap.add_argument("--driver-host", required=True)
    ap.add_argument("--driver-port", type=int, required=True)
    ap.add_argument("--teacher", required=True, choices=sorted(ADAPTERS))
    # Liveness bounds are explicit CLI parameters (no magic defaults hidden in
    # code); run_collect forwards the operator-chosen values.
    ap.add_argument("--connect-deadline-s", type=float, required=True)
    ap.add_argument("--connect-retries", type=int, default=3)
    ap.add_argument(
        "--episode-header-rows",
        action="store_true",
        help="use the ws2 evidence runner: one prompt/seed header row per "
        "episode on the per-step channel. Default OFF keeps every existing "
        "path's per-step row set byte-identical.",
    )
    ap.add_argument(
        "--max-cached-envs",
        type=int,
        default=0,
        help="bound the per-worker kitchen-env cache; <1 keeps the legacy "
        "unbounded cache (safe only where all task kitchens fit one GPU)",
    )
    ap.add_argument("--episode-deadline-s", type=float, required=True)
    ap.add_argument("--terminate-grace-s", type=float, required=True)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    runner = build_runner(args)

    def connect() -> socket.socket:
        return socket.create_connection((args.driver_host, args.driver_port))

    WorkerLoop(args.worker_id, args.server_key, runner, connect=connect).run_forever()


if __name__ == "__main__":
    main()
