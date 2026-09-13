"""Worker process entry point (plan §3/§4.2): one GPU/EGL slot, one server.

Launched by ``WorkerAgent`` as ``python -m examples.libero.worker_entry ...``.
Builds a ``LiberoEpisodeRunner`` (real WebSocket client + LIBERO env) and drives
a ``WorkerLoop`` that pulls episodes from the driver and reports results.

This is GPU/LIBERO-only (manual): it lazily imports LIBERO so the module stays
importable in CI, but ``main()`` requires a real environment + a running driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import socket
import sys

from openpi.conductor.worker import WorkerLoop


def parse_init_pools(spec: str, default_suite: str) -> dict[str, str]:
    """``--init-states-dir`` as a suite -> directory map.

    A bare path keeps the historical meaning (one pool, for the worker's own
    suite). ``libero_spatial=/a,libero_10=/b`` binds one pool per suite, which
    is what lets a single resident fleet serve waves of either suite: the
    frozen A-pools are per suite, and a worker that could switch suite but not
    pool would silently draw the wrong 500 inits. An empty spec keeps the
    benchmark's own pool for every suite.
    """
    if not spec:
        return {}
    if "=" not in spec:
        return {default_suite: spec}
    pools: dict[str, str] = {}
    for item in spec.split(","):
        suite, _, path = item.partition("=")
        if not suite or not path:
            raise ValueError(f"malformed init pool binding {item!r}")
        pools[suite] = path
    return pools


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

    The suite comes from each task (``EpisodeTask.experiment``), not from the
    worker's launch flag: a driver that runs libero_spatial waves and then
    libero_10 waves keeps one fleet, and the fleet follows the tasks. The launch
    flag remains the suite a bare ``--init-states-dir`` binds to.
    """
    from libero.libero import benchmark

    from examples.libero import main as m

    pools = parse_init_pools(init_states_dir, args.task_suite_name)
    suites: dict[str, object] = {}

    def suite_for(name: str):
        if name not in suites:
            suites[name] = benchmark.get_benchmark_dict()[name]()
        return suites[name]

    def setup(task):
        task_suite = suite_for(task.experiment)
        libero_task = task_suite.get_task(task.task_id)
        env, task_description = m._get_libero_env(libero_task, m.LIBERO_ENV_RESOLUTION, seed)  # noqa: SLF001
        init_states = m._load_init_states(  # noqa: SLF001
            libero_task, task_suite, task.task_id, pools.get(task.experiment, "")
        )
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


def build_probe(args, init_states_dir: str) -> dict:
    """Describe this worker's environment for the driver's census.

    The values a run's ledger depends on -- the wait/replan protocol, the
    per-suite step limits, and the exact init bytes each suite draws from --
    are hashed here, on the machine that runs the simulator, and attached to
    the worker's first pull. ``libero_code`` is folded into one digest so the
    payload stays small; the file list is reproducible from the same tree.
    """
    import numpy as np
    import robosuite
    import torch
    import libero

    from examples.libero import main as m

    libero_root = pathlib.Path(libero.__file__).resolve().parent
    code = [
        (str(path.relative_to(libero_root)), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(libero_root.rglob("*.py"))
    ]
    pools = parse_init_pools(init_states_dir, args.task_suite_name)
    per_task = {}
    for suite, directory in sorted(pools.items()):
        base = pathlib.Path(directory)
        if list(base.glob("*.pruned_init")):
            raise ValueError(f"{base}: unexpected .pruned_init override in the init pool")
        per_task[suite] = {
            path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(base.glob("*.init"))
        }
    return {
        "hostname": socket.gethostname(),
        "num_steps_wait": int(args.num_steps_wait),
        "replan_steps": int(args.replan_steps),
        "resize_size": int(args.resize_size),
        "max_steps": {s: m._get_max_steps(s) for s in ("libero_spatial", "libero_10")},  # noqa: SLF001
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "libero_file": str(libero_root / "__init__.py"),
        "robosuite_version": getattr(robosuite, "__version__", "unknown"),
        "libero_code_digest": hashlib.sha256(
            json.dumps(code, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "init_pools": dict(sorted(pools.items())),
        "per_task_digests": per_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="conductor LIBERO worker")
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--server-key", required=True, help='bound server endpoint "host:port"')
    ap.add_argument("--driver-host", required=True)
    ap.add_argument("--driver-port", type=int, required=True)
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument(
        "--init-states-dir",
        default="",
        help="one pool directory for --task-suite-name, or suite=dir[,suite=dir] "
        "to bind a pool per suite the fleet may be asked to run",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="hash this environment and the bound init pools and send the result on "
        "the first pull, so a driver on another machine can attest the client side",
    )
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

    probe = build_probe(args, a.init_states_dir) if a.probe else None
    WorkerLoop(a.worker_id, a.server_key, runner, connect=connect, probe=probe).run_forever()


if __name__ == "__main__":
    main()
