"""Per-machine worker agent: fork + supervise + restart workers (plan §3, §9.2).

One ``WorkerAgent`` runs (resident) on each client machine. It forks the
machine's worker processes — each bound to one GPU / EGL slot — and supervises
them: a worker that dies is respawned. Workers connect directly to the driver's
pull port; the agent owns their lifecycle on this host. If the agent itself
dies, the driver sees the worker connections drop and requeues their in-flight
episodes (driver ``_handle_conn``), so no episode is lost.

The spawn function is injectable so the supervision logic is unit-testable
without real subprocesses.

Coupling map:
  DEPENDS ON:  standard library (subprocess, threading)
  CONSUMED BY: conductor entry scripts (one agent process per machine)
  IF CHANGED:  worker process launch contract
  NOTE:        MUST NOT import exp.* — the worker entry module is named by spec.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import logging
import os
import signal
import subprocess
import threading
from typing import Literal, Protocol

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    """How to launch one worker on this machine."""

    worker_id: str
    server_key: str  # bound ServerEndpoint key "host:port"
    gpu_id: str  # value for CUDA_VISIBLE_DEVICES (EGL slot binding)
    worker_module: str = "examples.libero.worker_entry"  # python -m target
    # When set, the worker runs via ``conda run -p <conda_env>`` so it uses a
    # separate interpreter (e.g. the LIBERO sim env that has libero +
    # openpi_client) while the driver keeps its own (uv) interpreter. Empty =
    # spawn with the agent's own ``python`` (backward-compatible default).
    conda_env: str = ""
    # LIBERO benchmark suite the worker's env loads. MUST match the library /
    # strategy suite, otherwise the worker runs the wrong tasks and every cache
    # query's task_key mismatches the library (0 candidates -> all MISS). The
    # default preserves backward-compat with the original libero_spatial runs;
    # callers (run_phase2) forward their --task-suite here.
    task_suite_name: str = "libero_spatial"
    # Init-state pool the worker draws from, forwarded to
    # ``examples.libero.worker_entry --init-states-dir``. Empty (the default)
    # keeps the LIBERO benchmark's own pool, which is what every caller before
    # markov_sufficiency relied on. Set it to evaluate on a second, disjoint
    # pool -- e.g. ``exp/common/data/db_init/libero/<suite>`` -- without
    # rebuilding the benchmark.
    init_states_dir: str = ""
    # Which EpisodeTask identity selects an entry inside ``init_states_dir``.
    # "orig" preserves the historical contract for a full official pool;
    # "subset" is required when the directory itself is a materialised subset
    # while ``orig_init_state_idx`` remains the official provenance label.
    init_state_index_mode: Literal["orig", "subset"] = "orig"
    # Client-side rollout knobs, forwarded to ``worker_entry`` only when set so
    # a caller that omits them keeps the argv byte-identical. They live here
    # rather than in a fork of the spawn path because they are the same kind of
    # per-run fact as ``task_suite_name``: get them wrong and the fleet comes up
    # and then fails every episode. A GR00T checkpoint needs ``resize_size=256``
    # -- the official evaluator feeds the raw render and lets the transform
    # chain crop to 224, so main.Args' 224 default crops twice and changes the
    # field of view; the wire contract rejects the frame, but only once every
    # worker is already running.
    resize_size: int | None = None
    replan_steps: int | None = None
    seed: int | None = None
    # Extra process environment for this worker (merged last, so it wins).
    # ``MUJOCO_EGL_DEVICE_ID`` belongs here: ``CUDA_VISIBLE_DEVICES`` steers the
    # policy client, while EGL picks its render device independently.
    env: dict[str, str] = dataclasses.field(default_factory=dict)


class WorkerHandle(Protocol):
    """Minimal process handle the agent supervises (subprocess.Popen satisfies)."""

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...


# Repo layout: this file is <root>/src/openpi/conductor/agent.py. A worker that
# runs in a separate conda env (the LIBERO sim env has libero + openpi_client but
# NOT openpi) needs the repo's src + root on PYTHONPATH to import openpi.conductor
# and examples.libero.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(_SRC_DIR)


def _default_spawn(spec: WorkerSpec, driver_host: str, driver_port: int) -> WorkerHandle:
    """Launch a worker as a subprocess pinned to one GPU (EGL slot).

    When ``spec.conda_env`` is set, the worker runs via ``conda run -p <env>``
    (mirroring the legacy ``run_phase.py --conda-env`` path) and the repo src +
    root are injected onto PYTHONPATH so the conda interpreter can import
    openpi.conductor + examples.libero even though openpi is not installed there.
    """
    base_cmd = [
        "python",
        "-m",
        spec.worker_module,
        "--worker-id",
        spec.worker_id,
        "--server-key",
        spec.server_key,
        "--driver-host",
        driver_host,
        "--driver-port",
        str(driver_port),
        "--task-suite-name",
        spec.task_suite_name,
    ]
    if spec.init_states_dir:
        # Only appended when set, so a worker launched by an older caller keeps
        # the default LIBERO pool and the argv stays byte-identical.
        base_cmd += ["--init-states-dir", spec.init_states_dir]
    if spec.init_state_index_mode != "orig":
        base_cmd += ["--init-state-index-mode", spec.init_state_index_mode]
    if spec.resize_size is not None:
        base_cmd += ["--resize-size", str(spec.resize_size)]
    if spec.replan_steps is not None:
        base_cmd += ["--replan-steps", str(spec.replan_steps)]
    if spec.seed is not None:
        base_cmd += ["--seed", str(spec.seed)]
    if spec.conda_env:
        # Mirror legacy build_subprocess_cmd: strip the driver's uv-venv env
        # injections (VIRTUAL_ENV / PYTHONPATH / PYTHONHOME) and drop the uv venv
        # bin from PATH so the conda interpreter's own deps win; set MUJOCO_GL=egl
        # for headless LIBERO rendering; put repo root + src on PYTHONPATH so the
        # conda interpreter imports openpi.conductor + examples.libero (libero +
        # openpi_client come from the conda env's own site-packages).
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
        }
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv:
            venv_bin = os.path.join(venv, "bin")
            env["PATH"] = os.pathsep.join(
                p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin
            )
        env["MUJOCO_GL"] = "egl"
        env["PYTHONPATH"] = os.pathsep.join((_REPO_ROOT, _SRC_DIR))
        env_flag = "-p" if ("/" in spec.conda_env or os.path.isabs(spec.conda_env)) else "-n"
        cmd = ["conda", "run", "--no-capture-output", env_flag, spec.conda_env, *base_cmd]
    else:
        env = dict(os.environ)
        cmd = base_cmd
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    # glibc malloc tuning for the LIBERO/MuJoCo sim workers: cap per-thread
    # arenas and return freed memory to the OS past a threshold. MuJoCo render
    # workers otherwise hoard freed RAM in glibc's per-thread arenas, so a
    # worker's resident set climbs ~3.4G over a run; with these it plateaus at
    # ~1.6G (measured libero_10 Stage 2). Pure memory management -- no effect on
    # the simulation computation or eval correctness. setdefault so an explicit
    # caller export still wins.
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
    # Caller-supplied last: an explicit per-worker value (e.g. the EGL render
    # device) is a deliberate choice and must not be overridden by the defaults
    # above or by the inherited environment.
    env.update(spec.env)
    # start_new_session=True puts the worker (and, under `conda run`, the real
    # grandchild process) in its own process group so stop() can signal the
    # whole group rather than only the wrapper.
    return subprocess.Popen(cmd, env=env, start_new_session=True)


class WorkerAgent:
    """Forks and supervises this machine's workers; respawns the dead ones."""

    def __init__(
        self,
        specs: list[WorkerSpec],
        driver_host: str,
        driver_port: int,
        *,
        spawn_fn: Callable[[WorkerSpec, str, int], WorkerHandle] = _default_spawn,
        poll_s: float = 1.0,
    ) -> None:
        self._specs = specs
        self._driver_host = driver_host
        self._driver_port = driver_port
        self._spawn_fn = spawn_fn
        self._poll_s = poll_s
        self._handles: dict[str, WorkerHandle] = {}
        self._restart_counts: dict[str, int] = {}
        self._stop = threading.Event()

    def _spawn(self, spec: WorkerSpec) -> None:
        logger.info("agent: spawning worker %s on GPU %s", spec.worker_id, spec.gpu_id)
        self._handles[spec.worker_id] = self._spawn_fn(spec, self._driver_host, self._driver_port)

    def start(self) -> None:
        for spec in self._specs:
            self._spawn(spec)

    def supervise_once(self) -> None:
        """One supervision pass: respawn any worker whose process has exited."""
        if self._stop.is_set():
            return
        for spec in self._specs:
            handle = self._handles.get(spec.worker_id)
            if handle is None or handle.poll() is not None:
                if handle is not None:
                    # Reap the exited process so it does not linger as a zombie
                    # (poll() already saw it exit, so wait() returns at once).
                    wait_fn = getattr(handle, "wait", None)
                    if wait_fn is not None:
                        with contextlib.suppress(Exception):
                            wait_fn(timeout=5)
                self._restart_counts[spec.worker_id] = self._restart_counts.get(spec.worker_id, 0) + 1
                logger.warning(
                    "agent: worker %s died; restart #%d", spec.worker_id, self._restart_counts[spec.worker_id]
                )
                self._spawn(spec)

    def run(self) -> None:
        """Resident supervision loop until ``stop`` is called."""
        self.start()
        while not self._stop.is_set():
            self.supervise_once()
            self._stop.wait(self._poll_s)

    def stop(self) -> None:
        self._stop.set()
        for handle in self._handles.values():
            # Workers are spawned with start_new_session=True, so signalling the
            # process group also reaches the real worker behind a `conda run`
            # wrapper; terminate() alone would only hit the wrapper and orphan
            # the grandchild that holds the GPU/EGL slot.
            pid = getattr(handle, "pid", None)
            if isinstance(pid, int):
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            with contextlib.suppress(Exception):
                handle.terminate()

    @property
    def restart_counts(self) -> dict[str, int]:
        return dict(self._restart_counts)
