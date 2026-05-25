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
import subprocess
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    """How to launch one worker on this machine."""

    worker_id: str
    server_key: str  # bound ServerEndpoint key "host:port"
    gpu_id: str  # value for CUDA_VISIBLE_DEVICES (EGL slot binding)
    worker_module: str = "examples.libero.worker_entry"  # python -m target


class WorkerHandle(Protocol):
    """Minimal process handle the agent supervises (subprocess.Popen satisfies)."""

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...


def _default_spawn(spec: WorkerSpec, driver_host: str, driver_port: int) -> WorkerHandle:
    """Launch a worker as a subprocess pinned to one GPU (EGL slot)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    return subprocess.Popen(
        [
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
        ],
        env=env,
    )


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
            with contextlib.suppress(Exception):
                handle.terminate()

    @property
    def restart_counts(self) -> dict[str, int]:
        return dict(self._restart_counts)
