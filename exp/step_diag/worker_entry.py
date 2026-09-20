"""Worker entry of the step-vs-warm-start line: the production RoboCasa runner plus evidence (plan §3.2-2).

``StepDiagEpisodeRunner`` delegates the whole rollout to ``RobocasaEpisodeRunner`` (nothing of the
loop is copied) and adds three things around it:

* a **client proxy** that counts every ``infer`` (the worker-side decision index), keeps the
  ``__hit_meta__`` of every response and stamps the arm identity of the dispatched task
  (``launch_id / arm_id / experiment_id / config_sha / lane``) into ``episode_start``'s
  ``extra_metadata`` — the production runner forwards only its own provenance fields, and the
  server-side recorder joins on these; ``episode_end`` keeps its existing ``success=`` signature;
* a **counting env proxy** injected through the runner's ``gym_make`` seam, so the real number of
  ``env.step`` calls after the reset (including a successful last step) is recorded as
  ``n_env_steps`` — the runner's own ``n_steps`` is not incremented on the success step;
* one ``episode_summary`` row appended to ``EpisodeResult.per_step_rows`` after the runner returns
  (``reported_n_steps``, ``n_env_steps``, ``n_decisions``, the hit-meta list, the environment
  identity and the stamp), which the driver's ``per_step_writer`` persists next to the journal.

``step_diag_spawn_fn`` mirrors ``run_collect.robocasa_spawn_fn`` (same argv contract, same worker
environment) with this module as the entry point; a test pins the two against each other.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import functools
import hashlib
import pathlib
import platform
import importlib.metadata
from typing import Any, Callable

from openpi.conductor import WorkerSpec
from openpi.conductor import task as _task
from openpi.conductor.worker import WorkerLoop

from exp.robocasa365.episode_runner import (
    ADAPTERS,
    RobocasaEpisodeRunner,
    WatchdogRunner,
    default_client_factory,
    default_gym_make,
)
from exp.robocasa365.worker_entry import parse_args

# Keys copied from ``EpisodeTask.extra`` into ``episode_start.extra_metadata``.
STAMP_KEYS = ("launch_id", "arm_id", "experiment_id", "config_sha", "lane", "layout", "style")
# Sentinel step index of the summary row (real decision rows use step_idx >= 0, ws2 headers -1).
SUMMARY_STEP_IDX = -2


@functools.lru_cache(maxsize=1)
def worker_runtime() -> dict:
    """Actual simulator process identity; computed once in each worker process."""
    versions = {}
    for name in ("numpy", "torch", "mujoco", "robosuite", "robocasa", "libero"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = pathlib.Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in ("examples/libero", "exp/robocasa365", "exp/step_diag"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return {"host": socket.gethostname(), "python": platform.python_version(), "versions": versions,
            "gpu_slot": os.environ.get("CUDA_VISIBLE_DEVICES"), "source_sha256": digest.hexdigest()}


class _CountingEnv:
    """Transparent env proxy counting ``step`` calls since the last ``reset``."""

    def __init__(self, env: Any, on_reset: Callable[["_CountingEnv"], None]) -> None:
        self._env = env
        self._on_reset = on_reset
        self.n_steps = 0

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self.n_steps = 0
        self._on_reset(self)
        return self._env.reset(*args, **kwargs)

    def step(self, action: Any) -> Any:
        self.n_steps += 1
        return self._env.step(action)

    def close(self) -> Any:
        return self._env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _ClientProxy:
    """Transparent policy-client proxy: infer counter, hit-meta capture, identity stamp."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.stamp: dict[str, Any] = {}
        self.n_infer = 0
        self.hit_meta: list[dict[str, Any]] = []

    def episode_start(self, **kwargs: Any) -> Any:
        extra = dict(kwargs.get("extra_metadata") or {})
        extra.update(self.stamp)
        kwargs["extra_metadata"] = extra
        self.n_infer = 0
        self.hit_meta = []
        return self._inner.episode_start(**kwargs)

    def infer(self, obs: Any) -> Any:
        response = self._inner.infer(obs)
        idx = self.n_infer
        self.n_infer += 1
        meta = response.get("__hit_meta__") if isinstance(response, dict) else None
        if meta is not None:
            self.hit_meta.append({"decision_idx": idx, **{k: v for k, v in dict(meta).items() if _is_scalar(v)}})
        return response

    def episode_end(self, **kwargs: Any) -> Any:
        return self._inner.episode_end(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


class StepDiagEpisodeRunner(RobocasaEpisodeRunner):
    """RobocasaEpisodeRunner + client proxy + counting env + one summary row per episode."""

    def __init__(
        self,
        adapter: Any,
        *,
        client_factory: Callable[[_task.ServerEndpoint], Any] = default_client_factory,
        gym_make: Callable[..., Any] = default_gym_make,
        **kwargs: Any,
    ) -> None:
        self._active_env: _CountingEnv | None = None

        def _make(*args: Any, **kw: Any) -> _CountingEnv:
            return _CountingEnv(gym_make(*args, **kw), self._note_reset)

        super().__init__(adapter, client_factory=lambda server: _ClientProxy(client_factory(server)),
                         gym_make=_make, **kwargs)

    def _note_reset(self, env: _CountingEnv) -> None:
        self._active_env = env

    def run(self, task: _task.EpisodeTask, report: Any) -> _task.EpisodeResult:
        extra = task.extra or {}
        stamp = {k: extra[k] for k in STAMP_KEYS if k in extra}
        client = self._ensure_client(task)  # same proxy the base run() will reuse
        client.stamp = stamp
        self._active_env = None
        try:
            result = super().run(task, report)
        finally:
            client.stamp = {}
        env = self._active_env
        result.per_step_rows.append({
            "task_uid": task.task_uid, "yaml_id": task.yaml_id, "step_idx": SUMMARY_STEP_IDX,
            "row": "episode_summary", "attempt": task.attempt, "task_name": extra.get("task_name"),
            "task_id": task.task_id, "init_idx": task.orig_init_state_idx,
            "seed": int(extra["base_seed"]) + int(task.orig_init_state_idx), "pin_id": extra.get("pin_id"),
            "layout": extra.get("layout"), "style": extra.get("style"), **stamp,
            "reported_n_steps": result.n_steps, "n_env_steps": None if env is None else env.n_steps,
            "n_decisions": client.n_infer, "n_hit_meta": len(client.hit_meta), "hit_meta": list(client.hit_meta),
            "success": result.success, "error": result.error,
            "worker_runtime": worker_runtime(),
        })
        return result


def build_runner(args: argparse.Namespace) -> WatchdogRunner:
    try:
        adapter_factory = ADAPTERS[args.teacher]
    except KeyError:
        raise SystemExit(f"unknown --teacher {args.teacher!r}; expected one of {sorted(ADAPTERS)}") from None
    runner = StepDiagEpisodeRunner(
        adapter_factory(),
        connect_deadline_s=args.connect_deadline_s,
        connect_retries=args.connect_retries,
        max_cached_envs=None if args.max_cached_envs < 1 else args.max_cached_envs,
        pinned_objects_path=getattr(args, "pinned_objects", "") or None,
    )
    return WatchdogRunner(runner, episode_deadline_s=args.episode_deadline_s, terminate_grace_s=args.terminate_grace_s)


def step_diag_spawn_fn(
    spec: WorkerSpec,
    driver_host: str,
    driver_port: int,
    *,
    worker_python: str,
    robocasa_cwd: str,
    repo_root: str,
    egl_lib_dir: str,
    egl_vendor_dir: str,
    teacher: str,
    connect_deadline_s: float,
    episode_deadline_s: float,
    terminate_grace_s: float,
    max_cached_envs: int | None = None,
    pinned_objects_path: str | None = None,
) -> subprocess.Popen:
    """Launch one island-A worker running THIS module.

    A mirror of ``run_collect.robocasa_spawn_fn`` (argv contract, worker environment,
    ``start_new_session=True``) with the entry module swapped; ``tests/exp/step_diag/test_worker_entry.py``
    pins the two against each other so a change to the production spawn cannot drift past this one.
    """
    cmd, env = worker_command_and_env(
        spec, driver_host, driver_port, module="exp.step_diag.worker_entry", worker_python=worker_python,
        repo_root=repo_root, egl_lib_dir=egl_lib_dir, egl_vendor_dir=egl_vendor_dir, teacher=teacher,
        connect_deadline_s=connect_deadline_s, episode_deadline_s=episode_deadline_s,
        terminate_grace_s=terminate_grace_s, max_cached_envs=max_cached_envs, pinned_objects_path=pinned_objects_path,
    )
    return subprocess.Popen(cmd, env=env, cwd=robocasa_cwd, start_new_session=True)


def worker_command_and_env(
    spec: WorkerSpec, driver_host: str, driver_port: int, *, module: str, worker_python: str, repo_root: str,
    egl_lib_dir: str, egl_vendor_dir: str, teacher: str, connect_deadline_s: float, episode_deadline_s: float,
    terminate_grace_s: float, max_cached_envs: int | None, pinned_objects_path: str | None,
) -> tuple[list[str], dict[str, str]]:
    """The argv + environment of one worker (transcribed from ``robocasa_spawn_fn``)."""
    cmd = [
        worker_python, "-m", module,
        "--worker-id", spec.worker_id,
        "--server-key", spec.server_key,
        "--driver-host", driver_host,
        "--driver-port", str(driver_port),
        "--teacher", teacher,
        "--connect-deadline-s", str(connect_deadline_s),
        "--episode-deadline-s", str(episode_deadline_s),
        "--terminate-grace-s", str(terminate_grace_s),
    ]
    if max_cached_envs is not None:
        cmd += ["--max-cached-envs", str(max_cached_envs)]
    if pinned_objects_path:
        cmd += ["--pinned-objects", pinned_objects_path]
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = os.pathsep.join((repo_root, os.path.join(repo_root, "src")))
    env["MUJOCO_GL"] = "egl"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (egl_lib_dir, env.get("LD_LIBRARY_PATH", "")) if p)
    env["__EGL_VENDOR_LIBRARY_DIRS"] = egl_vendor_dir
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
    return cmd, env


def main() -> None:
    args = parse_args()
    runner = build_runner(args)

    def connect() -> socket.socket:
        return socket.create_connection((args.driver_host, args.driver_port))

    WorkerLoop(args.worker_id, args.server_key, runner, connect=connect).run_forever()


if __name__ == "__main__":
    main()
