"""Environment registry of the warm reset entry: benchmark adapters and the plugin hook.

An ``EntryEnv`` is everything the entry needs to know about one (policy,
benchmark) environment: the served executor family (``pi05`` / ``groot``),
the action horizon (the replan bound), the flow-matching schedule and K, the
``--arms all`` roster, and a ``BenchmarkAdapter`` that knows how the benchmark
exports its tasks, which EpisodeTask fields its worker reads, which
``episode_start`` identity keys admission may trust, which fields pair two
arms' episodes, and how its workers are launched.

The step_diag environments (``exp.step_diag.envs.ENVS``, read-only) are
registered with the standard LIBERO and RoboCasa adapters under their
existing ids, so every run directory prepared before this registry existed
replans to the same EpisodeTasks.

Hook for a new benchmark
------------------------
Create ``exp/<package>/warm_reset_env.py`` that calls, at import time::

    from exp.warm_reset.envs import BenchmarkAdapter, EntryEnv, register_env

    class MyAdapter(BenchmarkAdapter):
        name = "my_benchmark"
        # override validate_rollout, export_tasks (or _task_pairs), experiment,
        # episode_extra, identity, pairing and worker_agent (see BenchmarkAdapter)

    register_env(EntryEnv(env_id="pi05_mybench", policy="pi05", benchmark="mybench",
                          action_horizon=5, k_full=10, schedule_id="pi05_v1",
                          adapter=MyAdapter()))

``get_env`` / ``env_ids`` import every ``exp/*/warm_reset_env.py`` once (the
first time an id is not registered, and for the CLI's choice list), so the
new id works with every subcommand (``tasks`` / ``prepare`` / ``run`` /
``agent`` / ``admit`` / ``analysis``) without touching this package. The
plugin module must be import-light: no simulator or model import at module
level (the driver and admission run in the main venv). A plugin that fails
to import is reported when its id is requested, and never breaks the others.

Public interface: ``EntryEnv``, ``BenchmarkAdapter``, ``LiberoAdapter``,
``RoboCasaAdapter``, ``entry_env_from_spec``, ``register_env``, ``get_env``,
``env_ids``.
Depends on ``exp.step_diag.envs`` (read-only tables) and, lazily, the
conductor worker agent / RoboCasa spawn helpers.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import pathlib
from typing import Any

from exp.step_diag import envs as _step_diag_envs

logger = logging.getLogger(__name__)

_EXP_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PLUGIN_MODULE = "warm_reset_env"


# ------------------------------------------------------------------
# Environment and adapter
# ------------------------------------------------------------------


class BenchmarkAdapter:
    """What differs between benchmarks, as seen by the warm reset entry.

    Every method receives the ``EntryEnv`` and, where relevant, the frozen
    plan (``manifest``); none of them may read server evidence. Subclass and
    override for a new benchmark; the defaults raise so a missing piece is
    loud.
    """

    #: Short benchmark family name, recorded nowhere; for messages only.
    name = "base"
    #: Whether ``prepare --experiment-id`` may set a stable episode experiment id.
    supports_experiment_id = False
    #: Whether ``prepare --pinned-objects`` applies (RoboCasa PnP lane).
    supports_pinned_objects = False

    def validate_rollout(self, env: EntryEnv, rollout: dict) -> None:
        """Refuse missing / invalid rollout knobs (``replan_steps`` is checked by the entry)."""
        raise NotImplementedError(f"{self.name}: validate_rollout")

    def export_tasks(
        self, env: EntryEnv, *, task_ids: str, task_names: str, episodes: int, init_offset: int
    ) -> list[dict]:
        """The task roster ``[{task_id, name, init_indices}]`` of the ``tasks`` subcommand.

        Runs in the simulator environment. The default numbers ``_task_pairs``
        and gives every task the original init indices
        ``init_offset .. init_offset + episodes - 1``.
        """
        return [
            {"task_id": i, "name": name, "init_indices": list(range(init_offset, init_offset + episodes))}
            for i, name in self._task_pairs(env, task_ids=task_ids, task_names=task_names)
        ]

    def _task_pairs(self, env: EntryEnv, *, task_ids: str, task_names: str) -> list[tuple[int, str]]:
        raise NotImplementedError(f"{self.name}: export_tasks")

    def experiment(self, env: EntryEnv, manifest: dict) -> str:
        """``EpisodeTask.experiment`` (the episode_start ``experiment`` and a self-seed key)."""
        raise NotImplementedError(f"{self.name}: experiment")

    def episode_extra(self, env: EntryEnv, manifest: dict, task: dict) -> dict:
        """``EpisodeTask.extra`` fields the worker needs, besides ``num_trials_per_task``."""
        raise NotImplementedError(f"{self.name}: episode_extra")

    def identity(self, env: EntryEnv, manifest: dict, episode: Any) -> dict:
        """Extra episode_start identity keys the worker sends, rebuilt from the plan (trusted)."""
        return {}

    def pairing(self, env: EntryEnv, manifest: dict, episode: Any) -> dict:
        """Environment-identity fields two arms' episodes pair on, besides task and env id."""
        raise NotImplementedError(f"{self.name}: pairing")

    def worker_agent(self, env: EntryEnv, manifest: dict, *, specs: list, driver_host: str,
                     driver_port: int, options: dict) -> Any:
        """A ``WorkerAgent`` over the entry's ``WorkerSpec`` list (``options``: the agent CLI knobs)."""
        raise NotImplementedError(f"{self.name}: worker_agent")


@dataclasses.dataclass(frozen=True)
class EntryEnv:
    """One environment of the warm reset entry (see module docstring)."""

    env_id: str
    policy: str  # "pi05" | "groot": the served executor family
    benchmark: str  # LIBERO suite / "robocasa365" / a new benchmark's id
    action_horizon: int
    k_full: int
    schedule_id: str
    adapter: BenchmarkAdapter
    default_arms: tuple[str, ...] = ()

    @property
    def schedule(self):
        """The ``DenoiseSchedule`` of ``schedule_id`` (MISS arms may run fewer steps)."""
        from openpi.cache.types import schedule_from_id

        return schedule_from_id(self.schedule_id)

    @property
    def teacher(self) -> str:
        """The teacher label RoboCasa workers and the step_diag roots use."""
        return "groot_tp" if self.policy == "groot" else "pi05"


# ------------------------------------------------------------------
# Built-in adapters (the existing LIBERO / RoboCasa workers)
# ------------------------------------------------------------------


def _nonneg_int(rollout: dict, key: str, what: str) -> None:
    if type(rollout.get(key)) is not int or rollout[key] < 0:
        raise ValueError(f"{what} requires a nonnegative {key}")


class LiberoAdapter(BenchmarkAdapter):
    """Standard LIBERO worker (``examples/libero/episode_runner.py``); experiment = suite."""

    name = "libero"

    def validate_rollout(self, env: EntryEnv, rollout: dict) -> None:
        _nonneg_int(rollout, "seed", "LIBERO")

    def _task_pairs(self, env: EntryEnv, *, task_ids: str, task_names: str) -> list[tuple[int, str]]:
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[env.benchmark]()
        ids = range(suite.n_tasks) if task_ids == "all" else [int(i) for i in task_ids.split(",")]
        return [(i, suite.get_task(i).language) for i in ids]

    def experiment(self, env: EntryEnv, manifest: dict) -> str:
        return env.benchmark

    def episode_extra(self, env: EntryEnv, manifest: dict, task: dict) -> dict:
        return {}

    def pairing(self, env: EntryEnv, manifest: dict, episode: Any) -> dict:
        rollout = manifest["rollout"]
        pool = manifest.get("init_pool_sha256")
        if pool is None:
            # No digest recorded (pre-registry plan or a pool unreadable at
            # prepare): the pool is named by its worker path, or LIBERO's own.
            pool = f"path:{rollout['init_states_dir']}" if rollout.get("init_states_dir") else "libero_default"
        return {"init_idx": episode.orig_init_state_idx, "env_seed": rollout["seed"], "init_pool_sha256": pool}

    def worker_agent(self, env: EntryEnv, manifest: dict, *, specs: list, driver_host: str,
                     driver_port: int, options: dict) -> Any:
        from openpi.conductor.agent import WorkerAgent

        if options.get("pinned_objects"):
            raise ValueError("--pinned-objects is a RoboCasa option")
        return WorkerAgent(specs, driver_host, driver_port)


class RoboCasaAdapter(BenchmarkAdapter):
    """Island-A RoboCasa worker (``exp/robocasa365/episode_runner.py``), optionally pinned."""

    name = "robocasa"
    supports_experiment_id = True
    supports_pinned_objects = True

    def validate_rollout(self, env: EntryEnv, rollout: dict) -> None:
        for key in ("base_seed", "layout", "style"):
            _nonneg_int(rollout, key, "RoboCasa")

    def _task_pairs(self, env: EntryEnv, *, task_ids: str, task_names: str) -> list[tuple[int, str]]:
        names = [name.strip() for name in task_names.split(",") if name.strip()]
        if not names:
            raise ValueError("RoboCasa needs --task-names")
        return list(enumerate(names))

    def experiment(self, env: EntryEnv, manifest: dict) -> str:
        # A stable id (prepare --experiment-id) makes self-start noise common
        # across runs; without it the run token keeps runs apart (pre-registry).
        return manifest.get("experiment_id") or f"warm_reset_{manifest['token']}"

    def episode_extra(self, env: EntryEnv, manifest: dict, task: dict) -> dict:
        rollout = manifest["rollout"]
        extra = {
            "task_name": task["name"],
            "teacher": env.teacher,
            **{k: rollout[k] for k in ("base_seed", "layout", "style", "replan_steps")},
        }
        pins = manifest.get("pins")
        if pins is not None:
            from exp.robocasa365.pinned_objects import compute_pin_task_id

            slot_map = pins[task["name"]]
            extra.update(pin_id=manifest["pin_id"], pin_task_id=compute_pin_task_id(task["name"], slot_map),
                         pinned_objects=dict(slot_map))
        return extra

    def identity(self, env: EntryEnv, manifest: dict, episode: Any) -> dict:
        out = {"seed": manifest["rollout"]["base_seed"] + episode.orig_init_state_idx}
        if manifest.get("pins") is not None:
            out.update(pin_id=episode.extra["pin_id"], pin_task_id=episode.extra["pin_task_id"])
        return out

    def pairing(self, env: EntryEnv, manifest: dict, episode: Any) -> dict:
        rollout = manifest["rollout"]
        name = episode.extra["task_name"]
        try:
            lane = _step_diag_envs.lane_of(name)
        except KeyError:
            lane = None
        return {"init_idx": episode.orig_init_state_idx, "env_seed": rollout["base_seed"] + episode.orig_init_state_idx,
                "lane": lane, "pin_id": manifest.get("pin_id"), "layout": rollout["layout"],
                "style": rollout["style"]}

    def worker_agent(self, env: EntryEnv, manifest: dict, *, specs: list, driver_host: str,
                     driver_port: int, options: dict) -> Any:
        from functools import partial

        from exp.robocasa365.run_collect import robocasa_spawn_fn
        from openpi.conductor.agent import WorkerAgent

        rc = dict(options.get("rc") or {})
        if any(not rc.get(k) for k in ("worker_python", "robocasa_cwd", "egl_lib_dir", "egl_vendor_dir")):
            raise ValueError("RoboCasa requires worker Python, cwd and EGL library/vendor paths")
        pin_path = options.get("pinned_objects") or ""
        if (manifest.get("pin_id") is not None) != bool(pin_path):
            raise ValueError("a pinned plan needs --pinned-objects on its agent, and an unpinned plan refuses it")
        if pin_path:
            from exp.robocasa365.pinned_objects import (
                load_pin_manifest,
                resolve_manifest_path,
            )

            pin_path = resolve_manifest_path(pin_path)
            if load_pin_manifest(pin_path)[0] != manifest["pin_id"]:
                raise ValueError("this agent's pin manifest is not the plan's pin table (pin_id differs)")
            rc["pinned_objects_path"] = pin_path
        spawn = partial(
            robocasa_spawn_fn,
            repo_root=str(pathlib.Path(__file__).resolve().parents[2]),
            teacher=env.teacher,
            **rc,
        )
        return WorkerAgent(specs, driver_host, driver_port, spawn_fn=spawn)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_REGISTRY: dict[str, EntryEnv] = {}
_PLUGIN_ERRORS: dict[str, str] = {}
_discovered = False


def register_env(env: EntryEnv, *, replace: bool = False) -> EntryEnv:
    """Register ``env`` under its id; an id already taken is refused unless ``replace``."""
    if not isinstance(env, EntryEnv) or not isinstance(env.adapter, BenchmarkAdapter):
        raise TypeError("register_env takes an EntryEnv with a BenchmarkAdapter")
    if env.policy not in ("pi05", "groot"):
        raise ValueError(f"{env.env_id}: policy must be 'pi05' or 'groot' (the served executor family)")
    if env.env_id in _REGISTRY and not replace and _REGISTRY[env.env_id] != env:
        raise ValueError(f"warm reset env {env.env_id!r} is already registered")
    _REGISTRY[env.env_id] = env
    return env


def _default_arms(spec: Any) -> tuple[str, ...]:
    """The pre-registry ``--arms all`` roster: the step_diag self-family warm arms."""
    E = _step_diag_envs
    arms = (
        (*E.MACRO13_ARMS_BY_POLICY[spec.policy], *E.SELF13_ARMS_BY_POLICY[spec.policy])
        if spec.benchmark == "robocasa365"
        else E.LIBERO_SELF_ARMS_BY_POLICY[spec.policy]
    )
    return tuple(dict.fromkeys(a for a in arms if "_t" in a))


def entry_env_from_spec(spec: Any, adapter: BenchmarkAdapter, default_arms: tuple[str, ...] = ()) -> EntryEnv:
    """An ``EntryEnv`` from a step_diag ``EnvSpec`` row (``env_id`` / ``policy`` / ... fields)."""
    return EntryEnv(
        env_id=spec.env_id, policy=spec.policy, benchmark=spec.benchmark, action_horizon=spec.action_horizon,
        k_full=spec.k_full, schedule_id=spec.schedule_id, adapter=adapter, default_arms=tuple(default_arms),
    )


def _register_step_diag_envs() -> None:
    for spec in _step_diag_envs.ENVS.values():
        adapter = RoboCasaAdapter() if spec.benchmark == "robocasa365" else LiberoAdapter()
        register_env(entry_env_from_spec(spec, adapter, _default_arms(spec)))


def _discover() -> None:
    """Import every ``exp/*/warm_reset_env.py`` once; failures are kept per module."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    for path in sorted(_EXP_ROOT.glob(f"*/{_PLUGIN_MODULE}.py")):
        module = f"exp.{path.parent.name}.{_PLUGIN_MODULE}"
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - one broken plugin must not hide the others
            _PLUGIN_ERRORS[module] = f"{type(exc).__name__}: {exc}"
            logger.warning("warm reset env plugin %s failed to import: %s", module, exc)


def get_env(env_id: str) -> EntryEnv:
    """The registered environment ``env_id`` (plugins are discovered on the first miss)."""
    if env_id not in _REGISTRY:
        _discover()
    if env_id not in _REGISTRY:
        hint = f"; plugin import errors: {_PLUGIN_ERRORS}" if _PLUGIN_ERRORS else ""
        raise KeyError(f"unknown warm reset env {env_id!r} (registered: {sorted(_REGISTRY)}){hint}")
    return _REGISTRY[env_id]


def env_ids() -> list[str]:
    """Every registered environment id, plugins included."""
    _discover()
    return sorted(_REGISTRY)


_register_step_diag_envs()


__all__ = [
    "BenchmarkAdapter",
    "EntryEnv",
    "LiberoAdapter",
    "RoboCasaAdapter",
    "entry_env_from_spec",
    "env_ids",
    "get_env",
    "register_env",
]

