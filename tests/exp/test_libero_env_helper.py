"""Unit tests for ``exp/trajectory_deviation/_libero_env.py`` (cleanup/01, plan P0-3).

These tests must not depend on a real LIBERO install (G2 Watchpoint #1):
``libero.libero.*`` is stubbed via ``sys.modules`` patching before the
helper imports it. This keeps the suite runnable in the plain uv venv.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def libero_stubs(monkeypatch, tmp_path):
    """Install fake ``libero.libero`` submodules that the helper imports.

    Returns a dict of handles the tests can inspect:
        - ``env_ctor``   : the ``OffScreenRenderEnv`` constructor Mock
        - ``env``        : the single env instance ``env_ctor`` returns
        - ``bddl_root``  : the fake ``get_libero_path("bddl_files")`` root
        - ``task``       : the fake task object (has ``problem_folder`` /
          ``bddl_file``)
    """
    bddl_root = tmp_path / "bddl_files"
    bddl_root.mkdir()

    task = MagicMock()
    task.problem_folder = "libero_10_scene"
    task.bddl_file = "task_1.bddl"

    suite = MagicMock()
    suite.get_task = MagicMock(return_value=task)

    def _fake_get_benchmark_dict():
        return {"libero_10": lambda: suite, "libero_spatial": lambda: suite}

    def _fake_get_libero_path(kind: str) -> str:
        assert kind == "bddl_files"
        return str(bddl_root)

    env = MagicMock()
    env.seed = MagicMock()
    env_ctor = MagicMock(return_value=env)

    # ``from libero.libero import get_libero_path`` requires ``libero.libero``
    # to be a real package-ish module with ``get_libero_path`` attribute.
    libero_pkg = types.ModuleType("libero")
    libero_libero = types.ModuleType("libero.libero")
    libero_libero.get_libero_path = _fake_get_libero_path
    libero_pkg.libero = libero_libero

    libero_benchmark = types.ModuleType("libero.libero.benchmark")
    libero_benchmark.get_benchmark_dict = _fake_get_benchmark_dict

    libero_envs = types.ModuleType("libero.libero.envs")
    libero_envs.OffScreenRenderEnv = env_ctor

    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", libero_benchmark)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", libero_envs)

    return {
        "env_ctor": env_ctor,
        "env": env,
        "bddl_root": bddl_root,
        "task": task,
        "suite": suite,
    }


def test_resolve_bddl_path(libero_stubs):
    from exp.trajectory_deviation._libero_env import resolve_bddl_path

    path = resolve_bddl_path("libero_10", 1)

    assert path == libero_stubs["bddl_root"] / "libero_10_scene" / "task_1.bddl"
    libero_stubs["suite"].get_task.assert_called_once_with(1)


def test_build_env_seed_none_does_not_call_env_seed(libero_stubs):
    from exp.trajectory_deviation._libero_env import build_libero_env

    env = build_libero_env("libero_10", 1)

    assert env is libero_stubs["env"]
    env.seed.assert_not_called()
    libero_stubs["env_ctor"].assert_called_once()
    kwargs = libero_stubs["env_ctor"].call_args.kwargs
    assert kwargs["camera_heights"] == 256
    assert kwargs["camera_widths"] == 256
    assert kwargs["bddl_file_name"].endswith("task_1.bddl")


def test_build_env_seed_int_calls_env_seed_exactly_once(libero_stubs):
    from exp.trajectory_deviation._libero_env import build_libero_env

    env = build_libero_env("libero_10", 1, seed=7)

    env.seed.assert_called_once_with(7)


def test_build_env_resolution_passed_through(libero_stubs):
    from exp.trajectory_deviation._libero_env import build_libero_env

    build_libero_env("libero_10", 1, resolution=128)

    kwargs = libero_stubs["env_ctor"].call_args.kwargs
    assert kwargs["camera_heights"] == 128
    assert kwargs["camera_widths"] == 128


def test_build_env_default_resolution_is_256(libero_stubs):
    """Matches ``examples/libero/main.py::LIBERO_ENV_RESOLUTION`` and the
    current ``_SpawnCommon.make_env`` default. Regressing this would shift
    spawn obs byte-equality with GT HDF5."""
    from exp.trajectory_deviation._libero_env import build_libero_env

    build_libero_env("libero_10", 1)

    kwargs = libero_stubs["env_ctor"].call_args.kwargs
    assert kwargs["camera_heights"] == 256
