"""Offline guards for the Phase 0 smoke scripts.

Full end-to-end behaviour of
``scripts/verify_env_save_restore.py`` and
``scripts/verify_restore_obs_equivalence.py`` can only be exercised on a
real LIBERO install (MuJoCo + EGL + bddl assets). These tests therefore
only cover what is safe to run in CI:

1. The script modules import cleanly without pulling LIBERO at import time
   (both scripts must defer LIBERO imports until their ``run`` / ``verify``
   functions run — otherwise ``--help`` in the CI venv would fail).
2. ``--help`` exits 0 via ``argparse.SystemExit(0)``.
3. The argparse surface is the one Layer F expects (flag names fixed).

Any future refactor that either (a) moves LIBERO imports to module top-level,
or (b) renames a CLI flag, will break these tests — which is the whole point.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def _add_scripts_to_path():
    added = False
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
        added = True
    yield
    if added:
        sys.path.remove(str(_SCRIPTS_DIR))
    # Drop cached modules so each test re-imports fresh.
    for name in ("verify_env_save_restore", "verify_restore_obs_equivalence"):
        sys.modules.pop(name, None)


def test_verify_env_save_restore_imports_without_libero(_add_scripts_to_path):
    # libero is not installed in the test venv; if the script imports it
    # at module load this line raises ModuleNotFoundError.
    mod = importlib.import_module("verify_env_save_restore")
    assert hasattr(mod, "run")
    assert hasattr(mod, "main")


def test_verify_env_save_restore_help_exits_zero(
    _add_scripts_to_path, capsys: pytest.CaptureFixture[str]
):
    mod = importlib.import_module("verify_env_save_restore")
    with pytest.raises(SystemExit) as ei:
        mod.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--task-suite",
        "--task-id",
        "--init-state-idx",
        "--pre-steps",
        "--post-steps",
        "--seed",
        "--atol",
    ):
        assert flag in out, f"missing CLI flag in --help: {flag}"


def test_verify_restore_obs_equivalence_imports_without_libero(_add_scripts_to_path):
    mod = importlib.import_module("verify_restore_obs_equivalence")
    assert hasattr(mod, "verify")
    assert hasattr(mod, "main")


def test_verify_restore_obs_equivalence_help_exits_zero(
    _add_scripts_to_path, capsys: pytest.CaptureFixture[str]
):
    mod = importlib.import_module("verify_restore_obs_equivalence")
    with pytest.raises(SystemExit) as ei:
        mod.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--gt-h5",
        "--task-suite",
        "--task-id",
        "--cycle-idx",
        "--state-atol",
    ):
        assert flag in out, f"missing CLI flag in --help: {flag}"


def test_verify_restore_obs_equivalence_requires_gt_h5(
    _add_scripts_to_path, capsys: pytest.CaptureFixture[str]
):
    mod = importlib.import_module("verify_restore_obs_equivalence")
    with pytest.raises(SystemExit) as ei:
        mod.main([])
    # argparse returns 2 on missing required argument
    assert ei.value.code == 2
