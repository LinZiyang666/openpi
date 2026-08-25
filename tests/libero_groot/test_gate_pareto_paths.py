"""Both chain entry points resolve to the checkout they were launched from.

The scheduler runs on the serving box and the driver is invoked from whichever
checkout an operator is sitting in, and those two hosts hold the repository at
different absolute paths. A hardcoded root is therefore wrong on one of them by
construction -- and wrong in the most expensive way, because the failure is not
an import error at launch but a server that never binds, once per cell, until a
phase has burned its wall clock.

These checks need no GPU and no remote: they assert that what the entry points
resolve *right now*, in this checkout, actually exists.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from exp.libero_groot import orchestrate_search as sched

REPO = pathlib.Path(__file__).resolve().parents[2]
DRIVER = REPO / "exp/libero_groot/run_gate_pareto.sh"


def test_scheduler_resolves_this_checkout():
    assert sched.REPO_ROOT == REPO
    # The string constant must be derived, not restated: a second literal is
    # exactly how the two drift apart.
    assert sched.WEILAND_REPO == str(REPO)


def test_scheduler_pythonpath_points_at_this_checkout():
    parts = sched.GR00T_PATH.split(":")
    assert str(REPO) in parts
    assert f"{REPO}/src" in parts
    # The GR00T island is a separate install and legitimately absolute, but it
    # must be overridable rather than welded in.
    assert sched.GR00T_HOME in parts


def test_the_server_script_the_scheduler_launches_exists():
    assert (sched.REPO_ROOT / "exp/libero_groot/serve_groot_libero.py").is_file()


def test_the_venv_interpreter_the_driver_uses_exists():
    # Tied to the supported flow: `uv sync` puts the interpreter at the repo
    # root, and both CI and the driver invoke it from there. A tree extracted
    # without syncing (e.g. `git archive`) legitimately has no .venv, so a
    # failure here means the environment was never set up -- not that the
    # driver's path resolution is wrong.
    interpreter = REPO / ".venv/bin/python"
    assert interpreter.is_file(), f"{interpreter} missing"
    assert os.access(interpreter, os.X_OK), f"{interpreter} is not executable"


def test_the_analysis_entry_point_exists():
    analyzer = REPO / "exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py"
    assert analyzer.is_file()


def test_driver_resolves_the_same_root_as_the_scheduler():
    # Run the real script in its preflight-only mode rather than restating its
    # resolution expression here: a test carrying its own copy of the logic
    # proves nothing about the script it is supposed to be checking.
    proc = subprocess.run(
        ["bash", str(DRIVER)],
        capture_output=True, text=True,
        env={**os.environ, "GP_PREFLIGHT_ONLY": "1"},
    )
    assert proc.returncode == 0, f"driver preflight failed:\n{proc.stdout}{proc.stderr}"
    repo_line = next(ln for ln in proc.stdout.splitlines() if " repo=" in ln)
    resolved = repo_line.split("repo=", 1)[1].split(" host=", 1)[0]
    assert pathlib.Path(resolved) == REPO
    assert "preflight OK" in proc.stdout


def test_preflight_names_the_host_and_the_resolved_root(monkeypatch):
    # The message has to carry both, or an operator on the wrong box sees only
    # "missing file" and starts looking for the file instead of the host.
    monkeypatch.setattr(sched, "ISLAND_PY", "/nonexistent/python")
    with pytest.raises(SystemExit) as excinfo:
        sched.preflight()
    message = str(excinfo.value)
    assert "/nonexistent/python" in message
    assert str(REPO) in message
    assert "GROOT_N15_PYTHON" in message


def test_preflight_passes_when_the_island_is_present(monkeypatch):
    monkeypatch.setattr(sched, "ISLAND_PY", sys.executable)
    monkeypatch.setattr(sched, "GR00T_HOME", str(REPO))
    sched.preflight()
