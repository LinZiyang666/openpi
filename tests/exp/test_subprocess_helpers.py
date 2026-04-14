"""Unit tests for ``exp/common/_subprocess.py`` (cleanup/02, plan P0-4).

Five-plus cases per plan §2 P0-4 verification:
- ``conda_env=None`` returns ``(["uv", "run", *main_args], None)``.
- ``conda_env="libero_sim"`` strips ``VIRTUAL_ENV`` / ``PYTHONPATH`` /
  ``PYTHONHOME``.
- venv bin is removed from ``PATH``.
- ``MUJOCO_GL=egl`` is injected.
- ``extra_env`` can override ``MUJOCO_GL``.
- Keys not in ``extra_env`` retain defaults.
"""

from __future__ import annotations

import os

import pytest

from exp.common._subprocess import build_subprocess_cmd


def test_conda_env_none_returns_uv_run_and_no_env_override():
    cmd, env = build_subprocess_cmd(["examples/libero/main.py", "--seed", "7"])

    assert cmd == ["uv", "run", "examples/libero/main.py", "--seed", "7"]
    assert env is None


def test_conda_env_none_rejects_extra_env():
    with pytest.raises(ValueError, match="extra_env requires conda_env"):
        build_subprocess_cmd(
            ["main.py"], conda_env=None, extra_env={"FOO": "bar"}
        )


def test_conda_env_strips_uv_injections(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/fake/.venv")
    monkeypatch.setenv("PYTHONPATH", "/fake/pythonpath")
    monkeypatch.setenv("PYTHONHOME", "/fake/pythonhome")
    monkeypatch.setenv("KEEP_ME", "1")

    _, env = build_subprocess_cmd(["main.py"], conda_env="libero_sim")

    assert env is not None
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env.get("KEEP_ME") == "1"


def test_conda_env_removes_venv_bin_from_path(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/fake/.venv")
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(["/usr/bin", "/fake/.venv/bin", "/opt/conda/bin"]),
    )

    _, env = build_subprocess_cmd(["main.py"], conda_env="libero_sim")

    parts = env["PATH"].split(os.pathsep)
    assert "/fake/.venv/bin" not in parts
    assert "/usr/bin" in parts
    assert "/opt/conda/bin" in parts


def test_conda_env_injects_mujoco_gl_egl_by_default(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    _, env = build_subprocess_cmd(["main.py"], conda_env="libero_sim")

    assert env["MUJOCO_GL"] == "egl"


def test_extra_env_overrides_mujoco_gl(monkeypatch):
    _, env = build_subprocess_cmd(
        ["main.py"],
        conda_env="libero_sim",
        extra_env={"MUJOCO_GL": "osmesa"},
    )

    assert env["MUJOCO_GL"] == "osmesa"


def test_extra_env_leaves_other_defaults_alone(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/fake/.venv")

    _, env = build_subprocess_cmd(
        ["main.py"],
        conda_env="libero_sim",
        extra_env={"FOO": "bar"},
    )

    # MUJOCO_GL default preserved; VIRTUAL_ENV still stripped; new key present.
    assert env["MUJOCO_GL"] == "egl"
    assert "VIRTUAL_ENV" not in env
    assert env["FOO"] == "bar"


def test_cmd_shape_conda_path():
    cmd, _ = build_subprocess_cmd(
        ["examples/libero/main.py", "--port", "8001"],
        conda_env="libero_sim",
    )

    assert cmd == [
        "conda", "run", "--no-capture-output", "-n", "libero_sim", "python",
        "examples/libero/main.py", "--port", "8001",
    ]
