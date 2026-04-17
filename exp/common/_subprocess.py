"""Subprocess command builder for main.py dispatch (cleanup/02, plan P0-4).

Consolidates the two pre-cleanup copies of the "run main.py either under
uv or under conda, stripping uv's env injections so conda's interpreter
wins, and setting MUJOCO_GL=egl" recipe found at

- ``exp/common/run_cache_experiments.py`` (inline inside ``_execute_task_batch``)
- ``exp/trajectory_deviation/run_step1b_gt.py::_build_subprocess_cmd``

``conda_env`` defaults to ``None`` — matching both runners' pre-cleanup
behavior of "``--conda-env`` not passed → ``uv run``". Changing this
default would silently flip every existing invocation onto conda, so
keep it ``None``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple


def build_subprocess_cmd(
    main_args: List[str],
    *,
    conda_env: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Optional[Dict[str, str]]]:
    """Return ``(cmd, env)`` for ``subprocess.Popen`` / ``subprocess.run``.

    - ``conda_env=None`` → ``["uv", "run", *main_args]`` with ``env=None``
      (inherit caller's environment untouched).
    - ``conda_env="libero_sim"`` → conda run with ``VIRTUAL_ENV`` /
      ``PYTHONPATH`` / ``PYTHONHOME`` stripped and the uv venv removed
      from ``PATH`` so the conda interpreter's deps win. ``MUJOCO_GL=egl``
      is injected for headless rendering.
    - ``extra_env`` is applied **after** the ``MUJOCO_GL=egl`` default so
      callers can override (e.g. ``{"MUJOCO_GL": "osmesa"}``). Passing
      ``extra_env`` with ``conda_env=None`` is an error — there is no env
      dict to update in the uv-run path, and silently building one would
      drop the rest of the caller's environment.
    """
    if not conda_env:
        if extra_env:
            raise ValueError(
                "extra_env requires conda_env; uv-run path inherits caller env"
            )
        return ["uv", "run", *main_args], None

    env_flag = "-p" if ("/" in conda_env or os.path.isabs(conda_env)) else "-n"
    cmd = [
        "conda", "run", "--no-capture-output", env_flag, conda_env, "python",
        *main_args,
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
    }
    venv_bin = os.environ.get("VIRTUAL_ENV", "")
    if venv_bin:
        venv_bin = os.path.join(venv_bin, "bin")
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin
        )
    env["MUJOCO_GL"] = "egl"
    if extra_env:
        env.update(extra_env)
    return cmd, env
