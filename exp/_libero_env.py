"""LIBERO env construction helper (cleanup/01, plan P0-3).

Consolidates the three pre-cleanup copies of the
``get_benchmark_dict()[task_suite]().get_task(task_id) →
get_libero_path("bddl_files") / task.problem_folder / task.bddl_file →
OffScreenRenderEnv(...)`` sequence found at

- ``exp/run_spawn_experiment.py::_SpawnCommon.make_env``
- ``scripts/verify_env_save_restore.py::_build_env``
- ``scripts/verify_restore_obs_equivalence.py::verify`` (inline)

The V-3 regression (task_1 construction failure) happened because those
three copies drifted apart: only the examples/ layer had been patched.

Scope note (plan §7 D1):
    ``examples/libero/main.py::_get_libero_env`` is intentionally **not**
    folded into this helper — examples/ stays self-contained. The ``seed``
    parameter is exposed so this helper is capability-equivalent with the
    reference implementation (which always calls ``env.seed(seed)``), but
    every cleanup-range callsite passes ``seed=None`` to preserve current
    behavior. Whether the spawn runner should start using the GT HDF5 seed
    is tracked as follow-up F1, not part of this cleanup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_bddl_path(task_suite: str, task_id: int) -> Path:
    """Return the absolute ``.bddl`` path for ``(task_suite, task_id)``.

    Mirrors ``examples/libero/main.py::_get_libero_env`` path resolution:
    ``task.bddl_file`` is only a filename, the full path lives under
    ``get_libero_path("bddl_files") / task.problem_folder / task.bddl_file``.
    """
    # Lazy import: callers only pay the libero import cost when actually
    # constructing envs, and ``--help`` works in uv venv without libero.
    from libero.libero import get_libero_path  # type: ignore[import]
    from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]

    suite = get_benchmark_dict()[task_suite]()
    task = suite.get_task(int(task_id))
    return (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )


def build_libero_env(
    task_suite: str,
    task_id: int,
    *,
    resolution: int = 256,
    seed: int | None = None,
) -> Any:
    """Construct an ``OffScreenRenderEnv`` for ``(task_suite, task_id)``.

    ``seed=None`` (the default, matching all cleanup-range callsites) skips
    ``env.seed(...)`` entirely. Passing an int calls ``env.seed(seed)``
    exactly once, matching ``examples/libero/main.py::_get_libero_env``.
    """
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import]

    bddl_path = resolve_bddl_path(task_suite, task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    if seed is not None:
        env.seed(seed)
    return env
