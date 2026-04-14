"""Phase 0 smoke: verify restored env obs == GT HDF5 obs.

Referenced by ``logs/trajectory_deviation_corrective_implementation.log.md``
§19.6. Complements ``scripts/verify_env_save_restore.py`` — not only must the
env physics round-trip, but after ``env.set_init_state(sim_state)`` the
observation the policy sees must match the one captured in the GT HDF5 at
the same cycle. Otherwise the spawn rollout sees a different obs than GT at
the teleport point, and deviate-score comparisons are biased from step 0.

What this script does, for one HDF5 cycle ``cycle_idx``:
    1. Read ``sim_state``, ``agentview_image``, ``eye_in_hand_image``,
       ``robot_state`` and ``task_name`` from the GT HDF5.
    2. Build a fresh ``OffScreenRenderEnv`` for the recorded
       ``(task_suite, task_id)`` at ``LIBERO_ENV_RESOLUTION``.
    3. ``env.reset()`` then ``raw_obs = env.set_init_state(sim_state)``.
    4. Feed ``raw_obs`` through ``_obs_env_to_policy`` (the exact transform
       Layer F spawn uses) and assert equality against HDF5 contents:
         - images: ``np.array_equal``
         - state:  ``np.allclose(..., atol=--state-atol)``

**Exit conditions per §19.6**:
    - All cycles pass → Layer F may keep reading ``start_obs`` from HDF5
      directly (faster, avoids re-rendering).
    - Any cycle fails → Layer F MUST build ``start_obs`` via
      ``_obs_env_to_policy(env.set_init_state(sim_state))`` on the fly, and
      the spawn rollout must not rely on HDF5 obs caches.

Run example::

    uv run scripts/verify_restore_obs_equivalence.py \
        --gt-h5 data/deviation_experiment/gt_trajectories/task_3/episode_0.h5 \
        --task-suite libero_spatial --task-id 3 --cycle-idx 0
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


def _close_env_quietly(env) -> None:
    """Release MuJoCo/EGL handles; robosuite's atexit finaliser raises
    ``OpenGL.EGL._errors.EGLError`` on WSL GPU that clutters the smoke log
    — drain the env first and silence that particular stream during close.
    Mirrors the helper in ``verify_env_save_restore.py``.
    """
    close = getattr(env, "close", None)
    if not callable(close):
        return
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stderr(devnull):
            close()
    except Exception as e:  # noqa: BLE001
        print(f"warning: env.close() raised {e!r}", file=sys.stderr)
    finally:
        if devnull is not None:
            try:
                devnull.close()
            except Exception:  # noqa: BLE001
                pass


def _import_deps():
    """Imports are lazy so ``--help`` works without libero installed."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv

    # Re-use the transform from exp/run_spawn_experiment to ensure we're
    # testing the exact same code path Layer F will run.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from exp.run_spawn_experiment import _obs_env_to_policy  # noqa: E402
    from examples.libero.main import LIBERO_ENV_RESOLUTION  # noqa: E402

    return (
        get_libero_path,
        get_benchmark_dict,
        OffScreenRenderEnv,
        _obs_env_to_policy,
        LIBERO_ENV_RESOLUTION,
    )


def verify(
    gt_h5_path: Path,
    task_suite: str,
    task_id: int,
    cycle_indices: Sequence[int],
    *,
    state_atol: float = 1e-6,
) -> None:
    (
        get_libero_path,
        get_benchmark_dict,
        OffScreenRenderEnv,
        _obs_env_to_policy,
        LIBERO_ENV_RESOLUTION,
    ) = _import_deps()

    with h5py.File(gt_h5_path, "r") as f:
        task_name = str(f.attrs["task_name"])
        payloads = []
        for cycle_idx in cycle_indices:
            g = f[f"step_{cycle_idx:04d}"]
            payloads.append(
                {
                    "cycle_idx": cycle_idx,
                    "sim_state": g["sim_state"][...],
                    "ref_img": g["agentview_image"][...],
                    "ref_wrist": g["eye_in_hand_image"][...],
                    "ref_state": g["robot_state"][...],
                }
            )

    suite = get_benchmark_dict()[task_suite]()
    task = suite.get_task(task_id)
    # See examples/libero/main.py::_get_libero_env — bddl_file is a filename,
    # the real path lives under ``get_libero_path("bddl_files")/problem_folder``.
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )

    try:
        for p in payloads:
            env.reset()
            raw_obs = env.set_init_state(p["sim_state"])
            policy_obs = _obs_env_to_policy(raw_obs, task_name)

            img_ok = np.array_equal(policy_obs["observation/image"], p["ref_img"])
            wrist_ok = np.array_equal(
                policy_obs["observation/wrist_image"], p["ref_wrist"]
            )
            state_delta = float(
                np.abs(policy_obs["observation/state"] - p["ref_state"]).max()
            )
            state_ok = state_delta <= state_atol

            if not img_ok:
                raise AssertionError(
                    f"cycle {p['cycle_idx']}: agentview image mismatch"
                )
            if not wrist_ok:
                raise AssertionError(
                    f"cycle {p['cycle_idx']}: wrist image mismatch"
                )
            if not state_ok:
                raise AssertionError(
                    f"cycle {p['cycle_idx']}: state mismatch "
                    f"max_delta={state_delta:.3e} > atol={state_atol:.1e}"
                )
            print(
                f"OK: {gt_h5_path} cycle {p['cycle_idx']} "
                f"(state_max_delta={state_delta:.3e})"
            )
    finally:
        _close_env_quietly(env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-h5", required=True, type=Path)
    ap.add_argument("--task-suite", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument(
        "--cycle-idx",
        type=int,
        nargs="+",
        default=[0],
        help="one or more cycle indices to verify (default: 0)",
    )
    ap.add_argument("--state-atol", type=float, default=1e-6)
    args = ap.parse_args(argv)
    verify(
        args.gt_h5,
        args.task_suite,
        args.task_id,
        args.cycle_idx,
        state_atol=args.state_atol,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
