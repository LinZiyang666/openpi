"""Phase 0 smoke: verify LIBERO env restore is **deterministic**.

Referenced by ``logs/trajectory_deviation_corrective_implementation.log.md``
§11.0 / §19.8 and the Verify finding V-1 recorded in
``logs/trajectory_deviation_corrective_implementation_review.log.md`` §28.5.

Original §19.8 asserted that ``reset + set_init_state + restore timestep/
cur_time`` reproduces ``live continuation`` bit-exactly (``atol=1e-6``). Real
MuJoCo does not offer that guarantee: under the Layer F teleport sequence
(``env.reset(); env.set_init_state(sim_state); inner.timestep/cur_time=…``)
the ``robot0_eef_pos`` trajectory drifts from live continuation on the order
of ``1e-3`` due to solver warm-start residuals. That drift is symmetric
across Layer F's cache vs baseline branches so it does not bias the spawn
metric — but the smoke's assertion had to be redefined.

What Layer F actually depends on is **determinism**: given one
``(sim_state, timestep, cur_time)`` anchor, two separate spawn rollouts with
the same action sequence must yield the same trajectory. If determinism
holds, repeated measurements at the same unit are comparable; if it fails,
spawn success rate picks up random noise.

What this script does:
    1. Drive the env ``--pre-steps`` random steps to a mid-episode state.
    2. Checkpoint ``(sim_state, timestep, cur_time)``.
    3. **Live branch** (reference, information only):
       continue ``--post-steps`` more random steps — no teleport.
    4. **Replay A**: ``env.reset() → set_init_state(ckpt) → restore
       timestep/cur_time`` → run the identical ``post_actions``.
    5. **Replay B**: same teleport sequence, same actions.
    6. Print ``max_delta_vs_live`` (replay A vs live) as an informational
       metric of how far MuJoCo drifts under teleport — never assertive.
    7. Assert ``allclose(replayA, replayB, atol=--atol)`` and
       ``traj_success_A == traj_success_B``. This is what determinism
       actually means.

Run (libero_sim conda env)::

    /home/weiland/anaconda3/envs/libero_sim/bin/python \
        scripts/verify_env_save_restore.py \
        --task-suite libero_spatial --task-id 0

Exit code 0 on pass; any assertion failure propagates as non-zero.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np


def _build_env(task_suite: str, task_id: int, resolution: int):
    """Construct env + init_states. Imports are lazy so ``--help`` works
    without libero installed.

    Env construction is delegated to ``exp._libero_env.build_libero_env`` so
    all cleanup-range callsites share one path-resolution implementation.
    ``init_states`` is fetched separately because it is only used here.
    """
    import sys
    from pathlib import Path

    from libero.libero.benchmark import get_benchmark_dict

    # ``scripts/`` is not on sys.path by default when run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from exp._libero_env import build_libero_env  # noqa: E402

    env = build_libero_env(task_suite, task_id, resolution=resolution, seed=None)
    suite = get_benchmark_dict()[task_suite]()
    init_states = suite.get_task_init_states(task_id)
    return env, init_states


def _replay_from_checkpoint(
    env,
    inner,
    *,
    ckpt_sim: np.ndarray,
    ckpt_timestep: int,
    ckpt_cur_time: float,
    post_actions: list,
):
    """Replicate Layer F teleport exactly, then run ``post_actions``.

    Matches ``exp/run_spawn_experiment.py::_execute_spawn_unit`` (around
    line 509): ``env.reset() → env.set_init_state(sim_state) → inner.timestep
    = ... → inner.cur_time = ...``. Returns ``(eef_traj, success_traj)``.
    """
    env.reset()
    env.set_init_state(ckpt_sim)
    inner.timestep = ckpt_timestep
    inner.cur_time = ckpt_cur_time
    if hasattr(inner, "done"):
        try:
            inner.done = False
        except AttributeError:
            pass
    traj_eef: list = []
    traj_success: list = []
    for a in post_actions:
        obs, _, _, info = env.step(a.tolist())
        traj_eef.append(obs["robot0_eef_pos"].copy())
        traj_success.append(bool(info.get("success", False)))
    return np.stack(traj_eef), traj_success


def _close_env_quietly(env) -> None:
    """Release MuJoCo/EGL handles; ignore teardown exceptions that robosuite
    raises at atexit on WSL GPU (harmless but very noisy)."""
    close = getattr(env, "close", None)
    if not callable(close):
        return
    try:
        # Silence EGL errors coming out of robosuite's renderer finaliser.
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stderr(devnull):
            close()
    except Exception as e:  # noqa: BLE001
        print(f"warning: env.close() raised {e!r}", file=sys.stderr)
    finally:
        try:
            devnull.close()
        except Exception:  # noqa: BLE001
            pass


def run(
    *,
    task_suite: str,
    task_id: int,
    init_state_idx: int,
    resolution: int,
    pre_steps: int,
    post_steps: int,
    seed: int,
    atol: float,
) -> None:
    env, init_states = _build_env(task_suite, task_id, resolution)
    try:
        # OffScreenRenderEnv is a wrapper: timestep/cur_time live on the
        # inner robosuite env at ``env.env`` (see examples/libero/main.py
        # lines 206-207 and exp/run_spawn_experiment.py line 509).
        inner = env.env

        env.reset()
        env.set_init_state(init_states[init_state_idx])

        rng = np.random.default_rng(seed)
        pre_actions = [rng.uniform(-0.1, 0.1, 7) for _ in range(pre_steps)]
        for a in pre_actions:
            env.step(a.tolist())

        ckpt_sim = env.get_sim_state().copy()
        ckpt_timestep = int(inner.timestep)
        ckpt_cur_time = float(inner.cur_time)

        post_actions = [rng.uniform(-0.1, 0.1, 7) for _ in range(post_steps)]

        # --- Live branch (reference, information only) --------------------
        traj_live_eef = []
        traj_live_success = []
        for a in post_actions:
            obs, _, _, info = env.step(a.tolist())
            traj_live_eef.append(obs["robot0_eef_pos"].copy())
            traj_live_success.append(bool(info.get("success", False)))
        live_arr = np.stack(traj_live_eef)

        # --- Replay A (Layer F teleport sequence) -------------------------
        replay_a_arr, replay_a_success = _replay_from_checkpoint(
            env,
            inner,
            ckpt_sim=ckpt_sim,
            ckpt_timestep=ckpt_timestep,
            ckpt_cur_time=ckpt_cur_time,
            post_actions=post_actions,
        )
        a_end_timestep = int(inner.timestep)

        # --- Replay B (same teleport, same actions) -----------------------
        replay_b_arr, replay_b_success = _replay_from_checkpoint(
            env,
            inner,
            ckpt_sim=ckpt_sim,
            ckpt_timestep=ckpt_timestep,
            ckpt_cur_time=ckpt_cur_time,
            post_actions=post_actions,
        )
        b_end_timestep = int(inner.timestep)

        # --- Informational drift vs live (MuJoCo does not promise bit-exact
        # equality between teleport and live continuation; this is reported
        # so operators can see order-of-magnitude drift per run). ---------
        drift_vs_live = float(np.abs(replay_a_arr - live_arr).max())
        print(
            f"info: max_eef_pos drift replayA-vs-live over {post_steps} "
            f"steps = {drift_vs_live:.3e} (not asserted — MuJoCo does not "
            f"guarantee teleport == live continuation)"
        )

        # --- Determinism assertion (this is what Layer F depends on) -----
        max_delta_replay = float(np.abs(replay_a_arr - replay_b_arr).max())
        print(
            f"max_eef_pos delta replayA-vs-replayB over {post_steps} steps "
            f"= {max_delta_replay:.3e} (atol={atol:.1e})"
        )
        if not np.allclose(replay_a_arr, replay_b_arr, atol=atol):
            raise AssertionError(
                f"restore determinism failed: replayA vs replayB "
                f"max_delta={max_delta_replay:.3e} > atol={atol:.1e}"
            )
        if replay_a_success != replay_b_success:
            raise AssertionError(
                "success flag drifted between replay A and replay B — "
                "restore is non-deterministic"
            )
        if a_end_timestep - ckpt_timestep != len(post_actions):
            raise AssertionError(
                f"replay A timestep bookkeeping wrong: "
                f"{a_end_timestep - ckpt_timestep} != {len(post_actions)}"
            )
        if b_end_timestep - ckpt_timestep != len(post_actions):
            raise AssertionError(
                f"replay B timestep bookkeeping wrong: "
                f"{b_end_timestep - ckpt_timestep} != {len(post_actions)}"
            )
        print("restore determinism: OK")
    finally:
        _close_env_quietly(env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-idx", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--pre-steps", type=int, default=30)
    ap.add_argument("--post-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="determinism tolerance replayA vs replayB (default 1e-6).",
    )
    args = ap.parse_args(argv)
    run(
        task_suite=args.task_suite,
        task_id=args.task_id,
        init_state_idx=args.init_state_idx,
        resolution=args.resolution,
        pre_steps=args.pre_steps,
        post_steps=args.post_steps,
        seed=args.seed,
        atol=args.atol,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
