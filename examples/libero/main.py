import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import queue
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image, ImageDraw
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
VIDEO_FPS = 50


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    display: bool = False  # Show real-time render window (requires WSLg / X11)
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    init_states_dir: str = ""  # Custom init states directory. Expects {task_name}.pruned_init or .init files.

    #################################################################################################################
    # Task selection
    #################################################################################################################
    task_ids: Tuple[int, ...] = ()  # If non-empty, run only these task IDs (0-indexed). Default: all tasks.

    #################################################################################################################
    # Concurrency
    #################################################################################################################
    num_workers: int = 1  # Number of concurrent evaluation workers (1 = serial)

    #################################################################################################################
    # Trajectory-deviation experiment flags (plan §4 + §20.R1 + §19.B6)
    #################################################################################################################
    # Step 1b GT trajectory capture. When True, ``_run_episode`` buffers
    # (sim_state, imgs, robot_state, env_action_chunk, executed_actions) per
    # replan cycle and flushes to ``{save_trajectory_dir}/task_{id}/episode_{idx}.h5``.
    save_trajectory: bool = False
    save_trajectory_dir: str = "data/deviation_experiment/gt_trajectories"

    # Per-episode success JSON (Step 1a). On completion writes a list of
    # ``{task_id, init_state_idx, orig_init_state_idx, episode_id, seed,
    # success}`` records, one per executed episode.
    save_episode_results: bool = False
    episode_results_path: str = ""  # default: ``{task_suite_name}_episode_results.json``

    # Step 1b / Step 3 filter. JSON path to ``[{task_id, orig_init_state_idx,
    # subset_init_state_idx}, ...]``; only matching (task_id, episode_idx) are
    # executed. ``episode_idx`` inside the loop is the *subset* position.
    episode_filter: str = ""


def _get_max_steps(task_suite_name: str) -> int:
    """Return the maximum episode length for the given task suite."""
    limits = {
        "libero_spatial": 220,   # longest training demo has 193 steps
        "libero_object": 280,    # longest training demo has 254 steps
        "libero_goal": 300,      # longest training demo has 270 steps
        "libero_10": 520,        # longest training demo has 505 steps
        "libero_90": 400,        # longest training demo has 373 steps
    }
    if task_suite_name not in limits:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return limits[task_suite_name]


def _run_episode(env, client, initial_state, task_description, args, max_steps,
                 *, record_video: bool = False, step_callback=None) -> tuple:
    """Run a single episode.

    Args:
        step_callback: Optional callable(step_number) invoked each sim step
                       for live progress display.

    Returns:
        ``(success, images, timestamps, traj_buffer, final_env_timestep)``.
        ``traj_buffer`` is a list of per-replan-cycle dicts when
        ``args.save_trajectory`` is True (schema: plan §20.R1.2); otherwise
        ``None``. ``final_env_timestep`` is the real ``env.env.timestep``
        at loop exit (or the python-side step counter as a fallback for
        stub envs without the robosuite inner attribute). G2 §26 MF-3
        requires downstream spawn runs to use this as the env-step budget
        anchor rather than inferring ``num_steps_wait + num_steps``.

    Trajectory layout (plan §20.R1):
        For each replan cycle we record a dict with
        ``sim_state / env_timestep / env_cur_time / agentview_image /
        eye_in_hand_image / robot_state / env_action_chunk / executed_actions``.
        ``executed_actions`` has shape ``(K, 7)`` where K is the number of
        actions actually ``env.step``ed from this cycle (``1 <= K <=
        replan_steps``). Mid-chunk ``done=True`` finalises the last cycle with
        the partial ``_pending_executed`` (§20.R1.1).
    """
    env.reset()
    action_plan = collections.deque()
    obs = env.set_init_state(initial_state)

    images = []
    timestamps = []

    traj_buffer: Optional[List[Dict[str, Any]]] = [] if args.save_trajectory else None
    # Actions popped from ``action_plan`` since the last infer() call. Flushed
    # into ``traj_buffer[-1]["executed_actions"]`` on the next infer() boundary
    # and on end-of-loop (see §20.R1.1).
    _pending_executed: List[np.ndarray] = []

    t = 0
    done = False

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                if record_video:
                    _record_step(obs, images, timestamps, args.display)
                t += 1
                continue

            if step_callback is not None:
                step_callback(t - args.num_steps_wait)

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            if not action_plan:
                # Replan boundary: finalise the previous cycle's executed_actions
                # *before* kicking the next infer so traj_buffer stays consistent
                # even if infer() raises (the cycle just completed has real
                # data; drop only the cycle we haven't started yet).
                if traj_buffer is not None and traj_buffer and traj_buffer[-1]["executed_actions"] is None:
                    traj_buffer[-1]["executed_actions"] = np.stack(_pending_executed).astype(np.float32)
                    _pending_executed = []

                robot_state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": robot_state,
                    "prompt": str(task_description),
                }
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])

                if traj_buffer is not None:
                    # OffScreenRenderEnv exposes ``get_sim_state`` directly,
                    # but ``timestep`` / ``cur_time`` live on the inner
                    # robosuite env (``env.env``). Verified against
                    # libero_sim conda env: OffScreenRenderEnv ->
                    # Libero_Tabletop_Manipulation has both attrs.
                    traj_buffer.append(
                        {
                            "sim_state": np.asarray(env.get_sim_state()).copy(),
                            "env_timestep": int(env.env.timestep),
                            "env_cur_time": float(env.env.cur_time),
                            "agentview_image": img.copy(),
                            "eye_in_hand_image": wrist_img.copy(),
                            "robot_state": np.asarray(robot_state, dtype=np.float64).copy(),
                            "env_action_chunk": np.asarray(action_chunk, dtype=np.float32).copy(),
                            # Filled in on the next replan boundary / end-of-loop.
                            "executed_actions": None,
                        }
                    )

            action = action_plan.popleft()
            action_np = np.asarray(action, dtype=np.float32)
            obs, reward, done, info = env.step(action.tolist())
            if record_video:
                _record_step(obs, images, timestamps, args.display)

            if traj_buffer is not None:
                _pending_executed.append(action_np.copy())

            if done:
                break
            t += 1

        except RuntimeError:
            raise  # Server error — let it propagate and stop the worker.
        except Exception as e:
            logging.error(f"Caught exception: {e}")
            break

    # §20.R1.1: mid-chunk / end-of-loop finaliser. The tail cycle's
    # ``executed_actions`` is still None because we only flush on the *next*
    # replan boundary. If ``_pending_executed`` is empty (rare: infer returned
    # and ``done=True`` fired on the *next* loop iter before any env.step),
    # drop the cycle rather than write a zero-length dataset — downstream
    # ``load_gt_first_actions`` requires K >= 1.
    if traj_buffer is not None and traj_buffer and traj_buffer[-1]["executed_actions"] is None:
        if _pending_executed:
            traj_buffer[-1]["executed_actions"] = np.stack(_pending_executed).astype(np.float32)
        else:
            traj_buffer.pop()

    # G2 §26 MF-3: capture the real env-timestep at exit so downstream spawn
    # runs can compute a ``final - anchor`` budget on the exact same axis.
    # OffScreenRenderEnv forwards timestep via env.env.timestep; stub envs
    # without the inner attr fall back to the loop counter ``t`` (wall-equal
    # when num_steps_wait=0, and only used as a last resort).
    inner = getattr(env, "env", None)
    final_env_timestep = int(getattr(inner, "timestep", t)) if inner is not None else int(t)

    return done, images, timestamps, traj_buffer, final_env_timestep


def _compute_global_episode_id(task_id: int, episode_idx: int, num_trials_per_task: int) -> int:
    """Derive the global episode id from (task_id, subset episode idx).

    Both serial and concurrent evaluation paths MUST agree on this mapping
    (plan §4 / §19.B6): if they diverge, client-side GT HDF5, server-side
    collected HDF5, results JSON, and downstream Layer F unit keys fall out
    of alignment as soon as the caller uses ``--task_ids`` or
    ``--episode_filter`` to skip episodes. Keep as the only place this
    formula lives.
    """
    return int(task_id) * int(num_trials_per_task) + int(episode_idx)


def _count_filtered_episodes(
    filter_pairs: Optional[set],
    task_id_list: List[int],
    num_trials_per_task: int,
) -> int:
    """Compute expected episode count for the concurrent progress bar.

    G2 deferred should-fix: ``_eval_concurrent`` previously set
    ``total_episodes = num_tasks × num_trials_per_task`` unconditionally, so
    when the user passed ``--episode_filter step1b_filter.json`` (plan
    §19.B6) the overall ``tqdm`` bar promised many more episodes than were
    actually scheduled. We now mirror the worker's own ``continue`` guard:
    when a filter is active, count only pairs whose task is in
    ``task_id_list``. Extracted as a pure helper so tests can pin the math
    without constructing a real LIBERO env.
    """
    if filter_pairs is None:
        return len(task_id_list) * int(num_trials_per_task)
    task_id_set = set(task_id_list)
    return sum(1 for tid, _ in filter_pairs if tid in task_id_set)


def _load_episode_filter(
    path: str,
) -> Tuple[Optional[set], Dict[Tuple[int, int], int]]:
    """Load a ``step1b_filter.json`` (plan §19.B6) into two lookup structures.

    Returns:
        ``(filter_pairs, orig_map)``. ``filter_pairs`` is a set of
        ``(task_id, subset_init_state_idx)`` used to decide whether to run an
        episode; ``orig_map`` maps the same key to ``orig_init_state_idx`` so
        HDF5 output can record the original index even though the loop index
        is the subset position.  When ``path`` is empty returns
        ``(None, {})`` (i.e. "run every episode").
    """
    if not path:
        return None, {}
    entries = json.loads(pathlib.Path(path).read_text())
    filter_pairs = {(int(e["task_id"]), int(e["subset_init_state_idx"])) for e in entries}
    orig_map = {
        (int(e["task_id"]), int(e["subset_init_state_idx"])): int(e["orig_init_state_idx"])
        for e in entries
    }
    return filter_pairs, orig_map


def _flush_trajectory_h5(
    out_dir: str,
    task_id: int,
    subset_init_state_idx: int,
    *,
    task_name: str,
    orig_init_state_idx: int,
    episode_id: int,
    seed: int,
    success: bool,
    replan_steps: int,
    num_steps_wait: int,
    final_env_timestep: int,
    traj_buffer: List[Dict[str, Any]],
) -> Optional[pathlib.Path]:
    """Write one episode's ``traj_buffer`` to HDF5 (plan §20.R1.2 schema).

    Returns the written path, or ``None`` if ``traj_buffer`` is empty (no
    infer cycle completed — nothing worth recording). Writes under
    ``{out_dir}/task_{task_id}/episode_{subset_init_state_idx}.h5`` so downstream
    pipelines index by (task, subset position).

    G2 §26 MF-3 additions (env-timestep budget anchor for Layer F spawn):
      - ``num_steps_wait``       — ``args.num_steps_wait`` (warm-up offset).
      - ``final_env_timestep``   — real ``env.env.timestep`` at loop exit
        (captured by ``_run_episode``). Consumers use this to compute
        ``budget = final_env_timestep - start_env_timestep`` directly on
        the env-step axis, instead of fragile ``num_steps_wait + num_steps``
        arithmetic.
    """
    if not traj_buffer:
        return None
    out_path = (
        pathlib.Path(out_dir)
        / f"task_{task_id}"
        / f"episode_{subset_init_state_idx}.h5"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["task_name"] = task_name
        f.attrs["task_id"] = int(task_id)
        f.attrs["init_state_idx"] = int(subset_init_state_idx)
        f.attrs["orig_init_state_idx"] = int(orig_init_state_idx)
        f.attrs["episode_id"] = int(episode_id)
        f.attrs["seed"] = int(seed)
        f.attrs["num_cycles"] = len(traj_buffer)
        f.attrs["num_steps"] = int(
            sum(int(step["executed_actions"].shape[0]) for step in traj_buffer)
        )
        f.attrs["success"] = bool(success)
        f.attrs["replan_steps"] = int(replan_steps)
        # G2 §26 MF-3 schema additions.
        f.attrs["num_steps_wait"] = int(num_steps_wait)
        f.attrs["final_env_timestep"] = int(final_env_timestep)
        for i, step in enumerate(traj_buffer):
            g = f.create_group(f"step_{i:04d}")
            for k, v in step.items():
                if k == "executed_actions":
                    g.create_dataset(k, data=v)
                    g.attrs["executed_action_count"] = int(v.shape[0])
                else:
                    g.create_dataset(k, data=v)
    return out_path


def _write_episode_results_json(
    path: str,
    task_suite_name: str,
    rows: List[Dict[str, Any]],
) -> pathlib.Path:
    """Write the per-episode success JSON (plan §4 改动 5).

    When ``path`` is empty, defaults to ``{task_suite_name}_episode_results.json``
    in the current directory. Rows are written as-is so callers control the
    column set (``task_id`` / ``init_state_idx`` / ``orig_init_state_idx`` /
    ``episode_id`` / ``seed`` / ``success``).
    """
    out = pathlib.Path(path) if path else pathlib.Path(f"{task_suite_name}_episode_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    return out


def _eval_serial(args: Args, task_suite, task_id_list: List[int], max_steps) -> None:
    """Original serial evaluation path (num_workers=1)."""
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Plan §19.B6: episode_filter selects which (task_id, subset_init_state_idx)
    # pairs to run and remembers their original indices for HDF5 attrs.
    filter_pairs, orig_map = _load_episode_filter(args.episode_filter)
    per_episode_log: List[Dict[str, Any]] = []

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_id_list):
        task = task_suite.get_task(task_id)
        initial_states = _load_init_states(task, task_suite, task_id, args.init_states_dir)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            if filter_pairs is not None and (task_id, episode_idx) not in filter_pairs:
                continue
            orig_init_state_idx = orig_map.get((task_id, episode_idx), episode_idx)
            global_episode_id = _compute_global_episode_id(
                task_id, episode_idx, args.num_trials_per_task
            )

            logging.info(f"\nTask: {task_description}")
            # Plan §20.R1 / §4 改动 4: when saving trajectories, name the HDF5
            # artefacts by (task, subset-episode) so the Layer-B-4 collector
            # mirrors the naming scheme on the policy side.
            episode_name = (
                f"task_{task_id}/episode_{episode_idx}"
                if args.save_trajectory
                else ""
            )
            client.episode_start(
                experiment=args.task_suite_name,
                task=str(task_description),
                episode_id=global_episode_id,
                episode_name=episode_name,
            )

            done, images, timestamps, traj_buffer, final_env_timestep = _run_episode(
                env, client, initial_states[episode_idx],
                task_description, args, max_steps,
                record_video=True,
            )

            if args.display:
                cv2.destroyAllWindows()

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            out_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4"
            _save_video(images, timestamps, out_path)
            client.episode_end(success=done)

            if args.save_trajectory and traj_buffer is not None:
                _flush_trajectory_h5(
                    args.save_trajectory_dir,
                    task_id=task_id,
                    subset_init_state_idx=episode_idx,
                    task_name=str(task_description),
                    orig_init_state_idx=orig_init_state_idx,
                    episode_id=global_episode_id,
                    seed=args.seed,
                    success=bool(done),
                    replan_steps=args.replan_steps,
                    num_steps_wait=int(args.num_steps_wait),
                    final_env_timestep=int(final_env_timestep),
                    traj_buffer=traj_buffer,
                )

            if args.save_episode_results:
                per_episode_log.append(
                    {
                        "task_id": int(task_id),
                        "init_state_idx": int(episode_idx),
                        "orig_init_state_idx": int(orig_init_state_idx),
                        "episode_id": int(global_episode_id),
                        "seed": int(args.seed),
                        "success": bool(done),
                    }
                )

            if done:
                task_successes += 1
                total_successes += 1
            task_episodes += 1
            total_episodes += 1

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        if task_episodes > 0:
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    if args.save_episode_results:
        out = _write_episode_results_json(
            args.episode_results_path, args.task_suite_name, per_episode_log
        )
        logging.info(f"Wrote per-episode results to {out} ({len(per_episode_log)} rows)")


def _eval_concurrent(args: Args, task_suite, task_id_list: List[int], max_steps) -> None:
    """Concurrent evaluation path (num_workers > 1)."""
    # 1. Check server supports concurrent mode.
    probe_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_meta = probe_client.get_server_metadata()
    if not server_meta.get("concurrent", False):
        probe_client._ws.close()
        raise RuntimeError(
            "Server does not support concurrent mode. "
            "Start the server with --concurrent to enable multi-worker evaluation."
        )
    probe_client._ws.close()

    # 2. Build task queue.
    task_queue: queue.Queue[int] = queue.Queue()
    for task_id in task_id_list:
        task_queue.put(task_id)

    # 3. Progress bars: one overall + one per worker.
    lock = threading.Lock()
    counters = {"episodes": 0, "successes": 0, "steps": 0}
    step_times: collections.deque[float] = collections.deque()

    def _update_rate() -> float:
        """Recalc actions/s over a 10s sliding window. Caller must hold lock."""
        now = time.monotonic()
        cutoff = now - 10.0
        while step_times and step_times[0] < cutoff:
            step_times.popleft()
        if len(step_times) < 2:
            return 0.0
        span = step_times[-1] - step_times[0]
        return (len(step_times) - 1) / span if span > 0 else 0.0

    # Plan §19.B6: load the (task, subset-idx) filter once up-front; shared
    # read-only across workers so no lock is required for the lookups.
    filter_pairs, orig_map = _load_episode_filter(args.episode_filter)

    # G2 deferred should-fix: the progress bar total must reflect how many
    # episodes workers will actually run. See ``_count_filtered_episodes``
    # for the formula; extracted as a pure helper for unit testing.
    total_episodes = _count_filtered_episodes(
        filter_pairs, task_id_list, args.num_trials_per_task
    )
    pbar = tqdm.tqdm(total=total_episodes, desc="Total", unit="ep",
                     position=0, leave=True)
    worker_bars = []
    for i in range(args.num_workers):
        bar = tqdm.tqdm(total=max_steps, desc=f"W{i}: idle",
                        unit="step", position=i + 1, leave=True,
                        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [total:{postfix}]")
        bar.set_postfix_str("0")
        worker_bars.append(bar)
    per_episode_log: List[Dict[str, Any]] = []

    # 4. Worker function.
    init_lock = threading.Lock()  # Serialize env creation + server connection

    def worker(worker_id, stop: threading.Event):
        wbar = worker_bars[worker_id]
        total_steps = [0]  # mutable container for closure access
        with init_lock:
            client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        try:
            while not stop.is_set():
                try:
                    task_id = task_queue.get_nowait()
                except queue.Empty:
                    wbar.set_description_str(f"W{worker_id}: done")
                    break

                task = task_suite.get_task(task_id)
                initial_states = _load_init_states(task, task_suite, task_id, args.init_states_dir)
                with init_lock:
                    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                short_desc = task_description[:30]

                try:
                    for episode_idx in range(args.num_trials_per_task):
                        if stop.is_set():
                            break
                        if filter_pairs is not None and (task_id, episode_idx) not in filter_pairs:
                            continue
                        orig_init_state_idx = orig_map.get((task_id, episode_idx), episode_idx)
                        global_episode_id = _compute_global_episode_id(
                            task_id, episode_idx, args.num_trials_per_task
                        )

                        wbar.reset(total=max_steps)
                        wbar.set_description_str(
                            f"W{worker_id}: T{task_id} E{episode_idx} {short_desc}"
                        )

                        def step_cb(step, _wbar=wbar, _ts=total_steps):
                            _wbar.n = step
                            _wbar.set_postfix_str(str(_ts[0] + step))
                            _wbar.refresh()
                            with lock:
                                counters["steps"] += 1
                                step_times.append(time.monotonic())
                                if counters["steps"] % 20 == 0:
                                    aps = _update_rate()
                                    ep = counters["episodes"]
                                    sr = f"{counters['successes'] / ep:.1%}" if ep else "N/A"
                                    pbar.set_postfix_str(f"sr={sr}, {aps:.1f} act/s")

                        episode_name = (
                            f"task_{task_id}/episode_{episode_idx}"
                            if args.save_trajectory
                            else ""
                        )
                        client.episode_start(
                            experiment=args.task_suite_name,
                            task=str(task_description),
                            episode_id=global_episode_id,
                            episode_name=episode_name,
                        )

                        done, _, _, traj_buffer, final_env_timestep = _run_episode(
                            env, client, initial_states[episode_idx],
                            task_description, args, max_steps,
                            record_video=False,
                            step_callback=step_cb,
                        )
                        total_steps[0] += wbar.n
                        client.episode_end(success=done)

                        if args.save_trajectory and traj_buffer is not None:
                            # HDF5 write is per-episode and targets a
                            # task-scoped subdirectory, so cross-worker
                            # collisions are impossible (two workers never
                            # own the same task_id). No extra lock needed.
                            _flush_trajectory_h5(
                                args.save_trajectory_dir,
                                task_id=task_id,
                                subset_init_state_idx=episode_idx,
                                task_name=str(task_description),
                                orig_init_state_idx=orig_init_state_idx,
                                episode_id=global_episode_id,
                                seed=args.seed,
                                success=bool(done),
                                replan_steps=args.replan_steps,
                                num_steps_wait=int(args.num_steps_wait),
                                final_env_timestep=int(final_env_timestep),
                                traj_buffer=traj_buffer,
                            )

                        with lock:
                            counters["episodes"] += 1
                            if done:
                                counters["successes"] += 1
                            pbar.update(1)
                            ep = counters["episodes"]
                            sr = counters["successes"] / ep
                            aps = _update_rate()
                            pbar.set_postfix_str(f"sr={sr:.1%}, {aps:.1f} act/s")

                            if args.save_episode_results:
                                # Under ``lock`` so concurrent appends from
                                # multiple workers stay race-free.
                                per_episode_log.append(
                                    {
                                        "task_id": int(task_id),
                                        "init_state_idx": int(episode_idx),
                                        "orig_init_state_idx": int(orig_init_state_idx),
                                        "episode_id": int(global_episode_id),
                                        "seed": int(args.seed),
                                        "success": bool(done),
                                    }
                                )
                finally:
                    env.close()
        finally:
            client._ws.close()

    # 5. Launch workers.
    stop_event = threading.Event()
    threads: List[threading.Thread] = []
    for i in range(args.num_workers):
        t = threading.Thread(target=worker, args=(i, stop_event), daemon=True)
        t.start()
        threads.append(t)

    def _force_exit(signum, frame):
        """Second Ctrl+C: force immediate exit."""
        os._exit(1)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        logging.info("\nInterrupted — shutting down workers...")
        stop_event.set()
        signal.signal(signal.SIGINT, _force_exit)
        # Drain task queue so workers exit their loop
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
            except queue.Empty:
                break
        # Give workers a moment to notice stop_event, then force exit
        for t in threads:
            t.join(timeout=2.0)
        os._exit(0)

    for bar in worker_bars:
        bar.close()
    pbar.close()

    ep = counters["episodes"]
    sr = counters["successes"]
    if ep > 0:
        logging.info(f"Total success rate: {sr / ep:.1%} ({sr}/{ep})")
    else:
        logging.info("No episodes completed.")

    if args.save_episode_results:
        # Sort for determinism: concurrent workers append in completion order.
        rows = sorted(
            per_episode_log,
            key=lambda r: (r["task_id"], r["init_state_idx"]),
        )
        out = _write_episode_results_json(
            args.episode_results_path, args.task_suite_name, rows
        )
        logging.info(f"Wrote per-episode results to {out} ({len(rows)} rows)")


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # Resolve task ID list (filter or all).
    if args.task_ids:
        task_id_list = list(args.task_ids)
        for tid in task_id_list:
            if tid < 0 or tid >= num_tasks_in_suite:
                raise ValueError(f"task_id {tid} out of range [0, {num_tasks_in_suite})")
        logging.info(f"Running subset of tasks: {task_id_list}")
    else:
        task_id_list = list(range(num_tasks_in_suite))

    max_steps = _get_max_steps(args.task_suite_name)

    if args.num_workers > 5:
        logging.warning("num_workers capped at 5 (MuJoCo EGL context limit). Using 5.")
        args.num_workers = 5

    if args.num_workers > 1:
        _eval_concurrent(args, task_suite, task_id_list, max_steps)
    else:
        _eval_serial(args, task_suite, task_id_list, max_steps)


def _record_step(obs, images, timestamps, display):
    """Record agentview frame and wall-clock timestamp after each env.step()."""
    im = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    images.append(im)
    timestamps.append(time.monotonic())
    if display:
        bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow("LIBERO", bgr)
        cv2.waitKey(1)


def _save_video(images, timestamps, out_path):
    """Repeat each frame proportionally to its real wall-clock duration, then save."""
    if not images:
        return

    frames = []
    for i, (im, ts) in enumerate(zip(images, timestamps)):
        if i + 1 < len(timestamps):
            step_duration = timestamps[i + 1] - ts
        else:
            step_duration = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 1.0 / VIDEO_FPS

        latency_ms = step_duration * 1000
        annotated = _draw_latency(im, latency_ms)

        repeat = max(1, round(step_duration * VIDEO_FPS))
        frames.extend([annotated] * repeat)

    logging.info(f"Saving video to {out_path} ({len(frames)} frames @ {VIDEO_FPS}fps = {len(frames)/VIDEO_FPS:.1f}s real time)")
    imageio.mimwrite(str(out_path), frames, fps=VIDEO_FPS)


def _draw_latency(im: np.ndarray, latency_ms: float) -> np.ndarray:
    """Draw latency overlay on frame using PIL."""
    img = Image.fromarray(im.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    text = f"{latency_ms:.0f} ms"
    x, y = 8, 8

    # shadow for readability
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 80, 80))

    return np.array(img)


def _load_init_states(task, task_suite, task_id, init_states_dir: str):
    """Load init states from custom directory if specified, otherwise use default."""
    if not init_states_dir:
        return task_suite.get_task_init_states(task_id)
    base = pathlib.Path(init_states_dir)
    pruned = base / f"{task.name}.pruned_init"
    full = base / f"{task.name}.init"
    if pruned.exists():
        path = pruned
    elif full.exists():
        path = full
    else:
        raise FileNotFoundError(f"No init states found for {task.name} in {init_states_dir}")
    import torch
    states = torch.load(path, weights_only=False)
    logging.info(f"Loaded {len(states)} init states from {path}")
    return states


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
