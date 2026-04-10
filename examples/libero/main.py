import collections
import dataclasses
import logging
import math
import os
import pathlib
import queue
import signal
import threading
import time
from typing import Dict, List, Tuple

import cv2
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
        (success, images, timestamps) — images/timestamps are empty lists
        when record_video is False.
    """
    env.reset()
    action_plan = collections.deque()
    obs = env.set_init_state(initial_state)

    images = []
    timestamps = []

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
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            if record_video:
                _record_step(obs, images, timestamps, args.display)

            if done:
                break
            t += 1

        except RuntimeError:
            raise  # Server error — let it propagate and stop the worker.
        except Exception as e:
            logging.error(f"Caught exception: {e}")
            break

    return done, images, timestamps


def _eval_serial(args: Args, task_suite, task_id_list: List[int], max_steps) -> None:
    """Original serial evaluation path (num_workers=1)."""
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    global_episode_id = 0
    for task_id in tqdm.tqdm(task_id_list):
        task = task_suite.get_task(task_id)
        initial_states = _load_init_states(task, task_suite, task_id, args.init_states_dir)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            client.episode_start(
                experiment=args.task_suite_name,
                task=str(task_description),
                episode_id=global_episode_id,
            )

            done, images, timestamps = _run_episode(
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

            if done:
                task_successes += 1
                total_successes += 1
            task_episodes += 1
            total_episodes += 1
            global_episode_id += 1

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _eval_concurrent(args: Args, task_suite, task_id_list: List[int], max_steps) -> None:
    """Concurrent evaluation path (num_workers > 1)."""
    num_tasks_in_suite = len(task_id_list)

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

    total_episodes = num_tasks_in_suite * args.num_trials_per_task
    pbar = tqdm.tqdm(total=total_episodes, desc="Total", unit="ep",
                     position=0, leave=True)
    worker_bars = []
    for i in range(args.num_workers):
        bar = tqdm.tqdm(total=max_steps, desc=f"W{i}: idle",
                        unit="step", position=i + 1, leave=True,
                        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [total:{postfix}]")
        bar.set_postfix_str("0")
        worker_bars.append(bar)

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
                        global_episode_id = task_id * args.num_trials_per_task + episode_idx

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

                        client.episode_start(
                            experiment=args.task_suite_name,
                            task=str(task_description),
                            episode_id=global_episode_id,
                        )

                        done, _, _ = _run_episode(
                            env, client, initial_states[episode_idx],
                            task_description, args, max_steps,
                            record_video=False,
                            step_callback=step_cb,
                        )
                        total_steps[0] += wbar.n
                        client.episode_end(success=done)

                        with lock:
                            counters["episodes"] += 1
                            if done:
                                counters["successes"] += 1
                            pbar.update(1)
                            ep = counters["episodes"]
                            sr = counters["successes"] / ep
                            aps = _update_rate()
                            pbar.set_postfix_str(f"sr={sr:.1%}, {aps:.1f} act/s")
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
