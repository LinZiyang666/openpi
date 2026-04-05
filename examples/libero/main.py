import collections
import dataclasses
import logging
import math
import pathlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                 *, record_video: bool = False) -> tuple:
    """Run a single episode.

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

        except Exception as e:
            logging.error(f"Caught exception: {e}")
            break

    return done, images, timestamps


def _eval_serial(args: Args, task_suite, num_tasks_in_suite, max_steps) -> None:
    """Original serial evaluation path (num_workers=1)."""
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    global_episode_id = 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
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


def _eval_concurrent(args: Args, task_suite, num_tasks_in_suite, max_steps) -> None:
    """Concurrent evaluation path (num_workers > 1).

    Each worker thread runs all episodes for one task at a time, then pulls
    the next unfinished task from a shared queue.  Video recording and display
    are disabled; a single tqdm bar tracks overall progress.
    """
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
    for task_id in range(num_tasks_in_suite):
        task_queue.put(task_id)

    # 3. Shared state for progress reporting.
    lock = threading.Lock()
    counters = {"episodes": 0, "successes": 0}
    total_episodes = num_tasks_in_suite * args.num_trials_per_task
    pbar = tqdm.tqdm(total=total_episodes, desc="Eval", unit="ep")

    # 4. Worker function.
    def worker():
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        try:
            while True:
                try:
                    task_id = task_queue.get_nowait()
                except queue.Empty:
                    break

                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

                for episode_idx in range(args.num_trials_per_task):
                    global_episode_id = task_id * args.num_trials_per_task + episode_idx
                    client.episode_start(
                        experiment=args.task_suite_name,
                        task=str(task_description),
                        episode_id=global_episode_id,
                    )

                    done, _, _ = _run_episode(
                        env, client, initial_states[episode_idx],
                        task_description, args, max_steps,
                        record_video=False,
                    )
                    client.episode_end(success=done)

                    with lock:
                        counters["episodes"] += 1
                        if done:
                            counters["successes"] += 1
                        pbar.update(1)
                        ep = counters["episodes"]
                        sr = counters["successes"] / ep
                        pbar.set_postfix(sr=f"{sr:.1%}")

                env.close()
        finally:
            client._ws.close()

    # 5. Launch workers.
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(worker) for _ in range(args.num_workers)]
        for f in as_completed(futures):
            f.result()  # re-raises worker exceptions

    pbar.close()
    ep = counters["episodes"]
    sr = counters["successes"]
    logging.info(f"Total success rate: {sr / ep:.1%} ({sr}/{ep})")


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    max_steps = _get_max_steps(args.task_suite_name)

    if args.num_workers > 1:
        _eval_concurrent(args, task_suite, num_tasks_in_suite, max_steps)
    else:
        _eval_serial(args, task_suite, num_tasks_in_suite, max_steps)


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
    tyro.cli(eval_libero)
