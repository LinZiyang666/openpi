# LIBERO Remote Inference Guide

> **Status:** The code modifications described in this document have already been integrated into `examples/libero/main.py`. The current implementation includes episode lifecycle control (`episode_start`/`episode_end`), real-time display, and wall-clock video recording. **Do not replace main.py with the code below** — it is retained as a historical reference for the original design.
>
> **Current source of truth:** [`examples/libero/main.py`](../../examples/libero/main.py)

The simulator runs on the local WSL2 machine, while model inference runs on a remote GPU server, communicating via WebSocket.

```
[Local WSL2]                        [GPU Server]
  LIBERO simulator    <--websocket-->   π0.5 model inference
  main.py                              serve_policy.py
  port 9000 (via frp)                   port 8000 (model server)
```

Video recording strategy: Each frame is timestamped at capture time. When saving the video, frames are duration-weighted so that **inference latency is faithfully reflected** in the recorded video.

---

## 1. Code Modifications (Historical Reference)

> The changes below have already been applied. This section is kept for design context only.

The original upstream `examples/libero/main.py` only saves frames after `env.step()`, meaning inference latency is not captured in the video. The following modifications were made:

```python
import collections
import dataclasses
import logging
import math
import pathlib
import threading
import time

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RECORD_FPS = 30  # Recording frame rate, independent of simulation step rate


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    display: bool = False        # Whether to show a real-time rendering window (requires WSLg / X11)
    video_out_path: str = "exp/common/data/libero/videos"
    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan = collections.deque()

            # ── Recording state ──────────────────────────────────
            current_frame = [None]       # Shared frame read by background thread (list as mutable container)
            recorded_frames = []
            stop_event = threading.Event()

            def recorder_thread():
                while not stop_event.is_set():
                    t0 = time.perf_counter()
                    frame = current_frame[0]
                    if frame is not None:
                        recorded_frames.append(frame.copy())
                        if args.display:
                            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imshow("LIBERO", bgr)
                            cv2.waitKey(1)
                    elapsed = time.perf_counter() - t0
                    time.sleep(max(0.0, 1.0 / RECORD_FPS - elapsed))

            recorder = threading.Thread(target=recorder_thread, daemon=True)
            recorder.start()
            # ─────────────────────────────────────────────────────

            t = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        current_frame[0] = _get_display_frame(obs)
                        t += 1
                        continue

                    img = _preprocess_img(obs["agentview_image"], args.resize_size)
                    wrist_img = _preprocess_img(obs["robot0_eye_in_hand_image"], args.resize_size)
                    current_frame[0] = _get_display_frame(obs)  # Update display frame

                    if not action_plan:
                        # ── Background thread keeps repeating last frame during inference ──
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])
                        # ─────────────────────────────────────────────

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    current_frame[0] = _get_display_frame(obs)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # ── Stop recording and save video ────────────────────
            stop_event.set()
            recorder.join()
            if args.display:
                cv2.destroyAllWindows()

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4"
            if recorded_frames:
                imageio.mimwrite(str(video_path), recorded_frames, fps=RECORD_FPS)
                logging.info(f"Video saved: {video_path} ({len(recorded_frames)} frames)")
            # ─────────────────────────────────────────────────────

            task_episodes += 1
            total_episodes += 1
            logging.info(f"Success: {done}")
            logging.info(f"# episodes: {total_episodes}, # successes: {total_successes} "
                         f"({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Task success rate: {float(task_successes) / float(task_episodes):.2%}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.2%}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _preprocess_img(raw_img, resize_size):
    """Flip + resize to match training preprocessing."""
    img = np.ascontiguousarray(raw_img[::-1, ::-1])
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )


def _get_display_frame(obs):
    """Get the agentview image (flipped) for display/recording at original resolution."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
```

---

## 2. Server Side (GPU Machine)

Server-side setup is identical to aloha_sim.

### Installation

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

curl -LsSf https://astral.sh/uv/install.sh | sh

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Start the Policy Server

```bash
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

- Uses the `pi05_libero` config and explicitly loads the converted PyTorch checkpoint at `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
- Listens on `0.0.0.0:8000` by default — make sure port 8000 is open in the firewall
- ⚠ **Security**: the server has no authentication, and in `--concurrent` mode (the default) the `load_cache_config` control message accepts a client-supplied cache YAML whose `preload_path` is loaded via `pickle.load` with no path confinement — i.e. any client that can reach the port can achieve remote code execution as the server user. Only expose this port on a trusted network (not the public internet); treat the frp ingress as trusted-LAN-only. See [`logs/full_repo_audit_2026-05-26.log.md`](../../logs/full_repo_audit_2026-05-26.log.md) §3.1 for the accepted-risk record and the recommended allowlist fix.

---

## 3. Local WSL2 (Simulator)

### Install System Dependencies

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libglfw3 libglfw3-dev
```

For the real-time rendering window, also install:

```bash
sudo apt-get install -y libopencv-dev python3-opencv
```

### Create a Conda Environment

> **Python 3.8 is required.** LIBERO's dependency `numba==0.53.1` does not support Python 3.9+. You cannot reuse the aloha_sim conda environment.

```bash
conda create -n libero_sim python=3.8 -y
conda activate libero_sim
```

### Install Python Dependencies

```bash
cd /path/to/openpi

# Install dependencies (note: PyTorch requires the CUDA 11.3 index)
pip install -r examples/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install libero from PyPI (no git submodule needed)
pip install libero

pip install -e packages/openpi-client
```

---

## 4. Running

> **Note:** This script uses `tyro` nested argument parsing. All runtime arguments must be prefixed with `--`, e.g., `--host`, `--port`, `--display`. Using `--host` directly will cause an `Unrecognized options` error.

### Option A: Save Video Only (No Rendering Window, Most Stable)

```bash
cd /path/to/openpi
conda activate libero_sim
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

MUJOCO_GL=egl python examples/libero/main.py \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite-name libero_spatial
```

Video recording is off by default. Add `--save-video` to write
`exp/common/data/libero/videos/rollout_<task>_<success|failure>.mp4`.

### Option B: Real-Time Rendering Window + Video (Requires WSLg, Windows 11 Only)

```bash
MUJOCO_GL=egl python examples/libero/main.py \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite-name libero_spatial \
    --save-video \
    --display
```

> **Check if WSLg is available:** Run `echo $DISPLAY`. If there is output (e.g., `:0`), WSLg is available. Windows 10 does not support WSLg — use Option A instead.

### Common Parameters

| Parameter | Default | Description |
|---|---|---|
| `--host` | `155.98.36.32` | Server IP |
| `--port` | `9000` | Server port |
| `--task-suite-name` | `libero_spatial` | Task suite: `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` / `libero_90` |
| `--replan-steps` | `5` | Re-infer every N steps |
| `--num-trials-per-task` | `50` | Number of episodes per task |
| `--display` | `False` | Show real-time rendering window |
| `--save-video` | `False` | Save rollout videos in serial mode |
| `--video-out-path` | `exp/common/data/libero/videos` | Video output directory |
| `--init-states-dir` | `""` (use LIBERO default) | Custom init states directory (see §7) |

---

## 5. Verify Connection

Before starting the simulator, confirm the server is reachable:

```bash
nc -zv 155.98.36.32 9000
```

Start the local simulator only after the server log shows `Listening on port 8000`.

---

## 6. Video Recording Details

| Original Behavior | Modified Behavior |
|---|---|
| Frames saved only after `env.step()` | Background thread records continuously at **30fps** |
| Inference latency not captured in video | Last frame repeated during inference — latency fully reflected |
| No real-time rendering | `--display` flag opens a cv2 window |

The recording frame rate can be changed via `RECORD_FPS = 30` at the top of `main.py`.

---

## 7. Custom Init States

By default, `main.py` loads init states from LIBERO's built-in `.pruned_init` files (50 states per task). You can override this with `--init-states-dir` to use your own init states.

### LIBERO Default Init States

LIBERO provides two types of init state files per task:

| File type | Suffix | Description |
|-----------|--------|-------------|
| Full | `.init` | All collected init states (100 per task for most suites) |
| Pruned | `.pruned_init` | Subset used for standard evaluation (50 per task) |

| Suite | Tasks | `.init` per task | `.pruned_init` per task | State dim |
|-------|-------|-----------------|------------------------|-----------|
| libero_spatial | 10 | 100 | 50 | 92 |
| libero_object | 10 | 50 | 50 | 110 |
| libero_goal | 10 | N/A | 50 | 79 |
| libero_10 | 10 | 100 | 50 | 45-123 |
| libero_90 | 90 | 100 | 50 | 45-123 |

Default files are located at: `{libero_package}/libero/init_files/{suite_name}/`

### Using Custom Init States

1. Create a directory with your init state files, named `{task_name}.pruned_init` or `{task_name}.init` (PyTorch tensors, shape `[N, state_dim]`).

2. Run with `--init-states-dir`:

```bash
MUJOCO_GL=egl python examples/libero/main.py \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite-name libero_spatial \
    --init-states-dir /path/to/my/init_states/ \
    --num-trials-per-task 100
```

File lookup order: `.pruned_init` first, then `.init`. Set `--num-trials-per-task` to match your init state count.

### Generating New Init States

You can generate custom init states by repeatedly resetting the environment:

```python
import torch
from libero.libero.envs import OffScreenRenderEnv

env = OffScreenRenderEnv(bddl_file_name="path/to/task.bddl", camera_heights=256, camera_widths=256)
states = []
for i in range(100):
    env.seed(i)
    env.reset()
    states.append(env.env.sim.get_state().flatten())
torch.save(torch.stack(states), "task_name.init")
```

## Concurrent vs Non-concurrent serving (Phase 5 / M6)

``scripts/serve_policy.py`` now defaults to ``--concurrent``. Multiple
LIBERO workers share one server process; each connection binds to a
bundle by sending ``__ctrl__: select_bundle`` (or relying on the
``"default"`` bundle, the legacy fallback). ``__ctrl__: load_cache_config``
accepts an optional ``bundle_id`` field — the runner loads one bundle per
yaml, then dispatches workers to the right bundle.

Switch back to single-connection extreme-speed baseline with
``--non-concurrent`` (or ``--no-concurrent``). This path keeps the raw
single-connection structure (hard constraint C1: no ``BatchingCoordinator``,
no lazy lifecycle, no bundle indirection). Its numerics match the current sdpa
model, not the historical pre-Phase-5 eager baseline.

The runtime is write-frozen (hard constraint C2): cache backends refuse
``insert / batch_insert / delete / upsert / load_artifact`` after server
start, and ``write_policy`` MUST be ``"never"`` — any write-enabled policy
fails fast with a ``ConfigValidationError`` at server start and on every
``load_cache_config`` ctrl. Rebuild artifacts with offline tooling
(`exp/common/factor_postprocess.py`).

### Multi-replica scale-out + how many workers to run

One concurrent server is capped by the GIL-serialized CUDA kernel-launch path
(~12 inf/s on an A100). ``scripts/serve_policy.py --replicas N`` runs N server
processes behind one public port (a connection-sticky router) for ~N× the
throughput. On a memory-constrained host (jupyter, 32 GB host-RAM cgroup) add
``--replica-spawn-batch 2`` so the N model loads are staggered and don't OOM.

```bash
# a100 (exclusive, 40 GB): 3 replicas
BATCHING_MAX_WAIT_MS=25 BATCHING_MAX_BATCH_SIZE=32 \
python scripts/serve_policy.py --replicas 3 --port 8000 \
    --cache-config <yaml> policy:checkpoint --policy.config=pi05_libero --policy.dir=<dir>

# jupyter (H200, shared, 32 GB host RAM): 3 replicas, staggered
BATCHING_MAX_WAIT_MS=25 BATCHING_MAX_BATCH_SIZE=32 \
python scripts/serve_policy.py --replicas 3 --replica-spawn-batch 2 --port 8000 \
    --cache-config <yaml> policy:checkpoint --policy.config=pi05_libero --policy.dir=<dir>
```

LIBERO is closed-loop and ``--num-workers`` is capped at 15 per process (MuJoCo
EGL), so to drive a server with **N concurrent connections, launch N separate
``main.py`` processes** (``--num-workers 1`` each), one per tmux session. The
single public port fans connections across replicas automatically — the client
needs no replica/port awareness.

> **大规模实验改用新编排框架（推荐）**：上面"N 个 `main.py` 进程 + `replica_proxy` 单端口" 是遗留 client 编排。新的 **conductor 框架**（[教程](../experiments/conductor_tutorial.md) / [架构](../architecture/experiment_conductor.md)）由一个 driver 统一做 episode 级无空隙调度、按 server 分配 worker（48+48）、断点续跑/重试/监控。server 端点可为 **`--replicas N` 单公共端口**（`replica_proxy` 的 `fetch_dump` 已改为 aggregate——fan-out 到所有 child + 拼接各 replica 的 warmup dump 切片，warmup→eval 经 router 完整，driver 只注册一个端点）**或**多个独立单进程端点（各占一端口、按 server 细粒度分配 worker）。`--num-workers` 单进程多线程方式在迁移完成前仍可用于单机小规模实验。

**Auto-tuned optimal worker counts** (pi05_libero, phase5 cache mix):

| Server | replicas | client worker processes | `max_wait_ms` | throughput |
|--------|:--------:|:-----------------------:|:-------------:|:----------:|
| a100 (A100-40GB)  | 3 | **48** (clean, low backlog) | 25 | ~24 inf/s |
| jupyter (H200)    | 3 | **48** | 25 | ~31 inf/s |
| a100 + jupyter    | 3+3 | **48 + 48** from one sim host | 25 | ~48-51 inf/s |

并发 server 起法、调优、C1/C2、troubleshooting、`autotune_workers.py` re-tuning，
连同 client 编排（写 driver 策略），见端到端教程
[`docs/experiments/conductor_tutorial.md`](../experiments/conductor_tutorial.md)（§1 起 server / §8 调优 / §11 troubleshooting）。
