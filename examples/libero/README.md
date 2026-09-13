# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## 大规模 / 跨机实验编排（conductor）

上面的 `python examples/libero/main.py` 是**单机快速验证**入口（进程内多线程，单卡 ≤15 worker）。

对**大规模评测**（跨卡 / 跨机 / 多 server / warmup→eval / 断点续跑 / 重试 / 监控），改用新的**实验编排框架**：你只写一个 `ExperimentStrategy`（实验剧本），由通用 driver 做 episode 级无空隙调度，LIBERO 执行内核复用 [`episode_runner.py`](episode_runner.py)，worker 进程入口为 [`worker_entry.py`](worker_entry.py)（由 `WorkerAgent` 在各机 fork、直连 driver pull 端口）。

- **上手教程（重点：如何编写 driver 策略）**：[`docs/experiments/conductor_tutorial.md`](../../docs/experiments/conductor_tutorial.md)
- 架构与设计：[`docs/architecture/experiment_conductor.md`](../../docs/architecture/experiment_conductor.md)

> 旧的 `--num-workers N` 单进程多线程方式仅适合单机小规模；它无法跨卡（单进程钉一个 `CUDA_VISIBLE_DEVICES`）且单卡 ≤15 worker。跨卡/跨机一律用 conductor。

## `--save-episode-results` 记录格式

`--save-episode-results [--episode-results-path <json>]` 在 serial 与 concurrent 两条路径下写出同一 schema（每集一行）：

| 字段 | 含义 |
|---|---|
| `task_id` / `init_state_idx` / `orig_init_state_idx` / `episode_id` / `seed` / `success` | 原有身份与成功位（`init_state_idx` 是循环内的 subset 位置，`orig_init_state_idx` 来自 `--episode-filter` 映射） |
| `task_suite_name` | 运行的 suite |
| `termination_reason` | `_run_episode` 实际退出分支：`success`（env 报 done）/ `step_cap`（步数上限 `max_steps + num_steps_wait` 用尽）/ `exception`（一般异常 break）；`RuntimeError` 仍向上抛出，不形成记录 |
| `client_timing.steps` / `client_timing.infers` | 该集 env.step 次数（**含等待段**）与 `client.infer` 次数 |
| `max_steps` / `num_steps_wait` / `replan_steps` | suite 的步数上限与本次运行参数 |

后三组是 additive 的终态证据，供离线验收（如 `exp/libero_groot/verify_shadow_h5.py`）把 client 终态与采集侧 HDF5 绑定：HDF5 的步数只能证明内部一致，"正常到达上限的失败"与"截断"只有 client 侧能区分。

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
