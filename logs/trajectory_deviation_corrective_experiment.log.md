# Trajectory Deviation 纠偏实验 — 详细实验计划

> Status: Plan
> Date: 2026-04-13
> Task: 验证 cache 轨迹偏差的局部性假设，量化 inference 纠偏效果
> 关联文档: [trajectory_deviation_experiment_plan.log.md](trajectory_deviation_experiment_plan.log.md)

---

## 1. 实验目标

验证核心假设：**cache 轨迹失败是少数 deviate points 的局部偏差引发的级联效应，在这些点注入 inference 纠偏即可恢复轨迹成功**。

---

## 2. 术语定义

| 术语 | 定义 |
|------|------|
| **GT trajectory** | 纯 inference 跑出的成功轨迹。包含每步的 (env_state, observation, action, model_intermediates) |
| **背景 L2 距离** | 在 GT trajectory 某一步上，多次独立 inference 产出 action 之间的平均 L2 距离。衡量模型在该 step 的内在随机性（噪声水平） |
| **Cache L2 距离** | 在 GT trajectory 某一步上，cache 产出的 action 与 GT action 的 L2 距离 |
| **Deviate score** | Cache L2 距离 / 背景 L2 距离。信噪比。≈ 1 表示 cache 偏差在模型噪声范围内，>> 1 表示异常偏差 |
| **Deviate point** | Deviate score 显著大于 1 的 step |
| **Intervention** | 在某个 deviate point 处，不执行 cache action，改为沿 GT trajectory rollout n 步 |
| **Spawn** | 在 intervention 完成后的 env state 上新建环境，续跑纯 cache |
| **Trajectory depth (D)** | Cache 系统的 trajectory search 使用的历史窗口大小 |

---

## 3. 实验流水线总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1a: Cache 评估 — 找到失败 episode                                   │
│   纯 cache (always_hit) 跑 LIBERO，记录每个 episode 的 success/fail       │
│   同时记录每个 episode 使用的 init_state index                             │
│   → 输出: cache 失败 episode 列表 (task_id, init_state_idx, seed)         │
└─────────────────────┬───────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1b: GT 收集 — 只针对 cache 失败的 init 跑 inference                  │
│   对 Step 1a 筛选出的 (task, init_state) 组合跑纯 inference               │
│   收集成功轨迹的完整数据 (env_state + obs + action + model_intermediates)  │
│   → 输出: GT trajectories (仅覆盖 cache 失败的 episode)                   │
└─────────────────────┬───────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: 离线分析 — 计算 Deviate Score                                    │
│   对每条 GT trajectory 的每一步:                                          │
│     ① 重放 GT observation → 跑 M 次 inference → 背景 L2                  │
│     ② 重放 GT observation → 跑 1 次 cache    → Cache L2                 │
│   → 输出: 每步的 deviate score                                           │
└─────────────────────┬───────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: Oracle 纠偏 — Spawn 实验                                        │
│   选 top-k deviate points → 沿 GT rollout n 步 → spawn 环境 → 跑纯 cache │
│   扫描 k × n × D 参数组合                                                │
│   → 输出: success rate vs (k, n, D) 数据                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Step 1: 数据收集（Cache First → 定向收集 GT）

### 4.0 Init States 机制与数据对齐

**Init States 来源**：默认从 LIBERO 包内 `libero/init_files/<suite>/` 加载（`task_suite.get_task_init_states(task_id)`），也可通过 `--init-states-dir` 指定自定义目录（按 `{task.name}.pruned_init` 或 `.init` 查找）。每 task 50 个 init state，shape `(50, 92)`，由 `torch.load()` 加载。

**索引规则**：`init_state_idx = episode_idx`（task 内序号，`main.py:186`）。实验用前 5 个（idx 0~4）。

**Episode ID 编码**：`episode_id = task_id * num_trials_per_task + episode_idx`（concurrent 模式 `main.py:291`；serial 模式结果一致）。

**Client-Server 数据对齐 — `episode_name` 接口**：

当前 server `data_collector.py` 自动命名 `episode_{episode_id:04d}_{ts}.h5`，不存 `init_state_idx`。为避免反算，在 `episode_start` 接口新增 `episode_name` 参数（4 个文件透传，每个 < 5 行）：

```
client.episode_start(..., episode_name="task_3/episode_2")
  → websocket → server → data_collector
  → 保存为 {collect_dir}/{experiment}/task_3/episode_2.h5
```

Client-side trajectory 也保存为 `gt_trajectories/task_3/episode_2.h5`，**天然同名对齐**。不传 `episode_name` 时走原有命名逻辑，向后兼容。

**⚠️ Step 1a/1b 必须使用相同的 `--init-states-dir`**（或都用默认），否则同一 `(task_id, init_state_idx)` 对应不同场景。

### 4.1 Step 1a: 跑纯 Cache，找到失败 Episode

先跑纯 cache 评估，目的是找出哪些 (task, init_state) 组合下 cache 会失败。

**配置**：
- 服务器：`serve_policy.py --concurrent`，加载当前最优 cache 配置（`gate: always_search`, `judge: always_hit`）
- 客户端：`main.py`，正常评估流程
- 全量跑：10 tasks × 5 episodes = 50 episodes
- **Init states**：使用默认 pruned_init 或指定 `--init-states-dir`（两步必须一致）

**记录**：每个 episode 的完整标识保存到 JSON。

```json
// data/deviation_experiment/cache_eval_results.json
[
  {"task_id": 0, "init_state_idx": 0, "episode_id": 0, "seed": 42, "success": true},
  {"task_id": 0, "init_state_idx": 1, "episode_id": 1, "seed": 42, "success": false},
  {"task_id": 1, "init_state_idx": 0, "episode_id": 5, "seed": 42, "success": false},
  ...
]
```

`episode_id` 同时记录，用于与 server-side HDF5（如果开了 `--collect`）对齐。

### 4.2 Step 1b: 只对 Cache 失败的 Init 跑纯 Inference 收集 GT

编排脚本 `collect_gt_trajectories.py` 从 `cache_eval_results.json` 筛选 `success==false` 的 episode，生成 `failed_episodes.json`（`[{task_id, init_state_idx}, ...]`），然后调用 `main.py --save-trajectory --episode-filter failed_episodes.json` 只跑这些 episode。

Inference 在同一 init_state 下也可能失败，直接跳过（不满足"inference 成功 AND cache 失败"前提）。

### 4.3 GT Trajectory 每步保存的数据

| 数据 | 来源 | 用途 |
|------|------|------|
| `sim_state` | `env.get_sim_state()` — flattened numpy array | Step 3 spawn 恢复环境 |
| `env_timestep` / `env_cur_time` | `env.timestep` / `env.cur_time` | spawn 恢复 robosuite 内部计数器 |
| `agentview_image` / `eye_in_hand_image` | obs dict | Step 2 重放给 inference/cache |
| `robot_state` | eef_pos + eef_quat + gripper_qpos → 8D | Step 2 重放给 inference/cache |
| `action` | 模型输出的 action chunk | GT action，计算 L2 距离的参考 |
| model intermediates | server-side `--collect` flag 保存 | 离线回放 cache key builder |

`sim_state` 是新增需求（现有 data_collector 只保存 model-side 数据）。Client-side 保存 env 数据，server-side 保存 model 数据，通过 `episode_name` 对齐。

### 4.4 env state save/restore 验证（Phase 0）

LIBERO 已内置 `get_sim_state()` / `set_init_state()` API（§7 调研已确认）。Phase 0 需端到端验证：跑 50 步 → save checkpoint → 跑 50 步 → restore → 固定 seed 跑 50 步 → 验证轨迹一致。

Checkpoint save/restore 核心代码：
```python
# 保存
checkpoint = {"mujoco_state": env.get_sim_state(), "timestep": env.timestep, "cur_time": env.cur_time}
# 恢复
obs = env.set_init_state(checkpoint["mujoco_state"])  # 物理状态 + sim.forward() + 刷新观测
env.timestep, env.cur_time, env.done = checkpoint["timestep"], checkpoint["cur_time"], False
```

### 4.5 存储格式

Client-side HDF5（env 数据）：`gt_trajectories/task_{id}/episode_{idx}.h5`

```
metadata/: task_name, task_id, init_state_idx, episode_id, seed, num_steps, success, cache_success
step_0000/:
  sim_state: [N] float64          # env.get_sim_state() flattened
  env_timestep: int                # robosuite step counter
  env_cur_time: float64            # robosuite elapsed time
  agentview_image: [H,W,3] uint8
  eye_in_hand_image: [H,W,3] uint8
  robot_state: [8] float32
  action: [action_horizon, action_dim] float32
```

Server-side HDF5（model 数据，`--collect` flag）：同名 `task_{id}/episode_{idx}.h5`，包含 vision_emb, prompt_emb, noise_actions 等。

---

## 5. Step 2: 离线计算 Deviate Score

### 5.1 计算方法

对 GT trajectory 每一步 t，重放 observation（images + robot_state），分别计算：

- **背景 L2**：跑 M=10 次 inference（不同 noise sampling），取 C(10,2)=45 对两两 L2 均值
- **Cache L2**：按与 Step 3 相同的 `trajectory_depth=D` 跑 cache 检索，**history buffer 沿 GT 轨迹预填**（每步查询前 `record_query_keys()` + `record_action(GT_action[t-D..t-1])`），cache action 与 GT action 的 L2
- **Deviate score**：`cache_l2 / max(background_l2, floor)`，floor=0.1 防止除零

背景 L2 ≈ 0 的 step（模型输出几乎确定性）标记为 deterministic step，单独统计。

**Per-D 计算**：deviate score 随 D 变化（同一步在不同 D 下 cache 检索到的 action 不同），因此 Step 2 对每个候选 D ∈ {1, 3, 5} 各算一套 score。Step 3 扫 D 时用**对应 D** 的 deviate point 列表。D=1 时无 history 可填，等价于每步独立检索。

每步 M=10 次 inference 与 cache 检索开销相比可忽略；~220 步/轨迹 × 10 次 = ~2200 次 inference/episode，约 11 分钟/轨迹（0.3s/次）。Cache 部分 ×|D| 但无 inference 开销，可控。

### 5.2 产出与诊断分析

每条轨迹输出 JSON：`{background_l2[t], deterministic[t], per_D: {D: {cache_l2[t], deviate_score[t]}}}`（背景 L2 与 D 无关，共享一份）。

诊断分析：deviate score 分布直方图、top-k 覆盖率、位置分布、连续性、跨轨迹一致性。

**Go/No-Go 判定**：
- Go：deviate score >> 1 的 step 占比 < 20%，且存在明显长尾
- No-Go：分布平坦，无明显分离 → 假设不成立

---

## 6. Step 3: Oracle 纠偏 — Spawn 实验

### 6.1 实验参数

| 参数 | 符号 | 扫描范围 | 说明 |
|------|------|---------|------|
| 纠偏点数量 | k | 1, 2, 3, 5 | 选 deviate score 最高的 k 个 step 做 intervention |
| Rollout 长度 | n | 1, 3, 5, 10, 20 | intervention 后沿 GT trajectory rollout 几步 |
| Trajectory depth | D | 由配置决定：clip/w7 → D=4；spatial16/w8 → D=4；max_pool/w3 → D=5 | spawn 后 cache 的 trajectory search 历史窗口；每个配置取各自在 trajectory 实验中的最优 D，与 Step 2 计算 deviate score 时使用的 D 一致 |

### 6.2 单次 Spawn 实验流程

对 intervention point `s`、rollout 长度 `n`、trajectory depth `D`：

1. 恢复 env 到 `step_(s+n)` 的 sim_state + timestep + cur_time
2. `on_episode_start()` 初始化 cache orchestrator
3. 如果 D > 1：用 GT 的 `[step_(s+n-D+1) ... step_(s+n-1)]` 预填充 trajectory history（对每步调用 `key_builder.collect()` + `build()` → `record_query_keys()` + `record_action()`）
4. 从 `step_(s+n)` 开始跑纯 cache 直到 episode 结束
5. 记录 success/fail

**历史衰退**：预填充后前 D-1 步 history 混合 GT+cache，从第 D 步起 history 完全由 cache 产出。

### 6.3 多 Intervention Points 与 Baseline

**k > 1 策略**：先做策略 A（每个 intervention point 独立 spawn），验证单点效果。后续可做策略 B（从最后一个 deviate point spawn）或策略 C（连续 spawn 链，最接近真实但实现复杂）。

**连续 deviate 合并**：相邻 step 间距 ≤ merge_gap(=3) 时合并为一个 window，rollout 从 window 末尾开始。

**Baseline 对照组**：pure cache、pure inference、random-k（随机选 k 个 step）、equidistant-k（等间距选 k 个 step）。

---

## 7. LIBERO 环境可行性（Phase 0 前置调研结论）

**结论：完全可行。** LIBERO `ControlEnv` 已内置 save/restore API，无需 hack。

**API**：`env.get_sim_state()` 返回 flattened numpy array（qpos+qvel+time），`env.set_init_state(state)` 恢复物理 + `sim.forward()` + 刷新观测 + 返回 obs。

**必须额外保存**的 robosuite 变量：`env.timestep`（step 计数器，控制终止）、`env.cur_time`（累积时间）、`env.done`（restore 时设 False）。

**无需担心**：task goal 是 per-task 固定的，`_check_success()` 基于 `sim.data` 自动正确；controller 内部状态不影响结果。

**注意**：用 `get_sim_state()`（返回 flattened array，可存 HDF5）而非 `sim.get_state()`（返回 MjSimState 对象，不可跨版本 pickle）。始终用 `set_init_state()` 而非手动 `set_state()`（前者自动调 `sim.forward()`）。

---

## 8. 实验执行计划

### 8.1 固定配置

| 配置 | 值 |
|------|---|
| Task Suite | libero_spatial (10 tasks) |
| Episodes per task | 5 |
| Max steps per episode | 220 |
| Seed | 42 |
| Model | Pi0.5 LIBERO checkpoint |
| Cache 候选配置 | 3 个并行：`clip_w7_d4`（D=4）、`spatial16_w8_d4`（D=4）、`max_pool_w3_d5`（D=5），YAML 位于 `configs/cache_runs/deviate_exp/` |
| Cache Artifact | `data/cache_artifacts/libero_spatial/`（三个 key_builder 各需一份预构建 artifact）|
| M (inference 重复次数) | 10 |

### 8.2 执行顺序

```
Phase 0: 环境验证（§7 调研已确认 API 可用，需端到端验证）
  └── P0-1: 跑 §4.4 最小验证脚本，确认 checkpoint save/restore 后轨迹完全一致
  └── P0-2: 验证 restore 后 env._check_success() 判定、timestep 计数、episode 终止均正确

Phase 1: 数据收集 (Cache First)
  └── P1-1: 跑纯 cache (10 tasks × 5 episodes)，记录每个 episode 的成功/失败 + init_state_idx
  └── P1-2: 筛选 cache 失败的 (task_id, init_state_idx) 组合
  └── P1-3: 只对筛选出的组合跑纯 inference，收集 GT trajectories（含 env sim_state）
  └── P1-4: 过滤掉 inference 也失败的 episode，得到最终实验集

Phase 2: Deviate Score 计算
  └── P2-1: 对最终实验集计算背景 L2 (每步 M 次 inference)
  └── P2-2: 对最终实验集计算 cache L2 (每步 1 次 cache)
  └── P2-3: 计算 deviate score + 诊断分析
  └── P2-4: Go/No-Go 判定

Phase 3: Spawn 实验 (Phase 2 Go 之后)
  └── P3-1: 参数扫描 k × n × D，独立 spawn（策略 A）
  └── P3-2: Baseline 对照组 (random-k, equidistant-k)
  └── P3-3: 结果分析
```

### 8.3 预计工作量

| Phase | 人工 | GPU 时间 | 说明 |
|-------|------|---------|------|
| P0 | 0.5d | < 10min | 最小验证脚本 |
| P1-1 | 0.5d | ~30min | 纯 cache 评估 50 episodes，复用现有实验框架 |
| P1-2~4 | 1d | ~30min | 修改 main.py + 定向跑 inference（只跑失败 episode，数量远少于 50） |
| P2 | 1d | ~数小时 | 每条轨迹 ~2200 次 inference |
| P3 | 1d | ~数小时 | spawn 实验数量 = episodes × k_values × n_values × D_values |

---

## 9. 需要新建/修改的代码

### 9.1 总览

| 文件 | 类型 | 复杂度 | 说明 |
|------|------|--------|------|
| `packages/.../websocket_client_policy.py` | 修改 | 低 | `episode_start()` 新增 `episode_name` 参数 |
| `src/.../websocket_policy_server.py` | 修改 | 低 | 透传 `__episode_name__` 到 policy |
| `src/.../collection_policy.py` | 修改 | 低 | `on_episode_start()` 透传 `episode_name` |
| `src/.../data_collector.py` | 修改 | 低 | 支持 client 指定 HDF5 文件名（向后兼容） |
| `examples/libero/main.py` | 修改 | 低 | 增加 `--save-trajectory` flag + episode 结果 JSON + `episode_name` 调用 |
| `src/openpi/cache/cache_storage.py` | 修改 | 低 | facade 层新增 prefill 模式：`enter_prefill_mode(payload)` / `exit_prefill_mode()`，并在 `search()` / `fetch_payload()` 首行加 prefill 分支合成返回值（backend contract 不变）|
| `src/openpi/cache/interceptor.py` | 修改 | 中 | 新增 `prefill_trajectory(observations, actions, record, on_miss)` 方法 |
| `src/.../websocket_policy_server.py` | 修改 | 低 | 新增 `__ctrl__ == "prefill_trajectory"` 控制消息分支 |
| `packages/.../websocket_client_policy.py` | 修改 | 低 | 新增 `prefill_trajectory(observations, actions=None, record=False, on_miss="error")` 方法 |
| `src/.../collection_policy.py` | 修改 | 低 | 从 obs 提取 raw prompt 字符串，传给 collector（首步即可定，整 episode 常量） |
| `src/.../data_collector.py` | 修改 | 低 | Episode 级保存 `prompt` 字段（HDF5 episode attrs，平台无关 prefill 所需） |
| `src/openpi/cache/components/gate.py` | 修改 | 低 | 新增 `AlwaysSkipGate`（`__call__` return False），用于 Step 2a 强制走 inference；trajectory history 仍自然累积（见 `orchestrator.py:219-226`） |
| `src/openpi/cache/config.py` | 修改 | 低 | `gate.type` 接受 `"always_skip"`；`_build_gate` 分支 |
| `configs/cache_runs/deviate_exp/inference_*.yaml` | 新建 | 低 | 3 份（clip / spatial16 / max_pool），各自 `gate.type: always_skip`，其余字段与对应 cache YAML 一致（同 key_builder、同 weights、同 D，保证 trajectory history 语义一致） |
| `scripts/dump_step1a_failed_inits.py` | 新建 | 低 | 读 Step 1a 的 `episode_results.json` → 按 task 筛出失败 `(task_id, init_idx)` → `torch.save` 到 `data/step1b_inits/libero_spatial/{task_name}.init`；Q1 选 A，失败 init 以后可复用 |
| `exp/_cache_config_rpc.py` | 新建 | 低 | WebSocket 控制消息统一封装：`send_load_cache_config` + Step 3 的 `send_prefill_begin / send_prefill_end`；`run_cache_experiments.py` 迁移过去 |
| `exp/_run_state_base.py` | 新建 | 低 | 通用 RunState JSON 持久化骨架 + `--resume` + `_retry_failed_runs` 模板（三种 runner 都 import）|
| `exp/run_step1b_gt.py` | 新建 | 低 | Step 1b thin dispatcher：对每个 task 分批调用 `main.py --init-states-dir data/step1b_inits/... --collect ...`，实现 per-(task, init_idx) 粒度 resume + 失败重试 |
| `exp/compute_deviate_scores.py` | 新建 | **高** | Step 2：多连接并行 replay GT obs 序列，分两阶段（背景 L2 M-连接 + cache L2 1-连接），中间切 YAML，详见 §13.4 |
| `exp/run_spawn_experiment.py` | 新建 | **高** | Step 3：spawn 实验 + 调用 `prefill_trajectory`（详见 §13） |
| `exp/analyze_deviation_results.py` | 新建 | 低 | 统计分析 + 可视化 |

### 9.2 `episode_name` 接口（4 文件透传，每个 < 5 行）

`episode_start()` 新增 `episode_name: str = ""` 参数，沿 client → websocket(`__episode_name__`) → server → collection_policy → data_collector 透传。data_collector 中：若 `episode_name` 非空，HDF5 命名为 `{episode_name}.h5`（支持子目录）；否则走原有 `episode_{id:04d}_{ts}.h5` 逻辑。

### 9.3 `examples/libero/main.py` 改动

`Args` 新增：`save_trajectory` / `save_trajectory_dir` / `save_episode_results` / `episode_results_path` / `episode_filter`（JSON 文件指定要跑的 `(task_id, init_state_idx)` pairs）。

关键插入点（`_run_episode()`, line 146 `env.step()` 之后）：每步保存 `{sim_state, env_timestep, env_cur_time, images, robot_state, action}` 到 buffer，episode 结束写 HDF5。

GT 收集时 `episode_start()` 传 `episode_name=f"task_{task_id}/episode_{init_state_idx}"`，使 server-side model intermediates HDF5 与 client-side trajectory HDF5 同名对齐。

### 9.4 新建脚本

**`exp/collect_gt_trajectories.py`**：Step 1a/1b 编排。调用 main.py 跑纯 cache → 筛选失败 → 调用 main.py --save-trajectory --episode-filter 跑纯 inference → 过滤 inference 也失败的 → 输出 `experiment_set.json`。复用 `run_cache_experiments.py` 的 subprocess + JSON state 模式。

**`exp/compute_deviate_scores.py`**：Step 2。两阶段并行 replay，详见 §13.4。
- **阶段 1（背景 L2）**：发 `load_cache_config` 切到 `inference_<config>.yaml`（`gate.type: always_skip`）→ 开 N 个并发连接；每个 worker 处理一个 `(episode, sample_idx)` unit：从 HDF5 读该 episode 的 GT obs 序列 → 按步逐个发给 server 做 inference → 记录每步 action。M 条连接 per episode 即为该 episode 每步的 M 个背景样本。
- **阶段 2（cache L2）**：发 `load_cache_config` 切到正常 cache YAML → 开 N 个并发连接；每个 worker 处理一个 `(episode,)` unit：replay GT obs 序列，记录每步 cache 返回的 action。每 episode 只 1 次（cache 是 deterministic）。
- **阶段 3（聚合）**：离线读所有 action samples → 计算 background L2（同步两两 L2 距离均值）+ cache L2（vs GT action）→ deviate_score = cache_L2 / background_L2 → 输出 `deviate_score.json`。
- **关键语义**：每个连接独立 rollout 意味着 trajectory history 由连接自己的 infer 累积（喂的是 GT obs 序列，所以 history 和部署时语义一致）；**不能在一个连接里对同一点 M 次 infer**（trajectory depth history 会污染）。

**`exp/run_spawn_experiment.py`**（最复杂）：Step 3。参数扫描 k×n，对每个组合：选 top-k deviate points → 对每个 point s：`env.set_sim_state(gt[s+n].sim_state)`（teleport，不真跑 inference，见 §11 选项 A）→ 调用 `prefill_trajectory` 灌入 D-1 步 GT 历史（`obs_list` + `clean_action_list` 均从 HDF5 读）→ 纯 cache 从 s+n 跑到终点 → 记录 success。同时跑 random-k/equidistant-k baseline。**脚本内不调用 `interceptor.infer()`**。

**`exp/analyze_deviation_results.py`**：统计分析。输出 deviate score 分布、top-k 覆盖率、success rate vs (k,n) 热力图、baseline 对比等图表。

### 9.5 Cache 框架扩展：Prefill 模式 — Storage facade 层实现

**动机**：轨迹局部重放（spawn 实验、local replay、敏感性分析等）会反复用到"在 episode 中途把 cache 所有内部状态设置成好像过去 D-1 步真的发生过一样"的能力。与其作为实验一次性 hack，不如作为 cache 框架的正式能力发布。

**关键架构观察**：cache 系统中多个组件各自持有 episode 状态（**每个部件自己决定是否保存 trajectory**），不只是 `TrajectoryMixin`：

| 组件 | 状态字段 | 说明 |
|------|---------|------|
| `TrajectoryMixin`（search_strategy）| `_query_history`, `_action_history` | 构造 QuerySpec 的 trajectory 字段 |
| `CP1TemporalPruneKeyBuilder`（key_builder）| `_history: dict[str, _VisionHistoryBuffer]` | build() 的 temporal pruning 依赖它 —— **直接影响 query_keys 本身** |
| Orchestrator | `_step_counter` | 影响 SearchContext.current_step，可能影响 step_filter 策略 |
| Judge / Gate | 当前实现是 no-op，但有 hooks | 未来 trajectory-aware 组件的扩展点 |

因此"直接戳 `TrajectoryMixin` 内部 buffer"的设计是错的 —— 会绕过 key_builder 的 vision 历史，导致第一次正式查询时 `_VisionHistoryBuffer` 尚未 ready，temporal pruning 退化或输出错误 query_keys。

**正确设计（已定）**：**prefill 语义实现在 `CacheStorage` storage facade 层；framework 管线代码和 backend contract 完全不变**。所有上游对存储层的读操作都经过 `CacheStorage`（唯一入口，见 facade 文档），所以 prefill 分支只需要在 `search()` / `fetch_payload()` 最开头加 if 判断即可完整拦截。Framework 照常走完 collect→gate→build→search→judge→fetch 整条管线（让所有组件的 history 自然更新），但 facade 在 prefill 模式下返回合成 hit + 指定 payload，framework 不感知自己在 prefill 中。Judge 的决策无所谓（通常合成 hit 会被判 FULL_HIT，但无论什么决策都只作用在"丢弃的返回值"上）。

**为什么 facade 层而不是 backend**：
- 零 backend 改动：`VectorStoreBackend` ABC 不变，`in_memory_backend.py` / `qdrant_backend.py` / 未来新 backend 全部零接入成本
- 无代码重复：合成逻辑单一真相，不需要 N 份
- 语义正确：prefill 是"存储层对上游如何应答"的概念，不是存储本身的概念；backend 应保持纯存储契约
- 拦截面极小：`CacheStorage` 对外读 API 只有 `search` / `fetch_payload` / `search_and_fetch`，其中 `search_and_fetch` 内部调用 `self.search + self.fetch_payload`，自动继承 prefill 语义

**CacheStorage facade 新增**：

```python
class CacheStorage:
    def __init__(self, backend, metadata_db=None):
        ...
        self._prefill_mode: bool = False
        self._prefill_payload: CachePayload | None = None

    # ---- Prefill 模式（facade-only；backend 不感知）----
    def enter_prefill_mode(self, payload: CachePayload) -> None:
        """Enter prefill mode. Until exit_prefill_mode():
          - search(spec) returns exactly one synthetic SearchResultLite with
            score=1.0 (guarantees FULL_HIT under any reasonable judge threshold).
          - fetch_payload(id) returns the stored payload, ignoring id.
          - insert()/delete() still forward to backend; callers must not mutate
            during prefill (not enforced).
        Backend contract unchanged. Idempotent: calling while already in prefill
        replaces payload.
        """
        self._prefill_mode = True
        self._prefill_payload = payload

    def exit_prefill_mode(self) -> None:
        """Resume normal search/fetch. No-op if not currently in prefill."""
        self._prefill_mode = False
        self._prefill_payload = None

    def search(self, spec):
        self._check_query_dims(spec)   # 仍然校验：捕获 prefill 侧 query_keys 构造 bug
        self._check_filters(spec)
        if self._prefill_mode:
            return [SearchResultLite(
                id="__prefill__",
                score=1.0,
                checkpoint_id=spec.checkpoint_id,
            )]
        return self._backend.search(spec)

    def fetch_payload(self, id: str) -> CachePayload:
        if self._prefill_mode:
            return self._prefill_payload
        return self._backend.fetch_payload(id)
```

**流程（framework 代码完全不动）**：

```
for (obs, action) in prefill trajectory:
    cache_storage.enter_prefill_mode(payload_for(action))
    try:
        interceptor.infer(obs)   # 走完整 inference；丢弃返回值
        # 管线内部自然发生：
        #   key_builder.collect() → _VisionHistoryBuffer.push ✓
        #   key_builder.build() → query_keys（依赖上一步的 buffer）✓
        #   strategy.search() → facade 返回合成 hit → record_query_keys 触发 ✓
        #   judge() → 返回某个决策，无所谓 ✓
        #   fetch_payload() → facade 返回 prefill 指定 payload ✓
        #   _step_counter += 1 ✓
        #   interceptor.broadcast_action(prefill_action) → 所有组件的 record_action 触发 ✓
    finally:
        cache_storage.exit_prefill_mode()
```

**上层新 API**（`InferenceInterceptor`）：

```python
def prefill_trajectory(
    self,
    observations: list[dict],
    actions: list[np.ndarray] | None = None,   # None = 真 search，用 cache 返回值填（未实现）
    record: bool = False,                       # 是否将 synthetic 步写 HDF5（未实现）
    on_miss: str = "error",                     # 未实现；首版合成模式下不可能 MISS
) -> None:
    """Drive the cache framework along (obs, action) as if those steps really
    happened. After this call, all stateful components (key_builder vision
    buffer, strategy trajectory buffer, step_counter, and any future
    trajectory-aware gate/judge) are consistent with that trajectory.

    Use cases: spawn 实验 prefill D-1 步 GT 历史；local replay / 敏感性分析；
    未来 intervention / debugging 实验。

    First-version supports: actions provided, record=False, on_miss="error".
    Other combinations raise NotImplementedError (see docstring notes).
    """
    if actions is None:
        raise NotImplementedError(
            "actions=None (cache self-query mode). Future use case: run real "
            "search per step, record cache's own returned actions as history. "
            "Needed for 'what would cache remember if running pure-cache' analyses."
        )
    if record:
        raise NotImplementedError(
            "record=True. Future use case: capture prefill synthetic steps into "
            "HDF5 tagged as 'prefill' for audit/debugging. Requires "
            "CollectionPolicy to distinguish prefill from real inference steps."
        )
    if on_miss != "error":
        raise NotImplementedError(
            f"on_miss={on_miss!r}. Future use case: 'warn' allows sparse history "
            "on MISS; 'fallback_infer' runs real inference on MISS. First version "
            "uses synthetic-hit backend, so MISS cannot happen here."
        )
    for obs, action in zip(observations, actions):
        self._cache_storage.enter_prefill_mode(_build_prefill_payload(action))
        try:
            self.infer(obs)
        finally:
            self._cache_storage.exit_prefill_mode()
```

**WebSocket 协议层**：新增 `__ctrl__ == "prefill_trajectory"` 控制消息；字段 `observations`、`actions`、`record`、`on_miss`。Client 侧 `websocket_client_policy.py` 暴露同名方法。

**需要改动的文件（修订版）**：

| 文件 | 改动 |
|------|------|
| `src/openpi/cache/cache_storage.py` | 新增 `enter_prefill_mode(payload)` / `exit_prefill_mode()` 方法 + `_prefill_mode` / `_prefill_payload` 成员；`search()` / `fetch_payload()` 首行加 prefill 分支返回合成值 |
| `src/openpi/cache/interceptor.py` | 新增 `prefill_trajectory(...)` 方法，内部 enter/exit `cache_storage` 的 prefill 模式并跑完整 `self.infer(obs)` |
| `src/openpi/serving/websocket_policy_server.py` | 新增 `__ctrl__ == "prefill_trajectory"` 分支 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | 新增 `prefill_trajectory(...)` 方法 |
| `src/openpi/collect/collection_policy.py` | 从 obs 提取 raw prompt 字符串，传给 collector |
| `src/openpi/collect/data_collector.py` | Episode 级保存 `prompt` 字段（HDF5 episode attrs，首步即可定） |

（注：`VectorStoreBackend` ABC / `in_memory_backend.py` / `qdrant_backend.py` / `TrajectoryMixin` / key_builder / judge / gate **全部不需要改** —— 这是本设计最大的优势。）

**`--collect` 侧小扩展（平台无关的 prompt 收集）**：

当前 `--collect` 只存 `prompt_emb`（embedding），prefill 无法还原原始字符串。方案：在 `CollectionPolicy` 入口从 `obs["prompt"]` 取字符串，episode 级别写入 HDF5 attrs（整 episode 常量，一次写入，体积 ~几十字节/episode）。这使 prefill driver 只需读 HDF5 就能重建完整 obs dict，**不依赖任何 platform-specific task metadata API**（如 libero 的 `task.language`）—— 任何环境（Libero / ALOHA / DROID）通过 `--collect` 产出的 HDF5 都能直接用于 prefill。

**五个设计点的当前决定**：

| 设计点 | 首版 | 未来扩展方向 |
|--------|------|------------|
| 管线是否真跑 | 跑完整 infer；facade 合成 hit + payload；stage2/3 浪费可接受（D-1 次） | 可加快捷路径只跑到 CP1 以节省算力 |
| actions 来源 | 外部必给 | `actions=None` 时 facade 不进 prefill 模式，跑真 search，事后用返回 action 入 history |
| 采集交互 | 不采集 | `record=True` 在 CollectionPolicy 中加 prefill 标记 |
| MISS 处理 | 合成路径下不可能 MISS；其它模式抛错 | `warn` / `fallback_infer` 配合 `actions=None` |
| 生命周期地位 | **不是**第三个钩子；prefill 是 `CacheStorage` facade 的一个临时模式，由调用者显式 enter/exit | 同左 |

**推进策略（已定）**：**首版实现 + API 形状完整**。API signature 按完整版设计（`actions`/`record`/`on_miss` 参数全预留），首版只实现"actions 必给 + record=False + on_miss='error'"这一条路径，其它组合抛 `NotImplementedError`。未来补实现其它分支时 **API / 协议 / client 都不用改**。

**首版代码注释要求**：每个 `NotImplementedError` 分支必须注明（1）未实现的语义、（2）未来用例、（3）为什么暂不支持。

---

## 10. 风险

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| 假设不成立：偏差均匀分布 | 高 | Phase 2 Go/No-Go 判定 |
| Init states 不一致导致数据错位 | 高 | 编排脚本强制校验 `--init-states-dir` 一致 |
| 背景 L2 计算耗时（每步 M 次 inference） | 中 | 可先用 M=5 做初步分析 |
| History 预填充 key building 一致性 | 中 | 方案 A（server-side）保证一致 |
| Spawn 后 history 过渡期影响 | 中 | D=1 作为 baseline |
| Env state restore 不完整 | 低 | §7 已确认 API 可用，Phase 0 端到端验证 |

---

## 11. 待确认

**已确认**：
- env.sim 路径（§7）、get_sim_state() 返回 flattened array 可存 HDF5（§7）、init_state_idx = episode_idx（§4.0）、episode_id 编码规则（§4.0）。
- 纯 cache 配置：**3 个候选全部纳入实验，各自取其 trajectory 实验最优 D**，各自独立完成 Step 1/2/3，最终三组对照。YAML 位于 `configs/cache_runs/deviate_exp/`：

  | 配置 | key_builder | weights (v0/v1/rs) | D | Traj SR |
  |------|-------------|--------------------|---|---------|
  | clip_w7_d4 | clip | 0.1 / 0.1 / 0.8 | 4 | 68% |
  | spatial16_w8_d4 | cp1_spatial_pool_16 | 0.5 / 0.25 / 0.25 | 4 | 68% |
  | max_pool_w3_d5 | cp1_max_pool | 0.5 / 0 / 0.5 | 5 | 60% |

- Step 1a 现有 trajectory 实验数据不可复用：`experiment_state.json` 只有聚合 `task_results: [success_count, total]`，log 只有 total success rate，均无 per-episode `(task_id, init_state_idx, success)` 记录。Step 1a 必须重跑（顺便按 §9.3 加 `--save-episode-results`）。
- 三配置 artifact 已齐备，位于 `data/cache_artifacts/libero_spatial/`：`clip_vit_b_32.pkl`（188 MB）、`cp1_spatial_pool_16.pkl`（412 MB）、`cp1_max_pool.pkl`（37 MB）。
- Step 1a **不**收集 cache 轨迹的 per-step action，只记 `{task_id, init_state_idx, success}`。理由：本实验前提是 deviate point 由 GT 推出，面对一条纯 cache 轨迹我们无法直接识别 deviate point（需在该轨迹上跑 inference 才能算 deviate score），这属于后续工作，不在本实验范围。

- History 预填充：采用方案 A（server-side），**语义实现在 `CacheStorage` storage facade 层**（`enter_prefill_mode` / `exit_prefill_mode` + `search`/`fetch_payload` 开头加 prefill 分支），framework 管线和 `VectorStoreBackend` contract 完全不变，**任何 backend 零接入成本**。Framework 照常跑完整管线产生所有 side effect（key_builder vision buffer、strategy trajectory buffer、step_counter、broadcast_action 等），facade 在 prefill 模式下返回合成 hit（score=1.0） + 指定 payload，judge 决策无所谓。首版实现 actions 必给 + record=False + on_miss="error"；API 形状完整保留 `actions=None` / `record` / `on_miss` 参数。详见 §9.5。

- **Prefill obs 重建来源**：HDF5 里 `input_images/base_0_rgb` + `input_images/left_wrist_0_rgb` + `robot_state` 已足够；图片是 post-transform 224×224 uint8，通过 `_parse_image` 幂等，直接喂回 pre-transform key（如 libero 的 `observation/image`）即可。Prompt 字符串通过**扩展 `--collect` 保存到 episode attrs**（方案 B）实现平台无关，不依赖 `task_suite.get_task(task_id).language` 这类 platform-specific API。

- **Intervention 语义**：Step 3 的"介入 n 步"采用**选项 A（GT 替换 / oracle）**。`env.set_sim_state(gt[s+n].sim_state)` 直接 teleport 到 intervention 窗口末端，不真跑 `env.step()` 也不真调 inference；`prefill_trajectory` 用 HDF5 里 `clean_action[s+n-D+1 .. s+n-1]` 灌 D-1 步 GT 历史；纯 cache 从 s+n 接手跑到终点。
  - 关键语义澄清：GT trajectory 本身就是 Step 1b 用纯 inference 跑出来的成功轨迹，HDF5 `clean_action[t]` **就是**模型对 `obs[t]` 的 inference 输出。因此"介入时真跑 inference"（方案 B）= 把已存盘的 inference 结果丢掉再跑一遍，只会引入额外随机噪声，不带来新信息。
  - 对 §9.4 `run_spawn_experiment.py` 的影响：实现路径是"teleport env + `prefill_trajectory(obs_list, action_list)` + 纯 cache 续跑"，**整个 Step 3 流程不包含任何 `interceptor.infer()` 调用**。

- **Background L2 采样 seed 控制**：**不控 seed，直接复用 `CollectionPolicy` 落盘**。Step 2 算背景 L2 时对每个 obs 跑 M 次 inference，每次经过 `CollectionPolicy` 自然写入 HDF5（`clean_action` + `noise_action_steps` 已被既有 hook 捕获），下游分析脚本读 HDF5 计算两两 L2 距离均值即可。Flow matching 自带噪声采样（`sample_noise` 走 torch 全局 RNG）保证 M 个样本天然不同；具体数值不可复现但比值排序（Monte Carlo 估计 M≥20）对"找 top-k deviate points"足够稳定。不需要新增代码或存储逻辑，`--collect` 基础设施覆盖此需求。

- **三配置切换编排**：**沿用 `exp/run_cache_experiments.py` 的单 server + `load_cache_config` 热切换流程**（`_send_cache_config` at L79-99）。`build_shared_storage` 每次调用会 `_build_backend` + `load_artifact`，`build_per_connection_components` 会重造 `key_builder`（含 `ClipKeyBuilder`，后者 VRAM 按 lazy load 方式在首次 `collect()` 时分配 3 GB，切换后随引用被 GC，无需显式 `empty_cache`）。Phase 1/Phase 2 trajectory 实验（60 runs，含 CLIP ↔ non-CLIP 切换）已实际验证这条路径稳定（仅 1 例 segfault，重跑即恢复）。Step 3 的 3 个 deviate config YAML 放同一目录交给 runner 串行处理即可，不需起多 server / 不需重启。

**待确认**（实现阶段再讨论，但先钉在这里防止遗忘）：

1. **Spawn 策略 A vs B vs C**：§6.3 说先做策略 A（每个 intervention point 独立 spawn ×k 条轨迹）。但这与"一个 episode 里真实发生多次 intervention"的部署场景不一致。待确认：
   - A 是否足以验证假设（"单点偏差 → 级联失败"）？
   - 是否 Phase 3 一开始就必须上策略 C（连续 spawn 链）以贴近真实场景？
   - B（从最后一个 deviate point spawn）是否仅作为 k>1 的退化 baseline？

**已结论**：
- **Step 3 单独写一个 runner（`exp/run_spawn_experiment.py`），不复用 / 不魔改 `exp/run_cache_experiments.py`**。两者并列，各自管自己的 unit 粒度与 state schema。可复用子逻辑（`_send_cache_config` 的 WS 控制消息、`RunState` JSON 持久化骨架、`_retry_failed_runs` 重试机制、进度条 + log append 壳子）抽到共用小模块 `exp/_cache_config_rpc.py` 与 `exp/_run_state_base.py`，所有 runner `import`。详见 §13。

- **Step 1b init 子集传递：选项 A（dump 子集 init 文件）**。小脚本 `scripts/dump_step1a_failed_inits.py`：读 Step 1a 的 `episode_results.json` → per-task 筛失败 `(task_id, init_idx)` → `torch.save` 成 `{task.name}.init` 到 `data/step1b_inits/libero_spatial/` → `main.py --init-states-dir data/step1b_inits/libero_spatial --collect ...` 直接跑。`main.py` 零改动。失败 init 文件可复用于后续实验（对比新 gate、新 key_builder、新 D 的增量实验），不必每次重 dump。

- **Step 2 双阶段并行 replay（Q2 选 ①，单脚本）**：
  - **新增 `AlwaysSkipGate`**（`src/openpi/cache/components/gate.py`，约 10 行）+ config `gate.type: always_skip`。Orchestrator 对 gate skip 已正确处理（`orchestrator.py:219-226`）：`record_query_keys` 保持 strategy history 不断层、返回 MISS 让 Interceptor 走 inference、`broadcast_action` 把真 inference action 喂回各组件 hook。**零污染**，这正是背景 L2 需要的「always inference，trajectory history 正常累积」。
  - **关键并行模式**：因为框架会在单连接内累积 trajectory history（D-1 步），**不能在一个连接里对同一个 obs 点 M 次 infer**（第 2 次起 history 已污染）。改为「**M 条独立连接各自沿 GT obs 序列 rollout 一遍**」—— 每条连接内部的 history 来自自己的 infer 流水，语义与真实部署一致；每个 step 就有 M 个独立样本（flow matching `sample_noise` 走 torch 全局 RNG，M 条连接天然 decorrelated）。
  - **两阶段编排**：阶段 1 切到 `inference_<config>.yaml`（`AlwaysSkipGate`）跑 `K × M` 个 worker unit（K = GT episode 数）；阶段 2 切回正常 cache YAML 跑 `K` 个 unit（cache 是 deterministic，1 次就够）；阶段 3 离线聚合算 deviate score。
  - **并行性分析**：server `--concurrent` 模式下，每连接独立 per-connection components（独立 key_builder / strategy / gate，互不干扰）；单卡瓶颈是 GPU 显存（PaliGemma ~14 GB + per-connection buffer），N 受硬件限制，经验上 N=4-8。参考 `run_cache_experiments.py` 的 subprocess 并发模式（外层 N workers × 内层 main.py 各自起 env）。
  - **脚本不用 libero env**：Step 2 做的是「静态 obs 重放」，不调 `env.step`；`compute_deviate_scores.py` 自己读 HDF5 obs 序列，经 WebSocket 发给 policy，收 action，写结果。不走 `main.py`。

- **所有 exp 脚本硬性约束：`--resume` + 失败重试**：
  - `run_cache_experiments.py`（Step 1a）已满足。
  - `run_step1b_gt.py`（Step 1b thin dispatcher）：以 `(task_id, init_idx)` 为 unit key 做 per-unit resume；失败重试走 `_retry_failed_runs` 模板（`exp/_run_state_base.py` 提供）。
  - `compute_deviate_scores.py`（Step 2）：阶段 1 unit key = `(config, episode, sample_idx)`，阶段 2 unit key = `(config, episode)`；per-unit 粒度保存到 state JSON。
  - `run_spawn_experiment.py`（Step 3）：unit key = `(config, gt_episode, point_s, n, spawn_idx)`（§13.3）。
  - 所有 runner 复用 `exp/_run_state_base.py` 的 `BaseRunState` + `_retry_failed_runs` + `tqdm` 进度条 + log append 壳子。

---

## 12. 成功标准

实验成功的判定标准：

| 指标 | 目标 |
|------|------|
| Deviate points 占比 | < 20% of total steps |
| Top-3 spawn success rate vs pure cache | 提升 > 15 个百分点 |
| Top-3 spawn success rate vs random-3 | 显著高于 random baseline |
| 最小有效 n | n ≤ 10（rollout 长度合理） |

---

## 13. Runner 架构总览

### 13.0 每个 Step 用哪个 runner

| Step | Runner | 入口行为 | Unit 粒度 | 备注 |
|------|--------|---------|---------|------|
| 1a 全 benchmark 扫失败 | `exp/run_cache_experiments.py`（既有） | YAML(always_hit) → `main.py` 全 task × 5 init | `(yaml, task_id)` | 需加 `--save-episode-results` per-episode JSON 输出（§9.3） |
| 1b 对失败 init 跑 GT | `exp/run_step1b_gt.py`（新建，thin dispatcher） | dump 出的 per-task init 文件 → `main.py --init-states-dir ... --collect` | `(task_id, init_idx)` | `main.py` 零改动；runner 自己做 resume + retry |
| 2 背景 L2 + cache L2 | `exp/compute_deviate_scores.py`（新建） | 静态 obs 重放，不过 `main.py` / 不用 env | 阶段 1: `(config, episode, sample_idx)`；阶段 2: `(config, episode)` | 两阶段切 YAML（always_skip → 正常 cache） |
| 3 spawn 实验 | `exp/run_spawn_experiment.py`（新建） | 读 HDF5 → teleport env → prefill → 纯 cache rollout | `(config, gt_episode, point_s, n, spawn_idx)` | 需要 server 新增 `prefill_begin / prefill_end` 控制消息 |

### 13.1 为什么不复用 `exp/run_cache_experiments.py`

现有 runner 围绕「YAML → libero benchmark 全 task 跑 N episodes → 聚合 success rate」闭环设计，与 Step 3 需求有 6 处结构性错位（不是加 flag 能补的）：

| # | 错位点 | 现有 runner | Step 3 需求 |
|---|--------|------------|-------------|
| 1 | 入口脚本 | 调 `examples/libero/main.py`（不知 HDF5 / prefill / set_sim_state） | 需要读 HDF5 GT、teleport env、发 prefill 控制消息 |
| 2 | 参数空间 | `(yaml, task_id, trial_idx)` 3 维 | `(yaml, gt_episode, deviate_point_s, n, spawn_strategy, k, spawn_idx)` 7 维 |
| 3 | 结果指标 | 正则抓 stdout `"Total success rate: S/T"` | 结构化 `(point_s, n, D) → {success, final_step, action_l2, ...}` 矩阵，要 JSON/CSV |
| 4 | 状态粒度 | `task_progress[task_id]` | `spawn_progress[(gt_episode, point_s, n, spawn_idx)]` |
| 5 | 并发边界 | 丢给 `main.py --num-workers` 管 libero env pool | runner 自管 worker pool，单位 = 单次 spawn |
| 6 | 控制协议 | 只发 `episode_start/end + load_cache_config` | 新增 `prefill_begin(history) / prefill_end`（或经 obs 侧通道） |

### 13.2 目录与模块划分

```
exp/
  _cache_config_rpc.py        ← 新建：WebSocket 控制消息统一封装
                                 - send_load_cache_config(url, yaml_path)
                                 - send_prefill_begin(url, history_steps)  [new]
                                 - send_prefill_end(url)                   [new]
  _run_state_base.py          ← 新建：RunState 基类 + JSON 持久化骨架
                                 - load_state / save_state
                                 - --resume pattern
                                 - _retry_failed_runs 模板（max_retries=2，
                                   失败 unit copy 到 retry/ 目录重跑）
  run_cache_experiments.py    ← 保留：Step 1a（跑 always_hit 扫 libero benchmark）
                                 内部切换到 import _cache_config_rpc + _run_state_base
  run_step1b_gt.py            ← 新建：Step 1b thin dispatcher
                                 调 main.py --init-states-dir + --collect
  compute_deviate_scores.py   ← 新建：Step 2（详见 §13.4）
  run_spawn_experiment.py     ← 新建：Step 3（详见 §13.3）
scripts/
  dump_step1a_failed_inits.py ← 新建：Step 1a → Step 1b 的 init 子集 dump
```

### 13.3 `run_spawn_experiment.py` 职责

**输入**：
- 3 份 deviate config YAML（`configs/cache_runs/deviate_exp/`）
- Step 1b 产出的 GT HDF5 目录（per `(task_id, init_state_idx)`）
- Step 2 产出的 `deviate_score.json`（per episode 每步 score）
- 参数网格：`k ∈ {1, 3, 5}`, `n ∈ {3, 5, 10}`, `spawn_strategy ∈ {A}`, `spawn_idx ∈ [0, K)`

**Unit**：`(config, gt_episode_id, deviate_point_s, n, spawn_idx)`，每个 unit 独立 resume。

**核心循环**（伪码）：
```python
for config in configs:
    _send_load_cache_config(server_url, config.yaml)
    for episode in gt_episodes:
        top_k_points = read_deviate_score(episode, k)
        for s in top_k_points:
            for n in n_grid:
                for spawn_idx in range(K):
                    if state.is_done(unit_key): continue
                    history = read_hdf5_window(episode, s+n-D+1, s+n)  # D-1 步
                    obs = reconstruct_obs_from_hdf5(episode, s+n)
                    sim_state = read_sim_state(episode, s+n)

                    _send_prefill_begin(server_url, history)
                    env.reset(); env.set_sim_state(sim_state)
                    result = run_pure_cache_episode(env, conn_policy, obs, max_steps)
                    _send_prefill_end(server_url)

                    state.record(unit_key, result)
                    state.save()  # per-unit 粒度 resume
```

**输出**：
- `spawn_results.jsonl`：每行一个 unit，字段 `{config, gt_episode, point_s, n, spawn_idx, success, final_step, action_l2_to_gt, ...}`
- `spawn_aggregate.csv`：聚合矩阵 `(config, n, D) → success_rate`，供 §12 指标判定

### 13.4 `compute_deviate_scores.py`（Step 2）职责与并行性分析

**输入**：
- 3 份 deviate config，每份 2 个 YAML：
  - `inference_<config>.yaml`（`gate.type: always_skip`，强制 inference，trajectory history 仍累积）
  - `cache_<config>.yaml`（正常 gate + judge，用于 cache L2 replay）
- Step 1b 产出的 GT HDF5（每 episode 含 T 步 obs/robot_state/images/clean_action/sim_state）
- 参数：`M`（背景采样数，默认 M=20），`N`（worker 并发数，默认 N=4）

**两阶段流程**（per config）：

**阶段 1：背景 L2**
```python
_send_load_cache_config(server_url, f"inference_{config}.yaml")  # AlwaysSkipGate

tasks = [(episode, m) for episode in gt_episodes for m in range(M)]
# 单位工作：1 条 worker 沿一条 GT episode 跑 1 遍，T 步
# server 端 --concurrent 模式下 N 个连接并行

def worker_run(episode, sample_idx):
    conn = open_ws_connection(server_url)
    conn.send(episode_start)
    actions = []
    for t in range(T):
        obs = read_hdf5_obs(episode, t)   # GT obs from HDF5, no env.step
        action = conn.infer(obs)          # framework 框架强制走 inference
                                          # trajectory history 由 conn 自己累积
        actions.append(action)
    conn.send(episode_end)
    save_to_jsonl(config, episode, sample_idx, actions)

run_parallel(tasks, worker_run, num_workers=N)
```

**阶段 2：cache L2**
```python
_send_load_cache_config(server_url, f"cache_{config}.yaml")  # 正常 cache

tasks = [(episode,) for episode in gt_episodes]  # 每 episode 1 次
run_parallel(tasks, worker_run, num_workers=N)   # 同一 worker_run，换 YAML
```

**阶段 3：聚合**（离线，不用 server）
```python
for episode, t in grid:
    bg_actions = [sample[t] for sample in bg_samples[config][episode]]  # M 个
    bg_l2 = mean_pairwise_l2(bg_actions)                                # 背景
    cache_action = cache_samples[config][episode][t]                    # 1 个
    gt_action = read_gt_action(episode, t)
    cache_l2 = l2(cache_action, gt_action)
    deviate_score = cache_l2 / (bg_l2 + eps)
    record(config, episode, t, deviate_score)
```

**并行性分析**：

| 层级 | 并行度 | 上限 | 说明 |
|------|--------|------|------|
| config | 串行 | 1（3 个 config 顺序） | `load_cache_config` 切换，单 server 实例 |
| 阶段（背景 / cache） | 串行 | 1（阶段 1 → 阶段 2） | 中间切 YAML |
| Worker 连接（阶段 1） | 并行 | N（受 GPU 显存限制） | server `--concurrent` 模式；每连接独立 key_builder / strategy / gate |
| Worker 连接（阶段 2） | 并行 | N | 同上 |
| Episode 内步数 T | 串行 | 1（per 连接） | 单连接内必须顺序 infer（trajectory history 依赖） |

**关键限制**：
- **M 次采样不能在一个连接内反复 infer 同一个点**：trajectory history 会累积，第 2 次 infer 时 strategy 的 query_history 和 key_builder vision buffer 已被上一次污染。
- **必须用 M 条独立连接各自沿 GT rollout 一遍**：连接自己的 history 由自己的 infer 流积累（来自 GT obs 序列），语义与部署一致，每步天然 1 个采样 × M 条连接 = M 个。
- 每个 worker 任务完成一整条 episode rollout（T 步），而不是「单步」粒度；过细并发反而浪费握手开销。

**Unit 粒度**：
- 阶段 1 unit key = `(config, episode_id, sample_idx)`；unit 数 ≈ 3 × K × M（K = GT episode 数）
- 阶段 2 unit key = `(config, episode_id)`；unit 数 ≈ 3 × K
- 合计 ≈ 3 × K × (M + 1) 个 unit，按 config × 阶段 为边界分组并发。

**输出**：
- `bg_actions_<config>.jsonl`：每行 `{episode, sample_idx, t, action}`
- `cache_actions_<config>.jsonl`：每行 `{episode, t, action}`
- `deviate_score.json`：`{config: {episode: [score_0, score_1, ..., score_T-1]}}`

### 13.4b `run_spawn_experiment.py`（Step 3）并发策略

- **外层串行**：3 份 config 按顺序过（`load_cache_config` 切换）。
- **内层并发**：单 config 内，多个 worker 进程各自管一个 libero env，从 unit 队列取 unit 执行。worker 数 = `--num-workers`，默认 5。
- **单 unit 粒度 resume**：state 文件以 `unit_key` 为 key；worker 开工前先 check 状态，已 done 直接跳过。

### 13.5 与 server 的接口变更

`websocket_policy_server.py` 新增两条控制消息（需配合 §9.5 的 `CacheStorage.enter_prefill_mode` / `exit_prefill_mode`）：

```python
elif ctrl == "prefill_begin":
    history = obs.get("history_steps", [])  # list of {obs, action} dicts
    bundle.shared_storage.enter_prefill_mode(_build_payload_from_history(history))
    # 随后 N-1 次 infer 会走 prefill 路径灌历史
    await websocket.send(packer.pack({"__ack__": "prefill_begin"}))

elif ctrl == "prefill_end":
    bundle.shared_storage.exit_prefill_mode()
    await websocket.send(packer.pack({"__ack__": "prefill_end"}))
```

server 改动规模：约 20 行，纯控制路径，不侵入 infer 主干。

### 13.6 实现顺序建议

1. 先写 `_cache_config_rpc.py` + `_run_state_base.py`，让 `run_cache_experiments.py` 先迁移过去（保持行为等价），验证抽象没漏。
2. 补 `AlwaysSkipGate` + config 注册 + `configs/cache_runs/deviate_exp/inference_*.yaml`。
3. 写 `scripts/dump_step1a_failed_inits.py`（依赖 Step 1a 产出的 `episode_results.json`，跑完 Step 1a 再动工）。
4. 写 `run_step1b_gt.py`（thin dispatcher，per-(task_id, init_idx) 粒度 resume）。
5. 写 `compute_deviate_scores.py`（先单 config 单 episode smoke test → 扩全网格）。
6. 再写 `run_spawn_experiment.py` 的骨架（不含 prefill，先跑 teleport + pure cache 路径）。
7. 同步实现 server 的 `prefill_begin / prefill_end` 控制消息 + §9.5 的 `CacheStorage` prefill 模式。
8. 端到端连起来跑 1 个 spawn unit smoke test。
9. 再扩到完整参数网格。

### 13.7 所有 Runner 的硬约束：`--resume` + 失败重试

**硬约束**：以下所有 runner **必须**满足这两个能力，不接受「先不做，后面补」：

| Runner | `--resume` unit key | Retry 策略 |
|--------|--------------------|-----------|
| `run_cache_experiments.py`（Step 1a） | `(yaml, task_id)` ✓ 已有 | `_retry_failed_runs` ✓ 已有（max 2 次） |
| `run_step1b_gt.py`（Step 1b） | `(task_id, init_idx)` | 同上模板，max 2 次 |
| `compute_deviate_scores.py`（Step 2） | 阶段 1: `(config, episode_id, sample_idx)`；阶段 2: `(config, episode_id)` | 同上模板，max 2 次；区分两阶段独立 state 文件 |
| `run_spawn_experiment.py`（Step 3） | `(config, gt_episode, point_s, n, spawn_idx)` | 同上模板，max 2 次 |

**实现方式**：
- 所有 runner 强制 `import exp._run_state_base`，继承 `BaseRunState`，实现 `unit_key()` 与 `is_done()` 两个方法。
- `--resume` 默认 opt-in（命令行 flag）；不带 `--resume` 从零开始。
- 失败重试：首轮全跑完后，把 `status=failed` 的 unit 的 YAML/state entry copy 到 `<output_dir>/retry/`，按 `max_retries` 重跑；每次成功 → 从 retry/ 移除 + 更新原 state；max_retries 跑完仍失败 → 保留在 retry/ 供手工检查（参考 `run_cache_experiments.py:583-736` 既有实现）。
- State 文件 schema 随 unit 粒度变化，但 `status`、`start_time`、`end_time`、`retry_count` 字段统一。
- 所有 state 文件位于输出目录同级：`{output_dir}/run_state.json` + `{output_dir}/retry/*.yaml`。

**验收条件（写入 §12 旁注）**：每个 runner 实现时必须跑通以下自检：
1. Ctrl-C 中断任意 unit → 重启带 `--resume` → 未完成的 unit 继续，已完成的 unit 跳过。
2. 构造 1 个必然失败的 unit（如指向不存在的 HDF5）→ 确认进入 retry/ 目录并被重跑 2 次后留痕。
