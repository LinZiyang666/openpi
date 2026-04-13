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

### 4.1 Step 1a: 跑纯 Cache，找到失败 Episode

先跑纯 cache 评估，目的是找出哪些 (task, init_state) 组合下 cache 会失败。

**配置**：
- 服务器：`serve_policy.py --concurrent`，加载当前最优 cache 配置（`gate: always_search`, `judge: always_hit`）
- 客户端：`main.py`，正常评估流程
- 全量跑：10 tasks × 5 episodes = 50 episodes

**记录**：每个 episode 的 `(task_id, init_state_idx, seed, success)` 保存到 JSON。

```json
// data/deviation_experiment/cache_eval_results.json
[
  {"task_id": 0, "init_state_idx": 0, "seed": 42, "success": true},
  {"task_id": 0, "init_state_idx": 1, "seed": 42, "success": false},
  ...
]
```

**注意**：需要确保 `init_state_idx` 能被记录。当前 `main.py` 按 task 内 episode index 分配 init_state（`init_states[episode_idx]`），所以 `init_state_idx = episode_idx`。

### 4.2 Step 1b: 只对 Cache 失败的 Init 跑纯 Inference 收集 GT

从 Step 1a 筛选出 `success == false` 的 episode，只对这些 `(task_id, init_state_idx)` 组合跑纯 inference，收集完整 GT trajectory。

**优势**：
- 节省 GPU 时间 — 不需要全量跑 inference，只跑 cache 失败的那些
- 保证实验对象有意义 — 每条 GT trajectory 都对应一个 cache 失败的 episode

**可能的情况**：inference 在同一个 init_state 下也可能失败。此时该 episode 不满足实验前提（"inference 成功 AND cache 失败"），直接跳过。

### 4.3 GT Trajectory 需要收集的数据

每条成功 inference 轨迹需要保存 **每一步** 的以下数据：

| 数据 | 来源 | 用途 |
|------|------|------|
| `env_sim_state` | `env.sim.get_state()` (MuJoCo 完整状态: qpos, qvel, 全部物体) | Step 3 spawn 恢复环境 |
| `observation` | `env.step()` 返回的 obs dict | Step 2 重放给 inference/cache |
| `action` | 模型输出的 action chunk | GT action，计算 L2 距离的参考 |
| `model_intermediates` | data_collector 已有的 embeddings (vision, prompt, state, noise_actions) | 离线回放 cache key builder |
| `images_raw` | obs 中的 agentview_image + eye_in_hand_image | 重放给模型做 inference |
| `robot_state` | obs 中的 eef_pos + eef_quat + gripper_qpos → 8D | 重放给模型做 inference |

**关键：`env_sim_state` 是新增需求**。现有 data_collector 只保存 model-side 数据（embeddings, actions），不保存 MuJoCo simulator 的完整物理状态。spawn 需要的是完整 sim state（包含所有物体的 qpos/qvel，不只是 robot）。

### 4.4 收集方式

修改 `examples/libero/main.py` 增加 `--save-trajectory` flag，在每步 `env.step(action)` 前保存完整状态：

```python
# 伪代码
for step in range(max_steps):
    # 保存当前 env state（在执行 action 前）
    sim_state = env.sim.get_state()
    obs_snapshot = copy(obs)

    # 正常推理
    action = policy.infer(obs)

    # 保存
    trajectory_data[step] = {
        "sim_state": sim_state,          # MuJoCo full state
        "observation": obs_snapshot,      # obs dict
        "action": action,                 # model output
        "images": {
            "agentview": obs["agentview_image"],
            "eye_in_hand": obs["robot0_eye_in_hand_image"],
        },
        "robot_state": observation_state,  # 8D
    }

    obs, reward, done, info = env.step(action)
```

GT 收集只跑 Step 1a 中筛选出的 episode，通过 `--task-ids` 和指定 init_state_idx 控制。

### 4.5 env state save/restore 最小验证

**在正式收集前，必须验证 MuJoCo state 的 save/restore 能力：**

```python
# 最小验证脚本
env = make_libero_env(task)
obs = env.reset()
env.set_init_state(init_state)

# 跑 50 步
for i in range(50):
    action = policy.infer(obs)
    obs, reward, done, info = env.step(action)

# 保存 state
saved_state = env.sim.get_state()
saved_obs = copy(obs)

# 继续跑 50 步，记录后续轨迹
trajectory_A = []
for i in range(50):
    action = policy.infer(obs)
    obs, reward, done, info = env.step(action)
    trajectory_A.append(obs["robot0_eef_pos"])

# 恢复 state
env.sim.set_state(saved_state)
env.sim.forward()  # 重新计算物理量

# 用相同 seed 跑 50 步，验证轨迹一致
trajectory_B = []
for i in range(50):
    action = policy.infer(obs)  # 注意：如果 policy 有随机性，需要固定 seed
    obs, reward, done, info = env.step(action)
    trajectory_B.append(obs["robot0_eef_pos"])

# 验证
assert np.allclose(trajectory_A, trajectory_B, atol=1e-6)
```

**注意**：即使 env state 完全恢复，policy inference 有随机性（flow matching 的 noise sampling）。验证时需要固定 torch seed 以排除 policy 随机性。

### 4.6 存储格式

每条 GT trajectory 保存为一个 HDF5 文件：

```
data/deviation_experiment/gt_trajectories/<task_id>/episode_<init_idx>.h5
├── metadata/
│   ├── task_name: str
│   ├── task_id: int
│   ├── init_state_idx: int
│   ├── seed: int
│   ├── num_steps: int
│   ├── success: bool          # inference 是否成功（应该全是 true）
│   └── cache_success: bool    # 对应 cache 是否成功（应该全是 false）
├── step_0000/
│   ├── sim_state: bytes       # pickle(env.sim.get_state())
│   ├── agentview_image: [H, W, 3] uint8
│   ├── eye_in_hand_image: [H, W, 3] uint8
│   ├── robot_state: [8] float32
│   ├── action: [action_horizon, action_dim] float32
│   ├── prefix_embs: [N_tokens, D] float32       # vision encoder output
│   ├── prompt_emb: [N_prompt, D] float32
│   └── noise_action_1..9: [action_horizon, action_dim] float32
├── step_0001/
│   └── ...
└── ...
```

---

## 5. Step 2: 离线计算 Deviate Score

### 5.1 背景 L2 距离计算

对 GT trajectory 的每一步 t：

1. **恢复环境到该步的 state**：`env.sim.set_state(step_t.sim_state); env.sim.forward()`
2. **重放 observation**：用 `step_t.images` + `step_t.robot_state` 构造 policy input
3. **跑 M 次 inference**：每次用不同的 torch seed（控制 flow matching noise），得到 M 个 action
4. **计算背景 L2**：

```python
# M 个 action，每个 shape [action_horizon, action_dim]
actions = [inference(obs, seed=s) for s in seeds[:M]]

# 两两 L2 距离的均值
pairwise_l2 = []
for i in range(M):
    for j in range(i + 1, M):
        pairwise_l2.append(torch.norm(actions[i] - actions[j]).item())
background_l2 = np.mean(pairwise_l2)
```

**参数**：M = 10（10 次 inference，共 C(10,2) = 45 对，统计量足够）。

**注意**：这一步需要调用远端 GPU 服务器做 inference。每步 10 次 × 每条轨迹 ~220 步 = ~2200 次 inference per episode。如果单次 inference 约 0.3s，单条轨迹约 11 分钟。可并行化。

### 5.2 Cache L2 距离计算

对 GT trajectory 的每一步 t：

1. **重放 observation**：同上
2. **跑 1 次 cache 检索**：将 observation 喂给 cache 系统（key builder → search → judge → fetch payload → 返回 cached action）
3. **计算 Cache L2**：

```python
cache_action = cache_lookup(obs_t)  # shape [action_horizon, action_dim]
gt_action = step_t.action
cache_l2 = torch.norm(cache_action - gt_action).item()
```

**Cache 检索模式**：每步独立查询（`trajectory_depth=1`），不积累 trajectory history。原因：
- 此阶段目的是计算 deviate score，需要每步独立评估
- Trajectory history 的影响在 Step 3 的 spawn 实验中单独控制

### 5.3 Deviate Score

```python
deviate_score_t = cache_l2_t / max(background_l2_t, epsilon)
# epsilon = 1e-6 防止除零（某些 step 模型方差极小）
```

### 5.4 背景 L2 为零或极小的处理

某些 step 模型输出几乎确定性（如 gripper 保持不动的 step），背景 L2 ≈ 0。此时 deviate score 会爆炸但无意义。

处理：设定 `background_l2_floor`（如 0.1），低于此值的 step 标记为 **deterministic step**，deviate score 改为 `cache_l2 / background_l2_floor`，同时在分析中单独统计。

### 5.5 产出

每条 GT trajectory 生成一个 analysis JSON：

```json
{
  "task_id": 0,
  "episode_id": 3,
  "num_steps": 187,
  "cache_success": false,
  "steps": [
    {
      "step": 0,
      "background_l2": 0.523,
      "cache_l2": 0.481,
      "deviate_score": 0.920,
      "deterministic": false
    },
    {
      "step": 1,
      "background_l2": 0.034,
      "cache_l2": 2.871,
      "deviate_score": 84.44,
      "deterministic": true
    }
  ]
}
```

### 5.6 Phase A 诊断分析

在所有 trajectory 的 deviate score 数据上：

1. **分布直方图**：deviate score 的分布（预期：大量 ≈ 1，少数长尾 >> 1）
2. **Top-k 覆盖率**：选 top-k deviate points 能覆盖总偏差的百分比（类似 PCA 方差解释率）
3. **位置分布**：deviate points 在轨迹中的 normalized position (0~1)
4. **连续性**：相邻 deviate points 的间距分布
5. **跨轨迹一致性**：同一 task 的不同 episode 是否在相似位置出现 deviate？

**Go/No-Go 判定**：
- Go：deviate score >> 1 的 step 占比 < 20%，且存在明显长尾
- No-Go：deviate score 分布平坦，无明显分离 → 假设不成立，需重新审视

---

## 6. Step 3: Oracle 纠偏 — Spawn 实验

### 6.1 实验参数

| 参数 | 符号 | 扫描范围 | 说明 |
|------|------|---------|------|
| 纠偏点数量 | k | 1, 2, 3, 5 | 选 deviate score 最高的 k 个 step 做 intervention |
| Rollout 长度 | n | 1, 3, 5, 10, 20 | intervention 后沿 GT trajectory rollout 几步 |
| Trajectory depth | D | 1, 3, 5 | spawn 后 cache 的 trajectory search 历史窗口 |

### 6.2 单次 Spawn 实验流程

以一条 GT trajectory、一个 intervention point `s`、rollout 长度 `n`、trajectory depth `D` 为例：

```
GT trajectory: [step_0, step_1, ..., step_s, step_s+1, ..., step_s+n, ..., step_T]
                                      ↑ intervention point

实验执行:
1. 恢复 env 到 step_(s+n) 的 sim_state（即 rollout 后的 env state）
2. 初始化 cache orchestrator（on_episode_start）
3. 预填 trajectory history（如果 D > 1）
4. 从 step_(s+n) 开始跑纯 cache 直到 episode 结束或 max_steps
5. 记录最终是否 success
```

### 6.3 Trajectory History 预填充 — 核心细节

当 `D > 1` 时，cache 的 trajectory search 需要历史 step 的 query keys 和 action 来做多级检索。spawn 后 cache 从零开始，前 `D-1` 步没有足够历史。

**预填充策略**：用 GT trajectory 的历史数据来 prime cache 的 trajectory buffer。

#### 预填充窗口

spawn 点为 `step_(s+n)`，trajectory depth 为 `D`。需要从 GT trajectory 取 `D-1` 步历史：

```
GT history window: [step_(s+n-D+1), step_(s+n-D+2), ..., step_(s+n-1)]
                    ← D-1 步 →
```

#### 预填充操作

对 history window 中的每一步，按顺序：

1. 用 GT observation 构造 key builder input → 调用 `key_builder.collect()` + `key_builder.build()` 得到 `query_keys`
2. 调用 `search_strategy.record_query_keys(query_keys)` 写入 trajectory history buffer
3. 调用 `search_strategy.record_action(gt_action)` 写入 action history buffer

这样 spawn 后第一步 cache search 就已有 `D-1` 条历史。

#### 历史衰退过程

spawn 后的 cache rollout 中，trajectory history 的组成会逐步变化：

```
假设 D=5, spawn 点为 step_50:

cache step 1 (= GT step 50):
  history = [GT_46, GT_47, GT_48, GT_49]  ← 全部来自 GT
  当前 = cache 自己产出的 query

cache step 2 (= GT step 51):
  history = [GT_47, GT_48, GT_49, cache_50]  ← 3 GT + 1 cache

cache step 3:
  history = [GT_48, GT_49, cache_50, cache_51]  ← 2 GT + 2 cache

cache step 4:
  history = [GT_49, cache_50, cache_51, cache_52]  ← 1 GT + 3 cache

cache step 5+:
  history = [cache_50, cache_51, cache_52, cache_53]  ← 全部 cache
```

**从 step D 开始，history 完全由 cache 自己的数据组成。** 前 D-1 步是 GT→cache 的过渡期。

#### D=1 特殊情况

当 `D=1` 时，不使用 trajectory history（单步检索）。无需预填充，spawn 后直接跑。这是最干净的 baseline。

### 6.4 多 Intervention Points 的处理

当 k > 1 时，有多个 intervention points。两种策略：

**策略 A: 独立 Spawn（推荐）**

每个 intervention point 独立做 spawn 实验。即：对 top-k 个 deviate points，分别做 k 次独立的 spawn 实验，每次只处理一个 point。

优点：
- 实验简单，结果可解释
- 可以分析每个 intervention point 的独立贡献

缺点：
- 不反映多个 intervention 的协同效应

**策略 B: 最后一个 Spawn**

k 个 intervention points 按时序排列为 `s1 < s2 < ... < sk`。沿 GT trajectory 执行到 `sk + n` 后 spawn。

含义：假设前 k-1 个 deviate points 都被纠偏了（因为走的是 GT），只测试 "所有纠偏完成后 cache 能否接管" 。

**策略 C: 连续 Spawn 链（最完整但最复杂）**

```
step_0 → ... → s1: 跑纯 cache
s1 → s1+n: 切换为 GT rollout (intervention 1)
s1+n → ... → s2: 跑纯 cache
s2 → s2+n: 切换为 GT rollout (intervention 2)
...
sk+n → ... → T: 跑纯 cache
```

这最接近真实场景，但需要多次 spawn + cache restart，实现复杂。

**推荐**：先做策略 A（独立 spawn），验证单个 intervention 的效果。如果效果好，再做策略 B/C。

### 6.5 Baseline 对照组

| 对照组 | 说明 |
|--------|------|
| **Pure cache** | 不做任何 intervention，纯 cache 的 success rate |
| **Pure inference** | 不用 cache，纯 inference 的 success rate |
| **Random-k** | 随机选 k 个 step 做 intervention（而非 top-k deviate），验证 deviate score 的信息量 |
| **Equidistant-k** | 等间距选 k 个 step 做 intervention，验证是否"任何地方纠偏都有用" |

### 6.6 连续 Deviate Points 合并

如果 top-k 中有连续的 step（如 step 45, 46, 47），应合并为一个 intervention window：

```python
# 合并规则：相邻 step 间距 <= merge_gap 时合并
merge_gap = 3  # 可配置

# 例: top-5 deviate points = [45, 46, 47, 102, 150]
# 合并后: intervention_windows = [(45, 47), (102, 102), (150, 150)]
# 实际 intervention 次数 = 3（而非 5）
```

合并后，每个 window 的 rollout 从 window 末尾开始（即 rollout 起点 = window.end + 1）。

---

## 7. 实验执行计划

### 7.1 固定配置

| 配置 | 值 |
|------|---|
| Task Suite | libero_spatial (10 tasks) |
| Episodes per task | 5 |
| Max steps per episode | 220 |
| Seed | 42 |
| Model | Pi0.5 LIBERO checkpoint |
| Cache Artifact | `data/cache_artifacts/libero_spatial/` (使用当前最优 reducer) |
| M (inference 重复次数) | 10 |

### 7.2 执行顺序

```
Phase 0: 环境验证
  └── P0-1: MuJoCo state save/restore 最小验证
  └── P0-2: 验证 env.sim.get_state() 包含完整物体信息

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

### 7.3 预计工作量

| Phase | 人工 | GPU 时间 | 说明 |
|-------|------|---------|------|
| P0 | 0.5d | < 10min | 最小验证脚本 |
| P1-1 | 0.5d | ~30min | 纯 cache 评估 50 episodes，复用现有实验框架 |
| P1-2~4 | 1d | ~30min | 修改 main.py + 定向跑 inference（只跑失败 episode，数量远少于 50） |
| P2 | 1d | ~数小时 | 每条轨迹 ~2200 次 inference |
| P3 | 1d | ~数小时 | spawn 实验数量 = episodes × k_values × n_values × D_values |

---

## 8. 需要新建/修改的代码

| 文件 | 类型 | 说明 |
|------|------|------|
| `examples/libero/main.py` | 修改 | 增加 `--save-trajectory` flag，保存 env sim_state |
| `exp/collect_gt_trajectories.py` | 新建 | 封装 GT trajectory 收集 + cache 失败筛选 |
| `exp/compute_deviate_scores.py` | 新建 | 离线计算背景 L2 + cache L2 + deviate score |
| `exp/run_spawn_experiment.py` | 新建 | spawn 实验主脚本，参数扫描 k × n × D |
| `exp/analyze_deviation_results.py` | 新建 | 统计分析 + 可视化 |

---

## 9. 风险

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| MuJoCo state restore 不完整（遗漏某些 internal flags） | 高 | Phase 0 最小验证先行 |
| LIBERO env 有非 MuJoCo 的内部状态（task stage counter 等） | 高 | 验证 restore 后 `env.step()` 行为一致 |
| 背景 L2 计算耗时过长（每步 M 次 inference） | 中 | M=10 已是平衡值；可先用 M=5 做初步分析 |
| 假设不成立：偏差均匀分布 | 高 | Phase 2 的 Go/No-Go 判定，不盲目推进 Phase 3 |
| GT trajectory 的 inference 随机性导致不同 seed 跑出不同轨迹 | 低 | GT 收集时固定 seed，后续 M 次 inference 用不同 seed |
| Spawn 后的 cache trajectory history 过渡期影响结果 | 中 | D=1 作为 baseline 消除此因素，D>1 结果与 D=1 对比 |

---

## 10. 待确认

1. LIBERO `OffScreenRenderEnv` 的底层 sim 对象路径：`env.sim` 还是 `env.env.sim`？需要查 robosuite 封装层级
2. `env.sim.get_state()` 返回的对象是否可以直接 pickle 进 HDF5？
3. 纯 cache 跑的最优配置用哪个 reducer + 权重？（从现有 Phase 1 实验结果中选）
4. 是否需要同时收集 cache 轨迹的 per-step action（用于 Phase A 直接对比两条轨迹）？当前方案是 step 2 中用 GT observation 重放 cache，不需要真实 cache 轨迹

---

## 11. 成功标准

实验成功的判定标准：

| 指标 | 目标 |
|------|------|
| Deviate points 占比 | < 20% of total steps |
| Top-3 spawn success rate vs pure cache | 提升 > 15 个百分点 |
| Top-3 spawn success rate vs random-3 | 显著高于 random baseline |
| 最小有效 n | n ≤ 10（rollout 长度合理） |
