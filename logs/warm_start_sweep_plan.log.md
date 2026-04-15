---
name: warm_start_sweep_plan
description: Warm Start 成功率扫描实验计划：3 keybuilder × 3 start_t，对照 always_skip / always_hit
type: project
---

# Warm Start 成功率扫描实验计划

> Status: G1 Approved — ready for Code
> Date: 2026-04-15
> Task: 验证 CP1 warm start（从中间 timestep 起始的部分 denoise）能否提高 cache 命中下的任务成功率
> Level: L2（新 Judge 组件 + 新实验子目录）
> Owner: Ziyang Lin
> 依赖文档:
> - [`docs/architecture/cache_system.md`](../docs/architecture/cache_system.md) §6 / `warm_tiers` 描述
> - [`logs/archive/cp1_warm_start_impl_plan.log.md`](archive/cp1_warm_start_impl_plan.log.md)（warm start 实现总览）
> - [`docs/experiments/trajectory_deviation.md`](../docs/experiments/trajectory_deviation.md)（复用 server / 评估端 / artifact 来源）

---

## 1. 实验目标

回答一个问题：**在 cache 永远命中（`gate=always_search`）的前提下，把 FULL_HIT 替换为「从 timestep `start_t` 起的部分 denoise」（warm start）能否在牺牲少量延迟的情况下，把成功率从 always_hit baseline 拉回到接近纯 inference 的水平？**

**语义校准**（见 §2.1 表格 & `src/openpi/models_pytorch/pi0_pytorch.py` 的 `run_stage3_from`）：`start_t` 表示"从时间轴上哪个 t 开始跑剩余 denoise"。`start_t` **越小 → 剩余 denoise 步数越少 → 延迟越低 → 越接近 always_hit（完全采用缓存动作）**；`start_t` **越大 → 剩余 denoise 步数越多 → 延迟越高 → 越接近纯 inference**。极限情况：`start_t=0.0` 等价直接输出 cached action（FULL_HIT），`start_t=1.0` 等价从纯噪声重跑整个 Stage 3（MISS）。

具体形态：
- 若 warm start 能恢复成功率 → 这是一种"低成本纠偏"：缓存只提供初始 `x_t`，后续 denoise 用真实 stage2 condition 收敛。
- 预期曲线形状：`success_rate(start_t)` 从 `start_t=0.3`（接近 B1 下界）单调向 `start_t=0.7`（接近 B0 上界）上升 — 得到一条平滑的"成功率—延迟"权衡曲线，为后续在 `ThresholdJudge.warm_tiers` 里挑阈值提供经验依据。**（本实验不预设方向，以验证单调性为准；非单调则需排查 start_t 校验或 artifact intermediates 完整性。）**

---

## 2. 实验设计

### 2.1 自变量（3 × 3 网格）

| 维度 | 取值 |
|------|------|
| **Key builder**（来自 `configs/cache_runs/deviate_exp/`） | `cp1_max_pool` / `cp1_spatial_pool_16` / `clip_vit_b_32` |
| **start_t** | `0.7` / `0.5` / `0.3`（对应 noise_action_3 / 5 / 7，剩余 denoise 7 / 5 / 3 步） |

start_t 与剩余 denoise 步数对照（`num_steps=10`）：

| start_t | 对应 noise_action_i | 已 denoise | 剩余 denoise 步数 |
|---------|--------------------|-----------|------------------|
| 0.9 | noise_action_1 | 10% | 9 |
| 0.7 | noise_action_3 | 30% | 7 |
| 0.5 | noise_action_5 | 50% | 5 |
| 0.3 | noise_action_7 | 70% | 3 |
| 0.1 | noise_action_9 | 90% | 1 |

> 本次只跑 0.7 / 0.5 / 0.3 三档，覆盖中段；如有需要可扩到 0.9 / 0.1（pkl 已含全部 9 档，无需重建）。

### 2.2 对照组 — 复用 trajectory_deviation Step 1a 现成数据，不重跑

`trajectory_deviation` 的 Step 1a 已经在同一 `libero_spatial` 500-init 集上**对每个 keybuilder 都跑了纯 inference + always_hit 两套**，结果落在：

- `data/deviation_experiment/cache_eval_results_clip.json`
- `data/deviation_experiment/cache_eval_results_maxpool.json`
- `data/deviation_experiment/cache_eval_results_spatial16.json`

每条 record 包含 `(config_id, task_id, init_state_idx, seed, success)`，按 `config_id` split 后即得到上下界基线（汇总落盘 `data/warm_start_exp/baseline_failures.json`）：

| Config | **B0 Inference 上界** | **B1 Always-hit 下界** | Gap |
|--------|----------------------|------------------------|-----|
| `clip_w7_d4` | 99.2% (496/500, 4 fail) | 67.4% (337/500, **163 fail**) | +31.8 pp |
| `max_pool_w3_d5` | 98.4% (492/500, 8 fail) | 69.6% (348/500, **152 fail**) | +28.8 pp |
| `spatial16_w8_d4` | 98.4% (492/500, 8 fail) | 69.2% (346/500, **154 fail**) | +29.2 pp |

**Always-hit 失败集合的横截面**（500 init 中）：
- 三 cfg 失败并集：**277**
- 三 cfg 失败交集：**51**（"硬骨头"集合，warm start 在此集合的恢复率反映通用纠偏能力）
- 两两交集：`clip ∩ max=79`，`clip ∩ sp=76`，`max ∩ sp=88`

**Inference 失败集合**（纯模型 / env 噪声地板）：
- 三 cfg 各 4 / 8 / 8 fail，三者交集 **0**，并集 **15**。这 15 个 init 视为"模型本身就跑不通"的噪声地板，warm start 对它们的失败不归罪于 cache。

→ **本实验最终目标：9 个 warm start YAML × 500 ep = 4500 ep**（不重跑 B0 / B1）。

**执行范围**：
- **最终目标**：3 keybuilder × 3 start_t = **9 YAML × 500 ep = 4500 ep**
- **首批执行（P0b 未完成时）**：仅 `max_pool` + `spatial16` = **6 YAML × 500 ep = 3000 ep**；CLIP 的 3 个 YAML（`clip/` 子目录）等 P0b 完成后作为 P3 增量补跑 1500 ep。
- baseline 三 cfg 都已落 `baseline_failures.json`，后加 CLIP 不需补跑 baseline，P4 脚本需支持"3 cfg 完整"与"2 cfg 先行"两种输入形态（具体在 §3.5 P4 落实）。

### 2.3 主指标 / 副指标

| 类别 | 指标 | 计算方式 |
|------|------|----------|
| **主指标** | `success_rate` per (cfg, start_t) | `n_success / 500`（与 §2.2 baseline 同分母直接可比） |
| **主指标** | `recovery_rate` per (cfg, start_t) | `|warm_pass ∩ B1_fail(cfg)| / |B1_fail(cfg)|`，即 always-hit 下失败的那 152~163 个 init 中，warm start 救回了多少 |
| **主指标** | `incurred_loss` per (cfg, start_t) | `|warm_fail ∩ B0_pass(cfg)| / |B0_pass(cfg)|`，即原本 inference 能成功的 init 中，warm start 反而失败的比例（衡量"过激 warm 引入的新失败"） |
| 副指标 | `stage3_warm` / `stage3_flow` / `cp1_sum` 延迟均值 | timer probe；与 success_rate 配合给出权衡曲线 |
| 副指标 | WARM_START 被 Orchestrator 降级为 MISS 的次数 | 应为 0；不为 0 说明 P0 artifact intermediates 不完整 |
| 副指标 | warm start 失败 init 的三 cfg 交集大小 | 与 B1 交集（51）对照，看 warm start 是否解掉了"硬骨头" |

---

## 3. 阶段分解

```
P0  Artifact 重建（一次性）
    └── P1  新 Judge 组件 (L2 → G1/G2)
         └── P2  实验 YAML
              └── P3  跑实验
                   └── P4  分析 + 出图
                        └── P5  文档 + 索引
```

### 3.1 P0 — Artifact 重建（前置，max_pool + spatial16 先行）

**目的**：现有 `data/cache_artifacts/libero_spatial/{cp1_max_pool, cp1_spatial_pool_16, clip_vit_b_32}.pkl` 是 4 月 9 日构建，那时 builder 还未支持 `intermediates`，导致 `payload.intermediates=None`、`payload.denoising_num_steps=None`，**无法触发任何 WARM_START**（Orchestrator 校验失败一律降级为 MISS）。

**P0 范围**：
- `exp/cache_experiment/build_in_memory_cache_artifact.py:259` 已支持读全部 `noise_action_1..9`，`cp1_max_pool` / `cp1_spatial_pool_16` **本次直接重建即可**。
- `exp/cache_experiment/build_clip_cache_artifact.py:194` 目前只构造 `CachePayload(action_chunk, task_key)`，**不读 noise_action_* 也不写 intermediates / denoising_num_steps**。若直接用，CLIP 的 WARM_START 将全部被 Orchestrator 降级为 MISS → CLIP 实验无效。
- → **P0 只做 max_pool + spatial16 两个 artifact；CLIP 拆到 P0b**（见下）先把 builder 补齐再重建。第一轮实验允许只跑 max_pool + spatial16 (2 keybuilder × 3 start_t = 6 yaml)，CLIP 待 P0b 完成后并入全量。

源 HDF5 已确认：`data/db/libero_cache/libero_spatial/episode_*.h5`，50 episode，每 step 都含完整 `noise_action_1..9` 与 `input_images={base_0_rgb, left_wrist_0_rgb}` + `vision_0/1/2`、`prompt_emb`、`robot_state`。

**输出位置**：`data/cache_artifacts/libero_spatial_warm/{cp1_max_pool, cp1_spatial_pool_16}.pkl`
- **独立目录**，**不覆盖** `data/cache_artifacts/libero_spatial/*.pkl`，避免破坏 trajectory_deviation 实验复现性（[`logs/trajectory_deviation_corrective_implementation.log.md`](trajectory_deviation_corrective_implementation.log.md) §19.B8 明确要求 artifact 只读不可变）。

**命令**（2 个 keybuilder，每个一次）：

```bash
uv run python exp/cache_experiment/build_in_memory_cache_artifact.py \
    --data-dir data/db/libero_cache/libero_spatial \
    --builder-type cp1_max_pool \
    --output data/cache_artifacts/libero_spatial_warm/cp1_max_pool.pkl

uv run python exp/cache_experiment/build_in_memory_cache_artifact.py \
    --data-dir data/db/libero_cache/libero_spatial \
    --builder-type cp1_spatial_pool_16 \
    --output data/cache_artifacts/libero_spatial_warm/cp1_spatial_pool_16.pkl \
    --reducer-type spatial_pool --output-tokens 16
```

**验证**：

```bash
uv run python -c "
import pickle
for name in ['cp1_max_pool','cp1_spatial_pool_16']:
    with open(f'data/cache_artifacts/libero_spatial_warm/{name}.pkl','rb') as f:
        obj = pickle.load(f)
    e = obj['entries'][0]
    keys = sorted(e.payload.intermediates.keys())
    assert keys == [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], keys
    assert e.payload.denoising_num_steps == 10
    print(name, 'OK', len(obj['entries']), 'entries')
"
```

### 3.1b P0b — CLIP Artifact Builder 补齐 intermediates（CLIP 路线解阻）

**范围**：`exp/cache_experiment/build_clip_cache_artifact.py` 的 CachePayload 构造，复用 `build_in_memory_cache_artifact.py:250-266` 的 `noise_action_*` 读取 + canonical timestep 映射逻辑：

```python
# 现状（line 194）
payload = CachePayload(action_chunk=action, task_key=task)

# 目标：同步写入 intermediates + denoising_num_steps
_NUM_STEPS = 10
intermediates = None
denoising_num_steps = None
noise_indices = [
    int(k.split("_")[-1]) for k in group.keys()
    if k.startswith("noise_action_") and k.split("_")[-1].isdigit()
]
noise_indices = sorted(i for i in noise_indices if 1 <= i < _NUM_STEPS)
if noise_indices:
    denoising_num_steps = _NUM_STEPS
    intermediates = {
        round(1.0 - i / _NUM_STEPS, 4):
            torch.from_numpy(np.array(group[f"noise_action_{i}"])).float()
        for i in noise_indices
    }
payload = CachePayload(
    action_chunk=action, task_key=task,
    intermediates=intermediates, denoising_num_steps=denoising_num_steps,
)
```

**Level**：本身是 L1 小改动（机械抄写），但因为涉及缓存 payload 写入路径，**并入 P1 的 G2 审查**（避免多次 gate）。

**命令**（P0b 产出 CLIP artifact）：

```bash
uv run python exp/cache_experiment/build_clip_cache_artifact.py \
    --data-dir data/db/libero_cache/libero_spatial \
    --fields vision_0,vision_1,prompt_emb,robot_state \
    --output data/cache_artifacts/libero_spatial_warm/clip_vit_b_32.pkl
# 补齐后同样跑 §3.1 验证脚本（key 集合 {0.1..0.9}）
```

**`--fields` 必须显式写全**：`build_clip_cache_artifact.py` 默认 `--fields vision_0,robot_state`，而 `configs/cache_runs/deviate_exp/clip_w7_d4.yaml` 的 `keys` 启用了 `vision_0` / `vision_1` / `prompt_emb` / `robot_state` 四项，`search_strategy.field_similarity` 也为四项配了 similarity。如不传 `--fields`，新 pkl 的 `vector_dims` 只含 2 项 → `InMemoryBackend.load_artifact()` 与 warm YAML 的 `backend.vector_dims` 校验失败，要么启动失败，要么需改 warm YAML 禁用 `vision_1` + `prompt_emb`——后者会偏离 Step 1a 的 `clip_w7_d4` baseline，baseline 失效。故强制显式写 4-field。

**P0b 完成后**，§3.3 P2 的 `clip/` 子目录 3 份 YAML 接入全量实验；若 owner 决定第一轮不等 CLIP，可在 P3 结束后作为增量补跑（已有 baseline 不变）。

### 3.2 P1 — 新 Judge 组件 `AlwaysWarmStartJudge` (L2)

**位置**：`src/openpi/cache/components/judge.py`

**接口**：

```python
# 新增于 src/openpi/cache/types.py（leaf module，避免 config ↔ judge 循环导入）：
#     CANONICAL_DENOISE_TIMESTEPS: frozenset[float] = frozenset(
#         round(1.0 - i / 10, 4) for i in range(1, 10)
#     )  # {0.1, 0.2, ..., 0.9}
# 同步替换 src/openpi/cache/config.py:642 当前局部 `_canonical_timesteps`，
# 以及 judge.py 本模块，均改为从 types 导入同一常量。
from openpi.cache.types import CANONICAL_DENOISE_TIMESTEPS

class AlwaysWarmStartJudge:
    """Always returns WARM_START with a fixed start_t for the top-1 result.

    Used to sweep the success_rate ~ start_t curve under a constant (forced)
    warm-start regime, independent of similarity score. Empty result set
    falls back to MISS (cache truly empty / first step of episode).

    Restricted to CP1 — CP3 has no warm start support. Config-level
    validation rejects CP3 usage; the runtime FULL_HIT fallback below is
    defensive only and should be unreachable for validated configs.
    """

    def __init__(self, start_t: float) -> None:
        st = round(start_t, 4)
        if st not in CANONICAL_DENOISE_TIMESTEPS:
            raise ValueError(
                f"start_t must round to one of {sorted(CANONICAL_DENOISE_TIMESTEPS)}, got {start_t}"
            )
        self._start_t = st

    def __call__(self, results, checkpoint_id, cached_data) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        if checkpoint_id != CheckpointID.CP1:
            return JudgeResult(HitType.FULL_HIT, results[0].id)  # defensive, unreachable
        return JudgeResult(HitType.WARM_START, results[0].id, start_t=self._start_t)

    def on_episode_start(self) -> None: ...
    def record_action(self, action_chunk) -> None: ...
```

**配置层改动**：

`src/openpi/cache/config.py`
- `JudgeConfig` 新增字段 `start_t: float | None = None`（仅 `type=always_warm_start` 时使用）
- `_build_judge`（若尚无该工厂名，以 `config.py` 实际 judge 构造点为准）新增 branch：
  ```python
  elif cfg.type == "always_warm_start":
      return AlwaysWarmStartJudge(cfg.start_t)
  ```
- `validate_cache_config` 新增校验分支：
  - `judge.type == "always_warm_start"`：
    - `start_t` 必须提供（非 None）；`round(start_t, 4)` 必须 ∈ `CANONICAL_DENOISE_TIMESTEPS`（上文已移入 `types.py` 的共享常量；同一 validator 里 warm_tiers 分支也改为引用同一常量）
    - 禁止 CP3 使用（CP3 无 warm start payload）；违反时 fail-fast
    - 禁止与 `warm_tiers` 并用（二者语义冲突）
  - 校验通过后 **`cp_config.judge.start_t = round(cp_config.judge.start_t, 4)`** 规范化写回 cfg（避免 YAML `0.30000000000000004` 一类输入在 runtime `payload.intermediates[start_t]` 失配）。**注意**：这是 `always_warm_start` 的字段；现有 `warm_tiers` 的 `tier["start_t"]` 规范化逻辑（`config.py:680`）**保持不变**，两者互不影响
- YAML schema 解析白名单：若 `JudgeConfig` 的 type 字段有 `literal_values` / 预定义集合，需同步加入 `"always_warm_start"`（按 config.py 实际实现为准，G2 时逐条核验）

`src/openpi/cache/__init__.py`
- 导出 `AlwaysWarmStartJudge`

**测试**（**使用现有文件**，不要新建 `tests/cache/test_components_judge.py`）：
- `tests/cache/components/test_judge.py` 追加：
  - 空 results → `JudgeResult(MISS)`
  - CP1 命中 → `JudgeResult(WARM_START, winner_id, start_t=X)`
  - CP3 命中 → `JudgeResult(FULL_HIT, winner_id)` 并注释 "defensive, should be unreachable after config validation"
  - `AlwaysWarmStartJudge(0.5001)` → `__init__` 抛 ValueError（非 canonical timestep）
  - `AlwaysWarmStartJudge(0.30000000000000004)` → canonical 化为 `0.3`，不报错
- `tests/cache/test_config.py` 追加：
  - YAML 合法配置：`judge: {type: always_warm_start, start_t: 0.5}` + CP1 → `load_cache_config` + `validate_cache_config` 通过
  - 缺 start_t / 非法 start_t / CP3 配置 / 与 warm_tiers 并用 → `ConfigValidationError`
  - `load_cache_config(合法 yaml)` → `build_per_connection_components(cfg, storage)` 能构造并返回 `AlwaysWarmStartJudge` 实例

**P1 验收命令**：

```bash
uv run pytest tests/cache/components/test_judge.py tests/cache/test_config.py -v
```

### 3.3 P2 — 实验 YAML

**目录结构**（仅 9 份 YAML，无 baseline 子目录——B0/B1 用现成 deviate_exp 数据）：

```
configs/cache_runs/warm_start_exp/
├── max_pool/
│   ├── max_pool_w3_d5_warm_t0.7.yaml
│   ├── max_pool_w3_d5_warm_t0.5.yaml
│   └── max_pool_w3_d5_warm_t0.3.yaml
├── spatial16/
│   ├── spatial16_w8_d4_warm_t0.7.yaml
│   ├── spatial16_w8_d4_warm_t0.5.yaml
│   └── spatial16_w8_d4_warm_t0.3.yaml
└── clip/
    ├── clip_w7_d4_warm_t0.7.yaml
    ├── clip_w7_d4_warm_t0.5.yaml
    └── clip_w7_d4_warm_t0.3.yaml
```

**模板**：每份实验 YAML 完整继承对应 `configs/cache_runs/deviate_exp/{cfg}/{cfg}.yaml`，仅改三段：

```yaml
# 之前：
checkpoints:
  cp1:
    enabled: true
    gate: { type: always_search }
    judge: { type: always_hit }
    search_strategy: { ... }

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null        # ← deviate_exp 原状，latency 指标不会落盘

# 之后：
checkpoints:
  cp1:
    enabled: true
    gate: { type: always_search }                        # 不变
    judge:
      type: always_warm_start
      start_t: 0.5                                       # 三档分别 0.7 / 0.5 / 0.3
    search_strategy: { ... }                             # 不变

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: data/warm_start_exp/timing/max_pool_w3_d5_warm_t0.5   # ← 必填，否则 §2.3 延迟指标无法产出

backend:
  type: in_memory
  in_memory:
    preload_path: data/cache_artifacts/libero_spatial_warm/cp1_max_pool.pkl   # 指向新 artifact
```

每份 YAML 的 `output_csv_dir` 路径：`data/warm_start_exp/timing/<keybuilder>_warm_t<start_t>`。`write_policy` 保持 `never`（同 deviate_exp）。

**为什么必须在 YAML 里显式设 `output_csv_dir`**：
- `scripts/serve_policy.py:333-335` 的 `--timing_csv_dir` CLI flag 只对**非 `--cache_config` 路径**生效；
- 一旦 server 启动时带了 `--cache_config` 或收到 `load_cache_config` 控制消息，timer 由 cache_config 内部的 `timer.output_csv_dir` 决定（见 `build_per_connection_components`）；
- deviate_exp YAML 默认 `output_csv_dir: null`，不落 CSV；warm_start_exp 若沿用即拿不到 §2.3 要求的 `stage3_warm` / `stage3_flow` / `cp1_sum` 延迟数据。

### 3.4 P3 — 跑实验

**复用 trajectory_deviation 的 server + libero_sim 评估端**（[`docs/experiments/trajectory_deviation.md`](../docs/experiments/trajectory_deviation.md) §Step 0），只改启动 bundle 为新 artifact 之一即可。

#### 3.4.1 资源 & 端口约定

- **算力上限**：最多同时 6 个独立 GPU server + 6 个一一对应的 client。
- **端口分配**（从高到低，**从 8000 / 9000 优先用起**）：
  | Slot | Server 本机端口 | frp 外网端口 |
  |------|---------------|-------------|
  | 1 | 8000 | 9000 |
  | 2 | 7999 | 8999 |
  | 3 | 7998 | 8998 |
  | 4 | 7997 | 8997 |
  | 5 | 7996 | 8996 |
  | 6 | 7995 | 8995 |
- 每个 server 必须 `--concurrent` + `--collect`（和 trajectory_deviation Step 0 同），`--cache_config` 启动时指向**对应 keybuilder 的 inference bundle**（`configs/cache_runs/deviate_exp/{cfg}/inference_{cfg}.yaml`），driver 会用 `send_load_cache_config` 切到 warm start bundle。

#### 3.4.2 推荐方案 A — 3 server × 3 yaml（每 server 只服务同一 keybuilder）

YAML 已天然按 keybuilder 分目录，**不需要再拆**。每个 server 起始加载该 keybuilder 的 artifact；driver 依次跑 3 个 start_t yaml 时，`run_cache_experiments.py` 对每个 yaml 调 `send_load_cache_config` → server 端 `websocket_policy_server.py:287` 会 **每次都走 `build_shared_storage(cache_config)` → `InMemoryBackend().load_artifact(preload_path)`**（当前代码无 preload_path 缓存层，见 `src/openpi/cache/config.py:795-799`）。因此同一 keybuilder 的 3 份 yaml 虽然指向同一 pkl，**artifact 仍会被重复 pickle.load 3 次**。

**切换成本（实测等级）**：
| Artifact | 大小 | 预期每次 load | 3 yaml 累计 |
|---------|------|--------------|-------------|
| `cp1_max_pool.pkl` | 36 MB | <1 s | <3 s |
| `cp1_spatial_pool_16.pkl` | 412 MB | 数秒 | ~15-30 s |
| `clip_vit_b_32.pkl` | 189 MB | 1-2 s | 3-6 s |

相对于 1500 ep/slot 的跑量（分钟 → 小时级），累计重载几十秒可忽略。**本实验接受该成本，不引入 path-level storage 缓存**（属于独立优化，超出 plan 范围）。若 owner 后续想提速，可在另一 L2 plan 里加「按 preload_path 缓存已构建的 `shared_storage`」的小改动；本 plan 不涉及。

每个 server 的初始 bundle 及 driver 切换路径：

| Slot | Client `--yaml-dir` | Client `--port` | Server `--cache_config`（启动） | 加载的 artifact |
|------|--------------------|-----------------|--------------------------------|----------------|
| 1 | `warm_start_exp/max_pool` | 9000 | `deviate_exp/inference_max_pool_w3_d5.yaml` | `libero_spatial_warm/cp1_max_pool.pkl` |
| 2 | `warm_start_exp/spatial16` | 8999 | `deviate_exp/inference_spatial16_w8_d4.yaml` | `libero_spatial_warm/cp1_spatial_pool_16.pkl` |
| 3 | `warm_start_exp/clip` | 8998 | `deviate_exp/inference_clip_w7_d4.yaml` | `libero_spatial_warm/clip_vit_b_32.pkl` |

3 个 client 在 3 个独立终端同时启动，互不干扰（不同 port、不同 state-path、不同 yaml-dir）。

#### 3.4.3 备选方案 B — 6 server 极限并发（拆 yaml-dir）

若想压满 6 server，需要把 9 份 YAML 拆到独立 yaml-dir，每个 dir 1 份，6 个并发 + 3 个串行排队：

```
configs/cache_runs/warm_start_exp_split/
├── slot_max_pool_t07/ → max_pool_w3_d5_warm_t0.7.yaml   (软链或复制)
├── slot_max_pool_t05/ → max_pool_w3_d5_warm_t0.5.yaml
├── slot_max_pool_t03/ → max_pool_w3_d5_warm_t0.3.yaml
├── slot_spatial16_t07/ ...
... 共 9 个子目录
```

**收益有限**：方案 A 9 yaml × 500 ep / 3 并发 ≈ 1500 ep/slot；方案 B 9 yaml × 500 ep / 6 并发 ≈ 750 ep/slot 但有 3 份排队，实际 throughput 提升 < 50%，且 spatial16 (412MB) 重复加载到 3 个 server 显存翻倍。**默认采纳方案 A，方案 B 仅在显存充裕且 owner 主动要求时启用**。

#### 3.4.4 Smoke pass（单 yaml × 单 task × 5 ep，最小验通路径）

目标：最快验证 (i) YAML 被 server 正确加载 (ii) WARM_START 路径实际触发 (iii) Orchestrator 未降级。

```bash
# 仅启 1 个 server (slot 1, port 8000 / frp 9000)，默认 INFO 级别即可
# （P1 已把 WARM_START 降级日志从 debug 提升到 warning，见下文说明）
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/max_pool \
    --runs 2 \
    --task-ids 0 \
    --task-suite libero_spatial \
    --host <frp-host> --port 9000 \
    --episodes-per-run 5 \
    --num-workers 1 \
    --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_smoke.json
```

**WARM_START 降级日志可见性**：`scripts/serve_policy.py` 末尾固定 `logging.basicConfig(level=logging.INFO, force=True)` 且无 `--log-level` 参数，debug 日志默认不输出。因此 **P1 把 `src/openpi/cache/orchestrator.py:261` 的 `logger.debug("WARM_START payload incomplete ...")` 升级为 `logger.warning(...)`**（保持 format string 不变），使降级事件在默认 INFO 级别下可见（详见 §5 文件清单）。降级事件本就是"配置/artifact 不匹配"的错误信号，应当默认可见。

- `--runs 2` 只跑 `max_pool/` 下第 2 个 yaml（`max_pool_w3_d5_warm_t0.5.yaml`）；`run_cache_experiments.py:326` 支持 `1-8` 或 `1,3,5` 语法
  - **排序依赖说明**：`run_cache_experiments.py:677` 用 `sorted(yaml_dir.glob("*.yaml"))`，即 **lexicographic 字符串排序**。当前命名 `max_pool_w3_d5_warm_t0.3.yaml` / `_t0.5.yaml` / `_t0.7.yaml` 字典序为 `0.3 → 0.5 → 0.7`，因此 `--runs 2` = `t0.5`。本实验固定 3 档，接受此隐式依赖，**不引入 `01_/02_/03_` 前缀**（避免下游 state-path / log 路径跨实验不一致）；未来若扩档到 `t1.0` 再行重命名
- `--task-ids 0` 只跑 task 0
- `--episodes-per-run 5` + `--num-workers 1` = 5 个串行 episode

**通过条件**：
1. `cache_eval_results.json` 出现 `config_id == "max_pool_w3_d5_warm_t0.5"`，5 条 record
2. **server stderr / log file 中 `grep -cE "judge: WARM_START"` ≥ 1**（确认 WARM_START 实际生效）
3. **`grep -c "WARM_START payload incomplete"` == 0**（确认 artifact intermediates 完整，无降级）
4. `success_rate` 在合理区间（single task 5 ep 统计意义有限，目标是不崩）

若条件 3 不满足：停下来检查 P0 artifact 的 `intermediates` 字段（§3.1 验证脚本）。

**三档 start_t 加载验证**（可选，smoke 通过后追加）：

```bash
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/max_pool \
    --runs 1-3 --task-ids 0 --episodes-per-run 1 --num-workers 1 \
    --task-suite libero_spatial --host <frp-host> --port 9000 \
    --seed 42 --conda-env libero_sim \
    --state-path data/warm_start_exp/state_smoke_all3.json
```

确认 3 份 yaml 都能被 `load_cache_config` 接受（server log 出现 3 次 `Cache bundle updated to v*`）。

#### 3.4.5 全量（方案 A，3 路并发，每 path 500 init）

3 个独立终端同时执行：

```bash
# Terminal 1 — slot 1 (port 9000) — max_pool
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/max_pool \
    --task-suite libero_spatial \
    --host <frp-host> --port 9000 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_full_max_pool.json

# Terminal 2 — slot 2 (port 8999) — spatial16
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/spatial16 \
    --task-suite libero_spatial \
    --host <frp-host> --port 8999 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_full_spatial16.json

# Terminal 3 — slot 3 (port 8998) — clip
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/warm_start_exp/clip \
    --task-suite libero_spatial \
    --host <frp-host> --port 8998 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path data/warm_start_exp/state_full_clip.json
```

> ⚠️ 三个 client 的 `--state-path` **必须不同**（否则 RunState 互相覆盖）；`--num-workers` 是 client 内 task 级并发，与 server 并发数独立——3 client × 5 worker = 15 个 libero env 子进程，确保评估端 CPU 够。

#### 3.4.6 产物 & 失败追踪

每个 yaml-dir 根下生成独立的 `cache_eval_results.json`：
- `configs/cache_runs/warm_start_exp/max_pool/cache_eval_results.json`
- `configs/cache_runs/warm_start_exp/spatial16/cache_eval_results.json`
- `configs/cache_runs/warm_start_exp/clip/cache_eval_results.json`（P0b 完成后才有）

归档（P3 结束后立即执行，防止被二次运行覆盖）：

```bash
mkdir -p data/warm_start_exp/results
for cfg in max_pool spatial16 clip; do
    src=configs/cache_runs/warm_start_exp/$cfg/cache_eval_results.json
    [ -f "$src" ] && cp "$src" data/warm_start_exp/results/cache_eval_results_${cfg}.json
done
# 合并为单一文件供 P4 直接读入（P4 也可直接读 3 份，下方脚本给 merge 版本）
uv run python -c "
import json, glob
out = []
for fn in sorted(glob.glob('data/warm_start_exp/results/cache_eval_results_*.json')):
    out.extend(json.load(open(fn)))
json.dump(out, open('data/warm_start_exp/results/cache_eval_results.json','w'), indent=2)
print('merged', len(out), 'records')
"
```

**归档产物（P4 输入源）**：
- `data/warm_start_exp/results/cache_eval_results_{max_pool,spatial16,clip}.json` — 3 份分路结果
- `data/warm_start_exp/results/cache_eval_results.json` — 合并后单文件（P4 默认读此文件）

**Per-init 失败追踪**：driver 已在每条 record 写入 `(config_id, task_id, init_state_idx, seed, success)`，P4 分析脚本直接对 3 份 JSON 做 union + set 运算与 §2.2 baseline 比对，**runner 不需要任何改动**。

#### 3.4.7 预估规模

- 方案 A：9 YAML × 500 ep / 3 并发 ≈ 1500 ep / slot；按 deviate_exp Step 1a 的 throughput（owner 提供经验值后可填入预估时长）
- 总 episode：4500（不含 baseline）

### 3.5 P4 — 分析

新增脚本 `exp/cache_experiment/analyze_warm_sweep.py`：

**输入**：
- 实验：`data/warm_start_exp/results/cache_eval_results.json`
- 基线：`data/warm_start_exp/baseline_failures.json`（§2.2 已生成）

**输出**：
1. **主图 1 — Success Rate Sweep**：横轴 start_t（0.3 / 0.5 / 0.7），纵轴 success_rate；每个 keybuilder 一条曲线；
   加两条水平虚线：B0（每 cfg 自己的 inference rate，约 99%）、B1（每 cfg 自己的 always-hit rate，约 67-69%）。
2. **主图 2 — Recovery on B1 Failure Set**：
   横轴 start_t，纵轴 `recovery_rate = |warm_pass ∩ B1_fail(cfg)| / |B1_fail(cfg)|`；
   加水平虚线 0（B1 自身定义）和 ~1（B0 在 B1 fail 集上的恢复率，约 (163-4)/163 等）。
3. **主图 3 — Incurred Loss**：横轴 start_t，纵轴 `incurred_loss = |warm_fail ∩ B0_pass(cfg)| / |B0_pass(cfg)|`；越低越好。
4. **副图**：`mean_step_latency vs start_t`（来自 timer csv，若启用）。
5. **CSV 汇总** `summary.csv`：cfg, start_t, n_total, n_success, success_rate, recovery_rate, incurred_loss, p_value(McNemar vs B1)；
   另出 `failure_intersection.csv`：每 (cfg, start_t) 失败 init 集合与 B1_fail 三 cfg 交集（51）的重叠数，看 warm start 是否解掉了"硬骨头"。

### 3.6 P5 — 文档

| 文件 | 操作 |
|------|------|
| `docs/experiments/warm_start_sweep.md` | **新增** 运行手册（结构对齐 `trajectory_deviation.md`：实验总览 / 网络拓扑 / 前置条件 / Step 0–4） |
| `docs/experiments/README.md` | 索引追加一行 |
| `docs/README.md` | `experiments/` 章节追加一行 |
| `logs/warm_start_sweep_plan.log.md` | **本文件**，验收后状态从 `Plan` → `Implemented` → `Validated` 推进，最终归档至 `logs/archive/` |
| `logs/README.md` | Active Logs 区块新增一行 |

---

## 4. 依赖关系图

```
P0 (artifact rebuild)
  └── independent of P1
P1 (Judge + config + tests)            ← G1 must pass
  └── G1 review
       └── implementation
            └── G2 review
                 └── P2 (YAML)
                      └── P3 (run experiments, depends P0+P2)
                           └── P4 (analyze)
                                └── P5 (docs)
```

> P0 与 P1 可并行（仅依赖现状代码），P2 必须等 P1 G2 通过（YAML schema 依赖新 judge type）。

---

## 5. 改动文件清单

| 文件 | Phase | 改动 |
|------|-------|------|
| `data/cache_artifacts/libero_spatial_warm/cp1_max_pool.pkl` | P0 | 新建（`build_in_memory_cache_artifact.py` 已支持 intermediates，直接重建） |
| `data/cache_artifacts/libero_spatial_warm/cp1_spatial_pool_16.pkl` | P0 | 同上 |
| `exp/cache_experiment/build_clip_cache_artifact.py` | **P0b** | 复用 `build_in_memory_cache_artifact.py:250-266` 的 `noise_action_*` 读取 + canonical timestep 映射，为 `CachePayload` 补齐 `intermediates` / `denoising_num_steps`（解阻 CLIP WARM_START 路径） |
| `data/cache_artifacts/libero_spatial_warm/clip_vit_b_32.pkl` | **P0b** | P0b builder 改完后新建 |
| `src/openpi/cache/types.py` | P1 | 新增模块级常量 `CANONICAL_DENOISE_TIMESTEPS: frozenset[float] = frozenset(round(1.0 - i/10, 4) for i in range(1, 10))`；leaf module，无循环依赖风险 |
| `src/openpi/cache/components/judge.py` | P1 | `AlwaysWarmStartJudge`（引入 `CANONICAL_DENOISE_TIMESTEPS` + CP3 防御 fallback） |
| `src/openpi/cache/config.py` | P1 | (a) `JudgeConfig.start_t` 字段 + `_build_judge` 分支 + `validate_cache_config` 分支（复用 `CANONICAL_DENOISE_TIMESTEPS`；`cp_config.judge.start_t = round(..., 4)` 规范化写回；warm_tiers 校验里的 `_canonical_timesteps` 局部 set 一并替换为同一 import）；(b) judge type 白名单加 `always_warm_start` |
| `src/openpi/cache/orchestrator.py` | P1 | 把 `logger.debug("WARM_START payload incomplete ...")`（line 261）提升为 `logger.warning(...)`，使默认 INFO 级别下降级事件可观测 |
| `src/openpi/cache/__init__.py` | P1 | 导出 `AlwaysWarmStartJudge` |
| `tests/cache/components/test_judge.py` | P1 | **追加**用例（empty→MISS / CP1→WARM_START / CP3 防御 FULL_HIT / 非 canonical raise / `0.30000000000000004` 规范化通过） |
| `tests/cache/test_config.py` | P1 | **追加**用例（合法 YAML / 缺 start_t / 非法 start_t / CP3 / 与 warm_tiers 并用 → `ConfigValidationError`；`build_per_connection_components` 构造返回 `AlwaysWarmStartJudge`） |
| `configs/cache_runs/warm_start_exp/{max_pool,spatial16,clip}/*.yaml` | P2 | 9 份，每份显式写 `timer.output_csv_dir: data/warm_start_exp/timing/<cfg>_warm_t<start_t>`（否则 §2.3 latency 指标无产出），`backend.in_memory.preload_path` 指向 `libero_spatial_warm/` |
| `data/warm_start_exp/baseline_failures.json` | 预备 | 已生成；汇总 trajectory_deviation Step 1a 的 inference / always_hit 失败集 |
| `exp/cache_experiment/analyze_warm_sweep.py` | P4 | 新建；读 `data/warm_start_exp/results/cache_eval_results.json`（合并后单文件） |
| `docs/experiments/warm_start_sweep.md` | P5 | 新建 |
| `docs/experiments/README.md` | P5 | 索引同步 |
| `docs/README.md` | P5 | 索引同步 |
| `logs/warm_start_sweep_plan.log.md` | P5 | 本文件状态推进 |
| `logs/README.md` | P5 | 索引同步 |

**不改动的文件**：
- `data/cache_artifacts/libero_spatial/*.pkl`（trajectory_deviation 实验依赖）
- `data/db/libero_cache/libero_spatial/*.h5`（HDF5 已含完整 noise_action_1..9，无需重采）
- `src/openpi/cache/interceptor.py`（warm start 通路已就绪，仅 `orchestrator.py:261` 降级日志 level 调整为 warning，详见上表）
- `configs/cache_runs/deviate_exp/`（保持原状）

---

## 6. 风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| start_t=0.3 仅剩 3 步 denoise 不足以收敛，warm start 输出退化为噪声 | 中 | 这是要验证的核心问题；副图记录每步 `stage3_warm` 输出与 B0 (always_skip) 的 action L2 距离（事后分析） |
| 新 artifact 路径与 deviate_exp 旧路径并存导致磁盘膨胀 | 低 | 三个 pkl 总计 ~640 MB；可接受；跑完不归档时手动 prune |
| 校验逻辑放行新 judge 时不慎放过非法组合（如 always_warm_start + CP3） | 中 | 单元测试覆盖（§3.2 test_config.py 列表）；G2 代码审查重点核 |
| Server bundle 切换在 multi-worker 下抢改（trajectory_deviation 已遇过） | 低 | 方案 A 下每 server 只服务同一 keybuilder 的 3 个 yaml，driver 单串行控制切换；server 必须 `--concurrent` |
| 3 client 并发时 evaluation 端 CPU / 网络打满 | 中 | 3 client × `--num-workers 5` = 15 libero env 子进程；先单 client smoke 跑通后再起 3 路；评估端 nproc < 16 时降 num-workers |
| 三个 server 占用同一物理 GPU 显存超限（spatial16 412MB + clip 189MB + max_pool 36MB + 3×Pi0.5 权重） | 中 | 若单 GPU 不够，按 `CUDA_VISIBLE_DEVICES` 分卡（commit `8bb0289` 已支持）；显存估算后再决定是否走方案 B |
| AlwaysWarmStartJudge CP3 fallback 行为是否合理（不抛错而是降级 FULL_HIT） | 低 | 校验阶段已经禁止 always_warm_start + CP3 配置；运行时 fallback 是双保险，对外行为可观测 |

---

## 7. 已确认设计决策

| # | 决策点 | 采纳方案 |
|---|--------|----------|
| 1 | Artifact 路径 | 独立目录 `data/cache_artifacts/libero_spatial_warm/`（不覆盖 `libero_spatial/`，保留 deviate_exp 复现性） |
| 2 | 对照组 | 复用 deviate_exp Step 1a 现成 B0/B1 数据（§2.2，已落 `baseline_failures.json`），不重跑 baseline |
| 3 | 样本量 | 暴力跑满 500 init（50 ep × 10 task）；先 smoke 1 task × 5 ep 验通（§3.4.4） |
| 4 | 并行方案 | 方案 A：3 server (port 9000 / 8999 / 8998) × 3 yaml，按 keybuilder 子目录天然分组 |
| 5 | Judge 命名 / 空命中 | `AlwaysWarmStartJudge`；空 results → MISS；CP3 → FULL_HIT 防御（config 校验层已禁止 CP3，运行时分支不可达） |
| 6 | CP3 处理 | 保持 disabled（与 deviate_exp 一致） |

---

## 8. 验收标准（Verify 阶段）

- [ ] P0 两个 CP1 artifact（`cp1_max_pool.pkl` / `cp1_spatial_pool_16.pkl`）通过 §3.1 验证脚本（`intermediates` 含 9 个 canonical key，`denoising_num_steps == 10`）
- [ ] P0b CLIP artifact（`clip_vit_b_32.pkl`）通过同一验证脚本（若第一轮暂不跑 CLIP，P0b 可延后，但 §3.3 `clip/` 3 份 YAML 在 P0b 验收前**不得进入全量**）
- [ ] P1 单元测试全绿：`uv run pytest tests/cache/components/test_judge.py tests/cache/test_config.py -v`
- [ ] 端到端 smoke（严格对齐 §3.4.4 命令）：`--yaml-dir configs/cache_runs/warm_start_exp/max_pool --runs 2 --task-ids 0 --episodes-per-run 5 --num-workers 1` 跑完后同时满足：
  - [ ] `cache_eval_results.json` 出现 `config_id == "max_pool_w3_d5_warm_t0.5"` 且正好 5 条 record
  - [ ] **server log `grep -cE "judge: WARM_START"` ≥ 1**（WARM_START 路径实际触发）
  - [ ] **server log `grep -c "WARM_START payload incomplete"` == 0**（Orchestrator 未把 WARM_START 降级为 MISS；P1 已把该日志从 debug 升级为 warning，默认 INFO 级别即可可见；若不为 0，回到 §3.1 / §3.1b 重建 artifact）
  - [ ] `data/warm_start_exp/timing/max_pool_w3_d5_warm_t0.5/` 下至少 1 份 `timing_task_*.csv`（证实 timer CSV 通路有效）
- [ ] （可选）3 档 start_t 加载验证：`--runs 1-3 --task-ids 0 --episodes-per-run 1` 后 server log 出现 **3 次** `Cache bundle updated to v*`
- [ ] P3 全量结果产出：归档后的 `data/warm_start_exp/results/cache_eval_results.json` 包含 9 个 warm config × 500 init 共 **4500 record**（若 CLIP 延后则 6 × 500 = 3000），无 `attempt > 1` 异常
- [ ] P3 sanity：每个 cfg 的 `success_rate` 落在对应 `(B1, B0)` 区间 `[67%, 99%]`（超出 B0 或低于 B1 需排查）
- [ ] P4 主图 1-3 + `summary.csv` + `failure_intersection.csv` 全部生成；趋势可解释（**不要求方向，但单调性须可论证**；非单调时排查 start_t 校验或 artifact intermediates 完整性）
- [ ] P5 文档 + 索引全部更新；本 plan log 状态 `Plan → Implemented → Validated` 推进，最终移入 `logs/archive/`

---

## 9. 致审查者（G2 Review）

> Status: Awaiting G2 review. Code 阶段 P0b + P1 + P2 已落地，P0 artifact 重建 / P3 / P4 / P5 未动（按 plan 依赖图需等 G2 通过后再跑）。

### 9.1 本轮实际改动

| 文件 | 行数 / 改动 | 性质 |
|------|------------|------|
| `src/openpi/cache/types.py` | 新增 `CANONICAL_DENOISE_TIMESTEPS: frozenset[float]` 模块级常量（≈10 行，含注释） | 纯新增；leaf module 无循环依赖风险 |
| `src/openpi/cache/components/judge.py` | `+AlwaysWarmStartJudge` 类 + 顶部 import 添加 `CANONICAL_DENOISE_TIMESTEPS` | 纯新增；未动 `AlwaysHitJudge` / `ThresholdJudge` |
| `src/openpi/cache/orchestrator.py` | line 261 `logger.debug` → `logger.warning`（format string 未变） | 单字段改动；使 WARM_START 降级事件在 INFO 日志级别可见 |
| `src/openpi/cache/config.py` | (a) import `CANONICAL_DENOISE_TIMESTEPS`；(b) `JudgeConfig.start_t: float \| None = None`；(c) judge type 白名单 + 错误文案加 `always_warm_start`；(d) `_build_judge` 加 branch；(e) 新增 `always_warm_start` 独立校验块（CP1-only / 禁 warm_tiers / 必填 start_t / canonical / 规范化写回）；(f) `warm_tiers` 校验里的局部 `_canonical_timesteps` set 替换为同一 import | 配置层全部落点；语义互斥 `always_warm_start` ↔ `warm_tiers` 在校验层阻断 |
| `src/openpi/cache/__init__.py` | 导出 `AlwaysWarmStartJudge`（顺带导出此前已遗漏的 `AlwaysHitJudge`，供测试直接 import） | 公共 API 扩展 |
| `exp/cache_experiment/build_clip_cache_artifact.py` | line 194 附近：`CachePayload(action_chunk=action, task_key=task)` → 加 `intermediates` / `denoising_num_steps`，镜像 `build_in_memory_cache_artifact.py:250-266` | P0b；解阻 CLIP WARM_START |
| `tests/cache/components/test_judge.py` | 追加 5 个用例（`empty→MISS` / `CP1→WARM_START` / `CP3→FULL_HIT 防御` / `start_t=0.5001 ValueError` / `0.30000000000000004 → 0.3 规范化`）+ protocol 合规用例 | 单元测试 |
| `tests/cache/test_config.py` | 追加 7 个用例（合法 YAML / 缺 start_t / 非 canonical / CP3 / 与 warm_tiers 并用 / 0.30000…4 规范化 / `build_per_connection_components` 返回 `AlwaysWarmStartJudge`） | 集成测试 |
| `configs/cache_runs/warm_start_exp/{max_pool,spatial16,clip}/*.yaml` | 9 份（3 keybuilder × 3 start_t）；每份 3 改点：`judge` 块、`timer.output_csv_dir`、`backend.in_memory.preload_path` | P2 |

### 9.2 已跑过的自测

- `uv run pytest tests/cache/components/test_judge.py tests/cache/test_config.py -v` → **74 passed**（含原有 60+ 用例回归绿 + 新增 12 用例）。
- 9 份 warm YAML 全部过 `load_cache_config + validate_cache_config`（startup path 等价校验），`judge.start_t` 都按预期被规范化 / 保留。
- **未跑**：任何端到端 server/client / artifact 重建（按 plan §4 依赖图，这些是 P0/P3 内容，需 G2 通过后再做）。

### 9.3 审查重点建议

1. **`config.py` 校验块次序**：`always_warm_start` 块放在 `warm_tiers` 块之前独立 for-loop，两次遍历 checkpoints。逻辑对但多一次扫描；G2 需确认这种分离可读性是否优于合并（我选择分离是因为两块判空、失败路径不同，混写容易 regressions）。
2. **CP3 防御 fallback**：`AlwaysWarmStartJudge` 在 CP3 返回 `FULL_HIT` 而非 raise。plan §7 #5 已确认采纳；测试有覆盖。G2 请确认是否同意"运行时可观测 > fail-fast"的选择。
3. **`__init__.py` 顺带导出 `AlwaysHitJudge`**：不是 plan 显式列的改动，但 `test_config.py` 新增用例用到 `from openpi.cache.components.judge import AlwaysWarmStartJudge` 而非 `from openpi.cache`，不依赖这条导出。我仍然加了它（语义对称：`AlwaysHit` 既然是"总是命中"，与 `AlwaysWarmStart` 成对），若 G2 认为 scope creep 可回退为仅导出 `AlwaysWarmStartJudge`。
4. **orchestrator 日志**：从 `logger.debug` 升到 `logger.warning` 会让任何 WARM_START artifact 不全的场景在生产也刷日志。P2 的 9 个 YAML 都指向新重建 artifact，正常情况 0 触发；但 trajectory_deviation 的旧 artifact 若再跑 warm_tiers 流程会刷日志。G2 确认这是可接受代价（按 plan §3.4.4 判断：降级本就是错误信号，应可见）。
5. **P2 YAML**：保留 deviate_exp baseline 的 `trajectory_depth` / `trajectory_weights` / `rrf_k` / `field_similarity` / `keys` 权重全套不变；**唯一差异**是 judge 块 + timer csv 路径 + preload 路径，以便 warm vs baseline 可直接做 per-cfg 对比（P4 消除变量）。

### 9.4 未动文件（明示）

- `src/openpi/cache/interceptor.py`（WARM_START 通路之前就写好，无需改）
- `src/openpi/cache/components/{gate.py, key_builder.py, search_strategy.py}`
- `exp/cache_experiment/build_in_memory_cache_artifact.py`（已含 intermediates 读取）
- `configs/cache_runs/deviate_exp/`
- `scripts/serve_policy.py` / `src/openpi/serving/websocket_policy_server.py`

### 9.5 审查通过后下一步

1. 按 §3.1 / §3.1b 重建 3 份 artifact（先 max_pool + spatial16，CLIP 可并行或延后）
2. §3.4.4 smoke pass（1 server × 1 task × 5 ep）
3. 全量 P3（3 server 并发）→ P4 分析 → P5 文档 + 索引同步推进

---

## 10. G2 审查结论（2026-04-15）

**结论：G2 通过。** P0b + P1 + P2 的实现与 G1 放行范围一致，未发现阻断问题；可以进入 P0 artifact 重建与 §3.4.4 smoke。

### 10.1 审查范围

- 代码：`src/openpi/cache/types.py`、`src/openpi/cache/components/judge.py`、`src/openpi/cache/config.py`、`src/openpi/cache/orchestrator.py`、`src/openpi/cache/__init__.py`、`exp/cache_experiment/build_clip_cache_artifact.py`
- 测试：`tests/cache/components/test_judge.py`、`tests/cache/test_config.py`
- YAML：`configs/cache_runs/warm_start_exp/{max_pool,spatial16,clip}/*.yaml`
- 未把 P0 artifact 文件、P3 端到端实验、P4 分析脚本纳入本轮通过条件；这些仍按 §4 / §8 在后续阶段验收。

### 10.2 Findings

- **阻断问题：无。**
- `AlwaysWarmStartJudge` 的行为符合 plan：空结果为 `MISS`，CP1 强制 `WARM_START` 并规范化 `start_t`，CP3 defensive fallback 为 `FULL_HIT`；配置层已禁止非 CP1 与 `warm_tiers` 并用。
- `config.py` 校验覆盖点完整：judge 白名单、`start_t` 必填、canonical timestep、规范化写回、CP1-only、`warm_tiers` 互斥，以及 factory branch 都已落地。
- `orchestrator.py` 的 incomplete payload 日志从 debug 升为 warning，解决默认 INFO 级别不可 grep 的问题；这会让旧 artifact 的 warm downgrade 更显眼，但该 downgrade 本身是错误信号，接受。
- `build_clip_cache_artifact.py` 已补 `intermediates` / `denoising_num_steps`，逻辑与 `build_in_memory_cache_artifact.py` 对齐。后续 P0b 仍必须按 §8 对实际 `clip_vit_b_32.pkl` 做 9 个 canonical key 验证。
- 9 份 warm YAML 与 deviate_exp baseline 的有效差异限定在 judge、timer CSV 路径、warm artifact 路径；`--runs 2` 对应 t0.5 的 lexicographic 依赖已实测成立。

### 10.3 已跑验证

- `uv run python -m pytest tests/cache/components/test_judge.py tests/cache/test_config.py -v` → **74 passed**
- `uv run python -m pytest tests/cache/test_orchestrator.py tests/cache/test_interceptor.py -q` → **34 passed**
- 9 份 `configs/cache_runs/warm_start_exp/**/*.yaml` 均通过 `load_cache_config`
- `configs/cache_runs/warm_start_exp/{clip,max_pool,spatial16}` 内 YAML 排序均为 `t0.3`、`t0.5`、`t0.7`，因此 `--runs 2` 会选择 `t0.5`

### 10.4 非阻断备注

- 当前工作区直接跑 `uv run pytest ...` 会命中 `.venv/bin/pytest` 的旧 shebang，入口脚本指向已不存在的环境；这不是本轮代码失败。G2 验证改用 `uv run python -m pytest ...`，建议后续修复本地 venv 或把 §8 的单测命令改成 module 形式以避免同类环境漂移。
- `exp/cache_experiment/build_clip_cache_artifact.py` 对 `noise_action_*` 做的是 best-effort 读取；如果实际 HDF5 缺少某些 timestep，代码会产出部分 `intermediates`。这不阻断 G2，因为 §8 的 artifact 验证会强制检查 9 个 canonical key，但 P0b 重建后必须执行该检查。

### 10.5 放行项

1. 允许重建 `data/cache_artifacts/libero_spatial_warm/cp1_max_pool.pkl`
2. 允许重建 `data/cache_artifacts/libero_spatial_warm/cp1_spatial_pool_16.pkl`
3. 允许用显式 `--fields vision_0,vision_1,prompt_emb,robot_state` 重建 `data/cache_artifacts/libero_spatial_warm/clip_vit_b_32.pkl`
4. artifact 验证通过后，允许按 §3.4.4 先跑 max_pool t0.5 smoke，再进入 P3 全量
