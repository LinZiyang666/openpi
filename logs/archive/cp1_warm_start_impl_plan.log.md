# CP1 Warm Start 实现计划

> Status: Validated
> Date: 2026-04-10
> Task: 将 warm start 从 CP2 迁移到 CP1 Judge 决策
> Level: L3 (Architectural)
> 前置文档: [cp1_warm_start_investigation.log.md](cp1_warm_start_investigation.log.md)

---

## 实现分期

共 4 个 Phase，每个 Phase 可独立提交和验证。Phase 间有严格的依赖顺序。

```
Phase 0: 性能修复 (前置)
    └── Phase 1: 数据写入链路
         └── Phase 2: 命中判定 + 执行链路
              └── Phase 3: 文档更新
```

---

## Phase 0: `_stage3_with_intermediates` 性能修复

**目标**：消除 `_stage3_with_intermediates()` 中的 GPU→CPU 同步（`.item()`），降低 eager 模式延迟。不承诺 compile safe，compile 兼容性留待实测验证。

**这是后续所有 Phase 的硬性前置依赖**。`.item()` 每次调用都强制 CPU 等待 GPU 完成所有未完成计算，10 步循环 = 10 次 pipeline stall。无论 eager 还是 compiled 模式，此同步开销都会显著影响 `return_intermediates=True` 路径的延迟。

### P0-1: 修改 `_stage3_with_intermediates()`

文件：`src/openpi/models_pytorch/pi0_pytorch.py:624-663`

改动：
- 删除循环内的 `timestep.item()` 调用
- 删除 `abs(t_val - st) < half_dt` 浮点比较
- 新增循环前预计算：`save_at = {round((1.0 - st) * num_steps): st for st in save_timesteps}`
- 循环内改用 `step in save_at` 纯 Python int 判断
- 新增 `step` 计数器（Python int，从 0 开始）

改前核心循环：
```python
while timestep >= -dt / 2:
    t_val = timestep.item()           # GPU→CPU sync
    for st in save_timesteps:
        if abs(t_val - st) < half_dt:
            intermediates[st] = x_t.clone()
            break
    ...
```

改后核心循环：
```python
save_at = {}
for st in save_timesteps:
    step_idx = round((1.0 - st) * num_steps)
    save_at[step_idx] = st

step = 0
while timestep >= -dt / 2:
    if step in save_at:
        intermediates[save_at[step]] = x_t.clone()
    ...
    step += 1
```

**不改**：`_stage3_action_expert()`、`run_stage3()` 分支结构、`run_stage3_from()`、`denoise_step()`。

### P0-T: 测试

| 测试 | 内容 |
|------|------|
| 单元测试 | `run_stage3(return_intermediates=True)` 返回值验证：intermediates 包含 3 个 key (0.7, 0.5, 0.3)，tensor 形状正确 |
| 回归测试 | `run_stage3(return_intermediates=False)` 行为不变，`intermediates=None` |
| 数值一致性 | monkeypatch `denoise_step` 为确定性线性函数，验证保存点位置（denoise 前）、key 为预期 timestep、最终 action 与 reference loop allclose |
| 延迟对比（可选） | eager 模式下 `return_intermediates=True` vs `False` 延迟无显著差异（排除 `.item()` stall）|
| compile 检查（可选） | `torch._dynamo.explain` 检查 graph break 数量，记录结果但不作为放行条件 |

---

## Phase 1: 数据写入链路

**目标**：让 intermediates 数据能通过离线和在线两条路径写入缓存。Phase 1 完成后，缓存中的 `CachePayload` 会包含 `intermediates` 和 `denoising_num_steps`，为 Phase 2 的命中判定和执行提供数据基础。

### P1-1: StepRecord 增加 Optional 字段 (W2)

文件：`src/openpi/cache/storage_types.py:291-299`

```python
@dataclass
class StepRecord:
    query_keys: dict[str, torch.Tensor]
    action_chunk: torch.Tensor
    intermediates: Optional[dict[float, torch.Tensor]] = None
    denoising_num_steps: Optional[int] = None
```

向后兼容：新字段默认 `None`，所有现有构造 StepRecord 的代码零改动。

### P1-2: `buffer_for_write()` 增加参数 (W3)

文件：`src/openpi/cache/orchestrator.py:269-282`

新增 `intermediates` 和 `denoising_num_steps` 两个 Optional 参数，透传到 StepRecord。

向后兼容：参数默认 `None`，现有调用方（FULL_HIT 路径）无需改动。

### P1-3: `_build_entry_chain()` 填充 payload (W4)

文件：`src/openpi/cache/orchestrator.py:329-332`

CachePayload 构造新增 `intermediates=step.intermediates` 和 `denoising_num_steps=step.denoising_num_steps`。

### P1-4: Artifact Builder 读取 noise_action (W1)

文件：`exp/build_in_memory_cache_artifact.py:167-171`

从 HDF5 读取**全部** `noise_action_1..9`，映射到 `CachePayload.intermediates = {0.9: tensor, 0.8: tensor, ..., 0.1: tensor}`。

存储侧保留完整数据，不做裁剪。Judge 的 `warm_tiers` 配置的 `start_t` 值必须是存储中实际存在的 key，由 Orchestrator 在 payload 完整性校验中保证（P2-2）。

映射公式：`noise_action_i` → `t = round(1.0 - i / num_steps, 4)`。按数字后缀排序，跳过非数字后缀。

```python
_NUM_STEPS = 10

intermediates = None
denoising_num_steps = None
noise_indices = []
for k in group.keys():
    if k.startswith("noise_action_"):
        suffix = k.split("_")[-1]
        if suffix.isdigit():
            noise_indices.append(int(suffix))

if noise_indices:
    denoising_num_steps = _NUM_STEPS
    intermediates = {}
    for i in sorted(noise_indices):
        t = round(1.0 - i / _NUM_STEPS, 4)
        intermediates[t] = torch.from_numpy(np.array(group[f"noise_action_{i}"])).float()
```

向后兼容：旧版 HDF5 无 `noise_action_*` 时，`intermediates=None`。

### P1-5: Interceptor MISS 路径开启 intermediates 采集 (W5 + W6)

文件：`src/openpi/cache/interceptor.py:351-371`

新增模块级常量：

```python
_NUM_STEPS = 10  # matches pi0_pytorch.run_stage3 default
```

改动：
1. MISS 路径的 `run_stage3` 调用加 `return_intermediates=True` 和显式 `num_steps=_NUM_STEPS`
2. `stage3.intermediates` 中每个 tensor 做 `[0].cpu().float().contiguous()`（去 batch 维 + 转 CPU）
3. `buffer_for_write` 传入 `intermediates` 和 `denoising_num_steps=_NUM_STEPS`

注意：
- **不能使用编译后的 `self._stage3_fn`**。`run_stage3(return_intermediates=True)` 走 `_stage3_with_intermediates` 分支，与 `return_intermediates=False` 的 `_stage3_action_expert` 是不同的执行路径。`torch.compile` 编译的是 `run_stage3` 整体，内部分支在编译时确定。需要确认编译行为：
  - 如果 `torch.compile` 的 `run_stage3` 能根据 `return_intermediates` 参数走不同 trace，则直接传参即可
  - 如果编译时固定了 `False` 路径，则 MISS 路径需要调用 `self._model.run_stage3(stage2, noise=start_noise, return_intermediates=True)` 绕过编译版本
  - 决策点：实际测试确认，初期可用 eager `run_stage3` 调用，后续优化
- FULL_HIT 路径已有 early return，不受影响
- Orchestrator 为 None 时（无缓存模式），保持原有行为（使用编译版 `self._stage3_fn`，不采集 intermediates）

### P1-5 编译策略详细方案

当 `orchestrator is not None`（缓存模式开启）时，Stage 3 有两种场景：

| 场景 | 需要 intermediates | 调用 |
|------|-------------------|------|
| MISS | 是 | `self._model.run_stage3(stage2, ..., num_steps=_NUM_STEPS, return_intermediates=True)` (eager) |
| WARM_START (Phase 2) | 否 | `self._model.run_stage3_from(stage2, ...)` (eager) |
| 无缓存模式 | 否 | `self._stage3_fn(stage2, ...)` (compiled) |

MISS 路径频率随缓存命中率提升而降低。初期缓存为空时全部 MISS，此时 Stage 3 走 eager 会比 compiled 慢，但这是冷启动阶段，可接受。

### P1-T: 测试

| 测试 | 内容 |
|------|------|
| 离线路径 | Artifact Builder 从含 `noise_action_1..9` 的 HDF5 构建 artifact，验证 payload.intermediates 包含 9 个 key（0.9, 0.8, ..., 0.1），tensor 形状正确 |
| 在线路径 | Mock Orchestrator，验证 buffer_for_write 收到 intermediates 和 `denoising_num_steps == 10` |
| 写入→读取 round-trip | InMemoryBackend: insert → fetch_payload → 验证 intermediates 一致 |
| 向后兼容 | 旧 HDF5 (无 noise_action) 构建 artifact → intermediates=None |
| FULL_HIT 路径 | buffer_for_write 的 intermediates=None，不影响现有行为 |

---

## Phase 2: 命中判定 + 执行链路

**目标**：实现完整的三级判定（FULL_HIT / WARM_START / MISS）和 WARM_START 执行路径。Judge 决定命中类型和 warm start 的 `start_t`（通过 `warm_tiers` 配置），Orchestrator 负责 payload 完整性校验和语义降级，Interceptor 只执行。

### P2-1: HitType 枚举放开 + JudgeResult 返回类型

文件：`src/openpi/cache/components/judge.py`

1. 取消 `WARM_START = auto()` 的注释：

```python
class HitType(Enum):
    MISS = auto()
    FULL_HIT = auto()
    WARM_START = auto()
```

2. 新增 `JudgeResult` 数据类：

```python
@dataclass
class JudgeResult:
    hit_type: HitType
    winner_id: str | None = None
    start_t: float | None = None   # Judge 决定的 start_t，仅 WARM_START 时有值
```

3. `SimilarityJudge` Protocol 返回类型从 `tuple[HitType, Optional[str]]` 改为 `JudgeResult`：

```python
class SimilarityJudge(Protocol):
    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> JudgeResult:
        ...
```

4. `src/openpi/cache/__init__.py` 同步导出 `JudgeResult`（现有已导出 `SimilarityJudge`、`ThresholdJudge`、`HitType`）。

5. `AlwaysHitJudge` 同步修改返回值：

```python
def __call__(self, results, checkpoint_id, cached_data) -> JudgeResult:
    if not results:
        return JudgeResult(HitType.MISS)
    return JudgeResult(HitType.FULL_HIT, results[0].id)
```

### P2-2: Orchestrator `check()` 处理 WARM_START + payload 完整性校验

文件：`src/openpi/cache/orchestrator.py`

#### CheckResult 增加 start_t 字段

```python
@dataclass
class CheckResult:
    hit_type: HitType
    payload: Optional[CachePayload] = None  # non-None on FULL_HIT or WARM_START
    score: Optional[float] = None
    entry_id: Optional[str] = None
    query_keys: Optional[dict[str, torch.Tensor]] = None
    start_t: float | None = None            # Judge 决定的 start_t，透传给 Interceptor
```

#### check() 方法修改

Judge 调用方式变更（解包 JudgeResult）：

```python
judge_result = judge(results, checkpoint_id, self._key_builder.cached_data)
hit_type = judge_result.hit_type
winner_id = judge_result.winner_id
start_t = judge_result.start_t
```

FULL_HIT 和 WARM_START 都需要 fetch payload：

```python
if hit_type in (HitType.FULL_HIT, HitType.WARM_START) and winner_id is not None:
    with self._timer.measure(f"{prefix}_fetch"):
        payload = self._storage.fetch_payload(winner_id)

    # WARM_START payload 完整性校验（语义降级在此处完成）
    if hit_type == HitType.WARM_START:
        if (not payload.intermediates
                or payload.denoising_num_steps is None
                or start_t not in payload.intermediates):
            logger.debug(
                "[step %d] WARM_START payload incomplete (start_t=%s, "
                "has_intermediates=%s), downgrade to MISS.",
                self._step_counter, start_t,
                payload.intermediates is not None,
            )
            self._miss_by_checkpoint[checkpoint_id] = (
                self._miss_by_checkpoint.get(checkpoint_id, 0) + 1
            )
            return CheckResult(
                hit_type=HitType.MISS, query_keys=query_keys,
                score=results[0].score, entry_id=winner_id,
            )

    return CheckResult(
        hit_type=hit_type, payload=payload, start_t=start_t,
        score=results[0].score, entry_id=winner_id, query_keys=query_keys,
    )

return CheckResult(hit_type=HitType.MISS, query_keys=query_keys)
```

Interceptor 收到 `WARM_START` 时，可以信任 `payload.intermediates`、`payload.denoising_num_steps`、`start_t` 全部有效。

### P2-3: Interceptor WARM_START 分支 (E1)

文件：`src/openpi/cache/interceptor.py`

#### 完整控制流

```python
# CP1 check
if self._orchestrator is not None:
    ...
    cp1_result = self._orchestrator.check(CheckpointID.CP1, **cp1_kwargs)

    if cp1_result.hit_type == HitType.FULL_HIT:
        # early return（现有逻辑不变）
        ...
        return outputs

# --- 以下路径都需要 Stage 2（WARM_START 和 MISS 共用）---
with self._timer.measure("stage2_llm"):
    stage2 = self._stage2_fn(stage1)

# --- Stage 3 分支 ---
if (self._orchestrator is not None
        and cp1_result.hit_type == HitType.WARM_START):
    # WARM_START: 从缓存 x_t 开始跑部分 S3
    start_t = cp1_result.start_t
    start_x = cp1_result.payload.intermediates[start_t].to(self._pytorch_device)
    if start_x.ndim == 2:
        start_x = start_x[None, ...]

    with self._timer.measure("stage3_warm"):
        stage3 = self._model.run_stage3_from(
            stage2, start_x, start_t,
            num_steps=cp1_result.payload.denoising_num_steps,
        )
elif self._orchestrator is not None:
    # MISS (缓存模式): eager 调用，采集 intermediates
    with self._timer.measure("stage3_flow"):
        stage3 = self._model.run_stage3(
            stage2, noise=start_noise,
            num_steps=_NUM_STEPS, return_intermediates=True,
        )
else:
    # 无缓存模式: compiled 调用
    with self._timer.measure("stage3_flow"):
        stage3 = self._stage3_fn(stage2, noise=start_noise)

# --- 以下后处理完全共用（CP3 check, broadcast, buffer, output）---
```

关键设计点：
- **Stage 2 在 FULL_HIT early return 之后、Stage 3 之前统一执行，只跑一次**
- Stage 3 根据 `hit_type` 选择调用方式，`stage3` 变量在所有分支都有定义
- WARM_START 的 `buffer_for_write` 传 `intermediates=None`（`run_stage3_from` 不产出语义正确的 intermediates）
- MISS 的 `buffer_for_write` 传 `stage3.intermediates`（经 CPU 转换后）和 `denoising_num_steps=_NUM_STEPS`
- Interceptor 不做 payload 校验或语义降级（已由 Orchestrator 完成），仅保留 `assert` 级防御

### P2-4: Interceptor 时序 probe (E3)

文件：`src/openpi/cache/interceptor.py:173-175`

在 `__init__` 中注册 `stage3_warm` probe（仅当 orchestrator 不为 None）：

```python
if orchestrator is not None:
    self._timer.register_probe("cp1_sum", backend="cpu")
    self._timer.register_probe("cp3_sum", backend="cpu")
    self._timer.register_probe("stage3_warm", backend="cuda")
```

### P2-5: JudgeConfig 增加 warm_tiers (C1)

文件：`src/openpi/cache/config.py:87-89`

```python
@dataclass
class JudgeConfig:
    type: str = "threshold"
    threshold: float = 0.98
    warm_tiers: list[dict[str, float]] | None = None
```

`warm_tiers` 是一个从高阈值到低阈值排列的 tier 列表，每个 tier 包含 `threshold` 和 `start_t`。Judge 从上往下匹配第一个满足的 tier。

`warm_tiers=None`（默认）关闭 warm start，完全向后兼容。

配置示例：

```yaml
judge:
  type: threshold
  threshold: 0.98
  warm_tiers:
    - {threshold: 0.95, start_t: 0.3}   # 高相似 → 跳 70%
    - {threshold: 0.90, start_t: 0.5}   # 中等 → 跳 50%
    - {threshold: 0.85, start_t: 0.7}   # 较低 → 跳 30%
```

### P2-6: ThresholdJudge 多级阈值 (J2)

文件：`src/openpi/cache/components/judge.py`

ThresholdJudge 改为多级阈值判定。接受 `warm_tiers` 参数：

```python
class ThresholdJudge:
    def __init__(self, cp1_threshold, cp3_threshold,
                 warm_tiers: list[dict[str, float]] | None = None):
        self._thresholds = {
            CheckpointID.CP1: cp1_threshold,
            CheckpointID.CP3: cp3_threshold,
        }
        self._warm_tiers = warm_tiers or []

    def __call__(self, results, checkpoint_id, cached_data) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        top = results[0]
        threshold = self._thresholds.get(checkpoint_id, 0.98)
        if top.score >= threshold:
            return JudgeResult(HitType.FULL_HIT, top.id)
        if checkpoint_id == CheckpointID.CP1 and self._warm_tiers:
            for tier in self._warm_tiers:
                if top.score >= tier["threshold"]:
                    return JudgeResult(HitType.WARM_START, top.id, start_t=tier["start_t"])
        return JudgeResult(HitType.MISS)
```

WARM_START 只在 CP1 生效。`warm_tiers` 为空时行为与现在完全一致。

### P2-7: `_build_judge` 工厂更新

文件：`src/openpi/cache/config.py`

```python
def _build_judge(cfg: JudgeConfig):
    if cfg.type == "threshold":
        return ThresholdJudge(
            cp1_threshold=cfg.threshold,
            cp3_threshold=cfg.threshold,
            warm_tiers=cfg.warm_tiers,
        )
    elif cfg.type == "always_hit":
        return AlwaysHitJudge()
    ...
```

### P2-8: Config 校验更新

文件：`src/openpi/cache/config.py` 的 `validate_cache_config()`

新增校验：
- `warm_tiers` 非 None 时，每个 tier 必须有 `threshold` 和 `start_t`
- 第一个 tier 的 `threshold` 必须 < `judge.threshold`
- tier 间 `threshold` 必须严格递减
- tier 的 `start_t` 必须在 (0, 1) 开区间内，且为 `round(1.0 - i/num_steps, 4)` 形式的合法 timestep（即 0.1, 0.2, ..., 0.9）
- `warm_tiers` 非 None 时，`judge.type` 必须是 `threshold`（`always_hit` 不支持）
- CP3 的 `warm_tiers` 必须为 None 或空（CP3 不支持 warm start），违反时 fail-fast 报错
- `None` 和 `[]` 都表示关闭 warm start，语义等价
- `start_t` 校验时用规范化值比较：`canonical = {round(1.0 - i / 10, 4) for i in range(1, 10)}`，配置值也 `round(..., 4)` 后存回，避免 YAML 浮点表示差异导致 `start_t not in payload.intermediates` 误降级

### P2-T: 测试

| 测试 | 内容 |
|------|------|
| HitType | `HitType.WARM_START` 存在且可比较 |
| JudgeResult | 数据类字段正确，FULL_HIT/MISS 时 `start_t=None` |
| ThresholdJudge 多级 | score 落在不同 tier 区间时返回正确的 `start_t` |
| ThresholdJudge 向后兼容 | `warm_tiers=None` 时只返回 FULL_HIT/MISS，`start_t=None` |
| AlwaysHitJudge | 返回 `JudgeResult`，`start_t=None` |
| Orchestrator check() | WARM_START + payload 完整 → CheckResult 有 payload 和 start_t |
| Orchestrator 降级 | WARM_START + payload 缺 intermediates → 返回 MISS，miss 计数 +1 |
| Orchestrator 降级 | WARM_START + start_t 不在 intermediates keys → 返回 MISS |
| Interceptor 三路径 | FakeModel 扩展 `run_stage3(return_intermediates=True)` 和 `run_stage3_from()`；FULL_HIT 跳 S2+S3、WARM_START 跑 S2+`run_stage3_from`、MISS 跑 S2+完整 S3 |
| Config warm_tiers | 合法配置通过；tier threshold 未递减报错；start_t 非法值报错；CP3 设 warm_tiers 报错 |
| 离线/在线能力差异 | 离线 payload 含 t=0.1，warm_tiers 选 0.1 → WARM_START 成功；在线 payload 不含 0.1，同配置下 Orchestrator 降级 MISS 并保留 score/entry_id |
| 端到端 | FULL_HIT/WARM_START/MISS 三条路径各走一遍，输出格式正确 |

---

## Phase 3: 文档更新

**目标**：更新架构文档、教程和代码 docstring，反映 warm start 变更。

### P3-1: `docs/cache_system_architecture.md` (D1)

- 更新 CP1 命中行为：三级判定（FULL_HIT / WARM_START / MISS）
- 更新 CP2 章节：说明 warm start 已迁移至 CP1
- 更新时序图：增加 WARM_START 分支

### P3-2: `docs/cache_system_tutorial.md` (D2)

- 更新 Judge 配置说明：`warm_tiers` 多级阈值
- 增加 WARM_START YAML 配置示例

### P3-3: 代码 docstring 同步修正

以下代码 docstring 仍将 intermediates 描述为 CP2 warm start，需要修正为 CP1 warm start（英文）：

- `src/openpi/cache/storage_types.py:66-67`：`CP2 warm start` → `CP1 warm start`
- `src/openpi/cache/types.py`：`CheckpointID.CP2` docstring 中的 warm-start 描述
- `src/openpi/cache/interceptor.py` 模块 docstring：Step 4 / CP2 相关描述
- `src/openpi/cache/components/judge.py` 模块和 Protocol docstring：数据流 `-> (HitType, winner_id)` → `-> JudgeResult`
- `src/openpi/cache/orchestrator.py` 的 `CheckResult` 和 `check()` docstring：payload 语义更新

---

## 实现顺序与依赖关系

```
P0-1  (_stage3_with_intermediates 修复)
  │
  ├── P0-T (Phase 0 测试)
  │
  ▼
P1-1  (StepRecord)
P1-2  (buffer_for_write)       ← 这三个互不依赖，可并行
P1-3  (_build_entry_chain)
  │
  ├── P1-4  (Artifact Builder)   ← 依赖 P1-1 的 CachePayload 字段（已有）
  │
  ├── P1-5  (Interceptor 写入)   ← 依赖 P0-1 + P1-2
  │
  ├── P1-T  (Phase 1 测试)
  │
  ▼
P2-1  (HitType + JudgeResult)
P2-5  (JudgeConfig)             ← 这两个互不依赖
  │
  ├── P2-6  (ThresholdJudge)     ← 依赖 P2-1 + P2-5
  ├── P2-7  (_build_judge)       ← 依赖 P2-5 + P2-6
  ├── P2-8  (Config 校验)        ← 依赖 P2-5
  ├── P2-2  (Orchestrator)       ← 依赖 P2-1（JudgeResult 解包 + payload 校验）
  │
  ├── P2-3  (Interceptor 分支)   ← 依赖 P2-2
  ├── P2-4  (timing probe)       ← 依赖 P2-3
  │
  ├── P2-T  (Phase 2 测试)
  │
  ▼
P3-1 + P3-2 + P3-3  (文档 + docstring 更新)
```

---

## 改动文件清单

| 文件 | Phase | 改动点 |
|------|-------|--------|
| `src/openpi/models_pytorch/pi0_pytorch.py` | P0 | `_stage3_with_intermediates` 优化 |
| `src/openpi/cache/storage_types.py` | P1+P3 | StepRecord 增加 2 个字段；docstring 修正 CP2→CP1 |
| `src/openpi/cache/orchestrator.py` | P1+P2 | `buffer_for_write` 签名 + `_build_entry_chain` + `check()` JudgeResult 解包 + WARM_START payload 校验 + CheckResult.start_t |
| `src/openpi/cache/interceptor.py` | P1+P2+P3 | `_NUM_STEPS` 常量 + MISS 路径 intermediates 采集 + WARM_START 分支 + probe + docstring 修正 |
| `exp/build_in_memory_cache_artifact.py` | P1 | 读取全部 HDF5 noise_action_1..9，映射为 intermediates |
| `src/openpi/cache/components/judge.py` | P2 | HitType 枚举 + JudgeResult 数据类 + SimilarityJudge 返回类型 + ThresholdJudge 多级阈值 + AlwaysHitJudge 返回类型 |
| `src/openpi/cache/__init__.py` | P2 | 导出 JudgeResult |
| `src/openpi/cache/config.py` | P2 | JudgeConfig.warm_tiers + `_build_judge` + 校验（含 CP3 禁止 warm_tiers） |
| `src/openpi/cache/types.py` | P3 | CP2 docstring 修正 |
| `docs/cache_system_architecture.md` | P3 | CP1 三级判定、CP2 迁移说明、时序图 |
| `docs/cache_system_tutorial.md` | P3 | warm_tiers YAML 示例 |

---

## 风险清单

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| 跨推理 x_t 一致性（不同 past_key_values 下的 warm start） | 高 | 只在高相似度时触发 WARM_START；Judge 多级阈值控制激进程度；实验验证 |
| MISS 路径 Stage 3 从 compiled → eager 的性能回退 | 中 | 缓存命中率提升后 MISS 频率降低；监控 `stage3_flow` 延迟 |
| `_stage3_with_intermediates` 优化后 clone 位置错误 | 中 | monkeypatch denoise_step 的数值一致性测试 |
| warm_tiers 阈值选择不当导致低质量 warm start | 中 | 初期默认 `warm_tiers=None`（关闭）；需校准实验确定合适值 |
| WARM_START/FULL_HIT 写入的 entry 无 intermediates，日后被判为 WARM_START 时反复 fetch+降级 | 低 | Orchestrator 校验后降级为 MISS，正确性无影响；fetch 开销 backend-dependent（InMemory 很轻，Qdrant 可能更高）；降级日志用 `logger.debug` 限频（同一 entry_id 每 episode 告警一次） |

---

## YAML 配置示例

```yaml
# 启用 warm start（多级阈值）
checkpoints:
  cp1:
    judge:
      type: threshold
      threshold: 0.98                       # score >= 0.98 → FULL_HIT
      warm_tiers:
        - {threshold: 0.95, start_t: 0.3}  # 0.95 <= score < 0.98 → WARM_START, 跳 70%
        - {threshold: 0.90, start_t: 0.5}  # 0.90 <= score < 0.95 → WARM_START, 跳 50%
        - {threshold: 0.85, start_t: 0.7}  # 0.85 <= score < 0.90 → WARM_START, 跳 30%
                                            # score < 0.85 → MISS

# 关闭 warm start (默认/向后兼容)
checkpoints:
  cp1:
    judge:
      type: threshold
      threshold: 0.98
      # warm_tiers 不设置或为 null → 无 WARM_START
```

