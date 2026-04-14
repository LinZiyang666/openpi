# CP1 Warm Start 调查：Noise Action 可用性分析

> Status: In Progress
> Date: 2026-04-10
> Task: 将 warm start 从 CP2 迁移到 CP1 Judge 决策

---

## 1. 核心问题

CP2 因缺乏可用的检索 key 而挂起（Stage 2 只产生不透明的 `past_key_values`，无法用于向量检索）。新方案是：**用 CP1 的检索 key 来决定 warm start**，三级判定如下：

| 相似度区间 | 判定 | 行为 |
|---|---|---|
| 高 | FULL_HIT | 跳过 S2+S3，直接返回缓存 action |
| 中 | WARM_START | 正常跑 S2，S3 从缓存的 `x_t` 起步 |
| 低 | MISS | 全部正常执行 |

## 2. 调查项：缓存 Payload 中是否包含 Noise Action

### 2.1 数据模型层（支持）

`CachePayload`（`src/openpi/cache/storage_types.py:60-99`）已有预留字段：

```python
@dataclass
class CachePayload:
    action_chunk: torch.Tensor                          # [50, 32] CPU float32
    intermediates: Optional[dict[float, torch.Tensor]] = None   # {t: x_t}
    denoising_num_steps: Optional[int] = None
    task_key: str = ""
```

- `intermediates`: 设计用于存储 `{timestep: x_t}` 映射，如 `{0.7: tensor, 0.5: tensor, 0.3: tensor}`
- `denoising_num_steps`: 记录原始推理的总步数（如 10），用于计算 warm start 起始位置
- 文档注释中明确标注：`CP2 warm start : action_chunk + intermediates + denoising_num_steps required`

### 2.2 写入路径（未填充）

**Orchestrator `_build_entry_chain()`**（`src/openpi/cache/orchestrator.py:318-345`）：

```python
payload=CachePayload(
    action_chunk=step.action_chunk,
    task_key=record.task_key,
)
```

只写入 `action_chunk` 和 `task_key`，**`intermediates` 和 `denoising_num_steps` 未被填充**。

**`StepRecord`**（`src/openpi/cache/storage_types.py:291-300`）：

```python
@dataclass
class StepRecord:
    query_keys: dict[str, torch.Tensor]
    action_chunk: torch.Tensor
```

只有 `query_keys` 和 `action_chunk`，**没有 intermediates 字段**。

**`buffer_for_write()`**（`src/openpi/cache/orchestrator.py:269-282`）：

只接受 `query_keys` 和 `action_chunk` 两个参数。

### 2.3 Artifact Builder（未填充）

`exp/build_in_memory_cache_artifact.py:171`：

```python
payload = CachePayload(action_chunk=action, task_key=task)
```

只从 HDF5 读取 `clean_action`，**不读取 `noise_action_*`**。

### 2.4 InMemoryBackend（透传）

`src/openpi/cache/backends/in_memory_backend.py:67-74`：

```python
def insert(self, entry: CacheEntry) -> None:
    self._entries[entry.id] = entry

def fetch_payload(self, id: str) -> CachePayload:
    return self._entries[id].payload
```

InMemoryBackend 只做 dict 存取，**如果 payload 中有 intermediates，会原样存储和返回**。不存在后端层面的信息丢失。

### 2.5 QdrantBackend（支持序列化）

`src/openpi/cache/backends/qdrant_backend.py:406-434`：

已实现 `intermediates` 的 base64 序列化/反序列化（`_serialize_payload` / `_deserialize_payload`），但由于上游从未填充该字段，实际总是 `None`。

### 2.6 HDF5 数据源（有数据）

`src/openpi/collect/data_collector.py:89-90`：

```python
for i, noise_action in enumerate(embs.noise_action_steps, start=1):
    grp.create_dataset(f"noise_action_{i}", data=noise_action)
```

每步推理收集 9 个 noise action（`noise_action_1` 到 `noise_action_9`），对应 10 步 flow matching 中第 1 到第 9 步的中间状态 `x_t`。形状为 `(action_horizon, action_dim)`，如 `(10, 32)`。

### 2.7 Judge 层（未支持 WARM_START）

`src/openpi/cache/components/judge.py:27-36`：

```python
class HitType(Enum):
    MISS = auto()
    FULL_HIT = auto()
    # WARM_START = auto()  # Step 7: flow matching warm start
```

`WARM_START` 已作为注释预留，但未启用。现有 Judge（`AlwaysHitJudge`、`ThresholdJudge`）只返回 `MISS` 或 `FULL_HIT`。

### 2.8 Orchestrator `check()` 返回（未支持 WARM_START）

`src/openpi/cache/orchestrator.py:52-68`：

```python
@dataclass
class CheckResult:
    hit_type: HitType
    payload: Optional[CachePayload] = None  # non-None only on FULL_HIT
    ...
```

`check()` 方法（第 244-255 行）只在 `FULL_HIT` 时 fetch payload，其他情况返回 `HitType.MISS`。

---

## 3. 结论

| 层 | Noise Action 状态 | 说明 |
|---|---|---|
| 数据模型（CachePayload） | ✅ 字段已预留 | `intermediates` + `denoising_num_steps` |
| HDF5 数据源 | ✅ 有数据 | `noise_action_1..9`，每步推理 9 个中间态 |
| Artifact Builder | ❌ 未读取 | 只读 `clean_action`，不读 `noise_action_*` |
| StepRecord / buffer_for_write | ❌ 无字段 | 只有 `query_keys` + `action_chunk` |
| Orchestrator 写入路径 | ❌ 未填充 | `_build_entry_chain` 只写 `action_chunk` |
| InMemoryBackend 存取 | ✅ 透传 | dict 存取，不丢失任何字段 |
| QdrantBackend 序列化 | ✅ 已实现 | base64 round-trip 已写好 |
| Judge HitType | ❌ 未启用 | `WARM_START` 被注释掉 |
| Orchestrator check() | ❌ 未支持 | 只处理 `FULL_HIT` 路径 |
| Interceptor | ❌ 未支持 | 无 WARM_START 分支 |

**总结：底层数据模型和后端存储层已就绪，但从数据采集到写入缓存到命中判定到拦截器执行的整条链路都缺失 noise action 的处理。需要端到端补全。**

---

## 4. 已就绪的基础设施

以下组件已实现，可直接复用：

### 4.1 `run_stage3_from()` — 已实现

`src/openpi/models_pytorch/pi0_pytorch.py:573-622`

```python
def run_stage3_from(
    self,
    stage2: Stage2Output,
    start_x: torch.Tensor,
    start_t: float,
    *,
    num_steps: int = 10,
) -> Stage3Output:
```

从缓存的 `x_t` 和 `start_t` 开始跑剩余 flow matching 步骤。例如 `start_t=0.3, num_steps=10` → 只跑 3 步（省 70%）。已实现，未被调用。

### 4.2 `_stage3_with_intermediates()` — 已实现

`src/openpi/models_pytorch/pi0_pytorch.py:624-663`

当 `run_stage3(return_intermediates=True)` 时，在 `save_timesteps=(0.7, 0.5, 0.3)` 处克隆 `x_t`，返回 `{t: x_t}` 字典。已实现，但当前写入路径未使用该数据。

### 4.3 `Stage3Output.intermediates` — 已定义

`src/openpi/models_pytorch/pi0_pytorch.py:62-73`

```python
@dataclass
class Stage3Output:
    action_chunk: torch.Tensor
    intermediates: Optional[dict[float, torch.Tensor]] = None
```

### 4.4 `CachePayload.intermediates` — 已定义

`src/openpi/cache/storage_types.py:88-89`

```python
intermediates: Optional[dict[float, torch.Tensor]] = None   # {t: x_t}
denoising_num_steps: Optional[int] = None
```

### 4.5 QdrantBackend 序列化 — 已实现

`src/openpi/cache/backends/qdrant_backend.py:406-434`：`intermediates` 的 base64 序列化/反序列化已写好。

### 4.6 `HitType.WARM_START` — 已预留

`src/openpi/cache/components/judge.py:36`：`# WARM_START = auto()  # Step 7: flow matching warm start`

---

## 5. 需要修改的内容

为实现 CP1 warm start，需要修改以下模块。按数据流方向排列。

### 5.1 数据写入链路（把 noise action 存入缓存）

总览表：

| # | 模块 | 文件 | 现状 | 改动 |
|---|------|------|------|------|
| W1 | Artifact Builder | `exp/build_in_memory_cache_artifact.py` | 只读 `clean_action` | 读取 HDF5 中的 `noise_action_1..9`，映射到 `CachePayload.intermediates = {t: tensor}` |
| W2 | StepRecord | `src/openpi/cache/storage_types.py` | 只有 `query_keys` + `action_chunk` | 增加 `intermediates` 和 `denoising_num_steps` 两个 Optional 字段 |
| W3 | buffer_for_write() | `src/openpi/cache/orchestrator.py:269` | 只接受 `query_keys`, `action_chunk` | 新增 `intermediates` 和 `denoising_num_steps` 参数 |
| W4 | _build_entry_chain() | `src/openpi/cache/orchestrator.py:318` | `CachePayload(action_chunk=..., task_key=...)` | 填充 `intermediates` 和 `denoising_num_steps` |
| W5 | Interceptor 写入调用 | `src/openpi/cache/interceptor.py:369` | `buffer_for_write(query_keys, action_chunk)` | 传入 `stage3.intermediates` 和 `num_steps`（需要 `run_stage3(return_intermediates=True)`） |
| W6 | Interceptor run_stage3 调用 | `src/openpi/cache/interceptor.py:351` | `self._stage3_fn(stage2, noise=start_noise)` | 改为 `run_stage3(stage2, noise=start_noise, return_intermediates=True)` |

#### W1: Artifact Builder 详细改法（离线路径）

`exp/build_in_memory_cache_artifact.py:167-171`

现在：
```python
action = torch.from_numpy(np.array(group["clean_action"])).float()
if action.dim() == 1:
    action = action.unsqueeze(0)
payload = CachePayload(action_chunk=action, task_key=task)
```

改为：
```python
action = torch.from_numpy(np.array(group["clean_action"])).float()
if action.dim() == 1:
    action = action.unsqueeze(0)

# 读取 noise action 中间态
intermediates = None
denoising_num_steps = None
noise_keys = sorted([k for k in group.keys() if k.startswith("noise_action_")])
if noise_keys:
    num_steps = len(noise_keys) + 1  # 9 个 noise action → 10 步
    denoising_num_steps = num_steps
    intermediates = {}
    for k in noise_keys:
        # noise_action_3 → i=3 → t = 1.0 - 3/10 = 0.7
        i = int(k.split("_")[-1])
        t = round(1.0 - i / num_steps, 4)
        intermediates[t] = torch.from_numpy(np.array(group[k])).float()

payload = CachePayload(
    action_chunk=action,
    task_key=task,
    intermediates=intermediates,
    denoising_num_steps=denoising_num_steps,
)
```

向后兼容：旧版 HDF5 没有 `noise_action_*` 字段时 `noise_keys` 为空，`intermediates=None`，行为不变。

#### W2 + W3: StepRecord / buffer_for_write 详细改法（在线路径）

`src/openpi/cache/storage_types.py:291-299`

现在：
```python
@dataclass
class StepRecord:
    query_keys: dict[str, torch.Tensor]
    action_chunk: torch.Tensor
```

改为：
```python
@dataclass
class StepRecord:
    query_keys: dict[str, torch.Tensor]
    action_chunk: torch.Tensor
    intermediates: Optional[dict[float, torch.Tensor]] = None
    denoising_num_steps: Optional[int] = None
```

`src/openpi/cache/orchestrator.py:269-282`

现在：
```python
def buffer_for_write(
    self,
    query_keys: dict[str, torch.Tensor],
    action_chunk: torch.Tensor,
) -> None:
    self._episode_steps.append(StepRecord(
        query_keys=query_keys,
        action_chunk=action_chunk,
    ))
```

改为：
```python
def buffer_for_write(
    self,
    query_keys: dict[str, torch.Tensor],
    action_chunk: torch.Tensor,
    intermediates: Optional[dict[float, torch.Tensor]] = None,
    denoising_num_steps: Optional[int] = None,
) -> None:
    self._episode_steps.append(StepRecord(
        query_keys=query_keys,
        action_chunk=action_chunk,
        intermediates=intermediates,
        denoising_num_steps=denoising_num_steps,
    ))
```

向后兼容：新参数全部 `Optional` 默认 `None`。所有现有调用方（FULL_HIT 路径的 `buffer_for_write(query_keys, cached_action)` 等）零改动即可继续工作。

#### W4: Orchestrator _build_entry_chain 详细改法

`src/openpi/cache/orchestrator.py:329-332`

现在：
```python
payload=CachePayload(
    action_chunk=step.action_chunk,
    task_key=record.task_key,
)
```

改为：
```python
payload=CachePayload(
    action_chunk=step.action_chunk,
    task_key=record.task_key,
    intermediates=step.intermediates,
    denoising_num_steps=step.denoising_num_steps,
)
```

向后兼容：`step.intermediates` 为 `None` 时（FULL_HIT 路径或未开启采集），`CachePayload` 的字段就是 `None`，与现在行为完全一致。`CachePayload.validate_for_checkpoint()` 已有校验：`intermediates` 非 None 时要求 `denoising_num_steps` 也非 None。

#### 写入链路兼容性总结

三个改动点的共同设计原则：**全部用 Optional 默认 None，不改任何现有接口的签名语义。**

| 场景 | intermediates 值 | 行为变化 |
|------|-----------------|---------|
| 未开启采集（现有默认） | `None` | 无变化 |
| FULL_HIT early return | `None`（没跑 Stage 3） | 无变化 |
| 正常推理 + 在线采集 | `{0.7: tensor, 0.5: tensor, 0.3: tensor}` | 新增：写入缓存 |
| 旧版 HDF5 artifact 无 noise_action | `None` | 无变化 |
| 新版 HDF5 artifact 有 noise_action | `{t: tensor, ...}` | 新增：写入缓存 |

### 5.2 命中判定链路（决定 WARM_START）

总览表：

| # | 模块 | 文件 | 现状 | 改动 | 本轮范围 |
|---|------|------|------|------|---------|
| J1 | HitType 枚举 | `src/openpi/cache/components/judge.py:36` | `WARM_START` 被注释 | 取消注释 | ✅ 本轮 |
| J2 | ThresholdJudge | `src/openpi/cache/components/judge.py:106` | 单阈值 | 双阈值 | ❌ 后续 |
| J3 | AlwaysHitJudge | `src/openpi/cache/components/judge.py:74` | 总返回 FULL_HIT | 可能需要 AlwaysWarmJudge | ❌ 后续 |
| J4 | Orchestrator check() | `src/openpi/cache/orchestrator.py:244` | 只在 FULL_HIT 时 fetch payload | WARM_START 时也 fetch | ✅ 本轮 |
| J5 | CheckResult | `src/openpi/cache/orchestrator.py:52` | payload 注释为 "non-None only on FULL_HIT" | FULL_HIT 或 WARM_START 时都非 None | ✅ 本轮 |

#### J1: HitType 枚举 — 取消注释

`src/openpi/cache/components/judge.py:34-36`

现在：
```python
class HitType(Enum):
    MISS = auto()
    FULL_HIT = auto()
    # WARM_START = auto()  # Step 7: flow matching warm start
```

改为：
```python
class HitType(Enum):
    MISS = auto()
    FULL_HIT = auto()
    WARM_START = auto()
```

这是一个接口层改动。枚举放开后，下游的 Judge 实现、Orchestrator、Interceptor 即可使用该值。具体哪个 Judge 实现会返回 `WARM_START`（双阈值 ThresholdJudge、AlwaysWarmJudge 等）属于组件实现，留到后续。

#### J4: Orchestrator check() — 处理 WARM_START

`src/openpi/cache/orchestrator.py:244-255`

现在：
```python
if hit_type == HitType.FULL_HIT and winner_id is not None:
    with self._timer.measure(f"{prefix}_fetch"):
        payload = self._storage.fetch_payload(winner_id)
    return CheckResult(
        hit_type=hit_type, payload=payload,
        score=results[0].score, entry_id=winner_id, query_keys=query_keys,
    )
return CheckResult(hit_type=HitType.MISS, query_keys=query_keys)
```

改为：
```python
if hit_type in (HitType.FULL_HIT, HitType.WARM_START) and winner_id is not None:
    with self._timer.measure(f"{prefix}_fetch"):
        payload = self._storage.fetch_payload(winner_id)
    return CheckResult(
        hit_type=hit_type, payload=payload,
        score=results[0].score, entry_id=winner_id, query_keys=query_keys,
    )
return CheckResult(hit_type=HitType.MISS, query_keys=query_keys)
```

WARM_START 需要 fetch payload 是因为 Interceptor 要从 `payload.intermediates` 取 `x_t` 作为 `run_stage3_from` 的输入。

#### J5: CheckResult 文档更新

`src/openpi/cache/orchestrator.py:64`

现在：
```python
payload: Optional[CachePayload] = None  # non-None only on FULL_HIT
```

改为：
```python
payload: Optional[CachePayload] = None  # non-None on FULL_HIT or WARM_START
```

#### J4/J5 的性质

J4 和 J5 也是接口层改动，不涉及具体的判定逻辑。J4 把 Orchestrator 的决策条件从 `== FULL_HIT` 扩展为 `in (FULL_HIT, WARM_START)`，让 WARM_START 也能拿到 payload；J5 更新注释。Orchestrator 本身不决定"怎么算 WARM_START"——它只看 Judge 返回什么，然后决定要不要 fetch payload 并打包成 CheckResult 返回给 Interceptor。

#### J2/J3 暂不实现的原因

具体的 Judge 组件实现（`ThresholdJudge` 双阈值、`AlwaysWarmJudge` 测试用 Judge）需要先确定阈值来源和校准策略。这些属于实验设计范畴，与基础设施改动（J1/J4/J5）解耦，留到后续单独处理。

### 5.3 执行链路（用 noise action 做 warm start 推理）

总览表：

| # | 模块 | 文件 | 现状 | 改动 |
|---|------|------|------|------|
| E1 | Interceptor infer() | `src/openpi/cache/interceptor.py:326` | CP1 FULL_HIT 时跳过 S2+S3 | 新增 `WARM_START` 分支：正常跑 S2，调用 `run_stage3_from(stage2, start_x=cached_x_t, start_t=t, num_steps=...)` 跑部分 S3 |
| E2 | Stage 函数编译 | `src/openpi/cache/interceptor.py:181` | 只编译 `run_stage1/2/3` | 需要额外引用 `run_stage3_from`（可能不需要 compile，因为 warm start 路径频率低） |
| E3 | Interceptor 时序计时 | `src/openpi/cache/interceptor.py` | `stage3_flow` 统一计时 | WARM_START 时可能需要单独的 probe（如 `stage3_warm`）区分完整 S3 和部分 S3 |

#### E1: Interceptor infer() — WARM_START 分支

这是整个 warm start 的最终执行点。前面所有改动（写入 intermediates、Judge 返回 WARM_START、Orchestrator fetch payload）都是为了让数据到达这里。

`src/openpi/cache/interceptor.py:326-352`

现在只有两条路：
```python
if cp1_result.hit_type == HitType.FULL_HIT:
    # 跳过 S2+S3，直接返回缓存 action
    ...
    return outputs

# 没有 WARM_START 分支 → 掉到下面当 MISS 处理

# MISS：正常跑完整 S2 + 完整 S3
stage2 = self._stage2_fn(stage1)
stage3 = self._stage3_fn(stage2, noise=start_noise)
```

如果 Judge 返回 `WARM_START`，当前代码会把它当 MISS 处理——正常跑完整 S2+S3，白白浪费了缓存的 `x_t`。

改为三条路：
```python
if cp1_result.hit_type == HitType.FULL_HIT:
    # 跳过 S2+S3，直接返回缓存 action（现有逻辑不变）
    ...
    return outputs

elif cp1_result.hit_type == HitType.WARM_START:
    # 正常跑 S2，从缓存 x_t 开始跑部分 S3
    stage2 = self._stage2_fn(stage1)
    intermediates = cp1_result.payload.intermediates
    start_t = min(intermediates.keys())  # 如 0.3，省 70%
    start_x = intermediates[start_t].to(self._pytorch_device)
    stage3 = self._model.run_stage3_from(
        stage2, start_x, start_t,
        num_steps=cp1_result.payload.denoising_num_steps,
    )
    # 后续和正常推理一样：CP3 check + broadcast + buffer + build outputs

# MISS：正常跑 S2 + 完整 S3（现有逻辑不变）
stage2 = self._stage2_fn(stage1)
stage3 = self._stage3_fn(stage2, noise=start_noise)
```

注意事项：
- `start_t` 的选择策略（用最小的 t 省最多计算 vs 用较大的 t 保留更多修正空间）可配置化，初期硬编码取 `min(intermediates.keys())` 即可
- `start_x` 需要 `.to(self._pytorch_device)` 转到 GPU（payload 中的 tensor 是 CPU float32）
- WARM_START 分支走完后，后续的 CP3 check、broadcast_action、buffer_for_write 逻辑和 MISS 路径一样，可以复用

#### E1 补充：各路径的 intermediates 写入策略

`run_stage3_from()` 不采集中间态（它从别的推理的 `x_t` 开始跑部分步骤，采集出来的中间态语义不正确）。因此 WARM_START 和 FULL_HIT 一样，`buffer_for_write` 传 `intermediates=None`。

| 路径 | 调用方式 | intermediates 传入 buffer_for_write | 说明 |
|------|---------|--------------------------------------|------|
| MISS | `run_stage3(return_intermediates=True)` | `stage3.intermediates`（非 None） | 唯一产出中间态的路径 |
| FULL_HIT | 未跑 S3 | `None`（默认值） | 直接用缓存 action，无 S3 产出 |
| WARM_START | `run_stage3_from()` | `None`（默认值） | `run_stage3_from` 不采集中间态 |

`buffer_for_write` 的 `intermediates` 和 `denoising_num_steps` 参数默认为 `None`，FULL_HIT 和 WARM_START 的调用方无需任何改动。

### 5.4 配置链路

| # | 模块 | 文件 | 现状 | 改动 |
|---|------|------|------|------|
| C1 | YAML config | `src/openpi/cache/config.py`（待确认） | Judge 配置只有 `type` + 单阈值 | 增加 `warm_threshold` 参数 |
| C2 | save_timesteps 配置 | — | `run_stage3` 中硬编码 `(0.7, 0.5, 0.3)` | 可能需要配置化；warm start 的 `start_t` 也需要决定用哪个 timestep |

### 5.5 架构文档

| # | 文件 | 改动 |
|---|------|------|
| D1 | `docs/cache_system_architecture.md` | 更新 CP1 命中行为（增加 WARM_START 模式），更新 CP2 章节说明 warm start 已迁移至 CP1，更新时序图 |
| D2 | `docs/cache_system_tutorial.md` | 更新 Judge 配置说明（双阈值），增加 WARM_START YAML 示例 |

---

## 6. timestep 映射关系

HDF5 中 `noise_action_i` 与 flow matching timestep 的对应关系（10 步 Euler ODE，`dt = -0.1`）：

| HDF5 字段 | flow matching step | timestep `t` | 含义 |
|---|---|---|---|
| (初始噪声) | 0 | 1.0 | 纯噪声 `x_1` |
| `noise_action_1` | 1 | 0.9 | 第 1 步后的 `x_0.9` |
| `noise_action_2` | 2 | 0.8 | 第 2 步后的 `x_0.8` |
| `noise_action_3` | 3 | 0.7 | 第 3 步后的 `x_0.7` |
| `noise_action_4` | 4 | 0.6 | |
| `noise_action_5` | 5 | 0.5 | |
| `noise_action_6` | 6 | 0.4 | |
| `noise_action_7` | 7 | 0.3 | |
| `noise_action_8` | 8 | 0.2 | |
| `noise_action_9` | 9 | 0.1 | 第 9 步后的 `x_0.1` |
| `clean_action` | 10 | 0.0 | 最终 clean action `x_0` |

`run_stage3` 的 `save_timesteps=(0.7, 0.5, 0.3)` 对应 `noise_action_3, 5, 7`。

**注意**：`noise_action_i` 是 `action_in_proj` hook 在第 `i` 步**输入**时捕获的张量（即 denoise_step 之前的 `x_t`），与 `_stage3_with_intermediates` 中 "snapshot before denoise_step" 的语义一致。

---

## 7. `_stage3_with_intermediates()` 性能分析

### 7.1 当前状态

该方法是死代码——从未被调用。`--collect` 的数据收集通过 forward hook 捕获 noise action，不经过此方法。Interceptor 调用 `run_stage3` 时默认 `return_intermediates=False`，走的是 `_stage3_action_expert()` 快速路径。

### 7.2 性能问题

当前实现在循环内有 GPU→CPU 同步，会严重破坏 `torch.compile` 的编译图：

```python
# 当前实现 (pi0_pytorch.py:651-656)
while timestep >= -dt / 2:
    t_val = timestep.item()           # ← GPU→CPU 同步，每次迭代触发
    for st in save_timesteps:         # ← Python 遍历 + 浮点比较
        if abs(t_val - st) < half_dt:
            intermediates[st] = x_t.clone()
            break
    ...
```

问题清单：

| 问题 | 影响 | 触发频率 |
|------|------|---------|
| `timestep.item()` GPU→CPU 同步 | pipeline stall，编译器 graph break | 每次迭代（10 次/推理） |
| `abs(t_val - st) < half_dt` 浮点比较 | 依赖 CPU 值的分支，不可 trace | 每次迭代 |
| `intermediates[st] = x_t.clone()` dict 动态写入 | 编译器无法优化的 side effect | 3 次/推理 |

其中 `.item()` 是主要杀手：每次调用都强制 CPU 等待 GPU 完成当前所有未完成的计算。10 步循环 = 10 次同步 = 10 次 pipeline stall。

`x_t.clone()` 本身开销可忽略——tensor 很小（`[1, 10, 32]` = 1.25KB），且只在 3 个 timestep 处触发。

### 7.3 优化方案

**核心思路**：`save_timesteps` 和 `num_steps` 在循环开始前就是已知的 Python 常量。哪些步需要保存，可以预计算成一个 `set[int]`，循环内只做 `int in set` 检查（纯 Python，零 GPU 开销）。

```python
# 优化后
def _stage3_with_intermediates(self, state, prefix_pad_masks, 
                                past_key_values, noise, num_steps, save_timesteps):
    device = state.device
    bsize = state.shape[0]
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)

    # 预计算保存点 (纯 Python，循环外一次性)
    # step 0: t=1.0,  step 3: t=0.7,  step 5: t=0.5,  step 7: t=0.3
    save_at = {}
    for st in save_timesteps:
        step_idx = round((1.0 - st) * num_steps)
        save_at[step_idx] = st

    x_t = noise
    timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
    intermediates: dict[float, torch.Tensor] = {}

    step = 0
    while timestep >= -dt / 2:
        if step in save_at:
            intermediates[save_at[step]] = x_t.clone()

        expanded_time = timestep.expand(bsize)
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        timestep += dt
        step += 1

    return x_t, intermediates
```

改动对比：

| 项目 | 改前 | 改后 |
|------|------|------|
| 保存点判断 | `timestep.item()` + 浮点比较 | `int in dict`（纯 Python） |
| GPU→CPU 同步 | 每次迭代 1 次（共 10 次） | 0 次 |
| 编译图 break | 每次迭代 break | 无 break（`int in dict` 和 `x_t.clone()` 对编译器透明） |
| clone 调用 | 不变（3 次） | 不变（3 次） |
| denoise_step 调用 | 不变（10 次） | 不变（10 次） |
| 接口签名 | 不变 | 不变 |

### 7.4 不改什么

- `_stage3_action_expert()`：`return_intermediates=False` 的快速路径，完全不动
- `run_stage3()` 的分支结构：`False` 走原版，`True` 走优化后版本
- `run_stage3_from()`：warm start 执行路径，不涉及中间态采集
- `denoise_step()`：核心计算逻辑不变

### 7.5 与在线/离线两条数据路径的关系

中间态采集有两条独立路径：

| 路径 | 采集方式 | 使用场景 | 是否受此优化影响 |
|------|---------|---------|----------------|
| 离线 | `--collect` forward hook 从 `action_in_proj` 捕获 | HDF5 数据收集 → artifact builder 读入 | 否（hook 机制独立） |
| 在线 | `run_stage3(return_intermediates=True)` | Interceptor 实时采集 → buffer_for_write | 是（此优化的目标） |

离线路径已经在用（HDF5 中有 `noise_action_1..9`）。在线路径优化后可启用，使 Interceptor 在每次正常推理时同时采集中间态，无需依赖 `--collect`。

---

## 8. 在线采集可行性：集成到现有 trajectory 记录流程

### 8.1 现有 trajectory 记录数据流

```
Interceptor.infer()
  │
  ├─ run_stage3(stage2, noise=...)               → Stage3Output (只有 action_chunk)
  │
  ├─ action_chunk_cpu = stage3.action_chunk[0].detach().cpu().float().contiguous()
  │
  └─ orchestrator.buffer_for_write(query_keys, action_chunk_cpu)
       │
       └─ StepRecord(query_keys=..., action_chunk=...)   → 暂存到 _episode_steps[]
            │
            └─ on_episode_end()
                 └─ _build_entry_chain()
                      └─ CachePayload(action_chunk=..., task_key=...)  → batch_insert
```

### 8.2 结论：可行，改动很小

不需要新增类、新增流程、新增回调。沿着已有管道多带一份数据即可：

| 层 | 文件:行 | 现在 | 改为 |
|---|---|---|---|
| run_stage3 调用 | `interceptor.py:352` | `self._stage3_fn(stage2, noise=start_noise)` | 加 `return_intermediates=True` |
| intermediates 取出 | `interceptor.py:364` 附近 | 只取 `stage3.action_chunk` | 同时取 `stage3.intermediates`，逐 tensor `[0].cpu().float().contiguous()` |
| buffer_for_write | `interceptor.py:369` → `orchestrator.py:269` | `(query_keys, action_chunk)` | 加 `intermediates=...`, `denoising_num_steps=10` |
| StepRecord | `storage_types.py:291` | `query_keys` + `action_chunk` | 加 `intermediates: Optional[dict[float, Tensor]] = None` + `denoising_num_steps: Optional[int] = None` |
| _build_entry_chain | `orchestrator.py:329` | `CachePayload(action_chunk=..., task_key=...)` | 把 `step.intermediates` 和 `step.denoising_num_steps` 填入 |

### 8.3 FULL_HIT early return 路径

CP1 FULL_HIT 时走 early return（`interceptor.py:326-346`），没有跑 Stage 3，没有 intermediates。该路径的 `buffer_for_write` 传 `intermediates=None` 即可，与现有行为一致，StepRecord 的新字段默认就是 `None`。

### 8.4 前置依赖

**必须先完成第 7.3 节的 `_stage3_with_intermediates` 优化**。原因：

Interceptor 第 352 行的 `self._stage3_fn` 是 `torch.compile` 编译后的版本。`run_stage3(return_intermediates=True)` 会进入 `_stage3_with_intermediates` 分支。如果该分支内仍有 `.item()` GPU→CPU 同步，会导致：
- 编译图碎片化（graph break）
- 每次推理 10 次 pipeline stall
- 编译版本退化为接近 eager 的性能

优化顺序：**先修 7.3（消除 `.item()`）→ 再开启 `return_intermediates=True` → 再扩展 trajectory 写入路径**。

### 8.5 存储开销评估

每步推理额外存储 3 个中间态（`save_timesteps=(0.7, 0.5, 0.3)`）：

| 数据 | 形状 | 大小 | 每步数量 |
|------|------|------|---------|
| `action_chunk` (现有) | `[10, 32]` | 1.25 KB | 1 |
| `intermediates[t]` (新增) | `[10, 32]` | 1.25 KB | 3 |

每步新增 3.75 KB，一个 300 步的 episode 新增约 1.1 MB。InMemoryBackend 的内存开销和 QdrantBackend 的存储开销都可接受。

---

## 9. 关键技术风险

**跨推理 `x_t` 一致性**：warm start 使用的 `x_t` 来自另一次推理（不同的 `past_key_values`），用当前推理的 `past_key_values` 继续去噪。如果场景差异过大，剩余步骤的修正能力不足。

**缓解措施**：
- 初期只使用较高的 `t` 值（如 0.5 或 0.7），保留更多修正步骤
- warm start 阈值设置高于 MISS 但低于 FULL_HIT，确保只在高相似度场景触发
- 实验验证不同 `t` 值对任务成功率的影响
