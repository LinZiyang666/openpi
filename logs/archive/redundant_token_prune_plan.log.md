# 冗余 Token 剪枝方案设计 — Plan A / Plan B

> **Status**: `Plan`
> **前置讨论**: [redundant_token_prune_gpt.log.md](redundant_token_prune_gpt.log.md)（GPT 方案原始设计）
> **关联文档**: [key_dim_reduction_recommendations.log.md](key_dim_reduction_recommendations.log.md)（降维流水线）
> **关联架构**: [cache_system_tutorial.md](../docs/cache_system_tutorial.md) §4 KeyBuilder

---

## 1. 背景与目标

当前 KeyBuilder 对 vision token 的处理是"全量池化"：把 [256, 2048] 直接 mean/max/spatial pool 成一个向量。这种做法没有区分哪些 token 包含动作相关信息、哪些是静态背景冗余。

**目标**：在池化之前增加一个 token 剪枝阶段，去除冗余 token，使最终 cache key 更聚焦于任务相关的视觉区域。

两套方案：
- **Plan A (hard prune)**: 按时间变化度 → 删除静态 token → 按任务相关性 → 选 top K → 池化成 key
- **Plan B (compress then select)**: 按时间变化度 → 分成 static/dynamic → static 做 greedy merge 成 prototype → dynamic 选 top M → 合并 → 池化成 key

**实施顺序**: 先做 A，B 复用 A 的经验和部分实现。

---

## 2. 设计决策记录

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| D1 | 离线 vs 在线 | 都做，离线先写，在线后写 | 同一套 KeyBuilder 规则：离线预计算 artifact，在线推理时构建 key |
| D2 | 窗口化 vs 单帧 | 做窗口化，window_size 作为 KeyBuilder 参数 | 参数名需与 SearchStrategy 的 step_window 区分（后者是数据库查询参数） |
| D3 | A 和 B 同时还是分步 | 先做 A，B 在 A 之后做 | A 更简单适合做 baseline，B 可以复用 A 的 temporal scoring 等实现 |
| D4 | KeyBuilder 内部结构 | 显式分为两步：(1) 剪枝 (2) 降维/池化 | 第二步先写接口不着急实现，先用简单的 weighted mean pool |

---

## 3. 整体流水线定位

冗余 token 剪枝插入在现有降维流水线的**最前面**，形成三层结构：

```
原始 vision token         第 0 层：冗余剪枝            第 1 层：Token 池化        第 2 层：维度投影
[256, 2048]  ──────►  [K, 2048] (K<256)  ──────►  [2048]  ──────►  [D] (可选, 暂不实现)
                      ↑ 本次实现                   ↑ weighted mean pool      ↑ 接口预留
```

---

## 4. 影响面分析

### 4.1 需要新增/修改的文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `src/openpi/cache/components/key_builder.py` | **修改（核心）** | 新增 `CP1TemporalPruneKeyBuilder`（Plan A）类，需要：(1) 历史缓冲区存储过去帧的 vision token (2) temporal scoring 逻辑 (3) 两步式 `_temporal_prune()` → `reducer.reduce()` 结构 |
| `src/openpi/cache/config.py` | **修改** | (1) 在 `KeyBuilderConfig` 中增加 `prune_window_size` 等参数 (2) `_build_key_builder()` 增加新类型分支 (3) `validate_cache_config()` 增加新类型的校验规则 (4) `_valid_key_builder_types` 集合增加新类型 |
| `exp/build_in_memory_cache_artifact.py` | **修改** | (1) `_get_vector_dims()` 支持动态维度（从 reducer.output_dim 获取） (2) `_create_builder()` 增加新类型 (3) `_process_episode()` 改为支持窗口化：不再逐步独立处理，需要在步之间传递历史 token (4) CLI choices 和 main() 调用链扩展 |
| `tests/cache/components/test_temporal_prune.py` | **新增** | temporal prune + reducer 单元测试（独立文件） |

### 4.2 需要适配但不改逻辑的文件

| 文件 | 影响 | 说明 |
|------|------|------|
| `src/openpi/cache/orchestrator.py` | **协议适配** | 当前 `on_episode_start()` 会广播到 gate/judge/strategy，但不会广播到 key_builder。新的有状态 KeyBuilder 需要在 episode 开始时重置历史缓冲区。需要在 orchestrator 的 `on_episode_start()` 中增加对 key_builder 的调用。 |
| `src/openpi/cache/__init__.py` | **导出** | 新增类加入导出列表 |

### 4.3 不需要修改的文件

| 文件 | 理由 |
|------|------|
| `interceptor.py` | 不需要改 — interceptor 调用 orchestrator，orchestrator 调用 key_builder。只要 orchestrator 正确传递生命周期事件即可 |
| `cache_storage.py` / `backends/` | 不需要改 — 剪枝后的 key 仍然是 `{field: [dim]}` 格式，维度不变（仍然是 [2048]） |
| `search_strategy.py` | 不需要改 — SearchStrategy 的 `step_window` 是数据库侧的查询窗口参数，与 KeyBuilder 的 `prune_window_size` 完全独立 |
| `gate.py` / `judge.py` | 不需要改 — 它们消费的是 `cached_data` 和 `query_keys`，格式不变 |
| `storage_types.py` / `types.py` | 不需要改 — 字段名不变（仍然是 vision_0/vision_1/etc），只是值的内容不同 |

### 4.4 关键接口约束

**QueryKeyBuilder Protocol**（不可破坏）：
```python
class QueryKeyBuilder(Protocol):
    def collect(checkpoint_id, **stage_outputs) -> None   # 收集 GPU tensor
    def build(checkpoint_id) -> dict[str, Tensor]          # 输出 {field: [dim]} CPU float32
    def cached_data -> dict[str, Tensor]                   # Gate/Judge 读取
    def clear() -> None                                     # 每个推理周期结束释放
```

新增的生命周期方法（可选协议扩展）：
```python
def on_episode_start() -> None   # 重置历史缓冲区（有状态 KeyBuilder 需要）
```

---

## 5. KeyBuilder 两步内部架构

> **模态范围**: Step 1 和 Step 2 **仅作用于 vision 模态**（vision_0 / vision_1 / vision_2），每个摄像头独立走一遍 prune → reduce 流程。`prompt_emb` 和 `robot_state` **不经过此流水线**，沿用现有 `_CP1BaseKeyBuilder` 的处理方式原样传递（prompt_emb → mean pool → [2048]，robot_state → 原样 [32]）。

### 5.1 Step 1: Token Pruning（去除冗余）

**职责**: 去除跨时间步几乎不变的静态背景 vision token。只判断"是否冗余"，不判断"是否与任务相关"。
**每个摄像头独立处理**。

```
输入: tokens [W, 256, 2048]  (W 帧窗口的 vision tokens)

Plan A 流程:
  1. temporal scoring: 对每个 token 位置 i 算 temporal_score(i) = 平均相邻帧 cosine change
  2. temporal pruning: 按 temporal_score 保留变化最大的 top keep_ratio 个 (如 50% → 128 个)

输出: PruneResult:
  - tokens [W, K, 2048]       # 去除冗余后的 token (K = 256 * keep_ratio)
  - token_indices [K]          # 原始 256 中的位置索引 (SpatialPoolReducer 需要用来填回网格)
  - pruned: bool               # 是否执行了剪枝 (窗口不满时为 False)
  - temporal_scores [K]        # 每个保留 token 的时间变化度 (可选, 供 Step 2 参考)

Plan B 额外流程 (后续实现):
  2b. 分成 static set 和 dynamic set (而非直接删除)
  3b. static token 做 greedy merge → N 个 background prototypes
  4b. 输出: dynamic tokens + background prototypes
```

> **边界澄清**: task scoring / task-based selection 不属于 Step 1。Step 1 只回答"这个 token 是不是冗余的"，不涉及任务语义。任务相关性筛选是 Step 2 (TokenReducer) 内部的一种可选降维策略。

### 5.1.1 退化逻辑

当历史缓冲区帧数 < `prune_window_size` 时（episode 初始阶段），temporal scoring 不可用：

| 帧数 | 行为 | PruneResult |
|------|------|-------------|
| **1 帧** (第 0 步) | **跳过 Step 1**，全部 256 token 直接传给 Step 2 | `pruned=False`, `tokens=[1,256,2048]`, `token_indices=[0..255]` |
| **2 ~ W-1 帧** | **跳过 Step 1**，全部 256 token 直接传给 Step 2 | `pruned=False`, `tokens=[len,256,2048]`, `token_indices=[0..255]` |
| **W 帧（满窗口）** | **正常 Step 1**，temporal pruning | `pruned=True`, `tokens=[W,K,2048]`, `token_indices=[保留位置]` |

Step 2 的 reducer 通过 `pruned` 标记感知状态。所有 reducer 必须保证：**无论 `pruned` 为 True 还是 False，输出的 key 维度一致。** 这是 reducer 的接口契约。

### 5.2 Step 2: Token Reduction（降维，频繁实验区域）

**职责**: 把去冗余后的 token 集合压缩成最终的固定维度 key 向量。
**设计原则**: 这一步会频繁尝试不同方法，接口必须足够灵活。

```python
class TokenReducer(Protocol):
    """Reduce pruned token set to fixed-dim key vector(s).

    This is the experimentation hotspot — implementations will be
    swapped frequently. Keep the interface minimal and stable.
    """

    def reduce(self, prune_result: PruneResult, *,
               prompt_emb: torch.Tensor | None = None) -> torch.Tensor:
        """PruneResult → [output_dim] GPU tensor. KeyBuilder 负责 D2H 转换。"""
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of the output key vector."""
        ...
```

**可插拔实现**:

| 实现 | 参数 | 说明 | 输出维度 |
|------|------|------|----------|
| `MeanPoolReducer` | — | 时间+token 全平均，最简单 baseline | [2048] |
| `MaxPoolReducer` | — | 时间平均后 token 维 per-dim max | [2048] |
| `SpatialPoolReducer` | `output_tokens` | 填回 16x16 网格 → adaptive avg pool → flatten (见下方详细设计) | [output_tokens * 2048] |
| `TaskScoringReducer` | `select_k`, `temperature` | 用 cos(token, prompt_emb) 选 top-K 再加权池化 (GPT 原方案) | [2048] |
| `ProjectionReducer` | `output_dim` | 线性投影到低维 (未来) | [D] |
| `DualKeyReducer` | — | background + dynamic 分开池化拼接 (Plan B, 未来) | [4096] |

#### SpatialPoolReducer 详细设计

**参数**: `output_tokens`（如 16、4）。`pool_size = int(sqrt(output_tokens))`。

**统一流程**（无论是否剪枝）:

```
1. 时间平均: tokens [W, K, 2048] → mean over time → [K, 2048]
2. 用 token_indices 填回 16x16 网格:
     grid = zeros(16, 16, 2048)
     grid[indices] = token_values       # 缺失位置保持零
3. adaptive_avg_pool2d(grid, (pool_size, pool_size))  → [pool_size, pool_size, 2048]
4. flatten → [output_tokens * 2048]
```

- `pruned=False`: token_indices = [0..255]，网格完整，无零填充
- `pruned=True`: token_indices 是保留位置的子集，被剪掉的位置填零
- 只需一个实现，`output_tokens` 可配，取代原来 SpatialPool16 / SpatialPool64 两个类

`TaskScoringReducer` 的内部流程即 GPT 方案的 A-3 ~ A-5：

```
输入: PruneResult.tokens [W, K, 2048], prompt_emb [num_tokens, 2048]

1. 对每个 token 位置做时间平均: v_i = mean_t(token_i)        → [K, 2048]
2. 算 task_score(i) = cos(v_i, mean(prompt_emb))             → [K]
3. 选 top select_k 个                                         → [select_k, 2048]
4. weights = softmax(task_scores / temperature)
5. key = sum(weights * selected_tokens)                        → [2048]
```

> **适用场景**: 混合任务数据库（库中包含不同任务的 trajectory）时，task scoring 可以帮助区分"动态且相关" vs "动态但无关"的 token。在单任务数据库下 prompt_emb 区分度低，建议使用 `MeanPoolReducer` 或其他不依赖 prompt_emb 的 reducer。

**第一版实现**: `MeanPoolReducer` + `MaxPoolReducer` + `SpatialPoolReducer` + `TaskScoringReducer`。都是可插拔部件，实现成本低。

---

## 6. 窗口化机制设计

### 6.1 在线推理：历史缓冲区

KeyBuilder 内部维护一个 FIFO 缓冲区，存储过去 W 帧的 vision token：

```python
class _VisionHistoryBuffer:
    """Per-image history buffer for windowed temporal scoring."""
    
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._buffer: list[Tensor] = []   # 每项 [256, 2048]
    
    def push(self, tokens: Tensor) -> None:
        self._buffer.append(tokens)
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)
    
    def get_window(self) -> Tensor:
        # → [len, 256, 2048]，len <= window_size
        return torch.stack(self._buffer)
    
    def reset(self) -> None:
        self._buffer.clear()
```

**退化行为**: 窗口帧数 < `prune_window_size` 时完全跳过 Step 1（temporal pruning），全部 256 token 直接传给 Step 2（见 §5.1.1）。不做 partial-window scoring。

**生命周期**: 
- `on_episode_start()` → 重置缓冲区
- `collect()` → push 当前帧到缓冲区
- `build()` → 从缓冲区取窗口做剪枝

### 6.2 离线 artifact 构建

`_process_episode()` 当前逐步独立处理。需要改为：
- 维护一个 builder 实例跨步使用（每步仍调用 `clear()` 但只清 per-cycle cache，历史缓冲区保留）
- builder 内部的历史缓冲区在 `collect()` 时自然积累窗口
- episode 开始前调用 `on_episode_start()` 重置历史缓冲区

### 6.3 参数命名

| 参数 | 归属 | 含义 |
|------|------|------|
| `prune_window_size` | KeyBuilderConfig | token 剪枝的时间窗口帧数 (如 4) |
| `step_window` | SearchStrategyConfig | 数据库查询的步范围过滤 (如 ±5) |

两者完全独立，不会混淆。

---

## 7. 配置示例 (YAML)

```yaml
key_builder:
  type: cp1_temporal_prune        # Plan A
  prune_window_size: 4            # Step 1: 时间窗口 (默认 4)
  temporal_keep_ratio: 0.5        # Step 1: 保留变化最大的 50% (默认 128/256)
  reducer:                        # Step 2: 降维策略 (可替换)
    type: mean_pool               #   最简单 baseline
    # --- 或者 ---
    # type: task_scoring           #   GPT 原方案 (适合混合任务数据库)
    # select_k: 32                 #   从剩余 token 中按 task score 选 32 个
    # temperature: 1.0             #   softmax 温度
```

所有超参数均可通过 YAML 调整，默认值跟随 GPT 建议。

---

## 8. 数据可用性确认

| 场景 | vision token [256, 2048] | prompt_emb | 窗口化可行性 |
|------|--------------------------|------------|-------------|
| HDF5 数据集 | float16 逐步保存 | float16 逐步保存 | 完全可行：顺序读取 |
| 在线推理 | Stage1Output.prefix_embs 中切取 | prefix_embs 中切取 | 需要历史缓冲区 |
| 现有 artifact | 已被 pool，原始 token 丢失 | 已被 pool | **不可用** — 需要重建 |

---

## 9. Orchestrator 生命周期适配

当前 `on_episode_start()` 广播路径：

```
orchestrator.on_episode_start()
  → strategy.on_episode_start()     ✅ 已有
  → gate.on_episode_start()         ✅ 已有
  → judge.on_episode_start()        ✅ 已有
  → key_builder.on_episode_start()  ❌ 缺失 — 需要新增
```

改动方案：在 `orchestrator.on_episode_start()` 中检查 key_builder 是否有 `on_episode_start` 方法，如有则调用（duck typing，向后兼容）。

---

## 10. 已确认 & 待确认问题

### 已确认

| # | 问题 | 决策 |
|---|------|------|
| Q1 | 默认超参数 | 跟随 GPT 建议 (W=4, keep=50%, select=32)，但所有参数可配置 |
| Q2 | 多摄像头处理 | 每个摄像头 (vision_0/1/2) **独立剪枝** |
| Q3 | Step 2 降维 | 作为独立的 `TokenReducer` 接口，可插拔实现，方便频繁实验 |
| Q4 | Step 1/2 职责边界 | Step 1 只做 temporal pruning（去冗余），task scoring 属于 Step 2 (TokenReducer 实现) |
| Q5 | 模态范围 | Step 1+2 仅作用于 vision 模态，prompt_emb 和 robot_state 原样传递 |
| Q6 | 退化模式 | 窗口帧数 < prune_window_size 时跳过 Step 1，全 256 token 传给 Step 2，PruneResult.pruned=False |
| Q7 | SpatialPoolReducer | 统一为 `output_tokens` 参数，用 token_indices 填回 16x16 网格再 adaptive pool |
| Q8 | 第一版 reducer | MeanPool + MaxPool + SpatialPool + TaskScoring 四个全部实现 |

### 待确认

1. **prompt_emb 如何用于 task scoring**：直接 mean pool 后 cosine（GPT 方案），还是需要投影层？第一版建议直接 cosine。

---

## 11. 代码级实施计划

### 11.1 新增数据结构

**文件**: `src/openpi/cache/components/token_reducer.py`（新增文件，与 TokenReducer Protocol 同文件）

> `PruneResult` 放在 `token_reducer.py` 而非 `key_builder.py`，避免循环导入：
> `key_builder.py` 导入 `TokenReducer` 和 `PruneResult` ← `token_reducer.py`（单向依赖）。

```python
@dataclass
class PruneResult:
    """Output of Step 1 (token pruning), input to Step 2 (token reduction).

    Decouples pruning logic from reduction logic. All reducer implementations
    receive this uniform structure regardless of which pruner produced it.
    """
    tokens: torch.Tensor          # [W, K, emb_dim] — pruned token set
    token_indices: torch.Tensor   # [K] int64 — positions in original 256-token grid
    pruned: bool                  # False when window < prune_window_size (no pruning done)
    temporal_scores: torch.Tensor | None = None  # [K] optional, for Step 2 reference
```

### 11.2 TokenReducer Protocol + 4 个实现

**文件**: `src/openpi/cache/components/token_reducer.py`（新增文件）

这是独立文件，不放在 key_builder.py 里，因为 reducer 会频繁增减实现。

#### Protocol

```python
@runtime_checkable
class TokenReducer(Protocol):
    """Reduce pruned vision tokens to a fixed-dim key vector.

    Coupling:
      DEPENDS ON:  PruneResult (same file)
      CONSUMED BY: CP1TemporalPruneKeyBuilder.build()
      IF CHANGED:  output_dim must match backend vector_dims
    """

    def reduce(
        self,
        prune_result: PruneResult,
        *,
        prompt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """PruneResult → [output_dim] tensor (GPU, not yet transferred to CPU).

        prompt_emb: optional [num_tokens, emb_dim] for task-aware reduction.
        Reducers that don't use it simply ignore the parameter.
        """
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of the output key vector. Must be deterministic."""
        ...
```

#### MeanPoolReducer

```python
class MeanPoolReducer:
    """Mean over time and token dims. output_dim = emb_dim (2048)."""

    @property
    def output_dim(self) -> int:
        return _EMB_DIM  # 2048

    def reduce(self, prune_result: PruneResult, *,
               prompt_emb: torch.Tensor | None = None) -> torch.Tensor:
        # [W, K, D] → mean over W → [K, D] → mean over K → [D]
        return prune_result.tokens.mean(dim=0).mean(dim=0)
```

#### MaxPoolReducer

```python
class MaxPoolReducer:
    """Mean over time, then per-dim max over tokens. output_dim = emb_dim (2048)."""

    @property
    def output_dim(self) -> int:
        return _EMB_DIM

    def reduce(self, prune_result: PruneResult, *,
               prompt_emb: torch.Tensor | None = None) -> torch.Tensor:
        # [W, K, D] → mean over W → [K, D] → max over K → [D]
        return prune_result.tokens.mean(dim=0).max(dim=0).values
```

#### SpatialPoolReducer

```python
class SpatialPoolReducer:
    """Fill pruned tokens back into 16x16 grid, then adaptive avg pool.

    output_dim = output_tokens * emb_dim.
    output_tokens must be a perfect square (4, 16, 64, etc.).
    """

    _GRID_SIZE = 16  # sqrt(256), SigLIP patch grid

    def __init__(self, output_tokens: int = 16):
        self._output_tokens = output_tokens
        self._pool_size = int(output_tokens ** 0.5)
        assert self._pool_size ** 2 == output_tokens

    @property
    def output_dim(self) -> int:
        return self._output_tokens * _EMB_DIM

    def reduce(self, prune_result: PruneResult, *,
               prompt_emb: torch.Tensor | None = None) -> torch.Tensor:
        # 1. time average: [W, K, D] → [K, D]
        token_means = prune_result.tokens.mean(dim=0)

        # 2. fill back into 16x16 grid using token_indices
        D = token_means.shape[-1]
        grid = torch.zeros(self._GRID_SIZE * self._GRID_SIZE, D,
                           device=token_means.device, dtype=token_means.dtype)
        grid[prune_result.token_indices] = token_means
        # → [16, 16, D] → [1, D, 16, 16]
        grid = grid.reshape(self._GRID_SIZE, self._GRID_SIZE, D)
        grid = grid.permute(2, 0, 1).unsqueeze(0)

        # 3. adaptive avg pool → [1, D, pool, pool]
        pooled = F.adaptive_avg_pool2d(grid, (self._pool_size, self._pool_size))

        # 4. flatten → [output_tokens * D]
        return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)
```

#### TaskScoringReducer

```python
class TaskScoringReducer:
    """Select top-K by task relevance, then weighted pool. GPT Plan A-3~A-5.

    Requires prompt_emb to be passed via PruneResult or separately.
    Suitable for mixed-task databases. For single-task databases,
    prefer MeanPoolReducer.
    """

    def __init__(self, select_k: int = 32, temperature: float = 1.0):
        self._select_k = select_k
        self._temperature = temperature

    @property
    def output_dim(self) -> int:
        return _EMB_DIM

    def reduce(self, prune_result: PruneResult, *,
               prompt_emb: torch.Tensor | None = None) -> torch.Tensor:
        # 1. time average: [W, K, D] → [K, D]
        token_means = prune_result.tokens.mean(dim=0)

        if prompt_emb is None or prompt_emb.shape[0] == 0:
            # fallback: no task signal, plain mean pool
            return token_means.mean(dim=0)

        # 2. task scoring
        token_norm = F.normalize(token_means, dim=-1)
        prompt_vec = F.normalize(prompt_emb.mean(dim=0, keepdim=True), dim=-1)
        task_scores = (token_norm @ prompt_vec.T).squeeze(-1)  # [K]

        # 3. top-K selection
        k = min(self._select_k, token_means.shape[0])
        topk_indices = task_scores.topk(k).indices
        selected = token_means[topk_indices]        # [k, D]
        selected_scores = task_scores[topk_indices]  # [k]

        # 4. weighted pool
        weights = F.softmax(selected_scores / self._temperature, dim=0)
        return (weights.unsqueeze(-1) * selected).sum(dim=0)  # [D]
```

> 所有 reducer 的 `reduce()` 签名统一包含 `prompt_emb` keyword-only 参数。不需要的 reducer 直接忽略。KeyBuilder 统一传递 `prompt_emb=raw.get(PROMPT_EMB)`，无 isinstance 分支。

### 11.3 历史缓冲区

**文件**: `src/openpi/cache/components/key_builder.py`（在新增 class 之前）

```python
class _VisionHistoryBuffer:
    """Per-image FIFO buffer for windowed temporal scoring.

    Stores past W frames of vision tokens on the same device (GPU).
    Lifetime: created once per KeyBuilder, reset on on_episode_start().
    """

    def __init__(self, window_size: int):
        self._window_size = window_size
        self._buffer: list[torch.Tensor] = []  # each [256, emb_dim]

    def push(self, tokens: torch.Tensor) -> None:
        """Append current frame. Evicts oldest if over window_size."""
        self._buffer.append(tokens)
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)

    def get_window(self) -> torch.Tensor:
        """Return [len, 256, emb_dim], len <= window_size."""
        return torch.stack(self._buffer)

    @property
    def ready(self) -> bool:
        """True when buffer has enough frames for temporal scoring."""
        return len(self._buffer) >= self._window_size

    def reset(self) -> None:
        self._buffer.clear()
```

### 11.4 CP1TemporalPruneKeyBuilder（核心新增类）

**文件**: `src/openpi/cache/components/key_builder.py`（在文件底部新增）

```python
class CP1TemporalPruneKeyBuilder:
    """Two-step key builder: temporal pruning + pluggable token reduction.

    Step 1 (prune): Remove redundant (temporally static) vision tokens.
    Step 2 (reduce): Compress remaining tokens to fixed-dim key via TokenReducer.

    Only vision modalities go through prune → reduce.
    prompt_emb → mean pool → [emb_dim] (unchanged from _CP1BaseKeyBuilder).
    robot_state → raw [state_dim] (unchanged).

    Stateful: maintains per-image history buffers across inference steps.
    Must call on_episode_start() at episode boundaries.

    Data flow:
      collect() → push vision tokens into history buffers, cache state/prefix
      build()   → for each enabled vision field:
                     get_window() → _temporal_prune() → PruneResult
                     → reducer.reduce(PruneResult) → key vector
                   for prompt_emb: mean pool
                   for robot_state: raw
                   all → CPU float32
      clear()   → release per-cycle cache (history buffer NOT cleared)

    Coupling:
      DEPENDS ON:  Stage1Output field shapes (models_pytorch/pi0_pytorch.py),
                   TokenReducer protocol (token_reducer.py),
                   _slice_cp1_fields, _VISION_OFFSETS, _PROMPT_START (this file)
      CONSUMED BY: CacheOrchestrator.check()
      IF CHANGED:  Reducer output_dim must match backend vector_dims
    """

    def __init__(
        self,
        reducer: TokenReducer,
        enabled_fields: list[str] | None = None,
        prune_window_size: int = 4,
        temporal_keep_ratio: float = 0.5,
    ):
        self._reducer = reducer
        self._enabled = set(enabled_fields) if enabled_fields is not None else None
        self._window_size = prune_window_size
        self._keep_ratio = temporal_keep_ratio
        self._cache: dict[str, torch.Tensor] = {}

        # Per-image history buffers (only for enabled vision fields)
        self._history: dict[str, _VisionHistoryBuffer] = {}
        for field_name, _, _ in _VISION_OFFSETS:
            if self._enabled is None or field_name in self._enabled:
                self._history[field_name] = _VisionHistoryBuffer(prune_window_size)

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._cache["state"] = s1.state
            self._cache["prefix_embs"] = s1.prefix_embs

            # Push vision tokens into history ONLY on CP1.
            # Interceptor calls collect() for both CP1 and CP3 with the same
            # stage1 — pushing on both would double-count the same frame.
            # CP3 reuses the same key but does not advance the history window.
            if checkpoint_id == CheckpointID.CP1:
                raw = _slice_cp1_fields(s1.prefix_embs, s1.state, self._enabled)
                for field_name in self._history:
                    if field_name in raw:
                        self._history[field_name].push(raw[field_name])  # [256, D] GPU

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

        raw = _slice_cp1_fields(
            self._cache["prefix_embs"], self._cache["state"], self._enabled
        )
        keys: dict[str, torch.Tensor] = {}

        # ── Vision fields: prune → reduce ──
        for field_name in self._history:
            if field_name not in raw:
                continue
            buf = self._history[field_name]
            window = buf.get_window()  # [len, 256, D]

            if buf.ready:
                prune_result = self._temporal_prune(window)  # pruned=True
            else:
                # Degraded: window not full, skip pruning
                prune_result = PruneResult(
                    tokens=window,
                    token_indices=torch.arange(window.shape[1], device=window.device),
                    pruned=False,
                )

            reduced = self._reducer.reduce(
                prune_result, prompt_emb=raw.get(PROMPT_EMB),
            )  # [output_dim] GPU
            keys[field_name] = _to_cpu_float32(reduced)

        # ── prompt_emb: mean pool (unchanged) ──
        if PROMPT_EMB in raw:
            keys[PROMPT_EMB] = _to_cpu_float32(_mean_pool_tokens(raw[PROMPT_EMB]))

        # ── robot_state: raw (unchanged) ──
        if ROBOT_STATE in raw:
            keys[ROBOT_STATE] = _to_cpu_float32(raw[ROBOT_STATE])

        return keys

    def _temporal_prune(self, window: torch.Tensor) -> PruneResult:
        """Compute temporal change scores and keep top-K tokens.

        Args:
            window: [W, 256, emb_dim] vision tokens for one image across W frames.

        Returns:
            PruneResult with pruned=True, tokens [W, K, D], token_indices [K].
        """
        W, N, D = window.shape  # W=window_size, N=256, D=emb_dim

        # 1. Cast to float32 for scoring stability (input may be bf16/fp16)
        #    then L2 normalize each token
        normed = F.normalize(window.float(), dim=-1)  # [W, N, D] float32

        # 2. Cosine change between adjacent frames: 1 - cos(t, t+1)
        #    → [W-1, N], then mean over time → [N]
        cos_sim = (normed[:-1] * normed[1:]).sum(dim=-1)  # [W-1, N]
        temporal_scores = (1.0 - cos_sim).mean(dim=0)      # [N]

        # 3. Keep top-K by temporal score
        K = max(1, int(N * self._keep_ratio))
        topk = temporal_scores.topk(K)
        keep_indices = topk.indices                         # [K]
        keep_scores = topk.values                           # [K]

        # 4. Gather selected tokens across all frames
        #    window[:, keep_indices, :] → [W, K, D]
        selected = window[:, keep_indices, :]

        return PruneResult(
            tokens=selected,
            token_indices=keep_indices,
            pruned=True,
            temporal_scores=keep_scores,
        )

    def on_episode_start(self) -> None:
        """Reset history buffers. Called by Orchestrator at episode start."""
        for buf in self._history.values():
            buf.reset()

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        """Release per-cycle cache. History buffers are NOT cleared."""
        self._cache.clear()
```

### 11.5 config.py 改动

**文件**: `src/openpi/cache/config.py`

#### 11.5.1 新增 dataclass

```python
@dataclass
class ReducerConfig:
    type: str = "mean_pool"     # "mean_pool" | "max_pool" | "spatial_pool" | "task_scoring"
    output_tokens: int = 16     # only for spatial_pool
    select_k: int = 32          # only for task_scoring
    temperature: float = 1.0    # only for task_scoring
```

#### 11.5.2 扩展 KeyBuilderConfig

```python
@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"
    # ── temporal prune params (only for cp1_temporal_prune) ──
    prune_window_size: int = 4
    temporal_keep_ratio: float = 0.5
    reducer: ReducerConfig = field(default_factory=ReducerConfig)
```

#### 11.5.3 扩展 `_build_key_builder()`

在 `_build_key_builder()` 函数中添加分支：

```python
elif cfg.type == "cp1_temporal_prune":
    from openpi.cache.components.token_reducer import (
        MeanPoolReducer, MaxPoolReducer, SpatialPoolReducer, TaskScoringReducer,
    )
    from openpi.cache.components.key_builder import CP1TemporalPruneKeyBuilder

    reducer = _build_reducer(cfg.reducer)  # 新增工厂函数
    return CP1TemporalPruneKeyBuilder(
        reducer=reducer,
        enabled_fields=enabled_fields,
        prune_window_size=cfg.prune_window_size,
        temporal_keep_ratio=cfg.temporal_keep_ratio,
    )
```

新增 `_build_reducer()` 工厂函数：

```python
def _build_reducer(cfg: ReducerConfig):
    from openpi.cache.components.token_reducer import (
        MeanPoolReducer, MaxPoolReducer, SpatialPoolReducer, TaskScoringReducer,
    )
    if cfg.type == "mean_pool":
        return MeanPoolReducer()
    elif cfg.type == "max_pool":
        return MaxPoolReducer()
    elif cfg.type == "spatial_pool":
        return SpatialPoolReducer(output_tokens=cfg.output_tokens)
    elif cfg.type == "task_scoring":
        return TaskScoringReducer(select_k=cfg.select_k, temperature=cfg.temperature)
    else:
        raise ConfigValidationError(f"Unknown reducer.type '{cfg.type}'")
```

#### 11.5.4 扩展 `validate_cache_config()`

```python
# cp1_temporal_prune 校验
if config.key_builder.type == "cp1_temporal_prune":
    # CP1 必须启用：temporal prune 依赖 CP1 collect() 推进 history buffer
    cp1_cfg = config.checkpoints.get("cp1")
    if cp1_cfg is None or not cp1_cfg.enabled:
        errors.append("key_builder.type=cp1_temporal_prune requires checkpoints.cp1.enabled=true")
    for f in ("vision_0", "robot_state"):
        if f not in enabled_fields:
            errors.append(f"key_builder.type=cp1_temporal_prune requires keys.{f}.enabled=true")
    if not (0.0 < config.key_builder.temporal_keep_ratio <= 1.0):
        errors.append("temporal_keep_ratio must be in (0, 1]")
    if config.key_builder.prune_window_size < 1:
        errors.append("prune_window_size must be >= 1")
    # reducer.output_tokens must be perfect square for spatial_pool
    if config.key_builder.reducer.type == "spatial_pool":
        ot = config.key_builder.reducer.output_tokens
        ps = int(ot ** 0.5)
        if ps * ps != ot:
            errors.append(f"reducer.output_tokens={ot} must be a perfect square")
```

`_valid_key_builder_types` 集合增加 `"cp1_temporal_prune"`。

`_CONFIG_TYPES` dict 增加 `"ReducerConfig": ReducerConfig`，确保 YAML 嵌套的 `reducer` 字段被正确解析为 dataclass。

### 11.6 Orchestrator 生命周期适配

**文件**: `src/openpi/cache/orchestrator.py`

在 `_broadcast_episode_start()` 中增加一行：

```python
def _broadcast_episode_start(self) -> None:
    """Notify all components to clear their history buffers."""
    # ── 新增: key_builder ──
    if hasattr(self._key_builder, 'on_episode_start'):
        self._key_builder.on_episode_start()
    # ── 现有 ──
    for strategy in self._search_strategies.values():
        if hasattr(strategy, 'on_episode_start'):
            strategy.on_episode_start()
    # ... gate, judge 同上
```

duck typing，向后兼容：现有 KeyBuilder 没有 `on_episode_start` 方法则不调用。

### 11.7 离线 Artifact Builder 改动

**文件**: `exp/build_in_memory_cache_artifact.py`

#### 11.7.1 `_VECTOR_DIMS` 改为动态获取

`cp1_temporal_prune` 的 vision 维度取决于 reducer 类型（SpatialPoolReducer 输出 `output_tokens * 2048`，其他输出 `2048`）。静态 dict 不再适用。

修改 `_get_vector_dims()` 为：当 builder_type 是 `cp1_temporal_prune` 时，从 reducer 实例的 `output_dim` 属性动态计算维度。

```python
def _get_vector_dims(builder_type: str, reducer_type: str = "mean_pool",
                     output_tokens: int = 16) -> dict[str, int]:
    if builder_type in _VECTOR_DIMS:
        return _VECTOR_DIMS[builder_type]
    if builder_type == "cp1_temporal_prune":
        from openpi.cache.components.token_reducer import (
            MeanPoolReducer, MaxPoolReducer, SpatialPoolReducer, TaskScoringReducer,
        )
        reducer = _build_artifact_reducer(reducer_type, output_tokens)
        vision_dim = reducer.output_dim
        return {"vision_0": vision_dim, "vision_1": vision_dim,
                "prompt_emb": 2048, "robot_state": 32}
    raise ValueError(f"Unknown builder_type: {builder_type}")
```

#### 11.7.2 CLI 参数扩展

```python
# --builder-type choices 不能只从 _VECTOR_DIMS.keys() 生成，
# 因为 cp1_temporal_prune 使用动态维度，不在静态 dict 里。
_ALL_BUILDER_TYPES = list(_VECTOR_DIMS.keys()) + ["cp1_temporal_prune"]

parser.add_argument("--builder-type", required=True, choices=_ALL_BUILDER_TYPES)
parser.add_argument("--reducer-type", default="mean_pool",
                    choices=["mean_pool", "max_pool", "spatial_pool", "task_scoring"])
parser.add_argument("--output-tokens", type=int, default=16,
                    help="SpatialPoolReducer output_tokens (must be perfect square)")
parser.add_argument("--prune-window-size", type=int, default=4)
parser.add_argument("--temporal-keep-ratio", type=float, default=0.5)
```

**`main()` 调用 `build_artifact()` 时传入新参数**:

```python
artifact = build_artifact(
    args.data_dir, args.builder_type, args.checkpoint_id,
    workers=args.workers,
    reducer_type=args.reducer_type,
    output_tokens=args.output_tokens,
    prune_window_size=args.prune_window_size,
    temporal_keep_ratio=args.temporal_keep_ratio,
)
```

#### 11.7.3 参数传递链：CLI → `build_artifact()` → worker

多进程架构下，CLI 参数需要经过完整的传递链才能到达 worker 函数。改动如下：

**`build_artifact()` 函数签名扩展**:

```python
def build_artifact(
    ...,
    reducer_type: str = "mean_pool",
    output_tokens: int = 16,
    prune_window_size: int = 4,
    temporal_keep_ratio: float = 0.5,
) -> None:
```

**`ProcessPoolExecutor.submit()` 传参**:

```python
futures.append(executor.submit(
    _process_episode, h5_path_str, builder_type, checkpoint_id_str,
    reducer_type, output_tokens, prune_window_size, temporal_keep_ratio,
))
```

#### 11.7.4 `_process_episode()` 和 `_create_builder()` 改动

签名扩展为接收所有 builder 参数：

```python
def _process_episode(
    h5_path_str: str, builder_type: str, checkpoint_id_str: str,
    reducer_type: str = "mean_pool", output_tokens: int = 16,
    prune_window_size: int = 4, temporal_keep_ratio: float = 0.5,
) -> list | None:
```

当前代码在每步结束时调用 `builder.clear()`。对有状态 builder 需要改为：
- episode 开始时调用 `on_episode_start()`（如果有）
- 每步 `collect()` → `build()` → `clear()`（`clear()` 只清 per-cycle cache，不清历史缓冲区）
- 不再每步创建新 builder

```python
    # ... 现有代码 ...
    builder = _create_builder(builder_type, reducer_type, output_tokens,
                              prune_window_size, temporal_keep_ratio)

    # 新增: 通知 builder episode 开始
    if hasattr(builder, 'on_episode_start'):
        builder.on_episode_start()

    for step_name in step_names:
        group = f[step_name]
        fake_stage1 = _build_fake_stage1(group)
        builder.collect(cp_id, stage1=fake_stage1)
        query_keys = builder.build(cp_id)
        builder.clear()  # 只清 per-cycle cache, 历史缓冲区保留
        # ... 构建 CacheEntry ...
```

**`_create_builder()` 签名同步扩展**（原 §11.7.3）:

```python
def _create_builder(builder_type: str, reducer_type: str = "mean_pool",
                    output_tokens: int = 16, prune_window_size: int = 4,
                    temporal_keep_ratio: float = 0.5) -> QueryKeyBuilder:
    # ... 现有 builder 类型 ...
    elif builder_type == "cp1_temporal_prune":
        from openpi.cache.components.key_builder import CP1TemporalPruneKeyBuilder
        reducer = _build_artifact_reducer(reducer_type, output_tokens)
        return CP1TemporalPruneKeyBuilder(
            reducer=reducer,
            prune_window_size=prune_window_size,
            temporal_keep_ratio=temporal_keep_ratio,
        )
```

这个改动对现有 builder 类型无影响——它们的 `clear()` 本来就只清 `_cache`，额外参数被忽略。

### 11.8 `__init__.py` 导出

**文件**: `src/openpi/cache/__init__.py`

```python
from openpi.cache.components.key_builder import (
    QueryKeyBuilder, PlaceholderKeyBuilder, CP1TemporalPruneKeyBuilder,
)
from openpi.cache.components.token_reducer import (
    PruneResult, TokenReducer, MeanPoolReducer, MaxPoolReducer,
    SpatialPoolReducer, TaskScoringReducer,
)
```

### 11.9 测试计划

**文件**: `tests/cache/components/test_temporal_prune.py`（新增）

| 测试 | 验证内容 |
|------|----------|
| `test_temporal_scoring_static_tokens_removed` | 构造全静态 + 部分动态的 [W, 256, D] 输入，验证静态 token 被剪掉 |
| `test_temporal_scoring_all_dynamic` | 所有 token 都在变化，验证保留 top keep_ratio |
| `test_prune_result_indices_correct` | 验证 token_indices 对应正确的原始位置 |
| `test_degraded_mode_single_frame` | 只有 1 帧，验证 pruned=False 且 256 token 全部传递 |
| `test_degraded_mode_partial_window` | 2~W-1 帧，验证 pruned=False |
| `test_full_window_normal_prune` | W 帧满窗口，验证 pruned=True |
| `test_episode_start_resets_buffer` | 调用 on_episode_start() 后历史清空 |
| `test_history_buffer_fifo` | 超过 window_size 时旧帧被淘汰 |
| `test_mean_pool_reducer` | MeanPoolReducer 输出 [2048] |
| `test_max_pool_reducer` | MaxPoolReducer 输出 [2048] |
| `test_spatial_pool_reducer_full_grid` | 256 token 填满网格，验证 output_tokens * 2048 维度 |
| `test_spatial_pool_reducer_pruned` | 128 token + indices，验证填回网格后池化正确 |
| `test_task_scoring_reducer` | 构造已知 prompt_emb，验证 top-K 选择和加权池化 |
| `test_task_scoring_reducer_no_prompt` | prompt_emb=None 时退化为 mean pool |
| `test_full_pipeline_collect_build` | 端到端：多步 collect → build，验证 key 输出格式 |
| `test_output_dim_consistent_across_pruned_states` | pruned=True 和 pruned=False 输出维度一致 |
| `test_spatial_pool_reducer_varying_missing_ratio` | 不同缺失比例 (25%/50%/75%) 下零填充输出差异符合预期 |

**集成边界测试**（审查补充）:

| 测试 | 建议位置 | 验证内容 |
|------|----------|----------|
| `test_cp1_cp3_same_cycle_single_push` | `test_temporal_prune.py` | CP1+CP3 同一周期调用后，history 长度只增加 1 |
| `test_config_loads_temporal_prune_reducer_yaml` | `test_config.py` | YAML 中 `key_builder.reducer.type` 被解析成 `ReducerConfig` 并成功构建 |
| `test_artifact_matches_online_sequence` | `test_temporal_prune.py` | 同一 episode 顺序输入离线 builder 与在线 builder，得到一致 key |
| `test_builder_clear_preserves_history` | `test_temporal_prune.py` | `clear()` 只清 per-cycle cache，不清 history |
| `test_orchestrator_broadcasts_to_key_builder` | `test_orchestrator.py` | `on_episode_start()` 会广播到 key_builder |

### 11.10 Phase 汇总

| Phase | 范围 | 涉及文件 |
|-------|------|----------|
| **Phase 1: 核心实现** | PruneResult + TokenReducer protocol + 4 个 reducer + _VisionHistoryBuffer + CP1TemporalPruneKeyBuilder | `key_builder.py`, `token_reducer.py`(新增) |
| **Phase 2: 集成** | config 注册 + artifact builder 适配 + orchestrator 生命周期 + __init__.py | `config.py`, `build_in_memory_cache_artifact.py`, `orchestrator.py`, `__init__.py` |
| **Phase 3: 测试** | 单元测试 | `test_temporal_prune.py`(新增) |
| **Phase 4 (后续): Plan B** | greedy merge 分支 + DualKeyReducer | `key_builder.py`, `token_reducer.py` |

## 12. G2 代码审查记录（2026-04-12）

**审查结论**: **not approved**。核心结构基本按 G1 后的 plan 落地，但当前实现仍有几个会在合法配置下产生错误 key、NaN，或让离线 artifact 与在线查询 key 不一致的问题。建议修完下列阻塞项后再进入 G2 复审。

### 12.1 阻塞问题

1. `prune_window_size=1` 是当前配置校验允许的合法值，但会让 temporal score 变成 NaN。

   - 位置: `src/openpi/cache/config.py:529` 到 `src/openpi/cache/config.py:532` 只要求 `prune_window_size >= 1`。
   - 位置: `src/openpi/cache/components/key_builder.py:567` 到 `src/openpi/cache/components/key_builder.py:568` 中 buffer 满 1 帧就进入 `_temporal_prune()`。
   - 位置: `src/openpi/cache/components/key_builder.py:612` 到 `src/openpi/cache/components/key_builder.py:613` 对 `normed[:-1]` 和 `normed[1:]` 求相邻帧差；当 W=1 时相邻帧数量为 0，`mean(dim=0)` 得到 NaN。
   - 原因: temporal pruning 至少需要 2 帧才能计算相邻帧变化。当前实现会返回 `pruned=True` 且 `temporal_scores` 全 NaN，top-k 选择也没有实际意义。
   - 建议: 二选一即可。更简单的做法是在 `validate_cache_config()` 中改为要求 `prune_window_size >= 2`，并在 `CP1TemporalPruneKeyBuilder.__init__()` 或 `_VisionHistoryBuffer` 构造处 fail fast。另一种做法是保留 `1`，但 `buf.ready` 或 `build()` 中对 W<2 继续走 degraded mode。
   - 建议补测: 在 `tests/cache/components/test_temporal_prune.py` 增加 `prune_window_size=1` 的配置校验或 degraded 行为测试，确保不会产生 NaN。

2. `TaskScoringReducer` 的参数缺少边界校验，合法配置可以产生 NaN 或全零 key。

   - 位置: `src/openpi/cache/config.py:522` 到 `src/openpi/cache/config.py:539` 只校验了 `temporal_keep_ratio`、`prune_window_size` 和 `spatial_pool.output_tokens`，没有校验 `reducer.type`、`task_scoring.select_k`、`task_scoring.temperature`。
   - 位置: `src/openpi/cache/components/token_reducer.py:238` 到 `src/openpi/cache/components/token_reducer.py:245` 使用 `select_k` 做 top-k，并用 `selected_scores / self._temperature` 做 softmax。
   - 原因: `temperature=0` 会产生 NaN；`select_k=0` 会返回全零向量。这两种配置目前都能通过 `load_cache_config()`，直到运行时才污染 key。
   - 建议: 在 `validate_cache_config()` 中加入 reducer 类型集合校验；当 `reducer.type == "task_scoring"` 时要求 `select_k >= 1` 且 `temperature > 0`。也建议在 `TaskScoringReducer.__init__()` 中做构造期校验，避免绕过 config 工厂时出错。
   - 建议补测: 在 `tests/cache/test_config.py` 或 `tests/cache/components/test_temporal_prune.py` 覆盖 `temperature=0`、`select_k=0`、未知 `reducer.type`。

3. 离线 artifact builder 没有透传 `TaskScoringReducer` 的 `select_k` 和 `temperature`，会让非默认配置下离线库 key 与在线查询 key 不一致。

   - 位置: `exp/build_in_memory_cache_artifact.py:48` 到 `exp/build_in_memory_cache_artifact.py:64` 中 `_build_artifact_reducer()` 对 `task_scoring` 总是构造默认 `TaskScoringReducer()`。
   - 位置: `exp/build_in_memory_cache_artifact.py:283` 到 `exp/build_in_memory_cache_artifact.py:291` 的 `build_artifact()` 签名只接收 `reducer_type`、`output_tokens`、`prune_window_size`、`temporal_keep_ratio`。
   - 位置: `exp/build_in_memory_cache_artifact.py:355` 到 `exp/build_in_memory_cache_artifact.py:360` 的 CLI 也没有 `--select-k` 和 `--temperature`。
   - 原因: 在线配置已经暴露 `ReducerConfig.select_k` 和 `ReducerConfig.temperature`。如果在线使用非默认 task scoring 参数，而 artifact 用默认参数生成库，维度仍然都是 2048，`InMemoryBackend.load_artifact()` 无法发现这个语义不一致，实验结果会被静默污染。
   - 建议: 给 artifact CLI、`build_artifact()`、`_process_episode()`、`_create_builder()`、`_build_artifact_reducer()` 全链路增加 `select_k` 和 `temperature`，并在 artifact 元数据中记录 reducer 参数，至少便于人工核对。
   - 建议补测: 在 `tests/exp/test_build_in_memory_cache_artifact.py` 或 `tests/cache/components/test_temporal_prune.py` 增加非默认 task scoring 参数的 offline/online key 一致性测试。

### 12.2 非阻塞问题与建议

1. 建议在 config 阶段校验 reducer 输出维度与 `backend.vector_dims` 一致。

   - 位置: `src/openpi/cache/config.py:390` 到 `src/openpi/cache/config.py:398` 只检查启用字段是否出现在 `backend.vector_dims`，没有检查维度是否等于 reducer 输出。
   - 原因: 对 `cp1_temporal_prune + spatial_pool`，vision 维度取决于 `output_tokens * 2048`。如果 YAML 中 `backend.vector_dims.vision_0` 写错，当前会到第一次查询或写入时才由 `CacheStorage` 报 shape mismatch。
   - 建议: 对 `cp1_temporal_prune` 在 `validate_cache_config()` 中根据 reducer 配置计算 expected dim，并检查所有启用的 vision 字段。

2. `_VisionHistoryBuffer.push()` 建议保存 `tokens.detach().clone()` 或至少 `tokens.detach()`，并在注释中明确生命周期策略。

   - 位置: `src/openpi/cache/components/key_builder.py:460` 到 `src/openpi/cache/components/key_builder.py:464` 当前直接保存 stage output 切片 view。
   - 原因: 这个 builder 明确跨 infer 周期保留历史，而原 `QueryKeyBuilder.collect()` 文档强调 stage output 引用只在单次 infer 内有效。当前 no-grad 且 no-cudagraph 路径下大概率能工作，但直接保存 view 会保留整个 `prefix_embs` storage，并让生命周期依赖 PyTorch 输出复用行为。
   - 建议: 如果性能允许，保存 detached clone，使 history buffer 真正拥有自己的 W 帧 vision token；如果不 clone，也建议补充注释说明为什么跨周期引用是安全的。

3. 测试覆盖还缺少 G1 复审中要求的几个集成边界。

   - 已有: `tests/cache/components/test_temporal_prune.py` 覆盖了 CP1/CP3 单次 push、degraded mode、reducer 输出形状、config 工厂基础路径。
   - 缺少: `test_artifact_matches_online_sequence`、`test_orchestrator_broadcasts_to_key_builder`、YAML 解析到 `ReducerConfig` 的完整 `load_cache_config()` 测试。
   - 原因: 当前最容易出错的是跨组件契约，不是单个 reducer 的 shape。尤其是 artifact/online 一致性，需要真实顺序 episode 输入来保护。

4. `exp/build_in_memory_cache_artifact.py` 的现有测试路径需要单独整理。

   - 观察: `uv run python -m pytest tests/cache/test_config.py tests/cache/components/test_temporal_prune.py` 通过，结果为 70 passed。
   - 观察: `uv run pytest ...` 入口当前指向过期解释器，不能作为可靠验证命令；使用 `uv run python -m pytest ...` 可绕过。
   - 观察: 联合运行 `tests/exp/test_build_in_memory_cache_artifact.py` 时，在 artifact 构建测试处长时间无输出，未能完成验证。这个文件还保留了对 entry id 的旧断言，和当前 `exp/build_in_memory_cache_artifact.py` 中 `trajectory_id:step_idx` 的实现不一致。
   - 建议: 将 artifact 测试改为显式 `workers=1` 或提供同步 worker 路径，避免单元测试默认拉起所有 CPU worker；同时同步 entry id 断言或修正文档/实现。

### 12.3 疑问

1. `prune_window_size=1` 是否应该被支持？如果实验设计上没有单帧 temporal pruning 的意义，建议明确禁止，这比在算法里做特殊分支更清晰。

2. `TaskScoringReducer` 的非默认 `select_k`、`temperature` 是否计划纳入第一轮实验矩阵？如果不纳入，建议暂时不要在 YAML 暴露这些参数，或至少在 artifact builder 中记录“不支持非默认 task scoring 参数”的限制。

3. history buffer 是否允许保留 stage output view？如果这个选择是为了避免 clone 成本，建议在 plan 或代码注释中明确它依赖 `max-autotune-no-cudagraphs` 且会保留 W 帧 `prefix_embs` storage。

### 12.4 G2 审查回复（2026-04-12）

**总体**: 审查质量较好，3 个阻塞问题都是真实 bug。逐条回复如下。

#### 阻塞问题回复

| # | 审查意见 | 结论 | 处理 |
|---|---------|------|------|
| 12.1.1 | `prune_window_size=1` → NaN | **接受** | config 校验改为 `>= 2`；`CP1TemporalPruneKeyBuilder.__init__()` 加 fail-fast 检查；补测 `test_prune_window_size_1_rejected` + `test_config_rejects_prune_window_1` |
| 12.1.2 | TaskScoringReducer 缺参数边界校验 | **接受** | config 校验增加 `reducer.type` 集合校验、`select_k >= 1`、`temperature > 0`；`TaskScoringReducer.__init__()` 加构造期校验；补测 `test_task_scoring_select_k_0_rejected`、`test_task_scoring_temperature_0_rejected`、`test_config_rejects_temperature_0`、`test_config_rejects_unknown_reducer_type` |
| 12.1.3 | artifact builder 缺 `select_k`/`temperature` 透传 | **接受** | 全链路增加 `select_k` 和 `temperature` 参数：CLI → `build_artifact()` → `executor.submit()` → `_process_episode()` → `_create_builder()` → `_build_artifact_reducer()`。artifact 元数据新增 `reducer_params` 字段记录所有 reducer 参数。 |

#### 非阻塞问题回复

| # | 审查意见 | 结论 | 处理 |
|---|---------|------|------|
| 12.2.1 | config 阶段校验 reducer dim vs vector_dims | **接受** | `validate_cache_config()` 对 `cp1_temporal_prune` 增加 vision 字段的维度交叉校验，根据 reducer 类型计算 expected dim 并与 `backend.vector_dims` 比对。补测 `test_config_rejects_dim_mismatch` |
| 12.2.2 | history buffer 应 clone | **接受** | `_VisionHistoryBuffer.push()` 改为 `tokens.detach().clone()`，补充注释说明生命周期策略 |
| 12.2.3 | 缺少集成边界测试 | **部分接受** | 补测 `test_artifact_matches_online_sequence`（同一 episode 顺序输入两个独立 builder 对比 key 一致性）和 `test_orchestrator_broadcasts_to_key_builder`（验证 on_episode_start 广播到 key_builder）。YAML 完整 `load_cache_config()` 测试暂缓——工厂路径已由 `test_config_builds_*` 覆盖，完整 YAML fixture 成本偏高。 |
| 12.2.4 | artifact 测试路径整理 | **接受** | 修复 3 个问题：(1) entry id 断言 `"ok_step_042"` → `"ok:42"` 与代码一致；(2) `build_artifact()` 新增 `workers=-1` 串行模式，主进程直接处理避免 fork 开销；(3) 测试改用 `workers=-1`，从卡死降到 1s 完成。 |

#### 疑问回复

| # | 疑问 | 回复 |
|---|------|------|
| 12.3.1 | `prune_window_size=1` 是否支持？ | **明确禁止**。单帧没有 temporal scoring 意义，config 校验 + 构造器双重阻止。 |
| 12.3.2 | TaskScoring 非默认参数是否纳入实验矩阵？ | 第一轮实验以默认值为主，但 YAML 暴露参数是接口完整性要求，artifact builder 现已完整透传且在元数据中记录参数，不会有"静默不一致"风险。 |
| 12.3.3 | history buffer view vs clone？ | **已改为 clone**。跨 infer 周期保留 stage output view 违反 `QueryKeyBuilder.collect()` 文档契约，clone 成本是 [256, 2048] × float16 ≈ 1MB/帧，W=4 时 4MB，可以接受。 |

#### 修改清单

**代码修改**:
- `src/openpi/cache/config.py`: `prune_window_size >= 2`，reducer 类型/参数边界校验，vision dim vs vector_dims 交叉校验
- `src/openpi/cache/components/key_builder.py`: 构造器 fail-fast `prune_window_size < 2`，`push()` 改为 `detach().clone()`
- `src/openpi/cache/components/token_reducer.py`: `TaskScoringReducer.__init__()` 校验 `select_k >= 1`、`temperature > 0`
- `exp/build_in_memory_cache_artifact.py`: 全链路增加 `select_k`/`temperature`，artifact 元数据记录 `reducer_params`，新增 `workers=-1` 串行模式
- `tests/exp/test_build_in_memory_cache_artifact.py`: entry id 断言修正 `"ok_step_042"` → `"ok:42"`，测试改用 `workers=-1` 避免 fork 开销

**测试新增** (42 tests total, +10 vs G2 前):
- `TestConstructorValidation`: 4 tests (window_size=1, select_k=0, temperature=0, temperature<0)
- `TestConfigIntegration`: +4 tests (rejects_prune_window_1, rejects_temperature_0, rejects_unknown_reducer_type, rejects_dim_mismatch)
- `TestFullPipeline`: +1 test (artifact_matches_online_sequence)
- `TestOrchestratorIntegration`: +1 test (broadcasts_to_key_builder)

**测试结果**: 252 passed (cache) + 3 passed (artifact) = 255 passed, 0 failed。

### 12.5 G2 复审意见（2026-04-12）

**审查结论**: **not approved**。12.4 已经关闭 12.1 中的三个主要阻塞项，且本地复测 `uv run python -m pytest tests/cache tests/exp/test_build_in_memory_cache_artifact.py` 通过，结果为 255 passed。复审又发现两个同类的参数边界问题：它们不影响默认实验配置，但会让无效实验参数生成错误 key 或以非预期异常退出。建议修完后再进行 G2 最终确认。

#### 仍需处理的问题

1. `SpatialPoolReducer.output_tokens` 只校验 perfect square，没有校验正数。

   - 位置: `src/openpi/cache/components/token_reducer.py:154` 到 `src/openpi/cache/components/token_reducer.py:160` 中 `output_tokens=0` 会通过构造，`output_dim` 变成 0，`reduce()` 返回空向量。
   - 位置: `src/openpi/cache/config.py:545` 到 `src/openpi/cache/config.py:550` 中 `output_tokens=0` 也会通过 config 校验；`output_tokens=-1` 会在 `int(ot**0.5)` 处抛出 `TypeError`，而不是收敛为 `ConfigValidationError`。
   - 原因: 0 维 vision key 会让后端相似度退化成无意义结果，属于静默污染实验；负数则破坏 config 层统一错误语义。
   - 建议: 在 `SpatialPoolReducer.__init__()` 和 `validate_cache_config()` 中都要求 `output_tokens >= 1`，再检查 perfect square。补测 `output_tokens=0`、`output_tokens=-1` 两个路径。

2. `temporal_keep_ratio` 的边界只在 config 层校验，直接构造和 artifact CLI 仍可绕过。

   - 位置: `src/openpi/cache/config.py:529` 到 `src/openpi/cache/config.py:530` 已校验 `(0, 1]`，但 `src/openpi/cache/components/key_builder.py:520` 到 `src/openpi/cache/components/key_builder.py:535` 的构造器没有同等 fail-fast。
   - 位置: `exp/build_in_memory_cache_artifact.py:293` 到 `exp/build_in_memory_cache_artifact.py:303` 暴露 `temporal_keep_ratio`，但 artifact builder 不经过 `validate_cache_config()`。
   - 原因: `temporal_keep_ratio=0` 会被 `_temporal_prune()` 的 `max(1, int(N * ratio))` 静默改成保留 1 个 token；`temporal_keep_ratio > 1` 会在 `topk()` 处运行时报错。artifact 生成是实验关键入口，应该和在线配置保持同等参数约束。
   - 建议: 在 `CP1TemporalPruneKeyBuilder.__init__()` 中校验 `0.0 < temporal_keep_ratio <= 1.0`，artifact builder 会自然继承这个检查。补测直接构造和 `build_artifact(..., workers=-1)` 的非法 ratio 路径。

#### 已确认关闭的事项

- `prune_window_size=1` 已由 config 和构造器双重拒绝。
- `TaskScoringReducer.select_k` 和 `temperature` 已由 config 和构造器双重拒绝。
- artifact builder 已透传 `select_k` 和 `temperature`，并在 `reducer_params` 中记录。
- `_VisionHistoryBuffer.push()` 已改为 `detach().clone()`，跨 infer 周期保存 view 的生命周期风险已关闭。
- `workers=-1` 串行模式解决了 artifact 单测 fork 开销问题，相关测试已通过。

#### 非阻塞建议

- `tests/cache/components/test_temporal_prune.py` 中的 `test_artifact_matches_online_sequence` 实际比较的是两个在线 builder，没有经过 `exp/build_in_memory_cache_artifact.py` 或 HDF5 artifact 路径。建议后续改名为 `test_two_builders_match_same_sequence`，或另补真正的 artifact/online 一致性测试。
- `exp/build_in_memory_cache_artifact.py` 的 CLI help 建议说明 `--workers -1` 是串行模式，避免后续实验使用者只看到 “0 = all CPUs”。

### 12.6 G2 复审回复（2026-04-12）

#### 阻塞问题回复

| # | 审查意见 | 结论 | 处理 |
|---|---------|------|------|
| 12.5.1 | `SpatialPoolReducer.output_tokens` 缺正数校验 | **接受** | 构造器增加 `output_tokens < 1` 检查（在 perfect square 校验之前）；config 校验同步增加 `output_tokens >= 1` 检查。补测 `test_spatial_pool_output_tokens_0_rejected`、`test_spatial_pool_output_tokens_negative_rejected` |
| 12.5.2 | `temporal_keep_ratio` 构造器缺 fail-fast | **接受** | `CP1TemporalPruneKeyBuilder.__init__()` 增加 `0.0 < temporal_keep_ratio <= 1.0` 校验，与 config 层保持一致。artifact builder 自然继承。补测 `test_temporal_keep_ratio_0_rejected`、`test_temporal_keep_ratio_above_1_rejected` |

#### 非阻塞建议回复

| # | 审查意见 | 结论 | 处理 |
|---|---------|------|------|
| 1 | 测试改名 `test_artifact_matches_online_sequence` → `test_two_builders_match_same_sequence` | **接受** | 已改名 |
| 2 | CLI help 说明 `--workers -1` | **接受** | help 改为 `”Parallel workers (0 = all CPUs, -1 = serial in main process)”` |

#### 修改清单

**代码修改**:
- `src/openpi/cache/components/token_reducer.py`: `SpatialPoolReducer.__init__()` 增加 `output_tokens >= 1` 校验
- `src/openpi/cache/components/key_builder.py`: `CP1TemporalPruneKeyBuilder.__init__()` 增加 `temporal_keep_ratio ∈ (0, 1]` 校验
- `src/openpi/cache/config.py`: `validate_cache_config()` 对 `spatial_pool` 增加 `output_tokens >= 1` 前置校验
- `exp/build_in_memory_cache_artifact.py`: `--workers` help 补充 `-1` 说明
- `tests/cache/components/test_temporal_prune.py`: 测试改名 + 4 个新校验测试

**测试结果**: 259 passed (cache 256 + artifact 3), 0 failed。

### 12.7 G2 最终结论（2026-04-12）

**审查结论**: **code approved**。12.6 已关闭 12.5 中剩余的两个参数边界问题，G2 通过。

#### 复核结果

- `SpatialPoolReducer.output_tokens` 已在 `src/openpi/cache/components/token_reducer.py` 构造器中拒绝 `< 1`，避免 0 维 key 和负数参数。
- `src/openpi/cache/config.py` 已在 `validate_cache_config()` 中对 `spatial_pool.output_tokens` 做 `>= 1` 前置校验，再做 perfect square 校验，避免负数触发非预期 `TypeError`。
- `CP1TemporalPruneKeyBuilder` 已在 `src/openpi/cache/components/key_builder.py` 构造器中拒绝非法 `temporal_keep_ratio`，artifact builder 也会继承该 fail-fast。
- `exp/build_in_memory_cache_artifact.py` 的 `--workers` help 已说明 `-1` 串行模式。
- `tests/cache/components/test_temporal_prune.py` 已把误导性的 `test_artifact_matches_online_sequence` 改名为 `test_two_builders_match_same_sequence`，并补充 4 个构造期参数校验测试。

#### 验证

本地复测命令:

```bash
uv run python -m pytest tests/cache tests/exp/test_build_in_memory_cache_artifact.py
```

结果: `259 passed, 12 warnings`。warnings 均为既有环境/依赖警告或测试辅助类收集警告，不影响本次功能结论。

#### 残余建议

- 后续若要更强保护离线/在线一致性，可以另补一个真正经过 HDF5 artifact builder 的 `cp1_temporal_prune` 顺序 episode 测试。目前这不是 G2 阻塞项，因为参数链、生命周期、shape 和边界校验已覆盖，且 artifact 基础测试已通过。
