# Trajectory Search 实现计划

**Status**: `Plan`
**Date**: 2026-04-07
**需求文档**: [trajectory_search_requirements.log.md](trajectory_search_requirements.log.md)

---

## 总览

本计划将 trajectory search 需求拆解为 **7 个 Phase**，按依赖关系排序。每个 Phase 独立可测试，合并后不破坏现有功能（`trajectory_depth=1` 退化为单步搜索）。

**依赖图：**

```
Phase 1 (数据结构) ──┬──→ Phase 2 (Backend 轨迹搜索)
                     ├──→ Phase 3 (SearchStrategy 升级) ──→ Phase 5 (Orchestrator 重构)
                     ├──→ Phase 4 (Gate/Judge 预留接口) ──→ Phase 5
                     └──→ Phase 6 (WritePolicy + Config) ──→ Phase 5
Phase 7 (Ingest 脚本) 依赖 Phase 1
```

**可并行的 Phase**：Phase 2 / 3 / 4 / 6 / 7 均只依赖 Phase 1，可并行开发。Phase 5 是最终集成点。

---

## Phase 1: 数据结构变更

**目标**：修改 `storage_types.py` 和 `config.py` 中的 dataclass 定义，为后续 Phase 提供基础类型。纯类型变更，不改任何行为逻辑。

### 1.1 `src/openpi/cache/storage_types.py`

#### CachePayload：删除 `next_action_chunk`

```python
# 删除这个字段：
# next_action_chunk: Optional[torch.Tensor] = None   # [50, 32] CPU float32

# 删除 validate_for_checkpoint 中的 CP3 校验：
# if checkpoint_id == CheckpointID.CP3 and self.next_action_chunk is None:
#     raise ValueError("next_action_chunk is required for CP3")
```

**影响范围**：需全局搜索 `next_action_chunk` 的所有引用并清理：
- `storage_types.py` — 字段定义 + validate_for_checkpoint
- `orchestrator.py` — `schedule_next_action()` 中如有引用
- `backends/in_memory_backend.py` — artifact 加载 / 序列化中如有引用
- `backends/qdrant_backend.py` — payload 序列化中如有引用
- 测试文件 — 构造 CachePayload 的测试用例

#### CacheEntry：新增链表字段

```python
@dataclass
class CacheEntry:
    id: str
    checkpoint_id: CheckpointID
    query_keys: dict[str, torch.Tensor]
    payload: CachePayload
    step_idx: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    # ── 新增 ──
    prev_ids: list[str] = field(default_factory=list)
    next_ids: list[str] = field(default_factory=list)
    trajectory_id: Optional[str] = None
```

- 所有新增字段有默认值，现有构造代码无需修改
- `validate()` 方法不变（新字段无 CP 级约束需要校验）

#### QuerySpec：新增轨迹字段

```python
@dataclass
class QuerySpec:
    # ... 现有字段不变 ...

    # ── 新增 ──
    trajectory_history: Optional[list[dict[str, torch.Tensor]]] = None
    # query 侧历史 key 序列，newest-first: [当前step的keys, t-1的keys, t-2的keys, ...]
    # None 时退化为现有单步搜索行为

    trajectory_weights: Optional[list[float]] = None
    # 各层权重，newest-first: [w_当前, w_t-1, w_t-2, ...]
    # 必须与 trajectory_history 等长，None 时不做轨迹融合
```

#### 新增 StepRecord / EpisodeRecord

```python
@dataclass
class StepRecord:
    """单步暂存记录，用于 episode 结束时构造 CacheEntry 链。"""
    query_keys: dict[str, torch.Tensor]   # CPU float32
    action_chunk: torch.Tensor            # CPU float32，必填（无 action 的步不写入）

@dataclass
class EpisodeRecord:
    """整个 episode 的暂存记录，传给 WritePolicy 判断是否写入。"""
    steps: list[StepRecord]
    task_key: str
    miss_by_checkpoint: dict[CheckpointID, int]  # e.g. {CP1: 3, CP3: 50}
    total_steps: int
```

### 1.2 `src/openpi/cache/config.py`

#### SearchStrategyConfig 新增字段

```python
@dataclass
class SearchStrategyConfig:
    # ... 现有字段不变 ...
    # ── 新增 ──
    trajectory_depth: int = 1
    trajectory_weights: Optional[list[float]] = None
```

#### 新增 WritePolicyConfig

```python
@dataclass
class WritePolicyConfig:
    type: str = "on_any_miss"   # on_any_miss | always | never
```

#### CacheConfig 新增字段

```python
@dataclass
class CacheConfig:
    # ... 现有字段不变 ...
    # ── 新增 ──
    write_policy: WritePolicyConfig = field(default_factory=WritePolicyConfig)
```

#### 注册表更新

```python
_CONFIG_TYPES["WritePolicyConfig"] = WritePolicyConfig
```

#### validate_cache_config 新增校验

```python
# 在 validate_cache_config() 末尾追加：

# trajectory 校验
for cp_name, cp_cfg in config.checkpoints.items():
    ss = cp_cfg.search_strategy

    # trajectory_depth >= 1
    if ss.trajectory_depth < 1:
        raise ConfigValidationError(
            f"checkpoints.{cp_name}.search_strategy: "
            f"trajectory_depth must be >= 1, got {ss.trajectory_depth}"
        )

    if ss.trajectory_depth > 1:
        if ss.trajectory_weights is None:
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights required when trajectory_depth={ss.trajectory_depth}"
            )
        if len(ss.trajectory_weights) != ss.trajectory_depth:
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights length ({len(ss.trajectory_weights)}) "
                f"!= trajectory_depth ({ss.trajectory_depth})"
            )
        # weights 非负且总和 > 0
        if any(w < 0 for w in ss.trajectory_weights):
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights must be non-negative"
            )
        if sum(ss.trajectory_weights) <= 0:
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights sum must be > 0"
            )

    # Qdrant + trajectory_depth > 1 → 配置阶段 fail-fast
    # backend 是顶层 CacheConfig.backend，不是 CheckpointConfig 字段
    if ss.trajectory_depth > 1 and config.backend.type == "qdrant":
        raise ConfigValidationError(
            f"checkpoints.{cp_name}: trajectory_depth > 1 is not supported "
            f"with Qdrant backend. Use InMemoryBackend or set trajectory_depth=1."
        )

# write_policy.type 校验
_VALID_WRITE_POLICY_TYPES = {"on_any_miss", "always", "never"}
if config.write_policy.type not in _VALID_WRITE_POLICY_TYPES:
    raise ConfigValidationError(
        f"write_policy.type '{config.write_policy.type}' unknown, "
        f"valid: {_VALID_WRITE_POLICY_TYPES}"
    )
```

### 1.3 验收标准

- `uv run pytest` 全部通过（纯新增字段 + 默认值，不破坏现有测试）
- 全局搜索 `next_action_chunk`，所有引用已清理或更新
- 现有 YAML 文件无需修改（新字段都有默认值）

---

## Phase 2: InMemoryBackend 轨迹搜索

**目标**：在 `InMemoryBackend.search()` 中实现轨迹搜索。复用现有的 per-step fusion（RRF / score_sum），在此基础上增加 cross-step 轨迹融合。纯后端层变更，不涉及上层组件。

### 2.1 `src/openpi/cache/backends/in_memory_backend.py`

#### search() 入口分发

现有 `search()` 方法根据 `fusion_method` 分发。新增轨迹搜索分支：

```python
def search(self, spec: QuerySpec) -> list[SearchResultLite]:
    candidates = self._filter_entries(spec)
    if not candidates:
        return []

    active_fields = self._iter_active_fields(spec)
    if not active_fields:
        return []

    # ── 轨迹搜索 ──
    if (spec.trajectory_history is not None
            and spec.trajectory_weights is not None
            and len(spec.trajectory_weights) > 1):
        return self._search_with_trajectory(candidates, spec, active_fields)

    # ── 现有单步搜索（不变） ──
    if spec.fusion_method == "weighted_rrf":
        return self._search_weighted_rrf(candidates, spec, active_fields)
    elif spec.fusion_method == "weighted_score_sum":
        return self._search_weighted_score_sum(candidates, spec, active_fields)
    else:
        return self._search_single_field_cosine(candidates, spec)
```

#### 新增 `_search_with_trajectory()`：两次递归

```python
def _search_with_trajectory(
    self,
    candidates: list[CacheEntry],
    spec: QuerySpec,
    active_fields: list[tuple[str, float, dict[str, Any]]],
) -> list[SearchResultLite]:
    """轨迹搜索：两次递归 + 中间批量打分。

    完全复用现有的 per-step fusion（RRF / score_sum），在此之上增加
    cross-step 加权融合。

    流程：
      Phase A — 第一次递归（收集）：
        从 window 过滤后的候选出发，沿 prev_ids 回溯 depth 层，
        收集每层需要打分的 entry 集合（去重）。

      Phase B — 逐层批量打分：
        对每层的 entry 集合，用配置的 fusion_method（RRF / score_sum）
        批量计算 per-step 分数。RRF 的排名范围 = 该层收集到的 entry 集合。
        结果存入 level_scores[depth][entry_id] = step_score。

      Phase C — 第二次递归（求分）：
        沿相同路径再走一遍，从 level_scores 查表取预计算分数，
        按 trajectory_weights 加权求和，处理分叉（取 max）。

    不做归一化：链条断裂的候选天然得分更低，这是正确的信号。

    RRF 语义说明
    ------------
    当 fusion_method="weighted_score_sum" 时，两遍递归与逐路径递归在数学上等价。
    当 fusion_method="weighted_rrf" 时，每层 RRF 的排名范围是"该层所有从当前候选
    可回溯到的 entries"（per-level reachable-set RRF score），不是某个 entry 的
    绝对相似度。这是一个合理的扩展定义，但与原始单步 RRF 有语义差异，需单独测试。
    """
    history = spec.trajectory_history   # newest-first
    weights = spec.trajectory_weights   # newest-first
    max_depth = len(weights) - 1

    # ── Phase A: 第一次递归，收集每层需要打分的 entry ──
    # level_entries[idx] = set of entry_ids at that depth level
    level_entries: list[set[str]] = [set() for _ in range(len(weights))]
    for entry in candidates:
        self._collect_trajectory_entries(
            entry_id=entry.id,
            depth=max_depth,
            max_depth=max_depth,
            level_entries=level_entries,
            query_history_len=len(history),
            expected_checkpoint_id=spec.checkpoint_id,
        )

    # ── Phase B: 逐层批量打分 ──
    # level_scores[idx][entry_id] = per-step score
    level_scores: list[dict[str, float]] = []
    for idx in range(len(weights)):
        entry_ids = level_entries[idx]
        if not entry_ids:
            level_scores.append({})
            continue

        entries_at_level = [self._entries[eid] for eid in entry_ids if eid in self._entries]
        if not entries_at_level:
            level_scores.append({})
            continue

        # 用配置的 fusion_method 批量打分（复用现有逻辑）
        # active_fields 在 _batch_step_scores 内按当前层 query_keys 重新计算
        scores = self._batch_step_scores(entries_at_level, history[idx], spec)
        level_scores.append(scores)

    # ── Phase C: 第二次递归，查表求轨迹分数 ──
    scored: list[tuple[CacheEntry, float]] = []
    for entry in candidates:
        path_scores = self._score_trajectory(
            entry_id=entry.id,
            depth=max_depth,
            max_depth=max_depth,
            weights=weights,
            accumulated_sim=0.0,
            level_scores=level_scores,
            query_history_len=len(history),
        )
        traj_score = max(path_scores) if path_scores else 0.0
        scored.append((entry, traj_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        SearchResultLite(id=e.id, score=s, checkpoint_id=e.checkpoint_id)
        for e, s in scored[:spec.top_k]
    ]
```

#### 新增 `_collect_trajectory_entries()`（第一次递归）

```python
def _collect_trajectory_entries(
    self,
    entry_id: str,
    depth: int,
    max_depth: int,
    level_entries: list[set[str]],
    query_history_len: int,
    expected_checkpoint_id: CheckpointID | None = None,
) -> None:
    """第一次递归：收集每层需要打分的 entry id。

    checkpoint_id 一致性检查：跳过 checkpoint_id 不匹配的 prev entry，
    防止坏数据或剪枝/合并导致跨 checkpoint 指针污染轨迹分数。
    """
    entry = self._entries.get(entry_id)
    if entry is None:
        return

    # 跳过 checkpoint 不匹配的 entry（防御性检查）
    if expected_checkpoint_id is not None and entry.checkpoint_id != expected_checkpoint_id:
        return

    idx = max_depth - depth
    if idx >= query_history_len:
        return

    level_entries[idx].add(entry_id)

    if depth == 0 or not entry.prev_ids:
        return

    for prev_id in entry.prev_ids:
        self._collect_trajectory_entries(
            prev_id, depth - 1, max_depth,
            level_entries, query_history_len,
            expected_checkpoint_id,
        )
```

#### 新增 `_batch_step_scores()`（逐层批量打分）

```python
def _batch_step_scores(
    self,
    entries: list[CacheEntry],
    query_keys: dict[str, torch.Tensor],
    spec: QuerySpec,
) -> dict[str, float]:
    """对一层的 entry 集合批量计算 per-step 分数。

    复用现有的 per-step fusion 逻辑（RRF / score_sum / single cosine），
    排名范围 = 传入的 entries 集合。

    active_fields 按当前层的 query_keys 重新计算（历史 query 可能缺少
    某些 field），保证与现有单步搜索语义一致。

    返回 {entry_id: step_score}。
    """
    if not entries:
        return {}

    # 构造一个临时 QuerySpec，只替换 query_keys 为当前层的 query
    temp_spec = QuerySpec(
        query_keys=query_keys,
        top_k=len(entries),  # 返回全部，不截断
        checkpoint_id=spec.checkpoint_id,
        fusion_weights=spec.fusion_weights,
        fusion_method=spec.fusion_method,
        field_similarity=spec.field_similarity,
        score_normalization=spec.score_normalization,
        backend_hints=spec.backend_hints,
        # trajectory 字段不传，走单步逻辑
    )

    # 按当前层 query_keys 重新计算 active_fields
    # 历史 query 可能缺少某个 field，不能复用外层的 active_fields
    level_active_fields = self._iter_active_fields(temp_spec)
    if not level_active_fields:
        return {}

    # 复用现有的单步搜索方法
    if spec.fusion_method == "weighted_rrf":
        results = self._search_weighted_rrf(entries, temp_spec, level_active_fields)
    elif spec.fusion_method == "weighted_score_sum":
        results = self._search_weighted_score_sum(entries, temp_spec, level_active_fields)
    else:
        results = self._search_single_field_cosine(entries, temp_spec)

    return {r.id: r.score for r in results}
```

#### 新增 `_score_trajectory()`（第二次递归）

```python
def _score_trajectory(
    self,
    entry_id: str,
    depth: int,
    max_depth: int,
    weights: list[float],
    accumulated_sim: float,
    level_scores: list[dict[str, float]],
    query_history_len: int,
) -> list[float]:
    """第二次递归：查表取预计算分数，加权求和。

    索引映射：idx = max_depth - depth
    当 idx >= query_history_len 时提前终止（与第一遍 _collect 一致），
    避免在历史不足时产生无意义的额外分支。
    返回所有完整路径的非归一化 trajectory_sim 列表。
    """
    entry = self._entries.get(entry_id)
    if entry is None:
        return [accumulated_sim]

    idx = max_depth - depth
    if idx >= query_history_len:
        return [accumulated_sim]
    if idx >= len(level_scores):
        return [accumulated_sim]

    step_score = level_scores[idx].get(entry_id, 0.0)
    accumulated_sim += weights[idx] * step_score

    if depth == 0 or not entry.prev_ids:
        return [accumulated_sim]

    all_paths = []
    for prev_id in entry.prev_ids:
        all_paths.extend(self._score_trajectory(
            prev_id, depth - 1, max_depth,
            weights, accumulated_sim, level_scores,
            query_history_len,
        ))
    return all_paths
```

#### QdrantBackend 守卫

```python
# src/openpi/cache/backends/qdrant_backend.py
# 在 search() 方法开头追加：

if (spec.trajectory_history is not None
        and spec.trajectory_weights is not None
        and len(spec.trajectory_weights) > 1):
    raise NotImplementedError(
        "Trajectory search (trajectory_depth > 1) is not supported in QdrantBackend. "
        "Use InMemoryBackend for trajectory search, or set trajectory_depth=1."
    )
```

#### VectorStoreBackend ABC

```python
# src/openpi/cache/backend_base.py
# search() 的 docstring 中追加说明：

"""
...
Trajectory search
-----------------
If spec.trajectory_history is not None and len(spec.trajectory_weights) > 1,
the backend should compute trajectory-aware similarity for all filtered
candidates in a single pass — no two-phase "initial search + rerank".
Backends that do not support this MUST raise NotImplementedError.
The recursive traversal follows entry.prev_ids to compute cross-step
similarity fusion (see trajectory_search_requirements.log.md section 2.3).
"""
```

### 2.2 验收标准

- 单元测试：构造带 `prev_ids`/`next_ids` 的 CacheEntry 链，验证：
  - `trajectory_history=None` → 行为与修改前完全一致
  - `trajectory_depth=1, weights=[1.0]` → 结果与单步搜索一致
  - `trajectory_depth=3` → 正确回溯、正确加权、正确排序
  - 链条断裂（`prev_ids=[]`）→ 优雅终止，不报错
  - `query_history` 长度不足 → 优雅终止，分数等于只用已有历史的加权和，不产生额外分支
  - **排序反转测试**：构造 A 单步最高但轨迹差、B 单步次高但轨迹好 → 轨迹搜索后 B 排第一
  - **链条断裂惩罚测试**：同一 query，完整链候选得分 > 链断裂候选得分（断裂天然惩罚，不做归一化）
  - **历史 query 缺少某 field**：某层 query_keys 只有 vision_0 没有 robot_state → 不报错，按剩余 field 打分
  - **checkpoint 一致性**：prev_ids 指向不同 checkpoint 的 entry → 被跳过，不参与打分
  - **RRF 轨迹搜索**：`fusion_method="weighted_rrf"` + `trajectory_depth=3` → 正确排序（per-level reachable-set RRF 语义）
  - **score_sum vs RRF 语义差异**：同一数据集，对比两种 fusion_method 的轨迹分数，验证 score_sum 下两遍递归与逐路径递归等价
- QdrantBackend 测试：`trajectory_depth > 1` 时抛 `NotImplementedError`

---

## Phase 3: SearchStrategy 升级

**目标**：为所有现有 SearchStrategy 实现类增加 history buffer 维护、`trajectory_history`/`trajectory_weights` 填入 QuerySpec 的逻辑。

### 3.1 提取公共 Mixin：`TrajectoryMixin`

为避免在三个策略类中重复相同的 buffer 逻辑，提取一个 mixin class：

```python
# src/openpi/cache/components/search_strategy.py 新增

class TrajectoryMixin:
    """公共轨迹 buffer 逻辑，混入到各 SearchStrategy 实现类中。

    提供：
      - history buffer 管理（query_keys + action_chunk）
      - on_episode_start() 生命周期
      - record_action() 广播接收
      - _build_trajectory_fields() 构造 QuerySpec 的轨迹字段
    """

    def _init_trajectory(self, trajectory_depth: int, trajectory_weights: Optional[list[float]]) -> None:
        """在策略类 __init__ 末尾调用。"""
        self._trajectory_depth = trajectory_depth
        self._trajectory_weights = trajectory_weights
        self._query_history: list[dict[str, torch.Tensor]] = []
        self._action_history: list[Optional[torch.Tensor]] = []

    def on_episode_start(self) -> None:
        """清空 history buffer。由 Orchestrator 在 episode 开始时调用。"""
        self._query_history.clear()
        self._action_history.clear()

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """接收 Orchestrator 广播的 action。

        实现约束：纯本地 buffer 操作（append to list）。
        禁止回调 Backend / CacheStorage / Orchestrator 或获取任何外部锁。
        """
        self._action_history.append(action_chunk)

    def record_query_keys(self, query_keys: dict[str, torch.Tensor]) -> None:
        """暂存当前步的 query_keys 到轨迹历史。

        调用时机：
          - search() 内部自动调用（正常搜索路径）
          - Orchestrator 在 gate skip 时显式调用（保证轨迹历史完整）

        实现约束：纯本地 buffer 操作，禁止回调外部组件。
        """
        self._query_history.append(query_keys)

    def _build_trajectory_fields(self) -> dict[str, Any]:
        """返回填入 QuerySpec 的轨迹字段。

        depth=1 或历史为空时返回空 dict（QuerySpec 字段保持 None，退化为单步）。
        """
        if self._trajectory_depth <= 1 or not self._trajectory_weights:
            return {}

        # query_history 是 oldest-first 追加的，需反转为 newest-first
        actual_depth = min(self._trajectory_depth, len(self._query_history))
        if actual_depth <= 1:
            return {}

        history_newest_first = list(reversed(self._query_history[-actual_depth:]))
        weights_newest_first = self._trajectory_weights[:actual_depth]

        return {
            "trajectory_history": history_newest_first,
            "trajectory_weights": weights_newest_first,
        }
```

### 3.2 各策略类修改

以 `WeightedScoreSumKnnStrategy` 为例（其余两个同理）：

```python
class WeightedScoreSumKnnStrategy(TrajectoryMixin):
    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
        score_normalization: Optional[dict[str, Any]] = None,
        # ── 新增 ──
        trajectory_depth: int = 1,
        trajectory_weights: Optional[list[float]] = None,
    ) -> None:
        # ... 现有初始化不变 ...
        self._init_trajectory(trajectory_depth, trajectory_weights)

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        # ── 新增：记录当前步 query_keys ──
        self.record_query_keys(ctx.query_keys)

        filters = _build_step_filters(self._step_filter, self._step_window, ctx)
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_score_sum",
            field_similarity=self._field_similarity,
            score_normalization=self._score_normalization,
            # ── 新增：填入轨迹字段 ──
            **self._build_trajectory_fields(),
        )
        return self._storage.search(spec)
```

**对 `WeightedRrfKnnStrategy` 和 `QdrantWeightedRrfKnnStrategy` 做相同修改**：
1. 继承 `TrajectoryMixin`
2. `__init__` 新增 `trajectory_depth` / `trajectory_weights` 参数，末尾调用 `_init_trajectory()`
3. `search()` 开头调用 `self.record_query_keys(ctx.query_keys)`
4. 构造 `QuerySpec` 时用 `**self._build_trajectory_fields()` 展开轨迹字段

### 3.3 `_build_search_strategy()` 工厂函数更新

```python
# src/openpi/cache/config.py 的 _build_search_strategy() 中，
# 构造各策略实例时传入新参数：

def _build_search_strategy(cfg: SearchStrategyConfig, storage, fusion_weights):
    common_kwargs = {
        # ... 现有参数 ...
        "trajectory_depth": cfg.trajectory_depth,
        "trajectory_weights": cfg.trajectory_weights,
    }
    if cfg.type == "weighted_score_sum_knn":
        return WeightedScoreSumKnnStrategy(storage, **common_kwargs, ...)
    elif cfg.type == "weighted_rrf_knn":
        return WeightedRrfKnnStrategy(storage, **common_kwargs, ...)
    # ...
```

### 3.4 验收标准

- 单元测试：
  - `trajectory_depth=1`（默认）→ `_build_trajectory_fields()` 返回空 dict → QuerySpec 无轨迹字段 → 行为完全不变
  - `trajectory_depth=3` → 前两步 `_build_trajectory_fields()` 返回空 dict（历史不足）→ 第三步开始返回正确的 history 和 weights
  - `on_episode_start()` → buffer 清空
  - `record_action()` → action 正确追加
- 集成测试：与 Phase 2 的 Backend 联合测试完整轨迹搜索

---

## Phase 4: Gate / Judge 预留接口

**目标**：为 Gate 和 Judge 的具体实现类添加 `on_episode_start()` 和 `record_action()` 方法（no-op 实现），带详细注释说明未来扩展方式。**不修改 Protocol 定义**。

### 4.1 `src/openpi/cache/components/gate.py`

```python
class AlwaysSearchGate:
    """Gate that always permits cache search."""

    def __call__(self, checkpoint_id: CheckpointID, cached_data: dict[str, torch.Tensor]) -> bool:
        return True

    # ── 新增：轨迹感知预留接口 ──

    def on_episode_start(self) -> None:
        """清空内部历史 buffer。由 Orchestrator 在 episode 开始时调用。

        当前实现：no-op。AlwaysSearchGate 不维护历史状态。

        未来扩展：trajectory-aware gate 可在此清空 cached_data 历史 buffer，
        用于检测时序一致性（如连续多步观测变化量超过阈值才触发搜索）。
        扩展时在此方法中 clear self._cached_data_history。
        """
        pass

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """接收 Orchestrator 广播的 action。

        实现约束：纯本地 buffer 操作，禁止回调 Backend / CacheStorage /
        Orchestrator 或获取任何外部锁。

        当前实现：no-op。AlwaysSearchGate 不使用 action 数据。

        未来扩展：trajectory-aware gate 可暂存 action 历史，用于
        action-conditioned gate（如检测 action 漂移时强制 cache miss）。
        """
        pass
```

### 4.2 `src/openpi/cache/components/judge.py`

对 `AlwaysHitJudge` 和 `ThresholdJudge` 做相同处理：

```python
class ThresholdJudge:
    # ... 现有 __init__ 和 __call__ 不变 ...

    def on_episode_start(self) -> None:
        """清空内部历史 buffer。由 Orchestrator 在 episode 开始时调用。

        当前实现：no-op。ThresholdJudge 基于单步阈值判断，不维护历史状态。

        未来扩展：trajectory-aware judge 可在此清空历史，用于
        时序一致性校验（如连续多步命中分数趋势下降时判为 miss）。
        """
        pass

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """接收 Orchestrator 广播的 action。

        实现约束：纯本地 buffer 操作，禁止回调 Backend / CacheStorage /
        Orchestrator 或获取任何外部锁。

        当前实现：no-op。ThresholdJudge 不使用 action 数据。

        未来扩展：trajectory-aware judge 可暂存 action 历史，用于
        action 一致性校验（如候选 entry 的历史 action 与 query 侧历史 action
        偏差过大时判为 miss）。
        """
        pass
```

### 4.3 验收标准

- 现有测试全部通过（纯新增方法，不改已有行为）
- 每个预留方法都有详细的 docstring：当前行为、实现约束、未来扩展方式

---

## Phase 5: Orchestrator 重构

**目标**：重构 `CacheOrchestrator`，增加 action 广播、episode 生命周期管理、写入流程从逐步 insert 改为 episode 结束统一写入。

**依赖**：Phase 3（SearchStrategy 有 `on_episode_start()`/`record_action()`）、Phase 4（Gate/Judge 同上）、Phase 6（WritePolicy）。

### 5.1 `src/openpi/cache/orchestrator.py`

#### `__init__` 新增参数

```python
class CacheOrchestrator:
    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gates: dict[CheckpointID, GateFunction],
        judges: dict[CheckpointID, SimilarityJudge],
        search_strategies: dict[CheckpointID, SearchStrategy],
        timer: Optional[SystemTimer] = None,
        # ── 新增 ──
        write_policy: Optional[WritePolicy] = None,
    ) -> None:
        # ... 现有初始化 ...
        self._write_policy = write_policy

        # ── 新增：episode 级 buffer ──
        self._episode_steps: list[StepRecord] = []
        self._miss_by_checkpoint: dict[CheckpointID, int] = {}
        self._current_task_key: str = ""
```

#### `on_task_begin()` 扩展

```python
def on_task_begin(self, task_key: str = "") -> None:
    self._step_counter = 0
    self._current_task_key = task_key
    self._reset_episode_buffer()
    # 通知各组件
    self._broadcast_episode_start()
```

#### `on_episode_start()` 扩展

```python
def on_episode_start(self, task_key: str = "", episode_id: str = "") -> None:
    self._step_counter = 0
    if task_key:
        self._current_task_key = task_key
    self._current_episode_id = episode_id
    self._reset_episode_buffer()
    self._broadcast_episode_start()
```

#### 新增 `_broadcast_episode_start()`

```python
def _broadcast_episode_start(self) -> None:
    """通知所有组件 episode 开始，清空各自的 history buffer。"""
    for strategy in self._search_strategies.values():
        if hasattr(strategy, 'on_episode_start'):
            strategy.on_episode_start()
    for gate in self._gates.values():
        if hasattr(gate, 'on_episode_start'):
            gate.on_episode_start()
    for judge in self._judges.values():
        if hasattr(judge, 'on_episode_start'):
            judge.on_episode_start()
```

#### 新增 `broadcast_action()`

```python
def broadcast_action(self, action_chunk: torch.Tensor) -> None:
    """广播 action 给所有组件。

    由外部调用方在获得 action（无论来自 cache hit 还是模型推理）后调用。
    必须在 check() 返回之后调用（所有锁已释放，不存在死锁风险）。
    """
    for strategy in self._search_strategies.values():
        if hasattr(strategy, 'record_action'):
            strategy.record_action(action_chunk)
    for gate in self._gates.values():
        if hasattr(gate, 'record_action'):
            gate.record_action(action_chunk)
    for judge in self._judges.values():
        if hasattr(judge, 'record_action'):
            judge.record_action(action_chunk)
```

#### `check()` 修改：推理管线调整 + miss 记录

**⚠️ 关键变更：`build()` 上移到 gate skip 判断之前。**

现有顺序：`collect() → gate() → [skip时return] → build() → search() → judge()`
改为：`collect() → gate() → build()（始终执行）→ [skip时 record_query_keys+return] → search() → judge()`

这样 gate skip 时仍有 `query_keys`，可以记录到 SearchStrategy 历史，保证轨迹搜索无空洞。`build()` 只是 CPU 降维，开销远小于搜索。写入 buffer 不在 check() 中发生，由 Interceptor 在 action 产生后调用 `buffer_for_write()`。

```python
def check(self, checkpoint_id: CheckpointID, **stage_outputs) -> CheckResult:
    # ... collect ...
    # ... gate ...

    # ── 变更：build() 始终执行，不再被 gate skip 跳过 ──
    with self._timer.measure(f"{prefix}_build"):
        query_keys = self._key_builder.build(checkpoint_id)

    if not gate_pass:
        # gate skip 时：记录 query_keys 到 SearchStrategy 历史（轨迹搜索完整性）
        # 不调用 buffer_for_write()（action 尚未产生，由 Interceptor 在 action 产生后调用）
        strategy = self._search_strategies.get(checkpoint_id)
        if strategy and hasattr(strategy, 'record_query_keys'):
            strategy.record_query_keys(query_keys)
        self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1
        if checkpoint_id == CheckpointID.CP1:
            self._step_counter += 1
        return CheckResult(hit_type=HitType.MISS, query_keys=query_keys)
        # ↑ 返回 query_keys 供 Interceptor 后续调用 buffer_for_write(query_keys, action)

    # ── search：构造 SearchContext 时传入 task_key ──
    context = SearchContext(
        checkpoint_id=checkpoint_id,
        query_keys=query_keys,
        task_key=self._current_task_key,  # ← 新增：确保 task filter 生效
    )
    # ... strategy.build_query_spec(context) → spec → storage.search(spec) → results ...
    # ... judge(results) → hit_type ...

    if hit_type == HitType.MISS:
        self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1

    result.query_keys = query_keys  # 所有路径都填充，包括 FULL_HIT
    return result
```

#### CheckResult dataclass 变更

```python
@dataclass
class CheckResult:
    hit_type: HitType
    # ... 现有字段不变 ...
    # ── 新增 ──
    query_keys: Optional[dict[str, torch.Tensor]] = None
    # 所有返回路径（gate skip / miss / FULL_HIT）都填充。
    # 供 Interceptor 统一调用 buffer_for_write(query_keys, action)。
```

#### 新增 `buffer_for_write()`

```python
def buffer_for_write(
    self,
    query_keys: dict[str, torch.Tensor],
    action_chunk: torch.Tensor,
) -> None:
    """暂存当前步数据到 episode buffer，episode 结束时统一写入。

    由 Interceptor 在 action 产生后调用（无论来自 cache hit 还是模型推理）。
    不在 check() 中调用——check() 时 action 尚未产生。

    两个参数均为必填：
      - query_keys: 从 check() 返回的 CheckResult.query_keys 获取
      - action_chunk: 从 cache hit payload 或模型推理结果获取
    """
    self._episode_steps.append(StepRecord(
        query_keys=query_keys,
        action_chunk=action_chunk,
    ))
```

#### 新增 `on_episode_end()`

```python
def on_episode_end(self) -> None:
    """episode 结束时调用。根据 WritePolicy 决定是否将暂存数据写入 cache。

    写入流程：
      1. 构造 EpisodeRecord
      2. 调用 WritePolicy.should_write() 判断
      3. 若写入：将暂存的 StepRecord 列表构造为带链表的 CacheEntry 链
      4. batch_insert 一次写入
    """
    if not self._episode_steps:
        return

    if self._write_policy is None:
        return

    record = EpisodeRecord(
        steps=self._episode_steps,
        task_key=self._current_task_key,
        miss_by_checkpoint=dict(self._miss_by_checkpoint),
        total_steps=len(self._episode_steps),
    )

    if not self._write_policy.should_write(record):
        self._reset_episode_buffer()
        return

    # 构造 CacheEntry 链
    entries = self._build_entry_chain(record)
    if entries:
        self._storage.batch_insert(entries)

    self._reset_episode_buffer()
```

#### 新增 `_build_entry_chain()`

```python
def _build_entry_chain(self, record: EpisodeRecord) -> list[CacheEntry]:
    """将 EpisodeRecord 转为带链表的 CacheEntry 列表。"""
    import uuid

    trajectory_id = str(uuid.uuid4())
    entries: list[CacheEntry] = []

    # 第一遍：创建所有 entry（先不设 prev_ids / next_ids）
    for step_idx, step in enumerate(record.steps):
        entry_id = f"{trajectory_id}:{step_idx}"  # 唯一 id，不用 _stable_hash
        entry = CacheEntry(
            id=entry_id,
            checkpoint_id=CheckpointID.CP1,
            query_keys=step.query_keys,
            payload=CachePayload(
                action_chunk=step.action_chunk,
                task_key=record.task_key,
            ),
            step_idx=step_idx,
            trajectory_id=trajectory_id,
        )
        entries.append(entry)

    # 第二遍：串链表
    for i in range(len(entries)):
        if i > 0:
            entries[i].prev_ids = [entries[i - 1].id]
        if i < len(entries) - 1:
            entries[i].next_ids = [entries[i + 1].id]

    return entries
```

#### 新增 `_reset_episode_buffer()`

```python
def _reset_episode_buffer(self) -> None:
    self._episode_steps.clear()
    self._miss_by_checkpoint.clear()
```

#### 删除旧 `write()` / `write_with_keys()`

**直接删除**，不保留旧的逐步写入路径。理由：
- Phase 5b 会将 interceptor 的所有调用方迁移到新路径
- 保留旧接口会产生无链表 entry，混入后降低轨迹搜索效果
- 不存在需要兼容的外部调用方（写入路径完全在 interceptor 内部）

同步删除仅被旧 `write()` 使用的辅助逻辑（如 `_stable_hash()` 若不再需要）。

### 5.2 `on_task_begin()` 签名变更

现有 `on_task_begin()` 无参数。新增可选的 `task_key` 参数：

```python
def on_task_begin(self, task_key: str = "") -> None:
```

调用方如果不传 `task_key`，行为不变（`task_key="""`）。

### 5.3 验收标准

- 单元测试：
  - `on_episode_start()` → 所有组件的 `on_episode_start()` 被调用
  - `broadcast_action()` → 所有组件的 `record_action()` 被调用
  - `buffer_for_write()` + `on_episode_end()` → 正确构造 CacheEntry 链 + batch_insert
  - WritePolicy 为 `never` → 不写入
  - WritePolicy 为 `on_any_miss`，无 miss → 不写入
  - WritePolicy 为 `on_any_miss`，有 miss → 写入
  - 生成的 CacheEntry 链的 `prev_ids`/`next_ids` 正确
- 现有 `write()` / `write_with_keys()` 的测试**迁移**到 `buffer_for_write()` + `on_episode_end()` 新路径

### 5.4 Phase 5b: Interceptor + serve_policy 集成

**目标**：将 Interceptor 和 serve_policy.py 迁移到新的 episode 级写入路径，替代旧的逐步 `_bg_write()`。

#### `src/openpi/cache/interceptor.py` 修改

1. **`on_episode_start(experiment, task, episode_id)`**：
   - 调用 `orchestrator.on_episode_start(task_key=task, episode_id=episode_id)`
   - task 参数需 normalize（沿用 `CachePayload.task_key` 的约定）

2. **CP1 cache hit 早退路径**：
   - `check()` 返回 `FULL_HIT`，`CheckResult.query_keys` 非 None
   - 调用 `orchestrator.broadcast_action(cached_action_chunk)`
   - 调用 `orchestrator.buffer_for_write(result.query_keys, cached_action_chunk)`
   - 返回 cached action，跳过后续 Stage 2/3

3. **正常推理路径（CP1 miss）**：
   - Stage 3 推理完成后获得 `action_chunk`
   - 调用 `orchestrator.broadcast_action(action_chunk)`
   - 调用 `orchestrator.buffer_for_write(result.query_keys, action_chunk)`
   - **删除或旁路**旧的 `_bg_write()` 逐步写入逻辑

4. **`on_episode_end(success)`**：
   - 调用 `orchestrator.on_episode_end()`
   - WritePolicy 在 Orchestrator 内部判断是否写入

#### `scripts/serve_policy.py` 修改

- 构造 `CacheOrchestrator` 时传入 `write_policy=components["write_policy"]`

#### 5b 验收标准

- 端到端集成测试：
  - CP1 miss → 正常推理 → `buffer_for_write()` → `on_episode_end()` → batch_insert 形成链
  - CP1 hit 早退 → 仍 `broadcast_action()` + `buffer_for_write()`
  - `on_episode_start(task_key=...)` 后写入 payload.task_key 非空
- 旧 `_bg_write()` 路径不再被调用

---

## Phase 6: WritePolicy 实现 + Config 工厂

**目标**：实现 WritePolicy Protocol 和三个具体策略，以及 config.py 中的工厂函数。

### 6.1 新增 `src/openpi/cache/components/write_policy.py`

```python
"""Write policy: pluggable switch to decide whether to write trajectory at episode end.

Coupling map:
  DEPENDS ON:  storage_types.py (EpisodeRecord)
  CONSUMED BY: CacheOrchestrator.on_episode_end()
  IF CHANGED:  Orchestrator write decision logic
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openpi.cache.storage_types import EpisodeRecord
from openpi.cache.types import CheckpointID


@runtime_checkable
class WritePolicy(Protocol):
    """Decides whether to write the episode trajectory to cache at episode end."""

    def should_write(self, episode_record: EpisodeRecord) -> bool: ...


class OnAnyMissWritePolicy:
    """Write if the episode had any cache miss or gate skip.

    Rationale: episodes with zero misses are fully covered by existing cache;
    only episodes with novel steps benefit from being stored.
    """

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return episode_record.miss_by_checkpoint.get(CheckpointID.CP1, 0) > 0


class AlwaysWritePolicy:
    """Always write, regardless of cache hit/miss ratio."""

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return True


class NeverWritePolicy:
    """Never write (read-only mode)."""

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return False
```

### 6.2 `src/openpi/cache/config.py` 工厂函数

```python
def _build_write_policy(cfg: WritePolicyConfig):
    """Build WritePolicy from config."""
    from openpi.cache.components.write_policy import (
        AlwaysWritePolicy,
        NeverWritePolicy,
        OnAnyMissWritePolicy,
    )
    if cfg.type == "on_any_miss":
        return OnAnyMissWritePolicy()
    elif cfg.type == "always":
        return AlwaysWritePolicy()
    elif cfg.type == "never":
        return NeverWritePolicy()
    else:
        raise ConfigValidationError(f"Unknown write_policy.type '{cfg.type}'")
```

#### `build_cache_components()` 返回 dict 新增

```python
def build_cache_components(config: CacheConfig) -> dict[str, Any]:
    # ... 现有逻辑 ...
    return {
        # ... 现有 key ...
        "write_policy": _build_write_policy(config.write_policy),
    }
```

#### `build_per_connection_components()` 同步

```python
def build_per_connection_components(...) -> dict[str, Any]:
    # ... 现有逻辑 ...
    return {
        # ... 现有 key ...
        "write_policy": _build_write_policy(config.write_policy),
    }
```

### 6.3 验收标准

- `OnAnyMissWritePolicy`：`miss_by_checkpoint={CP1: 1}` → True，`miss_by_checkpoint={}` → False
- `AlwaysWritePolicy`：始终 True
- `NeverWritePolicy`：始终 False
- YAML 中 `write_policy.type` 为未知值 → `ConfigValidationError`
- `build_cache_components()` 返回的 dict 包含 `write_policy` key

---

## Phase 7: Ingest 脚本更新

**目标**：修改离线 ingest 脚本，在构造 CacheEntry 时填充 `prev_ids` / `next_ids` / `trajectory_id`。

### 7.1 定位 ingest 脚本

需要搜索 `exp/` 目录下的 ingest 脚本（如 `ingest_*.py` 或 `build_artifact.py`），找到构造 CacheEntry 并调用 `batch_insert()` 的位置。

### 7.2 修改逻辑

```python
# 伪代码：在 ingest 脚本中按 episode 构建链表

for episode_id, episode_steps in grouped_by_episode.items():
    entries = []
    for step_idx, step_data in enumerate(sorted(episode_steps, key=lambda s: s.step_idx)):
        entry = CacheEntry(
            id=f"{episode_id}:{step_idx}",
            checkpoint_id=CheckpointID.CP1,
            query_keys=step_data.query_keys,
            payload=CachePayload(
                action_chunk=step_data.action_chunk,
                task_key=step_data.task_key,
            ),
            step_idx=step_idx,
            trajectory_id=episode_id,         # ← 新增
        )
        entries.append(entry)

    # 串链表
    for i in range(len(entries)):
        if i > 0:
            entries[i].prev_ids = [entries[i - 1].id]
        if i < len(entries) - 1:
            entries[i].next_ids = [entries[i + 1].id]

    backend.batch_insert(entries)
```

### 7.3 Artifact 加载兼容

`InMemoryBackend.load_artifact()` 从 HDF5/pickle 加载 entry。需确认：
- 新字段 `prev_ids` / `next_ids` / `trajectory_id` 在旧 artifact 中不存在时，使用默认值（`[]` / `[]` / `None`）
- 新 artifact 格式需包含这三个字段的序列化

### 7.4 验收标准

- Ingest 后的 artifact 中，同 episode 内相邻 entry 的 `prev_ids`/`next_ids` 正确互指
- 所有 entry 的 `trajectory_id` 等于对应的 `episode_id`
- 旧 artifact 加载后新字段为默认值，不报错

---

## 文件变更汇总

| 文件 | Phase | 变更类型 |
|------|-------|----------|
| `src/openpi/cache/storage_types.py` | 1 | 修改：CachePayload 删字段、CacheEntry 加字段、QuerySpec 加字段、新增 StepRecord/EpisodeRecord |
| `src/openpi/cache/config.py` | 1, 6 | 修改：SearchStrategyConfig 加字段、新增 WritePolicyConfig、CacheConfig 加字段、注册表、校验、工厂 |
| `src/openpi/cache/backends/in_memory_backend.py` | 2 | 修改：search() 加轨迹重排序、新增 3 个私有方法 |
| `src/openpi/cache/backends/qdrant_backend.py` | 2 | 修改：search() 加 NotImplementedError 守卫 |
| `src/openpi/cache/backend_base.py` | 2 | 修改：docstring 追加轨迹搜索说明 |
| `src/openpi/cache/components/search_strategy.py` | 3 | 修改：新增 TrajectoryMixin、三个策略类继承 mixin 并传参 |
| `src/openpi/cache/components/gate.py` | 4 | 修改：AlwaysSearchGate 加 on_episode_start() + record_action() |
| `src/openpi/cache/components/judge.py` | 4 | 修改：AlwaysHitJudge + ThresholdJudge 加 on_episode_start() + record_action() |
| `src/openpi/cache/components/write_policy.py` | 6 | **新增**：WritePolicy Protocol + 3 个实现类 |
| `src/openpi/cache/orchestrator.py` | 5a | 修改：删除旧 write/write_with_keys、新增 write_policy 参数、episode buffer、broadcast、on_episode_end、buffer_for_write、`check()` 管线调整（build 上移） |
| `src/openpi/cache/interceptor.py` | 5b | 修改：episode lifecycle 转发、action 广播、buffer_for_write 替代旧 _bg_write |
| `scripts/serve_policy.py` | 5b | 修改：构造 Orchestrator 时传入 write_policy |
| `exp/` ingest 脚本 | 7 | 修改：构造 CacheEntry 时填充链表字段 |

**新增文件数**：1（`write_policy.py`）
**修改文件数**：12

---

## 风险与注意事项

1. **`next_action_chunk` 删除的影响范围**：Phase 1 中需全局搜索此字段的所有引用（包括测试、文档、序列化逻辑），遗漏会导致运行时 AttributeError
2. **旧 Artifact 兼容**：`InMemoryBackend.load_artifact()` 必须用 `getattr` 补齐旧 entry 缺少的 `prev_ids=[]` / `next_ids=[]` / `trajectory_id=None`
3. **递归深度**：`_collect_trajectory_entries()` 和 `_score_trajectory()` 的递归深度 = `trajectory_depth`。LIBERO episode 通常几百步但 trajectory_depth 一般不超过 10，不会触发 Python 递归限制
4. **性能**：轨迹搜索采用两次递归 + 逐层批量打分设计。第一次递归收集每层 entry 集合（开销 ≈ 遍历链表），`_batch_step_scores()` 逐层复用现有 fusion 方法批量打分，第二次递归仅查表求和。depth 通常 2-5，InMemoryBackend 本来就是全量遍历，新增开销可接受
5. **推理管线变更**：`check()` 中 `build()` 上移到 gate skip 之前（见 Phase 5a），gate skip 不再跳过 `build()`，增加少量 CPU 开销但保证轨迹完整
6. **CP3 链表替代路径未实现**：删除 `next_action_chunk` 不等于 CP3 已被链表替代。未来启用 CP3 需新增 `fetch_entry_metadata(id)` API。本次不实现
7. **Entry ID 语义变更**：轨迹 entry 使用 `trajectory_id:step_idx` 而非 `_stable_hash`，不再支持幂等 upsert。如需语义去重可新增 `semantic_key` 字段
8. **Trajectory 启用后 Judge threshold 需重新校准**：`_batch_step_scores()` 复用配置的 fusion_method，但轨迹最终分数是 `sum(weights[i] * step_score_i)`，scale 与单步搜索不同。启用 trajectory 后 ThresholdJudge 的阈值必须重新校准。注意：不要复用单步 `weighted_score_sum` 的 percentile calibration 数据——轨迹分数的分布不同

---

## 代码计划审查问题与建议（2026-04-07）

> Reviewer: Codex
>
> 结论：当前 plan 的总体方向合理，但不建议直接按现版本实现。下面问题需要先澄清或纳入 plan 修订，否则可能出现轨迹链写坏、运行时没有真正切到 episode 级写入、轨迹 rerank 分数语义与现有策略不一致等问题。

### P0 阻塞问题

1. **Entry ID 方案与轨迹链表语义冲突**
   - 当前 Phase 5 `_build_entry_chain()` 仍使用 `_stable_hash(CheckpointID.CP1, step.query_keys)` 作为 `CacheEntry.id`。
   - 现有 `_stable_hash` 的语义是“同 checkpoint + 同 query key 得到相同 id”，用于幂等 upsert；这适合单步 cache，但不适合作为轨迹节点 id。
   - 风险：同一个 observation 在多个 episode 或同一 episode 多次出现时会写到同一个 id，导致 `prev_ids` / `next_ids` 指向被覆盖或交叉污染，轨迹不再是一条真实 episode 链。
   - 需要确认：轨迹 entry 到底要“语义去重”还是“每次出现都是一个独立节点”？
   - 建议：轨迹节点 `id` 改为唯一 id，例如 `trajectory_id:step_idx` / UUID / episode 文件名 + step；如仍需语义去重，新增 `semantic_key` 字段而不要复用 `id`。

2. **漏掉 `InferenceInterceptor` 和 `serve_policy.py` 的运行时集成**
   - 当前文件变更汇总没有 `src/openpi/cache/interceptor.py` 和 `scripts/serve_policy.py`。
   - 但 action 产生、CP1 hit 早退、post-inference 写入、episode lifecycle 转发都在 `InferenceInterceptor` 中。
   - 风险：即使 Orchestrator 加了 `buffer_for_write()` / `broadcast_action()` / `on_episode_end()`，实际运行时也不会调用这些新接口，仍然会走旧的后台逐步 `write_with_keys()` 路径。
   - 建议：Phase 5 增加明确的 Interceptor 改动：
     - `on_episode_start(experiment, task, episode_id)` 将 `task` / `episode_id` 传给 Orchestrator。
     - `on_episode_end(success)` 调用 `orchestrator.on_episode_end()`。
     - CP1 cache hit 早退前也调用 `broadcast_action()`，并根据需要 `buffer_for_write()`。
     - Stage3 正常推理后调用 `broadcast_action(stage3.action_chunk[0])` 和 `buffer_for_write(cp1_query_keys, action_chunk)`，替代或旁路旧 `_bg_write()`。
     - `serve_policy.py` 构造 `CacheOrchestrator` 时传入 `components["write_policy"]`。

3. **`task_key` 没有真实数据流**
   - Plan 改 `on_task_begin(task_key="")`，但当前 WebSocket 生命周期里的任务信息来自 `on_episode_start(experiment, task, episode_id)`。
   - 现有 `SearchContext` 已有 `task_key`，但 Orchestrator 当前构造时未传入，导致 `step_filter` 之外的 task filter 实际不起作用。
   - 风险：`EpisodeRecord.task_key` 为空，运行时写入 payload 的 `task_key` 为空；搜索也不会按任务过滤，跨任务误匹配风险上升。
   - 建议：以 `on_episode_start(..., task, episode_id)` 作为 episode 级 task 来源，明确 task normalize 规则；Orchestrator 在构造 `SearchContext` 时传 `task_key=self._current_task_key`。

4. **轨迹 rerank 候选池太小**
   - Phase 2 计划先用现有 `search()` 返回的 `top_k` 结果，再对这 `top_k` 做 trajectory rerank。
   - 风险：当 `top_k=1` 时，轨迹搜索没有机会把“当前步略差但历史非常一致”的候选拉上来，等价于只给单步 top-1 改分数。
   - 需求文档 Q7 说可用现有 `candidate_multiplier`，但 plan 没有把它接入 InMemoryBackend 的初始候选池。
   - 建议：trajectory 模式初始候选数使用 `top_k * candidate_multiplier` 或新增 `trajectory_candidate_k`；InMemory 的 `WeightedRrfKnnStrategy` / `WeightedScoreSumKnnStrategy` 也需要接收并传递 `candidate_multiplier`。

5. **`_compute_step_similarity()` 没有真正复用现有 per-step fusion 语义**
   - Phase 2 伪代码对单个 entry 做 field similarity 加权平均。
   - 但现有 `weighted_score_sum` 会做 cosine 到 `[0,1]` 的转换、L2 `exp(-d/tau)` 转换、可选 percentile normalization；现有 `weighted_rrf` 是候选集合上的 rank fusion，不是单 entry 加权平均。
   - 风险：轨迹 rerank 的 step score 与初始 search score 语义不一致，Judge threshold 和排序行为会变得难解释。
   - 建议：先明确 trajectory search 是否只支持 `weighted_score_sum`。如果要支持 `weighted_rrf`，需要定义每个历史层的候选集合和 rank 计算规则，不能简单用单 entry cosine 替代 RRF。

6. **trajectory score scale 未归一化，threshold 不稳定**
   - 当前递归累加 `sum(weights[i] * step_sim_i)`。
   - 风险：历史不足、链条断裂、不同 depth 或不同 weight sum 会改变分数上界；`ThresholdJudge` 的 threshold 无法直接复用，也不容易校准。
   - 建议：按实际参与计算的 `sum(active_weights)` 做归一化，或在 plan 中明确 trajectory score 是非归一化累积分数，必须重新标定 Judge threshold。

7. **Gate skip 时 query history 不会记录**
   - Plan 在 `SearchStrategy.search()` 开头 `_record_query_keys(ctx.query_keys)`。
   - 但当 Gate 返回 False 时，Orchestrator 当前不会 build keys，也不会调用 SearchStrategy。
   - 风险：需求要求“每个 step 生成的 query key 暂存”，但 gate skip 的 step 会缺失，导致 query trajectory history 不完整。
   - 建议：如果 gate skip 也要参与历史，Orchestrator 需要在 gate skip 后仍 build 并记录 query keys，或新增明确的 `SearchStrategy.record_query_keys()` 由 Orchestrator 调用；如果不记录 gate skip，需要在需求和 plan 中声明这个行为。

### P1 设计缺口

8. **删除 `next_action_chunk` 的替代路径不完整**
   - Plan 删除 `CachePayload.next_action_chunk`，并认为可通过 `next_ids[0] -> fetch entry -> payload.action_chunk` 获取下一步 action。
   - 但当前 `CacheStorage.fetch_payload()` 只能按 id 拿 payload；search 结果 `SearchResultLite` 也不带 `next_ids`；Qdrant payload 目前不序列化 `next_ids`。
   - 风险：CP3 或未来 predictive skip 无法从 winner 获取下一步 entry id。
   - 建议：本阶段若 CP3 仍 suspended/disabled，应明确“删除 `next_action_chunk` 不等于 CP3 已被链表替代”；如果要实现替代路径，需要新增 `fetch_entry()` 或至少 `fetch_entry_metadata()` API，并在 InMemory/Qdrant 后端序列化链表字段。

9. **Qdrant / artifact 链表字段序列化不充分**
   - Phase 2 只要求 Qdrant 在 trajectory depth > 1 时抛 `NotImplementedError`，但 Phase 5/7 可能仍通过 Qdrant 写入带链表字段的 entry。
   - 当前 Qdrant `_to_point()` payload 只写 `checkpoint_id`、`timestamp`、`task_key` 和 payload tensors，不包含 `step_idx`、`prev_ids`、`next_ids`、`trajectory_id`。
   - InMemory pickle 旧 artifact unpickle 出来的 `CacheEntry` 也可能缺少新增属性，单靠 dataclass 默认值不一定自动补齐旧对象。
   - 建议：即使 Qdrant 暂不支持 trajectory search，也要明确是否支持链表字段持久化；`InMemoryBackend.load_artifact()` 应对旧 entry 用 `getattr` 补齐 `prev_ids=[]` / `next_ids=[]` / `trajectory_id=None`。

10. **`StepRecord.action_chunk: Optional` 与 `CachePayload` 校验冲突**
    - Phase 1 定义 `StepRecord.action_chunk: Optional[torch.Tensor] = None`，但 `CachePayload.validate_for_checkpoint()` 要求所有 checkpoint 都有 `action_chunk`。
    - 风险：`buffer_for_write()` 接收 None 后，`on_episode_end()` 构造 CP1 `CachePayload(action_chunk=None)`，`batch_insert()` 校验失败。
    - 建议：如果 CP1 trajectory 写入必须有 action，就把 `StepRecord.action_chunk` 改为非 Optional 并在 `buffer_for_write()` fail-fast；如果允许 action 缺失，需要同步调整 payload validation 和 cache hit 行为。

11. **Phase 5 的 `_episode_had_miss` 语义需要限定 checkpoint**
    - Plan 在 gate skip / judge miss 时直接设置 `_episode_had_miss=True`。
    - 风险：CP3 当前是 skeleton 或可能长期 miss，会导致 `on_any_miss` 几乎总是写入，即使 CP1 全命中。
    - 建议：`had_miss` 至少区分 checkpoint，例如只统计 CP1 miss，或 `EpisodeRecord` 改成 `miss_by_checkpoint: dict[CheckpointID, int]`，`OnAnyMissWritePolicy` 明确使用哪些 checkpoint。

12. **SearchStrategy history buffer 与 Orchestrator write buffer 可能重复且不同步**
    - SearchStrategy 维护 `_query_history` / `_action_history`，Orchestrator 也维护 `_episode_steps`。
    - 风险：CP1 hit 早退、gate skip、CP3 check、旧写入路径保留时，这两个 buffer 可能长度不同或内容不同；`record_action()` 在 SearchStrategy 中暂存 action，但本次搜索并不使用 action。
    - 建议：明确两个 buffer 的职责边界：SearchStrategy buffer 只用于 query trajectory search；Orchestrator buffer 只用于写入。并在测试中覆盖 CP1 hit 早退、gate skip、normal miss 三条路径的 buffer 长度一致性。

13. **保留旧 `write()` / `write_with_keys()` 会长期产生无链表 entry**
    - Plan 为兼容保留旧路径，但 trajectory search 对旧路径写入的数据只能当链条断裂处理。
    - 风险：运行时如果没有完全迁移到 `buffer_for_write()`，数据库会混入大量 `prev_ids=[]` 的单节点 entry，轨迹搜索收益不明显且难排查。
    - 建议：短期保留旧 API，但在 docstring/log 中标记为 legacy single-step write；如果 trajectory_depth > 1 且仍调用旧写入路径，建议 warning。

### P2 测试与验收建议

14. **增加端到端测试，而不只测 Orchestrator**
    - 当前 Phase 5 验收集中在 Orchestrator 单元测试，但真正的数据流跨 `InferenceInterceptor`。
    - 建议新增测试：模拟一次 CP1 miss 正常推理后 `buffer_for_write()`，episode_end 后 batch_insert 形成链；模拟 CP1 hit 早退后仍广播 action；验证 `on_episode_start(task=...)` 后写入 payload.task_key 非空。

15. **增加 trajectory rerank 反转排序测试**
    - 需要构造一个场景：单步 top-1 是 A，但 depth=3 轨迹分数让 B 胜出。
    - 这个测试能防止“只 rerank 已经返回的 top_k”导致 trajectory search 没有效果。

16. **增加分数归一化测试**
    - 对相同 query，在 history 不足、完整 history、链条断裂三种情况下验证 score scale 是否符合预期。
    - 如果采用 active weight normalization，应测试相同有效匹配在不同可用历史长度下分数可比较。

17. **补充配置校验**
    - `trajectory_depth >= 1`。
    - `trajectory_weights` 中是否允许负数或全 0，需要明确；建议要求非负且总和 > 0。
    - 当 backend.type == `qdrant` 且任一 checkpoint `trajectory_depth > 1` 时，是配置阶段报错，还是运行时 QdrantBackend 抛 `NotImplementedError`？建议配置阶段 fail-fast。

### 建议的 plan 修订顺序

1. 先修订 Phase 1：确定 trajectory entry id 语义、`StepRecord.action_chunk` 是否 Optional、score 归一化规则、配置校验规则。
2. 再修订 Phase 2：明确 `weighted_score_sum` / `weighted_rrf` 的 trajectory per-step scoring 语义，并加入初始候选池扩大机制。
3. 再修订 Phase 5：拆成 `Orchestrator core` 与 `InferenceInterceptor / serve_policy integration` 两个子阶段。
4. 最后修订 Phase 7：明确 `exp/build_in_memory_cache_artifact.py` 是当前 InMemory artifact 脚本；Qdrant ingest 是否纳入本次范围要单独说明。

---

## 审查回复（2026-04-07）

### P0 阻塞问题回复

#### #1 Entry ID 方案与轨迹链表语义冲突

**承认，这是真实 bug。**

`_stable_hash(CP1, query_keys)` 是语义去重 key——同一观测在不同 episode 出现时 id 相同，upsert 会覆盖前一个 entry 的 `prev_ids`/`next_ids`，链表交叉污染。

**修改方案**：
- 轨迹节点 id 改为 `f"{trajectory_id}:{step_idx}"`，保证每次写入的每个节点全局唯一
- 如未来需要语义去重（"这个观测是否见过"），新增 `semantic_key: Optional[str]` 字段（用现有 `_stable_hash` 计算），不复用 `id`
- 离线 ingest 同理改用 `trajectory_id:step_idx` 作为 id
- **影响 Phase**：Phase 1（CacheEntry 可选新增 `semantic_key`）、Phase 5（`_build_entry_chain()` 改 id 生成）、Phase 7（ingest 改 id 生成）

#### #2 漏掉 InferenceInterceptor 和 serve_policy.py

**承认。**

Plan 只改了 Orchestrator 内部，但实际调用链在 `interceptor.py` 和 `serve_policy.py`。不改这两个文件，新接口不会被调用，仍走旧的后台逐步 `write_with_keys()` 路径。

**修改方案**：Phase 5 拆为两个子阶段：
- **Phase 5a — Orchestrator core**：现有 Phase 5 内容不变
- **Phase 5b — Interceptor + serve_policy 集成**：
  - `interceptor.py`：
    - `on_episode_start()` 传 task/episode_id → Orchestrator
    - CP1 hit 早退前调 `broadcast_action()` + `buffer_for_write()`
    - Stage3 正常推理后调 `broadcast_action(action_chunk)` + `buffer_for_write(cp1_query_keys, action_chunk)`
    - `on_episode_end()` 调 `orchestrator.on_episode_end()`
    - 以上替代或旁路旧 `_bg_write()` 路径
  - `serve_policy.py`：构造 Orchestrator 时传入 `components["write_policy"]`
- **文件变更汇总新增**：`src/openpi/cache/interceptor.py`、`scripts/serve_policy.py`

#### #3 task_key 没有真实数据流

**承认。**

现有 Orchestrator 构造 SearchContext 时 `task_key` 为空，是已有的 gap。

**修改方案**：
- `on_episode_start(task_key: str = "", episode_id: str = "")` 接收 task 信息，存到 `self._current_task_key` 和 `self._current_episode_id`
- Orchestrator 构造 SearchContext 时传 `task_key=self._current_task_key`
- task normalize 规则：沿用现有 `CachePayload.task_key` 的约定（canonical task identifier，非原始 prompt）
- 在 Phase 5a 中完成

#### #4 轨迹 rerank 候选池太小

**驳回。问题源于 plan 的错误设计，而非需求本身。**

Reviewer 的问题基于 plan 中的两阶段设计（先单步搜索截断到 top_k，再对截断结果做轨迹 rerank）。但这个两阶段设计本身就是 plan 偏离需求文档的错误——需求文档从未要求两阶段。

InMemoryBackend 本来就是全量遍历所有 filtered entries，不存在 ANN 索引的截断问题。候选池就是全量 filtered entries，不需要 `candidate_multiplier`。

**修改方案**：Phase 2 已修订为两次递归设计：
1. 第一次递归沿 `prev_ids` 收集每层需要打分的 entry 集合（去重）
2. 逐层批量打分：`_batch_step_scores()` 复用配置的 fusion_method（RRF / score_sum / single cosine），排名范围 = 该层 entry 集合
3. 第二次递归查表取预计算分数，按 `trajectory_weights` 加权求和

详见 Phase 2 正文。

#### #5 `_compute_step_similarity()` 没有真正复用现有 per-step fusion 语义

**承认，已在 Phase 2 中彻底解决。**

新设计不再自行实现 per-step similarity 计算，而是通过 `_batch_step_scores()` **完全复用**现有的 per-step fusion 方法：
- 构造临时 QuerySpec（只替换 `query_keys` 为当前层的历史 query，不传 trajectory 字段以走单步逻辑）
- 根据配置的 `fusion_method` 分发到现有的 `_search_weighted_rrf()` / `_search_weighted_score_sum()` / `_search_single_field_cosine()`
- RRF 的排名范围 = 该层收集到的 entry 集合（两次递归的第一次递归已提前收集了每层的 entry 集合，为 RRF 提供了候选集排名基础）

因此不存在"轨迹搜索用 score sum，单步用 RRF"的语义不一致问题——**轨迹搜索的每一层 per-step 分数与单步搜索使用完全相同的 fusion 语义**。

#### #6 trajectory score scale 未归一化

**驳回，不做归一化。**

- **query 侧历史不足**（episode 前几步）：所有候选的 `active_weights_sum` 相同（同一个 `query_history`），归一化不改变排序
- **链条断裂**（某候选 `prev_ids=[]`）：链断了说明轨迹信息不完整，得分低于完整链是**正确的信号**，归一化反而会抹掉这个有用信号
- **ThresholdJudge 阈值**：部署时 `trajectory_depth` 是固定的，阈值按该 depth 下的分数 scale 校准即可，不需要跨不同 depth 可比

trajectory score 使用非归一化的加权累积分数 `sum(weights[i] * step_sim_i)`。

#### #7 Gate skip 时 query history 不会记录

**承认，需要修改推理管线。**

现有 `check()` 流程中，gate skip 发生在 `key_builder.build()` **之前**（`orchestrator.py:166-169`），gate skip 时 `query_keys` 未构建。但写入完整轨迹需要每步都有 `query_keys`，否则链表有空洞。

**修改方案**：gate skip 时**仍然调用 `build()`**，保证每步都有 `query_keys`。

**推理管线调整**（Phase 5a，`orchestrator.py` 的 `check()` 方法）：

现有顺序：
```
collect() → gate() → [skip时直接return] → build() → search() → judge()
```

改为：
```
collect() → gate() → build()（无论 gate 结果都执行）→ [skip时 buffer + return] → search() → judge()
```

- `build()` 从 gate skip 判断**之后**移到**之前**（上移几行）
- gate skip 的 `return` 下移到 `build()` 之后
- gate skip 时：`check()` 内调用 `strategy.record_query_keys(query_keys)`（轨迹搜索历史完整性）；`buffer_for_write()` 不在 `check()` 中调用，由 Interceptor 在 action 产生后调用
- `gate()` 用的是 `cached_data`（GPU），不依赖 `query_keys`，顺序调整无副作用
- `build()` 只是 CPU 降维，开销远小于搜索，gate skip 仍省下了搜索的开销

这样：
- query history 每步完整，轨迹搜索不出现空洞
- 写入 buffer 每步完整，episode 结束写入的链表无空洞
- Gate 的 `cached_data` 历史由 gate 在 `__call__` 中自行积累（gate 在 `build()` 之前已被调用）

### P1 设计缺口回复

#### #8 删除 next_action_chunk 的替代路径不完整

**部分承认。**

Reviewer 说得对：当前 `fetch_payload()` 返回 payload 不带 `next_ids`，删了 `next_action_chunk` 后 CP3 确实没有替代路径。但需求文档已明确"CP3 向前滚动的具体逻辑不在本次设计范围内"。

**修改方案**：在 plan 风险清单中补充说明：
- 删除 `next_action_chunk` 是消除冗余，不是声称 CP3 已被链表替代
- CP3 当前是 skeleton/disabled 状态，删除不影响运行时
- 未来启用 CP3 时需新增 `fetch_entry_metadata(id) -> CacheEntry`（不含 payload tensor）或在 `SearchResultLite` 中返回 `next_ids`
- 本次不实现该 API

#### #9 Qdrant / artifact 链表字段序列化

**承认。**

旧 pickle artifact unpickle 出来的对象不会自动获得新 dataclass 字段。

**修改方案**：
- `InMemoryBackend.load_artifact()` 中加 `getattr` 兜底：

```python
for entry in loaded_entries:
    entry.prev_ids = getattr(entry, 'prev_ids', [])
    entry.next_ids = getattr(entry, 'next_ids', [])
    entry.trajectory_id = getattr(entry, 'trajectory_id', None)
```

- Qdrant payload 序列化暂不纳入本次范围（Qdrant 不支持 trajectory search，链表字段不需要持久化），在 plan 中明确声明
- 新 artifact 构建脚本（Phase 7）自然包含这些字段

#### #10 StepRecord.action_chunk: Optional 与 CachePayload 校验冲突

**承认。**

`CachePayload.validate_for_checkpoint()` 要求 `action_chunk is not None`，但 `StepRecord` 允许 None 会在 `_build_entry_chain()` 中构造出非法 payload。

**修改方案**：
- `StepRecord.action_chunk` 改为**非 Optional**：`action_chunk: torch.Tensor`
- `buffer_for_write()` 签名改为 `buffer_for_write(query_keys, action_chunk)` 两个都是必传参数
- 调用时如果 action_chunk 尚未产生则不调用 `buffer_for_write()`（等 action 产生后再调用）

#### #11 `_episode_had_miss` 语义需要限定 checkpoint

**承认。**

CP3 是 skeleton 永远 miss，会导致 `on_any_miss` 总是写入。

**修改方案**：`EpisodeRecord` 改为：

```python
@dataclass
class EpisodeRecord:
    steps: list[StepRecord]
    task_key: str
    miss_by_checkpoint: dict[CheckpointID, int]  # e.g. {CP1: 3, CP3: 50}
    total_steps: int
```

`OnAnyMissWritePolicy` 默认只看 CP1 的 miss 数 > 0。未来可配置关注哪些 checkpoint。

#### #12 SearchStrategy buffer 与 Orchestrator buffer 重复

**驳回，这是 by design 不是 bug。**

两个 buffer 职责不同，记录时机不同：

- **SearchStrategy `_query_history`**：用于轨迹搜索。`check()` 内部记录——正常搜索时在 `search()` 中自动调用 `record_query_keys()`，gate skip 时由 Orchestrator 显式调用 `record_query_keys()`。**每步都记录**（因为 `build()` 已上移到 gate skip 之前）
- **Orchestrator `_episode_steps`**：用于 episode 结束写入。**不在 `check()` 中记录**，由 Interceptor 在 action 产生后调用 `buffer_for_write(query_keys, action_chunk)`。**每步都记录**（Interceptor 无论 cache hit 还是模型推理都会产生 action）

**不合并 buffer。** 在测试中覆盖：
- CP1 hit 早退：SearchStrategy 在 `search()` 中记录 query_keys；Interceptor 拿到 cache action 后调 `buffer_for_write()`
- Gate skip：SearchStrategy 由 Orchestrator 显式记录 query_keys；Interceptor 拿到模型推理 action 后调 `buffer_for_write()`
- Normal miss：同 gate skip

#### #13 保留旧 write() 会长期产生无链表 entry

**承认。直接删除旧 API，不保留。**

Phase 5b 已经要改 interceptor 把所有调用方迁移到新路径（`buffer_for_write()` + `on_episode_end()`），没有必要保留会产生无链表 entry 的旧接口。

**修改方案**：
- Phase 5a 中**删除** `Orchestrator.write()` 和 `Orchestrator.write_with_keys()`
- Phase 5b 中 interceptor 全部改用 `buffer_for_write()` + `broadcast_action()` + `on_episode_end()` 新路径
- 测试中所有使用旧 `write()` 的用例迁移到新路径
- 同步清理 `_stable_hash()` 如果只有旧 `write()` 在用（需确认 `_build_entry_chain()` 是否还需要）

### P2 测试与验收建议回复

#### #14 增加端到端测试

**承认。**

Phase 5b 验收标准新增端到端集成测试：
- 模拟 CP1 miss → 正常推理 → `buffer_for_write()` → `on_episode_end()` → batch_insert 形成链
- 模拟 CP1 hit 早退 → 仍 `broadcast_action()`
- 验证 `on_episode_start(task_key=...)` 后写入 payload.task_key 非空

#### #15 增加 trajectory rerank 反转排序测试

**承认，非常好的测试建议。**

Phase 2 验收标准新增：构造 A 单步 top-1 但轨迹差、B 单步 top-2 但轨迹好的场景，验证 rerank 后 B 排第一。

#### #16 增加分数归一化测试

**承认。**

Phase 2 验收标准新增：验证非归一化分数下完整链 > 断裂链；history 不足时只比较排序正确性，不要求跨不同历史长度同 scale。

#### #17 补充配置校验

**承认。**

Phase 1 的 `validate_cache_config()` 新增：
- `trajectory_depth >= 1`
- `trajectory_weights` 中所有值非负且总和 > 0
- 当 `backend.type == "qdrant"` 且任一 checkpoint `trajectory_depth > 1` 时，**配置阶段报错**（fail-fast），不等到运行时

### 建议的 plan 修订顺序

**承认 reviewer 建议的修订顺序合理。**
