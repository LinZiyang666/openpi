# Step 4 Config 配置系统 — 实现计划

> 日期：2026-04-04
> 状态：Plan
> 前置讨论：[step4_config_discussion.log.md](step4_config_discussion.log.md)

---

## 0. 概述

基于讨论阶段的所有决议 + 答辩修订，构建 cache 配置系统 + SearchStrategy 组件。完成后：

- `serve_policy.py` 保留现有 tyro CLI 主路径，新增 `--cache_config cache.yaml` 可选参数
- Cache 配置从 YAML 文件读取，支持 anchor 和环境变量替换
- SearchStrategy 成为 Orchestrator 第五个可插拔组件，通过 SearchContext 接收运行时信息
- CP1/CP3 可独立配置 gate、judge、search_strategy
- keys 统一配置，分发到 KeyBuilder + SearchStrategy（当前受 key_builder.type 限制，校验层保证语义不落空）
- InMemoryBackend 提升为正式代码，支持不依赖 Qdrant 的开发调试

---

## 1. 新增文件清单

| 文件 | 说明 |
|------|------|
| `src/openpi/cache/components/search_strategy.py` | SearchStrategy Protocol + SearchContext + SimpleKnnStrategy 实现 |
| `src/openpi/cache/backends/in_memory_backend.py` | InMemoryBackend 正式实现（从 conftest 提升） |
| `src/openpi/cache/config.py` | CacheConfig dataclass 树 + YAML 加载 + 校验 + 组件工厂 |
| `cache.yaml` | 项目根目录，cache 默认配置文件（全量注释） |
| `tests/cache/test_config.py` | Config 加载 + 校验 + 工厂 + 组装集成测试 |
| `tests/cache/test_search_strategy.py` | SearchStrategy 单元测试 |

## 2. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/openpi/cache/orchestrator.py` | 接入分检查点 gate/judge/search_strategy dict + step counter + SearchContext 构造 |
| `src/openpi/cache/storage_types.py` | QuerySpec 扩展 fusion_weights + backend_hints |
| `src/openpi/cache/interceptor.py` | on_task_begin() 转发给 orchestrator（重置 step_counter） |
| `scripts/serve_policy.py` | 保留 tyro CLI，新增 `--cache_config` 可选参数 |
| `tests/cache/conftest.py` | InMemoryBackend 改为 import 正式路径 |
| `tests/cache/test_orchestrator.py` | 适配 Orchestrator 新签名 |
| `tests/cache/test_interceptor.py` | 适配 Orchestrator 新签名 |
| `claude_log/README.md` | 新增 plan 条目 |

---

## 3. 实现步骤

### Phase 1：SearchStrategy 组件（独立于 Config）

**目标**：新增可插拔的搜索策略组件，从 Orchestrator 中抽离搜索逻辑。

#### 3.1.1 扩展 QuerySpec（`storage_types.py`）

QuerySpec 新增 fusion 参数字段，使 SearchStrategy 构造的 spec 能携带 fusion 信息下传到 Backend。

**模块级 Coupling map 更新**（在现有 docstring 中追加）：
```
Coupling map:
  DEPENDS ON:  types.py (CheckpointID)
  CONSUMED BY: KeyBuilder (CachePayload, CacheEntry construction),
               Orchestrator (SearchResultLite, SearchResult, CachePayload),
               SearchStrategy (QuerySpec construction — NEW),
               CacheStorage (all types), backends (all types)
  IF CHANGED:  SearchStrategy QuerySpec construction,
               Backend.search() fusion parameter reading,
               Orchestrator write path (CacheEntry unchanged)
```

**QuerySpec 变更**：
```python
@dataclass
class QuerySpec:
    """Everything a backend needs to execute one search.

    Data flow: SearchStrategy -> QuerySpec -> CacheStorage.search() -> Backend.search()

    Fusion parameters
    -----------------
    fusion_weights is a generic concept (any multi-vector backend may need weighted
    fusion). Backend-specific parameters (e.g. Qdrant's rrf_k, candidate_multiplier)
    are passed via backend_hints — backends read what they recognise, ignore the rest.

    Coupling:
      - CONSTRUCTED BY: SearchStrategy (the only constructor after this change)
      - CONSUMED BY: CacheStorage.search() (validation), Backend.search() (execution)
      - IF CHANGED: SearchStrategy construction logic, Backend.search() parameter reading
    """
    query_keys: dict[str, torch.Tensor]
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None
    # --- 新增：通用 fusion 参数 ---
    fusion_weights: Optional[dict[str, float]] = None     # 各字段融合权重（backend-agnostic）
    # --- 新增：backend-specific 参数通道 ---
    backend_hints: Optional[dict[str, Any]] = None        # e.g. {"rrf_k": 60, "candidate_multiplier": 5}
```

**影响范围**：
- QdrantVectorStore.search() 从 `spec.fusion_weights` 读权重，从 `spec.backend_hints` 读 rrf_k/candidate_multiplier
- InMemoryBackend 忽略 fusion_weights 和 backend_hints（单向量 cosine 搜索）
- FAISS 等未来 backend 可从 backend_hints 读取自己的参数，不污染 QuerySpec 公共字段

#### 3.1.2 新建 SearchStrategy Protocol + SimpleKnnStrategy（`components/search_strategy.py`）

```python
"""SearchStrategy: the single exit point for database search.

Overview
--------
SearchStrategy encapsulates all search parameters (top_k, fusion weights,
step filtering, etc.) and is the ONLY component that constructs QuerySpec
and calls CacheStorage.search().  This decouples search logic from both
Orchestrator (pure orchestration) and Backend (pure KNN execution).

Data flow: SearchContext (from Orchestrator) -> SearchStrategy.search()
             -> QuerySpec (with fusion params + backend_hints) -> CacheStorage.search()
             -> Backend.search() -> list[SearchResultLite]

Coupling map:
  DEPENDS ON:  CacheStorage.search() (Step 3 facade),
               storage_types (QuerySpec, QueryFilter, SearchResultLite)
  CONSUMED BY: CacheOrchestrator.check() — replaces inline QuerySpec construction
  DOES NOT depend on: VectorStoreBackend, KeyBuilder, Gate, Judge
  SHARES:      CacheStorage instance with Orchestrator (Orchestrator uses insert/fetch_payload)
  IF CHANGED:  Orchestrator.check() search call path,
               Judge thresholds may need recalibration (if search semantics change)
"""


@dataclass
class SearchContext:
    """Runtime context passed from Orchestrator to SearchStrategy.

    Data flow: Orchestrator constructs -> SearchStrategy reads -> used for QuerySpec/filter

    Coupling:
      - CONSTRUCTED BY: CacheOrchestrator.check() (fills from step counter + stage outputs)
      - CONSUMED BY: SearchStrategy.search()
      - IF CHANGED: Orchestrator construction logic, SearchStrategy read logic
    """
    query_keys: dict[str, torch.Tensor]   # from KeyBuilder.build()
    checkpoint_id: CheckpointID
    current_step: int = 0                  # inference cycle count within current task
    task_key: Optional[str] = None         # normalised task identifier (None = no task filter)


@runtime_checkable
class SearchStrategy(Protocol):
    """Encapsulate all search parameters and database interaction.

    Data flow: SearchContext -> build QuerySpec -> CacheStorage.search() -> results

    Coupling:
      - DEPENDS ON: CacheStorage.search()
      - CONSUMED BY: CacheOrchestrator.check()
      - DOES NOT: make hit/miss decisions (that's Judge's job)
      - IF CHANGED: Orchestrator.check() must adapt to new return type or semantics
    """

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        """Execute a search against the cache storage.

        Args:
            ctx: Runtime context containing query_keys, checkpoint_id,
                 current_step, and optional task_key. Constructed by Orchestrator.

        Returns:
            Search results sorted by descending score. Judge decides hit/miss.
        """
        ...


class SimpleKnnStrategy:
    """Standard KNN search with configurable fusion and step filtering.

    Data flow: SearchContext -> QueryFilter + QuerySpec(fusion, backend_hints)
              -> CacheStorage.search() -> results

    Coupling:
      - DEPENDS ON: CacheStorage.search() (thread-safe, RLock-protected)
      - HOLDS: all search parameters (top_k, step_filter, fusion_weights, etc.)
      - SHARES: CacheStorage instance with Orchestrator (search vs insert paths)
      - IF CHANGED: search behavior changes, Judge thresholds may need recalibration

    Parameters held (not from config — injected via constructor):
      - top_k: number of results to return
      - step_filter: "all" (no filter) | "exact" | "window"
      - step_window: window size for "window" mode
      - rrf_k: RRF fusion parameter k (backend-specific, passed via backend_hints)
      - fusion_weights: per-field fusion weights (from keys config, backend-agnostic)
      - candidate_multiplier: prefetch limit (backend-specific, passed via backend_hints)
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",      # "all" | "exact" | "window"
        step_window: int = 5,
        rrf_k: int = 60,
        fusion_weights: Optional[dict[str, float]] = None,
        candidate_multiplier: int = 5,
    ) -> None:
        ...

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        """Execute KNN search with configured fusion parameters.

        Flow:
          1. Build QueryFilter from step_filter + ctx.current_step (if applicable)
          2. Construct QuerySpec with fusion_weights + backend_hints
          3. Call self._storage.search(spec)
          4. Return results (Judge decides hit/miss downstream)
        """
        filters = self._build_filters(ctx)
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            backend_hints={"rrf_k": self._rrf_k, "candidate_multiplier": self._candidate_multiplier},
        )
        return self._storage.search(spec)

    def _build_filters(self, ctx: SearchContext) -> Optional[QueryFilter]:
        """Build QueryFilter based on step_filter mode and runtime context.

        Data flow: step_filter config + ctx.current_step -> QueryFilter or None
        """
        if self._step_filter == "all":
            return None
        elif self._step_filter == "exact":
            return QueryFilter(step_range=(ctx.current_step, ctx.current_step))
        elif self._step_filter == "window":
            lo = max(0, ctx.current_step - self._step_window)
            hi = ctx.current_step + self._step_window
            return QueryFilter(step_range=(lo, hi))
        else:
            raise ValueError(f"Unknown step_filter: {self._step_filter}")
```

**step_filter 逻辑**（三种模式全部实现）：
- `"all"`：不加 filter，搜全库
- `"exact"`：QueryFilter(step_range=(current_step, current_step))
- `"window"`：QueryFilter(step_range=(current_step - step_window, current_step + step_window))

**current_step 来源**：Orchestrator 维护 per-task step counter，通过 SearchContext 传入（见 3.1.3）。

#### 3.1.3 修改 Orchestrator（`orchestrator.py`）

**核心变更**：
1. 构造函数：gate/judge/search_strategy 从单值改为 `dict[CheckpointID, ...]` 映射
2. 新增 per-task step counter（`_step_counter`），on_task_begin 重置，每次 check 递增
3. check() 构造 SearchContext 传递运行时信息，委托 SearchStrategy
4. write() 保持不变（写入路径不经过 SearchStrategy）

```python
class CacheOrchestrator:
    """Orchestrate cache check and write operations.

    Combines pluggable components (KeyBuilder, Gate, Judge, SearchStrategy)
    with CacheStorage. All storage interaction goes through CacheStorage
    facade — never touches VectorStoreBackend directly.

    Data flow overview:
      check():  Interceptor -> collect -> gate -> build -> SearchContext
                -> search_strategy.search(ctx) -> judge -> CheckResult
      write():  Interceptor -> collect -> build -> CacheEntry -> storage.insert

    Coupling:
      - DEPENDS ON: QueryKeyBuilder, GateFunction, SimilarityJudge, SearchStrategy (Step 4 components)
      - DEPENDS ON: CacheStorage facade (Step 3) — insert/fetch_payload (search delegated to SearchStrategy)
      - CONSUMED BY: InferenceInterceptor (calls check/write at checkpoint slots)
      - DOES NOT depend on: VectorStoreBackend, Qdrant, or any specific backend
      - SHARES: CacheStorage instance with SearchStrategy (Orchestrator: insert/fetch, Strategy: search)
      - IF CHANGED: Interceptor's cache integration logic may need updating

    Per-checkpoint dispatch:
      gates, judges, search_strategies are dict[CheckpointID, Component].
      CP1 and CP3 can use different Gate/Judge/SearchStrategy instances.
      key_builder is shared across checkpoints (same data extraction logic).

    Step counter:
      _step_counter tracks inference cycles within a task (client connection).
      Reset by on_task_begin(), incremented by check(). Passed to SearchStrategy
      via SearchContext for step_filter="exact"/"window" modes.
    """

    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gates: dict[CheckpointID, GateFunction],
        judges: dict[CheckpointID, SimilarityJudge],
        search_strategies: dict[CheckpointID, SearchStrategy],
        timer: Optional[SystemTimer] = None,
    ) -> None:
        ...
        self._step_counter: int = 0

    def on_task_begin(self) -> None:
        """Reset per-task state. Called when a client connection opens."""
        self._step_counter = 0

    def check(self, checkpoint_id: CheckpointID, **stage_outputs) -> CheckResult:
        """Cache check pipeline: collect -> gate -> build -> ctx -> strategy.search() -> judge -> fetch.

        Flow:
          1. key_builder.collect(checkpoint_id, **stage_outputs)
          2. gates[checkpoint_id](checkpoint_id, cached_data) -> if False: MISS
          3. key_builder.build(checkpoint_id) -> query_keys
          4. Construct SearchContext(query_keys, checkpoint_id, current_step=_step_counter)
          5. search_strategies[checkpoint_id].search(ctx) -> results
             (SearchStrategy constructs QuerySpec internally, calls CacheStorage.search())
          6. judges[checkpoint_id](results, checkpoint_id, cached_data) -> (hit_type, winner_id)
          7. if FULL_HIT: storage.fetch_payload(winner_id) -> payload
          8. _step_counter += 1 (only on CP1 check, to count inference cycles)
          9. return CheckResult

        Note: collect() before gate() so Gate can access cached_data.
        fetch_payload called by Orchestrator (not delegated to SearchStrategy or Judge).
        Step counter incremented after CP1 check (one increment per inference cycle).
        """
        gate = self._gates[checkpoint_id]
        judge = self._judges[checkpoint_id]
        strategy = self._search_strategies[checkpoint_id]

        self._key_builder.collect(checkpoint_id, **stage_outputs)
        if not gate(checkpoint_id, self._key_builder.cached_data):
            if checkpoint_id == CheckpointID.CP1:
                self._step_counter += 1
            return CheckResult(hit_type=HitType.MISS)

        query_keys = self._key_builder.build(checkpoint_id)
        ctx = SearchContext(
            query_keys=query_keys,
            checkpoint_id=checkpoint_id,
            current_step=self._step_counter,
        )
        results = strategy.search(ctx)  # 委托 SearchStrategy
        hit_type, winner_id = judge(results, checkpoint_id, self._key_builder.cached_data)

        if checkpoint_id == CheckpointID.CP1:
            self._step_counter += 1

        ...
```

**向后兼容**：现有测试使用旧签名（单个 gate/judge）。需要更新测试 conftest 和 test 文件，将单值包装为 dict。

**step_counter 递增时机**：仅在 CP1 check 时递增（每个 inference cycle 只有一次 CP1 check），CP3 check 使用同一个 step 值。on_task_begin() 重置为 0。

#### 3.1.4 InMemoryBackend 提升（`backends/in_memory_backend.py`）

将 `tests/cache/conftest.py` 中的 InMemoryBackend 移到 `src/openpi/cache/backends/in_memory_backend.py`。

```python
"""In-memory vector store backend for development and testing.

Overview
--------
Stores entries in a Python dict, performs brute-force cosine similarity search.
No external dependencies. Suitable for:
  - Unit/integration tests (no Qdrant required)
  - Development and debugging (--cache_config with type: in_memory)
  - Small-scale cache validation (< 10k entries)

Data flow: CacheStorage -> InMemoryBackend.search/insert/... -> in-process dict

Coupling map:
  DEPENDS ON:  backend_base.py (VectorStoreBackend ABC), storage_types.py
  CONSUMED BY: CacheStorage (via VectorStoreBackend interface)
  IF CHANGED:  tests and development cache configs may need updating
  NOTE:        ignores fusion_weights and backend_hints in QuerySpec (single-field cosine)
"""
```

`tests/cache/conftest.py` 改为 `from openpi.cache.backends.in_memory_backend import InMemoryBackend`。

#### 3.1.5 SearchStrategy 单元测试（`tests/cache/test_search_strategy.py`）

测试内容：
- SimpleKnnStrategy 构造 + 基本搜索，走 InMemoryBackend 全链路（step_filter="all"，无 filter）
- QuerySpec 中 fusion_weights + backend_hints 正确传递（mock storage.search() 捕获 spec）
- step_filter="all"：不加 filter，走 InMemoryBackend 全链路
- step_filter="exact"：mock storage.search()，断言 QuerySpec.filters.step_range==(step, step)（InMemoryBackend 不支持 step_range，不走全链路）
- step_filter="window"：mock storage.search()，断言 QuerySpec.filters.step_range 正确（同上）
- SearchContext 各字段正确传递
- Protocol 合规性检查

---

### Phase 2：Config 系统

**目标**：YAML 配置 → dataclass 树 → 组件工厂。

#### 3.2.1 Config dataclass 树（`config.py`）

```python
"""Cache config system: YAML -> dataclass -> component factory.

Overview
--------
Config is a pure factory layer: reads YAML, instantiates cache components,
returns them for injection into Orchestrator and Interceptor. Components do
NOT import or depend on this module — they receive plain Python values via
constructors.

Scope: This module only covers CacheConfig (cache subsystem). Server, policy,
debug, and collect parameters remain in serve_policy.py's existing tyro CLI.
Full YAML-ization of all parameters is a separate future task.

Data flow:
  YAML file -> _substitute_env_vars() -> yaml.safe_load() -> CacheConfig
    -> validate_cache_config() -> build_cache_components() -> dict of component instances
    -> serve_policy.py injects into Orchestrator/Interceptor

Coupling map:
  DEPENDS ON:  all component constructors:
               - SystemTimer (timing.py)
               - CacheStorage (cache_storage.py)
               - InMemoryBackend (backends/in_memory_backend.py)
               - QdrantVectorStore (backends/qdrant_backend.py)
               - PlaceholderKeyBuilder (components/key_builder.py)
               - AlwaysSearchGate (components/gate.py)
               - ThresholdJudge (components/judge.py)
               - SimpleKnnStrategy (components/search_strategy.py)
  CONSUMED BY: serve_policy.py (the ONLY consumer, via --cache_config path)
  DOES NOT:    get imported by any component — components are config-unaware
  IF CHANGED:  serve_policy.py assembly logic must sync;
               YAML file format must match dataclass fields;
               adding new component types requires adding factory branch
"""

@dataclass
class KeyFieldConfig:
    enabled: bool = True
    weight: float = 1.0

@dataclass
class KeysConfig:
    vision_0: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    vision_1: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    vision_2: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    prompt_emb: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    robot_state: KeyFieldConfig = field(default_factory=KeyFieldConfig)

@dataclass
class GateConfig:
    type: str = "always_search"

@dataclass
class JudgeConfig:
    type: str = "threshold"
    threshold: float = 0.98

@dataclass
class SearchStrategyConfig:
    type: str = "simple_knn"
    top_k: int = 1
    step_filter: str = "all"        # "all" | "exact" | "window"
    step_window: int = 5
    rrf_k: int = 60                 # backend-specific, passed via backend_hints
    candidate_multiplier: int = 5   # backend-specific, passed via backend_hints
    # fusion_weights: None = 从 keys.weight 自动生成（config 加载逻辑处理）

@dataclass
class CheckpointConfig:
    gate: GateConfig = field(default_factory=GateConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    search_strategy: SearchStrategyConfig = field(default_factory=SearchStrategyConfig)

@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection_name: str = "openpi_cache"
    prefer_grpc: bool = False
    grpc_port: int = 6334
    request_timeout: int = 30

@dataclass
class BackendConfig:
    type: str = "in_memory"      # "in_memory" | "qdrant"
    vector_dims: dict[str, int] = field(default_factory=lambda: {"robot_state": 32})
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)

@dataclass
class TimerConfig:
    enabled: bool = True
    buffer_size: int = 10_000
    output_csv_dir: str | None = None

@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"

@dataclass
class CacheConfig:
    """Top-level cache configuration. This is the root dataclass for cache.yaml."""
    enabled: bool = False
    timer: TimerConfig = field(default_factory=TimerConfig)
    keys: KeysConfig = field(default_factory=KeysConfig)
    key_builder: KeyBuilderConfig = field(default_factory=KeyBuilderConfig)
    checkpoints: dict[str, CheckpointConfig] = field(default_factory=lambda: {
        "cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98)),
        "cp3": CheckpointConfig(judge=JudgeConfig(threshold=0.95)),
    })
    backend: BackendConfig = field(default_factory=BackendConfig)
```

**注意**：不包含 ServerConfig / PolicyConfig / DebugConfig / CollectConfig。这些参数保留在 serve_policy.py 的 tyro CLI 中。全量 YAML 化为后续独立任务。

#### 3.2.2 YAML 加载函数（`config.py` 内）

```python
def load_cache_config(path: str | Path) -> CacheConfig:
    """Load cache config from YAML file with environment variable substitution.

    Data flow: YAML file (disk) -> read text -> _substitute_env_vars()
              -> yaml.safe_load() -> raw dict -> _dict_to_dataclass()
              -> CacheConfig -> validate_cache_config() -> return

    Coupling:
      - DEPENDS ON: PyYAML (yaml.safe_load), os.environ (env var substitution)
      - CONSUMED BY: serve_policy.py main() via --cache_config path — the ONLY caller
      - IF CHANGED: YAML file format must match; serve_policy.py may need to
                    handle new exceptions

    Environment variable syntax: ${VAR_NAME} or ${VAR_NAME:-default_value}
    """
    # 1. 读取 YAML 文件内容
    # 2. 正则替换 ${VAR:-default} 模式
    # 3. yaml.safe_load() 解析
    # 4. 递归构建 CacheConfig dataclass 树
    # 5. 运行 validate_cache_config() 校验
    ...

def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR} and ${VAR:-default} patterns with environment values.

    Data flow: raw YAML text -> regex substitution -> text with env values resolved
    """
    ...
```

#### 3.2.3 启动时校验（`config.py` 内）

```python
def validate_cache_config(config: CacheConfig) -> None:
    """Cross-validate cache config consistency. Called once at startup.

    Data flow: CacheConfig -> cross-field validation -> raise or pass

    Coupling:
      - DEPENDS ON: types.CACHE_QUERY_FIELDS (valid field names),
                    CheckpointID (valid checkpoint names)
      - CALLED BY: load_cache_config() (automatically after parsing)
      - IF CHANGED: new validation rules may reject previously valid YAML files

    Checks:
    1. keys 中 enabled=true 的字段必须出现在 backend.vector_dims 中
    2. backend.vector_dims 的 key 必须是 CACHE_QUERY_FIELDS 的子集
    3. checkpoints 中只允许 "cp1"、"cp3"（当前支持的检查点）
    4. key_builder type 合法性
    5. gate/judge/search_strategy type 合法性
    6. key_builder.type ↔ enabled keys 交叉校验：
       当 type="placeholder" 时，只允许 robot_state 的 enabled=true，
       其余字段 enabled=true 则报错（防止配置写了但运行时不生效的语义落空）
    7. step_filter 值合法性：必须是 "all" | "exact" | "window"
    """
    ...
```

错误信息格式示例：
```
ConfigValidationError: keys.vision_0 is enabled but not found in backend.vector_dims.
  Enabled keys: ['vision_0', 'robot_state']
  Backend vector_dims: {'robot_state': 32}
  Fix: add 'vision_0' to backend.vector_dims or set keys.vision_0.enabled=false

ConfigValidationError: keys.vision_0 is enabled but key_builder type 'placeholder'
  only supports: ['robot_state'].
  Fix: set keys.vision_0.enabled=false, or use a key_builder that supports vision fields
```

#### 3.2.4 组件工厂函数（`config.py` 内）

```python
def build_cache_components(config: CacheConfig) -> dict:
    """Instantiate all cache components from config.

    Data flow:
      CacheConfig -> validate
        -> _build_backend() -> VectorStoreBackend
        -> CacheStorage(backend)
        -> _build_key_builder() -> QueryKeyBuilder
        -> for each checkpoint:
             _build_gate() -> GateFunction
             _build_judge() -> SimilarityJudge
             _build_search_strategy(storage) -> SearchStrategy
        -> return dict of all components

    Returns dict with keys: timer, storage, key_builder, gates, judges, search_strategies

    Coupling:
      - DEPENDS ON: all component constructors (see module-level coupling map)
      - CONSUMED BY: serve_policy.py main() — the ONLY caller
      - INSTANTIATION ORDER MATTERS:
          Backend -> CacheStorage -> SearchStrategy (needs storage ref)
          KeyBuilder, Gate, Judge are independent of each other
      - IF CHANGED: serve_policy.py assembly must match returned dict keys

    keys config dispatch:
      - enabled fields list -> KeyBuilder (decides which vectors to extract)
      - {field: weight} for enabled fields -> SearchStrategy (fusion_weights)
    """
    # 1. 构建 Timer
    timer = SystemTimer(
        enabled=config.timer.enabled,
        buffer_size=config.timer.buffer_size,
        output_csv_dir=config.timer.output_csv_dir,
    )

    # 2. 构建 Backend + CacheStorage
    backend = _build_backend(config.backend)
    storage = CacheStorage(backend)

    # 3. 构建 KeyBuilder
    #    传入 enabled fields 列表（从 keys config 提取）
    enabled_fields = [name for name, kf in _keys_iter(config.keys) if kf.enabled]
    key_builder = _build_key_builder(config.key_builder, enabled_fields)

    # 4. 按检查点构建 Gate / Judge / SearchStrategy
    #    fusion_weights 从 keys config 自动生成
    fusion_weights = {name: kf.weight for name, kf in _keys_iter(config.keys) if kf.enabled}
    gates = {}
    judges = {}
    search_strategies = {}
    for cp_name, cp_config in config.checkpoints.items():
        cp_id = CheckpointID[cp_name.upper()]
        gates[cp_id] = _build_gate(cp_config.gate)
        judges[cp_id] = _build_judge(cp_config.judge)
        search_strategies[cp_id] = _build_search_strategy(
            cp_config.search_strategy, storage, fusion_weights
        )

    return {
        "timer": timer,
        "storage": storage,
        "key_builder": key_builder,
        "gates": gates,
        "judges": judges,
        "search_strategies": search_strategies,
    }
```

#### 3.2.5 默认 YAML 文件（`cache.yaml`）

```yaml
# OpenPI Cache Configuration
# Cache 子系统的配置文件。通过命令行指定：
#   uv run scripts/serve_policy.py --cache_config cache.yaml
#   uv run scripts/serve_policy.py --cache_config configs/experiment_rrf.yaml
# Server/policy/debug/collect 参数仍通过 tyro CLI 传入。

enabled: true             # 总开关

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null    # null = 只打印终端，不写 CSV

# 查询向量字段配置
# enabled: 该字段是否参与 cache key 构建和搜索
# weight: fusion 时的权重（传给 SearchStrategy）
# ⚠️ 当前 key_builder.type=placeholder 仅支持 robot_state，
#    其余字段 enabled 必须为 false，否则启动报错。
#    后续 KeyBuilder 重构后可启用更多字段。
# task_key filtering: 当前未实现，需要 task normalization pipeline
keys:
  vision_0:    { enabled: false, weight: 1.0 }
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }
  robot_state: { enabled: true,  weight: 1.0 }

key_builder:
  type: placeholder       # "placeholder" = 仅用 robot_state

# 分检查点配置：CP1 和 CP3 可独立设置
checkpoints:
  _defaults: &cp_defaults
    gate:
      type: always_search
    search_strategy:
      type: simple_knn
      top_k: 1
      step_filter: all    # "all" | "exact" | "window"
      step_window: 5      # 仅 step_filter=window 时生效
      rrf_k: 60           # Qdrant RRF fusion 参数（in_memory backend 忽略）
      candidate_multiplier: 5  # Qdrant prefetch = top_k × multiplier（in_memory 忽略）

  cp1:
    <<: *cp_defaults
    judge:
      type: threshold
      threshold: 0.98     # ⚠️ 占位值，需真实数据校准
      # 阈值含义取决于搜索模式：
      #   单字段 cosine (robot_state only): score ∈ [-1, 1]，0.98 表示极高相似度
      #   多字段 RRF fusion: score 为小正数，量纲不同，不能直接沿用 cosine 阈值
      # 切换搜索模式后必须重新标定阈值

  cp3:
    <<: *cp_defaults
    judge:
      type: threshold
      threshold: 0.95     # ⚠️ 占位值，需真实数据校准
      # 同上：阈值与搜索模式绑定，切换后需重新标定

backend:
  type: in_memory         # "in_memory" | "qdrant"
  vector_dims:
    robot_state: 32       # 必须与 keys 中 enabled=true 的字段对齐
  qdrant:
    url: ${QDRANT_URL:-http://localhost:6333}
    collection_name: openpi_cache
    prefer_grpc: false
    grpc_port: 6334
    request_timeout: 30
```

---

### Phase 3：serve_policy.py 改造

#### 3.3.1 新增 `--cache_config` 参数

**serve_policy.py Coupling map 更新**：
```
Coupling map:
  DEPENDS ON:  config.py (load_cache_config, build_cache_components) — NEW, conditional import
               CacheOrchestrator, InferenceInterceptor (assembly, only when --cache_config)
               tyro (existing CLI), policy_config, training config (existing)
               PolicyRecorder, CollectionPolicy (optional wrappers)
               WebsocketPolicyServer (serving)
  CONSUMED BY: CLI entry point (user-facing)
  DATA FLOW:
    Default path (no --cache_config):
      tyro CLI args -> create_policy(args) -> optionally wrap -> serve
    Cache config path (--cache_config):
      tyro CLI args -> create_policy(args) -> load_cache_config(yaml)
        -> build_cache_components() -> CacheOrchestrator -> InferenceInterceptor -> serve
  IF CHANGED:  Dockerfile/compose.yml launch commands may need updating
```

**Args dataclass 变更**（新增一个字段，其余全部保留）：
```python
@dataclasses.dataclass
class Args:
    # ... 现有参数全部保留不动 ...

    # 新增：cache 配置文件路径。指定后从 YAML 加载 cache 组件，
    # 覆盖 --cache 及 --timing_csv_dir 参数。
    cache_config: str | None = None
```

**main() 变更**：
```python
def main(args: Args) -> None:
    _configure_torchinductor_cache_dir()
    policy = create_policy(args)  # 现有路径不动
    policy_metadata = policy.metadata

    # Cache 模式：两条路径
    if args.cache_config is not None:
        # YAML 路径：从配置文件加载全部 cache 组件
        from openpi.cache.config import load_cache_config, build_cache_components
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        cache_config = load_cache_config(args.cache_config)
        components = build_cache_components(cache_config)
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
        )
        policy = InferenceInterceptor(
            policy,
            timer=components["timer"],
            orchestrator=orchestrator,
        )
        logging.info("Cache mode enabled via config: %s", args.cache_config)
    elif args.cache:
        # 现有路径：简单 interceptor，无 orchestrator
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer
        timer = SystemTimer(enabled=True, output_csv_dir=args.timing_csv_dir)
        policy = InferenceInterceptor(policy, timer=timer)
        logging.info("Cache mode enabled (simple, no config).")

    # Wrapper ordering matters:
    #   1. InferenceInterceptor (innermost — needs direct Policy access)
    #   2. PolicyRecorder (records interceptor's output)
    #   3. CollectionPolicy (outermost — hooks into model internals via _model)
    # DO NOT reorder without verifying CollectionPolicy._model lookup.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    if args.collect:
        from openpi.collect.collection_policy import CollectionPolicy
        from openpi.collect.data_collector import EpisodeDataCollector
        collector = EpisodeDataCollector(base_dir=args.collect_dir)
        policy = CollectionPolicy(policy, collector)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy_metadata,
    )
    server.serve_forever()
```

**向后兼容说明**：
- 现有 tyro CLI 参数、`create_policy()`、`create_default_policy()`、`DEFAULT_CHECKPOINT` 全部保留不动
- `--cache` 简单路径（无 orchestrator）保持原样
- 新增 `--cache_config` 路径走 YAML 加载 + 完整 cache 组件组装
- 两条路径互斥逻辑：`--cache_config` 优先于 `--cache`

---

### Phase 4：测试

#### 3.4.1 Config 测试（`tests/cache/test_config.py`）

| 用例 | 说明 |
|------|------|
| `test_load_default_config` | 加载 cache.yaml，验证 CacheConfig dataclass 树结构正确 |
| `test_env_var_substitution` | `${VAR:-default}` 替换逻辑 |
| `test_validation_keys_vs_vector_dims` | enabled key 不在 vector_dims 中 → 报错 |
| `test_validation_key_builder_vs_enabled_keys` | placeholder + vision_0 enabled → 报错（Q1） |
| `test_validation_invalid_checkpoint` | 非法检查点名称 → 报错 |
| `test_validation_invalid_step_filter` | step_filter 非法值 → 报错 |
| `test_build_cache_components` | 工厂函数输出正确的组件类型和配置 |
| `test_build_and_assemble` | 从 YAML 加载 → build → 构造 Orchestrator，验证组装不报错（S6） |
| `test_yaml_anchor_merge` | YAML anchor 合并后 CP1/CP3 各自独立 |
| `test_fusion_weights_from_keys` | keys.weight 正确分发到 SearchStrategy |
| `test_wrapper_order` | config 组合 cache+record+collect 后验证 wrapper 链类型顺序（Q7） |

#### 3.4.2 SearchStrategy 测试（`tests/cache/test_search_strategy.py`）

| 用例 | 说明 |
|------|------|
| `test_simple_knn_basic_search` | 基本搜索流程（InMemoryBackend） |
| `test_query_spec_fusion_params` | fusion_weights + backend_hints 正确出现在 QuerySpec 中 |
| `test_step_filter_all` | step_filter="all" 不添加 filter |
| `test_step_filter_exact` | step_filter="exact"：mock storage.search()，验证 QuerySpec.filters.step_range==(step, step) |
| `test_step_filter_window` | step_filter="window"：mock storage.search()，验证 QuerySpec.filters.step_range 正确 |
| `test_search_context_fields` | SearchContext 各字段正确传递 |
| `test_protocol_compliance` | SimpleKnnStrategy 满足 SearchStrategy Protocol |

#### 3.4.3 更新现有测试

Orchestrator 构造函数签名变更后，更新：
- `tests/cache/test_orchestrator.py`
- `tests/cache/test_interceptor.py`
- `tests/cache/conftest.py`（InMemoryBackend 改为 import 正式路径）

将单值 gate/judge 包装为 `{CP1: gate, CP3: gate}` dict，新增 search_strategies dict。

---

## 4. 实现顺序

```
Phase 1  SearchStrategy 组件（独立于 Config）
  1.1  storage_types.py — QuerySpec 扩展（fusion_weights + backend_hints）
  1.2  components/search_strategy.py — SearchContext + Protocol + SimpleKnnStrategy
  1.3  backends/in_memory_backend.py — 从 conftest 提升为正式代码
  1.4  orchestrator.py — 接入分检查点 dict + step counter + on_task_begin() + SearchContext 构造
  1.5  interceptor.py — on_task_begin() 转发给 orchestrator（重置 step_counter）
  1.6  更新现有测试（Orchestrator 签名变更 + conftest import 路径 + interceptor task lifecycle）
  1.7  test_search_strategy.py — 新增测试（all 全链路 + exact/window mock 断言）

Phase 2  Config 系统
  2.1  config.py — CacheConfig dataclass 树 + YAML 加载 + 校验 + 工厂
  2.2  cache.yaml — 默认配置文件
  2.3  test_config.py — 新增测试（含组装集成测试）

Phase 3  serve_policy.py 改造
  3.1  新增 --cache_config 参数 + YAML 加载 + 组件组装
  3.2  手动验证：uv run scripts/serve_policy.py --cache_config cache.yaml 正常启动

Phase 4  收尾
  4.1  uv run pytest 全量回归
  4.2  更新 claude_log/README.md
  4.3  更新 src/openpi/cache/README.md
```

**依赖关系**：Phase 1 → Phase 2 → Phase 3 → Phase 4（严格顺序）

---

## 5. 设计原则提醒

1. **组件不知道 Config**：所有组件通过构造函数接收普通 Python 值，不 import config 模块
2. **Config 是唯一工厂**：读 YAML → 实例化组件 → 注入 Orchestrator。测试可绕过 Config 直接构造组件
3. **渐进式接入**：保留现有 tyro CLI 主路径不动，`--cache_config` 只接管 cache 子系统。全量 YAML 化为后续独立任务
4. **注释规范**：所有新增代码遵循项目耦合性注释规范（Coupling map / Data flow / DEPENDS ON / CONSUMED BY / IF CHANGED）
5. **不过度设计**：只实现讨论中确定的功能，不添加 CLI override、profile 系统、动态热加载等
6. **配置不落空**：YAML 中可写的参数必须在运行时生效；当前不支持的组合在 validate 时 fail-fast 报错

---

## 6. 部件稳定性评估

### 新增文件

| 文件 | 说明 | 预期状态 |
|------|------|----------|
| `components/search_strategy.py` | SearchContext + `SearchStrategy` Protocol + `SimpleKnnStrategy` 实现 | 🟡 单元测试覆盖，未集成验证 |
| `backends/in_memory_backend.py` | InMemoryBackend 正式实现（从 conftest 提升） | 🟡 间接测试覆盖（现有 Orchestrator/Interceptor 测试使用） |
| `config.py` | CacheConfig dataclass 树 + YAML 加载 + 校验 + 组件工厂 | 🟡 单元测试覆盖，未真实模型验证 |
| `cache.yaml` | 默认 cache 配置文件 | ✅ 稳定（纯数据文件） |

### 修改文件（风险变更）

| 文件 | 修改内容 | 修改后状态 | 修改前状态 |
|------|----------|------------|------------|
| `orchestrator.py` | 构造函数签名变更：gate/judge/search_strategy 改为 `dict[CheckpointID, ...]`；新增 step counter；check() 构造 SearchContext 委托 SearchStrategy | 🟡 单元测试需全量重跑 | 🟡 单元测试通过 |
| `storage_types.py` | QuerySpec 新增 `fusion_weights` + `backend_hints` 字段（有默认值） | ⚠️ 高危（backend_hints 通道需 Backend 适配） | ⚠️ 高危 |
| `interceptor.py` | on_task_begin() 转发给 orchestrator.on_task_begin()（重置 step_counter） | 🟡 单元测试需更新 | 🟡 FakeModel 测试通过 |
| `scripts/serve_policy.py` | 新增 `--cache_config` 参数 + YAML 加载路径（现有 CLI 不动） | 🟡 手动验证 + 组装集成测试 | ✅ 稳定 |
| `tests/cache/conftest.py` | InMemoryBackend import 路径变更 | ✅ 纯路径变更 | ✅ |

### 不稳定部件详述

#### `components/search_strategy.py`（新增·🟡）
- SimpleKnnStrategy 持有 CacheStorage 引用做搜索，与 Orchestrator 共享同一 CacheStorage 实例——并发安全依赖 CacheStorage 的 RLock
- 三种 step_filter 模式全部实现；current_step 从 SearchContext 获取（由 Orchestrator 的 step counter 填充）
- fusion_weights 传递路径：Config → SearchStrategy → QuerySpec.fusion_weights → Backend，中间任何一环 key 不匹配会导致搜索异常
- backend_hints 是 dict[str, Any] 无类型约束——Backend 拼写错误读取不到参数时静默降级为默认值
- SearchContext.task_key 当前始终为 None（task_key 来源未实现），不影响搜索行为

#### `backends/in_memory_backend.py`（新增·🟡）
- 从 conftest 提升，逻辑不变，只改文件位置和添加 Coupling map 注释
- 忽略 QuerySpec.fusion_weights 和 backend_hints（单向量 cosine 搜索），与 Qdrant 行为不对称——测试可能掩盖 fusion 传递问题
- 不支持 step_range filter：`supported_filters()` 返回 `frozenset({"checkpoint_id"})`，不含 `"step_range"`。CacheStorage 对含 step_range 的 QuerySpec 会 raise UnsupportedFilterError
- 因此 step_filter="exact"/"window" 的测试通过 mock storage.search() 断言 QuerySpec 内容，不走 InMemoryBackend 全链路；step_filter="all" 的测试走真实全链路

#### `config.py`（新增·🟡）
- YAML 加载中的 `${VAR:-default}` 环境变量替换使用正则，边界情况（嵌套 `$`、值中含 `}`）未穷举测试
- `validate_cache_config()` 新增 key_builder.type ↔ enabled keys 交叉校验，防止配置语义落空
- 工厂函数 `build_cache_components()` 的组件实例化顺序有隐含依赖：Backend → CacheStorage → SearchStrategy
- dataclass 树用 `field(default_factory=...)` 构造嵌套默认值，YAML 部分覆盖时的合并逻辑需仔细测试
- YAML anchor `<<: *defaults` 合并后，cp1/cp3 的 judge 覆盖是否正确取决于 YAML 解析器行为，需显式测试验证

#### `orchestrator.py`（修改·🟡）
- 构造函数签名破坏性变更：所有现有测试和 interceptor 的构造调用都需同步更新
- `self._gates[checkpoint_id]` / `self._judges[checkpoint_id]` 若 dict 缺少某个 CP 的 key 会 KeyError，需要在构造时校验
- 新增 `_step_counter`：仅在 CP1 check 时递增，on_task_begin() 重置。多 CP check 共享同一 step 值
- write() 路径不经过 SearchStrategy，仍直接调用 `self._storage.insert()`——两条路径共享 storage 引用的一致性

#### `storage_types.py`（修改·⚠️ 高危）
- QuerySpec 新增 `fusion_weights`（通用）和 `backend_hints`（backend-specific），有默认值 None
- QdrantVectorStore.search() 需同步修改为从 spec 读参数；backend_hints 的 key 拼写不一致时静默降级
- InMemoryBackend 忽略这两个字段，与 Qdrant 行为不对称——测试可能掩盖传递问题

#### `interceptor.py`（修改·🟡）
- on_task_begin() 新增转发：`self._orchestrator.on_task_begin()` 重置 step_counter
- 若遗漏此转发，step_filter="exact"/"window" 的 step 语义会跨 task 漂移
- on_task_end() 不需要转发（orchestrator 无 per-task 清理逻辑）

#### `scripts/serve_policy.py`（修改·🟡）
- 现有 tyro CLI 完全不动，仅新增 `--cache_config` 参数和对应分支
- `--cache_config` 和 `--cache` 互斥逻辑需明确（cache_config 优先）
- wrapper 顺序（cache → record → collect）已用注释显式标注，有测试验证

---

## 7. 已知风险与边界

| 风险 | 缓解 |
|------|------|
| QdrantBackend.search() 需从 spec.backend_hints 读参数 | Phase 1.1 扩展 QuerySpec 后同步修改 qdrant_backend.py |
| backend_hints key 拼写不一致时静默降级 | QdrantVectorStore 中对缺失 hint 使用 `.get(key, default)` 并 log warning |
| InMemoryBackend 不支持 step_range filter | supported_filters() 显式声明不支持；CacheStorage 会 raise UnsupportedFilterError |
| PlaceholderKeyBuilder 不支持 enabled_fields 参数 | validate_cache_config() 限制只有 robot_state 可 enabled；后续 KeyBuilder 重构时接入 |
| YAML anchor `<<: *defaults` 中 judge 被覆盖是否正确 | test_yaml_anchor_merge 显式验证 |
| --cache_config 和 --cache 同时指定 | cache_config 优先，log warning 提示 --cache 被忽略 |

---

## 8. 不在本次范围内

- KeyBuilder 重构（支持多 field 构建）— 后续单独 step
- CP3 DeferredWriter 实现 — Step 6
- QdrantBackend 集成测试 — 需要 Qdrant 实例
- ThresholdJudge 阈值校准 — 需要真实数据
- task_key 过滤 — 需要 task normalization pipeline
- 全量 server YAML 化（ServerConfig/PolicyConfig/DebugConfig/CollectConfig）— 后续独立任务
- CLI override / profile 系统 — 不做

---

## 9. 答辩模块

### 9.1 疑问

1. `keys` 统一配置是讨论里的核心决议，但 plan 的“已知风险与边界”又写了“Phase 1 不改 KeyBuilder，Config 工厂暂时忽略 enabled_fields”。如果 `keys.*.enabled` 在当前实现里不真正生效，那么 YAML 里的这组字段到底是当前配置还是未来占位？这里是否存在“配置可写但运行时不使用”的语义落空？

2. `SearchStrategy.search(query_keys, checkpoint_id)` 这个接口是否过窄？计划里已经把 `step_filter`、`step_window`、未来的 `task_key` 过滤都归到 SearchStrategy，但当前签名拿不到 `current_step`，也拿不到规范化后的 `task_key`。只靠内部 step counter 似乎不足以覆盖多任务、重连和未来扩展场景。

3. `step_filter="all" | "exact" | "window"` 的范围目前不一致。plan 正文写了要实现并测试 `"window"`，但“已知风险与边界”又写初始实现只支持 `"all"`。最终交付到底是哪一种？如果当前不做 `"exact"`/`"window"`，配置层是否应该直接禁止，而不是先暴露出来？

4. `BackendConfig.type` 对外暴露了 `"in_memory"`，但当前 in-memory backend 只存在于测试侧。生产配置是否应该允许选择一个测试专用 backend？如果允许，它的正式实现文件放在哪里；如果不允许，为什么还要进入 `config.py` 的公开接口？

5. `QuerySpec` 新增 `rrf_k`、`candidate_multiplier`、`fusion_weights`，本质上把 Qdrant 的融合细节上提到了公共类型层。这个抽象是否会把 `storage_types.py` 和 SearchStrategy 重新绑回 Qdrant 语义？如果未来接 FAISS 或单向量 backend，这些字段是被忽略、复用，还是还要继续往 QuerySpec 里加 backend-specific 参数？

6. `checkpoints` 这里为什么使用 `dict[str, CheckpointConfig]`，同时又在校验阶段强限制只能出现 `"cp1"` 和 `"cp3"`？既然 key 空间是封闭的，是否更适合用强类型 dataclass 直接表达，减少 string key、运行时 KeyError 和重复校验？

7. 现有 `scripts/serve_policy.py` 的 wrapper 顺序是 cache → record → collect。入口重构后，这个顺序是否会被显式保留并验证？这里一旦顺序漂移，`CollectionPolicy` 的 `_model` 查找、计时边界以及记录内容都会受到影响。

8. `CachePayload.task_key` 和 backend 的 `task_key` filter 已经在存储层留了口子，但现有写路径并没有真实来源；`src/openpi/cache/interceptor.py` 当前 CP1 write 只写了 `action_chunk`。如果本次 plan 里 SearchStrategy/Backend 要支持 task 过滤，那么 task_key 从哪里来、如何规范化？如果本次不做，默认 YAML 是否应该明确不开放 task filter？

9. `server.yaml` 放在仓库根目录是否合适？这个文件名很通用，但当前计划既想承载 server 全量参数，又想承载 cache 细粒度参数。后面一旦出现多环境、多实验、多 backend 配置，根目录单文件会不会很快失控？

### 9.2 建议

1. 建议把本次交付先收缩为“cache config 接入 `scripts/serve_policy.py --cache` 路径”，保留现有 tyro CLI 主路径不动；等 cache config 真正跑通，再单独做全量 server YAML 化。这样更符合当前 step 目标，也更容易回归。

2. 建议把 `keys` 配置分成“当前生效字段”和“未来预留字段”两层表述。若当前 `PlaceholderKeyBuilder` 只支持 `robot_state`，那默认配置和校验就应明确只允许这一条生效，避免出现伪配置。

3. 建议尽早把 SearchStrategy 的输入升级为一个轻量 `SearchContext`，至少容纳 `checkpoint_id`、`current_step`、`task_key` 和后续 runtime metadata。否则现在先做极简签名，后面为了动态过滤条件还是会再改一次接口。

4. 建议对未实现的配置项采用 fail-fast 策略。若当前版本只支持 `step_filter="all"`，就在 `validate_config()` 里明确报错，不要把 `"exact"` 和 `"window"` 暴露给用户再在运行时降级。

5. 建议不要把测试专用的 `"in_memory"` backend 直接暴露给生产 YAML；要么把它提升为 `src/openpi/cache/backends/` 下的正式实现，要么完全限制在测试中使用。

6. 建议为 `scripts/serve_policy.py` 增加最小装配测试，至少覆盖 config 加载后的 wrapper 顺序、default policy 路径，以及 `cache + record + collect` 的组合行为。否则这次入口级改造的主要风险会落在手动验证上。

7. 建议在 `config.py` 中加入 qdrant collection schema 的启动前预检查，特别是把 `robot_state` 维度从历史 14 切到当前 32 的兼容性风险提前暴露，而不是运行到 query/insert 时再报错。

8. 建议在默认 YAML 注释里把 `ThresholdJudge` 的默认阈值和 search mode 绑定说明写清楚，明确指出”单字段 cosine”和”多字段 RRF”不能直接沿用同一套阈值默认值，避免默认配置看起来可用但其实没有标定基础。

---

### 9.3 疑问回应

#### Q1：keys.*.enabled 语义落空
**接受。** 问题成立。当前 PlaceholderKeyBuilder 硬编码只用 `robot_state`，keys config 中的 enabled 字段写了但运行时不生效，是伪配置。

**修改方案**：
- `validate_config()` 增加校验：当 `key_builder.type == “placeholder”` 时，只允许 `robot_state` 的 enabled=true，其余字段必须 enabled=false，否则报错提示 “PlaceholderKeyBuilder only supports robot_state”
- YAML 注释明确标注哪些字段当前可用、哪些需要后续 KeyBuilder 实现才能生效
- 这样配置结构保持完整（不需要后续加字段），但校验层保证当前语义不落空

#### Q2：SearchStrategy.search() 接口过窄
**接受。** `current_step` 和 `task_key` 是动态运行时信息，不应由 SearchStrategy 自行维护（多任务重连时内部 counter 会出错）。

**修改方案**：
- 引入轻量 `SearchContext` dataclass，替换 `search(query_keys, checkpoint_id)` 的散列参数：
```python
@dataclass
class SearchContext:
    query_keys: dict[str, torch.Tensor]
    checkpoint_id: CheckpointID
    current_step: Optional[int] = None
    task_key: Optional[str] = None
```
- Protocol 签名改为 `search(ctx: SearchContext) -> list[SearchResultLite]`
- Orchestrator 负责构造 SearchContext（从 Interceptor 传入的运行时信息填充）
- 初始实现中 `current_step` 和 `task_key` 传 None，SimpleKnnStrategy 在 None 时等价于 step_filter=”all”

#### Q3：step_filter 交付范围不一致
**接受。** plan 正文和风险节自相矛盾。

**修改方案**（答辩后二次修订）：
- 三种模式全部实现：`”all”` / `”exact”` / `”window”`
- `current_step` 由 Orchestrator 的 per-task step counter 提供，通过 SearchContext 传入
- step counter 实现简单（一个整数 + on_task_begin 重置 + CP1 check 递增），无需推迟
- 测试计划保留 `test_step_filter_all`、`test_step_filter_exact`、`test_step_filter_window`

#### Q4：in_memory backend 暴露给生产 YAML
**接受。** 测试专用 backend 不应出现在生产配置选项中。

**修改方案**：
- 将 InMemoryBackend 从 `tests/cache/conftest.py` 提升到 `src/openpi/cache/backends/in_memory_backend.py`，成为正式实现
- 理由：cache 端到端验证需要不依赖 Qdrant 的轻量 backend（本周计划 1.2 “先用 InMemoryBackend 跑通”）；它不仅仅是测试工具，也是开发调试用的合法 backend
- 提升后加模块级 Coupling map 注释
- conftest 中改为 import 正式路径

#### Q5：QuerySpec fusion 字段绑定 Qdrant 语义
**部分接受。** `rrf_k` 和 `candidate_multiplier` 确实是 Qdrant RRF fusion 特有概念。但 `fusion_weights` 是通用的（任何多向量 backend 都可能需要加权融合）。

**修改方案**：
- QuerySpec 只保留通用字段：`fusion_weights: Optional[dict[str, float]]`
- `rrf_k` 和 `candidate_multiplier` 移回 backend-specific 配置——但不是移回 QdrantBackendConfig（讨论中已否决），而是作为 SearchStrategy 的内部参数，由 SimpleKnnStrategy 在构造 QuerySpec 时通过 `backend_hints: Optional[dict[str, Any]]` 附加字段下传
- 这样 QuerySpec 保持 backend-agnostic，backend-specific 参数通过 hints 通道传递，FAISS 等新 backend 可以忽略不认识的 hints

```python
@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None
    fusion_weights: Optional[dict[str, float]] = None      # 通用
    backend_hints: Optional[dict[str, Any]] = None          # backend-specific
```

SimpleKnnStrategy 构造时：
```python
spec = QuerySpec(
    ...,
    fusion_weights=self._fusion_weights,
    backend_hints={“rrf_k”: self._rrf_k, “candidate_multiplier”: self._candidate_multiplier},
)
```

QdrantVectorStore.search() 从 `spec.backend_hints` 读取。InMemoryBackend / FAISS 忽略。

#### Q6：checkpoints 用 dict[str, ...] 还是强类型
**驳回。** 理由：

1. YAML 天然表达为 string key 的 mapping，dict 直接对应 YAML 解析结果，不需要额外转换层
2. 当前支持 CP1/CP3 两个检查点，但 CP2 是明确的待实现项（Step 7），key 空间不是真正封闭的
3. dict 在 Config 层使用（纯数据），到 Orchestrator 层已经转为 `dict[CheckpointID, Component]`（强类型枚举 key）；运行时不存在 string key 的 KeyError 风险
4. 改为强类型 dataclass（`cp1: CheckpointConfig, cp3: CheckpointConfig`）在 CP2 加入时需要修改 dataclass 定义 + YAML 结构 + 工厂逻辑三处，而 dict 只需在 YAML 中加一个 key

校验阶段的 “只允许 cp1/cp3” 限制是版本门控，不是类型设计缺陷——后续版本放开即可。

#### Q7：wrapper 顺序保留
**接受。** wrapper 顺序 cache → record → collect 有隐含的行为依赖（CollectionPolicy 需要访问底层 policy 的 `_model`）。

**修改方案**：
- `serve_policy.py` 改造后的代码中显式添加注释标注 wrapper 顺序及其原因
- Phase 4 测试补充一条：`test_wrapper_order_preserved`（通过 config 组合 cache+record+collect 后验证 wrapper 链的类型顺序）

```python
# Wrapper ordering matters:
#   1. InferenceInterceptor (innermost — needs direct Policy access)
#   2. PolicyRecorder (records interceptor's output)
#   3. CollectionPolicy (outermost — hooks into model internals via _model)
# DO NOT reorder without verifying CollectionPolicy._model lookup.
```

#### Q8：task_key 来源
**接受。** 当前写路径确实没有 task_key 来源，本次不做 task 过滤。

**修改方案**：
- 本次实现中 task_key 相关功能不激活：
  - SearchContext.task_key 默认 None
  - SimpleKnnStrategy 在 task_key=None 时不添加 task_key filter
  - CachePayload 构造时 task_key 保持默认空字符串
- YAML 中不暴露 task_key 相关配置项（不写进 search_strategy 块）
- YAML 注释说明：`# task_key filtering: not yet implemented, requires task normalization pipeline`

#### Q9：server.yaml 放置位置
**驳回。** 理由：

1. 项目根目录放配置文件是常见实践（docker-compose.yml、pyproject.toml、.pre-commit-config.yaml 都在根目录）
2. 当前项目规模下，一个默认配置文件在根目录不构成混乱
3. 多环境/多实验场景通过 `--config configs/experiment_xxx.yaml` 指定不同文件解决，不需要动默认文件的位置
4. 讨论阶段已决议 “默认文件为根目录下的 server.yaml”，无新信息需要推翻

### 9.4 建议回应

#### S1：缩小交付范围，保留 tyro CLI 主路径
**接受。** 这与 Q1 的”配置不能语义落空”精神一致，且更符合 weekly plan 1.2 的目标（先用 InMemoryBackend 跑通 cache 链路）。

**修改方案**：
- `serve_policy.py` 保留现有 tyro CLI 作为主路径，不删除 Args dataclass
- 新增 `--config` 可选参数：当指定 `--config` 时走 YAML 路径，否则走原有 tyro 路径
- Phase 3 改造范围缩小：只在 `args.cache and args.config` 时从 YAML 加载 cache 组件，server/policy/debug/collect 参数仍走 tyro

```python
@dataclasses.dataclass
class Args:
    # ... 现有参数全部保留 ...
    # 新增：cache 配置文件路径。指定后覆盖 --cache 及相关参数。
    cache_config: str | None = None
```

```python
def main(args: Args) -> None:
    policy = create_policy(args)  # 现有路径不动

    if args.cache_config is not None:
        # YAML 路径：从配置文件加载 cache 组件
        from openpi.cache.config import load_cache_config, build_cache_components
        cache_config = load_cache_config(args.cache_config)
        components = build_cache_components(cache_config)
        orchestrator = CacheOrchestrator(...)
        policy = InferenceInterceptor(policy, timer=components[“timer”], orchestrator=orchestrator)
    elif args.cache:
        # 现有路径：简单 interceptor，无 orchestrator
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer
        timer = SystemTimer(enabled=True, output_csv_dir=args.timing_csv_dir)
        policy = InferenceInterceptor(policy, timer=timer)

    # record / collect 包装顺序不变
    ...
```

这意味着：
- Config dataclass 树简化：只需 `CacheConfig` 部分，不需要 `ServePolicyConfig` 全量结构
- `server.yaml` 简化为 `cache.yaml`（只包含 cache 配置块）
- Phase 2 工作量减少约 40%（不需要 ServerConfig/PolicyConfig/DebugConfig/CollectConfig）
- 全量 YAML 化作为后续独立任务

#### S2：keys 配置分层
**已在 Q1 回应中覆盖。** validate_config() 根据当前 key_builder.type 限制可用的 enabled 字段。

#### S3：SearchContext
**已在 Q2 回应中覆盖。** 引入 SearchContext dataclass。

#### S4：未实现配置项 fail-fast
**原则接受，Q3 已修订为全部实现。** step_filter 三种模式都做，不再需要 fail-fast 限制。validate_cache_config() 仍校验 step_filter 值的合法性（必须是 "all"/"exact"/"window" 之一）。

#### S5：in_memory backend 提升
**已在 Q4 回应中覆盖。** 提升到 `src/openpi/cache/backends/in_memory_backend.py`。

#### S6：serve_policy.py 最小装配测试
**接受。** 但范围收窄（配合 S1 的交付范围缩小）。

**修改方案**：
- 新增 `tests/cache/test_config.py` 中的 `test_build_and_assemble`：从 YAML 加载 → build_cache_components → 构造 Orchestrator，验证组装不报错
- 不需要真实 Policy 对象，用 mock 验证组件注入链
- wrapper 顺序测试（Q7）也放在此文件

#### S7：Qdrant collection schema 启动预检查
**驳回。** 理由：

1. config.py 的职责是”读配置 + 实例化组件”，不应承担 Qdrant 运行时健康检查
2. `vector_dims` 配置校验（keys ↔ vector_dims 一致性）已在 validate_config() 中覆盖
3. Qdrant collection 是否存在、schema 是否匹配是 QdrantVectorStore 初始化时的职责——在 backend 构造函数中检查更合理
4. 维度从 14→32 的迁移是部署层面的问题（需要重建 collection），不是 config 校验能解决的

#### S8：YAML 中阈值与 search mode 绑定说明
**接受。** 这是纯注释改进，零成本高收益。

**修改方案**：
在 YAML 的 judge.threshold 旁添加注释：
```yaml
judge:
  type: threshold
  threshold: 0.98     # ⚠️ 占位值，需真实数据校准
  # 阈值含义取决于搜索模式：
  #   单字段 cosine (robot_state only): score ∈ [-1, 1]，0.98 表示极高相似度
  #   多字段 RRF fusion: score 为小正数，量纲不同，不能直接沿用 cosine 阈值
  # 切换搜索模式后必须重新标定阈值
```

---

### 9.5 答辩后修订汇总

| 编号 | 来源 | 判定 | 影响 |
|------|------|------|------|
| Q1 | 疑问 | ✅ 接受 | validate_config() 增加 key_builder.type ↔ enabled keys 交叉校验 |
| Q2 | 疑问 | ✅ 接受 | 引入 SearchContext dataclass，替换 search() 散列参数 |
| Q3 | 疑问 | ✅ 接受（二次修订） | 三种 step_filter 全部实现，Orchestrator 加 step counter |
| Q4 | 疑问 | ✅ 接受 | InMemoryBackend 提升到 backends/ 正式代码 |
| Q5 | 疑问 | 🟡 部分接受 | rrf_k/candidate_multiplier 移到 backend_hints，fusion_weights 保留在 QuerySpec |
| Q6 | 疑问 | ❌ 驳回 | dict[str, CheckpointConfig] 保持不变 |
| Q7 | 疑问 | ✅ 接受 | 显式注释 wrapper 顺序 + 测试验证 |
| Q8 | 疑问 | ✅ 接受 | 本次不做 task_key 过滤，YAML 不暴露相关配置 |
| Q9 | 疑问 | ❌ 驳回 | server.yaml 位置不变 |
| S1 | 建议 | ✅ 接受 | 保留 tyro CLI，新增 --cache_config 可选参数，Config 范围缩小为 CacheConfig |
| S2 | 建议 | ✅ 已覆盖 | 同 Q1 |
| S3 | 建议 | ✅ 已覆盖 | 同 Q2 |
| S4 | 建议 | ✅ 已覆盖 | 同 Q3 |
| S5 | 建议 | ✅ 已覆盖 | 同 Q4 |
| S6 | 建议 | ✅ 接受 | test_config.py 增加组装集成测试 |
| S7 | 建议 | ❌ 驳回 | Qdrant schema 检查是 backend 构造职责，不属于 config |
| S8 | 建议 | ✅ 接受 | YAML 注释增加阈值与 search mode 绑定说明 |

### 9.6 答辩后修订的 Plan 条目（✅ 已全部同步到正文）

以下条目已根据答辩结论回溯修改到 Plan 正文中：

1. ✅ **Section 0 概述**：更新为保留 tyro CLI + 新增 --cache_config，描述 SearchContext、InMemoryBackend 提升
2. ✅ **Section 1 新增文件**：新增 `backends/in_memory_backend.py`；`server.yaml` → `cache.yaml`
3. ✅ **Section 2 修改文件**：serve_policy.py 描述改为"新增 --cache_config"；新增 conftest/test_orchestrator/test_interceptor 修改
4. ✅ **Phase 1.1**：QuerySpec 只保留 `fusion_weights` + 新增 `backend_hints`，删除 `rrf_k` 和 `candidate_multiplier`
5. ✅ **Phase 1.2**：SearchStrategy Protocol 签名改为 `search(ctx: SearchContext)`；新增 `SearchContext` dataclass；SimpleKnnStrategy 实现全部三种 step_filter 模式，rrf_k/candidate_multiplier 放入 backend_hints
6. ✅ **Phase 1.3**：新增 3.1.4 InMemoryBackend 提升（从 conftest 移到 backends/）
7. ✅ **Phase 1.3 Orchestrator**：新增 step counter + on_task_begin() 重置 + SearchContext 构造
8. ✅ **Phase 1.5 测试**：更新为三种 step_filter 测试 + SearchContext 测试
9. ✅ **Phase 2.1**：Config 范围缩小为 `CacheConfig`（删除 ServePolicyConfig/ServerConfig 等）
10. ✅ **Phase 2.2**：load_config → load_cache_config，返回 CacheConfig
11. ✅ **Phase 2.3**：validate_cache_config() 新增 key_builder.type ↔ enabled keys 校验 + step_filter 合法性校验
12. ✅ **Phase 2.5**：`server.yaml` → `cache.yaml`；YAML judge.threshold 注释增加 search mode 绑定说明；keys 注释标注当前受限于 placeholder；task_key 注释说明未实现
13. ✅ **Phase 3.1**：serve_policy.py 保留 tyro CLI，新增 `--cache_config`；wrapper 顺序显式注释
14. ✅ **Phase 4 测试**：新增 wrapper 顺序测试、组装集成测试、key_builder ↔ enabled keys 校验测试、step_filter_exact 测试
15. ✅ **Section 4 实现顺序**：更新步骤编号和描述
16. ✅ **Section 5 设计原则**：更新为"渐进式接入"和"配置不落空"
17. ✅ **Section 6 稳定性评估**：全面更新新增/修改文件表和不稳定部件详述
18. ✅ **Section 7 已知风险**：更新风险表，删除已解决的条目，新增 backend_hints/UnsupportedFilterError/互斥参数等
19. ✅ **Section 8 不在本次范围**：新增 task_key 过滤、全量 YAML 化
