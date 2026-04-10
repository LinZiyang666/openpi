# Step 4 Config System — Implementation Plan

> Date: 2026-04-04
> Status: Plan
> Prior discussion: [step4_config_discussion.log.md](step4_config_discussion.log.md)

---

## 0. Overview

Based on all decisions from the discussion phase + defense revisions, build the cache config system + SearchStrategy component. After completion:

- `serve_policy.py` retains the existing tyro CLI main path, adds optional `--cache_config cache.yaml` parameter
- Cache config is read from a YAML file, with anchor and environment variable substitution support
- SearchStrategy becomes the fifth pluggable component of the Orchestrator, receiving runtime information via SearchContext
- CP1/CP3 can independently configure gate, judge, search_strategy
- Keys are configured centrally, distributed to KeyBuilder + SearchStrategy (currently limited by key_builder.type; validation layer ensures no semantic gaps)
- InMemoryBackend is promoted to production code, supporting development/debugging without Qdrant dependency

---

## 1. New Files List

| File | Description |
|------|-------------|
| `src/openpi/cache/components/search_strategy.py` | SearchStrategy Protocol + SearchContext + QdrantWeightedRrfKnnStrategy implementation |
| `src/openpi/cache/backends/in_memory_backend.py` | InMemoryBackend production implementation (promoted from conftest) |
| `src/openpi/cache/config.py` | CacheConfig dataclass tree + YAML loading + validation + component factory |
| `cache.yaml` | Project root directory, cache default config file (fully commented) |
| `tests/cache/test_config.py` | Config loading + validation + factory + assembly integration tests |
| `tests/cache/test_search_strategy.py` | SearchStrategy unit tests |

## 2. Modified Files List

| File | Modifications |
|------|---------------|
| `src/openpi/cache/orchestrator.py` | Integrate per-checkpoint gate/judge/search_strategy dict + step counter + SearchContext construction |
| `src/openpi/cache/storage_types.py` | QuerySpec extended with fusion_weights + backend_hints |
| `src/openpi/cache/interceptor.py` | on_task_begin() forwards to orchestrator (resets step_counter) |
| `scripts/serve_policy.py` | Retain tyro CLI, add optional `--cache_config` parameter |
| `tests/cache/conftest.py` | InMemoryBackend changed to import from production path |
| `tests/cache/test_orchestrator.py` | Adapt to Orchestrator new signature |
| `tests/cache/test_interceptor.py` | Adapt to Orchestrator new signature |
| `logs/README.md` | Add plan entry |

---

## 3. Implementation Steps

### Phase 1: SearchStrategy Component (Independent of Config)

**Goal**: Add a pluggable search strategy component, extracting search logic from the Orchestrator.

#### 3.1.1 Extend QuerySpec (`storage_types.py`)

QuerySpec gains new fusion parameter fields so that QuerySpec constructed by SearchStrategy can carry fusion information down to the Backend.

**Module-level Coupling map update** (append to existing docstring):
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

**QuerySpec changes**:
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
    # --- New: generic fusion parameters ---
    fusion_weights: Optional[dict[str, float]] = None     # Per-field fusion weights (backend-agnostic)
    # --- New: backend-specific parameter channel ---
    backend_hints: Optional[dict[str, Any]] = None        # e.g. {"rrf_k": 60, "candidate_multiplier": 5}
```

**Impact scope**:
- QdrantVectorStore.search() reads weights from `spec.fusion_weights` and rrf_k/candidate_multiplier from `spec.backend_hints`
- InMemoryBackend ignores fusion_weights and backend_hints (single-vector cosine search)
- Future backends like FAISS can read their own parameters from backend_hints without polluting QuerySpec's public fields

#### 3.1.2 New SearchStrategy Protocol + QdrantWeightedRrfKnnStrategy (`components/search_strategy.py`)

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


class QdrantWeightedRrfKnnStrategy:
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

**step_filter logic** (all three modes fully implemented):
- `"all"`: No filter, searches entire store
- `"exact"`: QueryFilter(step_range=(current_step, current_step))
- `"window"`: QueryFilter(step_range=(current_step - step_window, current_step + step_window))

**current_step source**: Orchestrator maintains a per-task step counter, passed in via SearchContext (see 3.1.3).

#### 3.1.3 Modify Orchestrator (`orchestrator.py`)

**Core changes**:
1. Constructor: gate/judge/search_strategy changed from single value to `dict[CheckpointID, ...]` mapping
2. New per-task step counter (`_step_counter`), reset on on_task_begin, incremented on each check
3. check() constructs SearchContext to pass runtime info, delegates to SearchStrategy
4. write() remains unchanged (write path does not go through SearchStrategy)

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
        results = strategy.search(ctx)  # Delegate to SearchStrategy
        hit_type, winner_id = judge(results, checkpoint_id, self._key_builder.cached_data)

        if checkpoint_id == CheckpointID.CP1:
            self._step_counter += 1

        ...
```

**Backward compatibility**: Existing tests use the old signature (single gate/judge). Tests conftest and test files need updating to wrap single values into dicts.

**step_counter increment timing**: Only incremented on CP1 check (each inference cycle has only one CP1 check); CP3 check uses the same step value. on_task_begin() resets to 0.

#### 3.1.4 InMemoryBackend Promotion (`backends/in_memory_backend.py`)

Move InMemoryBackend from `tests/cache/conftest.py` to `src/openpi/cache/backends/in_memory_backend.py`.

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

`tests/cache/conftest.py` changed to `from openpi.cache.backends.in_memory_backend import InMemoryBackend`.

#### 3.1.5 SearchStrategy Unit Tests (`tests/cache/test_search_strategy.py`)

Test content:
- QdrantWeightedRrfKnnStrategy construction + basic search
- QuerySpec fusion_weights + backend_hints correctly passed (mock storage.search() to capture spec)
- step_filter="all": no filter, full chain through InMemoryBackend
- step_filter="exact": mock storage.search(), assert QuerySpec.filters.step_range==(step, step) (InMemoryBackend doesn't support step_range, not full chain)
- step_filter="window": mock storage.search(), assert correct QuerySpec.filters.step_range (same as above)
- SearchContext fields correctly passed
- Protocol compliance check

---

### Phase 2: Config System

**Goal**: YAML config -> dataclass tree -> component factory.

#### 3.2.1 Config Dataclass Tree (`config.py`)

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
               - QdrantWeightedRrfKnnStrategy (components/search_strategy.py)
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
    type: str = "qdrant_weighted_rrf_knn"
    top_k: int = 1
    step_filter: str = "all"        # "all" | "exact" | "window"
    step_window: int = 5
    rrf_k: int = 60                 # backend-specific, passed via backend_hints
    candidate_multiplier: int = 5   # backend-specific, passed via backend_hints
    # fusion_weights: None = auto-generated from keys.weight (handled by config loading logic)

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

**Note**: Does not include ServerConfig / PolicyConfig / DebugConfig / CollectConfig. Those parameters remain in serve_policy.py's tyro CLI. Full YAML-ization is a separate future task.

#### 3.2.2 YAML Loading Function (within `config.py`)

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
    # 1. Read YAML file content
    # 2. Regex-replace ${VAR:-default} patterns
    # 3. yaml.safe_load() to parse
    # 4. Recursively build CacheConfig dataclass tree
    # 5. Run validate_cache_config() validation
    ...

def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR} and ${VAR:-default} patterns with environment values.

    Data flow: raw YAML text -> regex substitution -> text with env values resolved
    """
    ...
```

#### 3.2.3 Startup Validation (within `config.py`)

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
    1. Enabled fields in keys must appear in backend.vector_dims
    2. Keys of backend.vector_dims must be a subset of CACHE_QUERY_FIELDS
    3. Only "cp1" and "cp3" are allowed in checkpoints (currently supported checkpoints)
    4. key_builder type validity
    5. gate/judge/search_strategy type validity
    6. key_builder.type <-> enabled keys cross-validation:
       When type="placeholder", only robot_state may have enabled=true;
       any other field with enabled=true triggers an error (prevents config that's written but ineffective at runtime)
    7. step_filter value validity: must be one of "all" | "exact" | "window"
    """
    ...
```

Error message format examples:
```
ConfigValidationError: keys.vision_0 is enabled but not found in backend.vector_dims.
  Enabled keys: ['vision_0', 'robot_state']
  Backend vector_dims: {'robot_state': 32}
  Fix: add 'vision_0' to backend.vector_dims or set keys.vision_0.enabled=false

ConfigValidationError: keys.vision_0 is enabled but key_builder type 'placeholder'
  only supports: ['robot_state'].
  Fix: set keys.vision_0.enabled=false, or use a key_builder that supports vision fields
```

#### 3.2.4 Component Factory Function (within `config.py`)

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
    # 1. Build Timer
    timer = SystemTimer(
        enabled=config.timer.enabled,
        buffer_size=config.timer.buffer_size,
        output_csv_dir=config.timer.output_csv_dir,
    )

    # 2. Build Backend + CacheStorage
    backend = _build_backend(config.backend)
    storage = CacheStorage(backend)

    # 3. Build KeyBuilder
    #    Pass in enabled fields list (extracted from keys config)
    enabled_fields = [name for name, kf in _keys_iter(config.keys) if kf.enabled]
    key_builder = _build_key_builder(config.key_builder, enabled_fields)

    # 4. Build Gate / Judge / SearchStrategy per checkpoint
    #    fusion_weights auto-generated from keys config
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

#### 3.2.5 Default YAML File (`cache.yaml`)

```yaml
# OpenPI Cache Configuration
# Config file for the cache subsystem. Specify via command line:
#   uv run scripts/serve_policy.py --cache_config cache.yaml
#   uv run scripts/serve_policy.py --cache_config configs/experiment_rrf.yaml
# Server/policy/debug/collect parameters are still passed via tyro CLI.

enabled: true             # Master switch

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null    # null = print to terminal only, no CSV output

# Query vector field configuration
# enabled: whether this field participates in cache key building and search
# weight: fusion weight (passed to SearchStrategy)
# ⚠️ Current key_builder.type=placeholder only supports robot_state;
#    all other fields must have enabled set to false, otherwise startup will error.
#    More fields can be enabled after future KeyBuilder refactoring.
# task_key filtering: not yet implemented, requires task normalization pipeline
keys:
  vision_0:    { enabled: false, weight: 1.0 }
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }
  robot_state: { enabled: true,  weight: 1.0 }

key_builder:
  type: placeholder       # "placeholder" = uses robot_state only

# Per-checkpoint configuration: CP1 and CP3 can be set independently
checkpoints:
  _defaults: &cp_defaults
    gate:
      type: always_search
    search_strategy:
      type: qdrant_weighted_rrf_knn
      top_k: 1
      step_filter: all    # "all" | "exact" | "window"
      step_window: 5      # Only effective when step_filter=window
      rrf_k: 60           # Qdrant RRF fusion parameter (ignored by in_memory backend)
      candidate_multiplier: 5  # Qdrant prefetch = top_k x multiplier (ignored by in_memory)

  cp1:
    <<: *cp_defaults
    judge:
      type: threshold
      threshold: 0.98     # ⚠️ Placeholder value, needs calibration with real data
      # Threshold meaning depends on search mode:
      #   Single-field cosine (robot_state only): score in [-1, 1], 0.98 = very high similarity
      #   Multi-field RRF fusion: score is a small positive number, different scale,
      #     cannot directly reuse cosine thresholds
      # Threshold must be recalibrated after switching search modes

  cp3:
    <<: *cp_defaults
    judge:
      type: threshold
      threshold: 0.95     # ⚠️ Placeholder value, needs calibration with real data
      # Same as above: threshold is tied to search mode, recalibrate after switching

backend:
  type: in_memory         # "in_memory" | "qdrant"
  vector_dims:
    robot_state: 32       # Must align with enabled=true fields in keys
  qdrant:
    url: ${QDRANT_URL:-http://localhost:6333}
    collection_name: openpi_cache
    prefer_grpc: false
    grpc_port: 6334
    request_timeout: 30
```

---

### Phase 3: serve_policy.py Modifications

#### 3.3.1 New `--cache_config` Parameter

**serve_policy.py Coupling map update**:
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

**Args dataclass changes** (one new field added, all existing fields preserved):
```python
@dataclasses.dataclass
class Args:
    # ... all existing parameters preserved unchanged ...

    # New: cache config file path. When specified, loads cache components from YAML,
    # overriding --cache and --timing_csv_dir parameters.
    cache_config: str | None = None
```

**main() changes**:
```python
def main(args: Args) -> None:
    _configure_torchinductor_cache_dir()
    policy = create_policy(args)  # Existing path unchanged
    policy_metadata = policy.metadata

    # Cache mode: two paths
    if args.cache_config is not None:
        # YAML path: load all cache components from config file
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
        # Existing path: simple interceptor, no orchestrator
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

**Backward compatibility notes**:
- All existing tyro CLI parameters, `create_policy()`, `create_default_policy()`, `DEFAULT_CHECKPOINT` are fully preserved
- The `--cache` simple path (no orchestrator) remains as-is
- The new `--cache_config` path uses YAML loading + full cache component assembly
- Mutual exclusion logic: `--cache_config` takes priority over `--cache`

---

### Phase 4: Tests

#### 3.4.1 Config Tests (`tests/cache/test_config.py`)

| Test Case | Description |
|-----------|-------------|
| `test_load_default_config` | Load cache.yaml, verify CacheConfig dataclass tree structure is correct |
| `test_env_var_substitution` | `${VAR:-default}` substitution logic |
| `test_validation_keys_vs_vector_dims` | Enabled key not in vector_dims -> error |
| `test_validation_key_builder_vs_enabled_keys` | placeholder + vision_0 enabled -> error (Q1) |
| `test_validation_invalid_checkpoint` | Invalid checkpoint name -> error |
| `test_validation_invalid_step_filter` | Invalid step_filter value -> error |
| `test_build_cache_components` | Factory function outputs correct component types and configuration |
| `test_build_and_assemble` | Load from YAML -> build -> construct Orchestrator, verify assembly doesn't error (S6) |
| `test_yaml_anchor_merge` | After YAML anchor merge, CP1/CP3 are independently configured |
| `test_fusion_weights_from_keys` | keys.weight correctly distributed to SearchStrategy |
| `test_wrapper_order` | After config combining cache+record+collect, verify wrapper chain type order (Q7) |

#### 3.4.2 SearchStrategy Tests (`tests/cache/test_search_strategy.py`)

| Test Case | Description |
|-----------|-------------|
| `test_qdrant_weighted_rrf_knn_basic_search` | Basic QuerySpec / storage delegation flow |
| `test_query_spec_fusion_params` | fusion_weights + backend_hints correctly present in QuerySpec |
| `test_step_filter_all` | step_filter="all" doesn't add filter |
| `test_step_filter_exact` | step_filter="exact": mock storage.search(), verify QuerySpec.filters.step_range==(step, step) |
| `test_step_filter_window` | step_filter="window": mock storage.search(), verify correct QuerySpec.filters.step_range |
| `test_search_context_fields` | SearchContext fields correctly passed |
| `test_protocol_compliance` | QdrantWeightedRrfKnnStrategy satisfies SearchStrategy Protocol |

#### 3.4.3 Update Existing Tests

After Orchestrator constructor signature changes, update:
- `tests/cache/test_orchestrator.py`
- `tests/cache/test_interceptor.py`
- `tests/cache/conftest.py` (InMemoryBackend changed to import from production path)

Wrap single-value gate/judge into `{CP1: gate, CP3: gate}` dict, add search_strategies dict.

---

## 4. Implementation Order

```
Phase 1  SearchStrategy Component (Independent of Config)
  1.1  storage_types.py — QuerySpec extension (fusion_weights + backend_hints)
  1.2  components/search_strategy.py — SearchContext + Protocol + QdrantWeightedRrfKnnStrategy
  1.3  backends/in_memory_backend.py — Promote from conftest to production code
  1.4  orchestrator.py — Integrate per-checkpoint dict + step counter + on_task_begin() + SearchContext construction
  1.5  interceptor.py — on_task_begin() forwards to orchestrator (resets step_counter)
  1.6  Update existing tests (Orchestrator signature change + conftest import path + interceptor task lifecycle)
  1.7  test_search_strategy.py — New tests (all full chain + exact/window mock assertions)

Phase 2  Config System
  2.1  config.py — CacheConfig dataclass tree + YAML loading + validation + factory
  2.2  cache.yaml — Default config file
  2.3  test_config.py — New tests (including assembly integration test)

Phase 3  serve_policy.py Modifications
  3.1  Add --cache_config parameter + YAML loading + component assembly
  3.2  Manual verification: uv run scripts/serve_policy.py --cache_config cache.yaml starts normally

Phase 4  Wrap-up
  4.1  uv run pytest full regression
  4.2  Update logs/README.md
  4.3  Update src/openpi/cache/README.md
```

**Dependency relationships**: Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 (strict order)

---

## 5. Design Principles Reminder

1. **Components don't know about Config**: All components receive plain Python values via constructors; they don't import the config module
2. **Config is the sole factory**: Reads YAML -> instantiates components -> injects into Orchestrator. Tests can bypass Config and construct components directly
3. **Gradual integration**: Retain existing tyro CLI main path unchanged; `--cache_config` only takes over the cache subsystem. Full YAML-ization is a separate future task
4. **Comment conventions**: All new code follows the project's coupling annotation conventions (Coupling map / Data flow / DEPENDS ON / CONSUMED BY / IF CHANGED)
5. **No over-engineering**: Only implement features confirmed during discussion; don't add CLI override, profile system, dynamic hot-reload, etc.
6. **Config must not be dead code**: Parameters writable in YAML must take effect at runtime; currently unsupported combinations fail-fast with errors at validate time

---

## 6. Component Stability Assessment

### New Files

| File | Description | Expected Status |
|------|-------------|-----------------|
| `components/search_strategy.py` | SearchContext + `SearchStrategy` Protocol + `QdrantWeightedRrfKnnStrategy` implementation | 🟡 Unit test covered, not integration-verified |
| `backends/in_memory_backend.py` | InMemoryBackend production implementation (promoted from conftest) | 🟡 Indirectly test-covered (existing Orchestrator/Interceptor tests use it) |
| `config.py` | CacheConfig dataclass tree + YAML loading + validation + component factory | 🟡 Unit test covered, not verified with real model |
| `cache.yaml` | Default cache config file | ✅ Stable (pure data file) |

### Modified Files (Risk Changes)

| File | Modification | Post-modification Status | Pre-modification Status |
|------|-------------|-------------------------|------------------------|
| `orchestrator.py` | Constructor signature change: gate/judge/search_strategy changed to `dict[CheckpointID, ...]`; new step counter; check() constructs SearchContext and delegates to SearchStrategy | 🟡 Unit tests need full re-run | 🟡 Unit tests passing |
| `storage_types.py` | QuerySpec gains `fusion_weights` + `backend_hints` fields (with defaults) | ⚠️ High risk (backend_hints channel needs Backend adaptation) | ⚠️ High risk |
| `interceptor.py` | on_task_begin() forwards to orchestrator.on_task_begin() (resets step_counter) | 🟡 Unit tests need updating | 🟡 FakeModel tests passing |
| `scripts/serve_policy.py` | New `--cache_config` parameter + YAML loading path (existing CLI unchanged) | 🟡 Manual verification + assembly integration test | ✅ Stable |
| `tests/cache/conftest.py` | InMemoryBackend import path change | ✅ Pure path change | ✅ |

### Unstable Components Detail

#### `components/search_strategy.py` (New, 🟡)
- QdrantWeightedRrfKnnStrategy holds a CacheStorage reference for search, sharing the same CacheStorage instance with Orchestrator — concurrency safety relies on CacheStorage's RLock
- All three step_filter modes fully implemented; current_step obtained from SearchContext (populated by Orchestrator's step counter)
- fusion_weights pass-through path: Config -> SearchStrategy -> QuerySpec.fusion_weights -> Backend; key mismatch at any link causes search anomalies
- backend_hints is dict[str, Any] with no type constraints — if Backend misspells a key when reading, it silently degrades to default values
- SearchContext.task_key is always None currently (task_key source not implemented); does not affect search behavior

#### `backends/in_memory_backend.py` (New, 🟡)
- Promoted from conftest; logic unchanged, only file location change plus Coupling map annotations added
- Ignores QuerySpec.fusion_weights and backend_hints (single-vector cosine search); asymmetric with Qdrant behavior — tests may mask fusion pass-through issues
- Does not support step_range filter: `supported_filters()` returns `frozenset({"checkpoint_id"})`, not including `"step_range"`. CacheStorage will raise UnsupportedFilterError for QuerySpec containing step_range
- Therefore step_filter="exact"/"window" tests use mock storage.search() to assert QuerySpec contents, not full InMemoryBackend chain; step_filter="all" tests use real full chain

#### `config.py` (New, 🟡)
- `${VAR:-default}` environment variable substitution in YAML loading uses regex; edge cases (nested `$`, `}` in values) not exhaustively tested
- `validate_cache_config()` adds key_builder.type <-> enabled keys cross-validation, preventing config semantic gaps
- Factory function `build_cache_components()` has implicit dependency order for component instantiation: Backend -> CacheStorage -> SearchStrategy
- Dataclass tree uses `field(default_factory=...)` for nested defaults; merge logic for partial YAML overrides needs careful testing
- Whether YAML anchor `<<: *defaults` merge correctly overrides cp1/cp3 judge depends on YAML parser behavior; needs explicit test verification

#### `orchestrator.py` (Modified, 🟡)
- Breaking constructor signature change: all existing tests and interceptor construction calls need synchronized updates
- `self._gates[checkpoint_id]` / `self._judges[checkpoint_id]` will KeyError if dict is missing a CP key; needs validation at construction time
- New `_step_counter`: only incremented on CP1 check, reset by on_task_begin(). Multiple CP checks share the same step value
- write() path doesn't go through SearchStrategy, still directly calls `self._storage.insert()` — consistency of two paths sharing the storage reference

#### `storage_types.py` (Modified, ⚠️ High risk)
- QuerySpec gains `fusion_weights` (generic) and `backend_hints` (backend-specific), both defaulting to None
- QdrantVectorStore.search() needs synchronized modification to read parameters from spec; inconsistent backend_hints key spelling silently degrades
- InMemoryBackend ignores both fields; asymmetric with Qdrant behavior — tests may mask pass-through issues

#### `interceptor.py` (Modified, 🟡)
- on_task_begin() adds forwarding: `self._orchestrator.on_task_begin()` resets step_counter
- If this forwarding is missed, step_filter="exact"/"window" step semantics will drift across tasks
- on_task_end() doesn't need forwarding (orchestrator has no per-task cleanup logic)

#### `scripts/serve_policy.py` (Modified, 🟡)
- Existing tyro CLI completely untouched; only new `--cache_config` parameter and corresponding branch added
- `--cache_config` and `--cache` mutual exclusion logic needs explicit handling (cache_config takes priority)
- Wrapper order (cache -> record -> collect) is explicitly annotated with comments, has test verification

---

## 7. Known Risks and Boundaries

| Risk | Mitigation |
|------|------------|
| QdrantBackend.search() needs to read params from spec.backend_hints | Synchronize modification of qdrant_backend.py after Phase 1.1 extends QuerySpec |
| backend_hints key spelling inconsistency causes silent degradation | QdrantVectorStore uses `.get(key, default)` for missing hints and logs a warning |
| InMemoryBackend doesn't support step_range filter | supported_filters() explicitly declares no support; CacheStorage will raise UnsupportedFilterError |
| PlaceholderKeyBuilder doesn't support enabled_fields parameter | validate_cache_config() restricts only robot_state to be enabled; to be connected when KeyBuilder is refactored later |
| Whether YAML anchor `<<: *defaults` correctly overrides judge | test_yaml_anchor_merge explicitly verifies |
| --cache_config and --cache specified simultaneously | cache_config takes priority; log warning that --cache is ignored |

---

## 8. Out of Scope for This Iteration

- KeyBuilder refactoring (supporting multi-field construction) — separate future step
- CP3 DeferredWriter implementation — Step 6
- QdrantBackend integration tests — requires Qdrant instance
- ThresholdJudge threshold calibration — requires real data
- task_key filtering — requires task normalization pipeline
- Full server YAML-ization (ServerConfig/PolicyConfig/DebugConfig/CollectConfig) — separate future task
- CLI override / profile system — not doing

---

## 9. Defense Module

### 9.1 Questions

1. Centralized `keys` configuration is a core decision from the discussion, but the plan's "Known Risks and Boundaries" also states "Phase 1 doesn't change KeyBuilder, Config factory temporarily ignores enabled_fields." If `keys.*.enabled` doesn't actually take effect in the current implementation, are these YAML fields current configuration or future placeholders? Is there a "config writable but not used at runtime" semantic gap here?

2. Is the `SearchStrategy.search(query_keys, checkpoint_id)` interface too narrow? The plan already assigns `step_filter`, `step_window`, and future `task_key` filtering to SearchStrategy, but the current signature can't access `current_step` or the normalized `task_key`. Relying solely on an internal step counter seems insufficient for multi-task, reconnection, and future extension scenarios.

3. The scope of `step_filter="all" | "exact" | "window"` is currently inconsistent. The plan body says to implement and test `"window"`, but "Known Risks and Boundaries" says the initial implementation only supports `"all"`. What is the actual deliverable? If `"exact"`/`"window"` aren't being done now, should the config layer directly prohibit them rather than exposing them first?

4. `BackendConfig.type` exposes `"in_memory"` externally, but the current in-memory backend only exists on the test side. Should production config allow selecting a test-only backend? If yes, where does its production implementation file go; if no, why is it in `config.py`'s public interface?

5. `QuerySpec` gains `rrf_k`, `candidate_multiplier`, `fusion_weights`, essentially lifting Qdrant's fusion details to the public type layer. Does this abstraction re-bind `storage_types.py` and SearchStrategy back to Qdrant semantics? If FAISS or a single-vector backend is added in the future, are these fields ignored, reused, or do we keep adding backend-specific parameters to QuerySpec?

6. Why does `checkpoints` use `dict[str, CheckpointConfig]` while also strictly limiting keys to only `"cp1"` and `"cp3"` during validation? Since the key space is closed, wouldn't a strongly typed dataclass be more appropriate, reducing string keys, runtime KeyErrors, and redundant validation?

7. The existing `scripts/serve_policy.py` wrapper order is cache -> record -> collect. After the entry point refactoring, will this order be explicitly preserved and verified? If the order drifts, `CollectionPolicy`'s `_model` lookup, timing boundaries, and recorded content would all be affected.

8. `CachePayload.task_key` and the backend's `task_key` filter already have hooks in the storage layer, but the existing write path has no actual source; `src/openpi/cache/interceptor.py` currently only writes `action_chunk` for CP1 write. If this plan requires SearchStrategy/Backend to support task filtering, where does task_key come from and how is it normalized? If not doing it this time, should the default YAML explicitly not expose task filter?

9. Is placing `server.yaml` in the repository root appropriate? The filename is very generic, but the current plan wants it to carry both full server parameters and fine-grained cache parameters. Once multi-environment, multi-experiment, and multi-backend configs appear, won't a single root-level file quickly become unmanageable?

### 9.2 Suggestions

1. Suggest narrowing the current deliverable to "cache config integration into `scripts/serve_policy.py --cache` path", preserving the existing tyro CLI main path; after the cache config is truly working, do full server YAML-ization as a separate task. This better fits the current step's goals and is easier to regression-test.

2. Suggest splitting `keys` config into "currently effective fields" and "future reserved fields" in two layers of expression. If the current `PlaceholderKeyBuilder` only supports `robot_state`, then the default config and validation should explicitly only allow this one to be enabled, avoiding pseudo-configuration.

3. Suggest upgrading SearchStrategy's input to a lightweight `SearchContext` early, accommodating at minimum `checkpoint_id`, `current_step`, `task_key`, and future runtime metadata. Otherwise, starting with a minimal signature now means the interface will need to change again later for dynamic filter conditions.

4. Suggest adopting a fail-fast strategy for unimplemented config items. If the current version only supports `step_filter="all"`, explicitly error in `validate_config()` rather than exposing `"exact"` and `"window"` to users only to degrade at runtime.

5. Suggest not directly exposing the test-only `"in_memory"` backend to production YAML; either promote it to a production implementation under `src/openpi/cache/backends/`, or restrict it entirely to tests.

6. Suggest adding a minimal assembly test for `scripts/serve_policy.py`, at least covering wrapper order after config loading, the default policy path, and the combined behavior of `cache + record + collect`. Otherwise, the primary risk of this entry-point-level modification will fall on manual verification.

7. Suggest adding a startup pre-check for the Qdrant collection schema in `config.py`, especially surfacing the compatibility risk of switching `robot_state` dimension from the historical 14 to the current 32 early, rather than only erroring at query/insert time.

8. Suggest clearly documenting in the default YAML comments that `ThresholdJudge`'s default threshold is tied to the search mode, explicitly noting that "single-field cosine" and "multi-field RRF" cannot directly reuse the same threshold defaults, avoiding a default config that appears usable but has no calibration basis.

---

### 9.3 Question Responses

#### Q1: keys.*.enabled Semantic Gap
**Accepted.** The issue is valid. Currently PlaceholderKeyBuilder is hardcoded to only use `robot_state`; enabled fields in the keys config are written but don't take effect at runtime — this is pseudo-configuration.

**Modification plan**:
- `validate_config()` adds validation: when `key_builder.type == "placeholder"`, only `robot_state` may have enabled=true; other fields with enabled=true trigger an error with the message "PlaceholderKeyBuilder only supports robot_state"
- YAML comments clearly indicate which fields are currently usable and which require future KeyBuilder implementation to take effect
- This way the config structure remains complete (no need to add fields later), but the validation layer ensures current semantics don't have gaps

#### Q2: SearchStrategy.search() Interface Too Narrow
**Accepted.** `current_step` and `task_key` are dynamic runtime information that should not be self-maintained by SearchStrategy (internal counter would be incorrect during multi-task reconnection).

**Modification plan**:
- Introduce a lightweight `SearchContext` dataclass, replacing the loose parameters of `search(query_keys, checkpoint_id)`:
```python
@dataclass
class SearchContext:
    query_keys: dict[str, torch.Tensor]
    checkpoint_id: CheckpointID
    current_step: Optional[int] = None
    task_key: Optional[str] = None
```
- Protocol signature changed to `search(ctx: SearchContext) -> list[SearchResultLite]`
- Orchestrator is responsible for constructing SearchContext (populated from runtime info passed by Interceptor)
- In the initial implementation, `current_step` and `task_key` are None; QdrantWeightedRrfKnnStrategy treats None as equivalent to step_filter="all"

#### Q3: step_filter Delivery Scope Inconsistency
**Accepted.** The plan body and risk section are self-contradictory.

**Modification plan** (second revision after defense):
- All three modes fully implemented: `"all"` / `"exact"` / `"window"`
- `current_step` provided by Orchestrator's per-task step counter, passed in via SearchContext
- Step counter implementation is simple (an integer + on_task_begin reset + CP1 check increment), no need to defer
- Test plan retains `test_step_filter_all`, `test_step_filter_exact`, `test_step_filter_window`

#### Q4: in_memory Backend Exposed to Production YAML
**Accepted.** A test-only backend should not appear in production config options.

**Modification plan**:
- Promote InMemoryBackend from `tests/cache/conftest.py` to `src/openpi/cache/backends/in_memory_backend.py` as a production implementation
- Rationale: Cache end-to-end validation needs a lightweight backend independent of Qdrant (this week's plan item 1.2 "first run through with InMemoryBackend"); it's not just a test tool, but a legitimate development/debugging backend
- After promotion, add module-level Coupling map annotations
- conftest changes to import from the production path

#### Q5: QuerySpec Fusion Fields Binding to Qdrant Semantics
**Partially accepted.** `rrf_k` and `candidate_multiplier` are indeed Qdrant RRF fusion-specific concepts. But `fusion_weights` is generic (any multi-vector backend may need weighted fusion).

**Modification plan**:
- QuerySpec only retains the generic field: `fusion_weights: Optional[dict[str, float]]`
- `rrf_k` and `candidate_multiplier` move back to backend-specific config — but not to QdrantBackendConfig (rejected during discussion); instead, as internal parameters of SearchStrategy, QdrantWeightedRrfKnnStrategy passes them when constructing QuerySpec via the `backend_hints: Optional[dict[str, Any]]` additional field
- This way QuerySpec stays backend-agnostic; backend-specific parameters pass through the hints channel; FAISS and other new backends can ignore unrecognized hints

```python
@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None
    fusion_weights: Optional[dict[str, float]] = None      # Generic
    backend_hints: Optional[dict[str, Any]] = None          # Backend-specific
```

QdrantWeightedRrfKnnStrategy at construction time:
```python
spec = QuerySpec(
    ...,
    fusion_weights=self._fusion_weights,
    backend_hints={"rrf_k": self._rrf_k, "candidate_multiplier": self._candidate_multiplier},
)
```

QdrantVectorStore.search() reads from `spec.backend_hints`. InMemoryBackend / FAISS ignores it.

#### Q6: checkpoints Using dict[str, ...] vs Strong Typing
**Rejected.** Rationale:

1. YAML naturally expresses as a string-key mapping; dict directly corresponds to the YAML parse result, no extra conversion layer needed
2. Currently supports CP1/CP3, but CP2 is an explicit TODO item (Step 7); the key space is not truly closed
3. dict is used at the Config layer (pure data); by the Orchestrator layer it's already converted to `dict[CheckpointID, Component]` (strongly typed enum key); there's no string key KeyError risk at runtime
4. Changing to strongly typed dataclass (`cp1: CheckpointConfig, cp3: CheckpointConfig`) would require modifying the dataclass definition + YAML structure + factory logic in three places when CP2 is added, while dict only needs a new key in YAML

The "only allow cp1/cp3" restriction in validation is version gating, not a type design flaw — later versions simply relax it.

#### Q7: Wrapper Order Preservation
**Accepted.** The wrapper order cache -> record -> collect has implicit behavioral dependencies (CollectionPolicy needs to access the underlying policy's `_model`).

**Modification plan**:
- In the modified serve_policy.py code, explicitly add comments marking wrapper order and its rationale
- Phase 4 tests add one case: `test_wrapper_order_preserved` (verify wrapper chain's type order after combining cache+record+collect via config)

```python
# Wrapper ordering matters:
#   1. InferenceInterceptor (innermost — needs direct Policy access)
#   2. PolicyRecorder (records interceptor's output)
#   3. CollectionPolicy (outermost — hooks into model internals via _model)
# DO NOT reorder without verifying CollectionPolicy._model lookup.
```

#### Q8: task_key Source
**Accepted.** The current write path indeed has no task_key source; task filtering is not being done this iteration.

**Modification plan**:
- In this implementation, task_key-related functionality is not activated:
  - SearchContext.task_key defaults to None
  - QdrantWeightedRrfKnnStrategy doesn't add task_key filter when task_key=None
  - CachePayload construction keeps task_key as default empty string
- YAML does not expose task_key-related config items (not written into the search_strategy block)
- YAML comment explains: `# task_key filtering: not yet implemented, requires task normalization pipeline`

#### Q9: server.yaml Placement Location
**Rejected.** Rationale:

1. Placing config files in the project root is common practice (docker-compose.yml, pyproject.toml, .pre-commit-config.yaml are all in the root)
2. At the current project scale, one default config file in the root doesn't cause clutter
3. Multi-environment/multi-experiment scenarios are handled by specifying different files via `--config configs/experiment_xxx.yaml`; no need to move the default file's location
4. The discussion phase already decided "default file is server.yaml in the root directory"; no new information warrants overturning this

### 9.4 Suggestion Responses

#### S1: Narrow Delivery Scope, Preserve tyro CLI Main Path
**Accepted.** This is consistent with Q1's "config must not have semantic gaps" spirit, and better fits weekly plan item 1.2's goal (first run through cache chain with InMemoryBackend).

**Modification plan**:
- `serve_policy.py` preserves the existing tyro CLI as the main path; Args dataclass is not deleted
- New `--config` optional parameter: when `--config` is specified, takes the YAML path; otherwise follows the original tyro path
- Phase 3 modification scope narrowed: only loads cache components from YAML when `args.cache and args.config`; server/policy/debug/collect parameters still go through tyro

```python
@dataclasses.dataclass
class Args:
    # ... all existing parameters preserved ...
    # New: cache config file path. When specified, overrides --cache and related parameters.
    cache_config: str | None = None
```

```python
def main(args: Args) -> None:
    policy = create_policy(args)  # Existing path unchanged

    if args.cache_config is not None:
        # YAML path: load cache components from config file
        from openpi.cache.config import load_cache_config, build_cache_components
        cache_config = load_cache_config(args.cache_config)
        components = build_cache_components(cache_config)
        orchestrator = CacheOrchestrator(...)
        policy = InferenceInterceptor(policy, timer=components["timer"], orchestrator=orchestrator)
    elif args.cache:
        # Existing path: simple interceptor, no orchestrator
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer
        timer = SystemTimer(enabled=True, output_csv_dir=args.timing_csv_dir)
        policy = InferenceInterceptor(policy, timer=timer)

    # record / collect wrapper order unchanged
    ...
```

This means:
- Config dataclass tree simplified: only needs the `CacheConfig` portion, no need for full `ServePolicyConfig` structure
- `server.yaml` simplified to `cache.yaml` (only contains cache config block)
- Phase 2 workload reduced by approximately 40% (no need for ServerConfig/PolicyConfig/DebugConfig/CollectConfig)
- Full YAML-ization as a separate future task

#### S2: Keys Config Layering
**Already covered in Q1 response.** validate_config() restricts available enabled fields based on current key_builder.type.

#### S3: SearchContext
**Already covered in Q2 response.** Introduced SearchContext dataclass.

#### S4: Fail-fast for Unimplemented Config Items
**Principle accepted, Q3 already revised to implement all.** All three step_filter modes are being done; fail-fast restriction no longer needed. validate_cache_config() still validates step_filter value legality (must be one of "all"/"exact"/"window").

#### S5: in_memory Backend Promotion
**Already covered in Q4 response.** Promoted to `src/openpi/cache/backends/in_memory_backend.py`.

#### S6: serve_policy.py Minimal Assembly Test
**Accepted.** But scope narrowed (aligned with S1's reduced delivery scope).

**Modification plan**:
- New `test_build_and_assemble` in `tests/cache/test_config.py`: Load from YAML -> build_cache_components -> construct Orchestrator, verify assembly doesn't error
- No real Policy object needed; use mock to verify component injection chain
- Wrapper order test (Q7) also placed in this file

#### S7: Qdrant Collection Schema Startup Pre-check
**Rejected.** Rationale:

1. config.py's responsibility is "read config + instantiate components"; it should not take on Qdrant runtime health checks
2. `vector_dims` config validation (keys <-> vector_dims consistency) is already covered in validate_config()
3. Whether the Qdrant collection exists and whether the schema matches is the responsibility of QdrantVectorStore initialization — checking in the backend constructor is more appropriate
4. The dimension migration from 14->32 is a deployment-level issue (requires collection rebuild), not solvable by config validation

#### S8: YAML Threshold and Search Mode Binding Documentation
**Accepted.** This is a pure comment improvement — zero cost, high benefit.

**Modification plan**:
Add comment next to judge.threshold in YAML:
```yaml
judge:
  type: threshold
  threshold: 0.98     # ⚠️ Placeholder value, needs calibration with real data
  # Threshold meaning depends on search mode:
  #   Single-field cosine (robot_state only): score in [-1, 1], 0.98 = very high similarity
  #   Multi-field RRF fusion: score is a small positive number, different scale,
  #     cannot directly reuse cosine thresholds
  # Threshold must be recalibrated after switching search modes
```

---

### 9.5 Post-Defense Revision Summary

| ID | Source | Verdict | Impact |
|----|--------|---------|--------|
| Q1 | Question | ✅ Accepted | validate_config() adds key_builder.type <-> enabled keys cross-validation |
| Q2 | Question | ✅ Accepted | Introduced SearchContext dataclass, replacing search() loose parameters |
| Q3 | Question | ✅ Accepted (second revision) | All three step_filter modes fully implemented, Orchestrator adds step counter |
| Q4 | Question | ✅ Accepted | InMemoryBackend promoted to backends/ production code |
| Q5 | Question | 🟡 Partially accepted | rrf_k/candidate_multiplier moved to backend_hints; fusion_weights kept in QuerySpec |
| Q6 | Question | ❌ Rejected | dict[str, CheckpointConfig] remains unchanged |
| Q7 | Question | ✅ Accepted | Explicit comments on wrapper order + test verification |
| Q8 | Question | ✅ Accepted | No task_key filtering this iteration; YAML doesn't expose related config |
| Q9 | Question | ❌ Rejected | server.yaml location unchanged |
| S1 | Suggestion | ✅ Accepted | Preserve tyro CLI, add --cache_config optional parameter, Config scope narrowed to CacheConfig |
| S2 | Suggestion | ✅ Covered | Same as Q1 |
| S3 | Suggestion | ✅ Covered | Same as Q2 |
| S4 | Suggestion | ✅ Covered | Same as Q3 |
| S5 | Suggestion | ✅ Covered | Same as Q4 |
| S6 | Suggestion | ✅ Accepted | test_config.py adds assembly integration test |
| S7 | Suggestion | ❌ Rejected | Qdrant schema check is backend constructor's responsibility, not config's |
| S8 | Suggestion | ✅ Accepted | YAML comments add threshold-to-search-mode binding explanation |

### 9.6 Post-Defense Plan Revisions (✅ All synced back to body text)

The following items have been retroactively modified in the Plan body based on defense conclusions:

1. ✅ **Section 0 Overview**: Updated to preserve tyro CLI + add --cache_config, describes SearchContext, InMemoryBackend promotion
2. ✅ **Section 1 New Files**: Added `backends/in_memory_backend.py`; `server.yaml` -> `cache.yaml`
3. ✅ **Section 2 Modified Files**: serve_policy.py description changed to "add --cache_config"; added conftest/test_orchestrator/test_interceptor modifications
4. ✅ **Phase 1.1**: QuerySpec only retains `fusion_weights` + adds `backend_hints`; removed `rrf_k` and `candidate_multiplier`
5. ✅ **Phase 1.2**: SearchStrategy Protocol signature changed to `search(ctx: SearchContext)`; added `SearchContext` dataclass; QdrantWeightedRrfKnnStrategy implements all three step_filter modes; rrf_k/candidate_multiplier placed in backend_hints
6. ✅ **Phase 1.3**: Added 3.1.4 InMemoryBackend promotion (moved from conftest to backends/)
7. ✅ **Phase 1.3 Orchestrator**: Added step counter + on_task_begin() reset + SearchContext construction
8. ✅ **Phase 1.5 Tests**: Updated to three step_filter tests + SearchContext test
9. ✅ **Phase 2.1**: Config scope narrowed to `CacheConfig` (removed ServePolicyConfig/ServerConfig, etc.)
10. ✅ **Phase 2.2**: load_config -> load_cache_config, returns CacheConfig
11. ✅ **Phase 2.3**: validate_cache_config() adds key_builder.type <-> enabled keys validation + step_filter legality validation
12. ✅ **Phase 2.5**: `server.yaml` -> `cache.yaml`; YAML judge.threshold comment adds search mode binding explanation; keys comment notes current restriction to placeholder; task_key comment explains not implemented
13. ✅ **Phase 3.1**: serve_policy.py preserves tyro CLI, adds `--cache_config`; wrapper order explicitly commented
14. ✅ **Phase 4 Tests**: Added wrapper order test, assembly integration test, key_builder <-> enabled keys validation test, step_filter_exact test
15. ✅ **Section 4 Implementation Order**: Updated step numbers and descriptions
16. ✅ **Section 5 Design Principles**: Updated to "gradual integration" and "config must not be dead code"
17. ✅ **Section 6 Stability Assessment**: Comprehensive update of new/modified file tables and unstable component details
18. ✅ **Section 7 Known Risks**: Updated risk table, removed resolved items, added backend_hints/UnsupportedFilterError/mutual exclusion parameters, etc.
19. ✅ **Section 8 Out of Scope**: Added task_key filtering, full YAML-ization
