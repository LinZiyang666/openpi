# Step 5 Config System — Discussion Log

> Start date: 2026-04-04
> Status: Under discussion

---

## Discussion 1: Review of existing CLI arguments in serve_policy.py

### Parameter overview

```
serve_policy.py
├── --env            → EnvMode enum (aloha/aloha_sim/droid/libero), selects environment and determines default checkpoint
├── --port           → int, WebSocket server port, default 8000
├── --default_prompt → str|None, fallback prompt (injected when client does not provide a prompt)
├── --record         → bool, enables PolicyRecorder, saves obs+action to .npy step by step (for debugging)
├── --collect        → bool, enables CollectionPolicy, forward hooks collect embeddings and write HDF5
│   └── --collect_dir → str, root directory for collected data, default ./data
├── --cache          → bool, enables InferenceInterceptor, runs CP1 check/write path
│   └── --timing_csv_dir → str|None, timing CSV output directory (None = print to terminal only)
└── policy (tyro union branch)
    ├── Default()        → uses preset checkpoint for the selected env
    └── Checkpoint()     → custom loading
        ├── --policy.config → training config name
        └── --policy.dir    → checkpoint directory path
```

### Decision

**The config system must cover all of the above parameters** — none may be omitted.

---

## Discussion 2: Inventory of configurable items in the cache system

By reading all source code in the cache subsystem, we extracted every configurable item from hardcoded values or constructor parameters.

### InferenceInterceptor (interceptor.py)

Very few config items — this is the assembly entry point. Core configuration is distributed across lower-level components.

| Configurable item | Current value / source | Description |
|-------------------|------------------------|-------------|
| `timer` | Passed at construction or defaults to `SystemTimer()` | Timer instance |
| `orchestrator` | Passed at construction or `None` | None = pure staged inference, non-None = cache enabled |

### SystemTimer (timing.py)

| Configurable item | Current value | Description |
|-------------------|---------------|-------------|
| `enabled` | `True` | False = measure() becomes a zero-cost no-op |
| `buffer_size` | `10_000` | Ring buffer size |
| `output_csv_dir` | `None` (already exposed to CLI) | Directory for writing CSV at task end |

### ThresholdJudge (components/judge.py)

| Configurable item | Current value | Description |
|-------------------|---------------|-------------|
| `cp1_threshold` | `0.98` | CP1 cosine threshold, placeholder value not yet calibrated |
| `cp3_threshold` | `0.95` | CP3 cosine threshold, placeholder value not yet calibrated |

### PlaceholderKeyBuilder (components/key_builder.py)

No configurable items currently. Only uses `ROBOT_STATE` 32 dimensions. In the future, which query fields to use could be configurable.

### AlwaysSearchGate (components/gate.py)

No configurable items currently (always returns True). In the future, gate strategy type could be selectable.

### QdrantBackendConfig (backends/qdrant_backend.py)

| Configurable item | Current value | Description |
|-------------------|---------------|-------------|
| `url` | `http://localhost:6333` | Qdrant service address |
| `collection_name` | `openpi_cache` | Collection name |
| `vector_dims` | `{vision_0: 1024, prompt_emb: 2048, robot_state: 14}` | Named vector dimensions |
| `prefer_grpc` | `False` | Whether to prefer gRPC |
| `grpc_port` | `6334` | gRPC port |
| `request_timeout` | `30` | Timeout in seconds |
| `rrf_k` | `60` | RRF fusion parameter k |
| `candidate_multiplier` | `5` | prefetch limit = top_k x multiplier |
| `fusion_weights` | `None` (equal weights) | Per-field fusion weights |

### InMemoryBackend (tests/cache/conftest.py, test-only)

Only one config item: `vector_dims`.

---

## Discussion 3: SearchStrategy component abstraction + search parameter ownership

### Background

Search logic is scattered across two layers:
- `top_k` is hardcoded in Orchestrator (`orchestrator.py:143`, hardcoded `top_k=1`)
- `rrf_k`, `fusion_weights`, `candidate_multiplier` are in QdrantBackendConfig
- `step_filter` / `step_window` exist only in exp/ experiment code, not integrated into the main system

Search strategy will change frequently (top_k mechanism, fusion method, custom re-ranking, etc.), and each change requires modifying both Orchestrator and Backend.

### Decision: Add a pluggable SearchStrategy component

SearchStrategy becomes the fifth pluggable component of Orchestrator (alongside Gate and Judge), serving as the sole entry point for database search. **All search parameters are owned by SearchStrategy.**

**Responsibility division**:

| Component | Responsibility | Does NOT handle |
|-----------|---------------|-----------------|
| **Orchestrator** | Pure orchestration (collect -> gate -> search -> judge) | top_k, filter, fusion, etc. |
| **SearchStrategy** | Search strategy + sole entry point for database communication | hit/miss determination |
| **Backend** | Pure KNN executor | No strategy decisions |

**SearchStrategy is responsible for**:
- Holding all search parameters: top_k, step_filter, step_window, rrf_k, fusion_weights, candidate_multiplier
- Receiving query_keys + checkpoint_id -> constructing a complete QuerySpec (including filters + fusion parameters)
- Calling CacheStorage.search()
- Future extensions: multi-round search, re-ranking, custom fusion

**Parameter passing path**:

```
SearchStrategy (holds search parameters)
  → constructs QuerySpec (with top_k, filters, fusion parameters)
  → CacheStorage.search(spec) (validates, locks, forwards)
    → Backend.search(spec) (executes search using parameters from spec)
```

Backend config degenerates to pure connection configuration (url, collection_name, etc.) and no longer holds search strategy parameters.

**Protocol interface**:

```python
class SearchStrategy(Protocol):
    def search(
        self,
        query_keys: dict[str, torch.Tensor],
        checkpoint_id: CheckpointID,
    ) -> list[SearchResultLite]:
        ...
```

**Orchestrator call chain change**:

```python
# Before: Orchestrator assembles QuerySpec itself
query_keys = key_builder.build(cp)
spec = QuerySpec(query_keys=query_keys, top_k=1, checkpoint_id=cp)
results = storage.search(spec)

# After: fully delegated to SearchStrategy
query_keys = key_builder.build(cp)
results = search_strategy.search(query_keys, checkpoint_id=cp)
```

**Storage access division**:
- SearchStrategy holds a CacheStorage reference, responsible for the search path (search)
- Orchestrator also holds a CacheStorage reference, responsible for the write path (insert/fetch_payload)
- Both share the same CacheStorage instance

---

## Discussion 4: Per-checkpoint configuration + unified keys configuration

### Decision 1: Independent CP1/CP3 configuration

CP1 and CP3 can use different Gate, Judge, and SearchStrategy instances.

Orchestrator uses dict mapping at construction:

```python
CacheOrchestrator(
    storage=storage,
    key_builder=key_builder,
    gates={CP1: always_gate, CP3: always_gate},
    judges={CP1: threshold_judge_098, CP3: threshold_judge_095},
    search_strategies={CP1: knn_strategy, CP3: knn_strategy_window},
)
```

### Decision 2: Unified keys configuration, dispatched to two layers

```json
"keys": {
    "vision_0":    {"enabled": true,  "weight": 1.0},
    "robot_state": {"enabled": true,  "weight": 10.0},
    "prompt_emb":  {"enabled": false, "weight": 1.0}
}
```

Dispatched at config load time:
- List of fields with `enabled=true` -> passed to KeyBuilder (determines which vectors to extract)
- Corresponding `weight` values -> passed to SearchStrategy's fusion_weights

### Current final config tree

```
CacheConfig
├── enabled: bool
├── timer
│   ├── enabled: bool
│   ├── buffer_size: int
│   └── output_csv_dir: str | None
├── keys                                      # unified config, dispatched to key_builder + search_strategy
│   ├── vision_0:    {enabled: bool, weight: float}
│   ├── vision_1:    {enabled: bool, weight: float}
│   ├── vision_2:    {enabled: bool, weight: float}
│   ├── prompt_emb:  {enabled: bool, weight: float}
│   └── robot_state: {enabled: bool, weight: float}
├── key_builder
│   └── type: "placeholder" | ...
├── checkpoints                                # per-checkpoint configuration
│   ├── cp1
│   │   ├── gate: {type: "always_search"}
│   │   ├── judge: {type: "threshold", threshold: 0.98}
│   │   └── search_strategy:
│   │       ├── type: "qdrant_weighted_rrf_knn"
│   │       ├── top_k: int
│   │       ├── step_filter: "all" | "exact" | "window"
│   │       ├── step_window: int
│   │       ├── rrf_k: int
│   │       ├── candidate_multiplier: int
│   │       └── fusion_weights: dict[str, float] | None  # or auto-generated from keys.weight
│   └── cp3
│       ├── gate: {type: "always_search"}
│       ├── judge: {type: "threshold", threshold: 0.95}
│       └── search_strategy: (same structure as above, may have different parameters)
└── backend                                    # degenerates to pure connection config
    ├── type: "in_memory" | "qdrant"
    ├── vector_dims: dict[str, int]            # for validation
    └── qdrant (only when type=qdrant)
        ├── url: str
        ├── collection_name: str
        ├── prefer_grpc: bool
        ├── grpc_port: int
        └── request_timeout: int
```

---

## Design principle: Relationship between components and config

**SearchStrategy (as well as Gate, Judge, KeyBuilder) are independent pluggable components, not part of the config system.**

- Components do not read config themselves, nor depend on config data structures
- Components receive parameters via constructor (plain Python values) and do not know where the parameters come from
- The config system is an "assembly factory": reads config file -> instantiates components -> injects into Orchestrator
- During testing, components can be constructed directly, completely bypassing the config system

```
Config system (factory)          Components (independent)
──────────────                   ──────────
Read YAML/dataclass              SearchStrategy(top_k=1, rrf_k=60, ...)
  → instantiate components       Gate(...)
  → inject into Orchestrator     Judge(threshold=0.98)
                                 KeyBuilder(...)
```

This means:
- Component Protocol interface design is not affected by config format
- Changing config format (YAML->TOML, CLI->file) does not require any component code changes
- Components can be used and tested independently without the config system

---

## Discussion 5: Config tree coverage gap analysis

### 5.1 Top-level config structure

CacheConfig is only a sub-config. The complete ServePolicyConfig must also cover all serve_policy.py parameters (env, port, default_prompt, record, collect, policy, etc.). To be discussed later.

### 5.2 QuerySpec extension

SearchStrategy is a new component; its introduction inevitably requires modifications to existing implementations (QuerySpec, Backend.search interface, etc.). These modifications are expected, but the detailed plan will be determined when the formal plan is drafted, not during the config discussion phase.

### 5.3 KeyBuilder and key contents

**Available raw data**:

| Field | Dimensions | Available at |
|-------|------------|--------------|
| vision_0/1/2 | 1024 each | After Stage 1 |
| prompt_emb | 2048 | After Stage 1 |
| robot_state | 32 | After Stage 1 |
| action_chunk | [50, 32] | After Stage 3 |

**Decisions**:
- KeyBuilder should allow adding different information per checkpoint — at CP1 time action is empty (not yet inferred), at CP3 time action can be included
- The specific key construction scheme needs to be redesigned, no longer limited to PlaceholderKeyBuilder's pure state mode

**Cross-cycle history memory maintenance**:
- Decision: Each component maintains its own required historical state (e.g., Gate stores previous state, Judge stores previous action)
- Rationale: Most flexible, does not introduce new shared components, does not reduce coding freedom
- Components expose lifecycle methods like clear() through Protocol interfaces

### 5.4 InMemoryBackend promotion

InMemoryBackend currently only exists in `tests/cache/conftest.py`. This is an efficiency optimization concern; not addressed now, will be promoted to production code when needed.

### 5.5 vector_dims consistency validation

Three declarations must be consistent: keys config (which fields are enabled), KeyBuilder (build output dimensions), Backend (vector_dims).

**Decision**: Validate at config load time; ensure good error messages elsewhere.
- Primary validation: Config factory cross-validates field and dimension consistency across keys/key_builder/backend at component instantiation time, failing at startup
- Fallback: CacheStorage runtime dimension validation is retained; error messages must clearly state which field, expected dimensions, and actual dimensions, to help diagnose whether it's a config error or a KeyBuilder output anomaly

---

## Discussion 6: Top-level structure + config format

### Decision 1: Top-level ServePolicyConfig with grouped nesting

```
ServePolicyConfig
├── server: {env, port, default_prompt}
├── policy: {config, dir} | Default
├── debug: {record}
├── collect: {enabled, dir}
└── cache: CacheConfig
```

Rationale: The cache subsystem config is highly complex; flattening it would result in dozens of CLI flags, which is unmaintainable.

### Decision 2: YAML file format

YAML is chosen as the config file format.

Rationale:
- Supports comments, convenient for documenting parameter meanings
- Good readability for deep nesting (natural indentation)
- YAML anchors (`&defaults` / `*defaults`) allow sharing defaults within a file, reducing duplication when CP1/CP3 share parameters
- Environment variable substitution (`${QDRANT_URL:-default}`) can be implemented in a few lines of Python, no extra dependencies needed

Variable capabilities:
- In-file variable references: Use YAML anchor native syntax
- Environment variables: Python-side `${VAR:-default}` substitution at load time
- No third-party dependencies like OmegaConf

---

### Decision 3: No CLI override support — YAML is the single source of truth

- All config parameters are read only from the YAML file; CLI override of individual parameters is not supported
- The CLI is only used to select the config file path: `--config path/to/config.yaml`
- There is a default config file that is auto-loaded when `--config` is not specified

```bash
# Use default config
uv run scripts/serve_policy.py

# Specify config file
uv run scripts/serve_policy.py --config configs/experiment_rrf.yaml
```

Rationale:
- Single source of configuration (YAML only) — no ambiguity about "where does the final value come from"
- Experiments are reproducible: one YAML file fully describes all parameters for a run
- Simplified implementation: no YAML + CLI merge logic needed

### Decision 4: Default file and CLI simplification

- Default config file: `server.yaml` in the project root
- When `--config` is not specified, `server.yaml` is auto-loaded
- serve_policy.py CLI arguments are simplified to only `--config`; all others move into YAML
- Every parameter in the YAML file must have a reasonable default value and a descriptive comment

```bash
# Use default server.yaml
uv run scripts/serve_policy.py

# Specify other config
uv run scripts/serve_policy.py --config configs/experiment.yaml
```

serve_policy.py Args dataclass becomes:

```python
@dataclasses.dataclass
class Args:
    config: str = "server.yaml"
```

---

### Decision 5: Code comment conventions

All new code must follow the project's existing comment conventions (refer to Step 4 components), including:

- **Module-level docstring**: Overview, Coupling map (DEPENDS ON / CONSUMED BY / IF CHANGED), Data flow
- **Class-level docstring**: Responsibility description, Data flow, Coupling relationships
- **Method-level docstring**: Parameter meanings, return values, side effects
- **Inline comments**: Annotate "why" at key decision points, do not explain "what"

Example (following `orchestrator.py` style):
```
Coupling map:
  DEPENDS ON:  ...
  CONSUMED BY: ...
  IF CHANGED:  ...
```

---

## Discussion 7: To be discussed

Pending topics:
- Whether any configurable items have been overlooked
- Whether we can begin drafting the formal plan
