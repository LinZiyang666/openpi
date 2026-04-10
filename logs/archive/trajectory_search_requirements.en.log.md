# Trajectory Search Requirements Document

**Status**: `Plan`
**Date**: 2026-04-06

---

## 1. Motivation

The existing cache search is **single-step matching**: at each inference step, a query key is constructed for the current step, compared against each independent entry in the database by similarity, and the most similar entry is returned. This approach discards **temporal context** -- two different episodes may have very similar observations at a given step, but their historical trajectories are entirely different, carrying different semantics.

**Goal**: Introduce **trajectory search**, which compares not only the current step but also traces back several previous steps' keys during search, leveraging temporal consistency to improve match quality.

---

## 2. Core Approach

### 2.1 Query Side: Buffering Historical Keys

During an inference episode, the query key generated at each step must be **buffered**. This gives the query side a complete key sequence from the start of the episode to the current step:

```
query_history = [key_step_0, key_step_1, ..., key_step_t]
```

Configurable **trajectory depth (trajectory_depth)**:
- `depth=1`: Current step only (degenerates to existing behavior)
- `depth=2`: Current step + previous step
- `depth=3`: Current step + two previous steps
- And so on

When `current_step < depth - 1` (insufficient history), only the available steps are used.

### 2.2 Database Side: Doubly Linked List + Trajectory Identifier

Database entries need **doubly linked list pointers** and a **trajectory identifier**, enabling traversal forward and backward along a trajectory from any entry, and identifying trajectory membership:

```
CacheEntry new fields:
    prev_ids:      list[str]       # List of predecessor entry ids
    next_ids:      list[str]       # List of successor entry ids
    trajectory_id: Optional[str]   # Trajectory membership identifier
    step_idx:      Optional[int]   # Existing field, step index within the trajectory
```

**Field rules:**

- `prev_ids` / `next_ids`: Use `list[str]` instead of a single value, reserving capacity for future trajectory branching/merging
  - The first entry in a trajectory has `prev_ids=[]`, the last has `next_ids=[]`
  - **Warning: Current phase constraint: lists contain at most one element**, but traversal and search interfaces are designed for multi-branch from the start (recursive implementation, see Section 2.3)
  - **Warning: Risk note: Management rules for trajectory_id in multi-branch/merge scenarios are not yet designed; deferred to the trajectory optimization phase**
- `trajectory_id`: Identifies which trajectory an entry belongs to. Initially assigned per episode during ingest (`trajectory_id = episode_id`); may change through subsequent operations:
  - **Merge/concatenation**: Two segments of entries are unified under a new `trajectory_id`, connecting head-to-tail pointers (the tail of the first segment appends the head id of the second segment to `next_ids`, and vice versa for `prev_ids`)
  - **Split**: Pointers are broken at the split point; the latter half is reassigned a new `trajectory_id`
  - **Pruning**: Entry is deleted, neighboring pointers are updated, `trajectory_id` remains unchanged
  - **Branching**: Multiple successor entries can share the same predecessor; each branch can have a different `trajectory_id` (Warning: specific rules deferred to the trajectory optimization phase)
- `step_idx`: Position index within the trajectory. Initially assigned in chronological order within an episode during ingest. Gaps may appear after pruning (continuity not required); renumbering may be needed after merging
- All new fields have `Optional` or empty list defaults, maintaining backward compatibility with existing entries

**Data source:** **libero_spatial** dataset. Steps within each episode are ordered chronologically; linked list relationships are automatically constructed during ingest.

### 2.3 Trajectory Comparison During Search

Search is structured in three layers:

**Layer 1 (per-field similarity)**: Same as the existing implementation. For each field within a single step (vision_0, vision_1, vision_2, prompt_emb, robot_state), compute cosine similarity or L2 distance.

**Layer 2 (per-step fusion)**: Same as the existing implementation. Fuse the similarity values of multiple fields within the same step into a single similarity value for that step via RRF or weighted score sum.

**Layer 3 (cross-step fusion, new)**: Through **recursive backtracking**, traverse all predecessor paths of a candidate entry, accumulating weighted similarity layer by layer, supporting both linear chain and branching tree topologies.

#### Recursive Trajectory Similarity Algorithm

```python
def _recursive_trajectory_sim(
    entry_id: str,
    query_history: list[dict[str, Tensor]],  # Query-side history [current, t-1, t-2, ...] (newest-first)
    depth: int,                               # Remaining backtrack depth (initial call = len(weights) - 1)
    max_depth: int,                           # Maximum backtrack depth (= len(weights) - 1, unchanged during recursion)
    weights: list[float],                     # Per-layer weights [w_current, w_t-1, w_t-2, ...] (newest-first)
    accumulated_sim: float,                   # Accumulated similarity passed from upper layer
) -> list[float]:                             # Returns trajectory_sim for all complete paths
    """Recursively compute trajectory similarity.

    Index mapping: idx = max_depth - depth
      depth=max_depth (current step) → idx=0 → query_history[0]=current, weights[0]=w_current
      depth=0         (oldest step)  → idx=max_depth → query_history[-1], weights[-1]

    Per layer:
      1. Compute step_similarity between the current entry and the corresponding query step (Layer 1 + Layer 2 fusion)
      2. Add weighted contribution to accumulated_sim
      3. If depth limit reached or no predecessors → return [accumulated_sim] (one complete path)
      4. Otherwise recursively traverse all prev_ids, collecting scores from all branch paths
    """
    idx = max_depth - depth
    step_sim = step_similarity(query_history[idx], entries[entry_id].query_keys)
    accumulated_sim += weights[idx] * step_sim

    if depth == 0 or not entries[entry_id].prev_ids:
        return [accumulated_sim]  # Leaf node, return score for one complete path

    all_scores = []
    for prev_id in entries[entry_id].prev_ids:
        all_scores.extend(_recursive_trajectory_sim(
            prev_id, query_history, depth - 1, max_depth, weights, accumulated_sim
        ))
    return all_scores
```

#### Branch Aggregation Strategy

The recursion returns a `list[float]` containing trajectory_sim for all complete paths:
- **No branching** (current phase): Returns `[one score]`, used directly
- **With branching**: Returns multiple scores, which need to be aggregated into a single value. Aggregation strategy is configurable:
  - `max`: Take the best path (optimistic matching)
  - `mean`: Take the average (comprehensive matching)
  - `min`: Take the worst path (conservative matching)
  - **Warning: Current phase defaults to `max`; the optimal choice for multi-branch aggregation strategy is to be determined experimentally during the trajectory optimization phase**

#### Weight Strategy

- The most recent step has the highest weight, decreasing with distance (e.g., exponential decay or linear decay)
- Specific weight values are configurable via YAML
- When `query_history` has fewer entries than depth, only the available steps are used; missing steps do not contribute to the computation

### 2.4 Search Flow Illustration

**Linear chain (current phase):**

```
Query Side (buffered during inference):    Database Side (linked list):

step 0: key_q0                    entry_A0 → entry_A1 → entry_A2 → entry_A3 (trajectory A)
step 1: key_q1                    entry_B0 → entry_B1 → entry_B2 → entry_B3 (trajectory B)
step 2: key_q2 (current)

depth=3 search for candidate entry_A2:
  recursive call: _recursive_trajectory_sim(entry_A2, [key_q2, key_q1, key_q0], depth=2, ...)
    depth=2: sim += w0 * similarity(key_q2, entry_A2.key)  → recurse prev_ids=[entry_A1]
    depth=1: sim += w1 * similarity(key_q1, entry_A1.key)  → recurse prev_ids=[entry_A0]
    depth=0: sim += w2 * similarity(key_q0, entry_A0.key)  → return [sim]
  result: [trajectory_sim]  → single path, used directly
```

**Branching tree (future scenario):**

```
                entry_X0 → entry_X1 ─┐
                                      ├→ entry_C2 → entry_C3
                entry_Y0 → entry_Y1 ─┘

depth=3 search for candidate entry_C3:
  recurse entry_C3 → entry_C2 → prev_ids=[entry_X1, entry_Y1]
    branch 1: entry_X1 → entry_X0 → return sim_path_1
    branch 2: entry_Y1 → entry_Y0 → return sim_path_2
  result: [sim_path_1, sim_path_2] → aggregate(max/mean/min) → final score
```

---

## 3. Components Requiring Changes

### 3.1 Data Structures (`storage_types.py`)

**CacheEntry new fields:**

```python
prev_ids:      list[str] = field(default_factory=list)   # List of predecessor entry ids
next_ids:      list[str] = field(default_factory=list)   # List of successor entry ids
trajectory_id: Optional[str] = None                       # Trajectory membership identifier
# step_idx already exists, no addition needed
```

- All new fields have Optional / empty list defaults, maintaining backward compatibility with existing entries

**CachePayload changes:**

- **Remove `next_action_chunk`**: With the `next_ids` linked list, the next step's action is obtained via `next_ids[0]` → fetch entry → `payload.action_chunk`, eliminating redundant storage
- **Retain `intermediates` / `denoising_num_steps`**: CP2 warm-start specific; set to None during runtime writes

**QuerySpec extensions:**

```python
# New fields (all Optional, None degenerates to existing behavior)
trajectory_history: Optional[list[dict[str, Tensor]]] = None  # Query-side historical key sequence
trajectory_weights: Optional[list[float]] = None               # Per-layer weights [w_current, w_t-1, ...]
```

### 3.2 Database Backends (`backends/`)

**No new Backend subclasses**; extend within the existing Backend's `search()` method:

- **InMemoryBackend**: Add a trajectory fusion branch in the existing `search()`. When `trajectory_history=None`, follow the original logic; when non-empty, execute: KNN initial retrieval → candidate backtracking along `prev_ids` → per-step similarity → weighted reranking (recursive implementation, see Section 2.3)
- **QdrantBackend**: When `trajectory_history` is non-empty and depth > 1, `raise NotImplementedError("trajectory search not supported in QdrantBackend")`; when depth=1 or no history, execute normally without error

### 3.3 Data Ingest

- When importing data from libero_spatial, group by episode, sort by step
- Construct doubly linked list: `entry[i].next_ids = [entry[i+1].id]`, `entry[i+1].prev_ids = [entry[i].id]`
- The first entry in an episode has `prev_ids=[]`, the last has `next_ids=[]`
- `trajectory_id` is initially set to `episode_id`

### 3.4 History Buffer: Component Self-Management

**No centralized tracker**; each component accumulates history from its own input and decides how to use it:

| Component | Input Data | History Content | Implemented This Phase |
|-----------|-----------|-----------------|----------------------|
| SearchStrategy | `ctx.query_keys` (CPU, after dimensionality reduction) | query_keys history buffer | Yes |
| Gate | `cached_data` (GPU, raw) | cached_data history | No -- reserve interface with detailed comments |
| Judge | `cached_data` (GPU, raw) | cached_data history | No -- reserve interface with detailed comments |

**New interfaces required per component:**

- `on_episode_start()` / `reset()`: Clear the history buffer, called by the Orchestrator during `on_task_begin()` / `on_episode_start()`
- `record_action(action_chunk: Tensor)`: Receive the action broadcast by the Orchestrator. **Must be a pure local buffer operation (append to list); callbacks to Backend / CacheStorage / Orchestrator or acquiring any external locks are prohibited**

**Action broadcast**: After the Orchestrator obtains the action, it calls `record_action()` on each component in sequence. Deadlock risk: none (`check()` has already returned and locks have been released; the entire process executes sequentially; concurrent connections have independent per_connection_components).

### 3.5 SearchStrategy: Upgrading Existing Strategies

**No new standalone trajectory strategy type**. All existing strategies (WeightedRrfKnnStrategy, WeightedScoreSumKnnStrategy, etc.) uniformly gain full trajectory search capabilities:

- Maintain query_keys + action_chunk history buffer
- `on_episode_start()` lifecycle management
- `record_action()` interface
- During search, construct `trajectory_history` / `trajectory_weights` from the buffer, populate into QuerySpec, and pass to the Backend
- When `trajectory_depth=1` (or unconfigured): Recursion depth is 1, naturally producing the same result as single-step search (degenerate behavior of the same code path, not branching to legacy logic)

### 3.6 Orchestrator: Write Flow Refactoring

**Original logic**: Each step calls `orchestrator.write()` → immediate `storage.insert()`

**New logic**: Read-write separation

1. **Read-only during inference**: Each step calls `orchestrator.buffer_for_write()`, buffering query_keys + action_chunk
2. **Batch write at episode end**: During `on_episode_end()`:
   - Call `WritePolicy.should_write(episode_record)` to decide whether to write
   - If writing: Construct a complete CacheEntry chain (with `prev_ids` / `next_ids` / `trajectory_id`) from buffered data, `batch_insert()` in one operation

**WritePolicy** (pluggable write switch):

```python
@dataclass
class StepRecord:
    query_keys: dict[str, Tensor]        # CPU float32
    action_chunk: Tensor                 # CPU float32, required (steps without actions are not written)

@dataclass
class EpisodeRecord:
    steps: list[StepRecord]              # Buffered query_keys + action_chunk per step
    task_key: str                         # Task identifier
    miss_by_checkpoint: dict[CheckpointID, int]  # e.g. {CP1: 3, CP3: 50}
    total_steps: int                      # Total number of steps

class WritePolicy(Protocol):
    def should_write(self, episode_record: EpisodeRecord) -> bool: ...
```

Implementation types: `on_any_miss` (default, write if any miss occurred during the episode), `always`, `never`

### 3.7 Config and YAML Configuration

**SearchStrategyConfig new fields:**

```python
# SearchStrategyConfig dataclass additions
trajectory_depth: int = 1                          # Default 1, degenerates to single-step search
trajectory_weights: Optional[list[float]] = None   # Default None, not needed when depth=1
```

```yaml
search_strategy:
  type: weighted_score_sum_knn              # Type name unchanged
  top_k: 5
  trajectory_depth: 3                       # Trajectory depth (1=degenerates to single-step, defaults to 1 if unconfigured)
  trajectory_weights: [1.0, 0.5, 0.25]     # Per-step weights, newest-first (from current to historical)
  # ... existing per-field configs remain
```

**CacheConfig top-level addition of WritePolicyConfig:**

```yaml
enabled: true

write_policy:
  type: on_any_miss        # on_any_miss | always | never

timer:
  # ...
checkpoints:
  # ...
```

**Config dataclass changes:**

```python
@dataclass
class WritePolicyConfig:
    type: str = "on_any_miss"

@dataclass
class CacheConfig:
    write_policy: WritePolicyConfig = field(default_factory=WritePolicyConfig)
    # ... remaining existing fields ...
```

**Factory function**: `_build_write_policy(cfg)` returns the corresponding implementation based on `cfg.type`; unknown types raise `ConfigValidationError`. The dict returned by `build_cache_components` gains a new `write_policy` key.

---

## 4. Backward Compatibility

- **All existing SearchStrategies are upgraded**: WeightedRrfKnnStrategy, WeightedScoreSumKnnStrategy, etc. all gain full trajectory search capabilities (history buffer maintenance, `on_episode_start()` lifecycle, `record_action()` interface, constructing `trajectory_history` / `trajectory_weights` into QuerySpec). No new standalone trajectory strategy type is added; all existing strategies uniformly support it
- **Controlled via YAML configuration `trajectory_depth` and `trajectory_weights`**: All strategies read these two parameters
- **When `trajectory_depth=1` (or unconfigured)**: Recursion depth is 1, naturally producing the same result as single-step search (degenerate behavior of the same code path, not branching to legacy logic)
- **CacheEntry new fields all have Optional / empty list defaults**: No impact on existing data
- **InMemoryBackend**: Trajectory branch is added within the existing `search()`; when `trajectory_history=None`, the original logic is followed

---

## 5. Items to Confirm

- [x] Specific defaults for trajectory depth and weights → **Configured via YAML, no hardcoded defaults; to be determined experimentally**
- [x] Relationship between initial candidate top-K and final return count → **No new parameters needed**, see Q7 conclusion
- [x] Whether inference-time actions need to be cached → **Confirmed yes**: Each step's action_chunk must also be buffered in the history buffer
- [x] Whether QdrantBackend needs simultaneous support → **Only InMemoryBackend is implemented first**, see Q1 conclusion
- [x] Whether the doubly linked list is built in the ingest script or auto-maintained during backend.insert → **Offline ingest batch construction + runtime batch write at episode end**, see Q5 conclusion

---

## 6. Discussion Conclusions

### Q1: History Buffer Ownership and Data Flow (Confirmed)

**Conclusion: Component self-management, no centralized tracker**

#### History Buffering Strategy

Each component accumulates history from its own input and decides how to use it:

| Component | Input Data | History Content | Implemented This Phase |
|-----------|-----------|-----------------|----------------------|
| SearchStrategy | `ctx.query_keys` (CPU, after dimensionality reduction) | query_keys history buffer | Yes |
| Gate | `cached_data` (GPU, raw) | cached_data history | No -- reserve interface with detailed comments explaining future use and extension approach |
| Judge | `cached_data` (GPU, raw) | cached_data history | No -- reserve interface with detailed comments explaining future use and extension approach |

#### Action Broadcast

After the Orchestrator obtains an action (whether from cache hit or model inference), it broadcasts to each component. Two-phase recording:
- **Phase 1**: Each component automatically accumulates input data during its normal call flow (Gate/Judge receive cached_data, SearchStrategy receives query_keys)
- **Phase 2**: The Orchestrator uniformly broadcasts action_chunk to components that need it

**Deadlock risk analysis: No risk**, because:
- Single-connection scenario: `check()` has already returned and all locks have been released before broadcasting; the entire process executes sequentially
- Concurrent-connection scenario: Each connection has independent per_connection_components; no cross-locking exists

**Implementation constraint**: Each component's `record_action()` method **must be a pure local buffer operation** (append to list), **callbacks to** Backend / CacheStorage / Orchestrator or acquiring any external locks are **prohibited**. This constraint must be explicitly stated in interface comments.

#### Lifecycle

Each component needs `on_episode_start()` / `reset()` interfaces, called by the Orchestrator during `on_task_begin()` / `on_episode_start()`.

#### Trajectory Search Logic Pushed Down to Backend

**Do not make two Backend calls from SearchStrategy**; instead, encapsulate the trajectory search logic within Backend.search() to complete in one call:

- Extend `QuerySpec` with new `trajectory_history` and `trajectory_weights` fields
- SearchStrategy is responsible for maintaining the query_keys history buffer and populating the history into QuerySpec during construction
- Backend.search() internally completes: KNN initial retrieval → candidate backtracking → per-step similarity → weighted reranking
- When `trajectory_history=None`, behavior fully degenerates to existing behavior

#### Backend Implementation Strategy

- **InMemoryBackend**: Extend the trajectory fusion branch within the existing `search()` method; do not create a new search method
- **QdrantBackend**: When `trajectory_history` is non-empty and depth > 1, raise `NotImplementedError("trajectory search not supported in QdrantBackend")`; when depth=1 or no history, execute normally without error
- No new Backend subclasses

#### Gate/Judge Signatures

Gate/Judge signatures are not modified this time. Known caveat: The data-level difference between Gate/Judge using GPU `cached_data` vs. SearchStrategy using CPU `query_keys` is an existing architectural issue not addressed this time. If trajectory-aware gate/judge is needed in the future, signatures can be extended incrementally.

### Q3: Whether action_chunk Participates in Search (Confirmed)

**Conclusion: Not this time**

- action_chunk is only buffered in each component's history buffer
- Whether it participates in similarity computation is decided by each component's specific implementation (SearchStrategy / Gate / Judge)
- The SearchStrategy, Gate, and Judge implemented this time do not use action data
- In future extensions, components can retrieve action from the buffer for action-conditioned retrieval or consistency checking

### Q4: Data Structure Design (Confirmed)

**Conclusion: Doubly linked list (list form) + trajectory_id + step_idx**

Not a pure episode_id + step_idx index scheme, nor a pure linked list scheme, but a combination of both:

- **Doubly linked list (`prev_ids: list[str]` / `next_ids: list[str]`)**: O(1) traversal, supports pruning/merging/splitting and other dynamic pointer operations. Lists reserve capacity for future trajectory branching/merging
- **`trajectory_id`**: Trajectory membership identifier, initially equals episode_id, may change after merge/split; used for trajectory-level batch queries and differentiation
- **`step_idx`** (existing field): Position within trajectory; gaps allowed after pruning

**Warning: Current phase constraint**: `prev_ids` / `next_ids` contain at most one element; multi-branch/merge logic deferred to the trajectory optimization phase

Rationale:
- A pure episode_id + step_idx scheme has step_idx gaps after pruning, requiring sorted lookups for backtracking (no longer O(1)); merging/insertion requires renumbering
- A pure linked list scheme lacks trajectory-level membership identifiers, unable to quickly answer "which trajectory does this entry belong to"
- Doubly linked list + trajectory_id balances traversal efficiency and trajectory management flexibility
- List form reserves extension space for future branching/merging; current phase implementation is simple (only access `[0]`)

### Q5: Ingest and Runtime Write Strategy (Confirmed)

#### Offline Ingest

Build the linked list in bulk offline in the ingest script:
- Group by episode, sort by step
- Fill in all `prev_ids` / `next_ids` / `trajectory_id` at once (initial `trajectory_id = episode_id`)
- Write via `backend.batch_insert()`; the Backend does not need to understand linked list semantics

#### Runtime Write: Batch Write at Episode End

**No more per-step inserts**; switched to read-write separation:
- **Read-only cache during inference**: SearchStrategy and other components accumulate query_keys and action_chunk history during their normal call flow
- **Batch write at episode end**: Construct a complete CacheEntry chain (with prev_ids / next_ids / trajectory_id) from buffered historical data, write in one `batch_insert` call

**Data needed for writing** (all buffered during the episode):
- Per-step `query_keys` (SearchStrategy's history buffer)
- Per-step `action_chunk` (buffered via the action broadcast mechanism)
- `task_key` (episode-level, one is sufficient)
- `intermediates` / `denoising_num_steps`: Set to None during runtime writes (CP2 not implemented; these two fields are coupled as CP2 warm-start data, currently always None; setting None passes existing validation)

**Changes to existing `Orchestrator.write()` / `write_with_keys()` flow**:
- Original logic: Each step calls `orchestrator.write()` → immediate `storage.insert()`
- New logic: Each step calls `orchestrator.buffer_for_write()` → buffer into Orchestrator's write buffer → construct linked list and `batch_insert()` during `on_episode_end()`

#### Pluggable Write Switch (WritePolicy)

Whether to write the trajectory at episode end is decided by a pluggable WritePolicy:

```python
class WritePolicy(Protocol):
    def should_write(self, episode_record: EpisodeRecord) -> bool: ...
```

Current default implementation: If any CP1 cache miss or gate skip occurred during the episode (`miss_by_checkpoint.get(CP1, 0) > 0`) → write the complete trajectory. By default, only CP1 is checked (CP3 skeleton always misses; it should not trigger writes). Can be replaced with other strategies in the future.

**Config and YAML design:**

New `WritePolicyConfig` dataclass, attached at the `CacheConfig` top level:

```python
# config.py additions
@dataclass
class WritePolicyConfig:
    type: str = "on_any_miss"       # Write policy type
    # Future extensible parameters, such as miss_ratio_threshold, etc.

# CacheConfig new field
@dataclass
class CacheConfig:
    enabled: bool = False
    write_policy: WritePolicyConfig = field(default_factory=WritePolicyConfig)
    # ... remaining existing fields ...
```

Corresponding YAML configuration:

```yaml
# cache.yaml
enabled: true

write_policy:
  type: on_any_miss          # Default: write if any miss occurred during the episode
  # type: always             # Write every episode
  # type: never              # No writes (read-only mode)

timer:
  # ...
checkpoints:
  # ...
```

**Factory function:**

```python
# config.py additions
def _build_write_policy(cfg: WritePolicyConfig):
    if cfg.type == "on_any_miss":
        return OnAnyMissWritePolicy()
    elif cfg.type == "always":
        return AlwaysWritePolicy()
    elif cfg.type == "never":
        return NeverWritePolicy()
    else:
        raise ConfigValidationError(f"Unknown write_policy.type '{cfg.type}'")
```

**The dict returned by build_cache_components / build_per_connection_components gains a new `write_policy` key**, used by the Orchestrator during `on_episode_end()` to call `write_policy.should_write(episode_record)` to decide whether to write.

**Validation rules:**
- `write_policy.type` must be a known type; otherwise `validate_cache_config()` raises an error
- `WritePolicyConfig` is added to the `_CONFIG_TYPES` registry to support recursive YAML parsing

#### CachePayload Field Cleanup

**Remove `next_action_chunk`**:
- With the `next_ids` linked list, the next step's action is obtained via `next_ids[0]` → fetch entry → `payload.action_chunk`
- The redundant `next_action_chunk` storage and the DeferredWriter backfill mechanism are no longer needed
- Update CP3 design documentation accordingly: obtain the next step's action via linked list traversal, replacing the `next_action_chunk` + DeferredWriter approach
- **The specific logic for CP3 forward rolling (how many steps, when triggered, scheduling) is out of scope for this design**

**Retain `intermediates` / `denoising_num_steps`**:
- These fields are CP2 warm-start specific; CP2 will genuinely need them in the future and they cannot be replaced by the linked list
- Set to None during runtime writes; passes existing validation without issues

#### Known Trade-offs

- **Crash data loss**: If the process crashes mid-episode, all buffered data for that episode is lost. Acceptable for experimental scenarios
- **Memory usage**: All steps' query_keys + action_chunk must be held in memory during the episode. LIBERO episodes are typically a few hundred steps, with a few small tensors (KB-scale) per step; total is manageable

### Q7: top-K and Final Return Count (Confirmed)

**Conclusion: No new parameters needed; `top_k` semantics unchanged**

Trajectory search is only a change in scoring method -- using trajectory similarity instead of single-step similarity to score candidates. The return value is still the `top_k` candidate entries for the current step (not entire trajectories). If Backend.search() internally needs a larger initial candidate pool, the existing `candidate_multiplier` parameter can be used.

### Q8: Handling Missing Entries During Backtracking (Confirmed)

**Conclusion: Naturally handled by recursion; no additional logic needed**

Both cases are covered by the recursive termination condition:
- **Insufficient history** (e.g., depth=5 but current step=3): Query-side history is insufficient; recursion terminates early
- **Broken chain** (e.g., entry deleted causing `prev_ids=[]`): Recursion encounters empty `prev_ids` and directly returns the accumulated score

In both cases, missing steps contribute no similarity (equivalent to 0). The existing recursive logic rigorously covers this; no special handling is needed.

### Q9: trajectory_depth and weights Defaults (Confirmed)

**Conclusion: Configured via YAML, no hardcoded defaults; to be determined experimentally**
