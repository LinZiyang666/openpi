# Cache CP1 In-Memory Implementation Plan

> Status: Plan
> Date: 2026-04-06

---

## Objective

Based on [cache_system_tutorial.md](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/docs/cache_system_tutorial.md) and [cache_experiment_plan.log.md](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/logs/cache_experiment_plan.log.md), implement the `CP1` path needed for the current large-scale experiment.

This round only serves the experiment; there is no aim to build a fully general production-grade system in one shot.

---

## Fixed Scope

- Only `CP1`
- `gate.type = always_search`
- `judge.type = always_hit`
- Database backend is uniformly `in_memory`
- Module implementation retains `vision_2` support; the LIBERO experiment in this round simply disables it in YAML
- `prompt_emb` weight is fixed to `0` in Phase 1 / 1.5
- Layer 1 fixed:
  - `vision_0 / vision_1 / prompt_emb = cosine`
  - `robot_state = L2`
- Layer 2 fixed comparison:
  - `weighted_rrf`
  - `weighted_score_sum`

---

## Design Principles

Strictly follow the module isolation principles in the tutorial:

- `Gate` only decides whether to search; no IO
- `Judge` only decides hit type; no IO
- Only `SearchStrategy` constructs `QuerySpec` and calls `storage.search()`
- Only `Orchestrator` calls `storage.insert()` / `storage.fetch_payload()`
- `Config` only handles YAML parsing, validation, and factory; no business logic
- `Backend` only handles storage, filtering, per-field scoring, fusion, and ranking; no awareness of model implementation details

Additional requirements:

- Dimensionality reduction logic must not be coupled to the backend
- cosine / L2 logic must not be coupled to the judge
- Cross-modal fusion logic must not be written into the orchestrator
- `in_memory` must be upgraded from a "test placeholder backend" to an "experiment primary backend"

---

## Current Implementation Issues

### 1. `in_memory` Backend Severely Insufficient

The current [in_memory_backend.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/backends/in_memory_backend.py) only has:

- Single-field brute-force cosine
- `break` on first matching field
- No `robot_state = L2` support
- No multi-field fusion
- No `weighted_rrf` support
- No `weighted_score_sum` support
- No score normalization
- No `task_key` / `step_range` filtering

It currently suffices only for unit tests, not for experiments.

### 2. `QuerySpec` Insufficient Expressiveness

The current [storage_types.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/storage_types.py) only has:

- `fusion_weights`
- `backend_hints`

Missing the explicit semantics the experiment actually needs:

- Fusion method type
- Per-field similarity definitions
- Score normalization definitions
- `tau`
- Percentile statistics parameters

### 3. `KeyBuilder` Lacks Experiment Dimensionality Reducers

The current [key_builder.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/components/key_builder.py) only has:

- `placeholder`
- `full_original`

Missing:

- `A: mean pool`
- `B1: spatial pooling 4x4`
- `B2: spatial pooling 2x2`
- `C: max pool`

### 4. YAML Schema Cannot Express the Experiment

The current [config.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/config.py) schema can only express placeholder cache, not experiment configurations.

---

## Module Boundaries

### 1. `KeyBuilder / Reducer`

Responsibilities:

- Extract `vision_0`, `vision_1`, `prompt_emb`, `robot_state` from `stage1` output
- Apply dimensionality reduction per field
- Output CPU float32 query keys

Not responsible for:

- Similarity computation
- Score normalization
- Fusion
- Database retrieval

Notes:

- `robot_state` is output as-is
- `vision_0 / vision_1 / vision_2` are all supported at the code level; this round's LIBERO YAML disables `vision_2`
- `vision_0 / vision_1` use A/B1/B2/C based on experiment selection
- `prompt_emb` participates in key construction per the plan, but its weight is fixed to `0` during coarse search
- The logic for splitting `prefix_embs` by modality directly reuses the approach from the current `FullOriginalKeyBuilder`
- The new CP1 builder only changes "how each modality is reduced after slicing," not "how the token sequence is sliced by modality"
- The existing key builder contract also reserves the `CP3` path: `collect()` can cache `stage3.action_chunk`, `build()` can accept `CheckpointID.CP3`
- Therefore the new builder must also maintain this interface compatibility, even though this round of experiments only runs `CP1`
- Specific requirements:
  - `collect()` continues to support caching `stage3.action_chunk`
  - `build()` continues to accept `CheckpointID.CP3`
  - For now, `CP3` can produce the same key as `CP1`; action concat remains deferred and is not implemented in this round

### 2. `SearchStrategy`

Responsibilities:

- Write the current experiment configuration into `QuerySpec`
- Assemble:
  - `top_k`
  - `filters`
  - `fusion_method`
  - `fusion_weights`
  - `field_similarity`
  - `score_normalization`

Not responsible for:

- Actually computing cosine / L2
- Performing fusion
- Deciding hit / miss

Recommendations:

- Retain the boundary that "only SearchStrategy constructs QuerySpec and calls storage.search()"
- The existing `QdrantWeightedRrfKnnStrategy` remains Qdrant-only
- This round adds separate in-memory strategy types for the experiment

### 3. `InMemoryBackend`

Responsibilities:

- Store entries
- Load entries from prebuilt artifacts
- Execute filtering per `QuerySpec`
- Compute per-field scores for each field
- Execute cross-modal fusion
- Sort and return `SearchResultLite`

Not responsible for:

- Dimensionality reduction
- Computing keys from raw data
- Hit determination
- Threshold logic

Recommend using private functions inside `in_memory_backend.py` for layering, rather than adding new files.

This way `InMemoryBackend` itself only handles orchestration:

1. Filter candidate entries
2. Compute raw scores for each field
3. Perform necessary score normalization
4. Call fusion to get final scores
5. Sort and return top-k

### 4. `Judge`

No business logic changes this round.

Fixed to use:

- `AlwaysHitJudge`

Responsibilities remain minimal:

- If `results` is non-empty, pass through top-1

Judge does not participate in:

- Threshold experiments
- Re-scoring
- Multi-field logic

### 5. `Orchestrator`

Minimize changes to responsibilities this round.

Responsibilities remain:

- collect
- gate
- build
- search
- judge
- fetch

It should not know:

- Whether the current fusion is RRF or Score Sum
- How the current fields compute cosine / L2
- What the current normalization parameters are

### 6. `Config / YAML`

Responsibilities:

- Describe the experiment, not execute it
- Validate field and parameter combinations
- Instantiate components

Not responsible for:

- Numerical logic
- Dimensionality reduction computation
- Fusion computation

### 7. `Experiment Runner / Server Control`

Responsibilities:

- Control 100+ experiments running sequentially
- Select which runs to execute
- Select the number of episodes per run
- Record experiment progress and support checkpoint-resume
- Notify the server to switch to a new YAML before each experiment round

Not responsible for:

- Rewriting the LIBERO client main logic
- Regenerating YAML at runtime
- Implementing experiment orchestration logic inside the server

Principles:

- Experiment orchestration belongs in the local control script, not in the backend
- The server only responds to "which YAML to load"
- `examples/libero/main.py` should be reused as much as possible, with only minimal wrapping or minimal modification

---

## Detailed Code Plan

This section converges on the extension points already defined in the tutorial, using only existing scaffolding:

- `components/key_builder.py`
- `components/search_strategy.py`
- `backends/in_memory_backend.py`
- `storage_types.py`
- `config.py`
- `tests/cache/...`
- `cache.yaml` or experiment-specific YAML
- `scripts/serve_policy.py`
- `src/openpi/serving/websocket_policy_server.py`
- `examples/libero/main.py` or its outer wrapper script
- `exp/generate_cache_run_yamls.py`
- `exp/build_in_memory_cache_artifact.py`
- `exp/calibrate_score_sum_stats.py`
- `exp/analyze_cache_results.py`
- `exp/run_cache_experiments.py`

No additional modules beyond those mentioned in the tutorial will be introduced.

### A. `components/key_builder.py`

This is one of the primary extension points for this round.

Implementation approach:

- Add 4 independent key builder classes in the existing file
- Do not create a single large class containing all experiment logic
- Each class corresponds to one reduction method

Planned new classes:

- `CP1MeanPoolKeyBuilder`
- `CP1SpatialPool16KeyBuilder`
- `CP1SpatialPool64KeyBuilder`
- `CP1MaxPoolKeyBuilder`

Implementation approach:

- Externally, retain 4 independent builder types to ensure a 1:1 mapping between YAML and experiment groups
- Internally, do not duplicate 4 copies of the implementation
- Use "4 thin wrapper classes + one set of shared private helpers" for logic reuse
- Do not introduce a separate `reducers.py`

Each class implements the existing `QueryKeyBuilder` protocol:

- `collect()`
- `build()`
- `cached_data`
- `clear()`

Responsibilities of the four classes:

1. `CP1MeanPoolKeyBuilder`
   - Mean pool `vision_0 / vision_1 / prompt_emb`
   - Output `robot_state` as-is
   - `collect()` retains caching of `stage3.action_chunk`
   - `build(CheckpointID.CP3)` remains callable; currently produces the same key as `CP1`
   - Output shapes:
     - `vision_*: [emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - After mean pool the result is already a 1D vector; no additional `flatten` needed
   - Corresponding experiment groups:
     - `A-RRF`
     - `A-SUM`

2. `CP1SpatialPool16KeyBuilder`
   - Pool `vision_0 / vision_1` from 16x16 token grid to 4x4
   - Mean pool `prompt_emb`
   - Output `robot_state` as-is
   - `collect()` retains caching of `stage3.action_chunk`
   - `build(CheckpointID.CP3)` remains callable; currently produces the same key as `CP1`
   - Output shapes:
     - `vision_*: [4, 4, emb_dim] -> flatten -> [16 * emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - Important clarification:
     - `vision_*` must be `flatten`ed after spatial pooling
     - Otherwise it violates the current 1D vector contract of `QuerySpec.query_keys[field] = [dim]`
   - Corresponding experiment groups:
     - `B1-RRF`
     - `B1-SUM`

3. `CP1SpatialPool64KeyBuilder`
   - Pool `vision_0 / vision_1` from 16x16 token grid to 2x2
   - Mean pool `prompt_emb`
   - Output `robot_state` as-is
   - `collect()` retains caching of `stage3.action_chunk`
   - `build(CheckpointID.CP3)` remains callable; currently produces the same key as `CP1`
   - Output shapes:
     - `vision_*: [2, 2, emb_dim] -> flatten -> [4 * emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - Important clarification:
     - `vision_*` must be `flatten`ed after spatial pooling
     - Otherwise it violates the current 1D vector contract of `QuerySpec.query_keys[field] = [dim]`
   - Corresponding experiment groups:
     - `B2-RRF`
     - `B2-SUM`

4. `CP1MaxPoolKeyBuilder`
   - Max pool `vision_0 / vision_1 / prompt_emb`
   - Output `robot_state` as-is
   - `collect()` retains caching of `stage3.action_chunk`
   - `build(CheckpointID.CP3)` remains callable; currently produces the same key as `CP1`
   - Output shapes:
     - `vision_*: [emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - After max pool the result is already a 1D vector; no additional `flatten` needed
   - Corresponding experiment groups:
     - `C-RRF`
     - `C-SUM`

Shared implementation approach:

- Directly reuse existing token offset constants in `key_builder.py`
- Directly reuse `FullOriginalKeyBuilder`'s current modality slicing:
  - `_VISION_OFFSETS`
  - `_PROMPT_START`
  - Slicing logic on `prefix_embs[0]`
- A small number of private helper functions is allowed, e.g.:
  - `_slice_cp1_fields()`
  - `_mean_pool_tokens()`
  - `_max_pool_tokens()`
  - `_spatial_pool_tokens()`

But these helpers stay inside `key_builder.py` only; no new modules are split out.

Key constraints:

- Do not duplicate hard-coded slicing logic in each of the four new builders
- Modality slicing logic should be consolidated into a single shared private implementation within `key_builder.py`
- If the token layout changes in the future, only this one location needs updating
- Each field written to `QuerySpec.query_keys` / `CacheEntry.query_keys` must be a 1D vector `[dim]`
- Therefore only the vision branches of `B1/B2` need explicit `flatten`
- Pooled results from `A/C` and `robot_state` itself are already 1D and do not need additional `flatten`

Explicitly not doing:

- Dynamic reducer registry
- General-purpose reducer framework

### B. `components/search_strategy.py`

This is the second major extension point for this round.

Implementation approach:

- Retain the boundary that "only SearchStrategy constructs `QuerySpec` and calls `storage.search()`"
- Add two experiment-specific strategies in the existing file

Planned new classes:

- `WeightedRrfKnnStrategy`
- `WeightedScoreSumKnnStrategy`

Responsibilities:

1. `WeightedRrfKnnStrategy`
   - Obtains from configuration:
     - `top_k`
     - `step_filter`
     - `fusion_weights`
     - `rrf_k`
     - `field_similarity`
   - Constructs into `QuerySpec`
   - Calls `self._storage.search(spec)`

2. `WeightedScoreSumKnnStrategy`
   - Obtains from configuration:
     - `top_k`
     - `step_filter`
     - `fusion_weights`
     - `field_similarity`
     - `score_normalization`
   - Constructs into `QuerySpec`
   - Calls `self._storage.search(spec)`

Explicitly not doing:

- cosine computation
- L2 computation
- score normalization
- fusion

If extending existing strategy implementations in the same file, the following must be satisfied:

- `QdrantWeightedRrfKnnStrategy` retains explicit Qdrant-only semantics
- The `weighted_rrf` and `weighted_score_sum` in-memory experiment paths are kept separate
- Multiple search strategies must not be mixed into an unreadable pile of `if` statements

### C. `storage_types.py`

This file needs type expansion but remains "backend-agnostic."

What needs to be added:

- Add fields to `QuerySpec` needed for experiment retrieval

Recommended additions:

- `fusion_method: str | None`
- `field_similarity: dict[str, dict[str, Any]] | None`
- `score_normalization: dict[str, dict[str, Any]] | None`
- `backend_hints: dict[str, Any] | None`

Usage:

- `fusion_method` (SearchStrategy drops `_knn` suffix when writing to QuerySpec)
  - `weighted_rrf`
  - `weighted_score_sum`
- `field_similarity`
  - `vision_0 / vision_1 / prompt_emb = cosine`
  - `robot_state = l2`
- `score_normalization`
  - cosine `[0,1]` mapping
  - `robot_state` `exp(-d/tau)`
  - percentile normalization parameters
- `backend_hints`
  - backend-specific parameters
  - Currently mainly `rrf_k`

Clear boundary:

- General search semantics go in top-level fields
- Backend private parameters remain in `backend_hints`
- Do not degrade `QuerySpec` into a large dictionary

No pursuit of an especially elegant type system here; first ensure:

- Configuration is expressible
- Strategy can pass through
- Backend can read

### D. `backends/in_memory_backend.py`

This is the most important rewrite point of this round.

Implementation approach:

- Do not add new backend files
- Directly rewrite the search logic in the existing `in_memory_backend.py` to an experiment-ready version

Features to implement:

1. Multi-field scoring
   - `vision_0 / vision_1 / prompt_emb` use cosine
   - `robot_state` uses L2

2. Two fusion methods
   - `weighted_rrf`
   - `weighted_score_sum`

3. Normalization required for `weighted_score_sum`
   - cosine first `(cos + 1) / 2`
   - `robot_state` first `exp(-d / tau)`
   - Then percentile normalization using `p5 / p95`

4. Filtering
   - `checkpoint_id`
   - `task_key`
   - `step_range`

Recommend adding private methods in `in_memory_backend.py`:

- `_filter_entries()`
- `_cosine_score()`
- `_l2_distance()`
- `_normalize_score()`
- `_search_weighted_rrf()`
- `_search_weighted_score_sum()`
- `_iter_active_fields()`

Key requirements:

- Backend only consumes already-built query vectors
- Backend only loads already-built entry vectors
- Backend does not touch stage1 / token layout
- Backend does not participate in hit/miss determination
- Fields with `weight == 0` are skipped entirely: no cosine / L2 / rank / sum computation

### E. `config.py`

This file only handles schema, validation, and factories; no algorithms.

What needs expanding:

1. `KeyBuilderConfig`
   - Support:
     - `cp1_mean_pool`
     - `cp1_spatial_pool_16`
     - `cp1_spatial_pool_64`
     - `cp1_max_pool`

2. `SearchStrategyConfig`
   - Support experiment-required fields:
     - `type`
     - `field_similarity`
     - `score_normalization`
     - `rrf_k`

3. `BackendConfig`
   - `in_memory` remains as a backend type
   - No additional complex backend registry

4. Validation logic
   - Module level allows `vision_2`
   - LIBERO experiment YAML for this round should have `vision_2` disabled
   - `prompt_emb` weight is `0` in Phase 1 / 1.5
   - Under `weighted_score_sum`, the following must be provided:
     - `tau`
     - `p5/p95`
   - `backend.in_memory.preload_path` must exist and match the current run's `key_builder.type` / `vector_dims`

5. Factory logic
   - `_build_key_builder()` adds four builder branches
   - `_build_search_strategy()` adds two experiment strategy branches

### F. YAML Plan

Still uses the YAML entry point required by the tutorial; no additional configuration system is introduced.

This round does not adopt "one YAML covering multiple experiments."

Requirements changed to:

- One run corresponds to one YAML
- One YAML is a single final runnable configuration
- No longer relies on external tables or scripts for secondary weight expansion

But 100+ YAMLs should not be hand-written; they should be script-generated.

Constraints:

- The generation script only produces final YAML files in batch
- Each generated YAML is a self-contained, complete, directly runnable experiment configuration
- When running experiments, these YAMLs are consumed directly without secondary parameter expansion at runtime

Per the current experiment plan:

- Phase 1 first generates `64 independent YAMLs` (8 combos x 8 weights)
- Phase 1.5 / Phase 2 YAMLs are generated after analyzing results (~45 + ~3)
- Total ~112 YAMLs, produced in batches

Recommended directory:

- `configs/cache_runs/phase1/`
- `configs/cache_runs/phase1_5/`
- `configs/cache_runs/phase2/`
- `exp/generate_cache_run_yamls.py`

Recommended naming:

- `phase1_run_001_a_rrf_w1.yaml`
- `phase1_run_002_a_rrf_w2.yaml`
- ...
- `phase1_run_064_c_sum_w8.yaml`
- `phase1_5_run_001_*.yaml`
- ...
- `phase2_run_003_*.yaml`

Each YAML should fully specify:

- `key_builder.type`
- `checkpoints.cp1.search_strategy.type`
- Field weights for the current run
- `backend.vector_dims` for the current run
- `score_normalization` for the current run
- `keys.vision_2.enabled` for the current run
- `backend.in_memory.preload_path` for the current run

Unified constraints for LIBERO experiment YAMLs this round:

- `keys.vision_2.enabled = false`
- The module supports `vision_2`, but this round's runs do not enable it
- `prompt_emb.weight = 0.0` in Phase 1 / 1.5

YAML generation script responsibilities:

- Read the combination definitions from the experiment plan
- Batch-generate `112` independent YAMLs by phase
- For each run, write:
  - `key_builder.type`
  - `search_strategy.type`
  - Field weights
  - `score_normalization`
  - `backend.in_memory.preload_path`
- File naming should be stable and traceable for reviewing experiment results

YAML generation script is not responsible for:

- Starting experiments
- Expanding parameters at runtime
- Building artifacts on the fly

### G. Offline Pre-Building `in_memory` Data

This round does not allow the `in_memory` backend to build the database on-the-fly from raw data and `key_builder` at experiment startup.

Reasons:

- 112 runs would repeat the same key extraction and reduction
- The backend should not depend on stage1 / token layout / dataset parsing details
- Different `key_builder.type` outputs have different dimensions and cannot share the same prebuilt data

Therefore responsibilities are split:

- Offline script
  - Reads raw data
  - Calls the corresponding `key_builder`
  - Generates a cache artifact that can be loaded directly
- `InMemoryBackend`
  - Only loads artifacts
  - Only performs search

The artifact must be bound to at least this information:

- Dataset identifier
- `checkpoint_id`
- `key_builder.type`
- Enabled fields
- `vector_dims`
- Entries themselves

For the current `data/libero_spatial`, the main path for artifact construction is:

- Directly read fields already present in HDF5
  - `vision_0`
  - `vision_1`
  - `vision_2`
  - `prompt_emb`
  - `robot_state`
- No additional offline stage1 inference needed
- No policy checkpoint needed
- No GPU needed

Only when switching to a different dataset where the raw data lacks these embedding fields would an "offline stage1 extraction" fallback path be considered; this is outside the current main plan.

Recommend adding an offline script, e.g.:

- `exp/build_in_memory_cache_artifact.py`

This script's responsibilities:

- Input:
  - Raw data path
  - `key_builder.type`
  - Output path
- Output:
  - An artifact that `InMemoryBackend` can load directly

For the current experiment, there will be at least 4 artifacts:

- `cp1_mean_pool`
- `cp1_spatial_pool_16`
- `cp1_spatial_pool_64`
- `cp1_max_pool`

Different weight experiments and different strategy experiments under the same builder should reuse the same artifact without rebuilding.

When generating YAMLs, the corresponding builder's artifact path should be written directly into each run:

- `cp1_mean_pool` -> corresponding mean pool artifact
- `cp1_spatial_pool_16` -> corresponding spatial-16 artifact
- `cp1_spatial_pool_64` -> corresponding spatial-64 artifact
- `cp1_max_pool` -> corresponding max-pool artifact

Before Phase 1 officially runs, an offline sanity check is needed for each builder:

- Sample three types of pairs:
  - same-episode near-step
  - same-task cross-episode
  - cross-task
- Compute the vision cosine distribution and separability for each builder
- Especially check whether `B1/B2`'s spatial-pool + flatten + cosine degenerates to near-noise

Handling principles:

- By default, do not automatically remove experiment groups based on sanity check results
- But if a builder's separability is close to random, or the distribution is clearly pathological, confirm with a human whether to stop that group before entering the 64 runs

### H. Experiment Control Script

A new local control script is needed, e.g.:

- `exp/run_cache_experiments.py`

Goal:

- Run 100+ experiments sequentially
- Avoid restarting the server for each YAML change
- Wrap the existing `examples/libero/main.py` as much as possible; do not rewrite the client main logic

This script is responsible for:

- Reading the list of YAMLs to run
- Supporting running only a subset of runs
- Supporting specifying the number of episodes per run for the current batch
- Notifying the server to switch to the corresponding YAML before each run
- Calling `examples/libero/main.py` or a lightweight wrapper to start the LIBERO evaluation
- Collecting exit status and result summary for each run
- Writing progress to a local state file; supporting checkpoint-resume
- Writing a structured result summary for each run for subsequent aggregation scripts

Recommended control parameters:

- `--runs`
  - Specify which runs to execute, e.g., by filename, index range, or phase filter
- `--episodes-per-run`
  - Override `num_trials_per_task` for each run
- `--num-workers`
  - Pass through to `examples/libero/main.py --num_workers`
  - Number of concurrent workers within a single run
- `--resume`
  - Resume from local state file, skipping completed runs
- `--state-path`
  - Specify the checkpoint state file path
- `--libero-args`
  - Pass through additional arguments to `examples/libero/main.py`

Result collection requirements:

- Each run uses a fixed `episodes-per-run` budget
- Phase 1's top 3 are ranked by `aggregate success rate` under a fixed budget
- For final conclusions, the best configuration should also have an additional larger-budget confirm run

Recommend adding a result aggregation script, e.g.:

- `exp/analyze_cache_results.py`

Responsibilities:

- Aggregate structured result files for each run
- Output ranked tables by phase
- Select Phase 1 top 3
- Generate subsequent input for Phase 1.5

Checkpoint-resume requirements:

- At minimum, support run-level checkpoint-resume
  - Completed YAMLs are not re-run
- Ideally support episode-level progress recording
  - After interruption, resume from the next episode in the current run

Recommended state file fields:

- run id / yaml path
- Number of episodes used
- Current status: pending / running / done / failed
- Number of episodes completed
- Start time / end time
- Result summary (success rate, exit code, log path)

Note: the "cache" here refers to experiment control-level progress caching, not the retrieval cache itself.

### I. Server Dynamic YAML Switching

The current `scripts/serve_policy.py` `--cache_config` is read once at server startup.

For 100+ experiments, this is inconvenient because every YAML change requires a server restart.

Therefore the server side needs a lightweight control capability:

- Add a "switch cache configuration" request in WebSocket control messages
- Sent by the local control script via a dedicated short-lived connection at the beginning of each experiment round
- The server validates the YAML, builds a new `shared_storage`, and atomically replaces the global `CurrentCacheBundle`

Overall constraints:

- **Support multiple workers concurrently evaluating within a single run under a fixed YAML**
- **Do not support multiple different runs sharing one server concurrently**
- Runs are serial; within a single run, `num_workers > 1` is allowed
- The experiment server uniformly starts with `--concurrent` (even if `num_workers=1`), because only concurrent mode has the `connection_policy_factory` entry point

Target boundary:

- The server only handles "serve new connections per bundle"
- Experiment control sequence is still the local script's responsibility
- Do not embed experiment plans or phase concepts into the server

Protocol form:

- Client sends control message:
  - `__ctrl__ = "load_cache_config"`
  - With `yaml_path`
- Server executes:
  - Validate YAML
  - `build_shared_storage(cache_config)` to construct new storage
  - Atomically replace the entire global `CurrentCacheBundle`
- Server returns:
  - `__ack__ = "load_cache_config"`
  - With `version` (bundle version number, monotonically increasing)

Effective timing:

- `load_cache_config` builds a new bundle and atomically replaces it; does not affect any currently active connections
- The new bundle takes effect on the next newly created evaluation connection
- Already-running old connections continue using the old bundle's `shared_storage`, unaffected
- No busy protection needed -- control messages can be sent at any time

Single run execution flow:

1. Control script sends `load_cache_config(yaml_path)`, waits for ack
2. Starts `examples/libero/main.py --num_workers N`
3. N workers each create connections; `connection_policy_factory` reads the current bundle snapshot
4. All workers share the same `bundle.shared_storage`
5. After all workers complete, switch to the next run

Implementation requirements:

- New logic goes in `websocket_policy_server.py` (`CurrentCacheBundle` + control message handling)
- `scripts/serve_policy.py`'s `connection_policy_factory` reads the latest configuration from the global bundle on each new connection
- Do not change the regular `infer` request protocol
- Do not require rewriting `openpi-client`'s main inference logic

Side effect notes:

- The control short-lived connection creates one empty task lifecycle (the server triggers `on_task_begin()` when a connection is established); this is acceptable

To maintain clear responsibilities:

- The server does not directly parse experiment tables
- The server is not responsible for selecting the next run
- The server is not responsible for experiment checkpoint-resume

### J. LIBERO Entry Point Reuse Approach

The preferred approach is not to rewrite `examples/libero/main.py`, but to wrap it as much as possible.

Reason:

- `examples/libero/main.py` already has:
  - Task suite selection
  - `num_trials_per_task`
  - Client lifecycle
  - Evaluation main loop
- The large-scale experiment mainly lacks "run switching control," not the individual LIBERO rollout logic

Recommended two-layer implementation:

1. Minimal modification to `examples/libero/main.py`
   - Expose a more reusable entry function
   - Support optional episode start offset / episode count override
2. Local control script wraps it
   - Notifies the server to switch YAML before each run
   - Then calls the LIBERO main entry to execute that run

If the client protocol main flow does not need to be changed, do not rewrite the client.

### K. Test Plan

Tests still follow the existing test structure under `tests/cache/`; no new test framework is invented.

Recommended new test files:

- `tests/cache/components/test_key_builder_cp1_experiment.py`
- `tests/cache/test_search_strategy_experiment.py`
- `tests/cache/test_search_strategy_weighted_rrf.py`
- `tests/cache/test_search_strategy_weighted_score_sum.py`
- `tests/cache/test_in_memory_backend_experiment.py`
- `tests/cache/test_config_experiment.py`

Test content:

1. key builder
   - Correct output shapes for all four builders
   - `vision_2` code path works correctly
   - Correct behavior when `vision_2` is disabled in LIBERO experiment YAML
   - `prompt_emb` output exists but can be configured with weight `0`
   - `collect(stage3=...)` caches `action_chunk`
   - `build(CheckpointID.CP3)` does not error and currently behaves identically to `CP1`

2. search strategy
   - `QuerySpec` correctly carries:
     - `fusion_method`
     - `field_similarity`
     - `score_normalization`
   - Assembly paths for both strategy types are correct

3. in_memory backend
   - cosine + L2 multi-field scoring is correct
   - `weighted_rrf` ranking is correct
   - `weighted_score_sum` ranking is correct
   - `tau = 0.334717` path is correct
   - Filtering logic is correct
   - Can correctly load artifact from `preload_path`
   - Refuses to start when `key_builder.type` / `vector_dims` mismatch

4. config
   - YAML can build the corresponding builder / strategy / backend
   - Missing `tau` or `p5/p95` produces an error under `weighted_score_sum`
   - Missing `backend.in_memory.preload_path` produces an error under experiment configuration

5. experiment runner / server control
   - Control script can execute runs in the given list order (runs are serial)
   - `--episodes-per-run` correctly overrides the LIBERO episode count
   - `--num-workers` passes through to `main.py`; supports multiple concurrent workers within a single run
   - `--resume` skips completed runs
   - Server uniformly starts with `--concurrent`
   - `load_cache_config` builds a new `CurrentCacheBundle` (containing shared_storage), atomically replaces the global bundle
   - `load_cache_config` does not affect the regular `infer` protocol
   - Multiple workers in the same run share the same `bundle.shared_storage`
   - When switching YAML between runs, shared_storage is rebuilt; old connections continue using the old bundle
   - Control short-lived connections create one empty task lifecycle (on_task_begin/on_task_end); tests must verify this does not produce harmful side effects

6. calibration / analysis
   - Can compute real `p5/p95` for each builder / field
   - Official `SUM` YAMLs must not use placeholder `p5=0.0, p95=1.0`
   - Can output builder sanity check results
   - Can aggregate each run's success rate and select Phase 1 top 3

---

## Implementation Breakdown by Experiment Group

To avoid "one key builder handling all functionality," this round splits builders by experiment group, but still places them in the tutorial's existing `components/key_builder.py` extension point:

Note:

- All four builders retain `vision_2` support
- Whether `vision_2` is enabled is determined by the specific YAML
- Code support scope is larger than what this round's LIBERO experiment uses

- Group A:
  - `CP1MeanPoolKeyBuilder`
- Group B1:
  - `CP1SpatialPool16KeyBuilder`
- Group B2:
  - `CP1SpatialPool64KeyBuilder`
- Group C:
  - `CP1MaxPoolKeyBuilder`

Layer 2 is no longer split into multiple backends by experiment group; instead, a single `InMemoryBackend` supports both fusion methods:

- `weighted_rrf`
- `weighted_score_sum`

This makes boundaries clearest:

- Experiment group differences mainly fall on `key_builder.type`
- Fusion differences mainly fall on `search_strategy.type`
- The backend uniformly handles execution

## Configuration Capabilities to Add or Modify

### `key_builder`

Must support:

- `type: cp1_mean_pool`
- `type: cp1_spatial_pool_16`
- `type: cp1_spatial_pool_64`
- `type: cp1_max_pool`

### `search_strategy`

Must support:

- `type: weighted_rrf_knn`
- `type: weighted_score_sum_knn`
- `fusion_method` (SearchStrategy drops `_knn` suffix when writing to QuerySpec)
  - `weighted_rrf`
  - `weighted_score_sum`
- `field_similarity`
  - `vision_0: cosine`
  - `vision_1: cosine`
  - `prompt_emb: cosine`
  - `robot_state: l2`
- `score_normalization`
  - cosine mapped to `[0,1]`
  - `robot_state` distance to similarity
  - percentile normalization
- `backend_hints`
  - `rrf_k`

### `backend`

Must support:

- `type: in_memory`
- `in_memory` specific config block
- `in_memory.preload_path`
- `index_type` can be extended later, but the first version uses brute-force

---

## Target YAML Format

Below is the target YAML format for a "single run." For `112 runs`, there should be `112` such independently runnable YAMLs.

```yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null

keys:
  vision_0:    { enabled: true,  weight: 0.75 }
  vision_1:    { enabled: true,  weight: 0.25 }
  vision_2:    { enabled: false, weight: 0.0 }
  prompt_emb:  { enabled: true,  weight: 0.0 }
  robot_state: { enabled: true,  weight: 0.25 }

key_builder:
  type: cp1_mean_pool

checkpoints:
  cp1:
    enabled: true
    gate:
      type: always_search
    judge:
      type: always_hit
    search_strategy:
      type: weighted_score_sum_knn
      top_k: 1
      step_filter: all
      field_similarity:
        vision_0:   { type: cosine }
        vision_1:   { type: cosine }
        prompt_emb: { type: cosine }
        robot_state:
          type: l2
          to_similarity:
            type: exp
            tau: 0.334717
      score_normalization:
        type: percentile
        fields:
          vision_0:   { p5: <calibrated>, p95: <calibrated> }
          vision_1:   { p5: <calibrated>, p95: <calibrated> }
          prompt_emb: { p5: <calibrated>, p95: <calibrated> }
          robot_state:{ p5: <calibrated>, p95: <calibrated> }

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    vision_2: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    preload_path: data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl
    index_type: brute_force
```

Notes:

- In Phase 1 / 1.5, `prompt_emb.weight = 0.0`
- Each run has its own independent YAML
- The module level retains `vision_2`
- This round's LIBERO YAMLs have `vision_2.enabled = false`
- If Phase 2 needs to verify `prompt_emb`, only the weight changes; module boundaries do not change
- The same schema should express both `weighted_rrf` and `weighted_score_sum`
- Multiple runs under the same `key_builder.type` share the same `preload_path`
- Official `SUM` YAMLs must contain real calibration values; they must not retain placeholder `p5: 0.0, p95: 1.0`

---

## Artifact Pre-Building and Loading Boundary

This round of experiments completely separates "building" and "retrieval":

- `KeyBuilder`
  - Defines how keys are constructed from model outputs
- Offline artifact build script
  - Uses `KeyBuilder` to batch-convert raw data into entry vectors
- `InMemoryBackend`
  - Only loads artifacts
  - Only performs search
- YAML
  - Only declares which artifact the current run uses

Constraints of this approach:

- The backend no longer repeatedly builds data for different experiment runs
- The backend does not need to know the raw data directory structure
- Products from different `key_builder.type` must be saved in separate files
- A run can only load one artifact that is already aligned with the current `key_builder.type`

---

## `Weighted Score Sum` Implementation Requirements

Must be clear: this is not raw score sum.

Field rules are fixed:

- `vision_0 / vision_1 / prompt_emb`:
  - Layer 1 uses cosine
  - Then mapped to `[0,1]`: `(cos + 1) / 2`
- `robot_state`:
  - Layer 1 uses L2
  - Then converted to similarity: `exp(-d / tau)`
  - Currently fixed at `tau = 0.334717`

Then percentile normalization:

- `s_hat_f = clip((s_f - p5_f) / (p95_f - p5_f), 0, 1)`

Only then is the following allowed:

- `Score(x) = sum_f w_f * s_hat_f(x)`

Therefore the backend must support:

- Returning both per-field raw scores and final fused scores simultaneously
- Under `weighted_score_sum`, strictly normalizing before summing

---

## `Weighted RRF` Implementation Requirements

The backend must support:

- Independent ranking per field
- Weighted RRF using field weights
- `robot_state`'s L2 results first produce ranks in "smaller distance = more similar" order

RRF's responsibility depends only on rank, not directly on score scale.

---

## `in_memory` Backend Rewrite Target

### First Version Must Support

- Multi-field entry/query
- `checkpoint_id` filtering
- `task_key` filtering
- `step_range` filtering
- `vision/prompt = cosine`
- `robot_state = L2`
- `weighted_rrf`
- `weighted_score_sum`
- brute-force top-k

### First Version Can Defer

- Approximate indexing
- Multi-threaded parallel scoring
- Disk persistence
- Qdrant-equivalent payload serialization complexity

---

## Recommended Implementation Order

### Phase 0: Data Pre-Check and Calibration

New:

- `exp/build_in_memory_cache_artifact.py`
- `exp/calibrate_score_sum_stats.py`

Goal:

- Confirm `data/libero_spatial` directly provides `vision_0/1/2`, `prompt_emb`, `robot_state`
- Produce artifacts for 4 builders
- Perform offline sanity check for each builder
- Compute statistics:
  - `tau`
  - `p5/p95` for each builder / field
- Produce calibration results that can be directly written into official `SUM` YAMLs

Notes:

- Official `SUM` experiments must not use placeholder `p5=0.0, p95=1.0`
- On the current dataset, this phase does not require additional offline stage1 inference

### Phase A: Extend Types and Configuration

Modify:

- `storage_types.py`
- `config.py`
- `search_strategy.py`

Goal:

- `QuerySpec` can express experiment semantics
- YAML can express reducer / fusion / normalization
- Factory can correctly instantiate new components

### Phase B: Add CP1 Experiment Dimensionality Reducers

Modify:

- `components/key_builder.py`

Goal:

- Support A / B1 / B2 / C reduction types
- Output dimensions aligned with `backend.vector_dims`
- 4 external builder types share one set of internal helpers

### Phase C: Rewrite `in_memory` Backend

Modify:

- `backends/in_memory_backend.py`

Goal:

- `in_memory` becomes an experiment-ready backend
- All Layer 1 / Layer 2 logic falls into backend internal private functions
- Fields with `weight == 0` are skipped entirely

### Phase D: Batch Generate YAMLs

New:

- `exp/generate_cache_run_yamls.py`

Goal:

- Phase 1 first generates `64` independent YAMLs (8 combos x 8 weights)
- Phase 1.5 / Phase 2 YAMLs are generated by calling the same script again after analyzing results
- Each YAML includes correct calibration and `preload_path`

### Phase E: Experiment Control and Result Aggregation

New:

- `exp/run_cache_experiments.py`
- `exp/analyze_cache_results.py`

Goal:

- Support specifying run subsets
- Support specifying episode count per run
- Support run-level checkpoint-resume
- Aggregate results and select Phase 1 top 3
- Reuse `examples/libero/main.py` as much as possible

### Phase F: Server Dynamic YAML Switching

Overall constraints:

- **Support multiple workers concurrently evaluating within a single run under a fixed YAML**
- **Do not support multiple different runs sharing one server concurrently**
- Runs are serial; within a single run, `num_workers > 1` is allowed
- The experiment server uniformly starts with `--concurrent` (even if `num_workers=1`)

Design: The server maintains a global `CurrentCacheBundle` (containing `cache_config` + `shared_storage` + `version`).

Flow:

1. Control script opens a short-lived WebSocket connection
2. Sends `{"__ctrl__": "load_cache_config", "yaml_path": "<abs_path>"}`
3. Server validates YAML, builds new `shared_storage`, atomically replaces the entire bundle
4. Server returns `{"__ack__": "load_cache_config", "version": N}`
5. Control connection disconnects
6. Starts `examples/libero/main.py --num_workers N`; N workers each create connections
7. `connection_policy_factory` reads the current bundle snapshot, calls `build_per_connection_components(bundle.cache_config, bundle.shared_storage)`
8. After all workers complete, switch to the next run

Storage sharing semantics:

- Multiple worker connections within the same run share the same `bundle.shared_storage`
- When switching YAML between runs, new `shared_storage` is built
- Already-running old connections continue using the old bundle, unaffected

Modify:

- `src/openpi/serving/websocket_policy_server.py`: add `CurrentCacheBundle` + `load_cache_config` control message
- `scripts/serve_policy.py`: `connection_policy_factory` reads the latest configuration from the global bundle

Do not modify:

- `packages/openpi-client/` -- control script uses `websockets` + `msgpack` directly
- `examples/libero/main.py` -- unchanged; each run naturally opens new connections

---

## Not Doing This Round

- `CP2`
- `CP3`
- Judge threshold experiments
- Gate strategy experiments
- Real Qdrant backend alignment
- Making in-memory a production-grade ANN engine

---

## Acceptance Criteria

Upon completion, the following should be satisfied:

1. A single YAML can fully express one experiment configuration
2. The `CP1` path can directly switch:
   - Reduction method
   - Field weights
   - Fusion method
3. The `in_memory` backend can actually run:
   - `weighted_rrf`
   - `weighted_score_sum`
4. Each module has clear responsibilities:
   - key builder only does reduction
   - strategy only assembles queries
   - backend only does retrieval
   - judge only passes through top-1
5. When later extending to Qdrant, the following can be reused:
   - key builder
   - config schema
   - search strategy
   - calibration configuration
6. Phase 1 top 3 selection rules are clear:
   - Fixed `episodes-per-run` budget
   - Ranked by `aggregate success rate`
   - Best configuration can have an additional larger-budget confirm run

---

## Next Steps

Next steps proceed in the following order:

1. First complete `Phase 0`: produce artifacts, sanity checks, `tau`, `p5/p95`
2. Then extend `QuerySpec` and YAML schema
3. Then add the 4 thin wrapper key builders
4. Then rewrite the `in_memory` backend
5. Then generate official experiment YAMLs and result aggregation scripts

---

## Code-Level Detailed Specifications

Below, in Phase order, are the precise modification specifications for each file: function signatures, data structures, algorithm pseudocode, test cases.

---

### Contract Revisions

This round of experiments makes the following revisions to existing system contracts; these need to be synchronized in code docstrings:

#### 1. `CacheEntry.query_keys` No Longer Requires All Fields to be L2-Normalized

The existing `storage_types.py:119` comment says "CPU float32, L2-normalised."

Revised to:

> Tensors in `query_keys` must be CPU float32 contiguous.
> Whether to L2 normalize depends on the field's similarity computation method:
>   - cosine fields (vision_0/1/2, prompt_emb): L2 normalization not required (`F.cosine_similarity` handles it internally)
>   - L2 distance fields (robot_state): must retain raw vector, no L2 normalization (otherwise the physical meaning of L2 distance is destroyed)

Reason: `robot_state` requires the backend to compute real L2 distance and then apply `exp(-d/tau)` conversion. If the builder pre-normalizes, the distance semantics change.

Actual impact is minimal: existing code (`CacheStorage.insert`, `InMemoryBackend.search`) never checks L2 norm at runtime; only the docstring stated this constraint.

#### 2. `robot_state` Explicitly Retained as Raw Vector

All CP1 builders only apply `_to_cpu_float32()` to `robot_state`; no L2 normalize, no mean pool.

This is not an "exception" but a design requirement: robot_state's L2 distance is a core experimental signal; normalization would lose distance information.

#### 3. Artifact Entry IDs May Deviate from Stable-Hash Semantics

The existing `storage_types.py:109` describes id's stable-hash semantics (`sha256(checkpoint_id + query_key_bytes)`).

Revision:
- Online write path (Orchestrator.write) still uses stable-hash
- Offline artifact entry IDs use `"{episode_file_stem}_{step_name}"`, deterministic and traceable
- Constraint: **In this round of experiments, the artifact-loaded backend does not mix with online writes** (a single InMemoryBackend instance either only loads from artifacts or only accepts online inserts, not both)

Reason: During offline construction, byte-level determinism of query_keys is not guaranteed (floating point precision); stable-hash is not guaranteed to be stable. File name + step name is more traceable.

#### 4. `CachePayload.action_chunk` Remains `[horizon, 32]` but Allows horizon < 50

The existing `storage_types.py:87` comment says `[50, 32]`.

Revision:
- In offline artifacts, action_chunk retains the data's actual horizon (e.g., `[10, 32]`)
- No zero-padding to `[50, 32]`, because the LIBERO client only uses the first few action steps; extra padding is meaningless
- The artifact build script's comments explain that action_chunk shape depends on the dataset

If it is later discovered that the Interceptor or client has a hard dependency on `[50, 32]`, padding logic will be added to the artifact script. Not added preemptively.

#### 5. vision_2 Support Scope Definition

Explicitly defined as **supported at the builder level; not produced in artifact/backend/YAML currently**:

- `_CP1BaseKeyBuilder.build()` has a code path for vision_2 (if `enabled_fields` includes vision_2)
- `_build_fake_stage1` uses zero-fill for the vision_2 position to maintain correct prefix_embs token layout (vision_1's offset is at 256-512; if the 512-768 region for vision_2 is not filled, the prompt start position would be wrong)
- This round's LIBERO experiment YAMLs uniformly set `keys.vision_2.enabled: false`
- `vector_dims` and artifacts do not include vision_2
- Builder tests cover the vision_2 enabled code path, but experiments do not use it
- To enable vision_2 in the future, only YAML + artifact rebuild + vector_dims expansion are needed; no builder code changes required

#### 6. Naming Conventions

Naming unified across two layers:

| Layer | Field | Value | Notes |
|---|---|---|---|
| config / YAML | `search_strategy.type` | `weighted_rrf_knn` / `weighted_score_sum_knn` | With `_knn` suffix, consistent with existing `qdrant_weighted_rrf_knn` style |
| QuerySpec / backend | `fusion_method` | `weighted_rrf` / `weighted_score_sum` | Without `_knn`; backend does not care whether retrieval is KNN or ANN |

SearchStrategy is responsible for dropping the `_knn` suffix when constructing QuerySpec.

---

### Phase A: Extend Types and Configuration

#### A1. `src/openpi/cache/storage_types.py` -- QuerySpec Extension

Add 3 new top-level fields to the existing `QuerySpec` dataclass:

```python
@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None
    fusion_weights: Optional[dict[str, float]] = None
    backend_hints: Optional[dict[str, Any]] = None

    # --- New fields ---
    fusion_method: Optional[str] = None
    # Values: "weighted_rrf" | "weighted_score_sum" | None
    # When None, backend falls back to existing single-field cosine behavior (backward compatible)

    field_similarity: Optional[dict[str, dict[str, Any]]] = None
    # Per-field similarity definition, e.g.:
    # {
    #   "vision_0":    {"type": "cosine"},
    #   "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
    # }

    score_normalization: Optional[dict[str, Any]] = None
    # Only needed for weighted_score_sum, e.g.:
    # {
    #   "type": "percentile",
    #   "fields": {
    #     "vision_0":    {"p5": 0.82, "p95": 0.99},
    #     "robot_state": {"p5": 0.15, "p95": 0.88},
    #   }
    # }
```

No modifications to `CacheEntry`, `CachePayload`, `SearchResultLite`, `SearchResult`.

No modifications to `QueryFilter` (`task_key` and `step_range` already exist).

#### A2. `src/openpi/cache/config.py` -- Configuration Schema Extension

##### A2.1 `KeyBuilderConfig` Extension

No additional fields needed; only extend the type enum:

```python
@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"
    # New valid values: "cp1_mean_pool", "cp1_spatial_pool_16",
    #                   "cp1_spatial_pool_64", "cp1_max_pool"
```

##### A2.2 `SearchStrategyConfig` Extension

```python
@dataclass
class FieldSimilarityConfig:
    type: str = "cosine"           # "cosine" | "l2"
    to_similarity: Optional[dict[str, Any]] = None
    # Only used for l2, e.g.: {"type": "exp", "tau": 0.334717}

@dataclass
class ScoreNormalizationConfig:
    type: str = "none"             # "none" | "percentile"
    fields: Optional[dict[str, dict[str, float]]] = None
    # Only used for percentile, e.g.: {"vision_0": {"p5": 0.82, "p95": 0.99}}

@dataclass
class SearchStrategyConfig:
    type: str = "qdrant_weighted_rrf_knn"
    # New valid values: "weighted_rrf_knn", "weighted_score_sum_knn"
    top_k: int = 1
    step_filter: str = "all"
    step_window: int = 5
    rrf_k: int = 60
    candidate_multiplier: int = 5

    # --- New fields ---
    field_similarity: Optional[dict[str, FieldSimilarityConfig]] = None
    # Keys are field names (vision_0, robot_state, ...)
    # Values are FieldSimilarityConfig
    # Only used by weighted_rrf_knn / weighted_score_sum_knn

    score_normalization: Optional[ScoreNormalizationConfig] = None
    # Only used by weighted_score_sum_knn
```

##### A2.3 `BackendConfig` Extension

```python
@dataclass
class InMemoryConfig:
    preload_path: Optional[str] = None    # artifact .pkl path
    index_type: str = "brute_force"       # currently only brute_force

@dataclass
class BackendConfig:
    type: str = "qdrant"
    vector_dims: dict[str, int] = field(default_factory=lambda: {"robot_state": 32})
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    in_memory: InMemoryConfig = field(default_factory=InMemoryConfig)
    # New in_memory config block
```

##### A2.4 `_dict_to_dataclass` Extension

New nested dataclasses need handling:
- `field_similarity` dict values -> `FieldSimilarityConfig`
- `score_normalization` -> `ScoreNormalizationConfig`
- `in_memory` -> `InMemoryConfig`

Add to the `_CONFIG_TYPES` registry:
```python
_CONFIG_TYPES: dict[str, type] = {
    ...  # Keep existing
    "FieldSimilarityConfig": FieldSimilarityConfig,
    "ScoreNormalizationConfig": ScoreNormalizationConfig,
    "InMemoryConfig": InMemoryConfig,
}
```

For `field_similarity` which is `dict[str, FieldSimilarityConfig]`, add special handling in `_dict_to_dataclass` (similar to `checkpoints`):

```python
elif key == "field_similarity" and isinstance(value, dict):
    result = {}
    for field_name, field_data in value.items():
        if isinstance(field_data, dict):
            result[field_name] = _dict_to_dataclass(FieldSimilarityConfig, field_data)
        else:
            result[field_name] = field_data
    kwargs[key] = result
```

##### A2.5 `validate_cache_config` Extension

New validation rules:

```python
# Existing rules remain unchanged

# New rule 8: key_builder.type valid values extended
_VALID_KEY_BUILDER_TYPES = frozenset({
    "placeholder", "full_original",
    "cp1_mean_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_64", "cp1_max_pool",
})
if config.key_builder.type not in _VALID_KEY_BUILDER_TYPES:
    errors.append(...)

# New rule 9: search_strategy.type valid values extended
_VALID_STRATEGY_TYPES = frozenset({
    "qdrant_weighted_rrf_knn", "weighted_rrf_knn", "weighted_score_sum_knn",
})

# New rule 10: weighted_score_sum_knn requires score_normalization
for cp_name, cp_config in config.checkpoints.items():
    ss = cp_config.search_strategy
    if ss.type == "weighted_score_sum_knn":
        if ss.score_normalization is None or ss.score_normalization.type == "none":
            errors.append(f"{cp_name}.search_strategy: weighted_score_sum_knn requires score_normalization")
        if ss.score_normalization and ss.score_normalization.type == "percentile":
            if not ss.score_normalization.fields:
                errors.append(f"{cp_name}.search_strategy: percentile normalization requires fields")

# New rule 11: weighted_rrf_knn / weighted_score_sum_knn require backend.type == "in_memory"
for cp_name, cp_config in config.checkpoints.items():
    ss = cp_config.search_strategy
    if ss.type in ("weighted_rrf_knn", "weighted_score_sum_knn"):
        if config.backend.type != "in_memory":
            errors.append(f"{cp_name}: {ss.type} requires backend.type='in_memory'")

# New rule 12: cp1_* key_builder requires certain enabled fields
_CP1_BUILDER_REQUIRED_FIELDS = frozenset({"vision_0", "robot_state"})
if config.key_builder.type.startswith("cp1_"):
    for f in _CP1_BUILDER_REQUIRED_FIELDS:
        if f not in enabled_fields:
            errors.append(f"key_builder.type={config.key_builder.type} requires {f} enabled")

# New rule 13: in_memory backend + cp1_* builder requires preload_path
if config.backend.type == "in_memory" and config.key_builder.type.startswith("cp1_"):
    if not config.backend.in_memory.preload_path:
        errors.append("in_memory backend + cp1 builder requires backend.in_memory.preload_path")
```

##### A2.6 Factory Function Extensions

`_build_key_builder` adds new branches:

```python
def _build_key_builder(cfg: KeyBuilderConfig, enabled_fields: list[str], vector_dims: dict[str, int]):
    if cfg.type == "placeholder":
        ...  # unchanged
    elif cfg.type == "full_original":
        ...  # unchanged
    elif cfg.type == "cp1_mean_pool":
        from openpi.cache.components.key_builder import CP1MeanPoolKeyBuilder
        return CP1MeanPoolKeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_spatial_pool_16":
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
        return CP1SpatialPool16KeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_spatial_pool_64":
        from openpi.cache.components.key_builder import CP1SpatialPool64KeyBuilder
        return CP1SpatialPool64KeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_max_pool":
        from openpi.cache.components.key_builder import CP1MaxPoolKeyBuilder
        return CP1MaxPoolKeyBuilder(enabled_fields=enabled_fields)
    else:
        raise ConfigValidationError(...)
```

`_build_search_strategy` adds new branches:

```python
def _build_search_strategy(cfg: SearchStrategyConfig, storage, fusion_weights: dict[str, float]):
    if cfg.type == "qdrant_weighted_rrf_knn":
        ...  # unchanged
    elif cfg.type == "weighted_rrf_knn":
        from openpi.cache.components.search_strategy import WeightedRrfKnnStrategy
        return WeightedRrfKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            fusion_weights=fusion_weights if fusion_weights else None,
            rrf_k=cfg.rrf_k,
            field_similarity=_field_similarity_to_dict(cfg.field_similarity),
        )
    elif cfg.type == "weighted_score_sum_knn":
        from openpi.cache.components.search_strategy import WeightedScoreSumKnnStrategy
        return WeightedScoreSumKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            fusion_weights=fusion_weights if fusion_weights else None,
            field_similarity=_field_similarity_to_dict(cfg.field_similarity),
            score_normalization=_score_norm_to_dict(cfg.score_normalization),
        )
    else:
        raise ConfigValidationError(...)
```

Add two new private helpers to convert config dataclasses to plain dicts (for QuerySpec):

```python
def _field_similarity_to_dict(
    cfg: Optional[dict[str, FieldSimilarityConfig]],
) -> Optional[dict[str, dict[str, Any]]]:
    """FieldSimilarityConfig dict -> plain dict for QuerySpec.field_similarity."""
    if cfg is None:
        return None
    result = {}
    for name, fs in cfg.items():
        d: dict[str, Any] = {"type": fs.type}
        if fs.to_similarity is not None:
            d["to_similarity"] = fs.to_similarity
        result[name] = d
    return result

def _score_norm_to_dict(
    cfg: Optional[ScoreNormalizationConfig],
) -> Optional[dict[str, Any]]:
    """ScoreNormalizationConfig -> plain dict for QuerySpec.score_normalization."""
    if cfg is None:
        return None
    d: dict[str, Any] = {"type": cfg.type}
    if cfg.fields is not None:
        d["fields"] = cfg.fields
    return d
```

`_build_backend` extends the in_memory branch:

```python
if cfg.type == "in_memory":
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    backend = InMemoryBackend(vector_dims=cfg.vector_dims)
    if cfg.in_memory.preload_path:
        backend.load_artifact(cfg.in_memory.preload_path)
    return backend
```

#### A3. `src/openpi/cache/components/search_strategy.py` -- Add Two New Strategies

##### A3.1 `WeightedRrfKnnStrategy`

```python
class WeightedRrfKnnStrategy:
    """In-memory weighted RRF search strategy.

    When constructing QuerySpec, writes:
      fusion_method = "weighted_rrf"
      field_similarity = from configuration
      backend_hints = {"rrf_k": self._rrf_k}
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._rrf_k = rrf_k
        self._field_similarity = field_similarity

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        filters = self._build_filters(ctx)  # Reuses same logic as Qdrant strategy
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_rrf",
            field_similarity=self._field_similarity,
            backend_hints={"rrf_k": self._rrf_k},
        )
        return self._storage.search(spec)

    def _build_filters(self, ctx: SearchContext) -> Optional[QueryFilter]:
        # Identical logic to QdrantWeightedRrfKnnStrategy._build_filters
        # Reuse approach: extract as module-level private function _build_step_filters()
        ...
```

##### A3.2 `WeightedScoreSumKnnStrategy`

```python
class WeightedScoreSumKnnStrategy:
    """In-memory weighted score sum search strategy.

    When constructing QuerySpec, writes:
      fusion_method = "weighted_score_sum"
      field_similarity = from configuration
      score_normalization = from configuration
    """

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
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._field_similarity = field_similarity
        self._score_normalization = score_normalization

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
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
        )
        return self._storage.search(spec)
```

##### A3.3 Extract Shared Filter Construction Function

```python
def _build_step_filters(
    step_filter: str,
    step_window: int,
    ctx: SearchContext,
) -> Optional[QueryFilter]:
    """Shared step filter construction logic, used by all three strategy classes."""
    task_filter = QueryFilter(task_key=ctx.task_key) if ctx.task_key else None

    if step_filter == "all":
        return task_filter
    elif step_filter == "exact":
        f = QueryFilter(step_range=(ctx.current_step, ctx.current_step))
        if ctx.task_key:
            f.task_key = ctx.task_key
        return f
    elif step_filter == "window":
        lo = max(0, ctx.current_step - step_window)
        hi = ctx.current_step + step_window
        f = QueryFilter(step_range=(lo, hi))
        if ctx.task_key:
            f.task_key = ctx.task_key
        return f
    else:
        raise ValueError(f"Unknown step_filter: {step_filter}")
```

Also change `QdrantWeightedRrfKnnStrategy._build_filters` to call this shared function.

---

### Phase B: Add CP1 Experiment Dimensionality Reducers

#### B1. `src/openpi/cache/components/key_builder.py` -- New Content

##### B1.1 Shared Private Helper Functions

```python
# ---------------------------------------------------------------------------
# CP1 experiment key builder helpers (private)
# ---------------------------------------------------------------------------

def _slice_cp1_fields(
    prefix_embs: torch.Tensor,
    state: torch.Tensor,
    enabled: set[str] | None,
) -> dict[str, torch.Tensor]:
    """Slice out per-modality raw token sequences from prefix_embs[0] and state[0].

    Returns dict with field names as keys, GPU tensors as values (not reduced, not on CPU).
    vision_*: [256, emb_dim]
    prompt_emb: [num_prompt_tokens, emb_dim]
    robot_state: [state_dim]
    """
    result: dict[str, torch.Tensor] = {}
    prefix = prefix_embs[0]  # [prefix_len, emb_dim], drop batch dim

    for field_name, start, end in _VISION_OFFSETS:
        if enabled is not None and field_name not in enabled:
            continue
        result[field_name] = prefix[start:end]  # [256, emb_dim]

    if enabled is None or PROMPT_EMB in enabled:
        result[PROMPT_EMB] = prefix[_PROMPT_START:]  # [num_prompt_tokens, emb_dim]

    if enabled is None or ROBOT_STATE in enabled:
        result[ROBOT_STATE] = state[0]  # [state_dim]

    return result


def _mean_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """[num_tokens, emb_dim] -> [emb_dim], mean over token dimension."""
    return tokens.mean(dim=0)


def _max_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """[num_tokens, emb_dim] -> [emb_dim], per-dimension max over token dimension."""
    return tokens.max(dim=0).values


def _spatial_pool_tokens(
    tokens: torch.Tensor,
    grid_size: int,
    pool_size: int,
) -> torch.Tensor:
    """Apply 2D adaptive average pooling to vision tokens.

    Args:
        tokens: [grid_size*grid_size, emb_dim], SigLIP flattened patch tokens
        grid_size: original grid side length (16, since 256 = 16x16)
        pool_size: grid side length after pooling (4 for B1, 2 for B2)

    Returns:
        [pool_size*pool_size, emb_dim] then flatten to [pool_size*pool_size * emb_dim]

    Implementation:
        1. reshape [grid_size*grid_size, emb_dim] -> [1, emb_dim, grid_size, grid_size]
           (treat emb_dim as channel, spatial dimensions as HxW)
        2. F.adaptive_avg_pool2d(..., (pool_size, pool_size))
        3. reshape -> [pool_size*pool_size * emb_dim]
    """
    emb_dim = tokens.shape[1]
    # [grid*grid, emb_dim] -> [1, emb_dim, grid, grid]
    x = tokens.reshape(grid_size, grid_size, emb_dim).permute(2, 0, 1).unsqueeze(0)
    # [1, emb_dim, pool, pool]
    pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
    # -> [pool*pool * emb_dim]
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def _to_cpu_float32(t: torch.Tensor) -> torch.Tensor:
    """GPU tensor -> CPU float32 contiguous, unified D2H exit point."""
    return t.cpu().float().contiguous()
```

##### B1.2 Shared Base Class `_CP1BaseKeyBuilder`

```python
class _CP1BaseKeyBuilder:
    """Shared base for CP1 experiment key builders.

    Subclasses only need to override _reduce_vision() and _reduce_prompt().
    """

    def __init__(self, enabled_fields: list[str] | None = None) -> None:
        self._cache: dict[str, torch.Tensor] = {}
        self._enabled = set(enabled_fields) if enabled_fields is not None else None

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._cache["state"] = s1.state               # [B, state_dim] GPU
            self._cache["prefix_embs"] = s1.prefix_embs   # [B, prefix_len, emb_dim] GPU
        if "stage3" in stage_outputs:
            self._cache["action_chunk"] = stage_outputs["stage3"].action_chunk

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

        raw = _slice_cp1_fields(
            self._cache["prefix_embs"],
            self._cache["state"],
            self._enabled,
        )
        keys: dict[str, torch.Tensor] = {}

        # Vision fields: call subclass _reduce_vision
        for field_name in (VISION_0, VISION_1, VISION_2):
            if field_name in raw:
                keys[field_name] = _to_cpu_float32(self._reduce_vision(raw[field_name]))

        # Prompt field: call subclass _reduce_prompt
        if PROMPT_EMB in raw:
            keys[PROMPT_EMB] = _to_cpu_float32(self._reduce_prompt(raw[PROMPT_EMB]))

        # Robot state: output as-is (no reduction, no L2 normalization)
        if ROBOT_STATE in raw:
            keys[ROBOT_STATE] = _to_cpu_float32(raw[ROBOT_STATE])

        return keys

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        """Subclass override: [256, emb_dim] -> [reduced_dim]"""
        raise NotImplementedError

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        """Subclass override: [num_tokens, emb_dim] -> [reduced_dim]"""
        raise NotImplementedError

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()
```

Note: `robot_state` is not L2 normalized. The reason is that in RRF mode the backend needs the original L2 distance, and in SUM mode it computes L2 first then applies `exp(-d/tau)`. If the builder pre-normalizes, the physical meaning of L2 distance changes.

##### B1.3 Four Thin Wrapper Classes

```python
class CP1MeanPoolKeyBuilder(_CP1BaseKeyBuilder):
    """Group A: Mean Pool. Mean pooling for vision/prompt.
    Output vision_*: [emb_dim=2048], prompt_emb: [emb_dim=2048].
    """
    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1SpatialPool16KeyBuilder(_CP1BaseKeyBuilder):
    """Group B1: Spatial Pool 4x4 (16x compression).
    Output vision_*: [16*2048=32768], prompt_emb: [emb_dim=2048] (mean pool).
    """
    _GRID_SIZE = 16   # sqrt(256)
    _POOL_SIZE = 4    # 16x16 -> 4x4

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1SpatialPool64KeyBuilder(_CP1BaseKeyBuilder):
    """Group B2: Spatial Pool 2x2 (64x compression).
    Output vision_*: [4*2048=8192], prompt_emb: [emb_dim=2048] (mean pool).
    """
    _GRID_SIZE = 16
    _POOL_SIZE = 2    # 16x16 -> 2x2

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1MaxPoolKeyBuilder(_CP1BaseKeyBuilder):
    """Group C: Max Pool. Per-dimension max for vision/prompt.
    Output vision_*: [emb_dim=2048], prompt_emb: [emb_dim=2048].
    """
    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)
```

##### B1.4 Output Dimension Quick Reference Table

| builder type | vision_0/1 | prompt_emb | robot_state |
|---|---|---|---|
| cp1_mean_pool | 2,048 | 2,048 | 32 |
| cp1_spatial_pool_16 | 32,768 | 2,048 | 32 |
| cp1_spatial_pool_64 | 8,192 | 2,048 | 32 |
| cp1_max_pool | 2,048 | 2,048 | 32 |

The YAML's `backend.vector_dims` must match this table, guaranteed by the YAML generation script.

---

### Phase C: Rewrite in_memory Backend

#### C1. `src/openpi/cache/backends/in_memory_backend.py` -- Complete Rewrite

##### C1.1 Class Signature and Constructor

```python
class InMemoryBackend(VectorStoreBackend):
    """In-memory backend supporting multi-field retrieval and two fusion methods.

    Features:
      - Multi-field entry/query (vision_0/1/2, prompt_emb, robot_state)
      - checkpoint_id / task_key / step_range filtering
      - Per-field similarity: cosine / L2
      - Fusion: weighted_rrf / weighted_score_sum
      - brute-force top-k
    """

    def __init__(self, vector_dims: dict[str, int]) -> None:
        self._dims = vector_dims
        self._entries: dict[str, CacheEntry] = {}
        self.search_call_count: int = 0
        self.fetch_payload_call_count: int = 0
```

##### C1.2 `supported_filters` Extension

```python
def supported_filters(self) -> frozenset[str]:
    return frozenset({"checkpoint_id", "task_key", "step_range"})
```

##### C1.3 New `load_artifact` Method

```python
def load_artifact(self, path: str) -> None:
    """Load prebuilt entries from a pickle file.

    Artifact format (dict):
      {
        "key_builder_type": str,
        "checkpoint_id": str,
        "vector_dims": dict[str, int],
        "entries": list[CacheEntry],
      }

    Validation:
      - Artifact's vector_dims must match self._dims
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data["vector_dims"] != self._dims:
        raise ValueError(
            f"Artifact vector_dims mismatch: "
            f"artifact={data['vector_dims']}, backend={self._dims}"
        )
    for entry in data["entries"]:
        self._entries[entry.id] = entry
    logger.info("Loaded %d entries from %s", len(data["entries"]), path)
```

##### C1.4 `_filter_entries` Private Method

```python
def _filter_entries(self, spec: QuerySpec) -> list[CacheEntry]:
    """Filter by checkpoint_id / task_key / step_range."""
    results = []
    for entry in self._entries.values():
        if spec.checkpoint_id is not None and entry.checkpoint_id != spec.checkpoint_id:
            continue
        if spec.filters is not None:
            if spec.filters.task_key is not None:
                if entry.payload.task_key != spec.filters.task_key:
                    continue
            if spec.filters.step_range is not None:
                # step_idx is in entry metadata
                # If entry has no step_idx info, skip filtering (compatible with artifacts without step_idx)
                step_idx = getattr(entry, "step_idx", None)
                if step_idx is not None:
                    lo, hi = spec.filters.step_range
                    if not (lo <= step_idx <= hi):
                        continue
        results.append(entry)
    return results
```

Note: `CacheEntry` currently has no `step_idx` field. For step_range filtering, there are two approaches:
- Approach A (recommended): This round's experiments fix step_filter to "all"; step_range filtering is not used and CacheEntry does not need modification
- Approach B: Add an optional `metadata: dict[str, Any]` field to CacheEntry

This round selects Approach A: step_range filtering code is written but never actually triggered.

##### C1.5 Per-Field Scoring Private Methods

```python
def _cosine_score(self, q: torch.Tensor, e: torch.Tensor) -> float:
    """Compute cosine similarity, returns [-1, 1]."""
    return float(F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)))


def _l2_distance(self, q: torch.Tensor, e: torch.Tensor) -> float:
    """Compute L2 distance (Euclidean distance), returns >= 0."""
    return float(torch.norm(q.float() - e.float(), p=2))


def _compute_field_score(
    self,
    field_name: str,
    q: torch.Tensor,
    e: torch.Tensor,
    field_sim_config: dict[str, Any],
) -> float:
    """Compute single-field raw score based on field_similarity config.

    cosine fields: return cosine similarity [-1, 1]
    l2 fields: return negative L2 distance (larger = more similar, for RRF ranking)
               or return similarity = exp(-d/tau) (for score sum)
               specific behavior decided by caller; this only returns raw value

    Return convention:
      cosine: returns cosine similarity (larger = more similar)
      l2: returns L2 distance (smaller = more similar); caller handles direction conversion
    """
    sim_type = field_sim_config.get("type", "cosine")
    if sim_type == "cosine":
        return self._cosine_score(q, e)
    elif sim_type == "l2":
        return self._l2_distance(q, e)
    else:
        raise ValueError(f"Unknown similarity type: {sim_type}")
```

##### C1.6 `_iter_active_fields` Private Method

```python
def _iter_active_fields(
    self,
    spec: QuerySpec,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Return active fields for this search: [(field_name, weight, sim_config), ...]

    Active conditions:
      1. Field is in spec.query_keys
      2. Field is in self._dims
      3. weight > 0 (or all participate when fusion_weights is None)
    """
    result = []
    weights = spec.fusion_weights or {}
    sim_configs = spec.field_similarity or {}
    for field_name in spec.query_keys:
        if field_name not in self._dims:
            continue
        w = weights.get(field_name, 1.0)
        if w <= 0:
            continue
        sim_cfg = sim_configs.get(field_name, {"type": "cosine"})
        result.append((field_name, w, sim_cfg))
    return result
```

##### C1.7 `_search_weighted_rrf` Private Method

```python
def _search_weighted_rrf(
    self,
    candidates: list[CacheEntry],
    spec: QuerySpec,
    active_fields: list[tuple[str, float, dict[str, Any]]],
) -> list[SearchResultLite]:
    """Weighted RRF fusion.

    Algorithm:
      1. For each active field, compute raw scores for all candidates
      2. Sort by score to get independent rank per field (rank 1 = best)
         - cosine: larger score = smaller rank
         - l2: smaller distance = smaller rank
      3. Weighted RRF: score(x) = sum_f w_f / (rrf_k + rank_f(x))
      4. Sort by RRF score descending, take top_k
    """
    rrf_k = 60
    if spec.backend_hints:
        rrf_k = spec.backend_hints.get("rrf_k", 60)

    # Per-field ranking
    per_field_ranks: dict[str, dict[str, int]] = {}  # field -> {entry_id: rank}
    for field_name, weight, sim_cfg in active_fields:
        sim_type = sim_cfg.get("type", "cosine")
        scores: list[tuple[str, float]] = []
        for entry in candidates:
            if field_name not in entry.query_keys or field_name not in spec.query_keys:
                continue
            raw = self._compute_field_score(
                field_name,
                spec.query_keys[field_name],
                entry.query_keys[field_name],
                sim_cfg,
            )
            scores.append((entry.id, raw))

        # Sort: cosine larger = better -> descending; L2 smaller = better -> ascending
        if sim_type == "cosine":
            scores.sort(key=lambda x: x[1], reverse=True)
        else:  # l2
            scores.sort(key=lambda x: x[1], reverse=False)

        ranks = {eid: rank + 1 for rank, (eid, _) in enumerate(scores)}
        per_field_ranks[field_name] = ranks

    # RRF fusion
    rrf_scores: dict[str, float] = {}
    entry_map: dict[str, CacheEntry] = {e.id: e for e in candidates}
    for entry in candidates:
        total = 0.0
        for field_name, weight, _sim_cfg in active_fields:
            rank = per_field_ranks.get(field_name, {}).get(entry.id)
            if rank is not None:
                total += weight / (rrf_k + rank)
        rrf_scores[entry.id] = total

    # Sort and take top_k
    sorted_ids = sorted(rrf_scores, key=lambda eid: rrf_scores[eid], reverse=True)
    results = []
    for eid in sorted_ids[: spec.top_k]:
        entry = entry_map[eid]
        results.append(
            SearchResultLite(id=eid, score=rrf_scores[eid], checkpoint_id=entry.checkpoint_id)
        )
    return results
```

##### C1.8 `_search_weighted_score_sum` Private Method

```python
def _search_weighted_score_sum(
    self,
    candidates: list[CacheEntry],
    spec: QuerySpec,
    active_fields: list[tuple[str, float, dict[str, Any]]],
) -> list[SearchResultLite]:
    """Weighted Score Sum fusion.

    Algorithm:
      1. For each active field, compute raw scores for all candidates
      2. Convert raw scores to [0, 1] similarity:
         - cosine: s_01 = (cos + 1) / 2
         - l2: s = exp(-d / tau)
      3. percentile normalization: s_hat = clip((s - p5) / (p95 - p5), 0, 1)
      4. weighted sum: Score(x) = sum_f w_f * s_hat_f(x)
      5. Sort by Score descending, take top_k
    """
    norm_config = spec.score_normalization or {}
    norm_type = norm_config.get("type", "none")
    norm_fields = norm_config.get("fields", {})

    entry_map: dict[str, CacheEntry] = {e.id: e for e in candidates}
    final_scores: dict[str, float] = {e.id: 0.0 for e in candidates}

    for field_name, weight, sim_cfg in active_fields:
        sim_type = sim_cfg.get("type", "cosine")

        for entry in candidates:
            if field_name not in entry.query_keys or field_name not in spec.query_keys:
                continue

            raw = self._compute_field_score(
                field_name,
                spec.query_keys[field_name],
                entry.query_keys[field_name],
                sim_cfg,
            )

            # Step 2: Convert to [0, 1] similarity
            if sim_type == "cosine":
                s = (raw + 1.0) / 2.0
            elif sim_type == "l2":
                to_sim = sim_cfg.get("to_similarity", {})
                tau = to_sim.get("tau", 1.0)
                s = math.exp(-raw / tau)
            else:
                s = raw

            # Step 3: percentile normalization
            if norm_type == "percentile" and field_name in norm_fields:
                p5 = norm_fields[field_name]["p5"]
                p95 = norm_fields[field_name]["p95"]
                denom = p95 - p5
                if denom > 0:
                    s = max(0.0, min(1.0, (s - p5) / denom))
                else:
                    s = 0.5  # degenerate case when p5 == p95

            # Step 4: weighted accumulation
            final_scores[entry.id] += weight * s

    # Step 5: Sort and take top_k
    sorted_ids = sorted(final_scores, key=lambda eid: final_scores[eid], reverse=True)
    results = []
    for eid in sorted_ids[: spec.top_k]:
        entry = entry_map[eid]
        results.append(
            SearchResultLite(id=eid, score=final_scores[eid], checkpoint_id=entry.checkpoint_id)
        )
    return results
```

##### C1.9 Rewritten `search` Main Method

```python
def search(self, spec: QuerySpec) -> list[SearchResultLite]:
    self.search_call_count += 1
    if not self._entries:
        return []

    # 1. Filter
    candidates = self._filter_entries(spec)
    if not candidates:
        return []

    # 2. Determine active fields
    active_fields = self._iter_active_fields(spec)
    if not active_fields:
        return []

    # 3. Dispatch by fusion_method
    method = spec.fusion_method
    if method == "weighted_rrf":
        return self._search_weighted_rrf(candidates, spec, active_fields)
    elif method == "weighted_score_sum":
        return self._search_weighted_score_sum(candidates, spec, active_fields)
    elif method is None:
        # Backward compatible: no fusion_method uses legacy single-field cosine behavior
        return self._search_single_field_cosine(candidates, spec)
    else:
        raise ValueError(f"Unknown fusion_method: {method}")


def _search_single_field_cosine(
    self, candidates: list[CacheEntry], spec: QuerySpec
) -> list[SearchResultLite]:
    """Backward compatible: original single-field cosine search (existing tests depend on this)."""
    results: list[SearchResultLite] = []
    for entry in candidates:
        score = 0.0
        for field in spec.query_keys:
            if field in entry.query_keys:
                q = spec.query_keys[field].float()
                e = entry.query_keys[field].float()
                score = float(F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)))
                break
        results.append(
            SearchResultLite(id=entry.id, score=score, checkpoint_id=entry.checkpoint_id)
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[: spec.top_k]
```

New imports:
```python
import math
import logging
logger = logging.getLogger(__name__)
```

---

### Phase D: Batch Generate YAMLs

#### Generation Scope Alignment

The first version of this script only generates Phase 1's 64 YAMLs (8 combos x 8 weights).

Phase 1.5 and Phase 2 YAMLs cannot be generated before Phase 1 runs because:
- Phase 1.5's weight neighborhood depends on each combination's best weight from Phase 1
- Phase 2's best combination depends on Phase 1.5's results

Phase 1.5 YAMLs are generated by calling this script's `--phase 1.5` mode after `exp/analyze_cache_results.py` outputs the top 3. Phase 2 likewise.

Total YAML count:
- Phase 1: 64 (generated directly by this script)
- Phase 1.5: ~45 (generated after analysis)
- Phase 2: ~3 (generated after analysis)

#### D1. `exp/generate_cache_run_yamls.py`

```python
"""Batch generate CP1 experiment YAMLs.

Usage:
    # Phase 1: Generate 64 directly
    uv run exp/generate_cache_run_yamls.py \
        --phase 1 \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
        --output-dir configs/cache_runs

    # Phase 1.5: Generate based on Phase 1 analysis results
    uv run exp/generate_cache_run_yamls.py \
        --phase 1.5 \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
        --phase1-analysis configs/cache_runs/phase1/analysis.json \
        --output-dir configs/cache_runs

    # Phase 2: Based on Phase 1.5 results
    uv run exp/generate_cache_run_yamls.py \
        --phase 2 \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
        --phase1_5-analysis configs/cache_runs/phase1_5/analysis.json \
        --output-dir configs/cache_runs

Output:
    configs/cache_runs/phase1/phase1_run_001_a_rrf_w1.yaml  (64 files)
    configs/cache_runs/phase1_5/phase1_5_run_001_*.yaml     (~45, second pass)
    configs/cache_runs/phase2/phase2_run_001_*.yaml         (~3, second pass)
"""
```

##### D1.1 Core Data Structures

```python
@dataclass
class ExperimentCombo:
    builder_type: str          # "cp1_mean_pool", ...
    builder_abbrev: str        # "a", "b1", "b2", "c"
    strategy_type: str         # "weighted_rrf_knn" | "weighted_score_sum_knn"
    strategy_abbrev: str       # "rrf" | "sum"
    vector_dims: dict[str, int]

COMBOS = [
    ExperimentCombo("cp1_mean_pool",       "a",  "weighted_rrf_knn",       "rrf", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_mean_pool",       "a",  "weighted_score_sum_knn", "sum", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_16", "b1", "weighted_rrf_knn",       "rrf", {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_16", "b1", "weighted_score_sum_knn", "sum", {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_64", "b2", "weighted_rrf_knn",       "rrf", {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_64", "b2", "weighted_score_sum_knn", "sum", {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_max_pool",        "c",  "weighted_rrf_knn",       "rrf", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_max_pool",        "c",  "weighted_score_sum_knn", "sum", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
]

WEIGHT_GRID_PHASE1 = [
    {"vision_0": 1.0,  "vision_1": 0.0,  "robot_state": 0.0 },  # W1
    {"vision_0": 0.75, "vision_1": 0.25, "robot_state": 0.0 },  # W2
    {"vision_0": 0.75, "vision_1": 0.0,  "robot_state": 0.25},  # W3
    {"vision_0": 0.5,  "vision_1": 0.25, "robot_state": 0.25},  # W4
    {"vision_0": 0.5,  "vision_1": 0.5,  "robot_state": 0.0 },  # W5
    {"vision_0": 0.5,  "vision_1": 0.0,  "robot_state": 0.5 },  # W6
    {"vision_0": 0.25, "vision_1": 0.5,  "robot_state": 0.25},  # W7
    {"vision_0": 0.25, "vision_1": 0.25, "robot_state": 0.5 },  # W8
]
```

##### D1.2 YAML Rendering Logic

```python
def render_yaml(
    combo: ExperimentCombo,
    weights: dict[str, float],
    weight_id: str,
    artifact_dir: str,
    calibration: dict,  # builder_type -> field -> {"p5": float, "p95": float}
) -> str:
    """Render a complete YAML string for a single run."""
    # prompt_emb weight fixed at 0.0
    full_weights = {**weights, "prompt_emb": 0.0}

    # artifact path
    preload_path = f"{artifact_dir}/{combo.builder_type}.pkl"

    # field_similarity fixed
    field_similarity = {
        "vision_0":    {"type": "cosine"},
        "vision_1":    {"type": "cosine"},
        "prompt_emb":  {"type": "cosine"},
        "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
    }

    # score_normalization (SUM only)
    score_normalization = None
    if combo.strategy_type == "weighted_score_sum_knn":
        cal = calibration[combo.builder_type]
        score_normalization = {
            "type": "percentile",
            "fields": cal,  # {"vision_0": {"p5": ..., "p95": ...}, ...}
        }

    # Render using PyYAML or template string
    ...
```

##### D1.3 Main Function

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["1", "1.5", "2"])
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--output-dir", default="configs/cache_runs")
    # Phase 1.5/2 need previous phase analysis results
    parser.add_argument("--phase1-analysis", default=None, help="Phase 1 analysis JSON (for phase 1.5)")
    parser.add_argument("--phase1_5-analysis", default=None, help="Phase 1.5 analysis JSON (for phase 2)")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration_file).read_text())

    if args.phase == "1":
        _generate_phase1(args, calibration)
    elif args.phase == "1.5":
        if not args.phase1_analysis:
            parser.error("--phase1-analysis required for phase 1.5")
        analysis = json.loads(Path(args.phase1_analysis).read_text())
        _generate_phase1_5(args, calibration, analysis)
    elif args.phase == "2":
        if not args.phase1_5_analysis:
            parser.error("--phase1_5-analysis required for phase 2")
        analysis = json.loads(Path(args.phase1_5_analysis).read_text())
        _generate_phase2(args, calibration, analysis)


def _generate_phase1(args, calibration):
    """Generate Phase 1: 8 combos x 8 weights = 64 YAMLs."""
    run_idx = 0
    for combo in COMBOS:
        for w_idx, weights in enumerate(WEIGHT_GRID_PHASE1):
            run_idx += 1
            filename = f"phase1_run_{run_idx:03d}_{combo.builder_abbrev}_{combo.strategy_abbrev}_w{w_idx+1}.yaml"
            yaml_str = render_yaml(combo, weights, f"w{w_idx+1}", args.artifact_dir, calibration)
            out_path = Path(args.output_dir) / "phase1" / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_str)
    print(f"Phase 1: generated {run_idx} YAML files")


def _generate_phase1_5(args, calibration, phase1_analysis):
    """Generate Phase 1.5: top 3 combos x ~15 weight neighbors = ~45 YAMLs.

    Reads the best combos and their best weights from phase1_analysis["top3"],
    samples within +/-0.2 range at step=0.1 around the best weights.
    """
    top3 = phase1_analysis["top3"]  # list of {"combo": ..., "best_weights": ..., ...}
    run_idx = 0
    for entry in top3:
        combo = _find_combo(entry["combo"])
        center = entry["best_weights"]
        fine_weights = _generate_fine_grid(center, step=0.1, radius=0.2)
        for w_idx, weights in enumerate(fine_weights):
            run_idx += 1
            filename = f"phase1_5_run_{run_idx:03d}_{combo.builder_abbrev}_{combo.strategy_abbrev}_f{w_idx+1}.yaml"
            yaml_str = render_yaml(combo, weights, f"f{w_idx+1}", args.artifact_dir, calibration)
            out_path = Path(args.output_dir) / "phase1_5" / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_str)
    print(f"Phase 1.5: generated {run_idx} YAML files")


def _generate_phase2(args, calibration, phase1_5_analysis):
    """Generate Phase 2: best combo+weights with prompt_emb=0.1 control, ~3 YAMLs."""
    best = phase1_5_analysis["best"]
    run_idx = 0
    for prompt_w in [0.0, 0.1, 0.2]:
        run_idx += 1
        combo = _find_combo(best["combo"])
        weights = {**best["best_weights"], "prompt_emb": prompt_w}
        # Renormalize (make weights sum to 1)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        filename = f"phase2_run_{run_idx:03d}_prompt_{prompt_w:.1f}.yaml"
        yaml_str = render_yaml(combo, weights, f"p{prompt_w}", args.artifact_dir, calibration)
        out_path = Path(args.output_dir) / "phase2" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_str)
    print(f"Phase 2: generated {run_idx} YAML files")
```

---

### Phase 0: Data Pre-Check and Calibration

#### 0.1 `exp/build_in_memory_cache_artifact.py`

```python
"""Offline InMemoryBackend artifact builder.

Usage:
    uv run exp/build_in_memory_cache_artifact.py \
        --data-dir data/libero_spatial \
        --builder-type cp1_mean_pool \
        --output data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl

Input: HDF5 episode files (with vision_0/1/2, prompt_emb, robot_state fields)
Output: pickle file loadable by InMemoryBackend.load_artifact()
"""

import argparse
import pickle
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)
```

##### Core Flow

```python
def build_artifact(
    data_dir: str,
    builder_type: str,
    checkpoint_id_str: str = "CP1",
) -> dict:
    """Build the artifact dict.

    Flow:
      1. Scan all .h5 files under data_dir
      2. For each step, read raw tensors, simulate stage1 output format
      3. Call the corresponding KeyBuilder's collect() + build()
      4. Construct CacheEntry (with placeholder action_chunk)
      5. Collect all entries, attach metadata
    """
    from openpi.cache.types import CheckpointID
    from openpi.cache.storage_types import CacheEntry, CachePayload

    cp_id = CheckpointID[checkpoint_id_str]
    builder = _create_builder(builder_type)
    vector_dims = _get_vector_dims(builder_type)

    h5_paths = sorted(Path(data_dir).rglob("*.h5"))
    entries: list[CacheEntry] = []

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            task = f.attrs.get("task", "")
            success = bool(f.attrs.get("success", False))
            if not success:
                continue  # Only use successful episodes

            step_names = sorted(k for k in f.keys() if k.startswith("step_"))
            for step_name in step_names:
                group = f[step_name]

                # Read raw tensors and simulate stage1 output
                fake_stage1 = _build_fake_stage1(group)

                builder.collect(cp_id, stage1=fake_stage1)
                query_keys = builder.build(cp_id)
                builder.clear()

                # Construct CacheEntry
                # id uses filename+step name (deterministic, traceable), not stable-hash
                # See "Contract Revisions" item 3
                entry_id = f"{h5_path.stem}_{step_name}"
                action = torch.from_numpy(np.array(group["clean_action"])).float()
                # clean_action shape: [action_horizon, action_dim]
                # Retain data's actual horizon, do not pad to [50, 32]
                # See "Contract Revisions" item 4
                if action.dim() == 1:
                    action = action.unsqueeze(0)  # [1, action_dim]

                payload = CachePayload(
                    action_chunk=action,
                    task_key=str(task),
                )
                entry = CacheEntry(
                    id=entry_id,
                    checkpoint_id=cp_id,
                    query_keys=query_keys,
                    payload=payload,
                )
                entries.append(entry)

    logger.info("Built %d entries for %s", len(entries), builder_type)
    return {
        "key_builder_type": builder_type,
        "checkpoint_id": checkpoint_id_str,
        "vector_dims": vector_dims,
        "entries": entries,
    }
```

##### `_build_fake_stage1` Helper

```python
class _FakeStage1:
    """Simulates Stage1Output structure so KeyBuilder.collect() can work."""
    def __init__(self, prefix_embs: torch.Tensor, state: torch.Tensor):
        self.prefix_embs = prefix_embs  # [1, prefix_len, emb_dim]
        self.state = state               # [1, state_dim]

def _build_fake_stage1(group: h5py.Group) -> _FakeStage1:
    """Construct fake stage1 output from HDF5 step group.

    HDF5 fields:
      vision_0: [256, 2048]  (already SigLIP embedding)
      vision_1: [256, 2048]
      vision_2: [256, 2048]  (may not exist)
      prompt_emb: [num_tokens, 2048]
      robot_state: [32]

    Need to rebuild prefix_embs = concat([vision_0, vision_1, vision_2, prompt_emb], dim=0)
    then add batch dim.
    """
    parts = []
    for vfield in ("vision_0", "vision_1", "vision_2"):
        if vfield in group:
            parts.append(torch.from_numpy(np.array(group[vfield])).float())
        else:
            # vision_2 may not exist, zero-fill
            emb_dim = parts[0].shape[1] if parts else 2048
            parts.append(torch.zeros(256, emb_dim))

    prompt = torch.from_numpy(np.array(group["prompt_emb"])).float()
    parts.append(prompt)

    prefix_embs = torch.cat(parts, dim=0).unsqueeze(0)  # [1, prefix_len, emb_dim]
    state = torch.from_numpy(np.array(group["robot_state"])).float().unsqueeze(0)  # [1, state_dim]

    return _FakeStage1(prefix_embs, state)
```

##### `_create_builder` and `_get_vector_dims` Helpers

```python
def _create_builder(builder_type: str):
    from openpi.cache.components.key_builder import (
        CP1MeanPoolKeyBuilder,
        CP1SpatialPool16KeyBuilder,
        CP1SpatialPool64KeyBuilder,
        CP1MaxPoolKeyBuilder,
    )
    builders = {
        "cp1_mean_pool": CP1MeanPoolKeyBuilder,
        "cp1_spatial_pool_16": CP1SpatialPool16KeyBuilder,
        "cp1_spatial_pool_64": CP1SpatialPool64KeyBuilder,
        "cp1_max_pool": CP1MaxPoolKeyBuilder,
    }
    return builders[builder_type]()

_VECTOR_DIMS = {
    "cp1_mean_pool":       {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
    "cp1_spatial_pool_16": {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32},
    "cp1_spatial_pool_64": {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32},
    "cp1_max_pool":        {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
}

def _get_vector_dims(builder_type: str) -> dict[str, int]:
    return _VECTOR_DIMS[builder_type]
```

##### Main

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--builder-type", required=True,
                        choices=list(_VECTOR_DIMS.keys()))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = build_artifact(args.data_dir, args.builder_type)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved {len(artifact['entries'])} entries to {args.output}")

if __name__ == "__main__":
    main()
```

#### 0.2 `exp/calibrate_score_sum_stats.py`

```python
"""Compute p5/p95 statistics per builder/field and offline sanity check.

Usage:
    uv run exp/calibrate_score_sum_stats.py \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --output data/cache_artifacts/libero_spatial/calibration.json

Output JSON format:
    {
      "cp1_mean_pool": {
        "vision_0":    {"p5": 0.82, "p95": 0.99, "mean": 0.91, "std": 0.04},
        "vision_1":    {"p5": ..., "p95": ...},
        "prompt_emb":  {"p5": ..., "p95": ...},
        "robot_state": {"p5": ..., "p95": ...}
      },
      ...
    }
"""
```

##### Core Algorithm

```python
def compute_field_stats(
    entries: list[CacheEntry],
    field_name: str,
    sim_type: str,
    tau: float = 0.334717,
    num_pairs: int = 50000,
) -> dict[str, float]:
    """Sample pairs for a field and compute similarity distribution's p5/p95.

    Sampling strategy:
      - Randomly sample num_pairs pairs (i, j) where i != j
      - Compute similarity score (direction-unified):
        - cosine: (cos + 1) / 2
        - l2: exp(-d / tau)
      - Return p5, p95, mean, std
    """
    import random
    n = len(entries)
    scores = []
    for _ in range(num_pairs):
        i, j = random.sample(range(n), 2)
        qi = entries[i].query_keys[field_name]
        ej = entries[j].query_keys[field_name]

        if sim_type == "cosine":
            cos = float(F.cosine_similarity(qi.unsqueeze(0), ej.unsqueeze(0)))
            s = (cos + 1.0) / 2.0
        elif sim_type == "l2":
            d = float(torch.norm(qi.float() - ej.float(), p=2))
            s = math.exp(-d / tau)
        scores.append(s)

    scores_np = np.array(scores)
    return {
        "p5":   float(np.percentile(scores_np, 5)),
        "p95":  float(np.percentile(scores_np, 95)),
        "mean": float(np.mean(scores_np)),
        "std":  float(np.std(scores_np)),
    }
```

##### Sanity Check Output

For each builder, additionally output a sanity check:
```python
def sanity_check(
    entries: list[CacheEntry],
    field_name: str,
) -> dict[str, float]:
    """Compare same-task vs cross-task cosine distributions.
    If the two distributions overlap heavily (AUC < 0.55), mark as WARNING.
    """
    ...
```

Output to console and to the `_sanity` key in JSON.

---

### Phase E: Experiment Control and Result Aggregation

#### E1. `exp/run_cache_experiments.py`

```python
"""Experiment control script.

Usage:
    # Run all Phase 1 (4 worker concurrency)
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 10 \
        --num-workers 4 \
        --host localhost --port 8000

    # Run only specified runs
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --runs 1-8 \
        --episodes-per-run 10 \
        --num-workers 1 \
        --host localhost --port 8000

    # Checkpoint-resume
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 10 \
        --num-workers 4 \
        --host localhost --port 8000 \
        --resume
"""
```

##### State File Format

```python
@dataclass
class RunState:
    yaml_path: str
    run_id: str
    status: str              # "pending" | "running" | "done" | "failed"
    episodes_total: int
    episodes_done: int
    start_time: Optional[str]
    end_time: Optional[str]
    success_rate: Optional[float]
    exit_code: Optional[int]
```

State file: `configs/cache_runs/phase1/experiment_state.json`

##### Single Run Execution Flow

```python
def execute_run(
    yaml_path: str,
    episodes_per_run: int,
    num_workers: int,
    host: str,
    port: int,
) -> RunResult:
    """Execute a single run.

    Flow:
      1. load_cache_config(yaml) -- switch YAML via WebSocket, wait for ack
      2. Start examples/libero/main.py --num_workers N --host H --port P
      3. Wait for all workers to complete
      4. Collect results
      5. Switch to next run

    Note: examples/libero/main.py accepts --host and --port (not --server_url),
    see examples/libero/main.py:34-35. Control messages via WebSocket require
    assembling the URL manually.
    """
    # Step 1: Notify server to switch YAML
    server_url = f"ws://{host}:{port}"
    _send_cache_config(server_url, yaml_path)

    # Step 2: Call LIBERO (N workers each create connections, sharing the same bundle)
    cmd = [
        "uv", "run", "examples/libero/main.py",
        "--host", host,
        "--port", str(port),
        "--task_suite_name", "libero_spatial",
        "--num_trials_per_task", str(episodes_per_run),
        "--num_workers", str(num_workers),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    ...
```

##### `_send_cache_config` Helper

```python
import asyncio
import msgpack
import websockets

def _send_cache_config(server_url: str, yaml_path: str) -> None:
    """Send load_cache_config control message via WebSocket."""
    async def _send():
        async with websockets.connect(server_url) as ws:
            # Read metadata
            raw = await ws.recv()
            # Send control message
            msg = {"__ctrl__": "load_cache_config", "yaml_path": yaml_path}
            await ws.send(msgpack.packb(msg))
            # Wait for ack
            resp = msgpack.unpackb(await ws.recv())
            if resp.get("__ack__") != "load_cache_config":
                raise RuntimeError(f"Unexpected ack: {resp}")
    asyncio.run(_send())
```

Note: This control connection is separate from the actual LIBERO evaluation connections. The control connection only switches the YAML and disconnects immediately. LIBERO evaluation connections are established inside `examples/libero/main.py`.

#### E2. `exp/analyze_cache_results.py`

```python
"""Aggregate experiment results and select top 3.

Usage:
    uv run exp/analyze_cache_results.py \
        --state-file configs/cache_runs/phase1/experiment_state.json \
        --output configs/cache_runs/phase1/analysis.json
"""

def main():
    # Read state file
    # Sort by success_rate descending
    # Output ranked table (markdown format to stdout)
    # Output JSON (for Phase 1.5 generation script)
    ...
```

---

### Phase F: Server Dynamic YAML Switching

#### Overall Constraints

- **Support multiple workers concurrently evaluating within a single run under a fixed YAML**
- **Do not support multiple different runs sharing one server concurrently**
- Runs are serial; within a single run, `num_workers > 1` is allowed

#### Runtime Mode

The experiment server **uniformly starts with `--concurrent`**, even if `num_workers=1` it goes through the concurrent path. Reason: only in concurrent mode does the server call `connection_policy_factory` for each new connection; this is the only entry point for dynamic YAML switching.

#### Core Design: CurrentCacheBundle

The server maintains a global `CurrentCacheBundle`, atomically replaced when a `load_cache_config` control message arrives. Multiple worker connections within the same run share the same bundle's `shared_storage`.

```python
@dataclass
class CurrentCacheBundle:
    """Current run's cache configuration snapshot.

    Atomically replaced as a whole when load_cache_config control message arrives.
    Multiple worker connections within the same run share the same shared_storage.
    When switching YAML between runs, a new bundle (with new shared_storage) is built.
    Already-running old connections continue using the old bundle's shared_storage, unaffected.
    """
    config_path: str
    cache_config: CacheConfig           # validated config
    shared_storage: CacheStorage        # built from build_shared_storage()
    version: int                        # monotonically increasing, for log tracing
```

#### F1. `src/openpi/serving/websocket_policy_server.py` Modifications

Module-level new global state:

```python
import threading
from dataclasses import dataclass
from typing import Optional

@dataclass
class CurrentCacheBundle:
    config_path: str
    cache_config: object        # CacheConfig
    shared_storage: object      # CacheStorage
    version: int

_bundle_lock = threading.Lock()
_current_bundle: Optional[CurrentCacheBundle] = None
_bundle_version: int = 0
```

In the existing `__ctrl__` handling block, after the `episode_end` branch, add:

```python
elif ctrl == "load_cache_config":
    yaml_path = obs.get("yaml_path", "")
    if not yaml_path:
        await websocket.send(packer.pack({"__ack__": "error", "msg": "missing yaml_path"}))
        continue
    try:
        from openpi.cache.config import load_cache_config, build_shared_storage
        # Validate YAML and build new shared_storage
        cache_config = load_cache_config(yaml_path)
        shared_storage = build_shared_storage(cache_config)
        # Atomically replace the entire bundle
        with _bundle_lock:
            global _current_bundle, _bundle_version
            _bundle_version += 1
            _current_bundle = CurrentCacheBundle(
                config_path=yaml_path,
                cache_config=cache_config,
                shared_storage=shared_storage,
                version=_bundle_version,
            )
        logger.info("Cache bundle updated to v%d: %s", _bundle_version, yaml_path)
        await websocket.send(packer.pack({
            "__ack__": "load_cache_config",
            "yaml_path": yaml_path,
            "version": _bundle_version,
        }))
    except Exception as e:
        logger.error("Failed to load cache config %s: %s", yaml_path, e)
        await websocket.send(packer.pack({"__ack__": "error", "msg": str(e)}))
    continue
```

Add a module-level function for `serve_policy.py` to call:

```python
def get_current_cache_bundle() -> Optional[CurrentCacheBundle]:
    """Return the current cache bundle snapshot (if any). Thread-safe."""
    with _bundle_lock:
        return _current_bundle
```

Notes:

- The control connection only "validates + builds shared_storage + writes global bundle"
- The control connection disconnects after sending the ack
- Side effect: the current server creates a conn_policy and triggers `on_task_begin()` when a connection is established; the control short-lived connection creates one empty task lifecycle. This is acceptable -- the timer may record one extra empty entry, but does not affect functional correctness

#### F2. `scripts/serve_policy.py` Modifications

Existing code structure:
- `_wrap_policy(base_policy, args, *, quiet, eager, shared_cache)` builds the wrapper chain
- In `main()`, the concurrent branch pre-builds `shared_cache`, then defines `_connection_policy_factory(shared_base_policy)` calling `_wrap_policy`

Changes:

1. `_wrap_policy` adds global bundle check **before the `args.cache_config` check**:

```python
def _wrap_policy(base_policy, args, *, quiet=False, eager=False, shared_cache=None):
    policy = base_policy

    # --- New: global bundle has highest priority (dynamic YAML switching scenario) ---
    # Must be placed before the args.cache_config branch,
    # otherwise when server starts without --cache_config, the bundle path won't take effect.
    from openpi.serving.websocket_policy_server import get_current_cache_bundle
    bundle = get_current_cache_bundle()
    if bundle is not None:
        from openpi.cache.config import build_per_connection_components
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        components = build_per_connection_components(
            bundle.cache_config,
            bundle.shared_storage,
            quiet=True,
        )
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
        )
        policy = InferenceInterceptor(
            policy, timer=components["timer"],
            orchestrator=orchestrator, eager=eager,
        )
    elif args.cache_config is not None:
        # --- Existing logic (--cache_config at startup, no dynamic switching) ---
        from openpi.cache.config import (
            build_cache_components,
            build_per_connection_components,
            load_cache_config,
        )
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        if shared_cache is not None:
            cache_config = load_cache_config(args.cache_config)
            components = build_per_connection_components(
                cache_config, shared_cache["storage"], quiet=True,
            )
        else:
            cache_config = load_cache_config(args.cache_config)
            components = build_cache_components(cache_config)
            if quiet:
                components["timer"]._quiet = True

        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
        )
        policy = InferenceInterceptor(
            policy, timer=components["timer"],
            orchestrator=orchestrator, eager=eager,
        )
    elif args.cache:
        # --- Existing logic (--cache without YAML) ---
        ...

    # Subsequent record / collect wrappers unchanged
    ...
    return policy
```

This way, even if the server starts without `--cache_config`, after `load_cache_config` control messages write a bundle, new connections correctly take the bundle path.

2. The concurrent branch in `main()` remains unchanged:

```python
# In main(), when starting in concurrent mode, still pre-build shared_cache as before
# If --cache_config specifies an initial YAML, pre-build shared_storage
# If --cache_config is not specified, shared_cache=None; the first run injects
# a bundle via load_cache_config from the control script, and subsequent new
# connections read from the bundle
if args.concurrent:
    shared_cache = None
    if args.cache_config is not None:
        from openpi.cache.config import build_shared_storage, load_cache_config
        cache_config = load_cache_config(args.cache_config)
        shared_cache = {"storage": build_shared_storage(cache_config)}

    def _connection_policy_factory(shared_base_policy):
        # _wrap_policy internally checks global bundle first
        return _wrap_policy(
            shared_base_policy, args, quiet=True, eager=True,
            shared_cache=shared_cache,
        )
    ...
```

Key point: `get_current_cache_bundle()` inside `_wrap_policy` has higher priority than the `shared_cache` parameter. After the control script sends `load_cache_config`, subsequent new connections take the bundle path; connections before that take the existing `shared_cache` path.

Storage sharing semantics:

- Multiple worker connections within the same run share the same `bundle.shared_storage` (consistent with existing concurrent mode semantics)
- When switching YAML between runs, `load_cache_config` builds new `shared_storage`, atomically replaces the entire bundle
- Already-running old connections continue using the old bundle's `shared_storage`, unaffected by the new bundle (Python reference counting ensures the old object is not garbage collected)

#### F3. Control Script Side (inside `exp/run_cache_experiments.py`)

```python
def _send_cache_config(server_url: str, yaml_path: str) -> None:
    """Send load_cache_config control message via a dedicated short-lived connection."""
    import asyncio
    import msgpack
    import websockets

    async def _send():
        async with websockets.connect(server_url) as ws:
            _metadata = await ws.recv()  # Read server metadata
            msg = {"__ctrl__": "load_cache_config", "yaml_path": str(Path(yaml_path).resolve())}
            await ws.send(msgpack.packb(msg))
            resp = msgpack.unpackb(await ws.recv())
            if resp.get("__ack__") != "load_cache_config":
                raise RuntimeError(f"Config switch failed: {resp}")
            logger.info("Server switched to bundle v%s: %s", resp.get("version"), yaml_path)
    asyncio.run(_send())
```

Single run flow:
1. `_send_cache_config(f"ws://{host}:{port}", yaml_path)` -- switch YAML, wait for ack
2. Start `examples/libero/main.py --host H --port P --num_workers N` -- N workers each create connections, reading the same bundle
3. Wait for all workers to complete
4. Switch to next run

Do not modify:

- `packages/openpi-client/` -- control script uses `websockets` + `msgpack` directly
- `examples/libero/main.py` -- unchanged; each run naturally opens new connections

---

### Test Specifications

#### T1. `tests/cache/components/test_key_builder_cp1_experiment.py`

```python
"""CP1 experiment key builder tests."""

class TestCP1MeanPoolKeyBuilder:
    def test_output_shapes(self):
        """Verify vision_0=[2048], prompt_emb=[2048], robot_state=[32]"""
        builder = CP1MeanPoolKeyBuilder()
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert keys["vision_0"].shape == (2048,)
        assert keys["prompt_emb"].shape == (2048,)
        assert keys["robot_state"].shape == (32,)

    def test_cp3_same_as_cp1(self):
        """Currently CP3 behaves identically to CP1."""
        ...

    def test_vision_2_included_when_enabled(self):
        """Output should include vision_2 when enabled_fields includes vision_2."""
        ...

    def test_vision_2_excluded_when_disabled(self):
        """Output should not include vision_2 when enabled_fields excludes vision_2."""
        ...

    def test_action_chunk_cached(self):
        """cached_data includes action_chunk when collect(stage3=...) is called."""
        ...

class TestCP1SpatialPool16KeyBuilder:
    def test_output_shapes(self):
        """vision_0=[32768], prompt_emb=[2048], robot_state=[32]"""
        ...

    def test_spatial_pool_correctness(self):
        """Hand-compute 4x4 pool result and verify consistency."""
        # Construct 256 tokens where 4x4 blocks are identical; after pooling should equal block value
        ...

class TestCP1SpatialPool64KeyBuilder:
    def test_output_shapes(self):
        """vision_0=[8192], prompt_emb=[2048], robot_state=[32]"""
        ...

class TestCP1MaxPoolKeyBuilder:
    def test_output_shapes(self):
        """vision_0=[2048], prompt_emb=[2048], robot_state=[32]"""
        ...

    def test_max_pool_picks_maximum(self):
        """Verify the maximum is actually taken."""
        ...
```

##### Test Fixtures

```python
def _make_fake_stage1(
    emb_dim: int = 2048,
    state_dim: int = 32,
    num_prompt_tokens: int = 20,
) -> _FakeStage1:
    """Construct a fake Stage1Output for testing."""
    prefix_len = 256 * 3 + num_prompt_tokens  # 3 images + prompt
    prefix_embs = torch.randn(1, prefix_len, emb_dim)
    state = torch.randn(1, state_dim)
    return _FakeStage1(prefix_embs, state)

class _FakeStage1:
    def __init__(self, prefix_embs, state):
        self.prefix_embs = prefix_embs
        self.state = state

class _FakeStage3:
    def __init__(self, action_chunk):
        self.action_chunk = action_chunk
```

#### T2. `tests/cache/test_in_memory_backend_experiment.py`

```python
"""InMemoryBackend multi-field retrieval and fusion tests."""

class TestWeightedRrf:
    def test_rrf_basic_ranking(self):
        """RRF result should be consistent when two fields rank consistently."""
        ...

    def test_rrf_conflicting_fields(self):
        """When two fields rank in opposite order, higher-weight field dominates."""
        ...

    def test_zero_weight_field_skipped(self):
        """Fields with weight=0 do not participate in RRF computation."""
        ...

class TestWeightedScoreSum:
    def test_sum_basic(self):
        """Hand-compute percentile norm + weighted sum and verify ranking is correct."""
        ...

    def test_tau_effect(self):
        """Verify robot_state's exp(-d/tau) conversion is correct."""
        ...

    def test_percentile_normalization(self):
        """Verify p5/p95 normalization clip behavior."""
        ...

class TestFiltering:
    def test_task_key_filter(self):
        """task_key filtering returns only matching entries."""
        ...

    def test_checkpoint_id_filter(self):
        """checkpoint_id filtering is correct."""
        ...

class TestBackwardCompat:
    def test_no_fusion_method_uses_single_field(self):
        """fusion_method=None uses legacy single-field cosine (doesn't break existing tests)."""
        ...

class TestArtifactLoading:
    def test_load_artifact(self, tmp_path):
        """Test load_artifact loading and vector_dims validation."""
        ...

    def test_load_artifact_dims_mismatch(self, tmp_path):
        """vector_dims mismatch should raise ValueError."""
        ...
```

#### T3. `tests/cache/test_config_experiment.py`

```python
"""Experiment configuration schema and factory tests."""

class TestConfigParsing:
    def test_full_experiment_yaml(self, tmp_path):
        """Complete experiment YAML parses correctly."""
        yaml_content = """
enabled: true
key_builder:
  type: cp1_mean_pool
keys:
  vision_0: {enabled: true, weight: 0.75}
  vision_1: {enabled: true, weight: 0.25}
  vision_2: {enabled: false, weight: 0.0}
  prompt_emb: {enabled: true, weight: 0.0}
  robot_state: {enabled: true, weight: 0.25}
checkpoints:
  cp1:
    enabled: true
    gate: {type: always_search}
    judge: {type: always_hit}
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      rrf_k: 60
      field_similarity:
        vision_0: {type: cosine}
        vision_1: {type: cosine}
        prompt_emb: {type: cosine}
        robot_state:
          type: l2
          to_similarity: {type: exp, tau: 0.334717}
backend:
  type: in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    prompt_emb: 2048
    robot_state: 32
"""
        ...

class TestConfigValidation:
    def test_score_sum_requires_normalization(self):
        """weighted_score_sum_knn without score_normalization should error."""
        ...

    def test_cp1_builder_requires_vision_0(self):
        """cp1_* builder requires vision_0 enabled."""
        ...

class TestConfigFactory:
    def test_build_mean_pool_builder(self):
        """Config factory correctly instantiates CP1MeanPoolKeyBuilder."""
        ...

    def test_build_weighted_rrf_strategy(self):
        """Config factory correctly instantiates WeightedRrfKnnStrategy."""
        ...
```

---

### File Change Checklist Summary

| Phase | File | Operation | Description |
|-------|------|-----------|-------------|
| A | `src/openpi/cache/storage_types.py` | Modify | QuerySpec add 3 fields + update query_keys docstring |
| A | `src/openpi/cache/config.py` | Modify | Add dataclasses, extend validation and factories |
| A | `src/openpi/cache/components/search_strategy.py` | Modify | Add 2 Strategies + shared filter |
| B | `src/openpi/cache/components/key_builder.py` | Modify | Add 4 builders + base class + helpers |
| C | `src/openpi/cache/backends/in_memory_backend.py` | Modify | Complete search logic rewrite |
| D | `exp/generate_cache_run_yamls.py` | **New** | Phase 1 generates 64 YAMLs; Phase 1.5/2 second pass |
| 0 | `exp/build_in_memory_cache_artifact.py` | **New** | Offline artifact building |
| 0 | `exp/calibrate_score_sum_stats.py` | **New** | Compute p5/p95 statistics |
| E | `exp/run_cache_experiments.py` | **New** | Experiment control script |
| E | `exp/analyze_cache_results.py` | **New** | Result aggregation |
| F | `src/openpi/serving/websocket_policy_server.py` | Modify | Add `load_cache_config` control message + global state |
| F | `scripts/serve_policy.py` | Modify | `connection_policy_factory` reads global cache config |
| T | `tests/cache/components/test_key_builder_cp1_experiment.py` | **New** | Builder tests |
| T | `tests/cache/test_in_memory_backend_experiment.py` | **New** | Backend fusion tests |
| T | `tests/cache/test_config_experiment.py` | **New** | Config tests |
| T | `tests/cache/test_search_strategy_experiment.py` | **New** | Strategy assembly tests |

Files not modified:
- `src/openpi/cache/orchestrator.py` -- unchanged
- `src/openpi/cache/interceptor.py` -- unchanged
- `src/openpi/cache/cache_storage.py` -- unchanged
- `src/openpi/cache/backend_base.py` -- unchanged
- `src/openpi/cache/types.py` -- unchanged
- `examples/libero/main.py` -- unchanged; each run naturally opens new connections
- `packages/openpi-client/` -- unchanged; control script uses websockets+msgpack directly
