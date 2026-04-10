# Step 4 Discussion: Orchestrator Skeleton (CP1 + CP3)

> This file keeps the design discussion and defense record only.
> The full implementation plan was moved to `logs/step4_plan.log.md`.

## Discussion 1: Component Stability Analysis

### Background

Step 3 (cache storage layer) was marked as `⚠️ Tentatively Complete · High Risk`, reasons:
- Vector DB choice undecided (Qdrant may be replaced)
- Query approach undecided (similarity type, multi-field fusion strategy, top_k, etc.)
- No test coverage, interfaces may change

Step 4's 6 submodules need individual analysis: which are unstable due to coupling with Step 3, and which are unstable due to their own undetermined design.

### Analysis Results

#### Stability Classification

| Component | Coupling with Step 3 | Self-Instability | Overall Risk | Notes |
|-----------|----------------------|-----------------|-------------|-------|
| 4.1 KeyBuilder | **High** | **High** | 🔴 High Risk | See below |
| 4.2 Gate | **None** | **Low** | 🟢 Stable | See below |
| 4.3 Judge | **High** | **Medium** | 🔴 High Risk | See below |
| 4.4 Orchestrator | **Medium** | **Low** | 🟡 Medium | See below |
| 4.5 Interceptor Integration | **Low** | **Low** | 🟢 Stable | See below |
| 4.6 End-to-End Tests | **High** | **Medium** | 🔴 High Risk | See below |

---

#### 4.1 KeyBuilder — 🔴 High Risk (Dual Instability)

**Coupling with Step 3 (High)**:
- `CacheEntry.query_keys` is a `dict[str, Tensor]` whose key names must match `VectorStoreBackend.vector_dims`
- Field names and dimensions output by KeyBuilder are directly constrained by backend configuration
- If the backend switches from Qdrant (supports named vectors + RRF fusion) to FAISS (single vector), the entire multi-field design must change
- `types.py` defines 5 canonical fields (vision_0/1/2, prompt_emb, robot_state), but which fields are actually useful depends on backend configuration and experimental results

**Self-Instability (High)**:
- Step 5 Experiment C will compare multiple key construction approaches (raw state / vision mean pool / state+vision concat / state+action); results will directly invalidate the current PlaceholderKeyBuilder
- CP1 and CP3 use different key construction strategies, but the specific strategies are experiment-driven
- Whether L2 normalize is the best normalization approach is also undetermined

---

#### 4.2 Gate — 🟢 Stable

**Coupling with Step 3 (None)**:
- Gate only decides "whether to query the cache" and does not touch the storage layer

**Self-Instability (Low)**:
- `AlwaysSearchGate` is the simplest possible implementation, almost impossible to get wrong
- Advanced Gates (StateChangeGate, IntervalGate) are deferred to Step 8
- Protocol interface `__call__(context) -> bool` is very simple and unlikely to need changes

---

#### 4.3 Judge — 🔴 High Risk (Dual Instability)

**Coupling with Step 3 (High)**:
- The semantics of `SearchResultLite.score` depend entirely on the backend:
  - Single-field cosine backend: score in [-1, 1]
  - Multi-field RRF fusion: score is a small positive number, magnitude determined by RRF k parameter
- `storage_types.py` comments explicitly state "SimilarityJudge thresholds must be calibrated to match the backend/mode in use"
- Switching backend or fusion strategy invalidates all threshold values and requires recalibration

**Self-Instability (Medium)**:
- Step 5 Experiment B will systematically sweep thresholds (0.99 to 0.80); ThresholdJudge's default 0.98 is just a starting point
- May need warm-start judgment logic (multi-level thresholds) in the future, but that's Step 7

---

#### 4.4 Orchestrator — 🟡 Medium

**Coupling with Step 3 (Medium)**:
- Orchestrator interacts through the `CacheStorage` facade, not directly with the backend
- `CacheStorage`'s interfaces (search/insert/fetch_payload) are relatively stable
- However, `QuerySpec` construction (top_k, filters, query_keys field selection) will change with the backend

**Self-Instability (Low)**:
- The `check()` flow of gate -> key -> search -> judge is well-defined
- `write_async()` starts with synchronous writes; this pattern is unlikely to change
- As a combinator, its stability depends on subcomponents; its own logic is simple

---

#### 4.5 Interceptor Integration — 🟢 Stable

**Coupling with Step 3 (Low)**:
- Interceptor only calls Orchestrator's `check()` / `write()`, does not touch storage

**Self-Instability (Low)**:
- Three TODO slots are clearly defined (line 285/290/294), insertion points won't change
- CP2 stays commented out; CP1 and CP3 insertion logic is well-defined
- Only possible change: how check() return values are handled (which stages to skip on hit), but this is already well-defined in the architecture docs

---

#### 4.6 End-to-End Tests — 🔴 High Risk (Dual Instability)

**Coupling with Step 3 (High)**:
- Tests need to instantiate a backend; if Qdrant backend is unstable/replaced, tests must change too
- Score assertions in tests (e.g., "should hit") depend on backend score semantics

**Self-Instability (Medium)**:
- "Load model and run 10 times" requires real model or mock — mock strategy undetermined
- Specific assertion values in test cases (threshold, L2 distance tolerance) need debugging

---

### Conclusion

**High Risk components (🔴)**: KeyBuilder, Judge, End-to-End Tests — these three are strongly coupled with Step 3 and will also change due to Step 5 experiments. Implementation should:
1. Keep interfaces minimal (Protocol), so concrete implementations can be replaced at any time
2. Avoid "clever" designs in these components; use the simplest placeholders
3. Use in-memory mock backend rather than Qdrant for tests, isolating Step 3 instability

**Stable components (🟢)**: Gate, Interceptor Integration — can be done well once, unlikely to need changes.

**Medium components (🟡)**: Orchestrator — flow is well-defined but parameter construction will change; design with flexibility.

### Revised Stability Assessment

Based on Discussion 1's initial analysis + user feedback, revised as follows:

| Component | Coupling with Step 3 | Self-Instability | Overall | Notes |
|-----------|----------------------|-----------------|---------|-------|
| 4.1 KeyBuilder | **High** | **High** | 🔴 | Needs flexible design: collect all data, config controls switches |
| 4.2 Gate | **None** | **High** (revised up) | 🟡 | Gate implementations will change frequently during experiments |
| 4.3 Judge | **High** | **High** (revised up) | 🔴 | Similar to Gate, will change frequently; also affected by Step 3 score semantics |
| 4.4 Orchestrator | **Medium** | **Low** | 🟢 (revised up) | Does not directly touch Step 3; subcomponents are unstable but call flow is well-defined |
| 4.5 Interceptor Integration | **Low** | **Low** | 🟢 | Well-defined, model-agnostic |
| 4.6 End-to-End Tests | — | — | Deferred | Consider after other code is written |

---

## Discussion Records

### Discussion 1

**Q (User)**: Step 3 is a high-risk unstable step. Which components in Step 4 become unstable due to coupling with Step 3? Which components may also change frequently on their own?

**A**: See the initial analysis table above. KeyBuilder and Judge have dual instability; Gate and Interceptor Integration are stable; Orchestrator is in between.

### Discussion 2

**User Feedback** (correcting initial analysis):
1. **Gate**: During the experiment phase, there will be many different Gates with frequently updated logic -> self-instability should be upgraded to high
2. **Orchestrator**: User considers it acceptable; although subcomponents (e.g., QuerySpec) are unstable, the call flow is well-defined -> overall risk can be lowered
3. **End-to-End Tests**: Not considered for now; do after code is written
4. **KeyBuilder**: Should be designed flexibly — collect all available data, use config/switches to decide which to use. `--collect` has already done simple data collection, consider integration
5. **Judge**: Similar to Gate, will change frequently, but additionally unstable because upstream data comes from Step 3
6. **Interceptor Integration**: Agree it's stable; the entire cache system should be transparent to the model (except that Gate/Judge can interrupt or resume the inference flow)

### Discussion 3: KeyBuilder Integration with --collect + Hook vs Stage Output

**Q (User)**: How should KeyBuilder integrate with --collect? What's the difference between approaches A and B?

**A**: Core differences between the two data acquisition approaches:

| Dimension | A: Forward Hook | B: Read Stage Output |
|-----------|----------------|---------------------|
| Extra copies | Each hook needs `.detach().clone()`, causing GPU memory allocation | Zero-copy, directly reads return values |
| Call overhead | Python callback inserted between CUDA kernels may break GPU pipeline | No extra calls |
| torch.compile compatibility | Hook is a Python callback, may break traced graph | Stage functions can be fully compiled |
| Data granularity | Can capture intermediate outputs from any model layer (e.g., noise action from each flow matching step) | Can only get fields exposed by the Stage API |

Key difference: The Stage API exposes stage1(vision+state), stage2(KV cache), stage3(final action) — **cannot access noise_action_steps and other internal model data**. But if KeyBuilder only needs state and final action (current Placeholder design), approach B has zero overhead and is fully sufficient.

**Decision**: Adopt approach B (read Stage output), do not integrate with --collect. Reasons:
- Zero performance overhead
- Does not break torch.compile
- Current and near-term needs (state, action) are met
- If finer-grained data is needed in the future, extend the Stage API output on a case-by-case basis

### Discussion 4: Full Skip vs Warm Start Decision Maker + Gate/Judge Data Relationship with KeyBuilder

**Q (User)**: For two cache usage modes (directly return clean action vs return noise action with partial skip), who makes the decision?

**A**: Responsibility breakdown across components:
- **Gate**: Only decides "whether to query the cache" (True/False)
- **Judge**: Determines hit type (MISS / FULL_HIT / WARM_START)
- **Orchestrator**: Orchestrates the flow, packages results
- **Interceptor**: Controls data flow based on hit type (skip/partial execution/full execution)

For CP2 as an example, Judge has three threshold levels:
- similarity > cp2_full_threshold -> FULL_HIT (return cached clean action)
- similarity > cp2_warm_threshold -> WARM_START (return cached noise action x_t, run remaining flow matching steps)
- similarity < cp2_warm_threshold -> MISS

**Q (User)**: Will Gate and Judge use the data acquired by KeyBuilder and the built keys?

**A**: Analysis as follows:

**Gate — Some implementations will use it**:
- `AlwaysSearchGate`: Does not need any data
- `StateChangeGate` (future): Needs to compare distance between current state and previous state -> can directly reuse keys built by KeyBuilder (already L2 normalized), compute cosine distance
- `VisionChangeGate` (hypothetical): Needs vision embeddings -> also a KeyBuilder data source

**Judge — Indirect dependency + possible future direct dependency**:
- `ThresholdJudge`: Only looks at score, does not directly use keys. But score is the result of query key search; if keys change -> score distribution changes -> thresholds become invalid (indirect coupling)
- Future `ReScoreJudge`: May use richer raw data for secondary scoring -> needs access to raw data collected by KeyBuilder

**Conclusion**: Both Gate and Judge may need access to data collected/built by KeyBuilder. The design needs a shared data channel.

**Decision**:
1. KeyBuilder is responsible for buffering collected raw data and built keys; Gate/Judge read from KeyBuilder
2. Gate and Judge interface design must be compatible with KeyBuilder's data (receive KeyBuilder reference or its buffered data)
3. **Hardware overhead minimization principle**:
   - Buffered data should stay on the original device as much as possible (no unnecessary CPU copies of GPU tensors)
   - Only move data when truly needed (e.g., `.cpu()` only when writing to storage)
   - If built keys are GPU tensors, Gate's cosine distance comparison also runs on GPU, avoiding sync
   - Buffer lifetime = single inference cycle; release references immediately after inference, don't accumulate memory
   - Avoid `.clone()` — if stage output won't be overwritten before the next inference, just hold a reference
4. **All interactions with Step 3 must go through abstraction layers**: Step 3 designed the `VectorStoreBackend`(ABC) -> `CacheStorage`(facade) multi-layer structure to isolate the underlying vector DB. All components in Step 4 that interact with storage (Orchestrator, KeyBuilder writes, Judge's indirect score dependency) must operate only through the `CacheStorage` facade, **never directly call backend or Qdrant API**. This way Step 4 code needs no changes when the backend is replaced.

### Discussion 5: Timing Integration

**User Requirement**: Step 4 code should integrate with Step 2's SystemTimer for precise timing, without significantly impacting performance.

**Design Decision**:
- **Orchestrator manages all timing centrally**: Components themselves (KeyBuilder/Gate/Judge) do not directly hold timer references; timing is done in Orchestrator's check()/write() using `with timer.measure()` for each substep
- **Use CPU backend throughout** (`PerfCounterBackend`): Gate/Judge are pure Python logic; KeyBuilder.build() contains GPU normalize but the main cost is D2H transfer; CacheStorage.search() is CPU dispatch. Using CUDA events would introduce unnecessary event record overhead
- **Minimal performance impact**: `perf_counter_ns()` call takes ~50ns; each check has about 6 probes = ~300ns, negligible compared to ~50ms inference latency; when `enabled=False`, `measure()` is a pure no-op (zero overhead)

**Probe List**:

Interceptor level (coarse): `cp1_check`, `cp1_write`, `cp3_check`
Orchestrator level (fine): `cp{1,3}_{collect, gate, build, search, judge, fetch}`

---


## Defense Section

> Role note: The following content only records the questioner's challenges. Each topic retains only the question, rationale, and evidence, for subsequent defense and plan revision.

### Topic 1: Is the actual integration point for CP3 missing?

**Question**:
The current `src/openpi/cache/interceptor.py` only has three `TODO(Step 4)` slots: after Stage 1, after Stage 2, and after Stage 3.  
But `docs/cache_system_architecture_chinese.md` defines `should_skip_inference()` as "called before inference begins; if CP3 pre-scheduled a cached action in the previous cycle, return immediately and skip the entire inference."  
If Step 4 insists that "the existing three TODO slots are sufficient," how does CP3 implement "skip the next complete inference"?

**Rationale**:
- The current plan conflates CP3's "judgment point" and "consumption point."
- Without a pre-inference consumption point, CP3 can at most "remember an action" but cannot actually skip.

**Evidence**:
- `src/openpi/cache/interceptor.py`
- `docs/cache_system_architecture_chinese.md`

**Defense**:

The challenge is valid. The current plan indeed missed CP3's "consumption integration point."

The architecture doc clearly describes that `should_skip_inference()` should be called before inference begins (at the top of `infer()`), while the existing three TODO slots are all inside the inference pipeline. CP3's complete workflow is:

1. **Write point** (after Stage 3): Record "current action -> next action" mapping
2. **Consumption point** (at the entry of the next `infer()` call): Check if there's a pre-scheduled action; if so, return directly and skip the entire inference

However, the complete mechanism of `should_skip_inference()` + `schedule_next_action()` depends on **Step 6's DeferredWriter** (because `next_action_chunk` can only be filled in the next cycle). Step 4 cannot implement complete CP3 skipping.

**Plan Correction**: In Step 4, CP3 is downgraded to "skeleton infrastructure":
- Reserve `schedule_next_action()` and `should_skip_inference()` interface stubs (empty implementations) in Orchestrator, and a CP3 consumption slot at the `infer()` entry in Interceptor
- **Do not** implement actual CP3 write and skip logic (deferred to Step 6)
- CP3 check is only for verifying infrastructure availability, expected to always MISS

### Topic 2: Does CP3 write semantics conflict with the current `CachePayload` contract?

**Question**:
The current `src/openpi/cache/storage_types.py` requires that `CheckpointID.CP3` payloads must contain `next_action_chunk`.  
But the architecture doc for Step 6 explicitly states: CP3's `next_action_chunk` can only be filled after the next cycle's action is produced, hence the need for `DeferredWriter`.  
If Step 4 plans to "enable CP1 and CP3" and `write_async()` degrades to synchronous writes first, where exactly does `next_action_chunk` come from during CP3 writes?

**Rationale**:
- The current data contract already defines CP3 as a "current action -> next action" mapping.
- Without a deferred writer, CP3 entries appear impossible to construct as valid `CachePayload`.

**Evidence**:
- `src/openpi/cache/storage_types.py`
- `docs/cache_system_architecture_chinese.md`

**Defense**:

The challenge is valid. `CachePayload.validate_for_checkpoint(CP3)` explicitly requires `next_action_chunk is not None`, and without a DeferredWriter, it's impossible to have cycle N+1's action when writing CP3 during cycle N.

This is consistent with Topic 1's conclusion: Step 4 cannot complete valid CP3 writes.

**Plan Correction**: Step 4 will not do CP3 writes. CP3 scope in Step 4 is limited to:
- Reserve CP3 check/write flow skeleton in Orchestrator
- Reserve CP3 consumption slot in Interceptor (at `infer()` entry)
- CP3's `CachePayload` contract remains unchanged, validation not relaxed
- Actual CP3 write + DeferredWriter deferred to Step 6

### Topic 3: Can the Stage API actually provide the fields planned for KeyBuilder?

**Question**:
`src/openpi/cache/types.py` defines `vision_0` / `vision_1` / `vision_2` / `prompt_emb` / `robot_state` as 5 canonical fields.  
But the current `src/openpi/models_pytorch/pi0_pytorch.py`'s `Stage1Output` only exposes `state`, `prefix_embs`, mask, and position ids — it does not separately expose per-camera vision embeddings or prompt embeddings.  
If Step 4's KeyBuilder is supposed to "collect all available data and use config switches to decide which fields to use," what are the actual sources for these fields?

**Rationale**:
- The fields stably available from the current public interface are far fewer than the field set discussed in the plan.
- If field sources are unclear, KeyBuilder's flexible design will be built on nonexistent data.

**Evidence**:
- `src/openpi/cache/types.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`
- `logs/step4_discussion.log.md`

**Defense**:

The challenge is valid. There is a clear gap between the fields exposed by the Stage API and the 5 canonical fields in `types.py`.

Actually available fields from current Stage outputs:
- `Stage1Output.state` -> maps to `ROBOT_STATE`, `[B, 32]`, **directly usable**
- `Stage1Output.prefix_embs` -> `[B, prefix_len, emb_dim]`, this is a mixed sequence of vision + language tokens, **cannot be directly split** into `vision_0/1/2` and `prompt_emb`
- `Stage3Output.action_chunk` -> `[B, 50, 32]`, usable for CP3 key construction

Fields not obtainable from the Stage API:
- `VISION_0/1/2`: Per-camera vision embeddings are internal outputs of `multi_modal_projector`, merged into `prefix_embs` and no longer distinguishable
- `PROMPT_EMB`: Language token embeddings are similarly mixed into `prefix_embs`

**Plan Correction**:
- `PlaceholderKeyBuilder` only uses `ROBOT_STATE` (`state` field), which is the only confirmed available field
- The 5 canonical fields in `types.py` are retained as "potentially available fields in the future," but Step 4 code comments clearly annotate which are currently available and which require Stage API extension or hooks to obtain
- KeyBuilder's "flexible field switch" design is narrowed to: **only the `robot_state` switch is currently available**; `prefix_embs` can serve as an experimental second option (requires mean pool dimension reduction on `[B, prefix_len, emb_dim]`); separate vision/prompt fields are deferred until Stage API extension

### Topic 4: Will KeyBuilder's "flexible field switch" be blocked by Step 3's existing validation?

**Question**:
Currently, `src/openpi/cache/cache_storage.py`'s `_check_entry_dims()` requires every field declared by the backend to appear in `entry.query_keys`.  
And `src/openpi/cache/README.md` already lists this rule as a known issue: it's too strict, and should be changed to validate only the intersection.  
If Step 4 adopts the "collect all data, select a subset of fields to write based on config" design, how can it avoid errors at the `CacheStorage.insert()` stage?

**Rationale**:
- Step 4's field flexibility depends on Step 3's validation semantics being lenient enough.
- The current implementation and current plan appear to be in direct conflict here.

**Evidence**:
- `src/openpi/cache/cache_storage.py`
- `src/openpi/cache/README.md`

**Defense**:

The challenge is valid. The `_check_entry_dims` logic will indeed block the "partial field write" design.

`cache_storage.py:156-168`'s `_check_entry_dims` iterates over `self._dims` (all fields declared by the backend), requiring the entry to contain all fields. If the backend declares `{robot_state: 32, vision_0: 2048}` but KeyBuilder only produces `{robot_state: [32]}`, insert will directly raise ValueError.

`README.md` line 51 already marks this as a fix item: `_check_entry_dims should validate only the intersection of query_keys and backend-declared fields, not require the full set`.

**Plan Correction**: Two solutions, choosing the first:

**(A) Fix `_check_entry_dims` in Step 4 (recommended)**: Change validation logic to intersection mode — only check that fields actually present in the entry match the backend-declared dimensions, without requiring all fields to be present. This is already a TODO marked in README; the change is small and clear.

(B) Step 4's backend only declares `{robot_state: 32}`: Avoids the problem but limits future experimental flexibility.

Choosing A: Fix `_check_entry_dims` to intersection validation in Step 4 implementation, while keeping `_check_query_dims` (search side) in intersection mode as well (it already uses intersection mode).

### Topic 5: Is the assumption of buffering GPU tensor references and avoiding `.clone()` too optimistic?

**Question**:
The current discussion proposes: KeyBuilder `collect()` buffers GPU tensor references, reusing them directly before inference ends, avoiding `.clone()` as much as possible.  
But `src/openpi/cache/interceptor.py` already explicitly calls `torch.compiler.cudagraph_mark_step_begin()`, indicating the staged compiled path is sensitive to output lifetimes.  
Additionally, the Step 3 storage layer explicitly requires all persisted tensors to be CPU contiguous float32.  
Given these premises, which objects can "just hold references," and which must be materialized? Where is the boundary?

**Rationale**:
- "Single-inference transient state" and "storable state" are not clearly distinguished.
- If the boundary is unclear, it's easy to mistakenly pass transient device-side objects across cycles, threads, or layers.

**Evidence**:
- `src/openpi/cache/interceptor.py`
- `src/openpi/cache/storage_types.py`
- `logs/step4_discussion.log.md`

**Defense**:

The challenge is partially valid, but the conclusion is "the current plan's approach is safe."

Regarding CUDAGraph risk: `interceptor.py:144-146` already forcibly downgrades compile mode from `max-autotune` to `max-autotune-no-cudagraphs`. This means the staged path **does not use CUDAGraph**, and output tensors are regular GPU tensors that won't be overwritten by CUDAGraph buffer reuse. The `cudagraph_mark_step_begin()` call is a defensive preservation, not an indication that CUDAGraph is actually enabled.

**Clear reference lifetime boundaries**:

| Data State | Location | Lifetime | Operation Constraints |
|-----------|----------|---------|----------------------|
| **Transient reference** | KeyBuilder._cache | Within a single `infer()` call | GPU tensor reference, no clone. Safety premise: stage outputs are not overwritten within the same `infer()` (staged path has no CUDAGraph) |
| **Query state** | KeyBuilder.build() return value | Within a check() call | CPU float32 L2-normalized. Already materialized (`.cpu().float()`), safe |
| **Storage state** | CacheEntry.query_keys / CachePayload | Persistent | CPU contiguous float32. Materialization is the responsibility of `build()` or the caller |

**Plan Correction**: Add clear annotations in KeyBuilder code comments marking these three state boundaries and safety premises:
```
# SAFETY: References are valid within a single infer() call.
# The staged path uses max-autotune-no-cudagraphs, so stage outputs
# are regular GPU tensors — not CUDAGraph-managed buffers.
# Crossing infer() boundary or writing to storage MUST materialize
# via .detach().cpu().contiguous().float().
```

### Topic 6: Why is Step 4's timing design inconsistent with Step 2's direction?

**Question**:
The current discussion advocates: all cache substeps in Step 4 use CPU backend timing, including paths like `KeyBuilder.build()` that involve GPU normalize / D2H.  
But `logs/step2.log` clearly describes the timing design for future cache components: GPU computation and GPU<->CPU transfers should support CUDA Events, while only CPU logic uses `perf_counter_ns`.  
If Step 4 universally uses CPU probes, are the measurements wall-clock time or actual component execution time? Does this contradict Step 2's "precise timing" goal?

**Rationale**:
- The current argument conflates "overall perceived latency" and "component-level precise timing."
- If timing semantics are not clarified first, subsequent performance analysis will be hard to align.

**Evidence**:
- `logs/step2.log`
- `src/openpi/cache/timing.py`
- `logs/step4_discussion.log.md`

**Defense**:

The challenge is partially valid. Step 2 log indeed lists KeyBuilder as a CUDA Event timing target, but this is based on a premise not yet implemented in Step 4.

Step 2 log lines 36-40 design expectations:

| Component | Device | Timing Approach |
|-----------|--------|----------------|
| Gate / Judge / FAISS CPU search | CPU | `perf_counter_ns` |
| KeyBuilder / VectorStore GPU search | GPU (`cache_stream`) | CUDA Event |
| GPU<->CPU transfer | CUDA `transfer_stream` | CUDA Event |

Step 2 log line 214 further explains: `CacheOrchestrator's cache_stream integrates via register_probe(..., stream=cache_stream)`.

**Key point**: `cache_stream` is a product of Step 8 (system efficiency optimization). Step 4 has no independent cache CUDA stream; all GPU operations (`F.normalize`) execute on the default stream, shared with inference.

Without cache_stream:
- `F.normalize` runs on the default stream, kernel launch ~1us
- `.cpu()` triggers implicit synchronization (waits for all kernels on the default stream to complete); the actual measurement is wall-clock time of D2H transfer
- Using CUDA Events to record these operations on the default stream **produces results essentially identical to perf_counter** (because `.cpu()` already forces synchronization)
- The extra CUDA Event record/synchronize overhead is actually wasteful

**Conclusion**: The current plan's use of CPU backend is a **pragmatic choice**, measuring wall-clock time (i.e., the actual time the inference pipeline is blocked), which is more practically meaningful for Step 5's latency analysis. After Step 8 introduces `cache_stream`, upgrade to CUDA Events per Step 2's design.

**Plan Correction**: Add a note in the timing comments:
```
# NOTE: Step 2 design envisions CUDA Event timing for KeyBuilder once
# cache_stream is introduced (Step 8). Current CPU backend measures
# wall-clock time, which equals GPU time when operations run on the
# default stream with implicit sync (.cpu() calls).
```

### Topic 7: Should `fetch_payload()` be called by Judge or by Orchestrator?

**Question**:
In the architecture doc, `ThresholdJudge` directly holds a `storage` reference and calls `fetch_payload()` on hit.  
But the current discussion splits probes into `search -> judge -> fetch` three stages, more like Orchestrator first lets Judge return "which candidate hit," then centrally fetches the payload.  
These two responsibility divisions are inconsistent. Which one does Step 4 ultimately adopt?

**Rationale**:
- If responsibility boundaries are not unified, subsequent probe ownership, component dependencies, and testing approaches will be unstable.
- Whether Judge is a pure judgment component or a component with storage side effects directly affects interface design.

**Evidence**:
- `docs/cache_system_architecture_chinese.md`
- `logs/step4_discussion.log.md`

**Defense**:

The challenge has some merit, but the "Judge calls fetch_payload" approach is rejected.

The architecture doc's pseudocode is concept-level design, where binding Judge to storage was for simplified illustration. In actual implementation, **Orchestrator calling fetch_payload** is superior, for these reasons:

1. **Judge stays pure judgment**: Judge's input is `(results, checkpoint_id, cached_data)` -> output is `(HitType, winner_id)`. Not holding a storage reference means Judge is fully testable (just pass in mock results), and replacing Judge doesn't require injecting storage dependencies.

2. **Timing separation**: `search -> judge -> fetch` three stages are timed independently. If Judge internally calls fetch, the judge probe measurement mixes in I/O time, making it impossible to distinguish Judge's own latency from fetch I/O latency.

3. **Consistency with two-phase search design**: Step 3's `CacheStorage` already designed a search() -> fetch_payload() two-phase pattern (lightweight search + selective retrieval); Orchestrator controlling this flow is more natural.

4. **Minimal coupling**: Judge does not depend on any Step 3 interfaces, only on `SearchResultLite` (a pure data class). This makes Judge the only judgment component in Step 4 that is not directly coupled to Step 3.

**Conclusion**: Maintain the plan design — Orchestrator calls fetch_payload, Judge is a pure judgment component. The architecture doc's pseudocode is treated as a conceptual reference rather than an interface specification.

### Topic 8: Should Step 4's actual scope be narrowed?

**Question**:
Step 4 currently promises "CP1 + CP3 runnable end-to-end," but from the preceding topics:
- CP3 lacks a proper consumption integration point;
- CP3 writes require `next_action_chunk`, conflicting with the current contract;
- KeyBuilder's flexible field design is blocked by Step 3 validation;
- The timing approach is inconsistent with Step 2's stated goals.

Given this situation, does Step 4's scope need to be redefined?

**Rationale**:
- The current scope may be pressing multiple misaligned prerequisites simultaneously.
- If the scope doesn't converge, implementation will easily become "skeleton looks complete, but no single path is truly end-to-end."

**Evidence**:
- `src/openpi/cache/README.md`
- `docs/cache_system_architecture_chinese.md`
- `logs/step4_discussion.log.md`

**Defense**:

The challenge is valid. Combining conclusions from Topics 1-7, Step 4's scope does need to converge.

**Revised Step 4 Scope**:

| Feature | Original Plan | Revised | Reason |
|---------|--------------|---------|--------|
| CP1 check + write | ✅ End-to-end | ✅ **End-to-end closed loop** | No blockers, can be fully implemented |
| CP1 hit early return | ✅ | ✅ | Skip stage2+3, return cached action |
| CP3 check | ✅ | ⚠️ **Skeleton only** | No CP3 entries -> always MISS, only verifies infrastructure |
| CP3 write | ✅ | ❌ **Deferred** | `next_action_chunk` contract conflict (Topic 2) |
| CP3 consumption (skip next cycle) | Implied | ❌ **Deferred** | Missing consumption integration point (Topic 1), needs Step 6 DeferredWriter |
| KeyBuilder flexible fields | "Collect all" | Narrowed to **`robot_state` only** | Stage API gap (Topic 3) |
| `_check_entry_dims` fix | Not planned | ✅ **Added** | Intersection validation, unblocks field flexibility (Topic 4) |
| Timing | All CPU | All CPU + **comment explanation** | Pragmatic choice, upgrade in Step 8 (Topic 6) |
| Interceptor CP3 consumption slot | None | ✅ **Reserved** | `should_skip_inference()` stub at `infer()` entry |

**Step 4's true closed-loop path**: CP1 end-to-end — first inference misses -> writes to cache -> second inference with same input -> CP1 hits -> skips stage2+3 -> returns cached action.

Complete CP3 implementation deferred to Step 6 (DeferredWriter + schedule_next_action + should_skip_inference).

---
