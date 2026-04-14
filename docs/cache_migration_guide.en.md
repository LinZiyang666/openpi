# Cache Framework Migration Guide

> **Version**: 1.0
> **Status**: Initial release
> **Last updated**: 2026-04-10
>
> **Design doc**: [cache_system_architecture.md](cache_system_architecture.md)
> **Component tutorial**: [cache_system_tutorial.md](cache_system_tutorial.md)
> **Workflow diagrams**: [cache_system_workflow.md](cache_system_workflow.md)

---

## Table of Contents

1. [Overview & Scope](#1-overview--scope)
2. [Architecture Overview (Migration Perspective)](#2-architecture-overview-migration-perspective)
3. [Minimum Viable Migration (CP1)](#3-minimum-viable-migration-cp1)
   - [Step 1: Analyze Your Inference Pipeline](#step-1-analyze-your-inference-pipeline)
   - [Step 2: Define the Minimum Interface Contract](#step-2-define-the-minimum-interface-contract)
   - [Step 3: Implement a Custom KeyBuilder](#step-3-implement-a-custom-keybuilder)
   - [Step 4: Implement a Custom Interceptor](#step-4-implement-a-custom-interceptor)
   - [Step 5: Register with the Config System](#step-5-register-with-the-config-system)
   - [Step 6: Data Collection & Artifact Building](#step-6-data-collection--artifact-building)
   - [Step 7: Verification](#step-7-verification)
4. [Optional Extensions](#4-optional-extensions)
5. [Experimental / Incomplete Items](#5-experimental--incomplete-items)
6. [YAML Config Reference](#6-yaml-config-reference)
7. [FAQ & Pitfalls](#7-faq--pitfalls)

---

## 1. Overview & Scope

### What This Framework Does

OpenPI Cache is a **multi-level inference caching system** that places cache checkpoints at key positions in the inference pipeline. When the current observation is sufficiently similar to a historical one, it skips subsequent computation and reuses the cached action, reducing end-to-end inference latency.

Core design principles:
- **Interceptor pattern**: Cache logic is an external plugin that hooks into the inference pipeline without modifying model internals
- **Pluggable components**: KeyBuilder, Gate, Judge, SearchStrategy, and WritePolicy are all replaceable
- **Backend-agnostic**: The storage layer is abstracted via `VectorStoreBackend` ABC; upper-layer logic is independent of any specific vector database

### Scope

**What this tutorial is**: A migration guide distilled from the current Pi0.5 implementation. It helps you integrate this cache framework with your own model.

**What this tutorial is not**: This repository does not provide out-of-the-box support for non-Pi0.5 models. Migration requires writing a custom KeyBuilder and Interceptor, and potentially adapting data collection and artifact building.

### What You Don't Need to Know

- OpenPI's training pipeline
- Pi0.5 model internals
- JAX-related code (this framework only supports the PyTorch path)

### Migration Path Decision

Before starting, answer these questions to determine your migration path:

| Question | Yes | No |
|----------|-----|----|
| Can your model's inference be split into multiple stages? | Use the Staged API pattern → Step 1 | See [§7 Single Forward Pass Models](#single-forward-pass-models) |
| Can you extract intermediate embeddings from your model? | Implement a custom KeyBuilder → Step 3 | Consider CLIP builder → [§4.3](#43-clip-builder-details) |
| Do you need online inference caching? | Follow Steps 1-7 in full | Offline artifact experiment only → [§4.4](#44-offline-artifact-experiments) |

---

## 2. Architecture Overview (Migration Perspective)

### Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│              You Need to Implement (model-specific)      │
│                                                         │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  Interceptor    │  │  KeyBuilder                  │  │
│  │  (inference     │  │  (intermediate repr →        │  │
│  │   flow control) │  │   query vectors)             │  │
│  └────────┬────────┘  └──────────────┬───────────────┘  │
│           │                          │                   │
├───────────┼──────────────────────────┼───────────────────┤
│           │     Reuse Directly (model-agnostic)          │
│           v                          v                   │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              CacheOrchestrator                      │ │
│  │    gate → collect → build → search → judge → write  │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                 │
│  ┌──────────────────────┴──────────────────────────────┐ │
│  │  SearchStrategy │ SimilarityJudge │ GateFunction    │ │
│  │  WritePolicy    │ SystemTimer     │ YAML Config     │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                 │
│  ┌──────────────────────┴──────────────────────────────┐ │
│  │         CacheStorage + VectorStoreBackend           │ │
│  │         (InMemoryBackend / custom backend)           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Checkpoint Semantics

The framework currently supports two checkpoint IDs:

| Checkpoint | Position in Pi0.5 | Semantics | What You Need to Do |
|------------|-------------------|-----------|---------------------|
| **CP1** | After Stage 1 (vision encoding) | Check cache after observation encoding; on hit, skip all subsequent computation | Find the boundary between "observation processing" and "decision/generation" in your model |
| **CP3** | After Stage 3 (action generation) | After current inference, predict whether the next step can be skipped (experimental) | Keep the framework code, but don't expect actual speedup currently |

> **Note**: The framework only supports `CP1` / `CP3` as checkpoint IDs (the `CheckpointID` enum). The config system also only accepts these two names. Adding new checkpoints requires modifying `types.py`, `config.py`, `orchestrator.py`, and related tests — this is a framework extension beyond the scope of this tutorial.

---

## 3. Minimum Viable Migration (CP1)

### Step 1: Analyze Your Inference Pipeline

**Goal**: Split your model's inference into stages and find the CP1-equivalent position.

The core semantics of CP1: **"After observation encoding is complete, if the current observation is sufficiently similar to a historical one, reuse the historical action and skip all subsequent computation."**

In Pi0.5, inference is split into three stages:

```
Stage 1: Vision encoding + token prep  →  [CP1]  →  Stage 2: LLM forward  →  Stage 3: Flow matching
```

For different model architectures, suggested CP1 positions:

| Model Type | Recommended CP1 Position | Rationale |
|------------|-------------------------|-----------|
| Vision-Language-Action (VLA) | After vision encoder | Vision embedding is the heaviest computation and best scene similarity indicator |
| Diffusion Policy | After condition encoder (image encoder + FiLM) | Skips the entire denoising loop |
| ACT (Action Chunking Transformer) | After CVAE encoder | Skips decoder |
| Single encoder-decoder model | After encoder | Skips decoder |

**Key question**: In your model, where is the boundary between "observation understanding" and "action generation"? That's your CP1.

### Step 2: Define the Minimum Interface Contract

You do not need to replicate Pi0.5's `Stage1Output` / `Stage3Output` data structures. The minimum contracts you need to satisfy are:

#### 2.1 CP1 Checkpoint Input Contract

When CP1 check runs, `orchestrator.check(CheckpointID.CP1, **kwargs)` passes `kwargs` to `KeyBuilder.collect()`. Your stage output must contain the information needed by KeyBuilder to construct query keys.

**Minimum requirement**:
- At least one observation embedding (visual, state, language, or a combination)
- Format: `torch.Tensor` on GPU

#### 2.2 Action Output Contract

Both the action returned on cache hit and the action stored on write are carried by `CachePayload.action_chunk`.

**Minimum requirement**:
- Shape: `[action_horizon, action_dim]`, CPU float32 contiguous
- This is the raw action sequence from your model's final output (before output transform)

#### 2.3 Episode Lifecycle Contract

The inference flow must send episode lifecycle signals at the correct times:

| Signal | When | Call |
|--------|------|------|
| episode start | At the beginning of each episode | `orchestrator.on_episode_start(task_key, episode_id)` |
| broadcast action | After each inference step | `orchestrator.broadcast_action(action)` |
| buffer for write | After each inference step | `orchestrator.buffer_for_write(query_keys, action)` |
| episode end | At the end of each episode | `orchestrator.on_episode_end()` |
| clear | At the end of each inference step | `orchestrator.clear()` |

#### 2.4 Pi0.5 Reference Example

```python
# Pi0.5 Stage Output definitions (for reference only — you don't need to copy these fields)

@dataclass
class Stage1Output:
    state: torch.Tensor           # [B, action_dim]
    prefix_embs: torch.Tensor     # [B, prefix_len, emb_dim] — SigLIP vision tokens + prompt tokens
    prefix_pad_masks: ...
    prefix_att_2d_masks_4d: ...
    prefix_position_ids: ...

@dataclass
class Stage3Output:
    action_chunk: torch.Tensor    # [B, action_horizon, action_dim]
    intermediates: Optional[...]  # flow matching intermediates (optional)
```

### Step 3: Implement a Custom KeyBuilder

KeyBuilder converts your model's intermediate representations into fixed-dimension query vectors for cache retrieval.

#### 3.1 Protocol Interface

```python
class QueryKeyBuilder(Protocol):
    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        """Extract raw tensor references from stage outputs (on GPU, no copy)."""
        ...

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Reduce + GPU→CPU transfer. Returns {field_name: [dim] CPU float32}.
        Cosine fields should be L2-normalized; L2 distance fields (robot_state) keep raw vectors."""
        ...

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        """Expose raw tensors cached by collect() (on GPU), for Gate/Judge use."""
        ...

    def clear(self) -> None:
        """Release cached references. Called at the end of each inference step."""
        ...
```

#### 3.2 Field Mapping Rules

Query key field names **must** be a subset of the 5 canonical fields:

```python
# src/openpi/cache/types.py
VISION_0 = "vision_0"      # Primary camera / first visual input
VISION_1 = "vision_1"      # Second visual input (e.g., wrist camera)
VISION_2 = "vision_2"      # Third visual input
PROMPT_EMB = "prompt_emb"  # Language / task embedding
ROBOT_STATE = "robot_state" # State vector
```

**Mapping strategy**:
- If your model has visual embeddings → map to `vision_0` (use `vision_1`, `vision_2` for multi-camera)
- If it has language/task embeddings → map to `prompt_emb`
- If it has a state vector → map to `robot_state`
- Fields can be omitted (set `enabled: false` in YAML) but new fields cannot be added

> **Deep modification**: If the existing 5 fields cannot cover your needs (e.g., you need `tactile_emb`), you must modify `CACHE_QUERY_FIELDS` in `src/openpi/cache/types.py`, validation logic in `src/openpi/cache/config.py`, backend `vector_dims` declarations, and related tests. This is a framework extension beyond the main scope of this tutorial.

#### 3.3 Implementation Example

Here is a hypothetical Diffusion Policy KeyBuilder example:

```python
from openpi.cache.components.key_builder import QueryKeyBuilder
from openpi.cache.types import VISION_0, ROBOT_STATE, CheckpointID

class DiffusionPolicyKeyBuilder:
    """KeyBuilder for a Diffusion Policy model.

    Extracts visual embedding from the image encoder output
    and robot state from the condition vector.
    """

    def __init__(self) -> None:
        self._cache: dict[str, torch.Tensor] = {}

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if "encoder_output" in stage_outputs:
            enc = stage_outputs["encoder_output"]
            self._cache["visual_feat"] = enc.visual_features  # [B, C, H, W] GPU
            self._cache["state"] = enc.state_vector            # [B, state_dim] GPU

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        import torch.nn.functional as F

        # Global average pooling: [B, C, H, W] -> [C]
        vis = self._cache["visual_feat"][0]          # drop batch: [C, H, W]
        vis_pooled = vis.mean(dim=(1, 2))            # [C]
        vis_key = F.normalize(vis_pooled, dim=0)     # cosine field: L2 normalize

        # L2 distance field: keep raw vector, do NOT normalize
        state = self._cache["state"][0]              # [state_dim]

        return {
            VISION_0: vis_key.cpu().float().contiguous(),
            ROBOT_STATE: state.cpu().float().contiguous(),
        }

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()
```

**Key points**:
- `collect()` only stores GPU tensor references — no copies
- `build()` is the only GPU→CPU data transfer point
- Returned tensors must be **1D, CPU, float32, contiguous**
- **Normalization rules differ by field**: cosine similarity fields (`vision_0/1/2`, `prompt_emb`) should be L2-normalized; L2 distance fields (`robot_state`) **must keep raw vectors** — normalizing would destroy distance semantics
- Output dimensions must match `backend.vector_dims` in the YAML config

#### 3.4 CLIP Low-Coupling Branch

If you don't want to (or can't) extract visual embeddings from model internals, you can use the CLIP KeyBuilder. It encodes raw input images with a standalone CLIP encoder, independent of the target model's internal representations.

**Prerequisites**: Your inference flow can provide raw input images (numpy arrays) and a `robot_state` vector.

**Limitations**:
- CLIP embeddings don't reflect features learned during model training; retrieval quality may be lower than model-internal embeddings
- Still requires `robot_state`
- CLIP encoding adds extra inference overhead (~5-10ms per image)

See [§4.3](#43-clip-builder-details) for usage details.

### Step 4: Implement a Custom Interceptor

The Interceptor is the integration layer between the cache system and the inference pipeline. It wraps your policy/model and inserts cache check and write logic into the inference flow.

#### 4.1 Host Runtime Minimum Contract

Your Interceptor must satisfy these conditions:

| Contract | Description |
|----------|-------------|
| **BasePolicy semantics** | Implement `infer(obs) -> dict` as a transparent drop-in for the original policy |
| **Access to staged inference methods** | Ability to call the split stage functions (e.g., `run_stage1()`, `run_stage2()`, etc.) |
| **Access to transform pipeline** | Ability to run input transform (raw obs → model input) and output transform (model output → action) |
| **Device information** | Know which GPU the model is on, for tensor transfers |
| **Episode lifecycle** | Call `on_episode_start` / `on_episode_end` at the correct times |
| **Wrapper ordering** | If integrating with `serve_policy.py`, the Interceptor's position in the wrapper chain must be correct |

#### 4.2 Implementation Template

```python
from openpi_client import base_policy as _base_policy
from openpi.cache.orchestrator import CacheOrchestrator, CheckResult
from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID


class MyInterceptor(_base_policy.BasePolicy):
    """Cache-aware inference wrapper for YourModel."""

    def __init__(self, policy, orchestrator: CacheOrchestrator, timer=None):
        self._policy = policy
        self._model = policy.model              # Your model instance
        self._input_transform = policy.input_transform
        self._output_transform = policy.output_transform
        self._device = policy.device
        self._orchestrator = orchestrator
        self._timer = timer

    # ---- Episode lifecycle ----

    def on_episode_start(self, task: str, episode_id: int) -> None:
        self._orchestrator.on_episode_start(task_key=task, episode_id=str(episode_id))

    def on_episode_end(self, success: bool) -> None:
        self._orchestrator.on_episode_end()

    # ---- Inference ----

    def infer(self, obs: dict) -> dict:
        # 1. Input transforms
        inputs = self._input_transform(obs)
        model_inputs = self._to_device(inputs)

        # 2. Stage 1: observation encoding
        encoder_output = self._model.encode(model_inputs)

        # 3. CP1 check
        cp1_result = self._orchestrator.check(
            CheckpointID.CP1,
            encoder_output=encoder_output,
        )

        if cp1_result.hit_type == HitType.FULL_HIT:
            # Cache hit: skip subsequent computation
            cached_action = cp1_result.payload.action_chunk
            self._orchestrator.broadcast_action(cached_action)
            if cp1_result.query_keys is not None:
                self._orchestrator.buffer_for_write(cp1_result.query_keys, cached_action)
            self._orchestrator.clear()
            return self._build_output(inputs, cached_action)

        # 4. Cache miss: continue normal inference
        action = self._model.decode(encoder_output)

        # 5. Post-inference: buffer for write
        action_cpu = action[0].detach().cpu().float().contiguous()
        self._orchestrator.broadcast_action(action_cpu)
        if cp1_result.query_keys is not None:
            self._orchestrator.buffer_for_write(cp1_result.query_keys, action_cpu)
        self._orchestrator.clear()

        return self._build_output(inputs, action)

    def _build_output(self, inputs, action) -> dict:
        # Note: when integrating within the OpenPI framework, output_transform
        # expects inputs as CPU numpy arrays with batch dim removed, not GPU tensors.
        # Actual code should be:
        #   outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        #   outputs = self._output_transform(outputs)
        # This is simplified pseudocode; adjust for your framework.
        outputs = {"actions": action, "state": inputs["state"]}
        return self._output_transform(outputs)
```

> **Note**: The template above is conceptual pseudocode. In the OpenPI framework, `output_transform` expects **CPU numpy arrays with the batch dimension removed** as input (see `policy.py:138` and `interceptor.py:341`). If integrating within OpenPI, you need to call `jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)` before `_output_transform`.

**Key points**:
- `orchestrator.check()` returns `CheckResult`; `hit_type` is either `FULL_HIT` or `MISS`
- On hit, `payload.action_chunk` has shape `[action_horizon, action_dim]` (no batch dim)
- `query_keys` is populated on all paths (hit, miss, gate skip), ensuring `buffer_for_write()` is always callable
- `orchestrator.clear()` must be called at the end of each inference step to release KeyBuilder cache

### Step 5: Register with the Config System

#### 5.1 Register a New KeyBuilder Type

Add your builder to `_build_key_builder()` in `src/openpi/cache/config.py`:

```python
def _build_key_builder(cfg: CacheConfig) -> QueryKeyBuilder:
    kb_type = cfg.key_builder.type
    if kb_type == "placeholder":
        return PlaceholderKeyBuilder()
    elif kb_type == "cp1_mean_pool":
        return CP1MeanPoolKeyBuilder(enabled_fields=...)
    # ... existing types ...
    elif kb_type == "my_diffusion_policy":  # <-- new
        from your_module import DiffusionPolicyKeyBuilder
        return DiffusionPolicyKeyBuilder()
    else:
        raise ConfigValidationError(f"Unknown key_builder type: {kb_type}")
```

#### 5.2 Update Config Validation

In `validate_cache_config()`, ensure the new key_builder type is recognized and add necessary cross-validation checks (e.g., verify `vector_dims` matches the new builder's output dimensions).

#### 5.3 YAML Configuration

```yaml
key_builder:
  type: my_diffusion_policy

backend:
  type: in_memory
  vector_dims:
    vision_0: 512      # Must match your KeyBuilder.build() output dimension
    robot_state: 14     # Must match your state vector dimension
```

Dimensions in `vector_dims` **must** match the tensor length for each field returned by KeyBuilder's `build()`. Mismatches will trigger dimension validation errors at `CacheStorage.insert()` / `CacheStorage.search()` time.

### Step 6: Data Collection & Artifact Building

The cache can **start empty** — entries accumulate online via `write_policy` at episode end, with no pre-built artifact required. However, if you want the cache to have data available from the very first episode, you can pre-build an artifact (pickle file) for `InMemoryBackend` to load at startup.

#### 6.1 Minimum Artifact Contract

The artifact is a pickle file containing a **dict** (not a bare list), with the following format:

```python
{
    "key_builder_type": "my_diffusion_policy",   # KeyBuilder type used during building
    "checkpoint_id": "CP1",                       # checkpoint identifier
    "vector_dims": {"vision_0": 512, "robot_state": 14},  # must match backend config
    "entries": [CacheEntry(...), CacheEntry(...), ...],    # list of CacheEntry objects
}
```

> **Important**: `InMemoryBackend.load_artifact()` validates that the artifact's `vector_dims` matches the backend's configured `vector_dims`. A mismatch will raise an error.

Each entry requires:

```python
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

entry = CacheEntry(
    id="trajectory_001:step_0",           # Unique ID
    checkpoint_id=CheckpointID.CP1,
    query_keys={
        "vision_0": torch.tensor([...]),  # [dim] CPU float32
        "robot_state": torch.tensor([...]),
    },
    payload=CachePayload(
        action_chunk=torch.tensor([...]), # [action_horizon, action_dim] CPU float32
    ),
    step_idx=0,
    prev_ids=[],
    next_ids=["trajectory_001:step_1"],
    trajectory_id="trajectory_001",
)
```

**Tensor contract**: All tensors must be CPU, contiguous, float32.

#### 6.2 Data Collection

You need to collect embeddings and action data from your target model's inference process.

**The existing `collect/` module** is a Pi0.5 reference implementation — it hardcodes Pi0.5-specific hook points such as `paligemma_with_expert`, `action_in_proj`, and `action_out_proj`. For other models, you have two options:

1. **Write your own collection logic** referencing the existing implementation: Add forward hooks or manually capture intermediate representations during your model's inference, outputting to HDF5 files
2. **Build artifacts directly**: If you already have offline data (e.g., demonstration trajectories), skip online collection and write a script to convert data into a `CacheEntry` list

**Minimum HDF5 schema** (if you choose the HDF5 intermediate format):

```
episode_NNNN/
  vision_0/          # [T, vision_dim] float32 — visual embedding time series
  robot_state/       # [T, state_dim] float32 — state vectors
  clean_action/      # [T, action_horizon, action_dim] float32 — action sequences
  prompt_emb/        # [T, prompt_dim] float32 (optional)
```

#### 6.3 Artifact Building

The existing `exp/cache_experiment/build_in_memory_cache_artifact.py` and `exp/cache_experiment/build_clip_cache_artifact.py` are reference scripts targeting the Pi0.5 HDF5 schema. When migrating, use their logic as reference to write your own build script.

Core flow:

```python
entries = []
for episode in episodes:
    trajectory_id = str(uuid.uuid4())
    for step_idx in range(len(episode)):
        entry = CacheEntry(
            id=f"{trajectory_id}:{step_idx}",
            checkpoint_id=CheckpointID.CP1,
            query_keys=build_query_keys(episode, step_idx),   # Your logic
            payload=CachePayload(
                action_chunk=episode.actions[step_idx],
                task_key=episode.task_name,
            ),
            step_idx=step_idx,
            prev_ids=[f"{trajectory_id}:{step_idx-1}"] if step_idx > 0 else [],
            next_ids=[f"{trajectory_id}:{step_idx+1}"] if step_idx < len(episode)-1 else [],
            trajectory_id=trajectory_id,
        )
        entries.append(entry)

# Serialize in the dict format required by InMemoryBackend.load_artifact()
import pickle

artifact = {
    "key_builder_type": "my_diffusion_policy",
    "checkpoint_id": "CP1",
    "vector_dims": {"vision_0": 512, "robot_state": 14},  # must match YAML config
    "entries": entries,
}
with open("my_artifact.pkl", "wb") as f:
    pickle.dump(artifact, f)
```

Specify the artifact path in YAML:

```yaml
backend:
  type: in_memory
  in_memory:
    preload_path: /path/to/my_artifact.pkl
```

### Step 7: Verification

#### 7.1 Unit Test: KeyBuilder

```python
def test_key_builder_output_dims():
    builder = DiffusionPolicyKeyBuilder()
    fake_output = make_fake_encoder_output()  # Your test data

    builder.collect(CheckpointID.CP1, encoder_output=fake_output)
    keys = builder.build(CheckpointID.CP1)

    assert "vision_0" in keys
    assert keys["vision_0"].shape == (512,)       # Must match vector_dims
    assert keys["vision_0"].device.type == "cpu"
    assert keys["vision_0"].dtype == torch.float32
    assert keys["vision_0"].is_contiguous()
```

#### 7.2 Integration Test: End-to-End Orchestrator

```python
def test_orchestrator_check_hit():
    # Build a backend with a known entry
    backend = InMemoryBackend(vector_dims={"vision_0": 512, "robot_state": 14})
    storage = CacheStorage(backend)
    storage.insert(known_entry)

    orchestrator = CacheOrchestrator(
        storage=storage,
        key_builder=DiffusionPolicyKeyBuilder(),
        gates={CheckpointID.CP1: AlwaysSearchGate()},
        judges={CheckpointID.CP1: AlwaysHitJudge()},
        search_strategies={CheckpointID.CP1: WeightedRrfKnnStrategy(...)},
    )

    # Check with the same input as known_entry
    result = orchestrator.check(CheckpointID.CP1, encoder_output=same_input)
    assert result.hit_type == HitType.FULL_HIT
    assert result.payload is not None
```

#### 7.3 Experiment Validation

- Run your model with cache enabled; observe cache hit rate
- Compare inference latency with and without cache
- Check cached action quality (difference from actual inference output)

---

## 4. Optional Extensions

### 4.1 Episode Write Path

By default, at the end of each episode the cache uses WritePolicy to decide whether to write the episode's inference data to storage. This allows the cache to accumulate experience over time.

Configuration:

```yaml
write_policy:
  type: on_any_miss    # Write only if there were misses (default)
  # type: always       # Write every episode
  # type: never        # Read-only mode
```

### 4.2 Trajectory Search

Single-step search only matches the current observation. Trajectory search additionally considers **whether the preceding steps also match**, favoring temporally coherent cached sequences.

Configuration:

```yaml
search_strategy:
  type: weighted_score_sum_knn    # Recommended for trajectory search
  trajectory_depth: 3              # Look back 3 steps
  trajectory_weights: [0.6, 0.3, 0.1]  # Most recent step weighted highest
```

> **Note**: Only `InMemoryBackend` supports trajectory search (`trajectory_depth > 1`).

### 4.3 CLIP Builder Details

The CLIP builder uses `open_clip`'s ViT-B-32 model to encode raw input images into 512-dimensional vectors, replacing model-internal visual embedding extraction.

**When to use**:
- You can reliably obtain raw input images but don't want to or can't extract embeddings from model internals
- You want to quickly validate the cache framework's effectiveness without investing in a custom KeyBuilder

**Configuration**:

```yaml
key_builder:
  type: clip

backend:
  type: in_memory
  vector_dims:
    vision_0: 512       # CLIP ViT-B-32 output dimension
    robot_state: 32     # Still requires state vector
```

**Artifact building**: Use `exp/cache_experiment/build_clip_cache_artifact.py` as reference. It reads raw images from HDF5, encodes them via CLIP, and builds the artifact.

### 4.4 Offline Artifact Experiments

If you only want to run offline experiments (evaluate cache hit rates without online inference):

1. Build an artifact from existing demonstration data (see Step 6)
2. Write an evaluation script: iterate over test data, call `orchestrator.check()` for each step, and collect hit rate statistics
3. No Interceptor implementation needed

---

## 5. Experimental / Incomplete Items

### CP3

CP3 in the current implementation is **used for infrastructure validation only**. It performs a check after Stage 3 completes, but there is no complete "predict whether the next step can be skipped" logic yet.

When migrating:
- You can keep the CP3 framework code (check + buffer_for_write) for future use
- Do not expect CP3 to provide actual inference speedup in the current version

### Custom Query Field / Checkpoint Extension

If the existing 5 canonical fields or 2 checkpoint IDs don't meet your needs, you'll need to modify the framework itself:

| Extension Type | Files to Modify |
|----------------|-----------------|
| New query field | `types.py` (constants), `config.py` (validation), backend `vector_dims`, related tests |
| New checkpoint | `types.py` (enum), `config.py` (checkpoint config), `orchestrator.py` (check logic), related tests |

---

## 6. YAML Config Reference

Here is a model-agnostic complete YAML template:

```yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null             # Set path to output timing CSV

keys:
  vision_0:    { enabled: true,  weight: 1.0 }
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }
  robot_state: { enabled: true,  weight: 1.0 }

key_builder:
  type: my_custom_builder          # Replace with your registered type name

checkpoints:
  _defaults: &cp_defaults
    gate:
      type: always_search
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      step_filter: all
      rrf_k: 60
      trajectory_depth: 1

  cp1:
    <<: *cp_defaults
    enabled: true
    judge:
      type: threshold
      threshold: 0.95              # Tune based on your data

  cp3:
    <<: *cp_defaults
    enabled: false                 # CP3 is experimental; recommend disabling

backend:
  type: in_memory
  vector_dims:
    vision_0: 512                  # Must match KeyBuilder output dimension
    robot_state: 14                # Must match state vector dimension
  in_memory:
    preload_path: /path/to/artifact.pkl

write_policy:
  type: on_any_miss
```

**Key adaptation points**:
- `keys`: Enable fields that your KeyBuilder actually outputs; disable the rest
- `key_builder.type`: The name you registered in `config.py`
- `backend.vector_dims`: Dimension of each enabled field, must match KeyBuilder output
- `judge.threshold`: Needs calibration based on your data; use `always_hit` initially for functional validation

---

## 7. FAQ & Pitfalls

### Different Vision Token Layout

Pi0.5's vision tokens come from SigLIP (256 tokens × 2048 dim per image, 3 images). Your model may have a completely different layout.

**Solution**: Handle it in your KeyBuilder. The only hard requirement is outputting a fixed-dimension 1D vector. You can use any pooling strategy (mean pool, max pool, spatial pool, CLS token) to reduce variable-length token sequences to a fixed dimension.

### Different Action Dimensions

Pi0.5's `action_chunk` shape is `[50, 32]` (50 steps × 32-dim actions). Your model might use `[16, 7]` or something else.

**Solution**: `CachePayload.action_chunk` has no hardcoded shape constraint. As long as your Interceptor can correctly handle the action tensor returned on hit, any shape works. However, action shapes must be consistent within a single artifact.

### Single Forward Pass Models

Some models (e.g., simple MLP policies) have no clear stage separation — just a single `forward()` call.

**Solutions**:
1. **Artificially split**: Call encoder and decoder separately, even if they were originally one `forward()`
2. **CP3 semantics only**: Don't check cache mid-inference; instead, after inference completes, judge whether the next step can be skipped. Note that CP3 is currently experimental
3. **Offline artifact experiments only**: Don't do online inference caching; just evaluate "if we had a cache, what would the hit rate be"

### vector_dims Mismatch

**Symptom**: `CacheStorage` throws dimension validation errors at `insert()` or `search()` time.

**Cause**: `backend.vector_dims` in YAML doesn't match the tensor dimensions returned by KeyBuilder `build()`, or doesn't match `query_keys` dimensions in the artifact.

**Fix**: Ensure all three are consistent — KeyBuilder output dimensions = YAML vector_dims = artifact query_keys dimensions.

### Performance Tuning

- **KeyBuilder choice**: mean pool is simplest but loses spatial information; spatial pool preserves spatial structure but higher dimensions; CLIP doesn't depend on model internals but has extra overhead
- **Search Strategy**: `weighted_rrf_knn` suits single-step search; `weighted_score_sum_knn` suits trajectory search
- **Top-k**: Default `top_k=1`; increasing it improves recall but slows search
- **Threshold**: Too high leads to almost no hits; too low leads to incorrect hits. Recommend starting with `always_hit` + offline analysis to determine appropriate thresholds
