# CLIP KeyBuilder Implementation Plan

> Status: `Plan`
> Date: 2026-04-10

## 1. Objective

Add a new KeyBuilder based on the CLIP vision encoder. Use open_clip to generate low-dimensional vectors from model input images as cache keys. Vision fields go through CLIP encoding; prompt_emb and robot_state reuse the existing `_CP1BaseKeyBuilder` logic (slice from `stage1.prefix_embs` + mean pool).

## 2. Design Overview

### 2.1 Data Flow

```
InferenceInterceptor.infer()
  │
  ├─ stage1 = run_stage1(obs)      ← prefix_embs, state (GPU tensor)
  ├─ input_images = extract_valid_images(inputs)  ← {slot: (224,224,3) uint8 numpy}
  │
  └─ orchestrator.check(CP1, stage1=stage1, input_images=input_images)
       │
       └─ CLIPKeyBuilder.collect(CP1, stage1=stage1, input_images=input_images)
            │
            ├─ Cache stage1.prefix_embs → used for prompt_emb (mean pool)
            ├─ Cache stage1.state       → used for robot_state (raw)
            └─ Cache input_images       → used for vision_0/1/2 (CLIP encode)
          
          CLIPKeyBuilder.build(CP1)
            │
            ├─ vision_0: input_images["base_0_rgb"]       → CLIP → [embed_dim]
            ├─ vision_1: input_images["left_wrist_0_rgb"] → CLIP → [embed_dim]
            ├─ vision_2: input_images["right_wrist_0_rgb"]→ CLIP → [embed_dim]
            ├─ prompt_emb: prefix_embs[768:] → mean pool  → [2048]
            └─ robot_state: state[0]                      → [32]
```

### 2.2 CLIP Image Encoding Details

open_clip API:
```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device="cuda"
)
model.eval()

# Input: uint8 numpy (224, 224, 3)
# → PIL Image → preprocess (Resize, CenterCrop, ToTensor, Normalize)
# → tensor [1, 3, 224, 224]
# → model.encode_image(tensor, normalize=True) → [1, embed_dim]

# embed_dim can be obtained from model.visual.output_dim (ViT-B-32=512, ViT-L-14=768)
```

### 2.3 Image Slot Mapping

| Image Slot | input_images Key | Output Field |
|----------|-----------------|---------|
| base cam | `base_0_rgb` | `vision_0` |
| left wrist | `left_wrist_0_rgb` | `vision_1` |
| right wrist | `right_wrist_0_rgb` | `vision_2` |

Follows the `MODEL_IMAGE_KEYS` order from `image_extract.py`.

### 2.4 enabled_fields Control

Consistent with existing builders, controlled via the constructor parameter `enabled_fields: list[str] | None`.
For example, when `enabled_fields=["vision_0", "robot_state"]`, only these two keys are output, and unnecessary CLIP encoding is skipped.

### 2.5 YAML Configuration

`KeyBuilderConfig` adds CLIP-specific fields:

```yaml
key_builder:
  type: clip                        # new type
  clip_model_name: ViT-B-32         # open_clip model name
  clip_pretrained: openai           # open_clip pretrained tag
```

Corresponding dataclass extension:

```python
@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"
    clip_model_name: str = "ViT-B-32"       # only used when type=clip
    clip_pretrained: str = "openai"          # only used when type=clip
```

### 2.6 vector_dims Configuration Examples

ViT-B-32 (embed_dim=512):
```yaml
backend:
  vector_dims:
    vision_0: 512
    robot_state: 32
```

ViT-L-14 (embed_dim=768):
```yaml
backend:
  vector_dims:
    vision_0: 768
    robot_state: 32
```

## 3. Dependency Analysis

### 3.1 New Dependency: `open-clip-torch`

PyPI package name: `open-clip-torch`
Import name: `open_clip`

Transitive dependencies of open_clip:
- `torch` — project already has `torch==2.7.1` ✅
- `torchvision` — **not currently installed in the project; open_clip needs it for image preprocessing**
- `timm` — used internally by open_clip for the vision backbone ✅ (bundled with open_clip)
- `pillow` — project already has `pillow>=11.0.0` ✅
- `huggingface_hub` — project already has it (transitive dependency of transformers) ✅

### 3.2 Conflict Risk Assessment

| Dependency | Current Version | open_clip Requirement | Conflict Risk |
|------|---------|---------------|---------|
| `torch` | ==2.7.1 | >=1.9 | **No conflict** |
| `torchvision` | Not installed | Needs installation | **Must add**; version must match torch 2.7.1 |
| `transformers` | ==4.53.2 | No direct dependency | **No conflict** |
| `pillow` | >=11.0.0 | >=8.0 | **No conflict** |
| `timm` | Not directly installed | Bundled with open_clip | **Need to verify no conflict with transformers' timm** |
| `jax` | 0.5.3 | Unrelated | **No conflict** |

### 3.3 Packages to Install

```toml
# pyproject.toml [project.dependencies] addition:
"open-clip-torch>=2.26.1",
```

`open-clip-torch` will automatically pull in `torchvision` and `timm`. Need to verify:
1. Whether `torchvision` version is compatible with `torch==2.7.1` (torch 2.7.1 corresponds to torchvision 0.22.1)
2. Whether `timm` conflicts with existing `transformers==4.53.2`

**Verification method**: Run `uv add open-clip-torch --dry-run` before installation to check dependency resolution.

### 3.4 GPU Memory Overhead

| CLIP Model | Parameters | Estimated VRAM |
|-----------|-------|---------|
| ViT-B-32 | ~88M | ~350 MB |
| ViT-L-14 | ~304M | ~900 MB |

The Pi0.5 model itself is ~3B parameters, occupying ~6-12 GB. The additional 350 MB for ViT-B-32 is acceptable.

## 4. File Change List

### 4.1 New Files

| File | Purpose |
|------|------|
| `src/openpi/cache/components/clip_key_builder.py` | CLIPKeyBuilder class implementation |
| `exp/build_clip_cache_artifact.py` | Build CLIP pkl artifact from HDF5 |

### 4.2 Modified Files

| File | Changes |
|------|---------|
| `src/openpi/cache/config.py` | (1) `KeyBuilderConfig` adds `clip_model_name`, `clip_pretrained` fields (2) `_build_key_builder()` adds `"clip"` branch (3) `_valid_key_builder_types` adds `"clip"` (4) `validate_cache_config()` adds CLIP validation: at least one vision field must be enabled |
| `scripts/serve_policy.py` | All 3 paths that create InferenceInterceptor must be covered: when `key_builder.type == "clip"`, force `collect_images = True` |
| `pyproject.toml` | Add `open-clip-torch` dependency |

### 4.3 Unchanged Files

- `key_builder.py` existing builders — no changes
- `interceptor.py` — no changes (already supports `collect_images` + `input_images` passthrough)
- `orchestrator.py` — no changes (`collect(**stage_outputs)` already passes through `input_images`)
- `in_memory_backend.py` — no changes (pkl format compatible)
- `storage_types.py` — no changes
- `data_collector.py` / `collection_policy.py` — no changes

## 5. Implementation Steps

### Phase 1: Dependency Installation and Verification

1. `uv add open-clip-torch --dry-run` to check for dependency conflicts
2. After confirming no conflicts: `uv add open-clip-torch`
3. Verify `python -c "import open_clip; print(open_clip.list_pretrained())"` works

### Phase 2: CLIPKeyBuilder Implementation

File: `src/openpi/cache/components/clip_key_builder.py`

```python
# ── Shared helpers (used by both online builder and offline artifact script) ──

def clip_prompt_key_from_tokens(prompt_tokens: torch.Tensor) -> torch.Tensor:
    """[num_tokens, emb_dim] → mean pool → [emb_dim] CPU float32."""
    return prompt_tokens.mean(dim=0).cpu().float().contiguous()

def clip_state_key(state: torch.Tensor) -> torch.Tensor:
    """[state_dim] → [state_dim] CPU float32."""
    return state.cpu().float().contiguous()


# ── CLIPKeyBuilder ──

class CLIPKeyBuilder:
    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        enabled_fields: list[str] | None = None,
    ):
        # Only store parameters, do not load model (lazy init)
        self._clip_model_name = clip_model_name
        self._clip_pretrained = clip_pretrained
        self._enabled = set(enabled_fields) if enabled_fields else None
        # CLIP model and preprocess, deferred to first collect() call
        self._clip_model = None
        self._preprocess = None
        self._device = None      # inferred from stage1 tensor
        self._embed_dim = None   # model.visual.output_dim

    def _ensure_model_loaded(self, device: torch.device):
        """Lazy init: infer device from stage1 tensor on first call and load CLIP."""
        if self._clip_model is not None:
            return
        import open_clip
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name, pretrained=self._clip_pretrained, device=device,
        )
        model.eval()
        self._clip_model = model
        self._preprocess = preprocess
        self._embed_dim = model.visual.output_dim

    def collect(self, checkpoint_id, **stage_outputs):
        # Cache prefix_embs, state from stage_outputs["stage1"] → self._cache (GPU tensor)
        # On first call, infer device from stage1.prefix_embs.device, lazy load CLIP model
        if "stage1" in stage_outputs:
            self._ensure_model_loaded(stage_outputs["stage1"].prefix_embs.device)
        # Cache image dict from stage_outputs["input_images"] → self._images (private, CPU numpy)

    def build(self, checkpoint_id) -> dict[str, torch.Tensor]:
        # vision_*: iterate over enabled vision fields, fetch corresponding images from self._images
        #           if image is missing (image_mask=False), skip that field (do not output zero vector)
        #           if image exists, CLIP encode → [embed_dim] CPU float32
        # prompt_emb: clip_prompt_key_from_tokens(prefix_embs prompt segment)
        # robot_state: clip_state_key(state[0])

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        # Only expose GPU tensors (prefix_embs, state), not self._images
        return self._cache

    def clear(self):
        self._cache.clear()
        self._images = None
```

Key implementation details:
- **Device inference**: Do not pass device in the constructor. Infer from `stage1.prefix_embs.device` on the first `collect()` call, lazy load the CLIP model. Reuse on subsequent calls.
- Image preprocessing: numpy uint8 → PIL Image → preprocess transform → GPU tensor
- Batch encoding: if multiple vision fields are enabled, stack into a batch for a single forward pass
- L2 normalize: use `model.encode_image(batch, normalize=True)`
- **Shared helpers**: `clip_prompt_key_from_tokens()` and `clip_state_key()` are called by both the online builder and the offline artifact script, avoiding logic divergence
- `cached_data` only contains GPU tensors (prefix_embs, state); `input_images` is stored in the private field `self._images`, not exposed to Gate/Judge
- Missing image slots: `build()` skips that vision field, does not produce a zero vector (compatible with InMemoryBackend fusion logic)

### Phase 3: Config Integration (Phase B, can be deferred until offline evaluation passes)

Modify `config.py`:
1. Add `clip_model_name` and `clip_pretrained` fields to `KeyBuilderConfig`
2. Add `"clip"` branch to `_build_key_builder()`, passing model name + pretrained + enabled_fields (no device — builder does lazy init)
3. Add `"clip"` to `_valid_key_builder_types`
4. `validate_cache_config()`: clip builder requires at least one vision field to be enabled (otherwise pointless)
5. Artifact compatibility check: do not change `_build_backend()` signature (it only takes `BackendConfig`). Move the check to the caller level — in `build_cache_components()` and `build_shared_storage()`, after `_build_backend()` returns but before the function returns, add a `_validate_artifact_metadata(backend, config.key_builder)` call. This function reads loaded artifact metadata (`clip_model_name`, `clip_pretrained`), compares with `KeyBuilderConfig`, and fails fast on mismatch. `build_per_connection_components()` does not need this check (it reuses already-validated shared_storage). No changes to `InMemoryBackend.load_artifact()` or `_build_backend()` interfaces.

Modify `serve_policy.py`:
6. **All** paths that create `InferenceInterceptor` (currently 3 locations: ~L201, L243, L251) must be covered: after reading the cache config and before creating the interceptor, uniformly check `if cache_config.key_builder.type == "clip": collect_images = True`. Do not only modify one branch.

### Phase 4: pkl Artifact Build Script (Phase A, implement first)

File: `exp/build_clip_cache_artifact.py`

**Does not reuse** `_build_fake_stage1()`. Instead reads directly from HDF5 from three sources:
- Images: read uint8 numpy from `step_xxxx/input_images/{base_0_rgb, ...}` → CLIP encode
- prompt_emb: read from `step_xxxx/prompt_emb` → `clip_prompt_key_from_tokens()` (shared helper) → [2048]
- robot_state: read directly from `step_xxxx/robot_state` → `clip_state_key()` (shared helper) → [32]

**Prompt/state logic sharing**: The offline script and online CLIPKeyBuilder call the same set of helper functions (`clip_prompt_key_from_tokens`, `clip_state_key`), avoiding two divergent implementations.

**Single-process GPU batch encoding** (no ProcessPoolExecutor):
1. Main process loads CLIP model to GPU once
2. Sequentially iterate HDF5 files, skip files without `input_images` group (warn and report available/unavailable file counts at the end)
3. Collect images, encode in batches (e.g., batch_size=64)
4. prompt_emb / robot_state read directly from HDF5

**Artifact metadata extension**: In addition to existing `key_builder_type`, `checkpoint_id`, `vector_dims`, `entries`, add:
- `clip_model_name`: CLIP model name (e.g., `"ViT-B-32"`)
- `clip_pretrained`: pretrained tag (e.g., `"openai"`)

CLI:
```bash
uv run exp/build_clip_cache_artifact.py \
    --data-dir data/libero_spatial \
    --clip-model ViT-B-32 \
    --clip-pretrained openai \
    --output data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl
```

### Phase 5: Testing

1. `test_clip_key_builder.py`: test with mock CLIP model:
   - Output dimension correctness
   - Missing image skipping (no zero vectors)
   - `enabled_fields` filtering
   - `cached_data` only contains tensors, not numpy images
2. `test_config_clip.py`: test config parsing:
   - `"clip"` type parses correctly
   - Validation error when no vision field is enabled
   - `clip_model_name` / `clip_pretrained` correctly passed through
3. Artifact script test: run build with small test HDF5, verify output format and metadata

### Phase 6: End-to-End Verification

1. Build artifact: run `build_clip_cache_artifact.py` with existing HDF5 data
2. Load verification: `InMemoryBackend.load_artifact()` loads the pkl
3. Offline retrieval evaluation: compare retrieval quality of CLIP artifact vs existing cp1_mean_pool artifact
4. (Phase B) Config run: start `serve_policy.py --cache_config clip_cache.yaml` with new YAML config
5. Dimension check: confirm vector_dims matches CLIP output

## 6. Complete YAML Configuration Example (ViT-B-32 + vision_0 + robot_state)

```yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000

keys:
  vision_0:    { enabled: true,  weight: 1.0 }
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }
  robot_state: { enabled: true,  weight: 1.0 }

key_builder:
  type: clip
  clip_model_name: ViT-B-32       # can switch to ViT-L-14, etc.
  clip_pretrained: openai          # can switch to laion2b_s34b_b79k, etc.

checkpoints:
  cp1:
    enabled: true
    judge:
      type: always_hit
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1

backend:
  type: in_memory
  vector_dims:
    vision_0: 512      # ViT-B-32 = 512, ViT-L-14 = 768
    robot_state: 32
  in_memory:
    preload_path: data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl

write_policy:
  type: never
```

## 7. Implementation Phase Division

| Phase | Includes | Goal | Prerequisites |
|------|-----------|------|---------|
| **Phase A (Offline)** | 1, 2, 4, 5 | Dependency install + CLIPKeyBuilder + artifact script + tests + offline evaluation | None |
| **Phase B (Online)** | 3, 6.4-6.5 | config.py integration + serve_policy.py hookup + end-to-end verification | Phase A offline evaluation results are satisfactory |

Phase B can proceed after Phase A verification results are satisfactory.

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|------|---------|
| `open-clip-torch` conflicts with existing dependencies | Installation failure | Phase 1: dry-run verification first |
| CLIP additional GPU VRAM | OOM | Default to ViT-B-32 (350MB); can run CLIP on CPU if constrained |
| CLIP inference latency (3 images ~10-15ms) | Increases cache check time | Only encode enabled vision fields; batch forward |
| HDF5 missing input_images | Artifact build skips that file | Script auto-detects and reports available/unavailable file statistics |
| Preprocess resize redundant for already 224x224 images | Minor latency | Images are already 224x224, resize is effectively a no-op |
| Same-dimension but different CLIP weights are semantically incompatible | Silent retrieval quality degradation | Artifact metadata records clip_model_name + clip_pretrained; config.py does automatic comparison at startup for fail-fast |

---

## Review (Historical Record — Conclusions Incorporated into Plan Above)

> Review date: 2026-04-10
> Status: `Historical` — all adopted changes are reflected in Sections 1-8 above; content below is retained as review record only.
> Review scope: `CLAUDE.md`, `docs/cache_system_architecture_chinese.md`, `src/openpi/cache/`, `src/openpi/collect/`, `exp/build_in_memory_cache_artifact.py`

### 1. Major Concerns

#### Concern 1: Introducing a separate CLIP encoding at runtime could directly negate CP1 cache benefits

- Reason:
  - CP1 currently fires after `run_stage1()`, where SigLIP computation cost is already paid.
  - Existing `cp1_*` builders all reuse `stage1.prefix_embs` for pooling, introducing virtually no additional model computation.
  - This proposal adds a new `open_clip` image encoding forward pass on both miss and hit paths — this is new computation, not "reusing existing intermediate results."
- Suggestion:
  - Gather baseline measurements before deciding whether this is worthwhile. At minimum compare:
    1. `cp1_mean_pool` build time
    2. CLIP encode time
    3. Search time saved from lower-dimensional retrieval
    4. Whether hit rate improvement is sufficient to cover the new encoding cost
- Question:
  - Is the goal "improve hit quality" or "reduce total latency"?
  - If CLIP encoding itself takes several to tens of milliseconds, does this tradeoff still hold?

#### Concern 2: The proposal treats `collect_images=True` as a config-validatable item, but it is not part of `CacheConfig`

- Reason:
  - `collect_images` is currently a CLI parameter in `scripts/serve_policy.py`, not in the dataclass tree in `src/openpi/cache/config.py`.
  - Therefore `validate_cache_config()` cannot know whether the user passed `--collect_images` at service startup.
  - The plan states "validate in `validate_cache_config()` that clip builder requires `collect_images=True`" — this is inconsistent with the current interface.
- Suggestion:
  - Choose one:
    1. Formally incorporate "whether input_images is needed" into cache config; or
    2. Do the startup-time fail-fast check in `serve_policy.py`, not in `validate_cache_config()`.
- Question:
  - Should this constraint belong to the YAML layer or the server startup parameter layer?

#### Concern 3: Artifact build depends on `input_images` in HDF5, but existing collected data may not have this group

- Reason:
  - `src/openpi/collect/data_collector.py` only writes `step_xxxx/input_images/...` when `input_images` is non-empty.
  - `docs/data_collection_guide.md` current main documentation also does not list `input_images` as a standard output field in the main workflow description.
  - This means historical HDF5 files likely only have `vision_*` / `prompt_emb` / `robot_state`, without raw images.
- Suggestion:
  - Explicitly state the data prerequisite in the plan:
    - Only support newly collected HDF5 with `input_images`; or
    - Provide a fallback strategy / inspection script to pre-scan which files are usable for CLIP artifact.
- Question:
  - Do you plan to reuse existing historical data, or accept re-collecting a batch of data with `input_images`?

#### Concern 4: Directly copying the multi-process pattern from `exp/build_in_memory_cache_artifact.py` will easily cause resource issues for the CLIP version

- Reason:
  - The existing script creates a builder within each worker process.
  - If the CLIP builder loads a GPU model within the worker, multiple processes will duplicate VRAM usage — this will almost certainly crash.
  - Even on CPU, multiple processes loading the model repeatedly incurs high model initialization and memory overhead.
- Suggestion:
  - Do not directly reuse the current "one builder per process" pattern.
  - A more stable approach:
    1. Single-process GPU batch encoding;
    2. Or multi-process for HDF5 reading only, with encoding centralized in a single process;
    3. At minimum, design the CLIP artifact script to work single-process first, then discuss parallelization.
- Question:
  - Is the priority for artifact building "fast" or "stable"?
  - Is the machine single-GPU by default, or do you plan CPU-only building?

#### Concern 5: `vector_dims` alone is insufficient to identify whether artifact and runtime builder truly match

- Reason:
  - `src/openpi/cache/backends/in_memory_backend.py`'s `load_artifact()` only validates `vector_dims`.
  - But different CLIP models / different pretrained tags may output same-dimension vectors, e.g., both 512-dimensional but in different semantic spaces.
  - This leads to "dimension is valid but semantics are incompatible" silent errors.
- Suggestion:
  - Artifact metadata should at minimum add:
    - `clip_model_name`
    - `clip_pretrained`
    - `enabled_fields`
    - Possibly a `builder_config_hash`
  - Do strict matching on load, not just dimension comparison.
- Question:
  - Should an artifact be reusable across different pretrained tags?
  - If not, this should be prohibited at the format level.

#### Concern 6: The `cached_data` interface contract currently assumes tensors; mixing in `numpy image` would break protocol boundaries

- Reason:
  - The `QueryKeyBuilder` Protocol's `cached_data` type and semantics are written as "tensors on original device, for gate/judge to read."
  - The plan's `collect()` caches both `prefix_embs/state` and `input_images`, where `input_images` is CPU numpy.
  - This would make `cached_data` a mixed container with "both GPU tensors and CPU numpy," inconsistent with the current protocol.
- Suggestion:
  - Store `input_images` in the builder's private cache, let only `build()` consume it.
  - Keep `cached_data` exposing only tensor data that gate/judge might read, avoiding protocol pollution.
- Question:
  - Do you intend to extend the `QueryKeyBuilder` protocol, or keep the gate/judge perspective unchanged?

#### Concern 7: The semantics for missing camera slots are not clearly defined, especially `vision_2`

- Reason:
  - `src/openpi/shared/image_extract.py` filters by `image_mask`, returning only valid images.
  - Existing `prefix_embs`-based builders produce `vision_0/1/2` via fixed slicing — even if a slot is perpetually empty in the environment, it has a fixed positional semantics.
  - For the CLIP version, if `right_wrist_0_rgb` does not exist, whether to output a zero vector, skip that field, or error at startup is not defined in the plan.
- Suggestion:
  - Lock down the strategy in the plan; do not decide ad-hoc during implementation.
  - Preferred approach:
    - Both query and artifact uniformly use "do not produce the field if missing";
    - Also require the config layer not to assign critical weight to perpetually missing slots.
- Question:
  - Does the target environment stably have 3 camera slots?
  - If LIBERO commonly lacks `vision_2`, should it still be included in the default example config?

#### Concern 8: Pretrained weight acquisition method is not specified; first run may fail outright on server or experiment machines

- Reason:
  - `open_clip.create_model_and_transforms(..., pretrained=...)` typically relies on local cache or online weight download.
  - The current plan only discusses pip dependencies, not model weight source, cache directory, or offline environment strategy.
  - This repo is used in many scenarios like remote inference, containers, clusters, or air-gapped environments — this cannot be left unspecified.
- Suggestion:
  - Add to the plan:
    - Whether internet access is allowed for first download
    - Where weight cache is stored
    - How to pre-stage weights for offline machines
    - Error messages and guidance on failure
- Question:
  - Does the deployment environment guarantee access to open_clip's weight source?

### 2. Additional Suggestions

#### Suggestion 1: Split the CLIP proposal into "offline artifact experiment" and "online inference integration" stages first

- Reason:
  - The biggest unknown right now is not "can we wire this into config" but "is it worth running online."
  - Doing the artifact and retrieval evaluation offline first answers the hit quality question faster.
- Suggestion:
  - Phase A: Only do offline artifact + retrieval experiment, do not touch `serve_policy.py`
  - Phase B: Only consider online key_builder when the quality benefit is clear

#### Suggestion 2: Clearly define the CLIP builder's responsibility boundary; do not continue reusing "fake stage1 reconstruction" as a long-term interface

- Reason:
  - The existing `_build_fake_stage1()` serves the reuse of `cp1_*` builders.
  - What the CLIP approach actually depends on is "prompt/state + raw input_images," not "reconstructing the full SigLIP prefix_embs."
- Suggestion:
  - Short-term reuse of old logic is fine, but the cleaner long-term approach is to prepare structured input for the artifact build script separately, rather than patching together a fake `Stage1Output`.

#### Suggestion 3: Add the test plan to this document; do not only list implementation steps

- Reason:
  - This repo already explicitly labels the cache subsystem as "high-risk, insufficiently integration-tested."
  - Adding an external vision model without tests makes it easy to defer problem discovery to the live service stage.
- Suggestion:
  - Add at least 4 types of tests:
    1. `config.py` clip branch and validation branch
    2. `clip_key_builder.py` dimensions, missing images, enabled_fields behavior
    3. Artifact script failure messages for missing `input_images`
    4. `serve_policy.py` startup fail-fast for missing `--collect_images`

### 3. Key Questions That Need Answering First

1. Is the primary goal of this proposal "retrieval quality improvement" or "end-to-end latency reduction"?
2. Has the cost of adding one CLIP encoding during online inference been roughly measured?
3. How many HDF5 files in the target dataset actually contain `input_images`?
4. Should the `collect_images` prerequisite ultimately be managed at the YAML or CLI layer?
5. Should artifacts be allowed to mix same-dimension but different CLIP weights?
6. What is the unified semantics for missing image slots: skip field, zero vector, or error?

### 4. Conclusion

The direction of this plan is viable, but I do not recommend starting implementation as-is. The two biggest risks are not "how to write the code" but:

1. Whether online CLIP encoding will cancel out cache benefits.
2. Whether existing data and config interfaces truly satisfy the `input_images` dependency.

If these two points are not nailed down first, the resulting implementation may easily become "code runs, but system benefit and usage path are both unstable."

---

## Review Response (Historical Record — Conclusions Incorporated into Plan Above)

> Response date: 2026-04-10
> Status: `Historical` — adopt/reject decisions are reflected in Sections 1-8 above; content below is retained as decision record only.
> Respondent: Claude (plan author)

### Point-by-Point Response to Major Concerns

#### Concern 1: CLIP encoding will negate CP1 benefits — **Partially agree, but does not block implementation**

The reviewer's observation is correct: existing `cp1_*` builders have zero additional model computation (pure tensor slice + pool), and CLIP introduces a new forward pass.

However, the reviewer conflates two things:
1. **The goal of this proposal is not "reduce total latency" but "use better semantic representations to improve retrieval quality."** The existing SigLIP prefix_embs mean pool is a coarse compression into 2048 dimensions; CLIP's 512-dimensional embedding is a purpose-trained image-level semantic representation, naturally more suitable for retrieval matching.
2. **CLIP encoding cost should be compared against Stage 2 + Stage 3 savings, not against build time.** On a CP1 hit, we skip Stage 2 (LLM, ~50-100ms) + Stage 3 (flow matching, ~30-50ms). ViT-B/32 encoding 3 images takes ~10-15ms — this tradeoff is positive when hit rate is sufficient.
3. The reviewer suggests "gather baseline measurements first" — this does not need to block implementation. Code implementation and performance evaluation can proceed in parallel, and the performance evaluation itself requires a usable CLIP artifact first.

**Conclusion**: Agree that performance evaluation is needed, but it is part of Phase 5 verification and does not need to be a prerequisite. Implementing the offline artifact build naturally enables evaluation.

#### Concern 2: `collect_images` is not in CacheConfig — **Agree, adopt approach 2**

The reviewer is correct that `collect_images` is a CLI parameter in `serve_policy.py`, not in YAML.

**Adopted approach**: In `serve_policy.py`, when `key_builder.type == "clip"`, **unconditionally force** `collect_images = True`. The overhead of `extract_valid_images()` is negligible (just slicing numpy from existing transform output by mask, no additional computation), so neither manual CLI parameter nor `validate_cache_config()` validation is needed.

**Plan modification**: In Phase 3 (config integration), add an automatic force logic line in the cache startup branch of `serve_policy.py`.

#### Concern 3: Historical HDF5 may lack `input_images` — **Agree, but limited impact**

The reviewer's factual observation is correct: whether historical HDF5 has `input_images` depends on whether it was enabled during collection.

But this does not affect the plan:
1. The artifact build script should naturally check whether `input_images` group exists when reading, skip if absent, and log a warning.
2. This is a data-layer prerequisite, not a code design issue. Users need data with `input_images` to build the artifact.

**Plan modification**: In Phase 4, explicitly state: the artifact script checks for `input_images` group in HDF5 during processing, skips if missing, counts skipped files, and reports available/unavailable file ratio at the end.

#### Concern 4: Multi-process CLIP model loading will crash VRAM — **Agree, adopt single-process approach**

This is a good catch. The existing `ProcessPoolExecutor` pattern is indeed unsuitable for builders that load GPU models.

**Adopted approach**: Artifact build script uses single-process design:
1. Main process loads CLIP model once
2. Sequentially iterate HDF5 files, batch-collect images
3. Encode in batches (e.g., batch_size=64)
4. prompt_emb / robot_state still read directly from HDF5 (no CLIP needed)

This is simpler and more stable than multi-process. Speed is sufficient when data volume is not large.

**Plan modification**: Phase 4 rewritten as single-process + GPU batch encoding.

#### Concern 5: `vector_dims` is insufficient to identify artifact match — **Partially agree, minor change**

The reviewer's concern is valid: same 512 dimensions but different pretrained CLIP models are indeed semantically incompatible.

However, the reviewer's suggestion of `builder_config_hash` is over-engineering. The existing artifact format already has a `key_builder_type` field.

**Adopted approach**: Add `clip_model_name` and `clip_pretrained` fields to artifact metadata. Do not change `load_artifact()` (keep it only validating `vector_dims`, since the backend should not know builder details), but the artifact build script output will include these fields for human confirmation. If automatic validation is needed later, it can be done in `config.py`'s startup logic by reading artifact metadata for matching.

**Plan modification**: In Phase 4, add `clip_model_name` and `clip_pretrained` fields to artifact dict.

#### Concern 6: `cached_data` mixed with numpy breaks protocol — **Agree, adopt isolation approach**

The reviewer is completely correct. `cached_data`'s contract is GPU tensors for Gate/Judge to read. `input_images` is CPU numpy and should not be exposed in `cached_data`.

**Adopted approach**: `input_images` is stored in the builder's private field `self._images`; `cached_data` only exposes `prefix_embs` and `state` (consistent with existing builders). `build()` takes images from `self._images` for CLIP encoding.

**Plan modification**: In Phase 2, explicitly state that `cached_data` only contains tensors; `_images` is a private field.

#### Concern 7: Missing camera slot semantics — **Agree, strategy defined**

The reviewer is correct that this needs to be pre-defined.

**Strategy**: Missing means do not produce that field. Specifically:
- `extract_valid_images()` only returns slots with `image_mask=True`
- In `build()`, if an enabled vision field's corresponding image is not in `input_images`, **skip that field** (do not output zero vector)
- This is compatible with InMemoryBackend's search logic: missing fields do not contribute to the fusion score for that candidate
- LIBERO environments indeed commonly lack `vision_2` (right_wrist), so the example config defaults to `vision_2: {enabled: false}`

**Plan modification**: In Phase 2, explicitly state: missing image = skip field, do not produce zero vector.

#### Concern 8: Pretrained weight acquisition — **Rejected, not this plan's responsibility**

The reviewer attributes a generic offline deployment strategy issue to this plan. open_clip's weight download behavior is identical to the project's existing HuggingFace transformers: download on first internet access, read from `~/.cache/huggingface/hub` cache thereafter. The project already depends on `transformers==4.53.2`, and Pi0.5 model weights themselves need downloading from HuggingFace. This is not a new problem introduced by CLIP.

If the deployment environment has no internet access, a project-level offline weight strategy is needed (pre-download all models to shared storage), not something for the CLIP builder documentation.

**Conclusion**: No plan modification. If offline deployment needs arise later, handle in a separate issue.

### Response to Additional Suggestions

#### Suggestion 1: Split into offline experiment + online integration stages — **Partially adopted**

Agree on priority: offline artifact + retrieval evaluation > online inference integration. But the plan itself is already in this order (Phase 4 artifact build can run independently before Phase 3 config integration).

**Adjustment**: Reorder phases more explicitly into two stages:
- Phase A (Offline): Dependency install → CLIPKeyBuilder implementation → artifact build script → build artifact and evaluate
- Phase B (Online): config.py integration → serve_policy.py hookup → end-to-end verification

Phase B can proceed after Phase A verification results are satisfactory.

#### Suggestion 2: Do not reuse `_build_fake_stage1()` — **Agree**

The CLIP builder needs `input_images` + `prompt_emb` + `robot_state`, not a reconstructed `prefix_embs`. The artifact build script should read these three sources directly from HDF5, without going through fake stage1.

**Modification**: Artifact script does not call `_build_fake_stage1()`. Instead:
- Images: read from `step_xxxx/input_images/`
- prompt_emb: read from `step_xxxx/prompt_emb` → reconstruct prefix_embs prompt segment → mean pool (or directly mean pool from HDF5 flat embedding)
- robot_state: read directly from `step_xxxx/robot_state`

#### Suggestion 3: Add test plan — **Agree, added**

**New Phase A.5 (Testing)**:
1. `test_clip_key_builder.py`: test with mock CLIP model for dimensions, missing image skipping, enabled_fields filtering, cached_data only contains tensors
2. `test_config_clip.py`: test `"clip"` type config parsing, validation (error when no vision field enabled, vector_dims matching)
3. Artifact script test: run artifact build with small test HDF5, verify output format

---

## Plan Revision Summary (Historical Record)

Based on review, the following changes have been incorporated into the plan text in Sections 1-8:

| Change | Source | Content |
|--------|------|------|
| `collect_images` auto-inference | Concern 2 | `serve_policy.py` automatically enables `collect_images` for clip builder |
| HDF5 missing image handling | Concern 3 | Artifact script skips files without `input_images`, reports statistics |
| Single-process artifact build | Concern 4 | No ProcessPoolExecutor; single-process GPU batch instead |
| Artifact metadata extension | Concern 5 | Add `clip_model_name`, `clip_pretrained` fields |
| `cached_data` isolation | Concern 6 | `input_images` in private field; `cached_data` only contains tensors |
| Missing image semantics | Concern 7 | Missing image = skip field, do not produce zero vector |
| Two-phase split | Suggestion 1 | Phase A offline first, Phase B online can be deferred |
| Do not reuse fake stage1 | Suggestion 2 | Artifact script reads HDF5 fields directly |
| Add test plan | Suggestion 3 | New Phase A.5 |

The following review items were **rejected**:

| Rejected Item | Source | Reason |
|--------|------|------|
| Block implementation pending baseline measurement | Concern 1 | Performance evaluation requires artifact first; implementation and evaluation can run in parallel |
| Pretrained weight offline strategy | Concern 8 | Generic issue, not this plan's responsibility; consistent with existing transformers weight management |
| `builder_config_hash` | Concern 5 | Over-engineering; metadata fields are sufficient |
