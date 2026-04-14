# Raw Image Collection Feature Design Plan

**Status**: Plan  
**Date**: 2026-04-09  
**Scope**: `--collect` system only. Zero changes to the cache framework.

---

## 1. Requirements Analysis

### Goal
During `--collect` data collection, save each step's **model input images** (post-transform RGB uint8 pixels, 224x224), rather than only saving embeddings.

### Scope Definition
- **What is saved**: The 3 canonical image slots after `input_transform` (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`) where `image_mask=True`.
- **What is NOT saved**: Not the environment's native-resolution images, nor images from all cameras in the environment. For example, ALOHA's `cam_low` does not enter the model input (see `aloha_policy.py:49-76`), so it will not be saved.
- **Adaptation logic**: Padding images are filtered via `image_mask`. Libero saves 2 images (`right_wrist_0_rgb` mask=False), ALOHA saves 3 images. This is not "automatically adapt to any camera" but rather "automatically filter padding slots in the model input".

### Constraints
- **Only modify the `--collect` system**: Two files under `src/openpi/collect/` + one shared utility function
- **Zero changes to the cache framework**: KeyBuilder does not currently need images; no data is added to `cached_data`. If future extension is needed, it can be passed through via `**stage_outputs` kwargs (a one-line change, see Appendix A)
- **Low overhead**: Image extraction is only performed when `--collect` is enabled

---

## 2. Implementation Plan

### 2.1 Data Flow

```
CollectionPolicy.infer(obs)
  │
  ├─ _extract_obs_fields(obs)  ←  single transform call, extracts both:
  │   ├─ robot_state (float32)
  │   └─ input_images: {"base_0_rgb": (224,224,3) uint8, ...}
  │       only includes slots where image_mask=True
  │
  ├─ register forward hooks → call self._policy.infer(obs) → collect embeddings
  │
  └─ record_inference(InferenceEmbeddings)  ← with input_images attached
        │
        └─ EpisodeDataCollector buffers in memory
              │
              └─ on_episode_end() → writes to HDF5, including raw_images/ group
```

### 2.2 File Changes

#### `src/openpi/collect/data_collector.py`

**InferenceEmbeddings** new field:

```python
@dataclass
class InferenceEmbeddings:
    vision_embs: list[np.ndarray]
    prompt_emb: np.ndarray
    robot_state: np.ndarray
    noise_action_steps: list[np.ndarray]
    clean_action: np.ndarray
    # ── new ──
    input_images: dict[str, np.ndarray] | None = None
    # {"base_0_rgb": (224,224,3) uint8, "left_wrist_0_rgb": ...}
    # only includes slots where image_mask=True, None means image collection not enabled
```

**HDF5 write logic** addition (in the step loop of `on_episode_end()`):

```python
if embs.input_images:
    img_grp = grp.create_group("input_images")
    for key, img in embs.input_images.items():
        img_grp.create_dataset(key, data=img, compression="lzf")
```

#### `src/openpi/collect/collection_policy.py`

**Merge transform calls**, replacing existing `_extract_robot_state()`:

```python
def _extract_obs_fields(self, obs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Single transform call to extract both robot_state and valid images.
    
    Returns:
        robot_state: (state_dim,) float32
        input_images: {slot_name: (H,W,3) uint8} only mask=True slots
    """
    inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
    
    robot_state = np.asarray(inputs["state"], dtype=np.float32).flatten()
    input_images = extract_valid_images(inputs)
    
    return robot_state, input_images
```

Call site modification:

```python
# old:
robot_state_np = self._extract_robot_state(obs)
# new:
robot_state_np, input_images = self._extract_obs_fields(obs)
```

`_record()` call also passes `input_images` accordingly.

#### `src/openpi/shared/image_utils.py` (new file)

```python
"""Image extraction utility — filters valid images from input_transform output."""

import numpy as np

# The model's 3 fixed image slots (see model.py:40-42)
MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def extract_valid_images(
    inputs: dict,
    image_keys: tuple[str, ...] = MODEL_IMAGE_KEYS,
) -> dict[str, np.ndarray]:
    """Extract images with image_mask=True from input_transform output.
    
    Note: Returns post-transform images (224x224), not environment native resolution.
    Cameras that do not enter the model input (e.g., ALOHA's cam_low) will not be extracted.
    
    Args:
        inputs: output of input_transform, containing "image" and "image_mask"
        image_keys: image slots to check
    
    Returns:
        {slot_name: (H, W, 3) uint8 ndarray} valid slots only
    """
    images = inputs.get("image", {})
    masks = inputs.get("image_mask", {})
    result = {}
    for key in image_keys:
        if key in images and bool(masks.get(key, False)):
            img = np.asarray(images[key])
            if np.issubdtype(img.dtype, np.floating):
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            result[key] = img
    return result
```

---

## 3. HDF5 Schema Changes

### Before (existing)

```
episode_XXXX.h5
├── attrs: experiment_name, task, episode_id, num_steps, timestamp, success
├── step_0000/
│   ├── vision_0          (float16, embedding)
│   ├── vision_1          (float16, embedding)
│   ├── vision_2          (float16, embedding)
│   ├── prompt_emb        (float16, embedding)
│   ├── robot_state       (float32)
│   ├── noise_action_1..N (float32)
│   └── clean_action      (float32)
└── step_0001/ ...
```

### After

```
episode_XXXX.h5
├── attrs: experiment_name, task, episode_id, num_steps, timestamp, success
├── step_0000/
│   ├── vision_0          (float16, embedding)
│   ├── ...
│   ├── clean_action      (float32)
│   └── input_images/                    ← new group
│       ├── base_0_rgb       (uint8, 224x224x3, lzf compressed)
│       └── left_wrist_0_rgb (uint8, 224x224x3, lzf compressed)
│       # right_wrist_0_rgb absent (mask=False under Libero)
└── step_0001/ ...
```

### Per-Step Size Estimate

| Data | Size |
|------|------|
| Existing embeddings + actions | ~2.5 MB (float16 vision x 3 + prompt + state + actions) |
| New images (Libero, 2 images) | ~295 KB raw, ~200 KB after lzf compression |
| New images (ALOHA, 3 images) | ~442 KB raw, ~300 KB after lzf compression |

Images account for < 15% of total size; overhead is acceptable.

---

## 4. What Is NOT Changed

| Component | Changed | Notes |
|-----------|---------|-------|
| `collection_policy.py` | Yes | Merge transform, extract images |
| `data_collector.py` | Yes | New field + HDF5 write logic |
| `shared/image_utils.py` | Yes (new) | Shared utility function |
| Entire cache framework | No | KeyBuilder / Orchestrator / Storage / Gate / Judge — zero changes |
| `serve_policy.py` | No | `--collect` already exists, no new CLI arguments needed |
| Model code | No | |
| Client code | No | |

---

## 5. Implementation Steps

### Phase 1: Utility Function
1. Create `src/openpi/shared/image_utils.py`, implement `extract_valid_images()`

### Phase 2: Data Collection Changes
1. `data_collector.py`: Add `input_images` field to `InferenceEmbeddings`
2. `data_collector.py`: Add `input_images/` group to HDF5 write logic
3. `collection_policy.py`: `_extract_robot_state()` -> `_extract_obs_fields()`, merge transform
4. `collection_policy.py`: Pass `input_images` to `_record()`

### Phase 3: Documentation Updates
1. Update `docs/data_collection_guide.md`: HDF5 schema, new field description, size estimates
2. Update `logs/README.md` index

### Phase 4: Verification
1. Unit tests: `extract_valid_images()` mask filtering for Libero (2 images) and ALOHA (3 images)
2. Integration tests: HDF5 written with `--collect` contains `input_images/` group with correct key count
3. Regression tests: Behavior unchanged when `--collect` is not enabled

---

## Appendix A: Future Cache Framework Extension Path

If KeyBuilder needs images in the future, the extension approach is as follows (not implemented now):

```python
# interceptor.py — pass one more kwarg
input_images = extract_valid_images(inputs)
orchestrator.check(CP1, stage1=stage1, input_images=input_images)

# orchestrator.py — zero changes, **stage_outputs passes through automatically
# key_builder.collect(CP1, stage1=stage1, input_images=input_images)

# new KeyBuilder subclass — reads as needed
def collect(self, checkpoint_id, **stage_outputs):
    super().collect(checkpoint_id, **stage_outputs)
    if "input_images" in stage_outputs:
        self._cache["input_images"] = stage_outputs["input_images"]
```

**Considerations** (to be addressed at that time):
- `cached_data`'s current type contract is `dict[str, torch.Tensor]` (GPU tensors). Mixing in numpy images would break the semantics. Solutions: either add a separate `cached_images` property, or relax the protocol to `dict[str, torch.Tensor | np.ndarray]` with documentation.
- If Gate/Judge need to read images, they must explicitly handle numpy types.
