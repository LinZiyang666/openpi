# CP1 LLM Layer Extract KeyBuilder Guide

> **Prerequisite**: read [tutorial.md](tutorial.md) §4 for KeyBuilder component basics, §10 for YAML configuration.
>
> **Design document**: full design and decisions in [`logs/cp1_llm_layer_extract_key_builder_plan.log.md`](../../logs/archive/cp1_llm_layer_extract_key_builder_plan.log.md) (Plan, G1 APPROVED).

---

## 1. Overview

`CP1LLMLayerExtractKeyBuilder` is a two-stage KeyBuilder. It independently runs the first `N+1` layers of the PaliGemma backbone inside the KeyBuilder (borrowing layer references from the model without modifying Stage 2), using the layer-N hidden state as the source data for the cache key:

```
Stage1Output                                   → KeyBuilder
   │                                              ┌─ Step A: LLMLayerExtractor (borrows layer 0..N + rotary_emb)
   ▼                                              │      runs N+1 layers forward (no_grad, no KV cache)
prefix_embs [1, 968, 2048]                        │      → LLMLayerExtractResult(hidden, mask, segments, layer)
prefix_pad_masks / prefix_position_ids /          │
  prefix_att_2d_masks_4d                          ▼
                                              ┌─ Step B: PrefixReducer (pluggable)
                                              │      prefix_mean_pool / per_modality_mean_pool
                                              │      → {field: [2048]}
                                              ▼
                                          {vision_0: …, robot_state: …} → CPU float32
```

**Core idea**: after one (or several) layers of prefix-LM full attention, each token's hidden state has already fused vision + lang + (Pi0.5 discretized state) cross-modal information, so **no hand-tuned multi-modal weighting is needed**.

**Use case**: CP1-checkpoint cache key construction; suitable for multi-task / cross-task generalization. Under a single-task setup, it does not necessarily beat the existing `cp1_mean_pool`.

> **Note**: this builder does **not** save Stage 2 compute on hits (this conflicts with CP1 timing and the design does not address it). It only swaps the key representation, with the expectation of improving hit quality. Per-step online cost ≈ 0.5–2 ms (A100/4090 bf16).

---

## 2. Quickstart

### 2.1 Build an offline artifact

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_llm_layer_extract \
    --extract-layer 0 \
    --prefix-reducer-type prefix_mean_pool \
    --checkpoint-dir <path-to-pi05-checkpoint> \
    --config-name pi05_libero \
    --output exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_meanpool.pkl
```

**Notes**:
- `--workers` is forced to `-1` (serial in-process). The model takes 5–10 GB of VRAM and cannot be parallelized with a ProcessPool.
- Default `--device cuda`. CPU mode is only suitable for smoke tests.
- At startup, a tokenizer self-check runs: re-tokenize the first step of the first episode, run `embed_language_tokens`, and `allclose` against the HDF5 `prompt_emb` at mask=True positions. On failure it aborts immediately, signaling a checkpoint/data mismatch.

### 2.2 Online inference YAML

```yaml
# cache_llm_l0_meanpool.yaml
enabled: true

key_builder:
  type: cp1_llm_layer_extract
  extract_layer: 0                  # gemma_2b: 0..17
  prefix_reducer:
    type: prefix_mean_pool          # or per_modality_mean_pool

keys:
  vision_0: { enabled: true, weight: 1.0 }
  # NOTE: under prefix_mean_pool, vision_1/2 and prompt_emb must NOT be enabled
  robot_state: { enabled: true, weight: 1.0 }

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048                  # gemma_2b width — must equal 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_meanpool.pkl

checkpoints:
  cp1:
    enabled: true                   # cp1_llm_layer_extract requires CP1 enabled
    judge:
      type: threshold
      threshold: 0.98
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
  cp3:
    enabled: true
    judge:
      type: threshold
      threshold: 0.95
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
```

### 2.3 Launch the inference server

```bash
uv run python scripts/serve_policy.py \
    --env LIBERO \
    --cache_config cache_llm_l0_meanpool.yaml
```

`InferenceInterceptor.__init__` automatically calls `key_builder.attach_model(model)` to lend PaliGemma's `layers[0..N]` and `rotary_emb` references to the KeyBuilder. No manual wiring required.

---

## 3. Two-Stage Architecture

### 3.1 Step A: LLMLayerExtractor — "where to extract from"

**Inputs** (from `Stage1Output`):
- `prefix_embs [1, 968, 2048]`: 3-camera SigLIP tokens (768) + lang tokens (200, right-padded)
- `prefix_pad_masks [1, 968]`: True = real token, False = padding
- `prefix_att_2d_masks_4d [1, 1, 968, 968]`: prefix-LM 4D additive mask
- `prefix_position_ids [1, 968]`: cumulative position over real tokens

**Algorithm**:
1. Borrow `model.paligemma.language_model.layers[0..extract_layer]` + `rotary_emb`.
2. Cast `prefix_embs` to the dtype of layer 0's weights (typically bf16).
3. Compute shared `(cos, sin) = rotary_emb(hidden, position_ids)`.
4. Forward `extract_layer + 1` layers sequentially (no_grad, no KV cache, `adarms_cond=None`).
5. Drop the batch dim → `[968, 2048]` bf16.
6. Wrap as `LLMLayerExtractResult(hidden_states, pad_mask, segment_offsets, extract_layer)`.

**Modality slice offsets** (still usable for per-modality pool after layer N):

| Range | Modality |
|-------|----------|
| `[0   : 256 )` | vision_0 (base_0_rgb) |
| `[256 : 512 )` | vision_1 (left_wrist_0_rgb) |
| `[512 : 768 )` | vision_2 (right_wrist_0_rgb) |
| `[768 : 968 )` | prompt (lang tokens, includes padding) |

**Model injection**: one-shot borrow via `KeyBuilder.attach_model(model)`. The `InferenceInterceptor` calls it automatically; the offline artifact builder calls it automatically too.

**Stateful?** No (no cross-step cache, no `on_episode_start` needed).

### 3.2 Step B: PrefixReducer — "how to build the key"

| Reducer | Input | Output | Purpose |
|---------|-------|--------|---------|
| `prefix_mean_pool` | `LLMLayerExtractResult` | `{vision_0: 2048}` | Original-spec baseline. Masked mean over the whole prefix, single key. |
| `per_modality_mean_pool` | same | `{vision_0/1/2: 2048, prompt_emb: 2048}` | Preserves modality ablation. Masked mean per segment; if a camera is missing the segment is omitted. |
| `per_modality_max_pool` | same | `{vision_0/1/2: 2048, prompt_emb: 2048}` | Per-segment masked max pool (padding → -inf); more sensitive to dominant activations. |
| `per_modality_spatial_pool_16` | same | `{vision_0/1/2: 32768, prompt_emb: 2048}` | Reshape vision segment back to a 16×16 grid, adaptive_avg_pool to 4×4 = 16 tokens; prompt segment is variable-length so falls back to masked mean. Aligned with legacy `cp1_spatial_pool_16`. |
| `per_modality_spatial_pool_4` | same | `{vision_0/1/2: 8192, prompt_emb: 2048}` | Same as above but pooled to 2×2 = 4 tokens — aggressive downsampling. Aligned with legacy `cp1_spatial_pool_4` (aka `cp1_spatial_pool_64`). |

**Key constraints**:
- Masked mean is mandatory (`pad_mask=False` positions do not enter the pool). The lang segment is 60%+ padding in normal conditions.
- A fully-False segment is omitted entirely (no zero vector emitted), matching `CLIPKeyBuilder` behavior.
- Output is a GPU tensor; the CPU transfer and `.float().contiguous()` are done in the KeyBuilder.

---

## 4. Parameters

### 4.1 KeyBuilder parameters

| Param | Type | Default | Constraint | Notes |
|-------|------|---------|------------|-------|
| `extract_layer` | int | 0 | 0 ≤ layer < 18 | gemma_2b depth=18; start at 0, sweep 0/2/5. |
| `apply_final_norm` | bool | False | must be False | Not implemented in v1; setting True raises `NotImplementedError` immediately. |

### 4.2 prefix_reducer parameters

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `type` | str | `prefix_mean_pool` | One of five: `prefix_mean_pool` / `per_modality_mean_pool` / `per_modality_max_pool` / `per_modality_spatial_pool_16` / `per_modality_spatial_pool_4`. |

### 4.3 `vector_dims` vs reducer correspondence

Vision/prompt field dimensions under `backend.vector_dims` **must** equal 2048 (gemma_2b width); otherwise config validation fails:

| reducer.type | Required enabled vision/prompt | vector_dims fields |
|--------------|-------------------------------|--------------------|
| `prefix_mean_pool` | Only `vision_0` (all other vision/prompt must be disabled) | `vision_0: 2048` |
| `per_modality_mean_pool` | Any subset of `vision_0/1/2` and `prompt_emb` | Every enabled field = 2048 |
| `per_modality_max_pool` | same | Every enabled field = 2048 |
| `per_modality_spatial_pool_16` | same | Vision fields = 32768, `prompt_emb` = 2048 |
| `per_modality_spatial_pool_4` | same | Vision fields = 8192, `prompt_emb` = 2048 |

`robot_state` goes through the original raw path (it does not enter layer N), can be enabled independently, and its dimensionality is model-determined (Pi0.5 → 32).

---

## 5. Offline Artifact CLI Reference

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir <HDF5 data dir> \
    --builder-type cp1_llm_layer_extract \
    --output <output .pkl path> \
    --extract-layer <int>             # default 0 \
    --prefix-reducer-type <prefix_mean_pool|per_modality_mean_pool|per_modality_max_pool|per_modality_spatial_pool_16|per_modality_spatial_pool_4> \
    --checkpoint-dir <PI0Pytorch weights dir containing model.safetensors> \
    --config-name <TrainConfig name, e.g. pi05_libero> \
    --device <cuda|cpu>               # default cuda \
    # Note: --workers is forced to -1
```

### 5.1 Offline pipeline steps

1. Load PI0Pytorch + PaligemmaTokenizer (max_len=200).
2. Run the **tokenizer self-check** on the first episode's first step: re-tokenize → `embed_language_tokens × √2048` → `allclose(rtol=1e-2, atol=1e-2)` against the HDF5 `prompt_emb` at mask=True positions; abort on failure.
3. Create `CP1LLMLayerExtractKeyBuilder` and call `attach_model`.
4. Serial loop over each episode's each step:
   - `_build_fake_stage1_with_masks(group, task, tokenizer, model, device)` re-tokenizes via `f.attrs['task']` + `step['robot_state']` to recover `lang_masks`.
   - Synthesize `prefix_pad_masks` (vision True / missing camera False, lang follows lang_masks).
   - `prefix_position_ids = cumsum(pad_masks) - 1`.
   - Call `model._prepare_attention_masks_4d(make_att_2d_masks(...))` to build the 4D additive mask.
   - `keybuilder.collect → build → clear`.
5. Write the artifact pickle (with metadata).

### 5.2 Artifact metadata fields

`reducer_params` dict records:

| Field | Meaning |
|-------|---------|
| `extract_layer` | Index of the extracted layer |
| `prefix_reducer_type` | Reducer type |
| `apply_final_norm` | Always `false` (v1) |
| `checkpoint_dir` | Path to the loaded checkpoint |
| `config_name` | TrainConfig name |
| `tokenizer_class` | `"PaligemmaTokenizer"` |
| `tokenizer_source` | `"gs://big_vision/paligemma_tokenizer.model"` |
| `tokenizer_max_len` | `200` |

---

## 6. Online / Offline Consistency

**Core claim**: for the same observation, the online and offline layer-N hidden states should be **bit-equivalent** at `prefix_pad_masks=True` positions (within bf16 tolerance); the post-masked-pool query_keys are strictly equal.

Reasons:
- The numerical source of `prefix_embs` is identical (vision from SigLIP, lang from `embed_language_tokens × √2048`, stored directly in HDF5).
- The 4D attention mask sets -2.38e38 at padding columns, so layer-N output at valid positions is independent of padding-position hidden states.
- Masked mean takes only `pad_mask=True` positions → result equals the online one.

**Mandatory verification** (must pass locally):

```bash
PI05_CHECKPOINT_DIR=/path/to/checkpoint \
PI05_CONFIG_NAME=pi05_libero \
uv run pytest tests/cache/test_llm_layer_extract_parity.py -m manual -v
```

The test asserts:
1. Online and offline `prefix_pad_masks` are exactly equal.
2. Online and offline layer-0 hidden states `allclose(rtol=1e-2, atol=1e-2)` at mask=True positions.
3. Online and offline `KeyBuilder.build()` output `allclose(rtol=1e-2, atol=1e-2)` per field.

Any failure indicates the offline contract is broken (tokenizer/checkpoint drift, padding handling bug, etc.). Verify stage must pass.

---

## 7. Comparison with Existing KeyBuilders

| Feature | `cp1_mean_pool` | `cp1_temporal_prune` | **`cp1_llm_layer_extract`** |
|---------|-----------------|----------------------|------------------------------|
| Data source | SigLIP tokens (Stage 1) | SigLIP tokens + cross-step prune | **LLM layer-N hidden (borrows model forward)** |
| Cross-modal fusion | None | None | **Yes (prefix-LM attention)** |
| Model dependency | None | None | Requires `attach_model` |
| Stateful? | No | Yes (FIFO history) | No |
| Online cost | < 0.1 ms | < 0.5 ms | 0.5–2 ms (one Gemma 2B layer forward) |
| Offline cost | Fast (multi-process) | Fast (multi-process) | Slow (must be serial + GPU) |
| Single-field vs multi-field | Multi-field | Multi-field | Depends on reducer (`prefix_mean_pool` single, `per_modality_mean_pool` multi) |

---

## 8. Module File Map

| File | Contents |
|------|----------|
| `src/openpi/cache/components/prefix_reducer.py` | `LLMLayerExtractResult`, `PrefixReducer` Protocol, 5 reducer implementations (mean/max/spatial×2 per-modality + global mean) |
| `src/openpi/cache/components/llm_layer_key_builder.py` | `CP1LLMLayerExtractKeyBuilder` (with `attach_model`) |
| `src/openpi/cache/orchestrator.py` | Exposes `key_builder` property (for Interceptor to call `attach_model`) |
| `src/openpi/cache/interceptor.py` | Auto-hooks `attach_model` at end of `__init__` (`hasattr` soft-probe) |
| `src/openpi/cache/config.py` | `PrefixReducerConfig`, `KeyBuilderConfig.extract_layer/prefix_reducer`, validation, factory |
| `exp/common/build_in_memory_cache_artifact.py` | `_build_fake_stage1_with_masks`, `_self_check_tokenizer_consistency`, `_load_pi05_for_llm_extract`, `_process_episode_with_model` |
| `tests/cache/components/test_prefix_reducer.py` | 17 reducer unit tests |
| `tests/cache/components/test_llm_layer_key_builder.py` | 22 KeyBuilder unit tests |
| `tests/cache/test_interceptor_attach_model.py` | 3 Interceptor hook tests |
| `tests/cache/test_llm_layer_extract_parity.py` | Online/offline parity test (`@pytest.mark.manual`) |

---

## 9. FAQ

### Q: What value of `extract_layer` should I pick?

For v1, start with `0` (the original spec). If hit quality is insufficient, sweep `0 / 2 / 5`:

| layer | Cumulative forward as % of Stage 2 | Cross-modal fusion depth |
|-------|------------------------------------|--------------------------|
| 0 | ~5.5% | 1 attention round |
| 2 | ~16.7% | 3 rounds |
| 5 | ~33.3% | 6 rounds |

### Q: Why isn't `robot_state` fed into layer N?

Pi0.5 already writes `robot_state` into the prompt **in a discretized text form** (`tokenizer.py:24-29`: `f"Task: {task}, State: {state_str};\nAction: "`), so the lang-segment hidden state already carries the state signal. But the raw 32-d continuous state has many times the L2-distance precision of the discrete text, so keeping it as a separate field pays off.

### Q: Is the single-task experiment worth running?

Not really. In LIBERO single-task, the per-step prompt is identical except for the discrete state field → the extra signal from lang-segment fusion is limited, and most of the difference comes from cross-camera vision fusion. The advantage of this approach over `cp1_mean_pool` shows up in **multi-task / cross-task** scenarios.

### Q: How much does online inference latency increase?

0.5–2 ms per step on A100/4090 bf16. Pi0.5's total inference latency is ~50 ms, so the relative increase is ~2% — acceptable.

### Q: What if the offline self-check fails?

Most common causes:
1. `--checkpoint-dir` does not match the checkpoint used when collecting the HDF5.
2. The PaligemmaTokenizer download source has been replaced or the local cache is corrupted.
3. The HDF5 was collected with a different model / older tokenizer version.

Triage path: first cross-check the `reducer_params` metadata (`tokenizer_source / checkpoint_dir`) against the collection-time setup; then verify the local cache hash of `gs://big_vision/paligemma_tokenizer.model`.

### Q: Can I reuse `cp1_temporal_prune`'s TokenReducer?

No. The two reducer protocols take different inputs: `TokenReducer` takes a `PruneResult` (pruned vision tokens), while `PrefixReducer` takes a `LLMLayerExtractResult` (with pad_mask + modality slices). Forcing reuse would break the padding semantics. The two pipelines are kept independent on purpose.
