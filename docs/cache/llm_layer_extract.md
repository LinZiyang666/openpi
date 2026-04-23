# CP1 LLM Layer Extract KeyBuilder 使用指南

> **前置知识**：阅读 [tutorial.md](tutorial.md) §4 了解 KeyBuilder 组件基础，§10 了解 YAML 配置。
>
> **设计文档**：完整方案设计与决策见 [`logs/cp1_llm_layer_extract_key_builder_plan.log.md`](../../logs/cp1_llm_layer_extract_key_builder_plan.log.md)（Plan，G1 APPROVED）。

---

## 1. 概述

`CP1LLMLayerExtractKeyBuilder` 是一个两步式 KeyBuilder。它在 KeyBuilder 内部独立跑 PaliGemma backbone 的前 `N+1` 层 forward（借用模型的 layer 引用，不修改 Stage 2），用 layer-N 的 hidden state 作为 cache key 的源数据：

```
Stage1Output                                   → KeyBuilder
   │                                              ┌─ Step A: LLMLayerExtractor (借用 layer 0..N + rotary_emb)
   ▼                                              │      跑前 N+1 层 forward (no_grad, no KV cache)
prefix_embs [1, 968, 2048]                        │      → LLMLayerExtractResult(hidden, mask, segments, layer)
prefix_pad_masks / prefix_position_ids /          │
  prefix_att_2d_masks_4d                          ▼
                                              ┌─ Step B: PrefixReducer (可插拔)
                                              │      prefix_mean_pool / per_modality_mean_pool
                                              │      → {field: [2048]}
                                              ▼
                                          {vision_0: …, robot_state: …} → CPU float32
```

**核心思想**：经过一层（或多层）prefix-LM full attention 后，每个 token 的 hidden state 已经融合了 vision + lang + (Pi0.5 离散化 state) 的跨模态信息，**不再需要人工调多模态权重**。

**适用场景**：CP1 检查点的 cache key 构造；适用于多任务 / 跨任务泛化场景，单任务下不一定优于现有 `cp1_mean_pool`。

> **注意**：本 builder 在 hit 时**不省 Stage 2 算力**（与 CP1 时序矛盾，方案不解决这个问题）。它只换 key 表征，期望提升 hit 质量。每步在线代价 ≈ 0.5–2 ms（A100/4090 bf16）。

---

## 2. 快速开始

### 2.1 构建离线 Artifact

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

**注意事项**：
- `--workers` 被强制设为 `-1`（serial in-process）。模型载入 5–10 GB VRAM，无法用 ProcessPool 并行。
- 默认 `--device cuda`。CPU 模式仅适合 smoke test。
- 启动时跑 tokenizer self-check：用首个 episode 首步重新 tokenize 后 `embed_language_tokens`，与 HDF5 `prompt_emb` 在 mask=True 位置 `allclose`。fail 立即 abort，提示 checkpoint/data 不匹配。

### 2.2 配置在线推理 YAML

```yaml
# cache_llm_l0_meanpool.yaml
enabled: true

key_builder:
  type: cp1_llm_layer_extract
  extract_layer: 0                  # gemma_2b: 0..17
  prefix_reducer:
    type: prefix_mean_pool          # 或 per_modality_mean_pool

keys:
  vision_0: { enabled: true, weight: 1.0 }
  # 注意：prefix_mean_pool 模式下 vision_1/2/prompt_emb 不可 enable
  robot_state: { enabled: true, weight: 1.0 }

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048                  # gemma_2b width，必须 = 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_meanpool.pkl

checkpoints:
  cp1:
    enabled: true                   # cp1_llm_layer_extract 要求 CP1 启用
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

### 2.3 启动推理服务

```bash
uv run python scripts/serve_policy.py \
    --env LIBERO \
    --cache_config cache_llm_l0_meanpool.yaml
```

`InferenceInterceptor.__init__` 会自动调 `key_builder.attach_model(model)` 把 PaliGemma 的 `layers[0..N]` 和 `rotary_emb` 引用借给 KeyBuilder。无需手动接线。

---

## 3. 两步架构详解

### 3.1 Step A: LLMLayerExtractor（决定"从哪抽"）

**输入**（来自 `Stage1Output`）：
- `prefix_embs [1, 968, 2048]`：3 相机 SigLIP token (768) + lang token (200, 含右 padding)
- `prefix_pad_masks [1, 968]`：True = 真实 token，False = padding
- `prefix_att_2d_masks_4d [1, 1, 968, 968]`：prefix-LM 4D additive mask
- `prefix_position_ids [1, 968]`：cumulative position over real tokens

**算法**：
1. 借用 `model.paligemma.language_model.layers[0..extract_layer]` + `rotary_emb`
2. cast prefix_embs 到 layer 0 weight 的 dtype（通常 bf16）
3. 计算共享的 `(cos, sin) = rotary_emb(hidden, position_ids)`
4. 顺序 forward `extract_layer + 1` 层（no_grad, no KV cache, `adarms_cond=None`）
5. drop batch 维 → `[968, 2048]` bf16
6. 包装为 `LLMLayerExtractResult(hidden_states, pad_mask, segment_offsets, extract_layer)`

**模态切片偏移**（layer-N 之后槽位仍可用于 per-modality pool）：

| 偏移 | 模态 |
|------|------|
| `[0   : 256 )` | vision_0 (base_0_rgb) |
| `[256 : 512 )` | vision_1 (left_wrist_0_rgb) |
| `[512 : 768 )` | vision_2 (right_wrist_0_rgb) |
| `[768 : 968 )` | prompt (lang token, 含 padding) |

**模型注入**：通过 `KeyBuilder.attach_model(model)` 一次性借用。`InferenceInterceptor` 自动调用；离线 artifact builder 也自动调用。

**有状态？** 否（无跨步缓存，无需 `on_episode_start`）。

### 3.2 Step B: PrefixReducer（决定"怎么 build key"）

| Reducer | 输入 | 输出 | 用途 |
|---------|------|------|------|
| `prefix_mean_pool` | `LLMLayerExtractResult` | `{vision_0: 2048}` | 教授原意 baseline。全 prefix masked mean，单 key。 |
| `per_modality_mean_pool` | 同上 | `{vision_0/1/2: 2048, prompt_emb: 2048}` | 保留模态消融。每段 masked mean，相机缺失时该段 omit。 |
| `per_modality_max_pool` | 同上 | `{vision_0/1/2: 2048, prompt_emb: 2048}` | 每段 masked max pool（padding 置 -inf）；对显著激活更敏感。|
| `per_modality_spatial_pool_16` | 同上 | `{vision_0/1/2: 32768, prompt_emb: 2048}` | vision 段重排为 16×16 grid 做 adaptive_avg_pool 到 4×4 = 16 tokens；prompt 段变长，fallback masked mean。对齐 legacy `cp1_spatial_pool_16`。|
| `per_modality_spatial_pool_4` | 同上 | `{vision_0/1/2: 8192, prompt_emb: 2048}` | 同上但 pool 到 2×2 = 4 tokens，激进下采样。对齐 legacy `cp1_spatial_pool_4`（又名 `cp1_spatial_pool_64`）。|

**关键约束**：
- 必须做 masked mean（`pad_mask=False` 位置不进入 pool）。lang 段 60%+ 通常是 padding。
- 全 False 段直接 omit field，不输出 zero vector（与 `CLIPKeyBuilder` 行为一致）。
- 输出 GPU tensor；CPU transfer 与 `.float().contiguous()` 由 KeyBuilder 完成。

---

## 4. 参数说明

### 4.1 KeyBuilder 参数

| 参数 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `extract_layer` | int | 0 | 0 ≤ layer < 18 | gemma_2b depth=18；建议起步 0，sweep 0/2/5 |
| `apply_final_norm` | bool | False | 必须 False | 首版不实现；为 True 立即抛 `NotImplementedError` |

### 4.2 prefix_reducer 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `type` | str | `prefix_mean_pool` | 五选一：`prefix_mean_pool` / `per_modality_mean_pool` / `per_modality_max_pool` / `per_modality_spatial_pool_16` / `per_modality_spatial_pool_4` |

### 4.3 vector_dims 与 reducer 的对应关系

`backend.vector_dims` 中 vision/prompt 字段维度**必须** = 2048（gemma_2b width），否则 config 校验会报错：

| reducer.type | 必须 enabled 的 vision/prompt | vector_dims 字段 |
|--------------|---------------------------------|------------------|
| `prefix_mean_pool` | 仅 `vision_0`（其他 vision/prompt 必须 disabled） | `vision_0: 2048` |
| `per_modality_mean_pool` | `vision_0/1/2`、`prompt_emb` 任意子集 | 每个 enabled 字段都 = 2048 |
| `per_modality_max_pool` | 同上 | 每个 enabled 字段都 = 2048 |
| `per_modality_spatial_pool_16` | 同上 | vision 字段 = 32768，`prompt_emb` = 2048 |
| `per_modality_spatial_pool_4` | 同上 | vision 字段 = 8192，`prompt_emb` = 2048 |

`robot_state` 走原 raw 路径（不进 layer N），可独立 enable，维度由 model 决定（Pi0.5 是 32）。

---

## 5. 离线 Artifact CLI 参考

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir <HDF5 数据目录> \
    --builder-type cp1_llm_layer_extract \
    --output <输出 .pkl 路径> \
    --extract-layer <int>             # 默认 0 \
    --prefix-reducer-type <prefix_mean_pool|per_modality_mean_pool|per_modality_max_pool|per_modality_spatial_pool_16|per_modality_spatial_pool_4> \
    --checkpoint-dir <PI0Pytorch 权重目录, 含 model.safetensors> \
    --config-name <TrainConfig 名, 例 pi05_libero> \
    --device <cuda|cpu>               # 默认 cuda \
    # 注: --workers 会被强制设为 -1
```

### 5.1 离线流程内部步骤

1. 加载 PI0Pytorch + PaligemmaTokenizer（max_len=200）
2. 用首 episode 首 step 跑 **tokenizer self-check**：重 tokenize → `embed_language_tokens × √2048` → 与 HDF5 `prompt_emb` 在 mask=True 位置 `allclose(rtol=1e-2, atol=1e-2)`；fail 即 abort
3. 创建 `CP1LLMLayerExtractKeyBuilder` 并 `attach_model`
4. 串行处理每个 episode 每步：
   - `_build_fake_stage1_with_masks(group, task, tokenizer, model, device)` 通过 `f.attrs['task']` + `step['robot_state']` 重 tokenize 还原 `lang_masks`
   - 合成 `prefix_pad_masks` (vision True / 缺失相机 False，lang 接 lang_masks)
   - `prefix_position_ids = cumsum(pad_masks) - 1`
   - 调 `model._prepare_attention_masks_4d(make_att_2d_masks(...))` 构造 4D additive mask
   - `keybuilder.collect → build → clear`
5. 写 artifact pickle（含 metadata）

### 5.2 Artifact metadata 字段

`reducer_params` 字典记录：

| 字段 | 含义 |
|------|------|
| `extract_layer` | 抽取的 layer 索引 |
| `prefix_reducer_type` | reducer 类型 |
| `apply_final_norm` | 始终 `false`（首版） |
| `checkpoint_dir` | 加载的 checkpoint 路径 |
| `config_name` | TrainConfig 名 |
| `tokenizer_class` | `"PaligemmaTokenizer"` |
| `tokenizer_source` | `"gs://big_vision/paligemma_tokenizer.model"` |
| `tokenizer_max_len` | `200` |

---

## 6. 在线/离线一致性

**核心声明**：在线和离线给同一观察生成的 layer-N hidden state 在 `prefix_pad_masks=True` 的位置上**应当 bit-equivalent**（bf16 容差），masked pool 后的 query_keys 严格相同。

理由：
- prefix_embs 数值来源相同（vision 来自 SigLIP、lang 来自 `embed_language_tokens × √2048`，HDF5 直接存）。
- attention 4D mask 在 padding 列填 -2.38e38，valid 位置的 layer-N 输出与 padding 位置 hidden 无关。
- masked mean 只取 `pad_mask=True` 位置 → 结果与在线相同。

**强制验证**（必须本地跑通）：

```bash
PI05_CHECKPOINT_DIR=/path/to/checkpoint \
PI05_CONFIG_NAME=pi05_libero \
uv run pytest tests/cache/test_llm_layer_extract_parity.py -m manual -v
```

测试断言：
1. 在线/离线 `prefix_pad_masks` 完全一致
2. 在线/离线 layer-0 hidden state 在 mask=True 位置 `allclose(rtol=1e-2, atol=1e-2)`
3. 在线/离线 KeyBuilder.build() 输出每个 field `allclose(rtol=1e-2, atol=1e-2)`

任何 fail 都说明离线契约被破坏（tokenizer/checkpoint 漂移、padding 处理 bug 等）。Verify 阶段必须跑通。

---

## 7. 与现有 KeyBuilder 对比

| 特性 | `cp1_mean_pool` | `cp1_temporal_prune` | **`cp1_llm_layer_extract`** |
|------|-----------------|----------------------|------------------------------|
| 数据源 | SigLIP token (Stage 1) | SigLIP token + 跨步 prune | **LLM layer-N hidden (借模型 forward)** |
| 跨模态融合 | 无 | 无 | **有（prefix-LM attention）** |
| 模型依赖 | 无 | 无 | 需 `attach_model` |
| 有状态 | 否 | 是（FIFO 历史） | 否 |
| 在线代价 | < 0.1 ms | < 0.5 ms | 0.5–2 ms (1 层 Gemma 2B forward) |
| 离线代价 | 快（多进程） | 快（多进程） | 慢（必须 serial + GPU） |
| 单字段 vs 多字段 | 多字段 | 多字段 | 取决于 reducer (`prefix_mean_pool` 单，`per_modality_mean_pool` 多) |

---

## 8. 模块文件一览

| 文件 | 内容 |
|------|------|
| `src/openpi/cache/components/prefix_reducer.py` | `LLMLayerExtractResult`、`PrefixReducer` Protocol、5 个 reducer 实现（mean/max/spatial×2 按模态 + 全局 mean） |
| `src/openpi/cache/components/llm_layer_key_builder.py` | `CP1LLMLayerExtractKeyBuilder`（含 `attach_model`） |
| `src/openpi/cache/orchestrator.py` | 公开 `key_builder` property（供 Interceptor 调 `attach_model`） |
| `src/openpi/cache/interceptor.py` | `__init__` 末尾自动 hook `attach_model`（`hasattr` 软探测） |
| `src/openpi/cache/config.py` | `PrefixReducerConfig`、`KeyBuilderConfig.extract_layer/prefix_reducer`、validation、factory |
| `exp/common/build_in_memory_cache_artifact.py` | `_build_fake_stage1_with_masks`、`_self_check_tokenizer_consistency`、`_load_pi05_for_llm_extract`、`_process_episode_with_model` |
| `tests/cache/components/test_prefix_reducer.py` | 17 个 reducer 单元测试 |
| `tests/cache/components/test_llm_layer_key_builder.py` | 22 个 KeyBuilder 单元测试 |
| `tests/cache/test_interceptor_attach_model.py` | 3 个 Interceptor hook 测试 |
| `tests/cache/test_llm_layer_extract_parity.py` | 在线/离线 parity 测试（`@pytest.mark.manual`） |

---

## 9. 常见问题

### Q: extract_layer 应该选多少？

首版建议 `0`（教授原意）。如果 hit 质量不够，sweep `0 / 2 / 5`：

| layer | 累计 forward 占 Stage 2 | 跨模态融合深度 |
|-------|--------------------------|-----------------|
| 0 | ~5.5% | 1 轮 attention |
| 2 | ~16.7% | 3 轮 |
| 5 | ~33.3% | 6 轮 |

### Q: 为什么 robot_state 不进 layer N？

Pi0.5 已经把 robot_state **以离散化文本形式**写进 prompt（`tokenizer.py:24-29`：`f"Task: {task}, State: {state_str};\nAction: "`），所以 lang 段的 hidden state 已经携带 state 信号。但原始 32-d 连续 state 的 L2 距离精度比离散文本高很多倍，单独保留作为独立字段更划算。

### Q: 单任务（single task）实验值得跑吗？

不太值得。LIBERO 单任务每步 prompt 完全相同（除离散 state 字段外）→ lang 段融合的额外信号有限，主要差异来自 vision 跨相机融合。**多任务/跨任务**场景才能看出本方案相对 `cp1_mean_pool` 的优势。

### Q: 在线推理时延会增加多少？

A100/4090 bf16 上单步 0.5–2 ms。Pi0.5 总推理时延 ~50 ms，相对增加 ~2%，可接受。

### Q: 离线 self-check fail 怎么办？

最常见原因：
1. `--checkpoint-dir` 与采集 HDF5 时用的 checkpoint 不一致
2. PaligemmaTokenizer 下载源被替换或本地缓存损坏
3. HDF5 是用其他模型 / 旧版 tokenizer 采集的

排查路径：先核对 `reducer_params` 元数据（`tokenizer_source / checkpoint_dir`）是否与采集时一致；再验证 `gs://big_vision/paligemma_tokenizer.model` 本地缓存哈希。

### Q: 能否复用 cp1_temporal_prune 的 TokenReducer？

不能。两边的 reducer 协议输入不同：`TokenReducer` 接 `PruneResult`（pruned vision tokens），`PrefixReducer` 接 `LLMLayerExtractResult`（带 pad_mask + 模态切片）。强行复用会破坏 padding 语义。设计上故意保持两条独立 pipeline。
