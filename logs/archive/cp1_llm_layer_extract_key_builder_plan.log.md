# CP1 LLM Layer Extract KeyBuilder — 实施计划

> Status: `Plan`
> Level: **L2**（新 KeyBuilder 组件，触动多文件，需 G1/G2）
> Authority: Execution
> Owner: Ziyang
> Date: 2026-04-22

---

## 1. 背景与动机

### 1.1 教授提出的方向

与其在 CP1 上反复调多个模态（vision_0/1/2、prompt_emb、robot_state）的融合权重，不如直接利用 LLM backbone 第一层的 hidden state 作为 cache key — 一层 prefix-LM full attention 已经把 vision + lang + (Pi0.5 的离散化 state) 跨模态融合好了，**不再需要人工调权重**。

### 1.2 不动 Stage 2 的约束

Stage 2 当前是 `paligemma.language_model.forward(...)` 完整 18 层 prefill，**不拆**。本方案在 KeyBuilder 内部独立跑 layer N 的 forward（借用模型的 layer 引用 + rotary_emb，零额外 VRAM）。

- hit 时**不省** Stage 2 算力（hit 后才能得到 key 反过来跳 Stage 2 — 与 CP1 时序矛盾，方案不解决这个问题）。
- 本方案**只换 key 表征**：用 layer N 输出取代 SigLIP 原始 token / CLIP 视觉 emb，期望提升 hit 质量与跨任务泛化。
- 与现有 `cp1_mean_pool` 直接对照即可观察 key 质量差异。

### 1.3 工程对应模式

与 `cp1_temporal_prune` 完全平行的两步可插拔架构：

| 维度 | cp1_temporal_prune | cp1_llm_layer_extract |
|------|-------------------|-----------------------|
| Step A | TemporalPrune（FIFO 跨步） | LLMLayerExtractor（无跨步） |
| Step A 输出 | `PruneResult` | `LLMLayerExtractResult` |
| Step B 协议 | `TokenReducer` | 新 `PrefixReducer`（独立） |
| 有状态 | 是 | 否 |

---

## 2. 架构设计

### 2.1 顶层数据流

```
Stage1Output
   │
   ▼
LLMLayerExtractor                       ← Step A: 决定"从哪抽"
   ├ 借用 model.paligemma.language_model.layers[0..N]  (引用, 零拷贝)
   ├ 借用 model.paligemma.language_model.rotary_emb
   ├ 跑前 N+1 层 forward (no KV cache, no_grad)
   │   输入: prefix_embs, prefix_pad_masks,
   │         prefix_att_2d_masks_4d, prefix_position_ids
   │   不传 adarms_cond (Pi0.5 paligemma 侧 use_adarms[0]=False)
   └ 输出 LLMLayerExtractResult(
         hidden_states: [L, 2048] bf16            # L=968 (Pi0.5)
         pad_mask: [L] bool                       # prefix_pad_masks[0]
         segment_offsets: dict                    # 模态边界（layer 0 后语义已融合，但槽位编号有效）
         extract_layer: int                       # 实际抽取的层号
     )
   │
   ▼
PrefixReducer                           ← Step B: 决定"怎么 build key"
   │   输入: LLMLayerExtractResult
   │   输出: dict[str, torch.Tensor]    # CPU float32 contiguous
   ├ PrefixMeanPoolReducer       → {vision_0: 2048}                          # 教授原意 baseline
   └ PerModalityPoolReducer      → {vision_0: 2048, vision_1: 2048,
                                     vision_2: 2048, prompt_emb: 2048}        # 保留模态消融维度
```

### 2.2 模态位置 (layer N 之后槽位仍可识别)

```
prefix layout (Pi0.5, prefix_len = 968 固定):
  [0    : 256 )   = vision_0  (base_0_rgb,        SigLIP 256 token)
  [256  : 512 )   = vision_1  (left_wrist_0_rgb,  SigLIP 256 token)
  [512  : 768 )   = vision_2  (right_wrist_0_rgb, SigLIP 256 token)
  [768  : 968 )   = prompt    (lang token, max_token_len=200, 右 padding)
```

layer N forward 后：每个位置的 hidden state 已经过 N+1 轮 prefix-LM full attention，**不再是原模态的纯 token**，但**槽位起止偏移仍然有效**，可以用于 per-modality pool。

---

## 3. 关键设计决策（已锁定）

| 决策 | 选择 | 理由 |
|------|------|------|
| 命名 | `cp1_llm_layer_extract` (KeyBuilder type) | 显式带"llm"防止与 ViT/SigLIP 抽取混淆 |
| 类名 | `CP1LLMLayerExtractKeyBuilder` | 与 type 字符串一一对应 |
| Step B 协议 | 新建 `PrefixReducer`，**不复用** `TokenReducer` | 输入带 pad_mask + segment 信息，签名不兼容 |
| 首版 reducer | `prefix_mean_pool`、`per_modality_pool` | A=教授原意，B=保留消融。`prefix_max_pool` / `prompt_token_scoring` 推迟到首版有信号后另起 L1 |
| `extract_layer` | 配置为 `int`，范围 0..17，默认 0 | 单 YAML 单值；实验 sweep 用多份 YAML |
| `apply_final_norm` | 默认 `False`，首版不实现 `True` 路径 | 仅 `extract_layer=17` 时才与 `last_hidden_state` 等价；引入即增加测试面 |
| dtype | hidden state bf16 内部，pool 后 cast float32 → CPU | 与现有 `_to_cpu_float32` 一致 |
| pad mask | masked mean 强制 | lang 段 60%+ 是 padding，naive mean 会被稀释成 padding embedding 中心 |
| 相机缺失 | 整段 mask 全 False 时 omit field | 与 CLIPKeyBuilder 一致（`clip_key_builder.py:155-161`） |
| 模型引用注入 | Interceptor 在 `__init__` 调 `key_builder.attach_model(self._model)`（如方法存在） | 类似 CLIPKeyBuilder lazy load；新 builder 实现 `attach_model` 协议方法 |
| robot_state | 走原 raw 路径（不进 layer N） | 32-d 连续 state 的 L2 距离信号比 Pi0.5 prompt 中文本化的离散 state 高很多倍精度；保留它作为可选独立字段 |

---

## 4. 文件触动清单

### 4.1 新增文件 (5)

| 路径 | 说明 |
|------|------|
| `src/openpi/cache/components/prefix_reducer.py` | `PrefixReducer` Protocol、`LLMLayerExtractResult` dataclass、`PrefixMeanPoolReducer`、`PerModalityPoolReducer` |
| `src/openpi/cache/components/llm_layer_key_builder.py` | `CP1LLMLayerExtractKeyBuilder` 类（含 `attach_model`、`collect`、`build`、`clear`、`cached_data`） |
| `tests/cache/components/test_prefix_reducer.py` | PrefixReducer 单元测试（pad mask 正确性、模态切片、相机缺失、dim） |
| `tests/cache/components/test_llm_layer_key_builder.py` | KeyBuilder 单元测试（attach_model 缺失报错、layer N 输出 shape、masked pool 与 unmasked 对比、reducer 接线、protocol 兼容、Interceptor 自动 hook attach_model） |
| `docs/cache/llm_layer_extract.md` | 组件使用指南：两步架构、YAML 配置、layer/reducer 选择、离线 artifact 构建、在线/离线一致性要求（与 `temporal_prune.md` 同结构） |

### 4.2 修改文件 (5)

| 路径 | 修改点 |
|------|--------|
| `src/openpi/cache/config.py` | 1) `KeyBuilderConfig` 加 `extract_layer: int = 0`、`prefix_reducer: PrefixReducerConfig`；2) 新建 `PrefixReducerConfig`；3) `validate_cache_config` 加分支（layer 范围、reducer 名、reducer 输出维度与 `vector_dims` 一致、enabled field 与 `reducer.emitted_fields` 一致）；4) `_build_key_builder` 加 factory 分支；5) 加 `_build_prefix_reducer` factory；6) `_valid_key_builder_types` 收录新名 |
| `src/openpi/cache/orchestrator.py` | 新增只读 `@property def key_builder(self) -> QueryKeyBuilder`（单行 accessor，返回 `self._key_builder`）。用于 Interceptor 做可选 `attach_model` hook，使接口显式且可测。 |
| `src/openpi/cache/interceptor.py` | `__init__` 末尾（`orchestrator is not None` 分支内）：`kb = self._orchestrator.key_builder; if hasattr(kb, "attach_model"): kb.attach_model(self._model)`。软探测保持向后兼容（现有 builder 无该方法时 noop）。 |
| `exp/common/build_in_memory_cache_artifact.py` | 1) 新增 `cp1_llm_layer_extract` 分支；2) 加 CLI 参数 `--checkpoint-dir` `--config-name` `--extract-layer` `--prefix-reducer-type`；3) 该 builder 强制 `--workers -1`（serial），首次循环加载 PI0Pytorch；4) 新增 `_build_fake_stage1_with_masks(group, task_str, tokenizer, model)` helper（详见 §6.2）；5) artifact 元数据记录 `extract_layer`、`prefix_reducer_type`、`apply_final_norm`、`checkpoint_dir`、`tokenizer_class`（`"PaligemmaTokenizer"`）、`tokenizer_source`（`"gs://big_vision/paligemma_tokenizer.model"`）、`tokenizer_max_len`（200） |
| `docs/README.md` | 在 `docs/cache/` 索引表追加 `llm_layer_extract.md` 行；与同 commit 的 plan + 文档一起落 |

### 4.3 不触动（明确）

- `src/openpi/models_pytorch/pi0_pytorch.py` — Stage 2 一行不改；仅在离线 builder 中 **读** `model.embed_prefix` / `make_att_2d_masks` / `_prepare_attention_masks_4d` 作为库函数调用，不修改
- `src/openpi/models_pytorch/gemma_pytorch.py`、`transformers_replace/...` — 不动
- `src/openpi/cache/components/key_builder.py` — `_VISION_OFFSETS` / `_PROMPT_START` 常量复用（import）
- `src/openpi/cache/components/token_reducer.py` — 不复用 protocol，但 import `_EMB_DIM` 常量
- `src/openpi/collect/data_collector.py` — **不** 增加字段。offline 所需的 `lang_masks` 从 HDF5 已存的 `attrs['task']` + `robot_state` 通过 `PaligemmaTokenizer` 重计算（详见 §6.2），避免破坏现有 HDF5 数据与重新采集

---

## 5. 接口定义

### 5.1 `prefix_reducer.py`

```python
@dataclass
class LLMLayerExtractResult:
    """Step A 输出，Step B 输入。

    All tensors GPU-resident (caller is KeyBuilder; CPU transfer is reducer's job
    for output, not its input).
    """
    hidden_states: torch.Tensor                  # [L, 2048] bf16, batch dim dropped
    pad_mask: torch.Tensor                       # [L] bool, True = real token
    segment_offsets: dict[str, tuple[int, int]]  # {"vision_0":(0,256), ..., "prompt_emb":(768,968)}
    extract_layer: int                           # for traceability / metadata

@runtime_checkable
class PrefixReducer(Protocol):
    """Reduce LLM layer-N hidden states to one or more cache key vectors.

    Coupling:
      DEPENDS ON: LLMLayerExtractResult (same file)
      CONSUMED BY: CP1LLMLayerExtractKeyBuilder.build()
      IF CHANGED: per-field output_dims must match backend vector_dims
    """
    def reduce(self, result: LLMLayerExtractResult) -> dict[str, torch.Tensor]:
        """Return {field_name: [dim] GPU tensor}. CPU transfer done by caller."""
        ...

    @property
    def output_dims(self) -> dict[str, int]:
        """Field name -> dim. Used for backend.vector_dims cross-validation."""
        ...

    @property
    def emitted_fields(self) -> frozenset[str]:
        """Subset of CACHE_QUERY_FIELDS this reducer can emit."""
        ...

class PrefixMeanPoolReducer:
    """Masked mean over the entire prefix → single 2048-d key under vision_0."""
    @property
    def output_dims(self) -> dict[str, int]: return {"vision_0": 2048}
    @property
    def emitted_fields(self) -> frozenset[str]: return frozenset({"vision_0"})
    def reduce(self, r): ...   # masked mean over r.pad_mask

class PerModalityPoolReducer:
    """Per-modality masked mean. Skips empty segments (e.g., absent camera)."""
    @property
    def output_dims(self) -> dict[str, int]:
        return {"vision_0": 2048, "vision_1": 2048, "vision_2": 2048, "prompt_emb": 2048}
    @property
    def emitted_fields(self) -> frozenset[str]:
        return frozenset({"vision_0", "vision_1", "vision_2", "prompt_emb"})
    def reduce(self, r): ...   # per-segment masked mean
```

### 5.2 `llm_layer_key_builder.py`

```python
class CP1LLMLayerExtractKeyBuilder:
    """Two-step KeyBuilder: LLM layer-N extraction + pluggable prefix reduction.

    Stateless across episodes (no history buffer; on_episode_start not required).
    Requires attach_model(model) before first collect().

    Coupling:
      DEPENDS ON: PI0Pytorch.paligemma_with_expert.paligemma.language_model
                  (specifically: layers[0..extract_layer], rotary_emb)
                  Stage1Output fields (prefix_embs, prefix_pad_masks,
                  prefix_att_2d_masks_4d, prefix_position_ids, state)
                  PrefixReducer (prefix_reducer.py)
      CONSUMED BY: CacheOrchestrator.check() via QueryKeyBuilder protocol
      IF CHANGED: backend.vector_dims must follow reducer.output_dims
    """

    def __init__(
        self,
        reducer: PrefixReducer,
        extract_layer: int = 0,
        enabled_fields: list[str] | None = None,
        apply_final_norm: bool = False,   # reserved; first version requires False
    ): ...

    def attach_model(self, model: "PI0Pytorch") -> None:
        """Borrow layer + rotary_emb references. Idempotent."""
        if apply_final_norm: raise NotImplementedError(...)  # first-version guard
        ...

    def collect(self, checkpoint_id, **stage_outputs):
        # Cache stage1 GPU tensors (no copy). robot_state separate path.
        ...

    def build(self, checkpoint_id) -> dict[str, torch.Tensor]:
        # 1. Layer N forward (no_grad, no_cache)
        # 2. result = LLMLayerExtractResult(hidden, pad_mask, segments, layer)
        # 3. keys = reducer.reduce(result)
        # 4. CPU transfer + L2 norm + contiguous on each field
        # 5. Add robot_state raw if enabled
        ...

    @property
    def cached_data(self) -> dict[str, torch.Tensor]: ...
    def clear(self) -> None: ...
```

### 5.3 `config.py` 增量

```python
@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"
    # -- temporal prune (existing) --
    prune_window_size: int = 4
    temporal_keep_ratio: float = 0.5
    reducer: ReducerConfig = field(default_factory=ReducerConfig)
    # -- llm layer extract (new) --
    extract_layer: int = 0                                         # 0..17 for gemma_2b
    prefix_reducer: "PrefixReducerConfig" = field(default_factory=...)

@dataclass
class PrefixReducerConfig:
    type: str = "prefix_mean_pool"   # "prefix_mean_pool" | "per_modality_pool"
```

新增 validation 规则：
- `cp1_llm_layer_extract` 要求 `cp1.enabled=true`
- `0 <= extract_layer <= 17`
- `prefix_reducer.type ∈ {"prefix_mean_pool", "per_modality_pool"}`
- `prefix_mean_pool` → enabled vision fields ⊆ `{vision_0}` AND `prompt_emb` 不可 enabled
- `per_modality_pool` → enabled vision/prompt fields ⊆ `{vision_0, vision_1, vision_2, prompt_emb}`
- 对每个 enabled field 校验 `backend.vector_dims[f]` 与 `reducer.output_dims[f]` 相等

### 5.4 `interceptor.py` 增量

```python
# 在 __init__ 末尾（orchestrator is not None 分支内，紧接现有 probe 注册之后）：
if self._orchestrator is not None:
    kb = self._orchestrator.key_builder    # public property (§5.5)
    if hasattr(kb, "attach_model"):
        kb.attach_model(self._model)
```

设计要点：
- **公开访问器**：通过 §5.5 新增的 `CacheOrchestrator.key_builder` 公开 property 拿到 builder。**不**用 `getattr` 软探测私有 `_key_builder`（reviewer G1 R1 #1 指出后修正）。
- **方法软探测**：`attach_model` 只是本 builder 的私有 hook，不入 `QueryKeyBuilder` Protocol 主签名。其他现有 builder 不实现，`hasattr` 走过即 noop，向后兼容。
- **失败模式**：若新 builder 在 YAML 启用但 Interceptor 路径未触发（例如 `orchestrator=None`），`build()` 会因 layer 引用未 attach 抛 `RuntimeError("attach_model not called; cannot run layer-N forward")`。明确错误优于沉默。

### 5.5 `orchestrator.py` 增量

唯一改动：单行只读 property，暴露已有的 `_key_builder` 字段。

```python
# In CacheOrchestrator class body:
@property
def key_builder(self) -> QueryKeyBuilder:
    """Public accessor for the bound KeyBuilder instance.

    Used by InferenceInterceptor to perform optional `attach_model`
    hook on type-specific builders (e.g. cp1_llm_layer_extract). The
    underlying builder reference is set once in __init__ and immutable;
    exposing it as a property keeps the interceptor → builder wiring
    explicit and unit-testable.
    """
    return self._key_builder
```

零行为变化（只读），不破坏现有代码。`test_orchestrator.py` 加一条断言：`orchestrator.key_builder is the_builder_passed_in`。

---

## 6. 离线 Artifact Builder 集成

### 6.1 新增 CLI

```bash
uv run exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_llm_layer_extract \
    --extract-layer 0 \
    --prefix-reducer-type prefix_mean_pool \
    --checkpoint-dir <path-to-pi05-checkpoint> \
    --config-name pi05_libero \
    --workers -1 \
    --output exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_meanpool.pkl
```

### 6.2 离线 Stage-1 重建契约（HDF5 → FakeStage1）

**HDF5 已有字段**（dump 验证，2026-04-22）：
- `f.attrs['task']` → str，episode 任务文本
- `step['vision_0/1/2']` → `(256, 2048) float16`，三相机 SigLIP 输出
- `step['prompt_emb']` → `(200, 2048) float16`，**已 padding 到 max_token_len=200**（`tokenizer.py:35-46`）
- `step['robot_state']` → `(32,) float32`

**HDF5 缺少**：`lang_masks`、`prefix_pad_masks`、`prefix_position_ids`、`prefix_att_2d_masks_4d`。注意 `prompt_emb` 第 `[lang_len:200]` 行是 padding token (sentencepiece pad_id) 的真实 embedding，**不是 0 vector**。

**重建步骤**（`_build_fake_stage1_with_masks` 新 helper）：

1. **重新 tokenize 恢复 `lang_masks`**。Pi0.5 tokenizer (`PaligemmaTokenizer`, `tokenizer.py:14-48`) 是确定性的，prompt 拼接规则固定为 `f"Task: {cleaned_text}, State: {state_str};\nAction: "`：
   ```python
   from openpi.models.tokenizer import PaligemmaTokenizer
   tokenizer = PaligemmaTokenizer(max_len=200)
   _, lang_masks = tokenizer.tokenize(task_str, state=robot_state_np)   # (200,) bool
   ```
   `task_str = f.attrs['task']`，`robot_state_np = np.array(group['robot_state'])`。
   - 这是一次性 CPU 计算，单步 < 1 ms。
   - **强制一致性自检**（artifact build 启动时一次）：取首 episode 首 step，重 tokenize 得 `lang_tokens`，调 `model.paligemma_with_expert.embed_language_tokens(lang_tokens) * sqrt(2048)` (注意 `embed_prefix:312-314` 的 normalize) → 与 HDF5 `prompt_emb` 在 mask=True 位置上 `torch.allclose(rtol=1e-2)`。fail → 抛 `RuntimeError("offline tokenizer/embed mismatch with HDF5; data was collected with a different tokenizer or model")`。

2. **重建 prefix_embs**（沿用现有 `_build_fake_stage1` 的 concat 逻辑）：
   ```python
   prefix_embs = torch.cat([
       vision_0, vision_1, vision_2,    # 各 (256, 2048)
       prompt_emb,                      # (200, 2048)
   ], dim=0).unsqueeze(0)               # → (1, 968, 2048) bf16 (cast)
   ```

3. **合成 prefix_pad_masks**：
   ```python
   prefix_pad_masks = torch.cat([
       torch.ones(3 * 256, dtype=torch.bool),     # vision: 全 True (相机缺失另议, 见 §6.2.4)
       torch.from_numpy(lang_masks),              # (200,) bool from tokenizer
   ]).unsqueeze(0)                                # (1, 968)
   ```

4. **合成 prefix_position_ids** 和 **prefix_att_2d_masks_4d**：调用 `pi0_pytorch.PI0Pytorch` 的 **public** 实例方法，避免重复实现 attention machinery：
   ```python
   from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
   # 与 embed_prefix 一致：vision + lang 全部 attend (att_masks 全 0)
   prefix_att_masks = torch.zeros(1, 968, dtype=torch.bool)
   prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
   prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
   prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
   ```
   `_prepare_attention_masks_4d` 是 PI0Pytorch 的私有但稳定的内部函数；本 builder 仅消费、不修改。

5. **打包 FakeStage1**：
   ```python
   FakeStage1(
       state=torch.from_numpy(robot_state_np).unsqueeze(0),      # (1, 32)
       prefix_embs=prefix_embs,                                   # (1, 968, 2048)
       prefix_pad_masks=prefix_pad_masks,                         # (1, 968)
       prefix_att_2d_masks_4d=prefix_att_2d_masks_4d,             # (1, 1, 968, 968)
       prefix_position_ids=prefix_position_ids,                   # (1, 968)
   )
   ```
   `_FakeStage1` dataclass 需扩展（现仅 2 字段），保持向后兼容（旧 builder 不读新字段）。

6. **相机缺失处理**：HDF5 `_build_fake_stage1` 当前对缺失相机填 zero (`(256, 2048)`)。本方案沿用，但同时把对应 vision 段的 `prefix_pad_masks` 置为 False。`PerModalityPoolReducer` 见全 False 段直接 omit field，与在线行为一致。

### 6.3 padding 语义与在线/离线 parity

**核心声明**：在线和离线给同一观察生成的 layer-N hidden state 在 `prefix_pad_masks=True` 的位置上**应当相等**（bf16 容差内），masked pool 后的 query_keys 严格相同（除 bf16 → float32 cast 的舍入误差）。

理由：
- prefix_embs 数值来源相同：vision 来自 SigLIP（HDF5 直接存）；lang 来自 `embed_language_tokens(tokens) * sqrt(2048)`（HDF5 存的就是这个）。
- attention mask 决定 padding 位置不参与 valid token 的 attention 计算（4D additive mask 在 padding 列填 -inf）。
- layer N forward 在 valid 位置的输出仅依赖 valid 位置的 K/V，与 padding 位置的 hidden 无关。
- `PrefixReducer` masked mean 只取 `prefix_pad_masks=True` 位置 → 输出与在线相同。

### 6.4 在线/离线一致性验证（hard requirement）

新增集成测试 `tests/cache/test_llm_layer_extract_parity.py @pytest.mark.manual`：
- 加载真实 Pi0.5 checkpoint (`PI05_LIBERO_CHECKPOINT` env var)
- 准备一组 LIBERO obs（含 task, robot_state, 3 张图），跑在线 `model.run_stage1(obs)` → KeyBuilder.collect/build → `keys_online`
- 用 `data_collector` 同步把同一 obs 写到临时 HDF5
- 调 artifact builder 处理该 HDF5 → KeyBuilder.collect/build → `keys_offline`
- 断言：所有 enabled field `torch.allclose(keys_online[f], keys_offline[f], rtol=1e-2, atol=1e-3)`（bf16 容差）

测试 fail → 离线契约被破坏，必须修。Test marker 为 `manual` 因为需要 GPU + checkpoint，CI 不跑；但本地 Verify 阶段必须跑通。

### 6.5 离线代价权衡

- 加载 PI0Pytorch ≈ 5–10 GB VRAM、首次启动 ~30s
- Tokenizer 模型加载 ~1s，每步重 tokenize < 1 ms
- 单 episode 内每步多跑 1 层 forward ≈ 1–2 ms（vs 现有 `cp1_mean_pool` < 0.05 ms）
- 1000 episodes × 100 step × 3 ms ≈ 5 分钟净额外算力（与模型加载相比可忽略）
- 强制 `--workers -1` 避免每个 worker 加载模型（VRAM 不够）

---

## 7. 在线推理代价

| 项目 | cp1_mean_pool | cp1_llm_layer_extract (layer=0) |
|------|---------------|----------------------------------|
| 单步算力 | < 0.1 ms | ~0.5–2 ms (A100/4090 bf16) |
| 单步 FLOPs | ≈ 5e5 | ≈ 7e10 (单层 Gemma 2B forward) |
| 额外 VRAM | 0 | 0（borrow 权重，激活 < 10 MB） |
| D2H 拷贝 | 1 次 | 1 次（与 baseline 同） |

总推理时延占比：在 ~50 ms 推理总时延下增加 ~2%，可接受。

---

## 8. 测试策略

### 8.1 `test_prefix_reducer.py` (~10 cases)

- `PrefixMeanPoolReducer.output_dims == {"vision_0": 2048}`
- `PerModalityPoolReducer.output_dims` 完整 4 字段
- masked mean 正确性：构造 hidden 全 1，pad_mask 一半 False → 输出仍为 1（验证 mask 起效）
- pad_mask 全 True 时退化为普通 mean
- per_modality 在某段 mask 全 False 时 omit（不输出 zero-vector）
- per_modality 段不重叠覆盖整个 prefix
- emitted_fields 与 output_dims keys 一致

### 8.2 `test_llm_layer_key_builder.py` (~14 cases)

- 未 attach_model 调 build → 抛清晰 `RuntimeError("attach_model not called...")`
- attach_model 后 collect / build / clear 全流程
- 输出字段与 reducer.output_dims 严格一致
- 输出 dtype = float32, device = cpu, contiguous = True
- L2 norm（如适用）验证
- `extract_layer=0` 与 `extract_layer=2` 输出 shape 相同但值不同
- `extract_layer<0` 或 `>=len(layers)` 在 attach_model 时报错（实际上限取自 `len(model.paligemma.language_model.layers)`）
- `apply_final_norm=True` 抛 `NotImplementedError`
- 模拟相机缺失：img_mask 全 False 段 → 输出无对应字段
- robot_state 若 enabled → 输出 raw 32-d vector
- CP3 与 CP1 调用路径行为一致（都会触发 forward，但仅 build 时行为，不入历史 — 本 builder 无状态）
- protocol compliance：`isinstance(builder, QueryKeyBuilder)` (`runtime_checkable`)
- **Interceptor hook 单测**：构造 dummy `Policy` + `CacheOrchestrator`（带 `CP1LLMLayerExtractKeyBuilder`），构造 `InferenceInterceptor` 后断言 `builder._layer is not None`（attach_model 已被自动调）
- **Interceptor 兼容性**：同样构造但用 `PlaceholderKeyBuilder`（无 `attach_model`），构造 `InferenceInterceptor` 不抛异常（`hasattr` noop 路径）

### 8.3 `test_orchestrator.py` 增补 (1 case)

- `orchestrator.key_builder is the_builder_passed_in_constructor` → 断言公开 property 返回原引用

### 8.4 `test_config.py` 增补

- 加载 `cp1_llm_layer_extract` + `prefix_mean_pool` YAML，校验通过
- `extract_layer=18` 报错（gemma_2b depth=18，合法范围 0..17）
- `prefix_reducer.type=foo` 报错
- `prefix_mean_pool` + 同时 enable `vision_1` → 报错
- `per_modality_pool` + `vector_dims.vision_0=999` → 报错（dim 不匹配 2048）
- `prefix_mean_pool` + `vector_dims.vision_0=2048` + `keys.vision_0.enabled=true` → 通过

### 8.5 集成测试（`@pytest.mark.manual`，需 GPU + checkpoint）

- **`test_llm_layer_extract_parity.py`**：在线/离线 parity（详见 §6.4）。Verify 阶段必须本地跑通。
- 真实 PI0Pytorch 模型走完整 collect→build→search→fetch 链路
- 校验 layer 0 输出与现成 `output_hidden_states=True` 完整 forward 的 `hidden_states[1]` 等价（同模型同输入）
- offline `_build_fake_stage1_with_masks` 自检：tokenizer 重算 lang_emb 与 HDF5 prompt_emb 在 mask=True 位置 `allclose`

---

## 9. 实验侧（首版仅 baseline 对照，详细 sweep 推后）

### 9.1 实验目标

验证教授假设：用 layer 0 的统一 hidden state 作为 key，**hit 质量是否优于** 现有 `cp1_mean_pool` baseline。指标：相同阈值下的 hit rate、success rate、SR-Hit 曲线。

### 9.2 首版对照组（最小 set）

```
Group  | KeyBuilder           | extract_layer | reducer            | Backend vector_dims
-------+----------------------+---------------+--------------------+--------------------
A0     | cp1_mean_pool        | -             | -                  | vision_0=2048, robot_state=32
A1     | cp1_llm_layer_extract| 0             | prefix_mean_pool   | vision_0=2048, robot_state=32
A2     | cp1_llm_layer_extract| 0             | per_modality_pool  | vision_0/1/2=2048, prompt_emb=2048, robot_state=32
```

**单任务（libero_spatial 单任务）跑首轮**：A0 vs A1 对照。如 A1 不显著优于 A0，layer index 提到 2、5 再对比（属于本任务后续 follow-up，不进首版）。

### 9.3 实验文件位置（artifact_layout 合规）

- 配置：`exp/common/config/llm_layer_extract/<sub>/`（new sub）
- artifact：`exp/common/data/cache_artifacts/<task_suite>/cp1_llm_l{0,2,5}_{reducer}.pkl`
- 实验脚本/runner：复用 `exp/common/run_cache_experiments.py`，无需新写
- 结果：`exp/common/data/llm_layer_extract/<sub>/`、分析 `exp/common/analysis/llm_layer_extract/<sub>/`

> 详细 YAML sweep + run command 文档**留待 G2 通过后另起 L1 任务**（参考 `phase1_libero_10_run_commands.log.md` 模式），不在本 plan 范围。

---

## 10. 风险登记

| Risk | 影响 | 缓解 |
|------|------|------|
| layer 0 单层融合不足 | hit 质量未必优于 mean_pool | 实验维度天然支持 layer index sweep；首版定 baseline 后再 L1 增量 |
| 离线 artifact build 必须加载模型，VRAM 5–10GB | build 慢、不能并行 | 强制 `--workers -1`；artifact 是一次性产物，可接受 |
| Pi0.5 训练 / 加载流程 (`load_pytorch`) 与 artifact builder 解耦差 | builder 需复用 `policy_config.create_trained_policy` 但又不需要全部 transforms | 抽 `_load_model_only` 私有 helper，只取 `model.paligemma_with_expert.paligemma.language_model` 子模块 |
| Lang 段 padding 占大头 | naive mean 失效 | 强制 masked mean，单元测试覆盖 |
| 输出 dtype/device 与下游 backend mismatch | 运行时报错 | 所有路径 cast `cpu().float().contiguous()`；测试覆盖 |
| 与 torch.compile 交互 | layer 0 单独 forward 可能触发 recompile | 首版 `extract_layer` 固定，shape 固定 [1,968,2048]；如观察到 recompile 警告，包 `torch._dynamo.disable()` |
| `attach_model` 协议无强制约束 | 用户漏配会在 build 时报错 | 默认 `_layer is None` 时 build 抛 `RuntimeError("attach_model not called")`；Interceptor 通过新 `orchestrator.key_builder` 公开 property 自动 hook，单测覆盖 |
| KeyBuilder runtime_checkable Protocol | 新方法 `attach_model` 不在协议里 | OK — `attach_model` 是该 builder 私有 hook，Interceptor 用 `hasattr` 软探测 |
| `apply_final_norm=True` 默认禁 | 用户配 True 启动失败 | validate 阶段直接报错；False 默认 |
| 离线 tokenizer/embed 与采集时不一致 | offline key 偏离 online，parity test fail | 启动 self-check（§6.2 step 1 末段）：首 step embed_lang vs HDF5 prompt_emb `allclose`；fail 立即 abort，附明确错误；artifact metadata 记录 `tokenizer_class` / `tokenizer_source` / `tokenizer_max_len` / `checkpoint_dir`，方便日后排查 PaligemmaTokenizer 下载源或本地缓存漂移 |
| 采集 HDF5 缺 `lang_masks` | 离线无法精确还原 prefix_pad_masks | 通过 `attrs['task']` + `step['robot_state']` 重 tokenize 还原（§6.2 step 1）；`PaligemmaTokenizer` 是确定性的，不需要改采集格式 |
| `model._prepare_attention_masks_4d` 私有方法 | 离线 builder 跨私有边界 | 该方法行为稳定（仅做形状/dtype 转换），与 Stage 2 forward 已经依赖；用 `# noqa: SLF001` 标注；如未来重构，更新 builder 同步 |
| 在线/离线 layer-N hidden state 不 bit-equal | query_keys 不可比 | §6.4 parity test 强制要求 `allclose(rtol=1e-2)` （bf16 容差）；fail 即 fail Verify |

---

## 11. 实施顺序（Code 阶段拆分）

1. **Phase 1 — Reducer 基础**：写 `prefix_reducer.py` + 测试，无任何模型依赖，纯 tensor 操作。
2. **Phase 2 — KeyBuilder 主体**：写 `llm_layer_key_builder.py`；`attach_model` 抓 layer 引用；`collect/build` 主路径；mock model 测试。
3. **Phase 3 — Orchestrator 接线**：`orchestrator.py` 加 `key_builder` public property；`test_orchestrator.py` 加单测断言。
4. **Phase 4 — Config 接线**：`config.py` 加 dataclass、validation、factory；`test_config.py` 补全。
5. **Phase 5 — Interceptor 接线**：`interceptor.py` 末尾用新公开 accessor 调 `attach_model`；`test_llm_layer_key_builder.py` 加 Interceptor hook 单测（含兼容性反例）。
6. **Phase 6 — Artifact Builder**：扩展 `_FakeStage1` 字段、`build_in_memory_cache_artifact.py` 加 `_build_fake_stage1_with_masks` + tokenizer self-check + 模型加载分支；写 `@pytest.mark.manual` 集成测试（需 checkpoint）。
7. **Phase 7 — Parity Test**：`tests/cache/test_llm_layer_extract_parity.py` 在线/离线一致性测试 (`@pytest.mark.manual`)；本地必须跑通。
8. **Phase 8 — 文档**：写 `docs/cache/llm_layer_extract.md`（与 `temporal_prune.md` 同结构，含 §6.2/6.3/6.4 的离线契约说明）；更新 `docs/README.md` 索引。
9. **Phase 9 — Verify**：`uv run pytest tests/cache/`、`uv run pytest tests/cache/components/` 全绿；本地手动跑 §8.5 manual 集成测试 + parity test；后进 §6 Verify。

---

## 12. Working Agreement 合规检查

- ✅ 文档位置：plan 在 `logs/`，文档在 `docs/`，无 `exp/` 内设计文档（WA §4）
- ✅ 命名：plan 文件 `.log.md` 后缀（execution_authority §2）
- ✅ Decoupled：interceptor / wrapper / hook 模式（WA §2.5），KeyBuilder 协议已存在；`attach_model` 是软扩展不入协议主签名
- ✅ Backward compatible：现有 5 个 KeyBuilder 不动；新 type 默认关闭（type 默认 `placeholder`）
- ✅ 注释语言：英文（WA §3.2）；docstring 待 Code 阶段写入
- ✅ 索引同步：本 plan 落盘后立即更新 `logs/README.md`（WA §4 红线）
- ✅ Comments 原则：少注释、why 而非 what、有 TODO 加 `(Phase N)` context
- ✅ G1/G2 必经（L2 强制，WA §2.4 / §2.6）

---

## 13. Open Items（不阻塞 G1）

- 是否在 artifact metadata 记录模型 weights 哈希以便 online/offline 强一致校验：留待 follow-up L1
- 是否支持 `extract_layer="last"` 字符串语义（= 17）：YAGNI，留待需求出现
- `prompt_token_scoring` reducer：留待首版有信号后 L1
- 多 layer 同时输出（多字段，每字段一层）：留待大规模实验阶段需要

## Review Log

### G2 Round 1 — Reviewer — APPROVED — 2026-04-22 21:56 CDT

- No blocking findings. Independent review confirmed plan/code consistency, docs/index sync, and passing non-manual test coverage for the implemented scope.
- Non-blocking Suggestion Keep `tests/cache/test_llm_layer_extract_parity.py` in the Verify checklist as a required local run whenever the tokenizer source, checkpoint-loading path, or HDF5 collection contract changes — reasoning: the online/offline parity guarantee is the main correctness risk of this feature and cannot be fully delegated to lightweight unit tests.

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-04-22 22:01 CDT

- [Blocking] [Concern] Approved test plan §8.5 was not implemented in full. The codebase adds `tests/cache/test_llm_layer_extract_parity.py`, but the other planned real-model checks are absent: there is no test covering the full `collect -> build -> search -> fetch` chain with the new builder, and no test comparing the builder's layer-0 replay against a full-model `output_hidden_states=True` reference (`hidden_states[1]`). — reasoning: these were part of the approved G1 validation strategy for the highest-risk path (real-model parity and end-to-end integration), so G2 cannot treat the current test set as complete.
- [Blocking] [Concern] The new offline guardrail `_self_check_tokenizer_consistency()` is not directly exercised by tests. Current coverage proves `_build_fake_stage1_with_masks()` on synthetic Stage1-derived HDF5 and proves unit-level config/builder behavior, but it does not prove that the actual self-check path trips correctly on mismatched prompt embeddings or runs successfully in the real artifact-build path. — reasoning: this self-check is the main defense against silently building incompatible offline artifacts; leaving it untested weakens the "offline and online keys are comparable" contract the feature depends on.

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-04-22 22:11 CDT

- [Non-blocking] [Concern] G2 Round 2 concern #2 is now closed. `tests/exp/test_llm_layer_extract_self_check.py` adds direct coverage for `_self_check_tokenizer_consistency()` pass/fail behavior and verifies that `build_artifact(..., builder_type="cp1_llm_layer_extract")` invokes the self-check path. Reviewer independently ran `uv run pytest tests/exp/test_llm_layer_extract_self_check.py` and observed `5 passed`. — reasoning: the offline guardrail is now exercised in isolation and at the artifact-builder call site.
- [Blocking] [Concern] The newly added real-model `collect -> build -> search -> fetch` manual test does not yet prove distinct-query retrieval. In `tests/cache/test_llm_layer_extract_parity.py`, `_synthetic_observation()` always seeds `np.random.default_rng(42)`, so `obs_a` and `obs_b` are initially identical; the attempted divergence via `obs_b._replace(...)` is a no-op because `Observation` is not a namedtuple and does not expose `_replace`. As written, the test can pass while only verifying a degenerate "query retrieves itself" path rather than two genuinely distinct entries. — reasoning: this leaves G2 Round 2 concern #1 only partially closed; the approved plan required a meaningful real-model end-to-end integration check, not a vacuous same-query retrieval.

### G2 Round 2 — Executor — 2026-04-22 22:30 CDT

- **Accepted (Round 1 non-blocking suggestion)** — parity test 已绑定到 Verify checklist。改动：plan §11 Phase 9 Verify 步骤明确包含 `tests/cache/test_llm_layer_extract_parity.py` 必须本地跑通；`docs/cache/llm_layer_extract.md` §6 也写明 "强制验证" + 触发命令；`reducer_params` metadata 把 `tokenizer_source` / `checkpoint_dir` / `tokenizer_max_len` / `tokenizer_class` / `config_name` 都纳入了，便于将来定位漂移。本轮代码无需进一步改动以对应该建议——文档与流程已闭环。
- **Accepted (Round 2 #1 — plan §8.5 缺两个 manual test)** — 已补齐 plan §8.5 列出的两条遗漏：
  - `tests/cache/test_llm_layer_extract_parity.py::test_layer0_partial_replay_matches_full_forward_hidden_states` (manual): 同一 Stage1 输入分别走 KeyBuilder 单层 replay 与 HF 完整 forward (`output_hidden_states=True`)，断言 `hidden_states[1]` 在 mask=True 位置 `allclose(rtol=1e-2, atol=1e-2)`。把 KeyBuilder._extract 的 layer-0 partial replay 锁死到 Stage 2 实际算的同一向量。
  - `tests/cache/test_llm_layer_extract_parity.py::test_full_chain_with_real_model_collect_build_search_fetch` (manual): 用真实模型 + `InMemoryBackend` + `CacheOrchestrator` 走完整 `collect → build → insert → check (search + judge + fetch)`：插两条不同 prompt 的 entry，分别用各自 obs 查询，断言每次 FULL_HIT 命中正确的 entry_id 和 payload。完整覆盖与现有 cache 框架的集成。
  - 两测试都标 `@pytest.mark.manual`，需 `PI05_CHECKPOINT_DIR` + GPU；CI 不跑，本地 Verify 必须跑。`uv run pytest tests/cache/test_llm_layer_extract_parity.py -v` 现显示 **4 skipped** (2 原有 + 2 新增)，全部正确 discover。
- **Accepted (Round 2 #2 — `_self_check_tokenizer_consistency` 无直接测试覆盖)** — 新增 `tests/exp/test_llm_layer_extract_self_check.py`，5 个 unit test (CI-runnable，无 checkpoint 依赖)：
  - `test_self_check_passes_on_matching_prompt`: 正例，aligned prompt_emb 不抛
  - `test_self_check_raises_on_mismatched_prompt`: **反例** — 故意把 prompt_emb 设成偏离 mock embed 的值，断言 `RuntimeError` 含 `"self-check failed"`，证明 fail-fast 路径生效
  - `test_self_check_ignores_padding_positions`: 边界 — padding 位置故意填脏数据，必须不触发（验证 mask=True 才算数）
  - `test_self_check_skips_when_prompt_is_empty`: 边界 — 全 padding prompt 走 warning 路径，不抛
  - `test_self_check_invoked_in_artifact_build`: **集成** — 用 monkeypatch 把 `_load_pi05_for_llm_extract` / `_create_builder` / `_process_episode_with_model` / `_self_check_tokenizer_consistency` 全 stub，断言 `build_artifact` 在 cp1_llm_layer_extract 分支下确实调到了 self-check（次数 == 1）。证明真实 artifact-build 路径会触发该守卫。
  - 5 个 test 全过；mock 用确定性 tokenizer + embedder（id=7→ones×√2048, id=0→zeros）覆盖 self-check 判定逻辑，不需要真实模型。
- **Local test output**：`uv run pytest tests/cache/ tests/exp/` → **571 passed, 4 skipped (manual), 0 regression**（旧 566 + 5 self-check 新；2 原 manual + 2 新增 manual）。完整覆盖 plan §8.1–§8.5。

### G2 Round 3 — Executor — 2026-04-22 22:48 CDT

- **Acknowledged (Round 3 #1 — self-check 已闭环)** — 同意 reviewer 判定。Round 2 concern #2 关闭确认；本条无需代码改动。
- **Accepted (Round 3 #2 — 端到端测试退化为自查自)** — 已核实 reviewer 事实主张：
  - `Observation` 在 `src/openpi/models/model.py:82-83` 用 `@struct.dataclass` 装饰（flax struct，frozen），方法名是 `replace()` 而非 `_replace()`。`hasattr(obs_b, "_replace")` 永远 False → 原代码 `obs_b._replace(...) if hasattr(obs_b, "_replace") else obs_b` 静默退化为 `obs_b = obs_b`，obs_a 与 obs_b 完全相同（同 seed=42 + 同 prompt）。
  - 实际行为：测试只验证"用 entry_a 的 obs 能查到 entry_a"，但无法区分 entry_b。如果 search 总返回第一条 entry，原测试也会通过 — 等于没测分辨率。
  - 修复 (`tests/cache/test_llm_layer_extract_parity.py`)：
    1. `_synthetic_observation()` 增加 `seed` 与 `task` 关键字参数（默认值与原行为一致），让 obs 可控且可分辨。
    2. `test_full_chain_with_real_model_collect_build_search_fetch` 用 `seed=42` + 抓取任务、`seed=99` + 开抽屉任务构造 obs_a / obs_b，images / state / prompt 三个维度全部不同。
    3. **加 pre-condition assertion**：`assert not torch.allclose(keys_a["vision_0"], keys_b["vision_0"], ...)`，在跑 retrieval 之前先证明两条 key 真的不同；防止将来 synthetic obs 又意外塌成一致而测试再次"通过"。
    4. retrieval assertion 加错误提示，明确指出"search picked the wrong entry"，便于失败定位。
  - 现 `uv run pytest tests/cache/test_llm_layer_extract_parity.py -v` → 4 manual tests 全部正常 discover + skip（无 checkpoint 时）。回归：`uv run pytest tests/cache/ tests/exp/` → **571 passed, 4 skipped, 0 regression**。
  - 该测试在 §6 Verify 阶段必须在真实 checkpoint 下跑通才算 plan §8.5 端到端要求达成。

### G2 Round 4 — Reviewer — APPROVED — 2026-04-22 22:17 CDT

- No blocking findings. G2 Round 3 concern is now closed: `tests/cache/test_llm_layer_extract_parity.py` makes the end-to-end real-model retrieval case genuinely discriminative by varying both `seed` and `task`, and by asserting the two built keys differ before retrieval.
- Reviewer independently ran `uv run pytest tests/cache/test_llm_layer_extract_parity.py tests/exp/test_llm_layer_extract_self_check.py` and observed `5 passed, 4 skipped`; also ran `uv run pytest tests/cache/ tests/exp/` and observed `571 passed, 4 skipped`. The four skipped tests are the expected `@pytest.mark.manual` real-checkpoint probes.
- Non-blocking Suggestion Keep the precondition `keys_a != keys_b` assertion in the manual full-chain test even if the synthetic-observation helper is refactored later — reasoning: it is a cheap guard against the test silently degrading back into a same-query retrieval check.

### G2 Round 4 — Executor — 2026-04-22 22:55 CDT

- **Acknowledged (Round 4 #1 — APPROVED)** — verdict 收到，G2 关闭。
- **Accepted (Round 4 NB suggestion — 保留 keys_a != keys_b 前置 assertion)** — 同意 reviewer 判定。该断言已落地（`tests/cache/test_llm_layer_extract_parity.py::test_full_chain_with_real_model_collect_build_search_fetch` 行 "Pre-condition: A's and B's keys are genuinely different"）；将作为分辨力守卫长期保留，未来 helper 重构时不可移除。本轮代码无需进一步改动。
