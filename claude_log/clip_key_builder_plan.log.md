# CLIP KeyBuilder 实现计划

> 状态: `Plan`
> 日期: 2026-04-10

## 1. 目标

新增一个基于 CLIP 视觉编码器的 KeyBuilder，用 open_clip 对模型输入图片生成低维向量作为 cache key。vision 字段走 CLIP 编码，prompt_emb 和 robot_state 沿用现有 `_CP1BaseKeyBuilder` 的逻辑（从 `stage1.prefix_embs` slice + mean pool）。

## 2. 设计概要

### 2.1 数据流

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
            ├─ 暂存 stage1.prefix_embs → 用于 prompt_emb (mean pool)
            ├─ 暂存 stage1.state       → 用于 robot_state (raw)
            └─ 暂存 input_images       → 用于 vision_0/1/2 (CLIP encode)
          
          CLIPKeyBuilder.build(CP1)
            │
            ├─ vision_0: input_images["base_0_rgb"]       → CLIP → [embed_dim]
            ├─ vision_1: input_images["left_wrist_0_rgb"] → CLIP → [embed_dim]
            ├─ vision_2: input_images["right_wrist_0_rgb"]→ CLIP → [embed_dim]
            ├─ prompt_emb: prefix_embs[768:] → mean pool  → [2048]
            └─ robot_state: state[0]                      → [32]
```

### 2.2 CLIP 图片编码细节

open_clip API:
```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device="cuda"
)
model.eval()

# 输入: uint8 numpy (224, 224, 3)
# → PIL Image → preprocess (Resize, CenterCrop, ToTensor, Normalize)
# → tensor [1, 3, 224, 224]
# → model.encode_image(tensor, normalize=True) → [1, embed_dim]

# embed_dim 可从 model.visual.output_dim 获取（ViT-B-32=512, ViT-L-14=768）
```

### 2.3 图片槽位映射

| 图片槽位 | input_images key | 输出字段 |
|----------|-----------------|---------|
| base cam | `base_0_rgb` | `vision_0` |
| left wrist | `left_wrist_0_rgb` | `vision_1` |
| right wrist | `right_wrist_0_rgb` | `vision_2` |

遵循 `image_extract.py` 的 `MODEL_IMAGE_KEYS` 顺序。

### 2.4 enabled_fields 控制

与现有 builder 一致，通过构造函数 `enabled_fields: list[str] | None` 控制。
例如 `enabled_fields=["vision_0", "robot_state"]` 时，只输出这两个 key，不执行多余的 CLIP encode。

### 2.5 YAML 配置

`KeyBuilderConfig` 新增 CLIP 专用字段：

```yaml
key_builder:
  type: clip                        # 新增类型
  clip_model_name: ViT-B-32         # open_clip 模型名
  clip_pretrained: openai           # open_clip pretrained tag
```

对应 dataclass 扩展：

```python
@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"
    clip_model_name: str = "ViT-B-32"       # 仅 type=clip 时使用
    clip_pretrained: str = "openai"          # 仅 type=clip 时使用
```

### 2.6 vector_dims 配置示例

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

## 3. 依赖分析

### 3.1 新增依赖: `open-clip-torch`

PyPI 包名: `open-clip-torch`
import 名: `open_clip`

open_clip 的传递依赖:
- `torch` — 项目已有 `torch==2.7.1` ✅
- `torchvision` — **项目当前未安装，open_clip 需要它做图片预处理**
- `timm` — open_clip 内部使用 timm 做 vision backbone ✅ (open_clip 自带)
- `pillow` — 项目已有 `pillow>=11.0.0` ✅
- `huggingface_hub` — 项目已有（transformers 传递依赖）✅

### 3.2 冲突风险评估

| 依赖 | 现有版本 | open_clip 要求 | 冲突风险 |
|------|---------|---------------|---------|
| `torch` | ==2.7.1 | >=1.9 | **无冲突** |
| `torchvision` | 未安装 | 需要安装 | **需新增**，版本需与 torch 2.7.1 匹配 |
| `transformers` | ==4.53.2 | 不直接依赖 | **无冲突** |
| `pillow` | >=11.0.0 | >=8.0 | **无冲突** |
| `timm` | 未直接安装 | open_clip 自带 | **需确认是否与 transformers 的 timm 冲突** |
| `jax` | 0.5.3 | 不相关 | **无冲突** |

### 3.3 需要安装的包

```toml
# pyproject.toml [project.dependencies] 新增:
"open-clip-torch>=2.26.1",
```

`open-clip-torch` 会自动拉入 `torchvision` 和 `timm`。需要验证:
1. `torchvision` 版本是否与 `torch==2.7.1` 兼容（torch 2.7.1 对应 torchvision 0.22.1）
2. `timm` 是否与现有 `transformers==4.53.2` 产生版本冲突

**验证方法**: 在安装前先运行 `uv add open-clip-torch --dry-run` 查看依赖解析结果。

### 3.4 GPU 显存开销

| CLIP 模型 | 参数量 | 估计显存 |
|-----------|-------|---------|
| ViT-B-32 | ~88M | ~350 MB |
| ViT-L-14 | ~304M | ~900 MB |

Pi0.5 模型本身约 3B 参数，占 ~6-12 GB。ViT-B-32 的 350 MB 额外开销可接受。

## 4. 改动文件清单

### 4.1 新建文件

| 文件 | 用途 |
|------|------|
| `src/openpi/cache/components/clip_key_builder.py` | CLIPKeyBuilder 类实现 |
| `exp/build_clip_cache_artifact.py` | 从 HDF5 构建 CLIP pkl artifact |

### 4.2 修改文件

| 文件 | 改动内容 |
|------|---------|
| `src/openpi/cache/config.py` | (1) `KeyBuilderConfig` 新增 `clip_model_name`, `clip_pretrained` 字段 (2) `_build_key_builder()` 新增 `"clip"` 分支 (3) `_valid_key_builder_types` 新增 `"clip"` (4) `validate_cache_config()` 新增 CLIP 校验：至少一个 vision 字段 enabled |
| `scripts/serve_policy.py` | 所有 3 处创建 InferenceInterceptor 的路径，当 `key_builder.type == "clip"` 时强制 `collect_images = True` |
| `pyproject.toml` | 新增 `open-clip-torch` 依赖 |

### 4.3 不修改的文件

- `key_builder.py` 中的现有 builder — 不动
- `interceptor.py` — 不动（已支持 `collect_images` + `input_images` 传递）
- `orchestrator.py` — 不动（`collect(**stage_outputs)` 已透传 `input_images`）
- `in_memory_backend.py` — 不动（pkl 格式兼容）
- `storage_types.py` — 不动
- `data_collector.py` / `collection_policy.py` — 不动

## 5. 实现步骤

### Phase 1: 依赖安装与验证

1. `uv add open-clip-torch --dry-run` 检查依赖冲突
2. 确认无冲突后 `uv add open-clip-torch`
3. 验证 `python -c "import open_clip; print(open_clip.list_pretrained())"` 正常

### Phase 2: CLIPKeyBuilder 实现

文件: `src/openpi/cache/components/clip_key_builder.py`

```python
# ── 共享 helper（在线 builder + 离线 artifact 脚本共用）──

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
        # 只记录参数，不加载模型（lazy init）
        self._clip_model_name = clip_model_name
        self._clip_pretrained = clip_pretrained
        self._enabled = set(enabled_fields) if enabled_fields else None
        # CLIP 模型和 preprocess，延迟到第一次 collect() 时加载
        self._clip_model = None
        self._preprocess = None
        self._device = None      # 从 stage1 tensor 推断
        self._embed_dim = None   # model.visual.output_dim

    def _ensure_model_loaded(self, device: torch.device):
        """Lazy init: 首次调用时从 stage1 tensor 推断设备并加载 CLIP。"""
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
        # 从 stage_outputs["stage1"] 暂存 prefix_embs, state → self._cache (GPU tensor)
        # 首次调用时从 stage1.prefix_embs.device 推断设备，lazy load CLIP 模型
        if "stage1" in stage_outputs:
            self._ensure_model_loaded(stage_outputs["stage1"].prefix_embs.device)
        # 从 stage_outputs["input_images"] 暂存图片 dict → self._images (私有，CPU numpy)

    def build(self, checkpoint_id) -> dict[str, torch.Tensor]:
        # vision_*: 遍历 enabled vision 字段，从 self._images 取对应图片
        #           若图片缺失（image_mask=False），跳过该字段（不输出零向量）
        #           有图片则 CLIP encode → [embed_dim] CPU float32
        # prompt_emb: clip_prompt_key_from_tokens(prefix_embs prompt 段)
        # robot_state: clip_state_key(state[0])

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        # 只暴露 GPU tensor (prefix_embs, state)，不暴露 self._images
        return self._cache

    def clear(self):
        self._cache.clear()
        self._images = None
```

关键实现细节:
- **设备推断**: 不在构造函数传 device。首次 `collect()` 时从 `stage1.prefix_embs.device` 推断，lazy load CLIP 模型。后续复用。
- 图片预处理: numpy uint8 → PIL Image → preprocess transform → GPU tensor
- 批量编码: 如果多个 vision 字段 enabled，stack 成 batch 一次 forward
- L2 normalize: 使用 `model.encode_image(batch, normalize=True)`
- **共享 helper**: `clip_prompt_key_from_tokens()` 和 `clip_state_key()` 同时被在线 builder 和离线 artifact 脚本调用，避免逻辑分叉
- `cached_data` 只含 GPU tensor（prefix_embs, state），`input_images` 存私有字段 `self._images`，不暴露给 Gate/Judge
- 缺失图片槽位: `build()` 跳过该 vision 字段，不产出零向量（与 InMemoryBackend 融合逻辑兼容）

### Phase 3: Config 集成（Phase B，可延后至离线评估通过后）

修改 `config.py`:
1. `KeyBuilderConfig` 加 `clip_model_name` 和 `clip_pretrained` 字段
2. `_build_key_builder()` 加 `"clip"` 分支，传入 model name + pretrained + enabled_fields（不传 device，builder 自行 lazy init）
3. `_valid_key_builder_types` 加 `"clip"`
4. `validate_cache_config()`: clip builder 要求至少一个 vision 字段 enabled（否则没意义）
5. artifact 兼容性校验：不改 `_build_backend()` 签名（它只接收 `BackendConfig`）。校验提到调用者层——在 `build_cache_components()` 和 `build_shared_storage()` 中，`_build_backend()` 返回后、函数返回前，新增 `_validate_artifact_metadata(backend, config.key_builder)` 调用。该函数读取已加载的 artifact metadata（`clip_model_name`, `clip_pretrained`），与 `KeyBuilderConfig` 比对，不匹配则 fail-fast。`build_per_connection_components()` 不需要校验（它复用已校验的 shared_storage）。不改 `InMemoryBackend.load_artifact()` 和 `_build_backend()` 的接口。

修改 `serve_policy.py`:
6. **所有**创建 `InferenceInterceptor` 的路径（当前有 3 处：约 L201, L243, L251）都要覆盖：在读取完 cache config 之后、创建 interceptor 之前，统一判断 `if cache_config.key_builder.type == "clip": collect_images = True`。不能只改一个分支。

### Phase 4: pkl Artifact 构建脚本（Phase A，优先实现）

文件: `exp/build_clip_cache_artifact.py`

**不复用** `_build_fake_stage1()`，直接从 HDF5 读取三个来源:
- 图片: 从 `step_xxxx/input_images/{base_0_rgb, ...}` 读取 uint8 numpy → CLIP encode
- prompt_emb: 从 `step_xxxx/prompt_emb` 读取 → `clip_prompt_key_from_tokens()` (共享 helper) → [2048]
- robot_state: 从 `step_xxxx/robot_state` 直接读取 → `clip_state_key()` (共享 helper) → [32]

**prompt/state 逻辑共享**: 离线脚本和在线 CLIPKeyBuilder 调用同一组 helper 函数 (`clip_prompt_key_from_tokens`, `clip_state_key`)，避免两套实现分叉。

**单进程 GPU batch 编码**（不用 ProcessPoolExecutor）:
1. 主进程加载一次 CLIP 模型到 GPU
2. 顺序遍历 HDF5 文件，跳过无 `input_images` group 的文件（打 warning，最后报告可用/不可用文件数量统计）
3. 收集图片，按 batch 编码（如 batch_size=64）
4. prompt_emb / robot_state 从 HDF5 直接读取

**artifact metadata 扩展**: 除现有的 `key_builder_type`, `checkpoint_id`, `vector_dims`, `entries` 外，新增:
- `clip_model_name`: CLIP 模型名（如 `"ViT-B-32"`）
- `clip_pretrained`: pretrained tag（如 `"openai"`）

CLI:
```bash
uv run exp/build_clip_cache_artifact.py \
    --data-dir data/libero_spatial \
    --clip-model ViT-B-32 \
    --clip-pretrained openai \
    --output data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl
```

### Phase 5: 测试

1. `test_clip_key_builder.py`: 用 mock CLIP 模型测试:
   - 输出维度正确性
   - 缺图跳过（不产出零向量）
   - `enabled_fields` 过滤
   - `cached_data` 只含 tensor，不含 numpy image
2. `test_config_clip.py`: 测试 config 解析:
   - `"clip"` 类型正常解析
   - 无 vision 字段 enabled 时校验报错
   - `clip_model_name` / `clip_pretrained` 正确传递
3. artifact 脚本测试: 用小型测试 HDF5 跑构建，验证输出格式和 metadata

### Phase 6: 端到端验证

1. 构建 artifact: 用已有 HDF5 数据跑 `build_clip_cache_artifact.py`
2. 加载验证: `InMemoryBackend.load_artifact()` 加载 pkl
3. 离线检索评估: 比较 CLIP artifact 与现有 cp1_mean_pool artifact 的检索质量
4. （Phase B）配置运行: 用新 YAML 配置启动 `serve_policy.py --cache_config clip_cache.yaml`
5. 维度检查: 确认 vector_dims 与 CLIP 输出一致

## 6. YAML 配置完整示例（ViT-B-32 + vision_0 + robot_state）

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
  clip_model_name: ViT-B-32       # 可换 ViT-L-14 等
  clip_pretrained: openai          # 可换 laion2b_s34b_b79k 等

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

## 7. 实施阶段划分

| 阶段 | 包含 Phase | 目标 | 前置条件 |
|------|-----------|------|---------|
| **Phase A (离线)** | 1, 2, 4, 5 | 依赖安装 + CLIPKeyBuilder + artifact 脚本 + 测试 + 离线评估 | 无 |
| **Phase B (在线)** | 3, 6.4-6.5 | config.py 集成 + serve_policy.py 接入 + 端到端验证 | Phase A 离线评估结果满意 |

Phase B 可在 Phase A 验证结果满意后再做。

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `open-clip-torch` 与现有依赖冲突 | 安装失败 | Phase 1 先 dry-run 验证 |
| CLIP 额外 GPU 显存 | OOM | 默认用 ViT-B-32 (350MB)；如紧张可将 CLIP 放 CPU |
| CLIP 推理延迟 (3张图 ~10-15ms) | 增加 cache check 时间 | 仅 enabled 的 vision 字段才编码；可 batch forward |
| HDF5 缺少 input_images | artifact 构建跳过该文件 | 脚本自动检测并报告可用/不可用文件统计 |
| preprocess resize 对已 224x224 图片多余 | 轻微延迟 | 图片已是 224x224，resize 实际是 no-op |
| 同维度不同 CLIP 权重语义不兼容 | 静默检索质量下降 | artifact metadata 记录 clip_model_name + clip_pretrained，config.py 启动期自动比对 fail-fast |

---

## 审查（历史记录，结论已纳入上方 plan 原文）

> 审查日期: 2026-04-10
> 状态: `Historical` — 所有被采纳的修改已反映在 Section 1-8 中，以下内容仅保留作审查记录。
> 审查范围: `CLAUDE.md`、`docs/cache_system_architecture_chinese.md`、`src/openpi/cache/`、`src/openpi/collect/`、`exp/build_in_memory_cache_artifact.py`

### 1. 主要疑虑

#### 疑虑 1: 运行时再引入一套 CLIP 编码，可能直接吃掉 CP1 cache 的收益

- 原因:
  - 当前 CP1 发生在 `run_stage1()` 之后，Stage 1 的 SigLIP 计算已经付过成本。
  - 现有 `cp1_*` builder 都是复用 `stage1.prefix_embs` 做池化，几乎不引入额外模型计算。
  - 这个方案会在 miss 路径和 hit 路径上都额外执行一次 `open_clip` 图像编码，属于新增前向，不是“复用已有中间结果”。
- 建议:
  - 先补一组基线测量，再决定是否值得做。至少比较:
    1. `cp1_mean_pool` build 时间
    2. CLIP encode 时间
    3. 低维检索节省的搜索时间
    4. 命中率变化是否足以覆盖新增编码成本
- 问题:
  - 目标是“提升命中质量”还是“降低总延迟”？
  - 如果 CLIP 编码本身就要几毫秒到十几毫秒，这个 tradeoff 是否还能成立？

#### 疑虑 2: 方案把 `collect_images=True` 当成 config 可校验项，但它现在不是 `CacheConfig` 的一部分

- 原因:
  - `collect_images` 目前是 `scripts/serve_policy.py` 的 CLI 参数，不在 `src/openpi/cache/config.py` 的 dataclass 树里。
  - 因此 `validate_cache_config()` 无法知道用户启动服务时有没有传 `--collect_images`。
  - 计划里写“在 `validate_cache_config()` 里校验 clip builder 要求 `collect_images=True`”，和现状接口不一致。
- 建议:
  - 二选一:
    1. 把“是否需要 input_images”正式纳入 cache config；或者
    2. 在 `serve_policy.py` 做启动期 fail-fast 校验，而不是放到 `validate_cache_config()`。
- 问题:
  - 你希望这个约束属于 YAML 层，还是属于 server 启动参数层？

#### 疑虑 3: artifact 构建依赖 HDF5 里的 `input_images`，但现有采集数据未必有这个 group

- 原因:
  - `src/openpi/collect/data_collector.py` 只有在 `input_images` 非空时才写 `step_xxxx/input_images/...`。
  - `docs/data_collection_guide.md` 当前主文档也没有把 `input_images` 作为标准输出字段写进主流程说明。
  - 也就是说，历史 HDF5 很可能只有 `vision_*` / `prompt_emb` / `robot_state`，没有原图。
- 建议:
  - 在计划里明确数据前提:
    - 只支持新采集且带 `input_images` 的 HDF5；或者
    - 提供降级策略 / 检查脚本，提前扫描哪些文件可用于 CLIP artifact。
- 问题:
  - 你打算复用现有历史数据，还是接受重新采集一批带 `input_images` 的数据？

#### 疑虑 4: 直接照搬 `exp/build_in_memory_cache_artifact.py` 的多进程模式，CLIP 版本很容易出资源问题

- 原因:
  - 现有脚本是每个 worker 进程内创建 builder。
  - 如果 CLIP builder 在 worker 内加载 GPU 模型，多进程会重复占用显存，基本会炸。
  - 即便放 CPU，多进程重复加载模型也会带来很高的模型初始化和内存开销。
- 建议:
  - 不要直接沿用当前“每进程一个 builder”的模式。
  - 更稳妥的方向是:
    1. 单进程 GPU 批量编码图片；
    2. 或者多进程只负责 HDF5 读取，编码集中在单进程完成；
    3. 至少先把 CLIP artifact 脚本设计成单进程可跑通，再讨论并行。
- 问题:
  - artifact 构建优先目标是“快”还是“稳”？
  - 机器默认是单卡 GPU，还是你打算 CPU-only 构建？

#### 疑虑 5: 仅靠 `vector_dims` 不足以识别 artifact 和运行时 builder 是否真正匹配

- 原因:
  - `src/openpi/cache/backends/in_memory_backend.py` 的 `load_artifact()` 只校验 `vector_dims`。
  - 但不同 CLIP 模型 / 不同 pretrained tag 可能输出同维度向量，例如同样 512 维，但语义空间不同。
  - 这样会出现“维度合法但语义不兼容”的静默错误。
- 建议:
  - artifact metadata 至少新增:
    - `clip_model_name`
    - `clip_pretrained`
    - `enabled_fields`
    - 可能的话再加一个 `builder_config_hash`
  - 加载时做严格匹配，而不是只比维度。
- 问题:
  - 是否允许一个 artifact 在不同 pretrained tag 之间复用？
  - 如果不允许，就应该在格式层直接禁止。

#### 疑虑 6: `cached_data` 的接口契约目前默认是 tensor，计划里混入 `numpy image` 会破坏协议边界

- 原因:
  - `QueryKeyBuilder` Protocol 里 `cached_data` 的类型和语义都写的是“原设备上的 tensor，给 gate/judge 读”。
  - 计划里的 `collect()` 同时缓存 `prefix_embs/state` 和 `input_images`，其中 `input_images` 是 CPU numpy。
  - 这会让 `cached_data` 变成“既有 GPU tensor 又有 CPU numpy”的混合容器，和当前协议不一致。
- 建议:
  - 把 `input_images` 放在 builder 的私有缓存里，只让 `build()` 自己消费。
  - `cached_data` 继续只暴露 gate/judge 可能会读的 tensor 数据，避免把协议做脏。
- 问题:
  - 你是打算扩展 `QueryKeyBuilder` 协议，还是保持 gate/judge 视角不变？

#### 疑虑 7: 缺失相机槽位的语义没有定义清楚，尤其是 `vision_2`

- 原因:
  - `src/openpi/shared/image_extract.py` 会按 `image_mask` 过滤，只返回有效图像。
  - 现有基于 `prefix_embs` 的 builder 会按固定切片产出 `vision_0/1/2`，即使某个槽位在环境里常年为空，也有固定位置语义。
  - CLIP 版本如果遇到 `right_wrist_0_rgb` 不存在，是输出零向量、跳过该 field，还是启动期报错，计划里没定。
- 建议:
  - 在计划中把策略写死，不要等实现时再拍脑袋。
  - 我倾向于:
    - query 和 artifact 都统一“缺失则不产出该 field”；
    - 同时要求配置层不要给长期缺失的槽位分配关键权重。
- 问题:
  - 目标环境是否稳定拥有 3 个相机槽位？
  - 如果 LIBERO 常缺 `vision_2`，是否还要把它放进默认示例配置？

#### 疑虑 8: 预训练权重的获取方式没有说明，首次运行可能在服务端或实验机上直接失败

- 原因:
  - `open_clip.create_model_and_transforms(..., pretrained=...)` 通常依赖本地 cache 或在线下载权重。
  - 当前计划只讨论了 pip 依赖，没有讨论模型权重来源、缓存目录、离线环境策略。
  - 这个仓库很多场景是远端推理、容器、集群、或无公网环境，这一点不能留空。
- 建议:
  - 在计划里补充:
    - 首次下载是否允许联网
    - 权重缓存放哪里
    - 离线机器如何提前准备权重
    - 失败时的报错和提示信息
- 问题:
  - 服务部署环境是否保证能访问 open_clip 的权重源？

### 2. 额外建议

#### 建议 1: 先把 CLIP 方案拆成“离线 artifact 实验”和“在线推理接入”两阶段

- 原因:
  - 现在最大的未知数不是“能不能接进 config”，而是“值不值得在线跑”。
  - 先离线做 artifact 和检索评估，可以更快回答命中质量是否提升。
- 建议:
  - Phase A: 只做离线 artifact + 检索实验，不接 `serve_policy.py`
  - Phase B: 只有当质量收益明显时，再考虑在线 key_builder

#### 建议 2: 明确 CLIP builder 的职责边界，不要继续复用“伪 stage1 重建”作为长期接口

- 原因:
  - 现有 `_build_fake_stage1()` 是为复用 `cp1_*` builder 服务的。
  - CLIP 方案真实依赖的是“prompt/state + raw input_images”，不是“重建整份 SigLIP prefix_embs”。
- 建议:
  - 可以短期复用旧逻辑，但长期更干净的做法是给 artifact 构建脚本单独准备结构化输入，而不是硬拼一个假 `Stage1Output`。

#### 建议 3: 把测试计划补进本文，不要只写实现步骤

- 原因:
  - 这个仓库对 cache 子系统已经明确标了“高危、未充分集成验证”。
  - 新增一个外部视觉模型，如果没有测试，很容易把问题拖到真实服务阶段才暴露。
- 建议:
  - 至少补 4 类测试:
    1. `config.py` 的 clip 分支和校验分支
    2. `clip_key_builder.py` 的维度、缺图、enabled_fields 行为
    3. artifact 脚本对缺失 `input_images` 的失败信息
    4. `serve_policy.py` 启动期对 `--collect_images` 缺失的 fail-fast

### 3. 我认为当前最需要先回答的几个问题

1. 这个方案的首要目标到底是“检索质量提升”还是“端到端延迟下降”？
2. 在线推理时新增一次 CLIP 编码的成本，是否已经做过粗测？
3. 目标数据集里有多少 HDF5 真的带 `input_images`？
4. `collect_images` 这个前提，最终准备放在 YAML 还是 CLI 层管理？
5. artifact 是否允许同维度但不同 CLIP 权重混用？
6. 缺图槽位的统一语义是什么: 跳过字段、零向量、还是报错？

### 4. 结论

这个计划的方向可以做，但我不建议直接按当前版本开工。最大的两个风险不是“怎么写代码”，而是:

1. 在线 CLIP 编码会不会抵消 cache 收益。
2. 现有数据和配置接口是否真的满足 `input_images` 这条新依赖。

如果这两点不先钉住，后面的实现很容易变成”代码能跑，但系统收益和使用路径都不稳”。

---

## 审查回应（历史记录，结论已纳入上方 plan 原文）

> 回应日期: 2026-04-10
> 状态: `Historical` — 采纳/驳回决定已反映在 Section 1-8 中，以下内容仅保留作决策记录。
> 回应人: Claude (plan 作者)

### 对主要疑虑的逐条回应

#### 疑虑 1: CLIP 编码会吃掉 CP1 收益 — **部分认同，但不阻塞实现**

审查人的观察是对的：现有 `cp1_*` builder 是零额外模型计算（纯 tensor slice + pool），CLIP 引入了新的前向传播。

但审查人混淆了两件事：
1. **这个方案的目标不是”降低总延迟”，而是”用更好的语义表示提升检索质量”**。现有 SigLIP prefix_embs 的 mean pool 是一个 2048 维的粗糙压缩，CLIP 的 512 维 embedding 是专门训练的图像级语义表示，在检索匹配任务上天然更适合。
2. **CLIP 编码的开销需要和 Stage 2 + Stage 3 的节省做对比，而不是和 build 时间做对比**。CP1 命中时跳过的是 Stage 2 (LLM, ~50-100ms) + Stage 3 (flow matching, ~30-50ms)。ViT-B/32 编码 3 张图约 10-15ms，这个 tradeoff 在命中率足够时是正的。
3. 审查人建议”先补基线测量”——这不需要阻塞实现。代码实现和性能评估可以并行，且性能评估本身需要先有可用的 CLIP artifact 才能跑。

**结论**: 认同需要做性能评估，但这是 Phase 5 验证的一部分，不需要前置。先实现离线 artifact 构建，自然就能评估。

#### 疑虑 2: `collect_images` 不在 CacheConfig 中 — **认同，采纳方案 2**

审查人说得对，`collect_images` 是 `serve_policy.py` 的 CLI 参数，不在 YAML 中。

**采纳方案**: 在 `serve_policy.py` 中，当 `key_builder.type == “clip”` 时，**无条件强制** `collect_images = True`。`extract_valid_images()` 的开销可忽略（只是从已有 transform 输出按 mask 切 numpy，无额外计算），所以不需要用户手动传 CLI 参数，也不需要在 `validate_cache_config()` 中校验。

**修改计划**: Phase 3 (config 集成) 中，在 `serve_policy.py` 的 cache 启动分支加一行自动强制逻辑。

#### 疑虑 3: 历史 HDF5 可能没有 `input_images` — **认同，但影响有限**

审查人说的事实正确：历史 HDF5 是否有 `input_images` 取决于采集时是否开启。

但这不影响计划：
1. artifact 构建脚本本来就应该在读取时检查 `input_images` group 是否存在，不存在则 skip 该文件并打 warning。
2. 这是数据层的前提条件，不是代码设计问题。用户需要用带 `input_images` 的数据构建 artifact。

**修改计划**: 在 Phase 4 中明确：artifact 脚本在处理 HDF5 时检查 `input_images` group，缺失则跳过并统计跳过数量，最后报告可用/不可用文件比例。

#### 疑虑 4: 多进程 CLIP 模型加载会炸显存 — **认同，采纳单进程方案**

这是一个好 catch。现有脚本的 `ProcessPoolExecutor` 模式确实不适用于需要加载 GPU 模型的 builder。

**采纳方案**: artifact 构建脚本使用单进程设计：
1. 主进程加载一次 CLIP 模型
2. 顺序遍历 HDF5 文件，批量收集图片
3. 按 batch 编码（例如 batch_size=64）
4. prompt_emb / robot_state 仍从 HDF5 直接读取（不需要 CLIP）

这比多进程更简单也更稳定。数据量不大时速度也够用。

**修改计划**: Phase 4 重写为单进程 + GPU batch 编码。

#### 疑虑 5: `vector_dims` 不足以识别 artifact 匹配 — **部分认同，小改**

审查人的担忧有道理：同为 512 维但 pretrained 不同的 CLIP 模型确实语义不兼容。

但审查人建议的 `builder_config_hash` 过度设计了。现有 artifact 格式已有 `key_builder_type` 字段。

**采纳方案**: artifact metadata 新增 `clip_model_name` 和 `clip_pretrained` 字段。`load_artifact()` 不改（保持只校验 `vector_dims`，因为 backend 不该知道 builder 细节），但 artifact 构建脚本的输出会包含这些字段供人工确认。如果后续需要自动校验，可以在 `config.py` 的启动逻辑中读取 artifact metadata 做匹配。

**修改计划**: Phase 4 中 artifact dict 加 `clip_model_name` 和 `clip_pretrained` 字段。

#### 疑虑 6: `cached_data` 混入 numpy 破坏协议 — **认同，采纳隔离方案**

审查人说得完全对。`cached_data` 的契约是 GPU tensor，给 Gate/Judge 读的。`input_images` 是 CPU numpy，不应暴露在 `cached_data` 中。

**采纳方案**: `input_images` 存在 builder 的私有字段 `self._images` 中，`cached_data` 只暴露 `prefix_embs` 和 `state`（和现有 builder 一致）。`build()` 从 `self._images` 取图片做 CLIP 编码。

**修改计划**: Phase 2 中明确 `cached_data` 只含 tensor，`_images` 是私有字段。

#### 疑虑 7: 缺失相机槽位的语义 — **认同，明确策略**

审查人说得对，需要提前定义。

**策略**: 缺失则不产出该 field。具体来说：
- `extract_valid_images()` 只返回 `image_mask=True` 的槽位
- `build()` 中，如果某个 enabled 的 vision 字段对应的图片不在 `input_images` 中，**跳过该字段**（不输出零向量）
- 这与 InMemoryBackend 的搜索逻辑兼容：缺失字段的 candidate 在融合时该字段不贡献分数
- LIBERO 环境确实常缺 `vision_2`（right_wrist），所以示例配置默认 `vision_2: {enabled: false}`

**修改计划**: Phase 2 中明确：缺图 = 跳过字段，不产出零向量。

#### 疑虑 8: 预训练权重获取 — **驳回，非本计划职责**

审查人把离线部署策略的通用问题算在了这个计划上。open_clip 的权重下载行为和项目中已有的 HuggingFace transformers 完全一致：首次联网下载，后续从 `~/.cache/huggingface/hub` 读缓存。项目已经依赖 `transformers==4.53.2`，Pi0.5 的模型权重本身就需要从 HuggingFace 下载。这不是 CLIP 引入的新问题。

如果部署环境无公网，需要的是项目级的离线权重方案（预下载所有模型到共享存储），不是 CLIP builder 文档里的内容。

**结论**: 不修改计划。如果用户后续有离线部署需求，另开 issue 处理。

### 对额外建议的回应

#### 建议 1: 拆成离线实验 + 在线接入两阶段 — **部分采纳**

认同优先级：离线 artifact + 检索评估 > 在线推理接入。但计划本身已经是这个顺序（Phase 4 artifact 构建在 Phase 3 config 集成之前可以独立运行）。

**调整**: 把 Phase 顺序调整为更明确的两阶段：
- Phase A (离线): 依赖安装 → CLIPKeyBuilder 实现 → artifact 构建脚本 → 构建 artifact 并评估
- Phase B (在线): config.py 集成 → serve_policy.py 接入 → 端到端验证

Phase B 可以在 Phase A 验证结果满意后再做。

#### 建议 2: 不要复用 `_build_fake_stage1()` — **认同**

CLIP builder 需要的是 `input_images` + `prompt_emb` + `robot_state`，不需要重建 `prefix_embs`。artifact 构建脚本应该直接从 HDF5 读取这三个来源，不经过 fake stage1。

**修改**: artifact 脚本不调用 `_build_fake_stage1()`，而是：
- 图片: 从 `step_xxxx/input_images/` 读取
- prompt_emb: 从 `step_xxxx/prompt_emb` 读取 → 重建 prefix_embs 的 prompt 段 → mean pool（或直接从 HDF5 的 flat embedding 做 mean pool）
- robot_state: 从 `step_xxxx/robot_state` 直接读取

#### 建议 3: 补测试计划 — **认同，补充**

**新增 Phase A.5 (测试)**:
1. `test_clip_key_builder.py`: 用 mock CLIP 模型测试维度、缺图跳过、enabled_fields 过滤、cached_data 只含 tensor
2. `test_config_clip.py`: 测试 `”clip”` 类型的 config 解析、校验（缺 vision 字段报错、vector_dims 匹配）
3. artifact 脚本测试: 用小型测试 HDF5 跑 artifact 构建，验证输出格式

---

## 计划修订摘要（历史记录）

根据审查，以下修改已纳入 plan 原文 Section 1-8：

| 修改项 | 来源 | 内容 |
|--------|------|------|
| `collect_images` 自动推断 | 疑虑 2 | `serve_policy.py` 中 clip builder 自动开启 `collect_images` |
| HDF5 缺图处理 | 疑虑 3 | artifact 脚本 skip 无 `input_images` 的文件，报告统计 |
| 单进程 artifact 构建 | 疑虑 4 | 不用 ProcessPoolExecutor，改为单进程 GPU batch |
| artifact metadata 扩展 | 疑虑 5 | 新增 `clip_model_name`, `clip_pretrained` 字段 |
| `cached_data` 隔离 | 疑虑 6 | `input_images` 存私有字段，`cached_data` 只含 tensor |
| 缺图语义 | 疑虑 7 | 缺失图片 = 跳过该 field，不产出零向量 |
| 两阶段拆分 | 建议 1 | Phase A 离线先行，Phase B 在线可延后 |
| 不复用 fake stage1 | 建议 2 | artifact 脚本直接读 HDF5 字段 |
| 补测试计划 | 建议 3 | 新增 Phase A.5 |

以下审查项**驳回**：

| 驳回项 | 来源 | 理由 |
|--------|------|------|
| 阻塞实现等基线测量 | 疑虑 1 | 性能评估需要先有 artifact，实现和评估可并行 |
| 预训练权重离线策略 | 疑虑 8 | 通用问题，非本计划职责，与现有 transformers 权重管理一致 |
| `builder_config_hash` | 疑虑 5 | 过度设计，metadata 字段足够 |
