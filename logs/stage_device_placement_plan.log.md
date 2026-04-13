# Stage Device Placement — 实现计划

**状态**: `In Progress`
**级别**: L3（跨模块架构变更：model loading + policy + interceptor + CLI）
**日期**: 2026-04-12

---

## 1. 需求

在 `serve_policy.py` 启动时，允许用户通过命令行参数分别指定 Stage 1/2/3 的设备放置：

```bash
uv run scripts/serve_policy.py \
    --stage1_device cuda:0 \
    --stage2_device meta \
    --stage3_device meta \
    ...
```

三种设置：
- `cuda` / `cuda:N`：放在 GPU 上
- `cpu`：放在 CPU 内存
- `meta`：释放常驻内存（启动时仍需完整 checkpoint 峰值内存），对应 stage 不可调用

**默认值**：三个参数均为 `None`（不覆盖），走现有 `pytorch_device=None` 自动选择逻辑（`cuda` if available else `cpu`），行为完全不变。

**作用域限定**：split device 或 meta 仅在 cache/interceptor 路径下可用。具体要求：
- split real devices（不含 meta）：需 `--cache` 或 `--cache_config`
- 含 meta stage：必须 `--cache_config`（`--cache` 简单模式无 orchestrator，stage2/3 必定执行，meta 会失败）
- 无 cache 时使用 split/meta 会在启动时报错。

**约束**：`stage1` 不允许 `meta`（所有推理路径都必须先执行 Stage 1 提取 cache key）。

---

## 2. 背景分析

### 2.1 Stage 与模块的映射

nn.Module 实际路径（相对于 `PI0Pytorch`）：

| Stage | nn.Module 路径 | 参数量 | VRAM (bf16) |
|-------|---------------|--------|-------------|
| Stage 1 | `paligemma_with_expert.paligemma.model.vision_tower`、`.model.multi_modal_projector`、`.model.language_model.embed_tokens` | ~930M | ~1.9 GB |
| Stage 2 | `paligemma_with_expert.paligemma.model.language_model.layers`、`.model.language_model.norm` | ~1.5B | ~3.0 GB |
| Stage 3 | `paligemma_with_expert.gemma_expert`、`action_in_proj`、`action_out_proj`、`time_mlp_in/out` (Pi0.5) 或 `state_proj`、`action_time_mlp_in/out` (Pi0) | ~600M | ~1.2 GB |

> **Tied weight**：`paligemma.lm_head.weight` 与 `paligemma.model.language_model.embed_tokens.weight` 是同一 tensor 对象（HuggingFace `_tied_weights_keys` + `post_init()` weight tying）。`lm_head` 不归属任何 stage——它在 staged inference 中不被调用（`_stage2_llm_backbone()` 只产生 KV cache，不做 logit 投影）。relocate 不单独移动 `lm_head`，但子模块级 `.to()` 会打断 tied 关系（PyTorch `.to()` 在子模块粒度创建新 tensor）。因此 **relocate 完成后必须调用 `paligemma.tie_weights()` 恢复绑定**，使 `lm_head.weight` 重新指向 `embed_tokens.weight`，最终跟随 Stage 1 device，不引入额外参数副本。

> **注意**：`PaliGemmaForConditionalGeneration` 上的 `@property vision_tower` 等是别名，指向 `.model.vision_tower`。用 `.to()` 方法操作实际模块时必须走完整 attribute path。

### 2.2 当前加载流程

```
policy_config.py: create_trained_policy()
  → model.py: load_pytorch()    # safetensors.torch.load_model(model, weight_path) — 全量加载
  → policy.py: Policy.__init__  # model.to(pytorch_device) — 全部移到同一设备
```

所有参数加载到同一个设备，无法分别控制。

### 2.3 设计约束

- **不修改 `PI0Pytorch` 模型类**：模型类不应知道 device placement 策略。
- **不修改 safetensors 加载逻辑**：仍然使用 `safetensors.torch.load_model()` 全量加载到 CPU。
- **不修改 upstream 的 transformers 代码**。
- **对 `meta` device 的处理**：meta 参数不占内存但也不能执行 forward。调用对应 stage 时应立即报错。

---

## 3. 方案设计

### 3.1 核心思路：加载后重放置（Post-Load Relocation）

三层加载路径：

1. **Legacy Default**（用户不传任何 stage device）：走现有 `create_trained_policy(pytorch_device=None)` 自动选择，行为零改动。
2. **All Same Device**（三个 stage 同一真实设备，如 `cpu/cpu/cpu` 或 `cuda:1/cuda:1/cuda:1`）：直接 `create_trained_policy(pytorch_device=device)`，无需 post-load relocate。
3. **Split/Meta**（不同设备或含 meta）：`create_trained_policy(pytorch_device="cpu")` 全量加载到 CPU → `relocate_model_stages()` 按模块映射表将参数 `.to(target_device)` → meta stage 的参数变为 meta tensor 释放 CPU 内存 → 更新 `Policy._pytorch_device = stage1_device`。

> **启动峰值内存**：Post-Load Relocation 方案的启动峰值 CPU 内存 ≈ 完整模型（~6GB），relocate 完成后 meta stage 参数释放，常驻内存降至所需 stage 大小。如需降低启动峰值，需要 safetensors 选择性加载（不在本次范围内）。

**`Policy` 不理解 stage placement**：`Policy` 只知道 `_pytorch_device`（= stage1 device，用于 input tensor 搬运）。`StageDeviceConfig` 显式传给 `InferenceInterceptor`，不挂在 Policy 或 model 上。

### 3.2 模块映射表

```python
# Stage 1: required (always present)
_STAGE1_MODULES = [
    "paligemma_with_expert.paligemma.model.vision_tower",
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens",
]

# Stage 2: required
# NOTE: lm_head is NOT included — it shares weight with embed_tokens (tied).
_STAGE2_MODULES = [
    "paligemma_with_expert.paligemma.model.language_model.layers",
    "paligemma_with_expert.paligemma.model.language_model.norm",
]

# Stage 3: required (always present)
_STAGE3_MODULES_REQUIRED = [
    "paligemma_with_expert.gemma_expert",
    "action_in_proj",
    "action_out_proj",
]

# Stage 3: optional, conditional on model.pi05
_STAGE3_MODULES_PI05 = ["time_mlp_in", "time_mlp_out"]           # pi05=True only
_STAGE3_MODULES_PI0  = ["state_proj", "action_time_mlp_in", "action_time_mlp_out"]  # pi05=False only
```

**错误处理**：required 模块路径解析失败 → raise（防止路径拼错导致权重留在错误设备）。optional 模块按 `model.pi05` 判断集合，不在集合内的跳过 + warning。

**Tied weight re-tie**：子模块级 `.to()` 会打断 `lm_head.weight is embed_tokens.weight` 的对象绑定。relocate 完成后必须调用 `model.paligemma_with_expert.paligemma.tie_weights()` 恢复，使 `lm_head.weight` 重新指向 `embed_tokens.weight`（跟随 Stage 1 device）。`tie_weights()` 是 HuggingFace `PreTrainedModel` 的标准方法，`scripts/train_pytorch.py` 中已有使用先例。

### 3.3 文件变更清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `scripts/serve_policy.py` | 修改 `Args` dataclass（3 个 `None` 默认参数）；`create_policy()` 三层路径；`create_default_policy()` 新增 `pytorch_device` 参数；启动校验（split real → `--cache`/`--cache_config`；meta → 必须 `--cache_config`） | CLI 入口 |
| `src/openpi/models_pytorch/stage_device_placement.py` | **新增** | `StageDeviceConfig` dataclass（三层语义 + validate）+ `relocate_model_stages()` 函数（required/optional 模块区分）|
| `src/openpi/models_pytorch/pi0_pytorch.py` | 小改 | `Stage1Output` / `Stage2Output` 添加 `.to(device)` 方法 + `_move_kv_cache()` helper |
| `src/openpi/cache/interceptor.py` | 中改 | 新增 `stage_config` 参数；meta sentinel + compile 跳过；stage 间 `.to()` 搬运（含 noise/start_x）；timer backend 按 device 选择 |
| `tests/` | 新增测试 | StageDeviceConfig、relocate、Stage*Output.to()、interceptor 防护、timer backend |
| `docs/openpi_reference.md` | 小改 | deployment / hardware 部分增加 stage device 说明 |
| `docs/cache_system_workflow.md` | 小改 | interceptor 设备语义变更 |

### 3.4 `stage_device_placement.py` 设计

```python
"""Per-stage device placement for PI0/PI0.5 models.

Relocates model sub-modules to different devices after checkpoint loading,
enabling memory-efficient configurations like stage1-only GPU inference.
"""

@dataclasses.dataclass(frozen=True)
class StageDeviceConfig:
    """Per-stage device assignment. None = no override (legacy default).

    Construct via classmethod create() which normalizes and validates.
    """
    stage1: str | None = None
    stage2: str | None = None
    stage3: str | None = None

    @classmethod
    def create(cls, stage1: str | None, stage2: str | None, stage3: str | None) -> "StageDeviceConfig":
        """Normalize device strings and validate before returning a frozen instance.

        Rules:
          - All three must be None (legacy default) or all three must be non-None.
          - stage1 cannot be "meta".
          - "cuda" is normalized to "cuda:0".
          - Device index validity is checked.
        """
        all_none = stage1 is None and stage2 is None and stage3 is None
        any_none = stage1 is None or stage2 is None or stage3 is None
        if not all_none and any_none:
            raise ValueError(
                "Either set all three stage devices or none. "
                f"Got stage1={stage1}, stage2={stage2}, stage3={stage3}"
            )
        if all_none:
            return cls()
        # Normalize and validate
        stage1 = _normalize_device(stage1)
        stage2 = _normalize_device(stage2)
        stage3 = _normalize_device(stage3)
        if stage1 == "meta":
            raise ValueError("stage1 cannot be meta: Stage 1 must always execute.")
        return cls(stage1=stage1, stage2=stage2, stage3=stage3)

    @property
    def is_legacy_default(self) -> bool:
        """No user override — use existing pytorch_device=None auto-select."""
        return self.stage1 is None and self.stage2 is None and self.stage3 is None

    @property
    def is_all_same_device(self) -> bool:
        """All stages on the same real device — direct load, no post-load relocate."""
        return (self.stage1 is not None
                and self.stage1 == self.stage2 == self.stage3
                and self.stage1 != "meta")

    @property
    def needs_relocation(self) -> bool:
        """Different devices or meta — load to CPU then relocate."""
        return not self.is_legacy_default and not self.is_all_same_device

    @property
    def has_meta_stage(self) -> bool:
        """True if any stage is on meta device."""
        return "meta" in (self.stage2, self.stage3)

    @property
    def primary_device(self) -> str:
        """Device for input tensors (always stage1's device)."""
        return self.stage1


def relocate_model_stages(model: nn.Module, config: StageDeviceConfig) -> None:
    """Move model sub-modules to their target devices in-place.

    Must be called AFTER weights are loaded (Policy.__init__ already called
    model.to("cpu") and model.eval()). eval() only sets training mode flags
    and does not affect device placement.

    For meta-device stages, parameters become zero-memory meta tensors.
    Required modules that fail to resolve raise an error.
    Optional modules (Pi0/Pi0.5 conditional) are skipped with a warning.

    After all modules are relocated, calls paligemma.tie_weights() to restore
    lm_head.weight → embed_tokens.weight binding (broken by submodule .to()).
    """
    # 1. Move stage modules to target devices
    ...
    # 2. Re-tie lm_head ↔ embed_tokens (submodule .to() breaks tied weight)
    model.paligemma_with_expert.paligemma.tie_weights()
    ...
```

### 3.5 `serve_policy.py` 变更

```python
@dataclasses.dataclass
class Args:
    ...
    # Per-stage device placement. Default: None (no override, legacy behavior).
    stage1_device: str | None = None
    stage2_device: str | None = None
    stage3_device: str | None = None


def create_policy(args: Args) -> _policy.Policy:
    stage_config = StageDeviceConfig.create(args.stage1_device, args.stage2_device, args.stage3_device)

    # Startup guard: split/meta requires cache/interceptor path
    if stage_config.needs_relocation and not (args.cache or args.cache_config):
        raise ValueError(
            "Split device placement requires --cache or --cache_config. "
            "Without cache, Policy.infer() uses single-device staged path."
        )
    # Meta stages require --cache_config (not just --cache).
    # --cache creates InferenceInterceptor without orchestrator, so CP1 never
    # hits and stage2/3 always execute — meta stages would fail.
    if stage_config.has_meta_stage and not args.cache_config:
        raise ValueError(
            "Meta stage placement requires --cache_config (not just --cache). "
            "--cache creates an interceptor without orchestrator, so stage2/3 "
            "always execute. Use --cache_config with always_hit judge."
        )

    # Determine pytorch_device for create_trained_policy()
    if stage_config.is_all_same_device:
        pytorch_device = stage_config.stage1
    elif stage_config.needs_relocation:
        pytorch_device = "cpu"
    else:
        pytorch_device = None  # legacy default

    # Both Checkpoint and Default branches use the same pytorch_device
    match args.policy:
        case Checkpoint():
            policy = _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir,
                default_prompt=args.default_prompt, pytorch_device=pytorch_device,
            )
        case Default():
            policy = create_default_policy(
                args.env, default_prompt=args.default_prompt,
                pytorch_device=pytorch_device,
            )

    if stage_config.needs_relocation:
        relocate_model_stages(policy._model, stage_config)
        policy._pytorch_device = stage_config.primary_device

    return policy
```

`create_default_policy()` 需要新增 `pytorch_device` 参数并传递给 `create_trained_policy()`。`Checkpoint` 和 `Default` 两条分支统一处理。

### 3.6 跨设备 tensor 传输（多 GPU 支持）

当不同 stage 在不同设备上时，stage 之间的 tensor 需要显式搬运。

**实现方式**：给 `Stage1Output` / `Stage2Output` 添加 `.to(device)` 方法：

```python
@dataclass
class Stage1Output:
    ...
    def to(self, device: str | torch.device) -> "Stage1Output":
        """Move all tensors to the target device. No-op if already there."""
        return Stage1Output(
            state=self.state.to(device),
            prefix_embs=self.prefix_embs.to(device),
            prefix_pad_masks=self.prefix_pad_masks.to(device),
            prefix_att_2d_masks_4d=self.prefix_att_2d_masks_4d.to(device),
            prefix_position_ids=self.prefix_position_ids.to(device),
        )
```

`Stage2Output` 同理（包含 `stage1` + `past_key_values`，其中 `past_key_values` 是 HuggingFace `DynamicCache`，需遍历 key/value tensors）。

**插入点**：在 `InferenceInterceptor.infer()` 中：

```python
stage1 = self._stage1_fn(observation)
stage1 = stage1.to(self._stage2_device)   # no-op if same device
stage2 = self._stage2_fn(stage1)
# stage3 needs state from stage1 + KV from stage2, both need to be on stage3_device
```

`torch.Tensor.to(same_device)` 是 **no-op**（返回 self，零开销），所以默认全 cuda:0 时无性能影响。

`Stage2Output.to()` 中 `past_key_values` 的搬运通过私有 helper 实现：

```python
def _move_kv_cache(past_key_values, device):
    """Move KV cache to target device. Handles None, DynamicCache, and tuple formats."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, 'to'):
        result = past_key_values.to(device)
        return result if result is not None else past_key_values
    # Fallback: tuple of (key, value) tuples
    return tuple((k.to(device), v.to(device)) for k, v in past_key_values)
```

### 3.7 InferenceInterceptor 变更

`InferenceInterceptor.__init__()` 新增 `stage_config: StageDeviceConfig | None = None` 参数（显式传参，不从 Policy 或 model 读取）。

**3.7.1 meta 防护 + compile 跳过**

在 `_get_or_compile_stages()` 中，meta stage 不编译，替换为 sentinel 函数：

```python
def _meta_guard(stage_name):
    def _fn(*args, **kwargs):
        raise RuntimeError(f"{stage_name} is on meta device (not loaded).")
    return _fn

# meta stages: sentinel; non-meta: compile or eager
s2 = (_meta_guard("stage2") if sc and sc.stage2 == "meta"
      else self._model.run_stage2)
# Only compile non-meta stages
if compile_mode and not (sc and sc.stage2 == "meta"):
    s2 = torch.compile(s2, mode=compile_mode)
```

meta sentinel 在 interceptor 级别持有，**不**写入 `model._compiled_stage*_fn`。非 meta stage 的 compiled function 照常缓存在 model 上。并发模式下 device placement 是模型级固定的（启动时 relocate），所有 interceptor 共享同一 device layout。

**3.7.2 跨设备 tensor 搬运**

stage 之间插入 `.to(next_device)`：

```python
stage1 = self._stage1_fn(observation)
stage1 = stage1.to(self._stage2_device)   # no-op if same device
stage2 = self._stage2_fn(stage1)
# stage2 output → stage3 device (via Stage2Output.to())
```

额外搬运点（G1 审查发现）：
- `start_noise`：`.to(self._stage3_device)` 而非 `self._pytorch_device`
- WARM_START 的 `start_x`：`.to(self._stage3_device)` 而非 `self._pytorch_device`
- CP1 FULL_HIT 的 `cached_action`：不受影响，只需 stage1 device（和 `inputs["state"]` 同设备）

**meta guard 时序**：availability check 在进入 `with self._timer.measure(...)` 之前执行，避免 sentinel 在 timing context 内抛出异常触发 fallback warning。

**3.7.3 Timer backend 按 device 选择**

```python
def _probe_backend(device_str: str | None) -> str:
    return "cuda" if device_str and device_str.startswith("cuda") else "cpu"

# stage_config 已知时：
self._timer.register_probe("stage1_vision", backend=_probe_backend(sc.stage1))
self._timer.register_probe("stage2_llm",    backend=_probe_backend(sc.stage2))
self._timer.register_probe("stage3_flow",   backend=_probe_backend(sc.stage3))
```

`stage3_warm` 也按 Stage 3 device 注册（与 `stage3_flow` 同 backend）。

`stage_config=None` 时保持现有行为（全 `"cuda"`）。meta stage 的 probe 不注册（不会被调用）。

### 3.8 与 ToyStage1Policy 的关系

实现后，`ToyStage1Policy` 的 device placement 功能可以用 `StageDeviceConfig(stage1="cuda", stage2="meta", stage3="meta")` 完全替代。但 `ToyStage1Policy` 有独立的 Qdrant HTTP 查询逻辑，不在本次重构范围内。

---

## 4. 典型使用场景

### 4.1 Stage1-only（cache always-hit 实验）

```bash
uv run scripts/serve_policy.py \
    --stage1_device cuda:0 \
    --stage2_device meta \
    --stage3_device meta \
    --cache_config cache.yaml \
    ...
```

- 启动峰值 CPU 内存：~6GB（加载完整 checkpoint 后释放 stage2/3）
- 运行时 VRAM 占用：~1.9 GB（可在 4GB 显卡上运行）
- 运行时 CPU 内存：接近零（meta 参数已释放）
- 要求 cache 配置使用 always_hit，否则 miss 时报错

### 4.2 Stage1 GPU + Stage2/3 CPU（低速回退）

```bash
--stage1_device cuda:0 --stage2_device cpu --stage3_device cpu
```

- 所有 stage 都可执行，cache miss 时回退到 CPU 推理（慢但可用）
- VRAM ~1.9 GB + CPU 内存 ~4 GB

### 4.3 全 GPU（默认，不变）

```bash
# 不传参数（legacy default，推荐）
uv run scripts/serve_policy.py ...

# 或显式指定同一设备（all_same_device 路径）
--stage1_device cuda:0 --stage2_device cuda:0 --stage3_device cuda:0
```

- 不传参数：走 legacy default，行为与现有代码完全一致
- 显式同一设备：`is_all_same_device` 路径，直接 `pytorch_device=device` 加载

### 4.4 多 GPU 分割

```bash
--stage1_device cuda:0 --stage2_device cuda:1 --stage3_device cuda:1
```

- Stage 1 在 GPU 0，Stage 2/3 在 GPU 1
- stage 之间的 tensor 通过 `Stage1Output.to()` / `Stage2Output.to()` 自动搬运
- 跨 GPU 传输有 PCIe 开销（每次 ~数 MB，延迟 <1ms），但远小于 stage 本身的计算时间

---

## 5. 风险与注意事项

### 5.1 `embed_tokens` 共享问题

`embed_tokens` 在 Stage 1（语言 token embedding）和 Stage 2（LLM backbone 的 input embedding）中都被使用。但在 `_stage2_llm_backbone()` 中，传入的是已经 embed 好的 `prefix_embs`（而非 raw tokens），所以 `embed_tokens` 只属于 Stage 1，不会在 Stage 2 执行时被调用。

**验证方式**：阅读 `_stage2_llm_backbone()` → 只接收 `prefix_embs`（已经是 float tensor），不调用 `embed_tokens()`。✅

### 5.2 `to_bfloat16_for_selected_params` 与 device placement 的交互

`policy_config.py` 在 `load_model()` 后调用 `to_bfloat16_for_selected_params("bfloat16")`，这是决定最终 dtype 的关键点。relocate 在此之后执行，`.to(device)` 不改变 dtype，所以 dtype 保持正确。测试需验证 relocate 后 dtype + device 同时符合预期。

### 5.3 `torch.compile` 与跨设备

如果不同 stage 在不同 device 上，`torch.compile` 需要分别编译。当前 `InferenceInterceptor._get_or_compile_stages()` 已经是分开编译 stage1/stage2/stage3 的。meta device 的 stage 替换为 sentinel 函数，不编译（见 §3.7.1）。

### 5.4 性能影响

- **Legacy default（不传参数）**：`is_legacy_default` 短路，零影响。
- **All same device**：直接 `pytorch_device=device`，无 relocate 开销。
- **Split device**：只影响启动时间（多几秒 relocate），运行时 `Tensor.to(same_device)` 是 no-op，同设备零开销；跨 GPU 时有 PCIe 传输延迟（<1ms），远小于 stage 计算时间。

### 5.5 Tied weight `lm_head` ↔ `embed_tokens`

子模块级 `.to()` 会打断 PyTorch 的 weight tying（创建新 tensor 而非原地搬运）。relocate 后 `lm_head.weight is embed_tokens.weight` 会变为 `False`。**必须在 relocate 后调用 `paligemma.tie_weights()` 恢复绑定**。测试 `test_tied_weight_retied` 覆盖此风险。

### 5.6 CPU stage 的 bf16 算子支持

`stage2=cpu` / `stage3=cpu` 是**实验性功能**。CPU 下 bf16 matmul 在部分硬件上有限支持（需要 AVX512_BF16 或 AMX）。如果 CPU 推理出现精度或算子错误，可能需要手动转换为 float32（类似 ToyStage1Policy 的 `.to(dtype=torch.float32)` 模式）。测试策略中以 `@pytest.mark.manual` 覆盖。

---

## 6. 实现步骤

### Phase 1: 核心模块

1. 创建 `src/openpi/models_pytorch/stage_device_placement.py`
   - `StageDeviceConfig` frozen dataclass（classmethod `create()` 负责规范化 + 校验，三层语义）
   - `relocate_model_stages()` 函数（required/optional 模块区分 + re-tie `lm_head`↔`embed_tokens`）
   - 模块映射表（完整 nn.Module 路径，含 `.model` 层）
   - 日志输出：每个 stage 的 device + 参数数量

### Phase 2: 跨设备 tensor 传输

2. 修改 `src/openpi/models_pytorch/pi0_pytorch.py`
   - `Stage1Output` 添加 `.to(device)` 方法（5 个 tensor 字段）
   - `Stage2Output` 添加 `.to(device)` 方法（stage1 + `_move_kv_cache()` helper）

### Phase 3: CLI 集成

3. 修改 `scripts/serve_policy.py`
   - `Args` 新增三个 `None` 默认参数
   - `create_policy()` 三层路径（legacy/all_same/relocate）
   - `create_default_policy()` 新增 `pytorch_device` 参数
   - 启动校验：split/meta 无 cache 报错；stage1=meta 报错
   - `_wrap_policy()` 传 `stage_config` 给 `InferenceInterceptor`

### Phase 4: Interceptor 变更

4. 修改 `src/openpi/cache/interceptor.py`
   - `__init__()` 新增 `stage_config` 参数
   - meta sentinel + compile 跳过
   - stage 间 `.to()` 搬运（含 noise、start_x）
   - timer probe backend 按 device 选择

### Phase 5: 测试

5. 添加 `tests/models_pytorch/test_stage_device_placement.py`
   - StageDeviceConfig 属性（legacy_default / all_same / needs_relocation）
   - relocate 模块映射（mock model）
   - meta device 参数 `is_meta=True`
   - required 模块缺失报错
   - optional 模块（pi05 条件）跳过
   - Stage1Output.to() / Stage2Output.to() 搬运
   - interceptor meta guard
   - interceptor tensor device 搬运（noise、start_x）
   - timer backend 选择
   - serve_policy 装配（legacy/all_same/split/meta+no-cache guard）
   - `@pytest.mark.manual`：真实模型 named_modules() 路径验证；CPU stage 推理

### Phase 6: 文档

6. 更新 `docs/openpi_reference.md` deployment/hardware 部分
7. 更新 `docs/cache_system_workflow.md` interceptor 设备语义

---

## 7. 测试策略

| 测试 | 类型 | 内容 |
|------|------|------|
| `test_config_legacy_default` | Unit | 三个 `None` → `is_legacy_default=True` |
| `test_config_all_same_device` | Unit | 同一真实设备 → `is_all_same_device=True` |
| `test_config_needs_relocation` | Unit | 不同设备或含 meta → `needs_relocation=True` |
| `test_config_validate_stage1_meta` | Unit | `stage1="meta"` → ValueError |
| `test_config_validate_partial_override` | Unit | `stage1="cuda:0", stage2=None` → ValueError |
| `test_relocate_moves_modules` | Unit | mock model 验证模块移到目标设备 |
| `test_relocate_meta_frees_memory` | Unit | meta device 后参数 `is_meta=True` |
| `test_relocate_required_missing_raises` | Unit | required 模块路径解析失败 → 报错 |
| `test_relocate_optional_pi05_skip` | Unit | Pi0 模型跳过 pi05 optional 模块 |
| `test_stage1_output_to_device` | Unit | `Stage1Output.to()` 正确搬运 5 个 tensor |
| `test_stage2_output_to_device` | Unit | `Stage2Output.to()` 搬运 stage1 + KV cache |
| `test_stage_output_to_same_noop` | Unit | `.to(same_device)` 返回相同 tensor |
| `test_interceptor_meta_guard` | Unit | stage2=meta 时 CP1 miss → RuntimeError |
| `test_interceptor_noise_device` | Unit | noise / start_x 搬运到 stage3 device |
| `test_timer_backend_by_device` | Unit | CPU stage → cpu probe，CUDA stage → cuda probe |
| `test_serve_split_no_cache_guard` | Unit | split device 无 --cache → ValueError |
| `test_serve_meta_simple_cache_guard` | Unit | meta stage + `--cache`（无 config） → ValueError |
| `test_tied_weight_retied` | Unit | relocate 后 `lm_head.weight is embed_tokens.weight` 仍为 True，且 `lm_head.weight.device == stage1_device` |
| `test_stage3_warm_timer_backend` | Unit | `stage3_warm` probe 按 Stage 3 device 注册 |
| `test_cli_args_parse` | Unit | tyro 解析 `--stage1_device cuda:0 --stage2_device meta --stage3_device meta` |
| `test_named_modules_paths` | `@pytest.mark.manual` | 真实模型 `named_modules()` 验证映射表路径 |
| `test_cpu_stage_inference` | `@pytest.mark.manual` | CPU stage2/3 推理端到端验证 |
| `test_full_stage1_only` | `@pytest.mark.manual` | 实际 checkpoint，stage2/3=meta，运行 stage1 |

---

## 8. Workflow Stage

```
WORKFLOW STATUS | Task: Stage Device Placement | Level: L3
Understand ✅ → Plan ✅ → G1 ✅ → Code ✅ → G2 ✅ → Verify ✅
```

**G1 已通过（§15，四轮审查）。可以进入 Code 阶段。**

---

## 9. G1 审查摘要

四轮审查（§9-§15 已归档），主要解决：

1. 默认值 `cuda:0` → `None`，保留 legacy auto-select
2. 三层语义：`is_legacy_default` / `is_all_same_device` / `needs_relocation`
3. 作用域限定：split real → `--cache`/`--cache_config`；meta → 必须 `--cache_config`
4. 模块路径修正：加 `.model` 层 + required/optional 区分
5. `lm_head` tied weight：从 Stage 2 移除，relocate 后 `tie_weights()` 恢复绑定
6. `StageDeviceConfig.create()` classmethod 规范化 + frozen 不可变
7. meta sentinel 在 interceptor 级别持有，不缓存到 model
8. meta guard 在 `timer.measure()` 外触发
9. `_move_kv_cache()` 处理 None / in-place / DynamicCache / tuple
10. noise / start_x 搬运到 `stage3_device`
11. timer backend 按 stage device 选择（含 `stage3_warm`）

**不在本次范围**：非 cache 路径 split device、选择性 safetensors 降低启动峰值、CPU bf16 自动 float32 fallback。

---

## 10. G2 审查记录

**结论：暂不通过 G2。** 当前实现方向与 G1 方案基本一致，新增单测也能通过，但仍有两个运行时风险会影响默认 cache 路径和 `stage3=meta` 的失败语义，需要修正后再复审。

### 问题 1：legacy default 的 stage timer backend 被误改成 CPU

位置：`src/openpi/cache/interceptor.py:185`

`scripts/serve_policy.py` 在 `main()` 中总是创建并传入 `StageDeviceConfig.create(None, None, None)`，因此 `InferenceInterceptor.__init__()` 里 `sc` 不为 `None`，但 `sc.stage1/stage2/stage3` 全是 `None`。当前 probe 注册逻辑使用：

```python
_probe_backend(sc and sc.stage1)
```

legacy default 下这里会变成 `_probe_backend(None)`，也就是 CPU backend。结果是未指定任何 `--stage*_device`、模型实际跑在 CUDA 时，`stage1_vision` / `stage2_llm` / `stage3_flow` 都会被注册成 CPU wall timer，改变现有 cache timing 行为。

建议：

1. 在 interceptor 内先归一化实际 stage device：
   - legacy default：stage1/stage2/stage3 都使用 `self._pytorch_device`
   - explicit config：使用 `sc.stage1/sc.stage2/sc.stage3`
2. probe 注册统一使用归一化后的实际 device。
3. 增加一个不跑真实模型的单测：`stage_config=StageDeviceConfig()`、`policy._pytorch_device="cuda:0"` 时，`stage1_vision` 注册为 CUDA backend。

原因：G1 批准范围要求 legacy default 保持旧行为，timer backend 只是按实际 stage device 选择，不能因为传入了 legacy config 而退化成 CPU。

### 问题 2：`stage3=meta` 的 cache MISS / WARM_START 没有走 meta sentinel

位置：`src/openpi/cache/interceptor.py:390`

`__init__()` 中虽然在 `stage3=meta` 时把 `self._stage3_fn` 替换成 `_meta_guard("stage3")`，但 `infer()` 的 cache 分支没有使用它：

```python
# WARM_START
stage3 = self._model.run_stage3_from(...)

# MISS
stage3 = self._model.run_stage3(...)
```

因此配置如 `stage1=cuda:0, stage2=cpu, stage3=meta` 时，如果 CP1 不是 FULL_HIT，会直接调用 meta 参数上的真实 Stage 3 计算，报错会变成 PyTorch/meta tensor 的底层错误，并且还会因为 `stage3_flow` / `stage3_warm` 未注册而触发 timer fallback warning。这个行为不符合 plan 中“meta stage 被调用时给清晰 RuntimeError”的要求。

建议：

1. 在进入 Stage 3 分支前显式判断 `self._stage_config.stage3 == "meta"`，只要不是 CP1 FULL_HIT 就直接抛出清晰 `RuntimeError`。
2. WARM_START 和 MISS 两条路径都要覆盖，不只覆盖 no-cache 的 `_stage3_fn` 路径。
3. 增加单测覆盖：
   - `stage2=cpu, stage3=meta` + CP1 MISS → 清晰 RuntimeError
   - `stage2=cpu, stage3=meta` + CP1 WARM_START → 清晰 RuntimeError

原因：`stage3=meta` 不一定总是和 `stage2=meta` 一起使用。当前 `stage2=meta, stage3=meta` 的 MISS 会先在 Stage 2 sentinel 处失败，但 `stage2=cpu, stage3=meta` 会绕过 sentinel，属于实际可配置组合的漏洞。

### 建议 3：`StageDeviceConfig.create()` 的 CUDA 字符串校验还不完整

位置：`src/openpi/models_pytorch/stage_device_placement.py:63`

当前 `_normalize_device("cuda:0:1")` 会被接受成 `cuda:0`，因为只读取了 `device.split(":")[1]`。此外函数文档写了“validate index”，但没有校验 `torch.cuda.device_count()`。如果暂时不想在无 GPU 环境拒绝 `cuda:N`，至少应把文档改成“格式校验”，并用更严格的解析拒绝多余冒号。

原因：CLI 参数错误最好在启动校验阶段暴露，避免后续 `.to(device)` 才抛出更难定位的错误。

### 建议 4：serve guard 单测没有真正测 guard

位置：`tests/models_pytorch/test_stage_device_placement.py:450`

`test_serve_split_no_cache_guard` 和 `test_serve_meta_simple_cache_guard` 目前只验证 `_get_stage_device_config()` 的属性，没有调用 `create_policy()`，因此不会覆盖 `scripts/serve_policy.py` 中真正的启动 guard。可以通过 monkeypatch `_policy_config.create_trained_policy` / `create_default_policy` 避免真实加载模型，然后断言 `create_policy(args)` 在 guard 阶段直接抛 `ValueError`。

原因：这两条 guard 是本功能的安全边界，尤其是 meta + `--cache` 的错误配置，应该有直接回归测试。

### 已验证

命令：

```bash
uv run python -m pytest tests/models_pytorch/test_stage_device_placement.py
```

结果：`39 passed, 3 skipped`。直接运行 `uv run pytest ...` 失败，原因是当前 `.venv/bin/pytest` 的 shebang 指向旧路径；改用 `python -m pytest` 后通过。

---

## 11. G2 复审记录

**结论：G2 通过。** §10 提出的 4 个问题均已修复，当前实现可以进入 Verify 阶段。

### 已确认修复

1. legacy default timer backend 已恢复按实际 `policy._pytorch_device` 注册
   - 位置：`src/openpi/cache/interceptor.py:161`
   - 现在先归一化 `self._stage1_device` / `self._stage2_device` / `self._stage3_device`，legacy default 使用 wrapped policy 的 `_pytorch_device`。
   - probe 注册改为使用归一化后的 stage device，因此默认 CUDA cache 路径不会退化成 CPU timer。
   - 新增测试覆盖：`tests/models_pytorch/test_stage_device_placement.py:422`

2. `stage3=meta` 的 MISS / WARM_START 路径已有显式 guard
   - 位置：`src/openpi/cache/interceptor.py:405`
   - 现在只要 CP1 不是 FULL_HIT 且 Stage 3 是 meta，就会在 Stage 3 分支前抛出清晰 `RuntimeError`。
   - 新增测试覆盖：`tests/models_pytorch/test_stage_device_placement.py:458`

3. CUDA device 字符串格式校验已收紧
   - 位置：`src/openpi/models_pytorch/stage_device_placement.py:76`
   - `cuda:0:1` 这类多冒号格式现在会被拒绝。
   - 文档说明已改成“格式校验，不校验 runtime GPU availability”，与实现一致。
   - 新增测试覆盖：`tests/models_pytorch/test_stage_device_placement.py:131`

4. serve startup guard 测试已改为直接调用 `create_policy()`
   - 位置：`tests/models_pytorch/test_stage_device_placement.py:553`
   - split 无 cache、meta 只有 `--cache`、meta 无任何 cache flag 三条错误配置现在都通过 `create_policy()` 断言 `ValueError`。
   - guard 发生在真实模型加载前，因此测试无需 monkeypatch 加载路径也能覆盖安全边界。

5. `_probe_backend()` 已兼容 `torch.device`
   - 位置：`src/openpi/cache/interceptor.py:91`
   - 这解决了测试和实际调用中 device 可能是字符串或 `torch.device` 的不一致。

### 非阻塞建议

1. `stage3=meta` guard 可以再提前一点
   - 位置：`src/openpi/cache/interceptor.py:402`
   - 当前顺序是先 `stage2 = stage2.to(self._stage3_device)`，再检查 `self._stage3_device == "meta"`。这不会再调用真实 Stage 3，但会在抛错前尝试把 `Stage2Output` / KV cache 搬到 meta。
   - 建议把 Stage 3 meta guard 放到 `stage2.to(self._stage3_device)` 之前，避免不必要的 KV cache 搬运，也减少未来 `DynamicCache` 表示变化时在清晰错误前抛底层异常的风险。
   - 这不是当前 G2 blocker，因为原问题“绕过 sentinel 调用真实 Stage 3”已经解决。

2. `docs/cache_system_workflow.md` 的章节编号出现重复
   - 位置：`docs/cache_system_workflow.md`
   - 新增 “Per-Stage Device Placement” 后，后面的 “Fully Pluggable Components” 仍然是 `6.`。建议顺手改成后续编号，避免文档目录歧义。

### 已验证

命令：

```bash
uv run python -m pytest tests/models_pytorch/test_stage_device_placement.py
```

结果：`43 passed, 3 skipped`。
