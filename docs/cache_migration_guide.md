# Cache 框架迁移指南

> **版本**: 1.0
> **状态**: 初始版本
> **最后更新**: 2026-04-10
>
> **设计文档**: [cache_system_architecture.md](cache_system_architecture.md)
> **组件教程**: [cache_system_tutorial.md](cache_system_tutorial.md)
> **工作流图**: [cache_system_workflow.md](cache_system_workflow.md)

---

## 目录

1. [概述与适用范围](#1-概述与适用范围)
2. [架构总览（迁移视角）](#2-架构总览迁移视角)
3. [最小可迁移主线（CP1）](#3-最小可迁移主线cp1)
   - [Step 1: 分析你的推理流程](#step-1-分析你的推理流程)
   - [Step 2: 定义最小接口契约](#step-2-定义最小接口契约)
   - [Step 3: 实现自定义 KeyBuilder](#step-3-实现自定义-keybuilder)
   - [Step 4: 实现自定义 Interceptor](#step-4-实现自定义-interceptor)
   - [Step 5: 注册到配置系统](#step-5-注册到配置系统)
   - [Step 6: 数据收集与 Artifact 构建](#step-6-数据收集与-artifact-构建)
   - [Step 7: 验证](#step-7-验证)
4. [可选扩展](#4-可选扩展)
5. [实验性 / 未完成项](#5-实验性--未完成项)
6. [YAML 配置参考](#6-yaml-配置参考)
7. [常见问题与陷阱](#7-常见问题与陷阱)

---

## 1. 概述与适用范围

### 本框架做什么

OpenPI Cache 是一个**多级推理缓存系统**，通过在推理管线的关键位置（checkpoint）设置缓存检查点，在场景相似时跳过后续计算，直接复用历史推理结果，从而降低端到端推理延迟。

核心设计原则：
- **Interceptor 模式**: 缓存逻辑作为外部插件挂载到推理管线上，不修改模型推理内部代码
- **可插拔组件**: KeyBuilder、Gate、Judge、SearchStrategy、WritePolicy 均可替换
- **Backend 无关**: 存储层通过 `VectorStoreBackend` ABC 抽象，上层逻辑不依赖具体向量数据库

### 适用范围

**本教程是什么**: 基于当前 Pi0.5 实现提炼的适配指南。帮助你将这套缓存框架接入你自己的模型。

**本教程不是什么**: 本仓库并未对非 Pi0.5 模型提供开箱即用支持。迁移需要你编写自定义的 KeyBuilder 和 Interceptor，并可能需要适配数据收集和 artifact 构建流程。

### 你不需要了解

- OpenPI 项目的训练流程
- Pi0.5 模型的内部架构细节
- JAX 相关代码（本框架仅支持 PyTorch 路径）

### 迁移路径判定

在开始之前，回答以下问题，确定你的迁移路径：

| 问题 | 是 | 否 |
|------|----|----|
| 你的模型推理能拆分为多个 stage？ | 直接使用 Staged API 模式 → Step 1 | 见 [§7 单 forward pass 模型](#单-forward-pass-模型) |
| 你能从模型中间层获取 embedding？ | 实现自定义 KeyBuilder → Step 3 | 考虑 CLIP builder → [§4.3](#43-clip-builder-路线详解) |
| 你需要在线推理缓存？ | 完整走 Step 1-7 | 仅做离线 artifact 实验 → [§4.4](#44-离线-artifact-预构建) |

---

## 2. 架构总览（迁移视角）

### 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                  你需要实现的（模型相关）                   │
│                                                         │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  Interceptor    │  │  KeyBuilder                  │  │
│  │  (推理流程控制)  │  │  (中间表示 → query vectors)   │  │
│  └────────┬────────┘  └──────────────┬───────────────┘  │
│           │                          │                   │
├───────────┼──────────────────────────┼───────────────────┤
│           │    直接复用的（模型无关）    │                   │
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
│  │         (InMemoryBackend / 自定义 backend)           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Checkpoint 语义

当前框架支持两个 checkpoint ID：

| Checkpoint | Pi0.5 中的位置 | 语义 | 你需要做的 |
|------------|---------------|------|-----------|
| **CP1** | Stage 1（视觉编码）之后 | 观测编码完成后检查缓存；命中则跳过后续所有计算 | 在你的模型中找到"观测处理完毕、开始决策/生成"的分界点 |
| **CP3** | Stage 3（动作生成）之后 | 当前推理完成后，预测下一步是否可跳过（实验性） | 保留框架代码，但当前不提供实际加速 |

> **注意**: 当前框架只支持 `CP1` / `CP3` 两个 checkpoint ID（`CheckpointID` enum）。配置系统也只接受这两个名称。如需新增 checkpoint，须修改 `types.py`、`config.py`、`orchestrator.py` 及相关测试——这属于框架扩展，不在本教程范围内。

---

## 3. 最小可迁移主线（CP1）

### Step 1: 分析你的推理流程

**目标**: 将你的模型推理拆分为 stages，找到 CP1 对应的位置。

CP1 的核心语义是：**"观测编码完成后，如果当前观测与历史某次观测足够相似，直接复用历史动作，跳过后续所有计算。"**

以 Pi0.5 为例，推理被拆分为三段：

```
Stage 1: 视觉编码 + token 准备  →  [CP1]  →  Stage 2: LLM 前向  →  Stage 3: 流匹配去噪
```

对于不同模型架构，CP1 位置的选择参考：

| 模型类型 | 推荐的 CP1 位置 | 理由 |
|----------|----------------|------|
| Vision-Language-Action (VLA) | 视觉编码器之后 | 视觉 embedding 是最重的计算，也是最好的场景相似度指标 |
| Diffusion Policy | 条件编码器（image encoder + FiLM）之后 | 跳过整个去噪循环 |
| ACT (Action Chunking Transformer) | CVAE encoder 之后 | 跳过 decoder |
| 单 encoder-decoder 模型 | encoder 之后 | 跳过 decoder |

**关键问题**: 你的模型中，哪一步是"观测理解"与"动作生成"的分界线？那就是 CP1。

### Step 2: 定义最小接口契约

你不需要复制 Pi0.5 的 `Stage1Output` / `Stage3Output` 数据结构。你需要满足的最低契约是：

#### 2.1 CP1 检查点的输入契约

CP1 check 时，`orchestrator.check(CheckpointID.CP1, **kwargs)` 的 `kwargs` 会传递给 `KeyBuilder.collect()`。你的 stage output 必须包含 KeyBuilder 构建 query key 所需的信息。

**最低要求**:
- 至少一种观测 embedding（视觉、状态、语言，或它们的组合）
- 格式：GPU 上的 `torch.Tensor`

#### 2.2 动作输出契约

缓存命中时返回的动作，以及缓存写入时存储的动作，都通过 `CachePayload.action_chunk` 承载。

**最低要求**:
- 形状：`[action_horizon, action_dim]`，CPU float32 contiguous
- 这是你的模型最终输出的动作序列（经过 output transform 之前的原始动作）

#### 2.3 Episode lifecycle 契约

推理流程必须在正确的时机发送 episode 生命周期信号：

| 信号 | 时机 | 调用 |
|------|------|------|
| episode start | 每个 episode 开始时 | `orchestrator.on_episode_start(task_key, episode_id)` |
| broadcast action | 每步推理后 | `orchestrator.broadcast_action(action)` |
| buffer for write | 每步推理后 | `orchestrator.buffer_for_write(query_keys, action)` |
| episode end | 每个 episode 结束时 | `orchestrator.on_episode_end()` |
| clear | 每步推理末尾 | `orchestrator.clear()` |

#### 2.4 Pi0.5 参考示例

```python
# Pi0.5 的 Stage Output 定义（仅供参考，你不需要复制这些字段）

@dataclass
class Stage1Output:
    state: torch.Tensor           # [B, action_dim]
    prefix_embs: torch.Tensor     # [B, prefix_len, emb_dim]  — SigLIP vision tokens + prompt tokens
    prefix_pad_masks: ...
    prefix_att_2d_masks_4d: ...
    prefix_position_ids: ...

@dataclass
class Stage3Output:
    action_chunk: torch.Tensor    # [B, action_horizon, action_dim]
    intermediates: Optional[...]  # flow matching 中间状态（可选）
```

### Step 3: 实现自定义 KeyBuilder

KeyBuilder 将模型的中间表示转换为固定维度的 query 向量，供缓存检索使用。

#### 3.1 Protocol 接口

```python
class QueryKeyBuilder(Protocol):
    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        """从 stage output 提取原始 tensor 引用（GPU 上，不拷贝）。"""
        ...

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """降维 + GPU→CPU 转换。返回 {field_name: [dim] CPU float32}。
        cosine 字段建议 L2 归一化；L2 距离字段（robot_state）保留原始向量。"""
        ...

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        """暴露 collect() 缓存的原始 tensor（GPU 上），供 Gate/Judge 使用。"""
        ...

    def clear(self) -> None:
        """释放缓存引用。每步推理末尾调用。"""
        ...
```

#### 3.2 字段映射规则

query key 的字段名**必须**是以下 5 个 canonical fields 的子集：

```python
# src/openpi/cache/types.py
VISION_0 = "vision_0"      # 主摄像头 / 第一视觉输入
VISION_1 = "vision_1"      # 第二视觉输入（如腕部摄像头）
VISION_2 = "vision_2"      # 第三视觉输入
PROMPT_EMB = "prompt_emb"  # 语言/任务 embedding
ROBOT_STATE = "robot_state" # 状态向量
```

**映射策略**:
- 如果你的模型有视觉 embedding → 映射到 `vision_0`（多摄像头则用 `vision_1`, `vision_2`）
- 如果有语言/任务 embedding → 映射到 `prompt_emb`
- 如果有状态向量 → 映射到 `robot_state`
- 字段可以省略（在 YAML 中设 `enabled: false`），但不能新增

> **深度改造**: 如果现有 5 个字段无法覆盖你的需求（例如你需要 `tactile_emb`），须修改 `src/openpi/cache/types.py` 中的 `CACHE_QUERY_FIELDS`、`src/openpi/cache/config.py` 中的校验逻辑、backend 的 `vector_dims` 声明及相关测试。这属于框架扩展，不在本教程的主线范围内。

#### 3.3 实现示例

以下是一个假设的 Diffusion Policy KeyBuilder 示例：

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

**要点**:
- `collect()` 只保存 GPU tensor 引用，不做拷贝
- `build()` 是唯一的 GPU→CPU 数据传输点
- 返回的 tensor 必须是 **1D、CPU、float32、contiguous**
- **归一化规则因字段而异**: cosine 相似度字段（`vision_0/1/2`, `prompt_emb`）建议 L2 归一化；L2 距离字段（`robot_state`）**必须保留原始向量**，归一化会破坏距离语义
- 输出维度必须与 YAML 配置中的 `backend.vector_dims` 一致

#### 3.4 CLIP 低耦合分支

如果你不想（或无法）从模型内部提取视觉 embedding，可以使用 CLIP KeyBuilder。它用独立的 CLIP 编码器处理原始输入图像，不依赖目标模型的内部表示。

**前提**: 你的推理流程能提供原始输入图像（numpy 数组）和 `robot_state` 向量。

**限制**:
- CLIP embedding 不反映模型训练学到的特征，检索质量可能不如模型内部 embedding
- 仍需 `robot_state` 字段
- CLIP 编码引入额外推理开销（约 5-10ms per image）

用法参见 [§4.3](#43-clip-builder-路线详解)。

### Step 4: 实现自定义 Interceptor

Interceptor 是缓存系统与推理管线的集成层。它包装你的 policy/model，在推理流程中插入缓存检查和写入逻辑。

#### 4.1 宿主运行时最小契约

你的 Interceptor 必须满足以下条件：

| 契约 | 说明 |
|------|------|
| **BasePolicy 语义** | 实现 `infer(obs) -> dict` 接口，作为原 policy 的透明替换 |
| **访问模型的 staged 推理方法** | 能调用拆分后的 stage 函数（如 `run_stage1()`, `run_stage2()` 等） |
| **访问 transform 管线** | 能执行 input transform（raw obs → model input）和 output transform（model output → action） |
| **获取 device 信息** | 知道模型在哪个 GPU 上，用于 tensor 转移 |
| **Episode lifecycle** | 能在正确时机调用 `on_episode_start` / `on_episode_end` |
| **Wrapper ordering** | 如果与 `serve_policy.py` 集成，Interceptor 在 wrapper 链中的位置须正确 |

#### 4.2 实现模板

```python
from openpi_client import base_policy as _base_policy
from openpi.cache.orchestrator import CacheOrchestrator, CheckResult
from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID


class MyInterceptor(_base_policy.BasePolicy):
    """Cache-aware inference wrapper for YourModel."""

    def __init__(self, policy, orchestrator: CacheOrchestrator, timer=None):
        self._policy = policy
        self._model = policy.model              # 你的模型实例
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

        # 2. Stage 1: 观测编码
        encoder_output = self._model.encode(model_inputs)

        # 3. CP1 check
        cp1_result = self._orchestrator.check(
            CheckpointID.CP1,
            encoder_output=encoder_output,
        )

        if cp1_result.hit_type == HitType.FULL_HIT:
            # 缓存命中：跳过后续计算
            cached_action = cp1_result.payload.action_chunk
            self._orchestrator.broadcast_action(cached_action)
            if cp1_result.query_keys is not None:
                self._orchestrator.buffer_for_write(cp1_result.query_keys, cached_action)
            self._orchestrator.clear()
            return self._build_output(inputs, cached_action)

        # 4. 缓存未命中：继续正常推理
        action = self._model.decode(encoder_output)

        # 5. Post-inference: buffer for write
        action_cpu = action[0].detach().cpu().float().contiguous()
        self._orchestrator.broadcast_action(action_cpu)
        if cp1_result.query_keys is not None:
            self._orchestrator.buffer_for_write(cp1_result.query_keys, action_cpu)
        self._orchestrator.clear()

        return self._build_output(inputs, action)

    def _build_output(self, inputs, action) -> dict:
        # 注意：如果在 OpenPI 框架内集成，output_transform 期望的输入是
        # 去掉 batch 维的 CPU numpy 数组，而非 GPU tensor。实际代码应为：
        #   outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        #   outputs = self._output_transform(outputs)
        # 此处为简化的伪代码，请根据你的框架调整。
        outputs = {"actions": action, "state": inputs["state"]}
        return self._output_transform(outputs)
```

> **注意**: 上面的模板是概念性伪代码。在 OpenPI 框架中，`output_transform` 期望输入为去掉 batch 维度的 **CPU numpy 数组**（参见 `policy.py:138` 和 `interceptor.py:341`）。如果你在 OpenPI 内部集成，需要在调用 `_output_transform` 之前执行 `jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)`。

**要点**:
- `orchestrator.check()` 返回 `CheckResult`，`hit_type` 为 `FULL_HIT` 或 `MISS`
- 命中时 `payload.action_chunk` 的形状是 `[action_horizon, action_dim]`（无 batch dim）
- `query_keys` 在所有路径（命中、未命中、gate skip）上都会被填充，确保 `buffer_for_write()` 始终可调用
- 每步推理结束时必须调用 `orchestrator.clear()` 释放 KeyBuilder 缓存

### Step 5: 注册到配置系统

#### 5.1 注册新的 KeyBuilder type

在 `src/openpi/cache/config.py` 的 `_build_key_builder()` 函数中添加你的 builder：

```python
def _build_key_builder(cfg: CacheConfig) -> QueryKeyBuilder:
    kb_type = cfg.key_builder.type
    if kb_type == "placeholder":
        return PlaceholderKeyBuilder()
    elif kb_type == "cp1_mean_pool":
        return CP1MeanPoolKeyBuilder(enabled_fields=...)
    # ... 现有 types ...
    elif kb_type == "my_diffusion_policy":  # <-- 新增
        from your_module import DiffusionPolicyKeyBuilder
        return DiffusionPolicyKeyBuilder()
    else:
        raise ConfigValidationError(f"Unknown key_builder type: {kb_type}")
```

#### 5.2 更新配置校验

在 `validate_cache_config()` 中确保新的 key_builder type 被识别，并添加必要的交叉校验（如检查 `vector_dims` 与新 builder 的输出维度是否匹配）。

#### 5.3 YAML 配置

```yaml
key_builder:
  type: my_diffusion_policy

backend:
  type: in_memory
  vector_dims:
    vision_0: 512      # 必须匹配你的 KeyBuilder.build() 输出维度
    robot_state: 14     # 必须匹配你的状态向量维度
```

`vector_dims` 中的维度**必须**与 KeyBuilder `build()` 返回的每个 field 的 tensor 长度一致。不一致会在 `CacheStorage.insert()` / `CacheStorage.search()` 时触发维度校验错误。

### Step 6: 数据收集与 Artifact 构建

缓存可以**空库启动**——通过 `write_policy` 在 episode 结束时在线累积条目，无需预构建任何东西。但如果你想让缓存从第一个 episode 就有数据可查，可以预构建一个 artifact（pickle 文件），供 `InMemoryBackend` 在启动时加载。

#### 6.1 最小 Artifact 契约

artifact 是一个 pickle 文件，包含一个 **dict**（不是裸列表），格式如下：

```python
{
    "key_builder_type": "my_diffusion_policy",   # 构建时使用的 KeyBuilder 类型
    "checkpoint_id": "CP1",                       # checkpoint 标识
    "vector_dims": {"vision_0": 512, "robot_state": 14},  # 必须与 backend 配置一致
    "entries": [CacheEntry(...), CacheEntry(...), ...],    # CacheEntry 列表
}
```

> **重要**: `InMemoryBackend.load_artifact()` 会校验 artifact 中的 `vector_dims` 与 backend 配置的 `vector_dims` 是否一致，不一致会报错。

每个 entry 需要：

```python
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

entry = CacheEntry(
    id="trajectory_001:step_0",           # 唯一 ID
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

**Tensor 契约**: 所有 tensor 必须是 CPU、contiguous、float32。

#### 6.2 数据收集

你需要从目标模型的推理过程中采集 embedding 和动作数据。

**现有 `collect/` 模块**是 Pi0.5 的参考实现——它硬编码了 `paligemma_with_expert`、`action_in_proj`、`action_out_proj` 等 Pi0.5 特有的 hook 点。对于其他模型，你有两个选择：

1. **参考现有实现自行编写采集逻辑**: 在你的模型推理流程中添加 forward hook 或手动采集中间表示，输出为 HDF5 文件
2. **直接构建 artifact**: 如果你已有离线数据（如 demonstration 轨迹），跳过在线采集，直接编写脚本将数据转换为 `CacheEntry` 列表

**最小 HDF5 schema**（如果你选择走 HDF5 中间格式）:

```
episode_NNNN/
  vision_0/          # [T, vision_dim] float32 — 视觉 embedding 时间序列
  robot_state/       # [T, state_dim] float32 — 状态向量
  clean_action/      # [T, action_horizon, action_dim] float32 — 动作序列
  prompt_emb/        # [T, prompt_dim] float32（可选）
```

#### 6.3 Artifact 构建

现有的 `exp/build_in_memory_cache_artifact.py` 和 `exp/build_clip_cache_artifact.py` 是面向 Pi0.5 HDF5 schema 的参考脚本。迁移时需参考其逻辑编写你自己的构建脚本。

核心流程：

```python
entries = []
for episode in episodes:
    trajectory_id = str(uuid.uuid4())
    for step_idx in range(len(episode)):
        entry = CacheEntry(
            id=f"{trajectory_id}:{step_idx}",
            checkpoint_id=CheckpointID.CP1,
            query_keys=build_query_keys(episode, step_idx),   # 你的逻辑
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

# 序列化为 InMemoryBackend.load_artifact() 要求的 dict 格式
import pickle

artifact = {
    "key_builder_type": "my_diffusion_policy",
    "checkpoint_id": "CP1",
    "vector_dims": {"vision_0": 512, "robot_state": 14},  # 必须与 YAML 配置一致
    "entries": entries,
}
with open("my_artifact.pkl", "wb") as f:
    pickle.dump(artifact, f)
```

在 YAML 中指定 artifact 路径：

```yaml
backend:
  type: in_memory
  in_memory:
    preload_path: /path/to/my_artifact.pkl
```

### Step 7: 验证

#### 7.1 单元测试：KeyBuilder

```python
def test_key_builder_output_dims():
    builder = DiffusionPolicyKeyBuilder()
    fake_output = make_fake_encoder_output()  # 你的测试数据

    builder.collect(CheckpointID.CP1, encoder_output=fake_output)
    keys = builder.build(CheckpointID.CP1)

    assert "vision_0" in keys
    assert keys["vision_0"].shape == (512,)       # 与 vector_dims 匹配
    assert keys["vision_0"].device.type == "cpu"
    assert keys["vision_0"].dtype == torch.float32
    assert keys["vision_0"].is_contiguous()
```

#### 7.2 集成测试：Orchestrator 端到端

```python
def test_orchestrator_check_hit():
    # 构建一个包含已知 entry 的 backend
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

    # 用与 known_entry 相同的输入做 check
    result = orchestrator.check(CheckpointID.CP1, encoder_output=same_input)
    assert result.hit_type == HitType.FULL_HIT
    assert result.payload is not None
```

#### 7.3 实验验证

- 运行你的模型 + cache，观察 cache hit rate
- 对比有/无 cache 的推理延迟
- 检查 cached action 的质量（与实际推理输出的差异）

---

## 4. 可选扩展

### 4.1 Episode Write Path

默认情况下，缓存在每个 episode 结束时，根据 WritePolicy 决定是否将该 episode 的推理数据写入存储。这使得缓存在运行过程中不断积累经验。

配置：

```yaml
write_policy:
  type: on_any_miss    # 有 miss 才写（默认）
  # type: always       # 每个 episode 都写
  # type: never        # 只读模式
```

### 4.2 Trajectory Search

单步搜索只匹配当前观测。Trajectory search 额外考虑**前几步的历史是否也匹配**，倾向选择时间上连贯的缓存序列。

配置：

```yaml
search_strategy:
  type: weighted_score_sum_knn    # 推荐用于 trajectory search
  trajectory_depth: 3              # 回看 3 步
  trajectory_weights: [0.6, 0.3, 0.1]  # 最近一步权重最大
```

> **注意**: 只有 `InMemoryBackend` 支持 trajectory search（`trajectory_depth > 1`）。

### 4.3 CLIP Builder 路线详解

CLIP builder 使用 `open_clip` 的 ViT-B-32 模型将原始输入图像编码为 512 维向量，替代从模型内部提取视觉 embedding。

**适用场景**:
- 你能稳定获取原始输入图像，但不想或无法从模型中间层提取 embedding
- 你想快速验证缓存框架的效果，不投入自定义 KeyBuilder 的开发

**配置**:

```yaml
key_builder:
  type: clip

backend:
  type: in_memory
  vector_dims:
    vision_0: 512       # CLIP ViT-B-32 输出维度
    robot_state: 32     # 仍需状态向量
```

**Artifact 构建**: 使用 `exp/build_clip_cache_artifact.py` 作为参考。它从 HDF5 中读取原始图像，通过 CLIP 编码后构建 artifact。

### 4.4 离线 Artifact 预构建

如果你只想做离线实验（评估缓存命中率，不做在线推理），可以：

1. 从已有的 demonstration 数据构建 artifact（见 Step 6）
2. 编写评估脚本：遍历测试数据，对每一步调用 `orchestrator.check()` 统计命中率
3. 不需要实现 Interceptor

---

## 5. 实验性 / 未完成项

### CP3

CP3 在当前实现中**仅用于 infrastructure validation**。它在 Stage 3 完成后做一次检查，但尚未有完整的"预测下一步跳过推理"逻辑。

迁移时：
- 可以保留 CP3 的框架代码（check + buffer_for_write）以备将来使用
- 不要期望 CP3 在当前版本提供实际的推理加速

### 自定义 Query Field / Checkpoint 扩展

如果现有的 5 个 canonical fields 或 2 个 checkpoint ID 不满足你的需求，需要修改框架本身：

| 扩展类型 | 需修改的文件 |
|----------|-------------|
| 新增 query field | `types.py` (常量), `config.py` (校验), backend `vector_dims`, 相关测试 |
| 新增 checkpoint | `types.py` (enum), `config.py` (checkpoint 配置), `orchestrator.py` (check 逻辑), 相关测试 |

---

## 6. YAML 配置参考

以下是一个模型无关的完整 YAML 模板：

```yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null             # 设置路径以输出 timing CSV

keys:
  vision_0:    { enabled: true,  weight: 1.0 }
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }
  robot_state: { enabled: true,  weight: 1.0 }

key_builder:
  type: my_custom_builder          # 替换为你注册的 type 名称

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
      threshold: 0.95              # 根据你的数据调整

  cp3:
    <<: *cp_defaults
    enabled: false                 # CP3 当前为实验性，建议关闭

backend:
  type: in_memory
  vector_dims:
    vision_0: 512                  # 必须匹配 KeyBuilder 输出维度
    robot_state: 14                # 必须匹配状态向量维度
  in_memory:
    preload_path: /path/to/artifact.pkl

write_policy:
  type: on_any_miss
```

**关键适配点**:
- `keys`: 启用你的 KeyBuilder 实际输出的字段，禁用其余
- `key_builder.type`: 你在 `config.py` 中注册的名称
- `backend.vector_dims`: 每个 enabled field 的维度，必须与 KeyBuilder 输出一致
- `judge.threshold`: 需要根据你的数据校准；初期可用 `always_hit` 做功能验证

---

## 7. 常见问题与陷阱

### 视觉 Token 布局不同

Pi0.5 的视觉 token 来自 SigLIP（256 tokens × 2048 dim per image, 3 images）。你的模型可能有完全不同的布局。

**解决**: 在 KeyBuilder 中自行处理。核心要求只是输出一个固定维度的 1D 向量。你可以用任何池化策略（mean pool、max pool、spatial pool、CLS token）将变长 token 序列降维为固定维度。

### Action 维度不同

Pi0.5 的 `action_chunk` 形状是 `[50, 32]`（50 步 × 32 维动作）。你的模型可能是 `[16, 7]` 或其他。

**解决**: `CachePayload.action_chunk` 的形状没有硬编码限制。只要你的 Interceptor 能正确处理 hit 时返回的 action tensor，任何形状都可以。但同一个 artifact 内的 action 形状必须一致。

### 单 Forward Pass 模型

有些模型（如简单的 MLP policy）没有明确的 stage 分割，只有一个 `forward()` 调用。

**解决方案**:
1. **人为拆分**: 将 encoder 和 decoder 分开调用，即使原本是一次 `forward()`
2. **只用 CP3 语义**: 不在推理中途做缓存检查，而是在推理完成后判断"下一步是否可以跳过"。但注意 CP3 当前为实验性
3. **只做离线 artifact 实验**: 不做在线推理缓存，只评估"如果有缓存，命中率是多少"

### vector_dims 不匹配

**症状**: `CacheStorage` 在 `insert()` 或 `search()` 时抛出维度校验错误。

**原因**: YAML 中的 `backend.vector_dims` 与 KeyBuilder `build()` 返回的 tensor 维度不一致，或与 artifact 中的 entry `query_keys` 维度不一致。

**解决**: 确保三者一致——KeyBuilder 输出维度 = YAML vector_dims = artifact 中的 query_keys 维度。

### 性能调优

- **KeyBuilder 选择**: mean pool 最简单但丢失空间信息；spatial pool 保留空间但维度更高；CLIP 不依赖模型内部但有额外开销
- **Search Strategy**: `weighted_rrf_knn` 适合单步搜索；`weighted_score_sum_knn` 适合 trajectory search
- **Top-k**: 默认 `top_k=1`，增大可以提高召回率但降低搜索速度
- **Threshold**: 过高会导致几乎不命中，过低会导致错误命中。建议先用 `always_hit` + 离线分析来确定合适的阈值
