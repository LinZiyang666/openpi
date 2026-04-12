# Cache 框架迁移教程 — 编写计划

> **Status**: Implemented
> **Task**: 编写教程文档，教导外部开发者如何将本项目的 cache 框架迁移到其他模型上
> **Output**: `docs/cache_migration_guide.md` (中文) + `docs/cache_migration_guide.en.md` (英文)
> **Created**: 2026-04-10

---

## 1. 需求摘要

- **目标读者**: 外部开发者，有自己的模型（非 Pi0.5），想使用我们的 cache 框架加速推理
- **前提**: 读者不需要了解 openpi 项目全貌，文档自包含
- **迁移范围**: CP1 为主线；CP3 作为实验性扩展单独附录
- **定位**: 基于当前 Pi0.5 实现提炼的适配指南，不是"本仓库已对任意模型提供开箱即用支持"
- **文档深度**: 接口级指引 + 适配清单，不深入模型内部细节
- **语言**: 中文 + 英文双份

---

## 2. 架构耦合分析

通过阅读源码，cache 框架与 Pi0.5 的耦合点分为 4 层：

### 2.1 必须适配（模型相关）

| 耦合点 | 文件 | 依赖内容 | 迁移动作 |
|--------|------|----------|----------|
| **Staged API** | `pi0_pytorch.py` | `run_stage1/2/3()` 三段推理接口 | 为目标模型定义等价的 stage 分割 |
| **Stage Output 数据结构** | `pi0_pytorch.py` | `Stage1Output`（prefix_embs, state）、`Stage2Output`、`Stage3Output`（action_chunk） | 定义目标模型的 stage output dataclass |
| **KeyBuilder** | `components/key_builder.py` | 从 `Stage1Output.prefix_embs` 提取 vision tokens（256×2048, SigLIP 特有） | 实现自定义 KeyBuilder，从目标模型的中间表示提取 query key |
| **Interceptor** | `cache/interceptor.py` | 直接调用 `self._model.run_stage1/2/3()`，依赖 `Policy` 的内部结构 | 实现自定义 Interceptor，适配目标模型的推理流程 |

### 2.2 可直接复用（模型无关）

| 组件 | 文件 | 说明 |
|------|------|------|
| CacheOrchestrator | `cache/orchestrator.py` | gate→search→judge→write 流程，不依赖模型 |
| CacheStorage + VectorStoreBackend | `cache/cache_storage.py`, `cache/backend_base.py` | 存储层完全解耦 |
| SearchStrategy (RRF / ScoreSum / Trajectory) | `cache/components/search_strategy.py` | 检索融合逻辑 |
| SimilarityJudge, GateFunction, WritePolicy | `cache/components/*.py` | 可插拔组件 |
| YAML 配置系统 | `cache/config.py` | 需要扩展 key_builder type 注册 |
| SystemTimer | `cache/timing.py` | 通用计时 |
| CacheEntry / CachePayload / QuerySpec | `cache/storage_types.py` | 数据模型通用 |
| CLIP KeyBuilder | `cache/components/clip_key_builder.py` | 用外部 CLIP 编码器，弱化对模型内部视觉表示的依赖（仍需原始图像 + robot_state） |
| 数据收集框架 | `collect/` | **Pi0.5 参考实现**，强耦合 PI0Pytorch 内部模块；其他模型需自行实现等价采集或产出最小 HDF5 schema |

---

## 3. 文档结构计划

### 第1节：概述与前提
- 本框架做什么（多级推理缓存，interceptor pattern）
- **适用范围与非承诺范围**: 明确声明——本教程基于当前 Pi0.5 实现提炼，不代表本仓库已对非 Pi0.5 模型提供开箱即用支持
- 迁移意味着什么：你需要做哪几件事
- 你不需要了解什么（openpi 具体模型、训练流程等）
- **迁移路径判定**: 根据你的模型特征（能否切 stage、能否拿中间表示、能否拿原始图像）引导到不同迁移路径

### 第2节：架构总览（迁移视角）
- 一张分层图：模型相关层 vs 模型无关层
- 明确标注：哪些你需要写、哪些直接复用
- Checkpoint 语义：CP1 / CP3 在 Pi0.5 中的含义，以及如何在你的模型中找到对应位置
  - 注意：当前框架只支持 CP1 / CP3 两个 checkpoint ID；新增 checkpoint 属于框架扩展，不在本教程范围

### 第3节：最小可迁移主线（CP1）

#### Step 1: 分析你的推理流程
- 将你的模型推理拆分为 stages（或识别可中断点）
- 在你的模型中找到对应 CP1 语义的位置（"观测编码完成后，跳过后续计算"）
- 确定每个 stage 的输出中哪些信息可以作为 cache key

#### Step 2: 定义最小接口契约
- **最低契约**（与 Pi0.5 dataclass 无关）:
  - CP1 检查点之前的输出必须能提供 KeyBuilder 所需输入（如视觉 embedding、状态向量）
  - 最终输出必须能提供 `action_chunk`（CachePayload 的核心）
  - 推理流程必须能支持 episode lifecycle（start/end/step 信号）
- 附 Pi0.5 的 `Stage1Output` / `Stage3Output` 作为参考示例

#### Step 3: 实现自定义 KeyBuilder
- 实现 `QueryKeyBuilder` protocol（`collect()` + `build()` + `cached_data` + `clear()`）
- `collect()`: 从 stage output 提取原始 tensor（GPU 上，不拷贝）
- `build()`: 降维 + GPU→CPU 转换，返回 `dict[str, Tensor]`
- **字段映射规则**: 必须映射到现有 5 个 canonical fields（`vision_0/1/2`, `prompt_emb`, `robot_state`）
  - 低成本路径：把目标模型的中间表示映射到这些字段
  - 深度改造（超出本教程范围）：如需新增字段，须修改 `types.py`、`config.py`、backend 维度声明及相关测试
- **CLIP 分支**: 当你能稳定拿到原始图像和状态、但不想复用模型内部视觉 embedding 时，可考虑 CLIP builder（仍需 `input_images` + `robot_state`）

#### Step 4: 实现自定义 Interceptor
- **宿主运行时最小契约清单**:
  - 包装对象满足 `BasePolicy` 语义（`infer(obs) -> dict`）
  - 能访问底层模型的 staged 推理方法
  - 能访问 input/output transform 管线
  - 能获取 pytorch device 信息
  - 支持 episode lifecycle 回调（`on_episode_start` / `on_episode_end`）
  - 如果与 `serve_policy.py` 集成，wrapper ordering 需正确（如 `CollectionPolicy` 能沿 `_policy` 链找到 `_model`）
- 在 stage 之间插入 `orchestrator.check()` 调用
- 处理 hit/miss 分支
- 实现 episode lifecycle 广播（broadcast_action / buffer_for_write）

#### Step 5: 注册到配置系统
- 在 `config.py` 的 `_build_key_builder()` 中注册新的 key_builder type
- 更新 YAML 配置的 `vector_dims` 以匹配新 key builder 的输出维度
- 验证配置交叉检查规则

#### Step 6: 数据收集与 Artifact 构建
- **最小 artifact 契约**: 最终需要产出包含 `query_keys` + `CachePayload` 的 `CacheEntry` 列表，序列化为 pickle
- **数据收集**: 现有 `collect/` 是 Pi0.5 参考实现（硬编码了 `paligemma_with_expert`、`action_in_proj` 等 hook 点）；对其他模型需自行实现等价采集，或直接产出最小 HDF5 schema
- **Artifact 构建脚本**: 现有 `exp/build_in_memory_cache_artifact.py` 和 `exp/build_clip_cache_artifact.py` 是 Pi0.5 HDF5 schema 的参考脚本，不是通用 builder；迁移时需参考其逻辑自行编写

#### Step 7: 验证
- 单元测试：KeyBuilder 的 collect/build 输出维度正确
- 集成测试：Orchestrator 端到端 check → hit/miss
- 实验验证：cache hit rate 和 action 质量

### 第4节：可选扩展
- Episode write path（trajectory linked list）
- Trajectory search（多步历史匹配）
- CLIP builder 路线详解
- 离线 artifact 预构建

### 第5节：实验性 / 未完成项
- **CP3**: 当前仅为 infrastructure validation，尚未有完整的"预测下一步跳过推理"逻辑；迁移时可保留框架代码但不期望获得实际加速
- 自定义 query field / 新 checkpoint 名称的框架扩展

### 第6节：YAML 配置参考
- 完整 YAML 模板（通用版，不特定于 Pi0.5）
- 各字段含义和适配要点

### 第7节：常见问题与陷阱
- vision token 布局不同怎么办
- action 维度不同怎么办
- 没有明确的 stage 分割怎么办（单 forward pass 模型）
- 性能调优建议

---

## 4. 涉及文件

| 操作 | 文件 |
|------|------|
| **新建** | `docs/cache_migration_guide.md` (中文) |
| **新建** | `docs/cache_migration_guide.en.md` (英文) |
| **更新** | `docs/README.md` (添加索引) |
| **更新** | `logs/README.md` (添加此 plan) |

---

## 5. 不做的事

- 不修改任何现有代码
- 不为某个具体外部模型写完整适配代码
- 不深入讲解 openpi 的训练、部署流程
- 不重复 `cache_system_architecture.md` 或 `cache_system_tutorial.md` 的内容，只引用

---

## 6. 审查

> 审查日期: 2026-04-10
> 角色: reviewer only
> 审查范围: `CLAUDE.md`、`WORKING_AGREEMENT.md`、`docs/cache_system_architecture.md`、`docs/cache_system_tutorial.md`、`docs/data_collection_guide.md`、`src/openpi/cache/`、`src/openpi/collect/`、`exp/build_in_memory_cache_artifact.py`、`exp/build_clip_cache_artifact.py`

### 6.1 主要问题

#### 问题 1: 文档支持边界没有说清，容易让读者误以为“本仓库已经官方支持任意模型迁移”

- 疑问:
  这篇教程是要讲“如何在本仓库内把 cache 接到别的模型上”，还是讲“如何把这套 cache 设计迁到你自己的外部项目里”？
- 原因:
  `WORKING_AGREEMENT.md` 和 `docs/cache_system_architecture.md` 都把当前系统的正式范围写得很明确：这是 **PyTorch + Pi0.5 only** 的实现。现在写“迁移到其他模型”的教程，如果不先写清楚支持边界，读者会自然理解成“这套代码已经抽象到模型无关，只差几处适配”。
- 建议:
  在教程开头单独加一节“适用范围 / 非承诺范围”。
  明确写成：本教程是“基于当前 Pi0.5 实现提炼出的适配指南”，不是“本仓库已经对非 Pi0.5 模型提供开箱即用支持”。

#### 问题 2: `CP1 / CP3 / 数据收集 / artifact / 实验流程` 全覆盖的表述过满，和当前实现状态不一致

- 疑问:
  这篇教程是否真的打算把 CP3 作为一条可落地、可复用的主线来教？
- 原因:
  `docs/cache_system_tutorial.md` 明确写了 `CP3` 当前只是 **infrastructure validation only**。
  `src/openpi/cache/README.md` 也写了 `CP3` 真正实现推迟到 Step 6。
  当前稳定可讲的主线其实是 `CP1`；`CP3` 还不能作为“迁移后即可获得的能力”来承诺。
- 建议:
  把范围改成：
  `CP1` 为主教程；
  `CP3` 作为“实验性扩展 / 当前限制”单独一节；
  “实验流程”只覆盖现有 `CP1` 流程，不要写成通用的 `CP1/CP3` 实验流水线。

#### 问题 3: plan 里把“字段命名规则（CACHE_QUERY_FIELDS 或自定义）”说得过头了，当前代码并不支持自定义字段名

- 疑问:
  你是想教用户“映射到现有 5 个 canonical fields”，还是想教用户“扩展框架以支持新的 field taxonomy”？
- 原因:
  `src/openpi/cache/types.py` 把 query fields 固定为 `vision_0`、`vision_1`、`vision_2`、`prompt_emb`、`robot_state`。
  `src/openpi/cache/config.py` 的校验也只接受这几个字段。
  所以现状不是“字段可以自定义”，而是“你必须先映射到这几个字段；如果做不到，就要改类型常量、config 校验、backend 约束以及相关文档”。
- 建议:
  在教程里把这件事拆成两档：
  1. 低成本迁移：把目标模型的中间表示映射到现有 canonical fields。
  2. 深度改造：如果需要新增字段，必须修改 `src/openpi/cache/types.py`、`src/openpi/cache/config.py`、backend 维度声明和相关测试。

#### 问题 4: “Checkpoint 概念：如何定义你自己的 checkpoint 位置” 需要更精确，不然容易误导成“可以随便新增 CP4/CP5”

- 疑问:
  这里说的“定义你自己的 checkpoint”，是重新解释 `CP1/CP3` 在新模型中的语义，还是新增新的 checkpoint ID？
- 原因:
  当前实现层面只支持 `cp1` / `cp3`。
  `src/openpi/cache/config.py` 明确只接受这两个 checkpoint 名。
  `src/openpi/cache/types.py` 里 `CheckpointID` 虽然还有 `CP2`，但实际配置和教程都只承认 `CP1`、`CP3`。
- 建议:
  文档措辞改成：
  “你需要在自己的模型里找到分别对应 `CP1` 和 `CP3` 语义的位置”；
  不要写成“自由定义 checkpoint 位置和命名”，除非教程同时准备讲清楚如何扩展 enum、config、orchestrator 配置和测试。

#### 问题 5: 对 Interceptor 适配成本描述偏轻，真实耦合点不止 `run_stage1/2/3()`

- 疑问:
  教程是否会把“只要实现一个自定义 Interceptor”说成轻量工作？
- 原因:
  `src/openpi/cache/interceptor.py` 现实中依赖的东西比表格里写的多：
  - 包装对象要满足 `BasePolicy` 语义
  - 依赖 `_model`、`_input_transform`、`_output_transform`、`_pytorch_device`
  - 依赖 staged API 的调用时机
  - 依赖 episode lifecycle：`on_episode_start()` / `on_episode_end()`
  - `scripts/serve_policy.py` 还要求 wrapper ordering 正确，`CollectionPolicy` 要能沿 `_policy` 链找到 `_model`
- 建议:
  教程里单独列一个“宿主运行时最小契约”清单，比“实现一个自定义 Interceptor”更准确。
  否则外部读者会低估接入成本。

#### 问题 6: 数据收集部分被写成“需要适配 hook 点”，但当前收集实现其实强依赖 Pi0.5 内部模块结构

- 疑问:
  这部分教程是要教“复用现有收集器”，还是教“参考现有收集器自己重写一份”？
- 原因:
  `src/openpi/collect/collection_policy.py` 明确只支持 PyTorch `PI0Pytorch` 路径，直接硬编码了：
  - `paligemma_with_expert`
  - `action_in_proj`
  - `action_out_proj`
  - 以及特定 hook 点的语义
  `docs/data_collection_guide.md` 的 HDF5 schema 也是按 Pi0.5 当前张量布局定义的。
- 建议:
  不要把 `collect/` 描述成“模型无关的数据收集框架”。
  更准确的写法是：
  现有 `collect/` 是 Pi0.5 参考实现；
  对其他模型，你要么提供等价 hook，要么直接自己产出教程要求的最小 HDF5 schema。

#### 问题 7: artifact 构建脚本并不通用，当前 plan 需要把它们降级成“参考样例”

- 疑问:
  教程是否准备直接告诉读者“适配 `exp/build_in_memory_cache_artifact.py` 就行”？
- 原因:
  `exp/build_in_memory_cache_artifact.py` 假定 HDF5 里有 `vision_0/1/2`、`prompt_emb`、`robot_state`、`clean_action`，还会重建 Pi0.5 的 `prefix_embs` token 布局。
  `exp/build_clip_cache_artifact.py` 也假定存在 `input_images`、`prompt_emb`、`robot_state`。
  这两份脚本都不是“模型无关 artifact builder”，而是“面向当前 HDF5 schema 的 builder”。
- 建议:
  教程里把 artifact 部分拆成：
  - 最小 artifact 契约：最终需要产出什么 `CacheEntry`
  - Pi0.5 参考脚本：现有两份 `exp/` 脚本只是 schema-specific example
  这样更稳，不会让读者误判为“只需改几行脚本”。

#### 问题 8: `CLIP KeyBuilder` 不能被表述成完全模型无关的万能退路

- 疑问:
  教程想把 CLIP 路线放成“如果你的模型中间表示不好取，就直接用 CLIP”吗？
- 原因:
  这个说法只对一半。
  当前 `clip` 路线仍然依赖：
  - 输入图片抽取路径（`input_images`）
  - 固定图片槽位命名
  - `robot_state` 字段
  - 某些配置下还依赖 `prompt_emb`
  所以它只是“弱化对模型内部视觉 token 的依赖”，不是“完全摆脱宿主系统耦合”。
- 建议:
  把 CLIP 方案写成一条特殊分支：
  “当你能稳定拿到原始图像和状态，但不想复用模型内部视觉 embedding 时，可考虑 CLIP builder。”
  不要把它写成默认通用方案。

### 6.2 结构建议

#### 建议 1: 文档结构按“稳定主线 / 实验扩展 / 深度改造”分层

- 原因:
  当前实现成熟度明显不均衡。把所有内容压成一个 7-step 主流程，会掩盖“哪些是稳定能力，哪些只是历史实现细节，哪些还在实验中”。
- 建议:
  推荐改成三层：
  1. 最小可迁移主线：`CP1 + 自定义 KeyBuilder + 自定义 Interceptor`
  2. 可选扩展：episode write、trajectory search、CLIP builder、离线 artifact
  3. 实验/未完成项：`CP3`、更深的 checkpoint 设计、自定义 field / checkpoint 扩展

#### 建议 2: 把“Stage Output 数据结构”从 Pi0.5 例子提升为“最小接口契约”

- 原因:
  当前 plan 写法容易让读者以为必须复制 `Stage1Output` / `Stage2Output` / `Stage3Output` 的字段设计。
  但对迁移教程来说，真正重要的是：
  - `stage1` 能提供 KeyBuilder 所需输入
  - `stage3` 能提供 `action_chunk`
  - lifecycle 能把 query_keys 和 action 串起来
- 建议:
  文档里先定义“最低契约”，再附 Pi0.5 示例 dataclass。
  这样教程才不会被当前模型的字段名绑死。

#### 建议 3: 单独增加“迁移判定树”

- 原因:
  外部读者最大的困难不是看懂组件，而是不知道自己属于哪种迁移类型。
- 建议:
  可以在开头放一个 decision tree：
  - 你能切 stage 吗？
  - 你能拿到中间表示吗？
  - 你能拿到原始图像吗？
  - 你需要在线 cache，还是只做离线 artifact 实验？
  这会比单纯线性步骤更实用。

### 6.3 我建议补充回答的几个问题

1. 教程目标到底是“在本仓库里接入别的模型”，还是“把这套 cache 设计迁到外部项目”？
2. 教程是否明确把 `CP1` 作为唯一主线，把 `CP3` 降为实验性附录？
3. 教程是否允许“新增 query field / 新 checkpoint 名称”？如果允许，是否要把这类修改列为“框架扩展”，而不是“普通迁移”？
4. 数据收集部分是否改成“给出最小 HDF5 契约”，而不是暗示现有 `collect/` 可以直接复用？
5. artifact 构建部分是否改成“定义产物契约 + 引用 Pi0.5 参考脚本”，而不是把现有 `exp/` 脚本表述成通用工具？

### 6.4 结论

这个 plan 的方向是对的，但当前版本把“可迁移的抽象边界”说得比实际代码更宽。
我建议先把教程定位收窄为：
“如何把当前 cache 框架的 **CP1 主线** 迁到其他模型，并说明哪些部分仍是 Pi0.5-specific reference implementation。”
先把这个边界讲清楚，后面的文档会稳很多，也更不容易和 `docs/cache_system_architecture.md`、`docs/cache_system_tutorial.md`、`docs/data_collection_guide.md` 的现状冲突。

---

## 7. 审查回复（Plan 作者）

> 回复日期: 2026-04-10

逐条回复：

### 问题 1（支持边界）：✅ 采纳

有道理。已在 §1 需求摘要中加入明确定位声明，并在 §3 第1节增加"适用范围与非承诺范围"小节。

### 问题 2（CP3 范围过满）：✅ 采纳

同意。CP3 当前只是 infrastructure validation，不应和 CP1 并列作为主教程。已将文档结构改为：
- §3 主线只覆盖 CP1
- §5 单独列出 CP3 作为"实验性/未完成项"

### 问题 3（自定义字段名）：✅ 采纳

确认 `CACHE_QUERY_FIELDS` 是硬编码的 5 个字段。已在 Step 3 中明确拆分为：
- 低成本路径：映射到现有 canonical fields
- 深度改造：须改 `types.py`、`config.py`、backend、测试（标注为超出本教程范围）

### 问题 4（自定义 checkpoint）：✅ 采纳

确认 `CheckpointID` 是固定 enum，config 只接受 `cp1`/`cp3`。已改为"在你的模型中找到对应 CP1/CP3 语义的位置"，删除"定义你自己的 checkpoint"的误导表述。新增 checkpoint 标注为框架扩展。

### 问题 5（Interceptor 成本）：✅ 部分采纳

审查者指出的额外依赖点（`_input_transform`、`_output_transform`、`_pytorch_device`、wrapper ordering）是真实的。已在 Step 4 中增加"宿主运行时最小契约清单"，列出全部实际依赖项。

但不同意"适配成本偏轻"这个判断。教程的目标就是把这些依赖摊开来讲清楚，让读者自行评估。教程不是在隐瞒成本，而是在提供成本清单。

### 问题 6（数据收集强耦合）：✅ 采纳

确认 `CollectionPolicy` 硬编码了 `paligemma_with_expert`、`action_in_proj`、`action_out_proj`。已在耦合分析表和 Step 6 中明确标注为"Pi0.5 参考实现"，改为"自行实现等价采集或直接产出最小 HDF5 schema"。

### 问题 7（artifact 脚本不通用）：✅ 采纳

已在 Step 6 中将 artifact 部分拆为：
- 最小 artifact 契约（`CacheEntry` 列表 → pickle）
- 现有脚本降级为"Pi0.5 HDF5 schema 参考脚本"

### 问题 8（CLIP 非万能）：⚠️ 部分采纳

同意 CLIP 不是零耦合方案——仍依赖 `input_images` 和 `robot_state`。已将措辞改为"弱化对模型内部视觉表示的依赖"。

但不同意把 CLIP 说成和其他 KeyBuilder 一样重的耦合。CLIP builder 的核心价值是**不需要访问模型内部 embedding**，只需要原始传感器输入，这对很多模型来说是最低门槛的路径。教程中会作为一条推荐的低耦合分支来介绍，但会加上限制条件说明。

### 结构建议 1（三层分层）：✅ 采纳

已将文档结构改为：
- §3 最小可迁移主线（CP1）
- §4 可选扩展（trajectory、CLIP、离线 artifact）
- §5 实验性/未完成项（CP3、框架扩展）

### 结构建议 2（最小接口契约）：✅ 采纳

已在 Step 2 中改为"最小接口契约"为主，Pi0.5 dataclass 降为参考示例。

### 结构建议 3（迁移判定树）：⚠️ 简化采纳

判定树的想法有价值，但完整决策树会让文档开头过于复杂。改为在第1节中加入简要的"迁移路径判定"引导（3-4 个关键问题），不做完整树状图。

### 对 §6.3 补充问题的回答

1. **教程定位**: "在本仓库基础上把 cache 接到别的模型上"。不是教人在完全独立的项目里从零搭建。
2. **CP1 为主线，CP3 降为附录**: 是，已修改。
3. **新增 field/checkpoint**: 属于框架扩展，超出本教程范围，但会在 §5 中提及存在这个可能性以及需要改哪些文件。
4. **数据收集**: 是，改为最小 HDF5 契约 + Pi0.5 参考实现。
5. **artifact**: 是，改为产物契约 + 参考脚本。

### 总结

8 个问题中 5 个完全采纳，3 个部分采纳。Plan 正文已同步修改。核心变化：
- 定位从"全覆盖迁移指南"收窄为"CP1 主线适配指南 + 可选扩展 + 实验附录"
- 所有 Pi0.5-specific 的组件（collect/、artifact 脚本）降级为参考实现
- 增加最小契约概念，解耦教程与 Pi0.5 具体字段设计
