# Step 4 Discussion: Orchestrator Skeleton (CP1 + CP3)

> This file keeps the design discussion and defense record only.
> The full implementation plan was moved to `logs/step4_plan.log.md`.

## 讨论 1: 组件稳定性分析

### 背景

Step 3（cache 存储层）标记为 `⚠️ 暂时完成·高危`，原因：
- Vector DB 选择未确定（Qdrant 可能被替换）
- 查询方式未确定（similarity 类型、multi-field fusion 策略、top_k 等）
- 无测试覆盖，接口可能变动

Step 4 的 6 个子模块需要逐一分析：哪些因与 Step 3 耦合而不稳定，哪些因自身设计未定而不稳定。

### 分析结果

#### 稳定性分级

| 组件 | 与 Step 3 耦合度 | 自身不稳定度 | 综合风险 | 说明 |
|------|------------------|-------------|---------|------|
| 4.1 KeyBuilder | **高** | **高** | 🔴 高危 | 见下文 |
| 4.2 Gate | **无** | **低** | 🟢 稳定 | 见下文 |
| 4.3 Judge | **高** | **中** | 🔴 高危 | 见下文 |
| 4.4 Orchestrator | **中** | **低** | 🟡 中等 | 见下文 |
| 4.5 Interceptor 集成 | **低** | **低** | 🟢 稳定 | 见下文 |
| 4.6 端到端测试 | **高** | **中** | 🔴 高危 | 见下文 |

---

#### 4.1 KeyBuilder — 🔴 高危（双重不稳定）

**与 Step 3 耦合（高）**：
- `CacheEntry.query_keys` 是一个 `dict[str, Tensor]`，键名必须匹配 `VectorStoreBackend.vector_dims`
- KeyBuilder 输出的 field 名称和维度直接受 backend 配置约束
- 如果 backend 从 Qdrant（支持 named vectors + RRF fusion）换成 FAISS（单向量），整个 multi-field 设计就要变
- `types.py` 定义了 5 个 canonical field（vision_0/1/2, prompt_emb, robot_state），但哪些 field 真正有用取决于 backend 配置和实验结果

**自身不稳定（高）**：
- Step 5 实验 C 会比较多种 key 构建方式（raw state / vision mean pool / state+vision concat / state+action），实验结果将直接推翻当前的 PlaceholderKeyBuilder
- CP1 和 CP3 用不同的 key 构建策略，但具体策略是实验驱动的
- L2 normalize 是否是最佳归一化方式也未确定

---

#### 4.2 Gate — 🟢 稳定

**与 Step 3 耦合（无）**：
- Gate 只决定"是否查缓存"，不接触 storage 层

**自身不稳定（低）**：
- `AlwaysSearchGate` 是最简实现，几乎不可能出错
- 高级 Gate（StateChangeGate, IntervalGate）推迟到 Step 8
- Protocol 接口 `__call__(context) -> bool` 非常简单，不太可能需要改

---

#### 4.3 Judge — 🔴 高危（双重不稳定）

**与 Step 3 耦合（高）**：
- `SearchResultLite.score` 的语义完全取决于 backend：
  - 单字段 cosine backend：score ∈ [-1, 1]
  - 多字段 RRF fusion：score 是小正数，量级由 RRF k 参数决定
- `storage_types.py` 注释明确说 "SimilarityJudge thresholds must be calibrated to match the backend/mode in use"
- 换 backend 或换 fusion 策略 → threshold 数值完全失效，需要重新标定

**自身不稳定（中）**：
- Step 5 实验 B 会系统地扫描 threshold（0.99 → 0.80），ThresholdJudge 的默认值 0.98 只是起点
- 未来可能需要 warm-start 判定逻辑（多级阈值），但这是 Step 7 的事

---

#### 4.4 Orchestrator — 🟡 中等

**与 Step 3 耦合（中）**：
- Orchestrator 通过 `CacheStorage` facade 交互，不直接接触 backend
- `CacheStorage` 的接口（search/insert/fetch_payload）相对稳定
- 但 `QuerySpec` 的构造（top_k、filters、query_keys 字段选择）会随 backend 变化

**自身不稳定（低）**：
- `check()` 的 gate → key → search → judge 流程是确定的
- `write_async()` 先用同步写入，这个模式不太会变
- 作为组合器，它的稳定性取决于子组件，自身逻辑简单

---

#### 4.5 Interceptor 集成 — 🟢 稳定

**与 Step 3 耦合（低）**：
- Interceptor 只调用 Orchestrator 的 `check()` / `write()`，不碰 storage

**自身不稳定（低）**：
- 三个 TODO 槽位明确（line 285/290/294），插入位置不会变
- CP2 保持注释，CP1 和 CP3 的插入逻辑是确定的
- 唯一可能变化：check() 返回值的处理方式（hit 时跳过哪些 stage），但这在架构文档中已经定义清楚

---

#### 4.6 端到端测试 — 🔴 高危（双重不稳定）

**与 Step 3 耦合（高）**：
- 测试需要实例化 backend，如果 Qdrant backend 不稳定/被替换，测试也得改
- 测试中的 score 断言（如 "应该 hit"）依赖 backend 的 score 语义

**自身不稳定（中）**：
- "加载模型跑 10 次" 需要真实模型或 mock — mock 策略尚未确定
- 测试用例的具体断言值（threshold、L2 distance 容差）需要调试

---

### 结论

**高危组件（🔴）**：KeyBuilder, Judge, 端到端测试 — 这三个与 Step 3 强耦合，且自身也会因 Step 5 实验而改变。实现时应：
1. 保持接口极简（Protocol），具体实现随时可替换
2. 避免在这些组件中做"聪明"的设计，用最简 placeholder
3. 测试用 in-memory mock backend 而非 Qdrant，隔离 Step 3 的不稳定性

**稳定组件（🟢）**：Gate, Interceptor 集成 — 可以一次做好，不太需要改。

**中等组件（🟡）**：Orchestrator — 流程确定但参数构造会变，设计时预留灵活性。

### 修订后的稳定性评估

基于讨论 1 的初始分析 + 用户反馈，修订如下：

| 组件 | 与 Step 3 耦合 | 自身不稳定 | 综合 | 备注 |
|------|---------------|-----------|------|------|
| 4.1 KeyBuilder | **高** | **高** | 🔴 | 需灵活设计：收集所有数据，config 控制开关 |
| 4.2 Gate | **无** | **高**（修订↑） | 🟡 | 实验阶段会频繁更换 Gate 实现 |
| 4.3 Judge | **高** | **高**（修订↑） | 🔴 | 类似 Gate 会频繁变动，且受 Step 3 score 语义影响 |
| 4.4 Orchestrator | **中** | **低** | 🟢（修订↑） | 不直接接触 Step 3，子组件不稳定但调用流程确定 |
| 4.5 Interceptor 集成 | **低** | **低** | 🟢 | 确定，对模型无感 |
| 4.6 端到端测试 | — | — | 推迟 | 写完其他代码后再考虑 |

---

## 讨论记录

### 讨论 1

**Q（用户）**: Step 3 是高危不稳定步骤，Step 4 中哪些组件因与 Step 3 耦合而变为 unstable？哪些组件自身也可能频繁改变？

**A**: 见上文初始分析表格。KeyBuilder 和 Judge 是双重不稳定，Gate 和 Interceptor 集成稳定，Orchestrator 居中。

### 讨论 2

**用户反馈**（修正初始分析）：
1. **Gate**：实验阶段会有各种各样的 Gate，频繁更新逻辑 → 自身不稳定度应提升到高
2. **Orchestrator**：用户认为还好，虽然子组件（如 QuerySpec）不稳定，但调用流程是确定的 → 综合风险可降低
3. **端到端测试**：暂不考虑，写完代码后再做
4. **KeyBuilder**：应设计为灵活模式——收集所有可用数据，通过 config/开关决定用哪些。`--collect` 功能已做了简单数据收集，可考虑集成
5. **Judge**：与 Gate 类似会频繁变动，但因数据上游是 Step 3 所以额外不稳定
6. **Interceptor 集成**：赞同稳定，且整套 cache 系统对模型应该是无感的（除了 Gate/Judge 会中断或开始推理流程）

### 讨论 3：KeyBuilder 与 --collect 集成 + Hook vs Stage 输出

**Q（用户）**: KeyBuilder 与 --collect 的集成方向？AB 两种方式有何区别？

**A**: 两种数据获取方式的核心区别：

| 维度 | A: Forward Hook | B: 读取 Stage 输出 |
|------|----------------|-------------------|
| 额外拷贝 | 每次 hook 需 `.detach().clone()`，产生 GPU 内存分配 | 零拷贝，直接读返回值 |
| 调用开销 | Python 回调插入 CUDA kernel 之间，可能破坏 GPU pipeline | 无额外调用 |
| torch.compile 兼容性 | Hook 是 Python 回调，可能打断 traced graph | Stage 函数可完整 compile |
| 数据粒度 | 能抓模型任意层中间输出（如 flow matching 每步的 noise action） | 只能拿 Stage API 暴露的字段 |

关键差异：Stage API 暴露的是 stage1(vision+state)、stage2(KV cache)、stage3(final action)，**拿不到 noise_action_steps 等模型内部中间数据**。但如果 KeyBuilder 只需要 state 和 final action（当前 Placeholder 设计），方式 B 零开销且完全够用。

**决定**：采用方式 B（读取 Stage 输出），不与 --collect 集成。理由：
- 零性能开销
- 不破坏 torch.compile
- 当前和近期需求（state, action）可满足
- 如果未来需要更细粒度数据，再针对性扩展 Stage API 输出

### 讨论 4：Full Skip vs Warm Start 决策者 + Gate/Judge 与 KeyBuilder 数据关系

**Q（用户）**: 两种 cache 使用方式（直接返回 clean action vs 返回 noise action 部分跳过），决策是谁做的？

**A**: 各组件职责分工：
- **Gate**：只决定"要不要查缓存"（True/False）
- **Judge**：判断命中类型（MISS / FULL_HIT / WARM_START）
- **Orchestrator**：编排流程，打包结果
- **Interceptor**：根据命中类型控制数据流（跳过/部分执行/完整执行）

以 CP2 为例，Judge 有三级阈值：
- similarity > cp2_full_threshold → FULL_HIT（返回 cached clean action）
- similarity > cp2_warm_threshold → WARM_START（返回 cached noise action x_t，跑剩余 flow matching 步骤）
- similarity < cp2_warm_threshold → MISS

**Q（用户）**: Gate 和 Judge 是否会用到 KeyBuilder 所获取的数据以及 build 的 key？

**A**: 分析如下：

**Gate — 部分实现会用到**：
- `AlwaysSearchGate`：不需要任何数据
- `StateChangeGate`（未来）：需要比较当前 state 与上次 state 的距离 → 可直接复用 KeyBuilder build 好的 key（已 L2 normalized），计算 cosine distance
- `VisionChangeGate`（假设）：需要视觉 embedding → 同样是 KeyBuilder 的数据源

**Judge — 间接依赖 + 未来可能直接依赖**：
- `ThresholdJudge`：只看 score，不直接用 key。但 score 是 query key 搜索的结果，key 变了 → score 分布变了 → threshold 失效（间接耦合）
- 未来 `ReScoreJudge`：可能用更丰富的原始数据做二次评分 → 需要访问 KeyBuilder 收集的原始数据

**结论**：Gate 和 Judge 都可能需要访问 KeyBuilder 收集/构建的数据。设计上需要一个共享数据通道。

**决定**：
1. KeyBuilder 负责暂存收集到的原始数据和 built key，Gate/Judge 从 KeyBuilder 读取
2. Gate 和 Judge 的接口设计需兼容 KeyBuilder 的数据（接收 KeyBuilder 引用或其暂存数据）
3. **硬件开销最小化原则**：
   - 暂存数据尽量保持在原始设备上（GPU tensor 不做不必要的 CPU 拷贝）
   - 只在真正需要时才移动数据（如写入 storage 时才 `.cpu()`）
   - built key 如果是 GPU tensor，Gate 做 cosine distance 比较也在 GPU 上完成，避免 sync
   - 暂存生命周期 = 单次推理周期，推理结束后立即释放引用，不累积内存
   - 避免 `.clone()` — 如果 stage 输出在下一次推理前不会被覆盖，直接持有引用即可
4. **与 Step 3 交互必须通过抽象层**：Step 3 设计了 `VectorStoreBackend`(ABC) → `CacheStorage`(facade) 的多层结构来隔离底层 vector DB。Step 4 中所有与 storage 交互的组件（Orchestrator、KeyBuilder 写入、Judge 间接依赖 score）必须只通过 `CacheStorage` facade 操作，**绝不直接调用 backend 或 Qdrant API**。这样 backend 替换时 Step 4 代码无需改动。

### 讨论 5：Timing 集成

**用户要求**：Step 4 代码需接入 Step 2 的 SystemTimer 实现精确计时，且不过多影响性能。

**设计决定**：
- **Orchestrator 统一管理计时**：组件本身（KeyBuilder/Gate/Judge）不直接持有 timer 引用，计时在 Orchestrator 的 check()/write() 中对每个子步骤做 `with timer.measure()`
- **全部使用 cpu backend**（`PerfCounterBackend`）：Gate/Judge 是纯 Python 逻辑；KeyBuilder.build() 虽含 GPU normalize 但主要耗时在 D2H transfer；CacheStorage.search() 是 CPU 调度。用 CUDA event 反而引入不必要的 event record 开销
- **性能影响极小**：`perf_counter_ns()` 调用 ~50ns，每次 check 约 6 个 probe = ~300ns，相对推理延迟 ~50ms 可忽略；`enabled=False` 时 `measure()` 是纯 no-op（零开销）

**Probe 列表**：

Interceptor 级（粗粒度）：`cp1_check`, `cp1_write`, `cp3_check`
Orchestrator 级（细粒度）：`cp{1,3}_{collect, gate, build, search, judge, fetch}`

---


## 答辩板块

> 角色说明：以下内容仅记录提问人的质疑点。每个议题只保留问题、原因和依据，用于后续答辩与 plan 修订。

### 议题 1：CP3 的真正接入点是否缺失？

**问题**：
当前 `src/openpi/cache/interceptor.py` 里只有三个 `TODO(Step 4)` 槽位：Stage 1 后、Stage 2 后、Stage 3 后。  
但 `docs/cache_system_architecture_chinese.md` 里 `should_skip_inference()` 的语义是“在推理开始前调用，如果上一周期 CP3 预调度了 cached action，直接返回并跳过整次推理”。  
如果 Step 4 仍坚持“现有三个 TODO 槽位足够”，那么 CP3 如何实现“跳过下一次完整推理”？

**原因**：
- 当前 plan 把 CP3 的“判定位置”和“消费位置”混在了一起。
- 若没有前置消费点，CP3 最多只能做到“记住一个 action”，不能做到真正 skip。

**依据**：
- `src/openpi/cache/interceptor.py`
- `docs/cache_system_architecture_chinese.md`

**答辩**：

质疑成立。当前 plan 确实遗漏了 CP3 的"消费接入点"。

架构文档明确描述了 `should_skip_inference()` 应在推理开始前调用（`infer()` 顶部），而现有三个 TODO 槽位都在推理内部。CP3 的完整工作流是：

1. **写入点**（Stage 3 之后）：记录"当前 action → 下一步 action"映射
2. **消费点**（下一次 `infer()` 入口处）：检查是否有预调度 action，有则直接返回跳过整次推理

但 `should_skip_inference()` + `schedule_next_action()` 的完整机制依赖 **Step 6 的 DeferredWriter**（因为 `next_action_chunk` 要等下一 cycle 才能补全）。Step 4 无法实现完整的 CP3 跳过。

**Plan 修正**：Step 4 中 CP3 降级为"骨架基础设施"：
- 在 Orchestrator 中预留 `schedule_next_action()` 和 `should_skip_inference()` 的接口桩（空实现），在 Interceptor `infer()` 入口预留 CP3 消费槽位
- **不做** CP3 的实际写入和跳过逻辑（推迟到 Step 6）
- CP3 的 check 仅用于验证基础设施可用，预期永远 MISS

### 议题 2：CP3 写入语义与当前 `CachePayload` 契约是否冲突？

**问题**：
当前 `src/openpi/cache/storage_types.py` 要求 `CheckpointID.CP3` 的 payload 必须包含 `next_action_chunk`。  
但架构文档 Step 6 又明确写到：CP3 的 `next_action_chunk` 要等到下一个 cycle 的 action 产出后才能补全，因此需要 `DeferredWriter`。  
如果 Step 4 计划写的是“开启 CP1 和 CP3”且 `write_async()` 先退化成同步写入，那么 CP3 写入时究竟从哪里拿到 `next_action_chunk`？

**原因**：
- 当前数据契约已经把 CP3 定义成“当前 action -> 下一 action”的映射。
- 在没有延迟写入器前，CP3 条目看起来无法构造为合法 `CachePayload`。

**依据**：
- `src/openpi/cache/storage_types.py`
- `docs/cache_system_architecture_chinese.md`

**答辩**：

质疑成立。`CachePayload.validate_for_checkpoint(CP3)` 明确要求 `next_action_chunk is not None`，而在没有 DeferredWriter 的情况下，cycle N 写入 CP3 时不可能拿到 cycle N+1 的 action。

这与议题 1 结论一致：Step 4 无法完成合法的 CP3 写入。

**Plan 修正**：Step 4 不做 CP3 写入。CP3 在 Step 4 的范围限于：
- Orchestrator 中预留 CP3 check/write 的流程骨架
- Interceptor 预留 CP3 消费槽位（`infer()` 入口处）
- CP3 的 `CachePayload` 契约维持不变，不放宽验证
- 实际 CP3 写入 + DeferredWriter 推迟到 Step 6

### 议题 3：KeyBuilder 计划中的字段，当前 Stage API 真的拿得到吗？

**问题**：
`src/openpi/cache/types.py` 里定义了 `vision_0` / `vision_1` / `vision_2` / `prompt_emb` / `robot_state` 五个 canonical field。  
但当前 `src/openpi/models_pytorch/pi0_pytorch.py` 的 `Stage1Output` 只公开了 `state`、`prefix_embs`、mask、position ids，并没有单独暴露 per-camera vision embedding，也没有单独暴露 prompt embedding。  
如果 Step 4 的 KeyBuilder 要“收集所有可用数据，通过 config 控制开关决定用哪些 field”，这些 field 的真实来源分别是什么？

**原因**：
- 当前公开接口里，可稳定使用的字段远少于计划里讨论的字段集合。
- 若 field 来源不清楚，KeyBuilder 的灵活设计会建立在不存在的数据上。

**依据**：
- `src/openpi/cache/types.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`
- `logs/step4_discussion.log.md`

**答辩**：

质疑成立。Stage API 暴露的字段与 `types.py` 的 5 个 canonical field 之间存在明确的 gap。

当前 Stage 输出的实际可用字段：
- `Stage1Output.state` → 映射到 `ROBOT_STATE`，`[B, 32]`，**可直接使用**
- `Stage1Output.prefix_embs` → `[B, prefix_len, emb_dim]`，这是 vision + language token 的混合序列，**无法直接拆分**为 `vision_0/1/2` 和 `prompt_emb`
- `Stage3Output.action_chunk` → `[B, 50, 32]`，可用于 CP3 key 构建

无法从 Stage API 获取的字段：
- `VISION_0/1/2`：per-camera vision embedding 是 `multi_modal_projector` 内部输出，被合并进 `prefix_embs` 后已不可区分
- `PROMPT_EMB`：language token embedding 同样混合在 `prefix_embs` 中

**Plan 修正**：
- `PlaceholderKeyBuilder` 只用 `ROBOT_STATE`（`state` 字段），这是当前确定可用的
- `types.py` 中的 5 个 canonical field 保留为"未来可能的字段集合"，但在 Step 4 代码注释中明确标注哪些当前可用、哪些需要扩展 Stage API 或加 hook 才能获取
- KeyBuilder 的"灵活字段开关"设计收敛为：**当前只有 `robot_state` 开关可用**，`prefix_embs` 可作为实验性第二选项（需要对 `[B, prefix_len, emb_dim]` 做 mean pool 降维），vision/prompt 单独字段推迟到 Stage API 扩展后

### 议题 4：KeyBuilder 的“灵活字段开关”是否会被 Step 3 现有校验直接拦住？

**问题**：
目前 `src/openpi/cache/cache_storage.py` 的 `_check_entry_dims()` 要求 backend 声明的每个字段都必须出现在 `entry.query_keys` 中。  
而 `src/openpi/cache/README.md` 已明确把这条规则列为已知问题：它过于严格，未来应改成只校验交集。  
如果 Step 4 现在就采用“收集所有数据，按 config 选择部分字段写入”的设计，如何避免在 `CacheStorage.insert()` 阶段直接报错？

**原因**：
- Step 4 的字段灵活性依赖 Step 3 的校验语义足够宽松。
- 当前实现和当前计划在这里看起来是直接冲突的。

**依据**：
- `src/openpi/cache/cache_storage.py`
- `src/openpi/cache/README.md`

**答辩**：

质疑成立。`_check_entry_dims` 的逻辑确实会阻塞"部分字段写入"的设计。

`cache_storage.py:156-168` 的 `_check_entry_dims` 遍历 `self._dims`（backend 声明的所有字段），要求 entry 必须包含全部字段。如果 backend 声明了 `{robot_state: 32, vision_0: 2048}` 但 KeyBuilder 只产出 `{robot_state: [32]}`，insert 会直接 raise ValueError。

`README.md` 第 51 行已将此标记为待修复项：`_check_entry_dims 改为只校验 query_keys 与 backend 声明的交集，而非要求全集`。

**Plan 修正**：两种解决方案，选择第一种：

**(A) Step 4 中修复 `_check_entry_dims`（推荐）**：将校验逻辑改为交集模式——只检查 entry 中实际包含的字段是否与 backend 声明的维度匹配，不要求全部字段都出现。这是 README 已标记的 TODO，改动小且明确。

(B) Step 4 的 backend 只声明 `{robot_state: 32}`：回避问题，但限制了未来实验灵活性。

选择 A：在 Step 4 实现中修复 `_check_entry_dims` 为交集校验，同时在 `_check_query_dims`（search 侧）保持交集逻辑不变（它已经是交集模式）。

### 议题 5：暂存 GPU tensor 引用、避免 `.clone()` 的假设是否过于乐观？

**问题**：
当前讨论里提出：KeyBuilder `collect()` 暂存 GPU tensor 引用，推理结束前直接复用，尽量避免 `.clone()`。  
但 `src/openpi/cache/interceptor.py` 当前已显式调用 `torch.compiler.cudagraph_mark_step_begin()`，说明 staged compiled path 对 output 生命周期是敏感的。  
再加上 Step 3 存储层明确要求所有落库 tensor 都必须是 CPU contiguous float32。  
在这种前提下，哪些对象允许“只持有引用”，哪些对象必须 materialize？边界在哪里？

**原因**：
- “单次推理临时态”和“可存储态”目前没有被清晰区分。
- 如果边界不清晰，后续很容易把临时设备侧对象错误地跨 cycle、跨线程或跨层传递。

**依据**：
- `src/openpi/cache/interceptor.py`
- `src/openpi/cache/storage_types.py`
- `logs/step4_discussion.log.md`

**答辩**：

质疑部分成立，但结论是"当前 plan 的做法是安全的"。

关于 CUDAGraph 风险：`interceptor.py:144-146` 已将 compile mode 从 `max-autotune` 强制降级为 `max-autotune-no-cudagraphs`。这意味着 staged path **不使用 CUDAGraph**，output tensor 是普通 GPU tensor，不会被 CUDAGraph 复用 buffer 覆盖。`cudagraph_mark_step_begin()` 调用是防御性保留，不代表 CUDAGraph 真的启用。

**明确的引用生命周期边界**：

| 数据状态 | 位置 | 生命周期 | 操作约束 |
|---------|------|---------|---------|
| **临时引用态** | KeyBuilder._cache | 单次 `infer()` 调用内 | GPU tensor 引用，不 clone。安全前提：同一 `infer()` 内 stage 输出不会被覆盖（staged path 无 CUDAGraph） |
| **查询态** | KeyBuilder.build() 返回值 | check() 调用内 | CPU float32 L2-normalized。已 materialize（`.cpu().float()`），安全 |
| **存储态** | CacheEntry.query_keys / CachePayload | 持久 | CPU contiguous float32。由 `build()` 或调用方负责 materialize |

**Plan 修正**：在 KeyBuilder 代码注释中明确标注这三种状态的边界和安全前提：
```
# SAFETY: References are valid within a single infer() call.
# The staged path uses max-autotune-no-cudagraphs, so stage outputs
# are regular GPU tensors — not CUDAGraph-managed buffers.
# Crossing infer() boundary or writing to storage MUST materialize
# via .detach().cpu().contiguous().float().
```

### 议题 6：Step 4 的 timing 设计为何与 Step 2 的方向不一致？

**问题**：
当前讨论主张：Step 4 的缓存子步骤全部使用 CPU backend 计时，包括 `KeyBuilder.build()` 这种带 GPU normalize / D2H 的路径。  
但 `logs/step2.log` 里对未来缓存组件的计时设计写得很清楚：GPU 计算和 GPU<->CPU transfer 应支持 CUDA Event，CPU 逻辑才用 `perf_counter_ns`。  
如果 Step 4 一律使用 CPU probe，最后得到的究竟是壁钟时间，还是组件真实执行时间？这和 Step 2 的“精确计时”目标是否矛盾？

**原因**：
- 当前说法把“总体感知延迟”和“组件精确计时”混在了一起。
- 计时语义如果不先说清楚，后面的性能分析会很难对齐。

**依据**：
- `logs/step2.log`
- `src/openpi/cache/timing.py`
- `logs/step4_discussion.log.md`

**答辩**：

质疑部分成立。Step 2 log 确实将 KeyBuilder 列为 CUDA Event 计时目标，但这是基于一个 Step 4 尚未实现的前提。

Step 2 log 第 36-40 行的设计预期：

| 组件 | 设备 | 计时方案 |
|------|------|---------|
| Gate / Judge / FAISS CPU 搜索 | CPU | `perf_counter_ns` |
| KeyBuilder / VectorStore GPU 搜索 | GPU (`cache_stream`) | CUDA Event |
| GPU↔CPU transfer | CUDA `transfer_stream` | CUDA Event |

Step 2 log 第 214 行进一步说明：`CacheOrchestrator 的 cache_stream 通过 register_probe(..., stream=cache_stream) 接入`。

**关键点**：`cache_stream` 是 Step 8（系统效率优化）的产物。Step 4 没有独立的 cache CUDA stream，所有 GPU 操作（`F.normalize`）在默认 stream 上执行，与推理共享 stream。

在没有 cache_stream 的前提下：
- `F.normalize` 在默认 stream 上执行，kernel launch ~1μs
- `.cpu()` 触发隐式同步（等待默认 stream 上所有 kernel 完成），实际测量的是 D2H transfer 的壁钟时间
- 用 CUDA Event 在默认 stream 上记录这些操作，**得到的结果与 perf_counter 基本一致**（因为 `.cpu()` 已经强制同步了）
- 额外的 CUDA Event record/synchronize 开销反而是浪费

**结论**：当前 plan 用 CPU backend 是**务实选择**，测量的是壁钟时间（即推理管线实际被阻塞的时间），这对 Step 5 的延迟分析更有实际意义。Step 8 引入 `cache_stream` 后，再按 Step 2 的设计升级为 CUDA Event。

**Plan 修正**：在 timing 注释中加一句：
```
# NOTE: Step 2 design envisions CUDA Event timing for KeyBuilder once
# cache_stream is introduced (Step 8). Current CPU backend measures
# wall-clock time, which equals GPU time when operations run on the
# default stream with implicit sync (.cpu() calls).
```

### 议题 7：`fetch_payload()` 应该由 Judge 调，还是由 Orchestrator 调？

**问题**：
架构文档中的 `ThresholdJudge` 直接持有 `storage` 并在命中时调用 `fetch_payload()`。  
但当前讨论里又把 probe 拆成 `search -> judge -> fetch` 三段，更像是 Orchestrator 先让 Judge 返回“命中哪个 candidate”，再统一 fetch payload。  
这两种职责划分并不一致。Step 4 最终到底采用哪一种？

**原因**：
- 如果职责边界不统一，后续 probe 归属、组件依赖关系和测试方式都会摇摆。
- 尤其 Judge 是纯判定组件还是带存储副作用组件，这会直接影响接口设计。

**依据**：
- `docs/cache_system_architecture_chinese.md`
- `logs/step4_discussion.log.md`

**答辩**：

质疑有一定道理，但驳回"由 Judge 调 fetch_payload"的方案。

架构文档的伪代码是概念级设计，将 Judge 与 storage 绑定是为了简化说明。在实际实现中，**Orchestrator 调 fetch_payload** 更优，理由：

1. **Judge 保持纯判定**：Judge 的输入是 `(results, checkpoint_id, cached_data)` → 输出是 `(HitType, winner_id)`。不持有 storage 引用意味着 Judge 完全可测试（传入 mock results 即可），且替换 Judge 时不需要注入 storage 依赖。

2. **计时分离**：`search → judge → fetch` 三段各自独立计时。如果 Judge 内部调 fetch，则 judge probe 的测量值混入了 I/O 时间，导致 Judge 自身延迟和 fetch I/O 延迟无法区分。

3. **两阶段搜索设计一致**：Step 3 的 `CacheStorage` 已设计了 search() → fetch_payload() 两阶段模式（轻量搜索 + 选择性获取），Orchestrator 控制这个流程更自然。

4. **耦合最小化**：Judge 不依赖 Step 3 的任何接口，只依赖 `SearchResultLite`（一个纯数据类）。这使得 Judge 成为 Step 4 中唯一不与 Step 3 直接耦合的判定组件。

**结论**：维持 plan 设计 — Orchestrator 调 fetch_payload，Judge 是纯判定组件。架构文档的伪代码视为概念参考而非接口规范。

### 议题 8：Step 4 的实际范围是否应收敛？

**问题**：
当前 Step 4 同时承诺“CP1 + CP3 可运行端到端”，但从前面几项来看：
- CP3 缺少真正的消费接入点；
- CP3 写入需要 `next_action_chunk`，与当前契约冲突；
- KeyBuilder 灵活字段设计受 Step 3 校验阻塞；
- 计时方案与 Step 2 的目标表述不一致。

在这种情况下，Step 4 的边界是否需要重新定义？

**原因**：
- 当前目标范围可能同时压了多个尚未对齐的前提。
- 如果范围不收敛，后面的实现容易变成“骨架看起来齐全，但没有一条路径真正闭环”。

**依据**：
- `src/openpi/cache/README.md`
- `docs/cache_system_architecture_chinese.md`
- `logs/step4_discussion.log.md`

**答辩**：

质疑成立。综合议题 1-7 的结论，Step 4 的范围确实需要收敛。

**修正后的 Step 4 范围**：

| 功能 | 原 plan | 修正后 | 理由 |
|------|---------|--------|------|
| CP1 check + write | ✅ 端到端 | ✅ **端到端闭环** | 无阻塞，可完整实现 |
| CP1 hit early return | ✅ | ✅ | 跳过 stage2+3，返回 cached action |
| CP3 check | ✅ | ⚠️ **骨架 only** | 无 CP3 entries → 永远 MISS，仅验证基础设施 |
| CP3 write | ✅ | ❌ **推迟** | `next_action_chunk` 契约冲突（议题 2） |
| CP3 消费（skip next cycle） | 隐含 | ❌ **推迟** | 缺消费接入点（议题 1），需 Step 6 DeferredWriter |
| KeyBuilder 灵活字段 | "收集所有" | 收敛为 **`robot_state` only** | Stage API gap（议题 3） |
| `_check_entry_dims` 修复 | 未计划 | ✅ **新增** | 交集校验，解除字段灵活性阻塞（议题 4） |
| Timing | 全 CPU | 全 CPU + **注释说明** | 务实选择，Step 8 升级（议题 6） |
| Interceptor CP3 消费槽位 | 无 | ✅ **新增预留** | `infer()` 入口处 `should_skip_inference()` 桩 |

**Step 4 的真正闭环路径**：CP1 端到端 — 第 1 次推理 miss → 写入 cache → 第 2 次相同输入 → CP1 hit → 跳过 stage2+3 → 返回 cached action。

CP3 完整实现推迟到 Step 6（DeferredWriter + schedule_next_action + should_skip_inference）。

---
