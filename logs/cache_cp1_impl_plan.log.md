# Cache CP1 In-Memory 实现计划

> Status: Plan
> Date: 2026-04-06

---

## 目标

基于 [cache_system_tutorial.md](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/docs/cache_system_tutorial.md) 和 [cache_experiment_plan.log.md](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/logs/cache_experiment_plan.log.md)，为当前大规模实验补齐 `CP1` 路径所需实现。

本轮只服务于实验，不追求把整套系统一次性做成通用生产方案。

---

## 固定范围

- 只做 `CP1`
- `gate.type = always_search`
- `judge.type = always_hit`
- 数据库后端统一使用 `in_memory`
- 模块实现保留 `vision_2` 支持；LIBERO 本轮实验只是在 YAML 中将其关闭
- `prompt_emb` 在 Phase 1 / 1.5 中权重固定为 `0`
- Layer 1 固定：
  - `vision_0 / vision_1 / prompt_emb = cosine`
  - `robot_state = L2`
- Layer 2 固定比较：
  - `weighted_rrf`
  - `weighted_score_sum`

---

## 设计原则

严格遵守教程里的模块隔离原则：

- `Gate` 只决定要不要搜，不做 IO
- `Judge` 只决定命中类型，不做 IO
- 只有 `SearchStrategy` 构造 `QuerySpec` 并调用 `storage.search()`
- 只有 `Orchestrator` 调 `storage.insert()` / `storage.fetch_payload()`
- `Config` 只负责 YAML 解析、校验、工厂，不承载业务逻辑
- `Backend` 只负责存储、过滤、字段打分、融合和排序，不感知模型实现细节

额外要求：

- 降维逻辑不能耦合到 backend
- cosine / L2 逻辑不能耦合到 judge
- 跨模态融合逻辑不能写进 orchestrator
- `in_memory` 要从“测试占位后端”升级为“实验主后端”

---

## 当前实现问题

### 1. `in_memory` 后端能力严重不足

当前 [in_memory_backend.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/backends/in_memory_backend.py) 只有：

- 单字段 brute-force cosine
- 命中第一个字段就 `break`
- 不支持 `robot_state = L2`
- 不支持多字段融合
- 不支持 `weighted_rrf`
- 不支持 `weighted_score_sum`
- 不支持 score normalization
- 不支持 `task_key` / `step_range` 过滤

它目前只够做单元测试，不够支撑实验。

### 2. `QuerySpec` 表达能力不够

当前 [storage_types.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/storage_types.py) 只有：

- `fusion_weights`
- `backend_hints`

缺少实验真正需要的显式语义：

- 融合方法类型
- 字段级相似度定义
- score normalization 定义
- `tau`
- percentile 统计参数

### 3. `KeyBuilder` 还没有实验版降维器

当前 [key_builder.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/components/key_builder.py) 只有：

- `placeholder`
- `full_original`

还没有：

- `A: mean pool`
- `B1: spatial pooling 4x4`
- `B2: spatial pooling 2x2`
- `C: max pool`

### 4. YAML schema 还不够表达实验

当前 [config.py](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/src/openpi/cache/config.py) 的 schema 只够表达占位式 cache，不够表达实验配置。

---

## 模块边界

### 1. `KeyBuilder / Reducer`

职责：

- 从 `stage1` 输出中抽取 `vision_0`、`vision_1`、`prompt_emb`、`robot_state`
- 对字段分别做降维
- 输出 CPU float32 query keys

不负责：

- 相似度计算
- 分数归一化
- 融合
- 数据库检索

说明：

- `robot_state` 原样输出
- `vision_0 / vision_1 / vision_2` 在代码层都支持；本轮 LIBERO YAML 中关闭 `vision_2`
- `vision_0 / vision_1` 根据实验选择 A/B1/B2/C
- `prompt_emb` 目前按 plan 参与 key 构造，但在粗搜阶段权重固定为 `0`
- 按模态切分 `prefix_embs` 的逻辑直接复用当前 `FullOriginalKeyBuilder` 的实现思路
- 新的 CP1 builder 只改“各模态切出来之后如何降维”，不改“token sequence 如何按模态切片”
- 现有 key builder 约定还预留了 `CP3` 路径：`collect()` 可缓存 `stage3.action_chunk`，`build()` 可接受 `CheckpointID.CP3`
- 因此新的 builder 也必须保持这个接口兼容性，即使本轮实验只跑 `CP1`
- 具体要求是：
  - `collect()` 继续支持缓存 `stage3.action_chunk`
  - `build()` 继续接受 `CheckpointID.CP3`
  - 当前阶段 `CP3` 可以先与 `CP1` 产出相同 key，action concat 仍保持 deferred，不在这轮实验实现

### 2. `SearchStrategy`

职责：

- 把本次实验配置写进 `QuerySpec`
- 组装：
  - `top_k`
  - `filters`
  - `fusion_method`
  - `fusion_weights`
  - `field_similarity`
  - `score_normalization`

不负责：

- 真的计算 cosine / L2
- 做融合
- 决定 hit / miss

建议：

- 保留“只有 SearchStrategy 负责构造 QuerySpec 并调用 storage.search()”这个出口边界
- 现有 `QdrantWeightedRrfKnnStrategy` 只保留给 Qdrant backend 使用
- 本轮实验再新增独立的 in-memory strategy type

### 3. `InMemoryBackend`

职责：

- 保存 entries
- 从预构建 artifact 加载 entries
- 按 `QuerySpec` 执行过滤
- 为每个字段计算字段级分数
- 执行跨模态融合
- 排序并返回 `SearchResultLite`

不负责：

- 降维
- 从原始数据现算 key
- 命中判定
- 阈值逻辑

建议在 `in_memory_backend.py` 内部用私有函数分层，而不新增文件。

这样 `InMemoryBackend` 本身只负责调度：

1. 过滤候选 entry
2. 计算每个字段的原始分数
3. 做必要的 score normalization
4. 调 fusion 得到最终分数
5. 排序返回 top-k

### 4. `Judge`

本轮不改业务逻辑。

固定使用：

- `AlwaysHitJudge`

职责保持最小化：

- 如果 `results` 非空，放行 top-1

Judge 不参与：

- 阈值实验
- re-score
- 多字段逻辑

### 5. `Orchestrator`

本轮尽量不改职责。

职责仍然是：

- collect
- gate
- build
- search
- judge
- fetch

它不应该知道：

- 当前 fusion 是 RRF 还是 Score Sum
- 当前字段怎么算 cosine / L2
- 当前 normalization 参数是什么

### 6. `Config / YAML`

职责：

- 描述实验，不执行实验
- 校验字段和参数组合是否合法
- 实例化组件

不负责：

- 数值逻辑
- 降维计算
- 融合计算

### 7. `Experiment Runner / Server Control`

职责：

- 控制 100+ 个实验按顺序运行
- 选择要跑哪些 run
- 选择每个 run 的 episode 个数
- 记录实验进度并支持断点重续
- 在每轮实验开始前通知 server 切换到新的 YAML

不负责：

- 重写 LIBERO 客户端主逻辑
- 在运行时重新生成 YAML
- 在 server 内实现实验编排逻辑

原则：

- 实验编排放在本地控制脚本，不塞进 backend
- server 只负责响应“加载哪个 YAML”
- `examples/libero/main.py` 尽量复用，只做最小包裹或最小改造

---

## 详细代码计划

这一节收敛到教程里已经定义的扩展点，只使用现有骨架：

- `components/key_builder.py`
- `components/search_strategy.py`
- `backends/in_memory_backend.py`
- `storage_types.py`
- `config.py`
- `tests/cache/...`
- `cache.yaml` 或实验专用 YAML
- `scripts/serve_policy.py`
- `src/openpi/serving/websocket_policy_server.py`
- `examples/libero/main.py` 或其外层包裹脚本
- `exp/generate_cache_run_yamls.py`
- `exp/build_in_memory_cache_artifact.py`
- `exp/calibrate_score_sum_stats.py`
- `exp/analyze_cache_results.py`
- `exp/run_cache_experiments.py`

不再额外引入教程里没有提到的一批新模块。

### A. `components/key_builder.py`

这是本轮最主要的扩展点之一。

实现方式：

- 在现有文件里新增 4 个独立 key builder 类
- 不做一个包含全部实验逻辑的大类
- 每个类只对应一种降维方式

计划新增类：

- `CP1MeanPoolKeyBuilder`
- `CP1SpatialPool16KeyBuilder`
- `CP1SpatialPool64KeyBuilder`
- `CP1MaxPoolKeyBuilder`

实现口径：

- 对外仍保留 4 个独立 builder type，保证 YAML 和实验组一一对应
- 对内不复制 4 份实现
- 采用“4 个很薄的 wrapper 类 + 一套共享私有 helper”的方式复用逻辑
- 不引入单独的 `reducers.py`

每个类都实现现有 `QueryKeyBuilder` 协议：

- `collect()`
- `build()`
- `cached_data`
- `clear()`

四个类的职责分别是：

1. `CP1MeanPoolKeyBuilder`
   - `vision_0 / vision_1 / prompt_emb` 做 mean pool
   - `robot_state` 原样输出
   - `collect()` 保留对 `stage3.action_chunk` 的缓存
   - `build(CheckpointID.CP3)` 保持可调用，当前可先与 `CP1` 同键
   - 输出 shape：
     - `vision_*: [emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - `mean pool` 后结果已经是一维向量，这里不需要额外 `flatten`
   - 对应实验组：
     - `A-RRF`
     - `A-SUM`

2. `CP1SpatialPool16KeyBuilder`
   - `vision_0 / vision_1` 从 16x16 token 网格池化到 4x4
   - `prompt_emb` 做 mean pool
   - `robot_state` 原样输出
   - `collect()` 保留对 `stage3.action_chunk` 的缓存
   - `build(CheckpointID.CP3)` 保持可调用，当前可先与 `CP1` 同键
   - 输出 shape：
     - `vision_*: [4, 4, emb_dim] -> flatten -> [16 * emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - 这里必须明确：
     - `vision_*` 在 spatial pooling 之后必须 `flatten`
     - 否则不符合当前 `QuerySpec.query_keys[field] = [dim]` 的一维向量契约
   - 对应实验组：
     - `B1-RRF`
     - `B1-SUM`

3. `CP1SpatialPool64KeyBuilder`
   - `vision_0 / vision_1` 从 16x16 token 网格池化到 2x2
   - `prompt_emb` 做 mean pool
   - `robot_state` 原样输出
   - `collect()` 保留对 `stage3.action_chunk` 的缓存
   - `build(CheckpointID.CP3)` 保持可调用，当前可先与 `CP1` 同键
   - 输出 shape：
     - `vision_*: [2, 2, emb_dim] -> flatten -> [4 * emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - 这里必须明确：
     - `vision_*` 在 spatial pooling 之后必须 `flatten`
     - 否则不符合当前 `QuerySpec.query_keys[field] = [dim]` 的一维向量契约
   - 对应实验组：
     - `B2-RRF`
     - `B2-SUM`

4. `CP1MaxPoolKeyBuilder`
   - `vision_0 / vision_1 / prompt_emb` 做 max pool
   - `robot_state` 原样输出
   - `collect()` 保留对 `stage3.action_chunk` 的缓存
   - `build(CheckpointID.CP3)` 保持可调用，当前可先与 `CP1` 同键
   - 输出 shape：
     - `vision_*: [emb_dim]`
     - `prompt_emb: [emb_dim]`
     - `robot_state: [state_dim]`
   - `max pool` 后结果已经是一维向量，这里不需要额外 `flatten`
   - 对应实验组：
     - `C-RRF`
     - `C-SUM`

共用实现方式：

- 直接在 `key_builder.py` 内部复用已有 token offset 常量
- 直接复用 `FullOriginalKeyBuilder` 当前的模态切分方式：
  - `_VISION_OFFSETS`
  - `_PROMPT_START`
  - `prefix_embs[0]` 上的切片逻辑
- 允许写少量私有 helper 函数，例如：
  - `_slice_cp1_fields()`
  - `_mean_pool_tokens()`
  - `_max_pool_tokens()`
  - `_spatial_pool_tokens()`

但这些 helper 只放在 `key_builder.py` 文件内部，不再拆新模块。

关键约束：

- 不允许在四个新 builder 里各自复制一份硬编码切片逻辑
- 模态切分逻辑应在 `key_builder.py` 内部尽量收敛成一处共享私有实现
- 如果以后 token layout 变化，只改这一处
- 最终写入 `QuerySpec.query_keys` / `CacheEntry.query_keys` 的每个字段都必须是一维向量 `[dim]`
- 因此只有 `B1/B2` 的 vision 分支需要显式 `flatten`
- `A/C` 的 pooled 结果以及 `robot_state` 本身已经是一维，不再额外 `flatten`

明确不做：

- 动态 reducer registry
- 通用 reducer 框架

### B. `components/search_strategy.py`

这是本轮第二个主要扩展点。

实现方式：

- 保留“只有 SearchStrategy 构造 `QuerySpec` 并调用 `storage.search()`”这个边界
- 在现有文件里新增两个实验专用 strategy

计划新增类：

- `WeightedRrfKnnStrategy`
- `WeightedScoreSumKnnStrategy`

职责：

1. `WeightedRrfKnnStrategy`
   - 从配置中拿到：
     - `top_k`
     - `step_filter`
     - `fusion_weights`
     - `rrf_k`
     - `field_similarity`
   - 构造成 `QuerySpec`
   - 调 `self._storage.search(spec)`

2. `WeightedScoreSumKnnStrategy`
   - 从配置中拿到：
     - `top_k`
     - `step_filter`
     - `fusion_weights`
     - `field_similarity`
     - `score_normalization`
   - 构造成 `QuerySpec`
   - 调 `self._storage.search(spec)`

明确不做：

- cosine 计算
- L2 计算
- score normalization
- fusion

如果一定要在同文件里扩展现有 strategy 实现，也必须满足：

- `QdrantWeightedRrfKnnStrategy` 仍明确保持 Qdrant-only 语义
- `weighted_rrf` 和 `weighted_score_sum` 两条 in-memory 实验路径分开
- 不能把多种 search strategy 混成不可读的一堆 `if`

### C. `storage_types.py`

这个文件需要扩类型，但仍保持“backend-agnostic”。

需要补的内容：

- 在 `QuerySpec` 里增加实验检索所需字段

建议直接增加：

- `fusion_method: str | None`
- `field_similarity: dict[str, dict[str, Any]] | None`
- `score_normalization: dict[str, dict[str, Any]] | None`
- `backend_hints: dict[str, Any] | None`

用途：

- `fusion_method`（SearchStrategy 写入 QuerySpec 时去掉 `_knn` 后缀）
  - `weighted_rrf`
  - `weighted_score_sum`
- `field_similarity`
  - `vision_0 / vision_1 / prompt_emb = cosine`
  - `robot_state = l2`
- `score_normalization`
  - cosine 的 `[0,1]` 映射
  - `robot_state` 的 `exp(-d/tau)`
  - percentile normalization 参数
- `backend_hints`
  - backend-specific 参数
  - 当前主要是 `rrf_k`

明确边界：

- 通用搜索语义放一级字段
- backend 私有参数继续放 `backend_hints`
- 不把 `QuerySpec` 退化成一个大字典

这里先不追求类型系统特别漂亮，先保证：

- 配置可表达
- strategy 能传递
- backend 能读取

### D. `backends/in_memory_backend.py`

这是本轮最重要的重写点。

实现方式：

- 不新增新的 backend 文件
- 直接在现有 `in_memory_backend.py` 里把搜索逻辑重写成实验可用版本

需要实现的功能：

1. 多字段打分
   - `vision_0 / vision_1 / prompt_emb` 用 cosine
   - `robot_state` 用 L2

2. 两种融合方式
   - `weighted_rrf`
   - `weighted_score_sum`

3. `weighted_score_sum` 所需归一化
   - cosine 先 `(cos + 1) / 2`
   - `robot_state` 先 `exp(-d / tau)`
   - 再按 `p5 / p95` 做 percentile normalization

4. 过滤
   - `checkpoint_id`
   - `task_key`
   - `step_range`

建议在 `in_memory_backend.py` 内部补私有方法：

- `_filter_entries()`
- `_cosine_score()`
- `_l2_distance()`
- `_normalize_score()`
- `_search_weighted_rrf()`
- `_search_weighted_score_sum()`
- `_iter_active_fields()`

关键要求：

- backend 只吃已经 build 好的 query vectors
- backend 只加载已经 build 好的 entry vectors
- backend 不碰 stage1 / token layout
- backend 不参与 hit/miss 判断
- `weight == 0` 的字段直接跳过，不做 cosine / L2 / rank / sum 计算

### E. `config.py`

这个文件只做 schema、校验、工厂，不做算法。

需要扩的内容：

1. `KeyBuilderConfig`
   - 支持：
     - `cp1_mean_pool`
     - `cp1_spatial_pool_16`
     - `cp1_spatial_pool_64`
     - `cp1_max_pool`

2. `SearchStrategyConfig`
   - 支持实验所需字段：
     - `type`
     - `field_similarity`
     - `score_normalization`
     - `rrf_k`

3. `BackendConfig`
   - `in_memory` 仍保留为 backend type
   - 不额外引入复杂 backend registry

4. 校验逻辑
   - 模块层允许 `vision_2`
   - LIBERO 本轮实验 YAML 中 `vision_2` 应为 disabled
   - `prompt_emb` 在 Phase 1 / 1.5 权重为 `0`
   - `weighted_score_sum` 下必须提供：
     - `tau`
     - `p5/p95`
   - `backend.in_memory.preload_path` 必须存在，且和当前 run 的 `key_builder.type` / `vector_dims` 一致

5. 工厂逻辑
   - `_build_key_builder()` 增加四个 builder 分支
   - `_build_search_strategy()` 增加两个实验 strategy 分支

### F. YAML 计划

仍然使用教程要求的 YAML 入口，不额外搞新的配置系统。

本轮不采用“一个 YAML 覆盖多个实验”。

要求改为：

- 一个 run 对应一个 YAML
- 一个 YAML 就是一次最终可运行配置
- 不再依赖外部表格或脚本去二次展开权重

但 100 多个 YAML 不应手写，应该脚本化生成。

约束是：

- 生成脚本只负责批量产出最终 YAML 文件
- 生成后的每个 YAML 仍然是独立、完整、可直接运行的一次实验配置
- 运行实验时直接消费这些 YAML，不再在运行阶段做二次参数展开

按当前实验计划：

- Phase 1 先生成 `64 个独立 YAML`（8 combos × 8 weights）
- Phase 1.5 / Phase 2 的 YAML 在分析结果后二次生成（~45 + ~3 个）
- 总计 ~112 个 YAML，分批产出

建议目录：

- `configs/cache_runs/phase1/`
- `configs/cache_runs/phase1_5/`
- `configs/cache_runs/phase2/`
- `exp/generate_cache_run_yamls.py`

建议命名：

- `phase1_run_001_a_rrf_w1.yaml`
- `phase1_run_002_a_rrf_w2.yaml`
- ...
- `phase1_run_064_c_sum_w8.yaml`
- `phase1_5_run_001_*.yaml`
- ...
- `phase2_run_003_*.yaml`

每个 YAML 都应完整写出：

- `key_builder.type`
- `checkpoints.cp1.search_strategy.type`
- 本次 run 的字段权重
- 本次 run 的 `backend.vector_dims`
- 本次 run 的 `score_normalization`
- 本次 run 的 `keys.vision_2.enabled`
- 本次 run 的 `backend.in_memory.preload_path`

LIBERO 本轮实验的 YAML 统一约束：

- `keys.vision_2.enabled = false`
- 模块虽然支持 `vision_2`，但本轮 run 不启用
- Phase 1 / 1.5 中 `prompt_emb.weight = 0.0`

YAML 生成脚本职责：

- 读取实验计划里的组合定义
- 按 phase 批量生成 `112` 个独立 YAML
- 为每个 run 写入：
  - `key_builder.type`
  - `search_strategy.type`
  - 字段权重
  - `score_normalization`
  - `backend.in_memory.preload_path`
- 文件命名稳定、可追踪，便于回看实验结果

YAML 生成脚本不负责：

- 启动实验
- 在运行时再展开参数
- 现场构建 artifact

### G. 离线预构建 `in_memory` 数据

本轮不让 `in_memory` backend 在实验启动时根据原始数据和 `key_builder` 现场建库。

原因：

- 112 个 run 会重复做同样的 key 提取和降维
- backend 不应该依赖 stage1 / token layout / dataset 解析细节
- 不同 `key_builder.type` 的输出维度不同，不能共享同一份预构建数据

因此职责拆分为：

- 离线脚本
  - 读取原始数据
  - 调用对应 `key_builder`
  - 生成可直接加载的 cache artifact
- `InMemoryBackend`
  - 只负责加载 artifact
  - 只负责搜索

artifact 至少要绑定这些信息：

- 数据集标识
- `checkpoint_id`
- `key_builder.type`
- enabled fields
- `vector_dims`
- entries 本体

对当前 `data/libero_spatial`，artifact 构建的主路径明确为：

- 直接读取 HDF5 里已有字段
  - `vision_0`
  - `vision_1`
  - `vision_2`
  - `prompt_emb`
  - `robot_state`
- 不需要额外离线 stage1 推理
- 不需要 policy checkpoint
- 不需要 GPU

只有以后换数据集、且原始数据里没有这些 embedding 字段时，才考虑增加“离线 stage1 提取”作为 fallback 路径；这不属于当前主方案。

建议增加一个离线脚本，例如：

- `exp/build_in_memory_cache_artifact.py`

这个脚本的职责是：

- 输入：
  - 原始数据路径
  - `key_builder.type`
  - 输出路径
- 输出：
  - 一份可被 `InMemoryBackend` 直接加载的 artifact

按当前实验，至少会有 4 份 artifact：

- `cp1_mean_pool`
- `cp1_spatial_pool_16`
- `cp1_spatial_pool_64`
- `cp1_max_pool`

同一个 builder 下的不同权重实验、不同 strategy 实验应复用同一份 artifact，不重复 build。

生成 YAML 时，应直接把对应 builder 的 artifact 路径写入每个 run：

- `cp1_mean_pool` -> 对应 mean pool artifact
- `cp1_spatial_pool_16` -> 对应 spatial-16 artifact
- `cp1_spatial_pool_64` -> 对应 spatial-64 artifact
- `cp1_max_pool` -> 对应 max-pool artifact

在 Phase 1 正式运行前，需要对每个 builder 额外做一轮离线 sanity check：

- 采样三类 pair：
  - same-episode near-step
  - same-task cross-episode
  - cross-task
- 统计各 builder 的 vision cosine 分布和分离度
- 尤其检查 `B1/B2` 的 spatial-pool + flatten + cosine 是否退化到近似噪声

处理原则：

- 默认不因为 sanity check 自动删除实验组
- 但如果某个 builder 的分离度接近随机、或分布明显病态，应在进入 64 runs 前人工确认是否停止该组实验

### H. 实验控制脚本

需要新增一个本地控制脚本，例如：

- `exp/run_cache_experiments.py`

目标：

- 顺序运行 100+ 个实验
- 不要求每换一个 YAML 就重启 server
- 尽量包裹现有 `examples/libero/main.py`，不重写客户端主逻辑

这个脚本负责：

- 读取待运行 YAML 列表
- 支持只跑其中一部分 run
- 支持为当前批次指定每个 run 的 episode 个数
- 每个 run 开始前先通知 server 切换到对应 YAML
- 然后调用 `examples/libero/main.py` 或一个轻量 wrapper 启动 LIBERO 评测
- 收集每个 run 的退出状态与结果摘要
- 将进度写入本地状态文件，支持断点重续
- 写出每个 run 的结构化结果摘要，供后续汇总脚本使用

建议支持的控制参数：

- `--runs`
  - 指定要跑哪些 run，例如按文件名、索引范围或 phase 过滤
- `--episodes-per-run`
  - 覆盖每个 run 的 `num_trials_per_task`
- `--num-workers`
  - 透传给 `examples/libero/main.py --num_workers`
  - 单个 run 内的并发 worker 数
- `--resume`
  - 从本地状态文件恢复，跳过已完成 run
- `--state-path`
  - 指定断点状态文件路径
- `--libero-args`
  - 透传给 `examples/libero/main.py` 的额外参数

结果收集要求：

- 每个 run 固定使用同一 `episodes-per-run` 预算
- `Phase 1` 的 top 3 以固定预算下的 `aggregate success rate` 排序
- 如需最终结论，最佳配置还应追加更大 budget 的 confirm run

建议增加结果汇总脚本，例如：

- `exp/analyze_cache_results.py`

职责：

- 汇总每个 run 的结构化结果文件
- 按 phase 输出排序表
- 选出 `Phase 1` top 3
- 为 `Phase 1.5` 生成后续输入

断点重续要求：

- 至少支持 run 级别断点重续
  - 已完成的 YAML 不重复跑
- 最好支持 episode 级别进度记录
  - 中断后可从当前 run 的下一个 episode 继续

建议状态文件记录：

- run id / yaml path
- 使用的 episode 个数
- 当前状态：pending / running / done / failed
- 已完成 episode 数
- 开始时间 / 结束时间
- 结果摘要（成功率、退出码、日志路径）

这里的“缓存”是实验控制层的进度缓存，不是检索缓存本身。

### I. Server 动态切换 YAML

当前 `scripts/serve_policy.py` 的 `--cache_config` 是 server 启动时一次性读取。

对 100+ 个实验，这种方式不方便，因为每换一个 YAML 都要重启 server。

因此 server 侧需要补一个轻量控制能力：

- 在 WebSocket 控制消息里新增”切换 cache 配置”的请求
- 每轮实验开始时由本地控制脚本通过专用短连接发给 server
- server 收到后校验 YAML、构建新的 `shared_storage`、原子替换全局 `CurrentCacheBundle`

总体约束：

- **支持单个 run 在固定 YAML 下使用多 worker 并发评测**
- **不支持多个不同 run 同时共享一个 server 并发执行**
- run 与 run 串行；单个 run 内允许 `num_workers > 1`
- 实验 server 统一用 `--concurrent` 启动（即使 `num_workers=1`），因为只有 concurrent 模式才有 `connection_policy_factory` 入口

目标边界：

- server 只负责”按 bundle 提供新连接”
- 实验控制顺序仍由本地脚本负责
- 不把实验计划、phase 概念写进 server

协议形态：

- 客户端发送控制消息：
  - `__ctrl__ = “load_cache_config”`
  - 附带 `yaml_path`
- server 执行：
  - 校验 YAML
  - `build_shared_storage(cache_config)` 构建新 storage
  - 原子替换全局 `CurrentCacheBundle`
- server 返回：
  - `__ack__ = “load_cache_config”`
  - 附带 `version`（bundle 版本号，单调递增）

生效时机：

- `load_cache_config` 构建新 bundle 并原子替换，不影响任何当前活跃连接
- 新 bundle 在下一条新建的评测连接生效
- 已在跑的旧连接继续用旧 bundle 的 `shared_storage`，不受影响
- 不需要 busy 保护——控制消息随时可发

单 run 执行流程：

1. 控制脚本发 `load_cache_config(yaml_path)`，等待 ack
2. 启动 `examples/libero/main.py --num_workers N`
3. N 个 worker 各自建连接，`connection_policy_factory` 读取当前 bundle 快照
4. 各 worker 共享同一个 `bundle.shared_storage`
5. 所有 worker 完成后，再切下一个 run

实现要求：

- 新逻辑加在 `websocket_policy_server.py`（`CurrentCacheBundle` + 控制消息处理）
- `scripts/serve_policy.py` 的 `connection_policy_factory` 每次新连接时从全局 bundle 读取最新配置
- 不改普通 `infer` 请求协议
- 不要求重写 `openpi-client` 的主推理逻辑

副作用说明：

- 控制短连接会产生一次空 task 生命周期（server 在连接建立时会触发 `on_task_begin()`），这是可接受的

为了保持职责清晰：

- server 不直接解析实验表格
- server 不负责选择下一个 run
- server 不负责实验断点重续

### J. LIBERO 入口复用方式

优先方案不是重写 `examples/libero/main.py`，而是尽量包裹它。

原因：

- `examples/libero/main.py` 已经有：
  - task suite 选择
  - `num_trials_per_task`
  - client lifecycle
  - 评测主循环
- 大实验主要缺的是“run 切换控制”，不是单个 LIBERO rollout 逻辑

推荐两层实现：

1. 最小改造 `examples/libero/main.py`
   - 暴露更容易复用的入口函数
   - 支持可选的 episode 起始偏移 / episode 数覆盖
2. 本地控制脚本包裹它
   - 在每个 run 前先通知 server 切 YAML
   - 再调用 LIBERO 主入口执行该 run

如果能不改客户端协议主流程，就不要重写客户端。

### K. 测试计划

测试仍按教程现有测试结构放在 `tests/cache/` 下，不额外发明新的测试体系。

建议新增测试文件：

- `tests/cache/components/test_key_builder_cp1_experiment.py`
- `tests/cache/test_search_strategy_experiment.py`
- `tests/cache/test_search_strategy_weighted_rrf.py`
- `tests/cache/test_search_strategy_weighted_score_sum.py`
- `tests/cache/test_in_memory_backend_experiment.py`
- `tests/cache/test_config_experiment.py`

测试内容：

1. key builder
   - 四种 builder 的输出 shape 正确
   - `vision_2` 代码路径可正常工作
   - LIBERO 实验 YAML 中 `vision_2` 关闭时行为正确
   - `prompt_emb` 输出存在但可配置权重为 `0`
   - `collect(stage3=...)` 时能缓存 `action_chunk`
   - `build(CheckpointID.CP3)` 不报错，且当前与 `CP1` 行为一致

2. search strategy
   - `QuerySpec` 正确携带：
     - `fusion_method`
     - `field_similarity`
     - `score_normalization`
   - 两个 strategy type 的装配路径分别正确

3. in_memory backend
   - cosine + L2 多字段打分正确
   - `weighted_rrf` 排序正确
   - `weighted_score_sum` 排序正确
   - `tau = 0.334717` 路径正确
   - 过滤逻辑正确
   - 能从 `preload_path` 正确加载 artifact
   - `key_builder.type` / `vector_dims` 不匹配时拒绝启动

4. config
   - YAML 能构建对应 builder / strategy / backend
   - 缺少 `tau` 或 `p5/p95` 时在 `weighted_score_sum` 下报错
   - 缺少 `backend.in_memory.preload_path` 时在实验配置下报错

5. experiment runner / server control
   - 控制脚本能按给定 run 列表顺序执行（run 间串行）
   - `--episodes-per-run` 能正确覆盖 LIBERO episode 个数
   - `--num-workers` 透传给 `main.py`，单 run 内支持多 worker 并发
   - `--resume` 能跳过已完成 run
   - server 统一用 `--concurrent` 启动
   - `load_cache_config` 构建新 `CurrentCacheBundle`（含 shared_storage），原子替换全局 bundle
   - `load_cache_config` 不影响普通 `infer` 协议
   - 同一 run 的多个 worker 共享同一个 `bundle.shared_storage`
   - 不同 run 切 YAML 时重建 shared_storage，旧连接继续用旧 bundle
   - 控制短连接会产生一次空 task 生命周期（on_task_begin/on_task_end），测试需验证这不会产生有害副作用

6. calibration / analysis
   - 能为每个 builder / field 统计真实 `p5/p95`
   - 正式 `SUM` YAML 禁止使用占位 `p5=0.0, p95=1.0`
   - 能输出 builder sanity check 结果
   - 能汇总每个 run 的 success rate 并选出 `Phase 1` top 3

---

## 按实验组拆分的实现方式

为了避免“一个 key builder 管全部功能”，本轮按实验组拆 builder，但仍放在教程已有的 `components/key_builder.py` 扩展点里：

注意：

- 四个 builder 都保留 `vision_2` 支持
- 是否启用 `vision_2` 由具体 YAML 决定
- 代码支持范围大于本轮 LIBERO 实验使用范围

- A 组：
  - `CP1MeanPoolKeyBuilder`
- B1 组：
  - `CP1SpatialPool16KeyBuilder`
- B2 组：
  - `CP1SpatialPool64KeyBuilder`
- C 组：
  - `CP1MaxPoolKeyBuilder`

而 Layer 2 不再按实验组拆成多个 backend，而是统一由同一个 `InMemoryBackend` 支持两种 fusion：

- `weighted_rrf`
- `weighted_score_sum`

这样边界最清楚：

- 实验组差异主要落在 `key_builder.type`
- 融合差异主要落在 `search_strategy.type`
- backend 统一负责执行

## 需要新增或修改的配置能力

### `key_builder`

需要支持：

- `type: cp1_mean_pool`
- `type: cp1_spatial_pool_16`
- `type: cp1_spatial_pool_64`
- `type: cp1_max_pool`

### `search_strategy`

需要支持：

- `type: weighted_rrf_knn`
- `type: weighted_score_sum_knn`
- `fusion_method`（SearchStrategy 写入 QuerySpec 时去掉 `_knn` 后缀）
  - `weighted_rrf`
  - `weighted_score_sum`
- `field_similarity`
  - `vision_0: cosine`
  - `vision_1: cosine`
  - `prompt_emb: cosine`
  - `robot_state: l2`
- `score_normalization`
  - cosine 映射到 `[0,1]`
  - `robot_state` 距离转相似度
  - percentile normalization
- `backend_hints`
  - `rrf_k`

### `backend`

需要支持：

- `type: in_memory`
- `in_memory` 专属配置块
- `in_memory.preload_path`
- 后续可扩 `index_type`，但第一版先 brute-force 即可

---

## YAML 目标形态

下面给出“单个 run”的目标 YAML 形态示例。`112 runs` 时，应产生 `112` 个这样可独立运行的 YAML。

```yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null

keys:
  vision_0:    { enabled: true,  weight: 0.75 }
  vision_1:    { enabled: true,  weight: 0.25 }
  vision_2:    { enabled: false, weight: 0.0 }
  prompt_emb:  { enabled: true,  weight: 0.0 }
  robot_state: { enabled: true,  weight: 0.25 }

key_builder:
  type: cp1_mean_pool

checkpoints:
  cp1:
    enabled: true
    gate:
      type: always_search
    judge:
      type: always_hit
    search_strategy:
      type: weighted_score_sum_knn
      top_k: 1
      step_filter: all
      field_similarity:
        vision_0:   { type: cosine }
        vision_1:   { type: cosine }
        prompt_emb: { type: cosine }
        robot_state:
          type: l2
          to_similarity:
            type: exp
            tau: 0.334717
      score_normalization:
        type: percentile
        fields:
          vision_0:   { p5: <calibrated>, p95: <calibrated> }
          vision_1:   { p5: <calibrated>, p95: <calibrated> }
          prompt_emb: { p5: <calibrated>, p95: <calibrated> }
          robot_state:{ p5: <calibrated>, p95: <calibrated> }

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    vision_2: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    preload_path: data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl
    index_type: brute_force
```

说明：

- Phase 1 / 1.5 时，`prompt_emb.weight = 0.0`
- 每个 run 都有自己的独立 YAML
- 模块层保留 `vision_2`
- LIBERO 本轮 YAML 中 `vision_2.enabled = false`
- Phase 2 如需验证 `prompt_emb`，只改权重，不改模块边界
- 同一 schema 应能表达 `weighted_rrf` 和 `weighted_score_sum`
- 同一 `key_builder.type` 下的多个 run 复用同一份 `preload_path`
- 正式 `SUM` YAML 必须写入真实 calibration 值，不能保留 `p5: 0.0, p95: 1.0` 占位

---

## Artifact 预构建与加载边界

这一轮实验把“建库”和“检索”彻底拆开：

- `KeyBuilder`
  - 定义 key 如何从模型输出中构造
- 离线 artifact 构建脚本
  - 用 `KeyBuilder` 批量把原始数据转换成 entry vectors
- `InMemoryBackend`
  - 只加载 artifact
  - 只执行搜索
- YAML
  - 只声明当前 run 使用哪份 artifact

这样做的约束是：

- backend 不再因为不同实验 run 重复 build 数据
- backend 不需要知道原始数据目录结构
- 不同 `key_builder.type` 产物必须分文件保存
- 一个 run 只能加载一份已经和当前 `key_builder.type` 对齐的 artifact

---

## `Weighted Score Sum` 落地要求

必须明确不是 raw score sum。

字段规则固定为：

- `vision_0 / vision_1 / prompt_emb`：
  - Layer 1 用 cosine
  - 再映射到 `[0,1]`：`(cos + 1) / 2`
- `robot_state`：
  - Layer 1 用 L2
  - 再转相似度：`exp(-d / tau)`
  - 当前固定 `tau = 0.334717`

然后再做 percentile normalization：

- `ŝ_f = clip((s_f - p5_f) / (p95_f - p5_f), 0, 1)`

最后才允许做：

- `Score(x) = Σ_f w_f * ŝ_f(x)`

因此 backend 必须支持：

- 同时返回字段原始分数与最终融合分数
- 在 `weighted_score_sum` 下严格先 normalization 后 sum

---

## `Weighted RRF` 落地要求

backend 必须支持：

- 每个字段先独立排序
- 按字段权重做加权 RRF
- `robot_state` 的 L2 结果先按“距离越小越相似”的顺序产生 rank

RRF 的职责只依赖 rank，不直接依赖 score 尺度。

---

## `in_memory` 后端重构目标

### 第一版必须支持

- 多字段 entry/query
- `checkpoint_id` 过滤
- `task_key` 过滤
- `step_range` 过滤
- `vision/prompt = cosine`
- `robot_state = L2`
- `weighted_rrf`
- `weighted_score_sum`
- brute-force top-k

### 第一版可以暂不支持

- 近似索引
- 多线程并行算分
- 磁盘持久化
- Qdrant 等价的 payload 序列化复杂性

---

## 推荐实施顺序

### Phase 0：数据前置检查与校准

新增：

- `exp/build_in_memory_cache_artifact.py`
- `exp/calibrate_score_sum_stats.py`

目标：

- 确认 `data/libero_spatial` 直接提供 `vision_0/1/2`、`prompt_emb`、`robot_state`
- 为 4 个 builder 产出 artifact
- 对各 builder 做离线 sanity check
- 统计：
  - `tau`
  - 每个 builder / field 的 `p5/p95`
- 产出可直接写入正式 `SUM` YAML 的 calibration 结果

说明：

- 正式 `SUM` 实验禁止使用占位 `p5=0.0, p95=1.0`
- 在当前数据集上，这一阶段不需要额外离线 stage1 推理

### Phase A：扩类型和配置

修改：

- `storage_types.py`
- `config.py`
- `search_strategy.py`

目标：

- `QuerySpec` 能表达实验语义
- YAML 能表达 reducer / fusion / normalization
- 工厂能正确实例化新组件

### Phase B：补 CP1 实验版降维器

修改：

- `components/key_builder.py`

目标：

- 支持 A / B1 / B2 / C 四类降维
- 输出维度和 `backend.vector_dims` 对齐
- 4 个对外 builder type 共用一套内部 helper

### Phase C：重写 `in_memory` 后端

修改：

- `backends/in_memory_backend.py`

目标：

- `in_memory` 成为实验可用后端
- 所有 Layer 1 / Layer 2 逻辑落到 backend 内部私有函数
- `weight == 0` 字段直接跳过

### Phase D：批量生成 YAML

新增：

- `exp/generate_cache_run_yamls.py`

目标：

- Phase 1 先生成 `64` 个独立 YAML（8 combos × 8 weights）
- Phase 1.5 / Phase 2 的 YAML 在分析结果后二次调用同一脚本生成
- 每个 YAML 都带上正确的 calibration 和 `preload_path`

### Phase E：实验控制与结果汇总

新增：

- `exp/run_cache_experiments.py`
- `exp/analyze_cache_results.py`

目标：

- 支持指定 run 子集
- 支持指定每个 run 的 episode 个数
- 支持 run 级断点重续
- 汇总结果并选出 `Phase 1` top 3
- 尽量复用 `examples/libero/main.py`

### Phase F：server 动态切换 YAML

总体约束：

- **支持单个 run 在固定 YAML 下使用多 worker 并发评测**
- **不支持多个不同 run 同时共享一个 server 并发执行**
- run 与 run 串行；单个 run 内允许 `num_workers > 1`
- 实验 server 统一用 `--concurrent` 启动（即使 `num_workers=1`）

设计：server 维护全局 `CurrentCacheBundle`（含 `cache_config` + `shared_storage` + `version`）。

流程：

1. 控制脚本开一个短生命周期 WebSocket 连接
2. 发送 `{"__ctrl__": "load_cache_config", "yaml_path": "<abs_path>"}`
3. Server 校验 YAML、构建新的 `shared_storage`、原子替换整个 bundle
4. Server 返回 `{"__ack__": "load_cache_config", "version": N}`
5. 控制连接断开
6. 启动 `examples/libero/main.py --num_workers N`，N 个 worker 各自建连接
7. `connection_policy_factory` 读取当前 bundle 快照，调 `build_per_connection_components(bundle.cache_config, bundle.shared_storage)`
8. 所有 worker 完成后，再切下一个 run

Storage 共享语义：

- 同一个 run 的多个 worker 连接共享同一个 `bundle.shared_storage`
- 不同 run 切 YAML 时构建新的 `shared_storage`
- 已在跑的旧连接继续用旧 bundle，不受影响

修改：

- `src/openpi/serving/websocket_policy_server.py`：新增 `CurrentCacheBundle` + `load_cache_config` 控制消息
- `scripts/serve_policy.py`：`connection_policy_factory` 从全局 bundle 读取最新配置

不修改：

- `packages/openpi-client/` — 控制脚本直接用 `websockets` + `msgpack`
- `examples/libero/main.py` — 不改，每个 run 自然开新连接

---

## 本轮不做

- `CP2`
- `CP3`
- Judge threshold 实验
- Gate 策略实验
- 真实 Qdrant 后端对齐
- 把 in-memory 做成生产级 ANN 引擎

---

## 交付标准

完成后应满足：

1. 单个 YAML 能完整表达一组实验配置
2. `CP1` 路径可直接切换：
   - 降维方法
   - 字段权重
   - 融合方法
3. `in_memory` 后端可以真实跑：
   - `weighted_rrf`
   - `weighted_score_sum`
4. 各模块职责清晰：
   - key builder 只做降维
   - strategy 只组装 query
   - backend 只做检索
   - judge 只放行 top-1
5. 后续扩展 Qdrant 时，可以复用：
   - key builder
   - config schema
   - search strategy
   - calibration 配置
6. Phase 1 的 top 3 选择规则明确：
   - 固定 `episodes-per-run` 预算
   - 按 `aggregate success rate` 排序
   - 最佳配置可追加更大 budget confirm run

---

## 下一步

下一步按以下顺序推进：

1. 先完成 `Phase 0`，把 artifact、sanity check、`tau`、`p5/p95` 都跑出来
2. 再扩 `QuerySpec` 和 YAML schema
3. 再补 4 个薄 wrapper key builder
4. 再重写 `in_memory` 后端
5. 再生成正式实验 YAML 和结果汇总脚本

---

## 代码级详细规格

下面按 Phase 顺序，给出每个文件的精确修改规格：函数签名、数据结构、算法伪码、测试用例。

---

### 契约修订

本轮实验对现有系统契约做以下修订，需要在代码中同步更新 docstring：

#### 1. `CacheEntry.query_keys` 不再要求全字段 L2-normalised

现有 `storage_types.py:119` 注释写 "CPU float32, L2-normalised"。

修订为：

> `query_keys` 中的 tensor 必须满足 CPU float32 contiguous。
> 是否 L2 normalise 取决于字段的相似度计算方式：
>   - cosine 字段（vision_0/1/2, prompt_emb）：不要求 L2 normalise（`F.cosine_similarity` 内部处理）
>   - L2 距离字段（robot_state）：必须保留 raw vector，不做 L2 normalise（否则 L2 distance 的物理含义被破坏）

原因：`robot_state` 需要 backend 计算真实 L2 distance 后做 `exp(-d/tau)` 转换。如果 builder 预先 normalize，距离语义就变了。

实际影响很小：现有代码（`CacheStorage.insert`、`InMemoryBackend.search`）从未在运行时检查 L2 norm，只有 docstring 声明了这个约束。

#### 2. `robot_state` 明确作为 raw vector 保留

所有 CP1 builder 对 `robot_state` 只做 `_to_cpu_float32()`，不做 L2 normalize、不做 mean pool。

这不是"例外"，而是设计要求：robot_state 的 L2 distance 是实验的核心信号，normalize 会丢失距离信息。

#### 3. artifact entry id 允许脱离 stable-hash 语义

现有 `storage_types.py:109` 描述了 id 的 stable-hash 语义（`sha256(checkpoint_id + query_key_bytes)`）。

修订：
- 在线写入路径（Orchestrator.write）仍使用 stable-hash
- 离线 artifact 的 entry id 使用 `"{episode_file_stem}_{step_name}"`，格式确定性且可追溯
- 约束：**本轮实验中，artifact-loaded backend 不混合在线写入**（同一个 InMemoryBackend 实例要么只从 artifact 加载，要么只接受在线 insert，不同时做两件事）

原因：离线构建时没有 query_keys 的字节级确定性（浮点精度），stable-hash 不保证稳定。用文件名+step名更可追溯。

#### 4. `CachePayload.action_chunk` 保持 `[horizon, 32]` 但允许 horizon < 50

现有 `storage_types.py:87` 注释写 `[50, 32]`。

修订：
- 离线 artifact 中，action_chunk 保留数据真实 horizon（如 `[10, 32]`）
- 不做 zero-pad 到 `[50, 32]`，因为 LIBERO 客户端只使用前几步 action，多余的 pad 无意义
- artifact 构建脚本的注释中说明 action_chunk shape 取决于数据集

如果后续发现 Interceptor 或客户端对 `[50, 32]` 有硬性依赖，再在 artifact 脚本中加 pad 逻辑。当前不预加。

#### 5. vision_2 的支持范围定义

明确为 **builder 层支持，artifact/backend/YAML 当前不产出**：

- `_CP1BaseKeyBuilder.build()` 对 vision_2 有代码路径（如果 `enabled_fields` 包含 vision_2）
- `_build_fake_stage1` 用零填充 vision_2 位置，是为了维持 prefix_embs 的 token layout 正确（vision_1 的 offset 在 256-512，如果不填 vision_2 的 512-768 区域，prompt 的起始位置就错了）
- 本轮 LIBERO 实验 YAML 统一设置 `keys.vision_2.enabled: false`
- `vector_dims` 和 artifact 不包含 vision_2
- builder 测试覆盖 vision_2 enabled 的代码路径，但实验不使用
- 未来如需启用 vision_2，只需改 YAML + 重建 artifact + 扩 vector_dims，不需要改 builder 代码

#### 6. 命名约定

两层命名统一：

| 层 | 字段 | 值 | 说明 |
|---|---|---|---|
| config / YAML | `search_strategy.type` | `weighted_rrf_knn` / `weighted_score_sum_knn` | 带 `_knn` 后缀，和现有 `qdrant_weighted_rrf_knn` 风格一致 |
| QuerySpec / backend | `fusion_method` | `weighted_rrf` / `weighted_score_sum` | 不带 `_knn`，backend 不关心检索是 KNN 还是 ANN |

SearchStrategy 在构造 QuerySpec 时负责去掉 `_knn` 后缀。

---

### Phase A：扩类型和配置

#### A1. `src/openpi/cache/storage_types.py` — QuerySpec 扩展

在现有 `QuerySpec` dataclass 中新增 3 个一级字段：

```python
@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None
    fusion_weights: Optional[dict[str, float]] = None
    backend_hints: Optional[dict[str, Any]] = None

    # --- 新增字段 ---
    fusion_method: Optional[str] = None
    # 取值: "weighted_rrf" | "weighted_score_sum" | None
    # None 时 backend 回退到现有单字段 cosine 行为（向后兼容）

    field_similarity: Optional[dict[str, dict[str, Any]]] = None
    # 每个字段的相似度定义，例如：
    # {
    #   "vision_0":    {"type": "cosine"},
    #   "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
    # }

    score_normalization: Optional[dict[str, Any]] = None
    # 仅 weighted_score_sum 需要，例如：
    # {
    #   "type": "percentile",
    #   "fields": {
    #     "vision_0":    {"p5": 0.82, "p95": 0.99},
    #     "robot_state": {"p5": 0.15, "p95": 0.88},
    #   }
    # }
```

不修改 `CacheEntry`、`CachePayload`、`SearchResultLite`、`SearchResult`。

不修改 `QueryFilter`（`task_key` 和 `step_range` 已经存在）。

#### A2. `src/openpi/cache/config.py` — 配置 schema 扩展

##### A2.1 `KeyBuilderConfig` 扩展

不需要额外字段，只扩 type 枚举：

```python
@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"
    # 新增合法值: "cp1_mean_pool", "cp1_spatial_pool_16",
    #            "cp1_spatial_pool_64", "cp1_max_pool"
```

##### A2.2 `SearchStrategyConfig` 扩展

```python
@dataclass
class FieldSimilarityConfig:
    type: str = "cosine"           # "cosine" | "l2"
    to_similarity: Optional[dict[str, Any]] = None
    # 仅 l2 使用，例如: {"type": "exp", "tau": 0.334717}

@dataclass
class ScoreNormalizationConfig:
    type: str = "none"             # "none" | "percentile"
    fields: Optional[dict[str, dict[str, float]]] = None
    # 仅 percentile 使用，例如: {"vision_0": {"p5": 0.82, "p95": 0.99}}

@dataclass
class SearchStrategyConfig:
    type: str = "qdrant_weighted_rrf_knn"
    # 新增合法值: "weighted_rrf_knn", "weighted_score_sum_knn"
    top_k: int = 1
    step_filter: str = "all"
    step_window: int = 5
    rrf_k: int = 60
    candidate_multiplier: int = 5

    # --- 新增字段 ---
    field_similarity: Optional[dict[str, FieldSimilarityConfig]] = None
    # 键为字段名 (vision_0, robot_state, ...)
    # 值为 FieldSimilarityConfig
    # 仅 weighted_rrf_knn / weighted_score_sum_knn 使用

    score_normalization: Optional[ScoreNormalizationConfig] = None
    # 仅 weighted_score_sum_knn 使用
```

##### A2.3 `BackendConfig` 扩展

```python
@dataclass
class InMemoryConfig:
    preload_path: Optional[str] = None    # artifact .pkl 路径
    index_type: str = "brute_force"       # 当前只有 brute_force

@dataclass
class BackendConfig:
    type: str = "qdrant"
    vector_dims: dict[str, int] = field(default_factory=lambda: {"robot_state": 32})
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    in_memory: InMemoryConfig = field(default_factory=InMemoryConfig)
    # 新增 in_memory 配置块
```

##### A2.4 `_dict_to_dataclass` 扩展

需要处理新的嵌套 dataclass：
- `field_similarity` 字典值 → `FieldSimilarityConfig`
- `score_normalization` → `ScoreNormalizationConfig`
- `in_memory` → `InMemoryConfig`

在 `_CONFIG_TYPES` 注册表中新增：
```python
_CONFIG_TYPES: dict[str, type] = {
    ...  # 保留现有
    "FieldSimilarityConfig": FieldSimilarityConfig,
    "ScoreNormalizationConfig": ScoreNormalizationConfig,
    "InMemoryConfig": InMemoryConfig,
}
```

对 `field_similarity` 这类 `dict[str, FieldSimilarityConfig]` 字段，在 `_dict_to_dataclass` 中增加特殊处理（与 `checkpoints` 类似）：

```python
elif key == "field_similarity" and isinstance(value, dict):
    result = {}
    for field_name, field_data in value.items():
        if isinstance(field_data, dict):
            result[field_name] = _dict_to_dataclass(FieldSimilarityConfig, field_data)
        else:
            result[field_name] = field_data
    kwargs[key] = result
```

##### A2.5 `validate_cache_config` 扩展

新增校验规则：

```python
# 现有规则保留不变

# 新增规则 8: key_builder.type 合法值扩展
_VALID_KEY_BUILDER_TYPES = frozenset({
    "placeholder", "full_original",
    "cp1_mean_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_64", "cp1_max_pool",
})
if config.key_builder.type not in _VALID_KEY_BUILDER_TYPES:
    errors.append(...)

# 新增规则 9: search_strategy.type 合法值扩展
_VALID_STRATEGY_TYPES = frozenset({
    "qdrant_weighted_rrf_knn", "weighted_rrf_knn", "weighted_score_sum_knn",
})

# 新增规则 10: weighted_score_sum_knn 必须有 score_normalization
for cp_name, cp_config in config.checkpoints.items():
    ss = cp_config.search_strategy
    if ss.type == "weighted_score_sum_knn":
        if ss.score_normalization is None or ss.score_normalization.type == "none":
            errors.append(f"{cp_name}.search_strategy: weighted_score_sum_knn 需要 score_normalization")
        if ss.score_normalization and ss.score_normalization.type == "percentile":
            if not ss.score_normalization.fields:
                errors.append(f"{cp_name}.search_strategy: percentile normalization 需要 fields")

# 新增规则 11: weighted_rrf_knn / weighted_score_sum_knn 需要 backend.type == "in_memory"
for cp_name, cp_config in config.checkpoints.items():
    ss = cp_config.search_strategy
    if ss.type in ("weighted_rrf_knn", "weighted_score_sum_knn"):
        if config.backend.type != "in_memory":
            errors.append(f"{cp_name}: {ss.type} 需要 backend.type='in_memory'")

# 新增规则 12: cp1_* key_builder 需要的 enabled fields
_CP1_BUILDER_REQUIRED_FIELDS = frozenset({"vision_0", "robot_state"})
if config.key_builder.type.startswith("cp1_"):
    for f in _CP1_BUILDER_REQUIRED_FIELDS:
        if f not in enabled_fields:
            errors.append(f"key_builder.type={config.key_builder.type} 需要 {f} enabled")

# 新增规则 13: in_memory backend + cp1_* builder 时 preload_path 应存在
if config.backend.type == "in_memory" and config.key_builder.type.startswith("cp1_"):
    if not config.backend.in_memory.preload_path:
        errors.append("in_memory backend + cp1 builder 需要 backend.in_memory.preload_path")
```

##### A2.6 工厂函数扩展

`_build_key_builder` 新增分支：

```python
def _build_key_builder(cfg: KeyBuilderConfig, enabled_fields: list[str], vector_dims: dict[str, int]):
    if cfg.type == "placeholder":
        ...  # 不变
    elif cfg.type == "full_original":
        ...  # 不变
    elif cfg.type == "cp1_mean_pool":
        from openpi.cache.components.key_builder import CP1MeanPoolKeyBuilder
        return CP1MeanPoolKeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_spatial_pool_16":
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
        return CP1SpatialPool16KeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_spatial_pool_64":
        from openpi.cache.components.key_builder import CP1SpatialPool64KeyBuilder
        return CP1SpatialPool64KeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_max_pool":
        from openpi.cache.components.key_builder import CP1MaxPoolKeyBuilder
        return CP1MaxPoolKeyBuilder(enabled_fields=enabled_fields)
    else:
        raise ConfigValidationError(...)
```

`_build_search_strategy` 新增分支：

```python
def _build_search_strategy(cfg: SearchStrategyConfig, storage, fusion_weights: dict[str, float]):
    if cfg.type == "qdrant_weighted_rrf_knn":
        ...  # 不变
    elif cfg.type == "weighted_rrf_knn":
        from openpi.cache.components.search_strategy import WeightedRrfKnnStrategy
        return WeightedRrfKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            fusion_weights=fusion_weights if fusion_weights else None,
            rrf_k=cfg.rrf_k,
            field_similarity=_field_similarity_to_dict(cfg.field_similarity),
        )
    elif cfg.type == "weighted_score_sum_knn":
        from openpi.cache.components.search_strategy import WeightedScoreSumKnnStrategy
        return WeightedScoreSumKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            fusion_weights=fusion_weights if fusion_weights else None,
            field_similarity=_field_similarity_to_dict(cfg.field_similarity),
            score_normalization=_score_norm_to_dict(cfg.score_normalization),
        )
    else:
        raise ConfigValidationError(...)
```

新增两个私有 helper 将 config dataclass 转为 plain dict（传入 QuerySpec）：

```python
def _field_similarity_to_dict(
    cfg: Optional[dict[str, FieldSimilarityConfig]],
) -> Optional[dict[str, dict[str, Any]]]:
    """FieldSimilarityConfig dict -> plain dict for QuerySpec.field_similarity."""
    if cfg is None:
        return None
    result = {}
    for name, fs in cfg.items():
        d: dict[str, Any] = {"type": fs.type}
        if fs.to_similarity is not None:
            d["to_similarity"] = fs.to_similarity
        result[name] = d
    return result

def _score_norm_to_dict(
    cfg: Optional[ScoreNormalizationConfig],
) -> Optional[dict[str, Any]]:
    """ScoreNormalizationConfig -> plain dict for QuerySpec.score_normalization."""
    if cfg is None:
        return None
    d: dict[str, Any] = {"type": cfg.type}
    if cfg.fields is not None:
        d["fields"] = cfg.fields
    return d
```

`_build_backend` 扩展 in_memory 分支：

```python
if cfg.type == "in_memory":
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    backend = InMemoryBackend(vector_dims=cfg.vector_dims)
    if cfg.in_memory.preload_path:
        backend.load_artifact(cfg.in_memory.preload_path)
    return backend
```

#### A3. `src/openpi/cache/components/search_strategy.py` — 新增两个 Strategy

##### A3.1 `WeightedRrfKnnStrategy`

```python
class WeightedRrfKnnStrategy:
    """In-memory weighted RRF search strategy.

    构造 QuerySpec 时写入:
      fusion_method = "weighted_rrf"
      field_similarity = 从配置传入
      backend_hints = {"rrf_k": self._rrf_k}
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._rrf_k = rrf_k
        self._field_similarity = field_similarity

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        filters = self._build_filters(ctx)  # 复用与 Qdrant strategy 相同的逻辑
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_rrf",
            field_similarity=self._field_similarity,
            backend_hints={"rrf_k": self._rrf_k},
        )
        return self._storage.search(spec)

    def _build_filters(self, ctx: SearchContext) -> Optional[QueryFilter]:
        # 与 QdrantWeightedRrfKnnStrategy._build_filters 逻辑完全相同
        # 复用方式: 提取为模块级私有函数 _build_step_filters()
        ...
```

##### A3.2 `WeightedScoreSumKnnStrategy`

```python
class WeightedScoreSumKnnStrategy:
    """In-memory weighted score sum search strategy.

    构造 QuerySpec 时写入:
      fusion_method = "weighted_score_sum"
      field_similarity = 从配置传入
      score_normalization = 从配置传入
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
        score_normalization: Optional[dict[str, Any]] = None,
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._field_similarity = field_similarity
        self._score_normalization = score_normalization

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        filters = _build_step_filters(self._step_filter, self._step_window, ctx)
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_score_sum",
            field_similarity=self._field_similarity,
            score_normalization=self._score_normalization,
        )
        return self._storage.search(spec)
```

##### A3.3 提取共享 filter 构建函数

```python
def _build_step_filters(
    step_filter: str,
    step_window: int,
    ctx: SearchContext,
) -> Optional[QueryFilter]:
    """共享的 step filter 构建逻辑，三个 strategy 类共用。"""
    task_filter = QueryFilter(task_key=ctx.task_key) if ctx.task_key else None

    if step_filter == "all":
        return task_filter
    elif step_filter == "exact":
        f = QueryFilter(step_range=(ctx.current_step, ctx.current_step))
        if ctx.task_key:
            f.task_key = ctx.task_key
        return f
    elif step_filter == "window":
        lo = max(0, ctx.current_step - step_window)
        hi = ctx.current_step + step_window
        f = QueryFilter(step_range=(lo, hi))
        if ctx.task_key:
            f.task_key = ctx.task_key
        return f
    else:
        raise ValueError(f"Unknown step_filter: {step_filter}")
```

同时将 `QdrantWeightedRrfKnnStrategy._build_filters` 改为调用这个共享函数。

---

### Phase B：补 CP1 实验版降维器

#### B1. `src/openpi/cache/components/key_builder.py` — 新增内容

##### B1.1 共享私有 helper 函数

```python
# ---------------------------------------------------------------------------
# CP1 experiment key builder helpers (private)
# ---------------------------------------------------------------------------

def _slice_cp1_fields(
    prefix_embs: torch.Tensor,
    state: torch.Tensor,
    enabled: set[str] | None,
) -> dict[str, torch.Tensor]:
    """从 prefix_embs[0] 和 state[0] 中切出各模态原始 token 序列。

    返回 dict，键为字段名，值为 GPU tensor（未降维、未转 CPU）。
    vision_*: [256, emb_dim]
    prompt_emb: [num_prompt_tokens, emb_dim]
    robot_state: [state_dim]
    """
    result: dict[str, torch.Tensor] = {}
    prefix = prefix_embs[0]  # [prefix_len, emb_dim]，drop batch dim

    for field_name, start, end in _VISION_OFFSETS:
        if enabled is not None and field_name not in enabled:
            continue
        result[field_name] = prefix[start:end]  # [256, emb_dim]

    if enabled is None or PROMPT_EMB in enabled:
        result[PROMPT_EMB] = prefix[_PROMPT_START:]  # [num_prompt_tokens, emb_dim]

    if enabled is None or ROBOT_STATE in enabled:
        result[ROBOT_STATE] = state[0]  # [state_dim]

    return result


def _mean_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """[num_tokens, emb_dim] -> [emb_dim]，对 token 维取均值。"""
    return tokens.mean(dim=0)


def _max_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """[num_tokens, emb_dim] -> [emb_dim]，对 token 维逐维取最大值。"""
    return tokens.max(dim=0).values


def _spatial_pool_tokens(
    tokens: torch.Tensor,
    grid_size: int,
    pool_size: int,
) -> torch.Tensor:
    """对 vision token 做 2D adaptive average pooling。

    Args:
        tokens: [grid_size*grid_size, emb_dim]，SigLIP 的扁平化 patch tokens
        grid_size: 原始网格边长（16，因为 256 = 16×16）
        pool_size: 池化后的网格边长（4 对应 B1，2 对应 B2）

    Returns:
        [pool_size*pool_size, emb_dim] 然后 flatten 为 [pool_size*pool_size * emb_dim]

    实现:
        1. reshape [grid_size*grid_size, emb_dim] -> [1, emb_dim, grid_size, grid_size]
           （把 emb_dim 当 channel，空间维度当 H×W）
        2. F.adaptive_avg_pool2d(..., (pool_size, pool_size))
        3. reshape -> [pool_size*pool_size * emb_dim]
    """
    emb_dim = tokens.shape[1]
    # [grid*grid, emb_dim] -> [1, emb_dim, grid, grid]
    x = tokens.reshape(grid_size, grid_size, emb_dim).permute(2, 0, 1).unsqueeze(0)
    # [1, emb_dim, pool, pool]
    pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
    # -> [pool*pool * emb_dim]
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def _to_cpu_float32(t: torch.Tensor) -> torch.Tensor:
    """GPU tensor -> CPU float32 contiguous，统一 D2H 出口。"""
    return t.cpu().float().contiguous()
```

##### B1.2 共享基类 `_CP1BaseKeyBuilder`

```python
class _CP1BaseKeyBuilder:
    """CP1 实验 key builder 的共享基础。

    子类只需覆盖 _reduce_vision() 和 _reduce_prompt()。
    """

    def __init__(self, enabled_fields: list[str] | None = None) -> None:
        self._cache: dict[str, torch.Tensor] = {}
        self._enabled = set(enabled_fields) if enabled_fields is not None else None

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._cache["state"] = s1.state               # [B, state_dim] GPU
            self._cache["prefix_embs"] = s1.prefix_embs   # [B, prefix_len, emb_dim] GPU
        if "stage3" in stage_outputs:
            self._cache["action_chunk"] = stage_outputs["stage3"].action_chunk

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

        raw = _slice_cp1_fields(
            self._cache["prefix_embs"],
            self._cache["state"],
            self._enabled,
        )
        keys: dict[str, torch.Tensor] = {}

        # Vision fields: 调用子类的 _reduce_vision
        for field_name in (VISION_0, VISION_1, VISION_2):
            if field_name in raw:
                keys[field_name] = _to_cpu_float32(self._reduce_vision(raw[field_name]))

        # Prompt field: 调用子类的 _reduce_prompt
        if PROMPT_EMB in raw:
            keys[PROMPT_EMB] = _to_cpu_float32(self._reduce_prompt(raw[PROMPT_EMB]))

        # Robot state: 原样输出（不降维、不 L2 normalize）
        if ROBOT_STATE in raw:
            keys[ROBOT_STATE] = _to_cpu_float32(raw[ROBOT_STATE])

        return keys

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        """子类覆盖：[256, emb_dim] -> [reduced_dim]"""
        raise NotImplementedError

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        """子类覆盖：[num_tokens, emb_dim] -> [reduced_dim]"""
        raise NotImplementedError

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()
```

注意：`robot_state` 不做 L2 normalize。原因是 backend 在 RRF 模式下需要原始 L2 distance，在 SUM 模式下会先算 L2 再做 `exp(-d/tau)`。如果 builder 预先 normalize 了，L2 distance 的物理含义就变了。

##### B1.3 四个薄 wrapper 类

```python
class CP1MeanPoolKeyBuilder(_CP1BaseKeyBuilder):
    """A 组: Mean Pool。vision/prompt 均值池化。
    输出 vision_*: [emb_dim=2048], prompt_emb: [emb_dim=2048]。
    """
    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1SpatialPool16KeyBuilder(_CP1BaseKeyBuilder):
    """B1 组: Spatial Pool 4×4 (16× 压缩)。
    输出 vision_*: [16*2048=32768], prompt_emb: [emb_dim=2048] (mean pool)。
    """
    _GRID_SIZE = 16   # sqrt(256)
    _POOL_SIZE = 4    # 16×16 -> 4×4

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1SpatialPool64KeyBuilder(_CP1BaseKeyBuilder):
    """B2 组: Spatial Pool 2×2 (64× 压缩)。
    输出 vision_*: [4*2048=8192], prompt_emb: [emb_dim=2048] (mean pool)。
    """
    _GRID_SIZE = 16
    _POOL_SIZE = 2    # 16×16 -> 2×2

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1MaxPoolKeyBuilder(_CP1BaseKeyBuilder):
    """C 组: Max Pool。vision/prompt 逐维取最大值。
    输出 vision_*: [emb_dim=2048], prompt_emb: [emb_dim=2048]。
    """
    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)
```

##### B1.4 输出维度速查表

| builder type | vision_0/1 | prompt_emb | robot_state |
|---|---|---|---|
| cp1_mean_pool | 2,048 | 2,048 | 32 |
| cp1_spatial_pool_16 | 32,768 | 2,048 | 32 |
| cp1_spatial_pool_64 | 8,192 | 2,048 | 32 |
| cp1_max_pool | 2,048 | 2,048 | 32 |

YAML 的 `backend.vector_dims` 必须与此表一致，由 YAML 生成脚本保证。

---

### Phase C：重写 in_memory 后端

#### C1. `src/openpi/cache/backends/in_memory_backend.py` — 完整重写

##### C1.1 类签名和构造函数

```python
class InMemoryBackend(VectorStoreBackend):
    """In-memory backend，支持多字段检索和两种融合方式。

    功能:
      - 多字段 entry/query (vision_0/1/2, prompt_emb, robot_state)
      - checkpoint_id / task_key / step_range 过滤
      - 字段级相似度: cosine / L2
      - 融合: weighted_rrf / weighted_score_sum
      - brute-force top-k
    """

    def __init__(self, vector_dims: dict[str, int]) -> None:
        self._dims = vector_dims
        self._entries: dict[str, CacheEntry] = {}
        self.search_call_count: int = 0
        self.fetch_payload_call_count: int = 0
```

##### C1.2 `supported_filters` 扩展

```python
def supported_filters(self) -> frozenset[str]:
    return frozenset({"checkpoint_id", "task_key", "step_range"})
```

##### C1.3 `load_artifact` 新增方法

```python
def load_artifact(self, path: str) -> None:
    """从 pickle 文件加载预构建的 entries。

    Artifact 格式 (dict):
      {
        "key_builder_type": str,
        "checkpoint_id": str,
        "vector_dims": dict[str, int],
        "entries": list[CacheEntry],
      }

    校验:
      - artifact 的 vector_dims 必须与 self._dims 一致
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data["vector_dims"] != self._dims:
        raise ValueError(
            f"Artifact vector_dims mismatch: "
            f"artifact={data['vector_dims']}, backend={self._dims}"
        )
    for entry in data["entries"]:
        self._entries[entry.id] = entry
    logger.info("Loaded %d entries from %s", len(data["entries"]), path)
```

##### C1.4 `_filter_entries` 私有方法

```python
def _filter_entries(self, spec: QuerySpec) -> list[CacheEntry]:
    """按 checkpoint_id / task_key / step_range 过滤。"""
    results = []
    for entry in self._entries.values():
        if spec.checkpoint_id is not None and entry.checkpoint_id != spec.checkpoint_id:
            continue
        if spec.filters is not None:
            if spec.filters.task_key is not None:
                if entry.payload.task_key != spec.filters.task_key:
                    continue
            if spec.filters.step_range is not None:
                # step_idx 存在 entry 的 metadata 中
                # 如果 entry 没有 step_idx 信息，不做过滤（兼容无 step_idx 的 artifact）
                step_idx = getattr(entry, "step_idx", None)
                if step_idx is not None:
                    lo, hi = spec.filters.step_range
                    if not (lo <= step_idx <= hi):
                        continue
        results.append(entry)
    return results
```

注意：`CacheEntry` 当前没有 `step_idx` 字段。如需 step_range 过滤，有两种处理方式：
- 方案 A（推荐）：本轮实验 step_filter 固定为 "all"，不使用 step_range 过滤，不需要改 CacheEntry
- 方案 B：在 CacheEntry 增加 optional `metadata: dict[str, Any]` 字段

本轮实验选方案 A：step_range 过滤代码写好但实际不触发。

##### C1.5 字段级打分私有方法

```python
def _cosine_score(self, q: torch.Tensor, e: torch.Tensor) -> float:
    """计算 cosine similarity，返回 [-1, 1]。"""
    return float(F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)))


def _l2_distance(self, q: torch.Tensor, e: torch.Tensor) -> float:
    """计算 L2 距离（欧氏距离），返回 >= 0。"""
    return float(torch.norm(q.float() - e.float(), p=2))


def _compute_field_score(
    self,
    field_name: str,
    q: torch.Tensor,
    e: torch.Tensor,
    field_sim_config: dict[str, Any],
) -> float:
    """根据 field_similarity 配置计算单字段原始分数。

    cosine 字段: 返回 cosine similarity [-1, 1]
    l2 字段: 返回负 L2 距离（越大越相似，用于 RRF 排序）
             或返回 similarity = exp(-d/tau)（用于 score sum）
             具体行为由调用方决定，这里只返回原始值

    返回约定:
      cosine: 返回 cosine similarity (越大越相似)
      l2: 返回 L2 distance (越小越相似)，调用方负责方向转换
    """
    sim_type = field_sim_config.get("type", "cosine")
    if sim_type == "cosine":
        return self._cosine_score(q, e)
    elif sim_type == "l2":
        return self._l2_distance(q, e)
    else:
        raise ValueError(f"Unknown similarity type: {sim_type}")
```

##### C1.6 `_iter_active_fields` 私有方法

```python
def _iter_active_fields(
    self,
    spec: QuerySpec,
) -> list[tuple[str, float, dict[str, Any]]]:
    """返回本次搜索中活跃字段列表: [(field_name, weight, sim_config), ...]

    活跃条件:
      1. field 在 spec.query_keys 中
      2. field 在 self._dims 中
      3. weight > 0 (或 fusion_weights 为 None 时全部参与)
    """
    result = []
    weights = spec.fusion_weights or {}
    sim_configs = spec.field_similarity or {}
    for field_name in spec.query_keys:
        if field_name not in self._dims:
            continue
        w = weights.get(field_name, 1.0)
        if w <= 0:
            continue
        sim_cfg = sim_configs.get(field_name, {"type": "cosine"})
        result.append((field_name, w, sim_cfg))
    return result
```

##### C1.7 `_search_weighted_rrf` 私有方法

```python
def _search_weighted_rrf(
    self,
    candidates: list[CacheEntry],
    spec: QuerySpec,
    active_fields: list[tuple[str, float, dict[str, Any]]],
) -> list[SearchResultLite]:
    """Weighted RRF 融合。

    算法:
      1. 对每个活跃字段，计算所有候选的原始分数
      2. 按分数排序，得到每个字段的独立 rank (rank 1 = 最佳)
         - cosine: 分数越大 rank 越小
         - l2: 距离越小 rank 越小
      3. 加权 RRF: score(x) = Σ_f w_f / (rrf_k + rank_f(x))
      4. 按 RRF score 降序排序，取 top_k
    """
    rrf_k = 60
    if spec.backend_hints:
        rrf_k = spec.backend_hints.get("rrf_k", 60)

    # 字段级排序
    per_field_ranks: dict[str, dict[str, int]] = {}  # field -> {entry_id: rank}
    for field_name, weight, sim_cfg in active_fields:
        sim_type = sim_cfg.get("type", "cosine")
        scores: list[tuple[str, float]] = []
        for entry in candidates:
            if field_name not in entry.query_keys or field_name not in spec.query_keys:
                continue
            raw = self._compute_field_score(
                field_name,
                spec.query_keys[field_name],
                entry.query_keys[field_name],
                sim_cfg,
            )
            scores.append((entry.id, raw))

        # 排序: cosine 越大越好 → 降序; L2 越小越好 → 升序
        if sim_type == "cosine":
            scores.sort(key=lambda x: x[1], reverse=True)
        else:  # l2
            scores.sort(key=lambda x: x[1], reverse=False)

        ranks = {eid: rank + 1 for rank, (eid, _) in enumerate(scores)}
        per_field_ranks[field_name] = ranks

    # RRF 融合
    rrf_scores: dict[str, float] = {}
    entry_map: dict[str, CacheEntry] = {e.id: e for e in candidates}
    for entry in candidates:
        total = 0.0
        for field_name, weight, _sim_cfg in active_fields:
            rank = per_field_ranks.get(field_name, {}).get(entry.id)
            if rank is not None:
                total += weight / (rrf_k + rank)
        rrf_scores[entry.id] = total

    # 排序取 top_k
    sorted_ids = sorted(rrf_scores, key=lambda eid: rrf_scores[eid], reverse=True)
    results = []
    for eid in sorted_ids[: spec.top_k]:
        entry = entry_map[eid]
        results.append(
            SearchResultLite(id=eid, score=rrf_scores[eid], checkpoint_id=entry.checkpoint_id)
        )
    return results
```

##### C1.8 `_search_weighted_score_sum` 私有方法

```python
def _search_weighted_score_sum(
    self,
    candidates: list[CacheEntry],
    spec: QuerySpec,
    active_fields: list[tuple[str, float, dict[str, Any]]],
) -> list[SearchResultLite]:
    """Weighted Score Sum 融合。

    算法:
      1. 对每个活跃字段，计算所有候选的原始分数
      2. 将原始分数转为 [0, 1] 区间的相似度:
         - cosine: s_01 = (cos + 1) / 2
         - l2: s = exp(-d / tau)
      3. percentile normalization: ŝ = clip((s - p5) / (p95 - p5), 0, 1)
      4. 加权求和: Score(x) = Σ_f w_f * ŝ_f(x)
      5. 按 Score 降序排序，取 top_k
    """
    norm_config = spec.score_normalization or {}
    norm_type = norm_config.get("type", "none")
    norm_fields = norm_config.get("fields", {})

    entry_map: dict[str, CacheEntry] = {e.id: e for e in candidates}
    final_scores: dict[str, float] = {e.id: 0.0 for e in candidates}

    for field_name, weight, sim_cfg in active_fields:
        sim_type = sim_cfg.get("type", "cosine")

        for entry in candidates:
            if field_name not in entry.query_keys or field_name not in spec.query_keys:
                continue

            raw = self._compute_field_score(
                field_name,
                spec.query_keys[field_name],
                entry.query_keys[field_name],
                sim_cfg,
            )

            # Step 2: 转换为 [0, 1] 相似度
            if sim_type == "cosine":
                s = (raw + 1.0) / 2.0
            elif sim_type == "l2":
                to_sim = sim_cfg.get("to_similarity", {})
                tau = to_sim.get("tau", 1.0)
                s = math.exp(-raw / tau)
            else:
                s = raw

            # Step 3: percentile normalization
            if norm_type == "percentile" and field_name in norm_fields:
                p5 = norm_fields[field_name]["p5"]
                p95 = norm_fields[field_name]["p95"]
                denom = p95 - p5
                if denom > 0:
                    s = max(0.0, min(1.0, (s - p5) / denom))
                else:
                    s = 0.5  # p5 == p95 退化情况

            # Step 4: 加权累加
            final_scores[entry.id] += weight * s

    # Step 5: 排序取 top_k
    sorted_ids = sorted(final_scores, key=lambda eid: final_scores[eid], reverse=True)
    results = []
    for eid in sorted_ids[: spec.top_k]:
        entry = entry_map[eid]
        results.append(
            SearchResultLite(id=eid, score=final_scores[eid], checkpoint_id=entry.checkpoint_id)
        )
    return results
```

##### C1.9 重写 `search` 主方法

```python
def search(self, spec: QuerySpec) -> list[SearchResultLite]:
    self.search_call_count += 1
    if not self._entries:
        return []

    # 1. 过滤
    candidates = self._filter_entries(spec)
    if not candidates:
        return []

    # 2. 确定活跃字段
    active_fields = self._iter_active_fields(spec)
    if not active_fields:
        return []

    # 3. 根据 fusion_method 分发
    method = spec.fusion_method
    if method == "weighted_rrf":
        return self._search_weighted_rrf(candidates, spec, active_fields)
    elif method == "weighted_score_sum":
        return self._search_weighted_score_sum(candidates, spec, active_fields)
    elif method is None:
        # 向后兼容: 无 fusion_method 时使用旧的单字段 cosine 行为
        return self._search_single_field_cosine(candidates, spec)
    else:
        raise ValueError(f"Unknown fusion_method: {method}")


def _search_single_field_cosine(
    self, candidates: list[CacheEntry], spec: QuerySpec
) -> list[SearchResultLite]:
    """向后兼容: 原始单字段 cosine 搜索（现有测试依赖）。"""
    results: list[SearchResultLite] = []
    for entry in candidates:
        score = 0.0
        for field in spec.query_keys:
            if field in entry.query_keys:
                q = spec.query_keys[field].float()
                e = entry.query_keys[field].float()
                score = float(F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)))
                break
        results.append(
            SearchResultLite(id=entry.id, score=score, checkpoint_id=entry.checkpoint_id)
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[: spec.top_k]
```

新增 import：
```python
import math
import logging
logger = logging.getLogger(__name__)
```

---

### Phase D：批量生成 YAML

#### 生成范围对齐

本脚本第一版只生成 Phase 1 的 64 个 YAML（8 combos × 8 weights）。

Phase 1.5 和 Phase 2 的 YAML 不可能在 Phase 1 运行前生成，因为：
- Phase 1.5 的权重邻域取决于 Phase 1 每个组合的最佳权重
- Phase 2 的最佳组合取决于 Phase 1.5 的结果

Phase 1.5 YAML 由 `exp/analyze_cache_results.py` 输出 top 3 后，二次调用本脚本的 `--phase 1.5` 模式生成。Phase 2 同理。

总计 YAML 数：
- Phase 1：64 个（本脚本直接生成）
- Phase 1.5：~45 个（分析后二次生成）
- Phase 2：~3 个（分析后二次生成）

#### D1. `exp/generate_cache_run_yamls.py`

```python
"""批量生成 CP1 实验 YAML。

用法:
    # Phase 1: 直接生成 64 个
    uv run exp/generate_cache_run_yamls.py \
        --phase 1 \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
        --output-dir configs/cache_runs

    # Phase 1.5: 基于 Phase 1 分析结果二次生成
    uv run exp/generate_cache_run_yamls.py \
        --phase 1.5 \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
        --phase1-analysis configs/cache_runs/phase1/analysis.json \
        --output-dir configs/cache_runs

    # Phase 2: 基于 Phase 1.5 结果
    uv run exp/generate_cache_run_yamls.py \
        --phase 2 \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
        --phase1_5-analysis configs/cache_runs/phase1_5/analysis.json \
        --output-dir configs/cache_runs

输出:
    configs/cache_runs/phase1/phase1_run_001_a_rrf_w1.yaml  (64 个)
    configs/cache_runs/phase1_5/phase1_5_run_001_*.yaml     (~45 个, 二次生成)
    configs/cache_runs/phase2/phase2_run_001_*.yaml         (~3 个, 二次生成)
"""
```

##### D1.1 核心数据结构

```python
@dataclass
class ExperimentCombo:
    builder_type: str          # "cp1_mean_pool", ...
    builder_abbrev: str        # "a", "b1", "b2", "c"
    strategy_type: str         # "weighted_rrf_knn" | "weighted_score_sum_knn"
    strategy_abbrev: str       # "rrf" | "sum"
    vector_dims: dict[str, int]

COMBOS = [
    ExperimentCombo("cp1_mean_pool",       "a",  "weighted_rrf_knn",       "rrf", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_mean_pool",       "a",  "weighted_score_sum_knn", "sum", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_16", "b1", "weighted_rrf_knn",       "rrf", {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_16", "b1", "weighted_score_sum_knn", "sum", {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_64", "b2", "weighted_rrf_knn",       "rrf", {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_spatial_pool_64", "b2", "weighted_score_sum_knn", "sum", {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_max_pool",        "c",  "weighted_rrf_knn",       "rrf", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
    ExperimentCombo("cp1_max_pool",        "c",  "weighted_score_sum_knn", "sum", {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}),
]

WEIGHT_GRID_PHASE1 = [
    {"vision_0": 1.0,  "vision_1": 0.0,  "robot_state": 0.0 },  # W1
    {"vision_0": 0.75, "vision_1": 0.25, "robot_state": 0.0 },  # W2
    {"vision_0": 0.75, "vision_1": 0.0,  "robot_state": 0.25},  # W3
    {"vision_0": 0.5,  "vision_1": 0.25, "robot_state": 0.25},  # W4
    {"vision_0": 0.5,  "vision_1": 0.5,  "robot_state": 0.0 },  # W5
    {"vision_0": 0.5,  "vision_1": 0.0,  "robot_state": 0.5 },  # W6
    {"vision_0": 0.25, "vision_1": 0.5,  "robot_state": 0.25},  # W7
    {"vision_0": 0.25, "vision_1": 0.25, "robot_state": 0.5 },  # W8
]
```

##### D1.2 YAML 渲染逻辑

```python
def render_yaml(
    combo: ExperimentCombo,
    weights: dict[str, float],
    weight_id: str,
    artifact_dir: str,
    calibration: dict,  # builder_type -> field -> {"p5": float, "p95": float}
) -> str:
    """渲染单个 run 的完整 YAML 字符串。"""
    # prompt_emb weight 固定为 0.0
    full_weights = {**weights, "prompt_emb": 0.0}

    # artifact path
    preload_path = f"{artifact_dir}/{combo.builder_type}.pkl"

    # field_similarity 固定
    field_similarity = {
        "vision_0":    {"type": "cosine"},
        "vision_1":    {"type": "cosine"},
        "prompt_emb":  {"type": "cosine"},
        "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
    }

    # score_normalization（仅 SUM）
    score_normalization = None
    if combo.strategy_type == "weighted_score_sum_knn":
        cal = calibration[combo.builder_type]
        score_normalization = {
            "type": "percentile",
            "fields": cal,  # {"vision_0": {"p5": ..., "p95": ...}, ...}
        }

    # 用 PyYAML 或模板字符串渲染
    ...
```

##### D1.3 main 函数

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["1", "1.5", "2"])
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--output-dir", default="configs/cache_runs")
    # Phase 1.5/2 需要前一阶段分析结果
    parser.add_argument("--phase1-analysis", default=None, help="Phase 1 analysis JSON (for phase 1.5)")
    parser.add_argument("--phase1_5-analysis", default=None, help="Phase 1.5 analysis JSON (for phase 2)")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration_file).read_text())

    if args.phase == "1":
        _generate_phase1(args, calibration)
    elif args.phase == "1.5":
        if not args.phase1_analysis:
            parser.error("--phase1-analysis required for phase 1.5")
        analysis = json.loads(Path(args.phase1_analysis).read_text())
        _generate_phase1_5(args, calibration, analysis)
    elif args.phase == "2":
        if not args.phase1_5_analysis:
            parser.error("--phase1_5-analysis required for phase 2")
        analysis = json.loads(Path(args.phase1_5_analysis).read_text())
        _generate_phase2(args, calibration, analysis)


def _generate_phase1(args, calibration):
    """生成 Phase 1: 8 combos × 8 weights = 64 YAML。"""
    run_idx = 0
    for combo in COMBOS:
        for w_idx, weights in enumerate(WEIGHT_GRID_PHASE1):
            run_idx += 1
            filename = f"phase1_run_{run_idx:03d}_{combo.builder_abbrev}_{combo.strategy_abbrev}_w{w_idx+1}.yaml"
            yaml_str = render_yaml(combo, weights, f"w{w_idx+1}", args.artifact_dir, calibration)
            out_path = Path(args.output_dir) / "phase1" / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_str)
    print(f"Phase 1: generated {run_idx} YAML files")


def _generate_phase1_5(args, calibration, phase1_analysis):
    """生成 Phase 1.5: top 3 组合 × ~15 权重邻域 = ~45 YAML。

    从 phase1_analysis["top3"] 读取最佳组合及其最佳权重，
    以最佳权重为中心，±0.2 范围内 step=0.1 采样。
    """
    top3 = phase1_analysis["top3"]  # list of {"combo": ..., "best_weights": ..., ...}
    run_idx = 0
    for entry in top3:
        combo = _find_combo(entry["combo"])
        center = entry["best_weights"]
        fine_weights = _generate_fine_grid(center, step=0.1, radius=0.2)
        for w_idx, weights in enumerate(fine_weights):
            run_idx += 1
            filename = f"phase1_5_run_{run_idx:03d}_{combo.builder_abbrev}_{combo.strategy_abbrev}_f{w_idx+1}.yaml"
            yaml_str = render_yaml(combo, weights, f"f{w_idx+1}", args.artifact_dir, calibration)
            out_path = Path(args.output_dir) / "phase1_5" / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_str)
    print(f"Phase 1.5: generated {run_idx} YAML files")


def _generate_phase2(args, calibration, phase1_5_analysis):
    """生成 Phase 2: 最佳组合+权重，加 prompt_emb=0.1 对照，~3 YAML。"""
    best = phase1_5_analysis["best"]
    run_idx = 0
    for prompt_w in [0.0, 0.1, 0.2]:
        run_idx += 1
        combo = _find_combo(best["combo"])
        weights = {**best["best_weights"], "prompt_emb": prompt_w}
        # 重新归一化（使权重之和为 1）
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        filename = f"phase2_run_{run_idx:03d}_prompt_{prompt_w:.1f}.yaml"
        yaml_str = render_yaml(combo, weights, f"p{prompt_w}", args.artifact_dir, calibration)
        out_path = Path(args.output_dir) / "phase2" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_str)
    print(f"Phase 2: generated {run_idx} YAML files")
```

---

### Phase 0：数据前置检查与校准

#### 0.1 `exp/build_in_memory_cache_artifact.py`

```python
"""离线构建 InMemoryBackend artifact。

用法:
    uv run exp/build_in_memory_cache_artifact.py \
        --data-dir data/libero_spatial \
        --builder-type cp1_mean_pool \
        --output data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl

输入: HDF5 episode 文件（vision_0/1/2, prompt_emb, robot_state 字段）
输出: pickle 文件，可被 InMemoryBackend.load_artifact() 加载
"""

import argparse
import pickle
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)
```

##### 核心流程

```python
def build_artifact(
    data_dir: str,
    builder_type: str,
    checkpoint_id_str: str = "CP1",
) -> dict:
    """构建 artifact dict。

    流程:
      1. 扫描 data_dir 下所有 .h5 文件
      2. 对每个 step，读取原始 tensor，模拟 stage1 输出格式
      3. 调用对应 KeyBuilder 的 collect() + build()
      4. 构造 CacheEntry（含 placeholder action_chunk）
      5. 收集所有 entries，附加 metadata
    """
    from openpi.cache.types import CheckpointID
    from openpi.cache.storage_types import CacheEntry, CachePayload

    cp_id = CheckpointID[checkpoint_id_str]
    builder = _create_builder(builder_type)
    vector_dims = _get_vector_dims(builder_type)

    h5_paths = sorted(Path(data_dir).rglob("*.h5"))
    entries: list[CacheEntry] = []

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            task = f.attrs.get("task", "")
            success = bool(f.attrs.get("success", False))
            if not success:
                continue  # 只使用成功 episode

            step_names = sorted(k for k in f.keys() if k.startswith("step_"))
            for step_name in step_names:
                group = f[step_name]

                # 读取原始 tensor 并模拟 stage1 输出
                fake_stage1 = _build_fake_stage1(group)

                builder.collect(cp_id, stage1=fake_stage1)
                query_keys = builder.build(cp_id)
                builder.clear()

                # 构造 CacheEntry
                # id 使用文件名+step名（确定性、可追溯），不使用 stable-hash
                # 见"契约修订"第 3 条
                entry_id = f"{h5_path.stem}_{step_name}"
                action = torch.from_numpy(np.array(group["clean_action"])).float()
                # clean_action shape: [action_horizon, action_dim]
                # 保留数据真实 horizon，不 pad 到 [50, 32]
                # 见"契约修订"第 4 条
                if action.dim() == 1:
                    action = action.unsqueeze(0)  # [1, action_dim]

                payload = CachePayload(
                    action_chunk=action,
                    task_key=str(task),
                )
                entry = CacheEntry(
                    id=entry_id,
                    checkpoint_id=cp_id,
                    query_keys=query_keys,
                    payload=payload,
                )
                entries.append(entry)

    logger.info("Built %d entries for %s", len(entries), builder_type)
    return {
        "key_builder_type": builder_type,
        "checkpoint_id": checkpoint_id_str,
        "vector_dims": vector_dims,
        "entries": entries,
    }
```

##### `_build_fake_stage1` helper

```python
class _FakeStage1:
    """模拟 Stage1Output 结构，让 KeyBuilder.collect() 可以工作。"""
    def __init__(self, prefix_embs: torch.Tensor, state: torch.Tensor):
        self.prefix_embs = prefix_embs  # [1, prefix_len, emb_dim]
        self.state = state               # [1, state_dim]

def _build_fake_stage1(group: h5py.Group) -> _FakeStage1:
    """从 HDF5 step group 构造 fake stage1 output。

    HDF5 字段:
      vision_0: [256, 2048]  (已有 SigLIP embedding)
      vision_1: [256, 2048]
      vision_2: [256, 2048]  (可能不存在)
      prompt_emb: [num_tokens, 2048]
      robot_state: [32]

    需要重建 prefix_embs = concat([vision_0, vision_1, vision_2, prompt_emb], dim=0)
    然后加 batch dim。
    """
    parts = []
    for vfield in ("vision_0", "vision_1", "vision_2"):
        if vfield in group:
            parts.append(torch.from_numpy(np.array(group[vfield])).float())
        else:
            # vision_2 可能不存在，用零填充
            emb_dim = parts[0].shape[1] if parts else 2048
            parts.append(torch.zeros(256, emb_dim))

    prompt = torch.from_numpy(np.array(group["prompt_emb"])).float()
    parts.append(prompt)

    prefix_embs = torch.cat(parts, dim=0).unsqueeze(0)  # [1, prefix_len, emb_dim]
    state = torch.from_numpy(np.array(group["robot_state"])).float().unsqueeze(0)  # [1, state_dim]

    return _FakeStage1(prefix_embs, state)
```

##### `_create_builder` 和 `_get_vector_dims` helpers

```python
def _create_builder(builder_type: str):
    from openpi.cache.components.key_builder import (
        CP1MeanPoolKeyBuilder,
        CP1SpatialPool16KeyBuilder,
        CP1SpatialPool64KeyBuilder,
        CP1MaxPoolKeyBuilder,
    )
    builders = {
        "cp1_mean_pool": CP1MeanPoolKeyBuilder,
        "cp1_spatial_pool_16": CP1SpatialPool16KeyBuilder,
        "cp1_spatial_pool_64": CP1SpatialPool64KeyBuilder,
        "cp1_max_pool": CP1MaxPoolKeyBuilder,
    }
    return builders[builder_type]()

_VECTOR_DIMS = {
    "cp1_mean_pool":       {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
    "cp1_spatial_pool_16": {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32},
    "cp1_spatial_pool_64": {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32},
    "cp1_max_pool":        {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
}

def _get_vector_dims(builder_type: str) -> dict[str, int]:
    return _VECTOR_DIMS[builder_type]
```

##### main

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--builder-type", required=True,
                        choices=list(_VECTOR_DIMS.keys()))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = build_artifact(args.data_dir, args.builder_type)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved {len(artifact['entries'])} entries to {args.output}")

if __name__ == "__main__":
    main()
```

#### 0.2 `exp/calibrate_score_sum_stats.py`

```python
"""统计每个 builder/field 的 p5/p95 和离线 sanity check。

用法:
    uv run exp/calibrate_score_sum_stats.py \
        --artifact-dir data/cache_artifacts/libero_spatial \
        --output data/cache_artifacts/libero_spatial/calibration.json

输出 JSON 格式:
    {
      "cp1_mean_pool": {
        "vision_0":    {"p5": 0.82, "p95": 0.99, "mean": 0.91, "std": 0.04},
        "vision_1":    {"p5": ..., "p95": ...},
        "prompt_emb":  {"p5": ..., "p95": ...},
        "robot_state": {"p5": ..., "p95": ...}
      },
      ...
    }
"""
```

##### 核心算法

```python
def compute_field_stats(
    entries: list[CacheEntry],
    field_name: str,
    sim_type: str,
    tau: float = 0.334717,
    num_pairs: int = 50000,
) -> dict[str, float]:
    """对一个字段采样 pair，计算相似度分布的 p5/p95。

    采样策略:
      - 随机采样 num_pairs 对 (i, j) 其中 i != j
      - 计算 similarity score (已经过方向统一):
        - cosine: (cos + 1) / 2
        - l2: exp(-d / tau)
      - 返回 p5, p95, mean, std
    """
    import random
    n = len(entries)
    scores = []
    for _ in range(num_pairs):
        i, j = random.sample(range(n), 2)
        qi = entries[i].query_keys[field_name]
        ej = entries[j].query_keys[field_name]

        if sim_type == "cosine":
            cos = float(F.cosine_similarity(qi.unsqueeze(0), ej.unsqueeze(0)))
            s = (cos + 1.0) / 2.0
        elif sim_type == "l2":
            d = float(torch.norm(qi.float() - ej.float(), p=2))
            s = math.exp(-d / tau)
        scores.append(s)

    scores_np = np.array(scores)
    return {
        "p5":   float(np.percentile(scores_np, 5)),
        "p95":  float(np.percentile(scores_np, 95)),
        "mean": float(np.mean(scores_np)),
        "std":  float(np.std(scores_np)),
    }
```

##### Sanity check 输出

对每个 builder 额外输出 sanity check：
```python
def sanity_check(
    entries: list[CacheEntry],
    field_name: str,
) -> dict[str, float]:
    """same-task vs cross-task cosine 分布对比。
    如果两者分布重叠严重（AUC < 0.55），标记为 WARNING。
    """
    ...
```

输出到 console 和 JSON 中的 `_sanity` 键。

---

### Phase E：实验控制与结果汇总

#### E1. `exp/run_cache_experiments.py`

```python
"""实验控制脚本。

用法:
    # 跑全部 Phase 1（4 worker 并发）
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 10 \
        --num-workers 4 \
        --host localhost --port 8000

    # 只跑指定 run
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --runs 1-8 \
        --episodes-per-run 10 \
        --num-workers 1 \
        --host localhost --port 8000

    # 断点重续
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 10 \
        --num-workers 4 \
        --host localhost --port 8000 \
        --resume
"""
```

##### 状态文件格式

```python
@dataclass
class RunState:
    yaml_path: str
    run_id: str
    status: str              # "pending" | "running" | "done" | "failed"
    episodes_total: int
    episodes_done: int
    start_time: Optional[str]
    end_time: Optional[str]
    success_rate: Optional[float]
    exit_code: Optional[int]
```

状态文件: `configs/cache_runs/phase1/experiment_state.json`

##### 单 run 执行流程

```python
def execute_run(
    yaml_path: str,
    episodes_per_run: int,
    num_workers: int,
    host: str,
    port: int,
) -> RunResult:
    """执行单个 run。

    流程:
      1. load_cache_config(yaml) — 通过 WebSocket 切换 YAML，等待 ack
      2. 启动 examples/libero/main.py --num_workers N --host H --port P
      3. 等所有 worker 完成
      4. 收集结果
      5. 再切下一个 run

    注意: examples/libero/main.py 接受 --host 和 --port（不是 --server_url），
    见 examples/libero/main.py:34-35。控制消息走 WebSocket 需要自行拼 URL。
    """
    # Step 1: 通知 server 切换 YAML
    server_url = f"ws://{host}:{port}"
    _send_cache_config(server_url, yaml_path)

    # Step 2: 调用 LIBERO（N 个 worker 各自建连接，共享同一个 bundle）
    cmd = [
        "uv", "run", "examples/libero/main.py",
        "--host", host,
        "--port", str(port),
        "--task_suite_name", "libero_spatial",
        "--num_trials_per_task", str(episodes_per_run),
        "--num_workers", str(num_workers),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    ...
```

##### `_send_cache_config` helper

```python
import asyncio
import msgpack
import websockets

def _send_cache_config(server_url: str, yaml_path: str) -> None:
    """通过 WebSocket 发送 load_cache_config 控制消息。"""
    async def _send():
        async with websockets.connect(server_url) as ws:
            # 读取 metadata
            raw = await ws.recv()
            # 发送控制消息
            msg = {"__ctrl__": "load_cache_config", "yaml_path": yaml_path}
            await ws.send(msgpack.packb(msg))
            # 等待 ack
            resp = msgpack.unpackb(await ws.recv())
            if resp.get("__ack__") != "load_cache_config":
                raise RuntimeError(f"Unexpected ack: {resp}")
    asyncio.run(_send())
```

注意：这个控制连接和实际 LIBERO 评测连接是分开的。控制连接只做切换 YAML，切完立即断开。LIBERO 评测连接在 `examples/libero/main.py` 内部建立。

#### E2. `exp/analyze_cache_results.py`

```python
"""汇总实验结果，选出 top 3。

用法:
    uv run exp/analyze_cache_results.py \
        --state-file configs/cache_runs/phase1/experiment_state.json \
        --output configs/cache_runs/phase1/analysis.json
"""

def main():
    # 读取 state file
    # 按 success_rate 降序排序
    # 输出排序表 (markdown 格式到 stdout)
    # 输出 JSON (供 Phase 1.5 生成脚本使用)
    ...
```

---

### Phase F：Server 动态切换 YAML

#### 总体约束

- **支持单个 run 在固定 YAML 下使用多 worker 并发评测**
- **不支持多个不同 run 同时共享一个 server 并发执行**
- run 与 run 串行；单个 run 内允许 `num_workers > 1`

#### 运行模式

实验 server **统一用 `--concurrent` 启动**，即使 `num_workers=1` 也走 concurrent 路径。原因：只有 concurrent 模式下 server 才会为每个新连接调用 `connection_policy_factory`，这是动态切 YAML 的唯一入口。

#### 核心设计：CurrentCacheBundle

Server 维护一个全局 `CurrentCacheBundle`，在 `load_cache_config` 控制消息到达时原子替换。同一 run 内的多个 worker 连接共享同一个 bundle 的 `shared_storage`。

```python
@dataclass
class CurrentCacheBundle:
    """当前 run 的 cache 配置快照。

    load_cache_config 控制消息到达时原子替换整个 bundle。
    同一 run 内的多个 worker 连接共享同一个 shared_storage。
    不同 run 切 YAML 时构建新的 bundle（新的 shared_storage）。
    已在跑的旧连接继续用旧 bundle 的 shared_storage，不受影响。
    """
    config_path: str
    cache_config: CacheConfig           # 已校验的 config
    shared_storage: CacheStorage        # 从 build_shared_storage() 构建
    version: int                        # 单调递增，用于日志追踪
```

#### F1. `src/openpi/serving/websocket_policy_server.py` 修改

模块级新增全局状态：

```python
import threading
from dataclasses import dataclass
from typing import Optional

@dataclass
class CurrentCacheBundle:
    config_path: str
    cache_config: object        # CacheConfig
    shared_storage: object      # CacheStorage
    version: int

_bundle_lock = threading.Lock()
_current_bundle: Optional[CurrentCacheBundle] = None
_bundle_version: int = 0
```

在现有 `__ctrl__` 处理块中，`episode_end` 分支之后新增：

```python
elif ctrl == "load_cache_config":
    yaml_path = obs.get("yaml_path", "")
    if not yaml_path:
        await websocket.send(packer.pack({"__ack__": "error", "msg": "missing yaml_path"}))
        continue
    try:
        from openpi.cache.config import load_cache_config, build_shared_storage
        # 校验 YAML 并构建新的 shared_storage
        cache_config = load_cache_config(yaml_path)
        shared_storage = build_shared_storage(cache_config)
        # 原子替换整个 bundle
        with _bundle_lock:
            global _current_bundle, _bundle_version
            _bundle_version += 1
            _current_bundle = CurrentCacheBundle(
                config_path=yaml_path,
                cache_config=cache_config,
                shared_storage=shared_storage,
                version=_bundle_version,
            )
        logger.info("Cache bundle updated to v%d: %s", _bundle_version, yaml_path)
        await websocket.send(packer.pack({
            "__ack__": "load_cache_config",
            "yaml_path": yaml_path,
            "version": _bundle_version,
        }))
    except Exception as e:
        logger.error("Failed to load cache config %s: %s", yaml_path, e)
        await websocket.send(packer.pack({"__ack__": "error", "msg": str(e)}))
    continue
```

新增模块级函数供 `serve_policy.py` 调用：

```python
def get_current_cache_bundle() -> Optional[CurrentCacheBundle]:
    """返回当前 cache bundle 快照（如有）。线程安全。"""
    with _bundle_lock:
        return _current_bundle
```

注意：

- 控制连接只做"校验 + 构建 shared_storage + 写全局 bundle"
- 控制连接发完 ack 后客户端即断开
- 副作用：当前 server 在连接建立时会创建 conn_policy 并触发 `on_task_begin()`，控制短连接会产生一次空 task 生命周期。这是可接受的——timer 可能多记一条空记录，不影响功能正确性

#### F2. `scripts/serve_policy.py` 修改

现有代码结构：
- `_wrap_policy(base_policy, args, *, quiet, eager, shared_cache)` 负责构建 wrapper 链
- `main()` 中 concurrent 分支预建 `shared_cache`，然后定义 `_connection_policy_factory(shared_base_policy)` 调用 `_wrap_policy`

改动点：

1. `_wrap_policy` 新增全局 bundle 检查，**在 `args.cache_config` 判断之前**：

```python
def _wrap_policy(base_policy, args, *, quiet=False, eager=False, shared_cache=None):
    policy = base_policy

    # --- 新增：全局 bundle 优先级最高（动态切 YAML 场景） ---
    # 必须放在 args.cache_config 分支之前，
    # 否则 server 不带 --cache_config 启动时 bundle 路径不会生效。
    from openpi.serving.websocket_policy_server import get_current_cache_bundle
    bundle = get_current_cache_bundle()
    if bundle is not None:
        from openpi.cache.config import build_per_connection_components
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        components = build_per_connection_components(
            bundle.cache_config,
            bundle.shared_storage,
            quiet=True,
        )
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
        )
        policy = InferenceInterceptor(
            policy, timer=components["timer"],
            orchestrator=orchestrator, eager=eager,
        )
    elif args.cache_config is not None:
        # --- 原有逻辑（启动时 --cache_config，无动态切换） ---
        from openpi.cache.config import (
            build_cache_components,
            build_per_connection_components,
            load_cache_config,
        )
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        if shared_cache is not None:
            cache_config = load_cache_config(args.cache_config)
            components = build_per_connection_components(
                cache_config, shared_cache["storage"], quiet=True,
            )
        else:
            cache_config = load_cache_config(args.cache_config)
            components = build_cache_components(cache_config)
            if quiet:
                components["timer"]._quiet = True

        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
        )
        policy = InferenceInterceptor(
            policy, timer=components["timer"],
            orchestrator=orchestrator, eager=eager,
        )
    elif args.cache:
        # --- 原有逻辑（--cache 无 YAML） ---
        ...

    # 后续 record / collect wrapper 不变
    ...
    return policy
```

这样即使 server 不带 `--cache_config` 启动，后续 `load_cache_config` 控制消息写入 bundle 后，新连接也能正确走 bundle 路径。

2. `main()` 中 concurrent 分支保持不变：

```python
# main() 中，concurrent 模式启动时仍按现有方式预建 shared_cache
# 如果 --cache_config 指定了初始 YAML，就预建 shared_storage
# 如果不指定 --cache_config，shared_cache=None，首个 run 由控制脚本通过
# load_cache_config 注入 bundle，之后的新连接从 bundle 读取
if args.concurrent:
    shared_cache = None
    if args.cache_config is not None:
        from openpi.cache.config import build_shared_storage, load_cache_config
        cache_config = load_cache_config(args.cache_config)
        shared_cache = {"storage": build_shared_storage(cache_config)}

    def _connection_policy_factory(shared_base_policy):
        # _wrap_policy 内部会优先检查全局 bundle
        return _wrap_policy(
            shared_base_policy, args, quiet=True, eager=True,
            shared_cache=shared_cache,
        )
    ...
```

关键点：`_wrap_policy` 内部 `get_current_cache_bundle()` 的优先级高于 `shared_cache` 参数。当控制脚本发了 `load_cache_config` 后，后续新连接会走 bundle 路径；在此之前的连接走原有 `shared_cache` 路径。

Storage 共享语义：

- 同一个 run 的多个 worker 连接共享同一个 `bundle.shared_storage`（和现有 concurrent 模式语义一致）
- 不同 run 切 YAML 时，`load_cache_config` 构建新的 `shared_storage`，原子替换整个 bundle
- 已在跑的旧连接继续用旧 bundle 的 `shared_storage`，不受新 bundle 影响（Python 引用计数保证旧对象不会被回收）

#### F3. 控制脚本侧（`exp/run_cache_experiments.py` 内）

```python
def _send_cache_config(server_url: str, yaml_path: str) -> None:
    """通过专用短连接发送 load_cache_config 控制消息。"""
    import asyncio
    import msgpack
    import websockets

    async def _send():
        async with websockets.connect(server_url) as ws:
            _metadata = await ws.recv()  # 读取 server metadata
            msg = {"__ctrl__": "load_cache_config", "yaml_path": str(Path(yaml_path).resolve())}
            await ws.send(msgpack.packb(msg))
            resp = msgpack.unpackb(await ws.recv())
            if resp.get("__ack__") != "load_cache_config":
                raise RuntimeError(f"Config switch failed: {resp}")
            logger.info("Server switched to bundle v%s: %s", resp.get("version"), yaml_path)
    asyncio.run(_send())
```

单 run 流程：
1. `_send_cache_config(f"ws://{host}:{port}", yaml_path)` — 切换 YAML，等待 ack
2. 启动 `examples/libero/main.py --host H --port P --num_workers N` — N 个 worker 各自建连接，读取同一个 bundle
3. 等所有 worker 完成
4. 再切下一个 run

不修改：

- `packages/openpi-client/` — 控制脚本直接用 `websockets` + `msgpack`
- `examples/libero/main.py` — 不改，每个 run 自然开新连接

---

### 测试规格

#### T1. `tests/cache/components/test_key_builder_cp1_experiment.py`

```python
"""CP1 实验版 key builder 测试。"""

class TestCP1MeanPoolKeyBuilder:
    def test_output_shapes(self):
        """验证 vision_0=[2048], prompt_emb=[2048], robot_state=[32]"""
        builder = CP1MeanPoolKeyBuilder()
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert keys["vision_0"].shape == (2048,)
        assert keys["prompt_emb"].shape == (2048,)
        assert keys["robot_state"].shape == (32,)

    def test_cp3_same_as_cp1(self):
        """当前 CP3 与 CP1 行为一致。"""
        ...

    def test_vision_2_included_when_enabled(self):
        """enabled_fields 包含 vision_2 时输出应包含 vision_2。"""
        ...

    def test_vision_2_excluded_when_disabled(self):
        """enabled_fields 不包含 vision_2 时输出不应包含 vision_2。"""
        ...

    def test_action_chunk_cached(self):
        """collect(stage3=...) 时 cached_data 包含 action_chunk。"""
        ...

class TestCP1SpatialPool16KeyBuilder:
    def test_output_shapes(self):
        """vision_0=[32768], prompt_emb=[2048], robot_state=[32]"""
        ...

    def test_spatial_pool_correctness(self):
        """手算 4×4 pool 结果，验证一致性。"""
        # 构造 256 tokens 其中 4×4 块内全相同，pool 后应等于块值
        ...

class TestCP1SpatialPool64KeyBuilder:
    def test_output_shapes(self):
        """vision_0=[8192], prompt_emb=[2048], robot_state=[32]"""
        ...

class TestCP1MaxPoolKeyBuilder:
    def test_output_shapes(self):
        """vision_0=[2048], prompt_emb=[2048], robot_state=[32]"""
        ...

    def test_max_pool_picks_maximum(self):
        """验证确实取了最大值。"""
        ...
```

##### 测试 fixture

```python
def _make_fake_stage1(
    emb_dim: int = 2048,
    state_dim: int = 32,
    num_prompt_tokens: int = 20,
) -> _FakeStage1:
    """构造测试用的 fake Stage1Output。"""
    prefix_len = 256 * 3 + num_prompt_tokens  # 3 images + prompt
    prefix_embs = torch.randn(1, prefix_len, emb_dim)
    state = torch.randn(1, state_dim)
    return _FakeStage1(prefix_embs, state)

class _FakeStage1:
    def __init__(self, prefix_embs, state):
        self.prefix_embs = prefix_embs
        self.state = state

class _FakeStage3:
    def __init__(self, action_chunk):
        self.action_chunk = action_chunk
```

#### T2. `tests/cache/test_in_memory_backend_experiment.py`

```python
"""InMemoryBackend 多字段检索和融合测试。"""

class TestWeightedRrf:
    def test_rrf_basic_ranking(self):
        """两个字段一致排序时 RRF 结果应一致。"""
        ...

    def test_rrf_conflicting_fields(self):
        """两个字段排序相反时，高权重字段主导。"""
        ...

    def test_zero_weight_field_skipped(self):
        """weight=0 的字段不参与 RRF 计算。"""
        ...

class TestWeightedScoreSum:
    def test_sum_basic(self):
        """手算 percentile norm + weighted sum，验证排序正确。"""
        ...

    def test_tau_effect(self):
        """验证 robot_state 的 exp(-d/tau) 转换正确。"""
        ...

    def test_percentile_normalization(self):
        """验证 p5/p95 normalization 的 clip 行为。"""
        ...

class TestFiltering:
    def test_task_key_filter(self):
        """task_key 过滤只返回匹配的 entries。"""
        ...

    def test_checkpoint_id_filter(self):
        """checkpoint_id 过滤正确。"""
        ...

class TestBackwardCompat:
    def test_no_fusion_method_uses_single_field(self):
        """fusion_method=None 时使用旧的单字段 cosine（不破坏现有测试）。"""
        ...

class TestArtifactLoading:
    def test_load_artifact(self, tmp_path):
        """测试 load_artifact 加载和 vector_dims 校验。"""
        ...

    def test_load_artifact_dims_mismatch(self, tmp_path):
        """vector_dims 不匹配时应 raise ValueError。"""
        ...
```

#### T3. `tests/cache/test_config_experiment.py`

```python
"""实验配置 schema 和工厂测试。"""

class TestConfigParsing:
    def test_full_experiment_yaml(self, tmp_path):
        """完整实验 YAML 能正确解析。"""
        yaml_content = """
enabled: true
key_builder:
  type: cp1_mean_pool
keys:
  vision_0: {enabled: true, weight: 0.75}
  vision_1: {enabled: true, weight: 0.25}
  vision_2: {enabled: false, weight: 0.0}
  prompt_emb: {enabled: true, weight: 0.0}
  robot_state: {enabled: true, weight: 0.25}
checkpoints:
  cp1:
    enabled: true
    gate: {type: always_search}
    judge: {type: always_hit}
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      rrf_k: 60
      field_similarity:
        vision_0: {type: cosine}
        vision_1: {type: cosine}
        prompt_emb: {type: cosine}
        robot_state:
          type: l2
          to_similarity: {type: exp, tau: 0.334717}
backend:
  type: in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    prompt_emb: 2048
    robot_state: 32
"""
        ...

class TestConfigValidation:
    def test_score_sum_requires_normalization(self):
        """weighted_score_sum_knn 缺少 score_normalization 时报错。"""
        ...

    def test_cp1_builder_requires_vision_0(self):
        """cp1_* builder 需要 vision_0 enabled。"""
        ...

class TestConfigFactory:
    def test_build_mean_pool_builder(self):
        """config 工厂能正确实例化 CP1MeanPoolKeyBuilder。"""
        ...

    def test_build_weighted_rrf_strategy(self):
        """config 工厂能正确实例化 WeightedRrfKnnStrategy。"""
        ...
```

---

### 文件修改清单总览

| Phase | 文件 | 操作 | 说明 |
|-------|------|------|------|
| A | `src/openpi/cache/storage_types.py` | 修改 | QuerySpec 新增 3 字段 + 更新 query_keys docstring |
| A | `src/openpi/cache/config.py` | 修改 | 新增 dataclass, 扩展校验和工厂 |
| A | `src/openpi/cache/components/search_strategy.py` | 修改 | 新增 2 个 Strategy + 共享 filter |
| B | `src/openpi/cache/components/key_builder.py` | 修改 | 新增 4 builder + 基类 + helpers |
| C | `src/openpi/cache/backends/in_memory_backend.py` | 修改 | 完全重写搜索逻辑 |
| D | `exp/generate_cache_run_yamls.py` | **新增** | Phase 1 生成 64 个 YAML；Phase 1.5/2 二次生成 |
| 0 | `exp/build_in_memory_cache_artifact.py` | **新增** | 离线构建 artifact |
| 0 | `exp/calibrate_score_sum_stats.py` | **新增** | 统计 p5/p95 |
| E | `exp/run_cache_experiments.py` | **新增** | 实验控制脚本 |
| E | `exp/analyze_cache_results.py` | **新增** | 结果汇总 |
| F | `src/openpi/serving/websocket_policy_server.py` | 修改 | 新增 `load_cache_config` 控制消息 + 全局变量 |
| F | `scripts/serve_policy.py` | 修改 | `connection_policy_factory` 读取全局 cache config |
| T | `tests/cache/components/test_key_builder_cp1_experiment.py` | **新增** | builder 测试 |
| T | `tests/cache/test_in_memory_backend_experiment.py` | **新增** | backend 融合测试 |
| T | `tests/cache/test_config_experiment.py` | **新增** | config 测试 |
| T | `tests/cache/test_search_strategy_experiment.py` | **新增** | strategy 装配测试 |

不修改的文件：
- `src/openpi/cache/orchestrator.py` — 不变
- `src/openpi/cache/interceptor.py` — 不变
- `src/openpi/cache/cache_storage.py` — 不变
- `src/openpi/cache/backend_base.py` — 不变
- `src/openpi/cache/types.py` — 不变
- `examples/libero/main.py` — 不变，每个 run 自然开新连接
- `packages/openpi-client/` — 不变，控制脚本直接用 websockets+msgpack
