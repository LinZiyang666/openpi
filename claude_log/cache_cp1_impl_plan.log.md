# Cache CP1 In-Memory 实现计划

> Status: Plan
> Date: 2026-04-06

---

## 目标

基于 [cache_system_tutorial.md](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/docs/cache_system_tutorial.md) 和 [cache_experiment_plan.log.md](/mnt/c/Users/lzy66/OneDrive%20-%20University%20of%20Illinois%20-%20Urbana/ai-gaming/openpi/claude_log/cache_experiment_plan.log.md)，为当前大规模实验补齐 `CP1` 路径所需实现。

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
  - `search_strategy_type`
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
- 按 `QuerySpec` 执行过滤
- 为每个字段计算字段级分数
- 执行跨模态融合
- 排序并返回 `SearchResultLite`

不负责：

- 降维
- 命中判定
- 阈值逻辑

建议内部再拆三层：

- `field_similarity.py`
  - `cosine_similarity`
  - `l2_distance`
- `score_normalization.py`
  - `cosine_to_unit_interval`
  - `l2_to_similarity_exp`
  - `percentile_normalize`
- `fusion.py`
  - `weighted_rrf`
  - `weighted_score_sum`

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

- `search_strategy_type: str | None`
- `field_similarity: dict[str, dict[str, Any]] | None`
- `score_normalization: dict[str, dict[str, Any]] | None`
- `rrf_k: int | None`

用途：

- `search_strategy_type`
  - `weighted_rrf_knn`
  - `weighted_score_sum_knn`
- `field_similarity`
  - `vision_0 / vision_1 / prompt_emb = cosine`
  - `robot_state = l2`
- `score_normalization`
  - cosine 的 `[0,1]` 映射
  - `robot_state` 的 `exp(-d/tau)`
  - percentile normalization 参数
- `rrf_k`
  - 仅 `weighted_rrf_knn` 使用

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

关键要求：

- backend 只吃已经 build 好的 query vectors
- backend 不碰 stage1 / token layout
- backend 不参与 hit/miss 判断

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

按当前实验计划：

- `112 runs = 112 个独立 YAML`

建议目录：

- `configs/cache_runs/phase1/`
- `configs/cache_runs/phase1_5/`
- `configs/cache_runs/phase2/`

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

LIBERO 本轮实验的 YAML 统一约束：

- `keys.vision_2.enabled = false`
- 模块虽然支持 `vision_2`，但本轮 run 不启用
- Phase 1 / 1.5 中 `prompt_emb.weight = 0.0`

### G. 测试计划

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
     - `search_strategy_type`
     - `field_similarity`
     - `score_normalization`
   - 两个 strategy type 的装配路径分别正确

3. in_memory backend
   - cosine + L2 多字段打分正确
   - `weighted_rrf` 排序正确
   - `weighted_score_sum` 排序正确
   - `tau = 0.334717` 路径正确
   - 过滤逻辑正确

4. config
   - YAML 能构建对应 builder / strategy / backend
   - 缺少 `tau` 或 `p5/p95` 时在 `weighted_score_sum` 下报错

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

---

## 推荐落地顺序

按教程骨架最小改动落地，顺序应为：

1. 先改 `storage_types.py`
2. 再改 `config.py`
3. 再改 `components/search_strategy.py`
4. 再改 `components/key_builder.py`
5. 最后重写 `backends/in_memory_backend.py`
6. 补 `tests/cache/...`
7. 写实验 YAML

理由：

- 先把 `QuerySpec` 和 config schema 立住
- 再接 strategy 和 builder
- backend 最后按确定好的 schema 实现，返工最少

---

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
- `field_similarity`
  - `vision_0: cosine`
  - `vision_1: cosine`
  - `prompt_emb: cosine`
  - `robot_state: l2`
- `score_normalization`
  - cosine 映射到 `[0,1]`
  - `robot_state` 距离转相似度
  - percentile normalization

### `backend`

需要支持：

- `type: in_memory`
- `in_memory` 专属配置块
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
          vision_0:   { p5: 0.0, p95: 1.0 }
          vision_1:   { p5: 0.0, p95: 1.0 }
          prompt_emb: { p5: 0.0, p95: 1.0 }
          robot_state:{ p5: 0.0, p95: 1.0 }

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    vision_2: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    index_type: brute_force
```

说明：

- Phase 1 / 1.5 时，`prompt_emb.weight = 0.0`
- 每个 run 都有自己的独立 YAML
- 模块层保留 `vision_2`
- LIBERO 本轮 YAML 中 `vision_2.enabled = false`
- Phase 2 如需验证 `prompt_emb`，只改权重，不改模块边界
- 同一 schema 应能表达 `weighted_rrf` 和 `weighted_score_sum`

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

新增：

- `components/reducers.py`

修改：

- `components/key_builder.py`

目标：

- 支持 A / B1 / B2 / C 四类降维
- 输出维度和 `backend.vector_dims` 对齐

### Phase C：重写 `in_memory` 后端

新增：

- `backends/field_similarity.py`
- `backends/score_normalization.py`
- `backends/fusion.py`

修改：

- `backends/in_memory_backend.py`

目标：

- `in_memory` 成为实验可用后端
- 所有 Layer 1 / Layer 2 逻辑落到 backend 内部，但拆成清晰子模块

### Phase D：补测试

新增测试重点：

- reducer 输出维度正确
- cosine / L2 字段打分正确
- percentile normalization 正确
- `weighted_rrf` 排序正确
- `weighted_score_sum` 排序正确
- `prompt_emb.weight = 0` 时不影响最终分数
- `task_key` / `step_range` 过滤正确
- `cp1 + always_search + always_hit + in_memory` 端到端可跑

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
   - reducer 只做降维
   - strategy 只组装 query
   - backend 只做检索
   - judge 只放行 top-1
5. 后续扩展 Qdrant 时，可以复用：
   - reducer
   - config schema
   - search strategy
   - calibration 配置

---

## 下一步

下一步按以下顺序推进：

1. 先扩 `QuerySpec` 和 YAML schema
2. 再补 `CP1ReducedKeyBuilder + reducers`
3. 再重写 `in_memory` 后端
4. 最后补测试和实验 YAML
