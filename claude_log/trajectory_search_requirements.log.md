# Trajectory Search 需求文档

**Status**: `Plan`
**Date**: 2026-04-06

---

## 1. 动机

现有的 cache 搜索是 **单步匹配 (single-step matching)**：每次推理时构建当前 step 的 query key，与数据库中每个独立 entry 做相似度比较，返回最相似的 entry。这种方式丢失了 **时序上下文**——两个不同 episode 可能在某一步的观测非常相似，但它们的历史轨迹完全不同，语义也不同。

**目标**：引入 **轨迹搜索 (trajectory search)**，在搜索时不仅比较当前 step，还回溯比较之前若干 step 的 key，从而利用时序一致性提高匹配质量。

---

## 2. 核心思路

### 2.1 Query 侧：暂存历史 key

在推理 episode 进行过程中，每个 step 生成的 query key 需要被 **暂存 (buffer)**。这样在搜索时，query 侧拥有从 episode 开始到当前 step 的完整 key 序列：

```
query_history = [key_step_0, key_step_1, ..., key_step_t]
```

可配置的 **轨迹深度 (trajectory_depth)**：
- `depth=1`：仅当前 step（退化为现有行为）
- `depth=2`：当前 step + 上一个 step
- `depth=3`：当前 step + 前两个 step
- 以此类推

当 `current_step < depth - 1` 时（历史不足），只使用已有的 step。

### 2.2 Database 侧：双向链表 + 轨迹标识

数据库中的 entry 需要增加 **双向链表指针** 和 **轨迹标识**，使得从任意一个 entry 可以沿轨迹前后遍历，并能识别轨迹归属：

```
CacheEntry 新增字段:
    prev_ids:      list[str]       # 前驱 entry id 列表
    next_ids:      list[str]       # 后继 entry id 列表
    trajectory_id: Optional[str]   # 轨迹归属标识
    step_idx:      Optional[int]   # 已有字段，在轨迹中的 step 序号
```

**字段规则：**

- `prev_ids` / `next_ids`：使用 `list[str]` 而非单值，为未来轨迹分叉/汇合预留能力
  - 轨迹首个 entry 的 `prev_ids=[]`，末尾 entry 的 `next_ids=[]`
  - **⚠️ 当前阶段约束：list 中最多只有一个元素**，但遍历和搜索接口从一开始就按多分支设计（递归实现，见 2.3 节）
  - **⚠️ 风险标注：多分叉/汇合场景的 trajectory_id 管理规则未设计，留待轨迹优化阶段专项研究**
- `trajectory_id`：标识 entry 所属轨迹。初始 ingest 时以 episode 为单位分配（`trajectory_id = episode_id`），后续操作可变：
  - **合并/拼接**：两段 entry 统一改为新 `trajectory_id`，接上首尾指针（前段尾 `next_ids` 追加后段头 id，后段头 `prev_ids` 追加前段尾 id）
  - **拆分**：断开指针处，后半段全部改为新 `trajectory_id`
  - **剪枝**：删除 entry，更新前后邻居指针，`trajectory_id` 不变
  - **分叉**：多个后继 entry 可共享同一个前驱，各分支可拥有不同 `trajectory_id`（⚠️ 具体规则待轨迹优化阶段设计）
- `step_idx`：在轨迹中的位置序号。初始 ingest 时按 episode 内时间顺序递增。剪枝后可能出现空洞（不要求连续），合并后可能需要重编号
- 所有新增字段均为 `Optional` 或空列表默认值，不影响已有 entry 的兼容性

**数据来源：** **libero_spatial** 数据集。每个 episode 中的 step 按时间顺序排列，在 ingest 时自动构建链表关系。

### 2.3 搜索时的轨迹比较

搜索分为三层：

**第一层 (per-field similarity)**：与现有实现相同。对于单个 step 中的每个 field（vision_0, vision_1, vision_2, prompt_emb, robot_state），计算 cosine similarity 或 L2 距离。

**第二层 (per-step fusion)**：与现有实现相同。将同一 step 内多个 field 的 similarity 通过 RRF 或 weighted score sum 融合为该 step 的单一 similarity 值。

**第三层 (cross-step fusion, 新增)**：通过 **递归回溯** 遍历候选 entry 的所有前驱路径，逐层累积加权 similarity，支持线性链和分叉树两种拓扑。

#### 递归 trajectory similarity 算法

```python
def _recursive_trajectory_sim(
    entry_id: str,
    query_history: list[dict[str, Tensor]],  # query侧历史 [当前, t-1, t-2, ...] (newest-first)
    depth: int,                               # 剩余回溯深度（初始调用时 = len(weights) - 1）
    max_depth: int,                           # 最大回溯深度（= len(weights) - 1，递归中不变）
    weights: list[float],                     # 各层权重 [w_当前, w_t-1, w_t-2, ...] (newest-first)
    accumulated_sim: float,                   # 上层传入的累积 similarity
) -> list[float]:                             # 返回所有完整路径的 trajectory_sim
    """递归计算轨迹 similarity。

    索引映射：idx = max_depth - depth
      depth=max_depth（当前步）→ idx=0 → query_history[0]=当前, weights[0]=w_当前
      depth=0        （最老步）→ idx=max_depth → query_history[-1], weights[-1]

    每层：
      1. 计算当前 entry 与对应 query step 的 step_similarity（第一层+第二层融合）
      2. 加权累积到 accumulated_sim
      3. 如果达到深度上限或无前驱 → 返回 [accumulated_sim]（一条完整路径）
      4. 否则递归遍历所有 prev_ids，收集所有分支路径的分数
    """
    idx = max_depth - depth
    step_sim = step_similarity(query_history[idx], entries[entry_id].query_keys)
    accumulated_sim += weights[idx] * step_sim

    if depth == 0 or not entries[entry_id].prev_ids:
        return [accumulated_sim]  # 叶子节点，返回一条完整路径的分数

    all_scores = []
    for prev_id in entries[entry_id].prev_ids:
        all_scores.extend(_recursive_trajectory_sim(
            prev_id, query_history, depth - 1, max_depth, weights, accumulated_sim
        ))
    return all_scores
```

#### 分支聚合策略

递归返回的 `list[float]` 包含所有完整路径的 trajectory_sim：
- **无分叉**（当前阶段）：返回 `[一个分数]`，直接使用
- **有分叉**：返回多个分数，需聚合为单一值。聚合策略可配置：
  - `max`：取最优路径（乐观匹配）
  - `mean`：取平均（综合匹配）
  - `min`：取最差路径（保守匹配）
  - **⚠️ 当前阶段默认 `max`，多分支聚合策略的最优选择待轨迹优化阶段实验确定**

#### 权重策略

- 最近的 step 权重最大，越远越小（例如指数衰减或线性衰减）
- 可通过 YAML 配置具体权重值
- `query_history` 不足 depth 时，只使用已有步数，缺失步不参与计算

### 2.4 搜索流程示意

**线性链（当前阶段）：**

```
Query Side (推理时暂存):          Database Side (链表):

step 0: key_q0                    entry_A0 → entry_A1 → entry_A2 → entry_A3 (trajectory A)
step 1: key_q1                    entry_B0 → entry_B1 → entry_B2 → entry_B3 (trajectory B)
step 2: key_q2 (current)

depth=3 时搜索 candidate entry_A2:
  递归调用: _recursive_trajectory_sim(entry_A2, [key_q2, key_q1, key_q0], depth=2, ...)
    depth=2: sim += w0 * similarity(key_q2, entry_A2.key)  → 递归 prev_ids=[entry_A1]
    depth=1: sim += w1 * similarity(key_q1, entry_A1.key)  → 递归 prev_ids=[entry_A0]
    depth=0: sim += w2 * similarity(key_q0, entry_A0.key)  → 返回 [sim]
  结果: [trajectory_sim]  → 一条路径，直接使用
```

**分叉树（未来场景）：**

```
                entry_X0 → entry_X1 ─┐
                                      ├→ entry_C2 → entry_C3
                entry_Y0 → entry_Y1 ─┘

depth=3 时搜索 candidate entry_C3:
  递归 entry_C3 → entry_C2 → prev_ids=[entry_X1, entry_Y1]
    分支1: entry_X1 → entry_X0 → 返回 sim_path_1
    分支2: entry_Y1 → entry_Y0 → 返回 sim_path_2
  结果: [sim_path_1, sim_path_2] → 聚合(max/mean/min) → 最终分数
```

---

## 3. 需要改动的组件

### 3.1 数据结构 (`storage_types.py`)

**CacheEntry 新增字段：**

```python
prev_ids:      list[str] = field(default_factory=list)   # 前驱 entry id 列表
next_ids:      list[str] = field(default_factory=list)   # 后继 entry id 列表
trajectory_id: Optional[str] = None                       # 轨迹归属标识
# step_idx 已有，不需新增
```

- 所有新增字段均为 Optional / 空列表默认值，不影响已有 entry 的兼容性

**CachePayload 变更：**

- **删除 `next_action_chunk`**：有了 `next_ids` 链表，下一步 action 通过 `next_ids[0]` → fetch entry → `payload.action_chunk` 获取，不再冗余存储
- **保留 `intermediates` / `denoising_num_steps`**：CP2 warm-start 专用，运行时写入时设为 None

**QuerySpec 扩展：**

```python
# 新增字段（均 Optional，None 时退化为现有行为）
trajectory_history: Optional[list[dict[str, Tensor]]] = None  # query 侧历史 key 序列
trajectory_weights: Optional[list[float]] = None               # 各层权重 [w_当前, w_t-1, ...]
```

### 3.2 数据库后端 (`backends/`)

**不新建 Backend 子类**，在现有 Backend 的 `search()` 方法内扩展：

- **InMemoryBackend**：在现有 `search()` 中增加轨迹融合分支。`trajectory_history=None` 时走原有逻辑；非空时执行：KNN 初始检索 → 候选沿 `prev_ids` 回溯 → 逐步 similarity → 加权重排序（递归实现，见 2.3 节）
- **QdrantBackend**：检测到 `trajectory_history` 非空且 depth > 1 时 `raise NotImplementedError("trajectory search not supported in QdrantBackend")`；depth=1 或无历史时正常执行不报错

### 3.3 数据 Ingest

- 从 libero_spatial 导入数据时，按 episode 分组，按 step 排序
- 构建双向链表：`entry[i].next_ids = [entry[i+1].id]`，`entry[i+1].prev_ids = [entry[i].id]`
- Episode 首个 entry 的 `prev_ids=[]`，末尾 entry 的 `next_ids=[]`
- `trajectory_id` 初始设为 `episode_id`

### 3.4 History Buffer：各组件自治

**不设中心化 tracker**，各组件从各自输入积累历史，自行决定如何使用：

| 组件 | 输入数据 | 历史内容 | 本次是否实现 |
|------|----------|----------|-------------|
| SearchStrategy | `ctx.query_keys`（CPU 降维后） | query_keys 历史 buffer | ✅ 实现 |
| Gate | `cached_data`（GPU 原始） | cached_data 历史 | ❌ 预留接口，需详细注释 |
| Judge | `cached_data`（GPU 原始） | cached_data 历史 | ❌ 预留接口，需详细注释 |

**各组件需新增的接口：**

- `on_episode_start()` / `reset()`：清空 history buffer，由 Orchestrator 在 `on_task_begin()` / `on_episode_start()` 时统一调用
- `record_action(action_chunk: Tensor)`：接收 Orchestrator 广播的 action。**必须是纯本地 buffer 操作（append to list），禁止回调 Backend / CacheStorage / Orchestrator 或获取任何外部锁**

**Action 广播**：Orchestrator 拿到 action 后依次调用各组件的 `record_action()`。死锁风险：无（`check()` 已返回、锁已释放，全程顺序执行；并发连接有独立的 per_connection_components）。

### 3.5 SearchStrategy：升级现有策略

**不新增独立的轨迹策略类型**。现有所有策略（WeightedRrfKnnStrategy、WeightedScoreSumKnnStrategy 等）统一增加完整的轨迹搜索功能：

- 维护 query_keys + action_chunk 历史 buffer
- `on_episode_start()` 生命周期管理
- `record_action()` 接口
- 搜索时从 buffer 构造 `trajectory_history` / `trajectory_weights` 填入 QuerySpec，传给 Backend
- `trajectory_depth=1`（或未配置）时：递归深度为 1，自然产出与单步搜索相同的结果（同一套代码路径的退化行为，不是分流到旧逻辑）

### 3.6 Orchestrator：写入流程重构

**原逻辑**：每步调用 `orchestrator.write()` → 立即 `storage.insert()`

**新逻辑**：读写分离

1. **推理时只读**：每步调用 `orchestrator.buffer_for_write()`，暂存 query_keys + action_chunk
2. **episode 结束时统一写入**：`on_episode_end()` 时：
   - 调用 `WritePolicy.should_write(episode_record)` 决定是否写入
   - 若写入：用暂存数据构造完整 CacheEntry 链（含 `prev_ids` / `next_ids` / `trajectory_id`），`batch_insert()` 一次写入

**WritePolicy**（可插拔写入开关）：

```python
@dataclass
class StepRecord:
    query_keys: dict[str, Tensor]        # CPU float32
    action_chunk: Tensor                 # CPU float32，必填（无 action 的步不写入）

@dataclass
class EpisodeRecord:
    steps: list[StepRecord]              # 每步暂存的 query_keys + action_chunk
    task_key: str                         # 任务标识
    miss_by_checkpoint: dict[CheckpointID, int]  # e.g. {CP1: 3, CP3: 50}
    total_steps: int                      # 总步数

class WritePolicy(Protocol):
    def should_write(self, episode_record: EpisodeRecord) -> bool: ...
```

实现类型：`on_any_miss`（默认，episode 中有任何未命中则写入）、`always`、`never`

### 3.7 Config 与 YAML 配置

**SearchStrategyConfig 新增字段：**

```python
# SearchStrategyConfig dataclass 新增
trajectory_depth: int = 1                          # 默认 1，退化为单步搜索
trajectory_weights: Optional[list[float]] = None   # 默认 None，depth=1 时不需要
```

```yaml
search_strategy:
  type: weighted_score_sum_knn              # 类型名不变
  top_k: 5
  trajectory_depth: 3                       # 轨迹深度（1=退化为单步，不配置默认为 1）
  trajectory_weights: [1.0, 0.5, 0.25]     # 各 step 权重，newest-first（从当前到历史）
  # ... 现有 per-field 配置继续保留
```

**CacheConfig 顶层新增 WritePolicyConfig：**

```yaml
enabled: true

write_policy:
  type: on_any_miss        # on_any_miss | always | never
  # 未来可扩展参数，如 miss_ratio_threshold 等

timer:
  # ...
checkpoints:
  # ...
```

**Config dataclass 变更：**

```python
@dataclass
class WritePolicyConfig:
    type: str = "on_any_miss"

@dataclass
class CacheConfig:
    write_policy: WritePolicyConfig = field(default_factory=WritePolicyConfig)
    # ... 其余现有字段 ...
```

**工厂函数**：`_build_write_policy(cfg)` 根据 `cfg.type` 返回对应实现，未知类型报 `ConfigValidationError`。`build_cache_components` 返回的 dict 新增 `write_policy` key。

---

## 4. 向后兼容性

- **现有 SearchStrategy 全部升级**：WeightedRrfKnnStrategy、WeightedScoreSumKnnStrategy 等都增加完整的轨迹搜索功能（history buffer 维护、`on_episode_start()` 生命周期、`record_action()` 接口、构造 `trajectory_history` / `trajectory_weights` 传入 QuerySpec）。不新增独立的轨迹策略类型，而是在现有策略中统一支持
- **通过 YAML 配置 `trajectory_depth` 和 `trajectory_weights` 控制**：所有策略都读取这两个参数
- **`trajectory_depth=1`（或未配置）时**：递归深度为 1，自然产出与单步搜索相同的结果（同一套代码路径的退化行为，不是分流到旧逻辑）
- **CacheEntry 新增字段均为 Optional / 空列表默认值**：不影响已有数据
- **InMemoryBackend**：在现有 `search()` 中扩展轨迹分支，`trajectory_history=None` 时走原有逻辑

---

## 5. 待确认事项

- [x] 轨迹深度和权重的具体默认值 → **由 YAML 配置，不设硬编码默认值，待实验确定**
- [x] 初始候选数 top-K 与最终返回数的关系 → **无需新参数**，见 Q7 结论
- [x] 是否需要缓存推理时的 action → **确认需要**：每步的 action_chunk 也要暂存到 history buffer 中
- [x] QdrantBackend 是否需要同步支持 → **仅先实现 InMemoryBackend**，见 Q1 结论
- [x] 双向链表的构建是在 ingest 脚本中完成，还是在 backend.insert 时自动维护 → **离线 ingest 批量构建 + 运行时 episode 结束统一写入**，见 Q5 结论

---

## 6. 讨论结论

### Q1: History Buffer 归属与数据流（已确认）

**结论：各组件自治，不设中心化 tracker**

#### 历史暂存策略

各组件从各自的输入积累历史，自行决定如何使用：

| 组件 | 输入数据 | 历史内容 | 本次是否实现 |
|------|----------|----------|-------------|
| SearchStrategy | `ctx.query_keys`（CPU 降维后） | query_keys 历史 buffer | ✅ 实现 |
| Gate | `cached_data`（GPU 原始） | cached_data 历史 | ❌ 预留接口，需详细注释说明未来用途和扩展方式 |
| Judge | `cached_data`（GPU 原始） | cached_data 历史 | ❌ 预留接口，需详细注释说明未来用途和扩展方式 |

#### Action 广播

Orchestrator 拿到 action（无论来自 cache 命中还是模型推理）后广播给各组件。两阶段记录：
- **阶段 1**：各组件在正常调用流程中自动积累输入数据（Gate/Judge 收到 cached_data，SearchStrategy 收到 query_keys）
- **阶段 2**：Orchestrator 统一广播 action_chunk 给需要的组件

**死锁风险分析：无风险**，理由：
- 单连接场景：`check()` 已返回、所有锁已释放后才广播，全程顺序执行
- 并发连接场景：每个连接有独立的 per_connection_components，不存在交叉锁

**实现约束**：各组件的 `record_action()` 方法**必须是纯本地 buffer 操作**（append to list），**禁止回调** Backend / CacheStorage / Orchestrator 或获取任何外部锁。此约束需在接口注释中明确标注。

#### 生命周期

各组件需要 `on_episode_start()` / `reset()` 接口，由 Orchestrator 在 `on_task_begin()` / `on_episode_start()` 时统一调用。

#### 轨迹搜索逻辑下沉到 Backend

**不在 SearchStrategy 中做两次 Backend 调用**，而是将轨迹搜索逻辑封装进 Backend.search() 一次完成：

- 扩展 `QuerySpec`，新增 `trajectory_history` 和 `trajectory_weights` 字段
- SearchStrategy 负责维护 query_keys 历史 buffer，构造 QuerySpec 时填入历史
- Backend.search() 内部完成：KNN 初始检索 → 候选回溯 → 逐步 similarity → 加权重排序
- `trajectory_history=None` 时完全退化为现有行为

#### Backend 实现策略

- **InMemoryBackend**：在现有 `search()` 方法内扩展轨迹融合分支，不新建搜索方法
- **QdrantBackend**：检测到 `trajectory_history` 非空且 depth > 1 时 raise `NotImplementedError("trajectory search not supported in QdrantBackend")`；depth=1 或无历史时正常执行不报错
- 不新建 Backend 子类

#### Gate/Judge 签名

本次不修改 Gate/Judge 签名。已知 caveat：Gate/Judge 用 GPU `cached_data` vs SearchStrategy 用 CPU `query_keys` 的数据层次差异是现有架构问题，本次不解决。未来如需 trajectory-aware gate/judge，可增量扩展签名。

### Q3: action_chunk 是否参与搜索（已确认）

**结论：本次不参与**

- action_chunk 只暂存到各组件的 history buffer 中
- 是否参与 similarity 计算由各组件（SearchStrategy / Gate / Judge）的具体实现自行决定
- 本次实现的 SearchStrategy、Gate、Judge 均不使用 action 数据
- 未来扩展时，组件可自行从 buffer 中取用 action 进行 action-conditioned 检索或一致性校验

### Q4: 数据结构方案（已确认）

**结论：双向链表（list 形式）+ trajectory_id + step_idx**

不用 episode_id + step_idx 纯索引方案，也不用纯链表方案，而是两者结合：

- **双向链表 (`prev_ids: list[str]` / `next_ids: list[str]`)**：用于 O(1) 遍历，支持剪枝/合并/拆分等动态指针操作。使用 list 为未来轨迹分叉/汇合预留能力
- **`trajectory_id`**：轨迹归属标识，初始等于 episode_id，合并/拆分后可变，用于轨迹级别的批量查询和区分
- **`step_idx`**（已有字段）：轨迹内位置，剪枝后允许空洞

**⚠️ 当前阶段约束**：`prev_ids` / `next_ids` 中最多一个元素，多分叉/汇合逻辑待轨迹优化阶段设计

选择理由：
- 纯 episode_id + step_idx 方案在剪枝后 step_idx 出现空洞，回溯需要排序查找，不再是 O(1)；合并/插入需要重编号
- 纯链表方案缺少轨迹级别的归属标识，无法快速回答"这个 entry 属于哪条轨迹"
- 双向链表 + trajectory_id 兼顾遍历效率和轨迹管理灵活性
- list 形式为未来分叉/汇合预留扩展空间，当前阶段实现简单（只取 `[0]`）

### Q5: Ingest 与运行时写入策略（已确认）

#### 离线 Ingest

在 ingest 脚本中离线批量构建链表：
- 按 episode 分组，按 step 排序
- 一次性填好所有 `prev_ids` / `next_ids` / `trajectory_id`（初始 `trajectory_id = episode_id`）
- 通过 `backend.batch_insert()` 写入，Backend 不需要理解链表语义

#### 运行时写入：episode 结束时统一写入

**不再逐步 insert**，改为读写分离：
- **推理时只读 cache**：SearchStrategy 等组件在正常调用流程中积累 query_keys 和 action_chunk 历史
- **episode 结束时统一写入**：用暂存的历史数据构造完整的 CacheEntry 链（带 prev_ids / next_ids / trajectory_id），通过 `batch_insert` 一次写入

**写入所需数据**（均在 episode 期间已暂存）：
- 每步的 `query_keys`（SearchStrategy 的 history buffer）
- 每步的 `action_chunk`（action 广播机制暂存）
- `task_key`（episode 级别，一个即可）
- `intermediates` / `denoising_num_steps`：运行时写入时设为 None（CP2 未实现，这两个字段绑定，都是 CP2 warm-start 数据，当前始终为 None，设 None 可直接通过现有校验）

**现有 `Orchestrator.write()` / `write_with_keys()` 流程变更**：
- 原逻辑：每步调用 `orchestrator.write()` → 立即 `storage.insert()`
- 新逻辑：每步调用 `orchestrator.buffer_for_write()` → 暂存到 Orchestrator 的写入 buffer → `on_episode_end()` 时统一构造链表并 `batch_insert()`

#### 可插拔写入开关（WritePolicy）

是否在 episode 结束时写入轨迹，由可插拔的 WritePolicy 决定：

```python
class WritePolicy(Protocol):
    def should_write(self, episode_record: EpisodeRecord) -> bool: ...
```

当前默认实现：episode 中 CP1 出现过任何 cache miss 或 gate skip（`miss_by_checkpoint.get(CP1, 0) > 0`）→ 写入完整轨迹。默认只看 CP1（CP3 skeleton 永远 miss，不应触发写入）。未来可替换为其他策略。

**Config 与 YAML 设计：**

新增 `WritePolicyConfig` dataclass，挂在 `CacheConfig` 顶层：

```python
# config.py 新增
@dataclass
class WritePolicyConfig:
    type: str = "on_any_miss"       # 写入策略类型
    # 未来可扩展参数，如 miss_ratio_threshold 等

# CacheConfig 新增字段
@dataclass
class CacheConfig:
    enabled: bool = False
    write_policy: WritePolicyConfig = field(default_factory=WritePolicyConfig)
    # ... 其余现有字段 ...
```

对应 YAML 配置：

```yaml
# cache.yaml
enabled: true

write_policy:
  type: on_any_miss          # 默认：episode 中有任何未命中则写入
  # type: always             # 每个 episode 都写入
  # type: never              # 不写入（纯读模式）

timer:
  # ...
checkpoints:
  # ...
```

**工厂函数：**

```python
# config.py 新增
def _build_write_policy(cfg: WritePolicyConfig):
    if cfg.type == "on_any_miss":
        return OnAnyMissWritePolicy()
    elif cfg.type == "always":
        return AlwaysWritePolicy()
    elif cfg.type == "never":
        return NeverWritePolicy()
    else:
        raise ConfigValidationError(f"Unknown write_policy.type '{cfg.type}'")
```

**build_cache_components / build_per_connection_components 返回的 dict 中新增 `write_policy` key**，由 Orchestrator 在 `on_episode_end()` 时调用 `write_policy.should_write(episode_record)` 决定是否写入。

**校验规则：**
- `write_policy.type` 必须是已知类型，否则 `validate_cache_config()` 报错
- `WritePolicyConfig` 加入 `_CONFIG_TYPES` 注册表以支持 YAML 递归解析

#### CachePayload 字段清理

**删除 `next_action_chunk`**：
- 有了 `next_ids` 链表，下一步 action 通过 `next_ids[0]` → fetch entry → `payload.action_chunk` 获取
- 不再需要冗余存储 `next_action_chunk`，也不再需要 DeferredWriter 回填机制
- 同步更新 CP3 设计说明：通过链表遍历获取下一步 action，替代 `next_action_chunk` + DeferredWriter 方案
- **CP3 向前滚动的具体逻辑（滚动多少步、何时触发、如何调度）不在本次设计范围内**

**保留 `intermediates` / `denoising_num_steps`**：
- 这两个字段是 CP2 warm-start 专用，未来 CP2 确实需要，不能被链表替代
- 运行时写入时设为 None，通过现有校验无问题

#### 已知 trade-off

- **崩溃丢数据**：进程 mid-episode 崩溃，整个 episode 的暂存数据丢失。对于实验场景可接受
- **内存占用**：episode 期间需要在内存中暂存所有步的 query_keys + action_chunk。LIBERO episode 通常几百步，每步几个小张量（KB 级），总量可控

### Q7: top-K 与最终返回数（已确认）

**结论：无需新参数，`top_k` 语义不变**

轨迹搜索只是评分方式的变化——用轨迹 similarity 替代单步 similarity 给候选打分。返回的仍然是当前 step 的 `top_k` 个候选 entry（不是整条轨迹）。Backend.search() 内部如需更大的初始候选池，用现有的 `candidate_multiplier` 参数控制即可。

### Q8: 回溯时 entry 缺失的处理（已确认）

**结论：递归自然处理，无需额外逻辑**

两种情况都由递归终止条件覆盖：
- **历史不足**（如 depth=5 但当前 step=3）：query 侧历史不够，递归提前终止
- **链条断裂**（如 entry 被删除导致 `prev_ids=[]`）：递归遇到空 `prev_ids` 直接返回累积分数

两种情况下缺失步不贡献 similarity（等效为 0），已有递归逻辑严密覆盖，无需特殊处理。

### Q9: trajectory_depth 和 weights 默认值（已确认）

**结论：由 YAML 配置，不设硬编码默认值，待实验确定**
