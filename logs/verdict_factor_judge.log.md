---
status: Plan
level: L3
authority: Execution
related:
  - logs/verdict_factor_candidates.log.md
  - docs/architecture/cache_system.md
  - src/openpi/cache/components/judge.py
  - src/openpi/cache/components/gate.py
  - src/openpi/cache/orchestrator.py
  - src/openpi/cache/storage_types.py
---

# Verdict Factor Judge — 实施计划

> **范围**：把 `logs/verdict_factor_candidates.log.md` 中列出的统计/运动学因子（F1a-A / F1a-T / F1b-A / F1b-T / F2）落到 cache 子系统的 Judge 阶段，作为 `ThresholdJudge` 之外的另一可插拔 Judge 实现。
>
> **结构**：§1-§7 描述设计意图与权衡，§8 是 executor 的代码级蓝图（file path / 类签名 / wiring 顺序 / 测试 / 文档交付物 / 风险）。两层应保持一致 —— 任何冲突以 §8 为准。
>
> **状态**：Plan，G1 APPROVED，post-G1 polish 完成（§2.8 intro / §3 因子矩阵 / §4 batch table 的 thin-subclass 措辞统一；Review Log 整段按 execution_authority §3.1 删除）。等待用户 §4 Code 启动指令。

---

## 1. 目标与非目标

**目标**
- 在 Judge 阶段引入"因子向量 → 组合器 → tier 映射"的三段式判定，替代单一 RRF / cosine 阈值。
- Factor 模块按 capability 协议组织（同一因子可同时实现写入侧与读取侧），离线 build pkl 与在线 server 共用同一注册表与同一份配置。
- 老 Judge / 老 entry / 老 backend / 老 KeyBuilder 全部保持可用。

**非目标**
- 不引入 Gate 端的因子（候选集中所有因子都在 search 之后才能算）。
- **不改 `VectorStoreBackend` ABC、不改 `QueryKeyBuilder`、不改 `WritePolicy`**。`SearchStrategy` 与 `SimilarityJudge` Protocol 仅做向后兼容的扩展（kw-only 默认参数）；详情见 §2.6 "有限度变更"。
- 不重训、不引入学习型组合器（S4 留作后续）。

## 2. 改动面（骨架）

### 2.1 Schema 扩展（向后兼容）

**已确认的现状**（实测 `exp/common/data/cache_artifacts/libero_10/cp1_mean_pool.pkl`）：

- `payload.action_chunk: [H, 32]` 是 **pre-output-transform** 的 normalized model output（interceptor.py:594 在 `_output_transform` 之前 detach 并 buffer）
- `query_keys["robot_state"]: [32]` 是 **post-input-transform** 的 normalized obs（KeyBuilder 拿到的就是 transform 之后的）—— **每条 entry 都已存**，沿 `prev_ids/next_ids` 链平滑演化
- 两侧 active 维（LIBERO: action 7 / state 8）都有 per-DOF std 0.24-1.0 的异质性；DOF 7-31 全部为 Pi0.5 universal-32 零填充

**因此 schema 仅需新增一个字段**：

| 类型 | 字段 | 来源 | 说明 |
|---|---|---|---|
| `CachePayload` | `factors: Optional[dict[str, float]] = None` | F1b-* 离线写入 | verdict 时 OnlineExtractor 读 |

**取消的字段**（基于实测数据）：

- ~~`CachePayload.future_state_chunk`~~ —— F1a-T 改用 `PayloadView.walk_next(winner_id, k)` 取后续 entry 的 `query_keys["robot_state"]`，不需要 payload 携带额外 tensor。代价：每次 F1a-T 算一次多 K 次 entry get（in-memory 几乎零成本）；收益：B3 不再涉及 schema 改动。
- ~~`StepRecord.robot_state`~~ —— state 已经在 `entry.query_keys["robot_state"]`，OfflineWriter 可直接读，无需在 StepRecord 多存一份；同理无需新增 `broadcast_state` lifecycle。

`factors` 字段 Optional + default None，老 entry 反序列化不受影响。

### 2.2 新建模块

```
src/openpi/cache/components/factors/
  __init__.py
  base.py                # OnlineExtractor / OfflineWriter Protocols
  registry.py            # name -> class, build_from_config()
  runtime_continuity.py  # F1a-A, F1a-T          (Online only) — 同类双参
  source_window.py       # F1b-A, F1b-T          (Online + Offline) — 同类双参
  consensus.py           # F2                    (Online only)
  composers/             # weighted_sum / and / or
  normalizers/           # percentile rolling

src/openpi/cache/components/payload_view.py   # PayloadView Protocol + default impl
```

**实现约定（防止 -A / -T 写成两份并行代码）**：F1a 与 F1b 各对应**一份实现基类 + 两个 thin subclass**。

- `source_window.py`：基类 `SourceWindowSmoothness` 持全部算法实现（descriptor 计算、z-score、windows 滑动、payload.factors key 拼接）；两个 thin subclass `SourceWindowSmoothnessAction` / `SourceWindowSmoothnessState` 仅重设 `source` / `requires_chain_walk` 等 class-level 属性，不重写任何方法；registry 注册 `f1b_a` → action subclass，`f1b_t` → state subclass。
- `runtime_continuity.py` 同样模式：`RuntimeContinuity` 基类 + `RuntimeContinuityAction` / `RuntimeContinuityState` 两个 thin subclass；registry 注册 `f1a_a` / `f1a_t`。

`source` 不通过 YAML params 传 —— 它由 class identity（即 registry 注册的 type 名）唯一决定。`__init__` 与 `describe(cls, params)` 都从 `cls.source` 读，不读 `params['source']`。

为什么这个模式优于"单 class + params['source']"：
- capability flag（`requires_chain_walk` 在 source="state" 时 True，source="action" 时 False）是 class-level，必须在 class 上设；用 thin subclass 让每个 class attribute 单值
- registry / validator 只走单一 class attribute 路径，不需要参数化 capability check
- describe / __init__ 不需要 YAML 多塞 `source` 字段（type 名已经隐含 source）

收益（仍然成立）：
- 描述子算法升级（B2 后扩 dirvar / path / freq / autocorr）只动**基类**一份代码
- σ / active_mask 选择逻辑集中在基类
- 测试只测基类的算法 + 各 thin subclass 的 capability flag 覆盖

新增 `CompositeJudge`（在 `judge.py` 同文件追加）：组合 Extractors + Normalizer + Composer → `JudgeResult`。

### 2.3 PayloadView：Judge 端的取数接口

设计原则：把"取候选 payload + 走前后链"包装成一个**对象**，Judge 通过它访问 storage 但**不持有 storage 句柄**。这是对 `docs/architecture/cache_system.md` §5.6 Judge purity 契约的 refinement —— 原契约的语义本意（"no side effects"）被精确化为"**Judge 不写 storage；可经 PayloadView 只读访问**"。详细论证见 §8.8.1 与 G1 Round 2 Item 1 回复。

#### 2.3.1 接口签名（B0 落地）

```python
class ForkPolicy(Enum):
    TRAJECTORY = auto()      # 默认：仅走 trajectory_id 与 anchor 相同的分支
    FIRST = auto()           # 取 prev_ids[0] / next_ids[0]，确定但语义任意
    STOP = auto()            # 遇 fork 即停，返回已收集的（< k）
    ALL_BRANCHES = auto()    # 返回 list[list[CacheEntry]]，每条一支路径
    SCORE = auto()           # 取与当前 query 相似度最高的分支

class PayloadView(Protocol):
    # 单点
    def get(self, entry_id: str) -> CachePayload: ...
    def get_entry(self, entry_id: str) -> CacheEntry: ...

    # 邻域窗口（newest-first，距 anchor 最近的在前）
    def walk_prev(
        self, entry_id: str, k: int,
        *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...

    def walk_next(
        self, entry_id: str, k: int,
        *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...

    # 批量取（top-k payload 用）
    def get_many(self, entry_ids: list[str]) -> list[CachePayload]: ...
```

#### 2.3.2 B0 实现范围（fork 部分留白）

| 能力 | B0 实现 |
|---|---|
| `get` / `get_entry` / `get_many` | ✅ 实现，内部 memo（per-`check()` 生命周期）|
| `walk_prev` / `walk_next` 默认链上行走（`prev_ids`/`next_ids` 长度 ≤ 1，无 fork）| ✅ 实现 |
| `fork_policy=TRAJECTORY` 在出现 fork 时的 anchor 选择 | ❌ `raise NotImplementedError`（当前数据库无 fork，触发即说明上游写入语义变了，应当显式失败而不是猜）|
| `fork_policy ∈ {FIRST, STOP, ALL_BRANCHES, SCORE}` | ❌ 全部 `raise NotImplementedError`（参数占位，签名锁定）|
| `cross_trajectory=True` | ❌ `raise NotImplementedError`（参数占位）|

实现层检测到 `len(prev_ids) > 1` 或 `len(next_ids) > 1` 时，按 fork_policy 路由；当前唯一可用的路径是"无 fork"（每节点至多 1 个邻居）的简单遍历。一旦未来引入 semantic-dedup id / 多 trajectory 合并产生真实 fork，再按需实现各 policy。

#### 2.3.3 Judge 接口扩展（kw-only，默认 None 兼容老 Judge）

- `SimilarityJudge.__call__` 新增 `view: PayloadView | None = None`、`history: HistoryView | None = None`（action + state 历史 view，由 Orchestrator 注入）。
- 老 Judge（`ThresholdJudge` / `AlwaysHitJudge` / `AlwaysWarmStartJudge`）以 `**kwargs` 吸收新参数，行为字节级一致。
- `Orchestrator.check()` 在 Judge 之前构造 `StoragePayloadView` + `HistoryView` 并注入。**state 历史不通过 broadcast 维护**——直接在 `check()` 内从 `query_keys["robot_state"]` 取（state 已是 `query_keys` 字段，无需新增 `broadcast_state` lifecycle）；action 历史复用现有 `broadcast_action` 路径。详细 lifecycle 表见 §8.2.3。

### 2.4 写入路径

**Interceptor**：不动。`buffer_for_write` 签名保持现状，不需要新增 `robot_state` 参数。

**Orchestrator `_build_entry_chain`**：build 完整条 entry 链后（含 `query_keys`、`payload.action_chunk`、`prev_ids/next_ids/trajectory_id` 全部就位），把链整体喂给每个 `OfflineWriter`：

```python
def compute_for_episode(
    self,
    entries: list[CacheEntry],
    library_stats: LibraryStats,
) -> list[dict[str, float]]:
    """Return per-entry factor dict to merge into entries[i].payload.factors."""
```

OfflineWriter 自取所需 —— F1b-T 读 `entries[i].query_keys["robot_state"]` 序列，F1b-A 读 `entries[i].payload.action_chunk[0]` 序列（每 chunk 第一步 = 实际即将执行的 action）。`library_stats` 由 Orchestrator 从 backend 取后透传（§8.2.3 / §8.3）。

**`required_payload_fields()` 协议保留**但 B0-B2 用不到（当前列出的因子无需新 raw payload field）；接口预留给未来真正需要"携带额外 tensor 才能算"的因子。

**离线 build pkl 工具**：复用同一 `OfflineWriter` + 同一 `compute_for_episode(entries, library_stats)` 入口，由 `enrich_artifact_with_factors` helper 集中调用。具体脚本与 helper 位置见 §2.5。

### 2.5 离线 build pkl 工具

现有 artifact build 脚本（实测）：

| 脚本 | 用途 | 改动 |
|---|---|---|
| `exp/common/build_in_memory_cache_artifact.py` | 主 artifact 构建路径（cp1_mean_pool / spatial_pool / max_pool / temporal_prune 等）| 写入 `entries` 后增加：(a) 调用 `LibraryStats.compute_from_entries(entries)` → 写入 artifact dict 的 `library_stats` 字段；(b) 对 config 中声明的每个 OfflineWriter 调 `compute_for_episode(per-trajectory entries)` → `entry.payload.factors`（per episode 切分依据 `entry.trajectory_id`）|
| `exp/common/build_llm_layer_matrix.py` | LLM-layer-extract artifact（cp1_llm_layer_extract）| 同上 |
| `exp/common/build_clip_cache_artifact.py` | CLIP-based artifact | 同上 |

三脚本共用一个新 helper（位置：`exp/common/factor_postprocess.py`）：

```python
def enrich_artifact_with_factors(
    entries: list[CacheEntry],
    offline_writers: list[OfflineWriter],
) -> LibraryStats:
    """In-place: compute library_stats + per-entry factors. Return library_stats."""
```

三脚本各加 ~3 行：import + 一行调用 + 一行写 artifact dict 的 `library_stats` 字段。

artifact pkl 顶层 dict 新格式：

```python
{
    "key_builder_type": str,
    "checkpoint_id": str,
    "vector_dims": dict[str, int],
    "entries": list[CacheEntry],          # entry.payload.factors 已填好
    "library_stats": LibraryStats,        # 新增（dataclass，pickle round-trip 安全）
}
```

**老 artifact 兼容**：`InMemoryBackend.load_artifact` 检测 `data.get("library_stats") is None` → 调 `LibraryStats.compute_from_entries(list(self._entries.values()))` 现算并 cache 到 `self.library_stats`（§8.2.5）。老 entry 的 `payload.factors is None` → CompositeJudge 的 F1b OnlineExtractor 检测缺字段 → 该因子返回 NaN → Composer 按 §2.8.8 规则处理。

`future_state_chunk` 字段已在 §2.1 中明确取消，本节不再涉及。

### 2.6 不动清单（解耦红线）

- `QueryKeyBuilder`、`WritePolicy`、`Gate` 的接口
- 现有 `ThresholdJudge` / `AlwaysHitJudge` / `AlwaysWarmStartJudge` 的运行时行为（接受新 kwargs 后行为字节级一致）
- `cache/types.py CACHE_QUERY_FIELDS`
- **`search_strategy.top_k` YAML 字段的语义**（当前指"该策略自身需要的候选数"，绝对不重定义）—— 老 YAML 全部 `top_k: 1`，行为零变化

**有限度变更**（在 §8 中精确指定）：

- `VectorStoreBackend` ABC：**真正不动**。新增的 chain-walk 能力走 facade-only 的 duck-typing 路径（见下）。
- `CacheStorage` facade：新增 `fetch_entry(id)` 方法。实现是 duck typing —— `getattr(self._backend, "fetch_entry", None)`，存在则透传，缺失则 raise `NotImplementedError(backend_class_name + 解释)`。这样 Backend ABC 真正不动；只有支持 in-memory entry 访问的 backend（当前 InMemoryBackend）需要新增一个**普通 public method**。
- `InMemoryBackend`：新增 `fetch_entry(id) -> CacheEntry` public method（**非** ABC override，就是一个普通方法），实现 `return self._entries[id]`。同时新增 `library_stats: Optional[LibraryStats] = None` 属性，由 `load_artifact` 填充或现算 fallback。
- `SearchStrategy` 三个 concrete class 的 `__init__`：各增一个 default-0 kwarg `min_top_k_hint`；`search()` 签名不动。
- `SimilarityJudge` Protocol：增两个 default-None kw-only 参数 `view` / `history`；老 Judge 用 `**kwargs` 吃掉，行为字节级一致。
- `docs/architecture/cache_system.md` §5.6 Judge purity 契约：refine 为"**Judge 不写 storage；可经 PayloadView 只读访问**"——这是 contract refinement，不是 violation（详见 §8.8.1 与 G1 Round 2 Item 1 的论证）。

非动 = 老 YAML / 老 entry / 老 backend / 老 Judge / 老 KeyBuilder 全部继续工作；新增能力通过新 YAML config 启用。

### 2.7 F2 / 多候选因子的 top_k 解决方案

**约束**：`search_strategy.top_k` YAML 字段的语义不能变；老 Judge 的 YAML 不能改；F2 需要 top-K candidate payload。

**做法**：让"策略自身要的 top_k"与"因子要的 top_k"是两个独立来源，运行时取 max。

| 配置位置 | 语义 | 谁拥有 |
|---|---|---|
| YAML `search_strategy.top_k`（已存在） | 该策略自身的候选需求 | SearchStrategy |
| YAML `judge.composite.factors.<name>.top_k`（新增） | 该因子算 consensus / 邻域所需的候选数 | Factor / Extractor |

**实现路径（最小耦合）**：

1. `Extractor` Protocol 加可选属性 `required_top_k: int = 0`
2. `CompositeJudge.__init__` 计算 `self.min_required_top_k = max((e.required_top_k for e in extractors), default=0)`
3. `SearchStrategy.__init__` 新增 kwarg `min_top_k_hint: int = 0`；内部 `self._effective_top_k = max(top_k, min_top_k_hint)`；`search()` 里用 `self._effective_top_k` 填 `QuerySpec.top_k`
4. `cache/config.py` 的 strategy builder：构造完 judge 后 `min_top_k = getattr(judge, "min_required_top_k", 0)`，作为 `min_top_k_hint` 传入 strategy 构造函数（per-CP 各自计算）

**对外可见的接口变化**：仅 SearchStrategy 各 concrete `__init__` 多一个 default-0 kwarg；`SearchStrategy.search()` 签名、`QuerySpec`、Backend ABC、所有老 YAML 全部不动。

**老配置回归**：老 Judge 没有 `min_required_top_k` 属性 → hint = 0 → `max(1, 0) = 1`，strategy 行为字节级一致。

**性能成本**（in-memory backend 实测分析）：`topk(5)` vs `topk(1)` 都是单一 op，差异忽略；`SearchResultLite` 多几个 id+float，无 payload，内存忽略；真正的 payload 拉取由 PayloadView 按需 `view.get_many([r.id for r in results[:k]])` 控制 —— 只在 F2 主动请求时才发生。

**F2 扩展位**：F2 内部使用时取 `min(self.K, len(results))`，避免 cold-start / 库太小时崩。`required_top_k` 与 F2 内部消费 K 短期等价，未来若想做 "拉 top-10 但 consensus 只看前 5、看 top-1 是否落在 consensus 中心" 这种再拆。

### 2.8 F1b 归一化基准与平滑度描述子集

> **实现路径**：F1b-A 与 F1b-T 共享一份算法实现基类 `_SourceWindowSmoothnessBase`，由两个 thin subclass `SourceWindowSmoothnessAction` / `SourceWindowSmoothnessState` 各自固定 `source` class attribute，registry 分别注册为 `f1b_a` / `f1b_t`（详见 §2.2 与 §8.1.6）。本节描述的描述子集 / 计算流程 / metadata 对两侧通用，仅输入序列与 σ-引用字段不同。

#### 2.8.1 为什么单一 |jerk| 不够

magnitude 与 direction 是平滑度的两个正交维度。仅靠 |jerk|：

- 多 DOF 协同直线运动（人觉得平滑）：|jerk| 低 ✅
- 突然转弯（不平滑）：|jerk| 局部尖峰 ✅
- 慢速随机抖动（不平滑）：|jerk| 也低 ❌ —— 错判为平滑

需要至少一个 direction 维度的描述子来分辨第三种情况。

#### 2.8.2 描述子集（F1b-A / F1b-T 共用）

每个窗口 (p, f) 输出**一个小向量**而非单一 scalar，写入 `payload.factors` 时各自带 key。

| 描述子 | 公式 | 维度语义 | 取向（高 = ?）| B2 状态 |
|---|---|---|---|---|
| `jerk` | `median \|Δ²a / σ\|`（active-DOF 内）| A. 加速度变化幅值 | **risky**（高 = 非平滑）| ✅ 实现 |
| `dir` | `mean cos(v[t], v[t+1])`（active-DOF 子空间，归一化后）| B. 方向一致性 | **safe**（高 = 方向一致）| ✅ 实现 |
| `curv_radius` | `mean \|\|p[t] − centroid(window)\|\|`（active-DOF 子空间内归一化后的 N-dim "位置"点云）| G. 窗口几何弥散度 | **non-monotonic**（中等 = 圆弧 / 平稳；极小 = 停滞；极大 = 大幅直线）| ✅ 实现 |
| `cum_disp` | `sum \|\|p[t+1] − p[t]\|\|`（active-DOF 子空间，归一化后）| H. 窗口累积路径长度 | **non-monotonic**（小 = 缓慢；大 = 快速移动；与 jerk 配合时小+低jerk = 静止 = safe）| ✅ 实现 |
| `dirvar` | `std(cos(v[t], v[t+1]))` | C. 方向分布形态 | risky（高 = 方向分布散）| ❌ NotImplementedError 占位 |
| `path` | `\|\|sum(v)\|\| / sum(\|\|v\|\|)` | D. 路径效率 | safe（高 = 直线）| ❌ NotImplementedError 占位 |
| `freq` | high-freq energy / total（per-DOF FFT）| E. 频域能量比 | risky（高 = 抖动）| ❌ NotImplementedError 占位 |
| `autocorr` | `autocorr(v, lag=1)` | F. 序列可预测性 | safe（高 = 可预测）| ❌ NotImplementedError 占位 |

**取向语义说明**：

- `safe` 描述子：`Composer` 用 `score = factor`，high score → 倾向 FULL_HIT
- `risky` 描述子：`Composer` 用 `score = -factor`（或 `1 - normalize(factor)`），high raw factor → 倾向 MISS
- `non-monotonic`：YAML `weight` 默认为 0（不进 weighted_sum），保留写入 `payload.factors` 用作分析；想入 verdict 必须显式配 `weight` + `direction: high|low|range:[lo,hi]`。这避免在不知道任务先验的情况下把 non-monotonic 描述子 force-fit 进单调聚合。

**B2 默认集 = {jerk, dir, curv_radius, cum_disp}**。前两者覆盖 magnitude / direction 两个正交维度；后两者覆盖 trajectory geometry 维度（"窗口里的运动是大幅直线、还是小幅打转、还是几乎不动"）—— 这一维与 jerk / dir 信息正交（同样的低 jerk 可对应大位移直线 vs 几乎不动的微抖，curv_radius + cum_disp 才能区分）。

**Embodiment-agnostic 注意**：`curv_radius` 与 `cum_disp` 都在 active-DOF 子空间（N-dim，N = active mask sum）内计算 N-dim 欧氏距离，不依赖 forward kinematics、不依赖 end-effector 提取，因此跨机器人 / 跨数据集零特化。物理意义不同于 Cartesian 曲率（joint-space 几何而非 task-space 几何），但作为统计描述子的判别力仍然有效。

**剩余 4 项（dirvar / path / freq / autocorr）**：留扩展位，实测发现解释力不够时再开。

#### 2.8.3 Library-level metadata（per-DOF 归一化基准）

| 字段 | 形状 / 类型 | 用途 |
|---|---|---|
| `action_sigma: Tensor[A]` | per-DOF std over all entries | jerk / dir 计算前的 z-score 分母 |
| `action_active_mask: Tensor[A] (bool)` | `action_sigma >= ε_a`（默认 0.01） | 聚合 / cos 计算时跳过零填充 DOF |
| `state_sigma: Tensor[S]` | per-DOF std over all entries | 同上，state 侧 |
| `state_active_mask: Tensor[S] (bool)` | `state_sigma >= ε_s`（默认 0.01） | 同上，state 侧 |

存放位置：

- **离线 artifact pkl**：作为 top-level dict 字段（与现有 `key_builder_type` / `vector_dims` / `entries` 同级）—— 老 artifact 缺字段时 F1b 自动跳过 / log warn 并按需现算
- **在线 server 启动时**：从加载的 artifact metadata 取；若 artifact 不带（旧文件），server 启动时扫一遍 entries 现算并 cache 到内存（一次性，非 hot path）

#### 2.8.4 计算流程（OfflineWriter 内部）

```python
# Step 1: per-DOF z-score（"位置"序列 p、速度 v、jerk j 都在 z-score 后空间）
p_norm = a                          / sigma.clamp_min(eps)     # [T, D]    ("位置"点)
v_norm = (a[t+1] - a[t])            / sigma.clamp_min(eps)     # [T-1, D]
j_norm = (a[t+1] - 2*a[t] + a[t-1]) / sigma.clamp_min(eps)     # [T-2, D]

# Step 2: restrict to active DOFs
p_act  = p_norm[..., active_mask]                              # [T,   A_active]
v_act  = v_norm[..., active_mask]                              # [T-1, A_active]
j_act  = j_norm[..., active_mask]                              # [T-2, A_active]

# Step 3: per-window descriptors
for (p, f) in windows:
    sl = slice(idx-p, idx+f+1)
    # A. magnitude — median over time, mean over active DOF
    jerk_factor = j_act[sl].abs().median(dim=0).values.mean()
    # B. direction — cosine of consecutive velocity vectors in active subspace
    v1, v2 = v_act[sl][:-1], v_act[sl][1:]
    cos = F.cosine_similarity(v1, v2, dim=-1)                  # [W-1]
    dir_factor = cos.mean()
    # G. geometric dispersion — mean distance from window points to centroid
    pts       = p_act[sl]                                      # [W, A_active]
    centroid  = pts.mean(dim=0, keepdim=True)                  # [1, A_active]
    curv_rad  = (pts - centroid).norm(dim=-1).mean()
    # H. cumulative path length — sum of consecutive Euclidean distances
    cum_disp  = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
    factors[f"f1b_a_jerk__p{p}_f{f}"]        = float(jerk_factor)
    factors[f"f1b_a_dir__p{p}_f{f}"]         = float(dir_factor)
    factors[f"f1b_a_curv_radius__p{p}_f{f}"] = float(curv_rad)
    factors[f"f1b_a_cum_disp__p{p}_f{f}"]    = float(cum_disp)
```

**为什么 median (jerk)**：吸收 gripper 双稳态的单帧脉冲。
**为什么 mean (dir)**：方向一致性是分布性指标，mean 反映平均一致度；后续 dirvar 描述子专门捕分布形态。
**为什么 curv_radius + cum_disp 同时存**：两者与窗口运动幅度耦合方式不同 —— `cum_disp` 是路径总长（直线小幅 vs 直线大幅、抖动小幅 vs 抖动大幅都能区分），`curv_radius` 是点云对中心的弥散（直线运动两端远离中心 → 大；停滞 → 小；圆弧 → 中等）。两者组合可分辨"快速直线 / 缓慢直线 / 停滞 / 圆弧 / 微抖"。

#### 2.8.5 配置入口（YAML）

实际 schema 见 §8.2.6 `JudgeConfig.factors: list[FactorConfig]`。下面是 composite judge 中 F1b-A / F1b-T 的 YAML 示例（`source` 不出现 —— 由 type 名 `f1b_a` / `f1b_t` 通过 thin subclass 决定，§2.2）：

```yaml
checkpoints:
  cp1:
    judge:
      type: composite
      factors:
        - type: f1b_a
          params:
            windows: [{past: 0, future: 5}, {past: 0, future: 10}, {past: 5, future: 5}]
            descriptors: [jerk, dir, curv_radius, cum_disp]   # B2 默认
            active_eps: 0.01                                  # σ_d 阈值；< 此值视为 inactive
        - type: f1b_t
          params:
            windows: [{past: 0, future: 5}, {past: 0, future: 10}, {past: 5, future: 5}]
            descriptors: [jerk, dir, curv_radius, cum_disp]
            active_eps: 0.01
      composer:
        type: weighted_sum
        weights: { ... }
        tier_thresholds: { full_hit: 0.80 }   # warm_start 可选
        directions: { ... }                   # non_monotonic key 必填，§2.8.2
      normalizer:
        type: percentile_rolling
        window_size: 200
        cold_start_strategy: force_miss
```

`windows` 内每项 `{past, future}` dict 由 `_normalize_windows` (§8.1.6) 转为 `(int, int)` tuple。未列入 `descriptors` 的项不计算；列入但 B2 未实现的（dirvar / path / freq / autocorr）→ config 校验阶段 `raise NotImplementedError`，避免静默失败。

**B0 拒收**：`type=composite` 在 B0 直接被 validator 拒收（§8.2.6 `_JUDGE_TYPES` 不含 `"composite"`，B1 启用 algorithm 时同 commit 加入）；本节示例适用于 B1 之后。

#### 2.8.6 存储成本估算

每条 entry：`4 描述子 × 2 因子族 (A/T) × N 窗口` 个 float。N=3 时 24 个 float ≈ 100 字节，相对 `payload.action_chunk [10, 32]` 的 1.3KB 仍可忽略。开 dirvar / path 各加 24 个 float —— 仍可忽略。

#### 2.8.7 跨域适应

换数据集 / 换机器人 → artifact rebuild 自动重算 σ_d / active_mask → F1b 代码完全不动。Schema-aware dim grouping（"dim 0-6 是关节、dim 7 是 gripper"）作为 opt-in 字段 `dim_groups` 在 config schema 占位，B2 实现时 `raise NotImplementedError`，等真有跨域漏判证据再开。

#### 2.8.8 NaN / 缺字段 / cold-start / 边界 处理规则

**NaN 来源与传播**：

| 来源 | Extractor 返回值 | Composer 行为 |
|---|---|---|
| `payload.factors is None`（老 entry 或写入跳过）| 该因子的所有描述子 = `nan` | weighted_sum: 跳过该 key（不计入和与权重和）；and-gate: 视为不通过 → MISS；or-gate: 视为通过被忽略，看其他因子 |
| 描述子分母为 0（如静止窗口算 dir）| 该描述子 = `nan` | 同上 |
| 窗口越界（episode 起始/终止 K 步内）| 该描述子 = `nan`（短窗口规则见下）| 同上 |
| Extractor 内部异常（implementation bug）| Extractor 自身 raise；CompositeJudge 不吞异常 | Orchestrator log + 该 verdict 退化为 MISS（不影响其他 verdict）|

**Cold-start（normalizer 滚窗未填满）**：

`PercentileRollingNormalizer` 配置项 `cold_start_strategy: Literal["force_miss", "passthrough", "lenient"]`：
- `force_miss`（默认）：window_size 满之前 normalizer 返回 **all-NaN dict**（每个 bound key 映射到 `nan`）；CompositeJudge 检测 all-NaN 即短路 MISS，不调 Composer
- `passthrough`：直接用 raw factor（normalize 跳过）—— 用于实验阶段对比 normalize 与不 normalize
- `lenient`：用当前已积累样本算 percentile（即使 N < window_size），N < 10 时按 `force_miss` 同样规则返回 all-NaN

**Cold-start 信号通路**（contract）：normalizer 不需要新增 status 字段；"force_miss / 信号不足"统一通过 **all-NaN return** 表达。CompositeJudge 在 `__call__` 中：

```python
if norm and all(math.isnan(v) for v in norm.values()):
    return JudgeResult(HitType.MISS)
```

短路 MISS 不依赖任何 Composer 类型对 all-NaN 的内在处理（避免不同 Composer 对 all-NaN 行为不一致）。

**边界窗口（episode 起始 < past_W 步 / 终止 < future_W 步）**：

OfflineWriter `compute_for_episode(entries)` 边界处理：
- `entry_idx < past_W` → 该窗口的所有描述子写入 `nan`（不抛错）
- `entry_idx + future_W >= len(entries)` → 同上
- 这样 entry 总有 `payload.factors` 字段，OnlineExtractor 读取时按统一 NaN 路径处理，不需要分支
- 短窗口（如 `(0, 2)`）大概率不会触发边界，长窗口（如 `(5, 10)`）头尾若干 entry 会有 NaN

**Composer 读取规则统一**：

CompositeJudge 收到的 `factors: dict[str, float]` 中 `nan` 表示"该信号缺失或无效"，`-inf` / `+inf` 视为 hard-MISS / hard-FULL-HIT 信号（保留语义但 B2 不主动产生）。Composer 必须文档化各类型对 NaN 的处理。

### 2.9 F1a-A 设计动机

**理论锚点（通用 VLA 连续性假设）**：

VLA policy 在相邻控制周期对相似输入产出的 action 之间存在天然连续性 —— 这是 policy 作为 Lipschitz-连续函数的直接推论，加上"相邻 obs 在物理上演化连续"这一前提：

```
||obs[t+1] - obs[t]|| 小（物理连续）
∧ ||π(o1) - π(o2)|| ≤ L · ||o1 - o2||（policy Lipschitz）
⇒ ||π(obs[t+1]) - π(obs[t])|| 小（输出连续）
```

也就是：**如果走 inference，下一步 action 与当前 action 之间会有可预期的小变化**。

**反推 F1a-A 的判定语义**：

cache retrieval 的候选 action 来自另一条 trajectory 的某个 chunk，**与当前 episode 的 action 历史没有因果联系**，因此不受上述 Lipschitz 推论约束。

F1a-A 测的不是"机器人物理动作抖不抖"，而是 **"用 cache 替换 inference 是否引入了 inference 自身不会产生的 action 不连续"**。这种 discontinuity 是 retrieval 的人工产物，不是任务本身的需要 —— 是真正应该被视为风险的信号。

**与 F1a-T 的设计对照**（两者并排做，靠实验回答各自的信号独立性）：

| 维度 | F1a-A | F1a-T |
|---|---|---|
| 锚点信号 | inference 输出连续性（policy Lipschitz）| 物理状态平滑性（机械连续）|
| 历史侧噪声 | action 是命令，**精确已知** | state 是传感器，可能含噪 |
| 候选侧与搜索分的关系 | action **不是** 检索字段（query 用 vision/prompt/state，不用 action）→ 提供搜索分以外的独立信号 | state **是** 检索字段 → 在 position-level 可能与搜索分高度重叠（待 B1 验证）|
| 不连续的语义 | "retrieval 引入了 inference 不会产生的 action 跳变" → 直接威胁 | "state-space 不连续" → 可能是物理事件（接触 / 滑脱），可能只是 retrieval 噪声放大 |

第三行是关键差异点：F1a-A 测的是搜索引擎完全没考虑的维度（action space）；F1a-T 与搜索引擎的 robot_state 字段在 position-level 可能重叠，但 velocity-level（K≥5 窗口）理论上能捕"action 平滑但 state 突变"的物理事件 —— 是否带来独立信号由 B1 数据回答（见 §7 待讨论）。

**操作化对应**（与 §2.8 描述子集对齐 —— `RuntimeContinuity` 复用 `SourceWindowSmoothness` 的同一组描述子）：

- `jerk` 描述子：测 "candidate first action vs last executed action" 的归一化二阶差分 —— inference 跑就不会出现的尖峰
- `dir` 描述子：测 last-K executed action velocity 与 candidate first-K action velocity 之间的方向一致性 —— inference 的连续性会保证 cos ≈ 1，retrieval 不会
- `curv_radius` / `cum_disp`：覆盖 "拼接窗口里的 action 是大幅直线 / 缓慢直线 / 停滞 / 圆弧 / 微抖" 的 trajectory geometry 维度

四个描述子各从一个维度回答 "如果不用 cache，inference 会给我一个与现在多接近的 action？"

## 3. 因子 ↔ 协议矩阵

| 因子 | OfflineWriter | OnlineExtractor | required_top_k | 数据源 | 备注 |
|---|---|---|---|---|---|
| F1a-A | — | ✅ | 0 | winner `payload.action_chunk[0]` + 已执行 action 历史 | 候选 action 衔接；`RuntimeContinuityAction`（thin subclass over `_RuntimeContinuityBase`，§2.2 / §8.1.5） |
| F1a-T | — | ✅ | 0 | winner `query_keys["robot_state"]` + `view.walk_next(winner_id, k)` 取后续 entry 的 state | `RuntimeContinuityState`（thin subclass over `_RuntimeContinuityBase`）。**信号有效性待实验观察**：`robot_state` 是检索字段（搜索分已强制保证 winner state position-level 相似），F1a-T 与搜索分可能在 position 维度高度重合；velocity-level（K≥5 窗口）理论上能捕"action 平滑 + state 突变"的物理事件（接触 / 滑脱）。是否带来搜索分以外的独立信号需 B1 数据回答 |
| F1b-A | ✅ | ✅ | 0 | OfflineWriter 读链上 `entries[i].payload.action_chunk[0]` 序列；OnlineExtractor 仅读 `payload.factors` | 输出描述子向量 `{jerk, dir, curv_radius, cum_disp}` × N 窗口（§2.8）；归一化用 library `action_sigma + active_mask`；`SourceWindowSmoothnessAction`（thin subclass over `_SourceWindowSmoothnessBase`，§2.2 / §8.1.6） |
| F1b-T | ✅ | ✅ | 0 | OfflineWriter 读链上 `entries[i].query_keys["robot_state"]` 序列；OnlineExtractor 仅读 `payload.factors` | 输出描述子向量 `{jerk, dir, curv_radius, cum_disp}` × N 窗口（§2.8）；归一化用 library `state_sigma + active_mask`；`SourceWindowSmoothnessState`（thin subclass over `_SourceWindowSmoothnessBase`，§2.2 / §8.1.6） |
| F2 | — | ✅ | **K（YAML，默认 5）** | top-k 个 `payload.action_chunk` 经 `view.get_many` | 驱动 strategy `min_top_k_hint`（§2.7）|

`CompositeJudge`（在线 server 进程）与离线 build pkl 脚本（经 `enrich_artifact_with_factors` helper，§2.5）都通过 `factors/registry.py` 取因子实例；factor 模块本身不知道自己被谁调用，从而离线写入与在线读取共享同一份配置与同一份代码。

## 4. 实施分批

| 批次 | 内容 | Orchestrator winner fetch | 备注 |
|---|---|---|---|
| **B0** | Factor / Composer / Normalizer Protocol + registry + 全部 factor 模块文件作为**导入安全空 stub**（`runtime_continuity.py` / `source_window.py` / `consensus.py` 文件存在但无 class 定义，便于 registry `from . import` 不 ImportError）+ `payload.factors` schema + CompositeJudge **类骨架**（无任何因子注册到 registry）+ PayloadView 接口与无-fork 实现 + **`InMemoryBackend.fetch_entry` 普通 public method（非 ABC override）+ `CacheStorage.fetch_entry` duck-typed facade**（Backend ABC **真正不动**）+ `InMemoryBackend.library_stats` attr 默认 None + `CacheStorage.library_stats` duck-typed facade accessor + Judge protocol 加 default-None kwargs + 老 Judge 加 `**kwargs` 吸收 | **不动**，仍走 `storage.fetch_payload(winner_id)` | **B0 严格定义"零运行时行为变化"**：所有改动都是新增 / 默认 None / 默认 no-op；老 YAML 不可能进入新代码路径。Orchestrator `check()` **不**注入 view+history（移到 B1）。CompositeJudge 即便构造也跑不起来（无 factor 注册）。Schema 测试与 round-trip 验证在此完成 |
| **B1** | `RuntimeContinuityAction` (= F1a-A) 与 `RuntimeContinuityState` (= F1a-T) + F2 + Orchestrator `check()` 注入 view+history + `_state_history` 维护 + winner fetch rewire 走 `view.get(winner_id)` + strategy `min_top_k_hint` wiring | rewire 走 PayloadView | 第一个真正"启用 CompositeJudge"的批次；fetcher / history 通路在此被首次执行；rewire 自带可观测收益（view memo 去重）；F1a-A 与 F1a-T 共享 `_RuntimeContinuityBase` 算法实现（thin subclass 仅 override class attribute），ship 成本几乎为零，便于实验阶段对比信号独立性 |
| **B2** | F1b-A + F1b-T 离线写入 + 在线读取 + library σ_d/active_mask metadata（§2.8）+ `LibraryStats` dataclass + artifact 重建工具（§2.5 三脚本）+ `_build_entry_chain` 调 OfflineWriter + InMemoryBackend `library_stats` 加载 / fallback | 不变 | 触发 build pkl 改动；旧 artifact 缺 metadata 时 server 启动现算 fallback |

**B1 fetch rewire 的成本评估**（已对调用面做实测；详细 wiring 见 §8.2.3 / §8.2.7）：

- production 唯一调用点：`orchestrator.py:378`，1 行替换（`storage.fetch_payload(winner_id)` → `view.get(winner_id)`）
- View 生命周期管理：约 5 行（`check()` 入口创建、注入 Judge、出口丢弃）
- `CacheStorage.fetch_payload` / `search_and_fetch` / Backend ABC 全部不删 —— PayloadView 内部仍依赖 storage 原语
- 测试：`tests/cache/conftest.py CountingStorage` 与 `tests/cache/test_orchestrator.py:138 test_judge_miss_does_not_fetch_payload` 都通过 storage 计数器工作，view → storage 透传后**计数器仍然递增**，老测试零修改
- `tests/cache/test_cache_storage.py` prefill mode：view.get → storage.fetch_payload → prefill 短路链路完整

净改动：production ≤ 10 行、测试 0 行。Composer 默认采用 S1（加权和 + percentile）作为首发；S2 / S3 留扩展位（§2.8 描述子取向语义说明）。

## 5. 测试策略

设计层覆盖范围概览（具体测试文件路径与每文件覆盖范围见 §8.5）：

- **单元**：每个 factor 类的 `compute_for_episode` / `extract` 用合成轨迹验证数值；`factors/registry.py` build/known 双向。
- **Composite**：Composer 取向翻转 / Normalizer 冷启动 / 缺字段 NaN 行为（规则见 §2.8.8）。
- **集成**：CompositeJudge 与 Orchestrator 走通 FULL_HIT / WARM_START / MISS 三路；老 `ThresholdJudge` 配置零变化仍然通过。
- **写入路径**：`_build_entry_chain` 在配置 / 未配置 OfflineWriter 两种情形分别 round-trip。
- **PayloadView**：`get` / `walk_prev` / `walk_next` 在无-fork 链上的正确性；遇 `len(prev_ids|next_ids) > 1` 触发 `NotImplementedError` 而非静默选择；`cross_trajectory=True` 与非 `TRAJECTORY` 的 fork policy 同样 raise。
- **B1 fetch rewire**：CountingStorage 验证 `view.get(winner_id)` + Judge 端任何 `view.get(winner_id)` 之间的 memo dedup（同一 verdict 内 storage 计数 ≤ 1）。
- **History lifecycle**：CP1-only / CP1+CP3 / gate-skip / FULL_HIT / WARM_START / episode reset / WebSocket 中断 共 6 路（详细列表与文件路径见 §8.2.3 与 §8.5）。
- **YAML round-trip**：composite judge 含 `factors: list[FactorConfig]` 的 YAML → `_dict_to_dataclass` → 验证 `cfg.checkpoints["cp1"].judge.factors[0]` 是 `FactorConfig` 而非 dict（§8.2.6）。

## 6. 风险（汇总）

风险列表（按已解决 / 仍开放分类）：

| 风险 | 状态 | 解决章节 |
|---|---|---|
| Top-k payload 取数开销 | ✅ 已解决 | §2.7 / §8.2.4：strategy `min_top_k_hint` + PayloadView 按需 `get_many`，in-memory 实测无可观测开销 |
| Normalizer 冷启动 fallback | ✅ 已解决 | §2.8.8 列出三策略 (`force_miss` 默认 / `passthrough` / `lenient`) |
| `payload.factors` key 命名漂移 | ✅ 已解决 | factor 类内部按 `f"f1b_{source[0]}_{descriptor}__p{p}_f{f}"` 模板生成 key（§2.8.4 / §8.1.6），写读两侧统一来自 registry，无字符串约定 |
| 老 artifact 缺 `factors` / `library_stats` 的 schema 迁移 | ✅ 已解决 | §2.5 / §8.2.5：load_artifact 检测缺字段 → fallback 现算（log warn）；OnlineExtractor 读到 None → 该因子返回 NaN（§2.8.8） |
| Build pkl 与 server 共享 factor config | ✅ 已解决 | §2.5 三脚本经同一 helper + 同一 registry，§8.3 wiring 顺序统一 |
| 是否独立 `verdict_factor_system.md` | ✅ 已决定 | §8.8.4：**不独立**，集中在 `cache_system.md` 新增 §5.11 / §5.12；WA §8 表无新增行 |
| Library `compute_from_entries` 阻塞 server startup（大 artifact）| ⚠️ 仍开放 | §8.7：log 时间统计；future 可加并行 |
| 不连续 regime 下 cache 是否安全 | ⚠️ 仍开放 | 候选文档 §5/§7 标记；属实验问题，B1 后回答 |

## 7. 待讨论（仍开放）

实验驱动 / 经验调参，不影响 plan 落地：

- **F1b 窗口 `(p, f)` 默认集合**：`{(0,5), (0,10), (5,5)}` 还是更小集？
- **F1b-A inter-step vs intra-chunk jerk**：是否同时算 inter-step（链上 `action_chunk[0]` 序列）与 intra-chunk（每 chunk 内 `action_chunk[0:H]` 序列）？短期建议只做前者（与 F1b-T 对偶），后者留扩展位。
- **B2 落地后描述子扩展**：是否需要扩 dirvar / path / freq / autocorr —— 数据驱动决定，先看 jerk + dir + curv_radius + cum_disp 的解释力。
- **不连续 regime 的 cache 安全策略**：候选文档 §5/§7 标记为开放问题；当前 plan 立场是描述子负责标记 regime，处置策略（禁用 cache vs 让 inference 接管 vs 照常）由实验回答。
- **F1a-T 信号独立性**：B1 跑完后对比 F1a-T 与搜索分（robot_state field similarity）的相关系数；窗口尺寸 K 对 velocity-level 噪声的影响（K=2 vs K=5 vs K=10）。判定 F1a-T 是否相对搜索分提供独立信号；若高度相关，下一阶段在 Composer 里降权或剔除。
- **`PercentileRollingNormalizer.window_size` 默认值**：当前 §8.2.6 暂定 200；实测后调整。
- **`active_eps` 默认 0.01 是否合适**：是否需要在 artifact build 时 log 哪些 DOF 被判 inactive 供 ops 校验？
- **Composer 默认权重来源**：YAML 硬编码 vs 标定脚本扫描？
- **Tier 映射规则**：`P_full_hit_percentile` / `P_warm_start_percentile` 默认值（B1+ 实验决定）。

设计决策已闭合（不再开放）：

- ~~首批是否包含 F1a-T~~ → 包含（B1，与 F1a-A 同 commit；§4 / §3 备注）
- ~~是否需要 `broadcast_state` lifecycle~~ → 不需要（state 在 query_keys，§2.3.3 / §2.1）
- ~~Build pkl 工具改造方式~~ → 改现有 3 脚本 + 共享 helper（§2.5）
- ~~是否独立 `verdict_factor_system.md`~~ → 不独立，集中在 `cache_system.md`（§8.8.4）
- ~~robot_state pre vs post input_transform~~ → post（用现有 query_keys，零 schema 改动；§2.1）

---

## 8. 代码级实施 Map

> **本节是 executor 的精确蓝图**。§1-§7 描述设计意图与权衡；本节描述每一个文件、类、方法、wiring 顺序的具体改动。所有 file path 都已对照当前 repo 实测；所有现有签名都引自当前代码，未来若 source 有更新需以 source 为准。

### 8.1 新建文件

#### 8.1.1 `src/openpi/cache/components/payload_view.py`

```python
"""PayloadView: per-check() lazy fetch + chain walk facade for Judge.

Wraps CacheStorage.fetch_payload + CacheStorage.fetch_entry (the latter
duck-types backend's optional fetch_entry capability — see §8.2.7). Judge
gets read-only access to candidate payload + neighbor entries; never holds
CacheStorage handle directly.

Coupling map:
  DEPENDS ON:  cache_storage.py (CacheStorage.fetch_payload, CacheStorage.fetch_entry)
  CONSUMED BY: CompositeJudge (via Orchestrator.check kw-only injection)
  IF CHANGED:  CompositeJudge factor extractors may need adaptation
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional, Protocol, runtime_checkable

import torch

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import CacheEntry, CachePayload


class ForkPolicy(Enum):
    TRAJECTORY = auto()
    FIRST = auto()         # B0 raise NotImplementedError
    STOP = auto()          # B0 raise NotImplementedError
    ALL_BRANCHES = auto()  # B0 raise NotImplementedError
    SCORE = auto()         # B0 raise NotImplementedError


@runtime_checkable
class PayloadView(Protocol):
    def get(self, entry_id: str) -> CachePayload: ...
    def get_entry(self, entry_id: str) -> CacheEntry: ...
    def get_many(self, entry_ids: list[str]) -> list[CachePayload]: ...
    def walk_prev(
        self, entry_id: str, k: int, *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...
    def walk_next(
        self, entry_id: str, k: int, *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...


class StoragePayloadView:
    """Default PayloadView implementation backed by CacheStorage.

    Memoizes (entry_id -> payload, entry_id -> entry) within the lifetime of
    one Orchestrator.check() call.
    """

    def __init__(self, storage: CacheStorage) -> None:
        self._storage = storage
        self._payload_memo: dict[str, CachePayload] = {}
        self._entry_memo: dict[str, CacheEntry] = {}

    def get(self, entry_id: str) -> CachePayload:
        if entry_id not in self._payload_memo:
            self._payload_memo[entry_id] = self._storage.fetch_payload(entry_id)
        return self._payload_memo[entry_id]

    def get_entry(self, entry_id: str) -> CacheEntry:
        if entry_id not in self._entry_memo:
            # Goes through the facade method `CacheStorage.fetch_entry`
            # (§8.2.7), which duck-types the backend's optional `fetch_entry`
            # capability. InMemoryBackend declares this as a plain public
            # method (§8.2.5) — NOT an ABC override; the Backend ABC stays
            # unchanged. Backends without the capability (Qdrant) cause the
            # facade to raise NotImplementedError; `validate_cache_config`
            # (§8.2.6) hard-rejects composite judges that need chain walking
            # against such backends.
            self._entry_memo[entry_id] = self._storage.fetch_entry(entry_id)
        return self._entry_memo[entry_id]

    def get_many(self, entry_ids: list[str]) -> list[CachePayload]:
        return [self.get(eid) for eid in entry_ids]

    def walk_prev(self, entry_id, k, *, fork_policy=ForkPolicy.TRAJECTORY, cross_trajectory=False):
        return self._walk(entry_id, k, "prev_ids", fork_policy, cross_trajectory)

    def walk_next(self, entry_id, k, *, fork_policy=ForkPolicy.TRAJECTORY, cross_trajectory=False):
        return self._walk(entry_id, k, "next_ids", fork_policy, cross_trajectory)

    def _walk(self, entry_id, k, link_attr, fork_policy, cross_trajectory):
        if cross_trajectory:
            raise NotImplementedError("cross_trajectory=True not implemented in B0")
        if fork_policy is not ForkPolicy.TRAJECTORY:
            raise NotImplementedError(f"fork_policy={fork_policy.name} not implemented in B0")
        result: list[CacheEntry] = []
        cur = self.get_entry(entry_id)
        anchor_traj_id = cur.trajectory_id
        for _ in range(k):
            link_ids: list[str] = getattr(cur, link_attr)
            if len(link_ids) == 0:
                break
            if len(link_ids) > 1:
                raise NotImplementedError(
                    f"fork detected at {cur.id}: {link_attr}={link_ids}; "
                    "fork_policy=TRAJECTORY anchor selection not implemented in B0"
                )
            nxt = self.get_entry(link_ids[0])
            if nxt.trajectory_id != anchor_traj_id:
                # Trajectory boundary — TRAJECTORY policy stops here
                break
            result.append(nxt)
            cur = nxt
        return result
```

#### 8.1.2 `src/openpi/cache/components/factors/__init__.py`

Empty namespace; re-export public names from submodules.

#### 8.1.3 `src/openpi/cache/components/factors/base.py`

```python
"""Factor protocols and shared library-level statistics.

A factor may implement OnlineExtractor (verdict-time, on cache hit candidates),
OfflineWriter (artifact-build / episode-end, populates payload.factors), or both.

Coupling map:
  DEPENDS ON:  storage_types.py (CacheEntry, CachePayload, SearchResultLite),
               components/payload_view.py (PayloadView)
  CONSUMED BY: CompositeJudge (OnlineExtractor),
               CacheOrchestrator._build_entry_chain + offline build pkl tool
                 (OfflineWriter)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch

from openpi.cache.components.payload_view import PayloadView
from openpi.cache.storage_types import CacheEntry, SearchResultLite


@dataclass
class LibraryStats:
    """Per-DOF library-wide statistics for F1b normalization.

    Tensors live on CPU, float32. Computed once per artifact (or by
    InMemoryBackend at server startup if missing from artifact).
    """
    action_sigma:        torch.Tensor    # [action_dim]
    action_active_mask:  torch.Tensor    # [action_dim] bool
    state_sigma:         torch.Tensor    # [state_dim]
    state_active_mask:   torch.Tensor    # [state_dim] bool

    @classmethod
    def compute_from_entries(
        cls, entries: list[CacheEntry], active_eps_action: float = 0.01,
        active_eps_state: float = 0.01,
    ) -> "LibraryStats":
        # Stack action_chunk[0] across all entries to get [N, A]
        # Stack query_keys["robot_state"] across all entries to get [N, S]
        # Compute per-DOF std; build active_mask via threshold.
        ...


@dataclass
class HistoryView:
    """Per-episode action / state history snapshot, injected into Judge."""
    actions: list[torch.Tensor]    # newest-last; from Orchestrator.broadcast_action
    states:  list[torch.Tensor]    # newest-last; from Orchestrator.record_query_keys


@runtime_checkable
class OnlineExtractor(Protocol):
    """Verdict-time factor extraction. Pure function (no I/O side effect).

    Two-tier metadata model:

    **Class-level capability flags** (read by validator BEFORE instantiation,
    no params needed):
      - required_top_k:           default min top_k for this factor type. May
                                  be overridden by per-instance `params.top_k`.
      - requires_library_stats:   whether the constructor needs `library_stats`
                                  kwarg injected by config builder (`_build_judge`).
                                  True for F1a-A / F1a-T / F1b-A / F1b-T (all use
                                  z-score against library σ); False for F2.
      - requires_chain_walk:      whether `extract()` calls `view.walk_prev` or
                                  `view.walk_next`. True for F1a-T (walks state
                                  via `view.walk_next` per §3); False for the
                                  rest. Used by validator to fail-fast when
                                  backend lacks `fetch_entry` capability.

    **Instance-level metadata** (set in __init__ from params, read by
    CompositeJudge after instantiation):
      - descriptor_orientations:  {key -> "safe"|"risky"|"non_monotonic"} for
                                  every key this *instance* will produce.
                                  Keys depend on params (e.g. F1b's keys
                                  encode `descriptors × windows × source`),
                                  so they cannot be class-level.

    **Static key introspection** (for validator, before instantiation):
      - describe(params) -> dict[str, str]: classmethod that computes the same
                                  key→orientation map from params alone, no
                                  library_stats needed. validator uses this to
                                  cross-check ComposerConfig.directions coverage
                                  for non_monotonic keys without instantiating
                                  the factor (which would need library_stats
                                  for F1b before backend is built).
                                  Instance __init__ should call
                                  `self.descriptor_orientations = self.__class__.describe(params)`.
    """

    required_top_k: int = 0
    requires_library_stats: bool = False
    requires_chain_walk: bool = False
    descriptor_orientations: dict[str, str]

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        """Compute descriptor_orientations from constructor params alone.

        Pure function over `params` — no library_stats / no I/O. Allows the
        validator to inspect what keys an instance WILL produce without
        constructing it. The classmethod and __init__ MUST agree:
        `instance.descriptor_orientations == cls.describe(params)`.
        """
        ...

    def extract(
        self,
        results: list[SearchResultLite],
        view: PayloadView,
        history: HistoryView,
        cached_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Return {key: value} factor descriptors for this verdict.

        Returned dict's key set MUST equal self.descriptor_orientations.keys()
        (CompositeJudge.__call__ asserts this — see §8.2.2 — otherwise
        Normalizer state, Composer weights, and orientation checks all
        silently desynchronize). Values may be NaN per §2.8.8 (missing-factor
        / boundary / cold-start).
        """
        ...


@runtime_checkable
class OfflineWriter(Protocol):
    """Artifact-build / episode-end factor computation."""

    def required_payload_fields(self) -> set[str]:
        """Extra payload fields this writer requires (B2 default: empty set)."""
        ...

    def compute_for_episode(
        self,
        entries: list[CacheEntry],
        library_stats: LibraryStats,
    ) -> list[dict[str, float]]:
        """Return per-entry factor dict (parallel to `entries`).
        Caller merges into entries[i].payload.factors.
        """
        ...
```

#### 8.1.4 `src/openpi/cache/components/factors/registry.py`

```python
"""Factor registry: name -> class.

Imports the three concrete-factor submodules at module load so their
@register(...) decorators run. The submodules MUST exist as importable
modules at every batch (B0 ships them as empty stubs to avoid ImportError;
B1/B2 fill in the class definitions + decorator calls).
"""

from __future__ import annotations
from typing import Any

# Populated by submodules via @register decorator at import time.
_REGISTRY: dict[str, type] = {}

def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"Factor name '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco

def get_class(name: str) -> type:
    """Return the registered class without instantiating. Used by config
    builder to introspect class-level capability flags
    (e.g. `requires_library_stats`) before deciding what kwargs to pass
    into the constructor."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown factor name '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]

def build(name: str, **kwargs) -> Any:
    return get_class(name)(**kwargs)

def known() -> set[str]:
    return set(_REGISTRY)

# Force submodule imports so @register decorators run.
# In B0 these modules exist as empty stubs (no class, no register call).
# In B1 runtime_continuity.py + consensus.py register their classes.
# In B2 source_window.py registers its class.
from . import runtime_continuity, source_window, consensus  # noqa: E402, F401
```

#### 8.1.5 `src/openpi/cache/components/factors/runtime_continuity.py`

**Architecture (per §2.2 thin-subclass model)**：base class `_RuntimeContinuityBase` 持全部算法实现 + describe 模板；两个 thin subclass `RuntimeContinuityAction` / `RuntimeContinuityState` 各自设 class-level `source` + `requires_chain_walk`。`source` 不出现在 YAML params 中 —— 由 registry type 名（`f1a_a` / `f1a_t`）唯一决定 → 命中对应 thin subclass → `cls.source` 已绑定。

**B0 ships**（**完整 metadata 层**：所有类定义 + capability flags + describe + __init__；只有 `extract` body 占位 raise NotImplementedError）：

```python
class _RuntimeContinuityBase:
    """Base. Subclasses set source + key_initial + requires_chain_walk class attributes."""
    source: str           # semantic data source ("action" / "state"), set per subclass
    key_initial: str      # key namespace ("a" / "t"), aligned with registry name suffix
    requires_library_stats: bool = True   # F1a 用 z-score（§2.8.4），需要 library σ
    requires_chain_walk: bool             # set per subclass
    required_top_k: int = 0

    def __init__(self, window_k: int, descriptors: list[str], library_stats: "LibraryStats"):
        self._window_k = window_k
        self._descriptors = descriptors
        self._library_stats = library_stats
        # instance-level orientation map computed via classmethod (B0 ready)
        self.descriptor_orientations = self.__class__.describe(
            {"window_k": window_k, "descriptors": descriptors}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        # source 与 key_initial 都来自 cls，不来自 params。
        # key_initial 必须与 registry 名后缀（"a" / "t"）一致，
        # 否则 ComposerConfig.weights 与 extractor 输出 key 会失配
        # （见 G2 Round 1 → Round 2 修订）。
        prefix = f"f1a_{cls.key_initial}"   # f1a_a / f1a_t
        from openpi.cache.components.factors.source_window import _DESCRIPTOR_ORIENTATIONS
        return {f"{prefix}_{d}": _DESCRIPTOR_ORIENTATIONS[d] for d in params["descriptors"]}

    def extract(self, results, view, history, cached_data):
        raise NotImplementedError("RuntimeContinuity.extract: B1 algorithm")


@register("f1a_a")
class RuntimeContinuityAction(_RuntimeContinuityBase):
    source = "action"
    key_initial = "a"
    requires_chain_walk = False   # 仅看 payload.action_chunk[0] + history.actions

@register("f1a_t")
class RuntimeContinuityState(_RuntimeContinuityBase):
    source = "state"
    key_initial = "t"
    requires_chain_walk = True    # 调 view.walk_next 取后续 entry state（§3）
```

**B1 ships**：`extract` body 完整算法（§2.8.4 jerk / dir / curv_radius / cum_disp 实现）。

#### 8.1.6 `src/openpi/cache/components/factors/source_window.py`

**Architecture**：同 §8.1.5 thin subclass 模式。base class `_SourceWindowSmoothnessBase` 实现 OnlineExtractor + OfflineWriter 全部算法；`SourceWindowSmoothnessAction` / `SourceWindowSmoothnessState` 各自设 class-level `source` (语义) + `key_initial`（key namespace，与 registry 名后缀对齐）；registry 注册 `f1b_a` / `f1b_t`。

**Window 表示规范化（Round 11 Item 1）**：YAML 写 `windows: [{past: 0, future: 5}, ...]`（list of dict，自描述），内部规范表示为 `list[tuple[int, int]]`（紧凑、便于数学）。两种形式之间的转换由 module-level helper `_normalize_windows` 集中处理；**`__init__` / `describe()` / validator 三处全部经过这个 helper**，永远不直接 unpack 入参 windows。

**B0 ships**（完整 metadata + 类定义 + register；`extract` / `compute_for_episode` body 占位）：

```python
_DESCRIPTOR_ORIENTATIONS: dict[str, str] = {
    "jerk":        "risky",
    "dir":         "safe",
    "curv_radius": "non_monotonic",
    "cum_disp":    "non_monotonic",
    # B2 后扩 dirvar / path / freq / autocorr
}

def _normalize_windows(windows) -> list[tuple[int, int]]:
    """Normalize YAML / params 'windows' to list[tuple[int, int]].

    Accepts either:
      - list[dict{"past": int, "future": int}]  (YAML / FactorConfig.params shape)
      - list[tuple[int, int]] / list[list[int]] (already-normalized internal shape)

    Used in three places — keep them in sync:
      - _SourceWindowSmoothnessBase.__init__
      - _SourceWindowSmoothnessBase.describe (classmethod, validator path)
      - validate_cache_config (when computing describe-based directions coverage)
    """
    out: list[tuple[int, int]] = []
    for w in windows:
        if isinstance(w, dict):
            out.append((int(w["past"]), int(w["future"])))
        else:
            p, f = w
            out.append((int(p), int(f)))
    return out


class _SourceWindowSmoothnessBase:
    source: str           # semantic data source ("action" / "state"), set per subclass
    key_initial: str      # key namespace ("a" / "t"), aligned with registry name suffix
    requires_library_stats: bool = True
    requires_chain_walk: bool = False
    required_top_k: int = 0

    def __init__(
        self,
        windows,                              # list[dict] OR list[tuple] — both accepted
        descriptors: list[str],
        active_eps: float,
        library_stats: "LibraryStats",
    ):
        self._windows = _normalize_windows(windows)   # normalized internal: list[tuple]
        self._descriptors = descriptors
        self._active_eps = active_eps
        self._library_stats = library_stats
        # describe takes the normalized form (consistency with __init__)
        self.descriptor_orientations = self.__class__.describe(
            {"windows": self._windows, "descriptors": descriptors}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        # key_initial 与 registry 名后缀对齐（"a" / "t"）；source 是语义字段，
        # 不参与 key 派生 —— 见 G2 Round 1 → Round 2 处置。
        prefix = f"f1b_{cls.key_initial}"   # f1b_a / f1b_t
        windows = _normalize_windows(params["windows"])
        out = {}
        for d in params["descriptors"]:
            for (p, f) in windows:
                out[f"{prefix}_{d}__p{p}_f{f}"] = _DESCRIPTOR_ORIENTATIONS[d]
        return out

    def extract(self, results, view, history, cached_data):
        raise NotImplementedError("SourceWindowSmoothness.extract: B2 algorithm")

    def compute_for_episode(self, entries, library_stats):
        raise NotImplementedError("SourceWindowSmoothness.compute_for_episode: B2 algorithm")

    def required_payload_fields(self) -> set[str]:
        return set()    # 当前实现不需要新 raw payload 字段


@register("f1b_a")
class SourceWindowSmoothnessAction(_SourceWindowSmoothnessBase):
    source = "action"
    key_initial = "a"

@register("f1b_t")
class SourceWindowSmoothnessState(_SourceWindowSmoothnessBase):
    source = "state"
    key_initial = "t"
```

**B2 ships**：`extract` body（仅读 `view.get(winner_id).factors`）+ `compute_for_episode` body（z-score / active_mask / windows / descriptors per §2.8.4）。

#### 8.1.7 `src/openpi/cache/components/factors/consensus.py`

**B0 ships**（完整 metadata + 类定义 + register；`extract` body 占位）：

```python
@register("f2")
class TopKActionConsensus:
    source = None    # F2 无 source 维度
    requires_library_stats: bool = False   # candidate-pool 内 variance scale-invariant
    requires_chain_walk: bool = False

    def __init__(self, K: int):
        self.K = K
        self.required_top_k = K
        self.descriptor_orientations = self.__class__.describe({"K": K})

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"f2_var": "risky"}    # 高 variance = 共识低 = 风险

    def extract(self, results, view, history, cached_data):
        raise NotImplementedError("TopKActionConsensus.extract: B1 algorithm")
```

**B1 ships**：`extract` body —— `view.get_many([r.id for r in results[:self.K]])`，compute action_chunk variance per DOF in active subspace, return `{"f2_var": float}`。

#### 8.1.8 `src/openpi/cache/components/factors/composers/__init__.py`

```python
"""Composer Protocol + concrete S1/S2/S3 classes.

B0 ships full interface signatures; the actual scoring algorithm bodies
land in B1+ (user decision: algorithm is later work, B0 ships the shell).
"""

from __future__ import annotations
from typing import Literal, Optional, Protocol, runtime_checkable

from openpi.cache.components.judge import JudgeResult


@runtime_checkable
class Composer(Protocol):
    """Aggregate normalized factor dict → JudgeResult.

    bind_orientations is called once by CompositeJudge.__init__ with the
    union of every extractor's `descriptor_orientations`. Composer uses
    these to decide flip-direction (safe / risky / non_monotonic).
    """

    def bind_orientations(self, orientations: dict[str, str]) -> None: ...

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        """Return JudgeResult with hit_type ∈ {FULL_HIT, WARM_START, MISS}.

        winner_id is the id Composer should attach when emitting FULL_HIT
        or WARM_START — Composer does not pick which candidate becomes
        the winner; it only decides hit-type for the already-selected one.
        """
        ...


class WeightedSumComposer:
    """S1: orientation-aware weighted percentile sum + tier thresholds.

    B0 stub: __init__ signature + bind_orientations + compose stub raising
    NotImplementedError. Full algorithm (per §2.8.8 NaN handling +
    orientation flip + tier mapping) lands in B1+.
    """

    def __init__(
        self,
        weights: dict[str, float],
        full_hit_threshold: float,
        warm_start_threshold: Optional[float] = None,
        warm_start_t: Optional[float] = None,
        directions: Optional[dict[str, str]] = None,
        # Per-key direction for non_monotonic descriptors.
        # Format: "high" | "low" | "range:[lo,hi]". Required (validated at
        # bind_orientations time) for any key whose orientation is
        # "non_monotonic" and whose weight != 0.
    ) -> None:
        ...

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        ...   # B0: store; B1+: cross-check directions vs non_monotonic keys

    def compose(self, factors, *, winner_id):
        raise NotImplementedError("WeightedSumComposer.compose: B1+ algorithm")


class AndGateComposer:
    """S2: every key (per per_factor_thresholds) must pass its threshold."""

    def __init__(
        self,
        per_factor_thresholds: dict[str, float],
        warm_start_t: Optional[float] = None,
    ) -> None:
        ...

    def bind_orientations(self, orientations: dict[str, str]) -> None: ...
    def compose(self, factors, *, winner_id):
        raise NotImplementedError("AndGateComposer.compose: B1+ algorithm")


class OrGateComposer:
    """S3: any key passing its threshold emits hit."""

    def __init__(
        self,
        per_factor_thresholds: dict[str, float],
        warm_start_t: Optional[float] = None,
    ) -> None:
        ...

    def bind_orientations(self, orientations: dict[str, str]) -> None: ...
    def compose(self, factors, *, winner_id):
        raise NotImplementedError("OrGateComposer.compose: B1+ algorithm")
```

#### 8.1.9 `src/openpi/cache/components/factors/normalizers/__init__.py`

```python
"""Normalizer Protocol + concrete PercentileRollingNormalizer.

B0 ships full interface signatures; algorithm body lands in B1+.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Normalizer(Protocol):
    """Per-key normalization (e.g. percentile rank over rolling window).

    bind_keys is called once by CompositeJudge.__init__ with the full set
    of keys all extractors will produce. Allows the normalizer to
    pre-allocate per-key state (ring buffers, running stats, etc.).
    """

    def bind_keys(self, keys: list[str]) -> None: ...

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        """Map raw factor values → normalized values (typically [0, 1]).

        NaN inputs are propagated as NaN (per §2.8.8 missing-factor rule);
        Composer is responsible for handling NaN per its own type.

        Cold-start sentinel: when the normalizer is in `force_miss` mode and
        the rolling window is not yet ready, it MUST return a dict where
        every key maps to NaN. CompositeJudge detects all-NaN and returns
        HitType.MISS directly without invoking the Composer. This makes
        force_miss representable through the existing dict[str, float]
        interface without adding a status channel.
        """
        ...

    def on_episode_start(self) -> None:
        """Lifecycle hook from Orchestrator. Default impls are no-op
        (rolling window survives across episodes for stable percentile)."""
        ...


class PercentileRollingNormalizer:
    """Per-key percentile rank over a rolling window.

    B0 stub: __init__ + bind_keys + __call__ stub raising NotImplementedError.
    Algorithm + cold_start_strategy semantics defined in §2.8.8.
    """

    def __init__(
        self,
        window_size: int = 200,
        cold_start_strategy: str = "force_miss",
        # "force_miss" | "passthrough" | "lenient" — see §2.8.8
    ) -> None:
        ...

    def bind_keys(self, keys: list[str]) -> None: ...
    def __call__(self, raw): raise NotImplementedError("B1+ algorithm")
    def on_episode_start(self) -> None: ...
```

### 8.2 修改的现有文件

#### 8.2.1 `src/openpi/cache/storage_types.py`

```diff
 @dataclass
 class CachePayload:
     action_chunk: torch.Tensor
     intermediates: Optional[dict[float, torch.Tensor]] = None
     denoising_num_steps: Optional[int] = None
     task_key: str = ""
+    factors: Optional[dict[str, float]] = None
     ...
     def validate_for_checkpoint(self, checkpoint_id):
-        if self.action_chunk is None: raise ValueError(...)
+        # No new validation needed — factors is fully optional.
+        if self.action_chunk is None: raise ValueError(...)
```

`StepRecord` 不动；`broadcast_state` 不引入。

#### 8.2.2 `src/openpi/cache/components/judge.py`

- 扩展 `SimilarityJudge.__call__` Protocol 签名加 kw-only：

```python
def __call__(
    self,
    results: list[SearchResultLite],
    checkpoint_id: CheckpointID,
    cached_data: dict[str, torch.Tensor],
    *,
    view: Optional[PayloadView] = None,        # NEW
    history: Optional[HistoryView] = None,     # NEW
) -> JudgeResult: ...
```

- 老 `ThresholdJudge` / `AlwaysHitJudge` / `AlwaysWarmStartJudge`：加 `**kwargs` 吃掉新 kwargs，行为不变。
- 新增 `CompositeJudge`：

```python
class CompositeJudge:
    def __init__(
        self,
        extractors: list[OnlineExtractor],
        composer: Composer,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        self._extractors = extractors
        self._composer = composer
        self._normalizer = normalizer
        self.min_required_top_k = max(
            (e.required_top_k for e in extractors), default=0,
        )

        # ---- Collect static metadata from extractors ----
        # union of every extractor's descriptor_orientations; conflicts (same key
        # claimed by two extractors with different orientations) are a config bug
        # and raise immediately.
        all_orientations: dict[str, str] = {}
        for ext in extractors:
            for k, ori in ext.descriptor_orientations.items():
                if k in all_orientations and all_orientations[k] != ori:
                    raise ValueError(
                        f"Composite judge factor key {k!r} has conflicting "
                        f"orientations: {all_orientations[k]!r} vs {ori!r}"
                    )
                all_orientations[k] = ori

        # ---- Bind metadata into composer / normalizer ----
        # Composer cross-checks `directions` config against non_monotonic keys
        # (per ComposerConfig validator §8.2.6) and stores the orientation map
        # for use in `compose`.
        composer.bind_orientations(all_orientations)
        # Normalizer pre-allocates per-key state (e.g. ring buffers).
        if normalizer is not None:
            normalizer.bind_keys(list(all_orientations.keys()))

    def __call__(self, results, checkpoint_id, cached_data, *, view=None, history=None):
        if not results:
            return JudgeResult(HitType.MISS)
        raw: dict[str, float] = {}
        for ext in self._extractors:
            out = ext.extract(results, view, history, cached_data)
            # Key contract enforcement (per §8.1.3 + Round 7 Item 5).
            # An extractor's runtime output keys MUST equal its declared
            # descriptor_orientations.keys(). Drift here silently desyncs
            # Normalizer state, Composer weights, and orientation lookup,
            # so we fail loud — bug surface = build/test time, not silent
            # mis-aggregation in production.
            expected = set(ext.descriptor_orientations.keys())
            actual = set(out.keys())
            if actual != expected:
                raise RuntimeError(
                    f"Extractor {type(ext).__name__} key contract violation: "
                    f"declared {expected}, returned {actual} "
                    f"(missing: {expected - actual}, extra: {actual - expected})"
                )
            raw.update(out)
        norm = self._normalizer(raw) if self._normalizer else raw
        # Cold-start sentinel: if normalizer returned all-NaN (force_miss not
        # ready), short-circuit MISS without bothering Composer. This makes
        # the cold-start path explicit, independent of how individual Composers
        # treat all-NaN inputs (per Item 2 of G1 Round 5 review).
        if norm and all(math.isnan(v) for v in norm.values()):
            return JudgeResult(HitType.MISS)
        return self._composer.compose(norm, winner_id=results[0].id)

    def on_episode_start(self):
        if self._normalizer is not None:
            self._normalizer.on_episode_start()

    def record_action(self, action_chunk):
        pass    # CompositeJudge holds no per-episode state itself
```

#### 8.2.3 `src/openpi/cache/orchestrator.py`

##### B0 改动（无 view+history 注入；零运行时行为变化）

- `__init__` 多两个可选参数：`offline_writers: list[OfflineWriter] = ()`、`library_stats: Optional[LibraryStats] = None`（B0 默认空 / None；B2 wiring 时填）
- `_build_entry_chain` 改造为可选附加 OfflineWriter 计算（B0 写无任何 OfflineWriter，所以走 fast path 不变）：

```python
def _build_entry_chain(self, record):
    ...    # existing chain-link build (unchanged)
    if self._offline_writers and self._library_stats is not None:
        for writer in self._offline_writers:
            per_entry_factors = writer.compute_for_episode(entries, self._library_stats)
            for entry, factors in zip(entries, per_entry_factors, strict=True):
                if entry.payload.factors is None:
                    entry.payload.factors = {}
                entry.payload.factors.update(factors)
    return entries
```

##### B1 改动（首次注入 view+history）

- 新增 `self._action_history: list[torch.Tensor] = []` 与 `self._state_history: list[torch.Tensor] = []`，在 `_reset_episode_buffer()` 中同时清空
- 新增 `self._state_history_anchor_cp: CheckpointID` —— 在 `__init__` 末尾设为 `min(self._gates.keys(), key=lambda cp: cp.value)`（即所有 enabled checkpoint 中 enum value 最小的一个；CP1 < CP2 < CP3，所以默认 CP1+CP3 → anchor=CP1，CP3-only → anchor=CP3，CP1-only → anchor=CP1）
- `broadcast_action` 末尾 append 到 `self._action_history`（与现有 strategy/gate/judge 广播并行；orchestrator 自己持有，不依赖 strategy buffer）
- `on_task_end` 改造：在 `_close_current_search_sessions()` 之后**显式**调 `_reset_episode_buffer()`（清掉 episode_steps + miss_by_checkpoint + action_history + state_history）。当前 `on_task_end`（orchestrator.py:478-485）只清 search sessions，episode buffer 不清；WebSocket 中断时若 episode_end 没触发，buffer 会带到下次连接。这是 pre-existing 的轻度泄漏，本计划顺带修上以防 history 跨连接污染。
- `check()` 的修改（**关键：state 在 anchor checkpoint 入口 append，避免重复**）：

```python
def check(self, checkpoint_id, *, request_context=None, **stage_outputs):
    ...
    # Existing: gate
    with self._timer.measure(f"{prefix}_build"):
        query_keys = self._key_builder.build(checkpoint_id)

    # NEW (B1): state history append — ONLY on the per-deployment "anchor"
    # checkpoint, so each inference cycle appends exactly once even when
    # multiple checkpoints (CP1 + CP3) are enabled. Anchor is the lowest-value
    # enabled CheckpointID (CP1 wins when enabled; CP3 takes over only if CP1
    # is disabled). Append happens regardless of gate/full-hit outcome because
    # the obs has been observed and built either way; episode history must
    # remain gap-free for velocity-level descriptors.
    if checkpoint_id == self._state_history_anchor_cp and "robot_state" in query_keys:
        self._state_history.append(query_keys["robot_state"])

    # Existing: gate-skip return path (unchanged)
    if not should_search:
        ...

    with self._timer.measure(f"{prefix}_search"):
        results = strategy.search(ctx)

    # NEW (B1): build PayloadView + HistoryView and inject into Judge.
    from openpi.cache.components.payload_view import StoragePayloadView
    from openpi.cache.components.factors.base import HistoryView
    view = StoragePayloadView(self._storage)
    history = HistoryView(
        actions=list(self._action_history),
        states=list(self._state_history),
    )

    with self._timer.measure(f"{prefix}_judge"):
        judge_result = judge(
            results, checkpoint_id, self._key_builder.cached_data,
            view=view, history=history,
        )

    ...
    # NEW (B1): winner fetch through view (memo dedup with judge-side view.get)
    if hit_type in (HitType.FULL_HIT, HitType.WARM_START) and winner_id is not None:
        with self._timer.measure(f"{prefix}_fetch"):
            payload = view.get(winner_id)
        ...
```

##### Lifecycle 语义（B1 增）

| 事件 | `_action_history` | `_state_history` | 备注 |
|---|---|---|---|
| `on_task_begin` / `on_episode_start` | clear | clear | 通过 `_reset_episode_buffer()` |
| **anchor CP** check (任意结果) | 不动（broadcast_action 之后才 append）| **append** query_keys["robot_state"] | 唯一 append 点 |
| anchor CP gate skip | 不动 | **仍然 append**（gap-free） | query_keys 已 build |
| anchor CP FULL_HIT / WARM_START | 不动 | append | 同上 |
| 非-anchor CP check（如 CP1+CP3 中的 CP3）| 不动 | **不 append** | 避免双 append |
| `broadcast_action(act)` | append act | 不动 | Interceptor 在 cache hit 与 inference 两路都调用，所以 action_history 与 state_history 同步增长 |
| `on_episode_end` | clear（finally 块）| clear（同左）| 通过 `_reset_episode_buffer()` |
| `on_task_end`（连接关闭 / WebSocket 中断） | clear | clear | **B1 新增显式调 `_reset_episode_buffer()`**，覆盖 episode_end 没触发的 leak case |

##### 测试覆盖（参见 §8.5 `test_orchestrator_history.py`）

- CP1-only config：每 cycle state append 一次（anchor=CP1）
- CP1+CP3 config：每 cycle state 仍只 append 一次（anchor=CP1，CP3 不 append）
- CP3-only config：每 cycle state append 一次（anchor=CP3）
- gate-skip 路径：state 仍 append
- FULL_HIT / WARM_START 路径：state 仍 append
- episode reset：清空
- WebSocket 中断模拟：on_task_end 清空 buffer（无跨连接污染）
- 老 Judge（不带 view kwargs）：依赖 `**kwargs` 吸收，正常工作

##### B2 改动（OfflineWriter wiring）

- `__init__` 接受非空 `offline_writers` 与 `library_stats`
- `_build_entry_chain` 已在 B0 落好骨架，B2 起开始有 writer 实例进入循环

#### 8.2.4 `src/openpi/cache/components/search_strategy.py`

三个 concrete strategy class（`QdrantWeightedRrfKnnStrategy` / `WeightedRrfKnnStrategy` / `WeightedScoreSumKnnStrategy`）的 `__init__` 各加一个 kw-only 参数：

```python
def __init__(self, storage, *, top_k=1, ..., min_top_k_hint: int = 0):
    ...
    self._top_k = max(top_k, min_top_k_hint)    # 直接覆盖 self._top_k
```

`search()` 内 `top_k=self._top_k` 不变（仍然用同一个属性）。

#### 8.2.5 `src/openpi/cache/backends/in_memory_backend.py`

- 增加 attribute `self.library_stats: Optional[LibraryStats] = None`
- 增加 public method（**非** ABC override）：

```python
def fetch_entry(self, id: str) -> CacheEntry:
    """Return the full CacheEntry by id. O(1) dict lookup.

    Used by CacheStorage.fetch_entry duck-typed call (§8.2.7) to support
    PayloadView chain-walk. Backend ABC does not declare this method —
    it is an InMemoryBackend-specific capability that the facade exposes
    via getattr.
    """
    if id not in self._entries:
        raise KeyError(id)
    return self._entries[id]
```

- `load_artifact` 末尾：

```python
self.library_stats = data.get("library_stats")
if self.library_stats is None:
    from openpi.cache.components.factors.base import LibraryStats
    logger.info("Artifact missing library_stats; computing from %d entries", len(self._entries))
    self.library_stats = LibraryStats.compute_from_entries(list(self._entries.values()))
```

`__init__` / `insert` / `delete` 等不动。

#### 8.2.6 `src/openpi/cache/config.py`

- `JudgeConfig` 增加新字段：

```python
@dataclass
class FactorConfig:
    type: str                              # registry name, e.g. "f1a_a", "f1b_a", "f2"
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class ComposerConfig:
    type: str = "weighted_sum"             # "weighted_sum" | "and" | "or"
    weights: Optional[dict[str, float]] = None
    tier_thresholds: Optional[dict[str, float]] = None    # {"full_hit": ..., "warm_start": ...}
    per_factor_thresholds: Optional[dict[str, float]] = None    # for and/or
    warm_start_t: Optional[float] = None    # which denoise timestep to warm-start at (must be in CANONICAL_DENOISE_TIMESTEPS)
    directions: Optional[dict[str, str]] = None
    # Per-key direction for non_monotonic descriptors (§2.8.2 取向语义说明).
    # Format: {key: "high" | "low" | "range:[lo,hi]"}.
    # validate_cache_config rejects: (a) any non_monotonic key with non-zero weight
    # but missing direction; (b) any direction value that's not in the recognized
    # forms.

@dataclass
class NormalizerConfig:
    type: str = "percentile_rolling"
    window_size: int = 200
    cold_start_strategy: str = "force_miss"

@dataclass
class JudgeConfig:
    type: str = "threshold"
    threshold: float = 0.98
    warm_tiers: list[dict[str, float]] | None = None
    start_t: float | None = None
    # NEW (only for type="composite")
    factors: Optional[list[FactorConfig]] = None
    composer: Optional[ComposerConfig] = None
    normalizer: Optional[NormalizerConfig] = None
```

- `_JUDGE_TYPES`：**B0 不加 `"composite"`**（即 `"composite"` 在 `_JUDGE_TYPES` 之外 → validator 拒收，给出 "judge.type='composite' not yet enabled in B0; available in B1+ when CompositeJudge algorithms land" 的明确错误）。**B1 加入 `"composite"`** 同 commit 算法 land，validator 接受；composite-specific 6 项校验（下面）随之激活。这样 B0 的"零运行时行为变化"保证扩展为"B0 拒收 composite YAML 在 config-load 阶段，而非首次 verdict 时 NotImplementedError"，符合 fail-fast at load 的 UX 期望（Round 11 Item 3）。
- `_CONFIG_TYPES` 字典登记新 dataclass：`FactorConfig` / `ComposerConfig` / `NormalizerConfig`。
- **`_dict_to_dataclass` 扩展 `list[Dataclass]` 处理（新能力，现有代码没有）**：实测 (`config.py:310-359`) 当前只对若干特定 key（`checkpoints` / `vector_dims` / `field_similarity`）做 dict-of-dataclass 特殊处理；现有的 `list[...]` 字段都是 `list[primitive]`（`warm_tiers: list[dict[str, float]]` / `trajectory_weights: list[float]`），所以从未触发"list of dataclass" 路径。`JudgeConfig.factors: list[FactorConfig]` 是首例。处理方式：在 `_dict_to_dataclass` 通用分支前增加：

```python
# Handle list[Dataclass] uniformly via field type annotation
import typing
origin = typing.get_origin(_resolve_type(field_types[key]))
if origin is list and isinstance(value, list):
    inner = typing.get_args(_resolve_type(field_types[key]))[0]
    inner_cls = _resolve_type(inner)
    if isinstance(inner_cls, type) and dataclasses.is_dataclass(inner_cls):
        kwargs[key] = [
            _dict_to_dataclass(inner_cls, item) if isinstance(item, dict) else item
            for item in value
        ]
        continue
```

字符串注解（`"list[FactorConfig]"`）的 `typing.get_origin` 返回 None，需要 fallback：用 `re.match(r"list\[(\w+)\]", clean)` 抽内层名再查 `_CONFIG_TYPES`。两条路径都走，确保前向（type）与字符串（PEP 563 / `from __future__ import annotations`）注解都被覆盖（实测当前 config.py 顶部有 `from __future__ import annotations`，所以字符串路径是主要触发的）。

测试覆盖：`tests/cache/test_config.py` 加一例 YAML round-trip：

```yaml
checkpoints:
  cp1:
    judge:
      type: composite
      factors:
        - type: f1a_a
          params: {window_k: 3, descriptors: [jerk, dir]}
        - type: f2
          params: {K: 5}
      composer:
        type: weighted_sum
      normalizer:
        type: percentile_rolling
        window_size: 200
```

→ `_dict_to_dataclass` 后 `cfg.checkpoints["cp1"].judge.factors` 应是 `list[FactorConfig]`（不是 `list[dict]`），`cfg.checkpoints["cp1"].judge.factors[0].type == "f1a_a"`。

- `validate_cache_config` 加 composite-specific 校验。**注意分层**：现有 `validate_cache_config(config: CacheConfig)`（config.py:399）被 `load_cache_config`（config.py:388）在 backend 构造**之前**调用，签名只接 `config`，**不**能 access storage / backend instance。因此所有校验只能基于 `config.*` 静态字段。下列 6 项全部为纯静态检查：
  1. 当 `type=="composite"` 时 `factors / composer` 必须存在
  2. 每个 `FactorConfig.type` 必须在 `factors.registry.known()`
  3. **`requires_library_stats` capability check**（纯静态，对 backend type 比较）：对每个 factor 取 `cls = registry.get_class(f.type)`，若 `getattr(cls, "requires_library_stats", False) is True`，则 `config.backend.type == "in_memory"` 必须成立（当前唯一暴露 library_stats 的 backend 是 InMemoryBackend；§8.2.5）—— 否则 raise ConfigValidationError 给出 backend type + factor name
  4. **`requires_chain_walk` capability check**（纯静态）：对每个 factor 检查 `getattr(cls, "requires_chain_walk", False)`，若 True 则 `config.backend.type == "in_memory"` 必须成立（当前唯一暴露 `fetch_entry` capability 的 backend 是 InMemoryBackend；§8.2.7）。**不再做任何 storage probe** —— Round 7 Item 2 指出 InMemoryBackend.fetch_entry 对缺失 id raise KeyError，sentinel probe 区分不了"capability 不存在"与"id 不存在"。改为静态 backend type 比较，cleaner 且不需要构造 storage
  5. **composite warm-start 完整校验**（Round 5 Item 5 + Round 11 Item 2）：
     - **5a. checkpoint gate**：若 `cp_config.judge.composer.warm_start_t is not None` 且 `cp_name != "cp1"`，raise ConfigValidationError（与现有 `always_warm_start` / `warm_tiers` 限制一致；CP3 hit 在 interceptor.py:583-615 路径中只走 broadcast + buffer，warm_start 语义无效）
     - **5b. canonical timestep**：若 `composer.warm_start_t is not None`，必须 `round(warm_start_t, 4) ∈ CANONICAL_DENOISE_TIMESTEPS`（与现有 `always_warm_start.start_t` 校验 config.py:872-877 同规则）；写回归一化值
     - **5c. pairwise rule**（warm-start 路径与 timestep 必须配对存在）：
       - 若 `composer.type == "weighted_sum"`：`composer.tier_thresholds.warm_start` 与 `composer.warm_start_t` 必须**同时存在或同时缺失**。同时存在 → composite judge 可发 WARM_START；同时缺失 → 仅发 FULL_HIT / MISS。仅一者存在 → raise（warm_start_t 无路径触发；或 warm-start tier 阈值无 timestep 可用）
       - 若 `composer.type ∈ {"and", "or"}`：暂不支持 WARM_START（`per_factor_thresholds` 仅决定单一 hit/miss path）；若 `composer.warm_start_t is not None` → raise，提示用 weighted_sum
     - **5d. tier ordering**（仅 weighted_sum）：若 warm_start 路径启用，则 `tier_thresholds.warm_start < tier_thresholds.full_hit` 必须成立（与现有 `warm_tiers` 严格递减规则 config.py:911-916 同义；warm_start 阈值若 ≥ full_hit 则 warm_start tier 不可达）
  6. **ComposerConfig.directions 覆盖率校验**（Round 7 Item 3）：对每个 factor 取 `cls = registry.get_class(f.type)`，调 `cls.describe(f.params)` 拿到该 instance 实际会产生的 `key -> orientation` map（**用 classmethod，不实例化**——避免 F1b 等 `requires_library_stats=True` 的因子在 validator 阶段就需要 library_stats）。union 所有 factor 的 describe 结果 → 找出 orientation == `"non_monotonic"` 的 key 集合 → 对每个这样的 key，若该 key 在 `composer.weights` 里权重非 0，则 `composer.directions[key]` 必须存在且 value 形式合法（`"high"` / `"low"` / `"range:[lo,hi]"`）；否则 raise

  **不在 validator 内做 / 由 build_per_connection_components 内部 assert 的事**：实际构造完 backend 后断言 `requires_library_stats` 因子配置下 `per_conn_storage.library_stats is not None`（应已被静态校验保证，此处是防御性 internal-error 而非 user-facing config error；与现有 cache_storage 内部 dim 校验同性质）。这条不挂 `validate_cache_config`，而是 builder 自检。
- `_build_judge` 加 `composite` 分支：构造 extractors list、composer、normalizer，组装 CompositeJudge。**任何 `requires_library_stats=True` 的 factor 都需要 `library_stats` 注入**（覆盖 F1a-A / F1a-T / F1b-A / F1b-T 四个；F2 不需要）。`_build_judge` 签名增加 `library_stats: Optional[LibraryStats] = None` 参数；构造时按 capability flag 选择性注入（§8.3 wiring）。
- `_build_composer(cfg: ComposerConfig)` 显式映射：

  ```python
  def _build_composer(cfg: ComposerConfig) -> Composer:
      if cfg.type == "weighted_sum":
          if cfg.weights is None:
              raise ConfigValidationError("composer.type=weighted_sum requires 'weights'")
          if cfg.tier_thresholds is None or "full_hit" not in cfg.tier_thresholds:
              raise ConfigValidationError(
                  "composer.type=weighted_sum requires tier_thresholds.full_hit"
              )
          return WeightedSumComposer(
              weights=cfg.weights,
              full_hit_threshold=cfg.tier_thresholds["full_hit"],
              warm_start_threshold=cfg.tier_thresholds.get("warm_start"),
              warm_start_t=cfg.warm_start_t,
              directions=cfg.directions,
          )
      if cfg.type == "and":
          if cfg.per_factor_thresholds is None:
              raise ConfigValidationError("composer.type=and requires 'per_factor_thresholds'")
          return AndGateComposer(
              per_factor_thresholds=cfg.per_factor_thresholds,
              warm_start_t=cfg.warm_start_t,
          )
      if cfg.type == "or":
          if cfg.per_factor_thresholds is None:
              raise ConfigValidationError("composer.type=or requires 'per_factor_thresholds'")
          return OrGateComposer(
              per_factor_thresholds=cfg.per_factor_thresholds,
              warm_start_t=cfg.warm_start_t,
          )
      raise ConfigValidationError(f"Unknown composer.type '{cfg.type}'")
  ```

  关键映射：YAML `tier_thresholds.full_hit` / `tier_thresholds.warm_start` → 构造函数 `full_hit_threshold` / `warm_start_threshold`（dict 名 → kw param）；YAML `warm_start_t` → 构造函数同名 kw；YAML `directions` 直接透传。
- `_build_normalizer(cfg: NormalizerConfig)` 同样模式：`PercentileRollingNormalizer(window_size=cfg.window_size, cold_start_strategy=cfg.cold_start_strategy)`。
- `_build_search_strategy` 增加 `min_top_k_hint: int = 0` 参数；在调用 strategy 构造时透传。
- `build_per_connection_components` wiring 顺序：

```python
for cp_name, cp_config in config.checkpoints.items():
    ...
    # 1. backend → library_stats
    library_stats = per_conn_storage.library_stats   # facade accessor (§8.2.7)
    # 2. judge needs library_stats for any factor with requires_library_stats=True (F1a + F1b; not F2)
    judges[cp_id] = _build_judge(cp_config.judge, library_stats=library_stats)
    # 3. strategy needs judge.min_required_top_k
    min_hint = getattr(judges[cp_id], "min_required_top_k", 0)
    search_strategies[cp_id] = _build_search_strategy(
        cp_config.search_strategy, per_conn_storage, fusion_weights,
        min_top_k_hint=min_hint,
    )

# offline writers (per-config 决定哪些 factors 同时是 OfflineWriter)
offline_writers = collect_offline_writers_from_judges(judges)

orch = CacheOrchestrator(
    ...,
    judges=judges, search_strategies=search_strategies,
    offline_writers=offline_writers,
    library_stats=library_stats,
)
```

辅助函数 `collect_offline_writers_from_judges` 遍历每个 CompositeJudge 的 extractors，挑出同时实现 OfflineWriter Protocol 的，去重（按 type+params hash）后返回 list。

#### 8.2.7 `src/openpi/cache/cache_storage.py`

新增 `fetch_entry(id) -> CacheEntry` facade 方法，**duck-typed** 调用 backend 的 optional `fetch_entry` capability。Backend ABC 不动；不是所有 backend 都需要支持。

```python
def fetch_entry(self, id: str) -> CacheEntry:
    """Fetch the full CacheEntry by id (duck-typed backend capability).

    Backends that store full entries in-memory (currently InMemoryBackend)
    expose a `fetch_entry(id)` method; the facade forwards to it. Backends
    without this capability (Qdrant) raise NotImplementedError, with a
    config-friendly message naming the backend type.

    Used by PayloadView chain walks. Composite judges that need chain
    walking are config-gated to backends that support fetch_entry — see
    `validate_cache_config` (§8.2.6).
    """
    fn = getattr(self._backend, "fetch_entry", None)
    if fn is None:
        raise NotImplementedError(
            f"Backend {type(self._backend).__name__} does not expose "
            "fetch_entry; composite judges that use chain-walking factors "
            "(e.g. F1a-T) require InMemoryBackend or another backend that "
            "implements `fetch_entry(id) -> CacheEntry`."
        )
    return fn(id)


@property
def library_stats(self) -> Optional["LibraryStats"]:
    """Return the backend's library-level statistics (or None).

    Same duck-typing pattern as fetch_entry: backends that compute /
    persist library_stats (currently InMemoryBackend, populated in
    `load_artifact`) expose a `library_stats` attribute; backends without
    it (Qdrant) return None. Component builders read this through the
    facade and pass the value into composite judge construction (§8.2.6
    wiring); this avoids private reach-through into `self._backend`.
    """
    return getattr(self._backend, "library_stats", None)
```

文档说明：PayloadView walk 仅在支持 `fetch_entry` 的 backend 上工作（当前 InMemoryBackend；与现有 trajectory search 限制一致 —— Qdrant backend 也不支持 trajectory search）。`validate_cache_config` 在启动时校验：composite judge 中**任何 `requires_chain_walk=True` 的 factor**（F1a-T）或 **任何 `requires_library_stats=True` 的 factor**（F1a-A / F1a-T / F1b-A / F1b-T 四个），都要求 `config.backend.type == "in_memory"`，否则 fail-fast。

### 8.3 Wiring 序列总结

```
load_cache_config(yaml_path)
  ├─ _dict_to_dataclass → CacheConfig (含 JudgeConfig.composite 字段)
  └─ validate_cache_config (新校验：composite, factor types, library_stats 可用性)

build_cache_components(config)
  └─ build_per_connection_components(config, storage)
      ├─ _build_backend → InMemoryBackend (.library_stats from artifact / computed)
      ├─ per-CP loop:
      │    ├─ library_stats = per_conn_storage.library_stats   # facade accessor (§8.2.7)
      │    ├─ judges[cp_id] = _build_judge(judge_cfg, library_stats)
      │    │     └─ if type=="composite":
      │    │          ├─ extractors = [_build_extractor(f, library_stats) for f in cfg.factors]
      │    │          │    where:  cls = registry.get_class(f.type)
      │    │          │           kwargs = dict(f.params)
      │    │          │           if getattr(cls, "requires_library_stats", False):
      │    │          │               kwargs["library_stats"] = library_stats   # validator already ensures non-None
      │    │          │           cls(**kwargs)
      │    │          ├─ composer = _build_composer(cfg.composer)
      │    │          ├─ normalizer = _build_normalizer(cfg.normalizer)
      │    │          └─ CompositeJudge(extractors, composer, normalizer)
      │    ├─ min_hint = judges[cp_id].min_required_top_k
      │    └─ strategies[cp_id] = _build_search_strategy(ss_cfg, storage, weights, min_top_k_hint=min_hint)
      ├─ offline_writers = collect_offline_writers_from_judges(judges)
      └─ Orchestrator(..., offline_writers=ow, library_stats=library_stats)
```

### 8.4 Backwards Compat / Migration

- **老 cache pkl artifact**（无 `library_stats` 字段、entries 无 `payload.factors`）：
  - `InMemoryBackend.load_artifact` 检测 `data.get("library_stats")` 缺失 → 调用 `LibraryStats.compute_from_entries(list(self._entries.values()))` 现算并 cache 到 `self.library_stats`
  - 老 entry `payload.factors is None` → CompositeJudge 的 F1b OnlineExtractor.extract 检测缺字段 → 返回 NaN → Composer 跳过该因子或按 `cold_start_strategy` 处理
- **老 YAML config**（judge.type=threshold 等）：零改动；新 fields default None；`_build_judge` 不进 composite 分支。
- **老 ThresholdJudge / AlwaysHitJudge / AlwaysWarmStartJudge**：吃 `**kwargs` 即可；测试 `tests/cache/test_judge.py` 等无需修改。
- **CountingStorage**（`tests/cache/conftest.py:33-42`）：通过 storage 计数器工作；PayloadView.get → storage.fetch_payload 透传 → 计数器仍递增。零修改。
- **Prefill mode**（`tests/cache/test_cache_storage.py:91`）：PayloadView.get → storage.fetch_payload → prefill 短路链路完整。零修改。

### 8.5 测试文件

| 路径 | 批次 | 内容 |
|---|---|---|
| `tests/cache/test_payload_view.py` (新建) | B0 | StoragePayloadView 的 get / get_entry / get_many memo；walk_prev / walk_next 在无-fork 链上的正确性；fork 路径 raise NotImplementedError；trajectory boundary stop |
| `tests/cache/components/factors/test_registry.py` (新建) | B0 | register / build / get_class / known；未知名 raise；重复注册 raise；空 stub 模块导入安全（registry 不报错） |
| `tests/cache/components/factors/test_base.py` (新建) | B2（依赖 LibraryStats 实现） | LibraryStats.compute_from_entries 用合成 entry 验证 σ_d 与 active_mask；HistoryView 字段语义 |
| `tests/cache/components/factors/test_composers_protocol.py` (新建) | B0 | Composer Protocol shape 测试：bind_orientations 接收 dict；compose 签名含 winner_id；stub 实现 raise NotImplementedError；非 monotonic key 缺 directions 时 ConfigValidationError |
| `tests/cache/components/factors/test_normalizer_protocol.py` (新建) | B0 | Normalizer Protocol shape 测试：bind_keys 接收 list[str]；__call__ 签名 dict→dict；stub 实现 raise NotImplementedError |
| `tests/cache/test_judge.py` 扩充 (B0 part) | B0 | CompositeJudge 类骨架可构造；`__init__` 调 bind_orientations + bind_keys；orientation 冲突 raise；`min_required_top_k` 计算正确；空 results → MISS（不调 extractor / composer）；**key contract assertion**：synthetic extractor 返回 keys 与 declared `descriptor_orientations` 不一致 → RuntimeError |
| `tests/cache/test_config.py` 扩充 (B0 part) | B0 | (a) `_dict_to_dataclass` 对 `list[FactorConfig]` 正常解析（dataclass 而非 dict）；ComposerConfig.directions 字段解析；WindowSpec dict→tuple 经 `_normalize_windows` 测试（dict 形式 / tuple 形式 / 混合都接受）；(b) **`type=composite` 在 B0 直接 raise ConfigValidationError**（错误信息含 "available in B1+"）—— 这是 B0 的核心 fail-fast 行为，与 R11 Item 3 修订一致；(c) factor 类直接构造测试（不走 validator）：F1a / F1b / F2 各 instantiate 后 `descriptor_orientations` 与 `cls.describe(params)` 一致；(d) `cls.describe(params)` 对 dict-windows / tuple-windows 输入都能产生正确 key；(e) 各 factor class 的 capability flags 正确 |
| `tests/cache/test_cache_storage.py` 扩充 | B0 | `fetch_entry(id)` 方法（duck-typed）；不支持 backend → NotImplementedError；`library_stats` property |
| `tests/cache/components/factors/test_runtime_continuity.py` (新建) | B1 | RuntimeContinuity(source="action") 与 (source="state") 各四个描述子的数值正确性（合成 sweep / turn / shake 三种 regime）；descriptor_orientations 类属性正确 |
| `tests/cache/components/factors/test_consensus.py` (新建) | B1 | TopKActionConsensus.required_top_k 正确；extract 输出方差合理；descriptor_orientations 类属性 |
| `tests/cache/components/factors/test_composers_algorithm.py` (新建) | B1 | WeightedSumComposer / AndGate / OrGate 的 tier 映射算法（**B1 才算法 land**）；orientation flip；NaN 跳过；非 monotonic direction 应用；warm_start_t emit |
| `tests/cache/components/factors/test_normalizer_algorithm.py` (新建) | B1 | PercentileRollingNormalizer 算法（**B1 才算法 land**）；force_miss → all-NaN sentinel；passthrough；lenient 阈值 |
| `tests/cache/test_orchestrator.py` 扩充 (B1 part) | B1 | check() 注入 view + history 给 Judge；老 Judge 收到 kwargs 不报错；CompositeJudge end-to-end FULL_HIT / WARM_START / MISS 三路 |
| `tests/cache/test_orchestrator_history.py` (新建) | B1 | anchor checkpoint 选择（CP1 / CP3 / CP1+CP3 各一例）；gate-skip / FULL_HIT / WARM_START 路径 state 仍 append；episode reset 清空；on_task_end 清空（WebSocket 中断模拟） |
| `tests/cache/test_config.py` 扩充 (B1 part) | B1 | composite YAML 在 B1 不再被拒收；6 项 composite-specific 校验全覆盖：(1) factors/composer 必填；(2) factor type 必须在 registry.known()；(3) requires_library_stats=True factor 配 Qdrant → raise；(4) requires_chain_walk=True factor (F1a-T) 配 Qdrant → raise；(5a-d) warm_start checkpoint gate / canonical timestep / pairwise rule / tier ordering 各路径全覆盖；(6) describe-based directions：F1b 实例的 `f1b_a_curv_radius__p5_f10` 这种参数化 key 在 `composer.directions` 里缺失（且 weight ≠ 0） → raise；`min_required_top_k` 反向喂回 strategy 的 wiring |
| `tests/cache/components/factors/test_source_window.py` (新建) | B2 | SourceWindowSmoothness OfflineWriter.compute_for_episode 在合成 episode 上输出符合预期；OnlineExtractor.extract 从 payload.factors 读取；缺字段 NaN；边界窗口 NaN |
| `tests/cache/test_orchestrator.py` 扩充 (B2 part) | B2 | `_build_entry_chain` 在配置 OfflineWriter 时填 payload.factors；library_stats 透传 |
| `tests/cache/test_artifact_roundtrip.py` (新建) | B2 | enrich_artifact_with_factors helper；新 artifact 格式 round-trip；老 artifact (无 library_stats) 加载 → fallback compute |

### 8.6 实施顺序（B0 → B1 → B2）

| 批次 | 文件改动清单 | "零运行时行为变化"保证 |
|---|---|---|
| **B0** | 8.1.1 (PayloadView)，8.1.2-8.1.4 (factor protocols + base + registry)，**8.1.5 / 8.1.6 / 8.1.7 完整 metadata 层**（base class + thin subclass + capability flags + `describe` classmethod + `__init__` 设 instance descriptor_orientations + `@register` 装饰器 —— 全部 ship；只有 `extract` / `compute_for_episode` body 占位 raise NotImplementedError），8.1.8 / 8.1.9 (Composer / Normalizer 类骨架 + Protocol + bind 方法 ship；algorithm body raise NotImplementedError)，8.2.1 (`CachePayload.factors` Optional 字段)，8.2.2 (CompositeJudge 类骨架 + collect+bind + key contract assertion + cold-start sentinel + Judge protocol 扩 default-None kwargs + 老 Judge `**kwargs` 吸收)，8.2.5 (InMemoryBackend.fetch_entry 普通方法 + library_stats attr 默认 None)，8.2.6 (config dataclass + `_dict_to_dataclass` list[Dataclass] 扩展 + `_build_judge` / `_build_composer` / `_build_normalizer` 完整 mapping + capability-flag injection；**`_JUDGE_TYPES` 不含 `"composite"` → validator 在 config-load 阶段拒收 composite YAML**，给 fail-fast 错误信息)，8.2.7 (CacheStorage.fetch_entry / library_stats duck-typed accessor)，docs §5.6 contract refinement + §5.11/5.12 骨架 + index 同步，测试：test_payload_view / test_registry / test_composers_protocol / test_normalizer_protocol / test_judge (B0 part) / test_config (B0 part：含 "composite YAML 在 B0 被 validator 拒收" 测试) / test_cache_storage 扩充 | ✅ Orchestrator `check()` **不**注入 view+history；composite YAML **在 config-load 阶段被拒收**（不是首次 verdict 才 fail），用户得到清晰错误信息"composite available in B1+"；老 YAML 完全不进新代码路径 |
| **B1** | **`_JUDGE_TYPES` 加 `"composite"`**（解除 B0 拒收 gate，与 algorithm 同 commit），**8.1.5 / 8.1.7 algorithm body 填实**（RuntimeContinuity / TopKActionConsensus 的 extract 实现，§2.8.4 jerk / dir / curv_radius / cum_disp + F2 variance），**8.1.8 / 8.1.9 algorithm 填实**（WeightedSumComposer / AndGateComposer / OrGateComposer / PercentileRollingNormalizer 的 compose / __call__ body），**§8.2.6 6 项纯静态 composite 校验激活**（与 `"composite"` 一同被 validator 接受 / 校验），**8.2.3 完整改造**（Orchestrator `_state_history` + anchor checkpoint policy + view+history 注入 + winner fetch rewire 走 view.get + on_task_end 显式 reset），8.2.4 (strategy 加 min_top_k_hint)，config builder wiring 完整接通（capability-flag library_stats injection + `min_required_top_k` 反向喂回 strategy），docs §5.11 / `tutorial.md` composite 章节，测试：test_runtime_continuity / test_consensus / test_composers_algorithm / test_normalizer_algorithm / test_orchestrator_history / test_orchestrator (B1 part) / test_config (B1 part：6 项校验全覆盖 + composite 不再被拒收) | 行为变化范围 = 选择 composite judge 的 YAML，影响仅限 verdict 阶段 |
| **B2** | **8.1.6 algorithm body 填实**（SourceWindowSmoothness 的 extract + compute_for_episode + LibraryStats 完整实现），8.2.5 完整 (`load_artifact` library_stats 加载 / fallback)，8.2.3 (`_build_entry_chain` 调 OfflineWriter)，离线 build pkl 工具更新（§2.5 三脚本 + factor_postprocess.py），docs §5.12 OfflineWriter / artifact 格式补完，测试：test_base / test_source_window / test_orchestrator (B2 part) / test_artifact_roundtrip | 老 artifact compute_from_entries fallback；写入侧改动经 offline_writers 列表 opt-in |

### 8.7 Risk Register（执行级）

| 风险 | 触发场景 | 缓解 |
|---|---|---|
| ~~`StoragePayloadView` reach 进 `_storage._backend._entries`~~ | (resolved per G1 R1 item 2) | StoragePayloadView 现在走 `CacheStorage.fetch_entry` facade 方法，facade 内部 duck-types backend 的可选 `fetch_entry` capability。Backend ABC 不动；Qdrant 等不支持的 backend 在被调用时 raise NotImplementedError，配置层 fail-fast |
| `_score_memo` 的 `SearchSessionActiveError` 与新 OfflineWriter 写 entry.payload.factors 冲突 | episode-end 调 `_build_entry_chain` 时 sessions 已被 `_close_current_search_sessions` finally 释放 (orchestrator.py:476) → 不冲突 | 单元测试覆盖 episode-end 顺序 |
| `library_stats` 是 None（非 in-memory backend，如 Qdrant）下使用 F1b | Qdrant + composite judge with f1b_* | `validate_cache_config` 校验 composite judge 含 f1b_* 时强制 backend.type=in_memory，否则 raise |
| Old artifact `compute_from_entries` 阻塞 server startup | 大 artifact (50k+ entries) 启动时全扫一遍 | log + 时间统计；future 可加并行；artifact build 工具默认填好 library_stats，老 artifact 重建一次即可 |
| F1a 的 `RuntimeContinuity` 没有 library_stats 时 | 任何走 in-memory backend 的 deployment 都有 → 不会发生 | None — 静态校验确保 backend.type=in_memory |
| Judge protocol 加 kwargs 可能漏改某个 third-party Judge 实现 | 项目目前 Judge 实现全部在 components/judge.py，无外部 | grep `def __call__` in judge.py + tests |

### 8.8 Documentation 与 Index 同步交付物

L3 architectural change，按 WA §4 (Index Sync Rule) 与 §8 (Subsystem Rules) 必须落实下面的 doc 与 index 更新。**所有 doc 改动与 code 改动必须在同一 commit 中**。

#### 8.8.1 必改 docs

| 文件 | 改动 | 触发批次 |
|---|---|---|
| `docs/architecture/cache_system.md` §5.6 SimilarityJudge | Refine purity 契约：从"DOES NOT call CacheStorage"改为"**No write to storage; read-only access via PayloadView allowed for verdict-time descriptor computation**"。同步 §5.6 的 Coupling 块。**这是 contract refinement 不是 violation** —— 原契约的语义本意是"无副作用"（见判句中 "no side effects"），新契约用更精确措辞固化此本意，并允许 read-only 取数 | B0（与 Judge protocol 扩 kwargs 同 commit）|
| `docs/architecture/cache_system.md` 新增 §5.11 PayloadView | 描述 PayloadView Protocol、StoragePayloadView 默认实现、ForkPolicy 五值的语义、per-`check()` 生命周期、duck-typed `fetch_entry` capability、当前仅 InMemoryBackend 支持等 | B0 |
| `docs/architecture/cache_system.md` 新增 §5.12 Verdict Factor System | 描述 OnlineExtractor / OfflineWriter Protocol、registry、CompositeJudge 流水线（extract → normalize → compose）、orientation 元信息收集机制、`payload.factors` schema、library-level metadata（σ + active_mask）、artifact 格式扩展 | B0 落骨架；B1/B2 各自补对应章节 |
| `docs/architecture/cache_system.md` §5.7 CachePayload schema | 加 `factors: Optional[dict[str, float]]` 字段说明 | B0 |
| `docs/cache/tutorial.md` §6 Component Judge | 增加 "composite" 类型一节：YAML 示例 + 因子/composer/normalizer 配置说明 + 与 ThresholdJudge 选用建议 | B1（首个真实可用 composite config）|
| `docs/cache/tutorial.md` §10 YAML Config 部分 | 新增 composite judge 的完整 YAML 示例与字段说明（FactorConfig / ComposerConfig / NormalizerConfig） | B1 |

#### 8.8.2 必改 index（WA §4 红线）

| 文件 | 改动 |
|---|---|
| `docs/README.md` §architecture/ 行 | 更新 `cache_system.md` 的 description 列以反映新增 §5.11 / §5.12 |
| `docs/README.md` §cache/ 行 | 更新 `tutorial.md` 的 description 列以反映新增 composite judge 章节 |
| `logs/README.md` Cache System 表 | 新增本 plan 行：`verdict_factor_judge.log.md` `In Progress` |
| `logs/README.md` 当本 plan 进入 Implemented / Validated 阶段 | 状态字段同步更新；最终归档时 `mv logs/verdict_factor_judge.log.md logs/archive/` 并更新行的位置与 status |

#### 8.8.3 不需要更新的 docs

- `docs/papers/*` —— 与 verdict factor 实现独立
- `docs/experiments/*` —— 实验 doc 在实际跑 cp1_cache 等 experiment 用 composite judge 时再单独更新
- `docs/cache/migration.md` —— 不影响非 Pi0.5 model 适配流程
- `docs/cache/temporal_prune.md` / `docs/cache/llm_layer_extract.md` —— 各 KeyBuilder 文档独立，与 verdict 阶段解耦

#### 8.8.4 WA §8 Subsystem Rules 表注册

WA §8 表格列出了 cache system 的 rule documents。当前已有 `docs/architecture/cache_system.md` + `docs/cache/tutorial.md`。本计划**不新增独立的 verdict_factor_system.md**（避免子系统文档碎片化）—— verdict factor 作为 cache system 的子模块，规则集中在 `cache_system.md` 的新增 §5.11 / §5.12 章节。WA §8 表格无需新增行。

如未来 verdict factor 复杂到独立子系统级别（多 backend 适配、多模型适配），再考虑独立 doc + WA §8 注册（需项目所有者批准）。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-04-26 17:58 CDT

- [Blocking] [Concern] State-side factor descriptor keys currently use the `"s"` namespace instead of the approved `"t"` namespace — reasoning: the approved plan registers the state/trajectory variants as `f1a_t` / `f1b_t`, and §8.1.5 / §8.1.6 explicitly annotate the key-prefix expressions as producing `f1a_a / f1a_t` and `f1b_a / f1b_t`. The implementation uses `cls.source[0]` while `RuntimeContinuityState.source == "state"` and `SourceWindowSmoothnessState.source == "state"` (`src/openpi/cache/components/factors/runtime_continuity.py:87`, `src/openpi/cache/components/factors/source_window.py:132`), so it emits `f1a_s_*` / `f1b_s_*` keys instead. This changes the metadata contract consumed by `ComposerConfig.weights` / `ComposerConfig.directions` and by docs §5.12; B1/B2 configs written against the registered `*_t` factor names will not match `describe()` / extractor keys. Fix by making the descriptor prefix explicit per thin subclass (or by an explicit source-to-key-prefix map) and add state-side tests for both `RuntimeContinuityState.describe(...)` and `SourceWindowSmoothnessState.describe(...)` expecting `*_t_*` keys.

### G2 Round 2 — Executor — 2026-04-26

- Accepted — Decoupled the descriptor key namespace from `source` by introducing an explicit class-level `key_initial: str` attribute on both `_RuntimeContinuityBase` and `_SourceWindowSmoothnessBase`. Thin subclasses now set both fields explicitly: `RuntimeContinuityAction` / `SourceWindowSmoothnessAction` set `key_initial = "a"`; `RuntimeContinuityState` / `SourceWindowSmoothnessState` set `key_initial = "t"`. The `describe()` classmethod prefix derivation switched from `f"f1a_{cls.source[0]}"` (which produced `f1a_s_*` for `source="state"`) to `f"f1a_{cls.key_initial}"` (now `f1a_t_*`); same for F1b. The `source` semantic field is retained — it still names the data the factor consumes ("action" / "state") and is independent of the YAML/payload key namespace. This matches the registered factor names (`f1a_t` / `f1b_t`) bijectively and aligns with both §8.1.5 / §8.1.6 plan comments and the B1+ ComposerConfig.weights / directions contract.
- Files touched (4): `src/openpi/cache/components/factors/runtime_continuity.py` (base attribute + describe + both thin subclasses), `src/openpi/cache/components/factors/source_window.py` (base attribute + describe + both thin subclasses), `tests/cache/components/factors/test_registry.py` (corrected `test_runtime_continuity_describe_state` to expect `f1a_t_jerk`; added `test_runtime_continuity_describe_state_multiple_descriptors`, `test_runtime_continuity_key_initial_attributes`, `test_source_window_describe_state_uses_t_namespace`, `test_source_window_key_initial_attributes`), `docs/architecture/cache_system.md` §5.12 key-template description (`<source_initial>` → `<key_initial>` with explicit a/t legend).
- Test evidence: `pytest tests/cache/components/factors/ tests/cache/test_payload_view.py tests/cache/test_cache_storage_factor_facade.py tests/cache/test_config_factor.py` → 90 passed (was 86; +4 new state-side namespace assertions). Full cache suite: `pytest tests/cache/` → 561 passed, 4 skipped, zero regression vs the G2 Round 1 baseline.

### G2 Round 3 — Reviewer — APPROVED — 2026-04-26 18:04 CDT

- [Non-blocking] [Concern] The polished plan body still contains historical `source[0]` examples in §8.1.5 / §8.1.6 — reasoning: Round 2 correctly fixes the executable contract in source, tests, and `docs/architecture/cache_system.md` by using explicit `key_initial`; prior G2 Review Log entries also record why `source[0]` is wrong for state-side `*_t` keys. Because the current code/docs/test contract is now internally consistent and the stale snippets are confined to the pre-G2 plan body, this does not block G2 approval, but final archival polish should avoid reusing those snippets as implementation guidance.

### G2 Round 4 — Executor — 2026-04-26

- Accepted (non-blocking polish) — Updated the §8.1.5 / §8.1.6 plan-body code snippets in this file so that future readers do not mistake the pre-Round-2 `source[0]` examples for implementation guidance. Specifically: (a) `_RuntimeContinuityBase` / `_SourceWindowSmoothnessBase` now declare both `source: str` and `key_initial: str` class attributes; (b) the `describe()` classmethod prefix derivation reads `cls.key_initial` (not `cls.source[0]`) with an inline comment cross-referencing the G2 R1→R2 decision; (c) all four thin subclasses (`RuntimeContinuityAction` / `RuntimeContinuityState` / `SourceWindowSmoothnessAction` / `SourceWindowSmoothnessState`) explicitly set `key_initial = "a"` / `key_initial = "t"` alongside their existing `source` assignments; (d) §8.1.6 prose now mentions both `source` (semantic) and `key_initial` (key namespace, aligned with registry name suffix). The Round 3 G2 verdict (APPROVED) stands; this entry records the polish per WA §10 single-response rule.
- Files touched (1, plan-body only): `logs/verdict_factor_judge.log.md` §8.1.5 + §8.1.6 — code, prose, and inline comments only. No source / test / docs / index changes (those were already consistent at G2 Round 3 APPROVED).
- Additionally, while writing the user-facing tutorial (`docs/cache/verdict_factor_judge.md`, separate user request after APPROVED — not a Review-Log item), the same `key_initial` story is repeated in §2 / §10 Q&A so independent documentation channels stay aligned.
