# TRACER Phase 1 — M3 动态链深 SearchStrategy（Plan）

- **Status**: Implemented（**G1 + G2 APPROVED 2026-07-07**；§6 Verify green 981 pass / 6 skip；待 owner 确认归档）
- **Date**: 2026-07-07
- **Level**: **L2**（新增一个 `SearchStrategy` 可插拔组件；不改任何 Protocol/orchestrator/interceptor/backend/verdict；零训练）
- **上位依据**: [`tracer_retrieval_refinement_roadmap.log.md`](tracer_retrieval_refinement_roadmap.log.md) Phase 1（M3）。需求由该 roadmap 的 Phase 1 spec 固化（owner 经 `/goal` 授权推进）。
- **子实验目录**: `exp/zixuan_proposal/`（源提案 PDF 已归此目录）。
- **提案锚点**: Eq 7（chain-aware 检索）、Eq 22–23（动态深度 `T_t = argmax ψᵀr_t − μ·cost(T)`）。本期只做**免训练启发式/常数**版；学习 ψ 版押到 roadmap Phase 6。

---

## 1. 无上下文摘要（供 reviewer / 实现者）

本项目推理 cache 系统里，`SearchStrategy` 是"构造 `QuerySpec` 并调 `storage.search()`"的唯一组件（隔离律：只有它 search）。现有三个策略（`weighted_rrf_knn` / `weighted_score_sum_knn` / `qdrant_weighted_rrf_knn`）的**轨迹深度是 init 时定死的固定值**（`SearchStrategyConfig.trajectory_depth`，config.py:345）。合作者提案要把深度做成**逐步（per-step）自适应**（Eq 22–23）。

本期目标：新增一个 **`DynamicDepthKnnStrategy`**，逐 `search()` 调用按廉价特征选一个深度 `T_t`，再复用现有融合（rrf / score_sum）。**关键安全性质**：当深度策略配成"常数"时，本策略与现有固定深度策略**逐值等价**（非回归），dynamic 是纯 opt-in。全期**不碰**框架抽象、不碰 verdict、不训练（策略是启发式规则）。

---

## 2. 现状事实（已亲验，附锚点）

| 事实 | 锚点 |
|---|---|
| `SearchStrategy` Protocol：`search(ctx)->list[SearchResultLite]` | `search_strategy.py:58-81` |
| `SearchContext` 字段：query_keys / checkpoint_id / current_step / task_key | `search_strategy.py:40-55` |
| `TrajectoryMixin` 提供 `_init_trajectory` / `on_episode_start` / `record_action` / `record_query_keys` / `_build_trajectory_fields` / `get_search_session_id` | `search_strategy.py:84-177` |
| `_build_trajectory_fields()` 用**固定** `self._trajectory_depth`/`_weights`，取 `actual_depth=min(depth, len(history))`，history newest-first + weights[:actual_depth] | `search_strategy.py:147-177` |
| backend 轨迹融合已支持变长深度 `L=min(len(history),len(weights))` | in_memory_backend `_search_with_trajectory`（Explore 测绘） |
| `SearchStrategyConfig` 字段：type/top_k/step_filter/step_window/rrf_k/candidate_multiplier/field_similarity/score_normalization/trajectory_depth/trajectory_weights | `config.py:335-346` |
| 策略类型 frozenset | `_valid_strategy_types` `config.py:1229-1231` |
| 工厂 `_build_search_strategy(cfg, storage, fusion_weights, *, min_top_k_hint=0)` | `config.py:2707-2772` |
| 校验：type / backend 兼容（非 qdrant 需 in_memory）/ 数值范围 / score_norm（score_sum 必需）/ trajectory 深度权重长度非负 | `config.py:1465-1525`、`1801-1820` |
| `QuerySpec` 轨迹字段：trajectory_history / trajectory_weights / search_session_id / trajectory_query_ids | `storage_types.py:231-250` |

---

## 3. 设计

### 3.1 新增 `DepthPolicy`（可插拔子部件，src，免训练）

Protocol：`select(features: DepthFeatures) -> int`，**返回值契约**：必须 ∈ 策略持有的 `allowed_depths` 集（下游 `DynamicDepthKnnStrategy.search` 以显式 `raise ValueError` 兜之，fail-loud；非 `assert`，避免 `python -O` 剥除）。

**`DepthFeatures`（dataclass）** — 本期只装免训练、pre-search、不依赖 D⁻ 的信号：
- `current_step: int`（= `ctx.current_step`）
- `history_len: int`（= `len(self._query_history)`）
- `action_smoothness: Optional[float]` — **精确定义**：
  - 数据源 `self._action_history`（`TrajectoryMixin.record_action` 逐步 append；orchestrator 每步 `broadcast_action` 广播**全 chunk** `[chunk_len, A]`，orchestrator.py:374；含 FULL_HIT / gate-skip 步 → gap-free）。
  - 时序：`search()`（step t）在 `broadcast_action`（step t）**之前** → 此刻 `_action_history` 含 step 0..t-1；两个最新即 `a_{t-1}, a_{t-2}`（对齐提案 Eq 22）。
  - `len(_action_history) < 2` → `None`。
  - 否则每步代表向量取**首动作** `chunk[0]`（`[A]`，与 verdict-factor 的 `action_chunk[0]` executed-action 约定一致）；`s = float(torch.linalg.vector_norm(a_prev[0].detach().float() - a_prev2[0].detach().float()))`（L2、float32、detach、原 device 上算再转 python float）。
- > 提案 Eq 22 的 `s⁻`（需 D⁻）与 `Δ⁺`（需先搜一次）**本期不可得**，押后 roadmap Phase 3+。

**`ConstantDepthPolicy(depth)`** — 恒返回 `depth`。**非回归默认**；`depth` 由 config 保证 ∈ `allowed_depths`。

**`HeuristicDepthPolicy(allowed_depths, smoothness_thresholds, fallback_depth)`** — 确定性分桶规则（写死）：
- 记 `depths_desc = sorted(allowed_depths, reverse=True)`（深→浅）；`smoothness_thresholds` 升序、长度 == `len(allowed_depths) − 1`。
- `select(features)`：`s = features.action_smoothness`
  - `s is None`（早步 <2 动作）→ 返回 `fallback_depth`（默认 `min(allowed_depths)`）。
  - 否则 `idx = sum(1 for t in smoothness_thresholds if s >= t)`；返回 `depths_desc[idx]`。
  - 语义：平滑（`s` 小 = 稳态）取更深；突变（`s` 大 = 重规划）取更浅。半开桶 `[t_i, t_{i+1})`，`>=` 定 tie，确定性、无随机。
- 构造即校验 `len(smoothness_thresholds)==len(allowed_depths)−1` 且严格升序、`fallback_depth ∈ allowed_depths`，否则 raise（与 §3.3 config 校验双保险）。输出恒 ∈ `allowed_depths`（by construction）。

### 3.2 新增 `DynamicDepthKnnStrategy(TrajectoryMixin)`（src）

构造参数：`storage`、`base_fusion`（"weighted_rrf" | "weighted_score_sum"）、复用现有融合参数（`top_k/step_filter/step_window/fusion_weights/rrf_k/field_similarity/score_normalization`）、`depth_policy: DepthPolicy`、`allowed_depths`、`max_depth: int`（= `trajectory_depth`）、`max_weights: list[float]`（= `trajectory_weights`，长度 == max_depth）。经 `_init_trajectory(max_depth, max_weights)` 装配 history/session 机构。

`search(ctx)` 流程：
1. `self.record_query_keys(ctx.query_keys)`（与现有策略一致，保 history/query_id gap-free）。
2. 组装 `DepthFeatures`（step / history_len / action_smoothness）。
3. `T_sel = depth_policy.select(features)`。**合法性契约**：`if T_sel not in self._allowed_depths: raise ValueError(...)`（fail-loud，防越界深度悄悄进 QuerySpec）。用**显式 `raise` 而非 `assert`**——后者在 `python -O` 下会被剥除。随后 `effective_depth = min(T_sel, len(self._query_history))`（history-availability clamp，与老 `_build_trajectory_fields` 的 `actual_depth` 同义）。
4. 构造 QuerySpec（**不做任何 renorm**，逐符号复刻老 `_build_trajectory_fields` 的权重切片语义）：
   - `effective_depth <= 1` → 单步（不带轨迹字段，等同现有单步路径）。
   - `effective_depth > 1` → `trajectory_history` = `reversed(self._query_history[-effective_depth:])`（newest-first）；`trajectory_weights` = `max_weights[:effective_depth]`（**原样前缀，不归一**）；带上 `search_session_id` + `reversed(self._query_id_history[-effective_depth:])`（与老 memo 接线一致，深度换成 `effective_depth`）。
   - `fusion_method` = `base_fusion`；**按 base_fusion 分支镜像现有两策略的 QuerySpec 形状**（rrf → `backend_hints={"rrf_k":..}`、无 score_normalization；score_sum → 传 `score_normalization`、无 backend_hints）以保逐值 parity。
   - > **无 renorm 的理由**：老策略在 partial history 时用**未归一**前缀权重 `trajectory_weights[:actual_depth]`（search_strategy.py:167）。若对截断向量归一，constant 档在早期步（history<max）就与老策略分叉。Phase 1 两条路径（constant / heuristic）**都**用原样前缀权重 → dynamic 选深度 T 等价"老策略在 depth=T 下的行为"，constant@max 逐值等价老策略（**所有** history 长度）。提案 Eq 7 的 Σα=1 归一**押后**：若 Phase 7 评测显示跨深度可比性重要，再加一个**与 history-clamp 分离**的可选 renorm 开关（明确区分"意图性选浅"与"历史不足降深"）。
5. `return self._storage.search(spec)`。

生命周期方法（`on_episode_start`/`record_action`/`get_search_session_id`）全部继承 `TrajectoryMixin` → 跨步 score-memo、search-session 与现有策略同构，无需重写。

**为什么是新类而非改老类**：隔离 + 最小改动。老三策略逐字节不动 → 现有 golden 与生产 YAML 零风险。

### 3.3 config 接线（factory 层，additive）

**新增 `DepthPolicyConfig` dataclass**（config.py，紧邻 `SearchStrategyConfig`）：

```python
@dataclass
class DepthPolicyConfig:
    type: str = "constant"                              # "constant" | "heuristic"
    depth: Optional[int] = None                         # constant: 固定深度（None → trajectory_depth）
    smoothness_thresholds: Optional[list[float]] = None # heuristic: 升序, 长度 == len(allowed_depths)-1
    fallback_depth: Optional[int] = None                # heuristic: smoothness 未定义时深度（None → min(allowed_depths)）
```

- **解析注册（已亲验 loader）**：在 `_CONFIG_TYPES`（config.py:505）注册 `"DepthPolicyConfig": DepthPolicyConfig`。之后 `depth_policy: Optional[DepthPolicyConfig]` 由 loader 的**通用单嵌套 dataclass 路径**（`_dict_to_dataclass` config.py:635，靠 `_resolve_type` config.py:544-553 读 `_CONFIG_TYPES` 把注解 `Optional[DepthPolicyConfig]` 解析成类型）materialize 成 `DepthPolicyConfig` 实例——**无需** `field_similarity` 那种 per-field 特判（:627，因其为 `dict[str, X]`）。仅注册即足；§5 有解析 round-trip 测试兜底。
- `_valid_strategy_types` 增 `"dynamic_depth_knn"`（config.py:1229）。
- `SearchStrategyConfig` 增 3 个**可选**字段（默认使非本策略 YAML 逐字节不变）：
  - `base_fusion: Optional[str] = None`（仅 dynamic_depth_knn 用）
  - `allowed_depths: Optional[list[int]] = None`（None → `[trajectory_depth]`，即常数=非回归）
  - `depth_policy: Optional[DepthPolicyConfig] = None`（None → 等价 constant@trajectory_depth）
  - 复用现有 `trajectory_depth`（max_depth）+ `trajectory_weights`（max_weights；未归一原样切片，见 §3.2/§3.4）。
- `_build_search_strategy` 增 `elif cfg.type == "dynamic_depth_knn"` 分支：据 `base_fusion` + 构造 `depth_policy`（constant/heuristic 实例）+ `allowed_depths`（None→`[trajectory_depth]`）构造 `DynamicDepthKnnStrategy`，透传 `min_top_k_hint`（effective_top_k=max(top_k,hint)）。
- **校验**（`validate_cache_config`，config.py:1465 区）新增：
  - `dynamic_depth_knn` → `backend.type=="in_memory"`。
  - `base_fusion ∈ {"weighted_rrf","weighted_score_sum"}`；`=="weighted_score_sum"` → 复用现有 `score_normalization` 必需校验。
  - `allowed_depths`：非空、去重严格升序、每个 ∈ `[1, trajectory_depth]`；`trajectory_weights` 长度 == `trajectory_depth` 且非负（复用现有 trajectory 校验，config.py:1801-1820）。
  - **policy 输出合法性**：
    - `depth_policy.type ∈ {"constant","heuristic"}`。
    - constant：`depth`（None→`trajectory_depth`）必须 ∈ `allowed_depths`。
    - heuristic：`smoothness_thresholds` 非空、严格升序、长度 == `len(allowed_depths)−1`；`fallback_depth`（None→`min(allowed_depths)`）∈ `allowed_depths`。
    - 运行期 `DynamicDepthKnnStrategy.search` 对 `select()` 输出用**显式 `raise ValueError`**（非 `assert`，避免 `python -O` 剥除）兜住自定义/未来 policy 的越界。

### 3.4 退化契约（非回归，roadmap D2 硬要求）

**`dynamic_depth_knn` + `depth_policy: constant`（depth=D=trajectory_depth）+ `base_fusion=F` + `trajectory_weights=W`** 必须与 **现有 `F` 策略 @ `trajectory_depth=D, trajectory_weights=W`** 在**任意 history 长度**（含早期 partial history）产出**逐值相同的 QuerySpec 与 `search()` 结果**。

**实现保证**：§3.2 已定 dynamic 路径**不做 renorm**，`effective_depth = min(select, history)`、`weights = max_weights[:effective_depth]`——与老 `_build_trajectory_fields`（search_strategy.py:158-177：`actual_depth=min(depth,len(history))`、`trajectory_weights[:actual_depth]` 未归一）逐符号一致。故 constant@max 档在 step 0/1/2…（history 从 0 增长）**每一步**都与老策略同值。**这条契约由 §5.1 的 partial-history golden（从 step 0 起整段配对，含 D=3）强制验证**。

---

## 4. 改动文件清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `src/openpi/cache/components/search_strategy.py` | 新增 `DepthFeatures` / `DepthPolicy` Protocol / `ConstantDepthPolicy` / `HeuristicDepthPolicy` / `DynamicDepthKnnStrategy` | 新增（套现有 Protocol） |
| `src/openpi/cache/config.py` | 新增 `DepthPolicyConfig` dataclass + 注册 `_CONFIG_TYPES`（:505）；`_valid_strategy_types` +1；`SearchStrategyConfig` +3 可选字段；`_build_search_strategy` +1 分支；`validate_cache_config` +校验（含 policy 输出合法性） | additive 工厂/校验 |
| `tests/cache/test_search_strategy.py` | 新增 dynamic 用例（见 §5，含 partial-history parity / policy 解析 / 越界 fail-loud）；不改现有用例 | 测试 |
| `exp/zixuan_proposal/config/` | 2 份示例 YAML：`dynamic_depth_constant_baseline.yaml`（非回归对照）+ `dynamic_depth_heuristic.yaml` | 实验配置 |
| `docs/cache/tutorial.md` | §7 SearchStrategy 表加 `dynamic_depth_knn` 一行 + 简述 | 文档 |
| `docs/cache/README.md` / `docs/README.md` / `logs/README.md` | 索引同步（tutorial 改动 → **cache 子目录索引 `docs/cache/README.md`** + 顶层 `docs/README.md` 同 commit；本 plan 归档时 `logs/README.md`） | 文档索引 |

**不改**：任何 Protocol 体、`orchestrator.py`、`interceptor.py`、`backend_base.py`、`in_memory_backend.py`、`cache_storage.py`、`storage_types.py`、`judge.py` 及 verdict 全链。

---

## 5. 测试策略

1. **非回归 golden（核心，含 partial history）**：从 **step 0 起整段**驱动一个 episode（history 由 0 增长），`DynamicDepthKnnStrategy(constant@D, F, W)` 与 `WeightedRrfKnnStrategy`/`WeightedScoreSumKnnStrategy(trajectory_depth=D, trajectory_weights=W)` **逐步**配对，断言每步 winner id 序列 + score 逐值相等——两种 `base_fusion` × **D∈{1,3,5}**，其中 **D=3 partial-history**（step 1/2 时 history<D）为必测点。
2. **HeuristicDepthPolicy 单测**：构造特征（`s=None` 早步 / 小 `s` 平滑 / 大 `s` 突变 / 恰在 threshold 边界 `s==t` 验 `>=` tie）→ 断言选出深度符合分桶规则且 ∈ `allowed_depths`。
3. **config load + 校验**：合法 dynamic YAML（constant + heuristic）能 build；非法（allowed_depths 越界/空/非升序、base_fusion 未知、score_sum 缺 score_norm、非 in_memory、constant depth ∉ allowed、heuristic thresholds 长度错/非升序、fallback_depth ∉ allowed）各触发对应报错。
4. **depth_policy 解析 round-trip**：`load_cache_config` 后断言 `cfg.checkpoints[cp].search_strategy.depth_policy` 是 `DepthPolicyConfig` 实例（**非 raw dict**），字段值与 YAML 一致（constant 与 heuristic 各一）。
5. **policy 输出合法性 + history clamp**：注入返回越界深度的假 policy → `search()` 显式 `raise ValueError` fail-loud；`history_len < select` → `effective_depth=min` 生效、weights 前缀截断、不越界。
6. **search-session memo 仍启用**：dynamic 路径带 `search_session_id`/`trajectory_query_ids`，跨步 memo 命中（`force_legacy_path` parity 或 memo 命中计数断言）。
7. **§6 Verify**：`uv run pytest tests/cache/`（本期 blast-radius；裸 pytest 改动目录，不 repo-wide、不碰 tests/serving/review_tests）。

---

## 6. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | dynamic 选浅深度使 score_sum 聚合量级变小（少几个权重项）→ 与 judge 阈值交互 | 此为**深度选择固有**（与老策略 partial-history 同语义），**非新 renorm artifact**（Phase 1 已彻底去 renorm）；constant 默认档不受影响；阈值敏感性留 Phase 7 评测标注 |
| R2 | 早步 < 2 动作 → smoothness 未定义 | policy 收 None → `fallback_depth`（默认最小允许深度） |
| R3 | 选中深度 > 可用历史 | §3.2 step 3 `effective_depth=min(select,history)` + 前缀权重截断（不归一），与老 `actual_depth` 同 |
| R4 | 变深度扰动跨步 memo | memo 键含稳定 `query_id`，与深度无关；§5.6 用例验证 memo 不污染 |
| R5 | config 向后兼容 | 新字段全可选 + 默认；非 dynamic YAML 逐字节不变（现有 test_config 回归护栏） |
| R6 | in_memory-only 未拦截导致 qdrant 深度>1 崩 | §3.3 校验强制 in_memory |
| R7 | history 不足致 constant parity 破坏 | **resolved-by-design**：无-renorm 前缀权重语义（§3.2/§3.4）使 constant@max 在任意 history 长度逐值等价老策略；§5.1 partial-history golden（含 D=3）强制守卫 |

---

## 7. 设计决策（G1 已定）

- **D-O1 HeuristicDepthPolicy 规则形状**：`constant` + 一条确定性 smoothness 分桶 `heuristic`（§3.1 已写死 schema/阈值/tie/fallback/`action_smoothness` 计算）。更复杂靶向留后期。
- **D-O2 深度语义与提案 `{0,3,5,8}` 对齐**：本项目约定深度 ≥1（1=单步）↔ 提案 `T=0`（无链）。allowed_depths 用 `{1,3,5,8}`，`docs/cache/tutorial.md` 标注映射。
- **D-O3 base_fusion 复用方式**：**定为新类内联复用融合参数**（不组合持有既有 strategy 实例）——更少间接、QuerySpec 构造透明、便于 constant 逐值 parity（按 base_fusion 分支镜像现有两策略的 QuerySpec 形状）。
- **D-O4 weights 归一**：**Phase 1 不做任何 renorm**，两条路径都用未归一前缀 `max_weights[:effective_depth]`（= 老策略 partial-history 语义），保证 constant 全 history 长度非回归。Eq 7 的 Σα=1 押后为"与 history-clamp 分离的可选 renorm 开关"（Phase 7 若需跨深度可比性再议）。

---

## 8. 出场条件（本期 Code→G2→Verify 的完成定义）

- 非回归 golden（§5.1）+ 全部 §5 用例通过；
- 现有 `tests/cache/` 全绿（老策略逐字节不变）；
- 2 份 exp YAML 能 `load_cache_config` 过校验。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-07 19:31 CDT

Scope: reviewed post-G1 implementation against this approved plan, including `src/openpi/cache/components/search_strategy.py`, `src/openpi/cache/config.py`, `tests/cache/test_search_strategy.py`, docs/index updates, and the two `exp/zixuan_proposal/config/` YAMLs.

Verdict: **NEEDS REVISION**. The core dynamic-depth implementation and existing cache-suite regression tests are green, but two approved-plan contracts are not yet satisfied.

Blocking findings:

1. **`allowed_depths: []` is silently accepted instead of rejected.**
   - Plan contract: §3.3 requires `allowed_depths` to be non-empty; §5.3 requires illegal `allowed_depths` empty/non-ascending/out-of-range cases to raise validation errors.
   - Implementation evidence: `src/openpi/cache/config.py:1543` uses `allowed = list(ss.allowed_depths) if ss.allowed_depths else [ss.trajectory_depth]`, so explicit `[]` takes the same default path as `None`. `_build_search_strategy` repeats the same falsey fallback at `src/openpi/cache/config.py:2859`.
   - Reviewer probe result: constructing a `dynamic_depth_knn` `CacheConfig` with `allowed_depths=[]` and calling `validate_cache_config(cfg)` printed `ACCEPTED_EMPTY_ALLOWED_DEPTHS`.
   - Test gap evidence: current dynamic config tests cover out-of-range, bad base fusion, constant-depth-not-in-allowed, and threshold length, but no empty/non-ascending/fallback-not-in-allowed YAML case is present around `tests/cache/test_search_strategy.py:691-720`.
   - Required fix: distinguish `None` from explicit empty list. `None` may default to `[trajectory_depth]`; `[]` must raise `ConfigValidationError`. Add the missing YAML validation tests from §5.3, especially empty list and non-ascending list.

2. **`logs/README.md` index is stale and contradicts the approved semantics.**
   - `logs/README.md:40` still marks this log as ``Plan (待 G1, 2026-07-07)`` even though the plan header is `G1 APPROVED 2026-07-07` and code has been implemented for G2.
   - The same row still says truncated weights are normalized to `Σ=1`, while this approved plan explicitly says Phase 1 does **not** renormalize (`§3.2`, `§7 D-O4`), and the implementation/test path uses unrenormalized prefixes.
   - Required fix: update the active log row to reflect the current G1-approved / code-done / G2-pending-or-needs-revision state and remove the stale renormalization claim.

Verification run by reviewer:

- `PYTHONPATH=. uv run pytest tests/cache/test_search_strategy.py tests/cache/test_config.py -q` — **182 passed**, 8 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/ -q` — **978 passed, 6 skipped**, 16 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python -c "... load_cache_config(exp/zixuan_proposal/config/*.yaml) ..."` — both dynamic-depth exp YAMLs loaded successfully.
- `git diff --check` — clean.
- Reviewer boundary probe for `allowed_depths=[]` — **failed expected behavior** by accepting the config.

### G2 Round 1 — Executor — 2026-07-07

回应 Reviewer 2 项 blocking，逐条以项目事实评估后**全部 Accepted**，代码/文档/测试已改。

- Accepted（finding 1，`allowed_depths: []` 被静默接受）— 真 bug，确认。两处 falsey coalescing 把显式空列表当 None 默认：`config.py:1543`（validation）与 `config.py:2859`（`_build_search_strategy`）都用 `if ss/cfg.allowed_depths else [trajectory_depth]`，空列表走了默认分支。**修正**：两处改为 `if ... is not None else ...`，区分 `None`（用默认 `[trajectory_depth]`）与显式 `[]`（validation 的 `if not allowed:` 随即报 "allowed_depths must be non-empty"；factory 只对 None 兜底，`[]` 若绕过 validation 也在下游 fail-loud）。**测试补齐（§5.3 缺口）**：新增 `test_dynamic_allowed_depths_empty_rejected`（constant YAML `allowed_depths: []` → match "non-empty"）、`test_dynamic_allowed_depths_non_ascending`（`[5,3,1]` → match "ascending"）、`test_dynamic_heuristic_fallback_not_in_allowed`（`fallback_depth: 2 ∉ [1,3,5]` → match "fallback_depth"）。
- Accepted（finding 2，`logs/README.md:40` 陈旧且矛盾）— 索引行仍写 `待 G1` 且保留旧的"截断按 Eq 7 归一 Σ=1"，与本 G1-APPROVED 计划的"Phase 1 不做 renorm"（§3.2 / §7 D-O4）冲突。**修正**：状态改为 `Plan (G1 APPROVED 2026-07-07; §4 Code done, G2 迭代中)`；renorm 措辞改为"截断用未归一前缀权重(Phase 1 不做 renorm，保 constant 全 history 长度非回归;Eq 7 的 Σ=1 押后)"，与计划/实现/测试一致。

本轮为 G2 code 修订。已跑本地 §4 advisory 自检（见下方汇报）；正式 §6 Verify 属 G2 APPROVED 之后。重新进入 G2，请 Reviewer 复审。

### G2 Round 2 — Reviewer — APPROVED — 2026-07-07 19:54 CDT

Scope: re-reviewed the executor response to G2 Round 1, focusing on the two prior blocking findings (`allowed_depths=[]` validation and `logs/README.md` stale index/renorm wording), then reran the dynamic-depth tests, cache-suite regression, exp YAML load checks, and an independent boundary probe.

Verdict: **APPROVED**.

Findings:

- Prior blocking finding 1 is resolved. `src/openpi/cache/config.py` now distinguishes `None` from an explicit empty list in both validation and factory construction, so `None` still defaults to `[trajectory_depth]` while `allowed_depths: []` is rejected. The reviewer probe now reports `REJECTED_EMPTY_ALLOWED_DEPTHS checkpoints.cp1.search_strategy: allowed_depths must be non-empty`.
- Prior blocking finding 2 is resolved. `logs/README.md` now reflects `G1 APPROVED 2026-07-07; §4 Code done, G2 迭代中` and describes Phase 1's unrenormalized prefix-weight semantics instead of the stale `Σ=1` truncation claim.
- No new blocking findings found in this G2 re-review.

Verification run by reviewer:

- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/test_search_strategy.py tests/cache/test_config.py -q` — **185 passed**, 8 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/ -q` — **981 passed, 6 skipped**, 16 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python -c "... load_cache_config(exp/zixuan_proposal/config/*.yaml) ..."` — both dynamic-depth exp YAMLs loaded successfully.
- Independent reviewer probe for `allowed_depths=[]` — rejected with `ConfigValidationError`.
- `git diff --check` — clean.
