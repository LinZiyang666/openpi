# TRACER Phase 3 — M2 失败感知骨架（惰性 D⁻ + 手设参数）Plan

- **Status**: Implemented（G1 R3 + G2 R1 APPROVED + §6 Verify green `tests/cache/` 1056 pass / 6 skip，2026-07-08；两示例 YAML build-verified；G2 Review Log 永久保留；待 owner 确认归档）
- **Date**: 2026-07-08
- **Level**: **L3**（跨模块 additive 缝 + 新子系统语义：双检索 + 失败感知判决门；触及 storage_types / backend / orchestrator / composite_judge / config 五处 additive 缝 + 2 新组件。需附架构文档更新）
- **上位依据**: [`tracer_retrieval_refinement_roadmap.log.md`](tracer_retrieval_refinement_roadmap.log.md) Phase 3（M2）。需求由该 roadmap 的 Phase 3 spec 固化（owner 经 `/goal` 授权推进）。roadmap §4 Phase 3 明列 3a/3b/3c 三可审子块、退化契约、出场 gate、Level L3。
- **子实验目录**: `exp/zixuan_proposal/`（源提案 PDF 已归此目录；Phase 1/2 沿用）。
- **提案锚点**: Eq 16–18（双检索 margin `m = s⁺ − λ·s⁻`）、Eq 19–21（三态门 `g_t = σ(β₀ + β₁m + β₂u + β₃Δ⁺)`）、Eq 20（正样本歧义 `Δ⁺`）。本期只做**免训练手设默认**版；标定 `τ/λ/β` 押 roadmap Phase 5，真 D⁻ 数据押 Phase 4。

---

## 1. 无上下文摘要（供 G1 reviewer / 实现者，零对话史可读）

**背景**：本项目推理 cache 系统是可插拔的（`QueryKeyBuilder` / `SearchStrategy` / `SimilarityJudge` 三个 Protocol + config 工厂 frozenset/`_build_*` 分支注册）。合作者提案 M2 = **失败感知双检索**：除现有成功库 D⁺ 外，引入失败库 D⁻；对 query 各检索一次，算 `s⁺`（正库最高相似度）、`s⁻`（负库最高相似度）、`margin = s⁺ − λ·s⁻`、`Δ⁺`（正库歧义度）；再由一个 σ 判决门 `g_t = σ(β₀+β₁·margin+β₂·u_t+β₃·Δ⁺)` 产出 FULL_HIT / WARM_START / MISS 三态，`u_t` 复用现成 4 层 verdict-factor judge 的 kinematic 因子。

**本期目标（骨架，非训练）**：把 M2 的**代码结构**全部立起来并单测，**参数全部 YAML 手设**、**D⁻ 用 fixture / 空池**。关键安全性质（roadmap D2）：**`outcome` 缺省 + D⁻ 空 + 手设默认参数时，整条链退化 == 现「success-only 单池 + 阈值判决」，且不扰动任何现有 config（逐字节）**。真数据在 Phase 4，标定在 Phase 5。

**三可审子块**：
- **3a 缝**（additive-only）：`QueryFilter.outcome` + `CacheEntry.outcome` 标签（D5 单-artifact 分池；含 `load_artifact` 旧 pickle backfill）+ in_memory backend 过滤；`retrieval_signals` 侧信道（strategy → orchestrator → judge）。默认 None → 全旧路径逐字节不变。
- **3b 策略**：新 `DualRetrievalKnnStrategy`（对 D⁺/D⁻ 各搜一次算 margin/Δ⁺；D⁺ 内部 over-fetch ≥2 供 Δ⁺；D⁻ 空 → s⁻=0 → margin=s⁺；复用 Phase 1 `DepthPolicy` 机构，常数深度默认=非回归）。
- **3c 门**：新 `SimilarityJudge` 类型 `failure_aware_gate`（σ 门；三态；直读 `results[0].score`+`retrieval_signals`；`β/τ/λ` YAML 手设）。默认参数下逐值退化为对 `s⁺` 的阈值判决（==`ThresholdJudge`）。**`β₂·u_t`（kinematic 项）分期 Phase 5**（依赖 4 层 calibration，与「零负担退化」冲突；见 §3c / D-P3 / D-P9）。

**隔离律遵守**：只 `SearchStrategy` 造 `QuerySpec`/调 `storage.search`；judge/composer 只读不写；所有框架触点 additive（默认惰性 → 旧行为）。不改任何 Protocol 语义、不改 backend ABC、不改 orchestrator/interceptor/judge 内核语义（仅按现有 `view`/`history` 先例加一个默认 None 的注入 kwarg）。

---

## 2. 现状事实（已亲验，附锚点）

| 事实 | 锚点 |
|---|---|
| `QueryFilter` 仅 `task_key` / `step_range`（各 `Optional=None`） | `storage_types.py:164-178`（字段 :177-178） |
| `CacheEntry` 字段：id/checkpoint_id/query_keys/payload/step_idx/timestamp/prev_ids/next_ids/trajectory_id —— **无 outcome/success 标签** | `storage_types.py:117-152`（字段 :143-152） |
| `CachePayload` 字段：action_chunk/intermediates/denoising_num_steps/task_key/factors —— **无 outcome 标签**；`factors` 为开放 `dict[str,float]` | `storage_types.py:60-114`（:101-105） |
| `SearchResultLite` 仅 id/score/checkpoint_id | `storage_types.py:258-283`（:281-283） |
| `_check_filters` 用 `dataclasses.asdict(spec.filters)` 反射所有非-None 字段 → `requested`；`requested − backend.supported_filters()` 非空即 `raise UnsupportedFilterError`（**新增 QueryFilter 字段被 set 时自动进 requested，必须同步进每个 backend 的 `supported_filters()`**） | `cache_storage.py:296-311`（search 顶部调用 :100） |
| in_memory `supported_filters()` = `frozenset({"checkpoint_id","task_key","step_range"})` | `in_memory_backend.py:112-113` |
| in_memory `_filter_entries()` 线性扫 `self._entries.values()`，按 checkpoint_id / `entry.payload.task_key` / `entry.step_idx` 过滤 | `in_memory_backend.py:340-357` |
| `InMemoryBackend.load_artifact` 反序列化旧 pickle entries；**已有 backfill 先例**（`for entry in data["entries"]: if not hasattr(entry,"prev_ids"): entry.prev_ids=[]` 等，**:238-245**）+ `library_stats` 惰性重算 → **新增 `CacheEntry` 字段必须同址 backfill**（旧 pickle unpickle 绕过 `__init__`，实例无新属性，直接 `entry.outcome` 会 AttributeError） | `in_memory_backend.py:238-245` |
| qdrant `_SUPPORTED_FILTERS` 同上三项；`_make_filter` 建服务端 Filter | `backends/qdrant_backend.py:123-125,441-468` |
| 单 backend / 单 storage 装配；`search()` 打单索引（无 dual/outcome 池概念）；`build_shared_storage`（:2059）/`build_cache_components`（:2075）各 `_build_backend(config.backend)` 一次 | `config.py:2059-2080`；`_build_backend` :2258-2276；`BackendConfig` :394-399 |
| `SearchStrategy` Protocol：唯一必需方法 `search(ctx: SearchContext) -> list[SearchResultLite]` | `search_strategy.py:59,:71` |
| `SearchContext` 字段：query_keys/checkpoint_id/current_step/task_key | `search_strategy.py:40-55` |
| `TrajectoryMixin` 提供 `_init_trajectory` / `on_episode_start`（mint session uuid）/ `get_search_session_id` / `record_action` / `record_query_keys` / `_build_trajectory_fields`（reversed newest-first + **未归一**前缀权重 + session/query_ids） | `search_strategy.py:84-177` |
| Phase 1 `DynamicDepthKnnStrategy` + `DepthFeatures`(:388) / `DepthPolicy` Protocol `select(features)->int`(:404-413) / `ConstantDepthPolicy`(:418) / `HeuristicDepthPolicy`(:432)；`_trajectory_fields_for_depth(depth)`(:595-615) 建变长深度轨迹字段；`_action_smoothness`(:577-593) | `search_strategy.py:388-615` |
| orchestrator `check()`：`results = strategy.search(ctx)`（:541）→ 立即 `judge(results, checkpoint_id, cached_data, view=view, history=history)`（:553-557）；`top_score = results[0].score`（:566） | `orchestrator.py:534-566` |
| orchestrator getattr 读回策略方法先例：`getter = getattr(strategy, "get_search_session_id", None); if getter is None: continue`（:302-310）；lifecycle 签名过滤 `_safe_call_lifecycle`（inspect.signature 剔除未声明 kwarg，:330-336） | `orchestrator.py:302-310,324-336` |
| `SimilarityJudge` Protocol `__call__(results, checkpoint_id, cached_data, *, view=None, history=None)->JudgeResult`；legacy judge 用 `**kwargs` 吸收 view/history（ThresholdJudge :225）；`JudgeResult(hit_type,winner_id,start_t,composer_score,factor_outputs)` | `judge.py:104-112,59-80`；ThresholdJudge 决策仅看 `results[0].score`（:220-237） |
| `CompositeJudge.__call__(..., *, view=None, history=None)`：空 results 短路 MISS（:160-161，不进 composer）→ Layer2 factors extract → Layer3 calibrate → `self._composer.compose(calibrated, winner_id=results[0].id)`（:188）；`bind_keys` 在 `__init__` fail-fast（:121-123）；`__call__` **无 `**kwargs`**（:149-157 仅 view/history）→ orchestrator 无条件传新 kwarg 需其显式接受 | `composite_judge.py:149-211` |
| **CompositeJudge 强制四层俱全 ≥1 factor**：`if not factors: raise ValueError("CompositeJudge requires at least one Factor")`（`composite_judge.py:79-80`）；`_build_composite_judge` 强制 `normalization`+`factors`(≥1)+`calibration`+`composer` 全非 None（`config.py:2536-2552`），且 `_build_normalization` 需 `library_stats` → **「零因子 no-op composite 退化门」在现架构下不可行**（3c 因此改新 `SimilarityJudge`，见 §3c / D-P3） | `composite_judge.py:79-80`；`config.py:2536-2552` |
| `SearchStrategyConfig.top_k` 默认 **1** → `Δ⁺=top1−top2` 需 top-2，策略须内部 over-fetch（否则 `b3≠0` 时 Δ⁺ 恒 0 静默关项） | `config.py:335-346` |
| **`DumpingJudge` wrapper 无 `**kwargs`**：`__call__(results,checkpoint_id,cached_data,*,view=None,history=None)`（`dumping_judge.py:133-141`）只收 view/history，且转发 inner 仅 `view=,history=`（:143-145）→ orchestrator 无条件传 `retrieval_signals=` 会 **wrapper 先 `TypeError`**（打爆所有 `JudgeConfig.dump:` 配置，即便 inner 能吸收）；且 wrapper 须**转发** inner 才能让 inner judge 拿到 signals | `dumping_judge.py:133-145` |
| **warm_tiers validator 仅 threshold**：`validate_cache_config` warm_tiers 块要求 `judge.type=='threshold'`（`config.py:2003-2007`）+ CP1-only（:2009-2012）+ tier 严格递减 `<judge.threshold`（:2022）+ `start_t∈CANONICAL_DENOISE_TIMESTEPS`（:2030）。`JudgeConfig` 有 `threshold`（默认 **0.98**，score-scale）/`warm_tiers`（:287-288）→ `failure_aware_gate` 复用 warm_tiers 须扩此 validator | `config.py:1994-2035`；JudgeConfig :285-316 |
| Layer-4 `Composer` Protocol：`declared_dependencies: set[str]` / `bind_orientations(orientations)` / `compose(calibrated: dict[str,float], *, winner_id: str) -> JudgeResult`；子类自管 NaN | `components/factors/composers/base.py:22-61` |
| 现成 composer：`WeightedSumComposer` / `WeightedSumWithWarmFallbackComposer`（all-NaN→WARM）/ `WeightedSumZeroNanComposer`（two-tier FH/WS）/ `AndGate` / `OrGate` | `composers/__init__.py:69,212,289,437,492` |
| `_build_composer(cfg)` 干净 if/elif + 末尾 raise；`ComposerConfig`（type/weights/tier_thresholds/per_factor_thresholds/warm_start_t/warm_fallback_start_t/directions） | `config.py:2730-2809`；ComposerConfig :123-158 |
| kinematic descriptor（`u_t` 候选）：`jerk`(risky)/`direction`(safe)/`dispersion`/`path_length`；online 8 因子 verdict 时活算，key 模板 `<descriptor>_online_<channel>__p<P>_f<F>` | `_descriptor_kernel.py:133-180`；`factors/online.py`；orientations :73-83 |
| 策略类型注册：**局部** frozenset `_valid_strategy_types`（validate 内 :1282-1285）+ 校验块 :1567 + in_memory-gate tuple :1533 + `_build_search_strategy` elif :2857 + else 串 :2962 | `config.py` |
| judge 类型：module-level `_JUDGE_TYPES`（:508-513，含 `composite`）；`_build_inner_judge` 分支 :2487-2516（composite → `_build_composite_judge`） | `config.py` |
| 嵌套 config dataclass 三步注册：定义 `@dataclass` → 加入 `_CONFIG_TYPES`（:541）→ 父 dataclass 加 typed 字段；`_dict_to_dataclass`/`_resolve_type` 反射 materialize（Phase 2 `ProjectionKeyBuilderConfig` 为模板 :429-452,566） | `config.py:541-685` |
| 机制/参数分离范式：`ScoreNormalizer.from_params_dict`（在线加载手设 params）vs `fit_from_scores`（离线拟合）；本期 `β/τ/λ` 即"手设 params"，直接 YAML 字段（无 fit，fit=Phase 5 `calibrate`） | `score_normalizers.py:71-153` |
| 现存 success 信号只到 `interceptor.on_episode_end(success)`，**不进** CachePayload/CacheEntry/QueryFilter | `interceptor.py:363` |

---

## 3. 设计

> 结构 = 3a 缝 / 3b 策略 / 3c 门 三可审子块（roadmap Phase 3 授权拆分）。贯穿硬约束：**additive-only + 惰性默认退化 = 现系统**（roadmap D2/D4）。

### 3a — Additive 框架缝（默认 None → 逐字节不变）

#### A1. `outcome` 标签 + `QueryFilter.outcome`（D5 单-artifact 分池）

- `storage_types.py`：
  - `QueryFilter` 增 `outcome: Optional[int] = None`（语义：`+1`=成功/D⁺，`-1`=失败/D⁻，`None`=不按 outcome 过滤）。
  - `CacheEntry` 增 `outcome: Optional[int] = None`（entry 级标签，与 `step_idx` 同层；`None`=未标注。D5「单 artifact + entry 带 y_d」落点）。
- `in_memory_backend.py`：
  - `supported_filters()` 返回集加 `"outcome"`（→ `frozenset({"checkpoint_id","task_key","step_range","outcome"})`）。
  - `_filter_entries()` 增一分支，**用 `getattr` 防旧 pickle 缺属性**：`ent_outcome = getattr(entry, "outcome", None); if spec.filters.outcome is not None and ent_outcome != spec.filters.outcome: continue`（旧实例无 `outcome` 属性 → 视作 None-tagged，不匹配显式 outcome filter，**不 AttributeError**）。
  - **`load_artifact()` 旧 pickle backfill**（镜像现有 `prev_ids`/`next_ids`/`trajectory_id` backfill 循环，`in_memory_backend.py:238-245`）：在该 `for entry in data["entries"]` 循环内加 `if not hasattr(entry, "outcome"): entry.outcome = None` → 旧 artifact 显式 None-tagged，与本节「现有 artifact 全 None-tagged」声明一致。两处（getattr 读 + load_artifact 写）互为兜底。
- **qdrant 不改**：M2 双检索 gated 到 in_memory（见 3b / 校验），qdrant config 永不 set `QueryFilter.outcome` → `_check_filters` 不会 raise。
- **退化/非回归**：`QueryFilter.outcome` 缺省 None → `_check_filters` 的 `requested` 不含它 → 现有配置逐字节不变；写侧无人 set `CacheEntry.outcome`（默认 None）→ 现有 artifact 全 None-tagged。**骨架退化路径不使用 outcome 过滤**（见 3b：dual-off 时单次全池搜，无 outcome filter）；outcome 过滤仅在显式 dual-on + tagged fixture 下走（单测），真数据 Phase 4。
- **未标注语义（明确）**：显式 `outcome=+1` 过滤**只匹配 `entry.outcome==+1`**，不匹配 None-tagged。故对现有全-None artifact 直接开 dual 会得空 D⁺——**这是有意的**：骨架 dual-off 默认走全池（无 filter），dual-on 需要 tagged 库（Phase 4 建库时统一打 `+1`/`-1`）。此语义写进 tutorial + validator 提示。

#### A2. `retrieval_signals` 侧信道（strategy → orchestrator → judge）

- 新 `RetrievalSignals` dataclass（**定于 `storage_types.py`** —— 跨层只读数据契约，与 `QueryFilter`/`QuerySpec`/`SearchResultLite` 同址；strategy/orchestrator/judge 均从此单向 import，杜绝 `judge → search_strategy` 循环依赖）：
  ```python
  @dataclass(frozen=True)
  class RetrievalSignals:
      s_pos: float          # 正库最高相似度
      s_neg: float          # 负库最高相似度（D⁻ 空 → 0.0）
      margin: float         # s_pos - lambda_ * s_neg
      delta_pos: float      # 正库歧义度 Δ⁺（见 3b 定义）
      lambda_: float        # 生效的 λ（便于诊断/复现）
  ```
  **per-query 标量 side-channel，不塞进 per-result `SearchResultLite`**（roadmap 风险 #2 定夺）。
- **生产者（strategy）**：`DualRetrievalKnnStrategy` 每次 `search()` 末尾把本步 `RetrievalSignals` 存入 `self._last_signals`，并暴露 `last_retrieval_signals(self) -> Optional[RetrievalSignals]`（返回最近一次；非-DualRetrieval 策略无此方法）。
- **orchestrator 缝**（`orchestrator.py`，`results = strategy.search(ctx)` 之后 :541、judge 调用之前 :553）：
  ```python
  _sig_getter = getattr(strategy, "last_retrieval_signals", None)   # 先例 :303
  retrieval_signals = _sig_getter() if _sig_getter is not None else None
  ```
  judge 调用（:554-557）追加末位 kwarg：`judge(results, checkpoint_id, cached_data, view=view, history=history, retrieval_signals=retrieval_signals)`。策略无该方法 → None → 逐字节不变（gate-skip 早返回路径 :481-532 本就不调 judge，天然 None-safe）。
- **judge 侧**：`SimilarityJudge` Protocol `__call__` 增 keyword-only `retrieval_signals: Optional[RetrievalSignals] = None`（additive，照 `view`/`history` B1 先例）。legacy judge（Threshold/AlwaysHit/AlwaysWarmStart）已 `**kwargs` 吸收 → 忽略 → 逐字节不变。**新 `failure_aware_gate` judge 直接消费它**（见 3c）——signals 到 judge 即被使用，**不再下传 composer**。
- **CompositeJudge 缝**（`composite_judge.py:149`，`__call__` 无 `**kwargs`，:149-157 仅 view/history）：仅增形参 `retrieval_signals: Optional[RetrievalSignals] = None` 并**接受即忽略**（现有 5 composer 的 `compose(calibrated,*,winner_id)` 不涉 signals，无需签名过滤/透传，其 compose 签名逐字节不变）。此改**唯一目的**是让 orchestrator 无条件传入的新 kwarg 不使 composite judge `TypeError`。**不改** `Composer` Protocol、不改任何现有 composer。
- **DumpingJudge wrapper 缝**（`dumping_judge.py:133-145`，`__call__` 无 `**kwargs`）：`JudgeConfig.dump` 把 inner judge 包成 `DumpingJudge`，orchestrator 调的是 **wrapper** → 必须 `__call__` 增 `retrieval_signals: Optional[RetrievalSignals] = None` 并**转发 inner**（`self._inner(..., view=view, history=history, retrieval_signals=retrieval_signals)`），dump-factor 写行本身忽略 signals。**这样任何 `dump:` 配置（含 warmup）在新 kwarg 下不炸，且 inner=`failure_aware_gate` 时能拿到 signals**（G1 R2 finding 1 Accepted）。**至此 judge 侧完整名单**：`SimilarityJudge` Protocol（加参）+ legacy 三 judge（`**kwargs` 吸收）+ `CompositeJudge`（接受即忽略）+ `DumpingJudge`（接受并转发）——orchestrator 唯一调 judge 处，wrapper 仅 DumpingJudge 一个。

### 3b — `DualRetrievalKnnStrategy`（双检索 + margin/Δ⁺；复用 M3 深度）

`DualRetrievalKnnStrategy(TrajectoryMixin)`，`search_strategy.py` 新增。构造参数：
- `storage`、`base_fusion`（"weighted_rrf" | "weighted_score_sum"）、复用现有融合参数（top_k/step_filter/step_window/fusion_weights/rrf_k/field_similarity/score_normalization）。
- `lambda_: float`（margin 的 λ，≥0，默认 0.0 → margin=s⁺，即无 D⁻ 影响）。
- `enable_dual: bool`（默认 False → 单池全搜=非回归；True → 打 outcome filter 做双检索）。
- 深度机构（**复用 Phase 1**）：`depth_policy: DepthPolicy`（默认 `ConstantDepthPolicy(trajectory_depth)`）、`allowed_depths`、`max_depth`(=trajectory_depth)、`max_weights`(=trajectory_weights)。经 `_init_trajectory(max_depth, max_weights)` 装配。

**M3 复用方式（并入同一策略，最小改动）**：把 Phase 1 `DynamicDepthKnnStrategy` 现有的 `_trajectory_fields_for_depth(depth)`（:595-615）、`_select_effective_depth`（新析出）、`_action_smoothness`（:577-593）、`DepthFeatures` 组装逻辑**上移到 `TrajectoryMixin`**（纯行为保持的搬移；`DynamicDepthKnnStrategy` 改为继承调用，逐值不变，由 Phase 1 golden 复验守卫）。`DualRetrievalKnnStrategy` 与 `DynamicDepthKnnStrategy` 由此**共享**深度机构。
> 说明：Eq 22–23 深度策略吃 `s⁻/Δ⁺` 的高级 `DepthFeatures` 输入在**空 D⁻ 骨架下恒退化**，故本期 `DepthFeatures` 维持 Phase 1 三字段（current_step/history_len/action_smoothness），`s⁻/Δ⁺`-aware 深度为 **Phase 5 分期上线**（非本期偏差，仅输入分期）。常数深度默认即非回归。

`search(ctx)` 流程：
1. `self.record_query_keys(ctx.query_keys)`。
2. `effective_depth = self._select_effective_depth(ctx)`（共享机构：select + `min(T_sel, len(history))` clamp；越界 `raise ValueError` fail-loud）。
3. **D⁺ 检索（内部 over-fetch 供 Δ⁺）**：`pos_top_k = max(self._top_k, 2)`（**始终取 ≥2** 令 `Δ⁺=top1−top2` 良定义 —— 解 finding 2 的「`b3≠0` + 默认 `top_k=1` 静默置零」）。`pos_spec = self._build_spec_at_depth(effective_depth, top_k=pos_top_k, extra_filter_outcome=(+1 if enable_dual else None))`（`_build_spec_at_depth` 增 `top_k` 形参；按 `base_fusion` 镜像现有两策略 QuerySpec 形状；`extra_filter_outcome` None → 不加 outcome filter=全池）。`pos_full = self._storage.search(pos_spec)`。
4. `s_pos = pos_full[0].score if pos_full else 0.0`。`Δ⁺`（正库歧义度，Eq 20）：`delta_pos = pos_full[0].score - pos_full[1].score if len(pos_full)>=2 else 0.0`（top-1 与 top-2 差；库越"果断"Δ⁺越大；空/单结果→0）。**over-fetch 不改 top-1**（多取候选不动最高分）→ NR2 parity 不破。
5. **D⁻ 检索**（仅 `enable_dual`）：`neg_spec = self._build_spec_at_depth(effective_depth, top_k=1, extra_filter_outcome=-1)`（`s⁻` 只需 top-1）；`neg = self._storage.search(neg_spec)`；`s_neg = neg[0].score if neg else 0.0`。否则 `s_neg = 0.0`（D⁻ 空/禁用）。
6. `margin = s_pos - self._lambda * s_neg`。
7. `self._last_signals = RetrievalSignals(s_pos, s_neg, margin, delta_pos, self._lambda)`。
8. `return pos_full[:self._top_k]`（**返回配置 `top_k` 数**，与现有策略返回列表长度一致 → NR2 parity；额外 fetch 仅内部算 Δ⁺。winner = `pos_full[0]`，下游 judge 定）。

生命周期（on_episode_start / record_action / get_search_session_id）继承 `TrajectoryMixin`，跨步 memo 与现有策略同构。
**为何新类**：隔离 + 最小改动；现有四策略逐字节不动。

**退化（非回归，roadmap D2）**：`enable_dual=False` + `depth_policy=constant@trajectory_depth` + `base_fusion=F` → 每步单池搜（内部 over-fetch 不改 top-1）、常数深度、未归一前缀权重 → 返回 `pos_full[:top_k]` 与现有 `F` 策略 @同深度**逐值等价**（同 Phase 1 无-renorm 语义），`last_retrieval_signals` = `{s_pos=pos_full[0].score, s_neg=0, margin=s_pos, delta_pos=top1−top2}`。由 §5 NR2 golden 强制。

### 3c — `failure_aware_gate`（新 `SimilarityJudge` 类型，σ 门三态）

> **关键设计修订（G1 R1 finding 1，Accepted）**：本门**不做 Layer-4 Composer**。现架构 `CompositeJudge` 硬性要求四层俱全 + ≥1 factor（`composite_judge.py:79-80` `if not factors: raise`；`config.py:2536-2552` 强制 normalization/factors/calibration/composer 全非 None），故「零因子 no-op composite 退化门」**不可行**——与 roadmap D2 要求的「退化=ThresholdJudge、无 Layer-2/3 负担」直接冲突。解法（reviewer 明列选项之一）：**改为新独立 `SimilarityJudge` 类型 `failure_aware_gate`**，直读 `results[0].score` + `retrieval_signals`，退化路径**不经任何 factor/normalization/calibration 机器**。

新 `FailureAwareGateJudge`（`src/openpi/cache/components/judge.py` 或独立模块 re-export 进 judge.py），实现 `SimilarityJudge` Protocol `__call__(results, checkpoint_id, cached_data, *, view=None, history=None, retrieval_signals=None) -> JudgeResult`。

构造参数（来自 `JudgeConfig` 新字段 + 复用现有字段）：
- `gate_betas: dict[str,float]`（键 `b0/b1/b3`；Eq 21 的 β₀/β₁/β₃。**β₂ 本期强制 0**，见下 u_t 分期）。
- 门阈值：复用现有 `JudgeConfig.threshold` 作 σ 门值 `g` 的 full_hit 阈（**`failure_aware_gate` YAML 须显式设 `threshold: 0.5`** —— `JudgeConfig.threshold` 默认 0.98 是 score-scale，不适用 `g∈[0,1]`）+ 可选 `warm_tiers`（对 `g` 的 WARM_START 分档，CP1-only）。**warm_tiers 语义**：tier 的 `threshold` 是对 `g` 的阈（非 raw score），`start_t`=去噪时步，全程复用现有 ThresholdJudge warm 校验规则（严格递减 `<threshold` + CP1-only + start_t canonical，见注册）。

`__call__` 流程：
1. 空 `results` → `JudgeResult(HitType.MISS)`（照 ThresholdJudge/CompositeJudge 空短路）。
2. `retrieval_signals is None` → 配置错误（`failure_aware_gate` 必须配 `dual_retrieval_knn` 策略，见校验）→ 防御性 `raise ValueError`（fail-loud；合法配置永非 None）。
3. `m = retrieval_signals.margin`；`d = retrieval_signals.delta_pos`。（`results[0].score == retrieval_signals.s_pos` 恒等，§5 断言守卫。）
4. `z = b0 + b1*m + b3*d`；`g = 1.0 / (1.0 + math.exp(-z))`（σ）。
5. 三态（对 `g` 卡阈）：`g >= full_hit阈` → `FULL_HIT`(winner_id=results[0].id)；elif warm 档配了且 `g >= warm阈` → `WARM_START`(winner_id, start_t)（CP1-only）；else → `MISS`。`composer_score=g` 入 `JudgeResult` 供观测（`factor_outputs` 亦可选带 `{margin,delta_pos,g}`）。

**`u_t`（β₂·u_t kinematic 项）分期 Phase 5（D-P9）**：roadmap 3c 的 `u_t`="现成 **4 层 calibrated** kinematic 因子"，而 calibrated 值**只**在 CompositeJudge 全 4 层管线里产出——独立 judge 拿不到 calibrated 因子（只能拿 raw descriptor，语义已不同，且 raw 需 Layer-1 normalization 才可比）。u_t 与其标定天然属 Phase 5（标定期）。**本期 validator 强制 `gate_betas.get("b2", 0.0) == 0.0`**（fail-loud，防手滑配一个不生效的 β₂）；judge `__call__` 已收 `view`/`history`（orchestrator 无条件注入）→ Phase 5 接 u_t 时零 orchestrator 改动。**本期交付 M2 核心 = margin(+Δ⁺) 失败感知门（Claim 1）**；u_t 为 Phase 5 增量，非本骨架承诺内容。

注册（idiom A，照 `threshold` judge）：
- `judge.py` 加 `FailureAwareGateJudge` 类。
- `config.py`：`_JUDGE_TYPES`（:508-513）加 `"failure_aware_gate"`；`_build_inner_judge`（:2487-2516）加 `elif cfg.type == "failure_aware_gate":` 分支（校验 `gate_betas` 含 b0/b1、`b2==0`；构造 judge）+ 末尾 valid-list 串；`JudgeConfig` 加 `gate_betas: Optional[dict[str,float]] = None`（默认 None，仅新 type 用 → 现有 judge YAML 逐字节不变；`threshold`/`warm_tiers` 复用现有 `JudgeConfig` 字段）。
- **warm_tiers validator 迁移（G1 R2 finding 2 Accepted）**：`config.py:2003` 的 `if cp_config.judge.type != "threshold":` 改为 `if cp_config.judge.type not in ("threshold", "failure_aware_gate"):`（放行两类用 warm_tiers）；其余 warm_tiers 规则**全部不变复用** —— `always_warm_start` 等仍禁 warm_tiers、CP1-only（:2009-2012）、tier 严格递减 `<threshold`（:2022）、`start_t` canonical（:2030）。效果：合法 CP1 `failure_aware_gate + warm_tiers` 可 load/build，CP3 仍 reject，非支持 judge 仍 reject。

**退化（非回归，roadmap D2）**：手设默认 `gate_betas={b0:-τ, b1:1, b3:0}`（b2 恒 0）+ `threshold=0.5`（对 `g`）+ 无 warm → `g=σ(m-τ)`，`g≥0.5 ⟺ m≥τ`。dual-off/D⁻ 空 → `m=s_pos=results[0].score`。故 **FULL_HIT ⟺ results[0].score≥τ ⟺ `ThresholdJudge(threshold=τ)`**，逐值等价（σ 单调过 0.5 恰在 `m=τ`，与 β₁>0 无关）。Δ⁺ 项默认关（b3=0）→ 纯 margin 门 → **不经任何 factor/normalization/calibration**（干净退化，NR3/NR4 直接落地，无需空-Layer-2 composite）。

### 退化契约总表（roadmap D2 硬要求）

| 层 | 骨架惰性配置 | 退化为 | 证据 |
|---|---|---|---|
| 缝（3a） | 所有新字段/ kwarg 默认 None | 现有 config 逐字节不变 | NR1（现有 tests/cache/ 全绿） |
| 策略（3b） | `enable_dual=False`,常数深度 | 现有 `weighted_score_sum/rrf_knn`@同深度 | NR2（from-step-0 golden 逐值 + signals） |
| 门（3c，**新 judge**） | `b2≡0, b3=0, b0=-τ, b1=1, threshold=0.5` | `ThresholdJudge(threshold=τ)`（干净，不经 factor/norm/calib） | NR3（门数学单测）+ NR4（整栈 judge-to-judge golden，见 §5） |

---

## 4. 改动文件清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `src/openpi/cache/storage_types.py` | `QueryFilter` +`outcome`；`CacheEntry` +`outcome`；**新 `RetrievalSignals` dataclass（定于此，跨层只读契约）** | additive 数据契约 |
| `src/openpi/cache/backends/in_memory_backend.py` | `supported_filters` +"outcome"；`_filter_entries` +outcome 分支（`getattr` 防旧 pickle）；**`load_artifact` +outcome backfill**（镜像 prev_ids/next_ids 循环，:238-245） | additive backend 过滤 + 旧 pickle 兼容 |
| `src/openpi/cache/components/search_strategy.py` | 深度机构上移 `TrajectoryMixin`（`_trajectory_fields_for_depth`/`_select_effective_depth`/`_action_smoothness`/`DepthFeatures` 组装，行为保持；`_build_spec_at_depth` 增 `top_k` 形参供 over-fetch）；新 `DualRetrievalKnnStrategy` + `last_retrieval_signals`；`DynamicDepthKnnStrategy` 改继承共享（逐值不变） | 新组件 + 保行为搬移 |
| `src/openpi/cache/components/judge.py` | `SimilarityJudge` Protocol `__call__` +`retrieval_signals=None`（keyword-only）；**新 `FailureAwareGateJudge` 类**（σ 门三态，读 retrieval_signals） | additive Protocol + 新组件 |
| `src/openpi/cache/components/composite_judge.py` | `__call__` +`retrieval_signals=None`（**接受即忽略**，唯为不炸 orchestrator 无条件 kwarg；**不改** compose 调用点 / `Composer` Protocol / 任何现有 composer） | additive 缝 |
| `src/openpi/cache/components/dumping_judge.py` | `__call__` +`retrieval_signals=None`（**接受并转发 inner**：`self._inner(...,retrieval_signals=retrieval_signals)`）→ `dump:` 配置在新 kwarg 下不炸，inner 拿到 signals | additive 缝（wrapper 转发） |
| `src/openpi/cache/orchestrator.py` | getattr 读 `strategy.last_retrieval_signals()` → judge 调用追加 `retrieval_signals=` kwarg | additive 缝（2 行区） |
| `src/openpi/cache/config.py` | `dual_retrieval_knn`：`_valid_strategy_types`(:1282)+校验块(:1567)+in_memory-gate(:1533)+`_build_search_strategy` elif(:2857)+else 串(:2962)+`SearchStrategyConfig` 可选字段(`lambda_`/`enable_dual`)；`failure_aware_gate`：`_JUDGE_TYPES`(:508)+`_build_inner_judge` elif(:2487-2516)+末尾串+`JudgeConfig` 字段(`gate_betas`)；**warm_tiers validator :2003 放行 `failure_aware_gate`**；跨字段校验（judge↔strategy 配对、`b2==0`、warm CP1-only、in_memory-only、betas 合法） | additive 工厂/校验 |
| `tests/cache/test_search_strategy.py` | 新增 DualRetrieval 用例（NR2 golden / margin / Δ⁺ / signals / 越界 fail-loud）；不改现有 | 测试 |
| `tests/cache/test_judge.py`（新 `FailureAwareGateJudge` 用例，或新 `test_failure_aware_gate_judge.py`） | **新文件/用例**：门数学（`g=σ(z)`）/ 三态 / retrieval_signals None fail-loud / WARM CP1-only | 测试 |
| `tests/cache/test_config.py` | dual_retrieval + failure_aware YAML build/校验（合法 + 各非法分支，含 `b2≠0` reject / judge↔strategy 配对） | 测试 |
| `tests/cache/test_orchestrator*.py`（定位后） | retrieval_signals 缝：无 `last_retrieval_signals` 策略→None 逐字节；有→judge 收到并消费；legacy judge + composite judge 接受即忽略 | 测试 |
| `tests/cache/test_storage*.py` / backend 测试 | QueryFilter.outcome：supported_filters/_filter_entries/`_check_filters` raise（qdrant unsupported）/默认 None 不过滤；**旧 pickle backfill**（无 outcome 属性 entry → `load_artifact` 补 None + `getattr` 读不炸） | 测试 |
| `docs/architecture/cache_system.md` | §5.6（SimilarityJudge +`retrieval_signals` kwarg + `failure_aware_gate` judge）/§5.7（`QueryFilter.outcome`+`CacheEntry.outcome`+`RetrievalSignals`）/§5.8（DualRetrieval 策略）additive 更新——**L3 架构文档更新** | 文档 |
| `docs/cache/tutorial.md` | §7 SearchStrategy 表 +`dual_retrieval_knn`；§6 Judge 表 +`failure_aware_gate`；`outcome` 语义 | 文档 |
| `docs/cache/README.md` / `docs/README.md` / `logs/README.md` | 索引同步（同 commit） | 文档索引 |
| `exp/zixuan_proposal/config/` | 示例 YAML：`dual_retrieval_degenerate.yaml`（非回归对照）+ `failure_aware_gate_skeleton.yaml`（dual-on + fixture 语义示例） | 实验配置 |

**不改**：`SearchStrategy`/`Composer`/`SimilarityJudge` Protocol 的**现有**方法语义；backend ABC（`backend_base.py`）；`interceptor.py`；`cache_storage.py` 内核（仅 `_check_filters` 自动识别新字段，无需改代码）；qdrant backend；现有四策略 / 五 composer / 三 legacy judge 的行为（逐字节）。

---

## 5. 测试策略

**NR1（缝 additivity — 主非回归保证）**：现有 `tests/cache/` 全绿。所有新字段/ kwarg 默认 None/False/absent → 现有 config/orchestrator/judge/composer 逐字节不变。这是 roadmap「用现有 orchestrator/judge golden 证非回归」的落地。

**NR2（策略退化 golden，核心）**：从 **step 0 整段**驱动一 episode，`DualRetrievalKnnStrategy(enable_dual=False, constant@D, base_fusion=F, weights=W)` 与 `WeightedScoreSumKnnStrategy`/`WeightedRrfKnnStrategy(trajectory_depth=D, trajectory_weights=W)` **逐步**配对：断言每步 winner id 序列 + score 逐值相等；且 `last_retrieval_signals()` = `{s_pos=results[0].score, s_neg=0.0, margin=s_pos}`。两 `base_fusion` × **D∈{1,3,5}**（含 D=3 partial-history）。

**NR3（门数学单测）**：`FailureAwareGateJudge(gate_betas={b0:-τ,b1:1,b3:0}, threshold=0.5)` → 喂 `results=[SearchResultLite(score=s)]` + `retrieval_signals=RetrievalSignals(margin=m,...)`：断言 `m≥τ`→FULL_HIT(winner=results[0].id)、`m<τ`→MISS、边界 `m==τ`→FULL_HIT（g=0.5≥0.5）；`composer_score==σ(m-τ)`。

**NR4（整栈退化 golden，judge-to-judge）**：`[dual_retrieval(dual-off,const@D,F) + failure_aware_gate(默认 b2≡0,b3=0,b0=-τ,b1=1,threshold=0.5)]` 的 `JudgeResult` 序列 == `[weighted_*_knn(D,F) + ThresholdJudge(threshold=τ)]` 序列（hit_type/winner_id/start_t 逐步等）。**两侧皆轻量 judge（无 factor/normalization/calibration）→ 干净可跑**；finding 1 已由 3c 改新 judge 消解「空-Layer-2 composite」不确定性，NR4 不再依赖 CompositeJudge，亦不再需 NR2+NR3 拆分兜底。

**功能单测**：
- **双检索 margin/Δ⁺ + over-fetch**：fixture artifact（+1/-1 tagged entries）→ 断言 s_pos/s_neg/margin=s_pos−λ·s_neg/Δ⁺=top1−top2 正确；D⁻ 空 → s_neg=0；λ 生效。**over-fetch 专项**：`top_k=1` 配置 + `b3≠0` + 库有 ≥2 正样本 → Δ⁺ 非 0（不被静默置零）、且 `search()` 返回列表长度 == 1（NR2 parity 不破）。
- **retrieval_signals 缝**：无 `last_retrieval_signals` 的策略 → orchestrator 传 None（现有 orchestrator 测试逐字节）；DualRetrieval → `failure_aware_gate` judge 收到并消费；legacy ThresholdJudge + CompositeJudge 接受即忽略（不炸）；**`DumpingJudge` wrapper（包 legacy 或 failure_aware inner）在 `retrieval_signals=` 下不炸并转发 inner**（回归守 `JudgeConfig.dump:` 配置——`dump` 包 legacy inner 验证不炸、`dump` 包 failure_aware inner 验证 signals 透传到位）。
- **QueryFilter.outcome 缝 + 旧 pickle backfill**：in_memory `supported_filters` 含 outcome；`_filter_entries` 按 outcome 过滤；`_check_filters` 对 qdrant（不支持）`raise UnsupportedFilterError`；`outcome=None` → 不过滤（全返回）。**旧 pickle**：构造无 `outcome` 属性的 entry（模拟旧序列化，如 `del entry.__dict__["outcome"]`）→ `load_artifact` 后补 None；`_filter_entries` 用 `getattr` 读**不 AttributeError**，显式 `outcome=+1` filter 不匹配它（视作 None-tagged）。
- **门三态**：`g=σ(z)` 三态分档正确；WARM_START 仅 CP1（validator reject cp3 warm）；retrieval_signals None → `raise`。
- **越界深度 fail-loud**：注入返回越界深度的假 policy → `search()` 显式 `raise ValueError`。

**config load + 校验**：合法 `dual_retrieval_knn` + `failure_aware_gate` YAML 能 build；**正向 warm_tiers（finding 2）**：CP1 `failure_aware_gate + warm_tiers`（`threshold: 0.5` on g）能 load/build（三态门配置面可用）；非法各触发对应报错：judge=failure_aware 但 strategy≠dual_retrieval（配对缺失）、非 in_memory backend、`gate_betas` 缺 b0/b1、**`gate_betas.b2 ≠ 0`（本期未支持 u_t，fail-loud）**、**CP3 `failure_aware_gate + warm_tiers`（reject：warm_tiers CP1-only）**、**`always_warm_start/composite + warm_tiers`（仍 reject：非支持 judge）**、λ<0、base_fusion 未知、`enable_dual=True` 但 backend 无 outcome 支持。

**§6 Verify**：`uv run pytest tests/cache/`（本期 blast-radius；裸 pytest 改动目录，**不 repo-wide、不 `-m`、不碰 tests/serving/review_tests**——遵 execution §6 + 记忆护栏）。库侧 artifact 若不改则不纳入；如动 `exp/common` 建库则加 `tests/exp/...`。

---

## 6. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 新数据契约字段（QueryFilter/CacheEntry.outcome）破坏向后兼容 | 全 `Optional=None` 默认；`_check_filters` 反射自动不请求 None；现有 artifact 全 None-tagged；NR1 现有 tests/cache 守卫 |
| R2 | `retrieval_signals` kwarg 改 judge 调用扰动现有 judge | additive keyword-only 默认 None；legacy judge `**kwargs` 吸收；`CompositeJudge` 接受即忽略、`DumpingJudge` wrapper 接受并转发 inner；仅 `failure_aware_gate` 消费；现有 judge/composer 逐字节不变；NR1 守卫 |
| R3 | 深度机构上移 `TrajectoryMixin` 改动 Phase 1 代码 | 纯行为保持搬移；Phase 1 golden（`tests/cache/test_search_strategy.py` 现有 dynamic 用例）复验逐值不变；若 golden 破 → 回退为 DualRetrieval 内联复制（隔离 Phase 1） |
| R4 | ~~CompositeJudge 不接受空 Layer-2 → NR4 不成立~~ **已消解**（finding 1 Accepted） | 3c 改新 `SimilarityJudge`（不经 CompositeJudge）→ NR4 干净 judge-to-judge，无空-Layer-2 依赖 |
| R5 | 未标注（None）entry 与显式 outcome 过滤语义混淆 | 骨架 dual-off 默认全池无 filter（不碰 outcome）；dual-on 语义写进 tutorial + validator 提示；真 tagged 库 Phase 4 统一打标 |
| R6 | `u_t`（calibrated kinematic 因子）需 CompositeJudge 全 4 层，独立 judge 拿不到 calibrated 值 | **u_t 分期 Phase 5**（D-P9）；本期 validator 强制 `b2==0`（fail-loud）；judge 已收 view/history → Phase 5 接入零 orchestrator 改动；本期交付 M2 核心 margin 门（Claim 1） |
| R7 | Δ⁺ 定义分歧（top1−top2 vs 其他） | 本期定 `top1−top2`（正库果断度）明确写死；若 Phase 7 评测显示别的判别力更好再议（属参数/定义调整，非结构） |
| R8 | 双检索两次 search 延迟翻倍 | 属机制固有（D⁻ 检索）；dual-off 默认单搜=零额外；enable_dual 为 opt-in；延迟评测留 Phase 7 |
| R9 | WARM_START 误挂 CP3 | validator 复用现有 warm-start-CP1-only 规则 reject（§5 校验用例守卫） |
| R10 | qdrant 路径误触 outcome | M2 validator 强制 in_memory-only（照 dynamic_depth_knn）；qdrant 不改、不 set outcome |
| R11 | `Δ⁺=top1−top2` 在默认 `top_k=1` 下被静默置零（`b3≠0` 时悄关项）（finding 2 Accepted） | 策略内部 `pos_top_k=max(top_k,2)` over-fetch 算 Δ⁺，返回 `pos_full[:top_k]` 保 parity；over-fetch 不改 top-1；§5 over-fetch 专项测试守卫 |
| R12 | 旧 pickle artifact 反序列化缺 `CacheEntry.outcome` → 显式 outcome filter `AttributeError`（finding 3 Accepted） | `load_artifact` backfill（镜像 prev_ids/next_ids）+ `_filter_entries` 用 `getattr(entry,"outcome",None)`，双兜底；§5 旧 pickle 测试守卫 |
| R13 | orchestrator 无条件 `retrieval_signals=` 打爆 `DumpingJudge` wrapper（无 `**kwargs`）→ 所有 `dump:` 配置 `TypeError`（G1 R2 finding 1 Accepted） | `DumpingJudge.__call__` +`retrieval_signals=None` 接受并**转发 inner**；judge 侧完整名单（Protocol+legacy `**kwargs`+CompositeJudge 忽略+DumpingJudge 转发）；§5 dump wrapper 回归测试守卫 |
| R14 | `failure_aware_gate + warm_tiers` 被 `threshold`-only validator（`config.py:2003`）拒 → 三态门配置面不可用（G1 R2 finding 2 Accepted） | validator :2003 放行 `type in ("threshold","failure_aware_gate")`，余规则（CP1-only/严格递减/start_t canonical）全复用；§5 正向 CP1 + 负向 CP3/非支持 judge config 测试守卫 |

---

## 7. 设计决策（提交 G1 定夺）

- **D-P1 分池落点**：**单 artifact + `CacheEntry.outcome` tag + `QueryFilter.outcome` 过滤**（roadmap D5 倾向）。理由：装配层零改（避免第二 backend），缝最小，`_filter_entries` 天然支持。备选（第二 backend）弃：需改单-backend 装配层。
- **D-P2 `retrieval_signals` 形状 + 落址**：**独立 `RetrievalSignals` dataclass 侧信道**（不塞 `SearchResultLite`）（roadmap 风险 #2）；**定于 `storage_types.py`**（跨层只读契约，杜绝 `judge→search_strategy` 循环依赖，采纳 G1 R1 non-blocking 建议）。per-query 标量，strategy 暴露 `last_retrieval_signals()`，orchestrator getattr 读、judge kwarg 传。
- **D-P3 失败感知门 = 新 `SimilarityJudge` 类型 `failure_aware_gate`**（**非** Layer-4 Composer；G1 R1 finding 1 Accepted 后修订）。理由：`CompositeJudge` 硬性要求四层俱全 ≥1 factor（`composite_judge.py:79-80` / `config.py:2536-2552`），Composer 路线无法承载「退化=ThresholdJudge、零 Layer-2/3 负担」（roadmap D2）；独立 judge 直读 `results[0].score`+`retrieval_signals`，退化干净。代价：u_t（calibrated 因子）不可原生取 → D-P9 分期。
- **D-P4 `retrieval_signals` 到达 judge 即消费**（不再经 composer 透传）。orchestrator 唯一调 judge 处无条件传 `retrieval_signals=` → **judge 侧完整覆盖**：`SimilarityJudge` Protocol 加参；legacy 三 judge `**kwargs` 吸收；`CompositeJudge` 接受即忽略；`DumpingJudge` wrapper 接受并**转发 inner**（G1 R2 finding 1）。`Composer` Protocol 与现有 composer 逐字节不变。
- **D-P10 warm_tiers 复用 vs 新字段**（G1 R2 finding 2）：failure_aware_gate 三态 WARM 档**复用现有 `warm_tiers`**（tier `threshold` 对 `g` 卡阈），仅需 validator :2003 放行本 type（一行）——比新增 `gate_warm_threshold` 字段更省，且免费复用 CP1-only/严格递减/`start_t` canonical 校验。`threshold` 亦复用作 `g` 的 full_hit 阈（config 须显式设 0.5，默认 0.98 是 score-scale）。
- **D-P5 M3 深度并入**：DualRetrieval **复用** Phase 1 深度机构（机构上移 `TrajectoryMixin` 共享），常数深度默认=非回归。`s⁻/Δ⁺`-aware `DepthFeatures` 输入**分期到 Phase 5**（空 D⁻ 下恒退化，非本期偏差）。若 G1 认为搬移 Phase 1 风险过高，回退 R3 缓解（DualRetrieval 内联复制、零 Phase-1 触碰）。
- **D-P6 非回归基线**：NR1（缝 additivity）+ NR2（策略逐值）+ NR3（门逐值）+ **NR4（整栈 judge-to-judge == ThresholdJudge，干净可跑）**。因 3c 改独立 judge（不经 CompositeJudge），NR4 不再有空-Layer-2 不确定性、无需拆分兜底。
- **D-P7 Δ⁺ 定义 + over-fetch**：`top1_score − top2_score`（正库歧义/果断度；≥2 结果否则 0）。策略内部 `pos_top_k=max(top_k,2)` over-fetch 供 Δ⁺、返回 `pos_full[:top_k]`（G1 R1 finding 2 Accepted）→ 默认 `top_k=1` 下 `b3≠0` 不被静默置零。
- **D-P8 λ 归属**：λ 在**策略**（margin=s⁺−λ·s⁻ 由 strategy 算），非 judge；judge 只收算好的 margin。
- **D-P9 `u_t`（β₂ 项）分期 Phase 5**（G1 R1 finding 1 连带）：roadmap 3c 的 u_t="4 层 **calibrated** kinematic 因子"，calibrated 值只在 CompositeJudge 全管线产出，独立 judge 拿不到；u_t 与其标定天然属 Phase 5（标定期）。本期 validator 强制 `gate_betas.b2==0`（fail-loud），judge 已收 view/history → Phase 5 接入零 orchestrator 改动。**本骨架交付 M2 核心 = 失败感知 margin(+Δ⁺) 门（Claim 1）**；u_t 为 Phase 5 增量。若 G1 坚持 u_t 入本期，备选 raw-descriptor 版（复用 `_descriptor_kernel`+`view.walk_next`，语义 ≠ calibrated，需 P/F 窗口配置）。

---

## 8. 出场条件（本期 Code→G2→Verify 的完成定义）

- NR1（现有 tests/cache 全绿）+ NR2（策略退化 golden，两 fusion × D∈{1,3,5}）+ NR3（门数学）+ NR4（整栈 judge-to-judge == ThresholdJudge）通过；
- 全部 §5 功能单测（margin/Δ⁺/over-fetch/signals 缝/outcome 缝+旧 pickle backfill/三态/越界 fail-loud）通过；
- config load + 校验用例（合法 + 各非法）通过；
- 示例 YAML 能 `load_cache_config` 过校验；
- L3 架构文档（`docs/architecture/cache_system.md`）additive 更新 + 索引同步。

---

## Review Log

### G2 Round 1 — Reviewer — APPROVED — 2026-07-08 09:30 CDT

G2 code review approved. Implementation matches the G1 R3-approved design: `QueryFilter.outcome` / `CacheEntry.outcome` are additive and old pickle backfill is implemented; `RetrievalSignals` lives in `storage_types.py`; orchestrator forwards signals through the judge kwarg; legacy judges, `CompositeJudge`, and `DumpingJudge` cover the new kwarg; `FailureAwareGateJudge` is an independent `SimilarityJudge`; `DualRetrievalKnnStrategy` handles D⁺ over-fetch, D⁻ top-1, margin, Δ⁺, and dual-off non-regression; config validation and factories register `dual_retrieval_knn` / `failure_aware_gate`; docs and indexes are updated.

- [Non-blocking] Suggestion make the two example YAMLs buildable against the checked-in preload artifact, not only `load_cache_config`-valid — reasoning: both `exp/zixuan_proposal/config/dual_retrieval_degenerate.yaml` and `failure_aware_gate_skeleton.yaml` pass `load_cache_config`, satisfying the plan exit condition, but `build_cache_components` fails with the current `cp1_mean_pool.pkl` because the artifact vector dims include `vision_1` and `prompt_emb` while the examples declare only `vision_0` and `robot_state`. This does not block G2 because the plan required config-load validation, and the actual new factory branches were independently smoke-tested with a no-preload minimal YAML, but the examples should be made runnable before users copy them into a server run.

Checklist: consistency with approved plan = PASS; test coverage and passing = PASS (`PYTHONPATH=. uv run pytest tests/cache/test_failure_aware_gate_judge.py tests/cache/test_search_strategy.py tests/cache/test_in_memory_backend_experiment.py tests/cache/test_orchestrator.py tests/cache/test_orchestrator_history.py tests/cache/test_config.py -q` → 314 passed; `PYTHONPATH=. uv run pytest tests/cache/ -q` → 1056 passed / 6 skipped); docs & indexes updated = PASS; no regressions = PASS (full cache suite green, dual-off golden tests present, old pickle / wrapper / warm_tiers regressions covered).
