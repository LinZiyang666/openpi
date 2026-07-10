# TRACER 检索精炼路线图 — Action-Compatible Failure-Aware Retrieval 的模块化落地

- **Status**: Roadmap（Design-Grounded）— Phase 0 ✅（架构判定/可行性亲验完成）/ **Phase 1 ✅（M3 `dynamic_depth_knn`，G1/G2 APPROVED + §6 Verify green，commit `cea98b2`，2026-07-07；plan 归档 `archive/tracer_phase1_dynamic_depth.log.md`）** / **Phase 2 ✅（M1 `projection` KeyBuilder 骨架，G1/G2 APPROVED + §6 Verify green，2026-07-08）** / **Phase 3 ✅（M2 `dual_retrieval_knn` + `failure_aware_gate` 骨架，G1 R3 / G2 R1 APPROVED + §6 Verify green，2026-07-08）** / **Phase 4 ✅（D⁻ 失败库建库：cache-OFF `serve_policy --collect` 采集；⚠ 首轮误采 pruned/eval 集(污染)→ **held-out 池重采修正**：spatial 18 失败/l10 85 失败,合并 D⁺/D⁻ artifact 两 suite 出场门 PASS,§4 Code + 待 G2 复核,2026-07-10）** / Phase 5–7 待逐期立项
- **Date**: 2026-07-07（创建）
- **来源**: 合作者提案 `TRACER_RETRIEVAL_REFINED_PROPOSAL.pdf`（*Action-Compatible Failure-Aware Retrieval for VLA Inference Caching*，2026-06-10，含显式方程 Eq 1–28）。文中 "TRACER" = 本 fork 的推理 cache 系统；full-hit / warm-start / miss = 我们的 `HitType`。
- **Level**: 本文件为**研究产物（L0 纯文档）**。它只负责给整条线**排期与定依赖**，不含代码、不走 G1。**每一期的实现仍是独立的 L2/L3，必须各自走 Understand → Plan → G1 → Code → G2 → Verify。**
- **Owner 指令对齐**: "training-free 先行"。本路线图把**全部训练/标定放到 Phase 5–6（晚期）**，Phase 1–4 全程零训练 → 与该指令一致。
- **与既有工作线关系**: 与 gate 探索线（N1/N2/N4，见 `gate_exploration_roadmap.log.md`）**正交**——gate 管"**要不要搜**"（省延迟），本线管"**搜得准不准 / 判得安不安全**"（检索质量 + 复用安全）。两者可叠加，互不前置。

---

## 0. TL;DR — 这条线做什么、怎么分期

提案把 3 个机制装进现有可插拔框架：**M1 结果兼容投影**（学一个投影头改相似度）、**M2 失败感知双检索**（成功库 D⁺ / 失败库 D⁻ 的 margin 挡不安全复用，全文最硬的 Claim 1）、**M3 动态链深**（逐步选 trajectory 深度）。

**核心工程原理 = 机制（代码）与参数（训练值）分离**：所有"要训练"的东西本质只是拟合一组参数（M1 投影权重 / M2 阈值 β,τ,λ / M3 策略 ψ）。把每个可训练旋钮做成"**可加载 + 有惰性默认**"（照 `ScoreNormalizer.from_params_dict/fit_from_scores` 范式），就能**先把代码结构全部立起来并测试，训练那天只换参数文件、零代码改动**。且惰性默认让整条链**退化成现系统行为** → 用现有 golden 测试即可证非回归。

**分期一句话**：Phase 1–3 = 纯代码（惰性默认，退化=现系统，可证非回归）→ Phase 4 = 建失败库数据（非训练）→ Phase 5–6 = 标定/训练 → Phase 7 = 集成评测 + ablation。**"先完成代码、之后再训练"** 就是这个顺序。

---

## 1. 目标与非目标

**目标**（提案 §1 Success condition）：在 (success, inference-ratio) Pareto 前沿上，超过 raw-feature / success-only / fixed-depth 的现状检索，同时**降低不安全 full-hit**，且**不把 cache 变成独立检索策略**。

**非目标**（提案 §1 Non-goals，继承）：不替换 host VLA；不训练新 policy；不把向量索引当主贡献；host VLA + action expert 全程冻结；保持三态 verdict 接口不变。

**提案 3 机制 → 我们部件的映射**（详细方程对照见附录 B）：

| 机制 | 落成 | 现有对应 |
|---|---|---|
| M1 结果兼容投影（Eq 8–15） | 新 `QueryKeyBuilder`（投影头）+ 库侧离线过同一头 | 现有 pool/CLIP KeyBuilder 的替代/增强 |
| M2 失败感知双检索（Eq 16–21） | 新 `SearchStrategy`（双检索算 margin）+ 失败感知 `Composer` + D⁻ 库 + 2 处 additive 缝 | 复用 4 层 judge 的 kinematic 因子作 `u_t` |
| M3 动态链深（Eq 22–23） | 新 `SearchStrategy`（逐步选深度）——与 M2 天然并进同一策略 | 现有固定 `trajectory_depth` 的推广 |

---

## 2. 架构事实基线（据实 — 已亲验，附锚点）

本路线图的分期建立在以下**已核实**的代码事实上（锚点见附录 A）：

1. **装配层是单 backend / 单 storage**：`load_cache_config → _build_backend(config.backend) → CacheStorage(backend)`（config.py:1938-1939），`BackendConfig`（config.py:373）单例，无 dual/failure pool 概念。→ **M2 分池需显式设计**（Phase 3 决策）。
2. **可插拔边界清楚**：新组件只要**符合现有 Protocol**（`QueryKeyBuilder` key_builder.py:31 / `SearchStrategy` search_strategy.py:58 / `SimilarityJudge` judge.py:83）即"不动框架"；类型经 config.py frozenset + `_build_*` 分支注册（`_valid_key_builder_types`:1213 / `_valid_strategy_types`:1229 / `_JUDGE_TYPES`:477），加分支是**既定插件口**，非框架手术。
3. **数据契约的容量**：`QueryFilter` 仅 `task_key`/`step_range`（storage_types.py:177-178）；`SearchResultLite` 仅 `id/score/checkpoint_id`（:281-283）。→ **分池过滤 + 把 margin 递给判决 需 additive 字段**（Phase 3）。
4. **judge 注入缝已有先例**：orchestrator 本地构造 `view/history` 并以关键字注入 `judge(..., view=view, history=history)`（orchestrator.py:547-556）；"从策略读回方法"先例 `getattr(strategy, "get_search_session_id", ...)`（:303）。→ **`retrieval_signals` 走同一条 additive 缝**（Phase 3），judge 签名 `**kwargs` 兼容（judge.py:104），旧 judge 逐字节不变。
5. **backend 已支持变长轨迹深度**：轨迹融合用 `L = min(len(history), len(weights))`。→ **M3 逐步变深度零 backend 改动**。
6. **"拟合逻辑内聚 + 离线执行"已有范式**：`ScoreNormalizer.fit_from_scores`（score_normalizers.py，拟合是组件 classmethod）离线拟合、在线只 `from_params_dict` 加载；verdict-factor 标定"预加载 + 启动 fail-fast、**无 cold-start**"（`composite_judge.bind_keys`）。→ **训练逻辑可内聚进模块类，但执行必须离线**（见 §3）。
7. **兼容标签所需数据现成**：`CachePayload` 有 `action_chunk` + `intermediates`（warm-start 快照）+ `factors`（storage_types.py:101-105）→ M1 的 `c^A`（next-H action）/`c^X`（denoise snapshot）离线可算。
8. **失败样本标签在、embedding 不在**（**Phase 4 census 修正**）：`exp/gate_research` 逐步 `success` 标签齐（182,899 步），但 `gate_rows.jsonl` 仅 `robot_state`、**无 vision/prompt 向量**（采集用 `--collect-embeddings none`）；全 5-key embedding 只存在于近乎全成功的库源 `db/libero_cache`（spatial 49成功/1失败、l10 50成功/0失败）。二者不相交 → **多模态 D⁻ 本地无料，须新 `serve_policy --collect`（cache-OFF）采集失败 rollout**（详见 [`tracer_phase4_failure_library.log.md`](tracer_phase4_failure_library.log.md) §0.1）。

---

## 3. 核心设计决策（贯穿全期）

- **D1 机制 vs 参数分离**：每个可训练旋钮 = "可加载文件 + 惰性默认"。代码不依赖训练完成即可运行。
- **D2 惰性默认 = 退化到现系统**：无投影权重→identity；D⁻ 空→`margin=s⁺`；失败 Composer 默认→对 `s⁺` 卡阈值≈`ThresholdJudge`；深度策略常数→现有固定深度。**每期落地都必须证"惰性默认下逐值/行为不回归"**。
- **D3 训练逻辑内聚、执行离线**：`fit()`/`calibrate()` 挂在组件类上（照 `fit_from_scores`），但由离线 driver 调用、参数冻进 artifact/YAML；**严禁在线自训练**（撞离线建库/在线加载分离、C2 <1ms 预算、可复现性、以及 verdict 已铲除的 cold-start 反模式）。
- **D4 additive-only 框架触点**：M2 需要的 `QueryFilter.outcome` 与 `retrieval_signals` 注入缝**只允许 additive**（默认 None → 全部旧路径逐字节不变），不得改 orchestrator/interceptor/backend-ABC/judge 内核语义。
- **D5 分池方案倾向单 artifact + outcome tag**：D⁺/D⁻ 合存一个 artifact、entry 带 `y_d`，用新增 `QueryFilter.outcome` 过滤（避免第二 backend 的装配层改动）。此为 Phase 3 Plan 的头号待定项。

---

## 4. 分期路线

> 每期表头：**目标 / 交付物 / 框架触点 / 退化契约（非回归）/ 前置 / 出场 gate / Level / 训练依赖**。
> 训练依赖：⚪ 无 / 🟡 轻标定 / 🔴 重训练。

### Phase 0 — 架构判定与可行性亲验 ✅（本路线图依据）

- 已完成：3 机制 → 部件映射；单 backend / Protocol / config 工厂 / QueryFilter / orchestrator 注入缝 / ScoreNormalizer 范式 全部亲验（§2）；确认"完整实现 = 4 新模块 + 3 additive 缝 + 离线训练/建库"，且完整失败门天生涉及 verdict（`u_t` = kinematic 因子）。
- 产出：本文件。

### Phase 1 — M3 动态链深（纯新 SearchStrategy）✅

- **✅ 完成（2026-07-07，commit `cea98b2`）**：`dynamic_depth_knn` + `DepthPolicy`（constant / heuristic）落地；G1 R2 / G2 R2 APPROVED；§6 Verify `tests/cache/` **981 pass / 6 skip**；constant@max 逐值等价现有固定深度策略（partial-history golden 守卫）。Plan 归档 [`archive/tracer_phase1_dynamic_depth.log.md`](archive/tracer_phase1_dynamic_depth.log.md)。
- **目标**：可逐步（per-step）选 trajectory 深度 `T_t ∈ {0,3,5,8}`（Eq 22–23），先用**启发式/常数**策略。
- **交付物**：新 `SearchStrategy` 子类（继承 `TrajectoryMixin`，内部算深度、建变长 `trajectory_history/weights`）+ config 工厂分支 + 单测。
- **框架触点**：**零**（套现有 `SearchStrategy` Protocol；backend 已支持变长深度，事实 §2.5）。零 verdict。
- **退化契约**：深度策略返回常数 → QuerySpec 与现有固定深度策略等价 → 结果逐值一致（对照 `WeightedScoreSumKnnStrategy` 同深度 golden）。
- **前置**：无。**出场 gate**：常数退化 golden 通过 + 启发式档位可跑。**Level**：L2。
- **意义**：最干净、无缝、无训练——**首期用来趟通 Plan→G1→Code→G2→Verify 全流程**。

### Phase 2 — M1 投影 KeyBuilder 骨架（identity 默认）✅

- **✅ 完成（2026-07-08）**：`projection` KeyBuilder（包无状态 pool inner）+ `ProjectionParams`（torch save/load）+ 类上 `fit()`（InfoNCE 机制，合成数据单测，本期不跑真实库）落地；G1 R2 / G2 R2 APPROVED；§6 Verify `tests/cache/` + 库侧脚本测 **1018 pass / 6 skip**；identity 逐值等价内层 pool（两侧同头 golden）+ 加权维度自洽 + projection 继承被包 cp1_* 的 key-enablement/preload 契约。Plan [`tracer_phase2_projection_key_builder.log.md`](tracer_phase2_projection_key_builder.log.md)（G2 Review Log 永久保留）。
- **目标**：KeyBuilder 支持对每模态过投影头 `z=h_θ`；**无权重时 identity**（等于现有 pool 输出）。`fit()` 逻辑内聚在类上但**本期不调用**。
- **交付物**：新 `QueryKeyBuilder` 子类（load 权重→投影 / 无权重→identity）+ 类上 `fit(library)->weights`（实现但不跑）+ config 工厂分支 + artifact build 侧投影 hook（identity 默认=no-op）+ 单测。
- **框架触点**：**零**（套 `QueryKeyBuilder` Protocol；投影两侧都做 → backend 仍普通 cosine，事实 §2）。零 verdict。
- **退化契约**：无权重 → `build()` 输出逐值 == 现有 pool KeyBuilder（golden 对照 keys）。
- **前置**：无（与 Phase 1 独立并行）。**出场 gate**：identity golden 通过 + 带权重时投影 shape/维度自洽。**Level**：L2。

### Phase 3 — M2 失败感知骨架（惰性 D⁻ + 手设参数）✅（仅手设默认，不训练）

> 本期最大，含全部 3 处 additive 缝。Plan 里可再拆 3a（缝）/3b（策略）/3c（composer）三个可审子块。

- **✅ 完成（2026-07-08）**：`dual_retrieval_knn` 双检索策略（D⁺ over-fetch≥2 / D⁻ top-1 / `margin=s⁺−λ·s⁻` / `Δ⁺`；`enable_dual=false` 逐值等价固定深度 base-fusion）+ **`failure_aware_gate` 独立 `SimilarityJudge`**（σ 门三态；默认 β₂=β₃=0 逐值 == `ThresholdJudge`）+ 3a additive 缝（`QueryFilter.outcome`/`CacheEntry.outcome`/`RetrievalSignals` 侧信道 orchestrator→judge，legacy/`CompositeJudge`/`DumpingJudge` 全覆盖）。G1 R3 / G2 R1 APPROVED；§6 Verify `tests/cache/` **1056 pass / 6 skip**；深度机构上移 `TrajectoryMixin` 共享（Phase 1 golden 复验）；**实现细化 vs 原提案**：3c 因 `CompositeJudge` 强制 ≥1 factor 落为独立 judge（非 Layer-4 Composer），`u_t` kinematic 项分期 **Phase 5**（validator 守 `b2==0`）。Plan [`tracer_phase3_failure_aware_skeleton.log.md`](tracer_phase3_failure_aware_skeleton.log.md)（G2 Review Log 永久保留）；示例 YAML `dual_retrieval_degenerate` / `failure_aware_gate_skeleton` build-verified。

- **目标**：双检索算 `s⁺/s⁻/margin/Δ⁺`（Eq 16–20）+ 失败感知 σ 判决门（Eq 21）跑通；**D⁻ 空 + 参数手设默认时退化 = 现 success-only + 阈值判决**。
- **交付物**：
  - **3a 缝**（additive-only，事实 §2.3–2.4）：`QueryFilter.outcome` 字段 + backend `supported_filters`/`_filter_entries`（各数行）；`retrieval_signals` 注入缝（orchestrator 读 `strategy.last_retrieval_signals()`（先例 :303）→ 关键字传 `judge(..., retrieval_signals=...)`（先例 :556））。
  - **3b 策略**：新双检索 `SearchStrategy`（对 D⁺/D⁻ 各搜一次算 margin；D⁺ winner 作 `results[0]`；D⁻ 空→`s⁻=0`→`margin=s⁺`）。含 M3（Phase 1）动态深度（Eq 22–23 正好吃 `s⁻/Δ⁺`，天然并入同一策略）。
  - **3c composer**：失败感知 `Composer`（σ 门；`u_t` 复用现成 4 层 judge kinematic 因子；`β/τ/λ` 走 YAML 手设默认）。**〔实现细化 · 见 [Phase 3 plan](tracer_phase3_failure_aware_skeleton.log.md) G1 R2〕**：因现架构 `CompositeJudge` 硬性要求 ≥1 factor + normalization + calibration（`composite_judge.py:79`），3c 落为**独立 `SimilarityJudge` 类型 `failure_aware_gate`**（直读 `results[0].score`+`retrieval_signals`，非 Layer-4 Composer）；`u_t`（calibrated 因子）分期 **Phase 5**，本期交付 M2 核心 = margin(+Δ⁺) 失败感知门（Claim 1）。
- **框架触点**：**2 处 additive 缝**（`QueryFilter.outcome`、`retrieval_signals`），默认 None → 全旧路径逐字节不变；新增 1 策略（`DualRetrievalKnnStrategy`）+ 1 judge（`failure_aware_gate`，见 Phase 3 plan R2 实现细化指针）+ config 分支。**不改** orchestrator/interceptor/backend-ABC/judge 内核。
- **退化契约**：`outcome=None` + 空 D⁻ + 默认参数 → margin=s⁺、σ 门≈`ThresholdJudge` → 整栈行为 == 现系统（用现有 orchestrator/judge golden 证非回归）。
- **前置**：Phase 1（动态深度并入策略）。**出场 gate**：惰性退化非回归 + 带小型 fixture D⁻ 时 margin/门 逻辑单测通过。**Level**：L3。
- **注意**：本期**不需要真 D⁻、不需要训练**——用 fixture / 空池即可开发测试。真数据在 Phase 4，标定在 Phase 5。

### Phase 4 — 失败库 D⁻ 构建（数据管线，非训练）✅

- **✅ 完成（2026-07-10，§4 Code + 待 G2 复核）**：cache-OFF `serve_policy --collect --replicas 1 --non-concurrent`（pi05 PyTorch 自然失败，seed 7）。⚠ **污染修正**：首轮 D⁻ 默认 `init_states_dir=""` → `get_task_init_states` = `pruned_init`（= Phase 7 评估集），与 D⁺ 的 held-out 划分相反且泄漏 → 全部污染数据/产物删除，D⁻ 在 **held-out 池 `db_init/libero/<suite>`（全集去掉 pruned）重采**（与 D⁺ 同划分、与 eval 不相交，D8 从根本满足）。修正后：spatial 500 rollout→**18 失败/792 D⁻ step**、l10 500 rollout→**85 失败/8840 D⁻ step**；`build_dual_artifact` 合并 D⁺(tag+1)∪D⁻(tag−1)→ **spatial 1810 / l10 11480 entries**（零 None）；`validate_dual_artifact` 两 suite **出场门 3-gate PASS**（coverage + 非平凡 margin 60/60 + 判别性）。builder 加 additive `--outcome-filter`（默认 success 零回归，14 测试全绿）+ in-experiment fix `main.py` `torch.load` 老 torch 兼容。held-out 失败率(3.6%/17%)高于 pruned(2.2%/5%)反利好 D⁻（D6）；provenance 存 `analysis/phase4_dminus_provenance.md`（索引 held-out 池，D8）。Plan [`tracer_phase4_failure_library.log.md`](tracer_phase4_failure_library.log.md)。
- **目标**：从失败 rollout 建带 `y_d∈{+1,-1}` 的 D⁻ artifact（§2.8 已 census 修正：gate_rows 仅 robot_state 无 vision/prompt → 多模态 D⁻ 须新采集）；M2 首次**非退化**运行，兑现 Claim 1（失败感知降低不安全复用）。
- **交付物**：`exp/` 建库脚本（复用 `build_in_memory_cache_artifact` 路径 + 成败标注）+ D⁺/D⁻ 合并 artifact（按 D5 单 artifact + `y_d` tag）+ 采集完整性校验。
- **框架触点**：无（exp/ 数据层）。零训练（参数仍手设）。
- **前置**：Phase 3。**出场 gate**：D⁻ artifact 可 load + 双检索在真库上产出非平凡 margin 分布。**Level**：L1/L2（数据脚本）。

### Phase 5 — 阈值 / 权重标定（轻标定）🟡

- **目标**：在 held-out 集上标定 `τ_hit/τ_warm/λ`（及可选 M3 的 ψ、M2 融合 `η`），最小化 `L_cal = BadHitRate + c_miss·MissRate + c_warm·WarmCost` s.t. `SR ≥ SR_base − ε`（Eq 24–25）。
- **交付物**：composer/strategy 类上的 `calibrate(held_out)->params`（内聚，照 `fit_from_scores` 范式）+ `exp/` 离线 driver + 标定产物入 YAML。
- **框架触点**：无（离线，事实 §2.6 / D3）。**轻**（阈值/权重搜索，非梯度训网络）。
- **前置**：Phase 4。**出场 gate**：标定参数使 M2 在验证集上 BadHitRate↓ @ 同 SR/inf。**Level**：L2。

### Phase 6 — M1 投影头训练（重训练，条件触发）🔴

- **目标**：按 InfoNCE（Eq 15）+ 兼容标签（Eq 10–12，`c^A` next-H action / `c^X` denoise snapshot，事实 §2.7）训投影头；host VLA + action expert 冻结。
- **交付物**：Phase 2 类上 `fit(library)->weights` 的离线 driver + 训练数据构造（从 payload 算兼容标签）+ 冻结权重入投影 artifact。
- **框架触点**：无（离线，D3）。**重**训练。
- **触发条件（重要）**：提案 Claim 2 是 **necessity check**（投影可能根本不需要）；**仅当 Phase 7 ablation 显示"候选质量是瓶颈、raw 特征不够"时才启动本期**。否则永久停在 identity（Phase 2 骨架已足）。
- **前置**：Phase 2 + Phase 7 初轮 ablation。**Level**：L2。

### Phase 7 — 集成评测 + ablation（Claim 1/2/3 验证）🟡

- **目标**：全线 (SR, inf_ratio) Pareto 评测 + 逐机制 ablation（提案 §11 Claim 1/2/3、§14 step 7）：raw vs 投影、success-only vs dual、fixed vs dynamic depth。
- **交付物**：`exp/` 评测 driver + 分析报告（`analysis/*.md`）+ 对 Phase 6 是否启动的裁决。
- **框架触点**：无（exp/ 评测）。**前置**：Phase 3（骨架）→ 首轮可跑；Phase 4/5 后为主轮。**Level**：L2。

---

## 5. 依赖图与关键路径

```
Phase 1 (M3 深度) ─┐
                   ├─► Phase 3 (M2 骨架) ─► Phase 4 (D⁻ 数据) ─► Phase 5 (标定) ─► Phase 7 (评测/ablation) ─┐
Phase 2 (M1 骨架) ─┘（并入策略）                                                                            │
        │                                                                                                   │
        └───────────────────────────────────────────────────────────► Phase 6 (M1 训练，条件触发) ◄────────┘
```

- **关键路径 = 1 → 3 → 4 → 5 → 7**（失败感知安全边界，全文最硬）。
- **Phase 2（M1 骨架）可与 1/3 并行**；**Phase 6（M1 训练）押到 ablation 之后**，可能永不触发。
- **"先代码后训练"** 体现为：Phase 1–3 纯代码 → Phase 4 数据 → Phase 5–6 训练/标定。

---

## 6. 风险登记 / 开放问题

1. **分池落点（Phase 3 头号决策）**：单 artifact + `QueryFilter.outcome`（D5 倾向）vs 第二 backend。前者装配零改、缝更小；Plan 定夺。
2. **`retrieval_signals` 缝的形状**：per-query 标量（margin/s⁻/Δ⁺）如何随 `results` 一起过 orchestrator——倾向独立 side-channel dict（不塞进 per-result 的 `SearchResultLite`）。Phase 3 Plan 定契约。
3. **D⁻ 语义**：失败库放哪些步？整条失败 episode，还是仅"近失败"步（提案 §9 near-miss / corrupted）？影响 margin 判别力与 Claim 1 强度。Phase 4 定义。
4. **投影是否值得（Phase 6 触发条件）**：Claim 2 是 necessity check；若 raw 特征够，M1 训练永久搁置。由 Phase 7 ablation 裁决。
5. **warm-start 兼容**：M2 门的 warm 分支需 winner 有 `intermediates`（复用现 `fetch_payload` 完整性降级逻辑）；CP3 无 warm（validator 已约束）。
6. **与 gate 线的潜在协同**（非前置）：M2 的失败 margin 可作 gate 特征反哺 N1/N4 的 `record_verdict`——列为后续可选，不进本路线图关键路径。

---

## 附录 A：已验证代码锚点

| 事实 | 锚点 |
|---|---|
| 单 backend / storage 装配 | `config.py:1938-1939`；`BackendConfig` `config.py:373` |
| KeyBuilder / Search / Judge Protocol | `key_builder.py:31` / `search_strategy.py:58` / `judge.py:83` |
| 类型注册 frozenset + 工厂 | `config.py` `_valid_key_builder_types:1213` / `_valid_strategy_types:1229` / `_JUDGE_TYPES:477`；`_build_key_builder:2197` / `_build_search_strategy:2707` / `_build_judge:2315` / `_build_backend:2133` |
| QueryFilter 现有字段 | `storage_types.py:177-178`（仅 task_key/step_range） |
| SearchResultLite 现有字段 | `storage_types.py:281-283`（id/score/checkpoint_id） |
| orchestrator 注入 view/history | `orchestrator.py:547-556`；读策略方法先例 `:303` |
| judge 签名 `**kwargs` 兼容 | `judge.py:104` |
| 拟合内聚 + 离线加载 范式 | `score_normalizers.py` `fit_from_scores`/`from_params_dict`；`composite_judge.bind_keys`（no cold-start） |
| 兼容标签数据源 | `CachePayload` `storage_types.py:101-105`（action_chunk/intermediates/factors） |
| 失败样本来源（Phase 4 修正） | `exp/gate_research` 逐步 success 标签齐，但**仅 robot_state、无 vision/prompt** → 多模态 D⁻ 须新 `serve_policy --collect`（cache-OFF）采集（Phase 4 plan §0.1） |

## 附录 B：提案方程 ↔ 我们部件 对照

| 提案 | 我们 |
|---|---|
| Eq 1 `C_t=Search(q_t,D⁺)`, `v_t=Verdict(C_t)` | `orchestrator.check()` → `SearchStrategy.search()` → `Judge` |
| Eq 6 `RRF=Σ wᵢ/(k+rankᵢ)` | `in_memory_backend._search_weighted_rrf`（逐字一致） |
| Eq 7 chain-aware `S_T=Σ αℓ·RRF(q_{t-ℓ},d^{-ℓ})` | 轨迹搜索（`trajectory_weights`=αℓ newest-first；`d^{-ℓ}`=walk_prev 链） |
| Eq 8–9 投影相似度 `sim=⟨z_t,z_d⟩/‖·‖` | M1：新 KeyBuilder 产 z，backend 普通 cosine |
| Eq 10–12 兼容标签 `c^A/c^X/c_ab` | 离线，从 `payload.action_chunk`/`.intermediates` 算 |
| Eq 16–18 双检索 margin `m=s⁺−λs⁻` | M2：新双检索 SearchStrategy 内部算 |
| Eq 19–21 `g_t=σ(β₀+β₁m+β₂u+β₃Δ⁺)` 三态门 | M2：失败感知 Composer；`u_t`=现 4 层 judge kinematic 因子 |
| Eq 22–23 动态深度 `T_t=argmax ψᵀr_t−μ·cost` | M3：策略内逐步选深度 |
| Eq 24–25 阈值标定 | Phase 5 离线 calibrate |
| Eq 26–28 IR/BHR/FFR 指标 | Phase 7 评测口径 |

---

> **下一步**：Phase 1（M3 动态链深）进入正式 **Understand → Plan → G1 → Code → G2 → Verify**。本路线图作为各期 Plan 的上位依据；每期落地后回填其状态与判决。
