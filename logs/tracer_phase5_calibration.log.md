# TRACER Phase 5 — 阈值/权重标定 + u_t 激活（轻标定）Plan

- **Status**: Implemented（G1 R3 + **G2 R4 APPROVED**；§6 Verify `tests/cache/ tests/exp/` **2035 pass/6 skip** green；机制代码 commit 完成。**运行时 Pass 1/2/3 rollout + 真实 calibrated YAML + Phase 5 report 仍属后续出场条件**，达出场门前不标 Validated，2026-07-11）
- **Date**: 2026-07-10
- **Level**: **L3**（owner 决定本期一并激活 β₂·u_t → 触及 src/ 三处 additive 缝：`FailureAwareGateJudge` u_t 通道 + `config.py` validator/工厂 + 架构文档更新。若仅标定 τ/λ 则为 L2；u_t 激活把它升为 L3）
- **上位依据**: [`tracer_retrieval_refinement_roadmap.log.md`](tracer_retrieval_refinement_roadmap.log.md) Phase 5（阈值/权重标定 🟡）。前置 Phase 4 ✅（held-out D⁻ 建库完成，commit `77ed0f5`）。
- **子实验目录**: `exp/zixuan_proposal/`（源提案 PDF + Phase 3/4 产物在此）。
- **提案锚点**: Eq 19–21（三态门 `g=σ(β₀+β₁m+β₂u_t+β₃Δ⁺)`）、Eq 24（`L_cal = BadHitRate + c_miss·MissRate + c_warm·WarmCost`）、Eq 25（约束 `SR ≥ SR_base − ε`）、Eq 26–28（IR / BHR / FFR 指标口径）。

---

## 1. 无上下文摘要（供 G1 reviewer，零对话史可读）

**背景**：本项目推理 cache 是可插拔的（`SearchStrategy` / `SimilarityJudge` Protocol + config 工厂）。TRACER M2 = 失败感知双检索：`DualRetrievalKnnStrategy` 对成功库 D⁺ / 失败库 D⁻ 各检索一次，算 `s⁺/s⁻/margin=s⁺−λ·s⁻/Δ⁺`（`RetrievalSignals` 侧信道）；`FailureAwareGateJudge` 用 σ 门 `g=σ(β₀+β₁·margin+β₃·Δ⁺)` 产 FULL_HIT/WARM_START/MISS 三态。Phase 3 落地了这套骨架但**参数全部 YAML 手设（illustrative，未标定）**、且 **β₂·u_t（kinematic 项）被 validator 强制为 0 分期到 Phase 5**。Phase 4 建好了真实的 held-out D⁻ 库（spatial 1810 / l10 11480 entries，出场门 3-gate PASS）。

**本期目标（两件事）**：
1. **激活 β₂·u_t**（owner 决定）：让 gate 消费一个 kinematic-quality 信号 u_t（提案 Eq 19 "computed from the positive candidate d⁺"），gate logit 变为 `z=β₀+β₁m+β₂u_t+β₃Δ⁺`。放宽 `b2==0` validator。
2. **离线标定 (λ, τ_hit, τ_warm, β) 于 held-out 集**（提案 Eq 24–25）：最小化 `L_cal = BadHitRate + c_miss·MissRate + c_warm·WarmCost`（真 `SR ≥ SR_base − ε` 由 Pass 3 强制，见 D-P4）；标定产物写**新** `dual_retrieval_calibrated{,_l10}.yaml`（spatial + l10；**不覆盖** active illustrative YAML）。

**数据划分（防泄漏，G1 R1 finding 1）**：held-out 池 = 50 init 态/任务 × 10 任务/suite。Phase 4 D⁻ 由 seed 7 在**全部 50/任务**上跑失败 rollout 建成（provenance 记录每条 D⁻ 的 `(task_id, init_state_idx)`）。为防 self-retrieval / 过拟合泄漏，本期用**三层分离**：
- **calibration split `I_cal` vs validation split `I_val` 物理不相交**：按 `init_idx` 确定性二分（`I_cal = {idx%2==0}`、`I_val = {idx%2==1}`，各 25/任务）。Pass 1 标定采集只在 `I_cal`，Pass 3 验证只在 `I_val` → 标定绝不在被验证的 init 上过拟合。
- **library（D⁻/D⁺）↔ calibration 泄漏用 LOEO 排除消解**：D⁻ 建自全部 50，与 `I_cal` 有交 → Pass 2 replay 每条 calibration query episode 时，**从检索池按 `trajectory_id`/provenance 排除该 episode 自身的 D⁻/D⁺ entries**（leave-one-episode-out，照 `calibrate_score_normalizers.py:102` LOEO 先例），并加 **fail-loud 断言**：任何 query 的 top-1 匹配 id ∉ 该 query 自身 trajectory。物理 rebuild D⁻ 于子 fold 被弃（Phase 4 rework + D⁻ 缩水；LOEO 是 sound 且有先例的等效解）。
- Phase 7 `pruned_init` eval 集与整个 held-out 天然不相交（Phase 4 已隔离）——那是泛化检验，不修 Phase 5 内部泄漏（本节才修）。

**方法学（owner Q1 选定 = warmup rollout + 离线解 + 验证 rollout）= 三阶段管线**，把「取真实成败标签 + 观测」与「算 gate 信号」分离，既忠实在线 chain-aware margin 又不被 cache 复用污染：
- **Pass 1（base 采集，cache-OFF，串行）**：在 `I_cal` 跑纯推理 rollout（复用 Phase 4 `serve_policy --collect` 机制，**强制 `--replicas 1 --non-concurrent`**，见 §2 `_validate_collect_isolation`），得每 episode 成败 + 每步 5-key embedding h5（`--collect` 落盘，非 `--save-episode-results`/`save_trajectory`）。需 ziyang10 server + timan107 client（**串行采集，非 eval 的 3replica**）。
- **Pass 2（离线 replay）**：把每条 episode 的 h5 **按轨迹顺序**喂生产 `DualRetrievalKnnStrategy`（不 reset history → TrajectoryMixin 累积 → 忠实重建深度5 chain-aware `s⁺/s⁻/margin/Δ⁺`）+ gate u_t 计算，LOEO 排除自身 entries，join 该 episode 成败 → threshold-**无关**的每步信号表 JSONL（§3b 有逐步时序 + orchestrator parity 表）。
- **离线 solve**：`MarginGateCalibrator.calibrate(rows)->params`（新 src 类，内聚照 `fit_from_scores` 范式），**唯一参数化 + 显式网格**（§3b 定死）搜 (λ, b0, β₂, β₃, τ_warm_g) 最小化 `L_cal`（`argmin`，MissRate 项即 SR 保护压力，离线不设 SR 约束——见 D-P4）。
- **Emit**：标定值 `yaml.safe_dump` 写**新** `dual_retrieval_calibrated{,_l10}.yaml`（**不覆盖** active illustrative YAML；后者留作标定前参照，前者为生产配置）。
- **Pass 3（验证 rollout，eval 拓扑）**：在 `I_val` 跑**三条同 init+seed 配对** rollout —— cache-OFF baseline（得 `SR_base(I_val)`）+ illustrative cache-ON + calibrated cache-ON；新 `analyze_phase5_rollout.py` 算 BHR/FFR/IR/SR（分子分母 + paired provenance），确认 **BadHitRate↓ @ `SR_calibrated(I_val)≥SR_base(I_val)−ε` 且 IR 不升**（路线图出场门）。cache-ON 部分为 eval（非 `--collect`）→ 可用 3replica 2batch/48worker 拓扑。

**隔离律遵守**：u_t 激活是 gate 内新增计算（gate 已从 orchestrator 无条件收 `view`/`history`，见 §2）；**不改** orchestrator/interceptor/backend/Protocol 语义、不改 `DualRetrievalKnnStrategy`、不改 `RetrievalSignals`。additive 默认（无 u_t 配置 + β₂=0 + `export_factor_outputs=false` → **verdict 与 wire 皆逐字节退化为 Phase 3 gate**）。标定与建库全离线（D3）。

---

## 2. 现状事实（已亲验，附锚点）

| 事实 | 锚点 |
|---|---|
| `FailureAwareGateJudge.__init__(*, gate_betas, threshold=0.5, warm_tiers=None)`；只读 `_b0/_b1/_b3`（**无 `_b2`**）；logit `z=b0+b1*m+b3*d`，`g=σ(z)`（数值稳定分支）；三态：`g≥threshold`→FULL_HIT / CP1 且 `g≥tier.threshold`→WARM_START(start_t) / else MISS；`composer_score=g` | `components/judge.py:252,276-289,291-329` |
| gate 只读 `retrieval_signals.margin` + `.delta_pos`（**不读 s_pos/s_neg**）；`retrieval_signals is None`→`raise ValueError`（fail-loud，合法配置永非 None） | `judge.py:310-312,300-308` |
| `SimilarityJudge.__call__(results, ckpt, cached_data, *, view=None, history=None, retrieval_signals=None)`——**gate 已收 `view`/`history`**（现经 `**kwargs` 吞掉，未用） | `judge.py:105-114`；`FailureAwareGateJudge.__call__` 签名 `judge.py:291-298`（收 `**kwargs`） |
| `JudgeResult(hit_type, winner_id, start_t, composer_score, factor_outputs)`；`factor_outputs` = 可选诊断 dict，**NaN 预转 None**，经 CheckResult→Interceptor `__hit_meta__`→client per-step JSONL | `judge.py:59-80` |
| orchestrator 在 **searched 路径无条件**建 `view=StoragePayloadView(storage)` + `history=HistoryView(actions,states)` 并传 `judge(..., view=, history=, retrieval_signals=)`（M2 用 `always_search` gate 永走此路径→gate 恒拿非 None view/history） | `orchestrator.py:541-564` |
| `DualRetrievalKnnStrategy(storage,*,base_fusion,depth_policy,allowed_depths,lambda_=0.0,enable_dual=False,top_k=1,...)`；`margin=s_pos−self._lambda*s_neg`；`last_retrieval_signals()->RetrievalSignals|None` | `components/search_strategy.py:667,693-731,761,771-773` |
| `RetrievalSignals(frozen)`: `s_pos/s_neg/margin/delta_pos/lambda_`（**无 u_t 槽**） | `storage_types.py:308-332` |
| **b2==0 validator**（本期放宽）：`failure_aware_gate` 要求 `gate_betas` 含 b0/b1；`float(gb.get("b2",0.0))!=0.0`→error "u_t kinematic term is deferred to Phase 5" | `config.py:1524-1538` |
| gate↔strategy 配对 validator：`failure_aware_gate` 须配 `dual_retrieval_knn` | `config.py:1557-1562` |
| `margin_lambda>=0` validator | `config.py:1662-1667` |
| `_build_judge(cfg, *, library_stats=..., yaml_id=...)` **已接 `library_stats` 但未转发给 gate**；gate 分支 `_build_inner_judge` :2561-2570 `FailureAwareGateJudge(gate_betas=dict(cfg.gate_betas or {}), threshold=cfg.threshold, warm_tiers=cfg.warm_tiers)` | `config.py:2513,2561-2570` |
| `JudgeConfig(type,threshold=0.98,warm_tiers=None,...,gate_betas:Optional[dict]=None)`（嵌套 dict 由 `_dict_to_dataclass` 泛型反序列化，无 bespoke 解析） | `config.py:284-322`；`load_cache_config` :704-733 |
| `ZScoreNormalization(library_stats, *, eps=0.01)`（library_stats None→raise）；`normalize_action(raw)`/`normalize_state(raw)`=`raw/sigma.clamp_min(eps)` 后 active-mask 选维 | `components/factors/normalization/zscore.py:34,43-65` |
| online kinematic factor：`_OnlineFactorBase.__init__(*, windows)`（windows=`[{past,future}]`）；`extract(ctx:FactorContext)->{key:float}`=splice`[hist[-P:],winner,walk_next(F)]`→`ctx.normalization.normalize_*`→`compute_descriptors`；边界/fork/缺链→NaN；`requires_chain_walk=True`（需 backend 有 fetch_entry，M2 in_memory 满足）；state 通道锚 `view.get_entry(winner_id).query_keys["robot_state"]` | `components/factors/online.py:71-117,210-267,275-312` |
| descriptor orientations：`jerk="risky"`、`direction="safe"`、`dispersion/path_length="non_monotonic"`；`compute_descriptors(descriptors,pts,v,j)->{d:float}` | `components/factors/_descriptor_kernel.py:73-83,90-125` |
| `FactorContext(results, view:PayloadView, history:HistoryView, normalization)` | `components/factors/base.py:200-221` |
| `LibraryStats.compute_from_entries(entries)`（D⁺-only，Phase 4 build_dual_artifact 保证）→ `action_sigma/state_sigma/*_active_mask`；in_memory backend 启动惰性算（若 artifact 缺） | `components/factors/base.py:56-120`；`in_memory_backend.py:283` |
| **两 active YAML** 现用未标定 illustrative 值：`judge: {type:failure_aware_gate, threshold:0.5, gate_betas:{b0:-0.9,b1:1.0,b3:0.0}}` + `search_strategy:{type:dual_retrieval_knn, base_fusion:weighted_rrf, trajectory_depth:5, allowed_depths:[5], depth_policy:{constant,5}, margin_lambda:0.5, enable_dual:true}`；preload_path→`cp1_mean_pool_dual.pkl` | `exp/zixuan_proposal/config/dual_retrieval_active.yaml` / `..._l10.yaml` |
| 合并 artifact 已存：spatial `cp1_mean_pool_dual.pkl` 86M（1810 entries）/ l10 510M（11480 entries） | `exp/common/data/cache_artifacts/{libero_spatial,libero_10}/` |
| held-out init 池 = `exp/common/data/db_init/libero/{suite}/<task>.init`（10 任务×50 态；Phase 4 采 D⁻ 的同池，与 pruned_init eval 集不相交）；anti-leak 逻辑 `init_holdout.py:42-53` | `phase4_dminus_provenance.md:3`；`exp/weighted_sum/init_holdout.py` |
| **指标现状**：SR **已存**（`_aggregate_sr_from_episode_json`，读 main.py `--save-episode-results` JSON）；MissRate/WarmCost 有原始 verdict 计数（`summarize_gate_log`→`{n_eval_verdicts,n_full_hit,n_warm_start,n_miss}`，读 `<gate_dir>/<yaml_id>.jsonl`）；**BadHitRate 不存在须新建**；`c_miss/c_warm/SR_base/ε` 全无 | `exp/verdict_factor_judge/common/run_phase.py:299-317`；`src/openpi/serving/per_step_recorder.py:287-330` |
| per-step dump：`PerStepWriter` row schema **caller 定义**，gate 模式 `flush_episode(success)` 给每行盖 `success`；`_commit` `allow_nan=False`（NaN 必须预转 None） | `per_step_recorder.py:82-120,89-107` |
| **`--collect` 隔离硬约束**：`_validate_collect_isolation` 对 `replicas>1` fail-fast（:642-646）、`concurrent 且非 non_concurrent` fail-fast（:648-652）→ 采集必 `--replicas 1 --non-concurrent` 串行 | `scripts/serve_policy.py:617-652` |
| **`export_factor_outputs` 是现存 JudgeConfig 字段**（默认 False，现只喂 CompositeJudge :2624）；interceptor `_build_hit_meta` 对**任何**非-None factor_outputs 上 `__hit_meta__` wire（`if factor_outputs is not None: meta["factor_outputs"]=...`）→ gate 无条件填会破 Phase 3 wire | `config.py:316,2624`；`interceptor.py:518-520` |
| **h5→query_keys 须走建库链**：`_process_episode` 每步 `_build_fake_stage1(group)`→`builder.collect(CP1, stage1)`→`builder.build(CP1)` 才得与 artifact 同空间的 query_keys（h5 存 token-level `vision_*`/`prompt_emb`/`robot_state`/`clean_action`，**非** KeyBuilder 产物） | `exp/common/build_in_memory_cache_artifact.py:289,661-663` |
| **在线历史时序**：`robot_state` 在 `check()` **开头**（search/judge **之前**、anchor=最低 enabled CP）append 到 `_state_history`；action 经 `broadcast_action` 在 check **返回后**取 `action_chunk[0]` append 到 `_action_history`（即当步 action 进的是**下一步**历史） | `orchestrator.py:475-479,356-383` |
| **search-session 生命周期**：`on_episode_start` 内 `storage.open_search_session(sid)`（:309）、`on_episode_end` 内 `storage.close_search_session(sid)`（:321），sid = strategy 生成（`get_search_session_id`） | `orchestrator.py:283-321` |
| 离线标定范式：`ScoreNormalizer.fit_from_scores`（classmethod 离线拟合返回实例）/`params_to_dict`（序列化 numeric）/`from_params_dict`（在线加载）；driver 先例 `calibrate_score_normalizers.py`（pkl→LOEO 打分→fit→JSON）；`verdict_factor_judge/phase3/threshold_solver.py`（复用生产类离线+分位切阈）；`emit_threshold_yamls.py`（唯一写 YAML：`yaml.safe_dump({type,threshold,warm_tiers})`） | `components/score_normalizers.py:103-153`；`exp/common/calibrate_score_normalizers.py`；`exp/weighted_sum/emit_threshold_yamls.py:48-112` |

---

## 3. 设计

> 结构 = 两块：**3a src/ u_t 激活缝**（框架触点，additive 退化=Phase 3）+ **3b exp/ 标定管线**（离线，三趟 + solve + emit）。贯穿硬约束：**additive-only + 无 u_t 配置 & β₂=0 → 逐值退化为 Phase 3 gate**。

### 3a — src/ u_t 激活（gate 内计算，additive）

**落点判定**：u_t 在 **gate 内**计算，不给 `RetrievalSignals` 加字段、不让 `DualRetrievalKnnStrategy` 承担 kinematic 计算。理由：(1) gate 已从 orchestrator 无条件收 `view`/`history`（§2 锚点）——u_t 需要的 winner 前向链 `walk_next` + history splice 天然可达；(2) 最大复用现成经测试的 online Factor + Layer-1 归一化 + descriptor kernel；(3) strategy 侧 `RetrievalSignals` 逐字节不变，缝最小。

**A1. `FailureAwareGateJudge` 增 u_t 计算**（`components/judge.py`）：
- `__init__` 增两个可选形参：`u_t_factor: Optional[dict] = None`（`{descriptor, channel, past, future}`，None→本期退化=Phase 3）、`library_stats: Optional[LibraryStats] = None`。存 `_b2 = float(gate_betas.get("b2", 0.0))`。
- 当 `u_t_factor is not None`：构造期 lazy-import 建 `self._norm = ZScoreNormalization(library_stats)` + `self._factor =` 对应 online Factor 实例（如 `DirectionOnlineState(windows=[{"past":P,"future":F}])`，按 `descriptor`×`channel` 选具体类）。`library_stats is None` 时 fail-loud（u_t 需归一化基）。
- `__call__` 增计算：`retrieval_signals` 分支后，若 `_u_t_active`：建 `FactorContext(results=results, view=view, history=history, normalization=self._norm)`，`u_raw = self._factor.extract(ctx)` 取**唯一** key 的值 → `u_t`。**NaN 处理**：`u_t` NaN（边界/fork/history<P/chain end）→ 该步**丢 β₂·u_t 项**（`z += 0`），门退化为 margin-only（优雅，且非回归——早期步/浅链本就无 kinematic 信号）。否则 `z = b0 + b1*m + b2*u_t + b3*d`。
- **factor_outputs 落盘（opt-in，G1 R1 finding 4）**：**复用现存 `JudgeConfig.export_factor_outputs`（config.py:316，默认 False）**，仅当为 True 时 `__call__` 才填 `factor_outputs = {"schema":"failure_gate_v1", "s_pos":..., "s_neg":..., "margin":m, "delta_pos":d, "u_t":(None if NaN else u_t), "g":g}`（NaN→None，合 `allow_nan=False` 契约）；**默认 False → factor_outputs 恒 None → `interceptor._build_hit_meta`（:518-520 `if factor_outputs is not None`）不上 wire → `__hit_meta__` 逐字节 == Phase 3**。此 dump **仅供 Pass 3 观测/诊断**，**不在标定关键路径**（Pass 2 离线 replay 直接取 `last_retrieval_signals()` + gate u_t，不经 wire）。带 schema version 便于演进。

**A2. `config.py` 工厂 + validator 放宽**：
- `_build_inner_judge` gate 分支（:2561-2570）：把已在 `_build_judge` 签名里的 `library_stats` **转发**给 `FailureAwareGateJudge(..., u_t_factor=cfg.u_t_factor, library_stats=library_stats, export_factor_outputs=cfg.export_factor_outputs)`（`export_factor_outputs` 已是 JudgeConfig 现存字段，现只喂 CompositeJudge :2624，本期 gate 分支也接）。
- `JudgeConfig` 增 `u_t_factor: Optional[dict[str, Any]] = None`（默认 None → 现有 YAML 逐字节不变）；`export_factor_outputs` **复用现存字段**（不新增）。
- **b2==0 validator 放宽**（:1533-1538）：改为 —— `b2!=0` 时要求 `u_t_factor` 非 None（且其 `descriptor∈{jerk,direction,dispersion,path_length}`、`channel∈{action,state}`、`past/future>=0`）；`u_t_factor` 非 None 时要求 backend=in_memory 且 artifact 带（或可惰性算）`library_stats`；`b2==0 且 u_t_factor is None` → 完全等价 Phase 3（现有 degenerate/skeleton/active YAML 全绿）。

**退化契约（非回归）**：`u_t_factor=None`（或 YAML 不写）+ `b2=0` + `export_factor_outputs=false`（默认）→ gate 不建 norm/factor、logit 无 b2 项、factor_outputs 恒 None（不上 wire）→ **verdict 与 `__hit_meta__` wire 皆逐值/逐字节 == Phase 3 `FailureAwareGateJudge`**。现有 3 示例 YAML + 2 active YAML（都 b2 缺省=0、无 u_t_factor、export 默认 False）行为逐字节不变，由 §5 NR1 + wire-invariance golden 守卫。

### 3b — exp/ 标定管线（离线，三阶段）

新脚本挂 `exp/zixuan_proposal/`（不建 design doc，WA §4）。数据划分见 §1（`I_cal`/`I_val` 不相交 + LOEO）。

**Pass 1 — base 采集（cache-OFF，串行，G1 R1 finding 2）**：
- **强制串行**：`serve_policy --collect --replicas 1 --non-concurrent`（`_validate_collect_isolation` 硬约束，replicas>1 或 concurrent 即 fail-fast，见 §2）。`--collect` cache-OFF → 每步落 5-key embedding h5（`vision_0/1`/`prompt_emb`/`robot_state`/`clean_action`；与 Phase 4 D⁻ 采集**同一机制、同 schema**）。episode 成败经 `--save-episode-results` JSON join（**h5 供 signal replay、JSON 供成败标签，两者按 (task,init,episode) 对齐**）。
- 范围：仅 `I_cal`（25 init/任务 × 10 任务 = 每 suite 250 rollout），**保留全部 h5（成功+失败）**——标定两类都要（bad-hit 需失败、safe-reuse 需成功）。
- `I_cal` 的 cache-OFF SR 仅作**数据质量记录**（不充当 validation 基线；Eq 25 的 `SuccessRate_base` 由 Pass 3 在 `I_val` 另跑 cache-OFF 得，见 D-P4/finding 3）。
- 设备：ziyang10 server（**单副本非并发**）+ timan107 client（**单进程**，因 `--non-concurrent` 单连接）。

**Pass 2 — 离线 replay（算 threshold-无关信号）**：新 `exp/zixuan_proposal/build_calibration_table.py`。逐步时序**逐行对齐 `orchestrator.check()`**（§2 anchors，G1 R2 finding 1 已按亲验修正）：

| orchestrator.check()（在线，锚点见 §2） | build_calibration_table（离线 replay） |
|---|---|
| 每步 `query_keys` 由**生产 KeyBuilder 建库链**产（`_build_fake_stage1(group)`→`builder.collect(CP1,stage1)`→`builder.build(CP1)`，build_in_memory_cache_artifact.py:661-663）| 逐步跑**同一建库链**（h5 存 token-level 模态，**非** keys）→ 与 artifact entry 同空间的 `query_keys`。builder 由 config 建（同 D⁺/D⁻ `cp1_mean_pool`）|
| episode 起：`strategy.on_episode_start()` 生成 sid + `storage.open_search_session(sid)`（orch:283-309）| 每 episode 起：`strat.on_episode_start()` → `sid=strat.get_search_session_id()` → `storage.open_search_session(sid)`（**显式注册**）|
| **check 开头**（search/judge **之前**）：`_state_history.append(query_keys["robot_state"])`（anchor=CP1，orch:475-479）| 每步**开头**先 append 该步 `robot_state` 到 `state_hist`（**先于** search/u_t，纠正 R1 表的"每步末"错位）|
| `results=strategy.search(ctx)`（`ctx=SearchContext(query_keys,CP1,current_step,task_key)`；内部 `record_query_keys`+建深度轨迹，**不 reset history**）| 同构建同调（`current_step`=step_idx）→ TrajectoryMixin 累积深度5 chain-aware |
| judge 收 `view=StoragePayloadView(storage)`、`history=HistoryView(list(action_hist),list(state_hist))`、`retrieval_signals` | 取 `strat.last_retrieval_signals()` 得 `s_pos/s_neg/delta_pos`；建 `FactorContext(results, StoragePayloadView(temp_storage), HistoryView(list(action_hist),list(state_hist)), ZScoreNormalization(library_stats))` → gate u_t Factor 得 `u_t` |
| check **返回后**：interceptor `broadcast_action(action_chunk)` → `_action_history.append(action_chunk[0])`（当步 action 进**下一步**历史，orch:356-383）| 每步 search **之后** append 该步 `action`（h5 `clean_action[0]`）到 `action_hist`（进下一步历史，纠正错位）；`record_action` 保 parity |
| episode 末 `on_episode_end`→`storage.close_search_session(sid)`（orch:321）| episode 末 `storage.close_search_session(sid)`（**显式关闭**） |

- **LOEO 防泄漏（G1 R1 finding 1 / R2 finding 2，唯一实现，不改生产接口）**：每条 calibration episode replay 前，**按 provenance 预过滤 entry 列表**——从合并 artifact entries 中剔除该 episode 自身的全部 entries，身份判定用 **`(task_id, trajectory_id)` 组合键**（builder 用 `h5_path.stem` 作 `trajectory_id`，跨 task 目录可撞同 stem → 仅 trajectory_id 会过度排除，故须组合 task 身份）。用剩余 entries 构造**临时只读 InMemoryBackend/CacheStorage**（`load_artifact` 等价路径，不碰 `QueryFilter`/strategy/backend 生产接口），**并显式沿用原合并 artifact 的 D⁺-only `library_stats`**（不从过滤后 D⁺∪D⁻ 重算——重算会把 D⁻ 折进 normalizer，破坏 Phase 4 的污染守卫）。原样 `DualRetrievalKnnStrategy` 跑之。**fail-loud**：断言临时检索池中身份键 == self 的 entry 数 **== 0**（而非弱断言"top-1 非自身"）。（不用 search 后删 id：D⁺ 只 over-fetch 2、D⁻ 只 top-1，删后无法恢复真 top-1/top-2。）
- 每行 = `{suite, task_id, init_state_idx, episode_id, step_idx, s_pos, s_neg, delta_pos, u_t(None if NaN), winner_id, winner_has_warm_snapshot(bool), episode_success}`（`winner_has_warm_snapshot` = winner payload 有 `start_t=0.5` 的 intermediates snapshot，Pass 2 查算，供 solver WS 分支 replay，finding 4）；写 `exp/zixuan_proposal/data/phase5_calib_rows_{suite}.jsonl`。u_t 非-NaN 覆盖率写入 driver 日志（R10）。

**离线 solve — `MarginGateCalibrator` + driver（G1 R1 finding 3/7）**：
- **归属固定**：新 **src 类** `src/openpi/cache/components/margin_gate_calibrator.py`::`MarginGateCalibrator`（离线-only，内聚照 `fit_from_scores` 范式；**运行时零耦合**——它只**产出** params，写进 YAML；在线 strategy 读 `margin_lambda`、judge 读 `gate_betas`，各自独立，λ 不反向耦合 judge）。公开接口：`classmethod calibrate(rows: list[dict], *, c_miss: float, c_warm: float, grid: GridSpec) -> dict`，返回 `{margin_lambda, gate_betas:{b0,b1,b2,b3}, threshold, warm_tiers}`（序列化边界 = 纯 JSON-able numeric，照 `params_to_dict`）。exp/ driver `solve_calibration.py` 只 load rows → 调 `calibrate` → dump JSON。
- **唯一参数化（固定项）**：`b1≡1.0`、`threshold≡0.5`（g-space FULL_HIT cutoff）、warm tier `start_t≡0.5`（∈ `CANONICAL_DENOISE_TIMESTEPS`，取 repo `WARM_START_T` 先例；WS 仅在 row `winner_has_warm_snapshot==True` 时触发——该 bool 已由 Pass 2 落进 rows，finding 4）。**搜索维（显式网格）**：`λ∈{0,.25,.5,.75,1,1.5,2}`(7) × `b0`(8，**per-λ 重算**：对每个 λ 先在全 rows 上算 `margin=s_pos−λ·s_neg` 分布，取分位[50,60,70,75,80,85,90,95] 的负值作该 λ 的 b0 候选，τ_hit=−b0) × `β₂∈{−2,−1,−.5,0,.5,1,2}`(7) × `β₃∈{0,.5,1}`(3) × `τ_warm_g∈{none,.2,.3,.4}`(4) = 4704 候选（离线 replay 极廉）。
- **L_cal 定义**（每候选在信号表逐步 replay）：`margin=s_pos−λ·s_neg`；`z=b0+1·margin+β₂·(u_t or 0 if None)+β₃·delta_pos`；`g=σ(z)`；三态（`g≥.5`→FH / `τ_warm_g` 非 none 且 `g≥τ_warm_g` 且 `row.winner_has_warm_snapshot`→WS / else MISS）。`bad(t):=episode_success==False`、`safe_reuse(t):=episode_success==True`（**complementary，全管线唯一 proxy**，D-P3）。`BadHitRate=Σ[FH∧bad]/(Σ[FH]+1e-8)`(Eq27)、`MissRate=Σ[MISS]/N`、`WarmCost=Σ[WS]·warm_cost(0.5)/N`（`warm_cost=1−.5·(1−start_t)`，repo 先例）。`L_cal=BadHitRate+c_miss·MissRate+c_warm·WarmCost`。
- **可行性 + tie-break（确定性，G1 R1 finding 3）**：**离线不设 SR 约束**（离线 replay base 轨迹，FULL_HIT 不真改 outcome → SR 不可离线测；MissRate 项即 SR 保护压力；真 `SR≥SR_base−ε` 由 Pass 3 强制，D-P4）。选 `argmin L_cal`；tie（|ΔL|<1e-9）确定性破：①更高 τ_hit（更低 b0，更保守）②更低 λ ③|β₂| 更小（简约）④参数元组字典序。
- **Emit**：`exp/zixuan_proposal/emit_calibrated_yaml.py`（照 `emit_threshold_yamls.py:48-112`）：读 base active YAML + solve JSON → `yaml.safe_dump` 写 `judge.gate_betas/threshold/warm_tiers/u_t_factor` + `search_strategy.margin_lambda` → **新** `config/dual_retrieval_calibrated{,_l10}.yaml`（不覆盖 active）。

**Pass 3 — 验证 rollout（确认出场门，G1 R1 finding 6 / R2 finding 3）**：
- 在 `I_val` 跑**三条同 init+seed 配对**的 rollout：① **cache-OFF baseline**（纯推理，**仅取 SR**：`--save-episode-results` 的 SR JSON → `SR_base(I_val)` = Eq 25 基线；**不产/不需 per-step JSONL**——裸 cache-OFF 无 gate → client 收不到 `__hit_meta__` → recorder 不触发，analyzer 对 baseline 只吃 episode-results，非 per-step）② illustrative（active YAML）cache-ON ③ calibrated YAML cache-ON。②③ 为 cache-ON eval（非 `--collect` → 可 3replica 2batch/48worker）；BHR/FFR/IR **仅由 ②③（有 gate → per-step JSONL）算**。②③ per-step JSONL 用 `per_step_recorder` gate 模式（`stamp_success=True`，每行 `hit_type`+`success`）+ `--save-episode-results`。
- 新 **`exp/zixuan_proposal/analyze_phase5_rollout.py`**（BHR aggregator 现无，须建）：读各跑 per-step JSONL，算 **分子/分母显式**的（`safe_reuse(t):=success==True`，与 solve 同一 proxy）：
  - `BHR=Σ[hit_type=FULL_HIT ∧ success==False]/Σ[hit_type=FULL_HIT]`（Eq27；零 FULL_HIT→BHR=0.0 且标 `n_full_hit=0` flag）；
  - `FFR=Σ[MISS ∧ success==True]/Σ[success==True]`（Eq28，safe_reuse proxy = episode 成功，与 solve/tests 一致）；
  - `IR=(Σ[MISS]+c_warm·Σ[WS])/N`（Eq26）；`SR` 用 `_aggregate_sr_from_episode_json`。
  - **`success==None`/不完整 episode**：从 BHR/FFR/SR 分子分母**剔除**（报告 `n_incomplete`）。
- **出场门判定**：calibrated vs illustrative（paired，同 I_val/seed）→ **BHR↓ 且 `SR_calibrated(I_val) ≥ SR_base(I_val) − ε` 且 IR_calibrated ≤ IR_illustrative**。报告 `exp/zixuan_proposal/analysis/phase5_calibration_report_{spatial,l10}.md`（纯 .md，最终报告入 `analysis/`）。

---

## 4. 改动文件清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `src/openpi/cache/components/judge.py` | `FailureAwareGateJudge`：`__init__` +`u_t_factor`/`library_stats`/`export_factor_outputs`；建 `ZScoreNormalization`+online Factor；`__call__` 算 u_t（NaN→丢项）+ `b2·u_t` 项 + **仅 export=True 时**填 `factor_outputs`（NaN→None）；lazy-import factor/norm 防循环 | src additive（gate 内） |
| `src/openpi/cache/components/margin_gate_calibrator.py` | **新**：`MarginGateCalibrator.calibrate(rows,*,c_miss,c_warm,grid)->params`（离线-only 内聚，照 fit_from_scores；L_cal 网格搜 + 确定性 tie-break；运行时零耦合） | src 新（离线机制） |
| `src/openpi/cache/config.py` | `_build_inner_judge` gate 分支转发 `library_stats`+`u_t_factor`+`export_factor_outputs`；`JudgeConfig` +`u_t_factor`（`export_factor_outputs` 复用现存）；b2==0 validator 放宽（b2≠0⇔u_t_factor 有效+in_memory+library_stats） | src additive（工厂/校验） |
| `docs/architecture/cache_system.md` | §5.6/§5.x：`failure_aware_gate` u_t 通道（β₂·u_t + export-gated factor_outputs）additive 更新——**L3 架构文档** | 文档 |
| `docs/cache/tutorial.md` | Judge 表 `failure_aware_gate` +`u_t_factor` 字段语义 + 标定说明指针 | 文档 |
| `docs/README.md` / `docs/cache/README.md` / `logs/README.md` | 索引同步（同 commit） | 文档索引 |
| `exp/zixuan_proposal/build_calibration_table.py` | 新：Pass 2 离线 replay（建库链 KeyBuilder + state-before/action-after 时序 + session 开关 + **per-episode 临时只读 storage 排除自身 trajectory** + fail-loud 0-self-entry + `winner_has_warm_snapshot`）→ `data/phase5_calib_rows_{suite}.jsonl` | exp 脚本 |
| `exp/zixuan_proposal/solve_calibration.py` | 新：load rows → `MarginGateCalibrator.calibrate` → params JSON（CLI：`--rows-jsonl --suite --c-miss --c-warm --out-json`） | exp 脚本 |
| `exp/zixuan_proposal/emit_calibrated_yaml.py` | 新：base YAML + params JSON → `yaml.safe_dump` 新标定 YAML | exp 脚本 |
| `exp/zixuan_proposal/analyze_phase5_rollout.py` | **新**：Pass 3 分析（读 cache-OFF baseline + illustrative + calibrated **三跑** per-step JSONL → BHR/FFR/IR/SR 分子分母 + paired provenance + success=None/零FH 处理；safe_reuse=success==True 与 solve 一致） | exp 脚本 |
| `exp/zixuan_proposal/config/dual_retrieval_calibrated{,_l10}.yaml` | 新：标定产物 YAML（emit 产出，含 u_t_factor + 标定 β/τ/λ；**不覆盖 active**） | 实验配置 |
| `exp/zixuan_proposal/analysis/phase5_calibration_report_{spatial,l10}.md` | 新：验证报告（BHR↓@SR≥SR_base−ε 且 IR 不升，paired） | 分析报告 |
| `tests/cache/test_failure_aware_gate_judge.py`（或新用例文件） | u_t 激活单测 + wire-invariance（export 默认 False）（见 §5） | 测试 |
| `tests/cache/test_config.py` | u_t_factor + export_factor_outputs 转发 + 放宽 validator config 用例 | 测试 |
| `tests/cache/test_margin_gate_calibrator.py`（新） | `calibrate` L_cal/网格/确定性 tie-break 单测 | 测试 |
| `tests/exp/test_calibration_pipeline.py`（新） | build_calibration_table（含 LOEO fail-loud + chain-aware parity）/ solve / emit / analyze 单测 | 测试 |

**不改**：`SimilarityJudge`/`SearchStrategy` Protocol 语义；`DualRetrievalKnnStrategy`；`RetrievalSignals`；orchestrator/interceptor/backend 内核；现有 online Factor / normalization / CompositeJudge；Phase 3 三示例 YAML 行为（逐字节）。

---

## 5. 测试策略

**NR1（u_t 关闭 → verdict + wire 双退化 — 主非回归）**：`tests/cache/` 全绿。`u_t_factor=None`+`b2=0`+`export_factor_outputs=false` → gate **verdict 逐值 == Phase 3**。**wire-invariance golden（G1 R1 finding 4）**：默认配置下 `judge_result.factor_outputs is None`（→ interceptor 不上 `__hit_meta__`），与 Phase 3 `__hit_meta__` 逐字节等。现有 `test_failure_aware_gate_judge.py` 用例 + 3 示例 YAML + 2 active YAML build 全绿。

**NR2（u_t 激活单测）**：
- **u_t 计算正确性**：fixture 库（tagged +1/-1）+ 构造 winner 有 F 步前向链 + history≥P → 断言 gate 取的 u_t == 直接调 online Factor `.extract` 的对应 key 值（同 `ZScoreNormalization`）。
- **logit + 三态**：给定 (m, u_t, d, β)，断言 `g=σ(b0+b1·m+b2·u_t+b3·d)` 逐值 + 三态分档正确；β₂ 生效（改 β₂ 改 g）。
- **NaN 优雅退化**：history<P / walk_next<F / winner 缺 robot_state → u_t NaN → **门退化为 margin-only**（`z=b0+b1·m+b3·d`，与 u_t_factor=None 逐值等价）。
- **factor_outputs opt-in（wire）**：`export_factor_outputs=true` → 填 `{schema,s_pos,s_neg,margin,delta_pos,u_t,g}` 且 NaN 全预转 None（`json.dumps(allow_nan=False)` 不炸）；`=false` → `factor_outputs is None`（NR1 wire golden）。

**config load + 校验**：合法 `failure_aware_gate + u_t_factor + b2≠0`（in_memory + library_stats）能 build；各非法触发 error：`b2≠0 但 u_t_factor 缺`、`u_t_factor 但非 in_memory`、`u_t_factor.descriptor/channel 非法`、`b2==0 且无 u_t_factor`（=Phase 3，须仍合法 build）；`export_factor_outputs` 正确转发 gate 分支。

**`MarginGateCalibrator` 单测**（`tests/cache/`）：构造已知信号表 → 断言 `BadHitRate/MissRate/WarmCost/L_cal` 逐值；网格搜返回全局 `argmin L_cal`；**确定性 tie-break**（构造两候选 L_cal 相等 → 断言按 τ_hit→λ→|β₂|→字典序 唯一选取，重跑同结果）。

**标定管线单测**（`tests/exp/`）：
- **build_calibration_table chain-aware + 时序 parity（G1 R2 finding 1）**：合成小 episode h5（≥P+1+F 步，token-level 模态）→ 经建库链得 keys → replay → 断言第 k 步 signals ≠ reset-history 单态 signals（轨迹累积）；**时序**：断言 state 在 search 前入 `state_hist`、action 在 search 后入 `action_hist`（当步 action 进下一步窗口）——与直接跑 orchestrator.check() 的 `_state_history/_action_history` 序列逐值对齐；行 schema 含 `winner_id/winner_has_warm_snapshot/u_t/episode_success`。
- **LOEO 0-self-entry（G1 R1 finding 1 / R2 finding 2）**：构造含 query episode 自身 trajectory 的 artifact → build 临时只读 storage → **断言池中 `trajectory_id==self` entry 数 == 0**（强断言，非"top-1 非自身"）；若绕过排除则 fail-loud `raise`。
- **solve 分支可 replay（G1 R2 finding 4）**：断言 WS 仅当 `row.winner_has_warm_snapshot`（无 snapshot 行永不判 WS）；`b0` 候选随 λ 变（per-λ margin 分位）；`safe_reuse=success==True` 与 analyze/L_cal 三处一致。
- **collect-topology fail-fast（G1 R1 finding 2）**：断言 Pass 1 采集命令构造用 `--replicas 1 --non-concurrent`；给 `replicas>1` / `concurrent` → `_validate_collect_isolation` fail-fast（守护串行采集不可绕）。
- **solve/emit**：solve JSON → `emit_calibrated_yaml` → `load_cache_config` 过校验 + `gate_betas/margin_lambda/u_t_factor` 值正确写入；**analyze**：合成**三跑**输入（baseline **episode-results-only**（仅 SR）+ illustrative/calibrated per-step JSONL）→ `SR_base(I_val)` 取自 baseline、BHR/FFR/IR 取自 ②③ → 分子分母逐值 + `success=None`/零-FH 处理正确。

**§4 Code 定向自检（无程序效力）**：编码期可随手 `uv run pytest tests/cache/test_failure_aware_gate_judge.py tests/cache/test_margin_gate_calibrator.py ...` 等子集自检（execution §4 Permitted，不满足 §6）。

**§6 Verify（唯一具程序效力，G1 R2 finding 5）**：目标 = **WA §2.7 全量非-manual 测试通过**（`uv run pytest`，本地 auto-skip `@pytest.mark.manual`/GPU 门控用例）。唯一**宪法级排除** = `tests/review_tests/`（execution_authority §1 明令 Execution 授权**禁止**读/列/跑 G2 reviewer 封闭空间——非裁量排除，而是授权律强制）。本改动 **blast-radius = `tests/cache/ tests/exp/`**（gate/config/calibrator + 标定脚本）作**主证据必须全绿**。其余非-manual 目录一并纳入 Verify 目标；Verify **执行时保存完整命令 + 结果输出**作可复现证据。**若** `tests/serving` 触发其**已知环境性挂起**，则捕获可复现证据（命令 + 卡住点）呈 owner 明示裁决是否临时缩范围——**任何缩范围都须 owner 明示,不在本 plan 预先按 fiat 排除**（仅 `tests/review_tests` 依 Review Authority §3.1 保持封闭，为唯一宪法级排除）。全绿（或 owner 裁决后的合规子集）方可进 §7。

---

## 6. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **bad(t) 用 episode 成败作 proxy 过粗**（失败 episode 的早期步未必"不安全复用"） | D-P3 明确定义 + 列为 G1 头号待定；Pass 3 真 rollout 验证兜底（真 SR/BHR）；备选：用 s⁻ 高（近失败）作细粒度 bad 标签，Phase 7 ablation 裁决 |
| R2 | u_t 激活破坏 Phase 3 非回归 | additive：`u_t_factor=None`+`b2=0` 逐值退化（NR1）；NaN u_t 也退化为 margin-only；现有 5 YAML build 全绿守卫 |
| R3 | 离线 replay 的 margin 分布 ≠ 在线（validate 用单态 reset-history） | Pass 2 **按轨迹顺序不 reset history** 忠实重建深度5 chain-aware（NR2 chain-aware 单测守卫）；Pass 3 真 rollout 终验 |
| R4 | cache-ON warmup 复用污染下游 signals/outcome | **三阶段分离**：Pass 1 cache-OFF 取纯 base 成败+观测，Pass 2 离线 replay 算 threshold-无关信号 → 零复用污染 |
| R4b | **self-retrieval / cal-val 泄漏**（Phase 4 D⁻ 与本期 query 同源，确定性下自身轨迹重合虚假优化 BHR/s⁻）（G1 R1 finding 1） | `I_cal`/`I_val` 物理不相交（按 init_idx 二分）+ Pass 2 replay 按 `trajectory_id` LOEO 排除自身 entries + **fail-loud 自匹配断言**；§5 LOEO 测试守卫 |
| R5 | u_t 描述符/通道选错（safe vs risky 方向、action vs state） | 默认 `direction_online_state`（orientation "safe"，robot_state 通道，β₂>0 期望）；descriptor/channel/window YAML 可配；标定自动定 β₂ 符号；G1 定默认（D-P7） |
| R6 | SR 离线不可测 | **离线不设 SR 约束**（`argmin L_cal`，MissRate 项即 SR 保护压力，D-P4）；**真 `SR_calibrated(I_val)≥SR_base(I_val)−ε` 由 Pass 3 强制**（同 split baseline，finding 3）；ε 默认 0.02（可调） |
| R7 | c_miss/c_warm/ε 无先例、取值主观 | 默认 c_miss=1.0（miss=满推理）、c_warm=0.75（repo WARM_COST 先例）、ε=0.02；作 CLI 参数 + 报告标注；Phase 7 敏感性 |
| R8 | gate 内建 Factor/norm 引入 judge→factors import 环 | lazy-import（照 `_build_inner_judge` 的 `from ... import` 先例）；FactorContext/kernel 已是被 CompositeJudge import 的稳定层 |
| R9 | Pass 1/3 需重起 GPU 设备（Phase 4 已关） | owner 已同意（Q1）；**Pass 1 串行采集**（`--replicas 1 --non-concurrent`）+ Pass 3 eval 拓扑（3replica）；具体设备起停/failover **执行时向 owner 明示请求/确认**（无 in-repo runbook，不作 plan 依赖，G1 R2 finding 6） |
| R10 | walk_next 在 winner 近链尾/fork 频繁 → u_t 大量 NaN → β₂ 失效 | NaN→丢项优雅退化（门仍靠 margin）；build_calibration_table 报告 u_t 非-NaN 覆盖率;覆盖率过低则 G1/Phase7 重议是否值得 u_t |
| R11 | 标定过拟合 held-out（与 Phase 7 eval 集 pruned_init 不同划分） | held-out 与 pruned_init 天然不相交（Phase 4 已隔离）；Phase 7 在 pruned_init 独立评测作泛化检验 |
| R12 | `calibrate()` 归属（横跨 strategy λ + judge β/τ）（G1 R1 finding 7） | **已定**：新 src 类 `MarginGateCalibrator`（离线-only，产出 params 写 YAML，运行时零耦合——strategy 读 λ / judge 读 β 各自独立）；D-P5 |
| R13 | factor_outputs 破坏 Phase 3 wire 兼容（G1 R1 finding 4） | 复用现存 `export_factor_outputs`（默认 False → factor_outputs None → 不上 `__hit_meta__`）；NR1 wire golden 守卫 |
| R14 | Pass 3 缺 BHR aggregator / 非 paired（G1 R1 finding 6） | 新 `analyze_phase5_rollout.py`（BHR/FFR/IR/SR 分子分母 + success=None/零FH 处理）；illustrative vs calibrated 同 I_val/seed paired |

---

## 7. 设计决策（G1 R1 已依 finding 全部定死）

- **D-P1 方法学 = 三阶段 rollout/replay/rollout**（owner Q1 选定）。Pass1(cache-OFF **串行**采集 base)/Pass2(离线 replay chain-aware 信号，不 reset history)/solve(`MarginGateCalibrator`)/Pass3(cache-ON eval 验证)。忠实 chain-aware（Pass2 累积 history）+ 零复用污染（阶段分离）。
- **D-P2 u_t 落点 = gate 内计算**（非 RetrievalSignals 字段、非 strategy）。gate 已收 view/history，复用 online Factor + ZScoreNormalization + kernel；strategy/RetrievalSignals 逐字节不变。
- **D-P3 bad/safe 标签 = episode 成败 proxy（全管线唯一，complementary）**：`bad(t):=success==False`、`safe_reuse(t):=success==True`（solve L_cal、analyze FFR、tests 三处**同一 proxy**，G1 R2 finding 4）。`BadHitRate=Σ[FH∧bad]/(Σ[FH]+1e-8)`（Eq 27）。粗粒度风险 R1 由 Pass 3 真 rollout 终验兜底；备选 s⁻-based 细标签留 Phase 7 裁决。
- **D-P4 SR 约束分层 + 同-split 基线（G1 R1 finding 3 / R2 finding 3）**：**离线 solve 不设 SR 约束**（replay base 轨迹 → FULL_HIT 不改 outcome → SR 离线不可测；MissRate 项即 SR 保护压力，`argmin L_cal`）。**`SR_base` ≝ Pass 3 在 `I_val` 上另跑的 cache-OFF baseline**（与 `SR_calibrated` 同 split+seed，否则跨 split 难度差使约束失效）；`I_cal` cache-OFF SR 仅数据质量记录。**真 `SR_calibrated(I_val) ≥ SR_base(I_val) − ε` 由 Pass 3 强制**。
- **D-P5 `calibrate()` 归属 = 新 src 类 `MarginGateCalibrator`（G1 finding 7，已定）**：`src/openpi/cache/components/margin_gate_calibrator.py`，离线-only classmethod `calibrate(rows,*,c_miss,c_warm,grid)->params`，产出纯 JSON-able numeric 写 YAML；**运行时零耦合**（strategy 独立读 margin_lambda、judge 独立读 gate_betas，λ 不反向耦合 judge 在线责任）。
- **D-P6 标定唯一参数化 + 显式网格（G1 R1 finding 3 / R2 finding 4，已定）**：固定 `b1≡1`、`threshold≡0.5`（g-space FH cutoff）、warm `start_t≡0.5`（∈CANONICAL_DENOISE_TIMESTEPS，WARM_START_T 先例；WS 仅当 row `winner_has_warm_snapshot==True`——该 bool Pass 2 落 rows）。网格 `λ`(7)×`b0`(8，**per-λ 重算 margin 分位**)×`β₂`(7)×`β₃`(3)×`τ_warm_g`(4)=4704。确定性 tie-break：τ_hit↑→λ↓→|β₂|↓→字典序。
- **D-P7 u_t 默认描述符 = `direction_online_state`**（orientation "safe"，robot_state 通道，单窗口 (P,F) YAML 可配）。理由：提案 "kinematic-quality"、"safe" 方向语义与 γ_hit≥ 对齐、robot_state 是观测侧运动质量。
- **D-P8 标定 suite = spatial + l10 两个**（与 Phase 4 建库一致，各出一套标定 YAML）。
- **D-P9 c_miss=1.0 / c_warm=0.75 / ε=0.02 默认**（c_warm 复用 repo WARM_COST 先例；均 CLI 可调 + 报告标注）。
- **D-P10 三层数据分离（G1 finding 1，已定）**：`I_cal`（init_idx 偶）/`I_val`（奇）物理不相交；library↔cal 泄漏用 Pass 2 `trajectory_id` LOEO 排除 + fail-loud 自匹配断言消解（物理 rebuild D⁻ 于子 fold 弃：Phase 4 rework + D⁻ 缩水，LOEO 等效且有先例）。
- **D-P11 factor_outputs opt-in（G1 finding 4，已定）**：复用现存 `export_factor_outputs`（默认 False → factor_outputs None → 不上 wire，Phase 3 逐字节）；带 `schema` version；仅 Pass 3 观测用，非标定关键路径。
- **D-P12 Pass 1 串行采集（G1 finding 2，已定）**：`serve_policy --collect --replicas 1 --non-concurrent`（`_validate_collect_isolation` 硬约束）；h5 供 replay、`--save-episode-results` 供成败；Pass 3 eval（非 collect）用 3replica。

---

## 8. 出场条件（本期 Code→G2→Verify 的完成定义）

- NR1（u_t 关闭 → verdict + wire 双退化 == Phase 3，tests/cache 全绿）+ NR2（u_t 激活/NaN 退化/factor_outputs opt-in 单测）通过；
- `MarginGateCalibrator` 单测（L_cal/网格/确定性 tie-break）+ 标定管线单测（build_table chain-aware parity + LOEO fail-loud + collect-topology fail-fast / solve / emit / analyze）+ config 放宽校验用例通过；
- Pass 1（`I_cal` 串行采集）/Pass 2（LOEO replay）/Pass 3（`I_val` **三跑**：cache-OFF baseline + illustrative + calibrated，同 init/seed paired）跑通两 suite，`phase5_calibration_report_{spatial,l10}.md` 显示 **BadHitRate↓ @ `SR_calibrated(I_val) ≥ SR_base(I_val) − ε` 且 IR 不升**（路线图出场门）；
- L3 架构文档（`cache_system.md`）additive 更新 + 索引同步；
- 标定 YAML `load_cache_config` 过校验 + build。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-10 23:20 CDT

- [Blocking] [Concern] Pass 2 的文档化命令无法产生任何 u_t 信号，因而本期“激活 β₂·u_t”的核心目标实际不可达——reasoning: `build_calibration_table.py` usage 和 `build_table()` 都从 `dual_retrieval_active{,_l10}.yaml` 构建 gate，但这两个 YAML 均无 `u_t_factor`；实现的 `FailureAwareGateJudge._compute_u_t()` 因此对每行返回 None。calibrator 在 u_t 全空时的目标与 b2 无关，其 `|b2|` tie-break 必然选 b2=0，emitter 随后又不写 `u_t_factor`。请为 Pass 2 增加明确的 u_t factor 输入（独立 replay config 或 CLI/config override），在建表前 fail-loud 要求该 factor 激活，并以 spatial/l10 实际命令形状的测试证明至少有可计算 u_t 行。Reviewer probe `test_pass2_reference_configs_activate_u_t` 已复现失败。
- [Blocking] [Concern] LOEO 身份使用 `(task_key, h5_path.stem)`，无法排除 Phase 4 中同 init 的自身轨迹——reasoning: 真实 spatial artifact 的 D⁻ `trajectory_id` 如 `episode_0275_20260710_042540_118699`，其中包含采集时间戳；Phase 5 重采同一 global episode 275 会得到新时间戳/stem。`_episode_identity` 与 `_entry_identity` 因此不相等，`_temp_storage_excluding` 保留所有真正的 self entries，而当前“池中该新 stem 为 0”断言仍会通过。请改用时间戳无关的 provenance 键（例如 `(task_id, init_state_idx/global_episode_id)`，可从 H5 attrs 与 `episode_<global-id>_...` 解析），并用“同 episode id、不同 timestamp”测试守卫。Reviewer 已对本地真 artifact 亲验 792 个 D⁻ entry 的 timestamped identity，`test_loeo_identity_survives_recollection_timestamp` 失败。
- [Blocking] [Concern] Pass 3 analyzer 没有实现计划承诺的 paired provenance / I_val / same-seed 门禁——reasoning: `analyze_phase5_rollout.py` 只分别求聚合指标，从不比较 baseline/illustrative/calibrated 的 `(task_id, init_state_idx, seed[, episode_id])` 集合，也不验证 `I_val` 的奇数 split；三份完全不同 init/seed 的伪造输入仍返回 exit-gate PASS/exit 0。请在计算前对 episode-results 三跑做唯一性与身份集合等值校验，并将 cache-ON per-step rows 绑回同一 episode provenance；任何不配必须 fail-loud。Reviewer probe `test_analyzer_rejects_unpaired_provenance` 已复现 analyzer 错误接受。`build_table()` 也应对 I_cal 偶数 split fail-loud，避免防泄漏设计完全依赖人工目录选择。
- [Blocking] [Concern] 文档索引未与本轮 docs 修改同步，构成 Working Agreement §4 Index Sync Rule 违规——reasoning: 工作树修改了 `docs/architecture/cache_system.md` 和 `docs/cache/tutorial.md`，但 `docs/README.md` 与 `docs/cache/README.md` 均无对应 diff；计划 §4 也明确把两份索引列为交付物。请在同一轮更新相应条目的 Phase 5 u_t/标定语义，并保持 `logs/README.md` 同步。
- [Blocking] [Concern] 新 `u_t_factor` 配置校验对恶意/误配类型不 fail-clean，并会静默截断非整数窗口——reasoning: validator 直接执行 `int(u_tf["past"])` / `int(...future)`；`past: bad` 会泄出原始 ValueError（而非项目 `ConfigValidationError`），`past: 1.5` 则被截断为 1 并通过，工厂再次 int 截断。请严格校验整数且 >=0，将所有类型/范围问题聚合为 `ConfigValidationError`，并补 string/float/bool/negative 测试。
- [Non-blocking] [Concern] `MarginGateCalibrator` 的 tie 契约与测试尚未精确覆盖——reasoning: plan 定义 `|ΔL|<1e-9` 视为 tie，实现却将 L_cal round 到 12 位；`test_calibrate_deterministic_and_tie_break` 只验“重跑相同”，没有构造等目标候选并断言 τ_hit→λ→|b2|→字典序。请统一 tolerance 实现并加真 tie-order 守卫。
- [Non-blocking] [Concern] G1 计划列出的真实 calibrated YAML 与 Phase 5 report 尚未出现，本轮只能审查机制代码——reasoning: 这些产物依赖后续 Pass 1/2/3 和 §6 Verify，因此不单独作为当前 G2 blocker；但未经 `load_cache_config` 的真实 emitted YAML 与未达出场门的报告不得在后续被宣称为 Validated。

### G2 Round 1 — Executor — 2026-07-11

逐条回应（顺序对应上方 G2 R1 reviewer 项）：

1. **Accepted** — Pass 2 命令不产生 u_t：`build_calibration_table.py` 加 `--u-t-descriptor/channel/past/future`（默认 D-P7 direction_online_state/past2/future1），`build_table` 在 `build_cache_components` **前注入** `config.checkpoints["cp1"].judge.u_t_factor` 并 `validate_cache_config` 复验，构建后 **fail-loud** 断言 `gate._u_t_factor is not None`，收尾再 **fail-loud** 断言 u_t 非-NaN 覆盖 >0（全 None 即 abort）。b2 与该结构因子正交（b2 仍由 calibrator 拟合）。测试：`test_reference_config_accepts_injected_u_t_factor`（reference 配置注入后过校验）+ `test_build_rows_schema_and_chain_aware_timing`（b2≠0 gate 实算出非-None u_t 行）。
2. **Accepted** — LOEO 时间戳漏排：改为**时间戳无关身份** `(task_key, global_episode_id)`——新增 `_parse_global_episode_id` 从 `episode_<id>_<timestamp>...` 解析稳定全局 id；`_entry_identity` 用 entry.trajectory_id 解析、`_episode_identity` 优先 h5 `episode_id` attr（否则解析 stem），**None 时 fail-loud**（不静默回退时间戳 stem）。测试 `test_loeo_identity_survives_recollection_timestamp`（`episode_0275_<ts_A>` 的 D⁻ entry 被 `(task,275)` 排除，`count()` 归 1）+ `test_parse_global_episode_id`。
3. **Accepted** — analyzer 无 paired 门禁：`analyze_phase5_rollout.py` 加 `check_paired_provenance(baseline,illustrative,calibrated, illus_steps, calib_steps)` —— 三跑 `(task_id,init_state_idx)` 集**必须相等** + 全为**奇 init（I_val）** + **单一共享 seed** + cache-ON per-step rows 绑回同一 episode 集，任一不配 fail-loud；main 计算前调用。`build_table` 亦对 **I_cal 偶 split** fail-loud（init_idx 非偶/None 即 abort）。测试：`test_analyzer_rejects_unpaired_provenance`（reviewer probe）+ reject even-init + reject seed-mismatch + accept-paired。
4. **Accepted** — doc 索引未同步：`docs/README.md`（cache_system.md §5.6 + tutorial 两条目）+ `docs/cache/README.md`（tutorial 条目）均加 Phase 5 β₂·u_t / u_t_factor / export_factor_outputs / MarginGateCalibrator 语义；`logs/README.md` 保持同步。
5. **Accepted** — validator 非 fail-clean + 静默截断：`config.py` u_t_factor past/future 校验改为 `isinstance(bool) or not isinstance(int)` → error、`<0` → error，**全部聚合为 ConfigValidationError**（不再 `int()` 泄 ValueError、不再截断 1.5）。测试 `test_u_t_factor_bad_window_rejected_clean`（string/float/bool/negative ×5 参数化）。
6. **Accepted** — tie 契约不精确：`round(l_cal, 12)` → `round(l_cal, 9)` 统一到 plan 的 1e-9 tolerance。测试 `test_calibrate_tie_break_prefers_higher_tau_hit`（构造 margins[0.1,0.3] 两 b0 候选同 verdict/同 L_cal → 断言选更高 τ_hit=b0=-0.28）。
7. **Accepted**（无需改码）— 真实 calibrated YAML / report 未现：确认约束——这些是 Pass 1/2/3 runtime 产物（G2 后 §6 Verify 及运行时生成）；emit 产物必经 `load_cache_config`+`validate_cache_config` 校验（已由 `test_apply_params_*` roundtrip 守卫），report 须达出场门方可标 Validated，本轮不作此宣称。

修订完成，全 blast-radius `tests/cache/ tests/exp/` **2029 pass / 6 skip**、ruff 全过。重新提交 G2 复审（R2）。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-07-11 01:52 CDT

- [Blocking] [Concern] Pass 3 的 paired provenance 校验不能消费生产端实际写出的 per-step JSONL——reasoning: `examples/libero/main.py` 两条 recorder 路径均把 rollout 子集索引写为 `subset_init_state_idx`，episode-results 才使用 `init_state_idx`；但 `analyze_phase5_rollout._provenance_keys()` 对所有输入一律强制读取 `init_state_idx`。因此用真实 illustrative/calibrated JSONL 运行 analyzer 会在指标计算前以“missing task_id/init_state_idx”退出，Pass 3 出场门不可执行。请区分 episode-results 与 per-step schema（或先规范化为统一 canonical key），并用 `main.py` 的真实字段形状测试。Reviewer probe `test_analyzer_accepts_production_per_step_schema` 已复现失败。
- [Blocking] [Concern] per-step provenance 只做 `s_keys.issubset(b_keys)`，会静默接受缺少整条 episode 的截断日志——reasoning: 只要残存行没有引用集合外 episode，illustrative/calibrated JSONL 即使只覆盖 I_val 的一个真子集也会通过，随后 BHR/FFR/IR 在不同或不完整样本上聚合，破坏 paired 比较。请要求每条 cache-ON per-step 日志覆盖的 episode identity 集与三跑 episode-results 的完整共享集相等（并对空/重复等不可验证情形 fail-loud）；Reviewer probe `test_analyzer_rejects_missing_per_step_episode` 已复现错误接受。
- [Blocking] [Concern] “单一共享 seed”校验在三份 episode-results 全部缺 seed 时被绕过——reasoning: 当前条件以 `if all_seeds` 为前置；三个 seed 集均为空会直接通过，报告仍宣称 same-seed paired。计划要求 fail-loud 验证同 seed，而非在无法验证时默认成立。请要求每个 episode row 都有 seed、每跑恰有一个 seed、且三跑 seed 相同；Reviewer probe `test_analyzer_rejects_unverifiable_missing_seed` 已复现错误接受。
- [Verified] Round 1 其余整改经代码与探针复核成立：Pass 2 CLI 注入并强制 u_t 覆盖、LOEO 使用时间戳无关 global episode id、I_cal/I_val 奇偶守卫、文档索引同步、u_t 窗口严格整数校验、1e-9 tie 行为测试均已落地。
- [Tests] 正式 blast-radius：`PYTHONPATH=. uv run pytest tests/cache/ tests/exp/ -q` → **2029 passed, 6 skipped**。审查者封闭探针：`PYTHONPATH=. uv run pytest tests/review_tests/test_tracer_phase5_g2.py -q` → **3 failed, 2 passed**；三项失败分别对应上述三个 blocker。

### G2 Round 2 — Executor — 2026-07-11

逐条回应（顺序对应上方 G2 R2 三个 blocker；[Verified]/[Tests] 为审查者确认项，无需 accept/reject）：

1. **Accepted** — analyzer 不消费生产 per-step schema：亲验 `examples/libero/main.py` —— 每步 recorder 行写 `subset_init_state_idx`（:621），episode-results 写 `init_state_idx`（:692）+ `seed`（:695），二者同值不同名。新增 `_init_key(row)`：先读 `init_state_idx` 再回退 `subset_init_state_idx`，`_provenance_keys` 全走 canonical key → 真实 illustrative/calibrated JSONL 可分析。测试 `test_analyzer_accepts_production_per_step_schema`（per-step 只含 subset 字段仍解析）+ 现有 analyzer 测试 `_step` 改用生产端 `subset_init_state_idx`。
2. **Accepted** — per-step `issubset` 放行截断日志：改为 **`s_keys == b_keys` 全等**（缺任一 episode / 有集合外 episode 均 fail-loud），并加**非空**守卫（baseline 空即 raise）。测试 `test_analyzer_rejects_missing_per_step_episode`（reviewer probe：per-step 少一条 episode → "full paired I_val set" raise）。
3. **Accepted** — 三跑全缺 seed 时 same-seed 被绕过：seed 改为**硬性必需**——`_provenance_keys(require_seed=True)` 对 episode-results 每行缺 seed 即 raise；`check_paired_provenance` 要求三跑各**恰一个 seed** 且**三者相等**（不再 `if all_seeds` 前置默认成立）。per-step 行无 seed 故 `require_seed=False`。测试 `test_analyzer_rejects_unverifiable_missing_seed`（reviewer probe：三跑无 seed → "missing 'seed'" raise）。

[Verified]/[Tests] 项已知悉：R1 六项整改经审查者复核成立，正式 blast-radius 2029→**2032 pass/6 skip**（本轮 +3 analyzer probe 对应测试），ruff 全过。重新提交 G2 复审（R3）。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-07-11 02:19 CDT

- [Verified] Round 2 三项 blocker 的直接修订均成立：canonical init key 可消费生产 `subset_init_state_idx`；per-step episode identity 集改为与共享 I_val 集全等；episode-results 每行 seed 必需且三跑各恰一个相同 seed。上一轮五个封闭探针现全部通过。
- [Blocking] [Concern] episode-results 的 identity 唯一性仍未校验，集合化会吞掉重复甚至互相矛盾的 episode 记录——reasoning: `_provenance_keys()` 直接向 `set` 加 `(task_id, init_idx)`，从不比较行数或拒绝已存在的 key；因此 baseline/calibrated 各一行而 illustrative 对同一 identity 同时含 `success=true` 与 `success=false` 时，三者 provenance sets 与 seed sets 仍完全相等并通过。随后 `_sr_from_rows()` 却按原始 rows 计数，重复/矛盾行会直接改变 `SR_illustrative` 或 `SR_calibrated`，使 paired 出场门建立在非一一对应样本上。这也未满足 Round 2 finding 明示的“空/重复等不可验证情形 fail-loud”和 Round 1 的“唯一性与身份集合等值校验”。请仅对三份 episode-results 强制 `(task_id, init_idx)` 唯一（per-step 同 episode 多行当然允许），发现重复立即 fail-loud，并补同 key 同/异 success 的测试。Reviewer probe `test_analyzer_rejects_duplicate_episode_identity` 已复现错误接受。
- [Tests] 正式 blast-radius：`PYTHONPATH=. uv run pytest tests/cache/ tests/exp/ -q` → **2032 passed, 6 skipped**；Ruff → **All checks passed**。审查者封闭探针：`PYTHONPATH=. uv run pytest tests/review_tests/test_tracer_phase5_g2.py -q` → **1 failed, 5 passed**；唯一失败对应上述 identity 唯一性 blocker。

### G2 Round 3 — Executor — 2026-07-11

1. **Accepted** — episode-results identity 唯一性未校验：`_provenance_keys` 加 `require_unique` 形参——对**三份 episode-results** 启用（`(task_id, init_idx)` 重复即 fail-loud，无论 success 相同或矛盾），**per-step 不启用**（同 episode 多行=多步为合法）。这样 set 化不再吞掉重复/矛盾行，`_sr_from_rows` 的按行 SR 计数与 paired provenance 一一对应。测试 `test_analyzer_rejects_duplicate_episode_identity`（reviewer probe：同 key 同/异 success 参数化两例均 raise "duplicate episode identity"）+ `test_analyzer_allows_per_step_multiple_rows_per_episode`（守 per-step 多行不误伤）。

[Verified]/[Tests] 项知悉：R2 三 blocker 直接修订经复核成立、上轮 5 探针全过。全 blast-radius 2032→**2035 pass/6 skip**（本轮 +3 测试），ruff 全过。重新提交 G2 复审（R4）。

### G2 Round 4 — Reviewer — APPROVED — 2026-07-11 02:26 CDT

- [Verified] Round 3 blocker 已完整修复：三份 episode-results 均以 `require_unique=True` 强制 `(task_id, init_idx)` 一行一 identity，重复记录无论 success 相同或矛盾均 fail-loud；per-step 保持允许同 episode 多步。审查者重复/矛盾 identity 探针及全部既有 provenance 探针通过。
- [Verified] G2 历轮阻断项均已闭环：Pass 2 u_t 注入与覆盖 fail-loud、时间戳无关 LOEO、I_cal/I_val 分割守卫、生产 per-step schema 兼容、三跑 identity 全等/唯一/same-seed、文档索引、配置 fail-clean、calibrator tie 契约均有代码与测试证据。
- [Tests] `PYTHONPATH=. uv run pytest tests/cache/ tests/exp/ -q` → **2035 passed, 6 skipped**；`PYTHONPATH=. uv run pytest tests/review_tests/test_tracer_phase5_g2.py -q` → **6 passed**；Ruff → **All checks passed**；staged diff whitespace 检查通过。
- [Scope] 本裁决批准机制代码进入 §6 Verify；真实 calibrated YAML、两 suite Pass 1/2/3 rollout 与 Phase 5 report 仍是后续运行时出场条件，本裁决不提前宣称这些产物已 Validated。
