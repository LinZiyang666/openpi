# logs/

Implementation plans and design records. For project architecture docs, see [`docs/README.md`](../docs/README.md).

## Directory Structure

```
logs/
├── README.md                              # This index
├── <active logs>                          # In Progress / Plan / Design Only
└── archive/                               # Validated / Implemented / Done-High-Risk / Historical
```

**Top level**: files actively being worked on (In Progress, Plan, Design Only).
**`archive/`**: completed or historical files — no longer the active source of truth.

English translations (`*.en.log.md`) are folded under the primary entry as `[EN]` and do not occupy their own row.

## Status Legend

| Status | Where | Meaning |
|--------|-------|---------|
| `In Progress` | top | Actively being executed or updated |
| `Plan` | top | Task breakdown or proposed workflow; not implemented yet |
| `Design Only` | top | Technical design exists, but implementation is not confirmed |
| `Implemented` | archive | Code exists, but final validation or sign-off is pending |
| `Validated` | archive | Implemented and explicitly verified |
| `Done-High-Risk` | archive | Code landed, but no test coverage; known risk points remain |
| `Historical` | archive | Kept for record; not the active source of truth |

---

## Active Logs

### Cache System

| File | Status | Description |
|------|--------|-------------|
| [tracer_retrieval_refinement_roadmap.log.md](tracer_retrieval_refinement_roadmap.log.md) | `Roadmap` (Design-Grounded, 2026-07-07 创建；Phase 0 ✅ 架构判定/可行性亲验) | **L0 研究文档**: 合作者提案 `TRACER_RETRIEVAL_REFINED_PROPOSAL.pdf`（Action-Compatible Failure-Aware Retrieval）的模块化落地路线图。3 机制映射我们部件：**M1 结果兼容投影**（新 KeyBuilder）/ **M2 失败感知双检索**（D⁺/D⁻ margin 挡不安全复用，Claim 1 最硬 = 新双检索 SearchStrategy + 失败感知 Composer + `QueryFilter.outcome`/`retrieval_signals` 两处 additive 缝）/ **M3 动态链深**（新 SearchStrategy，与 M2 并入同一策略）。核心原理 **机制(代码) vs 参数(训练值) 分离** + **惰性默认退化到现系统**（可证非回归）+ **训练逻辑内聚、执行离线**（照 `ScoreNormalizer.fit_from_scores` 范式，禁在线自训练）。**循序渐进 7 期**：Phase 1 M3 深度（纯策略/零框架/零训练，趟通流程）→ Phase 2 M1 投影骨架（identity 默认）→ Phase 3 M2 失败感知骨架（惰性 D⁻+手设参数，含 2 additive 缝）→ Phase 4 D⁻ 建库（数据非训练）→ Phase 5 阈值标定（轻）→ Phase 6 M1 投影训练（重，Claim 2 necessity-check 条件触发）→ Phase 7 集成评测+ablation。与 gate 线（`gate_exploration_roadmap`）正交（那条管"要不要搜"，本线管"搜得准/判得安全"）。每期实现仍为独立 L2/L3，各走 Plan→G1→Code→G2→Verify。附已验证代码锚点表 + 提案方程↔部件对照 |
| [tracer_phase2_projection_key_builder.log.md](tracer_phase2_projection_key_builder.log.md) | `Implemented` (G1 R2 + G2 R2 APPROVED + §6 Verify 1018/6 green 2026-07-08；待 owner 确认归档) | **L2**: TRACER roadmap Phase 2 — **M1 结果兼容投影 KeyBuilder 骨架**（identity 默认）。新 `ProjectionKeyBuilder` 包一个无状态 pool builder，`build()` 对余弦字段（vision_*/prompt_emb）过每模态线性投影 `z=xWᵀ`；**无权重→identity→逐值==内层 pool**（golden 非回归），`robot_state` 恒原样（L2 语义）。库侧 `_create_builder` 与在线 `_build_key_builder` 用**同一 builder 类** → 两侧过同一头 → backend 仍普通 cosine（零 backend 改）。`ProjectionParams` load/save 照 `ScoreNormalizer` 范式（torch.save 张量）；类上 `fit()` = InfoNCE 投影拟合**机制**（吃已备训练张量 + 合成数据单测），**本期不跑** —— 真实 payload→c^A/c^X 兼容标签 + 离线 driver + 重跑押 Phase 6（owner 定「机制版」边界）。config +`ProjectionKeyBuilderConfig`/`projection` 类型/工厂分支/校验，库侧 +`_create_builder`/`_get_vector_dims` 分支 + 2 CLI。零 orchestrator/interceptor/backend/judge/现有 KeyBuilder 改动。出场门 = identity golden + 带权维度自洽 |
| [tracer_phase3_failure_aware_skeleton.log.md](tracer_phase3_failure_aware_skeleton.log.md) | `Implemented` (G1 R3 + G2 R1 APPROVED + §6 Verify 1056/6 green, 2026-07-08; 待 owner 确认归档) | **L3**: TRACER roadmap Phase 3 — **M2 失败感知双检索骨架**（惰性 D⁻ + 手设参数，不训练）。三可审子块：**3a 缝**（`QueryFilter.outcome`+`CacheEntry.outcome` 单-artifact 分池 tag + in_memory `_filter_entries`；`retrieval_signals` 侧信道 strategy→orchestrator(getattr)→judge(kwarg 直接消费；DumpingJudge wrapper 转发)，全默认 None → 逐字节不变）/ **3b 策略**（新 `DualRetrievalKnnStrategy`：D⁺/D⁻ 各搜一次算 `margin=s⁺−λ·s⁻`/`Δ⁺=top1−top2`；D⁻ 空→s⁻=0→margin=s⁺；复用 Phase 1 `DepthPolicy` 机构上移 `TrajectoryMixin` 共享，常数深度默认=非回归）/ **3c 门**（新 `SimilarityJudge` 类型 `failure_aware_gate`：`g=σ(β₀+β₁m+β₃Δ⁺)` 三态；直读 results[0].score+retrieval_signals；β/τ/λ YAML 手设；默认 β₃=0 → σ(β₁(s⁺−τ))≥0.5⟺s⁺≥τ **逐值退化 ThresholdJudge**；u_t kinematic 项分期 Phase 5——G1 R1 finding 1 后 composer→judge，因 CompositeJudge 强制 ≥1 factor）。additive-only 触点 5 处 + 2 新组件 + config 工厂/校验；不改 Protocol 现有语义/backend-ABC/orchestrator/judge 内核。退化契约 NR1（缝）+NR2（策略逐值）+NR3（门数学）+NR4（整栈）。上位依据 [`tracer_retrieval_refinement_roadmap`](tracer_retrieval_refinement_roadmap.log.md) Phase 3。真 D⁻ 押 Phase 4、标定押 Phase 5 |
| [tracer_phase4_failure_library.log.md](tracer_phase4_failure_library.log.md) | `Implemented` (§4 Code + cache-OFF 采集/build/validate 两 suite 出场门 PASS；⚠ 污染修正重采 2026-07-10；待 G2 复核) | **L2**: TRACER roadmap Phase 4 — **失败库 D⁻ 构建**（数据管线，非训练）。owner 定 D⁻ = 新失败 rollout 采集（全 5-key，proposal-literal Eq-16）；⚠ **首轮 D⁻ 误采在 `pruned_init`（=Phase 7 评估集）→ 与 D⁺ held-out 划分相反+泄漏 → 全部污染数据删除，D⁻ 在 held-out 池 `db_init/libero` 重采**；**修正后实收 spatial 18 失败/792 D⁻ step（合并 1810 entries）+ l10 85 失败/8840 D⁻ step（合并 11480 entries），两 suite 出场门 3-gate PASS**；census 证伪 roadmap §2.8（gate_rows 失败步只有 robot_state、无 vision/prompt 向量 → 多模态 D⁻ 本地无料，须采集）。交付：① 共享 builder pool 路径 `_process_episode` +`--outcome-filter{success,failure,all}`（默认 success；`cp1_llm_layer_extract` model 路径非默认 filter **fail-loud**）② `build_dual_artifact.py`（load D⁺ tag +1 ∪ build D⁻ tag −1 → 单 artifact，`library_stats` 仅按 D⁺）③ `validate_dual_artifact.py`（出场 gate：outcome 覆盖 + 非平凡 margin → `analysis/*.md`）④ `dual_retrieval_active.yaml`（enable_dual:true）⑤ tests/exp 合成 fixture。采集须 **cache-OFF**（活 cache 下 CP1 FULL_HIT 短路跳 stage2/3 → CollectionPolicy 步洞）。**src/ 零改**（Phase 3 已接通消费侧）。消费硬契约：enable_dual=True 下 outcome=None 两池皆弃 → 全 D⁺ entry 必 tag +1。上位依据 [`tracer_retrieval_refinement_roadmap`](tracer_retrieval_refinement_roadmap.log.md) Phase 4 |
| [tracer_phase5_calibration.log.md](tracer_phase5_calibration.log.md) | `Implemented` (G1 R3 + **G2 R4 APPROVED**；§6 Verify `tests/cache/ tests/exp/` **2035 pass/6 skip** green；机制代码已 commit。运行时 Pass 1/2/3 + 真实 calibrated YAML + report 属后续出场条件，达门前不标 Validated，2026-07-11) | **L3**: TRACER roadmap Phase 5 — **阈值/权重标定 + u_t 激活**（轻标定）。两件事：① **激活 β₂·u_t**（owner 定）——gate 内算 kinematic-quality u_t（复用 online Factor + `ZScoreNormalization` + descriptor kernel；gate 已从 orchestrator 无条件收 view/history），logit 变 `z=β₀+β₁m+β₂u_t+β₃Δ⁺`，放宽 `b2==0` validator；u_t NaN→丢项优雅退化=Phase 3；additive（`u_t_factor=None`+β₂=0+`export_factor_outputs=false` → **verdict+wire 双逐字节退化**）② **离线标定 (λ,τ_hit,τ_warm,β) 于 held-out**（Eq 24–25，min `L_cal=BadHitRate+c_miss·MissRate+c_warm·WarmCost`；真 `SR≥SR_base−ε` 由 Pass3 强制）→ **新** `dual_retrieval_calibrated{,_l10}.yaml`（不覆盖 active）。方法学（owner Q1 选）= **三阶段** + **防泄漏三层分离**（`I_cal`/`I_val` init_idx 二分不相交 + Pass2 LOEO 排除自身 trajectory + fail-loud）：Pass1 cache-OFF **串行**采集（`--collect --replicas 1 --non-concurrent`，`_validate_collect_isolation` 硬约束）取真成败+5-key h5 于 `I_cal` → Pass2 离线 replay（轨迹顺序不 reset history 忠实重建深度5 chain-aware 信号+u_t，orchestrator parity 表）→ solve（新 src 类 `MarginGateCalibrator` 内聚照 fit_from_scores；固定 b1≡1/threshold≡0.5，网格搜 λ/b0/β₂/β₃/τ_warm_g，确定性 tie-break）→ emit 标定 YAML → Pass3 cache-ON eval 于 `I_val`（illustrative-vs-calibrated **paired** 同 init/seed，新 `analyze_phase5_rollout.py` 算 BHR/FFR/IR/SR 分子分母）确认 **BHR↓@SR≥SR_base−ε 且 IR 不升**（出场门）。BadHitRate/analyzer 须新建。src 改 `judge.py`（gate u_t + export-gated factor_outputs）+`config.py`（工厂/validator）+ 新 `margin_gate_calibrator.py`，additive 退化=Phase 3。上位依据 [`tracer_retrieval_refinement_roadmap`](tracer_retrieval_refinement_roadmap.log.md) Phase 5 |
| [tracer_phase6_projection_training.log.md](tracer_phase6_projection_training.log.md) | `Plan` (**G1 APPROVED R8**；§4 Code DONE；**G2 APPROVED R8**；**§6 Verify PASS** 1055 + 259/6；待用户指示提交；§9-D1 owner 裁=(b)；2026-07-13) | **L3**: TRACER roadmap Phase 6 — **M1 结果兼容投影训练 + 框架重测**。动机（已核实）：Phase 5 全程跑的是 raw-feature **消融**（投影是 identity 骨架从未训练）而非方法本身 → 框架未被测试。计划：从 D⁺/D⁻ payload 造动作兼容标签（`c^A` Eq10 / `c^X` Eq11 / `c_ab` Eq12），忠实 masked InfoNCE（Eq15，替换 group-label 代理）训练每模态投影头 → 投影空间重建库 → 重标定 → I_val paired rollout 判 `BHR↓@SR≥SR_base−ε`。多专家 workflow 起草 + 4 对抗审稿挖出 3 blocker（**批次混淆** D⁺全4月/D⁻全7月不可辨识、**even-init 泄漏**、**Pass-3 自检索泄漏**）。§0 以可证伪假设 + 3 区分性结局框定（不改写 Phase 5 的 calibration/SR-mismatch 结论）；§A 冻结标签口径、§B 向后兼容训练 API、§C 统计预注册（bootstrap/N_min/Holm–Bonferroni/Claim-2 等价性）。§9-D1 owner 已裁 = (b)（先补采 batch-matched 控制集，新增 Phase 6.0 前置 + 硬验收门）。上位依据 [`tracer_retrieval_refinement_roadmap`](tracer_retrieval_refinement_roadmap.log.md) Phase 6 |
| [tracer_phase7_negative_result_integration.log.md](tracer_phase7_negative_result_integration.log.md) | `Implemented` (**G1 APPROVED R5** 2026-07-14 / **G2 APPROVED R4** 2026-07-15 / §6 Verify green 2452 passed，2 既有环境失败 owner-adjudicated proceed；待 owner 确认归档) | **L2**: TRACER roadmap Phase 7 — **负结果整合 ablation/评测报告**（纯 `exp/` 写作，零新 GPU、零 src/exp 代码改动，复用已提交 Phase 5/6 证据）。owner 定性转向：不再证 Pareto 赢面，而是把已跑到的负判决整合成有出场门口径、可追溯的报告。交付 `analysis/phase7_tracer_ablation_report.md`（9 节）：**精确定级**判决——**Claim 1（M2，success-only vs dual）= NOT EVALUATED/de-scoped**（无 success-only comparator；已评估两 dual full-hit 操作点 FAIL 出场门 SR 崩 −19.6/−32pp）、**Claim 2（M1，raw vs projection）= reduced 离线 rescue 门 NO_GO**（仅 raw vs action-only 投影 B 的 I_cal AUROC ΔCI 含 0；B-vs-C+downstream 未评估）、**Claim 3（M3 深度）= NOT EVALUATED/owner-de-scoped**（逻辑闭合，不补 rollout）。warm-start 仅列不承重背景（无已提交 cost 产物 + runbook `start_t` 张力）；bottom line 收窄到「已评估配置/协议/操作点内**未观察到**安全且省算力工作点」（观察限定，非非存在性证明）；BHR/FFR proxy 因果上限（`analyze_phase5_rollout.py:6/78/80` episode 级关联非逐步反事实）入框架/风险/局限；provenance 区分**可追溯≠可复现**（原始产物 gitignored 在 ziyang10，绑 commit `2d0e4cc`）。§6 Verify 命令 `uv run pytest --ignore=tests/review_tests`（唯一宪法级排除显式编码守 sealed reviewer space）。G1 R1/R3 各 NEEDS REVISION（Claim 定级/证据边界/proxy 因果/Verify 命令）逐条依 §10.2 收敛，R5 APPROVED。上位依据 [`tracer_retrieval_refinement_roadmap`](tracer_retrieval_refinement_roadmap.log.md) Phase 7 |
| [gate_exploration_roadmap.log.md](gate_exploration_roadmap.log.md) | `Data-Grounded Roadmap` (2026-07-04 创建；**2026-07-05 Stage 1b live 判决修订**：F9–F12 + C10/C11 + §5 重排；前身 cache_gate_design_brainstorm 见 git `437bbc2`) | **L0 研究文档**: gate 探索路线图 — 用 `exp/gate_research` 7-config × 500 ep always-search 真 verdict 数据（182,899 决策步）对 brainstorm 方案谱系逐项判决。核心发现：**上一步 cp1_score 预测本步 MISS AUC 0.973–0.986（免费信号）**；verdict 块状粘滞（P(MISS\|MISS) 0.89–0.93，FH→MISS 直跳 0–2%，FH→WS→MISS 降解路径）；命中段与库轨迹 lockstep（winner 持久 93–98%，Δ+1 75–97%）；MISS 段恢复率 61–84% → 停搜必须配 probe；**B3 复用债务前提被驳回**（hazard 平坦/反向，降格为盲回放安全绳）、B4 停滞/N3 首步印象驳回（AUC≈0.5）；V1 省延迟分档判决（优化小库负 / stock +1.5~7.5ms / 50k 大库 +7~20ms 每步）。新方案：**N1 分数滞回门**（Stage 1 主打，零训练零新计算）+ **N2 追随赢家门**（Stage 2，锁定 winner 盲回放把省量上限从 MISS% 19–38% 翻到命中段 60–80%）。**Stage 1b live 判决（2026-07-05，F9–F12）**：N1 vs baseline SR 保真/提升 + net 全正、离线选点偏差 ≤1.8pp（自证成立）；但同 skip% 及格线 4/4 输 periodic——periodic SR +4.6~+7.8pp 却延迟 net 多数崩（靠推理暴增），揭示 **V2 机制**（skip=SR 干预，均匀注入 > 定向跳 MISS，剂量低档即饱和）；公理增补 **C10（skip% 不是预算轴）/ C11（禁止假设 gate SR 中性）**。重排阶段名单：S1 ✅=N1（1a/1b/1c 全收官）；**S2=V2 机制离线研究（H1 剂量截断/H2 on-manifold 反馈/H3 WS 执行中毒 + 公平 Pareto，0 GPU）；S3=N4 混合门（N1 跳 MISS + 定期注入）live，及格线换同 inf_ratio 轴；S4（押后）=N2（降级）/C1/A3/D1**。原公理 C8/C9 继承 |
| [gate_stage2_v2_mechanism.log.md](gate_stage2_v2_mechanism.log.md) | `Done` (G1/G2 APPROVED 2026-07-05；§4 Code + 产物 `analysis/stage2_*` + roadmap §5 回填完成；待 owner 确认后归档) | **L2**: gate roadmap Stage 2 — V2 增益机制离线研究（纯离线 / 0 GPU）。三线并进：**2a** SR 增益分解 + H1 剂量截断/H2 on-manifold 反馈/H3 WS 执行中毒裁决（复用 `analyze_n1_live` 纯函数，自写 run-length/Δinf 三分解 + 连续校正 chi2 与精确二项 McNemar p）；**2b** 公平 Pareto overlay（8 live + baseline + d1 前沿 7 config 同协议承重层 + RPG spatial-only 挖空参照层，正式核对 F11 "periodic 抬高前沿" + 第四设计轴判决，两图不对称 l10 无 RPG）；**2c** 危险步离线 join（scope 硬锁 libero_spatial，`deviate_score≥5` oracle × prev_score 廉价信号 AUC，跨配置迁移 proxy）。产出 `analysis/` 三份 .md + 两图，回填 roadmap §5。A2 预算重分配 C8 下离线 SR 不可靠 → 移出本阶段降 Stage 4 |
| [gate_stage3_n4_hybrid.log.md](gate_stage3_n4_hybrid.log.md) | `Done` (代码 G1/G2 APPROVED R2 + Verify green `251eddc`；**live 判决 2026-07-05：N4 胜出 L=6，4/6 pass**，报告 `analysis/stage3_n4_live.md` + roadmap §5 回填；待归档) | **L2**: gate roadmap Stage 3a — **N4 混合门 live 原型**（零 src）。N4 = V1（N1 跳预测-MISS，inference-neutral）⊕ V2（连续缓存执行 FULL_HIT run-length ≥ **L** 强制注入新推理，L≈{6,8,12} 源自 2a H1 剂量饱和）。**零 src 命门**：`interceptor.py:500` 回传 `__hit_meta__.hit_type`（FULL_HIT/WS/MISS）客户端可读 → V2 在客户端数 run，驱动现成 `ClientControlledGate`。新增 `n4_gate_client.py`（`N4GateState` 内嵌复用 `N1GateState` V1 子机 + fh_run 计数 + `_last_v2` 冻结避免污染 N1 相位）+ `worker_entry_n4.py`；改 `run_n1_live.py`（`--gate-family n4`/`--L` 分派）+ `analyze_n1_live.py`（N4 旁路 N1 离线重放 + matched-periodic 按 **inf_ratio** 轴配对，D6）。及格线（C10）：同 inf_ratio SR ≥ matched periodic ∧ net@34≥0 ∧ SR≥baseline−1pp，按 C9 三档。6 run（L×2 suite）500ep + 可能补 1–2 periodic 档；live 发射须 owner 确认。3b 定型下游、危险步靶向 2c 否 → 均范围外 |
| [gate_stage3b_n4_serverize.log.md](gate_stage3b_n4_serverize.log.md) | `In Progress` (G1 APPROVED R5 / §4 Code done: gate+config+n4_server yaml+tests+docs / 待 G2 2026-07-06) | **L2**: gate roadmap Stage 3b — **N4 服务器化**（3a 胜出 L=6 下游，**含 src**）。扩展 1c 的 `ScoreHysteresisGate` 加 V2 分支：`__call__` 加 `fh_run≥L→skip` 注入（PURE），`record_verdict` 按 `searched` 重构（fh_run 计数 + V1/V2 靠 `_searching` 状态重构区分、**无需客户端 `_last_v2` 标志**）。**server 侧比 `HitType` enum**（`== HitType.FULL_HIT/WARM_START`，非字符串——否则 fh_run 永不增静默退化纯 N1）。**零 orchestrator 改动**（`record_verdict` 已收 hit_type enum、`_feed_verdict_to_gate` 已喂 hit_type+searched）。`config.py` GateConfig **只加 `L`**（`include_ws` **不进 config**，避免 `bool=False` 触发 stray-field 让 legacy gate 全回归）+ validator + `_build_gate`。**L=None 行为等价 N1**（decide/observe 等价，1c golden 不动）。核心判据：`ScoreHysteresisGate(L=6)` ≡ 3a `N4GateState(L=6)` 等价 golden（喂真实 HitType enum）+ L=None=N1 兼容 + legacy 非回归 + n4_server YAML load 测试。docs 两 profile(延迟档 L 省略=N1-A / SR 档 L:6=N4) + docs/logs README index sync。无 live run（3a 已验）。 |
| [gate_stage4a_n2_follow_winner.log.md](gate_stage4a_n2_follow_winner.log.md) | `DONE` (G1 APPROVED R2 + **G2 APPROVED R2** 2026-07-06 / Phase A F5 复测 **GO**（报告 `exp/gate_research/analysis/stage4a_f5_recheck.md`）/ §4 Code + §6 Verify green / **Phase C live 完成 2026-07-07**：N2 6 点 **1 PASS**（l10 budget=3：SR +1.2 vs base、vs matched periodic +1.2、net@34 +8.2）；spatial 3 点 FAIL（同 inf periodic SR 90.4>>84-85）；l10 b5/b8 FAIL（budget↑ 漂移退化）；net@34 全正=延迟稳健价值。报告 `exp/gate_research/analysis/stage4a_n2_live.md`) | **L3**: gate roadmap Stage 4a — **N2 追随赢家门**（FollowWinnerGate / lockstep 盲回放，**含 src + 新执行原语**）。与 N1/N4 根本差异：skip=免搜仍全推理 vs **盲回放=不搜不推、直接回放锁定 winner 后续缓存动作**（省命中段 60–80% 的 search+judge+推理）。**必服务器端**（exp 层 `ClientControlledGate` 装不下：其 skip 必产推理、无 client 可控 replay 信号）。**L3 依据**：现 gate 契约仅 `__call__→bool` 二态、系统无"不搜不推回放"原语。设计：给 `GateFunction` 加**可选** `replay_target()→str\|None`（additive hook + `hasattr` 守卫，同 G0a record_verdict 惯例，`__call__` 仍 bool 向后兼容）；orchestrator `check()` 在 `not should_search` 时查 `replay_target`，非 None 则走盲回放分支——`view.walk_next(cursor,1)` 取 locked winner 后续 payload → 返回 **`FULL_HIT×searched=False`**（interceptor 短路回放 `payload.action_chunk`，跳 Stage2/3，保留 Stage1+build 契合 build/D2H 押后）。`FollowWinnerGate`：decide/observe 分裂（同 ScoreHysteresisGate），lock 触发靠解析 `winner_id=traj:step` 判同轨+Δ+1、游标+预算 M（B3 安全绳）、probe 掉带解锁。config 5 改（`follow_winner` 类型 + `lock_streak`/`budget` 字段 + `_build_gate` + validator + **backend 须 in_memory** fail-loud，`walk_next` 仅 InMemoryBackend）。**Phase A（gating 前置，0 GPU）**：在 Stage 3a N4 live per_step（每步含 `winner_id`）上离线重测 F5 lockstep，Go/No-Go 挡住昂贵 L3——NO-GO 即停。Phase C live（RPG 同构 (lock_streak,budget)↔(k,n) 网格，及格线同 inf_ratio SR≥matched periodic∧net@34≥0∧SR≥baseline−1pp）在 G2/commit 后。进入条件①（部署延迟档）非代码可验，如实交 G1/owner 判定 |
| [gate_stage4a_n2_live_enablement.log.md](gate_stage4a_n2_live_enablement.log.md) | `DONE` (G1 + **G2 APPROVED** 2026-07-06 / §6 Verify green / N4 byte-regression `cmp -s`=0 / 使能代码就绪；**Phase C 扫点 live 完成 2026-07-07**，见 `stage4a_n2_follow_winner` 行 + `analysis/stage4a_n2_live.md`) | **L2**: Stage 4a **Phase C 上机使能**（N2 已服务器化 `eaa4263` 下游）。缺口：N2 是**服务器端事件驱动 gate**，per-step `searched` 当前无处记录——`__collect_meta__` 有 C5 硬校验（`export_collect_meta` 要求 `always_search`），N1/N4 靠**客户端 stamp** 绕过、periodic 靠**闭式重建**绕过，**N2 两者皆不可**（无 client 决策 + 事件驱动不可重建）。`inf_ratio`/`SR` 不需 searched（`hit_type`+journal），但 **net@34 需 skip%=searched** → roadmap 及格线含 net。**解法**：给常开无 guard 的 `__hit_meta__` 加 `searched`（additive，源 `CheckResult.searched`），runner `collect_meta` 缺失时从 hit_meta 读——任何 server-side gate 从此可记 searched，不碰 C5。含：src `_build_hit_meta`+`episode_runner`；exp `run_n1_live` follow_winner 分派（同 periodic，DEFAULT_WORKER，无 client 信号）+ manifest lock_streak/budget/gate_family=n2；`analyze_n1_live` **修 inf 记账潜伏 bug**（searched=False 硬编码 inf=1.0 → 用 `inf_value`，N1/N4/periodic 数值不变、N2 盲回放 FULL_HIT×searched=False→0）+ `episode_searched` follow_winner 用记录 searched + `analyze()` 复用/泛化 N4 inf-matched 配对与 3 条件 overall。回归锁：Stage 3a N4 数据分析器输出逐值不变。**不含 live run**（使能代码；扫点 live 在此 commit 后另起）。进入条件①延迟档仍 owner 判断 |
| [n1_serverside_gate_stage1c.log.md](n1_serverside_gate_stage1c.log.md) | `Done` (G1/G2 APPROVED / Verify scope-green，committed `dc2815e`；操作点 YAML 化按 roadmap 留待 Stage 3 N4 定型；待 owner 确认归档) | **L2**: gate roadmap Stage 1c — **G0a hook 补丁 + N1 门服务器化**。补齐 server 侧 gate 缺的两条数据通路：**verdict 回传**（`orchestrator.check()` 内经新 `_feed_verdict_to_gate` 把 `cp1_score/hit_type/winner_id/start_t/searched` 回喂本 checkpoint 的门，供下一步决策；4 个 return 前各一次，`hasattr(record_verdict)` 守卫）+ **task_key 广播**（`_broadcast_episode_start` 给门的 `on_episode_start` 传 `task_key`，沿用 `_safe_call_lifecycle` 签名过滤）。新增 src 门 **`ScoreHysteresisGate`**（忠实复刻 1b G2-APPROVED 的 `N1GateState` decide/observe，`__call__` 纯读=decide、`record_verdict`=observe，θ/j/probe_interval 走 config）作为 verdict hook 的真实消费者（Owner 裁定 Scope B，避免 WA §3.1 dead code）。config +4 字段(`theta_low/theta_high/j/probe_interval`)+`_GATE_TYPES`+validator 分支+`_build_gate`。**零 interceptor/wire 改动**、5 个现有门逐字节不变（signature 过滤 + hasattr 守卫）。N2 属 Stage 2（仅在 verdict payload 预留 `winner_id/start_t`）。docs: cache_system.md §5.5 + tutorial gate 段 |
| [n1_live_validation_stage1b.log.md](n1_live_validation_stage1b.log.md) | `Done` (G1 R7 / G2 R7 APPROVED / Verify 797 pass / **live 8 run × 500 ep 完成 2026-07-05**：vs-periodic 及格线 4/4 FAIL、N1 vs baseline SR 保真 + net 全正 → 判决入 roadmap F9–F12/C10/C11，报告 `exp/gate_research/analysis/n1_live_results.md`；待 owner 确认归档) | **L2**: gate roadmap Stage 1b — N1 分数滞回门 **live 验证**。1a 选出的操作点 A(近免费)/B(平衡)在真实闭环 rollout 测**可观测量**(actual SR + skip% + actual inference_ratio + searched-step verdict mix;C8 下 live **无法测 lost%**,已删)。server 端零改(复用 `ClientControlledGate`+`__gate_decision__`+`__hit_meta__.cp1_score`+`__collect_meta__.searched` recorder seam,step3 已落地);**全部新文件落 `exp/gate_research/`,零 src/examples 改动**(G1 更正 agent.py:17 误读)——`n1_gate_client.py`(可导入纯 `N1GateState` + `N1GateClient` wrapper + 四情形异常契约)+ `worker_entry_n1.py`(worker 入口,θ 经 env)+ `run_n1_live.py`(单 config 单 θ + run manifest)+ `analyze_n1_live.py`(去重/完整性/配对/C9/N1-vs-periodic 裁决)+ **2 个 client_controlled yaml 已交付**（periodic yaml deferred 第二波，参数依赖首波 skip%）+ 单测。**L2 依据=正确性关键性**(gate bug→SR 污染→路线图误判),非落位。及格线=SR ≥ always_search 基线 −1pp(配对 Stage-0) **且** 同预算打败 matched-budget periodic(roadmap 强制)。范围 spatial fh75_ws10 + libero_10 fh5_ws40 各 A+B=4×500ep,periodic 第二波至多 +4×500ep(启动前 Owner 再确认) |
| [gate_data_collection_plan.log.md](gate_data_collection_plan.log.md) | `In Progress` (G1 APPROVED / §4 Code / **G2 R1→R10 迭代, R10 Executor applied 待重审**, 2026-07-03) | **L3**: GATE 研究数据采集 — 并发 serving 下用纯 per-connection wrapper(无 forward hook、不改 PI0Pytorch)逐推理步采集三条可 join 数据流(精简模型输入 + verdict-agnostic 判决 hit_type/cp1_score/winner_id/start_t + episode success)。**最终设计=采集字段 inline 进每步行**(`__collect_meta__` sibling key,`export_collect_meta` 默认关→wire byte-identical;客户端 codec ndarray→list;默认 `robot_state`,vision opt-in 仅 standalone,raw prefix_embs 移出 scope);conductor 下 collect 作 `per_step_rows` 额外 key(**不 bump protocol**,跨机无 NFS);新增 `src/openpi/serving/per_step_recorder.py`(`PerStepWriter` 两模式)+ `CheckResult.searched`(防 C5 把冷启动 always-search MISS 误标 skip);success 走客户端/conductor flush 盖章。R1 整合 `factor_outputs`(v2)+ canonical `--collect-gate-dir`(旧 `--per-step-log-dir` deprecated alias)/ R2 老 `--collect`(forward-hook 模型内部)独立共存。G1 5 轮迭代逐项全 Accept 后 APPROVED |
| [weighted_sum_trajectory_weight_alloc.log.md](weighted_sum_trajectory_weight_alloc.log.md) | `In Progress` (G1 APPROVED R3 2026-07-02 / §4 Code 完成 / 待 G2) | **L2**: 对 trajectory **每步权重** `trajectory_weights` 做 screening 搜索（合作者疑 d1>d3/d4/d5 源于历史固定递减方案）。每 depth 复用自己的 base 只搜每步权重形状（always_hit 下仅排名影响 SR）；搜索集 = S1 当前步主导×尾形梯度(含 c=0.9 near-d1 边界) ∪ S2 内部形状格点 ∪ S3 incumbent，实算锁死 **d3=52/d4=60/d5=59=171 config × 100 ep = 17,100 ep**（libero_spatial）。零改 src/：新 `emit_traj_weight_alloc.py`(从 tracked grid3+calibration 确定性重建 base，防 stale 断言 171 ID 集) + `analysis/analyze_stepweight.py`(YAML 读真实权重 + 互斥形状分类 + journal latest-ts 去重后配对 McNemar vs incumbent) + `tests/exp/test_traj_weight_alloc.py`(23 pass)。单 server ziyang10 `--replicas 3` + timan107 `--workers 48`；d1 作非裁决性 prior 参考、确认性重跑列后续 |
| [weighted_sum_trajectory_weight_alloc_libero10.log.md](weighted_sum_trajectory_weight_alloc_libero10.log.md) | `In Progress` (G1 APPROVED R3 2026-07-02 / §4 Code 完成 / 待 G2) | **L2**: 把 trajectory 每步权重 screening（搜索矩阵/配对分析/screening 框架完全复用 libero_spatial 已批准逻辑）搬到 **libero_10**，验证结论是否 dataset-general。计数不变 **52/60/59=171 × 100 ep = 17,100 ep**。核心新面：libero_10 base winner 非干净格点、CID 与 all_results.csv 均有损（`int(w*100)`）→ 新增 **tracked 非有损 `LIBERO10_BASE_MANIFEST`**（d3 0.62/0.37/0、d4 0.25/**0.4375**/0.3125、d5 0.5/0.5/0，值取自实际 base YAML；emitter 只读它、绝不从 CID/csv/grid 反推）+ 3 份 base YAML 入 `tests/exp/fixtures/`（deep-diff=0 锁唯一变量）。emitter/analyze 加 `--task-suite`（per-suite dispatch + 动态默认 None→post-parse + expected_ids 贯通 + provenance 参数化），libero_spatial 逐字节向后兼容（rollup 不变）。零改 src/；单 server ziyang10 + timan107 复用 libero_spatial 拓扑 |
| [history_similarity_markov_sufficiency_discussion.log.md](history_similarity_markov_sufficiency_discussion.log.md) | `Discussion Record` (2026-08-12 会议纪要；§6 检验项未排期) | **L0 研究文档**: trajectory search（history-frame similarity）在两 suite 强配置下无增量的成因讨论纪要。核心假说 = **Markov 充分性三层分解**：①结构层（近必然）—teacher Pi0.5 无记忆 ⇒ 检索目标按构造与历史条件独立 `I(a*;h|o_t)=0`，环境 POMDP 与否次要；②key 层（经验战场）—历史唯一可能贡献 = 补偿 key 有损性，增量 ⟺ `I(a*;h|k_t)>0` ⟺ key 不充分；③环境层—LIBERO 准静态近 obs-MDP。证据落位：rescue-the-weak（弱 base +10/+21pp vs 强 base 负）= key 质量↓⇒历史收益↑的剂量-反应曲线；wsweep 全平 / current-dominant 形状最优 / temporal_prune 窗口负效应均吻合；**gating 有效（prev cp1_score 预测 MISS AUC 0.973–0.986）vs ranking 无效的分界 = 假说双重预测**。残余：libero_10 d3-trough 信号（p=0.070 未裁决，混叠 vs winner's curse 不可分）+ threshold-pareto d3 赢（归因 calibration 收益而非信息收益）。提出 4 项离线检验（动作预测残差估 `I(a*;h|k)` / d1 失败尸检分混叠-覆盖 / 混叠率量化 / oracle 步对齐消融排除 H-B），检验 1、2 顺带裁决 74% 天花板属覆盖还是排序问题。**§9 增补（同日）**：**算子类分析**——trajectory search 两级信息封闭（逐帧标量化销毁 dynamics 特征 + 非负加性组合 `config.py:2067` 封死差分核）⇒ 可配置空间 = 纯低通平滑核，171 形状搜索在与 dynamics-aware 打分交集为空的函数类内，「空集」由算子类预先决定；(a) 目标时间依赖 × (b) 算子表达力双条件框架（本线 ranking 双✗ / gate 双✓）；检验 1 升级 A/B/C 三组特征判别设计（raw 拼接 vs Δk 差分）分离「无信息」vs「算子错」；judge 侧推测路线 = `walk_next` 作免训练世界模型做预测误差不信任信号。触发背景 WCM（arXiv 2607.29613）**经讨论定性为方法不可迁移**（RL 微调管线，无零件可搬），仅留任务族证据（同款 π0.5 backbone + LIBERO 系上单帧 policy 即 SOTA、历史仅在 critic 侧起效）作可引用数据点。**§10 讨论总结（同日定稿）**：cache 继承 teacher 无记忆 ⇒ 系统马尔可夫充分；trajectory search = 拐杖（救弱不助强，强 key 下加的是滞后偏差非噪声）；开放条款 = **被否定的是「加权平均边际相似度」这个方法而非 history 本身**（(a)(b) 分离，由 A/B/C 检验裁决）；实验清单定稿 5 项，含新增**记忆假肢实验**（构造性两分支观测混叠任务、无记忆 teacher≈50%，测 history-retrieval 能否超越 teacher——测的是超越命题而非逼近命题；任务挑选瞄准混叠而非难度）。**§11 致导师摘要**：§10 口径的一段式润色，供 owner 发送教授。**§12 文献定位与论文化评估**（同日 web 调研）：五条相邻线（causal confusion/copycat 训练期机制须划界 / VLA-history 激辩：⭐Present-but-Not-Remembered 2607.03372 = 第一层假说的表征层实证、HAMLET 拼接无增益 / 检索式操控：⭐ActionCache 2607.06370 同源系统必引，「检索 key 含历史」无人研究 / 理论与测量：λ-discrepancy、MVS、POBAX / 记忆 benchmark：MIKASA-Robo 2502.10550 + RoboMME 2603.04639 已在 π0.5 上比较 14 种 memory 变体）；地盘判定 =「模型是马尔可夫的」两端已被占，**未占位 = 继承定律（Markovness is Inherited：无记忆 teacher 钉死一切派生物的历史信息上界）**，本项目 cache = 该定律的干净测量仪器（无训练环节 + 14 万 ep 剂量-反应数据）；论文化差距 = 记忆假肢正面对照（场地用 MIKASA-Robo 免自建）+ A/B/C 落地 + 泛化性（复用 ablation_study 的 SmolVLA/ACT）+ 时效（PbNR 2026-07/RoboMME 2026-03，半年内更挤）。**§13 继承定律形式化 + ICML 评估**：可证但拆四件——引理 1 标签条件独立（一行，平凡）/ **引理 2 去噪上界** `I(a;h|k) ≤ I(a;o|k)`（三行链式法则，rescue-the-weak 的定理化，实验 1-B 组测左端）/ 定理 3 成功率天花板（需 obs-MDP 假设，标准 MDP 论证）/ **命题 4 例外通道：success 过滤 = collider 条件化重注历史信息**（两分支玩具闭式 50%→100% = 记忆假肢的数学原型；本项目库默认 success 过滤 → **§3 第一层绝对表述已修正**，LIBERO≈obs-Markov 故重注≈0 与数据自洽）；两通道分解（去噪 + 过滤）= 历史价值全部来源，与实验 1/5 一一对应；DT/RvS 为概念邻居；范围限制 = 仅模仿型派生、population 级（causal confusion = 有限样本病理叠加）。ICML 分档：仅理论不够 / +现有数据 borderline / +MVP 够格竞争（MVP = A/B/C + 假肢正对照 + 第二 backbone + 划界；双赢结构：C 组降残差则转「正确历史算子」故事仍可发）；至截稿约 5 个月可行 |

| [ablation_study_plan.log.md](ablation_study_plan.log.md) | `In Progress` (G1 APPROVED R3 / **G2 APPROVED R11** 2026-08-11；§6 Verify PASS 2288 passed/9 skipped；待 commit) | **L3**: cache 有效性双方向消融（合作者质疑 hit 内容 vs 路由信号价值）。共享同一 cache 判定链，仅换 hit/miss 槽执行体：**方向 1** `small_at_hit`（hit→小模型现算, miss→Pi0.5，检验 cache 内容是否优于便宜执行器）/ **方向 2** `small_at_miss`（hit→cache 回放, miss→小模型，检验 cache 能力放大器效应，cloud-edge 叙事）。小模型双臂 SmolVLA（主）+ ACT（per-task ×10）；**蒸馏路线** = 差集池全集 500 init/suite 采 Pi0.5 rollout（cache 库 50-init 的同源超集；train/val 45/5 防泄漏切分，pruned_init 冻结后仅一跑），复用 `main.py --save_trajectory` 现成采集 schema。实现：`InferenceInterceptor` 加 `hit_executor`/`miss_executor` 加法式默认惰性 hook + `CacheConfig.routing` yaml 段（bundle 热切换逐臂下发，正向 allowlist 校验）+ `SidecarExecutor`（有界超时/fail-closed/确定性 close）；小模型 sidecar 独立 lerobot venv 进程（localhost msgpack，避 transformers 冲突）。O2 基线已锁 gate_research fh40 d1 家族剥 warm_tiers。评测 8 臂/suite × pruned_init 500 ep，配对 McNemar + TOST(δ=3pp) + Holm–Bonferroni；拓扑 ziyang10(server+sidecar)+timan107(client)。L3 文档义务：cache_system.md external-executor 小节 |

> See [Archive › Verdict Factor Judge](#verdict-factor-judge) for prior phases + refactor history.

### Server Infrastructure

全部条目已于 2026-07-04 按「创建 >7 天归档」指令下移 → 见 Archive 小节 **Serving Infrastructure & Audits**。

---

## Archive

Completed and historical logs. See [`archive/`](archive/) for all files.

### Cache Latency Bench (archived 2026-07-04, >7d)

| File | Status | Description |
|------|--------|-------------|
| [cache_latency_bench_plan.log.md](archive/cache_latency_bench_plan.log.md) | `Implemented` (G1 APPROVED R5 / Code 完成；G2 未走，成果由最终报告收束) | L2: CP1 `check()` 六段延迟回放基准 `exp/cache_latency_bench/` |
| [cache_latency_bench_search_optimization_report.log.md](archive/cache_latency_bench_search_optimization_report.log.md) | `Validated` | 早期研究：cp1_search <10ms/<5ms 3 轮专家研究（pre-调优） |
| [cache_latency_bench_depth_study_plan.log.md](archive/cache_latency_bench_depth_study_plan.log.md) | `Implemented` (G1 APPROVED R2) | L2: depth∈{1,3,4,5} 六段延迟受控扫描 plan |
| [cache_latency_bench_depth_study.log.md](archive/cache_latency_bench_depth_study.log.md) | `Validated` | 六段延迟 ~ depth 扫描（weighted_sum, libero_10）；search 段占 95% |
| [cache_latency_bench_depth_study_rrf_kin.log.md](archive/cache_latency_bench_depth_study_rrf_kin.log.md) | `Validated` | 同扫描换 weighted_rrf + kinematic judge（libero_spatial） |
| [cache_latency_bench_round1_stack_elim.log.md](archive/cache_latency_bench_round1_stack_elim.log.md) | `Validated` | weighted_sum 调优 R1 = 预建矩阵 + 零拷贝（33.9→21ms, bit-equal） |
| [cache_latency_bench_round2_prenorm_dot.log.md](archive/cache_latency_bench_round2_prenorm_dot.log.md) | `Validated` | weighted_sum R2 = prenorm-dot GEMV（21→4.70ms, fp32-ONLY） |
| [cache_latency_bench_round3_lean.log.md](archive/cache_latency_bench_round3_lean.log.md) | `Validated` | weighted_sum R3 = LEAN 框架精简（→3.7ms, bit-equal vs R2） |
| [cache_latency_bench_round4_build.log.md](archive/cache_latency_bench_round4_build.log.md) | `Validated` | weighted_sum R4 = batched avg_pool2d keybuilder（build 1.28→0.45ms） |
| [cache_latency_bench_round5_mem_release.log.md](archive/cache_latency_bench_round5_mem_release.log.md) | `Validated` | weighted_sum R5 = 内存释放 1059.7MB（fail-closed 守卫） |
| [cache_latency_bench_rrf_round1.log.md](archive/cache_latency_bench_rrf_round1.log.md) | `Validated` | weighted_rrf R1 = 嫁接验证（winner-id parity） |
| [cache_latency_bench_rrf_round2.log.md](archive/cache_latency_bench_rrf_round2.log.md) | `Validated` | weighted_rrf R2 = fp32-ONLY 定档（dtype sweep） |
| [cache_latency_bench_rrf_round3.log.md](archive/cache_latency_bench_rrf_round3.log.md) | `Validated` | weighted_rrf R3 = LeanRRF（4.73→3.82ms） |
| [cache_latency_bench_rrf_round4.log.md](archive/cache_latency_bench_rrf_round4.log.md) | `Validated` | weighted_rrf R4 = batched keybuilder 复用 |
| [cache_latency_bench_rrf_round5.log.md](archive/cache_latency_bench_rrf_round5.log.md) | `Validated` | weighted_rrf R5 = RrfReleaseVision（MRO 零代码组合） |
| _(最终报告)_ | — | [`tuning_final_report.md`](../exp/cache_latency_bench/analysis/tuning_final_report.md)（weighted_sum 35.49→4.15ms）+ [`rrf_final_report.md`](../exp/cache_latency_bench/analysis/rrf_final_report.md)（weighted_rrf 32.88→4.88ms） |

### Weighted-Sum Two-Layer Line (archived 2026-07-04, >7d)

| File | Status | Description |
|------|--------|-------------|
| [weighted_sum_two_layer_refactor.log.md](archive/weighted_sum_two_layer_refactor.log.md) | `Validated` (G1/G2 APPROVED 2026-05-25) | L3: 两层重构（Layer-1 可插拔归一化 + Layer-2 加权和）——当前生产检索方案主设计日志；机理研究见 [`fusion_normalization_theory.md`](../exp/weighted_sum/analysis/fusion_normalization_theory.md) |
| [weighted_sum_trajectory_search.log.md](archive/weighted_sum_trajectory_search.log.md) | `Validated` (§6 Verify 1725 pass / 7200 ep) | L2: weighted_sum 最优配置上叠加 trajectory search（18 base × depth{3,4,5,6}） |
| [weighted_sum_trajectory_weight_research.log.md](archive/weighted_sum_trajectory_weight_research.log.md) | `Historical` (superseded by trajectory_weight_alloc) | L2: 权重×depth 加密扫描（~280 yaml），分离 H-3a vs H-机制 |
| [weighted_sum_threshold_pareto.log.md](archive/weighted_sum_threshold_pareto.log.md) | `Validated` | L2: 聚合总分阈值控三档，SR×inference_ratio 帕累托（两 suite 数据+分析已入 `exp/weighted_sum/`） |
| [weighted_sum_kinematic_phase5_replication.log.md](archive/weighted_sum_kinematic_phase5_replication.log.md) | `Validated` (G1 APPROVED R4) | L2: d1-best 检索上复刻 phase5 kinematic 237-cell 扫描（libero_spatial） |
| [weighted_sum_libero10_replication.log.md](archive/weighted_sum_libero10_replication.log.md) | `Validated` (G1 R3 / G2 R2 APPROVED; tests/exp 616 pass) | L2: weighted_sum 4 阶段整体移植 libero_10 |
| [verdict_phase5_libero10_systematic_sweep.log.md](archive/verdict_phase5_libero10_systematic_sweep.log.md) | `Implemented` (G2 R1 NEEDS REVISION → Executor R1 applied；Reviewer R2 未收到即按归档令下移) | L2: phase5 systematic sweep 移植 libero_10 |

### Serving Infrastructure & Audits (archived 2026-07-04, >7d)

| File | Status | Description |
|------|--------|-------------|
| [server_concurrency_resource_audit.log.md](archive/server_concurrency_resource_audit.log.md) | `Validated` (Audit R3 APPROVED；原 owner 决定留顶层作起点，2026-07-04 按 >7d 归档令下移) | L3: server 全模块资源占用画像（估算未实测，推进优化前先做附录 B.3 实测） |
| [serving_optimization_council.log.md](archive/serving_optimization_council.log.md) | `Validated` (9 议题全部决议) | L1: 议事框架 — A0–A8 决议 + C1/C2 硬约束，孵化 2 个 plan |
| [concurrent_serving_optimization_plan.log.md](archive/concurrent_serving_optimization_plan.log.md) | `Validated` (G1 R3 / G2 R4 APPROVED；1441 pass) | L3: 7-module 并发 serving 优化联合 plan（batching/bundle/pool/frozen guard/benchmark） |
| [serving_throughput_problem.md](archive/serving_throughput_problem.md) | `Historical` | L3: 单进程吞吐瓶颈问题陈述 + 外部专家问答；落地 = full-replica scale-out |
| [throughput_util_exploration.log.md](archive/throughput_util_exploration.log.md) | `Historical` (paused 2026-05-24 未续) | L3: jupyter+a100 replica/worker/batch-wait 扫描与瓶颈归因 |
| [concurrent_serving_scaleout.log.md](archive/concurrent_serving_scaleout.log.md) | `Implemented` (§4 完成 + Execution 自审；外部 G2 未走即归档；后续 6a2a2c0 审计覆盖) | L3: `--replicas N` scale-out + sticky router + Phase-7 监控 + autotune |
| [client_conductor_two_layer_refactor.log.md](archive/client_conductor_two_layer_refactor.log.md) | `Validated` (G1/G2 APPROVED / Verify 1646 pass) | L3: conductor 三层重构（episode 级中央队列 + 策略插件），日常在用 |
| [backend_c2_autoguard_decouple.log.md](archive/backend_c2_autoguard_decouple.log.md) | `Validated` (G1/G2 APPROVED / §6 Verify done) | L3: C2 write-frozen 守卫上提 `__init_subclass__` 自动包装 |
| [full_repo_audit_2026-05-26.log.md](archive/full_repo_audit_2026-05-26.log.md) | `Validated` (§6 Verify 1721 pass；3 项 PO 接受风险仍有效：C1 pickle RCE / M11 / M12) | L2/L3: 51e364b→HEAD 三轮 16 路全库审计 + 修复 |
| [full_repo_audit_commit_review_2026-05-27.log.md](archive/full_repo_audit_commit_review_2026-05-27.log.md) | `Validated` (Fix Applied) | G2-style 审查 `6a2a2c0` + 3 blocking 修复（proxy max_size 等） |

### Cache System Implementation

| File | Status | Description |
|------|--------|-------------|
| [tracer_phase1_dynamic_depth.log.md](archive/tracer_phase1_dynamic_depth.log.md) | `Implemented` (G1/G2 APPROVED + §6 Verify green 2026-07-07, commit `cea98b2`) | **L2**: TRACER roadmap Phase 1 — `dynamic_depth_knn` 动态链深 SearchStrategy（per-step 选深度；`ConstantDepthPolicy` 非回归默认 + 免训练 `HeuristicDepthPolicy`；零框架/零 verdict/零训练；constant@max 逐值等价现有固定深度策略）。上位依据 [`tracer_retrieval_refinement_roadmap`](tracer_retrieval_refinement_roadmap.log.md) |
| [step1.log](archive/step1.log) \[[EN](archive/step1.en.log)\] | `Validated` | Stage 1/2/3 public interface design, Pi0 vs Pi0.5 comparison |
| [step2.log](archive/step2.log) \[[EN](archive/step2.en.log)\] | `Validated` | CUDA Event timing system design |
| [step3_data_collection.log](archive/step3_data_collection.log) \[[EN](archive/step3_data_collection.en.log)\] | `Implemented` | Forward-hook HDF5 data collection |
| [step3_cache.log](archive/step3_cache.log) \[[EN](archive/step3_cache.en.log)\] | `Done-High-Risk` | Cache storage layer: VectorStoreBackend, Qdrant chunked RRF |
| [step4_discussion.log.md](archive/step4_discussion.log.md) \[[EN](archive/step4_discussion.en.log.md)\] | `Historical` | Step 4 discussion: stability analysis, design debates |
| [step4_plan.log.md](archive/step4_plan.log.md) \[[EN](archive/step4_plan.en.log.md)\] | `Historical` | Step 4 full plan: original + revised after review |
| [step4_test_plan.log.md](archive/step4_test_plan.log.md) \[[EN](archive/step4_test_plan.en.log.md)\] | `Validated` | Step 4 test plan: 6 files, 45 test cases passed |
| [step4_config_discussion.log.md](archive/step4_config_discussion.log.md) \[[EN](archive/step4_config_discussion.en.log.md)\] | `Validated` | SearchStrategy abstraction, YAML format, decoupling principles |
| [step4_config_plan.log.md](archive/step4_config_plan.log.md) \[[EN](archive/step4_config_plan.en.log.md)\] | `Implemented` | Config dataclass tree + YAML loading + serve_policy.py integration |
| [cache_private_access_plan.log.md](archive/cache_private_access_plan.log.md) | `Implemented` | L2: `CacheStorage.per_connection_facade()` + `CacheOrchestrator.prefill_mode()` context manager; collapses private-attribute reach-through in config.py / interceptor.py, with tests-layer white-box assertions explicitly exempted |
| [trajectory_search_optimization_plan.log.md](archive/trajectory_search_optimization_plan.log.md) | `Implemented` | L3: 优化 InMemoryBackend trajectory search — 单链假设主路径重写 + 跨 step `(search_session_id, query_id)` 双层身份 score memo；per-strategy sid；orchestrator `_broadcast_episode_start` / `_close_current_search_sessions` helper；7 src + 4 测试文件 27 tests + 2 docs + `exp/trajectory_search_benchmark/` P1 benchmark；G1 APPROVED R6；G2 R4 等审时归档 |

### Verdict Factor Judge

整套 verdict factor judge 链路（pre-refactor 5-factor stack 已被 4-layer 17-factor 重构替换；下面 9 份 log 是该重构落地及其前史的完整归档）。

| File | Status | Description |
|------|--------|-------------|
| [verdict_factor_judge_refactor.log.md](archive/verdict_factor_judge_refactor.log.md) | `Implemented` (G1 APPROVED R4 / G2 APPROVED) | L3: 运动学 judge 整体重构 — 4 desc (jerk/direction/dispersion/path_length) × 4 变体 (online/offline × action/state) + topk_action_variance = 17 因子；命名 `<desc>_<source>_<channel>`；4 层正交架构（Normalization → 因子 → Calibration → Composer）每层可插拔互不知道彼此；online 统一 splice `[history[-P:], winner, walk_next(F)]` action 与 state 完全对称；校准数据 **2 选 1 必备 + 启动 fail-fast** (offline LibraryStats/jsonl/pkl + warmup yaml + WarmupPool)，**no cold-start** 彻底废除（cold_start_strategy / force_miss / passthrough / lenient / sentinel / all_nan_fallback / requires_library_stats / record_action 7 项废除）；DumpingJudge 包 4 层外 + 自持独立 dump 因子列表 + 第 1 层 Norm 副本；诊断 `factor_outputs.{raw, calibrated, composer_score, schema_version=2}`；exp/verdict_factor_judge/v2_spec.py 全新 yaml 生成器；老代码（4 个 stub 文件）全删，超出 plan §14.4 import-only 承诺，已落 §6.14.1 deviation note。§11-§17 详细 Plan + B0-B7 落地 + 12+3 项 validator + 19 个测试重写 + 4 份 docs 同步 + factor_postprocess `enrich-existing-pkl` smoke gate；commit `5a51fa7` 已 push origin/Ziyang |
| [verdict_factor_candidates.log.md](archive/verdict_factor_candidates.log.md) | `Design Only` (superseded) | ENGRAM 缓存子系统 Judge / Gate 阶段的统计 / 运动学因子候选目录 — 列出 F1a-A / F1a-T / F1b-A / F1b-T / F2 五个候选因子的定义、数据源、计算时机（offline / online / hybrid）、风险与组合策略；服务于 verdict 实验规划（已被 17-factor 重构取代） |
| [verdict_factor_judge.log.md](archive/verdict_factor_judge.log.md) | `Implemented` (superseded) | L3: 把 verdict_factor_candidates 中的 5 因子落到 cache 子系统 Judge 阶段 — `factors/` 模块 + Protocol/registry/composers/normalizers + `payload_view.py` (PayloadView/StoragePayloadView) + `CompositeJudge` 骨架 + `CachePayload.factors` schema + facade-only fetch + warm_start CP1-only / canonical timestep / pairwise / tier ordering 校验 + cold-start all-NaN sentinel 短路 MISS；docs §5.6/§5.7/§5.11/§5.12；G1 APPROVED at R13；本路线已被 4-layer 重构整体替换 |
| [verdict_factor_judge_b1_b2.log.md](archive/verdict_factor_judge_b1_b2.log.md) | `Implemented` (superseded) | L3: B1+B2 合并实施 — 填实 F1a-A / F1a-T / F2 在线 extract、F1b-A / F1b-T 在线读取 + 离线 OfflineWriter、`LibraryStats.compute_from_entries`、3 Composer + PercentileRollingNormalizer 算法；Orchestrator B1 接线（view+history 注入 / `_state_history` / anchor CP / on_task_end leak fix / winner fetch rewire）；config 解禁 `composite` + 7 项校验激活；Backend `load_artifact` + fallback `library_stats`；新建 `factors/_descriptor_kernel.py` + `exp/common/factor_postprocess.py`；三 build 脚本接 `--factors-yaml` CLI |
| [verdict_factor_judge_dedicated_runner_plan.log.md](archive/verdict_factor_judge_dedicated_runner_plan.log.md) | `Implemented` (G1 APPROVED R5 / G2 APPROVED R1) | L3: B1+B2 — observability via `__hit_meta__` + warmup preload。**B1**：`InferenceInterceptor.infer()` FULL_HIT 早返 + 末返合并 WARM_START/MISS/no-orch attach `result["__hit_meta__"] = {hit_type, start_t, winner_id, cp1_score}`；client SDK 0 改动；`examples/libero/main.py` `--per-step-log-dir/--yaml-id/--phase` + worker line-buffered temp file + exit merge by `(task_id, subset_init_state_idx, episode_id, step_idx)`；docs §5.13。**B2**：每 yaml emit sibling `<stem>__warmup.yaml` (`AlwaysWarmStartJudge + DumpingJudge`, W=2 trial/task)；`run_phase.py` 7 步 orchestration；server-owned `--warmup-dump-root` mode 0o700 + uid/mode self-check；`fetch_dump` 双 .resolve() allowlist (拒 traversal + symlink escape)；`unload_warmup_buffer` server 派生 warmup name；`CurrentCacheBundle.yaml_id` + `WarmupPool[eval_yaml_id]` LRU/thread-safe/deep-copy + `PercentileRollingNormalizer.preload_buffer`；~1370 行 8 步实施 + 12 测试 file；G2 验 85 新增 + 159 回归全 PASS |
| [verdict_factor_judge_experiment_plan.log.md](archive/verdict_factor_judge_experiment_plan.log.md) | `Plan` (G1 APPROVED, superseded) | L2: verdict factor judge 上线实验 7 阶段 plan（Phase 0 baseline 复用 + calibration dump → Phase 1 单因子 → Phase 2 因子组合 ×2 tier → Phase 3 composer 类型 → Phase 4 窗口/normalizer/tier/描述子启用 → Phase 5 S-CALIB → Phase 7 cross-task）；KeyBuilder + Search Strategy 锁定 3 套 warm_start 同款；artifact `libero_spatial/{clip_vit_b_32,cp1_max_pool,cp1_spatial_pool_16}.pkl`；非 ThresholdJudge baseline；`inference_time_saved_ratio` 公式含 0.5 系数；F1b 长窗口 entry 链两端 NaN 兜底；W-MIX `(0,3)(1,1)(3,0)(0,5)(5,0)`；100 ep/run；paired McNemar p<0.10 + Wilson 95% CI 不重叠；117 run / 11,700 ep；DumpingJudge 透明包装 + `JudgeConfig.dump:{path,config_id,factors}` schema + `episode_start.extra_metadata` 5 wrapper 通道（已被 4-layer 重构 v2_spec.py 替换） |
| [verdict_factor_judge_phase0_phase1_run_commands.log.md](archive/verdict_factor_judge_phase0_phase1_run_commands.log.md) | `Plan` (superseded) | L1: Phase 0 (3 yaml × 100 ep AlwaysHit + DumpingJudge calibration) + Phase 1 (24 yaml 单因子 ablation × 100 ep) 执行命令清单；3 GPU server × 1 client/server；config 目录按 phase 分子目录；Phase 1 yaml 由 `phase1_spec.py` 笛卡尔生成 24 份；frp 端口 8998/8999/9000；含并发 / bundle global race / dump.config_id / `search_strategy.top_k=5` 必填提醒 |
| [verdict_factor_judge_phase2_run_commands.log.md](archive/verdict_factor_judge_phase2_run_commands.log.md) | `Plan` (superseded) | L1: Phase 2 Layer 1 — 6-server 执行命令清单（26 yaml × 3 cfg × 100 ep 单因子内部 desc/window 探索） |
| [verdict_factor_judge_phase2_layer2_run_commands.log.md](archive/verdict_factor_judge_phase2_layer2_run_commands.log.md) | `Plan` (superseded) | L1: Phase 2 Layer 2 redesign — 6-server / 6-client 运行教程（spatial16 only，240-cell threshold sweep） |


### Verdict Factor Judge — Phase 3/4/5 Sweeps (archived 2026-05-26, >5d)

| File | Status | Description |
|------|--------|-------------|
| [verdict_phase5_systematic_sweep.log.md](archive/verdict_phase5_systematic_sweep.log.md) | `Validated` (G1/G2 APPROVED 2026-05-09 / §6 Verify 334/334) | L2: Phase 5 online factor systematic sweep (5 group × 48 cell × 100ep, 6-server 4:4:4:3:3:3). Superseded by `verdict_phase5_libero10_systematic_sweep.log.md`. |
| [verdict_phase5_run_commands.log.md](archive/verdict_phase5_run_commands.log.md) | `Historical` | L1: Phase 5 systematic sweep 6-server run-command book (240 cell). |
| [verdict_phase4_weight_sweep.log.md](archive/verdict_phase4_weight_sweep.log.md) | `Implemented` (G1 APPROVED R3 / G2 进行中 at archive) | L2: Phase 4 weight sweep; src changes landed (`WeightedSumZeroNanComposer` real weighted sum + `reconstruct_scores(composer_weights=)`). Superseded by phase5. |
| [verdict_phase4_stage1_run_commands.log.md](archive/verdict_phase4_stage1_run_commands.log.md) | `Historical` | L1: Phase 4 Stage 1 (R1 α sweep) 6-server run-command book. |
| [verdict_phase4_stage2_run_commands.log.md](archive/verdict_phase4_stage2_run_commands.log.md) | `Historical` | L1: Phase 4 Stage 2 (R2 offline 4-desc weights) 6-server run-command book. |
| [verdict_phase4_stage5_run_commands.log.md](archive/verdict_phase4_stage5_run_commands.log.md) | `Historical` | L1: Phase 4 Stage 5 (48-cell × 500ep true-value re-test) 6-server run-command book. |
| [verdict_phase3_threshold_sweep.log.md](archive/verdict_phase3_threshold_sweep.log.md) | `Validated` (G1 APPROVED R5 / G2 APPROVED R2 — 2026-05-07) | L2: Phase 3 data-driven threshold sweep on 11 spatial16 gold-circle recipes; new `weighted_sum_zero_nan` composer + offline threshold solver. |
| [verdict_phase3_run_commands.log.md](archive/verdict_phase3_run_commands.log.md) | `Historical` | L1: Phase 3 threshold sweep 6-server run-command book (11 recipe × 16 cell). |
| [phase1_dead_loop_diagnosis.letter.md](archive/phase1_dead_loop_diagnosis.letter.md) | `Historical` (open-question letter, 2026-04-27) | Cold-start dead-loop diagnosis letter (Phase 1 single-factor ablation); superseded by the no-cold-start 17-factor refactor. |

### Cache Experiment / CP1

| File | Status | Description |
|------|--------|-------------|
| [cache_experiment_plan.log.md](archive/cache_experiment_plan.log.md) \[[EN](archive/cache_experiment_plan.en.log.md)\] | `Implemented` | CP1 experiment: 5 reducers (incl. CLIP) x RRF fusion |
| [cache_cp1_impl_plan.log.md](archive/cache_cp1_impl_plan.log.md) \[[EN](archive/cache_cp1_impl_plan.en.log.md)\] | `Implemented` | CP1 in-memory implementation plan for large-scale experiment |
| [cp1_warm_start_impl_plan.log.md](archive/cp1_warm_start_impl_plan.log.md) | `Validated` | CP1 warm start implementation plan: 4 phases (performance fix → write → judge + execute → docs) |
| [cp1_warm_start_investigation.log.md](archive/cp1_warm_start_investigation.log.md) | `Historical` | CP1 warm start feasibility investigation; output folded into the impl plan |
| [warm_start_sweep_plan.log.md](archive/warm_start_sweep_plan.log.md) | `Implemented` | Warm start success-rate sweep: 3 keybuilders × 3 start_t (0.7/0.5/0.3) + always_skip/always_hit controls; adds AlwaysWarmStartJudge |

### Retrieval System

| File | Status | Description |
|------|--------|-------------|
| [qdrant_design.log](archive/qdrant_design.log) | `Implemented` | Qdrant collection schema: named vectors vs multivector, payload structure |
| [qdrant_step_knn_experiment_plan.log](archive/qdrant_step_knn_experiment_plan.log) | `Implemented` | Step-KNN retrieval experiment: candidate generation, scoring, evaluation |
| [faiss_uv_toolchain_plan.log](archive/faiss_uv_toolchain_plan.log) | `Historical` | GPU Faiss build commands for uv environment |

### Feature Implementation

| File | Status | Description |
|------|--------|-------------|
| [trajectory_search_requirements.log.md](archive/trajectory_search_requirements.log.md) \[[EN](archive/trajectory_search_requirements.en.log.md)\] | `Implemented` | Trajectory search: linked list, history buffer, similarity fusion |
| [trajectory_search_impl_plan.log.md](archive/trajectory_search_impl_plan.log.md) \[[EN](archive/trajectory_search_impl_plan.en.log.md)\] | `Implemented` | Trajectory search: 7-phase rollout, code details |
| [clip_key_builder_plan.log.md](archive/clip_key_builder_plan.log.md) \[[EN](archive/clip_key_builder_plan.en.log.md)\] | `Implemented` | CLIP KeyBuilder: open_clip ViT-B-32 for cache keys |
| [cache_migration_guide_plan.log.md](archive/cache_migration_guide_plan.log.md) | `Implemented` | Cache framework migration tutorial plan: coupling analysis, 7-step guide, review |
| [concurrent_inference_plan.log.md](archive/concurrent_inference_plan.log.md) | `Implemented` | Server multi-connection + client multi-worker thread pool |
| [redundant_token_prune_plan.log.md](archive/redundant_token_prune_plan.log.md) | `Implemented` | Plan A redundant-token pruning: two-stage KeyBuilder via temporal scoring; includes G2 review record |
| [raw_image_collection_plan.log.md](archive/raw_image_collection_plan.log.md) \[[EN](archive/raw_image_collection_plan.en.log.md)\] | `Implemented` | Two-system (--collect + Cache Sidecar) raw image saving plan |
| [exp_reorg_plan.log.md](archive/exp_reorg_plan.log.md) | `Implemented` | Reorganize `exp/` directory by experiment: 4 experiment subpackages + `common/` shared package; G2 review approved and merged |
| [experiment_artifact_layout_plan.log.md](archive/experiment_artifact_layout_plan.log.md) | `Implemented` | Repo-wide audit of experiment scripts / configs / artifacts / data + unified layout (`exp/<exp>/{config,data,analysis}/`); 8 phases / 51 steps; Phases 0–8 executed; canonical rules live in [`docs/experiments/artifact_layout.md`](../docs/experiments/artifact_layout.md) |

### Trajectory Deviation

| File | Status | Description |
|------|--------|-------------|
| [trajectory_deviation_experiment_plan.log.md](archive/trajectory_deviation_experiment_plan.log.md) | `Historical` | 顶层 3-phase 纠偏方案 (offline diagnosis → signal analysis → Oracle correction)；由 step3_redesign 取代 |
| [trajectory_deviation_corrective_experiment.log.md](archive/trajectory_deviation_corrective_experiment.log.md) | `Historical` | 旧 Step 3 方案 (GT teleport + prefill + pure-cache rollout)；被 step3_redesign §1.2 明确废弃 |
| [trajectory_deviation_corrective_implementation.log.md](archive/trajectory_deviation_corrective_implementation.log.md) | `Implemented` | 代码级 implementation plan：每处改动锚点到文件 + 行号；落地后由 cleanup_plan 收尾 |
| [trajectory_deviation_corrective_implementation_review.log.md](archive/trajectory_deviation_corrective_implementation_review.log.md) | `Implemented` | G1 审查记录：Layer A+D+E / B / C / F APPROVED；审查意见已在实现中修正 |
| [trajectory_deviation_step2_parallel_commands.log.md](archive/trajectory_deviation_step2_parallel_commands.log.md) | `Historical` | Step 2 三服务器 / 三客户端并行 deviate-score 计算命令 |
| [trajectory_deviation_step3_redesign.log.md](archive/trajectory_deviation_step3_redesign.log.md) | `Validated` | Step 3 重设计：per-cycle policy selection，按预计算 deviate flag 在真实 env 中测纠偏效果 |
| [trajectory_deviation_corrective_cleanup_plan.log.md](archive/trajectory_deviation_corrective_cleanup_plan.log.md) | `Validated` | L2 post-hoc cleanup: three classes of compromise landed as 10 commits across three waves (squashed into 633acd8); Verify V1/V2/V3 all green |

### Design Only / Background Analysis

| File | Status | Description |
|------|--------|-------------|
| [key_dim_reduction_recommendations.log.md](archive/key_dim_reduction_recommendations.log.md) \[[EN](archive/key_dim_reduction_recommendations.en.log.md)\] | `Historical` | Two-layer pipeline (token pooling + dim projection) recommendations; did not enter the implementation path |
| [libero_env_init_analysis.log.md](archive/libero_env_init_analysis.log.md) \[[EN](archive/libero_env_init_analysis.en.log.md)\] | `Historical` | LIBERO env init analysis: main.py only uses 3 params; initial state comes from a pre-stored fixed set |
| [redundant_token_prune_gpt.log.md](archive/redundant_token_prune_gpt.log.md) | `Historical` | GPT-drafted preliminary proposal discussion; no project-specific implementation followed |

### Stage Device Placement

| File | Status | Description |
|------|--------|-------------|
| [stage_device_placement_plan.log.md](archive/stage_device_placement_plan.log.md) | `Historical` | L3: split device placement by Stage, supporting cuda/cpu/meta modes; G1 approved, shelved |

### Pi0.5 High-Level Autoregressive Decode

| File | Status | Description |
|------|--------|-------------|
| [pi05_hl_ar_decode_plan.log.md](archive/pi05_hl_ar_decode_plan.log.md) | `Historical` | L2: optional HL autoregressive decode (`lm_head` + incremental KV) on the inference path; Phase A probe gate not pursued |

### LLM Layer Extract KeyBuilder

| File | Status | Description |
|------|--------|-------------|
| [cp1_llm_layer_extract_key_builder_plan.log.md](archive/cp1_llm_layer_extract_key_builder_plan.log.md) | `Historical` | L2: `cp1_llm_layer_extract` KeyBuilder — KeyBuilder 内部独立跑 PaliGemma 第 N 层 forward；两步可插拔架构 (`LLMLayerExtractor` + `PrefixReducer`)；shelved |

### Data Artifact Build

| File | Status | Description |
|------|--------|-------------|
| [libero_10_cache_artifact_build_plan.log.md](archive/libero_10_cache_artifact_build_plan.log.md) | `Historical` | L1: 用 `exp/common/data/db_init/libero_cache/libero_10` 采样 init 驱动 LIBERO 推理 build 6 份 InMemoryBackend pkl artifact (4 pool + ViT-B-32 + ViT-L-14)；shelved |
| [libero_spatial_factor_artifact_rebuild.log.md](archive/libero_spatial_factor_artifact_rebuild.log.md) | `Implemented` | L1: 用 `--factors-yaml` CLI 重建 `libero_spatial/` 6 份 pkl，每 entry 168 keys (F1b-A + F1b-T × 4 描述子 × 21 窗口)；smoke + 6/6 acceptance pass |

### Phase1 Experiments

| File | Status | Description |
|------|--------|-------------|
| [phase1_libero_10_run_commands.log.md](archive/phase1_libero_10_run_commands.log.md) | `Historical` | L1: `exp/common/config/phase1/libero_10/batch{1,2,3}/` 共 60 个 run 的执行命令清单 |
| [phase1_libero_spatial_llm_run_commands.log.md](archive/phase1_libero_spatial_llm_run_commands.log.md) | `Historical` | L1: `exp/common/config/phase1/libero_spatial_llm/batch{1..6}/` 共 196 个 run 的执行命令清单 (5 LLM reducer × 4 extract_layer × 12 weight sweep) |

### Random & Periodic Gate Sweep

| File | Status | Description |
|------|--------|-------------|
| [random_periodic_gate_plan.log.md](archive/random_periodic_gate_plan.log.md) | `Implemented` | L2: 独立 gate baseline 实验 — `RandomGate(p_inference,seed)` + `PeriodicGate(cache_len,inference_len)`；3 套 keybuilder 权重 + AlwaysHitJudge，libero_spatial 全量 500 ep 扫参；G1 / G2 均 APPROVED |
| [random_periodic_gate_run_commands.log.md](archive/random_periodic_gate_run_commands.log.md) | `Historical` | L1: `exp/random_periodic_gate/config/batch{1..3}/` 共 114 个 YAML 的执行命令清单 |

### Trajectory Experiments (libero_10)

| File | Status | Description |
|------|--------|-------------|
| [trajectory_libero10_split_plan.log.md](archive/trajectory_libero10_split_plan.log.md) | `Historical` | L1: 按子实验 (libero_spatial / libero_10) 重组 `config/trajectory`、`data/{phase1,trajectory}`、`analysis/{phase1,trajectory}`，抽公共 `plot_common.py`；shelved |
| [trajectory_libero_10_run_commands.log.md](archive/trajectory_libero_10_run_commands.log.md) | `Historical` | L1: `exp/common/config/trajectory/libero_10/batch{1,2,3}/` 共 60 个 run 的执行命令清单 (d=4/5/6) |

### Historical

| File | Status | Description |
|------|--------|-------------|
| [doc_cleanup_plan.log](archive/doc_cleanup_plan.log) \[[EN](archive/doc_cleanup_plan.en.log)\] | `Historical` | Documentation cleanup plan |

---

## Maintenance Rules

> **AGENT: READ FIRST** — Log status system and lifecycle rules are defined in [`WORKING_AGREEMENT.md` §5 Log Management](../WORKING_AGREEMENT.md#5-log-management). The Working Agreement is authoritative.
