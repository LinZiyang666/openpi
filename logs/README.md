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
| [gate_exploration_roadmap.log.md](gate_exploration_roadmap.log.md) | `Data-Grounded Roadmap` (2026-07-04；前身 cache_gate_design_brainstorm 见 git `437bbc2`) | **L0 研究文档**: gate 探索路线图 — 用 `exp/gate_research` 7-config × 500 ep always-search 真 verdict 数据（185,899 决策步）对 brainstorm 方案谱系逐项判决。核心发现：**上一步 cp1_score 预测本步 MISS AUC 0.973–0.986（免费信号）**；verdict 块状粘滞（P(MISS\|MISS) 0.89–0.93，FH→MISS 直跳 0–2%，FH→WS→MISS 降解路径）；命中段与库轨迹 lockstep（winner 持久 93–98%，Δ+1 75–97%）；MISS 段恢复率 61–84% → 停搜必须配 probe；**B3 复用债务前提被驳回**（hazard 平坦/反向，降格为盲回放安全绳）、B4 停滞/N3 首步印象驳回（AUC≈0.5）；V1 省延迟分档判决（优化小库负 / stock +1.5~7.5ms / 50k 大库 +7~20ms 每步）。新方案：**N1 分数滞回门**（Stage 1 主打，零训练零新计算）+ **N2 追随赢家门**（Stage 2，锁定 winner 盲回放把省量上限从 MISS% 19–38% 翻到命中段 60–80%）。阶段名单：S1=N1 离线前沿+live+G0a 捎带；S2=N2 live+A2 预算重分配；S3（押后）=C1/A3/D1（D1 大概率不立项：免费信号 0.98 > 输入侧学习上限 0.86）。新增公理 C8 反事实口径 / C9 货币分档；及格线=同预算打败 periodic |
| [n1_live_validation_stage1b.log.md](n1_live_validation_stage1b.log.md) | `In Progress` (G1 R7 APPROVED / §4 Code 完成(65 tests, ruff clean) / G2 R1→R5 NEEDS REVISION→R6 applied 待重审, 2026-07-04) | **L2**: gate roadmap Stage 1b — N1 分数滞回门 **live 验证**。1a 选出的操作点 A(近免费)/B(平衡)在真实闭环 rollout 测**可观测量**(actual SR + skip% + actual inference_ratio + searched-step verdict mix;C8 下 live **无法测 lost%**,已删)。server 端零改(复用 `ClientControlledGate`+`__gate_decision__`+`__hit_meta__.cp1_score`+`__collect_meta__.searched` recorder seam,step3 已落地);**全部新文件落 `exp/gate_research/`,零 src/examples 改动**(G1 更正 agent.py:17 误读)——`n1_gate_client.py`(可导入纯 `N1GateState` + `N1GateClient` wrapper + 四情形异常契约)+ `worker_entry_n1.py`(worker 入口,θ 经 env)+ `run_n1_live.py`(单 config 单 θ + run manifest)+ `analyze_n1_live.py`(去重/完整性/配对/C9/N1-vs-periodic 裁决)+ **2 个 client_controlled yaml 已交付**（periodic yaml deferred 第二波，参数依赖首波 skip%）+ 单测。**L2 依据=正确性关键性**(gate bug→SR 污染→路线图误判),非落位。及格线=SR ≥ always_search 基线 −1pp(配对 Stage-0) **且** 同预算打败 matched-budget periodic(roadmap 强制)。范围 spatial fh75_ws10 + libero_10 fh5_ws40 各 A+B=4×500ep,periodic 第二波至多 +4×500ep(启动前 Owner 再确认) |
| [gate_data_collection_plan.log.md](gate_data_collection_plan.log.md) | `In Progress` (G1 APPROVED / §4 Code / **G2 R1→R10 迭代, R10 Executor applied 待重审**, 2026-07-03) | **L3**: GATE 研究数据采集 — 并发 serving 下用纯 per-connection wrapper(无 forward hook、不改 PI0Pytorch)逐推理步采集三条可 join 数据流(精简模型输入 + verdict-agnostic 判决 hit_type/cp1_score/winner_id/start_t + episode success)。**最终设计=采集字段 inline 进每步行**(`__collect_meta__` sibling key,`export_collect_meta` 默认关→wire byte-identical;客户端 codec ndarray→list;默认 `robot_state`,vision opt-in 仅 standalone,raw prefix_embs 移出 scope);conductor 下 collect 作 `per_step_rows` 额外 key(**不 bump protocol**,跨机无 NFS);新增 `src/openpi/serving/per_step_recorder.py`(`PerStepWriter` 两模式)+ `CheckResult.searched`(防 C5 把冷启动 always-search MISS 误标 skip);success 走客户端/conductor flush 盖章。R1 整合 `factor_outputs`(v2)+ canonical `--collect-gate-dir`(旧 `--per-step-log-dir` deprecated alias)/ R2 老 `--collect`(forward-hook 模型内部)独立共存。G1 5 轮迭代逐项全 Accept 后 APPROVED |
| [weighted_sum_trajectory_weight_alloc.log.md](weighted_sum_trajectory_weight_alloc.log.md) | `In Progress` (G1 APPROVED R3 2026-07-02 / §4 Code 完成 / 待 G2) | **L2**: 对 trajectory **每步权重** `trajectory_weights` 做 screening 搜索（合作者疑 d1>d3/d4/d5 源于历史固定递减方案）。每 depth 复用自己的 base 只搜每步权重形状（always_hit 下仅排名影响 SR）；搜索集 = S1 当前步主导×尾形梯度(含 c=0.9 near-d1 边界) ∪ S2 内部形状格点 ∪ S3 incumbent，实算锁死 **d3=52/d4=60/d5=59=171 config × 100 ep = 17,100 ep**（libero_spatial）。零改 src/：新 `emit_traj_weight_alloc.py`(从 tracked grid3+calibration 确定性重建 base，防 stale 断言 171 ID 集) + `analysis/analyze_stepweight.py`(YAML 读真实权重 + 互斥形状分类 + journal latest-ts 去重后配对 McNemar vs incumbent) + `tests/exp/test_traj_weight_alloc.py`(23 pass)。单 server ziyang10 `--replicas 3` + timan107 `--workers 48`；d1 作非裁决性 prior 参考、确认性重跑列后续 |
| [weighted_sum_trajectory_weight_alloc_libero10.log.md](weighted_sum_trajectory_weight_alloc_libero10.log.md) | `In Progress` (G1 APPROVED R3 2026-07-02 / §4 Code 完成 / 待 G2) | **L2**: 把 trajectory 每步权重 screening（搜索矩阵/配对分析/screening 框架完全复用 libero_spatial 已批准逻辑）搬到 **libero_10**，验证结论是否 dataset-general。计数不变 **52/60/59=171 × 100 ep = 17,100 ep**。核心新面：libero_10 base winner 非干净格点、CID 与 all_results.csv 均有损（`int(w*100)`）→ 新增 **tracked 非有损 `LIBERO10_BASE_MANIFEST`**（d3 0.62/0.37/0、d4 0.25/**0.4375**/0.3125、d5 0.5/0.5/0，值取自实际 base YAML；emitter 只读它、绝不从 CID/csv/grid 反推）+ 3 份 base YAML 入 `tests/exp/fixtures/`（deep-diff=0 锁唯一变量）。emitter/analyze 加 `--task-suite`（per-suite dispatch + 动态默认 None→post-parse + expected_ids 贯通 + provenance 参数化），libero_spatial 逐字节向后兼容（rollup 不变）。零改 src/；单 server ziyang10 + timan107 复用 libero_spatial 拓扑 |

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
