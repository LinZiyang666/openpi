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
| [cache_latency_bench_plan.log.md](cache_latency_bench_plan.log.md) | `In Progress` (G1 APPROVED R5 / §4 Code 完成 / 待 G2, 2026-05-30) | **L2**: 把 cache 系统 CP1 `check()` 六段(collect/gate/build/search/judge/fetch)延迟从推理栈抽出做**轻量回放基准** — 按真实 cache yaml 组装(`load_cache_config`/`build_cache_components`/`CacheOrchestrator`，与真 server 同源)、用 H5 真实 trajectory 逐 step 回放驱动 cache "真实工作"(含 trajectory history/score-memo/verdict 窗口等跨 step 记忆，生命周期 1:1 镜像 `InferenceInterceptor`)、`SystemTimer` 探针记录每请求每部件延迟。**不加载模型、零改 src/**，新建 `exp/cache_latency_bench/`。决策：CPU 最轻量(build 段不含 D2H = 已知偏差)/库冻结只读(=真 server runtime `_enforce_runtime_write_policy` never)/精简 per-step CSV+聚合 json。复用 `_build_fake_stage1`。 |
| [cache_latency_bench_depth_study_plan.log.md](cache_latency_bench_depth_study_plan.log.md) | `Plan` (G1 APPROVED R2 / §4 Code 完成 / 待 G2, 2026-05-31) | **L2**: 复用 `exp/cache_latency_bench/` 基础设施做 CP1 六段延迟 ~ `trajectory_depth` 受控扫描 — 唯一自变量 depth∈{1,3,4,5}，固定 3-key(vision_0/1+robot_state)+threshold+相似度+归一化+libero_10 库+50ep H5+repeats1+线程数；4 个派生 yaml 仅差 `trajectory_depth`/`trajectory_weights` 两键。主假设 H1：仅 search 段随 depth 升、余五段平(可证伪)。亲验三陷阱：T1 depth=1 与 depth≥2 走 backend 两条函数(`search_strategy.py:155`)、T2 episode 内 depth 爬升(`:158 actual_depth=min`)、T3 score-memo 复用非严格线性。零改 src/零改 infra，新增 4 yaml + `compare_depth.py` + 测试 + 研究 log |
| [cache_latency_bench_search_optimization_report.log.md](cache_latency_bench_search_optimization_report.log.md) | `Complete` | **早期研究**：cp1_search <10ms/<5ms 的 3 轮专家研究 — fp32 双 GEMV CPU fast path 方案探索（pre-调优） |
| [cache_latency_bench_depth_study.log.md](cache_latency_bench_depth_study.log.md) | `Complete` | CP1 六段延迟 ~ trajectory_depth(1/3/4/5) 受控扫描（weighted_sum，libero_10）；search 段占 95% |
| [cache_latency_bench_depth_study_rrf_kin.log.md](cache_latency_bench_depth_study_rrf_kin.log.md) | `Complete` | 同扫描换 weighted_rrf + kinematic judge（libero_spatial）；出现 d1→trajectory 台阶 |
| [cache_latency_bench_round1_stack_elim.log.md](cache_latency_bench_round1_stack_elim.log.md) | `Complete` | **weighted_sum 调优 R1** = 预建矩阵 + 零拷贝（33.9→21ms，bit-equal） |
| [cache_latency_bench_round2_prenorm_dot.log.md](cache_latency_bench_round2_prenorm_dot.log.md) | `Complete` | **weighted_sum R2** = prenorm-dot GEMV（21→**4.70ms** 达标，几何安全证明，fp32-ONLY） |
| [cache_latency_bench_round3_lean.log.md](cache_latency_bench_round3_lean.log.md) | `Complete` | **weighted_sum R3** = LEAN 框架精简（search→3.7ms，bit-equal vs R2） |
| [cache_latency_bench_round4_build.log.md](cache_latency_bench_round4_build.log.md) | `Complete` | **weighted_sum R4** = batched avg_pool2d keybuilder（build 1.28→0.45ms，query_key bit-equal） |
| [cache_latency_bench_round5_mem_release.log.md](cache_latency_bench_round5_mem_release.log.md) | `Complete` | **weighted_sum R5** = 内存释放 1059.7MB（哨兵 empty(0) + fail-closed 守卫） |
| [cache_latency_bench_rrf_round1.log.md](cache_latency_bench_rrf_round1.log.md) | `Complete` | **weighted_rrf R1** = 嫁接验证（复用 R1/R2 自动生效，winner-id parity） |
| [cache_latency_bench_rrf_round2.log.md](cache_latency_bench_rrf_round2.log.md) | `Complete` | **weighted_rrf R2** = fp32-ONLY 定档（dtype sweep RRF 翻档 4-12× 于 sum） |
| [cache_latency_bench_rrf_round3.log.md](cache_latency_bench_rrf_round3.log.md) | `Complete` | **weighted_rrf R3** = LeanRRF（唯一真 rrf-specific，4.73→3.82ms） |
| [cache_latency_bench_rrf_round4.log.md](cache_latency_bench_rrf_round4.log.md) | `Complete` | **weighted_rrf R4** = batched keybuilder 复用 |
| [cache_latency_bench_rrf_round5.log.md](cache_latency_bench_rrf_round5.log.md) | `Complete` | **weighted_rrf R5** = RrfReleaseVision（MRO 零代码组合） |
| _(cp1 latency 调优**最终报告**)_ | — | 两份总报告 [`tuning_final_report.md`](../exp/cache_latency_bench/analysis/tuning_final_report.md)（weighted_sum 6 轮 35.49→4.15ms）+ [`rrf_final_report.md`](../exp/cache_latency_bench/analysis/rrf_final_report.md)（weighted_rrf 6 轮 32.88→4.88ms）在 `exp/cache_latency_bench/analysis/`（`artifact_layout`: 最终报告 `.md` → `analysis/`；工作期间 log 留 `logs/`）。 |
| [weighted_sum_libero10_replication.log.md](weighted_sum_libero10_replication.log.md) | `In Progress` (G1 APPROVED R3 / §4 Code 完成 / G2 APPROVED R2, 2026-05-29; tests/exp 616 passed) | **L2**: 把 weighted_sum 系统化实验 4 阶段（① 两层校准+权重搜索 → ② trajectory(search+weight research) → ③ threshold-pareto → ④ kinematic phase5）整体移植到 `libero_10`。代码范围（全 exp/、零改 src/）：①③ 零代码、② `emit_trajectory_yamls` 的 `_EXPECT_BASE/_EXPECT_OVERLAP` 断言放宽、Stage1 `refine_round.py` 加 `rank_keybuilder(results,stem)`、④ ~60-80 行（`preload_pkl_override`+`cfg_id` 双透传进 3 builder + 新 `CFG_SPECS["spatial16_ws_d1_best_libero10"]`）、新 `emit_top10.py`。关键运行约束：§2.3 决策阶段单 server 钉死（跨 GPU 7pp>5pp）、§4.0 server 数据双推 sha256 三等、§8.6 路径/suite override 防覆盖 libero_spatial 产物。§3.1 锁定 handoff 确定性筛选准则（winner-kb / N_base / top10 恰10 / winner-kb-d1-best 自动导出；仅 Stage3 base 中期临时定）。100 ep/config、含 Stage2b、Stage4 用 Option B（libero_10 自己 winner）。预算 ≈109k ep（含 2b，Stage2a 随 N_base 浮动；双 jupyter H200 + timan107）。数据预收集见 §4 |
| [weighted_sum_kinematic_phase5_replication.log.md](weighted_sum_kinematic_phase5_replication.log.md) | `In Progress` (G1 APPROVED 2026-05-28 12:42 CDT R4; §3.1 Post-G1 polish 完成; §4 Code 进行中) | **L2**: 在 weighted_sum d1-best (`vision_0@6_vision_1@50_robot_state@43__d1`, SR=74%) 检索之上复刻 verdict phase5 的 5 group × 237 cell × kinematic factor + composer `tier_thresholds`（注：本项目无 `ThresholdJudge` class，判分入口是 composer 的 tier_thresholds 字段）系统化扫描；**用 1 份 super warmup + offline calibration source 替代 148 份 per-cell + WarmupPool 链路**（dump factor list 由 `set().union(*declared_keys)` 直接驱动 = 真实 50 keys，覆盖 237 cell declared_keys 并集；G5 grid 加 `fh+ws ≤ 0.9` 三角约束删 (0.5,0.5) → 15 pair × 3 recipe = 45 G5 cell；总 cell = 48*4 + 45 = 237）。代码级核实 5 不变量：DumpingJudge ⫫ inner judge / search ⫫ judge / `payload.factors` 已 enrich / `(5,0)` 仅 online 不需 enrich / PercentileRollingCalibration "extras silently ignored"。**所有 cell eval yaml 用 `samples_source.type=offline + path=super_warmup_raw.jsonl`**，server 启动时各自读盘构 CalibrationSamples，完全绕开 WarmupPool LRU 链路。新模块 `exp/weighted_sum/kinematic/` (spec/super_warmup/strategy/runner/analysis) + `v2_spec.py` 加 `CFG_SPECS["spatial16_ws_d1_best"]` + `run_phase2.py` `--strategy={weight,kinematic}` 开关 + wire `per_step_writer=strategy._write_per_step` 进 `ConductorDriver`（G1 R2 B2 mandated）+ **`src/openpi/conductor/driver.py` +~14 行 driver-internal per_step flush**（ctor 注入 `per_step_writer` + `RLock` + 新方法 `_flush_per_step_for_stage(yaml_id)` + `_complete_stage` 末尾自动 call；G1 R2 B1 mandated 路径，**strategy.py base 与既有 5 个 `on_stage_complete(stage, ctl, ctx)` override 零改动** — `warmup_eval_strategy.py:169`/`tests/conductor/{test_driver.py:118,249, conftest.py:104}` 兼容）。Stage 0 ziyang10 server 重启加 `--warmup_dump_root` 给 super warmup fetch_dump 用（须先 agentchat 通知 owner 释放 GPU）；eval 用 offline mode 不需要 fetch_dump，xuanle 3-replica 不重启。预算 ~6.7h（super warmup ~15min + 237 cell × 100 ep ~3h on dual-server 16/48 + always-WARM 自跑 3 cell × 100ep + 分析 1h）|
| [verdict_phase5_libero10_systematic_sweep.log.md](verdict_phase5_libero10_systematic_sweep.log.md) | `Plan` (G1 APPROVED 2026-05-22 / §4 Code 完成 / G2 R1 NEEDS REVISION → Executor R1 applied / 待 Reviewer R2) | **L2**: 把 phase5 systematic sweep 移植到 `libero_10` — 除 task_suite 切换、G5 三个历史 warmup raw 在 libero_10 上重建、libero_10/cp1_spatial_pool_16.pkl 一次 `enrich-existing-pkl` 补 `payload.factors` (64 keys) + `library_stats` 外，所有实验参数、cell 集合、6-server 4:4:4:3:3:3 分配与 libero_spatial 版本 (`verdict_phase5_systematic_sweep.log.md`) 严格一致。**Audit R1 否决"零代码改动"路线**：`common/v2_spec.py:80` 把 `preload_pkl` 写死成 libero_spatial pkl，必须改 `v2_spec.py` + `phase5/spec.py` + `phase5/runner.py` 加 `preload_pkl_override` keyword-only 透传（≤ 150 行向后兼容改动），并新建 `phase5/g5_warmup_libero10_driver.py`（Items 2/3/4 否决 phase3/phase4 现成 runner 复用）；G3 共享 warmup race（spec.py:382, 4 distinct id 覆盖 48 cell）通过 Stage 1.5 单机串行预跑消除（Item 5）；`src/openpi/` 不动。跳过 always-warm / pure-inference / random_periodic baseline 重建（Q3=b 只看 SR）；不画 Pareto / heatmap。新数据落 `data/phase5_libero10_systematic/`、G5 raw 落 `data/{phase3,phase4}_libero10/`、yaml 落 `config/spatial16/phase5_libero10/`；commit 边界严格按 `artifact_layout.md` §3（`data/**` 本地保留+tar 离线归档，code/yaml/analysis/plan log 入库）。预算 ~24 060 ep / ~6-7 h wall-clock（6-server）|
| [weighted_sum_threshold_pareto.log.md](weighted_sum_threshold_pareto.log.md) | `Plan` (G1 owner-APPROVED R2 / §4 Code 进行中 / 运行待 wsweep winner) | **L2**: 用 weighted_sum **检索最终聚合总分**（模态聚合+trajectory 加权后的 top-1 `cp1_result.score`）做阈值控 FULL_HIT/WARM_START/MISS 三档，扫 SR×inference_ratio 帕累托。移植 verdict phase3 data-driven threshold sweep：A warmup 收总分分布(前置门验展度)→B (fh_ratio,ws_ratio) 分位/zscore 反解 (T_fh,T_ws)→C ThresholdJudge eval 扫格(start_t=0.5)→D (inf_ratio,SR) 帕累托。`ThresholdJudge`(src 零改)+phase3 threshold_solver+Pareto plot 复用。base yaml 等 wsweep winner 定 |
| [weighted_sum_trajectory_weight_research.log.md](weighted_sum_trajectory_weight_research.log.md) | `Plan` (G1 owner-APPROVED / §4 Code 完成 / 运行中 ~57%) | **L2**: 续 trajectory 实验——分离「权重对 d1 regime 过拟合(H-3a)」vs「trajectory 对强索引本质拖累(H-机制)」。仅 `cp1_spatial_pool_16`，grid3 加密(step 0.0625, ~70 weight) × depth{1,3,4,5} = ~280 yaml × 100 ep ≈ 28000 ep。**含 d1 重测**(新网格点需无偏同配置基线)。判据：各 depth 重搜最优权重能否追回 d1 天花板(~74%) + 最优权重向量是否随 depth 漂移。极少代码（新 `emit_trajectory_weight_sweep.py` 复用 `build_eval_config`+`grid3_weight_configs` + 新分析脚本，零改 src/ 与评测基础设施）。owner 已用 top10 3 次重测 ≤1pp 排除选择偏差 |
| [weighted_sum_trajectory_weight_alloc.log.md](weighted_sum_trajectory_weight_alloc.log.md) | `In Progress` (G1 APPROVED R3 2026-07-02 / §4 Code 完成 / 待 G2) | **L2**: 对 trajectory **每步权重** `trajectory_weights` 做 screening 搜索（合作者疑 d1>d3/d4/d5 源于历史固定递减方案）。每 depth 复用自己的 base 只搜每步权重形状（always_hit 下仅排名影响 SR）；搜索集 = S1 当前步主导×尾形梯度(含 c=0.9 near-d1 边界) ∪ S2 内部形状格点 ∪ S3 incumbent，实算锁死 **d3=52/d4=60/d5=59=171 config × 100 ep = 17,100 ep**（libero_spatial）。零改 src/：新 `emit_traj_weight_alloc.py`(从 tracked grid3+calibration 确定性重建 base，防 stale 断言 171 ID 集) + `analysis/analyze_stepweight.py`(YAML 读真实权重 + 互斥形状分类 + journal latest-ts 去重后配对 McNemar vs incumbent) + `tests/exp/test_traj_weight_alloc.py`(23 pass)。单 server ziyang10 `--replicas 3` + timan107 `--workers 48`；d1 作非裁决性 prior 参考、确认性重跑列后续 |
| [weighted_sum_trajectory_weight_alloc_libero10.log.md](weighted_sum_trajectory_weight_alloc_libero10.log.md) | `In Progress` (G1 APPROVED R3 2026-07-02 / §4 Code 完成 / 待 G2) | **L2**: 把 trajectory 每步权重 screening（搜索矩阵/配对分析/screening 框架完全复用 libero_spatial 已批准逻辑）搬到 **libero_10**，验证结论是否 dataset-general。计数不变 **52/60/59=171 × 100 ep = 17,100 ep**。核心新面：libero_10 base winner 非干净格点、CID 与 all_results.csv 均有损（`int(w*100)`）→ 新增 **tracked 非有损 `LIBERO10_BASE_MANIFEST`**（d3 0.62/0.37/0、d4 0.25/**0.4375**/0.3125、d5 0.5/0.5/0，值取自实际 base YAML；emitter 只读它、绝不从 CID/csv/grid 反推）+ 3 份 base YAML 入 `tests/exp/fixtures/`（deep-diff=0 锁唯一变量）。emitter/analyze 加 `--task-suite`（per-suite dispatch + 动态默认 None→post-parse + expected_ids 贯通 + provenance 参数化），libero_spatial 逐字节向后兼容（rollup 不变）。零改 src/；单 server ziyang10 + timan107 复用 libero_spatial 拓扑 |
| [weighted_sum_trajectory_search.log.md](weighted_sum_trajectory_search.log.md) | `Validated` (G1 APPROVED 2026-05-27 / G2 owner-waived per WA §7 / §6 Verify 1725 pass / 7200 ep 跑完 + 分析完成) | **L2**: 在 weighted_sum 两层 `weighted_score_sum_knn` 检索的最优配置之上叠加 trajectory search（复刻老 trajectory 实验对 Phase1 的做法）。两组 base：① per-keybuilder top2 + 倒数第二（A 口径=正规网格同 zscore，排除 norm2/iso）× 4 keybuilder；② 全实验 top10。去重后 **18 base × depth{3,4,5,6} = 72 yaml × 100 ep = 7200 episode**；`trajectory_weights` 复用老递减方案；depth-1 基线复用 weighted_sum 已有 SR。**零改 src/ 与评测基础设施**，仅新增 `emit_trajectory_yamls.py`（import 现成 `build_eval_config`）+ `plot_trajectory_results.py`。拓扑 2 台：server=jupyter 单 H200 `--replicas 2`（临时，原 3；同机=无跨 GPU 污染）、client=timan107 conductor `--workers 48 --gpus 8`（每 yaml 100 ep = 10 task × eval-trials 10），经 tether 编排（a100 不用） |
| [weighted_sum_two_layer_refactor.log.md](weighted_sum_two_layer_refactor.log.md) | `In Progress` (G1 APPROVED 2026-05-25 / §4 Code 完成 / 待 G2) | **L3**: weighted_sum 两层重构 + 两阶段校准/权重搜索实验 — 把 `_search_weighted_score_sum` 的"raw→[0,1]→percentile→加权和"拆成 **Layer-1 可插拔归一化层**（新 `score_normalizers.py`：仿射/z-score/logit/−log(1−cos)/power/exp-l2 registry，**单调保幅、排除 rank/经验CDF**）+ **Layer-2 加权和**；config `ScoreNormalizationConfig` 扩 `per_field {method,params}` schema（保留 `percentile` 向后兼容）。修复旧失败三根因（校准分布来源=库内随机对 / percentile 在高基线 cosine 上 p5≈p95 塌缩成 0.5 / 与 keybuilder 不分层）。实验：**Phase 1 离线**用真实 query×全库分布（LOEO 过滤 query 自身链消除 self-match）数据驱动选每 (模态,keybuilder) 的归一化方法+参数（选择指标=幅值分离 mag_sep−λ·sat，因 rank 指标对单调变换不变）；**Phase 2** conductor 纯-eval 扫单模态隔离+权重网格找有用模态与最优权重。libero_spatial 复用现有 6 库 artifact，零额外采集。新建 `exp/weighted_sum/` |

> See [Archive › Verdict Factor Judge](#verdict-factor-judge) for prior phases + refactor history.

### Server Infrastructure

| File | Status | Description |
|------|--------|-------------|
| [client_conductor_two_layer_refactor.log.md](client_conductor_two_layer_refactor.log.md) | `In Progress` (G1 APPROVED 2026-05-25 / Post-G1 polish / §4 Code + 3 轮自审完成 / G2 进行中) | **L3**: client 实验基础设施两层重构 — 三物理层 `worker`(执行,绑单卡 EGL slot)/`agent`(每机常驻,本地 fork+心跳)/`driver`(中枢)，driver 内部再做**机制/策略分离**：通用引擎进 `src/openpi/conductor/`（`scheduler`/`driver`/`agent`/`worker`/`journal`/`protocol`/`health`/`monitor`），可编程 `ExperimentStrategy` + `EpisodeRunner` 接口；具体实验剧本（verdict warmup/eval）作为策略留 `exp/`。把派发粒度从 yaml→subtask 降到 **episode 级中央队列 + worker pull**，消除 yaml 间等待泡沫；调度遵 **yaml 亲和软约束 + 永不空转 + 单 server yaml 数最少**；warmup→eval barrier 由策略用核心「stage 依赖+完成回调」原语表达（核心不含实验语义）。完美断点续跑（账本 journal + server WarmupPool 自愈）+ 报错重试（可重试/致命分类）+ per-worker 进度/吞吐/健康监控 + 按 server 分配 worker（48+48）。**server WebSocket 协议不动**；`examples/libero/main.py` 抽出 `LiberoEpisodeRunner` 并保留 standalone 兼容；现有 `run_phase.py`/`phase5/runner.py`/`g5` 迁移为策略插件。新建 `docs/architecture/experiment_conductor.md` |
| [backend_c2_autoguard_decouple.log.md](backend_c2_autoguard_decouple.log.md) | `In Progress` (G1 APPROVED / G2 APPROVED 2026-05-25 / §6 Verify done / 待 commit) | **L3**: 把 C2（runtime write-frozen）守卫从"每个具体 backend 手抄 `_check_frozen`"上提到 ABC 的 `__init_subclass__` 透明自动包装——任意 `VectorStoreBackend` 子类（含 `tests/review_tests/` 看不见的 reviewer fake）零改动、不改方法名即自动获得 C2；删 in_memory/qdrant 手抄守卫 + in_memory 冗余 `batch_insert` override。owner 选定 `__init_subclass__` 路线（否决 template-method+抽象 `_xxx_impl`，因后者会破坏封闭审查区无法迁移的 fake 子类）。公共 API / 子类实现契约不变、严格更解耦；改 `backend_base.py` + 2 backend + `cache_system.md` C2 段 + 新建 `test_backend_frozen_autoguard.py` |
| [concurrent_serving_scaleout.log.md](concurrent_serving_scaleout.log.md) | `In Progress` (§4 Code 完成 / Execution 自审查完成 / **待外界专家 G2**) | **L3**: 多进程 scale-out — `serve_policy.py --replicas N` supervisor spawn N 个子 server 占内部 loopback 端口 + 进程内连接级 sticky WebSocket router (`src/openpi/serving/replica_proxy.py`)，对外单端口；infer 连接级 sticky 保 per-connection cache/KV，控制面 ctrl 广播全 replica，metrics 聚合；`--replica-spawn-batch` 错开启动防 jupyter 32GB host-RAM OOM；child-watchdog 任一子进程死即拆除。配套 Phase-7 监控（`monitor.py` 主开关 + coordinator no_grad 显存修复 + 探针 + 1Hz util）、client `[client.timing]` 探针、`autotune_workers.py`（bracket+golden-section+USL 找最适 worker/wait）。a100/jupyter 实测 ~2.4×/server、fleet ~4.3×。自审已修 2 BLOCKER (timing 主开关 6 测试 / supervisor watchdog) + 多 MAJOR (router 失败路径 / warm-start 步数 / autotune 窗口) + 死代码清扫 + 228MB 二进制移出 logs/。Review Log 待 G2 追加 |
| [serving_throughput_problem.md](serving_throughput_problem.md) | `In Progress` (问题陈述 + 外部专家问答记录) | **L3 研究记录**：单进程 serving 吞吐瓶颈调查 — 写给外部专家的问题陈述 + 多轮专家回复，含 2-process=2× 决定性证据、CUDA-graph capture 失败、bucket-first 中性结果、瓶颈定位为 GIL kernel-launch 路径 + 闭环延迟，最终落在 full-replica scale-out 方案（落地见 [concurrent_serving_scaleout.log.md](concurrent_serving_scaleout.log.md)）|
| [throughput_util_exploration.log.md](throughput_util_exploration.log.md) | `In Progress` | **L3 探索记录**：jupyter + a100 并行参数 / 吞吐 / 设备利用率探索全程 — replica 数、worker 数、batch wait 扫描，瓶颈归因（闭环推理延迟 86% / 单 client CPU 上限），fleet 联合 ~51 inf/s |
| [concurrent_serving_optimization_plan.log.md](concurrent_serving_optimization_plan.log.md) | `Implemented` (G1 APPROVED R3 / G2 APPROVED R4 — 2026-05-23) — 7 phase + Final landed; 1441 pytest pass / 0 regression; C2 frozen guard covers 5 mutation entries (insert / batch_insert / delete / upsert / load_artifact); operator guide merged into [`docs/experiments/conductor_tutorial.md`](../docs/experiments/conductor_tutorial.md) | **L3 联合 plan**：议事 [`serving_optimization_council.log.md`](serving_optimization_council.log.md) 9 议题决议落地。7 模块：M1 BatchingCoordinator (A2 = 组合 ① 全 batching + 动态 window + sub-batch split + per-request transform→stack) / M2 BundleDispatcher (A6 `_current_bundle` → `dict[bundle_id]`) / M3 BackendPool (A6 子优化 — pkl-shared backend 实例池) / M4 FrozenGuard (A7 + C2 — `BackendFrozenError` + runtime read-only contract，与 audit report §9.1 RLock 漏洞闭环解决) / M5 offline_writers 属性名修复 (A8 — `config.py:1768` 改 `_factors` + `isinstance(f, OfflineWriter)` filter，与 audit report §9.5 闭环) / M6 Single Process 默认 (A1 — `--concurrent` default True，sweep workflow 从多 server 迁移到 1-server × N-bundle) / M7 Throughput/Latency Benchmark Tool (Plan Kickoff Amendment — `exp/serving_benchmark/` 自动化探索吞吐 / latency 极限和关系，4 个测试 mode：sparse-to-dense 扫描 / freq sweep / yaml density / batch window 优化，对应 audit report §B.3 第 4 项 batched-infer microbench)。**两条硬约束 (PO 提出，G1 强制 verify)**：C1 保留非 `--concurrent` 模式作为极限速度基准（non-concurrent path + `torch.compile` 编译产物零改动）；C2 server runtime 禁止修改数据库内容（backend `insert/delete` 抛 `BackendFrozenError`，runtime backend `_entries` 完全 read-only + GIL 原子 dict lookup = 多 connection 并发 read 安全无 race）。7 phase 实施：Phase 1 M4+M5 (low-risk foundation) → Phase 2 M3 BackendPool → Phase 3 M2 BundleDispatcher → Phase 4 M1 BatchingCoordinator → Phase 5 M6 + sweep migration → Phase 6 M7 Benchmark → Phase 7 Comprehensive Verify。预估 ~2,400 行代码 + ~1,200 行测试 + ~500 行 docs |
| [serving_optimization_council.log.md](serving_optimization_council.log.md) | `Plan` (议事完成 2026-05-23 — 9 议题全部决议；待启动 2 个孵化 plan) | **L1 议事框架文档**：把 [`server_concurrency_resource_audit.log.md`](server_concurrency_resource_audit.log.md) §12 的 6 个优化机会 + §9.1 / §9.5 浮出的 2 个 latent code issue 拆为 9 个议题逐一决议。**议事结果**：A0=D 跳过前置实测（推迟到 Verify）/ A1=A 全切单进程 / A2=组合 ① 全 batching (CPU-1+动态 window+sub-batch split+per-request transform→stack) / A3=D 重 KeyBuilder 不优化 / A4=D 多 stream 不做 / A5=A 删 baseline `_sync()` 用 CUDA Event 替代 / A6=A `dict[bundle_id]` + pkl-shared backend pool / A7=B docstring + frozen 守护 / A8=A 修 `_extractors`→`_factors` filter。**两条硬约束 (PO 提出)**：C1 保留非 `--concurrent` 模式作为极限速度基准；C2 server runtime 禁止修改数据库内容。**孵化 2 plan**：L1 独立 Baseline `_sync()` 移除 + L3 联合 Concurrent Serving Optimization (A1+A2+A6+A7+A8)。议事 append-only |
| [server_concurrency_resource_audit.log.md](server_concurrency_resource_audit.log.md) | `Validated` (Audit R3 APPROVED 2026-05-23 — 仍保留在顶层 Active 作为后续 serving 优化工作的起点) | **L3 研究报告**：openpi server 全模块资源占用画像 — WebsocketPolicyServer / InferenceInterceptor / CacheOrchestrator / Policy.infer / pi0_pytorch stage1-3 / KeyBuilder (含 CLIP / LLMLayerExtract) / SearchStrategy / InMemoryBackend / CompositeJudge 17 因子 / DumpingJudge / CacheStorage / WarmupPool / SystemTimer / offline_writers / metadata_db 每层的 GPU vs CPU 占用、GIL 释放、batch 能力、多核可能性、per-connection vs shared、锁竞争画像。**核心架构事实**：①cache hot path batch=1 注入点覆盖 baseline `policy.py:87 + :138/199` 与 cache path `interceptor.py:512-516, 559-560, 668-669`；②`--concurrent` 路径 `serve_policy.py:419` 传 `eager=True`，**实验实际跑未编译 raw `run_stage1/2/3`**（`interceptor.py:172-178`），`torch.compile("max-autotune-no-cudagraphs")` 仅 single-connection 模式生效；③`backend_base.py:14-16` 声明的 "CacheStorage RLock 序列化" **代码未实现**，`InMemoryBackend` 共享状态 `_active_search_sessions` / `_score_memo` lock-free，依赖 GIL 单 op 原子性 + mutation guard；④`models/gemma.py:69-87` gemma_2b/300m 均 `depth=18, num_kv_heads=1, head_dim=256`（**不是 30 层**），KV cache 估算修正；⑤Transform `DataTransformFn` Protocol unbatched，request batching 不能"删一行 `[None, ...]`"；⑥`config.py:1768 _extractors` vs `composite_judge.py:137 _factors` 属性名不一致，**offline_writers 链路潜在代码 bug 嫌疑 ⚠**。Review 过程：R1 NEEDS REVISION (7 items) → Executor R1 (7/7 Accepted) → R2 REJECTED (1 Constitutional + 2 Blocking) → Executor R2 (4/4 Accepted 含 Constitutional 修复) → R3 APPROVED。**R3 scope 限制**：approval 仅限作为静态可行性审计；所有估算（毫秒、GB、SM 占用、收益倍数）仍为**未实测的架构推断**，未来推进任何优化 plan 前必须先按附录 B.3 完成 5 项廉价实测（`nvidia-smi dmon` / SystemTimer CSV / `py-spy top` / batched-infer microbench / `nsight-systems`）。研究方法：4 个 Explore sub-agent 并行调查 + executor 三轮亲验关键 line 锚点。不修改任何 src/ 代码（用户全程指令"暂时没有 code 环节"）。Review Log 永久保留（§10.1） |
| [full_repo_audit_2026-05-26.log.md](full_repo_audit_2026-05-26.log.md) | `Validated` (审计 + 修复完成 / §6 Verify 1721 pass 0 regression — 2026-05-27) | **L2/L3 全库审计**：51e364b(4-01)→HEAD 三轮 16 路 agent，过滤 yaml/data 聚焦 .py + docs/logs。修复 1 个 HEAD 红测试 (journal done/failed 双终态) + 10 个 CONFIRMED MAJOR (attach_model 共享 config eager 竞态 / WeightedSumZeroNan 零权重除零 / websocket 连接计数泄漏+单连接锁死 / batching worker 静默死锁 / bundle 切换残留已结束 policy / load_cache_config 阻塞事件循环 / driver 断连 requeue 代际栅栏 / 并发 Ctrl+C 丢 per-step 日志 / conductor MSG_SHUTDOWN 孤儿 / weighted_sum cid 取整不一致) + Round 1-3 追加修复（proxy fan-out timeout / submit backstop / agent process-group stop / WS frame cap / config validators / docs link repair 等）+ 死代码/中文注释/注释漂移/文档索引同步。**PO 决定"仅记录不改码"3 项**：C1 load_cache_config 远程未认证 pickle RCE (CONFIRMED，公网入口)、M11 eager→sdpa baseline 不可比 (保留 sdpa 重测)、M12 stage_device_placement CI 零覆盖 (维持 skip)。28 个核心改动文件 + docs/log/archive 同步 |
| [full_repo_audit_commit_review_2026-05-27.log.md](full_repo_audit_commit_review_2026-05-27.log.md) | `Fix Applied` (HEAD commit review + owner-approved fixes — 2026-05-27) | **G2-style 审查报告 + 修复记录**：审查 `6a2a2c0` 全库审计修复 commit，发现并修复 3 个 blocking：`ReplicaProxy.serve()` 多副本公网入口补 256MB `max_size`；`OPENPI_STAGE3_BUCKET_FIRST=1` stage3 bucket metrics 异常改为 non-fatal；docs/index 的 pre-Phase-5 bit-identical 说法改为当前 sdpa 数值语义。证据测试：review-only probes 在修复前 2 failed；常规回归测试已加入 `tests/serving/test_replica_proxy.py` 和 `tests/cache/test_serving_optimization.py` |

---

## Archive

Completed and historical logs. See [`archive/`](archive/) for all files.

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
