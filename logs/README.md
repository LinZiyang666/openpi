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
| [verdict_phase4_weight_sweep.log.md](verdict_phase4_weight_sweep.log.md) | `In Progress` (G1 APPROVED R3 / §4 Code 完成 / G2 进行中 — 2026-05-08) | L2: Phase 4 权重扫描 — 把 phase3 winner g1+g6 / g10+g6 融合成 2 个 10-factor recipe (p1, p2)，per-recipe 锁 ultra-cheap anchor (p1: (0.5,0.5) SR=0.95 / p2: (0.5,0.4) SR=0.96)，3 强制轮 (R1 α 7 点 / R2 offline 4-desc 9 pattern / R3 online 2-desc 5 pattern) + 1 条件轮 (R4 W-FUT 双窗 5 pattern)；总 42-52 cell ≈ 47-70 min wall-clock。**含 src/ 改动**：`WeightedSumZeroNanComposer._score_only` 真加权和（旧版等权平均；G1 R1 Blocking 1 修订）+ `phase3_threshold_solver.reconstruct_scores` 加 `composer_weights=None` keyword-only（向后兼容 None=phase3 行为）；**phase3_spec.py 不动** — phase4 直接调 `v2_spec.build_eval_yaml/build_warmup_yaml`，phase3 RECIPES 全程不被 mutate（G1 R2 Blocking 2 修订）；4-mode CLI (`emit-warmup-yaml` / `run-warmup` / `emit-eval-yamls` / `run-eval`) 把 server / no-server 工作分到不同命令；CLI per-recipe winner mapping (`--alpha-star "p1=0.4,p2=0.6"`) + `--recipe` 单 recipe 限制；preload-before-load 顺序与 phase3 line 380-382 一致；R4 触发条件改为对比 R3 baseline (uniform online) 而非 phase3 best；新建 `phase4_spec.py` / `run_phase4.py` / 2 分析脚本 + 5 测试文件（165 tests pass，含 numeric weight sensitivity / solver passthrough / phase3 RECIPES 不变性 / per-recipe dispatch / preload-before-load / round-specific decision_gate） |
| [verdict_phase3_run_commands.log.md](verdict_phase3_run_commands.log.md) | `Plan` | L1: Phase 3 数据驱动 threshold sweep 6-server 执行命令清单 — spatial16 only，11 recipe 切 6 batch（前 5 batch 各 2 recipe，batch6 单 g11）；端口拓扑同 phase2（S1-S3 frp `155.98.36.13:8998/8999/9000`，S4-S6 直连 `149.165.151.106:8001/8002/8003`）；§0 一次性 enrich pkl（168 老 key 保留 + 64 新 key 追加 = 232 keys，`exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl` canonical 路径）；§1 phase3_spec emit 11 warmup yaml；§2 6 server bootstrap (CUDA_VISIBLE_DEVICES=0 + `--warmup-dump-root /tmp/openpi_warmup_s{1..6}`)；§3 6 client `run_phase3.py` 每 batch 独立 summary file 避免并发冲突；§4 §4.1 cat 合并 6 summary → 176 rows + §4.4 `plot_pareto_phase3.py` 出图（warm cost 0.75 = phase2 0.85 -0.10）；§5 fail-fast `_NA` 自动写 16 cells；总预算 ≈ 17,820 ep / ~3.0 h |
| [verdict_phase3_threshold_sweep.log.md](verdict_phase3_threshold_sweep.log.md) | `In Progress` (G1 APPROVED R5; §4 Code 完成；G2 进行中) | L2: Phase 3 数据驱动 threshold sweep — 在 spatial16 phase2 layer1 11 个金圈点 recipe（因子配置完全冻结，仅 thresholds 改成从 warmup 反推）上扫 (FH_ratio, WS_ratio) ∈ {0.2,0.3,0.4,0.5}² 16 cell × 11 = 176 eval yaml。新 composer `weighted_sum_zero_nan`（NaN→0 仍计入，分母固定 N，**不带** all-NaN warm fallback，强制双 threshold FH+WS）；warmup 20 ep（W=2 trial/task 同 phase2）`AlwaysWarmStartJudge + DumpingJudge` 写 `factor_raw` JSONL；solver **离线重建** — 从 `factor_raw` 自建 `CalibrationSamples` → `PercentileRollingCalibration(samples, window_size=50).bind_keys(keys)` (saturated buffer，bind_keys 失败时 fail-fast `_NA`) → `WeightedSumZeroNanComposer._score_only(cal(raw))` 得 score 序列 → 降序 quantile 切位生成 (FH_thr, WS_thr)；start_t **0.5** （warm cost 0.75）；spatial16 cfg only；现有 `cp1_spatial_pool_16.pkl` 168 老命名 key 兼容保留 + 64 新命名 offline key 通过 `enrich-existing-pkl` 追加（不重 build）；canonical pkl path 锁定到 `exp/common/data/cache_artifacts/libero_spatial/` (修 v2_spec 三条 broken 路径)；§6.3 deviation: emit-on-demand 替代 placeholder eval yaml（runner 调 `build_eval_yaml_for_cell` 直接产 final yaml）。改 `src/openpi/cache/components/factors/composers/__init__.py` + `src/openpi/cache/config.py` (validator §5e + factory) + 新建 `exp/verdict_factor_judge/phase3_spec.py` / `phase3_threshold_solver.py` / `run_phase3.py` / `analysis/plot_pareto_phase3.py` + 5 测试文件 (73 tests)；G2 APPROVED Round 2 (2026-05-07) |

> See [Archive › Verdict Factor Judge](#verdict-factor-judge) for prior phases + refactor history.

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
