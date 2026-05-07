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
| [verdict_factor_judge_refactor.log.md](verdict_factor_judge_refactor.log.md) | `Implemented` (G1 APPROVED Round 4 / B0-B7 已 land / G2 进行中) | L3: 运动学 judge 整体重构 — 4 desc (jerk/direction/dispersion/path_length) × 4 变体 (online/offline × action/state) + topk_action_variance = 17 因子；命名 `<desc>_<source>_<channel>`；4 层正交架构（Normalization → 因子 → Calibration → Composer）每层可插拔互不知道彼此；online 统一 splice `[history[-P:], winner, walk_next(F)]` action 与 state 完全对称；校准数据 **2 选 1 必备 + 启动 fail-fast** (offline LibraryStats/jsonl/pkl + warmup yaml + WarmupPool)，**no cold-start** 彻底废除（cold_start_strategy / force_miss / passthrough / lenient / sentinel / all_nan_fallback / requires_library_stats / record_action 7 项废除）；DumpingJudge 包 4 层外 + 自持独立 dump 因子列表 + 第 1 层 Norm 副本；诊断 `factor_outputs.{raw, calibrated, composer_score, schema_version=2}`；exp/verdict_factor_judge/v2_spec.py 全新 yaml 生成器；老代码仅靠 git history（4 个 stub 文件全删，超出 plan §14.4 import-only 承诺，已落 §6.14.1 deviation note）。§11-§17 详细 Plan + B0-B7 落地 + 12+3 项 validator + 19 个测试重写 + 4 份 docs 同步 + factor_postprocess `enrich-existing-pkl` smoke gate；G1 APPROVED Round 4 (2026-05-07 13:34)；G2 Round 1 REJECTED 6 项已修，待重审 |
| [old_verdict_factor_candidates.log.md](old_verdict_factor_candidates.log.md) | `Design Only` | ENGRAM 缓存子系统 Judge / Gate 阶段的统计 / 运动学因子候选目录 — 列出 F1a-A / F1a-T / F1b-A / F1b-T / F2 五个候选因子的定义、数据源、计算时机（offline / online / hybrid）、风险与组合策略；服务于 verdict 实验规划，不为论文起草服务 |
| [old_verdict_factor_judge_b1_b2.log.md](old_verdict_factor_judge_b1_b2.log.md) | `Implemented` | L3: B1+B2 合并实施 — 填实 F1a-A / F1a-T / F2 在线 extract、F1b-A / F1b-T 在线读取 + 离线 OfflineWriter、`LibraryStats.compute_from_entries`、3 Composer + PercentileRollingNormalizer 算法；Orchestrator B1 接线（view+history 注入 / `_state_history` / anchor CP / on_task_end leak fix / winner fetch rewire）；config 解禁 `composite` + 7 项校验激活（含 5a-5d warm_start 子规则：CP1-only、canonical timestep、pairwise 配对、tier 排序）；Backend `load_artifact` 加载 / fallback `library_stats`；新建 `factors/_descriptor_kernel.py`（F1a/F1b 共用单窗口 helper）+ `exp/common/factor_postprocess.py`（`enrich_artifact_with_factors`）；三 build 脚本接 `--factors-yaml` CLI；docs status 由 B1/B2 → Implemented；不动 21-window 实验级 artifact 重建（属独立实验 plan） |
| [old_verdict_factor_judge.log.md](old_verdict_factor_judge.log.md) | `In Progress` | L3: 把 verdict_factor_candidates 中的因子落到 cache 子系统 Judge 阶段 — 新建 `factors/` 模块（OnlineExtractor / OfflineWriter Protocol 含 capability flags + `describe(cls, params)` classmethod + registry 含 `get_class` accessor + 5 个因子 / 3 个算法基类 + thin-subclass 模型按 source 绑定 capability + composers / normalizers Protocol with bind_orientations / bind_keys）+ `payload_view.py`（PayloadView + StoragePayloadView，per-`check()` memo + chain-walk 无 fork 实现，fork policy 全 raise NotImplementedError）+ `CompositeJudge` 类骨架（含 collect+bind / key contract assertion / cold-start all-NaN sentinel 短路 MISS）；schema 仅加 `CachePayload.factors`；走 facade-only `CacheStorage.fetch_entry` + `library_stats` duck-typing（Backend ABC 不动）；F2 经新 `min_top_k_hint` wiring 不破坏 `search_strategy.top_k` 语义；anchor-checkpoint policy 处理 CP1/CP3 enabled 组合的 state_history append 单点 + on_task_end 显式 reset；config validator 6 项纯静态 composite-specific 校验（含 capability vs backend.type / warm_start CP1-only + canonical timestep + pairwise rule + tier ordering / `_normalize_windows` dict→tuple / non_monotonic directions 等）；B0 `_JUDGE_TYPES` 不含 composite（fail-fast at config load），B1 解除 gate 同 commit algorithm；§8 是代码级蓝图；docs 同步 `cache_system.md` §5.6 contract refinement + 新增 §5.11 PayloadView + §5.12 Verdict Factor System；G1 APPROVED at R13。**B0 实现进行中：schema + protocols + factor metadata + Composer/Normalizer 骨架 + CompositeJudge + facade duck-typing + config dataclass/builder/validator + docs §5.6/§5.7/§5.11/§5.12 已落；剩余 B0 单元测试 + G2 评审** |
| [old_verdict_factor_judge_dedicated_runner_plan.log.md](old_verdict_factor_judge_dedicated_runner_plan.log.md) | `Implemented` (G1 APPROVED R5；§4 Code 完成；G2 APPROVED R1，2 非阻塞 doc 建议已 accept) | L3: B1+B2 一并 — observability via `__hit_meta__` + warmup preload。**B1**：`InferenceInterceptor.infer()` FULL_HIT 早返 + 末返合并 WARM_START/MISS/no-orch 三分支 attach `result["__hit_meta__"] = {hit_type=cp1_result.hit_type.name, start_t, winner_id=cp1_result.entry_id, cp1_score=cp1_result.score}`（no-orchestrator 走占位 MISS）；client SDK 0 改动（`infer()` return dict 自动透传）；`examples/libero/main.py` 加 `--per-step-log-dir/--yaml-id/--phase`，`_run_episode` 加 `infer_recorder` 钩子，每 worker line-buffered temp file 全程开 + exit 前 merge 按 `(task_id, subset_init_state_idx, episode_id integer, step_idx)` 4-tuple 排序；docs/architecture/cache_system.md §5.13 新增 wire-level observability + warmup preload 协议文档。**B2**：每 yaml emit sibling `<stem>__warmup.yaml` 用现成 `AlwaysWarmStartJudge + DumpingJudge` 跑 W=2 trial/task ≈ 420 verdict（`phase1_spec.build_warmup_yaml` + `dump.deferred=true` 让 CI/offline 校验通过）；新 `exp/verdict_factor_judge/run_phase.py` 7 步 orchestration（subprocess.run main.py 双 phase + ws ctrl 编排）；server-owned warmup dump root via `--warmup-dump-root` 启动 mkdir mode 0o700 + uid/mode self-check；`fetch_dump{warmup_yaml_id}` 双 .resolve() allowlist（拒 path traversal + symlink escape）；`unload_warmup_buffer{eval_yaml_id}` server 派生 warmup name 防 G1R3 conflation；`CurrentCacheBundle.yaml_id` 字段 + `load_cache_config{yaml_id}` 扩展 + `_fill_deferred_dump_paths` + 3 新 ctrl + `WarmupPool[eval_yaml_id]` LRU+thread-safe+deep-copy + `PercentileRollingNormalizer.preload_buffer`；`build_per_connection_components(yaml_id=)` 自动给所有 per-conn composite normalizer preload（drill 过 DumpingJudge wrapper）；`judge.py` 0 改，yaml schema optional，向后兼容。~1370 行 8 步实施 (B1.1→B1.2→B2.1→B2.2→B2.3→B2.4→B2.5→B2.6) + 12 测试 file（含 path traversal + symlink escape + 双 yaml_id e2e + 24 sibling warmup yaml validate / `/tmp/openpi_warmup` 不存在 case）。G2 reviewer 验 85 新增 + 159 回归 全 PASS |
| [old_verdict_factor_judge_phase0_phase1_run_commands.log.md](old_verdict_factor_judge_phase0_phase1_run_commands.log.md) | `Plan` | L1: Phase 0 (3 yaml × 100 ep AlwaysHit + DumpingJudge calibration) + Phase 1 (24 yaml 单因子 ablation × 100 ep) 执行命令清单；3 GPU server × 1 client/server（Phase 0/1 同款拓扑：server `_current_bundle` 是 module-level global，多 client 并发 `load_cache_config` 会全局覆盖，必须 1:1）；config 目录按 phase 分子目录 (`<cfg>/phase0/` `<cfg>/phase1/`)；Phase 1 yaml 由 `phase1_spec.py` 笛卡尔生成 24 份（3 cfg × 8 descriptor）；frp 端口 8998/8999/9000；含 DumpingJudge JSONL 并发撕裂 + bundle global race + dump.config_id 不变量 + search_strategy.top_k=5 必填提醒 |
| [old_verdict_factor_judge_phase2_run_commands.log.md](old_verdict_factor_judge_phase2_run_commands.log.md) | `Plan` | L1: Phase 2 Layer 1 — 6-server 执行命令清单（26 yaml × 3 cfg × 100 ep 单因子内部 desc/window 探索） |
| [old_verdict_factor_judge_phase2_layer2_run_commands.log.md](old_verdict_factor_judge_phase2_layer2_run_commands.log.md) | `Plan` | L1: Phase 2 Layer 2 redesign — 6-server / 6-client 运行教程（spatial16 only，240-cell threshold sweep） |
| [old_verdict_factor_judge_experiment_plan.log.md](old_verdict_factor_judge_experiment_plan.log.md) | `Plan` (G1 APPROVED) | L2: verdict factor judge 上线实验 plan — 7 阶段（Phase 0 baseline 复用 + calibration dump → Phase 1 单因子 ×（D-ALL/D-JERK/D-DIR + T-DUAL_07 WARM_START 探针）→ Phase 2 因子组合 ×2 tier → Phase 3 composer 类型 → Phase 4 窗口/normalizer/tier/描述子启用 → Phase 5 S-CALIB 学权重（paired McNemar/Wilson 决策）→ Phase 7 cross-task；Phase 6 取消已内嵌于每 phase ×3 cfg）；KeyBuilder + Search Strategy 锁定 warm_start 同款 3 套（CFG-CLIP `clip_w7_d4` / CFG-MAX `max_pool_w3_d5` / CFG-SP16 `spatial16_w8_d4`，共用 `weighted_rrf_knn` rrf_k=60 top_k=1）；artifact 用 `libero_spatial/{clip_vit_b_32,cp1_max_pool,cp1_spatial_pool_16}.pkl`（已带 168 keys F1b factors）；不用 ThresholdJudge baseline，复用 `warm_start/data/baseline_failures.json` + `warm_start/data/{clip,max_pool,spatial16}/` 现成 500 ep × 3 cfg baseline 不重跑；`inference_time_saved_ratio` 公式含 0.5 系数（warm_start 仅省 stage 3 ≈ 50% inference）；non_monotonic curv_radius / cum_disp default direction 按 normalizer 模式分两套（N-PCT percentile `range:[0.3,0.7]` / N-PASS z-score `range:[0.3,1.5]`，generate_yamls.py 自检）；F1b 长窗口 entry 链两端 NaN 兜底（实测 T median=21，`(5,5)`=48% NaN / `(7,7)`=67%；§3.3b 6 条规则）；W-MIX 默认 `(0,3)(1,1)(3,0)(0,5)(5,0)`；每 run 100 ep（10 task × init_idx 0..9 固定子集）；决策协议 paired McNemar p<0.10 + Wilson 95% CI 不重叠；总预算 117 run / 11,700 ep（3 cfg 全跑），如 Phase 0 数据支持收紧到 1 cfg ~4,400 ep；时间估算不在 plan 范围；主要代码改动：`src/openpi/cache/components/judge.py` 新增 `DumpingJudge` 透明包装类（filtered lifecycle dispatch + `__getattr__` fallback + `min_required_top_k` 合并 + JSONL append-only 双字段 identity）；`JudgeConfig.dump:{path,config_id,factors}` schema + `_build_composer` 给 and/or 补传 `directions`；`episode_start.extra_metadata` 通道 5 wrapper 显式扩签名（websocket_client_policy + websocket_policy_server + Policy/CollectionPolicy/DataCollector + InferenceInterceptor + Orchestrator）+ judge lifecycle 走 `_safe_call_lifecycle` filtering 不动现有 4 judge；cfg-prefixed yaml 命名约定锁定 yaml stem == `dump.config_id` == `cache_eval_results.json.config_id` 三处全局唯一（generate_yamls.py 强制）；runner 仅透传 `--episode-filter PATH` 给 `main.py:88 Args.episode_filter`（已有 mechanism 复用） |

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
