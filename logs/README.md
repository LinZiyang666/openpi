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

### Stage Device Placement

| File | Status | Description |
|------|--------|-------------|
| [stage_device_placement_plan.log.md](stage_device_placement_plan.log.md) | `In Progress` | L3: split device placement by Stage, supporting cuda/cpu/meta modes. G1 approved |

### Redundant Token Pruning

| File | Status | Description |
|------|--------|-------------|
| [redundant_token_prune_gpt.log.md](redundant_token_prune_gpt.log.md) | `Design Only` | GPT-drafted preliminary proposal discussion (no project-specific implementation details) |

### Pi0.5 High-Level Autoregressive Decode

| File | Status | Description |
|------|--------|-------------|
| [pi05_hl_ar_decode_plan.log.md](pi05_hl_ar_decode_plan.log.md) | `Plan` | L2: add optional HL autoregressive decode (`lm_head` + incremental KV) to the inference path; Phase B integration decided after the Phase A probe gate |

### LLM Layer Extract KeyBuilder

| File | Status | Description |
|------|--------|-------------|
| [cp1_llm_layer_extract_key_builder_plan.log.md](cp1_llm_layer_extract_key_builder_plan.log.md) | `Plan` | L2: 新 `cp1_llm_layer_extract` KeyBuilder — 在 KeyBuilder 内部独立跑 PaliGemma 第 N 层 forward (借引用, 不改 Stage 2)，两步可插拔架构 (`LLMLayerExtractor` + 新 `PrefixReducer` 协议)，首版 `prefix_mean_pool` / `per_modality_pool` 两个 reducer |

### Data Artifact Build

| File | Status | Description |
|------|--------|-------------|
| [libero_10_cache_artifact_build_plan.log.md](libero_10_cache_artifact_build_plan.log.md) | `Plan` | L1: 用 `exp/common/data/db_init/libero_cache/libero_10` 采样 init 驱动 LIBERO 推理，h5 落到 `exp/common/data/db/libero_cache/libero_10/`（与 `libero_spatial` 同约定），再 build 6 份 InMemoryBackend pkl artifact（4 pool + CLIP ViT-B-32 + ViT-L-14）到 `exp/common/data/cache_artifacts/libero_10/` |

### Phase1 Experiments

| File | Status | Description |
|------|--------|-------------|
| [phase1_libero_10_run_commands.log.md](phase1_libero_10_run_commands.log.md) | `Plan` | L1: `exp/common/config/phase1/libero_10/batch{1,2,3}/` 共 60 个 run 的执行命令清单（server + runner），含 prompt_emb 验证组；附 init-state 不匹配 caveat |

### Trajectory Experiments

| File | Status | Description |
|------|--------|-------------|
| [trajectory_libero10_split_plan.log.md](trajectory_libero10_split_plan.log.md) | `Plan` | L1: 把 `config/trajectory`、`data/{phase1,trajectory}`、`analysis/{phase1,trajectory}` 全部按子实验（libero_spatial / libero_10）重组，抽公共 `plot_common.py`；libero_10 trajectory YAML 留待 Phase1 libero_10 出榜后另起 L1 任务生成 |
| [trajectory_libero_10_run_commands.log.md](trajectory_libero_10_run_commands.log.md) | `Plan` | L1: `exp/common/config/trajectory/libero_10/batch{1,2,3}/` 共 60 个 run 的执行命令清单（server + runner），一个 depth 一个 batch（d=4/5/6），`--state-path` / `--log-dir` 显式指向 `data/` 以避免结果写回 `config/` |

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

### Historical

| File | Status | Description |
|------|--------|-------------|
| [doc_cleanup_plan.log](archive/doc_cleanup_plan.log) \[[EN](archive/doc_cleanup_plan.en.log)\] | `Historical` | Documentation cleanup plan |

---

## Maintenance Rules

> **AGENT: READ FIRST** — Log status system and lifecycle rules are defined in [`WORKING_AGREEMENT.md` §5 Log Management](../WORKING_AGREEMENT.md#5-log-management). The Working Agreement is authoritative.
