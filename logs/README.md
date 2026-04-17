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

### Warm Start

| File | Status | Description |
|------|--------|-------------|
| [warm_start_sweep_plan.log.md](warm_start_sweep_plan.log.md) | `Implemented` | Warm start 成功率扫描实验：3 keybuilder × 3 start_t (0.7/0.5/0.3) + always_skip/always_hit 对照；新增 AlwaysWarmStartJudge |

### Trajectory Deviation

| File | Status | Description |
|------|--------|-------------|
| [trajectory_deviation_experiment_plan.log.md](trajectory_deviation_experiment_plan.log.md) | `Plan` | Trajectory deviation 纠偏实验：3-Phase (离线诊断→信号分析→Oracle 纠偏) |
| [trajectory_deviation_corrective_experiment.log.md](trajectory_deviation_corrective_experiment.log.md) | `Plan` | 纠偏详细实验计划：GT 收集→Deviate Score→Spawn 纠偏，含 trajectory depth 预填充方案 |
| [trajectory_deviation_corrective_implementation.log.md](trajectory_deviation_corrective_implementation.log.md) | `In Progress` | 纠偏代码级实施跟踪记录（被 `src/openpi/cache/` 多处引用作为规范脚注） |
| [trajectory_deviation_corrective_implementation_review.log.md](trajectory_deviation_corrective_implementation_review.log.md) | `Plan/G2 Review/Verify` | 代码审查记录：Layer A+D+E、Layer B、Layer C、Layer F APPROVE；Verify 离线部分已落地，待端到端 dry run |

### Stage Device Placement

| File | Status | Description |
|------|--------|-------------|
| [stage_device_placement_plan.log.md](stage_device_placement_plan.log.md) | `In Progress` | L3: 按 Stage 分离 device placement，支持 cuda/cpu/meta 三种模式。G1 已通过 |

### Redundant Token Pruning

| File | Status | Description |
|------|--------|-------------|
| [redundant_token_prune_gpt.log.md](redundant_token_prune_gpt.log.md) | `Design Only` | GPT 初步方案讨论（不含项目实现细节） |

### Code Organization

| File | Status | Description |
|------|--------|-------------|
| [exp_reorg_plan.log.md](exp_reorg_plan.log.md) | `G2 Approved` | `exp/` 目录按实验重组：4 个实验子包 + `common/` 公用包；G2 复审已批准，可提交 |
| [experiment_artifact_layout_plan.log.md](experiment_artifact_layout_plan.log.md) | `Plan/G1 APPROVED` | 实验脚本/配置/产物/数据全仓普查 + 统一布局（`exp/<exp>/{config,data,analysis}/`）；owner G1 R2 指示"纯位置重构、24 个 result JSON 保持 tracked 用 `git mv`"；8 phase / 51 step；G1 R3 APPROVED，可进入 Phase 0 |

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

### Cache Experiment / CP1

| File | Status | Description |
|------|--------|-------------|
| [cache_experiment_plan.log.md](archive/cache_experiment_plan.log.md) \[[EN](archive/cache_experiment_plan.en.log.md)\] | `Implemented` | CP1 experiment: 5 reducers (incl. CLIP) x RRF fusion |
| [cache_cp1_impl_plan.log.md](archive/cache_cp1_impl_plan.log.md) \[[EN](archive/cache_cp1_impl_plan.en.log.md)\] | `Implemented` | CP1 in-memory implementation plan for large-scale experiment |
| [cp1_warm_start_impl_plan.log.md](archive/cp1_warm_start_impl_plan.log.md) | `Validated` | CP1 warm start 实现计划：4 Phase (性能修复→写入→判定+执行→文档) |
| [cp1_warm_start_investigation.log.md](archive/cp1_warm_start_investigation.log.md) | `Historical` | CP1 warm start 可用性调查，产出转为 impl plan |

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
| [redundant_token_prune_plan.log.md](archive/redundant_token_prune_plan.log.md) | `Implemented` | Plan A 冗余 token 剪枝：temporal scoring 两步 KeyBuilder，含 G2 审查记录 |
| [raw_image_collection_plan.log.md](archive/raw_image_collection_plan.log.md) \[[EN](archive/raw_image_collection_plan.en.log.md)\] | `Implemented` | Two-system (--collect + Cache Sidecar) raw image saving plan |
| [trajectory_deviation_corrective_cleanup_plan.log.md](archive/trajectory_deviation_corrective_cleanup_plan.log.md) | `Validated` | L2 post-hoc cleanup：三类妥协 10 commits 三波落地（squashed into 633acd8），Verify V1/V2/V3 全绿 |

### Design Only / Background Analysis

| File | Status | Description |
|------|--------|-------------|
| [key_dim_reduction_recommendations.log.md](archive/key_dim_reduction_recommendations.log.md) \[[EN](archive/key_dim_reduction_recommendations.en.log.md)\] | `Historical` | Two-layer pipeline (token pooling + dim projection) recommendations, 未进入实现路径 |
| [libero_env_init_analysis.log.md](archive/libero_env_init_analysis.log.md) \[[EN](archive/libero_env_init_analysis.en.log.md)\] | `Historical` | LIBERO env init analysis: main.py 仅使用 3 params，初始状态为预存固定集 |

### Historical

| File | Status | Description |
|------|--------|-------------|
| [doc_cleanup_plan.log](archive/doc_cleanup_plan.log) \[[EN](archive/doc_cleanup_plan.en.log)\] | `Historical` | Documentation cleanup plan |

---

## Maintenance Rules

> **AGENT: READ FIRST** — Log status system and lifecycle rules are defined in [`WORKING_AGREEMENT.md` §5 Log Management](../WORKING_AGREEMENT.md#5-log-management). The Working Agreement is authoritative.
