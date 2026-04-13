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

### Cache Experiment

| File | Status | Description |
|------|--------|-------------|
| [cache_experiment_plan.log.md](cache_experiment_plan.log.md) | `In Progress` | CP1 experiment: 5 reducers (incl. CLIP) x RRF fusion, Phase 1 running |
| [cache_cp1_impl_plan.log.md](cache_cp1_impl_plan.log.md) | `Plan` | CP1 in-memory implementation plan for large-scale experiment |
| [cp1_warm_start_investigation.log.md](cp1_warm_start_investigation.log.md) | `In Progress` | CP1 warm start: noise action 可用性调查，端到端链路分析 |
| [cp1_warm_start_impl_plan.log.md](cp1_warm_start_impl_plan.log.md) | `Validated` | CP1 warm start 实现计划：4 Phase (性能修复→写入→判定+执行→文档) |
| [trajectory_deviation_experiment_plan.log.md](trajectory_deviation_experiment_plan.log.md) | `Plan` | Trajectory deviation 纠偏实验：3-Phase (离线诊断→信号分析→Oracle纠偏) |
| [trajectory_deviation_corrective_experiment.log.md](trajectory_deviation_corrective_experiment.log.md) | `Plan` | Trajectory deviation 纠偏详细实验计划：GT收集→Deviate Score→Spawn纠偏，含 trajectory depth 预填充方案 |

### Retrieval System

| File | Status | Description |
|------|--------|-------------|
| [qdrant_design.log](qdrant_design.log) | `Design Only` | Qdrant collection schema: named vectors vs multivector, payload structure |
| [qdrant_step_knn_experiment_plan.log](qdrant_step_knn_experiment_plan.log) | `Plan` | Step-KNN retrieval experiment: candidate generation, scoring, evaluation |
| [faiss_uv_toolchain_plan.log](faiss_uv_toolchain_plan.log) | `Plan` | GPU Faiss build commands for uv environment |

### Cache Key Dimensionality Reduction

| File | Status | Description |
|------|--------|-------------|
| [key_dim_reduction_recommendations.log.md](key_dim_reduction_recommendations.log.md) | `Design Only` | Two-layer pipeline (token pooling + dim projection), with implementation path |

### Redundant Token Pruning

| File | Status | Description |
|------|--------|-------------|
| [redundant_token_prune_plan.log.md](redundant_token_prune_plan.log.md) | `Implemented` | Plan A 冗余 token 剪枝：temporal scoring 两步 KeyBuilder，含 G2 审查记录。代码已合入，待实验验证 |
| [redundant_token_prune_gpt.log.md](redundant_token_prune_gpt.log.md) | `Design Only` | GPT 初步方案讨论（不含项目实现细节） |

### Stage Device Placement

| File | Status | Description |
|------|--------|-------------|
| [stage_device_placement_plan.log.md](stage_device_placement_plan.log.md) | `In Progress` | L3: 按 Stage 分离 device placement，支持 cuda/cpu/meta 三种模式。G1 已通过 |

### LIBERO

| File | Status | Description |
|------|--------|-------------|
| [libero_env_init_analysis.log.md](libero_env_init_analysis.log.md) | `Design Only` | LIBERO env init analysis: main.py uses only 3 params, initial states are pre-stored fixed sets |

### Data Collection

| File | Status | Description |
|------|--------|-------------|
| [raw_image_collection_plan.log.md](raw_image_collection_plan.log.md) | `Plan` | Two-system (--collect + Cache Sidecar) raw image saving plan |

---

## Archive

Completed and historical logs. See [`archive/`](archive/) for all files.

### Cache System Implementation

| File | Status | Description |
|------|--------|-------------|
| [step1.log](archive/step1.log) | `Validated` | Stage 1/2/3 public interface design, Pi0 vs Pi0.5 comparison |
| [step2.log](archive/step2.log) | `Validated` | CUDA Event timing system design |
| [step3_data_collection.log](archive/step3_data_collection.log) | `Implemented` | Forward-hook HDF5 data collection |
| [step3_cache.log](archive/step3_cache.log) | `Done-High-Risk` | Cache storage layer: VectorStoreBackend, Qdrant chunked RRF |
| [step4_discussion.log.md](archive/step4_discussion.log.md) | `Historical` | Step 4 discussion: stability analysis, design debates |
| [step4_plan.log.md](archive/step4_plan.log.md) | `Historical` | Step 4 full plan: original + revised after review |
| [step4_test_plan.log.md](archive/step4_test_plan.log.md) | `Validated` | Step 4 test plan: 6 files, 45 test cases passed |
| [step4_config_discussion.log.md](archive/step4_config_discussion.log.md) | `Validated` | SearchStrategy abstraction, YAML format, decoupling principles |
| [step4_config_plan.log.md](archive/step4_config_plan.log.md) | `Implemented` | Config dataclass tree + YAML loading + serve_policy.py integration |

### Feature Implementation

| File | Status | Description |
|------|--------|-------------|
| [trajectory_search_requirements.log.md](archive/trajectory_search_requirements.log.md) | `Implemented` | Trajectory search: linked list, history buffer, similarity fusion |
| [trajectory_search_impl_plan.log.md](archive/trajectory_search_impl_plan.log.md) | `Implemented` | Trajectory search: 7-phase rollout, code details |
| [clip_key_builder_plan.log.md](archive/clip_key_builder_plan.log.md) | `Implemented` | CLIP KeyBuilder: open_clip ViT-B-32 for cache keys |
| [cache_migration_guide_plan.log.md](archive/cache_migration_guide_plan.log.md) | `Implemented` | Cache framework migration tutorial plan: coupling analysis, 7-step guide, review |
| [concurrent_inference_plan.log.md](archive/concurrent_inference_plan.log.md) | `Implemented` | Server multi-connection + client multi-worker thread pool |

### Historical

| File | Status | Description |
|------|--------|-------------|
| [doc_cleanup_plan.log](archive/doc_cleanup_plan.log) | `Historical` | Documentation cleanup plan |

---

## Maintenance Rules

> **AGENT: READ FIRST** — Log status system and lifecycle rules are defined in [`WORKING_AGREEMENT.md` §5 Log Management](../WORKING_AGREEMENT.md#5-log-management). The Working Agreement is authoritative.
