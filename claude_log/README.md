# claude_log/

Implementation plans and development records. For project architecture docs, see [`docs/README.md`](../docs/README.md).

## Status Legend

| Status | Meaning |
|--------|---------|
| `Plan` | Task breakdown or proposed workflow; not implemented yet |
| `Design Only` | Technical design exists, but implementation is not confirmed |
| `In Progress` | Actively being executed or updated |
| `Implemented` | Code and log exist, but final validation or sign-off is still pending |
| `Validated` | Implemented and explicitly verified |
| `Historical` | Kept for record; not the active source of truth for ongoing work |
| `⚠️ Done·High-Risk` | Code landed and functional, but no test coverage; known risk points remain; pending regression validation |

## Cache System Implementation

| File | Status | Description |
|------|--------|-------------|
| [step1.log](step1.log) | `Validated` | Stage 1/2/3 public interface design, Pi0 vs Pi0.5 architectural comparison |
| [step2.log](step2.log) | `Validated` | CUDA Event timing system: per-component timing strategy and API design |
| [step3_data_collection.log](step3_data_collection.log) | `Implemented` | Forward-hook based HDF5 data collection: hook points, schema, wrapper ordering |
| [step3_cache.log](step3_cache.log) | `⚠️ Done·High-Risk` | Cache storage layer abstraction + implementation: VectorStoreBackend ABC, named vectors, Qdrant chunked RRF; no test coverage, see `src/openpi/cache/README.md` |
| [step4_discussion.log.md](step4_discussion.log.md) | `Historical` | Step 4 discussion log: stability analysis, design debates, Q&A review and scope convergence |
| [step4_plan.log.md](step4_plan.log.md) | `Historical` | Step 4 full plan: original implementation summary + revised detailed plan after review |
| [step4_test_plan.log.md](step4_test_plan.log.md) | `Validated` | Step 4 test plan + execution: 6 files, 45 test cases all passed |
| [step4_config_discussion.log.md](step4_config_discussion.log.md) | `Validated` | Step 4 Config discussion: SearchStrategy abstraction, per-checkpoint config, YAML format, component-config decoupling principles |
| [step4_config_plan.log.md](step4_config_plan.log.md) | `Implemented` | Step 4 Config implementation plan: SearchStrategy component + Config dataclass tree + YAML loading + serve_policy.py integration |
| step4 (overall) | `⚠️ Done·High-Risk` | Orchestrator skeleton: CP1 end-to-end loop + CP3 skeleton; unstable components listed in `src/openpi/cache/README.md` |

## Cache Key 降维

| File | Status | Description |
|------|--------|-------------|
| [key_dim_reduction_recommendations.log.md](key_dim_reduction_recommendations.log.md) | `Design Only` | Cache key 降维方法推荐：两层流水线（Token 池化 + 维度投影），含实施路径 |
| [cache_experiment_plan.log.md](cache_experiment_plan.log.md) | `Plan` | Cache 实验方案组合：仅 CP1，Gate/Judge 固定 always-on，4 种降维 × 2 种跨模态融合（Weighted RRF / Weighted Score Sum） |

## Trajectory Search

| File | Status | Description |
|------|--------|-------------|
| [trajectory_search_requirements.log.md](trajectory_search_requirements.log.md) | `Plan` | 轨迹搜索需求：双向链表数据结构、query 历史暂存、三层 similarity 融合、YAML 配置扩展 |
| [trajectory_search_impl_plan.log.md](trajectory_search_impl_plan.log.md) | `Plan` | 轨迹搜索实现计划：7 Phase 分步实现，文件变更清单、代码细节、验收标准 |

## CLIP KeyBuilder

| File | Status | Description |
|------|--------|-------------|
| [clip_key_builder_plan.log.md](clip_key_builder_plan.log.md) | `Plan` | CLIP KeyBuilder 实现计划：open_clip 视觉编码器替代 SigLIP pooling 生成低维 cache key，YAML 可配置模型 |

## Retrieval System

| File | Status | Description |
|------|--------|-------------|
| [qdrant_design.log](qdrant_design.log) | `Design Only` | Qdrant collection schema: named vectors vs multivector, payload structure |
| [qdrant_step_knn_experiment_plan.log](qdrant_step_knn_experiment_plan.log) | `Plan` | Step-KNN retrieval experiment: candidate generation, scoring, evaluation |
| [faiss_uv_toolchain_plan.log](faiss_uv_toolchain_plan.log) | `Plan` | GPU Faiss build commands for uv environment |

## Concurrent Inference

| File | Status | Description |
|------|--------|-------------|
| [concurrent_inference_plan.log.md](concurrent_inference_plan.log.md) | `Implemented` | 并发推理方案：服务端多连接 + 客户端多 worker 线程池 |

## LIBERO 环境分析

| File | Status | Description |
|------|--------|-------------|
| [libero_env_init_analysis.log.md](libero_env_init_analysis.log.md) | `Design Only` | LIBERO 环境初始化分析：当前 main.py 仅用3个参数，大量可用选项未暴露；初始状态为预存固定集合 |

## 原始图片收集

| File | Status | Description |
|------|--------|-------------|
| [raw_image_collection_plan.log.md](raw_image_collection_plan.log.md) | `Plan` | 两套系统（--collect + Cache Sidecar）原始图片保存方案：解耦设计、image_mask 过滤、实施步骤 |

## Meta

| File | Status | Description |
|------|--------|-------------|
| [doc_cleanup_plan.log](doc_cleanup_plan.log) | `In Progress` | This documentation cleanup plan |

## Maintenance Rule

During development, keep the `Status` column in this index aligned with reality whenever a log changes role, moves from design to implementation, or becomes historical.

When a task is near completion, explicitly remind the user to manually confirm the final `Status` before treating the log entry as finalized.
