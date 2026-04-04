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
| `⚠️ 暂时完成·高危` | 代码落地可用，但无测试覆盖，存在已知风险点，待回归验证 |

## Cache System Implementation

| File | Status | Description |
|------|--------|-------------|
| [step1.log](step1.log) | `Validated` | Stage 1/2/3 public interface design, Pi0 vs Pi0.5 architectural comparison |
| [step2.log](step2.log) | `Validated` | CUDA Event timing system: per-component timing strategy and API design |
| [step3_data_collection.log](step3_data_collection.log) | `Implemented` | Forward-hook based HDF5 data collection: hook points, schema, wrapper ordering |
| [step3_cache.log](step3_cache.log) | `⚠️ 暂时完成·高危` | Cache 存储层抽象设计+落地：VectorStoreBackend ABC、named vectors、Qdrant chunked RRF 实现；无测试覆盖，详见 `src/openpi/cache/README.md` |
| [step4_discussion.log.md](step4_discussion.log.md) | `Historical` | Step 4 讨论记录：稳定性分析、设计争议、答辩问答与范围收敛 |
| [step4_plan.log.md](step4_plan.log.md) | `Historical` | Step 4 计划全文：原始实施摘要 + 答辩后修订版详细实现计划 |
| [step4_test_plan.log.md](step4_test_plan.log.md) | `Validated` | Step 4 测试计划+执行：6个文件45用例全部通过 |
| [step4_config_discussion.log.md](step4_config_discussion.log.md) | `Validated` | Step 4 Config 讨论记录：SearchStrategy 抽象、分检查点配置、YAML 格式、组件与 Config 解耦原则 |
| [step4_config_plan.log.md](step4_config_plan.log.md) | `Implemented` | Step 4 Config 实现计划：SearchStrategy 组件 + Config dataclass 树 + YAML 加载 + serve_policy.py 改造 |
| step4 (overall) | `⚠️ 暂时完成·高危` | Orchestrator骨架：CP1端到端闭环+CP3骨架；不稳定部件见 `src/openpi/cache/README.md` |

## Retrieval System

| File | Status | Description |
|------|--------|-------------|
| [qdrant_design.log](qdrant_design.log) | `Design Only` | Qdrant collection schema: named vectors vs multivector, payload structure |
| [qdrant_step_knn_experiment_plan.log](qdrant_step_knn_experiment_plan.log) | `Plan` | Step-KNN retrieval experiment: candidate generation, scoring, evaluation |
| [faiss_uv_toolchain_plan.log](faiss_uv_toolchain_plan.log) | `Plan` | GPU Faiss build commands for uv environment |

## Meta

| File | Status | Description |
|------|--------|-------------|
| [doc_cleanup_plan.log](doc_cleanup_plan.log) | `In Progress` | This documentation cleanup plan |

## Maintenance Rule

During development, keep the `Status` column in this index aligned with reality whenever a log changes role, moves from design to implementation, or becomes historical.

When a task is near completion, explicitly remind the user to manually confirm the final `Status` before treating the log entry as finalized.
