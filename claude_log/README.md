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

## Cache System Implementation

| File | Status | Description |
|------|--------|-------------|
| [step1.log](step1.log) | `Validated` | Stage 1/2/3 public interface design, Pi0 vs Pi0.5 architectural comparison |
| [step2.log](step2.log) | `Validated` | CUDA Event timing system: per-component timing strategy and API design |
| [step3_data_collection.log](step3_data_collection.log) | `Implemented` | Forward-hook based HDF5 data collection: hook points, schema, wrapper ordering |

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
