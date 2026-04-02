# claude_log/

Implementation plans and development records. For project architecture docs, see [`docs/README.md`](../docs/README.md).

## Cache System Implementation

| File | Description |
|------|-------------|
| [step1.log](step1.log) | Stage 1/2/3 public interface design, Pi0 vs Pi0.5 architectural comparison |
| [step2.log](step2.log) | CUDA Event timing system: per-component timing strategy and API design |
| [step3_data_collection.log](step3_data_collection.log) | Forward-hook based HDF5 data collection: hook points, schema, wrapper ordering |

## Retrieval System

| File | Description |
|------|-------------|
| [qdrant_design.log](qdrant_design.log) | Qdrant collection schema: named vectors vs multivector, payload structure |
| [qdrant_step_knn_experiment_plan.log](qdrant_step_knn_experiment_plan.log) | Step-KNN retrieval experiment: candidate generation, scoring, evaluation |
| [faiss_uv_toolchain_plan.log](faiss_uv_toolchain_plan.log) | GPU Faiss build commands for uv environment |

## Meta

| File | Description |
|------|-------------|
| [doc_cleanup_plan.log](doc_cleanup_plan.log) | This documentation cleanup plan |
