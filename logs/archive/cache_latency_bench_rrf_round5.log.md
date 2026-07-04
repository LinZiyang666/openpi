# Research Log — weighted_rrf 渐进调优 · Round 5：RrfReleaseVision（MRO 零代码组合）

> **Authority**: Execution（Owner 工作流 override G1/G2）· **Type**: 性能调优 / 内存（exp 内，零改 src，纯组合）· **Status**: In Progress（RRF Round 5 完成 / 进 Round 6）
> **前序**: RRF Round 4 [`rrf_round4`](cache_latency_bench_rrf_round4.log.md)

---

## RrfReleaseVisionBackend —— 多继承零代码组合

`RrfReleaseVisionBackend(ReleaseVisionBackend, LeanRRFBackend)`（`opt/safe_release_rrf_backend.py`），**纯 MRO 组合**两个已 ship + 已审查的 backend，class body 仅 docstring、零方法。MRO：`Rrf → ReleaseVision → LeanRRF → LeanSearch → PrenormDot → ...`，解析：`release_vision`/`_compute_field_scores`(guard) ← ReleaseVision、`_search_weighted_rrf` ← LeanRRF、`_lean_bucket`/`_prebuild` ← 共享祖先。`__init__` 链经每层 super() 正确初始化所有计数器。inject 加 `attach_rrf_release`。

## 全栈结果（lean_rrf + release + batched）

search 3.81 / build 0.495 / **total 4.88ms** / hit_rate 0.3852（不变）/ 释放 ~1GB 生效。

## 等价 + 安全

- 单测 4/4：MRO 解析 + **释放后独立 query search bit-same** + winner match LeanRRF + 守卫继承 raise（cross-bucket fallback）。
- **关键**：release 释放 entry.query_keys 的 vision（哨兵），但 lean_rrf search 用 `spec.query_keys`（keybuilder 独立 query）+ `self._mat`（未释放），`_lean_bucket` 只用 `candidate.id`，**从不读 entry.query_keys 内容** → 真实 replay（query 来自 keybuilder）零影响。守卫拦 release 后 fallback（cross-bucket → ReleaseUnsafeError）。
- 审查 APPROVE：`MRO SAFE` / `RELEASE GUARD COVERS RRF FALLBACK` / `SRC UNTOUCHED` YES。独立证 top_k>1 fallback bit-same + cross-bucket raise。

## 文件

`opt/safe_release_rrf_backend.py`（新，MRO 组合）、`opt/inject.py`（+`attach_rrf_release`）、`opt/run_rrf_latency.py`（+`rrf_release`）、`tests/exp/test_rrf_release_backend.py`（4）。

## 后续

- R6: 收口总报告 [`rrf_final_report`](cache_latency_bench_rrf_final_report.log.md)。
