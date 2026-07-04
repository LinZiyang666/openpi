# Research Log — weighted_rrf 渐进调优 · Round 3：LeanRRF（framework 开销消除）

> **Authority**: Execution（Owner 工作流 override G1/G2，本会话内审查）· **Type**: 性能调优（exp 内，零改 src，**唯一真 rrf-specific 新 backend**）· **Status**: In Progress（RRF Round 3 完成 / 进 Round 4）
> **前序**: RRF Round 2 [`rrf_round2`](cache_latency_bench_rrf_round2.log.md)

---

## roofline 定 gap（先验证再决定）

RRF lean probe（中位桶 N=258）：full `_search_weighted_rrf` 5.21ms vs lean(GEMV+argsort+RRF) 3.07ms+filter = 3.51ms → **framework gap 1.70ms（>1ms）→ 做 LeanRRF**（不降级）。gap 来自 `_batch_field_scores` wrapper + `_compute_field_scores` zeros+scatter + `mask.nonzero`，与 weighted_sum LEAN 同形。argsort（O(NlogN)）是 RRF 本征成本，lean 不救。

## LeanRRFBackend（唯一真 rrf-specific 新增）

`LeanRRFBackend(LeanSearchBackend)`，override `_search_weighted_rrf`：稳态（top_k==1 + 整桶有序 + 全 field 快路）→ 直接 per-field `mat.mv`(cosine)/`torch.norm`(l2) → argsort → `weight/(rrf_k+rank)` → topk(1)，跳 wrapper/scatter；否则 super()（Round-2 prenorm RRF）。

**继承 `LeanSearchBackend`** 以复用 `_lean_bucket`/`_verified_buckets`（整桶判定对 RRF 同样适用）。继承的 `_search_weighted_score_sum` override（weighted_sum LEAN）在 weighted_rrf config 下不触发，无害；`_lean_rrf_hits` 独立计数。

## 结果

| | RRF cp1_search median | p95 |
|---|---|---|
| prenorm (R2) | 4.73ms | 7.16 |
| **lean_rrf (R3)** | **3.82ms** | **5.66** |

**与 weighted_sum lean 的 3.7ms 几乎一样**（RRF 把 normalize 换成 argsort，framework 精简后同水平）。

## 等价（bit-equal vs prenorm）

- 单测 4/4：lean_rrf vs PrenormDot `_search_weighted_rrf` bit-equal（同 mv→同 argsort→同 RRF）+ winner parity vs stock + fallback（partial/top_k>1）。
- self-included 整桶：winner_mismatch=0、**rrf_score_abs_diff_max=0.0**、lean_rrf_hits=534。RRF score 是离散 rank 函数，prenorm 2e-6 不改 rank（此 subset 无 swap）→ 逐 bit 同。
- fp32-ONLY（R2 dtype sweep 已锁）。

## 文件（exp/，零 src）

`opt/lean_rrf_backend.py`（新）、`opt/inject.py`（+`attach_lean_rrf`）、`opt/run_rrf_latency.py`/`compare_rrf_equivalence.py`（+lean_rrf 分支）、`tests/exp/test_lean_rrf_backend.py`（4）。

## 后续

- **R4**: 复用 batched keybuilder（attach_batched_pool_keybuilder 叠加 lean_rrf）。
- **R5**: 复用 release vision（需 `RrfReleaseVisionBackend(LeanRRFBackend)` 解继承链）。
- **R6**: 收口 e2e + 同库对标 weighted_sum 报告。
