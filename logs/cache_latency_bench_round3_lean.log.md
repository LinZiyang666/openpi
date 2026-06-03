# Research Log — cp1_search 渐进调优 · ROUND 3：LEAN（框架开销消除）

> **Authority**: Execution（Owner 工作流 override G1/G2，本会话内审查）· **Type**: 性能调优（exp 内，零改 src）· **Status**: In Progress（Round 3 完成 / 进 Round 4）
> **前序**: Round 2 [`cache_latency_bench_round2_prenorm_dot.log.md`](cache_latency_bench_round2_prenorm_dot.log.md)

---

## 方向修正（推翻 Round 3 草拟）

Round 3 草拟 workflow 判定"latency 已触底，转内存释放"。**实测推翻**：roofline 纯算子 median 2.55ms，但 ReplayHarness 的 cp1_search 段 median 4.70ms（干净重测确认非污染），差 **~2ms 是 backend 框架冗余**（`_batch_field_scores` hit/miss wrapper、`_compute_field_scores` 的 `torch.zeros[n]`+scatter（整桶时恒等 no-op）、`topk(1)`、dispatch）。owner 要 latency，故 Round 3 改做 **Tier 1 框架精简（LEAN）**；内存释放（692MB，草拟已出 `safe_release_backend` 详细 plan）降到后续轮。

## LEAN（chosen）

`LeanSearchBackend(PrenormDotBackend)`，只 override `_search_weighted_score_sum`：稳态（top_k==1 + 整桶有序 + 全 field 快路）→ 直接对每 field 在常驻矩阵上 `mat.mv(qn)`（cosine）/`torch.norm`（l2）→ normalize → 加权和 → `topk(1)`；否则 `super()`（完整 Round-2 路径）。`_lean_bucket` 首次 O(n) 验证整桶有序 + 缓存 `(ckpt,task,n)` + **O(1) 端点检查**（G2 MAJOR：防 reordered 桶错位）。

## 结果

| | search median | search p95 | total median | total p95 |
|---|---|---|---|---|
| R2 prenorm | 4.70ms | 7.66ms | 6.27ms | 9.74ms (贴边) |
| **R3 LEAN** | **3.7ms** | **5.56ms** | **5.22ms** | **7.66ms** ✅ |

`lean_hits=2640 / fallbacks=0`（真实回放 100% 走 LEAN）。

## 等价

- self-included 122 subset（全走 LEAN）vs 老 InMemoryBackend：`verdict_flips=0 / winner=0 / geometric_safety_max=-5.76e-4`。
- 单测：LEAN vs Round-2 full path **bit-identical**（max_score_diff=0，同算子只去 plumbing）。LOO 路径 fallback 继承 Round 2 等价。

## 审查（步骤4，本会话）

APPROVE。`CACHE SAFE: YES`（`_verified_buckets` 信任后 + 冻结库 `_filter_entries` 保序 → 无错位；endpoints 守 reordered 隐患）、`SRC UNTOUCHED: YES`、数值 bit-identical。**响应**：MAJOR（O(1) 端点检查）、MINOR（`argmax`→`topk(1)` 对齐 tie 语义 + 2 回归测试 reordered/subset-after-verify）。6 单测。

## 文件（全 exp/ + tests/，零 src 改动）

`opt/lean_search_backend.py`、`opt/inject.py`（+`attach_lean_search`）、`opt/run_round3_lean_latency.py`、`opt/compare_prenorm_equivalence.py`（+`--backend lean`）、`tests/exp/test_lean_search_backend.py`（6）。

## 后续轮

- **Round 4**: build 段优化（1.33ms / 21%，现 search 之后第二大）。keybuilder `_spatial_pool_tokens`（`key_builder.py:192`）的 `adaptive_avg_pool2d`→固定 `avg_pool2d`、减 permute、`_to_cpu_float32` 冗余拷贝；exp 层 keybuilder 子类替换 `components["key_builder"]`，保 pool 输出数值等价。
- **Round 5+**: 内存释放（692MB，`safe_release_backend` plan 已就绪）；Tier 2 fp16+rescore（带宽减半，与释放权衡）。
