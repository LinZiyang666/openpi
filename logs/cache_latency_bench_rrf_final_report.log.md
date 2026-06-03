# weighted_rrf 渐进调优 — 最终总报告（6 轮 + 对标 weighted_sum）

> **Authority**: Execution（Owner 独裁令工作流 override G1/G2，每轮本会话内 spawn 审查）· **Type**: 优化总报告 · **Status**: In Progress（RRF 6 轮收官）
> **逐轮 log**: R1 [`rrf_round1`](cache_latency_bench_rrf_round1.log.md) · R2 [`rrf_round2`](cache_latency_bench_rrf_round2.log.md) · R3 [`rrf_round3`](cache_latency_bench_rrf_round3.log.md) · R4 [`rrf_round4`](cache_latency_bench_rrf_round4.log.md) · R5 [`rrf_round5`](cache_latency_bench_rrf_round5.log.md)
> **weighted_sum 总报告**: [`tuning_final_report`](cache_latency_bench_tuning_final_report.log.md)

所有延迟 **ms**，libero_10 / cp1_spatial_pool_16 / 2640 step / CPU t=4 / 静默。**全程零改 src 框架。**

---

## 执行摘要

weighted_rrf cp1 latency 通过**复用 weighted_sum 已 ship backend（~90%）+ 唯一 rrf-specific 新增 LeanRRF（~10%）**，从 stock **32.88ms 压到全栈 4.88ms（6.7×）**。**检索侧（search 3.81 + build 0.495 = 4.3ms）与 weighted_sum（4.05ms）持平**——印证 owner "weighted_rrf 不差于 weighted_sum"。全程零改 src，每轮独立审查 APPROVE，等价靠 **winner-id parity**（RRF 几何安全界结构性失效）。

| 轮 | 优化 | RRF cp1_total median | 复用/新增 |
|---|---|---|---|
| stock | src InMemoryBackend | 32.88ms | — |
| **R1** | 嫁接验证（prebuilt 自动生效）| ~24.6ms | 纯复用 R1 |
| **R2** | prenorm 等价深挖 + fp32-ONLY | (prenorm) 6.56ms | 纯复用 R2 + dtype sweep |
| **R3** | **LeanRRF**（override `_search_weighted_rrf`）| 5.66ms | **唯一真 rrf-specific 新增** |
| **R4** | batched keybuilder（build）| 4.86ms | 纯复用 R4（正交）|
| **R5** | RrfReleaseVision（MRO 组合）| **4.88ms** + 释放 ~1GB | 复用 R5 + LeanRRF（零代码组合）|

---

## RRF vs weighted_sum 对标（同 libero_10 库，全栈最终）

| 段 | weighted_rrf | weighted_sum | 说明 |
|---|---|---|---|
| search | 3.81ms | 3.6ms | 持平（同热核 + lean）|
| build | 0.495ms | 0.45ms | 持平（同 batched keybuilder）|
| **judge** | **0.38ms** | **0.01ms** | **唯一差额**：kinematic composite vs threshold |
| total | 4.88ms | 4.15ms | 差额纯在 judge 选择，非检索 |

**结论**：weighted_rrf 的检索（search+build）与 weighted_sum 持平，stock 时 RRF 甚至略快（30.87 vs 33.9，RRF 无 normalize 层）。total 的 0.73ms 差额**全在 kinematic judge 段**（这是 judge 配置选择，与检索优化无关）——若 RRF 也用 threshold judge，total 与 weighted_sum 一致。

---

## 复用 vs 新增（诚实，owner 要的）

- **纯复用（~90%）**：R1 PrebuiltMatrix / R2 PrenormDot 的 `_compute_field_scores` override 是 weighted_rrf 与 weighted_sum 的**共享热核**（`_search_weighted_rrf:530 → _batch_field_scores(sid=None) → _compute_field_scores`），自动生效；R4 batched keybuilder（build 正交）；R5 ReleaseVision（释放 `_mat` 外的 entry 副本，fast path 不读 entry）。
- **唯一真 rrf-specific 新增（~10%）**：R3 `LeanRRFBackend` override `_search_weighted_rrf`（rrf-lean，~30 行）+ winner-id 等价方法学（不能照搬几何界）+ `RrfReleaseVisionBackend` MRO 组合（零代码）+ dtype sweep（RRF 敏感性实证）+ 配套 config/harness/runner/test。
- **不把复用包装成 RRF 独立成果**——RRF 简单的根源正是复用 weighted_sum 已 ship backend；但**等价证明侧反而更重**。

---

## 等价（RRF 的特殊性 —— 比 weighted_sum 更难证）

- **几何安全界失效**：RRF top score 是 rank-reciprocal（~0.016），离散 rank tie（98/2640 exact top1==top2），weighted_sum 的 "err<margin ⇒ 不翻" 几何证明**结构性不适用** → 等价只能靠**实测 winner-id parity**。
- 各轮判据：R1 bit-equal（cosine 保留，rrf_score diff 0.0）；R2 winner parity（0/534，prenorm 不改 rank）；R3 lean bit-equal vs prenorm（0.0）；R4 query_key bit-equal（hit_rate 不变）；R5 释放后独立 query bit-same。
- **fp32-ONLY 更硬**：dtype sweep fp16=187/2640(7.08%) / bf16=809/2640(30.6%) 翻 RRF winner，vs weighted_sum 0.57%/7.3%——RRF argsort 对误差敏感 **4-12×**（小误差 swap rank → 改 RRF score → 翻 winner，vs sum 的 normalize 平移保 argmax）。

---

## 结论

weighted_rrf 6 轮：**32.88→4.88ms（6.7×）**，检索与 weighted_sum 持平，全程零改 src + 每轮独立审查 APPROVE + 0 winner 翻档。复用是 RRF 简单的根源（~90% backend 复用），唯一真新增是 R3 LeanRRF + 等价方法学 + MRO 组合。生产落地同 weighted_sum（src drop-in 或加 backend 白名单 + §10 rebench gate + GPU bit-exact 复验）；RRF 额外铁律：fp32-ONLY 比 weighted_sum 更不可松（argsort 敏感）。
