# Research Log — weighted_rrf 渐进调优 · Round 2：prenorm 等价深挖 + fp32-ONLY 定档

> **Authority**: Execution（Owner 工作流 override G1/G2，本会话内审查）· **Type**: 性能调优 / 等价验证（exp 内，零改 src，零新 backend）· **Status**: In Progress（RRF Round 2 完成 / 进 Round 3）
> **前序**: RRF Round 1 [`rrf_round1`](cache_latency_bench_rrf_round1.log.md)

---

## 目标

巩固 RRF Round 1 的 prenorm winner-id parity（更严格条件）+ 实证 RRF 对降精度的敏感性以锁定 fp32-ONLY。零新 backend（复用 PrenormDotBackend）。

## dtype sweep —— RRF 对降精度远比 weighted_sum 敏感

`rrf_dtype_sweep.py` 在真实 libero_10 task_pack（2640 LOO）上，对 fp32/fp16/bf16 prenorm-GEMV cosine 跑**忠实复刻的 RRF 融合**（per-field argsort 排名 + `weight/(rrf_k+rank)` + argmax），统计 winner 翻档 vs fp32 reference：

| dtype | RRF winner 翻档 | weighted_sum 对比 | 倍数 |
|---|---|---|---|
| **fp16** | **187/2640 (7.08%)** | 15/2640 (0.57%) | **12× 更糟** |
| **bf16** | **809/2640 (30.6%)** | 194/2640 (7.3%) | 4× 更糟 |

**根因**：RRF 用 `argsort` 排名融合，小误差（fp16 ~3e-4）足以 swap near-tie 的两个候选 rank → 改 RRF score → 翻 winner；而 weighted_sum 的 normalize+加权和对全候选误差是**平移**，保 argmax。故 **fp32-ONLY 对 RRF 是比 weighted_sum 更硬的铁律**。

## 多线程巩固 winner parity（非 schedule-specific）

| 条件 | winner_mismatch |
|---|---|
| R1: LOO threads=4 | 0 / 534 |
| R2: self-included(zero-copy) threads=4 | 0 |
| R2: LOO/self threads=1 | 0 |
| R2: self-included threads=8 | 0 |

prenorm winner parity 跨线程（1/4/8）× 跨路径（LOO index_select / self-included zero-copy）全 0-flip → 非单一 BLAS schedule artifact。

## 结论

- **fp32-ONLY 定档**（fp16/bf16 翻 RRF winner 7%/30%，远超 weighted_sum）。
- prenorm 对 RRF 的 winner parity 在 fp32 下稳健（跨线程跨路径 0-flip）。
- 几何安全界仍不适用 RRF（离散 rank tie），等价靠实测 winner-id parity。

## 文件（exp/，零 src，零新 backend）

`opt/rrf_dtype_sweep.py`（新）+ 复用 `compare_rrf_equivalence.py` 多线程。

## 后续

- **R3**: `LeanRRFBackend(PrenormDotBackend)` override `_search_weighted_rrf`（先 roofline 定 framework gap）。
- R4/R5 复用 batched keybuilder / release vision；R6 收口报告。
