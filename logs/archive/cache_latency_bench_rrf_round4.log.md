# Research Log — weighted_rrf 渐进调优 · Round 4：batched keybuilder 复用

> **Authority**: Execution（Owner 工作流 override G1/G2）· **Type**: 性能调优（exp 内，零改 src，纯复用）· **Status**: In Progress（RRF Round 4 完成 / 进 Round 5）
> **前序**: RRF Round 3 [`rrf_round3`](cache_latency_bench_rrf_round3.log.md)

---

## 复用 R4 batched keybuilder（与 fusion 正交）

build 段优化（keybuilder spatial_pool）与检索 fusion 无关，故已 ship + 已审查的 `CP1SpatialPool16BatchedKeyBuilder`（R4）对 weighted_rrf **直接复用**。`run_rrf_latency.py` 加 `--batched` flag 叠加 `attach_batched_pool_keybuilder`。零新代码。

## 结果（lean_rrf + batched）

| | lean_rrf | +batched | |
|---|---|---|---|
| build median | 1.198ms | **0.465ms** | 2.6× |
| search median | 3.817ms | 3.821ms | 不变（batched 只动 build）|
| total median | 5.661ms | **4.859ms** | |
| **hit_rate** | 0.3852 | **0.3852** | 不变 = query_key bit-equal → 检索不变 |

batched keybuilder 的 query_key bit-equal 已由 weighted_sum R4 审查 + 单测证明（与 fusion 无关）；RRF 的 hit_rate 不变是检索等价的间接证据（kinematic judge 按动作因子打档，RRF hit_rate≠1.0 是 judge 特性非检索）。

## 文件

`opt/run_rrf_latency.py`（+`--batched` flag）。零新 backend。

## 后续

- R5: RrfReleaseVisionBackend（MRO 组合 ReleaseVision + LeanRRF）。
- R6: 收口 e2e + 对标 weighted_sum 报告。
