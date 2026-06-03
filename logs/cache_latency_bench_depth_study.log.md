# Research Log — CP1 六段延迟 ~ trajectory_depth 受控扫描（libero_10, 3-key）

> **Authority**: Execution · **Level**: L2 · **Status**: Code 完成 / 待 G2
> **Plan**: [`cache_latency_bench_depth_study_plan.log.md`](cache_latency_bench_depth_study_plan.log.md)（G1 APPROVED R2）
> **基础设施**: [`cache_latency_bench_plan.log.md`](cache_latency_bench_plan.log.md)

所有延迟单位 **毫秒 (ms)**，每段 `median (p95)`。

---

## 方法

唯一自变量 `trajectory_depth ∈ {1,3,4,5}`，4 个派生 yaml 仅差 `trajectory_depth` + `trajectory_weights`（strip 这两键后 4 份 md5 一致 = 受控变量严格相同）。固定：3-key（vision_0/1 + robot_state）、threshold 0.997697、相似度、归一化、libero_10 库（`cp1_spatial_pool_16.pkl`, 1.1G, brute_force）、50 episode / 2640 step 全量回放、repeats=1、`OMP_NUM_THREADS=MKL_NUM_THREADS=8`、CPU。每 run 由 `replay.py` 经 orchestrator 自带 `SystemTimer` 采集 CP1 六段延迟。

---

## 结果

### ALL 桶 — median (p95) ms

| 段 | d1 | d3 | d4 | d5 | 随 depth |
|---|---|---|---|---|---|
| cp1_collect | 0.006 (0.007) | 0.005 (0.007) | 0.006 (0.008) | 0.006 (0.007) | 平 |
| cp1_gate | 0.002 (0.003) | 0.002 (0.003) | 0.003 (0.004) | 0.003 (0.004) | 平 |
| cp1_build | 1.136 (2.088) | 1.050 (1.784) | 1.124 (1.994) | 1.058 (1.906) | 平 |
| **cp1_search** | **37.549 (100.969)** | **35.774 (84.201)** | **38.996 (92.403)** | **36.825 (85.155)** | **无趋势** |
| cp1_judge | 0.011 (0.015) | 0.011 (0.016) | 0.013 (0.018) | 0.012 (0.017) | 平 |
| cp1_fetch | 0.004 (0.006) | 0.004 (0.006) | 0.004 (0.006) | 0.004 (0.006) | 平 |
| cp1_total | 39.364 (102.474) | 37.107 (85.562) | 40.539 (94.409) | 38.119 (86.791) | 无趋势 |

Steady-state（step_idx≥4）cp1_search：39.752 / 35.776 / 39.098 / 36.896 ms（与 ALL 同结论）。

### Hit counts（ALL，固定 threshold）

| run | n_steps | FULL_HIT | WARM_START | MISS |
|---|---|---|---|---|
| d1 | 2640 | 2640 | 0 | 0 |
| d3 | 2640 | 2590 | 0 | 50 |
| d4 | 2640 | 2540 | 0 | 100 |
| d5 | 2640 | 2490 | 0 | 150 |

---

## 结论

**1. 延迟绝对集中在 search 段**：cp1_search ≈ 37ms 占 cp1_total（≈39ms）的 **~95%**；其余五段合计 < 1.2ms。瓶颈是后端对 1.1G 库的全量 brute_force 相似度（vision 32768 维 cosine × 全库候选）。

**2. 主假设 H1 被数据否证**：H1 预测「search 随 depth 单调上升」。实测 search 在 36–39ms 间 **无单调趋势**（d4 最高、d1 > d3）。
机制（plan T3 score-memo，smoke 阶段已实测坐实）：后端 `_search_with_trajectory` 的 `(session_id, query_id)` score-memo 把主导成本——单次全库 cosine——**跨步复用**；加深 trajectory 的历史层大多命中 memo、不重算，故 depth 的边际成本被摊销至噪声级。**在「大库 brute_force + score-memo」regime 下，CP1 延迟由「单次全库 cosine」主导，而非 trajectory 深度。** 这是一个有价值的负面结果。

**3. 验收 §8.4 诊断（逆序非失败）**：search steady d1=39.75 > d3=35.78（逆序 10% > ε=5%）。诊断：① 跨 depth 差异（3.2ms）远小于单 run 段内 p50→p95 跨度（37→100ms）；② d4 重跑 search median 39.00→43.21ms（run-to-run +11%，loadavg 0.62 低负载）。即 **search median 自身 run-to-run 噪声 ~10% > 跨 depth 差异**，逆序源于「depth 无关 + 测量噪声」，非 host 负载尖峰。其余四段（collect/gate/build/judge）跨 depth 漂移亦在噪声内（H1 sanity check 通过：无段随 depth 系统漂移）。

**4. 附带语义发现**：固定 threshold 下 **MISS 数随 depth 线性增**（d1=0 → d3=50 → d4=100 → d5=150，每档 +50 = 每 episode +1）。trajectory 越深越改变聚合 score 分布，把部分步压到阈值下。这是 depth 的*语义*效应（非延迟），被受控设计干净暴露——延迟与 hit 率两条线在此分离。

---

## 已知偏差（plan §3/§7）

- **T1** depth=1 走单步 `_search_weighted_score_sum`、depth≥2 走 `_search_with_trajectory`（backend 两条函数）；d1 是单步基线，d1→d3 有函数级 break。但结论（depth 不影响 search 延迟）在两条路径下一致成立。
- **T2** episode 前 depth-1 步为 depth 爬升期；steady-state（step_idx≥4）表与 ALL 表结论一致，爬升期未改变结论。
- **T3** score-memo 跨步复用 → 见结论 2（正是它令 H1 证伪）。
- CPU build 段不含 GPU→CPU D2H（基础设施偏差1）；仅驱动 CP1（偏差2）；绝对 ms 与 CPU 负载相关，结论用段间相对占比 / depth 标度，非跨机绝对值（偏差4）。

数据（gitignore，不入库）：`exp/cache_latency_bench/data/depth_study/{d1,d3,d4,d5}/`、`comparison.json`、`compare_report.md`。
