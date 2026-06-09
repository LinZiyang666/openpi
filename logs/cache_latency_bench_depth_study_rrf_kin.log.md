# Research Log — CP1 六段延迟 ~ trajectory_depth 受控扫描（**weighted_rrf × kinematic judge**, libero_spatial）

> **Authority**: Execution · **Type**: 派生延迟扫描（复用 `exp/cache_latency_bench/` 基础设施，零改 src/infra） · **Status**: 完成
> **基础设施**: [`cache_latency_bench_plan.log.md`](../../../logs/cache_latency_bench_plan.log.md) · **同款 score_sum 扫描**: [`cache_latency_bench_depth_study.log.md`](cache_latency_bench_depth_study.log.md)

所有延迟单位 **毫秒 (ms)**，每段 `median (p95)`。

---

## 方法

与 score_sum 深度扫描同构，但换两处：**fusion = `weighted_rrf_knn`**（rrf_k=60，去 score_normalization），**judge = kinematic composite**（phase5 `g5_g6__fh0.3_ws0.3` 基底：`jerk_online_action` + `dispersion_online_action` 因子、zscore offline 归一化、percentile_rolling **offline 标定** `super_warmup_raw.jsonl`、composer `weighted_sum_zero_nan` fh0.49/ws0.11）。

唯一自变量 `trajectory_depth ∈ {1,3,4,5}`，4 个派生 yaml 仅差 `trajectory_depth` + `trajectory_weights`（strip 两键后 4 份 md5 一致 = 受控变量严格相同）。固定：3 个活跃字段（vision_0@0.0625 / vision_1@0.5 / robot_state@0.4375，prompt_emb enabled 但 weight0 → `_iter_active_fields` 跳过）、`cp1_spatial_pool_16` keybuilder、**libero_spatial 库**（`cp1_spatial_pool_16.pkl`, 430MB, brute_force）、libero_spatial 50 episode / 1062 step 全量回放、repeats=1、CPU。配置目录 `exp/cache_latency_bench/config/depth_study_rrf_kin/`。

**与 score_sum 扫描的差异（影响绝对值可比性）**：本扫描在 **libero_spatial**（库更小 → 每 query 候选 N 更少 → cosine 更快）上跑，故绝对 search 延迟（~12–16ms）低于 score_sum 在 libero_10 上的（~36–39ms）；这是库规模效应，**非** fusion 方法效应。两扫描只在**深度趋势的定性形状**上可比。

---

## 结果

### ALL 桶 — median (p95) ms

| 段 | d1 | d3 | d4 | d5 | 随 depth |
|---|---|---|---|---|---|
| cp1_collect | 0.006 (0.007) | 0.006 (0.007) | 0.007 (0.008) | 0.006 (0.007) | 平 |
| cp1_gate | 0.003 (0.004) | 0.003 (0.003) | 0.003 (0.004) | 0.003 (0.004) | 平 |
| cp1_build | 0.991 (1.708) | 0.995 (1.978) | 0.958 (1.721) | 0.950 (1.794) | 平 |
| **cp1_search** | **12.264 (33.033)** | **15.815 (28.012)** | **15.537 (29.243)** | **15.889 (30.956)** | **d1→d3 台阶 +3ms，后平** |
| cp1_judge | 0.434 (0.895) | 0.446 (0.952) | 0.441 (0.858) | 0.446 (0.883) | 平 |
| cp1_fetch | 0.001 (0.001) | 0.001 (0.002) | 0.001 (0.001) | 0.001 (0.001) | 平 |
| cp1_total | 13.849 (35.157) | 17.432 (30.173) | 17.146 (31.418) | 17.437 (32.995) | d1→d3 台阶 |

### Steady-state（step_idx≥4）— median (p95) ms

| 段 | d1 | d3 | d4 | d5 |
|---|---|---|---|---|
| **cp1_search** | **12.671 (34.382)** | **15.934 (27.819)** | **15.689 (29.675)** | **16.254 (30.999)** |
| cp1_total | 14.380 (37.226) | 17.525 (30.173) | 17.350 (32.059) | 17.880 (32.982) |

（其余五段与 ALL 桶一致，略。）

### Hit counts（ALL，kinematic composite judge）

| run | n_steps | FULL_HIT | WARM_START | MISS |
|---|---|---|---|---|
| d1 | 1062 | 316 | 314 | 432 |
| d3 | 1062 | 320 | 312 | 430 |
| d4 | 1062 | 317 | 312 | 433 |
| d5 | 1062 | 316 | 311 | 435 |

---

## 结论

**1. 延迟仍集中在 search 段**：cp1_search ≈ 12–16ms 占 cp1_total（≈14–18ms）的 **~88–91%**。与 score_sum 扫描的唯一段级差别：composite kinematic judge 让 cp1_judge 升到 **~0.45ms**（vs threshold judge 的 ~0.011ms，因要算 jerk/dispersion 因子 + percentile 标定），但仍远小于 search，不改瓶颈定位。

**2. 出现 score_sum 扫描没有的「d1→trajectory 台阶」**：d1（单步 `_search_weighted_rrf`）= 12.3ms，d3/d4/d5（trajectory `_search_with_trajectory` 的 RRF 分支）= 15.5–16.3ms。台阶 **~3.2ms（~25%）**，ALL 与 steady 两视图一致，且 d3/d4/d5 聚得很紧（spread ~5%）。这是 **T1 陷阱**（depth=1 与 depth≥2 走 backend 两条函数，`search_strategy.py:155`）被 RRF 显化——trajectory 路径每层做 argsort 排名融合 + `_walk_chain` + accumulate，比单步多一份**深度无关的固定开销**。score_sum 扫描里 d1–d3 差只在噪声内（~5%，d1 甚至略 > d3），RRF 把这台阶抬到噪声之上。

**3. d3→d4→d5 仍无趋势**（15.8/15.5/15.9 ALL；15.9/15.7/16.3 steady，spread ~5%）：复现 score_sum 的 **T3 score-memo 摊销**——加深 trajectory 新增的历史层大多命中 `(session,field,qid,sim_type)` memo、不重算全库 cosine，故深度的边际成本被摊到噪声级。**在「大库 brute_force + score-memo」regime 下，CP1 延迟由单步/轨迹路径选择 + 单次全库 cosine 主导，而非 trajectory 深度本身。**

**4. judge verdict 深度稳定**：hit counts 跨 depth 几乎不动（FULL ~316–320 / WARM ~311–314 / MISS ~430–435）。因 kinematic composite judge 按**动作因子值**（jerk/dispersion）打分判档，而非比检索 top_score；depth 改变的是检索 winner，对因子分布影响极小。（注：RRF top_score 本身是 rank-reciprocal 量级 ~0.016，与 judge 阈值无关。）

---

## 已知偏差 / caveat

- **repeats=1**：d1→d3 台阶（~25%）远超 d3–d5 聚类 spread（~5%），定性可信；但单 run 无法严格分离 run-to-run 噪声，台阶幅度为暗示性非定论；d3→d5 平坦复现稳健。
- **action 因子回放近似**：judge 用 `*_online_action` 因子，cache_latency_bench 对动作历史是 faithful-replay（MISS resample / WARM_START 近似，per_step.csv 的 `action_history_approx_active` 列标记，smoke 实测该列在 step≥1 置 1）。**只影响 judge verdict 段，不影响 cp1_search**（本扫描的测量目标）。
- **libero_spatial ≠ libero_10**：见方法节，绝对值与 score_sum 扫描不可直接比，只比深度趋势形状。
- CPU build 段不含 GPU→CPU D2H（基础设施偏差1）；仅驱动 CP1（偏差2）。

数据（gitignore，不入库）：`exp/cache_latency_bench/data/depth_study_rrf_kin/{depth_1,depth_3,depth_4,depth_5}/`、`comparison.json`。
