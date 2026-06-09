# cp1 Latency 调优 — 最终报告

cp1 cache 检索 + build latency 渐进调优的**最终总报告**（`artifact_layout.md`: 最终报告
`.md` → `exp/<exp>/analysis/`）。逐轮 research log、早期研究、plan 都是工作期间产物，留在
`logs/`（见下）。全程**零改 src 框架** —— 所有优化是 `InMemoryBackend` / keybuilder 的
exp 层子类，经 `components_hook` 注入 bench。

## 最终报告

| 报告 | 摘要 |
|------|------|
| [tuning_final_report.md](tuning_final_report.md) | **weighted_sum 6 轮**：cp1_total 35.49→**4.15ms (8.5×)** + 释放 1GB；Tier 2(fp16)/3(降维) 评估否决；portable 分析 + 生产落地建议 |
| [rrf_final_report.md](rrf_final_report.md) | **weighted_rrf 6 轮**：32.88→**4.88ms (6.7×)**；检索与 weighted_sum 持平（差额纯在 kinematic judge）；复用 ~90% + LeanRRF ~10%；fp32-ONLY 更硬 |

## 实验产物去向

- **脚本**：[`../opt/`](../opt/)（backend/keybuilder 子类 + harness + runner）· **config**：[`../config/`](../config/) · **data**：`../data/`（gitignored）
- **逐轮 research log**（工作期间产物）：`logs/cache_latency_bench_{round1..5,rrf_round1..5}.log.md`
- **早期研究**：`logs/cache_latency_bench_{search_optimization_report,depth_study,depth_study_rrf_kin}.log.md`
- **基础设施 plan**：`logs/cache_latency_bench_{plan,depth_study_plan}.log.md`
