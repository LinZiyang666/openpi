# R11 分析中期稿 / ANALYSIS INTERIM

在仓库根目录运行以下唯一命令，即可重新扫描已完成实验、更新全部表格、图和本报告：

```bash
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r11/astra/analysis/run.py
```

## 给负责人的说明（中期）

本次仅纳入四个指定目录中已完成的 42/94 个实验，每个实验 500 个回合。未完成或尚未同步到本机的实验不计入结果。不同方法完成比例不同，不能把现在的总体均值当作最终排名。

| 方法 | 已完成／计划 | 实际调用成本减目标 | 误差不超过零点零二 |
|---|---|---|---|
| 关闭额外调用 | 2/8 | — | 不适用 |
| 随机调用 | 8/17 | -0.0061 | 8/8 |
| 定期调用 | 8/17 | -0.0044 | 8/8 |
| 距离触发 | 5/17 | -0.1145 | 0/5 |
| 误差评分与随机混合 | 8/17 | -0.0289 | 3/8 |
| 自适应误差混合 | 5/8 | -0.0122 | 4/5 |
| 邻居分歧触发 | 3/4 | -0.0374 | 0/3 |
| 定期调用并追加一段 | 2/4 | -0.0205 | 1/2 |
| 随机调用并追加一段 | 1/2 | -0.0162 | 1/1 |

距离触发的主要问题是运行中的分数低于原校准分数，因而很少越过固定阈值。误差评分加入随机部分后，少调用的幅度减小；自适应方法会提高调用概率，但每回合从原始设定重新开始，短回合可能结束于尚未补足预算的状态。这些是冻结方法的测量结果，没有据此改阈值或重新训练。

这里把两种解释分开检查：一是可检索示范从原来的五分之四增加到全库，会让检索距离变小；二是闭环执行缓存动作后，状态可能更贴近已有示范。新增回放在未进入检索库的另一批示范上固定全部检索参数，只改变可检索示范的数量，因此第一项可以独立测量。直接用库里的示范本身查询全库只是“非常贴近示范”的极端参考，不能当作第二项的因果证明。

独立示范回放中，仅增加可检索示范造成的距离阈值越线率下降为 0.12–2.08 个百分点，不足以单独解释目前较大的欠支。最终检索参数与留出校准参数也不同；现有数据支持分布转移，但不能证明剩余差距都由闭环跟随示范造成。

成功率必须在实际调用成本相同处比较。主表只在两条曲线实际重叠的范围内插值；缺少点或没有重叠就标为待定。从关闭状态连到最低调用点的比较另列为次要分析，其结论依赖线性插值。当前不能宣布某一种状态评分已稳定优于随机或定期调用。

下一轮的建议是用不进入检索库的示范、完整检索库和最终预测器校准分数，再让一个全局、与任务无关的成本反馈器控制实际预算。状态分数负责决定把预算花在哪里，随机部分负责保留基本调用机会。只靠离线阈值无法保证闭环成本精确达标；原有强制保护成本超过目标时，额外调用只能关闭。接近纯策略成功率的单元默认关闭额外调用。以上均为未来假设。

## Technical scope and provenance

Snapshot started **2026-10-03T09:09:59.856589+00:00**, report generated **2026-10-03T09:10:15.959216+00:00**. Completed: 42 test arms and 20 separate dev arms. Data arriving after discovery waits for the next invocation. `r11_knob_3/4` are never read. `r11_local_idg` is used only for the requested hardware check. All numbers are reconstructed directly from journals, summaries and accepted decision logs; no coordinator aggregate is needed.

| Root | Completed | Planned |
|---|---|---|
| r11_knob_1 | 23 | 23 |
| r11_knob_2 | 2 | 24 |
| r11_local_k3 | 15 | 23 |
| r11_local_k4 | 2 | 24 |

Completion requires summary and journal to agree on 500 accepted, error-free episode outcomes (50 in dev). Decision logs are filtered to the accepted attempt and deduplicated by episode and step. Log counts must equal the cost ledger and episode steps must be contiguous; diagnostics are suppressed when they fail. The manifest records source hashes for summaries/journals, decision-file sizes/mtimes before and after reading, replay hashes, and exclusions. Caches are invalidated by source metadata changes. Original calibration artifacts and run roots are read-only.

Logs reconciled: 62/62 analyzed test+dev arms. All owner costs use `IR = c_v V/N + (1-c_v) M/N`, with `c_v=.152` for pi05 and `.148` for groot. This is neither measured latency nor the newer stage-pricing field in summary.json. N counts requested five-control slots; terminal slots may be partial.

## Realized IR and exact accounting decomposition

| Method | n / planned | IR − target | IR − library | within ±.02 | max abs error |
|---|---|---|---|---|---|
| random | 8/17 | -0.0061 | -0.0061 | 8/8 | 0.0189 |
| periodic | 8/17 | -0.0044 | -0.0044 | 8/8 | 0.0120 |
| distance | 5/17 | -0.1145 | -0.1145 | 0/5 | 0.1552 |
| error_hybrid | 8/17 | -0.0289 | -0.0289 | 3/8 | 0.0441 |
| adaptive_error_hybrid | 5/8 | -0.0122 | -0.0122 | 4/5 | 0.0220 |
| disagreement | 3/4 | -0.0374 | -0.0374 | 0/3 | 0.0437 |
| periodic_pgt1 | 2/4 | -0.0205 | -0.0205 | 1/2 | 0.0227 |
| random_tail2 | 1/2 | -0.0162 | -0.0162 | 1/1 | 0.0162 |

For each arm, let `v=V/N`, `g=G/V`, and `k=K/(V-G)`, with K counting knob-only calls. Then `IR=v[c_v+c_m(g+(1-g)k)]`. The table changes library v to live v, then library g to live g, then library conditional knob probability to the logged live probability, then probability to sampled calls. This ordered telescoping decomposition is exact; the attribution depends on this stated order. For adaptive arms the reference is the frozen 128-seed replay, and the score/controller term includes dynamic dose changes. `r11_knob` is the pre-OR sample and can overlap a guard; `os_r11_knob_call` is the additional call. They are not interchangeable.

| Method | n | look-rate contribution | guard contribution | score/controller contribution | coin contribution |
|---|---|---|---|---|---|
| distance | 5 | -0.0000 | -0.0072 | -0.1072 | +0.0000 |
| error_hybrid | 8 | +0.0005 | -0.0092 | -0.0204 | +0.0002 |
| adaptive_error_hybrid | 5 | +0.0001 | -0.0073 | -0.0037 | -0.0013 |
| disagreement | 3 | +0.0006 | -0.0134 | -0.0246 | +0.0000 |

Distance: mean IR error -0.1145; the logged score-probability component is -0.1072, guard component -0.0072, and look-rate component -0.0000. Zero coin contribution for deterministic distance thresholds and the probability reconstruction checks rule out an incorrectly implemented threshold as the observed explanation. This decomposition identifies changed score selection, not its causal origin.

| Arm | SR | IR | target | pred | v live/lib | g live/lib | k live/lib | Δv | Δg | Δscore/control | Δcoin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r11_pi05_l10_50_adaptive_error_hybrid_ir40 | 0.898 | 0.3938 | 0.40 | 0.4004 | 0.5036/0.5043 | 0.2126/0.2569 | 0.6733/0.6732 | -0.0005 | -0.0062 | -0.0001 | 0.0002 |
| r11_pi05_l10_50_error_hybrid_ir40 | 0.902 | 0.3690 | 0.40 | 0.4000 | 0.5038/0.5043 | 0.2011/0.2569 | 0.6051/0.6718 | -0.0003 | -0.0078 | -0.0228 | 0.0000 |
| r11_groot_l10_50_distance_ir40 | 0.796 | 0.3103 | 0.40 | 0.4000 | 0.5029/0.5029 | 0.3058/0.3156 | 0.3525/0.6492 | -0.0000 | -0.0015 | -0.0882 | 0.0000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | 0.930 | 0.3780 | 0.40 | 0.3996 | 0.5101/0.5104 | 0.1052/0.1664 | 0.6604/0.6943 | -0.0003 | -0.0081 | -0.0094 | -0.0038 |
| r11_pi05_l10_50_distance_ir32 | 0.870 | 0.2119 | 0.32 | 0.3200 | 0.5036/0.5043 | 0.2038/0.2569 | 0.1420/0.4201 | -0.0004 | -0.0132 | -0.0945 | 0.0000 |
| r11_pi05_spatial_50_error_hybrid_ir32 | 0.972 | 0.2796 | 0.32 | 0.3200 | 0.5123/0.5090 | 0.0873/0.1506 | 0.4131/0.4844 | 0.0021 | -0.0142 | -0.0317 | 0.0034 |
| r11_groot_spatial_50_disagreement_ir32 | 0.914 | 0.2777 | 0.32 | 0.3200 | 0.5107/0.5104 | 0.1022/0.1664 | 0.4034/0.4747 | 0.0002 | -0.0147 | -0.0279 | 0.0000 |
| r11_pi05_l10_200_error_hybrid_ir25 | 0.920 | 0.2306 | 0.25 | 0.2500 | 0.5044/0.5042 | 0.1615/0.1896 | 0.2364/0.2665 | 0.0001 | -0.0088 | -0.0103 | -0.0005 |
| r11_groot_l10_500_error_hybrid_ir25 | 0.896 | 0.2311 | 0.25 | 0.2500 | 0.5035/0.5036 | 0.2391/0.2742 | 0.1655/0.1857 | -0.0000 | -0.0122 | -0.0046 | -0.0020 |
| r11_groot_spatial_50_error_hybrid_ir25 | 0.902 | 0.2084 | 0.25 | 0.2500 | 0.5090/0.5104 | 0.1205/0.1664 | 0.2118/0.2816 | -0.0007 | -0.0143 | -0.0262 | -0.0004 |
| r11_groot_spatial_50_distance_ir25 | 0.862 | 0.1670 | 0.25 | 0.2500 | 0.5090/0.5104 | 0.1367/0.1664 | 0.0863/0.2816 | -0.0007 | -0.0092 | -0.0731 | 0.0000 |
| r11_pi05_spatial_50_distance_ir40 | 0.976 | 0.2448 | 0.40 | 0.4000 | 0.5115/0.5090 | 0.1016/0.1506 | 0.3157/0.7026 | 0.0019 | -0.0063 | -0.1508 | 0.0000 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.826 | 0.3763 | 0.40 | 0.4000 | 0.5031/0.5029 | 0.2980/0.3156 | 0.5785/0.6492 | 0.0002 | -0.0026 | -0.0202 | -0.0010 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.980 | 0.3559 | 0.40 | 0.4000 | 0.5118/0.5090 | 0.0776/0.1506 | 0.6106/0.7026 | 0.0021 | -0.0094 | -0.0392 | 0.0024 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.974 | 0.3817 | 0.40 | 0.3996 | 0.5111/0.5090 | 0.0775/0.1506 | 0.6762/0.7016 | 0.0016 | -0.0095 | -0.0085 | -0.0016 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.892 | 0.3083 | 0.32 | 0.3206 | 0.5034/0.5043 | 0.2057/0.2569 | 0.4248/0.4219 | -0.0006 | -0.0126 | 0.0011 | -0.0001 |
| r11_groot_spatial_50_distance_ir32 | 0.868 | 0.1838 | 0.32 | 0.3200 | 0.5091/0.5104 | 0.1399/0.1664 | 0.1281/0.4747 | -0.0008 | -0.0060 | -0.1293 | 0.0000 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.980 | 0.2763 | 0.32 | 0.3200 | 0.5125/0.5090 | 0.0863/0.1506 | 0.4051/0.4844 | 0.0022 | -0.0144 | -0.0315 | 0.0000 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.858 | 0.3080 | 0.32 | 0.3200 | 0.5038/0.5034 | 0.2589/0.2755 | 0.3844/0.4097 | 0.0002 | -0.0042 | -0.0079 | -0.0002 |
| r11_pi05_l10_50_disagreement_ir32 | 0.880 | 0.2938 | 0.32 | 0.3200 | 0.5035/0.5043 | 0.2114/0.2569 | 0.3771/0.4201 | -0.0005 | -0.0113 | -0.0145 | 0.0000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.778 | 0.3170 | 0.32 | 0.3198 | 0.5030/0.5029 | 0.3146/0.3156 | 0.3669/0.3757 | 0.0001 | -0.0003 | -0.0014 | -0.0012 |

Complete per-arm values for every comparator and the separate dev cohort are in [arms.csv](analysis/arms.csv). No calibration parameter is changed by this script.

## Per-arm score distributions and frozen thresholds

Reference quantiles below use actual even-step calibration looks; the original threshold/CDF was fitted on **all rows**. The CSV also includes the all-row quantiles, non-guard quantiles, CDF ranks, support tails, dose saturation and threshold exceedance. Static probability reconstruction is checked against `os_r11_p`. Dynamic arms have a changing rank cutoff `1-q`, so no fixed threshold is invented.

| Arm | live q05/q50/q95 | B-look q05/q50/q95 | threshold | live >t | B looks >t | live median rank |
|---|---|---|---|---|---|---|
| r11_pi05_l10_50_adaptive_error_hybrid_ir40 | 0.030 / 0.142 / 0.387 | 0.049 / 0.174 / 0.393 | — | — | — | 0.366 |
| r11_pi05_l10_50_error_hybrid_ir40 | 0.030 / 0.141 / 0.386 | 0.049 / 0.174 / 0.393 | 0.130 | 0.554 | 0.694 | 0.359 |
| r11_groot_l10_50_distance_ir40 | 6.903 / 9.406 / 22.950 | 7.720 / 11.189 / 20.990 | 9.882 | 0.427 | 0.703 | 0.231 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | 0.075 / 0.173 / 0.347 | 0.106 / 0.213 / 0.474 | — | — | — | 0.307 |
| r11_pi05_l10_50_distance_ir32 | 6.043 / 9.389 / 18.984 | 7.175 / 12.405 / 31.065 | 12.456 | 0.197 | 0.496 | 0.209 |
| r11_pi05_spatial_50_error_hybrid_ir32 | 0.036 / 0.167 / 0.425 | 0.097 / 0.230 / 0.450 | 0.225 | 0.312 | 0.506 | 0.270 |
| r11_groot_spatial_50_disagreement_ir32 | 0.045 / 0.167 / 0.486 | 0.053 / 0.190 / 0.612 | 0.190 | 0.425 | 0.501 | 0.415 |
| r11_pi05_l10_200_error_hybrid_ir25 | 0.022 / 0.097 / 0.274 | 0.030 / 0.110 / 0.300 | 0.154 | 0.221 | 0.281 | 0.419 |
| r11_groot_l10_500_error_hybrid_ir25 | 0.059 / 0.134 / 0.349 | 0.060 / 0.144 / 0.373 | 0.242 | 0.160 | 0.193 | 0.446 |
| r11_groot_spatial_50_error_hybrid_ir25 | 0.072 / 0.172 / 0.362 | 0.106 / 0.213 / 0.474 | 0.273 | 0.152 | 0.312 | 0.304 |
| r11_groot_spatial_50_distance_ir25 | 9.253 / 12.635 / 25.575 | 12.845 / 17.330 / 30.678 | 19.324 | 0.125 | 0.321 | 0.046 |
| r11_pi05_spatial_50_distance_ir40 | 8.288 / 13.244 / 25.276 | 11.026 / 18.480 / 34.296 | 15.068 | 0.332 | 0.733 | 0.160 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.087 / 0.209 / 0.503 | 0.108 / 0.230 / 0.481 | 0.198 | 0.538 | 0.648 | 0.405 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.034 / 0.169 / 0.434 | 0.097 / 0.230 / 0.450 | 0.168 | 0.504 | 0.714 | 0.282 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.039 / 0.170 / 0.419 | 0.097 / 0.230 / 0.450 | — | — | — | 0.289 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.025 / 0.137 / 0.378 | 0.049 / 0.174 / 0.393 | — | — | — | 0.341 |
| r11_groot_spatial_50_distance_ir32 | 9.320 / 12.662 / 25.973 | 12.845 / 17.330 / 30.678 | 17.282 | 0.165 | 0.506 | 0.046 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.039 / 0.166 / 0.565 | 0.054 / 0.196 / 0.608 | 0.193 | 0.420 | 0.511 | 0.395 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.071 / 0.161 / 0.399 | 0.080 / 0.171 / 0.398 | 0.190 | 0.376 | 0.411 | 0.456 |
| r11_pi05_l10_50_disagreement_ir32 | 0.031 / 0.136 / 0.519 | 0.034 / 0.152 / 0.542 | 0.164 | 0.405 | 0.454 | 0.440 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.085 / 0.207 / 0.499 | 0.108 / 0.230 / 0.481 | — | — | — | 0.397 |

![Score CDFs](analysis/score_cdfs.png)

Dev remains a separate, non-test cohort:

| Method | n | dev IR − target |
|---|---|---|
| distance | 4 | -0.0986 |
| error_hybrid | 4 | -0.0306 |

## Separating donor count, deployment transfer, and closed-loop state shift

No model is fitted in these replays. The serialized deployment metric, action scale, pooled predictor and each frozen knob setting remain fixed. Selected-library queries are compared in three forms: exclusion of the whole calibration fold; exclusion of only their entire own episode; and full donors including the query itself. The first two isolate donor eligibility under the same metric, but the metric has seen selected queries. The last is an on-demo limit with exact-self leakage, explicitly not independent value evidence.

For size 50, whole-fold exclusion and whole-episode exclusion leave the same four episodes per task. They therefore cannot identify a clean “4/5 to full donor” effect for a novel query. Instead, use ten deterministically sampled **unused parent-B episodes per task** (100 per eligible cell, failures retained), projected through the frozen selected-library PCA. Each exact query is replayed against all five reduced donor masks and against all donors. Queries are never donors or fitting data for these deployment fits. The metric, action scale, predictor and query path are held fixed. At size 500 no unused B episodes exist; only the disclosed selected-library ablations are available. No task-specific layer-4 settings are introduced; task identity is used only for inherited lower-layer retrieval and episode matching.

| Cell | independent external B episodes | selected episodes |
|---|---|---|
| groot_l10_200 | 100 | 200 |
| groot_l10_50 | 100 | 50 |
| groot_l10_500 | 0 | 500 |
| groot_spatial_50 | 100 | 50 |
| pi05_l10_200 | 100 | 200 |
| pi05_l10_50 | 100 | 50 |
| pi05_spatial_50 | 100 | 50 |

**Donor-count result:** with independent B queries and the deployment metric fixed, expanding donors lowers distance threshold exceedance by only 0.12–2.08 pp across these completed distance arms. This is much smaller than their crossfit-reference to live selection gaps. The simple donor-count explanation is therefore insufficient on the tested independent B states. This does not identify the remaining gap as demo tracking: fold metric/scale fitting, query distribution and guard overlap are separate changes.

| Arm | external 4/5 p | external full p | Δp full−4/5 | live fixed-setting p | live median / external full median |
|---|---|---|---|---|---|
| r11_pi05_l10_50_adaptive_error_hybrid_ir40 | 0.5478 | 0.5410 | -0.0067 | 0.5064 | 0.142 / 0.158 |
| r11_pi05_l10_50_error_hybrid_ir40 | 0.6659 | 0.6572 | -0.0087 | 0.6212 | 0.141 / 0.158 |
| r11_groot_l10_50_distance_ir40 | 0.4738 | 0.4531 | -0.0208 | 0.4274 | 9.406 / 9.618 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | 0.5527 | 0.5382 | -0.0145 | 0.5193 | 0.173 / 0.182 |
| r11_pi05_l10_50_distance_ir32 | 0.3179 | 0.3089 | -0.0090 | 0.1971 | 9.389 / 10.501 |
| r11_pi05_spatial_50_error_hybrid_ir32 | 0.4447 | 0.4402 | -0.0045 | 0.4182 | 0.167 / 0.184 |
| r11_groot_spatial_50_disagreement_ir32 | 0.4902 | 0.4369 | -0.0533 | 0.4254 | 0.167 / 0.171 |
| r11_pi05_l10_200_error_hybrid_ir25 | 0.3021 | 0.2936 | -0.0084 | 0.2508 | 0.097 / 0.109 |
| r11_groot_l10_500_error_hybrid_ir25 | — | — | — | 0.1768 | 0.134 / — |
| r11_groot_spatial_50_error_hybrid_ir25 | 0.2581 | 0.2456 | -0.0125 | 0.2320 | 0.172 / 0.182 |
| r11_groot_spatial_50_distance_ir25 | 0.1116 | 0.1105 | -0.0012 | 0.1253 | 12.635 / 13.886 |
| r11_pi05_spatial_50_distance_ir40 | 0.4484 | 0.4404 | -0.0080 | 0.3325 | 13.244 / 14.381 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.5932 | 0.5876 | -0.0056 | 0.5935 | 0.209 / 0.207 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.6407 | 0.6424 | +0.0016 | 0.6140 | 0.169 / 0.184 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.5711 | 0.5716 | +0.0005 | 0.5473 | 0.170 / 0.184 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.2296 | 0.2246 | -0.0050 | 0.1985 | 0.137 / 0.158 |
| r11_groot_spatial_50_distance_ir32 | 0.1945 | 0.1910 | -0.0035 | 0.1651 | 12.662 / 13.886 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.4824 | 0.4459 | -0.0366 | 0.4196 | 0.166 / 0.174 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.4059 | 0.3978 | -0.0081 | 0.3951 | 0.161 / 0.162 |
| r11_pi05_l10_50_disagreement_ir32 | 0.4510 | 0.4245 | -0.0265 | 0.4052 | 0.136 / 0.142 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.1557 | 0.1529 | -0.0028 | 0.1505 | 0.207 / 0.207 |

Here p is a **score-selection** probability, unconditioned on guard. For adaptive rows it is evaluated at the frozen **initial dose only**, not a simulated adaptive IR. The paired full-minus-reduced score estimates and episode-bootstrap intervals are in [library_replays.csv](analysis/library_replays.csv). Five donor masks are averaged per query and are not treated as independent episodes. Distances under a fixed geometry cannot increase when donors are added; this monotonicity is asserted. Disagreement and predicted error need not be monotone.

| Arm | external 4/5 guard | external full guard | external 4/5 IR | external full IR | ΔIR (fixed B paths) |
|---|---|---|---|---|---|
| r11_pi05_l10_50_adaptive_error_hybrid_ir40 | 0.2671 | 0.2634 | 0.3523 | 0.3490 | -0.0033 |
| r11_pi05_l10_50_error_hybrid_ir40 | 0.2671 | 0.2634 | 0.3911 | 0.3876 | -0.0035 |
| r11_groot_l10_50_distance_ir40 | 0.2868 | 0.2917 | 0.3253 | 0.3189 | -0.0064 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | 0.1367 | 0.1387 | 0.3350 | 0.3308 | -0.0043 |
| r11_pi05_l10_50_distance_ir32 | 0.2671 | 0.2634 | 0.2680 | 0.2641 | -0.0039 |
| r11_pi05_spatial_50_error_hybrid_ir32 | 0.0793 | 0.0783 | 0.2869 | 0.2853 | -0.0017 |
| r11_groot_spatial_50_disagreement_ir32 | 0.1367 | 0.1387 | 0.3091 | 0.2892 | -0.0199 |
| r11_pi05_l10_200_error_hybrid_ir25 | 0.2383 | 0.2356 | 0.2680 | 0.2642 | -0.0038 |
| r11_groot_l10_500_error_hybrid_ir25 | — | — | — | — | — |
| r11_groot_spatial_50_error_hybrid_ir25 | 0.1367 | 0.1387 | 0.2250 | 0.2218 | -0.0031 |
| r11_groot_spatial_50_distance_ir25 | 0.1367 | 0.1387 | 0.1705 | 0.1709 | +0.0004 |
| r11_pi05_spatial_50_distance_ir40 | 0.0793 | 0.0783 | 0.2866 | 0.2836 | -0.0030 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.2868 | 0.2917 | 0.3768 | 0.3753 | -0.0015 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.0793 | 0.0783 | 0.3659 | 0.3668 | +0.0009 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.0793 | 0.0783 | 0.3378 | 0.3384 | +0.0006 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.2671 | 0.2634 | 0.2523 | 0.2494 | -0.0029 |
| r11_groot_spatial_50_distance_ir32 | 0.1367 | 0.1387 | 0.1999 | 0.2000 | +0.0001 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.0793 | 0.0783 | 0.3007 | 0.2860 | -0.0147 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.2820 | 0.2826 | 0.3181 | 0.3156 | -0.0025 |
| r11_pi05_l10_50_disagreement_ir32 | 0.2671 | 0.2634 | 0.3192 | 0.3093 | -0.0099 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.2868 | 0.2917 | 0.2430 | 0.2441 | +0.0011 |

The preceding IRs include reconstructed no-progress guards and terminal cadence on these **fixed B episode paths**, with the unchanged thresholds. They are additional measurement diagnostics, not test results or closed-loop predictions. Adaptive rows keep q fixed at q0 for this contrast; they must not be read as a replay of the adaptive controller.

For disagreement, donor expansion has a more visible effect than for distance: the per-arm probability and fixed-path IR changes above should be considered alongside the live guard-rate contribution. For error_hybrid, the random half limits the cost effect of a shifted threshold distribution. The final-head-only transfer table below is small and often positive, so it does not by itself explain the systematic negative hybrid cost error.

The selected-query/full-metric ablations in the CSV also show large metric/scale-transfer shifts even with the reduced donor mask. For size 50 the reduced and leave-one-episode arrays are identical. Because the full fitted metric has seen these selected queries, their low scores cannot serve as an honest calibration replacement. The independent external-B experiment avoids that exposure. Batched float32 distance reductions agree with serving arithmetic away from exact-self zero within the checked tolerance; near-zero self distances are roundoff-sensitive (see numerical_validation.json).

The full-minus-reduced external-B comparison identifies the donor-eligibility component on those fixed B states. The difference between external full-library B and closed loop is consistent with state-distribution shift, but also includes B-vs-A episode composition, termination weighting and feedback. Decision logs lack the complete visual query keys needed for a full donor ablation on the **same live state**. Therefore these data do not identify a causal percentage of the total undershoot due to “following demos”. Lower live scores alone do not prove that mechanism.

Changing cross-fitted error heads to the final deployed head is a third transfer, not a donor-count effect:

| Arm | frozen IR | final head on same OOF features IR |
|---|---|---|
| r11_pi05_l10_50_error_hybrid_ir40 | 0.4000 | 0.4048 |
| r11_pi05_spatial_50_error_hybrid_ir32 | 0.3200 | 0.3306 |
| r11_pi05_l10_200_error_hybrid_ir25 | 0.2500 | 0.2508 |
| r11_groot_l10_500_error_hybrid_ir25 | 0.2500 | 0.2488 |
| r11_groot_spatial_50_error_hybrid_ir25 | 0.2500 | 0.2566 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.4000 | 0.3970 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.4000 | 0.4033 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.3200 | 0.3191 |

## Why adaptation can still undershoot

The script verifies every committed-ledger update. With gain η=.2, the cost error processed before the final proposal is `(c_m/η)(q0−q_last+Σ clipping_adjustment)`. Add the cost error of the final unprocessed segment for the exact whole-episode error. An upward change from q0 to q_last is therefore a finite-episode underspend term; resetting at every episode prevents carrying that deficit forward. This is an accounting explanation, not a proposed changed controller evaluated on test.

| Arm | q0 | mean final q | IR initial→final | IR clipping | IR final segment | max update error |
|---|---|---|---|---|---|---|
| r11_pi05_l10_50_adaptive_error_hybrid_ir40 | 0.5701 | 0.6660 | -0.0071 | -0.0020 | 0.0029 | 0.0000000000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | 0.6155 | 0.7660 | -0.0281 | -0.0044 | 0.0106 | 0.0000000000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.6528 | 0.7804 | -0.0253 | -0.0058 | 0.0127 | 0.0000000000 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.2332 | 0.4853 | -0.0187 | 0.0053 | 0.0017 | 0.0000000000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.1558 | 0.3844 | -0.0155 | 0.0103 | 0.0022 | 0.0000000000 |

## Success versus realized IR

![Realized-IR frontier](analysis/frontier.png)

Every cell-size plot includes completed methods, same-batch knob-off where available, and the historical R8 pure-policy point. Hollow diamonds show the historical R10 knob-off configuration, with common y-axis limits within each suite. Historical points do not substitute for missing same-batch controls in comparisons. Curves are piecewise linear through measured IR points, ordered by realized IR rather than target labels. Partial grids can have fewer than the planned three points.

Main matched comparisons use the midpoint of each pair of observed common-support intervals. Both methods need at least two measured points; single-point disagreement ablations instead appear in the per-arm comparator interpolation table. No extrapolation is performed. Intervals use 2000 paired episode bootstrap draws: the same evaluation episode draw is reused across all targets, nested sizes and models in a suite. Realized IR and interpolation weights are recalculated in each draw; unsupported draws are excluded and counted. These exploratory intervals describe finite-episode variation conditional on the observed rollout per arm; they do not capture repeated-policy-run variance or adjust for multiple comparisons.

| Cell | method − comparator | IR | ΔSR pp | 95% interval pp | valid bootstrap draws |
|---|---|---|---|---|---|
| groot_l10_200 | periodic − random | 0.2802 | +1.77 | [-1.06, 4.61] | 2000 |
| groot_l10_50 | periodic − random | 0.3604 | +2.09 | [-1.25, 5.38] | 2000 |
| pi05_spatial_50 | error_hybrid − periodic | 0.3178 | -0.22 | [-1.55, 1.08] | 2000 |

| Method − comparator | cells | mean ΔSR pp | paired 95% interval pp | valid draws |
|---|---|---|---|---|
| error_hybrid − periodic | 1 | -0.22 | [-1.55, 1.08] | 2000 |
| periodic − random | 2 | +1.93 | [-0.30, 4.14] | 2000 |

Unsupported main comparisons: 60/63. See [matched_frontier.csv](analysis/matched_frontier.csv) for ranges and missingness.

Per-arm interpolation against completed random/periodic points, with explicit labeling when a same-batch off anchor is needed:

| Arm | comparator | type | ΔSR pp | 95% interval pp | valid draws |
|---|---|---|---|---|---|
| r11_pi05_l10_50_error_hybrid_ir40 | periodic | secondary off-anchored | +0.80 | [-2.41, 4.20] | 2000 |
| r11_groot_l10_50_distance_ir40 | periodic | arm vs comparator interpolation | -0.48 | [-4.30, 3.83] | 2000 |
| r11_pi05_l10_50_distance_ir32 | random | secondary off-anchored | +1.48 | [-1.88, 4.68] | 2000 |
| r11_pi05_l10_50_distance_ir32 | periodic | secondary off-anchored | +1.49 | [-1.60, 4.88] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -0.32 | [-2.20, 1.48] | 2000 |
| r11_groot_spatial_50_disagreement_ir32 | random | arm vs comparator interpolation | +0.53 | [-2.78, 3.92] | 2000 |
| r11_pi05_spatial_50_distance_ir40 | periodic | arm vs comparator interpolation | +0.35 | [-1.57, 2.01] | 1750 |
| r11_groot_l10_50_error_hybrid_ir40 | random | arm vs comparator interpolation | +1.65 | [-2.09, 5.33] | 2000 |
| r11_groot_l10_50_error_hybrid_ir40 | periodic | arm vs comparator interpolation | -0.70 | [-4.48, 2.83] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir40 | periodic | arm vs comparator interpolation | -0.11 | [-1.67, 1.41] | 2000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | periodic | arm vs comparator interpolation | -0.91 | [-2.70, 0.83] | 2000 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | periodic | secondary off-anchored | +1.30 | [-1.71, 4.14] | 2000 |
| r11_pi05_spatial_50_disagreement_ir32 | periodic | arm vs comparator interpolation | +0.50 | [-1.00, 2.13] | 2000 |
| r11_groot_l10_200_error_hybrid_ir32 | random | arm vs comparator interpolation | +2.01 | [-1.81, 5.65] | 1999 |
| r11_groot_l10_200_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -2.23 | [-5.57, 1.70] | 1895 |
| r11_pi05_l10_50_disagreement_ir32 | periodic | secondary off-anchored | +0.46 | [-2.70, 3.57] | 2000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -2.61 | [-6.36, 1.30] | 2000 |

Pooled efficiency over available **same-batch off** controls is `Σ(SR−SR_off)/Σ(IR−IR_off)`. The shared bootstrap preserves reuse of the off episodes across targets. Different methods currently cover different cells/targets, so these ratios are descriptive, not a balanced method ranking:

| Method | arms / cells | ΔSR sum | ΔIR sum | SR gain per IR | paired 95% interval |
|---|---|---|---|---|---|
| random | 3/2 | 0.0600 | 0.2517 | 0.238 | [-0.068, 0.551] |
| periodic | 3/2 | 0.1360 | 0.4067 | 0.334 | [0.138, 0.536] |
| distance | 1/1 | 0.0260 | 0.0450 | 0.578 | [-0.141, 1.606] |
| error_hybrid | 2/2 | 0.0960 | 0.3205 | 0.300 | [0.123, 0.496] |
| adaptive_error_hybrid | 2/1 | 0.1020 | 0.3683 | 0.277 | [0.100, 0.474] |
| disagreement | 1/1 | 0.0360 | 0.1268 | 0.284 | [-0.015, 0.608] |
| periodic_pgt1 | 1/1 | 0.0300 | 0.1347 | 0.223 | [-0.055, 0.509] |
| random_tail2 | 1/1 | 0.0240 | 0.1369 | 0.175 | [-0.084, 0.455] |

No efficiency winner can yet be identified across the grid: method coverage differs, off controls are incomplete, and the state-method advantages in the matched-IR comparisons have intervals overlapping zero. A high ratio on a single low-spend arm is especially unstable and is not a reason to adopt that method.

The proposed periodic advantage of +0.6 percentage points is not established by a point estimate alone. Use the periodic−random common-support interval above; missing cells and intervals spanning zero prevent a general claim.

## Where the extra calls landed

Episode progress is observed decision step divided by the final episode length minus one. It is retrospective measurement, never an input to layer 4. Successful/failed labels are eventual outcomes, so conditional call rates are associations, not recovery effects. [placement.csv](analysis/placement.csv) gives per-arm progress quintiles and eventual-outcome groups. [guard_neighborhood.csv](analysis/guard_neighborhood.csv) compares eligible looks within two looks of a guard, farther away, and immediately after a guard.

| Arm | extra calls near guard | cache-only run median | cache-only run p95 |
|---|---|---|---|
| r11_pi05_l10_50_adaptive_error_hybrid_ir40 | 0.434 | 1.0 | 2.0 |
| r11_pi05_l10_50_error_hybrid_ir40 | 0.451 | 1.0 | 4.0 |
| r11_groot_l10_50_distance_ir40 | 0.703 | 2.0 | 9.0 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | 0.230 | 1.0 | 3.0 |
| r11_pi05_l10_50_distance_ir32 | 0.529 | 4.0 | 16.0 |
| r11_pi05_spatial_50_error_hybrid_ir32 | 0.226 | 2.0 | 5.0 |
| r11_groot_spatial_50_disagreement_ir32 | 0.256 | 2.0 | 5.0 |
| r11_pi05_l10_200_error_hybrid_ir25 | 0.430 | 2.0 | 8.0 |
| r11_groot_l10_500_error_hybrid_ir25 | 0.592 | 2.0 | 7.0 |
| r11_groot_spatial_50_error_hybrid_ir25 | 0.269 | 2.0 | 8.0 |
| r11_groot_spatial_50_distance_ir25 | 0.336 | 5.0 | 12.0 |
| r11_pi05_spatial_50_distance_ir40 | 0.226 | 4.0 | 10.6 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.626 | 1.0 | 4.0 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.204 | 1.0 | 4.0 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.184 | 1.0 | 2.0 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.431 | 1.0 | 4.0 |
| r11_groot_spatial_50_distance_ir32 | 0.316 | 5.0 | 12.0 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.227 | 2.0 | 7.0 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.597 | 1.0 | 5.0 |
| r11_pi05_l10_50_disagreement_ir32 | 0.453 | 2.0 | 7.0 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.571 | 1.0 | 4.0 |

Cache-only stretches count consecutive vision anchors with no policy call, including censored start/end runs. At nominal cadence one anchor spans ten controls (two five-control slots); a policy tail is not counted as cache-only.

## Hardware identity checks

| Model | status | local SR | remote SR | ΔSR pp [95%] | local IR / remote IR |
|---|---|---|---|---|---|
| pi05 | available | 0.844 | 0.826 | +1.80 [-1.00, 4.60] | 0.1669 / 0.1680 |
| groot | pending; one required arm missing | — | — | — [—, —] | — / — |

The pi05 comparison is local R11 knob-off versus historical H100 R10 distance-corrected control; it mixes hardware and batch. The groot duplicate compares the explicitly requested local identity run with the H100 R11 off run once synchronized. A small paired difference is not an equivalence test, and missing data cannot establish hardware identity.

## Future-round hypothesis: a library-only mapping that aims to hit target

1. Reserve whole B-pool episodes **outside** the deployed donor subset (or acquire additional B-only episodes at size 500). Use the complete deployment donor bank, exact deployed metric/action scale, and final pooled score predictor to generate calibration references. Preserve whole-episode cadence and mandatory guard logic. Never self-query a donor episode to claim an honest target match.

2. Fit one cell-wide score CDF and an initial dose from these external B sequences only; validate on a separate B split. Use `k_target=clip((target/v−c_v−c_m*g)/(c_m*(1−g)),0,1)` as an accounting constraint, not as a task-specific rule. Check the guard floor and all-call ceiling. Correct head/CDF transfer by calibrating the actual final head.

3. Retain a random floor and drive total cost with one global committed-ledger feedback state. For a future design, consider a persistent deficit budget across episodes or an analytically faster startup so repeated resets do not systematically lose the first part of each episode. Use bounded probabilities and anti-windup when guards force overspend. Tune any gain/floor/reset convention only on B sequences, freeze it, and measure a new untouched test round. Adaptation changes placement and feasibility: no library-only calculation guarantees exact cost on unknown short closed-loop trajectories.

4. If accurate budget delivery matters before state-specific value is demonstrated, use the measured random/periodic schedules as the operational reference and keep state-dependent placement experimental. Leave the knob off by default in near-pure cells. Adopt a state component only after a balanced, matched-realized-IR success comparison supports it; proxy error capture is insufficient.

Historical baseline context for that default (not substituted into the same-batch efficiency calculation):

| Cell | R10 off SR | R8 pure SR | off − pure pp | R10 off IR |
|---|---|---|---|---|
| pi05_l10_50 | 0.826 | 0.908 | -8.20 | 0.1680 |
| pi05_l10_200 | 0.904 | 0.908 | -0.40 | 0.1500 |
| pi05_l10_500 | 0.894 | 0.908 | -1.40 | 0.1504 |
| pi05_spatial_50 | 0.910 | 0.988 | -7.80 | 0.1353 |
| groot_l10_50 | 0.754 | 0.898 | -14.40 | 0.2071 |
| groot_l10_200 | 0.820 | 0.898 | -7.80 | 0.1913 |
| groot_l10_500 | 0.906 | 0.898 | +0.80 | 0.1754 |
| groot_spatial_50 | 0.896 | 0.940 | -4.40 | 0.1269 |

In this grid, pi05 L10-200/500 and groot L10-500 are the clearest **default-off candidates** from historical near-pure baselines (gaps of −0.4, −1.4, and +0.8 pp respectively). The unfinished same-batch off controls are still needed before a final adoption decision. These are cell-size recommendations, with no task-indexed configuration.

No item above is a refitted test result. No thresholds, fits, arm settings, serving code, run roots or running processes were changed.

## Executed script and artifacts

[Single entry script](analysis/run.py) produces [provenance.json](analysis/provenance.json), [inventory.csv](analysis/inventory.csv), [arms.csv](analysis/arms.csv), [method_summary.csv](analysis/method_summary.csv), [library_replays.csv](analysis/library_replays.csv), [matched_frontier.csv](analysis/matched_frontier.csv), [matched_arms.csv](analysis/matched_arms.csv), [pooled_frontier.csv](analysis/pooled_frontier.csv), [efficiency.csv](analysis/efficiency.csv), [hardware.json](analysis/hardware.json), placement tables and the two PNG figures. [Numerical validation](analysis/numerical_validation.json) checks serving feature and distance arithmetic. The command at the top was executed. Rerunning performs only CPU measurements and refreshes this **interim** report; it does not promote the output to a final analysis automatically.
