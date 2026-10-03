# R11 最终分析 / FINAL ANALYSIS

在仓库根目录运行以下唯一命令，即可重新扫描已完成实验、更新全部表格、图和本报告：

```bash
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r11/astra/analysis/run.py --final
```

## 给负责人的说明（最终）

已纳入全部 94/94 个实验，每个实验 500 个回合，共 47,000 次测试。补跑的五个实验使用远端完成版本，中止副本不计入；每个实验只计算一次。另有二十个开发集实验用于检查偏差是否重复出现，不混入测试结果。

| 方法 | 已完成／计划 | 实际调用成本减目标 | 误差不超过零点零二 |
|---|---|---|---|
| 关闭额外调用 | 8/8 | — | 不适用 |
| 随机调用 | 17/17 | -0.0078 | 16/17 |
| 定期调用 | 17/17 | -0.0058 | 17/17 |
| 距离触发 | 17/17 | -0.0802 | 2/17 |
| 误差评分与随机混合 | 17/17 | -0.0276 | 6/17 |
| 自适应误差混合 | 8/8 | -0.0141 | 5/8 |
| 邻居分歧触发 | 4/4 | -0.0317 | 1/4 |
| 定期调用并追加一段 | 4/4 | -0.0121 | 3/4 |
| 随机调用并追加一段 | 2/2 | -0.0085 | 2/2 |

距离触发的主要问题是运行中的分数低于原校准分数，因而很少越过固定阈值。误差评分加入随机部分后，少调用的幅度减小；自适应方法会提高调用概率，但每回合从原始设定重新开始，短回合可能结束于尚未补足预算的状态。这些是冻结方法的测量结果，没有据此改阈值或重新训练。

这里把两种解释分开检查：一是可检索示范从原来的五分之四增加到全库，会让检索距离变小；二是闭环执行缓存动作后，状态可能更贴近已有示范。新增回放在未进入检索库的另一批示范上固定全部检索参数，只改变可检索示范的数量，因此第一项可以独立测量。直接用库里的示范本身查询全库只是“非常贴近示范”的极端参考，不能当作第二项的因果证明。

独立示范回放中，仅增加可检索示范造成的距离阈值越线率下降为 0.12–2.74 个百分点，不足以单独解释目前较大的欠支。最终检索参数与留出校准参数也不同；现有数据支持分布转移，但不能证明剩余差距都由闭环跟随示范造成。

按实际调用成本匹配后，定期调用相对随机调用的成功率差为 +2.16 [0.80, 3.57] 个百分点（5 个单元），支持定期调用优于随机调用。误差混合相对随机为 +1.19 [-0.23, 2.49] 个百分点（5 个单元），相对定期为 -0.80 [-2.27, 0.56] 个百分点（5 个单元）；这两项区间都跨过零，尚不能认定误差混合在同样花费下更好。方括号表示配对抽样区间。

邻居分歧触发相对随机的差为 +1.83 [0.10, 3.48] 个百分点（4 个单元），相对定期为 -0.02 [-1.70, 1.62] 个百分点（4 个单元）。它提示了一点有用信号，但每个单元只测了一个设置，且这些探索性区间没有校正多项比较，仍需复现。

若现在需要一个能准确控制花费的开关，优先采用定期调用。误差混合的平均成本效率最高，值得保留为候选，但它花得更少，不能把这个比值当作在相同花费下优于定期调用的证据。距离触发的预算偏差过大，不建议作为通用预算开关。自适应方案改善了预算跟踪，仍有八个实验中的三个欠支超过零点零二。

下一轮的建议是用不进入检索库的示范、完整检索库和最终预测器校准分数，再让一个全局、与任务无关的成本反馈器控制实际预算。状态分数负责决定把预算花在哪里，随机部分负责保留基本调用机会。只靠离线阈值无法保证闭环成本精确达标；原有强制保护成本超过目标时，额外调用只能关闭。这套重新校准与反馈方案仅为未来假设，未在本轮重新拟合或作为测试成绩。

是否默认关闭，应看本轮对照的成功率，不能只看示范库大小。原先一组大库的历史对照成功率为九成以上，本轮却降到约八成六；因此已撤回把它归为接近纯策略、默认关闭的建议。两项硬件对照没有显示明确的系统差异，但区间较宽，也不能解释这次基线变化。

## Technical scope and provenance

Snapshot started **2026-10-03T14:23:42.374100+00:00**, report generated **2026-10-03T14:24:12.686843+00:00**. Completed: 94 test arms and 20 separate dev arms. `r11_knob_3` is never read. `r11_knob_4` contributes the five H100 completion runs; the other nineteen listed specs are completed in `r11_local_k4`. Its aborted `r11_pi05_l10_200_random_ir25` copy has no summary and is excluded. `r11_local_idg` is used only for the requested hardware check. All numbers are reconstructed directly from journals, summaries and accepted decision logs; no coordinator aggregate is needed.

| Root | Selected complete arms | Listed specs (overlap across migrated roots) |
|---|---|---|
| r11_knob_1 | 23 | 23 |
| r11_knob_2 | 24 | 24 |
| r11_local_k3 | 23 | 23 |
| r11_local_k4 | 19 | 24 |
| r11_knob_4 | 5 | 24 |

Unique planned arms: **94**. Counts are deduplicated by arm name, not summed over root manifests. More than one complete copy of any arm fails discovery instead of silently picking an outcome.

Completion requires summary and journal to agree on 500 accepted, error-free episode outcomes (50 in dev). Decision logs are filtered to the accepted attempt and deduplicated by episode and step. Log counts must equal the cost ledger and episode steps must be contiguous; diagnostics are suppressed when they fail. The manifest records source hashes for summaries/journals, decision-file sizes/mtimes before and after reading, replay hashes, and exclusions. Caches are invalidated by source metadata changes. Original calibration artifacts and run roots are read-only.

Logs reconciled: 114/114 analyzed test+dev arms. All owner costs use `IR = c_v V/N + (1-c_v) M/N`, with `c_v=.152` for pi05 and `.148` for groot. This is neither measured latency nor the newer stage-pricing field in summary.json. N counts requested five-control slots; terminal slots may be partial.

## Realized IR and exact accounting decomposition

| Method | n / planned | IR − target | IR − library | within ±.02 | max abs error |
|---|---|---|---|---|---|
| random | 17/17 | -0.0078 | -0.0078 | 16/17 | 0.0202 |
| periodic | 17/17 | -0.0058 | -0.0058 | 17/17 | 0.0160 |
| distance | 17/17 | -0.0802 | -0.0802 | 2/17 | 0.1916 |
| error_hybrid | 17/17 | -0.0276 | -0.0276 | 6/17 | 0.0475 |
| adaptive_error_hybrid | 8/8 | -0.0141 | -0.0140 | 5/8 | 0.0235 |
| disagreement | 4/4 | -0.0317 | -0.0317 | 1/4 | 0.0437 |
| periodic_pgt1 | 4/4 | -0.0121 | -0.0121 | 3/4 | 0.0227 |
| random_tail2 | 2/2 | -0.0085 | -0.0085 | 2/2 | 0.0162 |

For each arm, let `v=V/N`, `g=G/V`, and `k=K/(V-G)`, with K counting knob-only calls. Then `IR=v[c_v+c_m(g+(1-g)k)]`. The table changes library v to live v, then library g to live g, then library conditional knob probability to the logged live probability, then probability to sampled calls. This ordered telescoping decomposition is exact; the attribution depends on this stated order. For adaptive arms the reference is the frozen 128-seed replay, and the score/controller term includes dynamic dose changes. `r11_knob` is the pre-OR sample and can overlap a guard; `os_r11_knob_call` is the additional call. They are not interchangeable.

| Method | n | look-rate contribution | guard contribution | score/controller contribution | coin contribution |
|---|---|---|---|---|---|
| distance | 17 | -0.0000 | -0.0059 | -0.0742 | +0.0000 |
| error_hybrid | 17 | +0.0002 | -0.0099 | -0.0182 | +0.0003 |
| adaptive_error_hybrid | 8 | +0.0001 | -0.0086 | -0.0046 | -0.0009 |
| disagreement | 4 | +0.0005 | -0.0102 | -0.0220 | +0.0000 |

Distance: mean IR error -0.0802; the logged score-probability component is -0.0742, guard component -0.0059, and look-rate component -0.0000. Zero coin contribution for deterministic distance thresholds and the probability reconstruction checks rule out an incorrectly implemented threshold as the observed explanation. This decomposition identifies changed score selection, not its causal origin.

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
| r11_groot_spatial_50_error_hybrid_ir40 | 0.924 | 0.3574 | 0.40 | 0.4000 | 0.5106/0.5104 | 0.1183/0.1664 | 0.6007/0.6954 | 0.0002 | -0.0064 | -0.0369 | 0.0005 |
| r11_pi05_spatial_50_distance_ir32 | 0.944 | 0.1932 | 0.32 | 0.3200 | 0.5100/0.5090 | 0.1201/0.1506 | 0.1675/0.4844 | 0.0006 | -0.0068 | -0.1206 | 0.0000 |
| r11_groot_l10_50_error_hybrid_ir32 | 0.774 | 0.3116 | 0.32 | 0.3200 | 0.5026/0.5029 | 0.3154/0.3156 | 0.3484/0.3763 | -0.0002 | -0.0001 | -0.0092 | 0.0010 |
| r11_groot_l10_50_distance_ir32 | 0.796 | 0.2643 | 0.32 | 0.3200 | 0.5029/0.5029 | 0.3091/0.3156 | 0.1941/0.3763 | 0.0000 | -0.0017 | -0.0540 | 0.0000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | 0.966 | 0.2965 | 0.32 | 0.3194 | 0.5108/0.5090 | 0.0840/0.1506 | 0.4598/0.4829 | 0.0011 | -0.0149 | -0.0109 | 0.0018 |
| r11_pi05_l10_500_distance_ir25 | 0.896 | 0.2081 | 0.25 | 0.2500 | 0.5043/0.5037 | 0.1756/0.2043 | 0.1599/0.2536 | 0.0003 | -0.0092 | -0.0330 | 0.0000 |
| r11_pi05_l10_50_distance_ir25 | 0.842 | 0.1894 | 0.25 | 0.2500 | 0.5033/0.5043 | 0.2152/0.2569 | 0.0629/0.1998 | -0.0005 | -0.0143 | -0.0458 | 0.0000 |
| r11_pi05_spatial_50_error_hybrid_ir25 | 0.960 | 0.2153 | 0.25 | 0.2500 | 0.5115/0.5090 | 0.0936/0.1506 | 0.2467/0.2935 | 0.0012 | -0.0175 | -0.0246 | 0.0062 |
| r11_groot_l10_50_error_hybrid_ir25 | 0.778 | 0.2477 | 0.25 | 0.2500 | 0.5029/0.5029 | 0.3117/0.3156 | 0.1346/0.1376 | 0.0000 | -0.0015 | -0.0007 | -0.0002 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.826 | 0.3763 | 0.40 | 0.4000 | 0.5031/0.5029 | 0.2980/0.3156 | 0.5785/0.6492 | 0.0002 | -0.0026 | -0.0202 | -0.0010 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.980 | 0.3559 | 0.40 | 0.4000 | 0.5118/0.5090 | 0.0776/0.1506 | 0.6106/0.7026 | 0.0021 | -0.0094 | -0.0392 | 0.0024 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.974 | 0.3817 | 0.40 | 0.3996 | 0.5111/0.5090 | 0.0775/0.1506 | 0.6762/0.7016 | 0.0016 | -0.0095 | -0.0085 | -0.0016 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.892 | 0.3083 | 0.32 | 0.3206 | 0.5034/0.5043 | 0.2057/0.2569 | 0.4248/0.4219 | -0.0006 | -0.0126 | 0.0011 | -0.0001 |
| r11_groot_spatial_50_distance_ir32 | 0.868 | 0.1838 | 0.32 | 0.3200 | 0.5091/0.5104 | 0.1399/0.1664 | 0.1281/0.4747 | -0.0008 | -0.0060 | -0.1293 | 0.0000 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.980 | 0.2763 | 0.32 | 0.3200 | 0.5125/0.5090 | 0.0863/0.1506 | 0.4051/0.4844 | 0.0022 | -0.0144 | -0.0315 | 0.0000 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.858 | 0.3080 | 0.32 | 0.3200 | 0.5038/0.5034 | 0.2589/0.2755 | 0.3844/0.4097 | 0.0002 | -0.0042 | -0.0079 | -0.0002 |
| r11_pi05_l10_50_disagreement_ir32 | 0.880 | 0.2938 | 0.32 | 0.3200 | 0.5035/0.5043 | 0.2114/0.2569 | 0.3771/0.4201 | -0.0005 | -0.0113 | -0.0145 | 0.0000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.778 | 0.3170 | 0.32 | 0.3198 | 0.5030/0.5029 | 0.3146/0.3156 | 0.3669/0.3757 | 0.0001 | -0.0003 | -0.0014 | -0.0012 |
| r11_groot_l10_50_distance_ir25 | 0.728 | 0.2539 | 0.25 | 0.2500 | 0.5027/0.5029 | 0.3292/0.3156 | 0.1340/0.1376 | -0.0001 | 0.0050 | -0.0010 | 0.0000 |
| r11_pi05_l10_50_error_hybrid_ir25 | 0.888 | 0.2180 | 0.25 | 0.2500 | 0.5039/0.5043 | 0.1929/0.2569 | 0.1711/0.1998 | -0.0002 | -0.0219 | -0.0094 | -0.0005 |
| r11_pi05_l10_200_distance_ir25 | 0.910 | 0.2013 | 0.25 | 0.2500 | 0.5043/0.5042 | 0.1694/0.1896 | 0.1471/0.2665 | 0.0001 | -0.0063 | -0.0424 | 0.0000 |
| r11_groot_l10_200_distance_ir25 | 0.838 | 0.2380 | 0.25 | 0.2500 | 0.5033/0.5034 | 0.2695/0.2755 | 0.1532/0.1845 | -0.0001 | -0.0021 | -0.0098 | 0.0000 |
| r11_groot_spatial_50_distance_ir40 | 0.862 | 0.2084 | 0.40 | 0.4000 | 0.5087/0.5104 | 0.1457/0.1664 | 0.1890/0.6954 | -0.0013 | -0.0027 | -0.1875 | 0.0000 |
| r11_pi05_l10_50_distance_ir40 | 0.884 | 0.2889 | 0.40 | 0.4000 | 0.5036/0.5043 | 0.2113/0.2569 | 0.3626/0.6718 | -0.0005 | -0.0064 | -0.1041 | 0.0000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | 0.852 | 0.3950 | 0.40 | 0.3997 | 0.5034/0.5029 | 0.2843/0.3156 | 0.6470/0.6480 | 0.0004 | -0.0047 | -0.0003 | -0.0000 |
| r11_groot_l10_50_disagreement_ir32 | 0.806 | 0.3056 | 0.32 | 0.3200 | 0.5031/0.5029 | 0.3139/0.3156 | 0.3284/0.3763 | 0.0002 | -0.0005 | -0.0141 | 0.0000 |
| r11_groot_spatial_50_error_hybrid_ir32 | 0.914 | 0.2725 | 0.32 | 0.3200 | 0.5095/0.5104 | 0.1165/0.1664 | 0.3821/0.4747 | -0.0006 | -0.0114 | -0.0334 | -0.0021 |
| r11_pi05_l10_50_error_hybrid_ir32 | 0.896 | 0.2824 | 0.32 | 0.3200 | 0.5037/0.5043 | 0.2012/0.2569 | 0.3515/0.4201 | -0.0004 | -0.0138 | -0.0214 | -0.0020 |
| r11_groot_l10_200_distance_ir32 | 0.846 | 0.2739 | 0.32 | 0.3200 | 0.5035/0.5034 | 0.2656/0.2755 | 0.2711/0.4097 | 0.0000 | -0.0025 | -0.0437 | 0.0000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | 0.926 | 0.2968 | 0.32 | 0.3197 | 0.5092/0.5104 | 0.1116/0.1664 | 0.4489/0.4739 | -0.0007 | -0.0125 | -0.0071 | -0.0026 |
| r11_groot_l10_200_error_hybrid_ir25 | 0.864 | 0.2390 | 0.25 | 0.2500 | 0.5038/0.5034 | 0.2490/0.2755 | 0.1783/0.1845 | 0.0002 | -0.0093 | -0.0025 | 0.0005 |
| r11_pi05_l10_500_error_hybrid_ir25 | 0.924 | 0.2279 | 0.25 | 0.2500 | 0.5043/0.5037 | 0.1612/0.2043 | 0.2295/0.2536 | 0.0003 | -0.0138 | -0.0093 | 0.0007 |
| r11_groot_l10_500_distance_ir25 | 0.892 | 0.2265 | 0.25 | 0.2500 | 0.5040/0.5036 | 0.2464/0.2742 | 0.1424/0.1857 | 0.0002 | -0.0097 | -0.0140 | 0.0000 |
| r11_pi05_spatial_50_distance_ir25 | 0.932 | 0.1729 | 0.25 | 0.2500 | 0.5107/0.5090 | 0.1240/0.1506 | 0.1095/0.2935 | 0.0008 | -0.0081 | -0.0698 | 0.0000 |

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
| r11_groot_spatial_50_error_hybrid_ir40 | 0.072 / 0.175 / 0.374 | 0.106 / 0.213 / 0.474 | 0.170 | 0.525 | 0.706 | 0.316 |
| r11_pi05_spatial_50_distance_ir32 | 8.311 / 13.327 / 25.660 | 11.026 / 18.480 / 34.296 | 18.197 | 0.192 | 0.518 | 0.163 |
| r11_groot_l10_50_error_hybrid_ir32 | 0.090 / 0.212 / 0.526 | 0.108 / 0.230 / 0.481 | 0.266 | 0.350 | 0.373 | 0.421 |
| r11_groot_l10_50_distance_ir32 | 6.883 / 9.405 / 23.007 | 7.720 / 11.189 / 20.990 | 11.572 | 0.277 | 0.448 | 0.231 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | 0.037 / 0.169 / 0.437 | 0.097 / 0.230 / 0.450 | — | — | — | 0.283 |
| r11_pi05_l10_500_distance_ir25 | 4.502 / 6.317 / 10.271 | 4.681 / 6.782 / 11.111 | 7.668 | 0.198 | 0.298 | 0.377 |
| r11_pi05_l10_50_distance_ir25 | 6.090 / 9.424 / 20.540 | 7.175 / 12.405 / 31.065 | 15.691 | 0.111 | 0.271 | 0.212 |
| r11_pi05_spatial_50_error_hybrid_ir25 | 0.037 / 0.170 / 0.448 | 0.097 / 0.230 / 0.450 | 0.298 | 0.175 | 0.321 | 0.285 |
| r11_groot_l10_50_error_hybrid_ir25 | 0.086 / 0.213 / 0.549 | 0.108 / 0.230 / 0.481 | 0.381 | 0.165 | 0.136 | 0.424 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.087 / 0.209 / 0.503 | 0.108 / 0.230 / 0.481 | 0.198 | 0.538 | 0.648 | 0.405 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.034 / 0.169 / 0.434 | 0.097 / 0.230 / 0.450 | 0.168 | 0.504 | 0.714 | 0.282 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.039 / 0.170 / 0.419 | 0.097 / 0.230 / 0.450 | — | — | — | 0.289 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.025 / 0.137 / 0.378 | 0.049 / 0.174 / 0.393 | — | — | — | 0.341 |
| r11_groot_spatial_50_distance_ir32 | 9.320 / 12.662 / 25.973 | 12.845 / 17.330 / 30.678 | 17.282 | 0.165 | 0.506 | 0.046 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.039 / 0.166 / 0.565 | 0.054 / 0.196 / 0.608 | 0.193 | 0.420 | 0.511 | 0.395 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.071 / 0.161 / 0.399 | 0.080 / 0.171 / 0.398 | 0.190 | 0.376 | 0.411 | 0.456 |
| r11_pi05_l10_50_disagreement_ir32 | 0.031 / 0.136 / 0.519 | 0.034 / 0.152 / 0.542 | 0.164 | 0.405 | 0.454 | 0.440 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.085 / 0.207 / 0.499 | 0.108 / 0.230 / 0.481 | — | — | — | 0.397 |
| r11_groot_l10_50_distance_ir25 | 6.942 / 9.641 / 24.282 | 7.720 / 11.189 / 20.990 | 14.728 | 0.210 | 0.211 | 0.268 |
| r11_pi05_l10_50_error_hybrid_ir25 | 0.019 / 0.135 / 0.368 | 0.049 / 0.174 / 0.393 | 0.255 | 0.155 | 0.226 | 0.335 |
| r11_pi05_l10_200_distance_ir25 | 4.904 / 6.995 / 12.271 | 5.321 / 7.778 / 13.633 | 8.818 | 0.192 | 0.310 | 0.330 |
| r11_groot_l10_200_distance_ir25 | 5.894 / 7.595 / 15.575 | 6.229 / 8.118 / 15.122 | 9.594 | 0.209 | 0.224 | 0.356 |
| r11_groot_spatial_50_distance_ir40 | 9.339 / 12.703 / 25.223 | 12.845 / 17.330 / 30.678 | 15.500 | 0.231 | 0.715 | 0.048 |
| r11_pi05_l10_50_distance_ir40 | 6.125 / 9.521 / 20.450 | 7.175 / 12.405 / 31.065 | 10.099 | 0.424 | 0.723 | 0.218 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | 0.088 / 0.207 / 0.499 | 0.108 / 0.230 / 0.481 | — | — | — | 0.397 |
| r11_groot_l10_50_disagreement_ir32 | 0.042 / 0.173 / 0.742 | 0.050 / 0.190 / 0.638 | 0.246 | 0.340 | 0.360 | 0.444 |
| r11_groot_spatial_50_error_hybrid_ir32 | 0.072 / 0.167 / 0.358 | 0.106 / 0.213 / 0.474 | 0.214 | 0.305 | 0.488 | 0.276 |
| r11_pi05_l10_50_error_hybrid_ir32 | 0.024 / 0.136 / 0.373 | 0.049 / 0.174 / 0.393 | 0.185 | 0.312 | 0.455 | 0.339 |
| r11_groot_l10_200_distance_ir32 | 5.883 / 7.591 / 15.968 | 6.229 / 8.118 / 15.122 | 8.305 | 0.328 | 0.451 | 0.354 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | 0.072 / 0.171 / 0.343 | 0.106 / 0.213 / 0.474 | — | — | — | 0.297 |
| r11_groot_l10_200_error_hybrid_ir25 | 0.071 / 0.159 / 0.406 | 0.080 / 0.171 / 0.398 | 0.268 | 0.179 | 0.190 | 0.443 |
| r11_pi05_l10_500_error_hybrid_ir25 | 0.008 / 0.075 / 0.244 | 0.011 / 0.087 / 0.278 | 0.134 | 0.200 | 0.266 | 0.410 |
| r11_groot_l10_500_distance_ir25 | 5.534 / 7.025 / 11.978 | 5.676 / 7.347 / 12.270 | 8.496 | 0.174 | 0.226 | 0.398 |
| r11_pi05_spatial_50_distance_ir25 | 8.260 / 13.220 / 27.509 | 11.026 / 18.480 / 34.296 | 21.115 | 0.144 | 0.323 | 0.158 |

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
| pi05_l10_500 | 0 | 500 |
| pi05_spatial_50 | 100 | 50 |

**Donor-count result:** with independent B queries and the deployment metric fixed, expanding donors lowers distance threshold exceedance by only 0.12–2.74 pp across these completed distance arms. This is much smaller than the largest sparse-library crossfit-reference to live selection gaps. The simple donor-count explanation does not account for the major undershoots on the tested independent B states; its relative importance can be larger in arms with small errors. This does not identify the remaining gap as demo tracking: fold metric/scale fitting, query distribution and guard overlap are separate changes.

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
| r11_groot_spatial_50_error_hybrid_ir40 | 0.6486 | 0.6464 | -0.0022 | 0.6169 | 0.175 / 0.182 |
| r11_pi05_spatial_50_distance_ir32 | 0.2209 | 0.2184 | -0.0025 | 0.1915 | 13.327 / 14.381 |
| r11_groot_l10_50_error_hybrid_ir32 | 0.3527 | 0.3487 | -0.0040 | 0.3615 | 0.212 / 0.207 |
| r11_groot_l10_50_distance_ir32 | 0.2563 | 0.2473 | -0.0090 | 0.2767 | 9.405 / 9.618 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | 0.3527 | 0.3458 | -0.0069 | 0.3354 | 0.169 / 0.184 |
| r11_pi05_l10_500_distance_ir25 | — | — | — | 0.1977 | 6.317 / — |
| r11_pi05_l10_50_distance_ir25 | 0.1469 | 0.1440 | -0.0029 | 0.1109 | 9.424 / 10.501 |
| r11_pi05_spatial_50_error_hybrid_ir25 | 0.2629 | 0.2551 | -0.0077 | 0.2477 | 0.170 / 0.184 |
| r11_groot_l10_50_error_hybrid_ir25 | 0.1401 | 0.1372 | -0.0029 | 0.1520 | 0.213 / 0.207 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.5932 | 0.5876 | -0.0056 | 0.5935 | 0.209 / 0.207 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.6407 | 0.6424 | +0.0016 | 0.6140 | 0.169 / 0.184 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.5711 | 0.5716 | +0.0005 | 0.5473 | 0.170 / 0.184 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.2296 | 0.2246 | -0.0050 | 0.1985 | 0.137 / 0.158 |
| r11_groot_spatial_50_distance_ir32 | 0.1945 | 0.1910 | -0.0035 | 0.1651 | 12.662 / 13.886 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.4824 | 0.4459 | -0.0366 | 0.4196 | 0.166 / 0.174 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.4059 | 0.3978 | -0.0081 | 0.3951 | 0.161 / 0.162 |
| r11_pi05_l10_50_disagreement_ir32 | 0.4510 | 0.4245 | -0.0265 | 0.4052 | 0.136 / 0.142 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.1557 | 0.1529 | -0.0028 | 0.1505 | 0.207 / 0.207 |
| r11_groot_l10_50_distance_ir25 | 0.1276 | 0.1261 | -0.0015 | 0.2096 | 9.641 / 9.618 |
| r11_pi05_l10_50_error_hybrid_ir25 | 0.2228 | 0.2178 | -0.0050 | 0.1901 | 0.135 / 0.158 |
| r11_pi05_l10_200_distance_ir25 | 0.3486 | 0.3281 | -0.0205 | 0.1916 | 6.995 / 7.550 |
| r11_groot_l10_200_distance_ir25 | 0.2006 | 0.1896 | -0.0110 | 0.2091 | 7.595 / 7.701 |
| r11_groot_spatial_50_distance_ir40 | 0.3297 | 0.3231 | -0.0066 | 0.2309 | 12.703 / 13.886 |
| r11_pi05_l10_50_distance_ir40 | 0.5802 | 0.5591 | -0.0212 | 0.4239 | 9.521 / 10.501 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | 0.4967 | 0.4940 | -0.0027 | 0.4915 | 0.207 / 0.207 |
| r11_groot_l10_50_disagreement_ir32 | 0.3740 | 0.3453 | -0.0286 | 0.3398 | 0.173 / 0.176 |
| r11_groot_spatial_50_error_hybrid_ir32 | 0.4460 | 0.4344 | -0.0116 | 0.4071 | 0.167 / 0.182 |
| r11_pi05_l10_50_error_hybrid_ir32 | 0.4277 | 0.4204 | -0.0074 | 0.3805 | 0.136 / 0.158 |
| r11_groot_l10_200_distance_ir32 | 0.3758 | 0.3484 | -0.0274 | 0.3280 | 7.591 / 7.701 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | 0.3454 | 0.3323 | -0.0130 | 0.3062 | 0.171 / 0.182 |
| r11_groot_l10_200_error_hybrid_ir25 | 0.1920 | 0.1867 | -0.0053 | 0.1838 | 0.159 / 0.162 |
| r11_pi05_l10_500_error_hybrid_ir25 | — | — | — | 0.2338 | 0.075 / — |
| r11_groot_l10_500_distance_ir25 | — | — | — | 0.1743 | 7.025 / — |
| r11_pi05_spatial_50_distance_ir25 | 0.1323 | 0.1310 | -0.0013 | 0.1440 | 13.220 / 14.381 |

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
| r11_groot_spatial_50_error_hybrid_ir40 | 0.1367 | 0.1387 | 0.3713 | 0.3714 | +0.0001 |
| r11_pi05_spatial_50_distance_ir32 | 0.0793 | 0.0783 | 0.1984 | 0.1970 | -0.0014 |
| r11_groot_l10_50_error_hybrid_ir32 | 0.2868 | 0.2917 | 0.3035 | 0.3034 | -0.0001 |
| r11_groot_l10_50_distance_ir32 | 0.2868 | 0.2917 | 0.2580 | 0.2570 | -0.0010 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | 0.0793 | 0.0783 | 0.2497 | 0.2467 | -0.0029 |
| r11_pi05_l10_500_distance_ir25 | — | — | — | — | — |
| r11_pi05_l10_50_distance_ir25 | 0.2671 | 0.2634 | 0.2180 | 0.2161 | -0.0019 |
| r11_pi05_spatial_50_error_hybrid_ir25 | 0.0793 | 0.0783 | 0.2140 | 0.2107 | -0.0034 |
| r11_groot_l10_50_error_hybrid_ir25 | 0.2868 | 0.2917 | 0.2386 | 0.2398 | +0.0012 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.2868 | 0.2917 | 0.3768 | 0.3753 | -0.0015 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.0793 | 0.0783 | 0.3659 | 0.3668 | +0.0009 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.0793 | 0.0783 | 0.3378 | 0.3384 | +0.0006 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.2671 | 0.2634 | 0.2523 | 0.2494 | -0.0029 |
| r11_groot_spatial_50_distance_ir32 | 0.1367 | 0.1387 | 0.1999 | 0.2000 | +0.0001 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.0793 | 0.0783 | 0.3007 | 0.2860 | -0.0147 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.2820 | 0.2826 | 0.3181 | 0.3156 | -0.0025 |
| r11_pi05_l10_50_disagreement_ir32 | 0.2671 | 0.2634 | 0.3192 | 0.3093 | -0.0099 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.2868 | 0.2917 | 0.2430 | 0.2441 | +0.0011 |
| r11_groot_l10_50_distance_ir25 | 0.2868 | 0.2917 | 0.2229 | 0.2245 | +0.0016 |
| r11_pi05_l10_50_error_hybrid_ir25 | 0.2671 | 0.2634 | 0.2502 | 0.2472 | -0.0030 |
| r11_pi05_l10_200_distance_ir25 | 0.2383 | 0.2356 | 0.2661 | 0.2591 | -0.0070 |
| r11_groot_l10_200_distance_ir25 | 0.2820 | 0.2826 | 0.2418 | 0.2389 | -0.0029 |
| r11_groot_spatial_50_distance_ir40 | 0.1367 | 0.1387 | 0.2453 | 0.2443 | -0.0010 |
| r11_pi05_l10_50_distance_ir40 | 0.2671 | 0.2634 | 0.3529 | 0.3452 | -0.0077 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | 0.2868 | 0.2917 | 0.3473 | 0.3470 | -0.0004 |
| r11_groot_l10_50_disagreement_ir32 | 0.2868 | 0.2917 | 0.3116 | 0.3021 | -0.0096 |
| r11_groot_spatial_50_error_hybrid_ir32 | 0.1367 | 0.1387 | 0.2945 | 0.2914 | -0.0032 |
| r11_pi05_l10_50_error_hybrid_ir32 | 0.2671 | 0.2634 | 0.3135 | 0.3101 | -0.0034 |
| r11_groot_l10_200_distance_ir32 | 0.2820 | 0.2826 | 0.2966 | 0.2878 | -0.0088 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | 0.1367 | 0.1387 | 0.2569 | 0.2532 | -0.0037 |
| r11_groot_l10_200_error_hybrid_ir25 | 0.2820 | 0.2826 | 0.2521 | 0.2508 | -0.0013 |
| r11_pi05_l10_500_error_hybrid_ir25 | — | — | — | — | — |
| r11_groot_l10_500_distance_ir25 | — | — | — | — | — |
| r11_pi05_spatial_50_distance_ir25 | 0.0793 | 0.0783 | 0.1660 | 0.1653 | -0.0006 |

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
| r11_groot_spatial_50_error_hybrid_ir40 | 0.4000 | 0.4137 |
| r11_groot_l10_50_error_hybrid_ir32 | 0.3200 | 0.3161 |
| r11_pi05_spatial_50_error_hybrid_ir25 | 0.2500 | 0.2555 |
| r11_groot_l10_50_error_hybrid_ir25 | 0.2500 | 0.2482 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.4000 | 0.3970 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.4000 | 0.4033 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.3200 | 0.3191 |
| r11_pi05_l10_50_error_hybrid_ir25 | 0.2500 | 0.2541 |
| r11_groot_spatial_50_error_hybrid_ir32 | 0.3200 | 0.3356 |
| r11_pi05_l10_50_error_hybrid_ir32 | 0.3200 | 0.3262 |
| r11_groot_l10_200_error_hybrid_ir25 | 0.2500 | 0.2486 |
| r11_pi05_l10_500_error_hybrid_ir25 | 0.2500 | 0.2499 |

## Why adaptation can still undershoot

The script verifies every committed-ledger update. With gain η=.2, the cost error processed before the final proposal is `(c_m/η)(q0−q_last+Σ clipping_adjustment)`. Add the cost error of the final unprocessed segment for the exact whole-episode error. An upward change from q0 to q_last is therefore a finite-episode underspend term; resetting at every episode prevents carrying that deficit forward. This is an accounting explanation, not a proposed changed controller evaluated on test.

| Arm | q0 | mean final q | IR initial→final | IR clipping | IR final segment | max update error |
|---|---|---|---|---|---|---|
| r11_pi05_l10_50_adaptive_error_hybrid_ir40 | 0.5701 | 0.6660 | -0.0071 | -0.0020 | 0.0029 | 0.0000000000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | 0.6155 | 0.7660 | -0.0281 | -0.0044 | 0.0106 | 0.0000000000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | 0.4241 | 0.5933 | -0.0335 | 0.0006 | 0.0094 | 0.0000000000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.6528 | 0.7804 | -0.0253 | -0.0058 | 0.0127 | 0.0000000000 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.2332 | 0.4853 | -0.0187 | 0.0053 | 0.0017 | 0.0000000000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.1558 | 0.3844 | -0.0155 | 0.0103 | 0.0022 | 0.0000000000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | 0.5370 | 0.6106 | -0.0053 | -0.0019 | 0.0022 | 0.0000000000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | 0.4047 | 0.5906 | -0.0344 | 0.0007 | 0.0105 | 0.0000000000 |

The remaining adaptive delivery problem is concentrated in Spatial: all four L10 adaptive arms are within ±.02, but only one of four Spatial arms is. The shorter Spatial episodes show larger startup-to-final-dose underspend terms; the exact terminal/clipping decomposition above is more informative than attributing this to a broken update rule.

## Success versus realized IR

![Realized-IR frontier](analysis/frontier.png)

Every cell-size plot includes measured methods, same-batch knob-off, and the historical R8 pure-policy point. Hollow diamonds show the historical R10 knob-off configuration, with common y-axis limits within each suite. Historical points do not replace the same-batch controls in comparisons. Curves are piecewise linear through measured IR points, ordered by realized IR rather than target labels. Some dense cells have only one or two planned targets.

Main matched comparisons use the midpoint of each pair of observed common-support intervals. Both curves need at least two measured points, except that each single-point disagreement arm is compared at its own measured IR against the interpolated comparator curve. For that exception the arm IR is recalculated in each bootstrap draw. No extrapolation is performed. Intervals use 2000 paired episode bootstrap draws: the same evaluation episode draw is reused across all targets, nested sizes and models in a suite. Realized IR and interpolation weights are recalculated in each draw; unsupported draws are excluded and counted. These are percentile intervals conditional on supported draws, not unconditional coverage guarantees; distance has particularly narrow overlaps and substantial excluded draws. These exploratory intervals describe finite-episode variation conditional on the observed rollout per arm; they do not capture repeated-policy-run variance or adjust for multiple comparisons.

| Cell | method − comparator | IR | ΔSR pp | 95% interval pp | valid bootstrap draws |
|---|---|---|---|---|---|
| groot_l10_200 | distance − random | 0.2605 | -0.28 | [-3.50, 3.14] | 1989 |
| groot_l10_200 | distance − periodic | 0.2602 | -0.29 | [-3.36, 3.47] | 1991 |
| groot_l10_200 | error_hybrid − random | 0.2776 | +1.77 | [-0.99, 4.71] | 2000 |
| groot_l10_200 | error_hybrid − periodic | 0.2772 | +0.25 | [-2.35, 3.17] | 2000 |
| groot_l10_200 | periodic − random | 0.2802 | +1.77 | [-1.06, 4.61] | 2000 |
| groot_l10_50 | distance − random | 0.2849 | +2.66 | [-1.30, 6.15] | 2000 |
| groot_l10_50 | distance − periodic | 0.2829 | +0.07 | [-3.99, 3.77] | 2000 |
| groot_l10_50 | error_hybrid − random | 0.3179 | -1.04 | [-5.49, 2.95] | 2000 |
| groot_l10_50 | error_hybrid − periodic | 0.3159 | -4.35 | [-8.97, -0.02] | 2000 |
| groot_l10_50 | adaptive_error_hybrid − random | 0.3560 | +1.20 | [-1.85, 4.27] | 2000 |
| groot_l10_50 | adaptive_error_hybrid − periodic | 0.3560 | -1.89 | [-4.84, 1.05] | 2000 |
| groot_l10_50 | disagreement − random | 0.3056 | +2.40 | [-1.89, 6.42] | 2000 |
| groot_l10_50 | disagreement − periodic | 0.3056 | -0.70 | [-4.99, 3.42] | 2000 |
| groot_l10_50 | periodic − random | 0.3291 | +3.33 | [-0.60, 7.07] | 2000 |
| groot_spatial_50 | error_hybrid − random | 0.2989 | +0.00 | [-2.90, 2.80] | 2000 |
| groot_spatial_50 | error_hybrid − periodic | 0.3040 | +1.40 | [-1.11, 3.98] | 2000 |
| groot_spatial_50 | adaptive_error_hybrid − random | 0.3374 | +1.30 | [-1.08, 3.75] | 2000 |
| groot_spatial_50 | adaptive_error_hybrid − periodic | 0.3374 | +1.65 | [-0.74, 4.11] | 2000 |
| groot_spatial_50 | disagreement − random | 0.2777 | +0.53 | [-2.78, 3.92] | 2000 |
| groot_spatial_50 | disagreement − periodic | 0.2777 | +2.40 | [-0.67, 5.58] | 2000 |
| groot_spatial_50 | periodic − random | 0.3217 | -0.72 | [-3.69, 2.16] | 2000 |
| pi05_l10_50 | distance − random | 0.2600 | +1.87 | [-0.82, 4.70] | 2000 |
| pi05_l10_50 | distance − periodic | 0.2615 | -0.55 | [-3.16, 2.31] | 2000 |
| pi05_l10_50 | error_hybrid − random | 0.3000 | +3.72 | [0.39, 6.84] | 2000 |
| pi05_l10_50 | error_hybrid − periodic | 0.3015 | -0.84 | [-3.99, 2.12] | 2000 |
| pi05_l10_50 | adaptive_error_hybrid − random | 0.3500 | +1.48 | [-1.22, 4.19] | 2000 |
| pi05_l10_50 | adaptive_error_hybrid − periodic | 0.3508 | -0.89 | [-3.37, 1.63] | 2000 |
| pi05_l10_50 | disagreement − random | 0.2938 | +2.00 | [-1.29, 5.15] | 2000 |
| pi05_l10_50 | disagreement − periodic | 0.2938 | -2.16 | [-5.41, 0.89] | 2000 |
| pi05_l10_50 | periodic − random | 0.3128 | +4.50 | [1.31, 7.79] | 2000 |
| pi05_spatial_50 | distance − random | 0.2373 | +2.04 | [-0.07, 4.16] | 1781 |
| pi05_spatial_50 | distance − periodic | 0.2414 | +0.15 | [-1.96, 1.70] | 1410 |
| pi05_spatial_50 | error_hybrid − random | 0.2928 | +1.51 | [-0.41, 3.35] | 2000 |
| pi05_spatial_50 | error_hybrid − periodic | 0.2970 | -0.45 | [-2.13, 1.12] | 2000 |
| pi05_spatial_50 | adaptive_error_hybrid − random | 0.3391 | +0.67 | [-0.88, 2.21] | 2000 |
| pi05_spatial_50 | adaptive_error_hybrid − periodic | 0.3391 | -1.13 | [-2.56, 0.26] | 2000 |
| pi05_spatial_50 | disagreement − random | 0.2763 | +2.39 | [0.68, 4.15] | 2000 |
| pi05_spatial_50 | disagreement − periodic | 0.2763 | +0.39 | [-1.26, 1.92] | 2000 |
| pi05_spatial_50 | periodic − random | 0.3124 | +1.94 | [0.19, 3.78] | 2000 |

| Method − comparator | cells | mean ΔSR pp | paired 95% interval pp | valid draws |
|---|---|---|---|---|
| distance − random | 4 | +1.57 | [-0.05, 3.08] | 1771 |
| distance − periodic | 4 | -0.16 | [-1.77, 1.20] | 1403 |
| error_hybrid − random | 5 | +1.19 | [-0.23, 2.49] | 2000 |
| error_hybrid − periodic | 5 | -0.80 | [-2.27, 0.56] | 2000 |
| adaptive_error_hybrid − random | 4 | +1.16 | [-0.10, 2.41] | 2000 |
| adaptive_error_hybrid − periodic | 4 | -0.57 | [-1.79, 0.60] | 2000 |
| disagreement − random | 4 | +1.83 | [0.10, 3.48] | 2000 |
| disagreement − periodic | 4 | -0.02 | [-1.70, 1.62] | 2000 |
| periodic − random | 5 | +2.16 | [0.80, 3.57] | 2000 |

**Final interpretation:** periodic−random is +2.16 [0.80, 3.57] pp across 5 cells, 2000 supported bootstrap draws. Its interval excludes zero and the original +0.6 pp forecast has the correct positive direction. The matched-IR point estimate is larger, but the different weighting/support makes this a different estimand from the original pooled forecast, not a formal test of its numeric accuracy. Error_hybrid−random is +1.19 [-0.23, 2.49] pp across 5 cells, 2000 supported bootstrap draws, while error_hybrid−periodic is -0.80 [-2.27, 0.56] pp across 5 cells, 2000 supported bootstrap draws. Both hybrid intervals include zero. Distance is not shown to improve SR over periodic despite favorable portions of its low-IR curve; it fails budget calibration and has no knob-only overlap for groot Spatial-50. Adaptive error hybrid likewise does not establish superiority over periodic. Disagreement−random is +1.83 [0.10, 3.48] pp across 4 cells, 2000 supported bootstrap draws and disagreement−periodic is -0.02 [-1.70, 1.62] pp across 4 cells, 2000 supported bootstrap draws: a promising unadjusted single-setting comparison against random, not evidence that it beats periodic. Each pooled contrast weights its supported cells equally, and different methods can have different supports.

Unsupported main comparisons: 33/72. See [matched_frontier.csv](analysis/matched_frontier.csv). In the final grid, unsupported comparisons reflect single-point designs, methods not planned for that cell, or nonoverlapping realized IR—not unfinished arms.

Per-arm interpolation against completed random/periodic points, with explicit labeling when a same-batch off anchor is needed:

| Arm | comparator | type | ΔSR pp | 95% interval pp | valid draws |
|---|---|---|---|---|---|
| r11_pi05_l10_50_error_hybrid_ir40 | random | arm vs comparator interpolation | +1.28 | [-1.77, 4.44] | 2000 |
| r11_pi05_l10_50_error_hybrid_ir40 | periodic | arm vs comparator interpolation | -0.02 | [-2.96, 3.00] | 2000 |
| r11_groot_l10_50_distance_ir40 | random | arm vs comparator interpolation | +1.12 | [-3.64, 5.54] | 2000 |
| r11_groot_l10_50_distance_ir40 | periodic | arm vs comparator interpolation | -2.06 | [-6.49, 2.38] | 2000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | random | arm vs comparator interpolation | +2.60 | [-0.36, 5.43] | 2000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir40 | periodic | arm vs comparator interpolation | +1.95 | [-0.88, 4.82] | 2000 |
| r11_pi05_l10_50_distance_ir32 | random | secondary off-anchored | +1.48 | [-1.88, 4.68] | 2000 |
| r11_pi05_l10_50_distance_ir32 | periodic | secondary off-anchored | +0.86 | [-2.36, 4.11] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir32 | random | arm vs comparator interpolation | +1.54 | [-0.50, 3.44] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -0.45 | [-2.40, 1.29] | 2000 |
| r11_groot_spatial_50_disagreement_ir32 | random | arm vs comparator interpolation | +0.53 | [-2.78, 3.92] | 2000 |
| r11_groot_spatial_50_disagreement_ir32 | periodic | arm vs comparator interpolation | +2.40 | [-0.67, 5.58] | 2000 |
| r11_pi05_l10_200_error_hybrid_ir25 | random | secondary off-anchored | +1.77 | [-1.34, 4.57] | 1996 |
| r11_pi05_l10_200_error_hybrid_ir25 | periodic | secondary off-anchored | +0.75 | [-1.91, 3.35] | 2000 |
| r11_groot_l10_500_error_hybrid_ir25 | random | secondary off-anchored | +3.10 | [-0.03, 6.12] | 1956 |
| r11_groot_l10_500_error_hybrid_ir25 | periodic | secondary off-anchored | +0.21 | [-2.91, 3.21] | 1998 |
| r11_groot_spatial_50_error_hybrid_ir25 | random | secondary off-anchored | +0.86 | [-2.21, 3.80] | 2000 |
| r11_groot_spatial_50_error_hybrid_ir25 | periodic | secondary off-anchored | +2.05 | [-0.76, 4.88] | 2000 |
| r11_groot_spatial_50_distance_ir25 | random | secondary off-anchored | -3.07 | [-6.12, 0.04] | 2000 |
| r11_groot_spatial_50_distance_ir25 | periodic | secondary off-anchored | -2.49 | [-5.60, 0.50] | 2000 |
| r11_pi05_spatial_50_distance_ir40 | random | arm vs comparator interpolation | +2.40 | [0.43, 4.59] | 2000 |
| r11_pi05_spatial_50_distance_ir40 | periodic | arm vs comparator interpolation | +0.33 | [-1.48, 2.14] | 2000 |
| r11_groot_spatial_50_error_hybrid_ir40 | random | arm vs comparator interpolation | +1.44 | [-1.46, 4.30] | 2000 |
| r11_groot_spatial_50_error_hybrid_ir40 | periodic | arm vs comparator interpolation | +1.30 | [-1.54, 3.95] | 2000 |
| r11_pi05_spatial_50_distance_ir32 | random | secondary off-anchored | +0.46 | [-2.08, 2.86] | 2000 |
| r11_pi05_spatial_50_distance_ir32 | periodic | secondary off-anchored | -0.67 | [-3.35, 1.84] | 2000 |
| r11_groot_l10_50_error_hybrid_ir32 | random | arm vs comparator interpolation | -1.16 | [-5.55, 2.93] | 2000 |
| r11_groot_l10_50_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -4.36 | [-9.00, -0.03] | 2000 |
| r11_groot_l10_50_distance_ir32 | random | arm vs comparator interpolation | +3.91 | [-0.32, 8.47] | 2000 |
| r11_groot_l10_50_distance_ir32 | periodic | arm vs comparator interpolation | +1.51 | [-2.90, 6.27] | 2000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | random | arm vs comparator interpolation | +0.72 | [-1.60, 2.82] | 2000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -1.23 | [-3.09, 0.64] | 2000 |
| r11_pi05_l10_500_distance_ir25 | random | secondary off-anchored | -1.39 | [-4.18, 1.24] | 2000 |
| r11_pi05_l10_500_distance_ir25 | periodic | secondary off-anchored | -0.87 | [-3.71, 1.86] | 2000 |
| r11_pi05_l10_50_distance_ir25 | random | secondary off-anchored | -0.76 | [-3.88, 2.43] | 2000 |
| r11_pi05_l10_50_distance_ir25 | periodic | secondary off-anchored | -1.07 | [-4.29, 2.18] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir25 | random | secondary off-anchored | +1.42 | [-0.92, 3.78] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir25 | periodic | secondary off-anchored | -0.12 | [-2.32, 2.00] | 2000 |
| r11_groot_l10_50_error_hybrid_ir25 | random | secondary off-anchored | +2.54 | [-1.60, 6.23] | 2000 |
| r11_groot_l10_50_error_hybrid_ir25 | periodic | secondary off-anchored | +0.84 | [-3.89, 4.85] | 2000 |
| r11_groot_l10_50_error_hybrid_ir40 | random | arm vs comparator interpolation | +1.65 | [-2.09, 5.33] | 2000 |
| r11_groot_l10_50_error_hybrid_ir40 | periodic | arm vs comparator interpolation | -1.27 | [-4.80, 2.24] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir40 | random | arm vs comparator interpolation | +1.50 | [-0.19, 3.10] | 2000 |
| r11_pi05_spatial_50_error_hybrid_ir40 | periodic | arm vs comparator interpolation | -0.22 | [-1.71, 1.22] | 2000 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | random | arm vs comparator interpolation | +0.65 | [-1.35, 2.76] | 1891 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | periodic | arm vs comparator interpolation | -0.94 | [-2.67, 0.72] | 2000 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | random | arm vs comparator interpolation | +3.18 | [-0.53, 6.78] | 2000 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -1.58 | [-4.84, 1.61] | 2000 |
| r11_groot_spatial_50_distance_ir32 | random | secondary off-anchored | -2.50 | [-5.69, 0.67] | 2000 |
| r11_groot_spatial_50_distance_ir32 | periodic | secondary off-anchored | -1.67 | [-4.80, 1.45] | 2000 |
| r11_pi05_spatial_50_disagreement_ir32 | random | arm vs comparator interpolation | +2.39 | [0.68, 4.15] | 2000 |
| r11_pi05_spatial_50_disagreement_ir32 | periodic | arm vs comparator interpolation | +0.39 | [-1.26, 1.92] | 2000 |
| r11_groot_l10_200_error_hybrid_ir32 | random | arm vs comparator interpolation | +2.01 | [-1.81, 5.65] | 1999 |
| r11_groot_l10_200_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -2.23 | [-5.57, 1.70] | 1895 |
| r11_pi05_l10_50_disagreement_ir32 | random | arm vs comparator interpolation | +2.00 | [-1.29, 5.15] | 2000 |
| r11_pi05_l10_50_disagreement_ir32 | periodic | arm vs comparator interpolation | -2.16 | [-5.41, 0.89] | 2000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | random | arm vs comparator interpolation | -1.09 | [-5.37, 2.95] | 2000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | periodic | arm vs comparator interpolation | -4.39 | [-9.09, 0.02] | 2000 |
| r11_groot_l10_50_distance_ir25 | random | secondary off-anchored | -2.53 | [-7.09, 1.79] | 2000 |
| r11_groot_l10_50_distance_ir25 | periodic | secondary off-anchored | -4.51 | [-9.24, -0.08] | 2000 |
| r11_pi05_l10_50_error_hybrid_ir25 | random | secondary off-anchored | +3.12 | [-0.38, 6.23] | 2000 |
| r11_pi05_l10_50_error_hybrid_ir25 | periodic | secondary off-anchored | +2.42 | [-1.17, 5.48] | 2000 |
| r11_pi05_l10_200_distance_ir25 | random | secondary off-anchored | +1.21 | [-1.39, 3.83] | 2000 |
| r11_pi05_l10_200_distance_ir25 | periodic | secondary off-anchored | +0.56 | [-2.15, 3.38] | 2000 |
| r11_groot_l10_200_distance_ir25 | random | secondary off-anchored | -0.56 | [-4.57, 3.21] | 2000 |
| r11_groot_l10_200_distance_ir25 | periodic | secondary off-anchored | +0.44 | [-3.40, 4.09] | 2000 |
| r11_groot_spatial_50_distance_ir40 | random | secondary off-anchored | -3.14 | [-6.51, 0.06] | 2000 |
| r11_groot_spatial_50_distance_ir40 | periodic | secondary off-anchored | -1.95 | [-5.27, 1.09] | 2000 |
| r11_pi05_l10_50_distance_ir40 | random | arm vs comparator interpolation | +2.40 | [-1.02, 5.67] | 2000 |
| r11_pi05_l10_50_distance_ir40 | periodic | arm vs comparator interpolation | -1.51 | [-4.90, 1.84] | 2000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | random | arm vs comparator interpolation | +3.64 | [-0.16, 7.52] | 2000 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | periodic | arm vs comparator interpolation | +0.89 | [-2.82, 4.68] | 1973 |
| r11_groot_l10_50_disagreement_ir32 | random | arm vs comparator interpolation | +2.40 | [-1.89, 6.42] | 2000 |
| r11_groot_l10_50_disagreement_ir32 | periodic | arm vs comparator interpolation | -0.70 | [-4.99, 3.42] | 2000 |
| r11_groot_spatial_50_error_hybrid_ir32 | random | arm vs comparator interpolation | +0.73 | [-2.52, 3.94] | 2000 |
| r11_groot_spatial_50_error_hybrid_ir32 | periodic | arm vs comparator interpolation | +2.66 | [-0.74, 6.11] | 2000 |
| r11_pi05_l10_50_error_hybrid_ir32 | random | arm vs comparator interpolation | +3.60 | [0.24, 6.83] | 2000 |
| r11_pi05_l10_50_error_hybrid_ir32 | periodic | arm vs comparator interpolation | +0.04 | [-3.35, 3.43] | 2000 |
| r11_groot_l10_200_distance_ir32 | random | arm vs comparator interpolation | +0.24 | [-3.29, 3.98] | 2000 |
| r11_groot_l10_200_distance_ir32 | periodic | arm vs comparator interpolation | -0.97 | [-4.79, 3.09] | 2000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | random | arm vs comparator interpolation | +0.97 | [-2.33, 4.07] | 2000 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | periodic | arm vs comparator interpolation | +2.61 | [-0.29, 5.54] | 2000 |
| r11_groot_l10_200_error_hybrid_ir25 | random | secondary off-anchored | +2.00 | [-1.67, 5.66] | 2000 |
| r11_groot_l10_200_error_hybrid_ir25 | periodic | secondary off-anchored | +3.01 | [-0.79, 6.49] | 2000 |
| r11_pi05_l10_500_error_hybrid_ir25 | random | secondary off-anchored | +1.27 | [-1.55, 4.09] | 1999 |
| r11_pi05_l10_500_error_hybrid_ir25 | periodic | secondary off-anchored | +1.98 | [-0.84, 4.83] | 1998 |
| r11_groot_l10_500_distance_ir25 | random | secondary off-anchored | +2.77 | [-0.54, 5.86] | 1981 |
| r11_groot_l10_500_distance_ir25 | periodic | secondary off-anchored | +0.14 | [-3.46, 3.67] | 1986 |
| r11_pi05_spatial_50_distance_ir25 | random | secondary off-anchored | -0.15 | [-2.61, 2.44] | 2000 |
| r11_pi05_spatial_50_distance_ir25 | periodic | secondary off-anchored | -0.90 | [-3.63, 1.81] | 2000 |

Pooled efficiency over **same-batch off** controls is `Σ(SR−SR_off)/Σ(IR−IR_off)`. The shared bootstrap preserves reuse of the off episodes across targets. Random, periodic, distance and error_hybrid cover the same 17 planned cell-target points; adaptive and disagreement cover smaller designs. Realized spending still differs:

| Method | arms / cells | ΔSR sum | ΔIR sum | SR gain per IR | paired 95% interval |
|---|---|---|---|---|---|
| random | 17/8 | 0.4320 | 2.2981 | 0.188 | [0.095, 0.277] |
| periodic | 17/8 | 0.6840 | 2.3319 | 0.293 | [0.201, 0.383] |
| distance | 17/8 | 0.2300 | 1.0673 | 0.215 | [0.014, 0.413] |
| error_hybrid | 17/8 | 0.6660 | 1.9614 | 0.340 | [0.231, 0.448] |
| adaptive_error_hybrid | 8/4 | 0.4040 | 1.4931 | 0.271 | [0.180, 0.363] |
| disagreement | 4/4 | 0.1740 | 0.5163 | 0.337 | [0.186, 0.486] |
| periodic_pgt1 | 4/4 | 0.1720 | 0.5945 | 0.289 | [0.171, 0.413] |
| random_tail2 | 2/2 | 0.0820 | 0.2474 | 0.331 | [0.092, 0.587] |

Error_hybrid has the highest observed gain/spend ratio on the common 17-arm design (0.340), versus periodic 0.293, distance 0.215 and random 0.188. Paired differences of these ratios are reported below. They compare average returns at the actual budgets each method spent, and do not override the matched-realized-IR frontier:

| Method − control | common arms | efficiency difference | paired interval |
|---|---|---|---|
| periodic − random | 17 | +0.105 | [0.039, 0.177] |
| error_hybrid − random | 17 | +0.152 | [0.077, 0.229] |
| distance − random | 17 | +0.028 | [-0.129, 0.188] |
| error_hybrid − periodic | 17 | +0.046 | [-0.024, 0.119] |
| distance − periodic | 17 | -0.078 | [-0.226, 0.074] |

The error_hybrid−periodic efficiency difference is +0.046 SR per IR with a paired interval [−0.024, +0.119], so even the higher observed ratio does not establish a clear efficiency advantage over periodic. Both outperform random on this average-return metric. Matched-target error_hybrid−random below is positive while its matched-realized-IR interval includes zero because the estimand, supported cells and sampled budgets differ.

For reconciliation only, the next table reproduces the coordinator’s **matched-target** effects from the raw accepted outcomes. It is a different estimand from the realized-IR frontier. Our intervals resample the same episode identities jointly across all targets and cells, avoiding the assumption that repeated uses of one initial condition are independent:

| Method − control | arm-episode comparisons | ΔSR pp | paired interval pp | mean ΔIR |
|---|---|---|---|---|
| periodic − random | 8500 | +1.48 | [0.61, 2.42] | +0.0020 |
| error_hybrid − random | 8500 | +1.38 | [0.51, 2.29] | -0.0198 |
| distance − random | 8500 | -1.19 | [-2.19, -0.21] | -0.0724 |
| error_hybrid − periodic | 8500 | -0.11 | [-0.94, 0.72] | -0.0218 |
| distance − periodic | 8500 | -2.67 | [-3.61, -1.72] | -0.0744 |

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
| r11_groot_spatial_50_error_hybrid_ir40 | 0.264 | 1.0 | 4.0 |
| r11_pi05_spatial_50_distance_ir32 | 0.255 | 5.0 | 11.0 |
| r11_groot_l10_50_error_hybrid_ir32 | 0.627 | 1.0 | 5.0 |
| r11_groot_l10_50_distance_ir32 | 0.755 | 3.0 | 10.0 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir32 | 0.208 | 1.0 | 4.0 |
| r11_pi05_l10_500_distance_ir25 | 0.514 | 3.0 | 13.0 |
| r11_pi05_l10_50_distance_ir25 | 0.478 | 4.0 | 17.0 |
| r11_pi05_spatial_50_error_hybrid_ir25 | 0.249 | 2.0 | 8.0 |
| r11_groot_l10_50_error_hybrid_ir25 | 0.644 | 2.0 | 8.0 |
| r11_groot_l10_50_error_hybrid_ir40 | 0.626 | 1.0 | 4.0 |
| r11_pi05_spatial_50_error_hybrid_ir40 | 0.204 | 1.0 | 4.0 |
| r11_pi05_spatial_50_adaptive_error_hybrid_ir40 | 0.184 | 1.0 | 2.0 |
| r11_pi05_l10_50_adaptive_error_hybrid_ir32 | 0.431 | 1.0 | 4.0 |
| r11_groot_spatial_50_distance_ir32 | 0.316 | 5.0 | 12.0 |
| r11_pi05_spatial_50_disagreement_ir32 | 0.227 | 2.0 | 7.0 |
| r11_groot_l10_200_error_hybrid_ir32 | 0.597 | 1.0 | 5.0 |
| r11_pi05_l10_50_disagreement_ir32 | 0.453 | 2.0 | 7.0 |
| r11_groot_l10_50_adaptive_error_hybrid_ir32 | 0.571 | 1.0 | 4.0 |
| r11_groot_l10_50_distance_ir25 | 0.789 | 2.0 | 10.0 |
| r11_pi05_l10_50_error_hybrid_ir25 | 0.444 | 2.0 | 9.0 |
| r11_pi05_l10_200_distance_ir25 | 0.516 | 4.0 | 16.0 |
| r11_groot_l10_200_distance_ir25 | 0.752 | 3.0 | 10.0 |
| r11_groot_spatial_50_distance_ir40 | 0.306 | 4.0 | 11.0 |
| r11_pi05_l10_50_distance_ir40 | 0.484 | 3.0 | 14.0 |
| r11_groot_l10_50_adaptive_error_hybrid_ir40 | 0.601 | 1.0 | 2.0 |
| r11_groot_l10_50_disagreement_ir32 | 0.615 | 2.0 | 6.0 |
| r11_groot_spatial_50_error_hybrid_ir32 | 0.268 | 2.0 | 6.0 |
| r11_pi05_l10_50_error_hybrid_ir32 | 0.452 | 2.0 | 6.0 |
| r11_groot_l10_200_distance_ir32 | 0.698 | 2.0 | 9.0 |
| r11_groot_spatial_50_adaptive_error_hybrid_ir32 | 0.225 | 1.0 | 4.0 |
| r11_groot_l10_200_error_hybrid_ir25 | 0.597 | 2.0 | 7.0 |
| r11_pi05_l10_500_error_hybrid_ir25 | 0.444 | 2.0 | 8.0 |
| r11_groot_l10_500_distance_ir25 | 0.701 | 3.0 | 10.0 |
| r11_pi05_spatial_50_distance_ir25 | 0.324 | 5.0 | 11.0 |

Cache-only stretches count consecutive vision anchors with no policy call, including censored start/end runs. At nominal cadence one anchor spans ten controls (two five-control slots); a policy tail is not counted as cache-only.

Pooled placement below weights actual looks, rather than treating a short and long episode as equal exposure. The outcome association is descriptive: eventually failed episodes can be longer and more stalled. It cannot identify whether a particular extra call caused recovery.

| Method | group | looks | guard / look | extra call / eligible look |
|---|---|---|---|---|
| distance | eventual_success | 150681 | 0.149 | 0.152 |
| distance | eventual_failure | 50368 | 0.460 | 0.348 |
| distance | progress_0 | 43078 | 0.097 | 0.246 |
| distance | progress_1 | 38299 | 0.201 | 0.127 |
| distance | progress_2 | 38915 | 0.270 | 0.155 |
| distance | progress_3 | 39558 | 0.299 | 0.179 |
| distance | progress_4 | 41199 | 0.276 | 0.207 |
| error_hybrid | eventual_success | 153430 | 0.147 | 0.325 |
| error_hybrid | eventual_failure | 42152 | 0.459 | 0.339 |
| error_hybrid | progress_0 | 41948 | 0.086 | 0.278 |
| error_hybrid | progress_1 | 37214 | 0.196 | 0.323 |
| error_hybrid | progress_2 | 38003 | 0.256 | 0.322 |
| error_hybrid | progress_3 | 38185 | 0.286 | 0.354 |
| error_hybrid | progress_4 | 40232 | 0.257 | 0.374 |
| adaptive_error_hybrid | eventual_success | 64859 | 0.140 | 0.553 |
| adaptive_error_hybrid | eventual_failure | 17324 | 0.482 | 0.449 |
| adaptive_error_hybrid | progress_0 | 17789 | 0.080 | 0.390 |
| adaptive_error_hybrid | progress_1 | 15548 | 0.185 | 0.568 |
| adaptive_error_hybrid | progress_2 | 15916 | 0.258 | 0.591 |
| adaptive_error_hybrid | progress_3 | 15990 | 0.285 | 0.582 |
| adaptive_error_hybrid | progress_4 | 16940 | 0.262 | 0.612 |
| disagreement | eventual_success | 32275 | 0.140 | 0.373 |
| disagreement | eventual_failure | 9330 | 0.488 | 0.350 |
| disagreement | progress_0 | 8995 | 0.079 | 0.203 |
| disagreement | progress_1 | 7877 | 0.191 | 0.364 |
| disagreement | progress_2 | 8060 | 0.254 | 0.420 |
| disagreement | progress_3 | 8088 | 0.299 | 0.424 |
| disagreement | progress_4 | 8585 | 0.280 | 0.500 |

Placement interpretation: distance concentrates additional calls on eventually failed episodes (34.8% of eligible looks versus 15.2% on successes), while error_hybrid is much flatter (33.9% versus 32.5%). Adaptive makes fewer additional calls on failed episodes (44.9% versus 55.3%) while their guards consume much more of the budget (48.2% versus 14.0% of looks). This is consistent with guard-aware budget feedback, not evidence that reducing extra calls causes failure. Adaptive and disagreement increase their conditional call rates from early to late episode progress; the CSV retains every cell/arm so these pooled associations are not mistaken for task-dependent settings.

## Hardware identity checks

| Model | status | local SR | remote SR | ΔSR pp [95%] | local IR / remote IR |
|---|---|---|---|---|---|
| pi05 | available | 0.844 | 0.826 | +1.80 [-1.00, 4.60] | 0.1669 / 0.1680 |
| groot | available | 0.750 | 0.748 | +0.20 [-3.80, 4.20] | 0.2085 / 0.2087 |

The pi05 comparison is local R11 knob-off versus historical H100 R10 distance-corrected control; it mixes hardware and batch. The groot duplicate compares the explicitly requested local identity run with the H100 R11 off run. Both point differences are small and both paired intervals include zero: these two checks show no clear systematic hardware shift. They are not equivalence tests; the groot SR interval still spans several percentage points. Hardware cannot be assumed irrelevant to every other cell, and root allocation is not randomized replication.

Current knob-off versus historical knob-off, paired on the same initial-condition identities:

| Cell | current off SR | R10 off SR | ΔSR pp [paired interval] | ΔIR |
|---|---|---|---|---|
| groot_l10_200 | 0.820 | 0.820 | +0.00 [-3.20, 3.20] | -0.0018 |
| groot_l10_50 | 0.748 | 0.754 | -0.60 [-4.20, 3.00] | +0.0016 |
| pi05_l10_200 | 0.890 | 0.904 | -1.40 [-4.20, 1.20] | -0.0008 |
| groot_spatial_50 | 0.892 | 0.896 | -0.40 [-2.40, 1.40] | +0.0014 |
| pi05_l10_50 | 0.844 | 0.826 | +1.80 [-1.00, 4.80] | -0.0011 |
| groot_l10_500 | 0.858 | 0.906 | -4.80 [-8.00, -1.80] | +0.0063 |
| pi05_l10_500 | 0.906 | 0.894 | +1.20 [-1.40, 4.00] | -0.0021 |
| pi05_spatial_50 | 0.922 | 0.910 | +1.20 [-1.00, 3.40] | -0.0023 |

The largest negative historical-baseline change is groot L10-500: R10 off 0.906 versus current off 0.858 (−4.8 pp). The current run is an H100 completion in r11_knob_4, so this difference cannot be attributed to a 4090-versus-H100 switch. Batch/rollout variation or another historical-to-current difference remains unresolved; use current off for every R11 gain estimate. This revises the interim report’s historical near-pure/default-off classification.

## Future-round hypothesis: a library-only mapping that aims to hit target

1. Reserve whole B-pool episodes **outside** the deployed donor subset (or acquire additional B-only episodes at size 500). Use the complete deployment donor bank, exact deployed metric/action scale, and final pooled score predictor to generate calibration references. Preserve whole-episode cadence and mandatory guard logic. Never self-query a donor episode to claim an honest target match.

2. Fit one cell-wide score CDF and an initial dose from these external B sequences only; validate on a separate B split. Use `k_target=clip((target/v−c_v−c_m*g)/(c_m*(1−g)),0,1)` as an accounting constraint, not as a task-specific rule. Check the guard floor and all-call ceiling. Correct head/CDF transfer by calibrating the actual final head.

3. Retain a random floor and drive total cost with one global committed-ledger feedback state. For a future design, consider a persistent deficit budget across episodes or an analytically faster startup so repeated resets do not systematically lose the first part of each episode. Use bounded probabilities and anti-windup when guards force overspend. Tune any gain/floor/reset convention only on B sequences, freeze it, and measure a new untouched test round. Adaptation changes placement and feasibility: no library-only calculation guarantees exact cost on unknown short closed-loop trajectories.

4. **Operational recommendation from this round:** prefer periodic when an accurate budget knob is required: it is within ±.02 on all 17 arms and improves success over random on common realized-IR support. Keep error_hybrid as the most promising state-dependent cost-efficiency candidate, but do not advertise its target label as delivered IR or claim it beats periodic at equal actual cost. Distance is unsuitable as a general budget knob in this calibration; adaptive improves delivery but does not establish a success advantage over periodic. Disagreement has only four single-point cells and needs a wider frozen curve before a general adoption claim. These recommendations use the unchanged measured methods.

Same-batch off versus historical pure policy provides the default-off context:

| Cell | current off SR | R8 pure SR | off − pure pp [paired interval] | current off IR |
|---|---|---|---|---|
| groot_l10_200 | 0.820 | 0.898 | -7.80 [-12.00, -3.60] | 0.1896 |
| groot_l10_50 | 0.748 | 0.898 | -15.00 [-19.40, -10.60] | 0.2087 |
| pi05_l10_200 | 0.890 | 0.908 | -1.80 [-5.40, 1.80] | 0.1492 |
| groot_spatial_50 | 0.892 | 0.940 | -4.80 [-8.40, -1.40] | 0.1284 |
| pi05_l10_50 | 0.844 | 0.908 | -6.40 [-10.00, -2.60] | 0.1669 |
| groot_l10_500 | 0.858 | 0.898 | -4.00 [-7.80, 0.00] | 0.1816 |
| pi05_l10_500 | 0.906 | 0.908 | -0.20 [-3.60, 3.40] | 0.1483 |
| pi05_spatial_50 | 0.922 | 0.988 | -6.60 [-9.00, -4.20] | 0.1330 |

**Default off:** pi05 L10-500 is the clearest current near-pure case (0.906 versus 0.908 pure; IR 0.148). Pi05 L10-200 is a more cautious cost-first candidate (0.890 versus 0.908); extra spend does raise its observed SR, so it should remain optional rather than assumed useless. Do not blanket-disable groot L10-500: current off is 0.858 versus 0.898 pure, and periodic/error_hybrid reach 0.900/0.896 at IR about 0.239/0.231. Near-pure is a current cell-level judgment, not an equivalence claim or a task-indexed serving parameter.

No item above is a refitted test result. No thresholds, fits, arm settings, serving code, run roots or running processes were changed.

## Executed script and artifacts

[Single entry script](analysis/run.py) produces [provenance.json](analysis/provenance.json), [inventory.csv](analysis/inventory.csv), [arms.csv](analysis/arms.csv), [method_summary.csv](analysis/method_summary.csv), [library_replays.csv](analysis/library_replays.csv), [matched_frontier.csv](analysis/matched_frontier.csv), [matched_arms.csv](analysis/matched_arms.csv), [pooled_frontier.csv](analysis/pooled_frontier.csv), [efficiency.csv](analysis/efficiency.csv), [hardware.json](analysis/hardware.json), placement tables and the two PNG figures. Final paired checks are in [efficiency_comparisons.csv](analysis/efficiency_comparisons.csv), [target_crosscheck.csv](analysis/target_crosscheck.csv), [baseline_comparisons.csv](analysis/baseline_comparisons.csv), and [placement_pooled.csv](analysis/placement_pooled.csv). [Numerical validation](analysis/numerical_validation.json) checks serving feature and distance arithmetic. The command at the top was executed. The `--final` flag requires exactly 94 unique completed test arms and fully reconciled decision logs before producing **ANALYSIS.md**. Without that flag the pipeline writes ANALYSIS_INTERIM.md. No serving process is started; the write fence confines outputs to this owner directory.
