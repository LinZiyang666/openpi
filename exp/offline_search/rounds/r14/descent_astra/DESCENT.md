# 给负责人的结果说明

主规则：每个任务从纯推理逐级降低，在第一次低于纯推理成功率减 5 个百分点时停止，保留上一档。每任务使用相同的 30 个初始状态。以下是开发集上的选档结果。

档位从低到高为 off、ir25、ir32、ir40、ir45、c10；c10 表示纯推理。ir 数字是档位名称，实际算力比例见后表。

| 模型 / 任务组 | 任务 0 | 任务 1 | 任务 2 | 任务 3 | 任务 4 | 任务 5 | 任务 6 | 任务 7 | 任务 8 | 任务 9 | 统一档位 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| π0.5 / 长程任务 | ir25 | off | ir45 | off | ir45 | ir25 | off | off | off | off | ir25 |
| π0.5 / 空间任务 | ir40 | ir45 | off | off | off | ir45 | off | off | c10 | ir25 | off |
| GR00T / 长程任务 | ir45 | c10 | c10 | c10 | ir32 | off | ir40 | c10 | c10 | ir45 | ir45 |
| GR00T / 空间任务 | ir45 | off | off | ir45 | c10 | c10 | off | ir25 | off | off | off |

统一档位也按同一容差逐级下降，但它检查整个任务组的平均成功率，允许任务间相互抵消。因此两种选法不保证相同的实测成功率。

| 模型 / 任务组 | 纯推理成功率 % | 逐任务成功率 % | 统一成功率 % | 逐任务 IR | 统一 IR | 节省 IR 百分点 | 相对节省 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| π0.5 / 长程任务 | 88.0 | 90.0 | 85.7 | 0.226 | 0.236 | 0.94 | 4.0 |
| π0.5 / 空间任务 | 95.3 | 96.0 | 91.3 | 0.263 | 0.134 | -12.87 | -95.9 |
| GR00T / 长程任务 | 90.7 | 90.7 | 86.0 | 0.443 | 0.450 | 0.71 | 1.6 |
| GR00T / 空间任务 | 91.0 | 90.7 | 88.0 | 0.275 | 0.130 | -14.48 | -111.2 |

IR 按冻结的原始成本系数计算，表示归一化计算开销。正数表示节省，负数表示开销增加。逐任务结果由所选档位的完整任务轨迹拼合，并非已经独立运行的混合策略。选档与评估使用同一开发集，节省和成功率可能偏乐观。

库估计是否吻合：下表同时给出优先级与所需档位的排序相关性，以及原有预算分配对应档位的吻合比例。相关性为正表示方向一致；NA 表示档位全相同等情况，无法计算相关性。原有分配不是成功率边界预测，因此不能只凭其中一项宣布“预测正确”。

| 模型 / 任务组 | 原预算 | 排序相关 ρ | 排序相关 τ | 档位相同 % | 相差至多一档 % | 平均差几档 |
| --- | --- | --- | --- | --- | --- | --- |
| π0.5 / 长程任务 | ir32 | -0.110 | -0.056 | 0 | 20 | 1.90 |
| π0.5 / 长程任务 | ir40 | -0.110 | -0.056 | 0 | 20 | 2.60 |
| π0.5 / 空间任务 | ir32 | 0.279 | 0.205 | 0 | 20 | 1.90 |
| π0.5 / 空间任务 | ir40 | 0.279 | 0.205 | 0 | 40 | 2.20 |
| GR00T / 长程任务 | ir32 | 0.668 | 0.511 | 10 | 30 | 2.10 |
| GR00T / 长程任务 | ir40 | 0.668 | 0.511 | 20 | 40 | 1.50 |
| GR00T / 空间任务 | ir32 | 0.417 | 0.337 | 0 | 10 | 2.10 |
| GR00T / 空间任务 | ir40 | 0.417 | 0.337 | 0 | 30 | 2.20 |

优先级来自库内状态；预算偏置曾使用较早的开发集状态轨迹校准。有限上限映射无法预测完全关闭档位。

原有逐任务分配与统一周期策略的比较如下。相同名义预算不保证实际开销相同，实际 IR 差在 1 个百分点以内才标为近似匹配。精确匹配一栏是相邻统一周期档位的期望混合，仅作估算。

| 模型 / 任务组 | 原预算 | 逐任务 IR | 统一 IR | IR 差 百分点 | 近似匹配 | 成功率差 百分点 | 精确 IR 期望混合成功率差 百分点 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| π0.5 / 长程任务 | ir32 | 0.308 | 0.308 | 0.09 | 是 | -1.67 | -1.66 |
| π0.5 / 长程任务 | ir40 | 0.392 | 0.390 | 0.18 | 是 | 1.00 | 0.94 |
| π0.5 / 空间任务 | ir32 | 0.311 | 0.314 | -0.33 | 是 | 0.67 | 0.68 |
| π0.5 / 空间任务 | ir40 | 0.402 | 0.397 | 0.48 | 是 | 0.33 | 0.30 |
| GR00T / 长程任务 | ir32 | 0.321 | 0.323 | -0.16 | 是 | 1.33 | 1.50 |
| GR00T / 长程任务 | ir40 | 0.395 | 0.399 | -0.42 | 是 | 3.33 | 3.28 |
| GR00T / 空间任务 | ir32 | 0.321 | 0.319 | 0.20 | 是 | -2.33 | -2.26 |
| GR00T / 空间任务 | ir40 | 0.402 | 0.400 | 0.21 | 是 | 0.33 | 0.19 |

每任务 30 次试验的区间很宽。0 与 2 个百分点容差的任务表相同；5 个百分点允许净少成功一次。点估计达标不等于已统计证明不劣。配对下界、首次失败后出现的回升、所有容差及置信规则敏感性见下文。开发集存在历史复用，本报告不使用测试集。

## Audit and alternate tolerances

Frozen: `2026-10-05T04:13:48.459717+00:00`. Outcome inputs and code hashes are in `provenance.json`. All values are computed from accepted B episodes; no test-A evaluation was read or executed.

### Tolerance 0 pp

| Cell | Task | Selected | First fail | Lower recovery ignored | Gain/loss | ΔSR pp | Lower95 pp | Simultaneous lower95 pp | Bound sensitivity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 0 | ir32 | ir25 | none | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_l10_50 | 1 | ir40 | ir32 | ir25 | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_l10_50 | 2 | c10 | ir45 | off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_l10_50 | 3 | off | none | none | 4/3 | 3.33 | -22.77 | -41.48 | c10 |
| pi05_l10_50 | 4 | ir45 | ir40 | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| pi05_l10_50 | 5 | c10 | ir45 | ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_l10_50 | 6 | ir32 | ir25 | off | 5/4 | 3.33 | -25.08 | -44.98 | c10 |
| pi05_l10_50 | 7 | ir40 | ir32 | ir25 | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| pi05_l10_50 | 8 | off | none | none | 10/3 | 23.33 | -9.24 | -33.80 | c10 |
| pi05_l10_50 | 9 | off | none | none | 4/3 | 3.33 | -22.77 | -41.48 | c10 |
| pi05_spatial_50 | 0 | ir40 | ir32 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| pi05_spatial_50 | 1 | ir45 | ir40 | ir32,ir25,off | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 2 | ir32 | ir25 | none | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 3 | ir40 | ir32 | off | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 4 | off | none | none | 4/2 | 6.67 | -18.32 | -36.78 | c10 |
| pi05_spatial_50 | 5 | ir45 | ir40 | ir25 | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| pi05_spatial_50 | 6 | off | none | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| pi05_spatial_50 | 7 | c10 | ir45 | ir32 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_spatial_50 | 8 | c10 | ir45 | ir40,ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_spatial_50 | 9 | ir25 | off | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_l10_50 | 0 | ir45 | ir40 | none | 2/2 | 0.00 | -21.26 | -37.58 | c10 |
| groot_l10_50 | 1 | c10 | ir45 | ir40 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 2 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 3 | c10 | ir45 | ir40,ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 4 | ir32 | ir25 | none | 4/4 | 0.00 | -26.97 | -45.78 | c10 |
| groot_l10_50 | 5 | ir32 | ir25 | off | 2/2 | 0.00 | -21.26 | -37.58 | c10 |
| groot_l10_50 | 6 | ir45 | ir40 | none | 3/3 | 0.00 | -24.42 | -42.02 | c10 |
| groot_l10_50 | 7 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 8 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 9 | ir45 | ir40 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 0 | ir45 | ir40 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 1 | ir45 | ir40 | none | 3/3 | 0.00 | -24.42 | -42.02 | c10 |
| groot_spatial_50 | 2 | off | none | none | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| groot_spatial_50 | 3 | c10 | ir45 | ir32,off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 4 | c10 | ir45 | ir40,ir25,off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 5 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 6 | ir25 | off | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 7 | ir25 | off | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_spatial_50 | 8 | ir45 | ir40 | ir32,ir25,off | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_spatial_50 | 9 | ir32 | ir25 | none | 3/3 | 0.00 | -24.42 | -42.02 | c10 |

| Cell | Uniform | Bound uniform | IR saved pp | Conditional 95% interval pp | Task cost/episode | Uniform cost/episode |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | ir32 | c10 | -1.23 | [-1.79, -0.70] | 17.578 | 17.006 |
| pi05_spatial_50 | ir25 | c10 | -10.44 | [-11.03, -9.85] | 7.594 | 5.344 |
| groot_l10_50 | c10 | c10 | 4.58 | [4.40, 4.78] | 26.090 | 28.400 |
| groot_spatial_50 | ir45 | c10 | 6.66 | [6.19, 7.13] | 8.854 | 10.387 |

### Tolerance 2 pp

| Cell | Task | Selected | First fail | Lower recovery ignored | Gain/loss | ΔSR pp | Lower95 pp | Simultaneous lower95 pp | Bound sensitivity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 0 | ir32 | ir25 | none | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_l10_50 | 1 | ir40 | ir32 | ir25 | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_l10_50 | 2 | c10 | ir45 | off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_l10_50 | 3 | off | none | none | 4/3 | 3.33 | -22.77 | -41.48 | c10 |
| pi05_l10_50 | 4 | ir45 | ir40 | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| pi05_l10_50 | 5 | c10 | ir45 | ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_l10_50 | 6 | ir32 | ir25 | off | 5/4 | 3.33 | -25.08 | -44.98 | c10 |
| pi05_l10_50 | 7 | ir40 | ir32 | ir25 | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| pi05_l10_50 | 8 | off | none | none | 10/3 | 23.33 | -9.24 | -33.80 | c10 |
| pi05_l10_50 | 9 | off | none | none | 4/3 | 3.33 | -22.77 | -41.48 | c10 |
| pi05_spatial_50 | 0 | ir40 | ir32 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| pi05_spatial_50 | 1 | ir45 | ir40 | ir32,ir25,off | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 2 | ir32 | ir25 | none | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 3 | ir40 | ir32 | off | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 4 | off | none | none | 4/2 | 6.67 | -18.32 | -36.78 | c10 |
| pi05_spatial_50 | 5 | ir45 | ir40 | ir25 | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| pi05_spatial_50 | 6 | off | none | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| pi05_spatial_50 | 7 | c10 | ir45 | ir32 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_spatial_50 | 8 | c10 | ir45 | ir40,ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_spatial_50 | 9 | ir25 | off | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_l10_50 | 0 | ir45 | ir40 | none | 2/2 | 0.00 | -21.26 | -37.58 | c10 |
| groot_l10_50 | 1 | c10 | ir45 | ir40 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 2 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 3 | c10 | ir45 | ir40,ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 4 | ir32 | ir25 | none | 4/4 | 0.00 | -26.97 | -45.78 | c10 |
| groot_l10_50 | 5 | ir32 | ir25 | off | 2/2 | 0.00 | -21.26 | -37.58 | c10 |
| groot_l10_50 | 6 | ir45 | ir40 | none | 3/3 | 0.00 | -24.42 | -42.02 | c10 |
| groot_l10_50 | 7 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 8 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 9 | ir45 | ir40 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 0 | ir45 | ir40 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 1 | ir45 | ir40 | none | 3/3 | 0.00 | -24.42 | -42.02 | c10 |
| groot_spatial_50 | 2 | off | none | none | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| groot_spatial_50 | 3 | c10 | ir45 | ir32,off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 4 | c10 | ir45 | ir40,ir25,off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 5 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 6 | ir25 | off | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 7 | ir25 | off | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_spatial_50 | 8 | ir45 | ir40 | ir32,ir25,off | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_spatial_50 | 9 | ir32 | ir25 | none | 3/3 | 0.00 | -24.42 | -42.02 | c10 |

| Cell | Uniform | Bound uniform | IR saved pp | Conditional 95% interval pp | Task cost/episode | Uniform cost/episode |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | ir32 | c10 | -1.23 | [-1.79, -0.70] | 17.578 | 17.006 |
| pi05_spatial_50 | ir25 | c10 | -10.44 | [-11.03, -9.85] | 7.594 | 5.344 |
| groot_l10_50 | c10 | c10 | 4.58 | [4.40, 4.78] | 26.090 | 28.400 |
| groot_spatial_50 | ir45 | c10 | 6.66 | [6.19, 7.13] | 8.854 | 10.387 |

### Tolerance 5 pp

| Cell | Task | Selected | First fail | Lower recovery ignored | Gain/loss | ΔSR pp | Lower95 pp | Simultaneous lower95 pp | Bound sensitivity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 0 | ir25 | off | none | 1/2 | -3.33 | -21.99 | -37.64 | c10 |
| pi05_l10_50 | 1 | off | none | none | 0/1 | -3.33 | -17.22 | -32.33 | c10 |
| pi05_l10_50 | 2 | ir45 | ir40 | off | 0/1 | -3.33 | -17.22 | -32.33 | c10 |
| pi05_l10_50 | 3 | off | none | none | 4/3 | 3.33 | -22.77 | -41.48 | c10 |
| pi05_l10_50 | 4 | ir45 | ir40 | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| pi05_l10_50 | 5 | ir25 | off | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| pi05_l10_50 | 6 | off | none | none | 4/4 | 0.00 | -26.97 | -45.78 | c10 |
| pi05_l10_50 | 7 | off | none | none | 1/2 | -3.33 | -21.99 | -37.64 | c10 |
| pi05_l10_50 | 8 | off | none | none | 10/3 | 23.33 | -9.24 | -33.80 | c10 |
| pi05_l10_50 | 9 | off | none | none | 4/3 | 3.33 | -22.77 | -41.48 | c10 |
| pi05_spatial_50 | 0 | ir40 | ir32 | off | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| pi05_spatial_50 | 1 | ir45 | ir40 | ir32,ir25,off | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 2 | off | none | none | 0/1 | -3.33 | -17.22 | -32.33 | c10 |
| pi05_spatial_50 | 3 | off | none | none | 0/0 | 0.00 | -11.57 | -25.89 | c10 |
| pi05_spatial_50 | 4 | off | none | none | 4/2 | 6.67 | -18.32 | -36.78 | c10 |
| pi05_spatial_50 | 5 | ir45 | ir40 | ir32,ir25 | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| pi05_spatial_50 | 6 | off | none | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| pi05_spatial_50 | 7 | off | none | none | 0/1 | -3.33 | -17.22 | -32.33 | c10 |
| pi05_spatial_50 | 8 | c10 | ir45 | ir40,ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| pi05_spatial_50 | 9 | ir25 | off | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_l10_50 | 0 | ir45 | ir40 | none | 2/2 | 0.00 | -21.26 | -37.58 | c10 |
| groot_l10_50 | 1 | c10 | ir45 | ir40 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 2 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 3 | c10 | ir45 | ir40,ir32,ir25 | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 4 | ir32 | ir25 | none | 4/4 | 0.00 | -26.97 | -45.78 | c10 |
| groot_l10_50 | 5 | off | none | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_l10_50 | 6 | ir40 | ir32 | none | 4/5 | -3.33 | -30.97 | -49.78 | c10 |
| groot_l10_50 | 7 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 8 | c10 | ir45 | none | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_l10_50 | 9 | ir45 | ir40 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 0 | ir45 | ir40 | none | 1/1 | 0.00 | -17.13 | -32.33 | c10 |
| groot_spatial_50 | 1 | off | none | none | 3/4 | -3.33 | -28.61 | -46.32 | c10 |
| groot_spatial_50 | 2 | off | none | none | 1/0 | 3.33 | -11.49 | -25.89 | c10 |
| groot_spatial_50 | 3 | ir45 | ir40 | ir32,off | 0/1 | -3.33 | -17.22 | -32.33 | c10 |
| groot_spatial_50 | 4 | c10 | ir45 | ir40,ir32,ir25,off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 5 | c10 | ir45 | ir25,off | 0/0 | 0.00 | 0.00 | 0.00 | c10 |
| groot_spatial_50 | 6 | off | none | none | 1/2 | -3.33 | -21.99 | -37.64 | c10 |
| groot_spatial_50 | 7 | ir25 | off | none | 2/1 | 3.33 | -16.40 | -32.28 | c10 |
| groot_spatial_50 | 8 | off | none | none | 3/2 | 3.33 | -19.96 | -37.32 | c10 |
| groot_spatial_50 | 9 | off | none | none | 3/4 | -3.33 | -28.61 | -46.32 | c10 |

| Cell | Uniform | Bound uniform | IR saved pp | Conditional 95% interval pp | Task cost/episode | Uniform cost/episode |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | ir25 | c10 | 0.94 | [0.19, 1.67] | 12.943 | 13.796 |
| pi05_spatial_50 | off | c10 | -12.87 | [-13.63, -12.09] | 5.808 | 3.126 |
| groot_l10_50 | ir45 | c10 | 0.71 | [0.42, 1.00] | 25.290 | 26.261 |
| groot_spatial_50 | off | c10 | -14.48 | [-15.15, -13.78] | 6.408 | 3.141 |

## Interpretation limits

The exact paired task intervals use discordant gains/losses and conservative Clopper–Pearson tails. Cell intervals are stratified with Bonferroni correction over tasks. The simultaneous lower band covers all 200 non-reference task/level comparisons under within-task sampling assumptions. The bound-based choices are diagnostics only. The full numerical record includes every tested level, confusion matrix, original stage-ledger IR and nominal allocation comparison interval.

Savings and mixture bootstrap intervals condition on the selected levels and mixture weights; they omit selection uncertainty. The expected mixture is randomized across whole episodes, with the weight solved from total cost and total decision counts, not a linear interpolation of nominal IR labels. No mixture is extrapolated beyond observed periodic IRs. These development data do not establish independent deployment SR or rank predictiveness.
