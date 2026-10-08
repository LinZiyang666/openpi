# 给负责人的结论

先在开发集按“每任务成功率最多下降 5 个百分点、首次不达标即停”逐级选档，并冻结逐任务表和统一档位；随后在测试集独立执行三种策略，每组每种策略 500 回合。测试结果只作确认，没有重新选档。

| 模型 / 任务组 | 任务 0 | 任务 1 | 任务 2 | 任务 3 | 任务 4 | 任务 5 | 任务 6 | 任务 7 | 任务 8 | 任务 9 | 统一档位 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| π0.5 / 长程任务 | ir25 | off | ir45 | off | ir45 | ir25 | off | off | off | off | ir25 |
| π0.5 / 空间任务 | ir40 | ir45 | off | off | off | ir45 | off | off | c10 | ir25 | off |
| GR00T / 长程任务 | ir45 | c10 | c10 | c10 | ir32 | off | ir40 | c10 | c10 | ir45 | ir45 |
| GR00T / 空间任务 | ir45 | off | off | ir45 | c10 | c10 | off | ir25 | off | off | off |

c10 是纯推理；off 关闭额外旋钮调用，仍保留视觉和必要推理；ir 数字是档位名，不是实测算力。下表每格为“成功率 % / IR”，IR 越低越省；节省为负表示更费。

| 模型 / 任务组 | 纯推理 | 统一档位 | 逐任务表 | 逐任务超容差数 | 比纯推理省 IR 百分点 | 比统一省 IR 百分点 |
| --- | --- | --- | --- | --- | --- | --- |
| π0.5 / 长程任务 | 90.80 / 0.504 | 85.60 / 0.236 | 90.40 / 0.226 | 3 | 27.77 | 0.97 |
| π0.5 / 空间任务 | 98.60 / 0.512 | 91.80 / 0.133 | 95.20 / 0.260 | 3 | 25.14 | -12.69 |
| GR00T / 长程任务 | 88.00 / 0.504 | 87.20 / 0.448 | 85.60 / 0.442 | 3 | 6.14 | 0.60 |
| GR00T / 空间任务 | 95.40 / 0.512 | 92.60 / 0.121 | 92.80 / 0.272 | 2 | 23.95 | -15.07 |
| 汇总 | 93.20 / 0.506 | 89.30 / 0.281 | 91.00 / 0.317 | 11 | 18.94 | -3.57 |

（1）没有守住“每个任务都在容差内”。逐任务表有 11/40 个模型×任务的实测下降超过 5 个百分点，统一档位也有 11/40 个。逐任务表汇总比纯推理少成功 2.20 个百分点，配对 95% 区间为 [-3.65, -0.75]，相对节省 IR 37.44%；比统一档位多成功 1.70 个百分点，但 IR 增加 12.73%。平均达标不能代替逐任务达标。

（2）库估计只能提供有限的排序线索，尚不能可靠决定档位。四组排序相关依次为 -0.110、0.279、0.668、0.417；两种预算映射的档位完全吻合率为 0–20%。有限预算上限本来就不能预测 off；估计还用过较早的开发集校准。本次测试只运行冻结表，不能反推新的任务边界。

（3）开发集对成功率收益偏乐观，算力预测较准。逐任务表汇总成功率预测 91.83%，测试为 91.00%，高估 0.83 个百分点；扣除纯推理基线变化后，相对优势高估 2.78 个百分点。IR 预测 0.3161，实测 0.3165。这些差额也包含初始状态、单次运行及轨迹拼合与实际执行的差异，不能全算作选档偏差。

第 5 层建议：把逐任务降档保留为开发工具，暂不作为“逐任务守住成功率”的部署保证。下一轮只在开发集增加独立确认与重复运行、采用更保守的逐任务规则，再用新的未用评估数据确认；库估计只辅助安排试验。本次表保持冻结，不能按测试中的坏任务补档。

## Failure detail and limits

| Cell | Tasks below −5 pp | Task-descent ΔSR pp / exact paired lower95 pp |
| --- | --- | --- |
| pi05_l10_50 | 0, 4, 6 | 0: -8.00 / -21.76; 4: -6.00 / -19.18; 6: -14.00 / -35.81 |
| pi05_spatial_50 | 5, 6, 9 | 5: -8.00 / -19.23; 6: -8.00 / -19.23; 9: -10.00 / -21.81 |
| groot_l10_50 | 4, 5, 6 | 4: -6.00 / -23.06; 5: -10.00 / -21.81; 6: -10.00 / -30.39 |
| groot_spatial_50 | 1, 5 | 1: -10.00 / -26.25; 5: -14.00 / -29.06 |

All four cell-average task-descent point losses are within the registered tolerance, while each cell has task violations. Only 1/40 task comparisons has a marginal conservative paired lower bound at or above −5 pp; 0/40 meet the simultaneous lower-bound criterion. The point violations are observed losses, not forty individually proven population effects. Neither the cell means nor the bootstrap interval for the pooled effect certify every task.

GR00T spatial task 5 retains C10, yet its success changes from 98.00% to 84.00%. B stitching reused the same reference trajectory for a C10-selected task; A reruns it inside the mixed arm. This shows why not every observed loss can be attributed to a reduced knob. One run per arm cannot separate stochastic/runtime variation from controller effects. Do not delete this failure or substitute its pure outcome.

π0.5 long-task descent is the most promising cell-level tradeoff versus uniform: ΔSR 4.80 pp with interval [1.40, 8.20] and IR saving 0.97 pp, but its every-task requirement still fails. Both spatial cells pay substantially more than uniform off; GR00T spatial gains only 0.20 pp SR.

GR00T long-task descent lowers the IR ratio versus uniform by 0.60 pp, but has lower SR and increases normalized cost per episode by 0.27%. Its two arms have the same total policy-call count (12694); descent uses more requests. A lower ratio alone is not a reduction in total episode work. No same-realized-IR superiority claim follows from these three-arm A results.

## How optimistic was B?

| Cell | Descent B−A SR pp | B descent−pure pp | A descent−pure pp | Pure-relative optimism pp | B descent IR | A descent IR | A−B IR pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | -0.40 | 2.00 | -0.40 | 2.40 | 0.226420 | 0.226281 | -0.01 |
| pi05_spatial_50 | 0.80 | 0.67 | -3.40 | 4.07 | 0.262918 | 0.260189 | -0.27 |
| groot_l10_50 | 5.07 | 0.00 | -2.40 | 2.40 | 0.443268 | 0.442240 | -0.10 |
| groot_spatial_50 | -2.13 | -0.33 | -2.60 | 2.27 | 0.274974 | 0.272090 | -0.29 |
| pooled | 0.83 | 0.58 | -2.20 | 2.78 | 0.316073 | 0.316519 | 0.04 |

The pooled advantage over uniform shrinks from 4.08 pp on B to 1.70 pp on A. Negative raw B−A SR gaps and negative IR-saving optimism are retained in [TEST_A.md](TEST_A.md). The pooled IR may move in a different direction from every cell IR because each pool uses its own decision-count weights. B and A outcomes are not paired across pools. These are observed discrepancies, with no claim to isolate a causal winner’s-curse term.

## Library agreement with the frozen B descent

| Cell | Frozen allocation target | Spearman ρ | Kendall τ-b | Exact level % | Within one level % | Mean absolute ordinal error |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | ir32 | -0.110 | -0.056 | 0 | 20 | 1.90 |
| pi05_l10_50 | ir40 | -0.110 | -0.056 | 0 | 20 | 2.60 |
| pi05_spatial_50 | ir32 | 0.279 | 0.205 | 0 | 20 | 1.90 |
| pi05_spatial_50 | ir40 | 0.279 | 0.205 | 0 | 40 | 2.20 |
| groot_l10_50 | ir32 | 0.668 | 0.511 | 10 | 30 | 2.10 |
| groot_l10_50 | ir40 | 0.668 | 0.511 | 20 | 40 | 1.50 |
| groot_spatial_50 | ir32 | 0.417 | 0.337 | 0 | 10 | 2.10 |
| groot_spatial_50 | ir40 | 0.417 | 0.337 | 0 | 30 | 2.20 |

This reproduces the sealed B result, including both budgets and all cells. Priorities are library-derived; cap bias also used earlier B state-path calibration. Caps are budget allocations rather than calibrated SR tolerance boundaries, and the registered finite-cap mapping cannot output off. With ten tasks per cell, ties and discrete levels limit the rank evidence. A validates only the selected controller, not the unmeasured full A descent curve.

For context, the original frozen library allocation was also compared with uniform periodic on B:

| Cell | Target | Allocation−uniform IR pp | Paired ΔSR pp | Exact-IR expected-mixture ΔSR pp |
| --- | --- | --- | --- | --- |
| pi05_l10_50 | ir32 | 0.09 | -1.67 | -1.66 |
| pi05_l10_50 | ir40 | 0.18 | 1.00 | 0.94 |
| pi05_spatial_50 | ir32 | -0.33 | 0.67 | 0.68 |
| pi05_spatial_50 | ir40 | 0.48 | 0.33 | 0.30 |
| groot_l10_50 | ir32 | -0.16 | 1.33 | 1.50 |
| groot_l10_50 | ir40 | -0.42 | 3.33 | 3.28 |
| groot_spatial_50 | ir32 | 0.20 | -2.33 | -2.26 |
| groot_spatial_50 | ir40 | 0.21 | 0.33 | 0.19 |

These small B effects have mixed signs. All nominal pairs meet the previously registered approximate IR-gap criterion; the exact-IR comparison is an expected mixture of adjacent uniform trajectories, not an observed deterministic controller. Neither it nor the rank table demonstrates a general deployment benefit.

## Reproducibility and scope

The confirmation covers the four model×suite cells at library size 50. It does not answer the broader layer-5 component-ablation or larger-library questions. All A runs were on the local 4090 (the intended H100 host became unavailable); the R11 π0.5 host-insensitivity finding is inherited from the supplied handoff. No fresh host-equivalence claim or wall-time saving is inferred.

[TEST_A.md](TEST_A.md) contains every arm’s N/V/M, frozen-price SR/IR, 5,000-draw paired cluster-bootstrap intervals (seed 20261005), all task lower bounds, signed savings and full B/A comparisons. [test_a.json](test_a.json) contains episode arrays and hashed provenance. [PROTOCOL.md](PROTOCOL.md), [DESCENT.md](DESCENT.md), [levels.json](levels.json) and [uniform.json](uniform.json) remain unchanged. [analyze_test_a.py](analyze_test_a.py) regenerates this report and the handback as well as the measurement outputs.
