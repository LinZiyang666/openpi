# 给负责人的说明

我建议先试一种有短期记忆的单相机检索：当前只编码腕部相机，保留最近一次完整观察中的外部相机信息，再加入这段时间机器人真实移动了多少。是否少看一个相机，由示范库里直接学出的“少看是否安全”来决定，不按抓取阶段划分。

离线结果有希望，但还不能说成功率不掉。直接把旧相机特征塞进原来的检索器，误差明显变大；重新训练适配这种输入的检索度量和动作修正器，效果好得多。在六个 π0.5 库大小与任务组合中，安全门选择了约 **28.1%–37.1%** 的可评估观察只用腕部相机。最终动作误差相对完整相机的变化为 **+0.12% 到 +1.07%**。这些数字来自示范库回放，不能换算成闭环成功率。

最需要小跑检查的是守卫。动作很接近时，检索到的示范进度仍可能不同，导致本该调用大模型时没有调用。把进度一致性加入安全标签后有所改善，风险仍在。早期的数据请求已写好；现有开发集日志没有相机特征，无法补出这个答案。

开销空间也不大：理想情况下每隔一个控制时隙观察一次，全部改成单相机最多省 **0.0437** 的总推理比例；本方案要求单相机之后恢复完整观察，因此上限再减半。当前按请求切相机的完整观察路径还有额外开销。成功率只要稍微下降，补回它所需的策略调用就可能吃掉节省。因此我的预测是“原始视觉成本可能下降，匹配成功率后的总成本优势尚未确定”，而不是已经达标。

另一模型只有离线特征结果，尚没有验证过的真正单相机执行路径，也没有相应价格。不能把它列为已经省钱的方案。另外，加入机器人位移也能改善完整相机基线，实验必须把这个对照带上。

建议先做开发集配对小跑，核对守卫时机、真实相机成本和补调用成本；通过后再冻结官方测试格子。完整设计、预测和实验计划都已交接。

# Technical handback

Primary: pre-encoding, progress-aware safety atlas plus two-slot camera memory and observed displacement. Library-only task metric/head fits belong to layer 5; gate thresholds are task-agnostic. Prototype reference and fit/evidence code are included, but no deployable plugin is claimed.

Evidence covers the four cells at nested sizes 50/200/500, alternative token/image features on size 50, correction compatibility, fair full-plus-displacement controls, causal atlas replay, and accepted B-dev cost reconciliation. No R12 other-explorer data or A recordings were read. No launches, GPU work, or writes outside this directory occurred in the analysis.

Primary numerical evidence:

| cell / size | half looks % | corrected MSE change % | MSE delta | episode bootstrap interval | grip disagreement % | missed stall % | extra stall % |
|---|---|---|---|---|---|---|---|
| groot_l10_200 | 16.5 | -0.51 | -0.0012 | [-0.0019, -0.0005] | 0.10 | 1.30 | 0.94 |
| groot_l10_50 | 20.1 | 0.12 | 0.0004 | [-0.0028, 0.0037] | 0.29 | 1.26 | 1.01 |
| groot_l10_500 | 15.6 | -0.37 | -0.0007 | [-0.0011, -0.0004] | 0.05 | 1.38 | 0.87 |
| groot_spatial_200 | 25.4 | 0.19 | 0.0004 | [-0.0009, 0.0019] | 0.07 | 0.83 | 0.68 |
| groot_spatial_50 | 28.3 | -0.05 | -0.0002 | [-0.0039, 0.0038] | 0.48 | 1.43 | 0.44 |
| groot_spatial_500 | 22.2 | -0.30 | -0.0006 | [-0.0012, -0.0000] | 0.03 | 0.67 | 0.26 |
| pi05_l10_200 | 30.7 | 0.18 | 0.0003 | [-0.0003, 0.0009] | 0.13 | 1.73 | 0.71 |
| pi05_l10_50 | 28.1 | 0.12 | 0.0003 | [-0.0030, 0.0032] | 0.39 | 1.43 | 0.97 |
| pi05_l10_500 | 30.6 | 0.57 | 0.0007 | [0.0003, 0.0013] | 0.11 | 1.65 | 0.91 |
| pi05_spatial_200 | 35.0 | 0.36 | 0.0007 | [-0.0005, 0.0020] | 0.16 | 0.44 | 0.30 |
| pi05_spatial_50 | 28.2 | 0.73 | 0.0027 | [-0.0033, 0.0077] | 0.44 | 0.86 | 0.85 |
| pi05_spatial_500 | 37.1 | 1.07 | 0.0016 | [0.0009, 0.0023] | 0.14 | 0.38 | 0.32 |


OFFLINE.md documents the fixed-PCA and correction-calibration approximations. PREDICTION.md gives per-cell raw/conditional matched-SR forecasts with explicit assumptions; it does not manufacture closed-loop SR evidence. SPEC.md is implementation-ready for sol, EXPERIMENTS.md freezes the B-first workflow, and REQUESTS.md lists the missing recordings/cost measurements.
