# 给负责人的说明

按新的算力计价，两种模型只看一个相机都能省下一半视觉开销。此前把 GR00T 因为单请求延迟而排除的判断作废。但这轮还没有证据证明“成功率不掉且总开销下降”，也没有证明额外调用能划算地补回成功率。

我重新检查了 700 条 B 开发集轨迹、17,367 次观察，包含两种模型的小库、大库和刚完成的定期调用记录。它们都是原来的完整双相机控制器产生的轨迹；新方法只做离线影子判断，没有驱动机器人。旧方案的问题已经很清楚：动作看起来接近，并不保证检索进度相同，守卫会因此漏掉本来要做的调用。不能把漏调用带来的开销下降当成收益。

这里的统计只使用协调者当前认可的录制。空间任务采用本机重新录制的数据；被撤回的旧空间任务和旧中等库录制不再作为证据。预测文件保留旧段落，并追加了撤回说明和新的有效数据预测。

现在建议的小跑方案是：先看腕部相机，用最近一次完整观察和机器人位移估计另一相机；同时做一个只依赖当前腕部信息的独立检索。只有两个估计与拟执行动作、夹爪和进度都一致，才接受少看一半，否则补看另一相机并回到完整检索。这样直接检查当前观察，比只按上一次检索结果预判安全更可靠。所有检索度量和修正器仍只用所选 B 示范库拟合；新检查规则和阈值的选择明确属于冻结后的开发集调整。

在现有固定轨迹上，π0.5 的主要方案可省总 IR 0.001564–0.003491。它大幅减少了守卫差异，但部分格子仍会漏调用。更严格的版本在这些记录中没有出现漏调用、额外调用或夹爪符号差异，代价是省下的量更小；这仍不能证明闭环成功率不掉。

GR00T 的问题现在是准确性和数值路径，而不是定价。我实际比较了 160 组 B 开发图像重建后的联合编码与单相机编码。特征差异确实存在；进一步要求检索对这些差异稳定后，可接受的半观察更少。录下来的小图不能还原原始输入，所以这只是数值敏感性证据。补看另一相机后，分别编码得到的策略输入也不保证等于原来的联合编码，正式小跑必须把这一点核实。

定期调用记录虽然齐了，但和关闭调用的记录只重合 9 个初始状态，不能据此确定可靠的局部“多花多少开销能补回多少成功率”。因此建议先做严格配对的 B 小跑，优先检查 π0.5 的空间任务；GR00T 先通过真实单相机输入的数值检查。只有测到相同成功率下的总 IR 优势，才进入正式测试。

# Technical handback

Updated UTC: 2026-10-03T18:28:25.598035+00:00. DIRECTIVE_2 supersedes the prior latency economics. Primary pilot: wrist-first two-witness completion gate (`consensus2_r0.1`); strict diagnostic: the same gate at `.05`. The frozen pre-look atlas remains a rejected baseline, not a deployment recommendation. Camera choice, broad missing-view ensembles, and numerical sensitivity were also evaluated.

The table is fixed-state shadow evidence. The SR column belongs to the **full baseline**. Missed/extra calls are propagated fixed-schedule diagnostics, not actual new-controller outcomes. GR00T rows use joint-path recording keys and require the numerical-path qualification described above.

| cell / knob | N / looks / calls | baseline SR / IR | half looks | fixed-call IR saved | missed / extra calls | grip-disagreement looks |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 5722 / 2880 / 736 | 0.870 / 0.184081 | 69 | 0.000854 | 3 / 0 | 0 |
| π0.5 l10-400 / off | 5182 / 2615 / 325 | 0.970 / 0.129888 | 243 | 0.003491 | 5 / 2 | 0 |
| GR00T l10-50 / off | 6734 / 3384 / 1085 | 0.740 / 0.211650 | 72 | 0.000736 | 5 / 0 | 3 |
| π0.5 l10-50 / off | 6352 / 3198 / 754 | 0.800 / 0.177186 | 194 | 0.002309 | 2 / 1 | 1 |
| GR00T spatial-50 / off | 2497 / 1270 / 210 | 0.840 / 0.146928 | 63 | 0.001867 | 0 / 0 | 0 |
| π0.5 spatial-50 / off | 2391 / 1221 / 189 | 0.880 / 0.144652 | 80 | 0.002543 | 0 / 0 | 0 |
| GR00T l10-50 / periodic .32 | 2883 / 1450 / 820 | 0.880 / 0.316767 | 8 | 0.000205 | 0 / 0 | 0 |
| π0.5 l10-50 / periodic .32 | 2673 / 1349 / 723 | 0.940 / 0.306080 | 55 | 0.001564 | 0 / 0 | 0 |


The complete fit and acquisition contract is in [SPEC.md](SPEC.md), uncertainty and rejected alternatives in [OFFLINE.md](OFFLINE.md), new timestamped scenarios in [PREDICTION.md](PREDICTION.md), and the B-first controller plan in [EXPERIMENTS.md](EXPERIMENTS.md). Original delivery files are retained in `directive2/prior_delivery/`; the original prediction bytes remain untouched at the start of PREDICTION.md. No test-A evidence, other explorer's R12 files, launches, or external writes were used.
