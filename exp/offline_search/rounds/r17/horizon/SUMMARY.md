# R17 horizon / 纯减步最终摘要（2026-10-10）

**12个候选均未通过预注册的“两suite 2pp非劣＋实测成本”采用门。Fast-WAM C10最值得优先复核：两suite未校正界均过门，但尚未获family认证。**

Fast-WAM K1于19:05:52 CDT只读轮询确认PASS；DP/Fast完整重算快照UTC 2026-10-11T00:06:01.139090+00:00，11臂全部验收、pending=0。Cosmos沿用已封存配对分析。每臂两suite各500对，canonical task/init及初态hash匹配；SR控制分别为DP 91.2/79.6%、Fast 90.8/91.6%、Cosmos 96.4/98.6%。

下表所有双值依次为 **spatial / libero_10**。Δ为候选减默认，单位pp；“过/未”仅指该suite的SR门，两个suite均过才可联合通过。family与未校正都要求CP、bootstrap两种下界严格大于−2pp；未校正只作敏感性。
成本用真实模型调用M和实际NFE分别除以ACK确认的控制步T（不含settle）；↓为相对默认的减少百分比，负值表示增加，不是端到端时间加速。

| 模型/臂 | ΔSR，pp | family | 未校正 | 调用/T↓，% | NFE/T↓，% | 给owner的解读 |
|---|---:|---|---|---:|---:|---|
| DP/C10 | -1.4 / +4.2 | 未 / 未 | 未 / 过 | 48.93 / 49.63 | 48.93 / 49.63 | 长任务升、空间降；未获非劣认证 |
| DP/C15 | -4.8 / +4.4 | 未 / 未 | 未 / 过 | 65.26 / 66.12 | 65.26 / 66.12 | 空间掉4.8pp，不宜直接采用 |
| DP/C16 | -5.6 / +1.4 | 未 / 未 | 未 / 未 | 67.48 / 68.14 | 67.48 / 68.14 | 空间掉5.6pp，不宜直接采用 |
| DP/DDIM10 | -1.2 / -3.2 | 未 / 未 | 未 / 未 | 0.01 / 0.06 | 90.00 / 90.01 | 省90% NFE；长任务掉3.2pp |
| DP/DDIM5 | -0.6 / -4.2 | 未 / 未 | 未 / 未 | 0.05 / 0.03 | 95.00 / 95.00 | 省95% NFE；长任务掉4.2pp |
| Fast-WAM/C10 | +1.0 / +2.6 | 未 / 未 | 过 / 过 | 49.01 / 49.54 | 49.01 / 49.54 | 最有希望；未校正两suite均过 |
| Fast-WAM/C16 | +1.8 / -0.8 | 未 / 未 | 过 / 未 | 67.50 / 68.11 | 67.50 / 68.11 | 长任务近持平，非劣证据不足 |
| Fast-WAM/C20 | +1.2 / +0.0 | 未 / 未 | 过 / 未 | 73.50 / 74.35 | 73.50 / 74.35 | 长任务持平，非劣证据不足 |
| Fast-WAM/C32 | -0.8 / -6.2 | 未 / 未 | 未 / 未 | 82.85 / 83.62 | 82.85 / 83.62 | 长任务掉6.2pp，不宜直接采用 |
| Fast-WAM/K2 | +1.0 / -1.8 | 未 / 未 | 过 / 未 | -0.07 / 0.06 | 79.99 / 80.01 | 省80% NFE；长任务掉1.8pp |
| Fast-WAM/K1 | -0.4 / -3.0 | 未 / 未 | 未 / 未 | -0.03 / 0.02 | 90.00 / 90.00 | 省90% NFE；长任务掉3.0pp |
| Cosmos/K2含清理步 | +0.2 / -2.0 | 未 / 未 | 过 / 未 | 0.30 / 0.03 | 80.06 / 80.01 | 实际NFE1；长任务掉2.0pp，有节点混杂 |

默认与语义：DP为DDPM100/C5（chunk16）；DDIM10/5保持C5，使用同ε模型的trailing-DDIM、eta0，实际NFE100→10/5。Fast为K10/C5（chunk32），减步仍为原生shifted-flow，实际NFE10→2/1。horizon臂保持默认K。Cosmos为K5/C16（chunk16）；上游num_steps=2含clean预算后实际仅1次网络评估，不能记NFE2。

**采用建议：** 正式默认仍保留DP K100/C5、Fast K10/C5、Cosmos K5/C16。若另立验证，优先Fast C10（调用和NFE/控制步约减半、SR点差两suite均为正）；这不是改写本轮判决。DP C15/C16、Fast C32有明显观测掉点；DP DDIM10/5、Fast K1也有长任务损失，不能称为无损减步。Fast C16/C20/K2及Cosmos K2尚无两suite非劣证据。

**设计精度：** 24项family控制与每suite500对使接近持平但有不一致结局的臂较难跨过2pp门；这是精度与多重性的权衡，并非数学上不可能通过。未通过不等于证明损失超过2pp；未校正结果不用于事后更换主判据，也不能将明显点差下降仅归咎于检验保守。

**成本限制：** 调用/NFE节省已由逐步记录确认；所有臂的queue-free service/queue字段缺失，pricing不合格，host wall时间混有排队、batch组成和负载，服务成本门未认证。DP/Fast控制与臂分别匹配同GPU/batch配置；Cosmos同H100/B1，但K5渲染跨timan108/A5000与timan107/GTX1080，K2全部timan107；同timan107的274对仍下降2.19pp，其余l10的226对下降1.77pp，不能全归因于换节点。

| cache研究模型 | 最终L | 依据 |
|---|---:|---|
| DP | 10 | 无长C通过family门，按预注册fallback10；不是C10已认证 |
| Fast-WAM | 10 | 同上；未校正C10通过不参与L选择 |
| Cosmos | 16 | 默认已执行全chunk；长horizon组与对应默认组折叠 |

此处L是缓存研究的入组选择，不是纯推理horizon采用结论；按旧“首次失败即停”规则DP/Fast仍保留C5。未改STUDY、CACHE_STUDY或运行队列，未做GPU工作。

证据：[DP/Fast完整22格区间、成本与收据](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/RESULTS_DP_FASTWAM.md:1)、[可重算脚本](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/analyze_arms.py:1)、[DP/Fast JSON（164个输入SHA）](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/RESULTS_DP_FASTWAM.json:1)、[Cosmos K2（含早期NFE研究与混杂）](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/COSMOS_K2.md:1)、[预注册](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STUDY.md:125)、[cache选择规则](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/CACHE_STUDY.md:87)、[正式A集基线总表](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:34)。
