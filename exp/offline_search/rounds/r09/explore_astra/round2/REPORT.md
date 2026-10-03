# 第二轮：用共享修正模型改善动作，继续压低视觉成本

## 给负责人的结论

本轮严格把初始状态 **0–19 用于训练、20–29 用于评估**，没有使用 30–49 的结果，也没有查看另一位研究者的第二轮产出。新方法不按任务分配调用次数、修正强度或示范库；同一配置的十个任务共用一个小模型。**目前仍没有新的闭环成功率证据，也没有通过成功率下降不超过两个百分点的同时检验。**

最值得先验证的是：**用包含策略调用的轨迹补充训练，并让小模型同时提出运动和夹爪修正。** 只在夹爪预测分数足够大、且与缓存命令相反时修改夹爪；动作段仍完整执行十步。两个稀疏长任务配置中，运动误差在未参与训练的纯缓存轨迹上降低 **37.5% 和 45.4%**，在包含调用的轨迹上降低 **21.9% 和 26.4%**。预计推理比例仍约 **0.076 和 0.074**。这些是动作误差改善，不能换算成成功率；闭环收益的信心仍然偏低。

夹爪确实提供了一点新信号，但不应夸大：两个稀疏长任务配置的夹爪分歧只减少约 **1.2 和 1.3 个百分点**，首次修改的中位时间已经到第 **250 和 220 个控制步**。这可能帮助部分卡住的回合，也可能已经太晚。因此准备了“只改运动”“只改夹爪”“同时修改”的固定对照，不能把任何一项的局部误差下降当成完整解决长任务的证据。

为什么短任务的旧修正器收益很大，长任务却没有可靠提升？这轮数据支持一个更谨慎的判断：**长任务的困难不只是动作偏差大小，还包括后续子目标能否完成。** 在失败的长回合中，约 **38%–78%** 曾新完成至少一个目标条件；只有约 **7%–11%** 结束时完成的条件数少于途中最好状态。单纯防止已完成目标被破坏，不足以覆盖主要失败。小模型对错误夹爪的修正又大多发生得较晚。这里分析的是原始纯缓存轨迹，不能据此断言某个修正后回合失败的具体原因。

另一个直接降低成本的方向是：**只看腕部相机，再用小模型修正取出的动作。** 新模型在线完全不读取另一台相机的特征。在四个 π0.5 配置的未参与训练的腕部相机轨迹上，运动误差降低 **15%–43%**；预计推理比例为 **0.034–0.037**，比普通纯缓存再省约 **0.041–0.043**。建议先验证短任务、500 条示范库。这个方向的降成本把握较高，保持成功率的把握仍低；未经修正的腕部相机方案在长任务评估样本上已经损失 10–13 个百分点，不能因为离线误差变小就宣称补回来了。

学习“何时调用策略”的尝试没有得到可推广的证据：独立评估的调用收益区间很宽，不同配置甚至方向相反。本轮不推荐上线这类规则。也没有继续研究按任务分流。

已完成可复用离线工具、**11 项单元测试、560 次真实示范库查询检查**，并冻结 **28 个候选实现和 44 个含对照的实验配置**。小模型本地预测约 **0.084 毫秒**；尚未测远端端到端延迟。没有启动新闭环实验或占用显卡。准确的实验配置、仅含 20–29 的清单及启动命令已交付；批次时长不确定，按任务要求交协调者安排。**所有新方法的成功率仍待实测，不填未经支持的预测数字。**

## Technical findings

### A task-agnostic replacement is feasible, but its success rate is unmeasured

The new head pools all tasks within a cell and consumes observation/action/history features only. It preserves the existing retrieval and ten-control commitment. Mixed-path supervision is the main useful change: on sparse long tasks, transfer-path MSE improves materially compared with training only on pure-cache states. Adding one prior fresh observation helps more modestly. All evaluations share the same init split across source controllers, preventing a matching start from entering training through another arm.

The gripper extension is intentionally conservative and separately testable. It improves teacher-sign agreement on sparse cache paths and less on mixed-call paths. It barely fires on dense libraries. A large count of corrected control bits can come from a few repeated stalled states, so the tools also report affected episodes, first-change precision, and intervention timing. The .8 threshold is a regression-score threshold, not a probability guarantee.

No new switch uses task identity. The frozen base still uses the established task-conditioned retrieval metric; replacing that essential existing component was not part of this proposal. Per-cell head fitting reflects different policy/action/PCA spaces, not a per-task hardness rule.

### Wrist-only distillation is a separate compute-saving candidate

The wrist head is trained using the observations and retrieved actions that its cheap perception path actually sees, including real wrist-only trajectories. Its 143 inputs exclude the third-camera embedding. At wrist-path evaluation states it has lower motion error than both uncorrected wrist retrieval and fresh full-camera cache retrieval. This offers an actionable cost-saving hypothesis at roughly half the current cache IR.

It does not identify counterfactual SR. Full-camera retrieval on an already failed wrist trajectory is an especially weak success benchmark. The emitted primary wrist candidate keeps the gripper and ten-control commitment unchanged and starts on the short dense cell where the factual wrist baseline is least concerning.

### Call value must be distinguished from imitation error and failure risk

The randomized-call tool estimates first-entry excursions and local probability derivatives from the logged .25 coin. It trains moderators on 0–19 and evaluates them on 20–29. It does not use policy-shadow disagreement as a causal reward. First-entry learned-value intervals cross zero in all four sparse cells. A positive local derivative in GR00T long-50 does not generalize to Spatial and cannot be extrapolated to a deterministic gate.

This negative result is useful: neither a teacher disagreement score nor an observation-based failure score by itself certifies that spending on the policy will help. No call-policy deployment arm is promoted.

### What the physical audit does and does not explain

The audit distinguishes initially true goals, new progress, transient losses, and terminal regression, using explicitly addressed evaluation episodes only. It rejects a simple “most long failures destroy a completed goal” explanation. It is consistent with incomplete later subtasks and missed mode transitions, but does not demonstrate their causal mechanism. It never labels a future recovery as available from a given policy shadow.

The owner's old corrector results remain context: they reused training starts in closed loop. Their large Spatial gains justify investigating corrections, but are not accepted validation of any new disjoint-fit head. The new tools do not compare their split-specific MSE against the old five-fold numbers as a matched experiment.

## Recommended confirmation

1. Fixed five-arm motion/gripper factorial plus cache/P10 controls on π0.5 long-50, followed by the identical GR00T long-50 batch, **inits 20–29 only**.
2. Four-arm full-cache / wrist-cache / corrected-wrist / P10 comparison on π0.5 Spatial-500, same evaluation starts.
3. Expand to the remaining frozen cells only after recording the complete fixed comparisons, including negative results. No adaptive threshold search on these evaluation outcomes.

The candidate table with per-cell cost expectations, empirical SR references, failure modes, exact names, and final NI protocol is in [PROPOSALS.md](PROPOSALS.md). Full measurements and commands are in [DATA_ANALYSIS.md](DATA_ANALYSIS.md). Deployment and reproduction are in [HANDBACK.md](HANDBACK.md).

The scientific deliverable is a stricter offline workflow and several executable hypotheses, **not a newly established inference/success frontier**. Candidate SR remains unidentified until closed-loop confirmation. The coordinator owns the locked sample and the final simultaneous two-point claim.
