# 给负责人的结论

**本轮找到一个值得闭环验证的方向：发现长任务已经停滞后，让完整策略持续接管一小段，看到恢复迹象再交还缓存。还没有证明它能保持纯策略的成功率。** 它不需要知道哪个任务困难，也不使用任务编号分配预算。新组件只用初始状态 **0–19 训练、20–29 评估**；没有读取 30–49，没有查看其他研究者的第二轮目录，也没有启动或干扰正在运行的实验。

最有说服力的离线结果来自 **π0.5、50 条示范的长任务**：检测器在 100 个未参与训练的回合中，标出了全部 27 个失败回合，只误报了 1 个成功回合。第一次报警的中位时间是第 290 个控制步，还剩约 230 步。只看运行时间的对照也能找到这些失败，但晚约 60 步，并误报 11 个成功回合。说明观测和检索历史确实提供了额外信息。其他三个长任务配置的优势较弱，不能把这一结果推广成“已经准确识别了所有目标切换”。

建议比较四种固定响应：**调用一次、连续调用三次、持续调用直到出现恢复迹象或达到上限、随机位置连续调用三次**。每次调用仍完整执行十个控制步，每回合最多十二次完整策略调用。持续接管的理由是：在随机调用数据里，警报后仅调用一次没有显示出可靠的最终成功收益；缓存可能在策略刚把机器人带离卡住状态时又将它拉回去。这是待验证的机制假设，不是已证明的原因。

| 长任务配置 | 历史评估纯缓存成功率 | 连续三次调用的推理比例估算 | 持续接管的推理比例估算 | 新方法成功率 |
|---|---:|---:|---:|---|
| π0.5，50 条示范 | 73% | 0.102 | 0.117 | 未测 |
| π0.5，500 条示范 | 86% | 0.100 | 0.111 | 未测 |
| GR00T，50 条示范 | 62% | 0.100 | 0.124 | 未测 |
| GR00T，500 条示范 | 87% | 0.089 | 0.105 | 未测 |

这里的推理比例是假定轨迹长度和报警状态仍与原缓存轨迹相同的**成本估算**；接管后轨迹会变，实际值可能不同。作为收益敏感性分析，假如能救回四分之一被标出的失败，同时损害十分之一被误报的成功回合，四个配置的成功率分别约为 **79.7%、88.4%、69.3%、89.5%**。这些恢复概率没有被数据验证，所以这些数字**不是成功率预测**。特别是稀疏 GR00T 长任务，离纯策略仍有很大距离。

本轮还出现了必须正视的负结果。协调者正在运行的上一轮实验中，π0.5 稀疏长任务的严格独立评估已完成：**纯缓存 72%，共享运动修正 63%，运动加夹爪修正 63%，只改夹爪 74%，纯策略 92%**。运动修正曾把离线动作误差降低 37.5%，却没有带来成功率改善。样本还不足以精确确定损害大小，但足以否定“离线动作更接近策略就意味着闭环更好”的推断。本轮接管方案因此保留原缓存动作，没有叠加运动修正。

另外，直接把检索邻居推进到示范后面的动作，在四个配置中有三个使报警状态下的运动误差恶化；不建议部署。按夹爪模式筛选邻居也有不一致的结果，暂不占用闭环资源。

交付了可复用的进度标签、停滞检测、随机调用效果估计、成本计算和成功率敏感性工具，以及 **13 项单元测试、1,120 次真实示范库查询/尾段检查**。已冻结 **16 个候选实现、22 个含对照的实验配置**，全部只评估初始状态 20–29。优先验证 π0.5 稀疏长任务，其次用相同对照检验 GR00T；密集库配置排在后面。保持纯策略成功率的把握仍低，尚未通过两个百分点的非劣效检验。

## Technical interpretation

The new result is a useful observation-based **intervention selector**, not an identified causal transition detector. Its training target is a previously achieved new goal, at least 60 controls without another first-ever goal, and no new goal or success in the next 100 controls. Yet only 56–64% of first alerts match the current privileged later-stall label. The model also identifies failed regimes that do not satisfy that particular subgoal definition. High failure coverage must not be relabelled as high transition-boundary accuracy.

The serving feature vector has 57 scalar observation/action/retrieval-history summaries. It contains no task/init identity, language, goal predicate, object pose, teacher action, outcome, or absolute visual embedding. One shared model per policy/library cell is fitted across all ten tasks. The established task-conditioned retrieval remains unchanged. Four-fold initial-state cross-validation inside 0–19 selects model capacity and threshold; evaluation starts never calibrate a gate or budget.

The sustained-recovery hypothesis is deliberately different from a cheap action correction. It buys several consecutive policy decisions so that feedback can carry a transition through before cache retrieval resumes. It preserves ten-control commitment and uses a cap, hysteresis, and cooldown. The exact controller and its single-call, fixed-burst, and randomized placement controls are executable and frozen in [PROPOSALS.md](PROPOSALS.md).

Three cautions dominate the ranking:

- Alerts are late: median 270–300 controls. They may describe an already unrecoverable failure.
- Randomized first-alert one-call effects have intervals spanning benefit and harm. No offline estimator here identifies the sustained controller's SR.
- The newly completed correction experiment is a direct counterexample to using shadow action MSE as a success surrogate. Motion-only and joint losses versus cache are 7/16 and 11/20 paired wins/losses, respectively; exact two-sided p=.093 and .150. These are unfavorable estimates, not a precise claim of a nine-point population loss.

The complete evidence, including negative retrieval screens and the running-experiment snapshot, is in [DATA_ANALYSIS.md](DATA_ANALYSIS.md). Reproduction, frozen hashes, local dependency planning, and deferred launch commands are in [HANDBACK.md](HANDBACK.md). No candidate in this package establishes a new success-preserving frontier.
