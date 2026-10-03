负责人说明：这轮只交诊断，不发新方案。GR00T 的主要问题更像是少数轨迹进入反复开合抓手、短暂恢复后又停住的状态，而不是保护条件满足后没有及时调用模型。也没有证据支持在抓取、搬运或释放阶段关闭现有纠正器。协调方正在测试的升级调用方案，本轮没有重复制作。

在允许查看的 0–29 上，GR00T 配方成功 250/300，纯策略成功 268/300；配方失败而纯策略成功的有 41 回合，反向获益有 23 回合。这个差距主要来自 0–19：只看用于评估的 20–29，配方是 88/100，纯策略是 89/100，损失／获益为 8／7，不能据此认定评估集上有稳定的六个百分点差距。π0.5 在同样的 0–29 上是 272/300 对 272/300；20–29 是 89/100 对 87/100。两边来自不同批次，差值还包含随机运行差异。

GR00T 的 41 个损失回合，首次保护调用中位数在第 10 个决策，每个决策是五个控制步；27 回合在首次闭合抓手之前已经调用。所有满足无进展条件的时点都执行了调用。每个损失回合都在纯策略完成对应任务的时间之前调用至少两次，因此“普遍第一次调用太晚”不是最有力的解释；日志仍无法确定物体何时真正抓错或掉落。

首次调用后的下一次观察，GR00T 与 π0.5 的进展恢复率在 20–29 都约 92%。算入所有重复调用后，分别为 63% 与 70%；逐回合等权后是 82% 与 83%。这说明反复陷入困难状态的少数回合拉低了每次调用的表面效率。GR00T 损失回合中，70% 的调用落在释放附近或释放后的张开阶段。π0.5 另有持续升级调用，会改变留下来接受普通保护调用的状态，不能将这些数字解释成两种模型的纯粹恢复能力差异。

现有纠正器在另外三种已记录轨迹的 20–29 上，将 GR00T 运动动作误差降低 41.2%，五类抓手指令阶段都改善。预先规定的四个停用条件均使误差上升 6.2%–45.5%，因此没有足够依据冻结观察触发的新臂。离线标签不能证明纠正器绝不伤害闭环；配方自己的日志缺少同状态策略标签及物体真值，这是剩余的关键不确定性。

## Evidence and decision

- [DATA_ANALYSIS.md](DATA_ANALYSIS.md) contains split-specific counts, task lists, timing/recovery definitions, phase/distance comparisons, and all 41 GR00T lost episodes.
- [Lost evaluation timelines](results/lost_eval_timelines.svg) show actual gripper commands and policy calls for all eight evaluation losses. [lost_episodes.csv](results/lost_episodes.csv) includes both models; [episode_events.json](results/episode_events.json) includes every admitted episode's event and call sequence.
- No missed no-progress verdicts: GR00T 2,647/2,647 eligible fresh decisions call policy. The unmodified corrector does not change the gripper; reconstructed fresh-cache gripper error is below 4e-7 for both models.
- GR00T's 0–29 net losses are concentrated in task 9 (microwave mug, −10), task 7 (soup + cheese basket, −7), task 0 (soup + sauce basket, −4), and task 6 (mug + pudding, −3). Task 8 gains four. Task IDs are descriptive reporting only; no task-indexed rule was fitted or added.
- A concrete counterexample to a distance-only explanation is task 3/init 23: 33 calls, mean distance .090, correction RMS .0229, and final retrieved progress .976, yet failure. Retrieved demo progress can look nearly complete while the environment remains unsolved.
- 18 safety/statistical tests pass. Numerical artifact comparison verifies 602 arrays and all shared serving state across the source and deployed recipe pickles. Their bytes differ because eight optional pace-wrist metadata/attribute fields are absent in each earlier deployed pickle; stack behavior and correction parameters agree.

## Scope

No arms frozen; no new run root, PREDICTION.md, emitter package, sync, launch, or process change. All new files are inside this round8 directory. Fit cuts use only 0–19; evaluation uses 20–29; no prohibited roots or 30–49 payloads are admitted. Existing per-task recipe heads are audited unchanged. This is a development diagnosis on the reused evaluation population, not a new independent confirmation. [HANDBACK.md](HANDBACK.md) gives reproduction and coordinator guidance.
