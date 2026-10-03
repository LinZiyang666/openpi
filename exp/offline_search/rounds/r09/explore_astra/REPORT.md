# 降低推理成本：发现、建议与证据边界

## 给负责人的结论

这轮完成了 **70 组实验、21,000 个回合、952,842 次决策**的离线分析，全部只用初始状态 0–29。没有运行新的闭环实验，也没有使用 30–49 的保留集来选方案。目前**没有证据证明任何新方案通过了“成功率下降不超过 2 个百分点”的同时检验**。

最值得继续试的是：**把已经收集的影子策略动作变成更好的缓存监督，再用一个很小的模型修正缓存动作**。模型只读取现有视觉特征、机器人状态、缓存动作和已执行步数；保留夹爪命令，仍执行完整的 10 步动作段。使用一半修正量时，八种配置在未参与训练的初始状态上的运动误差降低 **12%–50%**；在另一种随机调用策略产生的轨迹上，四个稀疏库配置仍降低 **30%–35%**。预测头的本地 CPU 中位耗时约 **0.06 毫秒**，预计推理比例保持在纯缓存的 **0.074–0.078** 附近。

这里有两个必须说清楚的限制。第一，训练额外使用了发现集的影子标签，因此不能说是“只靠原来的 50 条示范”就获得提升。第二，动作误差不是成功率：本次找到直接反例——更频繁看图能降低影子动作误差，却让闭环成功率下降约 6–7 个百分点。**修正器的成功率现在无法可靠估计，必须实测**。置信度：局部动作改善较高，闭环收益仍较低。

还有一个较容易确认的方案：**在回合开始前，按任务选择全程纯策略或全程纯缓存**。对 π0.5 的长任务、50 条示范库，按初始状态交叉验证得到 **成功率 91.5%、推理比例 0.380**，同批纯策略为 **90.7%、0.504**。但重新拟合选择规则后的重采样显示明显不稳定，相对成功率变化的敏感性区间为 **−3.7 至 +1.8 个百分点**；不能据此宣布保持成功率。其他配置大多不支持大幅增加这种任务选择规则的复杂度。

最稳妥的已知配置调整仍是：**GR00T 的短任务、500 条示范库，关闭策略调用**。发现集纯缓存达到 **96.7% / 0.075**，纯策略为 **93.7% / 0.511**。这是对已有方向的支持，不是新突破；即使点估计高 3 个百分点，保守的同时检验仍未通过。

建议优先确认小型修正器和上述单个长任务配置的整回合选择。不要优先投入“只选最近一个示范”“只保留成功示范”或通用的高不确定性调用规则：这些方向本轮分别出现明确负面证据或没有稳定收益。现有检索距离还只用前 5 步动作训练，把监督改为实际提交的 10 步，在稀疏库上有 **4%–17%** 的动作误差改善，可作为不增加数据预算的低成本对照。

## Scope and reproducibility

All measured numbers below are discovery-only: ten fixed tasks × inits 0–29. Existing reports' full-500 results are context, not a selection dataset here. The extractor filters accepted journal attempts before admitting outcomes, verifies each episode's complete decision sequence, joins arrays by `decision_id`, and never inflates image members. Capture roots are read-only. Output locations, exact commands and tests are in [HANDBACK.md](HANDBACK.md); numbers and caveats are expanded in [DATA_ANALYSIS.md](DATA_ANALYSIS.md).

Definitions used in the technical body: A = ten-control pure cache; P10 = pure policy with a ten-control commitment; IP = independent .25 call coin; CU = uniform calls plus calibrated stall trigger. All errors use normalized **motion channels 0–5**; gripper sign error is separate. No padded action channels enter a distance. Metrics are equal-weight episode means, then equal-weight means over the ten tasks.

## Findings that change the next experiment

### 1. Relabel visited cache states with the policy, rather than only rearranging old neighbors

The fitted residual head uses 207 inputs: 128 PCA coordinates, 8 state values, 70 cached action coordinates and capped decision index. Per task, ridge regression uses linear features plus 384 fixed random Fourier features, ridge penalty 100, and equal training weight per episode. The target is the ten-step, six-channel motion residual. No success label enters training. Five folds keep every trajectory from the same init in one fold, including transfer arms. The proposed head applies half its residual and leaves the gripper unchanged.

| Model / suite / library | Pure cache SR @ IR | Half residual: motion MSE change on cache paths | On independent-call paths | On uniform-call paths | New SR estimate |
|---|---:|---:|---:|---:|---|
| π0.5 / long / 50 | .717 @ .0764 | −45.3% | −32.3% | −24.2% | Not identified |
| π0.5 / long / 500 | .857 @ .0766 | −18.5% | No such arm | −11.5% | Not identified |
| π0.5 / Spatial / 50 | .800 @ .0775 | −48.4% | −35.0% | −24.3% | Not identified |
| π0.5 / Spatial / 500 | .967 @ .0781 | −16.3% | No such arm | −14.3% | Not identified |
| GR00T / long / 50 | .637 @ .0743 | −50.3% | −34.7% | −24.6% | Not identified |
| GR00T / long / 500 | .827 @ .0745 | −12.8% | No such arm | −10.2% | Not identified |
| GR00T / Spatial / 50 | .880 @ .0753 | −37.2% | −30.3% | −20.6% | Not identified |
| GR00T / Spatial / 500 | .967 @ .0755 | −12.0% | No such arm | −8.8% | Not identified |

The expected owner IR is each row's pure-cache IR, subject to changed episode lengths and terminal rounding. This is an **equal-cost SR-improvement candidate**, not a demonstrated pure-policy-SR replacement. CPU overhead lies outside the owner's encoder/policy cost formula and must also be timed end to end.

Important controls:

- Gains survive the first 30 controls (6.7%–34.6% error reduction), moving-teacher states, and successful cache episodes (10.1%–33.3%). This is not solely fitting idle tails of failures.
- Zero actions and a per-task mean action are much worse. A fitted scalar shrinkage often **increases** error on IP/CU paths; the residual head does not.
- A nearest-neighbor memory of the **same extra shadow labels** achieves much of the sparse-library gain: a half blend reduces cache-path error 38.5%–52.6%, versus 37.2%–50.3% for the half residual head. The extra on-trajectory supervision is the main discovery; a special neural architecture is not established as necessary.
- On dense libraries the compact head transfers better than that simple extra memory. For π0.5 Spatial-500 on CU paths, shadow-memory blending increases error 4.5%; the half residual head reduces it 14.3%.
- Full corrections improve sparse cache-path errors more, but transfer less safely on dense paths. The single fixed half correction is the primary confirmation candidate; no per-cell blend search was used to select it.

The head learns from **52,741 fresh cache observations over 2,400 episodes**. Thus library-50 comparisons must disclose the extra 300 task-balanced rollouts per cell and the policy-shadow label budget. This is deliberately exploiting the owner's existing collection, not a library-only ablation.

### 2. Task-level routing is identifiable offline, but selection uncertainty is substantial

An episode lottery chooses a complete controller before acting. Every constituent episode outcome is available, so its expected SR and compute can be estimated without splicing trajectories. Optimize `sum(expected cost) / sum(expected decisions)`; averaging arm IRs would be incorrect. Paired task success differences are shrunk by twelve pseudo-episodes toward the cell mean; train for a one-point SR allowance. The optimizer is refit in five init folds.

| Cell | Binary router OOF SR @ IR | Matched P10 SR | Refit-bootstrap ΔSR interval, pp |
|---|---:|---:|---|
| π0.5 long-50 | .915 @ .380 | .907 | −3.7 to +1.8 |
| π0.5 long-500 | .876 @ .206 | .907 | See full sensitivity output; not retained |
| π0.5 Spatial-50 | .980 @ .445 | .990 | −3.1 to +0.1 |
| π0.5 Spatial-500 | .973 @ .163 | .990 | Not retained |
| GR00T long-50 | .891 @ .450 | .893 | Not an improvement over known frontier |
| GR00T long-500 | .875 @ .269 | .893 | −7.4 to −0.1 |
| GR00T Spatial-50 | .905 @ .234 | .937 | −5.9 to −0.2 |
| GR00T Spatial-500 | .967 @ .075 | .937 | −0.3 to +6.3; chooses cache everywhere |

The π0.5 long-50 frozen rule uses cache on tasks 2 and 3, policy on 0,1,4,5,6,7,9, and policy probability .149533 on task 8. Its full-data fitted replay is .914 @ .360; **.915 @ .380 is the OOF estimate of the learning procedure**, not the exact frozen policy's held-out value. These must not be interchanged.

Adding wrist/follow or CU choices often worsens out-of-fold SR. For example, π0.5 long-500's perception menu gives .849 @ .123 versus P10 .907. All 216 sensitivity configurations remain in `results/routing.json`; the report does not cherry-pick their best SR. Refit-bootstrap intervals are exploratory sensitivity intervals for this discrete learner, not a validated post-selection NI test.

### 3. Library-only changes are smaller; several intuitive ones are harmful

Refitting the same frozen PCA-space Mahalanobis metric using ten-step labels lowers shadow motion MSE by 17.4%, 11.8%, 8.9%, 4.4% on π0.5 long-50, π0.5 Spatial-50, GR00T long-50 and GR00T Spatial-50 respectively. Dense changes are all below 0.6%. This directly addresses the mismatch between five-step fitting and ten-step commitment without new observations. Sparse π0.5 Spatial improves on all ten tasks; much of the π0.5 long-50 improvement is concentrated in task 0.

At the same recorded states, top-1 synthesis increases error 27%–87%, motion-medoid synthesis 21%–34%, and filtering failed demonstrations increases error 5.6%–7.5% in dense libraries. Successful fragments of failed demonstrations remain useful. This does not contradict refusing to blindly extend through risky fragments: filtering an entire demonstration and gating a continuation are different interventions.

### 4. The call data do not support a universal uncertainty gate

All **26,687** discovery IP fresh anchors pass the actual `.25` probability and `coin < p` audit. We estimate centered first-entry excursions and local probability-shift derivatives with episode-level clustering. Moderators are pre-coin retrieval dispersion, relative nearest distance, state distance, elapsed time, kernel entropy, or a diagnostic policy-shadow disagreement.

Increasing call probability on the high-relative-distance half has an estimated local derivative of +.383 in π0.5 Spatial-50 but −.728 in π0.5 long-50. Shadow disagreement itself also changes sign. No generic high-risk score is supported across cells. These are derivatives under the IP future controller, not values of a deterministic trigger; dense libraries have no IP arm. Large action disagreement is not the same as positive policy-call value.

## Why the offline tool deliberately abstains from new SR predictions

Across 32 cache-only completed-arm contrasts, episode-balanced shadow motion error has Spearman correlation −.443 with SR change and gets the nonzero direction right in 22/32. The 58-contrast correlation is stronger (−.784), partly because policy calls trivially reduce policy disagreement. These contrasts share episodes and controls; nominal correlation p-values are descriptive.

A direct same-state falsification is stronger evidence of the limit:

| Sparse long-task cell | Blind refresh lowers shadow MSE | Actual every-5 versus every-10 SR change |
|---|---:|---:|
| π0.5 | −12.0% | −6.33 pp [−11.33, −1.33] |
| GR00T | −5.6% | −6.67 pp [−11.67, −1.67] |

Thus no conversion from a student MSE reduction to SR is justified. Shadows lack the next-state transition and subsequent recovery consequences, and represent a stochastic teacher that also fails. Small perturbations can change contacts, mode selection and future retrieval. New SR fields are explicitly missing rather than filled with optimistic numbers. Whole-episode routing is the exception because it selects observed complete controllers.

For GR00T Spatial-500, cache beats P10 by 18 wins / 9 losses on 300 pairs. The conservative paired CP lower bound is −2.024 pp nominal and −3.911 pp with an eight-cell family correction. Even this favorable point estimate fails the requested two-point bar. We do not substitute a narrower bootstrap interval to declare success.

## Hand-off

[PROPOSALS.md](PROPOSALS.md) ranks the proposals and specifies confirmation. There are **34 emitted discovery-only arms**, including eight residual candidates, four ten-step metric candidates, matched cache/policy controls, episode lotteries and two wrist controls. A four-arm π0.5 Spatial-50 dependency plan passes locally. Eleven contract/deployment tests pass, plus 80 real-library residual serving checks. No GPU, simulator, remote sync, server or worker was started.

The confirmation batch's wall time is uncertain and could exceed the brief's approximately 30-minute limit; the coordinator is asked to schedule it in [HANDBACK.md](HANDBACK.md). The unresolved result is closed-loop SR, not missing code or a claimed offline success certificate.
