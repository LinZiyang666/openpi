# R9 第二轮报告（fable）

## 一、给 owner 的结论（白话）

**规则执行**：这一轮所有分析只用初始状态 0–29；拟合用 0–19，评估（离线和闭环）只用 20–29，两者不重叠；没有打开任何 30–49 的文件；之前听到的保留集数字一律作废、不参与任何选择。没有任何"按任务分治"的东西：方法在决策时不知道当前任务难不难。

**问题**：修正器（给缓存动作加一半力度的运动修正）在短任务 Spatial 上涨 6–11 个百分点，在长任务 LIBERO-10 上几乎没用（π0.5 .743→.757，GR00T .583→.637），而纯大模型在 LIBERO-10 上领先 17–25 个点。为什么，以及怎么办。

**为什么（三条证据，均来自 0–29 的数据）**
1. 修正量在"走对"的时候很小（约 0.07–0.10 个动作标准差），一旦这集开始失败就涨到 2.5 倍，并且随检索距离增大——修正器主要在"往回拉已经偏离示教的状态"，半力度拉不回长任务。
2. Spatial 的失败有一个独特的物理特征：失败集的夹爪开合次数是成功集的 5 倍（每 100 步 6.3 对 1.3）——缓存在碗沿边反复合上、张开。修正器把这类集改好了。LIBERO-10 的失败没有这种特征（3.0 对 1.5），而且发生在更长的时程里。
3. 把失败时刻（取证标注的决定性事件）单独拿出来看：在 LIBERO-10 失败时刻，缓存的夹爪指令和大模型是一致的（分歧只有 .09，比平均还低），失败是位置没对准、不是夹爪时机；修正器在这些时刻也确实把动作拉近了大模型 21%，但一次 10 步、0.3 个标准差量级的修正改变不了"已经抓空"的结果。LIBERO-10 一半以上的失败是"合上夹爪但没夹到东西"（π0.5 53%，GR00T 58%）。

**这一轮找到的新东西（不分任务）**
1. **"抓空"检测器**：只看机器人自己的信号——合爪指令已持续 10 步、手指开度低于一个阈值（手指之间没有东西）。在没见过的初始状态上：π0.5 LIBERO-10 的 12 个抓空失败全部命中，Spatial 11/11，GR00T 23/25；所有失败的 70–90% 都被标出；成功集里误报 0–6%；比取证标注的失败时刻早 30–70 步。阈值用 0–19 的数据按"抓着东西时的开度的 1% 分位的一半"定，不看任务号。
2. **用它触发恢复**：标出抓空后，接下来两次决策调大模型（20 步），每集最多两次，其余时间纯缓存。推理占比只比纯缓存高约 0.005–0.02。
3. **修正器改成全力度、且在不重叠的初始状态上拟合**：离线上全力度在没见过的初始状态上把差距缩 31–41%（半力度 22–28%），且确认它跨初始状态泛化（不是靠记住训练时的场景）。
4. 已否定：改夹爪通道（失败时刻夹爪本来就对）；按任务做任何事（规则二）。

**闭环筛选（100 个没见过的初始状态 20–29，同拓扑配对）**：修正器**全力度**（在 0–19 拟合、20–29 评估）在 π0.5 LIBERO-10 五示教库上**明显有害**：按任务拟合的头 .63 对纯缓存 .74（−11 点，区间 −23 到 0），不分任务的单头 .54（−20 点）；纯大模型同组 .93。离线上全力度最接近大模型，闭环却最差——这是本轮第三次证明"离线更像大模型"不等于闭环更成功（扩库、全力度修正）。半力度（第一轮 +1.4 点）已接近最佳；修正缓存动作这条路在长任务上到头了。"抓空检测→调大模型恢复"的 5 条臂和第二批 6 条臂（半力度底座 + 检测器、无进展守卫 + 半力度、检测器 + 均匀抛硬币）在交接时仍在跑/已备好，结果用 `paired_r2` 工具读取（见 HANDBACK.md）。

**建议**：1. 不再加大修正力度；保留半力度作为底座。2. 以"抓空检测→恢复调用"为主候选：等 `r9f2_*_gm_corr0`（纯缓存 + 检测器）结果；若 LIBERO-10 上比纯缓存高 ≥ 5 点、占比 ≤ +.03，再扩到 300 集（检测器无拟合成分，可直接跑 0–29 全部）。3. 第二批臂（半力度 + 检测器、无进展守卫 + 半力度）在 r09_fable_r2 已 prefit/emit，待第一条链结束后同步并启动。

---

## 二、技术正文

See `DATA_ANALYSIS.md` for every table and command, `PROPOSALS.md` for frozen specs and exact arms, `HANDBACK.md`
for files and reproduction.

### 1. Why the half-strength corrector does not transfer to LIBERO-10
- Correction magnitude is state-dependent: ≈ .07–.10σ on-track, .17–.23σ once failing, corr(magnitude, d1_rel)
  .44–.61 (astra's arms, inits 0–29). The head mostly pulls off-manifold states back; half strength does not pull a
  failing long-horizon episode back.
- Spatial failures carry a gripper-chattering signature (6.3 toggles/100 controls in failures vs 1.3 in successes)
  that the corrector removes; LIBERO-10 failures do not (3.0 vs 1.5).
- At forensic onsets the cache's gripper command agrees with the policy (disagreement .04–.09 vs .10–.17 on
  average): failures are positioning errors. On LIBERO-10 the corrector changes the onset action by −21% in gap
  and the contact outcome does not change; 53–58% of LIBERO-10 cache failures are empty grasps.
- Per-phase held-out gaps (heads fitted on 0–19, evaluated on 20–29): the approach phase moves least on LIBERO-10
  (−11%), carry/post most (−40%); full strength beats half in every phase (−31/−36/−41% vs −22/−24/−28% overall).

### 2. Task-agnostic empty-grasp detector and recovery call
Executed close command held ≥ 10 controls and raw finger aperture < ½·p1(holding aperture on inits 0–19)
(.0010 π0.5 / .0009 GR00T): 12/12, 11/11, 23/25 grasp-miss failures flagged on inits 20–29; 22/27, 19/21, 28/38 of
all failures; 1/73, 0/79, 4/62 successes; 30–70 controls before the forensic onset. `GraspMissCalls`: on a flag, a
2-call policy burst (20 controls), ≤ 2 triggers per episode, otherwise pure cache (optionally corrected).

### 3. Closed-loop screening (inits 20–29, 100 pairs)
**Status at hand-back (2026-10-02 00:0x CDT): chain `r9f_r2` (tmux, self-terminating) has finished 2 of 11 arms; the remaining 9 (grasp-miss trigger arms, GR00T L10-50, Spatial-50) complete over the next ~30 min. Read them with `paired_r2` (HANDBACK.md).**

| arm (π0.5 L10-50, inits 20–29, 100 pairs) | SR @ IR | paired vs cache .740 (same pairs/topology) |
|---|---|---|
| pure cache (round-1 control) | .740 @ .0765 | — |
| pure policy (round-1 control) | .930 @ .504 | +19 pp [+10, +27] |
| corrector **full strength**, per-task head fitted on 0–19 | **.630** @ .0762 | **−11 pp** [−23, 0], +14/−25, p = .11 |
| corrector full strength, single task-agnostic head | **.540** @ .076 | **−20 pp** |
| grasp-miss trigger (+ full corrector / + plain cache), GR00T, Spatial | pending | pending |

**Reading.** The full-strength corrector — the offline optimum on held-out inits (−31% gap) — is clearly harmful
closed loop on LIBERO-10 (−11 and −20 pp), while the half strength of round 1 was +1.4 pp. Offline distance to the
policy mispredicts closed loop for the third time this round (grown library, full corrector). The per-task pattern
(task 2: 1.0 → .6, task 3: 1.0 → .8, task 0: .4 → .6) repeats the grown-library pattern: a stronger pull toward the
policy helps the tasks the demos handle badly and breaks the tasks they handle well. Consequence: correction strength
must be shrunk (≤ .5) and the remaining LIBERO-10 gap cannot be closed by correcting the cached action — the
grasp-miss recovery call (observation-keyed) is the live candidate; its arms are the pending ones, plus batch 2
(half-strength base + trigger, no-progress + half corrector, trigger + uniform coin) already prefitted and emitted in
the same run root (`arms_in_batch2.json`, `arms_in_batch2b.json`).

### 4. Limits
100 pairs resolve ±10 pp per contrast; the heads were fitted on test-task trajectories at inits 0–19 (deployable
version: B-pool rollouts); the aperture threshold is calibrated with simulator truth stages here (deployable: the
demonstrations' holding aperture); recovery relies on the policy re-grasping within 20 controls.
