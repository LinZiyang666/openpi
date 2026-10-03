# R9 自由探索报告（fable）

## 一、给 owner 的结论（白话）

**问题**：在 LIBERO 仿真里让机器人少调大模型（降低推理占比），同时成功率不掉。上一轮大数据里最值钱的是"影子标注"：机器人每做一次决策，事后都算出了大模型在同一画面下会给的动作。我用它做了一套离线工具，把 70 条实验、3.5 万集、160 万个决策翻了一遍，并在 h100 上跑了一次 4 臂的真闭环确认。结论：一条可靠的正结果，一条值得追的假设，两条明确的负结果。

**正结果：该不该调大模型，按"任务"分配预算，比按"时刻"放调用管用得多（置信度高）。**
每个库里都有 3–5 个任务扛起了几乎全部的"缓存不如大模型"的差距，另外 4–5 个任务缓存已经和大模型一样好，往那里花钱是白花。我用真实闭环结果做配对混合（同一组初始状态、同一随机种子，每集选哪条臂的结果只看任务号），并做了交叉验证（给任务排难度的数据和评估用的数据不重叠）：
- π0.5 LIBERO-10 五示教库：只给最难的 3 个任务用均匀调用（占比 0.3），其余纯缓存 → 成功率 .833、占比 .139，比同占比的均匀调用高 **+7.4 个百分点**（95% 区间 +3.8 到 +10.9）。
- GR00T LIBERO-10 五示教库：同样做法 .753@.147，高 **+6.4 点**（+1.8 到 +11.0）；最难 3 个任务直接用纯大模型 .810@.200，高 +8.1 点。
- π0.5 Spatial 五示教库：最难 2 个任务用调用 .887@.124，高 **+4.0 点**（+1.4 到 +6.6）；换个说法，达到 .943 成功率只要占比 .176，均匀调用要 .304（省 42%）。
- 五百示教库上 +2.5 到 +2.9 点；Spatial 五百示教库本来就接近大模型，没东西可省。
怎么知道哪个任务难、又不偷看测试结果？每个任务跑 10 集纯缓存、事后让大模型做影子标注、算平均动作差距（或夹爪分歧、或检索距离），排出的顺序和用 30 集排的一致（相关 .88–.97），不用任何成功/失败标签。四种排法在稀疏库上结论一致。

**值得追的假设：缓存库用"大模型事后标注的缓存轨迹"扩充，但只在难任务上用（置信度中低）。**
纯缓存给出的动作和大模型在同一画面下的动作差得很远（是大模型自身随机噪声的 2–10 倍），而且换任何平均方式都缩不了（最多 3%）。把"缓存自己跑出来的轨迹 + 大模型事后标签"加进库里，离线差距缩 31–45%，在线成本一分不加（占比仍约 .076）。但真闭环（π0.5 LIBERO-10 五示教库，100 个留出初始状态，同拓扑对照）**没有提升**：纯缓存 .74，扩库 .70 和 .73（差 −1 点，区间 ±10）。拆开看任务：难任务大涨（任务 0：.4→.8，任务 4：.4→.8），原本 5 条示教就够的任务反而大跌（任务 2、3：1.0→.5/.6，那两个任务上大模型本身就比缓存差）。事后按"检索距离排出的最难 3 个任务用扩充库、其余用原库"混合：.83 对 .74（+9 点，区间 +2 到 +16），占比不变。这是在 100 集上事后挑出来的，只能算假设，要按预注册规则再测。

**负结果 1**：想在一集刚开始时就判断"这集会不会失败"（比如开头调一次大模型看分歧）——不行。开头 10–50 步的分歧对最终成败几乎没有预测力（AUROC 约 .5）。这也解释了前几轮"按预测分歧放调用等于随机放"。
**负结果 2**：改合成规则（怎么把 16 条邻居平均成一个动作）没有空间；"离线更接近大模型"也不等于闭环更成功——扩库离线缩差距 38%，闭环 0 点，这是本轮对离线筛选工具最重要的一条校准。

**建议的下一步（按优先级）**：
1. 每任务 10 集影子标注校准 → 每任务调用预算，在三个稀疏库上和均匀调用做同拓扑配对（12 臂，约 2 小时）；达标线：同占比高 ≥ 4 点，或同成功率省 ≥ 40% 占比；再上留出集 30–49。
2. 预注册"难任务用扩充库、其余用原库"的规则，在 π0.5 LIBERO-10、GR00T LIBERO-10、π0.5 Spatial 五示教库上各跑 A / 混合 / 纯大模型（6 臂，约 40 分钟）；达标线：配对高于纯缓存 ≥ 5 点、占比不变。可部署版要用非测试初始状态重新采集扩库。
3. 两者叠加（共用同一次校准）。小网络代替检索暂不推进（前提和扩库一样是离线指标）。

---

## 二、技术正文

Names: A = pure cache (Commit-Cache, IR ≈ .076); CU = uniform random calls at ρ with calibrated stall trigger;
IP = independent call coin p .25; P10 = pure policy every 10 controls (IR .50). Shadow = deferred policy chunk
at the controller's own observation. Gap = RMS over executed 5 controls × 6 motion dims of served − shadow, in
library-σ units. Everything selects or fits on discovery inits 0–29 only. Details, commands and tables:
`DATA_ANALYSIS.md`; proposals with confirmation plans: `PROPOSALS.md`; files and runs: `HANDBACK.md`.

### 1. What I built (tools/, 14 unit tests)

1. **Extraction** of the 70-arm R8 collection into per-episode / per-decision tables and decision-aligned arrays
   (served block, policy shadow, PCA keys, state, 16 retrieval rows/weights), 7 GB, ~3 min with 12 workers.
2. **Dose-assignment simulator** (`paired.py`): paired mixtures of real closed-loop outcomes across arms,
   per-task Lagrangian hulls, K-fold cross-validation, IR-matched uniform reference with paired bootstrap.
3. **Shadow-gap profiler** (`shadow_gap.py`): per-decision/episode/task gaps, noise floor from independent
   policy draws, AUROC of early features for failure.
4. **Per-task calibration** (`task_alloc.py`): label-free task scores, correlation with the P10−A gap, fold and
   calibration-size stability, pre-declared allocation rules.
5. **Synthesis evaluator** (`synth_eval.py`): exact reconstruction of the deployed kernel from the frozen fit;
   alternative rules scored against the shadow.
6. **Student / kNN on shadow labels** (`student.py`, GPU): DAgger-style distillation, learning curves.
7. **Library grower** (`grow_library.py`): extends a frozen A fit with shadow-labelled rows as a registered
   library, deployable without new serving code; offline emulation check.
8. **Confirmation analysis** (`confirm_grown.py`).

### 2. Findings

**2.1 Per-task structure dominates.** Per-task A vs P10 success (inits 0–29): 3–5 tasks per sparse cell carry the
whole gap (π0.5 L10-50 task 0: .33 vs .97; GR00T L10-50 tasks 0/9: .23 vs .77/.93; π0.5 Sp-50 task 6: .37 vs 1.0)
while 4–5 tasks have A ≥ P10. The same LIBERO-10 tasks are hard for both policies.

**2.2 Per-task allocation beats uniform at matched cost (real outcomes, CV).** Rule "rank tasks by a label-free
shadow score fitted on the other folds; top-k get CU, rest A": π0.5 L10-50 top-3 .833 @ .139, +7.4 pp [+3.8,
+10.9] vs the IR-matched uniform mixture; GR00T L10-50 top-3 .753 @ .147, +6.4 [+1.8, +11.0] (top-3 P10 .810 @
.200, +8.1 [+4.6, +11.6]); π0.5 Sp-50 top-2 .887 @ .124, +4.0 [+1.4, +6.6]; π0.5 L10-500 top-5 .913 @ .131, +2.5
[+0.7, +4.3]; GR00T L10-500 top-2 .857 @ .108, +2.9 [+0.2, +5.3]; Spatial-500 ≈ 0. Four signals agree on the
sparse cells; the shadow gap is the most stable (fold-to-fold Spearman .85–.94; 10 episodes/task reproduce the
30-init ranking at .88–.97).

**2.3 Early disagreement does not predict failure.** AUROC of the gap over the first 1–10 decisions for episode
failure is .40–.65 in every cell; step-0 d1 .54–.75; the whole-episode gap is post hoc (.69–1.0).

**2.4 The cache is far from the policy and re-weighting cannot fix it.** Look-time gap .45/.48 (π0.5 sparse),
.52/.47 (GR00T sparse) vs policy noise floors .19 and .05; gripper disagreement .13–.16 vs ≤ .06. Top-1,
uniform-k, kernel widths 2–12 and local-linear corrections move the gap by −3% to +25%.

**2.5 Shadow-labelled rows close the offline gap; data, not architecture.** Train inits 0–19 (all arms), val =
pure-cache arm at inits 20–29: a 3×512 MLP on (PCA keys, state, task) reaches .238 vs .412 served (π0.5 L10-50),
.282 vs .525 (GR00T L10-50), .283 vs .518 (π0.5 Sp-50); kernel kNN on the same labelled rows .257/.298/.280, and
.301/.346/.319 with only the 200 pure-cache episodes. With the deployed metric and frozen fit: .412 → .285 (A
rows) / .254 (all rows); .525 → .336 / .293; .518 → .326 / .289. 500-demo cells gain ≤ 11%.

**2.6 Closed loop (π0.5 L10-50, 100 held-out pairs, h100 + timan107, standard mode): the uniform grown library
does not change success.** A .740 @ .0765; grown (6,763 A-arm rows) .700 @ .0763, −4 pp [−15, +6], 16/20, p .62;
grown (73,945 rows, all arms) .730 @ .0764, −1 pp [−11, +9], 17/18, p 1.0; P10 .930 @ .504 (+19 pp vs A [+10, +27], 24/5, p .0005). Per task the grown
library gains +30–40 pp on tasks 0, 4, 9 and loses 40–50 pp on tasks 2, 3 (where P10 < A). Post hoc, "grown rows on
the top-3 tasks by mean retrieval distance (ranking from inits 0–19), original library elsewhere" gives .83 vs
.74, +9 pp [+2, +16] at identical IR (18 variants examined; hypothesis only).

### 3. Limits and honesty notes

- Dose-simulator estimates are paired real outcomes but single-run (±3 pp per arm); the deployed per-task
  controller realizes doses inside one lottery controller rather than by switching arms.
- Shadow-gap numbers are screening evidence: a −38% held-out gap gave 0 ± 10 pp closed loop. They are used
  here only to rank tasks, where they agree with closed-loop outcomes.
- The grown artifacts use test-task trajectories from inits 0–19 (held out from the evaluated inits 20–29);
  the deployable version must be regrown from B-pool rollouts. Nothing used inits 30–49.
- The student was not run closed-loop; its premise is the same offline metric that failed to predict 2.6.
