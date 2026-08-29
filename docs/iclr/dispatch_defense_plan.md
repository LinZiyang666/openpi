# Dispatch Surface 线的攻防思路（graded warm-start 深度调度）

> 背景：教授质疑"别只用 threshold"；owner 定免训练路线；`latex/dispatch_note.tex` 给出双单调共形调度曲面 $\tau^*(s,v)$ 的形式化。外部审查（2026-08-27）确认方向可行、指出首版数学七处错误（已在 note v2 修正）并给出攻防条件。本文档 = 该线的完整攻防定位。
> 一句话判决：**换掉 threshold 产生了一个可能守住的方法贡献；直接对照与显著收益才真正完成防守。**

---

## 1. 这条线守住 novelty 的三个条件

方法升级后，与 ActionCache 只剩"都复用历史 action"这一上层共同点：

> ActionCache 问"是否接受一次固定深度的缓存命中"；我们问"**在昂贵 transformer 之前，根据局部检索风险与时间状态，为当前决策分配多少后续计算**"。

这足以把 ActionCache 从"撞车论文"降为"最接近的固定策略 baseline"——但仅当三个条件全部满足：

**条件 1：surface 必须实测优于固定阈值，不能只画一张漂亮的曲面。**
同一库、teacher、payload、硬件、标定预算下，主结论必须是二选一：matched SR 下降低实测 GPU-time，或 matched compute 下提高 SR。对照臂全集：fixed threshold + fixed depth / two-threshold 三档 / **surface** / learned router（E10）/ reduced-NFE teacher / ActionCache-style CP2 baseline（四臂方案 Arm1）。

**条件 2：CP2 ActionCache-style 对照必须在链条里。**
只拿 surface 与自家 CP1 threshold 比，证明不了对 ActionCache 的优势。完整递进链：

$$\text{CP2-AC 式} \;\to\; \text{CP1 fixed threshold} \;\to\; \text{CP1 graded surface} \;\to\; \text{CP1 surface + stateful}$$

四段分别回答：early tap point 有没有价值 / graded dispatch 有没有价值 / temporal state 有没有额外价值。这条链与四臂对照方案（`actioncache_response_plan.md` §5）合并：surface 作为 Arm2 与 Arm3 之间的新臂（或替换 Arm2 的执行层），臂序变为 Arm1(CP2) → Arm2(CP1+threshold) → **Arm2.5(CP1+surface)** → Arm3(+stateful)。

**条件 3：不得声称"动态分配去噪算力"本身是新的。**
前艺术两篇必进 related work：
- **D3P**（arXiv 2508.06804）：RL adaptor 逐状态动态分配去噪步数，有真机实验——"per-state dynamic denoising budget"已被占，且是**训练型**方案；
- **DVAC / Denoising Tells When to Replan**（arXiv 2606.03847，竞品表已有）：按去噪方差动态调 execution horizon。

我们真正独有的三件组合：**① 起点是检索到的历史中间态**（不是从纯噪声减步）；**② 深度由 retrieval risk 免训练决定**（排序统计+共形，非 RL/学习）；**③ 决策发生在 transformer 之前**（early key，成本地板 14%）。D3P 恰好还给 E10 的"训练型对照"提供第二个参照系：learned dynamic dispatcher 是有人做过的路线，我们对照的是"免训练能到多近"。

## 2. 中稿判断（外部估计，采信）

- 数学修好 + surface 在仿真与真机上显著优于 ActionCache-style / fixed threshold / reduced NFE：novelty objection 基本守住，总录用概率约 **45–55%**；
- 真机只是展示、或 surface 与 threshold 差异很小：**25–35%**。
- ⇒ 优先级：surface 的闭环增益是这条线的生死，先小规模验证"surface 确实比双阈值好"再投入全面对照。

## 3. Note v2 的数学修正清单（已落实到 `latex/dispatch_note.tex`）

首版七处错误与修法（记录在案，防止回退）：

1. **对期望再取分位数（型错误）**：先定义随机损失 $Y_\tau = \|W(a^{(\tau)} - a^{(0)})\|$，条件分位数 $q_\tau(s,v) = Q_{1-\alpha}(Y_\tau \mid s,v)$，$\tau^*(s,v) = \max\{\tau : q_\tau(s,v) \le \delta\}$。
2. **均值单调 ⇏ 分位数单调**：单调性假设 (A1) 直接落在条件分位数上（干净版 = 条件分布的一阶随机序）。
3. **Lipschitz ≠ contraction**：$L$ 可大于 1；"剩余步吸收误差"降为经验动机句，不作理论推导。
4. **Karlin–Rubin 表述收窄**：改为"固定标量 score、两类 likelihood ratio 随 score 单调时，固定错误接受率约束下 threshold 是仅依赖该 score 的规则中 ROC-最优（Neyman–Pearson）"——限定约束与损失，不泛称"任何规则不可能改进"。
5. **逐 $\tau$ 独立拟合会交叉**：改为三维联合单调拟合 $\hat q(\tau, s, v)$（对 $\tau, v$ 非降、对 $s$ 非升），杜绝"深跳安全浅跳不安全"的洞。
6. **Conformal 保证重建（最重）**：per-$\tau$ marginal 覆盖不给自适应选择后的覆盖；step 级 exchangeability 被 episode 内相关性破坏；Mondrian 分组若由同一标定数据定义亦破坏保证。修法：**episode 级三分 split**（fit / calibration / test）→ fit 集拟合三维单调曲面 → calibration 集用 **max-over-$\tau$ nonconformity** $R_i = \max_\tau \{Y_{i,\tau} - \hat q_\tau(s_i, v_i)\}$ → $R$ 的 conformal 分位数给统一修正量 → 从同时有效的上界中选最大安全 $\tau$；**以 episode 为 conformal block**，或明示只提供 teacher-visitation 下的经验覆盖、不声称 step 级有限样本有效性。
7. **"calibration is not training" 措辞**：isotonic 分位回归就是单调函数类上的统计拟合；改称 gradient-free / nonparametric post-hoc calibration / no policy fine-tuning，不强辩"绝对不是 training"。

**第二轮回归（2026-08-27，note v2 → v3）**：五处确认修好；再修四硬问题 + 协议三条——
- **NP 事件改为离线可测代理**：$Z = \mathbb{1}\{Y_N \le \delta\}$（direct replay 偏差在容忍内）。旧表述 "replay preserves the outcome" 是闭环反事实，离线不可标注、episode 标签逐步归因错误——与我们自己 E5 的 caveat 矛盾。MLR 检查对代理事件在 fit split 上做；任务成功仍归闭环。
- **$\tau=0$ fallback 与有限样本条件**：$Y_0 \equiv 0$（耦合下两支重合），设 $\tilde q_0 \equiv 0$、$\tau^* = \max(\{0\} \cup \{\tau \ge 1: \tilde q_\tau \le \delta\})$——集合永不空，fail-closed 数学化；conformal 只作用 $\tau \ge 1$。order statistic 需 $|\mathcal{E}| \ge (1-\alpha)/\alpha$（$\alpha=0.05$ 至少 19 个标定 episode），否则 $c = +\infty$、规则退化为全 teacher（有效但空洞）。
- **Coupled noise 可执行化**：正确对照是同一 $z_j$（缓存轨迹初始噪声）在当前条件下全程生成——同分布 ≠ 同 realization，不同 $z$ 可落不同 action branch。数据生成要求：标定 artifact 存 initial noise tensor（~6.4KB/entry）或确定性 seed（须保证跨版本逐位重现）；**当前 `CachePayload` 无此字段**。廉价退路：先测 teacher–teacher 噪声方差（同 observation 重复采样），若远小于 warm–teacher 偏差则免耦合、目标量改述为对随机 teacher 样本的总体偏差。
- **isotonic 措辞**：quantile 版是 pinball loss 在格序单调类上的凸优化（LP/网络流可解），不是投影；不再声称 sorting-and-pooling 复杂度。
- **协议三条**：三 split 职责互斥（超参与 A1/MLR 诊断在 fit；calibration 只算一次 $c$ 不得回访；闭环在 test）；多任务混合下 coverage 是 mixture marginal，逐任务有效需 task-wise Mondrian，真机三任务分别报告；**$\tau \leftrightarrow$ `start_t` 约定钉死**：$\tau$ = 跳过步数，`start_t` $= (N-\tau)/N$。
- **残余最大风险已从数学转为实证**：episode-max 构造可能过保守，把大量状态压回 $\tau^*=0$——增益预检（§5）的判据正是"校准后的保守性是否仍留下对双阈值的余量"。

建模修正四条：
- $v$ 改名 **local action disagreement**（top-$k$ 来自邻近而非同一状态，混合了检索误差/阶段混叠/多峰性/teacher 随机性/库密度，不得称 aleatoric uncertainty）；动作各维量纲不同，用标准化或加权 Mahalanobis 范数 $W$。
- **Noise 耦合**：$a^{(0)}$ 与 $a^{(\tau)}$ 必须同 noise seed（否则偏差混入 teacher 采样差异）；warm-start 中间态须注明来自哪条 cached noise trajectory。与 X15 shadow-RNG 隔离、E0 计时契约同一纪律。
- **Deviation 是 proxy**：$\delta$ 不是成功率保证；须展示 deviation 与闭环失败风险的相关性，操作点最终由 closed-loop SR 选择与验证。Eq.(coverage) 不得读作 SR 保证。
- **surface ≥ threshold 仅在 oracle 函数类意义下成立**：函数类包含 threshold 故最优解不更差，但有限样本拟合完全可能过拟合输给 threshold——写为待检验假设，不写预期定理。

## 3b. 深度轴悬崖证据与三档坍缩裁决（2026-08-28）

**Pure warm start 实测数据**（`exp/warm_start/`，libero_spatial，**always-hit + 强制 warm start**，3 keybuilder × 3 档 × 500 ep）：

| 执行方式 | SR（三 keybuilder 范围） |
|---|---|
| 纯推理 B0 | 98.4–99.2% |
| warm start 跳 30%（t=0.7） | 96.4–98.0% |
| warm start 跳 50%（t=0.5） | 95.2–96.6% |
| warm start 跳 70%（t=0.3） | 94.0–94.6% |
| 纯回放（跳 100%，always-hit） | **67.4–69.6%** |

**三个判决**：

1. **深度轴不是平坦景观——悬崖在 τ∈(70%, 100%]**：跳 70% → 跳 100% 之间 SR 从 ~94% 崩到 ~68%，26pp 动态范围全在深度轴上。此前担心的"NFE=1 现象 ⇒ 深度选择无利可图"不成立（那是从自己噪声减步的平坦；这里是从检索中间态起跳，去噪修正错配的效应真实巨大——always-hit 连最差匹配都强制回放，留 30% 去噪步仍能把 68% 修回 94%）。
2. **Surface 的收益模型**：主增益 = 把 threshold 判 MISS 的状态转成深 warm start（每个转化省 S2 + 大部分 S3 ≈ 50ms），94% always-hit 底线说明安全垫厚；full replay 档独自承担整个悬崖风险。⚠ caveat：94% 是全体平均，MISS 子集（分数最低的难状态）表现会低于此，实际可捞空间打折——预检要测的正是这个。
3. **历史先验修正**：过去"聪明 judge 全没赢"的尝试都在接受/拒绝轴上堆信息，warm start 深度一直是全局固定档——26pp 动态范围的深度轴从未被逐状态化。历史负结果对该轴不外推。**Surface 预检胜率从 35–45% 上调至 ~50–60%。**

**三档坍缩裁决（owner 提议，采纳）**：问题坍缩为 **full hit（τ=N）/ 跳 70%（t=0.3）/ miss（τ=0）** 三档。量化损失可忽略、复杂度全线下降：

- **价值结构**：算力侧跳满 ≈14.4ms / 跳70% ≈23ms / MISS 72.5ms——**MISS→跳70% 省 ~50ms 是大头，跳70%→跳满只差 ~9ms**；SR 侧 30/50/70% 三档只差 ~2pp、悬崖在其后 ⇒ 连续 τ 相对三档的全部增益 = 中间态上抠几毫秒，不值三维曲面的复杂度。中间档钉在悬崖前最激进的安全位（跳 70%）= 数据指认的最优量化。
- **数学简化**：三维联合单调拟合消失，剩 $(s,v)$ 平面上两条校准边界（两个二元事件 $Y_N \le \delta$、$Y_{0.7N} \le \delta$）；conformal 从 N 档 episode-max 缩到两事件同时校准（Bonferroni $\alpha/2$ 或 max-over-2）——**外审担心的保守性问题随之大幅缓解**。
- **存储/工程**：只需 t=0.3 一档中间态——在线路径现存 0.7/0.5/0.3 已覆盖，**九档扩存缺口消失**；实现 = 现有双阈值三档栈 + WS 档 start_t 钉 0.3 + 判定加 $v$ 维，预检可近零新代码起跑（threshold-pareto 栈开 warm_tiers 扫 $(f_{FH}, f_{WS})$）。
- **写作形态**：理论讲连续曲面 $\tau^*(s,v)$（note v3 保留），实现节写"深度网格量化到三点，由 sweep 数据正当化——30–70% 区间近平坦、悬崖在其后，三档捕获全部价值"；量化决策数据驱动，比任意工程选择更有说服力；审稿人问"为何不连续"的答案就是 sweep 图，连续版留 appendix/future。

## 4. 工程可行性（外部核实 + 代码锚点）

已有接口覆盖大部分：payload 支持保存 denoising intermediates 且离线 artifact 已存全部九档（`src/openpi/cache/storage_types.py:68`）；judge 已能动态返回 `start_t`（`components/judge.py:284`）；interceptor 已能从任意合法 `start_t` 恢复 Stage 3（`interceptor.py:1098`）。

新增件：top-$k$ payload action disagreement 计算（检索副产品，factor 基建近似现成）/ surface artifact + 新 Judge / $\tau \leftrightarrow$ `start_t` 映射 / 标定数据生成管线（离线回放，逐 $\tau$ 档 + 耦合 seed）/ fetch 与 disagreement 的延迟计时。

**已知缺口**：① 标定 artifact 需新增 initial noise（或 seed）字段——`CachePayload` 现无（v3 耦合协议的前置）；② 在线路径默认只存 0.7/0.5/0.3 三档中间态、离线路径存九档（`storage_types.py:70`）——若部署允许全 $\tau$ 选择，须让在线库存全部候选深度（内存代价待估），或把 surface 的候选深度限制为现有三档（首版建议后者，先验证增益再扩档）。

## 5. 行动顺序

1. Note v3 数学定稿（本文档 §3 已落实两轮修正）→ 发教授。
2. **第 0 步：noise-sensitivity 消融（近零成本）**——同 observation 重复全程采样测 teacher–teacher 偏差；若可忽略则免除 noise 耦合与存储字段，标定管线大幅简化。
3. **增益预检（小规模，先于全面对照）**：离线标定数据生成 + 三档受限 surface vs 双阈值，1 suite × ~1k ep——surface 无增益则此线降级、省下全链投入（episode-max 保守性正是预检要回答的问题）。
4. 预检通过 → 并入四臂链（Arm2.5），与 `actioncache_response_plan.md` §5 的预算合并排期。
5. Related work 增补 D3P；E10 对照叙事更新（learned dynamic dispatcher 参照系）。
