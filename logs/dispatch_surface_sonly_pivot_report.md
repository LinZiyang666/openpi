# S-only 风险索引转向报告（owner × Execution 讨论落稿，2026-08-30 晚）

> 性质：讨论纪要 + 转向提案草案，供 owner 复核后交 codex 立新 G1。全部数字均为 post-hoc 探索口径（A′ 开发集），不进冻结裁决链；冻结裁决 `stop_before_C` 及其全部记录不受影响。本文件未暂存（owner 指示：无指示不入暂存区）。

## 0. 一句话结论

放弃"(s,v) surface / CRD / H-CRD 超越 threshold"的定位。新定位：**s-only 校准分位切 = score-threshold 家族帕累托前沿的风险索引**——线上就是 threshold 策略（两次数值比较，零新机制），线下用一次小样本校准把 K 个切点耦合在一个风险水平 δ 上，**自动落在前沿附近，而不是靠网格搜索找前沿**。创新点定位在"自由度增多时组合的自动化"（K=1 情形与分位数倒推等价，主动让步）。

## 1. 讨论时间线与各步证据

### 1.1 H-CRD batch D（4 点，1200 ep）
q∈{0.60,0.75,0.85,0.925} × (γ=1, β=2δ, J=3, L=6, dwell=2)：41.71 ms/0.740、50.76/0.790、58.55/0.840、65.35/0.863。对本线三条旧 hull 同成本全胜（+2.2 … +19.1 pt vs threshold）；q0.60 首次超过 always-full（0.863 > 0.847）。golden 后验 75,856 行 0 违例。（交审文档 §16.2–16.3）

### 1.2 owner 提出 GTP 对照，胜利证据不成立
`exp/gate_threshold_pareto`（threshold 二值 + N4 gate，pruned 500，ws/cs 库 2640/2741 条目）同 IR 插值：IR 0.618 处 **−6.3 pt**、0.752 处 **−4.6**、0.867 处 −0.6、0.968 处 +1.7（GTP 右端外推）。口径差异（池不相交、库同级、动作空间不同、成本轴同表）详见交审文档 §16.5。判定：batch D 不构成对"dense threshold + production hysteresis"（codex §12.6 臂 1，必须打赢的强 baseline）的胜利证据。

### 1.3 参数诊断：不是没调好，是没油水
- 行为分解（§16.6）：RECOVERY-MISS 占四臂总成本 **51.6%**（q0.60 达 80%，中位停留 20 决策、最长 101 ≈ 整集）；仅因 dwell 被挡的可准入决策 711/75,856；fuse 0 次触发。
- 离线预演器 `analysis/crd_offline_screen.py`（dev 表回放状态机，校准误差 <0.5 ms）：reopen×{1.0–1.15} × J{3,5,8} × γ{1,0.9} × dwell{1,2} 全扫，**全部旋钮余地 ~1–3% 成本**，补不上 4.6–6.3 pt。绑定约束是 u 面在覆盖外段的判定（信号层），不是滞回参数。（§16.7）

### 1.4 机制结论：这类控制器只会"减命中"
状态层面 H-CRD 允许的缓存执行 ⊆ 同 δ 静态 surface 的（回入 u≤δ_reopen≤δ；debt/L 只做减法）。同 δ 对照：静态 `dsp_sv_p85` 45.78 ms/0.633 → H-CRD q0.85 50.76/0.790（+15.7 pt，+7.4% 成本）——滞回思想在我们侧同样有效；但 GTP 的 gate 把 threshold 侧抬了同量级（0.797→0.846）。两套"信号+时序状态机"栈爬同一条'把命中换教师步'的山脊，山脊高度由信号决定（决策序列决定论 931/931；H1≈0 ⇒ (s,v) 与 s 信息等价）。我们的 RECOVERY 还是退化形式：触发/回入用同一个 u 序列、带宽为零，抓不住"u 说行但其实错"。

### 1.5 per-task 不是护城河
task 条件化交叉拟合：threshold 0.621→0.739（+11.8），surface 0.636→0.742（+10.6）——**同涨后平手**。per-task 是任何家族都能用的条件变量；surface 唯一的量化差异是用 6 个候选臂达到 threshold 32 臂的水平（→ 归入"调参经济"主张）。

### 1.6 同密度下三线在区间内实质打平（owner 观察，已核实）
dense（部分）hull 逐成本对比（41.7/43.7/45.7/47.7/49.7 ms）：s0−thr = −0.9/−1.2/−1.0/−0.0/**+1.0** pt；sv−thr = +0.4/+0.1/+0.4/+1.4/+0.5 pt。全程 ≤1.4 pt（bootstrap 噪声半宽 ±4 pt）。冻结基上的 S0−T = −3.8 [CI −10.8, −0.9] 主要是支撑稀疏产物（S0 在 43.5–56.3 ms 无测点；`s0_p85` 补入后弦抬 3–5 pt）。**这正是 §11.5 禁令的道理：matched density 前禁止家族胜负表述——现在两边都 matched 了，结论是平齐。**
高成本区新事实：`s0_p75` = 63.84 ms（94.6%）SR **0.870** > always-full 0.847，且优于 H-CRD q0.60（0.863@96.8%）——"超过全推理"是救回效应（rescue 38%）的家族共性，非任何状态机的独门能力。

## 2. 新定位的完整表述

### 2.1 线上：s-only 就是 threshold 策略
每步同一检索、同一分数 s，两次比较：s ≥ s*_full → FULL；s ≥ s*_warm → WARM；否则 MISS。运行时零新机制（导出时 q̂ 单调曲线已反解成切点存入工件）。(s,v) 版线上也只是"按 v 分桶的阈值表"。

### 2.2 线下：速率索引 vs 风险索引
- threshold：切点 = dev 分数百分位（速率语义）或手调；**不预测风险**，(fh,ws) 组合落点只能 rollout 网格搜。
- s-only：cal rollout 拟合 q̂(s)（结果偏差 y10 的条件上分位，等距分位回归）；选风险容忍 α → δ，全部切点自动反解，落点语义事前给定。

### 2.3 创新点的分层让步-主张结构（owner 定稿）
- **K=1（中/不中）：让步**——分位数倒推与 δ 反解是同一条一维曲线，等价可证明（threshold 族 = 单调规则族 = 校准分位切族）。
- **K≥2：贡献**——"分位数自动化了单个切点，从未自动化切点的组合"。(fh,ws) 二维网格 29 格只有 5 个 hull 顶点（24 格 = 搜索烧掉的自由度代价）；风险阶梯 12 点贴 hull ≤1.4 pt。
- **理论骨架（拟写成命题）**：约束优化 max SR s.t. E[cost]≤B，在"校准单调风险 + 各档成本单调"假设下，K 档最优切点组合满足各档边际风险相等 ⇒ 全体切点由同一乘子 λ(=δ) 索引；**风险阶梯 = 解路径，网格搜索 = 逐格重发现解路径**。K=1 退化为分位数（与让步自洽）。
- **K=3 决胜实验（提案）**：`warm_tiers` 在 config/judge 原生是列表；成本公式本就是 `stage1+stage2+start_t·stage3`（当前钉死 0.3）。需：第三档单价解钉冻结（codex 签字）、第三档风险校准（小批 cal 或声明插值假设）、新 emit 协议。对照：~12 点风险阶梯 vs 预算配平的 3-D 粗网格 12 格，比较垂直 regret 分布。规模 ~4–7k ep。

### 2.4 措辞纪律与相关工作
- 不说"最优帕累托极限/支配/CMDP-optimal"；说"score-threshold 族前沿的风险索引：同天花板（1-D 可证明）+ 免搜索定位 + 事前风险语义 + per-task/跨库可迁移"。
- 必须主动引用并划界：selective prediction / reject option（Geifman & El-Yaniv）、conformal risk control / Learn-then-Test / RCPS、LLM cascade 阈值路由。差异：时序控制（无 i.i.d. 保证，不主张形式保证，与 §12.7 禁令自洽）、FULL/WARM 多档耦合、风险量为动作偏差上分位、机器人缓存系统落地。
- 弹药：GTP 文档自认"carrying a threshold across libraries is exactly the mistake the ratio-based design exists to prevent"——阈值不可迁移，风险索引换库只需重算校准。

### 2.5 处置
SV、CRD、H-CRD 从方法中移除，数据全部保留降为消融/附录：SV ⇒ "v 无增量"（选 s-only 是数据结论）；CRD/H-CRD ⇒ "时序状态机无增量、前沿由信号决定"；batch D + golden + 离线预演器 ⇒ 附录 B 探索记录。

## 3. 下一步实验（owner 已指示）

**s-only（原阶梯）× production hysteresis gate（老配置）**——测"整个系统"（gate + 风险索引 verdict）的综合表现：
- gate：`score_hysteresis`，θ_low=θ_high=0.9928（dev 分数 q0.15，与 Rev 1 secondary / GTP 同一约定）、j=3、probe_interval=3、L=6。
- verdict：s-only 校准阶梯（现有 s0 工件，q∈{0.50…0.975}，含 Rev 1 的 s0/p80/p95）。
- 机制：emit 走 secondary 式 gate 注入（`LAYER_SECONDARY` 期望 gate=score_hysteresis 的通道已存在）；server 无需重启（静态 judge + gate 均为现役代码）；A′ 300 ep/臂。
- 建议批次：先 6 臂（q∈{0.65,0.75,0.85,0.90(s0),0.925,0.975}）=1800 ep ≈ 95 min，覆盖前沿全段；视图形补其余。
- 产出：`pareto` 图新增 "s0+gate" 层；与 GTP 曲线的跨链对照升级为同链系统级对照；为论文提供"系统综合表现"主图素材。

## 4. 待 codex / owner

- 新 G1 提案：本报告 §2 定位 + §3 实验 + K=3 主实验设计；对应解除 §11.5 禁令的证据即本报告 §1.5/§1.6 表格。
- §15.6 未尽项照旧：reviewer patch 未 commit（owner 规则）、export record provenance 待 commit 后重建。
- 本文件与今日所有产出均未暂存。

## 5. 文件索引

- 图：`exp/dispatch_surface/analysis/figures/libero_10/pareto_hull_percent_dense{,_crd}.png`（18 臂正式版；旧版 pareto/family/delta 非 dense 图已按 owner 指示删除，仅留最新代）；报告页 https://claude.ai/code/artifact/5458a825-e32a-45b8-8674-c9289c9aa31e （图 1b = 18 臂 dense 总图，旧版已清理）。
- 数据：`exp/dispatch_surface/data/crd/libero_10/crd_batchD_summary.json`（sha `71008e0c…`）；sgrid 正式 summary `exp/dispatch_surface/data/sgrid/libero_10/sgrid_summary.json`（sha `78eb6491…`，18 臂）。
- 分析脚本：`analysis/crd_offline_screen.py`（参数预演）、`analysis/plot_budget_amendment.py`（`--crd-summary`）、`sgrid_sweep.py`（`summarize --arms`）。
- 交审文档新增节：§16（batch D）、§16.5（GTP 对照）、§16.6（行为分解）、§16.7（离线预演）。

## 6. 相关工作初查（2026-08-30 晚，WebSearch 快扫；投稿前须全文精读）

- **K=1 已知**：selective prediction / risk-coverage（Geifman & El-Yaniv 2017；SelectiveNet 2019）——按置信分数取阈值、扫阈值出 risk-coverage 曲线、给定目标风险反解阈值并给保证。我们的 K=1 让步必须引用这一支。
- **多阈值 + 风险控制（最近邻，须划界）**：Pareto Testing（Laufer-Goldshtein et al., ICLR 2023, arXiv:2210.07913）——多目标优化找参数 Pareto 前沿 + 沿前沿做多重假设检验；Learn-then-Test（Angelopoulos et al.）与 Quantile-LTT；**MultiRisk（arXiv:2512.24587，迭代分数阈值控多风险）**。共性：靠"搜索/检验"选多维阈值配置。我们的差异：单调逐档风险下**闭式等风险耦合**（一次等距分位拟合 → 一个 δ 反解全部切点，解路径无需搜索），并在 matched density 下实证贴合网格前沿——二者互补（他们的检验机器可为我们路径上的点做认证）。
- **级联影子价格（理论近邻）**：*Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades*（arXiv:2605.06350）——k 级 LLM 级联下"单一 shadow price 使各级边际质量/成本相等"的条件刻画，与我们的单乘子解路径同一经济学。划界：他们在有质量标签的 LLM 域做决策论刻画；我们在无标签闭环控制域用动作偏差代理量做校准并给实测。另有 Conformal Cascade（2607.25018）、Calibrate-Then-Delegate（2604.14251）、UCCI（2605.18796，"先校准后取阈"）。
- **本域（无直接前作）**：VLA-Cache（视觉 token 缓存）、Gated VLA-Cache（logit margin 启发式失效门）、BOKBO（VLA 校准弃权）、ElegantVLA（何时思考）、ActionCache（2607.06370，concurrent，已有攻防案）——均无"检索分数索引的多档动作重用 + 逐档校准风险阈值 + 前沿贴合实证"。
- **贡献声明层级（据此定稿）**：不声称发明"校准阈值"这一统计原语；声称 (a) 问题域首创（机器人 VLA 动作缓存的风险校准多档调度），(b) 机制形态（等风险闭式解路径 vs 搜索+检验），(c) matched-density 前沿贴合 + 调参预算 + 负结果的实证学。投稿前必须全文读 MultiRisk 与 2605.06350 并在 RW 中显式划界。
