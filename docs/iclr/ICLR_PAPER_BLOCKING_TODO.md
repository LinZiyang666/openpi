# 🚨 ICLR PAPER BLOCKING TODO — 任何论文级工作开始前必须先完成

> **最高优先级前置 Gate。** 在本文件所有 P0 项完成并经 Review Authority 复核前，禁止继续进行 ICLR 论文正文撰写、主图定稿、主结果表定稿、摘要/贡献点定稿或 confirmation rollout。
>
> 本 TODO 记录 2026-08-29 Phase 0 结果后确定的统一迁移：把 success–compute 主读出从“恰好位于 realized cost $c$ 的 exact-cost frontier”改为“计算预算不超过 $B$ 时的最优成功率”——即 **compute-budget value envelope**。
>
> 这不是修改成本口径。每个 arm 的真实解析成本、SR、原始 rollout 与三段单价全部保持不变；改变的是多个实测 operating point 如何形成方法族曲线，以及论文如何解释这条曲线。

## 0. 完成状态

- [ ] P0-A：数学对象与术语统一
- [ ] P0-B：统计协议与 analyzer 迁移
- [ ] P0-C：论文全文与图表迁移
- [ ] P0-D：历史记录、结果边界与 baseline 防守
- [ ] P0-E：验证、复核与最终放行

只有五项全部勾选，才允许在本文件顶部追加：

> `PAPER GATE PASSED — <date> — <review log / commit>`

## 1. 唯一目标定义

### 1.1 可达集合

对一个方法族 $F$（threshold、SV、S0、Action Cache、reduced-NFE 或 learned router）定义：

\[
\mathcal A_F
=
\left\{
\bigl(C(\pi),\operatorname{SR}(\pi)\bigr),\quad C(\pi)=\frac{\mathbb E[T(\pi)]}{\mathbb E[D(\pi)]}:
\pi\in\operatorname{mixtures}(F)
\right\}.
\]

`mixtures(F)` 只允许在已测 operating policies 之间做 **episode-level randomized mixture**。不允许凭空构造新 policy，不允许把逐 step 反事实混合伪装成实测结果。

### 1.2 主曲线：budget-value envelope

全文统一使用：

\[
V_F(B)
=
\sup_{\pi\in\operatorname{mixtures}(F)}
\left\{
\operatorname{SR}(\pi):
C(\pi)\le B
\right\}.
\]

其含义是：**在 model-forward compute 不超过预算 $B$ 时，方法族 $F$ 可达到的最高闭环成功率。**

### 1.3 唯一构造规则

1. 每个 marker 仍位于该 arm 的真实 `(realized analytic cost, measured SR)`。
2. **（2026-08-29 G1R2-B1 修订）不删除任何点、不做同成本去重**：在 ratio-of-sums 分式成本下，`(c,s)` 上被支配的 arm 可能因 `d_i` 更大成为最优混合基（反例：A(c=10,d=1,s=0.5)、B(c=20,d=100,s=0.4)、C(c=100,d=1,s=1.0)，B=30 时 A/C 最优 0.611、B/C 最优 0.961）。standalone Pareto 标记只可作图形描述，不得作为 LP 输入剪枝。
3. **（2026-08-29 G1 Round 3 冻结）** episode-level mixture 的成本是 decision-weighted ratio-of-sums 的分式：`C(p) = Σ p_i t_i / Σ p_i d_i`（`t_i`、`d_i` = arm `i` 每 episode 的总解析成本与决策数），SR 为 `Σ p_i s_i`。因此 $V_F(B)$ 不是 `(cost, SR)` 点的直线 upper concave hull，而是一维预算 LP：$V_F(B)=\max_p\{\sum_i p_i s_i:\ \sum_i p_i (t_i - B d_i)\le 0,\ \sum_i p_i = 1,\ p\ge 0\}$；最优基至多两臂，两臂片段为线性分式曲线（仅当各臂 `d_i` 相同才退化为直线）。
4. 不做 `(cost, SR)` 直线插值；积分为精确分片解析（断点 = 单臂成本 ∪ 三点共线的线性根；每片常数或线性分式闭式原函数）。
5. 预算语义由 LP 的 `≤` 约束内生：`B` 不小于全部臂的成本时 $V_F(B)=\max_i s_i$；不再需要"最后有效点之后水平保持"的附加规则（原规则的意图——较便宜 policy 仍满足更高预算——由约束本身给出）。
6. 所有臂可行后的常数段表示 `C(π) ≤ B` 的可行性，**不表示**算法实际消耗了 $B$，不表示进行了额外 rollout，也不表示应故意烧掉剩余计算。
7. 低于该 family 最低可行成本时仍为 support miss；禁止向左外推。
8. 曲线只画到预注册共同预算区间 $B_H$ 或明确的图域终点，禁止无限延伸制造全局支配观感。
9. threshold / SV / S0 / Action Cache / 其他 baseline 必须使用完全相同的构造规则。

### 1.4 主比较量

H1 统一改写为：

\[
\Delta_{\mathrm{H1}}
=
\frac{1}{B_H-B_L}
\int_{B_L}^{B_H}
\left[V_{\mathrm{SV}}(B)-V_{\mathrm T}(B)\right]\,dB.
\]

论文解释必须是：

> 在预注册 compute-budget 区间内，SV 相对 threshold 的**平均成功率提升**。

不得再把它写成“在每个点恰好消耗完全相同成本”。

## 2. P0-A — 数学与术语替换清单

- [ ] 在方法部分引入 $\mathcal A_F$ 与 $V_F(B)$（分式成本预算 LP），说明实测点、episode-level 随机混合（两臂分式片段）与全可行常数段各自的含义；不再使用 “upper concave envelope” 描述主曲线。
- [ ] 将 dispatch H1/H2 中的 `SR_F(c)` 或 exact-cost interpolation 改为 $V_F(B)$。
- [ ] 将“common cost interval `[c_L,c_H]`”统一改称“common compute-budget interval `[B_L,B_H]`”；数值可以不变，但符号和解释必须统一。
- [ ] 明确 $R_{\mathcal C}(\varepsilon)$ 与 $V_F(B)$ 是同一可达集合的两个约束视角：前者固定 SR 损失预算、最大化可回收计算；后者固定 compute budget、最大化 SR。
- [ ] 严格区分 `measured policies`（含 standalone-dominated 但进入 active mixture basis 的臂，须如实保留）与 `budget-value envelope`。可以把整图称为 success–compute budget frontier，但不能声称水平段本身是新实测点。
- [ ] 全文把 `at matched realized cost` / `at exactly the same cost` 改为 `under the same compute-budget cap` 或 `at or below the same compute budget`。
- [ ] 保留 `realized cost` 一词描述单个 arm、成本表和日志；不得机械地把所有 `cost` 替换成 `budget`。
- [ ] 任何“dominates”都限定在预注册预算区间和已比较 family；禁止写全局支配或“优于任何 scheduler”。

### 术语迁移表

| 旧写法 | 新写法 | 备注 |
|---|---|---|
| realized-cost frontier | compute-budget value envelope | family 曲线 |
| cost $c$ on the curve | budget cap $B$ | 曲线自变量 |
| matched compute / matched cost | same compute-budget cap | 比较语义 |
| SR at cost $c$ | best attainable SR under budget $B$ | $V_F(B)$ |
| cost support covers both endpoints | family is feasible at $B_L$ | 右端由预算语义自然覆盖 |
| AUC over costs | average SR difference across budgets | 结果解释 |

## 3. P0-B — 统计协议与代码迁移

- [ ] 新建正式 protocol amendment；旧 Phase 0 protocol 和 result 原样保留，不回写、不覆盖、不删除。
- [ ] **（G1R1-B1 修订）** 新增 budget-mixture 纯函数：输入每臂 sufficient stats `(T, D, S, E)`，按分式成本 LP 求 $V_F(B)$（单臂 / tight 两臂枚举，hull-at-zero 等价）；旧 exact-cost 函数原样保留。
- [ ] 全可行常数段必须来自 `C(π) ≤ B` 的 LP 定义；同时输出 measured-policy-only step envelope `max{s_i: T_i/D_i ≤ B}` 作为非 gating 敏感性。
- [ ] support：只在 $B_L$ 低于 family 最低单臂成本时 fail；LP 不做任何剪枝，因此不存在“端点被剪掉”的右端 miss。
- [ ] **（G1R1-B1 / G1R3 数值审计修订）** exact AUC 积分：断点为 `[B_L,B_H]` 两端、区间内单臂成本、三点共线的线性根；每片常数或线性分式闭式积分。自适应 Simpson 只审计 G2 随机/退化 fixture、development/C 的 full-sample + digest 派生最多 100 个 replicate、power 每个 `(N,outer_r)` 的 inner-0；不得进入全部 formal replicate 热路径或替代解析值。
- [ ] H1/H2、LOTO、per-task descriptive、A-2 和所有 family 共用同一个 budget-value 函数。
- [ ] 使用冻结的 Phase 0 输入生成**新的 amendment analysis artifact**；不得覆盖现有 `phase0_outcome_design.json`。
- [ ] 结果中同时记录 measured points、分片（活跃基、断点）、budget interval、AUC 与 support-miss；不再有“水平段起点”字段。
- [ ] 补旧实现必败的回归测试：(a) 被支配高成本点导致旧 `covers` 失败而新 LP 在更高预算仍可行；(b) G1R2-B1 反例——按 `(c,s)` 剪枝会把 `V(30)` 从 0.961 算成 0.611。
- [ ] 补三族同规则测试、左端不可行测试、分式片段与常数段的解析积分值、canonical tie 五类、同 family/同曲线差为零、A−B 反对称测试。
- [ ] confirmation analyzer、figure exporter 与表格 exporter 必须读取同一 frozen estimator version / digest。

### 功效估计必须同时修正

- [ ] 禁止把 support-miss sentinel `−1` 的混合 SD 直接代入正态功效公式后宣称真实 power = 0.264。
- [ ] 禁止删除 sentinel 后把 conditional SD 直接代入并宣称真实 power = 0.919。
- [ ] 对候选 `N/task` 做外层 paired、task-stratified pilot resampling；每个外层样本内部运行完整 bootstrap、budget frontier、support gate 与 q05 判定。
- [ ] 以“support gate 通过且单侧 q05 > 0”的频率作为完整裁决 power；候选 N、seed、replicate 数和最小通过阈值必须预先冻结。
- [ ] fresh C 的 N 只能由该 power record 机械选出，不能在看到 C outcome 后调整。

## 4. P0-C — 论文与图表逐文件迁移

下列是现行论文级入口，必须逐一审计；不能只改一张图：

- [ ] `docs/iclr/latex/paper_outline.tex`
  - 主 frontier 改成 compute-budget view。
  - 贡献与结果统一写 under-budget，而非 exact matched cost。
- [ ] `docs/iclr/latex/redundancy_note.tex`
  - 增加 $\mathcal A_F$、$V_F(B)$ 与 $R(\varepsilon)$ 的双视角关系。
  - 不改变 $R(\varepsilon)$ 已有定义，只补统一接口。
- [ ] `docs/iclr/latex/dispatch_note.tex`
  - primary readout 改为 budget-value envelope；保留单 arm 的真实 GPU-time / analytic cost。
- [ ] `docs/iclr/latex/experiment_list.tex`
  - E1 / dispatch-surface 相关 readout 改为 budget frontier；等成本措辞改为预算上限措辞。
- [ ] `docs/iclr/paper_rethink_discussion.md`
  - 主图、核心 claim、结果解释与 $R(\varepsilon)$ 的接口同步。
- [ ] `docs/iclr/dispatch_defense_plan.md`
  - `matched compute` 改为 `under a common compute-budget cap`。
  - surface vs threshold / Action Cache 的胜负只在共同预算区间陈述。
- [ ] `docs/iclr/actioncache_response_plan.md`
  - Action Cache、threshold、surface 采用同一 budget envelope；不得只给本方法横向延伸。
- [ ] `docs/iclr/actioncache_positioning_plan_codex.md`
  - success–compute frontier 的语义、图注与主张同步。
- [ ] `logs/dispatch_surface_rev2_protocol_draft.md`
  - 以 amendment 形式更新，不删除原冻结定义和修订轨迹。
- [ ] `logs/dispatch_surface_rev2_phase0_result.md`
  - 保留旧结果；链接新的 amendment artifact，并明确 development-only。
- [ ] 所有新正文、摘要、slides、图注、表格与 rebuttal 材料全文搜索：`matched cost`、`matched compute`、`exact cost`、`realized-cost frontier`、`support`、`Pareto`、`AUC`。

### 主图硬性规范

- [ ] x 轴：`Model-forward compute budget per decision (ms)`，不能只写含混的 `Cost`。
- [ ] y 轴：`Best attainable success rate under budget`，或正文已定义后的简写 `Success rate`。
- [ ] marker 表示真实测量 arm；marker 不得被线遮掉。
- [ ] curve 表示 budget-value envelope；两臂分式混合段与全可行常数段在 caption 中说明。
- [ ] 常数段只能延伸到冻结图域 / 共同预算区间终点。
- [ ] teacher / always-full anchor 仍放在其真实成本位置。
- [ ] 主图旁或表格继续报告每个 arm 的 realized cost，避免预算曲线掩盖真实开销。
- [ ] caption 固定包含：`Markers are measured operating points; curves are upper attainable envelopes under a compute-budget cap.`

## 5. P0-D — 结果边界与 baseline 防守

- [ ] Phase 0 的 old-estimand development 结果永久保留；新分析必须带 `posthoc_design_amendment=true` 或等价 provenance。
- [ ] 明确 fresh C 尚未触碰；只有 C 上按新冻结 analyzer 得到的结果才是 confirmation。
- [ ] l10 development 的推荐写法：在 `[41.7,47.7] ms` budget interval 上，SV 相对当前三点 threshold family 的平均 SR 差为 `+7.2 pp`；不得提前写成确认结果。
- [ ] spatial development 的推荐写法：budget 语义消除技术性右端 support miss 后，H1 仍为负；它是 near-ceiling negative control，不是“任何 scheduler 都一样”的证明。
- [ ] H2 / `v` 的地位另行由 final protocol 冻结；在它没有独立闭环证据前，不得把代数分解写成因果机制贡献。
- [ ] 在使用 `tuned threshold` 或 `dominates threshold` 前，验证 threshold development grid 足以形成可信 upper envelope；否则只能写 `the preregistered three-point threshold family`。
- [ ] 最接近 concurrent work 的 Action Cache baseline 必须用同一预算定义；budget migration 不能代替 baseline 强度审查。
- [ ] 解析 model-forward compute 与端到端 latency 继续分开；本迁移不授权把解析成本描述成实测延迟。

## 6. P0-E — 完成判据

以下全部满足才可勾选总 Gate：

- [ ] 数学定义、protocol、analyzer、figure exporter 和论文用词五方一致。
- [ ] 旧 exact-cost Phase 0 artifact 与新 budget amendment artifact 同时可追溯，且 SHA / provenance 完整。
- [ ] 新 analyzer 在冻结旧数据上复现设计诊断的方向：l10 H1 正、spatial H1 负；数值差异若存在必须解释，不得只改报告。
- [ ] 全部定向测试、对抗回归和 source-lock 通过。
- [ ] 独立 reviewer 检查常数段确实来自 `C(π) ≤ B` 的 LP、**没有做任何 Pareto / 同成本剪枝**、没有右侧无限外推、没有 family 特判。
- [ ] 独立 reviewer 检查 power 来自完整裁决模拟，而非 sentinel 混合 SD 的正态近似。
- [ ] 独立 reviewer 检查论文所有“提高多少”都写明 budget interval、baseline family、development/confirmation 身份和成本口径。
- [ ] owner 最终确认 H1/H2 地位、confirmation N、threshold baseline 强度和 fresh C 放行。

## 7. 明确不在本 TODO 中授权的动作

- 不授权启动 fresh C rollout。
- 不授权覆盖或删除 Phase 0 / Rev 1 产物。
- 不授权修改三段解析成本或改用另一成本轴。
- 不授权把 spatial H2 的事后正 q05 升格为确认性证据。
- 不授权因为横线存在而声称每个预算点都实测过。
- 不授权跳过新的 G1/G2；本文件是工作清单，不是 protocol amendment 本身。

## 8. 完成后的标准主张模板

只有 fresh C 通过后才能使用确认口吻：

> Across the preregistered model-forward compute-budget interval \([B_L,B_H]\), risk-calibrated three-way dispatch improves the best attainable closed-loop success rate over the frozen threshold baseline by an average of \(X\) percentage points.

在 confirmation 前只能写：

> On the development set, the budget-value envelope of risk-calibrated three-way dispatch exceeds that of the preregistered threshold family by an average of 7.2 percentage points on LIBERO-10 over 41.7--47.7 ms per decision; fresh-initial-state confirmation remains pending.
