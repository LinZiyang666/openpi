# Dispatch Surface Rev 2 — 协议草案 **v1**（Execution Authority，2026-08-29，按 §12 预 G1 裁决修订；待 Phase 0 后冻结数值再进正式 G1）

> 上游：[`dispatch_surface_rev1_aprime_result.md`](dispatch_surface_rev1_aprime_result.md) §7–§10；本文 §12 为 Review Authority 对 v0 的裁决，v1 全部采纳。
> Rev 1 已归档：`finalize_cross_suite` → `suite_specific_only`，两个 `line_demoted` 原样保留。
> 本文冻结**设计**；Phase 0 的代码计划见 [`dispatch_surface_rev2_phase0_plan.log.md`](dispatch_surface_rev2_phase0_plan.log.md)（G1 对象）。
> 标 **provisional** 的数值只是目标，按 §3.3 的纯成本规则在 Phase 0 后机械生成并写回，之后不可移动。

## 0. 假说与检验结构（冻结措辞）

在固定的 retriever、library、policy、D_dev、surface fit 下，**在 LIBERO-10 十个任务的独立重采样初始状态上**：

- **H1（唯一 primary）**：`(s,v)` verdict 的 dispatch frontier 相对 scalar-threshold frontier，在预注册共同成本区间 `[c_L, c_H]` 上的 normalized AUC difference `> 0`。
- **H2（fixed-sequence confirmatory key secondary）**：H1 通过后，以同一单侧 α = 0.05 正式检验 SV frontier 相对 S0 frontier 的 AUC difference `> 0`。H2 有确认性地位、无额外 multiplicity；H2 不过时可保留"surface frontier 改善"，**不得**声称改善由 `v` 建立。

libero_spatial 为 near-ceiling negative control（同 protocol，不要求过）。复杂 / 简单只作预注册的 descriptive heterogeneity（§6），不进 Gate。

## 1. 数据角色

| 集合 | 内容 | Rev 2 用途 |
|---|---|---|
| D_lib / D_dev | 官方 init 5 / 15 per task | 不变；D_dev 的 y10 分位数定义 δ 候选 |
| A′（官方 30/task，已看过结果） | **development set** | **两阶段解封**：Stage 1 只读 cost / tier-mix，拟合 δ→realized-cost 映射、按 §3.3 生成区间与锚点并选 δ，写出并冻结 cost-map SHA；Stage 2 才允许读取 outcome，只用于 posthoc development 的 A-2、H1/H2 effect、variance、LOTO、power/A-6。**禁止读取 SR 选点或移动区间；A′ 永不进入 confirmation verdict。** |
| pilot P（新采样 10/task） | 只跑 `always_full_inference` | 一次性验证生成分布（§5.3）；不进分析、不选 δ、与 C 不相交 |
| **确认池 C（新采样 40/task）** | 全部 10 臂配对 | **唯一确认数据**；G2 与 synthetic dry run 全部完成后才 materialize；任何 development 脚本不得获得其路径 |

## 2. 臂（每 suite 10 臂；roster 在 C materialize 前冻结）

| 族 | 臂 | 定义 |
|---|---|---|
| threshold ×3 | `dsp_t_fh30_ws20` / `fh50_ws20` / `fh70_ws10` | 与 Rev 1 逐字相同 |
| SV ×3 | `dsp_sv_a/b/c` | 同一 frozen SV fit；δ 从 development 点集 {p80, **p85**, p90 (+ spatial p95, p97.5)} 中按 §3.3 机械选三个 |
| S0 ×3 | `dsp_s0_a/b/c` | 同一 frozen S0 fit；δ 从 {p80, p90, **p95**} 中按 §3.3 机械选（**不再继承 SV 的 δ**） |
| anchor ×1 | `always_full_inference` | threshold judge，`threshold = 1.5`（cp1 score 为 L2-normalized 内积，上界 1.0），**无 warm tier**，`always_search` gate；matrix 写 `judge_role: always_full_inference_anchor`；dry validation 检查 threshold > 1.0 且无 warm tier；analyzer 机械要求该臂全部 verdict = MISS，否则整臂拒绝；成本声明区分 analytic compute（67.52 ms）与仍发生检索的端到端延迟 |

development 点的来源：p80 / p90 已有（SV−、SV）；**l10 SV p85、spatial SV p95 与 p97.5、两 suite S0 p80 与 p95** 由 Phase 0 补（§7）。跑完后不得在候选中按 SR 挑点。

## 3. 区间、锚点与 δ 的纯成本机械规则（§12.3，冻结）

### 3.1 输入
旧 A′ 上每个 development 臂的每个 (task, init) cell 的 `(cost_sum, n_decisions)`；**不读取 SR**。

### 3.2 frontier 定义（§12.4，H1/H2 共用同一函数）
每个 replicate、每个族的三点 `(cost, SR)`：(1) 取 episode-level randomized mixture 可实现集合的 **upper concave hull**；(2) 删除被同族另一点严格支配的点，同 cost 保留 SR 高者；(3) 只在 hull 的实际 cost support 内线性插值 / 积分。三族不允许不同处理。回归测试三类：中间点在 hull 上、中间点被 chord 支配、同 cost 不同 SR。

### 3.3 区间与锚点：非循环机械顺序（G1R1-B3 / B8，冻结）

输入只有 cost 行（协议 §3.1）。随机性冻结：`numpy.random.Generator(PCG64(20260829))`，10000 replicates，task-stratified、以 init 为 cluster 的 paired 索引**一次生成、全部族共享**并写入输出。

1. **候选集**在 rollout 前冻结（phase0 roster ∪ Rev 1 primary 点）；**SV / S0 族**对候选 δ 用 point-estimate ratio-of-sums cost 拟合 **decreasing isotonic**（δ↑ ⇒ cost↓；权重 = decision 数；PAV；平局按 δ 升序稳定）；**threshold 族不做 isotonic、不伪造 δ**：按预注册 aggressiveness 顺序固定 endpoints = `fh70` / `fh30`、middle = `fh50`。成本单价只来自共享的 `analytic_cost` 模块（全精度 stage 常数，canonical digest 写入所有产物）。
2. **endpoints**（SV / S0）= 每族 isotonic cost 最低与最高的候选。
3. 用固定 endpoints 与共享索引得到每 replicate 的 `L_r = max_family min_endpoint_cost(r)`、`H_r = min_family max_endpoint_cost(r)`：

```text
qL = np.quantile({L_r}, 0.995, method="higher")   qH = np.quantile({H_r}, 0.005, method="lower")   # R=10000: sorted[9950] / sorted[49]（零基）
c_L = ceil_to_0.1ms(qL)                              c_H = floor_to_0.1ms(qH)
c_1 = round_0.1( c_L + (c_H − c_L)/3 )                          c_2 = round_0.1( c_L + 2(c_H − c_L)/3 )
```

4. **middle**：族内只剩一个候选时即为该点；有多候选（spatial SV：p90 / p95）时取 isotonic cost 最接近 `(c_L + c_H)/2` 者，平局取较小 δ。middle 选定后区间**不再改动**。
5. 对所选三点重算 raw point-estimate cost：三点严格不同 ∧ endpoints 包围 `[c_L, c_H]` ∧ `c_H − c_L ≥ 4.0 ms`，否则 **Decision Gate A-3 fail closed**（不得缩区间、不得换点）。

### 3.4 δ 的冻结
上述三点的 δ 与 `[c_L, c_H], c_1, c_2` 在正式 G1 前写回 §4 冻结表；确认池上不得重选 δ、不得移动锚点。

## 4. Estimand 与检验（冻结）

- 每 replicate：按 task 分层、以 init 为 cluster 重采样，同一抽样用于全部 10 臂；每臂算 SR 与**实际** decision-weighted cost（ratio-of-sums）；按 §3.2 构造三条 frontier。
- **H1**：`AUC_norm = (1/(c_H−c_L)) ∫_{c_L}^{c_H} [SR_SV(c) − SR_T(c)] dc`；10000 replicates；**单侧 95% 下界 `> 0`** 判过。
- **H2**：同式以 S0 代 T；仅在 H1 通过后检验。
- **secondary**：`ΔSR(c_1)`, `ΔSR(c_2)` 的 simultaneous 95%（max-t bootstrap band）。
- **support-miss 处理（§12.2-B3，冻结）**：任一 replicate 中某族 frontier 不包围 `[c_L, c_H]` ⇒ 该 replicate 的对应 `AUC_norm` 赋理论最坏值 **−1.0** 并保留在分位数中（不删除、不 clamp、不移动区间）；joint support-miss 率 `> 1%` ⇒ verdict = `support_miss`，拒绝方向性裁决。
- 不做 endpoint clamp、不做 high/low 分支。

**冻结表（libero_10）**

| 项 | 值 | 状态 |
|---|---|---|
| `[c_L, c_H]` | 41.0 – 47.5 | **provisional**（§3.3 生成后替换） |
| `c_1, c_2` | 42.0 / 46.5 | **provisional** |
| δ_a, δ_b, δ_c（SV）/（S0） | — | Phase 0 后 |
| development AUC diff（SV vs T，两点折线，非 hull） | +0.0753，LB +0.030，support-miss 0/4000 | 设计输入，不是确认 |
| 是否被单一 task 驱动 | LOTO 全正（+0.041 … +0.097）：**删除任一 task 后聚合 AUC 仍为正**；不等于每 task 为正——每 task descriptive AUC 见 result 文档 §10.7（task 4/7 为负，task 8 为 floor，task 1 为 ceiling） | R2G1-B1 已改写 |

libero_spatial：development AUC −0.0035，LB −0.024；区间与锚点在 Phase 0（p95/p97.5）后按 §3.3 生成；若激进点仍被 threshold 支配，spatial 只作 control。

## 5. 确认池 C 与 pilot P（fresh sampled init）

1. 生成器：LIBERO 原生 BDDL placement 采样（`env.reset()` 在冻结 seed 序列下采样摆放，取 `sim.get_state()` 的 47 维行向量，写 `{task}.init`，与现有 `.init` 格式一致）；不手工筛除；拒绝 / 重采规则只允许"物理不合法"且写死；生成代码、seed authority、环境与资产 SHA（含 §6 的 assets rollup）在跑任何臂前冻结。
2. 去重：sim-state 内容 digest 对 D_lib / D_dev / 官方 A′ 全部 50 个官方 init 机械检查（预期零碰撞）。
3. **pilot 一次性**：P 上以 `always_full_inference` 跑一次，`|SR_P − SR_0a| ≤ 10 pt`；失败即"generator / distribution validation failed"，**不得换 seed 重试**。
4. C 一次性生成并封存（manifest + sha256 → data_authority）；G2 与 synthetic tests 全部完成后才 materialize；development 脚本不得获得其路径。
5. 论文表述固定："independently sampled initial states for the same task distribution"；不称 official benchmark test set；官方 A′ 上的 Rev 1 负结果完整报告。
6. 外部泛化支柱：真机（owner 已规划；multi-stage 与 simple 各至少一族，同一 frontier protocol）优先于重建第三套模拟链。

## 6. 复杂度：只作 descriptive heterogeneity（§12.6）

按**执行 task id** 重建（修正 v0 的 join 错误）：

| task | libero_10 任务 | goal atoms | n |
|---|---|---|---|
| 0 | LIVING_ROOM_SCENE2 both soup + sauce → basket | In, In | 2 |
| 1 | LIVING_ROOM_SCENE2 both cream cheese + butter → basket | In, In | 2 |
| 2 | KITCHEN_SCENE3 turn on stove + moka pot on it | Turnon, On | 2 |
| 3 | KITCHEN_SCENE4 bowl → bottom drawer + close | Close, In | 2 |
| 4 | LIVING_ROOM_SCENE5 mug → left plate + mug → right plate | On, On | 2 |
| **5** | **STUDY_SCENE1 book → caddy back compartment** | In | **1** |
| 6 | LIVING_ROOM_SCENE6 mug → plate + pudding right of plate | On, On | 2 |
| 7 | LIVING_ROOM_SCENE1 both soup + cream cheese → basket | In, In | 2 |
| 8 | KITCHEN_SCENE8 both moka pots → stove (+ Turnon) | On, On, Turnon | 3 |
| 9 | KITCHEN_SCENE6 mug → microwave + close | In, Close | 2 |

libero_spatial 十个 task 全部 `On(bowl, plate)`，n = 1。完整 task-id → task-name → BDDL path → raw goal atoms 表进入 manifest。
复杂 / 简单解释**不进入 H1**；模拟部分只按 `n_goal_atoms` 分层报告 descriptive 贡献。真机的 multi-stage 定义以**独立 goal-relevant state transitions**（两次放置 / 放置后关门 / 开炉后放置）在看结果前冻结，另设 single-stage 对照。

## 7. Phase 0（旧 A′，全部 `posthoc_exploratory=true`；**G1/G2 后**才放行 rollout）

| 项 | 内容 | 规模 |
|---|---|---|
| 0a | `always_full_inference` × 2 suite | 600 ep |
| 0b | spatial SV p95、p97.5 | 600 ep |
| 0b′ | **l10 SV p85** | 300 ep |
| 0e | S0 **p80、p95** × 2 suite | 1200 ep |
| 0c | HF assets provenance → data_authority（`external_asset` kind；完整 64 位 rollup、license 状态、不可变 manifest） | 0 |
| 0d | phase audit 前置：v3 夹爪弱标签 42/47，剩 5 条多计需视频 / 重放核对后才能用于 mismatch audit | 0（并行，独立于本 protocol） |
| 0f | task manifest join 修正、LOTO 措辞修正、每 task descriptive AUC | ✅ 已做 |

合计 **2700 ep ≈ 2.3 h**。全部产物标 `posthoc_exploratory=true`，不得进入 Rev 1 verdict。

**Decision Gate A**（跑 C 之前全部满足）：
A-1 l10 dev AUC LB > 0 且不由单一 task 驱动 ✅；A-2 spatial 激进点结果已知；A-3 三族共同支撑满足 §3.3（`c_H − c_L ≥ 4.0`，各族三个不同 cost）；A-4 `always_full_inference` **机械判定**：300/300 accepted ∧ verdict 100% MISS ∧ 每 cell decision 与 client_timing 完整一致 ∧ `math.isclose(ratio_of_sums, analytic_cost.unit_cost("MISS", None), rel_tol=0, abs_tol=1e-9)`（全精度 MISS = 67.518595 ms；不写截断字面量；SR 只记录不判定；异常只能走有日志的 protocol amendment）；**A-5** H2 三族共同支撑满足 §3.3；**A-6** H2 在 N = 40、预注册目标效应下功效 ≥ 0.80（不要求 dev LB > 0；点效应 ≤ 0 ⇒ 无方向假说，停止 H2，不得降级后在 C 上继续找 v 正结果）。

## 8. 执行顺序与止损（§12.8）

1. 0c、0f 立即；本文 v1 = 当前文件。
2. Phase 0 代码 G1（[plan](dispatch_surface_rev2_phase0_plan.log.md)）→ Code → G2 → **放行** 0a + 0b + 0b′ + 0e。
3. Phase 0 后按 §3.3 机械生成 `[c_L,c_H], c_1, c_2`、选 δ；补 H2 的效应 / 方差 / LOTO / 功效；过 A-2…A-6；把最终数值写回 §4。
4. 确认协议的**正式 G1**（此时无待裁数值）→ analyzer / generator 代码 → G2 → synthetic dry run → 封存 P、C。
5. **l10 确认**（10 臂 × 40/task = 4000 ep）。H1 的 LB ≤ 0 ⇒ 停止 surface 主线，不用 spatial / secondary / 真机追正结果。
6. l10 通过后：spatial control；真机（multi-stage / simple 两族）。
7. phase-aware retrieval 单独立项，仅在 0d audit 有明确效应后；不在 Rev 2 确认中换 retriever。

## 9. 功效

H1：development 上 AUC diff SD ≈ 0.027（N = 30，两点折线）；效应复现（0.075）时 N = 40 功效 ≈ 0.94；效应减半 ≈ 0.55——写入预注册。
H2：0e 完成后在 development 上按冻结区间、**与 §4 完全相同的 AUC / support 纯函数**（缺支撑 replicate 记 −1 并保留；joint miss > 1% ⇒ fail closed）计算效应、task-stratified 方差、LOTO 与 N = 40 功效：`effect` = 全部 300 个 development cells 的 **full-sample plug-in** AUC difference（hull 必须覆盖区间，否则 A-6 fail；不是 bootstrap mean、不含 −1）；`sd30` = 10000 个 paired bootstrap 差的 sample SD（support-miss replicate 记 −1 并保留）；`power = Phi(effect / (sd30·sqrt(30/40)) − z_0.95)`；`effect ≤ 0`、`sd30` 非有限 / 为 0、support gate 失败 ⇒ A-6 fail。

## 10. 实现清单（本文只列；计划与 G1 见 phase0 plan）

exploratory exporter / emitter `exploratory` 层与 anchor role / runner 校验 / analyzer 拒绝 exploratory + phase0 summary（含 all-MISS 检查）/ cost-map 脚本（§3.3、§3.4）/ data_authority `external_asset` / task manifest join。确认协议本身的 analyzer（hull、AUC、support-miss = −1.0、fixed sequence）与 fresh-init generator 属第二个 G1。

## 11. 待裁项（v0 五项已裁，见 §12.1）

无新增。provisional 数值（§4）不是待裁项：由 §3.3 生成，reviewer 与 executor 都不手选。

---

## 12. Review Authority 对 v0 的预 G1 裁决（Codex，2026-08-29）

**Verdict：NEEDS REVISION。五项未决已有明确裁决，但修完以下 protocol blockers 前不进入正式 G1，
server 继续保持停止。** Phase 0 rollout 的科学方向获批；实际运行仍须先完成 Rev 2 代码 G1/G2。

### 12.1 五项未决裁决

1. **Phase 0：条件放行。** 0a/0b/0e 允许进入代码计划与 G1/G2；G2 通过后可在旧 A′ 上运行，所有
   产物必须标 `posthoc_exploratory=true`，不得进入 Rev 1 verdict。0c 不应等待 rollout，补齐 license 与
   完整 64 位 rollup 后立即进入 data_authority。
2. **H2：不是第二 primary；定义为 fixed-sequence confirmatory key secondary。** H1 是唯一 primary。
   H1 通过后，H2 以同一单侧 α=0.05 正式检验，因此仍有确认性地位且无额外 multiplicity penalty；H2
   不过时可以保留“surface frontier 改善”，但不得声称改善由 v 建立。这个命名同时满足唯一 primary 与
   “v 必须正式验证”两项要求。
3. **确认池 N=40/task。** 当前增量成本相对整条链很小，而 development 效应减半时功效本就只有约
   0.55；没有理由为了节省约 1/4 rollout 把 full-effect 功效从约 0.94 降到 0.88。pilot 10/task 另计且
   永不进入确认。
4. **暂不批准 `[41.0,47.5]` 与 `42.0/46.5` 为冻结数值。** 当前 SV 两端到该区间仅约 1.3 ms，和正文
   `m=2.0 ms` 自相矛盾；S0 三点又尚不存在。锚点在 Phase 0 后按 §12.3 的纯成本机械规则生成，不能
   由 reviewer 手选。现有数字只保留为 provisional target。
5. **task 9 不以 horizon=520 自动归入 complex。** 520 是 suite 的评测超时预算，不是任务结构。
   更严重的是当前表存在 task-id 错配：权威 init map 中 task 5 才是 `STUDY...book...caddy`，task 9 是
   `KITCHEN...mug...microwave_and_close_it`；§6 把 task 9 写成 book，说明机械提取结果尚未按执行 task id
   正确 join。复杂度表必须重建，在此之前 task 9 不冻结分类。

### 12.2 新发现的三个 protocol blockers

#### R2G1-B1 — LOTO 被误读为“每个 task 都为正”

`leave-one-task-out 全正` 只说明删除任意一个 task 后**聚合 AUC**仍为正；它不等于十个 task-specific
AUC 全为正。草案可保留“not driven by any single task”，不得写“10 个 task 全正”。G1 前必须同时输出
每 task 的 descriptive AUC/discordance，明确 ceiling/floor task；不把它们作为选择门。

#### R2G1-B2 — 10 臂 roster 仍缺 l10 第三个 SV development 点

确认 roster 要三点 SV，但 l10 development 只有 p80/p90；仅因两点已覆盖成本区间，不能凭空选择第三个
δ。Phase 0 增加 **l10 SV p85**（exploratory，300 ep）作为中间形状点。S0 的“一个激进点”也必须在
rollout 前写死，冻结为 **p95**，所以每 suite S0 development 点固定为 p80/p90/p95。不得跑完后在
p95/p97.5 中挑更好看的。

spatial 的 p95/p97.5 两点仍都运行，用来判断低成本支撑及曲线形状；最终三点只能由纯 cost-map 规则选，
不能用 SR 选择。

#### R2G1-B3 — support_miss ≤1% 时如何处理未定义 replicate 没有规定

缺支撑的 bootstrap replicate 不能静默删除，否则 AUC 分布会条件化在“恰好覆盖”上并产生乐观偏差。
冻结处理如下：对 H1/H2，每个 support-miss replicate 将该 endpoint 的 `AUC_norm` 赋为理论最坏值
`−1.0` 并保留在分位数中；joint support-miss rate `>1%` 时仍输出 `support_miss`、拒绝方向性裁决。
测试必须钉死“不删除、不 clamp、不移动区间”。

### 12.3 cost-only 区间与锚点的机械生成规则

Phase 0 完成后，只用旧 A′ 的 cost/tier rows 做冻结 bootstrap；禁止读取 SR。每个 replicate `r` 对
`T/SV/S0` 三族分别得到三点 cost，定义：

```text
L_r = max_family min_point cost(r)
H_r = min_family max_point cost(r)
c_L = ceil_to_0.1ms(quantile_0.995({L_r}))
c_H = floor_to_0.1ms(quantile_0.005({H_r}))
c_1 = round_to_0.1ms(c_L + (c_H-c_L)/3)
c_2 = round_to_0.1ms(c_L + 2(c_H-c_L)/3)
```

这直接控制两端联合 support-miss 约不超过 1%，替代任意的 `m=2 ms`。若 `c_H-c_L < 4.0 ms`，或任一
family 没有三个严格不同的 point-estimate costs，则 Decision Gate A-3 失败，停止 H2/frontier 设计，不得
缩区间救结果。最终数值在 G1 前写回 §4，之后不可移动。

### 12.4 “frontier”必须定义为可实现的上包络

三点按 cost 排序直接连线可能包含被支配点，也可能因抽样噪声出现 SR 随成本下降/上升的折返；那不应
称 Pareto frontier。每个 replicate 对每个 family 按以下唯一算法构造：

1. 以 `(cost, SR)` 三点计算 episode-level randomized mixtures 可实现集合的 **upper concave hull**；
2. 删除被同族另一点严格支配的点；相同 cost 保留 SR 较高者；
3. 只在 hull 的实际 cost support 内线性积分/插值。

H1/H2 的 AUC 都使用该共享函数；threshold/SV/S0 不允许不同处理。应提供三类回归：中间点在 hull 上、
中间点被 chord 支配、相同 cost 不同 SR。若作者不愿采用 hull，则全文必须改称“three-point operating
curve”并保留原始折线；两种定义必须在 G1 二选一。Review Authority 推荐 upper concave hull。

### 12.5 H2 的功效和 Decision Gate 不能缺席

当前功效只估了 H1。0e 完成后必须在 development set 上按冻结区间计算 H2 `AUC(SV)-AUC(S0)` 的效应、
task-stratified variance、LOTO 与 N=40 功效。Decision Gate A 新增：

- A-5：H2 三族共同支撑满足 §12.3；
- A-6：H2 在 N=40 的预注册目标效应下功效 ≥0.80。

A-6 不要求 development LB>0（那会把确认假说按同一批 SR 结果筛得过狠），但若点效应 ≤0，则没有合理
方向假说，停止 H2；不得把 H2 降级后继续在确认集寻找 v 正结果。

### 12.6 复杂度主张的处理

当前 `predicates ≥2 OR horizon≥400` 实际上几乎等价于 suite label，且 task join 已出错，不能进入冻结假说。
Rev 2 H1 先准确限定为 **“on independently sampled initial states of the ten LIBERO-10 tasks”**。复杂/简单
解释在模拟部分只作预注册 descriptive heterogeneity，不作为 H1 Gate。

若要让真实机器人承担“复杂任务更受益”的外部假说，应在看结果前以**独立 goal-relevant state
transitions**定义 multi-stage（例如两个物体放置、放置后关门、开炉后放置），而非 suite horizon；同时设
simple 单阶段对照。task-id→task-name→BDDL path→raw goal atoms 的完整表必须进入 manifest。

### 12.7 fresh-init 与 pilot 的补充纪律

同分布重采 init 的确认地位获批，但 pilot 的 `|ΔSR|≤10 pt` 只能执行一次：生成器代码与 seed authority
必须先冻结；pilot 若失败，结论是 generator/distribution validation failed，不能换一批 seed 再试。确认池 C
仍须在全部 G2/synthetic tests 完成后才 materialize，且任何 development 脚本不得获得其路径。

### 12.8 修订后的执行顺序

1. 立即完成 0c（完整 SHA + license + `external_asset` schema）；修正 task manifest/join 与 LOTO 措辞。
2. 将本节 R2G1-B1…B3、§12.3–§12.7 写入正文，明确 l10 p85 与 S0 p95，形成 Rev 2 v1。
3. 对 Phase 0 exporter/emitter/runner/analyzer-support 准备 G1 plan；**G1/G2 通过后**，实际放行
   0a + spatial p95/p97.5 + l10 p85 + 两 suite S0 p80/p95。预计总量从原 2400 ep 增为 2700 ep。
4. Phase 0 后机械生成 `[c_L,c_H],c_1,c_2`，完成 H2 power 与 Decision Gate A-2…A-6；写回最终数值。
5. 再走 confirmation protocol 的最终 G1（此时无待裁数值）→ Code/G2 → P/C 封存与 l10 N=40 确认。

在这之前不启动任何 rollout。以上不是否定 Rev 2；相反，development AUC `+0.075` 且 LOTO 不被单一
task 翻转，是足以继续投入 Phase 0 的证据。需要防住的是第二次因 operating-point/support 定义不完整而
让一个本来有信号的方法输在协议上。
