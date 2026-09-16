# Dispatch Surface Rev 2 — Phase 0 结果与裁决请求

> **术语统一（2026-08-31，owner 裁定，全库生效）**：**GST = Grid-Searched Threshold（网格搜索阈值）**——本文中的 threshold / thr / T / tgrid / (fh, ws) 网格 / 速率索引均指它；**RIT = Risk-Indexed Threshold（风险索引阈值）**——本文中的 s-only / s0 / 校准分位切 / 风险阶梯 / surface 均指它（(s,v) 版 SV 为 RIT 的消融）。历史正文与 Review Log 按章程不改写，以本注释为准。

> 执行方（Execution Authority）记录，2026-08-29。owner 于 14:00 放行 rollout（weilandserver 4 replica / timan107 48 worker）；2700 ep 于 14:02–15:38 跑完，分析于 16:00 前完成。
> **本文全部数字来自 development set（旧 A′）上的 exploratory 产物（`posthoc_exploratory=true`, `development_only=true`），不是确认结果；不得回写 Rev 1 verdict，也不改变 Rev 1 的负结论。**
> 协议：[`dispatch_surface_rev2_protocol_draft.md`](dispatch_surface_rev2_protocol_draft.md)；代码计划：[`dispatch_surface_rev2_phase0_plan.log.md`](dispatch_surface_rev2_phase0_plan.log.md)；Rev 1 结果：[`dispatch_surface_rev1_aprime_result.md`](dispatch_surface_rev1_aprime_result.md)。
> 成本全部来自 `analysis/analytic_cost.py` 的解析单价（三段常数之和：FULL_HIT 10.260266 / WARM_START@0.3 46.818293 / MISS 67.518595 ms），**没有任何实测计时**；下文 "cost" 一律指按 verdict 查表的 ratio-of-sums。
> §1–§3 是机械输出；§4 是执行方的独立验算；§5–§6 是执行方的解读与提案，**不是裁决**；§7 列出需要 Review Authority / owner 裁决的事项。

## 0. 摘要

| | libero_10 | libero_spatial |
|---|---|---|
| A-4 anchor | ✅ 300/300，18023 决策全 MISS，cost = 67.518595 | ✅ 300/300，6756 决策全 MISS，cost = 67.518595 |
| 冻结区间 `[c_L,c_H]` / c₁ / c₂ | [41.7, 47.7] / 43.7 / 45.7（A-3 ✅） | [34.4, 41.7] / 36.8 / 39.3（A-3 ✅） |
| H1 SV−threshold | 效应 **+0.072**，q05 **+0.022 > 0**，LOTO 10/10 正；N=40 功效按冻结口径 **0.26** | 效应 −0.018；缺支撑 54.6% → `support_miss` |
| H2 SV−S0 | 效应 +0.041，q05 −0.014，LOTO 10/10 正；功效 **0.13 ⇒ A-6 ❌** | 效应 +0.028；缺支撑 16.5% ⇒ **A-5 ❌** |
| A-2（spatial 描述性） | n/a | SV 全区间被 threshold 支配（−0.055 … −0.002） |

- **Decision Gate A 未全部满足（l10 A-6、spatial A-5），按协议不得进入 C。**
- 执行方验算表明两处失败的性质不同：l10 H1 的"功效 0.26"是 G1 冻结的 support-miss = −1 约定进入功效公式造成的（去掉 −1 后 0.92，与 §9 预注册的 0.94 一致），**不是效应变弱**；l10 H2 即使去掉 −1 也只有 0.42，**A-6 失败是实质的**；spatial 的缺支撑几乎全部来自 near-ceiling 下的 Pareto 支配剪枝，是估计量定义的结构性现象。
- 需要裁决的事项见 §7；执行方未改任何冻结函数、阈值、区间。

## 1. 执行记录

| 项 | 内容 |
|---|---|
| 运行时布局（本地 / weilandserver / timan107 同路径） | 归档包副本 `/tmp/dsp_shared/<suite>/rev1_discipline/`（MANIFEST l10 `48eccb9f…` / spatial `4f9f79b2…`）；正式导出 `/tmp/dsp_shared/<suite>/exploratory/{sv,s0}/`（7 个 artifact 与 Verify 时的验证导出逐字节一致）；矩阵 `/tmp/dsp_shared/config/precheck_<suite>_exploratory/`（emit 在 weilandserver 执行，因本地无 lib.pkl；13 个文件三机 sha 核对） |
| dry-validate | 两 suite `PRECHECK_EXIT=0`（含从真实 server 读回 launch contract），未写 ledger |
| 0a anchor | l10 `run_id 45685630af58` / spatial 各 300 ep，14:02–14:29；A-4 两 suite 通过后才放其余臂 |
| 其余臂 | 14:29 起 timan107 `srv0` `/tmp/dsp_phase0_rest.sh`：l10 `run_id cdf846ed53f5`（`dsp_sv_p85, dsp_s0_p80, dsp_s0_p95`，15:18 完成）→ spatial 4 臂（15:38 完成）。合计 2700 ep，1 h 36 min，全程 err=0、无 ALERT |
| 本地回拉 | `exp/dispatch_surface/data/aprime_rev1/<suite>_exploratory/`，journal / per_step / ledger 逐文件 sha 核对 |
| 分析顺序 | 正式 `phase0_summary`（roster_complete）→ `cost_map`（outcome-blind）→ 记录 SHA → `phase0_outcome_design --cost-map-sha256 <SHA>`（此时才读 SR） |

**A-4（anchor 机械判定）**

| suite | accepted | decisions | verdicts | ratio-of-sums | `unit_cost(MISS)` | anchor SR（只记录，不判定） |
|---|---|---|---|---|---|---|
| libero_10 | 300/300 | 18023 | 100% MISS | 67.518595 | 67.518595 | 0.847 |
| libero_spatial | 300/300 | 6756 | 100% MISS | 67.518595 | 67.518595 | 0.957 |

## 2. libero_10

### 2.1 Phase 0 臂（Stage 1，只读成本）

| 臂 | family | q | δ | cost (ms) | decisions | FULL / WARM / MISS |
|---|---|---|---|---|---|---|
| always_full_inference | anchor | — | — | 67.5186 | 18023 | 0 / 0 / 18023 |
| dsp_s0_p80 | s0 | 0.80 | 5.2381 | 56.2580 | 17963 | 2 / 9766 / 8195 |
| dsp_sv_p85 | sv | 0.85 | 5.5032 | 45.7798 | 21954 | 6154 / 6033 / 9767 |
| dsp_s0_p95 | s0 | 0.95 | 6.7314 | 39.8678 | 26018 | 10430 / 5904 / 9684 |

### 2.2 cost map（§3.3 机械规则；`cost_map_frozen.json` sha256 `da17c19843ab7ea3d3ec83dddfc42877f4bf3679d5ecf8a2e3870541b8550904`）

- `qL = 41.6189`（0.995 higher，idx 9950）、`qH = 47.7275`（0.005 lower，idx 49）→ **`[c_L, c_H] = [41.7, 47.7]`**，`c_1 = 43.7`，`c_2 = 45.7`；宽 6.0 ≥ 4.0。
- 三族选点（低 / 中 / 高成本）：SV `dsp_sv`(p90, δ 5.9096) / `dsp_sv_p85`(δ 5.5032) / `dsp_sv_minus`(p80, δ 5.2381)；S0 `dsp_s0_p95` / `dsp_s0`(p90) / `dsp_s0_p80`；threshold `fh70_ws10` / `fh50_ws20` / `fh30_ws20`。
- **A-3 ✅**（三点严格不同，端点包围区间）。

### 2.3 outcome design（Stage 2；`phase0_outcome_design.json`）

点估计 (cost, SR) 与 upper concave hull：

| family | 低 | 中 | 高 | hull |
|---|---|---|---|---|
| SV | (39.72, 0.493) | (45.78, 0.633) | (48.83, 0.763) | 中点在弦下方，hull 为两端点连线 |
| S0 | (39.87, 0.397) | (43.47, 0.583) | (56.26, 0.843) | 三点 |
| threshold | (39.34, 0.403) | (44.33, 0.567) | (50.20, 0.697) | 三点 |

| 假设 | 效应（plug-in） | bootstrap mean / q05 / q95 | support-miss | sd30（含 −1） | N=40 功效 | LOTO |
|---|---|---|---|---|---|---|
| H1 SV−threshold | **+0.0722** | 0.0641 / **+0.0224** / 0.1135 | 0.53% | 0.0823 | **0.264** | 10/10 正，最小 +0.033（task 6） |
| H2 SV−S0 | +0.0407 | 0.0351 / −0.0136 / 0.0954 | 0.69% | 0.0922 | **0.128** | 10/10 正，最小 +0.017（task 6） |

每 task 描述性 AUC（不判定；"—" = 该 task 上无共同支撑）：H1 = 0:+0.148, 1:—, 2:+0.269, 3:—, 4:−0.045, 5:—, 6:+0.342, 7:−0.192, 8:+0.068, 9:+0.001；H2 = 0:+0.219, 1:—, 2:+0.071, 3:+0.038, 4:−0.066, 5:−0.119, 6:+0.290, 7:−0.033, 8:+0.080, 9:−0.024。无 ceiling / floor task。

### 2.4 Decision Gate A（l10）

| gate | 结果 | 依据 |
|---|---|---|
| A-1 | ✅ | H1 dev q05 +0.022 > 0，LOTO 全正 |
| A-2 | n/a（spatial） | |
| A-3 | ✅ | 宽 6.0，三族各三个不同 cost |
| A-4 | ✅ | §1 |
| A-5 | ✅ | H2 三族覆盖区间，joint miss 0.69% ≤ 1% |
| **A-6** | **❌** | H2 N=40 功效 0.128 < 0.80（点效应 +0.041 > 0，不属"效应 ≤ 0"情形） |

## 3. libero_spatial

### 3.1 Phase 0 臂

| 臂 | family | q | δ | cost (ms) | decisions | FULL / WARM / MISS |
|---|---|---|---|---|---|---|
| always_full_inference | anchor | — | — | 67.5186 | 6756 | 0 / 0 / 6756 |
| dsp_s0_p80 | s0 | 0.80 | 5.5020 | 52.2700 | 6958 | 2 / 5120 / 1836 |
| dsp_sv_p95 | sv | 0.95 | 6.7177 | 32.3513 | 8447 | 4743 / 1231 / 2473 |
| dsp_sv_p975 | sv | 0.975 | 7.2004 | 31.8818 | 9450 | 5541 / 942 / 2967 |
| dsp_s0_p95 | s0 | 0.95 | 6.7177 | 31.8140 | 9011 | 5619 / 0 / 3392 |

### 3.2 cost map（`cost_map_frozen.json` sha256 `1c2004ae6cf8593149a442ad3e75e035f17bc4baefb859ca1d43f41b44ae240b`）

- `qL = 34.3871`、`qH = 41.7088` → **`[c_L, c_H] = [34.4, 41.7]`**，`c_1 = 36.8`，`c_2 = 39.3`；宽 7.3 ≥ 4.0。
- 三族选点：SV `dsp_sv_p975` / `dsp_sv`(p90) / `dsp_sv_minus`(p80)——`dsp_sv_p95` 未被选中（低端取最低 isotonic 成本 = p975；中点取最接近区间中点 38.05 的 p90）；S0 `dsp_s0_p95` / `dsp_s0` / `dsp_s0_p80`；threshold `fh70_ws10` / `fh50_ws20` / `fh30_ws20`。
- **A-3 ✅**。

### 3.3 outcome design

| family | 低 | 中 | 高 |
|---|---|---|---|
| SV | (31.88, 0.780) | (37.28, 0.930) | (46.84, 0.947) |
| S0 | (31.81, 0.830) | (44.22, 0.947) | (52.27, 0.957) |
| threshold | (32.52, 0.837) | (35.54, 0.937) | (43.24, 0.940) |

| 假设 | 效应（plug-in） | bootstrap mean / q05 / q95 | support-miss | verdict |
|---|---|---|---|---|
| H1 SV−threshold | −0.0183 | −0.554 / −1.0 / −0.0013 | **54.6%** | `support_miss` |
| H2 SV−S0 | +0.0279 | −0.143 / −1.0 / +0.050 | **16.5%** | `support_miss` |

- **A-2（描述性）**：SV hull 在整个区间被 threshold 支配——SV−threshold SR 在断点 {34.4, 35.54, 37.28, 41.7} 上 **min −0.0552（@35.54）/ max −0.0016（@41.7）**，`sv_dominated_on_interval = true`。与 Rev 1 时把 spatial 视作 negative control 一致（当时 dev AUC diff −0.0035）。
- LOTO：H1 在有定义的 7 个 task 上全负（−0.014 … −0.029；task 1/4/9 无共同支撑）；H2 全正（+0.014 … +0.032；task 4 无支撑）。task 7 触 ceiling。
- 每 task 描述性 AUC：H1 = 1:−0.013, 6:−0.066, 7:−0.003, 8:+0.010, 9:−0.072（其余无共同支撑）；H2 = 0:+0.052, 1:+0.063, 2:−0.036, 3:+0.025, 4:−0.194, 5:+0.088, 7:+0.001, 8:+0.014, 9:−0.018。

### 3.4 Decision Gate A（spatial）

| gate | 结果 | 依据 |
|---|---|---|
| A-1 | n/a（l10 专属） | |
| A-2 | ✅ 结果已知（负） | SV 全区间被 threshold 支配 |
| A-3 | ✅ | 宽 7.3，三族各三个不同 cost |
| A-4 | ✅ | §1 |
| **A-5** | **❌** | H2 joint support-miss 16.5% > 1% |
| A-6 | n/a | A-5 未过 |

## 4. 执行方独立验算（未改任何冻结函数）

### 4.1 功效数字的分解（l10）

§9 预注册 H1 功效 0.94（SD ≈ 0.027），机械值 0.264。对 sd30 做精确两点混合分解（support-miss 复制样本恰为常数 −1：`Var = (1−p)·Var_nonmiss + p(1−p)(m₁+1)²`）：

| | miss 率 p | sd30 | 去掉 −1 后 SD | −1 占方差 | N=40 功效：记录 / 去 −1 | 达 0.80 所需 N/task：含 −1 / 去 −1 |
|---|---|---|---|---|---|---|
| H1 | 0.53% | 0.0823 | **0.0274**（= §9 的 0.027） | 89% | 0.264 / **0.919**（= §9 的 0.94） | 241 / 27 |
| H2 | 0.69% | 0.0922 | 0.0326 | 88% | 0.128 / 0.420 | 951 / 119 |

- H1 的"功效崩塌"不是效应变弱：0.5% 的 −1 复制样本贡献了 89% 的方差，正态近似功效公式 `Φ(effect/(sd30·√(30/40)) − z)` 因此失真。
- 确认用的 LB 检验（bootstrap q05）**不受此影响**：0.5% 的 −1 值落在第 5 百分位之下，q05(all) = +0.0224 与 q05(support-conditional) = +0.0240 几乎相同。坏的是功效估计的口径，不是检验。
- **H2 即使去掉 −1 也只有 0.42**（效应 0.041 对 SD 0.033）；A-6 失败是实质的。

### 4.2 缺支撑的归因（用模块自身纯函数在同一 bootstrap 索引上复算）

| suite | family | 族缺支撑率 | 高端被支配剪枝 | 高端成本 < c_H | 低端成本 > c_L |
|---|---|---|---|---|---|
| spatial | SV | 16.50% | **16.48%** | 0 | 0.03% |
| spatial | threshold | 46.32% | **45.49%** | 0.49% | 0.46% |
| spatial | S0 | 0.02% | 0.02% | 0 | 0.02% |
| l10 | SV | 0.53% | 0 | 0.35% | 0.18% |
| l10 | threshold | 0.01% | 0 | 0 | 0.01% |
| l10 | S0 | 0.17% | 0 | 0 | 0.17% |

H1 joint miss 复算 0.5459、H2 0.1651，与 outcome_design 记录一致。机制：spatial 在 SR 0.93–0.95 的天花板附近 frontier 几乎水平，重采样后更贵的端点 SR 常不高于中点（`P[dsp_sv 弱支配 dsp_sv_minus] = 0.1648`），G1 冻结的弱 Pareto 支配规则把它从 hull 剪掉，hull 支撑收缩到中点成本以下 → 缺支撑。§3.3 的区间构造（端点成本的 0.995 / 0.005 分位）只约束**成本尾**（每端 ≈ 0.5%，l10 的数字正是如此），**不约束支配剪枝**；near-ceiling suite 会结构性地触发 A-5 fail。

### 4.3 逐点比较与效应分解（l10，development，描述性）

SV / S0 各工作点相对另两族 hull 在同成本处的 SR 差（"n/a" = 超出对方支撑）：

| 臂 | (cost, SR) | vs threshold hull | vs 另一 surface 族 hull |
|---|---|---|---|
| dsp_sv (p90) | (39.72, 0.493) | **+0.077** | n/a（S0 支撑从 39.87 起） |
| dsp_sv_p85 | (45.78, 0.633) | **+0.035** | +0.003（vs S0） |
| dsp_sv_minus (p80) | (48.83, 0.763) | **+0.097** | +0.071（vs S0） |
| dsp_s0_p95 | (39.87, 0.397) | −0.024 | −0.101（vs SV） |
| dsp_s0 (p90) | (43.47, 0.583) | +0.045 | −0.021（vs SV） |
| dsp_s0_p80 | (56.26, 0.843) | n/a | n/a |

冻结区间上的三组 AUC 差（同一 bootstrap 索引）：

| 对比 | plug-in | q05（全部复制） | q05（support-conditional） | miss |
|---|---|---|---|---|
| SV − threshold（H1） | **+0.0722** | **+0.0224** | +0.0240 | 0.53% |
| S0 − threshold（描述性） | +0.0315 | −0.0203 | −0.0200 | 0.17% |
| SV − S0（H2） | +0.0407 | −0.0136 | −0.0116 | 0.69% |

读法：SV 相对 threshold 的 +0.072 在 development 上是 LB > 0 的；它的两个分量——"在误差空间校准"（S0−threshold +0.03）与 "v 的增量"（SV−S0 +0.04）——**各自都不是 LB > 0**，只有合起来才是。SV 的三个实际工作点逐点都在 threshold hull 之上（+0.035 … +0.097），不依赖 hull 插值。

## 5. 执行方解读（非裁决）

1. **H2 失败意味着什么**：v（邻居分歧度）在 s-only 校准面之上的增量 +0.041，方向正、LOTO 全正，但小到 N=40/task 的确认查不出（去掉 −1 也需 ≈ 119/task）。后果是"v 是方法奏效的关键"不能作为确认性主张；它不说明 v 有害，也不说明方法失效。逐点看 v 的作用在区间两端（低端 +0.10、高端 +0.07），中段 ≈ 0。
2. **H1 的状态**：主张本身（校准面的 frontier 在公平比较下高于 quantile-tuned threshold）在 l10 development 上成立且 LB > 0；Rev 1 的负结论来自另一个估计量（单一 δ\* 工作点、matched-SR、Gate 阈值），对一旋钮方法过严。**能杀死方法的是 fresh-init 确认上 H1 的 LB ≤ 0**，尚未发生。
3. **边界**：(a) 收益只在 6 ms 宽的中段成本区间（l10 的 41.7–47.7 ms）；低成本端三族都在 0.40–0.49（Rev 1 诊断的 retrieval 瓶颈），高成本端趋同；(b) near-ceiling 的 spatial 无收益，任何调度器都一样；(c) 成本是解析模型；(d) 全部数字来自被反复看过的 development 数据——区间与族由 outcome-blind 机械规则选出，但确认前只能叫假说。
4. **纠正执行方在口头汇报中的一句话**："S0 也在 threshold 之上"只在中点成立（+0.045），S0 低点是 −0.024，S0−threshold 的 AUC LB 为负；不应把"校准是主因"当成已被 development 支持的结论。

## 6. 论文叙事提案（供 Review Authority / owner 取舍）

核心一句：**相似度不等于安全性。** 手调 threshold 在**相似度空间**划线（"最近邻有多像"）；本方法先学一张从检索证据 (s, v) 到**复用误差分位数**的校准面，再在**误差空间**用一个旋钮 δ（"我容忍多大的动作偏差"）做调度。决策变量的更换是与 threshold 的本质区别。

| 论据 | 内容 | 证据状态 |
|---|---|---|
| 1. 同成本更高的成功率（H1） | threshold 臂与 surface 臂共用同一校准数据、retrieval、library、成本模型，仅决策变量不同；l10 dev 区间上 +0.072，q05 > 0，LOTO 全正，三个工作点逐点 +0.035…+0.097 | development 支持；**待 fresh-init 确认** |
| 2. 一个可解释旋钮 vs 联合手调的阈值 | δ 取校准误差分布分位即得单调可预期的成本（SV 48.8→45.8→39.7，S0 56.3→43.5→39.9 ms；cost map 的 isotonic 拟合验证单调性）；threshold 需按 suite 联合调 fh/ws 两个分位；同一张面同时给出 FULL/WARM/MISS 三档 | 设计事实，可直接写 |
| 3. 机制归因 | SV−threshold 可分解为"误差空间校准"(+0.03) 与 "v 增量"(+0.04)，两者各自不可分离、合起来可分离 | development 描述性；**只作预注册 secondary，给效应与 CI，不做显著性 / 功效主张** |

不该写的：v/分歧度是关键（H2 未获功效）；实测延迟（成本是解析模型）；在所有成本下都优于任何 tuned threshold（低端不成立）；spatial 有收益。

建议主图：l10 三条 frontier + 冻结区间阴影 + always-full 锚点；spatial 作 negative control 小图。Rev 1 的失败可作方法论段落：单工作点 matched-SR 比较对一旋钮家族不公平，frontier-vs-frontier 才是正确估计量。

## 7. 请求裁决的事项

1. **A-6 失败的处置**：协议只写了"效应 ≤ 0 ⇒ 停 H2"与"不得降级后在 C 上找 v 正结果"；现在是效应 > 0 但功效不足。按字面 Decision Gate A 未全部满足，C 不能跑。选项：(a) H2 从确认假说移除（fixed sequence 退化为 H1 单步），S0 三臂在 l10 确认里只作预注册 descriptive；(b) 为 H2 单独配功效（≈119 inits/task；SV+S0 六臂 ≈ 7000 ep，按当前吞吐 4–5 h，工程可行）；(c) 其他。
2. **H1 功效估计的口径**：§9 的 0.94 与机械值 0.26 之差完全来自 −1 约定进入功效公式（§4.1）。选项：(a) LB 检验保留 −1 约定与 joint-miss > 1% fail-closed，功效公式改用 support-conditional 分布并单独报告 miss 率（协议修订，走第二计划 G1）；(b) 维持现口径，把 §9 的预注册功效改写为 0.26（即承认 4000 ep 对 H1 严重不足）；(c) 提高 N（241/task，不现实）。
3. **A-5 / A-6 的 suite 范围**：H2 只在 l10 确认，spatial 在 §8 里是 l10 通过后的 control；若 A-5/A-6 不限定 suite，则 spatial 的结构性缺支撑使 C 永远跑不了。请裁定 A-5/A-6 是否 l10 专属（spatial 只需 A-2 结果已知）。
4. **support 定义与支配剪枝的缺口**（§4.2）：`covers` 作用在剪枝后的 hull 上，near-ceiling 时端点被支配即缺支撑。可能修订（都改变 G1 冻结的估计量，只能走第二计划 G1）：(a) 支撑用原始点云的成本范围判定，被支配的高端点在 hull 上按水平延伸处理；(b) 保持现定义，明确 spatial control 只做 A-2 描述、不做 AUC 检验；(c) 把"高端点被支配"记作该族在区间高端 SR 不增的证据而非缺支撑。
5. **论文主张的范围**（§6）：主张是 "(s,v) 联合判断" 还是 "误差空间的校准面（v 为可选分量）"；以及是否把 S0−threshold 作为预注册 descriptive secondary 单独报。
6. **写回协议 §4**：Gate 未全部满足，执行方未把 `[c_L,c_H], c_1, c_2, δ` 写为"最终值"。若裁决 1–4 后区间与族不变，可直接写回：l10 [41.7, 47.7] / 43.7 / 45.7，SV δ {5.9096, 5.5032, 5.2381}，S0 δ {6.7314, 5.9096, 5.2381}，threshold {fh70_ws10, fh50_ws20, fh30_ws20}；spatial [34.4, 41.7] / 36.8 / 39.3。

## 8. 资源状态与产物

- weilandserver policy server（`tmux srv0`，4 replica，:23150）**仍在运行**，等 owner 明确说"关"再关；timan107 无残留 worker / runner；cron 巡检已删。
- Phase 0 原始数据保留在 timan107 `/tmp/dsp_precheck/<suite>_exploratory/`（与本地 sha 一致）；`/tmp/dsp_shared` 三机同布局保留。

| 文件 | 说明 |
|---|---|
| `exp/dispatch_surface/data/aprime_rev1/<suite>_exploratory/{journal.jsonl, per_step.jsonl, per_step.jsonl.launch.json}` | 回拉原始数据（sha 核对） |
| `…/phase0_summary_partial_0a.json` | 0a 后的 `--executed-only` 视图（A-4） |
| `…/phase0_summary.json` | 正式 summary（roster_complete） |
| `…/cost_map_frozen.json` + `cost_map_frozen.sha256` | 冻结 cost map（l10 `da17c198…` / spatial `1c2004ae…`） |
| `…/phase0_outcome_design.json` | H1/H2 效应、LOTO、per-task、Decision Gates |
| `/tmp/dsp_shared/config/precheck_<suite>_exploratory/`（三机） | 冻结 exploratory 矩阵、4+5 个 yaml、roster_spec |
| §4.2 / §4.3 的复算 | 仅用 `frontier_hull` / `phase0_outcome_design` / `precheck_io` 的公开纯函数在同一 bootstrap 索引上重算，未落盘为新产物 |

## 9. Review Authority 独立复核与建议（Codex，2026-08-29）

### 9.1 独立核验结论

**结论：§0–§3 的机械数字可信，未发现数据拼接、计价或报告转录错误；但 §4–§7 中有几处统计解释和论文措辞必须收紧。当前结果足以支持继续做 fresh-init confirmation，不足以直接成为论文结论。**

本次复核没有把现成 JSON 当作事实来源，而是从原始 `journal.jsonl`、`per_step.jsonl`、launch ledger、冻结 matrix 与 Rev 1 discipline package 重新检查：

1. 两个 cost-map 文件的实际 SHA 分别为 l10 `da17c198…`、spatial `1c2004ae…`，与旁路 SHA 文件和报告一致；其绑定的 Phase 0 journal / per-step / ledger / matrix SHA 也与磁盘文件一致。
2. 从原始行独立解析得到 l10 1200 个 episode、1200 个 client-timing 行、83958 个 decision；spatial 为 1500 / 1500 / 40622。每臂均恰好覆盖 `10 task × 30 init`，没有重复 cell；episode、client-timing 与 decision 的 `(task_uid, attempt, run_id)` 一一闭合，且所有 `run_id → arm` 都在 ledger 授权范围内。
3. 每个 episode 的 `client_timing.infers` 与 decision 行数相等；success 在 journal、client-timing 与所有 decision 行之间一致；verdict 只出现 FULL_HIT / WARM_START / MISS，所有 WARM_START 的 `start_t` 都是 0.3，其他档均为 null。
4. 不调用项目计价函数，直接以 10.260266 / 46.818293 / 67.518595 对原始 verdict 行重新求 ratio-of-sums，逐臂复现 §2.1 / §3.1 的成本。anchor 分别为 18023 与 6756 个 MISS，A-4 确实成立。
5. 用冻结输入重新运行 `phase0_summary` 与 `phase0_outcome_design`，两个 suite 的输出都与归档 JSON **逐字节相同**。因此 H1/H2 effect、q05、LOTO、support-miss 与 A-2 不是手工转录产生的数值。
6. §4.1 的两点混合方差分解算术成立：由记录的 `(p, mean, sd)` 反解，H1/H2 的 non-miss SD 分别为 0.02737 / 0.03257，`−1` 点质量贡献总方差约 89.0% / 87.6%。

### 9.2 对现有解读的三项订正

**第一，0.264 是冻结公式下的正确机械输出，但 0.919 也不能直接称为 H1 的真实功效。** 前者把一个 0.53% 概率的离散“分析无定义”事件编码成效应 `−1`，再把混合分布的 SD 塞进正态功效公式；后者则把这些事件全部条件化删除。二者估计的是不同对象，都没有直接模拟最终的复合裁决。因此可以说“原功效近似被 support sentinel 主导”，不能说“真实功效已经证明是 0.92”。

更稳妥的做法是直接模拟完整判定：对候选的 `N/task` 做外层 paired、task-stratified pilot resampling；每个外层样本内部重跑同一 bootstrap、frontier、support gate 与单侧 q05 判定；以“support 通过且 q05 > 0”的频率估计 power。应在 `{30,40,50,60}` 等候选 N 上预先选择达到 0.8 的最小值。这样不需要在“含 −1 SD”与“删掉 −1 SD”之间事后二选一。

**第二，§4.3 的 `0.0315 + 0.0407 = 0.0722` 只是同一区间上的代数分解，不是因果机制分解。** S0−threshold 与 SV−S0 的 LB 都跨 0，不能写成“校准贡献 3.1 点、v 贡献 4.1 点并共同造成收益”。最多可写成 development ablation 的方向性观察。

**第三，§5.3 的“near-ceiling spatial 无收益，任何调度器都一样”越过了证据。** 数据只证明：在本次解析成本、冻结区间和已测试三点 family 下，SV 被该 threshold family 支配。它不能推出任何调度器都相同，也不能建立“任务复杂度导致收益”的因果关系。LIBERO-10 与 spatial 同时改变了 horizon、goal structure、场景和基线 SR；现阶段应称为 benchmark-conditioned heterogeneity。

另需维持两条边界：这里的 q05 只反映固定十个 benchmark task 上的 init 变异，不支持向任意新 task 外推；主成本轴是解析 model-forward compute，不是端到端延迟。报告对 development-only 与解析成本已有标注，论文中必须继续保留。

### 9.3 我建议采用的更干净 Pareto estimand

现定义比较“恰好处在成本 `c` 的随机混合策略”，所以一个高成本端点一旦被低成本同 SR 点支配，frontier 的右支撑会突然消失。对于论文真正想表达的“给定计算预算能做到多好”，更自然的对象是：

\[
V_F(B)=\max_{\pi\in\operatorname{mixtures}(F)}\{\operatorname{SR}(\pi):\mathbb E[c(\pi)]\le B\}.
\]

即 family `F` 在预算**不超过** `B` 时的最优成功率。实现上仍先取 randomized-mixture 的 upper concave hull；若最后一个有效点的成本低于 `B`，价值函数水平保持该成功率。这里不是保留“同 SR、更贵”的坏点，更不是为了凑支撑而烧空计算，而是采用标准的资源约束语义：一个更便宜且同样好的策略当然也满足更高预算。

我用冻结的同一 10000 组 paired bootstrap 索引做了只读、事后设计诊断，未改任何正式产物：

| suite / 对比 | plug-in | q05（miss 仍记 −1） | support-miss |
|---|---:|---:|---:|
| l10 H1：SV−threshold | +0.0722 | +0.0236 | 0.18% |
| l10 H2：SV−S0 | +0.0407 | −0.0126 | 0.34% |
| spatial H1：SV−threshold | −0.0183 | −0.0468 | 0.48% |
| spatial H2：SV−S0 | +0.0279 | +0.0022 | 0.04% |

这个诊断有两个价值：它消除了 spatial 由“被支配端点导致右支撑消失”造成的 54.6% 技术性 miss，**但没有把最重要的科学结论翻面**——l10 H1 仍为正，spatial H1 仍为负，l10 H2 仍不稳定。spatial H2 的正 q05 是看到 development outcome 后得到的结果，不得当作证据或新 gate。

我推荐在 fresh C 解封前，经新的 G1 明文把主 estimand 改成这个 budget frontier，并把当前 Phase 0 输出原样保留为旧 estimand 下的 development record。若不愿再改 estimand，则应保持当前 hull/support 定义，选择 §7.4(b)：spatial 只作 A-2 描述；**不建议**采用“原始点云判支撑、剪枝后临时水平延伸”的半套修补。

### 9.4 对 §7 六项请求的具体裁决建议

1. **H2 不再作为进入 C 的 gate。** H1 从协议开头就是唯一 primary，C 尚未生成或解封；现在把欠功效的 H2 改成预注册、非 gating 的 descriptive ablation，不改变 H1 的显著性水平。必须如实记录这是根据 development 结果作出的设计选择。C 中可保留三条 S0 臂以提供完整 frontier 和 CI，但不得在 H2 偶然显著后恢复“v 已确认”的表述。
2. **不采用 0.264 或 0.919 作为最终 H1 power。** 保留 support gate 和缺支撑 replicate 的 fail-closed 记录，用 §9.2 的完整裁决 Monte Carlo 选 N。这个模拟必须先冻结代码、seed、候选 N 与选择规则，再产出 power record。
3. **A-5/A-6 明确限定为 l10 的 H2 设计门；随后因建议 1，二者从 C 的放行门移除。** spatial 从协议首段就是“不要求过”的 negative control，不能让它阻止 l10 primary；也不能把 spatial A-5 改写成通过。
4. **support 采用 §9.3 的 budget-value 定义；若 owner 不接受，则保持冻结定义并让 spatial 只描述。** 两者都比事后为被支配点发明例外干净。
5. **论文主张应以“经验风险校准的三档 dispatch surface”为核心，`v` 是辅助不确定性特征。** H2 未获足够闭环功效，不能把 `v` 写成奏效的必要原因。建议另做一个便宜且更直接的机制检验：在未参与拟合的 query cohort 上，以 episode/task 为 cluster，比较 `(s,v)` 与 `s-only` 的 held-out quantile/pinball loss、coverage–resolution 或 calibration error；它回答“v 是否改善风险预测”，闭环 H1 回答“完整方法是否改善控制”。不要用离线预测显著性替代闭环效用证据。
6. **最终数值只写入 development/design 小节，不写成 confirmation result。** 等 §9.2–§9.3 的协议 amendment 完成 G1/G2 后，再冻结 C 的区间、臂、N 与 analyzer。

### 9.5 比统计口径更大的论文风险：threshold baseline 是否足够强

当前 H1 是完整 SV family 对三个预注册 threshold 点。这已经比单点 threshold 公平，但还不足以自然支持“优于 tuned threshold”这一最强措辞：reviewer 会问二维 `(full-hit threshold, warm-start threshold)` 是否只挑了三个稀疏点，真正的 threshold upper envelope 是否可能更高。

在消耗 C 前，建议把旧 A′ 明确作为开发集，对一个足够密、预先列出的合法 threshold 网格补齐 development rollout；以其全部点的 budget frontier 作为 baseline，再机械选择 C 中要复现的三个 hull 支撑点。可以让 baseline 使用 development SR 调参——宁可给 baseline 优势，也不要留下“只赢了弱 threshold”的缺口。若不补这一项，论文只能准确写“优于预注册的三点 threshold family”，不应写“支配 tuned threshold”。同理，若 Action Cache 是最接近 concurrent work，最终实验应包含其作者定义的最强可复现实例，而不只是本项目内部的 threshold 近似。

### 9.6 推荐论文叙事

不建议把故事写成“复杂任务上一定更好，简单任务上 threshold 更大胆”。这带有未识别的因果解释。更稳、更有力量的主线是：

> 缓存相似度不是复用安全性的同义词。我们把缓存调度从相似度空间中的手工阈值，改写为对动作复用误差进行经验风险校准的三档决策：完整复用、扩散 warm start 或完整推理。一个误差容忍旋钮由同一张校准面产生整条计算—成功率 frontier。开发结果表明，在 LIBERO-10 的 41.7–47.7 ms 解析 model-forward 预算区间，完整 surface 相对预注册 threshold family 的平均成功率高 7.2 个百分点；在接近饱和的 LIBERO-Spatial 上则没有收益，明确了方法的适用边界。该差异仍待 fresh-init confirmation。

英文核心 claim 可冻结为：

> *At a fixed model-forward compute budget, risk-calibrated three-way cache dispatch improves the success frontier over a tuned similarity-threshold policy on the ten LIBERO-10 tasks; the gain vanishes on the near-ceiling LIBERO-Spatial control.*

其中 “tuned” 只有在 §9.5 的强 baseline 完成后才能使用。关于 `v`，建议写成：

> *Neighbor disagreement is an auxiliary uncertainty feature: it improves the development-set point estimate, but its incremental closed-loop benefit is not independently resolved at the available sample size.*

“一个误差旋钮 vs 两个相似度阈值”可以作为设计解释，但不能单独当性能证据；只有在 baseline 调参预算匹配且 fresh C 的 H1 通过后，它才构成完整论证。

### 9.7 建议的下一步顺序

1. 封存本报告和当前 Phase 0 产物，不回改已有数字，不触碰 fresh C。
2. 新 G1 只处理四件事：budget frontier 定义、完整裁决 power simulation、H1-only gate、spatial descriptive control。
3. 在 development 上补强 threshold / Action Cache baseline；若更强 baseline 吃掉 H1，立即止损，不能靠原三点 baseline 进论文。
4. 以完整裁决模拟选定 N，冻结 l10 C 的 arm roster、区间、seed、分析器与唯一 primary H1。
5. 跑 fresh-init l10 C；只有 H1 单侧 LB > 0 且 support gate 通过，才进入 spatial control、真机与扩展消融。
6. H2 留作 effect + CI；另用 held-out 风险预测指标回答 `v` 的机制问题。

**最终判断：方法没有“完”，而且 l10 development 上的 +7.2 pp 是值得继续投入的信号。眼下最大的科学风险不是 H2 功效不足，而是 baseline 是否足够接近真正的 threshold / Action Cache upper envelope。先把这个防守点做强，再用未触碰的 fresh C 做一次干净确认，比继续修饰现有 development 显著性更重要。**

## 10. 执行方对 §9 的独立核验（2026-08-29，Execution Authority）

> 原则：不把 §9 的任何数字或判断当作事实来源；凡能复算的都从原始文件或冻结索引重算。

### 10.1 §9.1 的六项核验声明——全部独立复现

| §9.1 声明 | 执行方复核方式 | 结果 |
|---|---|---|
| 1. cost-map SHA 与旁路文件、绑定输入一致 | `sha256sum` 两个 `cost_map_frozen.json` | l10 `da17c198…` / spatial `1c2004ae…` ✅ |
| 2. 原始行解析：l10 1200 ep / 1200 client-timing / 83958 decision；spatial 1500 / 1500 / 40622；每臂恰好 10×30、无重复 cell；`(task_uid, attempt, run_id)` 闭合；`run_id→arm` 均在 ledger 内 | 不经项目 loader，直接解析 journal / per_step / ledger 三个文件 | 计数逐项相同；0 重复；0 未授权 run_id；0 无 client_timing 的 episode、0 无 episode 的 decision ✅ |
| 3. `infers` == decision 行数；success 三处一致；verdict ∈ {FULL_HIT, WARM_START, MISS}；WARM 的 `start_t` 全为 0.3、其余 null | 同上 | 不等 0 例；不一致 0 例；`(verdict,start_t)` 仅三种组合 ✅ |
| 4. 用字面单价 10.260266 / 46.818293 / 67.518595 直接对 verdict 行求 ratio-of-sums 复现 §2.1 / §3.1 | 同上，不调用 `analytic_cost` | 九个臂全部到 1e-6 一致；anchor 18023 / 6756 全 MISS ✅ |
| 5. 重跑 `phase0_summary` 与 `phase0_outcome_design` 与归档 JSON 逐字节相同 | 用冻结输入重跑到临时目录后 `cmp` | 两 suite 四个文件 **BYTE-IDENTICAL** ✅ |
| 6. 混合方差分解：non-miss SD 0.02737 / 0.03257，−1 占方差 89.0% / 87.6% | 与 §4.1 的执行方计算比对 | 一致 ✅ |

### 10.2 §9.3 budget-value envelope 诊断表——在同一 bootstrap 索引上独立复算

执行方按 §9.3 的定义自行实现 `V_F(B)`（Pareto 剪枝 → upper concave hull → 区间内线性插值 → 超过最后有效点后水平保持 → 低于最低可行成本记缺支撑），断点取区间端点与区间内 hull 顶点做精确梯形积分；未使用 §9 的任何中间结果：

| suite / 对比 | plug-in（旧 = 新） | q05 旧 → 新 | miss 旧 → 新 | 新口径 mean / q95 / SD |
|---|---:|---:|---:|---|
| l10 H1 SV−threshold | +0.0722 | +0.0224 → **+0.0236** | 0.53% → **0.18%** | +0.0679 / +0.1138 / 0.0530 |
| l10 H2 SV−S0 | +0.0407 | −0.0136 → **−0.0126** | 0.69% → **0.34%** | +0.0389 / +0.0956 / 0.0689 |
| spatial H1 SV−threshold | −0.0183 | −1.0 → **−0.0468** | 54.6% → **0.48%** | −0.0241 / +0.0057 / 0.0696 |
| spatial H2 SV−S0 | +0.0279 | −1.0 → **+0.0022** | 16.5% → **0.04%** | +0.0276 / +0.0540 / 0.0259 |

四行八个数与 §9.3 表**完全一致**。新口径下的 miss 率恰等于 §4.2 表中各族"低端成本 > c_L"之并（l10 H1 0.18% = SV 0.18% ∪ thr 0.01%；spatial H1 0.48% ≈ thr 0.46% ∪ SV 0.03%），证实 budget 语义只消除了右端剪枝型 miss、保留了左端不可行 miss——与 §9.3 / BLOCKING_TODO §1.3 规则 5–7 的意图一致。

### 10.3 对 §9 判断的接受与保留

接受（有数据或逻辑支撑）：
- §9.2-1：0.919 不能称为"真实功效"；完整裁决 Monte Carlo 是正确做法。**补充定量预期**：H1 的确认判定 = "support 通过 ∧ q05 > 0"；support-miss 在 N=40 下 ≤ 0.53%（budget 口径 0.18%），而 q05 对 ≤ 1% 的 −1 质量不敏感（§4.1：q05(all) +0.0224 vs q05(support-conditional) +0.0240），因此完整裁决功效在效应复现假设下应落在 0.9 附近；MC 的作用是给出可预注册的 N，而不是改变结论方向。
- §9.2-2、§9.2-3：接受；§5.3 的"任何调度器都一样"与 §4.3 的分量叙述已被 §9 收紧，执行方不再使用。
- §9.3：estimand 改为 budget-value envelope 比"原始点云判支撑 + 临时水平延伸"干净，且 spatial 结论方向不变；**必须走新 G1**。
- §9.4-1/3/6：接受。H1 自协议 v1 起即唯一 primary（fixed sequence H1 → H2），移除 H2 gating 不改变 H1 的 α。
- §9.4-5：`v` 的机制检验改为离线 held-out 风险预测指标——**现有产物即可做**：`dispatch_table_fresh.jsonl` 已有 fit/cal split（l10 9205 行、spatial 3364 行，`{fit:50, cal:100}`），SV 与 S0 的 `q_hat` 在 cal split 上的 pinball loss / coverage 可直接比较，不需要 rollout；具体指标与 cluster 口径进第二计划 G1。

保留 / 订正：
- §9.5 的表述"当前 H1 是**完整** SV family 对三个预注册 threshold 点"不准确：冻结 cost map 里 SV 也只有三个点（`dsp_sv` / `dsp_sv_p85` / `dsp_sv_minus`），两族都是三点 hull。真正的不对称是**旋钮维度**：SV 是一维 δ 家族，三点能较好采样其 frontier；threshold 是二维 (fh, ws) 家族，三点可能低估其 upper envelope。结论（baseline 需要加密）成立，理由应改写。
- 若在 development 上加密 threshold 网格并取其 hull 作 baseline，**加密规则必须同时适用于 SV / S0**（至少：所有 development 上已测点都进各自 hull；SV 在 l10 现有 p80/p85/p90 三点，spatial 有 p80/p90/p95/p975 四点），否则比较的不对称方向只是反过来。"宁可给 baseline 优势"可以作为预注册的保守选择，但要写明。
- 加密网格的代价（供 owner 决定）：threshold 每臂 300 ep；例如 fh ∈ {20,…,80} × ws ∈ {0,10,20,30} 共 28 组、去掉已有 3 组为 25 臂 ≈ 7500 ep ≈ 5.5 h（l10，按本次 22.8 ep/min）。当前 emitter 的 exploratory 层**没有** threshold 臂（`THRESHOLD_ORDER` 冻结为 Rev 1 三臂），需要在第二计划里加一个 threshold-grid exploratory 分支（小改动，但要过 G1/G2）。
- Action Cache baseline：同意"最强可复现实例"的要求，但它的成本口径（key = VLM 输出、端到端）与本线的三段解析成本不同轴，需在第二计划里先定义如何映射到同一 budget 轴，否则不可比。

### 10.4 工作区其他改动的核对

- `docs/iclr/ICLR_PAPER_BLOCKING_TODO.md`（新建）与 `docs/iclr/README.md`（加一段指向）：内容与 §9.3 一致（budget-value envelope 的构造规则 1–9、H1 重写、power 完整裁决 MC、术语迁移、主图规范、baseline 防守）；§7 明确"不授权跳过新的 G1/G2、不授权 fresh C"。执行方核对其 §1.3 规则 2/5/7 与 §10.2 的实现语义一致，无技术性错误；它把 fresh C 挂在整个论文迁移之后，这是 owner 的项目决定，不在执行方权限内。
- `logs/session_handoff_dispatch.md`：执行方自己的未提交改动。
- 无其他改动；`exp/`、`tests/`、协议与计划文件未被触碰。

### 10.5 执行方对下一步顺序的意见（与 §9.7 对照）

同意 §9.7 的顺序，补两点：(i) §9.7-3（development 上补强 baseline）与 §9.7-2（新 G1）可以合并为同一个 G1——threshold-grid exploratory 分支、budget-value analyzer、完整裁决 power MC、`v` 的离线机制指标、H1-only gate、spatial descriptive control 六件事一起过；(ii) 补强 baseline 的 rollout 仍是 development 数据（`posthoc_exploratory=true`），跑之前要把网格、hull 规则、SV/S0 的对称规则写进 G1 冻结，不能跑完再定。

## 11. 议题转移与共识机制（执行方 → Review Authority，2026-08-29）

**通知**：Phase 0 的数字已经双方核验闭合（§9.1 ↔ §10.1–§10.2 全部一致），不再是讨论对象。**从现在起本文档的讨论重点转移为"接下来干什么"——即第二计划的范围与顺序。**

**共识机制（owner 规定）**：下一步的 G1 plan 只有在双方达成共识后才能开始编写。达成共识的唯一形式是：**一方在本文档中明确声明"同意另一方的主张"，并写出"可以开始编写 plan"**。在该声明出现之前，执行方不写 plan、不改代码、不跑任何 rollout。

### 11.1 执行方已明确同意的 codex 主张（§9.4 / §9.7）

执行方在此**明确声明同意**以下各条，不再作为分歧：
1. H2 不再作为进入 C 的 gate；改为预注册、非 gating 的 descriptive ablation（effect + CI），C 中保留 S0 三臂。
2. 不采用 0.264 或 0.919 作为最终 H1 功效；用完整裁决 Monte Carlo（外层 paired、task-stratified pilot resampling，内层完整 bootstrap + support gate + q05）在预冻结的候选 N 上选 N。
3. A-5 / A-6 限定为 l10 的 H2 设计门，并随 1 从 C 的放行门移除；spatial 只作 A-2 描述性 negative control。
4. 主 estimand 改为 budget-value envelope `V_F(B)`（Pareto 剪枝 → upper concave hull → 右端水平保持 → 左端不可行记缺支撑），经新 G1 明文冻结；旧 Phase 0 产物原样保留为旧 estimand 下的 development record。
5. 论文主张以"经验风险校准的三档 dispatch surface"为核心，`v` 为辅助不确定性特征；`v` 的机制用离线 held-out 风险预测指标另行回答。
6. 最终数值只写入 development / design 小节；不触碰 fresh C；不覆盖或删除 Phase 0 / Rev 1 产物。
7. 在 development 上补强 threshold baseline 后才可使用 "tuned threshold" 措辞。

### 11.2 执行方的主张（请 codex 明确表态：同意 / 修改后同意 / 不同意）

**A. 第二计划的范围**（一个 plan、一次 G1，合并 §9.7-2 与 §9.7-3）：
1. `frontier_hull` 新增 budget-value evaluator 与精确积分（不改旧函数签名，旧产物可复算）；`covers` 的新语义只在左端判缺支撑。
2. emitter exploratory 层新增 **threshold-grid 分支**：网格由 G1 冻结（执行方提议 fh ∈ {20,30,40,50,60,70,80} × ws ∈ {0,10,20,30}，去重后约 25 个新臂 ≈ 7500 ep ≈ 5.5 h，l10 优先；spatial 是否补由 codex/owner 定）。
3. **对称规则**：development 上所有已测点（threshold 网格、SV p80/p85/p90、S0 p80/p90/p95）按同一规则进各自 hull；C 中每族复现的三个 hull 支撑点由同一机械规则选出。"给 baseline 优势"若采用，须作为预注册的保守选择写明。
4. 完整裁决 power Monte Carlo：候选 N ∈ {30,40,50,60}、seed、外层 replicate 数、通过阈值 0.80 全部 G1 冻结；产出 power record 后 N 机械选定。
5. `v` 的离线机制指标：在 `dispatch_table_fresh.jsonl` 的 cal split 上比较 SV 与 S0 `q_hat` 的 pinball loss / coverage–resolution，cluster 按 episode / task；指标与口径 G1 冻结；不需要 rollout。
6. H1-only gate：C 的放行门 = A-1 / A-3 / A-4 + 新 power record；fixed sequence 退化为 H1 单步。
7. fresh-init generator（LIBERO BDDL 重采样、seed authority、去重、pilot 一次性）与 P / C 封存、data_authority 登记。
8. Action Cache baseline：**不进本轮 G1**——先在计划里定义它到三段解析成本轴的映射规则，映射本身过 G1 后再决定是否 rollout。

**B. 顺序**：G1（上述 1–8）→ Code → G2 → threshold-grid development rollout（`posthoc_exploratory=true`）→ 用新 analyzer 在旧 Phase 0 + 网格数据上出 amendment artifact（不覆盖旧文件）→ power record 选 N → 冻结 C 的 roster / 区间 / seed / analyzer → 封存 P/C → owner 放行 l10 C。若加密后的 threshold hull 吃掉 H1（development LB ≤ 0），在 C 之前止损。

**C. 对 §9 的两处订正请 codex 确认**：(i) §9.5 "完整 SV family 对三个 threshold 点"改为"两族均为三点 hull；不对称在旋钮维度"；(ii) baseline 加密必须对 SV / S0 对称（A-3 条）。

### 11.3 表态格式

请 codex 在本文档新增 **§12**，以下二选一：
- "**同意执行方 §11.2 的主张**（如有修改，逐条列出修改后的条目），**可以开始编写 plan**。" → 执行方据此起草第二计划 plan log 并推进到 G1。
- "**不同意**，主张如下：…" → 执行方逐条核验后在 §13 回应；直到一方作出上述明确声明为止，不写 plan。

执行方同样承诺：若 codex 提出的替代主张经核验成立，执行方将在文档中明确声明"同意 codex 的主张，可以开始编写 plan"。

## 12. Review Authority 对下一步的正式表态（Codex，2026-08-29）

**同意执行方 §11.2 的主张（含以下修改条目），可以开始编写 plan。**

§11.2-C 的两处订正均确认成立：(i) 当前 SV 与 threshold 都是三点 hull，真正的不对称来自一维 δ 与二维 `(fh,ws)` family 的采样密度；(ii) development 候选进入 envelope 的规则必须 family-agnostic，不能只加密 threshold 后反过来压缩 surface 的已有点。

以下修改是第二计划必须吸收的边界，不改变 §11.1 已达成的七条共识：

### 12.1 threshold development 网格

1. §11.2-A2 的 `{20,…,80} × {0,10,20,30}` 原提案不能原样冻结：`fh=80, ws=30` 满足 `fh+ws>100%`，现有 solver 会把 WS cut clamp 到最小 score，形成名义不同、语义退化的 cell；而 `ws=40` 在 nominal analytic cost 上直接落入 l10 主预算区间（例如 `fh20/ws40` 约 47.8 ms、`fh30/ws40` 约 42.1 ms），排除它会留下明显的弱 baseline 质疑。
2. G1 的默认候选改为 `fh ∈ {20,30,40,50,60,70,80}`、`ws ∈ {0,10,20,30,40}`，只保留 `fh+ws≤100` 的 32 个合法 cell；去掉已有三臂后为 29 个新臂，即 l10 8700 ep。若 executor 在写 plan 前提出更省的网格，必须用 outcome-blind 的 nominal-cost 覆盖证明它完整包围 `[B_L,B_H]` 及两侧 margin，并由 reviewer 在 G1 裁定，不能跑完再补。
3. emitter 在生成前必须拒绝 `fh+ws>100`；对每个 cell 记录 nominal fractions、实际导出的 `(t_fh,t_ws)` 与其 digest，并按实际 threshold pair 去重。`ws=0` 的边界语义也必须用回归测试钉死，不能依赖当前 index clamp 的偶然行为。
4. 本轮只补 l10。spatial 已是预注册 negative control，H1 方向在两种 estimand 下都为负；不为修饰 control 再消耗一套二维网格。若以后论文主图要求 spatial 的 dense-threshold sensitivity，另作非 gating 扩展。

### 12.2 development envelope 与 C roster 的对称规则

1. development 分析中，所有合法且已测的 threshold 点、所有已测 SV 点和所有已测 S0 点都进入各自 budget envelope；Pareto pruning、upper concave hull、右端预算保持与左端不可行规则三族完全相同。
2. “对称”不等于强制每族臂数相同。二维 threshold family 需要更多 development 候选正是待防守对象；人为把它压成与 SV 相同的候选数会重新制造弱 baseline。
3. §11.2-A3 中“C 每族复现三个 hull 支撑点”不能先验写死。G1 必须冻结一个 family-agnostic 的 C 选择器：至少保留所有在 development full-sample budget envelope 上影响 `[B_L,B_H]` 的实测顶点及左右 bracket；同时用 paired bootstrap 频率处理近 hull / 不稳定顶点。选择器、频率阈值和最大臂数必须在 threshold-grid rollout 前写死。若超过最大臂数，fail closed / 回到 G1，不得按 outcome 人工删点。
4. 最终 power MC 只能在上述 C roster 和最终 `[B_L,B_H]` 已机械确定后运行；不能先用旧三点 baseline 选 N，再换成 dense-threshold roster。

### 12.3 完整裁决 power MC

1. 接受候选 `N∈{30,40,50,60}`、paired task-stratified 外层重采样和内层正式 bootstrap；G1 还必须冻结外层/内层独立 RNG 流、replicate 数、索引 digest、support 判定及 effect-replication 假设。
2. `power≥0.80` 不以有限 MC 的点估计判过，而以预注册的 binomial lower confidence bound（方法与置信水平在 G1 冻结）判过，或把 MC 误差控制到预注册容差后作保守判定。若 60/task 仍不过，默认止损并交 owner 决定是否扩 N，不能自动外推。
3. 每个外层样本必须真正执行最终 analyzer 的复合判定：`left-support gate pass ∧ one-sided q05>0`；不得再次退化为某个 SD 的正态近似。

### 12.4 `v` 的离线机制指标

1. cal split 可以评估**未用 cal 拟合的 raw `q_hat`** 的 pinball loss，作为 development diagnostic；但同一 cal split 已用于 conformal correction 时，不能再把 correction 后的 coverage 称为 held-out coverage。
2. coverage–resolution 若要写成 held-out 证据，必须使用 nested cross-fitting / OOF prediction 或另一个未参与 correction 的 query split，并以 episode/task cluster 统计；若现有产物无法做到，只能标成 calibration-resubstitution 描述量，不能通过改名制造 held-out test。
3. 该离线结果不替代 H1 的闭环证据，也不恢复 H2 gating。

### 12.5 C gate、fresh pool 与封存

1. 区分“进入 C 前的设计门”和“C 上的确认 verdict”：A-1/A-3/A-4、dense-baseline 止损与 power record 是 pre-C gate；C 的唯一 inferential primary 是 H1。S0 三臂和 always-full anchor 可保留，但均不改变 H1 的 α；H2 只报 effect + CI。
2. fresh-init generator 必须以内容级 digest 证明 C 与 library / fit / calibration / old A′ / pilot 全部互斥；task 配额、seed authority、去重、最大候选 N 与 prefix-selection 规则在生成前冻结。
3. C 在所有冻结臂完成前禁止读取 outcome-aware aggregate；partial arm 结果不能触发换点、加臂或调 N。P/C ledger、arm YAML、analyzer version、cost authority 和 pool digest 必须共同封存。

### 12.6 Action Cache 的时间边界

1. 接受 Action Cache 代码与 rollout 不并入本轮实现交付；但不能把“是否比较、比较哪些配置、如何计总 model-forward compute”留到 C outcome 解封之后。
2. 在 C roster 最终冻结前，必须另行冻结 Action Cache 的 inclusion decision、development 选点协议、C pool 复用规则及成本映射。共同横轴应是每个 family 的**总 model-forward compute budget**；不能把 CP2 命中强塞进 CP1 的 FULL/WARM/MISS 三单价表。
3. 若 Action Cache 不进入同一预冻结 C 对照，论文不得声称 confirmation 证明优于 Action Cache，只能报告其为另行进行的 post-confirmation / descriptive baseline。

### 12.7 更新后的执行顺序

第二计划顺序确认为：

1. 编写一个覆盖 budget evaluator、合法 threshold-grid exploratory 分支、对称 envelope/C-roster 选择器、完整 power MC、`v` 离线指标、H1-only gate、fresh generator 与封存纪律的 G1 plan；plan 同时挂接 `docs/iclr/ICLR_PAPER_BLOCKING_TODO.md` 的逐项完成证据。
2. G1 → Code → G2；任何 rollout 前完成 dry validation、source-lock 与旧 artifact 可复算验证。
3. 只在 l10 development 上跑冻结 threshold grid；所有结果保持 `posthoc_exploratory=true`。
4. 用旧 Phase 0 + 新网格生成 budget amendment artifact，不覆盖旧文件；机械确定最终 baseline、C roster 和 `[B_L,B_H]`。
5. 若 dense threshold 下 development H1 的单侧 LB `≤0`，在 C 前止损；若通过，运行完整裁决 MC，以保守规则选 N。
6. 在 C 冻结前完成 §12.6 的 Action Cache inclusion/cost-map 决定，并完成 blocking TODO 要求的论文语义/provenance 审计。
7. 生成并封存 P/C、roster、seed、analyzer 与 cost authority；等待 owner 明确放行 l10 C。

上述修改均可在一个第二计划中形成可审计条目；因此共识已经达到，执行方现在可以开始编写 plan，但仍不得在 plan 通过 G1/G2 前改代码或运行 rollout。

## 13. 执行方接收 §12 并进入第二计划（2026-08-29）

- §12 的表态满足 §11.3 的共识形式；执行方接受 §12.1–§12.7 全部修改条目，并核验其中可核的事实：网格合法 cell 32 / 新臂 29 / 8700 ep（枚举复算）；`derive_thresholds` 在 `fh+ws>100` 时 `i_ws` clamp 到 `n−1`（`threshold_solver.py:203`）；`fh20/ws40`、`fh30/ws40` 的 nominal cost 47.79 / 42.06 ms（算术复算）——但 Rev 1 三臂的 realized−nominal 偏移为 +14.0 / +9.6 / +4.0 ms，所以 nominal 落在区间内不等于 realized 落在区间内，计划把该偏移作为 outcome-blind 覆盖证明的输入（plan §3.2）；`fit_surface.py:805` 在 empirical 模式下 `dev_mask = fit|cal`，部署 q_hat 确实用了 cal 行，§12.4 成立，OOF 机制（`assign_folds/fit_fold_models/oof_predictions`）已存在可复用（plan §3.5）。
- 第二计划：[`dispatch_surface_rev2_confirmation_plan.log.md`](dispatch_surface_rev2_confirmation_plan.log.md)（L2；§3.1 budget evaluator、§3.2 tgrid、§3.3 amendment analyzer + C roster 选择器、§3.4 power MC、§3.5 `v` 离线指标、§3.6 H1-only gate、§3.7 fresh-init 与封存、§3.8 Action Cache 边界、§8 BLOCKING_TODO 挂接）。
- 本文档到此冻结为 Phase 0 的记录；后续讨论移到第二计划的 §9 Review Log。
