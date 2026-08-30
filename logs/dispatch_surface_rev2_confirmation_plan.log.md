# Dispatch Surface Rev 2 — 第二计划（budget-mixture estimand / dense threshold grid / 完整裁决 power MC / `v` 离线机制指标 / H1-only gate / fresh-init C 与封存）

> Status: **G1 APPROVED — Plan Rev 4**（2026-08-29；R1 B1–B7、R2 B1–B5 与 G1 Round 3 reviewer 直接修订全部闭合，见 §9）。Level **L2**（`exp/` 多文件 + 测试；不改 `src/`）。
> 共识依据：[`dispatch_surface_rev2_phase0_result.md`](dispatch_surface_rev2_phase0_result.md) §11 + §12。
> 协议权威：[`dispatch_surface_rev2_protocol_draft.md`](dispatch_surface_rev2_protocol_draft.md) v1；本计划的 §3.1 / §3.3 / §3.6 / §3.7 / §3.8 以 **§13 amendment** 追加，不删改旧文。**§13 的规则在 G1 APPROVED 的同一时刻冻结（记录 plan SHA + review SHA）**；之后只允许以独立的 `amendment_result` artifact 机械填入 tgrid package SHA、`[B_L,B_H]`、roster、N 等输出值，不得改规则（R2-B5）。适用范围 = budget amendment development + power MC + C。
> 论文 Gate：[`../docs/iclr/ICLR_PAPER_BLOCKING_TODO.md`](../docs/iclr/ICLR_PAPER_BLOCKING_TODO.md)；其 §1.1–§1.3 / P0-B 已按 budget-mixture 定义同步；§8 逐项挂接。
> Phase 0 产物（`exp/dispatch_surface/data/aprime_rev1/<suite>_exploratory/`、两个 `cost_map_frozen.json`、`phase0_outcome_design.json`）与 Rev 1 产物**永久保留、不覆盖、不回写**；本计划所有新分析写到新文件并带 `posthoc_design_amendment=true`。
> **本计划不含任何 rollout 自动启动**；G1/G2 通过前不改 `exp/`、不 emit tgrid 矩阵、不 rollout、不 materialize P/C；C 的放行只能由 owner 明确给出。

---

## 1. 范围与不在范围

**在范围**（一次 G1 → Code → G2 → Verify）：

1. **budget-mixture evaluator**（§3.1，B1）：`V_F(B)` = episode-level randomized mixture 在 **ratio-of-sums 成本 ≤ B** 约束下的最优 SR（一维预算 LP，最优基 ≤ 2 臂，分式混合曲线，精确分片解析积分）；旧 `frontier_hull` 函数逐字不动；估计器版本 digest 进入所有新产物。
2. **threshold-grid exploratory 分支**（§3.2）：emitter / runner 新层 `exploratory_tgrid`，冻结网格 `fh ∈ {20,…,80} × ws ∈ {0,10,20,30,40}`、`fh+ws ≤ 100`（32 合法 cell − Rev 1 已有 3 = **29 新臂**），只补 **libero_10**；**tgrid finalization package**（B4）作为下游唯一输入。
3. **budget amendment analyzer**（§3.3）：两阶段（cost-only → 冻结 SHA → outcome）在 **Rev 1 + Phase 0 + tgrid package** 上生成 amendment artifact：family-agnostic development envelope、`[B_L,B_H]`、**基于 active optimal basis 的 C roster 选择器**、H1 / H2 / S0−T / A-2 / LOTO / per-task；**dense-baseline 止损**。
4. **完整裁决 power MC**（§3.4，B6）：共享纯函数 `evaluate_h1_verdict`、逐 `(N, r)` 冻结的 SeedSequence 派生、canonical 索引 digest、非单调裁决规则、Clopper–Pearson 单侧下界常量。
5. **`v` 离线机制指标**（§3.5，B5）：两条并列 OOF pinball readout（horizon 7 / 10，分位 α）+ 预冻结的 coverage / sharpness 表；不使用 "resolution"；descriptive。
6. **确认协议 amendment + H1-only gate + confirmation analyzer + confirmation discipline**（§3.6，B2）：outcome-blind 完整性认证先于 unseal。
7. **fresh-init generator 与 P / C 封存**（§3.7，B7）：逐 state 生成状态机（env 构造边界、RNG 赋值、reset API、接受条件、`MAX_RETRIES=4`、失败占位）、真实 LIBERO round-trip smoke、跨机 digest、互斥、prefix 规则、pilot 一次性。
8. **Action Cache decision record**（§3.8，B3）：schema + validator + seal 条件分支；`inclusion=yes` 时本计划的 seal 拒绝，直至独立的 Action Cache package 通过 G1/G2。
9. 测试（§6）、`logs/README.md`、handoff、数据台账。

**不在范围**：spatial 的 dense grid；Action Cache 的代码与 rollout（若 `inclusion=yes`，另立 G1）；`src/` 变更；Rev 1 / Phase 0 冻结判据与产物的改动；论文 LaTeX 正文迁移（BLOCKING_TODO P0-C）；真机；phase-aware retrieval。

**硬约束**：Rev 1 与 Phase 0 的全部脚本对旧输入**逐字节等价**（六个 artifact、两个 verdict、两个 `cost_map_frozen.json`、两个 `phase0_outcome_design.json`）；不得以放宽旧校验兼容新层；`tests/review_tests/` 不读不列不搜；成本只来自 `analytic_cost`（**不改单价、不改 ratio-of-sums 成本轴**——B1 的修正是改 mixture 的可达集合定义，不是改成本口径）；所有新产物 `posthoc_exploratory=true` 或 `posthoc_design_amendment=true`；C 产物在 unseal 前不得被任何 outcome-aware 代码路径读取（AST source-lock）。

## 2. 文件清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `exp/dispatch_surface/analysis/frontier_hull.py` | 不改 | 旧 exact-cost 函数原样保留（Phase 0 复算） |
| `exp/dispatch_surface/analysis/budget_mixture.py` | 新增 | §3.1：`ArmStats(T, D, S, E)`、`value_at(stats, B)`（LP 枚举 + hull-at-zero 等价）、`pieces(stats, B_L, B_H)`（断点枚举）、`auc_norm(stats, B_L, B_H)`（分片解析积分）、`auc_with_support(stats_a, stats_b, B_L, B_H)`、`difference_extrema`、`active_basis_union(stats, B_L, B_H)`；纯函数，无 I/O |
| `exp/dispatch_surface/analysis/estimator_version.py` | 新增 | `EXACT_COST_V1`（旧，登记）与 `BUDGET_MIXTURE_V1` 的 canonical payload + digest |
| `exp/dispatch_surface/phase0_roster.py` | 扩展（追加） | §3.2：`THRESHOLD_GRID_FH/WS`、`tgrid_cells()`、`tgrid_arm_id`、`tgrid_roster_spec` + digest、`LAYER_TGRID`、`PROTOCOL_TGRID`；§3.3：`F_MIN = 0.20`、`M_MAX = 6` 常量 |
| `exp/dispatch_surface/emit_precheck_yamls.py` | 扩展 | `emit_tgrid(args)`；Rev 1 / Phase 0 分支不动 |
| `exp/dispatch_surface/run_precheck.py` | 扩展 | `LAYER_TGRID` 校验分支 + `TGRID_FROZEN_LAUNCH_KEYS`；`LAYER_CONFIRMATION` 分支：验证 seal / task-plan / fresh pool，按 task plan 构造 `EpisodeTask(orig_init_state_idx=None)`，ledger 冻结 `seal_sha256 / confirmation_task_plan_sha256`；不改 producer row schema；Rev 1 / Phase 0 分支不动 |
| `exp/dispatch_surface/analysis/phase0_discipline.py` | 扩展 | `validate_tgrid(...)`（与 Phase 0 同强度） |
| `exp/dispatch_surface/finalize_tgrid_package.py` | 新增 | §3.2-f（B4）：29×300 完整性 → 不可变 package + MANIFEST（role → member + SHA） |
| `exp/dispatch_surface/tgrid_package.py` | 新增 | role 解析 / `verify_package`（与 `rev1_package` 同模式；不复用以免改动 Rev 1 模块） |
| `exp/dispatch_surface/analysis/budget_cost_map.py` | 新增 | §3.3 Stage 1（cost-only；source-lock：不 import `budget_mixture` 的 outcome 路径——见 §4） |
| `exp/dispatch_surface/analysis/budget_outcome_design.py` | 新增 | §3.3 Stage 2 |
| `exp/dispatch_surface/analysis/h1_verdict.py` | 新增 | §3.4 / §3.6（B6）：`evaluate_h1_verdict(sufficient_stats, frozen_design, bootstrap_index) -> Verdict` 唯一实现 |
| `exp/dispatch_surface/analysis/confirmation_power_mc.py` | 新增 | §3.4 |
| `exp/dispatch_surface/analysis/v_offline_metric.py` | 新增 | §3.5 |
| `exp/dispatch_surface/analysis/confirmation_discipline.py` | 新增 | §3.6-b（B2 / R2-B4）：outcome-blind Cartesian 完整性认证；`task_uid` ↔ `confirmation_task_plan.json` 唯一 join |
| `exp/dispatch_surface/build_confirmation_task_plan.py` | 新增 | §3.6-b：在 seal **之前**由 frozen roster × task × prefix 与 C manifest 生成无环 `confirmation_task_plan.json`（arm × task × prefix → pool_id / fresh_state_sha256）；task-plan SHA 进入 seal |
| `examples/libero/episode_runner.py` | 不改 | R2-B4 采用方案 2（task_uid join），producer 不动 |
| `exp/dispatch_surface/analysis/confirmation_analyzer.py` | 新增 | §3.6-c |
| `exp/dispatch_surface/generate_fresh_inits.py` | 新增 | §3.7 |
| `exp/dispatch_surface/action_cache_decision.py` | 新增 | §3.8（B3）：decision record schema / validator / claim restriction |
| `exp/dispatch_surface/seal_confirmation.py` | 新增 | §3.7-h / §3.8：seal（条件分支）与 unseal |
| `exp/dispatch_surface/fit_surface.py` | 不改 | OOF 函数 import 复用 |
| `exp/data_authority/records/dispatch_surface__libero_10__tgrid_dev.json` | 新增 | kind `cache_artifact`：tgrid package（integrity = MANIFEST + 全成员） |
| `exp/data_authority/records/dispatch_surface__libero_10__fresh_pools.json` | 新增 | kind `init_pool`：P / C |
| `exp/data_authority/records/dispatch_surface__<suite>__budget_amendment.json` ×2 | 新增 | kind `cache_artifact`：budget cost map、outcome design、roster、power record |
| `logs/dispatch_surface_rev2_protocol_draft.md` | 追加 §13（FROZEN） | G1 Round 3 放行时冻结；以后只允许独立 result artifact 填机械输出值 |
| `docs/iclr/ICLR_PAPER_BLOCKING_TODO.md` | §1.1–§1.3 / P0-B 修订 | budget-mixture LP、无 Pareto 剪枝、step-envelope 敏感性 |
| `tests/dispatch_surface/test_rev2_confirmation.py` + `fixtures/` | 新增 | §6；fixture 含 l10 冻结 9 臂 sufficient stats（`(T, D, S, E)` 全样本）以钉住 §3.1 的回归值 |
| `logs/README.md`、`logs/session_handoff_dispatch.md` | 更新 | |

## 3. 设计

### 3.1 budget-mixture evaluator（B1；BLOCKING_TODO §1.3 已同步）

**输入**：每臂、每 replicate 的 sufficient statistics `ArmStats = (T, D, S, E)` = 该臂在重采 cell 集上的总解析成本、总决策数、总成功数、episode 数（`E` 各臂相同 = 重采 grid 大小）。每 episode 均值 `t = T/E`、`d = D/E`、`s = S/E`；单臂 ratio-of-sums 成本 `c = t/d = T/D`（与冻结口径一致）。

**可达集合与价值函数**：按 episode 以概率 `p_i` 选臂的 mixture 的 decision-weighted ratio-of-sums 成本为 `C(p) = Σ p_i t_i / Σ p_i d_i`，SR 为 `Σ p_i s_i`。定义

```text
V_F(B) = max_p  Σ p_i s_i    s.t.  Σ p_i (t_i − B d_i) ≤ 0,  Σ p_i = 1,  p ≥ 0
```

- 可行 ⇔ `min_i c_i ≤ B`；`B < min_i c_i` ⇒ **不可行（support miss）**，禁止左外推。
- 一维预算约束 ⇒ 最优基至多两臂。**机械枚举**：所有可行单臂（`t_i − B d_i ≤ 0`）取 `s_i`；所有 tight 两臂 `(i 可行, j 不可行)`：`p_j = g_i/(g_i − g_j)`（`g = t − B d`），值 `(1−p_j) s_i + p_j s_j`；取最大。**等价形式**（作为测试断言）：点集 `{(x_i(B) = t_i − B d_i, s_i)}` 的 upper concave hull 在 `x ≤ 0` 上的最大值。
- **不做任何 `(c, s)` 标记上的 Pareto 剪枝或同成本去重（R2-B1）**：roster 内全部臂的 `(T, D, S, E)` 都进入 LP。反例（B = 30）：A `(c=10, d=1, s=0.5)`、B `(c=20, d=100, s=0.4)`（在 `(c,s)` 上被 A 严格支配）、C `(c=100, d=1, s=1.0)`：A/C 最优 0.6111，**B/C 最优 0.9607**——被支配臂因 `d` 更大成为最优基。standalone dominance 只作输出中的描述性标记，绝不作为输入剪枝；测试钉住该反例。
- **canonical tie / 数值策略（R2-B2，冻结）**：(i) 臂总序 = 冻结 roster 的 id 顺序（索引 = 位置）；(ii) `value_at` 的候选值以 `abs_tol = 1e-12` 判同值，同值时 basis tie-break = 臂数少者优先（单臂 > 两臂），再按 canonical arm tuple（roster 索引升序元组的字典序最小）；返回 `(value, basis)`；(iii) 输入域校验：`T ≥ 0`、`D > 0`、`E > 0`、`0 ≤ S ≤ E`，float64，违者 `ValueError`（fail closed）；(iv) 断点：三点共线根的分母 `|den| < 1e-12` ⇒ 无根；候选断点限制在开区间 `(B_L, B_H)`，排序后以 `abs_tol = 1e-9` ms 合并近重复；(v) 每个**正长度开区间**在中点取 canonical basis；`B_L`、`B_H` 两端点的 canonical basis 单独记录，**不**并入区间活跃基并集；interior 零测度点上的共最优替代 basis 不入并集；(vi) 两臂片积分前断言 `|γB + δ| ≥ 1e-12` 于片两端（否则 fail closed）；(vii) 每次解析计算都做廉价内部不变量检查（分片覆盖无缝/无重叠、断点有限有序、片中点 basis 与 `value_at` 一致、相邻片端值连续至 `1e-10`），失败即 `NumericMismatch`；adaptive Simpson 不进入每个 formal replicate 的热路径，按下述冻结 audit policy 使用。
- 预算语义内生：`B ≥ max_i c_i` 时所有臂可行，`V = max_i s_i`；不再需要"右端水平保持"的附加规则。
- 两臂片段 `f_ij(B) = s_i + (s_j − s_i)·(B d_i − t_i) / (B (d_i − d_j) + (t_j − t_i))` 是线性分式，单调；`V_F` 连续、非降、分片（单臂常数片 / 两臂分式片）。

**精确积分**（`auc_norm(stats, B_L, B_H)`）：
1. 断点集 = `{B_L, B_H}` ∪ `{c_i ∈ (B_L, B_H)}` ∪ `{B_ijk ∈ (B_L, B_H)}`，其中 `B_ijk` 为三点 `(x_i, s_i), (x_j, s_j), (x_k, s_k)` 共线的根——`x` 对 `B` 线性，故为**线性方程**：`B_ijk = [(t_j − t_i)(s_k − s_i) − (t_k − t_i)(s_j − s_i)] / [(d_j − d_i)(s_k − s_i) − (d_k − d_i)(s_j − s_i)]`（分母为 0 ⇒ 无根）。hull-at-zero 的活跃基只在某点穿越 `x = 0`（`B = c_i`）或三点共线时改变，故该断点集完备。
2. 每片在中点用枚举确定活跃基；单臂片积分 = 常数 × 长度；两臂片用线性分式的闭式原函数 `∫ (αB+β)/(γB+δ) dB = (α/γ) B + (β/γ − αδ/γ²) ln|γB+δ|`（`γ = 0` 时为线性）。
3. `AUC_norm = (1/(B_H − B_L)) ∫ V_F dB`；`auc_with_support` 对任一族在 `B_L` 不可行的 replicate 返回 `(−1.0, True)` 并保留（joint miss > 1% 仍 fail closed）。
4. **数值 audit policy**：确定性自适应 Simpson（绝对容差 `1e-10`、最大深度 60）只用于：(a) G2 的 1000 组随机/退化 stats 与冻结 fixture；(b) development/C analyzer 的 full-sample + 由 `sha256(estimator_digest|artifact_input_digest)` 派生、无放回抽取的最多 100 个 bootstrap replicate；(c) power MC 每个 `(N, outer_r)` 的 inner replicate 0（共 800 个）。任一解析值与 Simpson 差 > `1e-8` ⇒ 整个对应 artifact / power record fail closed。不得在 800 万个 formal inner replicate 上逐一跑 Simpson，也不得用 Simpson 值替换解析值。l10 冻结 9 臂 full-sample 上 H1 = **0.0816441**（旧直线 hull 0.0722133）、H2 0.0447783、S0−T 0.0368658。

**A-2 极值**（`difference_extrema`）：`V_SV − V_T` 在每片是两个线性分式之差；极值只在断点（两族断点并集）或片内闭式驻点（导数差 `k₁/(γ₁B+δ₁)² − k₂/(γ₂B+δ₂)² = 0` 的实根）取得；输出断点集、驻点集与极值。

**active basis**（`active_basis_union`）：区间内各**正长度开区间**片的 canonical 活跃基并集（臂 id 集合，按 roster 顺序）；`B_L` / `B_H` 端点 basis 另存 `endpoint_bases`；bootstrap 每 replicate 输出 active-arm **bitset**（位序 = roster 顺序，big-endian 定长字节编码），rollup digest = sha256(按 replicate 顺序拼接的 bitset 字节)；结果与输入臂顺序无关（测试）。

**measured-policy-only step envelope（non-gating 敏感性，R2 non-blocking 3）**：`V^step_F(B) = max{ s_i : T_i/D_i ≤ B }`（只用实测单臂，不做混合）；对每个假设同时输出 step 版 AUC 差与 bootstrap q05/q95，作为“主结论是否主要来自 decision-count 加权 synthetic mixture”的敏感性证据；不替代主 LP、不 gating；不需要额外 rollout。

**估计器版本**：`BUDGET_MIXTURE_V1 = {"name": "budget_mixture_v1", "objective": "max episode-mixture SR", "cost_constraint": "C(p) = sum p_i T_i / sum p_i D_i <= B", "pruning": "none", "basis": "<=2 arms (single or tight pair)", "tie_break": "fewer arms, then canonical arm tuple", "value_tol": 1e-12, "breakpoint_tol_ms": 1e-9, "left_tail": "infeasible", "right_tail": "implicit (all arms feasible)", "integral": "exact piecewise: constant + linear-fractional closed form; breakpoints = single costs + three-point collinearity roots", "numeric_audit": "internal invariants every call; Simpson abs_tol 1e-10 on frozen audit subset; mismatch > 1e-8 fails artifact", "support_miss_value": -1.0, "max_joint_miss": 0.01, "sensitivity": "measured-policy-only step envelope"}`，canonical digest 写入 tgrid 矩阵（记录）、budget cost map、outcome design、power record、seal、analyzer 输出。

**记号**：横轴统一写 `C(π) = E[T(π)] / E[D(π)]`（episode 总解析成本期望 / episode 决策数期望），约束写 `C(π) ≤ B`；不使用会被理解为 `E[T/D]` 的 `E[c(π)]`。输出术语用 measured policies / active mixture bases，不用 “efficient hull vertices”。

**不改的东西**：单臂成本仍为 ratio-of-sums（owner 冻结）；`frontier_hull.py` 逐字不动；Phase 0 的 exact-cost 产物原样保留为旧 estimand 下的 development record。

### 3.2 threshold-grid exploratory 分支（§12.1；non-blocking 1；B4）

a. **网格常量**：`THRESHOLD_GRID_FH = (20,30,40,50,60,70,80)`、`THRESHOLD_GRID_WS = (0,10,20,30,40)`；合法 `fh + ws ≤ 100`；`tgrid_cells()` = 32 cell，去掉 Rev 1 `(30,20),(50,20),(70,10)` = 29 新 cell；臂名 `dsp_tg_fh{fh}_ws{ws}`；`tgrid_roster_spec("libero_10")` + digest。
b. **阈值导出**：`derive_thresholds` 逐字复用；输入 = 归档包 `fit.sv` 的 `input_digests.table` 所指表（`--table` 内容 SHA 必须相等）的 fit∪cal 分数；emitter 拒绝 `fh+ws>100`；**`ws = 0` ⇒ yaml 无 `warm_tiers` 键**（冻结的唯一 canonical 表示）；`ws > 0` ⇒ 恰一个 tier `{threshold: t_ws, start_t: 0.3}`；记录 `nominal`、实际 `(t_fh, t_ws|null)`、`threshold_pair_digest`；按实际 pair 去重——冲突 ⇒ emitter 拒绝并列出冲突 cell。
c. **nominal cost 与偏移**：写出每 cell 的 `nominal_cost_ms = (fh/100)·c_FULL + (ws/100)·c_WARM + (1 − (fh+ws)/100)·c_MISS`（`fh`、`ws` 为整数百分数） 与 Rev 1 三臂的 `realized − nominal`（+14.0 / +9.6 / +4.0 ms，成本行可算）；默认 32-cell 网格冻结，不做更省网格。
d. **矩阵**：`protocol = "dispatch_surface_rev2_tgrid_dev"`、`layer = "exploratory_tgrid"`、`posthoc_exploratory = true`、`suite`、`tgrid_roster_spec_sha256`、`rev1_package_manifest_path/sha256`、`rev1_matrix_sha256`、`table_sha256`、`gate_theta`（Rev 1 matrix，不重算）、`arms` / `arm_yaml_sha256` / `nominal` / `threshold_pairs` / `threshold_pair_digests` / `threshold_pair_rollup_sha256` / `nominal_cost_ms`、`library_pkl/sha256`、`template`、`cost_model` + digest、`estimator_version`（记录）、`contract_source = "rev1_package:artifact.dsp_sv"`。
e. **runner**：`--layer exploratory_tgrid`；gate `always_search`；arm 校验：judge `threshold`、阈值 finite 且 `≤ 1.0`、`ws>0` ⇒ 恰一个 tier 且 `t_fh > t_ws` 且 `start_t == 0.3`、`ws=0` ⇒ 无 `warm_tiers` 键、无其他 tier / 字段、`(t_fh, t_ws)` 与矩阵逐值相等且 pair digest 一致；roster digest 一致；`core_arms=[]`、全部 descriptive；ledger frozen keys 增 `tgrid_roster_spec_sha256`、`threshold_pair_rollup_sha256`、`contract_source`、`estimator_version`；subset launch 继承同一矩阵 / ledger；`--dry-validate` 覆盖全部。执行目录 `/tmp/dsp_precheck/libero_10_exploratory_tgrid/`（只是执行期落点，**不是**下游输入）。
f. **finalization package**（B4）：`finalize_tgrid_package.py` 在 **精确 29 臂 × 300 cell** 完整（每 cell 恰一个 accepted、`(run_id, arm)` 全部登记且被执行、无 off-grid / 重复 / stale）通过后，把 matrix、完整 ledger、29 个执行 yaml、split manifest 与 A′ pool attestation、journal、per_step、launch metadata（run-id / 批次集合、server bundle 指纹）、roster / pair rollup、cost model 打包到 `exp/dispatch_surface/data/tgrid_dev/libero_10/` 并写 `MANIFEST.json`（role → member + SHA + 打包时间）；data_authority 记录。`budget_cost_map` **只**经 `tgrid_package` 的 role 解析读取；不接受散落路径。负例：缺批次 / 混入不同 matrix 的行 / finalize 后成员漂移 / 少一个 cell ⇒ 拒绝。
g. **discipline**：`phase0_discipline.validate_tgrid`：与 Phase 0 同强度（唯一 run id、`(run_id, arm)` 认领、yaml 字节重算、pair digest、contract binding、A′ pool attestation、cost model digest）。
h. spatial 不跑网格；spatial 的 amendment 用 Phase 0 数据。

### 3.3 budget amendment analyzer（§12.2；B1-4；non-blocking 2）

**Stage 1 — `budget_cost_map.py`（cost-only；AST source-lock：不 import `budget_outcome_design` / `h1_verdict` / `analyze_precheck`，不访问 `success` / `status`）**
1. 候选集 = Rev 1 package（3 T + SV / SV− / S0）∪ Phase 0 suite roster ∪ tgrid package（l10：29 臂；spatial：无）。每臂 `(cost_sum, n)` per cell（`precheck_io` cost-only loader），各自 ledger 认领。
2. family 端点（outcome-blind）：SV / S0 沿用 §3.3 isotonic（δ 排序）；threshold 族端点 = point-estimate cost 最低 / 最高的臂（不伪造 δ）。
3. `[B_L, B_H]`：协议 §3.3 同式（共享 `PCG64(20260829)`、R = 10000、`qL = sorted[9950]`、`qH = sorted[49]`、`ceil/floor_0.1`）；`B_1, B_2` 同式。候选集变大后区间可变（cost-only、rollout 前不可预知）；旧 `[41.7,47.7]` 只作记录。A-3′：宽 ≥ 4.0 ∧ 每族 ≥ 3 个不同 point cost ∧ 每族 `min c ≤ B_L` ∧ 每族 `max c ≥ B_H`。
4. 输出 `budget_cost_map_frozen.json`（全部 input SHA、索引 SHA、估计器 digest）；记录 SHA 后 Stage 2 才可运行。

**Stage 2 — `budget_outcome_design.py`（先核 cost map SHA 与全部冻结输入 digest，再读 SR）**
1. development envelope：每族**全部**已测点的 `ArmStats`（不强制臂数相同），按 §3.1 求 `V_F`；输出 measured points、分片（活跃基、断点）、`[B_L,B_H]`。
2. **C roster 选择器**（family-agnostic，rollout 前冻结为代码）：
   - 必选：full-sample 上 `active_basis_union(stats, B_L, B_H)`（正长度开区间的 canonical 活跃基并集）∪ `endpoint_bases`（`B_L`、`B_H` 的 canonical basis）；interior 零测度共最优替代 basis 不入选；结果与臂输入顺序无关（测试）。
   - 稳定性：共享 10000 paired replicates 上每臂 `f_a` = 出现在区间活跃基并集的频率；`f_a ≥ F_MIN = 0.20` 者入选；per-replicate bitset rollup digest 写出。
   - 上限：每族 `≤ M_MAX = 6`（anchor 不计入任何族，固定进入 C）；超出 ⇒ `verdict = "roster_overflow"`，fail closed 回 G1，不得按 outcome 人工删点。
   - 输出 `c_roster.json`（臂、family、来源 package/role 或 tgrid cell、δ 或 nominal、full-sample 是否活跃、`f_a`、理由码）。
3. 假设量（全部 §3.1 estimand）：H1 SV−T（plug-in、bootstrap mean / q05 / q95 / SD、miss）、H2 SV−S0、S0−T（描述）、A-2（spatial，描述）、LOTO、per-task descriptive、`B_1 / B_2` 处的 `ΔV` 与 middle discordance；**每项同时输出 step-envelope 版本**（§3.1 敏感性）；每臂标注 standalone `(c,s)` dominance 状态（描述性，若被支配臂进入活跃基须如实保留）。
4. **dense-baseline 止损**：l10 development H1 单侧 q05 ≤ 0 ⇒ `verdict = "stop_before_C"`；power MC / seal 据此拒绝运行。
5. 输出 `budget_outcome_design.json`（`posthoc_design_amendment=true`、`development_only=true`、估计器 digest、cost map SHA、C roster）；不覆盖 Phase 0 文件；spatial 用 Phase 0 数据运行（A-2 + 描述，无 gate）。

### 3.4 完整裁决 power MC（§12.3；B6；non-blocking 4）

- **共享纯函数** `h1_verdict.evaluate_h1_verdict(sufficient_stats_by_arm_cell, frozen_design, bootstrap_index) -> Verdict{pass, effect, q05, q95, joint_miss, left_support_ok, active_bitsets, reason}`：`frozen_design = {roster, families, B_L, B_H, estimator_digest, R, max_joint_miss}`；power MC 与 confirmation analyzer **直接调用同一函数**（不是源码断言）。判定 = `left_support_ok ∧ joint_miss ≤ 0.01 ∧ q05 > 0`。
- 前置：只能在 `c_roster.json` 与 `[B_L,B_H]` 机械确定且止损未触发后运行；全部输入 SHA 写入 record。
- 外层：候选 `N_CANDIDATES = (30, 40, 50, 60)`，`R_OUTER = 200`（代码常量；CLI 不可改 formal 值，只有 `--smoke` 产 `smoke=true` 的非正式 record）。对每 `(N, r)`：外层 RNG = `np.random.Generator(PCG64(np.random.SeedSequence(20260830, spawn_key=(N, r))))`，每 task 从 30 个 development cell **有放回**抽 `N` 个（paired：同一抽样用于全部 roster 臂）；内层 RNG = `PCG64(SeedSequence(20260829, spawn_key=(N, r)))` 生成 R = 10000 的 task-stratified paired 索引；两个索引矩阵各以 canonical 编码（`json.dumps(list, separators=(",",":"))`）取 sha256；`(N, r)` 完全决定所有随机性，与并行调度无关。
- record：逐 `(N, r)` 的 `{verdict, reason, effect, q05, joint_miss, outer_index_sha256, inner_index_sha256}`；aggregate digest = sha256(按 `(N, r)` 排序的逐条 digest 串接)。
- **裁决**：`LCB(N)` = Clopper–Pearson 单侧 95% 下界 = `scipy.stats.beta.ppf(0.05, k, n − k + 1)`（`k` 通过数，`n = R_OUTER`；`k = 0 ⇒ 0`）。选 **最小的 N，使得该 N 及所有更大候选的 `LCB ≥ 0.80`**（序列出现 pass→fail 时不得取更小的 pass）；`N = 60` 不满足 ⇒ `verdict = "underpowered_stop"` 交 owner，不自动外推。
- 报告（非 gating）：effect ×0.5 情景通过率；MC 误差；实际用时。
- **假设注记**：effect-replication = "在 development 选定 roster 的条件下，以官方 A′ 的经验联合分布近似 fresh C"，是设计假设；P 上只跑 anchor 只能检验 full-inference 难度漂移，不能检验 H1 效应的 transport。
- 计算：每族 ≤ M_MAX+1 臂 ⇒ 断点 O(K³) 很小；4 × 200 × 10000 次 `auc_with_support`；numpy 向量化 + `multiprocessing`（weilandserver CPU）；G2 前 `--smoke`（`R_OUTER_SMOKE = 20`）核时长。

### 3.5 `v` 的离线机制指标（§12.4；B5；descriptive，无 gate）

- 数据：归档包 `fit.sv` 绑定的表（D_dev = fit∪cal；l10 9205 行、spatial 3364 行）；`alpha`（分位水平）、`ladder`、fold 分配与 fit record 一致（`fold_map_sha256` 复核）。
- **两条并列 readout**（horizon 7 = `y7`，horizon 10 = `y10`），各自独立报告，不聚合、不二选一。
- **OOF**：`assign_folds`（按 task 内 init 5-fold）；对 fold `f`：用其余 4 折 `choose_grid` + `bin_index` + `fit_bimonotone_quantile`（SV：`uses_disagreement=True`；S0：s-only），预测 fold `f` 的行（`oof_predictions`）。
- **pinball**：`ℓ_α(y, q̂) = α·max(y − q̂, 0) + (1 − α)·max(q̂ − y, 0)`；row 级损失 → **episode 均值** → task 内 episode 等权 → task 等权（各 task 一个数）→ 全局 = task 均值。SV − S0 差：**paired episode bootstrap**（unit = episode，task 内有放回，R = 10000，`PCG64(20260831)`），报 mean / q05 / q95；符号：负 = SV 更好。
- **coverage / sharpness 表**（无 bin、无 "resolution"）：OOF 上 `P(y ≤ q̂)` 的 episode 加权经验覆盖（全局与每 task）与 `mean(q̂)`（sharpness，越小越 sharp，只在覆盖相当时可比）；同一 bootstrap 给 CI。
- 标签：全部 `label = "oof"`；任何用到 final fit 的量（如与部署 artifact 的对照）标 `calibration_resubstitution`；输出中**禁止**出现 "held-out coverage" 字样（字符串断言）。
- 用途：回答 "`v` 是否改善风险预测"；不替代 H1，不恢复 H2 gating。

### 3.6 确认协议 amendment、H1-only gate、confirmation discipline 与 analyzer（§12.5；B2）

a. **协议 §13（G1 Round 3 FROZEN）**：estimand = §3.1；H1 唯一 inferential primary：`Δ_H1 = (1/(B_H−B_L)) ∫ [V_SV(B) − V_T(B)] dB`，10000 paired task-stratified bootstrap，判过 = `left_support ∧ joint miss ≤ 1% ∧ 单侧 95% q05 > 0`；H2 / S0−T 只报 effect + q05/q95；secondary：`ΔV(B_1), ΔV(B_2)` 的 **studentized max-t simultaneous 95% band**（每 replicate 的 `t_k = (Δ_k^{(r)} − Δ_k) / ŝ_k`，`ŝ_k` = 该点 bootstrap SD；临界值 = `max_k |t_k|` 的 0.95 分位；support-miss replicate 不进 band、其率单独报告；exploratory，不 gating）；A-2 spatial 描述。
b. **`confirmation_discipline.py`（outcome-blind，unseal 前必过）**：
   - Cartesian 完整性：`c_roster` × 10 task × prefix `0..N−1` 每 cell **恰一个** accepted `(task_uid, attempt, run_id)`；无 off-grid、无重复 accepted、无 stale / fenced / 其他 run 的行混入；
   - identity（R2-B4，采用**方案 2**：producer 不改）：在 seal 之前，`build_confirmation_task_plan.py` 由 frozen roster、N 与 sealed C-pool manifest 生成 `confirmation_task_plan.json`，映射 `task_uid = make_task_uid(arm, "eval", task_id, prefix_idx)` → `{arm, task_id, prefix_idx, pool_id: "C", fresh_state_sha256}`；**task plan 不含 seal SHA**，避免 `seal → task_plan_sha → seal_sha` 的哈希环。seal 绑定 task-plan SHA；runner ledger 同时冻结 seal SHA 与 task-plan SHA。discipline 以 `task_uid` 对 task plan 做唯一 join（每 uid 恰一条、无 plan 外 uid），核 task-plan roster/N/pool digest 与 seal、fresh digest 与 sealed C manifest 一致，并以 row `run_id` join ledger，要求 ledger 的 seal/task-plan SHA 与当前输入相等；所有臂对同一 `(task_id,prefix_idx)` 指向同一 state digest；行内 `subset_init_state_idx == prefix_idx`；**`orig_init_state_idx` 必须为 null**（confirmation strategy 置 `None`，subset loader 只用 `episode_idx`）；
   - 绑定：arm yaml SHA、matrix / ledger、server bundle 指纹、cost authority digest、estimator digest、journal / per-step producer schema 版本与 run id 逐项一致；
   - 每 accepted episode：per-step decision 行数 == `client_timing.infers`；verdict ∈ 三档、`WARM_START.start_t == 0.3`、其他为 null；缺行 / 未知 verdict / 多余 tier ⇒ 拒绝；
   - 产出 `confirmation_discipline.json`（输入 / 输出 SHA）；**AST source-lock**：本模块与 cost-only 路径不访问 `success` / `status`。
   - 负例（§6-9）：partial ledger、重复 accepted、stale attempt、错 pool state、某臂少一 cell、per-step 少行、off-grid cell、`orig_init_state_idx` 非 null、`task_uid` 不在 task plan / plan 内 digest 与 manifest 不符。
   - **真实路径 round-trip 测试**（非手造 fixture）：用真实 conductor 类构造 confirmation `EpisodeTask(orig_init_state_idx=None, extra={...})` → `protocol.task_to_wire/task_from_wire` → worker（stub episode 函数，不用 LIBERO）→ `EpisodeResult.per_step_rows` → driver 的 `_per_step_writer` 与 journal → `confirmation_discipline` 通过；断言行内 `orig_init_state_idx` 为 null、`task_uid` 可唯一 join。
c. **`confirmation_analyzer.py`**：需要 `unseal_record.json`（含 discipline artifact SHA、ledger SHA）；无 unseal ⇒ 只允许 `--cost-only` 视图；核 seal 内全部 digest；调用 `evaluate_h1_verdict`；输出 verdict（`h1_pass` / `h1_fail` / `support_miss`）+ 描述量 + secondary band；输出中不得出现任何 Action Cache 比较字段（§3.8 claim restriction）。
d. **pre-C gate**：A-1（development H1 q05 > 0）、A-3′、A-4（Phase 0 已过；C 上 anchor 再复核）、dense-baseline 止损通过、power record 存在且 N 由其机械选出、Action Cache decision record 已冻结且 seal 分支通过、P pilot 通过。

### 3.7 fresh-init generator 与 P / C 封存（协议 §5；§12.5-2/3；B7）

a. **逐 state 生成状态机**（每 `(suite, task, pool ∈ {P, C}, k)`）：
   1. **seed authority 与派生（R2-B3）**：attempt `a`（`a = 0` 为首次，`a = 1..MAX_RETRIES` 为 retry，`MAX_RETRIES = 4`，总 attempt ≤ 5）的 authority = `sha256(f"dsp_rev2_fresh|{suite}|{task_name}|{pool}|{k}|attempt|{a}")`（256 位，manifest 原样记录）；`ss = np.random.SeedSequence(int.from_bytes(authority, "big"))`；三处实际入参 **都** = `seed32 = int(ss.generate_state(1, dtype=np.uint32)[0])`（合法域 `[0, 2^32−1]`；`np.random.seed` 对 ≥ 2^32 实测抛 `ValueError`），分别记录 `py_seed / np_seed / env_seed`；跨机重放使用完全相同映射。
   2. **每个 attempt 新建 env**：`OffScreenRenderEnv(bddl_file_name=<task bddl>, camera_heights=256, camera_widths=256)`（与 `examples/libero/main.py` 同构造参数）；赋 RNG：`random.seed(seed32)`、`np.random.seed(seed32)`、`env.seed(seed32)`（robosuite placement sampler 走 `np.random` 全局态）；`obs = env.reset()`；`state = np.asarray(env.get_sim_state() if hasattr(env, "get_sim_state") else env.sim.get_state().flatten(), dtype=float64)`；attempt 结束 `env.close()`。
   3. **接受条件**：reset 无异常 ∧ `state.shape == (D_state,)`（`D_state` 由该 suite 官方 `.init` 的列数决定，l10 = 47）∧ `np.isfinite(state).all()`；否则下一 attempt `a+1`；`a` 达 `MAX_RETRIES` 仍失败 ⇒ 该 `k` 记 `failed`，**占用 k**（不重排、不补采）。
   4. **互斥**：`state_sha256 = sha256(np.ascontiguousarray(state).tobytes())`（与 `split_init_pools._sha256_states` 同口径的逐 state 版本）对官方 50/task 全部状态（D_lib 5 / fit 5 / cal 10 / A′ 30）及 P↔C 互查；碰撞 ⇒ 该 `k` 记 `collision`，占用 k，不重采。
   5. manifest 逐 `k`：`{pool, task_id, task_name, k, attempts: [{a, authority_sha256, seed32, outcome}], status ∈ {ok, failed, collision}, shape, dtype, state_sha256}`；环境与资产：libero / robosuite / mujoco / numpy 版本、bddl 文件 SHA、HF assets rollup、`task_manifest` SHA。
b. **配额与 prefix**：C 恰生成 `N_MAX = 60`/task，**要求 60/60 `status = ok`**（任一 `failed` / `collision` ⇒ `generator_validation_failed`，fail closed，无余量）；C 实际用 `k` 升序的前 `N` 个（prefix 规则在生成前冻结，N 来自 power record）；P = 10/task，独立 seed 流（`pool = "P"`），**要求 10/10 `ok`** 才 materialize / pilot，否则同样 `generator_validation_failed`。
c. **round-trip smoke（真实 LIBERO）**：每 task 至少 1 个生成 state：`env.set_init_state(state)` → `env.sim.forward()` → 重新读 state，与写入值逐元素相等；再经 runner 的 client 路径（`examples/libero/main.py` 的 `set_init_state` 分支）跑 1 步不崩；结果写入 manifest。
d. **materialize**：`{task_name}.init` = `torch.save(np.stack(states_ok))`（与 `materialize_pool` 相同格式），行序 = `k` 升序；`load_init_states` 可读。
e. **跨机确定性**：weilandserver 生成 → timan107 以同 seed 表重生成，逐 `k` `state_sha256` 相等才封存；不等 ⇒ fail closed（不得以任一方为准）。
f. **pilot 一次性**：P 上只跑 `always_full_inference`（100 ep）；`|SR_P − SR_0a| ≤ 10 pt`（0a = 0.847）；失败 ⇒ `generator_validation_failed`，不换 seed 重试。
g. **runner `confirmation` 层**：`validate_fresh_pool(seal, pool_dir, N)` 与 `validate_confirmation_task_plan(seal, task_plan)`；按 task plan 构造 `EpisodeTask(orig_init_state_idx=None)`，不改 per-step producer schema；ledger frozen keys 增 `seal_sha256`、`confirmation_task_plan_sha256`、`pool_digest`、`N`、`estimator_version`；subset launch 允许，outcome 封印直到 discipline 通过。
h. **seal / unseal**（`seal_confirmation.py`）：`confirmation_seal.json` = {P/C manifest digest、`c_roster.json` SHA、`confirmation_task_plan.json` SHA、arm yaml SHA、analyzer / estimator digest、cost authority digest、N、power record SHA、Action Cache decision record SHA 与其 validator 结果、协议 §13 SHA}；task plan 先生成、seal 后生成，二者无循环引用；C 路径不进任何 development 可读配置（source-lock）；`--unseal` 只在 `confirmation_discipline.json` 存在且通过、ledger `roster_complete` 时生成 `unseal_record.json`（记录 discipline SHA、ledger SHA、时间）。

### 3.8 Action Cache decision record（§12.6；B3）

- **schema**（`action_cache_decision.py`，validator 拒绝缺项；**schema 与分支本轮冻结，owner 的 `inclusion` 取值稍后签署**）：`{inclusion ∈ {"yes", "no", "post_confirmation_descriptive"}, reason_code, statistical_status ∈ {"descriptive", "secondary"}（yes 时必填；不得为 primary）, development_selection_protocol, config_digest, code_digest, cost_mapping: {axis: "total model-forward compute budget per family", cp2_rule: "not mapped into CP1 three-tier unit table"}, c_pool_binding, claim_restriction}`；`yes` ⇒ `development_selection_protocol / config_digest / code_digest / c_pool_binding` 必须为非空 SHA / 对象；`no` / `post_confirmation_descriptive` ⇒ 这四项为 canonical `null`，`reason_code` 与 `claim_restriction` 必填。
- **seal 分支**：
  - `inclusion = "yes"` ⇒ **本计划的 seal 拒绝**，直至一个独立通过 G1/G2 的 Action Cache package 提供：development 选点 artifact、config / code / 成本映射 digest、C arm roster、同一 fresh-pool 绑定、runner / analyzer 支持与 completeness discipline；这些 SHA 进入总 seal；其统计地位在 seal 前冻结为 descriptive / secondary。
  - `inclusion ∈ {"no", "post_confirmation_descriptive"}` ⇒ record 必须带 `reason_code` 与机器可读 `claim_restriction`；`confirmation_analyzer` 输出**不得**含任何 "vs Action Cache" 字段；论文只能报告为另行的 post-confirmation / descriptive baseline。
- **顺序**（§7）：若 `yes`，Action Cache 的 G1/G2 位于 P/C seal 与任何 C rollout 之前。

## 4. 接口与集成点

- `estimator_version.digest()` 进入：tgrid 矩阵（记录）、`budget_cost_map_frozen.json`、`budget_outcome_design.json`、power record、`confirmation_seal.json`、`confirmation_discipline.json`、analyzer 输出；任一不一致 ⇒ 拒绝。
- `budget_mixture` 是纯函数模块（无 I/O、不读文件）；`budget_cost_map` 只 import 其 `value_at` 的**成本可行性**部分？——**否**：cost-only 阶段不需要 SR，`budget_cost_map` **不 import `budget_mixture`**（区间只用成本分位），source-lock 钉住。
- `h1_verdict.evaluate_h1_verdict` 是 power MC 与 confirmation analyzer 的唯一裁决实现；`budget_outcome_design` 的 development H1 也调用它（同一函数、同一 R）。
- `tgrid_package` / `rev1_package` 角色解析是数据入口；Phase 0 数据经其 ledger 认领（与 Phase 0 一致）。

## 5. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 分式 estimand 改变 Phase 0 结论方向 | 设计漂移 | §6-1 真实数据回归：l10 H1 正（0.0816）、spatial H1 负必须复现；旧值 0.0722 同时钉住 |
| 断点枚举遗漏 | 积分不精确 | 每调用内部不变量 + 冻结 Simpson audit subset + G2 随机 stats；不把 Simpson 放进 800 万 replicate 热路径 |
| 网格臂 pair 碰撞 / ws=0 语义 | 名义臂数 ≠ 实际 | emitter 拒绝重复 pair；ws=0 无键；runner 逐值核 pair |
| `/tmp` 散落数据进入分析 | 不可审计 | B4 package 唯一入口；散落路径拒绝 |
| `[B_L,B_H]` 随候选集变化 | 与 Phase 0 区间不同 | cost-only 机械生成；旧区间只作记录 |
| C roster 超 `M_MAX` | 无法冻结 | fail closed 回 G1 |
| power MC 计算量 / 不可重放 | 数小时 / 无法复核 | 向量化 + multiprocessing；`(N, r)` 决定全部随机性；逐条 digest |
| 非单调 LCB | 选 N 含糊 | §3.4 规则冻结 |
| 生成器跨机不一致 / RNG 未覆盖 | C 不可复现 | §3.7-a 状态机 + e 双机 digest |
| 碰撞 / 生成失败 | 配额不足 | 占用 k、不重采；P 必须 10/10、C 必须 60/60 ok，否则 fail closed（无余量） |
| unseal 早于完整性 | 破坏确认 | B2 discipline 先于 unseal；source-lock |
| Action Cache "yes" 无实现 | seal 空转 | §3.8 分支拒绝 |
| 旧脚本行为漂移 | 冻结产物失效 | §6-0 字节等价回归 |
| dense baseline 吃掉 H1 | 主线止损 | §3.3-4 机械止损 |

## 6. 测试策略（`tests/dispatch_surface/test_rev2_confirmation.py`）

0. **冻结产物字节等价**：Phase 0 两 suite `phase0_summary` / `cost_map` / `phase0_outcome_design` 重跑 `cmp`；Rev 1 两 verdict；`frontier_hull.py` 与 G2 前快照 SHA 相等。
1. **budget_mixture**：(a) LP 枚举 == hull-at-zero（随机 stats 1000 组，含相同 `d`、相同 `s`、退化 pair）；(a′) **旧 Pareto 剪枝必败**：R2-B1 反例 `V(30)` 全臂 0.9607 vs 剪枝后 0.6111；(a″) tie / 数值：exact tie、同 cost 不同 `d/s`、三点近共线、重复断点、interior 单点共最优五类的 canonical basis 与 roster 与输入顺序无关；内部不变量或冻结 audit subset 交叉验证不一致 ⇒ `NumericMismatch`；输入域违规 ⇒ `ValueError`；step envelope 数值案例；(b) 相同 `d` 时退化为直线 hull；(c) 1000 组随机 stats 的解析积分与 adaptive Simpson 差 < 1e-8，另测 development/C audit-index 派生与 power 每 `(N,r)` inner-0 policy；(d) 冻结 l10 9 臂 fixture：H1 0.0816441 / 旧 0.0722133、H2 0.0447783、S0−T 0.0368658（1e-6）；(e) 左端不可行、右端常数、同族差 0、反对称；(f) `difference_extrema`；(g) `active_basis_union` / bitset；(h) estimator digest。
2. **tgrid emitter / runner**：32 / 29；`fh+ws>100` 拒绝；重复 pair 拒绝；ws=0 无 `warm_tiers` 键且 runner 接受、带键拒绝；ws>0 恰一个 tier、`t_fh > t_ws`、`start_t` 非 0.3 拒绝、非 finite 拒绝、pair 与矩阵不符拒绝；nominal cost 公式；表 SHA 不符拒绝；frozen keys；dry-validate；subset 继承；Rev 1 / Phase 0 分支对旧矩阵不变。
3. **tgrid package**：29×300 完整才产 MANIFEST；缺批次 / 混 matrix / 少 cell / finalize 后漂移 ⇒ 拒绝；role 解析；`budget_cost_map` 拒绝散落路径。
4. **discipline（tgrid）**：未登记 run id / 未执行臂 / pair digest 篡改 / yaml 篡改 ⇒ 拒绝。
5. **budget_cost_map**：cost-only（`success` / `status` 删除或替换后字节等价；AST source-lock 不 import `budget_mixture` / outcome 模块）；候选集精确；threshold 端点按 point cost；A-3′；索引 SHA 复现。
6. **outcome design / roster 选择器**：合成 stats 的活跃基并集、`f_a`、`F_MIN` / `M_MAX` 边界、anchor 不计入、超限 fail closed；止损 verdict 使 power MC / seal 拒绝；输入 SHA 漂移拒绝；不覆盖 Phase 0 文件。
7. **power MC**：`evaluate_h1_verdict` 被 power 与 analyzer 直接调用（同一对象）；`(N, r)` 重放得到相同 verdict 与索引 digest（换并行度不变）；已知强效 fixture 预期通过；**人工非单调 record**（例 LCB 30:0.82 / 40:0.78 / 50:0.85 / 60:0.86 ⇒ 选 50；30:0.85 / 40:0.86 / 50:0.79 / 60:0.81 ⇒ 选 60）机械裁决；全 fail ⇒ `underpowered_stop`；CP 下界数值断言（含 `k=0`、`k=n`）；formal 常量不可被 CLI 覆盖，`--smoke` 产 `smoke=true`。
8. **v 离线指标**：合成表上 SV / S0 的 OOF pinball 差已知值；两 horizon 分别输出；fold-local 训练（fold 内行不进训练，断言）；label 只允许 `oof` / `calibration_resubstitution`；输出不含 "held-out coverage"（字符串断言）。
9. **confirmation task plan / discipline**：task plan 无 `seal_sha256` 字段、先于 seal 生成且 SHA 被 seal/ledger 双绑定；人为构造 task-plan↔seal 循环 schema 必拒。七类 completeness 负例 + `orig_init_state_idx` 非 null + plan 外 uid / plan digest 与 manifest 不符 / row run_id 所属 ledger 的 seal 或 task-plan SHA 不符必拒；真实 `EpisodeTask → wire → worker stub → driver rows/journal → discipline` 路径通过；AST 不访问 `success/status`。
10. **generator**：seed 派生确定性（含 authority > 2^32−1 的真实案例：三处 `seed32` 合法且可重放）；attempt 编号（`a=0` base，`a=1..MAX_RETRIES=4` retry，总 ≤ 5）；P 10/10 与 C 60/60 的 fail closed；retry 状态机（合成 env stub：第 1 次异常、第 2 次成功 ⇒ attempts 记录）；重试耗尽 ⇒ `failed` 占位；碰撞 ⇒ `collision` 占位；prefix 规则；任一非 ok ⇒ fail closed；manifest 字段齐全；跨机 digest 比对逻辑；`.init` 与 `load_init_states` 往返。真实 LIBERO round-trip smoke 在 Verify 阶段于 timan107 / weilandserver 执行（非单元测试）。
11. **seal / unseal / Action Cache**：缺任一 digest ⇒ seal 拒绝；`inclusion=yes` 无 package ⇒ 拒绝；`no` 缺 reason / claim_restriction ⇒ 拒绝；analyzer 输出无 Action Cache 字段（键名断言）；未 unseal 时 analyzer 拒绝 outcome、允许 cost-only；unseal 需 discipline 通过 + `roster_complete`。
12. **confirmation analyzer**：H1 唯一 primary；H2 / S0−T 无 pass 字段；support / miss gate；studentized max-t band 数值案例；estimator digest 不符拒绝。
13. **data_authority**：新记录 validate；旧记录 integrity 不变。
14. **source-lock**：`budget_cost_map` / `confirmation_discipline` 不访问 outcome；development 脚本 import graph 不含 C 路径常量。

回归：全量 `pytest tests/`（`--ignore` 既有 `test_prebuilt_matrix_backend.py` 两个已知失败）+ ruff + `registry validate`。

## 7. 验收与放行顺序（§12.7 + B3/B4）

1. **G1 APPROVED 时**：协议 §13 的规则（estimator、grid、interval 机械式、roster selector、止损、power、H1、fresh discipline、Action Cache schema / 分支）改标 FROZEN 并记录 plan SHA + review SHA；之后 → Code → G2；G2 前：dry validation（tgrid 矩阵）、source-lock、§6-0 字节等价、§3.1 回归值。
2. owner 放行 **tgrid development rollout**（l10，29 臂 × 300 = 8700 ep ≈ 6.4 h；weilandserver 4 replica / timan107 48 worker；可分批，同一矩阵 / ledger）→ **`finalize_tgrid_package`**（29×300 完整）→ data_authority。
3. `budget_cost_map`（l10：Rev 1 + Phase 0 + tgrid package；spatial：Rev 1 + Phase 0）→ 记录 SHA → `budget_outcome_design` → 若 l10 H1 q05 ≤ 0 ⇒ **止损**。
4. 通过 ⇒ C roster 冻结（超 `M_MAX` ⇒ 回 G1）→ power MC → N 机械选出（60 不过 ⇒ 交 owner）。
5. `v_offline_metric`（可与 3 并行；descriptive）。
6. owner 签署 Action Cache decision record（schema / 分支已在 G1 冻结）；**若 `inclusion=yes`：Action Cache 独立 G1/G2 必须先于第 7 步**；写 `amendment_result` artifact（tgrid package SHA、`[B_L,B_H]`、roster、N、power record SHA 等**输出值**）——§13 规则本身自 G1 APPROVED 起已 FROZEN，此处不改规则；BLOCKING_TODO P0-A/B/D 证据勾选。
7. 生成 P / C（weilandserver）→ timan107 重生成核 digest → round-trip smoke → P pilot（100 ep）→ `build_confirmation_task_plan`（无 seal 引用）→ `seal_confirmation`（绑定 task-plan SHA）。
8. owner 明确放行 **l10 C**（roster ≤ 3·M_MAX + 1 臂 × N × 10）→ 全部臂完成 → `confirmation_discipline` → unseal → `confirmation_analyzer` → H1 verdict；fail ⇒ 停线。
9. 之后（不在本计划）：spatial control、真机。

## 8. 与 `ICLR_PAPER_BLOCKING_TODO.md` 的挂接

| TODO 项 | 本计划交付 |
|---|---|
| P0-A 数学与术语 | 协议 §13 amendment（G1 Round 3 已冻结）+ `estimator_version` payload；TODO §1.1–§1.3 已改为分式混合构造 |
| P0-B 统计协议与代码 | §3.1 / §3.3 / §3.4 / §6-1,5,6,7；TODO P0-B 中"upper concave hull + 右端水平"三条已改写 |
| P0-C 论文与图表 | 不在本计划；analyzer 输出 `value_envelope`（measured points、分片、活跃基、`[B_L,B_H]`）作为 figure exporter 唯一数据源 |
| P0-D 结果边界与 baseline | §3.2 dense grid + package、§3.3-4 止损、§3.6 H1-only、§3.8 decision record |
| P0-E 完成判据 | §6-0/1、§7-3/4/6；G2 清单列 reviewer 三项独立检查 |

## 9. Review Log

### G1 Round 1 — Review Authority（Codex，2026-08-29）

**Verdict: CHANGES REQUESTED（G1 未通过）。**

范围、两阶段解封、dense-threshold 止损、H1-only confirmation、fresh C 与旧产物不可覆盖等大方向与 §12 共识一致；32 个合法 threshold cell / 29 个新臂的枚举也正确。但以下 7 项会改变 estimand 的合法性、confirmation 的可审计性或预注册的唯一性，须在 Code 前修完。B1 是数学定义级 blocking；若不修，论文里“episode-level randomized mixture”的主曲线并不是所声明的可达集合。

#### G1R1-B1 — `ratio-of-sums` 与直线 upper hull 不相容；当前 `V_F(B)` 不是合法 episode-mixture envelope

本项目已经由 owner 冻结单臂成本为 decision-weighted ratio-of-sums。对 arm `i`，令每 episode 的均值为总解析成本 `t_i`、决策数 `d_i`、成功率 `s_i`。按 episode 以概率 `p_i` 随机选择 operating policy 时，真实量是

```text
C(p)  = sum_i p_i t_i / sum_i p_i d_i
SR(p) = sum_i p_i s_i
```

而不是 `C(p)=sum_i p_i (t_i/d_i)`。只有所有臂的 episode 决策数相同，当前 `(cost, SR)` 两点间的直线才等于 episode-level randomized mixture。这里该条件明显不成立：归档 l10 development 的各臂总 decisions 为 17,963–26,206，成功即提前结束也从代码路径上证明了 arm 间 episode 长度可变。

Review Authority 用冻结 l10 full-sample sufficient statistics 做了独立诊断：在旧区间上，当前直线 hull 的 H1 plug-in 是 `0.0722133`；按上式枚举单臂/两臂 episode mixtures 后做高密度数值积分约为 `0.0816441`，相差约 `+0.00943`（0.94 pp）。这只是 review diagnostic、不能写成新结果，但足以证明误差不是纯符号问题。

**Required fix：**

1. 新 evaluator 的输入不能只剩 `(ratio_cost, SR)`；每臂、每 replicate 必须保留至少 `(cost_sum, decisions, successes, episodes)`。
2. 对每个预算 `B` 共用唯一可达值求解：
   `max_p sum p_i s_i`，约束 `sum p_i(t_i-B d_i) <= 0, sum p_i=1, p_i>=0`。有限臂下一维预算的最优解至多使用两个臂，可用单臂/两臂机械枚举，无需通用黑箱 LP。
3. envelope 不再笼统称为 `(cost, SR)` 的 upper concave hull；两臂混合段通常是分式曲线。积分须冻结为解析的分式积分 + 交点枚举，或带预注册绝对误差界的确定性积分；所有 H1/H2/A-2/LOTO/per-task/power/C analyzer/figure exporter 共用同一纯函数与 digest。
4. C-roster 的“影响区间顶点/频率”随之改成“在区间内参与最优 LP basis 的 arm/频率”；full-sample 必选、bootstrap `f_a` 与 `M_MAX` 规则仍可保留。
5. 同步修订协议 §13、`ICLR_PAPER_BLOCKING_TODO.md` §1 的“直线/upper concave hull”构造、文件清单和测试；旧 exact-cost/Phase-0 函数与产物仍原样保留。若坚持直线 hull，唯一自洽替代是把主成本改成 expected total compute per episode，但这与 owner 已冻结的 ratio-of-sums 冲突，本轮不得暗改。

#### G1R1-B2 — `roster_complete` ledger 不能单独授权 unseal

§3.7-f 目前只要求 ledger 声称 `roster_complete`，随后即可生成 `unseal_record`；它没有在 outcome 解封前证明 C 的真实观测网格完整。历史上本线已经出现过 partial grid、stale/fenced attempt、producer schema 分叉与错误 init identity，不能把这些推迟到读 outcome 之后才发现。

**Required fix：**新增/扩展一个 outcome-blind `confirmation_discipline`，在 unseal 前机械证明：

- 精确 roster × 10 task × prefix `0..N-1` 的 Cartesian product；每 cell 恰一个 accepted `(task_uid, attempt, run_id)`，无 off-grid、重复 accepted、stale/fenced/其他 run 混入；
- 所有 arm 对同一 fresh state digest 成对，task/name/index/pool/seal identity 一致；不得复用 official `orig_init_state_idx` 的 0..49 语义冒充 fresh identity；
- arm YAML、matrix/ledger、server bundle、cost authority、journal/per-step producer schema 与 run id 均逐项绑定；每个 accepted episode 的 per-step decisions 与 client `infers` 无条件一致，未知 verdict/warm tier/缺行均拒绝；
- discipline artifact 的输入/输出 SHA 写入 `unseal_record`；上述验证完成前代码路径不得加载 `success/status` 字段。

测试至少加入 partial-complete ledger、重复 accepted、stale attempt、错 pool state、某 arm 少一个 cell、per-step 少行、off-grid cell 七类必拒。

#### G1R1-B3 — Action Cache 的 record 只有 SHA，没有可执行的 inclusion 语义

§3.8 允许 record 写 `inclusion=yes`，但当前 `c_roster.json`、runner、seal 和 analyzer 都只实现 SV/S0/threshold/anchor。这样可能出现“record 说纳入，实际 C 没跑 Action Cache，seal 仍通过”。

**Required fix：**冻结 decision-record schema 与 validator，并写成 seal 的条件分支：

- `inclusion=yes`：本计划的 seal 必须拒绝继续，直至另一个通过 G1/G2 的 Action Cache package 提供 development 选点 artifact、配置/代码/成本映射 digest、C arm roster、同一 fresh-pool 绑定、runner/analyzer 支持及 completeness discipline；这些 SHA 必须进入总 seal。还须冻结其在论文中的统计地位（H1 不变时至少明确是预注册 descriptive/secondary，不能事后升级 primary）。
- `inclusion=no` 或 `post_confirmation_descriptive`：record 必须携带理由码和机器可读 claim restriction；confirmation 输出不得生成“优于 Action Cache”的字段/句式。

§7 顺序应明确：若选择 `yes`，Action Cache 后续 G1/G2 位于 P/C seal 和任何 C rollout 之前，而不是仅签一个模板后直接进入第 7 步。

#### G1R1-B4 — tgrid 原始证据缺少持久、不可变的 finalization package

§3.2 把执行目录放在 `/tmp/dsp_precheck/...`，§2 仅登记一个 `journal` record；但新 grid 是 budget amendment、C roster 和 power 的直接输入。`/tmp` 文件、分批 ledger、YAML 与 per-step 若没有一次性 package，之后无法证明 analyzer 读到的就是已审核的 29 臂执行全集。

**Required fix：**在文件清单和执行顺序加入 tgrid finalizer/package：持久化到 `exp/dispatch_surface/data/...`（或 data_authority 支持的等价不可变位置），manifest 至少绑定 matrix、完整 ledger、全部执行 YAML、split/A′ pool attestation、journal、per_step、launch metadata、roster/pair rollup、cost model、run-id/批次集合及各文件 SHA。`budget_cost_map` 只能从已 finalise package 读取，不接收散落的 `/tmp` 路径；finalizer 必须在精确 29×300 完整性通过后才产 manifest。分批遗漏、不同 matrix 混包和 finalise 后文件漂移须有负例。

#### G1R1-B5 — `v` 的 coverage–resolution 尚未形成可实现的冻结统计量

§3.5 的“`q_hat` 的 resolution（分箱方差）”没有定义分箱对象、边界、tie、权重、目标或方向；τ7/τ10 也没有说明是两个并列 readout、预先聚合还是选择一个。这会让同一批 OOF 数据产生许多合法但结论不同的实现。

**Required fix：**在 G1 写公式而非只写名称：分别冻结 τ7/τ10 的 pinball 定义与符号；fold-local 训练/预测边界；row→episode→task 的权重；bootstrap resampling unit；coverage/calibration curve 的 bin 生成规则（边界只由相应 OOF 预测机械生成、ties/min-count 处理）；resolution/sharpness 的精确定义与“更好”的方向。若不愿新增一个有歧义的 scalar，可只预注册 OOF pinball + 预冻结 bins 的 empirical coverage/sharpness 表，并把 `resolution` 一词删除。所有结果继续保持 descriptive，不设 gate。

#### G1R1-B6 — power MC 尚缺可复现的子流和非单调候选裁决

仅写“outer PCG64 与 inner PCG64 独立”不足以唯一重放 800 个嵌套实验；并且有限 MC 下 `N=30/40/50/60` 的 LCB 可能非单调。§6-6 的“通过率单调于 N”不是一般真理，support gate 与 MC 误差都可使其违反。

**Required fix：**

1. 抽出正式共享纯函数 `evaluate_h1_verdict(sufficient_stats, frozen_design, bootstrap_index)`，power 与 confirmation analyzer 都直接调用；不要用“源码断言/调用计数”代替语义共享。
2. 冻结每个 `(N, outer_rep)` 的 inner SeedSequence/PCG64 派生规则、索引 canonical encoding/digest、并行调度无关性；record 保存 aggregate digest 及可抽查的逐 replicate verdict/reason。
3. 冻结非单调处理。建议选择最小的 `N`，使该 N **及所有更大候选 N** 的预注册 LCB 都 ≥0.80；若序列出现 pass→fail，不能取更小的 pass。也可用预注册 simultaneous binomial lower bounds，但不能看结果后决定。`N=60` 不满足仍按现有止损。
4. 明确 CP lower bound 的 tail、confidence 和有限样本实现；`R_outer=200`、one-sided 95% 可以接受，但必须作为常量而非 CLI 可变 formal 值。测试改为“已知强效 fixture 的预期通过”和“人工非单调 record 的机械裁决”，不要断言任意分布的 power 必单调。

#### G1R1-B7 — fresh-init 的 RNG 注入与 placement retry 仍留给实现者决定

§3.7-a 只给出 seed 哈希，却没有冻结 seed 如何进入 Python/NumPy/LIBERO/robosuite、每个 k 是新建 env 还是连续 reset、何谓“placement retry”、最大次数和 retry seed。任一选择都会改变 C 的采样分布；“实现时写死”晚于 G1。

**Required fix：**在 plan 中钉死逐 state 的生成状态机：环境构造/销毁边界、所有 RNG 的赋值位置、reset API、接受条件（正常 reset 即接受，除非哪一种明确异常/非有限 state）、`R_MAX`、第 r 次 retry 的 seed 派生、失败是否占用 k。manifest 保存每个 k 的 base/retry seed、attempt count、shape/dtype/content digest。`sim.get_state()` 与 runner `env.set_init_state()` 做真实 LIBERO round-trip smoke（至少每 task 一个 state），不能只用合成 47 维数组证明格式。跨机 exact digest 仍保留为额外 fail-closed 检查。

#### Non-blocking corrections（随 Rev 2 一并清理）

1. tgrid runner 校验不能只有 `threshold <= 1.0`：同时要求数值 finite、`t_fh > t_ws`（ws>0）、无额外 tiers、canonical `start_t=0.3`，并与 matrix 的实际 pair/digest 相等；ws=0 必须无 `warm_tiers` 键或为冻结的唯一 canonical 表示，二选一写死。
2. §3.3 的 `F_MIN=0.20`、`M_MAX=6` 本轮可接受，但在 B1 修正后的“active optimal basis”语义下冻结为代码常量；anchor 不计入任一 family 的 `M_MAX`。输出每个 bootstrap replicate 的 active-arm bitset rollup，便于重放频率。
3. secondary `B_1/B_2` max-t band 虽不 gating，也应写明 centered/studentized 与否、tail、critical value、缺支撑处理；否则降为逐点 effect + marginal CI，并明确 exploratory。
4. power 的 effect-replication assumption 还应注明：它是“在 development 选定 roster 条件下，以官方 A′ 经验联合分布近似 fresh C”的设计假设；P 上只跑 anchor 只能验证 full-inference 难度漂移，不能验证 H1 effect transport。

**Re-review entry condition：**执行方在本节后逐条响应 B1–B7，并提交自洽的 Rev 2 plan body；B1 修订后的 protocol/TODO diff 必须与 plan 同轮给 reviewer。G1 通过前不得改 `exp/`、不得 emit tgrid matrix、不得 rollout 或 materialize P/C。

### G1 Round 1 — Executor 响应（2026-08-29）

**B1（接受，定义级修订）**：独立复算与 Review Authority 一致——用冻结 l10 数据（Rev 1 包 + Phase 0）各臂 per-episode `(t, d, s)`：`dsp_s0_p80 59.9 dec/ep … dsp_t_fh70_ws10 87.4 dec/ep`（总 decisions 17,963–26,206），直线 hull 的 H1 plug-in = 0.0722133，按 `max_p Σ p s  s.t. Σ p (t − B d) ≤ 0` 枚举单臂 / tight 两臂并在 `[41.7, 47.7]` 上 400k 点数值积分 = **0.0816441**（+0.00943）；H2 0.0407 → 0.0448、S0−T 0.0315 → 0.0369，同向。§3.1 改为 budget-mixture LP：sufficient stats `(T, D, S, E)`、hull-at-zero 等价、断点 = 单臂成本 ∪ 三点共线线性根（完备性论证见 §3.1）、分片解析积分（常数 + 线性分式闭式）、自适应 Simpson 只作交叉验证测试；A-2 极值加片内闭式驻点；C roster 改为 active optimal basis 并集 + 频率；协议 §13 与 BLOCKING_TODO §1.3 / P0-B 已同轮修订（标 PROPOSED / 待 G1 R2）；`frontier_hull.py` 与 Phase 0 产物不动；单臂成本仍为 ratio-of-sums，未改成本轴。
**B2（接受）**：新增 `confirmation_discipline.py`（§3.6-b）：Cartesian 完整性、fresh identity（runner 注入 `pool_id / seal_sha256 / fresh_state_sha256`，`orig_init_state_idx` 必须为 null）、绑定、per-step 与 `infers` 一致、AST 不访问 outcome；discipline SHA 进 unseal record；七类负例 + 一类 identity 负例入 §6-9。
**B3（接受）**：`action_cache_decision.py` schema / validator（§3.8）；`inclusion=yes` ⇒ 本计划 seal 拒绝直至独立 G1/G2 的 Action Cache package 提供全部 digest 与 discipline；`no` / `post_confirmation_descriptive` ⇒ reason code + claim restriction，analyzer 输出无 Action Cache 字段（键名断言）；§7 顺序写明 `yes` 时 AC G1/G2 先于 seal / C。
**B4（接受）**：`finalize_tgrid_package.py` + `tgrid_package.py`（§3.2-f）：29×300 完整才产 MANIFEST，持久化到 `exp/dispatch_surface/data/tgrid_dev/libero_10/` 并入 data_authority；`budget_cost_map` 只经 role 读 package；负例：缺批次 / 混 matrix / 少 cell / 漂移。
**B5（接受）**：§3.5 改为两条并列 readout（horizon 7 / 10）的 OOF pinball（公式、fold-local 边界、row→episode→task 权重、episode paired bootstrap R=10000 seed 20260831、符号）+ 无 bin 的 coverage / sharpness 表；删除 "resolution"；label 约束与字符串断言保留。
**B6（接受）**：`h1_verdict.evaluate_h1_verdict` 唯一实现，power MC / analyzer / development H1 直接调用；`(N, r)` 的 outer / inner `SeedSequence(20260830|20260829, spawn_key=(N, r))`、canonical 索引 digest、逐条 record、aggregate digest；裁决改为"最小 N 使该 N 及所有更大候选 LCB ≥ 0.80"，pass→fail 不得取小；CP 单侧 95% 下界 `beta.ppf(0.05, k, n−k+1)` 常量化、`R_OUTER=200` 非 CLI；测试改为强效 fixture + 人工非单调 record。
**B7（接受）**：§3.7-a 逐 state 状态机：每 attempt 新建 `OffScreenRenderEnv`、`random/np.random/env.seed` 三处赋值、`env.reset()`、`get_sim_state`/`sim.get_state()`、接受条件（无异常 ∧ shape ∧ finite）、`R_MAX=5`、retry seed 派生、失败 / 碰撞占用 k 不重采；manifest 逐 k 记录；真实 LIBERO round-trip smoke（`set_init_state` → `sim.forward` → 相等；client 路径 1 步）；跨机 digest 保留。
**Non-blocking 1–4（接受）**：tgrid runner 校验加 finite / `t_fh > t_ws` / 无额外 tier / canonical `start_t` / 与矩阵 pair 相等，`ws=0` 无 `warm_tiers` 键；`F_MIN=0.20` / `M_MAX=6` 在 active-basis 语义下冻结为常量，anchor 不计入，bitset rollup 输出；secondary band 冻结为 studentized max-t、两侧 95%、miss 不进 band 且单独报告、exploratory；power 的 effect-replication 假设与 P 只能检验难度漂移已写入 §3.4。
**同轮交审的 diff**：`logs/dispatch_surface_rev2_protocol_draft.md` §13（PROPOSED）；`docs/iclr/ICLR_PAPER_BLOCKING_TODO.md` §1.3 规则 3–5 与 P0-B 三条（标 "待 G1 R2"）。**未改 `exp/`、未 emit、未 rollout、未 materialize P/C。**

G1 gate reached. Please initiate a separate Review Authority session to audit logs/dispatch_surface_rev2_confirmation_plan.log.md.

### G1 Round 2 — Review Authority（Codex，2026-08-29）

**Verdict: CHANGES REQUESTED（G1 仍未通过）。**

R1 的 B2–B6 主体修复成立；B1 的 LP、tight-pair 公式、单臂成本边界与分式积分方向也成立。Review Authority 另用随机 sufficient statistics 检查了“单臂成本 + 三点共线根”的分片主张，未找到片内 active basis 改变的反例。因此本轮不推翻 budget-mixture LP。剩余问题集中在 **LP 与权威文档仍不一致、active-basis 的离散选择未冻结、fresh generator 的 seed 实际不可执行、confirmation identity 没有 producer、以及 protocol 被计划成看完 tgrid 结果才冻结**。

#### G1R2-B1 — TODO/协议仍保留了对新 LP 不合法的 Pareto pruning；主成本泛函记号也未同步

`ICLR_PAPER_BLOCKING_TODO.md` §1.1/§1.2 仍把横轴写成 `E[c(pi)]`，但本轮真正冻结的是

```text
C(pi) = E[episode total compute] / E[episode decisions]
```

两者在 episode mixture 下不是同一个量。更严重的是 TODO §1.3-2 仍要求“相同成本只保留高 SR；删除 Pareto-dominated 点”，§2/P0-E 也仍写 upper-concave / 被支配点检查。该规则在新 ratio-of-sums LP 下是错误的：marker 上被支配的 arm 仍可能因 `d_i` 不同而成为最优 mixture basis。

机械反例（budget `B=30`）：

```text
arm A: c=10,  d=1,   s=0.5
arm B: c=20,  d=100, s=0.4   # 在 (c,s) 上被 A 严格支配
arm C: c=100, d=1,   s=1.0
```

混合 A/C 的最优 SR 为约 `0.6111`，而混合 B/C 为约 `0.9607`；预删 B 会直接算错 `V_F(30)`。同成本、较低 SR 的臂也可能因更大的 `d_i` 产生同一现象。因此新 evaluator **不得按 standalone `(c,s)` dominance 或 equal-cost marker 去重**；所有冻结 operating arms 的 `(T,D,S,E)` 都必须进入 LP。

**Required fix：**

1. TODO §1.1/§1.2 把可达集合的横轴改成明确定义的 `C(pi)=E[T]/E[D]`，后文统一写 `C(pi)≤B`，不得再用会被理解为 `E[T/D]` 的 `E[c(pi)]`。
2. 删除/改写 TODO §1.3-2、P0-A 的 “upper concave envelope”、P0-E 的“没有保留被支配点”等残留；明确 standalone Pareto 标记只可作图形描述，不能作为 LP 输入剪枝。增加上述“旧 Pareto prune 必败”的回归。
3. 协议 §13.1 的适用范围改为 **budget amendment development + power + C**；当前标题“仅对 C 生效”与 plan §3.3 的 development 止损相冲突。旧 Phase 0 artifact 仍按旧 estimator 冻结即可，不等于新 estimator 只能用于 C。
4. 图表/输出术语从“efficient hull vertices”改为 measured policies / active mixture bases；若一个 standalone-dominated arm 进入 active basis，必须如实保留并可追溯。

#### G1R2-B2 — active basis 的 tie、零测度事件与浮点退化会直接改变 C roster，目前未冻结

32 个 threshold arm 的 SR 是 `1/300` 粒度，full-sample 同 SR/tie 很可能发生。§3.1 只说“取最大”，§3.3 又把 active basis union 作为 C 必选集合，却没有定义：多个 basis 同值时选谁、只在单个 breakpoint 共最优的 arm 是否入选、近共线根如何去重、解析积分与数值交叉验证不一致时怎么办。不同遍历顺序会产生不同 `c_roster.json`，这不是 G2 才能决定的实现细节。

**Required fix：**冻结以下 canonical policy：

- arm 的总序来自 frozen roster id；`value_at` 在数值同值时的 basis tie-break（建议：较少臂优先，再按 canonical arm tuple）；
- `active_basis_union` 只计每个**正长度开区间**的 canonical active basis，并另外记录 `B_L/B_H` 的 canonical basis；不得把 interior measure-zero 共最优的全部替代 basis 自动塞入 roster；
- `f_a` 对每 replicate 使用同一规则；bitset 的 arm 顺序、编码与 digest 明定；
- float64 输入域校验（`T,D,E>0`、`0≤S≤E`）、breakpoint 过滤/排序/近重复容差、近零分母与 log domain 的处理。若解析积分与 frozen adaptive cross-check 相差超过 `1e-8`，该 family/replicate **fail closed**，不得退回近似值静默继续。

测试加入 exact tie、同 cost 不同 `d/s`、三点近共线、重复 breakpoint、只在 interior 单点共最优五类，并断言 roster 与输入 arm 顺序无关。

#### G1R2-B3 — fresh seed 是 64 位，不能直接传给 `np.random.seed`；P 配额和 attempt 编号也未闭合

§3.7-a 从 SHA 前 8 字节产生 `[0,2^64−1]` 的整数，随后调用 legacy `np.random.seed(seed)`；该 API 只接受 `[0,2^32−1]`，几乎所有生成 seed 都会在第一次 attempt 直接抛 `ValueError`。此外“`r=0` 即 base”与 `...|retry|0` 是两个不同哈希，`R_MAX=5` 是 5 次总 attempt 还是 base + 5 retry 也不明确。C 明确要求 60 个 ok，P 却没有写 `10/10 ok` 才可 pilot。

**Required fix：**

- 保留完整 seed digest 作为 authority，但机械派生各 API 的合法 seed（例如固定 `uint32_seed = SeedSequence(full_entropy).generate_state(1, dtype=uint32)[0]`）；Python、NumPy 与 `env.seed` 各自记录实际入参，跨机重放使用完全相同映射；
- attempt 0 明确使用 base seed；retry 仅为 `r=1..R_MAX−1`（若 `R_MAX` 表示总次数），或把常量改名 `MAX_RETRIES` 并明确总次数，二选一；manifest 不得把 base 与 `retry|0` 混为一谈；
- P 必须每 task `10/10 status=ok` 才 materialize/pilot，任一 failed/collision 即 `generator_validation_failed`；C 的 `60/60 ok` 规则保留，并把“含余量”误称删除（当前生成恰好 60 且要求 60 ok，没有余量）。

加入真实 seed-domain 单测：选择一个高于 `2^32−1` 的 authority seed，验证三处实际 seed 均合法且可重放。

#### G1R2-B4 — plan 声称 run_precheck “每行注入” fresh identity，但现有 producer 不传播这些字段

`run_precheck.py` 的 strategy 可以把 `pool_id/seal_sha256/fresh_state_sha256` 放进 `EpisodeTask.extra`，但当前 `examples/libero/episode_runner.py::_hit_row` 只输出固定字段；`client_timing` 行也不传播 `extra`。conductor journal 同样不保存 task.extra。文件清单没有 producer 改动，因此 §3.6-b 要求的逐行 identity 在真实路径上构造不出来。

**Required fix：**二选一并写入文件清单/测试：

1. additive 修改 `examples/libero/episode_runner.py`（以及共享 producer helper，如需要），从 authoritative `EpisodeTask.extra` 把冻结的三字段传播到 verdict/client-timing 行，拒绝 worker 自行覆盖；或
2. per-step 只保留 `task_uid`，由 outcome-blind discipline 对 frozen confirmation task plan 做唯一 join，再验证 task plan 中的 fresh identity；此时删除“每行直接携带”的虚假要求，并把 task-plan SHA 纳入 seal/discipline。

无论选哪条，都要加真实 `EpisodeTask → protocol serialization → worker → EpisodeResult.per_step_rows → driver stamping → discipline` round-trip 测试，而不只是手造 fixture。`orig_init_state_idx=null` 也须沿该路径验证；subset loader 必须继续只用 `episode_idx`。

#### G1R2-B5 — protocol 不能在看到 dense-tgrid outcome 后才从 PROPOSED 改为 FROZEN

§7-6 把“协议 §13 改标 FROZEN”放在 tgrid rollout、budget outcome design、C roster 和 power 之后。虽然 tgrid 是 development，但 active-basis 选择器、止损、power 和 fresh-C 规则正是本次 G1 要在看新 grid outcome **之前**冻结的内容；若整节一直标 PROPOSED，形式上仍允许结果出来后改协议。

**Required fix：**G1 通过时立即把 §13 中 estimator、grid、interval、roster selector、止损、power、H1 与 fresh discipline 的**规则**标为 FROZEN，并记录 plan/review SHA。之后只允许机械填入 tgrid package SHA、`[B_L,B_H]`、roster、power N 等输出值，以单独 amendment-result artifact 追溯，不得改规则。Action Cache 的 decision-record **schema/分支**本轮冻结，owner 的 `inclusion` 取值稍后签署；不得以该取值未定为由让整个 §13 保持可修改。

#### Non-blocking corrections

1. §3.2-c nominal cost 公式中的 `fh/ws` 是整数百分数，应明确写 `(fh/100)c_FULL + (ws/100)c_WARM + (1−(fh+ws)/100)c_MISS`，避免实现按 20/40 而非 0.2/0.4。
2. Action Cache `inclusion=no` 时 `config_digest/code_digest/development_selection_protocol/c_pool_binding` 应明确允许 canonical `null`；`yes` 时必须是非空 SHA/对象。否则当前 schema 对“不纳入”没有可构造的合法 record。
3. 建议 amendment analyzer 同时输出非 gating 的 **measured-policy-only** budget step envelope `max{s_i:T_i/D_i≤B}`。它不替代主 LP，但能显示论文正向结论是否主要来自 decision-count 加权的 synthetic mixture；无需任何额外 rollout，是应对 reviewer 对分母效应质疑的最好敏感性证据。

**Re-review entry condition：**Rev 3 同步修改 plan body、协议 §13 与 TODO；逐条响应 B1–B5。G1 通过前继续禁止改 `exp/` / `examples/`、emit、rollout 和 P/C materialization。

### G1 Round 2 — Executor 响应（2026-08-29）

**B1（接受；反例已独立复算）**：`V(30)`：全臂 LP = 0.9607，按 `(c,s)` 剪掉被支配的 B 后 = 0.6111——被支配臂因 `d` 更大成为最优基。§3.1 明确**不做任何 Pareto / 同成本剪枝**，全部 `(T,D,S,E)` 进 LP，standalone dominance 只作描述性标记；记号改为 `C(π) = E[T]/E[D]`，术语改为 measured policies / active mixture bases；协议 §13.1 适用范围改为 development amendment + power + C；TODO §1.1/§1.2/§1.3-2/P0-A/P0-B/P0-E 残留同步改写；§6-1 加“旧剪枝必败”回归。
**B2（接受）**：§3.1 冻结 canonical tie / 数值策略：roster 总序、`value_at` 同值 `1e-12` 与 tie-break（臂数少者 → canonical tuple）、输入域校验、断点 `1e-9` ms 合并与近零分母、正长度开区间才计活跃基且端点 basis 另存、bitset 位序 / 编码 / digest、分片解析与自适应 Simpson 差 > 1e-8 即 `NumericMismatch` fail closed；§6-1 加五类 tie 测试与顺序无关断言。
**B3（接受；`np.random.seed(2**32)` 实测 `ValueError`）**：256 位 authority → `SeedSequence` → `seed32 = generate_state(1, uint32)[0]` 供三处 API，各自记录；attempt `a = 0` 为 base、`a = 1..MAX_RETRIES(=4)` 为 retry（总 ≤ 5），manifest 逐 attempt 记 authority 与 `seed32`；P 10/10 `ok`、C 60/60 `ok` 才 materialize，删除“含余量”措辞；加 authority > 2^32−1 的真实 seed-domain 单测。
**B4（接受；采用方案 2）**：核对 `episode_runner._hit_row` 与 `client_timing` 行均为固定字段、`EpisodeTask.extra` 不传播，故放弃“每行注入”。改为 `confirmation_task_plan.json`（arm × task × prefix → pool / fresh digest / seal）+ `task_uid` 唯一 join，plan SHA 进 seal 与 discipline；`orig_init_state_idx` 由 confirmation strategy 置 `None`（已核 `conductor/protocol.py` wire 为 `dataclasses.asdict` ↔ `EpisodeTask(**d)`，`None` 可透传；dataclass `int` 注解运行期不强制）；加真实 conductor 路径 round-trip 测试；producer 不动。
**B5（接受）**：§7-1 改为 G1 APPROVED 即冻结 §13 全部规则（记录 plan SHA + review SHA）；§7-6 只写 `amendment_result` 输出值；Action Cache schema / 分支本轮冻结，owner 的 `inclusion` 取值稍后签署；协议 §13 抬头同步。
**Non-blocking 1–3（接受）**：nominal cost 公式改为 `/100`；AC schema 对 `no` / `post_confirmation_descriptive` 允许四项 canonical `null`、`yes` 必须非空；amendment analyzer 与 C analyzer 同时输出 **measured-policy-only step envelope** `max{s_i : T_i/D_i ≤ B}` 的 AUC 差与 CI 作为非 gating 敏感性。
**同轮 diff**：协议 §13（范围 / 记号 / 无剪枝 / tie 策略 / 冻结时点 / step envelope / seed 与配额 / task-plan join / AC schema null）；TODO §1.1/§1.2 记号、§1.3-2、§1.3-6/8、P0-A、P0-B、P0-E、图注措辞。**未改 `exp/` / `examples/`、未 emit、未 rollout、未 materialize P/C。**

G1 gate reached. Please initiate a separate Review Authority session to audit logs/dispatch_surface_rev2_confirmation_plan.log.md.

### G1 Round 3 — Review Authority（Codex，2026-08-29）

**Verdict: APPROVED（经 reviewer 直接修订后放行 Code；尚未放行 emit / rollout / P、C materialization）。**

Round 2 的 B1–B5 均已实质闭合：budget-mixture LP 不再按 standalone `(cost, SR)` 做 Pareto 剪枝；canonical basis / tie / active-union 与数值域已冻结；fresh seed 映射、attempt 语义及 P/C 完备配额已闭合；confirmation identity 改为 `task_uid` 对权威 task plan 的 outcome-blind join；协议 §13 已在读取 dense-tgrid outcome 前冻结。Review Authority 独立复核后又直接修正了四个会影响实现或可执行性的残余：

1. **解除 task-plan / seal 哈希环。** `confirmation_task_plan.json` 现在明确先于 seal 构造且不含 seal SHA；seal 绑定 task-plan SHA，launch ledger 同时绑定二者，discipline 分别核验 row `task_uid`、row `run_id` 与 ledger 的双摘要。此前响应中“task plan 含 seal”的措辞作废。
2. **删除 producer 注入残留。** 文件清单与 runner 设计统一采用 R2-B4 方案 2：`episode_runner.py` 不改，per-step row 不伪称携带 fresh identity；runner 从 task plan 构造 `EpisodeTask(orig_init_state_idx=None)`，discipline 通过 task plan、C manifest 与 ledger 完成身份认证。
3. **把 Simpson 从 formal MC 热路径移出。** 解析积分仍是唯一 estimand 实现；adaptive Simpson 只审计 G2 的 1000 组随机/退化案例、development/C 的 full sample 与 digest 派生至多 100 个 bootstrap replicate、以及 power 每个 `(N, outer_r)` 的 inner-0。任一差值 `>1e-8` 使相应 artifact / power record fail closed，且禁止用数值近似替换解析值。这样保留独立数值认证，同时避免在约 800 万次 formal inner replicate 上运行递归积分。
4. **清理冻结状态与术语漂移。** 当前权威正文统一使用 `MAX_RETRIES=4`（总 attempt ≤ 5）、P `10/10` 与 C `60/60` 完备而非“余量”，§13 标为 FROZEN；TODO 的数值审计与协议同步。历史 Review Log 中的 `R_MAX`、`PROPOSED` 等仅作为原轮次证据保留，不具当前规范效力。

**G1 freeze boundary：**暂存区中的 Executor Rev 3 是送审快照；上述 reviewer 修订保留在 working tree、未加入暂存。进入 Code 前，Execution Authority 应先审阅并接纳该未暂存 diff，再以最终字节生成 freeze record。Code/G2 可以开始；G2 通过且 §7 的先决条件满足前，仍禁止 emit tgrid matrix、任何 rollout、以及 P/C materialization。

## 10. Code 记录（2026-08-29，Execution Authority；G2 交审快照）

**范围内交付（未 emit、未 rollout、未 materialize P/C；`examples/` 与 `src/` 未改）**

| 文件 | 内容 |
|---|---|
| `exp/dispatch_surface/analysis/budget_mixture.py`（新） | `ArmStats(T,D,S,E)`、向量化 `value_at`（LP 枚举，canonical tie-break）、`hull_at_zero`（测试参照）、`breakpoints`（全量 O(K³) 参照）、**事件扫描分片** `pieces`（每片只解与当前活跃基相关的共线根 + 单臂成本；片端与两处内点用 `value_at` 复核，`NumericMismatch` 即中止）、分式闭式积分 `auc_norm`、`auc_with_support`（左端不可行记 −1）、`difference_extrema`（断点 ∪ 片内闭式驻点）、`active_basis_union` / `family_replicate` / bitset、`step_*`（measured-policy-only 敏感性）、`simpson_auc_norm` + `audit_family`（审计，非估计量）、`standalone_dominance`（描述性，不剪枝） |
| `analysis/estimator_version.py`（新） | `BUDGET_MIXTURE_V1` payload + digest；`EXACT_COST_V1` 登记 |
| `analysis/h1_verdict.py`（新） | `FrozenDesign`、`evaluate_hypothesis` / **`evaluate_h1_verdict`（唯一裁决实现，power MC 与 analyzer 直接调用同一对象）**、`audit_replicate_indices`（`sha256(estimator|input)` 派生 ≤ 100 个审计 replicate） |
| `phase0_roster.py`（追加） | `THRESHOLD_GRID_FH/WS`、`tgrid_cells`（32/29）、`tgrid_arm_id`、`tgrid_roster_spec` + digest、`F_MIN=0.20`、`M_MAX=6` |
| `emit_precheck_yamls.py`（扩展） | `emit_tgrid`：表 SHA 绑定 `fit.sv.input_digests.table`、同一 `derive_thresholds`、`ws=0` 无 `warm_tiers` 键、`t_fh>t_ws`、pair digest / rollup、重复 pair 拒绝、nominal cost `/100`、Rev 1 三臂 realized−nominal（cost-only 行） |
| `run_precheck.py`（扩展） | `LAYER_TGRID`（`validate_tgrid_matrix_artifacts` / `validate_tgrid_arms` / 契约取自包内 `artifact.dsp_sv` / `TGRID_FROZEN_LAUNCH_KEYS`）；`LAYER_CONFIRMATION`（`run_confirmation`：seal / task plan / `validate_fresh_pool` / `validate_confirmation_arms`（artifact 按**内容**绑定）/ `ConfirmationStrategy`（`orig_init_state_idx=None`）/ ledger 冻结 `seal_sha256`、`confirmation_task_plan_sha256`、`pool_digest`、`N`、`estimator_version`）；Rev 1 / Phase 0 分支未动 |
| `analysis/phase0_discipline.py`（追加） | `validate_tgrid`（与 Phase 0 同强度） |
| `tgrid_package.py` / `finalize_tgrid_package.py`（新） | 29×300 完整性（`assert_grid_complete`）→ 不可变 package + MANIFEST（role→member+SHA）；`verify_package` 核 matrix/ledger/yaml 绑定与完整性 |
| `analysis/budget_cost_map.py`（新） | cost-only Stage 1：Rev 1 包 + Phase 0 + tgrid 包；SV/S0 isotonic 端点、threshold 端点 = 最低/最高 point cost；`[B_L,B_H]`、`B_1,B_2`（协议 §3.3 同式）；A-3′；不 import `budget_mixture` / outcome 模块（AST 锁） |
| `analysis/budget_outcome_design.py`（新） | Stage 2：全部冻结 digest 复核后才读 SR；强制 `R=10000, seed=20260829`；三族全量点 envelope；H1（`evaluate_h1_verdict`）/ H2 / S0−T（描述，无 pass 字段）；A-2；LOTO；per-task；`B_1/B_2` ΔV；step-envelope；**C roster 选择器**（开区间活跃基并集 ∪ 端点 basis ∪ `f_a ≥ F_MIN`，每族 ≤ M_MAX，anchor 固定；超限 `roster_overflow`）；**止损** `stop_before_C`；输出 `budget_outcome_design.json` + `c_roster.json`（含各臂 yaml/artifact/pair 的内容 SHA） |
| `analysis/confirmation_power_mc.py`（新） | `N_CANDIDATES=(30,40,50,60)`、`R_OUTER=200`、`R_INNER=10000`、`SeedSequence(20260830|20260829, spawn_key=(N,r))`、canonical 索引 digest、逐条 record + aggregate digest、Clopper–Pearson 单侧 95% LCB、"最小 N 且所有更大候选达标"、`underpowered_stop`、inner-0 Simpson 审计、half-effect proxy（非 gating）、`--smoke`（R_OUTER=20，record 标 `smoke`）；测试用 `nonformal_r_inner` 只经显式 payload，结果标 `formal=false` |
| `analysis/v_offline_metric.py`（新） | OOF（复用 `assign_folds/fit_fold_models/oof_predictions`，fold map 与归档 record 逐 episode 相等才继续）；h7/h10 两条 readout；pinball（level = 1 − quantile_alpha）、coverage、sharpness；paired episode bootstrap R=10000 seed 20260831；label `oof`；输出禁含 "held-out coverage" |
| `generate_fresh_inits.py`（新） | 逐 state 状态机（authority → `SeedSequence` → `uint32` 三处赋 seed；每 attempt 新建 env；接受 = 无异常 ∧ shape ∧ finite；`MAX_RETRIES=4`；failed/collision 占位）；互斥（官方 50 + 另一池）；P 10/10 / C 60/60 才 materialize；`compare_manifests`（跨机） |
| `build_confirmation_task_plan.py`（新） | `task_uid → {arm, task, prefix, pool_id, fresh_state_sha256}`，无 seal 字段（`assert_no_cycle`），SHA 由 seal 与 ledger 绑定 |
| `seal_confirmation.py`（新） | seal（拒绝：止损/overflow verdict、smoke power record、N 未机械选出、pilot 失败或非一次性、跨机未验证、AC 分支不过、roster/yaml/artifact 内容漂移）；unseal（需 discipline 通过 + ledger roster_complete + seal 一致） |
| `analysis/confirmation_io.py` / `confirmation_discipline.py`（新） | outcome-blind：Cartesian 完整性、`task_uid`→task plan 唯一 join、run_id→ledger、ledger 的 seal/task-plan 双摘要、pool attestation、`orig_init_state_idx` 必须 null、decision 行数 == `infers`、verdict/tier 形状；源码不含 `success`/`status` 字面量（AST 锁） |
| `analysis/confirmation_analyzer.py`（新） | 无 unseal 只允许 `--cost-only`；unseal/discipline/ledger 三方 SHA 绑定；A-4 复核；H1 唯一 primary；H2 / S0−T 描述；studentized max-t band（B_1/B_2，exploratory）；输出禁含 Action Cache 字段 |
| `action_cache_decision.py`（新） | schema / validator（`no` 时四项 canonical null）/ `seal_branch`（`yes` 无独立 G1/G2 package 则拒绝）/ `assert_no_action_cache_fields` |
| `register_confirmation_records.py`（新） | data_authority 记录写入助手（tgrid_dev / fresh_pools / budget_amendment；只能在产物存在后运行） |
| `config/confirmation_freeze_record.json`（新） | G1 冻结的三份文档 SHA + 全部常量；测试断言与代码常量一致 |
| `tests/dispatch_surface/test_rev2_confirmation.py` + `fixtures/budget_mixture_dev_stats.json`（新） | §6 用例（见下） |

**未在本阶段做的（按计划留到后续步骤）**：data_authority 三类记录（产物存在后由 `register_confirmation_records` 写）；真实 LIBERO round-trip smoke 与跨机 digest（Verify / §7-7）；`amendment_result` artifact（§7-6）；论文文件迁移（P0-C）。

**实现决定（供 G2 审）**
1. §3.1 分片：生产路径用**事件扫描**（当前活跃基相关的共线根 + 单臂成本），O(K²)/片；全量 O(K³) 枚举保留为 `full_enumeration=True` 参照。两者在 1000 组随机实例（含 K=32）上 AUC 差 < 1e-10、基集合相同；每片在两端与两处内点用 `value_at` 复核，不一致即 `NumericMismatch`。K=32 时 11 ms/replicate（10000 次 ≈ 2 min）。
2. `validate_confirmation_arms` 对 surface 臂按 **artifact 内容**绑定（yaml 内路径的文件 SHA 与 seal 成员 SHA 均须等于 sealed digest），不比较路径字符串——Rev 1 yaml 内是历史绝对路径。
3. Rev 1 threshold 臂进 C roster 时，其 pair 从归档 yaml 读取（`load_cache_config`）写入 `c_roster.json`，seal 据此携带全部 threshold pair。
4. `pieces`/`design()` 对 R 不敏感；正式 R=10000 的门在 `run()`（拒绝非冻结 `replicates/seed`）；测试用 `design()` 直接跑 200 replicate 验证 overflow / 止损路径。

**验证证据**
- `budget_mixture` 冻结 fixture（真实 development 九臂 sufficient stats）：l10 H1 **0.0816441**（旧直线 hull 0.0722133）、H2 0.0447783、S0−T 0.0368658；spatial H1 −0.0167（方向不变）；R2-B1 反例 0.9607 vs 0.6111；hull-at-zero 等价 1000 组 0 失配；Simpson 审计最大差 2e-12。
- **Phase 0 字节等价（§6-0）**：两 suite `phase0_summary` / `cost_map` / `phase0_outcome_design` 重跑 `cmp` 全部一致；`frontier_hull.py` 未改（sha `99f8a962…`）。
- 真实 l10 归档包 `v_offline_metric`：fold map 逐 episode 复现；h10 OOF pinball SV 0.111 vs S0 0.129（差 −0.018，q05/q95 −0.023/−0.013）、h7 −0.007（−0.011/−0.003）；coverage 0.948/0.951（level 0.95）；sharpness 5.99 vs 6.07（descriptive）。
- 全量回归（排除 `review_tests`、既有 `test_prebuilt_matrix_backend.py` 两个已知失败、以及单独运行的新文件）：**4594 passed / 45 skipped / 0 failed**；ruff 全净；`git diff --check` 净。
- 新测试文件：合成链 tgrid emit → runner 校验（含 5 类负例）→ discipline → package（缺批次/少 cell/漂移必拒）→ budget cost map（R=10000，AST 锁）→ outcome design（R/seed 门、漂移拒绝、overflow / 止损）→ power MC（CP 下界、非单调裁决、(N,r) 重放、同一裁决函数）→ generator（seed 域 > 2³²−1、attempt 语义、retry/failed/collision/shape、配额）→ fresh pools → task plan（无环）→ seal（smoke record / pilot 失败 / AC=yes 无包 均拒）→ 合成 C rollout → discipline（九类负例）→ unseal → analyzer（H1 唯一 primary、A-4、band、无 AC 字段）→ 真实 conductor 路径（`EpisodeTask(orig=None)` → wire → `_hit_row` → `ConfirmationStrategy.plan`）。**结果：31 passed / 0 failed（618 s，单独运行 `pytest tests/dispatch_surface/test_rev2_confirmation.py`）。**

**Code 阶段完成，交 G2。** G2 通过前不 emit tgrid 矩阵、不 rollout、不 materialize P/C。

### G2 Round 1 — Review Authority（Codex，2026-08-29）

**Verdict: CHANGES REQUESTED（G2 未通过；继续禁止 emit、rollout 与 P/C materialization）。**

主 estimand 的方向是成立的：Review Authority 复核了 `ArmStats`、不剪枝的单臂/两臂 LP 枚举、分式积分、support 语义、step-envelope 敏感性以及 H1 单一裁决入口，没有发现会推翻 budget-mixture 主结论的错误；ruff 与 `git diff --check` 也通过。以下问题集中在**冻结可重放性和确认数据认证链**，其中多项已有可执行的错误放行复现，不能以现有合成测试通过替代。

#### G2R1-B1 — freeze record 对可追加的 plan log 做整文件哈希，Code 记录一写入测试就必败

`confirmation_freeze_record.json` 保存的是 G1 暂存快照中 plan 的 SHA；本文件按工作流追加 §10 Code 记录后，工作树字节自然变化。当前定向复现：

```text
pytest tests/dispatch_surface/test_rev2_confirmation.py::test_freeze_record_matches_code_constants -q
FAILED: dispatch_surface_rev2_confirmation_plan.log.md drifted since the G1 freeze
expected 0b357190...; current 8047029d...
```

因此 §10 声称的“31 passed”是在追加 Code 记录之前取得的，不能代表交审快照；整文件 SHA 与同文件追加 Review/Code log 在定义上互斥。

**Required fix：**不得把 freeze SHA 更新为看到代码后的当前 log SHA。应生成不可变 G1 snapshot（或冻结到明确 Round-3 结束标记的 canonical prefix / Git blob），freeze record 绑定该快照；可追加主 log 只验证冻结 prefix 与 snapshot 字节一致。新增回归：追加 Code/G2 记录不破坏 freeze test，而改动 G1 冻结正文任一字节必败。修复后重新跑本线完整测试并报告**交审后**结果。

#### G2R1-B2 — formal power 的 inner RNG 不等于 G1 冻结流，record digest 又未覆盖裁决数值

G1 冻结的是：

```python
Generator(PCG64(SeedSequence(20260829, spawn_key=(N, r))))
```

当前 `inner_index()` 先 `generate_state(uint64)`，再用该整数重新构造 `PCG64`，这是另一条随机流。独立复现 `(N=30,r=0)`：冻结流前六个值 `[18,20,13,29,16,27]`，实现为 `[21,15,6,0,25,19]`。此外 `aggregate_sha256` 只覆盖 `N/r/passed/reason/两个 index SHA`，不覆盖 `effect/q05/joint_miss/left_support/audit/formal`；修改正式裁决数值不会改变 aggregate。

**Required fix：**直接把冻结 `SeedSequence` 传给 `PCG64`；为 `(30,0)` 等钉住 canonical index digest。每个 replicate 生成覆盖**完整正式 row**的 canonical digest（明确排除的只能是 wall-clock），aggregate 按 `(N,r)` 串接这些 row digest；任意 effect、q05、miss、formal 或 audit 篡改必须改变 aggregate 并被 validator 拒绝。

#### G2R1-B3 — power record 与 C roster 没有认证；极简伪造 JSON 可机械选择 N 并进入 seal

`build_seal()` 目前只看 `smoke/verdict/selected_N/c_roster_sha256`。本轮测试 fixture 自己就用下面这个截断 record 成功建 seal：

```json
{"protocol":"...", "smoke":false, "verdict":"n_selected", "selected_N":4,
 "c_roster_sha256":"..."}
```

它没有 4×200 replicate、formal 常量、per-N power/LCB、aggregate、outcome/cost-map 绑定，`N=4` 甚至不在候选集。`confirmation_power_mc.run()` 也只检查 roster 的 `outcome_design_sha256`，不机械比对 roster arms/reasons/F_MIN/M_MAX 是否等于 outcome design 的选择结果。

**Required fix：**新增共享 `validate_power_record()`，至少拒绝：协议/键域不精确、非正式常量、非 800 个唯一 `(N,r)`、任一 row `formal != true`、index/row/aggregate digest 不一致、per-N 计数/CP-LCB/selection 不能由 rows 重算、selected N 不在冻结候选或不满足“本 N 及其以上全过”、outcome/cost/roster SHA 漂移。正式 power 还须产确定性 replay validation artifact（重放级别按冻结协议，不得用 self-digest 代替），seal 要求其 SHA。roster validator 必须从 outcome design 重建预期 roster，逐 arm/family/reason/digest 相等；`--trials` 在 formal cost/outcome/power 入口显式钉死 30。

#### G2R1-B4 — P pilot 没有可执行、可认证的 100-episode链；三个自报字段即可通过

当前没有 P pilot runner/finalizer/discipline。seal 只读取 `{arm, pool_id, sr, attempt}`，测试 fixture 的手写 `{"arm":"always_full_inference","pool_id":"P","sr":0.85,"attempt":1}` 即通过；没有 10 task × 10 state 完备性、accepted journal、run id/ledger、P manifest/state digest、anchor YAML/server/policy、100 episodes 或从 outcome 重算 SR 的证据。

**Required fix：**补一条真实 P pilot 执行与 finalization 路径：权威 P task plan、唯一 launch ledger、100 个 Cartesian accepted cell、anchor `always_full_inference` 配置与内容 SHA、server/policy/seed、P manifest/文件内容绑定；finalizer 从 journal 重算 SR 和 one-shot/attempt 纪律并生成 pilot artifact。seal 必须调用 validator 重算，而不是信任 `sr` 字段。合成测试必须复刻真实 producer schema，并加入少一 task/state、重复 accepted、错 run、手改 SR、错 pool/manifest、非 anchor 的必拒例。

#### G2R1-B5 — fresh pool 的“跨机/互斥/资产”目前是自我声明，不构成 G1 要求的 authority

生成器单次运行会检查传入的 official/P↔C digest，但最终 seal 不重算这些关系；`cross_machine` 也不是生成器产出的可验证 artifact，seal 仅相信 manifest 中可手写的 `{"verified":true,"problems":[]}`（当前测试正是这样造 fixture）。另外 `--state-dim` 由 CLI 自报而非从官方每-task `.init` 推导；未强制 task manifest 恰为 10 个唯一 `0..9`；manifest 缺 G1 冻结的 HF asset rollup；seal 不验证 materialized init-file SHA、P↔C、两池↔官方 50 的全内容互斥。

**Required fix：**实现 fresh-pool finalizer/validator：从官方文件逐 task 推导并交叉核 state dimension；校验 10 个唯一 task id/name/BDDL；逐 `k=0..quota-1` 校验 attempts、SHA 域和 materialized state bytes；同时加载 P、C、official 50 重算三方互斥；记录并验证 task manifest、BDDL、HF asset rollup。跨机记录必须绑定 local/peer 两份 manifest SHA、host/environment，并实际调用 `compare_manifests` 重算，不接受裸 `verified` 布尔值。seal 只接受该 validation artifact。data-authority writer 也不得在未重算的情况下声称 exclusivity。

#### G2R1-B6 — task-plan loader 没验证 exact Cartesian roster，seal 可冻结不可执行或错任务的 plan

`load_task_plan()` 只检查总条数和每条 UID 自洽，不检查 entry arm 属于 `roster_arms`、每个 roster arm 是否覆盖 `10×N`、task-name↔task-id、prefix 对应的原始 `k`，也只在顶层找 `seal` key。独立对抗复现：plan 声称 `roster_arms=["a"]`、N=1，但放入十个 `ghost*` arm、全部 task_id=0 的自洽 UID，当前 loader 返回成功。

**Required fix：**loader/validator 构造期与加载期都比较 exact Cartesian key set = `roster_arms × task_id(0..9) × prefix(0..N-1)`；拒绝额外/缺失/重复 arm，绑定 pool manifest 的 task-id↔name 映射，按 `k` 升序的**原始 entries**取 prefix（不得依赖 JSON 当前顺序或过滤后位置），校验 digest 格式与 status=ok；递归拒绝 seal 引用。seal、runner 与 discipline 共用同一 validator。加入上述旧实现必过的 ghost-arm 负例。

#### G2R1-B7 — unseal 可接受绑定到错误 ledger 的旧 discipline，绕过 outcome-blind ledger certification

`write_unseal()` 只检查 discipline 的 `passed` 和 `seal_sha256`，不检查 discipline protocol/schema、`ledger_sha256`、task-plan/pool/journal/per-step 摘要。独立复现中 discipline 明写 `ledger_sha256="WRONG"`，函数仍为另一个 ledger 生成 unseal；analyzer 又不比对 discipline 自身的 ledger SHA，因此该旧 certification 可继续用于读取 outcome。

**Required fix：**`write_unseal` 必须验证 discipline 的 protocol/schema/suite/N/arms、roster_complete、seal/task-plan/pool SHA，并要求 `discipline.ledger_sha256 == sha(actual ledger)`；最好让 unseal 接收 task-plan/journal/per-step 并重新调用同一 `certify()`，消除“信 passed 字段”的第二实现。analyzer 再次交叉核 discipline 的 seal/ledger/task-plan/pool 与 unseal。新增“同 seal、journal/per-step 不变、换 ledger 后复用旧 discipline”旧实现必过负例。

#### G2R1-B8 — tgrid package 不是自包含 authority；包内 matrix 仍打开包外 `/tmp` YAML/roster spec

`finalize_tgrid_package` 虽复制了 `yaml/*` 与 `tgrid_roster_spec.json`，但复制后的 matrix 保留 emit 目录的绝对路径。实测 fixture：

```text
matrix_yaml  = /tmp/.../tgrid/cfg/dsp_tg_fh20_ws0.yaml
package_yaml = /tmp/.../tgrid/pkg/yaml/dsp_tg_fh20_ws0.yaml
same = False
```

`budget_cost_map._load_tgrid()` 随后对包内 matrix 调 `phase0_discipline.validate_tgrid()`，后者按 matrix 绝对路径重新打开原 `/tmp` YAML；清理执行目录后，完整 MANIFEST 包也不可消费。这违反 B4“MANIFEST role 是下游唯一入口”，也让包外漂移影响已 finalise 的 package。

**Required fix：**保持原 matrix/ledger 字节绑定的同时，让 package validator/discipline 通过 MANIFEST role resolver 打开 `yaml.<arm>` 与 `roster_spec`，并逐 SHA 对照 matrix 的记录；下游不得解引用 matrix 中的历史执行路径。加入 finalize 后移动/删除整个原 cfg 执行目录、仅保留 package 仍能 `verify_package → budget_cost_map` 的集成测试；包内 member 漂移仍必拒。

#### G2R1-B9 — Action Cache `inclusion=yes` 只检查“像 SHA 的字符串”，没有验证独立 G1/G2 package

`seal_branch()` 对任意包含十个 64-hex 字符串的 dict 返回 `ok=true`，不加载 package manifest、不核成员内容、不验证 G1/G2 verdict 或 runner/discipline 实现。这与冻结规则“yes 必须先有独立通过 G1/G2 的 Action Cache package”不等价。

**Required fix：**在独立 package validator 尚未实现前，`yes` 分支必须无条件 fail closed；实现后应接受 package **路径**并验证 schema、成员 SHA、review verdict、C-pool/roster/config/code/cost/runner/discipline 的真实绑定，不能只接收自报 digest object。`no/post_confirmation_descriptive` 分支可保持现状。

#### 验证裁决与下一轮条件

- Review Authority 定向检查：ruff **通过**，`git diff --check` **通过**；前八个 budget-mixture/基础测试在长套件中通过。
- 完整新套件在出现 freeze failure 后仍进入约 10 分钟的重计算；为避免用已知失败继续耗时，reviewer 中止，并单独复现 B1 traceback。执行方修复后必须从最终交审工作树重跑完整文件，不得引用追加 log 前的 31-pass 数字。
- 对抗复现已钉住：冻结 inner RNG 不同；wrong-ledger discipline 被 unseal 接受；非 Cartesian ghost-arm task plan 被 loader 接受；tgrid package matrix 指向包外 cfg。

**Re-review entry condition：**逐条响应 B1–B9，给每项加入至少一个“旧实现必过/新实现必拒（或必保持可消费）”的回归；提交最终代码与当时工作树上的完整定向测试结果。G2 通过前继续禁止 emit、rollout、P/C materialization、pilot 与 seal。

### G2 Round 1 — Execution Authority 响应（2026-08-29；B1–B9 逐条）

**独立复现（未采信 reviewer 断言）**：B1 定向 freeze 测试确为失败（工作树 plan SHA `43aa9822…` ≠ 冻结 `0b357190…`，后者恰等于暂存的 G1 快照 blob `efa4585c`）；B2 `(N=30, r=0)` 冻结流前六值 `[18,20,13,29,16,27]`，原实现 `[21,15,6,0,25,19]`，与 reviewer 数值一致；B3–B9 由源码直接确认（`build_seal` 只读 4 个 power 字段与 3 个 pilot 字段；`load_task_plan` 不核 arm ⊂ roster；`write_unseal` 不核 `ledger_sha256`；包内 matrix 保留 emit 目录绝对路径且 `validate_tgrid` 按其解引用；`seal_branch` 只做 64-hex 正则；`--state-dim` 自报；`cross_machine` 为手写布尔）。

**修复（每项附"旧实现必过 / 新实现必拒或必保持可消费"回归）**

| 项 | 修复 | 回归（`tests/dispatch_surface/test_rev2_confirmation.py`） |
|---|---|---|
| B1 | 新 `freeze_record.py`：`documents_sha256`（整文件）+ `frozen_prefix`（append-only log：到唯一 `end_marker` 行为止的 canonical 前缀 SHA）。freeze record 中 plan 改为 `frozen_prefix{end_marker="**G1 freeze boundary：**", sha256=0b357190…（未改）, git_blob=efa4585c}`；未把冻结 SHA 更新为看到代码后的当前 SHA | `test_freeze_record_matches_code_constants`（工作树通过）、`test_freeze_prefix_survives_appends_but_not_edits`（追加不变、改一字节必败、重复 marker 必拒） |
| B2 | `inner_rng = Generator(PCG64(SeedSequence(INNER_SEED, spawn_key=(N, r))))` 直传；抽样顺序冻结（每 replicate、每 task 一次 `integers(0,N,N)`）；每 row 增加 `row_sha256` = 全部正式字段（verdict/reason/effect/q05/joint_miss/support/half-effect/两个 index digest/audit_inner0）的 canonical digest；`aggregate_sha256` = 按 `(N,r)` 串接 row digest | `test_inner_stream_is_the_frozen_pcg64_seedsequence`（前六值 + 旧流反例 + 钉住 formal `(30,0)` 索引 digest `2be4c679…`）、`test_power_replicate_…`（q05 篡改改变 row digest） |
| B3 | 共享 `validate_power_record()`（协议 / 精确键域 / 非 smoke / 冻结常量 / 4×R_OUTER 唯一 (N,r) / 每 row formal / row+aggregate digest / per-N 计数 CP-LCB selection 由 rows 重算 / N ∈ 候选且满足规则 / design–roster–cost map SHA）；`validate_c_roster()` 从 outcome design 重建预期 roster 逐 arm/family/reasons/active_freq/source/delta/yaml+artifact 内容/threshold pair（重读 yaml）/bitset rollup 相等；新 `replay` 子命令产 replay artifact（`REPLAY_PER_N=5`，subset 由 aggregate digest 派生，从源数据重算 row digest），seal 要求并校验；`budget_cost_map.main` / `budget_outcome_design.run` / `confirmation_power_mc.run` 钉死 `--trials 30` | `test_outcome_design_roster_and_verdict`（roster 篡改必拒）、`test_power_record_validates_and_forgeries_are_refused`（极简伪造 / effect 篡改 / 重 digest 后 aggregate / N=4 / formal=false / smoke / trials）、`test_power_replay_recomputes_a_digest_derived_subset`、`test_seal_refuses_bad_inputs` |
| B4 | 新 `pilot.py` + `run_precheck --layer pilot`（P task plan、`--pool-manifest`、`--anchor-yaml`、Rev 1 package 合同、ledger 仅允许一条 launch，第二次启动被拒）；`pilot.finalize` outcome-blind 先证（P plan↔P manifest、单 launch、100 Cartesian accepted、anchor yaml 内容、anchor 全 MISS A-4、decisions==infers）再从 journal 重算 SR；`validate_pilot` 从记录内路径重算整条记录并与文件逐字段相等；seal 只调用 validator，且 anchor yaml SHA 必须等于 C roster 的 anchor | `test_pilot_chain_and_negatives`（少 state / 重复 accepted / 错 run / 两条 launch / 错 pool manifest / 非 anchor plan / 手改 SR / 旧三字段 record / 超容差 ⇒ `generator_validation_failed`） |
| B5 | `generate_fresh_inits`：state width 从官方 `.init` 推导（`--state-dim` 删除）；task manifest 10 唯一 id/name/BDDL digest；manifest 记 `assets_rollup`；`cross_machine` 子命令产记录（local/peer manifest SHA、host、environment，host 必须不同，`compare_manifests` 重算）；`validate` 子命令产 `fresh_pool_validation.json`（逐 k attempts+seed 派生、SHA 域、materialized bytes、官方50/P/C 三方互斥、asset rollup、跨机记录重算）；seal 只接受该 artifact 并**重跑**；data_authority writer 必须带 artifact | `test_fresh_pool_validation_artifact_and_negatives`（同 host peer / peer digest 漂移 / 伪造 verified / state_dim 自报 / 缺 rollup / init 文件篡改 / 与官方碰撞 / artifact 篡改） |
| B6 | `validate_task_plan`（精确键域、`roster_arms × 10 × N` 精确 key set、arm∈roster、task_id↔name 一致、prefix/pool 域、digest 格式、递归禁 seal 引用）为唯一结构校验；`verify_task_plan_against_pool` 按原始 `k` 序 entries、`k == prefix_idx`、`status == ok`；seal / runner / discipline / pilot 共用；`pool_id ∈ {P, C}` | `test_task_plan_loader_requires_the_exact_cartesian_roster`（ghost-arm plan 旧计数检查通过、新必拒；缺 cell / 多臂 / 错名 / 乱序 / 超配额） |
| B7 | `write_unseal(..., task_plan, journal, per_step)` 核 discipline 的 protocol/seal/ledger/task-plan/journal/per-step/pool SHA、suite/N/arms/roster_complete，并**重新调用 `certify()`** 要求逐字节相等；analyzer 交叉核 discipline 与 unseal 的全部绑定 | `test_unseal_recertifies_and_refuses_a_foreign_ledger`（同 seal、journal/per-step 不变、换 ledger 复用旧 discipline 必拒；伪造 discipline 必拒）、`test_analyzer_verdict_and_cross_checks` |
| B8 | `validate_tgrid(..., yaml_paths=, rev1_manifest_path=)`、`validate_tgrid_matrix_artifacts(..., rev1_manifest_path=)`；`budget_cost_map._load_tgrid` 通过 MANIFEST role 解析全部 yaml 与 Rev 1 包，不解引用 matrix 中的执行路径；`verify_package` 对包内 yaml 成员做内容校验 | `test_tgrid_package_is_self_contained_after_execution_dir_is_gone`（整体移走 cfg/run 目录后 verify → `_load_tgrid` → `build` 仍可消费；旧解引用路径必败）、`test_tgrid_package_verify_and_negatives`（成员漂移必拒） |
| B9 | `seal_branch`：`inclusion=yes` 无条件 fail closed（`action_cache_package_validator_not_implemented`），`YES_PACKAGE_FIELDS` 仅作未来 validator 必须推导的 digest 清单 | `test_seal_refuses_bad_inputs`（全 10 个 64-hex digest 的 package 仍拒） |

**接口变化**：`confirmation_power_mc` 改为 `run | replay` 子命令；`seal_confirmation seal` 新增 `--budget-cost-map --power-replay --pool-validation`，`unseal` 新增 `--task-plan --journal --per-step`；`generate_fresh_inits` 改为 `generate | cross_machine | validate` 子命令（`generate` 新增 `--assets-dir`，删除 `--state-dim`）；`build_confirmation_task_plan` 新增 `--pool-id --arms`；`run_precheck` 新增 `--layer pilot` 与 `--pool-manifest --anchor-yaml --rev1-package-manifest`；`finalize_tgrid_package` 新增 `--rev1-package-manifest`；`register_confirmation_records fresh_pools` 必须 `--validation`。

**测试规模说明（供 G2 审）**：合成世界在 16 个 outer replicate 下机械裁决为 `underpowered_stop`（各 N 通过 4–10/16，主因 `joint_miss_exceeds`：R_INNER=64 时一次 miss 即 1.6% > 1%），规则本身工作正确，但会使确认链下游无法被测试；因此 `power` fixture 以 `pytest.MonkeyPatch` 把 `N_CANDIDATES / R_OUTER / R_INNER / POWER_TARGET` 缩为 `(16,20,24,28) / 16 / 64 / 0.05`，跑**真实**的 `run → validate → replay` 流水线（验证器读模块常量，正式 record 走完全相同代码；正式值由 `FROZEN_PMC` 与 freeze 测试钉住；`select_n` 的默认 target 改为调用期读取常量，避免定义期绑定），`confirmation` fixture 改为 assert 而非 skip。unseal 语义：对另一份仍完整的 ledger 重新 `certify()` 产生的新 artifact 可以 unseal（正是"重认证而非复用"），而声称某份 certify 必拒的 ledger 通过的伪造 discipline 被拒。

**仍不能机械保证、需 owner/reviewer 注意**：pilot 一次性在 ledger 层面强制（同一 ledger 第二条 launch 被拒、pilot record 只能绑定恰一条 launch），但换输出路径重跑 pilot 无法被代码发现——seal 只绑定一个 pilot record，运行历史需由 owner 核对远端目录。Action Cache `yes` 分支的 package validator 尚未实现（按 B9 要求先 fail closed）。

**测试**（G2 R1 修复后、当前工作树）：`pytest tests/dispatch_surface/test_rev2_confirmation.py` **40 passed / 0 failed / 0 skipped**（650 s，含 B1–B9 全部正负例；此前一次运行因合成世界 `underpowered_stop` 导致确认链 8 项 skip，已改为 assert 并按上文说明调整测试规模常量）；`tests/dispatch_surface/test_rev2_phase0.py` 67 passed、其余 `tests/dispatch_surface` 233 passed、`tests/cache/test_surface_binding.py` 24 passed（这三组在最后两次仅改测试文件与 `select_n` 默认参数读取方式之前运行；它们不导入 `confirmation_power_mc`）；ruff 全净；`git diff --check` 净；Phase 0 冻结模块 `frontier_hull.py` 未改。

**G2 R1 修复完成，交 G2 R2。** G2 通过前继续禁止 emit、rollout、pilot、P/C materialization 与 seal。
