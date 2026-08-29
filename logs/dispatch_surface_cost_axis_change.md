# Dispatch Surface —— 成本轴变更提案（提交评审）

> 2026-08-27 | 提交人：Execution Authority | 状态：**待裁决**
> 关联：[`dispatch_surface_plan.log.md`](dispatch_surface_plan.log.md)（§4.6 预检设计、§11 执行日志）
> 本文自包含，不需要先读 plan 也能判断。

---

## 0. 一句话

执行期发现 plan §4.6 的成本轴有两处需要改：**（一）独立成本 bench 应当删除**，成本可从主预检解析算出，多跑 2,800 个 episode 换不到精度；**（二）Gate 2 的成本守卫规则使整体把握度只有 0.70**，建议把守卫从「97.5% 置信上界 ≤ +5%」改为「点估计 ≤ +5%」（阈值不动）。两项都在看到任何 test 数据之前提出。

---

## 1. 这条线在做什么

免训练的三档调度：对每个决策步，用检索到的缓存条目算两个量——融合分数 `s` 与加权 top-k 动作分散度 `v`——据此三选一：

| 档 | 动作 | GPU 成本（CUDA-graph 档） |
|---|---|---|
| FULL_HIT | 直接用缓存动作 | stage1 = 10.26 ms |
| WARM_START | 从缓存的 t=0.3 中间态续跑 | stage1 + stage2 + 0.3 × stage3 = 46.82 ms |
| MISS | 全量推理 | stage1 + stage2 + stage3 = 67.52 ms |

要证的是：这个调度面比现有的双阈值 baseline 好——同成功率下更省 GPU。

**臂表**（各 300 episode，同一 A′ init 池，配对）：

- `T1/T2/T3` —— 双阈值 baseline，三个操作点，构成对照前沿
- `S0` —— 只用 `s`、不用 `v` 的阉割版（嵌套消融）
- `SV` —— 完整方法（primary）

**两道确认性门**（预注册，测后不得改）：

- **Gate 1**：`SV` vs `T` 前沿插值点。成功率不掉、GPU 成本省 ≥5%、且原有的延迟门。证明**调度面有用**。
- **Gate 2**：`SV` vs `S0`。成功率严格提升，且成本不显著变贵。证明 **`v` 这一维有用**。

Gate 2 是 fixed-sequence 的第二关，Gate 1 不胜即停。

---

## 2. 现状：已完成且可用

| 产物 | 状态 |
|---|---|
| `src/` 全部实现（`dispatch_surface` judge、契约绑定、policy attestation） | G1×5 + G2×4 APPROVED，Verify 过，commit `1985271` 已 push |
| init 四方划分（D_lib 50 / fit 50 / cal 100 / A′ 300，共 500 官方 init） | 已物化并逐字节复核（450/450 与其声称的官方索引相同），已登记 data_authority |
| 标定检索配置 `calibration_retrieval.yaml` | 已由 gtp 模板机械派生，三方契约 digest 实测相等 |
| query cohort 采集计划（150 ep） | 已生成 |
| weilandserver 执行环境 `/data/openpi_dispatch` | 已部署并端到端验证（远端重跑 split 与 plan，产物 digest 与本机逐字节相同） |
| GPU 步骤（重建库 → 采集 → 标定 → 拟合） | **未开始**，4090 被另一实验占用 |

**没有任何 test 数据被生成或查看过。** 本提案的全部数字来自一次**已完成的、无关的**历史实验（`gate_threshold_pareto` sweep，32 臂 × 498 共享 init cell），用作方差先验。

---

## 3. 变更一：删除独立成本 bench，成本改为解析计算

### 3.1 plan 现状

§4.6 规定一个**独立成本实验** `run_cost_bench.py`：单 server、两次互斥的 launch（`OPENPI_MONITOR_LEVEL=SNAPSHOT` 测 compute、`OFF` 测 latency）、block 设计（每 block 10 ep）、由 power simulation 从 R∈{5,10,15} 选块数、总量 5 臂 × R × 10 ep × 2 pass。

引入它的理由（§4.6 原文）：**"现有接口无法支撑主预检内的逐 decision 配对成本"**——`client_timing` 只有 episode 累计、`SystemTimer` 行没有 yaml/episode join key、conductor 持久连接跨臂使归属断裂。

### 3.2 为什么该删

那个理由针对的是**计时**。但成本口径是 **GPU inference ratio**——三档 verdict 的加权和，是个**解析量**：

```
cost(arm) = Σ_step  unit_cost(hit_type, start_t)
```

而主预检的 `per_step.jsonl` **逐 step 就记录着 `hit_type` 与 `start_t`**，且带完整的 `(yaml_id, task_id, orig_init_state_idx, episode_id)` join key。数一数三档各多少步、乘上单位成本，每臂的成本就出来了——与 SR 用同一批 episode、同一批 init、天然配对，**不需要多跑任何一个 episode**。

### 3.3 精度对比（实测）

在 gtp sweep 上取 30 init/task（模拟 A′ 配额），task 分层 init-cluster bootstrap，臂对按各 Gate 实际面对的差异程度分组：

| 口径 | 额外 episode | Gate 1 把握度 | Gate 2 把握度 |
|---|---|---|---|
| **解析成本，主预检 300 ep/臂** | **0** | **0.765** | **0.696** |
| 独立 block bench（block=40 ep、R=7） | 2,800 | SE 0.019（同等） | 同等 |

**独立 bench 多花 2,800 个 episode，换来的 SE 是 0.019 对 0.021——没有实质差别。**

此外，独立 bench 在 A′ 池内**根本无法达标**：`materialize_block_pools` 要求 block 之间 init 不重复，A′ 每 task 30 个（末位留 warmup，实际 29 可用），任何 block 尺寸都撑不到 80% 把握（最接近的 block=40 ep 用满池子仍只有 0.779，需要多 3 个 init/task）。

### 3.4 删除后一并消失的东西

block 设计、`R∈{5,10,15}`、power simulation（含 variance source 文件与 `validate_power_record` 的确定性重放）、两次独立 server launch、`stage_probe_backends` 的 CUDA 断言、A′ 池的 init 配额冲突。

`run_cost_bench.py`、`power_sim_cost_blocks.py` 及 `analyze_precheck.py` 中消费它们的纪律检查需重写或删除。

### 3.5 附带修正的一处实现错误

`decision_compute_ms` 原写 `frac = 1 - start_t`，把 WARM_START 当成跑 70% 的 stage3。源码 `pi0_pytorch.py:691` 明确：`start_t=0.3, num_steps=10 → 3 steps, saves 70%`，即**跑 `start_t` 那一份**。已改为 `frac = start_t` 并加测试钉死。

该错未污染任何已得数字——gtp 那批数据 warm_start 整条关闭，WS 占比 0%，该分支从未被触发。

### 3.6 另需裁决的遗留项

§4.6 的 Gate 1 第 3 条与 Gate 2 第 3 条是**延迟门**（`D_l` / `Δlatency` 上界 ≤ 0）。在 GPU inference ratio 口径下，检索的 CPU 开销不占 GPU、不进成本，这两条门**没有对应的测量物**，需明确删除或改写。

---

## 4. 变更二：Gate 2 的成本守卫使把握度只有 0.70

### 4.1 问题

Gate 2 由三条组成，全过才胜：

1. `ΔSR` 的 95% CI 下界 > 0 —— **主张**：v 提升成功率
2. `Δcompute` 的 **97.5% 上界 ≤ +5%** —— **守卫**：v 不是靠多烧 GPU 换 SR
3. `Δlatency` 上界 ≤ 0 —— 见 §3.6，无对应测量物

第 2 条是守卫，不是主张，但它写成"必须通过"，于是它的把握度直接乘进整体把握度。实测 SE = 0.0202（300 ep/臂、嵌套臂对），该守卫在 v 真的不增加成本时**只有 0.697 的概率放行**——也就是说，即使 v 完全无害，也有三成概率被自己的守卫拦下。

### 4.2 候选规则与实测把握度

| 方案 | 把握度 | 最坏臂对 | 漏放真实 +8% 回归 | 漏放真实 +15% 回归 |
|---|---|---|---|---|
| **A 现状**：97.5% 上界 ≤ +5% | 0.697 | 0.338 | 0.000 | 0.000 |
| B 放宽阈值：97.5% 上界 ≤ +8% | 0.977 | 0.695 | 0.025 | 0.000 |
| **C 点估计 ≤ +5%** | **0.993** | **0.939** | 0.069 | 0.000 |
| D 95% 上界 ≤ +5% | 0.797 | 0.460 | 0.001 | 0.000 |
| E 90% 上界 ≤ +5% | 0.884 | 0.603 | 0.003 | 0.000 |

「把握度」= v 真的不增加成本时守卫放行的概率；「漏放」= v 真的增加了那么多成本时守卫仍放行的概率（守卫的失效方向，越低越安全）。

**关键观察**：所有方案漏放 +15% 真实回归的概率都是 0。SE 只有 2%，任何有实质意义的成本回归都会被拦住——守卫的严格程度**几乎不影响它拦截真实回归的能力**，只影响它误伤的概率。

---

## 5. 推荐方案

### 5.1 变更一：采纳删除（信心高）

理由：多花 2,800 episode 换不到精度；且在 A′ 池内本就无法达标；且解析口径与 Pareto 前沿要画的量（GPU inference ratio）严格一致，而独立 bench 测的墙钟时间还需要额外论证它与该量的关系。

### 5.2 变更二：推荐 **C（点估计 ≤ +5%）**

**理由一：不放宽实质阈值。** 阈值仍是 +5%，改的是"用哪个统计量去判"，不是"容忍多少成本"。这个区分重要：在发现把握度不足后放宽**容忍度**（方案 B）是科学判断的让步，观感上接近 p-hacking；改**统计量形式**是测量框架问题，且本次框架本来就因为 §3 的口径变更被重写。

**理由二：符合守卫的语义。** 「97.5% 上界 ≤ +5%」要求的是"以 97.5% 置信**证明**成本没涨超 5%"，那是主张性要求。守卫要回答的是"观察到的成本有没有明显变贵"，那是筛查性要求。让守卫承担证明性负担，等于用主张的标准去要求一个保护性条款，代价是把主张自己的功效预算吃掉三成。

**理由三：保护力没有实质损失。** 真实回归 +15% 时拦截率仍是 100%，+8% 时漏放 6.9%。而 Gate 1 已经独立要求 SV 相对 baseline 省 ≥5% GPU，SV 的绝对成本另有约束。

**备选 E（90% 上界 ≤ +5%）**：若评审认为守卫必须保留置信上界的形式，E 把握度 0.884、漏放 +8% 仅 0.3%，是次优选择。

**不推荐 B**（放宽到 +8%）：它实质放宽了可容忍的成本回归，且是在看到把握度不足之后放宽，是三个选项里观感最差的。

**不推荐维持 A**：0.697 的把握度意味着 v 即使真的有用，也有三成概率因守卫误伤而"未获证"，而这个损失换来的保护力增量为零（见 §4.2 漏放列）。

### 5.3 Gate 1 的把握度（0.765）建议维持不动

Gate 1 的三条中，成本门是**主张**的一部分（"省 ≥5% GPU"正是要证的东西），不是守卫，其严格性有实质意义。0.765 略低于 0.80 的惯例，建议如实记录而非调整——调整主张性判据才是真正的 p-hacking。

---

## 6. 诚实性声明

- 两项变更均在**任何 test 数据生成之前**提出。GPU 步骤（重建库、cohort 采集、标定表、拟合）**一步都未开始**。
- 本提案的全部方差数字来自 `gate_threshold_pareto` 的**历史 sweep**，那是一条已完成的、不同的实验线，与本线的 δ\*、臂配置、surface artifact 无任何关联。
- 变更二确实是"在算出把握度不足之后调整判据"。这一点无法回避，故推荐方案刻意选择**不动阈值、只改统计量形式**的 C，并把观感更差的 B 明确标注为不推荐。是否接受这个区分，由评审判断。
- 变更一的动因是口径（§3.2），不是结果——即使把握度充足，多花 2,800 episode 换不到精度这件事仍然成立。

---

## 7. 影响面

**plan 需修订的段落**：§4.6「成本轴：独立成本实验」整段（block 设计、R 预定、双 pass 分离、采集生命周期）、§4.6 Gate 1 第 3 条与 Gate 2 第 2/3 条、§7 步骤 6、§9 风险登记中与 R 相关的行。

**代码需改动**：`exp/dispatch_surface/run_cost_bench.py`（删除或重写）、`power_sim_cost_blocks.py`（删除）、`analysis/analyze_precheck.py`（成本来源改为 per_step 解析、删除 power record 与双 pass 的纪律检查）、`emit_precheck_yamls.py`（不再需要 power record 绑定）。`src/` **零改动**。

**不受影响**：SR 轴全链（标定 → 拟合 → δ\* 机械冻结 → 预检 SR 比较）、init 四方划分、契约绑定、policy attestation。

---

## 8. 证据与复核路径

| 结论 | 脚本 | 产物 |
|---|---|---|
| 解析口径下两 Gate 的把握度、臂对分组 | `exp/dispatch_surface/analysis/analytic_cost_power.py` | 直接打印 |
| block 设计的方差、A′ 池可行域、单位成本敏感性 | `exp/dispatch_surface/analysis/block_variance_probe.py`（22 个单测） | `block_variance_probe.json` |
| 三档成本模型、`start_t` 语义 | `tests/dispatch_surface/test_block_variance_probe.py` | 114 tests pass |
| init 划分的逐字节复核 | `exp/data_authority/records/dispatch_surface__libero_spatial__init_pools.json` | registry validate + verify 全绿 |

方差先验数据源：`exp/gate_threshold_pareto/data/eval/libero_spatial/per_step.jsonl`（32 臂 × 498 共享 (task, init) cell，含逐 step `hit_type` 与逐 episode `client_timing`）。

单位成本来源：`exp/data_authority/records/latency_bench__libero_spatial__executor_costs.json` 的 CUDA-graph 档（stage1 10.26 / stage2 27.69 / stage3 29.57 ms）。该台账本次新增两条 caveat：15 个文件中仅 4 个带原始 per-call 列表（所有 pi05 文件只有汇总统计）；它是单次 forward 的 microbench，无 block 结构，供不了块级方差——这正是 §4.6 原本指定的 sigma 来源，也是本提案改从 gtp sweep 实测的原因。

---

## 9. Review Authority 裁决

### 9.1 裁决摘要

1. **变更一批准**：删除独立 cost bench；解析成本作为正式成本 estimand。三档单位成本按已冻结的 CUDA-graph 台账计算：

   \[
   c(\mathrm{FH})=10.26,
   \]

   \[
   c(\mathrm{WS})=10.26+27.69+0.3\times29.57=46.821,
   \]

   \[
   c(\mathrm{MISS})=10.26+27.69+29.57=67.52.
   \]

   `start_t=0.3` 表示执行 30% 的 stage3；该语义必须由测试固定。正式文字应称其为 **解析 GPU inference cost** 或 **model-forward compute proxy**，不能写成实测端到端延迟。

2. **延迟确认门批准删除**：当前主预检没有与原延迟门匹配、且不受多 worker 竞争污染的确认性测量物。已有 `client_timing` 可保留为描述性结果，但不得参与 Gate 1 或 Gate 2 的通过判定。

3. **变更二不批准方案 C，批准方案 D**：Gate 2 成本条件改为单侧 95% 上置信界不超过 `+5%`，不采用“点估计不超过 `+5%`”。

4. 本次是在任何 test 数据生成或查看前提出，允许作限定范围的预注册修订。先修订 plan 并冻结下述规则，再实现分析器；不得在看到 test 数据后重新选择 C/D/E。

### 9.2 Gate 2 的最终规则

定义

\[
\Delta SR=SR_{SV}-SR_{S0},\qquad
\Delta C=C_{SV}/C_{S0}-1.
\]

Gate 2 通过，当且仅当同一套配对 bootstrap draws 同时满足：

\[
q_{0.05}(\Delta SR)>0
\]

和

\[
q_{0.95}(\Delta C)\le 0.05.
\]

这是方向明确的 intersection-union test：论文主张要求“SR 严格提升”与“成本增幅不超过 5%”同时成立。在该联合原假设下，只要有一个分量原假设为真，联合主张即不成立，因此两个单侧分量各使用 `alpha=0.05`，无需对这两个合取条件作 Bonferroni 修正。Gate 1 → Gate 2 继续使用既有 fixed-sequence 顺序。

若为了最小化规则改动而保留 SR 的 `q0.025` 下界，可以更保守地执行，但正式推荐是与方向性主张一致的 `q0.05` 单侧下界。不得把 `q0.025` 与“单侧 95%”混称。

### 9.3 不采用方案 C 的原因

“点估计 ≤ +5%”没有确认性错误率控制。尤其当真实成本增幅恰好位于非劣界值 `+5%` 时，近似对称的点估计约有 50% 概率落在界值以下，即边界处误放率约为 50%，不能支撑“成本增幅不超过 5%”的论文结论。

§4.2 的表格只列出真实 `+8%` 和 `+15%` 时的漏放率，没有列出决定检验有效性的 `+5%` 边界。因此“+15% 样本内未观察到漏放”不能替代边界处的一类错误控制，也不应表述为真实拦截率必然为 100%。

方案 D 的统计含义清楚：检验 `H0: ΔC ≥ 0.05` 对 `H1: ΔC < 0.05`，单侧 `alpha=0.05`。历史方差模拟给出的放行概率约为 `0.797`，接近 80% 设计目标；在报告模拟设定下，真实 `+8%` 时漏放约 `0.001`。方案 E 使用 `alpha=0.10`，确认性标准偏弱；方案 B 则改变了科学容忍边界，均不优于 D。

报告中的 `0.697/0.797` 仅是相应**成本条件**在历史方差模型下的预计通过概率，不是整个 Gate 2 的联合功效；联合功效还取决于 SR 效应量及其与成本统计量的相关性。

### 9.4 解析成本 estimand 与重采样纪律

正式 per-decision 成本应定义为 decision-weighted ratio-of-sums：

\[
C_a=\frac{\sum_d c(h_d)}{N_{\mathrm{decisions}}}.
\]

不能先求每个 episode 的 per-decision 均值再对不等长 episode 等权平均，否则估计的是“随机 episode 的平均成本”，与本文声称的“随机 decision 的 GPU inference cost”不是同一个 estimand。每个 bootstrap replicate 都必须重新按被抽中的完整 init cluster 汇总分子与分母。

实现必须满足以下纪律：

- 在每个 task 内以 `orig_init_state_idx` 为 cluster 作有放回配对重采样；同一 replicate 对所有臂使用相同的 task/init 抽样索引。
- 同一 replicate 同时计算 SR 与解析成本；不得为两条轴分别抽样。Gate 1 的 threshold-frontier 插值也须在该 replicate 内用同一抽样重新计算。
- 主分析输入必须精确覆盖预注册的 `arm × task × init` 网格，每格恰有一个 accepted episode；缺格、重复格或身份不一致均 fail closed。
- `per_step` 的 decision 数必须与 episode 的 inference 数一致；`hit_type` 只能为 `FULL_HIT`、`WARM_START`、`MISS`。缺失、未知或 `UNPROBED` 不得静默按 MISS 计费，也不得静默丢弃 episode。
- 所有 WS 行必须验证 `start_t=0.3`；单位成本台账摘要、输入 `per_step.jsonl` 摘要和臂配置摘要必须进入分析 manifest。

### 9.5 结果判读

- 两项均通过：可以表述为“`v` 带来成功率增益，且解析 GPU inference cost 的增幅以单侧 95% 置信不超过 5%”。
- SR 条件通过、成本条件未通过：只能表述为“观察到/确认了 SR 增益，但成本非劣性证据不足”，不能据此判定 `v` 无效。
- SR 条件未通过：`v` 的独立成功率收益未获确认。
- `client_timing` 仅作描述性报告，并明确多 worker 竞争等限制。

### 9.6 对 power 证据的限定

`analytic_cost_power.py` 使用历史 threshold 臂对估计未来嵌套 `SV/S0` 对比的方差；臂对的逐 init verdict 不一致率和相关结构未必相同。因此 `0.797` 应写作设计阶段的 sensitivity estimate，而不是保证功效或精确的未来通过概率。该限制不阻止在 test 数据前采用方案 D，但必须随预注册修订一并记录。

**最终 verdict：有条件批准成本轴变更。** 条件是按本节采用解析成本、删除确认性 latency 门、Gate 2 使用方案 D，并落实 §9.4 的配对重采样和 fail-closed 输入纪律；方案 C 不获批准。

---

## 10. Execution Authority 接收（2026-08-27）

**裁决全部接受，无异议条目。** 复核后确认 §9.3 与 §9.4 各指出本提案一处实质错误，两处均已独立验算坐实。复核脚本 `exp/dispatch_surface/analysis/verify_cost_gate_rules.py`。

### 10.1 §9.3 成立：方案 C 在边界处无错误率控制

本提案 §4.2 的表格只列了真实 `+8%` 与 `+15%` 的漏放率——那是"明显超标"的区域，任何规则都拦得住；决定检验有效性的是**非劣界值 `+5%` 处**，该表漏了它。补算（SE 取 §10.2 的裁决 estimand 值 0.02151）：

| 规则 | 真 0% | **真 +5%（边界 = 一类错误率）** | 真 +8% | 真 +15% |
|---|---|---|---|---|
| A 97.5% 上界 ≤ +5% | 0.642 | 0.025 | 0.000 | 0.000 |
| **C 点估计 ≤ +5%** | 0.990 | **0.500** | 0.082 | 0.000 |
| **D 95% 上界 ≤ +5%（裁决）** | 0.752 | **0.050** | 0.001 | 0.000 |
| E 90% 上界 ≤ +5% | 0.851 | 0.100 | 0.004 | 0.000 |

方案 C 在边界处误放率恰为 0.500（对称分布落在界值以下的概率），确实不能支撑"成本增幅不超过 5%"的确认性表述；方案 D 恰为 `alpha=0.05`。原提案 §5.2「理由三：保护力没有实质损失」建立在只看 +8%/+15% 之上，**该理由不成立**。另接受 §9.3 关于「拦截率必然 100%」属过度断言的批评——那是有限模拟加正态近似的产物，不应写成确定性陈述。

### 10.2 §9.4 成立：原提案的 estimand 是错的，且该修正改变了数字

原实现取每 episode 的 per-decision 均值、再对 episode 等权平均，估的是"随机 episode 的平均成本"。实测 episode 决策数在 **14–44** 之间（中位 22），两个 estimand 因此不重合。按裁决改为 decision-weighted ratio-of-sums（分子分母在每个 bootstrap replicate 内按被抽中的 init cluster 重新汇总）：

| estimand | 嵌套臂对 SE | 方案 D 在真 0% 时的放行概率 |
|---|---|---|
| 裁决 §9.4（decision-weighted ratio-of-sums） | 0.02151 | **0.752** |
| 原提案（episode 等权） | 0.02023 | 0.797 |

**因此 §9.3 引用的 `0.797` 应更正为 `0.752`**——该数由本提案提供，是用错 estimand 算出的。结论不变（D 仍是四个候选中唯一兼顾错误率控制与可用把握度的），但正式记录应使用 0.752，且按 §9.6 标注为 design-stage sensitivity estimate。

### 10.3 一处提请 owner 注意（非异议）

§9.2 将 Gate 2 的 SR 条件由 plan 原文的「95% CI 下界 > 0」（即 `q_0.025`）改为单侧 `q_0.05`。就方向性主张而言这是恰当的，且 §9.2 已允许保留 `q_0.025` 作为更保守的执行方式。但需明确：**它同样是一次判据放宽，时机与变更二相同**。若希望本次变更包"改判据"的面积最小，可只改成本条件（那一条本来就必须重定）、SR 条件保留 `q_0.025`；若采纳 IUT 内部一致性的论证，则两条同用 `alpha=0.05`。两者皆可辩护，选择权在 owner，但应显式记录选了哪个及理由。

### 10.4 后续执行义务

按裁决 §9.4 落实的实现清单（`analysis/analyze_precheck.py`）：

- 成本 estimand 改为 decision-weighted ratio-of-sums，比值在每个 replicate 内重新形成；
- 同一 replicate 内以 `orig_init_state_idx` 为 cluster、task 分层、有放回配对重采样，**所有臂共用同一抽样索引**，SR 与成本**同 replicate 计算**（IUT 的有效性依赖两个统计量的联合分布），Gate 1 的前沿插值亦在同一 replicate 内重算；
- 输入必须精确覆盖预注册的 `arm × task × init` 网格，每格恰一个 accepted episode，缺格/重复/身份不一致 fail closed；
- `per_step` 决策数与 episode 的 inference 数必须一致；`hit_type` 仅允许三档，`UNPROBED`/缺失/未知**不得静默按 MISS 计费、也不得静默丢弃 episode**；
- 所有 WARM_START 行验证 `start_t=0.3`；单位成本台账摘要、`per_step.jsonl` 摘要、臂配置摘要进入分析 manifest；
- 正式文字一律称 **解析 GPU inference cost / model-forward compute proxy**，不得写成实测端到端延迟；`client_timing` 仅作描述性报告并注明多 worker 竞争限制。

plan §4.6 与 §7 的对应段落须先按上述修订并冻结，再实现分析器。**在 plan 修订落定前不 emit 臂矩阵、不启动闭环。**

---

## 11. G2 Round 1 修复（Execution Authority，2026-08-27）

Review Authority 对成本轴实现给出 **G2 NEEDS REVISION**（4 blocking）。四条全部核实成立并已修复；此轮未与评审争辩任何一条。

### 11.1 B1 — 真实数据会被 analyzer 直接拒绝（已修）

`episode_runner.py` 写出的 `client_timing` 行只有 `_kind` / `task_uid` / `yaml_id` / `task_id` / `subset_init_state_idx` + 计时字段，**没有 `orig_init_state_idx`**；而我的 loader 在 dispatch `_kind` 之前就强制读该字段，真实预检的第一条 timing 行就会炸。端到端测试没抓到，是因为我手造的 fixture 给 timing 行补了那个字段——**用简化 schema 造数据，测的就是自己的假设**。

修复：join 键统一为 **`task_uid`**（三类行都有），先 dispatch `_kind` 再取字段；`infers` 与计价 decision 数**无条件**比对（原写法 `if reported is not None` 等于该纪律从未执行）；每个 accepted episode **必须**恰有一条 `client_timing`，缺失即拒。测试 fixture 改为逐字段复刻两个 producer 的真实输出。

### 11.2 B2 — 失败重试与 fenced episode 会污染成本（已修）

conductor 会把 stale / fenced attempt 的 per-step 行一并持久化，用 `(task_uid, attempt, accepted, run_id)` 区分（`driver.py:323` 一带）。我的 loader 只按 `yaml_id × task_id × orig_init_state_idx` 聚合，于是旧 attempt 与 `accepted=false` 的 verdict 会被一起计费。

修复：新增 `load_accepted_episodes`，从 journal 选出每格**唯一**的 accepted `(task_uid, attempt, run_id)`（缺 attempt/run_id、同格两条 accepted、格外 episode 均拒）；成本 loader 按该三元组精确 join，stale/fenced/他 run 的行**排除并计数**（`excluded_stale_rows` 进 manifest），网格外 cell **fail closed** 而非静默 `continue`。四个对抗测试各覆盖一类。

### 11.3 B3 — A′ 的 official init 身份没有真正传进预检（已修）

`run_gtp.SweepStrategy` 写 `orig_init_state_idx = ep_idx`，那只在「池就是官方池」时成立；A′ 是官方 50 抽 30 物化的，subset 位置 0..29 不是 official index。沿用该 strategy 会给每个 episode 贴上它并不具有的 provenance。

修复（**分两半，我只做了前一半**）：

*driver 侧（本会话）*：新增 `PrecheckSweepStrategy`，读冻结 split manifest，`episode_idx` 保持 subset 位置、`orig_init_state_idx` 写真正的官方 0..49 索引（A′ 按 `sorted(indices)` 物化，故 subset i ↔ 第 i 小的 official）。analyzer 的网格从 split manifest 导出（不再硬编码 `range(30)`），并把每行的 `orig_init_state_idx` 与该映射**交叉验证**；launch ledger 绑定 split manifest 的 sha256。

*worker 侧（由 codex 补全，2026-08-27 22:36）*：**上述修复单独存在时会在真实运行中失败**——`worker_entry._build_episode_setup` 原本正是用 `task.orig_init_state_idx` 去索引物化池，而 A′ 池只有 30 个 state、official index 可达 49，于是越界或读到错误的 init。补法是把「索引」与「身份」分开：`worker_entry` 增 `--init-state-index-mode {orig,subset}`（预检用 `subset` 索引池、`orig_init_state_idx` 只作身份），并对索引加越界检查；`conductor/agent.py` 的 `WorkerSpec` 增同名字段并透传；`run_precheck` 的 `WorkerSpec` 传 `init_state_index_mode="subset"`。

这是本会话实现的一处实质缺陷：我改了写入端却没改读取端，而这条路径没有任何既有测试覆盖（真实 worker 需要 LIBERO env）。codex 同时补了 `tests/dispatch_surface/test_precheck_runner.py`，其中 `test_materialised_aprime_uses_subset_position_without_losing_official_identity` 正是钉住这一分离。

### 11.4 B4 — 运行时 YAML 未绑定到分析时 YAML（已修）

原 launch manifest 只记 arm 名、library、A′ 与 contract。analyzer 重算今日磁盘上的摘要，只能证明「今天的 yaml 与今天的 matrix 一致」，证明不了「正式 episode 当时跑的就是这些」——同名同库同 policy 但 threshold 或 artifact 路径不同的数据仍可通行。

修复：launch manifest 升为 **v2 ledger**，runner 在 episode 开跑前冻结 arm matrix sha、每个执行 yaml 的 sha、split manifest sha 与 driver 的 `run_id`；resume 追加条目。analyzer 要求：ledger 为 v2、所有条目的冻结摘要一致、run_id 无缺失无重复、**执行时 yaml sha == 分析时实测 sha**、每个 accepted episode 的 `run_id` 落在 ledger 内。为此给 `ConductorDriver` 加了只读 property `run_id`（`src/` 唯一改动，additive，零行为变化）。

### 11.5 plan 残留清理与 q0.05 冻结

§8 测试策略的「两 launch 生命周期 / block / backend / R 功效」整条、analyzer 纪律条里的 `D_l` 与 latency 回归样例、§9 风险登记的三条旧缓解，均已按新口径改写。owner 裁定 Gate 2 的 SR 条件冻结为单侧 `q_0.05`（IUT 两分量同用 α=0.05；`q_0.025` 只是额外保守、白损功效），§4.6 的「owner 未决项」已删除并改为冻结条款。

### 11.6 review_tests 失败的根因（采纳评审订正）

`tests/review_tests/test_cache_size_g2.py` 的失败与本轮无因果（相关文件均未修改），但我此前口头给出的根因**是错的**：我推断为 apool record 指向 weilandserver 绝对路径、本机不存在。评审能读该文件，实测根因是 **fixture 每 task 只造了 1 个 init，而生产代码要求 50 个**（`test_cache_size_g2.py:46`），临时池实际存在。此处如实记录正确根因；该目录对 Execution Authority 密封，我不应在无法读取时给出言之凿凿的诊断。

### 11.7 验证

本会话交付时 `tests/dispatch_surface` + `tests/conductor` 共 252 passed；codex 补全 worker 侧与 `run_precheck` 的完整实现（`FROZEN_LAUNCH_KEYS` / `validate_aprime_pool` / `validate_existing_launch_ledger` / `arms_with_accepted_work_left` 等）后，`tests/dispatch_surface` 为 **144 passed**。定向 ruff 通过。新增测试覆盖 B1–B4 的每一条放行条件，含真实 producer schema 的成功路径与四个对抗场景（stale attempt / fenced 同 attempt / 他 run / 网格外行）。

> **归属说明**：§11.1–11.4 的 driver/analyzer 侧修复出自本会话；`run_precheck.py` 的 A′ 池内容级校验、resume ledger 的 executed-arm 子集语义、accepted-only resume 过滤，以及 §11.3 的 worker 侧索引分离，由 codex 在其后补全。日志如实分记，避免把他人的补全算作本会话的交付。

**待下一轮定向 G2。** 在其通过前不 emit 臂矩阵、不启动闭环。

### 11.8 附带修掉一个本线早前引入的回归（以及它暴露的验证问题）

全仓跑（这次**不带 `-x`**）暴露 `tests/robocasa365/test_groot_cache_collector.py::test_collected_episode_builds_a_loadable_groot_artifact` 失败。查明是 **commit `1985271` 引入的**：该 commit 的 identity seam 给 `InMemoryBackend.artifact_meta` 增加了 7 个 additive 字段（`library_sha256` / `entry_count` / `action_horizon` / `action_dim` / `denoising_num_steps` / `schema_consensus_count` / `intermediates_completeness`），而该测试用**精确 dict 相等**钉住 3 个 builder-identity 键，于是被 additive 字段打破。

当时我处理过同类问题（改了 `tests/cache/groot/test_groot_load_guard.py` 的断言），但漏了这一处。**漏网的原因是验证方法**：那次 Verify 用 `uv run pytest -x`，在既有失败 `tests/exp/test_prebuilt_matrix_backend.py` 处就停了，其后的测试根本没跑到，"2768 passed"因此不是全量绿。**教训：`-x` 适合定位，不适合验收；有已知既有失败时应 `--ignore` 掉它再跑全量，而不是让 `-x` 停在那里。**

修复：把该断言改为按名比较 builder-identity 三元组（`key_builder_type` / `checkpoint_id` / `prompt_pool`），并注明 loader 另记的 additive schema 字段描述的是「加载了哪个库」而非「哪个 builder 造的」，不应让新记录字段读作 identity 变更。全仓已核实只有这一处同类精确相等断言残留（另一处 `test_groot_load_guard.py:232` 用的是 stub backend，不受影响）。

---

## 12. G2 最终复核的直接修复（Review Authority，2026-08-27）

复核没有直接放行 R1 实现；又发现三处会阻断正式运行或削弱 provenance 的问题，并按 owner 授权直接修复。以下修改均留在暂存区外，供 Execution Authority 独立检查。

### 12.1 正式 runner 原本无法执行（已修）

`PrecheckSweepStrategy` 原实现是普通类且用 `graph.stages.append(stage)`；真实 `TaskGraph.stages` 是 dict，因此第一臂构图即抛 `AttributeError`，并且该类没有 stage-begin 的 cache-config load hook。现改为继承 `ExperimentStrategy`、使用 `graph.add_stage(stage)`，在 `on_stage_begin` 加载该 arm YAML；回归测试同时执行 graph validate、20-episode 小图构建和控制面 load roundtrip。

### 12.2 A′ 误用了另一条实验线的 50/task verifier（已修）

原 runner 调用 cache-size 线的 `load_apool_digest`，该契约硬编码 10×50；dispatch 正式 A′ 明确是 10×30。用仓内真实 `test_aprime` 探测会在全部 task 报 `expected 50, actual 30`，因此此前实现无法启动正式预检。现以 dispatch split manifest 为唯一身份权威：逐 task 校验精确 30 条、official index assignment、实际 worker pool 文件集、反序列化后的 state-content sha，以及 raw-file rollup；默认 worker 目录就是 manifest 同级 `test_aprime`。真实仓内 A′ 已通过 300-init 内容复核。

### 12.3 resume ledger 与逐行身份绑定仍可绕过（已修）

原 analyzer 要求所有 launch 的 `executed_yaml_sha256` 完全相等，正常的单臂 resume 会被误拒；与此同时，它只验证第一条 launch 的 contract，并只要求 accepted `run_id` 在 ledger 的任一位置出现，无法证明该 run 实际执行了该 arm。现将 ledger 拆成：

- 实验级冻结字段：suite、完整 core/descriptive roster、trials/replan、policy/library、A′/split/matrix、**完整 arm YAML map**；所有 launch 必须一致；
- 运行级字段：`executed_arms` 与对应 YAML digest；允许 resume 为严格子集。

analyzer 对每条 launch 单独验证 contract，并要求每个 accepted episode 的 `(run_id, arm)` 由对应运行认领。per-step join 同时精确交叉核验 `task_uid` 内 arm/task/subset 与行上的 `yaml_id/task_id/subset_init_state_idx`、canonical `episode_id`、attempt、run_id、boolean accepted；缺字段、未知 `_kind` 或任一身份冲突均拒绝。新增正常单臂 resume 成功路径，以及“run_id 存在但该 run 未执行该 arm”、错误 uid arm、缺 accepted、重复身份字段漂移等对抗回归。

### 12.4 物化 A′ 的执行索引与 provenance 索引被混为一谈（已修）

即使完成 §12.2 的 30/task 内容核验，原 worker 仍固定用 `task.orig_init_state_idx` 取数组元素。这个字段现在是官方 0..49 身份，而 worker 加载的是仅有 30 条的物化 A′；因此会取错状态或直接越界。现给 `WorkerSpec/worker_entry` 增加默认关闭的 `init_state_index_mode=subset`：dispatch runner 开启后，用 `episode_idx` 0..29 选择物化数组，但 per-step 中的 `orig_init_state_idx` 仍保留官方身份。旧调用保持默认 `orig`，命令行字节不变。runner 还把已通过 artifact 契约校验的 `replan_steps` 和冻结 env seed 显式传入 worker，并将 seed 纳入跨 resume 冻结字段；避免 manifest 声称 h_exec/seed 正确而 rollout 实际沿用 worker 默认值。

### 12.5 最终验证与放行结论

- 仓内真实 A′：10 task × 30 init 的文件集、official assignment、state-content digest 与 raw-file rollup 全部通过；总数 300。
- 本线及共享执行边界：`tests/dispatch_surface` + `tests/conductor` + LIBERO episode-runner collect + 本次关联 RoboCasa 回归，**291 passed**。
- 新增回归覆盖：strategy 构图/cache-config load、30/task A′ 成功与改字节拒绝、subset/official 双索引、worker seed/index-mode CLI、accepted-only resume filter、严格子集 resume、run-arm 归属、逐行身份/结果/step 唯一性。
- 定向 Ruff、`git diff --check`、`git diff --cached --check` 全绿。

全仓扩大跑（排除已登记的 `test_prebuilt_matrix_backend.py`）为 **4636 passed / 60 skipped / 12 failed**。12 项均不在本线因果路径：cache-size 密封 review fixture 3 项、RoboCasa 过期 import fixture 1 项、RL-router 2 项、并行开发中的 ws2 4 项、需要下载 tokenizer 的 RoboCasa 环境测试 2 项；本线定向集合全绿。故本次 dispatch surface 修改可进入 Verify/正式启动前的 Execution Authority 复核，但不把“全仓全绿”作为结论。
