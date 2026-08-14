# E2 — d1 失败 episode 的 winner 尸检

> **两个数据源分开呈现、不合并**（plan §5.2）：
> **§2–§6 secondary** = 既有 `exp/gate_research/data/<suite>/gate_rows.jsonl`（threshold judge，**条件性** estimand）
> **§7 primary** = E4 的 A0 臂附带采集的 per-step（`always_hit`，**无条件** estimand，零额外 rollout）
> 判决层级：**效应量 + CI 估计**（plan §3.3.1c；proxy 功效在最保守基线下仅 0.67 / 0.78）
> 产物：`data/e2/primary_family.json`、`secondary_family.json`、`secondary_family_l10_first.json`

---

## 1. 一句话结论

**libero_10 上，失败 episode 的 winner 显著更常来自"错阶段"：无条件 estimand 下高出 +26.9pp（CI [+21.5, +31.9]，判 `aliasing`），条件性 estimand 下 +16.5pp。libero_spatial 上两个 estimand 都未过 10pp 实用界，判 `inconclusive`。** 两个 suite、两个数据源的"错任务" winner 率均为 **0**，即架构层面的 task_key 过滤没有漏洞，混叠若存在只能是时间维度上的。

**两源的差距本身是结果**：threshold judge 的接受过滤系统性地缩小了效应（§7.3）——secondary 的 `inconclusive` 不是"效应不存在"，而是条件化到了一个更干净的子总体上。

---

## 2. 口径与前置 gate（secondary）

推断单位是 episode，不是搜索步。每个 episode 在**等暴露窗口**内折叠为二值结局 `Y_e^K = 1{该 episode 前 K 个 cycle 内至少出现一次错阶段 winner}`，K 与 W 均在 G1 前由独立批次冻结、本轮不回填。

| 前置 gate | libero_spatial | libero_10 | 结论 |
|---|---|---|---|
| 时间轴校验（`replan_steps = 5`） | 非整除 0 / spacing 异常 0 / 非连续 0 | 同左 | 全过，无 yaml 被 quarantine |
| `winner_id` → 库 entry join | 30,637 / 30,637 = **100%** | 104,319 / 104,319 = **100%** | 全过 |
| **错任务率（data-integrity gate）** | **0** | **0** | 全过 |
| 冻结常量 | K=17, W=6 | K=34, W=8 | 与 plan §3.3.1b/c 一致 |

`replan_steps = 5` 的出处是 `examples/libero/main.py:56` 的 `Args.replan_steps` 默认值，并与 `exp/gate_research/run_n1_live.py:52` 的 `DEFAULT_REPLAN_STEPS` 一致；`run_collect.py` 未覆盖它。该值写入产物的 `replan_steps.provenance` 字段，不靠猜。

**错任务率为 0 的意义**：生产的 `_build_step_filters` 无条件下发 `QueryFilter(task_key=...)`，所以错任务 winner 在架构上不应出现——实测确实为 0，反过来说明 join 与标签管线本身是可信的。也因此，纪要 §6-2 里"高分错任务"那一格恒空，混叠的证据只能由"错阶段"承载。

---

## 3. 主结果（secondary）

| suite | 保留 / 总 episode | 错阶段率差（失败 − 成功） | 95% CI | CMH p | Holm | verdict |
|---|---|---|---|---|---|---|
| **libero_10** | 1790 / 2000 | **+16.48pp** | [+11.68, +21.26] | 6.2e−23 | 拒绝（α=0.025） | `aliasing` |
| **libero_spatial** | 1353 / 1500 | +2.05pp | [−0.07, +4.61] | 0.0023 | 拒绝（α=0.05） | `inconclusive` |

spatial 那一行是本报告最容易被误读的地方：**CMH 在 Holm 后拒绝了零假设，但配对风险差的 CI 仍然含 0，因此不下结论。** 分析管线按预注册规则输出 `inconclusive` 而不是 `aliasing`，这是刻意的——CMH 检验的是分层下的关联，与"率差是否可与 0 区分"不是同一个量，只报前者会把一个 2pp 的效应说成已确证。

次要 estimand（quasi-binomial GLM，`cbind(count, n−count) ~ success + factor(task_id)`，无 offset）与主结果同向：

| suite | n | `coef_success` | 95% CI | dispersion |
|---|---|---|---|---|
| libero_10 | 2000 | −1.122 | [−1.257, −0.988] | 14.89 |
| libero_spatial | 1500 | −0.633 | [−0.913, −0.353] | 6.00 |

系数为负 = 成功 episode 的错阶段计数更低。两个 suite 的 dispersion 都远大于 1，这正是不用二项 GLM 的原因。

---

## 4. 敏感性与混淆强度（secondary）

### 4.1 W 敏感性——spatial 的方向会翻转

| suite | W−2 | **W（冻结）** | W+2 |
|---|---|---|---|
| libero_spatial | W=4 → **+5.12pp** | W=6 → +2.05pp | W=8 → **−0.17pp** |
| libero_10 | W=6 → +15.66pp | W=8 → +16.48pp | W=10 → +12.85pp |

**libero_10 的效应在 W 的整个敏感性范围内稳定（12.8–16.5pp），libero_spatial 的则从 +5.1pp 一路翻到 −0.2pp。** 这是 spatial 判 `inconclusive` 的独立佐证：spatial 上的效应量与"多远算错阶段"这个定义选择同量级，不构成稳健的现象。

### 4.2 暴露长度混淆（R19）

| suite | K | 丢弃的短 episode | 其中成功占比 | 成功组 cycle 中位 | 失败组 cycle 中位 |
|---|---|---|---|---|---|
| libero_spatial | 17 | 147 / 1500 | 83.0% | 21.0 | 23.0 |
| libero_10 | 34 | 210 / 2000 | 69.5% | 50.5 | **70.0** |

失败组的 cycle 数系统性更多（libero_10 上 70 vs 50.5），这正是 primary 必须用等暴露窗口 `Y_e^K` 而不是"整条 episode 至少一次"的原因——后者即使在零假设下也会随暴露机械上升。代价是 K-window attrition：被丢弃的短 episode 里成功占多数（spatial 83%），所以**保留集相对总体略偏向失败**，这会让两组的基线率都上移，但不改变组间差的方向。

---

## 5. Discussion — 本实验**不**能证明什么

- **不能证明因果。** 这是观察性的尸检：错阶段 winner 与失败共同出现，不等于错阶段 winner 导致了失败。反向解释同样成立——episode 一旦开始偏离，其查询就落到库覆盖稀疏的状态上，于是最近邻自然来自别的阶段。本设计无法区分这两者。
- **estimand 是条件性的，不是无条件的。** 本报告的样本被过滤为 `searched == true` 且 `hit_type ∈ {FULL_HIT, WARM_START}`——即 threshold judge **接受**了的那些步。judge 的接受本身与"查询是否处在库覆盖良好的区域"相关，所以这里的率差是在"被接受的搜索步"这个子总体上的。无条件版本要等 `always_hit` 的 primary。
- **spatial 上没有结论。** 见 §3、§4.1：CI 含 0 且方向随 W 翻转。**不得**把它写成"spatial 也有混叠，只是弱一些"。
- **与 E3 的表面张力必须并列呈现。** E3 在同一批库上测得高相似对的动作分歧率极低（spatial 0.24%、libero_10 2.79%，均判 `almost_no_aliasing`）。E3 问的是"key 空间里相似的东西动作是否相似"，E2 问的是"失败时取回的东西阶段是否对"——前者说静态混叠几乎不存在，后者说失败时取回物的时间位置更偏。两者并不矛盾，但**任何把 E2 的结果读成"静态混叠是失败主因"的写法都与 E3 直接冲突**，合并解读放在 `synthesis.md`。
- **库的 `outcome` 全为 `None`。** 该 artifact 不带 per-entry 成功标签，因此"取回的是失败轨迹的帧"这一类主张在本数据上无法检验。
- 功效不在本轮重算。plan §3.3.3 预先把判决层级定为"效应量 + CI 估计"，正是为了避免拿到结果后再论证功效够不够。

---

## 6. 产物清单

```
exp/markov_sufficiency/data/e2/                     # gitignored
├── secondary_family.json            # suite A = libero_spatial（含 spatial 的 W 敏感性）
├── secondary_family_l10_first.json  # suite A = libero_10（含 l10 的 W 敏感性）
├── task_map_libero_spatial.json     # task_id -> task_key
├── task_map_libero_10.json
└── task_map_provenance.txt          # 映射取自 LIBERO benchmark 本身，并与库的 task_key 集合对拍
```

两份产物的 `family` 块互为镜像、数值一致；分开跑只是因为 CLI 的 W 敏感性只对 suite A 计算。

复现命令见 `logs/markov_sufficiency_plan.log.md` §13.5。

---

## 7. Primary（`always_hit`，无条件 estimand）

数据源：E4 的 A0 臂（d1 + `step_filter: all` + `always_hit`）附带采集的 per-step 记录，两 suite 各 **950 episode**（官方 pruned 池 500 + db_init 池 450，跨池的 episode 键经偏移隔离）。零额外 rollout 成本。

### 7.1 前置 gate（与 secondary 同一套，全部通过）

| gate | libero_spatial | libero_10 |
|---|---|---|
| 时间轴（`replan_steps=5`） | 非整除 0 / spacing 异常 0 / 非连续 0 / quarantine 0 | 同左 |
| `winner_id` → 库 join | 27,960 / 27,960 = **100%** | 76,648 / 76,648 = **100%** |
| **错任务率** | **0** | **0** |
| A0 的 MISS 率 | **0%**（无 `step_filter`，每步都有 winner） | **0%** |

A0 臂没有 `step_filter`，所以 plan §5.2 的"`always_hit` 下每步都有 winner"在这里成立（带 filter 的臂不成立，见 `e4_index_filter.md` §3）。

### 7.2 判决

| suite | 保留 / 总 episode | 错阶段率差（失败 − 成功） | CI | CMH p | Holm | verdict |
|---|---|---|---|---|---|---|
| **libero_10** | 904 / 950 | **+26.85pp** | [+21.52, +31.85]（97.5%） | 1.0e−17 | 拒绝 | **`aliasing`** |
| libero_spatial | 924 / 950 | **+6.88pp** | [+4.17, +9.85]（95%） | 6.5e−06 | 拒绝 | `inconclusive` |

spatial 的区间**排除 0**，但 +6.88pp 未达预注册的实用界 δ_E2 = 10pp，故按规则仍判 `inconclusive` —— 不因为"显著"就改口。

对齐份额：libero_10 仅 **65.8%**（即约三分之一的搜索步取回的是错阶段的 entry），spatial 92.8%。

次要 estimand（quasi-binomial GLM）同向且更强：libero_10 `coef_success = −2.482`（CI [−2.655, −2.308]）、spatial `−3.304`（CI [−3.531, −3.077]）。

### 7.3 primary 与 secondary 的差异，及其含义

| suite | secondary（threshold judge，条件性） | **primary（always_hit，无条件）** |
|---|---|---|
| libero_10 | +16.48pp [+11.68, +21.26] | **+26.85pp** [+21.52, +31.85] |
| libero_spatial | +2.05pp [−0.07, +4.61]（含 0） | **+6.88pp** [+4.17, +9.85]（排除 0） |

**无条件 estimand 下效应显著更大。** secondary 的样本被限制在 threshold judge **接受**了的搜索步上，而 judge 的接受与"查询是否落在库覆盖良好的区域"直接相关——它系统性地滤掉了最可能出问题的那些步。这正是 owner 当初裁决"补采 `always_hit`"的价值：secondary 的 `inconclusive` 不是"效应不存在"，而是条件化到了一个更干净的子总体上。

两源结果分开呈现、不合并（plan §5.2）。

### 7.4 暴露长度混淆在 primary 上同样存在

| suite | K | 丢弃的短 episode | 成功组 cycle 中位 | 失败组 cycle 中位 |
|---|---|---|---|---|
| libero_spatial | 17 | 26 / 950 | 21.0 | **44.0** |
| libero_10 | 34 | 46 / 950 | 50.0 | **104.0** |

失败组的暴露长度是成功组的约 2 倍，等暴露窗口 `Y_e^K` 因此是必需的而非保守选择。

### 7.5 W 敏感性（spatial）：primary 上方向稳定

| W−2 | W（冻结=6） | W+2 |
|---|---|---|
| +12.78pp | +6.88pp | +2.19pp |

与 secondary 不同（那里 W=8 时符号翻转为 −0.17pp），**primary 上三个 W 取值方向一致为正**。spatial 的结论在 primary 数据上比 secondary 稳健，但仍未过 10pp 实用界。
