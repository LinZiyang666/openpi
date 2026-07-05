# N1 分数滞回门 — Live 验证结果（Stage 1b）

- **Status**: Stage 1b live 完成（8 run × 500 ep），draft 待 owner review
- **Date**: 2026-07-05
- **上游**: `logs/gate_exploration_roadmap.log.md` §5 Stage 1；离线前沿 `exp/gate_research/analysis/n1_offline_frontier.md`
- **数据**: `exp/gate_research/analysis/n1_live_final.md`（analyzer 原始判决表）

---

## Headline

N1（score-hysteresis gate）的 live 验证给出一个**颠覆 roadmap 假设的结果**：

1. **N1 vs always_search（Wave1）**：SR 保真，spatial 甚至 +3.0/+5.2pp，libero_10 ±保真（A −0.6 / B −1.4）；延迟净值 net@34/70 全正。**N1 省搜索无 SR 代价、且有延迟价值 —— 这一层成立。**
2. **N1 vs matched periodic（Wave2, roadmap 及格线）**：**4 config 全 FAIL**——同 skip% 下固定周期盲跳的 SR 全面高于 N1（1.2~6.2pp）。
3. **但延迟轴完全相反**：N1 的 net 全正（+5.1~+17.1），periodic 的 net 全负（低到 −25.7）——periodic 的 SR 优势靠**推理量暴增**换来。

**结论**：roadmap 基于离线 dInf 的「N1 ≫ periodic」假设被 live SR **推翻**；但 N1 与 periodic 实际占据 **SR–延迟 Pareto 的不同点**，非简单优劣。live 揭示了一个离线看不到的机制：**skip 本身有 SR 增益（V2），而均匀盲跳比 score-targeted 跳更能兑现它。**

---

## 实验设计

| 项 | 值 |
|---|---|
| 拓扑 | server = ziyang10（3 replica / spawn-batch 2，H200，**一个 server 靠 yaml 热切换服务两 suite**）；client = timan107（48 worker）；broker force_single_active（expose/pull 走 --ack-alerts） |
| run 矩阵 | 8 run × 500 ep = 2 suite（libero_spatial fh75_ws10 / libero_10 fh5_ws40）× 2 点（A 近免费 / B 激进）× 2 gate（N1 client_controlled / matched periodic） |
| baseline | Stage-0 always_search 同 config 真 verdict，配对键 `(task_id, subset_init_state_idx)`，三方 unit 等集校验 |
| N1 操作点（1a 离线选） | spatial A θ0.968929/j3/M3、B θ0.967327/j1/M5；l10 A θ0.996873/j3/M3、B θ0.996873/j2/M8 |
| periodic matched | \|Δskip\|≤2pp：spatial A cache7/inf1、B cache4/inf1；l10 A cache4/inf1、B cache2/inf1 |

---

## 完整判决（配对 McNemar）

| config | N1 skip/SR/inf | periodic skip/SR/inf | ΔSR(N1−peri) | net@34 N1/peri |
|---|---|---|---|---|
| spatial_A | 12.8 / 85.6 / 0.283 | 10.9 / 90.4 / 0.280 | **−4.8** | +5.6 / +5.8 |
| spatial_B | 20.1 / 87.8 / 0.293 | 18.3 / 89.0 / 0.369 | **−1.2** | +5.1 / −18.1 |
| l10_A | 21.2 / 77.0 / 0.642 | 19.2 / 82.2 / 0.701 | **−5.2** | +5.5 / −12.8 |
| l10_B | 32.4 / 76.2 / 0.655 | 32.7 / 82.4 / 0.759 | **−6.2** | +5.5 / −25.7 |

（SR 为 %；inf = inf_ratio；配对后 baseline SR：spatial 82.6、l10 77.6。skip% 全部兑现离线预测，偏差 <1.5pp。）

**及格线判定**：overall pass 需 SR 保真（ΔSR≥−1pp）**且** 同预算 N1 SR≥periodic。4 config 的 vs-periodic 均 FAIL → **Stage1 roadmap 及格线未通过**。

---

## 科学解读

**(1) skip 本身提升 SR（V2 增益）。** N1 和 periodic 的 SR **都** > always_search（periodic +4.6~+7.8，N1 +3~+5 spatial）。即"跳过一些步走全推理"本身有益——印证 roadmap F7：缓存回放可能重复错误动作，跳步走新推理给轨迹新机会（尤其失败 ep 的边界步）。

**(2) 均匀盲跳比 score-targeted 跳更能兑现 V2。** periodic 的均匀 skip 覆盖了更多"需要新推理"的步；N1 把 skip 集中在 prev_score 低（预测 MISS）的步——这些步本就要全推理，跳它们省了搜索（V1）却没吃到 V2 的 SR 增益。**spatial_A 是最干净的证据**：periodic 与 N1 inf_ratio 几乎相同（0.280 vs 0.283），periodic SR 仍 +4.8——不是靠多推理，是均匀 skip 结构性更优。

**(3) periodic 的 SR 优势在其余 3 config 靠多推理换。** l10_A/B、spatial_B 的 periodic inf_ratio 比 N1 高 0.08~0.12（盲跳撞上 FULL_HIT 步→缓存回放换全推理），SR 换来但**延迟净值崩塌**（net@34 到 −25.7）。N1 只跳预测 MISS 步（本就 inf=1），skip 不增推理，net 全正。

**(4) skip% 不是公平预算轴。** 它把 V1（省搜索延迟）和 V2（提 SR，代价是推理）混为一谈。N1 与 periodic 在「同 skip%」下不可公平比 SR——N1 的 skip 便宜（省 inf）、periodic 的 skip 贵（增 inf）。真正的多目标是 **(SR, inf_ratio, search 延迟)** 三元。N1 = 延迟最优 + SR 保真；periodic = SR 最优 + 延迟崩。

---

## Discussion（反 narrative）

- **本实验不证明「N1 是坏 gate」。** N1 达成了它的设计目标（省搜索、SR 保真、延迟净值正）。它未达成的是 roadmap 设的一个**特定及格线**（同 skip% 打败 periodic SR），而该及格线的预算轴（skip%）事后看不公平。
- **本实验不证明「periodic 是更好的 gate」。** periodic 的 SR 优势 3/4 靠推理暴增换取，延迟净值全负——在延迟敏感/大库/远程部署档它是净负。它只在"SR 优先且不计推理成本"下占优。
- **未测的关键对照**：inf_ratio 对齐下的 N1 vs periodic（把 periodic 的 skip 加大到吃到 N1 的 inf_ratio，SR 会掉多少）。当前只在 skip% 对齐下比，结论受预算轴选择影响。
- **CI 限制**：每 config 500 ep，SR 差 1.2~6.2pp；配对 McNemar；单 seed、单 checkpoint、两个 suite。spatial_B 的 ΔSR −1.2pp 接近噪声，l10_B −6.2pp 稳健。多 seed / 更多 suite 未做。
- **离线-live 一致性**：skip% 4 点全兑现离线（偏差 <1.5pp），说明 N1 状态机 live 行为与离线重放一致；但离线 dInf 的"N1≫periodic"结论（基于 verdict-blind 假设）未能预测 live 的 V2 效应。

---

## 决策推荐（提请 owner）

| 场景 | 推荐 gate | 理由 |
|---|---|---|
| 延迟敏感 / 大库 / 远程 backend | **N1** | net@34/70 正，省搜索无 SR 代价 |
| SR 优先 / 推理成本不敏感 | periodic | SR 高但延迟净负，且靠多推理 |
| 学术结论 | 二者 Pareto 不同点 | 需 inf_ratio 对齐对照才能公平定优劣 |

**方向决策（待 owner，本 agent 不擅自定）**：
- (a) Stage2（N2 追随赢家门）是否续做？其前提（N1 机制成立）在 SR 轴被削弱。
- (b) 是否转向研究 **skip 的 V2 增益机制**（为何均匀盲跳 SR 更高）——这是本实验涌现的新问题，可能比 N1 本身更有价值。
- (c) 是否补 inf_ratio 对齐的 periodic 对照，做公平 Pareto。

---

## Artifact Layout

- **raw（timan107，gitignored）**：`exp/gate_research/data/n1_live/<run_id>/{journal.jsonl, rows.jsonl, manifest.json}`，8 run（4 N1 + 4 periodic）
- **periodic yaml（timan107）**：`exp/gate_research/config/{libero_spatial,libero_10}/periodic_{A,B}/`
- **本地 analysis（可 commit）**：`exp/gate_research/analysis/{n1_live_final.md, n1_live_spatial.md, n1_live_results.md}`
- **待 owner 显式**：raw 数据 rsync 回本地（tier-B 受阻，小文件走 base64-over-exec 已验证）；commit；关 server
