# Gate 探索路线图 — 基于 always-search 真 verdict 数据的方案判决与阶段名单

- **Status**: Data-Grounded Roadmap — Stage 0 ✅ / Stage 1 ✅（live 判决 2026-07-05）/ Stage 2 ✅（三线机制离线判决 2026-07-05，见 §5 回填）/ Stage 3a ✅（N4 live 胜出 L=6，4/6 pass，2026-07-05，见 §5 回填）/ Stage 3b ✅（N4 服务器化 `ScoreHysteresisGate`+L，G1/G2 APPROVED，2026-07-06，见 §5 回填）/ Stage 4a 🔨（N2 追随赢家门服务器化，Phase A F5 复测 GO + G1/G2 APPROVED 2026-07-06，见 §5 回填；**Phase C live 未跑**）/ Stage 4b+ 待
- **Date**: 2026-07-04（创建）/ 2026-07-05（Stage 1b live 判决修订：F9–F12、C10/C11、N2 降级、§5 重排）
- **Stage 1b live 数据**: 8 run × 500 ep（N1 A/B × 2 suite + matched periodic × 4，同 conductor/init 配对）；报告 `exp/gate_research/analysis/n1_live_results.md`、判决表 `n1_live_final.md`（commit `71f2b22`）；raw 已本地化 `exp/gate_research/data/n1_live/`（gitignored）
- **前身**: `cache_gate_design_brainstorm.log.md`（2026-07-02/03 头脑风暴，git 历史 `437bbc2` 可查）。本文件是其数据判决版：brainstorm 的方案谱系（G0/A/B/C/D）逐项对撞实测数据后重排，原文仍有效的资产（延迟账本、oracle 口径、RPG 基线、约束公理）已整合进来，不再回读原文。
- **数据**: `exp/gate_research` 采集（2026-07-04）——weighted_sum d1 Pareto 前沿 7 个标志性 config（libero_spatial 3 + libero_10 4）× 500 ep（全 50 inits 0..49），`gate: always_search` + 真 ThresholdJudge verdict，**182,899 决策步**（libero_spatial 39,136 + libero_10 143,763；"步" = 一次 CP1 决策 = 一个 action chunk ≈ 10 env steps），每步含 `robot_state[32] / hit_type / cp1_score / start_t / winner_id / success`。无选择偏置（C5 满足：always_search 采集）。
- **复现**: `exp/gate_research/gate_structure_analysis.py <gate_rows.jsonl> <suite>`（本文所有表）；`verify_gate.py`（采集完整性）；采集报告 `exp/gate_research/analysis/gate_research_results.md`。
- **Level**: 本文档为研究产物（L0 纯文档）。任何入选方案的实现仍为 L2（新 GateFunction 组件），需另走 Plan → G1 → Code → G2 → Verify。
- **Owner 指令**: training-free gate 先行探索；需要训练/标定的方案（C1/D1）押后。

---

## 0. TL;DR — 数据把 gate 问题改写成了什么

1. **全场最强信号是免费的**：上一步搜索的 `cp1_score` 预测本步 MISS 的 **AUC = 0.973–0.986**（7 config 全一致）。它不用算任何新东西——上一步搜索已经产出了它。gate 问题从"预测这一步难不难"塌缩成"**读昨天的温度计**"。由此产生本路线图的主打方案 **N1 分数滞回门**（training-free）。
2. **verdict 时序是块状的，不是逐点随机**：P(MISS|上一步 MISS)=0.89–0.93 vs P(MISS|上一步命中)=0.03–0.05；MISS 段均长 6–12 步，命中段 13–22 步；FH 几乎从不直接掉 MISS（P(FH→MISS)=0.00–0.02），标准降解路径是 **FH → WS（预警带）→ MISS**。
3. **命中段与单条库轨迹完美 lockstep**：连续 FULL_HIT 时 93–98% 是同一条库 episode、其中 75–97% winner step 恰好 +1。这打开了一个 brainstorm 没有的新象限——**N2 追随赢家门**：锁定 winner 后短预算盲回放，把命中段的 search 本身也省掉（省量上限从 MISS% 的 19–38% 翻到命中段的 60–80%）。
4. **B3 复用债务门的核心前提被驳回**：P(MISS | 连续命中 L 步) 不随 L 上升——L1-2 反而最高（0.04–0.17，刚恢复/边界带），L≥3 后平坦（0.02–0.07）。**在 verdict 监督下"复用债务"不积累**；"连续命中"是安全信号（AUC 反向 ≈0.9）。B3 重新定位为 N2 盲回放段的安全绳（那里才真的没有 verdict 监督），不再作为独立 gate 立项。
5. **B4 停滞门、N3 首步印象门驳回**（AUC≈0.5）；**A1 步位置**信号真实（AUC 0.78–0.90）但被 prev_score 完全覆盖且硬阈值化损失巨大——降为特征；**A2 任务先验**步级弱（AUC 0.63–0.67）、episode 级预算重分配保留为零成本离线验证项。
6. **V1（省延迟）的判决是分档的**：miss-skip 型 gate 的延迟净值 = skip%×search_ms − dInf×infer_ms。实测三档：优化栈小库（search 4 ms）**全负**；stock 栈（34 ms）probe 版 **+1.5~+7.5 ms/步**；大库 50k（70 ms）**+7~+20 ms/步**。**结论：miss-skip gate 只在大库 / stock / 远程 backend 部署档有 V1 价值；当前优化小库档不值得。** N2（hit 段省搜）不受此限（省的是 60–80% 的步）。
7. **oracle 与可实现的距离已量化**：oracle（只跳真 MISS）= 免费省 19–38% 搜索。sticky 计数规则在 probe=5 档做到 skip 11–31% / dInf +0.004–0.017；N1 用 AUC 0.98 的连续分数预期进一步支配该前沿（Stage 1a 离线扫描确认）。
8. **反事实口径**（一切离线数字的边界）：跳过真 MISS 步不改变动作（MISS 本来就全推理）→ 该部分离线数字**精确**；错跳的命中步把缓存回放换成新推理 → 轨迹反事实，dInf 是近似、SR 影响必须 live。**高 precision（低 lost）档位的离线结论最可信；N2 盲回放整体改变执行流，只能 live 评。**

**Stage 1b live 判决后记（2026-07-05；详见 §2 F9–F12 与 §5 重排）**：

9. **N1 自身承诺兑现**：vs always_search，SR 保真甚至提升（spatial A/B **+3.0/+5.2pp**、l10 A/B −0.6/−1.4pp，后二者 McNemar 不显著）；4 点 live skip% 与 1a 离线预测偏差全 ≤1.8pp（离线选点方法论成立）；Δinf ≈ 离线预测；net@34/70 全正。
10. **但本文原及格线 4/4 FAIL**：同 skip% 下 matched periodic 的 SR 全面高于 N1（+1.2~+6.2pp）。同时**延迟轴完全反转**：periodic 的 net@34 3/4 全负（至 −25.7 ms/步）——其 SR 优势多数靠推理暴增（Δinf +0.06~+0.12）购买；唯 spatial_A 档 periodic 在**同 inf_ratio** 下仍 +4.8pp（结构性优势，非买来）。
11. **新机制浮出（V2）**：skip 本身带 SR 增益——N1 与 periodic 的 SR **都** ≥ always_search。解释方向：缓存回放把轨迹锁在库轨迹上重复既往动作；周期性注入新推理给轨迹"回到自身流形"的机会（F7 的 live 印证）。**均匀盲跳比 score-targeted 跳 MISS 更能兑现 V2**——MISS 步本来就全推理，跳它省搜索但不改动作分布（这正是 N1 SR 保真的原因，也是它吃不到 V2 的原因）。
12. **判决改写问题本身**：skip% 不是预算轴（N1 的 skip 免推理、periodic 的 skip 费推理，C10）；N1 = 延迟最优 + SR 保真，periodic = SR 最优 + 延迟崩——Pareto 不同点。gate 问题从"何时省搜索"升维为"**(SR, inf_ratio, 延迟) 三元下的执行调度**"。

**探索名单（2026-07-05 重排，training-free 先行不变）**：Stage 1 ✅ = N1 滞回门（1a 离线前沿 → 1b live 8 run → 1c 服务器化，全部完成）；Stage 2 = **V2 增益机制研究**（纯离线，已有数据 0 GPU：H1/H2/H3 裁决 + 公平 Pareto）；Stage 3 = **N4 混合门**（N1 跳 MISS + 定期注入新推理）live；Stage 4（押后）= N2 追随赢家（降级，理由见 §4）/ C1 / A3 / D1。详见 §5。

---

## 1. 数据资产与口径

| 项 | 值 |
|---|---|
| 采集 | 7 config（d1 前沿标志点）× 500 ep = 3,500 ep；libero_spatial 39,136 步 + libero_10 143,763 步 |
| 每步字段 | `robot_state[32]`（模型输入侧）、`hit_type/cp1_score/start_t/winner_id`（verdict 侧）、`searched=true`、episode `success` |
| 无偏性 | gate=always_search（C5）；库内 init（5/task）vs held-out FH 率无差（54–62% vs 61–71%，libero_10 基本持平）→ 全量 500 ep 可用 |
| verdict 构成 | FH 28–70%，WS 5–34%，MISS 19–38%；精确 inf_ratio（warm cost=1−0.5(1−t)）：spatial 0.270/0.287/0.351，libero_10 0.314/0.369/0.417/0.636 |
| 延迟账本 | gate 可省段 search+judge+fetch：优化栈 ~4 ms、stock ~34 ms、50k 库外推 ~70 ms（`exp/cache_latency_bench/analysis/latency_breakdown.md`）；S2+S3 纯 GPU ~300 ms 量级，closed-loop ~1.7 s |
| 步的定义 | 1 步 = 1 次 CP1 决策（action chunk ≈10 env steps）——所有 run 长度按决策步计 |

**反事实结构**（决定哪些结论离线可信）：
- 跳过**真 MISS** 步：动作不变（MISS 本来走全推理）→ 轨迹不变 → 离线模拟**精确**。
- 错跳**命中**步：动作从缓存回放变为新推理 → 后续反事实 → 离线 dInf 为一阶近似（假设 verdict 序列不变），**SR 影响只能 live 测**。lost 越小离线越可信。
- **N2 盲回放**改变整个执行流 → 离线只能给机会规模（persistence 统计），效果必须 live。

---

## 2. 数据发现 — F1–F8（Stage 0 采集，7 config × 2 suite，跨 config min–max）+ F9–F12（Stage 1b live，本节末）

### F1. 上一步分数近乎完美预测本步 MISS（G0b 白搜目标的答案）

| 信号（training-free，线上可得） | AUC→MISS | 备注 |
|---|---|---|
| **上一步 cp1_score（连续）** | **0.973–0.986** | 免费（上一步搜索已产出）；需 verdict 回传（G0a hook 或 exp 层 ClientControlledGate——客户端已有 `__hit_meta__.cp1_score`，零 src） |
| 近 3 步分数均值 | 0.959–0.979 | 平滑略损即时性 |
| 上一步是否 MISS（二值） | 0.917–0.946 | sticky 计数规则的信号基础 |
| step_idx（A1） | 0.780–0.895 | 真实但被 prev_score 覆盖；硬阈值化 lost 45–73%（前轮 rule study） |
| task 先验（A2，in-sample 上限） | 0.626–0.669 | per-task MISS% 差异真实（7–56% spread）但步级弱 |
| 首 3 步分数（N3 首步印象） | 0.520–0.603 | ≈无信号 |
| 运动量 ‖Δrobot_state‖（B4 停滞） | 0.397–0.541 | ≈无信号甚至反向；LIBERO 无可检测的卡死形态 |
| 连续命中长度（B3 债务） | 0.049–0.101（反向 ≈0.9） | **方向与 B3 假设相反**：命中越久越安全 |
| （对照）robot_state 32 维线性探针 | 0.76–0.86 | 输入侧学习模型的上限参考——**低于免费的 prev_score**（D1 押后的核心依据） |

### F2. 三态转移矩阵：块状 + WS 是预警带

- P(FH→FH)=0.91–0.96，**P(FH→MISS)=0.00–0.02**（FH 从不突然死亡）；
- P(MISS→MISS)=0.89–0.93（MISS 同样自持）；
- WS 自持 0.41–0.86，WS→MISS 0.09–0.35，WS→FH 0.05–0.24 —— **FH→WS→MISS 是标准降解路径**，score 进入 WS 带即预警。

### F3. 段结构与恢复：停搜必须配 probe

- MISS 段：均长 6.4–11.7（中位 2–4，p90 20–38）；命中段：均长 12.7–22.4。
- **61–84% 的 MISS 段之后会恢复命中**（恢复后命中段均长 6.0–9.8 步）——libero_10 恢复率更高（76–84%，长程多段任务在子任务接缝处 MISS、段内恢复）。→ **永久停搜丢掉大量恢复段；probe（停搜期间周期性试探）是必需件**。
- episode 形态：spatial 以 all-hit（47–55%）+ 少块（25–34%）为主；libero_10 振荡型显著（oscillating 24–36%）。变点（首个 ≥3 步 MISS 段）中位发生在轨迹 29–50% 处。

### F4. B3 债务假设的直接检验（驳回）

P(MISS | 连续命中已 L 步)：L1-2 = 0.042–0.174（最高，刚恢复/边界带）→ L3-5 = 0.009–0.047 → L6+ 平坦 0.016–0.074。**无单调上升**。在 verdict 每步监督下，"复用越久风险越积累"不成立；长命中段 = 稳定 lockstep 段 = 最安全区。B3 的真正用武之地是 verdict **不在场**的盲回放段（见 N2）。

### F5. Winner 持久性：命中段是 lockstep 回放（N2 的结构基础）

连续 FH 步对中 **93–98% 同一条库 episode**；其中 winner step **+1 占 75–97%**（Δ0 占 2–20%，重规划密集段）。live 轨迹在命中段内逐步跟随同一条库轨迹前进。配合 F2（FH→MISS 直跳 0–2%），**短预算盲回放（不搜索直接执行 winner 的后续动作）在命中段内结构性安全**。

### F6. miss-skip 规则前沿与三档延迟净值（V1 判决）

sticky "连续 K 次 MISS 停搜 + 每 M 步 probe" 离线前沿（代表点，全 7 config 范围）：

| 档位 | skip%（省搜索） | dInf（精确口径） | net@4ms | net@34ms | net@70ms |
|---|---|---|---|---|---|
| K=3, probe=5（高 precision） | 13–28% | +0.006–0.011 | −1.1~−2.7 | **+2.5~+7.3** | **+7.2~+17.5** |
| K=2, probe=10（高 skip） | 19–38% | +0.020–0.036 | −4.5~−9.9 | −2.4~+6.8 | +6.0~+20.3 |
| K=∞ 对照（永停不 probe） | — | +0.019–0.101 | 全负 | 多数负 | 部分正 |

net = skip%×search_ms − dInf×300ms（ms/步）。**判决：优化栈小库下 miss-skip gate 延迟净值为负（V1 不成立）；stock/大库/远程档为正且随库规模放大**——brainstorm §2.1 的"随部署栈放大"现在有了精确的分档数字。oracle（只跳真 MISS）= 免费省 19.4–37.9% 搜索，N1 预期把可实现前沿推向它。

### F7. 成功纠缠与安全面

episode 级：成功 ep 的 MISS% = 5–24%，失败 ep = 34–59%。sticky 模拟中错跳的命中步落在失败 ep 的比例 13–76%（config 相关）。含义双面：(a) 停搜集中在"已偏离"轨迹上，对成功轨迹扰动小；(b) 失败 ep 里被强制换成真推理的步在反事实上可能**有益**（V2 方向）——但这只能 live 验证，离线数据对此沉默。

### F8. 输入侧信号被历史信号支配（C/D 类押后的依据）

robot_state 线性探针 AUC 0.76–0.86、cp1_score~robot_state 岭回归 R²=0.39–0.48：输入侧确有信息，但全面低于免费的 prev_score（0.98）。任何"从输入预测难度"的训练模型（D1）要在**开局无历史**或**跨 suite 泛化**上找增量，步级主战场已被历史信号占领。

### Stage 1b live 发现（2026-07-05；8 run × 500 ep；N1 = client 侧 `N1GateState`，periodic = server 侧 `PeriodicGate`；同 conductor / 同 inits，配对键 `(task_id, subset_init_state_idx)`）

**判决表**（SR/skip 为 %；inf = 实测 inf_ratio；net 单位 ms/步；配对 baseline = Stage-0 always_search：spatial 82.6 / l10 77.6）：

| config (periodic 比) | N1 skip/SR/inf | periodic skip/SR/inf | N1−peri ΔSR | net@34 N1/peri |
|---|---|---|---|---|
| spatial_A (7:1) | 12.8 / 85.6 / 0.283 | 10.9 / **90.4** / **0.280** | **−4.8** | +5.6 / +5.8 |
| spatial_B (4:1) | 20.1 / 87.8 / 0.293 | 18.3 / 89.0 / 0.369 | −1.2 | +5.1 / −18.1 |
| l10_A (4:1) | 21.2 / 77.0 / 0.642 | 19.2 / 82.2 / 0.701 | **−5.2** | +5.5 / −12.8 |
| l10_B (2:1) | 32.4 / 76.2 / 0.655 | 32.7 / 82.4 / 0.759 | **−6.2** | +5.5 / −25.7 |

#### F9. N1 兑现自身设计目标（vs always_search）

SR：spatial A +3.0（McNemar 连续校正 p≈0.068）/ B **+5.2（p≈0.0015）**；l10 A −0.6（p≈0.82）/ B −1.4pp（p≈0.48——及格线上 FAIL，统计上噪声级）。skip% 离线→live：13.3→12.8 / 21.9→20.1 / 20.4→21.2 / 32.2→32.4（偏差 ≤1.8pp）——**N1 状态机 live 行为与离线重放一致，1a 离线选点方法论被验证**。Δinf −0.004~+0.018 ≈ 离线预测；net@34/70 全正。1c 服务器化（`ScoreHysteresisGate`）已落 src。

#### F10. 原及格线 4/4 FAIL：同 skip% 下 periodic SR 全面反超

matched periodic（|Δskip|≤2pp 全满足）SR 高于 N1 1.2~6.2pp，且全部高于 baseline +4.6~+7.8pp。原判"N1 以 0.98 vs 0.93 的信号优势支配前沿"（TL;DR 7）在 **SR 轴**被 live 推翻——离线 dInf 只计"错跳命中步的推理成本"，完全没建模 skip 对**动作分布/轨迹**的正向效应。

#### F11. 延迟轴完全反转：periodic 的 SR 多数是买来的

periodic Δinf +0.064~+0.123（盲跳撞 FULL_HIT/WS 步 → 缓存回放换全推理），net@34 −12.8~−25.7；N1 只跳预测 MISS 步（本就 inf=1.0），skip 免推理，net@34 全 +5.1~+5.6。**例外 = spatial_A**：periodic 7:1 在 inf 0.280（≤ N1 0.283 ≤ baseline 0.287）下 SR 仍 90.4——同 inf 轴的**结构性**优势，该点在 RPG 坐标下严格支配 baseline 与 N1；粗插值它比既有 spatial d1 前沿（fh75_ws15 0.270/85 ↔ fh40_ws40 0.351/91 连线）高 ~4.7pp @同 inf，l10 两点比 fh5_ws40↔纯推理 anchor 连线高 ~2.7-3.3pp——**"gate 抬高既有 config 前沿"为初判，正式口径核对留 2b**（anchor 协议差异、跨 config 插值均未严格化）。

#### F12. V2 机制：skip 是 SR 干预；均匀注入 > 定向跳 MISS

两种 gate 的 SR 都 ≥ baseline ⇒ "跳过搜索、走新推理"本身有 SR 增益（V2），与 F7（失败 ep 中被强制换真推理的步反事实可能有益）同向。N1 把 skip 集中在预测 MISS 步——动作分布几乎不变（SR 保真的原因 = 吃不到 V2 的原因）；periodic 的 skip 均匀落进缓存执行段，等效于**限制最大连续缓存执行长度**（7:1/4:1/2:1 → cap ≤7/4/2；而 baseline/N1 的命中段 12.7–22.4 步全程缓存执行，F3）。**剂量已见饱和**：spatial 7:1（12.5% 名义剂量）SR 90.4 ≥ 4:1 的 89.0，l10 4:1 ≈ 2:1（82.2/82.4）——最低试验剂量即达满增益，低剂量注入是 N4 的先验甜点。三个候选机制假设（Stage 2a 裁决，全部可用已有数据离线检验）：

- **H1（剂量/截断）**：SR 增益 ~ 连续缓存执行 run 被截断的程度（rows 直接可算 run-length 分布 × ep 成败 × 3 档剂量）。
- **H2（on-manifold 反馈）**：定期新推理把轨迹拉回自身流形 → 后续搜索命中更好（spatial_A periodic 的 inf 反降 0.280<0.287、searched-step FH 率是证据坑）。
- **H3（WS 执行中毒）**：SR 损失集中于 WARM_START 部分去噪回放的执行；均匀 skip 顺带打断它（与 1a"WS-aware probe 0 增益"不矛盾——那是 probe 侧，这是执行侧）。

**公理增补（2026-07-05）**：

- **C10（预算轴）**：任何"同预算"比较必须声明在 (SR, inf_ratio, search 延迟) 三元的哪根轴对齐。**skip% 不是预算轴**——N1 的 skip 免推理、periodic 的 skip 费推理，同 skip% ≠ 同成本。1b 把"同预算"操作化为同 skip% 是本轮最大方法论教训。
- **C11（skip=干预）**：gate 决策改变动作分布、自带 SR 效应（V2）；任何 gate 评估禁止假设 SR 中性，必须同时报 (SR, inf_ratio, net)。离线 dInf 类模拟只对"动作不变"的跳步（真 MISS）可信——C8 的收紧版。

---

## 3. Brainstorm 方案逐项判决

| 方案 | 前提（brainstorm） | 数据判决 | 处置 |
|---|---|---|---|
| **G0a hook 补丁** | gate 缺 task_key / verdict 回传 | N1/N2/B3 全依赖 verdict 回传 → 重要性**上升** | ✅ 保留，随下一个 src 窗口捎带（exp 层原型不等它：客户端已有 `__hit_meta__`） |
| **G0b 离线信号研究（P0）** | 廉价信号预测力未知 | **本文 §2 即 G0b 的完成形**（白搜目标 AUC 榜）；危险步目标（deviate_score oracle 标签 join）未做，列为可选补充 | ✅ 已完成（白搜目标） |
| **A1 步位置门** | 相位能预测难步 | AUC 0.78–0.90 真实，但被 prev_score 覆盖、硬阈值 lost 45–73% | ⤵ 降为特征（并入 Stage 3 C1 特征集），不立项——与 brainstorm 判断一致，数据加固 |
| **A2 任务先验门** | task 间难度差异大 | per-task MISS 7–56% spread 真实；步级 AUC 仅 0.63–0.67 | ⤵ 步级不立项；**episode 级预算重分配**保留为 Stage 2 零成本离线验证 |
| **A3 库覆盖门** | 库密度可预测白搜 | 未直测（需离线密度表）；proxy（robot_state→score R² 0.43）中等；prev_score 掩盖步级增量；独特位置=开局无历史段 + 大库档 | ⏸ Stage 3 候补，优先级随库规模（50k 计划落地时上调） |
| **B3 复用债务门（原 P1 首推）** | 连续命中积累漂移债务 | **前提驳回**（F4：hazard 平坦/反向；连续命中=安全信号） | ✖ 独立 gate 撤销立项；**机制降格为 N2 盲回放段的预算安全绳**（那里 verdict 不在场，债务假设才成立） |
| **B4 停滞检测门** | 卡死形态可检测且危险 | AUC 0.40–0.54 ≈ 无信号/反向；500 ep 内无可检测卡死形态 | ✖ 驳回（防灾属性本数据无法证实，无正信号不立项） |
| **C1 标定组合门** | 多信号标定+conformal 预算旋钮 | 前景被数据强化（prev_score 0.98 为主特征），但属标定/训练类 | ⏸ 押后至 Stage 3（owner 指令：training-free 先行）；特征集由 Stage 1/2 幸存者决定 |
| **D1 学习难度门** | 廉价信号不足时上 MLP | **立项条件不成立**（免费信号 0.98 > 输入侧学习上限 0.86） | ✖ 大概率不立项；仅当出现"开局段/跨 suite"的明确需求再议 |

原公理集 C1–C7 全部继承，新增两条：

- **C8（反事实口径）**：离线评估只对"高 precision 跳真 MISS"型结论精确；任何改变命中步执行的方案（含 N2）必须 live 验证 SR。
- **C9（货币分档）**：V1 结论必须按部署档报告（4/34/70 ms 三档净值表），不允许单档结论外推。

---

## 4. 新方案（数据涌现，training-free）

### N1. 分数滞回门（ScoreHysteresisGate）— Stage 1 主打

- **规则**：维护上一次搜索的 `cp1_score`。score 跌破 θ_low（或连续 j 步落在 WS/MISS 带）→ 停搜（该步直接全推理）；停搜期间每 M 步 probe 一次搜索；probe score ≥ θ_high → 恢复正常搜索（双阈值滞回防抖）。
- **依据**：F1（prev_score AUC 0.973–0.986）+ F2（FH→WS→MISS 降解路径，θ 可直接钉在已知的 fh/ws 阈值坐标系上，可解释）+ F3（61–84% 恢复率 → probe 必需）。
- **成本**：零训练、零新计算（一个标量寄存器 + 比较）；在线成本远低于 C2 的 1 ms 红线。
- **预期**：以 0.98 vs 0.93 的信号优势支配 sticky-K 计数前沿（F6 表），把 (skip, dInf) 推向 oracle（免费省 19–38% 搜索）。
- **落地**：exp 层 `ClientControlledGate`（客户端已有 `__hit_meta__.cp1_score`，零 src 改动，step3 已趟通此路）；定型后经 G0a 服务器化。
- **失败模式**：score 在阈值带内高频振荡 → 滞回带宽 (θ_high−θ_low) 扫描解决；suite 结构差异（libero_10 振荡多）→ per-suite 的 M。
- **live 判决（2026-07-05，F9–F11）**：设计目标全兑现（SR 保真 + 延迟净正 + 离线选点精确迁移），已服务器化（1c `ScoreHysteresisGate`）；但同 skip% 的 SR 及格线输给 periodic（F10）。**定位修正**：N1 是**延迟档位**（stock/大库/远程）的正确工具，不是 SR 档位的；A 点可直接部署，l10 的 B 点不推荐（−1.4pp，虽不显著）。后续并入 N4（Stage 3）作为"免推理跳 MISS"分支。

### N2. 追随赢家门（FollowWinnerGate / lockstep 盲回放）— Stage 2 主打，V1 上限最大

- **规则**：连续 j 步 FULL_HIT 且同 winner、Δwinner_step=+1（lockstep 证据）→ 锁定该库 episode，接下来 M 步**不搜索**直接回放 winner 的后续动作；预算 M 用完或恢复搜索的 probe 不再命中/掉带 → 解锁回 always_search。M 为盲回放债务预算（**B3 机制的正确用武之地**：此段 verdict 不在场）。
- **依据**：F5（persistence 93–98%、Δ+1 75–97%）+ F2（FH→MISS 直跳 0–2%：锁定段内突然失效概率极低）+ RPG 的 periodic>random（成块结构有效）——N2 即 periodic「cache k 步 / infer n 步」的事件驱动智能版。
- **价值象限**：miss-skip 型省量上限 = MISS%（19–38%）；N2 省的是**命中段**的 search+judge+fetch（60–80% 的步）——把 V1 的省量上限翻了一倍以上，且在大库档（70 ms/步）意义放大。潜在还可省 build/D2H（当前 build 无条件执行以保轨迹历史 gap-free，改动属 L2+，Stage 2 先不动）。
- **风险与红线**：盲回放段无 verdict 监督 → M 从小起步（3–5）；**离线不可评**（改变执行流，C8）→ 直接 live，用 RPG 的 (k,n) 同构网格与评估管线；SR 及格线同 §5。
- **失败模式**：重规划密集段 Δ0 占比高（fh80_ws10 达 20%）→ 锁定条件需容忍 Δ∈{0,1}；库轨迹与 live 轨迹在锁定段内漂移 → 债务预算 + 解锁 probe 兜底。
- **重排判决（2026-07-05）**：**降级至 Stage 4 押后**。理由：(a) N2 是纯 V1（省 hit 段搜索），只在 stock/大库档变现（C9）；(b) F12 显示"长连续缓存执行"正是 SR 被压制之所在——盲回放把缓存执行 run 拉得更长，与 V2 方向相反，SR 风险先验上调；(c) 若 Stage 3 注入门成立，hit 段结构被注入改变，F5 的 lockstep 前提需重测。重启条件见 §5 Stage 4。

### N3. 首步印象门 — 驳回

episode 前 3 步分数对后续 MISS 的 AUC 仅 0.52–0.60（F1）；库覆盖在 episode 级基本均匀（与 in-lib≈held-out 一致）。不立项。

---

## 5. 探索名单：阶段与顺序（2026-07-05 按 live 判决重排；原 Stage 2/3 名单见 git `dc2815e`）

> 评估坐标（C6 + C10 修正）：一律在 (SR, inf_ratio, net 三档) 三元上报告；**"同预算"默认 = 同 inf_ratio 轴**（skip% 匹配已被 C10 废止）。training-free 先行不变（H1–H3 与 N4 全部零训练）。

### Stage 0 — 数据与判决 ✅（2026-07-04）

- G0b 白搜目标信号研究、结构分析、oracle/成本分档判决、B3/B4/N3 驳回、C1–C9 公理修订。

### Stage 1 — N1 滞回门 ✅（2026-07-05 收官，判决见 F9–F11）

| # | 项目 | 结果 |
|---|---|---|
| 1a | 离线前沿扫描 | ✅ 选出 A/B 操作点；live 复核 skip 预测偏差 ≤1.8pp（方法论成立） |
| 1b | live 验证 + matched periodic 对照（8 run × 500 ep） | ✅ N1 vs baseline：SR 保真/提升 + net 全正；vs periodic 同 skip% SR 及格线 **4/4 FAIL**（F10/F11，及格线本身被 C10 判为病态轴）。产物 `exp/gate_research/analysis/n1_live_results.md` |
| 1c | G0a hook + `ScoreHysteresisGate` 服务器化 | ✅（G2 APPROVED，`dc2815e`）；操作点 YAML 化留待 Stage 3 定型后一并做 |

### Stage 2 — V2 增益机制研究（纯离线，已有数据，0 GPU）

| # | 项目 | 方法 | 产出 / 及格线 |
|---|---|---|---|
| 2a | **SR 增益分解 + H1/H2/H3 裁决** | 用 8 run rows + Stage-0 gate_rows：连续缓存执行 run-length 分布 × ep 成败 × 3 档 periodic 剂量（H1 剂量-响应）；periodic searched-step verdict mix / FH 率 vs baseline（H2 反馈量化）；WS 执行量 × 失败相关（H3）；Δinf 三分解（skip 转换 / verdict-mix 迁移 / ep 长度构成）；全对照配对 McNemar 精确 p + per-task 切片 sanity（排单任务异常） | 机制判决（并入本文件）；及格 = ≥1 假设给出**可翻译成 gate 规则**的证据（如 H1 剂量曲线 → 注入间隔 L 取值）。fallback：三假设全不成立 → N4 仍按 uniform L 直接 live（periodic 已证 uniform 有效），2a 只影响注入靶向 |
| 2b | **公平 Pareto（RPG 坐标）** | 8 live 点 + baseline + d1 前沿 7 config + **RPG 锚点**（`exp/random_periodic_gate/analysis/aggregate.csv`，78 点 periodic(k,n)/random(p) × 3 keybuilder × 500ep，raw 在同目录 `data/batch1..6/`；⚠ **libero_spatial only、AlwaysHitJudge + 老 keybuilder 配置（clip_w7_d4/spatial16_w8_d4/max_pool_w3_d5），非 d1+ThresholdJudge，overlay 必标 "(different search/judge)" caveat**）+ warm_start Floor/Ceiling 锚点（`exp/warm_start/data/`）放同一 (SR, inf_ratio) 图；正式核对 F11 初判"periodic 点抬高既有前沿 ~3-5pp"（统一 50-init 协议、净化 l10 纯推理 anchor 口径；l10 无 RPG 锚点）；~~A2 预算重分配并入此处~~ **A2 → 降 Stage 4 押后**（stage2 plan G1 R1 裁决：C8/C11 下预算重分配跳命中步→反事实→离线 SR 不可靠，只得覆盖曲线非 SR 判决） | frontier overlay 图 + "gate 是否构成第四设计轴（keybuilder/judge/search 之外）"判决 |
| 2c | （可选）G0b 危险步补充 | 维持原计划：join `trajectory_deviation` deviate_score oracle 标签 | 若危险步可由 prev_score/其他信号预测 → N4 注入可做危险步靶向的证据 |

**Stage 2 判决（2026-07-05 回填；实现 plan `gate_stage2_v2_mechanism.log.md`，代码 `exp/gate_research/stage2_{common,a_sr_decomp,b_pareto_overlay,c_danger_join}.py`，产物 `exp/gate_research/analysis/stage2_{v2_mechanism,fair_pareto,danger_step}.md` + `stage2_pareto_{spatial,l10}.{png,pdf}`）**：

- **2a ✅（H1 达及格线）**：**H1 剂量/截断成立且可翻译成 N4 规则**——periodic 精确把连续缓存执行 run 截断至 cap=k（spatial 7/4、l10 4/2；baseline mean run-len 12.1/10.3 → periodic 5.3/3.5/3.2/1.8），SR 增益 +7.8/+6.4/+4.6/+4.8pp 且**低剂量已饱和**（spatial 7:1 +7.8 ≥ 4:1 +6.4）→ **N4 注入间隔 L 取宽松低剂量（先验 L≈6–8）**。H3（WS 中毒）有方向性支持（失败 ep WS 执行更多：spatial baseline 1.0/3.5，periodic 降至 0.8/2.6）。H2（on-manifold）弱/suite-specific（searched-FH 抬升主要是 N1 选择效应 69.9→80.8/87.2，periodic≈baseline；仅 spatial_A periodic verdict_mix −0.084 主导 Δinf、inf 反降为佐证）。Δinf 三分解证实 periodic Δinf 由 skip 转换项主导（spatial_A 例外 verdict_mix 主导）。全对照配对 McNemar 精确二项 p：N1 +3.0(p0.067)/+5.2(0.0013)/−0.6(0.82)/−1.4(0.48)，periodic +7.8(0.0001)/+6.4(0.0018)/+4.6(0.035)/+4.8(0.021)。
- **2b ✅**：同协议承重层上 **periodic 抬高既有 d1 前沿**——spatial_A periodic (0.280/90.4) gain **+6.8pp**；l10 periodic (0.70/82.2、0.76/82.4) 对 fh5_ws40↔纯推理(0.83@1.0) 连线 gain **+3.6/+3.0pp**；N1 点 +2.4/+4.4（spatial）、−0.7/−1.7（l10，~在前沿）。spatial_B periodic (0.369) 越 d1 inf 上界暂 OOR（未注入 spatial 纯推理锚，口径待 Stage 3 需要时补）。**判决：gate 构成第四设计轴（keybuilder/judge/search 之外）**，F11 初判 ~3-5pp 成立（spatial 更高 ~7pp）。RPG/异协议锚仅参照未承重。
- **2c（可选）✅ 完成、判决否定**：deviate_score≥5 危险步（libero_spatial，join 3254 步/6.1%）**不可由廉价信号预测**——neg_prev_score/cp1 全程 AUC 0.52/0.54，早期相位 cp1 0.61/step 0.64（弱）。**危险步 ≠ MISS 步**（prev_score 对 MISS AUC 0.98、对危险仅 0.52）→ **N4 注入不做危险步靶向**，用 2a H1 的 run-length/均匀触发（跨配置 proxy R2 限制，结论定性 suggestive；libero_10 无 deviate_score 离线不可做）。

**对 Stage 3 的净指令**：N4 的 V2 分支用 **uniform / 连续缓存执行 run-length ≥ L 触发**（L≈6–8 低剂量），**不做危险步靶向**（2c 否）；V1 分支沿用 N1 跳预测 MISS。

### Stage 3 — N4 混合门（N1 跳 MISS + 定期注入新推理）live

| # | 项目 | 方法 | 产出 / 及格线 |
|---|---|---|---|
| 3a | **N4 live 原型** | 规则：search，除非 (i) N1 滞回判预测 MISS → skip（免推理，V1 分支）或 (ii) 连续缓存执行 ≥ L 步 → skip（强制注入新推理，V2 分支）。(θ,j,M) 沿用 1a A 点；L 由 2a H1 剂量曲线定 2–3 档（先验 {6,8,12}——F12 剂量饱和提示低剂量足够）；exp 层 ClientControlledGate 客户端状态机（复用 1b 全套 harness/analyzer，零 src）；500 ep × 2 suite；若 2a 判 H2/H3 主因 → 注入触发换相应靶（同框架改 client 状态机） | 及格线（C10 轴）：**同 inf_ratio 下 SR ≥ matched periodic**（对照取 inf 最接近的 periodic 点，必要时补 1–2 个 periodic 档）且 net@34 ≥ 0 且 SR ≥ baseline − 1pp；按 C9 三档报告 |
| 3b | 定型服务器化 | N4 胜出 → 扩展 `ScoreHysteresisGate`（+缓存执行 run 计数器与注入分支，L2 小改，1c 管道现成）+ 操作点 YAML | src 门 + 部署配方（延迟档 N1-A / SR 档 N4 / 上限对照 periodic） |

**Stage 3a 判决（2026-07-05 回填；live 6 run×500ep + 补档 l10 cache12 periodic；报告 `exp/gate_research/analysis/stage3_n4_live.md`，代码 commit `251eddc`，raw 暂留 timan107 见报告 Artifact layout）**：
- **N4 胜出 ✅，赢点 L=6**（两 suite 唯一都 pass 的档）。及格线（同 inf_ratio SR ≥ matched periodic ∧ net@34≥0 ∧ SR≥baseline−1pp）：**4/6 pass**（spatial_L6、l10_L6/L8/L12），2/6 fail（spatial_L8/L12）。6/6 都 sr_ok + net34_ok（N4 从不倒退、延迟净正）。
- **剂量效应**：SR 随 L↑（注入变稀）单调降——spatial 92.4/88.8/85.8、l10 81.6/81.0/78.4（L=6/8/12）。**频繁注入（低 L）更好**，坐实 2a H1 方向且显示"更频繁更佳"至少到 L=6。
- **spatial**：仅 L=6 胜 matched periodic（spatial_A cache7/inf1，SR90.4@inf0.280），+2.0pp；L=8/12 退化输。
- **l10**：全 L 胜 matched periodic（补档 l10_c12 cache12/inf1，SR78.4@inf0.663），**双赢**——N4 net@34 全正而任何 l10 periodic net 全负（靠推理暴增换 SR，N4 不用）。
- **对 3b 的净指令**：**服务器化用 L=6**（延迟档 N1-A 纯 V1 / SR 档 N4 L=6，l10 尤推双赢）。caveat：spatial 剂量敏感（勿高 L）；单 run 无 CI（spatial L6 +2.0 在噪声量级，l10 +3.2/+2.6 较稳）；N4 只比了 matched-inf 的弱 l10 periodic，未证胜高-inf l10_A/B。

**Stage 3b 判决（2026-07-06 回填；plan `gate_stage3b_n4_serverize.log.md`，G1/G2 APPROVED）**：**N4 已服务器化 ✅**。扩展 `ScoreHysteresisGate` 加 V2 分支（`__call__` 连续缓存执行 run-length ≥ L 强制 skip 注入，PURE；`record_verdict` 按 searched 分派、用 `HitType` enum 数 run、V1/V2 靠 `_searching` 状态重构区分——**无需客户端 `_last_v2`**）；`config.py` GateConfig 只加 `L`（include_ws 构造器默认不进 config，避免 stray-field 回归）。**零 orchestrator/wire 改动**（record_verdict 已收 hit_type）。**L=None 行为等价 N1**（延迟档 N1-A），**L=6 = N4 SR 档**——同一 gate 类两档。正确性双证：`ScoreHysteresisGate(L=6)` ≡ 3a `N4GateState(L=6)` 等价 golden（含两赢点参数）+ L=None=N1 兼容（1c golden 不回归）。操作点 YAML `exp/gate_research/config/{spatial,l10}/n4_server/`（score_hysteresis L=6）。部署配方：延迟档 L=None(N1-A) / SR 档 L=6(N4) / periodic 上限。

**Stage 4a 判决（2026-07-06 回填；plan `gate_stage4a_n2_follow_winner.log.md`，G1/G2 APPROVED）**：**N2 已服务器化 ✅（Phase C live 未跑）**。进入条件②"重启前重测 F5 lockstep"以 **Phase A 离线复测**兑现（报告 `exp/gate_research/analysis/stage4a_f5_recheck.md`，脚本 `stage4a_f5_recheck.py`）——在 Stage 3a 6 个 N4 live run 上重算 within-FH-run persistence + Δwinner_step：**6/6 GO**（spatial 差基线 ≤2.1pp、l10 反高于基线 +3.7~3.9pp、Δ+1 全 >92% 主导、Δ0 从 14.4% 降 ~4%）——**注入不摧毁反浓缩 lockstep**，结构前提稳固。Phase B（L3）新增 `FollowWinnerGate`：命中段连续真搜 FULL_HIT + 同轨 Δ+1（容忍 Δ0）达 `lock_streak` transition → 锁定库 episode → `budget` 步**盲回放**（不搜不判不推、直接回放 winner 后续缓存动作，省命中段全部成本，与 N1/N4 skip 仍全推理相反）。契约扩展走**最小侵入 additive hook**：gate 加 docstring-only `replay_target()`（**不进 `@runtime_checkable` Protocol 体**，否则 legacy gate `isinstance` 全坏），orchestrator `check()` 的 `not should_search` 分支 `hasattr` 查询 → `PayloadView.walk_next` 取后续 payload → 返回 **`FULL_HIT × searched=False`**（interceptor 现成短路回放，跳 Stage2/3；Stage1+build 保留，build/D2H 押后）。**locked-tail fail-safe**：`walk_next` 空/异常落原 skip 分支单喂 `(MISS, searched=False, winner_id=None)` → gate `_unlock()`，无锁死。`config.py` 加 `follow_winner`（`lock_streak`/`budget`，**须 in_memory backend** fail-loud）。操作点 YAML `exp/gate_research/config/{spatial,l10}/n2_server/`。**注意**：N2 SR 效应离线不可评（C8/C11，改变执行流），必 Phase C live 验；(lock_streak,budget) 网格 + 及格线（同 inf_ratio SR ≥ matched periodic ∧ net@34≥0 ∧ SR≥baseline−1pp）见 plan §4。

### Stage 4 — 押后（进入条件明确）

| # | 项目 | 进入条件 |
|---|---|---|
| 4a | ✅ **服务器化完成（2026-07-06，G1/G2 APPROVED；Phase C live 未跑）** — N2 追随赢家门（自原 Stage 2 降级，理由见 §4 N2 条） | ~~stock/大库延迟为硬约束 **且** N4 落地后 hit 段搜索仍为主要成本；重启前重测 F5 lockstep~~ → Phase A F5 复测 **GO**；build/D2H 省取仍押后。剩 Phase C live（延迟档条件仍是 owner 部署判断） |
| 4b | C1 标定组合门（conformal 预算旋钮，V3 完全体） | 同原条件；特征集新增候选：连续缓存执行 run 长度 / 注入相位（2a 产出） |
| 4c | A3 库覆盖门 / D1 学习难度门 | 同原条件不变（A3 随 50k 库上调；D1 大概率不立项） |

---

## 6. 开放问题（2026-07-05 修订）

1. ~~N1 的 live-离线一致性~~ **已答（F9）**：skip%/Δinf 一致（≤1.8pp）；但 SR 效应离线**不可见**——dInf 模拟只对"动作不变"的跳步可信（C11），SR 必须 live。离线前沿降级为"skip 结构粗筛"，反事实修正模型不再立项（被 V2 机制研究取代）。
2. **V2 机制归因（Stage 2a 主问题）**：H1 剂量截断 / H2 on-manifold 反馈 / H3 WS 执行中毒，谁是主因？periodic 的 SR 是否已顶到纯推理 ceiling（需同协议 50-init anchor，2b）？
3. **注入的最优调度**：uniform（periodic 已证有效且低剂量饱和）vs state-aware（run-length / WS-band / 危险步触发）——N4 的 L 触发是最小 state-aware 版本；更聪明的靶向是否值得，由 2a 裁决。
4. **l10_B 的 N1 边界失败是否真实**：−1.4pp p≈0.48 不显著；若 Stage 3 复测同向，则 l10（振荡型）激进档正式标记不安全。
5. **WS 带**：probe 侧已否（1a 变体 B 0 增益）；**执行侧作为 H3 重开**。
6. **写路径交互 / 跨 suite 泛化**：维持原状（冻结库 write never；θ/M/L per-suite 标定，跨 suite 共享参数代价未测）。

---

## 附录 A：判决数字速查（7 config min–max）

- 转移：P(MISS|MISS) 0.89–0.93；P(MISS|hit) 0.03–0.05；P(FH→MISS) 0.00–0.02；WS→MISS 0.09–0.35。
- 段长：MISS 均 6.4–11.7（p90 20–38）；命中均 12.7–22.4；恢复率 61–84%，恢复后命中段 6.0–9.8。
- 形态：spatial all-hit 47–55%；libero_10 oscillating 24–36%；变点相位中位 0.29–0.50。
- hazard（B3）：L1-2 0.042–0.174 → L3-5 0.009–0.047 → L6+ 0.016–0.074（无单调上升）。
- winner：同 episode 93–98%；Δ+1 75–97%；Δ0 2–20%。
- AUC→MISS：prev_score 0.973–0.986；last3 0.959–0.979；prev_is_MISS 0.917–0.946；step 0.780–0.895；task 0.626–0.669；first3 0.520–0.603；motion 0.397–0.541；hit-streak 0.049–0.101（反向）。robot_state 探针 0.76–0.86。
- sticky 代表档：K3/probe5 → skip 13–28%、dInf +0.006–0.011、net −2.7~+17.5（按档）；oracle skip = 19.4–37.9% @ dInf 0。
- 纠缠：成功 ep MISS 5–24% vs 失败 ep 34–59%。
- 精确 inf_ratio：spatial 0.270/0.287/0.351；libero_10 0.314/0.369/0.417/0.636。
- **Stage 1b live（2026-07-05）**：N1 skip 12.8/20.1/21.2/32.4（spatial A/B、l10 A/B）；ΔSR vs baseline +3.0/+5.2/−0.6/−1.4pp（McNemar p 0.068/0.0015/0.82/0.48）；periodic ΔSR vs baseline +7.8/+6.4/+4.6/+4.8pp（配对 p 待 2a 精算）；N1−periodic ΔSR −4.8/−1.2/−5.2/−6.2pp；periodic Δinf −0.007/+0.081/+0.064/+0.123；net@34：N1 +5.1~+5.6 全正，periodic +5.8/−18.1/−12.8/−25.7。

## 附录 B：与前身 brainstorm 的差异清单

1. B3 从 P1 首推降为 N2 的内部安全绳（前提被 F4 驳回）；B4 驳回（无信号）。
2. 新增 N1（分数滞回，主打）与 N2（追随赢家，V1 上限翻倍）——均由 F1/F5 数据涌现，brainstorm 未含。
3. V1 从"有效收益方向"细化为**分档判决**（C9）：优化小库负、stock 中性偏正、大库显著正。
4. G0b 的白搜目标在本文完成；oracle 上限从 deviate_score 换算口径改为直接的 MISS% 免费上界（miss-skip 族）。
5. A1/A2 维持"不单独立项"，但有了定量依据（AUC 榜）；A3/C1/D1 押后并给出明确进入条件。
6. 评估框架新增 C8（反事实口径）——离线可信域的边界首次明确。

## 附录 C：Stage 1b live 判决对本文件的修订清单（2026-07-05）

1. 新增 F9–F12（§2 末）：N1 自证成立 / 原及格线 4/4 FAIL / 延迟轴反转 / V2 机制与三假设。
2. 公理增补 C10（skip% 不是预算轴，"同预算"须声明三元轴）、C11（skip=干预，禁止假设 SR 中性）——1b 及格线以 skip% 操作化"同预算"被认定为本轮最大方法论教训。
3. N1 定位修正（§4）：延迟档位工具，A 点可部署；并入 N4 作"免推理跳 MISS"分支。
4. N2 从 Stage 2 主打降级至 Stage 4 押后（§4 理由三条：纯 V1 / 与 V2 方向相反 / lockstep 前提将被注入改变）。
5. §5 重排：Stage 2 = V2 机制离线研究（H1–H3 + 公平 Pareto，0 GPU），Stage 3 = N4 混合门 live（同 inf_ratio 及格线），Stage 4 = 押后名单；原 Stage 2/3 见 git `dc2815e`。
6. §6 开放问题换代：live-离线一致性已答（skip 是、SR 否）；新主问题 = V2 归因与注入调度。
