# Gate 探索路线图 — 基于 always-search 真 verdict 数据的方案判决与阶段名单

- **Status**: Data-Grounded Roadmap（供后续 gate 探索按阶段执行）
- **Date**: 2026-07-04
- **前身**: `cache_gate_design_brainstorm.log.md`（2026-07-02/03 头脑风暴，git 历史 `437bbc2` 可查）。本文件是其数据判决版：brainstorm 的方案谱系（G0/A/B/C/D）逐项对撞实测数据后重排，原文仍有效的资产（延迟账本、oracle 口径、RPG 基线、约束公理）已整合进来，不再回读原文。
- **数据**: `exp/gate_research` 采集（2026-07-04）——weighted_sum d1 Pareto 前沿 7 个标志性 config（libero_spatial 3 + libero_10 4）× 500 ep（全 50 inits 0..49），`gate: always_search` + 真 ThresholdJudge verdict，**185,899 决策步**（"步" = 一次 CP1 决策 = 一个 action chunk ≈ 10 env steps），每步含 `robot_state[32] / hit_type / cp1_score / start_t / winner_id / success`。无选择偏置（C5 满足：always_search 采集）。
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

**探索名单（阶段制，training-free 先行）**：Stage 1 = N1 滞回门（离线前沿 → live 验证）+ G0a hook 捎带；Stage 2 = N2 追随赢家门（live）+ A2 预算重分配（离线）+ sticky 作对照基线；Stage 3（押后）= C1 标定组合 / A3 库覆盖（随库规模）/ D1（大概率不立项）。详见 §5。

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

## 2. 八项数据发现（7 config × 2 suite，报告范围为跨 config min–max）

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

### N2. 追随赢家门（FollowWinnerGate / lockstep 盲回放）— Stage 2 主打，V1 上限最大

- **规则**：连续 j 步 FULL_HIT 且同 winner、Δwinner_step=+1（lockstep 证据）→ 锁定该库 episode，接下来 M 步**不搜索**直接回放 winner 的后续动作；预算 M 用完或恢复搜索的 probe 不再命中/掉带 → 解锁回 always_search。M 为盲回放债务预算（**B3 机制的正确用武之地**：此段 verdict 不在场）。
- **依据**：F5（persistence 93–98%、Δ+1 75–97%）+ F2（FH→MISS 直跳 0–2%：锁定段内突然失效概率极低）+ RPG 的 periodic>random（成块结构有效）——N2 即 periodic「cache k 步 / infer n 步」的事件驱动智能版。
- **价值象限**：miss-skip 型省量上限 = MISS%（19–38%）；N2 省的是**命中段**的 search+judge+fetch（60–80% 的步）——把 V1 的省量上限翻了一倍以上，且在大库档（70 ms/步）意义放大。潜在还可省 build/D2H（当前 build 无条件执行以保轨迹历史 gap-free，改动属 L2+，Stage 2 先不动）。
- **风险与红线**：盲回放段无 verdict 监督 → M 从小起步（3–5）；**离线不可评**（改变执行流，C8）→ 直接 live，用 RPG 的 (k,n) 同构网格与评估管线；SR 及格线同 §5。
- **失败模式**：重规划密集段 Δ0 占比高（fh80_ws10 达 20%）→ 锁定条件需容忍 Δ∈{0,1}；库轨迹与 live 轨迹在锁定段内漂移 → 债务预算 + 解锁 probe 兜底。

### N3. 首步印象门 — 驳回

episode 前 3 步分数对后续 MISS 的 AUC 仅 0.52–0.60（F1）；库覆盖在 episode 级基本均匀（与 in-lib≈held-out 一致）。不立项。

---

## 5. 探索名单：阶段与顺序（training-free 先行）

> 每阶段内的项目**可并行**；进入下一阶段不要求上一阶段全部收尾，但 Stage 2 的 live 资源分配以 Stage 1b 的结论为准。所有 live 评估沿用 RPG 坐标系（SR vs inference_ratio + FULL_HIT 率三元组，C6），**及格线 = 同预算打败 periodic**（不是 random）；V1 结论按 C9 三档净值表报告。

### Stage 0 — 数据与判决（✅ 本文完成）

- G0b 白搜目标信号研究、结构分析、oracle/成本分档判决、B3/B4/N3 驳回、C1–C9 公理修订。

### Stage 1 — N1 滞回门（training-free 核心）

| # | 项目 | 方法 | 产出 / 及格线 |
|---|---|---|---|
| 1a | **N1 离线前沿扫描** | 复用 `gate_structure_analysis.py` 模拟管线，扫 (θ_low, θ_high, j, M) × 7 config；对照 sticky-K 与 periodic | (skip%, dInf) 前沿 + 三档净值表；及格 = 支配 sticky 前沿、逼近 oracle（skip→MISS%、dInf→0） |
| 1b | **N1 live 验证** | 离线前沿挑 2–3 操作点，exp 层 ClientControlledGate（零 src），500 ep × 2 suite | SR / inf_ratio / 实测省搜索延迟；及格 = 同 skip% 下 SR ≥ always_search − 1pp，且同预算 ≥ periodic |
| 1c | G0a hook 补丁（捎带项） | 随下一个 src 改动窗口：verdict 回传 + task_key 广播（L2 小改） | N1/N2 服务器化解锁；不阻塞 1a/1b |

### Stage 2 — N2 追随赢家门 + 廉价加法

| # | 项目 | 方法 | 产出 / 及格线 |
|---|---|---|---|
| 2a | **N2 live 原型** | ClientControlledGate 实现锁定/盲回放/解锁；M∈{3,5,8}、锁定条件 j∈{2,3}、Δ∈{0,1}；对照 periodic (k,n) 同构网格 | SR–预算前沿；及格 = 同"省搜比例"下 SR ≥ periodic；重点报告大库档净值 |
| 2b | A2 episode 级预算重分配（离线） | 用本数据按 task 重分配 N1/sticky 档位，合成聚合前沿 | 零成本判断"预算搬运"是否值得进 live |
| 2c | （可选）G0b 危险步目标补充 | join `trajectory_deviation` step2 的 deviate_score oracle 标签 | 若 prev_score 对危险步也 AUC 高 → N1 兼具 V2 属性的证据 |

### Stage 3 — 标定/学习类（押后，进入条件明确）

| # | 项目 | 进入条件 |
|---|---|---|
| 3a | C1 标定组合门（conformal 预算旋钮，V3 完全体） | Stage 1/2 幸存信号 ≥2 个且组合有离线增量；需要"可精确设定 inference_ratio"的部署需求出现 |
| 3b | A3 库覆盖门 | 库规模计划上 50k（V1 档位上调）或开局段成为瓶颈 |
| 3c | D1 学习难度门 | 仅当 C1 距 oracle 仍有大缺口 **且** 出现开局段/跨 suite 明确需求（当前证据下大概率不立项） |

---

## 6. 开放问题

1. **N1 的 live-离线一致性**：离线 dInf 在错跳命中步上是一阶近似（C8）；1b 的 live 点若系统性偏离离线前沿，需要建反事实修正模型（或接受离线仅作粗筛）。
2. **N2 的 build/D2H 省取**：当前 build 无条件执行（gap-free 轨迹历史）；盲回放段不产生新 key，理论上可连 build 一起省（把 4/34/70 ms 档再抬高），但动 orchestrator 契约（L2+），待 2a 证明机制价值后再议。
3. **WS 带的精细利用**：F2 显示 WS 是预警带；N1 目前只用标量阈值,是否给 WS 单独一档（如 WS 时减半 probe 间隔）留给 1a 扫描。
4. **写路径交互**：本轮全部冻结库（write never）。gate 与 write_policy 联动（如 N1 停搜段的真推理轨迹是否入库）是独立研究线，不混入 Stage 1/2。
5. **跨 suite 泛化**：θ/M/K 均 per-suite 标定（libero_10 振荡结构要求更频繁 probe）；跨 suite 共享参数的代价未测。

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

## 附录 B：与前身 brainstorm 的差异清单

1. B3 从 P1 首推降为 N2 的内部安全绳（前提被 F4 驳回）；B4 驳回（无信号）。
2. 新增 N1（分数滞回，主打）与 N2（追随赢家，V1 上限翻倍）——均由 F1/F5 数据涌现，brainstorm 未含。
3. V1 从"有效收益方向"细化为**分档判决**（C9）：优化小库负、stock 中性偏正、大库显著正。
4. G0b 的白搜目标在本文完成；oracle 上限从 deviate_score 换算口径改为直接的 MISS% 免费上界（miss-skip 族）。
5. A1/A2 维持"不单独立项"，但有了定量依据（AUC 榜）；A3/C1/D1 押后并给出明确进入条件。
6. 评估框架新增 C8（反事实口径）——离线可信域的边界首次明确。
