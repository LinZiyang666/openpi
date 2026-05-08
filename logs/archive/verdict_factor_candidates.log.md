# Verdict / Gate Factor Candidates — ENGRAM Cache

> **用途**：为 ENGRAM 缓存子系统的 Judge（命中判定）与 Gate（是否查 cache）阶段，建立可用的统计 / 运动学因子候选目录。
> **范围**：本文档**仅服务于实验规划**，**不为论文起草服务**。所选因子将驱动实现 + ablation；后续实验结果会反向喂回论文 §4.2 / §4.3 与 §6。
> **关联文档**：
> - 缓存子系统代码：`src/openpi/cache/`
> - CP1 实验文档：`docs/experiments/cp1_cache.md`
> - 轨迹偏差实验：`docs/experiments/trajectory_deviation.md`
> - Warm-start 扫描实验：`docs/experiments/warm_start_sweep.md`
> - LLM-layer extract 实验：`docs/experiments/llm_layer_extract.md`
> - Paper workbench：`docs/papers/paper_workbench.md`
> - 相关工作（HeiSD / NIRVANA）：`docs/papers/inference_cache_related_work.md`
> - 论文 Design 起草日志：cache thesis repo 的 `logs/design_drafting.log.md`

---

## 0. 决策上下文：为什么 threshold-on-fused-score 不可行

两个根本性问题使 raw similarity threshold 不能作为 verdict 信号：

1. **RRF 抹平绝对相似度尺度**。ENGRAM 选用 weighted RRF 作为多模态融合 —— rank-aggregation 让 fusion weights 在跨任务 / 跨场景上保持鲁棒，这是有意的设计取舍。代价是 RRF 输出的 fused score 不保留绝对相似度量级；两条 fused score 相同的条目，其各 modality 的实际 similarity 可能天差地别。直接 threshold fused score 会混淆 "真的像" 与 "矮子里拔将军"。

2. **Similarity threshold 不可跨任务移植**。即使换成 raw weighted-sum fusion，threshold 的合适值仍依赖任务的 similarity 分布、机器人的观测分布、cache library 的规模。任务 A 调好的阈值在任务 B 可能完全错位。

Verdict 信号必须从其他空间获取。**动作空间 + 轨迹统计** 是天然候选：信号源是 cache 内容自身，单位是运动量（可经归一化跨机器人），且多数因子计算便宜。

与 HeiSD 的核心区别：
- HeiSD 用 **任务-耦合的物理学公式** 做 safe / danger zone 分类；我们用 **任务-无关的 cache 统计** 聚合多因子。
- HeiSD 限定 autoregressive VLA；我们覆盖 flow-matching VLA action expert（更广泛适用）。

**Judge 接口约定**：本文档假设 Judge 在拿到搜索结果后能自取候选的 payload（含 `action_chunk` 与 `intermediates`），即 Judge 持有 winner_id 后由其负责 `fetch_payload`。具体接口如何放置不在本文档讨论范围。

---

## 0.5 平滑度的两种度量族（Action vs Trajectory）

> **术语约定（贯穿全文）**：本文档中所有"**action**"一词均指 **action chunk 形式的预测动作序列**（即 `action_chunk: [50, 32]` 这类张量），其来源既可以是 **VLA inference 直接输出**（model forward 的产出），也可以是 **cache retrieve 取上来的 cached `payload.action_chunk`**。**不指** environment-applied action / 控制器最终命令 / 实际执行后的物理量。所有 -A 变体因子的"动作"语义均按此约定理解（即同时覆盖 inference 输出与 cache 命中两种来源）。"Trajectory" 与 "robot_state" 才指涉实际状态轨迹。

所有 smoothness 类因子（F1a / F1b）都有**两种度量变体**，二者在数学上不等价、信息量不重合，应分别建模：

- **Action 变体（后缀 -A）**：从 **action chunk 序列**（inference 输出或 cache 命中均可）提取（二阶差分 / jerk / 步距）。
- **Trajectory 变体（后缀 -T）**：从 **robot_state 序列** 提取（实际状态 / 末端执行器位姿 / 关节位置的二阶差分 / 曲率）。

二者会差异显著的场景：
- **纯平移任务**：action 与 trajectory 都平滑（一致信号）。
- **圆周运动**：action 可能恒定关节速度命令（很平滑），但末端轨迹在笛卡尔空间走圆弧（曲率显著、按线性度量不平滑）。
- **接触瞬间**：action 阶跃命令（不平滑），但实际状态可能因机器人惯性平滑变化（不一致信号 → 二者结合可识别接触 regime）。

**单凭 Action 平滑度可能漏判轨迹层面的不连续；单凭 Trajectory 平滑度可能漏判命令层面的不连续。** 二者互补不冗余。每个 smoothness 因子条目下都标注其变体可选项与数据来源。

---

## 1. 因子目录

每条因子条目包含：定义 / 数据源 / 计算时机（`offline` 写入时预算 vs `online` per-cycle vs `hybrid` 混合）/ 适用决策点 / 风险 / 实现要点。

### F1a — Runtime continuity（运行时连续性，Judge-stage）

候选已被检索且 payload 由 Judge 取上来。比对候选 "未来若干步" 与本 episode "过去若干步" 的衔接平滑度，回答："如果接下来执行这个 cached 候选，运动是否仍会保持平滑？"

#### F1a-A — Action variant（动作连续性）

- **数据源**：candidate `payload.action_chunk[0:K]`（候选未来前 K 步动作）+ 当前 episode 最近 K 步**已执行** action（orchestrator 维护的执行历史）。
- **计算**：**online**（per-cycle，Judge 阶段）。
- **用途**：Judge（连续度低降权 / 拒绝）/ chain-walk 入口（首步必须够平滑才进链）。
- **不适用**：不能用于 Gate —— Gate 在 search 之前运行，无候选可对比。
- **操作化**：
  - L2 距离：`cached.action_chunk[0]` vs `last_executed_actions[-1]`
  - 多步对齐：候选首 K 步 vs 已执行末 K 步
  - per-DOF z-score 距离
  - DTW 柔性对齐
- **风险**：不连续 regime（接触 / 模式切换 / 抓 / 放）下因子会标记 jerky，但该 regime **究竟是否应该禁用 cache（真 danger）还是该照常 cache（仅 regime 切换）目前不明**——预设任一方向都可能错。需配 F1a-T 或额外 regime 检测以观察行为，但策略（放松 vs 严守）须由实验数据决定。跨自由度 / 跨机器人尺度差异需归一化。
- **灵感来源**：HeiSD 的 "纯 inference 输出在时间上平滑" 观察。

#### F1a-T — Trajectory variant（状态连续性）

- **数据源**：candidate `payload.future_state_chunk[0:K]`（候选未来前 K 步**机器人状态**，**需要 payload schema 扩展**）+ 当前 episode 最近 K 步**已观测** robot_state。
- **计算**：**online**（per-cycle，Judge 阶段）。
- **用途**：与 F1a-A 互补 —— 圆周运动 / 复杂控制时 action 平滑但状态不平滑，F1a-T 可捕获这类情形。
- **架构需求**：当前 `CachePayload` 不存 future_state_chunk；启用 F1a-T 需在 `storage_types.CachePayload` 增加 `future_state_chunk: Optional[torch.Tensor]` 字段（候选未来 K 步 robot_state，写入时从源 episode 取）。
- **操作化**：
  - 末端执行器位置 L2：`cached.future_state[0]` vs `last_state[-1]`
  - 关节位置二阶差分匹配
  - 笛卡尔曲率连续性
- **风险**：F1a-A 风险全部继承 + 额外的 schema / 存储成本。
- **替代方案**：若 schema 扩展过重，可用 **F1b-T `(0, +W)`** 作为 offline 替代 —— 后者度量"候选源轨迹未来 W 步状态平滑"，与 F1a-T 度量"候选状态衔接当前轨迹"略有不同（F1b-T 看候选侧自身平滑，F1a-T 看跨边界连续性），但两者部分覆盖相同信号。

### F1b — Source-trajectory window smoothness（离线，源轨迹窗口平滑度）★

- **定义**：每个 cache entry 的属性 —— 在其源轨迹中，该 entry 周围窗口内的运动有多平滑。窗口由 **`(past_W, future_W)` 元组**指定，**两分量独立可配，任一可为 0**。同一 entry 可预存多个 (p, f) 组合。
- **F8 已合并**：旧 F8 "future-trajectory smoothness" 是本因子在 `(0, +W)` 配置下的特例，不再独立列项。
- **数据源**：写入时已知的源 episode 完整 action + robot_state 序列。
- **计算**：**offline**（episode-write 时预算，存为 entry metadata）。

#### F1b-A — Action variant（源动作窗口平滑度）

- **数据源**：源 episode 的 `action_chunk` 序列。
- **操作化**：窗口 `[entry_idx − past_W, entry_idx + future_W]` 内的：
  - 平均二阶差分 / jerk
  - action velocity 方差
  - 最大相邻步距

#### F1b-T — Trajectory variant（源状态窗口平滑度）

- **数据源**：源 episode 的 `robot_state` 序列。
- **操作化**：窗口 `[entry_idx − past_W, entry_idx + future_W]` 内的：
  - 末端执行器位置二阶差分 / Cartesian jerk
  - 关节位置 jerk（与 F1b-A 在 action ≈ joint position 时部分重合，但 derived from achieved state 不是 commanded action）
  - 笛卡尔轨迹曲率 / 弧长偏差

- **窗口建议组合**（可灵活增减）：
  - `(0, +W)` 纯未来：服务 chain-walk 决策（旧 F8 用例）
  - `(W, 0)` 纯过去：判断"该 entry 是否来自稳定阶段"
  - `(W, W)` 对称：综合 entry 邻域质量
  - `(p, f)` 不对称：例如 `(2, 10)` 偏重未来，用于"当下衔接 OK 但未来值得长走链"
  - 多 W ∈ {2, 5, 10, 20} 各预算，让 verdict 按上下文取用
- **用途**：
  - Judge（偏好邻域平滑的命中）
  - chain-walk 决策（仅沿 future-window 平滑的链走）
  - 未来轨迹抽取（选稳定续段做 multi-step skip）
  - Write-time filter（可选 —— 写入时丢弃邻域过乱的 entry）
- **风险**：
  - 写入侧 schema 复杂度 / 存储成本上升 → 限定固定 (p, f) 集合。
  - 源轨迹窗口平滑 ≠ 当前轨迹拟合好 → 必须配 F1a 运行时检查。

### F2 — Top-k action consensus（候选共识 / 方差）

- **定义**：top-k 检索结果对应的 action 间的方差 / 离散度。
- **数据源**：搜索后 Judge 取 top-k payload 的 `action_chunk` 比较。
- **计算**：**online**。
- **用途**：Judge（低方差 = 高共识 = 安全 hit；高方差 = 模糊 = MISS）。
- **变体**：默认是 F2-A（动作空间）；若 payload 含 future_state_chunk（见 F1a-T），可同时算 F2-T 状态空间共识。
- **操作化**：关节空间 L2 方差 / 经验协方差 trace / top-1 与 top-k 均值距离。
- **风险**：对 k 敏感 → 用 percentile 或相对量；cold-start variance 恒低。



---

## 2. 计算时机汇总

| 因子 | 离线预算 | 在线 per-cycle | 单 entry 存储成本 |
|---|---|---|---|
| **F1a-A** runtime action continuity | — | yes | — |
| **F1a-T** runtime trajectory continuity | — | yes | 中（payload 新增 future_state_chunk） |
| **F1b-A** source action smoothness, multi-window | **yes** | — | 小（每 (p, f) 一个 float）|
| **F1b-T** source trajectory smoothness, multi-window | **yes** | — | 小（每 (p, f) 一个 float）|
| F2 top-k consensus | — | yes（搜索后）| — |

**预算经济性**：F1b-A 与 F1b-T 是离线预算受益最大的两个因子 —— 二者都需源轨迹的邻域信息（过去 / 未来），写入时全知，verdict 时不可再得。

---

## 3. 组合策略

因子必须组合为实际 Gate / Judge 决策。

### S1 — 加权和 + percentile 门槛

```
score = sum_i w_i * factor_i_normalized
score >= P_full_hit_percentile   → FULL_HIT
score >= P_warm_start_percentile → WARM_START
else                             → MISS
```

Percentile 在最近 verdict 分数的滑动窗口上计算。**完全免绝对阈值**。最易实现、最易 ablate。

### S2 — AND gate

k 个因子各自跨过本因子 percentile 才通过。严格；适用 FULL_HIT 标准。

### S3 — OR gate

任一因子跨过即通过。宽松；适用 WARM_START fallback。

### S4 — 学习型组合器

在因子向量上训一个小分类器。最强表达力，但与 "training-free" 定位冲突。

### Action / Trajectory 变体的组合策略

每个 smoothness 因子有 -A 与 -T 两种变体。组合方式：
- **AND**：F1a-A 与 F1a-T 都通过才接受 —— 严格，适用 FULL_HIT
- **OR**：任一通过即接受 —— 宽松，适用 WARM_START
- **Max gap**：取 max(action_smoothness, trajectory_smoothness) 作为最终风险 —— 任一指标 jerky 即视为 jerky
- **由 regime 切换**：识别"接触 / 模式切换"时偏重 F1*-T，常态时偏重 F1*-A

最终选择待实验决定，不预设。

---

## 5. 跨因子风险

| 风险 | 受影响因子 | 缓解 |
|---|---|---|
| 不连续 regime（接触 / 模式切换）下 cache 安全性未明 —— smoothness 因子标记 jerky 时，该 regime 究竟应**禁用 cache**（若是真 danger）还是**照常 cache**（若只是合法 regime 切换）目前不知，两种预设都不安全 | F1a-A / F1b-A / F1a-T / F1b-T | 配对 -A 与 -T 双侧观察以区分 regime 类型；实验阶段对不同不连续场景分别测 cache 命中后的成功率，**不预设**"放松"或"严守"|
| Action vs Trajectory 信号不一致（圆周运动等）| F1a / F1b 单独使用 | 强制 -A 与 -T 并用；不依赖单一变体 |
| Cold-start：consensus 样本不足 | F2 | 头 N 步强制 MISS |
| Per-DOF / per-robot 尺度差异 | F1a / F1b | 归一化动作 / 状态空间（per-DOF z-score / 单位归一化）|
| 离线因子的存储 / 写入成本 | F1b（多 (p, f) 配置）| 限定固定 (p, f) 集合；不全量持久化 |
| F1a-T 需 payload 扩展（future_state_chunk）| F1a-T | 评估收益 vs 存储成本；若收益不显著则用 F1b-T `(0, +W)` 替代 |
| 超参爆炸（每因子有 window / weight / threshold）| 全部 | 用 percentile / margin；先 ablate 因子再调组合 |

---

## 6. 推荐实验顺序

按"先做 → 再做"的实施顺序：

1. **F1a-A runtime action continuity**：简单、抓 HeiSD 核心观察。Judge 取 payload 后即可比对。
2. **F2 top-k consensus**：便宜、与 F1a-A 互补（动作空间方差 vs 衔接平滑度）。
3. **F1b-A & F1b-T source-window smoothness（offline）**：需写入侧 schema 改动但 verdict 时零开销，预期收益大。建议同步上 -A 与 -T 两变体；窗口先用 `(0, +5)` `(0, +10)` `(5, 5)` 三组，再据数据决定是否扩。
4. **F1a-T runtime trajectory continuity**：需 payload schema 扩展（`future_state_chunk`）。先评估 F1b-T 在 chain-walk / future-extraction 上的覆盖度，若仍有缺口再上 F1a-T。
5. **S4 学习型组合器**：仅在简单聚合证明不足时考虑。

---

## 7. 实验阶段开放问题

- **F1b 窗口配置**：哪些 `(past_W, future_W)` 组合跨任务泛化最好？对称窗口（`(W, W)`）vs 不对称（`(0, W)` 或 `(2, 10)` 等）哪个收益高？
- **Action vs Trajectory 信号**：在不同任务（平移 / 圆周 / 接触）下，二者各自 vs 联合的预测力如何？是否存在任务-类别决定哪个更主导？
- **F2 consensus**：top-k 共识是预测任务成功，还是只反映检索分布特征（与动作质量无关）？
- **Chain-walk**：命中后能可靠走多少步链才需重 verdict？步数与任务相关，还是 action / trajectory smoothness 的普适属性？
- **F1a-T payload 扩展决策**：F1b-T `(0, +W)` 能否充分替代 F1a-T 的 chain-walk / future-extraction 用例？若可，则不上 F1a-T 节省 schema 复杂度。
- **不连续 regime 的 cache 安全性**：当 smoothness 因子在接触 / 模式切换期标记 jerky 时，cache 命中究竟会损害任务成功率（→ 视为真 danger，禁用 cache）还是不影响（→ 视为合法 regime，照常 cache 或放松标准）？这是经验问题，必须由实验回答；候选检测机制：jerk 突变、action-vs-trajectory 不一致、任务阶段标注（若有）—— 但**检测出 regime 后的处置策略本身是被实验决定的对象，不预设**。
- **组合**：AND / OR / 加权和的 Pareto 前沿偏好？
- **存储预算**：每 entry 多预算因子是否在 verdict 准确性上付出递减回报？
- **Gate 阶段缺失**：当前因子集无 Gate 候选；实验阶段是否需要补一个早期过滤器？

