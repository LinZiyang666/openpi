# Trajectory Deviation 纠偏实验方案

> Status: Historical (superseded by `trajectory_deviation_step3_redesign.log.md`)
> Date: 2026-04-13
> Task: 验证 cache 轨迹偏差的局部性假设，设计 inference/warm start 纠偏实验

---

## 1. 核心假设

Cache 轨迹失败不是全局失败，而是少数关键 step 的局部偏差引发的级联效应。

具体地：

1. 同一 init scene 下，纯 inference 产出成功轨迹（ground truth），纯 cache 产出失败轨迹
2. 两条轨迹大部分 step 的 action L2 距离很近（重合段），但存在少数 **deviate points**（L2 距离突增）
3. 在 deviate points 用 inference 或 warm start 替代 cache，其余 step 仍用 cache，即可纠偏整条轨迹

如果假设成立，意味着只需极少量 inference 开销（针对 deviate points）就能大幅提高 cache 成功率。

---

## 2. 实验分期

共 3 个 Phase，按依赖顺序执行。Phase A 的结论决定是否继续 Phase B/C。

```
Phase A: 离线诊断（验证前提假设）
    └── Phase C: 在线信号分析（CP1 score 与 L2 距离相关性）
         └── Phase B: Oracle 纠偏（验证上界）
```

---

## Phase A: 离线诊断 — 验证偏差局部性

### 目标

确认 deviate pattern 是"少数 step 集中偏差"而非"均匀扩散"。

### 前提

需要同一 init scene 下的两类数据：
- **纯 inference 轨迹**：Pi0.5 正常推理，收集 per-step action
- **纯 cache 轨迹**：cache always_hit 模式，收集 per-step action

两条轨迹的 action shape 均为 `[action_horizon, action_dim]`（如 `[50, 32]`）。

### 数据收集

| 模式 | 服务器配置 | 说明 |
|------|-----------|------|
| 纯 inference | `--cache_config` 不设置或 cache 关闭 | Pi0.5 每步全量推理 |
| 纯 cache | `gate: always_search`, `judge: always_hit`, CP1 only | 每步都用 cache 返回的 action |

两组实验必须使用 **相同的 seed、相同的 task suite、相同的 episodes_per_task**，以保证 init scene 对齐。

### 分析方法

对每对（inference 轨迹, cache 轨迹）逐 step 计算：

```python
# action shape: [action_horizon, action_dim], e.g. [50, 32]
l2_distance = torch.norm(a_inference - a_cache).item()
```

统计指标：
1. **L2 距离分布**：直方图 + 百分位数（p50, p90, p95, p99）
2. **Deviate 占比**：L2 > threshold 的 step 数 / 总 step 数（多个 threshold 扫描）
3. **位置分布**：deviate points 在轨迹中的位置（前期/中期/后期），用 normalized step index
4. **连续性**：deviate points 是孤立的（偶尔一个）还是连续的（连续好几个）
5. **成功 vs 失败**：cache 成功轨迹和 cache 失败轨迹的 deviate pattern 对比

### 关键陷阱：观测漂移

纯 cache 的轨迹一旦在某个 step 偏离 ground truth，后续 step 的观测（image）就变了，导致后续所有 step 都无法直接与 ground truth 对齐比较。需要区分：

- **一阶偏差（action divergence）**：给定相同观测，cache 和 inference 的 action 不同
- **二阶偏差（observation drift）**：因前面 step 已偏离，观测本身就不同了，后续 action 自然不同

Phase A 直接比较两条轨迹看到的是**混合效应**。如果要分离一阶偏差，需要 Phase B 的 step-wise oracle 方法（每步实际执行被选中的 action 后，下一步的观测才是真实的）。

**但混合效应本身仍有价值**：如果在 drift 发生前能看到一个 clear deviate onset point，就说明偏差确实是局部触发的。

### 产出

- `exp/analyze_trajectory_deviation.py`：离线分析脚本
- 统计图表：L2 距离曲线、deviate 占比、位置分布
- **Go/No-Go 判定**：如果 deviate points 占比 < 20% 且集中在轨迹的特定区域 → Go；如果偏差均匀分布 → 重新审视假设

---

## Phase C: 在线 Deviate 检测信号探索

### 目标

找到一个在线可用的信号来预测哪些 step 是 deviate points。

### 前提

Phase A 完成，且假设得到支持。

### ⚠️ CP1 Similarity Score 不可直接使用

当前跨模态融合方法为 **Weighted RRF（Reciprocal Rank Fusion）**，产出的 score 是 `Σ w_i / (k + rank_i)` 的加权和。这是一个**基于排名的指标**，而非基于实际相似度的指标：

- score 只反映检索结果在各字段中的相对排名位置，不反映实际向量距离
- 不同 query 之间的 score 不可比较（排名分布不同）
- 同一 query 的 top-1 score 在完全不相关的场景下可能仍然很高（只要它在所有候选中排第一）

**结论：RRF score 不能用作 deviate point 检测的阈值信号。**

### 候选替代信号

需要探索其他在线可获取的检测信号：

#### C1: 原始字段级 Cosine Similarity

绕过 RRF 融合，直接取各字段的原始 cosine similarity（vision_0, vision_1, prompt_emb）。这是真实的向量距离，数值有绝对意义。

获取方式：在 orchestrator 的 search 流程中，除了返回 RRF fused score，额外返回 top-1 候选的各字段 raw cosine similarity。

分析：
- 对 Phase A 的数据，回放每步的 key builder output，与 artifact 中所有候选计算 raw cosine sim
- 取 top-1 候选的各字段 cosine sim，与对应 step 的 L2 distance 做相关性分析
- 可能只需要看 vision 字段的 cosine sim（vision 变化是 deviate 的主要来源）

#### C2: Top-1 vs Top-2 Score Gap

即使 RRF score 的绝对值不可用，top-1 和 top-2 之间的 **score gap** 可能有信息量：
- gap 大 → top-1 候选显著优于其他 → 高置信度匹配
- gap 小 → 多个候选都差不多 → 低置信度，可能是 deviate point

#### C3: Action Chunk 时间一致性

不依赖检索信号，而是看 cache 返回的 action chunk 的时间连续性：
- 连续两步 cache 返回的 action chunk 之间的 L2 距离突然变大 → 可能是 deviate point
- 这是一个纯执行端信号，不需要访问检索内部数据

#### C4: Vision Embedding 变化率

跟踪连续两步的 Stage 1 vision embedding 变化（L2 或 cosine distance）：
- 变化率突增 → 场景发生了预期之外的变化（可能因为前一步 action 偏差导致）
- 这是一个 early warning 信号，可以在 CP1 check 之前就触发

### 分析方法

对每个候选信号，与 Phase A 的 L2 distance ground truth 做：

1. **散点图**：x = 候选信号, y = l2_distance
2. **ROC 曲线**：deviate point 为正类，候选信号做分类器，计算 AUC
3. **对比**：哪个信号的 AUC 最高、哪个最实用（获取成本最低）

### 与 warm_tiers 的衔接

如果找到好的检测信号（AUC > 0.8），需要将该信号集成到 Judge 判定流程中：

- 如果 C1（raw cosine sim）胜出 → 可能需要改造 Judge 使其接收 raw sim 而非 RRF score
- 如果 C2（score gap）胜出 → Judge 需要接收 top-K results 而非仅 top-1
- 如果 C3/C4 胜出 → 需要新的 detection 组件，不在 Judge 内部

### 产出

- 各候选信号与 L2 距离的相关性分析 + AUC 对比
- 最优信号的选择和理由
- 如何将该信号集成到现有 cache 框架的方案建议

---

## Phase B: Oracle 纠偏 — 验证纠偏效果上界

### 目标

在完美 deviate detector 的前提下，量化纠偏能提升多少成功率。

### 前提

Phase A 确认偏差局部性。Phase C 可并行推进但不阻塞 Phase B。

### 实验设计

#### B1: Step-wise Oracle（纯 inference 替代）

每步同时产出 inference action 和 cache action。如果 L2 距离 > threshold，执行 inference action；否则执行 cache action。

| 参数 | 值 |
|------|---|
| oracle 类型 | L2 distance threshold |
| threshold 扫描 | 从 Phase A 的 p90/p95/p99 取 3~5 个值 |
| 执行 | 真实执行被选中的 action，观测是真实的 |

这给出了纯 oracle 纠偏的**成功率上界**。

#### B2: Warm Start Oracle

同 B1 的判定逻辑，但在 deviate point 不用纯 inference，而用 **warm start**（从 cache payload 的 `x_t` 开始跑部分 S3）。

| 参数 | 值 |
|------|---|
| oracle 类型 | L2 distance threshold |
| warm start 的 start_t | 0.3 / 0.5 / 0.7 |
| 对照组 | B1 的纯 inference 替代 |

这验证 warm start 的 action 质量是否足以纠偏（比纯 cache 好，可能不如纯 inference）。

#### B3: Intervention Budget

固定允许纠偏的次数（budget），测试不同 budget 下的成功率曲线。

| budget | 说明 |
|--------|------|
| 0 | 纯 cache（baseline） |
| 1 | 只允许 1 次 inference 纠偏 |
| 3 | 允许 3 次 |
| 5 | 允许 5 次 |
| ∞ | 纯 inference（upper bound） |

这回答核心问题：**纠偏几个 step 就够了？成本效益比如何？**

### 实现方式

B1/B2/B3 需要在运行时同时获取 inference 和 cache 的 action，有两种方案：

**方案 1: Interceptor 层面**（推荐）

修改 interceptor，在 cache hit 时同时跑一次 inference（不执行），比较两者 L2 距离，按策略选择。利用已有的 WARM_START 分支。

优点：完整利用已有 cache 框架，可直接产出与现有实验流水线兼容的结果。
缺点：需要修改 interceptor 代码，每步额外开销（额外一次 inference）。

**方案 2: 外部脚本**

新建 exp 脚本，每步分别调用两次 server（一次 cache 模式、一次 inference 模式），比较后选择执行。

优点：不动核心代码。
缺点：延迟加倍，需要管理两个 server 实例或两次调用。

### 产出

- 不同 oracle 策略下的 success rate 表
- budget vs success rate 曲线
- warm start 与 纯 inference 纠偏的效果对比
- 最终结论：warm start 机制是否足以替代纯 inference 纠偏

---

## 3. 评价指标

所有 phase 共用的指标：

| 指标 | 定义 | 用途 |
|------|------|------|
| Success Rate | 完成 task 的 episode 比例 | 主指标 |
| Deviate Ratio | L2 > threshold 的 step 占比 | Phase A 核心 |
| Intervention Count | 实际触发纠偏的 step 数 | Phase B 效率指标 |
| Speedup | 相比纯 inference 减少的 S2+S3 调用次数 | 成本指标 |

---

## 4. 固定配置

本轮实验沿用现有 LIBERO 实验设定：

| 配置 | 值 |
|------|---|
| Task Suite | libero_spatial (10 tasks) |
| Episodes per task | 5 |
| Max steps per episode | 220 |
| Seed | 42（评估用，区别于数据收集的 seed=7） |
| Model | Pi0.5 LIBERO checkpoint |
| Cache Artifact | `data/cache_artifacts/libero_spatial/` |

---

## 5. 待确认

1. 是否已有纯 inference 和纯 cache 的 per-step action 数据？还是需要重新收集？
2. deviate point 的 L2 threshold 初始值如何选取？建议先用 Phase A 的数据驱动
3. Phase B 的实现倾向：interceptor 层面（复用 WARM_START 分支）还是外部脚本？
4. Phase B 需要修改代码，Level 评估可能需要上调至 L2+

---

## 6. 风险

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| 假设不成立：偏差均匀分布，不存在局部 deviate point | 高 | Phase A 先验证，Go/No-Go 决策 |
| 观测漂移使 L2 对比失去意义 | 中 | Phase B 的 step-wise oracle 基于真实观测执行 |
| RRF score 不可用于阈值判定 | 已确认 | RRF 是排名指标非距离指标；Phase C 转向探索 raw cosine sim 等替代信号 |
| 所有候选检测信号与 L2 距离均无相关性 | 中 | Phase C 多信号对比；若全部失效则 warm_tiers 自动判定不可行，需转为固定策略或 budget-based |
| Phase B oracle 实验需要每步双重推理，时间成本高 | 中 | 先在少量 task/episode 上验证，再扩大 |
| warm start 的 action 质量不足以纠偏 | 低 | B2 vs B1 对比实验量化差距，调整 start_t |
