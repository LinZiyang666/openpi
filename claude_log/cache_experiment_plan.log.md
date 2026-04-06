# Cache 实验方案组合

> Status: Plan
> Date: 2026-04-06

---

## 实验范围与固定配置

本轮实验只做 `CP1`，不考虑 `CP2/CP3`。`Gate` 和 `Judge` 不作为实验变量，统一固定为：

- `gate.type = always_search`
- `judge.type = always_hit`

因此，这轮实验真正比较的是两件事：

1. cache key 的降维方式
2. 跨模态检索结果的融合方式

Judge 在这里仅用于放通 top-1 检索结果，不承担阈值判定职责；阈值校准留到后续单独实验。

---

## 实验目标

在 `CP1` 场景下，对 cache key 降维方法和跨模态融合方法进行组合实验，评估不同方案对 retrieval 质量、cache hit rate 和最终准确性的影响。

---

## 降维方法

| 方法 | 描述 | 适用字段 | Vision 降维后维度 | 压缩比 |
|------|------|---------|------------------|--------|
| **A: Mean Pooling** | 对 token 维度取均值 | `vision_0/1`、`prompt_emb` | 2,048 | 256× |
| **B1: Spatial Pooling (16×)** | 16×16 网格 → 4×4，保留空间结构 | 仅 `vision_0/1` | 16 × 2,048 = 32,768 | 16× |
| **B2: Spatial Pooling (64×)** | 16×16 网格 → 2×2，保留空间结构 | 仅 `vision_0/1` | 4 × 2,048 = 8,192 | 64× |
| **C: Max Pooling** | 对 token 维度逐维取最大值 | `vision_0/1`、`prompt_emb` | 2,048 | 256× |

注：

- 原始 vision 字段维度：256 tokens × 2,048 = 524,288
- LIBERO 实验只使用 `vision_0` 和 `vision_1`，不使用 `vision_2`
- `B1/B2` 仅作用于 `vision_0/1`；`prompt_emb` 在这两类实验中沿用 Mean Pool 到 2,048 维
- `robot_state` 固定为 32 维原始向量，不做降维

### Max Pooling 适用性分析

Max Pooling 取每个维度上最强激活值，适合关键物体突出的场景。不适合依赖全局结构或多物体空间关系的任务。对于 LIBERO 这类多物体操作任务，可能出现误命中或漏命中，因此保留为对照组而非默认优选方案。

---

## 两层检索结构

这里的“两层”不是“第一层检索 + 第二层 Judge”，而是：

1. `Layer 1: Field Similarity`
2. `Layer 2: Cross-Modal Fusion`

`Judge` 固定为 `always_hit`，不参与本轮方案组合。

### Layer 1: Field Similarity

所有实验样例都使用同一套字段级相似度定义：

| 字段 | 相似度 / 距离 |
|------|---------------|
| `vision_0` | Cosine similarity |
| `vision_1` | Cosine similarity |
| `prompt_emb` | Cosine similarity |
| `robot_state` | L2 distance |

说明：

- 所有实验都对 `vision_0/1` 和 `prompt_emb` 计算 cosine similarity
- 所有实验都对 `robot_state` 计算 L2 distance
- 由于 LIBERO 不存在 `vision_2`，本轮实验不包含该字段
- 也就是说，Layer 1 的字段级打分方式是固定的，实验变量不在这一层

### Layer 2: Cross-Modal Fusion

Layer 2 比较两种跨模态融合策略：

| 方法 | 描述 |
|------|------|
| **a: Weighted RRF** | 各字段先独立排序，再用加权 Reciprocal Rank Fusion 融合排名 |
| **b: Weighted Score Sum** | 将各字段分数先映射并归一化到统一尺度，再做加权求和 |

说明：

- `Weighted RRF` 主要融合 rank 信号，对各字段原始分值尺度不敏感
- `Weighted Score Sum` 不能直接使用 raw cosine 和 raw L2；必须先把各字段转换为同一语义、同一数值尺度上的相似度分数
- 这两种方法才是本轮实验中“第二层”的可变项，不是 Judge

### `Weighted Score Sum` 的归一化定义

本实验中的 `Weighted Score Sum` 明确定义为：

`Score(x) = Σ_f w_f * ŝ_f(x)`

其中：

- `w_f` 是字段权重
- `ŝ_f(x)` 是字段 `f` 在候选样本 `x` 上的归一化相似度分数
- 所有 `ŝ_f(x)` 都必须落在 `[0, 1]`，且语义统一为“越大越相似”

具体做法：

1. 先计算字段原始分数
   - `vision_0` / `vision_1` / `prompt_emb`：使用 cosine similarity，记为 `s_cos`
   - `robot_state`：使用 L2 distance，记为 `d_rs`
2. 把 `robot_state` 的距离转成相似度
   - `s_rs = exp(-d_rs / tau)`
   - `tau` 为温度参数，控制 L2 距离衰减速度；当前固定使用 `tau = 0.334717`
   - 该值由 `data/libero_spatial` 离线统计得到，脚本为 `exp/calibrate_robot_state_tau.py`
   - 采样规则：仅使用成功 episode；正样本为同一 episode 内 `Δt ∈ {1, 2}`；负样本为同 task 跨 episode 随机步 + 同一 episode 远距离步
3. 把每个字段分数单独归一化到 `[0, 1]`
   - 对 cosine 字段，先做方向统一：`s_cos_01 = (s_cos + 1) / 2`
   - 然后做 per-field percentile normalization：
     `ŝ_f = clip((s_f - p5_f) / (p95_f - p5_f), 0, 1)`
   - 其中 `p5_f` 和 `p95_f` 是字段 `f` 在离线统计集上的 5% / 95% 分位数
4. 最后做加权求和
   - `Score(x) = w_v0 * ŝ_v0(x) + w_v1 * ŝ_v1(x) + w_prompt * ŝ_prompt(x) + w_rs * ŝ_rs(x)`

补充约束：

- `A-SUM / B1-SUM / B2-SUM / C-SUM` 均指上述“归一化后的 score sum”，不是 raw score sum
- 如果 percentile 统计尚未准备好，`Weighted Score Sum` 方案不得与 `Weighted RRF` 直接做公平对比
- 实现优先级上，`Weighted RRF` 可先落地；`Weighted Score Sum` 依赖额外的离线统计与校准
- 只要更换数据集，或改变 `robot_state` 预处理/采样规则，就必须重新运行 `exp/calibrate_robot_state_tau.py` 计算新的 `tau`

---

## 全部组合（8 种）

| 编号 | 降维 | 适用字段 | Vision 降维后维度 | Layer 1: Field Similarity | Layer 2: Cross-Modal Fusion | 简记 |
|------|------|---------|------------------|---------------------------|-----------------------------|------|
| 1 | A: Mean Pool | 全部 | 2,048 | `v0/v1/prompt=cos`，`robot_state=L2` | a: Weighted RRF | A-RRF |
| 2 | A: Mean Pool | 全部 | 2,048 | `v0/v1/prompt=cos`，`robot_state=L2` | b: Weighted Score Sum | A-SUM |
| 3 | B1: Spatial Pool (16×) | 仅 Vision | 32,768 | `v0/v1/prompt=cos`，`robot_state=L2` | a: Weighted RRF | B1-RRF |
| 4 | B1: Spatial Pool (16×) | 仅 Vision | 32,768 | `v0/v1/prompt=cos`，`robot_state=L2` | b: Weighted Score Sum | B1-SUM |
| 5 | B2: Spatial Pool (64×) | 仅 Vision | 8,192 | `v0/v1/prompt=cos`，`robot_state=L2` | a: Weighted RRF | B2-RRF |
| 6 | B2: Spatial Pool (64×) | 仅 Vision | 8,192 | `v0/v1/prompt=cos`，`robot_state=L2` | b: Weighted Score Sum | B2-SUM |
| 7 | C: Max Pool | 全部 | 2,048 | `v0/v1/prompt=cos`，`robot_state=L2` | a: Weighted RRF | C-RRF |
| 8 | C: Max Pool | 全部 | 2,048 | `v0/v1/prompt=cos`，`robot_state=L2` | b: Weighted Score Sum | C-SUM |

---

## 权重探索实验设计

### 约束

- 每组测试 10 分钟，不可并行
- 实验时间预算：1 天（24h）
- 本轮目标是粗筛方案组合，不做 Judge 阈值调参
- 评估指标仍然记录 `hit rate`、准确性和下游任务指标，但这些指标不通过 Judge threshold 控制

### 输入字段与先验

| 字段 | 先验重要性 | 说明 |
|------|-----------|------|
| `vision_0` | **最高** | 主视角，信息量最大 |
| `robot_state` | 中 | 与动作直接相关 |
| `vision_1` | 中 | 左腕视角，补充局部细节 |
| `prompt_emb` | **固定为 0** | 本轮不参与权重搜索 |

### Phase 1：全量粗搜（64 runs，约 10.7h）

对全部 `8` 种“降维 × 融合”组合，各跑 `8` 种权重配置。不提前淘汰任何组合。

这里的权重作用于 Layer 2：

- 对 `Weighted RRF`，表示各字段 rank 的融合权重
- 对 `Weighted Score Sum`，表示各字段 score 的融合权重

`prompt_emb` 全部固定为 `0`，只调 3 个权重（归一化到和为 `1`），step=`0.25` 的粗网格：

| 编号 | `vision_0` | `vision_1` | `robot_state` |
|------|------------|------------|---------------|
| W1 | 1.0 | 0.0 | 0.0 |
| W2 | 0.75 | 0.25 | 0.0 |
| W3 | 0.75 | 0.0 | 0.25 |
| W4 | 0.5 | 0.25 | 0.25 |
| W5 | 0.5 | 0.5 | 0.0 |
| W6 | 0.5 | 0.0 | 0.5 |
| W7 | 0.25 | 0.5 | 0.25 |
| W8 | 0.25 | 0.25 | 0.5 |

`8` 组合 × `8` 权重 = `64` runs。

### Phase 1.5：对 Phase 1 top 3 组合做细粒度权重搜（45 runs，约 7.5h）

从 Phase 1 结果中选出 **top 3 组合**，围绕 Phase 1 各自最佳权重邻域做 step=`0.1` 细搜。

以 Phase 1 最佳权重为中心，`±0.2` 范围内 step=`0.1` 采样，约 `15` 个权重配置/组合：

示例（假设某组合 Phase 1 最佳为 `W3: v0=0.75, v1=0.0, rs=0.25`）：

| 编号 | `vision_0` | `vision_1` | `robot_state` |
|------|------------|------------|---------------|
| F1 | 0.85 | 0.05 | 0.1 |
| F2 | 0.8 | 0.1 | 0.1 |
| F3 | 0.8 | 0.0 | 0.2 |
| F4 | 0.75 | 0.1 | 0.15 |
| F5 | 0.75 | 0.05 | 0.2 |
| F6 | 0.7 | 0.1 | 0.2 |
| F7 | 0.7 | 0.0 | 0.3 |
| F8 | 0.7 | 0.15 | 0.15 |
| F9 | 0.65 | 0.1 | 0.25 |
| F10 | 0.65 | 0.05 | 0.3 |
| F11 | 0.6 | 0.15 | 0.25 |
| F12 | 0.6 | 0.1 | 0.3 |
| F13 | 0.55 | 0.15 | 0.3 |
| F14 | 0.55 | 0.1 | 0.35 |
| F15 | 0.65 | 0.15 | 0.2 |

注：实际采样点根据 Phase 1 各组合的最佳权重动态生成，上表仅为示例。

`15` 权重 × `3` 组合 = `45` runs。

### Phase 2（可选）：验证 `prompt_emb` 假设（3 runs，约 30 min）

对 Phase 1.5 最佳组合+权重，加一组 `prompt_emb=0.1` 对照，确认 `prompt_emb` 无用。

### 时间总结

| 阶段 | Runs | 时间 |
|------|------|------|
| Phase 1：全量粗搜 | 64 | ~10.7h |
| Phase 1.5：top 3 细搜 | 45 | ~7.5h |
| Phase 2：验证假设 | 3 | ~30 min |
| **总计** | **112** | **~18.7h** |

剩余 ~5h 可做额外实验或重复验证。

---

## 本轮需要实现的模块边界

为支撑上述实验，后续实现应围绕以下边界展开：

1. `CP1` 专用的 key builder / 降维模块
2. 字段级相似度计算模块
   - `vision_0/1`、`prompt_emb`: cosine
   - `robot_state`: L2
3. 跨模态融合模块
   - `weighted_rrf`
   - `weighted_score_sum`

本轮不做：

- `CP2/CP3`
- Gate 策略实验
- Judge 阈值实验
- 依赖现有通用 Qdrant RRF 逻辑直接替代全部实验逻辑
