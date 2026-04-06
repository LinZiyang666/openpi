# Cache Key 降维方法推荐

## 现状分析

当前 `FullOriginalKeyBuilder` 将所有原始 embedding 直接 flatten 作为 cache key：

| 字段 | 原始形状 | Flatten 后维度 |
|------|----------|---------------|
| vision_0 (base_0_rgb) | [256, 2048] | 524,288 |
| vision_1 (left_wrist) | [256, 2048] | 524,288 |
| vision_2 (right_wrist) | [256, 2048] | 524,288 |
| prompt_emb | [200, 2048] | 409,600 |
| robot_state | [32] | 32 |
| **总计** | | **~1,982,496** |

问题：
- 维度过高导致 Qdrant 需要分 chunk（>65k 维切块），搜索效率差
- 未做 L2 归一化，cosine similarity 结果不可靠
- 高维空间中距离度量退化（curse of dimensionality）

---

## 降维方法流水线

方法分为两层，按顺序组合使用：

```
原始 embedding          第一层：Token 池化              第二层：维度投影
[256, 2048]  ──────►  [2048] 或 [K, 2048]  ──────►  [D]
(~50万维/字段)         (2048~32768维/字段)            (128~512维/字段)
```

> **重要说明：** 第二层方法（PCA、随机投影等）的输入是第一层池化后的 2048 维向量，不是直接对原始 50 万维数据操作。对百万维数据直接做 PCA/投影既不现实（内存和计算开销太大），也没有必要——先池化再降维是标准做法。

---

## 第一层：Token 池化（必选，50万→2048）

### 方法 A：Mean Pooling（均值池化）

将 token 序列沿 token 维度取均值，得到单一向量。

```
[256, 2048] → mean(dim=0) → [2048]
```

| 字段 | 池化前 | 池化后 |
|------|--------|--------|
| vision_x (×3) | 524,288 | 2,048 |
| prompt_emb | 409,600 | 2,048 |
| robot_state | 32 | 32（不变） |
| **总计** | **1,982,496** | **8,224** |

- **压缩比：241×**
- SigLIP/CLIP 系列模型的标准 image-level representation 就是均值池化
- 实现最简单，适合作为 baseline

### 方法 B：Spatial Pooling（空间池化，仅 Vision）

Vision tokens 排列在 16×16 网格上（patch_size=14, image=224），可用 adaptive average pooling 保留空间结构：

```
[256, 2048] → reshape [16, 16, 2048] → adaptive_avg_pool2d → [H, W, 2048] → flatten
```

| 池化粒度 | Token 数 | 维度/字段 | 压缩比 |
|----------|----------|-----------|--------|
| 4×4 | 16 | 32,768 | 16× |
| 2×2 | 4 | 8,192 | 64× |
| 1×1 (=Mean) | 1 | 2,048 | 256× |

- 保留空间布局信息（哪个区域有什么物体）
- 适合场景差异主要体现在空间位置的任务
- Prompt 仍用 Mean Pooling

### 方法 C：Attention-Weighted Pooling

用 CLS token 或可学习 query 对 token 加权求和：

```
weights = softmax(tokens @ query)   # [256, 1]
pooled = (tokens * weights).sum(0)  # [2048]
```

- 比均值池化更有针对性，突出重要 token
- 需要训练或手动定义 query 向量
- 实现复杂度中等

---

## 第二层：维度投影（可选，2048→128~512）

> 以下方法均作用于第一层池化后的 **2048 维**向量。

### 方法 D：Random Projection（随机投影）

用固定种子的随机高斯矩阵投影，Johnson-Lindenstrauss 定理保证距离近似保持。

```
proj_matrix = normalize(randn(2048, 256), dim=0)  # 固定种子
reduced = pooled @ proj_matrix                      # [2048] → [256]
```

- **不需要任何训练数据**，生成矩阵即可
- 理论保证：N 个点投影到 O(log N / ε²) 维后，两两距离变化不超过 ε
- 适合快速验证

### 方法 E：PCA

对池化后的向量集合做主成分分析。

```
pca = PCA(n_components=256).fit(pooled_samples)  # 离线拟合
reduced = pca.transform(pooled)                    # [2048] → [256]
```

- 需要预先收集一批样本（几百到几千条）拟合
- 需保存变换矩阵供推理时使用
- 效果通常优于随机投影

### 方法 F：Learned Projection Head

训练小型 MLP 做降维，用对比损失（contrastive loss）优化。

```
class ProjectionHead(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
    def forward(self, x):
        return F.normalize(self.net(x))
```

- 为特定任务域优化的最佳方案
- 需要训练数据 + 训练流程
- 实现和维护成本最高

---

## 推荐实施路径

| 阶段 | 组合 | 每字段维度 | 总维度 | 压缩比 |
|------|------|-----------|--------|--------|
| **Phase 1** | Mean Pooling + L2 Norm | 2,048 | 8,224 | 241× |
| **Phase 2** | Mean Pooling + Random Projection (256d) | 256 | 1,060 | 1,871× |
| **Phase 3**（可选） | Mean Pooling + Learned Projection (128d) | 128 | 544 | 3,644× |

Phase 1 优先级最高的原因：
1. 压缩比已极大（241×），Qdrant 不再需要 chunk
2. Mean Pooling 是 SigLIP 检索的标准方案，几乎不损失语义
3. 修复了当前缺失 L2 归一化的问题
4. 实现最简单，可在现有 `QueryKeyBuilder` Protocol 内完成

Phase 2 可在 Phase 1 验证后按需追加，进一步减少存储和搜索开销。

Phase 3 仅在 Phase 2 效果不满足要求时考虑。

---

## 实现方式

在现有架构内，新建 `MeanPoolKeyBuilder`（或参数化的 `PooledKeyBuilder`）实现 `QueryKeyBuilder` Protocol，通过 `cache.yaml` 的 `key_builder.type` 切换。无需修改 Orchestrator、SearchStrategy 或 Backend 代码。

---
---

# Similarity 计算方法推荐

## 问题拆解

Similarity 计算自然分为两层：

```
              Layer 1: Field Similarity              Layer 2: Cross-Modal Fusion
query_key ──────────────────────────► per-field score ──────────────────────────► final score
(per field)    "两个同模态向量有多像"     (per field)      "多个模态的分数怎么合并"
```

| 层 | 职责 | 当前实现 |
|----|------|----------|
| **Layer 1: Field Similarity** | 对比同一字段（如 vision_0）的两个向量 | Cosine Similarity（Qdrant 原生 / InMemory F.cosine_similarity） |
| **Layer 2: Cross-Modal Fusion** | 将多个字段的相似度合并为最终排名/分数 | Qdrant Weighted RRF；InMemory 只取第一个字段 |

两层是独立的设计决策，可以自由组合。

---

## Layer 1: Field Similarity（单字段内的向量对比）

### 方法 1A：Cosine Similarity（当前方案）

```
sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

- 范围 [-1, 1]，1 = 完全相同方向
- **对向量幅度不敏感**，只关注方向
- 是 embedding 检索的标准选择（CLIP、SigLIP、sentence-transformers 全用这个）
- 如果 key 已经 L2 归一化，cosine similarity = 内积（dot product），计算更快

**适用场景：** 大多数情况下的默认选择。SigLIP embedding 天然是方向性语义，cosine 是最匹配的度量。

### 方法 1B：L2 Distance（欧氏距离）

```
dist(a, b) = ‖a - b‖₂
sim = -dist  或  sim = 1 / (1 + dist)
```

- 对向量幅度敏感——两个方向相同但长度不同的向量距离不为零
- 在 L2 归一化后的向量上，L2 距离和 cosine similarity 单调等价：`‖a-b‖² = 2(1 - cos(a,b))`
- Qdrant 原生支持（`Distance.EUCLID`）

**适用场景：** robot_state 字段。关节角度/末端位置的绝对数值有意义，不应只看方向。例如 `[0.1, 0.2]` 和 `[0.2, 0.4]` 方向相同但状态不同，L2 能区分。

### 方法 1C：Dot Product（内积）

```
sim(a, b) = a · b
```

- 如果向量已 L2 归一化，等价于 cosine similarity
- 如果未归一化，兼顾方向和幅度（幅度大 = 置信度高的 embedding 贡献更大）
- 是 Qdrant 最快的距离度量（无需算范数）

**适用场景：** 当 embedding 幅度本身携带信息时（如某些 transformer 层的输出幅度与 token 重要性正相关）。但需要验证这个假设是否在 SigLIP prefix_embs 上成立。

### 方法 1D：Learned Metric（学习型度量）

训练一个小网络判断两个向量的相似度：

```python
# Siamese 模式
sim = mlp(concat(a, b))  # 或 mlp(|a - b|)

# Mahalanobis 模式（学习一个加权矩阵 M）
sim = -sqrt((a-b)ᵀ M (a-b))
```

- 可以学到非线性的相似性定义
- Mahalanobis 是 L2 的推广：M=I 退化为 L2，M=对角矩阵 = 加权 L2
- 训练方式和降维 MLP 类似（对比学习 + 正负样本对）

**适用场景：** 当简单度量效果不够时。实际上如果降维阶段用了 Learned Projection，投影已经隐式学了一个度量（投影空间中的 cosine = 原始空间中的 learned metric），所以这一层通常不需要额外学习。

### Layer 1 推荐

| 字段 | 推荐度量 | 理由 |
|------|---------|------|
| vision_0/1/2 | **Cosine** | SigLIP embedding 语义在方向上，幅度无意义 |
| prompt_emb | **Cosine** | 同上，语言 embedding 标准做法 |
| robot_state | **L2 Distance** | 关节角/位置的绝对值有物理意义 |

> **关键洞察：** 不同字段可以用不同的度量，这是合理的。Qdrant 支持每个 named vector 独立配置 distance metric。当前实现所有字段统一用 cosine，对 robot_state 不太合适。

---

## Layer 2: Cross-Modal Fusion（多字段分数融合）

### 方法 2A：Weighted RRF（当前方案）

```
RRF_score(d) = Σ_field  w_field / (k + rank_field(d))
```

- 基于排名（rank）而非原始分数，对不同度量/尺度天然兼容
- k 参数控制头部排名的权重集中度（k 越小越集中于 top-1）
- 当前 k=60，fusion_weights 全部 1.0

**优点：**
- 不需要分数归一化，不同度量可以直接融合
- 排名融合比分数融合对噪声更鲁棒
- Qdrant 原生支持

**缺点：**
- 丢失了分数的绝对大小信息（两个 cosine=0.99 和 cosine=0.50 的结果，排名可能只差 1）
- k 和 weights 的最优值需要实验调
- 只有 Qdrant 支持，InMemory 后端完全不做多字段融合

### 方法 2B：Weighted Score Sum（加权分数求和）

```
final_score(d) = Σ_field  w_field · sim_field(q, d)
```

最直观的方法——每个字段的 similarity 乘以权重后求和。

**前提：** 所有字段的分数必须在同一尺度上。如果 vision 用 cosine ∈ [-1,1] 而 robot_state 用 L2 ∈ [0, +∞)，直接加权求和没有意义。需要先归一化到统一区间。

**归一化方式：**
- Min-max：`(score - min) / (max - min)` → [0, 1]（需要全局或 batch 统计）
- Z-score：`(score - μ) / σ`（需要历史统计）
- 固定区间映射：cosine 已在 [-1,1]，L2 用 `1/(1+dist)` 映射到 (0,1]

**优点：**
- 保留分数绝对值信息
- 容易理解和调试
- 可在任何后端实现（不依赖 Qdrant RRF）

**缺点：**
- 需要分数归一化，不同度量混用时归一化策略影响大
- 异常分数（如某字段全是低分）会拉偏总分

### 方法 2C：Learned Fusion（学习型融合）

训练一个小模型，输入每个字段的 similarity 分数，输出最终分数。

```python
# 输入: [sim_vision_0, sim_vision_1, sim_vision_2, sim_prompt, sim_state]
# 输出: final_score ∈ [0, 1]

fusion_net = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
)
```

**训练信号：** 需要标注"这两个 step 是否应该被认为相似"，可以从：
- 同 episode 相邻 step → 正样本（应该高分）
- 跨 episode 随机 step → 负样本（应该低分）
- 同 task 不同 episode 的对应阶段 → 正样本

**优点：**
- 自动学习各字段的最优权重和非线性组合
- 可以学到"vision 匹配但 state 不匹配 → 低分"这样的交互规则

**缺点：**
- 需要离线训练 + 部署
- 只有 5 维输入的 MLP 其实和手调权重差不多
- 无法利用 Qdrant 的原生融合加速，需要先取 top-K per field 再本地重排序

### 方法 2D：Two-Stage Retrieval（两阶段检索）

```
Stage 1 (召回): 单字段快速检索 top-100 候选
Stage 2 (精排): 对候选集计算全字段加权分数，重排序取 top-K
```

- Stage 1 用最有区分度的单字段（通常 vision_0）快速召回
- Stage 2 在小候选集上做精细的多字段融合

**优点：**
- 兼顾速度和精度
- Stage 2 可以用任意复杂的融合方法（包括 learned fusion）
- 减少多字段并行搜索的开销

**缺点：**
- Stage 1 的字段选择不当会漏掉好结果
- 实现复杂度更高

### Layer 2 推荐

| 方法 | 适用阶段 | 理由 |
|------|---------|------|
| **Weighted Score Sum** | Phase 1 首选 | 简单直观，保留分数信息，可在所有后端实现 |
| **Weighted RRF** | Phase 1 备选 | 当前已有实现，排名融合对不同度量天然兼容 |
| **Two-Stage** | Phase 2 优化 | 当数据量大时提升速度，保持精度 |
| **Learned Fusion** | Phase 3 | 大部分情况下与手调权重差别不大，优先级低 |

> **关键洞察：** 如果 Layer 1 各字段都用 cosine（归一化后分数可比），Weighted Score Sum 比 RRF 更好——它保留了"这个 vision 匹配度是 0.99 还是 0.60"的信息，而 RRF 只看排名。RRF 的优势场景是字段度量不统一时（如 vision 用 cosine，state 用 L2）。

---

## 组合推荐

| 阶段 | Layer 1 (Field Similarity) | Layer 2 (Fusion) |
|------|---------------------------|-------------------|
| **Phase 1** | vision/prompt: Cosine, state: Cosine | Weighted Score Sum（权重先均等，后续实验调） |
| **Phase 1-alt** | 全部 Cosine | 保持 Weighted RRF（已有实现，先验证降维效果） |
| **Phase 2** | vision/prompt: Cosine, state: L2 (归一化到 [0,1]) | Weighted Score Sum + 实验调权重 |
| **Phase 3** | 同 Phase 2 | Two-Stage（vision 召回 + 全字段精排） |

Phase 1 的核心建议是：**先不动 Layer 2，专注于降维（上一节的 Mean Pooling）**。降维带来的收益远大于换融合方法。等降维落地后，再用 A/B 实验对比 RRF vs Score Sum。

---

## 与现有架构的对应关系

```
Layer 1 (Field Similarity)  →  VectorStoreBackend 的 distance metric 配置
                                （Qdrant: per-vector distance; InMemory: F.cosine_similarity）

Layer 2 (Cross-Modal Fusion) → SearchStrategy 的职责
                                （当前: QdrantWeightedRrfKnnStrategy 通过 backend_hints 传 RRF 参数）
                                （改进: 可在 SearchStrategy 层做 score-level 融合）
```

修改 Layer 1 需要改 Backend 配置（Qdrant collection schema 的 distance 类型）。
修改 Layer 2 可以在 SearchStrategy 层做，不需要改 Backend——取回 per-field top-K，在 Strategy 内重排序。

---
---

# Dual Encoder vs Cross Encoder：学习型 Similarity 架构

## 概念与定位

Dual Encoder 和 Cross Encoder 是信息检索领域的两种核心架构，分别对应检索系统的**召回**和**精排**阶段。它们不仅仅是"field-level learned similarity"，而是对整个 similarity 流水线的重新思考。

```
                    Dual Encoder                    Cross Encoder
                    (Bi-Encoder)                    (Reranker)

输入方式:      query 和 candidate 独立编码         query 和 candidate 拼接后联合编码
输出:          各自的 embedding → 比较             直接输出 similarity score
预计算:        ✅ candidate 可离线编码              ❌ 每对都要实时计算
速度:          O(1) per candidate (向量检索)        O(n) per candidate (前向传播)
精度:          中等（无交叉注意力）                 高（完整交叉注意力）
典型用途:      Stage 1 召回 top-K                   Stage 2 对 K 个候选精排
```

在我们的 cache 系统中的对应关系：

```
当前架构:     KeyBuilder(frozen SigLIP) + cosine  ≈  frozen Dual Encoder

改进方向 1:   Trained Dual Encoder                →  替换 KeyBuilder + 降维层
改进方向 2:   Dual Encoder 召回 + Cross Encoder 精排  →  Two-Stage 架构
```

---

## Dual Encoder（双塔模型）

### 原理

Query 和 candidate 各自通过一个 encoder 映射到共享 embedding 空间，然后用简单度量（cosine/dot product）比较。两个 encoder 可以共享参数（Siamese）或独立（Asymmetric）。

```
query step:      [vision, prompt, state] → Encoder_Q → z_q ∈ ℝ^d
candidate step:  [vision, prompt, state] → Encoder_C → z_c ∈ ℝ^d

similarity = cosine(z_q, z_c)  或  z_q · z_c
```

### 与当前架构的关系

当前系统实际上已经是一个 dual encoder，只是 encoder 是 frozen 的：

```
当前:    SigLIP(frozen) → flatten/pool → cosine
         encoder 没有针对"相似 step 检索"这个任务优化

训练后:  SigLIP(frozen) → Projection MLP(trained) → cosine
         MLP 学习了什么样的 step 应该被认为相似
```

之前讨论的 Learned Projection Head（降维第二层的方法 F）本质上就是在训练一个 dual encoder 的投影层。

### 在我们场景中怎么做

**架构选择：在 frozen backbone 上加 projection head**

不微调 SigLIP/PaliGemma（太大、会破坏推理质量），只训练上面的投影层。

```python
class DualEncoderHead(nn.Module):
    """将多模态 stage1 输出投影到统一检索空间"""
    def __init__(self, vision_dim=2048, state_dim=32, out_dim=256):
        super().__init__()
        # 各模态独立的投影
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))
        self.prompt_proj = nn.Sequential(
            nn.Linear(vision_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))
        # 跨模态融合（可选）
        self.fusion = nn.Linear(out_dim * 5, out_dim)  # 3 vision + prompt + state

    def forward(self, vision_pools, prompt_pool, state):
        """
        vision_pools: list of 3 × [B, 2048] (mean-pooled per image)
        prompt_pool:  [B, 2048] (mean-pooled)
        state:        [B, 32]
        """
        v_embs = [self.vision_proj(v) for v in vision_pools]    # 3 × [B, out_dim]
        p_emb = self.prompt_proj(prompt_pool)                    # [B, out_dim]
        s_emb = self.state_proj(state)                           # [B, out_dim]

        fused = self.fusion(torch.cat(v_embs + [p_emb, s_emb], dim=-1))
        return F.normalize(fused, dim=-1)                        # [B, out_dim]
```

**关键设计决策：是 per-field 独立编码还是融合后单向量？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Per-field 独立投影**（每个字段各出一个向量） | 保留 per-field 权重可调性；兼容 Qdrant named vectors | 仍需 Layer 2 fusion；字段间交互被忽略 |
| **Fusion 后单向量**（所有字段融合成一个向量） | 一次 cosine 完成所有比较；天然学到跨模态交互 | 丢失 per-field 可解释性；权重不可调 |

**推荐：先做 per-field 独立投影**（和降维 MLP 一致），后续可加 fusion 层。

### 训练方法

与之前讨论的 Learned Projection Head 完全一致：

1. **数据：** HDF5 收集的 episode 数据，mean pool 后得到 per-step features
2. **正样本对：** 同 episode 相邻 step（时间窗 ±3）
3. **负样本：** in-batch negatives（同 batch 内其他 step）
4. **损失函数：** InfoNCE

```python
# query, candidate: [B, out_dim]  (L2 normalized)
sim_matrix = query @ candidate.T / temperature  # [B, B]
labels = torch.arange(B)
loss = F.cross_entropy(sim_matrix, labels)
```

5. **只训练投影层**，SigLIP backbone frozen

---

## Cross Encoder（交叉编码器）

### 原理

将 query 和 candidate 的原始特征**拼接后联合编码**，通过交叉注意力机制捕捉两者之间的细粒度交互，直接输出相似度分数。

```
输入: concat([query_features, [SEP], candidate_features])
      → Transformer / MLP
      → scalar score ∈ [0, 1]
```

### 为什么比 Dual Encoder 更准确

Dual Encoder 中，query 和 candidate 各自独立编码——它们之间没有信息交换。这意味着 encoder 必须把**所有可能需要比较的信息**都压缩进一个固定维度的向量里。

Cross Encoder 允许 query 的每个 token 直接 attend to candidate 的每个 token，可以做到：
- "query 的左手腕图像中有一个红色方块" + "candidate 的左手腕图像中也有红色方块" → 高分
- "query 的 state 显示夹爪打开" + "candidate 的 state 显示夹爪关闭" → 即使 vision 很像也给低分

这种 **条件性判断** 是 dual encoder（独立编码 + cosine）做不到的。

### 在我们场景中怎么做

**方案：轻量级 Cross Encoder 作为 reranker**

不用大 Transformer——输入维度已经很小（mean pooled 后每 step 只有 ~8k 维），用 MLP 即可。

```python
class CrossEncoderReranker(nn.Module):
    """对 Dual Encoder 召回的 top-K 候选做精排"""
    def __init__(self, per_step_dim=8224):
        super().__init__()
        # 输入: query features + candidate features + element-wise 交互特征
        input_dim = per_step_dim * 3  # [q, c, |q-c|]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_feats, candidate_feats):
        """
        query_feats:     [B, D]  (一个 query 重复 K 次)
        candidate_feats: [B, D]  (K 个候选)
        returns:         [B, 1]  similarity scores
        """
        interaction = torch.abs(query_feats - candidate_feats)
        combined = torch.cat([query_feats, candidate_feats, interaction], dim=-1)
        return self.net(combined)
```

**输入特征设计：**

| 特征 | 维度 | 含义 |
|------|------|------|
| `query_feats` | D | query step 的 pooled features |
| `candidate_feats` | D | candidate step 的 pooled features |
| `\|query - candidate\|` | D | 逐元素差异（哪些维度不同） |
| **总输入** | **3D** | 拼接后送入 MLP |

element-wise difference `|q-c|` 是关键——它直接编码了"哪里不同"，比让 MLP 从 `[q, c]` 拼接中自己学更高效。

### 训练方法

**与 Dual Encoder 不同——需要 pointwise/pairwise 标签，不是 contrastive。**

**方案 A：Pointwise（二分类）**

```
正样本: (query, candidate) 来自同 episode 相邻 step → label = 1
负样本: (query, candidate) 来自不同 episode 随机 step → label = 0
损失:   BCE loss
```

**方案 B：Pairwise（排序学习）**

```
三元组: (query, positive, negative)
  positive = 同 episode 相邻 step
  negative = 随机 step

损失: margin ranking loss
  loss = max(0, margin - score(q, pos) + score(q, neg))
```

**方案 C：Distillation from oracle（推荐）**

如果有一个"ground truth similarity"定义（比如两个 step 的 action chunk 之间的 L2 距离），可以用它作为 soft label：

```python
# oracle_sim: 基于 action chunk 距离计算的 "真实" 相似度
# pred_sim: cross encoder 输出的预测相似度
loss = F.mse_loss(pred_sim, oracle_sim)
```

这比二分类标签更有信息量——不只区分"相似/不相似"，而是学到"有多相似"。

### 训练数据量估计

Cross Encoder 比 Dual Encoder 需要更多训练数据（因为不能用 in-batch negatives，需要显式构造正负对）：

| 数据量 | 效果 |
|--------|------|
| 1k pairs | 基本可用，容易过拟合 |
| 10k pairs | 合理效果 |
| 100k+ pairs | 稳定泛化 |

从 HDF5 episode 数据生成 pair 很容易——每个 episode 有几百个 step，组合量很大。

---

## Two-Stage 完整架构

```
┌─────────────────────────────────────────────────────────┐
│                     Query Step                          │
│  [vision_0, vision_1, vision_2, prompt_emb, state]     │
│                        │                                │
│                   Mean Pool                             │
│                        │                                │
│              ┌─────────┴─────────┐                      │
│              ▼                   ▼                      │
│     Dual Encoder Head    (raw pooled features           │
│     → query_emb [256]     kept for Stage 2)             │
│              │                                          │
│              ▼                                          │
│  ┌───── Stage 1: 召回 ─────┐                            │
│  │  Qdrant cosine search   │                            │
│  │  query_emb vs all       │                            │
│  │  → top-K candidates     │                            │
│  └───────────┬─────────────┘                            │
│              │ K candidates (IDs + pooled features)     │
│              ▼                                          │
│  ┌───── Stage 2: 精排 ─────┐                            │
│  │  Cross Encoder MLP      │                            │
│  │  input: [q, c, |q-c|]  │                            │
│  │  → re-scored top-K      │                            │
│  └───────────┬─────────────┘                            │
│              │                                          │
│              ▼                                          │
│       Final ranked result                               │
└─────────────────────────────────────────────────────────┘
```

### 延迟预算分析

| 阶段 | 操作 | 预估延迟 |
|------|------|---------|
| Mean Pool | 3 个 mean + L2 norm | < 0.1 ms |
| Dual Encoder Head | MLP forward (5 个 Linear) | < 0.5 ms |
| Stage 1: Qdrant search | 单向量 cosine top-50 | 1-5 ms |
| Stage 2: Cross Encoder | 50 次 MLP forward (可 batch) | < 1 ms |
| **总计** | | **< 7 ms** |

对比当前：5 个 named vector × chunked RRF 搜索 ≈ 10-50 ms（取决于数据量和 chunk 数）

Two-Stage 架构反而可能**更快**，因为 Stage 1 只搜一个向量。

---

## Dual Encoder vs Cross Encoder 对比总结

| 维度 | Dual Encoder | Cross Encoder |
|------|-------------|---------------|
| **角色** | 召回（全库检索） | 精排（小候选集重排序） |
| **输入** | query 和 candidate 独立编码 | query + candidate 联合编码 |
| **速度** | 快（预计算 + ANN 索引） | 慢（每对实时计算） |
| **精度** | 中（无跨模态交互） | 高（完整交叉注意力） |
| **训练数据** | 少（in-batch negatives） | 多（需要显式正负对） |
| **可预计算** | ✅ candidate 离线编码入库 | ❌ 每次 query 都要重算 |
| **在我们系统中** | 替换 KeyBuilder + 降维层 | 新增 reranker，在 SearchStrategy 内 |

---

## 实施建议

| Phase | 做什么 | 原因 |
|-------|--------|------|
| **Phase 1** | Mean Pool + L2 Norm（无学习） | 先消除维度问题，建立 baseline |
| **Phase 2** | 训练 Dual Encoder Head | 让 embedding 空间对"step 相似性"优化 |
| **Phase 3** | 加 Cross Encoder Reranker | 提升精度，尤其在 borderline case |

Phase 2 和 Phase 3 可以独立训练和部署。Dual Encoder 先上线，积累一段时间的检索数据后，用这些数据训练 Cross Encoder。

### 与现有架构的集成点

```
Dual Encoder Head   →  新的 KeyBuilder 实现（DualEncoderKeyBuilder）
                       build() 内调用 MLP forward，输出单向量或 per-field 向量

Cross Encoder       →  新的 SearchStrategy 实现（TwoStageStrategy）
                       search() 内:
                         1. 调用 storage.search() 获取 top-K（Stage 1）
                         2. fetch candidate features
                         3. cross encoder rerank（Stage 2）
                         4. 返回重排序后的结果
```

不需要修改 Orchestrator、Gate、Judge 或 Backend。

---

## CP1 Calibration 实验发现与改进方向（2026-04-06）

### Calibration 结果

对 4 种池化方式（mean_pool / spatial_pool_16 / spatial_pool_64 / max_pool）构建 artifact（1040 entries）后，运行 `calibrate_score_sum_stats.py` 计算 same-task vs cross-task 区分度（separation = same_task_mean - cross_task_mean）：

| 字段 | 相似度类型 | separation 范围 | 评价 |
|------|-----------|----------------|------|
| vision_0 | cosine | 0.001 ~ 0.003 | 很低，池化后几乎无区分力 |
| vision_1 | cosine | 0.001 ~ 0.002 | 同上 |
| prompt_emb | cosine | ~0.000002 | 几乎为零，CP1 级别 prompt embedding 对所有 task 一模一样 |
| robot_state | L2 | ~0.06 | **最好**，是 vision 的 20-60 倍 |

**核心问题**：256 个 ViT patch token 中大部分对应静态背景（桌面、墙壁），只有少数 patch 捕获任务相关物体（机械臂、目标物体）。简单池化（mean/max/spatial）把所有 patch 平等对待，有用信号被背景淹没。

相对而言 spatial_pool_16 保留最多空间信息（sep=0.0025），mean_pool 最差（sep=0.001）。

### 改进方向：更智能的 Token 处理

#### 方向 1: Variance-based Token Selection（静态分析，最简单）

从已有数据预计算每个 token position 的方差，选最活跃的 top-k 个 token：

```python
# 对所有 entries 的 vision_0 [256, 2048] 做统计
all_tokens = torch.stack(...)  # [N, 256, 2048]
variance = all_tokens.var(dim=0)  # [256, 2048]
position_importance = variance.sum(dim=1)  # [256]
top_k_positions = position_importance.topk(k=32).indices  # 选最活跃的 32 个 token
```

只用这 32 个 token flatten 成 32×2048 = 65536d 向量做 cosine。

- **优点**：离线计算一次 mask，search 时直接用，改动小
- **缺点**：mask 是全局的，不同 task 的关键 token 可能不同

#### 方向 2: MaxSim（ColBERT 风格，不压缩）

完全不池化，直接做 token-level 匹配：

```python
def maxsim(query, candidate):
    # query: [256, 2048], candidate: [256, 2048]
    sim_matrix = query @ candidate.T  # [256, 256]
    # 每个 query token 找最相似的 candidate token
    max_per_query = sim_matrix.max(dim=1).values  # [256]
    return max_per_query.mean()  # scalar
```

信息检索领域（ColBERT）验证过的方法。

- **优点**：保留完整的 token 级匹配，不丢信息，理论上区分度最高
- **缺点**：计算量大（每对比较是 256×256 矩阵乘法），但 1040 entries batched 应该可接受

#### 方向 3: Residual Encoding（减去均值模板）

```python
# 离线：计算所有 entries 的 mean template
mean_template = all_tokens.mean(dim=0)  # [256, 2048]

# 每个 entry 的 key = tokens - mean_template（去掉共享的静态部分）
residual = tokens - mean_template
key = residual.flatten()  # 或者对 residual 再池化
```

- **核心思想**：所有 entry 共有的部分（背景）被减掉，剩下的是差异部分
- **优点**：池化后 cosine 区分度会显著提升，改动小
- **缺点**：依赖均值模板的代表性

#### 方向 4: Cross-step Attention（时序差异分析）

用每个 episode 内 token 的时间变化识别动态 token：

```python
# 同一 episode 的连续 step 对比
delta = tokens_t - tokens_{t-1}  # [256, 2048]
dynamic_mask = delta.norm(dim=1) > threshold  # 哪些 token 变化大
```

变化大的 token 对应机械臂运动轨迹、物体移动，是最有区分力的。

- **优点**：捕获真正的动态信号
- **缺点**：需要时序信息，构建 artifact 时更复杂

### 推荐实施顺序

| 优先级 | 方向 | 原因 |
|--------|------|------|
| **P0** | 方向 3（Residual）+ 方向 1（Variance Selection）组合 | 成本低，改动小，只需在 artifact 构建时预计算 mean_template 和 variance mask，新增一个 `CP1ResidualKeyBuilder`，与现有实验框架完全兼容 |
| **P1** | 方向 2（MaxSim） | 效果可能最好，但需要修改 similarity 计算逻辑（不再是简单向量 cosine），需要在 `in_memory_backend.py` 中新增 similarity type |
| **P2** | 方向 4（Cross-step） | 需要更多数据处理逻辑，但与方向 1 可以组合使用 |

这些方向与本文档前述的 Dual Encoder / Cross Encoder 方案是正交的——前者改进 token 表示/匹配方式（信号提取），后者改进整体 embedding 空间（学习优化）。两者可以叠加。
