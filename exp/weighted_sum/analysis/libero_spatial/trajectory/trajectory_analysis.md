# Weighted-Sum Trajectory Search — 实验分析

> 在 weighted_sum 两层 `weighted_score_sum_knn` 检索（Layer-1 per-field `zscore(tanh)` +
> Layer-2 加权和，`judge=always_hit` 纯回放）的最优配置之上叠加多步 query 历史聚合
> （`trajectory_depth > 1`）。libero_spatial，每配置 100 ep（10 task × 10 held-out trial）。
> server=jupyter H200（`--replicas 2`，与 weighted_sum depth-1 基线同机，可比）。
> 设计见 [`logs/weighted_sum_trajectory_search.log.md`](../../../logs/weighted_sum_trajectory_search.log.md)。

## 概览

- **18 base 配置**（去重）：① per-keybuilder（4 个 CP1 keybuilder 各 top1+top2+倒数第二，
  A 口径=正规网格同 zscore）；② 全实验 top10。两组重叠 4 个，并集 18。
- **depth ∈ {3,4,5,6}**，`trajectory_weights` 复用老实验递减方案。**depth-1 基线复用
  weighted_sum 已测 SR**（同 jupyter 机）。
- 共 72 trajectory yaml × 100 ep = **7200 episode**，全部完成（7200/7200，72/72 yaml）。

## 结果（成功率 %，按 base 角色分组）

| keybuilder · 权重 v0/v1/rs | 角色 | d1 | d3 | d4 | d5 | d6 | best | Δ |
|---|---|---|---|---|---|---|---|---|
| spatial_16 · 6/44/50 | top1/top10 | **74** | 62 | 62 | 62 | 62 | d3 | **−12** |
| spatial_16 · 6/50/44 | top2/top10 | **74** | 61 | 63 | 65 | 63 | d5 | −9 |
| max_pool · 31/25/44 | top1/top10 | 73 | 64 | 65 | 65 | 63 | d4 | −8 |
| max_pool · 6/25/69 | top2/top10 | 73 | 67 | 68 | 69 | 68 | d5 | −4 |
| spatial_16 · 19/31/50 | top10 | 73 | 67 | 63 | 66 | 68 | d6 | −5 |
| spatial_16 · 25/37/37 | top10 | 72 | 65 | 67 | 66 | 66 | d4 | −5 |
| spatial_16 · 25/44/31 | top10 | 73 | 63 | 66 | 68 | 66 | d5 | −5 |
| spatial_16 · 31/38/31 | top10 | 73 | 65 | 66 | 66 | 64 | d4 | −7 |
| spatial_16 · 31/44/25 | top10 | 73 | 67 | 67 | 66 | 65 | d3 | −6 |
| spatial_16 · 12/37/50 | top10 | 72 | 67 | 69 | 68 | 66 | d4 | −3 |
| mean_pool · 12/12/75 | top2 | 66 | 65 | 65 | 63 | 64 | d3 | −1 |
| mean_pool · 50/–/50 | top1 | 67 | 65 | 66 | 64 | 67 | d6 | 0 |
| spatial_64 · 12/12/75 | top1 | 67 | 68 | 64 | 67 | 67 | d3 | +1 |
| spatial_64 · 12/50/37 | top2 | 67 | 63 | 68 | 67 | 64 | d4 | +1 |
| **spatial_16 · 25/–/75** | **2nd-worst** | 60 | **67** | 65 | 65 | 63 | d3 | **+7** |
| **max_pool · 25/75/–** | **2nd-worst** | 56 | 61 | 61 | **64** | 63 | d5 | **+8** |
| **mean_pool · 87/–/12** | **2nd-worst** | 51 | 55 | 60 | **63** | 63 | d5 | **+12** |
| **spatial_64 · 62/37/–** | **2nd-worst** | 54 | 65 | 66 | 64 | **67** | d6 | **+13** |

### 分组聚合

| 角色 | n | 平均 Δ（最佳 traj depth − d1） |
|---|---|---|
| **2nd-worst（弱基线）** | 4 | **+10.0 pp** |
| top2 | 4 | −3.2 pp |
| top1 | 4 | −4.7 pp |
| top10-only（强基线） | 6 | −5.2 pp |

### Per-depth 聚合（全 18 base 均值）

| depth | mean SR |
|---|---|
| **1（基线）** | **67.7%** |
| 3 | 64.3% |
| 4 | 65.1% |
| 5 | 65.4% |
| 6 | 64.9% |

## 分析

### 1. 「救弱不救强」——与老 trajectory 实验核心结论一致

把 18 个 base 按 weighted_sum 角色拆开，效应非常锐利：

- **弱基线（2nd-worst）全部被 trajectory 救活，平均 +10.0 pp**：spatial_64/62-37（+13）、
  mean_pool/87（+12）、max_pool/25-75（+8）、spatial_16/25-75（+7）。
- **强基线（top1/top2/top10）几乎全部倒退**，平均 −3 ~ −5 pp，最差 spatial_16/6-44-50
  从 74% 掉到 62%（−12pp）。

这复刻了老实验（weighted_rrf）的判定：**trajectory 是鲁棒性机制，不是能力提升**。当单步
query 已是好索引（强基线），混入更旧的 query 只会引入噪声、稀释时间特异性；当单步检索弱
（弱基线），多步聚合平滑掉单步错误，显著救场。

### 2. 与老实验的关键差异：本实验「平均无增益」

老 trajectory 实验（基底 weighted_rrf）per-depth 均值在 **d5 见顶（54.1%→58.9%，+5pp）**；
本实验 per-depth 均值 **d1（67.7%）最高，d3-d6 均更低**。原因不是 trajectory 更差，而是
**base 集合构成不同**：本实验 18 base 里 14 个是强基线（top10 + 各 keybuilder top2），只有
4 个弱基线；强基线的倒退主导了均值。老实验 15 base 里 5 个是 2nd-worst，弱基线占比更高，
故均值能被救弱效应抬正。**结论：trajectory 的平均收益高度依赖 base 池的强弱构成；在已调到
~74% 天花板的强配置上，trajectory 平均是负收益。**

### 3. 救弱的最佳 depth 偏深（d5/d6）

4 个被救的弱基线，最佳 depth 为 d5/d5/d6/d3——多数落在 **d5-d6**，比强基线（多在 d3/d4 见顶
后继续负）更深。直觉：弱基线需要更长历史窗口来平滑单步噪声，收益随 depth 缓升；强基线则
depth 越深、旧 query 噪声越多、掉得越快。这与老实验「峰值 4-5」基本吻合（救场场景偏 5）。

### 4. keybuilder 维度

强基线倒退在 spatial_16（top10 主力，8/10）上最明显（−3 ~ −12pp），因其 depth-1 已接近
74% 天花板，无上行空间只剩噪声。弱基线救场在 4 个 keybuilder 上都成立（+7 ~ +13pp），其中
mean_pool/spatial_64 这两个本就偏弱的 keybuilder 的弱配置救场幅度最大（+12/+13），与「弱者
受益更多」自洽。

### 5. 方法学可信度

depth-1 基线复用 weighted_sum 同机（jupyter H200）SR，trajectory depths 在同一 server
实例上跑，**无跨 GPU 污染**（weighted_sum §7 已证同机 run-to-run ≤1pp）。held-out init 防
泄漏门生效，`always_hit` 纯检索隔离。864 条 "Connection refused" 为 worker 连接瞬时拒绝，
经 conductor requeue 全部重试成功（7200/7200 终态完整），不影响数据。

## 结论

1. **trajectory 对 weighted_sum 检索是「救弱」工具**：弱配置 +10pp（最高 +13），强配置 −4pp。
2. **不要在已调优的强配置上开 trajectory**：weighted_sum 的 ~74% 天花板配置开 trajectory 反而掉 3-12pp。
3. **若用于救弱/分布漂移场景，depth 取 5 左右**；强配置若必须开，depth 3-4 损失最小。
4. 平均增益取决于 base 池强弱构成——本强基线主导的池子平均为负，但救弱信号清晰且稳健。

## 文件

- 图：`trajectory_results.png`（各 keybuilder SR×depth 折线，按角色分样式）、
  `trajectory_delta.png`（角色 Δ 分组柱 + per-depth 聚合）。
- 数据：`../../data/libero_spatial/trajectory/journal.jsonl`（7200 ep）、`../../data/libero_spatial/trajectory/results.json`（72 yaml SR）。
- 复现：`emit_trajectory_yamls.py`（生成 72 yaml）→ `run_phase2.py`（conductor 评测）→
  `summarize.py` → `plot_trajectory_results.py`。
