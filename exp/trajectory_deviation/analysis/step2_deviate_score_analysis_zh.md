# Step 2 偏离分数分析

数据来源：`exp/trajectory_deviation/data/deviate_scores/deviate_score_{cfg}.json`。

完整分析和主要图像由下面这个脚本生成：

```bash
uv run python -m exp.trajectory_deviation.analysis.analyze_step2_deviate_scores \
    --score-dir exp/trajectory_deviation/data/deviate_scores \
    --gt-dir exp/trajectory_deviation/data/gt
```

只画连续分布图和简版 summary 的轻量脚本是：

```bash
uv run python -m exp.trajectory_deviation.analysis.plot_deviate_score_distribution \
    --score-dir exp/trajectory_deviation/data/deviate_scores
```

这份分析参考 Step 1a report 的结构，但数据只使用 Step 2 刚得到的偏离分数输出。每个 replan cycle 的分数定义为 `cache_l2 / max(background_l2, 0.1)`。分数越高，表示 cache replay 的动作相对 GT 的偏离，比纯 inference 自身背景波动更大。

## 核心结论

- 三个 keybuilder 的整体分布非常接近：中位数大约在 `1.82-1.86`，`>=5` 的大偏离点占全部 cycle 的 `6.2-6.7%`。
- 典型 episode 大约有 `21-22` 个 replan cycle，其中大约 `1.3-1.5` 个 cycle 的分数 `>=5`。按中位数看，一个 episode 通常刚好有 `1` 个大偏离点。
- 高风险点不是均匀分布在轨迹里：它们最常出现在轨迹中段，并且在轨迹末段再次出现。每个 episode 最开始的 20% 区间大偏离点最少。
- task 维度的差异比 keybuilder 维度更明显。`task_8` 是最清楚的高风险任务，三个 keybuilder 都高，尤其是 `max_pool_w3_d5`。

## 整体分布

![偏离分数连续分布，按阈值着色](plots/step2_score_distribution_colored.png)

| 配置 | episode 数 | cycle 数 | 均值 | 中位数 | p90 | p95 | p99 | 最大值 | 每个 episode 平均 >=5 数 | 没有 >=5 的 episode 数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clip_w7_d4` | 159 | 3467 | 2.56 | 1.86 | 4.40 | 5.60 | 20.27 | 27.67 | 1.46 | 40 |
| `spatial16_w8_d4` | 154 | 3301 | 2.45 | 1.82 | 4.16 | 5.37 | 20.10 | 25.77 | 1.34 | 45 |
| `max_pool_w3_d5` | 150 | 3292 | 2.54 | 1.83 | 4.34 | 5.62 | 20.26 | 27.44 | 1.46 | 38 |

按 cycle 统计的分数区间：

| 配置 | <1 | 1-2 | 2-5 | >=5 |
|---|---:|---:|---:|---:|
| `clip_w7_d4` | 657 (19.0%) | 1225 (35.3%) | 1353 (39.0%) | 232 (6.7%) |
| `spatial16_w8_d4` | 668 (20.2%) | 1183 (35.8%) | 1244 (37.7%) | 206 (6.2%) |
| `max_pool_w3_d5` | 690 (21.0%) | 1136 (34.5%) | 1247 (37.9%) | 219 (6.7%) |

解释：大部分 cycle 不是非常小的扰动。大约 44-46% 的 cycle 至少达到背景波动的 `2x`，约 6-7% 达到 `>=5x`。所以 Step 3 不应该只随机抽 cycle，而应该重点覆盖 top-k 高分 cycle。

## Episode 内部大偏离点数量

![每个 episode 中 >=5 点数量的直方图](plots/step2_episode_ge5_count_hist.png)

大多数 episode 只包含少量大偏离点，通常是 0-2 个，但存在长尾。这支持继续使用类似 `k-grid 1 3 5` 的设置：`k=1` 捕捉最主要的尖峰，`k=3/5` 覆盖少数包含多个高风险点的 episode。

## Task 维度风险

![task/keybuilder 的 >=5 比例热力图](plots/step2_task_keybuilder_ge5_heatmap.png)

![合并三个 keybuilder 后的 task 风险排序](plots/step2_pooled_task_risk_ranking.png)

| task | 任务描述 | 合并后 >=5 cycle 占比 | 合并后每 episode 平均 >=5 数 | 合并后平均分 |
|---|---:|---:|---:|---:|
| 8 | 盘子旁边 -> 放到盘子 | 12.9% | 2.67 | 3.40 |
| 0 | 盘子和小模子之间 -> 放到盘子 | 9.7% | 1.57 | 2.91 |
| 5 | 小模子上 -> 放到盘子 | 6.8% | 1.30 | 2.46 |
| 4 | 木柜顶层抽屉里 -> 放到盘子 | 6.1% | 1.60 | 2.58 |
| 1 | 小模子旁边 -> 放到盘子 | 5.7% | 1.29 | 2.20 |
| 9 | 木柜上 -> 放到盘子 | 5.7% | 1.35 | 2.46 |
| 6 | 饼干盒旁边 -> 放到盘子 | 5.6% | 1.16 | 2.36 |
| 3 | 饼干盒上 -> 放到盘子 | 5.3% | 0.94 | 2.44 |
| 7 | 炉灶上 -> 放到盘子 | 5.2% | 1.30 | 2.39 |
| 2 | 桌面中央 -> 放到盘子 | 3.5% | 0.70 | 2.12 |

关键 task 结论：`task_8`（盘子旁边 -> 放到盘子）是最强的异常点。它的合并 `>=5` 密度最高，并且在极端 episode 列表中占比很高。`task_0` 排第二，`task_2` 则稳定地更低风险。

## 高分点在 Episode 内的位置

![>=5 点在 episode 内的相对位置](plots/step2_ge5_relative_position.png)

`>=5` 的点主要集中在 episode 的 20-60% 中段，并且在 80-100% 末段再次增多。0-20% 的早期阶段相对安静。这说明偏离分数捕捉到的主要不是初始靠近阶段，而更可能是接触、抓取、搬运、最终放置和对齐这些交互阶段。

## 极端 Episode

![按 >=5 点数量排序的极端 episode](plots/step2_top_episodes_ge5.png)

| 排名 | 配置 | episode | cycle 数 | >=5 数量 | 最大分数 | >=5 的 cycle 索引 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | `max_pool_w3_d5` | `task_8/episode_13` | 29 | 12 | 20.22 | `[9, 10, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28]` |
| 2 | `max_pool_w3_d5` | `task_8/episode_28` | 31 | 10 | 27.44 | `[14, 16, 17, 19, 20, 23, 24, 28, 29, 30]` |
| 3 | `spatial16_w8_d4` | `task_8/episode_28` | 31 | 8 | 25.77 | `[14, 16, 17, 19, 27, 28, 29, 30]` |
| 4 | `clip_w7_d4` | `task_9/episode_5` | 25 | 8 | 12.88 | `[6, 7, 8, 9, 10, 11, 12, 16]` |
| 5 | `clip_w7_d4` | `task_4/episode_22` | 31 | 7 | 21.81 | `[7, 12, 16, 17, 23, 24, 25]` |
| 6 | `spatial16_w8_d4` | `task_8/episode_24` | 27 | 6 | 24.90 | `[6, 9, 13, 15, 16, 24]` |
| 7 | `max_pool_w3_d5` | `task_1/episode_4` | 30 | 6 | 21.50 | `[13, 14, 19, 20, 22, 24]` |
| 8 | `spatial16_w8_d4` | `task_5/episode_24` | 20 | 6 | 20.78 | `[7, 11, 14, 15, 16, 19]` |
| 9 | `spatial16_w8_d4` | `task_4/episode_20` | 32 | 6 | 16.17 | `[6, 16, 17, 18, 29, 30]` |
| 10 | `clip_w7_d4` | `task_9/episode_20` | 28 | 5 | 23.35 | `[6, 9, 21, 23, 27]` |
| 11 | `clip_w7_d4` | `task_9/episode_31` | 26 | 5 | 22.88 | `[8, 9, 10, 11, 25]` |
| 12 | `max_pool_w3_d5` | `task_4/episode_20` | 32 | 5 | 22.55 | `[6, 10, 17, 18, 29]` |

这些 episode 很适合作为 Step 3 的定性 case study，因为它们不是只有一个孤立尖峰，而是包含持续或重复出现的大偏离。

## 对 Step 3 的影响

- Step 3 仍然应该包含三个 keybuilder；它们整体分布接近，但 task 级别差异仍然明显。
- 保留 `k-grid 1 3 5`：大多数 episode 只有 1-2 个大偏离点，但长尾 episode 需要 3-5 个点才能覆盖重复偏离。
- `task_8`、`task_0`，以及极端 episode 表里的高计数 episode，应该优先做可视化检查和问题定位。
- 位置选择上，`n-grid` 应该覆盖轨迹中段和末段的失败。末段峰值说明接近结束时的恢复能力仍然很重要。
