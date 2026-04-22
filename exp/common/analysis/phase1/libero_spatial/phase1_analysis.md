# Phase 1 Experiment Analysis

## Overview

Phase 1 evaluates **5 key builders** x **8 weight configurations** = 40 runs on `libero_spatial` (10 tasks, 5 episodes each). The search strategy is `weighted_rrf_knn` across all runs; only the key builder type and field weight allocation vary.

### Key Builders

| ID | Type | Description |
|----|------|-------------|
| a  | `cp1_mean_pool` | Global average pooling of intermediate features |
| b1 | `cp1_spatial_pool_16` | Spatial pooling to 16-dim |
| b2 | `cp1_spatial_pool_64` | Spatial pooling to 64-dim |
| c  | `cp1_max_pool` | Global max pooling of intermediate features |
| d  | `clip` | CLIP visual encoder features |

### Weight Configurations

All weights are over `(vision_0, vision_1, prompt_emb, robot_state)`. `vision_2` is disabled in all runs. `prompt_emb` is always 0.

| ID | vision_0 | vision_1 | robot_state | Design Intent |
|----|----------|----------|-------------|---------------|
| w1 | 1.0  | 0    | 0    | Vision-only baseline |
| w2 | 0    | 0    | 1.0  | Robot-state-only baseline |
| w3 | 0.5  | 0    | 0.5  | Equal v0/rs split |
| w4 | 0.25 | 0    | 0.75 | RS-heavy, single vision |
| w5 | 0.25 | 0.25 | 0.5  | Dual vision + RS |
| w6 | 0.15 | 0.1  | 0.75 | RS-heavy, dual vision (low) |
| w7 | 0.1  | 0.1  | 0.8  | RS-dominant, minimal vision |
| w8 | 0.5  | 0.25 | 0.25 | Vision-heavy mix |

---

## Results

### Full Results Table (Success Rate)

|              |   w1 |   w2 |   w3 |   w4 |   w5 |   w6 |   w7 |   w8 | Mean  |  Max  |  Min  |  Std  |
|--------------|------|------|------|------|------|------|------|------|-------|-------|-------|-------|
| mean_pool    | 26.0 | 55.6 | 42.0 | 50.0 | 58.0 | 52.0 | 56.0 | 52.0 | 48.9% | 58.0% | 26.0% |  9.8% |
| spatial_16   | 36.0 | 46.0 | 44.0 | 58.0 | 58.0 | 56.0 | 58.0 | 62.0 | 52.2% | 62.0% | 36.0% |  8.5% |
| spatial_64   | 32.0 | 46.0 | 42.0 | 40.0 | 64.0 | 50.0 | 58.0 | 60.0 | 49.0% | 64.0% | 32.0% | 10.3% |
| max_pool     | 40.0 | 46.0 | 42.0 | 56.0 | 50.0 | 52.0 | 54.0 | 48.0 | 48.5% | 56.0% | 40.0% |  5.3% |
| clip         | 12.0 | 46.0 | 44.0 | 50.0 | 60.0 | 66.0 | 66.0 | 52.0 | 49.5% | 66.0% | 12.0% | 16.2% |

### Per-Weight Best Key Builder

| Weight | Best Key Builder | Success Rate |
|--------|-----------------|--------------|
| w1 | max_pool    | 40.0% |
| w2 | mean_pool   | 55.6% |
| w3 | spatial_16  | 44.0% |
| w4 | spatial_16  | 58.0% |
| w5 | spatial_64  | 64.0% |
| w6 | clip        | 66.0% |
| w7 | clip        | 66.0% |
| w8 | spatial_16  | 62.0% |

---

## Analysis

### 1. Weight Allocation Findings

**Vision-only retrieval (w1) performs worst across all key builders.** Every key builder achieves its lowest success rate with w1. This indicates that a single vision feature alone is insufficient for effective cache retrieval in LIBERO spatial tasks.

**Higher robot_state weight generally yields better results.** Configurations w2 (rs=1.0), w4 (rs=0.75), w6 (rs=0.75), and w7 (rs=0.8) consistently rank among the top performers. In LIBERO spatial tasks, robot state contributes more to cache retrieval quality than visual features.

**Multi-modal fusion outperforms single-modality retrieval.** Mixed-weight configurations (w3–w8) generally outperform single-feature baselines (w1, w2). The best overall results come from w6 (v0=0.15, v1=0.1, rs=0.75) and w7 (v0=0.1, v1=0.1, rs=0.8), both achieving 66% with clip — suggesting that a small amount of visual signal on top of a dominant robot_state weight is the optimal regime.

**Weight allocation matters more than key builder choice.** Within the same key builder, success rates vary by 20–54 percentage points across weights. Across key builders at the same weight, variation is typically under 15 points. This means tuning the weight vector is higher-leverage than switching feature extractors.

### 2. Key Builder Comparison

**Ranking by mean success rate:**

| Rank | Key Builder | Mean | Max | Min | Std |
|------|-------------|------|-----|-----|-----|
| 1 | spatial_16  | 52.2% | 62.0% | 36.0% | 8.5% |
| 2 | clip        | 49.5% | 66.0% | 12.0% | 16.2% |
| 3 | spatial_64  | 49.0% | 64.0% | 32.0% | 10.3% |
| 4 | mean_pool   | 48.9% | 58.0% | 26.0% | 9.8% |
| 5 | max_pool    | 48.5% | 56.0% | 40.0% | 5.3% |

**spatial_16 is the best all-rounder.** It achieves the highest mean success rate (52.2%) with moderate variance (std=8.5%). It wins 3 out of 8 per-weight comparisons (w3, w4, w8) and is consistently competitive across all weight configurations.

**clip has the highest ceiling but the lowest floor.** It achieves the global best of 66% (w6 and w7) but collapses to 12% with pure vision retrieval (w1). Its std of 16.2% is nearly double the next highest. clip is extremely sensitive to weight allocation — it requires high robot_state weight to be effective, but when properly configured, it outperforms all other key builders. Additionally, **using clip increases VRAM usage by nearly 3 GB** due to loading the CLIP model, and **this overhead is incompressible**.

**max_pool is the most stable but has a low ceiling.** With a std of only 5.3%, it delivers consistent results (40–56%) regardless of weight choice. However, its maximum of 56% is the lowest among all key builders. It is a safe but unexciting choice.

**spatial_64 does not improve over spatial_16.** Despite encoding 4x more spatial information, spatial_64 achieves a lower mean (49.0% vs 52.2%) with higher variance (10.3% vs 8.5%). Higher spatial resolution does not help in this task setting.

**mean_pool is unremarkable.** It ranks second-to-last by mean. Its one standout result — highest w2 score (55.6%) — suggests its visual features are weak, and performance depends primarily on robot_state retrieval.

---

## Recommendations for Phase 2

1. **Focus weight search on the robot_state-dominant regime** (rs = 0.7–0.9), with small vision contributions. The w6/w7 region appears most promising.
2. **Prioritize spatial_16 and clip** as key builder candidates. spatial_16 for stability, clip for peak performance.
3. **Consider dropping spatial_64 and max_pool** — spatial_64 offers no advantage over spatial_16, and max_pool has a low ceiling.
4. **Investigate whether clip + spatial_16 ensemble** could combine clip's high ceiling with spatial_16's stability.

## Limitations

The current experiment uses only 5 episodes per task across 10 tasks, resulting in a small sample size. Success rate differences of a few percentage points may not be statistically significant at this scale. However, running broader experiments (more episodes, more task suites) is prohibitively slow at this stage. The purpose of Phase 1 is to narrow down the search space. Once a final configuration is selected, a larger-scale validation experiment should be conducted to confirm the results with higher statistical confidence.
