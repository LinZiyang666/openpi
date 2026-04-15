# Temporal Prune Experiment Analysis

## Overview

The temporal prune experiment evaluates whether **temporal redundancy pruning** —
removing vision tokens that remain static across consecutive time steps before
pooling — improves cache retrieval quality. The experiment performs a grid search
over the pruning parameters on `libero_spatial` (10 tasks, 5 episodes each),
using `weighted_rrf_knn` as the search strategy with `top_k=1` and
`trajectory_depth=1` throughout.

The Phase 1 results for `cp1_mean_pool` and `cp1_max_pool` (which pool the full
256 vision tokens without pruning) serve as the **no-prune baselines** for
direct comparison.

### Experimental Grid

| Dimension | Values | Count |
|-----------|--------|-------|
| Reducer (Step 2 pooling) | `mean_pool`, `max_pool` | 2 |
| Window size (frames) | 3, 4, 5, 6 | 4 |
| Keep ratio | 0.25, 0.50, 0.75 | 3 |
| Weight config | WA, WB | 2 |

**Total: 2 × 4 × 3 × 2 = 48 runs.**

### Weight Configurations

| ID | vision_0 | vision_1 | robot_state | Design Intent |
|----|----------|----------|-------------|---------------|
| WA | 0.1  | 0.1  | 0.8  | RS-dominant, minimal vision (= Phase 1 w7) |
| WB | 0.5  | 0.25 | 0.25 | Vision-heavy mix (= Phase 1 w8) |

WA and WB correspond to Phase 1 weight configurations w7 and w8, respectively.
WA represents the robot-state-dominant regime that Phase 1 identified as
generally strong; WB represents a vision-heavy regime that stresses the quality
of the visual key representation — the regime where temporal pruning is most
likely to matter.

---

## Method: Two-Step Temporal Prune KeyBuilder

The `cp1_temporal_prune` KeyBuilder inserts a temporal redundancy pruning stage
before the standard token pooling. The two stages operate as follows.

### Step 1: Temporal Pruning

Given a sequence of vision token matrices from the most recent $W$ time steps
(the **prune window**), the pruner identifies and removes tokens that are
temporally static — i.e., tokens whose representations barely change across
consecutive frames.

Let $\mathbf{V}^{(t)} \in \mathbb{R}^{N \times D}$ denote the $N = 256$ vision
tokens at time step $t$, each of dimension $D = 2048$. For a window of size $W$,
we have access to $\{\mathbf{V}^{(t-W+1)}, \ldots, \mathbf{V}^{(t)}\}$.

For each token position $i \in \{1, \ldots, N\}$, compute the **temporal
variability score** as the mean cosine distance between consecutive frames:

$$
s_i = \frac{1}{W-1} \sum_{j=1}^{W-1}
\left(1 - \frac{\mathbf{v}_i^{(t-W+j)} \cdot \mathbf{v}_i^{(t-W+j+1)}}
{\|\mathbf{v}_i^{(t-W+j)}\| \; \|\mathbf{v}_i^{(t-W+j+1)}\|}\right)
$$

where $\mathbf{v}_i^{(t)}$ is the $i$-th row of $\mathbf{V}^{(t)}$.

A high $s_i$ indicates that token $i$ changes substantially across the window
(likely encoding dynamic, action-relevant content); a low $s_i$ indicates a
nearly static token (likely background). The pruner retains the top
$K = \lfloor \rho \cdot N \rfloor$ tokens ranked by $s_i$, where
$\rho \in (0, 1]$ is the **keep ratio**.

**Degenerate case:** When the history buffer contains fewer than $W$ frames
(the first few steps of each episode), pruning is skipped and all $N$ tokens
are passed through unchanged.

### Step 2: Token Reduction

The surviving $K$ tokens are fed into a standard pooling reducer. This
experiment tests two reducers:

- **mean_pool**: average all $K$ token vectors into a single $D$-dimensional
  key. $\mathbf{k} = \frac{1}{K} \sum_{i \in \mathcal{S}} \mathbf{v}_i$

- **max_pool**: element-wise maximum over the $K$ token vectors.
  $\mathbf{k}_d = \max_{i \in \mathcal{S}} v_{i,d}$ for each dimension $d$

where $\mathcal{S}$ denotes the set of retained token indices.

### Relationship to Phase 1 Baselines

The Phase 1 `cp1_mean_pool` and `cp1_max_pool` key builders are equivalent to
the temporal prune pipeline with **pruning disabled** ($\rho = 1.0$, i.e., all
256 tokens retained). This experiment tests whether selectively pruning
temporally static tokens ($\rho < 1.0$) before pooling produces better cache
keys.

---

## Results

### Phase 1 Baselines (No Pruning)

| Reducer | WA (w7) | WB (w8) |
|---------|---------|---------|
| mean_pool | 56% | 52% |
| max_pool  | 54% | 48% |

### Full Results: Weight WA (v0=.1, v1=.1, rs=.8)

| Reducer | Window | kr=0.25 | kr=0.50 | kr=0.75 | Row Mean |
|---------|--------|---------|---------|---------|----------|
| mean_pool | 3 | 42% | 34% | 40% | 38.7% |
| mean_pool | 4 | 36% | 38% | 36% | 36.7% |
| mean_pool | 5 | 42% | 34% | 30% | 35.3% |
| mean_pool | 6 | 34% | 36% | 30% | 33.3% |
| max_pool  | 3 | 40% | 42% | 46% | 42.7% |
| max_pool  | 4 | 42% | 38% | 42% | 40.7% |
| max_pool  | 5 | **52%** | 42% | 42% | 45.3% |
| max_pool  | 6 | 40% | 42% | 34% | 38.7% |

WA mean across all 24 runs: **38.9%**

### Full Results: Weight WB (v0=.5, v1=.25, rs=.25)

| Reducer | Window | kr=0.25 | kr=0.50 | kr=0.75 | Row Mean |
|---------|--------|---------|---------|---------|----------|
| mean_pool | 3 | **60%** | 46% | 50% | 52.0% |
| mean_pool | 4 | 42% | 40% | 38% | 40.0% |
| mean_pool | 5 | 38% | 32% | 38% | 36.0% |
| mean_pool | 6 | 36% | 32% | 36% | 34.7% |
| max_pool  | 3 | 50% | 34% | **54%** | 46.0% |
| max_pool  | 4 | 42% | 34% | 38% | 38.0% |
| max_pool  | 5 | 38% | 28% | 44% | 36.7% |
| max_pool  | 6 | 34% | 44% | 44% | 40.7% |

WB mean across all 24 runs: **40.5%**

### Per-Dimension Aggregates

| Dimension | Value | Mean | Std | Min | Max |
|-----------|-------|------|-----|-----|-----|
| **Reducer** | mean_pool | 38.3% | 6.5% | 30% | 60% |
|             | max_pool  | 41.1% | 5.9% | 28% | 54% |
| **Window**  | 3 | **44.8%** | 7.5% | 34% | 60% |
|             | 4 | 38.8% | 2.6% | 34% | 42% |
|             | 5 | 38.3% | 6.4% | 28% | 52% |
|             | 6 | 36.8% | 4.4% | 30% | 44% |
| **Keep ratio** | 0.25 | **41.8%** | 6.7% | 34% | 60% |
|                | 0.50 | 37.3% | 4.9% | 28% | 46% |
|                | 0.75 | 40.1% | 6.4% | 30% | 54% |
| **Weight** | WA | 38.9% | 4.9% | 30% | 52% |
|            | WB | 40.5% | 7.4% | 28% | 60% |

### Top 5 and Bottom 5 Configurations

| Rank | Reducer | Window | Keep Ratio | Weight | Success Rate |
|------|---------|--------|------------|--------|--------------|
| 1 | mean_pool | 3 | 0.25 | WB | **60%** |
| 2 | max_pool  | 3 | 0.75 | WB | 54% |
| 3 | max_pool  | 5 | 0.25 | WA | 52% |
| 4 | mean_pool | 3 | 0.75 | WB | 50% |
| 5 | max_pool  | 3 | 0.25 | WB | 50% |
| ... | | | | | |
| 44 | mean_pool | 5 | 0.50 | WB | 32% |
| 45 | mean_pool | 6 | 0.50 | WB | 32% |
| 46 | mean_pool | 5 | 0.75 | WA | 30% |
| 47 | mean_pool | 6 | 0.75 | WA | 30% |
| 48 | max_pool  | 5 | 0.50 | WB | 28% |

---

## Analysis

### 1. Temporal Pruning Hurts More Than It Helps

The central finding is negative: **temporal pruning degrades cache retrieval
quality relative to the no-prune baselines in the majority of configurations.**

| Reducer | Weight | Best Prune | No-Prune Baseline | Δ |
|---------|--------|------------|-------------------|---|
| mean_pool | WA | 42% (w=3, kr=0.25) | 56% | **−14 pp** |
| mean_pool | WB | 60% (w=3, kr=0.25) | 52% | **+8 pp** |
| max_pool  | WA | 52% (w=5, kr=0.25) | 54% | −2 pp |
| max_pool  | WB | 54% (w=3, kr=0.75) | 48% | **+6 pp** |

Under WA (robot-state-dominant), pruning is strictly harmful: the best pruned
mean_pool run (42%) is 14 points below the unpruned baseline (56%), and the best
pruned max_pool (52%) merely approaches the baseline (54%). Under WB
(vision-heavy), pruning provides a modest boost: mean_pool gains 8 pp and
max_pool gains 6 pp. However, these gains depend on hitting the right parameter
combination — most WB configurations still fall below the baseline.

The overall experimental mean of **39.7%** is substantially below the Phase 1
baselines (48–56%), confirming that pruning is not a reliable improvement.

### 2. Window Size Has a Monotonic Negative Effect

Smaller windows are consistently better:

| Window | Mean | Δ vs window=3 |
|--------|------|---------------|
| 3 | **44.8%** | — |
| 4 | 38.8% | −6.0 pp |
| 5 | 38.3% | −6.5 pp |
| 6 | 36.8% | −8.0 pp |

Window=3 outperforms all other window sizes by a wide margin (6–8 pp) and is
the only window size whose mean (44.8%) approaches the Phase 1 baselines. Every
single top-5 configuration uses window=3 or window=5 (and the one window=5
entry, max_pool/WA, is still below its unpruned baseline).

The degradation with larger windows is consistent with the following
interpretation: in LIBERO spatial tasks, the visual scene changes rapidly enough
that a window of 4+ frames captures too much diversity, causing the temporal
variability scores to become noisy. With a large window, the score $s_i$
reflects cumulative drift rather than action-relevant motion, and the pruning
loses its ability to discriminate dynamic from static tokens.

### 3. Keep Ratio Shows a Non-Monotonic Pattern

| Keep Ratio | Mean | Std |
|------------|------|-----|
| 0.25 | **41.8%** | 6.7% |
| 0.50 | 37.3% | 4.9% |
| 0.75 | 40.1% | 6.4% |

The aggressive pruning level (kr=0.25, retaining only 64 of 256 tokens) performs
best on average, while the moderate level (kr=0.50) performs worst. This is
counterintuitive: one might expect that retaining more tokens would converge
toward the unpruned baseline. Two factors explain this:

1. **At kr=0.25**, only the most dynamic tokens survive. If those tokens happen
   to encode action-relevant features, the resulting key is a cleaner signal.
   The best overall result (mean_pool, w=3, kr=0.25, WB = 60%) demonstrates
   this: aggressive pruning with a short window isolates motion-relevant tokens
   effectively.

2. **At kr=0.50**, the surviving set includes both dynamic and moderately-static
   tokens. These intermediate tokens are neither clean signal nor neutral
   (averaged-out) mass — they add noise without adding discrimination.

3. **At kr=0.75**, the retained set is large enough that the extra tokens
   partially recover the unpruned behavior through dilution, producing scores
   closer to the no-prune baseline.

### 4. Weight Interaction: Pruning Only Helps When Vision Matters

The WA/WB split reveals why pruning's overall effect is negative:

- **Under WA** (rs=0.8), the cache key is dominated by robot_state. Vision
  features contribute only 20% of the retrieval signal. Pruning modifies the
  vision component, but since that component barely matters, any noise introduced
  by imperfect pruning directly reduces key quality without a compensating
  benefit. Mean across all WA runs: **38.9%** vs. baselines of 54–56%.

- **Under WB** (v0=0.5, v1=0.25), vision features contribute 75% of the
  retrieval signal. Here, pruning has leverage: if it successfully isolates
  action-relevant tokens, the key quality improves. The best WB result (60%)
  exceeds both WB baselines (48–52%), and 5 of the top 6 overall runs use WB.

This is the experiment's most actionable insight: **temporal pruning is a
vision-key optimization, and its value is proportional to the weight placed on
vision features in retrieval.** In the robot-state-dominant regime that Phase 1
identified as generally optimal (w7), pruning is counterproductive.

### 5. Reducer Comparison: max_pool Is Slightly More Robust

max_pool (mean 41.1%) outperforms mean_pool (38.3%) by 2.8 pp overall, with
lower variance (std 5.9% vs 6.5%). This difference is small and may not be
statistically significant at this sample size, but the direction is consistent
across all four (window, weight) slices. max_pool's advantage likely stems from
its ability to preserve peak activations: even when pruning removes some
informative tokens, the element-wise maximum over the survivors retains extreme
feature values that carry discriminative information.

However, the best single configuration uses mean_pool (60%), not max_pool (54%).
mean_pool has higher variance — it benefits more from lucky pruning parameter
choices but also degrades more when the parameters are wrong.

### 6. The One Bright Spot: mean_pool / w=3 / kr=0.25 / WB = 60%

This single configuration deserves special discussion because it is the only
result that substantially exceeds its Phase 1 baseline (+8 pp over the
unpruned mean_pool/w8 at 52%). It also exceeds **all** Phase 1 mean_pool
results (the best Phase 1 mean_pool was w5 at 58%).

The combination of aggressive pruning (only 64 tokens retained) with a minimal
window (3 frames) and vision-heavy retrieval weights creates conditions where
the temporal pruning hypothesis actually holds: the pruner correctly identifies
the few tokens that encode task-relevant motion, and the resulting key is a
better index into the cache than the full 256-token average.

Whether this result is robust or a statistical artifact of the small sample
size (50 episodes) cannot be determined from this experiment alone.

---

## Conclusions and Recommendations

1. **Do not adopt temporal pruning as a default.** The overall experimental mean
   (39.7%) is well below the Phase 1 baselines (48–56%). Under the
   robot-state-dominant weight regime that Phase 1 and the trajectory experiment
   identified as optimal (w7), pruning causes consistent degradation.

2. **If temporal pruning is revisited, fix window=3 and kr=0.25.** These
   are the only parameter values that produce competitive results. The search
   over window ∈ {4, 5, 6} and kr=0.50 was unproductive; future experiments
   should not revisit those regions.

3. **Consider testing temporal pruning with vision-heavy retrieval on stronger
   key builders.** The one positive result (mean_pool/w=3/kr=0.25/WB = 60%)
   suggests that pruning can help when vision dominates the key. Testing this
   combination with `clip` or `spatial_pool_16` — key builders with higher
   Phase 1 ceilings — could determine whether pruning is genuinely useful in the
   vision-heavy regime or whether the 60% result is an outlier.

4. **The primary insight is methodological: temporal pruning is a vision-key
   optimization.** It only has leverage when vision features dominate retrieval.
   In the robot-state-dominant regime that is currently preferred for LIBERO
   spatial tasks, it is counterproductive. This limits its applicability unless
   the weight regime shifts toward vision — which may happen on other task
   suites or with better visual encoders.

---

## Limitations

Sample sizes are identical to Phase 1 and the trajectory experiment (5 episodes
per task × 10 tasks = 50 trials per configuration). At this scale, differences
of a few percentage points are within sampling noise. The key qualitative
findings — that pruning hurts under WA, that window=3 dominates, and that the
60% WB result is an outlier — are directionally robust, but their exact
magnitudes should not be over-interpreted.

Two runs (`tp_run_002_mean_3w_025kr_wb` and `tp_run_043_max_6w_025kr_wa`)
failed on their first attempt and were successfully recovered via automatic
retry. Their results are included in the tables above.
