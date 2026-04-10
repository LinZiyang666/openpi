# Cache Experiment Combination Plan

> Status: In Progress
> Date: 2026-04-06

---

## Experiment Scope and Fixed Configuration

This round of experiments only covers `CP1`; `CP2/CP3` are not considered. `Gate` and `Judge` are not experimental variables and are uniformly fixed as:

- `gate.type = always_search`
- `judge.type = always_hit`

Therefore, this round of experiments is truly comparing two things:

1. Dimensionality reduction method for cache keys
2. Fusion method for cross-modal retrieval results

Judge is used here only to pass through top-1 retrieval results and does not perform threshold determination; threshold calibration is deferred to a separate future experiment.

---

## Experiment Objectives

In the `CP1` scenario, conduct combinatorial experiments on cache key dimensionality reduction methods and cross-modal fusion methods to evaluate the impact of different approaches on retrieval quality, cache hit rate, and final accuracy.

---

## Dimensionality Reduction Methods

| Method | Description | Applicable fields | Post-reduction vision dims | Compression ratio |
|--------|-------------|-------------------|---------------------------|-------------------|
| **A: Mean Pooling** | Average over token dimension | `vision_0/1`, `prompt_emb` | 2,048 | 256x |
| **B1: Spatial Pooling (16x)** | 16x16 grid -> 4x4, preserves spatial structure | `vision_0/1` only | 16 x 2,048 = 32,768 | 16x |
| **B2: Spatial Pooling (64x)** | 16x16 grid -> 2x2, preserves spatial structure | `vision_0/1` only | 4 x 2,048 = 8,192 | 64x |
| **C: Max Pooling** | Per-dimension max over token dimension | `vision_0/1`, `prompt_emb` | 2,048 | 256x |

Notes:

- Original vision field dimensions: 256 tokens x 2,048 = 524,288
- LIBERO experiments use only `vision_0` and `vision_1`, not `vision_2`
- `B1/B2` apply only to `vision_0/1`; `prompt_emb` uses Mean Pool to 2,048 dims in these experiments
- `robot_state` is fixed at 32-dim raw vector, no dimensionality reduction

### Max Pooling Applicability Analysis

Max Pooling takes the strongest activation value per dimension, suitable for scenes with prominent key objects. Not suitable for tasks that rely on global structure or multi-object spatial relationships. For LIBERO-style multi-object manipulation tasks, false hits or missed hits may occur, so it is retained as a control group rather than the default preferred method.

---

## Two-Layer Retrieval Structure

The "two layers" here do not mean "Layer 1 retrieval + Layer 2 Judge", but rather:

1. `Layer 1: Field Similarity`
2. `Layer 2: Cross-Modal Fusion`

`Judge` is fixed to `always_hit` and does not participate in this round's method combinations.

### Layer 1: Field Similarity

All experiment cases use the same set of field-level similarity definitions:

| Field | Similarity / Distance |
|-------|-----------------------|
| `vision_0` | Cosine similarity |
| `vision_1` | Cosine similarity |
| `prompt_emb` | Cosine similarity |
| `robot_state` | L2 distance |

Notes:

- All experiments compute cosine similarity for `vision_0/1` and `prompt_emb`
- All experiments compute L2 distance for `robot_state`
- Since LIBERO does not have `vision_2`, this round of experiments does not include that field
- In other words, the field-level scoring method in Layer 1 is fixed; the experimental variable is not in this layer

### Layer 2: Cross-Modal Fusion

Layer 2 compares two cross-modal fusion strategies:

| Method | Description |
|--------|-------------|
| **a: Weighted RRF** | Each field is independently ranked, then fused using weighted Reciprocal Rank Fusion |
| **b: Weighted Score Sum** | Field scores are first mapped and normalized to a unified scale, then combined via weighted sum |

Notes:

- `Weighted RRF` primarily fuses rank signals and is insensitive to the raw score scales of each field
- `Weighted Score Sum` cannot directly use raw cosine and raw L2; scores from each field must first be converted to the same semantic and numerical scale
- These two methods are the actual variable in the "second layer" of this round's experiments, not the Judge

### Normalization Definition for `Weighted Score Sum`

In this experiment, `Weighted Score Sum` is explicitly defined as:

`Score(x) = Sigma_f w_f * s_hat_f(x)`

Where:

- `w_f` is the field weight
- `s_hat_f(x)` is the normalized similarity score for field `f` on candidate sample `x`
- All `s_hat_f(x)` must fall within `[0, 1]`, with unified semantics of "higher = more similar"

Specific procedure:

1. Compute raw field scores
   - `vision_0` / `vision_1` / `prompt_emb`: use cosine similarity, denoted `s_cos`
   - `robot_state`: use L2 distance, denoted `d_rs`
2. Convert `robot_state` distance to similarity
   - `s_rs = exp(-d_rs / tau)`
   - `tau` is a temperature parameter controlling L2 distance decay rate; currently fixed at `tau = 0.334717`
   - This value is derived from offline statistics on `data/libero_spatial`, script is `exp/calibrate_robot_state_tau.py`
   - Sampling rules: only successful episodes are used; positive samples are same-episode steps with `delta_t in {1, 2}`; negative samples are cross-episode random steps from the same task + distant steps within the same episode
3. Normalize each field score independently to `[0, 1]`
   - For cosine fields, first unify direction: `s_cos_01 = (s_cos + 1) / 2`
   - Then apply per-field percentile normalization:
     `s_hat_f = clip((s_f - p5_f) / (p95_f - p5_f), 0, 1)`
   - Where `p5_f` and `p95_f` are the 5th / 95th percentiles for field `f` on the offline statistics set
4. Finally, compute the weighted sum
   - `Score(x) = w_v0 * s_hat_v0(x) + w_v1 * s_hat_v1(x) + w_prompt * s_hat_prompt(x) + w_rs * s_hat_rs(x)`

Additional constraints:

- `A-SUM / B1-SUM / B2-SUM / C-SUM` all refer to the above "normalized score sum", not raw score sum
- If percentile statistics are not yet prepared, `Weighted Score Sum` approaches must not be directly compared with `Weighted RRF` in a fair comparison
- In terms of implementation priority, `Weighted RRF` can be implemented first; `Weighted Score Sum` depends on additional offline statistics and calibration
- Whenever the dataset changes, or the `robot_state` preprocessing/sampling rules change, `exp/calibrate_robot_state_tau.py` must be re-run to compute a new `tau`

---

## All Combinations (8 total)

| # | Reduction | Applicable fields | Post-reduction vision dims | Layer 1: Field Similarity | Layer 2: Cross-Modal Fusion | Short name |
|---|-----------|-------------------|---------------------------|---------------------------|-----------------------------|------------|
| 1 | A: Mean Pool | All | 2,048 | `v0/v1/prompt=cos`, `robot_state=L2` | a: Weighted RRF | A-RRF |
| 2 | A: Mean Pool | All | 2,048 | `v0/v1/prompt=cos`, `robot_state=L2` | b: Weighted Score Sum | A-SUM |
| 3 | B1: Spatial Pool (16x) | Vision only | 32,768 | `v0/v1/prompt=cos`, `robot_state=L2` | a: Weighted RRF | B1-RRF |
| 4 | B1: Spatial Pool (16x) | Vision only | 32,768 | `v0/v1/prompt=cos`, `robot_state=L2` | b: Weighted Score Sum | B1-SUM |
| 5 | B2: Spatial Pool (64x) | Vision only | 8,192 | `v0/v1/prompt=cos`, `robot_state=L2` | a: Weighted RRF | B2-RRF |
| 6 | B2: Spatial Pool (64x) | Vision only | 8,192 | `v0/v1/prompt=cos`, `robot_state=L2` | b: Weighted Score Sum | B2-SUM |
| 7 | C: Max Pool | All | 2,048 | `v0/v1/prompt=cos`, `robot_state=L2` | a: Weighted RRF | C-RRF |
| 8 | C: Max Pool | All | 2,048 | `v0/v1/prompt=cos`, `robot_state=L2` | b: Weighted Score Sum | C-SUM |

---

## Weight Exploration Experiment Design

### Constraints

- Each test run takes 10 minutes, cannot be parallelized
- Experiment time budget: 1 day (24h)
- This round's goal is coarse screening of method combinations, not Judge threshold tuning
- Evaluation metrics still record `hit rate`, accuracy, and downstream task metrics, but these metrics are not controlled by a Judge threshold

### Input Fields and Priors

| Field | Prior importance | Notes |
|-------|-----------------|-------|
| `vision_0` | **Highest** | Main viewpoint, most information |
| `robot_state` | Medium | Directly related to actions |
| `vision_1` | Medium | Left wrist viewpoint, supplementary local details |
| `prompt_emb` | **Fixed at 0** | Does not participate in weight search this round |

### Phase 1: Full coarse search (64 runs, ~10.7h)

For all `8` "reduction x fusion" combinations, run `8` weight configurations each. No combination is eliminated early.

The weights here apply to Layer 2:

- For `Weighted RRF`, these are the fusion weights for each field's rank
- For `Weighted Score Sum`, these are the fusion weights for each field's score

`prompt_emb` is fixed at `0` throughout; only 3 weights are tuned (normalized to sum to `1`), coarse grid with step=`0.25`:

| # | `vision_0` | `vision_1` | `robot_state` |
|---|------------|------------|---------------|
| W1 | 1.0 | 0.0 | 0.0 |
| W2 | 0.75 | 0.25 | 0.0 |
| W3 | 0.75 | 0.0 | 0.25 |
| W4 | 0.5 | 0.25 | 0.25 |
| W5 | 0.5 | 0.5 | 0.0 |
| W6 | 0.5 | 0.0 | 0.5 |
| W7 | 0.25 | 0.5 | 0.25 |
| W8 | 0.25 | 0.25 | 0.5 |

`8` combinations x `8` weights = `64` runs.

### Phase 1.5: Fine-grained weight search for Phase 1 top 3 combinations (45 runs, ~7.5h)

Select the **top 3 combinations** from Phase 1 results, and do step=`0.1` fine search around the neighborhood of each combination's best weight from Phase 1.

Using the Phase 1 best weight as center, sample at step=`0.1` within a `+/-0.2` range, approximately `15` weight configurations per combination:

Example (assuming some combination's Phase 1 best is `W3: v0=0.75, v1=0.0, rs=0.25`):

| # | `vision_0` | `vision_1` | `robot_state` |
|---|------------|------------|---------------|
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

Note: Actual sampling points are dynamically generated based on each combination's best weight from Phase 1; the table above is only an example.

`15` weights x `3` combinations = `45` runs.

### Phase 2 (optional): Validate `prompt_emb` hypothesis (3 runs, ~30 min)

For the best combination + weights from Phase 1.5, add one group with `prompt_emb=0.1` as a control to confirm that `prompt_emb` is not useful.

### Time Summary

| Phase | Runs | Time |
|-------|------|------|
| Phase 1: Full coarse search | 64 | ~10.7h |
| Phase 1.5: Top 3 fine search | 45 | ~7.5h |
| Phase 2: Validate hypothesis | 3 | ~30 min |
| **Total** | **112** | **~18.7h** |

Remaining ~5h can be used for additional experiments or repeated validation.

---

## Module Boundaries to Implement This Round

To support the above experiments, subsequent implementation should focus on the following boundaries:

1. CP1-specific key builder / dimensionality reduction module
2. Field-level similarity computation module
   - `vision_0/1`, `prompt_emb`: cosine
   - `robot_state`: L2
3. Cross-modal fusion module
   - `weighted_rrf`
   - `weighted_score_sum`

Not in scope this round:

- `CP2/CP3`
- Gate strategy experiments
- Judge threshold experiments
- Replacing all experiment logic by relying on existing general-purpose Qdrant RRF logic
