# Frozen round-2 proposals

These proposals were developed independently. No other researcher's round-2 outputs were read. The machine-readable freeze is `FROZEN_CANDIDATES.json`; it records hashes of 28 fitted candidates. `confirmation_specs.json` emits 44 arms including controls. All new heads fit **inits 0–19 only**. Every emitted arm evaluates **20–29 only**, 100 task/init pairs. None uses a new task-indexed parameter, threshold, budget, router, or choice of library. Existing task-conditioned retrieval is held fixed.

Ranking considers plausible benefit, strength of offline evidence, and the cost of falsifying the proposal. **No candidate has a measured closed-loop SR.** The offline data do not identify numerical candidate SR forecasts, and the round-1 discrepancy makes an error-to-SR extrapolation particularly indefensible. The tables give empirical reference SR, expected IR, and explicit experimental targets instead of invented success forecasts.

## 1. Shared motion and conservative gripper correction, with recovery-path supervision

**Priority:** first experiment on the two sparse long-task cells. Expected benefit: improve SR at the cache's IR, with a plausible additional route out of wrong-gripper states. Confidence is high for improved local imitation, low for SR improvement, and very low for closing the entire policy gap.

**Frozen recipe:** one pooled head per model/suite/library, shared over ten tasks. Inputs: 128 existing PCA coordinates, eight state coordinates, 70 cached action coordinates, capped elapsed decision count, and 16 previous-fresh-observation features. No task/init identity enters the feature vector. Train on pure-cache, uniform-plus-stall, and (where available) independent-call paths, all inits 0–19. Balance each arm/episode equally. Ridge alpha=100; 768 random Fourier features; seed=20261002. Predict 60 motion residuals and ten gripper-sign scores. Apply half the motion residual. Change a gripper control only if its sign opposes the score and |score|≥.8; use ±1 for the new command. The score is **not a calibrated probability**. Preserve the ten-control commitment.

The three fixed variants form a factorial comparison:

- `_motion`: half motion correction, original gripper.
- `_joint`: half motion correction plus conservative gripper changes.
- `_grip`: gripper changes only, original motion.

All three use the same fitted head; no outcome-tuned strengths. Inits 20–29 cache-path gripper disagreement decreases 1.20 pp / 1.32 pp in π0.5 / GR00T long-50. There are 20 / 33 affected episodes out of 100; first proposed changes occur at median control 250 / 220. This is a **small, late intervention**, not evidence that gripper correction alone repairs all long failures. On uniform-call paths, π0.5 long-50 per-control change precision drops to 75.8%, although first-change precision is 96.2%. Keep the gripper-only control to expose compounding harmful overrides.

| Cell | Supplied round-1 discovery cache / old corrector / policy SR¹ | New joint IR expectation | New joint SR estimate | Eval motion error reduction: cache / uniform-call paths |
|---|---|---:|---|---:|
| π0.5 long-50 | .743 / .757 / .917 | ≈.076 | Unidentified | 37.5% / 21.9% |
| π0.5 long-500 | .853 / .853 / .917 | ≈.077 | Unidentified | 8.7% / 5.7% |
| π0.5 Spatial-50 | .820 / .933 / .993 | ≈.077 | Unidentified | 44.1% / 30.5% |
| π0.5 Spatial-500 | .967 / .983 / .993 | ≈.078 | Unidentified | 17.0% / 9.6% |
| GR00T long-50 | .583 / .637 / .837 | ≈.074 | Unidentified | 45.4% / 26.4% |
| GR00T long-500 | .810 / .820 / .837 | ≈.075 | Unidentified | 10.9% / 11.4% |
| GR00T Spatial-50 | .867 / .933 / .943 | ≈.075 | Unidentified | 37.5% / 25.9% |
| GR00T Spatial-500 | .940 / .953 / .943 | ≈.076 | Unidentified | 9.7% / 7.7% |

¹ Context from the owner's inits-0–29 table. The old corrector was trained on those same starts. Its closed-loop gain is **not disjoint-fit validation**, and these SR numbers are not measurements of the new shared head. The new offline evaluation uses only 20–29 and has different factual baseline SR, recorded in `DATA_ANALYSIS.md`.

**Exact first batch**, all on the same 100-pair manifest:

```text
r9r2_astra_pi05_l10_50_cache
r9r2_astra_pi05_l10_50_motion
r9r2_astra_pi05_l10_50_joint
r9r2_astra_pi05_l10_50_grip
r9r2_astra_pi05_l10_P10
```

**Second batch:** replace `pi05` with `groot` in those five names. This is fixed now, not contingent on a favorable π0.5 outcome. Run negative results to completion. The other six cells have the identical three candidate suffixes and matched `_cache` arms, but broad expansion is lower priority. Dense gripper corrections affect only 1–4 of 100 cache episodes, so do not expect them to produce a large SR gain.

**Experimental target:** +2 pp versus newly measured matched cache at equal IR, with no evidence of material harm, and isolate `_joint−_motion` and `_grip−_cache`. This is a target, not a forecast or a significance claim. A 100-pair test has limited resolution and power.

**Failure modes:** policy-label multimodality; errors becoming irreversible before the late trigger; opening a correct grasp; feedback-distribution shift; history features trained on uncorrected actions; fewer independent training starts than round 1. Model supervision uses 600 arm/episodes per sparse cell (400 dense), but only 200 unique task/init pairs per cell. Extra shadow labels are part of the data budget.

## 2. Distill a wrist-only correction head to halve perception cost

**Priority:** start on π0.5 Spatial-500; then Spatial-50. Long-task arms are emitted as stress tests, with low confidence: uncorrected wrist perception loses 10–13 pp on the long-task evaluation starts.

**Frozen recipe:** inherited wrist-every-ten controller: both cameras at episode start, wrist camera thereafter, ten-control commitment, no calls. One shared head uses only 64 wrist PCA coordinates, state, actual wrist-cache action, and elapsed count (143 inputs). No third-camera feature, cached third-camera embedding, teacher action, or simulator predicate is used online. Fit on 0–19 wrist-only live trajectories and wrist retrieval shadows on pure-cache trajectories. The first anchor uses full-camera retrieval, matching deployment. Same ridge/RFF recipe; apply half motion residual; **keep gripper unchanged**. No stage or task gate.

| π0.5 cell | Eval factual full cache SR @ IR | Eval factual wrist cache SR @ IR | Corrected wrist expected IR | Corrected wrist SR | Motion MSE vs wrist cache / full-camera cache at the same wrist-path states |
|---|---:|---:|---:|---|---:|
| long-50 | .730 @ .07645 | .630 @ .03370 | ≈.034 | Unidentified | −39.9% / −45.4% |
| long-500 | .860 @ .07669 | .730 @ .03387 | ≈.034 | Unidentified | −14.5% / −11.1% |
| Spatial-50 | .790 @ .07728 | .790 @ .03600 | ≈.036 | Unidentified | −43.4% / −39.9% |
| Spatial-500 | .960 @ .07817 | .970 @ .03737 | ≈.037 | Unidentified | −20.8% / −13.6% |

These cost estimates include the first full-camera anchor and terminal rounding on existing paths; new episode lengths can shift them. Relative to ordinary cache, potential IR saving is approximately .041–.043. GR00T is **not supported** by the existing single-camera encoder path; no GR00T estimate or arm is fabricated.

**Exact first batch:**

```text
r9r2_astra_pi05_spatial_500_cache
r9r2_astra_pi05_spatial_500_wrist_cache
r9r2_astra_pi05_spatial_500_wrist
r9r2_astra_pi05_spatial_P10
```

For the other three π0.5 cells, substitute the cell suffix in the first three names and use its suite's P10. Keep batch budgets separate from the long-task correction experiment.

**Experimental target:** maintain ordinary-cache SR at ≈.037 IR on Spatial-500, then test preservation of P10 SR. The existing wrist .970 on 100 starts does not establish either claim. Confidence: high for compute reduction, moderate for local motion improvement, low for preserved SR. First test the fixed half correction; no full-strength search on the same evaluation outcomes.

**Failure modes:** wrist cannot observe remote objects/destinations; local imitation can be accurate along a globally wrong trajectory; recovery changes what the wrist sees; inherited camera shadows are a second encoder execution with numerical differences. Training on wrist-only paths reduces, but does not remove, this distribution shift. This is stronger evidence than applying a full-camera head to wrist-path observations, because the new head actually excludes the third camera.

## 3. Shared motion-only correction as the conservative portability control

The `_motion` arms above are independently useful if changing the gripper harms SR. Pooled recovery-path supervision improves transfer compared with a pooled pure-cache-only fit; short temporal memory adds a smaller benefit. At the same half strength, uniform-call-path MSE reductions increase from 14.9% to 21.9% for π0.5 long-50, and 19.8% to 26.4% for GR00T long-50. The uniform-call trajectories used to evaluate the head have **different initial states** from its uniform-call training trajectories.

Expected IR and unidentified SR are the first table's values. Rank below the joint experiment for new-method upside; it is principally a task-agnostic replacement and a clean control for the mechanism, not proof of improvement over the old per-task head. Do not compare its 20–29 offline MSE directly to round 1's five-fold metrics as if they were matched.

## Directions rejected or deferred by this round

- **Learned call-value gating:** held-out first-entry effects have wide intervals and inconsistent signs. GR00T long-50 has a positive local probability derivative, but the first-entry interval crosses zero and GR00T Spatial's derivative is negative. These are exploratory estimands under the original randomized future controller. No deployment arm is promoted.
- **High retrieval distance / low observed movement:** call-value signs change across cells. A generic observation-risk score is not evidence that calling helps.
- **Goal-regression rescue as the main long-task fix:** only 7–11% of failed long episodes finish below their maximum achieved predicate count. Some have transient regressions, but completed-goal destruction is not the dominant terminal pattern in these samples.
- **More history by itself:** the 16 history inputs give modest improvements over mixed-path supervision alone. No basis yet for an expensive recurrent-controller project.
- **Task routing, task budgets, task strength, or task library selection:** not pursued or proposed.

## Confirmation and final claim protocol

The emitted manifest contains only tasks 0–9 × inits 20–29. Never replace it with a 300- or 500-pair manifest: the models have been fitted on 0–19. Use timan108, ports 23230–23233, WPS=8, one chain at a time; synchronization port 23196. Full launch commands are in `HANDBACK.md`.

For each fixed batch, collect all outcomes, paired candidate/cache and candidate/P10 counts, pooled cost/decision sums, actual camera counts, and end-to-end latency. Motion/gripper errors remain mechanism checks, never surrogate success endpoints. Selection on 20–29 makes this development evaluation, not a locked final test.

Coordinator-only locked confirmation: after the scheduled development batches, choose and freeze **one** final candidate per cell before the coordinator opens any new 30–49 outcomes. If testing the currently proposed primary family, that family is π0.5 long-50 joint, GR00T long-50 joint, and π0.5 Spatial-500 wrist (K=3). Compare paired P10 on identical topology. Use `tools/statistics.py:paired_ni`, α=.05/K across claims, margin=.02, and require the lower bound >−.02 and lower IR. Do not refit these artifacts on 20–29 before that confirmation unless a new version and evaluation plan are explicitly frozen. The 200 locked pairs may be insufficient; an inconclusive bound remains inconclusive.
