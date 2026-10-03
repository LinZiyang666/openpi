# Ranked proposals and exact confirmation plan

These are discovery proposals, not new closed-loop results. Rank weighs plausible compute/SR benefit against strength of evidence and cost to falsify. Absolute new-controller SR cannot be inferred from action shadows; tables explicitly separate empirical SR estimates from confirmation targets.

## 1. Relabel cache-visited states with shadow policy actions; serve a small half-residual head

**Expected gain × confidence:** highest new-method upside at unchanged encoder cost. Strong evidence of local action improvement across inits and controllers; low-to-moderate confidence of positive closed-loop SR. No claim of preserving P10 SR yet. Prefer this to further call-placement searches.

**Frozen implementation:** `methods.py:ResidualCache`, blend=.5; per-task ridge alpha=100, 384 fixed random features, 207 inputs; motion correction only; original gripper and ten-control commitment. All eight fitted artifacts are ready. The extra-shadow nearest-neighbor baseline recovers much of the sparse benefit, so the proposal is primarily better supervision, with a compact serving head.

| Cell suffix | Expected owner IR | Current cache SR | New SR estimate | Half-residual error reduction: cache / CU paths |
|---|---:|---:|---|---:|
| `pi05_l10_50` | ≈.076 | .717 | Unidentified | 45.3% / 24.2% |
| `pi05_l10_500` | ≈.077 | .857 | Unidentified | 18.5% / 11.5% |
| `pi05_spatial_50` | ≈.077 | .800 | Unidentified | 48.4% / 24.3% |
| `pi05_spatial_500` | ≈.078 | .967 | Unidentified | 16.3% / 14.3% |
| `groot_l10_50` | ≈.074 | .637 | Unidentified | 50.3% / 24.6% |
| `groot_l10_500` | ≈.074 | .827 | Unidentified | 12.8% / 10.2% |
| `groot_spatial_50` | ≈.075 | .880 | Unidentified | 37.2% / 20.6% |
| `groot_spatial_500` | ≈.075 | .967 | Unidentified | 12.0% / 8.8% |

The experimental target is **at least +2 pp SR versus matched pure cache at essentially equal owner IR**, followed by comparison with P10. This target is an adoption criterion, **not an SR forecast**. CPU query overhead and actual work counts must be reported separately.

**Failure modes:** stochastic/multimodal teacher means; changed contact dynamics and retrieval distribution; original wrong gripper decisions remain wrong; regression on discovery initial states may not generalize; a small model might imitate recovery motion without changing the contact state needed to recover. Applying the full correction, retuning it separately by cell, or appending a call gate is outside this frozen test.

**Data-budget disclosure:** every “library-50” residual arm additionally uses 300 discovery cache episodes and their shadow labels. It is not a library-50-only result. Locked inits 30–49 remain necessary for the coordinator's generalization check. New discovery closed loops visit new states but reuse training initial configurations.

**Exact arms:** for each of the eight suffixes above,

```text
r9_astra_<suffix>_residual_half
r9_astra_<suffix>_cache
```

Matched pure-policy arms are `r9_astra_{pi05,groot}_{l10,spatial}_P10`. First batch is the four π0.5 Spatial-50 arms listed below, including the library-only metric control. Expand to π0.5 long-50 and GR00T long-50 only after inspecting the first fixed comparison. Keep unfavorable outcomes.

## 2. Whole-episode routing, specifically π0.5 long-50

**Expected gain × confidence:** modest compute gain with a more defensible offline SR estimand; medium confidence of savings, low-to-moderate confidence of SR preservation. OOF estimate **.915 SR @ .380 IR**, versus matched P10 **.907 @ .504** (about 25% less normalized work). Refitting the learner in bootstrap resamples gives a ΔSR sensitivity interval **[−3.7,+1.8] pp**; no noninferiority claim.

Freeze policy probability `[1,1,0,0,1,1,1,1,.14953271028037407,1]` for tasks 0–9. Draw one hash coin at episode start and use the selected controller throughout. The hash uses seed 20261001, domain `R9/astra/episode/pi05_l10_50`, task, init and decision index **0**, independent of future outcomes. No switching halfway through the episode.

**Exact primary comparison:**

```text
r9_astra_pi05_l10_50_episode
r9_astra_pi05_l10_50_cache
r9_astra_pi05_l10_P10
```

All other cell lotteries are emitted as transparent comparators, not equally recommended proposals. Their OOF estimates are:

| Cell | Expected SR @ IR from OOF episode replay | Disposition |
|---|---:|---|
| π0.5 long-500 | .876 @ .206 | Does not preserve matched P10 |
| π0.5 Spatial-50 | .980 @ .445 | Small saving, point −1 pp; insufficient confidence |
| π0.5 Spatial-500 | .973 @ .163 | Less attractive than existing frontier |
| GR00T long-50 | .891 @ .450 | Near reference, no compelling frontier improvement |
| GR00T long-500 | .875 @ .269 | Does not preserve matched P10 |
| GR00T Spatial-50 | .905 @ .234 | Reject for SR preservation |
| GR00T Spatial-500 | .967 @ .075 | Reduces to always-cache |

**Failure modes:** thirty examples per task cannot reliably select among many controllers; task-specific policy failures can be seed noise; train-on-all replay is optimistic; bootstrap model-selection instability is material. A new topology can change the relative strengths of both constituents. The full-data frozen rule's replay (.914 @ .360) must not be marketed as its test result.

## 3. Library-level no-call gate for GR00T Spatial-500

**Expected gain × confidence:** high confidence of lower work; moderate confidence that unnecessary calls should stay off, based on repeated prior rounds and this discovery sample. This is a deployment correction supported by existing evidence, not a novel algorithm.

Discovery estimates: pure cache **.9667 @ .07545**, matched P10 **.9367 @ .51119**. Do not borrow the optimistic point estimate as proof: paired counts 18 wins / 9 losses give CP lower bound −2.024 pp nominal, −3.911 pp for an eight-cell family. Neither passes the two-point bar.

**Exact confirmation:** `r9_astra_groot_spatial_500_cache` versus `r9_astra_groot_spatial_P10`. Keep the cache library and ten-control commitment unchanged. Existing CU/B results can inform a secondary comparison; a new B arm was not emitted because the main question is preservation of the matched P10 reference.

**Failure modes:** camera/render/topology effects, task distribution shift, optimistic discovery SR, and insufficient independent pairs for simultaneous NI. Do not extrapolate this gate to sparse Spatial libraries or to all π0.5 dense tasks.

For π0.5 Spatial-500, the discovery-only cheap-camera evidence is weaker than the full-500 headline: wrist-every-ten is **.960 @ .0373**, stage-gated wrist **.973 @ .0582**, versus P10 .990. Neither is a demonstrated preserved-SR replacement. Their exact controls `r9_astra_pi05_spatial_500_W10` and `_SW` are emitted for confirmation, not promoted as a new frontier.

## 4. Fit retrieval to the ten controls that are actually committed

**Expected gain × confidence:** smaller but clean library-only gain; medium confidence of better local retrieval, low confidence of SR gain. Expected IR equals the pure-cache rows above. New SR remains unidentified. No extra policy labels or discovery outcomes enter fitting.

Frozen change: action-similarity supervision uses the first ten valid seven-channel actions, retaining the original PCA, per-channel library scales, k=16, kernel shape, candidates and early fit. Sparse MSE improvements are 17.4%, 11.8%, 8.9%, 4.4% in π0.5 long, π0.5 Spatial, GR00T long, GR00T Spatial. Dense gains are negligible, so only sparse candidates are emitted.

**Exact arms:** `r9_astra_{pi05,groot}_{l10,spatial}_50_metric10`, each versus its `_cache` control. The head's confidence scales are retained only for logging: the arm uses pure-cache/always-hit behavior. Do not reuse these uncalibrated confidence values in a call guard.

**Failure modes:** ten-step teacher labels favor commitment consistency but may weaken immediate control; gains concentrated in a task; near-tied neighbors and vision numeric differences can change trajectories. The π0.5 Spatial-50 variant is the strongest general sparse-task control because all ten task-level local errors improve.

## Rejected or deferred

- Top-1, medoid, or removing failed neighbors: adverse local evidence, no basis for an expensive broad confirmation sweep.
- Generic uncertainty-triggered calls: randomized IP local effects are inconsistent across cells; shadow disagreement is not a call-value label.
- More complex task menus: often large train/test optimism. Keep all sensitivity results, do not choose the largest discovery gain.
- Reducing commitment or unconstrained continuation: the proxy fails known cadence experiments; no new evidence overturns the earlier losses.
- Full compact-policy replacement or gripper distillation: potentially useful, but not justified by the current motion-only evidence. The proposed half correction is intentionally a narrower first test.

## Exact runner plan and statistical decision

Owned run root: `/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation`.
`confirmation_specs.json` is the emitter input; the run root contains emitted `arms.json`, YAMLs and matrices. Every arm has **the same explicit discovery300 manifest**, tasks 0–9 × inits 0–29. No 30–49 manifest was created.

First batch, already dependency-planned locally:

```bash
R9_RUN=/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation
R9_ARMS=(r9_astra_pi05_spatial_50_cache r9_astra_pi05_spatial_50_residual_half r9_astra_pi05_spatial_50_metric10 r9_astra_pi05_spatial_P10)
```

Local plan: 37 dependency files, approximately **4.374 GiB** total logical assets. This is not necessarily transferred bytes; matching existing remote assets are reused. Source deployment and actual sync have not run. See the complete safe commands in `HANDBACK.md`.

Use only **timan108**, **23230–23233**, WPS=8 (32 workers), one chain at a time. Four arms run sequentially under one chain. The sync utility additionally requires a local 23100–23197 port; the assignment did not grant one, so the coordinator must reserve it. Do not silently use another researcher's sync port.

Discovery analysis reports paired SR differences, task breakdown, pooled owner IR, encoder/policy counts and latency. The fixed first-batch adoption target is +2 pp versus cache without a material cost increase; it is not a formal final selection claim. Do not continue trying variants until a favorable p-value appears.

For the owner-level claim, freeze one final candidate per selected cell **before the coordinator opens inits 30–49**. Evaluate candidate and P10 on the same host/checkpoint/controller conditions and paired initial states. Use the existing conservative paired CP lower bound with family α=.05/K for the K preregistered cell claims, and require lower bound >−.02 plus lower IR. Report bootstrap intervals as secondary; do not substitute them when CP fails. Repeated policy seeds are clustered by task/init, not counted as new independent pairs.

The 200-pair locked subset may simply be too small to establish simultaneous two-point noninferiority. An inconclusive bound is an inconclusive result, even with a good point estimate. Further sample/seed collection, if needed, is a coordinator decision rather than permission to tune on the locked set.
