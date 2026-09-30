# E1 — Kinematic stages: useful boundaries, unproven difficulty-to-compute mapping

## 1. Direction summary

**Recommend profiling two methods: soft allocation around gripper-event stages, and stage-contained extensions of Commit-Cache. Do not deploy the tested composite kinematic difficulty score as a call-placement rule.**

I built a library-only segmentation and difficulty annotation for all eight libraries, using successful demonstrations' proprioceptive states and executed action heads. It covers **2,005 successful demonstrations / 72,373 rows**, across 80 task–library cells. No serving method was implemented; the accompanying scripts are exploratory analysis.

The coarse structure transfers well: the two models agree on the modal gripper-transition count in **18/20 task pairs at size 50 and 20/20 at size 500**. A sparse geometric refinement yields medians of **16–17 segments on L10 and 7 on Spatial**. However, agreement of demonstrations is not a reliable measure of difficulty: the composite's hard/other under-motion risk ratio changes from **2.67 on GR00T L10-50 to 0.57 on π0.5 L10-50**. Tight boundary agreement alone has under-motion AUROC below .5 in **7/8 cells**.

Gripper events are more useful as *inspection locations*: their neighborhoods cover **21.2% of P3 cache anchors but 51.6% of under-motion onsets**. This does not identify failure causes. Among episodes with an onset, the first onset is event-adjacent in **70.9% of successful episodes versus 41.8% of failed episodes**. Normal contact/closing pauses are a major alternative explanation.

Read/checked: the R7 brief; library manifests and state/action/episode/step/success/next arrays; ledger §§5.1–5.3; R1 `FINDINGS.md`; R5 `ideation_A/REPORT.md`, `q1_commit/judge.py`, and `q3_callvalue/HANDBACK.md`; R6 P3 `SCHEMA_V2.md`; selected strict P3 and B-val tables; `closed_loop/README.md` and `closed_loop/blind.py`. These sources distinguish actual issued controls from proposed tails, and preserve R5's negative result on landmark call suppression.

### Relevant classical and demonstration-based methods

| Family / primary source | What it contributes | Transfer to R7 / limitation |
|---|---|---|
| Gripper events + stopped configurations: [PerAct's actual keypoint discovery code](https://raw.githubusercontent.com/peract/peract/main/helpers/demo_loading_utils.py) | Retains gripper changes, episode end, and stopped configurations with stable gripper state. | Good event vocabulary. Replace its absolute velocity tolerance and fixed stop buffer with library scales; do not import benchmark-specific constants. |
| Sparse waypoints: [Shi et al., Automatic Waypoint Extraction, CoRL 2023](https://proceedings.mlr.press/v229/shi23b.html) | Minimizes the waypoint set whose linear interpolation reconstructs a demonstration within an error tolerance. | Supplies geometric boundaries and an approximation certificate, not an SR or open-loop safety certificate. Our diagnostic uses time-linear error and mandatory event/slow knots, not an exact reproduction of AWE. |
| Bottleneck poses: [Johns, Coarse-to-Fine Imitation Learning](https://www.robot-learning.uk/coarse-to-fine-imitation-learning) | Reach an object-relative interaction pose, then replay a fine motion. | Motivates allocating observation near interaction entry and preserving its subsequent sequence. Absolute robot-pose agreement is weaker than object-relative alignment. No new pose-estimation model is proposed. |
| Multi-stage coarse/fine composition: [Learning Multi-Stage Tasks via Self-Replay](https://www.robot-learning.uk/self-replay) | Composes reaching and interaction phases for multiple stages. | Supports nested macro/micro stages. Its human-selected bottlenecks are not available here; our boundaries remain anonymous kinematic events. |
| DMPs and trajectory distributions: [Ijspeert et al., 2013](https://pubmed.ncbi.nlm.nih.gov/23148415/), [Paraschos et al., ProMPs](https://proceedings.neurips.cc/paper/2013/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) | DMPs represent attractor motions; ProMPs represent distributions over trajectories. | Borrow phase, endpoint, and dispersion concepts. Keep cached action chunks as the primitives; training another policy or replacing action synthesis is unnecessary. Low demonstrated variance does not establish sensitivity to perturbations. |
| Contact versus free motion: [Spector & Zacksenhouse, compliant movement primitives](https://arxiv.org/abs/2008.13223) | Separates free and contact-rich sub-tasks and adds compliance for contact. | Motivates a hybrid event/continuous representation. Our available state/action signals cannot reliably label contact, empty grasp, slip, or object-relative precision. We must not rename every slow segment “contact.” |

## 2. Frozen library construction and difficulty score

Implemented in [analyze_stages.py](analyze_stages.py); complete annotations are `labels_<cell>.npz`, `library_stages.csv`, `library_episodes.json`, and `calibration_<cell>.json`.

1. **Data selection.** Size 50 uses `current`; π0.5 size 500 uses `bpool_cs`; GR00T size 500 uses `bpool_all`. Fit only successful episodes. Preserve original row IDs; failed-library rows receive unknown stage/difficulty rather than “easy.” π0.5 Spatial-50 actually has 49 episodes. All 500 libraries contain failures.
2. **Representation.** Use `rs[:8]`; the first six coordinates define the pose embedding. Center each coordinate at its successful-library task median and scale by its 95th–5th percentile range; constant coordinates get unit scale. Pose distance is the six-coordinate Euclidean norm divided by √6. The prototype uses the stored normalized pose coordinates, not a physical SE(3) metric.
3. **Gripper events / macro stages.** Reduce each recorded executed head to its median gripper command. The manifest supplies the valid gripper channel and five-control recording stride. Fit deterministic two-means to successful-library commands; the midpoint separates modes. Remove isolated one-row reversals with a three-row majority. A change starts a macro stage. Mode polarity is learned separately; π0.5 and GR00T have opposite command conventions. No task text is inspected.
4. **Slow-motion knots.** Compute successive pose-embedding displacements. In each contiguous run below the task's successful-library first-quartile speed, retain its minimum. This is a low-speed proxy, not an assertion of zero physical velocity. Include episode endpoints and gripper changes as mandatory knots.
5. **Geometric refinement.** Between mandatory knots, dynamic programming finds the fewest time-linear segments with maximum pose-embedding error ≤ ε. Set ε to the median successful-library one-row displacement: one recording interval's motion resolution. All reconstructed paths satisfy this bound. Also report ε/2 and 2ε sensitivity, without selecting a winner on test outcomes.
6. **Boundary agreement.** Compare an endpoint with *other* successful episodes of the same task, initial command mode, and total transition count. Align the event ordinal and normalized arc length within that macro stage. Let σ be the median endpoint distance to interpolated peer poses and L the current macro-stage path length. Define precision demand `T = L/(σ + ε)`. This is a hypothesis about tightness, not a learned failure probability. Record peer support explicitly.
7. **Composite difficulty.** For a segment of duration Δ, `D = [rank(1/Δ) + rank(T) + E]/3`, where E says its endpoint lies within one stored row of a gripper event. Ranks are over the task's successful-library segments. Missing precision support gets the neutral rank .5 and an unknown-support flag. Define diagnostic “hard” by the successful-library row-weighted upper quartile of D. Also retain the simpler row-level event flag G: within one stored row of an event.

These are **general statistical rules**, not thresholds in LIBERO units: robust quantile scaling, quartile slow motion, median-resolution reconstruction, ranks, and one-sample neighborhoods. Equal feature weights are the symmetric baseline; their failure below argues against tuning weights on these test traces. The ±one-row event neighborhood is tied to recording resolution. The motion diagnostics below use fixed nominal 5% tails as descriptive reference rules, not safety guarantees.

For another robot, the adapter supplies valid pose/gripper dimensions, recording timestamps/stride, action horizon and normalization. Decode rotations and use geodesic orientation distance when the representation wraps; a parallel jaw, suction gripper, or continuous gripper requires the corresponding mode/event adapter. Refit scales, events, segment tolerances and reference distributions from that robot's library. If two command modes are not supported, omit that event channel and mark its signal unavailable. These adapter cases were not tested here.

### Library results

| Cell | Successful demos / nominal size | Median gripper events | Median micro segments | Median rows / retained knot | Mean task modal-event share | Rows without a peer |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 L10-50 | 50/50 | 3 | 16 | 3.10 | .940 | 2.1% |
| π0.5 L10-500 | 436/500 | 3 | 16 | 3.06 | .883 | 2.5% |
| π0.5 Sp-50 | 49/50 | 1 | 7 | 2.57 | .855 | 11.0% |
| π0.5 Sp-500 | 487/500 | 1 | 7 | 2.57 | .853 | 1.7% |
| GR00T L10-50 | 50/50 | 3 | 17 | 3.06 | .840 | 7.6% |
| GR00T L10-500 | 427/500 | 3 | 16 | 3.06 | .819 | 2.3% |
| GR00T Sp-50 | 50/50 | 1 | 7 | 2.56 | .860 | 1.5% |
| GR00T Sp-500 | 456/500 | 1 | 7 | 2.57 | .872 | 1.4% |

Source: [library_summary.csv](library_summary.csv), [library_tasks.csv](library_tasks.csv), [cross_model_consistency.csv](cross_model_consistency.csv).

The typical macro counts are therefore four and two, not a universally observed approach/grasp/transport/place/retreat sequence. **84.0–88.9% of successful Spatial demos finish in the opposite command mode from their start**, consistent with the R5 warning that success need not await release. A missing release is not a failure label.

Cross-model median absolute differences in per-task micro-stage counts are **1 at size 50 and 0 at size 500**. This is count consistency, not validated semantic alignment. Geometric granularity remains sensitive to ε: halving it changes L10 medians to **22–24**, doubling it to **12–12.5**; Spatial becomes **9–10.5** and **5–6**. Keep the robust event hierarchy separate from this resolution-dependent refinement.

## 3. Closed-loop and control-telemetry evidence

### Cohort and diagnostic definition

`check_telemetry.py` reads accepted strict tables under `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/<cell>/` and `r06_c_cal/tables/<cell>/`.
P3 contributes **60 A episodes per cell**: ten tasks × two inits × three replicates, **480 total**, with **391 successes / 89 failures and 10,691 anchors**. An additional 480 P10 episodes provide a policy comparison. B-val contributes **80 non-test A episodes / 1,817 anchors**, with 63 successes. Results were not used to fit segmentation or difficulty.

The primary online-stage approximation is nearest successful-library state in the eight-coordinate normalized proprioceptive space. It uses no image or shadow-policy feature. Reusing actual retrieval's top-1 stage is a separate sensitivity comparison; its binary hard labels agree with proprioceptive assignment on **75.0–86.1%** of known-label P3 A anchors. Neither assignment is ground truth. A causal sequence tracker is still needed before deployment.

**Under-motion:** compare observed position displacement over a complete ten-control commitment with the displacement expected from the matched library row over two recorded intervals. The cutoff is the task's 5th percentile of observed/expected displacement ratios when library rows match a nearest row from another successful episode. Position scales come from the same library. Exclude unobserved successors and nonconsecutive anchors. This is source-episode exclusion with frozen library normalization, not fully refitted LOEO or a conformal guarantee.

**Persistence:** replay those reference low-motion indicators at anchor stride two, in both parities. For each demo take its maximum consecutive low-motion run; use the task's empirical 95th percentile, rounded upward, as the run limit. Live persistent under-motion exceeds that limit. Limits range **1–10 anchors**, median **2**; they are library-derived, not a hand-set waiting time.

P3 anchors carry actual client after-head/after-commit robot states. I checked **19,397 A/P10 complete-commit successor pairs** against the next anchor's raw position: maximum discrepancy **6.13×10⁻⁸** in raw position coordinates. Thus the displacement analysis follows issued controls, not unexecuted candidate actions. This reads strict control summaries in `anchors.csv`; it does not reconstruct every intermediate physics substep. See [control_successor_audit.csv](control_successor_audit.csv).

### Does “hard” locate trouble?

| Cell | A successes / 60 | Hard / other under-motion RR [95% cluster interval] | Event-near / other RR | Event-near share of onset runs | Persistent episodes (failed) |
|---|---:|---:|---:|---:|---:|
| π0.5 L10-50 | 43 | .57 [.22, 1.32] | 2.45 | 31.1% | 7 (5) |
| π0.5 L10-500 | 54 | .63 [.19, 2.61] | 1.45 | 53.5% | 6 (6) |
| π0.5 Sp-50 | 44 | 2.08 [.70, 7.33] | 8.52 | 65.4% | 1 (1) |
| π0.5 Sp-500 | 55 | 1.25 [.38, 13.26] | 9.13 | 75.0% | 2 (2) |
| GR00T L10-50 | 34 | 2.67 [1.25, 6.09] | 7.20 | 53.0% | 11 (10) |
| GR00T L10-500 | 48 | .61 [.23, 1.28] | 1.64 | 40.7% | 8 (3) |
| GR00T Sp-50 | 55 | 1.09 [.37, 3.43] | 13.00 | 80.0% | 5 (4) |
| GR00T Sp-500 | 58 | 1.09 [.29, 3.79] | 17.60 | 76.2% | 0 (0) |

Source: [stage_outcome_evidence.json](stage_outcome_evidence.json), [telemetry_summary.csv](telemetry_summary.csv), `task_stalls_<cell>.csv`. Intervals use 2,000 bootstrap draws of task/init clusters, keeping all replicates together; they are marginal exploratory intervals, with no multiplicity correction. Ratios describe exposure on these paths, not effects of interventions. Very wide intervals and two inits per task limit generality.

The composite is higher in failed than successful episodes' average hard-stage occupancy in **only 1/8 cells** (GR00T L10-50). Across all cache episodes, persistent under-motion occurs in **40 episodes, 31 of which fail**; it covers only **31/89 failures**. Only **27.5% of persistent anchors are hard**, versus **29.4% hard exposure overall**. Thus the composite fails even as a general locator of persistent stalls. It should remain an ablation, not R7's safety signal.

The simpler event indicator has under-motion AUROC **.533–.815**, versus **.381–.726** for the composite; tightness alone is **.287–.645**, below chance in seven cells. Event-near risk ratios exceed one in all eight cells, but this may be legitimate closing/settling. For example, π0.5 L10-50 has hard/other RR **3.33 under P10 versus .57 under cache**. Controller-induced visitation changes the association.

B-val likewise puts **50.0% of onset runs near events at 19.6% event exposure**, but has only one episode per task/cell. Its composite ratios do not fix the cross-cell inconsistency. No test-derived task exceptions or score-weight tuning are recommended.

![Segmentation example and uncertainty of the composite stall association](evidence.png)

### Reproduction and CPU cost

From `/home/weiland/projects/openpi`:

```bash
taskset -c 22-23,66-67 bash exp/offline_search/rounds/r07/ideation/E1_kinematic_stages/run_analysis.sh
```

The wrapper runs each Python command with the brief's OMP/BLAS=1, CUDA disabled, no bytecode, and `PYTHONPATH=.:src`; commands are sequential. Intermediate selected-column table caches live in `/tmp/r7_E1_kinematic_stages/`. To reread changed source tables, remove only the corresponding named `anchors_<cohort>_<cell>.pkl` cache files before rerunning. Results and analysis scripts live here.

Segmentation fitting took **0.5–23.7 seconds per cell**. Individual nearest-neighbor lookups were approximately **.04 ms median and below .10 ms p95** on exact library-state probes; this is not a benchmark of the full online hook or difficult off-library queries. Stored compressed diagnostic labels occupy **24–463 KB/cell**. A compact serving annotation could use about 24 bytes/row (stage/phase/boundary IDs, score and support), under .72 MB at 30k rows, plus any lookup index. This is a proposed layout, not a measured implementation. See [cpu_cost.csv](cpu_cost.csv).

## 4. Method proposals

### P1 — Soft event-stage allocation with committed micro-sequences

**Hypothesis.** The useful classical distinction is a motion stage approaching a gripper event versus its interior. An event is a reason to consider observing/calling, while a chosen grasp/place sequence deserves continuity. Event-centered allocation may beat uniform allocation without needing the failed precision composite.

**Mechanism.** Use gripper-event macro stages; retain geometric subsegments as descriptive context. G marks the library event neighborhood. At vision anchors, use the current retrieval's distribution over stage labels plus a causal proprioceptive phase tracker. Unknown or conflicting stages retain uniform allocation. Do not infer stage from elapsed percentage of a test episode or its eventual length.

Atomic lever: **fewer calls** on stage interiors, shifting the same call budget toward event neighborhoods. Keep both cameras and existing ten-control commitments in the first comparison. This isolates stage-level allocation from camera selection or execution-horizon changes. The stage label and calibrated state deviation/stall status are signals; the budget knob remains the requested target IR.

A concrete soft baseline: estimate each task's event-neighborhood occupancy h from successful library demos, averaging episodes equally. Give event neighborhoods weight `1/h`, interiors weight 1; solve `p = min(1, λw)` for the requested call share under library occupancy. If there are no events, use weight 1 everywhere. This inverse-frequency rule retains nonzero interior calls, unlike event-only routing. λ is budget calibration, not a fitted test-success parameter. Closed-loop budget accounting must correct for changed occupancy and actual executed controls; it cannot assume the library occupancy remains exact. Cross-link: allocation-policy explorer owns that accounting.

Choose any event-associated call before executing its new chunk, then preserve the supported ten-control plan. Do not repeatedly restart a closing/release sequence merely because it remains near the boundary. Existing validated escape/stall logic can still request a fresh look. Profile a stage-entry inspection before adding any extra midpoint vision. G is **inspection priority**, not proof that the interior is safe to skip.

**Generality/calibration.** Uses library gripper commands, state and existing retrieval metadata. The event neighborhood, mode threshold and h come from the recording resolution and library. Task ID only selects the corresponding library, as A already does; no task-name/suite condition is introduced. A robot without discrete gripper events falls back to uniform until another library-supported event channel exists.

**Expected frontier effect.** With the same actual call and look shares, `ΔIR = c_v Δv + c_m Δm = 0`; the question is SR gain from timing. There is no defensible numerical SR forecast. The table below gives cell-specific priorities, not fitted cell-specific controllers. R5's unsupported landmark suppression and R6's random-placement result make the prior cautious.

**Cost tier / hook sketch.** T1 CPU metadata. `fit` stores stage annotations and occupancy; `reset` clears the per-connection tracker and commitment identity. `query` aggregates stage mass over the existing candidates, records ambiguity and deviation, and emits a proposed call weight. The judge combines this with the budget controller and existing guard verdict. `blind_step` / `policy_tail_step` keep the current supported source/chunk provenance; the stage label alone must not invalidate a good tail. Target added CPU below the existing 1–2 ms retrieval cost; only the approximately .04 ms nearest-state component is measured here.

**Kill criteria.** Drop event-based call weighting if a preregistered, equal-IR comparison against uniform gives no supported SR improvement across held-out tasks; do not rescue it by naming successful LIBERO cells. Drop deterministic stage gates if causal phase tracking cannot resolve repeated poses or unseen event sequences on held-out library episodes. The tested composite D has already failed the preliminary cross-cell consistency gate; do not make it the default allocation signal. Keep segmentation as telemetry even if call weighting dies.

### P2 — Extend execution only inside a supported kinematic segment

**Hypothesis.** Sparse geometric segments identify stretches where a little more execution can reuse a coherent plan. Their best immediate use may be to bound **look less**, not predict the value of a policy call.

**Mechanism.** Retain A's ten-control baseline. Consider one additional five-control block only when the represented trajectory has at least three recording intervals to the next geometric/slow/event boundary, all candidate continuations remain in that segment, and a current state/deviation check allows continuation. Ambiguous stage assignment, missing successor, or a boundary in the extension returns `LookReason` and the ordinary anchor path. This is an eligibility screen, not a safety certificate.

For GR00T first use the original chunk's controls 10:15, within H=16. For π0.5, a third block requires a true successor bridge: after ten controls advance original candidate rows by **`next(next(row))`**, then use their first five actions with coherent source/weight provenance. Validate same task/episode and exact step increments. Never clamp a terminal row or mistake `next(row)` for the ten-control successor. Do not shorten existing ten-control sequences just to hit a newly extracted micro boundary.

Phase compatibility is discrete, not a tuned distance. Calibrate continuous action-join and state-tracking diagnostics on successful library continuations using source-episode exclusion and empirical reference ranks. An initial nominal 5% alert-tail rule is a documented screening choice, not an SR guarantee; profile alert/coverage sensitivity before selecting a controller. Keep policy calls and their tails on their current contract while isolating cache extension.

**Generality/calibration.** Inputs are executed controls, current proprioception, library next links, action horizon and stage annotations. Another robot replaces the five-control stride and H with its interface contract, and refits all motion/join scales. No waypoint controller, DMP policy, encoder or learned model is added. Existing library action chunks remain the executed primitives.

**Expected frontier effect.** A necessary geometric screen retains **34.9–44.9% of successful-library rows for 15 controls**, versus **14.9–25.5% for 20 controls**; see [stage_eligibility.csv](stage_eligibility.csv). These are row-occupancy opportunities, not independent anchor opportunities or safe-skip rates. Actual multi-candidate and deviation gates can reduce them sharply.

For pure cache, if fraction f of actual ten-control blocks extends to fifteen, ideal vision cost changes from `c_v/2` to `c_v/(2+f)`. Substituting the *library* .349–.449 opportunities gives a purely illustrative **14.9–18.3% reduction in A's vision-only IR**. It is not a rollout forecast. At f=1 the ceiling is one third. For mixed controllers, recompute both call and vision shares over actual controls; do not bank unchanged-call assumptions.

The π0.5 bridge deserves a lower prior than GR00T's native tail. R5 `ideation_A/REPORT.md` found continuous tail/next-head RMS differences **.2527/.2852 on π0.5 L10 50/500** and an older phase-B2 result **.700 versus .806** for own-chunk tails at size 50. Thus geometry must earn its incremental protection over prior bridge screens. The R7 brief's “untested” refers to this idea on today's A, not an absence of adverse historical stitching evidence.

**Cost tier / hook sketch.** T1. `query` saves original row IDs, weights, stage endpoints, served chunk and commitment ID. `blind_step` checks actual completed controls and live state, then either serves the supported same-chunk tail / validated successor head or requests vision. `policy_tail_step` remains source-correct after a MISS. Store immutable fit arrays separately from per-connection cursors. The extra arithmetic is a small candidate/coordinate loop; added full-hook CPU is unmeasured and must be profiled, not inferred from the nearest-neighbor microbenchmark.

**Kill criteria.** Drop extension if the new stage screen does not improve the fidelity/coverage tradeoff over the existing continuation screen on held-out recordings; if savings disappear after actual gate/escape costs; or if randomized, matched-IR execution loses SR beyond the owner's accepted noninferiority margin. Never infer that margin from these test results. Any repetition of elapsed controls, endpoint clamping, cross-source tail, or planned-versus-issued confusion blocks an implementation before rollouts.

### Expected effect by cell — hypotheses, not measured improvements

| Cell | P1 event-stage call allocation | P2 stage-contained extension |
|---|---|---|
| π0.5 L10-50 | Event association 2.45×, but composite reverses; compare softly against uniform, no SR gain assumed. | 44.9% geometric opportunity; sparse peers and adverse historical stitching make this lowest-confidence π0.5 candidate. |
| π0.5 L10-500 | Event association 1.45× with broad uncertainty; benefit may be zero. | 44.4% opportunity, better peer support; plausible leftward IR movement if bridges preserve SR. |
| π0.5 Sp-50 | 8.52× event association largely includes closing pauses; high priority for checkpoint diagnostics, not automatic calls. | 36.8% opportunity; 11.0% of library rows lack a matching peer, so abstention may dominate. |
| π0.5 Sp-500 | 9.13× association but only five pilot failures; little evidence of extra-call value. | 35.9% opportunity; favorable cell to assess selective vision saving, conditional on successful bridge profiling. |
| GR00T L10-50 | Strongest composite association, but only six of ten tasks have positive hard-minus-other rate; avoid a task-specific rule. | 43.7% opportunity; native H16 extension removes the bridge confound, although SR risk remains. |
| GR00T L10-500 | Composite reverses, event ratio only 1.64×; no confident SR prediction. | 43.2% opportunity with native tail and more support; prioritize horizon comparison. |
| GR00T Sp-50 | 13.00× event association is not failure specificity; preserve calls outside events. | 34.9% opportunity; expected IR reduction is modest and requires episode-level verification. |
| GR00T Sp-500 | Only two pilot failures, no persistent stalls; do not spend more merely because an event exists. | 36.0% opportunity; prioritize preserving current high cache SR while lowering vision cost. |

These priorities do not authorize different task-name thresholds. Both proposals use the same library-derived rules across all eight cells. Neither justifies **look half** from proprioceptive geometry alone; camera sufficiency needs the separate lever/safety exploration.

## 5. Profile tools wanted

### Top request: `stage_trace` — stage transfer and event/persistence audit

- **Question:** Are the proposed stage boundaries recognizable causally, and does an event window contain intentional settling, a missed interaction, or a persistent stall? Is a low-difficulty stage actually distinguishable from unknown phase?
- **Inputs:** Frozen segmentation/fit hashes; library state/action/next arrays; existing P3 and B-val decisions, anchors, `action_steps`, `controls`, accepted outcomes and input archives. Use actual issued action heads and gripper aperture, not candidate tails. Privileged contact/object data, if inspected, are evaluation annotations only and never online features.
- **Outputs:** Per-episode timeline of gripper command events, observed aperture response, stage posterior/ambiguity, waypoint boundaries, deviation, phase advance, measured control speed, persistent stalls and terminal outcome. Compare static nearest state, current retrieval labels, and causal sequence alignment. Report event localization lag, backwards phase jumps, unmatched branches, episode-level failure recall/false alerts, and task/init-cluster uncertainty. Include ε sensitivity and source-episode-held-out alignment with fully refitted scalers as a stricter check.
- **Decision:** Whether P1 has a trustworthy observable stage signal and whether P2 may use a segment endpoint as an execution limit. It can kill semantic “hard stage” claims before a rollout.
- **New telemetry:** None for this offline audit. To profile a new implementation, log stage/branch IDs, candidate stage mass, boundary distance in *controls*, unknown reason, fit hash, tracker reset reason and hook timing alongside existing P3 identities.

### `stage_span_opportunity` — what can safely reach the experiment stage?

- **Question:** How much additional look-less opportunity survives geometry, source continuity, event exclusion and deviation checks, beyond the old bridge screen?
- **Inputs:** Existing P3 input archives with full candidate/executed/shadow chunks; state and issued-control summaries; original candidate weights; frozen stage labels and library next links.
- **Outputs:** Candidate-by-candidate 10→15→20 control lineage, next-boundary limits, join jumps, gripper minority/conflict, state divergence, rejection reasons, eligibility by task/stage and IR accounting under explicit unchanged-path assumptions. Separate GR00T's native tail from π0.5's successor bridge. Show baseline-screen versus added-stage-screen coverage/fidelity; benchmark fit memory and p50/p95/p99 CPU on actual off-library inputs.
- **Decision:** Implement P2, restrict its first experiment to native tails, or drop it if the stage screen adds no useful selectivity. Shadows measure same-observation disagreement, not recovery or changed-path SR.
- **New telemetry:** None for opportunity profiling. A P2 rollout needs actual completed-control counts, original anchor/chunk/row provenance, successor chain, remaining supported horizon, stage limit, and each `LookReason`.

### `stage_value_join` — cross-link to empirical-compute explorer

- **Question:** Does a call near a frozen event stage improve the episode outcome more than a call elsewhere, under the same continuation and total cost?
- **Inputs:** Frozen pre-treatment stage features joined to existing randomized uniform/dose/factorial recordings, accepted attempts and actual assignment propensities. Prefer strata based on information available before the assigned call; do not condition on realized final budget, downstream stage attainment, or eventual failure.
- **Outputs:** Supported stage-stratum treatment effects and uncertainty, honest unsupported cells, duration/hold package distinctions, and equal-budget uniform-versus-stage policy comparisons under valid overlap. Include all later costs and episode outcomes; short-horizon action disagreement is not the benefit target.
- **Decision:** Whether P1's inverse-frequency baseline merits a prospective experiment. The empirical-compute/allocation explorers own the causal estimator and budget policy; E1 supplies frozen labels and their uncertainty.
- **New telemetry:** Existing randomized recordings may suffice. If they lack support, a new randomized profile must log pre-treatment stage features, probabilities/overrides, intended and actual commitment, and future-controller package. An all-shadow A recording cannot supply this causal evidence.

## 6. Risks and open questions

- **Task progress is not arm progress.** Empty grasp, dropped object and contact failure can coexist with normal motion. Persistent under-motion misses 58/89 observed cache failures; even stage-correct replay is not evidence that the manipulated object is correct.
- **Difficulty is not demonstrated agreement.** Low variance can mean fixed initialization or a stereotyped easy motion. High variance can mean alternative valid routes. Object-relative bottlenecks are latent; the tested precision ratio is a poor universal trouble signal.
- **Contact pauses versus stalls.** The event association is strongest for brief under-motion; persistent-event occupancy is dominated by GR00T L10-50. Evaluate onset episodes and matched task/init clusters, not only repeated timeout anchors.
- **Phase aliasing and repairs.** Repeated visits, failed grasp retries and changed event counts can shift ordinal labels. Do not force an unseen sequence into the closest canonical stage. Failed-library rows and unsupported branches need explicit abstention.
- **Sample size.** Size-50 often provides only 3–4 peers; Spatial-50 π0.5 has no precision peer on 11.0% of rows. Empirical percentiles there are coarse and offer no closed-loop safety coverage guarantee.
- **Coordinates and cadence.** Stored normalized orientation coordinates can wrap or clip; robust scaling is not SE(3) invariance. Coarser recording can miss brief gripper events or speed minima. Production needs a pose/time adapter and sampling-sensitivity profiling.
- **Resolution and control semantics.** More waypoints are not automatically harder. A linear reconstruction bound does not cover contact dynamics. A library successor is a newly replanned action sequence, not the old chunk's continuation.
- **Evidence scope.** These are exploratory library fits and small existing-recording analyses, with post hoc diagnostic comparisons. No new SR/IR frontier point, safe-skip certificate, causal failure attribution, or deployable serving implementation is claimed.
