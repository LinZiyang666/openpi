# R7 E5 — Cross-domain ideas and generality

## 1. Direction summary

**Recommendation:** profile **M1, stage-limited self-triggered reuse**, first. It applies *look less* only while a library continuation remains inside a stage and observed proprioception remains compatible with that continuation. **M2, stage-consensus routing**, is a secondary, falsifiable proposal for *fewer calls*; do not select it just because it predicts action disagreement. Neither proposal has a measured SR gain.

I read the R7 brief first; then R1 FINDINGS, R5 Q3 HANDBACK, R6 ANALYSIS and Q1 FINAL, R6 ideation G REPORT, P3 SCHEMA_V2, stall INTERFACE, the C controller HANDBACK, the profile README, the plugin/BlindQueryView contracts, R4 blind code and owner cost table, and the RoboCasa input/output and episode adapters. I inspected P3 pilot schemas but computed new statistics **only on the eight non-test `r06_c_cal` A recordings and their deployed libraries**. [analyze.py](analyze.py), [evidence.json](evidence.json), and [anchor_features.csv](anchor_features.csv) retain the analysis and provenance. No implementation, servers, rollouts, GPU work, git changes, or external writes were performed.

Three findings change the framing:

- The 16 neighbours are **3.02–4.09 effective episodes** at library size 50 and **7.19–10.56** at size 500, by cell medians. Agreement is correlated evidence, not sixteen independent votes (§3).
- Long command-mode spans exist: median **45–55 controls** across the eight libraries. But their 10th percentiles are **1–4 controls**, and an unchanged closing command says nothing about whether an object is held. Semantic “transport” must not be inferred from this alone.
- Neighbour dispersion is already partly explored: R6 Q1 FINAL reports kernel dispersion rho **.434** [.388, .480] with same-observation cache–policy disagreement, versus **.468** for the selected LOEO residual in R6 ANALYSIS. The latter's call placement subsequently lost **0.15 pp** to uniform across eight cells. A useful uncertainty diagnostic is not thereby a call-value estimator.

### Ideas borrowed, and limits of the analogy

| Field | Useful idea for this problem | Judgment under the owner's constraints |
|---|---|---|
| LLM cascades / routing | Escalate from a cheap candidate when its acceptance conditions fail. FrugalGPT studies query-dependent combinations of models. | Our cheap candidate is the existing library; escalation is proprioception → own-policy vision → full policy. No small model, new encoder, or trained correctness classifier. Unlike isolated queries, a cheap action changes the next state. [FrugalGPT](https://arxiv.org/abs/2305.05176) |
| Speculative decoding | Draft a continuation, inspect it, limit how much is committed. | **No lossless robotics analogue here.** Text speculation verifies draft tokens with the target model before accepting them and can preserve its distribution. Executed motion cannot be rejected and undone; verifying every future observation would require an unavailable world model or the real future. Use the scheduling intuition, not the exactness claim. [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) |
| Early exit | Link local compute decisions to a whole-sequence quality criterion. | CALM explicitly addresses local confidence versus sequence-level consistency. For R7, evaluate whole-episode SR. Internal transformer exits require a suitable trained interface; few-step denoising is the shelved *cheaper call* lever. Neither is proposed. [CALM](https://arxiv.org/abs/2207.07061), [authors' training description](https://research.google/blog/accelerating-text-generation-with-confident-adaptive-language-modeling-calm/) |
| Event-triggered / self-triggered control | An event requests recomputation after measured departure; a self-trigger predicts the next review time in advance. | Combine a stage-end review deadline with cheap proprioceptive checks. Classical stability results do not transfer: we have no certified dynamics, Lyapunov function, or disturbance bound. This is an empirical scheduler. [Anta–Tabuada](https://arxiv.org/abs/1009.5208) |
| Anytime / receding-horizon control | Keep a usable plan; renew only a prefix; improve or replace it after fresh feedback. | Preserve A's useful committed micro-sequence. A cached suffix is available immediately, but there is no demonstrated progressively improving solver here. Background full-policy inference would consume compute even if discarded, so do not count latency hiding as IR savings. [Diehl et al.](https://cdn.syscop.de/publications/Diehl2005c.pdf) |
| Task-and-motion / coarse-to-fine planning | Refine the immediate subtask while retaining a coarser future plan. | Use stages as **review boundaries**, not a hand-written approach/grasp/place automaton. Object poses, symbolic predicates and an added planner are unnecessary. Gripper modes are only a minimal observable partition, not semantic preconditions. [Kaelbling–Lozano-Pérez, authors' institution](https://www.csail.mit.edu/news/csail-researchers-take-new-approach-robotic-planning) |

## 2. Method proposals

### Shared definitions and calibration contract

Let `R` be actual controls per decision, `H` the policy horizon, and `C0` the reference controller's commitment. In current A, `R=5`, `C0=10`; read these from the interface. Define time in actual controls and seconds, never request count alone. The 16-member kernel and its weights remain A's; changing the retriever is outside these proposals.

The minimal stage partition is each maximal run of one **declared gripper command mode** along the library's executed heads, plus episode ends. For the present approximately bimodal commands, the analysis uses the midpoint of the observed command extrema, without interpreting either sign as “closed.” A deployed adapter should supply discrete modes or validate a library-derived mode partition; for a continuously controlled hand without such a partition, use E1's frozen kinematic stages or leave stage adaptation inactive. A short mode run is not merged using a LIBERO-tuned debounce threshold. A stage shorter than `C0` is ineligible for extended commitment. Motion-defined boundaries from E1 can refine this partition, with the same scheduler and a separate ablation.

Stages are local intervals within each demonstration, **not** an ordinal grasp number shared across demonstrations. A library row supplies current mode, next mode and time to the next boundary. Online stage membership is aligned from the retrieved rows and executed gripper mode. Incompatible modes or unresolved alignment request review. No elapsed fraction of the *test episode's eventual duration*, task-name lookup, object pose, contact label or outcome enters a serving decision.

For a candidate prefix of `h` controls define:

`D(h)^2 = sum_i w_i ||(a_i[0:h] - sum_j w_j a_j[0:h]) / sigma_a||_F^2 / (h D_valid)`.

`P(h) = max_{t<h} [1 - max_mode sum_i w_i 1(mode_i(t)=mode)]` measures mode disagreement. Keep normalized time-to-boundary spread separately as an audit; do not confuse whole-episode progress dispersion with action risk. `N_episode = 1 / sum_e (sum_{i in e} w_i)^2` measures independent support concentration; it is not a success probability or a routing score.

Compute action/state scales from successful library episodes, with valid channels, gripper semantics and pose coordinates supplied by the manifest/adapter. Mask padding, **not** real but constant robot channels; a constant channel cannot be safely standardized and disables the affected deviation test until a scale is declared. Pose differences require an appropriate rotation difference, not subtraction of wrapped Euler angles. Calibration uses leave-one-episode-out retrieval, equal weight per source episode, frozen A representations and no test outcomes. This is neighbour exclusion, not a claim of fully refitted LOEO or conformal coverage.

For dimensional scores use their empirical library midranks (half weight for ties); omit a feature that is constant in calibration. Any rank cut is chosen by a single documented budget solve, not by suite. The budget target `rho` is an owner input. Candidate cut points are the finite set of calibration ranks; select the most conservative cut meeting the modeled IR target, breaking ties toward more observation. Report infeasible targets. Use at most the existing ten B-val recordings per cell to estimate occupancy/cadence, not fit task success labels. One recording per task cannot certify a task-specific failure probability. Actual SR is always evaluated separately.

### M1 — Stage-limited self-triggered reuse (priority)

**Hypothesis.** Some stage interiors permit a later vision review while retaining A's committed action sequence. At the last observed anchor, neighbourhood agreement bounds a *candidate* horizon; proprioceptive deviation can revoke its extension. This targets observation scheduling rather than predicting which MISS rescues a failure.

**Atomic levers / signals / allocation.** Main lever: *look less*. Signals: library stage boundary, direct neighbour action/mode agreement, valid successors and state deviation from the chosen continuation. Allocation: stage-level permission for a longer observation interval, with an event-triggered veto nested inside it. Existing stall detection and the *fewer calls* budget remain independent. No *look half* assumption is needed.

**Minimal mechanism.**

1. Keep the existing `C0`-control action commitment, including grasp/place micro-sequences. At its anchor, consider exactly one additional decision block, `C1=C0+R`; this is a general relative rule, not a new physical duration. Do not begin with arbitrary-length trajectory playback.
2. An easy stage has a contiguous candidate prefix through `C1`, no forthcoming stage boundary, and acceptable `D(C1)`. Treat any missing successor as a review requirement; never hold the terminal row and call it valid continuation. For the initial conservative variant, every returned neighbour must have a valid same-episode/same-task chain, stay in its command mode, and agree on that mode. This support rule has no tuned mass threshold; `P` is then a diagnostic, not another adjustable gate. The structural library-head screen passes **791/1,817 anchors (43.5%)**, with cell shares **28.2–61.7%** (§3). The final eligible share after checking the exact native-tail prefix, dispersion and online state remains unmeasured.
3. Retain the exact original synthesized chunk through its supported horizon. GR00T can supply the extra five controls from its 16-step chunk. For π0.5 beyond its 10-step chunk, the candidate extra head comes from each row's true successor chain at the matching control offset, with the original weights. Rebuilding earlier heads from successor rows would change A and is forbidden in this comparison. No policy tail is ever padded to create executable controls beyond `H`.
4. Before entering the extra block, compare the live **relative** proprioceptive displacement since the anchor with the weighted library displacement along these same chains. Use a normalized residual `e`; request vision if it exceeds its library-calibrated envelope, if alignment is unresolved, or if lifecycle/state validation fails. Absolute agreement with a world-frame demo pose is unnecessary. Missing state is not zero deviation.
5. At the end of the extra block, request a fresh observation. A stage boundary also expires permission. A LOOK recomputes retrieval; it is **not automatically a MISS**. The existing C stall/budget logic decides whether the policy must run. Count all extra looks and calls. Do not suppress a call already scheduled by the budget to manufacture vision savings.

**Calibration / hard versus easy.** Fit `D/P` rank distributions on successful library pseudo-queries using the same candidate construction. Use the maximum per-episode deviation over the existing native-commit horizon as an empirical reference envelope; if the extension exceeds that envelope at entry it is hard. A boundary/support failure is hard regardless of the rank cut. The budget solve chooses the joint agreement cut from observed calibration ranks. Fit-time tables store source episode counts, extrema and support; with sparse libraries these are empirical reference envelopes, not a safety certificate.

The library's observed state path was produced by executed demo heads, whereas A executes blended committed tails. Consequently the library envelope is only a starting hypothesis. The top profile request must check it against B-val A's actual states. If the discrepancy systematically rejects extensions or fails to precede deterioration, drop this tube test instead of adjusting its physical-unit threshold on test rollouts. No dynamics model is added.

**Generality.** Inputs are proprioception, executed controls, library chains and the existing vision encoder at review times. On another robot, replace interface metadata and re-fit ranks/scales; no named grasp/place classes or task IDs determine difficulty. On RoboCasa, include base displacement/orientation and every real action channel; a matching arm path with a moving base is insufficient. On a real arm, the same logic applies at the controller's actual update boundaries, with existing low-level safety control outside the compute budget. `H=C0` requires successors; `H=R` eliminates native-tail extension but not necessarily library continuation. If neither is valid, M1 reduces to the reference cadence.

**Expected frontier effect (hypothesis).** At unchanged calls per active control, shifting a fraction `f` of controls from 10-control to 15-control observation intervals gives `delta IR = c_v f(1/2-1/3)`. The all-controls upper bound is **.02533** for π0.5, **.02467** for GR00T; for `f=.5`, **.01267/.01233**. These are algebraic cost bounds, not predicted savings: stages, ends, vetoes and added calls reduce them. Pure-cache floors would become .05067/.04933 only in the ideal all-extended case. Re-solving a budget per anchor without accounting for the changed cadence would confound *look less* with *fewer calls*.

**Cost tier / CPU.** Cheap online, moderate CPU-only prefit; no extra deployed model forwards. Existing 16-row feature extraction measured median **.343–.381 ms** per anchor across cells (§3), including statistics unnecessary for M1. This is an analysis microbenchmark, not a plugin latency measurement. Future-chain materialization adds `O(K C1 D)` arithmetic; a cached-state deviation check costs `O(K D_state)`. Store chains/boundaries once. Measure actual extra p50/p95 latency against the reference 1–2 ms retrieval before accepting this as cheap.

**Plugin sketch.** `fit` reads metadata, successful chains, stage boundaries and calibration ranks; `reset` clears the saved anchor. `query` calls A once, preserves `Result.action`, records rows/weights, eligible extension, deadline and expected displacements. `blind_step` serves A's usual tail unchanged, then either one authorized extra block as `BlindResult` or `LookReason`. `policy_tail_step` remains bounded by real `H`; first test extends cache segments only, leaving policy commitments unchanged. The judge receives current stall/budget information only after a LOOK. Record selected horizon, actual executed controls and veto reason. Current `closed_loop/blind.py` and `plugin.py` hard-code offsets 5/10 and two cameras; those interfaces require parameterization for transfer, not a claim that this proposal already runs there. R4 `_BlindMixin._advance` holds dead-end rows: use `consecutive_next` validation but **do not inherit that terminal fallback**.

**Kill criteria.** Drop M1 if (a) all-neighbour support/boundary checks leave zero extensions; (b) added review/call/CPU cost cancels saved encoder time; (c) the extension group has worse held-recording transition consistency with no useful veto lead time; or (d) randomized closed-loop evaluation fails SR noninferiority to the identical controller without extension at a predeclared owner tolerance. For a concrete screening rule, use **1 pp absolute SR loss**, the already-used R5 Q3 tolerance, and require a one-sided paired interval to clear it before claiming success. An underpowered interval is inconclusive, not approval. Also compare a uniform extension arm at matched IR: no advantage over uniform means the stage signal is unnecessary even if the lever works.

### M2 — Stage-consensus routing (secondary; test the signal before coding an arm)

**Hypothesis.** Persistent ambiguity over a stage is more relevant to compute allocation than one anchor's predicted cache–policy disagreement. A stage with incompatible candidate continuations may deserve more full-policy calls; agreement alone may still be confidently wrong.

**Mechanism.** Keep A's observation cadence and `C0` commitment fixed. At stage entry, inspect the 16 library continuations to the earliest candidate stage boundary, capped by valid successor support. Let `D_stage` be the maximum per-R-block dispersion over that interval, and `P_stage` the mode disagreement at entry. Compute `S = max(F_D(D_stage), F_P(P_stage))`, where `F_D/F_P` are successful-library episode-balanced empirical midrank maps. This uses spread over the available **library stage continuation**, not future live observations, `sum_i w_i LOEO_residual_i`, or an estimated teacher disagreement. If no complete block is supported, use the uniform reference allocation. Freeze the stage score until its boundary; each subsequent anchor has call probability `p_s=min(1,lambda S)`. The budget regulator chooses `lambda` to meet target IR including stall overrides. A mode change must persist for one completed reference commitment to open a new stage; a mode inconsistent with the current library candidates requests immediate review. This persistence time derives from `C0`, not a benchmark timestep constant. A stall can override any stage allocation.

Only the *fewer calls* lever changes; *look less* and *look half* are held fixed for identification. “Easy” means low library-ranked continuation spread; “hard” means high spread or a stall override. The score is computed from information available at entry, never an average of the stage's later observed states. No group-level success classifier is fitted. The crucial comparison is **stage-frozen versus anchor-wise versus uniform allocation at matched realized IR and the same commitment**. This isolates stage aggregation from merely renaming R6 routing.

**Generality / calibration.** The same metadata, scales, mode definitions and empirical CDF rules apply on another robot. Keep K=16 as the inherited A contract, not sixteen statistical replicates. Recompute state/action transformations and rank tables for the new library. Do not hard-code a wrist camera, episode third or task identity. The same declared `rho` and inherited regulator determine `lambda`; no cell-specific signal sign is selected from outcomes. If valid stages are unavailable, use uniform C rather than inventing semantic stages. For noisy/continuous hands, E1's stage provider needs its own portability evidence.

**Expected effect / cost.** At equal IR this is an SR hypothesis with **no supportable positive point forecast**. Suppressing .1 full calls per five-control decision would save .0848/.0852 IR if success were preserved, over three times M1's all-extended vision saving; it is also the riskier claim. Measured direct-dispersion association in §3 is only a same-observation diagnostic, so it does not justify those suppressed calls or validate `D_stage`. Online work at stage entry is `O(K L_stage D)` for supported continuation length `L_stage`, then a scalar probability plus the existing hashed coin. CPU-only calibration stores rank maps and successor summaries; no added inference model. Long stages may exceed the chunk-feature microbenchmark substantially: profile their entry-time p95 before treating this as cheap. There is no dense blind-state tube update in M2.

**Plugin sketch.** `query` invokes A, feeds the existing stall tracker, detects stage entry, saves `S` and stage ID, and passes the latched probability to the existing C budget/coin mechanism. The judge's action is the ordinary HIT/MISS; `blind_step` and `policy_tail_step` preserve the original commitment. Log stage ID, entry step, entry score, current diagnostic score, p, coin, realized source and override. Stage duration may change after a call: the replay budget is only a calibration model, and realized control-normalized IR must be checked. Do not use per-request IR alone or silently import a stalled recording's future states as M2's counterfactual.

**Kill criteria.** First require supported positive call-benefit heterogeneity in pre-treatment stage strata from the randomized records, with task/init-cluster uncertainty and the actual continuation controller stated. If uniform is as good, or only teacher-disagreement prediction improves, drop routing. If the experiment lacks support, defer it. After that gate, drop if stage-frozen routing fails to improve the IR–SR frontier over uniform C, is matched by anchor-wise S (no stage-level contribution), or requires task-specific cut points. Use the same predeclared 1 pp screening tolerance as M1 for any cost-saving SR claim.

### Expected effects by cell, not tuning branches

These priorities are hypotheses based on the brief's measured reference frontier and §3 support counts. They do not enter either method's serving rule.

| Cell | Pure-SR-reaching R6 IR, from brief | M1 expectation | M2 expectation |
|---|---:|---|---|
| π0.5 L10-50 | Not reached; best .894 @ .458 | Smaller vision saving; fragile successors and only 3.61 effective episodes. Cannot plausibly close the .190 A–pure gap by saving vision alone. | Potential room for call improvement, but strongest warning against treating agreement as safety. |
| π0.5 L10-500 | .197 | Best π0.5 successor-continuation candidate: .792 stable 15-control weight mass. | Test only if call-benefit heterogeneity survives; R6's failure is the prior. |
| π0.5 Spatial-50 | .442 | .645 stable mass, frequent ends reduce opportunity; do not confuse a long closed-gripper run with successful grasp. | Harder than dense Spatial; possible benefit remains unestablished. |
| π0.5 Spatial-500 | .118 | Plausible small left shift; already strong A, so added calls can erase the gain. | Expect little headroom; preserve zero-call capability. |
| GR00T L10-50 | .453 | Native 15-control tail is mechanically simplest; 31.8% of anchors have majority weight on one episode. | Large SR gap, weak independent support; high-value stress test, not a likely win claim. |
| GR00T L10-500 | .184 | Native tail plus .781 stable mass is the clean first mechanism check. | More support than size 50; still needs causal gate. |
| GR00T Spatial-50 | .298 | Native tail feasible; .701 stable mass; beware known sensitivity of the no-progress guard. | Expect uncertainty; do not reuse π0.5's score sign or guard effect. |
| GR00T Spatial-500 | .052 | Existing frontier already near the ideal .04933 15-control vision floor: very little room. | Expect no useful rescue allocation at the bottom of the frontier; extra calls may hurt. |

## 3. Preliminary evidence and reproducibility

New analysis is **descriptive**, from 80 A episodes (ten per cell, one non-test initialization per task), **1,817 anchors**, and eight action libraries. Outcomes are retained for audit only; they do not select thresholds, identify call benefit, or yield reliable task-level SR. Same-observation shadow chunks are diagnostics. No pilot test-init features or outcomes were used in these calculations.

| Cell | Anchors | Median effective episodes | Majority weight on one episode | Median within-episode rho: raw dispersion vs shadow disagreement | 15-control stable successor weight mass | Median library mode-span controls |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 L10-50 | 346 | 3.61 | 14.2% | .590 | .705 | 55 |
| π0.5 L10-500 | 274 | 7.19 | 8.4% | .713 | .792 | 47 |
| π0.5 Spatial-50 | 152 | 3.96 | 3.9% | .753 | .645 | 45 |
| π0.5 Spatial-500 | 135 | 9.15 | 3.7% | .862 | .713 | 46 |
| GR00T L10-50 | 365 | 3.02 | 31.8% | .655 | .706 | 54 |
| GR00T L10-500 | 290 | 7.51 | 10.3% | .756 | .781 | 53 |
| GR00T Spatial-50 | 133 | 4.09 | 3.0% | .718 | .701 | 47 |
| GR00T Spatial-500 | 122 | 10.56 | 2.5% | .873 | .727 | 47 |

The stable-mass column is the mean over recorded anchors of `sum_i w_i 1(valid successors and unchanged command for 3R controls)`. It is **not** the share of anchors eligible under M1's stricter all-neighbour gate, a fraction of safe controls, or an SR estimate. It follows recorded library heads, not the open-loop future of the robot. At 20 controls, this mass falls to **.541–.732**, arguing for a one-extra-block first experiment. Large libraries contain failed episodes as well; only successful episodes enter the mode-span census and scale fit, whereas the recorded neighbour sets are preserved as deployed.

The stronger structural screen requires **all 16** valid 15-control head continuations, unchanged command along each, and the same mode across neighbours. In the table's order it passes **143/346, 169/274, 51/152, 65/135, 103/365, 136/290, 54/133, 70/122** anchors: **791/1,817 = 43.5%** overall. This counts anchors before dispersion or state vetoes. Native committed tails can differ from the successor heads used by this census, so P1 must additionally check the exact proposed executable prefix. Neither 43.5% nor the weight-mass column can be substituted for `f` in the IR formula without an actual scheduling/control-occupancy calculation.

Median within-episode correlations of **whole-episode progress SD** with shadow disagreement range only **.045–.425**; effective-episode count correlations range **−.263 to +.069**. Those fields should be support/ambiguity audits, not unvalidated “difficulty” scores. This agrees with R1's warning that its action oracle had *larger* phase error than B0 (.115 versus .073 in one cell). R1's “track” rows had the lowest error (.39–.46), but selection into tracking is observational and is not evidence that forcing playback improves SR. Sources: `r01/FINDINGS.md`, lines containing “Phase error” and “Temporal behaviour.”

In this new analysis, the ratio of median tail to head normalized dispersion for controls 5–9 versus 0–4 is **1.016–1.191**, depending on cell. Later actions can disagree more even when the current head agrees. The much higher B-val rho values than R6's .434 pooled estimate are from different recordings and a **median of episode correlations**, not the registered pooled estimator; there is no claimed replication improvement.

The measured feature-loop median is **.343–.381 ms**, cell p90 **.406–.434 ms**, using one Python process and the required pinned CPUs. This excludes CSV parsing, JSON decoding, retrieval, plugin plumbing, prefit and extra future-action construction. It includes full neighbour action reductions, episode grouping, structural-support checks and several diagnostic scalars. It is evidence that simple direct neighbourhood summaries are inexpensive, not a serving-latency promise.

Reproduce from `/home/weiland/projects/openpi`:

```bash
taskset -c 30-31,74-75 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E5_cross_domain/analyze.py
```

The script records input CSV/manifest hashes, source paths, shapes/counts and its own hash in `evidence.json`; only the two owned data outputs are written. It asserts 16 finite nonnegative weights and matching task IDs, and validates same-task/same-episode consecutive successor edges. Gripper census flattens the **executed first R controls of each row**, rather than concatenating overlapping H-step predictions. Horizon probes `2R,3R,4R` are exploratory measurements; only the relative one-extra-block proposal is recommended. No hypothesis-test p-values or confidence bounds are manufactured from repeated anchors.

## 4. Profile tools wanted

### P1 — `stage_horizon_audit` (top request; offline first)

**Question:** after accounting for actual continuation support and observation cost, is there any stage-specific reason to extend a commitment, and can proprioception veto trouble before the extra block?

**Existing inputs:** successful library actions/states/prev-next and manifest; B-val A anchors, decisions, `action_steps` and actual-control tables; P3 snapshots/input archives for logged blind observations and shadows; frozen E1 stages if selected. Use only proprioception, own encoder keys and actions to compute candidate decisions. Privileged contacts/object predicates may label failures in a separate evaluation column, never a feature.

**Outputs:** per-anchor candidate horizon and all-neighbour support, future `D/P`, command-boundary distance, effective episodes, proposed versus actual review time, state-tube residual and veto lead time; curves of admitted active controls versus horizon; added LOOK/MISS cost and maximum available IR saving; separate native-tail and successor results; task/init-cluster summaries. Include constant-cadence and randomly placed extension controls. Report current-observation shadow disagreement and actual subsequent-state consistency separately; both remain proxies.

**Decision informed:** whether M1 has enough eligible exposure to implement; whether the strict support gate is too restrictive; whether native tails and successor continuations require different hypotheses. Do not tune loosened gates on test tables. Existing A recordings execute only the original commitment, so they cannot establish SR for an extra blind block or a changed subsequent trajectory.

**New telemetry after implementation:** stage ID/version, anchor control index, granted horizon, pointer chain and weights, expected state delta, current normalized deviation, boundary/support/rank veto, LOOK reason, actual issued controls, camera subset, stage forward counts and timing. Full images/contacts need not be added to serving logs. A randomized extension-versus-reference campaign is required for causal SR and net IR; diagnostic replay alone cannot pass M1's final kill criterion.

### P2 — `stage_call_value` (coordinate with E2/E4; mostly existing randomized data)

**Question:** does stage-entry neighbourhood consensus modify the effect of a CALL under the recorded continuation controller, or only predict teacher disagreement?

**Inputs:** pre-treatment stage-entry features reconstructed from P3 `neighbours`, anchors and input archives; randomized assignment/actual propensity and accepted outcomes from P3 and the uniform-lottery records. Reuse strict attempt joins and the campaign catalog. Top-10 truncated logs are insufficient to reconstruct the 16-member signal; when full inputs are missing, mark unavailable rather than substitute a different score. Exclude forced/capped choices without local support.

**Outputs:** supported CALL–CACHE contrasts by pre-treatment stage score, stage-entry versus interior, actual propensities, effective task/init support and cluster intervals; downstream cost as well as final success. Compare uniform, anchor-score and stage-frozen allocations. Do not condition on final call count, eventual failure, realized stage duration, or future stage-average score. A local randomized-call effect does not identify an entire stage-routing policy without additional assumptions/support.

**Decision informed:** M2 is dropped/deferred unless there is supported, useful heterogeneity. This explicitly prevents repeating R6's disagreement-to-call-value leap. Existing records need no new rollouts for the first screen; exact online stage state/probabilities need the M2 telemetry listed above for implementation profiling and a stage-level randomized evaluation for the final claim.

### P3 — `portability_and_cost_audit` (offline interface replay, then measured timing)

**Question:** is the implementation one rule across interfaces, and is its claimed saving actually saved model work?

**Inputs:** manifest and adapter contracts; LIBERO libraries; locally available RoboCasa observation/action metadata; synthetic interface fixtures with C=1/2/3 cameras, H=R and H>R, changed action masks, gripper sign conventions, base movement and unequal timestamps. Fixtures test code semantics, not transfer SR. Actual robot/policy records are needed for deployment calibration.

**Outputs:** declared-versus-hard-coded fields, invariance to equivalent unit/coordinate representations, dead-end/partial-commit handling, rejected unsupported stages, action-wire parity, and real per-camera/stage dispatch/cost accounting. CPU `runprof` supplies query/blind timings; coordinator measurements with the profiler disabled supply encoder/full-call latency. Wrist-only .055 and .152/.148 total-vision shares are existing cost assumptions/measurements for their stated LIBERO configurations, not transferable constants.

**Decision informed:** whether either method passes the checklist below; whether a three-camera *look half* extension is worth a separate study. New timestamps/base-state and per-camera invocation telemetry are required wherever the target adapter does not already expose them. Full-vision shadow recordings cannot measure deployment latency or prove that skipping one camera preserves the encoder interface.

## 5. Portability checklist every R7 method must pass

| Check | Required evidence / failure behavior |
|---|---|
| **Inputs and dimensions** | Manifest/adapter declares cameras, H, R, control period, executed counts, action/state masks, coordinate frames, gripper semantics and output transform. No literal `:7`, `:8`, camera count 2, sign or offset 5 in the method. Constant valid channels remain valid. |
| **Stages without task names** | Same frozen segmentation/alignment rule; no task/suite lookup, benchmark object predicates or manually named phases. Unresolved/unsupported stages fall back to the reference cadence/allocation. A gripper mode is not a grasp-success label. |
| **Calibration and support** | Every scale, horizon, cut and time window has a manifest/library rule or declared owner budget; report episodes, not neighbour rows, as calibration units. Refitting on the new non-test library is permitted; outcome tuning on its test scenes is not. No conformal/SR guarantee from five demos/task. |
| **Time and embodiment** | Compare actual elapsed controls/seconds and physical pose differences. Include mobile-base state/action and preserve transformations on a real arm. Never port a 10-control interval as though it had the same duration at another rate. |
| **Observation/action validity** | Reuse only supported chunk tails or contiguous timestamped successor heads. Stop at terminals, missing state, dropped/partial controls and unresolved boundaries; distinguish rendered cameras from encoded cameras. Preserve native commit action parity. |
| **Cost completeness** | Re-measure the new policy's camera/prefix/head costs and account for all extra looks/calls and CPU/network work. Report SR versus actual-control-normalized IR and latency. If the encoder cannot omit a camera, the *look half* lever is unavailable. |
| **Cross-policy and cross-scene validation** | One algorithm/rule on both current policies, then RoboCasa scenes/instructions held out with proper pairing; a new embodiment needs new empirical validation. Success on two LIBERO models is not a demonstrated robot transfer. |
| **No hidden new model or sensor** | Only own-policy encoding, available proprioception/executed actions and library metadata in decisions. Simulator-only fields may explain evaluation failures but cannot grant laziness. Existing robot safety control remains active at its normal rate. |

### Concrete transfer stress test

**RoboCasa365:** the local `src/openpi/policies/robocasa_policy.py` requires all **three cameras for π0.5** and returns **12 action channels**. `exp/robocasa365/episode_runner.py` concatenates relative EE pose, base position/rotation and gripper state; it maps left agent view, wrist and right agent view explicitly and applies `convert_action`. The generic scheduler transfers as a hypothesis; a wrist-centric LIBERO representation, fixed-base displacement test, binary-only stage vocabulary, 8-state/7-action assumption and two-camera plugin do not. R6 ideation G documents the present π0.5 RoboCasa horizon as 50, but the deployed manifest must be checked before scheduling from it.

Do not assume the target runs at a particular frequency: upstream RoboCasa currently defaults to **20 Hz**, but this does not establish the local wrapper's effective wall-time/control cadence. Read actual configuration/timestamps and recalibrate duration-dependent quantities. A different rate changes both dynamics and the duration of a saved block. The two side views also mean “look half” must mean a validated selected camera subset, not literally half the cameras or always wrist-only. [Upstream constructor and frequency contract](https://raw.githubusercontent.com/robocasa/robocasa/main/robocasa/environments/kitchen/kitchen.py)

**Real arm:** proprioception may track a demo while the object slips or an obstacle moves. The stage deadline must force renewed vision even with a small state residual; there is no unlimited blind permission. Quantify action-to-observation latency, camera exposure age and encoder savings separately. Fresh robot/library recordings are needed; this proposal claims no physical safety guarantee. Task completion labels and scene pairing must be defined independently of LIBERO's timeout-based evaluation.

**Another policy:** direct neighbour dispersion works with any policy whose library actions share a declared executable coordinate system. It does not require diffusion steps, logits or uncertainty heads. Native-tail availability, policy statefulness and camera-selective execution remain interface preconditions. A recurrent policy might need its own observation-state updates even on a cache hit; neither caching keys nor skipping a forward is automatically valid. Preserve its state contract or mark that lever unsupported.

## 6. Risks and open questions

- **Confidently wrong retrieval:** every neighbour can miss the same unseen object state. Episode diversity helps audit dependence, but neither diversity nor low dispersion proves safety. Keep finite review deadlines and test low-dispersion failures explicitly.
- **Mode ambiguity versus valid multimodality:** two good trajectories can disagree, making a conservative gate waste compute. Conversely a narrow kernel can hide alternative modes. Report full neighbour membership and episode masses; do not present kernel variance as calibrated uncertainty.
- **Stage vocabulary:** continuous turning, pushing, wiping, mobile navigation and grasp-preserving placement can occur without a gripper change. The minimal partition is deliberately coarse. E1's kinematic stages are the relevant cross-link; any improvement must survive a frozen-stage ablation.
- **Commitment tradeoff:** R7 brief reports 36/97 cache failures with grasp outcomes contradicting the library, and R5 showed that five-control replanning could interrupt useful grasp/place sequences. “Boundary detected” should first mean “review,” not “interrupt everything and call.”
- **Weak causal prior for M2:** R5 Q3 found zero supported conservative saving in all fitted landmark suppression policies; R6 residual routing equaled uniform. M2's novelty is direct stage-level continuation ambiguity and persistence, but success is speculative. Profile its causal gate before consuming implementation capacity.
- **Distribution shift and arithmetic ceilings:** library pseudo-queries, B-val cache occupancy and an extended controller induce different states. A budget replay is not an off-policy SR evaluation. Extra calls can eliminate M1's at-most .0253/.0247 IR saving; dense GR00T Spatial already leaves little room.
- **Implementation versus method generality:** this memo proposes portable decisions, but the present plugin is a two-camera LIBERO implementation with hard-coded slicing. A manifest-only port is not complete until camera preparation, output transformation and policy history contracts also pass P3.
- **Selection status:** no proposed gate, stage routing, latency bound or SR improvement has passed closed-loop validation. Select M1 only after P1 exposes usable supported horizons; retain M2 only if P2 supplies direct evidence beyond disagreement prediction.
