# R5 ideation D: verify the outcome of the grasp

**Priority: test a command-matched finger-aperture contradiction before executing a cached transport action.** This is the strongest new mechanism I found in the completed R4 logs. It can target a small number of full-cost policy calls without changing retrieval, synthesis, library size, or the normal execution budget. At 50 episodes its historical exposure includes 36 of 97 failures; at 500 it includes 11 of 60. Those counts are opportunities, not recoveries. The biggest remaining failure cluster is partly outside this signal, so the evidence does **not** establish a new general 5–10-point SR lever.

A second, lower-confidence proposal changes the correlation of the noise used by consecutive full-policy rescues. It adds no inference calls, but needs a fixed-observation diagnostic before a rollout pilot. I rejected generic pose rebasing, generic motion-innovation alarms, and an unconditional “closed fingers means empty grasp” rule.

## 1. Evidence, scope, and costs

### Reproducibility and interpretation

All new scripts and outputs are in this directory; [RUN.md](RUN.md) gives exact commands and dependencies. Work used the specified Python, CPUs `14-17,58-61`, single-threaded BLAS, and no GPU. No policy inference, closed-loop run, server, remote worker, or sub-agent was started. Existing files were only read.

Binding interpretation comes from [R5 FINDINGS](../FINDINGS.md), [R2 ANALYSIS](../../r02/ANALYSIS.md), [R3 ANALYSIS](../../r03/ANALYSIS.md), R4 selection/coding briefs and hand-backs, K8/K9 reports, R4 ideation A/B/C including C's second report, the exploration ledger §§8–10, and the harness/plugin contracts. The missing K10 hand-back and missing completed GR00T R4 summaries are recorded in RUN.md. The cold store was used because the hot store was absent.

`diagnose_logs.py` follows `closed_loop/ops/kpi.py`'s accepted-attempt rule and deduplicates `(uid, step)`. Each K7 arm below has exactly 500 accepted episodes and complete state/action coverage. “Failed” means the episode's recorded final outcome. It does not label an individual action as erroneous. No outcome, teacher action, or future state is an online input to either proposal.

New calibration fits use the deployed library at the indicated size. **No new fit uses borrowed big-library information.** Closed-loop outcomes and recorded teacher actions are diagnostic evidence only. The servo diagnostic uses the harness's current-library action standard deviation for comparable error units; that normalization is not a deployed correction fit.

### Recomputed best-arm ledger

Source: [diagnose_logs.py](diagnose_logs.py) → [logs_summary.json](logs_summary.json). All decisions execute five controls. π0.5 owner cost is

`IR = (0.152 V + 0.848 M) / N`,

with every MISS priced at full inference. Search and the new CPU helper are outside this IR; their latency must be reported separately. Eager costs are not mixed into this basis.

| π0.5 arm | Library episodes | Success / 500 | Decisions N | Vision V | MISS M | IR |
|---|---:|---:|---:|---:|---:|---:|
| K7 l10 anchor-tail | 50 | 403 | 31,186 | 18,566 | 5,554 | .241513 |
| K7 l10 anchor-tail | 500 | 440 | 28,888 | 16,547 | 3,959 | .203281 |
| K7 l10 phase B2 | 50 | 350 | 34,229 | 21,941 | 6,884 | .267979 |
| K7 l10 phase B2 | 500 | 431 | 29,136 | 16,691 | 3,111 | .177621 |
| K7 spatial anchor-tail | 500 | 491 | 10,630 | 5,643 | 596 | .128236 |
| K7 l10 B0 | 500 | 415 | 30,242 | 30,242 | 3,300 | .244534 |

The proposal baseline is anchor-tail, held fixed. I do not propose it again. Spatial-50 K7 and GR00T mixed counterparts were not completed in the inspected inventory; their SR/IR must not be fabricated from these rows. The historical GR00T graph basis cited below is `exp/libero_groot/config/rit/cost_groot_libero_measured.json`: stages **6.145863/7.191965/28.104 ms**, total **41.441827 ms**; it does not certify the current serving stack.

In l10 anchor-tail, tasks 4 and 6 account for **37/97 failures at 50** and **24/60 at 500**. Task 4 is the two-mug placement task; task 6 combines a mug and pudding. The proposed aperture rule detects **zero task-4 episodes at either size**. Task 0 contributes 16 failures at 50, all reached by the rule, but only two failures at 500. This explains why a useful sparse-library intervention can have much less aggregate headroom at 500.

### A specific action-outcome contradiction survives the controls

Sources: [contact_audit.py](contact_audit.py), [decision_audit.py](decision_audit.py), and their JSON outputs. Let `g=(rs[6]-rs[7])/2`, and let `u=(g-q01)/(q99-q01)` using **that library's** aperture quantiles. `u` is a normalized finger-aperture coordinate, not a ground-truth object-contact label.

At a one-decision-old anchor, compare the observed aperture with `next[row]` for the anchor's exact 16 library members. Retain only members whose five gripper signs match the action actually executed. Require retained original mass ≥.5, retained mass with successor `u>.25` ≥.75, and retained mean successor `u>.25`. Alarm only after two fully closed executed heads, when current `u<.05`. Every evaluated alarm in this table is a scheduled blind HIT. This tests what happened after a command; it does not predict a future gripper sign change.

| l10 anchor-tail diagnostic | 50 episodes | 500 episodes |
|---|---:|---:|
| Unconditioned successor-mean alarm: episodes / failed episodes | 94 / 50 | 36 / 19 |
| **Command-matched, ≥75% contact consensus: episodes / failed episodes** | **73 / 36** | **24 / 11** |
| Successful episodes also reached | 37 | 13 |
| Alarm decisions before a once-per-episode cap | 223 | 47 |
| Observed failure fraction among reached episodes | 49.315% | 45.833% |
| Overall failure fraction | 19.400% | 12.000% |
| Expected reached failures using each task's own failure rate | 14.58 | 2.56 |
| Even inits: reached failures / alarms | 16 / 35 | 5 / 9 |
| Odd inits: reached failures / alarms | 20 / 38 | 6 / 15 |

The within-task comparison reduces the concern that the rule merely identifies difficult tasks. The init split is descriptive replication, not an untouched prospective test. Threshold sensitivity is modest: floors `.025/.05/.10` and expectation thresholds `.20/.25/.35`, with the other requirements fixed, reach **68–80 episodes / 33–40 failures at 50**, and **23–27 / 10–13 at 500**. These are sensitivity checks, not a selected optimum.

The subsequent physical/action evidence is more informative than final-outcome enrichment:

- At 50, **28/36** reached failures move upward by more than `.02` in normalized state-z during the alarmed cached head. **30/36** reach a policy MISS within three more decisions; **25/30** of those policy heads reopen the gripper. All 36 alarmed cached heads keep it closed.
- At 500, corresponding counts are **8/11 upward**, **8/11 with a MISS within three decisions**, and **7/8 reopening**. All 11 alarmed cached heads keep it closed.
- Reached successful episodes also exhibit this sequence: upward **33/37 and 11/13**; reopening on the next MISS within three decisions **28/28 and 11/13**. Thus an alarm in an eventually successful episode need not be a false physical alarm; it may mark an existing recovery.
- Among reached failures, median delay to the next policy call is **one decision at 50** and **three at 500**. There are already **80 and 17 MISSes** in the six-decision windows starting at those first alarms. An earlier call is not established to help merely because the eventual episode fails.

**Hypothesis:** some cache actions start transporting after a grasp outcome inconsistent with their source demonstrations; the policy subsequently reopens to recover. Intervening before that transport may prevent displacement or save recovery time. Images/contact ground truth and an intervention are still needed to distinguish missed grasps from other aperture mismatches and establish rescue value.

[contact_rule.py](contact_rule.py) is a pure online-legal reference helper. It reproduces **223/223 and 47/47** alarm decisions, with **zero disagreements** against the independent audit. On **12,120 / 11,841** eligible blind requests, measured CPU median is **13.880 / 11.515 μs**, p95 **78.982 / 75.686 μs**. This is helper timing, not server latency or a GPU benchmark. See [contact_rule_validation.json](contact_rule_validation.json).

### The policy's sampling variation is real; its contribution to failures is unknown

Source: [noise_probe.py](noise_probe.py) → [noise_probe.json](noise_probe.json). The store's existing resampling experiment has 500 fixed observations and four fresh policy draws per observation for each cell. All distances exclude padding and use the executed five-by-seven block.

| Model / suite | Fresh-draw vs recorded action RMS, σ units | Any gripper-sign disagreement / 2,000 comparisons | States with disagreement among the four fresh draws / 500 |
|---|---:|---:|---:|
| π0.5 spatial | .175208 | 161 (8.05%) | 63 |
| π0.5 l10 | .184585 | 205 (10.25%) | 77 |
| GR00T spatial | .052873 | 25 (1.25%) | 10 |
| GR00T l10 | .054612 | 78 (3.90%) | 28 |

Same-noise replay RMS is only **.004211/.003908** for π0.5 spatial/l10 and **.002844/.002287** for GR00T. These experiments establish sampling sensitivity, not harmful mode switching. They are library-independent evidence and therefore apply to the hypothesis at both sizes; they are not new calibration data for a deployed fit.

The l10 anchor-tail logs contain **2,870 / 1,175 consecutive MISS pairs at 50/500**, in **299 / 260 episodes**. Of these pairs, **2,226 / 707** occur in failed episodes; length and rescue-selection confounding prevent causal interpretation. Continuous action boundary jumps average **.372141 / .366414 σ**, with **136 / 78** gripper flips. The fixed-observation first-action continuous sampling difference is **.164139 σ** for π0.5 l10, versus **.044687** for GR00T l10. The statistics use different observations and are not a variance decomposition.

### Deployment bytes, both library sizes

Source: [decision_audit.py](decision_audit.py) → [decision_audit.json](decision_audit.json). MB is decimal. AWM sizes are measured `r02_g{50,500}/fits/oscl{50,500}_{p,g}_{sp,l10}_cl2.pkl` sizes. The last column is the owner's deployed-pkl comparison, not a newly statted file.

| Cell | Entries 50 / 500 | Actual AWM pkl MB 50 / 500 | Conservative D1 addition MB 50 / 500 | Deployed pkl MB |
|---|---:|---:|---:|---:|
| π0.5 spatial | 1,018 / 10,909 | 21.341661 / 46.993114 | .009170 / .098189 | 431 |
| π0.5 l10 | 2,640 / 29,472 | 24.631208 / 94.143229 | .023768 / .265256 | 1103 |
| GR00T spatial | 1,063 / 11,751 | 22.249307 / 58.156565 | .009575 / .105767 | 429 |
| GR00T l10 | 2,645 / 29,631 | 26.672701 / 117.303700 | .023813 / .266687 | 1068 |

π0.5 spatial's nominal 50-library has 49 episodes. D1's conservative addition is **9 bytes/entry + 8 bytes**: float32 aperture, uint8 five-sign pattern, int32 successor, and two float32 quantiles. K7 already retains states, actions and successor metadata, so an implementation can reuse them and retain just the quantiles. The table budgets separate compact arrays instead; it excludes Python/serialization overhead and is not a serialized new fit.

The actual K7 base pickles used here are **32,704,527 bytes l10-50**, **142,348,562 l10-500**, and **66,500,552 spatial-500**. Adding the conservative D1 representation yields calculated totals **32,728,295**, **142,613,818**, and **66,598,741 bytes**, respectively. No K7 spatial-50 or GR00T mixed pickle is substituted with an AWM pickle in these totals. Their existing AWM sizes are supplied as reference only.

D2 adds **zero library bytes**. Retaining one float32 noise tensor adds **1,280 bytes/connection for π0.5** or **2,048 for GR00T**, plus small counters and temporary generation buffers.

## 2. Ranked proposals

### D1 — Interrupt a cached transport when the grasp outcome contradicts its source

**Pitch and mechanism.** Use actual finger motion to verify the consequence of the preceding cached command. When a command-matched library kernel expects a wide grasp but the fingers close near their lower endpoint, acquire the current vision anchor and request one full policy rescue before executing the next cached head. The measured upward-then-reopen sequence motivates this intervention. It is an action-outcome check, distinct from R4's visual stillness guard, gripper-event-ahead gate, and phase continuation.

**Precise algorithm and integration.** Implement a `VisionConfirmedBlindMixedJudge` subclass for π0.5, with the existing `anchor_tail`, budget 1, `budget_only`, `noprog_span`, and `events='none'` settings held fixed.

1. `fit(lib, ctx)`: call the existing fit; calculate the aperture table/quantiles from the same deployed candidate library. Preserve the existing 16-member AWM synthesizer and all baseline guard calibration.
2. `reset(episode)`: clear the pending alarm, offered step, and one-intervention-per-episode budget. Track episode/task identity exactly as K7 does.
3. In `blind_step(bq)`, first enforce existing lifecycle validity. Evaluate the reference helper only when the anchor is from `step-1`, was a real vision HIT, `prev_hit=True`, and the normal proposal would be eligible for its one blind continuation. Use `hist_a_exec[-2:,:5,:7]`, current `rs[:8]`, and the saved full anchor rows/weights. The equation and thresholds are those in §1 and `contact_rule.py`.
4. On contradiction and unused budget, record a pending alarm and return `LookReason(new_code, 'aperture_contradiction')`. Do not broadcast an action or advance history on this path. The plugin then computes the normal full vision anchor.
5. In that decision's `query(q)`, call the unmodified superclass, then set `extras['os_force_miss']=1` and a distinct reason/diagnostic. Keep its proposed action, top-k, scores and confidence for audit. Use the existing `--os-judge guard_only`; the normal policy path executes all K10 denoising steps and records the true MISS.
6. Consume the budget only for an executed intervention. The wrapper can confirm this from the next request's `prev_hit=False` and matching offered step; guard against duplicate offers meanwhile. A failed/reset request must not leak pending state to another episode. Normal MISS anchor invalidation and the following real vision anchor remain in force.

**Plugin/server change:** none is needed for the π0.5 pilot's decision path: `blind_step → LookReason`, `os_force_miss`, and dense execution history already exist. Add arm-local Method code and log scalar `aperture`, expected aperture, matched mass, consensus mass, alarm and intervention count. The provided helper is not a complete installed Method. GR00T needs an analogous wrapper around its supported blind base, with negative meaning close; K7 itself explicitly rejects GR00T and must not be silently reused. A full-model endpoint is necessary wherever the new rule can MISS.

**Cost tier and attribution.** T1 library quantiles, T0 online comparisons, approximately 16-member arithmetic. It changes **control at a fixed library**. Synthesis, retrieval ranking and library content have zero intended contribution. No external model, new vision representation, or borrowed big-library fit is involved. Storage is in §1. Retain a vision anchor in every episode and all existing bounds on blindness.

**Predicted effect at both sizes — assumptions exposed.** Replacing the first alarmed blind HIT with a full MISS costs one unit, giving frozen-path increments **73/31,186=.002341** and **24/28,888=.000831**. Corresponding frozen-path IR is **.243854 / .204112**. These are bookkeeping counterfactuals, not realized-rollout IR predictions; all later path lengths, vision calls and MISSes must be remeasured.

If, purely as a planning scenario, the intervention recovers 20% of reached failures and harms 2% of reached successes, the forecast is:

| π0.5 l10 library | Reference SR | Assumed ΔSR | Scenario SR | Frozen-path IR |
|---|---:|---:|---:|---:|
| 50 | .806 | +.01292 | .81892 | .243854 |
| 500 | .880 | +.00388 | .88388 | .204112 |

Neither rescue rate is estimated by the observational data. Under a coupling that preserves the baseline until the first trigger, even rescuing **every reached failure** gives only **+7.2 pp / +2.2 pp** historical reach ceilings. Unreached task-4 failures remain untouched. At spatial-500 there are **zero alarms**, so the frozen-path prediction is exactly unchanged **.982 @ .128236**. Spatial-50 has no completed matching arm: a neutral ΔSR/ΔIR planning assumption is appropriate, but its absolute point and alarm rate are unknown. GR00T's actual alarm exposure and rescue value are unknown at **both 50 and 500**; do not transfer π0.5 rates. Its IR is `αG v + (1-αG)m`; the historical graph asset gives `αG=.148301`, pending confirmation of the current server basis. A new blind-to-MISS interruption still costs one full unit under either model.

**Cheapest diagnostic and kill criterion.** The offline incidence, task/init controls, sensitivity checks, action followups, and helper parity are complete. Next, inspect/replay the first alarm using actual observations and log the policy's proposed response. Reject the mechanism if command-matched contradictions are mainly harmless thin-object grasps, or if immediate calls do not reduce the observed transport-then-reopen sequence. Do not “repair” it by always calling on closed fingers. Kill the implementation on a changed non-alarm decision, a missing vision anchor, or an intervention budget above one. A realized IR increase above `.01` without a clear SR/recovery benefit is a practical rejection threshold.

**Closed-loop pilot, coordinator only.** Two arms: unchanged K7 anchor-tail and K7+D1, full K10, five controls/decision. Start with π0.5 l10 at **both libraries**. Use all tasks with inits `0..9` only as a collapse check; the adoption comparison is all **10×50 inits `0..49`**, paired, with repeated runs if the difference is small. A cheap mechanism-enriched pilot can instead use the exact **73 / 24 baseline-alarm `(task,init)` pairs** in `contact_episodes_*.json`, selected by alarm rather than outcome. Its SR is conditional and cannot be reported as whole-suite SR. Report first-trigger incidence, subsequent aperture, upward motion, reopen/retry loops, full-episode SR, N/V/M, and actual IR. Confirm spatial at both sizes before deployment; GR00T gets a log-only compatibility/activation audit before any efficacy claim. A single 500-init run cannot resolve the +.39-point 500-library planning scenario against the R5 noise floor.

**Variants.** After the one-call version passes, permit one intervention per newly observed close attempt with an episode cap of two; its calls must all be counted. A diagnostic-only variant asks for vision but lets the unchanged guard decide, separating the value of the extra observation from the forced rescue. Do not initially modify gripper commands, hold the robot in place, reweight candidates, or extend the blind budget.

### D2 — Correlate the noise within a pair of consecutive policy rescues

**Pitch and hypothesis.** On a second consecutive MISS, avoid an entirely independent latent draw while still running the complete conditioned policy. Fixed-observation resampling changes π0.5 l10 gripper signs in 10.25% of comparisons; repeated rescues are common in the remaining failures. Correlation might reduce arbitrary plan changes. It might also preserve a bad choice and hinder recovery. **The evidence identifies a mechanism to test, not a positive SR effect.** This ranks below D1 because its causal link is weaker, despite zero additional inference calls.

**Algorithm/API/server changes.** Keep Method `fit/reset/query`, retrieved actions, guard rules, vision cadence, execution horizon and denoising step count unchanged. Add opt-in per-connection policy-noise state in the plugin/server:

- On the first MISS of a pair use ordinary standard Gaussian `z1`. On the immediately following decision, only if it is also a MISS, use `z2=.9*z1+sqrt(1-.9²)*ε`, with independent standard Gaussian `ε`. The third consecutive MISS starts a new pair. Any executed HIT, episode/task reset, or failed request clears the pair.
- Couple the iid control and correlated arm through a counter-based seed derived from experiment seed, task, init and decision step, excluding port, request-completion order and arm name. Do not call global `manual_seed` per request. The iid control is necessary because existing `k4_eval/seeded_inference.py` seeds a process once and remains scheduling-dependent.
- For π0.5, `_ConnPolicy` can provide a connection-owned prospective `noise` argument on the ordinary inference branch and commit it only after actual MISS execution. `src/openpi/cache/interceptor.py:infer` already accepts explicit noise and passes it to the full stage-3 path; `models_pytorch/pi0_pytorch.py:run_stage3` accepts it too. Reset metadata at lifecycle hooks and log noise checksum, pair position and source. No Method API change is needed.
- GR00T's normal interceptor calls `run_stage3(stage2)` without an explicit noise argument. Add a request-local noise provider at those MISS call sites. `cache/groot/staged.py:run_stage3(noise=...)` uses the pinned explicit-noise denoising loop; test fixed-noise equivalence and graph compatibility. A π0.5 wrapper change alone does not implement GR00T support. Preserve full `(H,32)` noise, including padded action channels consumed by the model.

The marginal Gaussian formula does not prove unchanged state-conditioned policy behavior: states now depend on earlier correlated draws. This is the experimental change being tested. It uses no cached action as a denoising initial condition, no shortened denoising, and no additional policy calls.

**Cost, bytes and four-layer attribution.** T0 sampling/control change. Library bytes and synthesis/method/library effects are zero; per-connection retained bytes are in §1. At fixed paths, `v,m,N` and IR are identical. Realized paths can change both SR and IR. The pair rule would alter **1,749/5,554 MISSes at 50** and **851/3,959 at 500** in the existing l10 tail logs; spatial-500 has only **79/596** eligible MISSes. Report any explicit-noise graph overhead separately; its latency is unmeasured here.

**Prediction at 50 and 500.** Use a neutral planning center, **ΔSR=0**, until the diagnostic below succeeds. Thus l10 reference centers are **.806 @ .241513** and **.880 @ .203281**, with no warranted directional uplift; spatial-500 is **.982 @ .128236**, and spatial-50's matching mixed point remains unknown. These are planning centers rather than new measured results or confidence bounds. On GR00T's completed R2 pure-cache AWM arms the proposal is exactly a no-op at both sizes because there are no MISSes: spatial SR **.888/.966**, l10 **.552/.706**, and pure-cache cost `αG`. GR00T mixed efficacy at either size is unknown; its smaller resampling variation gives less mechanistic motivation than π0.5.

**Cheapest diagnostic, prerequisite and kill.** Coordinator: use 200 adjacent observation pairs from existing token/image-covered traces per tested cell, preserving each current raw state and prompt. For four seeds, evaluate the first observation once and the second with iid versus correlated noise: **12 full policy evaluations per pair, 2,400 total**, with reusable observation encodings accounted separately. Compare noise-induced boundary variability, gripper transitions, and preservation of responses to changed observations. This diagnostic has **not** been run here. Reject before a rollout pilot if the sampling change has little effect or suppresses legitimate transitions; a 20% reduction in the stochastic boundary component is a reasonable screening target, not an SR surrogate. Fixed-noise transport/parity and independent connection state are prerequisites.

**Closed-loop pilot.** Only after that gate: iid counter-seeded control versus correlated-pair variant, unchanged K7 tail on π0.5 l10, both 50/500 libraries, tasks `0..9`, inits `0..49`, with at least two paired seed namespaces if the first comparison is small. A 100-init all-task screen detects collapse only. Hold D1 off. Kill on a repeatable SR loss or absence of improved rescue outcomes; lower action jitter alone is insufficient. Confirm spatial at both sizes if there is a l10 gain. GR00T requires a matched mixed baseline first. Variants after evidence: `ρ=.5` or one identical-noise reuse; no episode-long frozen noise and no candidate averaging.

## 3. Rejected ideas and measured reasons

### R1 — Rebase every cached translation toward the library's mean pose

This was tested separately from R4 C's already-negative local affine action regression. [dynamics_probe.py](dynamics_probe.py) fits the physical map `Δrs_xyz = [1, mean(a[:5,:6])] B` from valid library successor edges, with five-fold whole-episode validation. A new controller could invert its translation block to turn the query-to-kernel position offset into an action correction. [servo_probe.py](servo_probe.py) tests gain `.25`, capped at `.15 σ` translation RMS, retaining rotations and gripper.

| Cell | Own-library forward CV R² 50 / 500 | Cache-trace action RMS before → after, 50 | Before → after, 500 |
|---|---:|---:|---:|
| π0.5 spatial | .95879 / .96063 | .589543 → .584816 | .509371 → .505425 |
| π0.5 l10 | .89293 / .87420 | .516744 → .517298 | .435160 → .439275 |
| GR00T spatial | .96534 / .96060 | .517589 → .513957 | .437336 → .434385 |
| GR00T l10 | .92812 / .89957 | .516453 → .516154 | .445893 → .449658 |

The physical map is learnable, but that does not make the mean demonstrated position the correct target for the current object placement. At l10, **69.945%/61.294%** of π0.5 and **70.124%/61.498%** of GR00T corrections hit the cap at 50/500. The direction has positive alignment with the recorded policy residual only **63.136%/59.295%** and **61.972%/56.238%**, respectively. This top-ten-pose diagnostic is approximate; the base action is exact. It does not rank SR. It supplies insufficient support for broad synthesis changes, particularly at 500, and agrees with R4's weak affine-correction evidence. Reject as the next pilot, not as a proof that every visual servo is impossible.

### R2 — Add a generic actuator-stall or forward-model-innovation guard

The same own-library forward model explains **.908660/.905863 R²** of actual π0.5 l10 tail translation at 50/500. Define a blocked transition as predicted normalized motion above the library median but actual projection along it below 25%; require two consecutive transitions. Before decision 30 this flags only **three episodes, two failed**, at 50, and **five episodes, none failed**, at 500. Later it flags **53 episodes/19 failures** and **35/14**, with existing MISS overlap. It is not an early general failure discriminator.

A broader residual-above-library-CV-p95 rule reaches **258 episodes/61 failures** at 50 and **144/31** at 500, including **197/113 successful episodes**. Spatial-500 reaches **166 episodes**, **158 successful**, while its repeated-blocked alarm never fires. Data: [feedback_logs.json](feedback_logs.json). Spatial-50 and GR00T lack matching R4 state logs, so their closed-loop alarm selectivity is unknown; the forward CV table above covers their own-library fit at both sizes. Do not promote a broadly noisy proprioceptive alarm over K7's already-tested visual guard.

### R3 — Call whenever closed fingers are near the lower aperture endpoint

The conditioning in D1 is essential. [aperture_floor.py](aperture_floor.py) finds near-floor aperture after two closed heads in the following fractions of own-library rows:

| Cell | 50 episodes | 500 episodes |
|---|---:|---:|
| π0.5 spatial | 341/473 = 72.093% | 3,630/5,077 = 71.499% |
| π0.5 l10 | 134/1,016 = 13.189% | 1,481/11,272 = 13.139% |
| GR00T spatial | 354/490 = 72.245% | 3,119/5,249 = 59.421% |
| GR00T l10 | 130/1,013 = 12.833% | 1,681/10,760 = 15.623% |

In actual π0.5 spatial-500 tail logs the unconditional rule reaches **422 episodes, 415 successful**, versus **zero** for D1's matched contradiction. In l10 it reaches **196 episodes/78 failures at 50**, and **158/43 at 500**. Calling once per reached episode adds frozen-path IR **.005068/.004349** in l10 and **.035530** in spatial-500, before any downstream path change. Near-floor width is neither a universal failed-grasp label nor a reason to rewrite gripper commands. This rejects an unconditional contact heuristic at both sizes and both models; it does not claim unmeasured GR00T closed-loop rates.

The remaining decisive facts are intervention outcomes for D1, paired-observation noise response for D2, and state/action logs from completed spatial-50 and GR00T mixed controls. Offline action error cannot supply any of them.
