# R4 ideation C, second pass: measure the value of a policy intervention

**Recommendation: one additional research proposal, `causal_rescue_credit`, below.** The new diagnostics do not justify another deployed pure-cache controller yet. I investigated recovery, policy/cache cooperation, local tokens, GR00T-specific action neighborhoods, and cross-task/suite transfer. Several initially promising ideas weakened when measured at both library sizes or with episode-balanced denominators. I have not filled the proposal quota with them.

This is independent of the first pass. `REPORT.md` and its original scripts/outputs are unchanged. This report does not re-propose its control-step library, pilot allocation, phase gates, affine synthesis, or successor-aware whitening. No GPU, simulator worker, server, remote host, git command, or subagent was used. The hot store is absent; reads used `/home/weiland/trace_runs/offline_search_store`. All new files are local `p2_*` files plus this report; no large array was written.

## 1. Measured facts driving the proposal

### Sources, reproduction, and what the data identifies

I read the first report and its scripts/outputs, the specified R4/R3/R2 findings, protocol §§8–10, brief, harness/plugin documentation and implementation, AWM/AWM3/MixedJudge/V6/V7, and the warm-start analysis. I also checked the R1/R2 token findings before considering a new token method.

Run the scripts from the repository root, with this prefix:

```bash
taskset -c 24-37,68-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_C/SCRIPT
```

| New script | Output and coverage |
|---|---|
| `p2_inventory.py` | `p2_inventory.json`: schemas, task descriptions, tokens, mixed inputs/log example |
| `p2_recoverability.py` | `p2_recovery_<arm>.json`, `p2_recoverability_summary.json`: **14 AWM-based mixed arms, 7,000 accepted episodes**; four workers |
| `p2_recovery_landmarks.py` | `p2_recovery_landmarks.json`: episode-balanced conditional outcomes and optimistic spending ceilings |
| `p2_cross_task.py` | `p2_cross_task_<model>_<50|500>.json`: both models, both library scales, inf/cache observations, other-task and spatial donors; four workers |
| `p2_tokens.py` | `p2_tokens_<model>_<suite>.json`, `p2_token_rows_*.npz`: **200 actual B0 closed-loop episodes, 10,880 token-bearing decisions**, replaying AWM candidates at both scales; four workers |
| `p2_token_projection.py` | `p2_projection_*.json`: fixed-seed 32-channel compression and soft reweighting at decisions 5/10, all eight library cells; four workers |
| `p2_kernel_variance.py` | `p2_kernel_*.json`: first actual AWM spell in each of **4,000 R2 episodes**, eight cells; four workers |
| `p2_summary.py` | `p2_summary.json`: compact tables and coverage |
| `p2_bytes.py` | `p2_bytes.json`: exact pickle bytes and analytic representation additions |
| `p2_token_early_check.py` | `p2_token_early_check.json`: decision-5 signal before the first B0 spell |
| `p2_validate.py` | `p2_validation.json`: counts, shapes, schema parsing, cost formulas and output-size checks |

Dependencies: recovery precedes landmarks; the final summary follows tokens/cross-task/kernel. The projection diagnostic can follow tokens independently. `p2_bytes.json` contains exact filesystem byte counts and the displayed storage arithmetic. The final recovery run is `p2_recoverability_corrected.log`; the earlier error logs document two corrected input-schema assumptions, not partial results used below.

Accepted journal attempts are joined to server decisions and deduplicated by `(uid, step)`, using the first pass's read-only parser. All action diagnostics use the executed `5×7` block. Policy actions are the logged `a_exec`, where present. Cached actions reconstructed from logs use the logged top **10** of **16** kernel members; these are approximate, not the exact served head. The first report already quantified this reconstruction limitation, so I did not repeat that experiment.

Three quantities must remain distinct:

1. **Paired episode rescue:** an init fails in CL2 and succeeds in a mixed arm. This identifies a whole-controller outcome difference on that init, including stochastic rollout differences.
2. **Eventual success after a MISS:** an observational conditional probability. It is not the probability that this particular call caused rescue.
3. **Short-term return:** within six decisions, success occurs or three consecutive HITs occur with library progress more than .03 above the intervention's progress. This is a retrieval/control proxy, not task completion. It depends on the controller's allowed HIT runs and must not compare periodic k=3 against other controllers.

**Individual causal rescue probability is not identifiable from the existing logs.** There is no randomized call/no-call branch from the same intervention distribution, and low confidence/guards choose which states receive a call. We can measure conditional outcomes, support and spending; we cannot honestly turn them into an intervention-benefit label.

### A. First-call recovery depends on context, and confidence is not a rescue-value score

Reproduced π0.5-l10 arm totals (`p2_recovery_*.json`):

| Library | Arm suffix | Success / 500 | Decisions | MISSes | Formula IR | CL2 failures rescued / CL2 successes lost |
|---|---|---:|---:|---:|---:|---:|
| 50 | `g` | 370 | 32,967 | 6,664 | .323416 | 80 / 25 |
| 50 | `awm_h70` | 408 | 30,385 | 10,436 | .443253 | 122 / 29 |
| 500 | `g500` | 432 | 29,340 | 2,983 | .238216 | 60 / 12 |
| 500 | `awm500_h70` | 436 | 29,046 | 9,905 | .441177 | 70 / 18 |

The 500-library confidence arm spends **6,922 additional MISSes** over guard-only for **four additional successful episodes**, with 294 fewer requests. This is an arm-level comparison, not a marginal price for individual calls. It illustrates why an error-prediction objective can be poorly aligned with inference value.

The next table uses **only the first MISS of each interrupted episode**, so long failures do not contribute multiple rows. Entries are eventual successes / episodes at that first intervention. `gexec` is the previously executed gripper sign; π0.5 positive means closed. Progress is the selected library row's normalized progress, not observed task completion.

| First-MISS context | `g`, 50 | `g500`, 500 | `awm_h70`, 50 | `awm500_h70`, 500 |
|---|---:|---:|---:|---:|
| Before first observed stall | 25/28 | 44/45 | 198/228 | 277/299 |
| At 0–2 decisions after first stall | 312/439 | 306/373 | 196/258 | 152/194 |
| Progress < .25 | 88/138 | 107/144 | 155/195 | 199/232 |
| Progress ≥ .75 | 75/83 | 90/92 | 40/47 | 35/35 |
| Gripper open | 154/239 | 172/213 | 181/234 | 203/236 |
| Gripper closed | 183/228 | 178/205 | 213/252 | 225/256 |
| Decision < 20 | 134/199 | 132/160 | 248/304 | 260/292 |
| Decision 20–39 | 156/214 | 151/187 | 126/156 | 154/183 |

Stall onset here is the first logged `stuck_n≥2` **or** `noprog_n≥2`, matching the current no-progress implementation's three-observation condition. It is not a simulator object-state label. Later first-call timing can indicate an easier episode that needed no early intervention; the table does not imply “wait longer.”

Conditioning further on **CL2 failure** changes the interpretation. In the confidence arms, first intervention before the mixed episode's first stall accompanies **60/80 = 75.0%** paired rescues at 50 and **31/44 = 70.45%** at 500. At/just after stall, the corresponding fractions are **61/104 = 58.65%** and **39/72 = 54.17%**. These are compatible with a recoverability window, but selection and different prior trajectories can produce the same pattern. They are not causal uplift estimates.

For later calls, `p2_recovery_landmarks.py` keeps only the **first MISS-run start per episode within each reported bin**. An episode may enter several bins, but never counts repeatedly inside one bin:

| Time since first stall | `g`, 50: eventual success | `g500`, 500 | `awm_h70`, 50 | `awm500_h70`, 500 |
|---|---:|---:|---:|---:|
| 0–2 decisions | 312/441 = .7075 | 308/375 = .8213 | 288/362 = .7956 | 235/289 = .8131 |
| 3–9 | 131/199 = .6583 | 93/133 = .6992 | 204/257 = .7938 | 180/221 = .8145 |
| ≥10 | 214/342 = .6257 | 184/252 = .7302 | 278/368 = .7554 | 266/326 = .8160 |

The crude decision-weighted late-call success fractions would have been only **516/1,783=.2894** for g50 and **439/1,111=.3951** for g500. Repeated unsuccessful attempts badly bias that reading. Even the episode-balanced table has survivor/selection bias, but it refutes a simple universal “late means hopeless” interpretation.

**Confidence scale does not transport directly across library sizes.** At a MISS-run start with `conf≤−.6`, g50 has **7/92** eventually successful episodes and awm_h70 has **0/40**; the Wilson upper bound for the latter is **.0876**. Neither 500-library arm has any MISS-run start in that bin. A threshold inferred at 50 therefore has no measured 500-library support. At the *first* intervention, high confidence `>−.3` accompanies 110/151 successes in g50 and 294/353 in g500: confidence alone is neither necessity nor benefit of calling the policy.

The JSON also reports descriptive phase bins from changes in the executed decision-boundary gripper sign, capped at ≥3. For example, g50 first-call success is **81/145** before the first sign change, **140/179** after one, **116/143** after ≥2; g500 is **75/109**, **120/144**, **155/165**. These are event-count descriptions, not validated subtask phases or a proposed gate. Within-chunk flips are not counted in this particular feature.

### B. Long MISS runs are not universally futile; a conservative futility rule has little 500-library leverage

One landmark per episode: first time a consecutive MISS run reaches the specified length.

| Arm | Success after reaching 5 MISSes | After 10 | After 20 |
|---|---:|---:|---:|
| g50 | 56/166 | 28/85 | 14/39 |
| g500 | 12/48 | 4/20 | 1/4 |
| awm_h70, 50 | 133/219 | 37/115 | 10/59 |
| awm500_h70, 500 | 240/295 | 83/126 | **23/50** |

This rejects a universal retry count inferred from failed tails. In particular, nearly half the 500-library confidence-arm episodes reaching 20 consecutive calls still succeed.

I also replayed a more selective **diagnostic**, not a proposed controller: first reach ≥10 consecutive MISSes, maximum retrieved progress gain since the run began <.03, and current `stuck_n≥2`. Count all remaining historical MISSes as if they could be removed without changing the trajectory. This deliberately optimistic ceiling is:

| Arm | Episodes flagged | Observed successes flagged | Historical suffix MISSes | Fixed-path IR saving ceiling |
|---|---:|---:|---:|---:|
| g50 | 35 | 6 | 667 | .017157 |
| g500 | 4 | 1 | 45 | **.001301** |
| awm_h70, 50 | 42 | 3 | 1,164 | .032486 |
| awm500_h70, 500 | 6 | 2 | 137 | **.004000** |

Actual switching changes future states, requests and opportunities to call again; these are not achieved savings or SR bounds for a new rollout. The measurement supports studying *value of another attempt*, but not shipping this futility heuristic.

### C. What the policy changes, and why warm starts do not currently supply the missing cooperation mechanism

At MISS-run starts with `a_exec`, compare the logged policy head to the reconstructed cached mean and to its best logged library member. These are event-weighted action diagnostics, not success predictors fitted on independent episodes:

| π0.5-l10 group | 50: mean-cache / best-member error to policy | 500: mean-cache / best-member | Policy/cache gripper disagreement, 50 / 500 |
|---|---:|---:|---:|
| g, eventually successful | .4554 / .3260 | .3186 / .2093 | .1082 / .1100 |
| g, eventually failed | .6600 / .5068 | .3960 / .2705 | .2591 / .1244 |
| confidence arm, successful | .3694 / .2714 | .2955 / .1936 | .0538 / .0554 |
| confidence arm, failed | .5411 / .3884 | .3253 / .2054 | .2533 / .1019 |

Units are current-library action σ, all seven valid dimensions. “Best member” uses the observed policy action to choose retrospectively; it is an oracle support check. It does **not** establish that snapping a policy action to a cached chunk improves success. Failed trajectories can disagree much more with the cache, but a useful correction and an unrecoverable state can both cause that disagreement. A gripper-only cooperation rule would also ignore most successful intervention steps in these logs.

The existing warm-start study, `exp/step_diag/analysis/step_vs_warmstart.md` §6.15, is directly relevant:

- π0.5 LIBERO exact continuation is **.926/.778** spatial/l10 versus full **.986/.844**; reset variants do not systematically beat equal-step plain inference.
- GR00T-l10 exact continuation is **.712** at one step and **.818** at two versus the study's full **.880**. Cached reset variants range **.826–.868**, with no significant improvement over matched-step plain inference.
- Those are that study's libraries/checkpoints, including its GR00T `libraries_w13` setup; they are **not** new 50-versus-500 AWM-library experiments. The 500-library warm-start counterfactual is unknown. No claim about its SR is extrapolated here.

Thus I do not propose a warm start, coordinate-wise policy/cache action splice, or teacher-action projection on the strength of an oracle L2 reduction. The cooperation proposal below uses the policy as an intervention whose **task-level effect** is learned; it does not blend or reinitialize its actions.

### D. Cheap token signals: an interesting GR00T lead, not a validated early guard

Mixed-run inventory finds **zero saved input NPZs** in r03_mx. No token histories can be recovered from those JSONL decision rows. The available token-bearing closed-loop data is B0's original cache trajectories: 50 episodes/cell, inits 0/10/20/30/40 on all ten tasks. AWM's nearest rows are replayed on those fixed observations at each library scale. **The 500-column changes the reference library, not the controller that produced the trajectory.**

New signal: the 95th percentile over 256 aligned wrist-image token positions of `1−cos(query_token, selected_row_token)`. Comparator: mean cosine residual over the existing 16 pooled positions. This is a local mismatch statistic; it does not require an external model or another stage-1 pass.

At decision 5, predicting eventual B0 failure:

| Cell | Library | AWM distance AUC | Wrist patch-p95 AUC | Wrist pooled-mean AUC | Patch-p95 within-task AUC |
|---|---:|---:|---:|---:|---:|
| π0.5-l10 | 50 | .4550 | .5650 | .5383 | .5833 |
| π0.5-l10 | 500 | .5233 | .5167 | .4917 | .5000 |
| GR00T-l10 | 50 | .5781 | **.7456** | .6908 | .6429 |
| GR00T-l10 | 500 | .4493 | **.6957** | .6409 | .6190 |

There are only **36/42** failure-success pairs within the same task on π0.5/GR00T-l10. The GR00T result is suggestive, not precise. For GR00T's 45 episodes whose first B0 repeated-pick spell starts at decision ≥7, decision-5 patch AUC is **.7134/.7115** at 50/500; this precedes that spell by at least two decisions, but is not evidence of preceding all MixedJudge guards. The deployed mixed trajectories lack the inputs needed to establish that comparison.

I tested a storage-feasible implementation before proposing a new key: unit-normalize each 2,048-channel token, multiply by a fixed Gaussian `2048×32` projection (seed 20260927), store float16 projected tokens, and use the patch-p95 of half squared projected distance. This adds **16,384 B/row + 262,144 fixed bytes** for one camera.

On GR00T-l10, full versus compressed mismatch rank correlation at decisions 5/10 is **.8740/.8637** at 50/500. Decision-5 failure AUC falls to **.6747/.6006**. Softly reweighting the logged top-10 kernel by `exp(−distance / median_candidate_distance)` changes its highest-weight row on **17%/23%** of those observations, but changes the served mean only **.00927/.00726 σ RMS**. On π0.5-l10, corresponding action changes are .02711/.02190 σ; the uncompressed early failure signal there was weak already.

This investigation also respects the existing R1 result: generic full-token cosine/MaxSim/Chamfer and high-variance patch selection did not beat pooled keys offline (`r01/ANALYSIS.md` §2.4; `r02/FINDINGS.md`). The new measurement is outcome prediction with a residual-tail statistic, not evidence overturning that retrieval result. I do not recommend a new token retrieval arm on the present sample.

### E. GR00T's gap: measured differences, and limits of the explanation

The method-layer premise needs a qualification: **spatial has the same +3.6 pp method effect on both models at 50 episodes**. The large difference is l10: +20.2 pp on π0.5 versus +8.6 pp on GR00T, and +23.0 versus +3.4 at 500 (R2 three-layer table). GR00T's smaller policy noise floor and its padding are already established; all diagnostics here exclude the padding.

`p2_kernel_variance.py` adds an episode-balanced check of the first real CL2 repeated-pick spell. It estimates the local action support from the logged kernel, not from a newly chosen candidate pool:

| l10 first spell | π0.5 50 / 500 | GR00T 50 / 500 |
|---|---:|---:|
| Failed episodes with a spell | 184 / 115 | 224 / 146 |
| Failed spells where mean translation retains <½ of weighted member translation norm | 5/184 / 3/115 | **31/224 / 25/146** |
| Successful spells with the same cancellation | 4/171 / 3/97 | **30/164 / 41/136** |
| Failed spells: mean kernel mass disagreeing with mean head on any gripper step | .2272 / .2776 | .2806 / .3640 |
| Successful spells: corresponding disagreeing mass | .1815 / .2770 | .2627 / **.4574** |

GR00T l10 neighborhoods have more cancellation and categorical ambiguity. Crucially, these features are at least as common in **successful** GR00T spells at 500. A global anti-cancellation or gripper-consistency rule therefore lacks the specificity it would need. This is consistent with, and does not reopen, the dead gripper rules.

I also measured whether the local kernel even contains a materially different demonstrated motion: continuous-head difference >.3 σ from the mean **and** that member's recorded next-state translation exceeds its own library's median. Kernel mass on such alternatives is ≥.1 in only **39/184, 14/115** failed π0.5 spells and **78/224, 21/146** failed GR00T spells, at 50/500. Its median is zero in all four cells. Under a hypothetical 10% whole-chunk lottery, the average probability of sampling such an alternative within ten decisions is only **.0656/.0350** for π0.5 and **.1110/.0362** for GR00T, holding the neighborhood fixed. This is support arithmetic, not an escape probability. I do not propose another library-side recovery mechanism.

Together, the live-log evidence favors **ambiguous control neighborhoods plus state-dependent recoverability**, not “GR00T needs a stronger global metric” as an explanation. The local wrist mismatch result suggests some missing visual detail, but its small sample and compression loss prevent a causal conclusion. These diagnostics do **not** fully explain the model-by-method interaction or prove a GR00T fix. No GR00T mixed controller is proposed; that remains R5's territory.

### F. Shared objects do not automatically make transferable control neighborhoods

Task names in the store identify plausible shared-object pairs: basket tasks 0↔7 and 1↔7, mugs/plate 4↔6, and stove/moka-pot 2↔8. These labels define only candidate donor sets; they do not tell us which subskill is active. Spatial donors add many bowl-placement trajectories but different scenes/goals.

`p2_cross_task.py` samples every fifth recorded decision, uses the l10 library's own PCA/main AWM metric, projects spatial rows into that same space, and compares nearest rows in the own-task, other-task, shared-object, and spatial sets. This deliberately uses the **main metric and nearest rows**, not AWM's special step-0/fresh branches or full kernel synthesis. Both inf and cache results are saved. The table below is cache observations:

| Model/library | Other-task row wins | On those queries: own → donor error | Spatial row wins | On those queries: own → donor error |
|---|---:|---:|---:|---:|
| π0.5, 50 | 359/8,177 = 4.39% | 1.0239 → .9421 | 144/8,177 = 1.76% | 1.2515 → 1.2872 |
| π0.5, 500 | 192/8,177 = 2.35% | .6479 → .8910 | 22/8,177 = .27% | .6747 → 1.6039 |
| GR00T, 50 | 307/8,078 = 3.80% | .8297 → 1.0213 | 131/8,078 = 1.62% | .9052 → 1.2679 |
| GR00T, 500 | 247/8,078 = 3.06% | .5755 → .8005 | 10/8,078 = .12% | .9620 → 1.0527 |

The shared-object restriction wins even less often: π0.5 **48/6,380 and 15/6,380**, GR00T **66/6,101 and 41/6,101**, at 50/500. On GR00T task 8, borrowing task 2 wins **0/994** queries at both scales; allowing its action oracle improves the same-task oracle by only **.0045/.0023 σ**. On task 7 there is more overlap: **52/945 and 38/945** donor wins, with oracle headroom .0325/.0137 σ. This is the most plausible targeted pair, but it still lacks an online selector for the correct subskill.

The unrestricted action oracle over other l10 tasks does improve: π0.5 **.27358→.22894 / .18785→.16390**; GR00T **.25527→.22046 / .17514→.15696**, at 50/500. Thus reusable motions exist. The failure is connecting those motions to the correct current object/goal, not a proof that transfer is impossible. Offline error does not rank SR; the reason to decline the naive transfer pilot is its rare intervention, conflicting actions on selected states, and missing subskill correspondence.

### G. Library and inference accounting

All MB are decimal. Baseline AWM pickle bytes come from the first pass's actual file measurements; new wrapper/token arithmetic is in `p2_bytes.json`.

| Cell | Episodes / entries, current | Episodes / entries, big | AWM pickle MB, 50 / 500 | Deployed pkl MB |
|---|---:|---:|---:|---:|
| π0.5-sp | 49 / 1,018 | 500 / 10,909 | 21.342 / 46.993 | 431 |
| π0.5-l10 | 50 / 2,640 | 500 / 29,472 | 24.631 / 94.143 | 1,103 |
| GR00T-sp | 50 / 1,063 | 500 / 11,751 | 22.249 / 58.157 | 429 |
| GR00T-l10 | 50 / 2,645 | 500 / 29,631 | 26.673 / 117.304 | 1,068 |

AWM retrieval representation: **580 B/entry**, action payload **280/448 B** for valid π0.5/GR00T full chunks, or 140 B for the executed head. Fixed overhead is approximately 17 MB PCA plus 1.5 MB transforms. Shipped pickles include padded chunks/auxiliaries. MixedJudge's representation is **626 B/entry**, with additional wrapper arrays in its pickle. Actual π0.5 mixed pickles are **26.036/66.084 MB spatial** and **32.602/141.227 MB l10**, at 50/500; the 500 spatial number is the existing confidence arm, not a newly run g500 arm.

A full float16 wrist-token table alone costs **2.773 GB / 31.070 GB** on GR00T-l10 at 50/500. The tested 32-channel table would make total pickles **70.271/603.040 MB**, below deployed 1,068 MB, but much larger than AWM and with the signal loss in §D. π0.5-l10 equivalents are 68.147/577.275 MB versus deployed 1,103 MB. No such deployment artifact was built.

π0.5 IR throughout is **`.152 + .848·m`**, with m the realized MISS share. Pure-cache variants retain .152 at both scales. GR00T's policy-stage split is not established by these pure-cache logs, so its pure-cache IR remains `α_G=s1/(s1+s23)`; no numeric GR00T IR is invented. The proposed experiment uses only π0.5. All policy calls, including unsuccessful rescue attempts and calls induced later by an intervention, count as cost.

## 2. Ranked additional proposal

### 1 — `causal_rescue_credit`: learn which call changes the episode, not which action looks inaccurate

**Pitch.** Run a small, randomized intervention experiment, then use a closed-form table of *incremental success and future inference cost* to decide whether a particular rescue opportunity earns a policy call. This is a new objective and evidence source for MISS placement/count, not another periodic interval, HIT cap, event burst, or fixed guard threshold.

**Status.** Research proposal with a concrete controller contract; not a fitted or validated controller. It is the only additional proposal I recommend from this pass. Its expected benefit is better allocation of existing calls; **the current logs do not support an honest positive causal SR estimate**. A default planning point is ΔSR=0. The experiment should earn a deployment proposal, rather than treating eventual success as a rescue label.

**Mechanism and supporting facts.** The first-call and retry tables show potentially useful heterogeneity, but also strong selection bias and scale dependence. A low-confidence failure predictor may spend more precisely on states from which neither policy nor cache can finish. Conversely, some long policy runs are necessary. Learning `P(success | state, call)` alone confuses these cases. The target should be the difference between call and no-call outcomes under the same downstream controller, including the calls they induce later.

**Precise first experiment and Method algorithm.**

1. Start from the existing π0.5 AWM + guard-only controller at each scale. Its selection/synthesis and stage-1 observation stay unchanged. The experiment is about interventions, not improving this baseline by decree. The existing periodic/confidence arms remain frontier references, owned by B.
2. Before each episode, use the episode's experiment RNG to assign a landmark class, equally likely: the **first** or **third** decision at which the baseline would request a MISS. This creates onset and retry observations without using future success or episode length. If the episode never reaches its assigned landmark, it is an unexposed episode and its behavior is unchanged.
3. At that landmark only, independently randomize with probability .5: **CALL**, execute the full policy as usual; or **CACHE**, execute that decision's ordinary AWM proposal. The baseline controller resumes at the next decision and may immediately request another MISS. This compares “call here” against “use the cache here,” not permanent policy withdrawal. There is at most one randomized action per episode.
4. Log the available context before randomization: progress `<.5/≥.5`; gripper sign; confidence `>−.3/≤−.3`; and stall age `before/0–9/≥10`. These define 24 contexts, or **48 bins including the two landmark classes**. The initial fit should pool sparse cells aggressively. The phase-event count, task and raw time are diagnostic covariates, not extra fit dimensions in the initial table.
5. At episode end, research-side labels are success Y, total requests N, and total MISSes M. Estimate within-context differences `ΔY, ΔN, ΔM` for CALL minus CACHE by inverse-probability means (known propensity .5), with episode/init-clustered uncertainty. Start with the two landmark-class parent estimates; shrink a child estimate toward its parent with weight `n/(n+100)`, and retain the baseline action for unsupported cells. Freeze this recipe before outcomes. **Any table fitted from these rollout outcomes is “borrowed big-library information”: the fit uses data beyond the deployed action library.** Fit separate tables at 50 and 500.
6. For an inference-ratio target ρ, the incremental normalized cost contribution is `ΔCρ = .848·ΔM + (.152−ρ)·ΔN`. The request term matters: an earlier completion changes the IR denominator. Choose CALL when the conservative estimated utility `ΔY−λ·ΔCρ` favors it, otherwise CACHE; unsupported bins retain the baseline. Select λ on training folds, then freeze it. Do not optimize λ on the confirmation outcomes.
7. The **first learned controller changes only the same single assigned landmark**, preserving the experiment's causal support. Applying the table at every decision would change the downstream intervention distribution and is a separate sequential-control experiment, not a justified extrapolation from this trial. It must collect new randomized support before promotion.

The 48-bin representation is a storage bound, not a claim that 500 episodes can precisely fit 48 independent treatment effects. At the proposed initial sample, the two parent effects and a few well-populated splits are the realistic estimands. “No supported split” is a legitimate outcome.

**Plugin/server changes, exactly.** Add an opt-in experimental verdict override in `PluginSession`/`PluginJudge` **after** the ordinary guard-only verdict is computed and **before** returning HIT/MISS to the interceptor. Keep a per-episode count of *baseline MISS opportunities*, sample landmark/assignment at reset, and override only that opportunity. On CACHE, fetch the existing `Result.action`; on CALL, use the current full-policy path. `on_executed` must receive the actual verdict/action so `q.hist_hit`, `q.prev_a_exec`, fresh/stale selection, and all subsequent guards remain truthful.

Log `trial_id`, landmark class/index, eligible, assigned treatment, propensity, baseline verdict, actual verdict, the 48-bin context and seed. Also log the actual served `[:5,:7]` on HITs so later analysis needs no top-10 reconstruction. The Method's `query()` can return context and baseline scores in `extras`; random assignment and the learned override belong in the plugin because it knows the final verdict. The offline harness cannot estimate this outcome effect: replay changes no future state. It can verify eligibility/reset/propensity and action contracts only. The server must load the full model as today. No denoising, warm-start, encoder, or library mutation is involved.

**SR/IR forecast at 50 and 500.** These are low-confidence planning forecasts for the *single-landmark learned overlay*, not measured effects and not confidence intervals:

| Cell/library | Measured guard-only reference | Planning point for overlay | Plausible useful result to investigate |
|---|---|---|---|
| π0.5-l10, 50 | SR .740, IR .323416 | SR .740, IR approximately .318 | SR within 1 pp, IR lower by .005–.015 |
| π0.5-l10, 500 | SR .864, IR .238216 | SR .864, IR approximately .235 | SR within 1 pp, IR lower by .002–.008 |
| π0.5-sp, 50 | SR .888, IR .2664 | No new arm initially; zero claimed effect | Transfer only after a supported l10 effect |
| π0.5-sp, 500 | Guard-only result unavailable | No numerical SR forecast | Must collect the missing same-scale reference if transferred |
| GR00T, both suites, both scales | Pure-cache SRs .888/.552 at 50; .966/.706 at 500 | Unchanged; no mixed proposal | IR `α_G`, no claimed change |

For cost intuition, removing exactly one MISS from every g50 episode while holding paths fixed would save only `.848×500/32967=.01286` IR; at g500 it would save `.01445`. The actual overlay reaches only some episodes and can induce a replacement call later. A large claimed saving requires measured downstream changes, not just counting the overridden call.

This does **not** pretend that slightly improving g50 beats the current frontier: existing periodic k=5 already has SR .792 at IR .3149. The single-landmark trial is an identification experiment. A broader learned judge is worth implementing only if its measured context effects plausibly clear the existing frontier at both scales. A −.005 IR result on an otherwise dominated g50 arm is not an R4 win.

Offline error/regret/bad-rate need not improve; the returned HIT action and the full MISS policy are unchanged. AURC for eventual failure also need not improve. The relevant scores are paired/causal ΔSR, ΔM, ΔN, and aggregate IR.

**Cost tier and bytes.** T1 counting, shrinkage and a tiny table; no neural fit. Five float32 quantities per context—effect estimates/count summaries—cost **960 bytes** for 48 bins, plus small configuration metadata. A production table may store uncertainty too; allow **<4 KB** total fixed overhead. No new per-entry array: MixedJudge remains 626 B/entry and the valid action payload remains 280 B for π0.5. Measured l10 wrapper pickle plus the minimal 960-byte table is **32,603,457 B / 141,227,807 B** at 50/500, against deployed **1,103 MB**. Spatial minimal equivalents are 26,037,097 / 66,085,200 B versus 431 MB, subject to the missing g500 spatial reference. GR00T libraries remain the baseline sizes in §1G. Query work is one small table lookup; wall-clock impact is unmeasured and system benchmarking is outside this round.

**Four-layer attribution.** Synthesis: unchanged AWM mean and full policy chunk. Method at fixed library: baseline retrieval unchanged; a new decision objective is isolated in the plugin. Library: unchanged rows and fit, repeated separately at 50/500. Control: the sole intended causal change; every direct and downstream policy call is charged. No library growth/pruning or GR00T mixed-mode work is hidden in this proposal.

**Cheapest diagnostic.** The observational diagnostic is delivered. The next necessary measurement is **randomized CALL versus CACHE at one live landmark**; no additional replay or action-error fit can identify the missing counterfactual. Before full collection, the coordinator can use 10 tasks × inits 0–9 at each scale to check assignment balance, treatment compliance, missing-landmark rate and gross collapse. Those 100 episodes cannot rank a small SR effect.

**Closed-loop pilot design.** The reviewable specification is `p2_causal_pilot.json`, explicitly marked proposal-only. Only the coordinator runs it:

- Primary cells: π0.5-l10, **both** 50 and 500 libraries.
- Two complementary randomized replicas per scale, all ten tasks × inits 0–49: **4 new arms × 500 = 2,000 episodes**. Landmark class is fixed per init; replicate two flips the CALL/CACHE assignment. Neither arm sees the outcome before acting. Prefix differences from stochastic execution are logged and enter the paired uncertainty; exact simulator-state equality is not assumed.
- Continue the same guard-only controller after the one intervention. Do not add a burst, candidate mask, new key, or policy-action modification. Randomization applies even when the baseline says MISS; that is the experiment's treatment, not a failure to obey the guard.
- First report intention-to-treat aggregate outcomes and exposure counts. Then report context/landmark effects with init-clustered intervals; sparse cells remain pooled. Frozen five-fold init splits can supply out-of-fold table decisions, but **a new closed-loop run of those decisions is still required**—off-policy labels alone do not confirm the learned controller.
- Confirmation: one baseline and one frozen-overlay run across 500 inits per scale, with fold-specific tables trained without those init outcomes if reusing the A-pool. Folds are ops-side experimental assignments, not an init feature available to the controller. Report pooled paired SR/IR and task regressions. Refitting a final deployment table on all data is distinct from the evaluated cross-fitted policy and must be disclosed.

**Kill criterion.** Stop this line if the randomized data cannot demonstrate stable heterogeneity in CALL benefit/cost beyond the parent estimates, if the learned preference reverses on held-out inits, or if the confirmation has >1 pp SR loss without an independently compelling frontier gain. A 1 pp noninferiority claim will generally need more data than a single 500-init comparison; an inconclusive interval is not success. For promotion, require a measured frontier improvement at both scales, not merely fewer calls in historically failed episodes. Do not label a value model learned solely from observational eventual-success labels as causal.

**Variants.** Only two initial variants are principled: first versus third baseline MISS opportunity; and pooled two-parent estimates versus supported context splits. A multi-decision value controller and token-based context are later experiments requiring fresh support, not extra arms slipped into this pilot.

## 3. Rejected additional ideas, with measured reasons

### 1. Stop spending after a generic number of failed-looking policy attempts

Rejected as a deployable rule at both scales. The 20-MISS landmark still succeeds in **10/59** confidence-arm episodes at 50 and **23/50** at 500. The more conservative “10 calls, stuck, no progress” rule flags three/two observed successes and has optimistic IR ceilings **.03249/.00400**; on the efficient g500 arm the ceiling is only **.00130**. Repeated-decision weighting exaggerates late failure, as §A shows. Storage would be negligible and IR arithmetic favorable only on fixed old paths, but the success cost and vanishing 500-library opportunity reject the heuristic. This does not reopen B's fixed schedules or the dead event/burst arms.

### 2. Unrestricted or name-based cross-task / cross-suite retrieval

Rejected as the next closed-loop arm, not as a universal impossibility of subskill transfer. Shared-object candidates win only **.752%/.235%** of π0.5 cache queries and **1.082%/.672%** of GR00T queries at 50/500. On GR00T task 8 they win zero queries at both scales; the donor action-oracle improvement is .0045/.0023 σ. Cross-suite winners become almost absent at 500 and usually disagree more with the teacher than the own-task winner. Adding another suite also changes the library layer: it means **49–50+50** or **500+500** collected episodes, not a free method gain. A fit or donor use beyond the originally deployed task library must be disclosed as **borrowed big-library information**. IR remains .152/`α_G`; extra shared action/key payload would be counted, and a separate shared-object correspondence diagnostic is needed before paying for a pilot.

### 3. A generic unpooled-token alarm or a larger token retrieval dictionary

Rejected for deployment now. Full wrist patch-p95 is promising specifically on GR00T-l10, but there are only 50 B0 episodes and 42 within-task outcome pairs; there is no mixed-loop token evidence and no demonstrated lead over the complete current guard set. A fixed, practical 32-channel compression lowers its decision-5 AUC **.7456→.6747** at 50 and **.6957→.6006** at 500, while the tested reweighting barely changes the mean action (.00927/.00726 σ). Uncompressed wrist tokens require **2.773/31.070 GB** on GR00T-l10 versus deployed **1.068 GB**; the compressed 70.271/603.040 MB alternative fits but has not earned the bytes. π0.5's corresponding uncompressed signal is .5650/.5167 AUC. Existing R1 generic token-retrieval negatives provide further reason to demand an incremental, outcome-grounded result first.

The retained lead is narrow: log this residual as a shadow statistic in future authorized GR00T pure-cache runs, including both scales, to test whether it predicts failure *within tasks and before the first current guard*. That would establish the missing fact; it is not a fourth proposal, a new GR00T mixed arm, or a claimed SR gain.
