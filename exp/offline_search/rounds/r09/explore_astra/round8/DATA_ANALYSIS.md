# Data analysis — R9 astra round 8

## Population, sources, and units

The only full-run outcome sources are `r09_recipe_full_g/r9eq_groot_l10_50`, `r09_recipe_full_p/r9eq_pi05_l10_50`, and `r08_main/{r8_groot_l10_P10,r8_pi05_l10_P10}`. Each contributes exactly tasks 0–9 × inits 0–29. `safe.py` lexes top-level task/init and UID scalars before decoding any outcome payload; nested/escaped identities cannot admit a row, and duplicate/conflicting identities fail closed. Forbidden root prefixes are checked before and after resolving paths. Only accepted, error-free terminal attempts are joined, and decision sequences must be contiguous with no conflicting duplicates. No full-500 summary, full-population dataframe, or mixed-log hash is used.

P10 actions come from existing compact debug NPZs. Only task/init members are opened first; any archive containing an inadmissible identity is rejected before action/state members are accessed. Admitted decision IDs are checked against the accepted attempt. Corrector auditing uses the similarly admitted R8 A (plain cache), CU (uniform randomized calls), and IP (independent randomized calls) compact trajectories. No raw mixed NPZ action member is loaded. Library actions and the frozen recipe artifacts are existing deployment inputs. Task descriptions come from exactly one admitted init-0 P10 episode per task.

Fit/descriptive training split = 0–19, evaluation = 20–29. Pooled 0–29 tables are descriptive, never threshold fitting. Quantile cuts use 0–19 only. Bootstrap intervals resample init clusters, preserving tasks; repeated use of this development population limits confirmatory interpretation. All scripts run under CPU affinity 10–21,54–65 and one BLAS thread. No serving logic, task selector, or policy-call escalation is introduced.

A decision executes five simulator controls; P10 and a guard MISS commit ten controls through one policy-tail decision. Step indices below are zero-based decisions, not seconds. P10 is this matched ten-control pure-policy baseline, not a different replanning-horizon pure-policy run. GR00T client resize is 256 in both arm specs; π0.5 P10 explicitly sets 224 while recipe uses its client default. Both specs set replan_steps=5. This is a historical, cross-batch comparison without common policy random draws. π0.5 recipe additionally uses persistent pace-lag escalation; ordinary guard-call comparisons exclude reason 91 but remain affected by that selection.

Source vs full-run recipe pickle bytes are not equal. `tools.provenance` checks the numerical/object state: 301 arrays per model and every shared scalar agree. Only the optional `stage_fit`, `wrist_fit`, `follow_kwargs`, `pace_lag` fields and their fit-info entries differ by absence in the earlier deployed stack artifact. [PROVENANCE.json](PROVENANCE.json) records hashes for artifacts and stat/identity provenance for admitted inputs.

## Success and paired loss population

| Model | Split | Recipe | P10 | Lost | Gained | Delta pp | Init-cluster 95% interval pp |
|---|---|---|---|---|---|---|---|
| groot | fit | 162/200 | 179/200 | 33 | 16 | -8.5 | [-15.0, -2.5] |
| groot | eval | 88/100 | 89/100 | 8 | 7 | -1.0 | [-8.0, +8.0] |
| groot | all | 250/300 | 268/300 | 41 | 23 | -6.0 | [-11.0, -1.0] |
| pi05 | fit | 183/200 | 185/200 | 15 | 13 | -1.0 | [-6.0, +4.0] |
| pi05 | eval | 89/100 | 87/100 | 7 | 9 | +2.0 | [-4.0, +8.0] |
| pi05 | all | 272/300 | 272/300 | 22 | 22 | +0.0 | [-4.0, +4.0] |

Lost = recipe fails and P10 succeeds. Gained = the reverse. These pairs identify trajectories to examine; they do not establish that the corrector caused the failures. GR00T fit delta is −8.5 pp, evaluation delta −1 pp. The full allowed pool delta is −6 pp, but it should not be presented as the evaluation estimate.

### Task inventory (descriptive only)

| Task | Instruction | G recipe/P10 wins | G lost inits | π recipe/P10 wins | π lost inits |
|---|---|---|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 19/23 | 0,1,5,7,19,24 | 30/29 | — |
| 1 | put both the cream cheese box and the butter in the basket | 30/27 | — | 30/30 | — |
| 2 | turn on the stove and put the moka pot on it | 29/29 | 5 | 24/25 | 0,19,22,28 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 29/30 | 23 | 29/27 | 2 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 24/25 | 7,15,18,20,24 | 25/28 | 0,5,11,12,19 |
| 5 | pick up the book and place it in the back compartment of the caddy | 30/29 | — | 30/30 | — |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 22/25 | 9,11,12,16,19,26 | 25/25 | 9,11,21,25 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 21/28 | 3,5,7,8,9,11,16,20 | 30/29 | — |
| 8 | put both moka pots on the stove | 28/24 | 17,18 | 22/21 | 2,9,10,23,29 |
| 9 | put the yellow and white mug in the microwave and close it | 18/28 | 0,1,2,6,7,11,13,14,16,18,22,25 | 27/28 | 10,18,25 |

Tasks 7 and 9 account for 20/41 gross GR00T losses and −17 of the pooled net −18 wins; other tasks partly offset one another. This concentration motivates the forensic description, not a task-based intervention. Evaluation GR00T losses are exactly `(0,24), (3,23), (4,20), (4,24), (6,26), (7,20), (9,22), (9,25)`.

## Guard timing and response

The deployed threshold is `noprog_n=3`, implemented as at least two nonadvancing decision intervals (`noprog_span`/`noprog_n >= 2`). It measures retrieved demonstration progress at real vision anchors; it is not object motion, contact, or task completion. Blind policy tails are not fresh progress observations.

| Model | Split | Guard calls | Escalation calls | Eligible missed | Lost first call median | Before first close | Before first open after close | Call no later than first far observation |
|---|---|---|---|---|---|---|---|---|
| groot | fit | 1896 | 0 | 0 | 10 | 23/33 | 29/33 | 31/32 |
| groot | eval | 751 | 0 | 0 | 10 | 4/8 | 5/8 | 7/7 |
| groot | all | 2647 | 0 | 0 | 10 | 27/41 | 34/41 | 38/39 |
| pi05 | fit | 675 | 634 | 0 | 20 | 4/15 | 11/15 | 14/15 |
| pi05 | eval | 396 | 279 | 0 | 16 | 3/7 | 5/7 | 6/7 |
| pi05 | all | 1071 | 913 | 0 | 19 | 7/22 | 16/22 | 20/22 |

All 41 GR00T losses received policy calls before the paired P10 episode ended: mean 10.88, minimum two. First call median 10, first far-distance observation median 42; 38/39 losses that ever cross the fit q90 distance already called by that crossing. On evaluation these figures are 9.25 calls before P10 end, minimum two, first call 10, first far 50, and 7/7. No-progress condition-to-call lag is zero at observed fresh decisions. This rules out missed or delayed decision-level verdicts on these recorded observations and weakens a universal late-first-call explanation. It does **not** rule out a guard that detects semantic mistakes too late: an initial grasp could be wrong before retrieved progress stalls. There is no object-error onset label in these logs.

### Recovery proxies, with censoring and call weighting made explicit

“Clear” means the next real vision decision after the ten-control call has no-progress <2. This is nearly the same as advancing more than half a demo decision; these are not independent evidence. End-of-episode calls without a next observation are censored, not counted as either rescued or failed. “Repeat4” means another actual call in the next four decisions and excludes calls without four observed future decisions. π0.5 reason-91 calls are excluded from the guard denominator, although a subsequent repeat can be an escalation call.

| Model | Split | Calls with next / total | Clear, per call | Clear, equal episode | Clear, first call | Repeat4 |
|---|---|---|---|---|---|---|
| groot | fit | 1835/1896 | 57.1% | 81.1% | 92.5% | 64.4% |
| groot | eval | 715/751 | 63.1% | 81.5% | 91.8% | 57.3% |
| groot | all | 2550/2647 | 58.8% | 81.2% | 92.3% | 62.4% |
| pi05 | fit | 657/675 | 76.6% | 85.6% | 90.2% | 42.1% |
| pi05 | eval | 381/396 | 70.1% | 82.8% | 91.4% | 46.9% |
| pi05 | all | 1038/1071 | 74.2% | 84.6% | 90.6% | 43.8% |

On evaluation, first-call rates are 89/97 for GR00T and 85/93 for π0.5: essentially equal. Equal-episode rates are 81.5% vs 82.8%; pooled-call rates are 63.1% vs 70.1%. Thus recurrent difficult episodes, not uniformly poor first rescues, explain much of the apparent per-call gap. GR00T lost-evaluation calls clear only 33.2% (223 observed nexts), compared with 82.5% in both-win episodes (359 nexts). These outcome-conditioned summaries are retrospective and selection-biased.

A descriptive adjustment uses only pre-call commanded state (approach/carry/post), within-model fit-distance bins, and fixed decision bins 0–19/20–39/40–59/60+. Shared cells receive the smaller of the two model call counts as weight. Evaluation standardized clear rates are 65.8% vs 71.1%, covering 688/715 GR00T and 381/381 π0.5 observed guard calls. The remaining difference is not a causal model-quality estimate: no randomization, different failure histories, coarse matching, and π0.5 escalation selection remain.

### Phase and retrieval distance

GR00T normalized negative gripper means close; π0.5 nonnegative means close. Majority of each executed five-control head supplies a command state. Approach precedes the first close; close transitions mark prior/current decisions as grasp; opening transitions mark prior/current/next as release; other closed/open decisions are carry/post. These phase labels use neighboring executed commands and are retrospective only. “Post” can include transit to a second object or drawer operation; “carry” does not prove an object is held. Adjustment above separately uses only past commands.

| Command phase | G calls | G clear | G repeat4 | π calls | π clear | π repeat4 |
|---|---|---|---|---|---|---|
| approach | 71 | 91.5% | 21.1% | 31 | 87.1% | 19.4% |
| grasp | 87 | 73.8% | 54.8% | 27 | 81.5% | 37.0% |
| carry | 221 | 72.9% | 53.3% | 140 | 82.2% | 37.1% |
| release | 130 | 51.3% | 65.4% | 51 | 66.0% | 55.1% |
| post | 242 | 47.7% | 69.1% | 147 | 53.6% | 62.5% |

Of the 1,170 calls in the 41 GR00T lost episodes, 608 are post and 214 release: 70.3% together. Lost-evaluation episodes make 6.88 close transitions on average versus 2.13 in paired successful P10 episodes; even after duration normalization, close-transition rates are 6.61 vs 3.96 per 100 decisions. This is evidence of repeated command cycles, not a count of failed physical grasps.

Recipe `dnn` is compared only within the recipe metric; debug-shadow `d1` is a different distance scale and is never equated to it. Fit fresh-decision median/q90 dnn cuts are GR00T .11634/.54807 and π0.5 .12664/.39634.

| Model | Within-model distance | Eval guard calls | Clear | Repeat4 | Median call step |
|---|---|---|---|---|---|
| groot | ≤ fit median | 271 | 73.3% | 40.6% | 26 |
| groot | median–q90 | 358 | 67.5% | 59.0% | 40 |
| groot | > fit q90 | 122 | 27.0% | 91.7% | 83 |
| pi05 | ≤ fit median | 107 | 92.5% | 22.1% | 24 |
| pi05 | median–q90 | 203 | 76.6% | 44.6% | 34 |
| pi05 | > fit q90 | 86 | 25.6% | 84.8% | 69 |

High-distance guard calls are difficult for both models, with evaluation clear rates about 27% and 26%. GR00T has more such calls in absolute count (122 versus 86), but a smaller fraction of its ordinary guard calls (16.2% versus 21.7%). The full call mix and escalation prevent a simple intrinsic-quality conclusion. Among all lost GR00T episodes, mean fresh dnn is .481 versus .110 for both-win episodes; observed mean correction RMS is .0475 versus .0266. These are episode means, and larger corrections can be a consequence of the bad trajectory.

## Corrector audit and rejected observation gates

At a fresh recipe cache hit, reconstruct the weighted first five library controls from logged rows and weights, then subtract from served_head. RMS over six motion channels measures the **applied half-strength** correction. Blind tails are excluded from this reconstruction; policy calls are excluded because their served head is policy output. Gripper reconstruction maximum error is 2.66e−7 for GR00T and 3.94e−7 for π0.5: no additional gripper correction or sign flip.

GR00T lost-vs-both-win correction RMS before the first call is .0245 vs .0229 in 0–29, but after that call is .0541 vs .0281. Evaluation is similar (.0257/.0237 before; .0525/.0284 after). The large overall correction difference is mostly on the subsequent divergent paths. This association neither proves correction harm nor proves harmlessness.

The direct error check evaluates the frozen deployed half-strength head on fresh R8 A/CU/IP observations and their same-state policy-shadow motion labels, retaining the full committed ten controls. There is no refitting. Heads take their original task one-hot/per-task dispatch because these are the inherited recipe being audited; none of the diagnostic gates uses a task selector. Cache-chunk reconstruction against the deployed library must agree to <1e−4. Input PCA keys, normalized state, action/sigma, and capped step match the serving feature layout.

| Model | Split | Anchors | Arm-episodes | Cache MSE | Recipe MSE | Relative change | Episodes improved |
|---|---|---|---|---|---|---|---|
| groot | fit | 20321 | 600 | 0.024876 | 0.013048 | -47.5% | 100.0% |
| groot | eval | 10219 | 300 | 0.023798 | 0.013982 | -41.2% | 99.7% |
| pi05 | fit | 18919 | 600 | 0.027184 | 0.014020 | -48.4% | 100.0% |
| pi05 | eval | 9432 | 300 | 0.022808 | 0.014239 | -37.6% | 100.0% |

MSE averages within each episode, then equally over arm-episodes; the 300 evaluation arm-episodes are three variants × 100 pairs, not 300 independent initial states. Bootstrap gate intervals keep init clusters together. Training scores overlap the original head training data and are descriptive; evaluation trajectories use only 20–29. Shadow labels come from different paths than the R9 recipe and are imitation targets, not oracle optimal actions.

| Phase | G motion MSE change | π motion MSE change | G cache/policy grip disagreement | π cache/policy grip disagreement |
|---|---|---|---|---|
| approach | -16.8% | -23.5% | 1.8% | 1.1% |
| grasp | -24.5% | -30.1% | 28.2% | 17.5% |
| carry | -47.3% | -40.2% | 10.4% | 6.3% |
| release | -40.5% | -34.7% | 31.0% | 18.0% |
| post | -34.8% | -32.5% | 4.2% | 2.0% |

Every phase improves on both models. Both the first five controls and five-control tail improve in aggregate. GR00T cache/policy gripper disagreement is larger (overall 10.2% vs 7.0%; release 31.0% vs 18.0%), although the inherited corrector leaves that channel alone. This suggests residual grasp/release timing or retrieval errors worth future observation-level study; it is not evidence for changing signs or a justified gripper-correction arm. Policy-shadow action sampling also contributes to disagreement.

[SCREEN_PROTOCOL.json](SCREEN_PROTOCOL.json) records four zero-correction diagnostics before their evaluation. Correction-RMS and shadow-distance q90 thresholds are fitted on pooled A/CU/IP inits 0–19 only: GR00T .09877/24.80203, π0.5 .10404/26.87250. The other conditions are sign-based and require no fitted threshold. These are ablations of the motion correction only; nothing changes gripper output or guard timing.

| Stop correction when… | G fit MSE vs recipe | G eval MSE vs recipe | G eval absolute ΔMSE 95% interval | π eval MSE vs recipe |
|---|---|---|---|---|
| crossing | 9.4% | 6.2% | [0.000640, 0.001152] | 5.7% |
| open | 30.2% | 20.7% | [0.002597, 0.003168] | 23.4% |
| large_correction | 57.9% | 45.5% | [0.004652, 0.008326] | 31.7% |
| far_retrieval | 32.2% | 26.1% | [0.002540, 0.005018] | 20.6% |

`crossing`: proposal changes gripper sign inside the committed ten controls; `open`: majority of first five proposed commands are open; `large_correction`: RMS above fit q90; `far_retrieval`: shadow d1 above fit q90. Positive change is worse. All four are worse in every evaluation variant A, CU, and IP, for both models. Thus no candidate meets the predeclared requirement even on training, and no evaluation-favored gate is selected.

## Individual GR00T lost episodes

All failures below last 104 decisions. `F` = fit init 0–19, `E` = evaluation init 20–29. `C/O` lists the first majority-close and subsequent open decision in recipe and P10; these are commanded events. `cyc R/P` counts recipe/P10 close transitions. `dnn` and correction are means; full maxima, both opening/closing counts, and exact event/call sequences are in the CSV/JSON artifacts. Correction units are normalized action RMS, not meters.

| task/init | split | first call / phase | calls | C/O recipe ; P10 | cyc R/P | dnn | corr | last demo progress | P10 end |
|---|---|---|---|---|---|---|---|---|---|
| 0/0 | F | 6 approach | 19 | 13/27 ; 11/15 | 7/4 | 0.675 | 0.0635 | 0.286 | 62 |
| 0/1 | F | 2 approach | 38 | 17/24 ; 11/14 | 3/3 | 0.503 | 0.0798 | 0.379 | 58 |
| 0/5 | F | 10 approach | 24 | 13/20 ; 15/31 | 9/4 | 0.735 | 0.0745 | 0.152 | 68 |
| 0/7 | F | 20 release | 25 | 10/20 ; 9/14 | 7/4 | 0.749 | 0.0725 | 0.286 | 65 |
| 0/19 | F | 6 approach | 22 | 12/28 ; 13/29 | 7/3 | 0.426 | 0.0403 | 0.800 | 57 |
| 0/24 | E | 2 approach | 21 | 10/24 ; 11/16 | 6/3 | 0.495 | 0.0482 | 0.245 | 61 |
| 2/5 | F | 20 release | 27 | 11/19 ; 10/18 | 11/2 | 0.401 | 0.0320 | 0.980 | 47 |
| 3/23 | E | 28 release | 33 | 15/28 ; 16/29 | 1/1 | 0.090 | 0.0229 | 0.976 | 40 |
| 4/7 | F | 4 approach | 31 | 9/24 ; 9/24 | 2/2 | 0.680 | 0.0355 | 0.533 | 48 |
| 4/15 | F | 16 carry | 29 | 11/21 ; 10/23 | 5/2 | 0.338 | 0.0279 | 0.523 | 50 |
| 4/18 | F | 2 approach | 32 | 9/21 ; 9/27 | 2/2 | 0.444 | 0.0262 | 0.523 | 51 |
| 4/20 | E | 10 release | 34 | 8/10 ; 9/22 | 3/2 | 0.474 | 0.0266 | 0.523 | 47 |
| 4/24 | E | 4 approach | 28 | 9/10 ; 8/20 | 8/2 | 0.262 | 0.0445 | 0.635 | 43 |
| 6/9 | F | 4 approach | 27 | 10/27 ; 10/24 | 14/2 | 0.283 | 0.0380 | 0.936 | 44 |
| 6/11 | F | 14 carry | 31 | 9/27 ; 9/32 | 5/2 | 0.327 | 0.0246 | 0.392 | 50 |
| 6/12 | F | 16 carry | 26 | 8/23 ; 9/28 | 12/2 | 0.316 | 0.0451 | 1.000 | 49 |
| 6/16 | F | 10 grasp | 31 | 10/14 ; 9/11 | 7/3 | 0.273 | 0.0487 | 0.824 | 48 |
| 6/19 | F | 14 carry | 33 | 9/23 ; 8/21 | 3/5 | 0.587 | 0.0254 | 0.605 | 80 |
| 6/26 | E | 36 grasp | 26 | 10/25 ; 9/27 | 13/3 | 0.320 | 0.0406 | 0.936 | 52 |
| 7/3 | F | 10 approach | 34 | 29/30 ; 9/20 | 8/2 | 0.593 | 0.0559 | 0.621 | 46 |
| 7/5 | F | 2 approach | 29 | 15/16 ; 11/22 | 8/2 | 0.787 | 0.0602 | 0.566 | 51 |
| 7/7 | F | 10 release | 32 | 9/10 ; 10/21 | 12/2 | 0.372 | 0.0600 | 1.000 | 49 |
| 7/8 | F | 2 approach | 34 | 35/40 ; 11/22 | 4/2 | 0.791 | 0.0574 | 0.208 | 48 |
| 7/9 | F | 2 approach | 29 | 21/28 ; 11/24 | 6/2 | 0.645 | 0.0614 | 0.849 | 50 |
| 7/11 | F | 10 grasp | 30 | 11/12 ; 27/28 | 7/4 | 0.867 | 0.0756 | 0.208 | 71 |
| 7/16 | F | 10 approach | 33 | 29/39 ; 11/20 | 7/2 | 0.625 | 0.0808 | 0.811 | 46 |
| 7/20 | E | 10 approach | 29 | 19/20 ; 11/22 | 9/2 | 0.711 | 0.0768 | 0.655 | 46 |
| 8/17 | F | 12 approach | 18 | 26/39 ; 28/43 | 4/2 | 0.243 | 0.0268 | 1.000 | 83 |
| 8/18 | F | 2 approach | 23 | 22/37 ; 24/38 | 5/3 | 0.206 | 0.0219 | 0.718 | 95 |
| 9/0 | F | 6 approach | 33 | 16/34 ; 16/29 | 1/1 | 0.405 | 0.0489 | 0.987 | 53 |
| 9/1 | F | 8 approach | 29 | 16/36 ; 16/29 | 1/1 | 0.263 | 0.0631 | 0.705 | 59 |
| 9/2 | F | 20 release | 31 | 19/20 ; 15/29 | 7/1 | 0.488 | 0.0390 | 0.987 | 48 |
| 9/6 | F | 10 approach | 28 | 17/39 ; 16/30 | 1/1 | 0.515 | 0.0436 | 0.987 | 51 |
| 9/7 | F | 16 grasp | 23 | 17/22 ; 16/20 | 6/3 | 0.338 | 0.0470 | 0.731 | 70 |
| 9/11 | F | 16 grasp | 28 | 17/41 ; 18/34 | 2/1 | 0.373 | 0.0330 | 0.987 | 55 |
| 9/13 | F | 8 approach | 25 | 15/18 ; 17/28 | 7/1 | 0.330 | 0.0438 | 0.261 | 58 |
| 9/14 | F | 2 approach | 30 | 18/23 ; 17/30 | 3/1 | 0.688 | 0.0492 | 0.722 | 49 |
| 9/16 | F | 14 approach | 30 | 17/28 ; 17/31 | 4/1 | 0.666 | 0.0580 | 1.000 | 51 |
| 9/18 | F | 22 carry | 25 | 19/38 ; 21/38 | 4/1 | 0.592 | 0.0426 | 0.372 | 57 |
| 9/22 | E | 20 carry | 29 | 19/25 ; 18/31 | 8/1 | 0.483 | 0.0396 | 0.987 | 50 |
| 9/25 | E | 4 approach | 31 | 17/21 ; 19/22 | 7/3 | 0.361 | 0.0456 | 0.987 | 90 |

Three instructive evaluation patterns:

- **Task 7/init 20 (soup + cheese basket):** call at 10 precedes close at 19; recipe opens at 20, while P10 closes at 11 and opens at 22. Nine recipe close/open cycles versus two P10, 29 calls, dnn .711 (max 1.546), correction .0768 (max .2037). This is repeated command cycling on a divergent trajectory, not simply an absent early call.
- **Task 9/init 22 (microwave mug):** first close is similar, 19 vs 18; recipe opens at 25 vs P10 31. It makes 29 calls, eight close transitions versus one, dnn .483, and finishes near demo progress .987 without solving. Early opening and later cycling are factual command differences; the logs cannot establish where the mug was.
- **Task 3/init 23 (bowl + drawer):** recipe/P10 close at 15/16 and open at 28/29, with one cycle each. First call at 28 and 33 total calls, despite low dnn .090, small correction .0229, and final demo progress .976. This is a low-distance, near-terminal failure that a blanket far-distance gate would miss.

[Eight evaluation timelines](results/lost_eval_timelines.svg) visualize every evaluation loss, not cherry-picked examples. Blue episode-end markers make clear how many recipe calls precede the P10 completion time.

## What the evidence answers

1. **Too late?** No dispatch defect or universal late-first-call pattern. The no-progress detector can still be semantically late or misled by retrieved progress; actual grasp-error onset is unobserved.
2. **Less effective per call?** Observed GR00T guard calls clear less often on the next observation and repeat more often within four decisions. First calls are comparable, and episode weighting greatly reduces the gap. Difficult-state composition and the π0.5 escalation policy remain major confounders. No marginal causal success gain per call is identified.
3. **Corrector harms a GR00T phase?** No phase-wide harm appears in the held-out fixed-state shadow audit; every tested observation-keyed removal worsens imitation error. Large corrections on failed recipe paths are mostly later, which cannot distinguish correction feedback from a response to bad states. A recipe-path policy-shadow or randomized correction ablation would be needed for that causal claim.
4. **Deployable fix?** None supported by this evidence. No arms frozen and no escalation duplication. Existing task-indexed recipe heads are not replaced or extended.

The strongest remaining diagnosis is a small subset of recurring, sometimes near-terminal semantic failures under sparse policy intervention, with extra gripper-event disagreement on GR00T. The present measurements do not prove which observation-only intervention would recover them.

