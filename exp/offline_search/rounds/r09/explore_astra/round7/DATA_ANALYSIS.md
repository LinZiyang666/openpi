# Round7: diagnosis of the GR00T gap

The evidence favors an underfit shared observation head plus a trajectory-distribution gap. It does not support a gripper-sign, 32-D padding, or chunk-length implementation error in round6. Missing row-table entries are ruled out in both GR00T cells; sparse or unrepresentative observation coverage remains possible. Larger shared heads improve held-out-init CV on exactly the same training corpus, so capacity is a directly supported contributor. The data do not isolate it as the sole cause of the closed-loop success deficit.

## Admission, split and estimands

`tools/data.py` rejects forbidden root prefixes before opening paths. JSONL records are lexed for top-level task/init or UID first; only admitted 20–29 records are deserialized. Conflicting/duplicate identities fail closed. Accepted terminal journal attempts are joined to server decisions, and every episode must have exactly contiguous step indices. All four GR00T round6 arms have 100 admitted complete episodes. No raw holdout root, init 30–49 trajectory, or pi0.5 outcome is used.

Offline input is the existing A/CU/IP compact corpus under the standard store. Lazy task/init members are checked before payload access; an archive with any init outside 0–29 is rejected. Training and CV subsets use only 0–19. Evaluation uses 20–29. The same init is held out jointly across tasks and A/CU/IP (`init % 3`); this prevents repeated paths of the same initialization from crossing a fold. Training balances arm-episodes. Per-cell selection was frozen at 2026-10-02T11:37:48.282011+00:00 before candidate evaluation.

Round6 20–29 outcomes were intentionally used to design round7. Reusing these inits makes round7 a development screen, not an independent final test. The older per-task controls used more trajectory variants: 80,108 long / 19,364 spatial anchors, versus 20,321 / 7,597 for both round6 and round7 shared heads. This is not a matched-data causal experiment on task identity.

Headline offline MSE is averaged within episode, then over 100 episodes. Diagnostic task/phase/distance tables below average anchors in each stratum and therefore have different overall totals. MSE uses model-normalized motion channels 0–5 and first ten controls. Labels are policy shadow draws, not optimal actions or success. Live phases describe commanded gripper state, not verified physical grasp/contact.

## Closed-loop reproduction

Owner IR uses the historical GR00T stage shares, stage1 6.145862978883088 ms, stage2 7.191964512458071 ms, stage3 28.104 ms, full K=8: `IR = look_share*Nlook/N + (1-look_share)*Ncall/N`. This reproduces the supplied owner numbers; it is not the pi0.5 .152/.848 pricing and not the newer eager cost table.

| Cell | Arm | Success / 100 | Decisions | Looks | Calls | Owner IR |
|---|---|---:|---:|---:|---:|---:|
| GR00T LIBERO-10-50 | control | 86 | 5905 | 2972 | 846 | 0.196662 |
| GR00T LIBERO-10-50 | taskfree | 80 | 6123 | 3079 | 972 | 0.209778 |
| GR00T Spatial-50 | control | 98 | 2195 | 1123 | 52 | 0.096050 |
| GR00T Spatial-50 | taskfree | 93 | 2272 | 1157 | 94 | 0.110759 |

Long: +9/−15 discordant pairs, SR delta −0.06; paired init-cluster 95% interval [−0.15, +0.03]. Spatial: +2/−7, delta −0.05, interval [−0.10, 0.00]. Bootstrap: 5000 draws, ten init clusters, all ten tasks retained. These are small-sample descriptive intervals.

All 846→972 long calls and 52→94 spatial calls carry `os_reason=4`, the unchanged no-progress guard. No new recovery trigger or call policy is present. Logs show larger state-nearest-neighbor distances (`dnn`: .1765→.2161 long, .1495→.1890 spatial) and larger progress lag (4.89→6.68, .44→.91). This is consistent with poorer subsequent trajectories, rather than a direct increase in the programmed call rate.

| Cell | Paired outcome group | Episodes | Control decisions / calls | Task-free decisions / calls |
|---|---|---:|---:|---:|
| GR00T LIBERO-10-50 | both_win | 71 | 3535 / 328 | 3583 / 394 |
| GR00T LIBERO-10-50 | both_lose | 5 | 520 / 155 | 520 / 142 |
| GR00T LIBERO-10-50 | lost | 15 | 914 / 112 | 1560 / 386 |
| GR00T LIBERO-10-50 | gained | 9 | 936 / 251 | 460 / 50 |
| GR00T Spatial-50 | both_win | 91 | 1939 / 27 | 1919 / 33 |
| GR00T Spatial-50 | both_lose | 0 | 0 / 0 | 0 / 0 |
| GR00T Spatial-50 | lost | 7 | 168 / 10 | 308 / 60 |
| GR00T Spatial-50 | gained | 2 | 88 / 15 | 45 / 1 |

The lost pairs generate +274 calls long and +50 spatial; gained pairs offset −201 and −14. Even pairs successful under both arms have +66 calls long and +6 spatial. In lost pairs, calls per decision rise from 12.25% to 24.74% long, and 5.95% to 19.48% spatial. Thus extra calls are not explained solely by failed episodes lasting longer. They still do not prove each preceding chunk is worse: the two trajectories diverge, success-conditioned groups are post-treatment, and round6 has no online policy-shadow labels.

At fresh cache hits, subtracting the weighted retrieved library action from the logged served head gives mean correction RMS .02921→.03484 long and .03341→.04111 spatial. Task-free corrections are larger despite the same .5 strength. Within task-free trajectories, corrections preceding a call in the next four decisions average .03806 versus .03233 long, and .05815 versus .03935 spatial. These associations can reflect difficult states and feedback, not a causal dose-response.

## Where the gap appears

Task IDs here are reporting strata only. No reported task value is a fitting feature or runtime selector. SR entries are successes out of ten. Offline deltas are round6 task-free minus current per-task MSE on the identical A-path states.

| Cell | Task | Control→task-free SR count | Control→task-free calls | Offline ΔMSE | Correction disagreement RMS |
|---|---:|---|---|---:|---:|
| GR00T LIBERO-10-50 | 0 | 9→5 | 79→170 | +0.002699 | 0.03779 |
| GR00T LIBERO-10-50 | 1 | 10→10 | 25→36 | +0.000897 | 0.03238 |
| GR00T LIBERO-10-50 | 2 | 10→9 | 27→66 | +0.000148 | 0.02258 |
| GR00T LIBERO-10-50 | 3 | 9→10 | 57→37 | +0.001431 | 0.03063 |
| GR00T LIBERO-10-50 | 4 | 6→6 | 152→143 | +0.002659 | 0.03459 |
| GR00T LIBERO-10-50 | 5 | 9→10 | 80→47 | +0.000604 | 0.02463 |
| GR00T LIBERO-10-50 | 6 | 7→8 | 121→97 | +0.001250 | 0.03221 |
| GR00T LIBERO-10-50 | 7 | 10→10 | 62→67 | +0.000988 | 0.03266 |
| GR00T LIBERO-10-50 | 8 | 9→6 | 102→147 | +0.000887 | 0.02553 |
| GR00T LIBERO-10-50 | 9 | 7→6 | 141→162 | +0.003579 | 0.04662 |
| GR00T Spatial-50 | 0 | 10→9 | 4→11 | +0.003195 | 0.04332 |
| GR00T Spatial-50 | 1 | 8→9 | 29→16 | -0.000074 | 0.03201 |
| GR00T Spatial-50 | 2 | 10→9 | 6→13 | -0.000093 | 0.02892 |
| GR00T Spatial-50 | 3 | 10→10 | 1→1 | +0.001297 | 0.03646 |
| GR00T Spatial-50 | 4 | 10→9 | 1→10 | +0.000451 | 0.03087 |
| GR00T Spatial-50 | 5 | 10→10 | 5→5 | +0.000391 | 0.03376 |
| GR00T Spatial-50 | 6 | 10→10 | 4→6 | +0.000370 | 0.03293 |
| GR00T Spatial-50 | 7 | 10→10 | 0→2 | -0.001824 | 0.03991 |
| GR00T Spatial-50 | 8 | 10→9 | 1→8 | -0.000439 | 0.03580 |
| GR00T Spatial-50 | 9 | 10→8 | 1→22 | +0.000726 | 0.03763 |

Long-task net losses concentrate in task 0 (−4) and task 8 (−3); task 9 and 2 lose one each, offset by tasks 3, 5 and 6 gaining one. The largest offline MSE deficits are tasks 9, 0 and 4. Task 8 has a smaller teacher-error deficit but a large success loss: mean teacher MSE is not a reliable task-success surrogate. Spatial loses two on task 9, one on 0/2/4/8 and gains one on task 1; task 0 has the largest offline deficit, but task 8 actually has lower task-free offline MSE.

Phase labels use GR00T normalized negative gripper = close. Approach is before first close; close transitions mark the previous/current decision as grasp; open transitions mark previous/current/next as release; remaining closed/open decisions are carry/post. They are retrospective diagnostics only, never serving inputs. Offline labels use all recorded executed commands before selecting fresh-look anchors.

| Cell | Phase | A anchors | Offline ΔMSE | Control→task-free live calls |
|---|---|---:|---:|---|
| GR00T LIBERO-10-50 | approach | 686 | +0.000250 | 74→105 |
| GR00T LIBERO-10-50 | carry | 1506 | +0.001995 | 268→279 |
| GR00T LIBERO-10-50 | grasp | 256 | +0.000921 | 78→85 |
| GR00T LIBERO-10-50 | post | 919 | +0.002577 | 312→369 |
| GR00T LIBERO-10-50 | release | 228 | +0.001759 | 114→134 |
| GR00T Spatial-50 | approach | 421 | +0.000255 | 2→1 |
| GR00T Spatial-50 | carry | 642 | +0.000944 | 33→44 |
| GR00T Spatial-50 | grasp | 106 | +0.000409 | 6→4 |
| GR00T Spatial-50 | post | 5 | +0.002480 | 5→24 |
| GR00T Spatial-50 | release | 56 | -0.005954 | 6→21 |

Long task-free extra calls occur in approach (+31), carry (+11), grasp (+7), post (+57), release (+20); the largest offline deficit is post (+.002577), then carry (+.001995). Spatial extra calls are primarily post (+19), release (+15), carry (+11), while approach/grasp decline. Only five spatial A anchors are post, so that stratum is highly uncertain. Spatial release has better task-free teacher MSE despite worse live release outcomes; the trajectories and denominators differ.

| Cell | Cached chunk contains a gripper sign transition | A anchors | Offline ΔMSE |
|---|---|---:|---:|
| GR00T LIBERO-10-50 | False | 3066 | +0.001792 |
| GR00T LIBERO-10-50 | True | 529 | +0.001297 |
| GR00T Spatial-50 | False | 975 | +0.000579 |
| GR00T Spatial-50 | True | 255 | -0.000506 |

A gripper event means any sign transition within the cached ten-control chunk. Long event/non-event deficits are +.001297/+.001792; Spatial is −.000506/+.000579. The gap is not confined to gripper transitions. Cache/policy shadow gripper-sign disagreement is 17.03% long and 9.96% spatial, but both correctors deliberately leave the same cached gripper untouched on the same state. This can limit either motion-only method without being a round6 sign bug.

## Retrieval distance and row coverage

Offline `d1` is the recorded retrieval metric, split by training-A median/p90 (long 10.0037/29.0289; spatial 13.0405/22.4821). Live `dnn` is the inherited guard’s nearest robot-state distance, not the same metric. Do not compare their numerical scales.

| Cell | Offline d1 bin | A anchors | Control MSE | Round6 MSE | ΔMSE |
|---|---|---:|---:|---:|---:|
| GR00T LIBERO-10-50 | ≤train median | 1773 | 0.007963 | 0.008280 | +0.000318 |
| GR00T LIBERO-10-50 | median–p90 | 1428 | 0.023921 | 0.026932 | +0.003011 |
| GR00T LIBERO-10-50 | >train p90 | 394 | 0.033293 | 0.036638 | +0.003345 |
| GR00T Spatial-50 | ≤train median | 706 | 0.009062 | 0.009362 | +0.000300 |
| GR00T Spatial-50 | median–p90 | 459 | 0.018863 | 0.018882 | +0.000019 |
| GR00T Spatial-50 | >train p90 | 65 | 0.030712 | 0.034022 | +0.003309 |

The far-distance deficit is clear in both cells. Live dnn bins share the control median/p90 thresholds within each cell. Long high-distance look counts increase 298→388 and calls 222→281; spatial increases 113→169 looks and 28→69 calls. Spatial high-distance states account for 41 of the 42 net extra calls. In long, the middle-distance bin also contributes +75 calls. This points to later-state drift and difficult retrieval neighborhoods; no absolute-distance threshold is introduced in either candidate.

| Cell | Library rows | Rows with zero training mass | Training retrieval mass p10 / median | Evaluation weight on unsupported rows |
|---|---:|---:|---|---:|
| GR00T LIBERO-10-50 | 2645 | 0 | 2.136 / 6.084 | 0.0% |
| GR00T Spatial-50 | 1063 | 0 | 2.510 / 6.450 | 0.0% |

Weighted row support is broad: the A-evaluation p10 row-support average is 45.7 long / 57.5 spatial arm-episodes (maximum 60 = 20 inits × three paths per library task neighborhood). Those paths are not 60 independent inits. Some rows have low effective mass and local table quality can still vary. However, unsupported rows cannot explain the gap: their evaluation mass is exactly zero. Shared residual RMS is .1520/.1240 versus local-table .0252/.0172 long/spatial; the table contributes a much smaller correction. This motivates increasing shared capacity before expanding the row table.

## Action-space and serving audit

- GR00T libraries are `[2645,16,32]` and `[1063,16,32]`. Only channels 0–6 are physical LIBERO actions. Channels 7–31 are noise-like padding, not zeros (observed max absolute 4.797/4.577). Both old and new corrector inputs exclude these channels and their residual targets use only 0–5. New confidence also excludes padding. Tests perturb padding and unused steps without changing physical correction.
- Normalized GR00T −1 means close, +1 means open. Both controls have the established GR00T stack sign. Reconstructed round6 fresh-cache served gripper differs from its retrieved gripper by <4.4e-7, compatible with arithmetic roundoff. The correction does not write channel 6.
- H=16, but these arms commit only the first ten controls in two five-control decisions. Corrected motion is applied exactly once on the judge `os_synth` path, then stored as the blind-tail anchor. Steps 10–15 remain byte-identical; no full-16 training or unintended padding normalization occurs.
- Round6 offline deficits exist in both halves: long first/second five ΔMSE +.001801/+.001638, spatial +.000355/+.000354. There is no evidence of a tail-specific off-by-five defect.
- The new correction API takes observation keys, robot state, elapsed step, cached action, retrieval rows and weights. A poison-identity object raises on task/episode/init access and still passes. Row/library permutation with corresponding retrieval remapping gives identical predictions. Existing retrieval and guard bookkeeping are inherited unchanged; task-indexed models remain only in the requested controls.

## Frozen alternatives and training evidence

Six head recipes were compared, each with confidence gain 0/.1/.25: round6 768 RFF, wider 1536/3072 without added features, 1536/3072 with 24 GR00T action/gripper features, and the enriched 3072 head at alpha 30. All other alphas are 100. Each cell independently selects one fixed-.5 capacity recipe and one positive-confidence-gain recipe. Both cells select enriched 3072/alpha100; confidence gain .25. No 20–29 score enters this choice.

| Recipe (same training data) | Long CV MSE/cache MSE | Spatial CV MSE/cache MSE |
|---|---:|---:|
| r6:0.0 | 0.629542 | 0.696271 |
| wide1536:0.0 | 0.610154 | 0.687587 |
| wide3072:0.0 | 0.598747 | 0.672960 |
| grip1536:0.0 | 0.610643 | 0.684386 |
| grip3072:0.0 | 0.593173 | 0.663804 |
| grip3072_a30:0.0 | 0.597919 | 0.691900 |
| grip3072:0.25 | 0.569907 | 0.647299 |

Width alone improves CV by 4.89% long / 3.35% spatial relative to round6; enriched width improves 5.78% / 4.66%. Lower regularization worsens Spatial strongly, so merely fitting harder is insufficient. The capacity result is a same-corpus ablation supporting shared-head underfit. More diverse training paths could also help, but expanding to unvalidated caches was not necessary for this round.

Both variants use the exact same shared head and row residual table per cell. Features are the prior 207 inputs plus closure fraction/first/last/event count (4), mean motion (6), second-half minus first-half motion (6), mean motion × closure fraction (6), and two gripper-state values × closure fraction (2). RFF seed is 260602. Row residual ridge lambda stays 1.

`capacity` applies .5*(shared+local). `confidence` applies `[.5+.25/(1+v/s)]*(shared+local)`. The dispersion v is the original retrieval-weighted squared motion spread across ten controls × six channels. s is the training-cell median. This is an agreement proxy, not a calibrated confidence probability: mutually agreeing retrieved chunks can still be distant or wrong. Low agreement returns strength towards .5; it never turns the correction off. The head/table fitted values and s never use task IDs.

## Frozen offline evaluation, 20–29

| Cell | Method | A episode MSE | A relative to control | CU relative | IP relative | A episodes better than control |
|---|---|---:|---:|---:|---:|---:|
| GR00T LIBERO-10-50 | capacity | 0.014525 | +0.81% | -0.61% | -0.77% | 51 / 100 |
| GR00T LIBERO-10-50 | confidence | 0.013422 | -6.85% | -3.12% | -4.83% | 69 / 100 |
| GR00T Spatial-50 | capacity | 0.011959 | -2.40% | -1.59% | -4.22% | 47 / 100 |
| GR00T Spatial-50 | confidence | 0.011459 | -6.48% | -3.09% | -8.46% | 46 / 100 |

Reference A MSE: current controls .014408 long / .012254 spatial; round6 task-free .015723 / .012548. Relative to round6, capacity reduces A MSE 7.62% / 4.69%, confidence 14.64% / 8.68%. Capacity therefore nearly closes the offline long gap and crosses the spatial mean; confidence beats control mean MSE in all six cell/path combinations.

| Cell | Variant | Paired A ΔMSE | Init-cluster 95% interval |
|---|---|---:|---|
| GR00T LIBERO-10-50 | capacity | +0.000117 | [-0.000138, +0.000386] |
| GR00T LIBERO-10-50 | confidence | -0.000987 | [-0.001439, -0.000536] |
| GR00T Spatial-50 | capacity | -0.000295 | [-0.000881, +0.000214] |
| GR00T Spatial-50 | confidence | -0.000795 | [-0.001587, -0.000048] |

These reuse round6’s ten evaluation init clusters and policy draws. Confidence Spatial improves the mean but only 46/100 individual episodes; there is meaningful heterogeneity. Candidate diagnostic task-average deltas improve for all ten Spatial tasks, which does not imply each episode improves. Its stronger correction may alter grasp geometry or future guard activation. These results justify the two frozen closed-loop screens, not a claim that the GR00T SR gap is already closed.

## Runtime and reproducibility

Serving prediction adds no new look/call. Single-thread CPU correction median is 1.24–1.31 ms, p95 1.35–1.55 ms (300 warmed synthetic-key queries, including projection; excluding retrieval/guard/GPU inference). Round6 measured roughly .80–.83 ms median on its fixture. This is local overhead, not H100 latency; owner IR does not price it. See `results/candidate_diagnostics.json` for tensor sizes and strength quantiles. Confidence median strength on evaluation A is .6226 long / .6290 spatial.

Evidence files: `results/diagnosis.json`, per-cell CV/evaluation files, `results/candidate_diagnostics.json`, `results/plugin_selftests.json`, unit-test and standard-plan logs; `SELECTION.json`, `FROZEN.json`, `INTEGRITY.json`. Source pointers: `tools/data.py` admission/phase definitions, `tools/analyze.py` live/offline comparisons, `tools/learn.py` training-only selection, `tools/numeric.py` serving math, `tools/methods.py` exact corrector swap.
