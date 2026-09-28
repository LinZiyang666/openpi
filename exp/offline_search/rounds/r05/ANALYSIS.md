# R5 analysis — offline action cache for π0.5 / GR00T N1.5 on LIBERO

Snapshot: **2026-09-28 05:08 CDT**, after both K7 replicates completed; the final four-run pool and the original planned three-run pool are now available. This report extends [R4](../r04/ANALYSIS.md), following [the R5 brief](ANALYSIS_BRIEF.md), [selection and both addenda](SELECTION.md), [findings](FINDINGS.md), [coding brief](CODING_BRIEF.md), ideation A–D reports, Q1–Q6 hand-backs, and [the coordinator ledger](../../../../logs/offline_search_exploration.log.md) §§8–10. Ledger outcome numbers were not used as computed results.

**Result:** longer cached commitments improve π0.5 in all four cells; GR00T needs suite-dependent commitment length and policy use. All four K7 repeats retain an advantage over three stock repeats; the completed prescribed three-run set also retains the same positive direction with an interval above zero. C10 lowers policy cost without establishing SR superiority; Q6 is promising under an assumed wrist cost model. Paid library growth and refitting help substantially. B1 does not validate its offline objective as a closed-loop SR selector. GPU shadow is faster at the median, but action parity remains incomplete.

Only completed arms with both a completion marker and a per-arm summary are used. Per the coordinator's clarification, `state/<arm>.manifest_<sha>.DONE` is a valid marker; the four grow250 saved manifests were checked against `q4_growth/evaluation_pairs.json`. **All 52 requested R5 arms are complete and included**, together with the R4 cell fill and historical references. Replicate B was admitted only after checking its DONE marker, per-arm summary and accepted journal pairs.

Provenance and conventions: let **R** = `/home/weiland/trace_runs/os_closed_loop`. An arm identified as `run/arm` always means `R/run/runs/arm/summary.json`, `R/run/runs/arm/client/journal.jsonl`, and its completion marker in `R/run/state/`. The source index in §9 maps every short alias. SR is success/accepted episodes, normally 500 = ten tasks × inits 0–49; grow250 uses 250 = ten tasks × inits 25–49. Accepted attempts were matched before reading server decisions. All 137 completed arms passed journal count, pair-identity, and summary-success checks. The raw-decision audit covered 86 arms / 1,849,535 deduplicated accepted decisions, including all 52 completed R5 arms; R5 had zero duplicate/conflicting decisions, discontinuous sequences, or `ok=false` decisions. One legacy R3 cost conflict is explicitly excluded from valid cost rankings (§9).

> Scripts are preserved in `analysis_scripts/` (copied from `/tmp/r5_analysis/` at 05:1x CDT); copy them back to `/tmp/r5_analysis/` to rerun the commands below.

Reproduction from the repository root (CPU only; outputs stay in `/tmp/r5_analysis/` except this report):

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/r5_analysis/analyze.py
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/r5_analysis/audit_logs.py
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/r5_analysis/write_report.py
```

`analysis.json` contains all arm counts/costs, 3,339 pair comparisons, pooled contrasts and source hashes; `journals.json` contains paired outcomes; `decisions.json` contains source paths, realized control counts, and shadow aggregates. The scripts read historical arm/K8 dictionaries using AST literals, without executing the R4 scripts. Analysis did not launch or alter any experiment.

Statistics: table contrasts are **A minus B**, in percentage points, followed by a 95% paired bootstrap interval, **W/L = A-only successes / B-only successes**, and exact two-sided McNemar p. Bootstraps use 10,000 draws, seed 20260928, resampling the `(task, init)` unit (R4's init convention), not individual decisions. Repeat comparisons average outcomes within each pair before resampling its entire repeat vector; thus four repeats still provide 500 clusters, not 2,000 independent initializations. This is conditional on these ten tasks and the observed run seeds; it does not estimate generalization to new tasks or fully integrate run-to-run uncertainty. Pooled B1 uses an equal-weight six-cell mean, stratifies resampling by suite (500 pair clusters in each of two suites), and keeps the outcomes for models and library scales together within each suite/task/init. Numeric task IDs in different suites are not treated as the same scene. Single-arm Wilson intervals and per-task success counts are in `analysis.json`. Unless specifically stated, p values are exploratory and unadjusted; non-significance is not equivalence.

Costs: owner IR is recomputed from each arm's `cost_ledger` counts, `v=V/N`, `m=M/N`: π0.5 `(.152 v + .848 m) × 5/L`; GR00T `(.148 v + .852 m) × 5/L`. L is controls/request: normally 5, or 10 for pure L=10 inference. A blind or policy-tail five-control slot incurs no new model inference. **Wrist ASSUMPTION:** `(.0552 v + (.848+.0499)m) × 5/L`, the R4 eager-to-owner ratio transfer, not an R5 graph measurement. Full-camera π0.5 dummy_cached repricing is the brief's stipulated exact `−.048 v × 5/L`; this is a counterfactual cost panel, not a newly measured R5 run. It is not stacked onto wrist or transferred to GR00T. Legacy R2/R3 summaries lack `cost_ledger`; their full-vision v=1 and m were reconstructed from counts and checked against raw decisions where possible. Summary eager/measured IR fields are not substituted for owner IR. Fit MB below means decimal serialized pickle MB, not resident memory or total deployment bytes.

## 1. Noise floor and acceptance rule

| Configuration / completed runs | SRs | Mean SR | Sample SD / range (pp) | Pooled decision-weighted owner IR |
| --- | --- | --- | --- | --- |
| Stock g500 | 0.864, 0.850, 0.832 | 0.84867 | 1.604 / 3.20 | 0.24444 |
| K7 tail: R4, B1 hand, replicates A/B | 0.880, 0.906, 0.876, 0.882 | 0.88600 | 1.356 / 3.00 | 0.20052 |
| K7 prescribed: R4, replicates A/B | 0.880, 0.876, 0.882 | 0.87933 | 0.306 / 0.60 | 0.20183 |

| A − B | Paired ΔSR [95% CI]; W/L; exact p | Discordance |
| --- | --- | --- |
| `g500` − `g500_repa` | +1.40 [-1.60, +4.40]; 33/26; p=0.4350 | 59/500 (11.8%) |
| `g500` − `g500_repb` | +3.20 [+0.20, +6.20]; 36/20; p=0.0440 | 56/500 (11.2%) |
| `g500_repa` − `g500_repb` | +1.80 [-1.20, +4.80]; 34/25; p=0.2976 | 59/500 (11.8%) |
| `k7_tail1ug` − `r5b_p_l10_500_hand` | -2.60 [-5.40, +0.20]; 20/33; p=0.0984 | 53/500 (10.6%) |
| `k7_tail1ug` − `r5x_p_l10_500_k7tail_repa` | +0.40 [-2.60, +3.40]; 30/28; p=0.8957 | 58/500 (11.6%) |
| `k7_tail1ug` − `r5x_p_l10_500_k7tail_repb` | -0.20 [-3.00, +2.60]; 25/26; p=1.0000 | 51/500 (10.2%) |
| `r5b_p_l10_500_hand` − `r5x_p_l10_500_k7tail_repa` | +3.00 [+0.20, +5.80]; 34/19; p=0.0534 | 53/500 (10.6%) |
| `r5b_p_l10_500_hand` − `r5x_p_l10_500_k7tail_repb` | +2.40 [-0.40, +5.20]; 32/20; p=0.1263 | 52/500 (10.4%) |
| `r5x_p_l10_500_k7tail_repa` − `r5x_p_l10_500_k7tail_repb` | -0.60 [-3.60, +2.40]; 29/32; p=0.7982 | 61/500 (12.2%) |

Identical configurations vary by about three SR points; even the stock .864 versus .832 repeat yields nominal p=0.0440. A single positive p<.05 is therefore insufficient grounds for accepting a small engineering gain. Retain R4's rule: use a paired comparison to the per-init mean of repeated references, require an interval above zero for an SR improvement, inspect repeat spread and task concentration, and repeat small effects before promotion. If preserving SR while saving cost is the claim, predeclare a noninferiority margin and test it; none was declared here. Spatial differences below one point require particular restraint.

The **four completed K7 runs** (R4 + B1 hand + replicates A/B) versus stock give **+3.73 pp, 95% CI [+1.32, +6.25]**. Owner IR falls from 0.24444 to 0.20052, a computed 18.0% reduction. Thus the positive K7 effect survives the final four-run comparison, and is not dependent on selecting the best .906 run. Net paired gains are concentrated in tasks 8 and 9 (10.75 and 7.08 mean additional successes across repeats); task 4 loses 4.92. This remains a conditional, benchmark-specific result.

The completed original planned set is **R4 + replicate A + replicate B**: mean SR 0.87933, with **+3.07 pp, 95% CI [+0.53, +5.67]** against stock. Both the prescribed three-run and all-four-run comparisons satisfy the paired interval-above-zero acceptance criterion. Adding the independently verified B1 hand repeat changes the point estimate but not that conclusion; neither interval fully integrates uncertainty over future run seeds.

Replicate B (`r05_x/r5x_p_l10_500_k7tail_repb`) independently yields 441/500 successes = 0.882. Its per-arm ledger has N=28454, V=16244, M=3787, hence v=0.57088634, m=0.13309201, owner IR=0.19964; the served fit is 142,348,562 bytes. Its accepted raw decisions match those counts. All nine within-configuration repeat comparisons are shown above; no individual run is selected as the configuration estimate.

## 2. Execution length on the cache and policy sides

| π0.5 cell | CL2 L=5 SR @ owner IR | Pure-cache 10-step tail SR @ IR | Tail − CL2: Δ pp [CI]; W/L; p |
| --- | --- | --- | --- |
| l10-50 | 0.630 @ 0.15200 | 0.706 @ 0.07640 | +7.60 [+3.60, +11.60]; 74/36; p=0.0004 |
| l10-500 | 0.768 @ 0.15200 | 0.828 @ 0.07650 | +6.00 [+2.40, +9.80]; 62/32; p=0.0026 |
| sp-50 | 0.800 @ 0.15200 | 0.838 @ 0.07754 | +3.80 [+0.80, +6.80]; 39/20; p=0.0183 |
| sp-500 | 0.954 @ 0.15200 | 0.982 @ 0.07819 | +2.80 [+0.80, +5.00]; 21/7; p=0.0125 |

All four cached-tail contrasts are positive; all survive Holm correction within this four-cell family (maximum adjusted p < .026). R5 extends R4's spatial-500 result to l10 and the 50 libraries. The three R5 pure-tail ledgers have **zero MISS**, so these gains cannot be purchased by additional policy calls. They retain an anchor every other five-control slot, with maximum blind run one. The comparison changes both reobservation cadence and commitment to the stored chunk's second half; it is a control/execution effect, not evidence that the visual or synthesis representation improved.

R4's policy result remains relevant: π0.5 l10 L=10 runs .904/.900 have mean .902 at IR .5; L=5 seeded runs .848/.860 have mean .854 at IR 1. Their paired pooled contrast is +4.80 pp [2.30, 7.20]. Historical trace_dual L=5 is .844 on l10 and .986 on spatial; spatial L=10 is also .986. R4 already established that longer policy execution can improve l10 while halving decisions. Do not attribute that component to cache retrieval. Trace_dual values are read from R4/brief and shown descriptively: **ASSUMPTION for this extension:** cross-harness L=5 traces are not treated as an independently verified paired execution-length experiment, particularly for GR00T.

| GR00T cell | CL2 L=5 | Tail L=10 (one blind block) | Tail L=15 (two blind blocks) | Pure policy L=10 | Historical policy L=5 |
| --- | --- | --- | --- | --- | --- |
| l10-50 | 0.552 @ 0.14800 | 0.608 @ 0.07428 | 0.570 @ 0.05004 | 0.866 @ 0.50000 | .870 @ 1 |
| l10-500 | 0.706 @ 0.14800 | 0.830 @ 0.07450 | 0.808 @ 0.05018 | 0.866 @ 0.50000 | .870 @ 1 |
| sp-50 | 0.888 @ 0.14800 | 0.868 @ 0.07529 | 0.864 @ 0.05127 | 0.938 @ 0.50000 | .940 @ 1 |
| sp-500 | 0.966 @ 0.14800 | 0.964 @ 0.07551 | 0.946 @ 0.05175 | 0.938 @ 0.50000 | .940 @ 1 |

| GR00T cell | 10-step tail − CL2 | 10-step tail − 15-step tail | 10-step tail − policy L=10 | 15-step tail − policy L=10 |
| --- | --- | --- | --- | --- |
| l10-50 | +5.60 [+1.40, +9.80]; 76/48; p=0.0150 | +3.80 [-0.60, +8.20]; 74/55; p=0.1127 | -25.80 [-30.80, -20.80]; 34/163; p<.0001 | -29.60 [-34.60, -24.60]; 31/179; p<.0001 |
| l10-500 | +12.40 [+8.40, +16.40]; 90/28; p<.0001 | +2.20 [-1.60, +6.00]; 54/43; p=0.3099 | -3.60 [-7.60, +0.40]; 45/63; p=0.1014 | -5.80 [-9.80, -1.60]; 42/71; p=0.0081 |
| sp-50 | -2.00 [-5.60, +1.60]; 38/48; p=0.3318 | +0.40 [-3.40, +4.20]; 49/47; p=0.9188 | -7.00 [-10.60, -3.40]; 25/60; p=0.0002 | -7.40 [-11.00, -3.80]; 23/60; p<.0001 |
| sp-500 | -0.20 [-1.80, +1.40]; 9/10; p=1.0000 | +1.80 [-0.20, +4.00]; 19/10; p=0.1360 | +2.60 [+0.20, +5.00]; 27/14; p=0.0596 | +0.80 [-1.80, +3.40]; 26/22; p=0.6655 |

GR00T l10 favors cached commitment over fresh five-step retrieval (+5.6/+12.4 pp at 50/500); spatial does not (−2.0/−0.2 pp). Every 10-versus-15 tail SR difference is unresolved in one run, including the newly completed spatial-50 .868 versus .864. Fifteen-step tails cost about .050–.052; ten-step tails cost .074–.076, so the modest numerical preference for ten does not prove its extra vision is worthwhile. R4's spatial-500 phase rule remains .976 @ .07834, beating its 15-step tail by 3.0 pp (R4); it also exceeds the new policy L=10 by 3.8 pp, W/L 31/12, p=.0054, CI [1.4, 6.4]. Stored tail continuation and selecting a new phase successor are different control operations.

GR00T policy L=10 is .866 l10 / .938 spatial at IR .5 versus historical L=5 .870/.940. Those tiny descriptive changes cannot establish a causal length effect. The useful variable is **source × commitment length × suite/library**, not a universal “commit” switch. Keep mandatory vision anchors and the measured one-/two-block blind caps; the favorable π0.5 L=10 result does not justify unbounded GR00T continuation.

## 3. C10 policy commitment and D1 grasp checks

| Cell | C10 SR @ owner IR | C10 − single R4 K7 | C10 − pure policy L=10 |
| --- | --- | --- | --- |
| l10-50 | 0.830 @ 0.18073 | +2.40 [-1.20, +6.00]; 47/35; p=0.2242 | -7.40 [-11.20, -3.60]; 32/69; p=0.0003 |
| l10-500 | 0.866 @ 0.16108 | -1.40 [-4.20, +1.40]; 23/30; p=0.4101 | -3.80 [-7.60, -0.20]; 35/54; p=0.0558 |
| sp-50 | 0.910 @ 0.14145 | +1.60 [-0.80, +4.00]; 24/16; p=0.2682 | -7.60 [-10.40, -5.00]; 7/45; p<.0001 |
| sp-500 | 0.982 @ 0.12058 | +0.00 [-1.40, +1.40]; 6/6; p=1.0000 | -0.40 [-2.00, +1.20]; 7/9; p=0.8036 |

For l10-500 the stronger comparator is repeated K7: C10 − four-run K7 mean is **-2.00 pp, 95% CI [-4.35, +0.30]**; versus stock mean it is +1.73 pp, 95% CI [-1.27, +4.67]. C10 reduces owner cost 19.7% against pooled K7 but has a −2.00 pp point change and allows a −4.35 pp loss in the interval. This is a cost/SR trade, not demonstrated preservation or improvement. Against the prescribed three-run pool, C10 gives -1.33 pp, 95% CI [-3.73, +1.00]; the interpretation is unchanged.

Spatial-500 K7's B1 hand repeat gives mean SR .980; C10's .982 differs by +0.20 pp, CI [−1.00, +1.40] (inverting the saved pooled comparison). Pure-cache tail already reaches .982 at .07819, so C10's additional policy calls buy no observed SR there. At 50, C10 improves K7 numerically but neither cell resolves superiority. Against pure policy L=10, l10-50 and spatial-50 lose 7.4/7.6 pp; l10-500 loses 3.8 pp with exact p=.0558. Its bootstrap interval narrowly excludes zero while exact McNemar does not: use the prespecified exact test, not whichever yields significance. The second l10 L=10 reference .900 yields −3.4 pp, W/L 29/46, p=.0639. No equality claim follows.

| C10 cell | N / V / M | v / m | Policy-tail slots / N | MISS with follow-up → tail | Terminal MISS | Max blind run |
| --- | --- | --- | --- | --- | --- | --- |
| l10-50 | 30480 / 15337 / 3747 | 0.50318 / 0.12293 | 3653 / 30480 = 11.985% | 3653/3653 | 94 | 1 |
| l10-500 | 28784 / 14497 / 2869 | 0.50365 / 0.09967 | 2794 / 28784 = 9.707% | 2794/2794 | 75 | 1 |
| sp-50 | 11771 / 6017 / 885 | 0.51117 / 0.07518 | 704 / 11771 = 5.981% | 704/704 | 181 | 1 |
| sp-500 | 10564 / 5429 / 529 | 0.51392 / 0.05008 | 265 / 10564 = 2.509% | 265/265 | 264 | 1 |

Every nonterminal C10 MISS is followed by its policy tail; terminal MISSes explain why policy-tail slots are fewer than MISSes, especially on short successful spatial episodes. The realized tail share is 2.5–12.0% of decisions, not 50%. Roughly half of all decisions are blind, mostly cache tails. Costs and SR therefore include both cached ten-step commitments and a smaller amount of ten-step policy execution. The lifecycle gate appears to be exercised as intended; it does not establish the causal value of each individual policy tail.

| D1 cell | SR @ owner IR | D1 − K7 | Triggered episodes /500 | Successful / failed triggered episodes | Trigger task counts (0…9) |
| --- | --- | --- | --- | --- | --- |
| l10-50 | 0.812 @ 0.24025 | +0.60 [-3.00, +4.20]; 45/42; p=0.8304 | 84 | 48 / 36 | 31,11,5,0,0,0,8,18,9,2 |
| l10-500 | 0.870 @ 0.20746 | -1.00 [-4.20, +2.20]; 30/35; p=0.6201 | 24 | 14 / 10 | 10,3,0,0,0,0,3,5,3,0 |

Raw `look_reason=9` / `extras.grasp_check=1` rows show 84/24 checks at 50/500, at most one per episode, all vision-bearing MISSes. Q1's pre-intervention replay reach was 73/24 episodes ([hand-back](q1_commit/HANDBACK.md)); changed trajectories mean replay reach is not an invariant exposure count. Reach is limited, and task 4 has no triggers despite being a weak K7 task. Success after a check is not the check's treatment effect. D1-500 versus pooled K7 is -1.60 pp, 95% CI [-4.10, +0.90]. **One run cannot resolve D1's small predicted effect** (Q1's ≈0.39 pp at 500); there is neither adoption evidence nor proof of zero benefit. The 50 result is also within the measured run noise.

## 4. GR00T CycleTail mixed mode

| Cell | G10 SR @ owner IR | G10 − tail L=10 | G10 − tail L=15 | G10 − policy L=10 |
| --- | --- | --- | --- | --- |
| l10-50 | 0.718 @ 0.18503 | +11.00 [+6.40, +15.80]; 104/49; p<.0001 | +14.80 [+10.20, +19.20]; 111/37; p<.0001 | -14.80 [-19.60, -10.00]; 45/119; p<.0001 |
| l10-500 | 0.828 @ 0.18605 | -0.20 [-3.80, +3.40]; 42/43; p=1.0000 | +2.00 [-1.60, +5.60]; 50/40; p=0.3428 | -3.80 [-7.80, +0.20]; 42/61; p=0.0756 |
| sp-50 | 0.920 @ 0.19794 | +5.20 [+1.80, +8.60]; 53/27; p=0.0049 | +5.60 [+2.00, +9.20]; 55/27; p=0.0026 | -1.80 [-4.60, +1.00]; 23/32; p=0.2806 |
| sp-500 | 0.952 @ 0.19897 | -1.20 [-2.80, +0.40]; 6/12; p=0.2379 | +0.60 [-1.60, +3.00]; 19/16; p=0.7359 | +1.40 [-1.20, +4.00]; 27/20; p=0.3817 |

| Cell | N / V / M | v / m | MISS / vision anchors | Policy tails / decisions | Nonterminal MISS → tail / terminal MISS |
| --- | --- | --- | --- | --- | --- |
| l10-50 | 33400 / 16785 / 4338 | 0.50254 / 0.12988 | 25.84% | 4296 / 33400 (12.86%) | 4296/4296 / 42 |
| l10-500 | 29923 / 15062 / 3918 | 0.50336 / 0.13094 | 26.01% | 3868 / 29923 (12.93%) | 3868/3868 / 50 |
| sp-50 | 11831 / 6025 / 1702 | 0.50926 / 0.14386 | 28.25% | 1629 / 11831 (13.77%) | 1629/1629 / 73 |
| sp-500 | 11211 / 5729 / 1623 | 0.51102 / 0.14477 | 28.33% | 1543 / 11211 (13.76%) | 1543/1543 / 80 |

G10 pays a scheduled policy call every fourth anchor, beginning with a paid anchor. Finite episode endings make the realized MISS share .130–.145, above the infinite-horizon .125 design value; 25.8–28.3% of anchors are paid. Every eligible policy tail is served and all observed blind runs have length one. This is GR00T mixed mode with actual policy cost, not a pure-cache length variant.

At **50**, policy calls buy +11.0 pp l10 and +5.2 pp spatial versus ten-step pure tails (p<.0001 / .0049), at IR .18503/.19794 instead of .07428/.07529. The two favorable 50-cell tests also survive Holm correction over the four G10-versus-ten-step-tail comparisons. G10 remains well below pure policy on l10-50; spatial-50's −1.8 pp versus policy is unresolved, not equivalent. At **500**, G10 is point-dominated by ten-step pure tail in both suites: l10 .828 @ .18605 versus .830 @ .07450, and spatial .952 @ .19897 versus .964 @ .07551. Spatial-500 phase is better still (.976 @ .07834). The result rejects a universal periodic policy schedule for large libraries, while supporting selective policy use with the smaller bank. It does not tell us which particular scheduled calls are useful.

## 5. Wrist plus tail (Q6) and the wrist spatial-50 repeat

| Cell | Q6 SR @ wrist owner IR (ASSUMPTION) | Q6 IR if charged full-camera owner prices | Q6 − K7 | Q6 − R4 wrist guard-only |
| --- | --- | --- | --- | --- |
| l10-50 | 0.792 @ 0.19853 | 0.24717 | -1.40 [-5.40, +2.60]; 48/55; p=0.5546 | +5.80 [+1.60, +10.00]; 77/48; p=0.0120 |
| l10-500 | 0.892 @ 0.16209 | 0.21066 | +1.20 [-2.20, +4.60]; 40/34; p=0.5614 | +7.20 [+3.60, +10.80]; 62/26; p=0.0002 |
| sp-50 | 0.902 @ 0.12999 | 0.17897 | +0.80 [-2.40, +4.00]; 36/32; p=0.7163 | -2.20 [-5.20, +0.80]; 24/35; p=0.1925 |
| sp-500 | 0.990 @ 0.07419 | 0.12273 | +0.80 [-0.60, +2.20]; 8/4; p=0.3877 | +1.60 [+0.20, +3.20]; 12/4; p=0.0768 |

Q6 combines wrist-confirmed stuck guards, a wrist visual key, and cached ten-step tails. At l10-500, .892 @ .16209 versus four-run K7 0.88600 @ 0.20052 gives **+0.60 pp, 95% CI [-2.25, +3.50]** and 19.2% lower assumed cost. Versus stock repeats it is +4.33 pp, 95% CI [+1.00, +7.67]. This makes Q6 a promising cost candidate, but one run cannot establish its small advantage over K7 or preservation of pure-policy L=10 SR (−1.2 pp, W/L 38/44, p=.5811, CI [−4.8,+2.4]). If wrist is charged as full-camera inference, its l10-500 IR is .21066 and the claimed cost edge over K7 disappears. Measuring the wrist graph price is therefore material. Against the prescribed three-run pool, Q6 gives +1.27 pp, 95% CI [-1.67, +4.20]; it also fails to resolve an SR improvement.

At l10-50, Q6 .792 @ .19853 is point-dominated by C10 .830 @ .18073. Tail addition to wrist guard-only helps l10 (+5.8/+7.2 pp), but does not establish an improvement on spatial; spatial-50 actually drops 2.2 pp. Spatial-500 Q6 is .990 @ .07419, versus pure tail .982 @ .07819: +0.8 pp, W/L 7/3, p=.3438. Versus two pooled K7 runs it is +1.00 pp, 95% CI [-0.30, +2.30]. Its appealing raw frontier point is conditional on the wrist cost assumption and single-run SR, not a confirmed new accuracy record.

The independent wrist guard-only spatial-50 repeat is **.928 @ .15539** versus R4 .924 @ .15612. Repeat − R4 is +0.40 [-1.60, +2.40]; 14/12; p=0.8450; repeat − stock .888 is +4.00 [+1.20, +6.80]; 37/17; p=0.0091. The two wrist runs average **.926**, with pooled difference from the shared stock run **+3.80 pp, 95% CI [+1.10, +6.60]** and owner IR 0.15575. This reproduces the direction and size of R4's wrist result; the common one-run stock comparator remains a limitation. The natural spatial-50 wrist choice is guard-only, not automatically the Q6 stack.

## 6. Paid library growth and the demo-data curve

**Grow250 uses a separate held-out manifest.** Acquisition is ten tasks × inits 0–24; evaluation is ten tasks × inits 25–49. Every comparison below intersects the reference with those same 250 pairs. Both grown variants use ordinary CL2, five controls/request, zero MISS and no blind tails; they are compared with CL2 references, not with the demo-tail arms. Static-50 and grow250 use kref=5; static-500 uses kref=8, so the latter is not a candidate-count-only comparison. This follows the Q4 [hand-back](q4_growth/HANDBACK.md), `evaluation_pairs.json`, and the run's saved manifests/specs.

| Suite / variant | SR on 250 pairs | Serving IR | vs CL2-50 on same pairs | vs CL2-500 on same pairs | Refit − frozen |
| --- | --- | --- | --- | --- | --- |
| l10 / refit | 0.744 | 0.15200 | +14.40 [+7.20, +21.60]; 61/25; p=0.0001 | -1.20 [-7.20, +4.80]; 29/32; p=0.7982 | +6.00 [+0.80, +11.20]; 32/17; p=0.0444 |
| l10 / frozen | 0.684 | 0.15200 | +8.40 [+1.20, +15.60]; 54/33; p=0.0314 | -7.20 [-13.60, -0.80]; 26/44; p=0.0414 | — |
| sp / refit | 0.952 | 0.15200 | +18.00 [+12.40, +24.00]; 53/8; p<.0001 | -1.20 [-4.40, +2.00]; 7/10; p=0.6291 | +7.60 [+3.20, +12.00]; 27/8; p=0.0019 |
| sp / frozen | 0.876 | 0.15200 | +10.40 [+4.00, +16.80]; 48/22; p=0.0025 | -8.80 [-13.60, -4.40]; 7/29; p=0.0003 | — |

The reference SRs on this manifest are l10 **.600 / .756**, spatial **.772 / .964** for CL2-50 / CL2-500. Grown refit reaches .744/.952: both are 1.2 pp below static-500, with intervals [−7.2,+4.8] and [−4.4,+2.0]. Thus growth closes 92.3%/93.75% of the observed static-50→500 gap, but the data do not establish equality. Frozen append improves SR too; refitting adds 6.0 pp l10 (p=.0444) and 7.6 pp spatial (p=.0019). Both refit-versus-frozen tests survive Holm correction over these two comparisons (adjusted p=.0444 l10 / .0038 spatial), although the l10 increment is close to the noise boundary; the large gains over static-50 are much clearer. One acquisition split cannot establish generality, and the reverse split / GR00T growth were not run.

| Suite | Initial episodes / rows | Paid episodes / successful admissions | Policy calls paid | Appended / total rows | Final stored episodes | Raw store bytes per copy |
| --- | --- | --- | --- | --- | --- | --- |
| l10 | 50 / 2,640 | 250 / 208 | 14,849 | 10,481 / 13,121 | 258 | 3,459,476,787 |
| spatial | 49 / 1,018 | 250 / 248 | 5,310 | 5,222 / 6,240 | 297 | 1,645,435,050 |

| Suite / fit | Fit bytes | Evaluation decisions N | Acquisition + serving numerator | Lifecycle IR |
| --- | --- | --- | --- | --- |
| l10 / refit | 52611819 | 16246 | 14849 + .152 × 16246 | 0.55695 |
| l10 / frozen | 52612030 | 17012 | 14849 + .152 × 17012 | 0.54722 |
| sp / refit | 35134074 | 5391 | 5310 + .152 × 5391 | 0.57279 |
| sp / frozen | 35134284 | 5874 | 5310 + .152 × 5874 | 0.55462 |

Acquisition counts/rows/raw-store bytes are read from Q4's published provenance and hand-back; fit sizes above are independently `stat`-read. All 250 paid episodes count toward acquisition, including failures; only successful rows enter the bank. Exact duplicate admissions were zero. The nominal spatial-50 base actually contains 49 episodes. Refitting uses all grown candidates; frozen append preserves the original representation, scales and calibration. This separates candidate coverage from refitting better than a simple 50→500 comparison, though it is not a factorial decomposition of every fit component.

**Lifecycle ASSUMPTION:** price an acquisition policy call as 1 and an evaluation CL2 five-control decision as .152, then normalize by the same number of nominal full-policy decision slots: `(A + .152 N)/(A+N)`. These are 250 acquisition + 250 evaluation episodes with their actual decision counts, not equal-length synthetic episodes. Refit's lifecycle IR is .55695 l10 / .57279 spatial; frozen is .54722/.55462. Longer, less successful evaluations can lower this ratio mechanically, so it is not a total cost-to-success metric. Absolute normalized work is A+.152N in the table, and initial demo acquisition is unpriced for all candidates. Serving-only .152 is a steady-state accounting boundary, not a free-data claim; acquisition amortizes only over later deployment traffic. Raw grow stores need 5,104,911,837 bytes per disk/RAM copy in addition to the compact fits.

| Suite / nominal episodes | Controller / kref | Rows | SR | v / m | Owner IR | Fit bytes |
| --- | --- | --- | --- | --- | --- | --- |
| l10 / 50 | anchor_tail / 5 | 2640 | 0.706 | 0.50261 / 0.00000 | 0.07640 | 26091371 |
| l10 / 100 | anchor_tail / 5 | 5764 | 0.748 | 0.50292 / 0.00000 | 0.07644 | 34145321 |
| l10 / 200 | anchor_tail / 5 | 11718 | 0.788 | 0.50271 / 0.00000 | 0.07641 | 49494769 |
| l10 / 300 | anchor_tail / 5 | 17612 | 0.806 | 0.50336 / 0.00000 | 0.07651 | 64689528 |
| l10 / 500 | anchor_tail / 8 | 29472 | 0.828 | 0.50328 / 0.00000 | 0.07650 | 95264533 |
| sp / 50 | anchor_tail / 5 | 1018 | 0.838 | 0.51016 / 0.00000 | 0.07754 | 21909613 |
| sp / 100 | anchor_tail / 5 | 2257 | 0.950 | 0.51109 / 0.00000 | 0.07769 | 25104085 |
| sp / 200 | anchor_tail / 5 | 4440 | 0.970 | 0.51170 / 0.00000 | 0.07778 | 30732026 |
| sp / 300 | anchor_tail / 5 | 6577 | 0.944 | 0.51231 / 0.00000 | 0.07787 | 36241257 |
| sp / 500 | anchor_tail / 8 | 10909 | 0.982 | 0.51440 / 0.00000 | 0.07819 | 47409015 |

| Suite / contrast | SR refit / frozen200 | Rows (both) | Frozen200 owner IR | Frozen200 fit bytes | Paired Δ pp [CI]; W/L; p |
| --- | --- | --- | --- | --- | --- |
| l10 | 0.788 / 0.684 | 11718 | 0.07637 | 49494984 | +10.40 [+6.20, +14.60]; 86/34; p<.0001 |
| sp | 0.970 / 0.808 | 4440 | 0.07726 | 30732276 | +16.20 [+12.80, +19.80]; 88/7; p<.0001 |

| Suite | 100 − 50 | 200 − 100 | 300 − 200 | 500 − 300 |
| --- | --- | --- | --- | --- |
| l10 | +4.20 [-0.40, +8.80]; 80/59; p=0.0895 | +4.00 [-0.00, +8.00]; 65/45; p=0.0696 | +1.80 [-2.00, +5.60]; 52/43; p=0.4119 | +2.20 [-1.20, +5.60]; 44/33; p=0.2543 |
| sp | +11.20 [+7.60, +15.00]; 76/20; p<.0001 | +2.00 [-0.40, +4.40]; 25/15; p=0.1539 | -2.60 [-5.00, -0.20]; 13/26; p=0.0533 | +3.80 [+1.80, +6.00]; 24/5; p=0.0005 |

The demo curve holds the pure-cache tail controller fixed and has zero MISS throughout. It is **not the paid grow250 curve**. The 100/200/300 demo subsets are nested within the 500 B-pool; the current 50 library is a different source and has no source-path/row/hash overlap with those subsets (Q4 hand-back). The 500 endpoint also changes kref from 5 to 8. B1's null result does not establish these kref settings are equivalent. Do not infer a clean marginal value of the first 50→100 or the last 300→500 demonstrations.

Refit l10 progresses .706→.748→.788→.806→.828. Spatial is .838→.950→.970→.944→.982: the 300 point is lower than 200 by 2.6 pp, p=.0533, and the paired bootstrap interval [−5.0,−0.2] illustrates another exact-test/interval boundary disagreement. This is not a proven monotone curve or proven saturation. Spatial-200 versus 500 is −1.2 pp (W/L 8/14, p=.2863, CI [−3.0,+0.6]); it is a plausible smaller bank, not demonstrated equivalent. At 200, refitting rather than projecting through frozen50 changes l10 .684→.788 (+10.4 pp) and spatial .808→.970 (+16.2 pp), both p<.0001. Additional candidates alone do not capture most of the possible benefit; fit geometry matters.

Rows are read from Q4's demo-curve metadata/hand-back; pickle sizes are from the exact served artifacts in the §9 source index. The 50→500 π0.5 tail fits are 26.091→95.265 MB l10 and 21.910→47.409 MB spatial. Compact representation accounting is approximately 580 B/row for AWM, 586 for BlindAWM, 632 for K7/C10, 370 for wrist guard-only, and 376 for Q6; D1 adds nine bytes/row to its K7 base. These summary representation counts are not the full pickle divided by row count. Two-camera PCA bases alone take 17,039,360 bytes (R4); action payloads of 280 B/π0.5 row or 448 B/GR00T row are already in the artifacts, with padded 32-column storage. GR00T has 2,645/29,631 l10 and 1,063/11,751 spatial rows at 50/500. Growth fits are .0526/.0351 GB while the full raw stores are 3.459/1.645 GB per copy; deployment must distinguish these storage boundaries.

Deployment recommendation: prefer a periodic refit after acquiring a material batch, keep acquisition outside evaluation, and retain raw provenance if future refits are required. When policy data are purchased, report its amortization separately. Where only 50 episodes are available, use the measured 50-cell controller choices in §9 instead of importing a 500-library metric. Spatial-200 is a candidate memory compromise for a later noninferiority test; l10 still shows a useful library-size trend. No online append speed, GPU-resident growth, opposite-fold result, or zero-cost acquisition is established.

## 7. B1: offline solver versus hand settings

| Cell | One changed parameter | Hand SR @ IR | Solve SR @ IR | Solve − hand | Offline LOTO action / state RMS Δ |
| --- | --- | --- | --- | --- | --- |
| p-sp-50 | ridge_main .1 → 1 | 0.910 @ 0.17259 | 0.900 @ 0.16556 | -1.00 [-3.40, +1.40]; 17/22; p=0.5224 | -0.013415 / -0.004306 |
| p-sp-500 | kref 8 → 5 | 0.978 @ 0.13122 | 0.988 @ 0.12807 | +1.00 [-0.40, +2.60]; 10/5; p=0.3018 | -0.005565 / -0.001529 |
| p-l10-500 | kref 8 → 5 | 0.906 @ 0.19653 | 0.886 @ 0.20161 | -2.00 [-5.20, +1.20]; 27/37; p=0.2604 | -0.002482 / -0.000732 |
| g-sp-50 | ridge_main .1 → 1 | 0.884 @ 0.14800 | 0.894 @ 0.14800 | +1.00 [-1.80, +3.80]; 28/23; p=0.5758 | -0.011326 / -0.004447 |
| g-sp-500 | state_scale 1 → 3 | 0.968 @ 0.14800 | 0.962 @ 0.14800 | -0.60 [-2.20, +1.00]; 7/10; p=0.6291 | -0.006471 / -0.005729 |
| g-l10-500 | state_scale 1 → 3 | 0.708 @ 0.14800 | 0.692 @ 0.14800 | -1.60 [-5.40, +2.20]; 43/51; p=0.4705 | -0.012615 / -0.007166 |

| Equal-weight paired pool | Solve mean SR | Hand mean SR | Δ pp [95% clustered CI] |
| --- | --- | --- | --- |
| p | 0.92467 | 0.93133 | -0.67 pp, 95% CI [-2.07, +0.73] |
| g | 0.84933 | 0.85333 | -0.40 pp, 95% CI [-2.07, +1.27] |
| all | 0.88700 | 0.89233 | -0.53 pp, 95% CI [-1.63, +0.57] |

The frozen offline objective predicts improvement in all six cells (lower action and successor RMS); closed-loop SR rises in only two of six, and no per-cell McNemar test is significant. The all-cell mean and suite-stratified interval are given in the pooled table. That pooled estimand weights each tested cell equally; it is not a pooled six-task deployment or six independent confirmations. The prospective test rejects the claim that these score improvements reliably rank closed-loop SR. It does not prove that every offline selection method is impossible, nor that the tested parameters have no effect.

Configuration checks use `r05_b1/arms_in.json` and `arms.json`, compared with reference specs. π0.5 l10-500 and spatial-500 hand are K7 tail repeats; GR00T l10-500 and spatial-500 hand are R2 CL2 repeats. Those are pooled in §1/§9 rather than cherry-picking B1 hand .906 as the K7 estimate. At spatial-50 the B1 hand and solve use the explicit ridge-capable AWM3 implementation; the paired intervention is clean inside B1, but equality to the original historical fit/tie handling has not been independently proved here. **ASSUMPTION:** do not pool spatial-50 B1 hand with the older implementation as a certified repeat. R5's K7 spatial-50 .894 remains the direct R4-cell fill. This conservative qualification does not change B1's six paired results.

Offline values above are read from `ideation_B/solver_choices.json`, not invented SR predictions. The proxy is conditional LOEO/LOTO in a library-fitted representation, not a fully inductive train-on-nine-tasks / evaluate-on-unseen-task pipeline. Ideation B's normalization audit and frozen candidate choices constrain leakage; action-unit conventions for the large library remain inherited and must be labeled, and no 50 fit is credited with 500-library statistics. Lower recorded-action error and successor error do not observe the controller's changed rollout distribution, rescue value or execution-length dynamics.

**Direct answer to the owner:** the selected finite parameter grid can be solved offline for its declared proxy, but **R4 + R5 do not support solving deployment SR by that proxy alone**. R4's tail/phase and noprog misrankings, unresolved K5 geometry-to-call-value mapping, and R5's six prospective cells are consistent with a missing control/outcome objective. RIT/LDA's solvable fit does not supply that objective automatically. Keep the hand defaults where the only justification for a replacement is lower proxy loss; use B1 as falsification evidence and avoid another retrospective winner search. A later owner-approved solver study should preregister one objective, establish inductive split/normalization rules, and validate rankings on fresh repeated rollouts.

As a secondary final-repeat check, l10-500 B1 solve versus the four-run K7 hand configuration gives +0.00 pp, 95% CI [-2.55, +2.55]; the pure-cache tail versus that same pool gives -5.80 pp, 95% CI [-8.75, -2.90]. This does not replace B1's prespecified one-to-one solve/hand contrast above.

## 8. Q5 live GPU retrieval shadow

Both arms are pure CL2 and serve CPU actions. This is live shadow measurement under load, not GPU-served SR and not the earlier stored-query microbenchmark. Raw sources are `R/r05_q5/runs/<arm>/server_*/decisions_*.jsonl`, restricted to each journal's accepted attempt. The audit also aggregates all raw shadow rows; accepted counts below are the relevant rollout cohort. Chunk tolerance is 1e-4 and confidence tolerance 1e-3, as specified by Q5.

| Metric | 50 (35,750 decisions) | 500 (31,835 decisions) |
| --- | --- | --- |
| top1_agree | 35749/35750 (99.99720%) | 31835/31835 (100.00000%) |
| top16_set_agree | 35750/35750 (100.00000%) | 31832/31835 (99.99058%) |
| top16_order_agree | 35709/35750 (99.88531%) | 31785/31835 (99.84294%) |
| chunk_agree | 35550/35750 (99.44056%) | 31730/31835 (99.67017%) |
| confidence_agree | 35677/35750 (99.79580%) | 31821/31835 (99.95602%) |
| Failed decisions | 0 | 0 |
| Step-zero / later chunk mismatches | 200 / 0 | 104 / 1 |
| Max chunk error | 0.0172251463 | 0.0138542056 |
| Max non-step-zero chunk error | 0.0000452399 | 0.0062419772 |
| Max confidence error | 2.00635213 | 0.32095606 |

| Latency metric (ms) | 50 p50 / p90 / max | 500 p50 / p90 / max |
| --- | --- | --- |
| GPU retrieval event | 1.965 / 3.558 / 17.624 | 2.539 / 4.793 / 34.556 |
| GPU retrieval wall incl. final copy | 2.873 / 6.379 / 31.450 | 3.700 / 8.431 / 42.482 |
| Pooling-to-GPU-retrieval event | 4.415 / 7.629 / 26.440 | 5.456 / 10.023 / 58.421 |
| CPU method/query | 5.073 / 8.983 / 25.526 | 8.123 / 14.664 / 40.282 |
| CPU complete recorded path | 9.123 / 15.449 / 33.513 | 12.599 / 21.976 / 59.490 |

| Shadow arm | SR @ owner IR | Shadow − R2 CL2 |
| --- | --- | --- |
| l10-50 | 0.644 @ 0.15200 | +1.40 [-1.40, +4.20]; 30/23; p=0.4101 |
| l10-500 | 0.756 @ 0.15200 | -1.20 [-4.00, +1.60]; 22/28; p=0.4799 |

The .644/.756 shadow SRs differ from CL2 .630/.768 by +1.4/−1.2 pp, both well inside observed run noise. The shadow action contract predicts the same action source, not identical realized success in a nondeterministic evaluation; neither exact test establishes an SR difference, and no formal equivalence margin was tested. Both owner IRs remain .152. Extra shadow work is a measurement overhead outside that model-inference ledger; do not claim that running both paths accelerates the robot.

The live **500-library post-initialization exception is material**: task 8, init 45, accepted attempt 1, step 46 (`r5q5_p_l10_500_cl2_shadow:eval:8:45`) has chunk error .0062419772 and confidence error .0132845941. Top-1 is 22292 on both paths, but the top-16 set/order differ. The stored-query hand-back had no such later chunk failure. A step-zero-only CPU fallback therefore does not close the live parity gap. Numerical cause is not established by this analysis; candidate boundary/tie behavior and action synthesis need investigation using this exact row.

GPU median wall query time is 2.873/3.700 ms versus CPU query 5.073/8.123 ms; the corresponding pooling-to-retrieval event medians are 4.415/5.456 ms versus CPU path 9.123/12.599 ms. Event and wall scopes differ and these are marginal quantiles under shared load, not sums or paired end-to-end speedups. Maximum GPU wall latency reaches 31.450/42.482 ms. Actual live timings are larger than the offline microbenchmarks, so §9's K8 surcharge and this live latency table must stay distinct.

Before promoting `serve`, resolve or explicitly contain **all** strict action/confidence mismatches, reproduce the live boundary case, define matching numerical/tie rules and a fallback policy, and have the coordinator run a paired GPU-served evaluation with measured end-to-end latency. Q5 currently implements a separate retrieval graph, not retrieval fused into the stage-1 graph; K7 blind guards, Q6 wrist and GR00T do not inherit this validation. No such follow-up is launched here. Q5 [hand-back](q5_gpu/HANDBACK.md) reports AWM GPU resident arrays 52,841,036 / 204,780,236 bytes, while fit pickles are 24,631,208 / 94,143,229 bytes and the native 1,103,155,631-byte preload is retained. A fast compact retrieval representation is not yet a reduced total-memory deployment.

The subsequent [K8 idle re-check](../r04/k8_search_latency/IDLE_RECHECK.md) reproduced single-process CPU search p50 of approximately **1.1–2.2 ms**, supporting K8's method-cost table. Live serving with **24 connections per server** instead measured CPU search **3–6× slower**, while GPU retrieval event p50 was approximately **2.0–2.7 ms**. The source attributes the CPU gap to GIL/thread contention; these concurrency-dependent serving costs must remain distinct from single-process method costs and from GPU wall latency including copies. This qualification changes latency interpretation, not the owner inference ledger.

## 9. Eight-cell frontiers, search cost, repricing, and deployable bytes

Tables below include every completed R4/R5 point in the inherited R4 arm inventory, together with its R2/R3 references. Growth/demo points are in §6 because they use different library sizes and, for grow250, a different pair manifest; mixing those into a 50/500 frontier would violate the comparison. K2 and randomized K5 points are listed separately below. Repeats and shadow observations remain visible in the raw tables; the deployment assessment then pools certified repeats.

**F** = nondominated observed point, **D** = another point has no greater cost and no lower SR, with at least one strict inequality. This is a point-estimate Pareto label, not statistical superiority. **A** = wrist-price assumption; **U** = unverified legacy cost, excluded from dominance; **H** = descriptive historical trace; **P** = proxy search measurement for an unbenchmarked R5 variant. Dominance is recomputed separately for owner (**O**), owner+search (**S**), and dummy_cached (**d**). A dash is an unavailable cost, not zero. Searchless pure inference has surcharge zero. Very close frontier memberships can be noise or artifact-timing differences; §1's repeat rule overrides a raw winner.

**Search panel:** K8's per-vision/per-blind CPU p50 query times are inherited from `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/an.py`'s `K8` table and the [K8 report](../r04/k8_search_latency/REPORT.md), not measured again here. Charge `(q_v v + q_b(1−v)) × 5/L` ms per five controls. The table uses R4's common 67.5-ms denominator for continuity: `IR_S = IR_owner + query_ms/67.5`. For π0.5 this is its owner full-inference reference; **for GR00T it is only a common-π-time surcharge proxy, not GR00T-native total IR**. A separate GR00T-native panel below uses 41.441827491 ms, the sum of `stage1_ms`, `stage2_ms`, and `stage3_full_loop_ms` in `exp/libero_groot/config/rit/cost_groot_libero_measured.json`. Stage owner coefficients stay as mandated. R5 C10/G10/Q6/B1 and one-block tails reuse the nearest K8 parent query costs; these exclude any unmeasured wrapper overhead. The exact q values used are included in the source index. Q5 live latencies remain §8; no GPU improvement is credited to other methods.

**Dummy panel:** full-camera π0.5 receives the stipulated −.048v×5/L only. Wrist and GR00T are held at owner prices with no claimed compatibility. The pure π0.5 tail moves to roughly .0523–.0535, and pure L=10 policy to .476. Thus dummy_cached makes the full-camera pure-cache spatial-500 tail cheaper than Q6 and restores that low-cost alternative to the frontier. No unmeasured joint wrist+dummy discount is invented.

### 9.1 p_l10, library 50: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r5t_p_l10_50_tail1uc` | 0.706 | 0.50261 | 0.00000 | 0.07640 | 0.08751 P | 0.05227 | F/F/F | 26.091 |
| `ph2c_l10_50` | 0.620 | 0.64166 | 0.00000 | 0.09753 | 0.11218 | 0.06673 | D/D/D | 26.091 |
| `cl2_l10_50` | 0.630 | 1.00000 | 0.00000 | 0.15200 | 0.16929 | 0.10400 | D/D/D | 24.631 |
| `cslG_50` | 0.114 | 1.00000 | 0.00000 | 0.15200 | 0.17068 | 0.10400 | D/D/D | 26.011 |
| `cslGS_50` | 0.628 | 1.00000 | 0.00000 | 0.15200 | 0.17141 | 0.10400 | D/D/D | 26.011 |
| `r5q5_p_l10_50_cl2_shadow` | 0.644 | 1.00000 | 0.00000 | 0.15200 | 0.16929 | 0.10400 | D/D/D | 24.631 |
| `r5q1_c10_p_l10_50` | 0.830 | 0.50318 | 0.12293 | 0.18073 | 0.19483 P | 0.15658 | F/F/F | 32.705 |
| `r5q6_p_l10_50_tail` A | 0.792 | 0.59746 | 0.18438 | 0.19853 | 0.21109 P | 0.19853 | D/D/D | 21.227 |
| `wrist_l10_50` A | 0.734 | 1.00000 | 0.20113 | 0.23580 | 0.25460 | 0.23580 | D/D/D | 21.125 |
| `r5q1_d1_p_l10_50` | 0.812 | 0.59569 | 0.17654 | 0.24025 | 0.25630 P | 0.21166 | D/D/D | 32.728 |
| `k7_tail1ug_50` | 0.806 | 0.59533 | 0.17809 | 0.24151 | 0.25755 | 0.21294 | D/D/D | 32.705 |
| `k7_ph2g_50` | 0.700 | 0.64101 | 0.20112 | 0.26798 | 0.28661 | 0.23721 | D/D/D | 32.705 |
| `k1_ph2k5_50` | 0.776 | 0.71279 | 0.19996 | 0.27791 | 0.29339 | 0.24369 | D/D/D | 26.091 |
| `per6_50` | 0.764 | 1.00000 | 0.16090 | 0.28845 | 0.30573 | 0.24045 | D/D/D | 25.990 |
| `g50_np4` | 0.690 | 1.00000 | 0.16651 | 0.29320 | 0.31758 | 0.24520 | D/D/D | 32.603 |
| `perk5_50` | 0.792 | 1.00000 | 0.19205 | 0.31486 | 0.33215 | 0.26686 | D/D/D | 25.990 |
| `g50` | 0.740 | 1.00000 | 0.20214 | 0.32342 | 0.34780 | 0.27542 | D/D/D | 32.602 |
| `perk3_50` | 0.832 | 1.00000 | 0.32680 | 0.42912 | 0.44641 | 0.38112 | F/F/F | 25.990 |
| `awm_h70_50` | 0.816 | 1.00000 | 0.34346 | 0.44325 | 0.46764 | 0.39525 | D/D/D | 32.602 |
| `inf_l10_L10` | 0.904 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.47600 | F/F/F | — (policy) |
| `inf_l10_L10b` | 0.900 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.47600 | D/D/D | — (policy) |
| `awm_h50_50` | 0.868 | 1.00000 | 0.53236 | 0.60345 | 0.62783 | 0.55545 | D/D/D | 32.602 |
| `inf_l10_s1001` | 0.848 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |
| `inf_l10_s2001` | 0.860 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |
| `trace_dual_L5` H | 0.844 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |

### 9.2 p_l10, library 500: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r5t_p_l10_500_tail1uc` | 0.828 | 0.50328 | 0.00000 | 0.07650 | 0.08877 P | 0.05234 | F/F/F | 95.265 |
| `ph2c_l10_500` | 0.778 | 0.59676 | 0.00000 | 0.09071 | 0.10617 | 0.06206 | D/D/D | 95.265 |
| `cl2_l10_500` | 0.768 | 1.00000 | 0.00000 | 0.15200 | 0.17281 | 0.10400 | D/D/D | 94.143 |
| `cslG_500` | 0.222 | 1.00000 | 0.00000 | 0.15200 | 0.17656 | 0.10400 | D/D/D | 94.379 |
| `cslGS_500` | 0.772 | 1.00000 | 0.00000 | 0.15200 | 0.17824 | 0.10400 | D/D/D | 94.379 |
| `r5q5_p_l10_500_cl2_shadow` | 0.756 | 1.00000 | 0.00000 | 0.15200 | 0.17281 | 0.10400 | D/D/D | 94.143 |
| `r5q1_c10_p_l10_500` | 0.866 | 0.50365 | 0.09967 | 0.16108 | 0.17836 P | 0.13690 | F/D/F | 142.349 |
| `r5q6_p_l10_500_tail` A | 0.892 | 0.57649 | 0.14508 | 0.16209 | 0.17702 P | 0.16209 | F/F/F | 117.133 |
| `wrist_l10_500` A | 0.820 | 1.00000 | 0.12256 | 0.16524 | 0.18871 | 0.16524 | D/D/D | 116.011 |
| `k7_ph2g` | 0.862 | 0.57287 | 0.10678 | 0.17762 | 0.19847 | 0.15012 | D/D/D | 142.349 |
| `r5b_p_l10_500_hand` | 0.906 | 0.56896 | 0.12978 | 0.19653 | 0.21563 P | 0.16922 | F/F/F | 142.349 |
| `r5x_p_l10_500_k7tail_repb` | 0.882 | 0.57089 | 0.13309 | 0.19964 | 0.21879 | 0.17223 | D/D/D | 142.349 |
| `r5b_p_l10_500_solve` | 0.886 | 0.57194 | 0.13523 | 0.20161 | 0.22080 P | 0.17416 | D/D/D | 142.349 |
| `r5x_p_l10_500_k7tail_repa` | 0.876 | 0.57283 | 0.13616 | 0.20253 | 0.22174 | 0.17504 | D/D/D | 142.349 |
| `k7_tail1ug` | 0.880 | 0.57280 | 0.13705 | 0.20328 | 0.22249 | 0.17579 | D/D/D | 142.349 |
| `k1_ph2k8` | 0.822 | 0.64196 | 0.12498 | 0.20356 | 0.21965 | 0.17274 | D/D/D | 95.265 |
| `r5q1_d1_p_l10_500` | 0.870 | 0.57547 | 0.14150 | 0.20746 | 0.22674 P | 0.17984 | D/D/D | 142.614 |
| `k1_ph2g` | 0.850 | 0.58642 | 0.14394 | 0.21119 | 0.23254 | 0.18305 | D/D/D | 142.348 |
| `g500_np4` | 0.808 | 1.00000 | 0.07314 | 0.21402 | 0.24507 | 0.16602 | D/D/D | 141.227 |
| `per12_500` | 0.828 | 1.00000 | 0.07510 | 0.21569 | 0.23650 | 0.16769 | D/D/D | 94.143 |
| `k1_tail1ug` | 0.878 | 0.58523 | 0.15830 | 0.22319 | 0.24301 | 0.19510 | D/D/D | 142.348 |
| `k7_ph1g` | 0.862 | 0.69455 | 0.14190 | 0.22590 | 0.24907 | 0.19256 | D/D/D | 142.349 |
| `g500` | 0.864 | 1.00000 | 0.10167 | 0.23822 | 0.26927 | 0.19022 | D/D/D | 141.227 |
| `g500_repa` | 0.850 | 1.00000 | 0.10859 | 0.24408 | 0.27513 | 0.19608 | D/D/D | 141.227 |
| `k7_b0g` | 0.830 | 1.00000 | 0.10912 | 0.24453 | 0.27551 | 0.19653 | D/D/D | 142.349 |
| `k1_clk1g` | 0.868 | 0.69783 | 0.16919 | 0.24954 | 0.27363 | 0.21605 | D/D/D | 142.348 |
| `g500_repb` | 0.832 | 1.00000 | 0.11657 | 0.25085 | 0.28190 | 0.20285 | D/D/D | 141.227 |
| `per8_500` | 0.850 | 1.00000 | 0.11974 | 0.25354 | 0.27435 | 0.20554 | D/D/D | 94.143 |
| `k1_ph1g` | 0.838 | 0.70485 | 0.18569 | 0.26460 | 0.28886 | 0.23077 | D/D/D | 142.348 |
| `k1_b0g` | 0.842 | 1.00000 | 0.13993 | 0.27066 | 0.30219 | 0.22266 | D/D/D | 142.348 |
| `awm500_h70` | 0.872 | 1.00000 | 0.34101 | 0.44118 | 0.47223 | 0.39318 | D/D/D | 141.227 |
| `inf_l10_L10` | 0.904 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.47600 | D/D/D | — (policy) |
| `inf_l10_L10b` | 0.900 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.47600 | D/D/D | — (policy) |
| `inf_l10_s1001` | 0.848 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |
| `inf_l10_s2001` | 0.860 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |
| `trace_dual_L5` H | 0.844 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |

### 9.3 p_sp, library 50: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r5t_p_sp_50_tail1uc` | 0.838 | 0.51016 | 0.00000 | 0.07754 | 0.08905 P | 0.05306 | F/F/F | 21.910 |
| `r5q6_p_spatial_50_tail` A | 0.902 | 0.56277 | 0.11017 | 0.12999 | 0.14198 P | 0.12999 | F/F/D | 15.430 |
| `r5q1_c10_p_sp_50` | 0.910 | 0.51117 | 0.07518 | 0.14145 | 0.15476 P | 0.11692 | F/F/F | 26.077 |
| `cl2_sp_50` | 0.800 | 1.00000 | 0.00000 | 0.15200 | 0.16910 | 0.10400 | D/D/D | 21.342 |
| `r5t_p_sp_g50_wrist_rep` A | 0.928 | 1.00000 | 0.11159 | 0.15539 | 0.17413 | 0.15539 | F/F/F | 15.389 |
| `wrist_sp_50` A | 0.924 | 1.00000 | 0.11239 | 0.15612 | 0.17486 | 0.15612 | D/D/D | 15.389 |
| `r5b_p_sp_50_solve` | 0.900 | 0.55537 | 0.09569 | 0.16556 | 0.17972 P | 0.13891 | D/D/D | 26.084 |
| `r5b_p_sp_50_hand` | 0.910 | 0.55895 | 0.10334 | 0.17259 | 0.18682 P | 0.14576 | D/D/D | 26.084 |
| `k7_sp_tail50` | 0.894 | 0.56204 | 0.10770 | 0.17676 | 0.19105 P | 0.14979 | D/D/D | 26.077 |
| `k1_sp50_ph2g` | 0.884 | 0.58821 | 0.13501 | 0.20390 | 0.22245 | 0.17567 | D/D/D | 26.076 |
| `sp_g50` | 0.888 | 1.00000 | 0.13496 | 0.26645 | 0.28912 | 0.21845 | D/D/D | 26.036 |
| `sp_perk3_50` | 0.938 | 1.00000 | 0.31686 | 0.42070 | 0.43780 | 0.37270 | F/F/F | 21.870 |
| `sp_awm_h70_50` | 0.980 | 1.00000 | 0.32825 | 0.43036 | 0.45303 | 0.38236 | F/F/F | 26.036 |
| `inf_sp_L10` | 0.986 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.47600 | F/F/F | — (policy) |
| `sp_awm_h50_50` U | 0.986 | 1.00000 | 0.57218 | 0.63721 † | 0.65987 | 0.58921 | U/U/U | 26.036 |
| `inf_sp_s1001` | 0.992 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |
| `inf_sp_s2001` | 0.994 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | F/F/F | — (policy) |
| `trace_dual_L5` H | 0.986 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |

### 9.4 p_sp, library 500: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r5q6_p_spatial_500_tail` A | 0.990 | 0.52736 | 0.05021 | 0.07419 | 0.08664 P | 0.07419 | F/F/F | 50.789 |
| `tail1uc_sp_500` | 0.982 | 0.51440 | 0.00000 | 0.07819 | 0.08997 | 0.05350 | D/D/F | 47.409 |
| `ph2c_sp_500` | 0.954 | 0.54355 | 0.00000 | 0.08262 | 0.09669 | 0.05653 | D/D/D | 47.409 |
| `wrist_sp_500` A | 0.974 | 1.00000 | 0.06863 | 0.11682 | 0.13746 | 0.11682 | D/D/D | 50.373 |
| `r5q1_c10_p_sp_500` | 0.982 | 0.51392 | 0.05008 | 0.12058 | 0.13592 P | 0.09591 | D/D/D | 66.501 |
| `r5b_p_sp_500_solve` | 0.988 | 0.53151 | 0.05576 | 0.12807 | 0.14383 P | 0.10256 | D/D/D | 66.501 |
| `k7_sp_tail1ug` | 0.982 | 0.53086 | 0.05607 | 0.12824 | 0.14397 | 0.10275 | D/D/D | 66.501 |
| `r5b_p_sp_500_hand` | 0.978 | 0.53264 | 0.05927 | 0.13122 | 0.14700 P | 0.10565 | D/D/D | 66.501 |
| `k7_sp_ph2g` | 0.968 | 0.54457 | 0.06754 | 0.14005 | 0.15778 | 0.11391 | D/D/D | 66.501 |
| `cl2_sp_500` | 0.954 | 1.00000 | 0.00000 | 0.15200 | 0.17228 | 0.10400 | D/D/D | 46.993 |
| `sp_per12_500` | 0.976 | 1.00000 | 0.06147 | 0.20413 | 0.22441 | 0.15613 | D/D/D | 46.993 |
| `sp_g500` | 0.974 | 1.00000 | 0.06377 | 0.20608 | 0.23738 | 0.15808 | D/D/D | 66.084 |
| `sp_awm500_h70` | 0.978 | 1.00000 | 0.33719 | 0.43794 | 0.46924 | 0.38994 | D/D/D | 66.084 |
| `inf_sp_L10` | 0.986 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.47600 | D/D/D | — (policy) |
| `inf_sp_s1001` | 0.992 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |
| `inf_sp_s2001` | 0.994 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | F/F/F | — (policy) |
| `trace_dual_L5` H | 0.986 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.95200 | D/D/D | — (policy) |

### 9.5 g_l10, library 50: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gb_l10_50_tail2u` | 0.570 | 0.33810 | 0.00000 | 0.05004 | 0.05840 | 0.05004 | F/F/F | 28.136 |
| `r5x_g_l10_50_tail1u` | 0.608 | 0.50189 | 0.00000 | 0.07428 | 0.08509 | 0.07428 | F/F/F | 28.136 |
| `gb_l10_50_ph2` | 0.566 | 0.67190 | 0.00000 | 0.09944 | 0.11397 | 0.09944 | D/D/D | 28.136 |
| `cl2_g_l10_50` | 0.552 | 1.00000 | 0.00000 | 0.14800 | 0.16560 | 0.14800 | D/D/D | 26.673 |
| `r5q2_g_l10_50_G10` | 0.718 | 0.50254 | 0.12988 | 0.18503 | 0.19585 P | 0.18503 | F/F/F | 28.136 |
| `r5q2_g_l10_policy_L10` | 0.866 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.50000 | F/F/F | — (policy) |
| `trace_dual_L5` H | 0.870 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | F/F/F | — (policy) |

### 9.6 g_l10, library 500: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gb_l10_500_tail2u` | 0.808 | 0.33905 | 0.00000 | 0.05018 | 0.05956 | 0.05018 | F/F/F | 118.431 |
| `r5x_g_l10_500_tail1u` | 0.830 | 0.50337 | 0.00000 | 0.07450 | 0.08682 | 0.07450 | F/F/F | 118.431 |
| `gb_l10_500_ph2` | 0.724 | 0.64119 | 0.00000 | 0.09490 | 0.11099 | 0.09490 | D/D/D | 118.431 |
| `cl2_g_l10_500` | 0.706 | 1.00000 | 0.00000 | 0.14800 | 0.16849 | 0.14800 | D/D/D | 117.304 |
| `r5b_g_l10_500_hand` | 0.708 | 1.00000 | 0.00000 | 0.14800 | 0.16849 P | 0.14800 | D/D/D | 117.304 |
| `r5b_g_l10_500_solve` | 0.692 | 1.00000 | 0.00000 | 0.14800 | 0.16849 P | 0.14800 | D/D/D | 117.304 |
| `r5q2_g_l10_500_G10` | 0.828 | 0.50336 | 0.13094 | 0.18605 | 0.19837 P | 0.18605 | D/D/D | 118.431 |
| `r5q2_g_l10_policy_L10` | 0.866 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.50000 | F/F/F | — (policy) |
| `trace_dual_L5` H | 0.870 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | F/F/F | — (policy) |

### 9.7 g_sp, library 50: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gb_sp_50_tail2u` | 0.864 | 0.34644 | 0.00000 | 0.05127 | 0.05960 | 0.05127 | F/F/F | 22.842 |
| `r5x_g_sp_50_tail1u` | 0.868 | 0.50874 | 0.00000 | 0.07529 | 0.08597 | 0.07529 | F/F/F | 22.842 |
| `gb_sp_50_ph2` | 0.834 | 0.60379 | 0.00000 | 0.08936 | 0.10281 | 0.08936 | D/D/D | 22.842 |
| `cl2_g_sp_50` | 0.888 | 1.00000 | 0.00000 | 0.14800 | 0.16486 | 0.14800 | D/D/D | 22.249 |
| `r5b_g_sp_50_hand` | 0.884 | 1.00000 | 0.00000 | 0.14800 | 0.16486 P | 0.14800 | D/D/D | 22.808 |
| `r5b_g_sp_50_solve` | 0.894 | 1.00000 | 0.00000 | 0.14800 | 0.16486 P | 0.14800 | F/F/F | 22.808 |
| `r5q2_g_spatial_50_G10` | 0.920 | 0.50926 | 0.14386 | 0.19794 | 0.20862 P | 0.19794 | F/F/F | 22.842 |
| `r5q2_g_spatial_policy_L10` | 0.938 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.50000 | F/F/F | — (policy) |
| `trace_dual_L5` H | 0.940 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | F/F/F | — (policy) |

### 9.8 g_sp, library 500: all raw points

| Arm alias | SR | v | m | Owner IR | +CPU search | dummy IR | O/S/d | Fit MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gb_sp_500_tail2u` | 0.946 | 0.34965 | 0.00000 | 0.05175 | 0.06063 | 0.05175 | F/F/F | 58.604 |
| `r5x_g_sp_500_tail1u` | 0.964 | 0.51020 | 0.00000 | 0.07551 | 0.08695 | 0.07551 | F/F/F | 58.604 |
| `gb_sp_500_ph2` | 0.976 | 0.52932 | 0.00000 | 0.07834 | 0.09187 | 0.07834 | F/F/F | 58.604 |
| `cl2_g_sp_500` | 0.966 | 1.00000 | 0.00000 | 0.14800 | 0.16656 | 0.14800 | D/D/D | 58.157 |
| `r5b_g_sp_500_hand` | 0.968 | 1.00000 | 0.00000 | 0.14800 | 0.16656 P | 0.14800 | D/D/D | 58.157 |
| `r5b_g_sp_500_solve` | 0.962 | 1.00000 | 0.00000 | 0.14800 | 0.16656 P | 0.14800 | D/D/D | 58.157 |
| `r5q2_g_spatial_500_G10` | 0.952 | 0.51102 | 0.14477 | 0.19897 | 0.21043 P | 0.19897 | D/D/D | 58.605 |
| `r5q2_g_spatial_policy_L10` | 0.938 | 1.00000 | 1.00000 | 0.50000 | 0.50000 | 0.50000 | D/D/D | — (policy) |
| `trace_dual_L5` H | 0.940 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | D/D/D | — (policy) |

† **Legacy R3 conflict:** `sp_awm_h50_50` has summary N=10,848, M=6,207, giving owner .63721; accepted raw-log deduplication gives N=10,851, M=5,874, with 764 duplicate decisions and 63 conflicting UIDs. The R4 printed cost is not used to conceal this mismatch. SR .986 is journal/summary-consistent; its cost is unverified and it cannot dominate a verified point. Other legacy summary count costs passed their raw audit. This anomaly does not affect the R5 ledger calculations.

K2 is excluded from the system by owner ruling (MISS step reduction is not an allowed system component). The next table preserves its observed SR and counts for inventory, and shows the mandated **full-MISS counterfactual accounting**, not its actual shortened-stage price; consult R4's K2 appendix for that different cost basis. K5 randomized interventions are diagnostic data, not ordinary deployable controller proposals.

| Excluded diagnostic alias | Reason | SR | v / m | Full-MISS counterfactual IR (K2) / owner IR (K5) | Fit MB |
| --- | --- | --- | --- | --- | --- |
| `inf_l10_k2_s1101` | K2 owner-excluded | 0.850 | 1.00000 / 1.00000 | 1.00000 | — (policy) |
| `inf_sp_k2_s1101` | K2 owner-excluded | 0.990 | 1.00000 / 1.00000 | 1.00000 | — (policy) |
| `g500_k2` | K2 owner-excluded | 0.842 | 1.00000 / 0.11425 | 0.24888 | 141.227 |
| `g50_k2` | K2 owner-excluded | 0.732 | 1.00000 / 0.21866 | 0.33742 | 32.602 |
| `perk5_500_k2` | K2 owner-excluded | 0.852 | 1.00000 / 0.19170 | 0.31456 | 94.143 |
| `perk5_50_k2` | K2 owner-excluded | 0.790 | 1.00000 / 0.19222 | 0.31501 | 25.990 |
| `sp_g500_k2` | K2 owner-excluded | 0.974 | 1.00000 / 0.06628 | 0.20821 | 66.084 |
| `sp_g50_k2` | K2 owner-excluded | 0.896 | 1.00000 / 0.13515 | 0.26661 | 26.036 |
| `k5_g500_r1` | K5 randomized | 0.826 | 1.00000 / 0.10798 | 0.24357 | 141.227 |
| `k5_g500_r2` | K5 randomized | 0.836 | 1.00000 / 0.10824 | 0.24379 | 141.227 |
| `k5_g50_r1` | K5 randomized | 0.768 | 1.00000 / 0.19330 | 0.31592 | 32.602 |
| `k5_g50_r2` | K5 randomized | 0.738 | 1.00000 / 0.19892 | 0.32068 | 32.602 |

### 9.9 Certified-repeat pooling and deployment interpretation

| Configuration pool | Runs | SR mean | Owner IR (pooled decisions) |
| --- | --- | --- | --- |
| stock3 | 3 | 0.84867 | 0.24444 |
| K7_l10_500_all4 | 4 | 0.88600 | 0.20052 |
| K7_sp500_2 | 2 | 0.98000 | 0.12973 |
| wrist_sp50_2 | 2 | 0.92600 | 0.15575 |
| CL2_g_l10_500_2 | 2 | 0.70700 | 0.14800 |
| CL2_g_sp500_2 | 2 | 0.96700 | 0.14800 |
| pi_policy_l10_L10_2 | 2 | 0.90200 | 0.50000 |
| pi_policy_l10_L5_2 | 2 | 0.85400 | 1.00000 |
| pi_policy_sp_L5_2 | 2 | 0.99300 | 1.00000 |

| Cell | Replicate-aware owner frontier (in ascending cost) | Owner+K8 frontier changes | dummy frontier changes |
| --- | --- | --- | --- |
| p_l10_50 | `r5t_p_l10_50_tail1uc` → `r5q1_c10_p_l10_50` → `perk3_50` → `pi_policy_l10_L10_2` | same membership | same membership |
| p_l10_500 | `r5t_p_l10_500_tail1uc` → `r5q1_c10_p_l10_500` → `r5q6_p_l10_500_tail` → `pi_policy_l10_L10_2` | added none; removed r5q1_c10_p_l10_500 | same membership |
| p_sp_50 | `r5t_p_sp_50_tail1uc` → `r5q6_p_spatial_50_tail` → `r5q1_c10_p_sp_50` → `wrist_sp50_2` → `sp_perk3_50` → `sp_awm_h70_50` → `inf_sp_L10` → `pi_policy_sp_L5_2` | same membership | added none; removed r5q6_p_spatial_50_tail |
| p_sp_500 | `r5q6_p_spatial_500_tail` → `pi_policy_sp_L5_2` | same membership | added tail1uc_sp_500; removed none |
| g_l10_50 | `gb_l10_50_tail2u` → `r5x_g_l10_50_tail1u` → `r5q2_g_l10_50_G10` → `r5q2_g_l10_policy_L10` | same membership | same membership |
| g_l10_500 | `gb_l10_500_tail2u` → `r5x_g_l10_500_tail1u` → `r5q2_g_l10_policy_L10` | same membership | same membership |
| g_sp_50 | `gb_sp_50_tail2u` → `r5x_g_sp_50_tail1u` → `r5b_g_sp_50_solve` → `r5q2_g_spatial_50_G10` → `r5q2_g_spatial_policy_L10` | same membership | same membership |
| g_sp_500 | `gb_sp_500_tail2u` → `r5x_g_sp_500_tail1u` → `gb_sp_500_ph2` | same membership | same membership |

Replicate pooling removes the misleading raw claim that B1 hand K7 .906 beats pure policy .904 at lower cost. Four-run K7 is .88600 @ .20052, policy L=10 .902 @ .5. Replicate B changes the pooled estimate slightly but does not change the replicate-aware frontier membership in any cost panel. Under wrist prices, Q6 .892 @ .16209 point-dominates pooled K7, but their paired interval crosses zero and the price is assumed. Pooled estimates therefore change which configurations deserve attention without declaring a statistically proven winner. The K8 proxy makes Q6 slightly cheaper than C10 at l10-500 (.17702 versus .17836), so C10 leaves that search frontier; the tiny cost ordering depends on assumed wrist prices and inherited query timings. Dummy_cached makes full-camera C10 dominate Q6 at spatial-50, while restoring pure tail to the spatial-500 frontier. Historical trace_dual is shown in the raw tables, not pooled across protocols in this configuration panel. There is no unique “best” without an SR/cost/memory utility; the following choices are explicit deployment judgments, not fitted optima.

| Cell | Balanced / simple deployable choice | Observed SR @ owner IR | Exact fit bytes | Interpretation / alternative |
| --- | --- | --- | --- | --- |
| p_l10_50 | `r5q1_c10_p_l10_50` | 0.830 @ 0.18073 | 32704614 | C10 is the balanced cost/SR point; pure tail .706 costs .07640. Policy L=10 remains the higher-SR choice. |
| p_l10_500 | `r5q6_p_l10_500_tail` | 0.892 @ 0.16209 | 117133059 | Conditional Q6 candidate; wrist price and SR need confirmation. Conservative two-camera alternative is pooled K7 (.88600 @ .20052, 142.349 MB); C10 is cheaper with possible SR loss. |
| p_sp_50 | `wrist_sp_50` | 0.924 @ 0.15612 | 15389123 | Guard-only wrist pool .926 @ .15575 has a reproduced SR gain. Fit shown is the identical guard-only representation; full-camera C10 is .910 @ .14145. |
| p_sp_500 | `tail1uc_sp_500` | 0.982 @ 0.07819 | 47409015 | Pure-cache tail .982 is a simple no-policy option, .05350 with dummy. Conditional Q6 .990 @ .07419 uses 50.789 MB but has no proven SR advantage. |
| g_l10_50 | `r5q2_g_l10_50_G10` | 0.718 @ 0.18503 | 28135738 | G10 policy use helps the small bank; tail15 .570 @ .05004 is the low-cost option, policy L=10 .866 is the accuracy option. |
| g_l10_500 | `r5x_g_l10_500_tail1u` | 0.830 @ 0.07450 | 118431046 | Tail10 .830 @ .07450; tail15 .808 @ .05018 remains a valid cheaper choice because the SR gap is unresolved. G10 is dominated. |
| g_sp_50 | `r5q2_g_spatial_50_G10` | 0.920 @ 0.19794 | 22842135 | G10 .920 @ .19794 buys SR over pure tails. Tail15 .864 @ .05127 is the low-cost choice; no significant reason to spend extra for tail10 .868. |
| g_sp_500 | `gb_sp_500_ph2` | 0.976 @ 0.07834 | 58604454 | R4 phase .976 @ .07834 is retained; tail15 .946 @ .05175 is the cheaper alternative. Scheduled policy does not help this cell. |

These fit bytes exclude policy checkpoints, raw library archives, the native preload where still required, process duplication and device copies. Original native cache pickles are 1,103,155,631 / 430,792,483 bytes for π0.5 l10/spatial and 1,068,314,575 / 429,351,282 for GR00T (R4 footprint and Q5 hand-back). Compact-fit claims are useful, but removing the preload is a separate deployment change. Fifty-episode fits use only their deployment library; B1's external action normalization and any inherited outcome-trained diagnostics retain the qualifications in §7.

### 9.10 GR00T-native search sensitivity

| GR00T arm | K8 query ms / five controls | Owner IR | +search / 67.5 common proxy | +search / 41.441827 native reference |
| --- | --- | --- | --- | --- |
| `cl2_g_l10_50` | 1.1880 | 0.14800 | 0.16560 | 0.17667 |
| `cl2_g_l10_500` | 1.3830 | 0.14800 | 0.16849 | 0.18137 |
| `cl2_g_sp_50` | 1.1380 | 0.14800 | 0.16486 | 0.17546 |
| `cl2_g_sp_500` | 1.2530 | 0.14800 | 0.16656 | 0.17824 |
| `gb_l10_500_ph2` | 1.0865 | 0.09490 | 0.11099 | 0.12111 |
| `gb_l10_500_tail2u` | 0.6329 | 0.05018 | 0.05956 | 0.06545 |
| `gb_l10_50_ph2` | 0.9804 | 0.09944 | 0.11397 | 0.12310 |
| `gb_l10_50_tail2u` | 0.5641 | 0.05004 | 0.05840 | 0.06365 |
| `gb_sp_500_ph2` | 0.9131 | 0.07834 | 0.09187 | 0.10037 |
| `gb_sp_500_tail2u` | 0.5996 | 0.05175 | 0.06063 | 0.06622 |
| `gb_sp_50_ph2` | 0.9078 | 0.08936 | 0.10281 | 0.11127 |
| `gb_sp_50_tail2u` | 0.5618 | 0.05127 | 0.05960 | 0.06483 |
| `r5q2_g_spatial_50_G10` | 0.7211 | 0.19794 | 0.20862 | 0.21534 |
| `r5q2_g_spatial_500_G10` | 0.7734 | 0.19897 | 0.21043 | 0.21763 |
| `r5q2_g_l10_50_G10` | 0.7301 | 0.18503 | 0.19585 | 0.20265 |
| `r5q2_g_l10_500_G10` | 0.8316 | 0.18605 | 0.19837 | 0.20612 |
| `r5q2_g_spatial_policy_L10` | 0.0000 | 0.50000 | 0.50000 | 0.50000 |
| `r5q2_g_l10_policy_L10` | 0.0000 | 0.50000 | 0.50000 | 0.50000 |
| `r5b_g_sp_50_hand` | 1.1380 | 0.14800 | 0.16486 | 0.17546 |
| `r5b_g_sp_50_solve` | 1.1380 | 0.14800 | 0.16486 | 0.17546 |
| `r5b_g_sp_500_hand` | 1.2530 | 0.14800 | 0.16656 | 0.17824 |
| `r5b_g_sp_500_solve` | 1.2530 | 0.14800 | 0.16656 | 0.17824 |
| `r5b_g_l10_500_hand` | 1.3830 | 0.14800 | 0.16849 | 0.18137 |
| `r5b_g_l10_500_solve` | 1.3830 | 0.14800 | 0.16849 | 0.18137 |
| `r5x_g_l10_500_tail1u` | 0.8316 | 0.07450 | 0.08682 | 0.09456 |
| `r5x_g_l10_50_tail1u` | 0.7294 | 0.07428 | 0.08509 | 0.09188 |
| `r5x_g_sp_500_tail1u` | 0.7725 | 0.07551 | 0.08695 | 0.09415 |
| `r5x_g_sp_50_tail1u` | 0.7205 | 0.07529 | 0.08597 | 0.09268 |

This denominator sensitivity leaves the key GR00T choices intact: ten-/fifteen-step tails remain much cheaper than mixed G10, G10 remains useful at 50 and point-dominated at 500. It does not turn K8 microbenchmarks into measurements of the loaded Q5 pipeline or a combined stage-1 graph.

### 9.11 Complete source index and exclusions

Each index entry expands to the per-arm summary/journal/DONE paths defined at the start. `fit_path`, exact bytes and SHA256 of summary/journal/marker are retained in `/tmp/r5_analysis/analysis.json`. q_v/q_b are the K8 ms values actually used; “—” means no measurement, with pure inference charged zero search. Every completed arm appears either in an eight-cell table, the diagnostic appendix, or §6. The source index supplies exact filenames instead of relying on changing root summaries.

| Alias | run / arm | Fit bytes | K8 q_v / q_b ms |
| --- | --- | --- | --- |
| `inf_l10_s1001` | `r04_cost/r4f_p_l10_inf_s1001` | 692443574 | 0.000 / 0 |
| `inf_l10_s2001` | `r04_cost/r4f_p_l10_inf_s2001` | 692443574 | 0.000 / 0 |
| `inf_sp_s1001` | `r04_cost/r4f_p_sp_inf_s1001` | 267012406 | 0.000 / 0 |
| `inf_sp_s2001` | `r04_cost/r4f_p_sp_inf_s2001` | 267012406 | 0.000 / 0 |
| `inf_l10_k2_s1101` | `r04_cost/r4b2_p_l10_inf_k2_s1101` | 692443574 | 0.000 / 0 |
| `inf_sp_k2_s1101` | `r04_cost/r4b2_p_sp_inf_k2_s1101` | 267012406 | 0.000 / 0 |
| `inf_l10_L10` | `r04_cost/r4f_p_l10_inf_k10_L10` | 692443574 | 0.000 / 0 |
| `inf_l10_L10b` | `r04_blind/r4b3_p_l10_50_inferL10` | 26091341 | 0.000 / 0 |
| `inf_sp_L10` | `r04_cost/r4f_p_sp_inf_k10_L10` | 267012406 | 0.000 / 0 |
| `cl2_l10_50` | `r02_g50/oscl50_p_l10_cl2` | 24631208 | 1.167 / 0 |
| `cl2_l10_500` | `r02_g500/oscl500_p_l10_cl2` | 94143229 | 1.405 / 0 |
| `cl2_sp_50` | `r02_g50/oscl50_p_sp_cl2` | 21341661 | 1.154 / 0 |
| `cl2_sp_500` | `r02_g500/oscl500_p_sp_cl2` | 46993114 | 1.369 / 0 |
| `cl2_g_l10_50` | `r02_g50/oscl50_g_l10_cl2` | 26672701 | 1.188 / 0 |
| `cl2_g_l10_500` | `r02_g500/oscl500_g_l10_cl2` | 117303700 | 1.383 / 0 |
| `cl2_g_sp_50` | `r02_g50/oscl50_g_sp_cl2` | 22249307 | 1.138 / 0 |
| `cl2_g_sp_500` | `r02_g500/oscl500_g_sp_cl2` | 58156565 | 1.253 / 0 |
| `g50` | `r03_mx/r3mx_p_l10_g` | 32602497 | 1.646 / 0 |
| `g500` | `r03_mx/r3mx_p_l10_g500` | 141226847 | 2.096 / 0 |
| `g500_repa` | `r04_rep/r4rep_p_l10_g500_a` | 141226847 | 2.096 / 0 |
| `g500_repb` | `r04_rep/r4rep_p_l10_g500_b` | 141226847 | 2.096 / 0 |
| `perk5_50` | `r03_mx/r3mx_p_l10_perk5` | 25989715 | 1.167 / 0 |
| `perk3_50` | `r03_mx/r3mx_p_l10_perk3` | 25989715 | 1.167 / 0 |
| `awm_h70_50` | `r03_mx/r3mx_p_l10_awm_h70` | 32602497 | 1.646 / 0 |
| `awm_h50_50` | `r03_mx/r3mx_p_l10_awm_h50` | 32602497 | 1.646 / 0 |
| `awm500_h70` | `r03_mx/r3mx_p_l10_awm500_h70` | 141226847 | 2.096 / 0 |
| `sp_g50` | `r03_mx/r3mx_p_sp_g` | 26036137 | 1.530 / 0 |
| `sp_perk3_50` | `r03_mx/r3mx_p_sp_perk3` | 21869602 | 1.154 / 0 |
| `sp_awm_h70_50` | `r03_mx/r3mx_p_sp_awm_h70` | 26036137 | 1.530 / 0 |
| `sp_awm_h50_50` | `r03_mx/r3mx_p_sp_awm_h50` | 26036137 | 1.530 / 0 |
| `sp_awm500_h70` | `r03_mx/r3mx_p_sp_awm500_h70` | 66084240 | 2.113 / 0 |
| `g500_np4` | `r04_frontier/r4_p_l10_g500_np4` | 141226864 | 2.096 / 0 |
| `per8_500` | `r04_frontier/r4_p_l10_per8_500` | 94143229 | 1.405 / 0 |
| `per12_500` | `r04_frontier/r4_p_l10_per12_500` | 94143229 | 1.405 / 0 |
| `g50_np4` | `r04_frontier/r4_p_l10_g50_np4` | 32602514 | 1.646 / 0 |
| `per6_50` | `r04_frontier/r4_p_l10_per6_50` | 25989715 | 1.167 / 0 |
| `sp_g500` | `r04_frontier/r4_p_sp_g500` | 66084240 | 2.113 / 0 |
| `sp_per12_500` | `r04_frontier/r4_p_sp_per12_500` | 46993114 | 1.369 / 0 |
| `g500_k2` | `r04_cost/r4b2_p_l10_g500_k2` | 141226847 | 2.096 / 0 |
| `g50_k2` | `r04_cost/r4b2_p_l10_g_k2` | 32602497 | 1.646 / 0 |
| `perk5_500_k2` | `r04_cost/r4b2_p_l10_perk5_500_k2` | 94143252 | 1.405 / 0 |
| `perk5_50_k2` | `r04_cost/r4b2_p_l10_perk5_k2` | 25989715 | 1.167 / 0 |
| `sp_g500_k2` | `r04_cost/r4b2_p_sp_g500_k2` | 66084240 | 2.113 / 0 |
| `sp_g50_k2` | `r04_cost/r4b2_p_sp_g_k2` | 26036137 | 1.530 / 0 |
| `k1_b0g` | `r04_blind/r4b3_p_l10_500_b0g` | 142348450 | 2.128 / 0.467 |
| `k1_ph1g` | `r04_blind/r4b3_p_l10_500_ph1g` | 142348450 | 2.128 / 0.467 |
| `k1_ph2g` | `r04_blind/r4b3_p_l10_500_ph2g` | 142348450 | 2.128 / 0.467 |
| `k1_tail1ug` | `r04_blind/r4b3_p_l10_500_tail1ug` | 142348462 | 2.128 / 0.223 |
| `k1_clk1g` | `r04_blind/r4b3_p_l10_500_clk1g` | 142348441 | 2.128 / 0.467 |
| `k1_ph2k8` | `r04_blind/r4b3_p_l10_500_ph2k8` | 95264515 | 1.426 / 0.478 |
| `k1_ph2k5_50` | `r04_blind/r4b3_p_l10_50_ph2k5` | 26091353 | 1.272 / 0.482 |
| `ph2c_l10_500` | `r04_blind/r4b3_p_l10_500_ph2c` | 95264525 | 1.426 / 0.478 |
| `ph2c_l10_50` | `r04_blind/r4b3_p_l10_50_ph2c` | 26091363 | 1.272 / 0.482 |
| `ph2c_sp_500` | `r04_blind/r4b3_p_sp_500_ph2c` | 47409007 | 1.335 / 0.491 |
| `tail1uc_sp_500` | `r04_blind/r4b3_p_sp_500_tail1uc` | 47409015 | 1.335 / 0.223 |
| `k1_sp50_ph2g` | `r04_blind/r4b3_p_sp_50_ph2g` | 26076410 | 1.796 / 0.475 |
| `k7_b0g` | `r04_k7/r4k7_p_l10_500_b0g` | 142348550 | 2.091 / 0 |
| `k7_ph2g` | `r04_k7/r4k7_p_l10_500_ph2g` | 142348550 | 2.096 / 0.483 |
| `k7_ph1g` | `r04_k7/r4k7_p_l10_500_ph1g` | 142348550 | 2.045 / 0.469 |
| `k7_tail1ug` | `r04_k7/r4k7_p_l10_500_tail1ug` | 142348562 | 2.097 / 0.223 |
| `k7_ph2g_50` | `r04_k7/r4k7_p_l10_50_ph2g` | 32704515 | 1.686 / 0.493 |
| `k7_tail1ug_50` | `r04_k7/r4k7_p_l10_50_tail1ug` | 32704527 | 1.657 / 0.238 |
| `k7_sp_ph2g` | `r04_k7/r4k7_p_sp_500_ph2g` | 66500540 | 1.798 / 0.478 |
| `k7_sp_tail1ug` | `r04_k7/r4k7_p_sp_500_tail1ug` | 66500552 | 1.804 / 0.223 |
| `wrist_l10_500` | `r04_b4w/r4b4_p_l10_g500_wrist` | 116011290 | 1.584 / 0 |
| `wrist_l10_50` | `r04_b4w/r4b4_p_l10_g50_wrist` | 21125076 | 1.269 / 0 |
| `wrist_sp_500` | `r04_b4w/r4b4_p_sp_g500_wrist` | 50372731 | 1.393 / 0 |
| `wrist_sp_50` | `r04_b4w/r4b4_p_sp_g50_wrist` | 15389123 | 1.265 / 0 |
| `cslG_500` | `r04_csl/r4b4_p_l10_500_cslG` | 94379345 | 1.658 / 0 |
| `cslGS_500` | `r04_csl/r4b4_p_l10_500_cslGS` | 94379347 | 1.771 / 0 |
| `cslG_50` | `r04_csl/r4b4_p_l10_50_cslG` | 26011134 | 1.261 / 0 |
| `cslGS_50` | `r04_csl/r4b4_p_l10_50_cslGS` | 26011136 | 1.310 / 0 |
| `k5_g500_r1` | `r04_k5/r4k5_p_l10_g500_r1` | 141226847 | 2.096 / 0 |
| `k5_g500_r2` | `r04_k5/r4k5_p_l10_g500_r2` | 141226847 | 2.096 / 0 |
| `k5_g50_r1` | `r04_k5/r4k5_p_l10_g50_r1` | 32602497 | 1.646 / 0 |
| `k5_g50_r2` | `r04_k5/r4k5_p_l10_g50_r2` | 32602497 | 1.646 / 0 |
| `gb_l10_500_ph2` | `r04_gblind/r4b3_g_l10_500_ph2` | 118431038 | 1.432 / 0.469 |
| `gb_l10_500_tail2u` | `r04_gblind/r4b3_g_l10_500_tail2u` | 118431046 | 1.432 / 0.223 |
| `gb_l10_50_ph2` | `r04_gblind/r4b3_g_l10_50_ph2` | 28135613 | 1.232 / 0.465 |
| `gb_l10_50_tail2u` | `r04_gblind/r4b3_g_l10_50_tail2u` | 28135621 | 1.232 / 0.223 |
| `gb_sp_500_ph2` | `r04_gblind/r4b3_g_sp_500_ph2` | 58604454 | 1.300 / 0.478 |
| `gb_sp_500_tail2u` | `r04_gblind/r4b3_g_sp_500_tail2u` | 58604462 | 1.300 / 0.223 |
| `gb_sp_50_ph2` | `r04_gblind/r4b3_g_sp_50_ph2` | 22842010 | 1.201 / 0.461 |
| `gb_sp_50_tail2u` | `r04_gblind/r4b3_g_sp_50_tail2u` | 22842018 | 1.201 / 0.223 |
| `r5t_p_l10_500_tail1uc` | `r05_ptail/r5t_p_l10_500_tail1uc` | 95264533 | 1.426 / 0.223 |
| `r5t_p_l10_50_tail1uc` | `r05_ptail/r5t_p_l10_50_tail1uc` | 26091371 | 1.272 / 0.223 |
| `r5t_p_sp_50_tail1uc` | `r05_ptail/r5t_p_sp_50_tail1uc` | 21909613 | 1.308 / 0.223 |
| `r5t_p_sp_g50_wrist_rep` | `r05_ptail/r5t_p_sp_g50_wrist_rep` | 15389123 | 1.265 / 0 |
| `r5q4_p_l10_grow250_refit` | `r05_growth/r5q4_p_l10_grow250_refit` | 52611819 | — |
| `r5q4_p_l10_grow250_frozen` | `r05_growth/r5q4_p_l10_grow250_frozen` | 52612030 | — |
| `r5q4_p_sp_grow250_refit` | `r05_growth/r5q4_p_sp_grow250_refit` | 35134074 | — |
| `r5q4_p_sp_grow250_frozen` | `r05_growth/r5q4_p_sp_grow250_frozen` | 35134284 | — |
| `r5q4d_p_l10_100_refit_tail1` | `r05_demo_curve/r5q4d_p_l10_100_refit_tail1` | 34145321 | — |
| `r5q4d_p_l10_200_refit_tail1` | `r05_demo_curve/r5q4d_p_l10_200_refit_tail1` | 49494769 | — |
| `r5q4d_p_l10_300_refit_tail1` | `r05_demo_curve/r5q4d_p_l10_300_refit_tail1` | 64689528 | — |
| `r5q4d_p_l10_200_frozen50_tail1` | `r05_demo_curve/r5q4d_p_l10_200_frozen50_tail1` | 49494984 | — |
| `r5q4d_p_sp_100_refit_tail1` | `r05_demo_curve/r5q4d_p_sp_100_refit_tail1` | 25104085 | — |
| `r5q4d_p_sp_200_refit_tail1` | `r05_demo_curve/r5q4d_p_sp_200_refit_tail1` | 30732026 | — |
| `r5q4d_p_sp_300_refit_tail1` | `r05_demo_curve/r5q4d_p_sp_300_refit_tail1` | 36241257 | — |
| `r5q4d_p_sp_200_frozen50_tail1` | `r05_demo_curve/r5q4d_p_sp_200_frozen50_tail1` | 30732276 | — |
| `r5q1_c10_p_l10_50` | `r05_q1/r5q1_c10_p_l10_50` | 32704614 | 1.657 / 0.238 |
| `r5q1_c10_p_l10_500` | `r05_q1/r5q1_c10_p_l10_500` | 142348649 | 2.097 / 0.223 |
| `r5q1_c10_p_sp_50` | `r05_q1/r5q1_c10_p_sp_50` | 26076609 | 1.530 / 0.238 |
| `r5q1_c10_p_sp_500` | `r05_q1/r5q1_c10_p_sp_500` | 66500639 | 1.804 / 0.223 |
| `r5q1_d1_p_l10_50` | `r05_q1/r5q1_d1_p_l10_50` | 32728495 | 1.657 / 0.238 |
| `r5q1_d1_p_l10_500` | `r05_q1/r5q1_d1_p_l10_500` | 142614036 | 2.097 / 0.223 |
| `r5q2_g_spatial_50_G10` | `r05_q2/r5q2_g_spatial_50_G10` | 22842135 | 1.201 / 0.223 |
| `r5q2_g_spatial_500_G10` | `r05_q2/r5q2_g_spatial_500_G10` | 58604579 | 1.300 / 0.223 |
| `r5q2_g_l10_50_G10` | `r05_q2/r5q2_g_l10_50_G10` | 28135738 | 1.232 / 0.223 |
| `r5q2_g_l10_500_G10` | `r05_q2/r5q2_g_l10_500_G10` | 118431163 | 1.432 / 0.223 |
| `r5q2_g_spatial_policy_L10` | `r05_q2/r5q2_g_spatial_policy_L10` | 278713319 | 0.000 / 0 |
| `r5q2_g_l10_policy_L10` | `r05_q2/r5q2_g_l10_policy_L10` | 693501089 | 0.000 / 0 |
| `r5q6_p_l10_50_tail` | `r05_q6/r5q6_p_l10_50_tail` | 21227220 | 1.269 / 0.223 |
| `r5q6_p_l10_500_tail` | `r05_q6/r5q6_p_l10_500_tail` | 117133059 | 1.584 / 0.223 |
| `r5q6_p_spatial_50_tail` | `r05_q6/r5q6_p_spatial_50_tail` | 15429622 | 1.265 / 0.223 |
| `r5q6_p_spatial_500_tail` | `r05_q6/r5q6_p_spatial_500_tail` | 50789097 | 1.393 / 0.223 |
| `r5b_p_sp_50_hand` | `r05_b1/r5b_p_sp_50_hand` | 26083701 | 1.530 / 0.238 |
| `r5b_p_sp_50_solve` | `r05_b1/r5b_p_sp_50_solve` | 26083853 | 1.530 / 0.238 |
| `r5b_p_sp_500_hand` | `r05_b1/r5b_p_sp_500_hand` | 66500574 | 1.804 / 0.223 |
| `r5b_p_sp_500_solve` | `r05_b1/r5b_p_sp_500_solve` | 66500534 | 1.804 / 0.223 |
| `r5b_p_l10_500_hand` | `r05_b1/r5b_p_l10_500_hand` | 142348584 | 2.097 / 0.223 |
| `r5b_p_l10_500_solve` | `r05_b1/r5b_p_l10_500_solve` | 142348512 | 2.097 / 0.223 |
| `r5b_g_sp_50_hand` | `r05_b1/r5b_g_sp_50_hand` | 22807846 | 1.138 / 0 |
| `r5b_g_sp_50_solve` | `r05_b1/r5b_g_sp_50_solve` | 22807850 | 1.138 / 0 |
| `r5b_g_sp_500_hand` | `r05_b1/r5b_g_sp_500_hand` | 58156609 | 1.253 / 0 |
| `r5b_g_sp_500_solve` | `r05_b1/r5b_g_sp_500_solve` | 58156636 | 1.253 / 0 |
| `r5b_g_l10_500_hand` | `r05_b1/r5b_g_l10_500_hand` | 117303744 | 1.383 / 0 |
| `r5b_g_l10_500_solve` | `r05_b1/r5b_g_l10_500_solve` | 117303771 | 1.383 / 0 |
| `r5q5_p_l10_50_cl2_shadow` | `r05_q5/r5q5_p_l10_50_cl2_shadow` | 24631208 | 1.167 / 0 |
| `r5q5_p_l10_500_cl2_shadow` | `r05_q5/r5q5_p_l10_500_cl2_shadow` | 94143229 | 1.405 / 0 |
| `r5x_g_l10_500_tail1u` | `r05_x/r5x_g_l10_500_tail1u` | 118431046 | 1.432 / 0.223 |
| `r5x_g_l10_50_tail1u` | `r05_x/r5x_g_l10_50_tail1u` | 28135621 | 1.232 / 0.223 |
| `r5x_g_sp_500_tail1u` | `r05_x/r5x_g_sp_500_tail1u` | 58604462 | 1.300 / 0.223 |
| `r5x_g_sp_50_tail1u` | `r05_x/r5x_g_sp_50_tail1u` | 22842018 | 1.201 / 0.223 |
| `r5x_p_l10_500_k7tail_repa` | `r05_x/r5x_p_l10_500_k7tail_repa` | 142348562 | 2.097 / 0.223 |
| `r5x_p_l10_500_k7tail_repb` | `r05_x/r5x_p_l10_500_k7tail_repb` | 142348562 | 2.097 / 0.223 |
| `k7_sp_tail50` | `r04_k7/r4k7_p_sp_50_tail1ug` | 26076522 | 1.530 / 0.238 |

SKIPPED markers (excluded, not unfinished numerical slots):

- `r03_mx/r3mx_p_l10_ev_h50`, `r03_mx/r3mx_p_sp_ev_h50`.
- `r04_blind/r4b3_p_l10_50_ph2g`, `r04_blind/r4b3_p_l10_50_b0g`.
- `r04_cost/r4f_p_l10_inf_k2_L10`, `r04_cost/r4f_p_sp_inf_k2_L10`, `r04_cost/r4b2_p_l10_inf_k2_s1102`, `r04_cost/r4b2_p_sp_inf_k2_s1102`.

The R4 brief's deferred `r4_p_{l10,sp}_inf_s{1,2}` plans are represented by the later completed seeded pure-inference references, not additional measured arms. Smokes are excluded. Q3 produced no supported causal overlay and its rollout queue was canceled; K10 was canceled in the selection addendum; no Q6 phase variant or GR00T growth arm was run. These are absent experiments, not pending evidence. All requested R5 outcome slots are now filled, including replicate B.

## 10. Synthesis, method, library, control, and cost: R4 → R5

| Layer | What is held / changed | R5 evidence and implication |
| --- | --- | --- |
| Synthesis | Kernel action combination; kref changes at fixed 500 bank | B1 π0.5 kref 8→5 yields −2.0 pp l10 and +1.0 pp spatial, both unresolved. Lower offline kernel error does not validate a better SR synthesis rule. Stored second-half execution belongs to control, not a new action synthesizer. |
| Method / representation | Ridge, state weight, fitted PCA/whitening/confidence, wrist key | B1 ridge/state changes do not improve SR reliably. Q4 refit versus frozen200 changes SR +10.4/+16.2 pp at identical candidate contents; refitting matters, but this does not identify which individual fit component causes the gain. Q6 changes observation and guard geometry as well as price. |
| Library | Contents / scale / acquisition provenance | Grow250 refit adds +14.4/+18.0 pp over same-manifest CL2-50; serving IR stays .152, acquisition is paid. The demo curve improves broadly, with a nonmonotone spatial point and endpoint source/kref confounds. Fifty- and 500-library conclusions are kept separate. |
| Control including execution | Anchors, cached tails, policy commitment, rescue timing | Pure π0.5 tail gains +7.6/+6.0 pp l10 and +3.8/+2.8 pp spatial without policy calls. G10 buys SR at 50 but not 500. C10 realizes all nonterminal policy tails and lowers m, while unresolved SR losses prevent a preservation claim. D1 has insufficient reach/evidence. |
| Cost implementation | Owner stage coefficients, camera work, retrieval path, storage | Dummy is a stipulated −.048v repricing; wrist is a ratio-transfer assumption. K8 adds CPU query cost outside inference. Q5 measures live latency but fails strict action parity. Fits shrink representation storage, while native preload/raw-bank/device memory remain separate. |

These axes interact. In particular, longer cached execution changes both the visited states and when guards request policy; refitting changes retrieval and hence guard reach; a wrist key changes observation quality and MISS rates as well as visual cost. The following arithmetic decompositions compare observed decision ratios, not causal mediation estimates and not a promise that the components can be added across independently run arms.

| Observed change | ΔSR pp | Vision term ΔIR | MISS term ΔIR | Total ΔIR |
| --- | --- | --- | --- | --- |
| π l10-50: CL2 → pure tail | +7.60 | -0.07560 | +0.00000 | -0.07560 |
| π l10-500: CL2 → pure tail | +6.00 | -0.07550 | +0.00000 | -0.07550 |
| π sp-50: CL2 → pure tail | +3.80 | -0.07446 | +0.00000 | -0.07446 |
| π sp-500: CL2 → pure tail | +2.80 | -0.07381 | +0.00000 | -0.07381 |
| π l10-50: R4 K7 → C10 | +2.40 | -0.01401 | -0.04678 | -0.06078 |
| π l10-500: R4 K7 → C10 | -1.40 | -0.01051 | -0.03169 | -0.04220 |
| G l10-50: tail10 → G10 | +11.00 | +0.00010 | +0.11066 | +0.11076 |
| G l10-500: tail10 → G10 | -0.20 | -0.00000 | +0.11156 | +0.11156 |
| G sp-50: tail10 → G10 | +5.20 | +0.00008 | +0.12257 | +0.12265 |
| G sp-500: tail10 → G10 | -1.20 | +0.00012 | +0.12334 | +0.12346 |

Pure-tail savings are entirely fewer visual decisions; C10 combines fewer vision anchors and fewer fresh policy calls per five controls. G10 versus a pure ten-step tail has almost the same vision cadence, so its extra cost is predominantly scheduled policy. Q6's full-camera counterfactual in §5 separates the observed control ratios from the assumed wrist stage price. Pure policy L=10 halves cost by executing ten controls/request and provides the execution baseline needed to avoid crediting every long-chunk gain to caching.

The R4 K5 result is inherited, not rederived: CALL−CACHE is +3.4 pp [0.4,6.6] at 500 and −0.2 pp [−4.0,3.2] at 50. Q3's [hand-back](q3_callvalue/HANDBACK.md) strengthens its limits: the scale contrast is +3.6 pp [−1.0,8.4], the 500 task-level interval includes zero, only 2/48 and 7/48 leaf contexts meet sample support, and no context has a positive conservative saving bound. All fitted suppression probabilities are zero under its declared 1 pp loss constraint. `arms_q3.json=[]` is a valid negative result. The apparent scale sign reversal is not a learned deployment rule, and these randomized stock-guard data cannot be transported automatically to K7, C10, GR00T or spatial. MISS inference is not an established general-purpose saving pool.

## 11. Settled results, rejected directions, and ranked proposals for the owner

**Settled within the measured protocol:** cached ten-step commitment improves π0.5 SR in all four cells while approximately halving pure-cache owner IR; the four-run K7 configuration retains a positive advantage over repeated stock on l10-500; wrist guard-only spatial-50 reproduces its favorable SR direction; paid growth and refitting improve held-out SR substantially; scheduled GR00T policy calls help 50-library tails but are unnecessary on the observed 500-library frontier. The 50 requirement materially changes the deployment answer: spatial-500 may drop policy, while spatial-50 and l10-50 still benefit from intervention. These are measured benchmark conclusions, not guarantees across tasks or hardware.

**Rejected for current adoption:** selecting B1 settings because offline RMS improves; promoting D1 from a single small-effect run; universal G10 scheduling at 500; automatic wrist+tail stacking at 50; treating kref endpoints as identical; equating a nonsignificant difference with preserved SR; claiming free library growth; suppressing policy calls using Q3's unsupported contexts; using K2 in the system; promoting GPU `serve` from top-1 agreement alone; or ranking deployment using the lucky .906 K7 repeat. R4's sparse-library phase continuation and relaxed no-progress ideas remain unrevived. “Rejected” here means unsupported by the current evidence, not mathematically impossible.

R5 is the last round in the current owner goal. The following are **ranked proposals for an owner decision**, not authorized or launched experiments:

1. **Close the reliability and price questions around the useful points.** The prescribed three-run and all-four-run K7 checks are complete and both pass the paired positive-effect criterion. If a new round is approved, repeat Q6/C10 against repeated K7 at both scales, predeclare an SR-loss margin and cost objective, and measure the wrist stage price on the actual graph. The final Q6-500 versus four-run K7 interval [−2.25,+3.50] pp and C10 interval [−4.35,+0.30] show why neither “same SR” claim is ready. Spatial-50 wrist guard-only deserves priority over unvalidated stack changes.
2. **Finish the retrieval implementation with strict parity and real memory accounting.** Reproduce Q5's task-8/init-45/step-46 top-16 boundary mismatch, resolve step-zero and later action/confidence differences, then ask the coordinator for a matched GPU-served evaluation and latency distribution. Integrate the supported controller only after this gate; measure native preload removal and actual resident bytes before claiming deployment memory reduction. Dummy_cached can be considered under its existing parity contract independently of unsupported wrist stacking.
3. **Test targeted GR00T intervention and commitment length where evidence supports value.** Focus on the 50 cells where G10 buys +11.0/+5.2 pp, retaining policy L=10 and both pure-tail lengths as controls. At 500, keep pure tail/phase as the baseline. A repeated, predeclared 10-versus-15 comparison is needed before paying roughly 50% more vision cost for the small numerical tail10 SR gains. Any adaptive rule needs bounded blindness, mandatory anchors and actual paid-call accounting.
4. **Validate library growth as a lifecycle decision.** Repeat the acquisition/evaluation split in reverse, hold controller and kref fixed, compare periodic refit with frozen append, and measure absolute cost-to-success as well as serving/lifecycle IR. Test the spatial-200 memory option with a declared noninferiority margin. Preserve the provenance distinction between demonstration data, paid policy rollouts, and outcome-trained fits; no GR00T growth extrapolation is justified yet.
5. **Reformulate offline selection around a prospective control objective.** Keep B1's negative SR-ranking result. A later study should distinguish synthesis error, dynamics/execution error and intervention value; enforce inductive/conditional split labels and library-only normalization where intended; preregister candidates and test ranks on new repeated loops. If causal call-value learning is reconsidered, collect randomized evidence under the exact intended controller and satisfy support requirements, rather than weakening Q3's constraints after seeing its null result.

**Completion:** all 52 R5 arms, both required K7 repeat pools, their dependent comparisons, and all eleven analysis sections are complete. The final frontier uses the four-run K7 estimate; the remaining proposals above require a separate owner decision.

