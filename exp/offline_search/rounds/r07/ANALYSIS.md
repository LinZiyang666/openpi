# R7 analysis — stage-level allocation: quantitative results

A1 final snapshot: **2026-09-30T05:59:26.910508-05:00 (CDT)**. All 28 frozen evaluation arms are complete. Rules follow [SELECTION §5 and the pre-test §8 clarifications](SELECTION.md); [the analysis brief](ANALYSIS_BRIEF.md) assigns the mechanism and generality discussion to A2.

**Success preservation fails for extra blind execution, while the stage/state checks help.** Allowing one extra five-control block only in stable stages lowers owner cost in all eight cells, but changes success by **-1.78 [-2.81, -0.76] pp** versus the no-call cache, failing the preregistered −1 pp lower-bound threshold. Stage and state checks improve success over extensions based on structural support alone by **+1.98 [+0.88, +3.08] pp**, so the signal rule passes. That comparison also spends more on vision; it is not a matched-cost test.

**Wrist-camera allocation and stage-tilted calls pass their respective rules.** On π0.5, using only the wrist camera in easy stages changes success by **-0.05 [-1.40, +1.28] pp**, with lower owner cost in all four cells under the stated wrist-price assumption. The lower bound clears its −1.5 pp threshold by only 0.10 pp. Tilting policy calls toward stage/state events improves success over uniform calls by **+2.70 [+1.05, +4.35] pp**, and all four comparisons meet the per-cell cost-match limit.

**The cheapest observed match to pure-policy success does not improve.** Twenty new points are non-dominated by the frozen historical points; fifteen remain on the combined frontier. None of the new points clears the conservative nominal 2 pp noninferiority bound against pure L10 or pure L5. Mechanism and generality are reserved for A2 in §§6–8.

## 1. Data quality and cost reconciliation

Each arm contains the same 500 accepted task/init pairs: ten tasks × fifty inits. Completion requires a summary and either the plain `.DONE` marker or a `.manifest_*.DONE` marker. The stage-bounded and uniform-follow arms used the full Cartesian set without a manifest field; the pair sets are rechecked here, and the init-pool hash agreement is recorded in SELECTION §8 by the coordinator/A2.

| Family | Arms | Accepted episodes | Decisions N | Vision V | Calls M | Count-mismatch arms | Accepted exceptions |
|---|---|---|---|---|---|---|---|
| Stage-bounded follow (SF1) | 8 | 4000 | 177,583 | 77,960 | 0 | 0 | 0 |
| Structural-only follow (UF1) | 8 | 4000 | 180,585 | 66,542 | 0 | 0 | 0 |
| Stage wrist (SW) | 4 | 2000 | 86,181 | 43,536 | 0 | 0 | 0 |
| Uniform calls + stall (CU) | 4 | 2000 | 84,068 | 43,780 | 21,986 | 0 | 0 |
| Stage-tilted calls + stall (CT) | 4 | 2000 | 81,861 | 42,471 | 21,590 | 0 | 0 |

**Audit:** summary successes equal accepted-journal successes in all 28 arms; accepted server streams reproduce every arm's N/V/M and client request count exactly. Across R7 and the 84 in-root historical references, all **56,000/56,000** accepted episodes have joined termination reasons, with **0 surviving exceptions**. R7 has 0 exception attempts in timing logs, 0 purged journal records and 0 purges recorded by the chain. No R7 exception purge was needed. The join uses `(task_uid, run_id, attempt)`; stale attempts do not become failures in the accepted set. [Per-arm exception evidence](analysis_r7/a1_exception_audit.json) and [chain evidence](analysis_r7/a1_snapshot.json) retain the audit details.

Accepted-stream selection discarded 0 stale decision-prefix rows and collapsed 0 exact duplicate decisions. It requires contiguous steps, a client/server request-count match and consistent within-episode cumulative vision counters. Historical references retain R6's registered ledger tolerances (1 of the 84 loaded references); no such tolerance is used for R7. The frozen R6 frontier retains its original seven flagged arms and pricing.

**Owner cost.** For each non-wrist arm, IR is recomputed as `[c_v·V/N + (1−c_v)·M/N]·5/L`, with `c_v=.152` for π0.5 and `.148` for GR00T. All R7 requests have nominal `L=5`. Blind decisions cost zero; pure L10 costs `.5`, pure L5 costs `1`. The denominator is nominal five-control blocks, not measured applied controls. The summary's eager `ir_per_five_controls` is a separate cost convention and is never substituted for owner IR.

| Family | Owner IR range | Eager-summary IR range | Owner − eager range |
|---|---|---|---|
| SF1 | 0.060382–0.068940 | 0.056393–0.064386 | +0.003989 to +0.009502 |
| UF1 | 0.052149–0.060229 | 0.044905–0.056251 | +0.003517 to +0.007640 |
| SW | 0.055692–0.066376 | 0.052013–0.061992 | +0.003679 to +0.004385 |
| CU | 0.297293–0.301240 | 0.294608–0.298099 | +0.002590 to +0.005299 |
| CT | 0.287822–0.308056 | 0.285046–0.305583 | +0.002473 to +0.005293 |

**Stage wrist pricing.** Primary IR is the sum of accepted per-decision `owner_cost`, independently recomputed from actual camera/call counters: full look `.152`, wrist look `.055198`, missing-camera completion `.049890`, policy call `.848`. The wrist/completion prices use **R4's proportional-latency assumption**; these are cost estimates, not measured hardware speedups. The two-camera column charges `.152·V/N + .848·M/N` to the same decisions. Historical wrist points retain R6's frozen full-camera prices.

| π0.5 cell | Wrist owner IR | Two-camera owner IR | Eager-summary IR | Equal-episode owner IR | Logged − counted IR |
|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | 0.061602 | 0.076371 | 0.057533 | 0.059667 | 0 |
| π0.5 LIBERO-10 / 500 | 0.066376 | 0.076541 | 0.061992 | 0.064985 | 0 |
| π0.5 Spatial / 50 | 0.062539 | 0.077348 | 0.058408 | 0.061959 | 0 |
| π0.5 Spatial / 500 | 0.055692 | 0.078125 | 0.052013 | 0.056166 | 0 |

**Validation passed:** 17 regression checks; all 44 profile arms (880 episodes, 37,999 decisions) reproduce the independent profile report's outcomes/counts/costs; four R6 uniform/calibrated-call arms pass raw decision reconciliation; 84 references match frozen R6 outcomes/costs; all eight R6 Pareto sets and pure-L10 crossing costs reproduce. [Validation](analysis_r7/a1_validation.json), [cost rows](analysis_r7/a1_costs.csv), [accepted arm data](analysis_r7/a1_arms.json).

## 2. Preregistered acceptance verdicts

All success-rate differences below are **percentage points (pp)**. Intervals are two-sided 95% task-stratified init bootstraps, 10,000 draws, seed **20260930**. Replicates of A/B are averaged within each task/init pair before resampling: 500 paired clusters per cell. Pooled comparisons give cells equal weight, with independent within-cell resampling as in R6. Rules use unrounded endpoints and strict `>` bounds; cost matching uses inclusive `≤`. These are the preregistered unadjusted comparisons, not simultaneous certification. Non-significance is not equivalence.

| Rule | Acceptance condition | Computed evidence (95% CI) | Verdict |
|---|---|---|---|
| 1. Stage-bounded follow preserves SR while saving vision | SF−A lower > −1.00 pp; IR lower in ≥6/8 | SF−A -1.78 [-2.81, -0.76]; IR lower in 8/8 | not supported |
| 2. Stage/state signal adds value | SF−UF lower >0; OR SF has no significant loss vs A and UF−A lower <−1.00 pp | SF−UF +1.98 [+0.88, +3.08]; SF−A upper -0.76; UF−A lower -4.90 | supported |
| 3. Stage wrist preserves SR while saving vision | SW−A lower >−1.50 pp; IR lower in 4/4 | SW−A -0.05 [-1.40, +1.28]; IR lower in 4/4 | supported |
| 4. Stage-tilted calls beat uniform calls | CT−CU lower >0; absolute aggregate ΔIR ≤.015 in every cell | CT−CU +2.70 [+1.05, +4.35]; 4/4 cells IR matched | supported |
| 5. Frontier placement | Non-dominated by frozen R6 single-run points; show CP bounds; no certified NI | 20/28 non-dominated by R6; 15/28 remain after all R7 | reported pointwise (§4) |

Rule 2's second branch interprets “no significant loss” as the pooled SF−A interval's upper endpoint being at least zero. It does not mean that rule 1 passed or that equality was established. Direct SF−UF branch: **True**; no-significant-SF-loss clause: **False**; UF lower-bound clause: **True**.

| Pooled comparison | Cells | ΔSR pp [95% CI] | Mean of per-cell aggregate ΔIR |
|---|---|---|---|
| SF1-A | 8 | -1.78 [-2.81, -0.76] | -0.010921 |
| UF1-A | 8 | -3.75 [-4.90, -2.63] | -0.020258 |
| SF1-UF1 | 8 | +1.98 [+0.88, +3.08] | +0.009337 |
| SW-A | 4 | -0.05 [-1.40, +1.28] | -0.015584 |
| CT-CU | 4 | +2.70 [+1.05, +4.35] | +0.000540 |

**Rule 4 cost match (fixed before any CT test episode).** A cell outside `.015` is labelled **IR not matched in that cell**; a favourable pooled mean cannot replace this condition. The cell SR intervals below remain descriptive; the acceptance SR test is the four-cell pooled bound.

| Cell | CU aggregate IR | CT aggregate IR | CT−CU aggregate ΔIR [95% CI] | Equal-episode IR: CU / CT | CT−CU SR pp [95% CI] | Cost-match verdict |
|---|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | 0.300688 | 0.308056 | +0.007368 [+0.004048, +0.010743] | 0.300843 / 0.307215 | -1.00 [-4.40, +2.60] | IR matched |
| π0.5 Spatial / 50 | 0.297293 | 0.287822 | -0.009471 [-0.014783, -0.004213] | 0.296527 / 0.286558 | +1.60 [-0.60, +4.00] | IR matched |
| GR00T LIBERO-10 / 50 | 0.300898 | 0.299262 | -0.001636 [-0.004753, +0.001593] | 0.301789 / 0.298192 | +8.40 [+4.20, +12.60] | IR matched |
| GR00T Spatial / 50 | 0.301240 | 0.307139 | +0.005899 [+0.000726, +0.010970] | 0.300996 / 0.306422 | +1.80 [-0.60, +4.20] | IR matched |

The four-cell mean aggregate ΔIR is **+0.000540**. Equal-cell averages of equal-episode IR are CU **0.300039** and CT **0.299597**, difference **-0.000442**. These alternative summaries do not change the per-cell aggregate-IR gate.

## 3. Per-cell success, owner cost and paired differences

**Method key:** **A** is the no-call cache with ten-control commitment; **B** adds guard-triggered policy rescue. A and B are three-replicate means. **SF1** allows one extra five-control block when successor support, stable gripper stage and the state valve permit it; **UF1** allows that extension with structural support alone. **SW** uses the wrist camera in easy stages (π0.5 only). **CU** places calls uniformly with a calibrated stall trigger; **CT** tilts call placement by stage/state events, with the same stall trigger and nominal `.30` budget. **Pure L10** calls the policy every ten controls. Library sizes are 50 or 500 episodes; “—” means that variant was not evaluated in that cell.

| Model / suite / library | A (3-rep) | SF1 | UF1 | SW | CU | CT | B (3-rep) | Pure L10 |
|---|---|---|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | 71.40% @ 0.076396 | 68.40% @ 0.064558 | 64.80% @ 0.057267 | 70.40% @ 0.061602 | 86.40% @ 0.300688 | 85.40% @ 0.308056 | 82.73% @ 0.180715 | 90.40% @ 0.500000 |
| π0.5 LIBERO-10 / 500 | 82.73% @ 0.076515 | 83.40% @ 0.068940 | 79.00% @ 0.053247 | 84.60% @ 0.066376 | — | — | 87.87% @ 0.158804 | 90.40% @ 0.500000 |
| π0.5 Spatial / 50 | 83.67% @ 0.077525 | 79.40% @ 0.065469 | 78.00% @ 0.060229 | 82.40% @ 0.062539 | 94.60% @ 0.297293 | 96.20% @ 0.287822 | 91.00% @ 0.140423 | 98.60% @ 0.500000 |
| π0.5 Spatial / 500 | 97.60% @ 0.078111 | 97.40% @ 0.060382 | 95.80% @ 0.055577 | 97.80% @ 0.055692 | — | — | 98.20% @ 0.119490 | 98.60% @ 0.500000 |
| GR00T LIBERO-10 / 50 | 61.07% @ 0.074276 | 58.00% @ 0.065519 | 56.60% @ 0.054996 | — | 75.60% @ 0.300898 | 84.00% @ 0.299262 | 72.47% @ 0.221125 | 86.60% @ 0.500000 |
| GR00T LIBERO-10 / 500 | 83.00% @ 0.074494 | 83.60% @ 0.068400 | 80.40% @ 0.052149 | — | — | — | 87.20% @ 0.187600 | 86.60% @ 0.500000 |
| GR00T Spatial / 50 | 86.73% @ 0.075280 | 82.80% @ 0.063363 | 83.20% @ 0.058439 | — | 93.20% @ 0.301240 | 95.00% @ 0.307139 | 87.60% @ 0.147126 | 93.80% @ 0.500000 |
| GR00T Spatial / 500 | 96.40% @ 0.075529 | 95.40% @ 0.064128 | 94.80% @ 0.054155 | — | — | — | 95.87% @ 0.126535 | 93.80% @ 0.500000 |

Entries are **SR% @ owner IR**, with IR pooled over decisions within each run. A/B IR is the arithmetic mean of their three count-derived run ratios, following R6. All R7 arms and pure L10 are single runs.

| Cell | Method | ΔSR vs A pp [95% CI] | ΔSR vs B pp [95% CI] | ΔSR vs pure L10 pp [95% CI] | ΔIR vs A |
|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | SF1 | -3.00 [-6.40, +0.33] | -14.33 [-17.93, -10.87] | -22.00 [-26.20, -18.00] | -0.011838 |
| π0.5 LIBERO-10 / 50 | UF1 | -6.60 [-10.20, -3.00] | -17.93 [-21.80, -14.00] | -25.60 [-29.60, -21.40] | -0.019129 |
| π0.5 LIBERO-10 / 50 | SW | -1.00 [-4.47, +2.47] | -12.33 [-15.87, -8.80] | -20.00 [-24.00, -16.00] | -0.014794 |
| π0.5 LIBERO-10 / 50 | CU | +15.00 [+11.13, +18.73] | +3.67 [+0.27, +7.00] | -4.00 [-7.80, -0.40] | +0.224293 |
| π0.5 LIBERO-10 / 50 | CT | +14.00 [+9.87, +18.20] | +2.67 [-0.87, +6.33] | -5.00 [-8.80, -1.20] | +0.231661 |
| π0.5 LIBERO-10 / 500 | SF1 | +0.67 [-1.73, +3.07] | -4.47 [-7.27, -1.73] | -7.00 [-10.60, -3.40] | -0.007575 |
| π0.5 LIBERO-10 / 500 | UF1 | -3.73 [-6.93, -0.60] | -8.87 [-11.87, -5.87] | -11.40 [-15.41, -7.40] | -0.023268 |
| π0.5 LIBERO-10 / 500 | SW | +1.87 [-0.93, +4.67] | -3.27 [-6.20, -0.40] | -5.80 [-9.40, -2.20] | -0.010138 |
| π0.5 Spatial / 50 | SF1 | -4.27 [-7.13, -1.40] | -11.60 [-14.53, -8.73] | -19.20 [-22.60, -15.80] | -0.012056 |
| π0.5 Spatial / 50 | UF1 | -5.67 [-8.60, -2.80] | -13.00 [-15.93, -10.07] | -20.60 [-24.20, -17.00] | -0.017296 |
| π0.5 Spatial / 50 | SW | -1.27 [-4.13, +1.60] | -8.60 [-11.33, -5.93] | -16.20 [-19.40, -13.20] | -0.014986 |
| π0.5 Spatial / 50 | CU | +10.93 [+7.93, +14.00] | +3.60 [+1.07, +6.07] | -4.00 [-6.40, -1.80] | +0.219768 |
| π0.5 Spatial / 50 | CT | +12.53 [+9.67, +15.47] | +5.20 [+2.73, +7.73] | -2.40 [-4.40, -0.40] | +0.210297 |
| π0.5 Spatial / 500 | SF1 | -0.20 [-1.67, +1.27] | -0.80 [-2.33, +0.67] | -1.20 [-3.00, +0.40] | -0.017729 |
| π0.5 Spatial / 500 | UF1 | -1.80 [-3.53, -0.13] | -2.40 [-4.27, -0.67] | -2.80 [-4.80, -1.00] | -0.022534 |
| π0.5 Spatial / 500 | SW | +0.20 [-1.13, +1.53] | -0.40 [-1.80, +0.93] | -0.80 [-2.40, +0.80] | -0.022419 |
| GR00T LIBERO-10 / 50 | SF1 | -3.07 [-7.00, +0.93] | -14.47 [-18.67, -10.13] | -28.60 [-33.20, -24.00] | -0.008756 |
| GR00T LIBERO-10 / 50 | UF1 | -4.47 [-8.80, -0.33] | -15.87 [-20.00, -11.73] | -30.00 [-34.40, -25.80] | -0.019280 |
| GR00T LIBERO-10 / 50 | CU | +14.53 [+10.00, +19.07] | +3.13 [-1.00, +7.27] | -11.00 [-15.20, -6.80] | +0.226622 |
| GR00T LIBERO-10 / 50 | CT | +22.93 [+18.73, +27.20] | +11.53 [+7.60, +15.47] | -2.60 [-6.60, +1.40] | +0.224986 |
| GR00T LIBERO-10 / 500 | SF1 | +0.60 [-1.93, +3.07] | -3.60 [-6.60, -0.47] | -3.00 [-7.00, +1.00] | -0.006095 |
| GR00T LIBERO-10 / 500 | UF1 | -2.60 [-6.27, +1.00] | -6.80 [-10.07, -3.53] | -6.20 [-10.20, -2.20] | -0.022345 |
| GR00T Spatial / 50 | SF1 | -3.93 [-7.60, -0.33] | -4.80 [-8.53, -1.20] | -11.00 [-14.80, -7.20] | -0.011917 |
| GR00T Spatial / 50 | UF1 | -3.53 [-7.13, +0.00] | -4.40 [-7.93, -0.87] | -10.60 [-14.20, -7.00] | -0.016841 |
| GR00T Spatial / 50 | CU | +6.47 [+3.13, +9.80] | +5.60 [+2.53, +8.67] | -0.60 [-3.60, +2.21] | +0.225960 |
| GR00T Spatial / 50 | CT | +8.27 [+4.93, +11.53] | +7.40 [+4.33, +10.47] | +1.20 [-1.40, +4.00] | +0.231859 |
| GR00T Spatial / 500 | SF1 | -1.00 [-2.80, +0.60] | -0.47 [-2.33, +1.33] | +1.60 [-1.00, +4.20] | -0.011402 |
| GR00T Spatial / 500 | UF1 | -1.60 [-3.60, +0.40] | -1.07 [-3.20, +0.93] | +1.00 [-1.60, +3.60] | -0.021375 |

| Cell | B−A pp [95% CI] | A−pure L10 pp [95% CI] | B−pure L10 pp [95% CI] |
|---|---|---|---|
| π0.5 LIBERO-10 / 50 | +11.33 [+8.40, +14.27] | -19.00 [-22.93, -15.13] | -7.67 [-10.87, -4.40] |
| π0.5 LIBERO-10 / 500 | +5.13 [+2.93, +7.40] | -7.67 [-11.07, -4.40] | -2.53 [-5.60, +0.60] |
| π0.5 Spatial / 50 | +7.33 [+5.60, +9.13] | -14.93 [-17.80, -12.20] | -7.60 [-9.93, -5.33] |
| π0.5 Spatial / 500 | +0.60 [-0.27, +1.53] | -1.00 [-2.47, +0.47] | -0.40 [-1.73, +1.00] |
| GR00T LIBERO-10 / 50 | +11.40 [+7.93, +14.93] | -25.53 [-29.80, -21.20] | -14.13 [-17.87, -10.40] |
| GR00T LIBERO-10 / 500 | +4.20 [+1.20, +7.27] | -3.60 [-7.60, +0.33] | +0.60 [-2.80, +4.00] |
| GR00T Spatial / 50 | +0.87 [-1.40, +3.13] | -7.07 [-10.53, -3.60] | -6.20 [-9.40, -3.00] |
| GR00T Spatial / 500 | -0.53 [-1.87, +0.87] | +2.60 [+0.20, +5.20] | +2.07 [-0.40, +4.73] |

**Paired stage-signal contrasts.** W/L counts candidate-only/reference-only successes; p is exact two-sided McNemar. Bootstrap intervals and p values test different summaries and are reported separately.

| Cell | Contrast | ΔSR pp [95% CI] | W/L | Exact p | Aggregate ΔIR |
|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | SF1−UF1 | +3.60 [-0.20, +7.40] | 59/41 | 0.088626 | +0.007291 |
| π0.5 LIBERO-10 / 50 | CT−CU | -1.00 [-4.40, +2.60] | 40/45 | 0.66465 | +0.007368 |
| π0.5 LIBERO-10 / 500 | SF1−UF1 | +4.40 [+1.40, +7.60] | 44/22 | 0.0092105 | +0.015693 |
| π0.5 Spatial / 50 | SF1−UF1 | +1.40 [-1.40, +4.20] | 30/23 | 0.4101 | +0.005239 |
| π0.5 Spatial / 50 | CT−CU | +1.60 [-0.60, +4.00] | 21/13 | 0.22948 | -0.009471 |
| π0.5 Spatial / 500 | SF1−UF1 | +1.60 [-0.40, +3.60] | 17/9 | 0.16864 | +0.004805 |
| GR00T LIBERO-10 / 50 | SF1−UF1 | +1.40 [-2.40, +5.20] | 51/44 | 0.53839 | +0.010523 |
| GR00T LIBERO-10 / 50 | CT−CU | +8.40 [+4.20, +12.60] | 85/43 | 0.00025853 | -0.001636 |
| GR00T LIBERO-10 / 500 | SF1−UF1 | +3.20 [-0.60, +7.00] | 56/40 | 0.12535 | +0.016250 |
| GR00T Spatial / 50 | SF1−UF1 | -0.40 [-3.40, +2.60] | 29/31 | 0.89742 | +0.004924 |
| GR00T Spatial / 50 | CT−CU | +1.80 [-0.60, +4.20] | 25/16 | 0.21102 | +0.005899 |
| GR00T Spatial / 500 | SF1−UF1 | +0.60 [-1.40, +2.60] | 16/13 | 0.71107 | +0.009973 |

[All 164 paired contrasts](analysis_r7/a1_contrasts.json) include pure L5, R6 call controls, per-replicate binary tests, equal-episode cost intervals and aggregate-cost intervals where both arms have episode costs. McNemar is not applied to fractional A/B replicate means. Historical cost estimates use summary counts, so no paired cost interval is inferred for them; their paired **SR** intervals are available. [Per-cell CSV](analysis_r7/a1_per_cell.csv), [paired-difference CSV](analysis_r7/a1_paired_deltas.csv).

## 4. Placement on the R6 frontier

A point is dominated when another eligible point has no greater owner cost and no lower SR, with at least one strict inequality. The comparator is the frozen R6 single-run set, including older eligible rounds and shared pure references; the A/B means in §3 are not frontier points. No interpolation or replicate pooling is used. “R6: yes” satisfies rule 5's point-estimate definition; “combined: yes” also survives comparison with every R7 point in that cell.

The conservative Q2 lower bound is `CP_lower(W/n; α/2) − CP_upper(L/n; α/2)`, `α=.05`. A bound strictly above **−2 pp** clears the nominal 2 pp NI margin for that comparison. For the π0.5 pure-L5 three-run mean, alpha is split over three binary comparisons and their bounds are averaged. These bounds **do not certify noninferiority** or adjust for selecting points across the frontier. Pure L5 also contains DUAL references from a different harness/seed, retained as in R6.

| Cell | Method | Non-dominated by R6? | On combined frontier? | CP lower vs pure L10 (pp) | CP lower vs pure L5 (pp) |
|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | SF1 | yes | no | -28.23 | -24.67 |
| π0.5 LIBERO-10 / 50 | UF1 | yes | yes | -31.63 | -28.40 |
| π0.5 LIBERO-10 / 50 | SW | yes | yes | -26.08 | -22.71 |
| π0.5 LIBERO-10 / 50 | CU | no | no | -9.20 | -5.53 |
| π0.5 LIBERO-10 / 50 | CT | no | no | -10.35 | -6.41 |
| π0.5 LIBERO-10 / 500 | SF1 | yes | no | -12.28 | -8.70 |
| π0.5 LIBERO-10 / 500 | UF1 | yes | yes | -17.07 | -13.26 |
| π0.5 LIBERO-10 / 500 | SW | yes | yes | -11.03 | -7.27 |
| π0.5 Spatial / 50 | SF1 | yes | no | -23.76 | -24.90 |
| π0.5 Spatial / 50 | UF1 | yes | yes | -25.33 | -26.26 |
| π0.5 Spatial / 50 | SW | yes | yes | -20.57 | -21.60 |
| π0.5 Spatial / 50 | CU | yes | no | -7.20 | -7.94 |
| π0.5 Spatial / 50 | CT | yes | yes | -5.31 | -6.09 |
| π0.5 Spatial / 500 | SF1 | yes | no | -3.84 | -4.48 |
| π0.5 Spatial / 500 | UF1 | yes | yes | -5.67 | -6.57 |
| π0.5 Spatial / 500 | SW | yes | yes | -3.34 | -4.09 |
| GR00T LIBERO-10 / 50 | SF1 | yes | yes | -35.05 | -35.39 |
| GR00T LIBERO-10 / 50 | UF1 | no | no | -36.26 | -36.70 |
| GR00T LIBERO-10 / 50 | CU | no | no | -16.86 | -17.24 |
| GR00T LIBERO-10 / 50 | CT | yes | yes | -8.32 | -8.75 |
| GR00T LIBERO-10 / 500 | SF1 | yes | yes | -8.75 | -9.19 |
| GR00T LIBERO-10 / 500 | UF1 | no | no | -11.79 | -12.32 |
| GR00T Spatial / 50 | SF1 | no | no | -16.20 | -16.30 |
| GR00T Spatial / 50 | UF1 | no | no | -15.77 | -15.82 |
| GR00T Spatial / 50 | CU | no | no | -4.80 | -4.96 |
| GR00T Spatial / 50 | CT | yes | yes | -2.83 | -2.92 |
| GR00T Spatial / 500 | SF1 | yes | yes | -2.36 | -2.52 |
| GR00T Spatial / 500 | UF1 | yes | yes | -2.92 | -3.16 |

| Method | Non-dominated by R6 | Survives combined frontier |
|---|---|---|
| SF1 | 7 | 3 |
| UF1 | 5 | 5 |
| SW | 4 | 4 |
| CU | 1 | 0 |
| CT | 3 | 3 |

**Twenty points extend the historical trade-off; fifteen survive the full R7 comparison.** All four wrist-camera points have higher success and lower owner IR than their corresponding stage-bounded-follow points. On π0.5 Spatial / 50, stage-tilted calls also dominate the new uniform-call point. Three stage-bounded-follow points, five structural-only-follow points, four wrist-camera points and three stage-tilted-call points remain on the combined frontier. None of the 28 new points clears the nominal 2 pp CP margin against either pure reference; the strongest pure-L10 lower bound is −2.36 pp (stage-bounded follow, GR00T Spatial / 500).

[Frontier output](analysis_r7/a1_frontier.json) identifies every R6 dominator, any R6 frontier points dominated by each R7 point, coincident points, the best R6 point at no greater IR, paired CP/McNemar comparisons with that neighbour, and complete before/after frontier membership. All eight original frontiers reproduce [R6 frontier_final](../r06/frontier_final/frontier_data.json) exactly.

## 5. Lowest owner IR reaching pure-L10 success before and after R7

This is an **observed point-estimate crossing**, not equivalence or NI: select the lowest-cost evaluated point with SR at least the cell's pure-L10 SR. The table includes the pure policy as a fallback and reports the non-pure crossing separately. A point below the target is never interpolated upward. R6 prices and its registered historical ledger tolerances are retained.

| Cell | Pure L10 SR | Before R7: SR @ IR | After R7: SR @ IR | IR change | Non-pure crossing before → after |
|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | 90.40% | P1: 90.40% @ 0.500000 | P1: 90.40% @ 0.500000 | +0.000000 | none → none |
| π0.5 LIBERO-10 / 500 | 90.40% | P2: 90.60% @ 0.196533 | P2: 90.60% @ 0.196533 | +0.000000 | 0.196533 → 0.196533 |
| π0.5 Spatial / 50 | 98.60% | P3: 98.60% @ 0.441500 | P3: 98.60% @ 0.441500 | +0.000000 | 0.441500 → 0.441500 |
| π0.5 Spatial / 500 | 98.60% | P4: 98.60% @ 0.118422 | P4: 98.60% @ 0.118422 | +0.000000 | 0.118422 → 0.118422 |
| GR00T LIBERO-10 / 50 | 86.60% | P5: 86.80% @ 0.452979 | P5: 86.80% @ 0.452979 | +0.000000 | 0.452979 → 0.452979 |
| GR00T LIBERO-10 / 500 | 86.60% | P6: 86.60% @ 0.184040 | P6: 86.60% @ 0.184040 | +0.000000 | 0.184040 → 0.184040 |
| GR00T Spatial / 50 | 93.80% | P7: 94.00% @ 0.297629 | P7: 94.00% @ 0.297629 | +0.000000 | 0.297629 → 0.297629 |
| GR00T Spatial / 500 | 93.80% | P8: 94.60% @ 0.051745 | P8: 94.60% @ 0.051745 | +0.000000 | 0.051745 → 0.051745 |

| Point label | Plain explanation | Source identity (`run/arm`) |
|---|---|---|
| P1 | Pure policy, ten-control requests | `r04_cost/r4f_p_l10_inf_k10_L10` |
| P2 | R5 cache with hand-set guard thresholds | `r05_b1/r5b_p_l10_500_hand` |
| P3 | Task-risk policy-call lottery, nominal IR .45 | `r06_frontier/r6q2_pi05_spatial_50_risk_rho0p45` |
| P4 | Guard-triggered committed rescue, third replicate | `r06_paper/r5q1_c10_p_sp_500_rep3` |
| P5 | Task-risk policy-call lottery, nominal IR .45 | `r06_frontier/r6q2_groot_l10_50_risk_rho0p45` |
| P6 | R6 calibrated calls with stall, nominal IR .18 | `r06_c_validation/r6c_groot_l10_500_C18` |
| P7 | R6 calibrated calls with stall, nominal IR .30 | `r06_c_validation/r6c_groot_spatial_50_C30` |
| P8 | No-call cache, fifteen-control commitment confirmation | `r06_frontier/r6q2_groot_spatial_500_A15_confirmation` |

**All eight crossing costs and selected point identities are unchanged.** π0.5 LIBERO-10 / 50 still has no non-pure point reaching pure-L10 success. The new GR00T Spatial / 50 stage-tilted-call point reaches the target at IR .307139, above the existing .297629 crossing. Both new follow variants on GR00T Spatial / 500 reach the target, also above the existing .051745 crossing. The other new points remain below their cells' pure-L10 success targets.

**Reproduction and provenance.** Run these CPU-only commands from the repository root. The analyzer takes a fresh finite inventory; no experiment is launched or polled.

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/analysis_scripts/a1_analyze.py --phase final
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/analysis_scripts/a1_validate.py
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/analysis_scripts/a1_report.py
``` 

The run root is `/home/weiland/trace_runs/os_closed_loop/r07_main/`. All numbers derive from arm summaries, accepted journals, client timing and server decision logs, or the frozen R6 frontier products. [Snapshot](analysis_r7/a1_snapshot.json) records source identities, hashes, conventions and rule evaluations; [references](analysis_r7/a1_references.json) records the 84 reloaded historical arms. No coordinator-ledger number enters a result.

*§§6–8 were written by analysis agent A2 (opus) from the decision logs and saved verbatim by the coordinator (2026-09-30 06:1x CDT). A1 wrote §§1–5.*

## 6. Mechanism by stage

All numbers in §§6–7 come from the arms' `summary.json`, client journals and server decision logs of the 28 evaluation arms (`r07_main`, 500 test pairs each, all complete at 05:5x CDT) and the A/B reference arms. SR differences are point estimates (arm SR minus the mean of A's three replicates on the same 500 pairs); the preregistered intervals and verdicts are in §§2–4. Scripts are `rounds/r07/analysis_scripts/a2_{common,audit,stage,flips,mech}.py`. Outputs are in `analysis_r7/a2/a2_*_r07_main.json`; the profile-run outputs `a2_*_r07_profile_bval1.json` are kept.

**Plain vocabulary.**
- **Look:** the robot encodes its cameras and retrieves the 16 nearest library rows. A looks once per 10 controls: the looked-up chunk plus one blind 5-control block.
- **Extension:** SF/UF serve one more 5-control block along the neighbours' own demonstrations, without looking.
- **Stage of a look:** read from C1's frozen library table using the 16 neighbours.
  - **Macro stage** S0, S1, S2, S3, S4+ counts gripper open/close segments of the matched demonstrations. S0 is the approach before the first grasp; S1 is after it. LIBERO-10 tasks usually reach S3 (second carry); Spatial tasks end in S1.
  - **Class** is one of four:
    - *interior*: all neighbours share the gripper state, and none is next to a gripper change.
    - *event*: all share the gripper state, but some neighbour is within one row of a gripper change.
    - *mixed*: neighbours disagree on the gripper state (a transition).
    - *unknown*: at least one neighbour comes from a failed library rollout. This only exists in the 500-episode libraries.
- **Sparse / dense:** the 50-episode (5 demos per task) and 500-episode libraries.

### 6.1 Look less: where SF1 and UF1 extended, refused, and were stopped by the valve

| Cell | SF1 ΔSR vs A | SF1 IR (A) | Extensions per look, SF1 / UF1 | UF1 ΔSR | Looks refused: missing successor / stage | Valve fires / checks | Early valve looks |
|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | −3.0 pp | .0646 (.0764) | .38 / .67 | −6.6 pp | 28% / 34% | 157 / 11,252 | 85 |
| π0.5 Sp-50 | −4.3 | .0655 (.0775) | .37 / .57 | −5.7 | 40% / 23% | 22 / 4,109 | 0 |
| GR00T L10-50 | −3.1 | .0655 (.0743) | .28 / .71 | −4.5 | 23% / 49% | 64 / 9,061 | 40 |
| GR00T Sp-50 | −3.9 | .0634 (.0753) | .38 / .59 | −3.5 | 39% / 22% | 49 / 4,295 | 25 |
| π0.5 L10-500 | +0.7 | .0689 (.0765) | .23 / .88 | −3.7 | 10% / 67% | 69 / 6,208 | 57 |
| π0.5 Sp-500 | −0.2 | .0604 (.0781) | .58 / .80 | −1.8 | 17% / 25% | 4 / 4,912 | 1 |
| GR00T L10-500 | +0.6 | .0684 (.0745) | .18 / .86 | −2.6 | 9% / 73% | 13 / 4,917 | 11 |
| GR00T Sp-500 | −1.0 | .0641 (.0755) | .37 / .80 | −1.6 | 18% / 46% | 3 / 3,538 | 2 |

**Where SF1 extended, by class.**
- Interior looks were extended 56–82% of the time and event looks 31–58%.
- Mixed and unknown looks were never extended, by design.
- Mixed looks are 28–49% of looks in the sparse cells. Unknown looks are 58% and 66% of looks in dense π0.5 and GR00T L10, and 42% in GR00T Sp-500.
- That is why SF1 extends so little on dense L10 (.18–.23 blocks per look): most looks there have a failed-rollout neighbour.
- By macro stage, extensions spread fairly evenly over S0–S3 on L10 (13–35% each). On Spatial they split roughly half S0 and half S1.

**What the state valve did.**
- It fires almost only while an object is held. On π0.5 L10-50: 116 fires in S3, 33 in S1, 8 in S2, 0 in S0. Spatial fires are in S1 only (except 7 GR00T fires in S0).
- It fires almost only on event-class looks: 2.2–4.7% of event checks in the sparse cells, against at most 0.26% of interior checks.
- Overall it fired on 0.1–1.4% of checks.

**Early valve looks.** These are looks forced one block before A would look — the documented identity exception. There were 221 across 4,000 SF1 episodes. They only add vision, and their cost is about .0004 IR on π0.5 L10-50.

**UF1** extends .57–.88 blocks per look. The share of UF's extensions that SF would have refused is 31–55% in the sparse cells and 27–76% in the dense cells. In sparse cells 54–70% of those refused extensions were in mixed (transition) kernels. In dense cells 77–83% were in unknown kernels (failed-rollout neighbours); π0.5 Sp-500 is the exception at 20%.

### 6.2 Which episodes flipped versus A, and at what stage

Flip counts per 500 pairs. A "loss" is an A-majority success that the arm failed; a "gain" is the reverse. The A noise column is one A replicate against the other two when those agree, for scale.

| Cell | SF1 loss / gain | UF1 loss / gain | A noise (loss / gain) |
|---|---|---|---|
| π0.5 L10-50 | 54 / 33 | 74 / 35 | 16.0 / 10.0 |
| π0.5 Sp-50 | 43 / 17 | 49 / 16 | 9.0 / 4.3 |
| GR00T L10-50 | 64 / 46 | 74 / 49 | 5.3 / 2.7 |
| GR00T Sp-50 | 58 / 39 | 56 / 39 | 0.3 / 1.0 |
| π0.5 L10-500 | 26 / 23 | 51 / 26 | 15.7 / 9.3 |
| π0.5 Sp-500 | 9 / 6 | 18 / 7 | 3.7 / 1.7 |
| GR00T L10-500 | 20 / 24 | 52 / 40 | 2.7 / 3.7 |
| GR00T Sp-500 | 12 / 7 | 17 / 9 | 0 / 0 |

- **GR00T flips are caused by the lever.** GR00T A replicates are almost bit-for-bit repeatable, and in 494–498 of 500 sparse-cell pairs the first decision where SF1/UF1 departs from A replicate 1 is exactly the first extension. So on GR00T every flip is the lever's doing.
  - The lever changes outcomes in both directions. In sparse cells 19–22% of pairs change outcome (GR00T L10-50: 110 of 500; Sp-50: 97), with more losses than gains.
  - In dense cells only 4–9% change (GR00T L10-500: 44; Sp-500: 19), and the two directions roughly balance.
- **π0.5 flips cannot be traced by comparing runs.** π0.5 A replicates already differ bit-for-bit at the first step in most pairs (same robot state, slightly different neighbour weights), so no cross-run divergence point exists there.
- **The stage of the first divergence tells us little.** The first lever event falls in S0 in 90% of SF1 episodes and 100% of UF1, at median decision 2 (the first 10–15 controls). Every episode starts diverging immediately.
- **Losses do not have more extensions than successes.** In the first 24 decisions (120 controls), SF1 sparse losses received as many extension blocks as same-outcome successes, or fewer: 4.5 vs 5.7, 3.8 vs 4.2, 4.5 vs 5.2 and 4.1 vs 4.2. There is no identifiable high-exposure subgroup; the risk is spread across all episodes.
- **Where losses end.** On Spatial, SF1 losses end still holding the object: every π0.5 loss (43/43) ends in S1, and 52 of 58 GR00T losses end in S1. On L10, π0.5 losses mostly reached the last carry stage (33 of 54 reached S3, 15 S4+). GR00T L10 losses end spread over S0–S4+.

### 6.3 Why SF1 loses about 3–4 pp on the 50-episode libraries but holds on the 500-episode libraries

Four measured differences point the same way, and two candidate explanations are ruled out.

1. **Few distinct demonstrations behind each extension.**
   - At looks SF1 extended, the 16 neighbours come from a weight-effective 3.6 / 4.1 / 3.6 / 4.2 demonstrations (median) in the sparse cells. That is all 5 distinct demos of the task, with 2–3 of the 16 neighbours being adjacent time steps of the top demo.
   - In the dense cells the same measure is 8.2 / 10.0 / 8.9 / 10.5 effective demos (13–15 distinct), with at most 1–2 neighbours from the top demo.
   - An extension replays "what these demonstrations did next". In a sparse library that is essentially 3–4 specific recordings, blended across slightly different moments.
2. **Larger drift, a looser valve.**
   - During the extension block, 90th-percentile proprioceptive drift (in library state standard deviations) is .176 / .109 / .111 / .104 in sparse cells versus .058 / .079 / .062 / .070 in dense.
   - The blind block A itself serves drifts less: 90th percentile .08–.09 sparse, .05–.07 dense.
   - The valve radius is calibrated per library by leaving one demo out. It comes out looser where the library is thinner: .40 / .34 / .37 / .32 sparse versus .26 / .27 / .37 / .29 dense.
   - So the valve tolerates the larger sparse drift. Even there, 90th-percentile drift stays under 44% of the radius, and the valve only acts on the far tail. It does not separate good from bad extensions.
3. **Later looks land further from the library.**
   - Mean log retrieval distance at looks rises versus A on the same pairs by +.05 / +.09 / +.05 / +.05 in sparse cells (about 5–9% further from the nearest demonstration). In dense cells it rises only +.01 / +.02 / +.02 / +.02. UF is further still in sparse cells (+.05 to +.12).
   - After an extension, the next look matches the same demonstration it was following in 29–42% of sparse cases, against 18–23% after an ordinary 10-control commitment in the same arm. In dense cells the two rates are equal (18–33% either way).
   - This is consistent with the robot tracking particular recordings rather than the scene. Selection caveat: extensions are granted in smooth segments, and some of this drift is a consequence of failing, since failing episodes run longer.
4. **In sparse libraries the gripper-stage gate does not pick out safe extensions.**
   - SR change per unit of exposure (ΔSR divided by extension blocks per look) is −8 to −12 pp for SF1 and −6 to −10 pp for UF1. SF's gated extensions are no safer per block; SF just takes fewer of them.
   - Pairs lost only by UF carry the same rate of SF-refused extensions as pairs both arms won: .26 vs .27, .08 vs .21, .36 vs .36 and .12 vs .22 per look.
5. **In dense libraries, the part of the gate that matters is refusing kernels with failed-rollout neighbours.**
   - UF1 loses 1.6–3.7 pp in the dense cells where SF1 holds. 77–83% of UF's extra extensions on dense L10 and GR00T Sp-500 are in unknown kernels.
   - Pairs lost only by UF carry more SF-refused extensions per look than pairs both won (.70 vs .52 π0.5 L10-500; .74 vs .57 GR00T L10-500; .50 vs .41 GR00T Sp-500).
   - Caveat: failing episodes in general wander into failed-rollout neighbourhoods (both-fail pairs .71–.75), so this is a marker as well as a possible cause.
   - Exposure alone does not explain the hold. π0.5 Sp-500 extends the most of any SF1 arm (.58 blocks per look) and still holds (−0.2 pp). Library density is what differs.
6. **Ruled out: early valve looks.** There are 221 of them and they only add vision.
7. **Ruled out: missing successor support.** That refusal is more common in sparse cells (23–40% of looks vs 9–18% dense), but it makes SF behave like A, so it cannot cost success.

**In one sentence:** in a 5-demos-per-task library, "follow the neighbours' next block" means following 3–4 specific recordings. Those continuations drift about twice as far as in a dense library, the valve's own calibration loosens accordingly, and each extension carries a small risk that no stage label predicts. That adds up to 3–4 pp over about 0.3–0.4 extension blocks per look.

### 6.4 Look half (SW, π0.5 only): camera modes by stage

| Cell | ΔSR vs A | Owner IR, per-decision (A) | Wrist-only share of looks | Loss / gain vs A |
|---|---|---|---|---|
| L10-50 | −1.0 pp | .0616 (.0764) | 30% | 50 / 39 |
| L10-500 | +1.9 | .0664 (.0765) | 21% | 29 / 32 |
| Sp-50 | −1.3 | .0625 (.0775) | 30% | 36 / 25 |
| Sp-500 | +0.2 | .0557 (.0781) | 45% | 7 / 6 |

- **Look rhythm is unchanged.** SW keeps A's cadence (1.95–1.99 decisions per look). It saves only on the price of each look.
- **Wrist-only looks happen only in interior stages.** They were planned only from interior looks: 67–72% of interior-planned looks used the wrist camera, and 0% of event, mixed or unknown ones did.
- **Why the rest stayed two-camera.** The logged reasons were a gripper change ahead (5,432 / 2,086 / 1,822 / 1,476 per cell), a non-unanimous or failed-rollout kernel (5,012 / 9,008 / 1,879 / 793), and a missing successor (145–788).
- **Where in the task.** On L10-50 the wrist share by stage is S0 53%, S1 32%, S2 33%, S3 11%.
- **What the wrist-only looks retrieved.** Their own retrieval, using only the wrist camera, landed in an interior kernel in 96% (L10-50) and 95% (Sp-50) of cases. In the dense libraries 10% (L10-500) and 2% (Sp-500) pulled in failed-rollout neighbours.
- **Outcome churn.** Flip counts are 2–4 times A's own noise. As with SF, they are balanced in dense cells and tilted toward losses in sparse cells, but smaller than SF's.
- **Completions.** No wrist-only look was followed by a policy call, so no missing-camera completion was ever priced.

### 6.5 Fewer calls: did the stage tilt (CT) actually move calls away from uniform (CU)?

**Yes.** Under CU the call rate was flat at .49–.51 in every stage class. Under CT:

| Stage class | CT call rate | CU call rate |
|---|---|---|
| mixed | .68–.77 | .49–.51 |
| event | .51–.60 | .49–.51 |
| interior | .32–.39 | .49–.51 |

- The mixed (transition) stage's share of all calls rose from 30% to 39% (π0.5 L10), 24% to 36% (π0.5 Sp), 42% to 51% (GR00T L10), and 25% to 35% (GR00T Sp).
- The deviation-entry factor fired 204–853 times per arm.
- **IR stayed matched.** CT−CU is +.0074 / −.0095 / −.0016 / +.0059.
- **SR, CT minus CU:** −1.0 pp (π0.5 L10-50), +1.6 (π0.5 Sp-50), +8.4 (GR00T L10-50), +1.8 (GR00T Sp-50); pooled point estimate +2.7 pp. The verdict is §2 rule 4.
- **The GR00T L10-50 gain is concentrated.**
  - Discordant pairs: 85 won only by CT, 43 won only by CU.
  - 34 of the +42 net successes come from four of the ten tasks (26→43, 35→42, 37→42, 31→36).
  - CT episodes spend fewer looks in transitions: 37.9% of looks mixed vs 41.7% under CU, and 31.3 vs 33.5 looks per episode.
- **Stall calls mark failure; they do not explain it.** Whichever arm failed a discordant pair made 0.22–0.25 stall-triggered calls per look, against 0.03–0.11 in the arm that succeeded, whether that was CU or CT.

## 7. Data quality and cost reconciliation

**Coverage.**
- R7: 28/28 arms, 14,000 accepted episodes, 610,278 server decisions.
- References: 48 A/B journals (24,000 episodes), all on the same two init pools, one hash per suite, identical to the R7 arms.
- Profile: 44 arms, 880 episodes, audited in phase 1 with zero flags.

**Exceptions.**
- End reasons of every accepted R7 episode are success 11,633 or step cap 2,367. Zero `exception` episodes.
- No `exception` row anywhere in the client logs.
- Zero `EXC_PURGED`, `ARM_INCOMPLETE`, `ARM_FAILED` or `COLLECT_FAILED` events in the chain logs.
- Every accepted run has a client timing row, with zero success mismatches.
- The references show the same: 24,000/24,000 covered, zero residual exceptions.

**Count reconciliation (all 28 arms).**
- Decisions, looks and calls agree exactly across four independent counts: my strict latest-run stream, `collect.py`'s own count, the summary ledger, and the client's per-step rows and MISS counts.
- Client inference counts equal server decisions in every episode.
- The only non-counted rows are 40 decisions from an abandoned first attempt of one uid in π0.5 L10-50 CT. The driver retried it and both counts exclude the old attempt correctly.
- Recomputed stage labels match the logged ones on 100% of SF1/UF1 and CT looks.
- Zero commitment violations: no cycle ran longer than A's rule plus the granted extension allows, and no policy chunk was extended.
- **Arms without a manifest.** The 16 SF/UF arms and 4 SW arms ran without a manifest (`state/<arm>.DONE`, no `manifest.json`). Their pairs are exactly the 10×50 test set, on the same init pools as the references; this is now recorded in SELECTION §8.

**Owner IR on each basis.** Pooled = over all decisions (the rule-4 basis). Episode mean = the average of per-episode IR (the CU/CT calibration basis). Ledger = the summary's eager cost-table value, which is not owner IR.

| Arm | Pooled | Episode mean | Ledger |
|---|---|---|---|
| π0.5 L10-50 SF1 / UF1 | .0646 / .0573 | .0634 / .0562 | .0603 / .0535 |
| π0.5 L10-500 SF1 / UF1 | .0689 / .0532 | .0677 / .0532 | .0644 / .0497 |
| π0.5 Sp-50 SF1 / UF1 | .0655 / .0602 | .0644 / .0589 | .0611 / .0563 |
| π0.5 Sp-500 SF1 / UF1 | .0604 / .0556 | .0603 / .0555 | .0564 / .0519 |
| GR00T L10-50 SF1 / UF1 | .0655 / .0550 | .0642 / .0542 | .0564 / .0474 |
| GR00T L10-500 SF1 / UF1 | .0684 / .0521 | .0670 / .0519 | .0589 / .0449 |
| GR00T Sp-50 SF1 / UF1 | .0634 / .0584 | .0625 / .0572 | .0569 / .0525 |
| GR00T Sp-500 SF1 / UF1 | .0641 / .0542 | .0640 / .0539 | .0576 / .0486 |
| π0.5 L10-50 CU / CT | .3007 / .3081 | .3008 / .3072 | .2981 / .3056 |
| π0.5 Sp-50 CU / CT | .2973 / .2878 | .2965 / .2866 | .2946 / .2850 |
| GR00T L10-50 CU / CT | .3009 / .2993 | .3018 / .2982 | .2956 / .2940 |
| GR00T Sp-50 CU / CT | .3012 / .3071 | .3010 / .3064 | .2973 / .3034 |

- Pooled and episode-mean IR differ by at most .0013.
- The ledger is .002–.009 lower because its stage-cost table prices a π0.5 look at .142 and a GR00T look lower still, instead of .152 / .148. Owner IR must always come from the counts.
- CU/CT realized IR is within .0007–.0122 of ρ = .30 on the pooled basis. The largest miss is π0.5 Sp-50 CT at −.0122. CT−CU |ΔIR| per cell is ≤ .0095.

**SW cost.**
- The summed per-decision `owner_cost` equals the recount from camera counters exactly: 0 mismatches, 0 missing.
- Owner IR on this basis: .0616 / .0664 / .0625 / .0557 (L10-50 / L10-500 / Sp-50 / Sp-500).
- Under two-camera pricing (.152 per look) SW costs .0764 / .0765 / .0773 / .0781, the same as A. All of SW's saving comes from the wrist price .055198, which is R4's proportional-latency assumption and has not been measured.
- The ledger prices wrist-only looks at .052 of a full call: .0575 / .0620 / .0584 / .0520.
- 0 completions.

**GR00T blind decisions are not free in wall time.**
- Median server time of a blind GR00T decision was 81–437 ms in SF/UF arms and 1.38–1.49 s in CU/CT arms. The same held for A on the profile run: 459 ms blind, 528 ms looking, of which the vision encoder is only 44 ms.
- On π0.5 a blind decision takes 1.9–3.0 ms.
- The time is spent before the method runs, consistent with requests waiting behind other workers' GPU work (22 concurrent clients per server); the method's own blind step takes 13–16 ms on GR00T.
- IR prices blind decisions at zero, so this is not an IR error. But no wall-clock saving can be claimed for looking less on GR00T until the serving queue is measured or fixed.

**Determinism.**
- π0.5 A replicates differ bit-for-bit in all 500 pairs. π0.5 flips therefore include rollout noise of about 9–26 flips per 500 on L10 (A noise, §6.2).
- GR00T A replicates are identical in 398–469 of 500 pairs per cell.

**Reproduce** (A2 prefix, CPUs 26-29,70-73):
```
python -m exp.offline_search.rounds.r07.analysis_scripts.a2_audit --run eval --refs
python -m exp.offline_search.rounds.r07.analysis_scripts.a2_stage --run eval
python -m exp.offline_search.rounds.r07.analysis_scripts.a2_flips --run eval
python -m exp.offline_search.rounds.r07.analysis_scripts.a2_mech
```
Detail tables are in `/tmp/r7_A2/*.csv.gz`.

## 8. Generality and R8 proposals

### 8.1 Portability scorecard against E5's checklist

Rating legend: ✓ passes; ~ partial; ✗ fails or not shown.

| Check | SF1 (stage-bounded follow) | UF1 (follow without gate) | SW (wrist-only in easy stages) | CU (uniform random calls + stall) | CT (stage-tilted calls + stall) |
|---|---|---|---|---|---|
| Inputs and dimensions from the manifest | ✓ | ✓ | ✗ | ~ | ~ |
| Stages without task names | ~ | n/a | ~ | n/a | ~ |
| Calibration from documented library rules | ✗ in sparse, ✓ in dense | ✓ | ✓ | ✓ | ✓ |
| Time and embodiment | ~ | ~ | ✓ | ~ | ~ |
| Observation/action validity | ✓ (one documented exception) | ✓ | ✓ | ✓ | ✓ |
| Cost completeness | ~ | ~ | ✗ | ✓ | ✓ |
| Cross-policy and cross-scene | ~ | ✗ | ✗ | ~ | ~ |
| No hidden model or sensor | ✓ | ✓ | ✓ | ✓ | ✓ |

Notes on the ratings:
- **Inputs:** SW's encoder path supports only π0.5's two-camera LIBERO layout. GR00T has no one-camera path, and π0.5 on RoboCasa needs three cameras. CU/CT reuse R6's timing, which supports only 5-control requests and 10-control commitments.
- **Stages:** the rule is task-free, but its vocabulary is gripper open/close only. A task with no gripper change (push, turn, wipe, drive) becomes one long stage.
- **Calibration:** SF's valve radius and support rules come from the library, but on 5 demos per task the valve loosens exactly where continuations are worst (§6.3). CU/CT solve their budget on 10 non-test recordings per cell and landed within ≤.012 of ρ.
- **Time:** extensions and stall windows are counted in control blocks, not seconds. The valve assumes a fixed-base arm's state.
- **Validity:** SF requires true successors for all 16 neighbours, never clamps at a terminal row, and never extends a policy chunk. Its one exception is 221 early looks that only add vision.
- **Cost:** SW's saving rests entirely on an unmeasured price. For every lever, the GR00T blind wall time (§7) is invisible to IR.
- **Cross-policy:** SF, CU and CT ran the same rule on both policies, and SF's sparse-vs-dense pattern repeats on both. There is no RoboCasa or real-arm test. SW is π0.5 only. UF fails on success rate in all eight cells.
- **Hidden inputs:** all methods use only proprioception, executed actions, the library and the policy's own encoder.

### 8.2 Ranked R8 proposals

1. **Confirm stage-tilted calls; do not adopt them yet.**
   - What R7 showed: CT moved calls from smooth motion (interior call rate .32–.39) to gripper transitions (.68–.77) at matched cost (|ΔIR| ≤ .0095). The pooled point estimate is +2.7 pp over CU, with +8.4 pp on GR00T L10-50 concentrated in four tasks.
   - Why that is not enough: one run per arm, and one cell drives most of the effect.
   - R8 test, fixed 16 arms: a second replicate of CT and CU on the four sparse cells (8 arms), plus CT and CU on the four dense cells (8 arms). Optionally add a fixed pair per sparse cell that splits CT's two factors (transition-only vs deviation-only).
   - Keep: calls placed by this rule if both the replicate and the dense cells agree. Drop the tilt if the pooled replicate interval includes zero.
2. **Replace SF's gripper-stage gate with a library-quality gate, and use it only where the library supports it.**
   - What R7 showed: SF1 saves .006–.018 IR at −1.0 to +0.7 pp on 500-episode libraries, but loses 3–4 pp on 50-episode ones. The protective part of its gate in dense libraries was refusing neighbourhoods that contain failed rollouts. The gripper-stage part added nothing measurable in sparse libraries.
   - Proposal: extend only when (a) all 16 neighbours come from successful demonstrations, and (b) they span enough distinct demonstrations. The threshold must come from a written library rule, for example the median effective-demo count in leave-one-out library replay. It must not be tuned on test outcomes.
   - Expected effect: sparse kernels (3.4–4.2 effective demos) would switch extension off, and dense kernels (8–10.5) would keep it.
   - This gate was designed after seeing R7's test outcomes. It must first be profiled on the non-test B-val starts, then evaluated on a fixed arm set: 8 cells × {gate (a) only, gates (a)+(b)}.
3. **Measure the wrist-only price before running any more SW arms.**
   - SW's .010–.022 IR saving at −1.3 to +1.9 pp is 100% the assumed .055198 price.
   - Measure: GPU latency of a wrist-only encode vs a two-camera encode, and of the completion path, on the actual serving stack at batch size 1.
   - If the measured price keeps at least half of the saving, SW stays as a π0.5-only lever. Otherwise drop it.
4. **Measure and fix wall time on the GR00T serving stack.**
   - Blind requests wait 0.08–1.5 s at the median behind other workers' GPU work.
   - Record per-control wall time and serialization before claiming any latency benefit from looking less or calling less on GR00T.
   - This is a systems item, not a method.
5. **Drop:**
   - UF, which was only a control and loses 1.6–6.6 pp.
   - SF on sparse libraries.
   - The state valve as a safety argument: it acts on 0.1–1.4% of checks, typical drift is at most 44% of its radius, and it is loosest where continuations are worst. Keep it only as telemetry, or recalibrate it with a density-aware rule inside proposal 2.
   - SF with a cap of 2 blocks, already dropped at Freeze 2.

### 8.3 What stays shelved, and why

- **Few-step denoising (cheaper policy calls):** untested in R7; nothing here argues for reopening it before proposals 1 and 4 are settled.
- **Library growth:** R7 adds one argument for it — looking less is only safe on dense libraries. But growth changes the library and costs paid episodes, so it stays shelved unless the owner reopens it.
- **Task-level knob:** no R7 evidence. CT's gain is a within-task stage effect, and a per-task knob would need per-task tuning, which conflicts with the no-benchmark-tuning principle.
- **SF+SW (composed):** neither half is settled — SF fails on sparse libraries and SW's price is unmeasured — and the composed arm was never profiled.
- **SA (full stage allocator):** it composes calls, looks and cameras and needs its own budget solve. Build it only after proposals 1–3 resolve which components deserve a place.

## Addendum A (coordinator, 2026-09-30 06:2x CDT) — measured wrist price

Rule 3's saving used R4's *assumed* wrist price. `analysis_scripts/a3_wrist_latency.py` now measures it on the actual R7
per-request path (π0.5, local RTX 4090, batch 1, eager, 12 recorded observations × 30 timed repetitions; output
`analysis_r7/a3_wrist_latency.json`). Median latencies: stock two-camera stage 1 **62.1 ms**, per-request full stage 1
**67.7 ms** (+9 % dispatch overhead), wrist-only **26.4 ms**, missing-camera completion **20.5 ms**, stage 2+3 (K10)
**375 ms**. The measured vision share of a full inference is .153, matching the owner basis .152. Proportionally
priced against the stock look (.152): per-request full look **.1656**, wrist look **.0646** (assumed .0552),
completion **.0502** (assumed .0499).

| π0.5 cell | SW IR, assumed prices | SW IR, measured (as implemented) | SW IR, measured with stock-speed full looks | A |
|---|---|---|---|---|
| LIBERO-10 / 50 | .0616 | .0678 | .0630 | .0764 |
| LIBERO-10 / 500 | .0664 | .0728 | .0674 | .0765 |
| Spatial / 50 | .0625 | .0688 | .0640 | .0775 |
| Spatial / 500 | .0557 | .0617 | .0579 | .0781 |

Rule 3's IR clause (lower than A in 4/4 cells) **still holds on measured prices**, with the saving shrinking from
.010–.022 to .004–.016 as implemented. The 9 % slower full look is an implementation artefact of the per-request path
(it re-encodes through the batcher instead of the stock stage 1); routing full looks through the stock path would recover
.005–.006 of IR. Prices remain proportional-latency estimates on one GPU, not a hardware-independent cost.

## Addendum B (coordinator, 2026-09-30 07:3x CDT) — rule-4 replicate on the sparse libraries (SELECTION §9.2)

CU(.30) and CT(.30) were rerun on the four sparse cells with a new lottery seed (26092904; calibration unchanged).
Script `analysis_scripts/a4_completion.py` → `analysis_r7/a4_completion.json` (task-stratified init bootstrap,
10,000 draws, seed 20260930 + cell index; the two replicates are averaged per (task, init) before resampling).

| Cell | CT rep1 / rep2 | CU rep1 / rep2 | CT−CU, two replicates [95% CI] | CT−CU, rep2 only | ΔIR rep1 / rep2 |
|---|---|---|---|---|---|
| π0.5 LIBERO-10 / 50 | .854 / .854 | .864 / .832 | +0.60 [−2.10, +3.40] | +2.20 | +.0074 / +.0074 |
| π0.5 Spatial / 50 | .962 / .962 | .946 / .948 | +1.50 [−0.10, +3.10] | +1.40 | −.0095 / **−.0153** |
| GR00T LIBERO-10 / 50 | .840 / .824 | .756 / .812 | +4.80 [+1.90, +7.80] | +1.20 | −.0016 / +.0000 |
| GR00T Spatial / 50 | .950 / .912 | .932 / .926 | +0.20 [−1.70, +2.10] | −1.40 | +.0059 / +.0077 |
| **Pooled** | | | **+1.78 [+0.60, +2.93]** | **+0.85 [−0.80, +2.55]** | |

**Preregistered verdict (§9.2): not supported** — the pooled two-replicate lower bound is above zero, but one replicate
pair (π0.5 Spatial / 50, rep2) misses the |ΔIR| ≤ .015 match by .0003 (CT spent *less*). The second replicate alone does
not reproduce a positive bound. **Reading:** tilting calls toward gripper transitions and first state-deviation entries is
worth roughly +1 to +2 pp at equal cost, not the +2.7 pp of the first run. The first run's GR00T LIBERO-10 / 50 gain
(+8.4 pp) was largely lottery-seed variance: CU alone moved 5.6 pp (.756 → .812) between seeds with identical
calibration, so single-seed call arms on long-horizon sparse cells carry about ±3 pp of seed noise on top of the
init-bootstrap interval.

## Addendum C (coordinator, 2026-09-30 08:2x CDT) — rule 4 on the dense (500-episode) libraries (SELECTION §9.3)

CU and CT at ρ = .18, calibrated by the C3 recipe on the non-test B-val recordings (feasible in all four cells), profiled
on 20 non-test episodes per arm (realized IR .170–.196, all within .18 ± .03), then 500 test pairs each
(`analysis_r7/a4_completion.json`).

| Cell | CT SR @ IR | CU SR @ IR | CT−CU pp [95% CI] | ΔIR |
|---|---|---|---|---|
| π0.5 LIBERO-10 / 500 | .898 @ .1863 | .902 @ .1885 | −0.40 [−3.20, +2.40] | −.0021 |
| π0.5 Spatial / 500 | .966 @ .1648 | .984 @ .1631 | −1.80 [−3.60, −0.20] | +.0017 |
| GR00T LIBERO-10 / 500 | .870 @ .1884 | .870 @ .1867 | +0.00 [−3.20, +3.20] | +.0017 |
| GR00T Spatial / 500 | .934 @ .1825 | .934 @ .1838 | +0.00 [−2.40, +2.40] | −.0013 |
| **Pooled** | | | **−0.55 [−1.85, +0.75]** | all matched |

**Preregistered verdict (§9.3): not supported.** On dense libraries the stage tilt does not change success at equal
cost. For reference against pure L10 (.904 / .986 / .866 / .938): GR00T LIBERO-10 / 500 CU and CT reach .870 at IR
≈ .187–.188, above the existing .184 crossing; the other three cells stay below pure L10 SR.

## Completion summary (coordinator)

- **Look less (one extra block in stable-gripper stages):** saves .006–.018 IR everywhere; holds success on 500-episode
  libraries, loses 3–4 pp on 50-episode libraries (rule 1 not supported). The stage/state gate beats ungated
  extension by +2.0 pp (rule 2 supported); in dense libraries the protective part is refusing neighbourhoods that contain
  failed demonstrations.
- **Look half (wrist camera in easy stages, π0.5):** success unchanged (rule 3 supported); on *measured* latency the
  saving is .004–.016 IR as implemented, .009–.020 if full looks used the stock path (Addendum A).
- **Fewer calls placed by stage:** first run +2.7 pp at equal cost (rule 4 supported), but the preregistered replicate
  does not confirm it (Addendum B: two-replicate +1.78 [+0.60, +2.93] with one cost-match miss of .0003; replicate alone
  +0.85 [−0.80, +2.55]), and on dense libraries it is flat (−0.55 [−1.85, +0.75]). Net: at most a small (+1–2 pp)
  effect on sparse libraries; single-seed call arms on long sparse tasks carry ~±3 pp of seed noise.
- **Frontier:** no cell's lowest IR reaching pure-L10 SR improved in R7.
