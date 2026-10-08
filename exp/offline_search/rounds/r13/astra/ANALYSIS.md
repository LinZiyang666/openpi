# R13 final analysis

The paper conclusion is: **with budget (cost) feedback, distance-threshold calls converge to almost the same cadence as periodic calls; distance mainly reorders calls within the budget-set cadence, with no detectable pooled matched-budget SR advantage. The simpler periodic schedule suffices in these measured settings.** B supplies the developmental evidence; the separate test-A measurement status is recorded below.

D5 (2026-10-04 20:38 CDT) ends the large grid and supersedes earlier scheduling/release language in PLAN, GATES and MECHANISM. This report is measurement and synthesis, not a design round. D=distance_feedback, P=periodic, S=distance_static_b without feedback. All preceding-test evidence below is B only.

**Q1: budget delivery and calibration**

The audited file contains 64 completed arms: 28 D, 28 P and 8 S. **The request's “32 primary B D arms” is a count mismatch: the actual completed D denominator is 28, all passing.** There are 32 periodic cap configurations in the full frozen grid, not 32 completed D arms. Target delivery uses `|IR−target|≤.02`, and paired matching separately uses `|IR_D−IR_P|≤.01`.

| Method | Passed / arms | IR error range |
| --- | --- | --- |
| distance_feedback | 28/28 | [-0.0153, +0.0061] |
| distance_static_b | 3/8 | [-0.0217, +0.0732] |
| periodic | 28/28 | [-0.0121, +0.0089] |

IR is pooled owner cost / pooled decision slots: GR00T `(.148 V+.852 M)/N`, π0.5 `(.152 V+.848 M)/N`; V counts real looks and M all policy calls including mandatory guards. It is not mean episode IR or wall-clock latency. S misses 5/8 targets, both under and over; B recalibration alone does not stabilize closed-loop spending.

| Cell | Target | Static IR | Error | Budget |
| --- | --- | --- | --- | --- |
| GR00T L10-50 | 0.32 | 0.3133 | -0.0067 | pass |
| GR00T L10-50 | 0.40 | 0.3783 | -0.0217 | FAIL |
| GR00T Spatial-50 | 0.32 | 0.3508 | +0.0308 | FAIL |
| GR00T Spatial-50 | 0.40 | 0.4302 | +0.0302 | FAIL |
| π0.5 L10-50 | 0.32 | 0.3311 | +0.0111 | pass |
| π0.5 L10-50 | 0.40 | 0.4147 | +0.0147 | pass |
| π0.5 Spatial-50 | 0.32 | 0.3932 | +0.0732 | FAIL |
| π0.5 Spatial-50 | 0.40 | 0.4522 | +0.0522 | FAIL |

π0.5 P slightly underspends in L10-50 (signed IR errors -0.0121 to -0.0044); Spatial is near target or slightly over. The direction is not uniform across all π0.5 cells. Exact errors and host records are in `final_b_budget.csv`. The mandatory-guard rate can shift a static cap's realized cost; D corrects the ledger online.

D needs a distance scale (two scalars per cell), a bias per cell/dose, a selected shared feedback width, and the running cost ledger. In the full frozen design this is 49 scalars. Its calibration used B trajectories where available and library-held-out transfer for GR00T L10-200 and size-500 (which lacks B pilot validation). P needs one cap per cell/dose, 32 across the full frozen design, from the existing whole-episode-held-out library ScheduleReplay solver; it needs no distance signal or online budget ledger. S uses eight B-calibrated thresholds on the same calibration paths as D, without feedback. These are configuration counts, not completed-arm counts; P is cheaper in calibration/state requirements, with no new latency benchmark claimed.

Pilot hosts: all GR00T and π0.5 L10-50 ran on local 4090; π0.5 L10-200 and Spatial-50 ran on H100. D2 authorizes the local π0.5 pilot; the obsolete GATES host hold is not a current verdict. All evaluated primary arms pass target delivery and D engagement; mechanics remain as audited in gates.json. The distinct paired cost misses are retained under Q2.

**Headline: cadence convergence, with local placement differences**

Budget feedback makes distance-threshold calls converge to approximately the periodic call cadence; distance still advances or delays calls within that cadence. At .32 rep0, pooled complete call gaps are 1.67 [1/2/2] (D) and 1.72 [1/2/2] (P). The full distributions differ, so “converge” describes their operating cadence, not equality of distributions or exact call times. Static S has no budget feedback and is not a matched-cost comparator when its budget gate fails.

The predecision threshold is `bias + (cv*V + cm*M − target*N)/(cm*width)`. Waiting replenishes budget and lowers the threshold even at constant distance. Identity-held-out probes give:

| Predictor | AUC | Accuracy |
| --- | --- | --- |
| gap | 0.832 | 76.7% |
| gap_distance_residual | 0.848 | 77.2% |
| gap_ledger | 0.937 | 85.1% |
| gap_ledger_distance_residual | 0.994 | 96.5% |

The folds group cell/task/init across doses and repeats. Gap+ledger explains much of the schedule; distance adds placement information conditional on that ledger. The last predictor nearly reconstructs the rule, not its benefit. Within-episode gap categories explain only 0.32–2.45% of raw-distance variation; median within-episode Spearman correlations across D cells range from -0.169 to -0.073. Thus distance is not a simple increasing clock since the last call. This is descriptive: triggering truncates long gaps, and neither the weak trend nor its sign is causal.

On the same recorded .32 rep0 states (both behavior paths, six cells), 9810/14727 = 66.6% of the union of non-guard selected positions are selected by only one rule. Each replay keeps its own ledger/call history, with recorded states/guards fixed. This approximately two-thirds difference is local placement, not different task stages or counterfactual SR. D also makes 17.8% of post-startup knob calls sooner than P's minimum gap and leaves 472 eligible cache looks at/beyond P's maximum.

Shadow action error at D-only versus P-only picks, conditional on cell × recorded behavior × task × actual gap, is 0.224 versus 0.201: Δ=0.0228 [0.0078, 0.0392], or 11.3% higher at D-only picks. There are 4040 common-support looks out of 5970 disagreement looks. The label is frozen-sigma-normalized motion MSE over the first ten controls, not action truth or causal call value; uncertainty uses 1000 physical-init cluster resamples. Spatial cells carry the positive signal; GR00T L10 reverses direction.

| Cell | D-only error | P-only error | D−P [95% CI] |
| --- | --- | --- | --- |
| GR00T L10-50 | 0.258 | 0.276 | -0.0184 [-0.0559, +0.0215] |
| GR00T Spatial-50 | 0.242 | 0.197 | +0.0444 [+0.0157, +0.0802] |
| π0.5 L10-50 | 0.174 | 0.162 | +0.0118 [-0.0087, +0.0309] |
| π0.5 Spatial-50 | 0.249 | 0.184 | +0.0648 [+0.0285, +0.1016] |

Complete between-call gaps below use real observations: mean [p10/median/p90], call-weighted, startup excluded; terminal incomplete intervals are right censored and excluded. All calls include guards. On these B traces the look spacing is ten controls; the CSV separately supplies directly recorded control gaps. This table preserves every observed cell/dose/rep, including the no-feedback diagnostic where run; a dash is unmeasured, never imputed.

| Cell | Target | Rep | Feedback gap | Periodic gap | Static gap |
| --- | --- | --- | --- | --- | --- |
| GR00T L10-200 | 0.25 | 0 | 2.29 [1/2/4] | 2.33 [1/2/4] | — |
| GR00T L10-200 | 0.32 | 0 | 1.68 [1/2/2] | 1.75 [1/2/2] | — |
| GR00T L10-200 | 0.40 | 0 | 1.35 [1/1/2] | 1.33 [1/1/2] | — |
| GR00T L10-200 | 0.45 | 0 | 1.16 [1/1/2] | 1.14 [1/1/2] | — |
| GR00T L10-50 | 0.25 | 0 | 2.09 [1/2/4] | 2.18 [1/2/4] | — |
| GR00T L10-50 | 0.32 | 0 | 1.61 [1/2/2] | 1.65 [1/2/2] | 1.73 [1/1/4] |
| GR00T L10-50 | 0.32 | 1 | 1.62 [1/2/2] | 1.66 [1/2/2] | — |
| GR00T L10-50 | 0.40 | 0 | 1.31 [1/1/2] | 1.30 [1/1/2] | 1.39 [1/1/2] |
| GR00T L10-50 | 0.45 | 0 | 1.14 [1/1/2] | 1.14 [1/1/2] | — |
| GR00T Spatial-50 | 0.25 | 0 | 2.00 [1/2/3] | 2.19 [1/2/3] | — |
| GR00T Spatial-50 | 0.32 | 0 | 1.77 [1/2/3] | 1.65 [1/2/2] | 1.44 [1/1/2] |
| GR00T Spatial-50 | 0.32 | 1 | 1.78 [1/2/3] | 1.69 [1/2/2] | — |
| GR00T Spatial-50 | 0.40 | 0 | 1.33 [1/1/2] | 1.30 [1/1/2] | 1.20 [1/1/1] |
| GR00T Spatial-50 | 0.45 | 0 | 1.14 [1/1/2] | 1.14 [1/1/2] | — |
| π0.5 L10-200 | 0.25 | 0 | 2.16 [1/2/3] | 2.35 [1/3/3] | — |
| π0.5 L10-200 | 0.32 | 0 | 1.66 [1/2/2] | 1.74 [1/2/2] | — |
| π0.5 L10-200 | 0.40 | 0 | 1.32 [1/1/2] | 1.33 [1/1/2] | — |
| π0.5 L10-200 | 0.45 | 0 | 1.14 [1/1/2] | 1.14 [1/1/2] | — |
| π0.5 L10-50 | 0.25 | 0 | 2.14 [1/2/3] | 2.48 [1/3/4] | — |
| π0.5 L10-50 | 0.32 | 0 | 1.63 [1/2/2] | 1.79 [1/2/2] | 1.59 [1/1/3] |
| π0.5 L10-50 | 0.32 | 1 | 1.63 [1/2/2] | 1.77 [1/2/2] | — |
| π0.5 L10-50 | 0.40 | 0 | 1.33 [1/1/2] | 1.35 [1/1/2] | 1.25 [1/1/1] |
| π0.5 L10-50 | 0.45 | 0 | 1.14 [1/1/2] | 1.15 [1/1/2] | — |
| π0.5 Spatial-50 | 0.25 | 0 | 2.12 [1/2/3] | 2.30 [1/2/3] | — |
| π0.5 Spatial-50 | 0.32 | 0 | 1.83 [1/2/3] | 1.69 [1/2/2] | 1.31 [1/1/1] |
| π0.5 Spatial-50 | 0.32 | 1 | 1.77 [1/2/3] | 1.70 [1/2/2] | — |
| π0.5 Spatial-50 | 0.40 | 0 | 1.37 [1/1/2] | 1.30 [1/1/2] | 1.13 [1/1/1] |
| π0.5 Spatial-50 | 0.45 | 0 | 1.15 [1/1/2] | 1.14 [1/1/2] | — |

Paper table (the fixed size-50/.32 rep0 subset):

| Cell (.32, rep0) | IR D / P / S | D gap | P gap | S gap |
| --- | --- | --- | --- | --- |
| GR00T L10-50 | 0.321 / 0.329 / 0.313 | 1.61 [1/2/2] | 1.65 [1/2/2] | 1.73 [1/1/4] |
| GR00T Spatial-50 | 0.325 / 0.327 / 0.351 | 1.77 [1/2/3] | 1.65 [1/2/2] | 1.44 [1/1/2] |
| π0.5 L10-50 | 0.322 / 0.309 / 0.331 | 1.63 [1/2/2] | 1.79 [1/2/2] | 1.59 [1/1/3] |
| π0.5 Spatial-50 | 0.314 / 0.322 / 0.393 | 1.83 [1/2/3] | 1.69 [1/2/2] | 1.31 [1/1/1] |

Figure source: `final_b_gap_cdf.csv`, exact ECDFs at every integer gap with denominator and count, faceted by cell/target/rep/method; select kind=`any`, unit=`real_looks` for the table's distribution. `knob_since_any` and `knob_since_knob` expose the guard contribution. `final_b_gap_summary.csv` includes SD and maxima. S comparisons at failed budgets diagnose loss of cost control; they do not isolate a causal cadence effect of feedback.

**Q2: paired B success at comparable budget**

Across all 2,800 paired B episode-runs, D SR=88.4%, P SR=87.5%; pooled D−P=+0.86 [-0.61, +2.29] percentage points. This is the complete nominal-dose descriptive pool, including the two unmatched cost points. There are 200 distinct suite/task/init clusters, not 2,800 independent initial states.

Restricting by the frozen, outcome-blind `|ΔIR|≤.01` budget label gives 2,600 pairs and D−P=+0.85 [-0.73, +2.35] pp. Unmatched points remain visible: GR00T L10-200 target 0.25 rep0: ΔIR=-0.0136; π0.5 L10-50 target 0.32 rep0: ΔIR=+0.0125. Both intervals include zero; the data do not distinguish SR at approximately matched budgets. This is not a formal equivalence/noninferiority result, and exact equal-IR performance is not identified by a nominal-dose comparison.

| B cell | Pairs (matched) | D SR | P SR | D−P pp [95% cluster CI] |
| --- | --- | --- | --- | --- |
| GR00T L10-200 | 400 (300) | 90.5% | 89.8% | +0.75 [-2.50, +3.75] |
| GR00T L10-50 | 500 (500) | 83.4% | 78.2% | +5.20 [+1.40, +9.00] |
| GR00T Spatial-50 | 500 (500) | 86.6% | 85.8% | +0.80 [-2.40, +3.80] |
| π0.5 L10-200 | 400 (400) | 90.2% | 91.2% | -1.00 [-4.50, +2.50] |
| π0.5 L10-50 | 500 (400) | 86.8% | 88.6% | -1.80 [-5.00, +1.20] |
| π0.5 Spatial-50 | 500 (500) | 93.6% | 92.8% | +0.80 [-0.40, +2.00] |

Intervals: 5,000 paired original suite/task/init cluster resamples within fixed suite/task, seed 20261004; carry all models, sizes, doses and repeats together, recomputing the equally weighted episode-pair mean. Per-cell results are descriptive, with no subgroup selection. No ordinary binomial/McNemar test is used on repeated init observations. Rep0-only pooled sensitivity: +0.92 [-0.50, +2.33] pp. The shared size-50/.32 subset gives rep0 +0.75 [-3.00, +4.25] and rep1 +0.50 [-3.25, +4.25] pp. These intervals are conditional on the recorded blocks; two repeats cannot establish future-block variance. The larger frozen A interpolation/precision plan was superseded by D5; no four- or two-block power claim is carried into the single-block lean confirmation.

**Test-A: fixed measurement only**

<!-- TEST_A_MEASUREMENT_BEGIN -->

**D5 lean confirmation: COMPLETE (12/12 arms validated at 2026-10-05T08:35:05.467449+00:00).** Fixed by rule before these outcomes: four size-50 cells, target .32, rep0, distance feedback / periodic / off. No tuning, selection, expansion, or stopping rule is inferred from A.

Scheduled by the owner: GR00T on the local 4090 during the night of October 4; π0.5 on H100 after approximately 07:30–08:00 CDT October 5. This schedule is not a claim of observed execution. Missing local summaries/logs remain pending; completed-arm host/timing evidence is in `final_test_a_numbers.json`.

| Fixed cell | Rule | Status | IR | SR | Gap mean [p10/50/90] |
| --- | --- | --- | --- | --- | --- |
| GR00T L10-50 | distance_feedback | complete | 0.3194 | 82.2% | 1.62 [1/2/2] |
| GR00T L10-50 | periodic | complete | 0.3230 | 81.6% | 1.69 [1/2/2] |
| GR00T L10-50 | off | complete | 0.2183 | 72.2% | 2.25 [1/2/5] |
| GR00T Spatial-50 | distance_feedback | complete | 0.3194 | 92.8% | 1.82 [1/2/3] |
| GR00T Spatial-50 | periodic | complete | 0.3197 | 92.0% | 1.70 [1/2/2] |
| GR00T Spatial-50 | off | complete | 0.1233 | 92.4% | 2.16 [1/1/5] |
| π0.5 L10-50 | distance_feedback | complete | 0.3210 | 87.8% | 1.63 [1/2/2] |
| π0.5 L10-50 | periodic | complete | 0.3092 | 88.2% | 1.79 [1/2/2] |
| π0.5 L10-50 | off | complete | 0.1725 | 81.8% | 2.64 [1/1/6] |
| π0.5 Spatial-50 | distance_feedback | complete | 0.3114 | 98.6% | 1.84 [1/2/3] |
| π0.5 Spatial-50 | periodic | complete | 0.3121 | 98.6% | 1.78 [1/2/2] |
| π0.5 Spatial-50 | off | complete | 0.1304 | 93.4% | 1.78 [1/1/4] |

Pooled fixed-set paired D−P: +0.25 [-1.35, +1.80] pp (2,000 episode pairs; 1000 suite/task/init clusters). Nominal-dose contrast; a matched-budget interpretation requires every cell to satisfy |ΔIR|≤.01. Failed cells remain in the fixed pool and are flagged.

| Cell | Pairs | D−P IR | Matched | D−P SR pp [95% CI] |
| --- | --- | --- | --- | --- |
| GR00T L10-50 | 500 | -0.0036 | yes | +0.60 [-3.40, +4.60] |
| GR00T Spatial-50 | 500 | -0.0003 | yes | +0.80 [-2.20, +3.80] |
| π0.5 L10-50 | 500 | +0.0118 | NO | -0.40 [-4.00, +3.00] |
| π0.5 Spatial-50 | 500 | -0.0008 | yes | +0.00 [-1.40, +1.40] |

IR uses the frozen owner cost coefficients and pooled accepted N/V/M, not current latency-derived summary IR. Gaps are between successive actual calls in real looks (guards included), with startup and terminal censoring separate; the figure CSV also has elapsed control gaps. SR pairs use the same suite/task/init; 5,000 paired cluster resamples within suite/task, seed 20261004, jointly carrying both models. The fixed-set pool weights cells equally because every complete arm uses the same full init count. A single rep0 cannot supply leave-one-block sensitivity or the superseded multi-block precision claim. Off is contextual and never used to choose a method.

**Pre-D5 supplementary measurements — all 7 arms, descriptive only.** Their accepted episodes finished before the stop; nothing is selected, promoted into the lean set, or used for tuning. The previously unspecified seventh arm is GR00T L10-500 periodic .32 rep0. The aborted GR00T L10-50 periodic .25 arm is discarded and is never read by this script.

| Prespecified supplementary arm | Episodes | IR | SR |
| --- | --- | --- | --- |
| r13_groot_l10_pure_c5_rep0 | 500 | 1.0000 | 85.4% |
| r13_groot_l10_200_distance_feedback_ir40_rep0 | 500 | 0.3969 | 85.4% |
| r13_groot_l10_200_periodic_ir40_rep0 | 500 | 0.3955 | 89.2% |
| r13_groot_l10_500_distance_feedback_ir25_rep0 | 500 | 0.2410 | 89.0% |
| r13_groot_l10_500_periodic_ir25_rep0 | 500 | 0.2420 | 86.6% |
| r13_groot_l10_500_distance_feedback_ir32_rep0 | 500 | 0.3132 | 87.6% |
| r13_groot_l10_500_periodic_ir32_rep0 | 500 | 0.3152 | 86.6% |

Re-run `final_test_a.py` with the CPU command in `final/HANDBACK.md` after local summaries, journals, manifests and decision logs are available. It replaces only this marked section in ANALYSIS/REPORT, writes `final_test_a_numbers.json`, arm/pair tables and gap CDF CSVs, and refuses frozen-file writes. The B conclusion and all controller artifacts are unchanged.

<!-- TEST_A_MEASUREMENT_END -->

**Reproducibility and limits**

`final_b_analysis.py` derives all B tables and numbers from `gates.json`, `mechanism_numbers.json` and the B-only mechanism cache; it reconciles per-arm looks/calls/episodes/N to the gate ledger. Existing mechanism probes and shadows are reused as audited, without refitting. `final_b_numbers.json` records inputs, code hashes, cluster intervals and the frozen-file check. `final/HANDBACK.md` gives exact CPU commands. B pilot outcomes support a developmental comparison; test-A supplies a separate measurement confirmation as its fixed arms complete. No A result enters calibration, dose/cell choice, or the B estimates.
