# Q1 final: reconstruction predicts disagreement; success and call value remain separate

> **Coordinator erratum 2026-09-29 (09:0x–09:5x CDT), added after this report was written.** 316 closed-loop episodes in 33
> R6 arms (7 paper, 11 ablation, 15 frontier) had ended on a client-side exception (keepalive timeout / render error
> under timan107 overload) and were journaled as ordinary failures. They were purged and re-run
> (`closed_loop/ops/remote/purge_exc.py`, ledger §10). Several numbers quoted below are therefore stale:
> - **A / B** — see the regenerated `../PAPER_AB.md`, e.g. π0.5 l10-50 B .827 (+11.3 pp), GR00T l10-50 B .725 (+11.4 pp).
> - **Ablations** — see `../ABLATIONS.md`. The direct-token-PCA losses on the Spatial-500 cells (−3.0 / −3.6 pp) were
>   artefacts; direct and pooled PCA now differ only on GR00T Spatial-50.
> - **Frontier** — see `../frontier_repaired/`. For example:
>   - GR00T l10-50 ρ .45 is .868 (was .838);
>   - π0.5 l10-500 ρ .24 is .910 and B+.25 is .916 (were .884 / .882);
>   - π0.5 Spatial-50 ρ .35 is .952 (was .934);
>   - GR00T Spatial-50 ρ .12 is .876 (was .854);
>   - π0.5 Spatial-500 A+1/32 is .976 (was .946).
>
> - Two further post-hoc statements are now wrong: π0.5 L10-500 B+.125 is .892 (was .842, so there is no "dip"), and
>   GR00T L10-500 A+1/8 is .842 (was .808, now above A). See `../ANALYSIS.md` §1.3 for the full before/after list.
>
> The preregistered pilot analysis is unaffected: the pilot used the P3 client and has no exception-terminated
> episodes.


**Preregistered decision: select R, the retrieval-weighted library leave-one-episode-out action reconstruction residual.** It predicts same-observation cache–policy disagreement in all eight cells. C fails the practical prediction-loss criterion; D is inconclusive; Qrisk passes but does not establish an improvement over R. **No score earns the status of a portable success-rate predictor or policy-call utility estimate.** The newer closed-loop evidence reinforces this distinction.

This report was produced on 2026-09-29. Sections 1–2 apply the unchanged preregistration or project its continuation. Section 3 is explicitly post-hoc. Section 4 proposes future validation, not an experiment performed here. All computations were CPU-only, with writes confined to this directory.

## 1. Preregistered all-cell result

### Execution and population

The authoritative inputs are the eight strict coordinator table directories under `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/`: `pi05_l10_50`, `pi05_l10_500`, `pi05_sp_50`, `pi05_sp_500`, and the four corresponding `groot` cells. `assemble_pilot.py` selected exactly those directories and preserved input hashes/audits in [joint_tables/source_manifest.json](joint_tables/source_manifest.json). No failed-table directory or repair archive was added. The repaired `groot_l10_50_window_r2` is included once through the final strict tables; the accepted-attempt correction is documented in `../p3_profiling/ATTEMPT_REUSE.md`. Its special history does not change the primary A population.

The **unchanged** [pilot_q1.py](pilot_q1.py), SHA256 `9a51ec7eb62378fedf8552aecb429253facdfc0c421e3d64c23bb68dc7a1de6d`, ran jointly over all eight cells, with 10,000 shared hierarchical bootstrap draws and seed 60601. It processed **4,320 episodes, 92,209 anchors, 180,155 decisions, and 1,210,330 action-step rows**. All **48/48** catalog-to-library provenance bridges passed. All 216 design slots were complete. [verify_joint.py](verify_joint.py) independently assembled the coordinator's per-cell episode scores and reran the frozen joint inference: estimates and intervals agree within `1e-10`, with the identical selection decision. This is not an average of eight per-cell intervals. See [run_manifest.json](pilot_all_cells/run_manifest.json), [completeness.json](pilot_all_cells/completeness.json), and [cross-check](joint_crosscheck/validation.json). The manifest's literal restriction string mentioning implementation-time smoke status is historical text in the frozen executable; its actual `smoke=false`, counts, and inputs describe this pilot run.

The primary test uses A, validation init 1: **30 episodes per cell, 240 episodes total, 5,321 anchors**. Each cell has ten distinct validation task/init groups, each repeated in three seed blocks. Across cells these are only **20 shared suite/task/init groups**, not 80 independent scenes or 5,321 independent samples. Calibration uses A init 0, another 30 episodes per cell. All four primary scores and labels are available; R has defined within-episode ranks in 240/240 episodes. The split and coefficients were not adjusted after seeing outcomes.

The target E is standardized RMS between the **counterfactual cache proposal** and the frozen primary shadow-policy chunk on the same observation, over ten intended controls. It is not executed-versus-policy error on policy-served decisions. Ten is `min(2 × interface block length, chunk horizon)` here; the action mask and coordinate SDs come from the library. Each candidate receives the preregistered two-parameter nonnegative calibration `Ehat=a+b*x`, fitted only on calibration trajectories. A constant fitted calibration mean is the loss baseline. Tasks, initializations, repeats, and anchors within episodes receive the registered hierarchical weights.

### Primary survival and selection

These are **99.375% simultaneous intervals** for the family of four candidates × two primary endpoints. The rule requires pooled rho ≥ .20, MAE reduction ≥ 5%, both simultaneous lower bounds > 0, nonnegative point association and gain in every cell, and rank availability ≥80%. A failed candidate is rejected for practical utility only if a simultaneous upper bound is below its practical threshold; otherwise it remains inconclusive.

| Candidate | Within-episode Spearman rho | Validation MAE reduction | Frozen decision |
|---|---:|---:|---|
| C: deployed LOEO distance CDF | .226 [.164, .278] | 0.63% [−3.00%, 3.76%] | **Practical utility rejected** |
| D: distance / library pair scale | .149 [.040, .252] | 9.77% [3.89%, 14.98%] | **Inconclusive** |
| R: weighted LOEO action residual | **.468 [.391, .542]** | **15.40% [9.38%, 21.95%]** | **Survives; selected** |
| Qrisk: D + R + successor residual | .439 [.355, .510] | 18.61% [11.20%, 24.75%] | Survives; does not replace R |

Baseline MAE is **.182916** standardized action units; R gives **.154750**, Qrisk **.148868**. The preregistered complexity comparison finds Qrisk's absolute MAE improvement over R is **.005882 [−.004135, .016142]**, with the six-comparison-family **99.1667%** interval. Its lower bound is not positive, so the simpler surviving R remains selected. C's loss-gain upper bound is below 5%, and its π0.5 Spatial-500 gain is −14.24%. D misses the pooled .20 rank threshold and has negative rank in GR00T L10-500 (−.0458); its upper bound still permits practical signal, hence “inconclusive,” not rejection. Sources: [state estimates](pilot_all_cells/state_estimates.csv), [paired candidate comparisons](pilot_all_cells/candidate_comparisons.csv), [decision](pilot_all_cells/decision.json).

![Frozen primary tests](final_figures/primary.png)

| Cell | R rho | R MAE reduction |
|---|---:|---:|
| π0.5 L10-50 | .372 | 5.43% |
| π0.5 L10-500 | .568 | 23.63% |
| π0.5 Spatial-50 | .418 | 18.29% |
| π0.5 Spatial-500 | .637 | 22.06% |
| GR00T L10-50 | .326 | 5.83% |
| GR00T L10-500 | .547 | 27.05% |
| GR00T Spatial-50 | .421 | 15.98% |
| GR00T Spatial-500 | .457 | 14.69% |

These are cell point estimates, not eight separate significance claims. Among 80 bank/task points, R's mean rank is positive in **76/80**, ranging **−.122 to .802**. Each task has only one validation initialization; the frozen code correctly leaves its init-based interval unavailable. Consequently the global survival result is **not** a certificate for every task or state. [Per-task output](pilot_all_cells/per_task_state_estimates.csv) and [retrieved-progress bins](pilot_all_cells/phase_descriptive.csv) locate failures; the latter is an anchor-weighted occupancy description, not another survival test or a semantic phase classifier. Fixed-task unseen-init intervals are unavailable in this pilot split.

### Registered secondary checks and outcome validation

Secondary dispersion and V7 remain comparators, not replacement winners. Their intervals below are **descriptive 95%**, outside the primary selection family.

| Secondary score | Pooled rho [95%] | MAE reduction [95%] |
|---|---:|---:|
| Kernel action dispersion | .434 [.388, .480] | 14.79% [12.13%, 17.75%] |
| V7 predicted error | .394 [.349, .441] | 18.53% [15.36%, 21.69%] |
| Negative effective-neighbour count | −.041 [−.087, .001] | −.06% [−.25%, .11%] |

R's rho advantage over C is **.242 [.185, .304]**, descriptive 95%. Leave-one-cell-out error calibration gives R **20.07% [15.76%, 24.96%]** pooled MAE gain against its own pooled-calibration constant baseline. The baseline differs from the primary one, so this is not a claim that leaving out data improves the same predictor. Every held-cell point gain is positive. R also retains positive pooled association under every other cohort (rho **.478–.526**; loss reduction **14.77–18.92%**). This tests these controllers' occupancies, not a new robot. Sources: [leave-cell-out](pilot_all_cells/leave_cell_out_state_estimates.csv), [other cohorts](pilot_all_cells/state_by_other_cohort.csv).

K4 gives **296** selected A-validation anchors. Cellwise selection-weight ESS is only **5.87–8.79 task/init groups**, with **4–14 zero-selection episodes** per cell. HT estimates of signed squared-error excess `L−V` are positive in all cells, with descriptive 95% lower bounds >0. The ratio of estimated head-sampling variance V to cache–draw squared loss L is **0.8–1.4% for GR00T** and **6.5–14.2% for π0.5**. Thus this measured component of sampling noise does not account for most disagreement. It does not measure uncertainty from resampling the entire upstream policy. Only **29/240 episodes** have enough K4 variation for R–excess rank estimates, including just one π0.5 Spatial-50 episode and zero in three other spatial cells; those correlations cannot support selection. See [policy-noise estimates](pilot_all_cells/policy_noise_estimates.csv) and [label sensitivities](pilot_all_cells/label_sensitivity.csv).

**The library-only global and per-task SR question is unresolved, with weak primary evidence.** Below are the registered LOEO R ranks against pilot outcomes; all intervals are descriptive 95%, with paired seeds and shared task/init bootstrap draws. Scores increase with risk, so positive correlation is the expected direction for each target.

| Target | Eight-bank global rho | 80 bank/task rho |
|---|---:|---:|
| A failure rate | .171 [−.229, .659] | .033 [−.176, .271] |
| P10 − A success rate | .193 [−.357, .539] | .001 [−.161, .173] |
| B − A success rate | −.204 [−.623, .270] | −.063 [−.262, .189] |

LOEO D and Qrisk global A-failure ranks are **.317 [−.209, .675]** and **.146 [−.169, .546]**. Held-inference reconstruction gives global A-failure rho **.464 [−.108, .659]**, still inconclusive. Across the **40 task-matched large-minus-small differences**, R differences associate with A-failure changes at **.311 [.091, .533]** and B-gain changes at **.329 [.087, .591]**. This is a limited within-model/suite size association, not 40 independent banks. The size-adjusted eight-bank R/A-failure correlation is **.211 [−.553, .952]**. All LOEO, inference-query and cache-query sources, task ranks, four model/suite size pairs, and size-adjusted sensitivities are in [bank_rank_correlations.csv](pilot_all_cells/bank_rank_correlations.csv). More repeated episodes cannot turn these eight banks into a transferable SR calibration sample.

The registered Q2 five-point screening rule flags only **π0.5 L10-500: B, dose .25, and dose .5**. B is the least-cost flagged choice: pilot validation SR **1.000**, IR **.1426**, versus P10 **.9333 @ .5047**; the simultaneous SR difference lower bound is **0**, and IR upper bound **−.3411**. This is an unstable ten-init screening result, not proof of near-pure-inference safety, nor authority to retune the rule. No other cell qualifies. The newer, larger experiments below are deliberately kept separate. See [budget screen](pilot_all_cells/q2_budget_screen.json) and [paired contrasts](pilot_all_cells/paired_contrasts.csv).

For the Q3 bridge, the frozen code uses only actual pre-guard coin opportunities, logged propensities and duration/hold probabilities; deterministic holds/guards are excluded. It keeps zero-opportunity episodes and bootstraps the ratio of macro expected HT contribution sum to expected opportunity count. Of **23 supported cell × C-tertile mixture contrasts**, only one has a positive descriptive 95% lower bound; none has a negative upper bound. GR00T Spatial-50's middle tertile has no eligible opportunities. Duration/hold packages are still less precise. No monotone quality-to-call-benefit rule is established; these are package excursions under the factorial controller, not isolated-call value under A. Assignment probability for the matched cohorts is one, not 1/9. See [support](pilot_all_cells/support.csv) and [factorial excursions](pilot_all_cells/factorial_excursions.csv). No new R-based causal subgroup was added to the preregistered tests.

## 2. What +31,680 continuation episodes would buy

The unchanged campaign yields 36,000 episodes total: **500 episodes / 250 distinct initializations per cohort/cell**, with **400 validation episodes / 200 initializations**. The validation repeat distribution is 80 initializations with three repeats, 40 with two, and 80 with one. Initializations retain equal weight, as frozen; the episode-weighted planning n_eff is not substituted.

[continuation_measured.py](continuation_measured.py) estimates paired-outcome components from all 60 pilot episodes/cohort/cell **for precision planning only**, before extrapolating. It uses balanced task × init × seed ANOVA (10 × 2 × 3; degrees of freedom 9/10/40): `v_init=(MS_init−MS_seed)/3`, `v_task=(MS_task−MS_init)/6`, `v_seed=MS_seed`. Raw ICC is `v_init/(v_init+v_seed)`. Negative components are exported and clipped to zero only for design projections; they are not evidence of negative physical variance. For K equally weighted initializations with repeats r_i,

`Var_fixed_tasks(mean) = v_init/K + v_seed * sum(1/r_i)/K²`.

For the preregistered task-superpopulation scope, add `v_task/10`. More initializations within the same ten tasks do not remove that term. These normal plug-in projections assume stationary variance and effect, and do not propagate variance-component uncertainty. Two initializations per task give very imprecise component estimates, particularly zero/negative ones.

The following replaces the planning variance .20 / ICC .30 for **A−P10**. V is the measured marginal paired sample variance; ICC is the raw within-task repeated-init estimate. Half-widths are projected **95%**, in percentage points, for the full **validation** sample; multiplicity-adjusted widths are larger.

| Cell | Measured V | Raw ICC | Fixed-task half-width | Task-superpopulation half-width |
|---|---:|---:|---:|---:|
| π0.5 L10-50 | .2836 | .586 | ±7.25 | ±7.25 |
| π0.5 L10-500 | .2023 | .250 | ±4.98 | ±11.57 |
| π0.5 Spatial-50 | .1989 | .250 | ±3.93 | ±19.69 |
| π0.5 Spatial-500 | .0777 | .000 | ±1.42 | ±16.07 |
| GR00T L10-50 | .3684 | −.167 | ±3.77 | ±33.74 |
| GR00T L10-500 | .2701 | .444 | ±6.77 | ±6.77 |
| GR00T Spatial-50 | .1523 | .069 | ±4.51 | ±4.51 |
| GR00T Spatial-500 | .1311 | .125 | ±4.17 | ±4.17 |

The narrow π0.5 Spatial-500 fixed-task projection reflects little repeat/init variance in these two scenes and substantial between-task variation; it is not a reliable guarantee of future near-ceiling precision. For **B−A**, projected fixed-task half-widths are **±2.47–6.04 pp**; task-superpopulation half-widths **±2.47–17.96 pp**. Full-validation within-task effective sample sizes for A−P10 are **235.8–315.8**, depending on the measured components. A simple independent-pair ±2 pp calculation using the observed marginal variances needs **747–3,538 effective pairs**, even before simultaneous coverage; a zero observed deficit is also needed for that width to certify a two-point margin. Power for noninferiority depends on the true gap, not just interval width.

For R's episode rank, measured marginal variances are **.0319–.0934** and raw repeated-init ICCs **−.285–.744**. At the full validation size, its projected per-cell **99.375%** half-width is **.024–.051** conditional on the fixed tasks, versus **.037–.143** with task-superpopulation variation. This makes better unseen-init and per-task calibration assessment plausible; it need not resolve every cell's loss gain or the R-versus-Qrisk complexity comparison. State precision calculations use all-init residuals, including calibration episodes, and are planning diagnostics only—not a second independent validation sample. See [SR projections](continuation_measured/SR_projections.csv), [state projections](continuation_measured/state_projections.csv), and [assumptions](continuation_measured/manifest.json). They also contain every fixed-dose comparison, actual discordance, 99% screening widths and 80%-power MDEs.

For the unresolved **Qrisk-versus-R** comparison, [pooled_precision.py](pooled_precision.py) first averages the four matched cells within each suite/task/init/seed, preserving cross-cell covariance, then applies the same variance-component projection. Its full-validation **99.1667%** half-width is **.006085** under the registered task-superpopulation scope, versus **.002633** for fixed tasks. The former still exceeds the current absolute improvement **.005882**: if that effect and these variance components persist, the continuation may still not justify replacing R. The narrower fixed-task projection cannot silently replace the frozen estimand. [Pooled projections](continuation_measured/pooled_projections.csv) also retain delta-method loss-gain precision; uncertainty in calibration and the estimated variance components is not included.

**Continuation decisions:** R's registered state-selection decision is already reached. Continue to test calibration stability, the weak individual tasks, K4 excess, dose contrasts and randomized package effects; retain D as inconclusive, without adding new tuning. Do not spend the continuation merely to reconfirm a pooled rank. The full campaign cannot supply independent bank acquisition, a new robot, or a counterfactual physical-success label at the same state. For global ranking, the preregistered minimum remains **three independently acquired banks per size × two sizes × four model/suite settings = 24 banks**, initially 50 distinct task/init pairs per bank/cohort for variance estimation, followed by measured-variance powering. A genuinely different benchmark/robot is required for the transfer claim.

## 3. Post-hoc / exploratory integration of later closed-loop evidence

**Everything in this section is post-hoc. It does not alter selection of R.** I read `../PAPER_AB.md`, `../ABLATIONS.md`, the 2026-09-28/29 §10 ledger entries, Q2's frontier specifications/calibration, and each included arm's own summary and accepted journal. [exploratory_final.py](exploratory_final.py) verified **102 completed 500-episode arms**, hashed **306 source files**, and formed **502 paired contrasts**. The fixed inventory includes 48 A/B replicate arms, 32 ablations, four P10 references, and 18 completed frontier arms; no unfinished arm or later completion is inferred. [arms.csv](exploratory_final/arms.csv) retains exact summary/journal paths, method names and kwargs; [input manifest](exploratory_final/input_manifest.json) freezes the source list.

Owner cost is recomputed from `cost_ledger`, never copied from its historical timing-weighted IR: `(.152*vision + .848*MISS)/(controls/5)` for π0.5 and `(.148*vision + .852*MISS)/(controls/5)` for GR00T. For the five-control request controllers this equals the requested `.152*v+.848*m` / `.148*v+.852*m`; an L10 pure-policy request executes ten nominal controls, so its IR is .5. Reported costs exclude profiling overhead. The pilot's frozen per-request and actual-control ledgers remain separate estimands.

A/B replicate means average seeds **within each of the same 500 (task, init) pairs**. Exploratory 95% intervals resample those 500 pairs within the ten fixed tasks, retaining all replicates (10,000 draws, seed 6062901). This differs from the primary pilot's task-superpopulation bootstrap. Nominal exact McNemar results are available for binary one-replicate comparisons; pooling 1,500 repeated-init outcomes as independent would overstate precision. All exploratory intervals are unadjusted and condition on these tasks/banks.

### Three-replicate A/B and trigger ablations

| Cell | A SR @ owner IR | B SR @ owner IR | B−A, pp [95% paired interval] |
|---|---:|---:|---:|
| π0.5 L10-50 | .714 @ .0764 | .8167 @ .1811 | +10.27 [7.33, 13.20] |
| π0.5 L10-500 | .8273 @ .0765 | .8733 @ .1589 | +4.60 [2.33, 6.80] |
| π0.5 Spatial-50 | .8367 @ .0775 | .9100 @ .1404 | +7.33 [5.60, 9.13] |
| π0.5 Spatial-500 | .9760 @ .0781 | .9820 @ .1195 | +0.60 [−.27, 1.53] |
| GR00T L10-50 | .6093 @ .0743 | .7187 @ .2209 | +10.93 [7.40, 14.53] |
| GR00T L10-500 | .8300 @ .0745 | .8667 @ .1874 | +3.67 [.53, 6.80] |
| GR00T Spatial-50 | .8673 @ .0753 | .8753 @ .1471 | +0.80 [−1.47, 3.07] |
| GR00T Spatial-500 | .9640 @ .0755 | .9587 @ .1265 | −0.53 [−1.87, .80] |

Source: [AB_three_replicates.csv](exploratory_final/AB_three_replicates.csv). These larger outcomes also expose how unrepresentative a pilot's two initializations can be; they are not folded into the preregistered validation set.

| No-progress removed, 50-episode library | SR @ owner IR | Change vs three-replicate B, pp [95%] |
|---|---:|---:|
| π0.5 L10 | .704 @ .1039 | −11.27 [−14.60, −7.93] |
| π0.5 Spatial | .858 @ .1124 | −5.20 [−7.07, −3.27] |
| GR00T L10 | .620 @ .0961 | −9.87 [−13.47, −6.33] |
| GR00T Spatial | **.906 @ .1053** | **+3.07 [1.33, 4.87]** |

This reproduces the guard reversal with lower cost. The other three individual guard deletions have no nominal significant difference against B's individual replicates (minimum p across those comparisons **.137**), which supports further simplification tests, not joint-removability or equivalence. For Q1, “poor progress/support/disagreement” cannot be treated as a universal causal reason to call the policy. R contains no LIBERO-specific progress guard, but its usefulness as a trigger still needs a direct test. Source: [paired contrasts](exploratory_final/paired_contrasts.csv), arm family `ablate_*` in [arms.csv](exploratory_final/arms.csv).

### Frontier completion: random calls help some libraries; equality is not established

Selected exact points and paired comparisons are below. B comparisons use its three-replicate mean; P10 is the matching historical 500-init L10 reference, with SR .904/.986 for π0.5 L10/Spatial and .866/.938 for GR00T L10/Spatial. Its nominal IR is .5.

| Cell / configuration | SR @ owner IR | Δ vs B, pp | Δ vs P10, pp [95%] |
|---|---:|---:|---:|
| π0.5 L10-50 B + .25 | .842 @ .2603 | +2.53 | −6.2 [−9.8, −2.6] |
| π0.5 L10-50 B + .5 | .880 @ .3388 | +6.33 | −2.4 [−5.6, 1.0] |
| π0.5 L10-50 B + .75 | .874 @ .4229 | +5.73 | −3.0 [−6.2, .2] |
| π0.5 L10-50 Q2 risk ρ=.35 | .862 @ .3416 | +4.53 | −4.2 [−7.8, −.6] |
| GR00T L10-50 B + .5 | .808 @ .3578 | +8.93 | −5.8 [−10.2, −1.4] |
| GR00T L10-50 B + .75 | .850 @ .4298 | +13.13 | −1.6 [−5.0, 1.8] |
| GR00T L10-50 Q2 risk ρ=.35 | .832 @ .3438 | +11.33 | −3.4 [−7.2, .4] |
| π0.5 L10-500 B + .125 | .842 @ .2062 | −3.13 | −6.2 [−10.0, −2.4] |
| π0.5 L10-500 B + .25 | .882 @ .2415 | +.87 | −2.2 [−5.6, 1.4] |
| π0.5 L10-500 Q2 risk ρ=.24 | .884 @ .2402 | +1.07 | −2.0 [−5.4, 1.4] |
| π0.5 Spatial-50 B + .5 | .962 @ .3153 | +5.20 | −2.4 [−4.4, −.4] |
| GR00T Spatial-50 B + .125 | .878 @ .1919 | +.27 | −6.0 [−9.405, −2.6] |
| GR00T Spatial-50 B + .25 | .902 @ .2340 | +2.67 | −3.6 [−6.8, −.4] |

All 18 points and their comparisons are in [frontier_summary.csv](final_figures/frontier_summary.csv). For example, the π0.5 L10-50 +.5 / GR00T L10-50 +.75 / π0.5 L10-500 risk .24 nominal P10 p-values are **.213 / .434 / .332**; none establishes a two-point noninferiority margin. In fact **none of the 18 points has an unadjusted 95% paired lower bound ≥−.02 versus P10**. Additional multiplicity control would not improve that conclusion.

The sparse L10 and π0.5 Spatial improvements from extra random calls are substantial. Dense π0.5 L10-500 does not show a reliable extra-call improvement over B: +.125 is worse in this run, whereas +.25 and risk .24 have small, uncertain gains. Dense Spatial A is already strong and B adds no established gain. This supports smaller budgets for well-performing dense banks, but does **not** prove that random calls are universally harmful in every dense bank. Increasing dose is not monotonically better even in the sparse π0.5 L10 sequence.

Q2's tested risk lottery is **not Q1 R**. Its adapter calibrates task allocation using standardized proprioceptive nearest-other-episode distances and episode-level lotteries between fixed doses. It does not use the selected action-residual signal at each state. Its mixed frontier performance therefore neither validates nor falsifies R-based placement. Exact existing configurations are retained in `arms.csv` and originate in `../ideation_Q2/frontier/adapters/`.

![Exploratory frontier](final_figures/frontier.png)

### Representation and global-score stress tests

Direct-token PCA versus pooled PCA produces **GR00T Spatial-50 .910 @ .0753**, a **+4.27 pp [1.07, 7.47]** change versus A's replicate mean; it gives **GR00T Spatial-500 .928**, **−3.60 pp [−6.00, −1.20]**, and **π0.5 Spatial-500 .946**, **−3.00 pp [−4.93, −1.20]**. The other five direct-PCA contrasts have intervals crossing zero. These are the arms' journal outcomes, consistent with `ABLATIONS.md` §2, not an assumption that higher-dimensional keys are better. A bank score belongs to **bank plus deployed representation/metric/kernel**, not just stored episodes. R must be recalculated after a PCA or metric change; reusing the pooled-PCA residuals would violate its provenance gate. The pilot validates neither that recalibration nor a prediction of this sign reversal.

Rechecking the old library-only LOEO R with the later A/B means gives global rho **.024 [−.024, .357]** with A failure and **.214 [−.048, .357]** with B gain. These are fixed-bank/fixed-task intervals, not new-domain intervals. The more precise outcomes do not rescue an absolute library-only SR score. [quality_rank_correlations.csv](exploratory_final/quality_rank_correlations.csv) contains all old sources/components and P10 gaps.

Two explicitly post-hoc checks address the owner's “few recorded trajectories” constraint:

* **Ten-trajectory refit:** keep only A init 0, first seed block (603), one trajectory per task per cell; leave all 240 validation episodes untouched. R's MAE reduction is **14.83% [9.38%, 20.62%]**, using the same endpoint-family confidence level; all eight cell point gains remain positive. R's rank is unchanged because nonnegative scalar calibration does not change the underlying score ranks. This supports a ten-trajectory starting recipe; it is not the preregistered 30-episode calibration result. Sources: [few_trajectory_check.py](few_trajectory_check.py), [estimates](few_trajectory_check/state_estimates.csv).
* **Occupancy-calibrated global score:** average that refit's predicted error over the ten calibration trajectories and compare against later A/B outcomes on **inits 2–49 only**, 480 distinct pairs per bank, excluding both pilot inits. Across the eight fixed banks, raw mean R has A-failure rho **−.048 [−.095, .286]**; calibrated mean error has **.333 [.286, .619]**. For B gain, calibrated mean error gives **.643 [.476, .714]**. These conditional intervals hold calibration and banks fixed and do not include their uncertainty. Moreover, the affine fit matches the calibration mean E, so this aggregate is largely a ten-trajectory shadow-disagreement estimate, not proof that R adds global ranking information. π0.5 Spatial-500 still receives lower Q than π0.5 L10-500 despite much higher A SR. This is a useful Q2 prior for study, not an SR certificate. Sources: [occupancy_bank_check.py](occupancy_bank_check.py), [bank scores](occupancy_bank_check/bank_scores.csv), [ranks](occupancy_bank_check/rank_correlations.csv).

## 4. Concrete R6 recommendation and falsification

### Portable signal and calibration

Use **R as the selected state-disagreement diagnostic** and a candidate input to allocation. Retire `D+R+S` as the default Q1 scalar: Qrisk survives, but its added terms do not pass the registered replacement test. Keep C, distance, dispersion, neighbour count and progress as support/audit fields, without promoting them to proven rescue signals. Do not report any scalar as a probability of task success.

The exact recipe is:

1. **Freeze the deployment contract.** Fingerprint the bank rows/episode identities, observation representation, retrieval fit, task/instruction conditioning, metric regimes, kernel, action validity mask, block length b and stored horizon H. Use exactly the deployed retriever. Unsupported task or fewer than two eligible library episodes yields an unsupported flag, not a fabricated cross-task score.
2. **Library units.** Compute population SD `sigma[d]` of each valid action coordinate over the first b controls of every stored chunk. Replace SD ≤`1e−12` by one, and expose that fallback/unsupported variation. Let `h=min(2b,H)`. This is h=10 here, not a robot-independent ten-step rule.
3. **Conditional LOEO residuals.** For every stored row j, exclude its entire episode from candidate retrieval, using the deployed metric regime for that row. Mix stored chunks with the actual kernel and save `r_j=RMS((mixed_chunk[:h]−stored_chunk_j[:h])/sigma)`. Freeze the deployed transform; call this conditional LOEO, since the fit may have seen the omitted episode. Full inductive refitting is a different, unvalidated procedure. In these measured A fits the kernel uses top 16 and `w_i ∝ exp(-((d_i−d_1)/max(d_kref−d_1,1e−6))²)`, with deployed kref 5 at small banks and 8 at large banks. These settings are inherited from A, not imposed on another robot.
4. **At an anchor.** Consume the current retrieval's normalized weights and row IDs and compute `R(s)=sum_i w_i r_i`. This needs one saved scalar per bank row and no shadow-policy call online. It predicts interpolation/reconstruction difficulty, not physical failure. Keep missingness, library fingerprint and nearest-distance support beside it.
5. **Optional calibration, at most ten recorded trajectories to start.** Preselect independent A-like initial conditions across the intended task mix before looking at errors/outcomes. Obtain same-observation frozen policy chunks along those recordings; a log with only served cache actions cannot supply these labels. Fit `a,b≥0` to `E≈a+bR` by weighted least squares, equal tasks, initializations, repeats and then anchors within episodes. Use all usable anchors but count trajectories, not anchors, for uncertainty. No success label, LIBERO geometry, gripper convention, camera count or simulator predicate enters this fit. Ten worked in the post-hoc pilot sensitivity, not as a universal sample-size guarantee. Tasks missing from the recordings get a calibration-coverage flag. With no policy-labeled recordings, retain raw R as an uncalibrated rank diagnostic.
6. **Expose the scalar with its meaning.** `Qstate=1/(1+a+bR)`. For each task average predicted error over anchors within each reserved trajectory, repeats within initialization, then initializations; `Qtask=1/(1+mean predicted error)`. Average task mean errors with declared deployment task weights before applying the same transform for Qglobal. Export uncertainty, score quantiles and coverage beside the scalar. On the library itself use equal-episode/task LOEO residual summaries and label them library audits, not sampled deployment occupancy. No task-specific regression is justified by one pilot initialization.
7. **Recalibrate and revalidate when the contract changes.** A new robot, action horizon, encoder/PCA, metric, kernel or materially changed bank gets new residuals and normalization. Reserve different trajectory groups for validation; never split adjacent anchors as independent train/test cases. `init%5` is this experiment's grouping convention, not part of the portable method. A genuinely new domain must rerun the rank/loss checks before making the transfer claim.

The primary coefficients and IDs are already in [calibration_parameters.json](pilot_all_cells/calibration_parameters.json); the ten-trajectory alternative is in [few_trajectory_check/calibration_parameters.json](few_trajectory_check/calibration_parameters.json). They are separately versioned; the alternative does not overwrite the frozen result.

**Q2 budget:** supply global/task predicted-disagreement distributions from reserved trajectories as a prior and a calibration-coverage audit. Select the actual budget using paired SR/IR evidence against the declared pure-inference reference. There is no validated map from Qglobal to “needed calls.” The dense/sparse frontier difference and the calibrated-mean-error association motivate testing such a map; the remaining global misrankings prohibit using it as a success constraint. Keep both inference commitment and the owner's cost ledger explicit.

**Q3 placement:** supply pre-assignment R, Ehat and support at the current anchor. At a fixed budget compare R-prioritized calls with uniform calls. A high score only means expected teacher disagreement; even a large, systematic disagreement may be harmless or the teacher may be worse. The no-progress reversal makes a separate utility test essential. Neither a policy shadow label nor a positive R correlation authorizes a universal hard gate.

### At most eight future configurations, paired against A, B and pure inference

This is a **new proposed test**, not part of the frozen analysis and not launched. Hold each cell's bank and pooled representation fixed. Use identical task/init/seed manifests across all configurations; keep scoring-calibration trajectories disjoint. Six configurations specify baseline/dose controls; two require an R-based lottery implementation by the coordinator. P5 still needs a matched reference collection.

| # | Exact configuration | Paired comparison and predicted result (hypothesis) |
|---|---|---|
| 1 | Deployed A: library's existing retriever, top-16 and recorded kref; vision anchor then one blind block; no policy calls | Cache-only SR/cost reference. |
| 2 | Deployed B unchanged, cell's existing `replay_baselines.json` method/kwargs | Measures benefit/cost of present guards; do not silently remove no-progress in this Q1 test. |
| 3 | P10: fresh policy at every vision anchor; commit `min(2b,H)` controls, no additional blind-tail recomputation | Primary ten-control pure-policy reference; nominal owner IR .5 here. |
| 4 | P5: fresh observation and policy every b controls | Distinguishes approximation to full-rate inference from approximation to P10; nominal owner IR 1 here. |
| 5 | A plus independent Bernoulli policy at each vision anchor, p=.25; policy serves the same h-control commitment, no B guards, holds or delayed calls | Uniform moderate-budget control. |
| 6 | Same as #5, p=.50 | Uniform higher-budget control; sparse-bank benefit is plausible, universal dense-bank benefit is not. |
| 7 | Same as #5, replace uniform p by `min(1,lambda_.25 * max(Ehat(s),1e−12))` | Compare with #5 at matched measured IR; hypothesize better use of calls if disagreement identifies recoverable errors. |
| 8 | Same as #6, with `lambda_.50` | Compare with #6; hypothesize smaller placement advantage as the uniform budget increases. |

For #7–8 calibrate lambda **only** on the at-most-ten reserved recordings with the same task/init/episode weights as the error fit: solve weighted mean p=d by **80 bisection steps**, bracket `[0, 1/min(max(Ehat,1e−12))]`, d=.25 or .50. If the score is constant this reduces to uniform d. Never calibrate lambda on validation SR. Log every actual propensity and assignment, use an independent coin keyed by the common seed/task/init/anchor, and preserve the same h-control commitment for cache and policy. These probabilities match the calibration occupancy, not necessarily future occupancy. Therefore treat placement superiority as unresolved if actual owner IR differs by more than **.01 absolute** between its paired R/uniform arms; compare dose–cost curves or independently recalibrate rather than claiming success from an unequal-cost comparison.

Predeclare the two dose-matched placement contrasts across eight cells (16 comparisons), shared-init clustered bootstrap and 95% family coverage. A useful placement result is a positive SR improvement at matched IR, while the ultimately selected deployment configuration must independently support a **two-percentage-point SR margin** to the owner's chosen pure reference and reduced IR. Report both P5 and P10 gaps; do not convert a nonsignificant difference into noninferiority. The two-point proposal is stricter than the original five-point pilot discussion screen. Start with the continuation's 250 distinct initializations/500 episodes per configuration/cell for variance estimation; the measured projections above explain why extra independent pairs may still be necessary. The existing continuation can provide A/B/P10/uniform controls, but does not itself run #7–8 or P5.

**What falsifies which claim:** a new-domain failure of the registered rank/loss thresholds falsifies the state-prediction transfer claim; systematic failure on individual tasks limits task applicability even if a pooled result survives. R-prioritized calls failing to improve SR over matched-cost uniform calls, or increasing losses relative to pure inference beyond the chosen margin, falsifies using disagreement as the placement signal at that operating point. A replicated higher calibrated Qglobal but lower SR across independently acquired, equal-size banks falsifies an SR-ordering claim; it does not invalidate an imitation-error diagnostic. A PCA/metric change without recomputed residuals is an invalid application rather than a successful transfer test. Counterfactual restored-state branch outcomes, independent banks, genuinely new robots and full-policy uncertainty remain missing; none is created by this analysis.

## Reproduction and delivered artifacts

All commands below were run from `/home/weiland/projects/openpi`. Output directories are immutable products; the assembly/exploratory/planning scripts intentionally refuse an existing output directory. For a fresh reproduction, use a fresh copy of the output location **inside Q1's directory**, not deletion or changes to sources. The frozen runner was not edited.

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/assemble_pilot.py
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/pilot_q1.py --tables exp/offline_search/rounds/r06/ideation_Q1/joint_tables --out exp/offline_search/rounds/r06/ideation_Q1/pilot_all_cells
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/verify_joint.py
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/continuation_measured.py
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/pooled_precision.py
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/exploratory_final.py
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/few_trajectory_check.py
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/occupancy_bank_check.py
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/final_figures.py
```

Corresponding `*.log` files record successful execution. The primary CSV/JSON outputs and registered plots are in [pilot_all_cells/](pilot_all_cells/); exploratory outcomes are in [exploratory_final/](exploratory_final/); measured continuation projections in [continuation_measured/](continuation_measured/). Standalone report figures are supplied as PNG and PDF in [final_figures/](final_figures/). [PREREG.md](PREREG.md), the original [REPORT.md](REPORT.md), and its [data wishlist](DATA_WISHLIST.md) remain unchanged historical records.
