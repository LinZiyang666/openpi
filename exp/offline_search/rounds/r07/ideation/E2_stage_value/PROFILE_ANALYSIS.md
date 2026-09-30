# E2: empirical stage value — PROFILE analysis

**Recommendation: freeze 28 candidate arms: SF1 ×8, UF1 ×8, SW ×4, CU30 ×4, CT30 ×4. Choose SF cap E=1 globally. Drop SF2 from this evaluation; defer SF+SW ×4 and do not build SA now.** Keep the paired A references. These are research evaluations, not declarations of safety or stage-call superiority.

Sources: [SELECTION](../../SELECTION.md), hand-backs [C1](../../c1_follow/HANDBACK.md), [C2](../../c2_wrist/HANDBACK.md), [C3](../../c3_calls/HANDBACK.md), [C4](../../c4_profile/HANDBACK.md), [closed-loop report](../../profile_results/closed_loop/profile_report.json), all four offline reports, and accepted raw logs under `/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/runs/`. Reproduction: [profile_stats.py](profile_stats.py); detailed numbers: [profile_metrics.csv](profile_metrics.csv), [profile_stats.json](profile_stats.json). No implementation, shared artifact, or evaluation configuration was changed.

**Denominators and uncertainty.** I independently reconciled **44 arms, 880 accepted episodes, 37,999 decisions**, with zero outcome/count/component-cost/lottery-coin/CT-weight mismatches. Each arm matches its cell's A on all 20 task/init pairs. Tables report owner IR = total accepted owner cost / requests; a request nominally covers five controls. “Paired saving” is the mean of the 20 episode IR differences, A minus variant, with C4's task-stratified init bootstrap. These two averaging conventions are distinct. All savings pass under either convention.

Twenty episodes (two per task) establish that routing, support, and cost saving actually occur on these starts. One outcome changes cell SR by **5 pp**. They cannot establish the −1/−1.5 pp noninferiority margins, stage-specific recovery, rare-event safety, robot transfer, or absence of harm. SR comparisons below are **descriptive only**, including pooled counts; repeated arms are not independent initial states. Cost intervals condition on these fixed tasks and the very small sampled init set. Dense banks include acquisition from the B pool: this is not an independent bank-generalization study. No new per-control telemetry exists, so nominal-request IR must not be presented as actual-control IR or measured latency.

**1. Gates and disposition of the built variants**

Each follow entry below is **IR / eligible anchors (%) / successes out of 20**. Eligibility counts a positive anchor grant once, not its repetitions in blind telemetry.

| Cell | A: IR / successes | SF1 | SF2 | UF1 |
|---|---|---|---|---|
| π0.5 L10-50 | .07629 / 15 | .06544 / 35.3 / 14 | .06140 / 26.9 / 11 | .05513 / 78.8 / 17 |
| π0.5 L10-500 | .07668 / 20 | .06788 / 26.6 / 20 | .06422 / 19.0 / 18 | .05599 / 73.9 / 13 |
| π0.5 Spatial-50 | .07728 / 15 | .06724 / 30.4 / 13 | .06181 / 25.8 / 16 | .06086 / 52.5 / 14 |
| π0.5 Spatial-500 | .07781 / 18 | .06238 / 48.4 / 18 | .05643 / 38.2 / 18 | .05527 / 81.0 / 18 |
| GR00T L10-50 | .07427 / 14 | .06487 / 31.0 / 14 | .06038 / 23.6 / 13 | .05502 / 71.2 / 14 |
| GR00T L10-500 | .07467 / 18 | .06796 / 19.1 / 17 | .06522 / 14.2 / 16 | .05292 / 82.7 / 17 |
| GR00T Spatial-50 | .07519 / 17 | .06416 / 36.4 / 16 | .05865 / 28.5 / 15 | .05981 / 53.0 / 15 |
| GR00T Spatial-500 | .07510 / 18 | .06536 / 32.7 / 18 | .06159 / 22.4 / 18 | .05410 / 77.0 / 18 |

**SF1: KEEP ×8.** Eligibility 19.1–48.4% exceeds 5%; paired IR savings .00795–.01695 exceed .005. Even the smallest lower 95% saving bound is **.00700** (GR00T L10-500; estimate .00795, interval [.00700,.00891]). Across eight cells, mean paired saving is .01145; 914 extra blocks were actually served. C1 reports exact disabled-A parity on 10,569 decisions and 6,612 enabled structural/stage-rejection checks. The early-valve exception below limits the stronger commitment-preservation claim, but does not fail §4's literal *lever-off* parity gate.

**SF2: DROP from the frozen list, not for a numerical §4 failure.** Eligibility 14.2–38.2%; paired savings .01146–.02364, minimum lower bound .01014. It passes those gates and disabled parity. It adds only **.004995** mean paired saving over SF1, while extending to 20 rather than 15 nominal controls and producing 20 valve LOOKs versus 13. Descriptive successes are 125/160 versus SF1 130/160 and A 135/160; these do not prove SF2 harmful. E=1 is the smaller intervention, already materially clears the cost gate, and preserves a matched-cap UF1 control. Do not choose different caps by cell from these outcomes.

**UF1: KEEP ×8 as the necessary allocation control.** Eligibility 52.5–82.7%; paired savings .01725–.02289, minimum lower bound .01563. Disabled parity passes. Its 126/160 successes versus SF1 130/160 are inconclusive. In π0.5 L10-500 it loses seven A-success pairs and rescues none; this is a warning to retain the control, not a validated cell-specific prohibition. SF versus UF changes both the stage gate and valve, so their contrast will identify that package, not the stage label alone.

**SW: KEEP ×4.** Real GPU parity is now passed: `/tmp/r7_C2/gpu_parity.json` records all four equality checks on all 12 observations, maximum policy absolute difference **0**. This supersedes C2's older pending-parity notice; its disabled identity covered 5,173 decisions. Actual wrist anchors and owner savings are:

| π0.5 cell | Wrist / vision anchors | Owner IR | Paired saving [95%] | SW / A successes |
|---|---|---|---|---|
| L10-50 | 217/624 = 34.8% | .05950 | .01864 [.01741,.01985] | 16/15 |
| L10-500 | 123/536 = 22.9% | .06538 | .01249 [.01133,.01374] | 19/20 |
| Spatial-50 | 75/274 = 27.4% | .06368 | .01533 [.01364,.01701] | 15/15 |
| Spatial-500 | 99/244 = 40.6% | .05742 | .02271 [.02069,.02474] | 17/18 |

All exceed even a 5% camera-use screen and the .005 saving gate. There were **zero policy calls and zero completions**: these rollouts exercise the saving path, not wrist-origin MISS completion. Same-observation offline B-val wrist-minus-full head RMS for L10-500 is **+.02377 [.00955,.04335]**; this flags a retrieval-quality tradeoff, not an SR kill criterion. It is an all-anchor diagnostic, not specifically the SW-admitted subset. Preserve all four cells in the full test.

**CU: KEEP ×4 as reference. CT: KEEP ×4 only as a falsifiable stage-allocation experiment.** All eight pass [.27,.33] under both request-weighted and equal-episode IR. CU/tilt-off identity is supported by C3's 26,992 R6 comparison pairs; CT retains that path when disabled. Call arms intentionally cost more than pure-cache A; the .005 *saving* gate applies to the saving variants, not to this .30 budget experiment.

| Sparse cell | CU IR | CT IR | Paired CT−CU IR [95%] | CU→CT successes /20 | CT wins/losses vs CU |
|---|---|---|---|---|---|
| π0.5 L10 | .29899 | .31842 | +.01561 [−.00067,.03231] | 18→19 | 2/1 |
| π0.5 Spatial | .30180 | .29265 | −.01189 [−.03248,.00920] | 18→16 | 0/2 |
| GR00T L10 | .30710 | .29661 | −.01373 [−.02333,−.00415] | 15→15 | 4/4 |
| GR00T Spatial | .30464 | .31619 | +.01234 [−.00266,.02703] | 17→18 | 1/0 |

Both total **68/80 successes**, with seven wins and seven losses. Equal-cell request-weighted ΔIR is +.00283; mean paired episode ΔIR is +.00058. π0.5 L10's cost gap is **+.01943** by request weighting and +.01561 by paired episodes, already slightly beyond the eventual .015 matched-budget margin. This does not fail the profile ±.03 target gate; it does prevent describing that cell's +5 pp as a matched-cost gain. Retain the frozen calibrations; do not tune to these SRs or tune on eval500 to rescue a claim.

**2. What happened to E2's proposals?**

**P1, first high-deviation entry call allocation: CHANGE its status to an unadmitted hypothesis.** A deployable P1 still fails my evidentiary admission criterion; keep uniform+stall as the default. Selected CT adds event weighting as well, so it is not a clean test of the deviation component. Its independent preregistered whole-controller comparison is still worth completing: the mechanism is active and passes every profile feasibility gate. The 20-episode tie is not adequate evidence to trigger my matched-IR *full-evaluation* kill rule.

CT has **35/10/26/10 deviation entries** in the four table-order cells, across 16/9/12/7 episodes. Current-anchor high-deviation occupancy is **21.9/25.5/29.3/18.2%** versus the nominal library upper quartile: no gross occupancy collapse, but this is not a calibration or representation-transfer proof. All **81 entries have nominal p=1** after clipping; 72 execute calls and nine are suppressed by stall cooldown. Twelve entry calls are already mandatory stall calls. Thus these logs contain no randomized cache-vs-call support at the entry itself; do not estimate an entry treatment effect from the nine cooldown exceptions.

The tilt changes much more than those 81 entries: event mass is positive at 247/584, 152/274, 349/651 and 124/247 anchors. Interior nominal probabilities fall to **.361/.295/.326/.358**, compared with CU λ≈.50. Nominal clipping occurs on **346/1,756 CT anchors (19.7%)**. These are implementation-confirmed allocation changes, not evidence those stages benefit from them. My weight/formula/coin checks found no mismatch.

Offline [stage_value](../../profile_results/offline/value_all/stage_value.json) uses **187,221 supported anchors from 23 R6 arms**. Of 244 natural-dose first-supported-stage-entry comparisons, eight lack an observed branch; **235/236 remaining intervals include zero**. The sole exclusion is `unknown` in π0.5 Spatial-500 risk ρ=.105: +1.317 pp [.018,3.098], only 49 entries/four calls (call ESS=4). This is not a usable hard-stage rule. Bonferroni adjustment is within arm/estimand/target, not across the entire search.

Crucially, C4's strata are gripper-run/event/mixed labels; it does **not** estimate the library-p75 deviation-entry-versus-interior interaction requested in P1. Its zero-for-unreached-episode probability-shift estimand also differs from my original conditional visited-entry contrast. Do not compare their magnitudes as replications. Held-task/continuation validation of the actual P1 quantity remains absent. R5 Q3's unsupported call suppression and R6's random=predicted-disagreement placement are therefore unrebutted; a failure marker still need not predict a beneficial intervention.

**P2, observation without commitment interruption: CHANGE the interpretation; defer the actual causal test.** SF/SW are useful descendants for saving vision, but neither implements randomized LOOK-only versus LOOK+replace versus continued tail. A valve LOOK performs normal fresh retrieval and can change actions. Calling this a successful implementation of my commitment-preserving LOOK-only proposal would violate its own kill criterion. Keep SF1/SW as separate look-less/look-half experiments; do not credit their costs or SR to a pure observation effect.

[failure_clock](../../profile_results/offline/clock_all/failure_clock.json) sharpens the signal distinction on 240 old A episodes (B-val plus P3 A_r0). Absolute state deviation ever crosses in **43/43 failures and 184/197 successes**; it is almost ubiquitous. Relative displacement valve crossings occur in **25/43 failures versus 12/197 successes** (58.1% versus 6.1%). Stall replay is available only for P3 anchors: confirmed stalls occur in **26/26 P3 failures versus 34/134 P3 successes**; B-val is missing, not negative. These are retrospective episode associations, not action-value estimates, and the clock evaluates displacement crossings without SF's eligibility restriction.

Among failures with both valve and stall alerts, the valve precedes stall in **7/16**. Its median lead to terminal failure is 334 controls among the 25 alerted failures, but terminal timeout is not irreversible-failure onset. The profiles supply neither grasp-intent annotations nor a counterfactual rescue. Thus stable, state-consistent interiors support an *opportunity to test* look less; deviations support caution; neither establishes a safe call-skip region. My P2 timing/recoverability kill criteria remain untested, not passed.

**3. Bugs, specification and accounting issues**

- **Early-valve semantic conflict is real.** SF1 has 13 valve LOOKs, **eight at age 1**, after only five controls (π0.5 L10-50/500: 4/1; GR00T L10-50: 3). SF2 has 20 valve LOOKs, also eight at age 1. These can replace A's committed tail before any extension. This is documented in C1, not a newly discovered arithmetic bug. Freeze SF1 explicitly as “follow plus immediate valve-triggered retrieval”; do not claim identity whenever no extra block is served or a pure longer-commitment experiment. If the coordinator instead requires unbroken initial ten-control commitment, the method must change and be reprofiled before freezing it.
- **Two legitimate IR bases are easy to mix.** Raw `summary.json` uses the operations eager-cost table. Example SF1 π0.5 L10-50: summary .06111 versus owner .06544; SW L10-50: .05557 versus owner .05950. C4 correctly recomputes owner costs; no lost/double-counted work was found. Wrist .055198 and completion .049890 remain proportional-latency assumptions. The summaries explicitly label `controls=N×5` nominal and actual controls null; server timings under these launches do not validate the owner prices.
- **Per-request fresh/carried semantics matter.** CT stage extras persist on blind tails, and cooldown overrides the nominal entry probability. Counting every logged entry flag or treating nominal p as actual p would create a false treatment analysis. My counts use `os_c_fresh=1` and effective `os_c_p`; 1,756 fresh CT anchors are distinct from its 3,383 total decisions.
- **Missing interaction and composition evidence is a reporting gap.** The offline stage tool does not answer P1's deviation-entry interaction, and the new stock client cannot answer physical failure timing. SF+SW and SA have **zero arms among the 44 closed-loop profile reports**; CPU composition parity and independent component savings do not establish their realized eligibility or saving. No other implementation/accounting error is established by this audit.
- **Stall-clock missingness must stay explicit.** All **3,606 B-val decisions** have `stall=not_replayed`; their `first_confirmed_stall=null` is not evidence of no stall. P3 has replay states on 3,524 anchors and `not_replayed` on 3,439 other decisions. Pooling the JSON's 26 stall-positive failures over all 43 failures would conflate missing replay with a negative signal; the comparable denominator is 26 P3 failures.

**4. Freeze recommendation and remaining evidence**

Freeze **SF1×8 + UF1×8 + SW×4 + CU×4 + CT×4 = 28 candidates**, with corresponding A references; do not withdraw any Freeze-1 arm on these data. Drop SF2×8 as the unchosen cap. **Defer SF+SW×4** from the present freeze: combining wrist retrieval with successor-follow changes the retrieved kernel, eligibility, and valve trajectory. Give that already-built combination its own non-test profile before any later admission; savings cannot be added. This is my conservative pruning recommendation, not a claim that SELECTION originally required a separate composition campaign.

**SA: not worth building now.** CT has no demonstrated benefit, SF safety remains unresolved, and composition changes the cadence underlying the .30 solve. C3 explicitly requires a fresh joint cost solve, combined telemetry audit and profile; reusing CU/CT λ would be invalid. If full evaluation supports vision savings, start a later composition with CU+SF1, optionally wrist after SF+SW is verified. That ordering is a proposed experiment, not an observed frontier gain.

Keep SELECTION §5's full-evaluation gates: SF pooled SR lower bound >−1 pp and lower IR in ≥6/8 cells; stage/valve advantage versus UF by its stated superiority-or-harm criterion; SW lower bound >−1.5 pp and lower IR in 4/4; CT−CU lower bound >0 at |ΔIR|≤.015. Do not promote profile point estimates into those claims.

**Top follow-up profile request:** extend `stage_value` with the *actual frozen library-LOEO deviation p75 and entry latch*, separate event and deviation contributions, report effective 0/1 support explicitly, and test entry-minus-interior interactions with held-task and continuation checks. Use existing randomized R6 logs for supported causal contrasts; CT's deterministic entry calls cannot supply that contrast. Preserve the separate randomized LOOK-only/LOOK+replace question for a later study.

Reproduce the audit from the repository root with:
`taskset -c 24-25,68-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E2_stage_value/profile_stats.py`
