# Q3 final: where to spend MISSes

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


**The preregistered pilot nominates no placement gate in any cell.** None of the 64 primary SR contrasts excludes zero under the frozen simultaneous intervals. The later, explicitly post-hoc evidence establishes that the **no-progress controller package helps on both LIBERO-10-50 cells and hurts on GR00T Spatial-50**. It does not establish a library-only signal that predicts this sign change. My recommendation is to test a replacement that calls on **reliably estimated slow progress**, while treating ambiguous trajectory alignment as a reason to reobserve. This is a concrete, unvalidated proposal; it is not a gate selected by the preregistration or a demonstrated solution at B's IR.

All results below were computed from the completed tables or the arms' own summaries and accepted episode journals. No experiment, simulator, server, repair, or deployment was run. All new files are in this ideation directory. The numerical supplement is [FINAL_TABLES.md](FINAL_TABLES.md); machine-readable provenance and checks are in [final_delivery.json](final_delivery.json). Scientific figure: [PNG](FINAL_evidence.png), [PDF](FINAL_evidence.pdf).

## 1. Preregistered all-cell result

### What was held fixed and checked

I ran the unchanged [analyze_pilot.py](analyze_pilot.py) once over all eight exact `r06_p3_pilot/tables/<cell>/` directories. Its SHA256 remains `d2b040c94d84af22f40259c304cbccf3ba2bfad6aeb9c48ae8dbda0bffd67a2c`; [PREREG_FROZEN.md](PREREG_FROZEN.md) remains `61b2e5696ec95e7e767d8e6ad87d3e4ed1ef67494b85e1d6bed5c22b199bfb13`. “Joint” means the prespecified family of eight cell-specific analyses, **not pooling different cells or continuation policies into one effect**.

The audit contains **216 arms, 4,320 episodes, 92,209 anchors, 180,155 decisions, and 893,276 actual controls**. All 92,209 assignments replayed. All eight joint per-cell effect tables reproduce the coordinator's previous tables to numerical tolerance `1e-12`. The independently reconstructed episode scores reproduce all 64 primary point estimates; their measured variance components recover all **54 estimable** frozen fixed-task standard errors. Sources: [pilot_all/audit.json](pilot_all/audit.json), [final_pilot/audit.json](final_pilot/audit.json), [score_reconstruction64.csv](final_pilot/score_reconstruction64.csv), [final_delivery.json](final_delivery.json).

The repaired `groot_l10_50_window_r2` is included through the final strict tables. I read [ATTEMPT_REUSE.md](../p3_profiling/ATTEMPT_REUSE.md) and did not recursively reread `runs/` or `repair_archive/`. The documented correction identifies the accepted driver attempt using control hashes and timing; it does not supply an extra episode. Source table hashes are retained in the joint audit.

The frozen estimator uses the logged free-coin propensity, including the distance-bin probabilities `.125/.25/.5`, and the HT score `Z*Y/p − (1−Z)*Y/(1−p)`. It averages over eligible **opportunities under that continuation controller**, with equal task/scheduled-episode weights; it never weights an anchor by inverse realized trajectory length. The factorial has **6,555 free roots, 2,329 assigned CALL**. Held and cooldown anchors are excluded from local randomization contrasts, while their downstream costs and outcomes remain included. Duration and hold use their own logged randomization probabilities. Fixed cohort assignments are not assigned an invented `1/9` propensity.

Clustering combines all anchors and three blocks within `(task, init)`, centers within task, and uses the frozen fixed-task sandwich and t degrees of freedom. Whole-pilot cohorts have 60 episodes but only **20 init clusters**, normally df=10. Intervals retain all 64 reserved primary slots; unsupported contrasts do not reduce multiplicity. The treatment is a CALL **package** with randomized duration/hold and subsequent factorial behavior, not an isolated shadow chunk or an estimate of the entire proposed controller's SR.

### Primary effects and simultaneous decisions

SR effects below are percentage points. The last column is the simultaneous 64-contrast interval. All eight contrasts per cell, pointwise intervals, costs, and support are in [primary64.csv](final_pilot/primary64.csv), [effects.csv](pilot_all/effects.csv), and [FINAL_TABLES.md §1](FINAL_TABLES.md).

| Cell | Overall factorial CALL effect | Simultaneous interval |
| --- | ---: | ---: |
| π0.5 LIBERO-10-50 | +0.54 | [−19.78, +20.86] |
| π0.5 LIBERO-10-500 | +2.57 | [−35.13, +40.27] |
| π0.5 Spatial-50 | +13.89 | [−23.17, +50.96] |
| π0.5 Spatial-500 | +2.71 | [−48.13, +53.55] |
| GR00T LIBERO-10-50 | +2.67 | [−19.02, +24.36] |
| GR00T LIBERO-10-500 | −4.15 | [−33.18, +24.88] |
| GR00T Spatial-50 | +15.46 | [−14.39, +45.31] |
| GR00T Spatial-500 | −17.05 | [−73.52, +39.43] |

There are **54 estimable intervals and 10 withheld intervals**; zero simultaneous exclusions of zero. Three pointwise findings must not be promoted to primary discoveries: GR00T Spatial-50 overall CALL `+15.46 [1.45,29.47]` pp; GR00T Spatial-500 coverage enrichment `+28.93 [2.74,55.12]`; π0.5 LIBERO-10-500 hold3−hold1 `+22.37 [1.36,43.39]`. Each fails its simultaneous test. Withheld intervals are not evidence of safety or no effect.

The frozen six-gate nomination rule requires **both** positive gate benefit and positive enrichment at the 48-slot selection confidence level, the prespecified assignment/ESS support, and df≥4, using calibration only (`init%5==0`). It returns **0/48 eligible nominations**. Forty-four gate entries have df=0 in both required contrasts; the four small-library G gates lack the positive stratum entirely. Pilot calibration has one init per task, so thousands of anchors cannot supply the missing init replication. I did not replace calibration by the whole pilot after seeing this limitation.

| Frozen gate | Outcome under the frozen rule | Whole-pilot secondary diagnostic, not a nomination |
| --- | --- | --- |
| Coverage-high: distance LOEO CDF≥2/3 | No nomination | Factorial enrichment changes sign across cells and fixed-dose continuations. **High means distant / poor coverage**, not good coverage. |
| Neighbour disagreement above A-calibration q90 | No nomination | Factorial enrichment positive in all four π0.5 cells, but all pointwise intervals include zero; GR00T directions vary. |
| Early matched phase <1/3 | No nomination | Factorial enrichment negative in six of eight cells. Its inverse was not an authorized search. |
| Non-advancing progress >0 | No nomination | Factorial enrichment changes sign; this broad feature is not the whole no-progress controller intervention. |
| Never previously called | No nomination | Factorial enrichment negative in all eight cells, with no positive selection evidence. |
| Six-test G, α=.20 | No nomination | Small-bank calibration resolution makes the gate vacuous. With n≤5/task, `6*p_min≥1`, exceeding .20. |

The full secondary tables explicitly test the signals against the **observed fixed-dose closed-loop continuations**. For example, GR00T LIBERO-10-500 coverage enrichment is +12.60 pp in factorial, but −6.13/−11.91/−0.89 pp under doses 1/8, 1/4, 1/2. GR00T Spatial-500 is a potentially interesting exception: coverage enrichment is +28.93/+56.71/+31.57/+11.75 pp across those four continuations. Its frozen primary interval nevertheless includes zero, and its calibration-only nomination fails. This is a continuation hypothesis, not permission to select that cell after looking.

Shadow disagreement and K4 cannot nominate an inference-free gate. The audit records 5,364 selected K4 anchors, but the frozen per-task A-calibration q90 construction provides insufficient positive-stratum support for `k4_high` enrichment in all eight cells. Missing reference/support is not zero spread. Shadow enrichment also varies by model and continuation. Library sample size, not additional evaluation episodes, controls the small-bank G-resolution problem. See [nomination48_audit.csv](final_pilot/nomination48_audit.csv), [calibration_resolution.csv](pilot_all/calibration_resolution.csv), [feature_references.json](pilot_all/feature_references.json), and [FINAL_TABLES.md §2](FINAL_TABLES.md).

The secondary SR/extra-MISS ratio is only reported when its future-MISS denominator has a positive lower bound. Overall factorial ratios are unstable on π0.5 LIBERO-10-50/500 and GR00T LIBERO-10-500. Even a positive pointwise ratio elsewhere does not establish the selected placement policy: numerator/denominator include later changed behavior, and the primary simultaneous evidence remains unresolved. Full vectors and ratios are exported, rather than replacing them with a single optimistic “SR per call” ranking.

### Timing, survival, commitment, and return to cache

Of 480 window episodes, **383 reached a trigger and 97 did not**. Ten triggered episodes terminated before executing their assigned delayed call; all ten remain in the trigger-conditional intention-to-treat contrast. No-trigger episodes remain in whole-cohort comparisons. All four Spatial cells lack a valid fixed-task interval for delay0−delay2 under the frozen support/variance rules. The other four do not show simultaneous superiority. No survivor-only or executed-call-only result is substituted. Source: [window_reach.csv](pilot_all/window_reach.csv).

The decisions actually reached are identical in all eight [method_screen.json](pilot_all/method_screen.json) entries:

* **No learned gate selected; no deployment validation.** The frozen fallback remains B plus coverage-high exploration for data collection only. Later harm evidence does not retrospectively change this preregistered output.
* Neither pre-guard nor at-guard benefit clears its simultaneous test. No causal claim about “before irreversible failure” follows from either operational label.
* No evidence-based move to shorter commitment, hold3, or intentional delay. Retain the design default `min(H,2R)` controls, hold1, then a fresh retrieval anchored to the actually executed tail; one-anchor cooldown. In this pilot that means 10 controls. This is an unresolved default, not an empirically established optimum.

For observed policy context, the pilot's SR@IR values are below. IR is recomputed from actual-control totals: `(c1*anchors+(1−c1)*MISSes)/(active_controls/5)`, with c1=.152/.148. These are **descriptive cohort outcomes**, not primary randomized placement effects. P10 can exceed .5 because an issued chunk may terminate before all ten controls execute. See [pilot_owner_IR.csv](final_pilot/pilot_owner_IR.csv) for per-request IR as well.

| Cell | A | B | P10 | Factorial |
| --- | ---: | ---: | ---: | ---: |
| π0.5 l10-50 | .717@.077 | .900@.171 | .950@.506 | .917@.318 |
| π0.5 l10-500 | .900@.077 | .950@.158 | .867@.507 | .967@.247 |
| π0.5 sp-50 | .733@.078 | .833@.173 | 1.000@.525 | .917@.338 |
| π0.5 sp-500 | .917@.079 | .950@.129 | 1.000@.522 | .983@.239 |
| GR00T l10-50 | .567@.075 | .617@.241 | .833@.506 | .717@.326 |
| GR00T l10-500 | .800@.075 | .867@.191 | .833@.506 | .867@.252 |
| GR00T sp-50 | .917@.076 | .933@.119 | .900@.517 | .933@.340 |
| GR00T sp-500 | .967@.077 | .917@.147 | .900@.517 | .967@.246 |

These small-init outcomes differ substantially from some 500-init historical outcomes. That cautions against treating the pilot's inits 0–1 as representative or selecting a controller from raw SR rankings.

### What the +31,680 continuation buys: measured-variance planning

This subsection is **a conditional precision forecast**, not additional experimental evidence. It replaces the planning variance .20 / ICC .3 with measured task-fixed components of **paired episode differences or linearized causal scores**, not the ICC of a terminal label duplicated across anchors. Code: [final_pilot.py](final_pilot.py); full forecasts: [measured_continuation_precision.csv](final_pilot/measured_continuation_precision.csv).

For the balanced three-block pilot, `sigma_init_raw=(MS_between−MS_within)/3`, `sigma_within=MS_within`. I retain raw ICC, including negative estimates. Conservative planning clips only `sigma_init_raw` to zero. At a new schedule with repeats r_i and N episode pairs, `Var(mean)=max(0,sigma_init_raw)*sum(r_i²)/N²+sigma_within/N`. The full schedule has 500 episodes/cohort/cell, **250 distinct task/init clusters**, and `sum(r_i²)=1200`; its fixed-task df is 240. This differs from the later 500-episode arms with 500 distinct task/init pairs. Local-effect forecasts use episode sums of linearized HT numerator/denominator contributions and retain zero contributions from unreached episodes; enrichment retains covariance between strata. They reproduce the 54 estimable pilot standard errors before projection.

| Cell | Measured B−A within-task variance | Raw ICC → planning ICC | Full B−A n_eff | Full B−A ±95%, pp | Full randomized CALL ±95% / simultaneous, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| π0.5 l10-50 | .2389 | .442 → .442 | 308.9 | 5.48 | 3.26 / 5.63 |
| π0.5 l10-500 | .1167 | .286 → .286 | 357.1 | 3.56 | 5.33 / 9.20 |
| π0.5 sp-50 | .1667 | .100 → .100 | 438.6 | 3.84 | 6.39 / 11.04 |
| π0.5 sp-500 | .0556 | −.200 → 0 | 500.0 | 2.27 | 8.77 / 15.15 |
| GR00T l10-50 | .2167 | .154 → .154 | 411.4 | 4.52 | 3.01 / 5.20 |
| GR00T l10-500 | .2333 | .214 → .214 | 384.6 | 4.85 | 4.13 / 7.14 |
| GR00T sp-50 | .0833 | −.200 → 0 | 500.0 | 2.79 | 6.63 / 11.45 |
| GR00T sp-500 | .0500 | ≈0 → 0 | 500.0 | 1.97 | 8.02 / 13.85 |

The full overall-CALL 80%-power pointwise MDE is **4.28–12.47 pp**. Full coverage-enrichment halfwidths are **6.49–36.95 pp pointwise, 11.22–63.83 pp simultaneous**. Full window-delay halfwidths remain **17.91–26.24 pp pointwise, 30.94–45.33 pp simultaneous**. These especially wide timing intervals reflect the measured variance of the frozen, uncentered HT estimator, as well as trigger support. I have not replaced it with a favorable post-hoc Hájek/DR estimate. A separately registered future estimator could use outcome-blind baseline centering or cross-fitted outcome regression to improve efficiency; that would not rewrite this result.

The reserved full validation split has 400 episodes/200 clusters; **continuation-only validation** has 370 episodes/190 clusters per cohort/cell. On the latter, projected B−A halfwidths are **2.29–6.34 pp**, overall-CALL simultaneous halfwidths **6.05–17.69 pp**, and coverage-enrichment simultaneous halfwidths **13.10–74.08 pp**. Full calibration has 100 episodes/50 clusters, so the structural df=0 barrier to nomination disappears, but support and 48-way selection still apply. No forecast guarantees a gate will qualify.

The measured B−P10 full 95% halfwidths are **1.97–5.73 pp**, already larger than the proposed one-pp retention margin; continuation-only validation gives **2.29–6.65 pp**. Using its measured planning variance, a *true-zero-gap*, pointwise one-sided 5%, 80%-power one-pp NI calculation needs **3,092–19,922 effective independent pairs per cell**, before candidate/cell multiplicity. These are planning illustrations using B's variance, not a sample-size claim for the untested candidate. Formula and cell counts: [NI_sample_size_measured.csv](final_pilot/NI_sample_size_measured.csv), `ceil((z_.95+z_.8)^2*variance/.01^2)`.

Thus the scheduled continuation is useful for broad call effects, replication under different doses, and possible nomination. It is unlikely to settle fine timing/enrichment universally or certify one-pp SR retention. Forecasts assume unchanged score variance, state visitation, and init covariance. They are noisy estimates from only 20 pilot clusters; negative ICCs are not evidence that repeats manufacture independent information. More evaluation episodes do not fix a five-episode library's G-calibration resolution, missing physical failure labels, or uncertified branch restoration.

## 2. Post-hoc / exploratory integration

### Sources and pairing

I read [PAPER_AB.md](../PAPER_AB.md), [ABLATIONS.md](../ABLATIONS.md), and the September 28/29 §10 ledger entries, then recomputed from each completed arm's `summary.json` and `client/journal.jsonl`. [final_exploratory.py](final_exploratory.py) audited **98 arms with 500 accepted episodes each**, verifying the exact same ten tasks ×50 inits, summary success counts, duplicate acceptance consistency, and ledger ratios. This is 49,000 recorded outcomes, **not 49,000 independent initializations**. All source paths/hashes are in [arm_audit.csv](final_exploratory/arm_audit.csv).

Owner IR is recomputed, not copied from a generic summary IR: `.152*v+.848*m` for π0.5 and `.148*v+.852*m` for GR00T. For references serving ten controls per request, I additionally normalize to nominal five-control units using `cost_ledger.controls`, giving P10 IR=.5. These legacy ledgers lack actual terminal partial-control totals, so they are not silently compared numerically with the pilot's actual-control denominator. Three-run A/B IR values below are means of the three ledger-derived arm ratios.

Each individual arm pair has an exact paired McNemar test. Comparisons to the mean of three references **first average their outcomes within task/init**, then form a task-stratified 500-init interval. Reusing one candidate against three references does not produce 1,500 independent candidate episodes. The paper's pooled discordance counts are reproduced, but its pooled McNemar p-values are not treated as independent-init inference here. All intervals in this section are exploratory pointwise intervals, not the frozen 64-test family. Sources: [paired_contrasts.csv](final_exploratory/paired_contrasts.csv), [FINAL_TABLES.md §4](FINAL_TABLES.md).

### B's gains and the no-progress reversal

| Cell | A mean SR@IR | B mean SR@IR | B−A, pp [95%] |
| --- | ---: | ---: | ---: |
| π0.5 l10-50 | .714@.076 | .817@.181 | +10.27 [7.28,13.25] |
| π0.5 l10-500 | .827@.077 | .873@.159 | +4.60 [2.32,6.88] |
| π0.5 sp-50 | .837@.078 | .910@.140 | +7.33 [5.54,9.12] |
| π0.5 sp-500 | .976@.078 | .982@.119 | +0.60 [−0.31,1.51] |
| GR00T l10-50 | .609@.074 | .719@.221 | +10.93 [7.35,14.51] |
| GR00T l10-500 | .830@.074 | .867@.187 | +3.67 [0.55,6.78] |
| GR00T sp-50 | .867@.075 | .875@.147 | +0.80 [−1.48,3.08] |
| GR00T sp-500 | .964@.076 | .959@.127 | −0.53 [−1.91,0.84] |

| Remove no-progress from B | SR@IR | Change vs B mean, pp [95%] | Change vs A mean, pp |
| --- | ---: | ---: | ---: |
| π0.5 l10-50 | .704@.104 | −11.27 [−14.60,−7.93] | −1.00 |
| GR00T l10-50 | .620@.096 | −9.87 [−13.45,−6.28] | +1.07 |
| π0.5 sp-50 | .858@.112 | −5.20 [−7.06,−3.34] | +2.13 |
| GR00T sp-50 | .906@.105 | **+3.07 [1.25,4.88]** | **+3.87** |

The GR00T Spatial-50 removal beats each B replicate, exact p=.00522/.00372/.00936, and lowers ledger IR from mean .14714 to .10529. This is the concrete counterexample to a universal raw no-progress trigger. The other three guards are individually removable without a detected paired SR change at .05, but non-rejection does **not** establish one-pp NI, nor their joint removability. I do not remove all three in the recommendation.

Crucially, [p2_ablations/judge.py](../p2_ablations/judge.py) masks the no-progress bit **and changes `blind_step` back to the base behavior when that guard is disabled**. It retains diagnostic histories. The observed intervention therefore removes both no-progress MISSes and its early blind-LOOK veto. Its effect can arise from policy takeover, changed observation cadence, or their interaction. The factorial contrast intervenes at free roots under a different future controller; its GR00T Spatial-50 point estimate cannot contradict or explain this whole-controller ablation by itself. A single local CALL can be helpful while the controller repeatedly calling/reobserving on that signal is harmful.

### Additional calls help some weak cells; risk allocation has not beaten dose control

I froze the exploratory inventory at **18 complete frontier arms**. Twenty-five specs without a completed 500-episode summary at that read were excluded and listed in [inventory.json](final_exploratory/inventory.json); I did not poll them. Exact configurations come from `ideation_Q2/frontier/adapters/emit_arms_all43.json`; every audited result and paired P10 comparison is in [frontier.csv](final_exploratory/frontier.csv).

Selected exact ledger-derived values:

| Cell | Configurations, SR@IR | Matched P10 comparison |
| --- | --- | --- |
| π0.5 l10-50 | B+.25 .842@.260; B+.5 .880@.339; B+.75 .874@.423; risk ρ=.35 .862@.342 | P10 .904@.5; B+.5 Δ=−2.4 pp [−5.75,+0.95], p=.213 |
| GR00T l10-50 | B+.5 .808@.358; B+.75 .850@.430; risk ρ=.35 .832@.344 | P10 .866@.5; B+.75 Δ=−1.6 pp [−5.11,+1.91], p=.434 |
| π0.5 l10-500 | B+.125 .842@.206; B+.25 .882@.242; risk ρ=.24 .884@.240 | P10 .904@.5; risk Δ=−2.0 pp [−5.50,+1.50], p=.332 |
| π0.5 sp-50 | B+.25 .922@.228; B+.5 .962@.315 | P10 .986@.5; B+.5 Δ=−2.4 pp [−4.39,−0.41], p=.0290 |
| GR00T sp-50 | B+.125 .878@.192; B+.25 .902@.234 | P10 .938@.5; B+.25 Δ=−3.6 pp [−6.78,−0.42], p=.0356 |

The risk lotteries are useful **budget-allocation comparators**, not established placement improvements. The adapter selects an episode dose from library-derived task risk at reset, then uses dose coins on A; it is not a per-anchor risk trigger on B ([methods.py:RiskLottery](../ideation_Q2/frontier/adapters/methods.py), [prepare_adapters.py](../ideation_Q2/frontier/adapters/prepare_adapters.py)). Consequently comparisons to B+extra-dose also change the underlying guard controller and cannot isolate within-trajectory timing. At similar IR, riskρ=.35 minus B+.5 is −1.8 pp [−5.66,+2.06] on π0.5 l10-50 and +2.4 pp [−2.00,+6.80] on GR00T l10-50. On π0.5 l10-500, riskρ=.24 minus B+.25 is +0.2 pp [−3.27,+3.67], at IR .24017 versus .24154. No matched risk-versus-uniform allocation result here establishes a timing advantage. On GR00T sp-50, B+.25 is −0.4 pp [−3.43,+2.63] versus no-progress removal while using **.23399 versus .10529 IR**. Spending more calls without repairing the harmful trigger is not the demonstrated efficient direction there.

The frontier narrows some gaps to P10, but `p>.05` is not SR retention. None of the quoted near-P10 comparisons certifies the frozen one-pp NI requirement. A universal success-at-B-budget claim remains unsupported, especially on the long-task sparse libraries.

### Visual representation and checks of tempting library proxies

Direct-token PCA raises GR00T Spatial-50 A from mean .867 to **.910**, difference +4.27 pp [1.02,7.52], at IR .07532. It lowers π0.5 Spatial-500 by −3.00 pp [−4.91,−1.09] and GR00T Spatial-500 by −3.60 pp [−6.08,−1.12]. Other pointwise intervals include zero. These recomputed results agree with the direction in [ABLATIONS.md §2](../ABLATIONS.md). They motivate investigating retrieval/phase ambiguity; they do not prove that ambiguity caused the no-progress harm. One cannot add the PCA gain to the guard-removal gain, or switch PCA by benchmark identity and call that calibration. A changed representation needs fresh LOEO calibration; owner inference IR also omits its CPU projection overhead.

I tested readily available library proxies against the four completed no-progress ablations before trusting them. Per-task successful-library LOEO records are from `p3_profiling/calibration_v2/<cell>.json`; rows below average tasks equally. The last column is the benefit of **keeping** no-progress, B minus its removal.

| Cell | Fraction of successful library episodes with NP alert | Mean LOEO chunk RMS | Mean maximum progress statistic | Benefit of keeping NP, pp |
| --- | ---: | ---: | ---: | ---: |
| π0.5 l10-50 | .900 | .1221 | 2.96 | +11.27 |
| GR00T l10-50 | 1.000 | .1343 | 3.44 | +9.87 |
| π0.5 sp-50 | .305 | .1659 | .69 | +5.20 |
| GR00T sp-50 | .380 | .1367 | .88 | −3.07 |

At four-cell level, alert fraction and maximum progress statistic each have Spearman .60, exact permutation p=.417; mean chunk RMS has −.80, p=.333. These are descriptive checks on four banks, not a trained benefit predictor. In particular, “disable the guard where it fires often on successful demonstrations” would most strongly disable the **beneficial** long-task guard. The mean imitation-error ordering also does not identify utility. Sources: [library_proxy_cell.csv](final_exploratory/library_proxy_cell.csv), [library_proxy_task.csv](final_exploratory/library_proxy_task.csv), [library_proxy_correlations.json](final_exploratory/library_proxy_correlations.json).

Two more post-hoc checks use current no-progress-flagged B anchors in the pilot. GR00T l10-50 has 802 such anchors: only **9.98%** have the vision-confirmed still statistic >0, versus **10.00%** of the 40 GR00T sp-50 anchors. A “must also be physically still” filter would discard about 90% in both. The existing coverage/action-error calibration p≤.20 retains **53.24%** of flagged GR00T l10-50 anchors, **39.89%** of π0.5 l10-50, and **77.50%** of harmful-cell GR00T sp-50. It also fails as an obvious sign separator. These are anchor frequencies, not probabilities of benefit. Local randomized NP-stratum effects have broad or unavailable intervals; no new classifier is nominated from them. Sources: [POSTHOC_guard_state.csv](final_pilot/POSTHOC_guard_state.csv), [POSTHOC_guard_call_effects.csv](final_pilot/POSTHOC_guard_call_effects.csv).

## 3. Recommendation: calibrate the reliability of progress before using it to trigger a call

### What is and is not recommended

**Do not claim a learned transferable MISS placement rule from this pilot.** Preserve the frozen “no nomination” result. For the next method experiment, replace the raw no-progress package with the following **post-hoc, prospective hypothesis**: a MISS is justified by a library-calibrated upper bound on recent progress that is unusually low; an uncertain estimate of progress should first request a fresh observation. This separates “we can tell that progress is slow” from “the nearest retrieved frame keeps moving between incompatible trajectory phases.” Neither is equivalent to the robot being physically stationary.

This is designed to keep useful long-task rescue while suppressing noisy triggers, **but no computed result here shows that it achieves that sign split**. Successful demonstrations alone cannot identify the causal benefit of switching to a policy in failed states. The calibration below estimates progress-estimation error and normal progress scale, not SR or call value. Its only route to acceptance is the closed-loop test below. No rule based on model/suite names is allowed. Until that test, the known GR00T sp-50 removal result is a benchmark-specific diagnostic/oracle control, not a portable method.

### Exact candidate algorithm for the next registration (not implemented or evaluated)

All constants below are declared design choices, not fitted to pilot success labels. The recipe reads action/state dimensions, valid channels, camera count, H, control period, and R from the library/adapter manifest. It uses A's fixed visual representation and learned retrieval metric. No object name, simulator predicate, gripper sign, LIBERO horizon, raw displacement cutoff, or benchmark-specific feature threshold is introduced.

1. **Library windows.** For each task bank, resample successful library observations at intervals `L=min(H,2R)` controls, selecting the latest available observation at or before each grid timestamp, removing repeated selections, and preserving the final observation and original control timestamps. Let `T_med` be the median library episode length in L-control intervals. Set `W=max(2,ceil(.05*T_med))`. Online use the latest W+1 distinct vision observations with their executed-control timestamps; before they exist, this component is inactive. Let S be the actual control span of the window. Express every estimated/true phase advance below as advance multiplied by `W*L/S`, so an extra LOOK does not make a shorter interval automatically count as slower progress. S must be positive. Normalized phase is original control index divided by that episode's final control index.

2. **Estimate progress using an episode-consistent alignment.** For every candidate library episode separately, align the observed W+1 keys to a nondecreasing sequence of its row indices, minimizing the sum of A-metric distances. Repeated indices and forward skips are allowed; ties choose the lexicographically earliest sequence. This dynamic program uses no mandatory forward motion. For a bank of E episodes use `K=min(5,E−1)` both online and in LOEO, taking the best K templates by mean alignment distance with equal template weight (episode identifier breaks template ties). Let `delta_hat` be the median end-minus-start normalized phase after the span adjustment in step 1, `phase_hat` the median start phase, `distance_hat` the median mean alignment distance, and `spread_hat` the empirical q90−q10 of adjusted template advances. The empirical quantile throughout is `inf{x:F(x)>=q}`. Missing keys or no valid window disable this component.

3. **LOEO calibration of estimation error and normal advance.** Replay every eligible window of each successful library episode against all *other* episodes of the same task, using exactly steps 1–2. Store its three-dimensional context `(phase_hat, distance_hat, spread_hat)`, known true normalized advance `delta_true`, and residual `delta_true−delta_hat`. Convert distance and spread to their episode-equal LOEO empirical CDFs; phase already lies in [0,1]. At an online context, select the nearest calibration window in L1 distance **within each calibration episode** (earliest window breaks ties), giving each episode one reference. Set `e90=q90(delta_true−delta_hat)` and `a10=q10(delta_true)` over these references. Require at least four contributing calibration episodes and finite references; otherwise this component is unavailable. No cross-task pooling or outcome-based threshold adjustment. This is an empirical conditional calibration, **not a conformal coverage theorem**; fixed-fit metric/PCA reuse and shifted cache states preclude that claim. Very small libraries will produce coarse, possibly conservative references.

4. **Trigger and observation decision.** At a free vision anchor, define `slow_confirmed = (delta_hat + e90 < a10)` and `slow_ambiguous = (delta_hat < a10 <= delta_hat + e90)`. A confirmed slow window is eligible for a MISS. For an ambiguous window, serve cache, request one extra LOOK after `min(R,L)` controls rather than waiting L, then recompute with the span adjustment in step 1. Permit at most one such extra LOOK per W*L executed controls; this is a vision cost, not a free measurement (when L≤R it advances no observation). Continue cache if it remains ambiguous; do not call solely because alignment is uncertain. All valid cache decisions remain bounded by their normal action horizon. This **replaces** the old no-progress MISS bit and its blind LOOK veto; it is not an additional AND on B's hand-set three-transition flag. The three other B guard mechanisms remain the fixed control condition in this Q3 experiment.

5. **Call budget and handback.** At a confirmed slow window not in cooldown, take an independent keyed Bernoulli(q) call. Commit `min(H,2R)` controls, hold1; after the actually executed tail, acquire fresh vision and retrieve using that tail for continuity. Apply one free-anchor cooldown to this new component; other B guards retain their existing behavior and take priority. No stale cached continuation is reused after takeover. Calibration cost target is B's owner IR on exactly **three recorded B trajectories per task**, selected in advance by recording index without outcome selection. Replay q∈{0,1/16,…,1}, averaging seeds 0..255; choose the largest q whose modeled IR, including extra LOOKs, is no greater than B's on those same streams. If no positive q is feasible, or the library reference is unavailable, label the candidate unavailable at that budget rather than inventing a fallback success claim. Replay chooses an initial budget only; it is not a causal rollout or assurance about realized IR.

The proposed online signal uses retrieved keys, trajectory indices, and executed-control history; it does not require shadow policy/K4 at deployment. Alignment incurs extra CPU retrieval/DP work, approximately proportional to the number of template rows times W per anchor; this has **not been implemented or timed**. Report its wall time separately from owner inference IR. The fixed B guard controls in the LIBERO test are not a claim that every B constant is portable: on another robot, declare its protocol fields and use the same independently frozen library-calibration recipe for the remaining guards in *both* B and candidate arms, following [ideation_G/REPORT.md](../ideation_G/REPORT.md). Do not silently pool small G references to make them fire. The Q3 replacement itself uses only the library recipe above and three recorded trajectories.

The rationale is falsifiable: agreeing, calibrated trajectory alignments could retain rescue during task-level stalls even while the robot moves; large alignment error could suppress phase-aliasing calls. The PCA and no-progress counterexamples motivate that possibility but do not measure it. Our checks specifically reject substituting raw library alert frequency, physical stillness, or the current error-risk threshold as if one of them had already solved the problem.

### At most eight validation configurations

Freeze code and calibration artifacts before new outcomes. Keep the same representation, task library, seeds, and init set in all eight arms; pair against **A, B, and both pure-inference commitment lengths**. This is a new experiment, not an amendment to the completed pilot nomination rule.

| # | Exact configuration | Paired question / prediction to test |
| --- | --- | --- |
| 1 | A, existing cache and L commitment | Does any rescue package improve SR enough to justify its calls? |
| 2 | B, all four existing guard packages | Practical rescue baseline and calibration budget. |
| 3 | Pure policy, commitment `min(H,2R)` | P10-equivalent retention reference; same commitment as candidate. |
| 4 | Pure policy, commitment `min(H,R)` | Native-inference retention reference. |
| 5 | B with no-progress MISS bit **and** NP blind-LOOK veto disabled; diagnostic histories retained | Replicate the observed sign reversal and define the low-call control. |
| 6 | B with only no-progress MISS bit disabled; keep its blind-LOOK veto and diagnostic histories | Is harm/benefit due to policy takeover or observation cadence? Direction unresolved; this is a mechanism test. |
| 7 | Replace B's entire NP package by steps 1–5 above, frozen q; other three B guards unchanged | Hypothesis: preserve B's benefit on both l10-50 cells while recovering some harmful-cell loss at no higher IR. No numerical gain is claimed. |
| 8 | Arm 5 + an independent uniform call coin at every free eligible anchor, commitment L, hold1, cooldown1; q_uniform from the same grid/256-seed replay, chosen for closest planned IR to arm7, ties choose lower q | Does arm7 buy more SR than untargeted calls at comparable realized cost? Other guards take priority; no extra ambiguity LOOKs. |

Primary practical acceptance is lower one-sided 95% SR differences of arm7 versus B, pure-L, and pure-R all above −.01, with upper one-sided 95% IR(arm7)−IR(B)≤0 on **both mean episode IR and pooled actual-control IR**. For a global eight-cell claim, reserve eight joint acceptance tests and use Holm across cells; each joint test is the intersection of its SR/cost requirements. Claim timing superiority only if arm7−arm8 has positive SR evidence and does not exceed its realized IR, with Holm across the eight cell-specific timing tests. A realized cost mismatch is reported as a frontier comparison, not called equal-budget superiority. Estimate uncertainty by paired init clusters across all repeated seeds, with fixed task strata; collection shadow costs are separate.

The key falsification target is stronger than “not significant”: arm7 must preserve B on the beneficial long-task cells and improve its inefficient GR00T sp-50 behavior without higher IR. A convincing >1 pp loss versus B, higher IR without the required SR retention, no advantage over uniform dosing, or alignment calibration that rejects nearly every beneficial long-task trigger falsifies the proposal or its budget premise. If arm6 explains the reversal while arm7's alignment gate adds nothing, the next correction belongs in observation cadence rather than a new MISS score. If arm5 remains as good as arm7 in the harmful cell, the extra alignment machinery needs independent justification. Failure to transfer with unchanged recipe on a new benchmark/robot falsifies the portability claim.

At B's IR, even a better gate may not reach pure-inference SR on weak long-task banks; current frontier evidence makes this a serious open constraint. If retention fails, do not call the method successful because it beats B or because `p>.05` versus inference. That outcome requires a larger compute budget or a different cache, with a new prospective comparison.

Use fresh init/scene diversity and independent library realizations, then another benchmark/robot. The pilot continuation's inits 2–24 are fresh relative to the pilot, but **not an untouched validation set for this post-hoc proposal**: the 500-episode A/B, ablation, and frontier arms already cover inits 0–49. Its preregistered continuation results remain valuable for the original frozen tests. The new candidate needs newly held-out episodes/conditions for a clean confirmatory test; where a benchmark cannot supply them, state the conditional repeated-init scope explicitly. Determine arm7's required count from its own blinded paired variance rather than treating the 3,092–19,922 B−P10 planning range as its measured requirement.

## 4. Reproduction, artifacts, and limits

Commands below were actually run from `/home/weiland/projects/openpi`; all Python used CPUs `22-25,66-69`, one BLAS/OpenMP thread and no GPU. The analysis programs preserve existing output directories by refusing to overwrite them; for a rerun, retain this delivery and use fresh output locations/a copied analysis tree inside this directory. No source, run-root, calibration, or profiler file was edited.

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q3/.mplconfig .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/analyze_pilot.py \
  --tables \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/groot_l10_50 \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/groot_l10_500 \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/groot_sp_50 \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/groot_sp_500 \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/pi05_l10_50 \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/pi05_l10_500 \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/pi05_sp_50 \
  /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/pi05_sp_500 \
  --calibration-dir /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/calibration \
  --stage pilot --out exp/offline_search/rounds/r06/ideation_Q3/pilot_all

taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q3/.mplconfig .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/final_pilot.py

taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q3/.mplconfig .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/final_exploratory.py

taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q3/.mplconfig .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/build_final_tables.py
```

`final_pilot.py` adds measured-variance planning and **explicitly named POSTHOC** guard diagnostics; it does not alter the frozen estimator. `final_exploratory.py` checks summary/journal pairing, recomputes owner IR, and records the completed-frontier snapshot. `build_final_tables.py` checks unchanged frozen hashes and reconstructs the frozen standard errors before rendering tables and the figure. The completed checks passed; no new serving controller or profiler was built.

Remaining limitations relevant to Q3: physical before/after-failure labels and recovery deadlines are not identified by a nearest-demo phase; logged simulator snapshots are not evidence of certified restores; shadow disagreement is paid inference and not a counterfactual successful action; successful-library calibration does not identify policy benefit off its visitation distribution; and seed repetition cannot replace init/library/benchmark diversity. The pilot resolves none of those by itself. They limit what can be concluded, rather than changing the recorded “no nomination” outcome.
