# Q2 final: how much policy inference to buy

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


**Frozen pilot decision: inconclusive in all eight cells; retain P10 as the SR reference. No risk, uniform, fixed-dose or B controller passes the preregistered promotion rule.** The pilot measures useful dose/cost curves but has only one reserved init per task. Post-hoc full evaluations narrow several SR gaps, but none of the 18 completed frontier-completion arms establishes 2 pp noninferiority to either native pure reference under the conservative paired procedure used here. The owner-facing knob remains **target owner IR ρ**, with uniform allocation as the unvalidated allocator's default; the current library-risk score is a challenger, not an SR certificate.

## 1. Preregistered all-cell result

### Provenance and unchanged analysis

Executed on CPUs **2–5,46–49**, CPU only, one BLAS/OMP thread. The joint input contains exactly the eight canonical `r06_p3_pilot/tables/<cell>` directories, linked read-only into `final_analysis/input_tables/`. The sibling `groot_l10_50.failed1` is excluded. The coordinator's repaired canonical `groot_l10_50` tables supply the accepted window-r2 attempt; nothing is read as an extra trial from `repair_archive/` or the obsolete repair paths in `ATTEMPT_REUSE.md`. No smoke outcomes, new experiment, server, worker, chain, remote host or git operation was used.

The unchanged `pilot_q2.py` completed with **216 arms, 4,320 accepted episodes, 180,155 decisions and 92,209 vision anchors**. Episode/decision/anchor cost totals, accepted attempts, source/propensity consistency, commitment lengths, paired environment seeds and complete campaign support passed in all cells. Upstream physical-control and snapshot presence audits were reused; simulator restore fidelity was not tested. `final_analysis/derived/verification.json` verifies the locked SHA256 of `PREREG.md`, `pilot_q2.py` and `library_quality.json` and exact equality of every joint cell result with the coordinator's corresponding `pilot_<cell>/estimates.json`. Only the owned shell wrapper's CPU mask changed.

The frozen fallback library risk was used; no later Q1 score was substituted. Equal task weights, equal distinct-init weights within task, and averages over repeated seed blocks within init are unchanged. Fixed cohorts have assignment probability 1; dose_mix has logged episode-dose probability .2. Its Horvitz–Thompson estimates keep unassigned episodes in the denominator. Anchors are not terminal-outcome replicates. All curves use 4,000 shared, within-task init-cluster bootstrap draws, seed 26092802 plus the script's fixed context salt.

Primary pilot IR divides `c1 V + (1−c1) M` by **actual active controls/5**. `IR_request` is also exported. Shadows and K4 resamples are excluded. Hence pilot P10 has IR slightly above .5 when terminal requests are partial; the later native frontier uses the requested owner ledger convention, with native L10=.5 and L5=1. These are explicitly different denominators, not measured runtime speedups. P10 is the contemporaneous profiler controller; it is not interchangeable with a native L10 result.

### Dose curves and selection

Each entry below is SR @ primary pilot IR, over 60 episodes / 20 init clusters per cohort and cell. These are **all-init descriptive** estimates; full intervals, realized M/V, M/N, calls per episode and request IR are in [estimates.csv](final_analysis/preregistered/estimates.csv), with eight curve figures in that directory.


| Cell | A (d=0) | d=1/8 | d=1/4 | d=1/2 | P10 | B |
|---|---|---|---|---|---|---|
| π0.5 L10 50 | 0.717 @ 0.077 | 0.850 @ 0.131 | 0.900 @ 0.190 | 0.867 @ 0.291 | 0.950 @ 0.506 | 0.900 @ 0.171 |
| π0.5 L10 500 | 0.900 @ 0.077 | 0.917 @ 0.131 | 0.950 @ 0.189 | 0.933 @ 0.295 | 0.867 @ 0.507 | 0.950 @ 0.158 |
| π0.5 Spatial 50 | 0.733 @ 0.078 | 0.800 @ 0.129 | 0.833 @ 0.190 | 0.933 @ 0.301 | 1.000 @ 0.524 | 0.833 @ 0.173 |
| π0.5 Spatial 500 | 0.917 @ 0.079 | 0.933 @ 0.131 | 0.917 @ 0.189 | 0.983 @ 0.304 | 1.000 @ 0.522 | 0.950 @ 0.129 |
| GR00T L10 50 | 0.567 @ 0.075 | 0.600 @ 0.128 | 0.700 @ 0.187 | 0.700 @ 0.291 | 0.833 @ 0.506 | 0.617 @ 0.241 |
| GR00T L10 500 | 0.800 @ 0.075 | 0.850 @ 0.130 | 0.850 @ 0.187 | 0.867 @ 0.293 | 0.833 @ 0.506 | 0.867 @ 0.191 |
| GR00T Spatial 50 | 0.917 @ 0.076 | 0.950 @ 0.131 | 0.917 @ 0.186 | 0.917 @ 0.295 | 0.900 @ 0.517 | 0.933 @ 0.119 |
| GR00T Spatial 500 | 0.967 @ 0.077 | 0.933 @ 0.129 | 0.950 @ 0.185 | 0.917 @ 0.300 | 0.900 @ 0.517 | 0.917 @ 0.147 |


The selection family stays **296 one-sided tests**, family α=.05, α/test=.000168918919, SR margin ε=.02. A candidate needs a simultaneous lower SR difference above −.02 versus P10; a non-B candidate also needs that bound versus B and a strictly negative upper cost difference versus B. Risk additionally needs held-out SR ≥ uniform and IR ≤ uniform at the same ρ. The calibration split is init%5=0, and validation is its complement.

**All eight validation recommendations are `P10_reference_inconclusive`, with zero eligible candidates.** Validation has 10 distinct init clusters, one per task, so its within-task population variance and all required simultaneous bounds are null. All-init comparisons have 20 clusters and 10 variance degrees of freedom; none is substituted for validation. The following all-init B–P10 intervals show the scale of uncertainty. The last column is the simultaneous lower bound, not a validation certificate.


| Cell | B−P10 pp | Descriptive paired 95% interval, pp | Family-adjusted lower, pp |
|---|---|---|---|
| π0.5 L10 50 | -5.00 | [-13.33, +3.33] | -36.97 |
| π0.5 L10 500 | +8.33 | [+3.33, +13.33] | -11.49 |
| π0.5 Spatial 50 | -16.67 | [-26.67, -6.67] | -56.32 |
| π0.5 Spatial 500 | -5.00 | [-8.33, -1.67] | -20.36 |
| GR00T L10 50 | -21.67 | [-31.67, -11.67] | -60.31 |
| GR00T L10 500 | +3.33 | [-8.33, +15.00] | -43.58 |
| GR00T Spatial 50 | +3.33 | [-3.33, +10.00] | -27.38 |
| GR00T Spatial 500 | +1.67 | [-5.00, +6.67] | -21.79 |


No survival model was preregistered for Q2. Terminal failures stay in the denominator; there is no survivor-only filtering, time-to-success substitution or later stopping rule. “Retain P10” is a statistical reference decision, not an instruction to change a running deployment.

One limitation of the frozen choice rule is now practically important: **it forbids a non-B candidate that costs more than B even when B misses pure SR.** Thus it cannot select several relevant frontier-completion budgets. This restriction is preserved above; the prospective recommendation below explicitly changes it rather than reinterpreting the preregistration.

### Marginal returns, task budgets and allocation

No monotonic or concave fit was imposed. For π0.5 L10-50, adjacent ΔSR/ΔIR is **2.456, .854, −.330, .386**; for GR00T L10-50 it is **.624, 1.705, 0, .621**. The curves do not support one universal diminishing-return law. Of 48 cell-level adjacent-slope-change checks (24 dose triples × two cost denominators), only four descriptive intervals exclude zero in the diminishing direction: the same two triples under both MISS-share and IR normalization. GR00T L10-50's 1/8→1/4 versus 1/4→1/2 IR-slope change is −1.705 [−2.806, −.632]; π0.5 Spatial-500's 1/4→1/2 versus 1/2→1 is −.505 [−.915, −.119]. The other 44 are unresolved. These secondary checks are not multiplicity-adjusted promotion evidence. [Slopes and intervals](final_analysis/derived/pilot_marginal_returns.csv) include negative returns without clipping.

The frozen gap-fraction denominator passes all its bootstrap support checks only in GR00T L10-50, π0.5 Spatial-50 and π0.5 Spatial-500. Elsewhere the fraction of the A→P10 gap recovered is left undefined, including cases where A exceeds P10 or some resampled gaps are zero/nonpositive. This is stricter than a positive point gap.

Realized per-task heterogeneity is large, but each task has just six episodes and two init clusters. The ranges below are measurements, **not fitted task-specific minimum budgets**. All 80 task-level gaps/counts and candidate curves remain in the frozen outputs.


| Cell | Task A→P10 gap range, pp | B calls/episode across tasks | Tasks with nonpositive point gap |
|---|---|---|---|
| π0.5 L10 50 | -16.67 to +66.67 | 2.33–15.33 | 3/10 |
| π0.5 L10 500 | -50.00 to +50.00 | 2.00–12.00 | 8/10 |
| π0.5 Spatial 50 | +0.00 to +100.00 | 0.50–6.83 | 5/10 |
| π0.5 Spatial 500 | +0.00 to +83.33 | 0.33–4.00 | 9/10 |
| GR00T L10 50 | -66.67 to +100.00 | 3.83–24.17 | 5/10 |
| GR00T L10 500 | -33.33 to +50.00 | 2.17–13.83 | 6/10 |
| GR00T Spatial 50 | -33.33 to +16.67 | 0.17–4.17 | 8/10 |
| GR00T Spatial 500 | -33.33 to +16.67 | 0.33–3.50 | 9/10 |


At the frozen ρ grid {.10,.15,.20,.25}, risk allocation point-dominates uniform on **only 3/32 held-out cell/budget comparisons**: GR00T Spatial-500 at .15, π0.5 Spatial-50 at .20, and π0.5 Spatial-500 at .10. None has estimable held-out uncertainty or passes the complete rule. For example, GR00T L10-50 risk at .25 gains 4.85 pp over uniform but costs .0043 extra IR, so it fails the frozen dominance preference. The all-init library IR prediction error across the 64 allocation checkpoints ranges from −.003665 to +.010046; predicting spending is substantially easier here than certifying SR.

The fallback risk's within-cell all-init task Spearman correlations with the A→P10 gap range .067–.716 (10 tasks per correlation); they are descriptive, not an outcome-fitted universal quality map. Positive correlation with a gap does not identify positive marginal benefit from calling the policy. Neither the weak Q2 score nor a later Q1 quality score can be promoted without its frozen risk-versus-uniform validation. In dose_mix, 144/400 all-init task×dose combinations have zero assigned support. The HT outputs are retained separately, never pooled into the fixed-dose curves; realized assignment counts are not new independent terminal trials. [Risk comparisons](final_analysis/derived/risk_vs_uniform.csv), [quality associations](final_analysis/preregistered/quality_associations.csv), [dose_mix](final_analysis/preregistered/dose_mix.csv).

### What +31,680 episodes would buy: measured precision sensitivity

This is a **post-pilot planning extension**, not an alteration of frozen selection. Calibration-only ICC is unidentifiable: there is only one calibration init per task. It would be misleading to replace the planning ICC with a supposedly calibration-only measured estimate. Instead, the following transparent sensitivity uses **all pilot inits**, with no resulting retuning or promotion. `derived/measured_precision.csv` gives every candidate−P10, candidate−B and risk−uniform contrast, and also the identifiable calibration-only within-init variance.

For paired per-episode difference D, compute task-fixed-effect between-init and within-init mean squares using `p3_profiling.pilot_stats.icc` (10 and 40 df). Because the pilot has three repeats per init, estimate marginal paired variance as `(MS_between + 2 MS_within)/3`, and ICC as `(MS_between−MS_within)/(MS_between+2 MS_within)`. Report signed ICC; use max(0,ICC) for projection so negative estimates do not manufacture extra effective samples. A zero observed variance is explicitly unresolved, not perfect equivalence.

For K equally weighted init clusters with repeat counts r_i, use `n_eff = K² / sum_i[(1+(r_i−1)ICC_used)/r_i]`. Full cumulative data have 500 episodes / 250 init clusters per cohort/cell; reserved validation has 400 episodes / 200 clusters (80×3 repeats, 40×2, 80×1). The table is for **B−P10**, not unpaired Bernoulli SR variance. Half-widths and MDEs use 1.959964 and 2.801585 times the projected SE and are unadjusted normal approximations. Exact effective-count and episode-weighted sensitivities are in the CSV.


| Cell | Paired variance | ICC (signed) | Full all-data ±95%, pp | Validation n_eff | Validation ±95%, pp | Validation 80% MDE, pp | Effective n for 2 pp NI, 296 tests |
|---|---|---|---|---|---|---|---|
| π0.5 L10 50 | 0.117 | 0.429 | 3.76 | 253.0 | 4.21 | 6.02 | 5714 |
| π0.5 L10 500 | 0.128 | -0.174 | 3.53 | 315.8 | 3.94 | 5.64 | 6258 |
| π0.5 Spatial 50 | 0.144 | 0.654 | 4.40 | 229.1 | 4.92 | 7.04 | 7075 |
| π0.5 Spatial 500 | 0.050 | -0.000 | 2.21 | 315.8 | 2.47 | 3.53 | 2449 |
| GR00T L10 50 | 0.261 | 0.106 | 5.19 | 297.5 | 5.81 | 8.30 | 12788 |
| GR00T L10 500 | 0.322 | 0.224 | 5.95 | 279.5 | 6.65 | 9.51 | 15781 |
| GR00T Spatial 50 | 0.133 | 0.250 | 3.85 | 275.9 | 4.31 | 6.16 | 6530 |
| GR00T Spatial 500 | 0.117 | -0.000 | 3.37 | 315.8 | 3.77 | 5.38 | 5714 |


The +31,680 continuation raises the total to 36,000 episodes, not 36,000 independent inits. It buys well-resolved cost curves, more useful dose contrasts and estimable held-out variance. For B−P10 the projected held-out 80% MDE is **3.53–9.51 pp**, and even at a true zero SR loss the approximate probability of passing the frozen family-adjusted 2 pp gate is only **0.11–1.96%** with this design. The 2 pp NI counts above assume true zero loss and 80% power; a truly greater-than-2 pp loss cannot be cured by more evaluation episodes. These predictions are uncertain: ICC has only 10 between-init df, task variances need not be homogeneous, and two inits need not represent the remaining stock pool. CSV columns also project ICC=0 and ICC=1; no precision guarantee is claimed.

The matched expected-mixture contrast can be much more precise: for risk−uniform at ρ=.20, projected unadjusted held-out 80% MDE ranges **.19–2.24 pp**. That is precision for the known mixtures of the five fixed controllers using shared outcomes; a separately executed lottery has additional episode-assignment variation. It is not evidence that individual calls or every task's SR loss can be estimated that precisely. Keep all nine continuation cohorts for the shared profiler; Q2's six core cohorts require 21,120 of the added episodes, and dose_mix another 3,520. If the objective is a 2 pp certificate, choose a small frozen candidate family and add new independent init/scene support; merely repeating these two pilot inits is inefficient, especially where the measured ICC is positive.

## 2. Post-hoc / exploratory integration of full closed-loop evidence

### Census and accounting

One completed-summary snapshot, **2026-09-29 12:10:58 UTC (07:10:58 CDT)**: 262 completed arm/reference records, 244 eligible for the common 500 task/init set; 202 summary-ledger cost rows, 54 hash-verified historical raw-audit rows, four native DUAL references and two conflicting-cost rows. There are **18 completed frontier-completion arms** in this snapshot. Other planned arms are not counted as results; no polling was performed. The canonical pilot is analyzed separately, never pooled with these runs. Library variants demo100/200/300, grow250 and extra-bank-prior fits remain outside the strict 50/500 frontiers. `frontier/skipped.csv` lists every exclusion, including incomplete, smoke/subset and conflicting records. Actual served-bank provenance is read from each summary/manifest, including the nominal 50 π0.5 Spatial bank's 49 source episodes.

Each accepted journal must match the summary's episode count, successes and tasks0–9 × inits0–49. Owner IR is recomputed as **.152 v + .848 m** (π0.5) or **.148 v + .852 m** (GR00T), with blind/tail decisions costing zero; native pure L10=.5 and L5=1. Do not copy `ir_per_request`, `ir_per_five_controls` or measured wall-clock ledger IR, which may use different stage prices. For example, π0.5 L10-500 risk ρ=.24 has V=13,503, M=5,164, N=26,779: requested owner IR is **.240171**, whereas the summary's differently priced `ir_per_request` is .237044. π0.5 Spatial-50 B+.25 similarly becomes **.227529**, and L10-500 K7 confirmation **.205440**. All table values below use the requested coefficients, not approximate ledger-note values.

[Updated per-cell Pareto tables](final_analysis/frontier/frontier.md), [all points](final_analysis/frontier/frontier_points.csv), [paired native tests](final_analysis/frontier/paired_mcnemar.csv), [PNG](final_analysis/frontier/frontier_all_cells.png), [PDF](final_analysis/frontier/frontier_all_cells.pdf). The staircase denotes empirical dominance only; it is not a causal interpolation between controllers. Individual-run frontier selection is optimistic, especially when a method has many replicates.

### A/B repeats and guard/representation evidence

The A/B means reproduce `PAPER_AB.md`. For uncertainty I keep all three outcomes of each init together, rather than treating its 1,500 repeated pairs as independent. The interval below is a task-stratified paired init-cluster bootstrap, descriptive and unadjusted.


| Cell | A SR @ IR | B SR @ IR | B−A pp [paired cluster 95%] |
|---|---|---|---|
| π0.5 L10 50 | 0.714 @ 0.076 | 0.817 @ 0.181 | +10.27 [+7.27, +13.20] |
| π0.5 L10 500 | 0.827 @ 0.077 | 0.873 @ 0.159 | +4.60 [+2.27, +6.80] |
| π0.5 Spatial 50 | 0.837 @ 0.078 | 0.910 @ 0.140 | +7.33 [+5.60, +9.20] |
| π0.5 Spatial 500 | 0.976 @ 0.078 | 0.982 @ 0.119 | +0.60 [-0.27, +1.53] |
| GR00T L10 50 | 0.609 @ 0.074 | 0.719 @ 0.221 | +10.93 [+7.33, +14.47] |
| GR00T L10 500 | 0.830 @ 0.074 | 0.867 @ 0.187 | +3.67 [+0.67, +6.73] |
| GR00T Spatial 50 | 0.867 @ 0.075 | 0.875 @ 0.147 | +0.80 [-1.53, +3.07] |
| GR00T Spatial 500 | 0.964 @ 0.076 | 0.959 @ 0.127 | -0.53 [-1.93, +0.87] |


Removing only B's no-progress guard has opposite effects across cells. The exact McNemar p range below is across the three individual B references, with 500 matched task/inits each; no repeated pairs are stacked. Other guards removed individually have no detected SR loss in these runs, which does **not** establish their joint removability or 2 pp NI.


| Cell | B without no-progress: SR @ IR | Δ versus B mean, pp [cluster 95%] | Exact p versus B reps |
|---|---|---|---|
| GR00T L10 50 | 0.620 @ 0.096 | -9.87 [-13.27, -6.33] | 1.27e-06–0.000699 |
| GR00T Spatial 50 | 0.906 @ 0.105 | +3.07 [+1.33, +4.93] | 0.00372–0.00936 |
| π0.5 L10 50 | 0.704 @ 0.104 | -11.27 [-14.47, -8.13] | 1.42e-09–7.03e-05 |
| π0.5 Spatial 50 | 0.858 @ 0.112 | -5.20 [-7.07, -3.40] | 3.47e-06–0.000156 |


For GR00T Spatial-50, removing no-progress raises SR to .906 at IR .105293; it also exceeds the A mean by 3.87 pp, with exact p=.000878/.000325/.000546 versus the three A runs. Conversely, the guard supplies essentially all of B's LIBERO-10-50 improvement and most of π0.5 Spatial-50's improvement. A global “more rescue is safer” rule is therefore unsupported. This is intervention evidence for the guard package, not a per-trigger causal attribution to every firing.

The direct-token-PCA ablation changes the same library's realized quality: GR00T Spatial-50 reaches .910 at IR .075320, +4.27 pp versus pooled-PCA A, with p=.017/.012/.015. Direct PCA is worse on dense Spatial libraries: π0.5 .946 (−3.0 pp; p=.0014/.0125/.0241) and GR00T .928 (−3.6 pp; p=.00792 for each A replicate). Other direct-PCA differences are unresolved. Thus “library size” and state-only LOEO distance omit the deployed retrieval representation; a risk score must be versioned with that fit. These tests do not measure direct PCA's wall-time cost under the fixed owner coefficients, and no universal switch to direct PCA follows. [All exact ablation values and tests](final_analysis/derived/ablations_paired.csv).

### New frontier points: SR gain is not a noninferiority certificate

All 18 complete frontier arms have 500 episodes. Below are their exact SR, requested owner IR, native L10 loss and two-sided paired McNemar p. Positive loss means below pure SR. The last column is the one-sided 95% **lower bound on candidate−L10 SR** in pp; NI requires it to exceed −2. L5 comparisons for every arm, including each π0.5 reference replicate, are in [new_frontier_native_tests.csv](final_analysis/derived/new_frontier_native_tests.csv).


| Cell | Method | SR | Owner IR | L10 loss pp | Paired p (L10) | NI lower pp |
|---|---|---|---|---|---|---|
| GR00T L10 50 | B+d=0.25 | 0.742 | 0.2905 | +12.40 | 1.61e-07 | -18.39 |
| GR00T L10 50 | B+d=0.5 | 0.808 | 0.3578 | +5.80 | 0.0134 | -11.78 |
| GR00T L10 50 | B+d=0.75 | 0.850 | 0.4298 | +1.60 | 0.434 | -6.54 |
| GR00T L10 50 | risk ρ=0.35 | 0.832 | 0.3438 | +3.40 | 0.114 | -8.88 |
| GR00T Spatial 50 | B+d=0.125 | 0.878 | 0.1919 | +6.00 | 0.00122 | -10.87 |
| GR00T Spatial 50 | B+d=0.25 | 0.902 | 0.2340 | +3.60 | 0.0356 | -8.09 |
| π0.5 L10 500 | B+d=0.125 | 0.842 | 0.2062 | +6.20 | 0.00322 | -11.66 |
| π0.5 L10 500 | B+d=0.25 | 0.882 | 0.2415 | +2.20 | 0.289 | -7.37 |
| π0.5 L10 500 | K7 confirmation | 0.878 | 0.2054 | +2.60 | 0.187 | -7.60 |
| π0.5 L10 500 | risk ρ=0.18 | 0.856 | 0.1786 | +4.80 | 0.0106 | -9.72 |
| π0.5 L10 500 | risk ρ=0.24 | 0.884 | 0.2402 | +2.00 | 0.332 | -7.09 |
| π0.5 L10 50 | B+d=0.25 | 0.842 | 0.2603 | +6.20 | 0.00192 | -11.41 |
| π0.5 L10 50 | B+d=0.5 | 0.880 | 0.3388 | +2.40 | 0.213 | -7.27 |
| π0.5 L10 50 | B+d=0.75 | 0.874 | 0.4229 | +3.00 | 0.101 | -7.72 |
| π0.5 L10 50 | risk ρ=0.35 | 0.862 | 0.3416 | +4.20 | 0.0295 | -9.22 |
| π0.5 L10 50 | risk ρ=0.45 | 0.874 | 0.4464 | +3.00 | 0.11 | -7.83 |
| π0.5 Spatial 50 | B+d=0.25 | 0.922 | 0.2275 | +6.40 | 9.43e-07 | -9.84 |
| π0.5 Spatial 50 | B+d=0.5 | 0.962 | 0.3153 | +2.40 | 0.029 | -5.31 |


π0.5 L10-50 gains SR from B+.25 to B+.5, but B+.75 and risk ρ=.45 remain at .874. This is an observed plateau, not proof that every higher-budget method must plateau. GR00T L10-50 reaches .850 at IR .429781 with B+.75; its p=.434 versus native L10 does not prove 2 pp NI. The π0.5 L10-500 B+.125 dip is significant only against the strongest B replicate, so it is insufficient to conclude that random calls causally damage dense libraries in general. At similar IR, risk versus B+random reverses point ordering between the two sparse L10 models; no universal winner follows.

The historical π0.5 L10-500 K7/hand point (.906 @ .196533) looks excellent, but the new confirmation is .878 @ .205440. Selecting the older point as a frontier minimum must not be mistaken for replicated performance. The strongest GR00T Spatial-50 low-cost alternatives are removal of no-progress and direct PCA; B+.25 costs .234 and reaches only .902. Placement and representation can matter more than buying extra random calls.

### Minimum tested IR at 2 pp NI, separately for both native references

Exact McNemar is the conditional binomial equality test on discordant pairs; it is **not** a 2 pp NI test. To keep the earlier Q2 frontier definition unchanged, nominal NI uses the conservative exact paired risk-difference lower bound

`Lα = CP_lower(wins/n; α/2) − CP_upper(losses/n; α/2)`, α=.05,

with endpoints 0/1 and NI iff Lα>−.02. These two one-sided Clopper–Pearson bounds cover jointly by Bonferroni. For the π0.5 L5 reference mean, split α over its three constituent comparisons and average the lower bounds; keep the same 500 init clusters. This is deliberately conservative and assumes exchangeable independent init pairs within the benchmark sampling interpretation. Failure means unresolved, not proven inferiority. The native references are π0.5 L10 .904/.986 and L5 means .850667/.990667; GR00T L10 .866/.938 and L5 .870/.940 (L10/Spatial). Historical DUAL pairing matches task/init identity, but not the complete profiler seed/harness experiment.

**These are minima among observed, eligible points passing a nominal test, not estimated global minimum budgets.** “Pure” denotes the reference itself as a known-cost option. The table reports B's remaining point gap as context, not as its NI result.


| Cell | B loss to L10 / L5, pp | Lowest nominal NI IR vs L10 | Lowest nominal NI IR vs L5 |
|---|---|---|---|
| π0.5 L10 50 | +8.73 / +3.40 | 0.50000 (pure) | 0.50000 |
| π0.5 L10 500 | +3.07 / -2.27 | 0.50000 (pure) | 0.19653 |
| π0.5 Spatial 50 | +7.60 / +8.07 | 0.50000 (pure) | 1.00000 (pure) |
| π0.5 Spatial 500 | +0.40 / +0.87 | 0.12273 | 1.00000 (pure) |
| GR00T L10 50 | +14.73 / +15.13 | 0.50000 (pure) | 1.00000 (pure) |
| GR00T L10 500 | -0.07 / +0.33 | 0.50000 (pure) | 1.00000 (pure) |
| GR00T Spatial 50 | +6.27 / +6.47 | 0.50000 (pure) | 1.00000 (pure) |
| GR00T Spatial 500 | -2.07 / -1.87 | 0.07551 | 0.07551 |


The nonidentity minima and their ordinary paired tests are:


| Cell / reference | Selected observed arm | SR | NI lower pp | Exact McNemar wins/losses; p |
|---|---|---|---|---|
| π0.5 L10 50 / L5 | `r04_cost/r4f_p_l10_inf_k10_L10` | 0.904 | -0.54 | 54/24; 0.000901; 52/24; 0.00176; 50/28; 0.0169 |
| π0.5 L10 500 / L5 | `r05_b1/r5b_p_l10_500_hand` | 0.906 | -0.68 | 61/30; 0.00152; 61/32; 0.00346; 51/28; 0.0128 |
| π0.5 Spatial 500 / L10 | `r05_q6/r5q6_p_spatial_500_tail` | 0.990 | -1.75 | 7/5; 0.774 |
| GR00T Spatial 500 / L5 | `r05_x/r5x_g_sp_500_tail1u` | 0.964 | -1.31 | 27/15; 0.0884 |
| GR00T Spatial 500 / L10 | `r05_x/r5x_g_sp_500_tail1u` | 0.964 | -1.06 | 27/14; 0.0596 |


For identity reference choices the paired difference is exactly zero by definition, so there is no independent test. All other arms' L5/L10 plain tests and nominal NI bounds are exported, including failed candidates with nonsignificant p-values. Searching the full census requires additional multiplicity protection: with α=.05/518 across the 518 cell/arm/reference tests, **no cheaper-than-reference point passes**. The corresponding minimum remains .5 versus L10 and 1 versus L5 in every cell. The nominal exceptions above are exploratory leads; the dense π0.5 Spatial and GR00T Spatial results warrant replication, and the old K7/hand L5 result particularly needs its weaker confirmation acknowledged.

## 3. Q2 recommendation and a portable validation recipe

### Method and knob

**Expose ρ, target deployment owner IR. Keep α internal.** A calibrated guard false-alarm level is not a call budget, and zero-to-38% successful-library alarm variation from `ideation_G/REPORT.md` does not calibrate SR loss. Do not expose both as independent owner controls or promise that successful-library calibration proves closed-loop safety. Use the lower-cost validated point only when it passes the chosen native SR criterion; otherwise the honest minimum remains unresolved below the reference controller's cost.

For R6, retain the existing A retrieval/commit and the deployed B comparison. Use **uniform allocation as the cost-calibrated budget baseline**, and keep the frozen risk allocator as an experimental alternative. There is no evidence here to adopt state-coverage weighting everywhere or to automatically convert a library-quality score into an ε=.02 budget. Dense-library A remains a useful low-cost candidate; sparse L10 cases need much more spending or better placement. No task-specific budget is learned from the pilot's two inits.

Exact portable budget algorithm:

1. Read the library manifest and robot adapter: task/context key, valid state/action dimensions, camera/key representation, action convention, request controls R and available chunk H. Set b=min(2,floor(H/R)); reject H<R. Keep the retrieval fit and its provenance fixed. Commitment is bR controls (10 here); the current adapters require R=5/H≥10, so another robot needs the corresponding protocol adapter, not a transplanted numeric slice.
2. Calibrate the production vision fraction c1 by replay-timing vision and full inference on recorded library inputs under the target backend, with profiling shadows disabled. For these analyses c1=.152/.148 is the requested owner accounting convention. Compute h_t=mean source-episode decision length and a_t=mean ceil(length/b), weighting source episodes equally. Traffic weights π_t are supplied; use equal task weights if absent. These lengths, costs and commitment yield the portable spending signal even when no useful SR-risk score exists.
3. Default q_t=1. The experimental risk alternative uses the **unchanged frozen** state LOEO distance: valid-state standardized RMS nearest-neighbour distance excluding the entire source episode, averaged first per source episode then per task. Normalize w_t=q_t/sum(πq). Use machine-epsilon times the maximum positive q for zero q; use uniform if all q are zero. A later Q1 score must be library-only, nonnegative, higher-is-worse and tagged with the deployed representation/fit; freeze it before new outcome evaluation. No outcome-ranked proxy substitution is allowed.
4. Solve p_t=min(1,λw_t) by the frozen 80-step bisection so that `sum π_t a_t [c1+(1−c1)p_t] / sum π_t h_t = ρ`. Clamp below the A vision floor to p=0 and above the every-anchor ceiling to p=1, recording the infeasible target. In particular ρ=0 yields A, whose vision cost is not zero. Predicted calls per task/episode are a_t p_t; long tasks consume more calls. Uniform means equal per-anchor p, not equal absolute episode MISS counts.
5. At episode start, independently choose the two adjacent fixed doses in {0,1/8,1/4,1/2,1} with probabilities giving expected p_t. Hold the chosen dose for the episode. At each vision anchor draw an independent keyed Bernoulli at that dose; execute policy with the same bR commit, otherwise A. Log the episode dose/probability, anchor coin/propensity, actual controls and served policy count. Use `frontier/adapters/methods.py:RiskLottery`, `allocation=uniform` or `risk`; disable shadow/profiling. The SHA-keyed episode/init/decision randomization in that version is already implemented; replay identity and seed tests are historical implementation evidence, not a new SR test here.
6. Library lengths need not equal induced deployment lengths. Report realized-minus-predicted IR, without outcome-based retuning. If three previously recorded, designated A trajectories are available on a new platform, a **prospective cost-only** correction may multiply every a_t by γ: observed `sum V / sum(C/R)` divided by the library `sum a_t / sum h_t` over those same trajectory task keys. Freeze γ and provenance before validation; γ=1 without those trajectories. If γa_t>h_t for any task, reject this correction and recollect protocol/calibration data rather than silently changing cadence. This optional correction is not part of the frozen pilot estimate and has no measured SR benefit here.

B+extra remains a useful deployed comparator, not a calibrated risk allocator: its total call probability is g_t+(1−g_t)d, and the guard rate g_t changes with the induced trajectory. Thus d is not ρ. If a later Q3/G guard ranks where to spend these budgets, its α/quantile must be selected internally from library LOEO or designated recorded calibration trajectories to match the budget; its SR requires a new arm. In particular, do not hard-code “disable no-progress on GR00T Spatial” as a portable rule. The current sign reversal falsifies a universal benefit assumption, not every possible calibrated progress guard.

### Revised SR acceptance objective (prospective, not the frozen pilot rule)

Freeze ε=.02 and the native L5/L10 reference definitions before evaluating candidates. Require a one-sided family-adjusted lower SR difference >−.02 against **both** references, then minimize measured owner IR over eligible candidates; use lower-complexity uniform allocation as the exact tie-break. Report B as a paired comparator, but **remove the requirement to be cheaper than B when B itself fails the native SR objective**. This is necessary to pursue the owner's stated objective. If no candidate qualifies, retain the relevant pure-reference option and report the unsolved crossing. The chosen higher-SR pure reference must not be replaced by a weaker cadence merely to make an NI claim easier.

### At most eight candidate configurations

The most useful missing comparison is the implemented **uniform allocator against risk at the same target cost**, rather than another comparison against a different-cost B. Propose exactly **eight candidate configurations total**, two in each 50-episode-library cell below. Each uses the exact frozen A fit/kwargs for that cell, `RiskLottery`, `rho=.45`, `allocation` as shown, 10-control commitment, no shadow, and a new declared lottery seed **26093001** with `randomization_key="Q2-native-NI-v2/<cell>"`. Share that key/seed between allocations within cell. ρ=.45 is a prospective high-budget checkpoint common to all four cells, chosen after these frontier results; it is not claimed to be the minimum crossing or a preregistered pilot grid point.


| Cell | Configuration 1 | Configuration 2 | Paired question / prediction |
|---|---|---|---|
| π0.5 L10 50 | uniform ρ=.45 | risk ρ=.45 | Same library-predicted IR; risk benefit and native 2 pp NI unresolved |
| π0.5 Spatial 50 | uniform ρ=.45 | risk ρ=.45 | Same library-predicted IR; risk benefit and native 2 pp NI unresolved |
| GR00T L10 50 | uniform ρ=.45 | risk ρ=.45 | Same library-predicted IR; risk benefit and native 2 pp NI unresolved |
| GR00T Spatial 50 | uniform ρ=.45 | risk ρ=.45 | Same library-predicted IR; risk benefit and native 2 pp NI unresolved |


Pair each candidate on the same `(task,init)` evaluation set against the existing three A runs, three B runs, native L10 and native L5 references; retain repeated init outcomes as clusters. Eight refers to new candidate configurations; those already completed references are reused, not counted as new experiments. Use 500 task/init pairs per candidate (4,000 candidate episodes) for an initial implementation/curve check, with outcomes generated under the declared new lottery seed. Record environment and policy seed provenance explicitly: historical native references match task/init but not the P3 seed experiment. Reusing old init scenes cannot establish new-scene or robot transfer. If a contemporaneous rerun of all reference controllers is required, that is additional work outside this eight-candidate budget and must be counted openly.

For the revised confirmatory selection, predeclare 16 candidate×native-reference SR tests, α=.05/16 (split each π0.5 L5 mean's α across its three component paired comparisons). Keep A/B and risk−uniform contrasts secondary. Rank only passing candidates by IR. A 500-pair check is not guaranteed 2 pp power: calculate the final sample size from the paired variances and actual repeat design above; if inconclusive, add independent init/scene support before making the NI claim. Because .45 is selected post-hoc, a strict new confirmatory claim additionally needs outcomes outside the selection data; new seed repeats are a conditional check on these scenes, not an untouched initialization holdout.

Prediction, explicitly a hypothesis: actual IR should be near .45 when library cost calibration holds; there is no supported numerical SR prediction for the unrun uniform controllers and no reason to assume risk wins. The observed π0.5 L10-50 risk .45 result (.874) warns that even this budget may miss the .904 L10 target. If both allocators fail, the answer is to improve placement/representation or retain pure inference, not to infer an unmeasured crossing. The shared +31,680-episode continuation remains the more efficient way to evaluate the frozen expected lotteries over the .10–.25 grid; these eight unprofiled configurations address implementation and the higher-budget region.

### Falsification and remaining uncertainty

- The state-coverage allocation proposal fails as a general recommendation if risk does not improve held-out SR at no greater realized IR than uniform across independent libraries. Three point-dominant pilot checkpoints cannot establish that property. A representation change that improves SR without changing state coverage, as in the direct-PCA ablation, already limits any state-only quality-to-budget claim.
- The portable cost rule fails its prospective cost calibration if repeated realized IR falls outside a preregistered ±.02 absolute band around a feasible ρ target; the observed pilot error range is encouraging but not a guarantee for new policies or robots. Freeze this tolerance before new evaluation. Persistent failure requires new cost/length calibration, not task-outcome tuning.
- Near-pure performance fails for a tested candidate if its SR loss is conclusively greater than 2 pp versus a required native reference. A failed NI test alone is inconclusive. Higher calls that leave SR flat or lower refute an automatic monotone-gap-closure controller; they do not identify the mechanism without matched placement tests.
- Transferring the budget algebra is supported by its manifest/cost/length inputs; transferring the SR frontier is unverified. A second benchmark/robot and independently built libraries, with the same frozen algorithm and only library plus designated recorded-trajectory calibration, are necessary. All eight current cells share one benchmark family and nonrandom library construction.

## Reproducible commands and artifacts

Run from `/home/weiland/projects/openpi`. The canonical input symlinks were made only in the owned directory. The following are the commands executed, with stdout/stderr saved as `final_analysis/preregistered.log`, `frontier.log` and `derived.log`. The first two writers intentionally refuse to overwrite their original output directories; use a fresh owned output directory for a new audit snapshot rather than mixing inventories.

```bash
bash exp/offline_search/rounds/r06/ideation_Q2/run_pilot_q2.sh   --tables exp/offline_search/rounds/r06/ideation_Q2/final_analysis/input_tables   --stage pilot --out exp/offline_search/rounds/r06/ideation_Q2/final_analysis/preregistered

taskset -c 2-5,46-49 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src   TMPDIR=exp/offline_search/rounds/r06/ideation_Q2/.tmp   MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q2/.mplconfig   .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/final_analysis/audit_frontier.py

taskset -c 2-5,46-49 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src   TMPDIR=exp/offline_search/rounds/r06/ideation_Q2/.tmp   MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q2/.mplconfig   .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/final_analysis/derive_results.py

taskset -c 2-5,46-49 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src   MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q2/.mplconfig   .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/final_analysis/plot_frontier.py

taskset -c 2-5,46-49 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src   .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/final_analysis/write_final.py
```

The frozen `power.json` is deliberately unchanged and still contains planning sensitivities; use `derived/measured_precision.csv` for the measured update. `frontier/summary_paths.json`, `source_evidence.json` and `outcomes.json` freeze the full-run sources, hashes, configs and accepted outcomes. Both the revised all-cell figure and its PDF are inside this ideation directory; the earlier external artifact directory was not modified. The PNG was visually inspected. All computations above use existing data; proposed calibration timings, future arms, new-scene evaluation and transfer remain unrun.

