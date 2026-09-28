# R6 Q2 — how much policy inference to buy

The measured answer is **task-, library-, and controller-dependent**. At the 50-episode libraries, B/G10 spends about 0.075–0.144 MISS per five-control slot and buys 5.2–12.4 SR points over A. At 500 episodes, π0.5 spatial needs no calls to match its measured B, and GR00T periodic calls are point-dominated by A. π0.5 l10 still benefits from rescue. A scalar MISS rate cannot identify SR without fixing commitment and call placement.

**Recommendation for R6:** expose a **target IR** knob, calibrate the task allocation and finite-episode call counts from the library, and test it with the eight arms below. Do not expose a library-only “SR loss ≤ ε” promise yet. The simple LOEO score has held-cell gap error around 11 SR points; the best tested gap-response model still misses a held cell by 6.3 points. The proposed allocation is an experiment, not a demonstrated improvement over B. For the currently measured strong spatial libraries, A is the evidence-based zero-call candidate; a target IR is a resource constraint, not evidence that spending it helps.

## 1. Measurement, scope, and reproduction

All work here is offline and CPU-only. No server, worker, chain, port, tmux session, GPU, git operation, or live experiment was started or changed. Writes are confined to this directory and `/tmp/r6_q2/`. The fixed snapshot is `snapshot.json`; a rerun reuses its completed-arm list even if more R6 arms finish later.


Snapshot UTC: **2026-09-28T17:51:27.459289+00:00**. The audit includes **177 arms, 87,500 accepted episodes**, and **4,094,415 decision counts**. There are 173 completed production arms from R2–R6 plus four original pure-inference trace arms. Production arms require both a DONE marker and summary; journal terminal attempts are matched before reading server decisions. Smoke runs, `r03_pilot`, incomplete arms, and evaluations other than 500 episodes or the four 250-pair grow250 arms are excluded. This includes all completed requested periodic, guard, quantile, A, B, G10, policy and K5 arms, plus supplemental production methods.


Sources: `/home/weiland/trace_runs/os_closed_loop/<run>/runs/<arm>/{summary.json,client/journal.jsonl,server_*/decisions_*.jsonl}`; original policy journals under `/home/weiland/trace_runs/dual_20260923/runs/`; library/trajectory arrays under `/home/weiland/trace_runs/offline_search_store/`. Exact paths, arm specs/kwargs, summary/journal SHA256 hashes, and decision-log sizes/mtimes are retained in `arms.json`. The original policy episode lengths are independently joined from the query store. Run `bash exp/offline_search/rounds/r06/ideation_Q2/run.sh` from the repository root. It executes these scripts, in order, each with:

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/r6_q2/matplotlib .venv/bin/python \
  exp/offline_search/rounds/r06/ideation_Q2/<script>.py
```

`measure.py` → `price_index.py` → `library_proxy.py` → `analyze.py` → `extend.py` → `verify.py` → `write_report.py`. Ingestion uses four worker processes; other scripts are single-process, with numeric libraries restricted to one thread. `verification.json` checks the frozen hashes, unique episode pairs, success counts, raw N/V/M totals, summary MISS totals for valid arms, and the proposed quota arithmetic.

Outputs:

- [Complete curves/arm index](CURVES.md), [arm CSV](arms.csv), [episode CSV](episodes.csv), [all task counts](all_task_counts.json).
- [SR vs logged MISS share](sr_vs_m.png), [SR vs normalized calls](sr_vs_m5.png), [SR vs IR](sr_vs_ir.png), [library-size curve](library_size_curve.png).
- [Detailed per-task A/B/policy tables](TABLES.md), [task budget CSV](task_budgets.csv), [marginal returns](marginal_returns.json), [quality validation](quality_validation.json), [K5 causal results](causal.json).
- [Exact proposed test configurations](test_plan.json), [response predictions](response_model.json), [episode-routing predictions](predictions.json), [ε frontier calculations](epsilon_frontier.json).

### Denominators and uncertainty

Let N be accepted requests, V vision requests, M fresh policy calls, L nominal controls/request, and D=N·L/5 be five-control equivalents. Report **m=M/N**, **m5=M/D**, and **IR=[c1·V+(1−c1)·M]/D**, with c1=.152 for π0.5 and .148 for GR00T. A/B/G10 have L=5 including their blind tail requests, so m=m5. Pure inference L10 has m=1, m5=.5, IR=.5. The m=1 endpoint must not be confused with one call per five controls. Cost is nominal requested controls; terminal partial chunks are not reconstructed as actual controls.

The primary reference is pure inference with the **same ten-control commitment** as A: π0.5 l10 uses both L10 repeats (.900/.904, pooled .902); the other cells use their one completed L10 arm. The L5 points remain in every curve. Pure-inference SR is an empirical comparator, not an upper bound: A can outperform it on some tasks. Supplemental wrist and K2 entries use the R4/R5 owner-price assumptions, explicitly tagged in the CSV; neither is used to calibrate the budget model. `ir_full_reference` preserves uniform full-camera/full-policy repricing.

SR differences use paired `(task, init)` outcomes. Intervals are 10,000 paired bootstrap draws, seed 20260928, preserving the within-init mean of policy repeats. Task numbers are never paired across suites. Slope intervals hold the observed cost difference fixed; they do not integrate uncertainty in the denominator. These are exploratory, pointwise, conditional on the tested tasks/seeds. Replicate runs are not independent additional initializations. No multiple-testing or new-task SR guarantee is claimed.

**Two legacy costs fail the raw-log audit.** `r3mx_p_l10_ev_h70` has 487 episodes with conflicting duplicate records; raw accepted M=11,203 versus summary M=12,112 at N=30,883. `r3mx_p_sp_awm_h50` has 63 conflicting episodes; raw N/M=10,851/5,874 versus summary 10,848/6,207. Their journal SRs (.822/.986) remain in the index, but neither cost enters a fitted curve/frontier. The latter's current summary implies m=.57218 and IR=.63721, unlike R4's reported .605 IR. It would be misleading to treat that historical h50 point as clean budget evidence. Details: `legacy_conflicts.json`.

## 2. Measured findings

### A → B/G10 → matched pure inference

Here B means π0.5 C10 committed rescue, and G10 means the historical GR00T scheduled call every fourth anchor. “Gap closed” is `(SR_B−SR_A)/(SR_P10−SR_A)` when the denominator is positive. A has m=0 in every row.


| Cell | Library | A SR @ IR | B/G10 SR @ IR | P10 SR | m(B) | MISS/episode | Gap closed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10 | 50 | 0.706 @ 0.076 | 0.830 @ 0.181 | 0.902 | 0.1229 | 7.49 | 63.3% |
| pi05_l10 | 500 | 0.828 @ 0.076 | 0.866 @ 0.161 | 0.902 | 0.0997 | 5.74 | 51.4% |
| pi05_spatial | 50 | 0.838 @ 0.078 | 0.910 @ 0.141 | 0.986 | 0.0752 | 1.77 | 48.6% |
| pi05_spatial | 500 | 0.982 @ 0.078 | 0.982 @ 0.121 | 0.986 | 0.0501 | 1.06 | 0.0% |
| groot_l10 | 50 | 0.608 @ 0.074 | 0.718 @ 0.185 | 0.866 | 0.1299 | 8.68 | 42.6% |
| groot_l10 | 500 | 0.830 @ 0.074 | 0.828 @ 0.186 | 0.866 | 0.1309 | 7.84 | -5.6% |
| groot_spatial | 50 | 0.868 @ 0.075 | 0.920 @ 0.198 | 0.938 | 0.1439 | 3.40 | 74.3% |
| groot_spatial | 500 | 0.964 @ 0.076 | 0.952 @ 0.199 | 0.938 | 0.1448 | 3.25 | A > policy |


At 50 episodes, the same affordable interior rate does **not** reach pure inference on l10: π0.5 remains 7.2 points below P10 and GR00T 14.8 points below P10. At spatial-500 the corresponding observed A→B/G10 gains are 0 and −1.2 points. Pure L5 versus L10 is a distinct execution-length intervention and must not be used to infer the value of extra calls. G10's nominal one-in-four paid anchors implies asymptotic m=.125, but its paid first anchor and finite endings produce measured m=.130–.145. The proposed randomized initial phase removes that forced-first-call bias.

The frozen snapshot also contains **GR00T guard B at l10-500**: .864 SR, m=.13155, IR=.18663, M=3,777. Periodic G10 has .828 SR, m=.13094, IR=.18605. Guard B−G10 is +3.6 points, paired interval [0.0, 7.4], McNemar p=.0693; guard B−A is +3.4 [0.0, 6.8], p=.0712. Guard B−P10 is −0.2 [−4.0, 3.6]. This single result is suggestive, not resolved equivalence or superiority. It is a concrete warning against predicting SR from m alone, since the two configurations buy almost the same number of calls. Source: `r6_completed_guard.json` and `r06_paper/r6p1_c10_g_l10_500`.

### Marginal SR per MISS share: diminishing returns is conditional

One unit of slope below is one unit of SR per unit m; equivalently, multiply by 10 to obtain SR percentage points per +.10 MISS share. These are same-family observed secants, not causal derivatives at a state.


| Sequence | Δm | ΔSR pp | ΔSR/Δm | 95% conditional interval |
| --- | --- | --- | --- | --- |
| oscl50_p_l10_cl2 → r4_p_l10_per6_50 | 0.1609 | 13.4 | 0.833 | [0.547, 1.119] |
| r4_p_l10_per6_50 → r3mx_p_l10_perk5 | 0.0311 | 2.8 | 0.899 | [-0.321, 2.119] |
| r3mx_p_l10_perk5 → r3mx_p_l10_perk3 | 0.1347 | 4.0 | 0.297 | [0.030, 0.579] |
| oscl500_p_l10_cl2 → r4_p_l10_per12_500 | 0.0751 | 6.0 | 0.799 | [0.373, 1.225] |
| r4_p_l10_per12_500 → r4_p_l10_per8_500 | 0.0446 | 2.2 | 0.493 | [-0.269, 1.255] |
| oscl50_p_sp_cl2 → r3mx_p_sp_perk3 | 0.3169 | 13.8 | 0.436 | [0.309, 0.568] |
| oscl500_p_sp_cl2 → r4_p_sp_per12_500 | 0.0615 | 2.2 | 0.358 | [0.098, 0.618] |
| r3mx_p_l10_awm_h70 → r3mx_p_l10_awm_h50 | 0.1889 | 5.2 | 0.275 | [0.085, 0.466] |


π0.5 l10-50's periodic slope falls to .297 from approximately .83–.90 at the lower rates; the .90 middle estimate is noisy and does not establish monotone curvature. At l10-500, the CL2→per12 slope is .799, and per12→per8 is .493 with an interval crossing zero. Confidence h70→h50 at l10-50 spends +.1889 m for +5.2 SR points: slope .275. R3 h70/guard-only/B points use different placement rules and cannot be concatenated as a pure dose-response experiment.

The A→B slopes are **1.009, .381, .958, 0** for π0.5 l10-50/l10-500/spatial-50/spatial-500. A→G10 slopes are **.847, −.015, .361, −.083** for the corresponding GR00T cells. Negative slopes are retained. An isotonic fit that deletes them would hide the main warning in the data. The three stock π0.5 l10-500 guard repeats are .864/.850/.832, while realized m increases .102/.109/.117: failures themselves generate extra downstream calls, so regressing episode success on realized M is not a causal call-value estimator.

### Randomized CALL/CACHE: one useful call can reduce total calls

The K5 treatment is one assigned first/third guard landmark, not a randomly assigned total episode budget. I independently matched all 2,000 exported randomized episode Y/N/M totals to this turn's raw audit, then recomputed CALL−CACHE using the complementary replicate pair per `(task, init)`. The 500-pair bootstrap keeps both treatments together. Sources: `causal.json` and R5 Q3's raw-audited assignment exports.


| Library | CALL−CACHE ΔSR [95%] | Δrequests [95%] | Δtotal MISS [95%] | CALL/CACHE m |
| --- | --- | --- | --- | --- |
| 50 | -0.002 [-0.038, +0.034] | -0.044 [-1.700, +1.622] | +0.472 [-0.772, +1.700] | 0.19980/0.19247 |
| 500 | +0.034 [+0.004, +0.066] | -1.582 [-2.970, -0.228] | -0.238 [-1.046, +0.558] | 0.10755/0.10866 |


At 500, assigned CALL improves SR by 3.4 points while its **total** MISS difference is −.238. Thus `ΔSR/Δtotal MISS` has the wrong sign for a “value per additional call” interpretation. At 50 the SR effect is unresolved. This result supplies no causal justification for indiscriminately suppressing guard calls at high-quality libraries, and does not transfer by itself to committed B, spatial, GR00T, or all decisions. It supports budgeting **interventions ex ante**, while reporting both realized total calls and episode duration ex post.

### Task and episode allocation

All 80 A/B/P10 task rows, actual MISS totals/means/medians/p90s, and gap estimates are in `TABLES.md`/`task_budgets.csv`; `all_task_counts.json` covers all 177 arms. Task IDs below are descriptive audit labels, never algorithm constants.

- π0.5 l10-50 task 0: A .30, B .76, P10 .99, B 10.00 calls/episode. Task 4: A .64, B .60, P10 .89, **13.68 calls/episode**. More calls and a large remaining gap do not establish that the existing rescue placement helps.
- π0.5 l10-500 task 0: A .62→B .96, only 4.44 calls/episode. Task 6: .72→.70 despite 8.90 calls/episode. Tasks 3/5/8 already have A≥P10 at the point estimate, yet receive **29.45%** of all B calls. This is a diagnostic concentration, not a causal estimate of “waste.”
- π0.5 spatial-50 task 6 contributes .36→.76 with 4.54 calls/episode, while task 2 is A=B=P10=1 and still gets .58 calls/episode. At spatial-500, B spends 1.058 calls/episode for zero aggregate SR gain; **67.49%** of calls fall on tasks with nonpositive measured A→P10 gap.
- GR00T l10-50 task 9 has A .14, G10 .40, P10 .92 and 10.88 calls/episode; task 5 has A .96, P10 .92 and still receives 5.12 scheduled calls. GR00T spatial-500 allocates **81.95%** of G10 calls to tasks with A≥P10 at the point estimate.

Failed episodes account for **55.51%/47.47%** of π0.5 l10 B calls at 50/500, despite failure rates of 17.0%/13.4%. Mean episode MISS fraction is .1001/.0829, below decision-weighted m=.1229/.0997; longer failures explain part of that difference. B call-count p90 is 24.0/17.1. Report both means and counts; do not enforce the same absolute cap on a short episode and a long episode, or interpret a post-failure budget overrun as the causal failure mechanism. No successful policy for truncating these late rescues has been identified by this analysis.

### Library quality predicts some gap, but not an accurate SR target

For each task, the new simple proxy standardizes **valid state coordinates only**, queries every library row against rows in other episodes of the same task, takes nearest-neighbor RMS distance, and averages rows within episodes then episodes within tasks. Smaller distance d means better coverage; q=mean_episode(exp(−mean_row_distance)) is stored as a descriptive coverage transform. State scales use the whole deployed task library, so this is conditional retrieval LOEO, not full-pipeline inductive LOEO. No full vision/metric quality design is attempted here.

Four model/suite cells × 50/500 plus the six π0.5 100/200/300 points provide **14 library settings and 140 task gaps**. The additional 80 task-level action/state-residual proxies are read from `r05/ideation_B/verified_library_<cell>_<size>.json`, whose metric refits exclude the held episode but whose PCA/scales are frozen. Those are library-only predictors; SR labels are used only to evaluate or fit the explicitly outcome-calibrated models below.


| Proxy | Task rows | Spearman(gap) | In-sample R² | Held-scene MAE / null | Held-cell MAE / null | Held-suite MAE / null |
| --- | --- | --- | --- | --- | --- | --- |
| state_loeo_distance | 140 | 0.541 | 0.213 | 0.103 / 0.114 | 0.110 / 0.118 | 0.131 / 0.129 |
| action_loeo | 80 | 0.322 | 0.062 | 0.131 / 0.133 | 0.140 / 0.134 | 0.150 / 0.144 |
| state_residual_loeo | 80 | 0.465 | 0.113 | 0.125 / 0.133 | 0.142 / 0.134 | 0.184 / 0.144 |
| inverse_sqrt_episodes | 140 | 0.348 | 0.121 | 0.107 / 0.114 | 0.111 / 0.118 | 0.125 / 0.129 |


The descriptive least-squares gap model is `P10−A ≈ −.111686 + .945821·d`. It explains only .213 of task-gap variance. “Held scene” keeps the same suite/task across all sizes and both models together; held cell removes both sizes and intermediate sizes of one model/suite; held suite removes the entire suite. The null predicts the training mean gap. The state score helps the scene/cell split modestly and **loses to the null on the suite split**. Action LOEO error loses to the null on held-cell and held-suite validation.

I also checked the same state proxy against residual policy gaps in all applicable completed families: Spearman is .541 for A (140 task-arm rows), .411 for B (50), .457 for G10 (40), .386 for CL2 (100), .202 for periodic (90), .426 for guard (80), and **.041 for confidence-quantile arms (60)**. These 560 rows share tasks and labels; they are not 560 independent validations. This tests the proxy against existing closed-loop behavior rather than treating offline agreement as SR.

The zero-call demo-size curves reinforce the caveat: π0.5 l10 A is .706/.748/.788/.806/.828 at nominal 50/100/200/300/500, but spatial is .838/.950/.970/**.944**/.982. The current 50 bank has a different source from the nested 100/200/300/500 demos, and 500 also changes kref. State coverage gets better with size without proving monotone closed-loop SR. The 100/200/300 libraries have only zero-call A points and policy endpoints; there are no observed interior MISS-budget points to fit a separate response curve at those sizes.

## 3. Budget models and exact proposed algorithms

### Descriptive gap-closing model — useful forecast, not an SR contract

Hold A, the policy commitment, and a call-placement family fixed. Let Δ=P10−A and x=m5∈[0,.5]. The endpoint-constrained response family is

`S(x) = A + Δ·[1−(1−2x)^γ]`.

For a positive interior gain below Δ, calibrate `γ=log(1−gain/Δ)/log(1−2m_B)`. Then the calls needed to close fraction f of the observed gap are `x(f)=.5·[1−(1−f)^(1/γ)]`. For target loss ε<Δ, use f=1−ε/Δ. This guarantees neither shape nor portability: it interpolates only one interior mixed arm and the two endpoints. We do not fit a positive individual γ when the gain is zero/negative or the policy gap is nonpositive.


| Cell | Lib | γ | m5 for 50% gap | 80% | 90% | m5 for ε=.02 |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10 | 50 | 3.549 | 0.089 | 0.182 | 0.239 | 0.237 |
| pi05_l10 | 500 | 3.241 | 0.096 | 0.196 | 0.254 | 0.166 |
| pi05_spatial | 50 | 4.090 | 0.078 | 0.163 | 0.215 | 0.193 |
| groot_l10 | 50 | 1.848 | 0.156 | 0.291 | 0.356 | 0.375 |
| groot_spatial | 50 | 4.003 | 0.079 | 0.166 | 0.219 | 0.134 |


A shared γ is fit by unweighted squared SR error over positive-gap A/B/G10 cell-size points, bounded to [0,20] with scalar tolerance 1e−10; its fitted value is **2.5595**. Holding out the whole model/suite cell gives SR MAE **0.0228**, maximum error **0.0630**, even though the held-out A/P endpoints are supplied. Therefore the tiny-ε extrapolations above are hypotheses, not calibrated guarantees. This model borrows closed-loop outcomes; it is not part of the library-only serving rule.



### Proposed serving rule: library-weighted paid-anchor quotas, one knob ρ

This is an exact **budget-layer experiment**. It leaves A's existing action representation and commitment lengths as supplied by the base controller. Its user-facing knob is target owner IR **ρ**. For a new robot, profiling supplies c1, the robot schema supplies valid state coordinates, and the base controller supplies the commitment length. None is inferred from LIBERO task IDs or a hard-coded horizon.

1. **Calibration input.** Group the deployed library by the same task/instruction key used by the base retriever. For each task t and each state coordinate, use the task-library mean and population standard deviation; omit exactly constant coordinates. Query each row against all rows of other episodes in that task. Distance is Euclidean distance divided by sqrt(number of active coordinates). Set d_t to the mean of episode-mean nearest distances. No percentiles, gripper signs, action-coordinate indices, or thresholds are introduced. If no active dimensions, fewer than two episodes, or nonfinite features prevent estimation, mark that task unknown and use weight 1. For valid tasks use `w_t=d_t/mean_valid(d)`; if the mean is zero use all weights 1. Unknown-task weight 1 is a neutral budget prior, not a quality claim.
2. **Length/cost calibration.** Let b be committed controls divided by the base logged control quantum. Here b=2 because A/P10 both commit ten controls and the logs use five. For every library episode of length h base quanta, compute a=ceil(h/b). Set H_t=mean(h) and A_t=mean(a) over its library episodes. Set deployment task probabilities π_t from a supplied traffic distribution; if absent, use equal task weights. The task set is discovered from the library, never assumed to contain ten tasks.
3. **Solve the budget.** Define `p_t=min(1, λ·w_t)`. Choose λ by monotone bisection so

   `Σ_t π_t [c1 A_t + (1−c1) A_t p_t] / Σ_t π_t H_t = ρ`.

   The minimum attainable library-predicted IR is the same expression with all p=0; the maximum uses p=1. Clamp an out-of-range target to that attainable interval and report the attained value. Use 80 bisection iterations and upper bound `1/min_positive(w)`. If zero-weight tasks exist, first saturate all positive-weight tasks; any target above that cost fills the zero-weight tasks with a common probability solved from the same linear cost equation. The all-policy endpoint sets every p=1. These are arithmetic conventions, not benchmark-fitted thresholds; every measured task here has positive weight.
4. **Split across episodes.** At reset draw one phase u uniformly in [0,1). For reproducibility, hash UTF-8 `20260928:task_key:episode_key` with SHA256, interpret its first eight bytes as an unsigned big-endian integer, and divide by 2^64. Use stable manifest keys independent of the arm; here they are decimal task ID and original init, so paired arms share phases. On vision anchor k=0,1,…, call iff `floor(u+(k+1)p_t)>floor(u+k p_t)`. Commit either the resulting policy chunk or A's cached chunk for the same b quanta. Do not force the first anchor to CALL. Reset the counter each episode; no calls occur in blind tails. No inherited reactive guard override is active in this diagnostic scheduler. The uniform comparator replaces all weights by 1 and otherwise uses the identical algorithm and common phases.
5. **Quota interpretation.** In any prefix of a_t realized anchors, paid calls differ from a_t p_t by less than one. Budget scales with actual episode progress; a short episode receives fewer calls. Report predicted/realized calls per task and absolute episode counts, not only aggregate m. The library equation predicts expected IR; shifted task durations and traffic can move realized IR. It is **not a hard fleet-wide IR cap**. Re-estimating durations from unlabeled deployment logs is permissible but is not included in these predictions or eight tests.

The rule is portable in the limited, testable sense that its **serving rates use only the deployed library, metadata, and profiling**, not evaluation success labels. It allocates more of a chosen budget to poorly covered tasks. It does **not** infer the globally optimal total budget from d: a fixed target IR spends a similar total amount even on a strong bank. The measured A/B/gap tables advise the total budget for these cells; a portable automatic ε→ρ mapping remains unvalidated. Library quality cannot identify unseen failure recovery or policy benefit from successful demos alone.

The exact per-task p values, original A method/kwargs, commitment settings, seed and counter rule are in `test_plan.json`. They are design specifications, not an installed server plugin. Replacing scheduled locations with a future Q3 placement rule changes the response curve and requires its own calibration; no location learner is designed here.

### Predictions per cell at ρ=.15

**SR columns below are hypotheses**, computed as the mean of `A_t+(P_t−A_t)[1−(1−p_t)^γ]`, using signed measured per-task gaps and γ fitted without that model/suite cell. The rates p_t are library-only, but the SR forecasts explicitly borrow endpoint outcomes. IR/M forecasts retain A's observed task lengths and anchor counts; this is why the realized-cost forecast can differ from the library target. The observed 6.3-point held-cell model error is a diagnostic warning, not a confidence band.


| Cell | Lib | Predicted m5 | Calls/episode | Predicted IR | Predicted SR | Uniform-budget SR |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10 | 50 | 0.089 | 5.92 | 0.151 | 0.781 | 0.771 |
| pi05_l10 | 500 | 0.088 | 5.23 | 0.151 | 0.858 | 0.853 |
| pi05_spatial | 50 | 0.083 | 2.05 | 0.148 | 0.888 | 0.890 |
| pi05_spatial | 500 | 0.086 | 1.81 | 0.151 | 0.984 | 0.983 |
| groot_l10 | 50 | 0.092 | 6.65 | 0.152 | 0.750 | 0.740 |
| groot_l10 | 500 | 0.091 | 5.42 | 0.152 | 0.857 | 0.848 |
| groot_spatial | 50 | 0.087 | 2.18 | 0.150 | 0.897 | 0.894 |
| groot_spatial | 500 | 0.088 | 1.92 | 0.150 | 0.955 | 0.954 |


The small forecasted allocation advantage is less than the historical l10 repeat spread. In particular the spatial-500 forecast decreases GR00T SR: this is a deliberate negative control against the notion that a chosen budget should always be spent. This rule is not ready to replace the measured B/G10 controllers merely because its cost is easy to calibrate.

### An identifiable episode split and an ε budget benchmark

There is one budget interpolation we can evaluate without assuming unseen mixed trajectories: choose A or pure P10 **once at episode reset**, independently of the episode outcome, with per-task probability p_t. Then `E[S_t]=(1−p_t)S_A,t+p_t S_P,t`; expected N/V/M are the same mixture of endpoint means. IR is the **ratio of the mixed expected counts**, not the average endpoint IR. This gives an empirically identifiable episode-routing benchmark; it generally sacrifices more SR than committed rescue at the same cost. Using the library-calibrated p_t at ρ=.15 yields:


| Cell | Lib | Endpoint-mixture SR | Expected IR | Expected m5 |
| --- | --- | --- | --- | --- |
| pi05_l10 | 50 | 0.745 | 0.140 | 0.075 |
| pi05_l10 | 500 | 0.844 | 0.147 | 0.083 |
| pi05_spatial | 50 | 0.862 | 0.140 | 0.074 |
| pi05_spatial | 500 | 0.983 | 0.150 | 0.086 |
| groot_l10 | 50 | 0.659 | 0.138 | 0.074 |
| groot_l10 | 500 | 0.841 | 0.149 | 0.088 |
| groot_spatial | 50 | 0.882 | 0.143 | 0.080 |
| groot_spatial | 500 | 0.960 | 0.152 | 0.090 |


These are computed expectations over the existing endpoint outcomes, **not new closed-loop runs**. They assume reset selection leaves each controller's outcome distribution unchanged. Sampling a finite lottery adds assignment noise; the expectation does not remove run-seed uncertainty.

For an ε=.02 comparison, I also solved the empirical routing problem over available A/B/P10 controllers (including the newly completed GR00T l10-500 guard B). The left budget uses one episode-mixture probability shared by all tasks. The right budget lets each task have its own A/B/P probabilities, minimizing the ratio of expected cost to expected control quanta subject to aggregate SR≥P10−.02. Fractional linear optimization uses bisection on IR and a linear program over per-task simplex weights. These are **in-sample achievable mixtures under stationarity**, not optimal within-episode budgets or deployable predictions. Task-wise optimization reuses outcome labels and is intentionally an optimistic allocation benchmark.


| Cell | Lib | Target SR | Shared-mixture IR / m5 | Task-oracle IR / m5 |
| --- | --- | --- | --- | --- |
| pi05_l10 | 50 | 0.882 | 0.406 / 0.389 | 0.205 / 0.152 |
| pi05_l10 | 500 | 0.882 | 0.310 / 0.275 | 0.103 / 0.031 |
| pi05_spatial | 50 | 0.966 | 0.401 / 0.383 | 0.218 / 0.166 |
| pi05_spatial | 500 | 0.966 | 0.078 / 0.000 | 0.078 / 0.000 |
| groot_l10 | 50 | 0.846 | 0.452 / 0.444 | 0.250 / 0.206 |
| groot_l10 | 500 | 0.846 | 0.126 / 0.061 | 0.087 / 0.015 |
| groot_spatial | 50 | 0.918 | 0.193 / 0.138 | 0.119 / 0.051 |
| groot_spatial | 500 | 0.918 | 0.076 / 0.000 | 0.076 / 0.000 |


The task allocation opportunity is large at the point estimate—π0.5 l10-500 needs 15.69 calls/episode under the shared mixture versus 1.75 under the task oracle—but the oracle is overfit. To test the tempting “a tiny pilot is enough” alternative, I used five paired A/P initializations per task to estimate positive task gaps and policy-call cost, solved a fractional-knapsack allocation for ε=.02, and evaluated on the other 45. Across ten disjoint five-init pilot folds in all eight cells, **43/80** held-out evaluations exceed the target loss. Those folds are dependent and are a diagnostic, not 80 independent trials. See `pilot_validation.json`.

An honest future ε interface requires a separate calibration stage. One exact conservative option is paired A/P pilot outcomes Z=P−A∈[−1,1], task gap bound `U_t=min(1,max(0,mean(Z_t)+sqrt(2 log(T/.05)/n_t)))`, with T task strata and n_t independent pilot pairs. Choose A/P routing probabilities to minimize expected pilot-measured cost subject to `Σπ_t(1−p_t)U_t≤ε`; independent evaluation remains required. This is a one-sided union-bound Hoeffding construction, with confidence .95 fixed as a statistical design choice. It is valid for stationary independent calibration episodes under the specified task distribution, not for arbitrary shift. Five pairs/task make the bound essentially vacuous here, so it defaults toward pure inference. This conservative method adds labeled paired rollouts beyond the library and is not the proposed eight-arm serving rule. No tiny pilot has been shown sufficient for a tight ε.

## 4. Closed-loop test plan — exactly eight candidate arms, coordinator only

Four allocation pairs test whether library weighting is useful at the same library-predicted IR=.15. Each uses the existing A retrieval configuration and ten-control cache/policy commits; the **only within-pair difference** is `w_t=d_t/mean(d)` versus all w=1. All use seed 20260928, common hashed per-episode phases, full stage 1, one blind tail block, no guard override, and the exact p arrays in `test_plan.json`. The coordinator should use the original paired 500 `(task, init)` evaluation manifest for each cell and hold policy sampling seeds common within each pair. No new library data are acquired. These are proposed specifications only; no experiments were launched.


| Arm | Cell/lib | Allocation | Forecast SR @ IR | Paired comparison |
| --- | --- | --- | --- | --- |
| 1 | pi05_l10/50 | uniform | 0.771 @ 0.149 | q2_budget_pi05_l10_50_library_weighted |
| 2 | pi05_l10/50 | library_weighted | 0.781 @ 0.151 | q2_budget_pi05_l10_50_uniform |
| 3 | pi05_l10/500 | uniform | 0.853 @ 0.150 | q2_budget_pi05_l10_500_library_weighted |
| 4 | pi05_l10/500 | library_weighted | 0.858 @ 0.151 | q2_budget_pi05_l10_500_uniform |
| 5 | groot_l10/50 | uniform | 0.740 @ 0.149 | q2_budget_groot_l10_50_library_weighted |
| 6 | groot_l10/50 | library_weighted | 0.750 @ 0.152 | q2_budget_groot_l10_50_uniform |
| 7 | groot_spatial/500 | uniform | 0.954 @ 0.150 | q2_budget_groot_spatial_500_library_weighted |
| 8 | groot_spatial/500 | library_weighted | 0.955 @ 0.150 | q2_budget_groot_spatial_500_uniform |


The four pairs address: π0.5 l10-50, whether weighting recovers part of its large deficit; π0.5 l10-500, whether weighting directs calls toward the few residual weak tasks; GR00T l10-50, whether the budget shape transfers to the weaker policy/library cell; GR00T spatial-500, whether coverage weighting avoids spending harmful calls when A already exceeds pure policy. Spatial-50 and GR00T l10-500 still have explicit forecasts above, but are not additional arms in this plan.

Use paired bootstrap SR contrasts, total calls, m5, IR, and task-level changes. Also compare every arm with its stored A/B/P10 reference, including the full cell-specific gap and cost. Accept a claimed allocation improvement only if its paired SR interval is above zero and there is no material cost increase; if realized IR differs, compare total work and report the mismatch instead of attributing the result to allocation alone. With only 500 pairs and predicted improvements around one point or less, these tests may reject large harms/model errors without resolving a small advantage. An unresolved interval is not equivalence; promotion would require a separately planned repeat, not an extra arm silently added here.

## 5. Risks and falsification

1. **Proxy transfer:** the state score already fails to beat a mean-gap predictor on held-suite error, and barely ranks the residual gap under quantile arms. Equal-cost uniform allocation matching/beating the weighted arm falsifies this particular library weighting, even if the aggregate budget is useful. The full Q1 score can replace d later only with fresh validation.
2. **No budget-only SR law:** the completed GR00T guard/periodic comparison and K5 downstream-call result undermine a universal SR=f(m). Systematic errors beyond the measured 6.3-point worst held-cell error falsify this response model for the new scheduler; errors smaller than that still cannot support ε=.02.
3. **Successful-library selection:** state support among demonstrations does not reveal irreversible failures, intervention timing, or cases where the policy itself is weaker than A. More coverage does not prove higher SR, as the spatial demo-size curve shows. No rule using d alone can certify zero necessary calls or equality to policy.
4. **Duration/traffic shift:** finite-episode anchor counts and longer failures change achieved IR. Check realized per-task H/A/M against calibration. The quota has a per-prefix call-count bound, not a universal IR guarantee under shifted episode lengths or task mix.
5. **Task-oracle and pilot optimism:** the ε frontier is selected on evaluation outcomes; it is not an externally validated policy. The 43/80 pilot violations falsify the claim that the tested five-pair calibration reliably meets ε=.02. Deployment needs more calibration or a loose target, not rebranding the oracle as library-only.
6. **Run noise and data integrity:** the stock l10 repeat spread is 3.2 SR points. Small new gains need repeat evidence. Two corrupt legacy costs remain excluded. The snapshot freezes one completed R6 guard result; later completions must be a new analysis snapshot, not selectively appended favorable outcomes.
7. **Portability boundary:** the budget formula uses discovered strata, library state scales, action-commit metadata and profiled costs. The current A/B controller itself still has the LIBERO-specific conventions listed in the R6 findings brief; this work does not claim to have generalized those controllers. No task ID, gripper sign, fixed episode horizon, or benchmark-specific state dimension appears in the proposed allocation formula.

**Practical decision:** use the measured strong-library spatial A points as zero-call candidates, preserve evidence-supported rescue where it helps, and test library-conditioned *allocation* at a declared IR before claiming a portable ε budget. Matching pure inference with minimal calls is plausible through task selectivity, but the current library-only proxies do not yet calibrate that promise.


## 6. Data we want but do not have

This follow-up ranks **Q2 budget-identification requirements** for the owner's one superset profiler. It does not build the profiler, launch experiments, or replace the eight-arm proposal above. “Missing” means absent as a complete, comparable dataset in this report's frozen audit; individual historical arms already contain some of the requested fields. The numerical planning calculations below are in [data_needs_precision.json](data_needs_precision.json), generated from the audited episode records by:

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  exp/offline_search/rounds/r06/ideation_Q2/data_needs_precision.py
```

**Already covered by the plan:** paired cache/shadow-policy chunks at visited vision anchors, the active retriever's internals, guard inputs/outputs, randomized anchor assignments and their propensities, and joins to episode outcomes. These are valuable new measurements; I am not requesting duplicate copies. They provide same-state action disagreement and randomized *local* treatment evidence. They do not reveal outcomes after unchosen chunks, the value of a different total-budget policy, success under another library, or performance on an unseen robot. Periodic snapshots preserve an opportunity to collect counterfactual continuation outcomes later; snapshots alone are not those outcomes. Nothing here requires full policy shadows at blind control steps while the commitment length remains fixed.

**Cost distinction:** injecting an already computed shadow chunk adds no policy forward **relative to this profiler**. Computing that shadow at every anchor still pays full-policy inference at every anchor. Log profiler acquisition cost separately from the counterfactual serving cost that would count only executed MISSes. Treat additional rollout episodes and branch suffixes as paid data acquisition even when they reuse the same instrumentation.

### Precision conventions used in the ranking

The useful unit for episode SR is an episode outcome, with matched initializations and repeats handled as clusters; anchors are not independent SR samples. For a paired outcome difference D, the planning approximation for a two-sided 95% interval of half-width h is `n ≈ ceil(1.96² Var(D)/h²)`. These are **conditional, normal-approximation planning estimates**, not guarantees for a new budget controller. They omit multiplicity, importance-weight variance and new-task/library uncertainty. Recompute variance after the pilot and use clustered inference; do not simply multiply the number of anchors by the number of episodes.

| Historical B/G10−A contrast | Measured paired variance | Approximate half-width at 500 pairs | Pairs for ±2 SR points | Pairs for ±5 SR points |
| --- | --- | --- | --- | --- |
| π0.5 l10, 50 | .21305 | 4.05 pp | 2,047 | 328 |
| π0.5 l10, 500 | .13282 | 3.19 pp | 1,276 | 205 |
| π0.5 spatial, 50 | .10302 | 2.81 pp | 990 | 159 |
| π0.5 spatial, 500 | .02405 | 1.36 pp | 231 | 37 |
| GR00T l10, 50 | .29449 | 4.76 pp | 2,829 | 453 |
| GR00T l10, 500 | .17034 | 3.62 pp | 1,636 | 262 |
| GR00T spatial, 50 | .15761 | 3.48 pp | 1,514 | 243 |
| GR00T spatial, 500 | .03593 | 1.66 pp | 346 | 56 |

For an illustrative paired variance of .20, ±5 points requires **308 pairs** and ±2 points **1,921 pairs**. Detecting a 3-point difference with two-sided 5% type-I error and 80% power requires approximately **1,745 pairs**. Demonstrating noninferiority with a 2-point margin, one-sided 5% type-I error, 80% power and true difference zero requires approximately **3,092 pairs**: precision and noninferiority power are different requirements. A pair costs two rollouts unless a valid endpoint can be shared. Unpaired comparisons lose the covariance benefit: ±2 points requires approximately **4,802 episodes per condition** in the worst Bernoulli case, or **1,729 per condition** if both SRs are .90. All of these numbers were computed, not assumed to be the actual future variance.

The existing 50 initializations per task give only about ±12.4 points for a task contrast with variance .20. Replaying those same 50 initializations under many policy seeds can improve conditional seed-averaged precision, but does not create new independent scene initializations or support new-task generalization. If the benchmark cannot supply more initializations, report that estimand explicitly and model initialization and seed variation separately.

### Ranked requests

#### 1. Assigned whole-episode dose and an actual budget-response surface

- **Quantity / granularity:** per-episode randomized budget-policy assignment, its assignment probability, and the resulting joint distribution of success, actual control duration, executed policy calls and inference cost, indexed by task and library. The assignment is an ex-ante treatment; realized total M is an outcome. We lack a comparable randomized SR/cost curve spanning A through near-policy behavior under fixed cache/policy commitment and fixed placement family.
- **Answers:** how many MISSes are needed to close a fraction of A's gap; diminishing returns; whether an ε target requires small, medium or nearly pure-policy spending. A regression on realized m cannot answer this causally because difficult/long episodes themselves consume more calls.
- **Measurement / cost:** within the single profiler job, assign an episode-level anchor CALL probability before execution and retain it throughout that episode, then perform the planned anchor randomization with that probability. Suggested **screening strata**, not fitted constants, are `p_anchor ∈ {0, 1/8, 1/4, 1/2, 1}`. Keep commitment and override semantics fixed. With two-quanta commitment these correspond asymptotically to `m5 ∈ {0, 1/16, 1/8, 1/4, 1/2}`; measure finite-episode values rather than imposing them. A sampled hard cap is a distinct budget policy and must be separately identified if we later want a cap. Log task-stratified assignment probabilities, rate/cap definitions, and all deviations. This adds almost no instrumentation cost beyond planned randomization; additional episodes cost complete shadow-instrumented rollouts.
- **Required scale:** **500 episodes per dose per cell** is a useful coarse screen, not a 2-point guarantee. Five doses across the eight current model×suite×library-size cells are **20,000 episode slots**. This is a data-volume requirement, not a request to launch separate experiments. Matched initializations across doses can reduce variance; the pure-policy endpoint can be shared across library sizes only after checking that the served policy, sampling, timing and side effects are identical. If constrained, prioritize the two l10-50 model cells, while acknowledging that this does not identify strong-library budgets. Refine only the frontier region using the precision table.
- **Still missing from the stated plan:** a fixed randomization probability at every anchor samples one behavior policy, not a precise whole-dose curve. Sequential importance weighting is theoretically possible with support but can be unusably variable. For an **illustrative fixed 30-anchor trajectory** under p=.25, the all-CACHE path has probability `.75^30 = .0001786`, and the all-CALL path `.25^30 = 8.67e−19`. These are arithmetic examples, not measured episode lengths. Direct endpoint strata are substantially more useful than hoping a dense anchor table evaluates them accurately.

#### 2. Effective treatment probabilities, availability, and sampling provenance

- **Quantity / granularity:** per-decision pre-randomization eligibility, budget remaining, anchor index, nominal and **effective probability of the action actually served**, sampled assignment, served action/source, every override/cancellation, and immutable episode/attempt/decision IDs. Also retain separate simulator, policy, assignment and controller RNG identifiers; chunk identifiers/hashes and policy/library/configuration versions; and whether a shadow computation mutates any future sampling or controller state.
- **Answers:** whether the dose-response estimates are causally interpretable and whether measured MISS quotas correspond to the intended policy. This is a prerequisite for all budget estimates, not another quality score.
- **Measurement / cost:** extend the already planned propensity record to include the complete precedence chain. A guard override, depleted budget, unavailable policy, deadline or dropped request can turn a nominal p into an actual probability of zero or one. Record those deterministic cases and never claim support for an unavailable action. Draw assignment independently of the shadow sample unless dependence is deliberate and logged. Maintain separate RNG streams or immutable per-anchor policy seeds so computing unused shadows does not silently change later policy samples. Cache the exact shadow draw selected for execution; do not resample it on CALL. Scalar metadata and chunk hashes add logging/CPU work and no policy inference; byte and timing overhead must be measured, not invented.
- **Required scale:** collect this on **every decision in every cell**; it is an integrity requirement rather than a large-n estimator. Before trusting outcomes, a suggested engineering pilot is **20 episodes per cell**, covering each assignment stratum and, where naturally encountered, guard/budget/termination overrides. This tests implementation paths, not the frequency of rare errors. The legacy conflicting records in this report are sufficient reason to require an authoritative accepted-attempt join and explicit duplicate checks.
- **Plan status:** propensities and guards are planned; post-override probabilities, unavailable treatments, RNG side effects and chunk identity are not explicitly promised and are the additions requested here.

#### 3. Paired continuation outcomes for additional calls and remaining budgets

- **Quantity / granularity:** for a sampled **pre-treatment** state, success and remaining total calls/controls/cost under two different remaining-budget continuations, with the suffix policy and evaluation horizon explicitly specified. Include the continuation with one more permitted call and the continuation without it. For a smaller subset, compare multiple remaining budgets to identify call interactions and saturation. These are per-checkpoint potential outcomes clustered by source episode, not action-error labels.
- **Answers:** the marginal value of another call after previous spending, when a per-episode cap becomes harmful, and whether an early call saves downstream calls. K5's −.238 total-call effect at 500 shows why an isolated call and the final budget need not move in the same direction.
- **Measurement / cost:** periodic simulator snapshots are only the starting point. Save/restore the simulator **and** controller state, pending committed tail, action history, guard counters, task/termination state, RNG state/keys, and pending request semantics needed for continuation equivalence. Verify restoration by replaying the same continuation before comparing different budgets. Select checkpoints before treatment with recorded sampling probabilities; retaining only eventual failures would bias population estimates. Branches need later simulation and policy inference at their newly visited states; the original shadow chunk only avoids recomputing the branch's first policy action. Cost is approximately the sum of executed branch suffixes, plus snapshot serialization/storage and restore overhead, all to be measured. No snapshot size or restore-time estimate is available in the current data.
- **Required scale:** first use **50 distinct source episodes** as a suggested restoration-feasibility audit. For outcome screening, use at least **500 distinct source episodes per selected cell and budget contrast**, initially π0.5 and GR00T l10-50; then extend to their 500 banks and a strong spatial cell before transferring the conclusion. With variance .20, 500 paired independent continuations would have about ±3.9-point precision; ±2 points would need about 1,921 independent source units. Multiple checkpoints or suffix seeds from one root episode do not multiply that independent count. Start with two continuations per sampled checkpoint; expand the budget ladder only where saturation is unresolved.
- **Still missing from the plan:** actual branch rollouts, restore fidelity, full controller/RNG snapshots, and the joint success/**downstream cost** outcome. Without these, anchor randomization estimates local intervention effects under the behavior suffix; it does not automatically identify a different hard-cap policy.

#### 4. Independent per-task budget calibration and tight-target validation

- **Quantity / granularity:** per-task estimates and uncertainty for A→P, A→budget-policy, and adjacent-budget SR/cost differences, with a calibration/validation split and independent seed-block IDs. Also needed is the variance decomposition over initializations, policy seeds, run blocks and tasks. Existing per-task outcomes exist; precise, independently validated budget assignments do not.
- **Answers:** which tasks should receive zero/few/many calls, whether the task-oracle saving survives estimation, and whether the single ε knob actually meets its advertised loss. It also determines how the collection budget itself should be split across tasks.
- **Measurement / cost:** reuse the outcome joins and assigned dose strata from item 1. Preserve reset/first-anchor features before the first served action and a stable initialization/block ID, so an episode-specific allocation uses ex-ante risk rather than post-treatment disagreement or eventual duration. The features should come from the planned retrieval logging; the missing quantity is their held-out relationship to budget benefit. Reserve evaluation initializations/seed blocks before fitting quality-to-budget maps; do not spend the same outcomes selecting tasks and validating the selected allocation. Suggested initial balancing is equal counts per task/dose, followed by additional independent data where uncertainty about the frontier is greatest. Record adaptive allocation probabilities if collection becomes adaptive. Use at least **three independently seeded collection blocks** as an engineering minimum to expose run variation, not as a precise estimate of its distribution. This requires extra rollouts but no new neural measurement beyond the superset profiler.
- **Required scale:** **50 episodes per task/dose** reproduces today's coarse resolution. For a **task-specific** ±5-point paired comparison, an illustrative variance .20 requires **308 pairs per task**; ±2 points requires 1,921 pairs per task. For an **aggregate** ε=.02 promise, the task-stratified estimator may need fewer episodes in easy strata, but 500 total episodes is not generally enough; the table and 3,092-pair noninferiority illustration give the relevant scale. Multiple comparisons or adaptive selection increase requirements. Do not apply the very small high-SR spatial plug-in sample estimates as a stopping rule before seeing enough discordant outcomes. A profiler cannot turn the present 50 initializations/task into a broad new-scene claim merely by rerunning them.
- **What it could change:** the 43/80 five-pair-pilot target violations could be replaced by a demonstrated calibration curve and an honest ε interface. More precise task means could also erase some apparent zero-call tasks or much of the optimistic task-oracle advantage.

#### 5. Quality × budget effects across independent libraries

- **Quantity / granularity:** per-library quality outputs from Q1, immutable membership/provenance/fit hashes, and per-task randomized budget outcomes for multiple bank sizes **and independently sampled banks at a fixed size**. We lack clean mixed-budget curves at the 100/200/300 sizes and uncertainty over which demonstrations happened to enter a bank.
- **Answers:** whether required MISS budget is a function of library quality rather than merely this bank's size/source, and whether a library-only calibration transfers to a new bank.
- **Measurement / cost:** vary the served library at episode reset, blocking on model, suite/task, initialization and assigned dose. Keep the retrieval representation fixed when measuring candidate-coverage effects; label refitting as a separate intervention when it is part of deployment. Log Q1's library score before rollout and never refit it on held-out outcomes. Reusing the current shadow policy costs no additional policy forward for alternate retrieval computations at the **same** state, but serving another bank changes future states and requires its own outcome episodes. Request retriever inputs sufficient to rescore another bank offline if “every internal” does not already include them: raw stage-1 keys or losslessly recoverable references, valid state/schema and fit version, not only the winning top-k. Two float32 32,768-dimensional keys are **262,144 bytes/anchor**, or **7.5 MiB for an illustrative 30-anchor episode**, before state/actions/metadata; this is a storage calculation, not a compression estimate. CPU retrieval/refit overhead is unmeasured. Any bank acquired from profiler episodes needs explicit source IDs and a disjoint evaluation split.
- **Required scale:** cover the **eight current model×suite×size settings**, then at least **four independently constructed held-out banks**, one per model/suite at a prespecified size, for a minimal bank-transfer screen. A stronger design with three banks at each current setting has **24 library settings**; three banks is a screening choice, not a precision guarantee. Use roughly **500 episodes per selected dose/bank** for coarse curves, then allocate by measured variance. Intermediate-size budgets need actual served-bank outcomes; alternate-bank retrieval on logged states cannot substitute for them. There is no defensible numerical power claim for library-level transfer from the current single-bank-per-setting evidence.
- **Still missing from the plan:** random library assignment, independent bank realizations and held-out bank outcomes. The profiler's all-internals promise may cover the feature logging; it does not cover those outcomes.

#### 6. Deployment-equivalent cost and actual control denominators

- **Quantity / granularity:** per-control executed steps and termination/truncation timestamps; per-decision counts for shadow-only policy evaluations, executed fresh MISSes, cache anchors, cache tails, policy tails, canceled/retried work and synchronization; per-episode total work/duration. For translating IR to hardware cost, also need stage/search/queue/serialization timing at the deployment batch/concurrency/camera mode, including the profiler's own overhead.
- **Answers:** how many policy calls were actually bought, whether a nominal IR cap holds, expected calls per successful episode, and whether an apparent efficiency gain comes from longer failed episodes or terminal partial chunks. Existing owner-price ratios and nominal N·L counts are not a complete physical-control or profiler-free serving-cost ledger.
- **Measurement / cost:** use authoritative client/simulator executed-control acknowledgments and timestamped request/action IDs, not requested chunk length alone. Maintain separate **profiler compute**, **executed-policy dose**, and **counterfactual serving compute** ledgers. The full-shadow run cannot directly measure the latency of a deployment that skips those forwards. Measure stage costs under a representative isolated calibration and retain whether inference or logging stalls alter observation age/control timing. Instrumentation consumes CPU/storage and possibly synchronization time; benchmark it rather than assuming zero. This is largely scalar logging and no extra policy inference; an actual shadow-disabled parity/timing condition requires additional replay or rollout work by the coordinator.
- **Required scale:** exact counters belong on **all episodes**. A suggested timing pilot is **1,000 anchors from at least 100 episodes per hardware/model/runtime configuration**, covering warmup and ordinary load; use the observed timing variance to choose `n ≥ (1.96·SD / desired_mean_error)^2`, with episode clustering. There is no measured millisecond precision claim yet. For the existing π0.5 l10-50/500 B−A call-count variances, approximately **302/166 paired episodes** suffice for a normal-approximation ±1-call mean difference; this does not certify p95 latency or rare failed-episode costs.
- **What it could change:** the numerical IR forecasts and quota-to-IR calibration, particularly small terminal/short-episode corrections. It does not by itself validate the SR response model.

#### 7. Budget exhaustion, censoring and recoverability after a failure signal

- **Quantity / granularity:** per-decision remaining assigned credits/eligibility and budget-exhaustion events; per-episode terminal reason separating success, environment time limit, controller abort, infrastructure error and any task-native irreversible failure indicator. For an episode called “hopeless,” the missing quantity is success/cost under a larger remaining budget at the **same evaluation horizon**.
- **Answers:** how to split spending across episodes, whether calls after a cap are wasted or necessary, and whether long-failure call concentration justifies stopping. Aggregate failure labels and stuck counters do not establish irreversibility.
- **Measurement / cost:** log the explicit deadline/remaining control horizon and the reason for every denied call. Avoid censoring the experiment merely because an assigned call budget ran out: continue with the specified fallback and preserve the outcome. Collect task-native terminal causes when available, without inventing a benchmark-specific gripper heuristic. Estimate recoverability using randomized larger-budget continuations or the branches in item 3; those continuations cost actual suffix rollouts. The logging is cheap; the recoverability label is not obtainable from one shadow chunk.
- **Required scale:** collect cause/credit fields on **all eight cells**. For each practically important cause stratum, a few dozen events are only a screen; target approximately **100 independent source episodes** for a coarse conditional rescue estimate, then use observed paired variance. At an **assumed 1% event rate**, merely observing 50 events needs **5,000 episodes in expectation**; that is not a success-effect precision guarantee. With zero observed events, the exact one-sided 95% binomial upper rate is about **.60% at n=500**, but **5.82% at n=50**. Branch-enriched samples must retain selection probabilities and must not be presented as natural event frequencies.
- **What it could change:** a per-episode maximum could become defensible. Today, spending 55.5% of π0.5 l10-50 B calls on eventual failures is an accounting fact, not evidence that cutting those calls is safe.

#### 8. Policy-sampling noise at the mixed-controller states

- **Quantity / granularity:** for a sampled anchor, repeated independent policy chunks conditional on the same observation, plus sampled-seed provenance and, when needed, continuation outcomes. Existing teacher-floor resampling covers other recorded populations; the planned single shadow draw does not identify policy variance at these mixed-controller states.
- **Answers:** whether a proposed quality-to-budget feature derived from shadow/cache disagreement reflects a reliable deficit or ordinary policy sampling variability. This is needed **if** Q2 uses that disagreement for ε calibration; a randomized whole-dose comparison can be estimated without reconstructing every state's policy distribution.
- **Measurement / cost:** reuse one planned shadow, take one additional independently seeded draw at a preselected sample of anchors, and log both seeds. A suggested sample is **one in 20 anchors**, which adds **5% policy-forward work** relative to the full-shadow baseline if each extra draw costs the same. Two draws screen variability; they do not characterize a multimodal action distribution. Extra continuation comparisons belong to item 3 and add rollout cost. Pure action distance remains a proxy, not a causal SR label.
- **Required scale:** a suggested first audit is **1,000 sampled states per model/suite across at least 200 distinct episodes**, with both library sizes and multiple assigned doses represented. That is a coverage/variance pilot, not an SR power claim. Increase sampling where variance is large; count episodes as clusters. Do not spend the whole profiler budget on repeated shadows before establishing the dose-outcome surface in item 1.
- **What it could change:** confidence in a new shadow-based deficit feature or the amount of shrinkage it needs. It cannot overturn the finding that the currently tested LOEO predictors are inaccurate on their historical validation split.

#### 9. Deployment task mixture, initial-state mixture and episode-cost tails

- **Quantity / granularity:** per-episode task/initial-state sampling probabilities, deployment arrival weights and missing-task/unavailable-library rates; per-task episode-length and call-count distributions under each selected budget. Existing balanced evaluation counts describe the benchmark, not a future workload.
- **Answers:** which tasks should receive a fleet budget, how to convert task rates to expected IR, and how much reserve is needed for long episodes or bursts. The serving rule's equal-task prior is not a measured deployment distribution.
- **Measurement / cost:** obtain workload weights from the caller if known; otherwise record real arrival/task IDs and pre-treatment context with privacy-appropriate identifiers. No extra inference is needed. Preserve raw counts instead of only mean m, and keep scheduling/concurrency metadata if queueing affects cost. A fixed balanced simulator experiment cannot identify the external traffic distribution even if it logs every anchor.
- **Required scale:** with a worst-case Bernoulli task-frequency variance, ±2-point frequency precision needs approximately **2,401 independent arrivals**; ±1 point needs **9,604**, before simultaneous-task corrections or drift. For cost tails, **1,000 episodes per selected cell/budget** yields about 50 observations in the upper 5% in expectation; this is only a tail-screening count, not a p95 cost-error guarantee. No arrival sample is needed if the evaluation objective explicitly remains an equal-task benchmark or the true weights are supplied.
- **What it could change:** the recommended task allocation and aggregate budget, even if every task's SR curve stays unchanged. It need not change a per-task curve.

#### 10. Held-out domain/robot budget calibration

- **Quantity / granularity:** per-library and per-task budget/outcome/cost curves in a held-out benchmark or robot, with its own state schema, commitment metadata, profiled costs and Q1 score calibration. Neither the existing LIBERO data nor a LIBERO-only superset profiler contains this transport test.
- **Answers:** whether “library-calibrated” also means portable, rather than simply executable without hard-coded task IDs. More episodes from the same two suites cannot supply this missing domain axis.
- **Measurement / cost:** run the same instrumentation and budget-assignment contract on at least **one held-out benchmark/robot**, with **two library-quality levels** and at least **three dose strata** (A, an interior rate, matched pure policy). Fit transformations from that domain's library/metadata without reusing LIBERO SR thresholds. These are additional domain rollouts and deployment profiling, not free shadow logging in the current experiment.
- **Required scale:** **500 episodes per dose/library level** is a suggested initial six-stratum screen (**3,000 episode slots**) in that domain, provided its task/initialization coverage makes that meaningful. Use the precision formulas and its measured variance for tight claims. A second domain can falsify portability; it does not establish universal robot transfer. Enough distinct tasks and independent banks are needed to estimate their variance; no defensible task-count power estimate exists yet.
- **What it could change:** the present restriction against transporting a library-only ε mapping. A successful test would expand the evidence to the tested domain, not turn the existing score into a universal SR guarantee.

#### 11. Acquisition and refit cost, if “MISS budget” includes building the bank

- **Quantity / granularity:** per-acquisition-episode policy calls and actual controls, including failed/rejected data; per-library admission IDs, acquisition-source cost and refit/search-build cost; the deployment episode horizon over which to amortize them. Some grow250 counts exist, but this is incomplete for the other banks and future profiler-derived banks.
- **Answers:** whether buying a stronger bank is cheaper than repeatedly spending MISSes during serving. This is conditional on a **lifecycle** objective; it is not needed for the serving-only IR claim above.
- **Measurement / cost:** audit every acquisition attempt and library admission, retain rejected attempts, record refit CPU/GPU time and policy-forward counts, and tag whether data were obtained anyway for profiling. Do not count successful admitted episodes as all the acquisition cost or charge the same forward twice. Bookkeeping is inexpensive; collecting another bank costs its actual acquisition work. Missing historical collection expense cannot be recovered from later serving traces.
- **Required scale:** one complete ledger per actual bank suffices to state that bank's realized cost; uncertainty over future acquisition requires independent banks/batches (suggested minimum **three**, with no precision guarantee). The break-even deployment horizon is supplied or varied analytically, not estimated from 500 evaluation episodes. This requirement can be deferred if the owner explicitly retains serving-only accounting.
- **What it could change:** the economic recommendation between “increase library quality” and “increase inference budget,” without changing the measured serving SR/MISS curves.

### What changes, and what remains established

The highest-value additions to the planned collection are **episode-level dose assignment with direct endpoints (1), an auditable effective-treatment/RNG record (2), independent outcome coverage for tasks/banks (4–5), and separate real-versus-counterfactual cost accounting (6)**. They can all be specified within one profiler implementation; adding them to the schema does not magically supply the required outcome volume. Snapshots plus actual branch continuations (3) are the most direct later route to remaining-budget/cap values. Items 8–11 are lower priority or conditional and should not crowd out the dose/outcome data.

| Current conclusion | Data that could change the decision | What the new evidence must show |
| --- | --- | --- |
| A single γ is too inaccurate for a tight SR-loss promise | 1, 3, 4 | Held-out dose/remaining-budget predictions and an independently validated target-loss rate, with adequate support near the chosen budget |
| Prefer a target-IR interface over a library-only ε guarantee | 1, 4, 5, 10 | A calibrated library-to-gap/budget map meeting a predeclared ε on unseen task/initialization, bank and claimed domain units |
| Strong spatial libraries are zero-call candidates | 1, 4 | A precise SR/cost benefit from calls at those libraries, or failure of A's predeclared noninferiority margin; absence of significance alone is insufficient |
| The task oracle is optimistic; five-pair calibration is unreliable | 4, 5 | Replicated task allocations retaining savings on held-out outcomes; the historical 43/80 violations remain a fact |
| We cannot justify truncating late rescues from failure-call concentration | 3, 7 | Randomized/paired continuation evidence that a cap saves cost while meeting the SR-loss criterion |
| Our quota predicts approximate rather than guaranteed IR under deployment | 2, 6, 9 | Verified control/cost ledgers and duration/traffic calibration at the actual runtime and workload |

New data can improve recommendations and falsify models; it does not retroactively change the audited historical counts, the two excluded legacy costs, or K5's identified effect under its specific guard controller. The warning that commitment, placement, state distribution and duration matter alongside total m also remains: richer data can estimate those dependencies, not make them disappear.
