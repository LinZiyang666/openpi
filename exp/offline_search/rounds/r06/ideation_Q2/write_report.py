from pathlib import Path
import json
import numpy as np

O=Path(__file__).resolve().parent
def load(name):return json.loads((O/name).read_text())
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x,n=3):return f'{x:.{n}f}'
K=load('key_contrasts.json');T=load('task_budgets.json');R=load('arms.json');Q=load('quality_validation.json');E=load('epsilon_frontier.json');M=load('response_model.json');P=load('predictions.json');plans=load('test_plan.json');causal=load('causal.json');extra=load('allocation_diagnostics.json');r6=load('r6_completed_guard.json');V=load('verification.json')

parts=[r'''# R6 Q2 — how much policy inference to buy

The measured answer is **task-, library-, and controller-dependent**. At the 50-episode libraries, B/G10 spends about 0.075–0.144 MISS per five-control slot and buys 5.2–12.4 SR points over A. At 500 episodes, π0.5 spatial needs no calls to match its measured B, and GR00T periodic calls are point-dominated by A. π0.5 l10 still benefits from rescue. A scalar MISS rate cannot identify SR without fixing commitment and call placement.

**Recommendation for R6:** expose a **target IR** knob, calibrate the task allocation and finite-episode call counts from the library, and test it with the eight arms below. Do not expose a library-only “SR loss ≤ ε” promise yet. The simple LOEO score has held-cell gap error around 11 SR points; the best tested gap-response model still misses a held cell by 6.3 points. The proposed allocation is an experiment, not a demonstrated improvement over B. For the currently measured strong spatial libraries, A is the evidence-based zero-call candidate; a target IR is a resource constraint, not evidence that spending it helps.

## 1. Measurement, scope, and reproduction

All work here is offline and CPU-only. No server, worker, chain, port, tmux session, GPU, git operation, or live experiment was started or changed. Writes are confined to this directory and `/tmp/r6_q2/`. The fixed snapshot is `snapshot.json`; a rerun reuses its completed-arm list even if more R6 arms finish later.
''']
parts += [f"Snapshot UTC: **{load('snapshot.json')['utc']}**. The audit includes **{len(R)} arms, {sum(r['n'] for r in R):,} accepted episodes**, and **{V['accepted_decision_count']:,} decision counts**. There are 173 completed production arms from R2–R6 plus four original pure-inference trace arms. Production arms require both a DONE marker and summary; journal terminal attempts are matched before reading server decisions. Smoke runs, `r03_pilot`, incomplete arms, and evaluations other than 500 episodes or the four 250-pair grow250 arms are excluded. This includes all completed requested periodic, guard, quantile, A, B, G10, policy and K5 arms, plus supplemental production methods."]
parts += [r'''
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
''']
parts += [table(['Cell','Library','A SR @ IR','B/G10 SR @ IR','P10 SR','m(B)','MISS/episode','Gap closed'],[[r['cell'],r['lib'],f"{r['srA']:.3f} @ {r['irA']:.3f}",f"{r['srB']:.3f} @ {r['irB']:.3f}",fmt(r['srP']),fmt(r['mB'],4),fmt(r['callsB'],2),f"{100*r['fraction_closed']:.1f}%" if r['fraction_closed'] is not None else 'A > policy'] for r in K])]
parts += [r'''
At 50 episodes, the same affordable interior rate does **not** reach pure inference on l10: π0.5 remains 7.2 points below P10 and GR00T 14.8 points below P10. At spatial-500 the corresponding observed A→B/G10 gains are 0 and −1.2 points. Pure L5 versus L10 is a distinct execution-length intervention and must not be used to infer the value of extra calls. G10's nominal one-in-four paid anchors implies asymptotic m=.125, but its paid first anchor and finite endings produce measured m=.130–.145. The proposed randomized initial phase removes that forced-first-call bias.

The frozen snapshot also contains **GR00T guard B at l10-500**: .864 SR, m=.13155, IR=.18663, M=3,777. Periodic G10 has .828 SR, m=.13094, IR=.18605. Guard B−G10 is +3.6 points, paired interval [0.0, 7.4], McNemar p=.0693; guard B−A is +3.4 [0.0, 6.8], p=.0712. Guard B−P10 is −0.2 [−4.0, 3.6]. This single result is suggestive, not resolved equivalence or superiority. It is a concrete warning against predicting SR from m alone, since the two configurations buy almost the same number of calls. Source: `r6_completed_guard.json` and `r06_paper/r6p1_c10_g_l10_500`.

### Marginal SR per MISS share: diminishing returns is conditional

One unit of slope below is one unit of SR per unit m; equivalently, multiply by 10 to obtain SR percentage points per +.10 MISS share. These are same-family observed secants, not causal derivatives at a state.
''']
sl=load('marginal_returns.json')
parts += [table(['Sequence','Δm','ΔSR pp','ΔSR/Δm','95% conditional interval'],[[r['low']+' → '+r['high'],fmt(r['dm'],4),fmt(100*r['delta'],1),fmt(r['slope']),f"[{r['slope_ci'][0]:.3f}, {r['slope_ci'][1]:.3f}]"] for r in sl])]
parts += [r'''
π0.5 l10-50's periodic slope falls to .297 from approximately .83–.90 at the lower rates; the .90 middle estimate is noisy and does not establish monotone curvature. At l10-500, the CL2→per12 slope is .799, and per12→per8 is .493 with an interval crossing zero. Confidence h70→h50 at l10-50 spends +.1889 m for +5.2 SR points: slope .275. R3 h70/guard-only/B points use different placement rules and cannot be concatenated as a pure dose-response experiment.

The A→B slopes are **1.009, .381, .958, 0** for π0.5 l10-50/l10-500/spatial-50/spatial-500. A→G10 slopes are **.847, −.015, .361, −.083** for the corresponding GR00T cells. Negative slopes are retained. An isotonic fit that deletes them would hide the main warning in the data. The three stock π0.5 l10-500 guard repeats are .864/.850/.832, while realized m increases .102/.109/.117: failures themselves generate extra downstream calls, so regressing episode success on realized M is not a causal call-value estimator.

### Randomized CALL/CACHE: one useful call can reduce total calls

The K5 treatment is one assigned first/third guard landmark, not a randomly assigned total episode budget. I independently matched all 2,000 exported randomized episode Y/N/M totals to this turn's raw audit, then recomputed CALL−CACHE using the complementary replicate pair per `(task, init)`. The 500-pair bootstrap keeps both treatments together. Sources: `causal.json` and R5 Q3's raw-audited assignment exports.
''']
parts += [table(['Library','CALL−CACHE ΔSR [95%]','Δrequests [95%]','Δtotal MISS [95%]','CALL/CACHE m'],[[r['scale'],f"{r['effect']['SR']:+.3f} [{r['ci']['SR'][0]:+.3f}, {r['ci']['SR'][1]:+.3f}]",f"{r['effect']['N']:+.3f} [{r['ci']['N'][0]:+.3f}, {r['ci']['N'][1]:+.3f}]",f"{r['effect']['M']:+.3f} [{r['ci']['M'][0]:+.3f}, {r['ci']['M'][1]:+.3f}]",f"{r['treatments']['CALL']['m']:.5f}/{r['treatments']['CACHE']['m']:.5f}"] for r in causal])]
parts += [r'''
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
''']
parts += [table(['Proxy','Task rows','Spearman(gap)','In-sample R²','Held-scene MAE / null','Held-cell MAE / null','Held-suite MAE / null'],[[r['feature'],r['n'],fmt(r['spearman']),fmt(r['r2']),*[f"{r['cv'][s]['mae']:.3f} / {r['cv'][s]['null_mae']:.3f}" for s in ['scene','cell','suite']]] for r in Q['regressions']])]
parts += [r'''
The descriptive least-squares gap model is `P10−A ≈ −.111686 + .945821·d`. It explains only .213 of task-gap variance. “Held scene” keeps the same suite/task across all sizes and both models together; held cell removes both sizes and intermediate sizes of one model/suite; held suite removes the entire suite. The null predicts the training mean gap. The state score helps the scene/cell split modestly and **loses to the null on the suite split**. Action LOEO error loses to the null on held-cell and held-suite validation.

I also checked the same state proxy against residual policy gaps in all applicable completed families: Spearman is .541 for A (140 task-arm rows), .411 for B (50), .457 for G10 (40), .386 for CL2 (100), .202 for periodic (90), .426 for guard (80), and **.041 for confidence-quantile arms (60)**. These 560 rows share tasks and labels; they are not 560 independent validations. This tests the proxy against existing closed-loop behavior rather than treating offline agreement as SR.

The zero-call demo-size curves reinforce the caveat: π0.5 l10 A is .706/.748/.788/.806/.828 at nominal 50/100/200/300/500, but spatial is .838/.950/.970/**.944**/.982. The current 50 bank has a different source from the nested 100/200/300/500 demos, and 500 also changes kref. State coverage gets better with size without proving monotone closed-loop SR. The 100/200/300 libraries have only zero-call A points and policy endpoints; there are no observed interior MISS-budget points to fit a separate response curve at those sizes.

## 3. Budget models and exact proposed algorithms

### Descriptive gap-closing model — useful forecast, not an SR contract

Hold A, the policy commitment, and a call-placement family fixed. Let Δ=P10−A and x=m5∈[0,.5]. The endpoint-constrained response family is

`S(x) = A + Δ·[1−(1−2x)^γ]`.

For a positive interior gain below Δ, calibrate `γ=log(1−gain/Δ)/log(1−2m_B)`. Then the calls needed to close fraction f of the observed gap are `x(f)=.5·[1−(1−f)^(1/γ)]`. For target loss ε<Δ, use f=1−ε/Δ. This guarantees neither shape nor portability: it interpolates only one interior mixed arm and the two endpoints. We do not fit a positive individual γ when the gain is zero/negative or the policy gap is nonpositive.
''']
parts += [table(['Cell','Lib','γ','m5 for 50% gap','80%','90%','m5 for ε=.02'],[[r['cell'],r['lib'],fmt(r['gamma']),*[fmt(r['m_for_gap_fractions'][str(f)]) for f in [.5,.8,.9]],fmt(r['m_for_epsilon_02'])] for r in K if r['gamma'] is not None])]
parts += [f'''
A shared γ is fit by unweighted squared SR error over positive-gap A/B/G10 cell-size points, bounded to [0,20] with scalar tolerance 1e−10; its fitted value is **{M['gamma']:.4f}**. Holding out the whole model/suite cell gives SR MAE **{M['mae']:.4f}**, maximum error **{M['max_error']:.4f}**, even though the held-out A/P endpoints are supplied. Therefore the tiny-ε extrapolations above are hypotheses, not calibrated guarantees. This model borrows closed-loop outcomes; it is not part of the library-only serving rule.
''']
parts += [r'''
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
''']
alloc=[r for r in M['allocations'] if r['method']=='library_weighted']
parts += [table(['Cell','Lib','Predicted m5','Calls/episode','Predicted IR','Predicted SR','Uniform-budget SR'],[[r['cell'],r['lib'],fmt(r['predicted_m']),fmt(r['predicted_calls_per_episode'],2),fmt(r['predicted_ir']),fmt(r['predicted_sr']),fmt(next(u['predicted_sr'] for u in M['allocations'] if u['cell']==r['cell'] and u['lib']==r['lib'] and u['method']=='uniform'))] for r in alloc])]
parts += [r'''
The small forecasted allocation advantage is less than the historical l10 repeat spread. In particular the spatial-500 forecast decreases GR00T SR: this is a deliberate negative control against the notion that a chosen budget should always be spent. This rule is not ready to replace the measured B/G10 controllers merely because its cost is easy to calibrate.

### An identifiable episode split and an ε budget benchmark

There is one budget interpolation we can evaluate without assuming unseen mixed trajectories: choose A or pure P10 **once at episode reset**, independently of the episode outcome, with per-task probability p_t. Then `E[S_t]=(1−p_t)S_A,t+p_t S_P,t`; expected N/V/M are the same mixture of endpoint means. IR is the **ratio of the mixed expected counts**, not the average endpoint IR. This gives an empirically identifiable episode-routing benchmark; it generally sacrifices more SR than committed rescue at the same cost. Using the library-calibrated p_t at ρ=.15 yields:
''']
parts += [table(['Cell','Lib','Endpoint-mixture SR','Expected IR','Expected m5'],[[r['cell'],r['lib'],fmt(r['sr']),fmt(r['ir']),fmt(r['m5'])] for r in P if r['method']=='library_weighted' and r['target_ir']==.15])]
parts += [r'''
These are computed expectations over the existing endpoint outcomes, **not new closed-loop runs**. They assume reset selection leaves each controller's outcome distribution unchanged. Sampling a finite lottery adds assignment noise; the expectation does not remove run-seed uncertainty.

For an ε=.02 comparison, I also solved the empirical routing problem over available A/B/P10 controllers (including the newly completed GR00T l10-500 guard B). The left budget uses one episode-mixture probability shared by all tasks. The right budget lets each task have its own A/B/P probabilities, minimizing the ratio of expected cost to expected control quanta subject to aggregate SR≥P10−.02. Fractional linear optimization uses bisection on IR and a linear program over per-task simplex weights. These are **in-sample achievable mixtures under stationarity**, not optimal within-episode budgets or deployable predictions. Task-wise optimization reuses outcome labels and is intentionally an optimistic allocation benchmark.
''']
parts += [table(['Cell','Lib','Target SR','Shared-mixture IR / m5','Task-oracle IR / m5'],[[r['cell'],r['lib'],fmt(r['target']),f"{r['best_uniform']['ir']:.3f} / {r['best_uniform']['m5']:.3f}",f"{r['task_oracle']['ir']:.3f} / {r['task_oracle']['m5']:.3f}"] for r in E])]
parts += [r'''
The task allocation opportunity is large at the point estimate—π0.5 l10-500 needs 15.69 calls/episode under the shared mixture versus 1.75 under the task oracle—but the oracle is overfit. To test the tempting “a tiny pilot is enough” alternative, I used five paired A/P initializations per task to estimate positive task gaps and policy-call cost, solved a fractional-knapsack allocation for ε=.02, and evaluated on the other 45. Across ten disjoint five-init pilot folds in all eight cells, **43/80** held-out evaluations exceed the target loss. Those folds are dependent and are a diagnostic, not 80 independent trials. See `pilot_validation.json`.

An honest future ε interface requires a separate calibration stage. One exact conservative option is paired A/P pilot outcomes Z=P−A∈[−1,1], task gap bound `U_t=min(1,max(0,mean(Z_t)+sqrt(2 log(T/.05)/n_t)))`, with T task strata and n_t independent pilot pairs. Choose A/P routing probabilities to minimize expected pilot-measured cost subject to `Σπ_t(1−p_t)U_t≤ε`; independent evaluation remains required. This is a one-sided union-bound Hoeffding construction, with confidence .95 fixed as a statistical design choice. It is valid for stationary independent calibration episodes under the specified task distribution, not for arbitrary shift. Five pairs/task make the bound essentially vacuous here, so it defaults toward pure inference. This conservative method adds labeled paired rollouts beyond the library and is not the proposed eight-arm serving rule. No tiny pilot has been shown sufficient for a tight ε.

## 4. Closed-loop test plan — exactly eight candidate arms, coordinator only

Four allocation pairs test whether library weighting is useful at the same library-predicted IR=.15. Each uses the existing A retrieval configuration and ten-control cache/policy commits; the **only within-pair difference** is `w_t=d_t/mean(d)` versus all w=1. All use seed 20260928, common hashed per-episode phases, full stage 1, one blind tail block, no guard override, and the exact p arrays in `test_plan.json`. The coordinator should use the original paired 500 `(task, init)` evaluation manifest for each cell and hold policy sampling seeds common within each pair. No new library data are acquired. These are proposed specifications only; no experiments were launched.
''']
parts += [table(['Arm','Cell/lib','Allocation','Forecast SR @ IR','Paired comparison'],[[i+1,r['model']+'_'+r['suite']+'/'+r['library'],r['allocation'],f"{r['predicted_sr']:.3f} @ {r['predicted_ir']:.3f}",r['paired_with']] for i,r in enumerate(plans)])]
parts += [r'''
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
''']
if (O/'DATA_NEEDS.md').exists():
    parts.append((O/'DATA_NEEDS.md').read_text().rstrip())
(O/'REPORT.md').write_text('\n\n'.join(parts)+'\n')
print('wrote REPORT.md',sum(len(x.split()) for x in parts),'words')
