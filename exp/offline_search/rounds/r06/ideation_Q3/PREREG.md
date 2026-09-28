# Q3 pilot preregistration: where to spend a policy call

Registered 2026-09-28, before reading any pilot outcomes. Only P3 documentation, source code, CSV headers from the designated smoke fixture, and the earlier Q3/G reports have been inspected at registration. No pilot run directory has been inspected. `PREREG_FROZEN.md` and `preregistration.sha256` preserve this initial specification; implementation notes and smoke results will be appended here. The smoke is a plumbing fixture, never scientific evidence.

## 1. Target and experimental units

The objective is the least deployment inference rate (IR) with success rate (SR) close to inference. I use **one percentage point** as the proposed SR noninferiority margin for eventual method acceptance, a design choice rather than an owner-approved tolerance or an achievable pilot precision. Report the full SR–IR tradeoff if that margin is not resolved. Primary inference reference in this campaign is **P10**, policy every anchor with a ten-control commitment. It is not an observed native five-control pure-inference arm. Claims about native pure inference require that additional matched reference.

Source of design: `../p3_profiling/HANDBACK.md`, “One pilot and continuation campaign”; `SCHEMA_V2.md`; `design.py`, `assignment.py`, `campaign_v2.py`, `read_v2.py`. Source of historical identification pattern: `../../r04/k5_rand/estimate.py`. Source of proposed calibrated guards: `../ideation_G/REPORT.md`, §2. The old Q3 report is prior evidence, not pilot data.

Each of eight cells has nine complete matched cohorts and three seed blocks. Pilot: ten tasks × two initializations × three blocks = 60 episodes/cohort/cell, but only **20 task/init clusters**. Model, suite and library define a cell. Task/init is the cluster, containing every anchor, all seed blocks and all compared cohorts. Attempt retries do not create units. Cohort probability is **one**, not 1/9: A/B/P10/fixed-dose cohort contrasts are matched configuration comparisons, not propensity-randomized cohort assignments. Within-anchor coins and independent duration/hold/delay assignments provide randomized causal identification.

The finite target is the scheduled task/init/seed population, with equal task weights and the scheduled proportions of seed blocks (equal in the pilot). Pool neither cells nor continuation controllers for a primary causal estimate. Report block-specific estimates as a stability diagnostic. Long trajectories supply more *opportunities*, not more independent episodes.

`init % 5 == 0` is calibration; the remainder is reserved validation. Pilot has only one distinct init per task in each split. The fixed, preregistered contrasts may be reported on the whole pilot and separately by split. Outcome-driven nomination/training uses calibration only; pilot validation is not used to tune a threshold. Full-campaign validation estimates must also be reported on **continuation-only validation** to make any pilot inspection explicit. No random anchor train/test split.

## 2. Estimands and weights

Let H be pre-assignment history at a vision anchor, Z the free source coin, p its **logged conditional nominal probability**, Y final success, and M/C/V remaining actual MISSes/active controls/anchors, including the current anchor. Costs use `c1 V + (1-c1) M`, with c1=.152 for pi05 and .148 for GR00T, exactly the existing P3 reader's accounting. This is a deployment compute-equivalent model, not profiler wall time. Actual-control-normalized episode IR is that cost divided by active_controls/5; also retain per-request IR. Do not count blind shadows or K4 extras as deployment MISSes. Keep collection dispatch counts separately.

### Randomized factorial excursions (primary)

Eligible roots have cohort=factorial, override=`coin`, both sources available, pre-guard randomization true and 0<p<1. Validate p against logged distance-CDF bin: [0,1/3), [1/3,2/3), [2/3,1] have p=.125,.25,.5 (right-sided ties). Root CALL executes a random first duration in {5,10}, starts a random hold in {1,2,3}, then follows the logged controller. Held anchors draw their own durations: a root duration is **not the duration of every held call**. Root CACHE commits ten controls then follows that same randomized controller. Estimate the value of this **excursion package under its continuation**, not the isolated physical chunk or a wholly different placement policy.

For a pre-action stratum S, the primary score is

`psi(O) = Z O/p - (1-Z) O/(1-p)`.

The opportunity-average effect is `sum(w_episode I[S] psi) / sum(w_episode I[S])`, with w_episode=1/(number of tasks × scheduled episodes in that task). This weights the visitation distribution of the randomized controller. **Never weight each anchor by inverse observed episode length or inverse number of future anchors**: those depend on treatment. No clipping or propensity estimation. Report the vector ΔY, ΔM, ΔC, Δcost, not just an unstable SR/call ratio. Report ΔY/ΔM and its delta-method interval only when the lower 95% bound on ΔM is positive; otherwise mark the ratio unidentified/unstable. Savings with retained success can be attractive even when this ratio is undefined.

Also estimate first free-root effects (at most one root/episode), including terminal nonexecution in Y. This avoids repeated-outcome amplification but targets earlier histories; it is a sensitivity, not interchangeable with the opportunity average. Fixed-dose cohorts get the same local estimator separately for each dose, ten controls and hold one, on their free supported roots. These are replication across continuations, not observations to pool blindly with factorial.

### Duration and hold (primary secondary dimensions of placement)

Among free factorial roots assigned CALL, estimate starting-duration 10 versus 5 using the logged duration probabilities, marginalizing hold; estimate hold3 versus hold1 using inverse logged hold probabilities, retaining hold2 roots in the common denominator with zero contrast numerator. The duration and hold streams are independent of the source coin. These contrasts target CALL-root histories, not all decision histories. Held/due/cooldown/cap/prefix/mandatory-guard anchors with source p=0 or 1 have **no local CALL-versus-CACHE support**. They remain in downstream costs and episode results.

### Delayed single call (primary)

At each episode's first window `scheduled_trigger`, compare delay0 versus delay2 with inverse **delay probabilities**, including delay1 episodes in the common trigger denominator. Also show 0 versus1 and1 versus2 as secondary. A trajectory ending before the scheduled call stays in its assigned delay group. This is the effect of delay assignment among reached triggers, followed by A, not a survivor-conditioned effect among executed calls. Report trigger reach and actual execution. Do not condition on delay and then use the marginal immediate propensity (nominal p/3). No-trigger episodes are absent only from this expressly trigger-conditional estimand, and remain in whole-window-versus-A comparisons.

### Eight primary SR contrasts per cell

1. Factorial free-root CALL package minus CACHE, all roots.
2. The same at `pre_guard_call == true`.
3. The same at `pre_guard_call == false`.
4. Difference of CALL effects: distance LOEO CDF >=2/3 versus <2/3.
5. Difference of CALL effects: pre-guard true versus false.
6. Starting duration10 versus5 among free factorial CALL roots.
7. Hold3 versus1 among free factorial CALL roots.
8. Window delay0 versus2 among first scheduled triggers.

Use two-sided 95% pointwise intervals and **Bonferroni 1-.05/64 intervals** for these eight contrasts across eight cells; missing/unsupported contrasts still consume their reserved slots. Secondary features and comparisons are explicitly exploratory and cannot establish selection by their pointwise intervals. Costs accompany every contrast; SR/call ratios are secondary. Matched B−A, B−P10, factorial−B, window−A and fixed-dose−B episode contrasts establish the observed closed-loop context. A proxy is not trusted because it predicts imitation error: inspect its randomized SR interaction and whether directions replicate under the separate fixed-dose continuations. Failure to replicate or reversal by cell blocks a transfer claim.

### Intervals and small samples

Use a ratio-estimator cluster sandwich: sum each row's weighted influence within task/init across blocks; center cluster influences within task; use the finite-stratum correction K_task/(K_task−1) and Student t with K−T degrees of freedom. Point estimates retain scheduled block weights. Also report task-cluster t intervals as an explicitly different, between-task generalization sensitivity. Pilot split-specific fixed-task intervals are unavailable when every task has one init, rather than fabricated from anchor counts. Report cluster counts, treatment-arm cluster counts, row and cluster Kish weight ESS, maximum inverse weight and support by override. Missing feature values form a reported missing category; they are never zero.

A contrast is eligible to nominate a method only with at least ten distinct task/init clusters contributing each contrasted assignment, cluster Kish ESS at least ten on each side, fixed-task df>=4, and a nonzero estimable variance. These are conservative preregistered analysis guards, not benchmark constants. No claim of finite-sample exact coverage from asymptotic t intervals. No degenerately zero-width binary-outcome interval can establish SR safety.

## 3. Candidate signals, preventive/reactive interpretation

All strata use the *current pre-assignment* record. Never use final trajectory length, eventual failure, future progress, actual call execution after a delay, or post-call guards as a covariate.

| Signal | Frozen definition / calibration | Role and limits |
|---|---|---|
| Coverage | `retrieval.d1_loeo_quantile`; thirds and high >=2/3; raw d1 retained | Library row-LOEO CDF, frozen representation. Not a conformal SR certificate. |
| Neighbour action disagreement | `retrieval.dispersion_rms`; high above q90 | Threshold from outcome-blind A calibration trajectories, per cell/task, episode-equal CDF; export exact samples/hash. It is a few-recorded-trajectories reference, **not mislabeled library LOEO**. Missing A reference disables this gate. |
| Shadow cache/policy disagreement | `distance.commit10_rms`; high above q90 of the same A calibration reference | Diagnostic teacher only; computing it has already paid for policy. No deployment-free gate. |
| Policy K4 spread | Mean logged dispersion over first ten controls; high above q90 of selected A calibration anchors | Diagnostic only. Selected anchors have independent inclusion p=1/16, use 1/selection_p sampling weights in numerator and denominator. Missing draws are not zero; tiny reference samples disable this gate. |
| K4-adjusted discrepancy | cache/policy disagreement greater than K4 spread | Exploratory noise-relative diagnostic; the logged RMS and mean-coordinate standard deviation have different scales, so this inequality has no universal error interpretation. |
| Calibrated guards | Each logged p-value for coverage, stuck, lag, overtime, progress, terminal; G score `6*min(p)` if all six present | Empirical episode-max calibration, not proven deployment conformal. Common α in {.05,.10,.20}; no pooled fallback. Missing tests disable this six-test gate. |
| Guard signals | pre_guard_call; vision-confirmed still statistic >0; progress non-advance statistic >0; raw lag/overtime retained | Raw >0 is presence of a condition, not a learned robot displacement cutoff. |
| Matched progress | top-ranked neighbour normalized progress; early <1/3, middle [1/3,2/3), late >=2/3 | Library time/phase proxy, not physical grasp/contact/release truth. |
| Time since MISS | no previous MISS; otherwise anchors_since_call / median number of A-calibration anchors, high >=1/3 | Portable normalized history clock; calibration per task. Separate never-called indicator. |

Each q90 is the left-continuous weighted empirical inverse CDF (`inf{x:F(x)>=.9}`). A calibration episode contributes total weight one; take every anchor without conditioning on its Y. For a resampled reference, divide by sampling propensity before equalizing episode totals. Require >=2 calibration episodes and >=20 finite reference observations (statistical stability choices); otherwise emit missing rather than borrow another task. Thresholds are built only from init%5==0 A trajectories and then fixed for validation. The library-supplied CDF and p-values need no trajectory refitting. Export provenance and missingness. Cross-cell raw magnitude thresholds are forbidden.

Operational **pre-guard** means pre_guard_call=false; **at-guard** means true. These do not prove “before irreversible physical failure” versus “after failure.” Separately show histories with no previous B diagnostic guard and no previous actual MISS, versus previously alerted/called histories. Physical contacts and predicates in successors are outcomes, never pre-action features. Sensor/entity event labeling and certified branch restore remain necessary to locate actual recovery deadlines. No contact/grasp semantics are inferred from simulator vector position or an undocumented gripper sign.

G's Bonferroni rule is a useful prespecified candidate, but P3's n=5/task reference has minimum p=1/6; even n=50/task has minimum 1/51. With six tests, α=.05 or .10 can be vacuous at both sizes; α=.20 can resolve at n=50 but not n=5. Report actual support from logs and library sample sizes. Do not silently introduce G's proposed pooling; P3 deliberately did not validate it. Frozen PCA/metric/V7 and cache-state distribution shift preclude a claimed conformal safety guarantee.

## 4. Decision rule: evidence to a portable placement proposal

**No pilot result automatically replaces B.** The primary family tests rank hypotheses; the final candidate needs a new closed-loop matched comparison. The current causal effects have the experiment's future controller, not the candidate's state distribution. No whole-policy importance product or replayed “counterfactual SR” will be reported.

Finite cost-free candidate gates are coverage-high, neighbour-disagreement-high, early-phase, non-advancing-progress, no previous MISS, and the complete six-test G gate at α=.20. Their inverses are not silently searched. A calibration-only nomination chooses the gate with the largest positive CALL-effect enrichment over its complement **only if** both the enrichment and the gate CALL effect meet the support guards and have positive simultaneous 1-.05/(6*8) lower bounds (48 secondary selection slots). If none qualifies, nominate no learned placement rule; continue B plus an explicitly experimental coverage-high gate for data collection only. An observed shadow/K4 interaction without a cost-free proxy requests further data, not a deployable gate. Pre-guard effect >0 nominates preventive placement; benefits confined to at-guard histories favor a reactive rule. Delay0 superiority favors acting at the earliest gate; an unresolved delay contrast does not justify intentional waiting.

Duration defaults to `min(H, 2R)` controls, hold=1, then a fresh vision/retrieval decision, with one-anchor cooldown. H and R come from the adapter/library interface (P3 R=5). Fresh policy calls, not repeated tails, implement any hold. Choose R controls only if the starting-duration contrast shows 5 is SR-noninferior to10 by the one-pp margin and lowers modeled cost; choose hold3 only if its SR effect is positive at simultaneous confidence and its whole factorial arm does not exceed B's measured IR. Otherwise preserve hold1; hold2 is exploratory. These dimension contrasts remain insufficient to certify their joint combination, hence a prospective package test.

**Exact candidate P(g):** at each anchor with no active commitment, compute the chosen gate g from pre-policy retrieval/history. With gate false use cache; with gate true, use one policy chunk with probability q from an independent keyed coin. During cooldown use cache. After commitment, retrieve anew with continuity to the actually executed tail; never hand back to the stale pre-call chunk. No independent hardcoded contact detector. No hidden mandatory guard overrides: replacing/suppressing reactive B is precisely the hypothesis to validate.

Calibrate q without outcome labels on the task's few recorded calibration trajectories: compute B's average modeled cost/actual controls, and the gate's eligible firing count after the proposed commitment/cooldown schedule on those frozen streams; select the largest q in {0,1/16,2/16,...,1} whose *replayed planned* IR does not exceed B's calibration IR. Integrate this replay over 256 fixed coin seeds, 0..255. Use task-pooled q only after normalizing features by each task's library, and explicitly label the task mix. No eligible calibration stream or no q>0 means do not nominate. This replay determines an initial budget parameter, **not a predicted closed-loop SR or guarantee of equal realized IR**; early termination and changed states invalidate such a guarantee. The pilot analyzer exports nominations/effect and support tables; it does not implement or certify this serving controller or budget replay.

Accept P(g) only on untouched matched closed-loop episodes if lower one-sided 95% bounds on SR(P)−SR(P10) and SR(P)−SR(B) exceed −.01, and upper one-sided 95% bound on IR(P)−IR(B) is <=0. Multiple candidate arms require Holm adjustment, using a joint SR/cost gate; compare to native pure inference separately before claiming the owner's literal target. If benefits require shadow/K4 every anchor, account their full inference cost; such a result usually defeats Q3's purpose. A nonpositive randomized gate benefit, absent enrichment, failed replication, or a noninferior-cost candidate losing SR falsifies the respective proposal.

## 5. What pilot and full campaign can resolve

HANDBACK's planning assumption is paired variance .20, not measured ICC. At repeated-init rho=.3, pilot n_eff=37.5/cohort/cell, 95% halfwidth14.3pp and 80%-power MDE20.5pp. Full cumulative500 episodes/cohort/cell contains250 distinct init clusters; n_eff352.1, halfwidth4.67pp, MDE6.68pp. At rho=1 full halfwidth6.07pp. These are whole-cohort figures; supported root strata and K4 are weaker, and 64-way simultaneous intervals are wider. Splitting pilot by init gives no within-task init variance estimate. Thousands of anchors do not repair this.

If inconclusive, request the already planned **same nine cohorts × same three blocks × eight cells**, disjoint continuation440 episodes/cohort/cell =31,680 new episodes, for36,000 cumulative. For Q3 alone the irreducible matched set is A, B, P10, factorial, window and one fixed-dose control (dose25): 6×440×8=21,120 continuation episodes; keeping all nine preserves the owner's superset design and dose-continuation checks. Do not change propensities after seeing outcomes without recording a new design version.

The full schedule is block0 inits0..24, block1 0..14, block2 0..9. Conditional fixed-task variance and score ESS will be measured from paired differences/cluster influence, not ICC of duplicated terminal Y. The script emits precision planning using the observed SE and `(current_SE/target_SE)^2` as a **conditional scaling assumption**, not a sample-size guarantee. Include sensitivities to fixed variance .20 and ICC, retain nonpositive/undefined variance warnings.

Even full500 cannot establish one-pp SR safety. The source formula `ceil(1.96^2*.20/delta^2)` requires 308 effective paired units for ±5pp and 7,684 for ±1pp (arithmetic checked by the script). A one-sided noninferiority test at80% power and true gap0 needs `ceil((1.644854+.841621)^2*.20/.01^2)` effective paired units; the script computes this rather than presenting a pilot promise. Fresh init/scene diversity is preferable to indefinitely repeating the same stock pool. More tasks/independent libraries and another robot benchmark are required for transfer, not just repeated seeds.

After nomination, request at most **five configurations** per chosen cell/benchmark, all matched: native pure inference at R; P10 at2R; exact B; P(g) with frozen q,duration,hold; a uniform-call control with the same calibration planned IR and same commitment/cooldown. P(g)−uniform answers timing at matched budget; P(g)−B answers the practical improvement; P(g)−both inference references answers retention. Compute episodes from the observed paired-difference variance with the one-pp target before scheduling; no fixed affordable count is falsely claimed sufficient. If duration/hold remains unresolved, spend remaining up-to-eight-arm capacity on P(g) atR, hold2 and hold3, with multiplicity adjustment. Certified snapshot branches can sharpen local deadlines but do not replace this deployment test.

## 6. Implementation and reproducible commands

All writes are confined to this directory. CPU-only, at most four processes; no live pilot access, polling, simulator jobs or profiler changes. `analyze_pilot.py` will accept `--tables`, or `--run-root` with `--arms`/`--cell`; run-root mode will invoke the strict P3 v2 reader once, with stage-count/snapshot checks, into this directory. It must refuse incomplete requested manifests and unverified client/control data. A partial arm list can be analyzed only with an explicit partial flag and cannot be labeled a complete cell. It will output audit/support/features, episode contrasts, randomized effects/interactions, planning and static plots, preserving smoke labels.

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q3/.mplconfig .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/analyze_pilot.py --tables /home/weiland/trace_runs/os_closed_loop/r06_p3_v2_client_smoke/tables_v2_smoke8b --smoke --allow-partial --out exp/offline_search/rounds/r06/ideation_Q3/smoke_analysis

# FUTURE invocation only, once the coordinator declares the full cell finished:
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q3/.mplconfig .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/analyze_pilot.py --run-root /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot --cell pi05_l10_50 --stage pilot --out exp/offline_search/rounds/r06/ideation_Q3/pilot_pi05_l10_50
```

Smoke results and any implementation deviations will be appended below; no scientific conclusion will be inferred from the fixture.

## 7. Implementation clarifications registered before pilot access

These clarifications follow code review and smoke plumbing, still with **no pilot outcomes or run-root inspection**. The original specification is preserved unchanged in `PREREG_FROZEN.md`, SHA256 `61b2e5696ec95e7e767d8e6ad87d3e4ed1ef67494b85e1d6bed5c22b199bfb13`.

1. **Two IR summaries.** `IR` is mean episode IR from §2. `IR_pooled` is total modeled deployment cost divided by total actual active controls/5, with a ratio-estimator cluster influence and paired covariance for differences. The latter matches aggregate compute accounting; both are exported. A prospective acceptance claim will require the no-higher-IR condition on **both**, rather than exploiting a denominator convention. The whole-factorial cost screen for a hold proposal uses pooled IR. Neither is profiler-disabled wall-clock latency.
2. **Diagnostic B history.** `assignment.pre_guard_call` refers to the active baseline's verdict. It identifies the intended pre-guard/at-guard strata in factorial, whose baseline is B; it does not necessarily describe B in A/fixed-dose arms. `diagnostic_guard` therefore uses `guards.inputs_outputs.os_force_miss`, and `previous_guard` accumulates that B diagnostic. Both fields are exported and the diagnostic has its own exploratory interaction. No call feature uses a future guard. For replication of B-guard heterogeneity in fixed doses, use `diagnostic_guard`, not their A baseline's `pre_guard_call`.
3. **Multiplicity and unsupported sides.** The 64 primary SR slots refer to the **all-split** eight contrasts per cell. Split-specific primary-name rows, other features and block summaries are descriptive unless used by the separately prespecified calibration-only 48-slot nomination rule. With no realized observation on either contrasted assignment, retain the raw HT total for audit but withhold intervals and p-values, label `point_only_no_realized_comparison`, and omit it from the forest. No interval is supplied from zero variance or a singleton init in any contributing task. HT finite-sample estimates need not lie in [-1,1]; they are contrasts of weighted totals, not fitted success probabilities.
4. **Learning versus fixed screens.** Gate nomination uses calibration only. The duration/hold/pre-guard/delay screens in `method_screen.json` use the fixed whole-pilot primary contrasts; these are provisional, prespecified hypothesis choices, not training/validation certification. A learned gate cannot pass this implementation's fixed-task support rule using the pilot's calibration split alone (one init/task). This is intentional. `method_screen.json` never marks a controller deployable; all joint packages still require the prospective comparison in §4.
5. **Verification boundary.** Table mode requires coordinator-produced v2 tables and checks accepted identity joins, full decision coverage, all assignment RNG/state-machine outputs, block/split provenance, actual-control and MISS accounting, independent stage counts, source availability and complete scheduled slots. It trusts upstream `read_v2` for raw input archive/snapshot payload checks. Run-root mode calls that reader with `--require-stage-counts --require-snapshots`. Snapshot restoration remains uncertified. A hash-matched library artifact is required before reporting a calibration sample size; a minimum p merely observed on trajectories is never used to invent n.
6. **What is implemented.** `analyze_pilot.py` builds reference thresholds, estimates, nominations and package-dimension screens. The reference samples include identities and provenance; input CSV and analysis hashes are in `audit.json`. It does **not** implement P(g), the 256-seed q budget replay, snapshot branches, or whole-policy off-policy evaluation. Those are explicit future method/experiment work, not missing pieces required to analyze the pilot. The run-root extraction route is implemented but was not run on live pilot arms; the requested end-to-end test uses the designated smoke tables.

The analyzer reuses the R4 estimator's HT signed weights and init-cluster accounting concept. It does not import R4's complementary-pair validation, hardcoded p=.5, single-episode landmark logic, or binary gripper contexts: none describes this v2 experiment. It replays v2's actual `Design.resolve` and SHA assignment and uses sandwich intervals instead of R4's percentile bootstrap. This avoids falsely treating deterministic holds or scheduled obligations as independently randomized calls.

## 8. Smoke execution and deliverables

Ran the smoke command in §6 successfully with the final analyzer. Console result:

```json
{"episodes":32,"anchors":1111,"decisions":2120,"arms":8,"assignments_replayed":1111,"estimate_rows":1878,"smoke":true,"eligible_nominations":0}
```

The source `tables_v2_smoke8b/audit.json` has 32 episodes, 1,111 anchors, 2,120 decisions, 10,876 control records, 17,776 neighbours and 14,182 action-step rows. The accepted active-control sum used in IR is **10,556**, verified against `decisions.actual_controls`; these are different accounting scopes. Measured stage dispatch totals are **2,120 / 2,120 / 5,453** for stages1/2/3. All **1,111** smoke anchors were K4-selected: the fixture uses a denser resampling setting than the pilot. The analysis reads the logged selection propensity, rather than assigning 1/16 to smoke rows. These are plumbing/accounting facts only.

The final `smoke_analysis/effects.csv` contains **1,878** rows across outcomes, splits, roots, strata and contrasts—not 1,878 independent experiments. It includes 16 all-split primary SR rows for two fixture cells. Status counts are 358 `estimated`, 768 `point_only_no_estimable_fixed_task_variance`, 168 `point_only_no_realized_comparison`, 340 `empty`, and 244 `missing_feature_side`. `episode_contrasts.csv` has 66 rows including explicit missing-cohort records. Smoke has no B/fixed-dose cohorts and only one A-calibration episode per task, so trajectory-reference thresholds correctly remain unavailable. All outputs carry `smoke=true` or an explicit smoke title, and automatic nominations are disabled.

The window fixture has four delay0 triggers per cell and no delay1/2 triggers. Thus **zero window contrast intervals** are emitted. The deterministic tests exercise all delay categories, including a terminal trajectory that never executes its scheduled call. This verifies the estimator logic without pretending the fixture supplies delayed-call outcome support.

Files ready for review:

| Artifact | Content |
|---|---|
| `analyze_pilot.py` | Finished-arm/table entry point; strict auditing, features, HT effects, paired comparisons, intervals and plotting |
| `smoke_analysis/audit.json` | Input hashes/counts, assignment replay count, cost accounting, analysis/specification hashes |
| `smoke_analysis/effects.csv` | Primary and exploratory causal excursion/package contrasts, cluster/ESS support and intervals |
| `smoke_analysis/episode_means.csv`, `episode_contrasts.csv` | Observed closed-loop SR and IR context, paired configuration differences |
| `smoke_analysis/support.csv`, `feature_missingness.csv`, `window_reach.csv` | Positivity, feature coverage, delayed-call nonexecution |
| `smoke_analysis/feature_references.json`, `anchor_features.csv` | Outcome-blind calibration samples and online candidate features |
| `smoke_analysis/calibration_resolution.csv` | SHA-matched per-task library n, attainable α and empirical observed p ranges |
| `smoke_analysis/gain_per_call.csv` | Ratios only with a positive, resolved incremental-MISS denominator |
| `smoke_analysis/nominations.csv`, `method_screen.json` | Frozen gate/package nomination rules; never deployment approval |
| `smoke_analysis/precision.json` | Paired-difference ICC, actual cluster-score SE planning, scheduled split sensitivities |
| `smoke_analysis/primary_effects.png`, `observed_arms.png` | Standalone labeled smoke plots; no performance interpretation |
| `test_analyze_pilot.py`, `analysis_test_results.json` | 16 deterministic statistical/design checks, all passing |

The hash-matched **library calibration artifacts**, independent of smoke outcomes, contain n=5 for the fixture's π0.5 tasks0/1 and n=43/48 for its GR00T tasks0/1. The script reports these actual counts rather than assuming nominal library500 implies exactly50 usable episodes per task. Source: `../p3_profiling/calibration_v2/{pi05_l10_50,groot_l10_500}.json`, checked against anchor calibration hashes; output `calibration_resolution.csv`. This is an artifact-resolution check, not a performance result.

Statistical/design test command actually run:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q3/.mplconfig .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/test_analyze_pilot.py
```

Result: **16 tests, zero failures, zero errors**. Tests check exact unequal-propensity HT recovery, unchanged uncertainty when identical anchors are replicated within init, paired covariance, zero-variance and single-init refusal, duration/hold marginalization, actual versus nominal delayed propensity, unexecuted delayed calls, source support before guards, K4 sampling weights, outcome-blind reference fitting, diagnostic B history, aggregate versus mean IR, manifest counts and the write boundary. The end-to-end smoke was rerun after code changes; `audit.json`'s analysis SHA matches the delivered script. The first smoke attempt found and fixed pandas index alignment in the control-count audit; that attempt produced no scientific output.

Also actually ran the smoke command with `--allow-partial` **without** `--smoke`, output `rejected_incomplete_pilot_check/`. It exited1 with `incomplete/unexpected scheduled pilot slots`, as required: four episodes/fixture arm cannot pass a 20-episode pilot-arm check. A partial cohort list does not waive per-arm manifest completeness. This negative test did not access the pilot.

Outputs intentionally require a fresh `--out` directory. Earlier smoke iterations are retained in `smoke_analysis_initial/`, `smoke_analysis_before_support_check/` and `smoke_analysis_before_precision/`; the authoritative final result is **`smoke_analysis/`**.

## 9. Precision after respecting the reserved split

The scheduled-split arithmetic is computed by `analyze_pilot.py:scheduled_precision` and written to `smoke_analysis/precision.json`; it is design arithmetic, not a smoke SR estimate. With the same assumed paired variance .20 and rho=.3:

| Population per cohort/cell | Episodes | Distinct task/init clusters | Planning n_eff | Pointwise 95% halfwidth | 80%-power MDE |
|---|---:|---:|---:|---:|---:|
| Whole pilot | 60 | 20 | 37.50 | 14.31pp | 20.46pp |
| Pilot validation only | 30 | 10 | 18.75 | 20.24pp | 28.94pp |
| Full calibration only | 100 | 50 | 70.42 | 10.45pp | 14.93pp |
| Full validation only | 400 | 200 | 281.69 | 5.22pp | 7.47pp |
| Continuation validation only | 370 | 190 | 263.27 | 5.40pp | 7.72pp |

Pilot validation has **zero fixed-task between-init degrees of freedom**, so the normal planning halfwidth is not an interval the analyzer will report from that split. Full validation is weaker than the full500 figure in §5; continuation-only validation is weaker again. Local feature/package contrasts require their measured cluster-score precision and support, not this binary paired planning table. The script emits those score-based scaling calculations separately, with the explicit assumption of stable visitation and repeat structure.

The proposed one-pp noninferiority target at80% power and true gap0 needs **12,366 effective paired units** under the declared .20 variance and one-sided normal formula. That is a planning illustration, not a request to manufacture independence with repeated seeds. The new five-configuration matched deployment comparison should be sized using its own paired discordance and independent initial scenes, with explicit error control; a few hundred episodes cannot certify this target by absence of observed regressions.

No pilot arm was inspected, extracted, polled or awaited. No server, worker, chain, simulator, port or remote host was used. The deliverable is preregistered analysis ready for coordinator-supplied finished cells, with honest support limits and a prospective route from a useful placement signal to a tested SR–IR improvement.
