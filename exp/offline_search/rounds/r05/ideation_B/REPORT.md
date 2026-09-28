# R5 ideation B — What can actually be solved offline?

**Recommendation:** retain K7’s control semantics and the winning tail recipe; solve a small, active set of retrieval parameters with grouped validation, then test one frozen configuration per cell. Do not fit CALL/CACHE decisions to observational success labels. Those decisions need K5. Offline solvers can determine statistics, numerical regularization and a requested cost budget; the present evidence does **not** identify an offline objective that reliably orders closed-loop success.

The strongest positive retrospective test is ridge on π0.5 spatial. The strongest negative test is sparse-library phase continuation versus anchor tail: both action error and my early failure-risk model choose the wrong controller. This report contains two ranked proposals, including one conditional on the randomized data arriving. It does not resurrect noprog-4 or 50-library phase continuation.

## 1. Scope, reproduction and validation boundaries

All new work is under `exp/offline_search/rounds/r05/ideation_B/`. No original source was changed, no simulator/server/GPU was started, and no git commands, external models or sub-agents were used. The advertised `/dev/shm/offline_search_store` was absent; reads used `/home/weiland/trace_runs/offline_search_store`. The nominal 50-library π0.5 spatial cell actually has 49 episodes.

Run every script from the repository root with this prefix:

```bash
taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python
```

Append `exp/offline_search/rounds/r05/ideation_B/<script>.py`. The scripts use only NumPy/SciPy and existing project code. Each numerical job was single-process; the final recomputation used four concurrent jobs on the assigned eight logical CPUs. No large arrays were saved; `/tmp/r05_ideation_B/` was reserved but not needed.

| Script | Product and purpose |
|---|---|
| `log_audit.py` | `log_results.json`, `landmarks.json`: accepted terminal attempts, contiguous decisions, paired outcomes, frozen-path no-progress replay, closed-form failure-risk LDA |
| `library_solver.py --tag verified_` | `verified_library_<cell>_<scale>.json`: 17 configurations (including duplicate baseline references), metric-refitted episode holdouts, task holdouts for configuration selection, action and successor-state losses |
| same, `--scales 500 --stock-sigma --restricted` | `stock_library_<cell>_500.json`: five configurations in the exact stock action units; prevents a normalization change from masquerading as hyperparameter improvement |
| `covariance.py` | NumPy closed-form OAS spherical covariance shrinkage, used by the metric and LDA audits |
| `calibration.py` | `calibration_results.json`: all eight libraries’ motion/camera calibration, LOEO threshold stability, guard false alarms on successful demos, rollout residual quantile transfer |
| `phase_prior.py` | `phase_results.json`: moment solution for a categorical clock-offset prior and its equivalent state-MSE penalty |
| `prior_solver.py` | `prior_results.json`: borrowed α LOEO sweep and episode-jackknife moment shrinkage |
| `cost_solver.py` | `cost_results.json`: exact IR inverses, finite-episode periodic arithmetic, and a fail-closed K5 contextual LP/held-init evaluation implementation |
| `select_configs.py` | `solver_choices.json`, `arms_proposed.json`: frozen, constructor-compatible candidate specifications; no arms launched |
| `verify_artifacts.py` | `verification.json`: AWM numerical parity of the metric factor, fixed evaluation populations, cost identities, constructor checks and unsupported-K5 fallback |
| `measurement_summary.py` | `measurement_summary.json`: final numeric index, file-stat bytes, normalization sensitivity and guard sensitivity |
| `frontier_references.py` | `frontier_reference_arms.json`: six best-known comparison recipes copied from original arm definitions, with new run names and no inherited fit-output path |

The two library runs can be reproduced separately with `--cells pi05_spatial pi05_l10 --tag verified_` and `--cells groot_spatial groot_l10 --tag verified_`; the stock-unit checks use the same cell lists without the tag, plus `--scales 500 --stock-sigma --restricted`. Run selection and verification after all twelve library output files exist. `verification.json` records library manifest/ID hashes and result hashes. JSON files retain task and episode values, not only means. Exploratory files in `discarded/` failed an evaluation-population check and are explicitly excluded; use the verified/stock files and selector hashes.

**Terminology matters.** The old `exp/rit_loto/build_loto_table.py` says *leave-one-trajectory-out*, not leave-one-task-out. Its own-trajectory exclusion and reconstruction/parity gate are useful precedents; its local shadow-deviation labels are not closed-loop rescue labels. Here **LOEO** removes an episode from supervised metric fitting and retrieval candidates; **LOTO** trains the configuration selector on nine tasks and evaluates the tenth, while retrieval always remains within the query’s task. A task held out from hyperparameter selection still has its own deployment library. This is not zero-shot retrieval without that task’s demos.

**Validation limits:** PCA and action normalization are frozen on the deployed library before the episode holdout. Thus these are metric-refitted, conditional LOEO results, **not fully inductive PCA-plus-metric LOEO**. At 50, every episode is held out once; at 500, ten evenly spaced episodes per task are held out individually, each against the other 49, with at most 24 evenly spaced query rows per held episode. Action losses exclude step zero; step zero is audited separately. Successor losses use the same nonterminal query population for every configuration; a terminal candidate predicts zero further displacement. Results average episodes, then tasks. The 500 episode sampling is deterministic, not a random sample with a sampling-confidence guarantee.

The primary library-only audit uses each deployed library’s own action standard deviations. Stock AWM instead uses `Context.action_sigma` from the **current** library even when deploying 500 episodes. The separate stock-unit audit inherits this benchmark normalization. Conservatively, that fit is labeled **borrowed big-library information (external normalization only)** under the instruction to disclose every fit using information outside the deployed rows; it does not add big-library information to a 50-library method. Every learned fit from rollout states/outcomes, and every α fit, is also explicitly **borrowed big-library information**. Neither the library-only solver nor its candidate retrieval uses logged `a_inf`.

Final verification **PASS**: 599 held-episode cases and 13,466 sampled query rows across the eight primary panels; every candidate uses the same successor-query population within an episode. The independent metric factor agrees with AWM to **1.11e−16** on the numerical check. All **16 proposed constructors** instantiate; the **8-episode K5 smoke** has three observed contexts and yields **zero supported overrides**. The log audit contains 29,500 arm-episodes with **zero noncontiguous accepted episodes**. Full deployment fits and new closed-loop success are deliberately not claimed by these checks.

## 2. Key measured facts

### 2.1 Raw decisions and the required cost arithmetic

`log_audit.py` processed **59 arms and 1,414,662 accepted, contiguous decisions**, selecting R2 CL2/CL3, R3 full/mixed and relevant R4 frontier/blind/K7 arms. This is a specified subset of the larger log collection, not a claim to have ingested every run. The input file sizes are frozen in `log_results.json`; unfinished/missing arms are listed separately. Legacy resumed event arms are retained descriptively and are not used to train the LDA or choose a proposal.

For π0.5, all costs below recompute **IR = (.152 V + .848 M)/N**, with one decision = five controls, full-price policy calls, and search excluded. Eager costs are not substituted. Some early R4 `summary.json` values disagree with their own raw counts under this formula. Preserve the published values as historical statements, but use the mandated arithmetic for this report:

| Cell/arm | SR | N | V | M | Recomputed IR |
|---|---:|---:|---:|---:|---:|
| l10 50 stock guards | .740 | 32,967 | 32,967 | 6,664 | .323416 |
| l10 50 noprog-4 | .690 | 34,353 | 34,353 | 5,720 | .293198 |
| l10 50 periodic 5 | .792 | 31,044 | 31,044 | 5,962 | .314858 |
| l10 50 periodic 6 | .764 | 32,156 | 32,156 | 5,174 | .288446 |
| l10 50 K7 phase B2 | .700 | 34,229 | 21,941 | 6,884 | .267979 |
| l10 50 K7 tail B1 | .806 | 31,186 | 18,566 | 5,554 | .241513 |
| l10 500 stock guards | .864 | 29,340 | 29,340 | 2,983 | .238216 |
| l10 500 noprog-4 | .808 | 30,450 | 30,450 | 2,227 | .214020 |
| l10 500 periodic 8 | .850 | 29,373 | 29,373 | 3,517 | .253536 |
| l10 500 periodic 12 | .828 | 29,986 | 29,986 | 2,252 | .215686 |
| l10 500 K1 dense B0 | .842 | 29,615 | 29,615 | 4,144 | .270660 |
| l10 500 K7 B0 | .830 | 30,242 | 30,242 | 3,300 | .244534 |
| l10 500 K7 phase B1 | .862 | 29,422 | 20,435 | 4,175 | .225903 |
| l10 500 K7 phase B2 | .862 | 29,136 | 16,691 | 3,111 | .177621 |
| l10 500 K7 tail B1 | .880 | 28,888 | 16,547 | 3,959 | .203281 |
| spatial 500 stock guards | .974 | 10,773 | 10,773 | 687 | .206077 |
| spatial 500 periodic 12 | .976 | 10,769 | 10,769 | 662 | .204129 |
| spatial 500 K7 phase B2 | .968 | 10,882 | 5,926 | 735 | .140051 |
| spatial 500 K7 tail B1 | .982 | 10,630 | 5,643 | 596 | .128236 |

For example, spatial g500’s summary prints `.19668`, but `.152 + .848*(687/10773) = .206077416`. Findings’ `.245/.206` for l10 periodic 8/12 likewise differ from `.253536/.215686` using raw counts. This is an accounting discrepancy, not a new SR finding or permission to change the owner’s coefficients. Conclusions about the failed noprog relaxation and the successful tail recipe survive the correction.

A newly available pure-cache phase arm, `r4b3_p_l10_500_ph2c`, has **.778 SR**, N=31,688, V=18,910, M=0, **IR .090707**. It was pending in FINDINGS; it is only a descriptive snapshot here, with no 50 counterpart or new ranking claim.

Paired raw-log differences (new minus old, 500 common inits): noprog-4 is **−.050, SE .017424** at 50 and **−.056, SE .016317** at 500. K7 phase B2 minus B1 is **0.000, SE .017453**. Tail minus phase B2 is **+.106, SE .023294** at 50 and **+.018, SE .016610** at 500. Therefore the sparse-library tail advantage is much stronger evidence than the 500 tail-versus-phase SR ordering.

### 2.2 Why frozen-path cost replay and failure-risk fitting are insufficient

Replaying noprog-4 on the stock guard paths gives:

| Library | Stock M/N | Frozen-path np4 M | Predicted IR | Executed np4 IR | Actual ΔSR |
|---|---:|---:|---:|---:|---:|
| 50 | 6,664 / 32,967 | 4,790 | .275212 | .293198 | −.050 |
| 500 | 2,983 / 29,340 | 1,632 | .199169 | .214020 | −.056 |

The replay reproduces the existing noprog-3 MISS count exactly at both scales. Its failure is the changed state distribution and episode length after changing decisions, not a bad implementation of the original guard. A replay is a conditional cost calculation, not off-policy evaluation.

I fitted a closed-form LDA at the first real vision anchor between decisions 10 and 13, using `dnn, disp, vis, stuck_n, overtime, lag, top1_prog, noprog_n`. There is one sample per episode, no final-duration feature, and no outcome in the inputs. Training uses the stock R3 l10 guard run separately at each scale. LOTO excludes the entire target task; external R4 predictions use that task-excluded fit. Covariance uses OAS shrinkage; posterior intercept uses training prevalence. This is **borrowed big-library information**, with eventual failure as an observational label.

| Scale | Episodes / failures | LOTO AUC | Held-init-fold AUC | LOTO Brier / constant Brier |
|---|---:|---:|---:|---:|
| 50 | 500 / 130 | .616736 | .716008 | .191883 / .201745 |
| 500 | 500 / 68 | .683892 | .726239 | .111399 / .121554 |

Despite useful within-arm discrimination, it reverses the important controller ranking:

| Scale | Predicted failure: phase B2 / tail B1 | Actual failure: phase / tail |
|---|---:|---:|
| 50 | .202175 / .247968 | .300 / .194 |
| 500 | .093234 / .107253 | .138 / .120 |

At 500 it also predicts B2 safer than B1 (`.093234` versus `.108916`), although observed SR is identical. Full-fit standardized visual-similarity coefficients change sign across scale: **−.538193 at 50, +.306762 at 500**. These are failure-risk coefficients under different behavior distributions, not weights for the causal value of CALL. There are no corresponding completed GR00T mixed/K7 data here; K7 explicitly refuses GR00T because the stock terminal-gripper semantics differ.

### 2.3 Demo thresholds: precision is measurable; rescue utility is not

`calibration.py` recomputed the stock global motion 10th percentile and minimum-camera cosine 95th percentile, with each camera centered by task, then counted two consecutive qualifying transitions. Motion is raw valid-state L2, not the normalized blind-motion gate.

| Cell | Scale | Motion threshold | Camera threshold | Successful demos alerted: vision-confirmed / motion-only |
|---|---:|---:|---:|---:|
| π0.5 spatial | 50 | .0931858614 | .8892200857 | 0 / .204082 |
| π0.5 spatial | 500 | .0938192621 | .9056166422 | .022587 / .266940 |
| π0.5 l10 | 50 | .0518229079 | .9574828026 | .240000 / .640000 |
| π0.5 l10 | 500 | .0385605428 | .9844978288 | .073394 / .470183 |
| GR00T spatial | 50 | .1658364296 | .8307866101 | .020000 / .480000 |
| GR00T spatial | 500 | .1190654784 | .8691121569 | .063596 / .217105 |
| GR00T l10 | 50 | .0424967110 | .9053388310 | .380000 / .760000 |
| GR00T l10 | 500 | .0247794464 | .9511453617 | .091335 / .344262 |

These are alerts in successful demonstrations, not proven harmful interventions. All 49/50 current-library episodes are successful. At 500, successes are **487, 436, 456, 427** in table cell order. Thus current-library outcome labels cannot identify an optimal rescue boundary: a classifier can minimize false alerts by never calling. Big-library failures add associations but still do not give the counterfactual effect of a call.

LOEO threshold stability is also not an optimality criterion. For π0.5 l10, deleting one episode moves the motion/camera thresholds over **[.051071,.055732] / [.942751,.958226]** at 50, versus **[.038491,.039224] / [.983745,.984579]** at 500. Camera centering remains fixed for this quantile-stability calculation. A robust percentile can still be the wrong control threshold.

Changing the percentile levels is consequential. At `stuck_thr=2`, a stricter `(motion p5, cosine p99)` versus looser `(p30,p90)` rule alerts successful π-l10 demos at **.060/.440 (50)** and **.011468/.188073 (500)**; GR-l10 rates are **.100/.640** and **.002342/.288056**. Keeping p10/p95 but changing run length 1→2→3 gives π-l10 **.320→.240→.140** at 50 and **.119266→.073394→.045872** at 500; GR-l10 **.640→.380→.200** and **.215457→.091335→.046838**. Full spatial and l10 grids are in `calibration_results.json` and `measurement_summary.json`. Minimizing these successful-demo alarm rates alone would favor excessive conservatism, not identify rescue timing.

### 2.4 Blind calibration and the phase-penalty closed form

`calibration.py` also reads R4 A’s saved, scalar-API-validated anchor/window diagnostics (`rounds/r04/ideation_A/measure_blind.py`). Fits on those logged full-inference states are **borrowed big-library information**. No new policy calls are made.

For π0.5 l10 h=1, fitting the displacement-residual 95th percentile on inference paths gives **.319731 / .293596** at 50/500. On pure-cache paths those thresholds fire on **11.4669% / 11.9313%**, not 5%. At h=2 the fitted values **.529404 / .497202** fire on **14.8951% / 14.6753%** of cache windows. The existing `.5` threshold fires on **16.6049% / 14.5322%** at h=2. Row quantiles therefore need a distribution label, horizon label and episode grouping; they are not guarantees for the modified controller.

The 95th percentile of each episode’s *maximum* h=1 residual is **.915531 / .872452**, much larger than the row percentile. This makes explicit the difference between “5% of decisions” and “5% of episodes have any alarm.” Per-task current libraries have only about five episodes; a distribution-free 95% episode-level bound cannot be nontrivially resolved from those five exchangeable calibration examples.

The same saved windows give l10 h=1 action RMS **phase .538430 versus tail .559949** at 50, and **phase .460683 versus tail .489711** at 500. Action error favors phase at both scales; the closed-loop tail recipe is substantially better at 50. This rules out selecting the serving choice by this error alone.

`phase_prior.py` supplies a genuine moment solution under an explicit, limited model. Match each demo row to three action-near rows from other episodes. At the next demo row, label the donor’s local offset −1/0/+1 by its best next-head action match. Estimate categorical probabilities π and normalized state-residual variance σ². Gaussian state likelihood plus that categorical prior yields an MSE penalty

`penalty(δ) = (2 σ² / 8) log(π0 / πδ)`.

The symmetric equivalent is the mean of the ±1 penalties. Median across tasks:

| Cell | 50 | 500 |
|---|---:|---:|
| π0.5 spatial | .051049 | .095722 |
| π0.5 l10 | .041956 | .061868 |
| GR00T spatial | .079455 | .105455 |
| GR00T l10 | .036395 | .066268 |

All 80 task fits have nonnegative symmetric penalties. This explains why `.05` is a plausible scale, but does **not** establish an optimal controller penalty: the phase labels are action-matching proxies, residuals are anisotropic, and the samples are dependent. Offsets outside the stored horizon are not made valid by any fit. This diagnostic does not reopen 50-library phase continuation.

## 3. Hyperparameter inventory and the earliest valid solver

A = deployed demos at library build; B = logged closed-loop observations/outcomes; C = randomized CALL/CACHE. “A possible” means the stated **proxy/statistic** can be solved, not that SR utility is identified. Source of implementation facts: `rounds/r02/g1_awm/awm.py`, `r03/h1_trap/awm3.py`, `r02/g3_recovery/{wrappers,g3_core}.py`, `r03/h3_judge/judge.py`, `r04/k1_blind/{blind_awm,judge,control_step}.py`, `r04/k7_guard/judge.py`, and `closed_loop/plugin.py`. Historical tuning evidence comes from R2/R3 ANALYSIS and R4 reports/arm manifests; absent controlled evidence is marked explicitly.

| Parameter and current setting | How set; sensitivity/evidence | Offline solver and recommendation |
|---|---|---|
| PCA dimension: 64/camera; full joint code 136 | Fixed R2 representation/cost choice. PCA fit is data-derived; dimension is hand-set. R2 reduced code-32 had similar offline error, not a controlled SR validation of PCA dimension. | A: grouped predictive validation of 16/32/64; reconstruction variance alone is insufficient. Implemented conditional sweep. Full inductive PCA CV remains unrun. Keep 64 in R5 serving. |
| PCA recipe: randomized rank 128, oversample 32, power 3, seed 0 | Numerical fitting recipe/cache identity, not behavior knobs with closed-loop ablations. Big basis fitted on 500 demos; current basis on current demos. | A: approximation/parity tolerance and memory budget; fix seeds and fingerprints. Do not optimize these using SR. |
| Kernel member count k=16 | Hand-set synthesis support; R2 mean-5 and AWM differ in metric and weighting, so that comparison does not identify k. | A: LOEO action/state prediction or conditional likelihood; swept 8/16/32. K1 explicitly requires 16-member anchors. Keep 16 in blind arms; k=8/32 would require relaxing and revalidating that contract. |
| Kernel bandwidth kref=5 at 50, 8 at 500 | Chosen from R2 offline regime tradeoffs. At 50, kr8 worsens step-zero RMS in all four cells; at 500 kr8 slightly improves recorded stale error. No same-library closed-loop kr5-vs-kr8 experiment. | A: leave-episode-out conditional likelihood/prediction, then LOTO selection; implemented. Candidate R5 comparison is same-library kr5 vs kr8, never infer its effect from CL2-50 vs CL2-500. |
| Kernel rule exp(−((d−d1)/(d_kref−d1))²), bandwidth floor 1e−6 | Functional form fixed by AWM. Floor protects zero/tied distances. | A: bandwidth CV can solve within this family; likelihood must specify outcome noise and use held episodes. Floor/tie tolerances are numerical requirements, not utility fits. |
| Whitening pairs: 3 nearest action heads, other episode, same task | AWM’s supervised definition; three rows may all come from the same donor episode. | A: covariance fit is closed form; pair definition remains a modeling choice. Swept nn=1/3/8 and distinct-donor variant. Likelihood prefers nn=1 in the broad audit; no corresponding SR validation, so retain 3 in proposals. |
| Main ridge .1; early ridge .1 | Hand ridge. R3 `ridge_main=1` improves π-spatial SR .800→.840, but GR-spatial .888→.894 is unresolved; l10 and big-ridge SR comparisons were not run. | A: OAS moment shrinkage or held-episode validation. Both implemented; OAS independence assumptions fail for overlapping temporal pairs. Prefer validated restricted ridge selection, with early ridge unchanged. |
| State multiplier 1 after whitening fit | Scale modifies standardized state coordinates after fitting W; it is not a separate prefit covariance weight. R2 state×3 showed mixed recorded-query gains, without a matching SR isolation. | A: dual action/state LOEO; implemented 0/1/3. Keep vision and fit state×3 only where selected. No state-only control proposal. |
| Early regime: enabled; fit steps≤2; used only at step 0 | R2 early fit particularly helped big GR l10 step-zero error (.229→.168). No isolated early-on/off SR comparison. | A: separate held-episode start-state objective and numerical stability. Current-library early covariance has rank about 11 in 136 dimensions in this audit. Keep branch and .1 ridge; do not let all-step averages turn it off. |
| Fresh continuity λ=.5; stale λ=0 | AWM uses the previous **executed** chunk’s tail only after a MISS. Hand coefficient, with demo-derived scale `s_c`; λ=1 had little offline benefit in R2. | A: demo fresh-tail CV (0/.5/1 implemented); B: real policy-tail states are needed to assess transfer. Never use cached self-output as fresh teacher evidence. Keep .5. |
| Confidence scales `s_c,s_d,s_a`, z means/std | Medians or moments of same-task other-episode library pseudoqueries; data-set, not individually hand-tuned. | A: already solved. Refit for changed selector; validate per regime. Existing calibration excludes self episode from candidates but not from the fitted metric. |
| AWM3 prior α=.5; ridge_main twin; shared β=0 | α=.5 retained after R3; α=1 weaker l10 pilot; shared-covariance pooling was offline-only. Every α value including 0 uses big PCA/standardization. | A only if big data is allowed, explicitly **borrowed big-library information** for 50. LOEO sweep and jackknife moment solution implemented. At 500 fit/deploy the big library directly: α adds no separate library prior. Do not ship the moment α≈1. |
| `stuck_thr=2` | Two qualifying motion-and-vision transitions. K1’s dense alternative changed the feature definition, not just this count. | A: successful-demo alarm rate for run lengths 1/2/3 implemented; B: trap association/timing; C: optimal rescue threshold. Retain 2 and K7 confirmation. |
| Motion percentile 10; camera percentile 95 | Percentile **levels** hand-set; raw thresholds library-derived. K7 restores stock semantics across real vision anchors. | A: quantiles/stability already solvable, measured above. B can calibrate operating-domain rates, C needed to attach call benefit. Preserve levels pending causal evidence. |
| Lag threshold 5; overtime threshold 1×task median demo length; require stuck≥1 | Geometric “behind demo” guard. Median length is fitted, multipliers/counts hand-set. No isolated R2–R4 lag/overtime ablation. | A: scale/quantiles; B: task-specific association and timing; C: call utility. Retain. Demo length is not an online episode endpoint. |
| `noprog_n=3`, `prog_eps=.5` demo decisions | Requires two nonadvancing progress transitions; stored `noprog_n` counts transitions. R4 np4 is a controlled negative at both scales. | A: estimate normal demo jitter only; B: calculate current trigger coverage and frozen-path cost; C: infer override value. Do not weaken to 4 or refit `.5` from failure correlation. |
| Blind progress `noprog_span`; no memo reset after MISS | Elapsed decision span between real anchors. Any positive span also blocks blindness before the 3-count MISS criterion. K7 refuses memo reset. | Structural online-history semantics. Preserve; changing reset behavior changes treatment history and requires a new controlled intervention. |
| Terminal-closed guard | Last library row plus positive executed gripper in stock π0.5. Not the discarded terminal-row mask. GR00T normalized sign differs. | Model/wire contract, not a learned threshold. Adapt and verify semantics before any GR00T mixed pilot; K7 currently refuses it. |
| V7: fixed signed z-sum, isotonic map, ncal=3000, MIN_CAL=30, stale fallback, tie ε=1e−6 | Library pseudoqueries fit moments and PAV; feature signs/weights and sample count hand-set. Isotonic mapping is monotone, so it cannot repair within-regime ranking. | A: honest cross-fitted error calibration; B: closed-form risk LDA implemented and rejected as controller selector; C: causal score. **Inactive for current guard-only verdicts**; refitting V7 alone cannot change their HIT/MISS decisions. |
| Event/burst extras: events=none, method burst=0; dormant dispersion 1.0 (optional quantile), vote .8, return margin .1/count1/hold4 | Event variants did not establish a better l10 frontier; defaults exist in MixedJudge. Plugin burst separately defaults to 1. | A can fit dispersion quantiles, B associations, C intervention utility. Keep disabled. Do not tune inactive settings. |
| Blind B; serving choice | Hand B1/B2 trials. Tail B1 wins at 50; phase B2 offers lower IR at 500. B2-vs-B1 SR equal in the measured 500 K7 run. | A gives the exact **tail** bound `B≤floor(H/5)−1`: π0.5 1, GR00T 2. It cannot certify a phase budget. B reports observed frontier, C/new randomized blindness would identify its effect. Keep π tail B1 as primary comparator. |
| Gripper-event/terminal mass .20; near-terminal last two rows | Heuristic look gates, not policy-MISS thresholds. Events inspect candidate chunk sign transitions and successor boundary. | A: estimate event probability/calibration; selecting .20 requires false-look versus missed-event utility, not just event labels. B/C for control value. **Unused by winning budget-only tail serving**. |
| Displacement residual .5; sensitivity .25/1 | Normalized expected kernel displacement vs actual displacement. Threshold fixed; distribution changes strongly with horizon and controller (measured above). | A: demo residual quantiles; B: domain-specific grouped calibration implemented on saved rollout windows; C/new randomized looks for benefit. **Unused by budget-only tail**. |
| Blind low motion: task normalized p10, two intervals; std floor .05 | Task state std and p10 fitted; floor/count hand-set. Different units from raw stock stuck motion. | A: percentile/scale; B: false-alarm timing; C: look benefit. Keep distinction explicit; unused by budget-only tail. |
| Phase penalty .05, offsets h−1,h,h+1, monotone advance≤2 | Hand local clock prior/structural support. Absolute-state alignment beat displacement alignment in R4 A’s offline windows, yet sparse phase still failed SR. | A: moment likelihood-prior fit implemented; B: validate alignment episodes. Offsets/horizon/monotonicity are legal-support constraints. No phase rule is used by anchor tail. |
| Control-step offsets 0..4 (K1 G/GS) | Analytic chord projection/rounding, not the phase-particle ±1 offsets. G and GS change representation/synthesis and were pending. | A: bounded projection is closed form; no outcome evidence presently ranks offset families. Not part of current winning baseline. |
| Quantile h, window W=1000, optional τ0, step0=judge, cap=0 | R3 tested h=.7/.5; W and initial thresholds engineered. Controller is server-wide, includes ±∞ for forced decisions. | A/B solve target h from a specified cost budget; B calibrates finite-window/transient behavior. C needed for optimal SR allocation. Guard-only has no active h. W/τ0 affect warmup and drift; no isolated SR evidence. |
| Periodic k and its clock | Tested 3/5 at 50, then 6; tested 8/12 at 500. Nonblind clock resets each episode; blind mode uses reserved server decision index. | A/B solve finite-length expected call rate, not SR. Actual trajectories change episode lengths. Never substitute “every kth vision request” for every kth decision. |

Additional inactive insurance knobs (`norm_cap=0`, `hyst=0`, AWM3 gripper commit/terminal masks off) remain off under the settled negatives. The kernel still synthesizes full stored chunks on valid action dimensions; only the first five steps are scored/executed at a request. No fitted threshold authorizes an unanchored episode.

## 4. Solver results and retrospective ranking test

### 4.1 A restricted solver that can produce concrete configurations

The broad library audit deliberately tests several objectives, rather than assuming action error is the answer:

- **Stale action RMS:** sigma-normalized five-step heads on metric-refitted held episodes.
- **Successor-state RMS:** predict normalized next-state displacement by the retrieved kernel. State standard deviation is task-local with the existing `.05` floor. This is a local dynamics proxy, not observation of object progress.
- **Conditional successor likelihood:** kernel-weighted Gaussian mixture over candidate displacements. Diagonal residual variance is fitted from cross-episode action-near training pairs, with numerical variance floor `1e-8`; evaluate held-query NLL per valid state dimension. Thus this is an actual leave-out likelihood calculation, not a rebranding of action RMS.

The likelihood fit can favor an overly specific residual model: it does not prove that maximizing density gives a better controller. In particular, `nn=1` is selected by that broad objective in every cell/scale; no corresponding closed-loop arm establishes its value. Changing `nn` also changes the fitted residual variance. Use its numeric result as model evidence, not an SR ranking.

For a reviewable R5 configuration I restricted selection to **one** existing change from `{main ridge 1, kref 5, kref 8, state multiplier 3}`, keeping 16 members, PCA-64, early branch, continuity and control fixed. A candidate must improve **both** held-episode losses by more than one standard error across ten task means. Among survivors choose the lowest successor-state loss. Recompute that choice in each nine-task training fold; retain baseline if the resulting held-task mean worsens either objective. The one-SE rule and finite candidate family are declared design choices, not a theorem about SR. This rule was devised after reading historical results: its historical checks are **retrospective**, not a secretly prospective benchmark.

`select_configs.py` freezes the following values. Scores below use current-library units at 50 and stock benchmark units at 500; see the provenance disclosure in §1.

| Cell | Scale | Solver setting | Action RMS: hand → selected | Successor RMS: hand → selected |
|---|---:|---|---:|---:|
| π0.5 spatial | 50 | main ridge 1; kr5 | .447174 → .433759 | .137857 → .133552 |
| π0.5 spatial | 500 | ridge .1; **kr5** | .301880 → .296315 | .083988 → .082459 |
| π0.5 l10 | 50 | retain ridge .1, kr5 | .363628 → .363628 | .099140 → .099140 |
| π0.5 l10 | 500 | ridge .1; **kr5** | .266090 → .263608 | .061428 → .060696 |
| GR00T spatial | 50 | main ridge 1; kr5 | .452935 → .441609 | .148113 → .143666 |
| GR00T spatial | 500 | **state multiplier 3**; kr8 | .355871 → .349400 | .100606 → .094877 |
| GR00T l10 | 50 | retain ridge .1, kr5, state 1 | .444217 → .444217 | .122321 → .122321 |
| GR00T l10 | 500 | **state multiplier 3**; kr8 | .340180 → .327565 | .081366 → .074200 |

The GR00T l10 50 retention is consequential: the attempted selector’s outer-task action delta was **+.002138**, despite a state delta of **−.000309**. It fails the two-objective retention rule. The six changing cells’ outer-task action/state deltas are respectively: π-sp50 **−.013415/−.004306**; π-sp500 **−.005565/−.001529**; π-l10-500 **−.002482/−.000732**; GR-sp50 **−.011326/−.004447**; GR-sp500 **−.006471/−.005729**; GR-l10-500 **−.012615/−.007166**. The two unchanged cells do not need duplicate solver arms.

The final selected settings are **identical when the 500-library audit uses only deployed-library action units** (`measurement_summary.json`). Thus the recommendation does not depend on borrowing the current library’s units; the stock-unit table is a parity/sensitivity check for the existing serving pipeline. Compare losses within a cell and scale, not raw normalized loss magnitudes across models or differently normalized libraries.

An analytic covariance answer is not automatically better. With `S=mean(dx dxᵀ)`, dimension p, n pair differences, μ=tr(S)/p and a=mean(S²), OAS gives

`γ = min(1, (a+μ²)/((n+1)(a−μ²/p)))`, `Sγ=(1−γ)S+γμI`.

Its equivalent AWM ridge is `λ=γ/(1−γ)` up to an irrelevant common distance scale. Mean fitted λ at 50 is **.062644, .030095, .075305, .030964** for π-sp, π-l10, GR-sp, GR-l10—below the hand `.1`, whereas both spatial LOEO fits favor stronger main ridge. The formula treats dependent pair differences as independent observations; it solves that covariance model, not closed-loop action retrieval risk. The separate LDA uses the same numerical shrinker in only eight dimensions; good AUC still does not identify CALL benefit.

The corresponding 500-library means are **.015129, .004684, .020923, .005664**. More pairs make the independence-based shrinkage estimate smaller, but there is no closed-loop OAS-versus-hand-ridge test at either scale.

The early branch warrants separate protection. Mean early covariance ranks at 50 are **10.53, 10.88, 10.84, 10.96 out of 136**. π0.5 l10 demo step-zero RMS favors main `.234421` over early `.266097`, despite R2’s recorded-query evidence favoring the early branch in important cells. That disagreement is another domain-transfer warning. The new metric candidates change **main** ridge only; no early-branch removal is proposed.

### 4.2 Borrowed prior α: LOEO and a closed form disagree

`prior_solver.py` fits both covariances in the **same big-library PCA and standardization**, then removes the current episode from its current covariance/candidates. This is **borrowed big-library information** at every α, including zero. In addition to the α grid, it estimates current covariance variance by the episode jackknife and computes

`α_moment = clip(V_jackknife / ||S_current−S_big||²_F, 0, 1)`.

This is a target-shrinkage moment heuristic. Its premise is imperfect here: nearest-action pairs at 5 versus 50 episodes/task need not estimate the same population covariance. It should not be sold as an exact SR-optimal empirical Bayes solution.

| Cell, current candidates | α selected by aggregate action/state LOEO grid | Action RMS at α0 / α.5 / α1 | Median moment α |
|---|---:|---:|---:|
| π0.5 spatial | .75 / .75 | .446384 / .416691 / .421641 | 1.000000 |
| π0.5 l10 | .5 / .5 | .363037 / .348403 / .351730 | .960091 |
| GR00T spatial | .75 / .75 | .447309 / .421243 / .421486 | 1.000000 |
| GR00T l10 | .5 / .5 | .442924 / .433207 / .435292 | 1.000000 |

The LOEO grid weakly supports an intermediate prior, but does not recover a universally optimal `.5`: `.75` was not run in the historical loop. The moment formula chooses nearly full borrowing. R3’s π l10 α1 pilot was `.49` versus `.53` for α.5 and `.59` for the rerun baseline; this is a **directional miss with low power**, not a significant α1-versus-.5 result. Full α.5 runs improved π spatial/l10 by **4.8/4.4 pp**, while GR spatial/l10 moved **+2.2/−.8 pp**, both unresolved on GR; GR l10 task 8 fell **.42→.08**. An α.5 action-error gain therefore did not reliably predict model/task transfer. At 500, the proposed counterpart is simply the deployed big-library fit; no additional α parameter or extra demonstrations are needed.

### 4.3 Hits, misses and tests that history cannot identify

| Historical contrast | Offline forecast and observed result | Verdict |
|---|---|---|
| π-sp50 main ridge .1→1 | Dual held-episode losses improve; actual SR .800→.840, paired p=.009 in R3 ANALYSIS | **Useful retrospective hit**, same library, no added calls. |
| GR-sp50 ridge .1→1 | Both new losses improve; actual .888→.894, unresolved | Weak directional agreement; **not evidence of a useful SR gain**. |
| noprog3→4, both scales | Replay predicts lower cost; actual cost is higher than replay and SR drops 5.0/5.6 pp | **Miss as a utility selector**. No-progress safety cannot be chosen by marginal call savings. |
| Periodic k5→6 at 50; k8→12 at 500 | Cost decreases; SR decreases .028/.022, each within roughly two paired SEs | Cost direction hit; **no offline SR ranking identified**. No monotone “largest affordable k” optimality claim. |
| K7 B1→B2 at 500 | Longer-horizon action/state drift predicts more error; actual SR .862/.862 and IR .225903→.177621 | Error cannot rank value per IR. SR equality itself remains noisy. At 50 no matched B1 phase arm exists; do not invent it. |
| Dense→vision-confirmed stuck | Successful-demo false alerts favor vision confirmation at both scales. At 500 B0 IR .270660→.244534; SR .842→.830 unresolved. B2 IR .211194→.177621; SR .850→.862 unresolved | **Cost/mechanism hit**, no proved SR improvement. No 50 dense-vs-K7 matched loop pair. |
| Phase B2→tail B1 | Both cached action error and cross-task risk LDA favor phase; actual tail gain +.106 at 50, +.018 at 500 | **Strong miss at 50**; unresolved SR ordering at 500. Do not optimize blind serving by either score. |
| Guard-only versus periodic / V7 / events | R3 l10 periodic k5 .792@.314858 beats g50 .740@.323416; spatial directed V7 had the opposite advantage at its higher budget (R3 ANALYSIS). Fitted risk weights also change by scale. | No universal fused confidence/guard objective demonstrated. Event/burst changes are not single-threshold isolates. |
| kref5 versus 8 | R2 recorded stale errors often favored 8 at 500; my held-demo objective favors 5 on π. R2 CL2 changes library **and** kref together. | **Unidentified in closed loop**; this is a prospective R5 test, not a retrospective hit. |
| α prior | Grid supports intermediate borrowing; moment solution goes near 1; full GR l10 transfer is poor despite proxy improvement | Partial π agreement, GR transfer miss; α1 pilot too small for a strong quantitative verdict. No separate α contrast at 500. |

There is no defensible “X out of Y ranking accuracy” from these rows: several contrasts are confounded, unrun or statistically unresolved. In particular, the new successor objective is more mechanically relevant than action error, but it has **not** passed a broad historical SR-ranking test. The R2 report’s eight method-error/SR correlations average roughly .02; that remains the correct warning against promoting a proxy into the exam.

## 5. Cost inverses, bytes and effect accounting

`cost_solver.py` implements the closed-form budget inverse

`m_target=(ρ−.152 v)/.848`, `h_target=1−m_target`.

For all-vision operation, target IR `.178/.203/.242/.280/.315` implies h **.969340/.939858/.893868/.849057/.807783**. These solve a **supplied budget**, not the optimal budget. A forced-guard MISS floor can make a target infeasible: stock l10 guards already spend m=.202142 at 50 and .101670 at 500. Holding that behavior fixed cannot meet all-vision targets .242 or .203 respectively by adjusting V7 confidence. Quantile adaptation also has finite-window, regime and forced-event transients; its configured h is not an exact realized hit fraction.

For nonblind periodic k on frozen episode lengths `n_e`, `M=Σ floor(n_e/k)`, not N/k exactly. On stock-guard l10 lengths, the minimum integer k meeting budgets `.242/.280/.315` is **9/7/6** at 50; for `.178/.203/.245` it is **26/15/9** at 500. These are conditional arithmetic answers, not recommendations to run those periods. For example, the actually executed k5 run already reaches `.314858`, while the frozen-length solver picks k6 for a `.315` target: trajectory lengths change after the control change. This is why an inverse cost formula needs closed-loop accounting even when no hyperparameter search remains.

Each extra MISS must save **.848/.152 = 5.578947 vision decisions** merely to break even in the owner model. For B1 tail, on a stationary simplified chain with MISS probability p per vision decision and tail admission r per eligible vision HIT, `v=1/[1+r(1−p)]`, `m=pv`. The measured controller can change p and r, so do not assume `v=1/2` just because B=1. Actual l10 tails have V/N=.595331 at 50 and .572798 at 500.

The following are **actual existing serialized fit sizes**, measured by file stat for R2 AWM and the available K7 tails, plus K7’s documented 50-spatial verification fit. MB is decimal, not MiB. A new scalar setting keeps the same array shapes; its exact new pickle size must be measured at coordinator prefit.

| Model/suite | Original deployed pkl MB | AWM fit bytes, 50 / 500 | K7 tail fit bytes, 50 / 500 |
|---|---:|---:|---:|
| π0.5 spatial | 430.792483 (≈431) | 21,341,661 / 46,993,114 | 26,076,510¹ / 66,500,552 |
| π0.5 l10 | 1,103.155631 (≈1103) | 24,631,208 / 94,143,229 | 32,704,527 / 142,348,562 |
| GR00T spatial | 429.351282 (≈429) | 22,249,307 / 58,156,565 | unsupported by K7 |
| GR00T l10 | 1,068.314575 (≈1068) | 26,672,701 / 117,303,700 | unsupported by K7 |

¹ Same representation, from `r04/k7_guard/HANDBACK.md`/`results/footprint.json`; a served 50-spatial tail fit was not present. The π l10 K7 fits are **2.96% / 12.90%** of the original pkl at 50/500. The largest GR AWM fit, l10-500, is **10.98%** of its original. These sizes include padded actions and fixed/auxiliary arrays; they are not just the advertised 580/626/632 bytes per entry. They do not include model weights, the cold store or duplicated server processes. Pure-cache GR blind-tail alternatives have documented K1 sizes 22.842/58.604 MB spatial and 28.136/118.431 MB l10, if subsequently needed; no new mixed-GR footprint is invented.

Both proposals below retain candidate libraries. They add at most a small scalar/table artifact (reserve **64 KiB**, a storage allowance, not a measured new fit). Main-ridge/state changes are **method at fixed library**; kref is a **synthesis-bandwidth change at fixed library/geometry**; 50→500 is a **library effect** and must remain a separate panel. Blind tail and learned CALL overrides are **control effects**, with every policy call charged. No new proposal changes the action representation or adds nonlibrary synthesis. The best observed tail SR includes an execution-horizon effect: FINDINGS’ π l10 full inference L10 is `.904 @ .5` versus L5 `.844–.850 @ 1`. Do not credit that gain to a metric solver.

Search remains outside the primary IR. FINDINGS’ 1.2–2.2 ms CPU method cost is an additional roughly **(.0178–.0326)×v** in the 67.52 ms owner denominator; the GPU prototype’s lower overhead is reported evidence, not used here. No GPU timing was run. GR00T’s primary stage shares are not supplied on this owner CUDA-graph basis: pure-cache numeric IR `.152` below is explicitly a **π-cost-equivalent proxy**. GR’s actual pure-cache ratio is its stage-1 share; do not substitute eager measured shares and call them the owner basis.

## 6. Ranked R5 proposals

### Rank 1 — `dual_loss_metric`: one frozen demo-solved setting, unchanged control

**Pitch.** Replace a small set of hand defaults with the restricted, grouped solver in §4.1. Its value per IR is potentially high because fitting is offline and serving adds no policy calls by design. The forecast is modest: it is a test of whether a locally better metric transfers, not a claim that offline error now ranks SR.

**Hypothesis/mechanism.** At 50, poorly estimated within-action covariance benefits from stronger main ridge; at 500 the π bandwidth can be narrower, and GR can use more state weight to localize the robot within a visual neighborhood. Requiring both action and successor-displacement improvement rejects some one-proxy choices. The π-spatial ridge result is supporting evidence; the GR-spatial non-result and phase counterexample limit confidence.

**Algorithm and implementation.** Use `library_solver.py`, then `select_configs.py`, freeze the resulting JSON/hash before any R5 outcomes. The fit remains AWM’s same-task action-neighbor covariance and kernel-16. `Method.fit` reads only candidate/fit libraries and fixed context units; `reset` and online `query` remain existing implementations. Return the normal `Result(topk,scores,confidence,action,library,extras)`. No query `a_inf`, episode success, future state or episode length is exposed online.

`arms_proposed.json` contains 16 reviewable hand/solver rows for the eight cells; two identical-cell pairs have `run_needed=false`. Names are `r5b_{p|g}_{sp|l10}_{50|500}_{hand|solve}`. π arms use `VisionConfirmedBlindMixedJudge` with `anchor_tail`, B=1, `gates=budget_only`, `noprog_span`, no events, `stuck_guard=vision_confirmed`, `noprog_n=3`, `--os-blind --os-judge guard_only --os-no-shadow-native`. Their base is `BlindAWM`, or `BlindAWM3` for the ridge comparison. The hand twin also uses `BlindAWM3(ridge_main=.1)` when required, so a change in tie handling is not mistaken for a ridge effect. GR arms use pure-cache `AWM`/`AWM3`; K7 is not silently ported to GR. Coordinator prefit must supply its own artifact/output paths and ordinary run YAMLs; the JSON is a proposed configuration list, not a claim to have emitted or launched servers.

**Plugin/server changes:** **none** for these settings. Existing kwargs implement `kref`, `state_scale` and `ridge_main`; early ridge remains `.1`. Existing fit/query/confidence calibration is rerun using the selected selector. W, ncal, the guard values and blind parameters remain fixed. k stays 16, so K1’s anchor contract is satisfied. Exact new fit bytes and API parity are coordinator preflight checks; constructor compatibility is verified here.

**Forecasts, not measurements:**

| Cell/control | 50-library forecast | 500-library forecast |
|---|---|---|
| π l10 K7 tail | Solver retains hand config: same behavior, expected `.806 @ .241513` up to run noise; no duplicate arm needed | Expected ΔSR 0 to +2 pp, uncertainty roughly −3 to +3 pp around hand `.880`; forecast IR **.19–.22** around `.203281`, because changed picks can change guards |
| π spatial | Ridge-1 pure-cache historical check already `.840 @ .152` vs AWM `.800`; the **K7 tail** hand SR/IR at 50 is unmeasured, so predict only relative ΔSR **−2 to +3 pp**, ΔIR **−.015 to +.015** | Forecast SR **.97–.99**, IR **.12–.14**, around hand tail `.982 @ .128236` |
| GR spatial, pure cache | Solver recovers historical ridge-1 `.894` versus `.888`; no established improvement beyond noise. IR proxy `.152` at either setting | Forecast SR **.95–.98** around AWM `.966`, IR proxy `.152`; no calls, every decision has vision |
| GR l10, pure cache | Retains hand AWM: `.552`, IR proxy `.152`; no duplicate arm | Forecast SR **.69–.75** around AWM `.706`, IR proxy `.152`; actual GR IR remains its unchanged stage-1 share |

The central expectation is small or zero SR gain; the ranges are explicitly judgmental forecasts, not fitted confidence intervals. At 50, this solver mainly automates known defaults; its prospective value is the four 500-library candidates. Comparison against the overall historical pure-cache frontier should also display borrowed α.5 and V6 results separately: beating CL2 does not establish beating every existing arm.

**Hand-best controls are explicit, not silently replaced by CL2.** `frontier_reference_arms.json` adds the original recipes for π-sp50 borrowed α.5 (pure-cache SR .848) and V7 h70 (.980 at its higher IR), GR-sp50 borrowed α.5 (.910), GR-l10-50 V6 (.606), GR-sp500 V6 (.976), and GR-l10-500 V6 (.736). Borrowed references are labeled. At π-l10 both scales and π-sp500, the matched K7-tail hand arm is already the relevant hand-best comparator. The primary GR-l10-500 pilot therefore has **AWM hand, AWM solver, and V6 hand-best**, not just the two AWM variants. If warranted, a fourth predeclared arm inserts the selected state multiplier into V6's unchanged base kwargs; that tests whether the method effect survives the best recovery controller. A gain only over CL2 is insufficient to claim a new frontier. At 50, retaining the AWM defaults does not imply that AWM beats V6 or the borrowed prior.

**Cost tier/bytes.** Served method **T1**, existing closed-form AWM fit, identical representation shapes and the corresponding §5 fit footprint plus at most the 64 KiB configuration allowance. The audit’s repeated LOEO/grid work is longer than a single seconds-level statistical fit; I did not benchmark it as deployment latency. No extra runtime neural network or library episode is stored.

**Cheapest diagnostic.** Already done: grouped losses, outer-task retention, fixed-population verification and constructor checks. Before launching, prefit the actual artifacts, verify action/score parity for unchanged configurations on held recorded queries, check both early and fresh regimes, and record realized bytes. No action-error gate alone is an SR acceptance test.

**Closed-loop pilot.** Primary: π l10-500 and GR l10-500, each **hand versus solver**; include the 50 counterparts as retained-configuration controls, not distinct methods. Parallel confirmation cells are π/GR spatial at **both scales**, with the 50 ridge cells serving as known positive/weak controls. Use the same 10 tasks × 50 pruned-A init indices `0..49`, environment seed 7 and paired original-init mapping as the prior protocol. If a cheap integration screen is needed, use every task × inits `{0,10,20,30,40}` solely to detect collapse, then complete all 500. Do not select a winner from that 50-episode screen. Repeat close frontier comparisons with a second policy-noise seed; <3 pp on l10 is unresolved in a single run under FINDINGS’ noise estimate.

For π l10-500 also retain the existing K7 phase-B2 hand point `.862 @ .177621` on the frontier plot; changing from tail to phase is not part of the metric comparison. The 50 phase arm stays dead.

**Kill criterion.** Reject the solver as an SR-selection objective if its frozen candidates repeatedly worsen SR by ≥3 pp or raise IR by >.015 without a compensating SR benefit. For a close result, use paired uncertainty/repeats rather than declare victory from a positive mean. If it fails, do not tune the one-SE multiplier on those same R5 outcomes and call the next answer offline prediction.

**Variants.** One small, predeclared variant is pure-cache π twins to separate a synthesis/method effect from a change in guard calls. A deployment-only normalization fit can be compared to the inherited benchmark-unit fit as a separately labeled method ablation. Do not combine it with the kref contrast without its own matched control. PCA/kernel-size/nn changes remain diagnostics, not additional closed-loop sweep arms.

### Rank 2 — `landmark_call_value`: solve a causal call budget offline after K5

**Pitch.** Use randomization to solve the quantity the guard should care about: CALL-minus-CACHE effect on success and total remaining cost. Unlike failure risk, that effect can identify which eligible calls can be omitted. The intervention remains **one designated opportunity per episode**, the support of K5.

**Current status.** Formal `r04_k5` accepted journals were absent for both scales when `cost_solver.py` ran. The product records “no fit/no effects invented.” Only the 8-episode smoke pair exists; it is not training evidence. The implemented fitter rejects it as unsupported and emits no overrides. GR00T, spatial and blind K7-tail causal effects are unknown.

**Algorithm.** Read raw accepted-attempt decisions through `r04/k5_rand/estimate.py:load_arm`, validate complementary assignments, original-init pairing, episode continuity and identical controller contracts. Require the complete 500-init pair separately for g50 and g500. K5 assigns the first or third baseline MISS landmark and CALL/CACHE at probability .5; do not train on post-intervention features.

For each supported pre-intervention context and landmark estimate HT differences in `Y=success`, `N=decisions`, `M=policy calls`, clustering both replicas by `(task, original init)`. For baseline target ρ use

`ΔCρ = .848 ΔM + (.152−ρ) ΔN`.

This charges *all* subsequent calls and changing episode lengths, rather than just the immediate saved call. For future blind randomization the correct extension is `.152 ΔV + .848 ΔM − ρ ΔN`.

`cost_solver.py:causal_fit` fits a small linear program: maximize a lower bound on population cost saving from suppressing eligible baseline calls, subject to an upper bound of **.01 absolute SR loss**. This 1-pp tolerance is a proposed utility constraint, not an estimated biological/numerical constant or an owner mandate. A context must have ≥30 init clusters and ≥10 samples in each treatment; otherwise its suppress probability is zero. Bounds are cluster-normal with Bonferroni correction across tested leaves, **approximate**, not exact small-sample guarantees. Fit on four init-modulo-5 folds and evaluate on the fifth before a full-data fit. The held-out IPW policy difference is in the returned artifacts. If no context clears support and uncertainty, the solution is baseline CALL everywhere. With up to 48 leaves, that is a plausible result at 1,000 episodes and must be accepted.

**Plugin/server change required for deployment.** Add an opt-in frozen-table path at the existing verdict stage, after the base’s guard verdict/context is computed and before the action is committed. Reuse K5’s original-init mapping, first/third opportunity assignment and pre-intervention context definitions. At the assigned eligible opportunity, hash an independent per-episode policy coin against the fitted suppression probability; keep a `used_override` bit. Suppression serves the already-computed cache candidate, logs baseline versus actual verdict and the table hash, and then ordinary baseline behavior resumes on actual history. Reset the bit/counter on episode reset. Preserve the full candidate for CACHE and the full-priced policy execution for CALL. No generic “skip after N bad calls,” multiple interventions, or extrapolation from a first landmark to every decision.

The `Method` selector itself is unchanged; the new table belongs in plugin/session control, because K5’s original-init assignment and treatment bookkeeping live there. The current randomized overlay rejects `--os-blind`; do not bypass that restriction and assume estimates transfer to K7. To apply the idea to the best tail controller, the coordinator must first collect the analogous one-landmark randomized data **under that exact tail baseline at 50 and 500**, with V included in the ledger. No such data are available now.

**Forecast and cost bound.** One omitted call per episode, with no downstream change, can directly save at most **.012861 IR at 50** (`.848*500/32967`) and **.014451 at 500** (`.848*500/29340`) on stock l10 guard lengths. The actual supported subset will save less; subsequent calls/length can offset or amplify it. Before K5, expected absolute SR change is **unknown**; the intended acceptance range is baseline minus 1 pp or better, not a forecast proved by observational data. A reasonable planning scenario is g50 near **.73–.76 @ .311–.323**, g500 near **.85–.87 @ .224–.238**; these are hypotheses conditional on harmless direct omissions and exclude trajectory-induced cost increases.

These stock-guard scenarios do **not** beat the existing K7 tail frontier by themselves. They are the cheapest causal identification step. For a later correctly randomized tail version, the analogous direct-only ceiling is .013596 at 50 and .014677 at 500; planning intervals would be IR **.228–.242 / .189–.203**, with SR near the matched tail baseline **.806 / .880** if omissions are harmless. Its SR effect is presently unmeasured. On spatial and GR00T at both scales, default to the unchanged hand controller/selector; there is no supported causal forecast or deployment transfer.

**Tier/bytes.** **T2, small CPU policy fitting** (effect estimation plus constrained LP), T0 lookup at serving, underlying AWM T1. No neural model. Existing g50/g500 π-l10 fits are about **32.602/141.227 MB** (`r04/ideation_B` storage table); add at most the reserved 64 KiB policy table versus the original 1,103.156 MB pkl. A later tail table uses the 32.705/142.349 MB fits in §5. A causal model trained from these rollouts is **borrowed big-library information** even if its stored table is tiny.

**Cheapest diagnostic.** Once formal pairs arrive, run the existing K5 identity/assignment audit, this fitter’s held-init evaluation, and a task-held-out sensitivity analysis before any new loop. Require positive cost saving and acceptable SR uncertainty on held-out data; neither high LDA AUC nor in-sample LP value qualifies. If context cells are unsupported, use the baseline or a predeclared parent-only first/third-landmark variant, not invented counterfactual labels.

**Pilot.** At **both 50 and 500 π l10**, compare baseline guard-only, the frozen solver table, and an equal-number-of-overrides randomized landmark control. Match 10 tasks × all 50 inits, same seed/mapping; no blind mode in this first test. This distinguishes selecting valuable opportunities from simply spending fewer calls. Include the hand K7-tail frontier in the report as an external performance comparator, not as an interchangeable treatment baseline. Only after a successful causal test collect/fit/pilot the tail-baseline extension at both scales.

**Kill criterion.** No arm if there is no supported held-out cost improvement. Kill the controller if paired SR loss exceeds the predeclared tolerance or if the saved immediate call is repaid by later calls/longer failures so total IR does not improve. A confidence interval too wide to distinguish a useful tradeoff means “not identified,” not permission to deploy every-decision suppression.

**Variants.** Parent-only first/third landmark tables reduce variance; shrinkage between parent/leaf effects can be fitted on training folds. A shadow-only LDA can be retained for descriptive monitoring. Neither variant may change the supported treatment frequency without new randomized data.

## 7. Rejected ideas

1. **Fit a failure classifier or guard LDA and call whenever its score is high.** The implemented LDA’s LOTO AUC improves over chance at both scales, but it reverses phase-versus-tail risk (`.202/.248` versus actual `.300/.194` at 50; `.093/.107` versus `.138/.120` at 500). Rescue opportunities, inherently hard states and states that are already beyond rescue share failure labels. V7 error calibration also cannot change a guard-only verdict. Keep the diagnostic; reject it as an offline causal solver.

2. **Choose the cheapest period, longest blindness or loosest no-progress count that looks acceptable on a frozen path.** Noprog-4’s replay underprices actual IR by **.017986 / .014851** at 50/500 and misses **5.0/5.6 pp** SR loss. The h=2 residual quantile transfers from inference to cache at about **15%**, not 5%, exceedance. The exact tail horizon bound is useful; it does not prove that phase B2 is safe in a sparse library. No new noprog-4, 50 phase-continuation, or unbounded blindness proposal.

3. **Use a single closed-form covariance/noise optimum as the SR optimum.** OAS’s 50-library mean ridge is only `.030–.075`, while the successful π-spatial ridge intervention is `1`. The episode-jackknife prior solution is median α≈`.96–1`, while the held-demo grid favors intermediate α, and the α1 l10 pilot is directionally worse. The likelihood objective selects nn1 throughout, without any matching SR evidence. Closed forms solve explicit statistical assumptions; they do not eliminate modeling choices or the closed-loop exam.

The deliverable is a frozen, falsifiable R5 configuration set and several useful offline solvers with explicit domains of validity. The unresolved fact is the causal value of CALL/CACHE on the best controller’s state distribution; formal K5 data and the coordinator’s paired R5 pilots are what can resolve it.
