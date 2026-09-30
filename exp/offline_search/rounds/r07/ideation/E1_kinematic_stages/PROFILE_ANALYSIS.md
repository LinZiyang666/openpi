**E1 PROFILE recommendation: freeze 28 eval500 arms — SF1 ×8, UF1 ×8, SW ×4, CU ×4, CT ×4. Choose SF cap E=1 everywhere. Drop SF2 from this evaluation; defer SF+SW and SA.** All individually profiled candidates pass their applicable §4 numerical gates; SF2 is a cap-selection decision, not a failed gate. Keeping CT authorizes a falsifiable experiment, not a claim that stage call allocation works.

Scope: [SELECTION](../../SELECTION.md), all four coder hand-backs, [C4 closed-loop report](../../profile_results/closed_loop/profile_report.json), its four offline output directories, and accepted journals/server decisions under `/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/runs/`. My [analysis script](analyze_profile.py) independently reconciles **44 arms, 880 episodes, 37,999 decisions**. Detailed derived evidence: [metrics](profile_metrics.csv), [raw audit](profile_raw_audit.csv), [call audit](profile_calls_audit.json). No serving implementation changed.

IR below is C4's owner-priced cost divided by accepted decision requests; these requests nominally cover five controls. “Paired saving” means the equal-episode mean of `IR(A episode) − IR(variant episode)` on the same 20 task/init pairs. It differs from subtracting arm-level ratios of totals. Both versions pass the saving gates. SR counts throughout are **descriptive only**.

| E1 proposal / descendant | Decision | Profile gates and own kill criteria |
|---|---|---|
| P2: stage-contained look less → SF1 | **Keep, cap 1** | Grants 19.09–48.42% of anchors; paired saving .007947–.016951 in 8/8 cells; minimum ratio-of-totals saving .006707. Disabled-A parity passes. Savings survive actual valve/LOOK work. SR noninferiority and added value of the gate remain unproven. |
| P2 → SF2 | **Drop from frozen list** | Grants 14.23–38.15%; paired saving .011463–.023638; minimum ratio saving .009449. Passes §4, but adds only .004446 mean arm IR saving over SF1, with 125/160 versus 130/160 successes. No demonstrated reason to accept the longer commitment. |
| P2 control → UF1 | **Keep** | Grants 52.47–82.74%; paired saving .017252–.022889; minimum ratio saving .015382. Parity passes. Required comparator for whether stage/valve signals add value. |
| P1: soft event allocation → CT; CU comparator | **Keep both at ρ=.30** | CU IR .298992–.307100; CT .292655–.318419, all within [.27,.33]. Both 68/80 successes. Equal-IR stage utility has not passed my original kill criterion; 20 episodes cannot resolve it. Retain the preregistered full test, with no cell-specific rescue. |
| Shared stage signal → SW | **Keep π0.5 ×4** | Wrist looks 22.95–40.57%; paired saving .012486–.022705; ratio saving .011302–.020393. Real-GPU key, completed-input and policy parity passes 12/12 observations. This is a separate camera sufficiency hypothesis, not a consequence of geometry. |
| P2 + camera lever → SF+SW | **Defer ×4** | Built with cap 1 and CPU replay checks, but absent from the 44 closed-loop arms. Joint eligibility and ≥.005 realized saving have not been measured. Component results do not establish the composed controller's gates. |
| P1/P2 composition → SA | **Do not build now** | No joint closed-loop profile or cadence/camera-aware .30 calibration. Hooks exist; reusing CU/CT λ after changing anchor cadence would not establish its budget gate. |
| Composite kinematic difficulty D / semantic phase gating | **Remain dropped / narrow the claim** | Original D failed cross-cell consistency; the profile does not repair it. Keep local gripper-mode segmentation as telemetry and a continuation screen. No validated causal tracker resolves repeated poses or unseen semantic event sequences. |

**Follow results across all eight libraries.** Entries are `successes/20 @ owner IR`; grant percentages are SF1/SF2/UF1. Sp denotes Spatial.

| Cell | A | SF1 | SF2 | UF1 | Grants % |
|---|---|---|---|---|---|
| π0.5 L10-50 | 15 @ .076294 | 14 @ .065437 | 11 @ .061404 | 17 @ .055131 | 35.26 / 26.89 / 78.77 |
| π0.5 L10-500 | 20 @ .076683 | 20 @ .067876 | 18 @ .064221 | 13 @ .055988 | 26.62 / 19.00 / 73.86 |
| π0.5 Sp-50 | 15 @ .077279 | 13 @ .067236 | 16 @ .061805 | 14 @ .060855 | 30.35 / 25.84 / 52.47 |
| π0.5 Sp-500 | 18 @ .077813 | 18 @ .062376 | 18 @ .056429 | 18 @ .055273 | 48.42 / 38.15 / 80.95 |
| GR00T L10-50 | 14 @ .074268 | 14 @ .064866 | 13 @ .060382 | 14 @ .055023 | 31.02 / 23.60 / 71.22 |
| GR00T L10-500 | 18 @ .074669 | 17 @ .067962 | 16 @ .065220 | 17 @ .052920 | 19.09 / 14.23 / 82.74 |
| GR00T Sp-50 | 17 @ .075189 | 16 @ .064162 | 15 @ .058652 | 15 @ .059806 | 36.44 / 28.50 / 53.00 |
| GR00T Sp-500 | 18 @ .075095 | 18 @ .065359 | 18 @ .061590 | 18 @ .054101 | 32.69 / 22.39 / 77.01 |

The single-cap choice is **one extra five-control block**, ordinarily extending a cache commitment from 10 to 15 controls. Equal-cell mean IR is A .075911, SF1 .065659, SF2 .061213, UF1 .056137. Successes are respectively 135, 130, 125, 126 out of 160 cell-episodes. SF2's five-success deficit versus SF1 is a caution, not significant harm established by this profile; it also improves π0.5 Sp-50 by three successes. Do not select different caps by these cell outcomes.

SF1's four-success advantage over UF1 is concentrated in π0.5 L10-500 (20 versus 13); π0.5 L10-50 goes the other way (14 versus 17). The stage signal's value is still open. The selected SF implementation uses gripper-mode continuity and displacement, not my original geometric/slow-knot boundaries: these results do not validate waypoint density or bottleneck tightness as difficulty measures.

The stage screen does change the served actions as intended: **0/2,248 SF1+SF2 extension blocks cross a gripper-command mode**, versus **154/1,902 UF1 blocks (8.10%)**. This includes transitions from the preceding served head and within the extra head, using frozen library mode midpoints. All SF extensions pass logged all-member structural/stage support, age and displacement-valve assertions. These checks address my source/continuation-integrity kill criterion; they do not prove grasp correctness or full execution of terminal heads.

The distinction between models remains important. SF1 serves **495 π0.5 successor-head blocks and 419 GR00T native-tail blocks**. SF2 serves 713 π0.5 bridges, 311 GR00T native blocks and 310 GR00T successor bridges. Thus cap 2 introduces stitching on GR00T as well. [C1](../../c1_follow/HANDBACK.md) additionally establishes exact disabled-A parity on **10,569 decisions / 52,449 recorded controls / 240 episodes**, and 6,612 enabled structural/stage-rejection checks. My new raw audit checks logged provenance/age, not an independent recomputation of every 16-member synthesis.

My P2 fidelity/coverage criterion is only partly addressed: mode changes are excluded while eligibility remains useful, and realized savings persist. There is no matched-coverage action-fidelity or matched-IR randomized SR proof. The [offline follow audit](../../profile_results/offline/follow_all/follow_audit.json) covers 5,341 original-controller anchors, 10,682 anchor×cap records; its pooled old-B-val SF1/SF2 eligibility is 35.06%/29.55%. Those fixed-path opportunities are not current-controller success effects or realized savings.

**P1 and stage difficulty evidence.** The current CT adds first-entry deviation weighting to my event weighting and uses the selected row-weighted occupancy rule. It is therefore a test of the combined stage/event allocation, not an event-only ablation. Fresh-anchor probabilities match `min(1, λw)` in all **1,756** CT anchors; there are **81** high-deviation entries. Clipping occurs on 17.20–23.48% of anchors by cell, so realized budget accounting must retain it.

| Sparse cell | CU successes @ IR | CT successes @ IR | CT−CU IR, ratios | CT−CU IR, paired episodes |
|---|---|---|---:|---:|
| π0.5 L10-50 | 18 @ .298992 | 19 @ .318419 | +.019426 | +.015607 |
| π0.5 Sp-50 | 18 @ .301804 | 16 @ .292655 | −.009149 | −.011894 |
| GR00T L10-50 | 15 @ .307100 | 15 @ .296609 | −.010491 | −.013728 |
| GR00T Sp-50 | 17 @ .304635 | 18 @ .316186 | +.011551 | +.012340 |

All eight call arms pass the .30±.03 profile gate under either aggregation. **π0.5 L10-50 is not matched within .015**, even though both arms pass that gate; its extra CT success cannot be advertised as a stage benefit at equal cost. Across four cells the mean ratio difference is +.002834 and mean paired difference +.000581. Preserve §5's pooled acceptance rule and explicitly report cell costs; do not tune λ on these SR outcomes or silently switch estimands.

The [offline stage-value report](../../profile_results/offline/value_all/) offers no broad reason to expect CT to win: 235/236 reported natural-dose first-entry intervals contain zero, with eight additional comparisons lacking an observed CALL or CACHE branch. The lone positive stratum is `unknown`, with four CALLs and CALL ESS=4. Keep CT as the prospective falsification of P1, then drop stage call weighting if the preregistered full comparison fails; retain segmentation telemetry even then. Held-out-task transfer remains a separate, untested part of my original criterion.

The requested failure chronology also does not validate a “hard stage” failure detector. In [C4's old B-val/P3 clock](../../profile_results/offline/clock_all/failure_clock.json), absolute-deviation crossings occur in **43/43 failures but 184/197 successes**; valve alerts occur in **25/43 failures and 12/197 successes**, confirmed stalls in **26/43 and 34/197**. These are episodes with crossings, not physical failure-onset labels. My original D located only 27.5% of persistent-stall anchors against 29.4% overall hard exposure ([proposal](PROPOSAL.md)); keep D dropped. Local stage continuity is observable, but neither normal proprioception nor event timing establishes that the object interaction succeeded.

**SW and composition.** SW successes/20, IR, and wrist share are:

| π0.5 cell | Successes | IR | Wrist / looks |
|---|---:|---:|---:|
| L10-50 | 16 | .059502 | 217/624 = 34.78% |
| L10-500 | 19 | .065381 | 123/536 = 22.95% |
| Sp-50 | 15 | .063681 | 75/274 = 27.37% |
| Sp-500 | 17 | .057421 | 99/244 = 40.57% |

SW totals **67/80**, paired A **68/80**. [C2](../../c2_wrist/HANDBACK.md) reports 5,173 disabled identity decisions and 1,235 independent wrist-bank anchor matches. The later [GPU artifact](/tmp/r7_C2/gpu_parity.json) supersedes the hand-back's pending-GPU note: all 12 full-path, wrist-key, completed-policy-input and K10 output checks pass, maximum policy difference zero. The offline camera audit still flags π0.5 L10-500: wrist−full head RMS **+.02377 [.00955,.04335]** on old B-val; this is a risk signal, not a success estimate or parity failure.

The four SW rollouts have **zero policy calls**. They validate pre-request camera selection and pure-cache cost, not live mixed-controller wrist-origin MISS handling. SF+SW changes retrieval and subsequent follow eligibility jointly; there is no joint 20-episode profile. Defer it from this freeze rather than asserting its gates passed. If revisited, use E=1 and run the same joint profile gates before test episodes. SA additionally needs a frozen controller, new B-val budget calibration for changed cadence/camera prices, and mixed MISS/completion accounting. The current null CT benefit and unvalidated composition give no basis to build SA now.

**Accounting and possible contract bugs.** I found no confirmed decision-count, owner-cost arithmetic, or logged extension-integrity error in the 44 arms. The following distinctions must be preserved in the hand-off:

1. **Two price tables:** `summary.json.cost_ledger.ir_per_request` uses default eager prices, not the owner table used by C4. Example π0.5 L10-50 SF1: **.061114 versus .065437**; SW: **.055572 versus .059502**. Using the summary value in the R7 frontier would be an accounting error. My recomputation charges actual camera modes, completion forwards and calls, matching C4 within 1e−10 in every arm. Wrist costs remain the labelled R4 proportional-latency assumption; CPU/wall-clock speedup is not established.
2. **Two cost estimands:** SF1 π0.5 L10-50 saves **.010857** by arm ratios, **.012304** by paired episode means. C3 calibration targets mean episode IR; C4 also exposes ratios of totals. Both are valid when labelled; they must not be exchanged mid-comparison. Changed path lengths and termination contribute to the difference.
3. **Granted is not served:** SF1 grants 927 extensions but serves **914** blocks; SF2 grants 686 two-block extensions but serves **1,334**, not 1,372. At least one block is served for 671 SF2 grants. Actual served eligibility remains ≥19.09% for SF1 and ≥14.04% for SF2. Use actual looks/calls for IR, not grants or fixed-path projections.
4. **Early valve LOOK is a real specification tension:** SF1 has 13 valve LOOKs, including **8 after the first five controls**; SF2 has 20, also **8 after five**. Enabled SF can interrupt A's original ten-control commitment before any extension is served. C1 discloses this; it follows §2's every-blind-check valve, and does not violate §4's lever-off parity gate. Freeze and describe that exact behavior explicitly. It narrows my original “preserve ten controls” proposal and means SF−UF tests stage plus valve, including early inspection. Do not claim “no extension served implies A identity.” Changing the valve now would require new profiling.
5. **No applied-control table:** all 44 `IR_active_controls` values are unavailable. Server heads and nominal five-control request counts cannot reconstruct partial terminal execution or physical stall onset. No physical gripper-transition, failure-lead-time or control-normalized-cost claim follows from the new runs. The old P3 clock is separate evidence.

**What 20 episodes establish:** the levers activate frequently, realized request-priced savings exceed the specified threshold, .30 controllers land in their allowed budget band, and recorded branch/accounting behavior is auditable. One success changes cell SR by **5 percentage points**. This profile cannot establish a 1 pp SF or 1.5 pp SW noninferiority margin, stage superiority, rare-error rates, new-task transfer, or a frontier gain. The same pairs recur across arms/models/library sizes; pooled cell counts are not independent new initial states. C4 also notes that dense-bank acquisition used these B-pool states, further limiting generalization claims.

Freeze the **28 arms above (14,000 new test episodes)** against the stipulated A/B references, without modifying the §5 tests: SF−A pooled lower bound >−1 pp and lower IR in ≥6/8 cells; the specified SF−UF stage-value test; SW−A lower bound >−1.5 pp and lower IR in 4/4; CT−CU lower bound >0 at pooled |ΔIR|≤.015. Use the registered task-stratified paired analysis. SF2 is excluded uniformly; SF+SW and SA are deferred, not described as failed methods or silently added after test results.

Reproduce my raw audit from the repository root (CPU only; writes only this directory):

```bash
taskset -c 22-23,66-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E1_kinematic_stages/analyze_profile.py
```
