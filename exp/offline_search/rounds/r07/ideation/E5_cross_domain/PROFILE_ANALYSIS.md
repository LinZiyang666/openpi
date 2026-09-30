# E5 — R7 PROFILE analysis

**Recommendation: freeze 28 candidate eval500 arms: SF1 ×8, UF1 ×8, SW ×4, CU(.30) ×4, CT(.30) ×4. Choose the common SF cap E=1. Drop SF2 from this evaluation; defer SF+SW and SA.** “Keep” means proceed to the preregistered test, not that SR noninferiority or stage superiority has been established.

Sources: [SELECTION §§4–8](../../SELECTION.md), all four coder hand-backs, [closed-loop report](../../profile_results/closed_loop/profile_report.json), the four offline profile products, accepted raw journals/server logs under `/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/runs/`, and `/tmp/r7_C2/gpu_parity.json`. My independent [profile_audit.py](profile_audit.py) and [profile_audit.json](profile_audit.json) reconcile **44 arms, 880 accepted episodes, 37,999 requests**, N/V/M, outcomes and owner request costs. Only E5 analysis files were written.

## 1. What the gates say

All IR values below use the **owner prices and realized request counts**. They are not verified active-control IR: all 44 report arms have `active_control_episodes=0`, `IR_active_controls=null`. All comparisons use the same 20 task/init pairs within a cell, with no unmatched pairs. Grant eligibility is granted vision anchors / vision anchors; repeated grant fields on blind requests are counted once.

| Variant | Observed eligibility | Saving versus paired A: difference of aggregate request IRs | Mean paired episode-IR saving range | §4 decision |
|---|---:|---:|---:|---|
| SF1, 8 cells | 19.09–48.42% | .00671–.01544 | .00795–.01695 | Pass ≥5% and ≥.005 in 8/8; keep |
| SF2, 8 cells | 14.23–38.15% | .00945–.02138 | .01146–.02364 | Pass in 8/8; drop as cap choice, not a profile-gate failure |
| UF1, 8 cells | 52.47–82.74% | .01538–.02254 | .01725–.02289 | Pass in 8/8; keep the control |
| SW, 4 cells | Actual wrist fraction of looks 22.95–40.57% | .01130–.02039 | .01249–.02271 | Pass cost and camera-use screen; keep |

The ≥.005 saving test concerns the saving levers; CU/CT use the [.27,.33] target-IR gate. CU request IR is **.29899–.30710**, CT **.29266–.31842**: 4/4 each pass. Their equal-episode mean IRs also pass: CU .29944–.31004, CT .29002–.31783. Thus the weighting convention changes numerical deltas but none of these profile pass/fail decisions.

**Parity:** C1/C4 report disabled SF/UF exact identity on **10,569 decisions / 52,449 recorded active controls / 240 episodes**, zero failures; enabled structural/stage rejection has 6,612 checked blind/LOOK comparisons. C3 reports 26,992 tilt-off/reference decision pairs with zero differences. C2's real GPU artifact has **12/12 PASS**, exact wrist keys, completed policy inputs and K10 outputs (`policy_max_abs=0`); this supersedes its earlier hand-back's pending-GPU note. My raw-log audit finds zero within-episode stage-1 counter-increment mismatches, zero camera-counter/cost mismatches, and zero logged call-coin, CT weight or nominal-p mismatches. These checks do not establish physical action execution or successful transfer to another robot.

## 2. M1: stage-limited reuse → SF1/SF2 and UF1

Table entries are **successes out of 20 @ owner request IR**. SR changes are descriptive. Eligibility columns are SF1 / SF2 / UF1 percentages.

| Cell | A | SF1 | SF2 | UF1 | Eligibility % |
|---|---|---|---|---|---|
| π0.5 L10-50 | 15 @ .07629 | 14 @ .06544 | 11 @ .06140 | 17 @ .05513 | 35.3 / 26.9 / 78.8 |
| π0.5 L10-500 | 20 @ .07668 | 20 @ .06788 | 18 @ .06422 | 13 @ .05599 | 26.6 / 19.0 / 73.9 |
| π0.5 Spatial-50 | 15 @ .07728 | 13 @ .06724 | 16 @ .06181 | 14 @ .06085 | 30.4 / 25.8 / 52.5 |
| π0.5 Spatial-500 | 18 @ .07781 | 18 @ .06238 | 18 @ .05643 | 18 @ .05527 | 48.4 / 38.2 / 81.0 |
| GR00T L10-50 | 14 @ .07427 | 14 @ .06487 | 13 @ .06038 | 14 @ .05502 | 31.0 / 23.6 / 71.2 |
| GR00T L10-500 | 18 @ .07467 | 17 @ .06796 | 16 @ .06522 | 17 @ .05292 | 19.1 / 14.2 / 82.7 |
| GR00T Spatial-50 | 17 @ .07519 | 16 @ .06416 | 15 @ .05865 | 15 @ .05981 | 36.4 / 28.5 / 53.0 |
| GR00T Spatial-500 | 18 @ .07510 | 18 @ .06536 | 18 @ .06159 | 18 @ .05410 | 32.7 / 22.4 / 77.0 |

**Keep SF1; cap E=1 for every cell.** It serves **914 extension requests** from 927 granted anchors: 495 π0.5 successor heads and 419 GR00T native-tail blocks. There are zero policy calls. This addresses my empty-support and vanished-model-cost-saving kill criteria: extension exposure is real, and all eight request-IR savings exceed .005. Across cells, mean request IR falls **.07591→.06566**. The implementation does not include my proposed action-dispersion rank gate, and uses C1's row-p95 displacement radius rather than my episode-envelope sketch; these results support the selected structural/stage/valve variant only.

SF1 has **130/160 successes versus A's 135/160**: three paired gains and eight losses, descriptive −3.125 pp. It therefore **has not passed my 1 pp SR noninferiority requirement** or SELECTION §5 acceptance. Twenty starts per cell cannot turn this into a reliable NI conclusion. Keep it to test the frozen hypothesis, not because the profile proves laziness safe.

**Drop SF2 as the cap choice.** It has 125/160 successes, versus SF1's 130: eight paired gains and thirteen losses. Its additional saving versus SF1 is only **.00274–.00595 per cell**, mean **.00445**, and it requires successor execution beyond GR00T's native chunk. π0.5 L10-50 is 11/20 versus 14/20 for SF1 and 15/20 for A; Spatial-50 reverses that ordering (16/20 versus 13/20), so do not choose different caps using these outcomes. The common cap 1 is the simpler cross-policy experiment; cap 2's five fewer pooled successes are a descriptive caution, not a significant harm claim.

**Keep UF1 as an essential control.** It has 126/160 successes, eight gains/seventeen losses versus A, mean cell IR .05614, and 1,902 extension requests. π0.5 L10-500 is the strongest descriptive separation: SF1 20/20 versus UF1 13/20; π0.5 L10-50 goes the other way, 14/20 versus 17/20. Requested extension heads have **0/914 gripper-sign-flip-containing blocks for SF1 versus 154/1,902 (8.10%) for UF1**, counting flips within a head or from the preceding requested head. This is evidence that the signal changes which command sequences get extended, not a measured grasp/slip label or verified execution of every terminal control.

My stronger **matched-IR** stage-value criterion remains untested: SF1 observes more often than UF1, and SF/UF jointly ablate stage and valve. Their comparison cannot isolate the valve or show that stage placement beats a uniformly randomized extension scheduler at equal observation cost. Follow SELECTION's frozen acceptance test, but keep that scope explicit.

The offline `follow_all` result already found SF1 eligibility **21.72–48.15%** on the earlier ten B-val recordings per cell. Real eligibility **19.09–48.42%** and real savings above support screening are encouraging, not evidence that its fixed-stream modeled IR predicts every renewed trajectory. Offline `clock_all` reports **43 failures: 43 absolute-deviation crossings, 25 valve alerts, 26 confirmed stalls**. Its timestamps lead to terminal failure, not known failure onset. With no new per-control telemetry, my transition-consistency/veto-before-physical-failure criterion remains unresolved.

## 3. M2: neighbourhood-consensus routing, and the different built CT

**Defer my original M2.** Direct neighbour action dispersion over the stage and a latched stage score were not implemented. CT instead uses library event-neighbourhood occupancy and first entry above the state-deviation p75. It is an adjacent stage/event hypothesis, not a validation or refutation of neighbourhood-consensus routing or a revival of R6's action LOEO residual.

My causal-evidence gate is still unmet: offline `value_all` has **244** natural-dose first-entry comparisons, **236** with intervals, **235/236 including zero**. The one positive `unknown`-stage result has 49 entries, four CALLs, CALL ESS=4 and effect .01317 [ .00018, .03098 ]; it cannot justify a portable difficulty rule. Do not build a new consensus router or fit task-specific thresholds from these profiles.

**Keep CU and CT for the already-preregistered uniform-versus-tilted test**, not for deployment endorsement. Both have **68/80 successes**, CT versus CU has **7 paired gains / 7 losses**. CU makes 907 calls, CT 908; mandatory stall calls fall **218→187**, so total call count alone hides changed placement. Logged CT has 81 fresh deviation entries; weights and clipped nominal probabilities agree with C3's formula on all profiled fresh anchors.

| Sparse cell | CU successes @ IR | CT successes @ IR | CT−CU request IR | CT−CU mean paired episode IR |
|---|---|---:|---:|---:|
| π0.5 L10-50 | 18 @ .29899 | 19 @ .31842 | +.01943 | +.01561 |
| π0.5 Spatial-50 | 18 @ .30180 | 16 @ .29266 | −.00915 | −.01189 |
| GR00T L10-50 | 15 @ .30710 | 15 @ .29661 | −.01049 | −.01373 |
| GR00T Spatial-50 | 17 @ .30463 | 18 @ .31619 | +.01155 | +.01234 |

Each arm passes [.27,.33], but **π0.5 L10-50 does not satisfy the separate |ΔIR|≤.015 matched-cost condition**, under either aggregation. Its one-success advantage cannot be described as better call placement at equal cost. Do not retune to these outcomes; preserve the frozen non-test calibration for evaluation, report both IR aggregations, and withhold a matched-cost claim wherever §5's condition fails. A target-rho calibration is an expectation under recorded occupancy, not a hard online spending guarantee.

## 4. SW, composition and the frozen arm list

SW's actual wrist looks / all looks are **217/624, 123/536, 75/274, 99/244** for π0.5 L10-50, L10-500, Spatial-50, Spatial-500. Successes and IR are **16 @ .05950, 19 @ .06538, 15 @ .06368, 17 @ .05742**. That is **67/80 versus paired A's 68/80**, three gains/four losses. Keep all four cells: the actual camera use, real GPU parity and cost gates pass, while descriptive SR does not select a cell-specific wrist policy. The offline L10-500 wrist-minus-full shadow-error increase **+.02377 [.00955,.04335]** is a caution, not a parity failure or causal SR estimate.

**Frozen recommendation: SF1 ×8 + UF1 ×8 + SW ×4 + CU ×4 + CT ×4 = 28 candidate arms / 14,000 eval500 episodes**, retaining SELECTION's reference and paired-statistics plan. This preserves the 20 Freeze-1 arms, adds SW and CT, and uses one cap everywhere. No R7 test-pair outcomes were read for this recommendation.

**Defer SF+SW ×4.** It is implemented but absent from the 44-arm closed-loop profile. Independent SW and SF passes do not measure composed eligibility, state-valve behavior or cost: wrist retrieval changes the very neighbourhood used to grant follow. It should receive its own non-test composed profile and parity/cost checks before a later freeze, not be inferred safe or effective by adding marginal savings.

**Do not build SA now.** We have no call-tilt advantage here, and SF's NI remains unproven. SA needs new cadence/camera budget calibration; reusing CU/CT's λ would not target .30 after follow/wrist changes. SW's 514 wrist looks all had zero MISSes/completions, so these profile arms do not exercise the mixed wrist-origin CALL path end-to-end. C2's GPU parity covers completed inputs, but it is not a mixed serving/cost smoke. If future evaluation justifies composition, start with **CU+SF1**, then consider wrist; do not add CT and wrist together without separate evidence.

## 5. Bugs, contract discrepancies and accounting traps

- **Early valve abort changes the native commitment.** SF1 has **13** valve LOOKs, **8 at age 1 (five controls)**: four π0.5 L10-50, one π0.5 L10-500, three GR00T L10-50. SF2 has 20 valve LOOKs, eight at age 1. `FollowExtension.check` runs before the ordinary first blind tail whenever an extension plan exists. This is the documented C1 specification conflict, **not a disabled-lever parity failure**. Keep the profiled SF1 implementation, but change the frozen method description to explicitly permit these early reviews; it is not “always unchanged ten-control commitment.” If preserving ten controls is mandatory, that is a different variant requiring another profile, not a silent fix.
- **Default summary IR uses a different cost table.** Example SW π0.5 L10-50: `summary.cost_ledger.ir_per_request=.0555715`, owner report **.0595023**, from 407 full looks and 217 wrist looks over 1,241 requests. N/V/M reconcile; this is an explicit eager-table/owner-table mismatch, not missing camera work. Use the C4 owner result consistently for gates/frontiers. Wrist/completion prices remain the R4 proportional-latency assumption, not new measured latency savings.
- **Aggregation differs.** SF1 π0.5 L10-50 saves **.0108574** by the difference of aggregate request IRs, versus **.0123037** by mean paired episode differences. C3 calibration targets the equal-episode statistic; long failures receive more weight in aggregate request IR. State the denominator rather than treating these as inconsistent results.
- **Counter and grant fields repeat.** `stage1_calls` is cumulative; raw summation would vastly overcount work. Its within-episode increments match `vision` on all 37,999 requests. Count `os_sf_granted` only at vision anchors: SF2 has **686 grants / 1,372 granted blocks / 1,334 served extension requests**, three distinct quantities. A granted block is not a proven full block of applied controls.
- **No physical-control/reset attestation in these stock-client profiles.** The selection manifests and task/init identities match, but absence of P3 client traces means no independently verified reset-state hash, terminal partial-control count or failure-onset time. `summary.controls` is explicitly nominal requests×5; do not relabel it actual controls or fill C4's null control-IR fields with it.
- **No end-to-end latency claim.** SF1 blind-query medians are **.448–.621 ms**; SW look-query medians are **6.47–12.53 ms**. These are method wall timings under the profile execution configuration, not controlled CPU/GPU speed comparisons. IR excludes CPU, transport and batching changes; SW uses B=1. My complete-cost/portability check therefore remains incomplete even though owner IR passes. No new robot, control rate or three-camera path has been validated by these LIBERO runs.

## 6. What twenty episodes establish, and reproduction

One outcome changes a cell's SR by **5 pp**, and there are only two starts per task. These profiles establish exercised code paths, observed eligibility, nominal-request compute use and gross accounting consistency. They do **not** establish ≤1 pp/1.5 pp NI, a causal stage-specific rescue benefit, real-arm safety, independent dense-library generalization, or a moved test frontier. In particular, SF1's −3.125 pp pooled descriptive change is a reason to test the NI hypothesis seriously, not a pass; failure to prove NI in this screen is not proof of harm. The same B-val scenes can underlie the dense banks, as C4's hand-back records.

Reproduce the raw audit from repository root; it reads existing artifacts and writes only E5's `profile_audit.json`:

```bash
taskset -c 30-31,74-75 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E5_cross_domain/profile_audit.py
```
