# R8 E3: Failure forensics on the real data (DATA_ANALYSIS.md)

Explorer E3 (Opus), R8 callback. Run root `/home/weiland/trace_runs/os_closed_loop/r08_main`. Outputs are in `/tmp/r8cb_E3_failure_forensics/`. Analysis scripts are in `exp/offline_search/rounds/r08/ideation/E3_failure_forensics/` (`r8_labels.py`, `r8_grasps.py`, `r8_oracle.py`). Everything ran on CPU 0-3/44-47 with at most 3 Python processes.

**Holdout.** I chose no threshold or segmentation. Every table is descriptive over all 500 pairs per arm. The only fitted quantity is the per-task near-radius used by the forensic labels. It was frozen once from **P10 successful episodes on discovery inits 0-29** (the tool enforces this) and then reused for every arm via `--adapter-config`.

**Short answer to the main question.** Perfectly placed grasp-window calls (O5a) recover only 35-40% of the gap between A and pure policy on LIBERO-10, at about 1.9× owner IR. The tight version at about 1.2× owner IR recovers 14-17% on L10 and 5-14% on Spatial. At the same call share, perfect grasp-window placement beats random placement by about +4.5 pp on L10 and about 0 on Spatial. **Segmentation of the grasp window is not the bottleneck.** The cache's excess failures are spread over the approach before the window, the grasp itself, and the place/release/fixture stage.

## 1. What I ran

| Tool (CLI, all with `--run-root r08_main --procs 2`) | Arms | Coverage |
|---|---|---|
| `physical.forensics` (frozen calibration from the P10 run) | 46 arms: P10 ×4, A ×8, CU ×8, CT ×8, IP ×4, O5a ×2, O5b ×4; levers pi05 L10-50 FL/SF1/SW/W10/W5/A5 and GR00T L10-50 FL/SF1 | 23,000 episodes; 0 unavailable rows |
| `physical.grasp_audit --reference-arms <P10>` | pi05 L10 {P10, A50, O5a, O5b, CU50}; GR00T L10 {same}; Spatial {P10, A50, O5b} × 2 models | 16 arms, about 19k close-onsets |
| `physical.drop_audit` | pi05 sp50 A, groot sp50 A, groot sp50 IP, groot sp500 CT | 2,000 episodes, 145 carry losses |
| `physical.paired_diverge --reference-arm r8_pi05_l10_50_A` | pi05 L10: P10 and O5a vs A50 | 1,000 pairs |
| `physical.selfcheck` | pi05 L10 P10, groot sp50 O5b | 1,000 episodes × 6 checks |
| `decision.trigger_vs_onset --onsets <forensics CSV>` | CU50 ×4 (`os_c_stall_call`, `os_c_extra_look`); CT50 L10 ×2 (adds `os_c3_dev_entry`) | 6 arms |
| SR and owner IR from `runs/<arm>/summary.json` `cost_ledger` (v, m) | all 67 finished arms | — |

Example command (prefix `taskset -c 0-3,44-47 env OMP_NUM_THREADS=1 … PYTHONPATH=.:src .venv/bin/python`):
```
-m exp.offline_search.debug.tools.physical.forensics --run-root …/r08_main --arms r8_pi05_l10_P10 r8_groot_l10_P10 r8_pi05_spatial_P10 r8_groot_spatial_P10 --out /tmp/r8cb_E3_failure_forensics/forensics_P10
-m exp.offline_search.debug.tools.physical.forensics --run-root …/r08_main --arms <group> --adapter-config /tmp/r8cb_E3_failure_forensics/forensics_P10/adapter_calibration.json --out …/forensics_<group>
python exp/offline_search/rounds/r08/ideation/E3_failure_forensics/{r8_labels,r8_grasps,r8_oracle}.py
```

**Not run (time):**
- Forensics on 21 finished arms: the FL/SF1 arms outside L10-50, SW/W10/W5 on 500 and Spatial, and SHIFT.
- `episode_card`, `arm_rollup`, `segmentation_bench` (E1's), `blind_drift` and `twin_divergence`.
- None of my tools needs the deferred augmentation, so no result here is aug-partial.

## 2. Do the tools and data answer my questions?

| Tool / field | Verdict | Concrete gap |
|---|---|---|
| **selfcheck** | **Works** | 0 failed checks out of 5,000 on 1,000 episodes: geom naming, post-settle reference, seed, contiguity, resting contacts. All three P3 v2 defects are fixed; `before` is no longer stored. |
| **forensics (labels + onsets)** | **Works**, with 4 gaps | (a) `unknown_release` when the destination is a fixture region. In A50+P10 this is microwave heating region 43, stove cook region 14, and one each for cabinet bottom, caddy and the table-plate region. It is 40/195 GR00T L10-50 A failures, so drop vs misplace is undecidable on L10 tasks 8 and 9. (b) `grasp_miss` onset = window entry (first near), not the first failed closure, so onsets are early. (c) Without `--adapter-config`, `calibrate()` pools successes of whichever arms are passed, so labels change with the arm set; it must be frozen as I did. (d) Every row embeds `adapter_settings` with all training keys (89 MB for 2,000 rows). Labels are still `truth_validated=false`; the human audit has not been done. |
| **truth stages** (`decisions_truth`, `controls_truth`) | **Works** | Used indirectly through grasp/oracle joins. |
| **grasp_audit** | **Works** | Object-frame pose, aperture, 40-control lift, issuing source, same task/init P10 reference. No gap found. |
| **drop_audit** | **Works** | Named contacts at loss are available (table_collision etc.). |
| **paired_diverge** | **Partial** | Outcome flips and transitions work. The envelope exit, object-divergence onset and `stage_at_object_onset` are unavailable in 1,000/1,000 pairs because no frozen `--envelope` exists. R8 has no replicate arm to calibrate natural variation. `first_event_difference` and `first_vision_difference` are null in all pairs; `first_action/state_difference` is trivially control 10. I computed the eef-gap divergence from `divergence_curves.csv` myself. |
| **trigger_vs_onset** | **Works** | Its value is capped by the heuristic onsets (gap b above). Its "false alert" is episode-level (any alert in a success). |
| **Oracle arms O5a/O5b** | **Delivered as specified** | Per-decision distance, window and object are logged. But the spec (mine) measured the tight 5 cm radius to the **object centre** and checked the window only at fresh anchors; both limit what the arms can answer (section 3). |
| sim_replay (T7) | **Missing** | Not built. It is not needed now that every control is captured. |

## 3. Findings

### 3.1 Oracle arms: did perfect grasp-window calls recover pure-policy SR?

**Notes for the table:**
- dSR is paired against A on the same task/init and env_seed 7, with a 95% task/init cluster bootstrap.
- Gap = P10 − A.
- "vs random @ same m" = oracle SR minus A-to-IP linear interpolation at the oracle's call share m. IP is an independent p=.25 coin at m ≈ .125-.13. Random placement is concave, so this is an *upper* bound on the oracle's advantage.

| Cell | A SR @ IR | O5a SR @ IR (m) | O5b SR @ IR (m) | IP SR @ IR | CU / CT SR @ IR ≈ .30 | P10 SR |
|---|---|---|---|---|---|---|
| π0.5 L10-50 | .718 @ .076 | **.794** @ .141 (.076); dSR +.076 [+.034, +.114]; **40% of gap**; +.047 vs random @ same m | **.744** @ .092 (.018); +.026 [−.008, +.058]; 14%; +.019 | .768 @ .187 | .854 / .876 (72% / 83%) | .908 |
| GR00T L10-50 | .610 @ .074 | **.712** @ .147 (.085); +.102 [+.064, +.142]; **35%**; +.045 | **.658** @ .089 (.017); +.048 [+.022, +.076]; 17%; +.037 | .694 @ .181 | .796 / .810 (65% / 69%) | .898 |
| π0.5 Spatial-50 | .824 @ .078 | — | **.832** @ .102 (.029); +.008 [−.016, +.032]; 5%; −.009 | .898 @ .183 | .940 / .948 | .988 |
| GR00T Spatial-50 | .868 @ .075 | — | **.878** @ .098 (.026); +.010 [−.006, +.028]; 14%; +.001 | .912 @ .185 | .930 / .936 | .940 |

**Why the oracle falls short** (`r8_oracle.py`, `r8_grasps.py`):

**Calls do arrive (O5a).**
- O5a fires in 88.8% (π0.5) and 86.4% (GR00T) of episodes, at 4.66 and 5.64 calls per episode. Median call distance to the object centre is 6.6 and 6.2 cm.
- In A's grasp-miss episodes, O5a placed at least one call in 67/69 (π0.5) and 100/106 (GR00T).
- Yet only 37/69 (54%) and 40/106 (38%) of those succeed, against 64/69 (93%) and 91/106 (86%) under P10.

**Many closures still come from the cache (O5a).**
- In O5a, 637/1,335 (48%, π0.5) and 760/1,505 (50%, GR00T) of grasp closures are still issued by the cache. The window is checked only at fresh anchors (every 10 controls), so fast entries are missed.
- When the policy does issue the closure, lift rates are .846 (policy) and .755 (policy tail) on π0.5, and .691 / .647 on GR00T.
- For comparison, cache closures in A lift at .640 / .726 (π0.5) and .468 / .451 (GR00T); P10 lifts at .820 and .745.

**O5b's tight window never opens where it is needed.**
- In A's Spatial grasp-miss episodes, O5b made zero calls in 51/51 (π0.5) and 15/16 (GR00T). The failing cache never brings the eef within 5 cm of the bowl *centre* at an anchor. Call distances p10/p90 are 3.5/4.9 cm, i.e. the bowl-rim geometry.
- On L10, O5b calls reach 39/69 and 66/106 of A's grasp-miss episodes and rescue 11 and 18.
- O5b's policy-issued closures lift .85-.94, but they are only 13-17% of all closures.

**The residual after O5a is mostly outside the grasp window.**
- π0.5 L10 O5a, 103 failures: grasp_miss 37, undone 24, misplace 21, fixture_not_done 12. That is 55% on the place/fixture side.
- GR00T L10 O5a, 144 failures: grasp_miss 62, unknown_release 35 (microwave task), misplace 21.

**Conclusion.** For stage-level allocation, the grasp window is a real hard stage: per-closure lift rate is .45-.73 for the cache vs .75-.94 for the policy. But "call the policy when inside the window" recovers at most 35-40% of the gap at about 2× owner IR. Uniform random calls at IR ≈ .30 recover 65-94%. The hard part starts **before** the window (the approach the cache sets up) and continues **after** it (place, release and fixtures).

### 3.2 Failure modes by variant (50-demo cells plus P10; failures only)

| Suite / variant | Failures | grasp_miss | drop | place side (misplace + unknown_release + held + undone) | other (fixture / never_reached) |
|---|---|---|---|---|---|
| L10 A | 336 | 175 (52%) | 11 (3%) | 137 (41%) | 13 |
| L10 CU / CT | 175 / 157 | 42% / 41% | 7% / 6% | 45% / 46% | 10 / 11 |
| L10 IP | 269 | 47% | 3% | 44% | 15 |
| L10 O5a / O5b | 247 / 299 | 40% / 50% | 4% / 3% | 48% / 42% | 19 / 15 |
| L10 P10 | 97 | 35 (36%) | 0 | 51 (53%) | 11 |
| Spatial A | 154 | 67 (44%) | 19 (12%) | 56 (36%) | 12 |
| Spatial CT / CU / IP / O5b | 58 / 65 / 95 / 145 | 28% / 31% / 40% / 50% | 24% / 15% / 23% / 12% | 41% / 45% / 29% / 30% | — |
| Spatial P10 | 36 | 31% | 19% | 47% | 1 |

**What P10 rescues.** Pure policy fixes almost every cache grasp miss on the same task/init/seed: 64/69 and 91/106 on L10, 50/51 and 16/16 on Spatial. The cache-specific excess is mostly grasp_miss, then undone, misplace and fixture.

**Two L10 failures that are task-specific:**
- π0.5 task 9 fixture_not_done (microwave not closed): A 10, O5a 11, P10 2.
- π0.5 task 8 undone (moka pots): A 7, **O5a 16**, CU 11, P10 4.

**Where drops matter.** Drops are rare on L10 (3%) but matter on Spatial. The pilot's large L10 drop share (28%) did not replicate at n=500. That pilot covered only inits 0-1.

### 3.3 Grasp alignment

- **Object-frame pose error predicts a failed lift.** Error is measured against P10's grasp on the same task/init/object. It predicts a failed lift better than the centre offset:

  | Cell | AUROC, pose-error reference | AUROC, centre offset |
  |---|---|---|
  | π0.5 L10 A | .754 | .707 |
  | GR00T L10 A | .672 | .611 |
  | GR00T L10 CU | .814 | .636 |
  | π0.5 Spatial A | .716 | .408 |

- **Typical error sizes.** Median pose error is 2.7-4.2 cm for failed closures and 1.5-1.9 cm for successful ones.
- **What this means.** A cache-side "pre-grasp alignment" signal exists, but it needs the object frame and a reference grasp, which the robot does not have online.
- **Attempts per episode.** Closures per episode with any attempt are 2.78 (π0.5 A) vs 2.33 (P10), and 4.21 (GR00T A) vs 2.43. The cache retries more often.

### 3.4 Drops (Spatial)

- Of 145 carry losses away from the destination, 144 are slips with the close command still on, and 1 is a premature open command.
- Finger width at loss has a median of 0.48 cm (pads nearly closed, an edge or rim pinch).
- 101/145 of these episodes still succeed after a regrasp.
- Sources of the losses: cache 82, cache tail 50, policy 13.
- The blended-gripper "premature open" hypothesis from the pilot is not supported.

### 3.5 Paired divergence and outcome churn (π0.5 L10-50, A reference)

- **A vs P10.** The eef gap exceeds 3 cm in 499/500 pairs, at a median control of 52 (IQR 29-83). This replicates the pilot's 54. Outcome flips occur in 27.3% of pairs [24.4, 30.4]: 131 rescued, 36 lost.
- **O5a vs A.** The trajectories stay together until the first oracle call: the gap first exceeds 3 cm at a median control of 128 (385/500 pairs). 72 rescued, 34 lost.
- **Churn is large.** Even arms that lose SR overall rescue 37-49 A failures each. Without a replicate or placebo arm in the 50-demo cells, per-episode rescue counts cannot be separated from chaotic divergence; only paired SR differences are interpretable.

### 3.6 Triggers vs onset (CU/CT, 50-demo cells)

| Trigger | Fires in the 20 controls before onset | Also fires in successful episodes |
|---|---|---|
| Stall call | 21-32% of failures with an onset | 70-86% (L10), 26-33% (Spatial) |
| Deviation-entry (CT) | 16-19% | 72-83% |
| Extra look | 10-21% | — |

Measured against these onsets, none of the existing call triggers is a failure predictor. Caveat: the grasp_miss onset is window entry (section 2, forensics gap b).

### 3.7 Onset and waste

- On L10, failures spend a median 65-78% of their controls after the decisive event: about 350-410 of 520 controls.
- On Spatial, the figure is 57-76%.
- Every failure is a step-cap timeout.

### 3.8 Lazy levers (π0.5 L10-50 and GR00T L10-50, paired vs A)

| Lever | dSR vs A | grasp_miss count (A = 69) |
|---|---|---|
| W5 | −.128 [−.174, −.086] | 99 |
| FL | −.084 [−.128, −.044] | 105 |
| W10 | −.080 [−.124, −.036] | 93 |
| A5 (A every 5 controls) | −.076 [−.118, −.034] | 98 |
| SF1 | −.036 [−.074, .000] | 83 |
| SW | −.018 [−.056, +.020] | 75 |

- W5 also raises undone from 21 to 30.
- On GR00T L10-50, FL and SF1 are each −.036.
- **Looking less or with fewer cameras hurts mainly at the grasp.** A5 (more frequent cache re-retrieval) also hurts, so the loss is not simply blind time.
- **Implication for allocation.** Any look-less lever must be gated off from the approach through the grasp.

## 4. Bugs and data-quality problems

1. **Fixture-region destinations are unresolved** (forensics adapter): `microwave_1_heating_region`, `flat_stove_1_cook_region`, `white_cabinet_1_bottom_region`, `desk_caddy_1_back_contain_region`, `living_room_table_plate_right_region`. The result is `unknown_release`, e.g. 40/195 GR00T L10-50 A failures. Fix: put region site positions (`site_xpos`) in the catalog.
2. **Label calibration depends on the arm set** unless `--adapter-config` is passed (`adapters.calibrate` pools successes of the passed arms). Any multi-group forensic run without a frozen config yields labels that cannot be compared across groups. Use `/tmp/r8cb_E3_failure_forensics/forensics_P10/adapter_calibration.json`, or freeze it in the catalog.
3. **`grasp_miss` onset is window entry, not the first failed closure** (`forensics.py`, `onset_control=first(near)`). This biases `trigger_vs_onset` hit rates and lead times.
4. **`paired_diverge` envelope fields are 100% unavailable.** There is no `--envelope` and no replicate arm to calibrate one. `first_event_difference` and `first_vision_difference` are null in 1,000/1,000 rows.
5. **`episodes_forensics.csv` repeats the full `adapter_settings`, including training-key lists, on every row** (89 MB per 2,000 episodes).
6. **Oracle design flaw (my spec, not an implementation bug).** The O5b radius is measured to the object centre and the window is checked only at fresh anchors (`debug/client/adapter.py:oracle`, `methods.py:OracleGraspCalls`). On Spatial, the tight window never opens in the failing episodes, and in O5a half of all closures are issued by the cache.
7. **Arms not at owner IR.** CU/CT run at ρ .30 (50-demo) and .18 (500-demo), giving owner IR .17-.31. No call arm sits at owner IR ≈ .076; O5b (.089-.102) is the closest.

## 5. What R9 should do

1. **Do not fund sharper grasp-window segmentation alone.** Its ceiling at about owner IR is +2-5 pp on L10 and about 0 on Spatial.
2. **Run oracle v2 diagnostics (privileged, labelled) that cover the whole hard span:**
   - (a) Check the window every decision, not only at anchors.
   - (b) Measure the window as surface or object-frame distance, not centre distance.
   - (c) Keep the policy through closure until lift (or 20 controls).
   - (d) Add a place/release/fixture oracle window around the destination.
   - Run grasp-only, place-only and both, on the L10-50 cells of both models. This tells whether *complete* stage coverage at IR .10-.15 can reach P10. If even "both" stays far below P10, stage-level allocation is not the route and the cache itself must improve.
3. **Add matched-call-share random arms** (IP at p ≈ .15 and p ≈ .04), so oracle vs random is measured, not interpolated.
4. **Add one replicate or placebo arm per 50-demo L10 cell** (same seed, A rerun or SHIFT), to calibrate the `paired_diverge` envelope and the churn floor.
5. **Fix tool gaps 1-5 in section 4.** Run the E3 human audit (50 failures + 20 successes via `episode_card`) before any label or onset becomes ground truth. Redefine the `grasp_miss` onset as the first failed closure.
6. **Target the two cache-specific stages directly** (a better cache, not only better calls):
   - Pre-grasp alignment: object-frame pose error AUROC .67-.81.
   - Place/fixture execution: π0.5 task 9 microwave close and task 8 moka-pot undo.
   - Online, only robot-side proxies of these are available. E1/E2 should score their candidate segmentations against `decisions_truth` stages on discovery inits only.
