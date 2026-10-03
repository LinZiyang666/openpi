# E4 data analysis: looking less and looking half (R8 callback)

**Headline.** Every look-saving lever loses success on LIBERO-10 except the stage-gated wrist camera. Looking more often, every 5 controls instead of 10, also loses success, on both models. Almost all the extra failures are missed grasps. The tools answer the questions about arm-level effects and about how trajectories drift apart. They do not answer which stage an extension hurts in: the dedicated tool collapses every stage to "unknown", and one 500-episode lottery arm is too small to show it anyway.

Read-only analysis, CPUs 26-29/70-73, at most 3 Python processes at a time. Scripts are in `exp/offline_search/rounds/r08/ideation/E4_lazy_levers/`: `fl_lottery.py` and `cadence_mech.py` are new; `follow_vs_look.py` and `twin_divergence.py` are from the proposal. Outputs are in `/tmp/r8cb_E4_lazy_levers/`.

**Holdout rule I followed.**
- Analyses split by stage, the lottery analysis and the mechanism metrics use only inits 0–29.
- Success rate per arm, paired churn, forensics labels and twin counts use all 500 pairs. They are descriptive and were not used to choose any threshold or segmentation.

**Names used below.**

| Name | Meaning |
|---|---|
| P10 | pure policy |
| A | cache, looks every 10 controls |
| A5 | same as A but looks every 5 controls |
| SF1 | one extra blind block, only when the stage gate and state valve allow it |
| FL | follow lottery: 0, 1 or 2 extra blind blocks drawn at random, gate computed but not enforced |
| W10 / W5 | wrist camera only, every 10 / every 5 controls |
| SW | wrist camera only in easy stages |
| SHIFT | placebo: A with all looks shifted by 5 controls |

## 1. What I ran

**Arms (all 500 accepted pairs).**
- π0.5, four cells (L10-50, L10-500, Sp-50, Sp-500): A, SF1, FL, SW, W10, W5, P10. A5 on L10-50 only.
- GR00T, four cells: A, SF1, FL. SHIFT on L10-50 and L10-500. A5 on L10-50 (finished during this analysis).
- References: R6 A replicates and R7 SF1 (journals and decision logs).

**Augmentation coverage.** At analysis time only GR00T arms had `AUG_DONE` (15 arms). No π0.5 arm was augmented, so `camera_shadow` could not run. `follow_vs_look` ran on GR00T A and FL, L10-50 and L10-500.

**Commands.** All use the brief's CPU and environment prefix, `--run-root .../r08_main`.
- `decision.churn`, 9 cell runs. `--reference A`; `--placebos SHIFT` on GR00T L10.
- `decision.exposure_hazard` on `r8_groot_l10_50_FL`.
- `decision.follow_vs_look` on GR00T L10 A/FL at 50/500, run twice: once with defaults and once with `--metadata meta_groot_l10_current.json`. That file sets `gripper_dim=6`, `gripper_threshold=0`, and `action_scale` = library successful-row action std.
- `physical.twin_divergence` on GR00T L10-50 A/SF1/FL/SHIFT (`--reference-arm A --library-root groot_l10/current`).
- `physical.forensics` on π0.5 L10-50 A, A5, W10, W5, SW, SF1, FL and π0.5 L10 P10.
- `physical.grasp_audit` on A, A5, FL, SF1, W10 and P10 (`--reference-arms P10`).
- `physical.blind_drift` on GR00T L10-50 A/SF1/FL. **It did not finish.** I stopped my own process (by PID) after 57 CPU-minutes with no output.
- My scripts:
  - `fl_lottery.py`: all FL arms, all lottery anchors.
  - `cadence_mech.py`: 15 arms, measuring demo switching, seam jumps and gripper toggles.
  - Two inline checks: replicate churn of R8 A against R6 A and of R8 SF1 against R7 SF1; and whether debug mode changed GR00T decisions (identical rows/weights, R8 A against R6 A).

## 2. Do the tools and data answer my questions?

| Requested tool | Verdict | Concrete gap |
|---|---|---|
| churn (with SHIFT placebo) | **Works** | Reads one run root, so π0.5 has no placebo and no replicate control inside the tool. I computed R8 A vs R6 A churn by hand. |
| exposure_hazard (FL) | **Partial; my main question is unanswered** | (a) Every stage stratum collapses to `unknown` ("assignment-time stage certification absent"). Yet the stage is computable from the anchor's own neighbours, which are fixed before the coin, and FL even logs `diag.stage_ok_by_e`. (b) Only the first-entry estimand is used: one assignment per episode, 149–189 per arm. Success E1−E0 = −0.6 pp [−20.4, +22.4]. (c) robot-demo drift, object drift and stall entry are all unavailable. |
| My all-anchor lottery replacement | **Partial** | Randomized local contrasts are resolvable (§3.6). Success effects by stage are not: summed-anchor HT intervals are ±0.2–0.5 at 300 discovery episodes. One 500-episode lottery arm cannot place extension harm by stage. |
| twin_divergence (GR00T) | **Works** | 1,500/1,500 pairs admitted with a verified bit-identical prefix. Without `--envelope` the physical divergence onset is unavailable; I report raw object-gap curves instead. `--limit` samples episodes per arm, so it yields 0 matched pairs. Runtime was ~30 min and 9.5 GB RSS for 4 arms. |
| follow_vs_look (closed loop) | **Works on augmented GR00T arms** | With default metadata, gripper flips (n = 0) and action-σ units are missing, because manifests lack `gripper_threshold` and `action_scale`. It needs `--metadata`. π0.5 is pending augmentation. |
| camera_shadow | **Missing** (π0.5 augmentation pending) | I explained the wrist arms with forensics, grasp_audit and cadence metrics instead. Wrist-vs-full retrieval at the same observation is still unmeasured. |
| blind_drift | **Not delivered** | Robot-state only, because there is no library backfill. >57 min for 1,500 GR00T episodes without finishing. FL's live valve statistic (`shadow_absolute`) served as the drift proxy. |
| forensics, grasp_audit | **Work** | Labels are heuristic (`truth_validated=false`). Grasp contrasts are post-treatment selections and are reported as descriptive only. |

## 3. Findings

### 3.1 Camera × cadence on the owner-cost axis

Success rate @ owner IR (π0.5: full look .152, wrist look .0646, call .848; GR00T .148/.852). 500 pairs per arm.

| π0.5 cell | P10 | A (full, 10) | A5 (full, 5) | SW | SF1 | FL | W5 (wrist, 5) | W10 (wrist, 10) |
|---|---|---|---|---|---|---|---|---|
| L10-50 | .908@.504 | .718@.076 | .642@.152 | .700@.063 | .682@.065 | .634@.058 | .590@.066 | .638@.034 |
| L10-500 | .908@.504 | .848@.077 | – | .846@.067 | .830@.069 | .780@.054 | .726@.066 | .786@.034 |
| Sp-50 | .988@.511 | .824@.078 | – | .826@.064 | .794@.066 | .792@.061 | .820@.068 | .800@.036 |
| Sp-500 | .988@.511 | .974@.078 | – | .976@.058 | .962@.061 | .980@.056 | .956@.069 | .976@.037 |

Paired difference versus A, in pp, with 95 % task/init bootstrap intervals from the churn tool:

| π0.5 arm | L10-50 | L10-500 | Sp-50 | Sp-500 |
|---|---|---|---|---|
| A5 | −7.6 [−11.6, −3.4] | – | – | – |
| W10 | −8.0 [−12.4, −3.8] | −6.2 [−9.8, −2.4] | −2.4 [−6.0, +1.4] | +0.2 [−1.4, +1.8] |
| W5 | −12.8 [−17.0, −8.2] | −12.2 [−16.0, −8.4] | −0.4 [−4.0, +3.0] | −1.8 [−3.8, +0.2] |
| SW | −1.8 [−5.6, +2.0] | −0.2 [−3.4, +2.8] | +0.2 [−3.2, +3.6] | +0.2 [−1.4, +1.8] |
| SF1 | −3.6 [−7.2, 0.0] | −1.8 [−4.6, +1.0] | −3.0 [−6.2, +0.4] | −1.2 [−3.0, +0.6] |
| FL | −8.4 [−12.4, −4.4] | −6.8 [−10.4, −3.2] | −3.2 [−6.8, +0.4] | +0.6 [−0.8, +2.4] |

- **Looking more often hurts a cache.** A5 costs twice A and loses 7.6 pp on π0.5 L10-50. On GR00T L10-50 it loses 6.2 pp [−10.4, −2.2] (.548 @ .148), or −7.4 [−11.6, −3.0] against the placebo.
- **On L10-50 the two effects add roughly.** Going from every 10 to every 5 controls costs −7.6 (full cameras) / −4.8 (wrist). Going from full to wrist costs −8.0 (at 10) / −5.2 (at 5).
- **At matched IR ≈ .06–.07, SW is best in all four π0.5 cells** and W5 is worst on LIBERO-10. SW is the only lever with no detectable loss anywhere.
- **On Spatial-500, the wrist camera alone holds.** W10 matches A (.976 vs .974) at less than half the cost (.037 vs .078). It still does not reach P10's .988.

### 3.2 Looking less: the lever versus the placebo and replicates

| GR00T, SR @ IR | L10-50 | L10-500 | Sp-50 | Sp-500 |
|---|---|---|---|---|
| A | .610 @ .074 | .814 @ .075 | .868 @ .075 | .964 @ .076 |
| SHIFT | .622 @ .076 | .828 @ .076 | – | – |
| SF1 | .574 @ .066 | .830 @ .068 | .832 @ .063 | .954 @ .064 |
| FL | .574 @ .056 | .768 @ .053 | .830 @ .059 | .934 @ .055 |

**Churn on GR00T L10-50 (pairs whose outcome differs from A).**
- SF1 116 (49 gained / 67 lost); FL 116 (49 / 67); SHIFT 114 (60 / 54); A5 125 (47 / 78).
- Run-to-run A churn (R8 A vs R6 A replicates): 11–20.
- So the churn itself is the closed loop's generic sensitivity: a 5-control timing shift creates the same number of flips. **The lever's specific harm is the net bias.**

**Lever minus placebo, in pp.**
- L10-50: SF1 −4.8 [−9.0, −1.2]; FL −4.8 [−9.0, −0.8].
- L10-500: SF1 +0.2 [−3.0, +3.6] (46 flips vs placebo 87); FL −6.0 [−9.6, −2.2].

**π0.5 replicate churn** (R8 A vs the three R6 A replicates): 46–52 flips per 500 on L10-50, 47–48 on L10-500, 29–32 on Sp-50, 9–12 on Sp-500. The lever arms on L10-50 flip 94 (SF1) to 148 (W5).

**SF1 replicates R7 closely.** Success is within 0.6 pp of R7 in 6/8 cells, 1.2 pp on π0.5 Sp-500 and GR00T L10-50. Example: π0.5 L10-50 .682 vs .684, with 29/30 flips between runs.

**Conclusion.** Gating makes extensions safe on dense libraries but not on sparse ones. The ungated lottery (about one extra block per supported look) loses 3.0–8.4 pp in 7 of 8 cells.

### 3.3 Twins on GR00T L10-50 (1,500 pairs)

| | SF1 | FL | SHIFT |
|---|---|---|---|
| Median perturbation control | 20 | 20 | 15 |
| Perturbation was the lever's blind decision | 99 % | 98 % | 0 % |
| Action gap at perturbation, p50 (library σ) | .228 | .185 | .149 |
| Gripper mode differs at perturbation | 0 % | 0.2 % | 0 % |
| AUROC of the action gap for an outcome flip | .51 | .52 | .49 |
| State gap after 1 / 4 / 16 / 32 decisions (σ) | .020 / .037 / .119 / .201 | .015 / .033 / .122 / .238 | .011 / .027 / .114 / .171 |
| AUROC of the state gap for a flip, lags 1–4 | .47–.51 | .50–.51 | .46–.51 |
| AUROC of the state gap for a flip, lags 8–32 | .57–.61 | .56–.60 | .54–.60 |

- The objects stay bit-identical for a median of 39–43 controls after the perturbation.
- The end-effector gap 10 controls after the perturbation is 0.6–0.9 cm.
- The object gap exceeds 1 cm at a median of 56–57 controls in flipped pairs versus 63–75 in non-flipped pairs. It exceeds 3 cm at 68–75 versus 98–103.
- About 99 % of all pairs, placebo included, eventually differ by more than 1 cm.

**Reading.** The perturbation is tiny and the grasp/release decision is never changed. The outcome is decided 40–80 controls later. The divergence profile of the lever is the same shape as the placebo's. **No per-episode cause can be pinned to the extension event.** The harm is a small systematic bias riding on generic closed-loop amplification.

### 3.4 How much a follow block differs from a fresh look, measured in the closed loop

GR00T, same observation, the deferred shadow look as reference, in library action-σ. "Grip flip" is open-vs-closed disagreement on the gripper command.

| Block | Motion gap p50 (L10-50 / L10-500) | Grip flip (L10-50 / L10-500) |
|---|---|---|
| A's own blind tail, 5 controls after a look | .194 / .155 | 6.5 % / 8.7 % |
| FL follow, 10 controls after a look | .263 / .205 | 8.8 % / 11.6 % |
| FL follow, 15 controls after a look | .247 / .184 | 8.1 % / 11.0 % |
| FL at 10 controls, gate would allow (inits 0–29) | .211 / .169 | 0.2 % / 0.1 % |
| FL at 10 controls, gate would refuse | .327 / .221 | 15.3 % / 14.7 % |
| FL at 10 controls, interior / mixed stage | .192 / .358 (L10-50) | 0.7 % / 17.8 % |

- This confirms in the closed loop what my proposal found offline: the gate removes gripper disagreements.
- Blocks the gate would allow are about as close to a fresh look as A's own accepted blind tail.
- **SF1 still loses on sparse libraries.** So the loss is not information discarded at the extension. That matches the twins.

### 3.5 Where the extra failures are (π0.5 L10-50 forensics, all 500, heuristic labels)

| | P10 | A | SW | SF1 | W10 | A5 | W5 | FL |
|---|---|---|---|---|---|---|---|---|
| Successes | 454 | 359 | 350 | 341 | 319 | 321 | 295 | 317 |
| `grasp_miss` | 11 | 69 | 75 | 82 | 93 | 100 | 101 | 104 |

Extra missed grasps account for most of each lever's loss versus A:

| Arm | Extra missed grasps / lost successes | Other notable increases |
|---|---|---|
| A5 | +31 / −38 | – |
| FL | +35 / −42 | – |
| W10 | +24 / −40 | drops +4 |
| W5 | +32 / −64 | never-reached +8, undone +9, unknown-release +8, failed re-grasp +7 |
| SF1 | +13 / −18 | – |

**Grasp audit (9,141 close attempts).**

| | P10 | A | SF1 | FL | W10 | A5 |
|---|---|---|---|---|---|---|
| Close attempts per episode | 2.35 | 2.83 | 3.05 | 2.85 | 3.06 | 4.37 |
| Lift rate per attempt | .813 | .663 | .625 | .634 | .599 | .661 |
| Lift rate on the first attempt | .953 | .904 | .889 | .840 | .913 | .878 |

- Closes issued at a fresh wrist-only look (W10) lift .549, versus .628 at A's looks.

### 3.6 Why looking more often hurts (`cadence_mech`, inits 0–29)

Per 100 controls. "Seam jumps" = action jumps at look seams, in units of the within-chunk median delta. "Gripper toggles" = gripper sign changes.

| Arm | Demo switches | Seam jumps | Gripper toggles |
|---|---|---|---|
| π0.5 L10-50 A | 4.0 | 59 | 2.16 |
| π0.5 L10-50 A5 | 5.3 | 85 | 2.70 |
| π0.5 L10-50 W5 | 5.0 | 88 | 2.64 |
| π0.5 L10-50 W10 | 3.9 | 67 | 2.18 |
| π0.5 L10-50 SW | 4.2 | 60 | 2.21 |
| π0.5 L10-50 SF1 | 4.1 | 61 | 2.29 |
| π0.5 L10-50 FL | 4.3 | 66 | 2.41 |
| π0.5 L10 P10 | – | 60 | 1.59 |
| π0.5 L10-500 A | – | 49 | 1.86 |
| π0.5 L10-500 W10 | – | 56 | 2.21 |
| π0.5 L10-500 W5 | – | 77 | 3.23 |
| GR00T L10-50 A | 4.1 | 73 | 2.69 |
| GR00T L10-50 SHIFT | – | 71 | 2.82 |
| GR00T L10-50 A5 | 6.0 | 93 | 4.03 |

- Re-retrieving every 5 controls stitches more demonstrations together and toggles the gripper 25–50 % more often. That matches the +54 % close attempts in A5.
- **The 10-control commitment is a quality feature of the cache, not only a saving.**
- Wrist-only retrieval at 10 controls stitches slightly less consistent continuations: seam jump 6.69 vs 5.89, progress slip p50 .021 vs .017.
- Seam jumps alone are not harmful: P10 has the same seam mass as A (60 vs 59).

### 3.7 Follow lottery, all anchors (inits 0–29; 1,775–5,708 anchors per arm; 92–98 % fully supported)

**Effect of an extra block on how far the next look lands from the library** (log distance ratio, E1 − E0):
- Where the gate would refuse: GR00T L10-50 +.021 [+.011, +.032]; GR00T Sp-50 +.032 [+.002, +.061]; π0.5 L10-50 +.020 [+.005, +.037]; π0.5 Sp-50 +.079 [+.037, +.118]; π0.5 Sp-500 +.049 [+.014, +.087]; π0.5 L10-500 −.002.
- Where the gate would allow: every interval includes 0, except π0.5 L10-500 at +.018 [+.003, +.035].

**Stage of the next look.** Extending from an interior anchor raises the share of next looks landing in mixed/unknown neighbourhoods from .08–.25 to .19–.42 (E0 → E2). Part of this is mechanical: a later look sits nearer the gripper event.

**Success effects by stratum are unresolvable.** Example: π0.5 L10-50 all-anchor E1 − E0 = +0.33 [−0.19, +0.89], which even contradicts the arm-level −8.4 pp in point estimate.

### 3.8 Debug-mode invariance (GR00T, decision level)

| Comparison | Pairs with identical rows/weights over the whole common prefix | Median first-divergence decision | Diverge at step 0 |
|---|---|---|---|
| R8 A (debug on) vs R6 A rep1 (debug off), L10-500 | 407/500 | 18 | 9 |
| Same, L10-50 | 403/500 | 20 | 8 |
| R6 rep1 vs rep2 (both debug off), L10-500 | 400/500 | 16 | 13 |
| Same, L10-50 | 398/500 | 26 | 3 |

- Debug mode shows no behaviour change beyond the existing nondeterminism.
- GR00T A is not fully deterministic: about 20 % of pairs diverge somewhere. Twin attribution still holds, because the lever acts at decision 2–4, before typical natural divergence.

## 4. Bugs and data-quality problems

1. **exposure_hazard (`tools/decision/exposure_hazard.py`).**
   - It puts every stage into `unknown`, even though the anchor's stage comes from its own pre-assignment neighbours (and FL logs `diag.stage_ok_by_e`).
   - It uses only the first supported assignment per episode.
   - robot-demo drift reports "aligned library robot-state reference unavailable", although `catalog` + `rs.npy` + the logged `successor_rows` suffice at decision checks.
   - Stall entry is unavailable.
2. **follow_vs_look / divergence.** The default output has no gripper flips (`gripper_flip_n = 0`) and is in normalized units, because arm manifests lack `gripper_threshold` and `action_scale`. Put both in the R8 arm manifest or `meta_*.json`.
3. **twin_divergence / paired_diverge.**
   - `--limit` applies per arm, so a limited run admits 0 pairs.
   - Object onset requires an `--envelope` that nobody has frozen yet.
   - About 30 min and 9.5 GB RSS for 2,000 episodes.
4. **blind_drift.** Did not finish in 57 CPU-min for 1,500 episodes (robot-only fallback). It needs profiling or vectorization before R9.
5. **churn.** A single run root means no in-tool π0.5 replicate or placebo control.
6. **State marker.** `state/r8_groot_l10_500_CU.AUG_DONE.mixed_610613bf` has a non-standard name ("mixed"). That is not my lane; the coordinator should check it.
7. **Wrist look path.** The per-request full look in the W/SW arms is priced at the stock .152. R7 Addendum A measured .1656 for the per-request path. The W10/W5 cost advantage is therefore slightly overstated: for the first look this is negligible; SW's full looks are affected.

## 5. What R9 should do

1. **Drop as methods:**
   - SF1 on sparse libraries.
   - FL and any ungated extension.
   - A5 / W5 (more frequent looks).
   - W10 on LIBERO-10.

   Keep W10 as a candidate only for dense Spatial-type tasks (.976 @ .037). Keep SW as the safe look-half (no loss in 4/4 cells, IR −.011 to −.020). Keep SF1 on dense libraries only (lever − placebo +0.2 on GR00T L10-500).
2. **Treat commitment length as a quality knob, not only a cost knob.** Measure cadence 10 vs 15 vs 20 with gating against the SHIFT placebo, rather than adding looks. The data say chattering across demos is what hurts.
3. **Where to search for "hard" stages:** the grasp approach. Every lever's extra failures are mostly missed grasps. Any segmentation should be validated on whether it isolates the pre-grasp alignment window, using forensics `grasp_miss` onsets on inits 0–29.
4. **Fix exposure_hazard** (all-anchor estimand, pre-assignment stage from neighbours, drift from the catalog). For stage-level success effects, use ≥ 3 lottery arms per cell or a fixed-stage factorial (extend only in stage s, one arm per s) against the placebo. One 500-episode lottery arm cannot resolve per-stage harm.
5. **Unblock camera_shadow on π0.5** once augmentation lands, to test whether W10's grasp losses come from different retrieved demos at the pre-grasp look. Add the missing GR00T one-camera path only if SW-type gating is pursued for GR00T.
6. **Add a SHIFT placebo for π0.5 L10** (two arms). With π0.5's higher rollout noise, a lever without a placebo or replicate control cannot be read.

Files:
- `/tmp/r8cb_E4_lazy_levers/{arm_sr.txt, replicate_churn.txt, churn_*/, eh_groot_l10_50/, fl_lottery.json, fl_anchors_*.csv.gz, cadence_mech.json, fvl_groot_l10_meta/, twin_groot_l10_50/, forensics_pi05_l10_50/, grasp_pi05_l10_50/}`
- Scripts: `/home/weiland/projects/openpi/exp/offline_search/rounds/r08/ideation/E4_lazy_levers/{fl_lottery.py, cadence_mech.py}`
