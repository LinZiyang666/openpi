# R9 round 2 — data analysis (fable)

**Rule-1 compliance.** Every loader in `round2/tools/` filters to inits 0–29 before any computation
(`common.discovery_mask`, `init < 30` on journals/forensics/server logs; see the comments marked "rule 1"). Heads are
fitted on inits **0–19** (`TRAIN_INITS`) and every offline evaluation uses the pure-cache arm's decisions at inits
**20–29** (`EVAL_INITS`); the closed-loop screening uses the 100 pairs tasks 0–9 × inits 20–29. No file of
`r09_astra_holdout*` was opened; the round-1 holdout numbers I was told were discarded and did not enter any choice.
Rule 2: nothing below is task-indexed at decision time (the per-task ridge head is the *existing* corrector design and
is reported as such next to a single task-agnostic head; the empty-grasp trigger uses one threshold per robot).

Prefix: `P="taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python"`,
`T2=exp.offline_search.rounds.r09.explore_fable.round2.tools`. Gap = RMS over the executed 5 controls × 6 motion
channels of (served − policy shadow) in library-σ units. LIBERO gripper convention: **+1 = close, −1 = open**
(verified on the captured controls: close command → aperture .04 → .009 and the object rises).

## 1. Why the half-strength corrector works on Spatial-50 and not on LIBERO-10 (astra's arms, inits 0–29 only)

Server logs of `r09_astra_confirmation` (corrector and cache arms, inits < 30), `r9_correction_rms` = RMS of the
half correction in σ units per look decision:

| cell | SR cache → corr | controls/episode cache → corr | gripper toggles /100 controls, cache (succ / fail) → corr | mean half-correction: successes / failures | correction by episode quartile, failures |
|---|---|---|---|---|---|
| π0.5 Spatial-50 | .820 → .933 | 125 → 112 | 2.21 (1.30 / **6.32**) → 1.53 | .099 / .234 | .11 → .25 → .30 → .29 |
| π0.5 L10-50 | .743 → .757 | 327 → 317 | 1.90 (1.53 / 2.97) → 1.84 | .070 / .170 | .09 → .16 → .22 → .22 |
| GR00T L10-50 | .583 → .637 | 366 → 348 | 2.39 (1.78 / 3.24) → 2.17 | .066 / .170 | .07 → .15 → .21 → .24 |

- In every cell the correction is small while the episode is on track (≈ .07–.10σ) and grows 2.5× once an episode
  is failing; it tracks retrieval distance (corr(correction, d1_rel) = .44–.61). The corrector is therefore mostly
  "pulling back" states that are already off the demonstrations; a half-strength pull does not bring a failing
  LIBERO-10 episode back.
- Spatial failures are a distinct physical mode: the gripper toggles 5× more in failed than in successful episodes
  (6.3 vs 1.3 per 100 controls: the cache closes on nothing, reopens, retries around the bowl rim). The corrector
  removed most of those episodes. LIBERO-10 failures do not have that signature (3.0 vs 1.5) and are longer-horizon.

**Per-phase held-out gap** (`$P -m $T2.corrector [--per-task]`; heads fitted on inits 0–19; evaluation = pure-cache
arm's cache decisions at inits 20–29; phases from the executed gripper history: approach = open before the first
close, grasp = ±1 decision around a close transition, carry = closed, release = around an open transition, post =
open after a release, i.e. the approach to the next object in two-object tasks):

| cell / head | all | approach | grasp | carry | release | post |
|---|---|---|---|---|---|---|
| π0.5 Sp-50 served | .500 | .305 (n 357) | .545 (193) | .573 (708) | .630 (17) | .892 (19) |
| π0.5 Sp-50 per-task head ×.5 / ×1.0 | .361 / **.293** | .245 / .238 | .389 / .297 | .403 / .315 | .391 / .256 | .645 / .532 |
| π0.5 L10-50 served | .407 | .280 (590) | .338 (340) | .451 (1383) | .402 (214) | .454 (781) |
| π0.5 L10-50 per-task head ×.5 / ×1.0 | .317 / **.281** | .246 / **.248** | .273 / .252 | .328 / .270 | .334 / .309 | .366 / .331 |
| π0.5 L10-50 single task-agnostic head ×.5 / ×1.0 | .355 / .339 | — | — | — | — | — |
| GR00T L10-50 served | .517 | — | .515 (296) | .440 (1661) | .411 (290) | .634 (1348) |
| GR00T L10-50 per-task head ×.5 / ×1.0 | .393 / **.330** | — | .409 / .377 | .352 / .315 | .356 / .344 | .447 / .336 |

- The corrector generalizes across initial states: fitted on 0–19 it removes 41% (Sp-50), 31% (π0.5 L10-50), 36%
  (GR00T L10-50) of the gap at inits 20–29 at full strength, 28% / 22% / 24% at half strength. Full strength is
  uniformly better offline in every phase; the half strength used in round 1 leaves a third of the achievable
  reduction on the table.
- On LIBERO-10 the phase that moves least is the **approach** (−11%, .280 → .248); the carry/post phases move most
  (−40%). The approach is where grasp alignment is decided.

**Failure onsets** (R8 forensic labels of the pure-cache arms, inits 0–29; onset decision = the labelled
decisive event; corrector = per-task head fitted on 0–19, evaluated at the onset decisions of inits 20–29):

| cell | failures (0–29) | dominant labels | gap at/just before onsets: served → corrected | gripper-sign disagreement with the policy: all decisions / at onsets |
|---|---|---|---|---|
| π0.5 Sp-50 | 60 | grasp_miss 34, misplace 12, drop 8 | .473 → .462 (−2%) | .169 / **.043** |
| π0.5 L10-50 | 85 | grasp_miss 45, undone 13, misplace 12 | .421 → .335 (−21%) | .101 / .094 |
| GR00T L10-50 | 109 | grasp_miss 63, unknown_release 19, misplace 16 | .419 → .363 (−13%) | .159 / .076 |

- At the decisive moment the cache's **gripper command agrees with the policy** (disagreement at onsets is lower
  than average); failures are positioning errors, not gripper timing. A gripper head (offline it halves the sign
  disagreement, .101 → .059) is therefore not the lever and was not pursued.
- On Spatial the corrector barely changes the action at the onset itself (−2%) yet removes most failures closed
  loop: it acts earlier, keeping the approach on the demonstrations. On LIBERO-10 it changes the action at the
  onset more (−21%) and still does not rescue: a 10-control, .3σ-level correction of an already misaligned grasp
  does not change the contact outcome. Half of LIBERO-10 failures are grasps that close on nothing.

## 2. A task-agnostic failure-onset detector: the empty grasp (`$P -m $T2.aperture`)

Robot-observable signals only: executed gripper command (+1 = close) and finger aperture (wire state dims 6, 7).
"Empty grasp" = close command held ≥ 10 controls **and** aperture below a threshold. Threshold calibrated on inits
0–19 as half the 1st percentile of the aperture while an object is held (truth stage *carry*; the deployable
calibration uses the same statistic on the demonstrations); evaluated on inits 20–29:

| cell | holding aperture p1 / p10 / p50 (m) | empty close p50 | threshold | grasp-miss failures flagged | all failures flagged | successes flagged | flag vs forensic onset |
|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | .0021 / .0029 / .0147 | .0005 | .0010 | **12 / 12** | 22 / 27 | **1 / 73** | 40 controls earlier (median) |
| π0.5 Sp-50 | .0018 / .0019 / .0024 | .0018 | .0009 | **11 / 11** | 19 / 21 | **0 / 79** | 31 controls earlier |
| GR00T L10-50 | .0017 / .0029 / .0147 | .0006 | .0009 | **23 / 25** | 28 / 38 | 4 / 62 | 73 controls earlier |

Every grasp-miss failure of π0.5 (and 92% of GR00T's) is flagged, 70–90% of all failures are flagged, with 0–6% false
alarms in successful episodes, and the flag precedes the forensic onset by 30–70 controls (the forensic onset is the
window entry of the final attempt). This is the sharp, cheap version of the no-progress guard: it fires at the
physical event (fingers meet nothing) instead of after a progress stall.

## 3. Candidates screened closed loop (inits 20–29, 100 pairs, h100 + timan107; run root `r09_fable_r2`)

Heads fitted on inits 0–19 only; controls on the same 100 pairs and topology: π0.5 L10-50 pure cache .740 / pure
policy .930 (`r09_fable_grown`, round 1), GR00T L10-50 and Spatial-50 pure-cache arms run in this chain.

| arm | method |
|---|---|
| `r9f2_<cell>_corr1pt` | CorrectedCache: frozen cache + per-task ridge/RFF head (existing design), **full strength**, motion only |
| `r9f2_pi05_l10_50_corr1single` | same with one task-agnostic head (task one-hot as an input feature) |
| `r9f2_<cell>_gm_corr1pt` | GraspMissCalls: corr1pt + empty-grasp trigger → policy call burst (2 calls = 20 controls), ≤ 2 triggers/episode, no other calls |
| `r9f2_<cell>_gm_corr0` | the trigger alone on the plain cache (blend 0) |
| `r9f2_<cell>_corr0` | plain cache control (Spatial-50, GR00T L10-50) |

**Status at hand-back (2026-10-02 00:0x CDT): chain `r9f_r2` (tmux, self-terminating) has finished 2 of 11 arms; the remaining 9 (grasp-miss trigger arms, GR00T L10-50, Spatial-50) complete over the next ~30 min. Read them with `paired_r2` (HANDBACK.md).**

| arm (π0.5 L10-50, inits 20–29, 100 pairs) | SR @ IR | paired vs cache .740 (same pairs/topology) |
|---|---|---|
| pure cache (round-1 control) | .740 @ .0765 | — |
| pure policy (round-1 control) | .930 @ .504 | +19 pp [+10, +27] |
| corrector **full strength**, per-task head fitted on 0–19 | **.630** @ .0762 | **−11 pp** [−23, 0], +14/−25, p = .11 |
| corrector full strength, single task-agnostic head | **.540** @ .076 | **−20 pp** |
| grasp-miss trigger (+ full corrector / + plain cache), GR00T, Spatial | pending | pending |

**Reading.** The full-strength corrector — the offline optimum on held-out inits (−31% gap) — is clearly harmful
closed loop on LIBERO-10 (−11 and −20 pp), while the half strength of round 1 was +1.4 pp. Offline distance to the
policy mispredicts closed loop for the third time this round (grown library, full corrector). The per-task pattern
(task 2: 1.0 → .6, task 3: 1.0 → .8, task 0: .4 → .6) repeats the grown-library pattern: a stronger pull toward the
policy helps the tasks the demos handle badly and breaks the tasks they handle well. Consequence: correction strength
must be shrunk (≤ .5) and the remaining LIBERO-10 gap cannot be closed by correcting the cached action — the
grasp-miss recovery call (observation-keyed) is the live candidate; its arms are the pending ones, plus batch 2
(half-strength base + trigger, no-progress + half corrector, trigger + uniform coin) already prefitted and emitted in
the same run root (`arms_in_batch2.json`, `arms_in_batch2b.json`).

## 4. Tools and tests

`round2/tools/corrector.py` (head fitting, phase labels, held-out evaluation), `aperture.py` (detector study),
`methods.py` (`CorrectedCache`, `GraspMissCalls` serving classes; CPU plugin selftests PASS with `--blind
--policy-tail --judge guard_only`), `paired_r2.py` (paired closed-loop analysis across run roots);
`tools/tests/test_round2.py` (6 tests: phase labels, ridge recovery, weight normalization, parameter validation,
trigger logic incl. burst and cap). Head artifacts: `round2/out/corrector/head_*.npz` (+ `eval_*.csv`).
