# R8 E3: Failure forensics. PROPOSAL (explorer E3, Opus)

Scripts are in `exp/offline_search/rounds/r08/ideation/E3_failure_forensics/`: `extract_compact.py`, `forensics.py` and `grasp_drop_pairs.py`. Outputs are in `/tmp/r8_E3_failure_forensics/{compact,out}/`. All work was CPU-only on CPUs 22-25/66-69. I started no simulator, server or worker.

## 1. Summary

**What I read.** The brief (all sections); R6 P3 v2 `SCHEMA_V2.md` and `telemetry.py` (client physical tap); the snapshot npz layout; the R7 C4 `failure_clock` and E2 findings; the trace-collector HDF5 layout (`dual_20260923`); and the stock R7 client and server logs (`r07_main`).

**What I measured.** I converted the R6 superset pilot's raw client telemetry into a compact physical series for every A (pure cache) and P10 (pure policy, 10-control commitment) episode:
- Scope: 8 cells × 2 arms × 3 seeds × 10 tasks × inits 0-1.
- 971 attempts. After excluding 11 infrastructure exceptions, 960 accepted episodes and 209,658 controls.
- I then ran an automatic failure taxonomy on kinematics plus goal predicates, a grasp-attempt audit, a drop-mechanism audit, onset timing, and paired divergence on the same (task, init).

**What surprised me.**
1. **Cache failures and policy failures are different kinds of failure.**
   - A fails 89/480 (18.5%). 48% are grasp misses and 28% are drops, so 76% happen while acquiring or holding the object.
   - P10 fails 43/480 (9.0%). 51% are misplacements and only 26% are grasp miss or drop.
   - Grasp miss and drop together account for more than all of A's 46 excess failures (+35 and +22; misplacement is −10).
   - When A succeeds, its sim-truth stage durations match the policy's exactly. The cache is not slow; it fails at the grasp.
   - R7's gripper split labels exactly this pre-grasp alignment as "interior = easy".
2. **Every failure is a step-cap timeout (132/132).** LIBERO never terminates early, so "timeout" is not a failure mode. The decisive event comes at a median control of 88 (A) and 110 (P10). After it, 65-70% of the episode (about 300 controls) is spent going nowhere.
3. **Three defects in the R6 superset client data.** None of them was documented.
   - (a) `before.*` equals the previous `after.*` in 209,658 of 209,658 controls. That is about 50% of the 72 KB per control.
   - (b) Contact geom IDs do not line up with `entity_ids.geom_names` (155 names, but IDs go up to at least 158). For example, geom 10 resolves to `robot0_g0_vis` but is really the table. **No P3 v2 contact can be named correctly.**
   - (c) Objects are spawned above the table and settle 6-10 cm in z during the 10 wait controls. Any "moved" or "lifted" test that uses control 0 as reference is wrong. My first pass flagged 565 of 828 successes as "disturbed"; after the fix it flags 0.
4. **The environment seed changes physics.** With different seeds, the sim state differs during the wait phase (controls 0-3), before any issued action differs, in 40% of pairs. Pairing across arms therefore needs a shared seed.
5. **Pure cache is nearly deterministic across seeds.** 24/480 A-vs-A pairs are bitwise identical for the whole episode. On Spatial, more than half of A-vs-A pairs never separate by 3 cm, and A-vs-A outcome concordance is 89%. Seed replicates of a cache-only arm add almost no information.

## 2. Profile tools wanted

All tools run offline on CPU over the debug data. "Compact series" means the client per-control fields in section 3.

| Tool | Question it answers | Inputs | Outputs | Stage decision it informs | Pri | CPU |
|---|---|---|---|---|---|---|
| **T1 `forensics.label`** | Why did this episode fail, and when? | Compact series plus BDDL goal predicates (target objects, destinations) | `episodes_forensics.csv`: label (grasp_miss / drop / drop_regrasp_fail / misplace / held_not_placed / undone / fixture_not_done / wrong_object / never_reached), onset control and decision, subgoals done, grasp attempts, tail motion (stall / oscillation / moving), per-predicate detail | Which sim-truth stage failures come from, per arm and model | P0 | About 6 s for 960 episodes (measured, after extraction) |
| **T2 `forensics.events`** (sim-truth stages) | Where are approach, grasp window (near→lift), carry, place, release and retreat on each control? | Same as T1 | Per-control and per-decision `truth_stage`, joined to server decisions (source, look reason, neighbours' library stage) | Gold standard for scoring any robot-only segmentation from E1: boundary error and "hard" coverage of real failures | P0 | Trivial |
| **T3 `grasp_audit`** | Is failure caused by pre-grasp misalignment? How far ahead does it show? | Compact series; policy-arm successful grasps on the same (task, init) as reference; decision records (source, age since last look, neighbour library episode/step) | One row per close-command onset near a goal object: eef pose **in the object frame** (xyz plus yaw), width at closure, lift within 40 controls yes/no, issuing source. Offset AUROC. Per-task grasp-pose clusters. | Whether a "final alignment" stage exists, and its radius and lead time; this defines the hard stage | P0 | Seconds |
| **T4 `drop_audit`** | Is a drop a premature open command (blended gripper channel) or a slip under a closed command? | Compact series plus served chunks | Per carry segment that ends away from the destination: command at loss, finger width, lift height, eef acceleration, issuing decision and its neighbours | Whether the "carry" stage needs calls or only a gripper-channel fix | P1 | Seconds |
| **T5 `paired_diverge`** | Where does the same (task, init, seed) split between the cache arm and the policy arm, relative to the sim-truth events? | Compact series across arms (requires the shared seed schedule) | First exit from the policy-replicate envelope; event-aligned (not control-aligned) comparison; a transition table (A-fail/P-success by mode); the first sim-truth event where discordant pairs differ | Validates that "first divergence" is not onset, and localizes decisive stages | P0 | Seconds |
| **T6 `trigger_vs_onset`** | Which online signal fires before the decisive event, and with what lead and false-alarm rate per sim-truth stage? | T1/T2 plus server decision logs (stall, state valve, neighbour dispersion, shadow cache-vs-policy divergence, R7 stage label) | Lead-time distribution, per-stage ROC and hit rate, and share of calls that land inside the grasp window ("call coverage") | Operating point of every call trigger per stage. It also shows whether R7's tilted calls ever landed in the grasp window. | P0 | Seconds |
| **T7 `sim_replay`** | Can we rebuild any field offline (named contacts, renders, fixture joints) from (init, seed, issued actions)? | Init index, seed, `action_issued`, per-control `sim_state` digests | Replayed series plus a bitwise digest-match report | If it holds, the online client stays slim; if not, the compact series is the only ground truth | P0 validation, P1 tool | About 5-10 s per episode without render (estimate) |
| **T8 `episode_card` / `arm_rollup`** | The standard report | T1-T6 | Per-episode card: decision strip (cache / policy / look, truth stage, triggers), top-down XY of eef and objects, target z and finger width, label and onset. Per-arm roll-up: mode × stage table and paired transitions. | Human validation and the headline tables | P0 | Seconds (PNG) |
| **T9 `collector_selfcheck`** | Is the debug data physically sane? | Reset record | Geom→name alignment (resting objects must touch the table at reset), post-settle reference frame, `before`==`after` redundancy, seed logged | Prevents a repeat of defects (b) and (c) | P0 | Trivial |

**Label validation (needed before T1 counts as ground truth):**
- Re-render 50 failures and 20 successes, stratified by label (from `sim_state` via T7, or from decision images), and hand-label them once.
- Target: at least 85% agreement per major label.
- Thresholds are explicit: lift 3 cm; carry radius = per-task p95 + 2 cm, clipped to [8, 15] cm; lift within 40 controls.

## 3. Debug-mode data fields required

| Field | Side | Why | Size | Pri |
|---|---|---|---|---|
| `control`, `decision_id`, `chunk_offset`, `source` (cache / policy / blend), `wait_phase` | client / control | Joins and truth-stage × source | ~16 B | P0 |
| `action_issued` (7, float32, post-transform) | client / control | Replay, gripper-command audit | 28 B | P0 |
| eef pos+quat, gripper qpos (2), joint pos (7) | client / control | All kinematic labels | ~64 B | P0 |
| Pose (pos+quat) of **every movable object**, float32 | client / control | Grasp, lift, drop, misplace, wrong object, disturbance | 28 B × K (K = 5-9) | P0 |
| Goal-predicate booleans (BDDL `goal_state` via `_eval_predicate`) | client / control | Subgoal progress, "undone" | <8 B | P0 |
| **Named** contact summary: body-level pairs involving the finger pads, the hand, or a goal object, with normal force (via `geom_bodyid` / `body_id2name`) | client / control | Pad contacts separate pinch, touch and slip | ~10 × 12 B | P1 (must pass T9) |
| `sim_state` (123 float64 on L10) | client / control (at least per decision) | Exact re-render, fixture joints, replay check | ~1 KB raw, ~0.5 KB compressed | P0 |
| Articulated-fixture joint qpos (stove knob, microwave door, cabinet drawer) | derivable from `sim_state` | `fixture_not_done` diagnosis | 0 | P1 |
| Per-episode: task_id, init idx, init-state SHA, **env seed**, MuJoCo/robosuite/LIBERO versions, geom/body name tables (once), BDDL goal list, max steps, termination reason | client / episode | Pairing, replay, portability | ~5 KB | P0 |
| Images at decisions only (both cameras, 224², JPEG) | client / decision, optional per arm | Visual check of a label sample; re-render covers the rest | ~15 KB per image | P2 |
| Server / decision: look/call flag and reason, age since last look, neighbours' (library episode, step, success, stage), shadow-policy gripper and position at the first 10 controls | server / decision | T3, T4, T6 (reuse E5's spec) | Small (per E5) | P0 |

**Budget.** The compact series measured 53 KB per episode (971 episodes → 51 MB npz). Adding `sim_state` per control makes it about 0.3-0.6 MB per episode on the client. For 50 arms × 500 episodes that is about 15 GB, compared with 60 MB per episode (1.5 TB) for the R6 superset.

**Not worth collecting:**
- `before.*`: it is identical to the previous `after`.
- `entity_ids` repeated on every control.
- Actuator commands at every physics substep (2.4% of bytes; torques follow from state plus action).
- Raw `cvel`, `cfrc_ext`, `efc_force`, and unnamed raw contacts.
- Concatenated `object-state` and `robot0_proprio-state` (duplicates of the per-object fields).
- Images on every control.
- `step_N.npz` controller snapshots, except in branch experiments. `restore_certified=false`; replaying from reset avoids needing them.

## 4. Arms wanted (500 episodes each)

**Common rules for all arms:**
- Use the standard 10 tasks × 50 test inits.
- **The same environment seed per (task, init) in every arm** (finding 4).
- The debug client is on in every arm.
- A shadow policy is needed only where E2/E5 ask for it.

| Group | Arm(s) | Cells | Why (E3 lens) | Pri |
|---|---|---|---|---|
| 1 | Pure policy, 10-control commitment (P10), debug on, cache proposal computed in shadow | 4 (π0.5 / GR00T × L10 / Spatial; library-independent) | The paired reference for T5; the policy's grasp poses are T3's reference distribution. The trace collector has no sim truth and uses a 5-control cadence. | P0 |
| 2a | A (ρ = 0) | 8 (× library 50 / 500) | Failure-mode baseline. A is nearly seed-deterministic, so one seed per (task, init) is enough. | P0 for the 50-lib cells, P1 for 500 |
| 2b | C = A + stall + uniform random calls at owner IR | 8 | Do uniform calls shift grasp_miss or drop? T6 call coverage of the grasp window. | P0 for 50-lib, P1 for 500 |
| 3 | R7 stage-tilted calls at the same ρ | 4 (50-lib) + 4 (500-lib) | Measure whether tilted calls actually land in the sim-truth grasp window, and their per-mode effect | P1 |
| 4 | E4's factorial | per E4 | I only ask that T1/T2 be run on it: does look-less or look-half move failures toward drop or misplacement? | Per E4 |
| 5a | **Oracle grasp-window calls (diagnostic, privileged)**: call the policy at every anchor while the eef is inside the sim-truth window (near an un-lifted, unsatisfied goal object); cache elsewhere | L10-50, both models (2 arms) | Upper bound on what perfect segmentation of the hard stage can buy. The window covers 21.7-22.5% of decisions on L10 (33% on Spatial), so the estimated IR is about .2, not .076. | P0 |
| 5b | **Oracle-tight at owner IR**: same, but only inside a 5 cm radius and at most 2 calls per goal object | L10-50 both models, plus Spatial-50 both models (4 arms) | Is final alignment alone enough at about IR .08? If even this does not recover SR, stage segmentation is not the bottleneck. | P0 |

- **Total from my lens:** 4 + 16 + 8 + 6 = 34 arms, plus E4's, which fits 40-60 arms.
- **If cuts are needed:** keep group 1 (4), groups 2a and 2b on 50-lib (8), and 5a/5b (6). That is 18 arms.
- **Oracle arms are privileged.** The client passes the sim-truth window flag to the method. They must be labelled diagnostic and never compared as methods.

## 5. Preliminary evidence (R6 P3 v2 pilot, inits 0-1)

**Commands** (repo root, `PY = taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python`, scripts under `D = exp/offline_search/rounds/r08/ideation/E3_failure_forensics`):
```
$PY $D/extract_compact.py --cells pi05_l10_50 groot_l10_50 pi05_l10_500 pi05_sp_50 pi05_sp_500 groot_l10_500 groot_sp_50 groot_sp_500   # about 0.45 s per episode
$PY $D/forensics.py          # episodes.csv, pairs.csv, carry_calibration.json, determinism.json
$PY $D/grasp_drop_pairs.py   # section 5 tables; grasp_drop_pairs.window_share() gives the window sizes
```

**E-1. Failure modes (T1).** Accepted episodes: A 480, P10 480.

| | A (89 failures) | P10 (43 failures) |
|---|---|---|
| grasp_miss | 43 (48%) | 8 (19%) |
| drop + drop_regrasp_fail | 21 + 4 (28%) | 3 (7%) |
| misplace | 12 (13%) | 22 (51%) |
| fixture_not_done / held_not_placed / undone | 4 / 3 / 2 | 6 / 1 / 3 |

- L10: A has 33 grasp misses and 14 drops out of 61 failures; P10 has 14 misplacements out of 31.
- Spatial: A has 10 grasp misses and 11 drops out of 28; P10 has 8 misplacements out of 12.
- GR00T P10: 20 of its 32 failures are misplacements.
- wrong_object: 1 episode. never_reached: 0.

**E-2. Grasp attempts (T3).** An attempt is a close-command onset near a goal object; "failed" means no lift within 40 controls.
- A: 1,163 attempts, 37% fail (L10 40.8%, Spatial 27.3%).
- P10: 870 attempts, 13.3% fail (L10 16.5%, Spatial 6.3%).
- Eef-to-object-centre XY offset at closure for A: failed 5.22 cm vs succeeded 3.99 cm. AUROC is .62 for A and .59 for P10.
- The centre offset explains only part of it. An object-frame pose against a reference grasp distribution is needed (this is why T3 exists).

**E-3. Drops (T4).**
- Of A's 25 drops, 17 happen with the close command still on. Finger width at loss is 0.2-1.9 cm in most, which suggests a rim or edge pinch that slips.
- 8 are an open command issued while carrying away from the destination: a premature release, all served by the cache.
- π0.5 Spatial task 9: 7/7 A drops fall at controls 70-103. This is systematic, not random.

**E-4. Sim-truth stage durations in successes** (medians, in controls):

| | Approach | Near→lift | Lift→goal |
|---|---|---|---|
| L10, A | 76 | 25 | 48 |
| L10, P10 | 76 | 25 | 51 |
| Spatial, A | 39 | 21 | 40 |
| Spatial, P10 | 39.5 | 20 | 41 |

Grasp attempts per object in successes: mean 1.7 (A) vs 1.3 (P10) on L10.

**E-5. Onset and wasted controls.**
- Decisive-event onset median: control 88 for A (IQR 56-200), 110 for P10.
- Controls after onset: median 295 (A) and 307 (P10), which is 70% and 65% of the episode.
- Success median length is 137/138 controls, so a failure consumes about 3.9× the controls of a success.

**E-6. Pairing and determinism (T5, T7 feasibility).**
- A vs P10 on the same (task, init) separate by 3 cm at median control 54 (IQR 36-76), in all 1,440 pairs.
- P10 vs P10 (different seeds) separate at control 96. A vs A separate at 440; on Spatial the median A-vs-A pair never separates.
- So divergence from the policy is universal and comes about 3-4 decisions before the decisive event. It is not an onset marker.
- Same-seed pairs: the sim state first differs exactly at the first differing issued action in 480/480. This is only a weak test, because the actions already differ at the first decision.
- Different seeds: the state differs during the wait phase in 192 of 480 pairs per pair type.
- 24 A-vs-A pairs (GR00T L10) are bitwise identical over the whole episode, although their seeds differ.

**E-7. Format findings.** These are the three defects in section 1 finding 3: `before` is redundant, contact geom IDs are misaligned, and objects settle during the wait phase. Compact series vs raw: 53 KB vs 38 MB per episode (about 700×).

## 6. Risks and open questions

1. **Heuristic labels.** The labels are kinematic heuristics. A center-of-object offset is crude for mugs, moka pots and bowls (rim grasps). Human validation (section 2) must pass before the labels become ground truth. `fixture_not_done` is undiagnosed until fixture joints are extracted. The `living_room_table_plate_right_region` destination has no body, so "misplace" there relies on the predicate alone.
2. **Replay determinism is untested.** It needs the simulator. The cheap conclusive check is T7 on 20 R6 pilot episodes, which already contain `action_issued` and per-control `sim_state`. Replay must start from reset with the logged seed. Mid-episode restore also needs `qacc_warmstart` and controller goal state (present in `step_N.npz`, but never certified).
3. **Seed schedule in the stock client.** I did not confirm that the stock R7/R8 client derives the environment seed identically per (task, init) across arms. If it does not, cross-arm pairing (T5) and oracle comparisons are confounded by settle-phase physics.
4. **Oracle arms send privileged simulator information to the method.** This must be an explicit, labelled exception to the invariance rule. Their IR must be reported by the same ledger.
5. **Small evidence sample.** Inits 0-1 only, i.e. 20 (task, init) per cell. The shares here are pooled and only indicative. Per-cell mode shares need the R8 500-episode arms.
6. **Porting.** The taxonomy interface should be "goal objects + destinations + success sub-predicates" per benchmark adapter:
   - LIBERO: BDDL `goal_state`.
   - RoboCasa365: `_check_success` sub-conditions.
   - MetaWorld: its `info` dict already exposes `near_object`, `grasp_success`, `obj_to_target` and `success`.
   - Generic manipulators: eef pose, gripper width and object poses. Named contacts are a bonus.
7. **Open question for the coordinator and E1/E2.** Should the R8 hard stage be defined, for evaluation, as the sim-truth grasp window (T2)? Robot-only segmentations would then be scored against it, and the oracle arms (5a/5b) would decide whether stage-level allocation can pay off at owner IR before more segmentation work is funded.
