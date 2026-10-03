# R8 selection — debug-mode data collection

Written 2026-09-30 11:5x CDT by the coordinator after the explorer proposals
(`ideation/E1_segmentation`, `E2_call_value`, `E4_lazy_levers`, `E5_debug_architecture`; E3 failure forensics is
folded in when it lands). Pause point 1 is waived. R8 is a **data-collection round**: no method claims, no new
allocation rule. Every arm runs the standard 500 test pairs (10 tasks × inits 0–49) with the debug mode on.

## 1. Decisions

1. **One debug mode for the whole closed-loop system** (E5 architecture): a passive observer at the server decision
   boundary plus a client capture at every control, one schema, one reader. It never changes an action.
   Specification: `CODING_BRIEF.md` and `exp/offline_search/debug/SCHEMA.md`.
2. **Everything the live path already sees is captured at every decision of every episode**: exact wire images of
   both cameras (lossless), exact raw state and prompt, compact deployed keys, served / cache / policy chunks,
   the 16 neighbours with scores and weights, every method reason, randomization record and dispatch counter.
   Full 2×32768 raw keys only on an independent hashed 1/16 decision sample (they are recomputable from the images).
3. **Every control is captured on the client**: issued action, robot joint/EE/finger state, every movable object
   and articulation pose, entity-resolved contacts, goal predicates, actuator commands per physics substep,
   reward/done. Entity catalogs once per episode. Snapshots of the simulator at reset, at a hashed 1/16 sample of
   pre-decision points and at the end (`restore_certified=false`; no branching in R8).
4. **Shadows are deferred** (E1/E2/E5 agree): after collection, a separate GPU job computes from the stored
   observations (a) a policy chunk at every decision with a private seed, (b) three extra policy draws on a hashed
   1/32 sample (noise floor), (c) a full-look A retrieval ("shadow look") at every decision, and on π0.5 wrist-only
   and third-only retrievals (camera shadow), (d) for pure-policy arms, cache proposals against both libraries.
   Nothing is computed inline, so behaviour invariance reduces to "the observer only copies".
5. **Storage on `/home`**, target ≤ 20 MB per episode on average (images dominate); measured in the smoke and
   admitted before the full campaign. Old data is moved to `/archive` only if the measured forecast needs it.
6. **Analysis holdout (E1):** within each task, inits 30–49 are a locked holdout for any new segmentation or stage
   rule; discovery uses inits 0–29. Arm-level descriptive statistics use all 500.
7. **Determinism:** GR00T A is bit-deterministic, so GR00T lever-vs-A twin analysis is available; π0.5 twins are not
   (batched bf16 encoder), and R8 does not change π0.5 serving to force it.
8. **E3 findings adopted:** cache failures are mostly missed grasps (48 %) and drops (28 %) during pre-grasp
   alignment and carry, which R7 labelled "easy"; policy failures are mostly misplacements. Consequences: (a) the
   stock client's constant environment seed (7) must stay identical in every arm so (task, init) pairs are physically
   comparable; (b) contact geom ids must be named from the MuJoCo model's own id→name maps (the R6 client's contact
   names were misaligned) with a reset self-check; (c) "moved/lifted" tests use the post-settle state, not control 0;
   (d) the diagnostic oracle arms below test whether perfectly placed grasp-window calls can reach pure-policy SR near
   owner IR .08 — if not, segmentation is not the bottleneck.
9. **Library physical backfill (E4/E1/E3):** a one-time CPU job replays library demonstrations in the simulator
   through the same client capture, admitted only where replayed state and success match the library.

## 2. Arms (500 episodes each; debug on)

Cells: π0.5 / GR00T × LIBERO-10 / Spatial × library 50 (`current`) / 500 (π0.5 `bpool_cs`, GR00T `bpool_all`).
Budgets as R7: ρ = .30 for library 50, .18 for library 500. Fits are R7's frozen artifacts where the configuration is
unchanged (sha-checked); new variants get new fits from the same library-only rules.

| Group (owner) | Arm | Cells | Arms | Priority |
|---|---|---|---:|---|
| 1 pure policy | Policy every 10 controls, 5-control decisions (policy tail), live cache proposal where available | 2 models × 2 suites | 4 | P0 |
| 2 no stages | A (pure cache, 10-control commitment) | all 8 | 8 | P0 |
| 2 no stages | Uniform calls + calibrated stall (R7 "CU") | all 8 | 8 | P0 |
| 3 R7 stages | Stage-tilted calls + calibrated stall (R7 "CT") | all 8 | 8 | P0 |
| 3 R7 stages | Stage-gated look-less, one extra block (R7 "SF1") | all 8 | 8 | P1 |
| 3 R7 stages | Stage-gated wrist-only looks (R7 "SW") | π0.5 × 4 | 4 | P1 |
| 4 look less | Follow lottery: at each structurally supported cache anchor a hashed coin picks 0, 1 or 2 extra blind blocks (1/3 each among supported values); stage gate and state valve computed but not enforced | L10 × 4 (P0), Spatial × 4 (P1) | 8 | P0/P1 |
| 4 look half | Wrist-only at every look except the first/forced ones (no stage gate), 10-control cadence | π0.5 × 4 | 4 | P1 |
| 4 look half | Wrist-only looks every 5 controls (no blind block) | π0.5 × 4 | 4 | P1 |
| 5 identification | A + independent call coin p = .25 at each fresh anchor, no stall, no cooldown | 2 models × 2 suites × library 50 | 4 | P1 |
| 5 placebo | A with one extra look at the start (shifts every later look by 5 controls) | GR00T L10 × 2 | 2 | P2 |
| 5 cadence | A looking every 5 controls, both cameras | π0.5 L10-50, GR00T L10-50 | 2 | P2 |
| 5 oracle (diagnostic, privileged) | Oracle grasp-window calls: policy at every anchor while the end effector is inside the simulator-truth grasp window of an un-lifted, unsatisfied goal object (E3 5a) | L10-50, both models | 2 | P1 |
| 5 oracle (diagnostic, privileged) | Oracle-tight: only within 5 cm, at most 2 calls per goal object (E3 5b; aims at owner IR ≈ .08) | L10-50 and Spatial-50, both models | 4 | P1 |

**Total 70 arms / 35,000 episodes** (P0 32, P1 34, P2 4). Oracle arms read simulator truth through a separate,
labelled channel; they are upper bounds on what a perfect hard-stage segmentation could buy, never methods. Scheduling order P0 → P1 → P2; within a priority, GR00T
L10 (slowest) first. Deferred shadow augmentation runs P0 arms first.

## 3. Admission before the full campaign (smoke, non-test B-val inits)

1. Replay parity (CPU): request tapes replayed through each method family with debug off / on give byte-identical
   responses, method histories and counters.
2. Closed-loop parity (coordinator): GR00T A, 10 non-test episodes debug off vs on: identical issued actions.
3. Completeness: every accepted smoke episode has server + client + receipt records joined with zero gaps; images
   decode; per-control physics present; decision ids match on both sides.
4. Capacity: measured MB/episode p50/p95 per model × suite; forecast ≤ 1.6 TB total; receiver keeps up at 4 lanes.
5. Deferred shadow: π0.5 and GR00T shadow chunks on smoke data; throughput measured; noise-floor sample present.
