# R8 report — debug-mode profile system and a full real-trajectory collection

Written 2026-10-01 by the coordinator. R8 is a data-collection round (owner directive, `SELECTION.md` §0): build the
profile tools into the system's own debug mode, collect real closed-loop trajectories on the full 500-pair test set,
and let the explorers judge whether the data answer the stage-segmentation questions. No method is claimed here.

**Plain names used below.** *Pure policy* = call the policy every 10 controls (IR .5). *Pure cache* = retrieve from
the demonstration library at every look, execute the whole 10-control chunk, never call the policy. *Uniform calls* =
pure cache plus a random policy call at each look with a fixed budget, plus the calibrated stall trigger. *Stage-tilted
calls* = R7's version that moves calls toward gripper changes. *Independent call coin* = a call with probability .25
at every look, no stall rule. *Look less (gated)* = R7's one extra blind block when the stage gate and state valve
allow it. *Follow lottery* = 0, 1 or 2 extra blind blocks drawn at random (gate computed, not enforced). *Wrist in easy
stages* = R7's look-half. *Wrist every 10 / every 5* = wrist camera only at every look, at 10- or 5-control cadence.
*Look every 5* = pure cache looking every 5 controls. *Placebo* = pure cache with every look shifted by 5 controls.
*Oracle window (wide / tight)* = privileged diagnostic: call the policy only when simulator truth says the gripper is
inside the grasp window of an un-lifted goal object (tight: within 5 cm, at most 2 calls per object).

## 1. What was built

| Part | Where | What it does |
|---|---|---|
| Data contract | `exp/offline_search/debug/SCHEMA.md` (`osdebug.v1`) | Identity, file layout, every field and its status semantics |
| Server observer | `debug/server/`, hooks in `closed_loop/plugin.py`, `blind.py` | Copies exact wire images/state/prompt, keys, all chunks, the 16 neighbours, every method decision and reason, randomization, dispatch counts and timing for every request; per-process block files; never touches the live path |
| Client capture + transport | `debug/client/`, `debug/transport/`, `debug/ops/` | Every control: action, robot state, all movable object poses, named contacts with forces, goal predicates, actuator substeps; sampled simulator snapshots; streamed to a receiver on weilandserver with receipts and spill fallback; `chain_debug.sh` campaign chain |
| Reader / validation / capacity / catalogs | `debug/reader.py`, `validate.py`, `capacity.py`, `catalog.py` | One reader API for every tool; per-arm PASS/INCOMPLETE; byte accounting; library row catalogs with R7 stage labels |
| Deferred shadows | `debug/aug/` | From stored observations: a policy chunk at every decision, 3 extra draws on a 1/32 sample, a full-look retrieval at every decision, wrist-only / third-only retrievals for π0.5 |
| Decision-level profile tools | `debug/tools/decision/` | provenance, divergence, follow_vs_look, camera_shadow, stage_ledger, call_value, exposure_hazard, churn, trigger_vs_onset |
| Physical profile tools | `debug/tools/physical/` | selfcheck, forensics (failure labels + onsets + simulator-truth stages), grasp_audit, drop_audit, paired_diverge, twin_divergence, blind_drift, segmentation_bench, episode_card, arm_rollup |
| R8 arm methods and specs | `rounds/r08/methods/`, `rounds/r08/ops/` | Follow lottery, wrist every look, look every 5, placebo, independent call coin, pure policy with 5-control decisions, oracle window calls; 70 arm specs; fits; augmentation fit config; arm table |

Process: five explorers (3 astra, 2 Opus) specified tools and fields (`ideation/E*/PROPOSAL.md`); six sol coders built
them (`debug/handbacks/`, `methods/HANDBACK.md`); an Opus reviewer audited all code (`REVIEW_1.md`: 1 blocker, 7 major,
9 minor); one fix round by the same coders; a performance round for the deferred shadows.

## 2. Validation

- **Debug on/off parity on the real GPU:** 11 method families (π0.5: pure cache, uniform calls, stage-tilted calls,
  look less, wrist in easy stages, pure policy; GR00T: the same minus wrist), 20 recorded episodes each, served actions
  byte-identical with debug off and on (496–521 decisions per family). Rechecked after the fix round.
- **Real-simulator capture parity:** the same action tape with client capture off and on gives byte-identical joint
  states (6 non-test episodes, both suites).
- **Closed-loop consistency:** GR00T pure-cache decisions with debug on agree with R6's debug-off runs as often as two
  debug-off runs agree with each other (E4 §3.8).
- **Latency:** after replacing a profiling hook, debug-on GR00T looks cost 62.0 → 63.4 ms (before the fix 87 ms).
- **Smoke:** 70 arm types × 20 non-test episodes all passed receipt and schema validation before the campaign.
- **Campaign audit (E5):** 70/70 arms pass capture validation; 35,000 accepted episodes, 1,592,477 decisions,
  8,253,989 controls, 202,448 snapshots; all pairs share reset-state hashes across arms (constant environment seed);
  25.5 M neighbour references resolve; all writers drained with zero errors.
- **Shadow agreement:** on 552 real GR00T policy calls, the deferred policy shadow differs from the live chunk by MSE
  .0020, below the .0082 between two independent policy draws. GR00T shadow looks reproduce the live keys to ≤5e-7
  (serial vision stage; the batched variant drifted 1.3 % and was rejected).

Incidents, all resolved: real-LIBERO environment unwrapping and robosuite structs (contacts/predicates failed in the
first smoke), wire actions stored as float32 (fixed to float64), a file-name collision between two coders' fixes,
transient `tether pull` failures (retry added), GPU overcommit between collection servers and shadow jobs (gates and
lane counts adjusted), and one arm's shadows written by a half-edited code version during the performance round
(`r8_groot_l10_500_CU`, quarantined as `aug__mixed_610613bf` and regenerated).

## 3. Data

Run root `/home/weiland/trace_runs/os_closed_loop/r08_main`: **70 arms × 500 official test pairs = 35,000 episodes**,
259 GB (images 81 %); smoke run `r08_smoke` (70 × 20 non-test episodes). Storage per episode: π0.5 L10 9.3 MB,
π0.5 Spatial 3.5, GR00T L10 11.8, GR00T Spatial 4.4. `/home` keeps ≈1.9 TB free; nothing was moved to `/archive`.

Success rate @ owner IR (pooled over decisions; π0.5 .152 look + .848 call, GR00T .148/.852, wrist look .0646).
Table `analysis/arm_table.json` (`rounds/r08/ops/arm_table.py`). Library 50 / 500.

| Arm | π0.5 L10 | π0.5 Spatial | GR00T L10 | GR00T Spatial |
|---|---|---|---|---|
| Pure policy | .908 @ .504 | .988 @ .511 | .898 @ .504 | .940 @ .511 |
| Pure cache | .718 @ .076 / .848 @ .077 | .824 @ .078 / .974 @ .078 | .610 @ .074 / .814 @ .074 | .868 @ .075 / .964 @ .076 |
| Uniform calls | .854 @ .301 / .906 @ .188 | .940 @ .303 / .980 @ .166 | .796 @ .303 / .846 @ .187 | .930 @ .301 / .952 @ .177 |
| Stage-tilted calls | .876 @ .309 / .880 @ .188 | .948 @ .292 / .984 @ .165 | .810 @ .301 / .844 @ .191 | .936 @ .307 / .938 @ .179 |
| Independent call coin | .768 @ .187 | .898 @ .183 | .694 @ .181 | .912 @ .185 |
| Look less (gated) | .682 @ .065 / .830 @ .069 | .794 @ .066 / .962 @ .061 | .574 @ .065 / .830 @ .068 | .832 @ .063 / .954 @ .064 |
| Follow lottery | .634 @ .058 / .780 @ .054 | .792 @ .061 / .980 @ .056 | .574 @ .055 / .768 @ .053 | .830 @ .059 / .934 @ .055 |
| Wrist in easy stages | .700 @ .063 / .846 @ .067 | .826 @ .064 / .976 @ .058 | — | — |
| Wrist every 10 | .638 @ .034 / .786 @ .034 | .800 @ .036 / .976 @ .037 | — | — |
| Wrist every 5 | .590 @ .066 / .726 @ .066 | .820 @ .068 / .956 @ .069 | — | — |
| Look every 5 | .642 @ .152 (50) | — | .548 @ .148 (50) | — |
| Placebo | — | — | .622 @ .076 / .828 @ .076 | — |
| Oracle window, wide | .794 @ .141 (50) | — | .712 @ .147 (50) | — |
| Oracle window, tight | .744 @ .092 (50) | .832 @ .102 (50) | .658 @ .088 (50) | .878 @ .098 (50) |

Pure-cache and look-less points reproduce R6/R7 within run-to-run noise (e.g. R7 look-less π0.5 L10-50 .684 vs .682).

**Standard profile pack (all 70 arms, all accepted pairs, descriptive):** `r08_main/profile/{forensics,provenance,
stage_ledger}/<arm>/` (18 GB), failure labels with one frozen per-task calibration
(`profile/forensics_calibration.json`, fitted on pure-policy successes of inits 0–29), so labels are comparable across
arms. Built by `tmp/r8/profile_pack.sh`; 70/70 succeeded. Analyses that choose a segmentation or threshold must still
filter to inits 0–29.

## 4. What the explorers found (details in `ideation/E*/DATA_ANALYSIS.md`)

Segmentation choices used discovery inits 0–29 only; inits 30–49 of every task stay a locked holdout.

1. **Is stage segmentation the bottleneck? Not the grasp window alone (E3).** Calling the policy exactly when the
   gripper is in the grasp window recovers only 35–40 % of the pure-cache → pure-policy gap on LIBERO-10, at about
   twice the cost (.14–.15); the tight version at ≈ .09 recovers 14–17 % on LIBERO-10 and ≈ 0 on Spatial. Uniform calls
   at .30 recover 65–94 %. The cache's extra failures are spread over the approach before the window, the grasp, and
   the place/release/fixture actions. Pure policy rescues almost every cache grasp miss on the same reset.
2. **Where the cache fails (E3, E4).** Missed grasps dominate (about half of cache failures on LIBERO-10), then
   misplacement/undone/fixture; drops are rim slips with the gripper still closed, not premature opening. Object-frame
   pre-grasp pose error predicts a failed lift (AUROC .67–.81) but needs object poses the robot does not have.
3. **Segmentation quality (E1).** R7's gripper split matches physical stage boundaries no better than an equal-count
   uniform split (F1 .18). Motion change points beat uniform in 36/36 arms (F1 .36). A trajectory-waypoint risk score
   flags upcoming cache failures better than R7's stage in 6/8 pure-cache cells, but elapsed time is an equally strong
   baseline; no hard/easy rule is validated.
4. **Where a call is worth more (E2).** Every one of 127,341 call coins replays exactly. Stage-tilted vs uniform calls:
   +0.08 pp (discovery). A placement-stage call effect (+5.4 pp in stage-tilted 50-demo cells) is a hypothesis only
   (0/150 simultaneous intervals exclude zero). Blind continuation drifts further from a fresh look with age, mostly in
   mixed (gripper-transition) kernels.
5. **Look less / look half (E4).** At IR ≈ .06–.07 the only lever without a detectable loss is *wrist in easy stages*.
   Looking every 5 controls is worse than every 10 for both models (−6 to −8 pp): re-retrieving stitches more
   demonstrations and toggles the gripper more, so the 10-control commitment is a quality feature. Ungated following
   loses 3–8 pp in 7/8 cells; gated following is safe only on 500-demo libraries. A 5-control placebo causes as many
   outcome flips as any lever; the levers' harm is a small net bias, mostly extra missed grasps.
6. **Data and tools (E5).** Capture is complete and lossless; scientific capability is partial (next section).

## 5. Gaps to fix before R9 (from E1–E5 and the review)

1. No *pre-decision* stage is logged (`stage_pre`); stage-conditioned causal tools fall back to `unknown`. Add an
   explicit, prefix-only stage record with its source and fit hash.
2. Model manifests lack gripper channel, threshold and action scale, so action-distance metrics mix the gripper in.
3. Forensic labels are kinematic heuristics (`truth_validated=false`): grasp-miss onset is window entry, fixture
   destinations are unresolved (`unknown_release`), calibration must be frozen per suite. A human double-annotation
   audit is needed before labels become ground truth.
4. Retrieval scores are saved for the top 10 of 16 neighbours; server and catalog library hashes use different domains.
5. Readers write derived caches under the capture root by default; tools have no `--split discovery` enforcement;
   reports do not bind tool/config/augmentation-generation hashes; `AUG_DONE` is not a validated generation.
6. `blind_drift` does not finish at campaign scale; `paired_diverge` needs a frozen envelope and a replicate arm;
   `exposure_hazard` collapses stages and uses only first entries.
7. Library physical backfill: only the two π0.5 500-demo libraries have an exact replay source; not run.
8. Six stage-tilted GR00T L10-500 attempts raised "stage masses/occupancies must be finite probabilities" and were
   retried; the accepted population is complete but the serving fault should be fixed.

## 6. Deferred shadows and full validation (final, 10-01 19:4x)

- **Coverage.** Deferred augmentation (policy shadow at every decision, independent policy draws on a 1/32
  sample, shadow look, π0.5 camera shadow) finished for **70/70 arms** at 10-01 19:36.
  - Lanes `r8aug_A*` (`tmp/r8/r8_aug_lane.sh`) ran at ~20–26 decisions/s with three π0.5-scale jobs at once.
  - Three OOM failures (four concurrent jobs or an admission race); each was re-run to completion from its
    resumable parts.
  - One arm generated with mid-edit code (`r8_groot_l10_500_CU`) was regenerated; the contaminated parts are kept
    apart as `aug__mixed_610613bf`.
  - GR00T ran in serial stage mode, reproducing live keys to ≤5e-7; π0.5 ran in batches of 32, as served live.
- **Full validation.** `debug.validate` (capture + augmentation completeness, all accepted pairs) is **PASS for
  every arm**. Per-arm reports are in `r08_main/validation_full/<arm>.json` (status / capture / augmentation all
  PASS, 500 accepted episodes, no missing kinds).
- **Agreement check (`aug.validate`, descriptive).**
  - 30 arms have live policy calls to compare.
  - The ratio "live-call vs shadow MSE / independent-draw vs independent-draw MSE" has median 1.11 (p10 .66,
    p90 1.91). Shadows are as close to the live call as a second independent draw is (MISS noise is unseeded).
  - All six stage-tilted (CT) arms sit at 1.25–2.15 while uniform-call (CU), independent-coin (IP) and pure-policy
    (P10) arms sit at ≈.8–1.3.
  - **Unverified hypothesis:** CT places its calls in transition stages, where the policy is more variable, while the
    independent-draw reference is sampled from all decisions. The R9 check is a population-matched reference
    (draws at the same decisions).
  - The 38 arms without live calls (A, look-less, wrist, follow-lottery, etc.) have nothing to compare, by
    construction.
- Standard profile pack (§3) and the full catalog are complete for all 70 arms.

## 7. Supplementary ablations (owner 10-01 12:2x; standard mode, no debug capture)

Owner decisions of 09-29 (complete the B-layer guard ablations; add a CLIP retrieval-key ablation) run on the new
topology: policy servers on h100 (CUDA MPS from cell 2 on) with LIBERO workers on timan108 (fleet A, run root
`os_closed_loop/r08_abl`) and timan107 (fleet B, `os_closed_loop/r08_abl_t107`). 44 arms × 500 official pairs.
Every cell has its own same-topology controls (B and A re-runs), because a review showed the new hosts diverge
from the old reference runs from step 1 (render / GPU numerics); the old three R6 replicates are a cross-topology
reference only (controls land within 0–2.4 pp of them). Full table with paired exact McNemar:
`abl/RESULTS.md` (`python -m exp.offline_search.rounds.r08.abl.analyze`); arm specs `abl/ARMS.md`, CLIP design
`abl/clip/CLIP.md`; runner `closed_loop/ops/h100/` (RUNBOOK, HANDBACK_SB).

**Guards (B = A + four guard-triggered policy rescues).**
- Removing *stuck*, *terminal* or *overtime* alone changes SR by −1.4…+1.6 pp in all four 500-demo cells (all n.s.).
- Removing *no_progress* costs −5.0 pp (π0.5 L10-500, p=.002) and −4.4 pp (GR00T L10-500, p=.02), and leaves
  SR ≈ A (discordant pairs vs A only +5/−1 for GR00T): no_progress is where B's gain comes from.
- "A + only no_progress" recovers B on the sparse/long cells:
  - π0.5 L10-50: .832 vs B .834 (A .724);
  - π0.5 Sp-50: .892 vs B .892 (A .828), at lower IR (.128 vs .145);
  - GR00T L10-50: .716 vs B .730 (A .610);
  - GR00T L10-500: .870 vs B .858 (A .806);
  - π0.5 L10-500: .876 vs B .894 (A .826; −1.8 pp n.s.).
- Dense / Spatial cells:
  - π0.5 Sp-500: everything ≈ .98.
  - GR00T Sp-500: B .940 < A .956; removing no_progress is +2.0 pp vs B (p=.006).
  - GR00T Sp-50: only-no_progress −2.4 pp vs B (p=.04) and ≈ A. The other guards matter a little here.
  - These confirm R6's "no library-level do-not-call gate" gap.
- Implication for the method: drop stuck / terminal / overtime; keep no_progress plus a library-level gate that
  turns calls off on dense libraries (GR00T Spatial).

**CLIP key (A with the visual key replaced by CLIP ViT-B/32 global embedding → PCA-64 → same per-task metric),
π0.5 only.** The GR00T libraries store no raw RGB, so GR00T was not built.
- SR vs same-topology A: L10-50 .424 vs .724 (−30.0 pp), Sp-50 .556 vs .828 (−27.2), L10-500 .644 vs .826 (−18.2),
  Sp-500 .948 vs .978 (−3.0). All p ≤ .017.
- An independent read-only diagnosis found no implementation mismatch:
  - step-0 re-encoding reproduces the h100 online top-10 neighbours exactly (relative distance error 3e-4);
  - flip, resolution, camera order, colour and preprocessing are all identical;
  - there are no cross-task neighbours.
- Mechanism: one global vector cannot tell the phase within a task.
  - Library self-retrieval nearest/median distance ratio: .583 for CLIP vs .434 for A.
  - Consecutive looks jump back in the demo 16.4% of the time vs 8.6% for A.
  - SR drops in all 10 tasks.
- The 90-query offline check in CLIP.md understated the gap. On 4,126 logged queries: RMS10 .666 vs .585, and the
  top-1 neighbour agrees with A only 22% of the time.
- Conclusion: the policy's own spatially pooled tokens are necessary. A fairer CLIP variant (patch tokens with the
  same 4×4 pooling) was not built.
