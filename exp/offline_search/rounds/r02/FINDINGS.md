# R2 input: what R1 established, under the owner's new hard constraint (2026-09-26)

## ⛔ Owner ruling that frames R2
**Vision is mandatory.** Every method must use visual observation (the policy's own vision-encoder output: pooled keys
or tokens; the current system already computes them at stage 1 even on a HIT) in every regime — step 0, after a MISS,
after a HIT. State-only / action-continuity-only retrieval ("a blind person operating") is rejected as a solution; such
signals may only be auxiliary terms fused with a visual term. The primary exam is the **raw strength of the cache**:
the stale regime (cache cells: every decision after a HIT) — plus step 0; the fresh regime (after a MISS) must also be
reported. Every conclusion must state the library scale (episodes, entries, bytes/entry, total volume) vs the deployed
library (pkl on disk π0.5 431/1103 MB, GR00T 429/1068 MB; 262 KB key per entry), at BOTH current size (~50 episodes)
and the 10× library (500 episodes) until the owner rules which is deployable.

Full R1 report: `rounds/r01/ANALYSIS.md`; ledger: protocol §10. Cells order below: π0.5-sp / π0.5-l10 / GR00T-sp /
GR00T-l10. err = executed-segment RMS in σ units vs the teacher's `a_inf` (lower is better).

## Closed-loop reference (pure cache, B0, current library, A-pool 500 inits; from trace_dual)
Success rate: pure cache (every decision a HIT) .668 / .452 / .736 / .468; pure inference .986 / .844 / .940 / .870.

## Stale regime (cache cells) — the primary exam
| method | library | err | AURC | notes |
|---|---|---|---|---|
| B0 (current fused, top-1) | current (~50 ep; 262 KB/entry) | .643 .597 .621 .644 | .406 .390 .408 .437 | deployed |
| M4: B0 ranking + mean of top-5 | current | .589 .540 .548 .568 | .413 .380 .382 .409 | synthesis alone −.05…−.07 |
| M9c: two-camera pooled z-sum (+state), top-1 | current, raw keys | .635 .584 .625 .620 | .444 .394 .419 .459 | |
| M8: B0 formula over ALL candidates + mean-5 | 10× (500 ep; 262 KB/entry → 2.9–7.8 GB) | **.526 .480 .453 .488** | .333 .322 .323 .346 | best; 69 ms/query; not deployable as is |
| M8 top-1 | 10× | .618 .554 .527 .569 | .396 .378 .389 .399 | mean-5 is worth −.07…−.09 here |
| M9c pca128 z-sum + state, med3 | 10× (1056 B/entry) | .562 .495 .496 .517 | .389 .361 .382 .409 | median synthesis costs .03–.04 vs mean |
| M7 cascade: state/continuity prefilter to 16 (or 64), PCA-32 two-cam re-rank | 10× (428 B/entry) | .590 .528 .490 .526 (M=64: .573 .509 .510 .533) | .343 | the narrow state prefilter throws away the vision gain |
| (vision-free reference, not allowed alone) M2 state window mean-8 | 10× (96 B/entry) | .598 .528 .445 .499 | .368 .349 .309 .354 | on GR00T state ≈ vision; on π0.5 vision −.06…−.08 |
| oracle (best candidate) | current / 10× | .394 .277 .361 .261 / .282 .190 .229 .179 | | |

## Step 0 (500 decisions/cell; robot state ≈ constant across inits → only vision can tell layouts apart)
B0 .317 .268 .392 .248; M8 (10×, mean-5) .219 .181 .274 .197; M7 raw both .244 .177 .294 .189; state-only mean-5
.247 .198 .271 .198; oracle (current) .212 .140 .233 .132. The M9 probe: privileged true object layout is NOT better
than the policy's own vision tokens; pooled 4×4 keys of both cameras, z-summed, are the best training-free representation
at steps 0–2; aligning candidates to the same step index helps GR00T-l10.

## Fresh regime (inf cells, after a MISS)
The previous chunk's unexecuted tail predicts the teacher strongly: tail itself err .249 .282 .211 .239; vision methods:
M8 .329 .302 .362 .347, B0 .492 .467 .524 .535, M5 (B0's top-10 re-ranked by tail continuity, current lib)
.380 .363 .392 .395. Vision re-rank on top of continuity: neutral (π0.5) to slightly harmful (GR00T). Under the new
constraint continuity may still be fused with a visual term (the tail was produced by the policy looking at the image).

## Representation / compression facts (all paired, CIs in ANALYSIS.md)
- Raw 32768-d vs PCA-32 per camera: ≤ .008 err in every cell/regime; PCA-128 vs PCA-32 ≤ .006 → useful visual info
  survives ~1000× compression (PCA-32 two cameras = 256 B/entry; basis 8.4 MB/suite).
- Two cameras z-summed ≥ wrist-only > agentview-only; task-centring ≤ .005; full-token cosine, MaxSim, Chamfer,
  top-variance tokens: no better (MaxSim/Chamfer ~256× cost); raw pixels worse than tokens.
- Within-task cosine of pooled keys is saturated (π0.5 p50 .97–.98); rank correlation with action error is weak
  (.1–.5). Fused-score tanh-z normalization changes are ≤ .012 (not the lever).
- Vision must score the whole candidate set (or ≥ 128); a narrow state prefilter (16) forfeits the vision gain on π0.5.

## Synthesis / confidence facts
- Mean of top-k beats top-1 everywhere and beats per-step median (+.026…+.033 worse at k=5 stale); k 5→8 small gain.
- Confidence: in the stale regime observation-side signals rank best (B0 score for M8: AURC .338; z(B0)+z(−disp5)
  +z(−cont): .332); top1–top2 margin is anti-informative; continuity terms after a HIT are spuriously high.
- Drift: when the robot-state window is > 3× the library 1-NN median away (19–35% of stale decisions), err is .7–1.0
  while B0's confidence stays ~.98 (saturated vision on a stuck arm) — a visual/state drift detector is needed.
- 58–80% of stale-regime error comes from episodes B0 failed (drifted states); on episodes B0 completed stale err is
  .33–.35.

## Costs
B0: 262 KB/entry, 1.3–8.3 ms/query (current lib); M8 on 10×: 69 ms/query (memory-bandwidth bound).
PCA-32 two-camera: 256 B/entry (+ basis 8.4 MB/suite), ~1 ms/query on 10×.
