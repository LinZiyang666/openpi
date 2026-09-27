# R3 input: what R2 established (offline full runs + first closed-loop arms), 2026-09-26 23:0x CDT

Rules still in force (protocol §9): vision mandatory in every regime; report the three-layer effect decomposition
(synthesis / method at fixed library / library); report library scale at current (~50 episodes) and 10× (500 episodes);
closed loop is the real exam. Cells: π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10. err = executed-segment error vs the
teacher (lower better). Full R2 material: `rounds/r02/` (SELECTION, NOTES, code), results `results/r02/`, scoreboard.

## Closed loop (pure cache = every decision a HIT, A-pool 500 inits, current 50-episode library)
| arm | π0.5-sp SR | π0.5-l10 SR |
|---|---|---|
| CL0 B0 native (deployed) | .668 | .440 |
| CL1 B0 ranking + mean of top-5 (synthesis only) | .764 | .428 |
| CL2 AWM (action-whitened joint metric, fitted on the same 50 episodes, kref 5, kernel-16) | **.800** | **.630** |
| history S3 (5 traj/task, B0-style top-1) | .688 | .456 |
| history S6 (~44 traj/task ≈ our 10× library, B0-style top-1) | .810 | .516 |
GR00T arms, CL3 (AWM + stuck recovery) and the whole 500-episode group are running (results land in
`/home/weiland/trace_runs/os_closed_loop/r02_g50|r02_g500/summary.json` and `runs/chain.log`).
Server-side per-decision logs (inputs digests, picks, extras, latency) for every closed-loop arm:
`/home/weiland/trace_runs/os_closed_loop/r02_g50/runs/<arm>/server_<port>/decisions_<arm>_<port>.jsonl`; client
journals/per_step pulled under `runs/<arm>/` — these are the FIRST cache-arm traces driven by non-B0 selectors.

**Offline→closed-loop mapping is not proportional.** Synthesis-only (CL1) cut offline stale err by .054 on π0.5-sp and
.057 on π0.5-l10, but gave +9.6 pp SR on spatial and −1.2 pp on l10. AWM's ranking cut offline stale err only by
≈0/−.03 beyond synthesis, yet gave +3.6 pp (sp) and +20 pp (l10) SR. Ranking quality matters much more on long-horizon
tasks than the offline mean err suggests; AWM episodes also finish in fewer decisions (π0.5-sp 25.4 dec/ep).
AWM agrees with the native B0 pick on only 38–45% of closed-loop decisions.

## Offline (full 8 cells; stale = cache cells, fresh = inf cells)
| method | library / fit | stale err | stale AURC | fresh err |
|---|---|---|---|---|
| B0 | current | .643 .597 .621 .644 | .406 .390 .408 .437 | .492 .467 .524 .535 |
| M4 (B0 rank + mean-5) | current | .589 .540 .548 .568 | .413 .380 .382 .409 | .436 .414 .458 .469 |
| AWM kr5 | current / current | .579 .513 .511 .513 | .367 .334 .352 .370 | .337 .317 .344 .331 |
| V4 PCA-cosine z-sum | current / current | .574 .514 .562 .539 | .351 .332 .348 .376 | .419 .392 .452 .455 |
| AWM | 10× / 10× | .499 .432 .431 .442 | .293 .271 .294 .318 | .257 .273 .254 .267 |
| AWM vision-only | 10× / 10× | .492 .429 .438 .448 | .289 .271 .297 .327 | ≈ AWM |
| V4 per-model state weight | 10× / 10× | .513 .450 .435 .465 | .302 .281 .294 .319 | .311 .275 .353 .344 |
| V5 fresh fusion λ1 | 10× | = V4 | | .251 .271 .246 .262 |
| T2 MLKR metric (trained) | 10× / 10× | .494 .423 .416 .445 | .282 .260 .288 .313 | ≈ AWM |
| M8 (B0 formula over all 10× cands + mean-5) | 10× | .526 .480 .453 .488 | .333 .322 .323 .346 | .329 .302 .362 .347 |
- V7 predicted-err confidence (on a vision z-sum base): stale AURC 10× .300 .288 .293 .319 vs base .356 .335 .347 .377;
  one threshold works across regimes (pooled AURC below regime-wise rank normalization).
- V6 stuck recovery: offline blend effect tiny (≤ .005); recovery itself only measurable closed loop (CL3 pending).
- AWM fitted on 50 episodes loses ~.03 stale err vs fitting on 500 ("borrowed" fit); the metric needs many pairs.
- Insurance (norm-preserving mean + gripper hysteresis) costs +.02–.035 offline err; not yet tested closed loop.
- Library scale: AWM 580 B/entry (+~19 MB PCA/whitening per suite): current 21–27 MB, 10× 47–117 MB fit pickles;
  deployed pkl 431–1103 MB (262 KB/entry); M4/B0 fit pickles 0.27–0.70 GB at current size.

## Open questions R3 should attack
1. Why does ranking matter so much more in closed loop on l10? Mine the closed-loop decision logs (AWM vs B0 arms on
   the same inits: where do B0 episodes fail and AWM succeed — phase, task, gripper, stuck?).
2. The deployable constraint may be the 50-episode library: can AWM's fit be improved without more episodes
   (augmented pairs, temporal neighbours, cross-task sharing of the vision whitening, shrinkage toward a big-library
   prior fitted once offline)?
3. A real cache is mixed HIT/MISS: with a calibrated confidence (V7-style on AWM), what SR do we get at a given hit rate
   (inference ratio)? This needs the full model on the server (MISS decisions run stage 2/3) — allowed for R3.
4. Stuck/deadlock recovery and gripper chatter in closed loop (CL3 results will tell).
