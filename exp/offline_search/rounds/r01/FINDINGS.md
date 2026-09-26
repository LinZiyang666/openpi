# R0 findings (input to round 1 ideation) — 2026-09-26

All numbers: full store, 188,844 decisions, 8 cells (model × suite × arm; `inf` = on-policy states,
`cache` = states drifted by the online cache method). err = RMS over executed segment [:5,:7] in units of
per-dim library action std (see harness README). Order in lists: π0.5-sp-inf / π0.5-sp-cache / π0.5-l10-inf /
π0.5-l10-cache / GR00T-sp-inf / GR00T-sp-cache / GR00T-l10-inf / GR00T-l10-cache. Raw: `exp/offline_search/results/r00/`,
logs in `results/_logs/r00_*.log`, profile reports in `<store>/profile_cache/reports/`.

## Store / infrastructure
- Store root for runs: `/dev/shm/offline_search_store` (hot arrays pinned in RAM, 70 GB; tok/ dirs symlink to the SSD
  store `/home/weiland/trace_runs/offline_search_store`). Use `--root /dev/shm/offline_search_store`.
- Libraries: `current` (online libs: 1018 / 2640 / 1063 / 2645 entries, ~50 successful trajectories, 5 per task);
  `bpool_all` (GR00T: 500 episodes/suite incl. failures, 11,751 / 29,631 rows; π0.5: ≈ current, April batch);
  `bpool_cs` (π0.5 only: 500 episodes/suite, Aug 2026 batch, 10,909 / 29,472 rows, incl. failures).
- Gates all PASS: B0 reproduces the recorded online top-1 on 99.96–100% of decisions (all disagreements are
  near-ties, score gap ≤ 3.1e-5); oracle ≤ B0; random ≫ B0; bit-deterministic; current lib bit-exact vs pkl.

## Teacher noise floor (same observation, fresh noise; err units as above)
- π0.5 is stochastic: err p50 0.12 (p90 0.36–0.40), gripper flip ≈ 3%. Early 0.10, mid 0.12–0.14, late 0.14–0.16.
- GR00T is nearly deterministic: err p50 0.03–0.05 (rel-L2 ≈ 2%), gripper flip ≈ 1%.
- ⇒ `indist` (err ≤ floor median) is ≈ 0 for every retrieval method on GR00T; use err / regret / AURC there.

## Reference methods (err mean; lower is better)
| method | values |
|---|---|
| B0 current (grid weights, zscore+tanh, top-1) | .492 .643 .467 .597 .524 .621 .535 .644 |
| B1 LDA weights | .493 .638 .470 .600 .523 .620 .533 .636 (≈ B0 everywhere) |
| B4 robot_state only | .523 .661 .529 .652 .529 **.584** .552 **.606** (beats B0 on GR00T cache cells) |
| B4 vision_1 (wrist) only | .509 .657 .487 .605 .570 .661 .596 .690 |
| B4 vision_0 (agentview) only | .609 .688 .602 .696 .615 .649 .650 .690 (worst field everywhere) |
| B3 oracle (best in task, current lib) | .308 .394 .239 .277 .326 .361 .265 .261 |
| B2 random in task | ≈ 1.19–1.31 |
- B0 regret (err − oracle): .18 .25 .23 .32 .20 .26 .27 .38; B0 = 3.2–4.1× π0.5 floor, 10–19× GR00T floor.
- B0 AURC .38 .41 .37 .39 .43 .41 .43 .44 vs B3 (oracle picks ordered by true err) .18 .21 .14 .16 .20 .21 .16 .16; B0 risk@30% coverage
  .35 .33 .34 .31 .39 .36 .39 .36 — the fused score is a weak confidence signal.
- Gripper mismatch (executed 5 steps): B0 .04 .19 .07 .17 .06 .14 .09 .23 vs oracle .01 .06 .01 .02 .01 .03 .01 .01.
- Phase error: oracle picks have LARGER phase error than B0 (e.g. .115 vs .073 π0.5-l10-inf) — phase alignment is
  not the objective; action-relevant state is.
- Cost of B0: 262 KB per library entry (two 32768-d f32 keys + state), 1.3–8.3 ms/query single-thread (l10
  larger), latency ∝ L^1.28 (98% of time in the two 32768-d cosines).

## Coverage (library-side ceiling; `coverage` tool)
- Candidates per task: ~110 (spatial) / ~265 (l10) in `current`.
- Queries with ≥1 candidate within teacher noise: π0.5 .11 .09 .19 .12, GR00T ≈ 0 — whole-chunk reuse is
  intrinsically far from the teacher.
- Library size is a big lever: 10× library (`bpool_cs` π0.5, `bpool_all` GR00T) cuts oracle err ≈ 40%:
  .308→.186, .394→.282, .239→.140, .277→.190 (π0.5); .326→.204, .361→.229, .265→.158, .261→.179 (GR00T).
- B0's pick has rank p50 3–17 among task candidates; cache-arm late steps degrade to rank p50 32–49 and regret
  .42–.50 (drifted states are where retrieval breaks).

## Representation diagnostics (`repr` on library current; 2000 queries/cell)
- Within-task cosine is saturated: π0.5 pooled keys p50 .97–.98 (v0: 14–22% of pairs > .99); GR00T wider
  (v0 .93–.95, v1 .85). Query-side spread of sims ~.025 (π0.5).
- Similarity ↔ action-error rank correlation is weak: Spearman(sim, −err) key_v0 .10–.38, key_v1 .22–.48,
  rs .27–.53. AUROC(top-10% candidates) .68–.90.
- Oracle rank under a single field p50 4–15; oracle is in the top-10 of a single field for 33–80% of queries
  ⇒ a better re-ranker over a short list has room.
- tokmean (mean over 256 tokens) is worse than 4×4 pooling (more saturated, lower rho).
- PCA participation ratio of pooled keys is small (≈ 7–54) ⇒ 32768-d keys live in a low-dim subspace.

## Failure modes (`breakdown`, `timeline` on B0)
- Gripper transitions (teacher gripper flips within/at the chunk): 7–18% of decisions, err .62–.95 vs .45–.63
  steady, gripper mismatch 32–41% there — and B0's confidence is barely lower on them (.91–.99 vs .95–.99).
- Temporal behaviour: B0 moves along a library trajectory ("track": picks prev pick's next entry) 38–63% of the time
  with the lowest err (.39–.46); "stay" (re-picks the same entry) 11–35% with the highest err (.62–1.13, worst in
  cache arms); "switch" 11–35%. The oracle switches 45–60% of the time, stays 10–34%, run length ≈ 2.
- Cache arm vs inf arm: err +0.07–0.15, gripper mismatch 3–5×, stay rate doubles — drift + stickiness.
