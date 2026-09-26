# Ideation agent B (diagnosis-driven refinement) — round 1 proposals

All numbers below are my own measurements (CPU, read-only) on the full store unless marked "subsample". Cell order everywhere: π0.5-sp-inf / π0.5-sp-cache / π0.5-l10-inf / π0.5-l10-cache / GR00T-sp-inf / GR00T-sp-cache / GR00T-l10-inf / GR00T-l10-cache. B0 err = .492 .643 .467 .597 .524 .621 .535 .644; B0 AURC = .382 .406 .367 .390 .425 .408 .430 .437. Scripts + raw json/txt: `/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d{1,1b,2,3,4,5,6}_*.py|txt|json` (D1/D1b/D2/D5/D6 all 188,844 decisions; D3 2,500 and D4 1,200 queries per cell, seed 0).

## Headline diagnosis (what the leads turned into)

1. **The strongest untapped online signal is action-chunk continuity, not a better key.** `q.prev_a_exec[5:10, :7]` (the unexecuted tail of the previous chunk, which the policy or the cache returned in full) forecasts exactly the window the current chunk must execute. Scoring every task candidate by `cont = RMS_σ(cand[0:5,:7] − prev_tail)` and nothing else gives err .371 .607 .350 .589 .372 .570 .356 .620 on the current library (D2), i.e. −24…−34 % on inf arms and −1…−8 % on cache arms, at ~0 cost. A crucial fact for reading the two arms: in the trace `prev_a_exec` is the teacher chunk on 100 % of inf-arm decisions and a library chunk on 100 % of cache-arm decisions (D1b: hit_rate_prev = 0.0 vs 1.0). So inf = "after a MISS", cache = "after a HIT"; a deployed system sits between, weighted by its miss rate. The offline harness feeds the *trace's* executed chunk back, never the method's own pick, so stickiness of a continuity-driven chain cannot be measured offline → closed-loop check is mandatory (protocol §3.3 already demands it).
2. **Lead 1 (μ/σ saturation) is mechanically real but not the bottleneck.** B0's top-10 candidates sit at z-scores with median 2.0–3.3 on both vision fields (|z|>2 for 50–98 % of them; tanh′(2.2)=.06, tanh′(3.3)=.005), rs z median 1.6–2.2. Yet per-task μ/σ, linear z (no squash), per-task linear z, product and min fusion all change err by ≤ .012 (D3, subsample): the sim↔err relation is weak (Spearman .1–.5) so de-saturating a weak signal buys nothing. Rejected below.
3. **Lead 2 (rs beats B0 on GR00T cache, v0 worst):** vision re-ranking of a good short list is as bad as B0 (D4: B0 fused score over an rs+cont-pruned top-16 of the current library = .488 .651 .440 .591 .521 .608 .496 .610, ≈ B0). Vision only pays on the *big* library after pruning (P4). rs alone on the 10× library already beats B0 on GR00T cache cells (.518 vs .621, .552 vs .644) and matches it on π0.5-l10-inf (.423 vs .467).
4. **Lead 3 (oracle in short list):** the task oracle is inside B0's own top-10 for 82/67/61/48/81/71/56/44 % of queries; the best-of-top-10 is .330 .468 .285 .396 .352 .423 .325 .401. Re-ranking B0's top-10 by continuity alone recovers most of the inf-arm gap: .381 .627 .364 .594 .391 .595 .391 .622 (D1). Medoid, rs-only re-rank and gripper-consistency filters recover nothing (rejected).
5. **Lead 4 (gripper transitions):** the fraction of B0's top-10 whose gripper flips vs the last executed gripper predicts a teacher transition with AUROC .91 .72 .92 .72 .91 .81 .88 .72. It is a poor *ranking* signal (filter net ≈ 0: steady −.005…−.03, transitions +.10…+.20) but a usable *veto tier* in the confidence (P3 variant).
6. **Lead 5 (track/stay):** forcing `next(prev pick)` on "stay" decisions: .494 .642 .479 .594 .527 .618 .551 .627 (≈ B0); "track if next(prev) is in B0's top-3": .495 .643 .472 .593 .524 .617 .542 .641. Dead; rejected.
7. **Lead 6 (confidence):** on B0's own picks, −continuity of the top-1 alone reaches AURC .301 .288 .326 .325 on inf arms (B0 .382 .367 .425 .430; attainable .269 .243 .302 .286) but is worse than B0 after a HIT (.438 .464 .412 .493). A rank-average of (B0 score, top-5 action agreement, −continuity) beats B0 on all 8 cells: AURC .313 .390 .292 .384 .347 .376 .341 .423; risk@30 % .260 .304 .238 .301 .297 .314 .281 .325 vs B0 .351 .325 .341 .311 .390 .357 .394 .357. **B0's top-1/top-2 margin is anti-informative** (AURC .56–.84, worse than random .49–.64): near-ties are neighbours on the same library trajectory (a dense, reliable region), a large margin means an isolated candidate.
8. **Lead 7 (library):** with a key-free score the 10× library costs nothing: rs+cont on `bpool_cs`/`bpool_all` gives err .293 .604 .301 .589 .284 .537 .289 .623 (top-1) and the fused top-10 still contains a candidate at .217 .453 .208 .450 .228 .357 .224 .477 (big-library oracle .186 .282 .140 .190 .204 .229 .158 .179). Bytes/entry 32 B (rs) instead of 262 KB.
9. **Top-k per-step median synthesis** (dims 0–5 median, gripper majority over top-k) is a universal −4…−8 % on all 8 cells, even on B0's unchanged top-5: .452 .621 .428 .565 .478 .574 .487 .597, grip_mis equal or lower. Included as a component of P1/P2/P4.

---

## P1 `chunk_continuity_search` — key-free retrieval by action-chunk overlap + robot state over the 10× library (T0)

**Pitch.** Replace the two 32768-d cosines with a 35-d GEMM against the previous chunk's tail; use robot_state as the tie-breaker/step-0 fallback; return the per-step median of the top-3.

**Hypothesis.** The policy is temporally consistent: its chunk at t−1 already predicts the window executed at t (π0.5 floor p50 .12, GR00T .03–.05, whereas B0 is at .47–.64). Pooled vision keys are saturated and only weakly action-relevant (rho .1–.5), so the candidate whose *head* continues the executed *tail* is a far better proxy for "same action-relevant state" than image similarity. Because the score is 35-d + 8-d, the library can be 10× larger (oracle −40 %) at negligible latency, and the median over the top-3 removes single-neighbour noise (multi-modality is handled by the gripper majority + per-dim median, not a mean).

**Algorithm (against the harness API).**
- `fit(lib, ctx)`: `big = ctx.open_library("bpool_cs" if ctx.model=="pi05" else "bpool_all")`; for π0.5 also `ctx.register_library("bpool_cs", big.action, progress=big.progress, task_id=big.task_id)` (Result.library must be "current"/"bpool_all"/registered). σ = `ctx.action_sigma` (7,). Per task t: `rows_t = big.rows_of_task(t)`; `AF_t = (big.action[rows_t][:, :5, :7] / σ).reshape(-1, 35).astype(f32)` contiguous, `a2_t = ||AF_t||²`; `RS_t = big.rs[rows_t][:, :8]` (dims.valid_state), `r2_t`. Scales (T1, library-only, seconds): `s_d` = median over library rows of the rs L2 to the nearest row of another episode in the same task; `s_c` = median over library rows of RMS_σ(entry.action[5:10,:7] − nearest other-episode head[0:5,:7]) (i.e. the library's own tail→head continuity); `s_a` = median pairwise RMS_σ among each row's 3 nearest heads. Optional: drop failed episodes (`big.success`) — measured neutral-to-worse for rs-only (.449→.449 π0.5-sp, .423→.456 π0.5-l10), so default keep all.
- `reset(episode)`: nothing to store (all state comes from `q.prev_a_exec` / `q.hist_a_exec`).
- `query(q)`: `d = sqrt(r2 − 2 RS·rs_q + |rs_q|²)`; if `q.step == 0`: `f = d/s_d`; else `tail = q.prev_a_exec[5:10, :7]/σ` (35,), `c = sqrt(max(a2 − 2 AF·tail + |tail|², 0)/35)`, `f = c/s_c + α·d/s_d` (α = 0.25 default; GR00T variant adds `β·c2/s_c2` with `c2` against `q.hist_a_exec[-2][10:15, :7]` when `q.step ≥ 2`). `o = argsort(f)[:10]`; `action[:5,:7]` = per-step median over `big.action[rows_t[o[:3]]]` for dims 0–5, dim 6 = sign of the majority (ties → +1); fill the rest of the (H,32) with `o[0]`'s chunk. Confidence (fixed-scale, online-computable): `conf = −c[o0]/s_c − d[o0]/s_d − disp3/s_a` where `disp3` = mean pairwise RMS_σ among the 3 heads (at step 0 use `−d/s_d − disp3/s_a` only). Return `Result(topk=rows_t[o], scores=−f[o], confidence=conf, action=..., library=big_name, extras={"cont": c[o0], "rs": d[o0], "disp3": disp3, "flipfrac10": fraction of the 10 whose gripper flips vs prev_a_exec[4,6]})`.
- `bytes_per_entry()` = 8·4 = 32 (actions are the shared payload).

**Predicted effect** (D2/D5 measured with query-side scale medians and cell-level rank-average confidence; the library-fitted scales and the fixed-scale sum should land within ~.01/.02): err top-3-median on the 10× library .29 .59 .29 .57 .29 .50 .29 .59 (vs B0 .49 .64 .47 .60 .52 .62 .54 .64); by third on π0.5-sp-inf [.23 .32 .29] vs B0 [.40 .56 .52]; regret vs the *current* oracle becomes ≈ 0 on inf arms (.29 vs .31/.24/.33/.27) and ≈ .30 on cache arms. AURC .20 .42 .20 .42 .19 .36 .20 .45 (vs .38 .41 .37 .39 .43 .41 .43 .44); risk@30 % ≈ .17–.19 on inf arms (B0 .34–.39), ≈ .32–.42 on cache arms (B0 .31–.36 → slightly worse there, see P3). grip_mis .033 .20 .050 .19 .039 .14 .070 .22 (B0 .042 .19 .065 .17 .064 .14 .094 .23). Cost: ~0.1–0.3 ms/query at 3,000 candidates (one 1×35·35×3000 GEMM + 8-d L2), fit seconds, 32 B/entry, 30 k-row library = 1 MB of rs + actions.

**Tier:** T0 for the scores (T1 for the three library scales). Deployment: no image keys, no key build (saves the pooling too), library growth is free; needs the previous chunk (already in the client).

**Kill criteria.** (a) Full-run cache-arm err not ≤ B0 − .02 on ≥ 3 of the 4 cache cells, or inf-arm err > .35 anywhere. (b) Closed-loop: SR drop > 2 pp vs B0 at matched hit rate, or the run-length of consecutive hits grows with err (open-loop drift along a library trajectory) — this is the risk the offline harness cannot see.

**Cheapest diagnostic first.** Already done (D2/D5). Next cheapest is the harness `smoke` + full run, then `timeline` on the run to compare its switch/track/stay rates against the oracle's (45–60 % switch, run length ≈ 2) and `breakdown --slices grip,third`.

**Variants.** α ∈ {0, 0.25, 1} (α=0 best on inf arms .371/.350/.372/.356, flat on cache arms; α=1 costs +.03 on inf); GR00T β ∈ {0, 1} for the 2-back chunk (helps cache arms only: GR00T-sp-cache med3 .544→.521 current, .509→.481 bpool; l10-cache .608→.590/.610→.595; −.005…−.01 on inf arms); top-k median k ∈ {1, 3, 5} (k=5 slightly better on l10-cache: .559/.584); library ∈ {current, big}; success-only filter.

---

## P2 `b0_shortlist_continuity_rerank` — keep B0 exactly, re-rank its top-10 by continuity, return the top-3 median (T0)

**Pitch.** Minimal-diff upgrade of the deployed pipeline: same keys, same search, same latency; only the final selection changes.

**Hypothesis.** B0's short list is good (contains the oracle 44–82 % of the time, best-of-10 .33–.47) but its order is nearly uninformative among saturated scores (pick rank p50 3–17). Continuity is orthogonal to the vision/rs score, so it resolves the ties the fused score cannot, while the vision-grounded shortlist prevents the pure-continuity chain from drifting off the observation.

**Algorithm.** Wrap `baselines.FusedKNN` (or copy `score_all`): `rows, f, parts = self.score_all(q)`; `o = argsort(−f)[:10]`; if `q.step > 0`: `c` = RMS_σ(cand head − prev tail) for the 10 (10×35 ops), order by `rank_B0 + λ·rank_cont` with λ=2 (Borda; λ→∞ = pure continuity order) else keep B0 order; `action` = per-step median (dims 0–5) + gripper majority over the first 3 of the new order, rest of the chunk from the first row. Confidence as P3 (B0 fused score of the pick, agreement among the 3, −continuity). Extras as P1 plus B0's per-field sims. `bytes_per_entry` = B0's.

**Predicted effect** (D1/D1b measured): pure continuity order + median-3: .380 .620 .363 .574 .392 .581 .395 .604; Borda λ=1 + median-5: .408 .605 .383 .549 .428 .563 .433 .587 (best on cache arms); B0 order + median-5 only (no continuity at all): .452 .621 .428 .565 .478 .574 .487 .597. AURC with P3's signals .28–.31 on inf arms, .38–.43 on cache arms. Cost unchanged (1.3–8.3 ms, 262 KB/entry) + ~10 µs.

**Tier:** T0. Deployment: a few lines in the selection stage of `_search_weighted_score_sum`; no new data.

**Kill criterion.** Full run: err not ≤ B0 − .10 on inf arms or ≤ B0 − .02 on cache arms; or `flip_rate` gains that vanish in `compare`'s episode-bootstrap CI.

**Cheapest diagnostic first.** Done (D1: re-ranking saved npz top-10 needs no search). Next: the full run + `compare` vs B0 and vs P1.

**Variants.** λ ∈ {1, 2, ∞}; k ∈ {3, 5}; shortlist size 5 vs 10 (10 is better everywhere by .01–.02); replace the B0 shortlist with `bpool` rs+cont pruning (→ P4).

---

## P3 `consistency_confidence` — a drop-in confidence head for any ranker (T0/T1)

**Pitch.** Confidence = agreement of independent evidence about the chosen action, not the retrieval score.

**Hypothesis.** The fused score is a saturated similarity, so it barely moves between good and bad picks (conf .91–.99 on transitions vs .95–.99 steady). Three cheap quantities do move: (i) the continuity residual of the pick vs the previous chunk's tail (after a MISS this is a direct teacher check; after a HIT it still measures smoothness), (ii) the dispersion of the top-k actions (if the neighbourhood agrees on the action, the choice is robust — the same reason B0's margin is *anti*-informative), (iii) rs distance / B0 score as observation grounding, which is what carries the after-HIT case.

**Algorithm.** For the returned pick: `conf = −w1·cont/s_c − w2·disp_k/s_a + w3·g` with `g` = B0 fused score (P2/B0) or `−d/s_d` (P1), equal weights after library-fitted scaling (`s_c, s_a, s_d` as in P1; for B0-score use its library-internal top-1 score spread). At `step 0`: drop the continuity term. Variant "gripper veto tier": if `flipfrac10 ≥ τ` (τ = .3) subtract a constant larger than the conf range (accept these only when everything else is accepted). Fitted parameters: three scales, from the library only (T1, seconds); nothing from queries.

**Predicted effect** (D1/D6, measured on B0's own picks with cell-level rank-averaging, so expect ±.02 from the fixed-scale sum): AURC .313 .390 .292 .384 .347 .376 .341 .423 (B0 .382 .406 .367 .390 .425 .408 .430 .437; attainable .269 .308 .243 .292 .302 .310 .286 .326); risk@30 % .260 .304 .238 .301 .297 .314 .281 .325 (B0 .351 .325 .341 .311 .390 .357 .394 .357) — i.e. at the same 30 % hit rate the accepted err drops 25 % on inf arms; on π0.5 cache arms risk@30 is ≈ B0 (the after-HIT regime is where B0's score remains useful, hence keep it as term (iii)). Veto tier τ=.3: AURC +.005…+.02 worse, but gripper mismatch among the accepted 30 % falls 3–10× on inf arms (e.g. GR00T-l10-inf .036→.010, π0.5-l10-inf .005→.001); on cache arms the flip fraction is less predictive (AUROC .72) and the veto buys little. Cost: ~µs; no bytes.

**Tier:** T0 (T1 for scales). Deployment: replaces the threshold variable of the HIT/MISS judge; thresholds must be re-tuned (different scale).

**Kill criterion.** Full-run AURC not below B0 on ≥ 6 of 8 cells, or risk@30 % worse than B0 on any inf cell.

**Cheapest diagnostic first.** Done for B0's picks; for P1/P2 picks the `extras` (`cont, disp3, rs, flipfrac10`) make it a 1-minute post-hoc `risk_coverage` recomputation from the npz — do this before spending a run on weight variants.

**Variants.** weights (1,1,1) vs (1,1,0) after a MISS and (0,1,1) after a HIT (the HIT/MISS state is known online; offline the arm tells it: AURC .309 .390 .291 .375 .344 .386 .332 .415 measured for this mix); k ∈ {3, 5, 10} for dispersion; veto τ ∈ {off, .3, .5}.

---

## P4 `cascade_biglib_shortlist` — rs+continuity prune the 10× library to M=16, then vision re-rank on the 16 (T0/T1)

**Pitch.** Use the big library and keep an observation-grounded vision term, at 10–50× lower latency than B0 today.

**Hypothesis.** The 35-d+8-d score is an excellent *recall* filter (its top-16 of `bpool` holds a candidate at .213 .426 .187 .430 .218 .323 .211 .432 vs the big-library oracle .19 .28 .14 .19 .20 .23 .16 .18), and vision similarity, useless for ranking 100–300 saturated candidates, is a useful tie-breaker among 16 that are already action-consistent — and it is the grounding P1 lacks after a run of HITs. Latency ∝ L^1.28 disappears because the cosines are only computed on 16 rows.

**Algorithm.** `fit`: as P1 on the big library plus per-task key access (`big.key_v0/key_v1` memmaps in the offline harness; for deployment fit per-field PCA-64 on the big library — participation ratio 7–54 — and store 2×64 f32 = 512 B/entry; T1, randomized SVD, seconds-minutes); B0 μ/σ/weights from `ctx.current_params()`. `query`: `f_pre = c/s_c + α d/s_d` (P1); `cand = argsort(f_pre)[:16]`; gather 16 keys, `fB0` = B0's weighted tanh-z sum on the 16 (global params are fine — D3 shows recalibration is worthless), `f = fB0 + 0.5·(−c/s_c)`; order by `f`; top-3 median action; confidence P3 with `g = fB0` of the pick. `bytes_per_entry` = 32 + 2·32768·4 (offline) or 32 + 512 (PCA variant).

**Predicted effect** (D4, 1,200-query subsample, M=16 on bpool, full keys): err .291 .586 .283 .580 .273 .511 .290 .582 vs B0 on the same subsample .490 .656 .459 .607 .527 .612 .535 .621 — ≈ P1 offline (P1 bpool med3 .293 .590 .289 .566 .291 .495 .293 .593): better on 5 cells, worse on the two l10-cache cells by ~.015. AURC by `−cont` .209 .471 .221 .492 .199 .410 .208 .531, by the 3-signal average .207 .456 .218 .469 .197 .392 .206 .501. Cost: 16×2×32768 MACs ≈ 1 M + the P1 GEMM ≈ 0.1–0.4 ms/query independent of L (B0: 1.3–8.3 ms and growing); with PCA-64 keys ≈ 2 k MACs for vision.

**Tier:** T0 with raw keys, T1 with PCA. Deployment: the only proposal that keeps an image term while using a 10× library; value is closed-loop grounding, not offline err.

**Kill criterion.** Full run not ≤ P1 on ≥ 4 cells *and* no closed-loop advantage over P1 → drop the vision stage entirely. PCA-64 variant: cache-arm err worse than raw keys by > .01.

**Cheapest diagnostic first.** Extend D4 to PCA-64 keys (one randomized SVD per field per suite on `bpool`, minutes on CPU) before writing the method; M ∈ {16, 32, 64} is already flat (±.003), so fix M=16.

**Variants.** M=16 fixed; vision weight on the 16 ∈ {B0 fused, v1 only (wrist beats v0 everywhere), none}; PCA dim ∈ {32, 64, 128}.

---

## Ranking (gain × probability / cost)

1. **P1** — largest measured gain (−40 % err, AURC −.18 on inf arms; −5…−20 % err on cache arms), ~zero cost, trivial to implement; probability discounted by the unmeasurable closed-loop stickiness.
2. **P2** — half of P1's inf-arm gain and the same cache-arm gain with zero deployment risk (unchanged keys/search, shortlist stays vision-grounded); highest probability.
3. **P3** — the number that actually sets the online hit rate; measured on B0's own picks already (risk@30 % −25 % on inf arms), composable with all of the above.
4. **P4** — offline ≈ P1, its value is grounding + latency independence from L; carry it as P1's safety twin rather than a standalone bet.

## Considered and rejected (with the measurement)

- **Per-task μ/σ, linear z without tanh, product/min fusion, rs-50 → B0 vision cascade** (lead 1, §7 "消除饱和"): B0 top-10 candidates are indeed in tanh's flat region (z medians 2.0–3.3, |z|>2 for 50–98 %), but every recalibration changes err by ≤ .012 (subsample: B0 .490/.647/.467/.603/.527/.609/.534/.646 → per-task linear z .487/.645/.454/.593/.524/.612/.539/.638; min-fusion worse by up to .04). The similarity is weakly action-relevant; its normalization is not the lever.
- **Trajectory tracking / HMM-style "follow `next`"** (lead 5): forcing `next(prev pick)` on stay decisions ≈ B0 (.494 .642 .479 .594 .527 .618 .551 .627); "track if in top-3" no better. The oracle switches 45–60 % of the time with run length ≈ 2 — the library trajectories are not the teacher's trajectory; continuity in *action* space (P1) is the right version of this idea.
- **Gripper-consistency filter as a ranker** (lead 4): net zero (steady −.005…−.03, transitions +.10…+.20 because at true transitions the flipping candidates are the right ones). Survives only as the veto tier in P3.
- **Medoid of top-k / rs re-rank within B0's top-10 / mean instead of median**: medoid .484 .662 .453 .596 .516 .611 .520 .638 (≈ B0), rs re-rank ≈ B0 except GR00T-l10-cache (.621), i.e. the per-step median + gripper vote is the only synthesis that pays.
- **Not re-proposed** per the brief: score smoothing over history (P1 uses one previous chunk as a *feature*, not a smoothed score — the 2-back chunk variant for GR00T is a second feature with its own weight and is only a variant), percentile normalization, RRF.
