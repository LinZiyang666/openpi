# R1 analysis — offline retrieval-method exploration (2026-09-26, analysis agent)

Inputs: 406 (method × cell) runs of round r01 plus the r00 references (`results/scoreboard.csv`, latest `run_ts` per
method × cell; per-decision npz under `results/r01/<method>/`). All numbers are the harness metric (err = RMS over the
executed block `[:5,:7]` in σ_d units, GT = `a_inf`), 188,844 decisions. Cell order in every 4-tuple: π0.5-spatial /
π0.5-l10 / GR00T-spatial / GR00T-l10. `ms/query` are the batch timing pass at `timing_concurrency = 4` (one pinned
physical core each) for every r01 method; the r00 references were timed in-run (no concurrency field), so B0's
1.3–8.3 ms is not on the same footing. Scratch (tables, compare/breakdown/timeline/explain/runprof reports, scripts
`s01..s06`): `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01/`.

**Regimes.** In the trace, every step ≥ 1 decision of an `inf` cell follows a MISS (the previous chunk is the
teacher's; "fresh" regime) and every step ≥ 1 decision of a `cache` cell follows a HIT (the previous chunk is a
library row served by B0; "stale" regime). Step 0 has no previous chunk. The methods' `x_regime` extras confirm this
100 % (500 step-0 + only-fresh / only-stale rows in every cell). The cache-arm states were produced by **B0 with the
current library**, not by any new method — every stale-regime number below is "how the method does on the states B0
drove the robot into", including B0's failures (failed episodes are 50 / 71 / 42 / 70 % of the cache-arm decisions).

## 0. Sanity checks (artefacts looked for)

| check | result |
|---|---|
| leakage: query `a_inf[:5,:7]` bit-identical to a same-task library chunk | 0.0000 in all 16 (cell × library) pairs; `a_exec` of cache arms is in `current` 100 % (B0's hits, as designed) and in `bpool_all` 100 % for GR00T (current ⊂ bpool_all), 0 % in `bpool_cs` |
| query / library init pools | queries = A-pool inits, libraries = B-pool inits (init-index mapping verified by the M9 probe, ρ .46–.56 vs controls ≈ 0) |
| duplicate variants | M1 α∈{0,.25,1} and stale∈{window,cont} are bit-identical in the cells where the varied term is unused; M3 mass/ent/lev/nocont identical in cache cells (continuity unused after a HIT); M4_k·_med ≡ M5_·_k· in cache cells (no re-rank after a HIT); M6 ≡ B0 in err. These rows are not independent evidence. M1_med3 vs M7_none in inf cells are **not** duplicates (top-1 agree 57–63 %; M7 re-orders its 16 by continuity alone) but land within .001 |
| regime identity | `x_regime` = step0/fresh/stale exactly as the arm predicts in every cell (see above) |
| metric | executed segment `[:5,:7]` everywhere; synthesized actions are scored on the same block; `phase_err` uses top-1 |
| teacher-noise floor | π0.5 floor p50 .11–.13 (p90 .37–.40), GR00T .03–.05. Best fresh method is 1.6–1.7× floor p50 on π0.5 (indist .21–.23) and 4–6× on GR00T (indist ≈ 0, as for every method). Stale regime: 3.2–3.7× (π0.5), 7.5–15× (GR00T) |
| timing provenance | all 406 r01 cells: `timing_only`, concurrency 4. M8 (B0 formula on the 10× library, raw keys) is memory-bandwidth bound: 69 ms/query at concurrency 4 (75–630 ms in the 36–44-worker accuracy run) |

## 1. Per-method verdicts

### 1.1 Fresh regime (inf cells; every step ≥ 1 decision follows a MISS)

err mean per cell; the other columns are means over the 4 inf cells. Big-library oracle (bpool_cs / bpool_all): .186 / .140 / .204 / .158; current-library oracle .308 / .239 / .326 / .265; B0 .492 / .467 / .524 / .535 (AURC .401, risk@30 .369, grip .066).

| method | err (4 cells) | err p50 | AURC | risk@30 | grip | regret vs big oracle | bad@100 | ms/q | B/entry | fit s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R_tail (reference, not a library action) | .249 .282 .211 .239 | .177 | .184 | .168 | .039 | .073 | .24/.25/.86/.82 | 0.44 | 268 | 4.3 | reference: fresh-regime ceiling |
| **M1 big α.25 kmean5 stale=window** | **.260 .274 .250 .265** | .202 | **.191** | **.172** | .039 | .090 | .26/.24/.95/.91 | 0.46 | 268 | 4.2 | **KEEP (best)** |
| M1 big α.25 med3 | .279 .287 .273 .283 | .221 | .204 | .183 | .039 | .109 | | 0.42 | 268 | 4.2 | drop variant (kmean5 dominates, −.013…−.024 paired) |
| M1 big α.25 top1 | .295 .299 .287 .292 | .235 | .211 | .189 | .039 | .121 | | 0.35 | 268 | | drop |
| M1 big α0 / α1 (med3) | .280 .293 .272 .282 / .293 .289 .291 .292 | | .207 / .210 | | | | | | | | α≤.25 flat (state term ±.005), α=1 hurts +.015 |
| M1 big β1 (GR00T) | – – .283 .298 | | .203 | | .047 | | | | | | drop (+.010 / +.016 vs β0) |
| M1 current med3 | .370 .345 .373 .360 | .308 | .256 | .225 | .045 | .190 | | 0.27 | 268 | 0.04 | drop (library size −.058…−.102) |
| M7 cascade none (state+cont prefilter 16, med3) | .281 .291 .271 .282 | .221 | .205 | .184 | .039 | .109 | | 0.39 | 172 | 3.0 | ≡ M1 med3; keep as M7's control |
| M7 pca32 both / raw both | .282 .283 .287 .289 / .282 .282 .291 .289 | .226 | .207 / .209 | .184 | .039 | .113 | | 0.95 / 2.66 | 428 / 262 K | | vision re-rank neutral (π0.5) to harmful (GR00T +.017 spatial, +.007 l10) after a MISS |
| M7 raw b0fused | .279 .290 .271 .281 | .220 | .204 | .182 | .039 | .108 | | 2.66 | 262 K | | ≈ none; drop |
| M3 HMM lev (em1, cont) | .314 .301 .320 .313 | .257 | .215 | .184 | .042 | .140 | | 0.54 | 176 | 1.4 | DROP: transition prior costs +.03 vs plain continuity kNN (M1 med3); α=.8 worse than .6 |
| M8 B0-formula on big lib, mean5 | .329 .302 .362 .347 | .284 | .279 | .262 | .040 | .163 | | 68.6 | 262 K | 6.3 | not a fresh-regime method |
| M9c vision z-sum pca128 st1 med3 | .342 .303 .394 .373 | .290 | .272 | .246 | .041 | .181 | | 4.6 | 1056 | 0.6 | not a fresh-regime method |
| M2 state window m3 k8 mean | .367 .341 .376 .373 | .312 | .269 | .241 | .049 | .192 | | 0.36 | 96 | 1.1 | not for fresh (no tail) |
| M5 B0 top-10 re-ranked by continuity (λ=∞, k3) | .380 .363 .392 .395 | .322 | .316 | .295 | .049 | .211 | | 5.7 | 262 K | 1.1 | DROP: recovers < ½ of M1's gain at B0's cost |
| M4 B0 consensus kernel5 | .425 .414 .452 .469 | .375 | .356 | .331 | .060 | .268 | | 5.7 | 262 K | 3.2 | fold into others (synthesis rule only) |
| M6 consistency conf. on B0 (eq / split) | = B0 | | .323 | .269 | .066 | .333 | | 5.7 | 262 K | 3.4 | idea KEEP (dAURC −.067…−.097, CI excludes 0); base obsolete; veto τ.3 hurts AURC (+.013) |
| B0 (r00) | .492 .467 .524 .535 | .431 | .401 | .369 | .066 | .333 | .67/.54/1.0/.99 | 1.3–8.3* | 262 K | 0.9 | baseline |

\* in-run timing, not concurrency 4.

Compare (episode bootstrap, 2000 reps) M1 kmean5 vs B0: d err −.232 / −.193 / −.275 / −.271 (all CIs within ±.01), win rate .80–.92, dAURC −.198 / −.162 / −.247 / −.234, d indist +.164 / +.178 (π0.5). Vs R_tail: +.011 / −.008 / +.039 / +.025 — M1 is within .01 of executing the teacher's own tail on π0.5 and .03–.04 behind on GR00T (GR00T's teacher is near-deterministic, so its tail is a better forecast than any library chunk). Vs the current-library oracle: M1 is below it in 3/4 cells (regret vs current −.049 / +.035 / −.076 / −.001).

### 1.2 Stale regime (cache cells; every step ≥ 1 decision follows a HIT served by B0)

Big-library oracle .282 / .190 / .229 / .179; current oracle .394 / .277 / .361 / .261; B0 .643 / .597 / .621 / .644 (AURC .410, risk@30 .337, grip .184).

| method | err (4 cells) | err p50 | AURC | risk@30 | grip | regret vs big oracle | ms/q | B/entry | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **M8 B0 formula on big lib + mean5** | **.526 .480 .453 .488** | .395 | **.331** | .277 | .166 | .266 | 68.6 | 262 K | KEEP as the stale-regime accuracy reference; **not deployable as is** (memory-bound, 1000× the entry size) → refine into "PCA-32/64 keys + mean-k" |
| M2 state window m3 k8 mean | .598 .528 **.445** .499 | .430 | .345 | .285 | .178 | .297 | 0.36 | 96 | **KEEP** (vision-free stale + step-0 branch; ties/beats M8 on GR00T) |
| M9c vision z-sum pca128 st1 med3 | .562 .495 .496 .517 | .412 | .385 | .342 | .159 | .297 | 4.6 | 1056 | drop as a method (M8 −.015…−.042 better; its confidence is poor); its representation finding stands |
| M7 pca32 both, M=64 | .573 .509 .510 .533 | .425 | .343 | .278 | .164 | .311 | 0.95 | 428 | REFINE (wider shortlist helps π0.5 −.02, hurts GR00T +.02/+.007) |
| M7 pca32 both, M=16 | .590 .528 .490 .526 | .430 | .343 | .276 | .163 | .313 | 0.95 | 428 | REFINE (the only cheap vision method; PCA-32 ≡ raw) |
| M7 raw both / raw b0fused | .598 .532 .485 .522 / .595 .538 .484 .528 | .431 | .342 / .347 | .273 / .279 | .165 | .314 / .316 | 2.66 | 262 K | drop (no gain over pca32) |
| M7 pca32/64/128 v1only | .587 .531 .502 .535 (pca32) | .437 | .348 | .280 | .164 | .318 | 0.68–1.29 | 300–684 | drop (wrist-only ≈ both −.00…+.01) |
| M7 none (state prefilter 16, med3) | .644 .585 .480 .525 | .465 | .373 | .306 | .162 | .338 | 0.39 | 172 | control |
| M1 kmean5 (its window branch) | .603 .535 .468 .509 | .442 | .357 | .295 | .173 | .308 | 0.46 | 268 | = M2 k5-mean on the same window (−.01 vs M2 k8 on GR00T, +.01 on π0.5) |
| M1 stale=cont (med3) | .586 .573 .511 .601 | .468 | .386 | .326 | .164 | .347 | 0.35 | 172 | drop (−.061 π0.5-sp but +.018 / +.051 GR00T; continuity confidence spurious) |
| M3 HMM (any conf) | .636 .579 .492 .536 | .462 | lev .410 / mass .712 / ent .695 | .363 / .756 | .173 | .340 | 0.54 | 176 | DROP (+.02…+.04 vs M2 m1 k5; mass/entropy **anti-informative**: risk@30 .92 on π0.5-spatial) |
| M4 B0 consensus mean5 | .589 .540 .548 .568 | .461 | .396 | .337 | .194 | .341 | 5.7 | 262 K | fold into others |
| M5 / M4 med (any) | .621–.631 .565–.574 .574–.576 .597–.614 | | .408 | .342 | .192 | .369–.378 | 5.7 | 262 K | drop |
| M6 on B0 (eq) | = B0 | | .389 | .307 | .184 | .406 | 5.7 | 262 K | dAURC −.013…−.026 only |
| R_tail (keep executing the previous library chunk) | .594 .590 .537 .625 | .490 | .405 | .344 | .166 | .366 | 0.44 | 268 | reference; note it beats B0's re-retrieval (−.048 / −.006 / −.084 / −.018) |
| B0 | .643 .597 .621 .644 | .511 | .410 | .337 | .184 | .406 | 1.3–8.3* | 262 K | baseline |

Per-model split of the stale verdict: on **π0.5** the order is M8 (.526/.480) ≪ M9c (.562/.495) < M7-M64 (.573/.509) < M7 (.590/.528) ≈ M2 (.598/.528); on **GR00T** it is M2 (.445/.499) ≈ M8 (.453/.488) < M7 none (.480/.525) ≈ M7 (.490/.526) < M9c (.496/.517). The vision-carrying methods only separate from the state-only ones on π0.5.

### 1.3 Step 0 (500 decisions per cell; inf and cache step-0 queries are the same inits)

| method | err (4 cells, inf) | AURC (step 0) |
|---|---|---|
| oracle (current) | .212 .140 .233 .132 | .112 |
| M8 mean5 | **.219 .181** .274 .197 | .213 |
| M9c st1 med3 | .234 **.167** .304 .211 | .197 |
| M7 raw both / pca32 both | .244 .177 .294 .189 / .246 .177 .293 .197 | **.184** / .188 |
| M2 m1 k5 mean | .247 .198 **.271** .198 | .189 |
| M2 m3 k8 mean | .245 .204 .274 .204 | .195 |
| M1 (state only at step 0) | .270 .210 .296 .219 | .202 |
| B0 | .317 .268 .392 .248 | .284 |

## 2. Cross-method findings

### 2.1 What carries the gain (fresh regime, mean of 4 inf cells, step ≥ 1)

B0 .510 → B0 + kernel-mean-5 (M4) .445 → continuity on the *current* library (M1 current, med3) .364 → continuity on the 10× library (M1 med3) .282 → kernel-mean-5 (M1 kmean5) .262 → executing the teacher's own tail (R_tail) .245. Paired contributions (episode-bootstrap CIs all exclude 0): **continuity vs state-only on the same big library** −.09…−.11 (M1 kmean5 vs M2 k8); **library 10×** −.058…−.102 (M1 big vs current); **synthesis** top1→med3 −.009…−.014, med3→kmean5 −.013…−.024; **state term** in the fresh score ±.005 (α 0 vs .25), α=1 +.015; **vision re-rank** 0…+.02 (harmful on GR00T). The fresh regime is now regret .07–.15 from the big-library oracle and, on π0.5, 1.6–1.7× the teacher noise floor (indist .21–.23 vs B0 .04–.06).

### 2.2 What carries the gain (stale regime, mean of 4 cache cells, step ≥ 1)

B0 .626 → B0 + mean-5 .561 → state window on the 10× library, mean-8 (M2) .524 → + B0's vision fusion on the same library (M8) .493. Contributions: **library 10×** −.025…−.10 (M1 big vs current, cache); **synthesis** top1→med3 −.028…−.035, med3→mean/kernel-5 −.026…−.044, mean-5→mean-8 −.004…−.009, and **median is worse than mean everywhere** (+.026…+.033 at k=5 in the stale regime, +.015…+.018 fresh) — B's "median + gripper vote" rule loses to A's mean on the same shortlist; **vision** (M8 vs M2 m1 k5, same library, same k, same synthesis) −.077 / −.061 on π0.5, −.000 / −.025 on GR00T; **transition prior** (HMM) +.02…+.04; **continuity after a HIT** (M1 stale=cont) −.061 on π0.5-spatial but +.018 / +.051 on GR00T and its confidence is spurious. Regret vs the big-library oracle remains .27–.31 for the best method: the stale regime is where the remaining error is.

### 2.3 The vision question (owner)

Where vision earns its cost, paired per-decision differences (A − B, episode-bootstrap 95 % CI; `*` = CI excludes 0):

| contrast | regime | π0.5-sp | π0.5-l10 | GR00T-sp | GR00T-l10 |
|---|---|---|---|---|---|
| M8 (B0 vision fusion, raw, mean5) − M2 m1 k5 mean (state only) | step 0 | −.028* | −.017* | +.003 | −.001 |
| | fresh | −.046* | −.051* | −.028* | −.039* (but both ≫ M1: vision is irrelevant after a MISS) |
| | stale | **−.077*** | **−.061*** | −.000 | −.025* |
| M7 pca32-both re-rank − M7 none (state+cont prefilter, M=16) | step 0 | −.021* | −.027* | −.003 | −.016* |
| | fresh | +.002 | −.008* | **+.017*** | +.007* |
| | stale | −.054* | −.058* | +.010* | +.001 |
| M7 raw both − M7 pca32 both (key size 262 KB vs 256 B) | any | −.002…+.008 | ±.004 | ±.005 | ±.008 |
| M7 pca128 v1 − pca32 v1 | any | ≤ .004 | ≤ .003 | ≤ .006 | ≤ .003 |
| M9c pca128 z-sum st1 top1 − M8 top1 (repr. + fusion, top-1) | stale | −.014* | −.023* | +.007* | −.015* |
| M9c st1 − st0 (state term inside the vision score) | stale | +.002 | −.001 | −.028* | −.033* |

Reading:
- **After a MISS vision is worthless**: the continuity signal (35-d) beats every vision method by .05–.10 and adding a vision re-rank on top of it is neutral (π0.5) or harmful (GR00T). Vision-free pipeline = M1 for that regime, full stop.
- **After a HIT vision is worth −.06…−.08 on π0.5 and ≈ 0 on GR00T** (M8 vs M2, same library and synthesis). At step 0: −.02…−.03 on π0.5, 0 on GR00T. Mixed at 50 % hits, a vision-free system (M1 + M2) loses .024–.036 err on π0.5 and 0.000–.005 on GR00T versus M1 + M8 (§2.5). Its cost: 96–268 B/entry and 0.4–2.3 ms/query vs 262 KB and 15–65 ms.
- **Representation size**: inside M7, raw 2×32768-d keys vs PCA-32 (task-agnostic, big-library fit) differ by ≤ .008 in every cell and regime; PCA-128 vs PCA-32 by ≤ .006. So the visual information that matters survives a 1000× compression. The M9 probe agrees: pooled 4×4 keys of both cameras, task-centred and z-summed, recover the privileged-layout gain fully (recovery ≥ .57 in all 19 slots), and token-level MaxSim/Chamfer, top-variance tokens and raw pixels are all worse or equal (§2.4). The 262 KB entry buys nothing over ~256–512 B.
- **Why M7 does not recover M8's π0.5 gain** (M7 pca32 both .590/.528 vs M8 .526/.480 stale): two things, both measured. (i) The state prefilter is too tight in the stale regime — drifted states make the 8-d distance unreliable, and widening M 16→64 already gives −.018/−.020 on π0.5 (while hurting GR00T +.020/+.007, where state is the better signal). (ii) med3 synthesis vs mean-5: −.03…−.04 (M1 med3→kmean5 in cache; M9c med3→top1 shows median-3 is still +.04 better than top-1, but mean-5 is better still). M8's own "top1 → mean5" is −.066…−.093 in cache cells. So the deployable target is "vision score over all (or ≥ 64–128) candidates with PCA-32/64 keys + mean-5", not "vision on a state-prefiltered 16".
- **Fusion formula**: B0's tanh-z weighted sum vs a plain z-sum of the two centred cosines: ≤ .008 either way (M7 b0fused vs both); the M9 probe found the same (recalibration of μ/σ moves err by ≤ .012). Not a lever.
- **Wrist vs both cameras**: v1-only is ≈ both in M7 (−.00…+.01 stale); the probe puts `pool_both+st` ahead of `pool_v1+st` by .006 mean gain. Keep both, but it is a second-order choice.

### 2.4 What the M9 probe says (steps 0–2, step-aligned candidates, ~50 per task)

- The privileged object-layout picker (`priv_*`) is barely better than state at step 0 (gain over state .002–.057, largest on GR00T-l10 .056) and *worse* than state at steps 1–2 in most cells: object placement is a weak determinant of the executed action once the arm has moved; the oracle (.08–.16) is far below all of them, i.e. the action-relevant variation is mostly not layout.
- Among training-free visual representations, `pool_both`, `res_pool_tm_both` (task-mean centred) and their `+st` versions are the best at every step (e.g. step 0 π0.5-sp .262/.260 vs state .317, B0 .270; GR00T-l10 .199 vs state .255), and they recover the whole privileged gain (mean recovery ≈ 4×, min .57–.72). Token-level MaxSim / Chamfer are no better (.264/.265), top-variance token subsets and raw pixels are worse (.27–.31), wrist-only is worse than both. Ranking quality (Spearman sim ↔ −err) is .25–.36 for the pooled reps, .16–.18 for `pool_v0` alone.
- The vision edge at step 0 in the probe (−.05 on π0.5-sp, −.056 on GR00T-l10 vs state) shrinks to −.03 / 0 in the full methods because the state-window kNN with a mean-5 over *all* rows (M2, .247/.198 at step 0) already closes most of it; with top-1 selection the gap persists (M8 top1 .271/.229 vs M2 m1 .247/.198 — vision top-1 is *worse* than state mean-5 at step 0 on GR00T).

### 2.5 Regime-mixed estimate (owner)

Deployed caches alternate. Estimate for a combined policy — after a MISS: M1 kmean5; after a HIT: one of {M8, M2 k8, M7 pca32 both, M1's own window branch}; step 0: the best step-0 picker of the stale method family — with `mix(p) = w0·e0 + (1−w0)·(p·e_stale + (1−p)·e_fresh)`, w0 = share of step-0 decisions (.046 spatial, .017 l10). Stale rows are the B0-driven cache-arm states (pessimistic for a better cache); the "succ-only" column restricts the stale rows to cache-arm episodes B0 actually completed (optimistic bound). B0 mixed the same way is given for reference.

| model × suite | fresh (M1) | stale method | stale all / succ-only | mix p=.3 | p=.5 | p=.7 | mix@.5 succ-only | B0 mix@.5 | ms/q @conc4 (p=.3/.5/.7) | bytes/entry (union) |
|---|---|---|---|---|---|---|---|---|---|---|
| π0.5-spatial | .259 | M8 mean5 | .536 / .348 | .337 | .389 | .442 | .304 | .565 | 14.5 / 22.6 / 30.8 | 262.5 K |
| | | M2 k8 | .611 / .402 | .358 | .425 | .492 | .330 | | 2.3 / 2.3 / 2.3 | 364 |
| | | M7 pca32 both | .603 / .385 | .355 | .421 | .487 | .322 | | 2.5 / 2.6 / 2.7 | 696 |
| π0.5-l10 | .275 | M8 mean5 | .484 / .333 | .335 | .376 | .417 | .304 | .531 | 28 / 46 / 65 | 262.5 K |
| | | M2 k8 | .532 / .380 | .349 | .400 | .450 | .328 | | 0.5 / 0.5 / 0.5 | 364 |
| | | M7 pca32 both | .532 / .359 | .349 | .400 | .450 | .317 | | 0.7 / 0.8 / 0.9 | 696 |
| GR00T-spatial | .248 | M8 mean5 | .460 / .337 | .310 | .350 | .391 | .293 | .572 | 16 / 26 / 35 | 262.5 K |
| | | M2 k8 | .451 / .353 | **.307** | **.346** | **.385** | .301 | | 2.3 | 364 |
| | | M7 pca32 both | .497 / .375 | .320 | .368 | .416 | .311 | | 2.4–2.6 | 696 |
| GR00T-l10 | .265 | M8 mean5 | .492 / .352 | .331 | .376 | .420 | .309 | .588 | 25 / 41 / 57 | 262.5 K |
| | | M2 k8 | .503 / .380 | .334 | .381 | .427 | .323 | | 0.4 | 364 |
| | | M7 pca32 both | .530 / .390 | .342 | .394 | .447 | .328 | | 0.6–0.8 | 696 |

(π0.5-spatial ms/q at concurrency 4 are inflated by the spatial cells' timing pass sharing bandwidth with the l10 jobs; the l10 rows are the cleaner numbers. B0-current at in-run timing is 1.3–8.3 ms.) Every combination is .18–.24 below B0 at the same hit rate. The vision-free combination is within .005 of the vision one on GR00T and .024–.036 behind on π0.5 at p=.5, at 1/700 of the entry size and 1/20–1/100 of the query time. The M7-pca32 middle ground gains nothing over M2 today (it needs the M/synthesis fixes of §2.3).

### 2.6 Confidence (owner)

AURC by signal, step ≥ 1, mean over the 4 cells of each regime (`opt` = ordering by the true err):

| method | signal | fresh AURC | stale AURC |
|---|---|---|---|
| M1 kmean5 | opt | .134 | .282 |
| | own = −c0/s_c − disp/s_a (fresh) / −w0/s_w − disp/s_a (stale) | **.188** | **.364** |
| | −c0 alone (continuity of the top-1) | .190 | .396 (stale: mildly anti-informative vs own) |
| | −d0 (state L2) / −w0 (window L2) alone | .208 / .260 | .390 / .388 |
| | −disp alone | .199 | .414 |
| | z(−c0)+z(−d0)+z(−disp) | .182 | .365 |
| M8 mean5 | opt | .189 | .255 |
| | own = B0 fused score | .284 | **.338** |
| | −cont_act (continuity of the returned action) | **.216** | .352 |
| | −disp5 | .249 | .373 |
| | margin (top1−top2) / −margin | .395 / .309 | .625 / .393 (margin anti-informative, as B found) |
| | z(b0)+z(−disp5)+z(−cont) | .216 | .332 |
| | z(−disp5)+z(−dist_rs) | .242 | .331 |
| M2 k8 | own = cw+ca | .272 | .352 |
| M7 pca32 both | own (−c0−disp fresh; −d0−disp+g stale) | .205 | .351 |
| | −c0 in the stale regime | | .560 (spurious, as predicted) |
| M3 HMM | lev (log predictive evidence) | .216 | .413 |
| | mass5 / −entropy | .373 / .368 | .713 / .698 (anti-informative) |
| M6 on B0 | own (3-term) | .326 (B0 .408) | .394 (B0 .415) |

Findings: (i) in the **fresh** regime the continuity residual is the signal (M1 own .188 vs opt .134; even M8 is best ranked by the continuity of its returned action); adding dispersion and state helps by ≤ .006. (ii) In the **stale** regime observation-side signals rank best — B0's fused score on the big library (.338), or z(−disp5)+z(−dist_rs) (.331, no vision needed), M1/M2's window distance + dispersion (.352–.364) — and any continuity term must be removed (M7 −c0: .560; M1 stale=cont AURC .386). The stale gap to `opt` (.08–.10) is twice the fresh gap. (iii) Posterior concentration of the HMM is anti-informative after a HIT (risk@30 .92 on π0.5-spatial): the belief sharpens exactly when the robot is stuck and the same rows keep winning. (iv) **Scale compatibility across regimes** (pooled test, one threshold for both regimes, weights p/(1−p), §D of `s03_regime.txt`): M1's own confidence has the same form in both regimes and its stale values are legitimately lower (median −3.7…−4.4 vs −2.1 fresh), so one threshold accepts fresh decisions first — pooled AURC with the raw scores (.208–.235 at p=.3) is *better* than regime-wise rank normalization (.233–.253); the accepted 30 % contains 7–20 % stale decisions when their weight is 30 %. Pairing M1 with **M8's B0-score confidence is broken**: the 0.98-scale score outranks every M1 value, the accepted set at any threshold is 100 % stale, and pooled AURC is .35–.36 vs .22–.24 rank-normalized. So one threshold can serve both regimes only if the stale-regime confidence is expressed on the same "distance / dispersion" scale (M2/M7-style), or both are mapped to a predicted-err scale (§5).

### 2.7 Other cross-method facts

- **B0's own selection is what fails in the stale regime, not only the library**: keeping the previous library chunk running (R_tail in cache cells = replan-10 from the hit) beats B0's next lookup in 3/4 cells (−.048 / −.006 / −.084 / −.018).
- **Track/stay/switch**: M1 after a MISS switches 69–82 % (run length 1.2–1.4), closer to the oracle (54–60 %, 1.6–1.8) than B0 (14–35 %, 2.6–8.0); its err after a "stay" is .33–.42 vs B0's .58–.73. In the stale regime every state-based method stays 25–35 % with err .64–.96 (the robot is not moving, kNN returns the same row); M8 stays 22–34 % with err .59–.77; the oracle itself stays 20–34 % there with err .26–.52. "Stay" in the stale regime is a property of the state, not of the selector.
- **Drift is measurable and predictive**: the M1 window distance to the nearest library window exceeds 3× the library's own 1-NN median for 19–35 % of stale decisions (p90 5.3–8.0 vs 3.6–4.5 fresh); those decisions have err .71–.99 vs .41–.47 for the rest, and they are 41–50 % of the decisions of failed cache-arm episodes vs 2–6 % of successful ones.

## 3. Failure profile of the current best per regime

**Fresh (M1 kmean5, 4 inf cells, step ≥ 1: err .259 / .275 / .248 / .265).**
- Gripper transitions: 7–10 % of decisions, err .46–.60 (steady .23–.25), grip mismatch .22–.44 (oracle .06–.14 → real headroom, largest on GR00T-l10), 14–23 % of the summed error. The 25 worst decisions per cell are almost all `grip_mis = 1` cases where the tail forecasts no flip but the teacher flips (err 1.5–2.3): continuity cannot see an imminent flip; all 10 candidates near the tail share the wrong sign (explain: `x_c0` .18–.26, i.e. a *good* continuity match).
- Late third: .27–.32 vs early .22 (share of error .34–.39); mid third worst on spatial (.29).
- Failed inf-arm episodes: 3–28 % of decisions, err .33–.58 (share .07–.33), e.g. π0.5-spatial task 4 episodes 230/240 (bowl-in-drawer) dominate the worst list.
- Tasks: spread .10–.14 between best and worst (π0.5-sp t2 .194 vs t0 .299; GR00T-l10 t5 .212 vs t7 .324).
- Coverage: 49–67 % (π0.5) / 99–100 % (GR00T) of decisions have no candidate within the floor; on those M1 is .31–.34 (π0.5) — the library is the ceiling for the rest.
- Confidence deciles are monotone and steep (π0.5-sp d0 .51 → d9 .13; grip .09 → .00), so a 70 % acceptance keeps err ≤ .21–.23.

**Stale (M8 mean5, 4 cache cells, step ≥ 1: err .536 / .484 / .460 / .492).**
- Failed cache-arm episodes carry 58–80 % of the summed error (51–71 % of decisions, err .55–.72); on episodes B0 completed the stale err is .33–.35 — within .03 of the fresh-regime err of the same method. The "stale penalty" is mostly the states of runs that were already lost.
- Drifted states (window distance > 3× library 1-NN): err .71–.99 (§2.7); these are also where B0's confidence stays at .98 (explain: cos_v0 .997, cos_v1 .988 on a stuck arm with phase error .5 — saturated vision says "similar", the picked library step is at progress .8 while the query is at .26).
- Late third .59–.61 vs early .33–.36; gripper transitions 12–19 % of decisions with err .60–.77 and mismatch .35–.39 (share of error 15–27 %).
- Tasks: spread .22–.38 (π0.5-sp t0 .706 / t9 .659 vs t2 .322; GR00T-l10 t9 .640 vs t8 .369); the worst tasks coincide with the highest failed-episode shares.
- Confidence deciles: d0–d3 err .76–.86 with grip mismatch .3–.5, d4+ .48 → .21 — the fused score does separate the lost states, but only in the lowest 3–4 deciles.

## 4. Risks the offline harness cannot see, and cheap offline proxies

1. **Closed-loop compounding of continuity.** After a HIT the next query's tail is the library chunk, so a chain of hits is an open-loop replay along one library trajectory; the harness feeds the *trace's* executed chunk, never the method's own pick, so stickiness is invisible. Proxy: an offline "self-fed" rollout on the trace — replace `prev_a_exec` by the method's own previous synthesized chunk for a run of k HITs while keeping the trace observation (states do not respond, but the tail does), and report err growth vs run length; plus the R_tail-in-cache result (§2.7) as an upper-level sanity check. The real check is closed loop (§5).
2. **Regime mixing / state distribution shift.** Cache-arm states are B0's; a better cache visits fewer lost states (the succ-only bound is .05–.20 lower). Proxy: report stale numbers on both all and success-only cache episodes (done above) and weight by the *new* method's expected failure rate once one closed-loop run exists.
3. **Confidence scale across regimes.** Shown in §2.6: two confidences of different form cannot share a threshold. Proxy: library-LOEO calibration of every signal to predicted err (isotonic or 10-bin), then evaluate the pooled AURC/risk at fixed thresholds with the weighted pooling used here.
4. **Gripper flips.** Offline err punishes a flip by ≈ 2/σ_g per step; closed-loop a wrong flip is a dropped object. Proxy: report grip mismatch on transition steps at the accepted coverage (already in breakdown) and a transition-aware veto in the confidence (M6's flip10 idea, re-fitted).
5. **Batch / render drift.** π0.5 queries (Sep, timan107) vs bpool_cs (Aug, weilandserver) vs current (Apr). Both M8 and M2 use bpool_cs, so the vision-vs-state contrast is drift-free, but the size of the π0.5 vision gain could move with a same-batch library. Proxy: the M9 task-centring recovered ≤ .01, so drift is not dominant; a step-0 paired check on the tok subsample (same inits) would settle it cheaply.
6. **Timing.** Concurrency-4 numbers overstate bandwidth-bound methods (M8) and the spatial cells; ranking of the cheap methods (< 1 ms) is robust, absolute numbers for deployment need a concurrency-1 pass.

## 5. Recommendations for round 2

**Refine (with concrete changes)**
1. **M1 chunk_cont** — freeze as the after-MISS selector: big library, α = 0.25, kernel-mean-5 (try k = 8 and the kernel width), stale branch = window; drop α = 1, β, stale = cont, med3, current library. Add: (a) a gripper-flip-aware candidate set (candidates whose head gripper differs from the tail's last gripper are allowed but weighted by a library-fitted flip prior given step/progress), targeting the 14–23 % of error on transitions; (b) confidence = −c0/s_c − disp/s_a (keep) with a library-LOEO mapping to predicted err.
2. **M2 state_window** — the vision-free after-HIT / step-0 branch: k = 8 mean (try 12–16 and a kernel), m = 3; add the same predicted-err calibration; test velocity (Δrs) as an explicit feature and a per-task scale.
3. **M7 cascade → "vision over a wide set"**: PCA-32 (or 64) keys of both cameras, task-centred cosine z-sum, scored over *all* task candidates or M ≥ 128 (not 16), fused with the state window, **mean-5/8 synthesis** instead of med3, regime-aware confidence without continuity after a HIT. This is the deployable version of M8 (target: M8's .526/.480 on π0.5 stale at ≤ 1 ms and ≤ 512 B/entry). Kill if it does not beat M2 by ≥ .03 on π0.5 stale cells or costs > .01 on GR00T.
4. **M8** — keep only as the accuracy reference for #3 (raw keys); one variant "M8 with PCA-32 keys, mean-8" to confirm the representation-size result outside the cascade.
5. **M6 consistency confidence** — port to M1/M2/M7: fresh = (−cont, −disp), stale = (−window dist, −disp, +vision z), no B0 score, no veto; deliver the LOEO-calibrated predicted-err version so one threshold works (§2.6 iv).

**Drop**: M3 HMM (all variants), M5 shortlist re-rank, M4 as a standalone (its synthesis rule lives on in M1/M2/M7), M9c as a method, M7 raw/v1only/b0fused variants, M1 current-library and β variants, the median-3 synthesis.

**New directions suggested by the data**
- **Flip-aware synthesis / "transition detector"**: the residual fresh error is gripper transitions the tail cannot forecast; a library-fitted P(flip | step-in-task, state window, tail) used to (i) veto or (ii) choose between the flip / no-flip candidate clusters instead of averaging them (mean-k across a flip boundary produces a hovering gripper).
- **Predicted-err calibration as the confidence** (T1): map each regime's signals to err via library LOEO (isotonic / binned), which also gives a single deployable threshold and lets the stale branch reject the drifted states (window distance > 3× 1-NN median already flags err .7–1.0 decisions).
- **Success-aware library**: the stale regime's error sits on lost states; weight or filter big-library rows by progress/success and test whether the stale-branch kNN restricted to "on-track" rows lowers err on successful episodes without hurting recovery.
- **HIT continuation policy** (systems, not retrieval): after a HIT, continuing the same library chunk (replan-10) beats B0's re-retrieval offline by .006–.084; a cheap variant is "continue while the window distance stays below τ, re-retrieve otherwise".

**Closed-loop check to propose to the owner now**: yes — the fresh-regime result is large (err halves, AURC halves, 268 B/entry, < 0.5 ms) but its closed-loop value hinges on the untestable stickiness of continuity chains. A single A/B on one suite (π0.5-spatial or GR00T-spatial, 500 episodes, timan107): B0 vs (M1 kmean5 after MISS + M2 k8 after HIT + calibrated confidence at matched hit rate), with the replan-10 reference arm (R_tail semantics) as the ceiling. This also produces the first cache-arm trace whose stale states come from the new method (§4.2), which the next offline rounds need.
