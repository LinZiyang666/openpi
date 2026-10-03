# R10 offline analysis: in-library corrector training (LOEO vs PAIRS), library data only

Analyst: opus, 2026-10-02. Code: `tools/` (core.py, run_offline.py, run_regime.py, analyze.py, diag_metric_optimism.py,
test_core.py: 8 tests pass). Raw per-row outputs: `out/kfold/*.npz`, `out/regime/*.npz`; tables: `out/tables.md`,
`out/per_cell.md`, `out/summary.json`, `out/regime_summary.json`.

## 0. Data boundary

Every array comes from `offline_search_store/library/<model>_<suite>/{bpool_cs (π0.5), bpool_all (GR00T)}` (B pool) plus
the r01 PCA basis of those same libraries. `core.assert_library_path` refuses `trace_runs/os_closed_loop/**` and R8-derived
paths. Nothing from the LIBERO test set A, no closed-loop run, no R8 decision table was opened. The sol code read for
alignment (`rounds/r10/{data,method,build,train}.py`) contains constructor specs only; I did not open its run roots.

## 1. Protocol

* Subsets: identical to sol's (`default_rng(20261002 + task).permutation(episodes)`, first size/10 per task; checked
  `np.array_equal` against `rounds/r10/subsets/*.npy` for sizes 50 and 500 in all four cells). Failed episodes kept.
* 5-fold by episode: an episode's fold = its permutation position mod 5. At size 50 each fold holds out one episode per
  task (training fold = 4 episodes / task); at 500, 10 per task.
* Cache, refitted on the training fold only (deployment-faithful for a new episode): PCA-64 per camera (`awm.pca_fit`, the
  recipe sol uses for subsets; the stored r01 basis at 500), per-task joint metric `awm.fit_metric` (nn 3, lam .1), early
  metric for step 0, top-16, kernel with kref 5 (size 50) / 8 (≥100) — sol's rule. Retrieval regime 2 (a look after a
  cache HIT, no continuity term) unless stated; regime 1 in §7.
* Held-out evaluation: each held-out row is served from the training-fold rows ("full"), and from a random half of the
  training episodes ("half", a sparser-library stress test). Error = RMS over 10 steps × motion channels 0–5 in units of
  the 500-library action σ, against the row's own stored policy chunk. Corrections at blend .5 (spec) unless stated.
  CIs: 95 % episode-cluster bootstrap (2000 resamples) of the row-mean difference.
* Head: R9 recipe exactly (fable round 2): inputs PCA-128 + rs[:8] + served[:10,:7]/σ + min(step,120)/120 + task one-hot,
  standardize + clip 8, 384 cos RFF (seed 0), weighted ridge α 100, one head per task, target = motion residual / σ.
  `core.Head` = fable's `fit_head` with chunked normal equations (tested equal) plus an explicit total-weight argument.

Training schemes (all equal weight per episode):

| name | training rows / pairs |
|---|---|
| `loeo` | each row once; served chunk = cache synthesis from other episodes (deployed metric, fitted on all training rows) |
| `xfit` | as loeo, but the synthesis uses a metric refitted without the row's inner group of episodes (4 inner groups) |
| `loeo_aug`, `xfit_aug` | + two more syntheses per row from a random 50 % / 25 % of the other episodes (drift augmentation), ⅓ weight each |
| `pair_r50_c16` | partners = other-episode rows within the per-task median LOEO 16-NN distance, ≤16 nearest; input chunk = partner's chunk, target = own − partner; each anchor row's pairs share weight 1 |
| `pair_r95_c64` | radius = per-task 95th pct of LOEO 16-NN distance, ≤64 nearest, row-normalized |
| `pair_all_c64` | 64 uniformly random other-episode rows of the task ("all points"), row-normalized |
| `pair_r95_c64_raw` | same pairs as r95_c64 but weights normalized to the pair count (what fable's `fit_head` does with per-pair weights): α 100 becomes ~n_pairs/n_rows times weaker |

## 2. Main result (full library, blend .5): relative change of held-out motion error vs no corrector, % [95 % CI]

`*` = CI excludes 0. "new-ep d1 ratio" = median over held-out rows of (distance to the nearest training row) / (median
in-library LOEO nearest-neighbour distance), both in the deployed metric.

| cell | size | new-ep d1 ratio | none e10 | loeo | loeo_aug | xfit | pair_r50_c16 | pair_r95_c64 | pair_all_c64 | pair_r95_c64_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| groot l10 | 50 | 1.36 | 0.496 | +0.7 [-0.6,+2.0] | +0.5 [-0.7,+1.8] | +1.1* [+0.1,+2.3] | +1.0* [+0.3,+1.8] | +2.5* [+1.4,+3.6] | +4.0* [+2.7,+5.2] | +13.5* [+10.3,+16.9] |
| groot l10 | 200 | 1.07 | 0.429 | +0.1 [-0.7,+0.8] | +0.3 [-0.4,+0.9] | +0.3 [-0.4,+1.0] | +1.8* [+1.2,+2.4] | +2.6* [+2.0,+3.2] | +5.2* [+4.6,+5.9] | +7.8* [+6.4,+9.4] |
| groot l10 | 500 | 1.03 | 0.389 | -0.8* [-1.2,-0.4] | -0.3 [-0.7,+0.0] | -0.4* [-0.8,-0.0] | +2.7* [+2.2,+3.2] | +3.3* [+2.8,+3.9] | +7.1* [+6.5,+7.7] | +5.3* [+4.6,+5.9] |
| pi05 l10 | 50 | 1.72 | 0.449 | -0.4 [-1.3,+0.6] | -0.7 [-1.6,+0.3] | -0.0 [-1.1,+1.2] | -1.1 [-2.5,+0.5] | +0.6 [-1.3,+2.4] | +2.6* [+0.4,+4.7] | +13.6* [+9.1,+18.5] |
| pi05 l10 | 200 | 1.11 | 0.335 | -2.5* [-3.0,-2.0] | -1.9* [-2.4,-1.4] | -1.6* [-2.1,-1.0] | +0.8* [+0.4,+1.3] | +2.0* [+1.5,+2.5] | +6.5* [+5.9,+7.2] | +7.1* [+5.9,+8.4] |
| pi05 l10 | 500 | 1.03 | 0.304 | -3.3* [-3.7,-2.8] | -2.8* [-3.2,-2.4] | -2.8* [-3.2,-2.4] | +0.8* [+0.4,+1.2] | +2.0* [+1.5,+2.5] | +7.7* [+7.1,+8.3] | +4.1* [+3.4,+4.8] |
| groot spatial | 50 | 2.01 | 0.519 | -0.2 [-1.0,+0.5] | -0.8 [-2.0,+0.2] | -0.2 [-1.2,+0.8] | -1.3* [-2.8,-0.1] | -1.0 [-2.9,+0.5] | -0.7 [-2.6,+0.9] | +7.3* [+3.7,+11.2] |
| groot spatial | 200 | 1.20 | 0.432 | -0.5 [-1.1,+0.2] | -0.3 [-0.9,+0.4] | +0.0 [-0.6,+0.7] | +1.1* [+0.5,+1.7] | +1.9* [+1.0,+2.7] | +3.3* [+2.4,+4.2] | +11.1* [+9.3,+12.9] |
| groot spatial | 500 | 1.07 | 0.395 | -1.0* [-1.4,-0.6] | -0.8* [-1.2,-0.4] | -0.6* [-1.0,-0.2] | +1.5* [+1.1,+2.0] | +1.7* [+1.2,+2.2] | +3.7* [+3.1,+4.3] | +5.9* [+5.0,+6.8] |
| pi05 spatial | 50 | 2.66 | 0.515 | -1.4* [-2.5,-0.3] | -1.6* [-2.6,-0.4] | -0.6 [-1.7,+0.6] | -1.4* [-2.5,-0.1] | -0.8 [-2.1,+0.7] | -0.0 [-1.5,+1.5] | +11.5* [+7.5,+16.4] |
| pi05 spatial | 200 | 1.25 | 0.389 | -3.1* [-4.0,-2.1] | -2.8* [-3.6,-2.0] | -2.5* [-3.3,-1.6] | -1.1* [-1.6,-0.5] | -0.3 [-0.9,+0.4] | +1.3* [+0.6,+2.0] | +8.1* [+6.4,+9.9] |
| pi05 spatial | 500 | 1.07 | 0.346 | -4.7* [-5.2,-4.3] | -4.2* [-4.7,-3.8] | -4.1* [-4.6,-3.7] | -1.0* [-1.5,-0.5] | -0.6* [-1.1,-0.1] | +2.1* [+1.5,+2.6] | +2.8* [+1.9,+3.7] |

Reading:
* LOEO vs best PAIR (`pair_r50_c16`), paired difference: at 200/500 LOEO is better in all 8 cell-sizes by 1.6–4.0
  points (all CIs exclude 0; e.g. π0.5 L10-500 −3.3 vs +0.8, GR00T L10-500 −0.8 vs +2.7). At 50 they tie: differences
  −0.7 … +0.3 points (CIs include 0) except GR00T Spatial-50 where PAIR is 1.1 points better (CI −2.4 … −0.0).
* PAIRS gets worse as the radius widens / partners become arbitrary (r50_c16 < r95_c64 < all_c64) and is clearly harmful
  (+2.8 … +13.6 %) when the weights are normalized to the pair count.
* Cross-fitting the synthesis metric (`xfit`) never helps (0.0–0.6 points worse than `loeo`); drift augmentation
  (`loeo_aug`) costs 0.2–0.6 points in the full condition.
* The corrector itself is weak: best case π0.5 ≥200 episodes (−2.5 … −4.7 %); GR00T −1.0 … +0.7 %; 50-episode
  libraries −1.4 … +0.7 %.

## 3. Sparser-library stress ("half": candidates = random half of the training episodes)

| cell | size | none e10 | loeo | loeo_aug | xfit | pair_r50_c16 | pair_r95_c64 | pair_all_c64 | pair_r95_c64_raw |
|---|---|---|---|---|---|---|---|---|---|
| groot l10 | 50 | 0.540 | -2.1* | -4.0* | -3.3* | -3.4* | -2.9* | -1.5 | +6.2* |
| groot l10 | 200 | 0.455 | -1.9* | -2.4* | -1.9* | -1.2* | -0.9* | +1.6* | +3.4* |
| groot l10 | 500 | 0.415 | -2.0* | -2.4* | -2.1* | -0.3 | -0.2 | +3.3* | +1.2* |
| pi05 l10 | 50 | 0.483 | -2.9* | -4.3* | -3.3* | -4.6* | -3.5* | -1.7 | +7.3* |
| pi05 l10 | 200 | 0.365 | -4.4* | -5.1* | -4.4* | -3.3* | -2.8* | +1.3* | +0.8 |
| pi05 l10 | 500 | 0.331 | -4.5* | -5.2* | -4.6* | -2.8* | -2.5* | +2.6* | -1.3* |
| groot spatial | 50 | 0.559 | -2.6* | -5.1* | -3.3* | -5.2* | -5.1* | -4.7* | +1.2 |
| groot spatial | 200 | 0.457 | -1.7* | -2.7* | -2.1* | -1.8* | -1.5* | -0.1 | +6.4* |
| groot spatial | 500 | 0.418 | -2.4* | -2.8* | -2.3* | -1.2* | -1.4* | +0.5 | +1.9* |
| pi05 spatial | 50 | 0.560 | -3.8* | -5.8* | -4.4* | -5.7* | -5.4* | -4.8* | +4.4 |
| pi05 spatial | 200 | 0.428 | -4.6* | -6.5* | -5.4* | -5.7* | -5.6* | -4.2* | +0.7 |
| pi05 spatial | 500 | 0.377 | -5.9* | -6.9* | -6.2* | -4.8* | -5.0* | -2.6* | -2.9* |

When the served chunk is synthesized from a sparser library (larger kernel-mean bias), every reasonable corrector helps,
and at 50 episodes the PAIR variants (r50/r95) and LOEO-aug are 1–2.6 points better than plain LOEO. The augmentation
result is partly home advantage (it trains on the same kind of perturbation). At ≥200, LOEO(-aug) stays best.

## 4. Mechanism: why more pairs do not help

| cell | size | loeo b*/slope | pair_r50_c16 b*/slope | pair_r95_c64 b*/slope | pair_all_c64 b*/slope | pair_r95_c64_raw b*/slope | loeo e10 @1.0 vs @.5 |
|---|---|---|---|---|---|---|---|
| groot l10 | 50 | 0.20 / -0.04 | 0.21 / -0.16 | 0.19 / -0.19 | 0.16 / -0.19 | 0.07 / -0.44 | 0.536 vs 0.499 |
| groot l10 | 500 | 0.38 / -0.04 | 0.17 / -0.27 | 0.18 / -0.30 | 0.10 / -0.33 | 0.16 / -0.71 | 0.407 vs 0.385 |
| pi05 l10 | 50 | 0.27 / -0.02 | 0.40 / -0.15 | 0.32 / -0.18 | 0.26 / -0.18 | 0.11 / -0.38 | 0.474 vs 0.447 |
| pi05 l10 | 500 | 0.52 / -0.02 | 0.28 / -0.23 | 0.25 / -0.27 | 0.12 / -0.30 | 0.19 / -0.60 | 0.306 vs 0.294 |
| pi05 spatial | 500 | 0.56 / -0.01 | 0.35 / -0.21 | 0.34 / -0.25 | 0.24 / -0.27 | 0.22 / -0.58 | 0.343 vs 0.330 |

(all 11 cell-sizes in `out/tables.md`.) b* = least-squares optimal blend on the held-out rows; slope = response of the
correction to a small perturbation of the input chunk (−1 = the head throws the served chunk away and substitutes its
own estimate; 0 = correction independent of the served chunk).

* The pair target a_i − a_j moves one-for-one with the partner chunk a_j, so a pair-trained head learns to partly
  *replace* the input chunk (slope −0.15 … −0.33; −0.3 … −0.7 with pair-count weights). At serving the input is a
  16-member kernel mean, which is far smoother than a single partner chunk; the head treats its legitimate content as
  partner noise and over-corrects (b* falls to 0.1–0.3).
* The many pairs of one anchor share one target row; the effective sample size is still the number of rows. With
  weights normalized to the pair count the ridge is 15–60× weaker for the same α, and the head overfits
  (`pair_r95_c64_raw`, the worst arm everywhere at 50 and 200).
* LOEO heads barely depend on the input chunk (slope ≈ −0.02) and learn a state/step-dependent bias of the kernel mean.
  b* ≈ .45–.56 for π0.5 at ≥200 (blend .5 is right), .2–.38 elsewhere (blend .5 is somewhat too strong; blend 1.0 is
  worse than .5 in every cell).

## 5. Drift: error change by distance of the query to the library (full library, pooled over cells)

Bin = held-out row's nearest-training-row distance / median in-library LOEO nearest distance (deployed metric). Cell
bin means averaged over cells.

Size 50 (4 cells):

| d1 bin | rows | none e10 | loeo | loeo_aug | pair_r50_c16 | pair_r95_c64 |
|---|---|---|---|---|---|---|
| [0.75,1) | 228 | 0.342 | -6.2% | -5.0% | -2.2% | +0.3% |
| [1,1.25) | 1126 | 0.380 | -2.6% | -2.0% | +0.3% | +2.9% |
| [1.25,1.5) | 1680 | 0.414 | -0.5% | -0.3% | +0.9% | +2.7% |
| [1.5,2) | 2479 | 0.482 | -0.2% | -0.6% | -0.1% | +1.0% |
| [2,3) | 1967 | 0.556 | +1.4% | +0.8% | -0.2% | +0.5% |
| [3,inf) | 710 | 0.711 | +0.3% | -0.8% | -4.3% | -4.9% |

Sizes 200 + 500 (8 cells):

| d1 bin | rows | none e10 | loeo | loeo_aug | pair_r50_c16 | pair_r95_c64 |
|---|---|---|---|---|---|---|
| [0,0.75) | 4059 | 0.293 | -4.8% | -3.8% | +2.6% | +3.9% |
| [0.75,1) | 38182 | 0.317 | -3.2% | -2.4% | +0.7% | +2.1% |
| [1,1.25) | 46167 | 0.360 | -2.4% | -1.9% | +0.5% | +1.4% |
| [1.25,1.5) | 14780 | 0.425 | -1.7% | -1.6% | +0.6% | +0.8% |
| [1.5,2) | 7082 | 0.506 | +0.0% | -0.1% | +2.0% | +1.7% |
| [2,3) | 2857 | 0.640 | +1.6% | +0.2% | +1.2% | +0.4% |
| [3,inf) | 809 | 0.751 | +4.3% | +4.2% | +3.2% | +2.4% |

* The corrector helps only close to the library: LOEO's gain is −3 … −6 % below 1× and gone by 1.5×; beyond 2× all
  schemes are neutral-to-harmful at ≥200 episodes. In the far tail at 50 episodes PAIRS is the only scheme with a gain
  (−4 … −5 % on 10 % of the rows), consistent with it acting as a parametric "replace the bad chunk" regressor; this
  is the one regime where PAIRS beats LOEO.
* A new episode is already farther from a small library than library rows are from each other in the deployed metric:
  median ratio 1.36–2.66 at 50 episodes, 1.07–1.25 at 200, 1.03–1.07 at 500. `diag_metric_optimism.py` shows this is the
  per-task whitening fitted on the library rows themselves (fitting the metric with the held-out rows included brings the
  ratio back to ~0.98 and lowers the held-out served error 0.430 → 0.358 for π0.5 L10-50, 0.495 → 0.427 GR00T L10-50;
  PCA in-sample effects are negligible). Hence in-library LOEO pseudo-queries are optimistic about retrieval quality at
  small sizes, and every library-LOEO-calibrated distance (pair radius, AWM confidence z-scores) under-states the
  distance of real queries. Cross-fitting the synthesis metric (`xfit`) removes this optimism in training but did not
  improve the held-out result (§2), so no trainer change is needed for it.
* Closed-loop cache rollouts move further from the library than these held-out B episodes; the offline gains above are
  therefore an upper bound for the bulk of decisions.

## 6. Head capacity at 50 episodes (pool mode, same folds)

| cell (50 ep) | per-task LOEO α100 (spec) | per-task LOEO α1000 | one pooled LOEO head | per-task PAIR r50 | pooled PAIR r50 |
|---|---|---|---|---|---|
| groot l10 | +0.7 | -0.3 | +1.3* | +1.0* | +2.6* |
| pi05 l10 | -0.4 | -0.8* | +1.0 | -1.1 | +1.0 |
| groot spatial | -0.2 | -0.7* | +0.2 | -1.3* | -0.1 |
| pi05 spatial | -1.4* | -1.6* | -1.0 | -1.4* | -1.3 |

A single task-agnostic head (task one-hot input, all 10 tasks pooled) is worse than per-task heads even at 50
episodes. A 10× stronger ridge is slightly better (−0.1 … −1.0 points vs α 100) and makes blend 1.0 harmless
(b* .29–.71), i.e. the α 100 head at 50 episodes is mostly noise that the .5 blend partially absorbs. Not a spec change
for R10 (hyper-parameters are fixed), only context.

## 7. Retrieval regime used by sol's trainer (`rounds/r10/train.py`)

sol's trainer poses every library row with `prev_hit=False`, i.e. AWM's fresh-after-MISS regime 1 (score =
d / median d + 0.5 · continuity / s_c, continuity measured against the previous row's own policy tail). At deployment
regime 1 requires `prev_hit=False`, i.e. a look immediately after a policy call; looks after a cache HIT use regime 2
(see the first bullet below for how rare regime 1 is in R10's arms). sol's PAIR also normalizes weights to the pair count (fable `fit_head` with per-pair weights, cap 16, so
the effective ridge is ~15× weaker) and draws partners by regime-1 score within a pooled 95th-percentile radius.
`run_regime.py` reproduces both and evaluates held-out episodes served in regime 2 and in regime 1:

| cell | size | held-out regime | none e10 | loeo_r2 | loeo_r1 | loeo_mix | sol_pair | pair_r1_rn | pair_r2_r50 | sol pairs/row |
|---|---|---|---|---|---|---|---|---|---|---|
| groot l10 | 50 | r2 | 0.496 | +0.7 | +1.2* | +0.4 | +5.7* | +0.8 | +1.0* | 15.5 |
| groot l10 | 50 | r1 | 0.405 | +2.0 | -2.2* | -0.7 | +11.5* | +3.4* | +5.1* | 15.5 |
| groot l10 | 500 | r2 | 0.389 | -0.8* | +0.6* | -0.7* | +1.9* | +0.7* | +2.7* | 15.4 |
| groot l10 | 500 | r1 | 0.310 | +2.9* | -4.3* | -2.7* | +1.8* | -0.9* | +8.1* | 15.4 |
| pi05 l10 | 50 | r2 | 0.449 | -0.4 | -0.1 | -0.8 | +6.1* | -0.8 | -1.1 | 15.6 |
| pi05 l10 | 50 | r1 | 0.374 | -0.3 | -2.2* | -1.8* | +12.4* | +1.7* | +2.2* | 15.6 |
| pi05 l10 | 500 | r2 | 0.304 | -3.3* | -0.8* | -2.3* | +1.5* | +0.2 | +0.8* | 15.4 |
| pi05 l10 | 500 | r1 | 0.306 | -2.1* | -8.4* | -7.4* | -5.8* | -7.3* | -4.2* | 15.4 |
| groot spatial | 50 | r2 | 0.519 | -0.2 | +0.5 | -0.2 | +3.4* | -1.3 | -1.3* | 15.4 |
| groot spatial | 50 | r1 | 0.436 | +0.3 | -2.2* | -1.8* | +8.9* | +1.5 | +1.6 | 15.4 |
| groot spatial | 500 | r2 | 0.395 | -1.0* | +0.4* | -0.9* | +3.2* | +0.5* | +1.5* | 15.3 |
| groot spatial | 500 | r1 | 0.333 | +1.5* | -4.8* | -3.3* | +4.3* | -1.0* | +4.5* | 15.3 |
| pi05 spatial | 50 | r2 | 0.515 | -1.4* | -0.6 | -1.4* | +6.5* | -1.1 | -1.4* | 15.4 |
| pi05 spatial | 50 | r1 | 0.443 | -2.3* | -3.3* | -3.2* | +10.9* | +0.7 | +0.9 | 15.4 |
| pi05 spatial | 500 | r2 | 0.346 | -4.7* | -2.3* | -3.8* | +1.6* | -1.2* | -1.0* | 15.2 |
| pi05 spatial | 500 | r1 | 0.322 | -3.2* | -6.7* | -6.1* | -0.3 | -4.5* | -2.1* | 15.2 |

(relative change of held-out e10 at blend .5 vs no corrector, %; `*` = episode-bootstrap CI excludes 0; "r2" = look
after a cache HIT, "r1" = look right after a policy call; served error itself is much lower in r1 because the
continuity term sees the policy's own previous tail.)

* Which regime does the corrector actually see? In pure cache A every look at step ≥ 1 follows a HIT (regime 2). In G
  (`--os-policy-tail --os-policy-tail-blocks 1`) a policy call at decision t is followed by a policy-tail step at t+1
  that the plugin commits as a HIT (`closed_loop/plugin.py`, blind/policy-tail commit with `hit=True`), so the next look
  at t+2 again has `prev_hit=True` → regime 2. Regime 1 only arises when the tail is invalid. So for R10's arms the
  relevant rows are the "r2" rows.
* On r2, LOEO trained in regime 2 is best or tied in all 8 cell-sizes; sol's regime-1 LOEO loses 1.4–2.5 points at
  500 episodes (π0.5 L10 −0.8 vs −3.3, GR00T L10 +0.6* vs −0.8*, GR00T Sp +0.4* vs −1.0*, π0.5 Sp −2.3 vs −4.7) and
  is significantly harmful for GR00T L10 at both sizes. Matching the regime matters as much as LOEO-vs-PAIR.
* sol's PAIR (regime-1 partners, pooled p95 radius, cap 16, pair-count weights; 15.2–15.6 pairs per row) is harmful on
  r2 in every cell (+1.5 … +6.5 %); with row-normalized weights the same pairs are roughly neutral (−1.3 … +0.8 %).
  This matches sol's own 3-fold CV for π0.5 L10-50 (PAIR blend-.5 MSE 0.240 vs baseline 0.181; LOEO 0.176).
* Training on both regimes (`loeo_mix`) is a reasonable hedge if regime 1 were frequent; for R10's arms regime 2 alone is
  the faithful choice.
* Note: sol's CV fixes the full-subset PCA/metric while holding out episodes, which includes the held-out rows in the
  metric fit — the in-sample optimism of §5 (largest at 50 episodes). It is labelled diagnostic-only; do not read its
  absolute errors as deployment errors.

## 8. Breakdowns (full library, blend .5; `analyze.breakdown`)

* Failed library episodes as queries: corrector neutral-to-harmful for GR00T (L10-500 LOEO +1.7 %, PAIR +4.7 %;
  L10-50 LOEO +6.2 %); π0.5 failed-episode rows still gain at 500 (−1.9 %). Successful-episode rows carry the gains.
* Episode phase: π0.5 (500) gains in every third of the episode (−3.0 … −5.2 %); GR00T L10-500 gains shrink over the
  episode (−1.6 % first third, −0.4 % last third); no clear pattern at 50 episodes.
* Step 0 (early metric): the corrector does nothing useful or hurts (GR00T L10-500 LOEO +4.1 %, PAIR +15 %; 500 rows).

## 9. What this offline analysis cannot tell

* Success. The metric is a 10-step motion RMS against one sample of a stochastic policy; part of it is irreducible
  sampling noise, and a 1–5 % change of an average is small next to what decides grasps (timing and alignment at a few
  decisions). Gripper is not corrected and not evaluated.
* Closed-loop distribution. Queries in A rollouts are generated by the cache itself and drift away from library states;
  §5 shows the gain vanishes beyond ~1.5× in-library distance. How far A queries sit is not measurable without A data.
* Guard interaction (G): the corrector changes trajectories, hence when the no-progress guard fires; the decisions right
  after a policy call use regime 1 (§7). Policy-tail blocks and blind steps are not modelled beyond the 10-step chunk.
* Fold sizes: at 50 the training fold has 4 episodes per task (deployment: 5), so absolute errors are slightly
  pessimistic; relative scheme comparisons are paired and unaffected.
* Test initial states (A) differ from B; nothing here says how far A's scenes are from B's.
