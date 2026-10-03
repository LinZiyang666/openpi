# R11 opus: offline LOEO knob curves and value evidence for schedule-type IR knobs

Snapshot 2026-10-02 ~21:30 CDT. Library only:
- R10 nested subsets of `bpool_cs` (π0.5) and `bpool_all` (GR00T);
- whole-episode-out 5-fold replay of the deployed retrieval rule (IR_MODEL.md §3).

No `os_closed_loop` file was read. Two disclosed exceptions use **published arm-level aggregates** only to check the model, never to set a knob:
- R6's ledger table, for the overlap rule (§5);
- R10's REPORT IRs, for the base guard rate (§1).

Full machine-readable output is in `out/curves.json`, `out/arm_grid.json` and `out/value_diag.json`. Every table in `out/OFFLINE_TABLES.md` covers all 12 cell-sizes and all levels.

## 0. Findings in brief

1. **The knob → IR map is reliable on the library and nearly linear in the knob.**
   - For random miss, the closed form `IR = v·(c_v + c_m·(g + (1 − g)·ρ))` and the anchor-by-anchor simulation agree to < .002.
   - The overlap rule (guard first, knob on the rest) reproduced R6's 13 closed-loop "B + random dose" arms within −.003 to +.011 IR, *given* the closed-loop base guard rate.
2. **The only material offline unknown is the closed-loop base guard rate g.**
   - The library replay puts g at .15–.29 on the R11 grid.
   - Compared with the rate implied by R10's published 3-layer IRs (a check only), it is high by .00–.045 on 7 of 8 cell-sizes and low by .02 on GR00T LIBERO-10-50.
   - The guard-rate stress (×.75 / ×1.25) moves realized IR by ±.01 to ±.027 at the proposed levels.
   - The periodic gap cap is 15–45% less sensitive to this than random, because guard calls reset its counter.
   - Expected realized IR: within ±.02 of target for every proposed arm, with π0.5 cells slightly low.
3. **Schedule placement captures the same LOEO policy-vs-cache error as random, within ±8%.**
   - Capture ratio: random 1.00, periodic .92–1.02, periodic plus post-guard tail 1.01–1.06 at the proposed settings.
   - The only offline-visible structural difference is cache-run length. At matched IR, random leaves a longest cache-only stretch of 3.7–6.4 anchors per episode on average (p90 6–10 anchors = 60–100 controls). Periodic caps it at 1–3 anchors and halves-to-eliminates uncovered high-error streaks.
   - Whether that matters for success is a closed-loop question. Open-loop replay cannot show error compounding.
4. **Rejected offline:**
   - **Front-loading:** early anchors carry 0.68–0.95× the mean cache error.
   - **Refractory after guard calls:** post-guard anchors carry 1.02–1.27× the mean.
   - **Per-episode total-budget sigma-delta:** the most IR-robust option, but it starves guard-heavy failing episodes and spaces calls less evenly. Kept as a documented fallback if IR tracking proves to be the problem.

## 1. Base (knob off) from the library replay

| cell-size | episodes | lib success | mean slots | v | guard rate g | base IR | IR succ / failed eps | g stress ×.75 / ×1.25 |
|---|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | 50 | .88 | 58.5 | .5043 | .259 | .187 | .155 / .308 | .193 / .319 |
| GR00T L10-50 | 50 | .88 | 59.1 | .5029 | .291 | .199 | .171 / .305 | .220 / .369 |
| π0.5 Sp-50 | 50 | .92 | 23.2 | .5090 | .151 | .142 | .120 / .269 | .113 / .190 |
| GR00T Sp-50 | 50 | .90 | 24.0 | .5104 | .150 | .141 | .115 / .256 | .103 / .186 |
| π0.5 L10-200 | 200 | .91 | 57.2 | .5042 | .189 | .158 | .137 / .255 | .145 / .232 |
| GR00T L10-200 | 200 | .88 | 58.5 | .5034 | .271 | .191 | .163 / .287 | .208 / .336 |
| π0.5 L10-500 | 500 | .87 | 58.9 | .5037 | .204 | .164 | .138 / .254 | .155 / .253 |
| GR00T L10-500 | 500 | .85 | 59.3 | .5036 | .274 | .192 | .163 / .276 | .205 / .340 |

**Disclosed accuracy check (`r10_check.py`).** R10's published 3-layer IR implies the following closed-loop g, compared with library g:

| cell-size | implied closed-loop g | library g |
|---|---|---|
| π0.5 L10-50 | .214 | .259 |
| GR00T L10-50 | .309 | .291 |
| π0.5 Sp-50 | .133 | .151 |
| GR00T Sp-50 | .118 | .150 |
| π0.5 L10-200 | .172 | .189 |
| GR00T L10-200 | .272 | .271 |
| π0.5 L10-500 | .172 | .204 |
| GR00T L10-500 | .234 | .274 |

**Library-only explanation for part of the gap (`size_sensitivity.py`).** The 50-episode replay serves each held-out episode from 4 of the 5 episodes per task. On the 500 libraries, g falls with library size:

| library | g at 3/task | 4/task | 5/task | 8/task | 16/task | 40/task |
|---|---|---|---|---|---|---|
| π0.5 L10 | .283 | .264 | .250 | .230 | .210 | .204 |
| π0.5 Sp | .117 | .101 | .091 | .073 | .058 | .045 |
| GR00T L10 | .323 | .310 | .302 | .294 | .287 | .274 |
| GR00T Sp | .178 | .155 | .153 | .143 | .125 | .118 |

- **Sparsity bias.** The 4-versus-5 sparsity accounts for .002–.014 of upward bias on the 50-cells. A library-only extrapolation from 3 → 4 episodes per task predicts it well (π0.5 L10: predicted .011, actual .014). It is not applied to the frozen settings: the effect is ≤ .006 IR.
- **The remainder is genuine closed-loop difference.** The GC_dist corrector reduces stalls, and closed-loop failure mix and lengths differ.

## 2. Knob curves (knob → anchor miss fraction f / owner IR), library LOEO

| cell-size | base f / IR | random ρ=.2 | ρ=.4 | ρ=.6 | ρ=.8 | periodic K=4 | K=2 | K=1 | K=.5 |
|---|---|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | .259 / .187 | .409 / .251 | .556 / .314 | .710 / .380 | .853 / .442 | .346 / .225 | .448 / .268 | .583 / .326 | .730 / .389 |
| GR00T L10-50 | .291 / .199 | .431 / .259 | .573 / .320 | .719 / .383 | .857 / .442 | .362 / .230 | .459 / .271 | .578 / .322 | .740 / .392 |
| π0.5 Sp-50 | .151 / .142 | .321 / .216 | .491 / .289 | .666 / .365 | .832 / .436 | .257 / .188 | .371 / .237 | .526 / .305 | .693 / .377 |
| GR00T Sp-50 | .150 / .141 | .320 / .215 | .491 / .289 | .661 / .363 | .832 / .437 | .253 / .185 | .367 / .235 | .519 / .301 | .692 / .376 |
| π0.5 L10-200 | .189 / .158 | .351 / .227 | .511 / .295 | .676 / .365 | .838 / .435 | .297 / .203 | .405 / .250 | .552 / .312 | .712 / .381 |
| GR00T L10-200 | .271 / .191 | .415 / .253 | .558 / .314 | .706 / .377 | .854 / .441 | .342 / .221 | .439 / .263 | .574 / .321 | .732 / .388 |
| π0.5 L10-500 | .204 / .164 | .363 / .231 | .523 / .300 | .683 / .368 | .842 / .436 | .306 / .207 | .414 / .253 | .556 / .314 | .716 / .383 |
| GR00T L10-500 | .274 / .192 | .418 / .254 | .563 / .316 | .709 / .379 | .855 / .442 | .346 / .223 | .440 / .263 | .572 / .320 | .731 / .388 |

**Slopes and reach.**
- Random's IR slope is `c_m·v·(1 − g)` ≈ .30–.37 IR per unit ρ.
- Periodic is convex in K. K → ∞ returns the base. K = 0 is pure policy (.503–.510).
- Both reach every target in [base, .50].

## 3. Target solves and robustness to the unknown closed-loop guard rate

Frozen settings and per-arm predictions are in SPEC §5 and PREDICTION.md. The table below shows, at the proposed levels, the change in realized IR when the closed-loop guard rate is ×.75 or ×1.25 of the library's. The knob setting stays the one solved on the unperturbed library.

| cell-size | target | random | periodic (gap cap) | sigma-delta budget (not proposed) |
|---|---|---|---|---|
| π0.5 L10-50 | .25 | −.022 / +.020 | −.019 / +.014 | −.018 / +.010 |
| | .32 | −.017 / +.014 | −.014 / +.009 | −.014 / +.003 |
| | .40 | −.010 / +.008 | −.008 / +.007 | −.005 / +.001 |
| GR00T L10-50 | .25 | −.025 / +.027 | −.019 / +.018 | −.017 / +.012 |
| | .32 | −.018 / +.019 | −.012 / +.013 | −.011 / +.004 |
| | .40 | −.010 / +.010 | −.009 / +.009 | −.003 / +.001 |
| π0.5 Sp-50 | .22 | −.013 / +.013 | −.011 / +.008 | −.013 / +.007 |
| | .30 | −.009 / +.009 | −.006 / +.006 | −.007 / +.004 |
| | .40 | −.004 / +.004 | −.004 / +.004 | −.004 / +.001 |
| GR00T Sp-50 | .22 | −.016 / +.012 | −.011 / +.006 | −.011 / +.006 |
| | .30 | −.011 / +.008 | −.006 / +.005 | −.005 / +.002 |
| π0.5 L10-200 | .25 | −.014 / +.014 | −.010 / +.009 | −.007 / +.003 |
| GR00T L10-200 | .25 | −.022 / +.023 | −.015 / +.016 | −.011 / +.010 |
| | .32 | −.016 / +.016 | −.010 / +.009 | −.008 / +.003 |
| π0.5 L10-500 | .25 | −.016 / +.016 | −.012 / +.009 | −.009 / +.003 |
| GR00T L10-500 | .25 | −.024 / +.023 | −.017 / +.016 | −.011 / +.010 |

- **Error band.** All proposed arms stay within ±.027 even at a ±25% guard-rate error, and within ±.02 at ±15%.
- **Expected direction.** The disclosed check suggests the closed-loop g is mostly *below* the library's. Realized IR should therefore land at or slightly below target (PREDICTION "informed" column: −.017 to +.008).
- **Sensitivity shrinks with target.** At higher targets, less of the spend depends on the guard.

## 4. Offline value evidence and how much weight it deserves

**Risk proxy.** The proxy is the LOEO motion error between the row's stored policy chunk and the GC_dist-corrected cache chunk (10 controls, 6 motion channels, σ units). It is reconstructed exactly from R10 opus's library-only k-fold head terms. Values below are ratios to the mean over non-guard anchors.

| cell-size | mean err | at guard anchors | anchors 0..3 | time quintiles | lag-1 autocorr | after guard +1 / +2 |
|---|---|---|---|---|---|---|
| π0.5 L10-50 | .399 | 1.43 | .92 .89 .87 .95 | .94 .91 .98 1.09 1.14 | .62 | 1.25 / 1.24 |
| GR00T L10-50 | .493 | 1.00 | .81 .77 .85 .91 | .88 .93 .96 1.02 1.26 | .59 | 1.06 / 1.01 |
| π0.5 Sp-50 | .481 | 1.38 | .68 .84 .78 1.09 | .78 1.03 1.14 1.05 1.10 | .49 | 1.27 / 1.13 |
| GR00T Sp-50 | .483 | 1.45 | .76 .70 .87 1.07 | .78 1.09 1.13 .97 1.12 | .50 | 1.24 / 1.18 |
| π0.5 L10-200 | .311 | 1.25 | .83 .83 .83 1.00 | .87 .97 1.00 1.08 1.12 | .52 | 1.10 / 1.09 |
| GR00T L10-200 | .412 | 1.11 | .72 .89 .86 .94 | .90 .97 .96 1.07 1.15 | .55 | 1.04 / 1.04 |
| π0.5 L10-500 | .278 | 1.28 | .79 .80 .84 .92 | .85 .95 1.00 1.10 1.16 | .58 | 1.13 / 1.09 |
| GR00T L10-500 | .373 | 1.09 | .70 .93 .88 .89 | .87 .95 .99 1.06 1.18 | .54 | 1.02 / 1.05 |

**At the frozen arm settings (`value_diag.py`).** "Uncovered high-error streaks" are runs of ≥ 3 consecutive anchors, all above the cell's 75th error percentile and none called.

| cell-size | target | streaks per episode: base / random / periodic | capture: random / periodic | longest cache run, mean (p90): random / periodic |
|---|---|---|---|---|
| π0.5 L10-50 | .25 / .32 / .40 | .44 / .25, .12, .02 / .18, 0, 0 | .98–1.00 / .92–1.00 | 6.4 (10), 3.9 (6), 2.2 (3) / 3.0 (3), 1.5 (2), 1.0 (1) |
| GR00T L10-50 | .25 / .32 / .40 | .34 / .23, .10, .02 / .12, 0, 0 | .99–1.00 / .97–1.01 | 6.0 (9), 3.7 (6), 2.2 (3) / 3.0 (3), 1.2 (2), 1.0 (1) |
| π0.5 Sp-50 | .22 / .30 / .40 | .22 / .11, .05, .01 / .02, 0, 0 | 1.00–1.01 / 1.00–1.02 | 4.9 (8), 3.0 (5), 1.6 (3) / 2.9 (3), 1.2 (2), 1.0 (1) |
| GR00T Sp-50 | .22 / .30 | .16 / .10, .03 / .04, 0 | 1.00 / .98–1.02 | 4.8 (8), 3.0 (5) / 2.8 (3), 1.1 (1) |
| GR00T L10-200 | .25 / .32 | .43 / .26, .11 / .13, 0 | 1.00 / 1.00–1.01 | 5.7 (9), 3.6 (6) / 2.9 (3), 1.1 (1) |
| π0.5 L10-200 | .25 | .56 / .25 / 0 | 1.00 / .96 | 5.8 (9) / 2.0 (2) |
| π0.5 L10-500 | .25 | .47 / .25 / .03 | 1.00 / .98 | 5.9 (9) / 2.4 (3) |
| GR00T L10-500 | .25 | .42 / .26 / .13 | 1.00 / 1.01 | 5.8 (9) / 2.9 (3) |

**Variants at the middle level:**

| Variant | Capture | Longest cache run | Uncovered streaks | Verdict |
|---|---|---|---|---|
| Periodic + post-guard tail | 1.01–1.06 | 1.9–2.7 anchors | 0–.04 | Kept as one exploratory arm per weak cell |
| Random two-chunk segments | .97–.99 | 4.5–4.7 anchors (longer than plain random) | .14–.17 | Kept on the two L10-50 cells only, as the "segment length" probe |
| Front-loaded first anchor | .89–.97 | — | — | Rejected |
| Refractory 2 anchors after a guard call | .90–.98 | — | — | Rejected |
| Skip step 0 for random | 1.00–1.06 | — | — | Negligible change; not proposed |

**Weight of this evidence: low for SR, high for IR.**
- **Precedent.** R6 placed calls by a LOEO disagreement predictor that captured far more error than any schedule here, and it did no better than uniform placement (−0.15 pp over 8 cells).
- **Size of the differences.** Capture differences of ±8% are an order of magnitude smaller than that and should carry essentially no SR weight.
- **Cache-run structure.** It is the one real difference between random and periodic. Its value rests on a closed-loop hypothesis (drift compounds over long cache stretches) that the open-loop replay cannot test. The streak statistic is close to tautological: periodic bounds runs by construction.
- **Prior closed-loop hints**, all different bases and confounded, from R6's published aggregates; the trend is consistent but none is significant:

| Comparison | SR @ IR | SR @ IR |
|---|---|---|
| GR00T L10-50: periodic every 2 anchors vs uniform random | .794 @ .291 | .770 @ .301 |
| GR00T Sp-50: periodic every 3 anchors vs B + random .25 | .934 @ .233 | .902 @ .234 |
| π0.5 L10-50: B + run cap vs interpolated B + random | .866 @ .292 | ≈ .857 |

**Recommendation.** Use random vs periodic at matched IR as the main closed-loop contrast.

## 5. Overlap-rule validation (disclosed, R6 aggregates)

Source: `r6_overlap_check.py` → `out/r6_overlap_check.md`. Inputs are R6 B replicates (g_B = M/V) and 13 "B + random dose" arms. Nothing was fitted.

| cell | dose | B guard rate | miss frac obs / pred | IR obs / pred |
|---|---|---|---|---|
| GR00T L10-50 | .25 / .5 / .75 | .343 | .505/.507, .661/.671, .828/.836 | .290/.292, .358/.362, .430/.433 |
| GR00T Sp-50 | .125 / .25 | .166 | .269/.270, .366/.375 | .192/.192, .234/.238 |
| π0.5 L10-50 | .25 / .5 / .75 | .244 | .431/.433, .614/.622, .811/.811 | .260/.261, .339/.342, .423/.423 |
| π0.5 L10-500 | .125 / .25 | .192 | .299/.293, .384/.394 | .205/.202, .241/.245 |
| π0.5 Sp-50 | .25 / .5 / .75 | .145 | .346/.359, .547/.572, .770/.786 | .227/.233, .315/.326, .411/.418 |

**Reading.** Extra calls lower the guard's own firing rate slightly (implied g .145 → .08–.13 on π0.5 Sp-50), so the model over-predicts IR by ≤ .011.

## 6. Reproduction (opus CPUs; total ≈ 17 min)

```bash
P="taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r11.opus"
$P.loeo_anchor_table --workers 16     # 4 min; out/anchor/*.npz (60 files), exact e10 match with R10 opus k-fold
$P.offline_curves                     # 12 min; out/curves.json, out/OFFLINE_TABLES.md
$P.r10_check; $P.r6_overlap_check     # disclosed aggregate checks
$P.arm_grid; $P.value_diag; $P.size_sensitivity; $P.predict
$P.test_ir_model                      # self-tests (guard port vs frozen judge, closed form, cap bounds)
```
