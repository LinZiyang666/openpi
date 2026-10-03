# R11 opus: predictions for the schedule-knob arms

**Timestamp: 2026-10-02 21:10 CDT.**
- Written before any R11 closed-loop result exists. No R11 arm had been emitted or run when this was written.
- No `os_closed_loop` episode, journal, log or summary was read.
- Inputs: the library model (`out/arm_grid.json`) plus published aggregates (R10 REPORT 3-layer SR / IR and pure-policy SR; R6 ANALYSIS / frontier aggregates as priors), disclosed below.
- Machine-readable copy: `out/predictions.json` (`predict.py`).

## How the numbers were made

**Predicted realized IR, two columns:**
- **Library** (`pred IR library`): the frozen basis of the setting.
- **Informed** (`pred IR informed`): the same setting re-simulated with the library guard flags rescaled to the closed-loop base guard rate implied by R10's published 3-layer IR. The informed column is my best guess. It uses a test-set aggregate for prediction only; no knob setting depends on it.

**Predicted SR:**
- **Formula:** `SR = SR_base + (SR_pure − SR_base)·φ(IR_informed − IR_base) + schedule prior`, with `φ(d) = d²/(d² + .12²)`.
- **Where φ comes from:** a hand-set shape read off R6's published "B + random dose" aggregates (about 20–35% of the base-to-pure gap closed at +.08 IR, 65–70% at +.16, 80–90% at +.24).
- **Schedule priors (hand-set):**

| Arm type | Prior on SR |
|---|---|
| Periodic, weak LIBERO-10 cells, target ≤ .32 | +1.0 pp |
| Periodic, Spatial-50, target ≤ .30 | +0.5 pp |
| Periodic, target .40 | 0 |
| Post-guard tail (on top of periodic) | +0.3 pp |
| Random two-chunk segments | −0.5 pp |
| Any arm in a cell whose base is already at pure | −0.5 pp |

**Uncertainty.**
- A single arm vs the same-batch base has a 95% interval of roughly ±3.5 pp on the weak cells and ±2.5 pp on the strong ones (500 paired episodes).
- The schedule priors are well inside single-cell noise.

## Per-arm predictions

The base is the R10 3-layer SR @ IR (test A aggregate). The same-batch base will be re-measured.

| # | cell-size | target | method | setting | pred IR library | pred IR informed | base SR @ IR | pred SR | Δ vs base (pp) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GR00T L10-50 | .32 | periodic | K 1.0252 | .320 | .323 | .754 @ .207 | .834 | +8.0 |
| 2 | GR00T L10-50 | .32 | random | ρ .3998 | .321 | .326 | .754 @ .207 | .826 | +7.2 |
| 3 | GR00T Sp-50 | .30 | periodic | K 1.0109 | .300 | .297 | .896 @ .127 | .930 | +3.4 |
| 4 | GR00T Sp-50 | .30 | random | ρ .4310 | .300 | .296 | .896 @ .127 | .925 | +2.9 |
| 5 | π0.5 L10-50 | .32 | periodic | K 1.0657 | .320 | .312 | .826 @ .168 | .884 | +5.8 |
| 6 | π0.5 L10-50 | .32 | random | ρ .4177 | .320 | .309 | .826 @ .168 | .874 | +4.8 |
| 7 | π0.5 Sp-50 | .30 | periodic | K 1.0549 | .300 | .297 | .910 @ .135 | .965 | +5.5 |
| 8 | π0.5 Sp-50 | .30 | random | ρ .4292 | .300 | .295 | .910 @ .135 | .960 | +5.0 |
| 9 | GR00T L10-50 | .25 | periodic | K 2.7237 | .250 | .255 | .754 @ .207 | .784 | +3.0 |
| 10 | GR00T L10-50 | .25 | random | ρ .1707 | .250 | .257 | .754 @ .207 | .775 | +2.1 |
| 11 | GR00T L10-50 | .40 | periodic | K .4512 | .399 | .401 | .754 @ .207 | .858 | +10.4 |
| 12 | GR00T L10-50 | .40 | random | ρ .6555 | .400 | .403 | .754 @ .207 | .859 | +10.5 |
| 13 | GR00T Sp-50 | .22 | periodic | K 2.4123 | .221 | .217 | .896 @ .127 | .917 | +2.1 |
| 14 | GR00T Sp-50 | .22 | random | ρ .2144 | .220 | .214 | .896 @ .127 | .911 | +1.5 |
| 15 | π0.5 L10-50 | .25 | periodic | K 2.6125 | .250 | .238 | .826 @ .168 | .857 | +3.1 |
| 16 | π0.5 L10-50 | .25 | random | ρ .1950 | .248 | .233 | .826 @ .168 | .844 | +1.8 |
| 17 | π0.5 L10-50 | .40 | periodic | K .4386 | .399 | .394 | .826 @ .168 | .890 | +6.4 |
| 18 | π0.5 L10-50 | .40 | random | ρ .6610 | .399 | .392 | .826 @ .168 | .890 | +6.4 |
| 19 | π0.5 Sp-50 | .22 | periodic | K 2.5795 | .219 | .215 | .910 @ .135 | .939 | +2.9 |
| 20 | π0.5 Sp-50 | .22 | random | ρ .2112 | .219 | .213 | .910 @ .135 | .933 | +2.3 |
| 21 | π0.5 Sp-50 | .40 | periodic | K .3931 | .397 | .395 | .910 @ .135 | .974 | +6.4 |
| 22 | π0.5 Sp-50 | .40 | random | ρ .6885 | .398 | .395 | .910 @ .135 | .974 | +6.4 |
| 23 | GR00T L10-200 | .25 | periodic | K 2.4585 | .250 | .250 | .820 @ .191 | .845 | +2.5 |
| 24 | GR00T L10-200 | .25 | random | ρ .1910 | .250 | .251 | .820 @ .191 | .835 | +1.5 |
| 25 | GR00T L10-200 | .32 | periodic | K 1.0069 | .320 | .320 | .820 @ .191 | .872 | +5.2 |
| 26 | GR00T L10-200 | .32 | random | ρ .4221 | .322 | .322 | .820 @ .191 | .862 | +4.2 |
| 27 | GR00T L10-500 | .25 | periodic | K 2.4699 | .250 | .240 | .906 @ .175 | .901 | −0.5 |
| 28 | GR00T L10-500 | .25 | random | ρ .1869 | .250 | .236 | .906 @ .175 | .901 | −0.5 |
| 29 | π0.5 L10-200 | .25 | periodic | K 2.0000 | .250 | .246 | .904 @ .150 | .899 | −0.5 |
| 30 | π0.5 L10-200 | .25 | random | ρ .2663 | .250 | .245 | .904 @ .150 | .899 | −0.5 |
| 31 | π0.5 L10-500 | .25 | periodic | K 2.0924 | .250 | .242 | .894 @ .150 | .899 | +0.5 |
| 32 | π0.5 L10-500 | .25 | random | ρ .2537 | .250 | .240 | .894 @ .150 | .899 | +0.5 |
| 33 | GR00T L10-50 | .32 | periodic + post-guard tail | K 2.3165 | .320 | .328 | .754 @ .207 | .839 | +8.5 |
| 34 | GR00T L10-50 | .32 | random two-chunk | ρ .2777 | .321 | .325 | .754 @ .207 | .820 | +6.6 |
| 35 | GR00T Sp-50 | .30 | periodic + post-guard tail | K 1.4117 | .300 | .295 | .896 @ .127 | .933 | +3.7 |
| 36 | π0.5 L10-50 | .32 | periodic + post-guard tail | K 1.8009 | .320 | .304 | .826 @ .168 | .885 | +5.9 |
| 37 | π0.5 L10-50 | .32 | random two-chunk | ρ .2839 | .319 | .309 | .826 @ .168 | .869 | +4.3 |
| 38 | π0.5 Sp-50 | .30 | periodic + post-guard tail | K 1.4050 | .299 | .293 | .910 @ .135 | .968 | +5.8 |

## Falsifiable statements (to be scored after the sweep)

**P1, IR linkage.**
- Realized owner IR is within ±.02 of `pred IR library` in ≥ 34 of 38 arms.
- The mean signed error (realized − library) is negative on the π0.5 arms (about −.008) and non-negative on GR00T L10-50 (about +.005).
- |realized − informed| ≤ .012 in ≥ 32 of 38 arms.

**P2, matched spend.** Random and periodic at the same target land within .01 of each other in ≥ 14 of 16 pairs. Periodic is closer to its library prediction than random in ≥ 10 of 16.

**P3, SR rises with spend on the weak cells.**
- The 3-layer base SR is the floor, and SR is non-decreasing in target in each 50-cell, within noise.
- At .40, π0.5 LIBERO-10-50 reaches about .89, still below pure .908.
- GR00T LIBERO-10-50 reaches about .86, below pure .898.
- π0.5 Spatial-50 reaches about .97.
- No arm passes a 2 pp noninferiority test against pure policy on the LIBERO-10-50 cells.

**P4, periodic vs random at matched IR.**
- Point estimates: +1 pp on the LIBERO-10 50 / 200 cells at targets ≤ .32, +0.5 pp on Spatial-50, about 0 at .40 and on the near-pure cells.
- Pooled over the 16 matched pairs: **+0.6 pp** (80% prediction interval −0.9 to +2.1).
- No single cell is significant.
- If the pooled contrast is ≤ −1 pp, the even-spacing hypothesis is wrong for this base.

**P5, near-pure cells** (π0.5 LIBERO-10-200 / -500, GR00T LIBERO-10-500).
- At .25, extra spend changes SR by less than 1.5 pp in either direction for both methods.
- Spending there is not worth it, because the base is already within 1.4 pp of pure policy.

**P6, variants.**
- Post-guard tail minus plain periodic at the same target: +0.3 pp (no direction claimed).
- Two-chunk segments minus random at the same target: −0.5 pp (no direction claimed).
- Neither difference is expected to be detectable with one arm per cell.
