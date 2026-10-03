# Prediction for the running LIBERO-10-50 residual_half arms (written before their results)

Timestamp: 2026-10-01 21:20:32 CDT. At this time `/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation/runs/` contains only the
four π0.5 Spatial-50 arms; no LIBERO-10 result has been read or exists locally.

## Basis
- Astra's Spatial-50 result per task (discovery, 300 pairs): the half residual recovered 65% of the (P10 − cache) gap
  overall (.820 → .933, P10 .993); per task it recovered 0.6–1.0 of the gap on tasks 1, 4, 5, 6 (gaps .16–.63) and
  0.3–0.4 on tasks 8, 9; it never hurt a task (worst −.03 on task 3).
- My offline evaluation of the same residual head on held-out inits 20–29 (A arm): the half residual lowers the
  motion gap by 29% (π0.5 L10-50) and 33% (GR00T L10-50), uniformly over tasks (every task −25% to −40%); the head
  keeps the gripper unchanged (grip disagreement .104 / .163 unchanged).
- My grown-library closed-loop result on π0.5 L10-50 (full replacement incl. gripper, 100 pairs): +30–40 pp on tasks
  0, 4, 9, −40–50 pp on tasks 2, 3 (where P10 < cache), net 0. The half residual is a shrunk, gripper-preserving
  correction of the cache's own chunk, so I expect the gains on the hard tasks to survive and the losses on tasks 2, 3
  to shrink to ≤ 10 pp.
- Long-horizon tasks compound errors and need grasp precision, so I expect a smaller recovered fraction than on
  Spatial: c ≈ .4 of the per-task gap where P10 > cache, and about half of the (negative) gap where P10 < cache.

## Prediction, π0.5 LIBERO-10-50 (300 pairs, inits 0–29). Cache / P10 columns = R8 discovery per-task SR.
| task | cache | P10 | predicted residual_half |
|---|---|---|---|
| 0 | .33 | .97 | .55–.65 (point .58) |
| 1 | .90 | 1.00 | .92–.97 |
| 2 | .90 | .83 | .83–.90 (no gain; possible small loss) |
| 3 | .97 | .90 | .90–.97 (no gain; possible small loss) |
| 4 | .53 | .93 | .63–.75 (point .69) |
| 5 | 1.00 | 1.00 | .97–1.00 |
| 6 | .53 | .83 | .58–.72 (point .65) |
| 7 | .63 | .97 | .70–.83 (point .76) |
| 8 | .63 | .70 | .60–.72 |
| 9 | .73 | .93 | .75–.87 (point .81) |
| **all** | **.717** (same-topology cache arm expected .70–.76) | **.907** | **.76–.82 (point .79)**: +4 to +10 pp over the paired cache arm, recovering 25–50% of the P10 gap (less than Spatial's 65%) |

Falsifiable statements: (a) the gain is concentrated on tasks 0, 4, 6, 7, 9 (each ≥ +.07); (b) tasks 2 and 3 do not gain
(≤ +.03) and may lose; (c) the overall residual_half SR stays below P10 by ≥ 8 pp; (d) owner IR stays within .002 of the
cache arm (≈ .0765).

## Prediction, GR00T LIBERO-10-50 (300 pairs)
| task | cache | P10 | predicted residual_half |
|---|---|---|---|
| 0 | .23 | .77 | .38–.52 (point .44) |
| 1 | .77 | .90 | .80–.87 |
| 2 | 1.00 | .97 | .95–1.00 |
| 3 | .93 | 1.00 | .93–.98 |
| 4 | .57 | .83 | .62–.73 (point .67) |
| 5 | .97 | .97 | .93–1.00 |
| 6 | .60 | .83 | .63–.75 (point .69) |
| 7 | .43 | .93 | .53–.72 (point .63) |
| 8 | .63 | .80 | .65–.75 |
| 9 | .23 | .93 | .40–.60 (point .51) |
| **all** | **.637** | **.893** | **.70–.78 (point .74)**: +5 to +12 pp over the paired cache arm |

Falsifiable: (a) gains concentrated on tasks 0, 7, 9 (largest gaps) and 4, 6; (b) tasks 2, 3, 5 unchanged within ±.05;
(c) GR00T's absolute gain ≥ π0.5's (its cache gap to the policy is 10× the policy's noise floor vs 2.4× for π0.5);
(d) the residual_half arm remains ≥ 10 pp below P10.

If instead the residual_half gains ≥ +15 pp on either L10-50 cell, my "long-horizon compounding limits the half
residual" reading is wrong and the Spatial recovery fraction (65%) transfers; if it gains ≤ +2 pp, the Spatial result
was cell-specific (dominated by one task) and the grown-library null on L10-50 generalizes to residual corrections.
