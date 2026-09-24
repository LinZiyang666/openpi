## Warm-start continuation variants — pi05 (m=1, t=0.1; descriptive, 95% paired bootstrap)

### CloseFridge

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.02 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.22 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 50 | 0.22 | ok | 1.0 | 0.0 | – |
| warmshoot_t0.1 | 50 | 0.80 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.1 | 50 | 0.24 | ok | 1.0 | 0.0 | – |
| midfinal_t0.1 | 50 | 0.54 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.1 - plain_k1 | 50 | +0.20 | [+0.08, +0.32] | 11 / 1 |
| warmreset_t0.1 - warm_t0.1 | 50 | +0.00 | [-0.18, +0.18] | 11 / 11 |
| warmreset_t0.1 - resetfinal_t0.1 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| warmreset_t0.1 - full | 50 | -0.40 | [-0.58, -0.22] | 5 / 25 |
| warmshoot_t0.1 - plain_k1 | 50 | +0.78 | [+0.66, +0.88] | 39 / 0 |
| warmshoot_t0.1 - warm_t0.1 | 50 | +0.58 | [+0.40, +0.74] | 33 / 4 |
| warmshoot_t0.1 - warmreset_t0.1 | 50 | +0.58 | [+0.42, +0.74] | 31 / 2 |
| warmshoot_t0.1 - resetfinal_t0.1 | 50 | +0.56 | [+0.40, +0.72] | 30 / 2 |
| warmshoot_t0.1 - full | 50 | +0.18 | [+0.00, +0.36] | 16 / 7 |
| resetfinal_t0.1 - plain_k1 | 50 | +0.22 | [+0.10, +0.36] | 12 / 1 |
| resetfinal_t0.1 - warm_t0.1 | 50 | +0.02 | [-0.16, +0.18] | 10 / 9 |
| resetfinal_t0.1 - warmreset_t0.1 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| resetfinal_t0.1 - full | 50 | -0.38 | [-0.56, -0.18] | 6 / 25 |
| midfinal_t0.1 - plain_k1 | 50 | +0.52 | [+0.38, +0.66] | 26 / 0 |
| midfinal_t0.1 - warm_t0.1 | 50 | +0.32 | [+0.14, +0.50] | 21 / 5 |
| midfinal_t0.1 - warmreset_t0.1 | 50 | +0.32 | [+0.12, +0.52] | 23 / 7 |
| midfinal_t0.1 - resetfinal_t0.1 | 50 | +0.30 | [+0.12, +0.48] | 20 / 5 |
| midfinal_t0.1 - full | 50 | -0.08 | [-0.30, +0.12] | 13 / 17 |
| warm_t0.1 - plain_k1 | 50 | +0.20 | [+0.08, +0.32] | 11 / 1 |
| warm_t0.1 - warmreset_t0.1 | 50 | +0.00 | [-0.18, +0.18] | 11 / 11 |
| warm_t0.1 - resetfinal_t0.1 | 50 | -0.02 | [-0.20, +0.14] | 9 / 10 |
| warm_t0.1 - full | 50 | -0.40 | [-0.56, -0.24] | 2 / 22 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.84 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.98 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.12 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 50 | 1.00 | ok | 1.0 | 0.0 | – |
| warmshoot_t0.1 | 50 | 0.08 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.1 | 50 | 1.00 | ok | 1.0 | 0.0 | – |
| midfinal_t0.1 | 50 | 0.42 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.1 - plain_k1 | 50 | +0.02 | [+0.00, +0.06] | 1 / 0 |
| warmreset_t0.1 - warm_t0.1 | 50 | +0.88 | [+0.78, +0.96] | 44 / 0 |
| warmreset_t0.1 - resetfinal_t0.1 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| warmreset_t0.1 - full | 50 | +0.16 | [+0.06, +0.26] | 8 / 0 |
| warmshoot_t0.1 - plain_k1 | 50 | -0.90 | [-0.98, -0.82] | 0 / 45 |
| warmshoot_t0.1 - warm_t0.1 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| warmshoot_t0.1 - warmreset_t0.1 | 50 | -0.92 | [-0.98, -0.84] | 0 / 46 |
| warmshoot_t0.1 - resetfinal_t0.1 | 50 | -0.92 | [-0.98, -0.84] | 0 / 46 |
| warmshoot_t0.1 - full | 50 | -0.76 | [-0.88, -0.64] | 0 / 38 |
| resetfinal_t0.1 - plain_k1 | 50 | +0.02 | [+0.00, +0.06] | 1 / 0 |
| resetfinal_t0.1 - warm_t0.1 | 50 | +0.88 | [+0.78, +0.96] | 44 / 0 |
| resetfinal_t0.1 - warmreset_t0.1 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| resetfinal_t0.1 - full | 50 | +0.16 | [+0.06, +0.26] | 8 / 0 |
| midfinal_t0.1 - plain_k1 | 50 | -0.56 | [-0.70, -0.42] | 0 / 28 |
| midfinal_t0.1 - warm_t0.1 | 50 | +0.30 | [+0.18, +0.42] | 15 / 0 |
| midfinal_t0.1 - warmreset_t0.1 | 50 | -0.58 | [-0.72, -0.44] | 0 / 29 |
| midfinal_t0.1 - resetfinal_t0.1 | 50 | -0.58 | [-0.72, -0.44] | 0 / 29 |
| midfinal_t0.1 - full | 50 | -0.42 | [-0.58, -0.26] | 3 / 24 |
| warm_t0.1 - plain_k1 | 50 | -0.86 | [-0.94, -0.76] | 0 / 43 |
| warm_t0.1 - warmreset_t0.1 | 50 | -0.88 | [-0.96, -0.78] | 0 / 44 |
| warm_t0.1 - resetfinal_t0.1 | 50 | -0.88 | [-0.96, -0.78] | 0 / 44 |
| warm_t0.1 - full | 50 | -0.72 | [-0.84, -0.60] | 0 / 36 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 2 | 0.730 |
| plain_k1 | 2 | 0.500 |
| warm_t0.1 | 2 | 0.170 |
| warmreset_t0.1 | 2 | 0.610 |
| warmshoot_t0.1 | 2 | 0.440 |
| resetfinal_t0.1 | 2 | 0.620 |
| midfinal_t0.1 | 2 | 0.480 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.1 - plain_k1 | 2 | +0.110 | [+0.050, +0.180] |
| warmreset_t0.1 - warm_t0.1 | 2 | +0.440 | [+0.340, +0.540] |
| warmreset_t0.1 - resetfinal_t0.1 | 2 | -0.010 | [-0.070, +0.050] |
| warmreset_t0.1 - full | 2 | -0.120 | [-0.220, -0.010] |
| warmshoot_t0.1 - plain_k1 | 2 | -0.060 | [-0.130, +0.010] |
| warmshoot_t0.1 - warm_t0.1 | 2 | +0.270 | [+0.160, +0.370] |
| warmshoot_t0.1 - warmreset_t0.1 | 2 | -0.170 | [-0.260, -0.090] |
| warmshoot_t0.1 - resetfinal_t0.1 | 2 | -0.180 | [-0.270, -0.090] |
| warmshoot_t0.1 - full | 2 | -0.290 | [-0.400, -0.180] |
| resetfinal_t0.1 - plain_k1 | 2 | +0.120 | [+0.060, +0.190] |
| resetfinal_t0.1 - warm_t0.1 | 2 | +0.450 | [+0.350, +0.550] |
| resetfinal_t0.1 - warmreset_t0.1 | 2 | +0.010 | [-0.050, +0.070] |
| resetfinal_t0.1 - full | 2 | -0.110 | [-0.220, +0.000] |
| midfinal_t0.1 - plain_k1 | 2 | -0.020 | [-0.120, +0.080] |
| midfinal_t0.1 - warm_t0.1 | 2 | +0.310 | [+0.200, +0.420] |
| midfinal_t0.1 - warmreset_t0.1 | 2 | -0.130 | [-0.250, -0.010] |
| midfinal_t0.1 - resetfinal_t0.1 | 2 | -0.140 | [-0.250, -0.030] |
| midfinal_t0.1 - full | 2 | -0.250 | [-0.390, -0.110] |
| warm_t0.1 - plain_k1 | 2 | -0.330 | [-0.410, -0.250] |
| warm_t0.1 - warmreset_t0.1 | 2 | -0.440 | [-0.540, -0.340] |
| warm_t0.1 - resetfinal_t0.1 | 2 | -0.450 | [-0.540, -0.350] |
| warm_t0.1 - full | 2 | -0.560 | [-0.660, -0.460] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
