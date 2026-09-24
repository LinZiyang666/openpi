## Warm-start continuation variants — pi05 (m=2, t=0.2; descriptive, 95% paired bootstrap)

### CloseFridge

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.08 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.30 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.52 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.76 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.58 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.44 | [+0.30, +0.58] | 23 / 1 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.22 | [+0.04, +0.40] | 18 / 7 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | -0.24 | [-0.38, -0.10] | 2 / 14 |
| warmreset_t0.2 - full | 50 | -0.10 | [-0.30, +0.10] | 10 / 15 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.30 | [-0.44, -0.18] | 0 / 15 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.52 | [-0.66, -0.38] | 0 / 26 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.76 | [-0.88, -0.64] | 0 / 38 |
| warmshoot_t0.2 - full | 50 | -0.62 | [-0.74, -0.48] | 0 / 31 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.68 | [+0.54, +0.82] | 35 / 1 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.46 | [+0.28, +0.62] | 26 / 3 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | +0.24 | [+0.10, +0.38] | 14 / 2 |
| resetfinal_t0.2 - full | 50 | +0.14 | [-0.04, +0.32] | 14 / 7 |
| midfinal_t0.2 - plain_k2 | 50 | +0.50 | [+0.36, +0.64] | 26 / 1 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.28 | [+0.08, +0.48] | 22 / 8 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | +0.06 | [-0.12, +0.24] | 13 / 10 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.18 | [-0.36, +0.00] | 6 / 15 |
| midfinal_t0.2 - full | 50 | -0.04 | [-0.24, +0.16] | 12 / 14 |
| warm_t0.2 - plain_k2 | 50 | +0.22 | [+0.06, +0.38] | 15 / 4 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.22 | [-0.40, -0.04] | 7 / 18 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.46 | [-0.62, -0.28] | 3 / 26 |
| warm_t0.2 - full | 50 | -0.32 | [-0.50, -0.12] | 6 / 22 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.84 | ok | 10.0 | 0.0 | – |
| plain_k2 | 100 | 0.94 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 100 | 0.16 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.72 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.32 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | -0.12 | [-0.24, +0.00] | 2 / 8 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.66 | [+0.50, +0.80] | 35 / 2 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.10 | [-0.02, +0.22] | 8 / 3 |
| warmreset_t0.2 - full | 50 | -0.02 | [-0.16, +0.12] | 6 / 7 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.94 | [-1.00, -0.86] | 0 / 47 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.16 | [-0.26, -0.06] | 0 / 8 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.82 | [-0.92, -0.70] | 0 / 41 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.72 | [-0.84, -0.58] | 0 / 36 |
| warmshoot_t0.2 - full | 50 | -0.84 | [-0.94, -0.74] | 0 / 42 |
| resetfinal_t0.2 - plain_k2 | 50 | -0.22 | [-0.38, -0.08] | 3 / 14 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.56 | [+0.40, +0.70] | 29 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.10 | [-0.22, +0.02] | 3 / 8 |
| resetfinal_t0.2 - full | 50 | -0.12 | [-0.28, +0.04] | 6 / 12 |
| midfinal_t0.2 - plain_k2 | 50 | -0.62 | [-0.76, -0.46] | 1 / 32 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.16 | [+0.04, +0.30] | 10 / 2 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.50 | [-0.64, -0.34] | 1 / 26 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.40 | [-0.56, -0.24] | 2 / 22 |
| midfinal_t0.2 - full | 50 | -0.52 | [-0.70, -0.32] | 5 / 31 |
| warm_t0.2 - plain_k2 | 100 | -0.78 | [-0.86, -0.70] | 0 / 78 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.66 | [-0.80, -0.50] | 2 / 35 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.56 | [-0.70, -0.42] | 1 / 29 |
| warm_t0.2 - full | 50 | -0.68 | [-0.82, -0.54] | 1 / 35 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 2 | 0.730 |
| plain_k2 | 2 | 0.510 |
| warm_t0.2 | 2 | 0.230 |
| warmreset_t0.2 | 2 | 0.670 |
| warmshoot_t0.2 | 2 | 0.000 |
| resetfinal_t0.2 | 2 | 0.740 |
| midfinal_t0.2 | 2 | 0.450 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 2 | +0.160 | [+0.070, +0.250] |
| warmreset_t0.2 - warm_t0.2 | 2 | +0.440 | [+0.320, +0.560] |
| warmreset_t0.2 - resetfinal_t0.2 | 2 | -0.070 | [-0.160, +0.020] |
| warmreset_t0.2 - full | 2 | -0.060 | [-0.180, +0.060] |
| warmshoot_t0.2 - plain_k2 | 2 | -0.510 | [-0.560, -0.460] |
| warmshoot_t0.2 - warm_t0.2 | 2 | -0.230 | [-0.310, -0.150] |
| warmshoot_t0.2 - warmreset_t0.2 | 2 | -0.670 | [-0.750, -0.580] |
| warmshoot_t0.2 - resetfinal_t0.2 | 2 | -0.740 | [-0.820, -0.650] |
| warmshoot_t0.2 - full | 2 | -0.730 | [-0.810, -0.640] |
| resetfinal_t0.2 - plain_k2 | 2 | +0.230 | [+0.130, +0.330] |
| resetfinal_t0.2 - warm_t0.2 | 2 | +0.510 | [+0.400, +0.620] |
| resetfinal_t0.2 - warmreset_t0.2 | 2 | +0.070 | [-0.020, +0.160] |
| resetfinal_t0.2 - full | 2 | +0.010 | [-0.110, +0.130] |
| midfinal_t0.2 - plain_k2 | 2 | -0.060 | [-0.160, +0.040] |
| midfinal_t0.2 - warm_t0.2 | 2 | +0.220 | [+0.100, +0.340] |
| midfinal_t0.2 - warmreset_t0.2 | 2 | -0.220 | [-0.340, -0.100] |
| midfinal_t0.2 - resetfinal_t0.2 | 2 | -0.290 | [-0.400, -0.170] |
| midfinal_t0.2 - full | 2 | -0.280 | [-0.420, -0.140] |
| warm_t0.2 - plain_k2 | 2 | -0.280 | [-0.370, -0.190] |
| warm_t0.2 - warmreset_t0.2 | 2 | -0.440 | [-0.560, -0.320] |
| warm_t0.2 - resetfinal_t0.2 | 2 | -0.510 | [-0.620, -0.390] |
| warm_t0.2 - full | 2 | -0.500 | [-0.610, -0.380] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
