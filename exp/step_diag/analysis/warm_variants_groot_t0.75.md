## Warm-start continuation variants — groot (m=1, t=0.75; descriptive, 95% paired bootstrap)

missing arms: warmshoot_t0.75

### TurnOnSinkFaucet

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.18 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.04 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.28 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.24 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.32 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.32 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.20 | [+0.08, +0.32] | 11 / 1 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | -0.08 | [-0.20, +0.04] | 3 / 7 |
| warmreset_t0.75 - full | 50 | +0.06 | [-0.06, +0.20] | 7 / 4 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.28 | [+0.14, +0.42] | 15 / 1 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| resetfinal_t0.75 - full | 50 | +0.14 | [-0.02, +0.30] | 12 / 5 |
| midfinal_t0.75 - plain_k1 | 50 | +0.28 | [+0.14, +0.42] | 15 / 1 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.18, +0.18] | 11 / 11 |
| midfinal_t0.75 - full | 50 | +0.14 | [+0.00, +0.28] | 11 / 4 |
| warm_t0.75 - plain_k1 | 50 | +0.24 | [+0.12, +0.38] | 13 / 1 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| warm_t0.75 - resetfinal_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| warm_t0.75 - full | 50 | +0.10 | [-0.06, +0.26] | 12 / 7 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 3, serving runtimes 3 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.82 | ok | 4.0 | 0.0 | – |
| plain_k1 | 100 | 0.94 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 100 | 0.49 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.94 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.94 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.88 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.02 | [-0.08, +0.04] | 1 / 2 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.52 | [+0.36, +0.68] | 28 / 2 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| warmreset_t0.75 - full | 50 | +0.12 | [+0.00, +0.24] | 8 / 2 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.52 | [+0.38, +0.66] | 26 / 0 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| resetfinal_t0.75 - full | 50 | +0.12 | [+0.02, +0.22] | 7 / 1 |
| midfinal_t0.75 - plain_k1 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.46 | [+0.32, +0.60] | 23 / 0 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | -0.06 | [-0.16, +0.04] | 2 / 5 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | -0.06 | [-0.14, +0.02] | 1 / 4 |
| midfinal_t0.75 - full | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| warm_t0.75 - plain_k1 | 100 | -0.45 | [-0.56, -0.34] | 4 / 49 |
| warm_t0.75 - warmreset_t0.75 | 50 | -0.52 | [-0.68, -0.36] | 2 / 28 |
| warm_t0.75 - resetfinal_t0.75 | 50 | -0.52 | [-0.66, -0.38] | 0 / 26 |
| warm_t0.75 - full | 50 | -0.40 | [-0.58, -0.22] | 4 / 24 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 2 | 0.500 |
| plain_k1 | 2 | 0.490 |
| warm_t0.75 | 2 | 0.385 |
| warmreset_t0.75 | 2 | 0.590 |
| warmshoot_t0.75 | 0 | – |
| resetfinal_t0.75 | 2 | 0.630 |
| midfinal_t0.75 | 2 | 0.600 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 2 | +0.090 | [+0.020, +0.160] |
| warmreset_t0.75 - warm_t0.75 | 2 | +0.240 | [+0.130, +0.350] |
| warmreset_t0.75 - resetfinal_t0.75 | 2 | -0.040 | [-0.110, +0.030] |
| warmreset_t0.75 - full | 2 | +0.090 | [+0.000, +0.180] |
| resetfinal_t0.75 - plain_k1 | 2 | +0.130 | [+0.060, +0.210] |
| resetfinal_t0.75 - warm_t0.75 | 2 | +0.280 | [+0.170, +0.390] |
| resetfinal_t0.75 - warmreset_t0.75 | 2 | +0.040 | [-0.030, +0.110] |
| resetfinal_t0.75 - full | 2 | +0.130 | [+0.040, +0.230] |
| midfinal_t0.75 - plain_k1 | 2 | +0.100 | [+0.020, +0.180] |
| midfinal_t0.75 - warm_t0.75 | 2 | +0.250 | [+0.150, +0.350] |
| midfinal_t0.75 - warmreset_t0.75 | 2 | +0.010 | [-0.090, +0.110] |
| midfinal_t0.75 - resetfinal_t0.75 | 2 | -0.030 | [-0.130, +0.070] |
| midfinal_t0.75 - full | 2 | +0.100 | [+0.010, +0.190] |
| warm_t0.75 - plain_k1 | 2 | -0.105 | [-0.190, -0.020] |
| warm_t0.75 - warmreset_t0.75 | 2 | -0.240 | [-0.350, -0.130] |
| warm_t0.75 - resetfinal_t0.75 | 2 | -0.280 | [-0.380, -0.180] |
| warm_t0.75 - full | 2 | -0.150 | [-0.270, -0.030] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
