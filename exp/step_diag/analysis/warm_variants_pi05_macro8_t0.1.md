## Warm-start continuation variants — pi05 (m=1, t=0.1; descriptive, 95% paired bootstrap)

### CloseBlenderLid

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.14 | ok | 10.0 | 0.0 | – |
| plain_k1 | 0 | – | NO | – | None | – |
| warm_t0.1 | 0 | – | NO | – | None | – |
| warmreset_t0.1 | 0 | – | NO | – | None | – |
| warmshoot_t0.1 | 0 | – | NO | – | None | – |
| resetfinal_t0.1 | 50 | 0.04 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| resetfinal_t0.1 - full | 50 | -0.10 | [-0.22, +0.00] | 2 / 7 |

### OpenCabinet

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.66 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.26 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.50 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 0 | – | NO | – | None | – |
| warmshoot_t0.1 | 0 | – | NO | – | None | – |
| resetfinal_t0.1 | 50 | 0.54 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| resetfinal_t0.1 - plain_k1 | 50 | +0.28 | [+0.16, +0.40] | 14 / 0 |
| resetfinal_t0.1 - warm_t0.1 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| resetfinal_t0.1 - full | 50 | -0.12 | [-0.28, +0.06] | 7 / 13 |
| warm_t0.1 - plain_k1 | 50 | +0.24 | [+0.10, +0.38] | 14 / 2 |
| warm_t0.1 - full | 50 | -0.16 | [-0.34, +0.02] | 7 / 15 |

### PickPlaceCounterToCabinet

cross-arm model/env identity: identical; worker islands 1, serving runtimes 1 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.54 | ok | 10.0 | 0.0 | – |
| plain_k1 | 0 | – | NO | – | None | – |
| warm_t0.1 | 0 | – | NO | – | None | – |
| warmreset_t0.1 | 0 | – | NO | – | None | – |
| warmshoot_t0.1 | 0 | – | NO | – | None | – |
| resetfinal_t0.1 | 50 | 0.62 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| resetfinal_t0.1 - full | 50 | +0.08 | [-0.12, +0.28] | 15 / 11 |

### PickPlaceDrawerToCounter

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.32 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.06 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.10 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 0 | – | NO | – | None | – |
| warmshoot_t0.1 | 0 | – | NO | – | None | – |
| resetfinal_t0.1 | 50 | 0.24 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| resetfinal_t0.1 - plain_k1 | 50 | +0.18 | [+0.06, +0.32] | 11 / 2 |
| resetfinal_t0.1 - warm_t0.1 | 50 | +0.14 | [+0.00, +0.28] | 10 / 3 |
| resetfinal_t0.1 - full | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| warm_t0.1 - plain_k1 | 50 | +0.04 | [-0.06, +0.14] | 4 / 2 |
| warm_t0.1 - full | 50 | -0.22 | [-0.36, -0.08] | 2 / 13 |

### PickPlaceSinkToCounter

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 1.00 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.62 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.26 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 0 | – | NO | – | None | – |
| warmshoot_t0.1 | 0 | – | NO | – | None | – |
| resetfinal_t0.1 | 50 | 0.80 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| resetfinal_t0.1 - plain_k1 | 50 | +0.18 | [+0.02, +0.34] | 13 / 4 |
| resetfinal_t0.1 - warm_t0.1 | 50 | +0.54 | [+0.38, +0.70] | 29 / 2 |
| resetfinal_t0.1 - full | 50 | -0.20 | [-0.32, -0.10] | 0 / 10 |
| warm_t0.1 - plain_k1 | 50 | -0.36 | [-0.52, -0.20] | 3 / 21 |
| warm_t0.1 - full | 50 | -0.74 | [-0.86, -0.62] | 0 / 37 |

### PickPlaceToasterToCounter

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.40 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.00 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.06 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 0 | – | NO | – | None | – |
| warmshoot_t0.1 | 0 | – | NO | – | None | – |
| resetfinal_t0.1 | 50 | 0.00 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| resetfinal_t0.1 - plain_k1 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| resetfinal_t0.1 - warm_t0.1 | 50 | -0.06 | [-0.14, +0.00] | 0 / 3 |
| resetfinal_t0.1 - full | 50 | -0.40 | [-0.54, -0.26] | 0 / 20 |
| warm_t0.1 - plain_k1 | 50 | +0.06 | [+0.00, +0.14] | 3 / 0 |
| warm_t0.1 - full | 50 | -0.34 | [-0.48, -0.20] | 1 / 18 |

### CloseFridge

cross-arm model/env identity: identical; worker islands 3, serving runtimes 3 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.02 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.22 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 50 | 0.22 | ok | 1.0 | 0.0 | – |
| warmshoot_t0.1 | 50 | 0.80 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.1 | 50 | 0.24 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.1 - plain_k1 | 50 | +0.20 | [+0.08, +0.32] | 11 / 1 |
| warmreset_t0.1 - warm_t0.1 | 50 | +0.00 | [-0.18, +0.18] | 11 / 11 |
| warmreset_t0.1 - full | 50 | -0.40 | [-0.58, -0.22] | 5 / 25 |
| warmshoot_t0.1 - plain_k1 | 50 | +0.78 | [+0.66, +0.88] | 39 / 0 |
| warmshoot_t0.1 - warm_t0.1 | 50 | +0.58 | [+0.40, +0.74] | 33 / 4 |
| warmshoot_t0.1 - warmreset_t0.1 | 50 | +0.58 | [+0.42, +0.74] | 31 / 2 |
| warmshoot_t0.1 - full | 50 | +0.18 | [+0.00, +0.36] | 16 / 7 |
| resetfinal_t0.1 - plain_k1 | 50 | +0.22 | [+0.10, +0.36] | 12 / 1 |
| resetfinal_t0.1 - warm_t0.1 | 50 | +0.02 | [-0.16, +0.18] | 10 / 9 |
| resetfinal_t0.1 - warmreset_t0.1 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| resetfinal_t0.1 - full | 50 | -0.38 | [-0.56, -0.18] | 6 / 25 |
| warm_t0.1 - plain_k1 | 50 | +0.20 | [+0.08, +0.32] | 11 / 1 |
| warm_t0.1 - warmreset_t0.1 | 50 | +0.00 | [-0.18, +0.18] | 11 / 11 |
| warm_t0.1 - full | 50 | -0.40 | [-0.56, -0.24] | 2 / 22 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 3, serving runtimes 3 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.84 | ok | 10.0 | 0.0 | – |
| plain_k1 | 50 | 0.98 | ok | 1.0 | 0.0 | – |
| warm_t0.1 | 50 | 0.12 | ok | 1.0 | 0.0 | – |
| warmreset_t0.1 | 50 | 1.00 | ok | 1.0 | 0.0 | – |
| warmshoot_t0.1 | 50 | 0.08 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.1 | 50 | 1.00 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.1 - plain_k1 | 50 | +0.02 | [+0.00, +0.06] | 1 / 0 |
| warmreset_t0.1 - warm_t0.1 | 50 | +0.88 | [+0.78, +0.96] | 44 / 0 |
| warmreset_t0.1 - full | 50 | +0.16 | [+0.06, +0.26] | 8 / 0 |
| warmshoot_t0.1 - plain_k1 | 50 | -0.90 | [-0.98, -0.82] | 0 / 45 |
| warmshoot_t0.1 - warm_t0.1 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| warmshoot_t0.1 - warmreset_t0.1 | 50 | -0.92 | [-0.98, -0.84] | 0 / 46 |
| warmshoot_t0.1 - full | 50 | -0.76 | [-0.88, -0.64] | 0 / 38 |
| resetfinal_t0.1 - plain_k1 | 50 | +0.02 | [+0.00, +0.06] | 1 / 0 |
| resetfinal_t0.1 - warm_t0.1 | 50 | +0.88 | [+0.78, +0.96] | 44 / 0 |
| resetfinal_t0.1 - warmreset_t0.1 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| resetfinal_t0.1 - full | 50 | +0.16 | [+0.06, +0.26] | 8 / 0 |
| warm_t0.1 - plain_k1 | 50 | -0.86 | [-0.94, -0.76] | 0 / 43 |
| warm_t0.1 - warmreset_t0.1 | 50 | -0.88 | [-0.96, -0.78] | 0 / 44 |
| warm_t0.1 - full | 50 | -0.72 | [-0.84, -0.58] | 0 / 36 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 8 | 0.565 |
| plain_k1 | 6 | 0.323 |
| warm_t0.1 | 6 | 0.210 |
| warmreset_t0.1 | 2 | 0.610 |
| warmshoot_t0.1 | 2 | 0.440 |
| resetfinal_t0.1 | 8 | 0.435 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| resetfinal_t0.1 - full | 8 | -0.130 | [-0.183, -0.078] |
| resetfinal_t0.1 - plain_k1 | 6 | +0.147 | [+0.103, +0.193] |
| resetfinal_t0.1 - warm_t0.1 | 6 | +0.260 | [+0.203, +0.313] |
| warm_t0.1 - plain_k1 | 6 | -0.113 | [-0.160, -0.063] |
| warm_t0.1 - full | 6 | -0.430 | [-0.490, -0.370] |
| warmreset_t0.1 - plain_k1 | 2 | +0.110 | [+0.050, +0.180] |
| warmreset_t0.1 - warm_t0.1 | 2 | +0.440 | [+0.340, +0.540] |
| warmreset_t0.1 - full | 2 | -0.120 | [-0.220, -0.010] |
| warmshoot_t0.1 - plain_k1 | 2 | -0.060 | [-0.130, +0.010] |
| warmshoot_t0.1 - warm_t0.1 | 2 | +0.270 | [+0.160, +0.370] |
| warmshoot_t0.1 - warmreset_t0.1 | 2 | -0.170 | [-0.260, -0.080] |
| warmshoot_t0.1 - full | 2 | -0.290 | [-0.400, -0.180] |
| resetfinal_t0.1 - warmreset_t0.1 | 2 | +0.010 | [-0.050, +0.070] |
| warm_t0.1 - warmreset_t0.1 | 2 | -0.440 | [-0.540, -0.340] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
