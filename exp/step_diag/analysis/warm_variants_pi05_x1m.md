## Warm-start continuation variants — pi05 (m=2, t=0.2; descriptive, 95% paired bootstrap)

missing arms: full, resetfinal_t0.2

### CloseFridge

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k2 | 50 | 0.04 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.32 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.66 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.62 | [+0.48, +0.76] | 32 / 1 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.34 | [+0.16, +0.52] | 22 / 5 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.04 | [-0.10, +0.00] | 0 / 2 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.32 | [-0.46, -0.20] | 0 / 16 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.66 | [-0.78, -0.52] | 0 / 33 |
| warm_t0.2 - plain_k2 | 50 | +0.28 | [+0.16, +0.40] | 14 / 0 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.34 | [-0.52, -0.16] | 5 / 22 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k2 | 50 | 0.92 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.14 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.80 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | -0.12 | [-0.26, +0.00] | 3 / 9 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.66 | [+0.52, +0.78] | 33 / 0 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.92 | [-0.98, -0.84] | 0 / 46 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.14 | [-0.24, -0.06] | 0 / 7 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.80 | [-0.90, -0.68] | 0 / 40 |
| warm_t0.2 - plain_k2 | 50 | -0.78 | [-0.90, -0.64] | 1 / 40 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.66 | [-0.78, -0.52] | 0 / 33 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 0 | – |
| plain_k2 | 2 | 0.480 |
| warm_t0.2 | 2 | 0.230 |
| warmreset_t0.2 | 2 | 0.730 |
| warmshoot_t0.2 | 2 | 0.000 |
| resetfinal_t0.2 | 0 | – |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 2 | +0.250 | [+0.150, +0.350] |
| warmreset_t0.2 - warm_t0.2 | 2 | +0.500 | [+0.390, +0.610] |
| warmshoot_t0.2 - plain_k2 | 2 | -0.480 | [-0.520, -0.430] |
| warmshoot_t0.2 - warm_t0.2 | 2 | -0.230 | [-0.310, -0.150] |
| warmshoot_t0.2 - warmreset_t0.2 | 2 | -0.730 | [-0.810, -0.640] |
| warm_t0.2 - plain_k2 | 2 | -0.250 | [-0.340, -0.160] |
| warm_t0.2 - warmreset_t0.2 | 2 | -0.500 | [-0.610, -0.390] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
