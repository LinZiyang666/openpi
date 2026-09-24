## Warm-start continuation variants — pi05 (m=3, t=0.3; descriptive, 95% paired bootstrap)

missing arms: warmshoot_t0.3

### CloseFridge

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 10.0 | 0.0 | – |
| plain_k3 | 50 | 0.14 | ok | 3.0 | 0.0 | – |
| warm_t0.3 | 50 | 0.30 | ok | 3.0 | 0.0 | – |
| warmreset_t0.3 | 50 | 0.66 | ok | 3.0 | 0.0 | – |
| resetfinal_t0.3 | 50 | 0.84 | ok | 3.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.3 - plain_k3 | 50 | +0.52 | [+0.36, +0.66] | 27 / 1 |
| warmreset_t0.3 - warm_t0.3 | 50 | +0.36 | [+0.20, +0.52] | 21 / 3 |
| warmreset_t0.3 - full | 50 | +0.04 | [-0.16, +0.24] | 14 / 12 |
| resetfinal_t0.3 - plain_k3 | 50 | +0.70 | [+0.56, +0.82] | 36 / 1 |
| resetfinal_t0.3 - warm_t0.3 | 50 | +0.54 | [+0.38, +0.70] | 29 / 2 |
| resetfinal_t0.3 - warmreset_t0.3 | 50 | +0.18 | [+0.00, +0.36] | 15 / 6 |
| resetfinal_t0.3 - full | 50 | +0.22 | [+0.04, +0.40] | 17 / 6 |
| warm_t0.3 - plain_k3 | 50 | +0.16 | [+0.02, +0.32] | 12 / 4 |
| warm_t0.3 - warmreset_t0.3 | 50 | -0.36 | [-0.52, -0.20] | 3 / 21 |
| warm_t0.3 - full | 50 | -0.32 | [-0.52, -0.12] | 7 / 23 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 3, serving runtimes 3 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.84 | ok | 10.0 | 0.0 | – |
| plain_k3 | 50 | 0.94 | ok | 3.0 | 0.0 | – |
| warm_t0.3 | 50 | 0.16 | ok | 3.0 | 0.0 | – |
| warmreset_t0.3 | 50 | 0.64 | ok | 3.0 | 0.0 | – |
| resetfinal_t0.3 | 50 | 0.42 | ok | 3.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.3 - plain_k3 | 50 | -0.30 | [-0.44, -0.16] | 1 / 16 |
| warmreset_t0.3 - warm_t0.3 | 50 | +0.48 | [+0.34, +0.62] | 25 / 1 |
| warmreset_t0.3 - full | 50 | -0.20 | [-0.38, -0.02] | 6 / 16 |
| resetfinal_t0.3 - plain_k3 | 50 | -0.52 | [-0.66, -0.36] | 1 / 27 |
| resetfinal_t0.3 - warm_t0.3 | 50 | +0.26 | [+0.10, +0.42] | 16 / 3 |
| resetfinal_t0.3 - warmreset_t0.3 | 50 | -0.22 | [-0.36, -0.08] | 2 / 13 |
| resetfinal_t0.3 - full | 50 | -0.42 | [-0.58, -0.24] | 3 / 24 |
| warm_t0.3 - plain_k3 | 50 | -0.78 | [-0.88, -0.66] | 0 / 39 |
| warm_t0.3 - warmreset_t0.3 | 50 | -0.48 | [-0.64, -0.32] | 1 / 25 |
| warm_t0.3 - full | 50 | -0.68 | [-0.82, -0.52] | 2 / 36 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 2 | 0.730 |
| plain_k3 | 2 | 0.540 |
| warm_t0.3 | 2 | 0.230 |
| warmreset_t0.3 | 2 | 0.650 |
| warmshoot_t0.3 | 0 | – |
| resetfinal_t0.3 | 2 | 0.630 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.3 - plain_k3 | 2 | +0.110 | [+0.010, +0.210] |
| warmreset_t0.3 - warm_t0.3 | 2 | +0.420 | [+0.310, +0.530] |
| warmreset_t0.3 - full | 2 | -0.080 | [-0.210, +0.050] |
| resetfinal_t0.3 - plain_k3 | 2 | +0.090 | [-0.010, +0.190] |
| resetfinal_t0.3 - warm_t0.3 | 2 | +0.400 | [+0.290, +0.510] |
| resetfinal_t0.3 - warmreset_t0.3 | 2 | -0.020 | [-0.130, +0.090] |
| resetfinal_t0.3 - full | 2 | -0.100 | [-0.220, +0.020] |
| warm_t0.3 - plain_k3 | 2 | -0.310 | [-0.400, -0.210] |
| warm_t0.3 - warmreset_t0.3 | 2 | -0.420 | [-0.530, -0.310] |
| warm_t0.3 - full | 2 | -0.500 | [-0.620, -0.370] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
