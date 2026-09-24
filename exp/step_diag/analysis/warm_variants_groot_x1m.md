## Warm-start continuation variants — groot (m=1, t=0.75; descriptive, 95% paired bootstrap)

missing arms: full, warmshoot_t0.75

### TurnOnSinkFaucet

cross-arm model/env identity: identical; worker islands 1, serving runtimes 1 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k1 | 50 | 0.08 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.34 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.14 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.20 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.20 | [-0.36, -0.04] | 5 / 15 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.12 | [+0.00, +0.26] | 9 / 3 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.14 | [-0.32, +0.04] | 7 / 14 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.06 | [-0.06, +0.18] | 7 / 4 |
| warm_t0.75 - plain_k1 | 50 | +0.26 | [+0.10, +0.42] | 16 / 3 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.20 | [+0.04, +0.36] | 15 / 5 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 1, serving runtimes 1 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k1 | 50 | 0.90 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.52 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.92 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.86 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.02 | [-0.04, +0.08] | 2 / 1 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.40 | [+0.26, +0.54] | 21 / 1 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.04 | [-0.12, +0.04] | 1 / 3 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.34 | [+0.16, +0.52] | 21 / 4 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | -0.06 | [-0.14, +0.02] | 1 / 4 |
| warm_t0.75 - plain_k1 | 50 | -0.38 | [-0.54, -0.22] | 3 / 22 |
| warm_t0.75 - warmreset_t0.75 | 50 | -0.40 | [-0.54, -0.26] | 1 / 21 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 0 | – |
| plain_k1 | 2 | 0.490 |
| warm_t0.75 | 2 | 0.430 |
| warmreset_t0.75 | 2 | 0.530 |
| warmshoot_t0.75 | 0 | – |
| resetfinal_t0.75 | 2 | 0.530 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 2 | +0.040 | [-0.030, +0.110] |
| warmreset_t0.75 - warm_t0.75 | 2 | +0.100 | [-0.010, +0.210] |
| resetfinal_t0.75 - plain_k1 | 2 | +0.040 | [-0.040, +0.110] |
| resetfinal_t0.75 - warm_t0.75 | 2 | +0.100 | [-0.020, +0.220] |
| resetfinal_t0.75 - warmreset_t0.75 | 2 | +0.000 | [-0.080, +0.080] |
| warm_t0.75 - plain_k1 | 2 | -0.060 | [-0.170, +0.050] |
| warm_t0.75 - warmreset_t0.75 | 2 | -0.100 | [-0.210, +0.010] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
