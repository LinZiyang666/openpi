## Warm-start continuation variants — pi05 (m=2, t=0.2; descriptive, 95% paired bootstrap; init_idx >= 50)

missing arms: full, resetfinal_t0.2

### CloseFridge

cross-arm model/env identity: identical; worker islands 3, serving runtimes 3 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k2 | 450 | 0.06 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 450 | 0.29 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 450 | 0.62 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 450 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 450 | +0.56 | [+0.51, +0.61] | 257 / 5 |
| warmreset_t0.2 - warm_t0.2 | 450 | +0.34 | [+0.27, +0.40] | 196 / 45 |
| warmshoot_t0.2 - plain_k2 | 450 | -0.06 | [-0.08, -0.04] | 0 / 28 |
| warmshoot_t0.2 - warm_t0.2 | 450 | -0.29 | [-0.33, -0.24] | 0 / 129 |
| warmshoot_t0.2 - warmreset_t0.2 | 450 | -0.62 | [-0.67, -0.58] | 0 / 280 |
| warm_t0.2 - plain_k2 | 450 | +0.22 | [+0.18, +0.27] | 125 / 24 |
| warm_t0.2 - warmreset_t0.2 | 450 | -0.34 | [-0.40, -0.28] | 45 / 196 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k2 | 450 | 0.95 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 450 | 0.17 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 450 | 0.78 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 450 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 450 | -0.16 | [-0.21, -0.12] | 16 / 90 |
| warmreset_t0.2 - warm_t0.2 | 450 | +0.62 | [+0.57, +0.66] | 283 / 6 |
| warmshoot_t0.2 - plain_k2 | 450 | -0.95 | [-0.97, -0.93] | 0 / 427 |
| warmshoot_t0.2 - warm_t0.2 | 450 | -0.17 | [-0.20, -0.14] | 0 / 76 |
| warmshoot_t0.2 - warmreset_t0.2 | 450 | -0.78 | [-0.82, -0.75] | 0 / 353 |
| warm_t0.2 - plain_k2 | 450 | -0.78 | [-0.82, -0.74] | 0 / 351 |
| warm_t0.2 - warmreset_t0.2 | 450 | -0.62 | [-0.66, -0.57] | 6 / 283 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 0 | – |
| plain_k2 | 2 | 0.506 |
| warm_t0.2 | 2 | 0.228 |
| warmreset_t0.2 | 2 | 0.703 |
| warmshoot_t0.2 | 2 | 0.000 |
| resetfinal_t0.2 | 0 | – |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 2 | +0.198 | [+0.166, +0.230] |
| warmreset_t0.2 - warm_t0.2 | 2 | +0.476 | [+0.438, +0.513] |
| warmshoot_t0.2 - plain_k2 | 2 | -0.506 | [-0.521, -0.490] |
| warmshoot_t0.2 - warm_t0.2 | 2 | -0.228 | [-0.254, -0.201] |
| warmshoot_t0.2 - warmreset_t0.2 | 2 | -0.703 | [-0.732, -0.673] |
| warm_t0.2 - plain_k2 | 2 | -0.278 | [-0.309, -0.247] |
| warm_t0.2 - warmreset_t0.2 | 2 | -0.476 | [-0.513, -0.438] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
