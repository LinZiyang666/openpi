## Warm-start continuation variants — pi05 (m=2, t=0.2; descriptive, 95% paired bootstrap)

missing arms: full, resetfinal_t0.2

### CloseFridge

cross-arm model/env identity: identical; worker islands 3, serving runtimes 3 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k2 | 500 | 0.06 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 500 | 0.29 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 500 | 0.61 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 500 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 500 | +0.54 | [+0.50, +0.59] | 279 / 7 |
| warmreset_t0.2 - warm_t0.2 | 500 | +0.32 | [+0.26, +0.38] | 213 / 52 |
| warmshoot_t0.2 - plain_k2 | 500 | -0.06 | [-0.09, -0.04] | 0 / 32 |
| warmshoot_t0.2 - warm_t0.2 | 500 | -0.29 | [-0.33, -0.25] | 0 / 143 |
| warmshoot_t0.2 - warmreset_t0.2 | 500 | -0.61 | [-0.65, -0.57] | 0 / 304 |
| warm_t0.2 - plain_k2 | 500 | +0.22 | [+0.18, +0.27] | 139 / 28 |
| warm_t0.2 - warmreset_t0.2 | 500 | -0.32 | [-0.38, -0.26] | 52 / 213 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| plain_k2 | 500 | 0.95 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 500 | 0.16 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 500 | 0.79 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 500 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 500 | -0.16 | [-0.20, -0.12] | 18 / 97 |
| warmreset_t0.2 - warm_t0.2 | 500 | +0.63 | [+0.58, +0.67] | 320 / 6 |
| warmshoot_t0.2 - plain_k2 | 500 | -0.95 | [-0.97, -0.93] | 0 / 474 |
| warmshoot_t0.2 - warm_t0.2 | 500 | -0.16 | [-0.19, -0.13] | 0 / 81 |
| warmshoot_t0.2 - warmreset_t0.2 | 500 | -0.79 | [-0.83, -0.75] | 0 / 395 |
| warm_t0.2 - plain_k2 | 500 | -0.79 | [-0.82, -0.75] | 0 / 393 |
| warm_t0.2 - warmreset_t0.2 | 500 | -0.63 | [-0.67, -0.58] | 6 / 320 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 0 | – |
| plain_k2 | 2 | 0.506 |
| warm_t0.2 | 2 | 0.224 |
| warmreset_t0.2 | 2 | 0.699 |
| warmshoot_t0.2 | 2 | 0.000 |
| resetfinal_t0.2 | 0 | – |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 2 | +0.193 | [+0.163, +0.223] |
| warmreset_t0.2 - warm_t0.2 | 2 | +0.475 | [+0.438, +0.511] |
| warmshoot_t0.2 - plain_k2 | 2 | -0.506 | [-0.521, -0.492] |
| warmshoot_t0.2 - warm_t0.2 | 2 | -0.224 | [-0.250, -0.199] |
| warmshoot_t0.2 - warmreset_t0.2 | 2 | -0.699 | [-0.726, -0.671] |
| warm_t0.2 - plain_k2 | 2 | -0.282 | [-0.312, -0.253] |
| warm_t0.2 - warmreset_t0.2 | 2 | -0.475 | [-0.511, -0.439] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
