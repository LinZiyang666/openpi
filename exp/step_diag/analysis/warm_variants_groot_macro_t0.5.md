## Warm-start continuation variants — groot (m=2, t=0.5; descriptive, 95% paired bootstrap)

missing arms: warmshoot_t0.5

### OpenCabinet

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.90 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.90 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.88 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.92 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.06 | [-0.06, +0.18] | 7 / 4 |
| warmreset_t0.5 - full | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.10 | [-0.02, +0.22] | 7 / 2 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| resetfinal_t0.5 - full | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| warm_t0.5 - plain_k2 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.20, +0.06] | 4 / 7 |
| warm_t0.5 - full | 50 | -0.08 | [-0.20, +0.04] | 3 / 7 |

### SlideDishwasherRack

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.54 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.46 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.44 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.44 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.50 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| warmreset_t0.5 - full | 50 | -0.10 | [-0.22, +0.02] | 3 / 8 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.04 | [-0.06, +0.14] | 4 / 2 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.06 | [-0.02, +0.14] | 4 / 1 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.06 | [-0.04, +0.16] | 5 / 2 |
| resetfinal_t0.5 - full | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| warm_t0.5 - plain_k2 | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| warm_t0.5 - warmreset_t0.5 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| warm_t0.5 - full | 50 | -0.10 | [-0.22, +0.00] | 2 / 7 |

### TurnOnSinkFaucet

cross-arm model/env identity: identical; worker islands 3, serving runtimes 3 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.18 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.36 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.30 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.38 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | +0.06 | [-0.12, +0.22] | 11 / 8 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.12 | [-0.06, +0.30] | 15 / 9 |
| warmreset_t0.5 - full | 50 | +0.24 | [+0.08, +0.40] | 16 / 4 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 10 / 6 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.22, +0.14] | 10 / 12 |
| resetfinal_t0.5 - full | 50 | +0.20 | [+0.06, +0.34] | 13 / 3 |
| warm_t0.5 - plain_k2 | 50 | -0.06 | [-0.24, +0.12] | 9 / 12 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.12 | [-0.30, +0.06] | 9 / 15 |
| warm_t0.5 - full | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |

### PickPlaceCounterToCabinet

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 4.0 | 0.0 | – |
| plain_k2 | 0 | – | NO | – | None | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.44 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.48 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - full | 50 | -0.18 | [-0.34, -0.02] | 4 / 13 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| resetfinal_t0.5 - full | 50 | -0.14 | [-0.30, +0.02] | 6 / 13 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.82 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.96 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.88 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.86 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| warmreset_t0.5 - full | 50 | +0.06 | [-0.06, +0.18] | 7 / 4 |
| resetfinal_t0.5 - plain_k2 | 50 | -0.10 | [-0.18, -0.02] | 0 / 5 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.12, +0.06] | 2 / 3 |
| resetfinal_t0.5 - full | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| warm_t0.5 - plain_k2 | 50 | -0.14 | [-0.26, -0.04] | 1 / 8 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| warm_t0.5 - full | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |

### PickPlaceDrawerToCounter

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.66 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.50 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.54 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.58 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| warmreset_t0.5 - full | 50 | -0.12 | [-0.26, +0.02] | 4 / 10 |
| resetfinal_t0.5 - plain_k2 | 50 | -0.02 | [-0.18, +0.14] | 7 / 8 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| resetfinal_t0.5 - full | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| warm_t0.5 - plain_k2 | 50 | -0.10 | [-0.28, +0.08] | 8 / 13 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| warm_t0.5 - full | 50 | -0.16 | [-0.34, +0.02] | 7 / 15 |

### PickPlaceSinkToCounter

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.78 | ok | 4.0 | 0.0 | – |
| plain_k2 | 0 | – | NO | – | None | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 1.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.92 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - full | 50 | +0.22 | [+0.12, +0.34] | 11 / 0 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| resetfinal_t0.5 - full | 50 | +0.14 | [+0.04, +0.26] | 8 / 1 |

### PickPlaceToasterToCounter

cross-arm model/env identity: identical; worker islands 2, serving runtimes 2 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.64 | ok | 4.0 | 0.0 | – |
| plain_k2 | 0 | – | NO | – | None | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.62 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - full | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| resetfinal_t0.5 - full | 50 | -0.02 | [-0.18, +0.16] | 9 / 10 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 8 | 0.643 |
| plain_k2 | 5 | 0.656 |
| warm_t0.5 | 5 | 0.576 |
| warmreset_t0.5 | 8 | 0.650 |
| warmshoot_t0.5 | 0 | – |
| resetfinal_t0.5 | 8 | 0.657 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 5 | -0.024 | [-0.080, +0.032] |
| warmreset_t0.5 - warm_t0.5 | 5 | +0.056 | [-0.004, +0.116] |
| warmreset_t0.5 - full | 8 | +0.007 | [-0.040, +0.055] |
| resetfinal_t0.5 - plain_k2 | 5 | -0.008 | [-0.060, +0.048] |
| resetfinal_t0.5 - warm_t0.5 | 5 | +0.072 | [+0.012, +0.132] |
| resetfinal_t0.5 - warmreset_t0.5 | 8 | +0.008 | [-0.035, +0.050] |
| resetfinal_t0.5 - full | 8 | +0.015 | [-0.033, +0.065] |
| warm_t0.5 - plain_k2 | 5 | -0.080 | [-0.144, -0.020] |
| warm_t0.5 - warmreset_t0.5 | 5 | -0.056 | [-0.120, +0.004] |
| warm_t0.5 - full | 5 | -0.044 | [-0.108, +0.016] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
