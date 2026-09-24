## Warm-start continuation variants — pi05 (m=2, t=0.2; descriptive, 95% paired bootstrap)

### CloseFridge

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.08 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.30 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.52 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.44 | [+0.30, +0.58] | 23 / 1 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.22 | [+0.04, +0.40] | 18 / 7 |
| warmreset_t0.2 - full | 50 | -0.10 | [-0.30, +0.10] | 10 / 15 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.30 | [-0.42, -0.18] | 0 / 15 |
| warmshoot_t0.2 - full | 50 | -0.62 | [-0.74, -0.48] | 0 / 31 |
| warm_t0.2 - plain_k2 | 50 | +0.22 | [+0.06, +0.38] | 15 / 4 |
| warm_t0.2 - full | 50 | -0.32 | [-0.50, -0.12] | 6 / 22 |

### PickPlaceCounterToStove

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.84 | ok | 10.0 | 0.0 | – |
| plain_k2 | 100 | 0.94 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 100 | 0.16 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | -0.12 | [-0.24, +0.00] | 2 / 8 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.66 | [+0.50, +0.80] | 35 / 2 |
| warmreset_t0.2 - full | 50 | -0.02 | [-0.16, +0.12] | 6 / 7 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.94 | [-1.00, -0.86] | 0 / 47 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.16 | [-0.26, -0.06] | 0 / 8 |
| warmshoot_t0.2 - full | 50 | -0.84 | [-0.94, -0.74] | 0 / 42 |
| warm_t0.2 - plain_k2 | 100 | -0.78 | [-0.86, -0.70] | 0 / 78 |
| warm_t0.2 - full | 50 | -0.68 | [-0.82, -0.54] | 1 / 35 |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`. n10 = variant success / reference failure; n01 the reverse.
