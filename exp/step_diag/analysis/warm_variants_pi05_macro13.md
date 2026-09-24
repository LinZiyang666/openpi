## Warm-start continuation variants — pi05 (m=2, t=0.2; descriptive, 95% paired bootstrap)

### CloseBlenderLid

cross-arm model/env identity: identical; worker islands 5, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.14 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.04 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.02 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.34 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.34 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.34 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.30 | [+0.18, +0.44] | 15 / 0 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.32 | [+0.18, +0.46] | 17 / 1 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| warmreset_t0.2 - full | 50 | +0.20 | [+0.02, +0.38] | 16 / 6 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.04 | [-0.10, +0.00] | 0 / 2 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.02 | [-0.06, +0.00] | 0 / 1 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.34 | [-0.48, -0.22] | 0 / 17 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.34 | [-0.48, -0.22] | 0 / 17 |
| warmshoot_t0.2 - full | 50 | -0.14 | [-0.24, -0.06] | 0 / 7 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.30 | [+0.14, +0.44] | 17 / 2 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.32 | [+0.20, +0.46] | 16 / 0 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| resetfinal_t0.2 - full | 50 | +0.20 | [+0.02, +0.38] | 16 / 6 |
| midfinal_t0.2 - plain_k2 | 50 | +0.30 | [+0.16, +0.44] | 16 / 1 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.32 | [+0.18, +0.46] | 17 / 1 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | +0.00 | [-0.18, +0.18] | 11 / 11 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | +0.00 | [-0.20, +0.20] | 13 / 13 |
| midfinal_t0.2 - full | 50 | +0.20 | [+0.04, +0.36] | 15 / 5 |
| warm_t0.2 - plain_k2 | 50 | -0.02 | [-0.08, +0.04] | 1 / 2 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.32 | [-0.46, -0.18] | 1 / 17 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.32 | [-0.46, -0.20] | 0 / 16 |
| warm_t0.2 - full | 50 | -0.12 | [-0.22, -0.02] | 1 / 7 |

### CloseFridge

cross-arm model/env identity: identical; worker islands 5, serving runtimes 5 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.08 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.30 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 100 (50 ids × replicates) | 0.51 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 100 (50 ids × replicates) | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.76 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.58 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.43 | [+0.29, +0.57] | 27 / 2 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.21 | [+0.03, +0.38] | 19 / 10 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | -0.25 | [-0.39, -0.11] | 4 / 18 |
| warmreset_t0.2 - full | 50 | -0.11 | [-0.29, +0.07] | 10 / 19 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.30 | [-0.42, -0.18] | 0 / 15 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.51 | [-0.63, -0.38] | 0 / 30 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.76 | [-0.88, -0.64] | 0 / 38 |
| warmshoot_t0.2 - full | 50 | -0.62 | [-0.76, -0.48] | 0 / 31 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.68 | [+0.54, +0.82] | 35 / 1 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.46 | [+0.28, +0.62] | 26 / 3 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | +0.25 | [+0.11, +0.39] | 18 / 4 |
| resetfinal_t0.2 - full | 50 | +0.14 | [-0.04, +0.32] | 14 / 7 |
| midfinal_t0.2 - plain_k2 | 50 | +0.50 | [+0.36, +0.64] | 26 / 1 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.28 | [+0.08, +0.48] | 22 / 8 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | +0.07 | [-0.12, +0.26] | 17 / 13 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.18 | [-0.34, +0.00] | 6 / 15 |
| midfinal_t0.2 - full | 50 | -0.04 | [-0.24, +0.16] | 12 / 14 |
| warm_t0.2 - plain_k2 | 50 | +0.22 | [+0.06, +0.38] | 15 / 4 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.21 | [-0.39, -0.03] | 10 / 19 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.46 | [-0.62, -0.28] | 3 / 26 |
| warm_t0.2 - full | 50 | -0.32 | [-0.50, -0.12] | 6 / 22 |

### CoffeeSetupMug

cross-arm model/env identity: identical; worker islands 5, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.40 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.36 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.04 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.48 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.50 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.06 | [-0.14, +0.26] | 16 / 13 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.38 | [+0.22, +0.54] | 21 / 2 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| warmreset_t0.2 - full | 50 | +0.02 | [-0.18, +0.20] | 12 / 11 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.36 | [-0.50, -0.24] | 0 / 18 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.04 | [-0.10, +0.00] | 0 / 2 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.42 | [-0.56, -0.28] | 0 / 21 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.48 | [-0.62, -0.34] | 0 / 24 |
| warmshoot_t0.2 - full | 50 | -0.40 | [-0.54, -0.26] | 0 / 20 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.12 | [-0.06, +0.30] | 15 / 9 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.44 | [+0.30, +0.58] | 23 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | +0.06 | [-0.10, +0.22] | 10 / 7 |
| resetfinal_t0.2 - full | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| midfinal_t0.2 - plain_k2 | 50 | +0.14 | [-0.06, +0.34] | 17 / 10 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.46 | [+0.32, +0.60] | 23 / 0 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | +0.08 | [-0.12, +0.28] | 15 / 11 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | +0.02 | [-0.16, +0.20] | 12 / 11 |
| midfinal_t0.2 - full | 50 | +0.10 | [-0.10, +0.30] | 15 / 10 |
| warm_t0.2 - plain_k2 | 50 | -0.32 | [-0.46, -0.18] | 1 / 17 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.38 | [-0.54, -0.22] | 2 / 21 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.44 | [-0.58, -0.30] | 1 / 23 |
| warm_t0.2 - full | 50 | -0.36 | [-0.52, -0.20] | 2 / 20 |

### OpenCabinet

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.66 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.50 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.90 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.86 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.48 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.40 | [+0.26, +0.54] | 20 / 0 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.48 | [+0.32, +0.64] | 25 / 1 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| warmreset_t0.2 - full | 50 | +0.24 | [+0.08, +0.40] | 15 / 3 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.50 | [-0.64, -0.36] | 0 / 25 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.42 | [-0.56, -0.28] | 0 / 21 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.90 | [-0.98, -0.80] | 0 / 45 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.86 | [-0.94, -0.76] | 0 / 43 |
| warmshoot_t0.2 - full | 50 | -0.66 | [-0.78, -0.52] | 0 / 33 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.36 | [+0.20, +0.52] | 20 / 2 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.44 | [+0.30, +0.58] | 23 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| resetfinal_t0.2 - full | 50 | +0.20 | [+0.06, +0.34] | 13 / 3 |
| midfinal_t0.2 - plain_k2 | 50 | -0.02 | [-0.18, +0.16] | 9 / 10 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.06 | [-0.08, +0.22] | 9 / 6 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.42 | [-0.56, -0.28] | 0 / 21 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.38 | [-0.54, -0.22] | 3 / 22 |
| midfinal_t0.2 - full | 50 | -0.18 | [-0.36, +0.02] | 8 / 17 |
| warm_t0.2 - plain_k2 | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.48 | [-0.62, -0.34] | 1 / 25 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.44 | [-0.58, -0.30] | 1 / 23 |
| warm_t0.2 - full | 50 | -0.24 | [-0.38, -0.10] | 2 / 14 |

### OpenDrawer

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 10.0 | 0.0 | – |
| plain_k2 | 100 | 0.75 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 100 | 0.71 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.88 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.82 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.14 | [+0.04, +0.26] | 8 / 1 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.20 | [+0.10, +0.32] | 10 / 0 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.06 | [-0.02, +0.16] | 4 / 1 |
| warmreset_t0.2 - full | 50 | +0.26 | [+0.12, +0.40] | 15 / 2 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.74 | [-0.86, -0.62] | 0 / 37 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.68 | [-0.80, -0.54] | 0 / 34 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.88 | [-0.96, -0.78] | 0 / 44 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.82 | [-0.92, -0.70] | 0 / 41 |
| warmshoot_t0.2 - full | 50 | -0.62 | [-0.74, -0.48] | 0 / 31 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.14 | [+0.04, +0.26] | 8 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.06 | [-0.16, +0.02] | 1 / 4 |
| resetfinal_t0.2 - full | 50 | +0.20 | [+0.06, +0.34] | 13 / 3 |
| midfinal_t0.2 - plain_k2 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.14 | [+0.00, +0.28] | 11 / 4 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | +0.00 | [-0.12, +0.10] | 4 / 4 |
| midfinal_t0.2 - full | 50 | +0.20 | [+0.06, +0.34] | 13 / 3 |
| warm_t0.2 - plain_k2 | 100 | -0.04 | [-0.14, +0.06] | 10 / 14 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.20 | [-0.32, -0.10] | 0 / 10 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.14 | [-0.26, -0.04] | 1 / 8 |
| warm_t0.2 - full | 50 | +0.06 | [-0.08, +0.22] | 9 / 6 |

### OpenStandMixerHead

cross-arm model/env identity: identical; worker islands 5, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.16 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.20 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.44 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.72 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.68 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.48 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.52 | [+0.36, +0.66] | 27 / 1 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.28 | [+0.14, +0.42] | 15 / 1 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.04 | [-0.06, +0.16] | 5 / 3 |
| warmreset_t0.2 - full | 50 | +0.56 | [+0.40, +0.70] | 29 / 1 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.20 | [-0.32, -0.10] | 0 / 10 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.44 | [-0.58, -0.30] | 0 / 22 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.72 | [-0.84, -0.60] | 0 / 36 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.68 | [-0.80, -0.54] | 0 / 34 |
| warmshoot_t0.2 - full | 50 | -0.16 | [-0.26, -0.06] | 0 / 8 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.48 | [+0.32, +0.62] | 25 / 1 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.24 | [+0.12, +0.36] | 13 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.04 | [-0.14, +0.06] | 3 / 5 |
| resetfinal_t0.2 - full | 50 | +0.52 | [+0.34, +0.68] | 29 / 3 |
| midfinal_t0.2 - plain_k2 | 50 | +0.28 | [+0.12, +0.44] | 18 / 4 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.24 | [-0.38, -0.12] | 1 / 13 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.20 | [-0.34, -0.06] | 3 / 13 |
| midfinal_t0.2 - full | 50 | +0.32 | [+0.16, +0.48] | 19 / 3 |
| warm_t0.2 - plain_k2 | 50 | +0.24 | [+0.08, +0.40] | 16 / 4 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.28 | [-0.42, -0.14] | 1 / 15 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.24 | [-0.38, -0.12] | 1 / 13 |
| warm_t0.2 - full | 50 | +0.28 | [+0.12, +0.44] | 18 / 4 |

### SlideDishwasherRack

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.56 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.88 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.06 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.92 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.76 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.28 | [+0.14, +0.42] | 16 / 2 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.28 | [+0.14, +0.42] | 16 / 2 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | -0.04 | [-0.10, +0.00] | 0 / 2 |
| warmreset_t0.2 - full | 50 | +0.32 | [+0.16, +0.48] | 19 / 3 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.54 | [-0.68, -0.40] | 0 / 27 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.54 | [-0.68, -0.38] | 1 / 28 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.82 | [-0.92, -0.70] | 0 / 41 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.86 | [-0.94, -0.76] | 0 / 43 |
| warmshoot_t0.2 - full | 50 | -0.50 | [-0.64, -0.34] | 1 / 26 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.32 | [+0.18, +0.46] | 17 / 1 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.32 | [+0.18, +0.48] | 18 / 2 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | +0.04 | [+0.00, +0.10] | 2 / 0 |
| resetfinal_t0.2 - full | 50 | +0.36 | [+0.20, +0.52] | 20 / 2 |
| midfinal_t0.2 - plain_k2 | 50 | +0.16 | [-0.02, +0.34] | 14 / 6 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.16 | [+0.00, +0.32] | 14 / 6 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.12 | [-0.24, -0.02] | 1 / 7 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.16 | [-0.26, -0.06] | 0 / 8 |
| midfinal_t0.2 - full | 50 | +0.20 | [+0.04, +0.36] | 14 / 4 |
| warm_t0.2 - plain_k2 | 50 | +0.00 | [-0.16, +0.16] | 8 / 8 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.28 | [-0.42, -0.14] | 2 / 16 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.32 | [-0.48, -0.16] | 2 / 18 |
| warm_t0.2 - full | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |

### TurnOnSinkFaucet

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.86 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.74 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.22 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.94 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.88 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.78 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.20 | [+0.08, +0.32] | 11 / 1 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.72 | [+0.58, +0.84] | 37 / 1 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.06 | [-0.02, +0.14] | 4 / 1 |
| warmreset_t0.2 - full | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.74 | [-0.86, -0.62] | 0 / 37 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.22 | [-0.34, -0.12] | 0 / 11 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.94 | [-1.00, -0.86] | 0 / 47 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.88 | [-0.96, -0.78] | 0 / 44 |
| warmshoot_t0.2 - full | 50 | -0.86 | [-0.94, -0.76] | 0 / 43 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.14 | [+0.00, +0.28] | 10 / 3 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.66 | [+0.52, +0.80] | 34 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.06 | [-0.14, +0.02] | 1 / 4 |
| resetfinal_t0.2 - full | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midfinal_t0.2 - plain_k2 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.56 | [+0.38, +0.72] | 31 / 3 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.16 | [-0.30, -0.04] | 2 / 10 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.10 | [-0.26, +0.04] | 5 / 10 |
| midfinal_t0.2 - full | 50 | -0.08 | [-0.24, +0.08] | 7 / 11 |
| warm_t0.2 - plain_k2 | 50 | -0.52 | [-0.68, -0.36] | 2 / 28 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.72 | [-0.84, -0.58] | 1 / 37 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.66 | [-0.80, -0.52] | 1 / 34 |
| warm_t0.2 - full | 50 | -0.64 | [-0.78, -0.50] | 1 / 33 |

### PickPlaceCounterToCabinet

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.54 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.58 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.14 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.68 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.64 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.30 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.10 | [-0.08, +0.28] | 13 / 8 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.54 | [+0.36, +0.70] | 30 / 3 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| warmreset_t0.2 - full | 50 | +0.14 | [-0.04, +0.32] | 14 / 7 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.58 | [-0.72, -0.44] | 0 / 29 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.14 | [-0.24, -0.06] | 0 / 7 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.68 | [-0.80, -0.54] | 0 / 34 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.64 | [-0.76, -0.50] | 0 / 32 |
| warmshoot_t0.2 - full | 50 | -0.54 | [-0.68, -0.40] | 0 / 27 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.06 | [-0.14, +0.26] | 14 / 11 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.50 | [+0.36, +0.64] | 26 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| resetfinal_t0.2 - full | 50 | +0.10 | [-0.08, +0.28] | 13 / 8 |
| midfinal_t0.2 - plain_k2 | 50 | -0.28 | [-0.46, -0.10] | 5 / 19 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.16 | [+0.00, +0.32] | 13 / 5 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.38 | [-0.56, -0.20] | 4 / 23 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.34 | [-0.50, -0.18] | 3 / 20 |
| midfinal_t0.2 - full | 50 | -0.24 | [-0.38, -0.10] | 2 / 14 |
| warm_t0.2 - plain_k2 | 50 | -0.44 | [-0.60, -0.26] | 4 / 26 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.54 | [-0.70, -0.36] | 3 / 30 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.50 | [-0.64, -0.34] | 1 / 26 |
| warm_t0.2 - full | 50 | -0.40 | [-0.56, -0.24] | 3 / 23 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 5, serving runtimes 5 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.84 | ok | 10.0 | 0.0 | – |
| plain_k2 | 100 | 0.94 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 100 | 0.16 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 100 (50 ids × replicates) | 0.82 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 100 (50 ids × replicates) | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.72 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.32 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | -0.12 | [-0.24, -0.01] | 2 / 9 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.66 | [+0.50, +0.80] | 36 / 2 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.10 | [-0.02, +0.23] | 9 / 3 |
| warmreset_t0.2 - full | 50 | -0.02 | [-0.16, +0.12] | 6 / 8 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.94 | [-1.00, -0.86] | 0 / 47 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.16 | [-0.26, -0.06] | 0 / 8 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.82 | [-0.92, -0.71] | 0 / 42 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.72 | [-0.84, -0.60] | 0 / 36 |
| warmshoot_t0.2 - full | 50 | -0.84 | [-0.94, -0.74] | 0 / 42 |
| resetfinal_t0.2 - plain_k2 | 50 | -0.22 | [-0.36, -0.08] | 3 / 14 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.56 | [+0.42, +0.70] | 29 / 1 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.10 | [-0.22, +0.02] | 3 / 9 |
| resetfinal_t0.2 - full | 50 | -0.12 | [-0.28, +0.04] | 6 / 12 |
| midfinal_t0.2 - plain_k2 | 50 | -0.62 | [-0.76, -0.46] | 1 / 32 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.16 | [+0.04, +0.28] | 10 / 2 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.50 | [-0.64, -0.35] | 1 / 27 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.40 | [-0.56, -0.24] | 2 / 22 |
| midfinal_t0.2 - full | 50 | -0.52 | [-0.70, -0.32] | 5 / 31 |
| warm_t0.2 - plain_k2 | 100 | -0.78 | [-0.86, -0.70] | 0 / 78 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.66 | [-0.80, -0.50] | 2 / 36 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.56 | [-0.70, -0.42] | 1 / 29 |
| warm_t0.2 - full | 50 | -0.68 | [-0.82, -0.54] | 1 / 35 |

### PickPlaceDrawerToCounter

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.32 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.18 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.10 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.48 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.22 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.30 | [+0.16, +0.44] | 16 / 1 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.38 | [+0.22, +0.54] | 22 / 3 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.06 | [-0.10, +0.22] | 10 / 7 |
| warmreset_t0.2 - full | 50 | +0.16 | [+0.00, +0.32] | 14 / 6 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.18 | [-0.30, -0.08] | 0 / 9 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.10 | [-0.20, -0.02] | 0 / 5 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.48 | [-0.62, -0.34] | 0 / 24 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.42 | [-0.56, -0.28] | 0 / 21 |
| warmshoot_t0.2 - full | 50 | -0.32 | [-0.46, -0.20] | 0 / 16 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.24 | [+0.08, +0.40] | 15 / 3 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.32 | [+0.16, +0.48] | 19 / 3 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| resetfinal_t0.2 - full | 50 | +0.10 | [-0.06, +0.26] | 11 / 6 |
| midfinal_t0.2 - plain_k2 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.26 | [-0.46, -0.06] | 9 / 22 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.20 | [-0.36, -0.04] | 5 / 15 |
| midfinal_t0.2 - full | 50 | -0.10 | [-0.28, +0.08] | 8 / 13 |
| warm_t0.2 - plain_k2 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.38 | [-0.54, -0.22] | 3 / 22 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.32 | [-0.48, -0.16] | 3 / 19 |
| warm_t0.2 - full | 50 | -0.22 | [-0.36, -0.08] | 3 / 14 |

### PickPlaceSinkToCounter

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 1.00 | ok | 10.0 | 0.0 | – |
| plain_k2 | 100 | 1.00 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 100 | 0.37 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 1.00 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 1.00 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.94 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.58 | [+0.44, +0.72] | 29 / 0 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| warmreset_t0.2 - full | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| warmshoot_t0.2 - plain_k2 | 50 | -1.00 | [-1.00, -0.93] † | 0 / 50 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.42 | [-0.56, -0.28] | 0 / 21 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -1.00 | [-1.00, -0.93] † | 0 / 50 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -1.00 | [-1.00, -0.93] † | 0 / 50 |
| warmshoot_t0.2 - full | 50 | -1.00 | [-1.00, -0.93] † | 0 / 50 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.58 | [+0.44, +0.72] | 29 / 0 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| resetfinal_t0.2 - full | 50 | +0.00 | [-0.07, +0.07] † | 0 / 0 |
| midfinal_t0.2 - plain_k2 | 50 | -0.06 | [-0.14, +0.00] | 0 / 3 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.52 | [+0.38, +0.66] | 26 / 0 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.06 | [-0.14, +0.00] | 0 / 3 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.06 | [-0.14, +0.00] | 0 / 3 |
| midfinal_t0.2 - full | 50 | -0.06 | [-0.14, +0.00] | 0 / 3 |
| warm_t0.2 - plain_k2 | 100 | -0.63 | [-0.72, -0.53] | 0 / 63 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.58 | [-0.72, -0.44] | 0 / 29 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.58 | [-0.72, -0.44] | 0 / 29 |
| warm_t0.2 - full | 50 | -0.58 | [-0.72, -0.44] | 0 / 29 |

### PickPlaceToasterToCounter

cross-arm model/env identity: identical; worker islands 4, serving runtimes 4 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.40 | ok | 10.0 | 0.0 | – |
| plain_k2 | 50 | 0.28 | ok | 2.0 | 0.0 | – |
| warm_t0.2 | 50 | 0.08 | ok | 2.0 | 0.0 | – |
| warmreset_t0.2 | 50 | 0.64 | ok | 2.0 | 0.0 | – |
| warmshoot_t0.2 | 50 | 0.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.2 | 50 | 0.68 | ok | 2.0 | 0.0 | – |
| midfinal_t0.2 | 50 | 0.56 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 50 | +0.36 | [+0.18, +0.52] | 22 / 4 |
| warmreset_t0.2 - warm_t0.2 | 50 | +0.56 | [+0.40, +0.72] | 30 / 2 |
| warmreset_t0.2 - resetfinal_t0.2 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| warmreset_t0.2 - full | 50 | +0.24 | [+0.04, +0.44] | 20 / 8 |
| warmshoot_t0.2 - plain_k2 | 50 | -0.28 | [-0.40, -0.16] | 0 / 14 |
| warmshoot_t0.2 - warm_t0.2 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| warmshoot_t0.2 - warmreset_t0.2 | 50 | -0.64 | [-0.76, -0.50] | 0 / 32 |
| warmshoot_t0.2 - resetfinal_t0.2 | 50 | -0.68 | [-0.80, -0.54] | 0 / 34 |
| warmshoot_t0.2 - full | 50 | -0.40 | [-0.54, -0.26] | 0 / 20 |
| resetfinal_t0.2 - plain_k2 | 50 | +0.40 | [+0.24, +0.56] | 23 / 3 |
| resetfinal_t0.2 - warm_t0.2 | 50 | +0.60 | [+0.44, +0.74] | 32 / 2 |
| resetfinal_t0.2 - warmreset_t0.2 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| resetfinal_t0.2 - full | 50 | +0.28 | [+0.12, +0.44] | 18 / 4 |
| midfinal_t0.2 - plain_k2 | 50 | +0.28 | [+0.08, +0.46] | 21 / 7 |
| midfinal_t0.2 - warm_t0.2 | 50 | +0.48 | [+0.32, +0.62] | 25 / 1 |
| midfinal_t0.2 - warmreset_t0.2 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| midfinal_t0.2 - resetfinal_t0.2 | 50 | -0.12 | [-0.26, +0.02] | 4 / 10 |
| midfinal_t0.2 - full | 50 | +0.16 | [-0.04, +0.36] | 18 / 10 |
| warm_t0.2 - plain_k2 | 50 | -0.20 | [-0.36, -0.04] | 4 / 14 |
| warm_t0.2 - warmreset_t0.2 | 50 | -0.56 | [-0.72, -0.40] | 2 / 30 |
| warm_t0.2 - resetfinal_t0.2 | 50 | -0.60 | [-0.76, -0.44] | 2 / 32 |
| warm_t0.2 - full | 50 | -0.32 | [-0.48, -0.16] | 2 / 18 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 13 | 0.548 |
| plain_k2 | 13 | 0.481 |
| warm_t0.2 | 13 | 0.277 |
| warmreset_t0.2 | 13 | 0.708 |
| warmshoot_t0.2 | 13 | 0.005 |
| resetfinal_t0.2 | 13 | 0.708 |
| midfinal_t0.2 | 13 | 0.545 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.2 - plain_k2 | 13 | +0.228 | [+0.189, +0.268] |
| warmreset_t0.2 - warm_t0.2 | 13 | +0.430 | [+0.388, +0.471] |
| warmreset_t0.2 - resetfinal_t0.2 | 13 | +0.001 | [-0.032, +0.034] |
| warmreset_t0.2 - full | 13 | +0.161 | [+0.117, +0.204] |
| warmshoot_t0.2 - plain_k2 | 13 | -0.475 | [-0.506, -0.445] |
| warmshoot_t0.2 - warm_t0.2 | 13 | -0.274 | [-0.305, -0.243] |
| warmshoot_t0.2 - warmreset_t0.2 | 13 | -0.704 | [-0.735, -0.672] |
| warmshoot_t0.2 - resetfinal_t0.2 | 13 | -0.703 | [-0.735, -0.671] |
| warmshoot_t0.2 - full | 13 | -0.543 | [-0.577, -0.511] |
| resetfinal_t0.2 - plain_k2 | 13 | +0.228 | [+0.188, +0.269] |
| resetfinal_t0.2 - warm_t0.2 | 13 | +0.429 | [+0.388, +0.469] |
| resetfinal_t0.2 - warmreset_t0.2 | 13 | -0.001 | [-0.034, +0.032] |
| resetfinal_t0.2 - full | 13 | +0.160 | [+0.117, +0.203] |
| midfinal_t0.2 - plain_k2 | 13 | +0.065 | [+0.022, +0.109] |
| midfinal_t0.2 - warm_t0.2 | 13 | +0.266 | [+0.223, +0.309] |
| midfinal_t0.2 - warmreset_t0.2 | 13 | -0.164 | [-0.206, -0.122] |
| midfinal_t0.2 - resetfinal_t0.2 | 13 | -0.163 | [-0.205, -0.120] |
| midfinal_t0.2 - full | 13 | -0.003 | [-0.049, +0.045] |
| warm_t0.2 - plain_k2 | 13 | -0.204 | [-0.242, -0.166] |
| warm_t0.2 - warmreset_t0.2 | 13 | -0.430 | [-0.472, -0.388] |
| warm_t0.2 - resetfinal_t0.2 | 13 | -0.429 | [-0.469, -0.388] |
| warm_t0.2 - full | 13 | -0.269 | [-0.311, -0.228] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
