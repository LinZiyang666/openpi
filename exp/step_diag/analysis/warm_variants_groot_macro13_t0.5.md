## Warm-start continuation variants — groot (m=2, t=0.5; descriptive, 95% paired bootstrap)

missing arms: warmshoot_t0.5

### CloseBlenderLid

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.80 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.56 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.54 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.74 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.76 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.74 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.74 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.04 | [-0.18, +0.10] | 5 / 7 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | +0.02 | [-0.12, +0.18] | 8 / 7 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | -0.18 | [-0.32, -0.06] | 2 / 11 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.18 | [-0.32, -0.04] | 2 / 11 |
| warmreset_t0.5 - full | 50 | -0.24 | [-0.38, -0.10] | 2 / 14 |
| resetfinal_t0.5 - plain_k2 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.18, +0.12] | 7 / 8 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | -0.20 | [-0.34, -0.06] | 3 / 13 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | -0.20 | [-0.34, -0.06] | 2 / 12 |
| resetfinal_t0.5 - full | 50 | -0.26 | [-0.42, -0.10] | 4 / 17 |
| midfinal_t0.5 - plain_k2 | 50 | +0.14 | [-0.02, +0.30] | 12 / 5 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | +0.18 | [+0.06, +0.32] | 11 / 2 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | +0.20 | [+0.06, +0.34] | 13 / 3 |
| midfinal_t0.5 - midreset_t0.5 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midfinal_t0.5 - full | 50 | -0.06 | [-0.22, +0.12] | 8 / 11 |
| midfinal50_t0.5 - plain_k2 | 50 | +0.16 | [-0.02, +0.32] | 14 / 6 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | +0.20 | [+0.06, +0.34] | 13 / 3 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | +0.22 | [+0.08, +0.36] | 13 / 2 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| midfinal50_t0.5 - full | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midreset_t0.5 - plain_k2 | 50 | +0.14 | [+0.00, +0.28] | 10 / 3 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.18 | [+0.06, +0.32] | 11 / 2 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | +0.20 | [+0.06, +0.34] | 12 / 2 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midreset_t0.5 - full | 50 | -0.06 | [-0.20, +0.08] | 6 / 9 |
| midreset50_t0.5 - plain_k2 | 50 | +0.14 | [-0.02, +0.30] | 12 / 5 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | +0.18 | [+0.06, +0.30] | 10 / 1 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | +0.20 | [+0.02, +0.38] | 16 / 6 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | +0.00 | [-0.16, +0.16] | 8 / 8 |
| midreset50_t0.5 - midreset_t0.5 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| midreset50_t0.5 - full | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |

### CloseFridge

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.34 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.48 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.54 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.70 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.86 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.68 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.64 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.70 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | +0.06 | [-0.12, +0.24] | 12 / 9 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.16 | [-0.30, -0.02] | 4 / 12 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | -0.32 | [-0.48, -0.16] | 3 / 19 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.10 | [-0.28, +0.08] | 8 / 13 |
| warmreset_t0.5 - full | 50 | +0.20 | [+0.00, +0.40] | 18 / 8 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.22 | [+0.06, +0.38] | 16 / 5 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.16 | [+0.02, +0.30] | 12 / 4 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | -0.16 | [-0.30, -0.02] | 3 / 11 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | +0.06 | [-0.12, +0.24] | 12 / 9 |
| resetfinal_t0.5 - full | 50 | +0.36 | [+0.18, +0.54] | 22 / 4 |
| midfinal_t0.5 - plain_k2 | 50 | +0.38 | [+0.20, +0.56] | 23 / 4 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | +0.32 | [+0.16, +0.48] | 19 / 3 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | +0.16 | [+0.02, +0.30] | 11 / 3 |
| midfinal_t0.5 - midreset_t0.5 | 50 | +0.22 | [+0.08, +0.36] | 14 / 3 |
| midfinal_t0.5 - full | 50 | +0.52 | [+0.36, +0.68] | 28 / 2 |
| midfinal50_t0.5 - plain_k2 | 50 | +0.20 | [+0.00, +0.40] | 18 / 8 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | +0.14 | [-0.02, +0.30] | 13 / 6 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.02 | [-0.20, +0.14] | 9 / 10 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | -0.18 | [-0.34, -0.02] | 4 / 13 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| midfinal50_t0.5 - full | 50 | +0.34 | [+0.16, +0.52] | 22 / 5 |
| midreset_t0.5 - plain_k2 | 50 | +0.16 | [-0.02, +0.34] | 15 / 7 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.10 | [-0.08, +0.28] | 13 / 8 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.24, +0.12] | 9 / 12 |
| midreset_t0.5 - midfinal_t0.5 | 50 | -0.22 | [-0.38, -0.08] | 3 / 14 |
| midreset_t0.5 - full | 50 | +0.30 | [+0.12, +0.48] | 20 / 5 |
| midreset50_t0.5 - plain_k2 | 50 | +0.22 | [+0.04, +0.38] | 16 / 5 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | +0.16 | [+0.00, +0.32] | 14 / 6 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | +0.00 | [-0.18, +0.18] | 11 / 11 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | -0.16 | [-0.30, -0.02] | 3 / 11 |
| midreset50_t0.5 - midreset_t0.5 | 50 | +0.06 | [-0.08, +0.20] | 8 / 5 |
| midreset50_t0.5 - full | 50 | +0.36 | [+0.18, +0.54] | 22 / 4 |

### CoffeeSetupMug

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.58 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.46 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.52 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.54 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.40 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.48 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.48 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.48 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | +0.06 | [-0.10, +0.22] | 9 / 6 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.02 | [-0.14, +0.10] | 5 / 6 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | +0.12 | [-0.04, +0.28] | 12 / 6 |
| warmreset_t0.5 - midreset_t0.5 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| warmreset_t0.5 - full | 50 | -0.06 | [-0.22, +0.10] | 8 / 11 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.02 | [-0.10, +0.16] | 6 / 5 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.14 | [+0.02, +0.26] | 9 / 2 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | +0.06 | [-0.06, +0.20] | 7 / 4 |
| resetfinal_t0.5 - full | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midfinal_t0.5 - plain_k2 | 50 | -0.06 | [-0.22, +0.12] | 8 / 11 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | -0.12 | [-0.28, +0.04] | 6 / 12 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | -0.14 | [-0.26, -0.02] | 2 / 9 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| midfinal_t0.5 - full | 50 | -0.18 | [-0.34, -0.02] | 5 / 14 |
| midfinal50_t0.5 - plain_k2 | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.22, +0.14] | 9 / 11 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | +0.00 | [-0.14, +0.14] | 7 / 7 |
| midfinal50_t0.5 - full | 50 | -0.10 | [-0.26, +0.06] | 6 / 11 |
| midreset_t0.5 - plain_k2 | 50 | +0.02 | [-0.16, +0.18] | 10 / 9 |
| midreset_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.18, +0.06] | 4 / 7 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 10 / 6 |
| midreset_t0.5 - full | 50 | -0.10 | [-0.26, +0.06] | 7 / 12 |
| midreset50_t0.5 - plain_k2 | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | +0.08 | [-0.06, +0.22] | 8 / 4 |
| midreset50_t0.5 - midreset_t0.5 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| midreset50_t0.5 - full | 50 | -0.10 | [-0.26, +0.06] | 6 / 11 |

### OpenCabinet

cross-arm model/env identity: identical; worker islands 7, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.90 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.90 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.88 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.92 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.86 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.94 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.84 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.06 | [-0.06, +0.18] | 7 / 4 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.04 | [-0.12, +0.04] | 1 / 3 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | +0.06 | [-0.06, +0.20] | 7 / 4 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.06 | [-0.14, +0.00] | 0 / 3 |
| warmreset_t0.5 - full | 50 | -0.02 | [-0.08, +0.04] | 1 / 2 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.10 | [-0.02, +0.22] | 7 / 2 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.10 | [-0.02, +0.24] | 8 / 3 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| resetfinal_t0.5 - full | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| midfinal_t0.5 - plain_k2 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| midfinal_t0.5 - warm_t0.5 | 50 | +0.00 | [-0.14, +0.14] | 7 / 7 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.18, +0.06] | 4 / 7 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | -0.10 | [-0.22, +0.02] | 3 / 8 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.12 | [-0.24, +0.00] | 2 / 8 |
| midfinal_t0.5 - full | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.04 | [-0.18, +0.10] | 5 / 7 |
| midfinal50_t0.5 - warm_t0.5 | 50 | +0.04 | [-0.10, +0.18] | 7 / 5 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.16, +0.10] | 5 / 6 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.16, +0.04] | 2 / 5 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | +0.04 | [-0.06, +0.16] | 5 / 3 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.20, +0.04] | 3 / 7 |
| midfinal50_t0.5 - full | 50 | -0.04 | [-0.18, +0.10] | 5 / 7 |
| midreset_t0.5 - plain_k2 | 50 | +0.04 | [+0.00, +0.10] | 2 / 0 |
| midreset_t0.5 - warm_t0.5 | 50 | +0.12 | [+0.02, +0.24] | 7 / 1 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.06 | [+0.00, +0.14] | 3 / 0 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.12 | [+0.00, +0.24] | 8 / 2 |
| midreset_t0.5 - full | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| midreset50_t0.5 - plain_k2 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| midreset50_t0.5 - warm_t0.5 | 50 | +0.02 | [-0.04, +0.08] | 2 / 1 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.08 | [-0.18, +0.00] | 1 / 5 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | +0.02 | [-0.10, +0.14] | 6 / 5 |
| midreset50_t0.5 - midreset_t0.5 | 50 | -0.10 | [-0.20, +0.00] | 1 / 6 |
| midreset50_t0.5 - full | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| warm_t0.5 - plain_k2 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.18, +0.06] | 4 / 7 |
| warm_t0.5 - resetfinal_t0.5 | 50 | -0.10 | [-0.22, +0.02] | 2 / 7 |
| warm_t0.5 - midfinal_t0.5 | 50 | +0.00 | [-0.14, +0.14] | 7 / 7 |
| warm_t0.5 - midreset_t0.5 | 50 | -0.12 | [-0.24, -0.02] | 1 / 7 |
| warm_t0.5 - full | 50 | -0.08 | [-0.20, +0.04] | 3 / 7 |

### OpenDrawer

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.72 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.70 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.72 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.56 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.64 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.72 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.72 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | +0.02 | [-0.10, +0.14] | 6 / 5 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | +0.16 | [+0.02, +0.30] | 11 / 3 |
| warmreset_t0.5 - midreset_t0.5 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| warmreset_t0.5 - full | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| resetfinal_t0.5 - plain_k2 | 50 | -0.10 | [-0.26, +0.08] | 7 / 12 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.12 | [-0.26, +0.02] | 4 / 10 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.04 | [-0.10, +0.18] | 7 / 5 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | -0.12 | [-0.26, +0.00] | 3 / 9 |
| resetfinal_t0.5 - full | 50 | -0.12 | [-0.28, +0.04] | 5 / 11 |
| midfinal_t0.5 - plain_k2 | 50 | -0.14 | [-0.30, +0.02] | 5 / 12 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | -0.16 | [-0.30, -0.02] | 3 / 11 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | -0.04 | [-0.18, +0.10] | 5 / 7 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.16 | [-0.28, -0.04] | 2 / 10 |
| midfinal_t0.5 - full | 50 | -0.16 | [-0.32, +0.00] | 5 / 13 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| midfinal50_t0.5 - full | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| midreset_t0.5 - plain_k2 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | +0.12 | [+0.00, +0.26] | 9 / 3 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.16 | [+0.04, +0.28] | 10 / 2 |
| midreset_t0.5 - full | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midreset50_t0.5 - plain_k2 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | +0.00 | [-0.10, +0.12] | 4 / 4 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | +0.12 | [-0.04, +0.28] | 11 / 5 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | +0.16 | [+0.02, +0.32] | 12 / 4 |
| midreset50_t0.5 - midreset_t0.5 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midreset50_t0.5 - full | 50 | +0.00 | [-0.10, +0.12] | 4 / 4 |

### OpenStandMixerHead

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.72 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.70 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.68 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.76 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.74 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.66 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.70 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.70 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.08 | [-0.18, +0.02] | 2 / 6 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| warmreset_t0.5 - full | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.06 | [-0.04, +0.16] | 5 / 2 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.02 | [-0.04, +0.10] | 2 / 1 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| resetfinal_t0.5 - full | 50 | +0.04 | [-0.06, +0.16] | 5 / 3 |
| midfinal_t0.5 - plain_k2 | 50 | +0.04 | [-0.06, +0.14] | 4 / 2 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| midfinal_t0.5 - midreset_t0.5 | 50 | +0.04 | [-0.06, +0.16] | 5 / 3 |
| midfinal_t0.5 - full | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.10 | [-0.20, +0.00] | 1 / 6 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | -0.08 | [-0.18, +0.00] | 1 / 5 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| midfinal50_t0.5 - full | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| midreset_t0.5 - plain_k2 | 50 | +0.00 | [-0.12, +0.10] | 4 / 4 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| midreset_t0.5 - midfinal_t0.5 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midreset_t0.5 - full | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| midreset50_t0.5 - plain_k2 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.16, +0.04] | 2 / 5 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| midreset50_t0.5 - midreset_t0.5 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midreset50_t0.5 - full | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |

### SlideDishwasherRack

cross-arm model/env identity: identical; worker islands 7, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.54 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.46 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.44 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.44 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.50 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.52 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.34 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.44 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.16, +0.04] | 2 / 5 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| warmreset_t0.5 - midreset_t0.5 | 50 | +0.02 | [-0.04, +0.08] | 2 / 1 |
| warmreset_t0.5 - full | 50 | -0.10 | [-0.22, +0.02] | 3 / 8 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.04 | [-0.06, +0.14] | 4 / 2 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.06 | [-0.02, +0.16] | 4 / 1 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.06 | [-0.04, +0.16] | 5 / 2 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| resetfinal_t0.5 - full | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midfinal_t0.5 - plain_k2 | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| midfinal_t0.5 - warm_t0.5 | 50 | +0.08 | [-0.02, +0.18] | 6 / 2 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midfinal_t0.5 - midreset_t0.5 | 50 | +0.10 | [+0.00, +0.20] | 6 / 1 |
| midfinal_t0.5 - full | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.12 | [-0.24, +0.00] | 2 / 8 |
| midfinal50_t0.5 - warm_t0.5 | 50 | -0.10 | [-0.22, +0.02] | 2 / 7 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.10 | [-0.20, +0.00] | 1 / 6 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.16 | [-0.28, -0.06] | 1 / 9 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | -0.18 | [-0.30, -0.08] | 0 / 9 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.18, +0.00] | 1 / 5 |
| midfinal50_t0.5 - full | 50 | -0.20 | [-0.32, -0.10] | 0 / 10 |
| midreset_t0.5 - plain_k2 | 50 | -0.04 | [-0.14, +0.06] | 2 / 4 |
| midreset_t0.5 - warm_t0.5 | 50 | -0.02 | [-0.08, +0.04] | 1 / 2 |
| midreset_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | -0.08 | [-0.18, +0.00] | 1 / 5 |
| midreset_t0.5 - midfinal_t0.5 | 50 | -0.10 | [-0.20, +0.00] | 1 / 6 |
| midreset_t0.5 - full | 50 | -0.12 | [-0.22, -0.02] | 1 / 7 |
| midreset50_t0.5 - plain_k2 | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| midreset50_t0.5 - warm_t0.5 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.16, +0.02] | 1 / 4 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | -0.08 | [-0.18, +0.00] | 1 / 5 |
| midreset50_t0.5 - midreset_t0.5 | 50 | +0.02 | [-0.04, +0.10] | 2 / 1 |
| midreset50_t0.5 - full | 50 | -0.10 | [-0.20, +0.00] | 1 / 6 |
| warm_t0.5 - plain_k2 | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| warm_t0.5 - warmreset_t0.5 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| warm_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.14, +0.02] | 1 / 4 |
| warm_t0.5 - midfinal_t0.5 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| warm_t0.5 - midreset_t0.5 | 50 | +0.02 | [-0.04, +0.08] | 2 / 1 |
| warm_t0.5 - full | 50 | -0.10 | [-0.22, +0.02] | 2 / 7 |

### TurnOnSinkFaucet

cross-arm model/env identity: identical; worker islands 7, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.18 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.36 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.30 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.38 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.50 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.34 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.52 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.30 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | +0.06 | [-0.12, +0.22] | 11 / 8 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.12 | [-0.08, +0.30] | 15 / 9 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | +0.04 | [-0.14, +0.22] | 12 / 10 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.10 | [-0.28, +0.08] | 9 / 14 |
| warmreset_t0.5 - full | 50 | +0.24 | [+0.08, +0.40] | 16 / 4 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 10 / 6 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.22, +0.14] | 10 / 12 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | -0.12 | [-0.30, +0.06] | 8 / 14 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | -0.14 | [-0.32, +0.04] | 7 / 14 |
| resetfinal_t0.5 - full | 50 | +0.20 | [+0.06, +0.34] | 13 / 3 |
| midfinal_t0.5 - plain_k2 | 50 | +0.14 | [-0.04, +0.32] | 15 / 8 |
| midfinal_t0.5 - warm_t0.5 | 50 | +0.20 | [+0.00, +0.40] | 19 / 9 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | +0.12 | [-0.06, +0.30] | 14 / 8 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.02 | [-0.20, +0.16] | 10 / 11 |
| midfinal_t0.5 - full | 50 | +0.32 | [+0.14, +0.50] | 21 / 5 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.02 | [-0.18, +0.14] | 8 / 9 |
| midfinal50_t0.5 - warm_t0.5 | 50 | +0.04 | [-0.14, +0.22] | 11 / 9 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | -0.16 | [-0.32, +0.00] | 5 / 13 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.18 | [-0.34, -0.02] | 4 / 13 |
| midfinal50_t0.5 - full | 50 | +0.16 | [+0.04, +0.28] | 10 / 2 |
| midreset_t0.5 - plain_k2 | 50 | +0.16 | [+0.00, +0.32] | 13 / 5 |
| midreset_t0.5 - warm_t0.5 | 50 | +0.22 | [+0.06, +0.38] | 16 / 5 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.10 | [-0.08, +0.28] | 14 / 9 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | +0.14 | [-0.04, +0.32] | 14 / 7 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.02 | [-0.16, +0.20] | 11 / 10 |
| midreset_t0.5 - full | 50 | +0.34 | [+0.18, +0.50] | 19 / 2 |
| midreset50_t0.5 - plain_k2 | 50 | -0.06 | [-0.22, +0.08] | 6 / 9 |
| midreset50_t0.5 - warm_t0.5 | 50 | +0.00 | [-0.16, +0.16] | 9 / 9 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | -0.12 | [-0.30, +0.08] | 9 / 15 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | -0.20 | [-0.38, -0.02] | 6 / 16 |
| midreset50_t0.5 - midreset_t0.5 | 50 | -0.22 | [-0.38, -0.04] | 5 / 16 |
| midreset50_t0.5 - full | 50 | +0.12 | [-0.04, +0.28] | 11 / 5 |
| warm_t0.5 - plain_k2 | 50 | -0.06 | [-0.24, +0.12] | 9 / 12 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.12 | [-0.30, +0.08] | 9 / 15 |
| warm_t0.5 - resetfinal_t0.5 | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| warm_t0.5 - midfinal_t0.5 | 50 | -0.20 | [-0.40, +0.00] | 9 / 19 |
| warm_t0.5 - midreset_t0.5 | 50 | -0.22 | [-0.38, -0.06] | 5 / 16 |
| warm_t0.5 - full | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |

### PickPlaceCounterToCabinet

cross-arm model/env identity: identical; worker islands 6, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.40 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.44 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.48 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.56 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.50 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | +0.04 | [-0.10, +0.18] | 7 / 5 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.12 | [-0.26, +0.02] | 4 / 10 |
| warmreset_t0.5 - full | 50 | -0.18 | [-0.34, -0.02] | 4 / 13 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.08 | [-0.06, +0.22] | 8 / 4 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.06 | [-0.08, +0.20] | 8 / 5 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.18, +0.02] | 2 / 6 |
| resetfinal_t0.5 - full | 50 | -0.14 | [-0.30, +0.02] | 6 / 13 |
| midfinal_t0.5 - plain_k2 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.16, +0.12] | 6 / 7 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.14 | [-0.26, -0.04] | 1 / 8 |
| midfinal_t0.5 - full | 50 | -0.20 | [-0.34, -0.06] | 3 / 13 |
| midfinal50_t0.5 - plain_k2 | 50 | +0.02 | [-0.12, +0.16] | 8 / 7 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.18, +0.14] | 8 / 9 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | +0.00 | [-0.16, +0.16] | 9 / 9 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.14 | [-0.30, +0.02] | 6 / 13 |
| midfinal50_t0.5 - full | 50 | -0.20 | [-0.38, -0.02] | 7 / 17 |
| midreset_t0.5 - plain_k2 | 50 | +0.16 | [+0.02, +0.30] | 11 / 3 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.14 | [+0.04, +0.26] | 8 / 1 |
| midreset_t0.5 - full | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| midreset50_t0.5 - plain_k2 | 50 | +0.10 | [-0.04, +0.24] | 10 / 5 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | +0.06 | [-0.08, +0.20] | 8 / 5 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | +0.02 | [-0.16, +0.20] | 10 / 9 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 10 / 6 |
| midreset50_t0.5 - midreset_t0.5 | 50 | -0.06 | [-0.24, +0.12] | 8 / 11 |
| midreset50_t0.5 - full | 50 | -0.12 | [-0.28, +0.04] | 6 / 12 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.82 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.96 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.82 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.88 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.86 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.86 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.54 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.90 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.82 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| warmreset_t0.5 - full | 50 | +0.06 | [-0.06, +0.20] | 7 / 4 |
| resetfinal_t0.5 - plain_k2 | 50 | -0.10 | [-0.20, -0.02] | 0 / 5 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.00 | [-0.12, +0.12] | 4 / 4 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | -0.04 | [-0.10, +0.00] | 0 / 2 |
| resetfinal_t0.5 - full | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| midfinal_t0.5 - plain_k2 | 50 | -0.10 | [-0.20, +0.00] | 1 / 6 |
| midfinal_t0.5 - warm_t0.5 | 50 | +0.04 | [-0.06, +0.14] | 5 / 3 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | +0.00 | [-0.12, +0.10] | 4 / 4 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.04 | [-0.14, +0.06] | 2 / 4 |
| midfinal_t0.5 - full | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.42 | [-0.58, -0.26] | 2 / 23 |
| midfinal50_t0.5 - warm_t0.5 | 50 | -0.28 | [-0.44, -0.12] | 4 / 18 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.34 | [-0.50, -0.16] | 4 / 21 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.32 | [-0.48, -0.16] | 4 / 20 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | -0.32 | [-0.48, -0.16] | 3 / 19 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.36 | [-0.52, -0.20] | 3 / 21 |
| midfinal50_t0.5 - full | 50 | -0.28 | [-0.46, -0.10] | 6 / 20 |
| midreset_t0.5 - plain_k2 | 50 | -0.06 | [-0.14, +0.02] | 1 / 4 |
| midreset_t0.5 - warm_t0.5 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | +0.04 | [+0.00, +0.10] | 2 / 0 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.04 | [-0.06, +0.14] | 4 / 2 |
| midreset_t0.5 - full | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| midreset50_t0.5 - plain_k2 | 50 | -0.14 | [-0.26, -0.04] | 1 / 8 |
| midreset50_t0.5 - warm_t0.5 | 50 | +0.00 | [-0.06, +0.06] | 1 / 1 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.20, +0.06] | 4 / 7 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midreset50_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| midreset50_t0.5 - full | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| warm_t0.5 - plain_k2 | 50 | -0.14 | [-0.26, -0.04] | 1 / 8 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| warm_t0.5 - resetfinal_t0.5 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| warm_t0.5 - midfinal_t0.5 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| warm_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| warm_t0.5 - full | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |

### PickPlaceDrawerToCounter

cross-arm model/env identity: identical; worker islands 5, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.66 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 50 | 0.50 | ok | 2.0 | 0.0 | – |
| warmreset_t0.5 | 50 | 0.54 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.58 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.58 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.66 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.56 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.52 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| warmreset_t0.5 - warm_t0.5 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | -0.04 | [-0.18, +0.10] | 6 / 8 |
| warmreset_t0.5 - midreset_t0.5 | 50 | -0.02 | [-0.16, +0.12] | 6 / 7 |
| warmreset_t0.5 - full | 50 | -0.12 | [-0.26, +0.02] | 4 / 10 |
| resetfinal_t0.5 - plain_k2 | 50 | -0.02 | [-0.18, +0.14] | 7 / 8 |
| resetfinal_t0.5 - warm_t0.5 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.04 | [-0.08, +0.16] | 6 / 4 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | +0.02 | [-0.12, +0.14] | 6 / 5 |
| resetfinal_t0.5 - full | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| midfinal_t0.5 - plain_k2 | 50 | -0.02 | [-0.20, +0.14] | 9 / 10 |
| midfinal_t0.5 - warm_t0.5 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | +0.04 | [-0.10, +0.18] | 8 / 6 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| midfinal_t0.5 - midreset_t0.5 | 50 | +0.02 | [-0.12, +0.14] | 6 / 5 |
| midfinal_t0.5 - full | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| midfinal50_t0.5 - plain_k2 | 50 | +0.06 | [-0.10, +0.24] | 11 / 8 |
| midfinal50_t0.5 - warm_t0.5 | 50 | +0.16 | [+0.00, +0.32] | 12 / 4 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | +0.12 | [-0.06, +0.30] | 14 / 8 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | +0.10 | [-0.04, +0.26] | 10 / 5 |
| midfinal50_t0.5 - full | 50 | +0.00 | [-0.16, +0.16] | 8 / 8 |
| midreset_t0.5 - plain_k2 | 50 | -0.04 | [-0.18, +0.10] | 6 / 8 |
| midreset_t0.5 - warm_t0.5 | 50 | +0.06 | [-0.08, +0.20] | 9 / 6 |
| midreset_t0.5 - warmreset_t0.5 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | -0.02 | [-0.16, +0.10] | 5 / 6 |
| midreset_t0.5 - midfinal_t0.5 | 50 | -0.02 | [-0.16, +0.12] | 5 / 6 |
| midreset_t0.5 - full | 50 | -0.10 | [-0.26, +0.06] | 6 / 11 |
| midreset50_t0.5 - plain_k2 | 50 | -0.08 | [-0.26, +0.10] | 9 / 13 |
| midreset50_t0.5 - warm_t0.5 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.20, +0.16] | 9 / 10 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.24, +0.12] | 9 / 12 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | -0.06 | [-0.22, +0.10] | 8 / 11 |
| midreset50_t0.5 - midreset_t0.5 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| midreset50_t0.5 - full | 50 | -0.14 | [-0.32, +0.04] | 8 / 15 |
| warm_t0.5 - plain_k2 | 50 | -0.10 | [-0.28, +0.08] | 8 / 13 |
| warm_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| warm_t0.5 - resetfinal_t0.5 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| warm_t0.5 - midfinal_t0.5 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| warm_t0.5 - midreset_t0.5 | 50 | -0.06 | [-0.20, +0.10] | 6 / 9 |
| warm_t0.5 - full | 50 | -0.16 | [-0.34, +0.02] | 7 / 15 |

### PickPlaceSinkToCounter

cross-arm model/env identity: identical; worker islands 6, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.78 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.92 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 1.00 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.92 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.94 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.90 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.98 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.90 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | +0.08 | [+0.02, +0.16] | 4 / 0 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | +0.08 | [+0.02, +0.16] | 4 / 0 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | +0.06 | [+0.00, +0.14] | 3 / 0 |
| warmreset_t0.5 - midreset_t0.5 | 50 | +0.02 | [+0.00, +0.06] | 1 / 0 |
| warmreset_t0.5 - full | 50 | +0.22 | [+0.12, +0.34] | 11 / 0 |
| resetfinal_t0.5 - plain_k2 | 50 | +0.00 | [-0.12, +0.10] | 4 / 4 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | -0.06 | [-0.16, +0.02] | 1 / 4 |
| resetfinal_t0.5 - full | 50 | +0.14 | [+0.04, +0.26] | 8 / 1 |
| midfinal_t0.5 - plain_k2 | 50 | +0.02 | [-0.04, +0.08] | 2 / 1 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.14, +0.00] | 0 / 3 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.04 | [-0.12, +0.04] | 1 / 3 |
| midfinal_t0.5 - full | 50 | +0.16 | [+0.04, +0.30] | 10 / 2 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.10 | [-0.18, -0.02] | 0 / 5 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | -0.04 | [-0.16, +0.08] | 3 / 5 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| midfinal50_t0.5 - full | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |
| midreset_t0.5 - plain_k2 | 50 | +0.06 | [-0.02, +0.14] | 4 / 1 |
| midreset_t0.5 - warmreset_t0.5 | 50 | -0.02 | [-0.06, +0.00] | 0 / 1 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | +0.06 | [-0.02, +0.14] | 4 / 1 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| midreset_t0.5 - full | 50 | +0.20 | [+0.10, +0.32] | 10 / 0 |
| midreset50_t0.5 - plain_k2 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | -0.10 | [-0.20, -0.02] | 0 / 5 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | -0.04 | [-0.14, +0.06] | 3 / 5 |
| midreset50_t0.5 - midreset_t0.5 | 50 | -0.08 | [-0.18, +0.00] | 1 / 5 |
| midreset50_t0.5 - full | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |

### PickPlaceToasterToCounter

cross-arm model/env identity: identical; worker islands 6, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.64 | ok | 4.0 | 0.0 | – |
| plain_k2 | 50 | 0.64 | ok | 2.0 | 0.0 | – |
| warm_t0.5 | 0 | – | NO | – | None | – |
| warmreset_t0.5 | 50 | 0.60 | ok | 2.0 | 0.0 | – |
| resetfinal_t0.5 | 50 | 0.62 | ok | 2.0 | 0.0 | – |
| midfinal_t0.5 | 50 | 0.42 | ok | 2.0 | 0.0 | – |
| midfinal50_t0.5 | 50 | 0.52 | ok | 2.0 | 0.0 | – |
| midreset_t0.5 | 50 | 0.54 | ok | 2.0 | 0.0 | – |
| midreset50_t0.5 | 50 | 0.56 | ok | 2.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 50 | -0.04 | [-0.18, +0.10] | 6 / 8 |
| warmreset_t0.5 - resetfinal_t0.5 | 50 | -0.02 | [-0.18, +0.14] | 8 / 9 |
| warmreset_t0.5 - midfinal_t0.5 | 50 | +0.18 | [+0.00, +0.36] | 15 / 6 |
| warmreset_t0.5 - midreset_t0.5 | 50 | +0.06 | [-0.10, +0.22] | 9 / 6 |
| warmreset_t0.5 - full | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| resetfinal_t0.5 - plain_k2 | 50 | -0.02 | [-0.18, +0.14] | 8 / 9 |
| resetfinal_t0.5 - warmreset_t0.5 | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| resetfinal_t0.5 - midfinal_t0.5 | 50 | +0.20 | [+0.04, +0.36] | 15 / 5 |
| resetfinal_t0.5 - midreset_t0.5 | 50 | +0.08 | [-0.08, +0.24] | 10 / 6 |
| resetfinal_t0.5 - full | 50 | -0.02 | [-0.18, +0.16] | 9 / 10 |
| midfinal_t0.5 - plain_k2 | 50 | -0.22 | [-0.40, -0.04] | 7 / 18 |
| midfinal_t0.5 - warmreset_t0.5 | 50 | -0.18 | [-0.36, +0.00] | 6 / 15 |
| midfinal_t0.5 - resetfinal_t0.5 | 50 | -0.20 | [-0.36, -0.04] | 5 / 15 |
| midfinal_t0.5 - midreset_t0.5 | 50 | -0.12 | [-0.30, +0.06] | 7 / 13 |
| midfinal_t0.5 - full | 50 | -0.22 | [-0.40, -0.04] | 7 / 18 |
| midfinal50_t0.5 - plain_k2 | 50 | -0.12 | [-0.30, +0.08] | 9 / 15 |
| midfinal50_t0.5 - warmreset_t0.5 | 50 | -0.08 | [-0.28, +0.12] | 11 / 15 |
| midfinal50_t0.5 - resetfinal_t0.5 | 50 | -0.10 | [-0.28, +0.08] | 8 / 13 |
| midfinal50_t0.5 - midfinal_t0.5 | 50 | +0.10 | [-0.08, +0.28] | 13 / 8 |
| midfinal50_t0.5 - midreset_t0.5 | 50 | -0.02 | [-0.20, +0.16] | 11 / 12 |
| midfinal50_t0.5 - full | 50 | -0.12 | [-0.30, +0.06] | 9 / 15 |
| midreset_t0.5 - plain_k2 | 50 | -0.10 | [-0.26, +0.06] | 7 / 12 |
| midreset_t0.5 - warmreset_t0.5 | 50 | -0.06 | [-0.20, +0.10] | 6 / 9 |
| midreset_t0.5 - resetfinal_t0.5 | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| midreset_t0.5 - midfinal_t0.5 | 50 | +0.12 | [-0.06, +0.30] | 13 / 7 |
| midreset_t0.5 - full | 50 | -0.10 | [-0.26, +0.06] | 7 / 12 |
| midreset50_t0.5 - plain_k2 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| midreset50_t0.5 - warmreset_t0.5 | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midreset50_t0.5 - resetfinal_t0.5 | 50 | -0.06 | [-0.22, +0.10] | 8 / 11 |
| midreset50_t0.5 - midfinal_t0.5 | 50 | +0.14 | [-0.02, +0.30] | 13 / 6 |
| midreset50_t0.5 - midreset_t0.5 | 50 | +0.02 | [-0.16, +0.18] | 10 / 9 |
| midreset50_t0.5 - full | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 13 | 0.638 |
| plain_k2 | 13 | 0.629 |
| warm_t0.5 | 5 | 0.576 |
| warmreset_t0.5 | 13 | 0.632 |
| warmshoot_t0.5 | 0 | – |
| resetfinal_t0.5 | 13 | 0.646 |
| midfinal_t0.5 | 13 | 0.643 |
| midfinal50_t0.5 | 13 | 0.600 |
| midreset_t0.5 | 13 | 0.669 |
| midreset50_t0.5 | 13 | 0.632 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.5 - plain_k2 | 13 | +0.003 | [-0.032, +0.040] |
| warmreset_t0.5 - resetfinal_t0.5 | 13 | -0.014 | [-0.049, +0.022] |
| warmreset_t0.5 - midfinal_t0.5 | 13 | -0.011 | [-0.049, +0.028] |
| warmreset_t0.5 - midreset_t0.5 | 13 | -0.037 | [-0.072, -0.002] |
| warmreset_t0.5 - full | 13 | -0.006 | [-0.046, +0.034] |
| resetfinal_t0.5 - plain_k2 | 13 | +0.017 | [-0.022, +0.055] |
| resetfinal_t0.5 - warmreset_t0.5 | 13 | +0.014 | [-0.022, +0.049] |
| resetfinal_t0.5 - midfinal_t0.5 | 13 | +0.003 | [-0.034, +0.040] |
| resetfinal_t0.5 - midreset_t0.5 | 13 | -0.023 | [-0.058, +0.011] |
| resetfinal_t0.5 - full | 13 | +0.008 | [-0.032, +0.048] |
| midfinal_t0.5 - plain_k2 | 13 | +0.014 | [-0.026, +0.055] |
| midfinal_t0.5 - warmreset_t0.5 | 13 | +0.011 | [-0.029, +0.049] |
| midfinal_t0.5 - resetfinal_t0.5 | 13 | -0.003 | [-0.040, +0.034] |
| midfinal_t0.5 - midreset_t0.5 | 13 | -0.026 | [-0.062, +0.009] |
| midfinal_t0.5 - full | 13 | +0.005 | [-0.037, +0.046] |
| midfinal50_t0.5 - plain_k2 | 13 | -0.029 | [-0.072, +0.014] |
| midfinal50_t0.5 - warmreset_t0.5 | 13 | -0.032 | [-0.074, +0.009] |
| midfinal50_t0.5 - resetfinal_t0.5 | 13 | -0.046 | [-0.086, -0.005] |
| midfinal50_t0.5 - midfinal_t0.5 | 13 | -0.043 | [-0.083, -0.003] |
| midfinal50_t0.5 - midreset_t0.5 | 13 | -0.069 | [-0.108, -0.031] |
| midfinal50_t0.5 - full | 13 | -0.038 | [-0.082, +0.005] |
| midreset_t0.5 - plain_k2 | 13 | +0.040 | [+0.003, +0.077] |
| midreset_t0.5 - warmreset_t0.5 | 13 | +0.037 | [+0.002, +0.072] |
| midreset_t0.5 - resetfinal_t0.5 | 13 | +0.023 | [-0.012, +0.058] |
| midreset_t0.5 - midfinal_t0.5 | 13 | +0.026 | [-0.009, +0.062] |
| midreset_t0.5 - full | 13 | +0.031 | [-0.008, +0.069] |
| midreset50_t0.5 - plain_k2 | 13 | +0.003 | [-0.037, +0.043] |
| midreset50_t0.5 - warmreset_t0.5 | 13 | -0.000 | [-0.038, +0.038] |
| midreset50_t0.5 - resetfinal_t0.5 | 13 | -0.014 | [-0.054, +0.026] |
| midreset50_t0.5 - midfinal_t0.5 | 13 | -0.011 | [-0.051, +0.028] |
| midreset50_t0.5 - midreset_t0.5 | 13 | -0.037 | [-0.074, +0.000] |
| midreset50_t0.5 - full | 13 | -0.006 | [-0.046, +0.034] |
| warmreset_t0.5 - warm_t0.5 | 5 | +0.056 | [-0.008, +0.120] |
| resetfinal_t0.5 - warm_t0.5 | 5 | +0.072 | [+0.012, +0.132] |
| midfinal_t0.5 - warm_t0.5 | 5 | +0.080 | [+0.012, +0.148] |
| midfinal50_t0.5 - warm_t0.5 | 5 | -0.028 | [-0.096, +0.040] |
| midreset_t0.5 - warm_t0.5 | 5 | +0.092 | [+0.036, +0.148] |
| midreset50_t0.5 - warm_t0.5 | 5 | +0.008 | [-0.036, +0.052] |
| warm_t0.5 - plain_k2 | 5 | -0.080 | [-0.140, -0.020] |
| warm_t0.5 - warmreset_t0.5 | 5 | -0.056 | [-0.116, +0.004] |
| warm_t0.5 - resetfinal_t0.5 | 5 | -0.072 | [-0.132, -0.012] |
| warm_t0.5 - midfinal_t0.5 | 5 | -0.080 | [-0.148, -0.012] |
| warm_t0.5 - midreset_t0.5 | 5 | -0.092 | [-0.148, -0.036] |
| warm_t0.5 - full | 5 | -0.044 | [-0.108, +0.020] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
