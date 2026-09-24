## Warm-start continuation variants — groot (m=1, t=0.75; descriptive, 95% paired bootstrap)

missing arms: warmshoot_t0.75

### CloseBlenderLid

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.80 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.50 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.76 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.56 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.60 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.58 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.80 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.56 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.64 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.06 | [-0.10, +0.22] | 9 / 6 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.20 | [-0.36, -0.04] | 5 / 15 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.22, +0.18] | 13 / 14 |
| warmreset_t0.75 - full | 50 | -0.24 | [-0.42, -0.06] | 5 / 17 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.10 | [-0.06, +0.26] | 12 / 7 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.16 | [-0.32, +0.02] | 6 / 14 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| resetfinal_t0.75 - full | 50 | -0.20 | [-0.34, -0.06] | 3 / 13 |
| midfinal_t0.75 - plain_k1 | 50 | +0.08 | [-0.10, +0.26] | 13 / 9 |
| midfinal_t0.75 - warm_t0.75 | 50 | -0.18 | [-0.34, -0.02] | 5 / 14 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.02 | [-0.18, +0.22] | 14 / 13 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | -0.02 | [-0.18, +0.14] | 8 / 9 |
| midfinal_t0.75 - full | 50 | -0.22 | [-0.38, -0.08] | 3 / 14 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.30 | [+0.14, +0.46] | 19 / 4 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.04 | [-0.10, +0.18] | 8 / 6 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.24 | [+0.06, +0.42] | 19 / 7 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.20 | [+0.02, +0.38] | 16 / 6 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.22 | [+0.08, +0.36] | 13 / 2 |
| midfinal50_t0.75 - full | 50 | +0.00 | [-0.16, +0.16] | 9 / 9 |
| midreset_t0.75 - plain_k1 | 50 | +0.06 | [-0.12, +0.24] | 11 / 8 |
| midreset_t0.75 - warm_t0.75 | 50 | -0.20 | [-0.38, -0.02] | 6 / 16 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.16, +0.16] | 9 / 9 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | -0.04 | [-0.16, +0.08] | 4 / 6 |
| midreset_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.16, +0.12] | 6 / 7 |
| midreset_t0.75 - full | 50 | -0.24 | [-0.42, -0.06] | 5 / 17 |
| midreset50_t0.75 - plain_k1 | 50 | +0.14 | [-0.04, +0.30] | 13 / 6 |
| midreset50_t0.75 - warm_t0.75 | 50 | -0.12 | [-0.26, +0.02] | 4 / 10 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.06 | [-0.08, +0.20] | 8 / 5 |
| midreset50_t0.75 - full | 50 | -0.16 | [-0.32, +0.00] | 6 / 14 |
| warm_t0.75 - plain_k1 | 50 | +0.26 | [+0.10, +0.42] | 17 / 4 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.20 | [+0.04, +0.36] | 15 / 5 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.16 | [+0.00, +0.32] | 14 / 6 |
| warm_t0.75 - midfinal_t0.75 | 50 | +0.18 | [+0.02, +0.34] | 14 / 5 |
| warm_t0.75 - full | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |

### CloseFridge

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.34 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.36 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.62 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.30 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.28 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.58 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.86 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.50 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.84 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.06 | [-0.24, +0.12] | 9 / 12 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.32 | [-0.48, -0.16] | 3 / 19 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.28 | [-0.46, -0.10] | 6 / 20 |
| warmreset_t0.75 - full | 50 | -0.04 | [-0.24, +0.14] | 11 / 13 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.34 | [-0.50, -0.18] | 2 / 19 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | -0.02 | [-0.16, +0.12] | 6 / 7 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | -0.30 | [-0.48, -0.12] | 5 / 20 |
| resetfinal_t0.75 - full | 50 | -0.06 | [-0.24, +0.12] | 9 / 12 |
| midfinal_t0.75 - plain_k1 | 50 | +0.22 | [+0.04, +0.40] | 18 / 7 |
| midfinal_t0.75 - warm_t0.75 | 50 | -0.04 | [-0.24, +0.16] | 11 / 13 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.28 | [+0.10, +0.46] | 20 / 6 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.30 | [+0.12, +0.48] | 20 / 5 |
| midfinal_t0.75 - full | 50 | +0.24 | [+0.08, +0.40] | 16 / 4 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.50 | [+0.36, +0.64] | 25 / 0 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.24 | [+0.10, +0.38] | 14 / 2 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.56 | [+0.40, +0.72] | 30 / 2 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.58 | [+0.44, +0.72] | 29 / 0 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.28 | [+0.14, +0.42] | 15 / 1 |
| midfinal50_t0.75 - full | 50 | +0.52 | [+0.34, +0.68] | 29 / 3 |
| midreset_t0.75 - plain_k1 | 50 | +0.14 | [-0.02, +0.30] | 12 / 5 |
| midreset_t0.75 - warm_t0.75 | 50 | -0.12 | [-0.30, +0.06] | 8 / 14 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.20 | [+0.02, +0.38] | 16 / 6 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.22 | [+0.06, +0.38] | 16 / 5 |
| midreset_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.26, +0.10] | 9 / 13 |
| midreset_t0.75 - full | 50 | +0.16 | [-0.02, +0.34] | 15 / 7 |
| midreset50_t0.75 - plain_k1 | 50 | +0.48 | [+0.34, +0.62] | 24 / 0 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.22 | [+0.06, +0.38] | 15 / 4 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.54 | [+0.40, +0.68] | 28 / 1 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.56 | [+0.42, +0.70] | 28 / 0 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.26 | [+0.10, +0.42] | 16 / 3 |
| midreset50_t0.75 - full | 50 | +0.50 | [+0.34, +0.66] | 27 / 2 |
| warm_t0.75 - plain_k1 | 50 | +0.26 | [+0.08, +0.44] | 19 / 6 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.32 | [+0.16, +0.48] | 19 / 3 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.34 | [+0.18, +0.50] | 19 / 2 |
| warm_t0.75 - midfinal_t0.75 | 50 | +0.04 | [-0.14, +0.22] | 13 / 11 |
| warm_t0.75 - full | 50 | +0.28 | [+0.08, +0.48] | 22 / 8 |

### CoffeeSetupMug

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.58 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.52 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.30 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.46 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.46 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.46 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.60 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.38 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.16 | [+0.00, +0.32] | 13 / 5 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | -0.02 | [-0.08, +0.04] | 1 / 2 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| warmreset_t0.75 - full | 50 | -0.12 | [-0.28, +0.04] | 5 / 11 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.04 | [-0.18, +0.10] | 5 / 7 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.18 | [+0.02, +0.34] | 14 / 5 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.02 | [-0.04, +0.10] | 2 / 1 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | +0.02 | [-0.10, +0.14] | 6 / 5 |
| resetfinal_t0.75 - full | 50 | -0.10 | [-0.26, +0.06] | 6 / 11 |
| midfinal_t0.75 - plain_k1 | 50 | -0.06 | [-0.22, +0.10] | 6 / 9 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.16 | [+0.00, +0.32] | 13 / 5 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | -0.02 | [-0.16, +0.10] | 5 / 6 |
| midfinal_t0.75 - full | 50 | -0.12 | [-0.28, +0.04] | 5 / 11 |
| midfinal50_t0.75 - plain_k1 | 50 | -0.06 | [-0.22, +0.10] | 6 / 9 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.16 | [+0.00, +0.32] | 14 / 6 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.18, +0.18] | 10 / 10 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | -0.02 | [-0.20, +0.16] | 9 / 10 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.18, +0.18] | 10 / 10 |
| midfinal50_t0.75 - full | 50 | -0.12 | [-0.30, +0.06] | 8 / 14 |
| midreset_t0.75 - plain_k1 | 50 | +0.08 | [-0.08, +0.24] | 10 / 6 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.30 | [+0.14, +0.46] | 18 / 3 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.14 | [+0.02, +0.26] | 9 / 2 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.12 | [+0.00, +0.24] | 8 / 2 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.14 | [+0.06, +0.24] | 7 / 0 |
| midreset_t0.75 - full | 50 | +0.02 | [-0.14, +0.18] | 9 / 8 |
| midreset50_t0.75 - plain_k1 | 50 | -0.14 | [-0.30, +0.02] | 5 / 12 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | -0.10 | [-0.26, +0.06] | 6 / 11 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.24, +0.08] | 7 / 11 |
| midreset50_t0.75 - full | 50 | -0.20 | [-0.34, -0.06] | 3 / 13 |
| warm_t0.75 - plain_k1 | 50 | -0.22 | [-0.38, -0.06] | 4 / 15 |
| warm_t0.75 - warmreset_t0.75 | 50 | -0.16 | [-0.32, +0.00] | 5 / 13 |
| warm_t0.75 - resetfinal_t0.75 | 50 | -0.18 | [-0.34, -0.02] | 5 / 14 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.16 | [-0.32, +0.00] | 5 / 13 |
| warm_t0.75 - full | 50 | -0.28 | [-0.44, -0.12] | 4 / 18 |

### OpenCabinet

cross-arm model/env identity: identical; worker islands 7, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.90 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.90 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.78 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.84 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.78 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.86 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.80 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.84 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.86 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.06 | [-0.14, +0.02] | 1 / 4 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.08, +0.20] | 8 / 5 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.06 | [-0.04, +0.16] | 5 / 2 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| warmreset_t0.75 - full | 50 | -0.06 | [-0.16, +0.04] | 2 / 5 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.12 | [-0.24, +0.00] | 2 / 8 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.00 | [-0.16, +0.16] | 8 / 8 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | -0.06 | [-0.16, +0.04] | 2 / 5 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| resetfinal_t0.75 - full | 50 | -0.12 | [-0.22, -0.04] | 0 / 6 |
| midfinal_t0.75 - plain_k1 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.08 | [-0.04, +0.22] | 8 / 4 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| midfinal_t0.75 - full | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midfinal50_t0.75 - plain_k1 | 50 | -0.10 | [-0.24, +0.04] | 4 / 9 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | -0.04 | [-0.18, +0.10] | 6 / 8 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.12, +0.16] | 8 / 7 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | -0.06 | [-0.18, +0.06] | 4 / 7 |
| midfinal50_t0.75 - full | 50 | -0.10 | [-0.22, +0.02] | 2 / 7 |
| midreset_t0.75 - plain_k1 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.08, +0.20] | 8 / 5 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.08, +0.08] | 2 / 2 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.06 | [-0.02, +0.16] | 4 / 1 |
| midreset_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| midreset_t0.75 - full | 50 | -0.06 | [-0.16, +0.04] | 2 / 5 |
| midreset50_t0.75 - plain_k1 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midreset50_t0.75 - full | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| warm_t0.75 - plain_k1 | 50 | -0.12 | [-0.26, +0.02] | 4 / 10 |
| warm_t0.75 - warmreset_t0.75 | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.16, +0.16] | 8 / 8 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| warm_t0.75 - full | 50 | -0.12 | [-0.26, +0.02] | 3 / 9 |

### OpenDrawer

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.72 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.72 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.68 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.64 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.62 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.74 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.58 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.64 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.62 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.10 | [-0.26, +0.06] | 6 / 11 |
| warmreset_t0.75 - full | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.10 | [-0.22, +0.02] | 3 / 8 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.06 | [-0.24, +0.12] | 8 / 11 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | -0.12 | [-0.26, +0.02] | 3 / 9 |
| resetfinal_t0.75 - full | 50 | -0.10 | [-0.24, +0.04] | 5 / 10 |
| midfinal_t0.75 - plain_k1 | 50 | +0.02 | [-0.14, +0.18] | 8 / 7 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.06, +0.18] | 7 / 4 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.10 | [-0.06, +0.26] | 11 / 6 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.12 | [+0.00, +0.26] | 9 / 3 |
| midfinal_t0.75 - full | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| midfinal50_t0.75 - plain_k1 | 50 | -0.14 | [-0.28, +0.00] | 4 / 11 |
| midfinal50_t0.75 - warm_t0.75 | 50 | -0.10 | [-0.24, +0.04] | 4 / 9 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | -0.16 | [-0.30, -0.02] | 3 / 11 |
| midfinal50_t0.75 - full | 50 | -0.14 | [-0.28, +0.00] | 4 / 11 |
| midreset_t0.75 - plain_k1 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| midreset_t0.75 - warm_t0.75 | 50 | -0.04 | [-0.18, +0.10] | 6 / 8 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.14, +0.14] | 7 / 7 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| midreset_t0.75 - midfinal_t0.75 | 50 | -0.10 | [-0.22, +0.02] | 2 / 7 |
| midreset_t0.75 - full | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| midreset50_t0.75 - plain_k1 | 50 | -0.10 | [-0.24, +0.04] | 4 / 9 |
| midreset50_t0.75 - warm_t0.75 | 50 | -0.06 | [-0.22, +0.08] | 6 / 9 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | -0.02 | [-0.20, +0.16] | 9 / 10 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.14, +0.14] | 7 / 7 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | -0.12 | [-0.26, +0.02] | 3 / 9 |
| midreset50_t0.75 - full | 50 | -0.10 | [-0.22, +0.02] | 3 / 8 |
| warm_t0.75 - plain_k1 | 50 | -0.04 | [-0.18, +0.10] | 5 / 7 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.06 | [-0.12, +0.22] | 11 / 8 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.06 | [-0.18, +0.06] | 4 / 7 |
| warm_t0.75 - full | 50 | -0.04 | [-0.18, +0.10] | 5 / 7 |

### OpenStandMixerHead

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.72 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.68 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.70 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.70 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.68 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.70 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.70 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.68 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.72 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.12, +0.10] | 4 / 4 |
| warmreset_t0.75 - full | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| resetfinal_t0.75 - full | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| midfinal_t0.75 - plain_k1 | 50 | +0.02 | [-0.06, +0.10] | 3 / 2 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.10, +0.12] | 4 / 4 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midfinal_t0.75 - full | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midfinal50_t0.75 - full | 50 | -0.02 | [-0.10, +0.06] | 2 / 3 |
| midreset_t0.75 - plain_k1 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midreset_t0.75 - warm_t0.75 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| midreset_t0.75 - warmreset_t0.75 | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midreset_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.08, +0.04] | 1 / 2 |
| midreset_t0.75 - full | 50 | -0.04 | [-0.14, +0.06] | 2 / 4 |
| midreset50_t0.75 - plain_k1 | 50 | +0.04 | [-0.06, +0.14] | 4 / 2 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.02 | [-0.04, +0.10] | 2 / 1 |
| midreset50_t0.75 - full | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| warm_t0.75 - plain_k1 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| warm_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| warm_t0.75 - full | 50 | -0.02 | [-0.14, +0.10] | 4 / 5 |

### SlideDishwasherRack

cross-arm model/env identity: identical; worker islands 8, serving runtimes 8 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.54 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.40 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.42 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.40 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.40 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.00 | [-0.10, +0.12] | 4 / 4 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.18, +0.00] | 1 / 5 |
| warmreset_t0.75 - full | 50 | -0.14 | [-0.26, -0.02] | 2 / 9 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.08 | [-0.02, +0.18] | 6 / 2 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.10, +0.10] | 4 / 4 |
| resetfinal_t0.75 - full | 50 | -0.06 | [-0.22, +0.08] | 6 / 9 |
| midfinal_t0.75 - plain_k1 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.02, +0.14] | 4 / 1 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.10, +0.12] | 4 / 4 |
| midfinal_t0.75 - full | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.02, +0.14] | 4 / 1 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.12, +0.10] | 4 / 4 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midfinal50_t0.75 - full | 50 | -0.06 | [-0.18, +0.06] | 4 / 7 |
| midreset_t0.75 - plain_k1 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.04, +0.16] | 5 / 2 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.08 | [-0.02, +0.20] | 6 / 2 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.12, +0.12] | 4 / 4 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.10, +0.10] | 3 / 3 |
| midreset_t0.75 - full | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |
| midreset50_t0.75 - plain_k1 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| midreset50_t0.75 - warm_t0.75 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.20, +0.02] | 2 / 6 |
| midreset50_t0.75 - full | 50 | -0.14 | [-0.26, -0.02] | 2 / 9 |
| warm_t0.75 - plain_k1 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| warm_t0.75 - resetfinal_t0.75 | 50 | -0.06 | [-0.18, +0.06] | 3 / 6 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.06 | [-0.16, +0.02] | 1 / 4 |
| warm_t0.75 - full | 50 | -0.12 | [-0.24, -0.02] | 1 / 7 |

### TurnOnSinkFaucet

cross-arm model/env identity: identical; worker islands 8, serving runtimes 8 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.18 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.04 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.28 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 100 (50 ids × replicates) | 0.22 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 100 (50 ids × replicates) | 0.25 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.32 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.44 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.32 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.40 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.18 | [+0.08, +0.28] | 15 / 1 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.06 | [-0.20, +0.08] | 9 / 11 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | -0.03 | [-0.14, +0.07] | 10 / 10 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.10 | [-0.26, +0.06] | 11 / 14 |
| warmreset_t0.75 - full | 50 | +0.04 | [-0.09, +0.17] | 11 / 7 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.21 | [+0.11, +0.32] | 18 / 1 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.03 | [-0.18, +0.11] | 12 / 11 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.03 | [-0.07, +0.14] | 10 / 10 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | -0.07 | [-0.22, +0.07] | 12 / 12 |
| resetfinal_t0.75 - full | 50 | +0.07 | [-0.05, +0.19] | 13 / 7 |
| midfinal_t0.75 - plain_k1 | 50 | +0.28 | [+0.14, +0.42] | 15 / 1 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.10 | [-0.06, +0.26] | 14 / 11 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.07 | [-0.08, +0.22] | 12 / 12 |
| midfinal_t0.75 - full | 50 | +0.14 | [+0.00, +0.28] | 11 / 4 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.40 | [+0.26, +0.54] | 21 / 1 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.16 | [-0.02, +0.32] | 14 / 6 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.22 | [+0.10, +0.35] | 16 / 4 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.19 | [+0.04, +0.34] | 19 / 7 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.12 | [-0.06, +0.30] | 13 / 7 |
| midfinal50_t0.75 - full | 50 | +0.26 | [+0.14, +0.38] | 13 / 0 |
| midreset_t0.75 - plain_k1 | 50 | +0.28 | [+0.14, +0.42] | 15 / 1 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.10 | [-0.04, +0.24] | 12 / 8 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.07 | [-0.07, +0.22] | 13 / 11 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.00 | [-0.16, +0.16] | 8 / 8 |
| midreset_t0.75 - full | 50 | +0.14 | [+0.00, +0.28] | 10 / 3 |
| midreset50_t0.75 - plain_k1 | 50 | +0.36 | [+0.22, +0.50] | 19 / 1 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.12 | [-0.08, +0.32] | 16 / 10 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.18 | [+0.05, +0.32] | 15 / 6 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.15 | [+0.00, +0.30] | 17 / 9 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.08 | [-0.10, +0.26] | 13 / 9 |
| midreset50_t0.75 - full | 50 | +0.22 | [+0.08, +0.36] | 13 / 2 |
| warm_t0.75 - plain_k1 | 50 | +0.24 | [+0.12, +0.38] | 13 / 1 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.06 | [-0.08, +0.20] | 11 / 9 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.03 | [-0.11, +0.18] | 11 / 12 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 7 / 9 |
| warm_t0.75 - full | 50 | +0.10 | [-0.06, +0.26] | 12 / 7 |

### PickPlaceCounterToCabinet

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.62 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.46 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.32 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.40 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.46 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.58 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.56 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.52 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.06 | [-0.24, +0.10] | 8 / 11 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.08 | [-0.06, +0.22] | 9 / 5 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | -0.08 | [-0.22, +0.06] | 4 / 8 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.06 | [-0.22, +0.10] | 6 / 9 |
| warmreset_t0.75 - full | 50 | -0.22 | [-0.38, -0.06] | 5 / 16 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.02 | [-0.12, +0.16] | 7 / 6 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.16 | [+0.00, +0.32] | 12 / 4 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.08 | [-0.06, +0.22] | 8 / 4 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | +0.02 | [-0.14, +0.18] | 8 / 7 |
| resetfinal_t0.75 - full | 50 | -0.14 | [-0.30, +0.02] | 6 / 13 |
| midfinal_t0.75 - plain_k1 | 50 | +0.00 | [-0.16, +0.16] | 8 / 8 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.14 | [+0.00, +0.28] | 11 / 4 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.06 | [-0.10, +0.22] | 9 / 6 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | -0.02 | [-0.18, +0.14] | 7 / 8 |
| midfinal_t0.75 - full | 50 | -0.16 | [-0.32, -0.02] | 4 / 12 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.12 | [-0.04, +0.28] | 12 / 6 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.26 | [+0.10, +0.42] | 17 / 4 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.18 | [+0.00, +0.36] | 15 / 6 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.10 | [-0.10, +0.30] | 15 / 10 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.12 | [-0.02, +0.26] | 10 / 4 |
| midfinal50_t0.75 - full | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midreset_t0.75 - plain_k1 | 50 | +0.10 | [-0.04, +0.26] | 10 / 5 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.24 | [+0.08, +0.40] | 15 / 3 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.16 | [+0.02, +0.30] | 11 / 3 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.10 | [-0.04, +0.24] | 9 / 4 |
| midreset_t0.75 - full | 50 | -0.06 | [-0.20, +0.08] | 5 / 8 |
| midreset50_t0.75 - plain_k1 | 50 | +0.06 | [-0.10, +0.22] | 10 / 7 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.20 | [+0.04, +0.36] | 14 / 4 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.12 | [-0.06, +0.30] | 13 / 7 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.04 | [-0.14, +0.22] | 11 / 9 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.06 | [-0.10, +0.22] | 9 / 6 |
| midreset50_t0.75 - full | 50 | -0.10 | [-0.28, +0.08] | 8 / 13 |
| warm_t0.75 - plain_k1 | 50 | -0.14 | [-0.30, +0.02] | 6 / 13 |
| warm_t0.75 - warmreset_t0.75 | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| warm_t0.75 - resetfinal_t0.75 | 50 | -0.16 | [-0.32, -0.02] | 4 / 12 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.14 | [-0.28, +0.00] | 4 / 11 |
| warm_t0.75 - full | 50 | -0.30 | [-0.48, -0.12] | 4 / 19 |

### PickPlaceCounterToStove

cross-arm model/env identity: identical; worker islands 7, serving runtimes 7 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.82 | ok | 4.0 | 0.0 | – |
| plain_k1 | 100 | 0.94 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 100 | 0.49 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 100 (50 ids × replicates) | 0.94 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 100 (50 ids × replicates) | 0.94 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.88 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.70 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.92 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.86 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.02 | [-0.07, +0.02] | 1 / 2 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.52 | [+0.36, +0.67] | 28 / 2 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.06, +0.06] | 2 / 2 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | +0.06 | [-0.03, +0.15] | 5 / 2 |
| warmreset_t0.75 - full | 50 | +0.12 | [+0.01, +0.23] | 8 / 2 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.02 | [-0.07, +0.02] | 1 / 2 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.52 | [+0.39, +0.66] | 27 / 0 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.06, +0.06] | 2 / 2 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | +0.06 | [-0.02, +0.14] | 5 / 1 |
| resetfinal_t0.75 - full | 50 | +0.12 | [+0.02, +0.23] | 8 / 1 |
| midfinal_t0.75 - plain_k1 | 50 | -0.08 | [-0.16, -0.02] | 0 / 4 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.46 | [+0.32, +0.60] | 23 / 0 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | -0.06 | [-0.16, +0.03] | 2 / 5 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | -0.06 | [-0.14, +0.02] | 1 / 5 |
| midfinal_t0.75 - full | 50 | +0.06 | [-0.06, +0.18] | 6 / 3 |
| midfinal50_t0.75 - plain_k1 | 50 | -0.26 | [-0.40, -0.12] | 2 / 15 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.28 | [+0.14, +0.42] | 15 / 1 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | -0.24 | [-0.39, -0.09] | 4 / 15 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | -0.24 | [-0.39, -0.09] | 4 / 15 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | -0.18 | [-0.34, -0.02] | 5 / 14 |
| midfinal50_t0.75 - full | 50 | -0.12 | [-0.28, +0.04] | 6 / 12 |
| midreset_t0.75 - plain_k1 | 50 | -0.04 | [-0.10, +0.00] | 0 / 2 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.50 | [+0.36, +0.64] | 26 / 1 |
| midreset_t0.75 - warmreset_t0.75 | 50 | -0.02 | [-0.07, +0.02] | 1 / 2 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | -0.02 | [-0.08, +0.04] | 1 / 3 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| midreset_t0.75 - full | 50 | +0.10 | [+0.00, +0.20] | 6 / 1 |
| midreset50_t0.75 - plain_k1 | 50 | -0.10 | [-0.22, +0.02] | 2 / 7 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.44 | [+0.30, +0.58] | 22 / 0 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | -0.08 | [-0.20, +0.04] | 4 / 7 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | -0.08 | [-0.20, +0.04] | 4 / 7 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.16, +0.12] | 6 / 7 |
| midreset50_t0.75 - full | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| warm_t0.75 - plain_k1 | 100 | -0.45 | [-0.56, -0.34] | 4 / 49 |
| warm_t0.75 - warmreset_t0.75 | 50 | -0.52 | [-0.67, -0.36] | 2 / 28 |
| warm_t0.75 - resetfinal_t0.75 | 50 | -0.52 | [-0.65, -0.38] | 0 / 27 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.46 | [-0.60, -0.32] | 0 / 23 |
| warm_t0.75 - full | 50 | -0.40 | [-0.58, -0.22] | 4 / 24 |

### PickPlaceDrawerToCounter

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.66 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.40 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.36 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.36 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.44 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.54 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.50 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.56 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.04 | [-0.14, +0.06] | 3 / 5 |
| warmreset_t0.75 - warm_t0.75 | 50 | -0.12 | [-0.28, +0.04] | 6 / 12 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.24, +0.08] | 6 / 10 |
| warmreset_t0.75 - full | 50 | -0.30 | [-0.44, -0.16] | 2 / 17 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.04 | [-0.18, +0.10] | 6 / 8 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.12 | [-0.28, +0.04] | 6 / 12 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.14, +0.14] | 6 / 6 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | -0.08 | [-0.24, +0.08] | 7 / 11 |
| resetfinal_t0.75 - full | 50 | -0.30 | [-0.46, -0.14] | 3 / 18 |
| midfinal_t0.75 - plain_k1 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| midfinal_t0.75 - warm_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.08 | [-0.08, +0.24] | 10 / 6 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| midfinal_t0.75 - full | 50 | -0.22 | [-0.38, -0.06] | 4 / 15 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.14 | [-0.02, +0.30] | 12 / 5 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.06 | [-0.08, +0.20] | 8 / 5 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.18 | [+0.02, +0.34] | 14 / 5 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.18 | [+0.02, +0.34] | 13 / 4 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.10 | [-0.06, +0.26] | 11 / 6 |
| midfinal50_t0.75 - full | 50 | -0.12 | [-0.30, +0.06] | 8 / 14 |
| midreset_t0.75 - plain_k1 | 50 | +0.10 | [-0.06, +0.26] | 11 / 6 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.02 | [-0.14, +0.18] | 8 / 7 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.14 | [+0.00, +0.28] | 11 / 4 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.14 | [+0.00, +0.28] | 11 / 4 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.06 | [-0.06, +0.20] | 7 / 4 |
| midreset_t0.75 - full | 50 | -0.16 | [-0.32, +0.02] | 6 / 14 |
| midreset50_t0.75 - plain_k1 | 50 | +0.16 | [-0.02, +0.34] | 15 / 7 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.20 | [+0.04, +0.36] | 14 / 4 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.20 | [+0.04, +0.36] | 14 / 4 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.12 | [-0.04, +0.28] | 12 / 6 |
| midreset50_t0.75 - full | 50 | -0.10 | [-0.26, +0.08] | 7 / 12 |
| warm_t0.75 - plain_k1 | 50 | +0.08 | [-0.08, +0.24] | 11 / 7 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.12 | [-0.04, +0.28] | 12 / 6 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.12 | [-0.04, +0.28] | 12 / 6 |
| warm_t0.75 - midfinal_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| warm_t0.75 - full | 50 | -0.18 | [-0.36, +0.00] | 7 / 16 |

### PickPlaceSinkToCounter

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.78 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.90 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.80 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.90 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.96 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.94 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.98 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.98 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.92 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | +0.00 | [-0.12, +0.12] | 5 / 5 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.10 | [-0.04, +0.24] | 9 / 4 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | -0.06 | [-0.14, +0.02] | 1 / 4 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | -0.04 | [-0.16, +0.06] | 3 / 5 |
| warmreset_t0.75 - full | 50 | +0.12 | [-0.02, +0.26] | 9 / 3 |
| resetfinal_t0.75 - plain_k1 | 50 | +0.06 | [-0.04, +0.16] | 5 / 2 |
| resetfinal_t0.75 - warm_t0.75 | 50 | +0.16 | [+0.06, +0.26] | 8 / 0 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | +0.06 | [-0.02, +0.14] | 4 / 1 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | +0.02 | [-0.04, +0.10] | 2 / 1 |
| resetfinal_t0.75 - full | 50 | +0.18 | [+0.06, +0.32] | 11 / 2 |
| midfinal_t0.75 - plain_k1 | 50 | +0.04 | [-0.06, +0.16] | 5 / 3 |
| midfinal_t0.75 - warm_t0.75 | 50 | +0.14 | [+0.04, +0.26] | 8 / 1 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | +0.04 | [-0.06, +0.16] | 5 / 3 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | -0.02 | [-0.10, +0.04] | 1 / 2 |
| midfinal_t0.75 - full | 50 | +0.16 | [+0.04, +0.28] | 10 / 2 |
| midfinal50_t0.75 - plain_k1 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| midfinal50_t0.75 - warm_t0.75 | 50 | +0.18 | [+0.06, +0.30] | 10 / 1 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.04, +0.10] | 2 / 1 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| midfinal50_t0.75 - full | 50 | +0.20 | [+0.08, +0.32] | 11 / 1 |
| midreset_t0.75 - plain_k1 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.18 | [+0.06, +0.30] | 10 / 1 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.08 | [+0.00, +0.18] | 5 / 1 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.04, +0.08] | 2 / 1 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.04 | [-0.04, +0.12] | 3 / 1 |
| midreset_t0.75 - full | 50 | +0.20 | [+0.10, +0.32] | 10 / 0 |
| midreset50_t0.75 - plain_k1 | 50 | +0.02 | [-0.08, +0.12] | 4 / 3 |
| midreset50_t0.75 - warm_t0.75 | 50 | +0.12 | [+0.00, +0.26] | 9 / 3 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | +0.02 | [-0.10, +0.14] | 5 / 4 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | -0.04 | [-0.14, +0.06] | 2 / 4 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.12, +0.08] | 3 / 4 |
| midreset50_t0.75 - full | 50 | +0.14 | [+0.00, +0.28] | 11 / 4 |
| warm_t0.75 - plain_k1 | 50 | -0.10 | [-0.24, +0.06] | 5 / 10 |
| warm_t0.75 - warmreset_t0.75 | 50 | -0.10 | [-0.24, +0.04] | 4 / 9 |
| warm_t0.75 - resetfinal_t0.75 | 50 | -0.16 | [-0.26, -0.06] | 0 / 8 |
| warm_t0.75 - midfinal_t0.75 | 50 | -0.14 | [-0.26, -0.04] | 1 / 8 |
| warm_t0.75 - full | 50 | +0.02 | [-0.16, +0.20] | 10 / 9 |

### PickPlaceToasterToCounter

cross-arm model/env identity: identical; worker islands 6, serving runtimes 6 (recorded, not gated)

| arm | n | SR | admissible | steps | miss | problems |
|---|---|---|---|---|---|---|
| full | 50 | 0.64 | ok | 4.0 | 0.0 | – |
| plain_k1 | 50 | 0.66 | ok | 1.0 | 0.0 | – |
| warm_t0.75 | 50 | 0.58 | ok | 1.0 | 0.0 | – |
| warmreset_t0.75 | 50 | 0.58 | ok | 1.0 | 0.0 | – |
| resetfinal_t0.75 | 50 | 0.54 | ok | 1.0 | 0.0 | – |
| midfinal_t0.75 | 50 | 0.50 | ok | 1.0 | 0.0 | – |
| midfinal50_t0.75 | 50 | 0.48 | ok | 1.0 | 0.0 | – |
| midreset_t0.75 | 50 | 0.58 | ok | 1.0 | 0.0 | – |
| midreset50_t0.75 | 50 | 0.56 | ok | 1.0 | 0.0 | – |

| paired difference | n | Δ | 95% CI | n10 / n01 |
|---|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| warmreset_t0.75 - warm_t0.75 | 50 | +0.00 | [-0.20, +0.18] | 12 / 12 |
| warmreset_t0.75 - resetfinal_t0.75 | 50 | +0.04 | [-0.10, +0.18] | 8 / 6 |
| warmreset_t0.75 - midfinal_t0.75 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| warmreset_t0.75 - full | 50 | -0.06 | [-0.24, +0.12] | 10 / 13 |
| resetfinal_t0.75 - plain_k1 | 50 | -0.12 | [-0.28, +0.06] | 7 / 13 |
| resetfinal_t0.75 - warm_t0.75 | 50 | -0.04 | [-0.24, +0.16] | 13 / 15 |
| resetfinal_t0.75 - warmreset_t0.75 | 50 | -0.04 | [-0.18, +0.10] | 6 / 8 |
| resetfinal_t0.75 - midfinal_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 10 / 8 |
| resetfinal_t0.75 - full | 50 | -0.10 | [-0.28, +0.08] | 9 / 14 |
| midfinal_t0.75 - plain_k1 | 50 | -0.16 | [-0.34, +0.02] | 7 / 15 |
| midfinal_t0.75 - warm_t0.75 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| midfinal_t0.75 - warmreset_t0.75 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| midfinal_t0.75 - resetfinal_t0.75 | 50 | -0.04 | [-0.20, +0.12] | 8 / 10 |
| midfinal_t0.75 - full | 50 | -0.14 | [-0.30, +0.02] | 5 / 12 |
| midfinal50_t0.75 - plain_k1 | 50 | -0.18 | [-0.34, +0.00] | 6 / 15 |
| midfinal50_t0.75 - warm_t0.75 | 50 | -0.10 | [-0.28, +0.08] | 8 / 13 |
| midfinal50_t0.75 - warmreset_t0.75 | 50 | -0.10 | [-0.26, +0.06] | 6 / 11 |
| midfinal50_t0.75 - resetfinal_t0.75 | 50 | -0.06 | [-0.22, +0.10] | 8 / 11 |
| midfinal50_t0.75 - midfinal_t0.75 | 50 | -0.02 | [-0.18, +0.14] | 9 / 10 |
| midfinal50_t0.75 - full | 50 | -0.16 | [-0.34, +0.02] | 8 / 16 |
| midreset_t0.75 - plain_k1 | 50 | -0.08 | [-0.26, +0.10] | 10 / 14 |
| midreset_t0.75 - warm_t0.75 | 50 | +0.00 | [-0.20, +0.18] | 12 / 12 |
| midreset_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.18, +0.18] | 10 / 10 |
| midreset_t0.75 - resetfinal_t0.75 | 50 | +0.04 | [-0.12, +0.20] | 9 / 7 |
| midreset_t0.75 - midfinal_t0.75 | 50 | +0.08 | [-0.04, +0.20] | 7 / 3 |
| midreset_t0.75 - full | 50 | -0.06 | [-0.22, +0.10] | 7 / 10 |
| midreset50_t0.75 - plain_k1 | 50 | -0.10 | [-0.28, +0.08] | 9 / 14 |
| midreset50_t0.75 - warm_t0.75 | 50 | -0.02 | [-0.18, +0.16] | 9 / 10 |
| midreset50_t0.75 - warmreset_t0.75 | 50 | -0.02 | [-0.20, +0.16] | 11 / 12 |
| midreset50_t0.75 - resetfinal_t0.75 | 50 | +0.02 | [-0.16, +0.20] | 10 / 9 |
| midreset50_t0.75 - midfinal_t0.75 | 50 | +0.06 | [-0.12, +0.24] | 12 / 9 |
| midreset50_t0.75 - full | 50 | -0.08 | [-0.22, +0.06] | 5 / 9 |
| warm_t0.75 - plain_k1 | 50 | -0.08 | [-0.26, +0.10] | 8 / 12 |
| warm_t0.75 - warmreset_t0.75 | 50 | +0.00 | [-0.20, +0.20] | 12 / 12 |
| warm_t0.75 - resetfinal_t0.75 | 50 | +0.04 | [-0.16, +0.24] | 15 / 13 |
| warm_t0.75 - midfinal_t0.75 | 50 | +0.08 | [-0.10, +0.26] | 12 / 8 |
| warm_t0.75 - full | 50 | -0.06 | [-0.22, +0.08] | 6 / 9 |

### Macro (mean over admissible tasks)

| arm | tasks | macro SR |
|---|---|---|
| full | 13 | 0.638 |
| plain_k1 | 13 | 0.575 |
| warm_t0.75 | 13 | 0.555 |
| warmreset_t0.75 | 13 | 0.562 |
| warmshoot_t0.75 | 0 | – |
| resetfinal_t0.75 | 13 | 0.573 |
| midfinal_t0.75 | 13 | 0.611 |
| midfinal50_t0.75 | 13 | 0.646 |
| midreset_t0.75 | 13 | 0.628 |
| midreset50_t0.75 | 13 | 0.637 |

| macro paired difference | tasks | Δ | 95% CI |
|---|---|---|---|
| warmreset_t0.75 - plain_k1 | 13 | -0.015 | [-0.050, +0.020] |
| warmreset_t0.75 - warm_t0.75 | 13 | +0.012 | [-0.029, +0.055] |
| warmreset_t0.75 - resetfinal_t0.75 | 13 | -0.012 | [-0.044, +0.021] |
| warmreset_t0.75 - midfinal_t0.75 | 13 | -0.049 | [-0.090, -0.009] |
| warmreset_t0.75 - full | 13 | -0.077 | [-0.118, -0.035] |
| resetfinal_t0.75 - plain_k1 | 13 | -0.004 | [-0.039, +0.032] |
| resetfinal_t0.75 - warm_t0.75 | 13 | +0.024 | [-0.018, +0.066] |
| resetfinal_t0.75 - warmreset_t0.75 | 13 | +0.012 | [-0.021, +0.044] |
| resetfinal_t0.75 - midfinal_t0.75 | 13 | -0.038 | [-0.075, -0.001] |
| resetfinal_t0.75 - full | 13 | -0.065 | [-0.105, -0.025] |
| midfinal_t0.75 - plain_k1 | 13 | +0.034 | [-0.006, +0.072] |
| midfinal_t0.75 - warm_t0.75 | 13 | +0.062 | [+0.022, +0.103] |
| midfinal_t0.75 - warmreset_t0.75 | 13 | +0.049 | [+0.009, +0.089] |
| midfinal_t0.75 - resetfinal_t0.75 | 13 | +0.038 | [+0.001, +0.075] |
| midfinal_t0.75 - full | 13 | -0.028 | [-0.066, +0.011] |
| midfinal50_t0.75 - plain_k1 | 13 | +0.069 | [+0.031, +0.109] |
| midfinal50_t0.75 - warm_t0.75 | 13 | +0.097 | [+0.057, +0.137] |
| midfinal50_t0.75 - warmreset_t0.75 | 13 | +0.085 | [+0.043, +0.125] |
| midfinal50_t0.75 - resetfinal_t0.75 | 13 | +0.073 | [+0.032, +0.115] |
| midfinal50_t0.75 - midfinal_t0.75 | 13 | +0.035 | [-0.003, +0.075] |
| midfinal50_t0.75 - full | 13 | +0.008 | [-0.034, +0.049] |
| midreset_t0.75 - plain_k1 | 13 | +0.051 | [+0.012, +0.089] |
| midreset_t0.75 - warm_t0.75 | 13 | +0.078 | [+0.037, +0.120] |
| midreset_t0.75 - warmreset_t0.75 | 13 | +0.066 | [+0.030, +0.103] |
| midreset_t0.75 - resetfinal_t0.75 | 13 | +0.055 | [+0.019, +0.089] |
| midreset_t0.75 - midfinal_t0.75 | 13 | +0.017 | [-0.017, +0.051] |
| midreset_t0.75 - full | 13 | -0.011 | [-0.051, +0.028] |
| midreset50_t0.75 - plain_k1 | 13 | +0.060 | [+0.020, +0.100] |
| midreset50_t0.75 - warm_t0.75 | 13 | +0.088 | [+0.046, +0.129] |
| midreset50_t0.75 - warmreset_t0.75 | 13 | +0.075 | [+0.035, +0.116] |
| midreset50_t0.75 - resetfinal_t0.75 | 13 | +0.064 | [+0.025, +0.103] |
| midreset50_t0.75 - midfinal_t0.75 | 13 | +0.026 | [-0.014, +0.065] |
| midreset50_t0.75 - full | 13 | -0.002 | [-0.042, +0.040] |
| warm_t0.75 - plain_k1 | 13 | -0.021 | [-0.062, +0.021] |
| warm_t0.75 - warmreset_t0.75 | 13 | -0.012 | [-0.055, +0.030] |
| warm_t0.75 - resetfinal_t0.75 | 13 | -0.024 | [-0.065, +0.017] |
| warm_t0.75 - midfinal_t0.75 | 13 | -0.062 | [-0.103, -0.022] |
| warm_t0.75 - full | 13 | -0.089 | [-0.134, -0.045] |

Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); `warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).
