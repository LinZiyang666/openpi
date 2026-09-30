# Direct-token PCA: observed results

| arm | total fit s | PCA s v0 / v1 | explained variance v0 / v1 | process peak GiB | projection median ms (2 cameras, 3 threads) | A median ms |
|---|---:|---:|---:|---:|---:|---:|
| r6p2_direct_g_l10_50 | 741.0 | 384.2 / 355.2 | 0.6988 / 0.3608 | 3.07 | 10.471 | 0.797 |
| r6p2_direct_g_l10_500 | 5701.5 | 2967.4 / 2721.8 | 0.6267 / 0.3190 | 3.11 | 11.752 | 0.828 |
| r6p2_direct_g_sp_50 | 339.3 | 169.0 / 168.7 | 0.6940 / 0.4415 | 3.06 | 14.025 | 1.089 |
| r6p2_direct_g_sp_500 | 1905.5 | 1000.9 / 901.6 | 0.6151 / 0.3799 | 3.13 | 9.544 | 0.691 |
| r6p2_direct_p_l10_50 | 806.9 | 424.7 / 379.8 | 0.8161 / 0.6114 | 3.07 | 12.119 | 0.901 |
| r6p2_direct_p_l10_500 | 6662.9 | 3272.6 / 3380.0 | 0.7704 / 0.5752 | 3.11 | 9.531 | 0.742 |
| r6p2_direct_p_sp_50 | 450.9 | 229.9 / 219.2 | 0.8712 / 0.7005 | 3.06 | 10.913 | 0.839 |
| r6p2_direct_p_sp_500 | 2336.6 | 1253.9 / 1080.0 | 0.8338 / 0.6586 | 3.13 | 11.064 | 0.833 |

Peak combined fit-worker RSS (sampled every 2 s): 6.81 GiB. Four workers × three BLAS threads plus the one-thread coordinator; the single three-thread check process fits within 16 total. Each projection matrix is 134,217,728 bytes (128 MiB), or 256 MiB for two cameras; means add 4 MiB.

CPU timings are observed under shared load, with recorded token arrays already on CPU. They exclude stage-1 inference and live GPU-to-CPU transfer. Full-query timings and projection p95 are in summary.json. GPU was not run under the CPU-only constraint.

## Held-out recorded-query sanity checks (descriptive)

| arm | stream | queries | mean top-16 overlap with A | mean sigma-RMS action error: direct | A |
|---|---|---:|---:|---:|---:|
| r6p2_direct_g_l10_50 | cache | 345 | 0.6842 | 0.4515 | 0.4421 |
| r6p2_direct_g_l10_50 | inf | 344 | 0.7818 | 0.4358 | 0.4336 |
| r6p2_direct_g_l10_500 | cache | 345 | 0.5274 | 0.4085 | 0.3989 |
| r6p2_direct_g_l10_500 | inf | 344 | 0.6072 | 0.3229 | 0.3219 |
| r6p2_direct_g_sp_50 | cache | 343 | 0.8508 | 0.4959 | 0.5005 |
| r6p2_direct_g_sp_50 | inf | 337 | 0.8815 | 0.4402 | 0.4492 |
| r6p2_direct_g_sp_500 | cache | 343 | 0.6376 | 0.4150 | 0.4213 |
| r6p2_direct_g_sp_500 | inf | 337 | 0.6604 | 0.3649 | 0.3702 |
| r6p2_direct_p_l10_50 | cache | 347 | 0.8392 | 0.5148 | 0.5223 |
| r6p2_direct_p_l10_50 | inf | 343 | 0.8714 | 0.3857 | 0.3940 |
| r6p2_direct_p_l10_500 | cache | 347 | 0.7651 | 0.4401 | 0.4409 |
| r6p2_direct_p_l10_500 | inf | 343 | 0.8453 | 0.2743 | 0.2764 |
| r6p2_direct_p_sp_50 | cache | 338 | 0.8774 | 0.5014 | 0.5044 |
| r6p2_direct_p_sp_50 | inf | 337 | 0.9108 | 0.4350 | 0.4382 |
| r6p2_direct_p_sp_500 | cache | 338 | 0.7986 | 0.4243 | 0.4180 |
| r6p2_direct_p_sp_500 | inf | 337 | 0.8327 | 0.3137 | 0.3122 |

PASS: 5,468 query comparisons with stock BlindAWM on the new visual features and with exact grid4 A control; unchanged state block; 160 independent main/early metric refits; 5,336 anchor-tail checks.

The grid4 control separately matches all payload fields, including the optional still diagnostic, on 5,468 recorded queries. The direct-grid comparison excludes only that unused diagnostic when prior full tokens are unavailable.

Real-plugin smoke PASS: 2 arms × 4 complete recorded episodes, 202 decisions, no MISS, one blind tail maximum. Token extraction/serialization PASS on 64 recorded query rows and 64 library rows, both cameras; exact HDF5/store/serializer/plugin values. Library and query source-file sets are disjoint. See token_equality.json for the existing pi05 bf16-pooling vs f32-pooling difference.

## Final prefits

| file | bytes | SHA256 |
|---|---:|---|
| r6p2_direct_p_l10_50.pkl | 281685608 | `66807254b32dd22a30c87218246b57ad399b9419f3bf2355da936b58fc33cac7` |
| r6p2_direct_p_l10_500.pkl | 350858772 | `2f0a6de17acc18d7d846fffbcf99bc0cd063f24c447de23587c133674416f367` |
| r6p2_direct_p_sp_50.pkl | 277503854 | `07129646a57b450d67fcb2d24f3ae2fb18a5d7bf7baafed30bfdb6f77af87ecd` |
| r6p2_direct_p_sp_500.pkl | 303003262 | `53656435170de32fcbf185fe8c26b9c446085298ad1458bf904a820b2c362ee2` |
| r6p2_direct_g_l10_50.pkl | 283729860 | `b3532ffb9b9f30332f61bab15cea22e22152bfc75cf818fd24892c40be1d0430` |
| r6p2_direct_g_l10_500.pkl | 374025289 | `6d94faa1ef5420c2de21a5eeae959edbb495f986370aac461d07ba42709b6621` |
| r6p2_direct_g_sp_50.pkl | 278436261 | `308fa8c812b7aa0bc5cbf6cd39db8fbbf1c09d2901ab78f579ba91e25a9b87bf` |
| r6p2_direct_g_sp_500.pkl | 314198713 | `b18970a8360ab57e2b3cf037cca2a16f61c388f4b3f486ad7865424c7d78a33e` |
