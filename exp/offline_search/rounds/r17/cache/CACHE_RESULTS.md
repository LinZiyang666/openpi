# R17 cache study: accepted episode snapshot

UTC: 2026-10-11T20:40:58.824016+00:00. Selected sizes: [50, 100, 200, 300, 400, 500].

F500 与 Fsize 各固定40比较/5%；不随已到达 lane 缩小族，不声称合并族5%。中间尺寸与敏感性均为描述性。
Cosmos ch/chr/chg/chrg 分别复用 c/cr/cg/crg。CI为逐点双侧95% CP；差值LB为单侧。
Pairs use canonical task/init keys; only common keys contribute. The fixed 32-init sensitivity exclusion leaves spatial489/10:479 when complete.
Receipt status is information only. Cells with fewer than 500 accepted episodes are INTERIM; missing outcomes are never imputed.

## Lane receipts and evidence

| Lane | Receipt status | Receipts | accepted | cost episodes | missing cost |
|---|---|---|---:|---:|---:|
| /data/mw_finetune/eval/dpc8713063796_r2_d1dd | PASS | RECEIPT.json:PASS | 8000 | 8000 | 0 |
| /data/mw_finetune/eval/fastwamc5481985003 | PASS | RECEIPT.json:PASS | 8000 | 8000 | 0 |
| /data/mw_finetune/eval/hs5dp_ctl_142812 | PASS | RECEIPT.json:PASS | 1000 | 1000 | 0 |
| /data/mw_finetune/eval/hs6fw_res_152931 | PASS | RECEIPT.json:PASS | 1000 | 1000 | 0 |
| /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380 | FAIL | RECEIPT.json:FAIL | 726 | 726 | 0 |
| /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043 | PASS | RECEIPT.json:FAIL; RECOVERED_RECEIPT.json:PASS | 1000 | 1000 | 0 |
| /data/mw_finetune/eval/fastwamc6066547406 | FAIL | RECEIPT.json:FAIL | 40000 | 40000 | 0 |
| /data/mw_finetune/eval/csztd_r3_l48 | FAIL | RECEIPT.json:FAIL | 40000 | 40000 | 0 |
| /data/mw_finetune/eval/cosmos500w85_220813 | PASS | RECEIPT.json:PASS | 4000 | 4000 | 0 |
| /data/mw_finetune/eval/cosmossizes145_r8c | FAIL | RECEIPT.json:FAIL | 20000 | 20000 | 0 |

144/240 cells have accepted outcomes; 144/240 have all 500 episodes.

## N500 vs inference control (full cohort)

| model | group | suite | n | SR% | pairs | Δ pp | g/l | F500 CP LB pp | status |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| Cosmos | c | libero_10 | 500 | 75.20 | 500 | -23.40 | 5/122 | -30.91 | COMPLETE |
| Cosmos | c | libero_spatial | 500 | 95.00 | 500 | -1.40 | 12/19 | -6.61 | COMPLETE |
| Cosmos | cg | libero_10 | 500 | 80.00 | 500 | -18.60 | 5/98 | -25.72 | COMPLETE |
| Cosmos | cg | libero_spatial | 500 | 95.80 | 500 | -0.60 | 13/16 | -5.70 | COMPLETE |
| Cosmos | cr | libero_10 | 500 | 79.80 | 500 | -18.80 | 6/100 | -26.09 | COMPLETE |
| Cosmos | cr | libero_spatial | 500 | 95.40 | 500 | -1.00 | 12/17 | -6.08 | COMPLETE |
| Cosmos | crg | libero_10 | 500 | 85.60 | 500 | -13.00 | 7/72 | -19.80 | COMPLETE |
| Cosmos | crg | libero_spatial | 500 | 95.80 | 500 | -0.60 | 13/16 | -5.70 | COMPLETE |
| DP | c | libero_10 | 500 | 64.80 | 500 | -14.80 | 46/120 | -25.08 | COMPLETE |
| DP | c | libero_spatial | 500 | 94.60 | 500 | 3.40 | 30/13 | -2.64 | COMPLETE |
| DP | cg | libero_10 | 500 | 87.40 | 500 | 7.80 | 76/37 | -1.47 | COMPLETE |
| DP | cg | libero_spatial | 500 | 93.00 | 500 | 1.80 | 31/22 | -4.90 | COMPLETE |
| DP | cr | libero_10 | 500 | 68.20 | 500 | -11.40 | 46/103 | -21.41 | COMPLETE |
| DP | cr | libero_spatial | 500 | 93.60 | 500 | 2.40 | 30/18 | -4.00 | COMPLETE |
| DP | crg | libero_10 | 500 | 88.80 | 500 | 9.20 | 78/32 | 0.10 | COMPLETE |
| DP | crg | libero_spatial | 500 | 94.00 | 500 | 2.80 | 33/19 | -3.83 | COMPLETE |
| Fast-WAM | c | libero_10 | 500 | 69.00 | 500 | -22.60 | 22/135 | -31.81 | COMPLETE |
| Fast-WAM | c | libero_spatial | 500 | 87.80 | 500 | -3.00 | 15/30 | -9.08 | COMPLETE |
| Fast-WAM | cg | libero_10 | 500 | 81.40 | 500 | -10.20 | 27/78 | -18.74 | COMPLETE |
| Fast-WAM | cg | libero_spatial | 500 | 89.20 | 500 | -1.60 | 17/25 | -7.58 | COMPLETE |
| Fast-WAM | cr | libero_10 | 500 | 70.40 | 500 | -21.20 | 24/130 | -30.48 | COMPLETE |
| Fast-WAM | cr | libero_spatial | 500 | 88.40 | 500 | -2.40 | 14/26 | -8.20 | COMPLETE |
| Fast-WAM | crg | libero_10 | 500 | 83.40 | 500 | -8.20 | 25/66 | -16.31 | COMPLETE |
| Fast-WAM | crg | libero_spatial | 500 | 90.20 | 500 | -0.60 | 18/21 | -6.43 | COMPLETE |

## Sizes 50–400 so far (full cohort)

| model | group | suite | N50 n / SR% | N100 n / SR% | N200 n / SR% | N300 n / SR% | N400 n / SR% |
|---|---|---|---|---|---|---|---|
| DP | c | libero_spatial | 500 / 72.20 | 500 / 86.60 | 500 / 88.40 | 500 / 91.80 | 500 / 93.00 |
| DP | c | libero_10 | 500 / 24.20 | 500 / 46.60 | 500 / 48.80 | 500 / 57.80 | 500 / 61.20 |
| DP | cr | libero_spatial | 500 / 77.80 | 500 / 89.20 | 500 / 93.40 | 500 / 94.40 | 500 / 94.80 |
| DP | cr | libero_10 | 500 / 32.40 | 500 / 50.40 | 500 / 51.00 | 500 / 60.60 | 500 / 63.60 |
| DP | cg | libero_spatial | 500 / 83.80 | 500 / 91.40 | 500 / 92.20 | 500 / 92.80 | 500 / 92.00 |
| DP | cg | libero_10 | 500 / 50.60 | 500 / 73.60 | 500 / 83.80 | 500 / 82.20 | 500 / 85.80 |
| DP | crg | libero_spatial | 500 / 85.40 | 500 / 90.20 | 500 / 93.40 | 500 / 93.00 | 500 / 93.80 |
| DP | crg | libero_10 | 500 / 59.40 | 500 / 76.40 | 500 / 80.40 | 500 / 84.80 | 500 / 87.60 |
| Fast-WAM | c | libero_spatial | 500 / 78.60 | 500 / 83.20 | 500 / 84.40 | 500 / 84.80 | 500 / 85.40 |
| Fast-WAM | c | libero_10 | 500 / 39.00 | 500 / 51.00 | 500 / 56.60 | 500 / 62.60 | 500 / 64.80 |
| Fast-WAM | cr | libero_spatial | 500 / 77.00 | 500 / 83.40 | 500 / 86.40 | 500 / 87.40 | 500 / 87.60 |
| Fast-WAM | cr | libero_10 | 500 / 45.00 | 500 / 53.80 | 500 / 63.00 | 500 / 67.20 | 500 / 71.20 |
| Fast-WAM | cg | libero_spatial | 500 / 84.20 | 500 / 87.60 | 500 / 86.20 | 500 / 87.00 | 500 / 88.20 |
| Fast-WAM | cg | libero_10 | 500 / 66.20 | 500 / 71.80 | 500 / 74.80 | 500 / 77.20 | 500 / 79.60 |
| Fast-WAM | crg | libero_spatial | 500 / 85.60 | 500 / 87.20 | 500 / 87.80 | 500 / 89.20 | 500 / 88.80 |
| Fast-WAM | crg | libero_10 | 500 / 71.80 | 500 / 72.20 | 500 / 79.80 | 500 / 81.20 | 500 / 84.80 |
| Cosmos | c | libero_spatial | 500 / 11.20 | 500 / 85.60 | 500 / 93.60 | 500 / 95.20 | 500 / 94.00 |
| Cosmos | c | libero_10 | 500 / 12.00 | 500 / 45.00 | 500 / 66.00 | 500 / 70.60 | 500 / 77.20 |
| Cosmos | cr | libero_spatial | 500 / 36.80 | 500 / 90.60 | 500 / 96.00 | 500 / 96.00 | 500 / 95.60 |
| Cosmos | cr | libero_10 | 500 / 29.80 | 500 / 60.20 | 500 / 71.40 | 500 / 75.60 | 500 / 81.40 |
| Cosmos | cg | libero_spatial | 500 / 25.60 | 500 / 86.60 | 500 / 92.80 | 500 / 94.60 | 500 / 94.60 |
| Cosmos | cg | libero_10 | 500 / 22.40 | 500 / 54.00 | 500 / 74.40 | 500 / 77.00 | 500 / 84.40 |
| Cosmos | crg | libero_spatial | 500 / 45.20 | 500 / 91.80 | 500 / 94.80 | 500 / 96.20 | 500 / 95.20 |
| Cosmos | crg | libero_10 | 500 / 38.40 | 500 / 65.20 | 500 / 76.40 | 500 / 83.60 | 500 / 84.00 |

## SR and costs

| model | N | group | suite | cohort | status | n | SR% [CP95%] | cost n | fresh hit% | MISS/control | NFE/control | calls/ep |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| Cosmos | 50 | c | libero_10 | full | COMPLETE | 500 | 12.00 [9.28,15.18] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | c | libero_10 | sensitivity | COMPLETE | 479 | 12.53 [9.70,15.83] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | c | libero_spatial | full | COMPLETE | 500 | 11.20 [8.57,14.30] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | c | libero_spatial | sensitivity | COMPLETE | 489 | 11.25 [8.59,14.39] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | cg | libero_10 | full | COMPLETE | 500 | 22.40 [18.82,26.31] | 500 | 74.85 | 0.01599 | 0.07995 | 7.45 |
| Cosmos | 50 | cg | libero_10 | sensitivity | COMPLETE | 479 | 23.38 [19.66,27.43] | 479 | 74.83 | 0.01601 | 0.08003 | 7.42 |
| Cosmos | 50 | cg | libero_spatial | full | COMPLETE | 500 | 25.60 [21.83,29.66] | 500 | 76.49 | 0.01508 | 0.07539 | 2.98 |
| Cosmos | 50 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 25.97 [22.14,30.10] | 489 | 76.42 | 0.01513 | 0.07564 | 2.99 |
| Cosmos | 50 | cr | libero_10 | full | COMPLETE | 500 | 29.80 [25.82,34.02] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | cr | libero_10 | sensitivity | COMPLETE | 479 | 30.48 [26.38,34.82] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | cr | libero_spatial | full | COMPLETE | 500 | 36.80 [32.56,41.20] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 37.01 [32.72,41.46] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 50 | crg | libero_10 | full | COMPLETE | 500 | 38.40 [34.12,42.82] | 500 | 77.94 | 0.01404 | 0.07022 | 5.94 |
| Cosmos | 50 | crg | libero_10 | sensitivity | COMPLETE | 479 | 39.04 [34.65,43.57] | 479 | 78.03 | 0.01399 | 0.06994 | 5.88 |
| Cosmos | 50 | crg | libero_spatial | full | COMPLETE | 500 | 45.20 [40.78,49.68] | 500 | 80.56 | 0.01253 | 0.06266 | 2.16 |
| Cosmos | 50 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 45.60 [41.12,50.14] | 489 | 80.57 | 0.01253 | 0.06265 | 2.15 |
| Cosmos | 100 | c | libero_10 | full | COMPLETE | 500 | 45.00 [40.58,49.48] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | c | libero_10 | sensitivity | COMPLETE | 479 | 45.51 [40.99,50.09] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | c | libero_spatial | full | COMPLETE | 500 | 85.60 [82.21,88.56] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | c | libero_spatial | sensitivity | COMPLETE | 489 | 85.69 [82.26,88.67] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | cg | libero_10 | full | COMPLETE | 500 | 54.00 [49.52,58.43] | 500 | 85.11 | 0.00949 | 0.04747 | 3.68 |
| Cosmos | 100 | cg | libero_10 | sensitivity | COMPLETE | 479 | 54.28 [49.70,58.81] | 479 | 85.10 | 0.00950 | 0.04752 | 3.67 |
| Cosmos | 100 | cg | libero_spatial | full | COMPLETE | 500 | 86.60 [83.30,89.46] | 500 | 92.84 | 0.00469 | 0.02344 | 0.57 |
| Cosmos | 100 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 86.50 [83.15,89.41] | 489 | 92.93 | 0.00463 | 0.02314 | 0.56 |
| Cosmos | 100 | cr | libero_10 | full | COMPLETE | 500 | 60.20 [55.76,64.52] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | cr | libero_10 | sensitivity | COMPLETE | 479 | 60.54 [56.01,64.95] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | cr | libero_spatial | full | COMPLETE | 500 | 90.60 [87.70,93.01] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 90.59 [87.65,93.03] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 100 | crg | libero_10 | full | COMPLETE | 500 | 65.20 [60.84,69.37] | 500 | 87.59 | 0.00792 | 0.03960 | 2.80 |
| Cosmos | 100 | crg | libero_10 | sensitivity | COMPLETE | 479 | 65.55 [61.11,69.81] | 479 | 87.59 | 0.00792 | 0.03961 | 2.79 |
| Cosmos | 100 | crg | libero_spatial | full | COMPLETE | 500 | 91.80 [89.04,94.05] | 500 | 95.13 | 0.00322 | 0.01612 | 0.37 |
| Cosmos | 100 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 91.82 [89.03,94.09] | 489 | 95.28 | 0.00312 | 0.01561 | 0.35 |
| Cosmos | 200 | c | libero_10 | full | COMPLETE | 500 | 66.00 [61.66,70.15] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | c | libero_10 | sensitivity | COMPLETE | 479 | 66.81 [62.39,71.01] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | c | libero_spatial | full | COMPLETE | 500 | 93.60 [91.08,95.58] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | c | libero_spatial | sensitivity | COMPLETE | 489 | 93.46 [90.89,95.48] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | cg | libero_10 | full | COMPLETE | 500 | 74.40 [70.34,78.17] | 500 | 89.20 | 0.00691 | 0.03455 | 2.24 |
| Cosmos | 200 | cg | libero_10 | sensitivity | COMPLETE | 479 | 74.74 [70.60,78.57] | 479 | 89.29 | 0.00685 | 0.03426 | 2.21 |
| Cosmos | 200 | cg | libero_spatial | full | COMPLETE | 500 | 92.80 [90.17,94.91] | 500 | 96.01 | 0.00263 | 0.01314 | 0.29 |
| Cosmos | 200 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 92.64 [89.95,94.79] | 489 | 95.94 | 0.00268 | 0.01339 | 0.29 |
| Cosmos | 200 | cr | libero_10 | full | COMPLETE | 500 | 71.40 [67.22,75.32] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | cr | libero_10 | sensitivity | COMPLETE | 479 | 71.82 [67.56,75.80] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | cr | libero_spatial | full | COMPLETE | 500 | 96.00 [93.89,97.54] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 95.91 [93.75,97.48] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 200 | crg | libero_10 | full | COMPLETE | 500 | 76.40 [72.43,80.06] | 500 | 90.78 | 0.00590 | 0.02950 | 1.86 |
| Cosmos | 200 | crg | libero_10 | sensitivity | COMPLETE | 479 | 76.83 [72.78,80.53] | 479 | 90.92 | 0.00581 | 0.02903 | 1.82 |
| Cosmos | 200 | crg | libero_spatial | full | COMPLETE | 500 | 94.80 [92.47,96.58] | 500 | 96.68 | 0.00221 | 0.01107 | 0.24 |
| Cosmos | 200 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 94.68 [92.31,96.50] | 489 | 96.61 | 0.00226 | 0.01129 | 0.24 |
| Cosmos | 300 | c | libero_10 | full | COMPLETE | 500 | 70.60 [66.39,74.56] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | c | libero_10 | sensitivity | COMPLETE | 479 | 71.40 [67.12,75.41] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | c | libero_spatial | full | COMPLETE | 500 | 95.20 [92.94,96.90] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | c | libero_spatial | sensitivity | COMPLETE | 489 | 95.30 [93.03,97.00] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | cg | libero_10 | full | COMPLETE | 500 | 77.00 [73.06,80.62] | 500 | 89.79 | 0.00653 | 0.03266 | 2.06 |
| Cosmos | 300 | cg | libero_10 | sensitivity | COMPLETE | 479 | 77.45 [73.44,81.12] | 479 | 89.90 | 0.00647 | 0.03233 | 2.03 |
| Cosmos | 300 | cg | libero_spatial | full | COMPLETE | 500 | 94.60 [92.24,96.41] | 500 | 96.20 | 0.00253 | 0.01263 | 0.27 |
| Cosmos | 300 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 94.48 [92.07,96.33] | 489 | 96.15 | 0.00256 | 0.01279 | 0.28 |
| Cosmos | 300 | cr | libero_10 | full | COMPLETE | 500 | 75.60 [71.59,79.30] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | cr | libero_10 | sensitivity | COMPLETE | 479 | 76.62 [72.57,80.34] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | cr | libero_spatial | full | COMPLETE | 500 | 96.00 [93.89,97.54] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 96.11 [94.00,97.64] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 300 | crg | libero_10 | full | COMPLETE | 500 | 83.60 [80.06,86.74] | 500 | 92.08 | 0.00507 | 0.02536 | 1.51 |
| Cosmos | 300 | crg | libero_10 | sensitivity | COMPLETE | 479 | 84.13 [80.55,87.29] | 479 | 92.28 | 0.00494 | 0.02472 | 1.47 |
| Cosmos | 300 | crg | libero_spatial | full | COMPLETE | 500 | 96.20 [94.13,97.70] | 500 | 97.30 | 0.00180 | 0.00898 | 0.19 |
| Cosmos | 300 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 96.11 [94.00,97.64] | 489 | 97.33 | 0.00178 | 0.00889 | 0.19 |
| Cosmos | 400 | c | libero_10 | full | COMPLETE | 500 | 77.20 [73.27,80.81] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | c | libero_10 | sensitivity | COMPLETE | 479 | 77.66 [73.66,81.32] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | c | libero_spatial | full | COMPLETE | 500 | 94.00 [91.55,95.92] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | c | libero_spatial | sensitivity | COMPLETE | 489 | 93.87 [91.36,95.82] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | cg | libero_10 | full | COMPLETE | 500 | 84.40 [80.92,87.47] | 500 | 92.08 | 0.00508 | 0.02539 | 1.51 |
| Cosmos | 400 | cg | libero_10 | sensitivity | COMPLETE | 479 | 84.55 [81.00,87.67] | 479 | 92.23 | 0.00499 | 0.02493 | 1.47 |
| Cosmos | 400 | cg | libero_spatial | full | COMPLETE | 500 | 94.60 [92.24,96.41] | 500 | 96.06 | 0.00262 | 0.01312 | 0.28 |
| Cosmos | 400 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 94.48 [92.07,96.33] | 489 | 95.98 | 0.00267 | 0.01337 | 0.29 |
| Cosmos | 400 | cr | libero_10 | full | COMPLETE | 500 | 81.40 [77.71,84.72] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | cr | libero_10 | sensitivity | COMPLETE | 479 | 81.42 [77.64,84.80] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | cr | libero_spatial | full | COMPLETE | 500 | 95.60 [93.41,97.22] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 95.71 [93.51,97.32] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 400 | crg | libero_10 | full | COMPLETE | 500 | 84.00 [80.49,87.10] | 500 | 92.28 | 0.00495 | 0.02473 | 1.46 |
| Cosmos | 400 | crg | libero_10 | sensitivity | COMPLETE | 479 | 84.13 [80.55,87.29] | 479 | 92.39 | 0.00488 | 0.02438 | 1.43 |
| Cosmos | 400 | crg | libero_spatial | full | COMPLETE | 500 | 95.20 [92.94,96.90] | 500 | 96.57 | 0.00228 | 0.01140 | 0.24 |
| Cosmos | 400 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 95.30 [93.03,97.00] | 489 | 96.61 | 0.00226 | 0.01128 | 0.24 |
| Cosmos | 500 | c | libero_10 | full | COMPLETE | 500 | 75.20 [71.17,78.93] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | c | libero_10 | sensitivity | COMPLETE | 479 | 75.78 [71.69,79.55] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | c | libero_spatial | full | COMPLETE | 500 | 95.00 [92.71,96.74] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | c | libero_spatial | sensitivity | COMPLETE | 489 | 94.89 [92.55,96.66] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | cg | libero_10 | full | COMPLETE | 500 | 80.00 [76.22,83.42] | 500 | 90.62 | 0.00601 | 0.03004 | 1.84 |
| Cosmos | 500 | cg | libero_10 | sensitivity | COMPLETE | 479 | 80.58 [76.75,84.03] | 479 | 90.79 | 0.00590 | 0.02952 | 1.80 |
| Cosmos | 500 | cg | libero_spatial | full | COMPLETE | 500 | 95.80 [93.65,97.38] | 500 | 96.63 | 0.00223 | 0.01116 | 0.24 |
| Cosmos | 500 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 95.71 [93.51,97.32] | 489 | 96.57 | 0.00228 | 0.01138 | 0.24 |
| Cosmos | 500 | cr | libero_10 | full | COMPLETE | 500 | 79.80 [76.01,83.23] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | cr | libero_10 | sensitivity | COMPLETE | 479 | 80.17 [76.31,83.65] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | cr | libero_spatial | full | COMPLETE | 500 | 95.40 [93.18,97.06] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 95.30 [93.03,97.00] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Cosmos | 500 | crg | libero_10 | full | COMPLETE | 500 | 85.60 [82.21,88.56] | 500 | 92.67 | 0.00470 | 0.02348 | 1.37 |
| Cosmos | 500 | crg | libero_10 | sensitivity | COMPLETE | 479 | 85.59 [82.13,88.62] | 479 | 92.77 | 0.00463 | 0.02316 | 1.35 |
| Cosmos | 500 | crg | libero_spatial | full | COMPLETE | 500 | 95.80 [93.65,97.38] | 500 | 96.96 | 0.00201 | 0.01005 | 0.21 |
| Cosmos | 500 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 95.71 [93.51,97.32] | 489 | 96.90 | 0.00205 | 0.01025 | 0.22 |
| DP | 50 | c | libero_10 | full | COMPLETE | 500 | 24.20 [20.51,28.20] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | c | libero_10 | sensitivity | COMPLETE | 479 | 24.63 [20.84,28.75] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | c | libero_spatial | full | COMPLETE | 500 | 72.20 [68.05,76.09] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | c | libero_spatial | sensitivity | COMPLETE | 489 | 72.39 [68.20,76.31] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | cg | libero_10 | full | COMPLETE | 500 | 50.60 [46.13,55.07] | 500 | 83.94 | 0.03220 | 3.22010 | 12.35 |
| DP | 50 | cg | libero_10 | sensitivity | COMPLETE | 479 | 51.15 [46.57,55.71] | 479 | 83.89 | 0.03230 | 3.23030 | 12.35 |
| DP | 50 | cg | libero_spatial | full | COMPLETE | 500 | 83.80 [80.27,86.92] | 500 | 85.80 | 0.02879 | 2.87925 | 3.62 |
| DP | 50 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 84.05 [80.50,87.18] | 489 | 85.58 | 0.02923 | 2.92342 | 3.67 |
| DP | 50 | cr | libero_10 | full | COMPLETE | 500 | 32.40 [28.31,36.70] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | cr | libero_10 | sensitivity | COMPLETE | 479 | 32.57 [28.39,36.97] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | cr | libero_spatial | full | COMPLETE | 500 | 77.80 [73.90,81.37] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 77.91 [73.97,81.51] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | crg | libero_10 | full | COMPLETE | 500 | 59.40 [54.95,63.74] | 500 | 86.36 | 0.02738 | 2.73830 | 9.75 |
| DP | 50 | crg | libero_10 | sensitivity | COMPLETE | 479 | 60.13 [55.58,64.54] | 479 | 86.41 | 0.02727 | 2.72712 | 9.65 |
| DP | 50 | crg | libero_spatial | full | COMPLETE | 500 | 85.40 [82.00,88.38] | 500 | 85.78 | 0.02881 | 2.88114 | 3.51 |
| DP | 50 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 85.28 [81.82,88.30] | 489 | 85.61 | 0.02916 | 2.91591 | 3.55 |
| DP | 100 | c | libero_10 | full | COMPLETE | 500 | 46.60 [42.16,51.08] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | c | libero_10 | sensitivity | COMPLETE | 479 | 46.97 [42.43,51.55] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | c | libero_spatial | full | COMPLETE | 500 | 86.60 [83.30,89.46] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | c | libero_spatial | sensitivity | COMPLETE | 489 | 86.71 [83.37,89.59] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | cg | libero_10 | full | COMPLETE | 500 | 73.60 [69.50,77.41] | 500 | 86.92 | 0.02628 | 2.62828 | 8.96 |
| DP | 100 | cg | libero_10 | sensitivity | COMPLETE | 479 | 74.32 [70.16,78.18] | 479 | 86.96 | 0.02620 | 2.61962 | 8.86 |
| DP | 100 | cg | libero_spatial | full | COMPLETE | 500 | 91.40 [88.59,93.71] | 500 | 88.51 | 0.02334 | 2.33405 | 2.67 |
| DP | 100 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 91.62 [88.80,93.92] | 489 | 88.67 | 0.02304 | 2.30359 | 2.63 |
| DP | 100 | cr | libero_10 | full | COMPLETE | 500 | 50.40 [45.93,54.87] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | cr | libero_10 | sensitivity | COMPLETE | 479 | 51.15 [46.57,55.71] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | cr | libero_spatial | full | COMPLETE | 500 | 89.20 [86.14,91.78] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 89.37 [86.29,91.96] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 100 | crg | libero_10 | full | COMPLETE | 500 | 76.40 [72.43,80.06] | 500 | 87.85 | 0.02442 | 2.44184 | 8.09 |
| DP | 100 | crg | libero_10 | sensitivity | COMPLETE | 479 | 77.45 [73.44,81.12] | 479 | 87.94 | 0.02425 | 2.42461 | 7.95 |
| DP | 100 | crg | libero_spatial | full | COMPLETE | 500 | 90.20 [87.25,92.66] | 500 | 88.33 | 0.02375 | 2.37508 | 2.70 |
| DP | 100 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 90.18 [87.20,92.67] | 489 | 88.35 | 0.02371 | 2.37145 | 2.70 |
| DP | 200 | c | libero_10 | full | COMPLETE | 500 | 48.80 [44.34,53.28] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | c | libero_10 | sensitivity | COMPLETE | 479 | 49.27 [44.70,53.84] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | c | libero_spatial | full | COMPLETE | 500 | 88.40 [85.26,91.07] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | c | libero_spatial | sensitivity | COMPLETE | 489 | 88.34 [85.16,91.05] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | cg | libero_10 | full | COMPLETE | 500 | 83.80 [80.27,86.92] | 500 | 87.15 | 0.02583 | 2.58322 | 8.16 |
| DP | 200 | cg | libero_10 | sensitivity | COMPLETE | 479 | 84.97 [81.45,88.05] | 479 | 87.39 | 0.02535 | 2.53531 | 7.91 |
| DP | 200 | cg | libero_spatial | full | COMPLETE | 500 | 92.20 [89.49,94.39] | 500 | 90.32 | 0.01969 | 1.96887 | 2.23 |
| DP | 200 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 92.02 [89.26,94.27] | 489 | 90.31 | 0.01969 | 1.96928 | 2.23 |
| DP | 200 | cr | libero_10 | full | COMPLETE | 500 | 51.00 [46.52,55.46] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | cr | libero_10 | sensitivity | COMPLETE | 479 | 51.57 [46.99,56.12] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | cr | libero_spatial | full | COMPLETE | 500 | 93.40 [90.86,95.41] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 93.25 [90.65,95.31] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 200 | crg | libero_10 | full | COMPLETE | 500 | 80.40 [76.64,83.79] | 500 | 86.81 | 0.02651 | 2.65127 | 8.40 |
| DP | 200 | crg | libero_10 | sensitivity | COMPLETE | 479 | 81.00 [77.20,84.42] | 479 | 86.82 | 0.02650 | 2.65038 | 8.34 |
| DP | 200 | crg | libero_spatial | full | COMPLETE | 500 | 93.40 [90.86,95.41] | 500 | 90.93 | 0.01845 | 1.84520 | 2.04 |
| DP | 200 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 93.25 [90.65,95.31] | 489 | 90.89 | 0.01854 | 1.85431 | 2.06 |
| DP | 300 | c | libero_10 | full | COMPLETE | 500 | 57.80 [53.34,62.17] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | c | libero_10 | sensitivity | COMPLETE | 479 | 58.66 [54.11,63.11] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | c | libero_spatial | full | COMPLETE | 500 | 91.80 [89.04,94.05] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | c | libero_spatial | sensitivity | COMPLETE | 489 | 92.02 [89.26,94.27] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | cg | libero_10 | full | COMPLETE | 500 | 82.20 [78.56,85.45] | 500 | 88.08 | 0.02397 | 2.39684 | 7.50 |
| DP | 300 | cg | libero_10 | sensitivity | COMPLETE | 479 | 82.67 [78.98,85.96] | 479 | 87.91 | 0.02432 | 2.43215 | 7.56 |
| DP | 300 | cg | libero_spatial | full | COMPLETE | 500 | 92.80 [90.17,94.91] | 500 | 91.34 | 0.01761 | 1.76094 | 1.96 |
| DP | 300 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 92.84 [90.19,94.96] | 489 | 91.44 | 0.01741 | 1.74103 | 1.94 |
| DP | 300 | cr | libero_10 | full | COMPLETE | 500 | 60.60 [56.16,64.91] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | cr | libero_10 | sensitivity | COMPLETE | 479 | 61.59 [57.07,65.96] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | cr | libero_spatial | full | COMPLETE | 500 | 94.40 [92.01,96.25] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 94.48 [92.07,96.33] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 300 | crg | libero_10 | full | COMPLETE | 500 | 84.80 [81.35,87.83] | 500 | 88.38 | 0.02338 | 2.33840 | 7.14 |
| DP | 300 | crg | libero_10 | sensitivity | COMPLETE | 479 | 85.59 [82.13,88.62] | 479 | 88.34 | 0.02346 | 2.34611 | 7.10 |
| DP | 300 | crg | libero_spatial | full | COMPLETE | 500 | 93.00 [90.40,95.08] | 500 | 91.59 | 0.01711 | 1.71106 | 1.89 |
| DP | 300 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 92.84 [90.19,94.96] | 489 | 91.56 | 0.01717 | 1.71730 | 1.90 |
| DP | 400 | c | libero_10 | full | COMPLETE | 500 | 61.20 [56.77,65.49] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | c | libero_10 | sensitivity | COMPLETE | 479 | 62.21 [57.70,66.57] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | c | libero_spatial | full | COMPLETE | 500 | 93.00 [90.40,95.08] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | c | libero_spatial | sensitivity | COMPLETE | 489 | 93.05 [90.42,95.14] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | cg | libero_10 | full | COMPLETE | 500 | 85.80 [82.43,88.74] | 500 | 88.98 | 0.02217 | 2.21685 | 6.75 |
| DP | 400 | cg | libero_10 | sensitivity | COMPLETE | 479 | 86.43 [83.03,89.37] | 479 | 88.96 | 0.02221 | 2.22100 | 6.72 |
| DP | 400 | cg | libero_spatial | full | COMPLETE | 500 | 92.00 [89.27,94.22] | 500 | 91.76 | 0.01675 | 1.67455 | 1.88 |
| DP | 400 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 92.23 [89.49,94.44] | 489 | 91.91 | 0.01646 | 1.64621 | 1.84 |
| DP | 400 | cr | libero_10 | full | COMPLETE | 500 | 63.60 [59.21,67.83] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | cr | libero_10 | sensitivity | COMPLETE | 479 | 64.93 [60.47,69.20] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | cr | libero_spatial | full | COMPLETE | 500 | 94.80 [92.47,96.58] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 94.89 [92.55,96.66] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 400 | crg | libero_10 | full | COMPLETE | 500 | 87.60 [84.39,90.36] | 500 | 89.15 | 0.02184 | 2.18398 | 6.50 |
| DP | 400 | crg | libero_10 | sensitivity | COMPLETE | 479 | 88.31 [85.09,91.05] | 479 | 89.12 | 0.02190 | 2.19046 | 6.47 |
| DP | 400 | crg | libero_spatial | full | COMPLETE | 500 | 93.80 [91.31,95.75] | 500 | 92.06 | 0.01616 | 1.61623 | 1.77 |
| DP | 400 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 93.87 [91.36,95.82] | 489 | 92.24 | 0.01579 | 1.57939 | 1.73 |
| DP | 500 | c | libero_10 | full | COMPLETE | 500 | 64.80 [60.44,68.99] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | c | libero_10 | sensitivity | COMPLETE | 479 | 65.97 [61.53,70.21] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | c | libero_spatial | full | COMPLETE | 500 | 94.60 [92.24,96.41] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | c | libero_spatial | sensitivity | COMPLETE | 489 | 94.48 [92.07,96.33] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | cg | libero_10 | full | COMPLETE | 500 | 87.40 [84.17,90.18] | 500 | 89.57 | 0.02099 | 2.09870 | 6.32 |
| DP | 500 | cg | libero_10 | sensitivity | COMPLETE | 479 | 87.27 [83.94,90.12] | 479 | 89.51 | 0.02111 | 2.11119 | 6.33 |
| DP | 500 | cg | libero_spatial | full | COMPLETE | 500 | 93.00 [90.40,95.08] | 500 | 92.03 | 0.01621 | 1.62066 | 1.80 |
| DP | 500 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 93.05 [90.42,95.14] | 489 | 92.10 | 0.01605 | 1.60527 | 1.78 |
| DP | 500 | cr | libero_10 | full | COMPLETE | 500 | 68.20 [63.92,72.26] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | cr | libero_10 | sensitivity | COMPLETE | 479 | 69.31 [64.97,73.42] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | cr | libero_spatial | full | COMPLETE | 500 | 93.60 [91.08,95.58] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 93.46 [90.89,95.48] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | crg | libero_10 | full | COMPLETE | 500 | 88.80 [85.70,91.43] | 500 | 90.32 | 0.01948 | 1.94832 | 5.73 |
| DP | 500 | crg | libero_10 | sensitivity | COMPLETE | 479 | 88.94 [85.78,91.60] | 479 | 90.27 | 0.01957 | 1.95738 | 5.73 |
| DP | 500 | crg | libero_spatial | full | COMPLETE | 500 | 94.00 [91.55,95.92] | 500 | 92.36 | 0.01555 | 1.55527 | 1.70 |
| DP | 500 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 94.07 [91.59,95.99] | 489 | 92.46 | 0.01534 | 1.53444 | 1.68 |
| Fast-WAM | 50 | c | libero_10 | full | COMPLETE | 500 | 39.00 [34.70,43.43] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | c | libero_10 | sensitivity | COMPLETE | 479 | 39.87 [35.46,44.42] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | c | libero_spatial | full | COMPLETE | 500 | 78.60 [74.74,82.12] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | c | libero_spatial | sensitivity | COMPLETE | 489 | 78.53 [74.62,82.09] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | cg | libero_10 | full | COMPLETE | 500 | 66.20 [61.87,70.34] | 500 | 80.97 | 0.03819 | 0.38194 | 13.25 |
| Fast-WAM | 50 | cg | libero_10 | sensitivity | COMPLETE | 479 | 66.18 [61.75,70.41] | 479 | 80.89 | 0.03836 | 0.38363 | 13.28 |
| Fast-WAM | 50 | cg | libero_spatial | full | COMPLETE | 500 | 84.20 [80.70,87.29] | 500 | 92.20 | 0.01581 | 0.15809 | 1.98 |
| Fast-WAM | 50 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 84.25 [80.72,87.37] | 489 | 92.10 | 0.01601 | 0.16007 | 2.00 |
| Fast-WAM | 50 | cr | libero_10 | full | COMPLETE | 500 | 45.00 [40.58,49.48] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | cr | libero_10 | sensitivity | COMPLETE | 479 | 45.72 [41.19,50.30] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | cr | libero_spatial | full | COMPLETE | 500 | 77.00 [73.06,80.62] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 76.69 [72.68,80.37] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | crg | libero_10 | full | COMPLETE | 500 | 71.80 [67.63,75.71] | 500 | 81.78 | 0.03660 | 0.36596 | 12.19 |
| Fast-WAM | 50 | crg | libero_10 | sensitivity | COMPLETE | 479 | 72.44 [68.21,76.40] | 479 | 81.76 | 0.03663 | 0.36634 | 12.12 |
| Fast-WAM | 50 | crg | libero_spatial | full | COMPLETE | 500 | 85.60 [82.21,88.56] | 500 | 91.23 | 0.01778 | 0.17777 | 2.19 |
| Fast-WAM | 50 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 85.69 [82.26,88.67] | 489 | 91.12 | 0.01799 | 0.17993 | 2.21 |
| Fast-WAM | 100 | c | libero_10 | full | COMPLETE | 500 | 51.00 [46.52,55.46] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | c | libero_10 | sensitivity | COMPLETE | 479 | 51.77 [47.20,56.33] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | c | libero_spatial | full | COMPLETE | 500 | 83.20 [79.63,86.37] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | c | libero_spatial | sensitivity | COMPLETE | 489 | 83.03 [79.40,86.25] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | cg | libero_10 | full | COMPLETE | 500 | 71.80 [67.63,75.71] | 500 | 85.42 | 0.02928 | 0.29277 | 9.81 |
| Fast-WAM | 100 | cg | libero_10 | sensitivity | COMPLETE | 479 | 71.61 [67.34,75.61] | 479 | 85.43 | 0.02926 | 0.29259 | 9.78 |
| Fast-WAM | 100 | cg | libero_spatial | full | COMPLETE | 500 | 87.60 [84.39,90.36] | 500 | 93.14 | 0.01392 | 0.13917 | 1.69 |
| Fast-WAM | 100 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 87.73 [84.49,90.50] | 489 | 93.00 | 0.01419 | 0.14192 | 1.72 |
| Fast-WAM | 100 | cr | libero_10 | full | COMPLETE | 500 | 53.80 [49.32,58.24] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | cr | libero_10 | sensitivity | COMPLETE | 479 | 54.28 [49.70,58.81] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | cr | libero_spatial | full | COMPLETE | 500 | 83.40 [79.84,86.56] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 83.64 [80.06,86.81] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 100 | crg | libero_10 | full | COMPLETE | 500 | 72.20 [68.05,76.09] | 500 | 86.35 | 0.02742 | 0.27417 | 9.08 |
| Fast-WAM | 100 | crg | libero_10 | sensitivity | COMPLETE | 479 | 72.65 [68.42,76.60] | 479 | 86.43 | 0.02727 | 0.27269 | 8.98 |
| Fast-WAM | 100 | crg | libero_spatial | full | COMPLETE | 500 | 87.20 [83.95,90.00] | 500 | 93.13 | 0.01393 | 0.13933 | 1.72 |
| Fast-WAM | 100 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 87.53 [84.27,90.32] | 489 | 92.98 | 0.01423 | 0.14232 | 1.75 |
| Fast-WAM | 200 | c | libero_10 | full | COMPLETE | 500 | 56.60 [52.13,60.99] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | c | libero_10 | sensitivity | COMPLETE | 479 | 56.58 [52.00,61.07] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | c | libero_spatial | full | COMPLETE | 500 | 84.40 [80.92,87.47] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | c | libero_spatial | sensitivity | COMPLETE | 489 | 85.07 [81.60,88.11] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | cg | libero_10 | full | COMPLETE | 500 | 74.80 [70.75,78.55] | 500 | 83.77 | 0.03261 | 0.32607 | 10.42 |
| Fast-WAM | 200 | cg | libero_10 | sensitivity | COMPLETE | 479 | 74.95 [70.82,78.77] | 479 | 83.80 | 0.03256 | 0.32564 | 10.37 |
| Fast-WAM | 200 | cg | libero_spatial | full | COMPLETE | 500 | 86.20 [82.86,89.10] | 500 | 90.54 | 0.01918 | 0.19178 | 2.35 |
| Fast-WAM | 200 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 86.91 [83.60,89.77] | 489 | 90.60 | 0.01907 | 0.19072 | 2.32 |
| Fast-WAM | 200 | cr | libero_10 | full | COMPLETE | 500 | 63.00 [58.60,67.24] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | cr | libero_10 | sensitivity | COMPLETE | 479 | 63.47 [58.98,67.79] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | cr | libero_spatial | full | COMPLETE | 500 | 86.40 [83.08,89.28] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 87.12 [83.82,89.96] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 200 | crg | libero_10 | full | COMPLETE | 500 | 79.80 [76.01,83.23] | 500 | 84.78 | 0.03060 | 0.30597 | 9.49 |
| Fast-WAM | 200 | crg | libero_10 | sensitivity | COMPLETE | 479 | 80.17 [76.31,83.65] | 479 | 84.87 | 0.03042 | 0.30416 | 9.38 |
| Fast-WAM | 200 | crg | libero_spatial | full | COMPLETE | 500 | 87.80 [84.61,90.54] | 500 | 91.06 | 0.01814 | 0.18137 | 2.20 |
| Fast-WAM | 200 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 88.34 [85.16,91.05] | 489 | 91.19 | 0.01786 | 0.17865 | 2.16 |
| Fast-WAM | 300 | c | libero_10 | full | COMPLETE | 500 | 62.60 [58.19,66.86] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | c | libero_10 | sensitivity | COMPLETE | 479 | 62.84 [58.34,67.18] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | c | libero_spatial | full | COMPLETE | 500 | 84.80 [81.35,87.83] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | c | libero_spatial | sensitivity | COMPLETE | 489 | 85.28 [81.82,88.30] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | cg | libero_10 | full | COMPLETE | 500 | 77.20 [73.27,80.81] | 500 | 84.07 | 0.03201 | 0.32012 | 10.12 |
| Fast-WAM | 300 | cg | libero_10 | sensitivity | COMPLETE | 479 | 77.45 [73.44,81.12] | 479 | 84.12 | 0.03192 | 0.31923 | 10.06 |
| Fast-WAM | 300 | cg | libero_spatial | full | COMPLETE | 500 | 87.00 [83.73,89.82] | 500 | 90.83 | 0.01860 | 0.18602 | 2.25 |
| Fast-WAM | 300 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 87.73 [84.49,90.50] | 489 | 91.02 | 0.01822 | 0.18219 | 2.19 |
| Fast-WAM | 300 | cr | libero_10 | full | COMPLETE | 500 | 67.20 [62.89,71.30] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | cr | libero_10 | sensitivity | COMPLETE | 479 | 67.64 [63.25,71.81] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | cr | libero_spatial | full | COMPLETE | 500 | 87.40 [84.17,90.18] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 87.93 [84.71,90.69] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 300 | crg | libero_10 | full | COMPLETE | 500 | 81.20 [77.49,84.53] | 500 | 85.58 | 0.02901 | 0.29007 | 8.83 |
| Fast-WAM | 300 | crg | libero_10 | sensitivity | COMPLETE | 479 | 81.42 [77.64,84.80] | 479 | 85.61 | 0.02893 | 0.28929 | 8.76 |
| Fast-WAM | 300 | crg | libero_spatial | full | COMPLETE | 500 | 89.20 [86.14,91.78] | 500 | 91.97 | 0.01629 | 0.16287 | 1.93 |
| Fast-WAM | 300 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 89.98 [86.97,92.49] | 489 | 92.24 | 0.01574 | 0.15739 | 1.85 |
| Fast-WAM | 400 | c | libero_10 | full | COMPLETE | 500 | 64.80 [60.44,68.99] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | c | libero_10 | sensitivity | COMPLETE | 479 | 65.14 [60.68,69.40] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | c | libero_spatial | full | COMPLETE | 500 | 85.40 [82.00,88.38] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | c | libero_spatial | sensitivity | COMPLETE | 489 | 85.89 [82.48,88.85] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | cg | libero_10 | full | COMPLETE | 500 | 79.60 [75.80,83.05] | 500 | 84.72 | 0.03072 | 0.30722 | 9.53 |
| Fast-WAM | 400 | cg | libero_10 | sensitivity | COMPLETE | 479 | 79.54 [75.65,83.07] | 479 | 84.73 | 0.03070 | 0.30704 | 9.51 |
| Fast-WAM | 400 | cg | libero_spatial | full | COMPLETE | 500 | 88.20 [85.04,90.89] | 500 | 90.67 | 0.01892 | 0.18920 | 2.29 |
| Fast-WAM | 400 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 88.96 [85.84,91.59] | 489 | 90.87 | 0.01852 | 0.18515 | 2.22 |
| Fast-WAM | 400 | cr | libero_10 | full | COMPLETE | 500 | 71.20 [67.01,75.13] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | cr | libero_10 | sensitivity | COMPLETE | 479 | 71.82 [67.56,75.80] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | cr | libero_spatial | full | COMPLETE | 500 | 87.60 [84.39,90.36] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 88.34 [85.16,91.05] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 400 | crg | libero_10 | full | COMPLETE | 500 | 84.80 [81.35,87.83] | 500 | 86.88 | 0.02639 | 0.26391 | 7.79 |
| Fast-WAM | 400 | crg | libero_10 | sensitivity | COMPLETE | 479 | 85.18 [81.67,88.24] | 479 | 86.98 | 0.02619 | 0.26192 | 7.69 |
| Fast-WAM | 400 | crg | libero_spatial | full | COMPLETE | 500 | 88.80 [85.70,91.43] | 500 | 91.35 | 0.01755 | 0.17554 | 2.09 |
| Fast-WAM | 400 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 89.57 [86.52,92.14] | 489 | 91.72 | 0.01681 | 0.16808 | 1.99 |
| Fast-WAM | 500 | c | libero_10 | full | COMPLETE | 500 | 69.00 [64.74,73.03] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | c | libero_10 | sensitivity | COMPLETE | 479 | 69.31 [64.97,73.42] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | c | libero_spatial | full | COMPLETE | 500 | 87.80 [84.61,90.54] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | c | libero_spatial | sensitivity | COMPLETE | 489 | 88.55 [85.39,91.23] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | cg | libero_10 | full | COMPLETE | 500 | 81.40 [77.71,84.72] | 500 | 85.24 | 0.02968 | 0.29684 | 9.09 |
| Fast-WAM | 500 | cg | libero_10 | sensitivity | COMPLETE | 479 | 81.42 [77.64,84.80] | 479 | 85.22 | 0.02972 | 0.29719 | 9.09 |
| Fast-WAM | 500 | cg | libero_spatial | full | COMPLETE | 500 | 89.20 [86.14,91.78] | 500 | 91.62 | 0.01701 | 0.17012 | 2.02 |
| Fast-WAM | 500 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 89.98 [86.97,92.49] | 489 | 91.92 | 0.01641 | 0.16405 | 1.94 |
| Fast-WAM | 500 | cr | libero_10 | full | COMPLETE | 500 | 70.40 [66.19,74.37] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | cr | libero_10 | sensitivity | COMPLETE | 479 | 71.40 [67.12,75.41] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | cr | libero_spatial | full | COMPLETE | 500 | 88.40 [85.26,91.07] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 88.96 [85.84,91.59] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | crg | libero_10 | full | COMPLETE | 500 | 83.40 [79.84,86.56] | 500 | 86.18 | 0.02780 | 0.27800 | 8.33 |
| Fast-WAM | 500 | crg | libero_10 | sensitivity | COMPLETE | 479 | 83.51 [79.87,86.72] | 479 | 86.22 | 0.02772 | 0.27716 | 8.29 |
| Fast-WAM | 500 | crg | libero_spatial | full | COMPLETE | 500 | 90.20 [87.25,92.66] | 500 | 91.40 | 0.01745 | 0.17454 | 2.07 |
| Fast-WAM | 500 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 90.80 [87.88,93.21] | 489 | 91.61 | 0.01705 | 0.17046 | 2.01 |

## Paired comparisons (Δ/LB in pp)

| model/N/group/suite | cohort | reference | family | status | pairs | cost pairs | Δ | g/l | CP LB family | boot LB family | CP/boot LB pointwise | NI |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|
| Cosmos/50/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -86.60 | 0/433 | — | — | -89.46/-88.60 | 描述性 |
| Cosmos/50/c/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -63.20 | 2/318 | -70.40 | -69.20 | -67.78/-66.40 | 未证明 |
| Cosmos/50/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -86.22 | 0/413 | — | — | -89.18/-88.31 | 描述性 |
| Cosmos/50/c/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -63.26 | 2/305 | -70.61 | -69.52 | -67.94/-66.60 | 描述性 |
| Cosmos/50/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -85.20 | 1/427 | — | — | -88.37/-87.00 | 描述性 |
| Cosmos/50/c/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -83.80 | 2/421 | -89.03 | -87.60 | -87.24/-85.80 | 未证明 |
| Cosmos/50/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -85.28 | 1/418 | — | — | -88.48/-87.12 | 描述性 |
| Cosmos/50/c/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -83.64 | 2/411 | -88.95 | -87.32 | -87.13/-85.69 | 描述性 |
| Cosmos/50/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -76.20 | 1/382 | — | — | -80.05/-79.00 | 描述性 |
| Cosmos/50/cg/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -57.60 | 13/301 | -66.29 | -64.40 | -63.13/-61.40 | 未证明 |
| Cosmos/50/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 10.40 | 56/4 | — | — | 6.54/8.00 | 描述性 |
| Cosmos/50/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -75.37 | 1/362 | — | — | -79.35/-78.29 | 描述性 |
| Cosmos/50/cg/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -57.20 | 13/287 | -66.13 | -64.51 | -62.88/-61.17 | 描述性 |
| Cosmos/50/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 10.86 | 56/4 | — | — | 6.83/8.56 | 描述性 |
| Cosmos/50/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -70.80 | 2/356 | — | — | -75.09/-73.60 | 描述性 |
| Cosmos/50/cg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -70.20 | 3/354 | -77.09 | -75.40 | -74.63/-73.00 | 未证明 |
| Cosmos/50/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 14.40 | 76/4 | — | — | 10.13/12.00 | 描述性 |
| Cosmos/50/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -70.55 | 2/347 | — | — | -74.90/-73.42 | 描述性 |
| Cosmos/50/cg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -69.73 | 3/344 | -76.74 | -74.85 | -74.24/-72.60 | 描述性 |
| Cosmos/50/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 14.72 | 76/4 | — | — | 10.36/12.27 | 描述性 |
| Cosmos/50/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -68.80 | 0/344 | — | — | -72.84/-71.60 | 描述性 |
| Cosmos/50/cr/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -50.00 | 8/258 | -58.48 | -57.20 | -55.37/-53.80 | 未证明 |
| Cosmos/50/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 17.80 | 105/16 | — | — | 12.37/14.60 | 描述性 |
| Cosmos/50/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -68.27 | 0/327 | — | — | -72.42/-71.19 | 描述性 |
| Cosmos/50/cr/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -49.69 | 8/246 | -58.38 | -56.78 | -55.19/-53.44 | 描述性 |
| Cosmos/50/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 17.95 | 102/16 | — | — | 12.34/14.61 | 描述性 |
| Cosmos/50/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -59.60 | 4/302 | — | — | -64.50/-62.20 | 描述性 |
| Cosmos/50/cr/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -58.60 | 2/295 | -66.02 | -63.50 | -63.30/-61.20 | 未证明 |
| Cosmos/50/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 25.60 | 138/10 | — | — | 20.08/23.00 | 描述性 |
| Cosmos/50/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -59.51 | 4/295 | — | — | -64.47/-62.17 | 描述性 |
| Cosmos/50/cr/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -58.28 | 2/287 | -65.80 | -63.60 | -63.04/-60.94 | 描述性 |
| Cosmos/50/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 25.77 | 136/10 | — | — | 20.15/23.11 | 描述性 |
| Cosmos/50/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -60.20 | 1/302 | — | — | -64.71/-63.20 | 描述性 |
| Cosmos/50/crg/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -47.20 | 14/250 | -56.28 | -54.20 | -52.93/-51.00 | 未证明 |
| Cosmos/50/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 16.00 | 121/41 | — | — | 9.55/12.20 | 描述性 |
| Cosmos/50/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 8.60 | 57/14 | — | — | 4.10/6.00 | 描述性 |
| Cosmos/50/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -59.71 | 0/286 | — | — | -64.13/-62.84 | 描述性 |
| Cosmos/50/crg/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -46.56 | 13/236 | -55.79 | -53.44 | -52.39/-50.52 | 描述性 |
| Cosmos/50/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 15.66 | 116/41 | — | — | 9.01/11.69 | 描述性 |
| Cosmos/50/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 8.56 | 54/13 | — | — | 3.99/5.85 | 描述性 |
| Cosmos/50/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -51.20 | 6/262 | — | — | -56.41/-54.00 | 描述性 |
| Cosmos/50/crg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -50.60 | 4/257 | -58.57 | -55.50 | -55.64/-53.40 | 未证明 |
| Cosmos/50/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 19.60 | 126/28 | — | — | 13.46/16.60 | 描述性 |
| Cosmos/50/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 8.40 | 53/11 | — | — | 4.14/6.00 | 描述性 |
| Cosmos/50/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -50.92 | 6/255 | — | — | -56.20/-53.78 | 描述性 |
| Cosmos/50/crg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -50.10 | 4/249 | -58.18 | -55.32 | -55.21/-52.97 | 描述性 |
| Cosmos/50/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 19.63 | 124/28 | — | — | 13.39/16.56 | 描述性 |
| Cosmos/50/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 8.59 | 53/11 | — | — | 4.24/6.13 | 描述性 |
| Cosmos/100/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -53.60 | 1/269 | — | — | -58.23/-56.60 | 描述性 |
| Cosmos/100/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -30.20 | 19/170 | — | — | -36.03/-34.20 | 描述性 |
| Cosmos/100/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -53.24 | 1/256 | — | — | -57.98/-56.37 | 描述性 |
| Cosmos/100/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -30.27 | 17/162 | — | — | -36.17/-34.03 | 描述性 |
| Cosmos/100/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -10.80 | 13/67 | — | — | -15.31/-13.40 | 描述性 |
| Cosmos/100/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -9.40 | 13/60 | — | — | -13.78/-12.00 | 描述性 |
| Cosmos/100/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -10.84 | 13/66 | — | — | -15.43/-13.50 | 描述性 |
| Cosmos/100/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -9.20 | 13/58 | — | — | -13.64/-11.66 | 描述性 |
| Cosmos/100/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -44.60 | 1/224 | — | — | -49.27/-47.80 | 描述性 |
| Cosmos/100/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -26.00 | 27/157 | — | — | -32.08/-30.00 | 描述性 |
| Cosmos/100/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 9.00 | 64/19 | — | — | 4.13/6.40 | 描述性 |
| Cosmos/100/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -44.47 | 1/214 | — | — | -49.25/-47.81 | 描述性 |
| Cosmos/100/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -26.30 | 25/151 | — | — | -32.49/-30.27 | 描述性 |
| Cosmos/100/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 8.77 | 60/18 | — | — | 3.82/6.05 | 描述性 |
| Cosmos/100/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -9.80 | 15/64 | — | — | -14.36/-12.40 | 描述性 |
| Cosmos/100/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -9.20 | 9/55 | — | — | -13.25/-11.60 | 描述性 |
| Cosmos/100/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.00 | 19/14 | — | — | -2.35/-0.80 | 描述性 |
| Cosmos/100/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -10.02 | 14/63 | — | — | -14.61/-12.68 | 描述性 |
| Cosmos/100/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -9.20 | 9/54 | — | — | -13.32/-11.66 | 描述性 |
| Cosmos/100/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.82 | 18/14 | — | — | -2.56/-1.02 | 描述性 |
| Cosmos/100/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -38.40 | 1/193 | — | — | -43.02/-41.40 | 描述性 |
| Cosmos/100/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -19.60 | 24/122 | — | — | -25.31/-23.20 | 描述性 |
| Cosmos/100/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 15.20 | 99/23 | — | — | 9.57/12.00 | 描述性 |
| Cosmos/100/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -38.20 | 1/184 | — | — | -42.93/-41.34 | 描述性 |
| Cosmos/100/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -19.62 | 22/116 | — | — | -25.41/-23.38 | 描述性 |
| Cosmos/100/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 15.03 | 94/22 | — | — | 9.29/11.69 | 描述性 |
| Cosmos/100/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -5.80 | 14/43 | — | — | -9.87/-8.20 | 描述性 |
| Cosmos/100/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -4.80 | 15/39 | — | — | -8.82/-7.00 | 描述性 |
| Cosmos/100/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.00 | 40/15 | — | — | 0.88/2.60 | 描述性 |
| Cosmos/100/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -5.93 | 13/42 | — | — | -10.01/-8.38 | 描述性 |
| Cosmos/100/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -4.70 | 15/38 | — | — | -8.78/-6.95 | 描述性 |
| Cosmos/100/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 4.91 | 39/15 | — | — | 0.72/2.45 | 描述性 |
| Cosmos/100/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -33.40 | 1/168 | — | — | -37.92/-36.60 | 描述性 |
| Cosmos/100/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -20.40 | 23/125 | — | — | -26.10/-24.00 | 描述性 |
| Cosmos/100/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 11.20 | 93/37 | — | — | 5.23/7.60 | 描述性 |
| Cosmos/100/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 5.00 | 37/12 | — | — | 1.11/2.80 | 描述性 |
| Cosmos/100/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -33.19 | 1/160 | — | — | -37.82/-36.53 | 描述性 |
| Cosmos/100/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -20.04 | 22/118 | — | — | -25.85/-23.59 | 描述性 |
| Cosmos/100/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 11.27 | 89/35 | — | — | 5.18/7.72 | 描述性 |
| Cosmos/100/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 5.01 | 36/12 | — | — | 0.98/2.71 | 描述性 |
| Cosmos/100/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -4.60 | 14/37 | — | — | -8.52/-6.80 | 描述性 |
| Cosmos/100/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -4.00 | 13/33 | — | — | -7.75/-6.20 | 描述性 |
| Cosmos/100/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.20 | 41/15 | — | — | 1.05/2.80 | 描述性 |
| Cosmos/100/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.20 | 17/11 | — | — | -1.91/-0.40 | 描述性 |
| Cosmos/100/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -4.70 | 13/36 | — | — | -8.62/-6.95 | 描述性 |
| Cosmos/100/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -3.89 | 13/32 | — | — | -7.69/-6.13 | 描述性 |
| Cosmos/100/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 5.32 | 40/14 | — | — | 1.15/2.86 | 描述性 |
| Cosmos/100/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 1.23 | 16/10 | — | — | -1.85/-0.41 | 描述性 |
| Cosmos/200/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -32.60 | 2/165 | — | — | -37.26/-35.60 | 描述性 |
| Cosmos/200/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -9.20 | 33/79 | — | — | -14.71/-12.60 | 描述性 |
| Cosmos/200/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -31.94 | 2/155 | — | — | -36.70/-35.07 | 描述性 |
| Cosmos/200/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -8.98 | 30/73 | — | — | -14.51/-12.32 | 描述性 |
| Cosmos/200/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -2.80 | 14/28 | — | — | -6.45/-4.80 | 描述性 |
| Cosmos/200/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.40 | 14/21 | — | — | -4.81/-3.20 | 描述性 |
| Cosmos/200/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.07 | 13/28 | — | — | -6.75/-5.11 | 描述性 |
| Cosmos/200/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.43 | 14/21 | — | — | -4.92/-3.48 | 描述性 |
| Cosmos/200/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -24.20 | 3/124 | — | — | -28.70/-27.20 | 描述性 |
| Cosmos/200/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -5.60 | 46/74 | — | — | -11.41/-9.20 | 描述性 |
| Cosmos/200/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 8.40 | 51/9 | — | — | 4.30/6.00 | 描述性 |
| Cosmos/200/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -24.01 | 3/118 | — | — | -28.62/-27.14 | 描述性 |
| Cosmos/200/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -5.85 | 42/70 | — | — | -11.71/-9.60 | 描述性 |
| Cosmos/200/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 7.93 | 47/9 | — | — | 3.76/5.64 | 描述性 |
| Cosmos/200/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.60 | 14/32 | — | — | -7.38/-5.80 | 描述性 |
| Cosmos/200/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -3.00 | 11/26 | — | — | -6.42/-5.00 | 描述性 |
| Cosmos/200/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -0.80 | 3/7 | — | — | -2.74/-1.80 | 描述性 |
| Cosmos/200/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.89 | 13/32 | — | — | -7.69/-6.13 | 描述性 |
| Cosmos/200/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -3.07 | 11/26 | — | — | -6.57/-5.11 | 描述性 |
| Cosmos/200/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -0.82 | 3/7 | — | — | -2.80/-1.84 | 描述性 |
| Cosmos/200/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -27.20 | 5/141 | — | — | -32.04/-30.20 | 描述性 |
| Cosmos/200/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -8.40 | 34/76 | — | — | -13.90/-11.60 | 描述性 |
| Cosmos/200/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.40 | 60/33 | — | — | 0.14/2.40 | 描述性 |
| Cosmos/200/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -26.93 | 4/133 | — | — | -31.78/-30.06 | 描述性 |
| Cosmos/200/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -8.35 | 31/71 | — | — | -13.89/-11.69 | 描述性 |
| Cosmos/200/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 5.01 | 56/32 | — | — | -0.35/1.88 | 描述性 |
| Cosmos/200/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -0.40 | 14/16 | — | — | -3.61/-2.20 | 描述性 |
| Cosmos/200/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | 0.60 | 13/10 | — | — | -2.26/-1.00 | 描述性 |
| Cosmos/200/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.40 | 20/8 | — | — | -0.67/0.60 | 描述性 |
| Cosmos/200/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -0.61 | 13/16 | — | — | -3.84/-2.45 | 描述性 |
| Cosmos/200/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | 0.61 | 13/10 | — | — | -2.31/-1.02 | 描述性 |
| Cosmos/200/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.45 | 20/8 | — | — | -0.68/0.82 | 描述性 |
| Cosmos/200/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -22.20 | 3/114 | — | — | -26.61/-25.01 | 描述性 |
| Cosmos/200/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -9.20 | 35/81 | — | — | -14.80/-12.60 | 描述性 |
| Cosmos/200/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.00 | 49/39 | — | — | -3.17/-1.00 | 描述性 |
| Cosmos/200/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 5.00 | 36/11 | — | — | 1.19/2.80 | 描述性 |
| Cosmos/200/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -21.92 | 3/108 | — | — | -26.43/-24.84 | 描述性 |
| Cosmos/200/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -8.77 | 33/75 | — | — | -14.44/-12.11 | 描述性 |
| Cosmos/200/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.09 | 47/37 | — | — | -3.19/-1.04 | 描述性 |
| Cosmos/200/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 5.01 | 34/10 | — | — | 1.16/2.92 | 描述性 |
| Cosmos/200/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -1.60 | 14/22 | — | — | -5.05/-3.60 | 描述性 |
| Cosmos/200/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.00 | 11/16 | — | — | -4.04/-2.60 | 描述性 |
| Cosmos/200/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.00 | 21/11 | — | — | -1.28/0.20 | 描述性 |
| Cosmos/200/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -1.20 | 3/9 | — | — | -3.27/-2.40 | 描述性 |
| Cosmos/200/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -1.84 | 13/22 | — | — | -5.31/-3.68 | 描述性 |
| Cosmos/200/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.02 | 11/16 | — | — | -4.13/-2.86 | 描述性 |
| Cosmos/200/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.04 | 21/11 | — | — | -1.31/0.20 | 描述性 |
| Cosmos/200/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -1.23 | 3/9 | — | — | -3.34/-2.45 | 描述性 |
| Cosmos/300/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -28.00 | 3/143 | — | — | -32.66/-31.00 | 描述性 |
| Cosmos/300/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -4.60 | 36/59 | — | — | -9.86/-7.60 | 描述性 |
| Cosmos/300/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -27.35 | 3/134 | — | — | -32.10/-30.48 | 描述性 |
| Cosmos/300/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -4.38 | 34/55 | — | — | -9.72/-7.52 | 描述性 |
| Cosmos/300/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -1.20 | 13/19 | — | — | -4.48/-3.00 | 描述性 |
| Cosmos/300/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | 0.20 | 13/12 | — | — | -2.76/-1.40 | 描述性 |
| Cosmos/300/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -1.23 | 12/18 | — | — | -4.48/-3.07 | 描述性 |
| Cosmos/300/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | 0.41 | 13/11 | — | — | -2.57/-1.23 | 描述性 |
| Cosmos/300/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -21.60 | 4/112 | — | — | -26.09/-24.60 | 描述性 |
| Cosmos/300/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -3.00 | 46/61 | — | — | -8.58/-6.40 | 描述性 |
| Cosmos/300/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 6.40 | 46/14 | — | — | 2.16/4.00 | 描述性 |
| Cosmos/300/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -21.29 | 3/105 | — | — | -25.77/-24.22 | 描述性 |
| Cosmos/300/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -3.13 | 42/57 | — | — | -8.75/-6.47 | 描述性 |
| Cosmos/300/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 6.05 | 42/13 | — | — | 1.80/3.55 | 描述性 |
| Cosmos/300/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -1.80 | 13/22 | — | — | -5.19/-3.60 | 描述性 |
| Cosmos/300/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.20 | 9/15 | — | — | -4.07/-2.80 | 描述性 |
| Cosmos/300/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -0.60 | 3/6 | — | — | -2.47/-1.60 | 描述性 |
| Cosmos/300/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -2.04 | 12/22 | — | — | -5.46/-3.89 | 描述性 |
| Cosmos/300/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.23 | 9/15 | — | — | -4.16/-2.86 | 描述性 |
| Cosmos/300/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -0.82 | 2/6 | — | — | -2.60/-1.64 | 描述性 |
| Cosmos/300/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -23.00 | 5/120 | — | — | -27.67/-26.00 | 描述性 |
| Cosmos/300/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -4.20 | 35/56 | — | — | -9.37/-7.20 | 描述性 |
| Cosmos/300/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.00 | 47/22 | — | — | 0.40/2.40 | 描述性 |
| Cosmos/300/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -22.13 | 4/110 | — | — | -26.77/-25.05 | 描述性 |
| Cosmos/300/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -3.55 | 33/50 | — | — | -8.74/-6.68 | 描述性 |
| Cosmos/300/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 5.22 | 43/18 | — | — | 0.70/2.51 | 描述性 |
| Cosmos/300/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -0.40 | 12/14 | — | — | -3.41/-2.00 | 描述性 |
| Cosmos/300/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | 0.60 | 11/8 | — | — | -2.02/-0.80 | 描述性 |
| Cosmos/300/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.80 | 10/6 | — | — | -1.63/-0.40 | 描述性 |
| Cosmos/300/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -0.41 | 11/13 | — | — | -3.38/-2.04 | 描述性 |
| Cosmos/300/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | 0.82 | 11/7 | — | — | -1.80/-0.61 | 描述性 |
| Cosmos/300/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.82 | 10/6 | — | — | -1.67/-0.41 | 描述性 |
| Cosmos/300/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -15.00 | 5/80 | — | — | -19.19/-17.80 | 描述性 |
| Cosmos/300/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -2.00 | 32/42 | — | — | -6.77/-4.80 | 描述性 |
| Cosmos/300/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 6.60 | 54/21 | — | — | 1.87/3.80 | 描述性 |
| Cosmos/300/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 8.00 | 43/3 | — | — | 4.55/6.00 | 描述性 |
| Cosmos/300/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -14.61 | 4/74 | — | — | -18.77/-17.33 | 描述性 |
| Cosmos/300/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -1.46 | 31/38 | — | — | -6.29/-4.38 | 描述性 |
| Cosmos/300/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 6.68 | 51/19 | — | — | 1.91/3.97 | 描述性 |
| Cosmos/300/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 7.52 | 39/3 | — | — | 4.03/5.43 | 描述性 |
| Cosmos/300/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -0.20 | 12/13 | — | — | -3.16/-1.80 | 描述性 |
| Cosmos/300/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | 0.40 | 9/7 | — | — | -2.04/-0.80 | 描述性 |
| Cosmos/300/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.60 | 15/7 | — | — | -1.17/0.00 | 描述性 |
| Cosmos/300/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.20 | 3/2 | — | — | -1.31/-0.60 | 描述性 |
| Cosmos/300/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -0.41 | 11/13 | — | — | -3.38/-2.04 | 描述性 |
| Cosmos/300/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | 0.41 | 9/7 | — | — | -2.08/-0.82 | 描述性 |
| Cosmos/300/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.64 | 15/7 | — | — | -1.20/0.00 | 描述性 |
| Cosmos/300/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.00 | 2/2 | — | — | -1.42/-0.61 | 描述性 |
| Cosmos/400/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -21.40 | 4/111 | — | — | -25.88/-24.40 | 描述性 |
| Cosmos/400/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | 2.00 | 34/24 | — | — | -2.30/-0.60 | 描述性 |
| Cosmos/400/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -21.09 | 4/105 | — | — | -25.67/-24.01 | 描述性 |
| Cosmos/400/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | 1.88 | 32/23 | — | — | -2.50/-0.63 | 描述性 |
| Cosmos/400/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -2.40 | 11/23 | — | — | -5.72/-4.20 | 描述性 |
| Cosmos/400/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.00 | 10/15 | — | — | -3.94/-2.60 | 描述性 |
| Cosmos/400/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -2.66 | 10/23 | — | — | -5.99/-4.50 | 描述性 |
| Cosmos/400/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.02 | 10/15 | — | — | -4.02/-2.66 | 描述性 |
| Cosmos/400/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -14.20 | 5/76 | — | — | -18.33/-17.00 | 描述性 |
| Cosmos/400/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | 4.40 | 42/20 | — | — | 0.01/1.80 | 描述性 |
| Cosmos/400/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 7.20 | 42/6 | — | — | 3.53/5.20 | 描述性 |
| Cosmos/400/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -14.20 | 5/73 | — | — | -18.44/-17.12 | 描述性 |
| Cosmos/400/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | 3.97 | 38/19 | — | — | -0.45/1.46 | 描述性 |
| Cosmos/400/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 6.89 | 39/6 | — | — | 3.15/4.80 | 描述性 |
| Cosmos/400/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -1.80 | 11/20 | — | — | -5.01/-3.60 | 描述性 |
| Cosmos/400/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.20 | 9/15 | — | — | -4.07/-2.80 | 描述性 |
| Cosmos/400/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.60 | 6/3 | — | — | -1.30/-0.40 | 描述性 |
| Cosmos/400/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -2.04 | 10/20 | — | — | -5.26/-3.89 | 描述性 |
| Cosmos/400/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.23 | 9/15 | — | — | -4.16/-2.86 | 描述性 |
| Cosmos/400/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.61 | 6/3 | — | — | -1.33/-0.41 | 描述性 |
| Cosmos/400/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -17.20 | 5/91 | — | — | -21.54/-20.00 | 描述性 |
| Cosmos/400/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | 1.60 | 40/32 | — | — | -3.14/-1.20 | 描述性 |
| Cosmos/400/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 4.20 | 46/25 | — | — | -0.48/1.40 | 描述性 |
| Cosmos/400/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -17.33 | 5/88 | — | — | -21.80/-20.04 | 描述性 |
| Cosmos/400/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | 1.25 | 37/31 | — | — | -3.56/-1.67 | 描述性 |
| Cosmos/400/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 3.76 | 43/25 | — | — | -1.04/1.04 | 描述性 |
| Cosmos/400/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -0.80 | 12/16 | — | — | -3.90/-2.40 | 描述性 |
| Cosmos/400/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | 0.20 | 9/8 | — | — | -2.30/-1.20 | 描述性 |
| Cosmos/400/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.60 | 16/8 | — | — | -1.29/0.00 | 描述性 |
| Cosmos/400/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -0.82 | 11/15 | — | — | -3.88/-2.45 | 描述性 |
| Cosmos/400/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | 0.41 | 9/7 | — | — | -2.08/-0.82 | 描述性 |
| Cosmos/400/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.84 | 16/7 | — | — | -1.05/0.20 | 描述性 |
| Cosmos/400/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -14.60 | 5/78 | — | — | -18.76/-17.20 | 描述性 |
| Cosmos/400/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -1.60 | 27/35 | — | — | -6.01/-4.20 | 描述性 |
| Cosmos/400/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | -0.40 | 24/26 | — | — | -4.43/-2.60 | 描述性 |
| Cosmos/400/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 2.60 | 26/13 | — | — | -0.98/0.60 | 描述性 |
| Cosmos/400/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -14.61 | 5/75 | — | — | -18.89/-17.33 | 描述性 |
| Cosmos/400/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -1.46 | 26/33 | — | — | -5.96/-4.18 | 描述性 |
| Cosmos/400/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | -0.42 | 23/25 | — | — | -4.54/-2.71 | 描述性 |
| Cosmos/400/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 2.71 | 26/13 | — | — | -1.02/0.63 | 描述性 |
| Cosmos/400/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -1.20 | 12/18 | — | — | -4.38/-3.00 | 描述性 |
| Cosmos/400/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -0.60 | 8/11 | — | — | -3.21/-2.00 | 描述性 |
| Cosmos/400/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.60 | 13/10 | — | — | -2.26/-1.00 | 描述性 |
| Cosmos/400/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -0.40 | 3/5 | — | — | -2.19/-1.40 | 描述性 |
| Cosmos/400/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -1.23 | 11/17 | — | — | -4.38/-3.07 | 描述性 |
| Cosmos/400/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.41 | 8/10 | — | — | -3.02/-1.84 | 描述性 |
| Cosmos/400/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.82 | 13/9 | — | — | -2.04/-0.82 | 描述性 |
| Cosmos/400/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -0.41 | 3/5 | — | — | -2.24/-1.43 | 描述性 |
| Cosmos/500/c/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -23.40 | 5/122 | -30.91 | -28.80 | -28.08/-26.40 | 未证明 |
| Cosmos/500/c/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -22.96 | 5/115 | -30.64 | -28.60 | -27.75/-26.10 | 描述性 |
| Cosmos/500/c/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -1.40 | 12/19 | -6.61 | -4.60 | -4.62/-3.20 | 未证明 |
| Cosmos/500/c/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -1.64 | 11/19 | -6.87 | -4.91 | -4.87/-3.48 | 描述性 |
| Cosmos/500/cg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -18.60 | 5/98 | -25.72 | -23.80 | -23.03/-21.60 | 未证明 |
| Cosmos/500/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 4.80 | 34/10 | — | — | 1.11/2.80 | 描述性 |
| Cosmos/500/cg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -18.16 | 5/92 | -25.43 | -23.80 | -22.69/-21.09 | 描述性 |
| Cosmos/500/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 4.80 | 32/9 | — | — | 1.08/2.71 | 描述性 |
| Cosmos/500/cg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -0.60 | 13/16 | -5.70 | -3.80 | -3.75/-2.40 | 未证明 |
| Cosmos/500/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.80 | 7/3 | — | — | -1.18/-0.20 | 描述性 |
| Cosmos/500/cg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -0.82 | 12/16 | -5.94 | -4.09 | -3.98/-2.66 | 描述性 |
| Cosmos/500/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.82 | 7/3 | — | — | -1.20/-0.20 | 描述性 |
| Cosmos/500/cr/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -18.80 | 6/100 | -26.09 | -24.00 | -23.34/-21.60 | 未证明 |
| Cosmos/500/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 4.60 | 46/23 | — | — | -0.01/1.80 | 描述性 |
| Cosmos/500/cr/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -18.58 | 5/94 | -25.89 | -23.90 | -23.13/-21.50 | 描述性 |
| Cosmos/500/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 4.38 | 43/22 | — | — | -0.30/1.67 | 描述性 |
| Cosmos/500/cr/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -1.00 | 12/17 | -6.08 | -4.20 | -4.14/-2.80 | 未证明 |
| Cosmos/500/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.40 | 10/8 | — | — | -2.17/-1.00 | 描述性 |
| Cosmos/500/cr/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -1.23 | 11/17 | -6.33 | -4.50 | -4.38/-3.07 | 描述性 |
| Cosmos/500/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.41 | 10/8 | — | — | -2.21/-1.02 | 描述性 |
| Cosmos/500/crg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -13.00 | 7/72 | -19.80 | -18.00 | -17.22/-15.60 | 未证明 |
| Cosmos/500/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.60 | 48/20 | — | — | 1.05/3.00 | 描述性 |
| Cosmos/500/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 5.80 | 33/4 | — | — | 2.55/4.00 | 描述性 |
| Cosmos/500/crg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -13.15 | 6/69 | -20.01 | -18.16 | -17.41/-15.87 | 描述性 |
| Cosmos/500/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 5.01 | 44/20 | — | — | 0.38/2.30 | 描述性 |
| Cosmos/500/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 5.43 | 30/4 | — | — | 2.14/3.55 | 描述性 |
| Cosmos/500/crg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -0.60 | 13/16 | -5.70 | -3.80 | -3.75/-2.40 | 未证明 |
| Cosmos/500/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.00 | 10/10 | — | — | -2.68/-1.40 | 描述性 |
| Cosmos/500/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.40 | 3/1 | — | — | -0.99/-0.20 | 描述性 |
| Cosmos/500/crg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -0.82 | 12/16 | -5.94 | -4.09 | -3.98/-2.45 | 描述性 |
| Cosmos/500/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.00 | 10/10 | — | — | -2.74/-1.43 | 描述性 |
| Cosmos/500/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.41 | 3/1 | — | — | -1.01/-0.20 | 描述性 |
| DP/50/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -55.40 | 12/289 | — | — | -60.93/-59.20 | 描述性 |
| DP/50/c/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -40.60 | 17/220 | -49.95 | -47.40 | -46.48/-44.40 | 未证明 |
| DP/50/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -56.16 | 12/281 | — | — | -61.81/-59.92 | 描述性 |
| DP/50/c/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -41.34 | 17/215 | -50.94 | -48.64 | -47.38/-45.30 | 描述性 |
| DP/50/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -19.00 | 36/131 | — | — | -25.19/-22.80 | 描述性 |
| DP/50/c/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -22.40 | 15/127 | -31.00 | -29.00 | -27.77/-25.80 | 未证明 |
| DP/50/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -19.02 | 34/127 | — | — | -25.24/-22.70 | 描述性 |
| DP/50/c/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -22.09 | 15/123 | -30.79 | -28.02 | -27.52/-25.56 | 描述性 |
| DP/50/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -29.00 | 32/177 | — | — | -35.35/-32.60 | 描述性 |
| DP/50/cg/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -36.80 | 13/197 | -45.79 | -42.30 | -42.44/-40.00 | 未证明 |
| DP/50/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 26.40 | 141/9 | — | — | 20.91/23.60 | 描述性 |
| DP/50/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -29.65 | 29/171 | — | — | -36.08/-33.40 | 描述性 |
| DP/50/cg/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -36.12 | 13/186 | -45.33 | -41.86 | -41.91/-39.46 | 描述性 |
| DP/50/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 26.51 | 136/9 | — | — | 20.86/23.59 | 描述性 |
| DP/50/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -7.40 | 36/73 | — | — | -12.91/-10.60 | 描述性 |
| DP/50/cg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -9.20 | 25/71 | -17.44 | -14.80 | -14.31/-12.20 | 未证明 |
| DP/50/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 11.60 | 76/18 | — | — | 6.54/8.60 | 描述性 |
| DP/50/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -7.36 | 34/70 | — | — | -12.87/-10.43 | 描述性 |
| DP/50/cg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -9.00 | 24/68 | -17.26 | -14.52 | -14.12/-11.86 | 描述性 |
| DP/50/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 11.66 | 75/18 | — | — | 6.50/8.59 | 描述性 |
| DP/50/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -47.20 | 19/255 | — | — | -53.16/-50.80 | 描述性 |
| DP/50/cr/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -35.80 | 21/200 | -45.41 | -42.50 | -41.82/-39.40 | 未证明 |
| DP/50/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 8.20 | 77/36 | — | — | 2.52/5.20 | 描述性 |
| DP/50/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -48.23 | 17/248 | — | — | -54.25/-51.77 | 描述性 |
| DP/50/cr/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -36.74 | 20/196 | -46.55 | -43.32 | -42.90/-40.50 | 描述性 |
| DP/50/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 7.93 | 74/36 | — | — | 2.08/4.80 | 描述性 |
| DP/50/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -13.40 | 37/104 | — | — | -19.36/-17.01 | 描述性 |
| DP/50/cr/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -15.80 | 21/100 | -24.42 | -22.20 | -21.16/-19.20 | 未证明 |
| DP/50/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.60 | 80/52 | — | — | -0.52/1.80 | 描述性 |
| DP/50/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -13.50 | 35/101 | — | — | -19.48/-17.18 | 描述性 |
| DP/50/cr/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -15.54 | 21/97 | -24.27 | -21.68 | -20.97/-19.02 | 描述性 |
| DP/50/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 5.52 | 77/50 | — | — | -0.63/1.84 | 描述性 |
| DP/50/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -20.20 | 38/139 | — | — | -26.52/-23.60 | 描述性 |
| DP/50/crg/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -29.40 | 8/155 | -37.65 | -34.20 | -34.56/-32.20 | 未证明 |
| DP/50/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 8.80 | 65/21 | — | — | 3.83/6.00 | 描述性 |
| DP/50/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 27.00 | 143/8 | — | — | 21.55/24.00 | 描述性 |
| DP/50/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -20.67 | 34/133 | — | — | -27.04/-24.01 | 描述性 |
| DP/50/crg/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -28.81 | 8/146 | -37.25 | -33.82 | -34.09/-31.52 | 描述性 |
| DP/50/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 8.98 | 63/20 | — | — | 3.88/6.25 | 描述性 |
| DP/50/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 27.56 | 138/6 | — | — | 22.09/24.63 | 描述性 |
| DP/50/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -5.80 | 37/66 | — | — | -11.22/-9.00 | 描述性 |
| DP/50/crg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -8.60 | 21/64 | -16.39 | -13.80 | -13.43/-11.60 | 未证明 |
| DP/50/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.60 | 43/35 | — | — | -3.31/-1.40 | 描述性 |
| DP/50/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 7.60 | 63/25 | — | — | 2.53/4.60 | 描述性 |
| DP/50/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -6.13 | 35/65 | — | — | -11.59/-9.41 | 描述性 |
| DP/50/crg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -8.79 | 20/63 | -16.65 | -14.31 | -13.66/-11.66 | 描述性 |
| DP/50/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.23 | 41/35 | — | — | -3.73/-1.64 | 描述性 |
| DP/50/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 7.36 | 61/25 | — | — | 2.22/4.29 | 描述性 |
| DP/100/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -33.00 | 31/196 | — | — | -39.38/-36.80 | 描述性 |
| DP/100/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -18.20 | 38/129 | — | — | -24.44/-22.00 | 描述性 |
| DP/100/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -33.82 | 27/189 | — | — | -40.25/-37.79 | 描述性 |
| DP/100/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -19.00 | 35/126 | — | — | -25.35/-22.96 | 描述性 |
| DP/100/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -4.60 | 35/58 | — | — | -9.81/-7.60 | 描述性 |
| DP/100/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -8.00 | 18/58 | — | — | -12.59/-10.60 | 描述性 |
| DP/100/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -4.70 | 33/56 | — | — | -9.92/-7.77 | 描述性 |
| DP/100/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -7.77 | 18/56 | — | — | -12.42/-10.43 | 描述性 |
| DP/100/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -6.00 | 52/82 | — | — | -12.08/-9.60 | 描述性 |
| DP/100/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -13.80 | 21/90 | — | — | -19.04/-17.00 | 描述性 |
| DP/100/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 27.00 | 148/13 | — | — | 21.22/23.60 | 描述性 |
| DP/100/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -6.47 | 47/78 | — | — | -12.60/-10.02 | 描述性 |
| DP/100/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -12.94 | 21/83 | — | — | -18.29/-16.28 | 描述性 |
| DP/100/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 27.35 | 143/12 | — | — | 21.45/23.80 | 描述性 |
| DP/100/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | 0.20 | 31/30 | — | — | -4.20/-2.20 | 描述性 |
| DP/100/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.60 | 24/32 | — | — | -5.82/-4.00 | 描述性 |
| DP/100/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 4.80 | 40/16 | — | — | 0.63/2.40 | 描述性 |
| DP/100/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 0.20 | 30/29 | — | — | -4.23/-2.25 | 描述性 |
| DP/100/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.43 | 24/31 | — | — | -5.71/-3.89 | 描述性 |
| DP/100/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 4.91 | 39/15 | — | — | 0.72/2.45 | 描述性 |
| DP/100/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -29.20 | 33/179 | — | — | -35.59/-33.00 | 描述性 |
| DP/100/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -17.80 | 37/126 | — | — | -23.98/-21.60 | 描述性 |
| DP/100/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 3.80 | 80/61 | — | — | -2.50/0.20 | 描述性 |
| DP/100/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -29.65 | 31/173 | — | — | -36.16/-33.40 | 描述性 |
| DP/100/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -18.16 | 36/123 | — | — | -24.52/-21.92 | 描述性 |
| DP/100/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 4.18 | 77/57 | — | — | -2.24/0.42 | 描述性 |
| DP/100/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -2.00 | 35/45 | — | — | -6.93/-4.80 | 描述性 |
| DP/100/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -4.40 | 19/41 | — | — | -8.66/-6.80 | 描述性 |
| DP/100/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.60 | 36/23 | — | — | -1.73/0.20 | 描述性 |
| DP/100/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -2.04 | 33/43 | — | — | -6.97/-4.91 | 描述性 |
| DP/100/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -4.09 | 19/39 | — | — | -8.39/-6.54 | 描述性 |
| DP/100/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.66 | 35/22 | — | — | -1.70/0.20 | 描述性 |
| DP/100/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -3.20 | 57/73 | — | — | -9.25/-6.60 | 描述性 |
| DP/100/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -12.40 | 14/76 | — | — | -17.11/-15.20 | 描述性 |
| DP/100/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.80 | 58/44 | — | — | -2.71/-0.40 | 描述性 |
| DP/100/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 26.00 | 142/12 | — | — | 20.33/22.80 | 描述性 |
| DP/100/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -3.34 | 52/68 | — | — | -9.43/-6.68 | 描述性 |
| DP/100/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -11.48 | 14/69 | — | — | -16.27/-14.41 | 描述性 |
| DP/100/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 3.13 | 57/42 | — | — | -2.53/-0.21 | 描述性 |
| DP/100/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 26.30 | 136/10 | — | — | 20.59/22.96 | 描述性 |
| DP/100/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -1.00 | 33/38 | — | — | -5.70/-3.80 | 描述性 |
| DP/100/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -3.80 | 21/40 | — | — | -8.12/-6.20 | 描述性 |
| DP/100/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | -1.20 | 27/33 | — | — | -5.56/-3.80 | 描述性 |
| DP/100/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.00 | 20/15 | — | — | -2.44/-0.80 | 描述性 |
| DP/100/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -1.23 | 31/37 | — | — | -5.93/-3.89 | 描述性 |
| DP/100/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -3.89 | 21/40 | — | — | -8.29/-6.54 | 描述性 |
| DP/100/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | -1.43 | 25/32 | — | — | -5.78/-3.89 | 描述性 |
| DP/100/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.82 | 19/15 | — | — | -2.65/-1.23 | 描述性 |
| DP/200/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -30.80 | 25/179 | — | — | -36.91/-34.60 | 描述性 |
| DP/200/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -16.00 | 39/119 | — | — | -22.18/-19.80 | 描述性 |
| DP/200/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -31.52 | 22/173 | — | — | -37.70/-35.28 | 描述性 |
| DP/200/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -16.70 | 35/115 | — | — | -22.95/-20.67 | 描述性 |
| DP/200/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -2.80 | 28/42 | — | — | -7.43/-5.40 | 描述性 |
| DP/200/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -6.20 | 12/43 | — | — | -10.16/-8.40 | 描述性 |
| DP/200/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.07 | 26/41 | — | — | -7.70/-5.73 | 描述性 |
| DP/200/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -6.13 | 12/42 | — | — | -10.16/-8.38 | 描述性 |
| DP/200/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | 4.20 | 70/49 | — | — | -1.67/0.80 | 描述性 |
| DP/200/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -3.60 | 33/51 | — | — | -8.61/-6.40 | 描述性 |
| DP/200/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 35.00 | 183/8 | — | — | 29.24/31.60 | 描述性 |
| DP/200/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | 4.18 | 64/44 | — | — | -1.69/0.63 | 描述性 |
| DP/200/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -2.30 | 33/44 | — | — | -7.35/-5.22 | 描述性 |
| DP/200/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 35.70 | 178/7 | — | — | 29.83/32.15 | 描述性 |
| DP/200/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | 1.00 | 35/30 | — | — | -3.53/-1.60 | 描述性 |
| DP/200/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -0.80 | 25/29 | — | — | -4.96/-3.20 | 描述性 |
| DP/200/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 3.80 | 35/16 | — | — | -0.22/1.60 | 描述性 |
| DP/200/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 0.61 | 33/30 | — | — | -3.95/-2.04 | 描述性 |
| DP/200/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.02 | 24/29 | — | — | -5.24/-3.48 | 描述性 |
| DP/200/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 3.68 | 34/16 | — | — | -0.40/1.43 | 描述性 |
| DP/200/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -28.60 | 30/173 | — | — | -34.87/-32.60 | 描述性 |
| DP/200/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -17.20 | 36/122 | — | — | -23.32/-21.00 | 描述性 |
| DP/200/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.20 | 70/59 | — | — | -3.88/-1.40 | 描述性 |
| DP/200/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -29.23 | 27/167 | — | — | -35.57/-33.19 | 描述性 |
| DP/200/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -17.75 | 35/120 | — | — | -24.04/-21.71 | 描述性 |
| DP/200/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.30 | 67/56 | — | — | -3.90/-1.46 | 描述性 |
| DP/200/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | 2.20 | 33/22 | — | — | -2.00/-0.20 | 描述性 |
| DP/200/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -0.20 | 18/19 | — | — | -3.72/-2.20 | 描述性 |
| DP/200/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.00 | 34/9 | — | — | 1.37/3.00 | 描述性 |
| DP/200/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 1.84 | 31/22 | — | — | -2.38/-0.61 | 描述性 |
| DP/200/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.20 | 18/19 | — | — | -3.81/-2.25 | 描述性 |
| DP/200/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 4.91 | 33/9 | — | — | 1.23/2.86 | 描述性 |
| DP/200/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | 0.80 | 65/61 | — | — | -5.22/-2.80 | 描述性 |
| DP/200/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -8.40 | 22/64 | — | — | -13.27/-11.40 | 描述性 |
| DP/200/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | -3.40 | 32/49 | — | — | -8.33/-6.20 | 描述性 |
| DP/200/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 29.40 | 159/12 | — | — | 23.58/26.00 | 描述性 |
| DP/200/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | 0.21 | 59/58 | — | — | -5.86/-3.34 | 描述性 |
| DP/200/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -7.93 | 22/60 | — | — | -12.93/-10.86 | 描述性 |
| DP/200/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | -3.97 | 28/47 | — | — | -8.91/-6.89 | 描述性 |
| DP/200/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 29.44 | 153/12 | — | — | 23.45/26.10 | 描述性 |
| DP/200/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | 2.20 | 36/25 | — | — | -2.20/-0.40 | 描述性 |
| DP/200/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -0.60 | 20/23 | — | — | -4.36/-2.80 | 描述性 |
| DP/200/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.20 | 24/18 | — | — | -2.53/-1.00 | 描述性 |
| DP/200/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.00 | 14/14 | — | — | -3.11/-1.60 | 描述性 |
| DP/200/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 1.84 | 34/25 | — | — | -2.59/-0.61 | 描述性 |
| DP/200/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.82 | 19/23 | — | — | -4.62/-3.07 | 描述性 |
| DP/200/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.23 | 24/18 | — | — | -2.59/-0.82 | 描述性 |
| DP/200/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.00 | 14/14 | — | — | -3.18/-1.64 | 描述性 |
| DP/300/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -21.80 | 32/141 | — | — | -27.95/-25.60 | 描述性 |
| DP/300/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -7.00 | 38/73 | — | — | -12.57/-10.40 | 描述性 |
| DP/300/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -22.13 | 30/136 | — | — | -28.39/-25.89 | 描述性 |
| DP/300/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -7.31 | 36/71 | — | — | -13.01/-10.86 | 描述性 |
| DP/300/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | 0.60 | 27/24 | — | — | -3.47/-1.80 | 描述性 |
| DP/300/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -2.80 | 8/22 | — | — | -5.89/-4.60 | 描述性 |
| DP/300/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 0.61 | 26/23 | — | — | -3.47/-1.64 | 描述性 |
| DP/300/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -2.45 | 8/20 | — | — | -5.54/-4.09 | 描述性 |
| DP/300/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | 2.60 | 63/50 | — | — | -3.15/-0.80 | 描述性 |
| DP/300/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -5.20 | 24/50 | — | — | -9.87/-8.00 | 描述性 |
| DP/300/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 24.40 | 131/9 | — | — | 19.01/21.20 | 描述性 |
| DP/300/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | 1.88 | 56/47 | — | — | -3.88/-1.46 | 描述性 |
| DP/300/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -4.59 | 24/46 | — | — | -9.37/-7.52 | 描述性 |
| DP/300/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 24.01 | 124/9 | — | — | 18.48/20.67 | 描述性 |
| DP/300/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | 1.60 | 30/22 | — | — | -2.50/-0.80 | 描述性 |
| DP/300/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -0.20 | 21/22 | — | — | -3.97/-2.40 | 描述性 |
| DP/300/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.00 | 15/10 | — | — | -1.96/-0.60 | 描述性 |
| DP/300/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 1.43 | 28/21 | — | — | -2.65/-0.82 | 描述性 |
| DP/300/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.20 | 20/21 | — | — | -3.97/-2.25 | 描述性 |
| DP/300/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.82 | 14/10 | — | — | -2.15/-0.82 | 描述性 |
| DP/300/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -19.00 | 38/133 | — | — | -25.27/-22.80 | 描述性 |
| DP/300/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -7.60 | 38/76 | — | — | -13.22/-11.00 | 描述性 |
| DP/300/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.80 | 57/43 | — | — | -2.66/-0.40 | 描述性 |
| DP/300/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -19.21 | 33/125 | — | — | -25.48/-22.96 | 描述性 |
| DP/300/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -7.72 | 35/72 | — | — | -13.41/-11.27 | 描述性 |
| DP/300/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.92 | 54/40 | — | — | -2.61/-0.21 | 描述性 |
| DP/300/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | 3.20 | 32/16 | — | — | -0.73/1.00 | 描述性 |
| DP/300/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | 0.80 | 14/10 | — | — | -2.11/-0.80 | 描述性 |
| DP/300/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.60 | 24/11 | — | — | -0.80/0.80 | 描述性 |
| DP/300/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 3.07 | 30/15 | — | — | -0.83/0.82 | 描述性 |
| DP/300/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | 1.02 | 14/9 | — | — | -1.89/-0.61 | 描述性 |
| DP/300/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.45 | 22/10 | — | — | -0.89/0.61 | 描述性 |
| DP/300/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | 5.20 | 67/41 | — | — | -0.42/1.80 | 描述性 |
| DP/300/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -4.00 | 23/43 | — | — | -8.47/-6.60 | 描述性 |
| DP/300/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.60 | 46/33 | — | — | -2.33/-0.40 | 描述性 |
| DP/300/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 24.20 | 131/10 | — | — | 18.75/21.00 | 描述性 |
| DP/300/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | 4.80 | 60/37 | — | — | -0.79/1.46 | 描述性 |
| DP/300/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -3.34 | 23/39 | — | — | -7.89/-6.05 | 描述性 |
| DP/300/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.92 | 45/31 | — | — | -2.13/0.00 | 描述性 |
| DP/300/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 24.01 | 125/10 | — | — | 18.41/20.67 | 描述性 |
| DP/300/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | 1.80 | 30/21 | — | — | -2.26/-0.60 | 描述性 |
| DP/300/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.00 | 17/22 | — | — | -4.59/-3.00 | 描述性 |
| DP/300/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.20 | 20/19 | — | — | -3.41/-1.80 | 描述性 |
| DP/300/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -1.40 | 6/13 | — | — | -3.96/-2.80 | 描述性 |
| DP/300/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 1.43 | 28/21 | — | — | -2.65/-0.82 | 描述性 |
| DP/300/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.23 | 16/22 | — | — | -4.85/-3.27 | 描述性 |
| DP/300/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.00 | 19/19 | — | — | -3.65/-2.04 | 描述性 |
| DP/300/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -1.64 | 5/13 | — | — | -4.17/-3.07 | 描述性 |
| DP/400/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -18.40 | 37/129 | — | — | -24.61/-22.20 | 描述性 |
| DP/400/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -3.60 | 38/56 | — | — | -8.86/-6.80 | 描述性 |
| DP/400/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -18.58 | 35/124 | — | — | -24.91/-22.34 | 描述性 |
| DP/400/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -3.76 | 36/54 | — | — | -9.13/-7.10 | 描述性 |
| DP/400/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | 1.80 | 31/22 | — | — | -2.33/-0.60 | 描述性 |
| DP/400/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.60 | 5/13 | — | — | -4.08/-3.00 | 描述性 |
| DP/400/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 1.64 | 30/22 | — | — | -2.56/-0.82 | 描述性 |
| DP/400/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.43 | 5/12 | — | — | -3.91/-2.86 | 描述性 |
| DP/400/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | 6.20 | 72/41 | — | — | 0.48/2.80 | 描述性 |
| DP/400/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -1.60 | 31/39 | — | — | -6.26/-4.20 | 描述性 |
| DP/400/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 24.60 | 132/9 | — | — | 19.20/21.40 | 描述性 |
| DP/400/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | 5.64 | 64/37 | — | — | -0.05/2.30 | 描述性 |
| DP/400/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -0.84 | 31/35 | — | — | -5.58/-3.55 | 描述性 |
| DP/400/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 24.22 | 123/7 | — | — | 18.83/21.09 | 描述性 |
| DP/400/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | 0.80 | 30/26 | — | — | -3.44/-1.60 | 描述性 |
| DP/400/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.00 | 12/17 | — | — | -4.14/-2.80 | 描述性 |
| DP/400/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -1.00 | 10/15 | — | — | -3.94/-2.60 | 描述性 |
| DP/400/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 0.82 | 30/26 | — | — | -3.52/-1.64 | 描述性 |
| DP/400/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.82 | 12/16 | — | — | -3.98/-2.66 | 描述性 |
| DP/400/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -0.82 | 10/14 | — | — | -3.77/-2.45 | 描述性 |
| DP/400/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -16.00 | 43/123 | — | — | -22.32/-19.80 | 描述性 |
| DP/400/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -4.60 | 39/62 | — | — | -10.01/-7.80 | 描述性 |
| DP/400/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.40 | 63/51 | — | — | -3.37/-1.00 | 描述性 |
| DP/400/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -15.87 | 40/116 | — | — | -22.28/-19.62 | 描述性 |
| DP/400/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -4.38 | 38/59 | — | — | -9.92/-7.72 | 描述性 |
| DP/400/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.71 | 61/48 | — | — | -3.18/-0.84 | 描述性 |
| DP/400/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | 3.60 | 32/14 | — | — | -0.23/1.40 | 描述性 |
| DP/400/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | 1.20 | 16/10 | — | — | -1.81/-0.40 | 描述性 |
| DP/400/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.80 | 20/11 | — | — | -1.44/0.00 | 描述性 |
| DP/400/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 3.48 | 30/13 | — | — | -0.33/1.23 | 描述性 |
| DP/400/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | 1.43 | 16/9 | — | — | -1.58/-0.20 | 描述性 |
| DP/400/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.84 | 19/10 | — | — | -1.37/0.00 | 描述性 |
| DP/400/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | 8.00 | 75/35 | — | — | 2.38/4.60 | 描述性 |
| DP/400/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -1.20 | 29/35 | — | — | -5.68/-3.80 | 描述性 |
| DP/400/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.80 | 43/34 | — | — | -3.08/-1.00 | 描述性 |
| DP/400/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 24.00 | 133/13 | — | — | 18.37/20.60 | 描述性 |
| DP/400/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | 7.52 | 69/33 | — | — | 1.84/4.18 | 描述性 |
| DP/400/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -0.63 | 29/32 | — | — | -5.21/-3.34 | 描述性 |
| DP/400/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 1.88 | 40/31 | — | — | -3.03/-1.04 | 描述性 |
| DP/400/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 23.38 | 124/12 | — | — | 17.68/20.04 | 描述性 |
| DP/400/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | 2.60 | 28/15 | — | — | -1.15/0.60 | 描述性 |
| DP/400/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -0.20 | 17/18 | — | — | -3.64/-2.00 | 描述性 |
| DP/400/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.80 | 21/12 | — | — | -1.54/0.00 | 描述性 |
| DP/400/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -1.00 | 6/11 | — | — | -3.46/-2.40 | 描述性 |
| DP/400/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | 2.45 | 27/15 | — | — | -1.34/0.41 | 描述性 |
| DP/400/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.20 | 16/17 | — | — | -3.63/-2.04 | 描述性 |
| DP/400/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.64 | 20/12 | — | — | -1.73/-0.20 | 描述性 |
| DP/400/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -1.02 | 5/10 | — | — | -3.40/-2.25 | 描述性 |
| DP/500/c/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -14.80 | 46/120 | -25.08 | -22.00 | -21.18/-18.60 | 未证明 |
| DP/500/c/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -14.82 | 43/114 | -25.26 | -21.92 | -21.30/-18.79 | 描述性 |
| DP/500/c/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 3.40 | 30/13 | -2.64 | -0.30 | -0.32/1.20 | 未证明 |
| DP/500/c/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 3.07 | 28/13 | -2.99 | -0.82 | -0.66/1.02 | 描述性 |
| DP/500/cg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | 7.80 | 76/37 | -1.47 | 1.80 | 2.11/4.40 | PASS |
| DP/500/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 22.60 | 126/13 | — | — | 17.05/19.40 | 描述性 |
| DP/500/cg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | 6.47 | 67/36 | -2.83 | 0.42 | 0.75/3.13 | 描述性 |
| DP/500/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 21.29 | 115/13 | — | — | 15.65/17.95 | 描述性 |
| DP/500/cg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 1.80 | 31/22 | -4.90 | -2.40 | -2.33/-0.60 | 未证明 |
| DP/500/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -1.60 | 10/18 | — | — | -4.67/-3.20 | 描述性 |
| DP/500/cg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 1.64 | 30/22 | -5.16 | -2.66 | -2.56/-0.61 | 描述性 |
| DP/500/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -1.43 | 10/17 | — | — | -4.52/-3.07 | 描述性 |
| DP/500/cr/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -11.40 | 46/103 | -21.41 | -18.40 | -17.60/-15.20 | 未证明 |
| DP/500/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 3.40 | 68/51 | — | — | -2.47/-0.20 | 描述性 |
| DP/500/cr/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -11.48 | 44/99 | -21.71 | -18.27 | -17.82/-15.24 | 描述性 |
| DP/500/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 3.34 | 65/49 | — | — | -2.67/-0.21 | 描述性 |
| DP/500/cr/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 2.40 | 30/18 | -4.00 | -1.70 | -1.55/0.20 | 未证明 |
| DP/500/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | -1.00 | 10/15 | — | — | -3.94/-2.60 | 描述性 |
| DP/500/cr/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 2.04 | 28/18 | -4.38 | -2.04 | -1.92/-0.20 | 描述性 |
| DP/500/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | -1.02 | 10/15 | — | — | -4.02/-2.66 | 描述性 |
| DP/500/crg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | 9.20 | 78/32 | 0.10 | 3.40 | 3.62/6.00 | PASS |
| DP/500/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.40 | 31/24 | — | — | -2.81/-1.00 | 描述性 |
| DP/500/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 20.60 | 114/11 | — | — | 15.29/17.40 | 描述性 |
| DP/500/crg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | 8.14 | 69/30 | -0.94 | 2.09 | 2.56/4.80 | 描述性 |
| DP/500/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 1.67 | 31/23 | — | — | -2.68/-0.84 | 描述性 |
| DP/500/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 19.62 | 105/11 | — | — | 14.22/16.49 | 描述性 |
| DP/500/crg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 2.80 | 33/19 | -3.83 | -1.20 | -1.28/0.60 | 未证明 |
| DP/500/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.00 | 18/13 | — | — | -2.26/-0.80 | 描述性 |
| DP/500/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.40 | 14/12 | — | — | -2.62/-1.20 | 描述性 |
| DP/500/crg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 2.66 | 31/18 | -3.94 | -1.64 | -1.41/0.41 | 描述性 |
| DP/500/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.02 | 17/12 | — | — | -2.21/-0.82 | 描述性 |
| DP/500/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.61 | 14/11 | — | — | -2.42/-1.02 | 描述性 |
| Fast-WAM/50/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -52.60 | 11/274 | — | — | -58.12/-56.00 | 描述性 |
| Fast-WAM/50/c/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -30.00 | 16/166 | -39.07 | -36.40 | -35.68/-33.80 | 未证明 |
| Fast-WAM/50/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -51.77 | 11/259 | — | — | -57.45/-55.32 | 描述性 |
| Fast-WAM/50/c/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -29.44 | 16/157 | -38.74 | -36.33 | -35.26/-33.19 | 描述性 |
| Fast-WAM/50/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -12.20 | 18/79 | — | — | -17.15/-15.00 | 描述性 |
| Fast-WAM/50/c/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -9.20 | 27/73 | -17.61 | -14.80 | -14.42/-12.20 | 未证明 |
| Fast-WAM/50/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -12.88 | 16/79 | — | — | -17.84/-15.75 | 描述性 |
| Fast-WAM/50/c/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -10.02 | 24/73 | -18.42 | -15.64 | -15.23/-13.09 | 描述性 |
| Fast-WAM/50/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -25.40 | 17/144 | — | — | -30.99/-29.00 | 描述性 |
| Fast-WAM/50/cg/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -15.20 | 37/113 | -24.96 | -22.00 | -21.26/-19.00 | 未证明 |
| Fast-WAM/50/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 27.20 | 150/14 | — | — | 21.36/23.60 | 描述性 |
| Fast-WAM/50/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -25.47 | 17/139 | — | — | -31.23/-29.02 | 描述性 |
| Fast-WAM/50/cg/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -15.24 | 36/109 | -25.25 | -21.92 | -21.46/-19.00 | 描述性 |
| Fast-WAM/50/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 26.30 | 140/14 | — | — | 20.33/22.76 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -6.60 | 18/51 | — | — | -11.05/-9.20 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -5.00 | 21/46 | -12.21 | -9.70 | -9.46/-7.60 | 未证明 |
| Fast-WAM/50/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 5.60 | 44/16 | — | — | 1.32/3.20 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -7.16 | 16/51 | — | — | -11.60/-9.82 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -5.73 | 18/46 | -12.87 | -10.43 | -10.15/-8.38 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 5.73 | 44/16 | — | — | 1.35/3.27 | 描述性 |
| Fast-WAM/50/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -46.60 | 11/244 | — | — | -52.17/-50.00 | 描述性 |
| Fast-WAM/50/cr/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -25.40 | 24/151 | -34.91 | -32.00 | -31.33/-29.20 | 未证明 |
| Fast-WAM/50/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 6.00 | 55/25 | — | — | 1.10/3.20 | 描述性 |
| Fast-WAM/50/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -45.93 | 11/231 | — | — | -51.65/-49.48 | 描述性 |
| Fast-WAM/50/cr/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -25.68 | 22/145 | -35.32 | -32.67 | -31.70/-29.44 | 描述性 |
| Fast-WAM/50/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 5.85 | 53/25 | — | — | 0.79/2.92 | 描述性 |
| Fast-WAM/50/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -13.80 | 25/94 | — | — | -19.24/-17.00 | 描述性 |
| Fast-WAM/50/cr/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -11.40 | 34/91 | -20.61 | -17.40 | -17.11/-14.80 | 未证明 |
| Fast-WAM/50/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | -1.60 | 45/53 | — | — | -6.99/-4.80 | 描述性 |
| Fast-WAM/50/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -14.72 | 21/93 | — | — | -20.10/-17.79 | 描述性 |
| Fast-WAM/50/cr/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -12.27 | 30/90 | -21.43 | -18.40 | -17.95/-15.54 | 描述性 |
| Fast-WAM/50/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | -1.84 | 43/52 | — | — | -7.27/-5.11 | 描述性 |
| Fast-WAM/50/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -19.80 | 19/118 | — | — | -25.27/-23.00 | 描述性 |
| Fast-WAM/50/crg/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -11.60 | 26/84 | -20.21 | -17.40 | -16.95/-14.80 | 未证明 |
| Fast-WAM/50/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.60 | 62/34 | — | — | 0.27/2.60 | 描述性 |
| Fast-WAM/50/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 26.80 | 151/17 | — | — | 20.81/23.20 | 描述性 |
| Fast-WAM/50/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -19.21 | 18/110 | — | — | -24.75/-22.55 | 描述性 |
| Fast-WAM/50/crg/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -11.06 | 25/78 | -19.81 | -17.12 | -16.50/-14.20 | 描述性 |
| Fast-WAM/50/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 6.26 | 61/31 | — | — | 0.82/3.13 | 描述性 |
| Fast-WAM/50/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 26.72 | 144/16 | — | — | 20.62/23.17 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -5.20 | 29/55 | — | — | -10.16/-8.00 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -4.60 | 29/52 | -12.51 | -9.50 | -9.50/-7.40 | 未证明 |
| Fast-WAM/50/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.40 | 44/37 | — | — | -3.59/-1.40 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 8.60 | 58/15 | — | — | 4.03/6.00 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -5.73 | 26/54 | — | — | -10.66/-8.38 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -5.11 | 26/51 | -12.98 | -10.43 | -9.98/-7.77 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.43 | 42/35 | — | — | -3.55/-1.43 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 9.00 | 58/14 | — | — | 4.37/6.34 | 描述性 |
| Fast-WAM/100/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -40.60 | 14/217 | — | — | -46.33/-44.20 | 描述性 |
| Fast-WAM/100/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -18.00 | 22/112 | — | — | -23.53/-21.40 | 描述性 |
| Fast-WAM/100/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -39.87 | 14/205 | — | — | -45.76/-43.42 | 描述性 |
| Fast-WAM/100/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -17.54 | 21/105 | — | — | -23.16/-21.09 | 描述性 |
| Fast-WAM/100/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -7.60 | 22/60 | — | — | -12.40/-10.40 | 描述性 |
| Fast-WAM/100/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -4.60 | 22/45 | — | — | -9.08/-7.20 | 描述性 |
| Fast-WAM/100/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -8.38 | 19/60 | — | — | -13.16/-11.25 | 描述性 |
| Fast-WAM/100/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -5.52 | 18/45 | — | — | -9.92/-7.98 | 描述性 |
| Fast-WAM/100/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -19.80 | 22/121 | — | — | -25.42/-23.20 | 描述性 |
| Fast-WAM/100/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -9.60 | 34/82 | — | — | -15.19/-13.00 | 描述性 |
| Fast-WAM/100/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 20.80 | 121/17 | — | — | 15.12/17.40 | 描述性 |
| Fast-WAM/100/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -20.04 | 21/117 | — | — | -25.79/-23.59 | 描述性 |
| Fast-WAM/100/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -9.81 | 33/80 | — | — | -15.56/-13.15 | 描述性 |
| Fast-WAM/100/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 19.83 | 111/16 | — | — | 14.10/16.28 | 描述性 |
| Fast-WAM/100/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.20 | 20/36 | — | — | -7.37/-5.60 | 描述性 |
| Fast-WAM/100/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.60 | 19/27 | — | — | -5.46/-3.80 | 描述性 |
| Fast-WAM/100/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 4.40 | 35/13 | — | — | 0.52/2.20 | 描述性 |
| Fast-WAM/100/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.68 | 18/36 | — | — | -7.85/-6.13 | 描述性 |
| Fast-WAM/100/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -2.25 | 16/27 | — | — | -6.05/-4.50 | 描述性 |
| Fast-WAM/100/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 4.70 | 34/11 | — | — | 0.87/2.45 | 描述性 |
| Fast-WAM/100/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -37.80 | 13/202 | — | — | -43.46/-41.20 | 描述性 |
| Fast-WAM/100/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -16.60 | 26/109 | — | — | -22.26/-20.20 | 描述性 |
| Fast-WAM/100/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.80 | 52/38 | — | — | -2.42/-0.20 | 描述性 |
| Fast-WAM/100/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -37.37 | 13/192 | — | — | -43.17/-40.71 | 描述性 |
| Fast-WAM/100/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -17.12 | 24/106 | — | — | -22.88/-20.67 | 描述性 |
| Fast-WAM/100/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.51 | 49/37 | — | — | -2.83/-0.63 | 描述性 |
| Fast-WAM/100/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -7.40 | 18/55 | — | — | -11.93/-10.20 | 描述性 |
| Fast-WAM/100/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -5.00 | 23/48 | — | — | -9.59/-7.80 | 描述性 |
| Fast-WAM/100/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.20 | 29/28 | — | — | -4.07/-2.20 | 描述性 |
| Fast-WAM/100/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -7.77 | 16/54 | — | — | -12.28/-10.43 | 描述性 |
| Fast-WAM/100/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -5.32 | 21/47 | — | — | -9.90/-7.98 | 描述性 |
| Fast-WAM/100/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.61 | 28/25 | — | — | -3.62/-1.84 | 描述性 |
| Fast-WAM/100/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -19.40 | 22/119 | — | — | -25.00/-22.80 | 描述性 |
| Fast-WAM/100/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -11.20 | 29/85 | — | — | -16.67/-14.40 | 描述性 |
| Fast-WAM/100/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.40 | 49/47 | — | — | -4.97/-2.80 | 描述性 |
| Fast-WAM/100/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 18.40 | 111/19 | — | — | 12.76/15.20 | 描述性 |
| Fast-WAM/100/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -19.00 | 22/113 | — | — | -24.75/-22.55 | 描述性 |
| Fast-WAM/100/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -10.86 | 28/80 | — | — | -16.43/-14.20 | 描述性 |
| Fast-WAM/100/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 1.04 | 48/43 | — | — | -4.42/-2.30 | 描述性 |
| Fast-WAM/100/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 18.37 | 106/18 | — | — | 12.62/15.03 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.60 | 20/38 | — | — | -7.82/-6.20 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -3.00 | 21/36 | — | — | -7.21/-5.40 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | -0.40 | 20/22 | — | — | -4.13/-2.40 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 3.80 | 32/13 | — | — | 0.01/1.60 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.89 | 18/37 | — | — | -8.08/-6.34 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -3.27 | 19/35 | — | — | -7.46/-5.73 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | -0.20 | 20/21 | — | — | -3.97/-2.45 | 描述性 |
| Fast-WAM/100/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 3.89 | 32/13 | — | — | 0.02/1.64 | 描述性 |
| Fast-WAM/200/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -35.00 | 14/189 | — | — | -40.67/-38.60 | 描述性 |
| Fast-WAM/200/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -12.40 | 23/85 | — | — | -17.65/-15.60 | 描述性 |
| Fast-WAM/200/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -35.07 | 14/182 | — | — | -40.90/-38.62 | 描述性 |
| Fast-WAM/200/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -12.73 | 21/82 | — | — | -18.06/-16.08 | 描述性 |
| Fast-WAM/200/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -6.40 | 15/47 | — | — | -10.62/-8.80 | 描述性 |
| Fast-WAM/200/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -3.40 | 16/33 | — | — | -7.30/-5.60 | 描述性 |
| Fast-WAM/200/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -6.34 | 15/46 | — | — | -10.62/-8.79 | 描述性 |
| Fast-WAM/200/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -3.48 | 16/33 | — | — | -7.47/-5.73 | 描述性 |
| Fast-WAM/200/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -16.80 | 21/105 | — | — | -22.22/-20.20 | 描述性 |
| Fast-WAM/200/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -6.60 | 33/66 | — | — | -11.90/-9.80 | 描述性 |
| Fast-WAM/200/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 18.20 | 106/15 | — | — | 12.80/15.00 | 描述性 |
| Fast-WAM/200/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -16.70 | 20/100 | — | — | -22.23/-20.25 | 描述性 |
| Fast-WAM/200/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -6.47 | 32/63 | — | — | -11.90/-9.60 | 描述性 |
| Fast-WAM/200/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 18.37 | 103/15 | — | — | 12.79/15.03 | 描述性 |
| Fast-WAM/200/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -4.60 | 12/35 | — | — | -8.35/-6.80 | 描述性 |
| Fast-WAM/200/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -3.00 | 16/31 | — | — | -6.85/-5.20 | 描述性 |
| Fast-WAM/200/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.80 | 26/17 | — | — | -1.96/-0.40 | 描述性 |
| Fast-WAM/200/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -4.50 | 12/34 | — | — | -8.31/-6.75 | 描述性 |
| Fast-WAM/200/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -3.07 | 16/31 | — | — | -7.00/-5.32 | 描述性 |
| Fast-WAM/200/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 1.84 | 26/17 | — | — | -2.01/-0.20 | 描述性 |
| Fast-WAM/200/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -28.60 | 22/165 | — | — | -34.54/-32.20 | 描述性 |
| Fast-WAM/200/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -7.40 | 33/70 | — | — | -12.77/-10.60 | 描述性 |
| Fast-WAM/200/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 6.40 | 54/22 | — | — | 1.63/3.60 | 描述性 |
| Fast-WAM/200/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -28.18 | 22/157 | — | — | -34.28/-31.94 | 描述性 |
| Fast-WAM/200/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -7.93 | 30/68 | — | — | -13.38/-11.27 | 描述性 |
| Fast-WAM/200/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 6.89 | 54/21 | — | — | 1.96/3.97 | 描述性 |
| Fast-WAM/200/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -4.40 | 13/35 | — | — | -8.21/-6.60 | 描述性 |
| Fast-WAM/200/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -2.00 | 17/27 | — | — | -5.77/-4.20 | 描述性 |
| Fast-WAM/200/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.00 | 22/12 | — | — | -1.38/0.20 | 描述性 |
| Fast-WAM/200/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -4.29 | 13/34 | — | — | -8.16/-6.54 | 描述性 |
| Fast-WAM/200/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.84 | 17/26 | — | — | -5.66/-3.89 | 描述性 |
| Fast-WAM/200/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.04 | 22/12 | — | — | -1.41/0.20 | 描述性 |
| Fast-WAM/200/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -11.80 | 25/84 | — | — | -17.11/-15.20 | 描述性 |
| Fast-WAM/200/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -3.60 | 36/54 | — | — | -8.76/-6.60 | 描述性 |
| Fast-WAM/200/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.00 | 54/29 | — | — | -0.01/2.20 | 描述性 |
| Fast-WAM/200/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 16.80 | 102/18 | — | — | 11.32/13.60 | 描述性 |
| Fast-WAM/200/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -11.48 | 25/80 | — | — | -16.94/-14.82 | 描述性 |
| Fast-WAM/200/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -3.34 | 35/51 | — | — | -8.62/-6.47 | 描述性 |
| Fast-WAM/200/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 5.22 | 52/27 | — | — | 0.12/2.30 | 描述性 |
| Fast-WAM/200/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 16.70 | 98/18 | — | — | 11.06/13.36 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.00 | 14/29 | — | — | -6.68/-5.20 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -2.40 | 16/28 | — | — | -6.15/-4.60 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.60 | 22/14 | — | — | -1.88/-0.40 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.40 | 20/13 | — | — | -1.94/-0.40 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.07 | 14/29 | — | — | -6.83/-5.11 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -2.45 | 16/28 | — | — | -6.29/-4.70 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.43 | 21/14 | — | — | -2.08/-0.61 | 描述性 |
| Fast-WAM/200/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 1.23 | 19/13 | — | — | -2.15/-0.61 | 描述性 |
| Fast-WAM/300/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -29.00 | 18/163 | — | — | -34.76/-32.60 | 描述性 |
| Fast-WAM/300/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -6.40 | 28/60 | — | — | -11.42/-9.60 | 描述性 |
| Fast-WAM/300/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -28.81 | 18/156 | — | — | -34.72/-32.36 | 描述性 |
| Fast-WAM/300/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -6.47 | 25/56 | — | — | -11.51/-9.39 | 描述性 |
| Fast-WAM/300/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -6.00 | 13/43 | — | — | -10.02/-8.40 | 描述性 |
| Fast-WAM/300/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -3.00 | 15/30 | — | — | -6.77/-5.20 | 描述性 |
| Fast-WAM/300/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -6.13 | 12/42 | — | — | -10.16/-8.59 | 描述性 |
| Fast-WAM/300/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -3.27 | 14/30 | — | — | -7.07/-5.52 | 描述性 |
| Fast-WAM/300/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -14.40 | 24/96 | — | — | -19.83/-17.80 | 描述性 |
| Fast-WAM/300/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -4.20 | 31/52 | — | — | -9.16/-7.20 | 描述性 |
| Fast-WAM/300/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 14.60 | 96/23 | — | — | 9.02/11.40 | 描述性 |
| Fast-WAM/300/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -14.20 | 24/92 | — | — | -19.79/-17.75 | 描述性 |
| Fast-WAM/300/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -3.97 | 29/48 | — | — | -8.97/-6.89 | 描述性 |
| Fast-WAM/300/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 14.61 | 92/22 | — | — | 8.90/11.27 | 描述性 |
| Fast-WAM/300/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.80 | 15/34 | — | — | -7.68/-6.00 | 描述性 |
| Fast-WAM/300/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -2.20 | 12/23 | — | — | -5.58/-4.20 | 描述性 |
| Fast-WAM/300/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 2.20 | 21/10 | — | — | -1.03/0.40 | 描述性 |
| Fast-WAM/300/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.68 | 15/33 | — | — | -7.62/-5.93 | 描述性 |
| Fast-WAM/300/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -2.25 | 12/23 | — | — | -5.70/-4.29 | 描述性 |
| Fast-WAM/300/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 2.45 | 21/9 | — | — | -0.79/0.61 | 描述性 |
| Fast-WAM/300/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -24.40 | 25/147 | — | — | -30.35/-28.00 | 描述性 |
| Fast-WAM/300/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -3.20 | 39/55 | — | — | -8.47/-6.40 | 描述性 |
| Fast-WAM/300/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 4.60 | 47/24 | — | — | -0.07/2.00 | 描述性 |
| Fast-WAM/300/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -24.01 | 25/140 | — | — | -30.12/-27.56 | 描述性 |
| Fast-WAM/300/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -3.76 | 36/54 | — | — | -9.13/-6.89 | 描述性 |
| Fast-WAM/300/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 4.80 | 45/22 | — | — | 0.06/2.09 | 描述性 |
| Fast-WAM/300/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.40 | 13/30 | — | — | -7.06/-5.40 | 描述性 |
| Fast-WAM/300/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.00 | 16/21 | — | — | -4.51/-3.00 | 描述性 |
| Fast-WAM/300/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.60 | 26/13 | — | — | -0.98/0.60 | 描述性 |
| Fast-WAM/300/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.48 | 13/30 | — | — | -7.22/-5.73 | 描述性 |
| Fast-WAM/300/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.02 | 16/21 | — | — | -4.61/-3.07 | 描述性 |
| Fast-WAM/300/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.66 | 25/12 | — | — | -0.91/0.61 | 描述性 |
| Fast-WAM/300/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -10.40 | 26/78 | — | — | -15.66/-13.60 | 描述性 |
| Fast-WAM/300/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -2.20 | 32/43 | — | — | -6.99/-5.00 | 描述性 |
| Fast-WAM/300/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 4.00 | 51/31 | — | — | -1.00/1.20 | 描述性 |
| Fast-WAM/300/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 14.00 | 91/21 | — | — | 8.57/11.00 | 描述性 |
| Fast-WAM/300/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -10.23 | 25/74 | — | — | -15.60/-13.36 | 描述性 |
| Fast-WAM/300/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -2.09 | 30/40 | — | — | -6.93/-4.80 | 描述性 |
| Fast-WAM/300/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 3.97 | 49/30 | — | — | -1.16/1.04 | 描述性 |
| Fast-WAM/300/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 13.78 | 86/20 | — | — | 8.24/10.65 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -1.60 | 15/23 | — | — | -5.13/-3.60 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.00 | 17/22 | — | — | -4.59/-3.00 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.20 | 21/10 | — | — | -1.03/0.40 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.80 | 20/11 | — | — | -1.44/0.00 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -1.43 | 15/22 | — | — | -5.01/-3.48 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.82 | 17/21 | — | — | -4.45/-2.86 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.25 | 21/10 | — | — | -1.05/0.41 | 描述性 |
| Fast-WAM/300/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 2.04 | 20/10 | — | — | -1.21/0.20 | 描述性 |
| Fast-WAM/400/c/libero_10 | full | control | — | PAIRED | 500 | 500 | -26.80 | 19/153 | — | — | -32.54/-30.40 | 描述性 |
| Fast-WAM/400/c/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -4.20 | 26/47 | — | — | -8.88/-7.00 | 描述性 |
| Fast-WAM/400/c/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -26.51 | 19/146 | — | — | -32.41/-30.06 | 描述性 |
| Fast-WAM/400/c/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -4.18 | 25/45 | — | — | -8.96/-7.10 | 描述性 |
| Fast-WAM/400/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -5.40 | 15/42 | — | — | -9.50/-7.80 | 描述性 |
| Fast-WAM/400/c/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -2.40 | 16/28 | — | — | -6.15/-4.60 | 描述性 |
| Fast-WAM/400/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -5.52 | 14/41 | — | — | -9.63/-7.98 | 描述性 |
| Fast-WAM/400/c/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -2.66 | 15/28 | — | — | -6.44/-4.91 | 描述性 |
| Fast-WAM/400/cg/libero_10 | full | control | — | PAIRED | 500 | 500 | -12.00 | 27/87 | — | — | -17.42/-15.40 | 描述性 |
| Fast-WAM/400/cg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | -1.80 | 32/41 | — | — | -6.54/-4.60 | 描述性 |
| Fast-WAM/400/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 14.80 | 92/18 | — | — | 9.47/11.80 | 描述性 |
| Fast-WAM/400/cg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -12.11 | 26/84 | — | — | -17.67/-15.45 | 描述性 |
| Fast-WAM/400/cg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | -1.88 | 30/39 | — | — | -6.70/-4.80 | 描述性 |
| Fast-WAM/400/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 14.41 | 85/16 | — | — | 9.06/11.27 | 描述性 |
| Fast-WAM/400/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -2.60 | 15/28 | — | — | -6.30/-4.80 | 描述性 |
| Fast-WAM/400/cg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.00 | 18/23 | — | — | -4.68/-3.00 | 描述性 |
| Fast-WAM/400/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 2.80 | 27/13 | — | — | -0.82/0.80 | 描述性 |
| Fast-WAM/400/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -2.45 | 15/27 | — | — | -6.21/-4.51 | 描述性 |
| Fast-WAM/400/cg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.02 | 18/23 | — | — | -4.78/-3.07 | 描述性 |
| Fast-WAM/400/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 3.07 | 27/12 | — | — | -0.58/1.02 | 描述性 |
| Fast-WAM/400/cr/libero_10 | full | control | — | PAIRED | 500 | 500 | -20.40 | 23/125 | — | — | -26.10/-23.80 | 描述性 |
| Fast-WAM/400/cr/libero_10 | full | N500 | — | PAIRED | 500 | 500 | 0.80 | 36/32 | — | — | -3.82/-1.80 | 描述性 |
| Fast-WAM/400/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 6.40 | 56/24 | — | — | 1.51/3.60 | 描述性 |
| Fast-WAM/400/cr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -19.83 | 23/118 | — | — | -25.68/-23.17 | 描述性 |
| Fast-WAM/400/cr/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | 0.42 | 33/31 | — | — | -4.27/-2.30 | 描述性 |
| Fast-WAM/400/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 6.68 | 55/23 | — | — | 1.65/3.76 | 描述性 |
| Fast-WAM/400/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.20 | 15/31 | — | — | -7.00/-5.40 | 描述性 |
| Fast-WAM/400/cr/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -0.80 | 16/20 | — | — | -4.27/-2.80 | 描述性 |
| Fast-WAM/400/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.20 | 27/16 | — | — | -1.56/0.00 | 描述性 |
| Fast-WAM/400/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -3.07 | 15/30 | — | — | -6.92/-5.32 | 描述性 |
| Fast-WAM/400/cr/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -0.61 | 16/19 | — | — | -4.12/-2.66 | 描述性 |
| Fast-WAM/400/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.45 | 27/15 | — | — | -1.34/0.41 | 描述性 |
| Fast-WAM/400/crg/libero_10 | full | control | — | PAIRED | 500 | 500 | -6.80 | 29/63 | — | — | -11.91/-10.00 | 描述性 |
| Fast-WAM/400/crg/libero_10 | full | N500 | — | PAIRED | 500 | 500 | 1.40 | 36/29 | — | — | -3.13/-1.20 | 描述性 |
| Fast-WAM/400/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.20 | 48/22 | — | — | 0.58/2.60 | 描述性 |
| Fast-WAM/400/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 13.60 | 87/19 | — | — | 8.31/10.60 | 描述性 |
| Fast-WAM/400/crg/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -6.47 | 29/60 | — | — | -11.74/-9.60 | 描述性 |
| Fast-WAM/400/crg/libero_10 | sensitivity | N500 | — | PAIRED | 479 | 479 | 1.67 | 36/28 | — | — | -3.02/-1.04 | 描述性 |
| Fast-WAM/400/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 5.64 | 48/21 | — | — | 0.86/2.92 | 描述性 |
| Fast-WAM/400/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 13.36 | 82/18 | — | — | 7.98/10.44 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -2.00 | 16/26 | — | — | -5.69/-4.20 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | full | N500 | — | PAIRED | 500 | 500 | -1.40 | 14/21 | — | — | -4.81/-3.40 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.60 | 20/17 | — | — | -2.93/-1.40 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.20 | 20/14 | — | — | -2.19/-0.60 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -1.84 | 16/25 | — | — | -5.57/-4.09 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | sensitivity | N500 | — | PAIRED | 489 | 489 | -1.23 | 14/20 | — | — | -4.67/-3.27 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.61 | 20/17 | — | — | -2.99/-1.43 | 描述性 |
| Fast-WAM/400/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 1.23 | 20/14 | — | — | -2.24/-0.61 | 描述性 |
| Fast-WAM/500/c/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -22.60 | 22/135 | -31.81 | -29.00 | -28.34/-26.00 | 未证明 |
| Fast-WAM/500/c/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -22.34 | 22/129 | -31.80 | -29.02 | -28.24/-25.89 | 描述性 |
| Fast-WAM/500/c/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -3.00 | 15/30 | -9.08 | -7.00 | -6.77/-5.20 | 未证明 |
| Fast-WAM/500/c/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -2.86 | 15/29 | -9.03 | -6.85 | -6.68/-5.11 | 描述性 |
| Fast-WAM/500/cg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -10.20 | 27/78 | -18.74 | -16.20 | -15.49/-13.60 | 未证明 |
| Fast-WAM/500/cg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 12.40 | 83/21 | — | — | 7.10/9.40 | 描述性 |
| Fast-WAM/500/cg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -10.23 | 26/75 | -18.97 | -16.08 | -15.65/-13.57 | 描述性 |
| Fast-WAM/500/cg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 12.11 | 78/20 | — | — | 6.71/9.19 | 描述性 |
| Fast-WAM/500/cg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -1.60 | 17/25 | -7.58 | -5.40 | -5.30/-3.60 | 未证明 |
| Fast-WAM/500/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.40 | 18/11 | — | — | -1.75/-0.40 | 描述性 |
| Fast-WAM/500/cg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -1.43 | 17/24 | -7.49 | -5.32 | -5.18/-3.68 | 描述性 |
| Fast-WAM/500/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 1.43 | 18/11 | — | — | -1.79/-0.41 | 描述性 |
| Fast-WAM/500/cr/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -21.20 | 24/130 | -30.48 | -27.20 | -26.98/-24.60 | 未证明 |
| Fast-WAM/500/cr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.40 | 47/40 | — | — | -3.75/-1.60 | 描述性 |
| Fast-WAM/500/cr/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -20.25 | 23/120 | -29.66 | -26.41 | -26.12/-23.59 | 描述性 |
| Fast-WAM/500/cr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.09 | 45/35 | — | — | -3.08/-0.84 | 描述性 |
| Fast-WAM/500/cr/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -2.40 | 14/26 | -8.20 | -6.20 | -5.99/-4.40 | 未证明 |
| Fast-WAM/500/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.60 | 19/16 | — | — | -2.84/-1.40 | 描述性 |
| Fast-WAM/500/cr/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -2.45 | 14/26 | -8.38 | -6.34 | -6.12/-4.50 | 描述性 |
| Fast-WAM/500/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.41 | 18/16 | — | — | -3.06/-1.43 | 描述性 |
| Fast-WAM/500/crg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -8.20 | 25/66 | -16.31 | -13.60 | -13.22/-11.20 | 未证明 |
| Fast-WAM/500/crg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.00 | 49/39 | — | — | -3.17/-1.20 | 描述性 |
| Fast-WAM/500/crg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 13.00 | 92/27 | — | — | 7.34/10.00 | 描述性 |
| Fast-WAM/500/crg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -8.14 | 25/64 | -16.52 | -13.57 | -13.33/-11.27 | 描述性 |
| Fast-WAM/500/crg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.09 | 46/36 | — | — | -3.14/-0.84 | 描述性 |
| Fast-WAM/500/crg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 12.11 | 84/26 | — | — | 6.38/9.19 | 描述性 |
| Fast-WAM/500/crg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -0.60 | 18/21 | -6.43 | -4.20 | -4.20/-2.60 | 未证明 |
| Fast-WAM/500/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.00 | 21/16 | — | — | -2.53/-1.00 | 描述性 |
| Fast-WAM/500/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 1.80 | 19/10 | — | — | -1.34/0.00 | 描述性 |
| Fast-WAM/500/crg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -0.61 | 18/21 | -6.57 | -4.50 | -4.29/-2.66 | 描述性 |
| Fast-WAM/500/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.82 | 20/16 | — | — | -2.74/-1.23 | 描述性 |
| Fast-WAM/500/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 1.84 | 19/10 | — | — | -1.37/0.00 | 描述性 |

## 两套件主判定

| family | model | group | NI（两套件） | 成本/采用 |
|---|---|---|---|---|
| F500 | DP | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | ch | PENDING | PENDING |
| F500 | DP | chr | PENDING | PENDING |
| F500 | DP | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | chg | PENDING | PENDING |
| F500 | DP | chrg | PENDING | PENDING |
| F500 | Fast-WAM | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | ch | PENDING | PENDING |
| F500 | Fast-WAM | chr | PENDING | PENDING |
| F500 | Fast-WAM | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | chg | PENDING | PENDING |
| F500 | Fast-WAM | chrg | PENDING | PENDING |
| F500 | Cosmos | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Cosmos | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Cosmos | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Cosmos | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | DP | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | DP | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | DP | ch | PENDING | PENDING |
| Fsize | DP | chr | PENDING | PENDING |
| Fsize | DP | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | DP | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | DP | chg | PENDING | PENDING |
| Fsize | DP | chrg | PENDING | PENDING |
| Fsize | Fast-WAM | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | ch | PENDING | PENDING |
| Fsize | Fast-WAM | chr | PENDING | PENDING |
| Fsize | Fast-WAM | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | chg | PENDING | PENDING |
| Fsize | Fast-WAM | chrg | PENDING | PENDING |
| Fsize | Cosmos | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Cosmos | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Cosmos | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Cosmos | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |

## 限制与待到达

NFE是动作头计算量，不含视觉/检索/修正器；零NFE不等于零耗时。service无完整可比证据时不宣称墙钟加速。
Cosmos旧控制含timan108与timan107 renderer；JSON按旧控制节点分层，不能把跨节点差异解释为纯缓存因果效应。
Q3实际校准来自各尺寸B成功episode LOEO，不是独立B验证录制；门不可用任务在JSON逐组列出。

Missing cells, resume segments, duplicate counts and missing evidence are listed in CACHE_RESULTS.json; CSV includes paired counts and raw cost totals. Curves leave missing outcomes blank.
This CPU analysis performs no GPU/model/campaign operations. Input files are read without hashes or seals.
