# R17 cache study: accepted episode snapshot

UTC: 2026-10-11T05:00:57.646950+00:00. Selected sizes: [50, 100, 200, 300, 400, 500].

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
| /data/mw_finetune/eval/fastwamc6066547406 | PARTIAL | RECEIPT.json:PARTIAL | 6380 | 6380 | 0 |
| /data/mw_finetune/eval/csztd_r2_c383_r2_e5b0 | PARTIAL | RECEIPT.json:PARTIAL | 2618 | 2618 | 0 |

51/240 cells have accepted outcomes; 49/240 have all 500 episodes.

## N500 vs inference control (full cohort)

| model | group | suite | n | SR% | pairs | Δ pp | g/l | F500 CP LB pp | status |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| DP | c | libero_10 | 500 | 64.80 | 500 | -14.80 | 46/120 | -25.08 | COMPLETE |
| DP | c | libero_spatial | 500 | 94.60 | 500 | 3.40 | 30/13 | -2.64 | COMPLETE |
| DP | cg | libero_10 | 500 | 87.40 | 500 | 7.80 | 76/37 | -1.47 | COMPLETE |
| DP | cg | libero_spatial | 500 | 93.00 | 500 | 1.80 | 31/22 | -4.90 | COMPLETE |
| DP | ch | libero_10 | 500 | 84.80 | 500 | 5.20 | 73/47 | -4.37 | COMPLETE |
| DP | ch | libero_spatial | 500 | 91.40 | 500 | 0.20 | 28/27 | -6.61 | COMPLETE |
| DP | chg | libero_10 | 500 | 87.40 | 500 | 7.80 | 79/40 | -1.68 | COMPLETE |
| DP | chg | libero_spatial | 500 | 92.00 | 500 | 0.80 | 29/25 | -5.96 | COMPLETE |
| DP | chr | libero_10 | 500 | 87.20 | 500 | 7.60 | 76/38 | -1.71 | COMPLETE |
| DP | chr | libero_spatial | 500 | 93.80 | 500 | 2.60 | 32/19 | -3.97 | COMPLETE |
| DP | chrg | libero_10 | 500 | 88.60 | 500 | 9.00 | 80/35 | -0.30 | COMPLETE |
| DP | chrg | libero_spatial | 500 | 94.00 | 500 | 2.80 | 31/17 | -3.59 | COMPLETE |
| DP | cr | libero_10 | 500 | 68.20 | 500 | -11.40 | 46/103 | -21.41 | COMPLETE |
| DP | cr | libero_spatial | 500 | 93.60 | 500 | 2.40 | 30/18 | -4.00 | COMPLETE |
| DP | crg | libero_10 | 500 | 88.80 | 500 | 9.20 | 78/32 | 0.10 | COMPLETE |
| DP | crg | libero_spatial | 500 | 94.00 | 500 | 2.80 | 33/19 | -3.83 | COMPLETE |
| Fast-WAM | c | libero_10 | 500 | 69.00 | 500 | -22.60 | 22/135 | -31.81 | COMPLETE |
| Fast-WAM | c | libero_spatial | 500 | 87.80 | 500 | -3.00 | 15/30 | -9.08 | COMPLETE |
| Fast-WAM | cg | libero_10 | 500 | 81.40 | 500 | -10.20 | 27/78 | -18.74 | COMPLETE |
| Fast-WAM | cg | libero_spatial | 500 | 89.20 | 500 | -1.60 | 17/25 | -7.58 | COMPLETE |
| Fast-WAM | ch | libero_10 | 500 | 75.60 | 500 | -16.00 | 18/98 | -24.37 | COMPLETE |
| Fast-WAM | ch | libero_spatial | 500 | 89.20 | 500 | -1.60 | 17/25 | -7.58 | COMPLETE |
| Fast-WAM | chg | libero_10 | 500 | 78.20 | 500 | -13.40 | 19/86 | -21.60 | COMPLETE |
| Fast-WAM | chg | libero_spatial | 500 | 89.20 | 500 | -1.60 | 18/26 | -7.71 | COMPLETE |
| Fast-WAM | chr | libero_10 | 500 | 76.80 | 500 | -14.80 | 25/99 | -23.66 | COMPLETE |
| Fast-WAM | chr | libero_spatial | 500 | 89.40 | 500 | -1.40 | 17/24 | -7.33 | COMPLETE |
| Fast-WAM | chrg | libero_10 | 500 | 80.00 | 500 | -11.60 | 24/82 | -20.05 | COMPLETE |
| Fast-WAM | chrg | libero_spatial | 500 | 88.20 | 500 | -2.60 | 16/29 | -8.71 | COMPLETE |
| Fast-WAM | cr | libero_10 | 500 | 70.40 | 500 | -21.20 | 24/130 | -30.48 | COMPLETE |
| Fast-WAM | cr | libero_spatial | 500 | 88.40 | 500 | -2.40 | 14/26 | -8.20 | COMPLETE |
| Fast-WAM | crg | libero_10 | 500 | 83.40 | 500 | -8.20 | 25/66 | -16.31 | COMPLETE |
| Fast-WAM | crg | libero_spatial | 500 | 90.20 | 500 | -0.60 | 18/21 | -6.43 | COMPLETE |

## Sizes 50–400 so far (full cohort)

| model | group | suite | N50 n / SR% | N100 n / SR% | N200 n / SR% | N300 n / SR% | N400 n / SR% |
|---|---|---|---|---|---|---|---|
| DP | c | libero_spatial | 500 / 72.20 | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | c | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | cr | libero_spatial | 500 / 77.80 | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | cr | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | ch | libero_spatial | 500 / 69.60 | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | ch | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | chr | libero_spatial | 500 / 77.80 | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | chr | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | cg | libero_spatial | 500 / 83.80 | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | cg | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | crg | libero_spatial | 118 / 79.66 | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | crg | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | chg | libero_spatial | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | chg | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | chrg | libero_spatial | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| DP | chrg | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | c | libero_spatial | 500 / 78.60 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | c | libero_10 | 500 / 39.00 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | cr | libero_spatial | 500 / 77.00 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | cr | libero_10 | 500 / 45.00 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | ch | libero_spatial | 500 / 85.00 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | ch | libero_10 | 500 / 52.80 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | chr | libero_spatial | 500 / 84.60 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | chr | libero_10 | 500 / 60.60 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | cg | libero_spatial | 500 / 84.20 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | cg | libero_10 | 380 / 72.89 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | crg | libero_spatial | 500 / 85.60 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | crg | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | chg | libero_spatial | 500 / 85.60 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | chg | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | chrg | libero_spatial | 500 / 87.00 | 0 / — | 0 / — | 0 / — | 0 / — |
| Fast-WAM | chrg | libero_10 | 0 / — | 0 / — | 0 / — | 0 / — | 0 / — |

## SR and costs

| model | N | group | suite | cohort | status | n | SR% [CP95%] | cost n | fresh hit% | MISS/control | NFE/control | calls/ep |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| DP | 50 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | c | libero_spatial | full | COMPLETE | 500 | 72.20 [68.05,76.09] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | c | libero_spatial | sensitivity | COMPLETE | 489 | 72.39 [68.20,76.31] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | cg | libero_spatial | full | COMPLETE | 500 | 83.80 [80.27,86.92] | 500 | 85.80 | 0.02879 | 2.87925 | 3.62 |
| DP | 50 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 84.05 [80.50,87.18] | 489 | 85.58 | 0.02923 | 2.92342 | 3.67 |
| DP | 50 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | ch | libero_spatial | full | COMPLETE | 500 | 69.60 [65.36,73.61] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | ch | libero_spatial | sensitivity | COMPLETE | 489 | 69.33 [65.03,73.39] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chr | libero_spatial | full | COMPLETE | 500 | 77.80 [73.90,81.37] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | chr | libero_spatial | sensitivity | COMPLETE | 489 | 77.71 [73.76,81.32] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | cr | libero_spatial | full | COMPLETE | 500 | 77.80 [73.90,81.37] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 77.91 [73.97,81.51] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 50 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 50 | crg | libero_spatial | full | INTERIM | 118 | 79.66 [71.27,86.51] | 118 | 92.37 | 0.01544 | 1.54351 | 1.87 |
| DP | 50 | crg | libero_spatial | sensitivity | INTERIM | 116 | 80.17 [71.75,87.00] | 116 | 92.20 | 0.01576 | 1.57632 | 1.91 |
| DP | 100 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 100 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 200 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 300 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 400 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| DP | 500 | c | libero_10 | full | COMPLETE | 500 | 64.80 [60.44,68.99] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | c | libero_10 | sensitivity | COMPLETE | 479 | 65.97 [61.53,70.21] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | c | libero_spatial | full | COMPLETE | 500 | 94.60 [92.24,96.41] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | c | libero_spatial | sensitivity | COMPLETE | 489 | 94.48 [92.07,96.33] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | cg | libero_10 | full | COMPLETE | 500 | 87.40 [84.17,90.18] | 500 | 89.57 | 0.02099 | 2.09870 | 6.32 |
| DP | 500 | cg | libero_10 | sensitivity | COMPLETE | 479 | 87.27 [83.94,90.12] | 479 | 89.51 | 0.02111 | 2.11119 | 6.33 |
| DP | 500 | cg | libero_spatial | full | COMPLETE | 500 | 93.00 [90.40,95.08] | 500 | 92.03 | 0.01621 | 1.62066 | 1.80 |
| DP | 500 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 93.05 [90.42,95.14] | 489 | 92.10 | 0.01605 | 1.60527 | 1.78 |
| DP | 500 | ch | libero_10 | full | COMPLETE | 500 | 84.80 [81.35,87.83] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | ch | libero_10 | sensitivity | COMPLETE | 479 | 85.18 [81.67,88.24] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | ch | libero_spatial | full | COMPLETE | 500 | 91.40 [88.59,93.71] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | ch | libero_spatial | sensitivity | COMPLETE | 489 | 91.62 [88.80,93.92] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | chg | libero_10 | full | COMPLETE | 500 | 87.40 [84.17,90.18] | 500 | 93.30 | 0.00702 | 0.70248 | 2.01 |
| DP | 500 | chg | libero_10 | sensitivity | COMPLETE | 479 | 87.89 [84.63,90.68] | 479 | 93.39 | 0.00693 | 0.69311 | 1.96 |
| DP | 500 | chg | libero_spatial | full | COMPLETE | 500 | 92.00 [89.27,94.22] | 500 | 94.24 | 0.00605 | 0.60534 | 0.67 |
| DP | 500 | chg | libero_spatial | sensitivity | COMPLETE | 489 | 92.23 [89.49,94.44] | 489 | 94.49 | 0.00579 | 0.57861 | 0.64 |
| DP | 500 | chr | libero_10 | full | COMPLETE | 500 | 87.20 [83.95,90.00] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | chr | libero_10 | sensitivity | COMPLETE | 479 | 87.47 [84.17,90.30] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | chr | libero_spatial | full | COMPLETE | 500 | 93.80 [91.31,95.75] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | chr | libero_spatial | sensitivity | COMPLETE | 489 | 93.87 [91.36,95.82] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| DP | 500 | chrg | libero_10 | full | COMPLETE | 500 | 88.60 [85.48,91.25] | 500 | 93.54 | 0.00678 | 0.67807 | 1.93 |
| DP | 500 | chrg | libero_10 | sensitivity | COMPLETE | 479 | 88.73 [85.55,91.42] | 479 | 93.46 | 0.00686 | 0.68649 | 1.94 |
| DP | 500 | chrg | libero_spatial | full | COMPLETE | 500 | 94.00 [91.55,95.92] | 500 | 95.24 | 0.00499 | 0.49901 | 0.54 |
| DP | 500 | chrg | libero_spatial | sensitivity | COMPLETE | 489 | 94.07 [91.59,95.99] | 489 | 95.31 | 0.00491 | 0.49148 | 0.53 |
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
| Fast-WAM | 50 | cg | libero_10 | full | INTERIM | 380 | 72.89 [68.13,77.30] | 380 | 80.88 | 0.03841 | 0.38414 | 12.35 |
| Fast-WAM | 50 | cg | libero_10 | sensitivity | INTERIM | 364 | 73.08 [68.21,77.57] | 364 | 80.88 | 0.03842 | 0.38419 | 12.34 |
| Fast-WAM | 50 | cg | libero_spatial | full | COMPLETE | 500 | 84.20 [80.70,87.29] | 500 | 92.20 | 0.01581 | 0.15809 | 1.98 |
| Fast-WAM | 50 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 84.25 [80.72,87.37] | 489 | 92.10 | 0.01601 | 0.16007 | 2.00 |
| Fast-WAM | 50 | ch | libero_10 | full | COMPLETE | 500 | 52.80 [48.32,57.25] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | ch | libero_10 | sensitivity | COMPLETE | 479 | 53.03 [48.45,57.57] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | ch | libero_spatial | full | COMPLETE | 500 | 85.00 [81.56,88.02] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | ch | libero_spatial | sensitivity | COMPLETE | 489 | 84.87 [81.38,87.93] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 50 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 50 | chg | libero_spatial | full | COMPLETE | 500 | 85.60 [82.21,88.56] | 500 | 93.50 | 0.00686 | 0.06855 | 0.85 |
| Fast-WAM | 50 | chg | libero_spatial | sensitivity | COMPLETE | 489 | 85.48 [82.04,88.48] | 489 | 93.40 | 0.00697 | 0.06967 | 0.87 |
| Fast-WAM | 50 | chr | libero_10 | full | COMPLETE | 500 | 60.60 [56.16,64.91] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | chr | libero_10 | sensitivity | COMPLETE | 479 | 61.59 [57.07,65.96] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | chr | libero_spatial | full | COMPLETE | 500 | 84.60 [81.13,87.65] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | chr | libero_spatial | sensitivity | COMPLETE | 489 | 84.46 [80.94,87.56] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 50 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 50 | chrg | libero_spatial | full | COMPLETE | 500 | 87.00 [83.73,89.82] | 500 | 93.66 | 0.00670 | 0.06702 | 0.82 |
| Fast-WAM | 50 | chrg | libero_spatial | sensitivity | COMPLETE | 489 | 87.12 [83.82,89.96] | 489 | 93.57 | 0.00680 | 0.06802 | 0.83 |
| Fast-WAM | 50 | cr | libero_10 | full | COMPLETE | 500 | 45.00 [40.58,49.48] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | cr | libero_10 | sensitivity | COMPLETE | 479 | 45.72 [41.19,50.30] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | cr | libero_spatial | full | COMPLETE | 500 | 77.00 [73.06,80.62] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | cr | libero_spatial | sensitivity | COMPLETE | 489 | 76.69 [72.68,80.37] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 50 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 50 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 50 | crg | libero_spatial | full | COMPLETE | 500 | 85.60 [82.21,88.56] | 500 | 91.23 | 0.01778 | 0.17777 | 2.19 |
| Fast-WAM | 50 | crg | libero_spatial | sensitivity | COMPLETE | 489 | 85.69 [82.26,88.67] | 489 | 91.12 | 0.01799 | 0.17993 | 2.21 |
| Fast-WAM | 100 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 100 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 200 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 300 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | c | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | c | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | c | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | c | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | ch | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | ch | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | ch | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | ch | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chrg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chrg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chrg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | chrg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cr | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cr | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cr | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | cr | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | crg | libero_10 | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | crg | libero_10 | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | crg | libero_spatial | full | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 400 | crg | libero_spatial | sensitivity | INTERIM | 0 | — [—,—] | 0 | — | — | — | — |
| Fast-WAM | 500 | c | libero_10 | full | COMPLETE | 500 | 69.00 [64.74,73.03] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | c | libero_10 | sensitivity | COMPLETE | 479 | 69.31 [64.97,73.42] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | c | libero_spatial | full | COMPLETE | 500 | 87.80 [84.61,90.54] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | c | libero_spatial | sensitivity | COMPLETE | 489 | 88.55 [85.39,91.23] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | cg | libero_10 | full | COMPLETE | 500 | 81.40 [77.71,84.72] | 500 | 85.24 | 0.02968 | 0.29684 | 9.09 |
| Fast-WAM | 500 | cg | libero_10 | sensitivity | COMPLETE | 479 | 81.42 [77.64,84.80] | 479 | 85.22 | 0.02972 | 0.29719 | 9.09 |
| Fast-WAM | 500 | cg | libero_spatial | full | COMPLETE | 500 | 89.20 [86.14,91.78] | 500 | 91.62 | 0.01701 | 0.17012 | 2.02 |
| Fast-WAM | 500 | cg | libero_spatial | sensitivity | COMPLETE | 489 | 89.98 [86.97,92.49] | 489 | 91.92 | 0.01641 | 0.16405 | 1.94 |
| Fast-WAM | 500 | ch | libero_10 | full | COMPLETE | 500 | 75.60 [71.59,79.30] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | ch | libero_10 | sensitivity | COMPLETE | 479 | 75.78 [71.69,79.55] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | ch | libero_spatial | full | COMPLETE | 500 | 89.20 [86.14,91.78] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | ch | libero_spatial | sensitivity | COMPLETE | 489 | 89.37 [86.29,91.96] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | chg | libero_10 | full | COMPLETE | 500 | 78.20 [74.32,81.74] | 500 | 88.52 | 0.01184 | 0.11840 | 3.66 |
| Fast-WAM | 500 | chg | libero_10 | sensitivity | COMPLETE | 479 | 78.29 [74.32,81.90] | 479 | 88.50 | 0.01186 | 0.11858 | 3.66 |
| Fast-WAM | 500 | chg | libero_spatial | full | COMPLETE | 500 | 89.20 [86.14,91.78] | 500 | 93.89 | 0.00638 | 0.06377 | 0.75 |
| Fast-WAM | 500 | chg | libero_spatial | sensitivity | COMPLETE | 489 | 89.57 [86.52,92.14] | 489 | 94.03 | 0.00623 | 0.06225 | 0.73 |
| Fast-WAM | 500 | chr | libero_10 | full | COMPLETE | 500 | 76.80 [72.85,80.43] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | chr | libero_10 | sensitivity | COMPLETE | 479 | 76.83 [72.78,80.53] | 479 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | chr | libero_spatial | full | COMPLETE | 500 | 89.40 [86.36,91.96] | 500 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | chr | libero_spatial | sensitivity | COMPLETE | 489 | 89.98 [86.97,92.49] | 489 | 100.00 | 0.00000 | 0.00000 | 0.00 |
| Fast-WAM | 500 | chrg | libero_10 | full | COMPLETE | 500 | 80.00 [76.22,83.42] | 500 | 90.22 | 0.01012 | 0.10115 | 3.08 |
| Fast-WAM | 500 | chrg | libero_10 | sensitivity | COMPLETE | 479 | 79.96 [76.09,83.45] | 479 | 90.15 | 0.01019 | 0.10185 | 3.10 |
| Fast-WAM | 500 | chrg | libero_spatial | full | COMPLETE | 500 | 88.20 [85.04,90.89] | 500 | 93.87 | 0.00641 | 0.06414 | 0.76 |
| Fast-WAM | 500 | chrg | libero_spatial | sensitivity | COMPLETE | 489 | 88.75 [85.61,91.41] | 489 | 94.01 | 0.00626 | 0.06261 | 0.74 |
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
| DP/50/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/c/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/c/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/c/libero_spatial | full | control | — | PAIRED | 500 | 500 | -19.00 | 36/131 | — | — | -25.19/-22.80 | 描述性 |
| DP/50/c/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -22.40 | 15/127 | -31.00 | -29.00 | -27.77/-25.80 | 未证明 |
| DP/50/c/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -19.02 | 34/127 | — | — | -25.24/-22.70 | 描述性 |
| DP/50/c/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -22.09 | 15/123 | -30.79 | -28.02 | -27.52/-25.56 | 描述性 |
| DP/50/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cg/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cg/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -7.40 | 36/73 | — | — | -12.91/-10.60 | 描述性 |
| DP/50/cg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -9.20 | 25/71 | -17.44 | -14.80 | -14.31/-12.20 | 未证明 |
| DP/50/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 11.60 | 76/18 | — | — | 6.54/8.60 | 描述性 |
| DP/50/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -7.36 | 34/70 | — | — | -12.87/-10.43 | 描述性 |
| DP/50/cg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -9.00 | 24/68 | -17.26 | -14.52 | -14.12/-11.86 | 描述性 |
| DP/50/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 11.66 | 75/18 | — | — | 6.50/8.59 | 描述性 |
| DP/50/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/ch/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/ch/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/ch/libero_spatial | full | control | — | PAIRED | 500 | 500 | -21.60 | 35/143 | — | — | -27.86/-25.20 | 描述性 |
| DP/50/ch/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -21.80 | 26/135 | -31.26 | -27.80 | -27.70/-25.20 | 未证明 |
| DP/50/ch/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | -2.60 | 64/77 | — | — | -8.87/-6.40 | 描述性 |
| DP/50/ch/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -22.09 | 33/141 | — | — | -28.38/-25.77 | 描述性 |
| DP/50/ch/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -22.29 | 25/134 | -31.85 | -28.43 | -28.25/-25.77 | 描述性 |
| DP/50/ch/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | -3.07 | 61/76 | — | — | -9.38/-6.75 | 描述性 |
| DP/50/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -13.40 | 34/101 | — | — | -19.24/-16.80 | 描述性 |
| DP/50/chr/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -16.00 | 20/100 | -24.55 | -21.80 | -21.32/-19.20 | 未证明 |
| DP/50/chr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 8.20 | 76/35 | — | — | 2.57/4.80 | 描述性 |
| DP/50/chr/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 0.00 | 59/59 | — | — | -5.85/-3.40 | 描述性 |
| DP/50/chr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -13.70 | 32/99 | — | — | -19.57/-17.18 | 描述性 |
| DP/50/chr/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -16.16 | 20/99 | -24.86 | -22.09 | -21.57/-19.22 | 描述性 |
| DP/50/chr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 8.38 | 75/34 | — | — | 2.68/5.11 | 描述性 |
| DP/50/chr/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | -0.20 | 57/58 | — | — | -6.11/-3.68 | 描述性 |
| DP/50/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cr/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cr/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/cr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -13.40 | 37/104 | — | — | -19.36/-17.01 | 描述性 |
| DP/50/cr/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -15.80 | 21/100 | -24.42 | -22.20 | -21.16/-19.20 | 未证明 |
| DP/50/cr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 5.60 | 80/52 | — | — | -0.52/1.80 | 描述性 |
| DP/50/cr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -13.50 | 35/101 | — | — | -19.48/-17.18 | 描述性 |
| DP/50/cr/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -15.54 | 21/97 | -24.27 | -21.68 | -20.97/-19.02 | 描述性 |
| DP/50/cr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 5.52 | 77/50 | — | — | -0.63/1.84 | 描述性 |
| DP/50/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/50/crg/libero_spatial | full | control | — | INTERIM | 118 | 118 | -14.41 | 7/24 | — | — | -26.32/-21.19 | INTERIM |
| DP/50/crg/libero_spatial | full | N500 | Fsize | INTERIM | 118 | 118 | -15.25 | 5/23 | -32.80 | -27.12 | -26.41/-21.19 | INTERIM |
| DP/50/crg/libero_spatial | full | corrector_ablation | — | INTERIM | 118 | 118 | 10.17 | 17/5 | — | — | -0.99/4.24 | INTERIM |
| DP/50/crg/libero_spatial | full | gate_ablation | — | INTERIM | 118 | 118 | 6.78 | 11/3 | — | — | -2.50/1.69 | INTERIM |
| DP/50/crg/libero_spatial | sensitivity | control | — | INTERIM | 116 | 116 | -13.79 | 7/23 | — | — | -25.79/-20.69 | INTERIM |
| DP/50/crg/libero_spatial | sensitivity | N500 | Fsize | INTERIM | 116 | 116 | -14.66 | 5/22 | -32.32 | -25.86 | -25.88/-20.69 | INTERIM |
| DP/50/crg/libero_spatial | sensitivity | corrector_ablation | — | INTERIM | 116 | 116 | 10.34 | 17/5 | — | — | -1.00/4.31 | INTERIM |
| DP/50/crg/libero_spatial | sensitivity | gate_ablation | — | INTERIM | 116 | 116 | 6.90 | 11/3 | — | — | -2.54/1.72 | INTERIM |
| DP/100/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/100/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/200/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/300/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| DP/400/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
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
| DP/500/ch/libero_10 | full | control | F500 | PAIRED | 500 | 500 | 5.20 | 73/47 | -4.37 | -1.20 | -0.68/1.60 | 未证明 |
| DP/500/ch/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | 20.00 | 116/16 | — | — | 14.42/16.80 | 描述性 |
| DP/500/ch/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | 4.38 | 66/45 | -5.26 | -1.98 | -1.55/0.84 | 描述性 |
| DP/500/ch/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | 19.21 | 107/15 | — | — | 13.57/15.87 | 描述性 |
| DP/500/ch/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 0.20 | 28/27 | -6.61 | -4.20 | -4.01/-2.20 | 未证明 |
| DP/500/ch/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | -3.20 | 12/28 | — | — | -6.75/-5.20 | 描述性 |
| DP/500/ch/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 0.20 | 26/25 | -6.53 | -3.99 | -3.95/-2.25 | 描述性 |
| DP/500/ch/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | -2.86 | 12/26 | — | — | -6.42/-4.91 | 描述性 |
| DP/500/chg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | 7.80 | 79/40 | -1.68 | 1.30 | 1.98/4.40 | PASS |
| DP/500/chg/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | 0.00 | 30/30 | — | — | -4.37/-2.60 | 描述性 |
| DP/500/chg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 2.60 | 26/13 | — | — | -0.98/0.60 | 描述性 |
| DP/500/chg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | 7.10 | 72/38 | -2.46 | 0.84 | 1.22/3.55 | 描述性 |
| DP/500/chg/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | 0.63 | 30/27 | — | — | -3.83/-1.88 | 描述性 |
| DP/500/chg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 2.71 | 25/12 | — | — | -0.93/0.63 | 描述性 |
| DP/500/chg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 0.80 | 29/25 | -5.96 | -3.60 | -3.37/-1.60 | 未证明 |
| DP/500/chg/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | -1.00 | 17/22 | — | — | -4.59/-3.00 | 描述性 |
| DP/500/chg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.60 | 8/5 | — | — | -1.62/-0.60 | 描述性 |
| DP/500/chg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 0.82 | 27/23 | -5.86 | -3.37 | -3.30/-1.43 | 描述性 |
| DP/500/chg/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | -0.82 | 16/20 | — | — | -4.36/-2.86 | 描述性 |
| DP/500/chg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.61 | 8/5 | — | — | -1.66/-0.61 | 描述性 |
| DP/500/chr/libero_10 | full | control | F500 | PAIRED | 500 | 500 | 7.60 | 76/38 | -1.71 | 1.60 | 1.88/4.20 | PASS |
| DP/500/chr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.40 | 39/27 | — | — | -2.15/-0.20 | 描述性 |
| DP/500/chr/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | 19.00 | 110/15 | — | — | 13.54/15.80 | 描述性 |
| DP/500/chr/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | 6.68 | 68/36 | -2.66 | 0.42 | 0.94/3.34 | 描述性 |
| DP/500/chr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 2.30 | 37/26 | — | — | -2.36/-0.42 | 描述性 |
| DP/500/chr/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | 18.16 | 102/15 | — | — | 12.60/14.82 | 描述性 |
| DP/500/chr/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 2.60 | 32/19 | -3.97 | -1.60 | -1.45/0.40 | 未证明 |
| DP/500/chr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.40 | 22/10 | — | — | -0.87/0.60 | 描述性 |
| DP/500/chr/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 0.20 | 20/19 | — | — | -3.41/-1.80 | 描述性 |
| DP/500/chr/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 2.45 | 30/18 | -4.08 | -1.64 | -1.58/0.20 | 描述性 |
| DP/500/chr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 2.25 | 21/10 | — | — | -1.05/0.41 | 描述性 |
| DP/500/chr/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 0.41 | 20/18 | — | — | -3.24/-1.64 | 描述性 |
| DP/500/chrg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | 9.00 | 80/35 | -0.30 | 3.00 | 3.29/5.60 | PASS |
| DP/500/chrg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.20 | 34/28 | — | — | -3.24/-1.40 | 描述性 |
| DP/500/chrg/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | -0.20 | 28/29 | — | — | -4.47/-2.60 | 描述性 |
| DP/500/chrg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 1.40 | 24/17 | — | — | -2.29/-0.60 | 描述性 |
| DP/500/chrg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | 7.93 | 72/34 | -1.45 | 2.09 | 2.17/4.59 | 描述性 |
| DP/500/chrg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 0.84 | 32/28 | — | — | -3.72/-1.67 | 描述性 |
| DP/500/chrg/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | -0.21 | 27/28 | — | — | -4.59/-2.71 | 描述性 |
| DP/500/chrg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 1.25 | 22/16 | — | — | -2.47/-0.84 | 描述性 |
| DP/500/chrg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | 2.80 | 31/17 | -3.59 | -1.10 | -1.14/0.60 | 未证明 |
| DP/500/chrg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 2.00 | 20/10 | — | — | -1.19/0.20 | 描述性 |
| DP/500/chrg/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 0.00 | 18/18 | — | — | -3.48/-2.00 | 描述性 |
| DP/500/chrg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.20 | 6/5 | — | — | -1.88/-0.80 | 描述性 |
| DP/500/chrg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | 2.66 | 29/16 | -3.68 | -1.23 | -1.25/0.41 | 描述性 |
| DP/500/chrg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.84 | 19/10 | — | — | -1.37/0.00 | 描述性 |
| DP/500/chrg/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 0.00 | 17/17 | — | — | -3.47/-1.84 | 描述性 |
| DP/500/chrg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.20 | 6/5 | — | — | -1.92/-0.82 | 描述性 |
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
| Fast-WAM/50/cg/libero_10 | full | control | — | INTERIM | 380 | 380 | -19.47 | 13/87 | — | — | -25.62/-23.16 | INTERIM |
| Fast-WAM/50/cg/libero_10 | full | N500 | Fsize | INTERIM | 380 | 380 | -10.79 | 27/68 | -21.41 | -17.89 | -17.39/-14.74 | INTERIM |
| Fast-WAM/50/cg/libero_10 | full | gate_ablation | — | INTERIM | 380 | 380 | 29.47 | 121/9 | — | — | 22.74/25.53 | INTERIM |
| Fast-WAM/50/cg/libero_10 | sensitivity | control | — | INTERIM | 364 | 364 | -19.51 | 13/84 | — | — | -25.84/-23.35 | INTERIM |
| Fast-WAM/50/cg/libero_10 | sensitivity | N500 | Fsize | INTERIM | 364 | 364 | -10.71 | 27/66 | -21.68 | -17.86 | -17.54/-14.84 | INTERIM |
| Fast-WAM/50/cg/libero_10 | sensitivity | gate_ablation | — | INTERIM | 364 | 364 | 28.85 | 114/9 | — | — | 21.94/24.73 | INTERIM |
| Fast-WAM/50/cg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -6.60 | 18/51 | — | — | -11.05/-9.20 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -5.00 | 21/46 | -12.21 | -9.70 | -9.46/-7.60 | 未证明 |
| Fast-WAM/50/cg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 5.60 | 44/16 | — | — | 1.32/3.20 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -7.16 | 16/51 | — | — | -11.60/-9.82 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -5.73 | 18/46 | -12.87 | -10.43 | -10.15/-8.38 | 描述性 |
| Fast-WAM/50/cg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 5.73 | 44/16 | — | — | 1.35/3.27 | 描述性 |
| Fast-WAM/50/ch/libero_10 | full | control | — | PAIRED | 500 | 500 | -38.80 | 14/208 | — | — | -44.52/-42.40 | 描述性 |
| Fast-WAM/50/ch/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -22.80 | 23/137 | -32.10 | -29.40 | -28.60/-26.41 | 未证明 |
| Fast-WAM/50/ch/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | 13.80 | 97/28 | — | — | 8.03/10.40 | 描述性 |
| Fast-WAM/50/ch/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -38.62 | 14/199 | — | — | -44.50/-42.38 | 描述性 |
| Fast-WAM/50/ch/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -22.76 | 22/131 | -32.25 | -29.65 | -28.68/-26.51 | 描述性 |
| Fast-WAM/50/ch/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | 13.15 | 91/28 | — | — | 7.24/9.81 | 描述性 |
| Fast-WAM/50/ch/libero_spatial | full | control | — | PAIRED | 500 | 500 | -5.80 | 28/57 | — | — | -10.76/-8.60 | 描述性 |
| Fast-WAM/50/ch/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -4.20 | 30/51 | -12.13 | -9.20 | -9.11/-7.00 | 未证明 |
| Fast-WAM/50/ch/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 6.40 | 57/25 | — | — | 1.46/3.60 | 描述性 |
| Fast-WAM/50/ch/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -6.54 | 25/57 | — | — | -11.50/-9.41 | 描述性 |
| Fast-WAM/50/ch/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -4.50 | 28/50 | -12.45 | -9.82 | -9.42/-7.36 | 描述性 |
| Fast-WAM/50/ch/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 6.34 | 56/25 | — | — | 1.31/3.48 | 描述性 |
| Fast-WAM/50/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -5.20 | 27/53 | — | — | -10.05/-8.00 | 描述性 |
| Fast-WAM/50/chg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -3.60 | 24/42 | -10.85 | -8.40 | -8.09/-6.20 | 未证明 |
| Fast-WAM/50/chg/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 1.40 | 40/33 | — | — | -3.37/-1.40 | 描述性 |
| Fast-WAM/50/chg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.60 | 28/25 | — | — | -3.54/-1.80 | 描述性 |
| Fast-WAM/50/chg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -5.93 | 24/53 | — | — | -10.77/-8.59 | 描述性 |
| Fast-WAM/50/chg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -4.09 | 22/42 | -11.36 | -9.10 | -8.59/-6.75 | 描述性 |
| Fast-WAM/50/chg/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 1.23 | 39/33 | — | — | -3.61/-1.64 | 描述性 |
| Fast-WAM/50/chg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.61 | 28/25 | — | — | -3.62/-1.84 | 描述性 |
| Fast-WAM/50/chr/libero_10 | full | control | — | PAIRED | 500 | 500 | -31.00 | 18/173 | — | — | -36.80/-34.60 | 描述性 |
| Fast-WAM/50/chr/libero_10 | full | N500 | Fsize | PAIRED | 500 | 500 | -16.20 | 35/116 | -25.90 | -23.00 | -22.23/-20.00 | 未证明 |
| Fast-WAM/50/chr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 7.80 | 65/26 | — | — | 2.65/4.80 | 描述性 |
| Fast-WAM/50/chr/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | 15.60 | 112/34 | — | — | 9.44/12.00 | 描述性 |
| Fast-WAM/50/chr/libero_10 | sensitivity | control | — | PAIRED | 479 | 479 | -30.06 | 18/162 | — | — | -36.01/-33.82 | 描述性 |
| Fast-WAM/50/chr/libero_10 | sensitivity | N500 | Fsize | PAIRED | 479 | 479 | -15.24 | 35/108 | -25.18 | -22.34 | -21.42/-19.00 | 描述性 |
| Fast-WAM/50/chr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 8.56 | 65/24 | — | — | 3.27/5.43 | 描述性 |
| Fast-WAM/50/chr/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | 15.87 | 109/33 | — | — | 9.54/12.11 | 描述性 |
| Fast-WAM/50/chr/libero_spatial | full | control | — | PAIRED | 500 | 500 | -6.20 | 28/59 | — | — | -11.20/-9.00 | 描述性 |
| Fast-WAM/50/chr/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -4.80 | 27/51 | -12.56 | -10.00 | -9.60/-7.40 | 未证明 |
| Fast-WAM/50/chr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | -0.40 | 35/37 | — | — | -5.13/-3.20 | 描述性 |
| Fast-WAM/50/chr/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 7.60 | 74/36 | — | — | 1.97/4.20 | 描述性 |
| Fast-WAM/50/chr/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -6.95 | 25/59 | — | — | -11.95/-9.82 | 描述性 |
| Fast-WAM/50/chr/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -5.52 | 24/51 | -13.26 | -10.43 | -10.31/-8.18 | 描述性 |
| Fast-WAM/50/chr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | -0.41 | 35/37 | — | — | -5.24/-3.27 | 描述性 |
| Fast-WAM/50/chr/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 7.77 | 73/35 | — | — | 2.07/4.50 | 描述性 |
| Fast-WAM/50/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/chrg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -3.80 | 26/45 | — | — | -8.43/-6.40 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -1.20 | 33/39 | -8.85 | -6.00 | -5.92/-3.80 | 未证明 |
| Fast-WAM/50/chrg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.40 | 35/28 | — | — | -3.07/-1.20 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 1.40 | 39/32 | — | — | -3.31/-1.40 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 2.40 | 29/17 | — | — | -1.47/0.20 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -4.29 | 24/45 | — | — | -8.95/-6.95 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -1.64 | 31/39 | -9.34 | -6.54 | -6.39/-4.29 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.64 | 35/27 | — | — | -2.90/-1.02 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 1.43 | 38/31 | — | — | -3.32/-1.43 | 描述性 |
| Fast-WAM/50/chrg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 2.66 | 29/16 | — | — | -1.25/0.41 | 描述性 |
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
| Fast-WAM/50/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_10 | full | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_10 | sensitivity | N500 | Fsize | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/50/crg/libero_spatial | full | control | — | PAIRED | 500 | 500 | -5.20 | 29/55 | — | — | -10.16/-8.00 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | full | N500 | Fsize | PAIRED | 500 | 500 | -4.60 | 29/52 | -12.51 | -9.50 | -9.50/-7.40 | 未证明 |
| Fast-WAM/50/crg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.40 | 44/37 | — | — | -3.59/-1.40 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 8.60 | 58/15 | — | — | 4.03/6.00 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | control | — | PAIRED | 489 | 489 | -5.73 | 26/54 | — | — | -10.66/-8.38 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | N500 | Fsize | PAIRED | 489 | 489 | -5.11 | 26/51 | -12.98 | -10.43 | -9.98/-7.77 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 1.43 | 42/35 | — | — | -3.55/-1.43 | 描述性 |
| Fast-WAM/50/crg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 9.00 | 58/14 | — | — | 4.37/6.34 | 描述性 |
| Fast-WAM/100/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/100/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/200/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/300/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/c/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/ch/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chr/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | full | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | sensitivity | long_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/chrg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/cr/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_10 | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | full | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | full | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | full | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | full | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | sensitivity | control | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | sensitivity | N500 | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | sensitivity | corrector_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
| Fast-WAM/400/crg/libero_spatial | sensitivity | gate_ablation | — | PENDING_PAIRS | 0 | 0 | — | — | — | — | — | PENDING |
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
| Fast-WAM/500/ch/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -16.00 | 18/98 | -24.37 | -21.80 | -21.21/-19.20 | 未证明 |
| Fast-WAM/500/ch/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | 6.60 | 71/38 | — | — | 0.98/3.40 | 描述性 |
| Fast-WAM/500/ch/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -15.87 | 18/94 | -24.48 | -21.61 | -21.23/-19.21 | 描述性 |
| Fast-WAM/500/ch/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | 6.47 | 67/36 | — | — | 0.75/3.13 | 描述性 |
| Fast-WAM/500/ch/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -1.60 | 17/25 | -7.58 | -5.40 | -5.30/-3.80 | 未证明 |
| Fast-WAM/500/ch/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 1.40 | 26/19 | — | — | -2.45/-0.80 | 描述性 |
| Fast-WAM/500/ch/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -2.04 | 15/25 | -8.00 | -5.93 | -5.73/-4.09 | 描述性 |
| Fast-WAM/500/ch/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 0.82 | 23/19 | — | — | -3.00/-1.43 | 描述性 |
| Fast-WAM/500/chg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -13.40 | 19/86 | -21.60 | -19.00 | -18.50/-16.60 | 未证明 |
| Fast-WAM/500/chg/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | -3.20 | 34/50 | — | — | -8.22/-6.20 | 描述性 |
| Fast-WAM/500/chg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 2.60 | 39/26 | — | — | -1.92/0.00 | 描述性 |
| Fast-WAM/500/chg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -13.36 | 19/83 | -21.81 | -19.00 | -18.62/-16.49 | 描述性 |
| Fast-WAM/500/chg/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | -3.13 | 33/48 | — | — | -8.28/-6.05 | 描述性 |
| Fast-WAM/500/chg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 2.51 | 37/25 | — | — | -2.11/0.00 | 描述性 |
| Fast-WAM/500/chg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -1.60 | 18/26 | -7.71 | -5.80 | -5.38/-3.80 | 未证明 |
| Fast-WAM/500/chg/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 0.00 | 19/19 | — | — | -3.57/-2.00 | 描述性 |
| Fast-WAM/500/chg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | 0.00 | 13/13 | — | — | -3.01/-1.60 | 描述性 |
| Fast-WAM/500/chg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -1.84 | 17/26 | -8.01 | -5.93 | -5.66/-4.09 | 描述性 |
| Fast-WAM/500/chg/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | -0.41 | 17/19 | — | — | -3.96/-2.25 | 描述性 |
| Fast-WAM/500/chg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | 0.20 | 13/12 | — | — | -2.82/-1.43 | 描述性 |
| Fast-WAM/500/chr/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -14.80 | 25/99 | -23.66 | -21.00 | -20.31/-18.20 | 未证明 |
| Fast-WAM/500/chr/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.20 | 35/29 | — | — | -3.30/-1.40 | 描述性 |
| Fast-WAM/500/chr/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | 6.40 | 76/44 | — | — | 0.53/3.20 | 描述性 |
| Fast-WAM/500/chr/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -14.82 | 25/96 | -23.96 | -20.88 | -20.51/-18.16 | 描述性 |
| Fast-WAM/500/chr/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 1.04 | 34/29 | — | — | -3.61/-1.67 | 描述性 |
| Fast-WAM/500/chr/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | 5.43 | 69/43 | — | — | -0.52/2.09 | 描述性 |
| Fast-WAM/500/chr/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -1.40 | 17/24 | -7.33 | -5.00 | -5.07/-3.40 | 未证明 |
| Fast-WAM/500/chr/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | 0.20 | 18/17 | — | — | -3.24/-1.80 | 描述性 |
| Fast-WAM/500/chr/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | 1.00 | 25/20 | — | — | -2.85/-1.20 | 描述性 |
| Fast-WAM/500/chr/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -1.43 | 16/23 | -7.35 | -5.11 | -5.09/-3.48 | 描述性 |
| Fast-WAM/500/chr/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | 0.61 | 18/15 | — | — | -2.81/-1.23 | 描述性 |
| Fast-WAM/500/chr/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | 1.02 | 24/19 | — | — | -2.83/-1.23 | 描述性 |
| Fast-WAM/500/chrg/libero_10 | full | control | F500 | PAIRED | 500 | 500 | -11.60 | 24/82 | -20.05 | -17.60 | -16.84/-14.80 | 未证明 |
| Fast-WAM/500/chrg/libero_10 | full | corrector_ablation | — | PAIRED | 500 | 500 | 1.80 | 38/29 | — | — | -2.79/-0.80 | 描述性 |
| Fast-WAM/500/chrg/libero_10 | full | long_ablation | — | PAIRED | 500 | 500 | -3.40 | 35/52 | — | — | -8.49/-6.40 | 描述性 |
| Fast-WAM/500/chrg/libero_10 | full | gate_ablation | — | PAIRED | 500 | 500 | 3.20 | 42/26 | — | — | -1.41/0.40 | 描述性 |
| Fast-WAM/500/chrg/libero_10 | sensitivity | control | F500 | PAIRED | 479 | 479 | -11.69 | 23/79 | -20.33 | -17.64 | -17.06/-15.03 | 描述性 |
| Fast-WAM/500/chrg/libero_10 | sensitivity | corrector_ablation | — | PAIRED | 479 | 479 | 1.67 | 35/27 | — | — | -2.95/-1.04 | 描述性 |
| Fast-WAM/500/chrg/libero_10 | sensitivity | long_ablation | — | PAIRED | 479 | 479 | -3.55 | 32/49 | — | — | -8.68/-6.68 | 描述性 |
| Fast-WAM/500/chrg/libero_10 | sensitivity | gate_ablation | — | PAIRED | 479 | 479 | 3.13 | 39/24 | — | — | -1.51/0.42 | 描述性 |
| Fast-WAM/500/chrg/libero_spatial | full | control | F500 | PAIRED | 500 | 500 | -2.60 | 16/29 | -8.71 | -6.40 | -6.38/-4.80 | 未证明 |
| Fast-WAM/500/chrg/libero_spatial | full | corrector_ablation | — | PAIRED | 500 | 500 | -1.00 | 14/19 | — | — | -4.33/-2.80 | 描述性 |
| Fast-WAM/500/chrg/libero_spatial | full | long_ablation | — | PAIRED | 500 | 500 | -2.00 | 15/25 | — | — | -5.60/-4.00 | 描述性 |
| Fast-WAM/500/chrg/libero_spatial | full | gate_ablation | — | PAIRED | 500 | 500 | -1.20 | 5/11 | — | — | -3.58/-2.60 | 描述性 |
| Fast-WAM/500/chrg/libero_spatial | sensitivity | control | F500 | PAIRED | 489 | 489 | -2.66 | 15/28 | -8.77 | -6.54 | -6.44/-4.91 | 描述性 |
| Fast-WAM/500/chrg/libero_spatial | sensitivity | corrector_ablation | — | PAIRED | 489 | 489 | -0.82 | 14/18 | — | — | -4.18/-2.66 | 描述性 |
| Fast-WAM/500/chrg/libero_spatial | sensitivity | long_ablation | — | PAIRED | 489 | 489 | -2.04 | 14/24 | — | — | -5.64/-4.09 | 描述性 |
| Fast-WAM/500/chrg/libero_spatial | sensitivity | gate_ablation | — | PAIRED | 489 | 489 | -1.23 | 5/11 | — | — | -3.66/-2.66 | 描述性 |
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
| F500 | DP | ch | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | chr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | chg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | DP | chrg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | ch | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | chr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | cg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | crg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | chg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Fast-WAM | chrg | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| F500 | Cosmos | c | PENDING | PENDING |
| F500 | Cosmos | cr | PENDING | PENDING |
| F500 | Cosmos | cg | PENDING | PENDING |
| F500 | Cosmos | crg | PENDING | PENDING |
| Fsize | DP | c | INTERIM | INTERIM |
| Fsize | DP | cr | INTERIM | INTERIM |
| Fsize | DP | ch | INTERIM | INTERIM |
| Fsize | DP | chr | INTERIM | INTERIM |
| Fsize | DP | cg | INTERIM | INTERIM |
| Fsize | DP | crg | INTERIM | INTERIM |
| Fsize | DP | chg | PENDING | PENDING |
| Fsize | DP | chrg | PENDING | PENDING |
| Fsize | Fast-WAM | c | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | cr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | ch | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | chr | NI_NOT_ESTABLISHED | NOT_ESTABLISHED |
| Fsize | Fast-WAM | cg | INTERIM | INTERIM |
| Fsize | Fast-WAM | crg | INTERIM | INTERIM |
| Fsize | Fast-WAM | chg | INTERIM | INTERIM |
| Fsize | Fast-WAM | chrg | INTERIM | INTERIM |
| Fsize | Cosmos | c | PENDING | PENDING |
| Fsize | Cosmos | cr | PENDING | PENDING |
| Fsize | Cosmos | cg | PENDING | PENDING |
| Fsize | Cosmos | crg | PENDING | PENDING |

## 限制与待到达

NFE是动作头计算量，不含视觉/检索/修正器；零NFE不等于零耗时。service无完整可比证据时不宣称墙钟加速。
Cosmos旧控制含timan108与timan107 renderer；JSON按旧控制节点分层，不能把跨节点差异解释为纯缓存因果效应。
Q3实际校准来自各尺寸B成功episode LOEO，不是独立B验证录制；门不可用任务在JSON逐组列出。

Missing cells, resume segments, duplicate counts and missing evidence are listed in CACHE_RESULTS.json; CSV includes paired counts and raw cost totals. Curves leave missing outcomes blank.
This CPU analysis performs no GPU/model/campaign operations. Input files are read without hashes or seals.
