## Library size 100

# Frozen test-A measurement

Partial measurement: this library-size group uses the two suite cells of one complete model. Pooled rows include only that model; combine both stages for the frozen all-cell summary. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 917/1000 | 91.70 | 39677 | 20071 | 20071 | 0.505860 | 20.071000 |
| pooled | uniform | 886/1000 | 88.60 | 41157 | 20780 | 6774 | 0.214955 | 8.846888 |
| pooled | task_descent | 899/1000 | 89.90 | 40760 | 20613 | 12556 | 0.337302 | 13.748436 |
| groot_l10_100 | pure_c10 | 440/500 | 88.00 | 28473 | 14339 | 14339 | 0.503600 | 28.678000 |
| groot_l10_100 | uniform | 411/500 | 82.20 | 29971 | 15075 | 6260 | 0.252398 | 15.129240 |
| groot_l10_100 | task_descent | 418/500 | 83.60 | 29672 | 14939 | 9277 | 0.340893 | 20.229952 |
| groot_spatial_100 | pure_c10 | 477/500 | 95.40 | 11204 | 5732 | 5732 | 0.511603 | 11.464000 |
| groot_spatial_100 | uniform | 475/500 | 95.00 | 11186 | 5705 | 514 | 0.114632 | 2.564536 |
| groot_spatial_100 | task_descent | 481/500 | 96.20 | 11088 | 5674 | 3279 | 0.327693 | 7.266920 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 55/73 | -1.80 | [-4.00, 0.40] |
| pooled | uniform_vs_pure_c10 | 60/91 | -3.10 | [-5.50, -0.70] |
| pooled | task_descent_vs_uniform | 77/64 | 1.30 | [-1.00, 3.50] |
| groot_l10_100 | task_descent_vs_pure_c10 | 35/57 | -4.40 | [-8.20, -0.60] |
| groot_l10_100 | uniform_vs_pure_c10 | 38/67 | -5.80 | [-9.80, -1.80] |
| groot_l10_100 | task_descent_vs_uniform | 55/48 | 1.40 | [-2.60, 5.40] |
| groot_spatial_100 | task_descent_vs_pure_c10 | 20/16 | 0.80 | [-1.60, 3.20] |
| groot_spatial_100 | uniform_vs_pure_c10 | 22/24 | -0.40 | [-3.00, 2.20] |
| groot_spatial_100 | task_descent_vs_uniform | 22/16 | 1.20 | [-1.20, 3.60] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 16.86 | 33.32 |
| pooled | vs_uniform | -12.23 | -56.92 |
| groot_l10_100 | vs_pure_c10 | 16.27 | 32.31 |
| groot_l10_100 | vs_uniform | -8.85 | -35.06 |
| groot_spatial_100 | vs_pure_c10 | 18.39 | 35.95 |
| groot_spatial_100 | vs_uniform | -21.31 | -185.87 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.


## Library size 200

# Frozen test-A measurement

Partial measurement: this library-size group uses the two suite cells of one complete model. Pooled rows include only that model; combine both stages for the frozen all-cell summary. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 917/1000 | 91.70 | 39677 | 20071 | 20071 | 0.505860 | 20.071000 |
| pooled | uniform | 885/1000 | 88.50 | 40953 | 20703 | 4620 | 0.170935 | 7.000284 |
| pooled | task_descent | 915/1000 | 91.50 | 39623 | 20038 | 11442 | 0.320879 | 12.714208 |
| groot_l10_200 | pure_c10 | 440/500 | 88.00 | 28473 | 14339 | 14339 | 0.503600 | 28.678000 |
| groot_l10_200 | uniform | 405/500 | 81.00 | 29971 | 15086 | 4138 | 0.192129 | 11.516608 |
| groot_l10_200 | task_descent | 441/500 | 88.20 | 28403 | 14304 | 8113 | 0.317898 | 18.058536 |
| groot_spatial_200 | pure_c10 | 477/500 | 95.40 | 11204 | 5732 | 5732 | 0.511603 | 11.464000 |
| groot_spatial_200 | uniform | 480/500 | 96.00 | 10982 | 5617 | 482 | 0.113092 | 2.483960 |
| groot_spatial_200 | task_descent | 474/500 | 94.80 | 11220 | 5734 | 3329 | 0.328426 | 7.369880 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 50/52 | -0.20 | [-2.20, 1.70] |
| pooled | uniform_vs_pure_c10 | 55/87 | -3.20 | [-5.50, -0.90] |
| pooled | task_descent_vs_uniform | 76/46 | 3.00 | [0.80, 5.10] |
| groot_l10_200 | task_descent_vs_pure_c10 | 32/31 | 0.20 | [-2.80, 3.20] |
| groot_l10_200 | uniform_vs_pure_c10 | 32/67 | -7.00 | [-10.80, -3.00] |
| groot_l10_200 | task_descent_vs_uniform | 60/24 | 7.20 | [3.60, 10.60] |
| groot_spatial_200 | task_descent_vs_pure_c10 | 18/21 | -0.60 | [-3.20, 1.80] |
| groot_spatial_200 | uniform_vs_pure_c10 | 23/20 | 0.60 | [-2.00, 3.20] |
| groot_spatial_200 | task_descent_vs_uniform | 16/22 | -1.20 | [-3.60, 1.20] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 18.50 | 36.57 |
| pooled | vs_uniform | -14.99 | -87.72 |
| groot_l10_200 | vs_pure_c10 | 18.57 | 36.87 |
| groot_l10_200 | vs_uniform | -12.58 | -65.46 |
| groot_spatial_200 | vs_pure_c10 | 18.32 | 35.80 |
| groot_spatial_200 | vs_uniform | -21.53 | -190.41 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.


## Library size 500

# Frozen test-A measurement

Partial measurement: this library-size group uses the two suite cells of one complete model. Pooled rows include only that model; combine both stages for the frozen all-cell summary. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 917/1000 | 91.70 | 39677 | 20071 | 20071 | 0.505860 | 20.071000 |
| pooled | uniform | 924/1000 | 92.40 | 38996 | 19718 | 3867 | 0.159323 | 6.212948 |
| pooled | task_descent | 921/1000 | 92.10 | 39557 | 20009 | 10107 | 0.292552 | 11.572496 |
| groot_l10_500 | pure_c10 | 440/500 | 88.00 | 28473 | 14339 | 14339 | 0.503600 | 28.678000 |
| groot_l10_500 | uniform | 447/500 | 89.40 | 27991 | 14105 | 3433 | 0.179074 | 10.024912 |
| groot_l10_500 | task_descent | 448/500 | 89.60 | 28386 | 14297 | 7335 | 0.294701 | 16.730752 |
| groot_spatial_500 | pure_c10 | 477/500 | 95.40 | 11204 | 5732 | 5732 | 0.511603 | 11.464000 |
| groot_spatial_500 | uniform | 477/500 | 95.40 | 11005 | 5613 | 434 | 0.109086 | 2.400984 |
| groot_spatial_500 | task_descent | 473/500 | 94.60 | 11171 | 5712 | 2772 | 0.287093 | 6.414240 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 60/56 | 0.40 | [-1.60, 2.50] |
| pooled | uniform_vs_pure_c10 | 64/57 | 0.70 | [-1.50, 2.80] |
| pooled | task_descent_vs_uniform | 50/53 | -0.30 | [-2.20, 1.70] |
| groot_l10_500 | task_descent_vs_pure_c10 | 39/31 | 1.60 | [-1.60, 4.80] |
| groot_l10_500 | uniform_vs_pure_c10 | 42/35 | 1.40 | [-2.00, 4.80] |
| groot_l10_500 | task_descent_vs_uniform | 37/36 | 0.20 | [-3.20, 3.60] |
| groot_spatial_500 | task_descent_vs_pure_c10 | 21/25 | -0.80 | [-3.40, 1.80] |
| groot_spatial_500 | uniform_vs_pure_c10 | 22/22 | 0.00 | [-2.60, 2.60] |
| groot_spatial_500 | task_descent_vs_uniform | 13/17 | -0.80 | [-2.80, 1.40] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 21.33 | 42.17 |
| pooled | vs_uniform | -13.32 | -83.62 |
| groot_l10_500 | vs_pure_c10 | 20.89 | 41.48 |
| groot_l10_500 | vs_uniform | -11.56 | -64.57 |
| groot_spatial_500 | vs_pure_c10 | 22.45 | 43.88 |
| groot_spatial_500 | vs_uniform | -17.80 | -163.18 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.
