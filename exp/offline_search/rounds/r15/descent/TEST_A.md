## Library size 100

# Frozen test-A measurement

Each library-size group uses four model/suite cells. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 1864/2000 | 93.20 | 78072 | 39502 | 39502 | 0.505969 | 19.751000 |
| pooled | uniform | 1782/2000 | 89.10 | 81673 | 41276 | 10098 | 0.180978 | 7.390516 |
| pooled | task_descent | 1842/2000 | 92.10 | 79136 | 40057 | 22186 | 0.314271 | 12.435082 |
| pi05_l10_100 | pure_c10 | 454/500 | 90.80 | 27796 | 14009 | 14009 | 0.503993 | 28.018000 |
| pi05_l10_100 | uniform | 423/500 | 84.60 | 29507 | 14856 | 2878 | 0.159239 | 9.397312 |
| pi05_l10_100 | task_descent | 453/500 | 90.60 | 27716 | 13968 | 7292 | 0.299710 | 16.613504 |
| pi05_spatial_100 | pure_c10 | 493/500 | 98.60 | 10599 | 5422 | 5422 | 0.511558 | 10.844000 |
| pi05_spatial_100 | uniform | 473/500 | 94.60 | 11009 | 5640 | 446 | 0.112225 | 2.470976 |
| pi05_spatial_100 | task_descent | 490/500 | 98.00 | 10660 | 5476 | 2338 | 0.264069 | 5.629952 |
| groot_l10_100 | pure_c10 | 440/500 | 88.00 | 28473 | 14339 | 14339 | 0.503600 | 28.678000 |
| groot_l10_100 | uniform | 411/500 | 82.20 | 29971 | 15075 | 6260 | 0.252398 | 15.129240 |
| groot_l10_100 | task_descent | 418/500 | 83.60 | 29672 | 14939 | 9277 | 0.340893 | 20.229952 |
| groot_spatial_100 | pure_c10 | 477/500 | 95.40 | 11204 | 5732 | 5732 | 0.511603 | 11.464000 |
| groot_spatial_100 | uniform | 475/500 | 95.00 | 11186 | 5705 | 514 | 0.114632 | 2.564536 |
| groot_spatial_100 | task_descent | 481/500 | 96.20 | 11088 | 5674 | 3279 | 0.327693 | 7.266920 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 98/120 | -1.10 | [-2.50, 0.30] |
| pooled | uniform_vs_pure_c10 | 100/182 | -4.10 | [-5.70, -2.50] |
| pooled | task_descent_vs_uniform | 157/97 | 3.00 | [1.40, 4.55] |
| pi05_l10_100 | task_descent_vs_pure_c10 | 37/38 | -0.20 | [-3.60, 3.20] |
| pi05_l10_100 | uniform_vs_pure_c10 | 34/65 | -6.20 | [-9.80, -2.60] |
| pi05_l10_100 | task_descent_vs_uniform | 58/28 | 6.00 | [2.60, 9.60] |
| pi05_spatial_100 | task_descent_vs_pure_c10 | 6/9 | -0.60 | [-2.00, 0.80] |
| pi05_spatial_100 | uniform_vs_pure_c10 | 6/26 | -4.00 | [-6.00, -2.00] |
| pi05_spatial_100 | task_descent_vs_uniform | 22/5 | 3.40 | [1.60, 5.40] |
| groot_l10_100 | task_descent_vs_pure_c10 | 35/57 | -4.40 | [-8.20, -0.60] |
| groot_l10_100 | uniform_vs_pure_c10 | 38/67 | -5.80 | [-9.80, -1.80] |
| groot_l10_100 | task_descent_vs_uniform | 55/48 | 1.40 | [-2.60, 5.40] |
| groot_spatial_100 | task_descent_vs_pure_c10 | 20/16 | 0.80 | [-1.60, 3.20] |
| groot_spatial_100 | uniform_vs_pure_c10 | 22/24 | -0.40 | [-3.00, 2.20] |
| groot_spatial_100 | task_descent_vs_uniform | 22/16 | 1.20 | [-1.20, 3.60] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 19.17 | 37.89 |
| pooled | vs_uniform | -13.33 | -73.65 |
| pi05_l10_100 | vs_pure_c10 | 20.43 | 40.53 |
| pi05_l10_100 | vs_uniform | -14.05 | -88.21 |
| pi05_spatial_100 | vs_pure_c10 | 24.75 | 48.38 |
| pi05_spatial_100 | vs_uniform | -15.18 | -135.30 |
| groot_l10_100 | vs_pure_c10 | 16.27 | 32.31 |
| groot_l10_100 | vs_uniform | -8.85 | -35.06 |
| groot_spatial_100 | vs_pure_c10 | 18.39 | 35.95 |
| groot_spatial_100 | vs_uniform | -21.31 | -185.87 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.


## Library size 200

# Frozen test-A measurement

Each library-size group uses four model/suite cells. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 1864/2000 | 93.20 | 78072 | 39502 | 39502 | 0.505969 | 19.751000 |
| pooled | uniform | 1826/2000 | 91.30 | 79762 | 40346 | 7229 | 0.152936 | 6.099226 |
| pooled | task_descent | 1855/2000 | 92.75 | 78122 | 39526 | 17468 | 0.266077 | 10.393216 |
| pi05_l10_200 | pure_c10 | 454/500 | 90.80 | 27796 | 14009 | 14009 | 0.503993 | 28.018000 |
| pi05_l10_200 | uniform | 451/500 | 90.20 | 28239 | 14232 | 2356 | 0.147355 | 8.322304 |
| pi05_l10_200 | task_descent | 451/500 | 90.20 | 27791 | 14008 | 4034 | 0.199707 | 11.100096 |
| pi05_spatial_200 | pure_c10 | 493/500 | 98.60 | 10599 | 5422 | 5422 | 0.511558 | 10.844000 |
| pi05_spatial_200 | uniform | 490/500 | 98.00 | 10570 | 5411 | 253 | 0.098109 | 2.074032 |
| pi05_spatial_200 | task_descent | 489/500 | 97.80 | 10708 | 5480 | 1992 | 0.235541 | 5.044352 |
| groot_l10_200 | pure_c10 | 440/500 | 88.00 | 28473 | 14339 | 14339 | 0.503600 | 28.678000 |
| groot_l10_200 | uniform | 405/500 | 81.00 | 29971 | 15086 | 4138 | 0.192129 | 11.516608 |
| groot_l10_200 | task_descent | 441/500 | 88.20 | 28403 | 14304 | 8113 | 0.317898 | 18.058536 |
| groot_spatial_200 | pure_c10 | 477/500 | 95.40 | 11204 | 5732 | 5732 | 0.511603 | 11.464000 |
| groot_spatial_200 | uniform | 480/500 | 96.00 | 10982 | 5617 | 482 | 0.113092 | 2.483960 |
| groot_spatial_200 | task_descent | 474/500 | 94.80 | 11220 | 5734 | 3329 | 0.328426 | 7.369880 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 91/100 | -0.45 | [-1.80, 0.90] |
| pooled | uniform_vs_pure_c10 | 96/134 | -1.90 | [-3.40, -0.40] |
| pooled | task_descent_vs_uniform | 106/77 | 1.45 | [0.05, 2.80] |
| pi05_l10_200 | task_descent_vs_pure_c10 | 36/39 | -0.60 | [-3.80, 2.60] |
| pi05_l10_200 | uniform_vs_pure_c10 | 35/38 | -0.60 | [-3.81, 2.60] |
| pi05_l10_200 | task_descent_vs_uniform | 27/27 | 0.00 | [-2.80, 2.80] |
| pi05_spatial_200 | task_descent_vs_pure_c10 | 5/9 | -0.80 | [-2.20, 0.60] |
| pi05_spatial_200 | uniform_vs_pure_c10 | 6/9 | -0.60 | [-2.00, 0.80] |
| pi05_spatial_200 | task_descent_vs_uniform | 3/4 | -0.20 | [-1.20, 0.80] |
| groot_l10_200 | task_descent_vs_pure_c10 | 32/31 | 0.20 | [-2.80, 3.20] |
| groot_l10_200 | uniform_vs_pure_c10 | 32/67 | -7.00 | [-10.80, -3.00] |
| groot_l10_200 | task_descent_vs_uniform | 60/24 | 7.20 | [3.60, 10.60] |
| groot_spatial_200 | task_descent_vs_pure_c10 | 18/21 | -0.60 | [-3.20, 1.80] |
| groot_spatial_200 | uniform_vs_pure_c10 | 23/20 | 0.60 | [-2.00, 3.20] |
| groot_spatial_200 | task_descent_vs_uniform | 16/22 | -1.20 | [-3.60, 1.20] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 23.99 | 47.41 |
| pooled | vs_uniform | -11.31 | -73.98 |
| pi05_l10_200 | vs_pure_c10 | 30.43 | 60.38 |
| pi05_l10_200 | vs_uniform | -5.24 | -35.53 |
| pi05_spatial_200 | vs_pure_c10 | 27.60 | 53.96 |
| pi05_spatial_200 | vs_uniform | -13.74 | -140.08 |
| groot_l10_200 | vs_pure_c10 | 18.57 | 36.87 |
| groot_l10_200 | vs_uniform | -12.58 | -65.46 |
| groot_spatial_200 | vs_pure_c10 | 18.32 | 35.80 |
| groot_spatial_200 | vs_uniform | -21.53 | -190.41 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.


## Library size 500

# Frozen test-A measurement

Each library-size group uses four model/suite cells. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 1864/2000 | 93.20 | 78072 | 39502 | 39502 | 0.505969 | 19.751000 |
| pooled | uniform | 1876/2000 | 93.80 | 76569 | 38760 | 6187 | 0.144637 | 5.537346 |
| pooled | task_descent | 1875/2000 | 93.75 | 77589 | 39282 | 17408 | 0.266703 | 10.346620 |
| pi05_l10_500 | pure_c10 | 454/500 | 90.80 | 27796 | 14009 | 14009 | 0.503993 | 28.018000 |
| pi05_l10_500 | uniform | 464/500 | 92.80 | 26909 | 13574 | 2080 | 0.142223 | 7.654176 |
| pi05_l10_500 | task_descent | 462/500 | 92.40 | 27459 | 13851 | 6167 | 0.267124 | 14.669936 |
| pi05_spatial_500 | pure_c10 | 493/500 | 98.60 | 10599 | 5422 | 5422 | 0.511558 | 10.844000 |
| pi05_spatial_500 | uniform | 488/500 | 97.60 | 10664 | 5468 | 240 | 0.097023 | 2.069312 |
| pi05_spatial_500 | task_descent | 492/500 | 98.40 | 10573 | 5422 | 1134 | 0.168900 | 3.571552 |
| groot_l10_500 | pure_c10 | 440/500 | 88.00 | 28473 | 14339 | 14339 | 0.503600 | 28.678000 |
| groot_l10_500 | uniform | 447/500 | 89.40 | 27991 | 14105 | 3433 | 0.179074 | 10.024912 |
| groot_l10_500 | task_descent | 448/500 | 89.60 | 28386 | 14297 | 7335 | 0.294701 | 16.730752 |
| groot_spatial_500 | pure_c10 | 477/500 | 95.40 | 11204 | 5732 | 5732 | 0.511603 | 11.464000 |
| groot_spatial_500 | uniform | 477/500 | 95.40 | 11005 | 5613 | 434 | 0.109086 | 2.400984 |
| groot_spatial_500 | task_descent | 473/500 | 94.60 | 11171 | 5712 | 2772 | 0.287093 | 6.414240 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 102/91 | 0.55 | [-0.80, 1.85] |
| pooled | uniform_vs_pure_c10 | 109/97 | 0.60 | [-0.80, 1.95] |
| pooled | task_descent_vs_uniform | 81/82 | -0.05 | [-1.30, 1.20] |
| pi05_l10_500 | task_descent_vs_pure_c10 | 36/28 | 1.60 | [-1.40, 4.60] |
| pi05_l10_500 | uniform_vs_pure_c10 | 39/29 | 2.00 | [-1.20, 5.20] |
| pi05_l10_500 | task_descent_vs_uniform | 24/26 | -0.40 | [-3.00, 2.20] |
| pi05_spatial_500 | task_descent_vs_pure_c10 | 6/7 | -0.20 | [-1.60, 1.20] |
| pi05_spatial_500 | uniform_vs_pure_c10 | 6/11 | -1.00 | [-2.60, 0.60] |
| pi05_spatial_500 | task_descent_vs_uniform | 7/3 | 0.80 | [-0.40, 2.00] |
| groot_l10_500 | task_descent_vs_pure_c10 | 39/31 | 1.60 | [-1.60, 4.80] |
| groot_l10_500 | uniform_vs_pure_c10 | 42/35 | 1.40 | [-2.00, 4.80] |
| groot_l10_500 | task_descent_vs_uniform | 37/36 | 0.20 | [-3.20, 3.60] |
| groot_spatial_500 | task_descent_vs_pure_c10 | 21/25 | -0.80 | [-3.40, 1.80] |
| groot_spatial_500 | uniform_vs_pure_c10 | 22/22 | 0.00 | [-2.60, 2.60] |
| groot_spatial_500 | task_descent_vs_uniform | 13/17 | -0.80 | [-2.80, 1.40] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 23.93 | 47.29 |
| pooled | vs_uniform | -12.21 | -84.40 |
| pi05_l10_500 | vs_pure_c10 | 23.69 | 47.00 |
| pi05_l10_500 | vs_uniform | -12.49 | -87.82 |
| pi05_spatial_500 | vs_pure_c10 | 34.27 | 66.98 |
| pi05_spatial_500 | vs_uniform | -7.19 | -74.08 |
| groot_l10_500 | vs_pure_c10 | 20.89 | 41.48 |
| groot_l10_500 | vs_uniform | -11.56 | -64.57 |
| groot_spatial_500 | vs_pure_c10 | 22.45 | 43.88 |
| groot_spatial_500 | vs_uniform | -17.80 | -163.18 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.
