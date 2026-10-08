## Library size 100

# Frozen test-A measurement

Partial measurement: this library-size group uses the two suite cells of one complete model. Pooled rows include only that model; combine both stages for the frozen all-cell summary. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 947/1000 | 94.70 | 38395 | 19431 | 19431 | 0.506082 | 19.431000 |
| pooled | uniform | 896/1000 | 89.60 | 40516 | 20496 | 3324 | 0.146464 | 5.934144 |
| pooled | task_descent | 943/1000 | 94.30 | 38376 | 19444 | 9630 | 0.289809 | 11.121728 |
| pi05_l10_100 | pure_c10 | 454/500 | 90.80 | 27796 | 14009 | 14009 | 0.503993 | 28.018000 |
| pi05_l10_100 | uniform | 423/500 | 84.60 | 29507 | 14856 | 2878 | 0.159239 | 9.397312 |
| pi05_l10_100 | task_descent | 453/500 | 90.60 | 27716 | 13968 | 7292 | 0.299710 | 16.613504 |
| pi05_spatial_100 | pure_c10 | 493/500 | 98.60 | 10599 | 5422 | 5422 | 0.511558 | 10.844000 |
| pi05_spatial_100 | uniform | 473/500 | 94.60 | 11009 | 5640 | 446 | 0.112225 | 2.470976 |
| pi05_spatial_100 | task_descent | 490/500 | 98.00 | 10660 | 5476 | 2338 | 0.264069 | 5.629952 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 43/47 | -0.40 | [-2.20, 1.40] |
| pooled | uniform_vs_pure_c10 | 40/91 | -5.10 | [-7.20, -3.00] |
| pooled | task_descent_vs_uniform | 80/33 | 4.70 | [2.70, 6.60] |
| pi05_l10_100 | task_descent_vs_pure_c10 | 37/38 | -0.20 | [-3.60, 3.20] |
| pi05_l10_100 | uniform_vs_pure_c10 | 34/65 | -6.20 | [-9.80, -2.60] |
| pi05_l10_100 | task_descent_vs_uniform | 58/28 | 6.00 | [2.60, 9.60] |
| pi05_spatial_100 | task_descent_vs_pure_c10 | 6/9 | -0.60 | [-2.00, 0.80] |
| pi05_spatial_100 | uniform_vs_pure_c10 | 6/26 | -4.00 | [-6.00, -2.00] |
| pi05_spatial_100 | task_descent_vs_uniform | 22/5 | 3.40 | [1.60, 5.40] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 21.63 | 42.73 |
| pooled | vs_uniform | -14.33 | -97.87 |
| pi05_l10_100 | vs_pure_c10 | 20.43 | 40.53 |
| pi05_l10_100 | vs_uniform | -14.05 | -88.21 |
| pi05_spatial_100 | vs_pure_c10 | 24.75 | 48.38 |
| pi05_spatial_100 | vs_uniform | -15.18 | -135.30 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.


## Library size 200

# Frozen test-A measurement

Partial measurement: this library-size group uses the two suite cells of one complete model. Pooled rows include only that model; combine both stages for the frozen all-cell summary. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 947/1000 | 94.70 | 38395 | 19431 | 19431 | 0.506082 | 19.431000 |
| pooled | uniform | 941/1000 | 94.10 | 38809 | 19643 | 2609 | 0.133942 | 5.198168 |
| pooled | task_descent | 940/1000 | 94.00 | 38499 | 19488 | 6026 | 0.209674 | 8.072224 |
| pi05_l10_200 | pure_c10 | 454/500 | 90.80 | 27796 | 14009 | 14009 | 0.503993 | 28.018000 |
| pi05_l10_200 | uniform | 451/500 | 90.20 | 28239 | 14232 | 2356 | 0.147355 | 8.322304 |
| pi05_l10_200 | task_descent | 451/500 | 90.20 | 27791 | 14008 | 4034 | 0.199707 | 11.100096 |
| pi05_spatial_200 | pure_c10 | 493/500 | 98.60 | 10599 | 5422 | 5422 | 0.511558 | 10.844000 |
| pi05_spatial_200 | uniform | 490/500 | 98.00 | 10570 | 5411 | 253 | 0.098109 | 2.074032 |
| pi05_spatial_200 | task_descent | 489/500 | 97.80 | 10708 | 5480 | 1992 | 0.235541 | 5.044352 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 41/48 | -0.70 | [-2.50, 1.10] |
| pooled | uniform_vs_pure_c10 | 41/47 | -0.60 | [-2.40, 1.20] |
| pooled | task_descent_vs_uniform | 30/31 | -0.10 | [-1.60, 1.40] |
| pi05_l10_200 | task_descent_vs_pure_c10 | 36/39 | -0.60 | [-3.80, 2.60] |
| pi05_l10_200 | uniform_vs_pure_c10 | 35/38 | -0.60 | [-3.81, 2.60] |
| pi05_l10_200 | task_descent_vs_uniform | 27/27 | 0.00 | [-2.80, 2.80] |
| pi05_spatial_200 | task_descent_vs_pure_c10 | 5/9 | -0.80 | [-2.20, 0.60] |
| pi05_spatial_200 | uniform_vs_pure_c10 | 6/9 | -0.60 | [-2.00, 0.80] |
| pi05_spatial_200 | task_descent_vs_uniform | 3/4 | -0.20 | [-1.20, 0.80] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 29.64 | 58.57 |
| pooled | vs_uniform | -7.57 | -56.54 |
| pi05_l10_200 | vs_pure_c10 | 30.43 | 60.38 |
| pi05_l10_200 | vs_uniform | -5.24 | -35.53 |
| pi05_spatial_200 | vs_pure_c10 | 27.60 | 53.96 |
| pi05_spatial_200 | vs_uniform | -13.74 | -140.08 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.


## Library size 500

# Frozen test-A measurement

Partial measurement: this library-size group uses the two suite cells of one complete model. Pooled rows include only that model; combine both stages for the frozen all-cell summary. Two new 500-episode arms per cell are measured against the existing R14 pure C10 reference. No A selection or refitting. GR00T intended host is local; pi05 intended host is h100. Startup hosts are retained in JSON.

| Cell | Arm | Success / episodes | SR % | N | V | M | Owner IR | Cost / episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pure_c10 | 947/1000 | 94.70 | 38395 | 19431 | 19431 | 0.506082 | 19.431000 |
| pooled | uniform | 952/1000 | 95.20 | 37573 | 19042 | 2320 | 0.129395 | 4.861744 |
| pooled | task_descent | 954/1000 | 95.40 | 38032 | 19273 | 7301 | 0.239818 | 9.120744 |
| pi05_l10_500 | pure_c10 | 454/500 | 90.80 | 27796 | 14009 | 14009 | 0.503993 | 28.018000 |
| pi05_l10_500 | uniform | 464/500 | 92.80 | 26909 | 13574 | 2080 | 0.142223 | 7.654176 |
| pi05_l10_500 | task_descent | 462/500 | 92.40 | 27459 | 13851 | 6167 | 0.267124 | 14.669936 |
| pi05_spatial_500 | pure_c10 | 493/500 | 98.60 | 10599 | 5422 | 5422 | 0.511558 | 10.844000 |
| pi05_spatial_500 | uniform | 488/500 | 97.60 | 10664 | 5468 | 240 | 0.097023 | 2.069312 |
| pi05_spatial_500 | task_descent | 492/500 | 98.40 | 10573 | 5422 | 1134 | 0.168900 | 3.571552 |


| Cell | Contrast | Gains / losses | SR delta pp | Paired bootstrap 95% CI pp |
| --- | --- | --- | --- | --- |
| pooled | task_descent_vs_pure_c10 | 42/35 | 0.70 | [-1.00, 2.40] |
| pooled | uniform_vs_pure_c10 | 45/40 | 0.50 | [-1.30, 2.20] |
| pooled | task_descent_vs_uniform | 31/29 | 0.20 | [-1.30, 1.70] |
| pi05_l10_500 | task_descent_vs_pure_c10 | 36/28 | 1.60 | [-1.40, 4.60] |
| pi05_l10_500 | uniform_vs_pure_c10 | 39/29 | 2.00 | [-1.20, 5.20] |
| pi05_l10_500 | task_descent_vs_uniform | 24/26 | -0.40 | [-3.00, 2.20] |
| pi05_spatial_500 | task_descent_vs_pure_c10 | 6/7 | -0.20 | [-1.60, 1.20] |
| pi05_spatial_500 | uniform_vs_pure_c10 | 6/11 | -1.00 | [-2.60, 0.60] |
| pi05_spatial_500 | task_descent_vs_uniform | 7/3 | 0.80 | [-0.40, 2.00] |


| Cell | Reference | IR saved pp | Relative IR saved % |
| --- | --- | --- | --- |
| pooled | vs_pure_c10 | 26.63 | 52.61 |
| pooled | vs_uniform | -11.04 | -85.34 |
| pi05_l10_500 | vs_pure_c10 | 23.69 | 47.00 |
| pi05_l10_500 | vs_uniform | -12.49 | -87.82 |
| pi05_spatial_500 | vs_pure_c10 | 34.27 | 66.98 |
| pi05_spatial_500 | vs_uniform | -7.19 | -74.08 |


Frozen R14 paired CP intervals and 5,000-draw paired within-task bootstrap, seed 20261005, are unchanged. Forty task comparisons form each library-size contrast family. JSON retains all task bounds, audit hashes, N/V/M, retries, hosts and B/A discrepancies. Shared C10 data are never pooled across the three sizes. Signed losses and negative savings remain signed. B estimates were selected and stitched; A measures the frozen actual mixed controller. The B size-500 estimate excludes own serving episodes while retaining them in fits; official A identities do not belong to B.
