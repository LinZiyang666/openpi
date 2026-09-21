## shadow diagnostics — pi05_rc

episodes seen 130, admitted 130; rejections: none
missing labels by outcome: failure: 64 eps (0 rejected), error rows 0/8910, missing decision rows 0, unknown decision counts 0 eps; success: 66 eps (0 rejected), error rows 0/5758, missing decision rows 0, unknown decision counts 0 eps

| task | eps | decisions | disp_K | d_1 | d_2 | d_3 | d_5 | r_1 | r_2 | r_3 | r_5 | ΔBIC>10 frac (n dense) | SR(shadow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CloseBlenderLid | 10 | 1757 | 1.365 | 4.762 | 2.219 | 1.522 | 0.932 | 3.324 | 1.520 | 1.069 | 0.661 | 0.317 (104) | 0.200 |
| CloseFridge | 10 | 1338 | 2.012 | 7.629 | 2.642 | 1.807 | 1.096 | 3.277 | 1.273 | 0.839 | 0.522 | 0.365 (85) | 0.600 |
| CoffeeSetupMug | 10 | 1063 | 1.351 | 5.679 | 2.135 | 1.533 | 0.905 | 4.165 | 1.596 | 1.139 | 0.666 | 0.486 (70) | 0.500 |
| OpenCabinet | 10 | 1373 | 2.107 | 6.289 | 2.940 | 1.980 | 1.120 | 2.977 | 1.322 | 0.919 | 0.537 | 0.258 (89) | 0.700 |
| OpenDrawer | 10 | 1153 | 1.609 | 6.239 | 2.307 | 1.548 | 0.974 | 3.673 | 1.421 | 0.980 | 0.582 | 0.311 (61) | 0.400 |
| OpenStandMixerHead | 10 | 815 | 1.533 | 6.082 | 2.397 | 1.731 | 0.951 | 3.874 | 1.620 | 1.111 | 0.634 | 0.396 (53) | 0.300 |
| PickPlaceCounterToCabinet | 10 | 1209 | 1.909 | 7.051 | 2.873 | 1.872 | 1.080 | 3.230 | 1.403 | 0.937 | 0.549 | 0.567 (67) | 0.400 |
| PickPlaceCounterToStove | 10 | 913 | 1.803 | 5.997 | 2.931 | 1.937 | 1.102 | 3.184 | 1.576 | 1.029 | 0.590 | 0.250 (44) | 0.600 |
| PickPlaceDrawerToCounter | 10 | 1478 | 2.151 | 7.521 | 2.783 | 1.828 | 1.095 | 2.861 | 1.241 | 0.798 | 0.493 | 0.311 (103) | 0.200 |
| PickPlaceSinkToCounter | 10 | 729 | 1.629 | 5.701 | 2.416 | 1.624 | 0.982 | 3.497 | 1.479 | 1.010 | 0.622 | 0.128 (47) | 1.000 |
| PickPlaceToasterToCounter | 10 | 1141 | 1.262 | 5.738 | 2.172 | 1.545 | 0.907 | 4.650 | 1.684 | 1.188 | 0.706 | 0.449 (78) | 0.500 |
| SlideDishwasherRack | 10 | 676 | 1.651 | 8.156 | 2.724 | 1.858 | 1.083 | 5.060 | 1.675 | 1.113 | 0.625 | 0.000 (37) | 0.500 |
| TurnOnSinkFaucet | 10 | 1023 | 1.048 | 4.830 | 2.146 | 1.433 | 0.858 | 4.430 | 2.047 | 1.317 | 0.801 | 0.143 (56) | 0.700 |

Spearman rho(d_1, g) = 0.08528206142151305 n=13 ci95=[-0.4206163425234888, 0.6306968956322994] gap_range=0.78 → **no_conclusion**

LOTO (descriptive): hits 3, missed cliffs 4, false cliffs 4, correct flat 2
## shadow diagnostics — groot_rc

episodes seen 130, admitted 130; rejections: not_accepted_terminal=4
missing labels by outcome: failure: 44 eps (0 rejected), error rows 0/5520, missing decision rows 0, unknown decision counts 0 eps; success: 86 eps (0 rejected), error rows 0/6427, missing decision rows 0, unknown decision counts 0 eps

| task | eps | decisions | disp_K | d_1 | d_2 | d_3 | r_1 | r_2 | r_3 | ΔBIC>10 frac (n dense) | SR(shadow) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CloseBlenderLid | 10 | 999 | 1.223 | 1.708 | 0.808 | 0.402 | 1.412 | 0.653 | 0.325 | 0.200 (65) | 0.800 |
| CloseFridge | 10 | 1216 | 1.591 | 2.085 | 1.005 | 0.461 | 1.173 | 0.590 | 0.267 | 0.290 (69) | 0.800 |
| CoffeeSetupMug | 10 | 872 | 1.145 | 1.726 | 0.797 | 0.400 | 1.455 | 0.688 | 0.333 | 0.180 (61) | 0.600 |
| OpenCabinet | 10 | 952 | 1.370 | 1.829 | 0.876 | 0.417 | 1.352 | 0.646 | 0.301 | 0.087 (69) | 1.000 |
| OpenDrawer | 10 | 917 | 1.213 | 1.759 | 0.818 | 0.397 | 1.415 | 0.676 | 0.323 | 0.227 (66) | 0.900 |
| OpenStandMixerHead | 10 | 593 | 1.497 | 1.784 | 0.874 | 0.423 | 1.205 | 0.607 | 0.280 | 0.194 (36) | 0.500 |
| PickPlaceCounterToCabinet | 10 | 1026 | 1.198 | 1.706 | 0.797 | 0.406 | 1.385 | 0.657 | 0.320 | 0.234 (64) | 0.500 |
| PickPlaceCounterToStove | 10 | 698 | 1.320 | 1.799 | 0.852 | 0.410 | 1.375 | 0.652 | 0.313 | 0.103 (29) | 0.700 |
| PickPlaceDrawerToCounter | 10 | 890 | 1.392 | 1.882 | 0.872 | 0.438 | 1.269 | 0.590 | 0.291 | 0.259 (54) | 0.900 |
| PickPlaceSinkToCounter | 10 | 989 | 1.201 | 1.767 | 0.826 | 0.403 | 1.421 | 0.674 | 0.331 | 0.154 (65) | 0.800 |
| PickPlaceToasterToCounter | 10 | 963 | 1.244 | 1.689 | 0.805 | 0.401 | 1.347 | 0.639 | 0.319 | 0.141 (64) | 0.600 |
| SlideDishwasherRack | 10 | 707 | 1.242 | 1.778 | 0.860 | 0.411 | 1.360 | 0.648 | 0.310 | 0.091 (44) | 0.400 |
| TurnOnSinkFaucet | 10 | 1125 | 1.222 | 1.769 | 0.825 | 0.402 | 1.434 | 0.661 | 0.319 | 0.156 (77) | 0.100 |

Spearman rho(d_1, g) = 0.37689169079829965 n=13 ci95=[-0.35227679734305806, 0.8991613923445548] gap_range=0.30000000000000004 → **no_conclusion**

LOTO (descriptive): hits 3, missed cliffs 1, false cliffs 4, correct flat 5
## shadow diagnostics — pi05_libero_spatial

episodes seen 100, admitted 100; rejections: none
missing labels by outcome: failure: 1 eps (0 rejected), error rows 0/44, missing decision rows 0, unknown decision counts 0 eps; success: 99 eps (0 rejected), error rows 0/2074, missing decision rows 0, unknown decision counts 0 eps

| task | eps | decisions | disp_K | d_1 | d_2 | d_3 | d_5 | r_1 | r_2 | r_3 | r_5 | ΔBIC>10 frac (n dense) | SR(shadow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pick up the black bowl between the plate and the ramekin and place it on the plate | 10 | 155 | 0.372 | 0.306 | 0.222 | 0.169 | 0.097 | 0.806 | 0.592 | 0.459 | 0.260 | 0.125 (8) | 1.000 |
| pick up the black bowl from table center and place it on the plate | 10 | 202 | 0.288 | 0.219 | 0.167 | 0.129 | 0.075 | 0.800 | 0.604 | 0.461 | 0.262 | 0.000 (8) | 1.000 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 10 | 256 | 0.334 | 0.261 | 0.189 | 0.144 | 0.081 | 0.786 | 0.567 | 0.452 | 0.259 | 0.154 (13) | 1.000 |
| pick up the black bowl next to the cookie box and place it on the plate | 10 | 209 | 0.278 | 0.229 | 0.175 | 0.134 | 0.077 | 0.848 | 0.621 | 0.489 | 0.274 | 0.133 (15) | 1.000 |
| pick up the black bowl next to the plate and place it on the plate | 10 | 225 | 0.363 | 0.285 | 0.211 | 0.158 | 0.088 | 0.768 | 0.569 | 0.437 | 0.254 | 0.083 (12) | 0.900 |
| pick up the black bowl next to the ramekin and place it on the plate | 10 | 219 | 0.282 | 0.232 | 0.172 | 0.132 | 0.078 | 0.806 | 0.603 | 0.467 | 0.273 | 0.062 (16) | 1.000 |
| pick up the black bowl on the cookie box and place it on the plate | 10 | 171 | 0.308 | 0.243 | 0.181 | 0.145 | 0.082 | 0.793 | 0.603 | 0.475 | 0.270 | 0.200 (10) | 1.000 |
| pick up the black bowl on the ramekin and place it on the plate | 10 | 189 | 0.327 | 0.267 | 0.193 | 0.150 | 0.087 | 0.802 | 0.586 | 0.455 | 0.261 | 0.167 (12) | 1.000 |
| pick up the black bowl on the stove and place it on the plate | 10 | 251 | 0.279 | 0.226 | 0.164 | 0.126 | 0.069 | 0.795 | 0.583 | 0.455 | 0.260 | 0.000 (14) | 1.000 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 10 | 241 | 0.334 | 0.268 | 0.197 | 0.147 | 0.086 | 0.806 | 0.581 | 0.443 | 0.258 | 0.308 (13) | 1.000 |
## shadow diagnostics — pi05_libero_10

episodes seen 100, admitted 100; rejections: none
missing labels by outcome: failure: 14 eps (0 rejected), error rows 0/1456, missing decision rows 0, unknown decision counts 0 eps; success: 86 eps (0 rejected), error rows 0/4423, missing decision rows 0, unknown decision counts 0 eps

| task | eps | decisions | disp_K | d_1 | d_2 | d_3 | d_5 | r_1 | r_2 | r_3 | r_5 | ΔBIC>10 frac (n dense) | SR(shadow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pick up the book and place it in the back compartment of the caddy | 10 | 337 | 0.240 | 0.191 | 0.148 | 0.114 | 0.065 | 0.799 | 0.609 | 0.485 | 0.280 | 0.033 (30) | 1.000 |
| put both moka pots on the stove | 10 | 973 | 0.264 | 0.210 | 0.154 | 0.120 | 0.065 | 0.811 | 0.584 | 0.456 | 0.255 | 0.300 (50) | 0.500 |
| put both the alphabet soup and the cream cheese box in the basket | 10 | 652 | 0.328 | 0.283 | 0.206 | 0.152 | 0.083 | 0.805 | 0.574 | 0.444 | 0.256 | 0.268 (41) | 0.800 |
| put both the alphabet soup and the tomato sauce in the basket | 10 | 575 | 0.257 | 0.218 | 0.157 | 0.121 | 0.067 | 0.822 | 0.595 | 0.470 | 0.272 | 0.114 (35) | 1.000 |
| put both the cream cheese box and the butter in the basket | 10 | 513 | 0.315 | 0.256 | 0.195 | 0.147 | 0.081 | 0.802 | 0.593 | 0.465 | 0.265 | 0.200 (35) | 1.000 |
| put the black bowl in the bottom drawer of the cabinet and close it | 10 | 577 | 0.349 | 0.278 | 0.206 | 0.151 | 0.085 | 0.766 | 0.561 | 0.426 | 0.238 | 0.029 (35) | 0.900 |
| put the white mug on the left plate and put the yellow and white mug on the right plate | 10 | 515 | 0.287 | 0.231 | 0.178 | 0.138 | 0.077 | 0.804 | 0.596 | 0.468 | 0.271 | 0.200 (35) | 0.900 |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 10 | 556 | 0.257 | 0.217 | 0.158 | 0.121 | 0.068 | 0.816 | 0.580 | 0.456 | 0.263 | 0.343 (35) | 0.800 |
| put the yellow and white mug in the microwave and close it | 10 | 568 | 0.413 | 0.334 | 0.237 | 0.171 | 0.095 | 0.783 | 0.554 | 0.415 | 0.233 | 0.098 (41) | 0.900 |
| turn on the stove and put the moka pot on it | 10 | 613 | 0.328 | 0.264 | 0.191 | 0.146 | 0.081 | 0.784 | 0.568 | 0.443 | 0.250 | 0.073 (41) | 0.800 |
## shadow diagnostics — groot_libero_spatial

episodes seen 100, admitted 100; rejections: none
missing labels by outcome: failure: 8 eps (0 rejected), error rows 0/352, missing decision rows 0, unknown decision counts 0 eps; success: 92 eps (0 rejected), error rows 0/1995, missing decision rows 0, unknown decision counts 0 eps

| task | eps | decisions | disp_K | d_1 | d_2 | d_4 | d_6 | r_1 | r_2 | r_4 | r_6 | ΔBIC>10 frac (n dense) | SR(shadow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pick up the black bowl between the plate and the ramekin and place it on the plate | 10 | 171 | 0.130 | 0.164 | 0.115 | 0.060 | 0.029 | 1.245 | 0.903 | 0.482 | 0.221 | 0.000 (14) | 1.000 |
| pick up the black bowl from table center and place it on the plate | 10 | 211 | 0.093 | 0.133 | 0.090 | 0.047 | 0.022 | 1.343 | 0.943 | 0.493 | 0.240 | 0.091 (11) | 1.000 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 10 | 292 | 0.117 | 0.156 | 0.105 | 0.053 | 0.027 | 1.346 | 0.902 | 0.477 | 0.236 | 0.000 (12) | 0.900 |
| pick up the black bowl next to the cookie box and place it on the plate | 10 | 218 | 0.106 | 0.141 | 0.097 | 0.051 | 0.025 | 1.303 | 0.912 | 0.498 | 0.235 | 0.056 (18) | 1.000 |
| pick up the black bowl next to the plate and place it on the plate | 10 | 296 | 0.116 | 0.159 | 0.114 | 0.062 | 0.029 | 1.341 | 0.952 | 0.508 | 0.247 | 0.062 (16) | 0.600 |
| pick up the black bowl next to the ramekin and place it on the plate | 10 | 219 | 0.106 | 0.136 | 0.099 | 0.053 | 0.025 | 1.319 | 0.928 | 0.494 | 0.241 | 0.053 (19) | 1.000 |
| pick up the black bowl on the cookie box and place it on the plate | 10 | 201 | 0.113 | 0.154 | 0.107 | 0.055 | 0.026 | 1.297 | 0.902 | 0.478 | 0.230 | 0.118 (17) | 0.900 |
| pick up the black bowl on the ramekin and place it on the plate | 10 | 225 | 0.119 | 0.155 | 0.106 | 0.056 | 0.027 | 1.260 | 0.884 | 0.477 | 0.227 | 0.043 (23) | 0.900 |
| pick up the black bowl on the stove and place it on the plate | 10 | 257 | 0.102 | 0.134 | 0.096 | 0.051 | 0.024 | 1.350 | 0.960 | 0.499 | 0.241 | 0.091 (11) | 0.900 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 10 | 257 | 0.123 | 0.154 | 0.107 | 0.056 | 0.027 | 1.262 | 0.875 | 0.468 | 0.223 | 0.000 (13) | 1.000 |
## shadow diagnostics — groot_libero_10

episodes seen 100, admitted 100; rejections: none
missing labels by outcome: failure: 13 eps (0 rejected), error rows 0/1352, missing decision rows 0, unknown decision counts 0 eps; success: 87 eps (0 rejected), error rows 0/4381, missing decision rows 0, unknown decision counts 0 eps

| task | eps | decisions | disp_K | d_1 | d_2 | d_4 | d_6 | r_1 | r_2 | r_4 | r_6 | ΔBIC>10 frac (n dense) | SR(shadow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pick up the book and place it in the back compartment of the caddy | 10 | 361 | 0.066 | 0.113 | 0.075 | 0.043 | 0.020 | 1.686 | 1.125 | 0.632 | 0.302 | 0.095 (21) | 1.000 |
| put both moka pots on the stove | 10 | 921 | 0.067 | 0.119 | 0.078 | 0.043 | 0.020 | 1.786 | 1.151 | 0.641 | 0.311 | 0.127 (55) | 0.500 |
| put both the alphabet soup and the cream cheese box in the basket | 10 | 544 | 0.079 | 0.134 | 0.093 | 0.050 | 0.024 | 1.681 | 1.136 | 0.633 | 0.300 | 0.100 (40) | 0.900 |
| put both the alphabet soup and the tomato sauce in the basket | 10 | 568 | 0.064 | 0.117 | 0.076 | 0.042 | 0.020 | 1.781 | 1.151 | 0.634 | 0.305 | 0.179 (39) | 1.000 |
| put both the cream cheese box and the butter in the basket | 10 | 497 | 0.070 | 0.123 | 0.085 | 0.047 | 0.022 | 1.738 | 1.189 | 0.663 | 0.312 | 0.192 (26) | 1.000 |
| put the black bowl in the bottom drawer of the cabinet and close it | 10 | 467 | 0.073 | 0.125 | 0.082 | 0.049 | 0.023 | 1.665 | 1.149 | 0.663 | 0.305 | 0.061 (33) | 1.000 |
| put the white mug on the left plate and put the yellow and white mug on the right plate | 10 | 745 | 0.066 | 0.117 | 0.077 | 0.043 | 0.020 | 1.809 | 1.170 | 0.658 | 0.308 | 0.109 (46) | 0.500 |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 10 | 502 | 0.062 | 0.113 | 0.073 | 0.042 | 0.020 | 1.765 | 1.141 | 0.653 | 0.312 | 0.321 (28) | 0.900 |
| put the yellow and white mug in the microwave and close it | 10 | 567 | 0.090 | 0.137 | 0.096 | 0.054 | 0.026 | 1.558 | 1.079 | 0.618 | 0.294 | 0.067 (45) | 1.000 |
| turn on the stove and put the moka pot on it | 10 | 561 | 0.070 | 0.124 | 0.084 | 0.047 | 0.022 | 1.702 | 1.114 | 0.643 | 0.310 | 0.000 (36) | 0.900 |
## Q-B pi05 (m*=2, t*=0.2) — verdict **inconclusive** ⚠ harmful_on_flat

| task | group | n | SR full | SR plain | SR warm | g | Δ | H50 | H25 | recovery | admissible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CloseFridge | cliff | 50 | +0.620 | +0.080 | +0.300 | +0.540 | +0.220 | -0.050 | +0.085 | +0.407 | ok |
| OpenCabinet | cliff | 50 | +0.660 | +0.500 | +0.420 | +0.160 | -0.080 | -0.160 | -0.120 | -0.500 | ok |
| PickPlaceToasterToCounter | cliff | 50 | +0.400 | +0.280 | +0.080 | +0.120 | -0.200 | -0.260 | -0.230 | -1.667 | ok |
| PickPlaceDrawerToCounter | cliff | 50 | +0.320 | +0.180 | +0.100 | +0.140 | -0.080 | -0.150 | -0.115 | -0.571 | ok |
| PickPlaceSinkToCounter | flat | 100 | +1.000 | +1.000 | +0.370 | – | -0.630 | – | – | – | ok |
| OpenDrawer | flat | 100 | +0.620 | +0.750 | +0.710 | – | -0.040 | – | – | – | ok |
| PickPlaceCounterToStove | flat | 100 | +0.840 | +0.940 | +0.160 | – | -0.780 | – | – | – | ok |

cliff macro (n_tasks=4, boot=100000, α=0.00417): g +0.240 [+0.130, +0.350], delta -0.035 [-0.145, +0.075], H50 -0.155 [-0.253, -0.055], H25 -0.095 [-0.194, +0.006]
flat PickPlaceSinkToCounter: Δ -0.630 [-0.771, -0.405] (n10=0, n01=63, n=100)
flat OpenDrawer: Δ -0.040 [-0.244, +0.170] (n10=10, n01=14, n=100)
flat PickPlaceCounterToStove: Δ -0.780 [-0.891, -0.565] (n10=0, n01=78, n=100)

Flat n counts all primary P/W pairs (100); its full reference has 50 episodes. Per-arm Wilson intervals, per-task paired bootstrap intervals and episode NFE totals are in the JSON.

Exploratory budget m=1 (95% intervals; no formal verdict):

| task | paired n | g | Δ | Δ CI95 | status |
|---|---|---|---|---|---|
| CloseFridge | 50 | +0.600 | +0.200 | [+0.080, +0.320] | ok |
| OpenCabinet | 50 | +0.400 | +0.240 | [+0.100, +0.380] | ok |
| PickPlaceToasterToCounter | 50 | +0.400 | +0.060 | [+0.000, +0.140] | ok |
| PickPlaceDrawerToCounter | 50 | +0.260 | +0.040 | [-0.060, +0.140] | ok |
| PickPlaceSinkToCounter | 50 | +0.380 | -0.360 | [-0.520, -0.200] | ok |
| OpenDrawer | 50 | -0.160 | -0.100 | [-0.220, +0.020] | ok |
| PickPlaceCounterToStove | 50 | -0.140 | -0.860 | [-0.940, -0.760] | ok |

Exploratory budget m=3 (95% intervals; no formal verdict):

| task | paired n | g | Δ | Δ CI95 | status |
|---|---|---|---|---|---|
| CloseFridge | 50 | +0.480 | +0.160 | [+0.020, +0.320] | ok |
| OpenCabinet | 50 | +0.060 | -0.120 | [-0.280, +0.060] | ok |
| PickPlaceToasterToCounter | 50 | +0.020 | -0.320 | [-0.460, -0.200] | ok |
| PickPlaceDrawerToCounter | 50 | +0.120 | -0.040 | [-0.180, +0.100] | ok |
| PickPlaceSinkToCounter | 50 | +0.000 | -0.420 | [-0.560, -0.280] | ok |
| OpenDrawer | 50 | -0.060 | +0.020 | [-0.100, +0.160] | ok |
| PickPlaceCounterToStove | 50 | -0.100 | -0.780 | [-0.880, -0.660] | ok |

similarity bins (shadow idx 0..9 only, descriptive): task | joined | low Δ (n) | high Δ (n) | cut
- CloseFridge: 10 | +0.600 (5) | +0.000 (5) | 0.996
- OpenCabinet: 10 | -0.200 (5) | -0.200 (5) | 0.997
- PickPlaceToasterToCounter: 10 | -0.200 (5) | -0.200 (5) | 0.997
- PickPlaceDrawerToCounter: 10 | +0.000 (5) | -0.200 (5) | 0.991
- PickPlaceSinkToCounter: 10 | -0.200 (5) | -0.600 (5) | 0.997
- OpenDrawer: 10 | +0.000 (5) | -0.400 (5) | 0.989
- PickPlaceCounterToStove: 10 | -1.000 (5) | -0.800 (5) | 0.994
## Q-B groot (m*=1, t*=0.75) — verdict **inconclusive** ⚠ harmful_on_flat

| task | group | n | SR full | SR plain | SR warm | g | Δ | H50 | H25 | recovery | admissible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PickPlaceDrawerToCounter | cliff | 50 | +0.660 | +0.400 | +0.480 | +0.260 | +0.080 | -0.050 | +0.015 | +0.308 | ok |
| SlideDishwasherRack | cliff | 50 | +0.540 | +0.400 | +0.420 | +0.140 | +0.020 | -0.050 | -0.015 | +0.143 | ok |
| TurnOnSinkFaucet | cliff | 50 | +0.180 | +0.040 | +0.280 | +0.140 | +0.240 | +0.170 | +0.205 | +1.714 | ok |
| OpenCabinet | cliff | 50 | +0.900 | +0.900 | +0.780 | +0.000 | -0.120 | -0.120 | -0.120 | – | ok |
| PickPlaceCounterToStove | flat | 100 | +0.820 | +0.940 | +0.490 | – | -0.450 | – | – | – | ok |

cliff macro (n_tasks=4, boot=100000, α=0.00417): g +0.135 [+0.035, +0.235], delta +0.055 [-0.045, +0.155], H50 -0.012 [-0.107, +0.080], H25 +0.021 [-0.074, +0.115]
flat PickPlaceCounterToStove: Δ -0.450 [-0.641, -0.196] (n10=4, n01=49, n=100)

Flat n counts all primary P/W pairs (100); its full reference has 50 episodes. Per-arm Wilson intervals, per-task paired bootstrap intervals and episode NFE totals are in the JSON.

Exploratory budget m=2 (95% intervals; no formal verdict):

| task | paired n | g | Δ | Δ CI95 | status |
|---|---|---|---|---|---|
| PickPlaceDrawerToCounter | 50 | +0.060 | -0.100 | [-0.280, +0.080] | ok |
| SlideDishwasherRack | 50 | +0.080 | -0.020 | [-0.100, +0.060] | ok |
| TurnOnSinkFaucet | 50 | -0.180 | -0.060 | [-0.240, +0.120] | ok |
| OpenCabinet | 50 | +0.000 | -0.080 | [-0.200, +0.020] | ok |
| PickPlaceCounterToStove | 50 | -0.140 | -0.140 | [-0.260, -0.040] | ok |

similarity bins (shadow idx 0..9 only, descriptive): task | joined | low Δ (n) | high Δ (n) | cut
- PickPlaceDrawerToCounter: 10 | +0.200 (5) | +0.600 (5) | 0.996
- SlideDishwasherRack: 10 | +0.000 (5) | -0.200 (5) | 0.986
- TurnOnSinkFaucet: 10 | +0.200 (5) | +0.000 (5) | 0.996
- OpenCabinet: 10 | -0.400 (5) | -0.200 (5) | 0.997
- PickPlaceCounterToStove: 10 | -0.400 (5) | -0.200 (5) | 0.997
