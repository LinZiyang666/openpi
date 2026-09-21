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
