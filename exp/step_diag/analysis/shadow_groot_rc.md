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
