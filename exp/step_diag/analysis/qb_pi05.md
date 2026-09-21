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
