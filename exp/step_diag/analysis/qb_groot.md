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
