**K8 search latency — measured 2026-09-27, CPU only.** All table times are milliseconds. `server_stats.json` also stores milliseconds while retaining the original field names (`q_us`, etc.). Triples are p50/p90/p99; Repeat columns show two independent process medians: runs 1/2 except GR00T l10/500 BlindAWM, which uses runs 2/3 without helper overlap. Its first run is retained in raw JSON but excluded from controlled summaries because a verification helper briefly shared CPU 34. Costs below use arithmetic means, not sums of percentiles. Sources, exclusions, commands and caveats follow the tables.

**Controlled warm query replay (CPU 34; ≥1,024 timed vision calls per process/configuration).**

| Model/suite | Episodes | Method | Library entries | Timed vision N | Vision p50/p90/p99 | Repeat p50 | Blind p50/p90/p99 | Blind repeat p50 | Blind templates/run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_spatial | 50 | AWM | 1018 | 2048 | 1.154/1.249/1.342 | 1.159→1.148 | n/a | n/a | n/a |
| pi05_spatial | 50 | MixedJudge | 1018 | 2048 | 1.530/1.686/1.964 | 1.573→1.500 | n/a | n/a | n/a |
| pi05_spatial | 50 | BlindMixedJudge | 1018 | 2048 | 1.796/2.135/2.544 | 1.965→1.661 | 0.475/0.521/0.623 | 0.489→0.459 | 96/96 |
| pi05_spatial | 50 | BlindAWM | 1018 | 2048 | 1.308/2.080/2.337 | 1.902→1.170 | 0.459/0.492/0.533 | 0.456→0.462 | 106/106 |
| pi05_spatial | 50 | WristAWM | 1018 | 2048 | 0.846/0.933/1.093 | 0.864→0.833 | n/a | n/a | n/a |
| pi05_spatial | 50 | WristMixedJudge | 1018 | 2048 | 1.265/1.474/1.875 | 1.351→1.209 | n/a | n/a | n/a |
| pi05_spatial | 50 | BlindWristMixedJudge | 1018 | 2048 | 1.439/1.647/2.080 | 1.495→1.400 | 0.466/0.508/0.651 | 0.467→0.465 | 88/88 |
| pi05_spatial | 50 | ControlG | 1018 | 2048 | 1.290/1.944/2.405 | 1.551→1.186 | n/a | n/a | n/a |
| pi05_spatial | 50 | ControlGS | 1018 | 2048 | 1.405/1.958/2.463 | 1.636→1.273 | n/a | n/a | n/a |
| pi05_spatial | 500 | AWM | 10909 | 2048 | 1.369/1.665/2.079 | 1.472→1.270 | n/a | n/a | n/a |
| pi05_spatial | 500 | MixedJudge | 10909 | 2048 | 2.113/2.806/3.762 | 2.508→1.747 | n/a | n/a | n/a |
| pi05_spatial | 500 | BlindMixedJudge | 10909 | 2048 | 2.184/3.111/3.541 | 2.859→1.853 | 0.478/0.552/0.705 | 0.500→0.467 | 102/102 |
| pi05_spatial | 500 | BlindAWM | 10909 | 2048 | 1.335/1.561/1.787 | 1.452→1.281 | 0.491/0.559/0.605 | 0.482→0.503 | 106/106 |
| pi05_spatial | 500 | WristAWM | 10909 | 2048 | 0.902/1.002/1.086 | 0.913→0.896 | n/a | n/a | n/a |
| pi05_spatial | 500 | WristMixedJudge | 10909 | 2048 | 1.393/1.525/1.884 | 1.421→1.369 | n/a | n/a | n/a |
| pi05_spatial | 500 | BlindWristMixedJudge | 10909 | 2048 | 1.539/1.689/1.946 | 1.581→1.519 | 0.471/0.509/0.570 | 0.484→0.459 | 97/97 |
| pi05_spatial | 500 | ControlG | 10909 | 2048 | 1.424/1.572/1.788 | 1.452→1.398 | n/a | n/a | n/a |
| pi05_spatial | 500 | ControlGS | 10909 | 2048 | 1.463/1.630/1.783 | 1.460→1.468 | n/a | n/a | n/a |
| pi05_l10 | 50 | AWM | 2640 | 2282 | 1.167/1.263/1.363 | 1.170→1.163 | n/a | n/a | n/a |
| pi05_l10 | 50 | MixedJudge | 2640 | 2282 | 1.646/2.019/2.384 | 1.800→1.560 | n/a | n/a | n/a |
| pi05_l10 | 50 | BlindMixedJudge | 2640 | 2282 | 1.786/2.073/2.504 | 1.859→1.740 | 0.468/0.525/0.592 | 0.464→0.475 | 201/201 |
| pi05_l10 | 50 | BlindAWM | 2640 | 2282 | 1.272/1.440/1.617 | 1.287→1.261 | 0.482/0.530/0.589 | 0.471→0.498 | 228/228 |
| pi05_l10 | 50 | WristAWM | 2640 | 2282 | 0.861/0.931/1.135 | 0.864→0.858 | n/a | n/a | n/a |
| pi05_l10 | 50 | WristMixedJudge | 2640 | 2282 | 1.269/1.366/1.516 | 1.269→1.268 | n/a | n/a | n/a |
| pi05_l10 | 50 | BlindWristMixedJudge | 2640 | 2282 | 1.457/1.606/1.791 | 1.490→1.433 | 0.468/0.500/0.541 | 0.472→0.462 | 202/202 |
| pi05_l10 | 50 | ControlG | 2640 | 2282 | 1.261/1.467/1.649 | 1.246→1.276 | n/a | n/a | n/a |
| pi05_l10 | 50 | ControlGS | 2640 | 2282 | 1.310/1.445/1.586 | 1.317→1.302 | n/a | n/a | n/a |
| pi05_l10 | 500 | AWM | 29472 | 2282 | 1.405/1.733/2.229 | 1.514→1.356 | n/a | n/a | n/a |
| pi05_l10 | 500 | MixedJudge | 29472 | 2282 | 2.096/2.641/3.642 | 2.337→1.958 | n/a | n/a | n/a |
| pi05_l10 | 500 | BlindMixedJudge | 29472 | 2282 | 2.128/2.386/2.821 | 2.151→2.104 | 0.467/0.508/0.576 | 0.470→0.464 | 204/204 |
| pi05_l10 | 500 | BlindAWM | 29472 | 2282 | 1.426/1.575/1.755 | 1.408→1.443 | 0.478/0.537/0.585 | 0.478→0.477 | 240/240 |
| pi05_l10 | 500 | WristAWM | 29472 | 2282 | 1.025/1.141/1.312 | 1.031→1.022 | n/a | n/a | n/a |
| pi05_l10 | 500 | WristMixedJudge | 29472 | 2282 | 1.584/1.802/2.406 | 1.631→1.552 | n/a | n/a | n/a |
| pi05_l10 | 500 | BlindWristMixedJudge | 29472 | 2282 | 1.796/2.004/2.238 | 1.812→1.782 | 0.488/0.534/0.606 | 0.482→0.493 | 200/200 |
| pi05_l10 | 500 | ControlG | 29472 | 2282 | 1.658/1.934/2.256 | 1.681→1.648 | n/a | n/a | n/a |
| pi05_l10 | 500 | ControlGS | 29472 | 2282 | 1.771/2.074/2.478 | 1.762→1.794 | n/a | n/a | n/a |
| groot_spatial | 50 | AWM | 1063 | 2048 | 1.138/1.239/1.419 | 1.119→1.154 | n/a | n/a | n/a |
| groot_spatial | 50 | BlindAWM | 1063 | 2048 | 1.201/1.290/1.394 | 1.192→1.212 | 0.461/0.494/0.540 | 0.457→0.463 | 99/99 |
| groot_spatial | 50 | ControlG | 1063 | 2048 | 1.201/1.286/1.367 | 1.205→1.198 | n/a | n/a | n/a |
| groot_spatial | 50 | ControlGS | 1063 | 2048 | 1.281/1.415/1.508 | 1.281→1.281 | n/a | n/a | n/a |
| groot_spatial | 500 | AWM | 11751 | 2048 | 1.253/1.393/1.528 | 1.247→1.264 | n/a | n/a | n/a |
| groot_spatial | 500 | BlindAWM | 11751 | 2048 | 1.300/1.397/1.520 | 1.300→1.300 | 0.478/0.542/0.900 | 0.482→0.472 | 107/107 |
| groot_spatial | 500 | ControlG | 11751 | 2048 | 1.366/1.545/1.740 | 1.364→1.368 | n/a | n/a | n/a |
| groot_spatial | 500 | ControlGS | 11751 | 2048 | 1.542/1.801/2.022 | 1.479→1.589 | n/a | n/a | n/a |
| groot_l10 | 50 | AWM | 2645 | 2088 | 1.188/1.283/1.384 | 1.171→1.204 | n/a | n/a | n/a |
| groot_l10 | 50 | BlindAWM | 2645 | 2088 | 1.232/1.327/1.598 | 1.226→1.238 | 0.465/0.508/0.596 | 0.458→0.472 | 213/213 |
| groot_l10 | 50 | ControlG | 2645 | 2088 | 1.253/1.379/1.500 | 1.250→1.258 | n/a | n/a | n/a |
| groot_l10 | 50 | ControlGS | 2645 | 2088 | 1.320/1.495/1.635 | 1.328→1.311 | n/a | n/a | n/a |
| groot_l10 | 500 | AWM | 29631 | 2088 | 1.383/1.510/1.681 | 1.383→1.384 | n/a | n/a | n/a |
| groot_l10 | 500 | BlindAWM | 29631 | 2088 | 1.432/1.576/1.794 | 1.411→1.461 | 0.469/0.516/0.564 | 0.453→0.484 | 213/213 |
| groot_l10 | 500 | ControlG | 29631 | 2088 | 1.681/1.972/2.282 | 1.674→1.685 | n/a | n/a | n/a |
| groot_l10 | 500 | ControlGS | 29631 | 2088 | 1.768/2.149/2.533 | 1.773→1.767 | n/a | n/a | n/a |
| pi05_l10 | 500 | K7 b0g | 29472 | 2282 | 2.091/2.474/2.999 | 2.084→2.110 | n/a | n/a | n/a |
| pi05_l10 | 500 | K7 ph1g | 29472 | 2282 | 2.045/2.290/2.735 | 2.047→2.044 | 0.469/0.535/0.689 | 0.475→0.465 | 145/145 |
| pi05_l10 | 500 | K7 ph2g | 29472 | 2282 | 2.096/2.419/2.765 | 2.084→2.114 | 0.483/0.519/0.560 | 0.479→0.487 | 204/204 |
| pi05_l10 | 500 | K7 tail1ug | 29472 | 2282 | 2.097/2.466/3.000 | 2.090→2.103 | 0.223/0.245/0.290 | 0.225→0.221 | 216/216 |
| pi05_l10 | 50 | K7 ph2g | 2640 | 2282 | 1.686/1.910/2.188 | 1.696→1.674 | 0.493/0.546/0.627 | 0.491→0.495 | 201/201 |
| pi05_l10 | 50 | K7 tail1ug | 2640 | 2282 | 1.657/1.884/2.187 | 1.663→1.654 | 0.238/0.300/0.354 | 0.264→0.225 | 224/224 |
| pi05_spatial | 500 | K7 ph2g | 10909 | 2048 | 1.798/1.934/2.087 | 1.798→1.798 | 0.478/0.515/0.574 | 0.466→0.484 | 102/102 |
| pi05_spatial | 500 | K7 tail1ug | 10909 | 2048 | 1.804/1.941/2.205 | 1.821→1.787 | 0.223/0.256/0.319 | 0.227→0.218 | 113/113 |

The nominal 50-episode π0.5 spatial library contains 49 stored episodes (1,018 entries). The 500-library AWM is CL2 with constructor defaults (`lib="big", kref=8`); the 50-library CL2 uses `lib="current", kref=5`. For the original 52 configurations, blind rows are successful B=2, `phase_particles`, all-gates decisions; failed attempts that request vision have separate raw distributions. WristAWM is the fitted base extracted from the exact K3 WristMixedJudge pickle. ControlG/GS use the exact existing G/GS fits. See `configs.json` for every path, constructor argument and per-task row count. Supplemental K1 fits in `/tmp/k1_blind_fits` cover cells without a closed-loop fit; no refit was performed.

**Coverage and unavailable cells.**

| Method | Model | Suites | Scales | Reason unmeasured |
| --- | --- | --- | --- | --- |
| BlindMixedJudge | groot | l10, spatial | 50 and 500 | no matching fitted artifact in inventory |
| BlindWristMixedJudge | groot | l10, spatial | 50 and 500 | implementation pi05-only |
| MixedJudge | groot | l10, spatial | 50 and 500 | no matching fitted artifact in inventory |
| WristAWM | groot | l10, spatial | 50 and 500 | implementation pi05-only |
| WristMixedJudge | groot | l10, spatial | 50 and 500 | implementation pi05-only |

The initial fit inventory contains `os_closed_loop/*/fits/*.pkl`, the K1 handback fits, and K3 derived fits; `late_fit_inventory.json` records eight K7 artifacts created afterward and discovered during the final inventory check. It includes nonrequested AWM3/recovery/inference variants for audit; the measurement matrix selects the requested method families with the deployed default guard recipe, not every arm-level ablation. GR00T MixedJudge/BlindMixedJudge can execute in existing CPU selftests, but no matching saved fit was found in these fit inventories. The original 52 measured configurations cover every initially available model × suite × scale × requested method tuple; BlindWristMixedJudge adds four configurations beyond the minimum wrist request. Eight late K7 VisionConfirmedBlindMixedJudge artifacts (which export a BlindMixedJudge alias) were then measured in two fresh processes each, giving 60 total configurations. The K7 labels preserve each arm suffix: b0g disables blindness; ph1g/ph2g use phase particles with B=1/B=2 and all gates; tail1ug uses anchor-tail, B=1, and budget-only gates. No successful blind timing exists for B=0 by definition. Their timings are supplemental; the original frontier cost ledger remains tied to the original methods.

**Exclusive component means from separate instrumented replays.**

| Cell/episodes | Method | PCA/state | Score + whitening | Top-k | Synthesis/selection | Judge/guards | Other/timers | Profile total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_spatial/50 | AWM | 0.663 | 0.124 | 0.017 | 0.160 | 0.000 | 0.250 | 1.214 |
| pi05_spatial/50 | MixedJudge | 0.681 | 0.217 | 0.029 | 0.122 | 0.530 | 0.091 | 1.670 |
| pi05_spatial/50 | BlindMixedJudge | 0.809 | 0.223 | 0.032 | 0.169 | 0.589 | 0.183 | 2.005 |
| pi05_spatial/50 | BlindAWM | 0.939 | 0.134 | 0.018 | 0.182 | 0.000 | 0.308 | 1.580 |
| pi05_spatial/50 | WristAWM | 0.357 | 0.121 | 0.017 | 0.159 | 0.000 | 0.265 | 0.919 |
| pi05_spatial/50 | WristMixedJudge | 0.411 | 0.202 | 0.030 | 0.125 | 0.552 | 0.101 | 1.421 |
| pi05_spatial/50 | BlindWristMixedJudge | 0.393 | 0.206 | 0.031 | 0.172 | 0.607 | 0.196 | 1.604 |
| pi05_spatial/50 | ControlG | 0.838 | 0.196 | 0.015 | 0.082 | 0.000 | 0.320 | 1.452 |
| pi05_spatial/50 | ControlGS | 0.777 | 0.200 | 0.015 | 0.138 | 0.000 | 0.327 | 1.457 |
| pi05_spatial/500 | AWM | 0.767 | 0.195 | 0.030 | 0.174 | 0.000 | 0.276 | 1.442 |
| pi05_spatial/500 | MixedJudge | 1.025 | 0.353 | 0.054 | 0.131 | 0.620 | 0.096 | 2.279 |
| pi05_spatial/500 | BlindMixedJudge | 1.054 | 0.373 | 0.056 | 0.179 | 0.672 | 0.190 | 2.525 |
| pi05_spatial/500 | BlindAWM | 0.677 | 0.188 | 0.029 | 0.187 | 0.000 | 0.313 | 1.394 |
| pi05_spatial/500 | WristAWM | 0.351 | 0.163 | 0.028 | 0.159 | 0.000 | 0.271 | 0.972 |
| pi05_spatial/500 | WristMixedJudge | 0.360 | 0.266 | 0.053 | 0.124 | 0.607 | 0.099 | 1.510 |
| pi05_spatial/500 | BlindWristMixedJudge | 0.373 | 0.272 | 0.054 | 0.176 | 0.663 | 0.194 | 1.732 |
| pi05_spatial/500 | ControlG | 0.707 | 0.295 | 0.063 | 0.085 | 0.000 | 0.332 | 1.483 |
| pi05_spatial/500 | ControlGS | 0.664 | 0.289 | 0.063 | 0.134 | 0.000 | 0.329 | 1.479 |
| pi05_l10/50 | AWM | 0.699 | 0.137 | 0.018 | 0.158 | 0.000 | 0.255 | 1.268 |
| pi05_l10/50 | MixedJudge | 0.811 | 0.250 | 0.035 | 0.130 | 0.564 | 0.097 | 1.887 |
| pi05_l10/50 | BlindMixedJudge | 0.691 | 0.237 | 0.035 | 0.160 | 0.581 | 0.179 | 1.884 |
| pi05_l10/50 | BlindAWM | 0.672 | 0.139 | 0.019 | 0.182 | 0.000 | 0.306 | 1.318 |
| pi05_l10/50 | WristAWM | 0.351 | 0.128 | 0.018 | 0.156 | 0.000 | 0.265 | 0.918 |
| pi05_l10/50 | WristMixedJudge | 0.366 | 0.218 | 0.035 | 0.126 | 0.575 | 0.102 | 1.421 |
| pi05_l10/50 | BlindWristMixedJudge | 0.373 | 0.215 | 0.035 | 0.167 | 0.610 | 0.194 | 1.594 |
| pi05_l10/50 | ControlG | 0.677 | 0.206 | 0.022 | 0.086 | 0.000 | 0.326 | 1.318 |
| pi05_l10/50 | ControlGS | 0.682 | 0.205 | 0.021 | 0.133 | 0.000 | 0.318 | 1.360 |
| pi05_l10/500 | AWM | 0.773 | 0.292 | 0.046 | 0.175 | 0.000 | 0.293 | 1.578 |
| pi05_l10/500 | MixedJudge | 0.829 | 0.496 | 0.083 | 0.129 | 0.682 | 0.095 | 2.313 |
| pi05_l10/500 | BlindMixedJudge | 0.730 | 0.476 | 0.084 | 0.168 | 0.725 | 0.186 | 2.370 |
| pi05_l10/500 | BlindAWM | 0.701 | 0.275 | 0.046 | 0.183 | 0.000 | 0.323 | 1.529 |
| pi05_l10/500 | WristAWM | 0.364 | 0.237 | 0.047 | 0.169 | 0.000 | 0.299 | 1.116 |
| pi05_l10/500 | WristMixedJudge | 0.383 | 0.372 | 0.085 | 0.126 | 0.689 | 0.104 | 1.758 |
| pi05_l10/500 | BlindWristMixedJudge | 0.362 | 0.367 | 0.085 | 0.175 | 0.738 | 0.199 | 1.926 |
| pi05_l10/500 | ControlG | 0.681 | 0.432 | 0.154 | 0.087 | 0.000 | 0.344 | 1.699 |
| pi05_l10/500 | ControlGS | 0.691 | 0.443 | 0.156 | 0.140 | 0.000 | 0.356 | 1.785 |
| groot_spatial/50 | AWM | 0.656 | 0.122 | 0.016 | 0.157 | 0.000 | 0.243 | 1.194 |
| groot_spatial/50 | BlindAWM | 0.665 | 0.126 | 0.016 | 0.178 | 0.000 | 0.289 | 1.274 |
| groot_spatial/50 | ControlG | 0.664 | 0.183 | 0.015 | 0.086 | 0.000 | 0.305 | 1.253 |
| groot_spatial/50 | ControlGS | 0.666 | 0.186 | 0.015 | 0.152 | 0.000 | 0.309 | 1.327 |
| groot_spatial/500 | AWM | 0.683 | 0.194 | 0.030 | 0.174 | 0.000 | 0.271 | 1.352 |
| groot_spatial/500 | BlindAWM | 0.669 | 0.190 | 0.029 | 0.190 | 0.000 | 0.306 | 1.384 |
| groot_spatial/500 | ControlG | 0.663 | 0.285 | 0.063 | 0.090 | 0.000 | 0.318 | 1.418 |
| groot_spatial/500 | ControlGS | 0.740 | 0.301 | 0.064 | 0.165 | 0.000 | 0.337 | 1.608 |
| groot_l10/50 | AWM | 0.677 | 0.134 | 0.018 | 0.162 | 0.000 | 0.261 | 1.253 |
| groot_l10/50 | BlindAWM | 0.688 | 0.144 | 0.019 | 0.188 | 0.000 | 0.318 | 1.357 |
| groot_l10/50 | ControlG | 0.687 | 0.210 | 0.022 | 0.091 | 0.000 | 0.328 | 1.337 |
| groot_l10/50 | ControlGS | 0.670 | 0.202 | 0.021 | 0.157 | 0.000 | 0.321 | 1.372 |
| groot_l10/500 | AWM | 0.681 | 0.273 | 0.045 | 0.171 | 0.000 | 0.285 | 1.456 |
| groot_l10/500 | BlindAWM | 0.689 | 0.277 | 0.045 | 0.190 | 0.000 | 0.319 | 1.519 |
| groot_l10/500 | ControlG | 0.688 | 0.442 | 0.153 | 0.094 | 0.000 | 0.348 | 1.725 |
| groot_l10/500 | ControlGS | 0.683 | 0.440 | 0.154 | 0.171 | 0.000 | 0.350 | 1.799 |
| pi05_l10/500 | K7 b0g | 0.710 | 0.473 | 0.084 | 0.175 | 0.720 | 0.132 | 2.293 |
| pi05_l10/500 | K7 ph1g | 0.703 | 0.463 | 0.082 | 0.165 | 0.698 | 0.125 | 2.236 |
| pi05_l10/500 | K7 ph2g | 0.717 | 0.475 | 0.083 | 0.169 | 0.704 | 0.126 | 2.274 |
| pi05_l10/500 | K7 tail1ug | 0.709 | 0.468 | 0.082 | 0.167 | 0.697 | 0.127 | 2.250 |
| pi05_l10/50 | K7 ph2g | 0.684 | 0.247 | 0.036 | 0.167 | 0.577 | 0.128 | 1.840 |
| pi05_l10/50 | K7 tail1ug | 0.692 | 0.246 | 0.035 | 0.166 | 0.592 | 0.128 | 1.860 |
| pi05_spatial/500 | K7 ph2g | 0.683 | 0.324 | 0.054 | 0.171 | 0.634 | 0.127 | 1.992 |
| pi05_spatial/500 | K7 tail1ug | 0.694 | 0.332 | 0.055 | 0.173 | 0.640 | 0.129 | 2.024 |

Instrumented totals are separate from the primary uninstrumented timings. Existing `prof.section` scopes and small function wrappers accumulate exclusive time, including wrapper overhead. Score includes per-task GEMV, distance normalization/median, fresh-policy-tail continuity, and camera similarity auxiliaries. Synthesis includes residual selection work and bookkeeping inside synthesis scopes. Judge includes V7 features/calibration, visual/proprioceptive motion tests and guards. R4 final scalar bookkeeping outside existing scopes remains in Other; these columns should not be interpreted as a cycle-accurate additive decomposition of the uninstrumented median. Raw JSON retains every individual component distribution and per-query profile row; profile wrapping retained identical sampled actions/top-k.

**Deployed native CP1 reference (same original backend for method scales 50 and 500).**

| Model/suite | Native entries | p50/p90/p99 | R1→R2 p50 | Matched recorded top-1 | Compared N |
| --- | --- | --- | --- | --- | --- |
| pi05/spatial | 1018 | 2.118/2.933/4.169 | 2.102→2.130 | 578 | 578 |
| pi05/l10 | 2640 | 7.458/9.211/12.836 | 7.410→7.497 | 1196 | 1196 |
| groot/spatial | 1063 | 2.152/2.921/4.196 | 2.045→2.267 | 502 | 502 |
| groot/l10 | 2645 | 7.390/11.594/13.211 | 7.099→7.665 | 1082 | 1082 |

This is `WeightedScoreSumKnnStrategy.search(SearchContext)` using the real configuration, frozen in-memory backend, library pickle and task filter. CUDA is disabled and Torch intra/inter-op thread counts are one. The live `native_us` shadow path searches the original current library even when AWM uses its separate 500-episode library; reporting it as native 500-episode retrieval would be incorrect. No synthetic 500-native backend was substituted.

**Key construction and plugin overhead.**

| Operation | p50/p90/p99 | R1→R2 p50 |
| --- | --- | --- |
| groot_l10_500_AWM_whiten_state | 0.004/0.004/0.017 | 0.004→0.004 |
| groot_l10_50_AWM_whiten_state | 0.004/0.004/0.016 | 0.004→0.004 |
| groot_l10_pool2_cpu | 7.248/7.620/7.973 | 7.189→7.281 |
| groot_spatial_500_AWM_whiten_state | 0.005/0.005/0.016 | 0.005→0.005 |
| groot_spatial_50_AWM_whiten_state | 0.005/0.005/0.016 | 0.005→0.005 |
| groot_spatial_pool2_cpu | 7.261/7.632/8.053 | 7.192→7.312 |
| history_two_pooled_key_copies_128 | 0.017/0.023/0.039 | 0.017→0.017 |
| history_two_pooled_key_copies_16 | 0.016/0.021/0.038 | 0.016→0.015 |
| history_two_pooled_key_copies_64 | 0.016/0.017/0.035 | 0.016→0.016 |
| pi05_l10_500_AWM_whiten_state | 0.005/0.005/0.017 | 0.004→0.005 |
| pi05_l10_500_WristAWM_whiten_state | 0.003/0.003/0.012 | 0.003→0.003 |
| pi05_l10_50_AWM_whiten_state | 0.005/0.005/0.016 | 0.005→0.005 |
| pi05_l10_50_WristAWM_whiten_state | 0.003/0.003/0.013 | 0.003→0.003 |
| pi05_l10_pool2_cpu | 7.293/7.629/7.953 | 7.173→7.356 |
| pi05_spatial_500_AWM_whiten_state | 0.005/0.005/0.018 | 0.004→0.005 |
| pi05_spatial_500_WristAWM_whiten_state | 0.003/0.003/0.011 | 0.003→0.003 |
| pi05_spatial_50_AWM_whiten_state | 0.005/0.005/0.017 | 0.005→0.005 |
| pi05_spatial_50_WristAWM_whiten_state | 0.003/0.003/0.009 | 0.003→0.003 |
| pi05_spatial_pool2_cpu | 7.165/7.620/8.050 | 7.199→7.137 |
| plugin_emit_blind | 0.419/0.476/0.523 | 0.403→0.425 |
| plugin_emit_vision | 0.314/0.358/0.411 | 0.305→0.322 |

CPU pooling calls the deployed `_spatial_pool_tokens` twice on real stored token tensors, converted to float32 before timing. This is a CPU reference only: live stage-1 GPU pooling/D2H was not measured and is outside `q_us`. Stored fp16 token quantization can prevent exact equality to the historical pooled key; raw auxiliary JSON records maximum differences. Whitening is the exact task-specific `x @ Wf - shift` expression after PCA; its inputs are prepared before timing. History rows time two real `_Buf.append()` pooled-key copies with preallocated capacity, excluding buffer-growth outliers and the rest of `_push_inputs`. `plugin_emit` invokes the actual cleaning/JSON/open/append/write/close path on representative real rows into `/tmp/k8_latency`; it excludes server filesystem/GIL contention. Server residual `search−q−native`, below, measures the full in-search bookkeeping boundary. Emit runs after that boundary, so it is additional.

**Fixed-task scaling (task 0, both cached and policy-history inputs).**

| Cell/base scale | Synthetic multiplier | Suite episodes | Entries in task block | p50/p90/p99 | R1→R2 p50 |
| --- | --- | --- | --- | --- | --- |
| groot_l10_500_AWM | 1 | 500 | 3138 | 1.360/1.502/1.669 | 1.369→1.354 |
| groot_l10_500_AWM | 2 | 1000 | 6276 | 1.571/1.749/1.921 | 1.581→1.559 |
| groot_l10_500_AWM | 4 | 2000 | 12552 | 1.973/2.332/2.598 | 1.998→1.956 |
| groot_l10_500_AWM | 10 | 5000 | 31380 | 3.716/4.687/5.487 | 3.750→3.692 |
| groot_l10_50_AWM | 1 | 50 | 276 | 1.176/1.315/1.516 | 1.182→1.170 |
| groot_spatial_500_AWM | 1 | 500 | 1223 | 1.244/1.377/1.553 | 1.243→1.247 |
| groot_spatial_500_AWM | 2 | 1000 | 2446 | 1.311/1.462/1.670 | 1.324→1.302 |
| groot_spatial_500_AWM | 4 | 2000 | 4892 | 1.452/1.610/1.923 | 1.471→1.441 |
| groot_spatial_500_AWM | 10 | 5000 | 12230 | 1.949/2.280/2.647 | 1.975→1.923 |
| groot_spatial_50_AWM | 1 | 50 | 82 | 1.143/1.238/1.319 | 1.126→1.157 |
| pi05_l10_500_AWM | 1 | 500 | 2841 | 1.352/1.465/1.566 | 1.357→1.349 |
| pi05_l10_500_AWM | 2 | 1000 | 5682 | 1.515/1.671/1.806 | 1.517→1.512 |
| pi05_l10_500_AWM | 4 | 2000 | 11364 | 1.914/2.187/2.474 | 1.935→1.900 |
| pi05_l10_500_AWM | 10 | 5000 | 28410 | 3.365/4.269/5.081 | 3.511→3.075 |
| pi05_l10_50_AWM | 1 | 50 | 284 | 1.176/1.271/1.374 | 1.184→1.168 |
| pi05_spatial_500_AWM | 1 | 500 | 950 | 1.229/1.340/1.532 | 1.216→1.243 |
| pi05_spatial_500_AWM | 2 | 1000 | 1900 | 1.292/1.404/1.518 | 1.279→1.304 |
| pi05_spatial_500_AWM | 4 | 2000 | 3800 | 1.417/1.549/1.675 | 1.394→1.439 |
| pi05_spatial_500_AWM | 10 | 5000 | 9500 | 1.823/2.075/2.331 | 1.771→1.859 |
| pi05_spatial_50_AWM | 1 | 50 | 80 | 1.150/1.235/1.302 | 1.147→1.157 |

The real 50/500 fits have different PCA bases and metrics as deployed. Synthetic 1k/2k/5k episode cases tile every row-indexed fitted task array of the 500 fit by 2/4/10; repeated IDs still address the original actions. PCA/whitening dimensions and query inputs stay fixed. This isolates candidate-block size and exact deployed AWM scoring/top-k costs. Duplicate rows/ties and repeated payloads are not a forecast of a newly fitted diverse 5k-episode library; no synthetic SR is claimed.

**Concurrency: one Python process, eight allowed logical CPUs, shared fitted arrays and per-thread method state.**

| Configuration | Threads | p50/p90/p99 | R1→R2 p50 | Queries/s mean |
| --- | --- | --- | --- | --- |
| groot_l10_500_AWM | 1 | 1.431/2.056/2.742 | 1.438→1.425 | 643.2 |
| groot_l10_500_AWM | 4 | 5.790/8.022/10.528 | 5.770→5.806 | 620.4 |
| groot_l10_500_AWM | 8 | 12.361/17.496/23.230 | 12.080→12.647 | 570.9 |
| groot_l10_500_AWM | 16 | 22.858/32.616/42.372 | 22.339→23.515 | 596.8 |
| groot_l10_500_AWM | 24 | 34.397/52.182/68.288 | 33.540→35.195 | 585.4 |
| groot_l10_500_AWM | 32 | 43.095/65.236/85.809 | 42.906→43.272 | 590.6 |
| groot_l10_50_AWM | 1 | 1.195/1.336/1.527 | 1.215→1.183 | 812.4 |
| groot_l10_50_AWM | 4 | 5.146/7.505/11.807 | 4.962→5.424 | 652.2 |
| groot_l10_50_AWM | 8 | 10.861/16.932/24.402 | 10.162→11.630 | 609.3 |
| groot_l10_50_AWM | 16 | 20.815/33.720/48.182 | 19.717→21.848 | 601.7 |
| groot_l10_50_AWM | 24 | 31.173/51.259/76.764 | 29.790→32.873 | 597.0 |
| groot_l10_50_AWM | 32 | 38.586/66.315/98.953 | 39.741→37.423 | 600.2 |
| pi05_l10_500_AWM | 1 | 1.360/1.474/1.602 | 1.368→1.353 | 726.1 |
| pi05_l10_500_AWM | 4 | 6.026/8.117/10.440 | 6.116→5.942 | 640.9 |
| pi05_l10_500_AWM | 8 | 12.425/17.439/22.813 | 12.431→12.403 | 614.8 |
| pi05_l10_500_AWM | 16 | 24.943/35.793/47.853 | 24.830→25.130 | 601.6 |
| pi05_l10_500_AWM | 24 | 35.899/54.636/71.906 | 36.733→35.253 | 592.9 |
| pi05_l10_500_AWM | 32 | 49.475/73.190/98.234 | 49.577→49.329 | 601.1 |
| pi05_l10_500_BlindMixedJudge | 1 | 2.163/2.425/2.886 | 2.128→2.197 | 449.1 |
| pi05_l10_500_BlindMixedJudge | 4 | 10.168/13.242/17.110 | 9.769→10.729 | 383.1 |
| pi05_l10_500_BlindMixedJudge | 8 | 24.425/35.786/46.897 | 30.399→20.360 | 318.6 |
| pi05_l10_500_BlindMixedJudge | 16 | 46.985/63.440/80.560 | 52.680→42.095 | 327.4 |
| pi05_l10_500_BlindMixedJudge | 24 | 70.221/95.028/117.341 | 73.676→67.053 | 304.7 |
| pi05_l10_500_BlindMixedJudge | 32 | 100.295/135.085/168.162 | 98.089→102.244 | 296.8 |
| pi05_l10_500_MixedJudge | 1 | 1.991/2.200/2.487 | 1.979→2.004 | 492.2 |
| pi05_l10_500_MixedJudge | 4 | 8.821/11.053/13.854 | 8.867→8.763 | 445.1 |
| pi05_l10_500_MixedJudge | 8 | 18.657/23.681/29.146 | 18.583→18.728 | 422.4 |
| pi05_l10_500_MixedJudge | 16 | 37.105/49.072/59.053 | 36.886→37.292 | 411.0 |
| pi05_l10_500_MixedJudge | 24 | 53.707/71.212/88.223 | 53.407→54.350 | 408.3 |
| pi05_l10_500_MixedJudge | 32 | 74.556/99.499/118.987 | 75.008→74.019 | 404.6 |
| pi05_l10_50_AWM | 1 | 1.172/1.284/1.373 | 1.168→1.178 | 837.7 |
| pi05_l10_50_AWM | 4 | 5.600/7.757/9.761 | 5.593→5.619 | 696.2 |
| pi05_l10_50_AWM | 8 | 11.482/17.117/24.278 | 11.549→11.371 | 661.6 |
| pi05_l10_50_AWM | 16 | 21.969/35.422/51.733 | 22.163→21.815 | 656.4 |
| pi05_l10_50_AWM | 24 | 31.454/52.863/74.062 | 31.286→31.695 | 646.0 |
| pi05_l10_50_AWM | 32 | 44.202/70.907/104.049 | 44.452→43.947 | 648.0 |
| pi05_l10_50_MixedJudge | 1 | 1.599/1.767/1.961 | 1.597→1.600 | 612.9 |
| pi05_l10_50_MixedJudge | 4 | 7.784/9.967/12.775 | 7.784→7.781 | 502.9 |
| pi05_l10_50_MixedJudge | 8 | 16.374/21.546/26.713 | 16.351→16.384 | 478.9 |
| pi05_l10_50_MixedJudge | 16 | 31.942/44.733/55.769 | 31.922→31.971 | 477.3 |
| pi05_l10_50_MixedJudge | 24 | 45.088/65.116/83.359 | 44.987→45.236 | 471.3 |
| pi05_l10_50_MixedJudge | 32 | 62.903/89.386/111.873 | 62.348→63.512 | 469.5 |

Each thread count replays the same 1,024 calls per process run, after warmup and a barrier. Sixteen-call episode segments receive isolated method clones warmed on their exact episode prefixes. Output digests match across all thread counts. The one-thread baseline uses CPU 34; the concurrent runs use all eight allowed logical CPUs. Timers start inside each call, excluding executor queue time. No global lock encloses query; arrays are shared with `plugin.clone_method(strict=True)`, matching K6’s state-isolation design. Counts 16/24/32 deliberately oversubscribe these eight logical CPUs (four physical cores); NumPy/Torch/BLAS remain single-threaded. These numbers combine GIL scheduling, CPU oversubscription and memory-bandwidth contention: they quantify shared-process contention, not a uniquely identified GIL-only causal effect. The sweep covers l10 AWM for both models/scales, R3 guard-only both scales, and R4 500 BlindMixedJudge vision queries. It does not model GPU batching, RPC queueing, or live blind/vision mixes.

**Logged in-server evidence, read-only snapshot.** Triples p50/p90/p99 in ms; N is the q sample count. Native dashes mean disabled/absent, never zero-cost native retrieval.

| Mode | Model/suite | Library | Method/verdict | Decision | N | q | search | native | Blind prepare | Blind output | Search bookkeeping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CONCURRENT | groot/l10 | 50 | AWM CL2 pure_cache | vision | 38173 | 2.752/3.479/4.376 | 7.956/10.110/13.255 | 4.688/6.270/8.309 | — | — | 0.336/0.514/3.628 |
| CONCURRENT | groot/l10 | 50 | AWM3 | vision | 37652 | 2.780/3.365/4.147 | 7.433/9.359/12.871 | 4.150/5.735/7.800 | — | — | 0.333/0.503/3.753 |
| CONCURRENT | groot/l10 | 50 | B0TopkConsensus | vision | 39962 | 8.702/12.521/16.263 | 14.459/20.041/25.979 | 5.225/7.253/9.503 | — | — | 0.362/0.551/4.111 |
| CONCURRENT | groot/l10 | 50 | StuckRecovery | vision | 36720 | 3.493/4.523/5.539 | 8.913/11.304/14.697 | 4.886/6.631/8.636 | — | — | 0.348/0.529/3.854 |
| CONCURRENT | groot/l10 | 50 | native | vision | 39669 | 4.753/6.398/8.609 | 4.753/6.398/8.609 | 4.753/6.398/8.609 | — | — | — |
| CONCURRENT | groot/l10 | 500 | AWM CL2 pure_cache | vision | 33390 | 2.957/3.564/4.432 | 7.523/9.544/12.691 | 4.076/5.654/7.738 | — | — | 0.344/0.505/3.586 |
| CONCURRENT | groot/l10 | 500 | B0BigLibConsFast | vision | 70272 | 14.864/50.163/71.353 | 19.840/55.013/77.410 | 4.136/5.867/8.231 | — | — | 0.319/0.468/3.177 |
| CONCURRENT | groot/l10 | 500 | StuckRecovery | vision | 32430 | 4.115/5.037/6.298 | 8.768/11.029/14.567 | 4.107/5.622/7.774 | — | — | 0.362/0.536/4.109 |
| CONCURRENT | groot/spatial | 50 | AWM CL2 pure_cache | vision | 12296 | 2.733/3.401/4.108 | 5.918/7.272/9.287 | 2.779/3.508/4.433 | — | — | 0.351/0.543/1.225 |
| CONCURRENT | groot/spatial | 50 | AWM3 | vision | 24023 | 2.710/3.251/3.973 | 5.593/6.726/8.783 | 2.473/3.217/4.057 | — | — | 0.339/0.500/1.098 |
| CONCURRENT | groot/spatial | 50 | B0TopkConsensus | vision | 12626 | 3.895/4.816/6.177 | 7.148/8.787/11.921 | 2.823/3.586/4.544 | — | — | 0.375/0.581/1.431 |
| CONCURRENT | groot/spatial | 50 | StuckRecovery | vision | 12415 | 3.465/4.466/5.434 | 6.711/8.365/10.609 | 2.815/3.540/4.463 | — | — | 0.365/0.569/1.329 |
| CONCURRENT | groot/spatial | 50 | native | vision | 13820 | 2.812/3.551/4.374 | 2.812/3.551/4.374 | 2.812/3.551/4.374 | — | — | — |
| CONCURRENT | groot/spatial | 500 | AWM CL2 pure_cache | vision | 11047 | 2.704/3.244/3.926 | 5.479/6.581/8.287 | 2.372/3.114/3.912 | — | — | 0.333/0.485/0.994 |
| CONCURRENT | groot/spatial | 500 | B0BigLibConsFast | vision | 24551 | 9.668/12.560/20.502 | 12.607/15.956/25.837 | 2.490/3.280/4.204 | — | — | 0.312/0.462/1.085 |
| CONCURRENT | groot/spatial | 500 | StuckRecovery | vision | 10953 | 3.493/4.225/5.226 | 6.306/7.512/9.664 | 2.377/3.105/3.935 | — | — | 0.341/0.496/1.009 |
| CONCURRENT | pi05/l10 | 50 | AWM CL2 periodic | vision | 92625 | 6.058/11.473/17.567 | 37.890/50.764/62.488 | 25.656/37.518/49.128 | — | — | 3.390/8.879/15.434 |
| CONCURRENT | pi05/l10 | 50 | AWM CL2 pure_cache | vision | 36192 | 4.487/8.533/14.158 | 24.790/39.534/54.142 | 17.389/28.130/39.899 | — | — | 1.622/5.712/12.132 |
| CONCURRENT | pi05/l10 | 50 | AWM3 | vision | 35190 | 4.635/8.675/14.361 | 24.092/39.006/55.458 | 16.814/27.670/41.099 | — | — | 1.300/5.385/11.507 |
| CONCURRENT | pi05/l10 | 50 | B0Current | vision | 69298 | 11.355/26.588/46.843 | 54.471/89.214/114.744 | 28.088/61.249/91.297 | — | — | 5.282/22.182/41.710 |
| CONCURRENT | pi05/l10 | 50 | B0TopkConsensus | vision | 41376 | 13.406/29.542/52.918 | 50.011/87.062/128.278 | 25.577/52.570/86.650 | — | — | 4.595/19.889/39.369 |
| CONCURRENT | pi05/l10 | 50 | MixedJudge guard_only | vision | 67320 | 11.700/19.317/26.555 | 43.178/59.410/72.256 | 25.801/37.895/49.484 | — | — | 3.220/8.209/14.617 |
| CONCURRENT | pi05/l10 | 50 | MixedJudge quantile | vision | 96254 | 8.656/17.092/25.484 | 31.107/52.990/71.980 | 18.596/33.067/48.444 | — | — | 2.018/6.920/13.692 |
| CONCURRENT | pi05/l10 | 50 | SeededInference | vision | 59019 | 16.104/36.488/59.915 | 24.937/51.827/77.060 | — | — | — | 5.818/22.765/42.702 |
| CONCURRENT | pi05/l10 | 50 | StuckRecovery | vision | 35002 | 8.361/15.639/24.007 | 29.945/49.225/68.034 | 18.640/30.651/44.526 | — | — | 1.789/5.803/11.899 |
| CONCURRENT | pi05/l10 | 50 | native | vision | 40283 | 10.612/26.425/43.570 | 10.612/26.425/43.570 | 10.612/26.425/43.570 | — | — | — |
| CONCURRENT | pi05/l10 | 500 | AWM CL2 periodic | vision | 59359 | 11.835/19.604/27.535 | 50.061/64.886/80.370 | 31.823/43.501/56.335 | — | — | 4.174/9.800/16.705 |
| CONCURRENT | pi05/l10 | 500 | AWM CL2 pure_cache | vision | 31716 | 4.917/9.000/14.271 | 20.391/31.318/42.828 | 13.288/20.331/28.002 | — | — | 1.105/4.068/8.529 |
| CONCURRENT | pi05/l10 | 500 | B0BigLibConsFast | vision | 77074 | 24.367/43.816/71.152 | 41.732/68.814/101.594 | 14.818/25.579/41.525 | — | — | 0.637/4.526/11.328 |
| CONCURRENT | pi05/l10 | 500 | BlindMixedJudge guard_only | blind | 14726 | 0.911/1.594/4.231 | 0.000/0.000/0.000 | — | 0.690/1.262/3.909 | 0.406/1.132/3.997 | — |
| CONCURRENT | pi05/l10 | 500 | BlindMixedJudge guard_only | vision | 29072 | 14.279/26.122/37.055 | 17.832/31.081/42.717 | — | — | — | 2.874/7.202/13.291 |
| CONCURRENT | pi05/l10 | 500 | MixedJudge guard_only | vision | 59790 | 21.092/31.889/42.984 | 55.430/74.788/93.548 | 29.512/41.351/54.099 | — | — | 3.624/8.478/14.865 |
| CONCURRENT | pi05/l10 | 500 | MixedJudge quantile | vision | 29046 | 13.595/24.526/34.804 | 37.088/58.287/72.519 | 19.741/32.000/41.891 | — | — | 2.163/6.033/11.817 |
| CONCURRENT | pi05/l10 | 500 | StuckRecovery | vision | 31486 | 8.503/15.347/23.106 | 24.509/37.776/51.279 | 13.966/20.515/27.170 | — | — | 0.940/3.967/8.132 |
| CONCURRENT | pi05/spatial | 50 | AWM CL2 periodic | vision | 11172 | 4.121/8.217/13.737 | 24.009/40.292/50.180 | 16.349/28.833/39.701 | — | — | 2.142/6.592/11.654 |
| CONCURRENT | pi05/spatial | 50 | AWM CL2 pure_cache | vision | 12718 | 3.488/6.959/11.590 | 11.368/28.276/41.461 | 5.908/18.917/29.634 | — | — | 0.584/4.296/9.137 |
| CONCURRENT | pi05/spatial | 50 | AWM3 | vision | 24516 | 3.038/5.666/9.027 | 8.999/23.318/34.800 | 4.257/15.916/24.841 | — | — | 0.504/3.309/6.867 |
| CONCURRENT | pi05/spatial | 50 | B0Current | vision | 12736 | 3.622/7.573/14.206 | 27.001/45.917/56.727 | 18.607/34.781/47.053 | — | — | 2.057/8.873/16.658 |
| CONCURRENT | pi05/spatial | 50 | B0TopkConsensus | vision | 13320 | 4.399/8.220/14.235 | 18.648/34.026/48.102 | 11.137/23.113/35.946 | — | — | 0.557/5.779/13.051 |
| CONCURRENT | pi05/spatial | 50 | MixedJudge guard_only | vision | 11981 | 7.243/13.396/19.516 | 27.455/43.913/55.263 | 16.843/27.375/38.406 | — | — | 2.145/5.872/10.932 |
| CONCURRENT | pi05/spatial | 50 | MixedJudge quantile | vision | 33989 | 7.053/13.342/19.568 | 25.763/42.574/54.264 | 15.285/26.639/37.844 | — | — | 2.089/5.943/10.553 |
| CONCURRENT | pi05/spatial | 50 | StuckRecovery | vision | 12762 | 4.847/8.878/13.731 | 12.131/28.695/40.482 | 5.691/17.604/26.126 | — | — | 0.529/3.485/7.135 |
| CONCURRENT | pi05/spatial | 50 | native | vision | 14653 | 4.949/10.982/21.819 | 4.949/10.982/21.819 | 4.949/10.982/21.819 | — | — | — |
| CONCURRENT | pi05/spatial | 500 | AWM CL2 periodic | vision | 10769 | 5.743/10.503/15.823 | 25.407/39.213/47.678 | 16.459/25.495/33.392 | — | — | 2.458/5.723/9.877 |
| CONCURRENT | pi05/spatial | 500 | AWM CL2 pure_cache | vision | 10990 | 3.324/6.234/9.731 | 8.727/22.274/31.952 | 3.816/14.347/21.197 | — | — | 0.512/2.960/5.818 |
| CONCURRENT | pi05/spatial | 500 | B0BigLibConsFast | vision | 25139 | 11.610/22.446/33.466 | 21.436/39.712/59.045 | 8.654/16.156/26.633 | — | — | 0.424/2.784/7.556 |
| CONCURRENT | pi05/spatial | 500 | MixedJudge guard_only | vision | 10773 | 9.821/17.483/23.942 | 30.577/47.814/57.110 | 17.535/27.385/34.040 | — | — | 2.289/5.479/9.477 |
| CONCURRENT | pi05/spatial | 500 | MixedJudge quantile | vision | 10709 | 9.745/18.741/27.253 | 28.348/48.161/60.874 | 15.486/26.700/37.813 | — | — | 1.818/5.207/9.460 |
| CONCURRENT | pi05/spatial | 500 | StuckRecovery | vision | 10976 | 5.319/9.736/14.859 | 10.726/26.221/36.340 | 4.170/14.808/20.926 | — | — | 0.506/2.794/5.457 |
| SERIALIZED | pi05/l10 | 50 | MixedJudge guard_only | vision | 571 | 3.069/3.446/4.341 | 8.365/9.465/18.376 | 4.841/5.541/6.342 | — | — | 0.355/0.543/6.068 |
| SERIALIZED | pi05/l10 | 500 | BlindAWM | blind | 50 | 0.877/1.021/1.223 | 0.000/0.000/0.000 | — | 0.673/0.890/1.054 | 0.233/0.286/0.433 | — |
| SERIALIZED | pi05/l10 | 500 | BlindAWM | vision | 55 | 3.082/3.405/5.407 | 3.520/4.250/8.622 | — | — | — | 0.392/0.601/5.472 |
| SERIALIZED | pi05/l10 | 500 | BlindMixedJudge guard_only | blind | 12414 | 1.014/1.440/1.856 | 0.000/0.000/0.000 | — | 0.695/0.967/1.284 | 0.223/0.385/0.589 | — |
| SERIALIZED | pi05/l10 | 500 | BlindMixedJudge guard_only | vision | 47205 | 4.080/5.280/6.696 | 4.485/6.003/8.776 | — | — | — | 0.382/0.591/3.587 |
| SERIALIZED | pi05/l10 | 500 | MixedJudge guard_only | vision | 29883 | 3.749/4.290/5.487 | 7.996/9.889/13.150 | 3.793/5.214/7.102 | — | — | 0.331/0.482/3.283 |

SERIALIZED means an R4-mode startup before K6’s atomic installation at **2026-09-27 15:47:26.508504 CDT (20:47:26.508504 UTC)**. Legacy arms are CONCURRENT; R4 startups after the cutoff are also CONCURRENT. Classification is per startup record, not per decision timestamp, so a pre-install process remains serialized after the cutoff. Already-running processes retain old imported code. This inference follows the documented installation and flags; logs do not carry the imported plugin SHA. Config roots without server decisions at the snapshot remain listed in `log_inventory.json`/the root counts below.

**Coordinator first-cut verification (named arms, not all-arm pools).**

| Root/arm | Mode | Decision | N | q p50/p90/p99 | Prepare | Output |
| --- | --- | --- | --- | --- | --- | --- |
| r02_g50/oscl50_g_l10_cl2 | CONCURRENT | vision | 38173 | 2.752/3.479/4.376 | — | — |
| r02_g50/oscl50_p_l10_cl2 | CONCURRENT | vision | 36192 | 4.487/8.533/14.158 | — | — |
| r02_g500/oscl500_g_l10_cl2 | CONCURRENT | vision | 33390 | 2.957/3.564/4.432 | — | — |
| r02_g500/oscl500_p_l10_cl2 | CONCURRENT | vision | 31716 | 4.917/9.000/14.271 | — | — |
| r03_mx/r3mx_p_l10_g | CONCURRENT | vision | 32967 | 10.233/17.715/25.286 | — | — |
| r03_mx/r3mx_p_l10_g500 | CONCURRENT | vision | 29340 | 19.393/28.400/36.572 | — | — |
| r04_blind/r4b3_p_l10_500_b0g | SERIALIZED | vision | 29615 | 4.108/5.314/6.792 | — | — |
| r04_blind/r4b3_p_l10_500_ph2g | SERIALIZED | vision | 17535 | 4.025/5.223/6.504 | — | — |
| r04_blind/r4b3_p_l10_500_ph2g | SERIALIZED | blind | 12367 | 1.015/1.441/1.856 | 0.696/0.967/1.284 | 0.223/0.385/0.589 |
| r04_bsmoke/r4b3_p_l10_500_ph2g | SERIALIZED | vision | 55 | 3.869/4.600/8.305 | — | — |
| r04_bsmoke/r4b3_p_l10_500_ph2g | SERIALIZED | blind | 47 | 0.851/1.188/1.379 | 0.664/0.789/1.091 | 0.220/0.279/0.394 |
| r04_cost/r4b2_p_l10_g500_k2 | SERIALIZED | vision | 29883 | 3.749/4.290/5.487 | — | — |
| r04_k5_smoke/r4k5_p_l10_g50_r1 | SERIALIZED | vision | 276 | 3.087/3.557/4.193 | — | — |
| r04_k5_smoke/r4k5_p_l10_g50_r2 | SERIALIZED | vision | 295 | 3.063/3.402/4.605 | — | — |

The quoted AWM 4.5/4.9 ms and GR00T 2.8/3.0 ms, and R3 concurrent guard-only 10.2/19.4 ms, reproduce in their specific arms. Serialized R3 guard-only is about 3.1 ms at 50 (K5 smoke) and 3.75 ms at 500 (K2-cost arm). R4 gap-aware BlindMixedJudge vision is a different method: its serialized 500 medians are about 4.0–4.1 ms. The approximately 0.95 ms blind estimate is method-only `q_us`, not complete blind latency; preparation and output add their own distributions. Do not add separate percentiles as though they were a percentile of total time. `search_us=0` on blind decisions is a logging convention, while `q_us` times the blind method.

**Policy cost frame.** Owner graph basis: s1/s2/s3 = 10.26/27.69/29.57 ms; use the owner’s rounded full denominator 67.5 ms and shares .152/.410/.438. Eager π0.5 basis: 65.78312/33.42163/364.18918 ms, total 463.39393 ms (K=10).

| Method | Mean q | Owner s1 | Owner s2 | Owner s3 | Owner full | Eager s1 | Eager s2 | Eager s3 | Eager full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50_AWM | 1.177 | 11.47% | 4.25% | 3.98% | 1.74% | 1.79% | 3.52% | 0.32% | 0.25% |
| pi05_l10_500_AWM | 1.464 | 14.27% | 5.29% | 4.95% | 2.17% | 2.23% | 4.38% | 0.40% | 0.32% |
| pi05_l10_50_MixedJudge | 1.706 | 16.63% | 6.16% | 5.77% | 2.53% | 2.59% | 5.10% | 0.47% | 0.37% |
| pi05_l10_500_MixedJudge | 2.203 | 21.48% | 7.96% | 7.45% | 3.26% | 3.35% | 6.59% | 0.61% | 0.48% |
| pi05_l10_500_BlindMixedJudge | 2.160 | 21.05% | 7.80% | 7.30% | 3.20% | 3.28% | 6.46% | 0.59% | 0.47% |
| pi05_spatial_500_MixedJudge | 2.180 | 21.25% | 7.87% | 7.37% | 3.23% | 3.31% | 6.52% | 0.60% | 0.47% |
| Blind phase-particle step | 0.469 | 4.57% | 1.69% | 1.59% | 0.70% | 0.71% | 1.40% | 0.13% | 0.10% |

**GR00T eager cost fractions (flat cost-table basis, averaged over the two prompt/suite shapes).**

| Configuration | Mean q | Eager s1 | Eager s2 | Eager s3 | Eager full |
| --- | --- | --- | --- | --- | --- |
| groot_spatial_50_AWM | 1.149 | 3.53% | 3.95% | 0.61% | 0.46% |
| groot_spatial_50_BlindAWM | 1.207 | 3.71% | 4.15% | 0.64% | 0.48% |
| groot_spatial_500_AWM | 1.266 | 3.89% | 4.35% | 0.67% | 0.51% |
| groot_spatial_500_BlindAWM | 1.306 | 4.01% | 4.49% | 0.69% | 0.52% |
| groot_l10_50_AWM | 1.194 | 3.67% | 4.10% | 0.63% | 0.48% |
| groot_l10_50_BlindAWM | 1.242 | 3.82% | 4.27% | 0.66% | 0.50% |
| groot_l10_500_AWM | 1.391 | 4.28% | 4.78% | 0.74% | 0.56% |
| groot_l10_500_BlindAWM | 1.447 | 4.45% | 4.97% | 0.77% | 0.58% |

GR00T eager s1/s2/s3: 32.53043/29.09824/188.50252 ms, full 250.13119 ms. No GR00T owner CUDA-graph denominator was supplied, so none is invented.


| Frontier point (given SR) | SR | Vision share | MISS share | Added method ms/decision | Owner IR before→after | Eager IR before→after |
| --- | --- | --- | --- | --- | --- | --- |
| l10 500 guard-only | 0.864 | 1.0 | 0.10100 | 2.203 | 0.238→0.271 | 0.229→0.233 |
| l10 500 blind B=2 | 0.85 | 0.586 | 0.14400 | 1.460 | 0.211→0.233 | 0.207→0.210 |
| spatial 500 guard-only | 0.974 | 1.0 | 0.05307 | 2.180 | 0.197→0.229 | 0.187→0.192 |

**Server-wall-time surcharge scenarios for the supplied frontier points.** These use logged means and keep the supplied SR/shares fixed; they are workload-dependent accounting illustrations, not intrinsic GPU inference-cost estimates.

| Timing proxy | Mean method ms/decision | Owner IR + method | Mean search/blind boundary ms | Owner IR + boundary |
| --- | --- | --- | --- | --- |
| l10 guard: serialized | 3.836 | 0.295 | 8.393 | 0.362 |
| l10 guard: concurrent | 19.677 | 0.530 | 49.637 | 0.973 |
| l10 blind B2: serialized | 2.903 | 0.254 | 3.607 | 0.264 |
| spatial guard: concurrent | 10.380 | 0.351 | 29.408 | 0.633 |

Vision boundary uses logged search_us (including native shadow when enabled and in-search bookkeeping); blind boundary uses q_us + preparation + output. These are means, so addition is valid for expected cost. They exclude final log emission and failed blind attempts. R4 cost K2 is a timing proxy for the same guard method, not a claim that its rollout/SR equals the supplied K10 point. Concurrent wall time includes waiting for shared CPU/GIL resources and overlap with other connections, so adding it to isolated owner stages is deliberately a stressed-serving scenario, not a hardware-normalized measure. The controlled table above is the single-thread algorithm-cost comparison.

Formula: `ΔIR = [v·E(q_vision)+(1−v)·E(q_blind)]/full_ms`. MISS decisions already pay vision retrieval before the judge and therefore add no second search charge. This table holds the supplied SR, vision/MISS shares and K=10 constant; it does not predict a new closed-loop SR. Spatial MISS share is inferred from the supplied rounded owner IR: `(.197−.152)/.848`, not independently observed for that frontier point. Eager baselines are recomputed from eager stage costs and those shares; owner IR is never reused as an eager baseline. Means reflect the fixed offline input sample, not that frontier’s exact closed-loop state distribution. Native shadow, plugin overhead, failed blind attempts before vision, and GPU key pooling/D2H are excluded from the method-only addition. An operational ledger adds their measured **means** at the appropriate decision frequency; native shadow can be removed through the existing `--os-no-shadow-native` flag without changing selected actions.

**Snapshot and measurement load.**
Read 235 server log files containing 1,672,208 decisions, with 0 malformed/truncated lines. 4 files grew during the sequential scan; this is a read-only per-file snapshot, not a transactionally simultaneous sample. Exact byte counts and startup rows are retained in log_inventory.json. Roots with zero files had no server decision logs at this snapshot.

| Closed-loop root | Files | Decisions |
| --- | --- | --- |
| r02_g50 | 64 | 411987 |
| r02_g500 | 64 | 370024 |
| r03_full | 24 | 121381 |
| r03_mx | 51 | 397961 |
| r04_b4 | 0 | 0 |
| r04_blind | 8 | 103315 |
| r04_bsmoke | 2 | 207 |
| r04_cost | 6 | 88902 |
| r04_csl | 0 | 0 |
| r04_frontier | 14 | 177860 |
| r04_gblind | 0 | 0 |
| r04_k5 | 0 | 0 |
| r04_k5_smoke | 2 | 571 |
| r04_k7 | 0 | 0 |
| r04_k7_tailemit | 0 | 0 |

Snapshot time: `2026-09-27T22:09:42.484813+00:00`. Main-query 1-minute system load ranged 9.64–19.60; other workloads remained active. CPU 34 is the sole main-query timing core. Logical 34–37 and 78–81 are the only CPUs used; 78–81 are SMT siblings of 34–37 on this host. Every raw run contains its start/end load and per-core `/proc/stat` idle fraction, including the timing process’s own work. The initial parity-check helper briefly overlapped the first main sweep and, before its import guard was corrected, inherited CPU 34; this is an additional possible contributor to first-pass tails (notably GR00T l10/500 BlindAWM). Those original samples are retained. That configuration received a third fresh-process run after the rest of the experiments; its controlled table uses runs 2/3. The complete second pass had no such helper overlap.

| Assigned CPU | Idle percentage range during main replay runs |
| --- | --- |
| cpu34 | 0.00–0.27 |
| cpu35 | 0.00–94.31 |
| cpu36 | 88.16–94.72 |
| cpu37 | 87.52–94.05 |
| cpu78 | 91.00–100.00 |
| cpu79 | 41.54–99.86 |
| cpu80 | 87.74–94.00 |
| cpu81 | 87.74–94.15 |

Across configurations, relative between-process p50 variation (max/min−1) ranges 0.00%–62.54%; all repeats are retained. CPU cache placement, SMT/socket traffic and the machine’s background load are not controlled beyond affinity and thread limits.

**Measured headroom and hypotheses (no optimization implemented).**

| Configuration | PCA mean ms | Share of profile total | Top-k mean ms | 16-chunk kernel mean ms | Maximum owner ΔIR if PCA were free |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_500_AWM | 0.773 | 49.0% | 0.046 | 0.071 | 0.0115 |
| pi05_l10_500_MixedJudge | 0.829 | 35.8% | 0.083 | 0.072 | 0.0123 |
| pi05_l10_500_BlindMixedJudge | 0.730 | 30.8% | 0.084 | 0.071 | 0.0108 |

PCA reads two fixed 64×32768 float32 bases on every full-camera decision; its cost is substantially independent of library size. The candidate score, normalization/median, V7 candidate-state scan and top-k grow with per-task rows. Kernel synthesis is only 16 chunks; V7’s final scalar calibration and the small whitening matvec are cheap relative to the complete query. Wrist-only reduces the PCA input count to one camera, with new fitted metric/guard calibration; its measured latency difference is useful evidence of headroom but it changes the model’s vision/caching intervention and cannot be treated as a free equivalent optimization.

Hypotheses: keep PCA and candidate matvecs on an already-resident GPU or combine PCA/whitening affine maps to reduce CPU memory traffic; benchmark synchronization/transfer overhead before claiming a gain. NumPy vectorization alone cannot remove the already-vectorized PCA GEMV. Precomputed contiguous per-task V7 state blocks could avoid repeated gather/allocation, while preserving formulas; optimize top-k only if the measured scaling justifies it and retain deterministic tie behavior. Removing native shadow avoids its entire reference-search cost; deleting unused diagnostics/log serialization can reclaim only their measured component budgets. Shared-process oversubscription can dwarf kernel cost, so concurrency scheduling or process isolation deserves a separate capacity experiment. None of these changes was implemented or tested on GPU, and no SR-preserving speedup is claimed.

![Fixed-task scaling and thread contention](latency.png)

**Method and reproducibility.**
Working directory: `/home/weiland/projects/openpi`. The requested relative `rounds/r04/k8_search_latency` lives under `exp/offline_search/`. All writes are in this directory or `/tmp/k8_latency`; input stores, fit pickles and server logs are read-only. No GPU, server, port, chain, simulator, remote host, git command, or review-test path was used. No existing method/plugin/harness file was edited. Python bytecode writes are disabled. The absent `/dev/shm/offline_search_store` was replaced by the same on-disk store at `/home/weiland/trace_runs/offline_search_store`.

The primary benchmark loads the exact plugin fit payload and invokes its actual `query()` through `plugin.OnlineQueryView`, backed by plugin `_Buf` instances. It uses the first recorded cache and inference episode for each of ten tasks, up to 64 decisions per episode, preserving logged keys/state/executed-action/hit histories and resetting method state per episode. Every complete input sequence is warmed; ≥1,024 sequential calls are then timed with `perf_counter_ns`. Shorter spatial sets cycle to reach 1,024; larger l10 sets run all selected inputs. Model inference, fit/loading, disk faults during initial input loading, and reset outside the query timer are excluded. The primary vision timings use the recorded all-vision histories. R4/K7 vision calls immediately following blind gaps have different guard paths and are not separately isolated in the main vision table; actual blind/vision masks are exercised during blind-template collection, and the older R4 live mixture is represented by server logs. This is replay of real logged inputs, not a counterfactual rollout: cached and policy histories remain the recorded histories rather than being replaced by the benchmark’s synthesized actions.

Blind inputs are collected by sequentially exercising the real all-gates method on recorded cached episodes, calling query on LookReason and tracking the actual vision mask. The benchmark saves successful pre-call anchor/phase states and replays those exact `BlindQueryView` inputs ≥1,024 times after 128 warmups; restoring the saved small anchor state occurs outside the timer. Every checked replay action/row digest matches the collection call. This supplies enough valid blind decisions without weakening gates or inventing eligible states. The main table and raw results disclose the number of distinct successful templates, failed-LookReason counts and their latency; those failures do not get relabeled as successful blind steps.

Independent existing-log verification: **282/282 top-k/actions bit-identical**, including **55 blind decisions**, across R3 l10 both scales and R4 both π0.5 suites/scales. The exact NPZ paths and source-fit paths are in `replay_verification.json`; these are pre-existing K6 CPU plugin integration logs using real recorded store keys, not newly acquired GPU trajectories. Live closed-loop roots in scope did not save the full visual query inputs needed for equivalent replay. The existing logs verify representative deployed paths; wrist, CSL and supplemental K7 have profile/unprofile and process-repeat determinism checks, but no independent saved closed-loop top-k/action trace in this task. Native separately compares real backend winners to `rec_top1` from the original store.

Exact driver invocations (individual expanded commands are preserved in `benchmark_commands.json`, `extended_commands.json`, and `late_commands.json`):
```bash
K8=exp/offline_search/rounds/r04/k8_search_latency
PY=(taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
"${PY[@]}" "$K8/inventory.py"
"${PY[@]}" "$K8/select_configs.py"
"${PY[@]}" "$K8/run_benchmarks.py"
"${PY[@]}" "$K8/summarize_logs.py"
"${PY[@]}" "$K8/verify_replay.py"
"${PY[@]}" "$K8/run_extended.py"
"${PY[@]}" "$K8/benchmark.py" --config groot_l10_500_BlindAWM --run 3 --profile
"${PY[@]}" "$K8/run_late.py"
"${PY[@]}" "$K8/audit.py"
"${PY[@]}" "$K8/plot.py"
"${PY[@]}" "$K8/report.py"
```

**Caveats.**
Two fresh processes per primary/extended configuration do not make this a quiet-machine hardware benchmark. Wall-clock tails include preemption and shared memory bandwidth; all raw samples, affinity, and load/idle metadata are retained. Percentiles from pooled repeats are descriptive, not confidence intervals. Serial R4, legacy concurrency and post-K6 concurrency are different deployment conditions; do not infer the tenfold library effect from differently loaded arms. The synthetic block sweep and fixed offline replay do not reproduce live state occupancy or policy/GPU concurrency. The native CPU replay pins Torch to one thread; the deployed launcher permits OMP=4, which can reduce memory-heavy native wall time. Consequently a single-thread native replay need not equal serialized-server native_us. Native shadow’s 50-library footprint and disabled-native zeros must remain explicit. Blind preparation/output cover policy-specific transforms in live logs; their GPU pieces are not reproducible under the CPU-only constraint. The primary cost correction therefore reports method-only accounting and leaves the additional measured boundaries visible rather than silently conflating them.
