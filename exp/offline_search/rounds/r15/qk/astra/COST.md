# Light-key cost and IR projections

Measured locally, batch one and both cameras. **CPU is feasible and preferred.** Primary final measurements use the CPU-trained action-aware student. Prior GPU measurements are retained as historical comparisons; no further GPU benchmark is needed or authorized by convenience.

Sources: `latency_cpu_{50,200}.json`, `latency_cuda_{50,200}.json` (historical), `latency_cpu_extra_{50,200}.json`, `cost_counts.json/.csv`, and cited existing policy benchmarks. Each variant has 20 warmups per task and 100 timed iterations for each of tasks 0/2/5/8 (400 samples). Input is two actual selected-library 224² uint8 camera images; timing includes nearest resize, conversion, encoder, task metric, and the usable code on CPU. Historical GPU timing additionally includes host/device and code-return transfers. Common neighbor search/synthesis is excluded from both key costs. These are shared-host measurements, not an exclusive-GPU study.

## Per-look latency

| Cell size | Candidate | Backend / CPU threads | Median ms | IQR ms | p95 ms |
|---|---|---|---:|---|---:|
| 50 | symmetric_rgb_spatial_awm | cpu / 1 | 0.227 | 0.216–0.239 | 0.274 |
| 50 | asymmetric_ridge_0.1 | cpu / 1 | 0.218 | 0.211–0.233 | 0.262 |
| 50 | local_residual_krr_l1_a0.01 | cpu / 1 | 0.363 | 0.346–0.380 | 0.417 |
| 50 | cnn_distill | cpu / 1 | 1.114 | 1.069–1.194 | 1.322 |
| 50 | symmetric_rgb_spatial_awm | cuda / 1 | 0.559 | 0.537–0.591 | 0.796 |
| 50 | asymmetric_ridge_0.1 | cuda / 1 | 0.533 | 0.506–0.574 | 0.682 |
| 50 | local_residual_krr_l1_a0.01 | cuda / 1 | 0.858 | 0.818–0.903 | 1.491 |
| 50 | cnn_distill | cuda / 1 | 1.135 | 1.091–1.193 | 1.786 |
| 50 | symmetric_cnn_action_cpu | CPU eager / 1 | 1.191 | 1.159–1.244 | 1.316 |
| 50 | symmetric_cnn_action_cpu | CPU eager / 4 | 1.098 | 1.072–1.125 | 1.164 |
| 50 | symmetric_cnn_action_cpu | CPU torchscript / 1 | 0.948 | 0.923–0.981 | 1.053 |
| 50 | symmetric_cnn_action_cpu | CPU torchscript / 4 | 0.893 | 0.877–0.915 | 0.955 |
| 50 | action_qk_a1_s1 | CPU eager / 1 | 0.265 | 0.257–0.281 | 0.319 |
| 50 | action_qk_a1_s1 | CPU eager / 4 | 0.246 | 0.244–0.257 | 0.280 |
| 200 | symmetric_rgb_spatial_awm | cpu / 1 | 0.256 | 0.229–0.278 | 0.315 |
| 200 | asymmetric_ridge_0.01 | cpu / 1 | 0.218 | 0.210–0.231 | 0.256 |
| 200 | local_residual_krr_l1_a0.01 | cpu / 1 | 0.519 | 0.476–0.638 | 0.759 |
| 200 | cnn_distill | cpu / 1 | 1.106 | 1.064–1.182 | 1.329 |
| 200 | symmetric_rgb_spatial_awm | cuda / 1 | 0.559 | 0.537–0.596 | 0.835 |
| 200 | asymmetric_ridge_0.01 | cuda / 1 | 0.520 | 0.501–0.558 | 0.823 |
| 200 | local_residual_krr_l1_a0.01 | cuda / 1 | 0.870 | 0.828–0.999 | 1.686 |
| 200 | cnn_distill | cuda / 1 | 1.122 | 1.065–1.237 | 1.705 |
| 200 | symmetric_cnn_action_cpu | CPU eager / 1 | 1.192 | 1.176–1.201 | 1.248 |
| 200 | symmetric_cnn_action_cpu | CPU eager / 4 | 1.033 | 1.026–1.047 | 1.101 |
| 200 | symmetric_cnn_action_cpu | CPU torchscript / 1 | 0.971 | 0.942–1.002 | 1.044 |
| 200 | symmetric_cnn_action_cpu | CPU torchscript / 4 | 0.833 | 0.826–0.848 | 0.885 |
| 200 | action_qk_a0.1_s1 | CPU eager / 1 | 0.234 | 0.227–0.243 | 0.259 |
| 200 | action_qk_a0.1_s1 | CPU eager / 4 | 0.222 | 0.221–0.228 | 0.234 |

CPU export parity is checked against eager outputs on the timed inputs (`max_abs_export_parity` in the JSON). This timing trace is not a complete deployment export; all-task/dynamic-input parity remains part of the sol handoff. The historical CUDA CNN measurements use the same architecture with distillation weights; no new GPU run was made for the final CPU-trained variant. GR00T has no trained light model: corresponding cost conversions below are architecture proxies, not GR00T accuracy or deployability evidence.

## Policy references and epsilon

| Existing reference | Tower/stage-1 ms | Full-policy ms | Provenance / qualification |
|---|---:|---:|---|
| π0.5 CUDA-graph model timing | 10.26 | 67.48 | `exp/ablation_study/latency_bench/analysis/analysis.md`; owner basis, historical 4090 |
| π0.5 production eager cost table | 65.783 | 463.394 | `closed_loop/ops/cost_table.json`; sum of recorded stages |
| GR00T production eager cost table | 32.530 | 250.131 | Same cost table; full K8 policy |
| π0.5 R12 measured stock tower | 68.484 | 483.215 | R12 price table; full column is tower + remainder medians, not an end-to-end median |
| GR00T L10 R12 tower | 35.898 | 261.987 | R12 price table; sum of tower/remainder medians |
| GR00T Spatial R12 tower | 32.426 | 235.843 | R12 price table; sum of tower/remainder medians |

The fixed owner split is retained: π0.5 .152/.848, GR00T .148/.852. For a stated tower timing `t_v`, epsilon = `c_v * light_ms / t_v`. Do not refit the owner split to unrelated wall-time stage shares. The primary conservative π0.5 comparison uses the historical 10.26-ms tower; the same-production-stack R12 conversion is shown separately. CPU latency expressed in these units is a cost proxy, not a measurement of GPU utilization saved.

| Size | Primary CPU TorchScript ms, one thread | ε π0.5 graph basis | ε π0.5 R12 eager basis | ε GR00T L10 R12 proxy | ε GR00T Spatial R12 proxy |
|---|---:|---:|---:|---:|---:|
| 50 | 0.9478 | 0.01404 | 0.00210 | 0.00391 | 0.00433 |
| 200 | 0.9714 | 0.01439 | 0.00216 | 0.00400 | 0.00443 |

The graph-basis median budget is 1.026 ms (epsilon≤.0152); an eager-only pass must not conceal a miss against that conservative sensitivity. Deployment-host H100 timing is not measured here. No policy was loaded for a new latency benchmark: existing authoritative measurements suffice and a full GPU model would exceed this task’s 4-GiB prototype limit.

## Correct counterfactual ledger

`IR_old = (c_v V + c_m M)/N` and `IR_new = (epsilon V + M)/N`, because a light-key MISS still executes the complete policy including the tower. Thus `saved = [c_v(V−M) − epsilon V]/N`. The benefit comes from cache-only looks. At all-policy looks a student is overhead; a separately implemented forced-policy bypass can avoid that overhead, but cannot save the policy tower.

All following projections hold observed N/V/M fixed. Actual accuracy changes may alter M, V, episode length and success. `cost_counts.py` reconstructed accepted attempts and contiguous decisions and reconciled all 52 B arms with summary ledgers. No new-method outcome was read. Full arm/task counts are in `cost_counts.csv`; hashes of every source read are in `cost_counts.json`.

## Uniform operating points at the target epsilon = c_v/10

| Round | Cell | Knob | N | V | M | Current IR | Projected IR | Relative saving |
|---|---|---|---:|---:|---:|---:|---:|---:|
| R13 | pi05_l10_50 | ir25 | 5888 | 2966 | 1120 | 0.2379 | 0.1979 | 16.8% |
| R13 | pi05_l10_50 | ir32 | 5676 | 2859 | 1556 | 0.3090 | 0.2818 | 8.8% |
| R13 | pi05_l10_50 | ir32 | 5641 | 2842 | 1561 | 0.3112 | 0.2844 | 8.6% |
| R13 | pi05_l10_50 | ir40 | 5637 | 2842 | 2083 | 0.3900 | 0.3772 | 3.3% |
| R13 | pi05_l10_50 | ir45 | 5536 | 2785 | 2410 | 0.4456 | 0.4430 | 0.6% |
| R13 | pi05_spatial_50 | ir25 | 2274 | 1161 | 454 | 0.2469 | 0.2074 | 16.0% |
| R13 | pi05_spatial_50 | ir32 | 2262 | 1156 | 652 | 0.3221 | 0.2960 | 8.1% |
| R13 | pi05_spatial_50 | ir32 | 2233 | 1145 | 644 | 0.3225 | 0.2962 | 8.2% |
| R13 | pi05_spatial_50 | ir40 | 2307 | 1177 | 892 | 0.4054 | 0.3944 | 2.7% |
| R13 | pi05_spatial_50 | ir45 | 2341 | 1196 | 1040 | 0.4544 | 0.4520 | 0.5% |
| R13 | groot_l10_50 | ir25 | 6606 | 3322 | 1418 | 0.2573 | 0.2221 | 13.7% |
| R13 | groot_l10_50 | ir32 | 6319 | 3180 | 1887 | 0.3289 | 0.3061 | 6.9% |
| R13 | groot_l10_50 | ir32 | 6121 | 3079 | 1815 | 0.3271 | 0.3040 | 7.1% |
| R13 | groot_l10_50 | ir40 | 6266 | 3150 | 2396 | 0.4002 | 0.3898 | 2.6% |
| R13 | groot_l10_50 | ir45 | 5847 | 2941 | 2562 | 0.4478 | 0.4456 | 0.5% |
| R13 | groot_spatial_50 | ir25 | 2498 | 1270 | 521 | 0.2529 | 0.2161 | 14.6% |
| R13 | groot_spatial_50 | ir32 | 2487 | 1269 | 735 | 0.3273 | 0.3031 | 7.4% |
| R13 | groot_spatial_50 | ir32 | 2485 | 1262 | 712 | 0.3193 | 0.2940 | 7.9% |
| R13 | groot_spatial_50 | ir40 | 2408 | 1227 | 921 | 0.4013 | 0.3900 | 2.8% |
| R13 | groot_spatial_50 | ir45 | 2498 | 1274 | 1110 | 0.4541 | 0.4519 | 0.5% |
| R14 | pi05_l10_50 | ir45 | 16570 | 8363 | 7235 | 0.4470 | 0.4443 | 0.6% |
| R14 | pi05_l10_50 | ir40 | 16806 | 8465 | 6212 | 0.3900 | 0.3773 | 3.3% |
| R14 | pi05_l10_50 | ir32 | 16589 | 8365 | 4517 | 0.3075 | 0.2800 | 9.0% |
| R14 | pi05_l10_50 | ir25 | 17547 | 8829 | 3298 | 0.2359 | 0.1956 | 17.1% |
| R14 | pi05_l10_50 | off | 18557 | 9335 | 2022 | 0.1689 | 0.1166 | 30.9% |
| R14 | pi05_spatial_50 | ir45 | 6558 | 3352 | 2850 | 0.4462 | 0.4424 | 0.9% |
| R14 | pi05_spatial_50 | ir40 | 6623 | 3380 | 2496 | 0.3972 | 0.3846 | 3.2% |
| R14 | pi05_spatial_50 | ir32 | 6606 | 3375 | 1843 | 0.3142 | 0.2868 | 8.7% |
| R14 | pi05_spatial_50 | ir25 | 6653 | 3396 | 1282 | 0.2410 | 0.2005 | 16.8% |
| R14 | pi05_spatial_50 | off | 6987 | 3565 | 467 | 0.1342 | 0.0746 | 44.4% |
| R14 | groot_l10_50 | ir45 | 17492 | 8807 | 7717 | 0.4504 | 0.4486 | 0.4% |
| R14 | groot_l10_50 | ir40 | 17763 | 8937 | 6768 | 0.3991 | 0.3885 | 2.7% |
| R14 | groot_l10_50 | ir32 | 17876 | 9006 | 5211 | 0.3229 | 0.2990 | 7.4% |
| R14 | groot_l10_50 | ir25 | 19187 | 9653 | 4114 | 0.2571 | 0.2219 | 13.7% |
| R14 | groot_l10_50 | off | 19604 | 9858 | 3124 | 0.2102 | 0.1668 | 20.6% |
| R14 | groot_spatial_50 | ir45 | 6992 | 3566 | 3038 | 0.4457 | 0.4420 | 0.8% |
| R14 | groot_spatial_50 | ir40 | 7184 | 3652 | 2736 | 0.3997 | 0.3884 | 2.8% |
| R14 | groot_spatial_50 | ir32 | 7054 | 3597 | 2015 | 0.3188 | 0.2932 | 8.0% |
| R14 | groot_spatial_50 | ir25 | 7255 | 3688 | 1491 | 0.2503 | 0.2130 | 14.9% |
| R14 | groot_spatial_50 | off | 7237 | 3679 | 467 | 0.1302 | 0.0721 | 44.7% |

## R14 frozen selected task tables

These are B-selected whole-task stitches of existing uniform arms, not independently executed mixed-policy results. They describe economic headroom at the existing table, not a new descent. The target-epsilon and zero-cost ceilings are shown together.

| Cell | Current IR | IR at ε=c_v/10 | Saving | Maximum saving at ε=0 | Extra calls/100 looks before all savings disappear |
|---|---:|---:|---:|---:|---:|
| pi05_l10_50 | 0.2264 | 0.1843 | 18.6% | 22.0% | 8.35 |
| pi05_spatial_50 | 0.2629 | 0.2264 | 13.9% | 16.8% | 7.16 |
| groot_l10_50 | 0.4433 | 0.4402 | 0.7% | 2.4% | 0.61 |
| groot_spatial_50 | 0.2750 | 0.2417 | 12.1% | 14.8% | 6.52 |

| Cell | Task | Fixed level | N | V | M | Current IR | Projected IR at cost gate |
|---|---:|---|---:|---:|---:|---:|---:|
| pi05_l10_50 | 0 | ir25 | 1837 | 925 | 326 | 0.2270 | 0.1851 |
| pi05_l10_50 | 1 | off | 1505 | 763 | 68 | 0.1154 | 0.0529 |
| pi05_l10_50 | 2 | ir45 | 1543 | 779 | 668 | 0.4439 | 0.4406 |
| pi05_l10_50 | 3 | off | 1541 | 778 | 161 | 0.1653 | 0.1122 |
| pi05_l10_50 | 4 | ir45 | 1492 | 754 | 649 | 0.4457 | 0.4427 |
| pi05_l10_50 | 5 | ir25 | 1258 | 634 | 254 | 0.2478 | 0.2096 |
| pi05_l10_50 | 6 | off | 1901 | 955 | 291 | 0.2062 | 0.1607 |
| pi05_l10_50 | 7 | off | 1817 | 913 | 137 | 0.1403 | 0.0830 |
| pi05_l10_50 | 8 | off | 2518 | 1263 | 253 | 0.1614 | 0.1081 |
| pi05_l10_50 | 9 | off | 1737 | 877 | 223 | 0.1856 | 0.1361 |
| pi05_spatial_50 | 0 | ir40 | 613 | 314 | 242 | 0.4126 | 0.4026 |
| pi05_spatial_50 | 1 | ir45 | 663 | 336 | 279 | 0.4339 | 0.4285 |
| pi05_spatial_50 | 2 | off | 620 | 316 | 54 | 0.1513 | 0.0948 |
| pi05_spatial_50 | 3 | off | 505 | 261 | 2 | 0.0819 | 0.0118 |
| pi05_spatial_50 | 4 | off | 807 | 410 | 62 | 0.1424 | 0.0846 |
| pi05_spatial_50 | 5 | ir45 | 615 | 315 | 262 | 0.4391 | 0.4338 |
| pi05_spatial_50 | 6 | off | 646 | 330 | 23 | 0.1078 | 0.0434 |
| pi05_spatial_50 | 7 | off | 718 | 365 | 31 | 0.1139 | 0.0509 |
| pi05_spatial_50 | 8 | c10 | 694 | 354 | 354 | 0.5101 | 0.5178 |
| pi05_spatial_50 | 9 | ir25 | 746 | 378 | 140 | 0.2362 | 0.1954 |
| groot_l10_50 | 0 | ir45 | 1792 | 902 | 793 | 0.4515 | 0.4500 |
| groot_l10_50 | 1 | c10 | 1533 | 775 | 775 | 0.5055 | 0.5130 |
| groot_l10_50 | 2 | c10 | 1586 | 799 | 799 | 0.5038 | 0.5112 |
| groot_l10_50 | 3 | c10 | 1444 | 728 | 728 | 0.5042 | 0.5116 |
| groot_l10_50 | 4 | ir32 | 1629 | 823 | 473 | 0.3222 | 0.2978 |
| groot_l10_50 | 5 | off | 1249 | 632 | 173 | 0.1929 | 0.1460 |
| groot_l10_50 | 6 | ir40 | 1843 | 928 | 738 | 0.4157 | 0.4079 |
| groot_l10_50 | 7 | c10 | 1780 | 896 | 896 | 0.5034 | 0.5108 |
| groot_l10_50 | 8 | c10 | 2560 | 1284 | 1284 | 0.5016 | 0.5090 |
| groot_l10_50 | 9 | ir45 | 1700 | 856 | 748 | 0.4494 | 0.4475 |
| groot_spatial_50 | 0 | ir45 | 702 | 358 | 319 | 0.4626 | 0.4620 |
| groot_spatial_50 | 1 | off | 735 | 374 | 71 | 0.1576 | 0.1041 |
| groot_spatial_50 | 2 | off | 616 | 317 | 22 | 0.1066 | 0.0433 |
| groot_spatial_50 | 3 | ir45 | 564 | 290 | 249 | 0.4522 | 0.4491 |
| groot_spatial_50 | 4 | c10 | 791 | 402 | 402 | 0.5082 | 0.5157 |
| groot_spatial_50 | 5 | c10 | 608 | 311 | 311 | 0.5115 | 0.5191 |
| groot_spatial_50 | 6 | off | 682 | 347 | 20 | 0.1003 | 0.0369 |
| groot_spatial_50 | 7 | ir25 | 754 | 383 | 149 | 0.2435 | 0.2051 |
| groot_spatial_50 | 8 | off | 724 | 370 | 53 | 0.1380 | 0.0808 |
| groot_spatial_50 | 9 | off | 815 | 413 | 41 | 0.1179 | 0.0578 |

## Projections using measured primary CPU epsilon

| R14 π0.5 cell | Off: current → projected | ir25: current → projected | Selected table: current → projected |
|---|---|---|---|
| pi05_l10_50 | 0.1689 → 0.1160 (31.3%) | 0.2359 → 0.1950 (17.3%) | 0.2264 → 0.1838 (18.8%) |
| pi05_spatial_50 | 0.1342 → 0.0740 (44.9%) | 0.2410 → 0.1999 (17.1%) | 0.2629 → 0.2258 (14.1%) |

Spatial’s last table is an architecture-cost projection only: its trained model and all-look accuracy gate are not established. The R13/R14 ir40–ir45 regimes have little removable cache-only tower work. GR00T L10’s chosen table cannot deliver a large total reduction at matched calls even with a free key. The economical initial target is π0.5 off/ir25, subject to the still-failing accuracy gate.
