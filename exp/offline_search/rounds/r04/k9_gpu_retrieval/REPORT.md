**K9 parity: real stored queries, batch 1, two fresh processes per cell and precision.** Counts below are **per process**; repeated inputs are not counted as new independent queries. R1/R2 agreement counts are shown explicitly. Strict diagnostic gates: action atol 1e-4; other real-valued fields atol 1e-3, except `motion`, `vself` and `vis` atol 1e-5. Discrete fields require equality. The top-1 target is ≥99.9%. These are empirical replay checks, not universal error bounds.

**float32:**

| Suite / episodes / method | N | Top-1 agree R1/R2 | Top-16 set R1/R2 | Top-16 order R1/R2 | Max \|Δchunk\| | Max \|Δconfidence\| | Chunk >1e-4 R1/R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 1037 | 1037/1037 | 1037/1037 | 1037/1037 | 0.000642 | 0.00159 | 16/16 |
| spatial / 50 / MixedJudge | 1037 | 1037/1037 | 1037/1037 | 1037/1037 | 0.000642 | 4.44e-07 | 16/16 |
| spatial / 500 / AWM | 1037 | 1036/1036 | 1037/1037 | 1034/1034 | 0.000425 | 0.000808 | 10/10 |
| spatial / 500 / MixedJudge | 1037 | 1036/1036 | 1037/1037 | 1034/1034 | 0.000425 | 1.58e-06 | 10/10 |
| l10 / 50 / AWM | 2200 | 2200/2200 | 2200/2200 | 2194/2194 | 0.00171 | 0.0997 | 20/20 |
| l10 / 50 / MixedJudge | 2200 | 2200/2200 | 2200/2200 | 2194/2194 | 0.00171 | 0.00635 | 20/20 |
| l10 / 500 / AWM | 2200 | 2200/2200 | 2196/2196 | 2190/2190 | 0.0125 | 0.0269 | 19/19 |
| l10 / 500 / MixedJudge | 2200 | 2200/2200 | 2196/2196 | 2190/2190 | 0.0125 | 1.28e-06 | 19/19 |

**float64:**

| Suite / episodes / method | N | Top-1 agree R1/R2 | Top-16 set R1/R2 | Top-16 order R1/R2 | Max \|Δchunk\| | Max \|Δconfidence\| | Chunk >1e-4 R1/R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 1037 | 1037/1037 | 1037/1037 | 1035/1035 | 0.000313 | 0.000785 | 12/12 |
| spatial / 50 / MixedJudge | 1037 | 1037/1037 | 1037/1037 | 1035/1035 | 0.000313 | 6.76e-07 | 12/12 |
| spatial / 500 / AWM | 1037 | 1037/1037 | 1037/1037 | 1032/1032 | 0.00043 | 0.00199 | 8/8 |
| spatial / 500 / MixedJudge | 1037 | 1037/1037 | 1037/1037 | 1032/1032 | 0.00043 | 1.29e-06 | 8/8 |
| l10 / 50 / AWM | 2200 | 2200/2200 | 2200/2200 | 2196/2196 | 0.000783 | 0.095 | 17/17 |
| l10 / 50 / MixedJudge | 2200 | 2200/2200 | 2200/2200 | 2196/2196 | 0.000783 | 0.00635 | 17/17 |
| l10 / 500 / AWM | 2200 | 2200/2200 | 2196/2196 | 2194/2194 | 0.0124 | 0.0267 | 4/4 |
| l10 / 500 / MixedJudge | 2200 | 2200/2200 | 2196/2196 | 2194/2194 | 0.0124 | 1.5e-06 | 4/4 |

**Float32 maximum absolute differences by output, across both final processes.** AWM confidence is also independently audited for each MixedJudge base. The complete per-field distributions, worst UID/step and every strict mismatch are in the raw JSON. Integer ID differences are reported as mismatches, not excused by a numeric tolerance.


| Cell | d1 | disp5 | dst | w_eff | scores | stuck_n | vself | disp | vis | zsum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 0.00066 | 9.97e-05 | 0.0004 | 0.0109 | 0.0015 | — | — | — | — | — |
| spatial / 50 / MixedJudge | 0.00066 | 9.97e-05 | 0.0004 | 0.0109 | 150 | 0 | 3.37e-07 | 2.38e-07 | 6.56e-07 | 3.64e-06 |
| spatial / 500 / AWM | 0.00241 | 5.12e-05 | 0.000632 | 0.0181 | 0.00328 | — | — | — | — | — |
| spatial / 500 / MixedJudge | 0.00241 | 5.12e-05 | 0.000632 | 0.0181 | 0.0556 | 0 | 4.28e-07 | 2.38e-07 | 6.56e-07 | 6.99e-06 |
| l10 / 50 / AWM | 0.00634 | 0.00678 | 0.000187 | 0.0157 | 0.0108 | — | — | — | — | — |
| l10 / 50 / MixedJudge | 0.00634 | 0.00678 | 0.000187 | 0.0157 | 162 | 0 | 4.28e-07 | 0.0099 | 5.96e-07 | 0.796 |
| l10 / 500 / AWM | 0.00685 | 0.00179 | 0.000401 | 0.0947 | 0.0138 | — | — | — | — | — |
| l10 / 500 / MixedJudge | 0.00685 | 0.00179 | 0.000401 | 0.0947 | 0.0823 | 0 | 3.96e-07 | 1.19e-07 | 5.36e-07 | 1.64e-05 |

**Remaining audited float32 scalar/weight differences (maximum across both processes).** Dashes mean the deployed result does not expose that field for this method; additional implemented intermediate values are not claimed as independently audited extras.


| Cell | awm_confidence | weights | d1_rel | c0 | lag | pred_err | still | vote | lib_ep | lib_step | term1 | top1_prog |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | — | 0.00214 | 3.85e-06 | 2.67e-06 | — | — | 5.69e-07 | — | 0 | 0 | — | — |
| spatial / 50 / MixedJudge | 0.00159 | 0.00214 | 3.85e-06 | — | 3.55e-15 | 4.44e-07 | — | 8.39e-06 | — | — | 0 | 0 |
| spatial / 500 / AWM | — | 0.000515 | 9.75e-06 | 2.16e-06 | — | — | 6.98e-07 | — | 40 | 0 | — | — |
| spatial / 500 / MixedJudge | 0.000808 | 0.000515 | 9.75e-06 | — | 3.55e-15 | 1.58e-06 | — | 2.27e-06 | — | — | 0 | 0 |
| l10 / 50 / AWM | — | 0.00823 | 1.05e-05 | 4.95e-06 | — | — | 6.21e-07 | — | 0 | 0 | — | — |
| l10 / 50 / MixedJudge | 0.0997 | 0.00823 | 1.05e-05 | — | 0.2 | 0.00635 | — | 1.02e-05 | — | — | 0 | 0 |
| l10 / 500 / AWM | — | 0.00187 | 1.26e-05 | 5.42e-06 | — | — | 6.1e-07 | — | 0 | 0 | — | — |
| l10 / 500 / MixedJudge | 0.0269 | 0.00187 | 1.26e-05 | — | 1.42e-14 | 1.28e-06 | — | 4.15e-06 | — | — | 0 | 0 |

**Float32 action error by regime (maximum over both processes).** Regimes 0/1/2 mean step 0 / fresh after MISS / stale after HIT or unknown, respectively.


| Cell | N per process 0/1/2 | Δchunk step 0 | Δchunk fresh | Δchunk stale |
| --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 40/428/569 | 0.000642 | 4.35e-06 | 8.37e-06 |
| spatial / 50 / MixedJudge | 40/428/569 | 0.000642 | 4.35e-06 | 8.37e-06 |
| spatial / 500 / AWM | 40/428/569 | 0.000425 | 7.32e-06 | 4.54e-06 |
| spatial / 500 / MixedJudge | 40/428/569 | 0.000425 | 7.32e-06 | 4.54e-06 |
| l10 / 50 / AWM | 40/1040/1120 | 0.00171 | 7.03e-06 | 1.48e-05 |
| l10 / 50 / MixedJudge | 40/1040/1120 | 0.00171 | 7.03e-06 | 1.48e-05 |
| l10 / 500 / AWM | 40/1040/1120 | 0.0125 | 2.39e-05 | 4.41e-06 |
| l10 / 500 / MixedJudge | 40/1040/1120 | 0.0125 | 2.39e-05 | 4.41e-06 |

Every tested float32 non-step-0 chunk satisfies atol 1e-4 (maximum 2.39089131e-05); the chunk failures are at step 0. This does not imply that all other outputs satisfy their own strict gates.

**Retrieval alone, float32, milliseconds per call under concurrent load.** Each eager and captured path has 10 warmups + 80 timed calls/process. Resident inputs; transfers, fit/loading and capture setup excluded. Every entry is event p50 R1→R2 / synchronized wall p50 R1→R2. Batch 8 is total batch latency, not per-query latency.


| Cell | B1 eager event / wall | B1 graph event / wall | B1 graph min event / wall | B8 eager event / wall | B8 graph event / wall | B8 graph min event / wall |
| --- | --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 4.540 → 4.654 / 4.572 → 4.678; min 3.929/3.970 | 0.337 → 0.290 / 0.352 → 0.306; min 0.284/0.302 | 0.284 / 0.302 | 4.676 → 4.924 / 4.700 → 4.946; min 4.264/4.286 | 0.458 → 0.348 / 0.474 → 0.364; min 0.318/0.333 | 0.318 / 0.333 |
| spatial / 50 / MixedJudge | 10.031 → 9.305 / 10.056 → 9.330; min 8.426/8.451 | 0.808 → 0.924 / 0.833 → 0.945; min 0.556/0.575 | 0.556 / 0.575 | 10.188 → 9.724 / 10.213 → 9.749; min 8.926/8.950 | 0.889 → 0.601 / 0.914 → 0.630; min 0.595/0.609 | 0.595 / 0.609 |
| spatial / 500 / AWM | 4.649 → 4.711 / 4.695 → 4.739; min 4.230/4.260 | 0.308 → 0.436 / 0.333 → 0.460; min 0.295/0.309 | 0.295 / 0.309 | 5.002 → 4.924 / 5.042 → 4.949; min 4.515/4.537 | 0.447 → 0.372 / 0.466 → 0.390; min 0.335/0.350 | 0.335 / 0.35 |
| spatial / 500 / MixedJudge | 9.383 → 10.641 / 9.407 → 10.665; min 8.595/8.620 | 0.598 → 0.679 / 0.615 → 0.704; min 0.558/0.574 | 0.558 / 0.574 | 9.474 → 9.829 / 9.498 → 9.857; min 9.041/9.063 | 0.688 → 0.708 / 0.706 → 0.735; min 0.631/0.646 | 0.631 / 0.646 |
| l10 / 50 / AWM | 6.000 → 5.097 / 6.029 → 5.132; min 4.011/4.033 | 0.348 → 0.329 / 0.389 → 0.350; min 0.302/0.318 | 0.302 / 0.318 | 4.931 → 5.512 / 4.955 → 5.543; min 4.213/4.255 | 0.346 → 0.853 / 0.364 → 0.889; min 0.326/0.342 | 0.326 / 0.342 |
| l10 / 50 / MixedJudge | 8.877 → 9.255 / 8.914 → 9.286; min 8.058/8.079 | 0.632 → 0.647 / 0.666 → 0.671; min 0.557/0.576 | 0.557 / 0.576 | 9.740 → 9.706 / 9.766 → 9.735; min 8.611/8.661 | 0.725 → 1.146 / 0.757 → 1.168; min 0.611/0.628 | 0.611 / 0.628 |
| l10 / 500 / AWM | 4.558 → 5.249 / 4.593 → 5.276; min 4.088/4.109 | 0.678 → 0.401 / 0.699 → 0.419; min 0.341/0.358 | 0.341 / 0.358 | 5.010 → 4.943 / 5.035 → 4.972; min 4.485/4.510 | 0.517 → 0.509 / 0.538 → 0.530; min 0.444/0.463 | 0.444 / 0.463 |
| l10 / 500 / MixedJudge | 9.198 → 8.766 / 9.222 → 8.790; min 7.882/7.907 | 0.665 → 0.764 / 0.684 → 0.786; min 0.631/0.646 | 0.631 / 0.646 | 9.447 → 9.624 / 9.470 → 9.671; min 8.260/8.295 | 0.859 → 0.929 / 0.878 → 0.954; min 0.793/0.808 | 0.793 / 0.808 |

**Float64 retrieval graph tradeoff, milliseconds per call.**


| Cell | B1 graph event p50 R1→R2 | B1 graph wall p50 R1→R2 | B1 minimum event / wall | Resident MiB f32 / f64 |
| --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 0.599 → 0.561 | 0.616 → 0.577 | 0.401 / 0.416 | 22.152 / 41.870 |
| spatial / 50 / MixedJudge | 1.154 → 1.125 | 1.180 → 1.146 | 0.665 / 0.684 | 25.852 / 46.512 |
| spatial / 500 / AWM | 0.445 → 0.523 | 0.463 → 0.545 | 0.425 / 0.439 | 51.084 / 81.078 |
| spatial / 500 / MixedJudge | 0.764 → 1.159 | 0.784 → 1.187 | 0.756 / 0.774 | 63.938 / 102.062 |
| l10 / 50 / AWM | 0.534 → 0.578 | 0.568 → 0.595 | 0.431 / 0.452 | 28.441 / 50.393 |
| l10 / 50 / MixedJudge | 0.700 → 0.798 | 0.724 → 0.818 | 0.676 / 0.692 | 34.132 / 58.588 |
| l10 / 500 / AWM | 0.760 → 0.623 | 0.777 → 0.640 | 0.529 / 0.547 | 135.366 / 195.294 |
| l10 / 500 / MixedJudge | 0.918 → 2.624 | 0.947 → 2.644 | 0.903 / 0.925 | 174.882 / 263.877 |

**Real π0.5 stage 1 → deployed 4×4 pooling → retrieval/synthesis, milliseconds, batch 1, step-0 observation.** Every entry is event p50 R1→R2 / wall p50 R1→R2; 5 warmups + 40 timed calls/process. CPU path ends with the host chunk and verdict data; GPU path includes pinned-host copies of only the final chunk and verdict packet. CUDA-event spans around the CPU path include host gaps: they are not a pure kernel sum. Input transforms and common observation H2D are excluded from both paths.


| Cell | Stock eager S1 + CPU | All eager GPU | Graph S1 + CPU | Combined graph + final D2H | Combined min event / wall |
| --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 70.931 → 70.385 / 70.962 → 70.417; min 63.378/63.411 | 68.765 → 71.502 / 68.790 → 71.539; min 63.981/64.019 | 13.024 → 13.963 / 13.062 → 14.023; min 11.941/11.980 | 10.601 → 17.422 / 10.631 → 17.460; min 10.533/10.560 | 10.533 / 10.560 |
| spatial / 50 / MixedJudge | 68.199 → 70.373 / 68.236 → 70.406; min 62.794/62.824 | 76.556 → 78.548 / 76.590 → 78.587; min 68.566/68.590 | 12.130 → 16.887 / 12.155 → 16.917; min 11.750/11.776 | 10.854 → 17.270 / 10.890 → 17.292; min 10.784/10.833 | 10.784 / 10.833 |
| spatial / 500 / AWM | 66.821 → 68.965 / 66.853 → 69.003; min 61.641/61.671 | 71.058 → 71.998 / 71.087 → 72.027; min 64.579/64.626 | 11.954 → 17.757 / 11.990 → 17.798; min 11.723/11.758 | 10.623 → 16.369 / 10.648 → 16.420; min 10.570/10.595 | 10.570 / 10.595 |
| spatial / 500 / MixedJudge | 76.929 → 72.932 / 76.967 → 72.966; min 67.162/67.210 | 78.216 → 78.213 / 78.245 → 78.248; min 72.467/72.509 | 13.394 → 21.517 / 13.438 → 21.548; min 12.392/12.419 | 10.885 → 16.459 / 10.928 → 16.494; min 10.849/10.880 | 10.849 / 10.880 |
| l10 / 50 / AWM | 71.113 → 71.676 / 71.148 → 71.710; min 64.510/64.557 | 70.999 → 75.827 / 71.032 → 75.859; min 67.289/67.318 | 12.989 → 16.258 / 13.023 → 16.288; min 11.912/11.942 | 10.641 → 18.012 / 10.671 → 18.046; min 10.616/10.646 | 10.616 / 10.646 |
| l10 / 50 / MixedJudge | 67.438 → 74.164 / 67.471 → 74.197; min 63.828/63.856 | 77.910 → 75.772 / 77.941 → 75.800; min 67.961/67.998 | 12.816 → 15.463 / 12.844 → 15.493; min 11.910/11.937 | 10.875 → 13.231 / 10.910 → 13.262; min 10.831/10.850 | 10.831 / 10.850 |
| l10 / 500 / AWM | 71.569 → 130.103 / 71.602 → 130.136; min 67.141/67.173 | 72.100 → 209.379 / 72.130 → 209.430; min 69.305/69.337 | 17.001 → 87.002 / 17.037 → 87.058; min 13.432/13.479 | 17.139 → 108.707 / 17.210 → 108.814; min 11.041/11.108 | 11.041 / 11.108 |
| l10 / 500 / MixedJudge | 71.660 → 193.765 / 71.794 → 193.810; min 64.296/64.340 | 79.241 → 184.700 / 79.267 → 184.775; min 74.206/74.235 | 21.582 → 72.038 / 21.615 → 72.075; min 13.386/13.423 | 14.226 → 118.365 / 14.258 → 118.488; min 11.430/11.476 | 11.430 / 11.476 |

**Stage-1 graph and increment.** Combined paired samples exclude final D2H in both arms and alternate execution order. Positive/negative samples are retained: load fluctuations can exceed the retrieval increment. These are manual CUDA graphs of the current kernels, not the owner’s historical compiled 10.26-ms stage-1 cost basis.


| Cell | S1 eager event / wall | S1 graph event / wall | Separate graphs + D2H event / wall | Paired Δ event p50 R1→R2 | Paired Δ wall p50 R1→R2 | Paired Δ event mean R1→R2 | Paired Δ event min / p90 (pooled) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 67.198 → 66.419 / 67.227 → 66.460; min 61.170/61.215 | 10.237 → 14.006 / 10.278 → 14.032; min 10.179/10.205 | 10.563 → 15.382 / 10.587 → 15.422; min 10.495/10.515 | 0.358 → 0.497 | 0.358 → 0.489 | 0.356 → 4.561 | -81.758 / 2.882 |
| spatial / 50 / MixedJudge | 65.847 → 68.714 / 65.877 → 68.750; min 61.185/61.211 | 10.255 → 16.293 / 10.306 → 16.325; min 10.202/10.227 | 10.887 → 17.709 / 10.922 → 17.738; min 10.811/10.838 | 0.582 → 0.635 | 0.586 → 0.629 | 0.583 → 5.184 | -12.491 / 4.582 |
| spatial / 500 / AWM | 64.742 → 67.201 / 64.779 → 67.228; min 57.826/57.888 | 10.280 → 14.134 / 10.317 → 14.176; min 10.207/10.230 | 10.646 → 15.745 / 10.675 → 15.802; min 10.594/10.623 | 0.358 → 0.427 | 0.361 → 0.419 | 0.357 → 1.359 | -73.741 / 8.545 |
| spatial / 500 / MixedJudge | 67.971 → 68.210 / 68.014 → 68.263; min 61.683/61.712 | 10.307 → 15.052 / 10.348 → 15.077; min 10.233/10.260 | 10.898 → 18.248 / 10.929 → 18.292; min 10.846/10.871 | 0.633 → 0.783 | 0.632 → 0.793 | 0.633 → 8.659 | -30.390 / 8.982 |
| l10 / 50 / AWM | 68.271 → 68.105 / 68.301 → 68.133; min 61.668/61.695 | 10.847 → 11.845 / 10.878 → 11.922; min 10.190/10.217 | 10.609 → 17.083 / 10.642 → 17.122; min 10.560/10.591 | 0.401 → 0.655 | 0.396 → 0.653 | 0.503 → -0.972 | -65.737 / 6.671 |
| l10 / 50 / MixedJudge | 64.708 → 65.027 / 64.734 → 65.081; min 60.676/60.699 | 10.304 → 14.195 / 10.334 → 14.241; min 10.187/10.213 | 10.863 → 15.222 / 10.900 → 15.256; min 10.807/10.841 | 0.633 → 0.890 | 0.630 → 0.908 | 0.630 → -0.898 | -59.481 / 1.791 |
| l10 / 500 / AWM | 68.869 → 153.923 / 68.913 → 153.958; min 64.284/64.318 | 11.917 → 29.794 / 11.939 → 29.843; min 10.271/10.303 | 16.333 → 35.026 / 16.401 → 35.139; min 10.964/11.004 | -0.243 → 3.975 | -0.252 → 3.971 | 1.302 → 12.977 | -165.339 / 77.538 |
| l10 / 500 / MixedJudge | 68.594 → 175.889 / 68.633 → 175.921; min 62.856/62.891 | 14.518 → 132.441 / 14.536 → 132.560; min 10.505/10.544 | 22.188 → 47.134 / 22.242 → 47.223; min 11.513/11.554 | 1.362 → 0.894 | 1.361 → 0.877 | -0.638 → -0.231 | -239.475 / 77.605 |

**PCIe bytes per decision at the measured retrieval boundary.** Common observation upload is listed separately. No raw tokens or pooled camera keys cross D2H in the GPU design; no native shadow/logging is included.


| Cell | Actual pooled dtype | Current keys D2H | Host f32 key bytes | GPU chunk + verdict D2H | Common observation H2D | Fixed candidate capacity/task |
| --- | --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | torch.bfloat16 | 131328 | 262272 | 1360 | 1808395 | 192 |
| spatial / 50 / MixedJudge | torch.bfloat16 | 131328 | 262272 | 1456 | 1808395 | 192 |
| spatial / 500 / AWM | torch.bfloat16 | 131328 | 262272 | 1360 | 1808395 | 1664 |
| spatial / 500 / MixedJudge | torch.bfloat16 | 131328 | 262272 | 1456 | 1808395 | 1664 |
| l10 / 50 / AWM | torch.bfloat16 | 131328 | 262272 | 1360 | 1808395 | 512 |
| l10 / 50 / MixedJudge | torch.bfloat16 | 131328 | 262272 | 1456 | 1808395 | 512 |
| l10 / 500 / AWM | torch.bfloat16 | 131328 | 262272 | 1360 | 1808395 | 5952 |
| l10 / 500 / MixedJudge | torch.bfloat16 | 131328 | 262272 | 1456 | 1808395 | 5952 |

The current D2H total is 131,072 bytes of bf16 cameras plus 256 bytes of observed float64 stage state; the builder materializes float32 keys/state on CPU. A float32-on-device pooled-key design with float32 state would transfer 262,272 bytes. New per-decision control inputs are three int64 values (task/step/previous-HIT: 24 H2D bytes) if updated from host; histories/actions should remain resident. This benchmark preloads those controls/history and does not time their upload. The current interceptor can additionally upload the synthesized 1,280-byte CPU chunk beyond the measured boundary; that and eventual wire/output transforms are not charged here.

**Observed load and memory.** Utilization is device-wide, including K9 during measurement; it cannot isolate another process’s utilization. Timings are under load, and all quoted minima are observed loaded-machine minima.


| Cell | Before K9 CUDA util % | During timing util % | Min free GiB during timing | Max observed own GiB | Peak Torch reserved GiB |
| --- | --- | --- | --- | --- | --- |
| spatial / 50 / AWM | 0.0–74.0 | 0.0–100.0 | 15.303 | 2.498 | 2.000 |
| spatial / 50 / MixedJudge | 0.0–82.0 | 0.0–100.0 | 14.650 | 2.504 | 2.000 |
| spatial / 500 / AWM | 0.0–79.0 | 0.0–100.0 | 13.955 | 2.539 | 2.041 |
| spatial / 500 / MixedJudge | 0.0–41.0 | 0.0–100.0 | 14.754 | 2.547 | 2.043 |
| l10 / 50 / AWM | 0.0–96.0 | 0.0–100.0 | 13.445 | 2.498 | 2.000 |
| l10 / 50 / MixedJudge | 0.0–100.0 | 0.0–100.0 | 12.201 | 2.523 | 2.020 |
| l10 / 500 / AWM | 42.0–100.0 | 23.0–100.0 | 15.082 | 2.623 | 2.125 |
| l10 / 500 / MixedJudge | 40.0–100.0 | 0.0–100.0 | 14.660 | 2.680 | 2.176 |


| Cell | Read-only fitted artifact |
| --- | --- |
| spatial / 50 / AWM | /home/weiland/trace_runs/os_closed_loop/r02_g50/fits/oscl50_p_sp_cl2.pkl |
| spatial / 50 / MixedJudge | /home/weiland/trace_runs/os_closed_loop/r03_mx/fits/r3mx_p_sp_g.pkl |
| spatial / 500 / AWM | /home/weiland/trace_runs/os_closed_loop/r02_g500/fits/oscl500_p_sp_cl2.pkl |
| spatial / 500 / MixedJudge | /home/weiland/trace_runs/os_closed_loop/r04_frontier/fits/r4_p_sp_g500.pkl |
| l10 / 50 / AWM | /home/weiland/trace_runs/os_closed_loop/r02_g50/fits/oscl50_p_l10_cl2.pkl |
| l10 / 50 / MixedJudge | /home/weiland/trace_runs/os_closed_loop/r03_mx/fits/r3mx_p_l10_g.pkl |
| l10 / 500 / AWM | /home/weiland/trace_runs/os_closed_loop/r02_g500/fits/oscl500_p_l10_cl2.pkl |
| l10 / 500 / MixedJudge | /home/weiland/trace_runs/os_closed_loop/r03_mx/fits/r3mx_p_l10_g500.pkl |

**Design note.**

The prototype is an isolated `torch.nn.Module`; nothing imports it from serving. It loads the existing plugin fit, registers PCA/whitening/calibration and library data as device buffers, and chooses a task through tensor indexing. The common capacity is `ceil64(max(1.25 * largest_task, largest_task + 64))`; unused rows are excluded by a validity mask. Median uses the two central valid ranks, including the even-count case. Both early and main distances and continuity are evaluated at fixed shapes and selected with tensor conditions. The 16-member kernel uses kref=5 for the current fits and kref=8 for the big fits; float64 weights are normalized and cast to float32 before mixing every element of the 10×32 chunk. Stable sort chooses ascending row order for exact computed ties. Pure CPU AWM's `argpartition` can select a different subset at an exact rank-16 boundary tie; MixedJudge explicitly resolves that boundary by row.

`append_prepared` accepts every precomputed row field, copies into spare slots, updates the count, and publishes the validity mask last. The caller must serialize maintenance against graph execution and provide globally ascending IDs and rows computed with the existing frozen fit. It does not refit PCA, task means, scales or confidence calibration. Capacity exhaustion requires a new allocation and graph. Each final module process appends a duplicate of its selected row with a new ID and checks that all registered addresses remain unchanged, the existing graph sees the row, and graph/eager outputs agree. This is an address/lifecycle check, not an evaluation of an enlarged real library.

Guard-only MixedJudge can keep one device slot per connection/episode: prior real vision keys and state, previous executed action/tail and HIT flag, stuck count, top-1 progress and prior progress step, consecutive no-progress count, burst/return counters and a small HIT-history ring. Motion, raw task-centred camera cosine, terminal/gripper predicates, lag/overtime, dispersion, V7 interpolation and Boolean guard combination are tensor operations. This prototype computes retrieval, AWM confidence/extras, V7 confidence/features and the stuck-count recurrence; it takes history tensors as inputs. It does **not** implement the final MixedJudge flag/burst/progress state machine or publish a HIT/MISS verdict. A production version must preserve reset, repeated-query, skipped-decision and connection-reuse semantics, and update executed-action history only after the actual served/policy outcome is known. The raw-key task means for stuck detection differ from the PCA task means used by V7 candidate similarity; both are retained.

R4 blind continuation can store its 16 anchor rows, weights, phase rows, anchor state/chunk, anchor/last steps, age, episode generation and last two normalized motions in device tensors. Consecutive-next gathers, bounded offset scans, phase argmin, gripper/terminal weighted masses, displacement residual and anchor-tail slicing are fixed-shape operations. K1's progress-span guard uses real vision anchors; do not advance it with fabricated blind keys. K7 additionally needs the last real vision anchor and dense motion intervals since it: endpoint cosine confirms only the intervening low-motion intervals, carrying a prior stuck count only across a fully qualifying gap. It cannot establish that no out-and-back visual motion occurred. R4/K7 blind execution is a design proposal here, not implemented or benchmarked.

The deployed quantile controller is **server-wide**, not per episode. A GPU ring of W float64 effective confidences, count, head and n_seen can reproduce its sorted order statistic at `clamp(round((1-h)*n), 0, n-1)`, its warm-start threshold, and ±infinity entries for forced decisions. Python's rounding and serialized tau/read/decide/push order must be reproduced. A batch cannot silently give all connections the same pre-batch threshold: that changes today's atomic controller semantics. Fixed-size sorting and a deterministic serial order within a batch are possible; no controller is ported here.

An initial serving integration should leave the current adapter behind an explicit opt-in flag, hold fixed input/output buffers per graph slot, and retain the stage-1 prefix on GPU. Copy the chunk plus compact verdict inputs to pinned host buffers, wait once, and let the existing host scheduler choose HIT versus stage 2/3. That branch is the remaining required host synchronization in this proposed host-controlled design, not an intrinsic GPU limitation. Once the entire guard/controller is on device, a scalar verdict is sufficient for the host branch. A CUDA conditional graph can instead run the MISS subgraph from a device-set condition: CUDA 12.4 introduced IF/WHILE, while optional ELSE and SWITCH arrived in 12.8. This needs a native graph integration and eligible stage-2/3 body, not just a Python `if` inside `torch.cuda.graph`; it was not tested here. [NVIDIA conditional-node description](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/), [body restrictions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements).

Integration remains gated on an agreed numerical policy for near ties and strict action/confidence differences, then shadow replay against the unchanged CPU method, episode/state isolation checks, and measured scheduling/load behavior. No online success-rate preservation is inferred from these offline comparisons.

**Numerical findings and caveats.**

float32: lowest cell top-1 agreement 99.903568%; largest full-chunk absolute difference 0.0124661475. The strict 1e-4 chunk gate fails. This prototype is not a demonstrated strict numerical replacement for the CPU path. The supplied float64 variant is a diagnostic option, not a promise of reproducing CPU float32 rounding.
float64: lowest cell top-1 agreement 100.000000%; largest full-chunk absolute difference 0.0124055743. The strict 1e-4 chunk gate fails. This prototype is not a demonstrated strict numerical replacement for the CPU path. The supplied float64 variant is a diagnostic option, not a promise of reproducing CPU float32 rounding.

The mismatch classes are: near ties changing top-1 identity and its episode metadata; order changes, including rank-5 boundary swaps within an unchanged top-16 set; exact/near rank-16 boundary ties changing membership; continuous distance/kernel-weight/synthesis drift with identical selected rows; confidence/extras drift propagated through dispersion and calibration; and large absolute transformed-score drift for strongly downweighted members. All eight selected fits are loaded without refitting. `mismatch_catalog.json` groups every observed discrete ranking mismatch and every strict per-field count; each raw benchmark JSON gives every offending UID/step, both 16-row selections, deltas, CPU top-1 and rank-16 gaps, and CPU distances at the GPU-selected rows. The action maximum is over all 320 chunk values, not only the executed 35.

The CPU query uses float32 PCA GEMV, whitening and expanded squared distances, then float64 score/kernel calculations and float32 synthesis. GPU reductions/FMA and GEMM/GEMV execution order differ even with TF32 disabled. Early distances subtract large, nearly equal terms using fitted float32 norm arrays; resulting ties or small gaps can change order/membership. Float64 runtime math still begins with those quantized fitted buffers and need not agree more closely with deployed float32 decisions. Stable sorting only resolves exact ties in the computed GPU distances; it cannot recover a tie created by CPU rounding. Full-chunk synthesis amplifies a rank-16 swap when the displaced row retains non-negligible kernel weight. These mechanisms follow the code and the logged gap/delta evidence; no success-rate effect is measured.

A separate CPU-only diagnosis (`numerical_probe.py`, `numerical_probe.json`) replays 16 real step-0 cases. Across their first 18 neighbors, the sum of absolute expanded-distance terms divided by the final squared distance ranges 2530.7–16489.0. It records PCA f32/f64 differences, each norm/cross term, and exact CPU boundary ties. This diagnoses cancellation; the f64 recomputation is not ground truth for the deployed f32 method.

Coverage: π0.5 spatial/l10 × 50/500 × pure CL2 AWM/guard-only MixedJudge, both float32 and float64 retrieval; 40 real recorded episodes per suite (first two episodes per task in cache and inference arms, up to 64 decisions/episode), preserving executed histories. Spatial has 1,037 and l10 2,200 query calls per process. This bounded replay limits shared-GPU occupation; remaining store episodes and later decisions were not tested. Identical episode-start observations can appear in both arms; repeats and duplicates do not increase independent coverage. Batch-8 latency repeats one stored stale query and records its comparison to batch 1; the full selected-query parity sweep is batch 1. No synthetic rollout was substituted for the real query inputs.

Stage 1 uses the same checkpoint and mixed bf16/fp32 weights as K3, with unused stages 2/3 placed on meta. It loads an exact CPU stage-1-only scratch cache after its first normal checkpoint load; no serving weights/configuration change. Direct stock manual capture failed at `embed_prefix`’s Python-list `torch.tensor` construction (retained in `stage_dev1.log`). The isolated adapter replaces that all-zero attention-mask allocation with `zeros_like`; it runs all three image towers. The final raw stage files test all five stage-output fields on six changed real observations across tasks 0/4/9, and compare graph/eager retrieval/chunk/verdict packet outputs. Changing a test image at a later recorded step while holding the timing regime at step 0 tests graph/input reuse, not a complete chronological episode. The existing stage compile/graph options and K3 adapter were inspected; manual graph capture supplies the requested combined graph, so no additional torch.compile experiment was run and the historical compiled cost basis was not reproduced. No stages 2/3 inference, online guard/controller, R4 blind GPU implementation, K7 GPU implementation, GR00T, simulator, SR, RPC, logging, native-shadow retrieval or serving integration was performed. GR00T was optional; the other omitted items are outside this retrieval/feature experiment or explicitly design-only deliverables.

The GPU timing endpoint is the host chunk plus confidence/guard inputs. Final host guard/progress updates, controller thresholding and the HIT/MISS branch are excluded. The deployed CPU method call computes its own guards as part of the baseline. These timings therefore compare the requested stage/retrieval/synthesis boundary; they do not establish latency of a complete GPU serving replacement. All eager paths retain Python launch overhead. The separately captured path also packs its verdict outside the retrieval graph, whereas the combined graph includes that packing.

Final stage adapter checks: 96/96 all-field bitwise matches; maximum graph/eager chunk difference 0, packet difference 0. Same-eager-stage CPU/GPU chunk error is separately retained for every stage run. All 32 final append/address checks completed successfully.

Module between-process wall-median variation ranges 0.00%–179.28% (`variation.json`). Paired stage increments and all raw event/wall samples remain available; two fresh processes do not establish quiet-machine latency or multi-client throughput. The historical owner s1/s2/s3=10.26/27.69/29.57-ms and full=67.5-ms constants are reference context only; no IR or SR projection is claimed from this shared-load manual-graph experiment.

Every Python command is pinned to CPUs 26–29,70–73, with OMP/OpenBLAS/MKL threads=1 and bytecode writes disabled. `GPUWatch` invokes nvidia-smi before GPU phases, admits at ≥12 GiB free, polls free/own memory about every 0.2 s, exits its own process below 4 GiB or above 8 GiB own memory, and caps the Torch allocator at 7 GiB to leave context/library allowance. The serial scheduler waits for additional headroom, preserves rejected attempts, and each measurement process exits to release its GPU state. GPU timings include other active server workloads; MPS and other process PIDs are recorded in admission snapshots. Admission failures, including later capture/timing phase checks, are retained in logs; no OOM recovery was attempted. No servers, ports, remote machines, LIBERO workers, git commands, review tests, or running code paths were touched. Writes are confined to this directory and `/tmp/k9_*`. The store fallback is `/home/weiland/trace_runs/offline_search_store`; `/dev/shm/offline_search_store` was absent.

**Exact commands and artifacts.** Working directory `/home/weiland/projects/openpi`; requested `rounds/` is under `exp/offline_search/`. `final_commands.json` contains every expanded taskset/env/Python invocation, attempt, timing, exit code and log path. Successful final jobs alone populate these tables. Earlier dev/smoke artifacts are retained and excluded.

Environment: Python 3.11.15, NumPy 1.26.4, PyTorch 2.7.1+cu126 / CUDA runtime 12.6. nvidia-smi identifies the shared device as NVIDIA GeForce RTX 4090, 49,140 MiB total memory, driver 595.71.05; those are observed device-reported values. `environment.json`, per-attempt admission snapshots and `reference_hashes.txt` retain the environment and reference-source evidence. The final audit checks unchanged retrieval, stage-adapter and measurement-code hashes across all 48 successful processes, ≥99.9% top-1 per cell, graph/input/append checks and memory limits; `audit.json` separately reports the failed strict chunk gate. The CPU references and deployed stage/key-builder sources still match `reference_hashes.txt` at completion (`reference_hashes_final_check.txt`).

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/run_jobs.py
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/audit.py
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/report.py
```

