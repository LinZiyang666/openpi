# Round-2 gate application

The numerical gates in GATES.md are unchanged (SHA256 188d3cc1d3beb807174c3bff84696542be72f00bcb66ac0cdc66cbc913a2d937). Their old offline-veto permission language is superseded by owner directive 4. Pilot eligibility requires the image-use and own-library rules; it is separate from an offline accuracy/cost pass.

| Cell | Selected recipe / epochs | Library gate | B-dev gate | Image audit | CPU median / p95 ms | Local cost gate |
|---|---|---|---|---|---|---|
| pi05_l10_50 | state_drop_aug / 40 | True | False | True | 1.028 / 1.202 | False |
| pi05_l10_200 | state_drop_aug / 120 | True | False | True | 0.958 / 1.127 | True |

| Cell | One-thread CPU IQR ms | Epsilon: graph basis | Epsilon: historical R12 eager sensitivity |
|---|---|---:|---:|
| pi05_l10_50 | 0.963–1.124 | 0.015229 | 0.002282 |
| pi05_l10_200 | 0.932–0.998 | 0.014188 | 0.002126 |

The R12 eager sensitivity uses its previously measured 68.484-ms π0.5 tower and the same .152 owner coefficient (sources in COST.md). It is not H100 deployment timing and cannot convert a conservative graph-budget failure into a pass.

Local timing uses one CPU thread, 400 timed raw-image-to-main-AWM-code calls across all ten tasks after warmup; four-thread timing is also saved. Raw two-camera bundles are prepared outside the timer; resize, float conversion, state/task tensor conversion, model inference and AWM projection are timed. Generic loader checks, camera-bundle construction and common neighbor synthesis are not timed. These are encoder-path measurements, not complete serving latency. H100 deployment-host timing and the complete five-layer runtime verification belong to sol/coordinator and are not asserted here. Failure of an old offline numerical gate is reported explicitly but no longer vetoes this owner-authorized pilot.

## Cost projection at the frozen size-50 epsilon

| R14 B-dev operating point | Current IR | Projected IR | Relative saving |
|---|---:|---:|---:|
| r14d_pi05_l10_50_periodic_ir25 | 0.2359 | 0.1956 | 17.1% |
| r14d_pi05_l10_50_off | 0.1689 | 0.1166 | 30.9% |
| fixed R14 task table (stitched) | 0.2264 | 0.1844 | 18.6% |

These use the already reconciled B-only counts in cost_counts.json and hold N/V/M fixed: IR_new=(epsilon*V+M)/N. They do not predict the extra policy calls or changed trajectory lengths caused by the new key. No size-200 count projection is borrowed from size 50; the matched pilot must measure its own cost. The task table is only a fixed-table projection, not an added pilot arm.

Exact checks: `gate_report.json`. Pilot design and artifacts: `PILOT.md`.
