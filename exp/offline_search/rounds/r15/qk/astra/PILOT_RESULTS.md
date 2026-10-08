# R15-B appendix: replacing the vision tower with a light image key

The light key remains an ablation/appendix comparison, following [owner decision D5](../../COORDINATOR_DECISIONS.md). It uses images and removes tower work from cache hits, but loses closed-loop success and misses the frozen joint accuracy/savings criteria. The vision-tower key remains the main design. No fitting, model changes, new rollouts or GPU work were performed for this analysis.

Both students use current agentview/wrist images, normalized robot state and task identity; training uses only the exact served library. Both library and query use the student features, followed by the same per-task action-supervised AWM, early metric and 16-NN kernel. Round 2 selected the augmented state-dropout recipe using two within-library episode hold-outs (40 epochs at size 50; 120 at size 200).

## Offline progression

| Library | Student round | Library donor-LOEO error ratio | Unseen B-dev error excess | Shuffle / black-image error rise | CPU key median / p95 (ms) |
|---|---|---:|---:|---:|---:|
| 50 | 1 | 0.9950 | +12.50% | 156.0% / 114.2% | 0.948 / 1.053 |
| 50 | 2 | 1.0146 | +9.83% | 169.0% / 117.5% | 1.028 / 1.202 |
| 200 | 1 | 1.0168 | +10.00% | 203.5% / 128.3% | 0.971 / 1.044 |
| 200 | 2 | 1.0299 | +8.02% | 217.5% / 131.9% | 0.958 / 1.127 |

Error is equal-episode/equal-task RMS10 relative to the deployed tower key. B-dev contains 100 unseen episodes / 3,198 looks at size 50 and 73 / 2,289 at size 200, from existing size-50 trajectories. Both rounds pass the image-use audit (state held fixed); state-only is rejected. Image sensitivity does not imply adequate retrieval accuracy: both final students fail the unchanged B-dev gate. Local one-thread timing includes image conversion and AWM code, excludes common retrieval, and is not H100 serving latency. The frozen graph-basis median budget is 1.026 ms: round-2 size 50 narrowly fails; size 200 passes locally. [Full offline analysis](PROTOTYPE2.md).

## Paired closed-loop success

All eight arms contain 300 accepted, error-free episodes, 30/task, with identical task/B-init pairs from the R15 list outside the 200-episode subset. Six arms were newly collected; the two size-200 current arms reuse descent runs under the [baseline equivalence proof](../impl/baseline_equivalence.json). Accepted attempts, contiguous server decisions, client verdicts, episode-end counts and summaries were independently reconciled. This is reused B development data, not independent validation.

| Library | Level | Current successes | Light successes | Δ SR (pp) | Paired gains / losses | Conservative paired bounds (pp) |
|---|---|---:|---:|---:|---:|---:|
| 50 | off | 255/300 (85.00%) | 228/300 (76.00%) | -9.00 | 29 / 56 | [-17.0, -0.8] |
| 50 | ir25 | 258/300 (86.00%) | 244/300 (81.33%) | -4.67 | 26 / 40 | [-12.0, +2.7] |
| 200 | off | 269/300 (89.67%) | 259/300 (86.33%) | -3.33 | 20 / 30 | [-9.8, +3.3] |
| 200 | ir25 | 274/300 (91.33%) | 268/300 (89.33%) | -2.00 | 17 / 23 | [-7.9, +4.0] |

Δ is light minus current. The frozen lower bound is `L = CP_low(g, n, .025) − CP_high(l, n, .025)`; the displayed upper companion reverses the bounds. Each endpoint is a conservative one-sided 95% bound; the displayed pair is not claimed as a simultaneous two-sided 95% interval. The gate uses L. The machine-readable results also include a simultaneous two-sided 95% Bonferroni interval using .0125 tails. Pairing controls task/init identity; process-seeded stochastic rollout order is not an identical-noise guarantee.

Per-task Δ SR and the same conservative paired bounds (pp; 30 paired episodes/task):

| Task | 50 off | 50 ir25 | 200 off | 200 ir25 |
|---|---:|---:|---:|---:|
| 0 | -6.7 [-34.8, +23.0] | +3.3 [-25.1, +31.0] | +0.0 [-17.1, +17.1] | -3.3 [-17.2, +11.5] |
| 1 | -10.0 [-26.5, +9.5] | +0.0 [-11.6, +11.6] | +0.0 [-11.6, +11.6] | -3.3 [-17.2, +11.5] |
| 2 | -6.7 [-22.1, +10.8] | -13.3 [-37.7, +14.4] | +3.3 [-11.5, +17.2] | +0.0 [-17.1, +17.1] |
| 3 | -6.7 [-26.4, +15.1] | -3.3 [-22.0, +16.4] | -3.3 [-22.0, +16.4] | +0.0 [-17.1, +17.1] |
| 4 | -6.7 [-40.5, +28.6] | +10.0 [-20.8, +38.5] | +0.0 [-24.4, +24.4] | -6.7 [-29.9, +18.3] |
| 5 | +0.0 [-21.3, +21.3] | -3.3 [-22.0, +16.4] | +0.0 [-17.1, +17.1] | +0.0 [-17.1, +17.1] |
| 6 | -3.3 [-31.0, +25.1] | +0.0 [-24.4, +24.4] | -33.3 [-61.9, +2.4] | +13.3 [-11.6, +34.6] |
| 7 | -10.0 [-30.6, +13.5] | -10.0 [-26.5, +9.5] | -3.3 [-17.2, +11.5] | +3.3 [-11.5, +17.2] |
| 8 | -20.0 [-51.7, +15.9] | -16.7 [-45.6, +16.0] | +3.3 [-28.6, +34.6] | -23.3 [-48.6, +7.3] |
| 9 | -20.0 [-47.3, +11.8] | -13.3 [-40.2, +16.6] | +0.0 [-21.3, +21.3] | +0.0 [-27.0, +27.0] |

## Actual work and modeled cost

Compute from accepted decision logs, without repricing: `IR_current=(.152 V+.848 M)/N`; `IR_light=(ε Q+M)/N`. V is current-system tower calls, Q student calls, M full-policy calls, N five-control requests. The frozen ε values are 0.01522897 (size 50), 0.01418826 (size 200). IR uses actual work counts with frozen cost coefficients; it is not measured wall time.

| Library | Level | IR current → light | Relative IR saving | M/N current → light | Executed controls/episode current → light | Cost/episode current → light |
|---|---|---:|---:|---:|---:|---:|
| 50 | off | 0.16474 → 0.14088 | 14.49% | 10.40% → 13.32% | 298.73 → 328.70 | 9.893 → 9.305 |
| 50 | ir25 | 0.23476 → 0.20562 | 12.41% | 18.66% → 19.80% | 290.60 → 305.04 | 13.721 → 12.610 |
| 200 | off | 0.15030 → 0.10986 | 26.91% | 8.70% → 10.27% | 281.05 → 293.85 | 8.502 → 6.496 |
| 200 | ir25 | 0.24385 → 0.20959 | 14.05% | 19.72% → 20.24% | 276.27 → 276.91 | 13.560 → 11.685 |

Current-arm summary IR fields use the generic eager stage-cost ratio, not the owner coefficients; the table recomputes the required .152/.848 formula from audited V/M/N. Light summaries reproduce the frozen-epsilon formula exactly. Cost/episode is in full-policy-call equivalents. Executed controls come from accepted `client_timing.steps`, subtracting the configured warm-up; every episode satisfies `5(N−1) < controls ≤ 5N`, retaining its final partial chunk. Request counts remain the frozen IR denominator. Longer light-key episodes weaken savings per episode further; failure/termination is not credited as useful efficiency.

| Library | Level | No-progress guard calls current → light | Guard calls / 100 requests current → light | Scheduled calls current → light | Saving with current counts fixed → actual |
|---|---|---:|---:|---:|---:|
| 50 | off | 1873 → 2640 | 10.40 → 13.32 | 0 → 0 | 32.23% → 14.49% |
| 50 | ir25 | 1759 → 2275 | 10.03 → 12.37 | 1512 → 1367 | 17.27% → 12.41% |
| 200 | off | 1476 → 1822 | 8.70 → 10.27 | 0 → 0 | 37.38% → 26.91% |
| 200 | ir25 | 1357 → 1560 | 8.13 → 9.33 | 1933 → 1826 | 16.19% → 14.05% |

Every guard MISS is logged as `force:4` (no progress); every additional scheduled MISS is `force:62` (periodic knob). Guard calls take priority, so more guard calls partially replace scheduled calls at ir25. The observed mechanism is consistent with the offline retrieval deficit: less reliable retrieved actions/progress tracking → more no-progress fallbacks and longer episodes → higher M/N, consuming the tower-skipping benefit. The decomposition is exact: relative to holding current counts fixed, the IR penalty is the increase in M/N plus ε times the change in look rate. It does not identify whether each individual guard was caused by a wrong progress estimate or a physically poor action; the trajectories differ.

## Tower-skipping audit and frozen verdict

| Library | Level | Current N / V / M | Light N / Q / T / M | Light tower-on-HIT violations |
|---|---|---:|---:|---:|
| 50 | off | 18015 / 9076 / 1873 | 19816 / 9954 / 2640 / 2640 | 0 |
| 50 | ir25 | 17534 / 8832 / 3271 | 18398 / 9257 / 3642 / 3642 | 0 |
| 200 | off | 16969 / 8545 / 1476 | 17739 / 8936 / 1822 / 1822 | 0 |
| 200 | ir25 | 16683 / 8409 / 3290 | 16726 / 8428 / 3386 / 3386 | 0 |

Every light decision satisfies logged T=M, with exactly one tower/full-policy call on each MISS and none on any HIT. Legal blind continuations and policy tails have Q=T=M=0. Per-request work, cumulative episode counters, terminal totals and summary ledgers agree; model hashes and fixed ir25 settings match the frozen artifacts. The implementation achieves tower skipping; the retrieval/SR trade-off causes the negative design result.

| Library | Level | Δ SR ≥−2 pp | Every task ≥−5 pp | Paired L ≥−5 pp | IR saving ≥25% off / ≥15% ir25 | Joint result |
|---|---|---|---|---|---|---|
| 50 | off | FAIL | FAIL | FAIL | FAIL | FAIL |
| 50 | ir25 | FAIL | FAIL | FAIL | FAIL | FAIL |
| 200 | off | FAIL | FAIL | FAIL | PASS | FAIL |
| 200 | ir25 | PASS | FAIL | FAIL | FAIL | FAIL |

Logged H100-host encoder-only medians are 4.09–4.40 ms (p95 12.69–20.12 ms). These runtime timings exclude the subsequent AWM projection and do not certify the frozen end-to-end deployment cost gate; all exceed its historical graph-basis sensitivity budgets. The primary IR comparison retains the frozen epsilon rather than substituting new timing prices after the outcomes.

The boundary Δ=−2 pp is a pass for that individual criterion; it does not override task or paired-bound failures. No cell/setting is promoted by pooling away another failure. The separate frozen R14 task-table saving criterion remains a labeled projection in PROTOTYPE2.md, not an outcome of these off/ir25 arms. GATES.md and PILOT.md are hash-verified unchanged. The conclusion supports D5: retain the vision tower in the main design and report this light, image-sensitive student as the unsuccessful comparison.

Reproduce from the repository root:

```bash
taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r15/qk/astra/pilot_results.py
```

Sources and exact results: [PILOT_RESULTS.json](PILOT_RESULTS.json) (all input paths/SHA256, checks, offline values, counts and intervals); [paired episode rows](pilot_results_episodes.csv); [analysis script](pilot_results.py). Only the eight named B-dev arms were read. No test-A data or run-root writes.
