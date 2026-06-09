# cp1 Latency Breakdown — Before vs After Optimization

Per-segment cp1 `check()` latency, **median ms** (libero_10 / cp1_spatial_pool_16 / 2640
steps / CPU, 4 threads). **Before** = stock src `InMemoryBackend`; **After** = full
optimization stack. Zero src changes — all optimizations are exp-layer subclasses injected
via `components_hook`. Full reports: [`tuning_final_report.md`](tuning_final_report.md) /
[`rrf_final_report.md`](rrf_final_report.md).

## weighted_sum

| segment | before | after | speedup |
|---------|-------:|------:|--------:|
| collect | 0.006 | 0.005 | — |
| gate    | 0.002 | 0.002 | — |
| build   | 0.948 | 0.446 | 2.1× |
| search  | 33.921 | 3.539 | **9.6×** |
| judge   | 0.013 | 0.009 | — |
| fetch   | 0.005 | 0.004 | — |
| **total** | **35.489** | **4.147** | **8.6×** |

*After = prebuilt-matrix + prenorm-dot GEMV + LEAN search + batched-avgpool build. R5 vision
release additionally reclaims 1059.7 MB with no latency change.*

## weighted_rrf

| segment | before | after | speedup |
|---------|-------:|------:|--------:|
| collect | 0.005 | 0.005 | — |
| gate    | 0.002 | 0.002 | — |
| build   | 1.180 | 0.495 | 2.4× |
| search  | 30.873 | 3.812 | **8.1×** |
| judge   | 0.380 | 0.431 | — (kinematic composite) |
| fetch   | 0.001 | 0.001 | — |
| **total** | **32.876** | **4.875** | **6.7×** |

*After = same R1/R2 backend (auto-applies) + LeanRRF search + batched build + vision release.
Retrieval (search+build = 4.31 ms) matches weighted_sum (3.99 ms); the total gap is entirely
the kinematic-composite judge segment (0.43 ms vs threshold 0.01 ms), not retrieval.*
