# Stage 2b — 公平 Pareto overlay（RPG 坐标）

承重层 = 同协议（always_search + ThresholdJudge + 50-init）；RPG = 参照层（libero_spatial only，异 judge/keybuilder，不承重）。

## libero_spatial

| 点 | 层 | inf | SR% | frontier SR | gain pp |
|---|---|---|---|---|---|
| fh75_ws15_quantile | d1 | 0.270 | 85.0 | — | — |
| fh75_ws10_quantile | d1 | 0.287 | 82.6 | — | — |
| fh40_ws40_quantile | d1 | 0.351 | 90.8 | — | — |
| spatial_fh75_ws10_A | client_controlled | 0.283 | 85.6 | 83.2 | +2.4 |
| spatial_fh75_ws10_B | client_controlled | 0.293 | 87.8 | 83.4 | +4.4 |
| spatial_A_periodic | periodic | 0.280 | 90.4 | 83.6 | +6.8 |
| spatial_B_periodic | periodic | 0.369 | 89.0 | OOR | —(inf 越界) |
| RPG×78 | 参照(不承重) | — | — | — | — |

## libero_10

| 点 | 层 | inf | SR% | frontier SR | gain pp |
|---|---|---|---|---|---|
| fh80_ws10_quantile | d1 | 0.314 | 57.0 | — | — |
| fh60_ws30_quantile | d1 | 0.369 | 60.6 | — | — |
| fh40_ws40_quantile | d1 | 0.417 | 67.6 | — | — |
| fh5_ws40_quantile | d1 | 0.636 | 77.6 | — | — |
| pure_inf | pure_inf | 1.000 | 83.0 | — | — |
| l10_fh5_ws40_A | client_controlled | 0.642 | 77.0 | 77.7 | -0.7 |
| l10_fh5_ws40_B | client_controlled | 0.655 | 76.2 | 77.9 | -1.7 |
| l10_A_periodic | periodic | 0.701 | 82.2 | 78.6 | +3.6 |
| l10_B_periodic | periodic | 0.759 | 82.4 | 79.4 | +3.0 |
