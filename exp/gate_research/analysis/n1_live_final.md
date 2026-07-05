# N1 Live 验证结果（Stage 1b）

| run | gate | skip% | SR(N1/base) | ΔSR pp | SR-ok | vs-periodic | overall |
|---|---|---|---|---|---|---|---|
| spatial_fh75_ws10_A | client_controlled | 12.8 | 85.6/82.6 | +3.0 | ok | ΔSR -4.8pp |Δskip|=1.9pp FAIL | fail |
| spatial_fh75_ws10_B | client_controlled | 20.1 | 87.8/82.6 | +5.2 | ok | ΔSR -1.2pp |Δskip|=1.8pp FAIL | fail |
| l10_fh5_ws40_A | client_controlled | 21.2 | 77.0/77.6 | -0.6 | ok | ΔSR -5.2pp |Δskip|=2.0pp FAIL | fail |
| l10_fh5_ws40_B | client_controlled | 32.4 | 76.2/77.6 | -1.4 | FAIL | ΔSR -6.2pp |Δskip|=0.3pp FAIL | fail |
| spatial_A_periodic | periodic | 10.9 | 90.4/82.6 | +7.8 | ok | — | — |
| spatial_B_periodic | periodic | 18.3 | 89.0/82.6 | +6.4 | ok | — | — |
| l10_A_periodic | periodic | 19.2 | 82.2/77.6 | +4.6 | ok | — | — |
| l10_B_periodic | periodic | 32.7 | 82.4/77.6 | +4.8 | ok | — | — |

| run | inf(live/base) | Δinf | net@4/34/70 | b/c |
|---|---|---|---|---|
| spatial_fh75_ws10_A | 0.283/0.287 | -0.004 | +1.7/+5.6/+10.2 | 22/37 |
| spatial_fh75_ws10_B | 0.293/0.287 | +0.006 | -0.9/+5.1/+12.3 | 18/44 |
| l10_fh5_ws40_A | 0.642/0.636 | +0.006 | -0.8/+5.5/+13.2 | 40/37 |
| l10_fh5_ws40_B | 0.655/0.636 | +0.018 | -4.2/+5.5/+17.1 | 40/33 |
| spatial_A_periodic | 0.280/0.287 | -0.007 | +2.6/+5.8/+9.7 | 26/65 |
| spatial_B_periodic | 0.369/0.287 | +0.081 | -23.6/-18.1/-11.5 | 34/66 |
| l10_A_periodic | 0.701/0.636 | +0.064 | -18.6/-12.8/-5.9 | 43/66 |
| l10_B_periodic | 0.759/0.636 | +0.123 | -35.5/-25.7/-14.0 | 38/62 |

口径：skip% 由权威 `searched`（periodic 按 ordinal 重建）；SR/periodic 配对按 `(task_id, subset_init_state_idx)`，要求 live/baseline-journal/baseline-gate-rows 三方 unit 等集；baseline_inf_ratio 来自 Stage-0 gate_rows（episode-global max-attempt 去重）。**overall pass** 需 SR 保真（ΔSR ≥ −1pp）**且** 同预算 N1 SR ≥ periodic（|Δskip| ≤ 2pp，越界=FAIL）；periodic 未跑 = pending（非 pass）。
