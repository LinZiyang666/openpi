# N1 Live 验证结果（Stage 1b）

| run | gate | skip% | SR(N1/base) | ΔSR pp | SR-ok | vs-periodic | overall |
|---|---|---|---|---|---|---|---|
| spatial_fh75_ws10_A | client_controlled | 12.8 | 85.6/82.6 | +3.0 | ok | pending | pending |
| spatial_fh75_ws10_B | client_controlled | 20.1 | 87.8/82.6 | +5.2 | ok | pending | pending |

| run | inf(live/base) | Δinf | net@4/34/70 | b/c |
|---|---|---|---|---|
| spatial_fh75_ws10_A | 0.283/0.287 | -0.004 | +1.7/+5.6/+10.2 | 22/37 |
| spatial_fh75_ws10_B | 0.293/0.287 | +0.006 | -0.9/+5.1/+12.3 | 18/44 |

口径：skip% 由权威 `searched`（periodic 按 ordinal 重建）；SR/periodic 配对按 `(task_id, subset_init_state_idx)`，要求 live/baseline-journal/baseline-gate-rows 三方 unit 等集；baseline_inf_ratio 来自 Stage-0 gate_rows（episode-global max-attempt 去重）。**overall pass** 需 SR 保真（ΔSR ≥ −1pp）**且** 同预算 N1 SR ≥ periodic（|Δskip| ≤ 2pp，越界=FAIL）；periodic 未跑 = pending（非 pass）。
