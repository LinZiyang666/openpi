# Proposed state-dependent arm grid

Frozen 2026-10-03T01:55:50.277415+00:00.

44 proposed arms, excluding the coordinator’s same-batch knob-off controls and opus’s schedules.

All cell-size parameters are shared across tasks. Target levels follow the brief’s small grid; no A outcomes select parameters.

| Method | Cells and targets | Arms | Rationale |
|---|---|---|---|
| distance | Every size-50 cell: .25/.32/.40; L10-200/500: .25 | 16 | Requested simple state trigger; conservative benchmark for richer signals. |
| error_hybrid | Every size-50 cell: .25/.32/.40; L10-200/500: .25 | 16 | Strictly nested error score plus half-uniform floor; strongest consistent error proxy. |
| adaptive_error_hybrid | Every size-50 cell: .32/.40 | 8 | Track actual committed cost; tested only at feasible higher targets. |
| disagreement | Every size-50 cell: .32 | 4 | Cheap nonlearned comparator; the predictor’s incremental gain is often small. |

Standalone gripper triggers, pure learned-error thresholds, and dense low-budget adaptive arms are not proposed. Pure learned thresholds remain an offline baseline, to keep the live grid small. The knob remains off unless a spend target is explicitly selected; the library cannot establish positive causal spend utility.

If the combined arm budget needs trimming, remove disagreement ablations first, then adaptive Spatial arms. Keep the distance and hybrid comparisons plus same-batch knob-off controls.

Exact numeric parameters and per-arm predictions: [arm_grid.csv](arm_grid.csv), [freeze.json](freeze.json), [PREDICTION.md](PREDICTION.md).
