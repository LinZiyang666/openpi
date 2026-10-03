# B-library episode-fold comparison

Five outer episode folds; four inner episode folds calibrate each outer training set.
Metric, sigma, residual head and distance reference exclude each outer validation episode.
PCA is fixed to the deployed B-subset basis. Failed episodes are retained.
No-corrector = same G cache synthesis without a residual; guard interventions are not simulated.
MSE = episode-balanced mean normalized 10-step × 6-motion squared residual.
Distance bins use held-out query d1 / inner-calibrated scalar; step 0 is separate.
CV chooses the gate and reports it on those same folds. This is tuning CV, not independent evidence.
Intervals are 2,000 task-stratified episode bootstraps, conditional on the chosen rule.

Selected rule: `{"peak": 0.5, "plateau": 0.75, "cutoff": 2.0}`. All 24 arms use this shape.
Equal-cell-size mean changes vs none: LOEO -3.669%; dist -4.096%.
Dist improves over LOEO in 19/24 cell-sizes. 252,218 validation rows in total (nested sizes reuse episodes).

| Cell-size | none MSE | LOEO MSE | dist MSE | LOEO vs none | dist vs none | dist vs LOEO [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| pi05_l10_50 | 0.254868 | 0.255044 | 0.252307 | +0.07% | -1.00% | -1.07% [-1.93,-0.27] |
| pi05_l10_100 | 0.194940 | 0.186390 | 0.186368 | -4.39% | -4.40% | -0.01% [-0.62,+0.63] |
| pi05_l10_200 | 0.153078 | 0.144265 | 0.144207 | -5.76% | -5.80% | -0.04% [-0.57,+0.46] |
| pi05_l10_300 | 0.138827 | 0.130581 | 0.130254 | -5.94% | -6.18% | -0.25% [-0.99,+0.41] |
| pi05_l10_400 | 0.136620 | 0.128576 | 0.128357 | -5.89% | -6.05% | -0.17% [-1.11,+0.65] |
| pi05_l10_500 | 0.126302 | 0.117193 | 0.118191 | -7.21% | -6.42% | +0.85% [+0.19,+1.57] |
| pi05_spatial_50 | 0.335988 | 0.325699 | 0.326112 | -3.06% | -2.94% | +0.13% [-0.27,+0.52] |
| pi05_spatial_100 | 0.267102 | 0.259836 | 0.257951 | -2.72% | -3.43% | -0.73% [-1.60,-0.03] |
| pi05_spatial_200 | 0.188638 | 0.176395 | 0.176133 | -6.49% | -6.63% | -0.15% [-0.97,+0.60] |
| pi05_spatial_300 | 0.172075 | 0.159229 | 0.159696 | -7.47% | -7.19% | +0.29% [-0.24,+0.78] |
| pi05_spatial_400 | 0.161552 | 0.147889 | 0.149240 | -8.46% | -7.62% | +0.91% [+0.34,+1.45] |
| pi05_spatial_500 | 0.150893 | 0.136774 | 0.138151 | -9.36% | -8.44% | +1.01% [+0.55,+1.44] |
| groot_l10_50 | 0.346383 | 0.351692 | 0.344901 | +1.53% | -0.43% | -1.93% [-2.48,-1.34] |
| groot_l10_100 | 0.271077 | 0.271244 | 0.266492 | +0.06% | -1.69% | -1.75% [-2.50,-1.04] |
| groot_l10_200 | 0.244009 | 0.238282 | 0.236171 | -2.35% | -3.21% | -0.89% [-1.38,-0.45] |
| groot_l10_300 | 0.219628 | 0.214165 | 0.211967 | -2.49% | -3.49% | -1.03% [-1.46,-0.63] |
| groot_l10_400 | 0.214159 | 0.208961 | 0.206845 | -2.43% | -3.42% | -1.01% [-1.52,-0.58] |
| groot_l10_500 | 0.205687 | 0.198630 | 0.198201 | -3.43% | -3.64% | -0.22% [-0.54,+0.12] |
| groot_spatial_50 | 0.307533 | 0.303887 | 0.302836 | -1.19% | -1.53% | -0.35% [-0.83,+0.11] |
| groot_spatial_100 | 0.263441 | 0.260416 | 0.259252 | -1.15% | -1.59% | -0.45% [-0.90,+0.06] |
| groot_spatial_200 | 0.222389 | 0.219376 | 0.217210 | -1.35% | -2.33% | -0.99% [-1.42,-0.58] |
| groot_spatial_300 | 0.212596 | 0.207204 | 0.205651 | -2.54% | -3.27% | -0.75% [-1.37,-0.26] |
| groot_spatial_400 | 0.199773 | 0.193976 | 0.192081 | -2.90% | -3.85% | -0.98% [-1.54,-0.50] |
| groot_spatial_500 | 0.192884 | 0.186769 | 0.185602 | -3.17% | -3.78% | -0.62% [-0.99,-0.29] |

## Distance bins, equal-cell-size means of relative changes

| Calibrated-distance bin | Rows | Cell-sizes | LOEO vs none | dist vs none | dist vs LOEO |
|---|---:|---:|---:|---:|---:|
| step0 | 6200 | 24 | +1.13% | +0.00% | -0.90% |
| [0,0.75) | 24374 | 24 | -6.75% | -6.75% | +0.00% |
| [0.75,1) | 117546 | 24 | -4.74% | -4.84% | -0.09% |
| [1,1.25) | 70064 | 24 | -4.43% | -4.48% | -0.02% |
| [1.25,1.5) | 17966 | 24 | -2.78% | -3.00% | -0.15% |
| [1.5,2) | 10022 | 24 | -1.09% | -1.30% | -0.08% |
| [2,3) | 4454 | 24 | +0.86% | +0.00% | -0.51% |
| [3,inf) | 1592 | 19 | +11.23% | +0.00% | -8.47% |

All per-cell-size bins: `distance_bins.csv`; row sufficient statistics: `cv/*/rows.npz`.
Those statistics retain episode, original parent row, task, step, fold, raw d1, training-only scale,
mean y², mean y·prediction and mean prediction², sufficient to reproduce every candidate MSE.
Calibration artifacts contain one scalar per cell-size and no task-specific gate parameter.
Scales use smaller held-out-fold donor libraries (outer training 80%; nested calibration 60% of full);
no size extrapolation is applied. This tends to soften the gate relative to a full-library reference.
The raw scale and fractional attenuation can differ between final serving and outer validation.
The cutoff is in this held-out-episode reference, not in Opus’s in-library LOEO units.
Vectorized float32 offline arithmetic is numerically close, not byte-identical, to single-query serving.
Serving reuses the exact G retrieval and exact original head; selftests check its correction bytes.
Action imitation error is not closed-loop success; no A or closed-loop trajectory was used.
