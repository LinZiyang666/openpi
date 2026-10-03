# GC_dist selection protocol

Recorded 2026-10-02T18:09:02Z, before computing astra validation results.

Use only the 24 nested B libraries; retain failed episodes. Final G fits and
Stage 2b LOEO heads are copied unchanged. No augmentation or head refit for serving.
The only change is a multiplier on the existing half-strength residual correction.

Five outer folds: selected episode's position in sol's task permutation modulo 5.
Within each fold, fit main/early AWM metrics and LOEO heads on training episodes
only; validation donors are those same training episodes. LOEO training excludes
the entire query episode and uses regime 2, with the existing early step-0 branch.
Use sol's exact weighted ridge routine, feature layout, regularization and RFF seed.
The unsupervised PCA basis is held fixed to the corresponding deployed G basis;
this is a fixed-PCA diagnostic, not a fully independent representation evaluation.
Action sigma is recomputed on each training fold. No validation actions enter its
metric, head, sigma or distance calibration. All tasks and all query rows are used.

Each outer training set gets four inner episode folds. Refit the metric excluding
each inner fold and measure its queries against the remaining episodes. Pool
non-step-0 nearest-neighbor distances across tasks, with equal total episode mass;
their weighted median is one scalar distance scale for that outer training set.
This calibration deliberately includes metric exclusion, unlike fixed-metric LOEO.
For final serving the scalar is the weighted median of the five outer-held-out
distance distributions. The smaller calibration donor libraries make the scale
conservative; no extrapolation to full library size or task-specific scale is used.

For non-step-0 look queries: r = raw nearest retrieved-neighbor distance / scale;
strength = b * clip((cutoff-r)/(cutoff-plateau), 0, 1).
At step 0 strength is zero (the early metric is a separate, weakly supported regime).
For an exceptional fresh-after-MISS look use the geometric distance of the actual
retrieved rows, rather than its continuity-augmented ranking score. Nonfinite
distance gives strength zero. Blind continuation retains the corrected anchor.

Predeclared candidates: b in {0.25, 0.35, 0.5}, plateau in {0.5, 0.75}, cutoff in
{1.25, 1.5, 2.0}. Select ONE global triple by lowest arithmetic mean across 24
cell-sizes of episode-balanced corrected-MSE / no-corrector-MSE. Exact ties use
smaller b, then smaller cutoff, then smaller plateau. No task-specific gate or
cell/size-specific shape selection. GC_loeo comparator is always blend 0.5,
including step 0. No-corrector comparator uses the same synthesized chunk.

Report normalized 10-step six-motion-channel MSE, RMS as a secondary display,
and distance bins [0,.75), [.75,1), [1,1.25), [1.25,1.5), [1.5,2), [2,3), [3,inf).
Step 0 is separate. All comparisons use the same queries. Conditional episode
bootstrap intervals describe the chosen rule; they do not remove selection bias.
The CV used to choose the triple is explicitly tuning CV, not an untouched test.
Prediction and all calibration/selection hashes must be written before emission.
No closed-loop success prediction is inferred numerically from action MSE.
