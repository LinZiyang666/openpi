# Pre-emission prediction — GC_dist

UTC timestamp: 2026-10-02T18:22:47.770823+00:00

Frozen before emitting either r10_corr3 run. No closed-loop results, clients,
journals, summaries or server logs were read. Inputs are B libraries and their
nested subsets plus the permitted G/Stage-2b LOEO fits and arm specifications.

One variant, one global shape for all 24 cell-sizes:

`r = min geometric distance to the actual retrieved rows / held-out-episode scale`

`strength = 0.5 * clip((2.0 - r) / 1.25, 0, 1)` for step > 0.

Step 0 strength is zero. Thus strength is .5 at r <= .75, .4 at r=1,
.2 at r=1.5, and zero at r >= 2 or nonfinite distance. No task gate or
task-specific scale is added. Each cell-size has one equal-episode-weighted
median d1 from five folds with held-out episodes excluded from metric fitting
and candidates. Final G and Stage 2b LOEO head arrays remain unchanged.
The corrected anchor continues to supply the existing blind tail.

This shape won the 18 predeclared candidates by mean relative MSE across the
24 B-only tuning-CV panels. Mean change vs no corrector: LOEO
-3.669%, GC_dist -4.096%; GC_dist
beats LOEO in 19/24 panels. These are action errors,
not success rates. Candidate selection and these comparisons use the same CV;
PCA is fixed to the deployed B-subset basis. Calibration donor sets are smaller
than the final library, without a fitted size extrapolation. See OFFLINE.md.

Prediction fixed now: GC_dist should reduce overcorrection on distant looks and
is most likely to help or tie LOEO for GR00T and the smaller libraries. A large
closed-loop success gain is unlikely. It may lose a little of LOEO's useful bias
correction on the largest pi05 libraries; offline pi05 size-500 already shows
that tradeoff. At very large calibrated distances it returns exactly to the raw
G cache action. This does not make the entire trajectory equal to G, because
earlier corrections can change later states and no-progress guard timing.

No numerical closed-loop success lift or policy-call saving is claimed.
Drift augmentation is not used, preserving a clean strength-only ablation.
No evaluation is launched by this task.
