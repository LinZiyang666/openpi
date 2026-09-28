<task>
You are R5 ideation agent B: solve the hyperparameters offline (owner's R5 direction, protocol §9 item 10).
The owner asks: how should our hyperparameters be set, and can they be solved offline — from the h5 demos when the
library is built, or from logged closed-loop data — by leave-one-out (leave-one-task-out LOTO / leave-one-episode-out
LOEO) or in closed form, the way the old implementation calibrated RIT thresholds by LOTO (exp/rit_loto/: fit_loto.py,
noise_floor.py, verify_closed_loop.py; logs/ mention the RIT LOTO line) and solved LDA fusion weights in closed form
(exp/weighted_sum/analysis/lcw_fit_weights.py, emit_keybuilder_lda.py), instead of hand-tuning or closed-loop sweeps?
1. Inventory every hyperparameter of the current best methods (AWM: PCA dims, kernel k, kref, whitening pair rule and
   ridge, state weight, early-regime and continuity λ; guard-only MixedJudge / K7: stuck_thr, motion and camera-cosine
   percentiles, lag_thr, overtime, noprog_n, prog_eps, V7 calibration; blind: budget, serving choice, gate thresholds
   (gripper-event mass .20, displacement residual .5, low-motion), phase penalty .05, offsets; quantile h / periodic k).
   For each: how it was set, what evidence exists (R2–R4 arms differ in exactly some of these knobs), sensitivity.
2. For each, say at which time an offline solver is possible — (a) library build time from the demos only,
   (b) with logged closed-loop decisions and outcomes (≈1.6 M decisions under /home/weiland/trace_runs/os_closed_loop/,
   with confounding: labels are observational), (c) with randomized CALL/CACHE data (K5, arriving later) — and build the
   solvers you can now (LOTO / LOEO, closed form where it exists: e.g. Fisher/LDA-type metric or fused guard score,
   quantile thresholds, kernel bandwidth by leave-one-out likelihood).
3. Decisive test: would the offline-solved values have predicted the closed-loop ranking of arms we actually ran
   (noprog 3 vs 4, periodic k, B=1 vs 2, kref 5 vs 8, guard variants, dense vs vision-confirmed stuck, α prior, ridge)?
   Report hits and misses honestly; offline action error does not rank closed-loop methods, so find objectives that
   do (e.g. stale-state error, stall/trap predictors, per-decision cost-benefit).
4. Propose R5 arms: "solver-set" configurations vs the hand-set best, at both library scales.
</task>
