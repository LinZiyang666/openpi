"""RIT offline calibration (LOTO) for the GR00T N1.5 x LIBERO line.

Replaces the rollout-based shadow calibration of the RIT ladder with a purely
offline one: every decision of the W13 build corpus is re-scored against the
S3 library with its own trajectory left out (leave-one-trajectory-out), the
warm rungs are re-denoised from the winner's stored intermediates under the
recorded observation, and the reference is the chunk the teacher executed at
collection time. The resulting shadow table is fitted with the unchanged
``fit_ladders`` LP and compared, curve by curve, with the deployed shadow fit.

Modules (plan: ``logs/rit_loto_calibration_plan.log.md``):
  build_loto_table   LOTO shadow table + parity sample + parity gate (island, GPU)
  noise_floor        two-seed full-inference deviation floor (island, GPU)
  fit_loto           fits, shadow re-fit audit, curve comparison, bootstrap band (CPU)
  emit_verify_arm    verification arm from the frozen fit + verify/smoke init pools
  loto_logger        server-side per-decision log wrapped around the cache interceptor
  verify_closed_loop closed-loop merge/identity checks, offline labels, exceedance report
"""
