"""RIT-Pareto line: IR-addressed risk-indexed thresholds on the GTP libraries.

Four Pareto frontiers (libero_spatial / libero_10 x no gate / H gate) traced
with the RIT-PL estimator (``exp.dispatch_surface.rit_pl``) on a shadow
calibration collected directly on the official pruned_init evaluation pool.

Modules:
  shadow_cohort   sample + materialise the shadow cohort from the A-pool
  export_rit      calibration yaml; shadow table -> IR-addressed artifacts
  emit_arms       arm yamls + matrices for the no-gate / H-gate layers
  aggregate_rit   three-tier inference ratio + success rate; frontier plots
"""
