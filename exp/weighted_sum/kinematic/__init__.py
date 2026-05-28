"""Kinematic-factor phase5 replication on weighted_sum d1-best retrieval.

This package replicates the verdict_factor_judge phase5 systematic sweep
(5 groups × kinematic factors × tier_thresholds) but with three changes
relative to the original phase5:

  1. Underlying retrieval: weighted_score_sum_knn d1 (weighted_sum series
     spatial16 best, SR=74%) instead of weighted_rrf_knn d4.
  2. Single super warmup (one yaml that dumps the union of all 240 cells'
     declared factor keys) replaces 148 per-cell warmups.
  3. Eval-side calibration uses ``samples_source.type=offline`` pointing
     at the super warmup raw jsonl on the server filesystem, bypassing
     WarmupPool / preload_normalizer_buffer entirely.

G5 grid is filtered by ``fh + ws <= 0.9`` (mirror threshold_pareto §2.2)
which drops the single degenerate cell (0.5, 0.5); per-recipe 15 pairs ×
3 recipes = 45 G5 cells. Total = G1+G2+G3+G4 + G5 = 48*4 + 45 = 237 cells.

Public surface:
  - ``spec``:         237-cell generator + super_warmup_declared_keys()
                      + build_eval_yaml_for_cell (offline calibration mode)
  - ``super_warmup``: build_super_warmup_yaml + run + 7-check verify
  - ``strategy``:     KinematicSearchStrategy (driver-internal flush
                      writer; no hook overrides)
  - ``runner``:       7-mode CLI (emit-warmup / run-warmup / verify-raw
                      / emit-eval-yamls / run-eval / run-always-warm
                      / analyze)

See ``logs/weighted_sum_kinematic_phase5_replication.log.md`` for the
full G1-APPROVED plan.
"""
