"""Cache-size ablation (X9b): how library size affects pure-replay success rate.

Experiment 2 of the ``ablation_study`` family. Measures the closed-loop success
rate of a *pure cache* configuration (``gate: always_search`` +
``judge: always_hit``, no threshold) as the library grows from 1 to 45 successful
trajectories per task.

Design and pre-registration: ``logs/cache_size_ablation_plan.log.md``.

Module map:
    emit_size_grid      build the deterministic nested size grid from collected h5
    emit_episode_lists  turn a size grid into ``--episode-list`` files for the builder
    build_size_artifacts drive the 12 per-size library builds
    emit_size_yamls     derive the 16 arm yamls from the baseline config
    run_recal           re-run production LOEO calibration at the S1/S6 endpoints
    run_size_eval       conductor driver for the paired A-pool evaluation
    analysis/           paired statistics and plots
"""
