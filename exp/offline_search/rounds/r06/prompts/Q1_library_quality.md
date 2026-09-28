<task>
Your question (Q1): **how do we judge the quality of a library — globally, per task and per state — using only the
library and recorded trajectories, in a way that transfers to other benchmarks and robots?** Owner's idea: leave one
trajectory out and search the library along it (as in the earlier LOTO/LOEO offline solving). Build and evaluate that
idea and alternatives:
- Leave-one-episode-out over library episodes, and held-out recorded trajectories (`queries/*_inf` and `*_cache`),
  searching with the deployed A retrieval (same features, metric, kernel): coverage (distance to the nearest
  other-episode row, relative to the library's own distance scale), action / successor prediction error along the
  trajectory, where along the trajectory coverage breaks (phase / progress), per task and per state.
- Compute the scores for every library we have: `current` (50), `demo100/200/300`, `bpool_*` (500), `grow250`, for both
  models and both suites, and test whether they predict the closed-loop SR of A (and the gap to pure inference, and
  the gain of B over A) across all existing cells and library sizes (`r05_demo_curve`, `r05_growth`, `r05_ptail`,
  `r05_x`, `r04_blind`, `r02_g*`, `r05_q1`, `r05_q2`), including per-task SR. Report rank correlations with intervals.
- Deliver a portable library-quality score (global and per task, and a per-state version usable online at an anchor)
  with its calibration recipe.
Your directory: `exp/offline_search/rounds/r06/ideation_Q1/`. CPU range: 10-13,54-57.
</task>
