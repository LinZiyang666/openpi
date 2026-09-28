<task>
Your question (Q3): **where in a trajectory should MISS be placed to buy the most SR per call, so SR is kept while IR is
pushed down?**
- Estimate the value of a MISS at a decision from the data we have: the randomized CALL vs CACHE experiment (`r04_k5`,
  causal, estimator `rounds/r04/k5_rand/estimate.py`, analysis `rounds/r05/q3_callvalue/`), paired A vs B episodes
  (`r05_ptail`/`r05_x` vs `r05_q1`/`r05_q2`: find where B-only successes diverge from A and what B's MISS did there),
  guard-fired points and their outcomes, and leave-one-out trajectories (where the cache's predicted chunk deviates
  from what the trajectory actually did).
- Candidate placement signals available online at an anchor: retrieval coverage (nearest-neighbour distance vs the
  library's LOEO distance scale), neighbour action disagreement, progress / phase along the matched library episode
  (e.g. before grasp / contact / release events), stall and stuck signals, time since last MISS. Measure which ones
  predict "a MISS here helps", with intervals; distinguish preventive (before failure) from reactive (after failure).
- Propose a placement rule (which decisions get the policy, how long the policy keeps control, how control returns to
  the cache) with thresholds calibrated from the library, portable across benchmarks, and compare it on paper against
  the current reactive guards.
Your directory: `exp/offline_search/rounds/r06/ideation_Q3/`. CPU range: 22-25,66-69.
</task>
