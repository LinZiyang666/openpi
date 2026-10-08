# Handoff to coordinator and sol

**Ready to implement and run the independent B pilot.** Algorithm/scalars, forecasts and the exact test grid froze at 2026-10-04T16:40:30.172320+00:00. Start with SPEC.md and `configs.json`; run `pilot_grid.json` before interpreting A arms as target-controlled. No launch or remote operation was performed here.

The proposed distance threshold uses true nearest geometric distance and one episode-local committed-cost feedback. Width is `1.0` globally; cell/dose scalar fits use library/B data only. It passed 44/44 correlated development replay budget checks (worst error 0.0188) and retained material score-dependent placement. Closed-loop delivery and value are still unmeasured. Static B recalibration alone passed 32/44 checks. Do not promote replay to a live success claim.

Files: SPEC.md (exact implementation/gates), PREDICTION.md + predictions.csv (prospective forecasts), PLAN.md (measurement/statistics), OFFLINE.md (executed B evidence), REQUESTS.md (early and final requests), configs.json (immutable scalar payload), test_grid.csv (full paired/host-pinned grid), pilot_grid.json + static_pilot_grid.json (new B jobs), validation_selection.json (exact feasible fresh pairs, common across models/sizes per suite), used_b_identities.json + development_splits.csv (selection audit), controller.py/selfcheck.py (reference arithmetic).

Test plan: 256 arms, 128,000 episode runs; four replicates at the two primary doses and two at the curve extensions, with repeated off and pure controls. Approximate test time 3.7–6.4 fleet days from the brief's throughput, plus pilot. The frozen cell order keeps GR00T local and pi05 H100. Do not trim replicates after reading outcomes and retain the dependence caveat in the power calculation.

Important integration correction: recorder `distances` are transformed wrapper scores here, not geometric distance. Use `_dist(q)[7].min()`. We reconstructed corrected cache proposals against saved HIT actions to float precision before computing shadow labels. A MISS action compared with its own shadow does not measure wasted-call error.

Open items: real-plugin tests and overhead; true independent B selection; pilot cost/engagement/replicate-variance results; size-500 library-only transfer; active/Spatial shadow labels; pure-control lifecycle and rollout-seed verification. If gates fail, keep the failed candidate and require a separately versioned B-only redesign; never adjust using A. Deep B mechanism work can continue without modifying frozen config/forecast files.

ANALYSIS.md and the Chinese-first REPORT.md are intentionally deferred until the coordinator resumes with the completed frozen grid. TEST_DESCRIPTIVE.md contains the explicit A boundary and no newly computed A results.
