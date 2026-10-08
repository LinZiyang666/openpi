# R14 design handback

Deliverables through the offline design phase are complete. Start with `SPEC.md`; exact parameters are in `allocation_fits.json`, switches in `design.json`, and arm/identity contracts in `pilot_grid.csv`, `pilot_selection.json` and `test_grid.csv`. `PREDICTION.md` is frozen before new closed-loop outcomes.

The main code finding is that task-indexed guard dictionaries do not control the deployed active no-progress verdict. They are a CPU equivalence check, saving a GPU arm. The extra continuity scale is action-relevant only in the fresh branch and is isolated explicitly. Task geometry, donor scope and correction heads remain distinct switches; foreign donors must be scored in the query's coordinate system.

For sol: implement the full-row pooled fits and lower-layer switches, use the original periodic controller with the frozen task caps, and perform real-plugin lifecycle/row-mapping/coin equivalence checks. The CPU reference compared 6604 proposals to the actual periodic query logic, but is not a serving adapter. Keep all GR00T arms on the local 4090 and all pi05 arms on H100.

For the coordinator: `REQUESTS.md` queues 68 B pilot arms / 3400 episodes after the R13 scheduling decision. It launches nothing. The frozen A grid has 128 arms / 64,000 episodes. Budget or engagement failures stop the affected matched-budget claim; retain them and request a separately versioned B-only redesign. No A-based fitting, dose change, host migration or best-replicate selection is authorized by this design.

This is a design handback, not the later results handback. `ANALYSIS.md` and the owner-first Chinese `REPORT.md` are intentionally deferred until the coordinator resumes this work with completed runs. The outstanding data/capacity questions are in `STATUS.md`.
