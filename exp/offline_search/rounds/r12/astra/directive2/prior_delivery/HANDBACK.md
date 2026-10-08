# Handback

The BRIEF §6 offline deliverables are complete in this directory. Start with REPORT.md, then SPEC.md and PREDICTION.md. The primary proposal is frozen: a progress-aware pre-look safety atlas, a short memory of the last genuine third-person observation, current wrist vision and observed robot displacement. The gate is task-agnostic; task-indexed metrics and correction heads are explicitly layer 5.

This is a proposal to pilot, **not** a claim of lower matched-SR IR. Forecasts on the existing pi05 request-full path are marginal; the spatial larger-library scenarios are more promising than l10, and stock-speed full routing would improve the economics. Guard timing remains the key uncertainty. GR00T has offline evidence only until an actual one-camera path and costs exist.

## Concrete handoff to sol

- `atlas_reference.py`: pure prelook planner, safety-label action comparison and half feature construction.
- `fit_selected_atlas.py`, `final_atlas/*/{table.npz,fit.json}`: final B-library safety tables, row maps, successors, state/sigma scales and main metric references. These are **not** complete plugin fits; half correction heads and exact production distance calibration remain sol's implementation work.
- `SPEC.md`: exact fit, serving, memory-age, tail/correction and ledger contracts; primary plus a secondary post-wrist acquisition idea.
- `EXPERIMENTS.md`: B-first costs/shadow/controller pilot and frozen A-grid procedure, including full-plus-displacement control and matched-SR buyback.
- `OFFLINE.md`, raw JSON/NPZ outputs and scripts: reproducible evidence; `validation.json` and `FINAL_MANIFEST.json`: checks and freeze hashes.
- `REQUESTS.md`: early recording requests. No design-specific closed-loop recording was received or read during this exploration.

The primary requires explicit per-camera availability and mode-specific correction. R7's metric-only wrapper is insufficient. Preserve the existing action/guard/knob lifecycle. The narrow permission to use valid full-policy camera memory after its committed tail must be separately verified; it must not bypass arbitrary lifecycle failures. Never label stale or deferred camera keys as fresh observations.

## Open questions, recorded rather than sent as questions

1. Does the safety map remain calibrated on cache-driven B trajectories? Existing allowlisted dev logs lack the visual keys needed to answer this.
2. How much do half-view progress errors delay guard intervention, and does the resulting policy-call change erase the saving?
3. Can full looks retain stock-path speed under per-request camera control? The measured extra full-look overhead is large relative to the available saving.
4. Why do separate wrist-plus-completion measurements cost less than full encoding, and does that relation survive the real mixed request path and another GPU? A full-information split-camera timing control is needed.
5. Does observed displacement improve the full-information closed-loop baseline as the offline result suggests? Compare against it before attributing an advantage to camera omission.
6. What is the local B-dev SR-per-IR buyback slope around each target? The coarse existing dev grid does not identify it reliably.
7. How should a half query interact with the distance-threshold knob? Primary support is off/periodic; raw full/half distances must never share an unadjusted threshold. A separate B-only calibration pilot is deferred.
8. GR00T requires genuine one-camera feature/dispatch validation and prices. There is no positive deployment claim to transfer from pi05.
9. Curated current-library replication and additional nonserved B inits for full-size libraries remain outside the completed replay scope.
10. Fixed selected-library PCA and diagnostic correction-scale/training conventions are documented approximations. Sol's exact production replay and B pilot must check the resulting gap.

## Operational status

Only offline CPU work was performed under the assigned affinity, with single-threaded BLAS and the specified virtualenv/PYTHONPATH. The GPU driver was unavailable; no GPU work was attempted beyond its status check. No server, chain, worker, sync or evaluation launch occurred. No prohibited commands were used. Writes stayed in this directory. The other R12 explorer directory and forbidden A recordings were not read. All queued offline subprocesses are checked for completion before handback; there is no intentional background work to resume.

`PREDICTION.md` contains timestamped subjective forecasts and sensitivity assumptions, clearly separate from measured offline evidence. Do not overwrite it after receiving this design's closed-loop results. Any later revision should be a separately timestamped addendum.
