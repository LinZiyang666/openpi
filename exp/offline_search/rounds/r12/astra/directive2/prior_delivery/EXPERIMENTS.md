# Experiment plan

No launches were performed. All pilot selection/fitting is B-only. Freeze PREDICTION.md and the chosen fit/source hashes before receiving any changed-controller outcomes.

## First: costs and shadow recordings

Fulfil REQUESTS.md on B dev initial states outside each served library. Verify per-view availability and camera mapping, record both keys and shadow half proposals on unchanged cache-driven trajectories, and evaluate the frozen atlas without modifying the trajectory. Existing R11 B-dev decision logs contain states/actions but not the keys needed for this check.

Measure batch-one full stock, request-full, wrist-only and half-MISS completion. Also measure two separately encoded cameras as a full-information control, because the supplied wrist-plus-completion price is lower than either full-look price. Price all arms on the same owner denominator and report actual latency separately. Charge CPU retrieval/head/planner overhead in wall-time reporting even though the defined owner IR omits it. Do not credit encoding both cameras and discarding one as a half look.

Require actual one-camera feature parity/dispatch before a GR00T rollout. For pi05, an optimization of full dispatch must also be applied to the full-information control. If the old full-path overhead consumes the predicted saving, report that result; do not silently reprice it at stock speed.

## B dev controller pilot

Use the primary SPEC rule unchanged across tasks. Start pi05 l10 and spatial at nested size 50; add size 200 and then 500 if direction is acceptable. Use ten tasks with ten nonserved B inits per task, paired across arms. Size 500 needs additional B inits outside the parent library. Preserve the disjointness manifest. The initial pilot is a mechanism/cost check; it has insufficient power to certify a sub-percentage-point SR loss.

Initial arms per cell-size:

- Unchanged five-layer baseline, periodic knob off.
- Primary half atlas, periodic knob off.
- Full-camera displacement control, knob off.
- Baseline and primary at the same frozen periodic settings calibrated to legacy owner targets `.25` and `.32`.
- On the first size-50 cells only: request-full/no-half camera-dispatch control and a strict-atlas ablation (`state_valve=.1`). These are dev diagnostics, not a menu to select using A outcomes.

The same knob **setting** comparison identifies raw camera savings and guard feedback; setting the same total IR target can intentionally spend the savings and hide them. Then sweep the primary's periodic setting more densely around a useful baseline SR to test buying back any loss. Choose those finer settings with B dev only and freeze them before test. Recalibrate target-to-setting costs using the actual half/completion ledger; retain the full reference settings unchanged as an audit comparison.

For distance-threshold knob compatibility, first do a separate B-only pilot with distinct half/full calibration maps. Keep it off the primary frozen grid until its threshold semantics and underspending are understood; the periodic knob supplies the initial matched-SR comparison.

Inspect paired SR, total work/nominal five-control slots, actual controls if available, true full/half/completion counts, first guard-call time, extra and missed stall events, policy/guard/knob counts, mode exposure, safety mass/state residual, camera memory age, and action/gripper discrepancy on shadow looks. A lower action MSE alone never passes the pilot. In particular, saved vision that disappears into additional guard calls is a failed saving even if raw half rate is high.

Use B results to decide go/no-go and which cells are worth the A grid; do not fit an SR conversion factor from previous A summaries. The recorded R11 B pilot offers only coarse/noisy knob slopes and even has nonmonotone l10 outcomes, so the buyback slope in PREDICTION.md is a sensitivity assumption, not a fitted success model.

## Frozen A grid after B selection

Candidate grid: pi05 × {l10, spatial} × {50, 200, 500}; unchanged full, primary half, and full-plus-displacement control; each at knob off and frozen periodic settings corresponding to `.25` and `.32`, plus at most one finer B-selected setting on each side of a prespecified target SR. Publish the exact arm manifest, fit SHA, source SHA, target SR, settings, seeds, and pricing before launch. Each included configuration gets one complete official set of ten tasks × fifty initial states. Do not read any A outcome until the grid is frozen. Do not tune/retest a failed A configuration.

GR00T uses the analogous grid only after a B-validated actual one-camera path and measured costs. Otherwise keep its proposal as an offline handback. Curated current libraries were inventoried but not replayed in this exploration; add a B-only current-library replication before making a deployment claim about them.

## Matched-SR analysis

Prespecify a target SR from B data lying within both frozen curves. Compare measured total IR at that target, including completion, changed guard calls and the added periodic calls. Report the matched **setting** contrast alongside the matched-SR contrast.

For interpolation, mix frozen policies across episodes rather than assuming a linear pooled-IR curve. With mixture probability p, SR is linear in p; pooled IR is `((1-p)*work0+p*work1)/((1-p)*slots0+p*slots1)`. Episode length changes otherwise distort a straight line between reported IR ratios. If a target is not bracketed, report “not identified”; do not extrapolate or treat an unsuccessful arm as matched.

Bootstrap paired initial states within task, carrying whole episodes and all their counts together. Report intervals for SR difference, IR difference and interpolated matched-SR advantage. Distinguish policy stochasticity from finite-init uncertainty; retain a paired B replicate to estimate batch noise. A point-estimate SR tie is not proof of losslessness. A wide interval is an inconclusive test, not a pass. The go criterion is positive savings at the same SR with uncertainty stated; layer-4 buyback must cost less than the saved IR.
