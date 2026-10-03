# Handback to sol and the coordinator

**Ready for implementation, not launched.** Start with [REPORT.md](REPORT.md), then implement [SPEC.md](SPEC.md). The authoritative numerical payload is [freeze.json](freeze.json); [arm_grid.csv](arm_grid.csv) is its flat arm table. [PREDICTION.md](PREDICTION.md) was timestamped before this explorer read any R11 closed-loop result. No such result has been read at handback either.

## What sol must implement

* Add a task-agnostic layer-4 wrapper around the existing R10Recipe real-look query. Preserve the existing corrector, no-progress judge, one policy-tail block and all history/reset semantics. No changes to blind cadence for score collection.
* Implement the geometric-distance threshold, the limited neighbour-disagreement comparator, the pooled learned-error half-random hybrid, and the adaptive version. The exact feature vector, probability formula, tie handling and cost-feedback update are in SPEC. There is no per-task layer-4 table, coefficient vector, threshold, rate or schedule.
* Load each cell's `data/<cell>/predictor.json` for error-score arms. For adaptive arms, use the sorted `predicted_error` array in the matching `signals.npz` as the empirical CDF reference. These are scalar scores from outer-held-out predictor models, not actual error labels at runtime. Embed the needed immutable arrays into the fit artifact; serving should not load exploration tables by path.
* Read `dose`, `threshold`, `tie_probability`, `beta` and `eta` from the exact frozen arm row. Keep guard overlap as OR, never an extra policy call. The static half-random mixture is `q/2 + indicator/2`, not floor-coin OR hard threshold.
* Synchronize the adaptive controller against committed five-control ledger counts. Include guard calls and anomalous lifecycle looks. Keep episode-local state; make repeated same-step proposals idempotent. `controller.py` demonstrates this API but is not a production plugin adapter.
* Keep the knob-off baseline as the unchanged R10Recipe. Preserve its policy-tail anchor before applying a knob-only miss. Do not reset progress memos after a knob call, alter the corrector, overwrite actual hit history or serve the discarded cache tail.

## Important files

| File | Purpose |
|---|---|
| `SPEC.md` | Complete implementation and library-calibration contract |
| `ARM_GRID.md`, `arm_grid.csv`, `freeze.json` | Proposed methods, exact scalar settings and immutable predictions |
| `PREDICTION.md` | Per-arm expected IR, explicit offline stress ranges and subjective SR direction/size |
| `OFFLINE.md`, `offline_curves.csv` | Every cell's dose → miss fraction → IR curves |
| `value_evidence.csv`, `matched_ir_value.csv`, `supplement.json` | Episode-balanced capture, added capture after guard overlap, RMS/outlier/within-episode checks |
| `data/<cell>/signals.npz` | B-held-out scores, risk labels, folds and parent query IDs; `top` is subset-local |
| `data/<cell>/predictor.json` | One pooled, task-free score model per cell |
| `data/<cell>/provenance.json` | Nested episode splits and source hashes |
| `data/<cell>/analysis.json` | Full calibration and controller-simulation results |
| `experiment.py` | Strictly nested error-score generation; lower-layer whole-episode refits |
| `signals.py`, `analyze.py` | Minimal cadence accounting, overlap, threshold solves and adaptive simulation |
| `crosscheck.py`, `crosscheck.json` | Agreement with opus's shared accounting on identical input sequences |
| `controller.py`, `selfcheck.py`, `selfcheck.json` | Runtime arithmetic reference and independent checks |
| `RUN.md`, `final_audit.json`, `MANIFEST.sha256` | Executed workflow, ownership/consistency audit and artifact hashes |

## Open questions and integration decisions

These are recorded for handoff, not requests to stop the work.

1. **Matched-IR controls:** opus's current Spatial grid uses different target levels. Keep this frozen grid intact and add random controls at the missing matching targets, or let the coordinator select a budget subset before any new evaluation. `COORDINATION.md` lists the exact missing pairs. A nearest target is not a matched-budget comparator. Counts remain comfortably inside the brief's suggested overall budget with those additions.
2. **Representation:** astra uses the frozen corrector's fixed selected-library PCA convention; opus's separate table refits PCA in smaller libraries. Their numerical thresholds, risks and guard rates must not be mixed. The cost model agrees exactly when fed the same table. Sol should reproduce astra's table for these frozen state-dependent arms, even if schedule arms use opus's table.
3. **Runtime score access:** wrapper scores are kernel logits, not geometric distances. Extract distance before that transformation. The error score needs the uncorrected weighted chunk and exact selected donor weights; using the corrected returned chunk changes the frozen feature definition.
4. **Reason code:** SPEC proposes a common new knob-only reason; sol should reconcile it with the schedule wrapper's reserved code. Preserve the original guard reason when the guard fires. No algorithm or budget depends on that logging choice.
5. **Adaptive lifecycle:** the reference accepts committed ledger counters; the integration must not count speculative proposals or discarded tails. Missing policy tails, an episode ending mid-commit, and invalid-lifecycle extra looks are already distinguishable in the real plugin ledger. Reuse that truth instead of reconstructing costs from the desired target.
6. **Default off / utility:** library policy-action error cannot establish that extra calls improve success. Do not add a learned “spend is beneficial” classifier from these data or silently enable dense-library spending. The higher-budget adaptive variants are experiments, not a universal hard-budget guarantee.
7. **Scope limits:** no curated current-library arms or non-test B-val recordings were used. The optional non-test pilot is the coordinator's work. A pilot can check realized cost and lifecycle handling; it must not become a route to test-A fitting.
8. **Forecast limits:** prospective SR intervals are declared subjective priors, not statistical intervals. The numerical freeze records the explorer's timestamp and absence of outcome access; global timing against coordinator jobs was not checked by reading run outputs.

## Dropped methods and limitations

Standalone imminent-gripper and gripper-disagreement triggers are dropped after weak long-task motion-error evidence. Their Spatial signal does not justify a new standalone branch. Dense low-target adaptive initialization failed the offline feasibility check and is not in the grid. Pure predicted-error thresholds and disagreement hybrids remain offline baselines to avoid unnecessary live arms.

The strongest score still only predicts action discrepancy on pure-policy B trajectories. It does not observe rescue utility, action-induced state drift, success or the eventual closed-loop guard rate. R6's lack of a matched-budget SR advantage for disagreement remains the main prior. The report therefore claims better proxy concentration, not established success efficiency.

The numerical prediction freeze is unchanged. A subsequent reference-code clarification made static off/all-call endpoints work outside the library score support; none of the proposed static doses is at an endpoint. See `POST_FREEZE_NOTES.md` and the original/final source hashes.

## Checks to retain during production integration

Reuse the analysis checks and add integration checks for ordinary cache hit/tail, knob-only miss/policy tail, simultaneous guard+knob, next-anchor handback, same-step retries, independent connection state, episode reset, and nonfinite-score fallback. Verify actual ledger `N,V,M`, whole-episode exclusion in the builder, and unchanged knob-off action bytes. The explorer has not modified or tested a production adapter, launched a model, or evaluated a closed-loop arm.
