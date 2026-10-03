## Data admission and split

Read-only inputs are the existing discovery compact files under `/home/weiland/trace_runs/offline_search_store/derived/r09_astra/compact/r8_<cell>_<variant>.npz`, for four 50-demo cells and variants A/CU/IP. `tools/data.py` opens only lazy task/init members first. An archive containing any init outside 0–29 is rejected before any observation/action payload is accessed; the requested train/eval mask is applied before constructing a dataset. No outcome/success/cost ledger is loaded. No raw run outcome or decision log is needed.

Fit, feature standardization, label fitting and model selection use only inits 0–19. Three CV folds hold out whole init groups (`init % 3`), jointly across all tasks and all three trajectory variants. Thus the same physical initialization cannot appear in both train and validation via another path. Selection is a single equal-cell mean of episode-balanced relative motion MSE. The recipe was frozen in `SELECTION.json` at **2026-10-02 10:32:00.520112 UTC**, before evaluation access.

Final evaluation uses inits 20–29, ten tasks × ten inits = 100 episodes per variant/cell. The episode loss averages fresh look anchors, then averages episodes. These are recorded paths, not counterfactual corrected rollouts. No new closed-loop result was read.

| Cell | Training anchors, A/CU/IP combined | Arm-episodes | Library rows | A evaluation anchors |
|---|---:|---:|---:|---:|
| π0.5 LIBERO-10-50 | 18,919 | 600 | 2,640 | 3,308 |
| π0.5 Spatial-50 | 7,359 | 600 | 1,018 | 1,294 |
| GR00T LIBERO-10-50 | 20,321 | 600 | 2,645 | 3,595 |
| GR00T Spatial-50 | 7,597 | 600 | 1,063 | 1,230 |

The 600 arm-episodes per cell share 200 task/init identities. Bootstrap uncertainty respects that grouping. Library actions are demonstration data, not evaluation outcomes. The weighted retrieved demonstration motion reconstructs the recorded cache shadow motion to better than 1e-4 in every training cell.

For CPU replay tests only, a local reader admits two training episodes per cell before loading array rows. It also filters the plugin's episode identity map. Mixed JSON metadata is framed and identities are parsed before an accepted object is decoded. Tests use the first admitted episode of tasks 0 and 1 with at least 24 recorded rows. The initial short Spatial test failure is retained under the new run root; selecting a sufficiently long test episode fixed only the test fixture. Serving arguments still point to the standard store.

## Model and selection

Let `b_i` be the retrieved ten-control chunk, `y_i` the policy shadow motion minus `b_i`, and `W_ij` the normalized retrieval weight on library row `j`. The observation input has 207 values: 128 projected visual values, eight robot-state values, 70 cached-action values, and capped elapsed decision index. Task/init metadata only forms episode weights and CV groups.

1. Fit one shared RFF ridge `h(x_i)` to the 60-dimensional motion residual. Standardize using training observations only; clip standardized inputs to ±8; append 768 cosine features; ridge alpha 100; fixed seed 260602. No task one-hot is appended.
2. Fit a library-row table `C` by minimizing `sum_i u_i ||y_i - h(x_i) - W_i C||² + lambda ||C||²`, where `u_i` balances arm-episodes and has mean 1. This is a sparse joint ridge solve over existing retrieval incidence. It does not fit or select heads by task ID. Rows unsupported by training have zero local correction.
3. Serve `b_i[:10,:6] + 0.5 * (h(x_i) + W_i C)`. No gripper correction, guard modification, additional look or policy call is introduced.

The row-only ablation omits `h`; because cache motion reconstructs as the weighted demo actions, its objective is equivalent to attaching fitted policy-minus-demo residuals to library rows and retrieving the residuals with the original weights. The selected combination adds a shared observation-dependent component, allowing correction to vary within a retrieval neighborhood.

Seven recipes were declared before evaluation. All have blend 0.5; one common winner was selected for all cells:

| Recipe | Mean training-CV MSE / uncorrected cache MSE |
|---|---:|
| Shared observation head | 0.68007 |
| Row table, lambda 1 | 0.76859 |
| Row table, lambda 10 | 0.88290 |
| Row table, lambda 100 | 0.97415 |
| **Shared + row, lambda 1** | **0.65722** |
| Shared + row, lambda 10 | 0.67338 |
| Shared + row, lambda 100 | 0.67905 |

The same winner also has the lowest CV error in every individual cell; no task-specific or cell-specific recipe switch is used. Final standardization and coefficients are refitted on all admitted 0–19 examples only. No 20–29 row is added to either head or residual table.

## Evaluation on 20–29

Motion MSE below is in each model's normalized action coordinates, averaged over ten controls × six motion channels and then over episodes. Cross-model MSE magnitudes are not physically comparable. Labels are policy shadow draws, not demonstrated optimal actions or success values.

| Cell, A path | No correction | Current per-task head | Shared only | Row only, lambda 1 | Selected task-free |
|---|---:|---:|---:|---:|---:|
| π0.5 LIBERO-10-50 | 0.025958 | 0.013893 | 0.016382 | 0.017782 | 0.015444 |
| π0.5 Spatial-50 | 0.044653 | 0.021697 | 0.024957 | 0.032281 | 0.024181 |
| GR00T LIBERO-10-50 | 0.029501 | 0.014408 | 0.016690 | 0.018561 | 0.015723 |
| GR00T Spatial-50 | 0.020078 | 0.012254 | 0.012937 | 0.014726 | 0.012548 |

Relative to the shared head alone, local residuals reduce A-path MSE by 5.72%, 3.11%, 5.79%, and 3.01%, respectively. A row table alone is insufficient. The selected method retains `(cache MSE - selected MSE)/(cache MSE - current MSE)` = 87.15%, 89.18%, 91.29%, and 96.23% of the current corrector's A-path error reduction.

Transfer to paths containing policy calls remains weaker than the current corrector, but materially better than no correction:

| Cell | A: selected/current − 1 | CU: selected/current − 1 | IP: selected/current − 1 |
|---|---:|---:|---:|
| π0.5 LIBERO-10-50 | +11.16% | +9.74% | +9.47% |
| π0.5 Spatial-50 | +11.45% | +4.03% | +7.17% |
| GR00T LIBERO-10-50 | +9.12% | +4.92% | +6.13% |
| GR00T Spatial-50 | +2.40% | +3.10% | +2.99% |

On A, paired selected-minus-current MSE with 95% init-cluster bootstrap intervals (5,000 draws; fixed ten tasks) is:

| Cell | MSE difference | 95% interval | Episodes better than current / 100 |
|---|---:|---:|---:|
| π0.5 LIBERO-10-50 | +0.001551 | [+0.001295, +0.001841] | 9 |
| π0.5 Spatial-50 | +0.002485 | [+0.001638, +0.003479] | 12 |
| GR00T LIBERO-10-50 | +0.001315 | [+0.000994, +0.001655] | 18 |
| GR00T Spatial-50 | +0.000295 | [−0.000142, +0.000681] | 36 |

There are only ten independent init clusters; these intervals describe this screen, not a general task population. GR00T Spatial is closest to the current head. The other three cells have clear offline teacher-error deficits, so a claim of universal no-loss replacement is not supported.

## Control provenance and limits

The controls are the requested existing production heads and stack fits, copied byte for byte. Their metadata asserts inits 0–19, per-task heads, task one-hot inputs, 384 RFFs and alpha 100. They were trained on more trajectory variants: 73,945 / 20,848 / 80,108 / 19,364 anchors for the four cells in table order. The new head uses only the identity-validated A/CU/IP compact corpus, and 768 shared RFFs. Consequently this is a comparison against the actual serving control, **not a matched-data causal ablation of task identity**. Lower data coverage, differing feature standardization and head capacity can all contribute to the gap. No extra task-indexed model was fitted for this study.

The new head's metadata, label coverage and runtime tensors have no task dispatcher. Existing library retrieval and inherited stack bookkeeping remain unchanged as required. Reordering table rows with an equivalent retrieval permutation leaves the correction unchanged. A poison-identity test makes `task_id`, `episode` and `init` inaccessible to the correction function and still produces the same valid correction API behavior.

No SR can be estimated reliably from the shadow-error deltas alone. The new policy changes later observations, retrieval and guard activation. Actual same-batch closed-loop comparisons are necessary. Owner IR prices count look/call fractions and do not directly include CPU milliseconds.

## Runtime and launch checks

- The new numerical artifacts are approximately 1.04–1.41 MB compressed. Shared head tensors occupy 874,872 bytes/cell; row tables occupy 244,320–634,800 bytes/cell. The serving objects contain neither old per-task head arrays nor fitting observations/labels.
- All library rows receive training retrieval mass except two of 2,640 π0.5 long-task rows. Those unsupported rows carry only 0.025% of A evaluation retrieval weight; their local residual is zero, with the shared head still available.
- CPU correction latency (300 warmed single-query repetitions, synthetic valid inputs; required CPU affinity and one BLAS thread) has median 0.802–0.825 ms and p95 0.886–0.900 ms. This includes visual projection and head/table synthesis, not retrieval, guard or GPU inference. Remote timing is unmeasured.
- Ten unit tests verify split rejection, frozen source identity, standard store/manifest, task independence, row permutation behavior, attribute collision rejection, identical non-corrector fitted state, judge correction exactly once, and remembered corrected anchor.
- Sixteen real CPU plugin tests pass: eight production artifacts plus eight forced grasp-trigger variants. Each covers two connections, two reset episodes and 48 decisions; forced variants exercise policy miss/tail paths. Forced parameters are confined to tests; all emitted arms have `max_calls=0` and no forced trigger indices.
- Standard `exp.offline_search.closed_loop.ops.h100.control plan` exits 0 with all eight arms, 102 files and 22.231 GiB. Standard episode identity-map dependencies remain under `store/queries/...`; they are not replaced by a run-local store. The plan hashes/copies standard metadata as the owner-required deployment dependency; it does not run analysis or deserialize outcome trajectories. The local CPU test's filtered readers are absent from all emitted methods/arguments.

Detailed machine-readable evidence: `SELECTION.json`, `results/evaluation.json`, `results/diagnostics.json`, `results/plugin_selftests.json`, `results/unit_tests.log`, `results/standard_control_plan.log`, and the new run root's `h100_sync/plan.json`.
