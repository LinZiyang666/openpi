# Discovery evidence and reproduction

## Population and provenance

- Source: `/home/weiland/trace_runs/os_closed_loop/r08_main`.
- 70 arms × 300 accepted episodes = **21,000 arm-episodes; 952,842 decisions**.
- Four IP arms: **26,687 fresh randomized anchors**, all with actual p=.25 and matching recorded treatment/coin.
- Independent policy-draw augmentation joins on **29,757 discovery decisions** across 70 arms.
- Eight pure-cache training populations: **52,741 fresh observations, 2,400 episodes**. Per-cell counts below.
- All selected pairs are tasks 0–9 × original/subset init 0–29. The UID subset identity is authoritative, consistent with the debug schema. No split-selection or evaluation includes inits 30–49.
- Historical source files can contain holdout records. The stream filter checks identity before admitting accepted outcomes; NPZ joins retain only selected IDs. No holdout summary is created.
- Every compact output has an adjacent JSON with source stat fingerprint, arm-spec hash, extractor source hash, augmentation metadata hashes, accepted counts and the split string. `results/extract.json` inventories them. These are immutable-run provenance checks, not a cryptographic rehash of the full 278 GB collection.
- No original debug-reader derived cache is written or used. No image decompression, simulator restoration, GPU execution, remote access or closed-loop launch was needed.

`derived/r09_astra/compact/*.npz` contains normalized valid action prefixes, state, keys, randomization and the per-pair success/cost/decision arrays. It is approximately 1.4 GB including student evaluation arrays; the owned research directory is approximately 1 GB, mainly ready-to-load fitted methods.

## Estimands and conventions

1. **Cost:** pooled owner cost divided by the number of five-control decisions, including terminal rounding. Use summed numerator/denominator for controller mixtures. Pure P10 is nominally .5, but its logged ratios here are .504–.511.
2. **Motion:** mean squared error in normalized channels 0–5 over the first ten controls of a fresh anchor. Gripper sign disagreement is a separate measure. Synthesis and student tables average decisions within each episode, then the 300 episodes equally. No physical units or success probabilities are inferred from these MSEs.
3. **Episode pairing:** task/init pairs are resampled within each of ten fixed tasks. Policy seeds are not replicated. Paired bootstrap intervals are exploratory.
4. **Learning:** `init % 5` defines five folds across *all* paths. Each student is trained only on pure-cache fresh states from the other four folds. All outcomes, successes and failures contribute equally by episode; outcome labels do not train the student. Transfer paths use the matching init fold's model, so matching starts cannot leak through a different arm.
5. **Call effect:** first supported visit to a pre-coin score half, then the original IP future controller. Centered HT scores use other-fold task success as baseline; sum within an episode before bootstrapping. All-anchor sums estimate an infinitesimal probability-shift derivative, not SR under a replacement controller. A policy shadow is an offline diagnostic moderator, not an available-free online variable.
6. **Router:** only whole episodes are selected. OOF values estimate the training procedure; full-data values evaluate a fitted rule on its own discovery data. Neither is the untouched holdout value of the final frozen rule.

## Exact discovery baselines

| Cell | Cache SR | Cache IR | P10 SR | P10 IR | Fresh training anchors |
|---|---:|---:|---:|---:|---:|
| π0.5 long-50 | .716667 | .076414 | .906667 | .503990 | 10,071 |
| π0.5 long-500 | .856667 | .076591 | .906667 | .503990 | 8,744 |
| π0.5 Spatial-50 | .800000 | .077452 | .990000 | .511180 | 3,895 |
| π0.5 Spatial-500 | .966667 | .078119 | .990000 | .511180 | 3,299 |
| GR00T long-50 | .636667 | .074303 | .893333 | .503887 | 10,669 |
| GR00T long-500 | .826667 | .074453 | .893333 | .503887 | 8,875 |
| GR00T Spatial-50 | .880000 | .075286 | .936667 | .511191 | 3,835 |
| GR00T Spatial-500 | .966667 | .075451 | .936667 | .511191 | 3,353 |

These differ from the brief's historical/full-500 references because the population and policy realization differ. They are not contradictory re-estimates of the same sample.

## Supervision versus architecture

`student.py` fits per-task ridge residuals, alpha=100. Inputs are 207-dimensional, standardized using training data only and clipped at eight standard deviations. Nonlinear features are 384 fixed cosines, with seed 20261001. The half-residual candidate retains all gripper and padding coordinates and ten-control cadence. No classifier, segmentation, task outcome, contact or object-pose feature is used.

| Cell | Linear full correction | Nonlinear full correction | Nonlinear half correction | Extra-shadow NN half teacher blend |
|---|---:|---:|---:|---:|
| π0.5 long-50 | −53.6% | −59.4% | −45.3% | −46.5% |
| π0.5 long-500 | −14.7% | −20.1% | −18.5% | −12.4% |
| π0.5 Spatial-50 | −58.7% | −64.6% | −48.4% | −45.2% |
| π0.5 Spatial-500 | −12.0% | −15.8% | −16.3% | −3.9% |
| GR00T long-50 | −61.3% | −65.4% | −50.3% | −52.6% |
| GR00T long-500 | −7.5% | −10.0% | −12.8% | −10.3% |
| GR00T Spatial-50 | −44.9% | −47.3% | −37.2% | −38.5% |
| GR00T Spatial-500 | −8.0% | −8.6% | −12.0% | −5.6% |

All entries are relative changes in motion MSE on held-out-in-fold cache paths. The memory baseline uses the same shadow-label budget, a frozen library Mahalanobis metric, and 16 neighbors from other inits. Its gripper remains the original cache gripper, like the student. This matched-label baseline prevents attributing the effect solely to the nonlinear head.

Transfer evidence is in `results/student_audit.json`: half corrections reduce MSE on CU trajectories by 8.8%–24.6% in all eight cells, and on IP trajectories by 30.3%–35.0% in all four available cells. The CU controller's guard and fresh-after-call paths have a different state distribution; transfer improvement is useful but does not cover the corrected controller's future distribution.

The W10 transfer entries in that file use **full-camera shadow retrieval on states reached by W10**. They do not demonstrate a wrist-input student or justify charging wrist-only cost to this head.

Negative controls in the same file:

- `zero_motion` and `task_teacher_mean` lose heavily.
- A per-task scalar motion shrinker helps some stalled cache paths but increases CU error by 19.1% in GR00T long-50 and 6.2% in π0.5 Spatial-50.
- Removing visual inputs from the half-residual head still helps, but less on sparse π0.5: long-50 and Spatial-50 improve more with visual inputs. This uses state **and the retrieved action**, so it is not evidence that perception can be omitted.
- Improvements remain on successful episodes, the first thirty controls, and teacher motion RMS > .1. These subsets are diagnostic; success is not a deployable gate.

Inference artifacts contain ten sets of numeric coefficients, approximately 4.3 MB compressed per cell. Real-library checks time the NumPy head at approximately .06 ms median; full query timing includes repeated PCA projection and retrieval and is approximately 2–3 ms on this CPU. This is not an end-to-end H100 latency measurement.

## Retrieval and synthesis

`horizon_metric.py` reconstructs library features from frozen full-rank metric codes (maximum inverse reconstruction error <5e-13), refits the within-action-neighbor metric using ten-step labels, and reretrieves on the same recorded states. PCA, candidate library, k=16 and each cell's kernel reference count stay fixed. Early-state fitting still uses library steps <=2.

| Cell | Ten-step metric MSE change | All-success-neighbor filter | Motion medoid | Top-1 |
|---|---:|---:|---:|---:|
| π0.5 long-50 | −17.36% | ~0 | +20.8% | +32.7% |
| π0.5 long-500 | −0.07% | +5.7% | +30.6% | +55.2% |
| π0.5 Spatial-50 | −11.80% | ~0 | +21.2% | +27.0% |
| π0.5 Spatial-500 | −0.41% | +6.2% | +33.6% | +72.1% |
| GR00T long-50 | −8.91% | ~0 | +21.7% | +42.2% |
| GR00T long-500 | −0.56% | +7.5% | +22.6% | +68.4% |
| GR00T Spatial-50 | −4.44% | ~0 | +25.7% | +37.3% |
| GR00T Spatial-500 | −0.52% | +5.6% | +29.8% | +86.5% |

The successful-neighbor filter renormalizes the original sixteen neighbors, falling back to original weights if none succeeds. It is not a reretrieval over all successful rows; the negative result is scoped accordingly.

GR00T's replayed mixture agrees with the live cache to <=4.8e-7. π0.5 shadow-based replay is not bit-identical: worst absolute differences are .076–.094; aggregate teacher MSE changes from the reconstruction alone are <=.025%. We expose this mismatch and use the live cache as the synthesis reference. Student training uses shadow retrieval consistently. The zero-blend **online method** separately passes exact identity against its frozen base on eighty real library queries.

Other tested choices—top-4/8, halved/squared kernel weights, episode-balanced weights, gripper majority vote and a 1.15 norm expansion—are all saved in `results/synthesis.json`. None gives a compelling broad improvement. Norm expansion here is in raw normalized motion coordinates; it is not a claim about every possible sigma-scaled cap.

## Offline success estimates that are legitimate, and those that are not

The router solves a fractional linear program by Dinkelbach iterations, minimizing total expected cost / expected decisions subject to a shrunk task-average SR-loss constraint. `test_ratio_not_average` guards the episode-length accounting. Three fixed menus, three shrinkage strengths and three tolerances yield **216** sensitivity runs. The main reported rule is binary P10/cache, prior=12, loss allowance=.01; it was not selected by maximizing the reported cross-validation SR.

The frozen π0.5 long-50 task policy probabilities are `[1,1,0,0,1,1,1,1,.14953271028037407,1]`. Its OOF point is **.915184 @ .379761**. Conditional frozen-fold ΔSR interval is [−.016012,+.034464]. Refitting all fold LPs in each of 1,000 task-stratified bootstraps gives the much less reassuring percentile sensitivity interval **[−.037025,+.018043]**, with substantial IR instability [.2491,.4319]. The asymmetry and difference from the point estimate reflect unstable discrete selection and finite-sample bootstrap behavior; this is not calibrated post-selection confidence.

The call analysis output keeps all first-entry and derivative contrasts, including adverse ones. For relative distance, local high-minus-low derivatives are:

| Cell | Derivative | Pointwise bootstrap interval |
|---|---:|---|
| π0.5 long-50 | −.728 | [−1.421, −.056] |
| π0.5 Spatial-50 | +.383 | [.081, .692] |
| GR00T long-50 | +.414 | [−.339, 1.140] |
| GR00T Spatial-50 | +.315 | [−.003, .655] |

Units are SR change per unit infinitesimal probability tilt, not percentage points obtained by deploying that tilt. Formal support exists only for the logged interior coin. The approximate family-width diagnostic in `evidence.json` is explicitly descriptive; no causal subgroup is promoted on it.

The shadow proxy fails a direct design check. At A's blind states a fresh look improves motion MSE 12.0% / 5.6% in π0.5 / GR00T long-50. The actual every-five controller loses 6.33 / 6.67 pp SR against A, with paired intervals excluding zero. This is why `REPORT.md` leaves new student/metric SR estimates unfilled rather than extrapolating an empirical error-to-SR slope.

## Tests and commands

Run from `/home/weiland/projects/openpi`:

```bash
R9PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
R9MOD=exp.offline_search.rounds.r09.explore_astra.tools
"${R9PY[@]}" -m "$R9MOD.extract" --workers 6
"${R9PY[@]}" -m "$R9MOD.routing"
"${R9PY[@]}" -m "$R9MOD.shadows"
"${R9PY[@]}" -m "$R9MOD.student"
"${R9PY[@]}" -m "$R9MOD.student_audit"
"${R9PY[@]}" -m "$R9MOD.shadow_memory"
"${R9PY[@]}" -m "$R9MOD.horizon_metric"
"${R9PY[@]}" -m "$R9MOD.call_value"
"${R9PY[@]}" -m "$R9MOD.routing_uncertainty"
"${R9PY[@]}" -m "$R9MOD.evidence"
"${R9PY[@]}" -m "$R9MOD.prepare_confirmation"
"${R9PY[@]}" -m unittest "$R9MOD.test_tools" "$R9MOD.test_deployment" -v
```

The same sequence is `bash tools/reproduce.sh`. Extraction resumes existing compact files; use `--overwrite` when intentionally changing the capture generation/extractor. All other analyses overwrite only owned outputs. Reports themselves are authored interpretation and are not regenerated by the script.

**11 tests pass** (`tests.log`). They cover discovery rejection, conflicting attempts, identity joins/missing augmentation, padded dimensions, failure-filter fallback, cost ratio, paired resampling, LP behavior, regression recovery, episode-constant lottery assignments, and real fitted metric query/tail execution. Additional `prepare_confirmation` checks cover zero-blend identity, gripper/padding invariance, corrected blind tails and clone isolation on ten real library queries in each of eight cells.

A first attempt to use scikit-learn found it absent. The head was implemented with NumPy/SciPy weighted normal equations; no environment or package installation was performed. Raw extraction took about three minutes with six one-thread workers; each fitted student cell took seconds. The bootstrap router is the slowest small-data computation. Exact job logs are retained alongside this document.
