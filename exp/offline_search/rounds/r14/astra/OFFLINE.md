# R14 offline partition evidence

Generated 2026-10-04T17:35:06.095796+00:00. All numerical tables are recomputed by scripts in this directory. No new variant has run closed loop. Action imitation loss is not SR, and fixed-state replay is not an environment simulation.

## What is actually task-indexed

The source audit covers `r10/recipe/recipe.py`, `fitting.py`, `build.py`, AWM, the inherited guard/blind/policy-tail implementations, `r11/knob/recipe.py`, and the serving plugin. `deployment_audit.json` hashes each actual recipe artifact.

* Retrieval scope restricts donor rows by query task. Both normal and early Mahalanobis transforms, including centering and standardization, are fitted per task.
* There is a separate task-specific continuity scale `T.s_c` in the fresh-after-MISS branch. It can affect actions when the normal policy tail falls back to a fresh look. `T.s_d` and projected visual centering affect confidence/diagnostics. With `guard_only`, no recovery or blending, those confidence changes cannot alter acceptance.
* The motion corrector has task-specific heads. Its task one-hot is constant within each head. The proposed pooled head removes that input to prevent an implicit task-conditioned correction model.
* Guard dictionaries `M0/M1` contain task-centred raw-camera means and `med_len` contains task median episode lengths. Motion/cosine thresholds themselves are shared. The deployed recipe disables stuck, terminal and overtime, leaving only no-progress, whose threshold is shared and whose inputs are retrieved-row progress and episode length. The row's progress/length are metadata, not learned task tables. Pooling guard dictionaries cannot change the active verdict under the deployed settings.
* Blind state scales and motion percentiles are task-indexed but their gates are inactive with budget-only anchor tails. Task identity checks on episode/anchor lifecycle remain necessary in every arm.
* Both camera PCAs and action sigma are fitted once per selected cell library. GC_dist uses one distance scale per cell/variant, and kref is cell-wide. Nothing is pooled across policies, suites or library sizes.

## Whole-episode library diagnostic

`partition.py` fits metrics on all training-fold rows and leaves every such row available as a donor. It evaluates 5 outer episode folds, with at most 16 deterministic evenly spaced anchors per episode for head fitting and evaluation. Head weights preserve the represented full-row mass. An inner episode holdout fits the gate scale without exposing the outer validation episodes. All failed library episodes remain. Rows and labels never come from A.

This is an economical diagnostic, not the production fitter: head input moments and ridge sufficient statistics use anchor quadrature; gate scale uses one inner holdout rather than the production complete fold schedule; early distances use explicit float32 codes rather than the deployment affine representation. The selected-library PCA is fixed across folds and thus sees heldout covariates, as in the prior design. `SPEC.md` requires full-row production fitting and the original distance calibration. The earlier train-LOEO scale experiment was superseded because its in-metric-fit query distances suppressed correction; its log is retained, but none of its tables enters this freeze.

Loss below is episode-balanced mean squared motion error against stored policy chunks, scaled by training action sigma. Signed percentages compare corrected loss with the same diagnostic baseline; positive is worse. Gripper errors, raw loss, unattenuated half-strength correction, per-episode records and fold manifests are in the CSV/JSON payload.

| cell | baseline loss | scope pooled | metric pooled | head pooled | scope+metric pooled | scope only task | all pooled |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 0.2546 | +0.8% | +5.7% | +1.2% | +6.5% | +8.6% | +9.8% |
| pi05_l10_200 | 0.1499 | -0.3% | +5.4% | +2.0% | +5.9% | +7.5% | +8.2% |
| pi05_l10_500 | 0.1209 | -0.4% | +3.2% | +3.1% | +3.8% | +7.0% | +7.9% |
| pi05_spatial_50 | 0.3279 | +2.0% | -9.0% | +0.8% | -6.7% | -7.3% | -0.8% |
| groot_l10_50 | 0.3633 | +0.4% | +4.5% | +2.9% | +4.9% | +9.2% | +9.6% |
| groot_l10_200 | 0.2497 | +0.3% | +4.5% | +0.8% | +4.9% | +6.3% | +7.0% |
| groot_l10_500 | 0.2046 | +0.2% | +4.6% | +0.1% | +4.9% | +5.2% | +5.7% |
| groot_spatial_50 | 0.3043 | +0.4% | -4.3% | +0.4% | -5.9% | -2.7% | -2.2% |

Corrector benefit is `mse_raw - mse_corrected`; head pooling is assessed at identical retrieval and gate scale. It is small relative to changing retrieval geometry in most cells. Library observations suggest retaining task geometry for the long-horizon suite, while pooling can help the small spatial libraries. This is not yet a closed-loop recommendation.

| cell | foreign top1: scope pooled | foreign top1: scope+metric pooled | S×M loss interaction |
| --- | --- | --- | --- |
| pi05_l10_50 | 1.12% | 1.88% | -0.06% |
| pi05_l10_200 | 0.75% | 0.91% | +0.77% |
| pi05_l10_500 | 0.49% | 0.76% | +0.98% |
| pi05_spatial_50 | 9.29% | 11.36% | +0.29% |
| groot_l10_50 | 0.75% | 1.75% | -0.05% |
| groot_l10_200 | 0.69% | 1.03% | +0.15% |
| groot_l10_500 | 0.62% | 1.30% | +0.09% |
| groot_spatial_50 | 0.50% | 5.00% | -1.95% |

“Foreign top1” measures task-identity agreement, not a ground-truth semantic retrieval score. Scope pooling often leaves task retrieval almost unchanged because the query's task metric already separates other tasks. The interaction is `(SM - S - M + base) / base`; it is an action-loss interaction. `scope_only` explicitly asks whether retaining only the retrieval boundary suffices after pooling learned geometry and correction.

## The additional continuity scale and guard null

`continuity.py` forces the fresh branch on library states, excludes the complete query episode from donors, and pools the tail-to-head continuity reference across tasks. This stress test conditions on the full fitted representation; it is not held-out metric performance and is not a forecast of fallback frequency. These are uncorrected action errors, isolating the scale's retrieval effect.

| cell | fresh top1 changed | raw action loss change |
| --- | --- | --- |
| pi05_l10_50 | 3.07% | +0.56% |
| pi05_l10_200 | 4.90% | +0.64% |
| pi05_l10_500 | 4.45% | +0.55% |
| pi05_spatial_50 | 2.86% | +0.44% |
| groot_l10_50 | 2.93% | +0.43% |
| groot_l10_200 | 4.37% | +0.23% |
| groot_l10_500 | 4.20% | +0.35% |
| groot_spatial_50 | 2.13% | +0.28% |

The guard-null test changes all inactive task guard/blind tables and their shared thresholds much more strongly than ordinary pooling, while exercising dense, gapped and stationary histories. It preserves actions, donor IDs and active force/reason/flag fields on 1440 proposals, including 400 active no-progress decisions. This accompanies the source-level dependency argument; real plugin lifecycle checks are still sol's responsibility. No GPU arm is allocated to guard-only pooling. Continuity pooling is different: it has a real fresh-branch action effect and gets its own arm.

## Recorded B cache states

`dev_extract.py` is adapted from the frozen earlier extractor with its outputs and read fence redirected here. It reconciles accepted attempts, B contracts, journal outcomes, complete decision ledgers and shard hashes. Reconstructing the actual corrected cache on recorded HITs gives maximum absolute action error 3.58e-07. `dev_partition.py` scores all recorded real looks with the same economical head/scale approximations described above.

| cell | proposal | shadow motion loss | actual deployed loss | guard disagreement on fixed states |
| --- | --- | --- | --- | --- |
| pi05_l10_50 | base | 0.2490 | 0.2357 | 0.11% |
| pi05_l10_50 | scope | 0.2404 | 0.2357 | 0.83% |
| pi05_l10_50 | metric | 0.2434 | 0.2357 | 13.97% |
| pi05_l10_50 | head | 0.2510 | 0.2357 | 0.11% |
| pi05_l10_50 | all_pool | 0.2468 | 0.2357 | 14.13% |
| pi05_l10_200 | base | unlabelled | unlabelled | 0.08% |
| pi05_l10_200 | scope | unlabelled | unlabelled | 0.08% |
| pi05_l10_200 | metric | unlabelled | unlabelled | 11.56% |
| pi05_l10_200 | head | unlabelled | unlabelled | 0.08% |
| pi05_l10_200 | all_pool | unlabelled | unlabelled | 11.55% |
| pi05_spatial_50 | base | unlabelled | unlabelled | 0.00% |
| pi05_spatial_50 | scope | unlabelled | unlabelled | 1.28% |
| pi05_spatial_50 | metric | unlabelled | unlabelled | 8.01% |
| pi05_spatial_50 | head | unlabelled | unlabelled | 0.00% |
| pi05_spatial_50 | all_pool | unlabelled | unlabelled | 8.83% |
| groot_l10_50 | base | 0.3338 | 0.3205 | 0.04% |
| groot_l10_50 | scope | 0.3344 | 0.3205 | 0.48% |
| groot_l10_50 | metric | 0.3004 | 0.3205 | 21.64% |
| groot_l10_50 | head | 0.3339 | 0.3205 | 0.04% |
| groot_l10_50 | all_pool | 0.3069 | 0.3205 | 21.66% |
| groot_spatial_50 | base | unlabelled | unlabelled | 0.00% |
| groot_spatial_50 | scope | unlabelled | unlabelled | 1.08% |
| groot_spatial_50 | metric | unlabelled | unlabelled | 8.69% |
| groot_spatial_50 | head | unlabelled | unlabelled | 0.00% |
| groot_spatial_50 | all_pool | unlabelled | unlabelled | 8.65% |

The diagnostic baseline does not exactly reproduce the deployed full-row head; the actual deployed shadow loss is shown to make that gap visible. Compare pooled diagnostic proposals with their diagnostic baseline, not with the deployed column as if only one switch had changed. The tiny baseline guard disagreement also reflects prototype numerical differences. Guard comparisons hold the recorded state/look sequence fixed and omit feedback through actions and blind-veto timing; they are not IR predictions.

Shadow policy labels are available here only in the two small long-horizon cells. Spatial and middle-size recorded states still support retrieval/guard diagnostics but not a policy-action error claim. The requested shadow recordings and queued pilot fill that gap after the reservation is released. `task_descriptives.csv` reports old per-task off/periodic SR and IR without fitting to them; those methods spend different budgets.

## Task budget allocation

Priority uses only cross-fitted library cache/policy motion gaps, averaged within episode before log compression, then averaged per task and shrunk by the number of episodes. The two available task-rank checks against independent B shadow states are pi05_l10_50=0.624, groot_l10_50=0.648. This supports limited transfer of ranking, not a causal claim that an extra call will improve success. Neither SR, task success labels nor test outcomes set a priority.

The final shared intercept uses all eligible B-dev off/periodic paths to match reference spend; the final replay is therefore development evidence. An earlier identity-holdout fit/check is retained separately in `allocation_split_checks.json`: budget band passes 20/30, and paired matching passes 21/30. Some sparse splits omit an entire task. These failures motivated retaining the final full-cohort refit as development calibration, not relabeling those splits as passed validation. Gain and shrinkage were not increased to rescue a weak result. Library identity-holdout checks fit both priorities and the intercept on the calibration identities; their input error tables themselves come from whole-episode cross-fitting on the selected library.

| cell | dose | budget-band checks | paired-budget checks | worst \|ΔIR\| | min changed placement |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 0.32 | 5/5 | 5/5 | 0.0022 | 0.272 |
| pi05_l10_50 | 0.4 | 5/5 | 5/5 | 0.0074 | 0.120 |
| pi05_l10_200 | 0.32 | 2/2 | 2/2 | 0.0038 | 0.391 |
| pi05_l10_200 | 0.4 | 2/2 | 2/2 | 0.0067 | 0.124 |
| pi05_l10_500 | 0.32 | 1/1 | 1/1 | 0.0018 | 0.385 |
| pi05_l10_500 | 0.4 | 1/1 | 1/1 | 0.0028 | 0.152 |
| pi05_spatial_50 | 0.32 | 4/4 | 3/4 | 0.0211 | 0.371 |
| pi05_spatial_50 | 0.4 | 3/4 | 3/4 | 0.0134 | 0.093 |
| groot_l10_50 | 0.32 | 5/5 | 5/5 | 0.0048 | 0.260 |
| groot_l10_50 | 0.4 | 5/5 | 5/5 | 0.0055 | 0.098 |
| groot_l10_200 | 0.32 | 1/1 | 1/1 | 0.0020 | 0.424 |
| groot_l10_200 | 0.4 | 1/1 | 1/1 | 0.0035 | 0.185 |
| groot_l10_500 | 0.32 | 1/1 | 1/1 | 0.0001 | 0.453 |
| groot_l10_500 | 0.4 | 1/1 | 1/1 | 0.0014 | 0.219 |
| groot_spatial_50 | 0.32 | 4/4 | 4/4 | 0.0018 | 0.315 |
| groot_spatial_50 | 0.4 | 4/4 | 4/4 | 0.0071 | 0.064 |

There are 46 rows in this combined table: full B-development replay where available plus a separate library identity-holdout diagnostic per cell/dose. In particular, the small pi05 spatial library has a heldout-library budget/matching warning. The full-cohort fits do not erase it. `allocation_checks.json` retains each source and both realized IR values.

`allocation_robustness.json` bootstraps task episodes with the scarce-sample cap while keeping the final bias fixed. It measures fragility, not an independent validation interval. `allocation_tasks.csv` retains every task priority/cap; no task effect is selected from SR. Missing tasks in a diagnostic split have null task rates, not estimated zero-call rates. Full fit tasks all have library evidence. Task-level changes in knob-call fraction are recorded relative to the periodic reference on the same states, so existing task guard/rate differences do not masquerade as allocation engagement.

At the large library size there is no B complement, so both utility and budget calibration use only the verified heldout-library tables. State distribution, episode length and guard-rate shifts remain unvalidated. Endpoints are explicitly tested: cap zero calls at every real look, a cap larger than an episode adds no calls, and mandatory guards always survive. No cost debt or learned state crosses episodes.

## Reproduction and limits

Run only the commands in `RUN.md`. Provenance files list every data read; the fence blocks test roots, query stores, reserved validation files, subprocesses and out-of-directory writes. No GPU, policy query, remote command or rollout was used. `controller_check.json` records 6604 proposals and 6604 identical retries against actual R11 periodic query logic with its lower-layer query stubbed, including extra looks. This checks schedule arithmetic, not the production plugin.

The production full-row pooled fits, real serving parity, full B pilot and all test SR/IR effects remain future work. This deliverable freezes how to perform them, not their outcome.
