# Offline evidence

Generated UTC: 2026-10-03T17:01:37.140060+00:00. All numerical tables were produced by this directory's executed code. Evidence is B-library or allowlisted B-dev only. These are action/retrieval/ledger diagnostics, not measured SR of look-half.

## Data and validation

Four model/suite cells, nested selected libraries of 50, 200 and 500 episodes. All failed library episodes remain. Five episode folds exclude held-out episodes from action sigma, metrics, donor sets, correction-head training and atlas-label fitting. The existing R10 PCA files are verified against selected-row hashes and remain fixed across folds; this is transductive unsupervised B-library representation, not fully held-out PCA. The alternative representations use the same convention. No test-A recordings, queries store, or other R12 explorer data were accessed.

Retrieval screens use even query steps at least two, modelling ten-control commits after hits; they do not simulate policy-induced state changes. Atlas screens begin at step four and discard the initial look pair; deployment exposure is consequently smaller. Equal episode weights prevent failed/long episodes dominating row means. State/action values and library successes are observations, not official-test labels.

Correction diagnostics refit heads inside each outer fold, with LOEO donor synthesis inside its training library. They use noninitial training rows and training-LOEO distance scales. Production R10 heads include their initial-row branch and five-fold distance calibration. Thus these are controlled integration diagnostics, not exact end-to-end R10 replay. The specification calls for the exact production calibration before serving.

Validation checked **235,064** donor-row cases for task and episode-fold exclusion, finite actions and normalized weights. Main full metric distances differ from the R10 equation by at most **3.33786011e-06**. **11** pure planner checks passed. The planner arithmetic averaged **114.4 microseconds** in a local CPU loop; this is not end-to-end retrieval latency. CPU affinity and read audit are in validation.json. GPU was not used because its driver was unavailable.

Camera mapping was checked in library metadata and source: pi05 img0 is base_0_rgb and img1 is left_wrist_0_rgb; GR00T's LIBERO builder orders video.image then video.wrist_image. R7 replay explicitly poisoned key_v0 on wrist requests. The actual current pi05 deferred encoder uses a zero placeholder until completion, which is another reason availability must be tracked explicitly rather than inferred from finite arrays.

## Retrieval and representation screen

Values are episode-balanced normalized motion MSE over ten controls and six channels. Lower is better. “Stale v0” simply substitutes a two-slot-old key into the current full metric. “Refit memory” fits the correct delayed representation; “transport” adds a neighbour-predicted v0 innovation before full-metric retrieval.

| cell / library size | rows | full | wrist | third-person | wrist L2 | stale v0 | refit memory | transport |
|---|---|---|---|---|---|---|---|---|
| groot_l10_200 | 5691 | 0.2483 | 0.2653 | 0.2717 | 0.2912 | 0.3865 | 0.2490 | 0.2615 |
| groot_l10_50 | 1436 | 0.3534 | 0.3792 | 0.3829 | 0.4203 | 0.4783 | 0.3691 | 0.3725 |
| groot_l10_500 | 14421 | 0.2091 | 0.2272 | 0.2422 | 0.2538 | 0.3718 | 0.2150 | 0.2215 |
| groot_spatial_200 | 2151 | 0.2290 | 0.2503 | 0.2529 | 0.2856 | 0.4990 | 0.2394 | 0.2505 |
| groot_spatial_50 | 563 | 0.3229 | 0.3316 | 0.3413 | 0.3683 | 0.5351 | 0.3365 | 0.3559 |
| groot_spatial_500 | 5483 | 0.1995 | 0.2234 | 0.2199 | 0.2510 | 0.4845 | 0.2064 | 0.2159 |
| pi05_l10_200 | 5566 | 0.1561 | 0.1728 | 0.1965 | 0.1897 | 0.2954 | 0.1637 | 0.1651 |
| pi05_l10_50 | 1425 | 0.2565 | 0.2791 | 0.3224 | 0.3110 | 0.3875 | 0.2662 | 0.2796 |
| pi05_l10_500 | 14344 | 0.1292 | 0.1440 | 0.1721 | 0.1556 | 0.2766 | 0.1366 | 0.1370 |
| pi05_spatial_200 | 2063 | 0.1976 | 0.2129 | 0.2435 | 0.2346 | 0.3746 | 0.2051 | 0.2114 |
| pi05_spatial_50 | 541 | 0.3602 | 0.3657 | 0.3868 | 0.3772 | 0.4974 | 0.3698 | 0.3798 |
| pi05_spatial_500 | 5082 | 0.1575 | 0.1706 | 0.2028 | 0.1853 | 0.3516 | 0.1642 | 0.1688 |


Unconditional wrist increases error in every screened cell. Pure third-person search is usually worse and its price is unmeasured. Naive stale-key substitution is especially poor; age-four substitution is worse still. A Schur-complement distance, minimizing the full metric over the missing view, nearly reproduces the independently refitted wrist metric. Both use the same action-pair covariance; marginalizing does not restore missing camera information. Euclidean wrist distance does not rescue it.

The useful direction is treating camera memory as a different representation and including observed displacement. Initial-only camera memory and longer refresh gaps were screened in history/*/summary.json; they did not justify dropping the short age contract. Raw state history expressed as a displacement was more useful than appending another absolute-state block. This is an empirical regularization effect, not additional physical information.

Alternative single-wrist features on size-50 libraries:

| cell / size | token mean | 2x2 pool | PCA128 | token max | 8x8 sketch | RGB | RGB + PCA64 |
|---|---|---|---|---|---|---|---|
| groot_l10_50 | 0.3909 | 0.3851 | 0.3943 | 0.3958 | 0.3805 | unavailable | unavailable |
| groot_spatial_50 | 0.3304 | 0.3298 | 0.3309 | 0.3382 | 0.3158 | unavailable | unavailable |
| pi05_l10_50 | 0.2938 | 0.2802 | 0.2800 | 0.2843 | 0.2901 | 0.3989 | 0.3072 |
| pi05_spatial_50 | 0.3787 | 0.3765 | 0.3808 | 0.3709 | 0.3617 | 0.4330 | 0.3943 |


The 8x8 representation mean-pools tokens spatially after a fixed signed channel sketch; it retains finer spatial information than the original 4x4 keys. It helps the small GR00T spatial cell in this screen, but does not win consistently and has no measured one-camera deployment price. RGB features use downsampled wrist pixels; they fail to improve the main candidates. Larger PCA rank also does not consistently help. These representations are not selected for the primary.

## Correction and the fair full-information control

| cell / size | full | wrist | memory | wrist + displacement | memory + displacement | full + displacement |
|---|---|---|---|---|---|---|
| groot_l10_200 | 0.2395 | 0.2569 | 0.2407 | 0.2311 | 0.2260 | 0.2233 |
| groot_l10_50 | 0.3464 | 0.3727 | 0.3646 | 0.3462 | 0.3463 | 0.3311 |
| groot_l10_500 | 0.2008 | 0.2182 | 0.2067 | 0.1916 | 0.1898 | 0.1841 |
| groot_spatial_200 | 0.2236 | 0.2412 | 0.2336 | 0.2178 | 0.2207 | 0.2115 |
| groot_spatial_50 | 0.3228 | 0.3276 | 0.3365 | 0.3087 | 0.3173 | 0.3087 |
| groot_spatial_500 | 0.1918 | 0.2133 | 0.1984 | 0.1880 | 0.1843 | 0.1786 |
| pi05_l10_200 | 0.1477 | 0.1647 | 0.1556 | 0.1549 | 0.1516 | 0.1449 |
| pi05_l10_50 | 0.2532 | 0.2756 | 0.2629 | 0.2521 | 0.2476 | 0.2391 |
| pi05_l10_500 | 0.1210 | 0.1358 | 0.1283 | 0.1286 | 0.1250 | 0.1183 |
| pi05_spatial_200 | 0.1870 | 0.2019 | 0.1953 | 0.1926 | 0.1905 | 0.1844 |
| pi05_spatial_50 | 0.3591 | 0.3640 | 0.3689 | 0.3512 | 0.3587 | 0.3435 |
| pi05_spatial_500 | 0.1449 | 0.1584 | 0.1518 | 0.1525 | 0.1491 | 0.1426 |


The half head must match its actual input. Copying old camera data into the original correction head is worse than fitting a matching head (`correction/*/summary.json`). Displacement also improves full vision: the final column is the necessary control for any claimed representational improvement. Memory-plus-displacement can approach or beat the old full cache on some cells while remaining worse than full vision with the same motion information. That is why the experiment grid includes both controls.

## Direct camera-safety segmentation

The primary atlas is trained from raw full/half action agreement, gripper agreement and progress-comparison agreement. At runtime the previous full donor mixture predicts safety two slots ahead; current state displacement checks whether the robot followed that prediction. No current image is used to choose the camera. Fixed label RMS .15, safe mass .75, support .99 and state residual .2 are B-screened choices, not test-A fits. `atlas_reference.py` contains the pure prelook function; SPEC.md gives the complete production recipe.

The following table applies the frozen atlas causally along held-out B policy trajectories, forcing a full look after each half. It substitutes the separately fitted corrected half action into the unchanged corrected-full branch. The bootstrap interval resamples whole episodes and is conditional on the fitted models; it is not an SR interval or a correction for selecting the proposal on B data.

This replay includes the camera-score and camera-age rules only. It does **not** simulate the runtime pending-no-progress veto, periodic-due veto, policy-tail lifecycle, new guard calls or changed trajectories. Thus the half fraction is an offline exposure opportunity, not a forecast of the complete serving state machine. Gripper disagreement averages sign mismatches over the ten controls, including zero mismatches on full looks; it is not the fraction of episodes harmed.

| cell / size | half looks % | corrected MSE change % | MSE delta | episode bootstrap interval | grip disagreement % | missed stall % | extra stall % |
|---|---|---|---|---|---|---|---|
| groot_l10_200 | 16.5 | -0.51 | -0.0012 | [-0.0019, -0.0005] | 0.10 | 1.30 | 0.94 |
| groot_l10_50 | 20.1 | 0.12 | 0.0004 | [-0.0028, 0.0037] | 0.29 | 1.26 | 1.01 |
| groot_l10_500 | 15.6 | -0.37 | -0.0007 | [-0.0011, -0.0004] | 0.05 | 1.38 | 0.87 |
| groot_spatial_200 | 25.4 | 0.19 | 0.0004 | [-0.0009, 0.0019] | 0.07 | 0.83 | 0.68 |
| groot_spatial_50 | 28.3 | -0.05 | -0.0002 | [-0.0039, 0.0038] | 0.48 | 1.43 | 0.44 |
| groot_spatial_500 | 22.2 | -0.30 | -0.0006 | [-0.0012, -0.0000] | 0.03 | 0.67 | 0.26 |
| pi05_l10_200 | 30.7 | 0.18 | 0.0003 | [-0.0003, 0.0009] | 0.13 | 1.73 | 0.71 |
| pi05_l10_50 | 28.1 | 0.12 | 0.0003 | [-0.0030, 0.0032] | 0.39 | 1.43 | 0.97 |
| pi05_l10_500 | 30.6 | 0.57 | 0.0007 | [0.0003, 0.0013] | 0.11 | 1.65 | 0.91 |
| pi05_spatial_200 | 35.0 | 0.36 | 0.0007 | [-0.0005, 0.0020] | 0.16 | 0.44 | 0.30 |
| pi05_spatial_50 | 28.2 | 0.73 | 0.0027 | [-0.0033, 0.0077] | 0.44 | 0.86 | 0.85 |
| pi05_spatial_500 | 37.1 | 1.07 | 0.0016 | [0.0009, 0.0023] | 0.14 | 0.38 | 0.32 |


“Missed stall” means the half top donor advances while the full top donor does not, relative to the previous full donor; “extra stall” is the converse. These are single-comparison diagnostics, not measured additional policy calls. Adding progress agreement to the atlas labels reduces many such errors versus action-only labels, but does not remove them. **The no-progress guard is the main unresolved system risk.**

The after-wrist gate in gates_dynamic.json was also explored. It has access to the current wrist proposal and can abstain based on corrected-action/progress disagreement, but needs a new transactional second-camera fallback. It remains secondary. Its measured feature gate must not be described as an existing plugin capability.

## Owner IR accounting and B dev limits

Let N be requested five-control slots, V all looks, H actual half looks, C half-to-policy completions, and M policy calls. At fixed trajectories/calls, pi05 current request-path savings against stock full vision are:

`S = (0.152-0.1656)*V/N + (0.1656-0.0646)*H/N - 0.0502*C/N`.

If full looks truly retain stock speed, `S = 0.0874*H/N - 0.0502*C/N`. Additional calls cost at least their policy computation; recompute the complete ledger rather than subtracting this fixed-trajectory formula after behavior changes. With no completions the current path needs half share above **0.1347** just to cover full-look dispatch overhead. Prices are supplied local-4090 proportional-latency measurements, not hardware-independent FLOPs.

The all-half ceiling on a baseline with calls is `.0874*v - .0502*m`. Restricting half to no-call looks gives the more operational ceiling `.0874*(v-m)`. These ceilings ignore warm-up and safety rejection and cannot be read as achievable controller savings.

| B dev arm | baseline owner IR | V/N | M/N | all-half upper saving | noncall-only upper saving | legacy eligible/look | valid-tail eligible/look |
|---|---|---|---|---|---|---|---|
| r11_dev_pi05_l10_200 | 0.1494 | 0.5040 | 0.0859 | 0.0397 | 0.0365 | 0.4438 | 0.4274 |
| r11_dev_pi05_spatial_200 | 0.1092 | 0.5137 | 0.0366 | 0.0431 | 0.0417 | 0.4563 | 0.4171 |
| r11_devknob_pi05_l10_50_off | 0.1630 | 0.5037 | 0.1020 | 0.0389 | 0.0351 | 0.4243 | 0.4109 |
| r11_devknob_pi05_l10_50_periodic_ir32 | 0.3116 | 0.5039 | 0.2772 | 0.0301 | 0.0198 | 0.0903 | 0.3646 |
| r11_devknob_pi05_spatial_50_off | 0.1640 | 0.5090 | 0.1021 | 0.0394 | 0.0356 | 0.4151 | 0.3781 |
| r11_devknob_pi05_spatial_50_periodic_ir32 | 0.3095 | 0.5071 | 0.2740 | 0.0306 | 0.0204 | 0.0368 | 0.3614 |


All selected B-dev journals include both accepted successes and accepted failed episodes. Their N/V/M counts reconcile exactly with summaries. Owner-basis IR was recomputed; some stored stage-price IR fields use a different basis. Existing dev recordings lack query keys, so they support cost/opportunity bounds but cannot evaluate this camera policy on cache-driven states.

The legacy camera override discards many opportunities after a policy tail. A narrow, explicitly validated camera-memory path recovers some without altering the tail action lifecycle; blanket waiver of lifecycle errors is prohibited. Prediction scenarios use the new valid-tail column. No size-500 B-dev baseline exists in the allowlisted roots; its forecasts are stated scenarios. GR00T half/completion prices and a one-camera path remain unmeasured/unvalidated.

## Reproduction and scope

All commands run from repository root with:

```sh
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Append `-m exp.offline_search.rounds.r12.astra.<module>` for inventory, retrieval, correction, representations, history, atlas, dev_cost, assess_atlas, check, make_reports. Also run `correction --dynamic`, `gates`, `gates --dynamic`, and `assess_atlas --guard`. `representations` defaults to size 50. Run independent cell fits with their bounded process pools; all BLAS stays single-threaded inside the assigned CPU set. No server/worker/sync/test run is launched by any of these scripts.

`make_reports` regenerates numerical prose and predictions; it must not be rerun after receiving design closed-loop outcomes as if that were the original preregistration. PREDICTION.md is frozen by FINAL_MANIFEST.json. Curated current libraries were inventoried, not analysed with these candidate fits. Complete closed-loop requests, pilot/grid, unresolved questions and implementation work are in REQUESTS.md, EXPERIMENTS.md, SPEC.md and HANDBACK.md.

`fit_selected_atlas` builds the final selected-library safety tables and main metric references under final_atlas/. These are library-only ingredients for sol, not deployable full-system artifacts. [Forecast sensitivity figure](forecast.png) visualizes explicitly assumed buyback economics; it is not an experimental SR plot.
