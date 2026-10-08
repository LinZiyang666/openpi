# DIRECTIVE_2 offline evidence

Generated UTC: 2026-10-03T18:28:25.598035+00:00. All numbers below come from executed scripts in this directory. This is post-freeze development evidence, entirely B-library/B-dev. No changed-controller SR was measured. Earlier library/token/representation/correction evidence remains in [the original OFFLINE](directive2/prior_delivery/OFFLINE.md); its latency-priced economics are superseded here.

## Data and baseline reconstruction

Read with the official `look_recording.read_looks` hash/B-contract checks: `r12_devlooks_50L`, `r12_devlooks_400`, now-complete `r12_devlooks_knob`, and the completed local replacement roots represented below. There are 700 accepted episodes and 17,367 actual looks across these cells. Failed episodes are retained. Each recording contract excludes the served B subset; no queries store or A output is opened.

**Source correction:** the coordinator's 13:0x note, discovered during final handback review, retired `r12_devlooks_50S` and `r12_devlooks_200`. All current tables exclude them. Their earlier screens are superseded historical artifacts only. `r12_devlooks_50S_local` is the replacement; `r12_devlooks_200_local` is included only if complete and validated. Method/thresholds were not changed after this source correction. The 18:22 prediction's Spatial/200 rows were explicitly withdrawn in an append-only correction; their replacements below are post-freeze validation, not a silent rewrite. Current authority is the `VALID_ROOTS` allowlist; it blocks new source reads from the retired roots.

`dev_validation.py` reconstructs the deployed full ranking, kernel weights, correction and guard inputs; source counts reconcile to accepted journals and summaries. The current numerical replay uses the **actual recorded full donor rows and weights** for memory, preserving source behavior when tiny ranking differences occur. Independently reconstructed full-path parity is:

| cell / knob | rank mismatches | max served-HIT action difference |
| --- | --- | --- |
| GR00T l10-400 / off | 0 | 0 |
| π0.5 l10-400 / off | 0 | 0 |
| GR00T l10-50 / off | 0 | 0 |
| π0.5 l10-50 / off | 0 | 0 |
| GR00T spatial-50 / off | 0 | 0 |
| π0.5 spatial-50 / off | 0 | 0 |
| GR00T l10-50 / periodic .32 | 0 | 0 |
| π0.5 l10-50 / periodic .32 | 0 | 0 |


Forced-full replay exactly reproduces stored full proposal actions, no-progress spans, calls, and look timing. Periodic due status is independently reconstructed from pre-look histories and the actual keyed cap, including guard-suppressed due events. The 24 poisoning checks reproduce complete half proposals without any current missing-camera feature. `directive2/validation.json` records checks, source hashes and CPU affinity.

## Why the frozen atlas is not enough

Repricing changes the economics, not the guard errors. At the new prices, the original pre-look atlas gives:

| cell / knob | frozen atlas half looks | IR saved, new price | missed / extra calls | grip-disagreement looks |
| --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 366 | 0.003957 | 43 / 35 | 4 |
| π0.5 l10-400 / off | 762 | 0.009768 | 62 / 56 | 25 |
| GR00T l10-50 / off | 480 | 0.004220 | 80 / 56 | 28 |
| π0.5 l10-50 / off | 725 | 0.007610 | 71 / 48 | 38 |
| GR00T spatial-50 / off | 301 | 0.008446 | 14 / 12 | 8 |
| π0.5 spatial-50 / off | 333 | 0.009599 | 19 / 16 | 18 |
| GR00T l10-50 / periodic .32 | 121 | 0.003029 | 19 / 24 | 6 |
| π0.5 l10-50 / periodic .32 | 245 | 0.006596 | 26 / 29 | 5 |


These missed calls can make a naive replay ledger look cheaper. That is not evidence of preserved SR. Later full looks can also inherit a changed progress reference from an earlier half decision. Small action error alone was therefore an inadequate acceptance rule.

## Current-observation acceptance and explicit abstention

The revised method uses a current-wrist-only witness and a full-metric witness with a library-transported estimate of the missing view. It requires corrected action, gripper and progress agreement before committing. Rejection pays the other half of vision and restores a full-information decision; this is additive under DIRECTIVE_2.

All retrieval metrics, safety rows and heads are selected-library-only. Structural selection and the `.1`/`.05` comparison occurred **after the frozen atlas had been evaluated on B dev**, so they are expressly post-freeze dev calibration. The raw candidate artifacts separate legal proposal inputs from hidden-view evaluation. Full/half correction heads and the five-fold H scale are in `dev_validation/fits/`; wrist-only sensor fits are in `directive2/fits/`.

| cell / knob | N / looks / calls | baseline SR / IR | half looks | fixed-call IR saved | missed / extra calls | grip-disagreement looks |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 5722 / 2880 / 736 | 0.870 / 0.184081 | 69 | 0.000854 | 3 / 0 | 0 |
| π0.5 l10-400 / off | 5182 / 2615 / 325 | 0.970 / 0.129888 | 243 | 0.003491 | 5 / 2 | 0 |
| GR00T l10-50 / off | 6734 / 3384 / 1085 | 0.740 / 0.211650 | 72 | 0.000736 | 5 / 0 | 3 |
| π0.5 l10-50 / off | 6352 / 3198 / 754 | 0.800 / 0.177186 | 194 | 0.002309 | 2 / 1 | 1 |
| GR00T spatial-50 / off | 2497 / 1270 / 210 | 0.840 / 0.146928 | 63 | 0.001867 | 0 / 0 | 0 |
| π0.5 spatial-50 / off | 2391 / 1221 / 189 | 0.880 / 0.144652 | 80 | 0.002543 | 0 / 0 | 0 |
| GR00T l10-50 / periodic .32 | 2883 / 1450 / 820 | 0.880 / 0.316767 | 8 | 0.000205 | 0 / 0 | 0 |
| π0.5 l10-50 / periodic .32 | 2673 / 1349 / 723 | 0.940 / 0.306080 | 55 | 0.001564 | 0 / 0 | 0 |


Saving holds the baseline denominator, look schedule and calls fixed and charges completion on baseline calls. The propagated missed/extra counts hold the recorded states/look slots fixed while recomputing donor progress and periodic histories. Once actions or timing diverge, these are counterfactual diagnostics, **not realizable changed-controller episodes or SR estimates**. Recorded execution metadata is likewise conditional on the full baseline; a serving adapter must use its own accepted history.

The stricter `.05` alternative has the following exposure:

| cell / knob | strict half looks | IR saved | missed / extra / grip |
| --- | --- | --- | --- |
| GR00T l10-400 / off | 4 | 0.000052 | 0 / 0 / 0 |
| π0.5 l10-400 / off | 22 | 0.000323 | 0 / 0 / 0 |
| GR00T l10-50 / off | 1 | 0.000011 | 0 / 0 / 0 |
| π0.5 l10-50 / off | 20 | 0.000239 | 0 / 0 / 0 |
| GR00T spatial-50 / off | 4 | 0.000119 | 0 / 0 / 0 |
| π0.5 spatial-50 / off | 9 | 0.000286 | 0 / 0 / 0 |
| GR00T l10-50 / periodic .32 | 0 | 0.000000 | 0 / 0 / 0 |
| π0.5 l10-50 / periodic .32 | 2 | 0.000057 | 0 / 0 / 0 |


Zero observed call/gripper discrepancies in this small exposure does not certify lossless SR. Even motion-only changes can alter later states.

## Uncertainty, split and policy shadows

The following intervals bootstrap whole episodes within task, with fits/gate fixed. They do not account for development selection and are not SR intervals. The split takes the first half of sorted B init IDs within each task as calibration, the remainder as evaluation. Earlier atlas work had already inspected these recordings, so neither half is described as pristine confirmation. Periodic recording groups are smaller.

| cell / knob | fixed-call saving 95% interval | saving charging extra calls, no missed-call credit | calibration H / missed / extra / grip | evaluation H / missed / extra / grip |
| --- | --- | --- | --- | --- |
| GR00T l10-400 / off | [0.000678, 0.001063] | 0.000854 | 30 / 0 / 0 / 0 | 39 / 3 / 0 / 0 |
| π0.5 l10-400 / off | [0.003080, 0.003922] | 0.003163 | 128 / 3 / 1 / 0 | 115 / 2 / 1 / 0 |
| GR00T l10-50 / off | [0.000574, 0.000908] | 0.000736 | 36 / 2 / 0 / 1 | 36 / 3 / 0 / 2 |
| π0.5 l10-50 / off | [0.001969, 0.002683] | 0.002176 | 99 / 1 / 0 / 0 | 95 / 1 / 1 / 1 |
| GR00T spatial-50 / off | [0.001475, 0.002308] | 0.001867 | 29 / 0 / 0 / 0 | 34 / 0 / 0 / 0 |
| π0.5 spatial-50 / off | [0.002143, 0.002960] | 0.002543 | 40 / 0 / 0 / 0 | 40 / 0 / 0 / 0 |
| GR00T l10-50 / periodic .32 | [0.000102, 0.000314] | 0.000205 | 3 / 0 / 0 / 0 | 5 / 0 / 0 / 0 |
| π0.5 l10-50 / periodic .32 | [0.001207, 0.001935] | 0.001564 | 23 / 0 / 0 / 0 | 32 / 0 / 0 / 0 |


The extra-call column subtracts policy computation for observed additional calls while assigning **no saving to missed calls**. It still cannot predict actual total IR after trajectories change.

Only L10-50 off has a policy shadow at every look. The proposal motion MSE difference over ten controls/six standardized motion channels is:

| cell / knob | proposal MSE delta vs full [95% episode bootstrap] |
| --- | --- |
| GR00T l10-50 / off | -0.00002689 [-0.00018749, 0.00013461] |
| π0.5 l10-50 / off | 0.00002652 [-0.00022438, 0.00027790] |


These differences are close to zero; the single stochastic shadow sample is not optimal-action truth. Its MSE cannot be converted into a predicted success rate. No policy-shadow error is invented for the other roots.

## Additional method screens

`d2_screen.py` measures 33 configurations per cell: frozen/strict pre-look gates, wrist-only witness, two witnesses, more individual-donor missing-view hypotheses, exact-top agreement, reduced atlas mass, and tighter action tolerances. Broad ensembles largely eliminate acceptance; relaxing the old atlas mass adds exposure but brings back guard/gripper errors. Detailed rows are in each `screen.json`, not silently discarded.

External-camera-only retrieval and causal camera choice were separately fitted for L10-50/400 in both models. The alternative saves the last genuine wrist instead of the external view, fits matching metrics/heads and a five-fold scale, and chooses its first camera solely from the two safety masses on the previous full donor mixture. It never chooses after inspecting both current views. At the same `.1` gate:

| cell / knob | camera method | half looks | IR saved | missed / extra / grip |
| --- | --- | --- | --- | --- |
| GR00T l10-400 / off | external | 47 | 0.000595 | 3 / 2 / 0 |
| GR00T l10-400 / off | prelook_choice | 71 | 0.000892 | 3 / 0 / 0 |
| π0.5 l10-400 / off | external | 167 | 0.002332 | 9 / 5 / 1 |
| π0.5 l10-400 / off | prelook_choice | 252 | 0.003579 | 9 / 4 / 0 |
| GR00T l10-50 / off | external | 47 | 0.000505 | 3 / 1 / 2 |
| GR00T l10-50 / off | prelook_choice | 80 | 0.000835 | 4 / 0 / 3 |
| π0.5 l10-50 / off | external | 122 | 0.001460 | 1 / 4 / 1 |
| π0.5 l10-50 / off | prelook_choice | 201 | 0.002405 | 2 / 1 / 2 |


The small coverage gains do not resolve the guard risk, so wrist-first remains the pilot proposal. The alternate fits and all outcomes are retained in `alternate.pkl`, `camera_choice.json` and the candidate NPZ files.

## GR00T numerical mismatch

Vision-only offline computation used the required interpreter, bounded allocation, and the deployed GR00T torch/FA2 dependency stack. No model server or rollout was launched. Peak reserved allocation was 2684 MiB. The first attempts exposed missing/incompatible FA2 dependencies in the workspace stack; the final run used the matching installed packages together without modifying either environment. Logs retain the failures and successful run.

On 160 paired reconstructed B observations, maximum token delta was 5.08203125; mean token RMS delta 0.04096293; mean pooled-key RMS delta 0.01037304. Projected wrist delta RMS was 0.06853035 at size 50 and 0.06998462 at size 400. These are own executed measurements, not copied latency-prep values.

The images were upsampled from 56px and deterministically cropped/resized. This isolates batch-one versus batch-two numerical differences on identical inputs, but cannot reconstruct original input detail. The selected GR00T libraries lack original RGB. We therefore did **not** re-encode the actual library or claim these are exact single-camera keys for the recorded states.

The eight largest measured projected-norm perturbations, both signs, were applied to legal current-wrist inputs. Requiring the candidate and acceptance to remain stable gives:

| GR00T L10 cell | proposals stressed | top changed under stress | robust half looks | IR saved | missed / extra / grip |
| --- | --- | --- | --- | --- | --- |
| GR00T l10-400 / off | 93 | 13 | 28 | 0.000336 | 2 / 0 / 0 |
| GR00T l10-50 / off | 86 | 5 | 35 | 0.000374 | 1 / 0 / 1 |
| GR00T l10-50 / periodic .32 | 25 | 1 | 1 | 0.000026 | 0 / 0 / 0 |


This filter is post-freeze dev numerical calibration, not a certified perturbation bound. It shows the risk reaches retrieval and eligibility; it is insufficient for a deployment claim. The spatial checkpoint has no corresponding numerical fit in this exploration. A genuine single-camera library/query study remains required there too. Completed separate-camera policy prefixes also require validation; the current replay assumes the original full fallback and cannot establish its actual numerical equivalence.

## Knob evidence and compute-share accounting

The full-controller off/periodic roots do not share all their B initial states. Comparing only their actual intersection gives:

| model | common B pairs / tasks | off SR → knob SR | off IR → knob IR | SR change 95% interval, pp | chord pp/IR [95%] |
| --- | --- | --- | --- | --- | --- |
| pi05_l10 | 9 / 7 | 0.778 → 0.889 | 0.150832 → 0.311708 | [-11.11, 33.33] | 69.07 [-60.91, 242.17] |
| groot_l10 | 9 / 7 | 0.889 → 1.000 | 0.167670 → 0.304586 | [0.0, 22.22] | 81.15 [0.00, 195.84] |


This overlap is too small and incomplete across tasks to identify a reliable local buyback slope. Unpaired aggregate SR differences between the roots are not a paired knob response. The forecast uses explicit sensitivity slopes, not a fitted slope from these sparse intersections.

With c=.076/.074, work is `2c*F+c*H+c*C+(1-2c)*M`; a rejected half acquisition plus completion is one full look. At fixed states/calls, saving is `c*(H-C)/N`. A policy call costs one in either camera plan. There is no request-overhead penalty or latency credit. With look rate .5 and no calls, all-half ceilings are 0.038000/0.037000; alternating ceilings are 0.019000/0.018500 for π0.5/GR00T. The observed-cell ceilings below ignore safety, warmup and future behavior:

| cell / knob | all-half ceiling | alternating ceiling |
| --- | --- | --- |
| GR00T l10-400 / off | 0.027727 | 0.018623 |
| π0.5 l10-400 / off | 0.033585 | 0.019176 |
| GR00T l10-50 / off | 0.025264 | 0.018593 |
| π0.5 l10-50 / off | 0.029242 | 0.019132 |
| GR00T spatial-50 / off | 0.031414 | 0.018819 |
| π0.5 spatial-50 / off | 0.032803 | 0.019405 |
| GR00T l10-50 / periodic .32 | 0.016171 | 0.016171 |
| π0.5 l10-50 / periodic .32 | 0.017799 | 0.017799 |


## Reproduction and provenance

Run from repository root with `taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r12.astra.<module>`. Use `CUDA_VISIBLE_DEVICES=` for CPU scripts. No script starts a server, worker, evaluation or sync. The common audit hook rejects foreign writes, the other R12 explorer directory, non-library store paths and nonallowlisted closed-loop roots.

Order: `dev_validation extract`/`fit` for the relevant root; `d2_consensus --roots ...`; `d2_screen`; `d2_camera_choice`; `d2_groot_mismatch`; `d2_stress`; `d2_validate`; `d2_evidence`. The GPU script only loads the vision tower/projector and caps tensors below the user limit. Inspect `nvidia-smi` before running it. `d2_documents` writes reports and appends a prediction once; do not rerun it as an original preregistration after receiving changed-controller outcomes. All raw fits, NPZ arrays, JSON summaries, read audits and logs stay in this directory.

Size 400 replaces the earlier size-500 dev ambition because size 500 has no nonserved B init in the existing pool. Earlier size-500 library replay remains historical evidence, not a new closed-loop validation. No new per-cell SR estimate is fabricated for unrecorded cells.
