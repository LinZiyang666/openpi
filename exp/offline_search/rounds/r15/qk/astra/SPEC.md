# R15-B frozen experimental specification: spatial light query key

Status: **offline exploration only; not approved for deployment or a closed-loop pilot**. The prototype package does not include a refitted five-layer runtime. `GATES.md` controls promotion; the coordinator must approve any online work after those offline gates pass. `OWNER_ADDENDUM.md` is part of this contract. This specification is frozen with the final handback SHA256 manifest before any new-method closed-loop outcome.

## Decision and candidate identity

The primary experimental build is **symmetric action-aware spatial CNN**, `symmetric_cnn_action_cpu`, independently trained for each cell. Both library and query use the same frozen student. It retains the two camera streams and spatial positions. The asymmetric comparison uses the identical student query but the original cell's offline teacher keys, `asymmetric_cnn_action_cpu`. Run this comparison offline; do not add another online arm by default.

The small-library B-dev comparisons currently do not justify replacing the tower. The primary is a concrete implementation target for further offline improvement, not a claim that its accuracy gate passed. Ridge distillation is a fast CPU integration/debug control. State-only, direct action query keys, local kernel residuals and global-image representations are not promoted.

## Input and train recipe

One artifact per `(model, suite, library size, exact episode subset)`. Train only on complete rows of that cell's build-cache library; verify `tok/rows.npy`, parent row IDs, episode IDs and original B init identities. No cross-size warm start, extra library episodes, B-dev training examples, test-A data, external pretrained weights, or synthetic policy labels from outside the library. Stored tower tokens/keys and that cell's library-only PCA/AWM are permitted distillation targets. A matching R10 artifact may be reused only after its subset and provenance are verified.

The current implementation is π0.5 only. Input: two RGB uint8 224×224 images, in agentview/wrist order, nearest sampling at `4*j` to 56×56; normalized valid robot state `rs[:8]`; task ID 0…9. RGB uses the fixed scale `pixel / 127.5 - 1` at the convolution input; learned state and output normalizers come only from the selected library. Do not substitute the 56-pixel recording for training data. No random crops, flips or phase/time labels are used. Each query is single-frame; the ordinary controller history remains outside the network.

Student, exactly as `neural.TinySpatial`:

1. Shared camera convolution: `3→16`, 5×5, stride 2, padding 2; SiLU; `16→24`, 3×3, stride 2, padding 1; SiLU; `24→32`, 3×3, stride 2, padding 1; SiLU. Each camera retains a 7×7 grid. No global pooling.
2. Concatenate the two flattened grids (3,136 values), eight library-standardized state values clipped to ±8, and a learned eight-dimensional task embedding.
3. Linear 3,152→256, SiLU, linear 256→128. These outputs predict the two 64-dimensional teacher PCA blocks after undoing library target normalization. A 128→35 action auxiliary head is training-only in the deployed export. Prototype parameter count including the auxiliary head: 856,299.
4. The 128 outputs are feature blocks, not calibrated original camera tokens. Their names must identify student features. Do not fabricate 32,768-dimensional vision keys or pretend a teacher tower ran.

CPU training is the default and is feasible. Pin CPUs 10–21,54–65; use four Torch/BLAS threads per training process, at most 24 total across concurrent work. Fresh initialization, seed 20261005, 80 epochs, shuffled batches of 96, AdamW learning rate .001 and weight decay .001, gradient norm cap 5. No early stopping on B-dev. Refuse a final artifact that stopped before the declared epoch count. The earlier capped GPU experiments remain labeled historical; future GPU use requires documented CPU infeasibility, not a speed preference.

Loss, implemented in `neural.py`:

- MSE of the 128 standardized teacher PCA coordinates.
- MSE in the main teacher AWM code: map PCA prediction error through the task's vision block of `Wf`, then divide by that task's mean teacher code energy.
- Weight .1 on smooth-L1 error for the sigma-normalized first 5×7 action block.
- Weight .5 on an action-pair hinge. For each query, sample one of its three closest action heads from a **different episode of the same task**. Sample a same-task negative and keep it only if it is from a different episode and its mean squared action distance exceeds the positive's by .05. Against the fixed teacher PCA features of those positive/negative library rows, in teacher AWM geometry including exact state, enforce normalized squared-distance margin .1. No own-episode positive or negative enters this term.

The metric-only distillation ablation omits the hinge and retains the other losses. Library-dependent means, sigma, positives and code-energy scales must be stored with provenance. The fixed recipe was selected using existing B-dev, so this is development reuse, not an independently validated choice.

## Fitting the five layers

| Layer | Required handling |
|---|---|
| 1: retrieval/synthesis | Freeze the trained student, encode **all selected library rows**, concatenate its two 64-D blocks and exact eight-dimensional state. Refit per-task AWM means/stds and metric, including the step≤2 early branch, using unchanged cross-episode nearest-action-pair recipe (`nn=3`, `lambda=.1`). Keep 16 donors, the cell's existing kernel reference (`kref=5` at size 50 and `kref=8` at size 200 in these R10 artifacts), stable row tie-breaking, action payloads and 10-control commit semantics. All donor exclusions happen before sorting and distance reference computation. |
| 2: only-no-progress | Preserve the existing progress-span calculation and threshold, anchor steps, donor progress/episode length, no memo reset after MISS, and blind veto. Replace the feature accessor so a real light look is a fresh observation. Disabled stuck/terminal/overtime branches must not force a tower merely to compute unused diagnostics. Diagnostics that need original tokens are explicitly unavailable; do not invent them. If retaining V7 diagnostics, refit their representation-dependent components on the same library and keep them out of verdict logic. |
| 3: LOEO corrector | Refit all heads and distance attenuation scale on the new frozen feature representation using the unchanged R10 library-only LOEO/outer-fold recipe. Own episode excluded from every donor lookup. Features remain student-128 + state-8 + synthesized normalized 10×7 chunk + capped step + task one-hot. Preserve zero correction at step 0, six motion channels, blend .5 and the `.75/2.0` distance-ratio ramp. Do not reuse old heads/scales just because shapes agree. |
| 4: periodic knob | Preserve each baseline arm's **actual fixed gap caps**, lifecycle, counters and policy dose for the first paired comparison. `ir25` is its legacy knob label, not a claim that new achieved IR equals .25. Change only the ledger. Distance/disagreement/error knobs are outside this pilot and would require fresh library-only calibration. Repricing the caps to spend savings on extra policy calls is a separate experiment. |
| 5: task descent | Retain the existing task table for a matched-setting comparison. A fresh descent is later work under the frozen R14 first-failure protocol, with new B-dev outcomes. Do not transfer a newly selected table to A before freezing it. |

For the asymmetric diagnostic, the student predicts original teacher PCA coordinates and retrieves the original offline AWM library codes; do not silently refit those code coordinates. Query-dependent confidence scales, corrector heads and distance calibration still need new library pseudoqueries with the light query path. Symmetric and asymmetric artifacts must have different representation IDs.

As in the existing R10 recipe, the final representation is held fixed during donor-LOEO corrector fitting. This conditions on an encoder trained on the query episode and can be optimistic. Record this explicitly. An additional episode-cross-fitted encoder diagnostic is useful before claiming robustness; it must remain library-only and cannot replace unseen B-dev evidence. Do not relabel the existing donor-LOEO check as fully held-out learning.

## Serving interface and tower bypass

Add an opt-in light-key capability before `_ConnPolicy._osp_infer` falls through to the full policy. Reuse the existing CPU state/output adapter and per-connection transaction/lock. The flow is:

1. Validate observation, execution receipt, episode identity and duplicate request exactly as today.
2. Serve a legal blind cache/policy-tail continuation without either encoder.
3. On a real look, use both wire images and CPU state normalization to compute student features. Run retrieval, only-no-progress guard, correction and frozen knob against a feature-aware query view.
4. On HIT, apply existing output transforms and commit once. **Zero calls** to the policy stage-1/tower, stage-2 or stage-3 functions.
5. On MISS, execute one complete policy call on the retained current observation, including its tower. There is no reusable teacher prefix from the light look. Commit that policy chunk once; invalidate cache anchor and preserve ordinary policy-tail rules.

Do not route the MISS back through a second student search or commit a provisional cache action. Exact forced-policy paths may bypass the student in a later explicitly measured optimization; the initial cost projection conservatively charges it at every look. No periodic teacher refresh, high-uncertainty tower probe without a policy call, or hidden fallback is part of this design.

Introduce explicit `light_features[2,64]`, `has_look`, `hist_has_look` and representation identity in the query API. Old `hist_has_vision` serves as a documented compatibility alias for observation-anchor validity only where needed; cost accounting uses separate actual-work fields. Teacher-token access remains unavailable on light HITs. Update the AWM projection, corrector feature accessor and guard feature access consistently; changing only `_dist` is insufficient.

## Artifact and verification contract

Bundle: model state and architecture/recipe version; library/subset/row/input hashes; library-only target and state normalizers; encoder train log/device/seed/epoch count; student library features; metric and early metric; confidence scales; corrector heads/scales; guard metadata; parent action payload identity; fixed knob caps; row-space mapping; parity and gate reports. `feature_kind`, dimension, camera order, resize and state normalization are checked on load. Reject missing RGB, wrong subset, cross-cell reuse and partial training.

Prototype checkpoint files and fitted symmetric metrics exist under `pi05_l10_{50,200}/`; they are **not** a deployable R10 artifact. Sol builds the remaining runtime/refits in its authorized directory. `REQUESTS.md` specifies verification and ledger requirements. No GR00T model may be trained until exact build-library frames are recovered and verified.

## Economic B-dev pilot plan

Only after all offline gates pass and coordinator approval: π0.5 L10-50 first, one student configuration, baseline/student pairs at **off** and the identical **ir25 fixed caps**, 100 paired B episodes per arm (10 per task), same host and seeds. Choose identities from the complement of the size-200 subset so a later L10-200 comparison remains unseen. This is 400 episodes for the first cell; no pure-inference arm is needed because the comparison is against the current system at matched settings.

Run L10-200 only if the first cell passes its development gates. Reuse a baseline only with exact episode/seed/host/runtime/cap/artifact identity; otherwise collect the paired baseline. Extend passing cells to 30 B episodes/task for the confirmatory development check. Spatial requires all-look image+policy labels first; MISS-only data cannot clear its offline gate. GR00T and size 500 are outside the initial pilot. No runs, servers, chains, remote operations or A emission are requested from this exploration process.
