# Round 2 selection plan — supersedes the initial round-2 plan

Started 2026-10-05 16:09 UTC after the fourth owner directive; target handoff by 17:39 UTC. CPU only, no online launches. GATES.md stays byte-identical: `188d3cc1d3beb807174c3bff84696542be72f00bcb66ac0cdc66cbc913a2d937`. Its numerical thresholds are still reported; its old offline-veto authorization is superseded by the owner. The image-use and own-library requirements remain mandatory.

Primary comparison for every reported candidate: symmetric light features plus the existing valid eight-dimensional state, same task-specific action-supervised AWM and early fit, same 16-NN kernel, cell kref 5 at size 50 / 8 at 200. No asymmetric candidate is eligible for the pilot. State-only is a negative control and ineligible.

Explore: (1) augmented visual encoder without a state shortcut inside the network, (2) the same with spatial token-map distillation, (3) token-supervised spatial-softmax keypoints, (4) augmented state-dropout student. Keep a shared serving interface and 128 output features. Inputs used by this bounded search are both current images, state and task; permitted history is not required.

Select architecture and training epoch using two deterministic within-library episode hold-outs per cell, never B-dev optimization. Fit student normalizers and AWM on training episodes only within each split. Teacher PCA coordinates are the fixed cell artifact's target coordinate system, but fold AWM and all student normalizers are fit without held-out episodes. Report that qualification. Fit the selected recipe afresh on the complete exact cell library. Library donor-LOEO precedes final B-dev assessment.

Blindness audit before selection for pilot: shuffle paired camera images across looks within task with three fixed seeds, and set both raw images to black, keeping state/task/step/history and the library index fixed. Report equal-task/episode RMS10 rise and per-task deltas. Before seeing audit results, operationalize 'barely rises' as less than 5% relative error rise for either the mean shuffle result or the zero-image result; require at least eight of ten tasks to worsen under shuffling. This new explicit image-use screen implements directive 3 and does not relax any frozen gate. Offline accuracy rankings cannot rescue a blind candidate.

Publish PILOT.md with exact artifact hashes and R15 outside-200 identities, two sizes x current/new x off/ir25 x 300 episodes. Audit reusable baseline identity from manifests/counts without selecting from online outcomes. Existing accuracy/cost/SR thresholds stay unchanged in the report; pilot permission is tracked separately.

## Control added at 16:32 UTC, before new-candidate B-dev assessment

Add `r1_style`, a fresh original-style global-target-normalization/state-conditioned/no-augmentation/no-dropout recipe, to the same two episode hold-outs and 40/80/120 epoch comparison. This separates the new regularization package from a retrained incumbent-style control using the same validation method. It uses the stable v2 module, so its state/task input concatenation and initialization are not byte-identical to the frozen round-1 checkpoint. It remains eligible only if image audits pass. Select from all five recipes by the already declared mean hold-out ratio and blindness rule; no new candidate B-dev metrics have been read when adding this control.
