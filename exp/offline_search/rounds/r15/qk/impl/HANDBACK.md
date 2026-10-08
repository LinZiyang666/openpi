# R15 QK final candidates active

Swapped both frozen `r15_qk_student_v2` / `state_drop_aug` students: L10-50 **40 epochs**, L10-200 **120 epochs**. Active fits use these jointly produced 128 features (two 64-value accessor blocks), with same-library main/early AWM, confidence, only-no-progress representation, LOEO correctors/distance scale and unchanged 16-NN/action/parent-row recipes. No serving-code change was needed. HIT runs zero towers; MISS one full policy; blind/policy-tail neither encoder.

Both astra manifest hashes match [PILOT.md](../astra/PILOT.md). Active artifact paths below are relative to `impl/`, hashes SHA256. Complete provenance/refit bindings: [50 manifest](artifacts/pi05_l10_50/manifest.json), [200 manifest](artifacts/pi05_l10_200/manifest.json). Original round-1 fits/reports/arrays are archived under `impl/round1/` and derived `r15_qk/impl/round1/`.

| Artifact | SHA256 |
|---|---|
| ../astra/round2/pilot/pi05_l10_50/student.ts | 7c493bf5d1708a7320593595ac6ebf7082d408aac4b5188299d764697a45faee |
| artifacts/pi05_l10_50/off.pkl | 1c6a280cdd9edbbc2b4e7d41d6df2b737b79a9127eff644bcad16e5a9a78c292 |
| artifacts/pi05_l10_50/ir25.pkl | bdf1859c9276bdfaede79a2af357b170f2e7c6ed3ea12da4a4ac003983dbbd2c |
| ../astra/round2/pilot/pi05_l10_200/student.ts | 4272d04dfcfa4929d2e9fed5efb393aa8ffd47f061d3799430d7c4ba0b059a00 |
| artifacts/pi05_l10_200/off.pkl | dea131640ca91cb8e0298c2dff4b1005a44088bca4fe76cbd943c30a8bf02844 |
| artifacts/pi05_l10_200/ir25.pkl | 8cb6cfd032c54ec8d9df04d59ac1f43247876c3b8e61fe30eff5c723c2946869 |

Regenerated roots/contracts: `/home/weiland/trace_runs/os_closed_loop/r15_qk_pilot_pi05_l10_50` (light/current × off/ir25), `r15_qk_pilot_pi05_l10_200` (light off/ir25). [Queue](queue_pilot.txt) retains six H100 lines, 300 B episodes/arm outside size 200; seed 2026101401, knob seed 0, caps 2.6125/2.0000. [Size-200 baseline reuse proof](baseline_equivalence.json) preserves descent off/ir25 identities, seeds, fits, arguments and templates. Reuse remains conditional on collecting all 300 accepted outcomes; no completion or SR claim. Policy RNG remains warmup/request-order dependent.

[Completion audit](completion_audit.json): six pytest contracts, both final-student [plugin replays](plugin_replay.json), [v2 parity](v2check_replay.json), all-task features/geometry/kernel/corrector/image/boundary checks and four blocked [fleet plans](plans/verification.json) PASS. Default teacher actions/decision semantics match the saved original. All-row donor-LOEO covers 2,925/11,437 rows; B-dev covers 3,198/2,289 looks (100/73 episodes). Encoder uses the full cell library; donor-LOEO is not encoder cross-fitting.

[Gate report](gate_report.json): **accuracy FAIL**, balanced corrected RMS10 ratios library 1.0775/1.0589, B-dev 1.1000/1.0891. Local key+task-code medians 1.073/1.119 ms; measured projected epsilon .015892/.016582, local cost FAIL. Artifacts retain astra's final frozen epsilon **.01522897037037037 / .01418825925925926**; `IR=(epsilon*Q+M)/N`, T=M, old-arm pricing unchanged. H100 timing and new closed-loop SR are unmeasured. Owner directive 4 permits the B pilot despite numerical failures.

[Deployment manifest](deployment_manifest.json) lists hashes for the nine required code files, complete conservative code/dependency inventory, both frozen exports, four active fits (source and H100-relocated served hashes), root definitions/contracts/queue and existing B-worker inputs/configs. Use H100 staged fit paths/hashes, not local unrelocated pickle hashes. Existing full-policy environment needs CPU TorchScript support; local Python 3.11.15, Torch 2.7.1+cu126, NumPy 1.26.4; one Torch/interop/OMP/BLAS thread, affinity intersecting CPUs 2–9,46–53. H100 was not inspected remotely.

Shared-file deployment while descent chains run: [DEPLOYMENT_SAFETY.md](DEPLOYMENT_SAFETY.md) identifies direct server `closed_loop/plugin.py`, shared planner/potential transitive server dependency `closed_loop/ops/h100/assets.py`, and collector `rounds/r04/k4_eval/cost_ledger.py`. All three have a [default-off source proof](descent_safety.json); 64 configured B descent arms omit the flag, and their fit hashes remain unchanged. All six QK serving files and these three shared files remain byte-identical to the first handoff. Use complete atomic file replacements; retain unchanged shared dependencies, descent roots and fits. Loaded Python modules are not reloaded by copying source; subsequent starts use the default-off equivalent path.

No deployments, launches, servers, GPU, remote operations, git, process killing or test-A data reads. Active serving requires no training checkpoint, exploratory metric, library RGB or derived evaluation arrays.
