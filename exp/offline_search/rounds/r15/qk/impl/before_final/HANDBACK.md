# R15 QK implementation handback

Round-1 starter complete: default-off `--os-light-key`, frozen CPU `cnn_action_cpu` (80 epochs), exact-cell main/early AWM, 16-NN and layers 1–3 refits, unchanged action payload/parent rows and periodic caps. HIT runs zero towers, MISS one full policy, blind/cache/policy-tail neither encoder. The causal look interface exposes both images, state, prior looks/features, executed actions, previous policy outputs, task/instruction/step and HIT history. B-only light recording is supported; pilot recording is off.

Actual-work collector: Q/T/M/N, T=M, `IR_new=(epsilon*Q+M)/N`, epsilon .0140/.0144, latency and donor/guard/cap audit. Old formula is preserved; logical looks and nominal `5*N` controls are separate from tower work and actual terminal controls.

Paths below are relative to `impl/artifacts/`; SHA256 bindings and complete refits/provenance: [50 manifest](artifacts/pi05_l10_50/manifest.json), [200 manifest](artifacts/pi05_l10_200/manifest.json). Arrays: `/home/weiland/trace_runs/offline_search_store/derived/r15_qk/impl/pi05_l10_{50,200}/`.

| Artifact | SHA256 |
|---|---|
| pi05_l10_50/student.ts | d2a667d07d93aaf28fb3e0be77f3dd29b8555ccdf0fb8b207eea5513394cae90 |
| pi05_l10_50/off.pkl | f2e0116e0b392e6aed20bf14762e812b30f60f5429f5a592d864bd50cecc08af |
| pi05_l10_50/ir25.pkl | 1f34ec267d552fe6b780a2f9e6ce5b8cc4657c5d2f9d020fd044f4e8871d368f |
| pi05_l10_200/student.ts | b18cc15b9e41f291a67ba1c7a27c6ef0487e9763217b2907a0c00f8444c0c7cc |
| pi05_l10_200/off.pkl | 2619fc60c784fa0ea64e23d8612ea61e20bf004a6bb6e081e3391c3f67dc39b0 |
| pi05_l10_200/ir25.pkl | 23e60a8100a58ddb0fdd87ea3defd4d197b4ca62e9e49b9a90d1ec380d5d36a8 |

Roots: `/home/weiland/trace_runs/os_closed_loop/r15_qk_pilot_pi05_l10_50` (light/current × off/ir25), `r15_qk_pilot_pi05_l10_200` (light off/ir25). [Queue](queue_pilot.txt): six H100 lines, 300 B episodes/arm, descent identities outside size 200. Reuse descent `r15d_pi05_l10_200_{off,periodic_ir25}`: [proof](baseline_equivalence.json) verifies identical manifest bytes/identities, seeds, fits, arguments and templates. Process seed 2026101401, knob seed 0; policy RNG remains warmup/request-order dependent.

[Verification](completion_audit.json): six pytest contracts; poisoned-tower plugin replay; duplicate/reset/failed-receipt/policy-tail/preflight and single-broadcast checks; default actions/audited decision fields byte-equal; all-task parity, sigma, metric/kernel/correction/parent-map, stable ties, image sensitivity and fit-boundary checks. Four [fleet plans](plans/verification.json) pass with remote/process/server operations blocked. All-row donor-LOEO: 2,925/11,437 rows; accepted unseen B-dev: 3,198/2,289 looks, 100/73 episodes. Encoder uses the full cell library: donor-LOEO is not encoder cross-fitting.

[Gate report](gate_report.json): **accuracy FAIL**, B-dev RMS10 ratios 1.1329/1.1168; LOEO 1.0042/1.0302. Local key+task-code medians 1.064/1.262 ms, projected epsilon .01577/.01870: local cost FAIL; H100 cost unmeasured. Requested frozen coefficients stay .0140/.0144. These are recorded-trajectory proposals, with no new SR. Owner directive 4 permits the B pilot despite numerical failures.

Swap after astra's final [PILOT.md](../astra/PILOT.md) freeze: run below, then repeat with `--size 200`. It validates hashes/subset/epochs, binds authorized affinity, archives round 1 and refits by config. Interim v2 size-50 Tensor `[N,128]` already passed an isolated refit/parity/plugin [replay](v2check_replay.json). Final costs and size-200 selection are pending; active roots remain round 1.

```bash
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r15.qk.impl.swap --size 50 --refit
```

Regenerate with that CPU prefix: `impl.prepare`, `impl.verify`, pytest (`-p no:cacheprovider`, external basetemp), `impl.evaluate --size {50,200} --mode {loeo,dev}` (four combinations), `impl.gates`, `impl.plans`, `impl.audit`, `impl.deployment`. `impl` abbreviates `exp.offline_search.rounds.r15.qk.impl`.

Deploy [inventory](deployment_manifest.json): every fleet Python source/payload hash, source/relocated served hashes and root seals. New serving files: `impl/{__init__,encoder,recipe,runtime,recording,accounting}.py`. Shared edits: `closed_loop/plugin.py`, `closed_loop/ops/h100/assets.py`, `rounds/r04/k4_eval/cost_ledger.py`; [diff](shared_changes.diff) and before/after hashes included. Plans stage exports, fits, existing payload tables and contracts.

H100 needs CPU TorchScript in its existing full-policy environment; no new package expected. Local: Python 3.11.15, Torch 2.7.1+cu126 (CUDA uninitialized), NumPy 1.26.4. Coordinator must check compatible CPU export loading/forward, affinity intersecting CPUs 2–9,46–53, one Torch/interop and OMP/BLAS thread. No training checkpoint/code, library RGB or evaluation arrays needed for serving. H100 environment was not inspected remotely.

No launches, servers, GPU, deployment, remote operations, git or test-A reads. [Interim handback](HANDBACK_INTERIM.md) retained.
