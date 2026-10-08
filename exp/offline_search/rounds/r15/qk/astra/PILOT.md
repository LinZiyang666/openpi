# R15-B round-2 frozen pilot handoff

Owner directive 4 authorizes this B-dev pilot even when the existing offline gates fail. GATES.md stays byte-identical; its numerical SR/accuracy/cost standards are unchanged. This document supersedes the old offline-veto language and 100-episode-first pilot in the round-1 SPEC/REQUESTS. Astra has launched nothing. Sol completes the same-library runtime refits and integration verification; the coordinator executes the authorized pilot.

## Chosen artifacts

| Cell | Recipe | Epochs | Library hold-out ratio | B-dev ratio | Shuffle / zero error rise | One-thread CPU key ms |
|---|---|---:|---:|---:|---|---:|
| pi05_l10_50 | state_drop_aug | 40 | 1.0863 | 1.0983 | 169.0% / 117.5% | 1.028 |
| pi05_l10_200 | state_drop_aug | 120 | 1.0850 | 1.0802 | 217.5% / 131.9% | 0.958 |

Recipe and epoch were selected using two within-library episode hold-outs, before the selected model's B-dev evaluation. The original checkpoint is a separately audited incumbent. No test-A data, GPU, current-look teacher token, or B-dev training label was used. All candidates have the same symmetric AWM comparison form. See PROTOTYPE2.md and gate_report.json for failures as well as improvements.

### pi05_l10_50

Artifact directory: `/home/weiland/projects/openpi/exp/offline_search/rounds/r15/qk/astra/round2/pilot/pi05_l10_50`. Manifest SHA256: `ec6ce9c06ea277e0fad4a0ba1b313cc18802a4aadb31c3390403da7202f24bc8`.

| File | SHA256 |
|---|---|
| `student.ts` | `7c493bf5d1708a7320593595ac6ebf7082d408aac4b5188299d764697a45faee` |
| `student.pt` | `7696176a70c84085ed9aa573a950dd30fb6d73a2cbb478450d80665d056eab30` |
| `library_features.npz` | `d57ca5c55945497ebf4b2dcf91120c819c06921f57a287e9ef2b649d306a75ee` |
| `metric.pkl` | `5da8271cac4be7df62c7ec60e18f4b466768df9565afc7cf46f3c66acf8a8c8d` |
| `latency.json` | `804647ae1337b90e8f7f3f34224a0671902668e91148103d3aa62b4846ec0ed1` |
| `encoder_config.json` | `9e2f5fc8646bb399df6c496f2901b57f948f804681020335bad9a5f0e0fe7be7` |

### pi05_l10_200

Artifact directory: `/home/weiland/projects/openpi/exp/offline_search/rounds/r15/qk/astra/round2/pilot/pi05_l10_200`. Manifest SHA256: `e48a7e44aa49807e995b79f93e44cf9e6030ee0d9a136227785ef3eca1ab9f98`.

| File | SHA256 |
|---|---|
| `student.ts` | `4272d04dfcfa4929d2e9fed5efb393aa8ffd47f061d3799430d7c4ba0b059a00` |
| `student.pt` | `ffe667f53bacd8a6863063a849c35f1421bf01f2924a5bf8507c4801890cb03d` |
| `library_features.npz` | `f486728bae2127c4ac261cc33f7339a933f6d11456adfda2e1fda036dcfa125d` |
| `metric.pkl` | `ea992f8f2f0973f04f65bcabae3afb53b4f308a28aa8509567a20760d8bba2b3` |
| `latency.json` | `934bc3d87aa32ee4300b24e0b3d47afe4b555e07276041e14edb808c3af9a11e` |
| `encoder_config.json` | `9a8cb4541272a79bdb534664b9a6f86ce079e8aa1a34e070bf88c69602d5e1c6` |

## Stable artifact format: r15_qk_student_v2

`student.ts` is a frozen CPU TorchScript module with `forward(images, state, task_id) -> features`. Images are float32 `[N,2,3,56,56]` in [0,1], agentview then wrist, from raw 224² uint8 RGB via `image[::4,::4]`, CHW and `/255`. State is float32 `[N,8]`, the existing valid normalized `rs[:8]`; student-specific normalization is internal. Task is int64 `[N]`, 0..9. Output is a single float32 `[N,128]` Tensor. No history input or new state channel is required by these selected models.

`library_features.npz` binds float32 `features[L,128]` to int64 parent `rows[L]` in exact selected-library order. Split the 128 features into two 64-value blocks only for existing accessors; they are jointly produced light features, not original teacher tokens or independent camera embeddings. Append the unchanged valid state and refit the same AWM/main/early/confidence and R10 corrector from this cell only. Do not apply the teacher PCA again. `metric.pkl` is the exploratory per-task main/early affine metric; sol's complete layer-1–3 refit is the runtime artifact.

`encoder_config.json` is directly compatible with `exp.offline_search.rounds.r15.qk.impl.build --size SIZE --encoder-config PATH`. Factory is `impl.encoder:CNNStudent`; **`kwargs.output_index` is null**, because the module returns a Tensor. `declared_epochs` matches the selected recipe rather than assuming 80. The config binds model, training record, exact subset and cost coefficient; `library_features_reference` also enables sol's all-row feature-parity check after encoding. `student.pt` is the portable config/state_dict checkpoint. All-task perturbation checks and all-library export parity are in the manifest; `round2/verification.json` additionally checks sol's actual FrozenEncoder loader on 30 raw-image examples/cell.

Local encoder configs use one Torch thread and the authorized CPU affinity 10–21,54–65. When staging H100, sol/coordinator must bind paths to the frozen copies, record that host's available CPU allocation, and recompute its serving-config hash if paths or operational settings change. Student weight hashes stay fixed. Deployment-host timing is still required; local epsilon is the measured one-thread raw-image-to-AWM-code cost normalized by the owner π0.5 10.26-ms graph tower and .152 coefficient. Initial export timings are retained separately; final timing controls both Torch and NumPy BLAS threads and uses 400 distinct own-library image pairs.

## Eight arms, 300 paired B episodes per arm

Use `/home/weiland/projects/openpi/exp/offline_search/rounds/r15/descent/selection.json`, SHA256 `c6cfbe28d3a1942a2621636b6be4e76b169d4c1160619b730bfef3ad25b17ac8`, field `model_suites.pi05_l10.pairs`. It has 30 identities/task, 300 total, verified disjoint from both served subsets. Rollout replicate seed: 2026101401; knob coin seed: 0. This is development reuse, not independent validation.

| Cell | Current off | New key off | Current ir25 | New key ir25 | Matched ir25 cap |
|---|---:|---:|---:|---:|---:|
| pi05_l10_50 | 300 | 300 | 300 | 300 | 2.6125 |
| pi05_l10_200 | 300 | 300 | 300 | 300 | 2.0000 |

Both sizes are included directly, per the coordinator's revised directive. Total design: 2,400 arm-episodes. Preserve actual fixed caps, only-no-progress logic, same-library LOEO correction, commit/replan/policy-tail semantics, and paired host/seeds. Keep the baseline π0.5 H100 stack, five-control replan and full policy call. Do not reprice ir25 to consume the saved budget, introduce a new task table, or add teacher refresh probes.

## Baseline reuse from R15 descent

L10-200 current off and ir25 match the R15 descent selection, seed, library, fixed caps and intended H100 host:

- `/home/weiland/trace_runs/os_closed_loop/r15_descent_pi05_l10_200/runs/r15d_pi05_l10_200_off`
- `/home/weiland/trace_runs/os_closed_loop/r15_descent_pi05_l10_200/runs/r15d_pi05_l10_200_periodic_ir25`

At the metadata audit, neither had a local completion summary or run manifest. These are eligible planned baselines, not completed evidence. Reuse once all 300 accepted paired outcomes exist and host/runtime/fit/cap identity is verified; no duplicate collection is needed if those conditions hold. There is no R15 size-50 descent root, so collect both size-50 current-system baselines on the exact same list. A size-100/200 baseline cannot substitute for size 50. With the two size-200 baselines reused, the pilot needs 1,800 additional arm-episodes: 1,200 student and 600 current-system size-50. Source hashes and identity checks are in `round2/pilot_identity_baseline.json`; no baseline success outcomes were inspected for this selection.

## Runtime and interpretation

Sol must complete its generic tower-skipping integration and the same-library layers 1–3 refits using these encoder configs. Verify zero tower on a light HIT, one complete policy/tower on MISS, neither encoder on legal blind continuation, single commit, exact parent rows, reset/receipt semantics and default teacher-path parity. Separate Q=student calls, T=tower calls and M=full-policy calls; the specified path has T=M and IR=(epsilon*Q+M)/N. Logical look history remains separate from tower work.

Report the original numerical offline gate results without relabeling failures. The owner override permits the pilot, not a claim of accuracy equivalence. For the 300-episode matched comparison retain the frozen cell SR delta ≥−2 pp, each task ≥−5 pp, conservative paired cell lower95 ≥−5 pp, and the original large-saving targets (off ≥25%, ir25 ≥15%). Report actual IR, task/episode paired gains/losses, errors, executed controls and cost per episode. The fixed-table saving remains a separately labeled projection/comparison, not an extra pilot arm. No new A work is authorized here.

Recommendation: continue with this bounded pilot, then decide from matched SR and real cost. The image audits pass, and the selected size-50 regularization improves its unseen error, but the offline gap remains. Any further architecture/epoch/knob change must be separately versioned before its outcomes are read.
